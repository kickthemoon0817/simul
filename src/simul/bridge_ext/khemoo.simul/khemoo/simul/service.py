"""Typed command service for the Simul Isaac bridge."""

from __future__ import annotations

import time
from typing import Any, Optional

from .executor import ScriptExecutor
from .prim_detail import BULK_GEOMETRY_ATTRIBUTES, PRIM_DETAIL_ASPECTS, PrimDetailReader
from .protocol import DEFAULT_REQUEST_TIMEOUT_SECONDS, BridgeRequest, BridgeResponse


# Actions that only read state. They are safe to dispatch without the request
# lock: Kit is single-threaded, so they cannot interleave mid-operation with a
# mutating action, and holding them behind a long-running step is what made a
# health-check ping look like an unreachable instance. get_runtime_info is
# here so its busy report can be read while the lock is held by the very
# action it reports on.
READ_ONLY_ACTIONS = frozenset(
    {
        "ping",
        "capabilities",
        "get_stage_info",
        "get_simulation_state",
        "get_prim_info",
        "get_prim_detail",
        "list_prims",
        "get_runtime_info",
    }
)

# Actions dispatched without the request lock. interrupt changes nothing in
# the scene but exists to reach a script that is holding the lock, so it must
# not queue behind it.
LOCK_FREE_ACTIONS = READ_ONLY_ACTIONS | frozenset({"interrupt"})

__all__ = [
    "BULK_GEOMETRY_ATTRIBUTES",
    "LOCK_FREE_ACTIONS",
    "POSE_ASPECTS",
    "READ_ONLY_ACTIONS",
    "BridgeCommandService",
    "usd_readback_note",
]

# Prim detail aspects whose values come from USD transforms or velocities.
POSE_ASPECTS = frozenset({"info", "transform", "bounding_box", "rigid_body"})

# Isaac Sim's SimulationManager switches PhysX to Fabric for GPU physics:
# enable_fabric() enables omni.physx.fabric and sets /physics/updateToUsd and
# /physics/updateVelocitiesToUsd to false (v5.1.0 simulation_manager.py,
# v6.0.1 set_device). The simulation then stops writing poses and velocities
# to USD, so a USD read returns the values from before play. Same wording as
# simul's USD_READBACK_STALE_WARNING; this module cannot import simul.
USD_READBACK_STALE_WARNING = (
    "PhysX is writing simulation results to Fabric, not USD "
    "(/physics/updateToUsd or /physics/updateVelocitiesToUsd is false, as Isaac "
    "Sim's SimulationManager sets for GPU physics), so the poses and velocities "
    "read from USD here are the values from before play, not the simulated state. "
    "Read the simulated state through a physics view (omni.physics.tensors, or "
    "isaacsim.core.experimental.prims on 6.0) with execute_isaac_script."
)


def usd_readback_note() -> Optional[dict[str, Any]]:
    """Describe whether USD poses and velocities lag the simulation right now.

    Returns:
        None while the timeline is stopped (USD then holds the authored state,
        which is the state) or when nothing points at Fabric; otherwise
        ``update_to_usd``, ``update_velocities_to_usd``,
        ``physx_fabric_enabled`` and ``stale_under_fabric``. Never raises.
    """
    try:
        import carb.settings
        import omni.timeline

        if omni.timeline.get_timeline_interface().is_stopped():
            return None
        settings = carb.settings.get_settings()
        update_to_usd = settings.get("/physics/updateToUsd")
        update_velocities_to_usd = settings.get("/physics/updateVelocitiesToUsd")
    except Exception:
        return None
    fabric_enabled: Optional[bool] = None
    try:
        import omni.kit.app

        manager = omni.kit.app.get_app().get_extension_manager()
        fabric_enabled = bool(manager.is_extension_enabled("omni.physx.fabric"))
    except Exception:
        pass
    stale = update_to_usd is False or update_velocities_to_usd is False
    if not stale and not fabric_enabled:
        return None
    return {
        "update_to_usd": update_to_usd,
        "update_velocities_to_usd": update_velocities_to_usd,
        "physx_fabric_enabled": fabric_enabled,
        "stale_under_fabric": stale,
    }


def _physics_scene_rate() -> tuple[Optional[str], Optional[float], Optional[str]]:
    """Find the physics scene near the stage root and read its step rate.

    Only the top two levels are searched (``/physicsScene``,
    ``/World/physicsScene``), so a large stage is not walked on Kit's main
    thread. Without PhysxSceneAPI PhysX steps at the schema fallback of 60 Hz,
    which is also what Isaac Sim's SimulationManager reports then.

    Returns:
        ``(scene_path, steps_per_second, source)``; all None when no scene
        is found.
    """
    import omni.usd
    from pxr import PhysxSchema, Usd, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return None, None, None
    prims = iter(Usd.PrimRange(stage.GetPseudoRoot()))
    for prim in prims:
        if prim.GetPath().pathElementCount >= 2:
            prims.PruneChildren()
        if not prim.IsA(UsdPhysics.Scene):
            continue
        if prim.HasAPI(PhysxSchema.PhysxSceneAPI):
            rate = PhysxSchema.PhysxSceneAPI(prim).GetTimeStepsPerSecondAttr().Get()
            return str(prim.GetPath()), rate, "physxScene:timeStepsPerSecond"
        return str(prim.GetPath()), 60.0, "PhysxSceneAPI fallback"
    return None, None, None


def _physics_runtime_state() -> dict[str, Any]:
    """Report the Carb settings and extension that decide where physics writes.

    These are the keys Isaac Sim's own SimulationManager reads and sets
    (/physics/cudaDevice, /physics/suppressReadback, /physics/updateToUsd,
    /physics/updateVelocitiesToUsd, the omni.physx.fabric extension) plus
    /physics/resetOnStop from the app's .kit file. The omni.physx interface
    has no physics statistics to report: get_physics_stats and
    is_cuda_lib_present are not on it in Kit 107 or 110.
    """
    import carb.settings

    settings = carb.settings.get_settings()
    state: dict[str, Any] = {
        "cuda_device": settings.get("/physics/cudaDevice"),
        "suppress_readback": settings.get("/physics/suppressReadback"),
        "update_to_usd": settings.get("/physics/updateToUsd"),
        "update_velocities_to_usd": settings.get("/physics/updateVelocitiesToUsd"),
        "reset_on_stop": settings.get("/physics/resetOnStop"),
        "physx_fabric_enabled": None,
    }
    try:
        import omni.kit.app

        manager = omni.kit.app.get_app().get_extension_manager()
        state["physx_fabric_enabled"] = bool(manager.is_extension_enabled("omni.physx.fabric"))
    except Exception:
        pass
    return state


def _physics_scene_config() -> dict[str, Any]:
    """Read the physics scene's own settings, the ones SimulationManager reads.

    The previous keys (/physics/gpuDynamicsEnabled,
    /persistent/simulation/defaultPhysicsDt, /persistent/physics/solverType)
    appear nowhere in Isaac Sim and always read None. The rate, GPU dynamics,
    solver and broadphase live on the scene's PhysxSceneAPI.
    """
    import omni.usd
    from pxr import PhysxSchema, Usd, UsdPhysics

    config: dict[str, Any] = {
        "scene_path": None,
        "time_steps_per_second": None,
        "physics_dt": None,
        "gpu_dynamics_enabled": None,
        "solver_type": None,
        "broadphase_type": None,
    }
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return config
    prims = iter(Usd.PrimRange(stage.GetPseudoRoot()))
    for prim in prims:
        if prim.GetPath().pathElementCount >= 2:
            prims.PruneChildren()
        if not prim.IsA(UsdPhysics.Scene):
            continue
        config["scene_path"] = str(prim.GetPath())
        if prim.HasAPI(PhysxSchema.PhysxSceneAPI):
            api = PhysxSchema.PhysxSceneAPI(prim)
            rate = api.GetTimeStepsPerSecondAttr().Get()
            config["time_steps_per_second"] = rate
            config["physics_dt"] = 1.0 / rate if rate else None
            config["gpu_dynamics_enabled"] = api.GetEnableGPUDynamicsAttr().Get()
            solver = api.GetSolverTypeAttr().Get()
            config["solver_type"] = str(solver) if solver is not None else None
            broadphase = api.GetBroadphaseTypeAttr().Get()
            config["broadphase_type"] = str(broadphase) if broadphase is not None else None
        break
    return config


def _physics_rates(timeline: Any, frame_dt: Optional[float], time_delta: float) -> dict[str, Any]:
    """Report the rates that decide how many physics steps a frame runs.

    Every read is guarded: a missing piece leaves its field None rather than
    failing the step that already happened.
    """
    rates: dict[str, Any] = {
        "physics_scene": None,
        "physics_steps_per_second": None,
        "physics_dt": None,
        "rate_source": None,
        "timeline_time_codes_per_second": None,
        "rate_limit_hz": None,
        "physics_steps_per_frame": None,
        "physics_steps_estimated": None,
    }
    try:
        rates["timeline_time_codes_per_second"] = timeline.get_time_codes_per_seconds()
    except Exception:
        pass
    try:
        import carb.settings

        settings = carb.settings.get_settings()
        if settings.get("/app/runLoops/main/rateLimitEnabled"):
            rates["rate_limit_hz"] = settings.get("/app/runLoops/main/rateLimitFrequency")
    except Exception:
        pass
    try:
        scene, rate, source = _physics_scene_rate()
    except Exception:
        scene, rate, source = None, None, None
    rates["physics_scene"] = scene
    rates["rate_source"] = source
    if rate:
        rate = float(rate)
        rates["physics_steps_per_second"] = rate
        rates["physics_dt"] = 1.0 / rate
        if frame_dt:
            rates["physics_steps_per_frame"] = round(frame_dt * rate, 3)
        rates["physics_steps_estimated"] = int(round(time_delta * rate))
    return rates


def _annotate_usd_readback(payload: dict[str, Any]) -> None:
    """Flag a pose or velocity payload read while USD lags the simulation."""
    note = usd_readback_note()
    if note is None:
        return
    payload.setdefault("usd_readback", note)
    if note["stale_under_fabric"]:
        payload.setdefault("stale_under_fabric", True)
        warnings = payload.setdefault("warnings", [])
        if isinstance(warnings, list):
            warnings.append(USD_READBACK_STALE_WARNING)


class BridgeCommandService:
    """Dispatch typed bridge actions inside Isaac Sim."""

    def __init__(
        self,
        executor: ScriptExecutor,
        allow_unsafe_execution: bool,
        request_timeout: float | None = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self._executor = executor
        self._allow_unsafe_execution = allow_unsafe_execution
        # The lifecycle abandons a request after ``request_timeout``; a script
        # budget at or above it would never fire. Keep the ceiling a margin
        # below so the executor interrupts first and reports why.
        self._script_timeout_ceiling: float | None = (
            request_timeout - min(5.0, request_timeout * 0.1)
            if request_timeout and request_timeout > 0
            else None
        )
        # The mutating action in flight, if any. Reads run concurrently and
        # are not tracked: they cannot be what a client is waiting behind.
        self._current_action: str | None = None
        self._busy_since: float | None = None

    @property
    def capabilities(self) -> dict[str, Any]:
        """Return the bridge capability description."""
        return {
            "transport": "simul_bridge",
            "actions": [
                "ping",
                "capabilities",
                "execute_script",
                "interrupt",
                "get_stage_info",
                "list_prims",
                "get_prim_info",
                "get_prim_transform",
                "get_prim_detail",
                "search_prims",
                "get_runtime_info",
                "get_simulation_state",
                "simulation_control",
            ],
            "allow_unsafe_execution": self._allow_unsafe_execution,
        }

    @property
    def activity(self) -> dict[str, Any]:
        """Describe what the bridge is doing right now.

        Lets a client tell a busy bridge from a hung one: a mutating action
        holds the request lock for as long as it runs, and a script may keep
        running after its own client gave up waiting.

        Returns:
            ``busy``, ``busy_since`` (epoch seconds or None),
            ``busy_for_seconds`` and ``current_action``. A script still
            executing outside a tracked action (the VS Code compat path)
            reports as ``execute_script``.
        """
        action = self._current_action
        since = self._busy_since
        if action is None:
            executor_state = self._executor.state
            if executor_state["busy"]:
                action = "execute_script"
                since = executor_state["busy_since"]
        return {
            "busy": action is not None,
            "busy_since": since,
            "busy_for_seconds": (
                round(time.time() - since, 3) if since is not None else None
            ),
            "current_action": action,
        }

    async def dispatch(self, request: BridgeRequest) -> BridgeResponse:
        """Dispatch a bridge request to the matching typed handler.

        Mutating actions are recorded in ``activity`` for the duration of the
        call; lock-free actions are not.
        """
        if request.action in LOCK_FREE_ACTIONS:
            return await self._dispatch_action(request)
        self._current_action = request.action
        self._busy_since = time.time()
        try:
            return await self._dispatch_action(request)
        finally:
            self._current_action = None
            self._busy_since = None

    async def _dispatch_action(self, request: BridgeRequest) -> BridgeResponse:
        """Route one request to its typed handler."""
        if request.action == "ping":
            return BridgeResponse.success(
                request.request_id,
                {"reachable": True, "transport": "simul_bridge"},
            )
        if request.action == "capabilities":
            payload = dict(self.capabilities)
            payload["protocol_version"] = request.protocol_version
            return BridgeResponse.success(request.request_id, payload)
        if request.action == "execute_script":
            return await self._handle_execute_script(request)
        if request.action == "interrupt":
            return self._handle_interrupt(request)
        if request.action == "get_stage_info":
            return self._handle_get_stage_info(request)
        if request.action == "list_prims":
            return self._handle_list_prims(request)
        if request.action == "get_prim_info":
            return self._handle_get_prim_info(request)
        if request.action == "get_prim_transform":
            return self._handle_get_prim_transform(request)
        if request.action == "get_prim_detail":
            return self._handle_get_prim_detail(request)
        if request.action == "search_prims":
            return self._handle_search_prims(request)
        if request.action == "get_runtime_info":
            return self._handle_get_runtime_info(request)
        if request.action == "get_simulation_state":
            return self._handle_get_simulation_state(request)
        if request.action == "simulation_control":
            return await self._handle_simulation_control(request)
        return BridgeResponse.failure(
            request.request_id,
            "UnknownAction",
            f"Unsupported bridge action: {request.action}",
        )

    async def _handle_execute_script(self, request: BridgeRequest) -> BridgeResponse:
        """Execute arbitrary Python source via the bridge."""
        if not self._allow_unsafe_execution:
            return BridgeResponse.failure(
                request.request_id,
                "UnsafeExecutionDisabled",
                "execute_script is disabled in bridge settings.",
            )
        code = request.payload.get("code")
        if not isinstance(code, str) or not code.strip():
            return BridgeResponse.failure(
                request.request_id,
                "InvalidRequest",
                "execute_script requires a non-empty string payload.code.",
            )
        raw_timeout = request.payload.get("timeout")
        timeout_seconds: float | None = None
        if raw_timeout is not None:
            try:
                timeout_seconds = float(raw_timeout)
            except (TypeError, ValueError):
                return BridgeResponse.failure(
                    request.request_id,
                    "InvalidRequest",
                    "execute_script payload.timeout must be a number of seconds.",
                )
            if timeout_seconds <= 0:
                timeout_seconds = None
        ceiling = self._script_timeout_ceiling
        if ceiling is not None and (timeout_seconds is None or timeout_seconds > ceiling):
            timeout_seconds = ceiling
        output, exception, trace = await self._executor.execute(
            code, timeout_seconds=timeout_seconds
        )
        if output.endswith("\n"):
            output = output[:-1]
        if exception is not None:
            return BridgeResponse.failure(
                request.request_id,
                type(exception).__name__,
                str(exception),
                traceback=trace,
            )
        return BridgeResponse.success(
            request.request_id,
            {"output": output, "transport": "simul_bridge"},
        )

    def _handle_interrupt(self, request: BridgeRequest) -> BridgeResponse:
        """Stop the script currently running in the executor, if any.

        Only a coroutine script, or one suspended at an await, can be reached
        this way: a synchronous script that never yields also blocks the loop
        this request arrives on, so for it the per-request timeout sent with
        execute_script is the stop that works.
        """
        activity = self.activity
        phase = self._executor.state["phase"]
        interrupted = self._executor.interrupt("interrupted by request")
        return BridgeResponse.success(
            request.request_id,
            {
                "transport": "simul_bridge",
                "interrupted": interrupted,
                "was_busy": activity["busy"],
                "phase": phase,
                "current_action": activity["current_action"],
                "busy_for_seconds": activity["busy_for_seconds"],
            },
        )

    def _handle_get_stage_info(self, request: BridgeRequest) -> BridgeResponse:
        """Return metadata for the currently open stage."""
        import omni.usd
        from pxr import UsdGeom

        ctx = omni.usd.get_context()
        stage = ctx.get_stage()
        if stage is None:
            return BridgeResponse.failure(
                request.request_id,
                "StageUnavailable",
                "No stage is currently open.",
            )

        root = stage.GetPseudoRoot()
        root_prims = [str(prim.GetPath()) for prim in root.GetChildren()]
        default_prim = stage.GetDefaultPrim()
        # Counting prims walks the entire stage on Kit's main thread. Callers
        # that only need stage metadata can opt out; the key stays present so
        # the response shape does not change. Older clients omit the flag and
        # keep the previous behaviour.
        include_prim_count = request.payload.get("include_prim_count", True)
        payload = {
            "transport": "simul_bridge",
            "stage_url": str(ctx.get_stage_url()),
            "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
            "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
            "time_codes_per_second": stage.GetTimeCodesPerSecond(),
            "start_time": stage.GetStartTimeCode(),
            "end_time": stage.GetEndTimeCode(),
            "frame_rate": stage.GetFramesPerSecond(),
            "total_prims": (
                sum(1 for _ in stage.Traverse()) if include_prim_count else None
            ),
            "root_prims": root_prims,
            "layer_count": len(stage.GetLayerStack()),
            "default_prim": str(default_prim.GetPath()) if default_prim else None,
        }
        return BridgeResponse.success(request.request_id, payload)

    def _handle_list_prims(self, request: BridgeRequest) -> BridgeResponse:
        """Return a filtered prim listing for the current stage."""
        import omni.usd
        from pxr import Usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return BridgeResponse.failure(
                request.request_id,
                "StageUnavailable",
                "No stage is currently open.",
            )

        root_path = str(request.payload.get("root_path", "/"))
        prim_type = request.payload.get("prim_type")
        max_depth = int(request.payload.get("max_depth", 5))
        max_results, offset = self._page_bounds(request.payload, default_limit=100)

        root = stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            return BridgeResponse.failure(
                request.request_id,
                "InvalidRootPath",
                f"Invalid root path: {root_path}",
            )

        root_depth = len(str(root.GetPath()).rstrip("/").split("/"))
        prims = []
        matched = 0
        truncated = False
        # continue skips emitting a prim but still descends its subtree, so the
        # depth limit bounded the response and not the work. Prune instead —
        # and the page-full break below is unreachable while we continue past
        # every deep prim on the stage.
        prim_iter = iter(Usd.PrimRange(root))
        for prim in prim_iter:
            path_str = str(prim.GetPath())
            depth = len(path_str.rstrip("/").split("/")) - root_depth
            if max_depth >= 0 and depth >= max_depth:
                prim_iter.PruneChildren()
            if max_depth >= 0 and depth > max_depth:
                continue
            prim_type_name = prim.GetTypeName()
            if prim_type and prim_type_name != prim_type:
                continue
            matched += 1
            if matched <= offset:
                continue
            if len(prims) >= max_results:
                truncated = True
                break
            prims.append(
                {
                    "path": path_str,
                    "type": prim_type_name,
                    "name": prim.GetName(),
                    "active": prim.IsActive(),
                }
            )

        return BridgeResponse.success(
            request.request_id,
            {
                "transport": "simul_bridge",
                "root_path": root_path,
                "type_filter": prim_type,
                "max_depth": max_depth,
                "count": len(prims),
                "offset": offset,
                "applied_limit": max_results,
                "truncated": truncated,
                "next_offset": offset + len(prims) if truncated else None,
                "prims": prims,
            },
        )

    def _handle_get_prim_info(self, request: BridgeRequest) -> BridgeResponse:
        """Return detailed information about a single prim."""
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return BridgeResponse.failure(
                request.request_id,
                "StageUnavailable",
                "No stage is currently open.",
            )

        prim_path = str(request.payload.get("prim_path", ""))
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return BridgeResponse.failure(
                request.request_id,
                "PrimNotFound",
                f"Prim not found: {prim_path}",
            )

        payload = PrimDetailReader(stage).info(prim)
        payload["path"] = prim_path
        payload["transport"] = "simul_bridge"
        _annotate_usd_readback(payload)
        return BridgeResponse.success(request.request_id, payload)

    def _handle_get_prim_transform(self, request: BridgeRequest) -> BridgeResponse:
        """Return local or world transform for a prim."""
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return BridgeResponse.failure(
                request.request_id,
                "StageUnavailable",
                "No stage is currently open.",
            )

        prim_path = str(request.payload.get("prim_path", ""))
        world_space = bool(request.payload.get("world_space", True))
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return BridgeResponse.failure(
                request.request_id,
                "PrimNotFound",
                f"Prim not found: {prim_path}",
            )

        payload = PrimDetailReader(stage).transform(prim, world_space)
        if "error" in payload:
            return BridgeResponse.failure(
                request.request_id,
                "PrimNotXformable",
                payload["error"],
            )
        payload["prim_path"] = prim_path
        payload["transport"] = "simul_bridge"
        _annotate_usd_readback(payload)
        return BridgeResponse.success(request.request_id, payload)

    def _handle_get_prim_detail(self, request: BridgeRequest) -> BridgeResponse:
        """Return every requested aspect of one prim in a single response."""
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return BridgeResponse.failure(
                request.request_id,
                "StageUnavailable",
                "No stage is currently open.",
            )

        prim_path = str(request.payload.get("prim_path", ""))
        aspects = request.payload.get("aspects") or ["info"]
        if not isinstance(aspects, list) or not all(isinstance(a, str) for a in aspects):
            return BridgeResponse.failure(
                request.request_id,
                "InvalidRequest",
                "get_prim_detail requires payload.aspects to be a list of strings.",
            )
        unknown = [aspect for aspect in aspects if aspect not in PRIM_DETAIL_ASPECTS]
        if unknown:
            return BridgeResponse.failure(
                request.request_id,
                "InvalidAspect",
                f"Unknown aspect(s): {', '.join(unknown)}. "
                f"Valid aspects: {', '.join(PRIM_DETAIL_ASPECTS)}.",
            )

        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return BridgeResponse.failure(
                request.request_id,
                "PrimNotFound",
                f"Prim not found: {prim_path}",
            )

        reader = PrimDetailReader(stage)
        payload: dict[str, Any] = {
            "transport": "simul_bridge",
            "prim_path": prim_path,
            "aspects": list(aspects),
        }
        for aspect in aspects:
            # Aspects are independent reads: one that fails is reported in
            # place rather than discarding the rest of the answer, and each
            # carries its own success flag like the per-aspect tools do.
            try:
                detail = reader.read(prim, aspect)
            except Exception as exc:
                detail = {"error": str(exc), "error_type": type(exc).__name__}
            detail["success"] = "error" not in detail
            payload[aspect] = detail
        if POSE_ASPECTS.intersection(aspects):
            _annotate_usd_readback(payload)
        return BridgeResponse.success(request.request_id, payload)

    def _handle_search_prims(self, request: BridgeRequest) -> BridgeResponse:
        """Search for prims by type or name."""
        import omni.usd
        from pxr import Usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return BridgeResponse.failure(
                request.request_id,
                "StageUnavailable",
                "No stage is currently open.",
            )

        search_type = str(request.payload.get("search_type", "type"))
        query = request.payload.get("query")
        if not isinstance(query, str) or not query.strip():
            return BridgeResponse.failure(
                request.request_id,
                "InvalidRequest",
                "search_prims requires a non-empty string payload.query.",
            )
        root_path = str(request.payload.get("root_path", "/"))
        max_results, offset = self._page_bounds(request.payload, default_limit=100)
        if search_type not in ("type", "name"):
            return BridgeResponse.failure(
                request.request_id,
                "InvalidSearchType",
                f"Unsupported search_type: {search_type}",
            )

        root = stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            return BridgeResponse.failure(
                request.request_id,
                "InvalidRootPath",
                f"Invalid root path: {root_path}",
            )

        matches = []
        matched = 0
        truncated = False
        for prim in Usd.PrimRange(root):
            if search_type == "type":
                match = prim.GetTypeName() == query
            else:
                match = query.lower() in prim.GetName().lower()
            if not match:
                continue
            matched += 1
            if matched <= offset:
                continue
            if len(matches) >= max_results:
                truncated = True
                break
            matches.append(
                {
                    "path": str(prim.GetPath()),
                    "type": prim.GetTypeName(),
                    "name": prim.GetName(),
                }
            )

        return BridgeResponse.success(
            request.request_id,
            {
                "transport": "simul_bridge",
                "search_type": search_type,
                "query": query,
                "root_path": root_path,
                "count": len(matches),
                "offset": offset,
                "applied_limit": max_results,
                "truncated": truncated,
                "next_offset": offset + len(matches) if truncated else None,
                "matches": matches,
            },
        )

    def _handle_get_runtime_info(self, request: BridgeRequest) -> BridgeResponse:
        """Return consolidated runtime diagnostics."""
        import sys

        info: dict[str, Any] = {"transport": "simul_bridge", "bridge": self.activity}

        try:
            import omni.kit.app

            app = omni.kit.app.get_app()
            info["app"] = {
                "version": str(app.get_build_version()),
                "python_version": sys.version.split()[0],
                "update_number": int(app.get_update_number()),
            }
            # Kit build string above is what get_build_version returns; the
            # Isaac Sim release (5.1.0, 6.0.1, ...) is what callers need to
            # pick API namespaces.
            if hasattr(app, "get_app_version"):
                info["app"]["isaac_version"] = str(app.get_app_version())
        except Exception as exc:
            info["app_error"] = str(exc)

        try:
            import omni.timeline

            timeline = omni.timeline.get_timeline_interface()
            info["timeline"] = {
                "is_playing": timeline.is_playing(),
                "is_stopped": timeline.is_stopped(),
                "current_time": timeline.get_current_time(),
                "start_time": timeline.get_start_time(),
                "end_time": timeline.get_end_time(),
                "fps": timeline.get_time_codes_per_second(),
            }
        except Exception as exc:
            info["timeline_error"] = str(exc)

        try:
            info["physics"] = _physics_runtime_state()
        except Exception as exc:
            info["physics_error"] = str(exc)

        try:
            info["physics_config"] = _physics_scene_config()
        except Exception as exc:
            info["physics_config_error"] = str(exc)

        try:
            import carb.settings

            settings = carb.settings.get_settings()
            info["renderer"] = {
                "active_gpu": settings.get("/renderer/activeGpu"),
                "gpu_name": settings.get("/renderer/gpuName"),
                "hgi_driver": settings.get("/renderer/hgi/driver"),
                "raytracing_mode": settings.get("/rtx/rendermode"),
                "realtime_mode": settings.get("/rtx/ecoMode/enabled"),
            }
        except Exception as exc:
            info["renderer_error"] = str(exc)

        try:
            from omni.kit.viewport.utility import get_active_viewport

            viewport = get_active_viewport()
            if viewport:
                info["viewport"] = {
                    "camera_path": str(viewport.camera_path),
                    "resolution": list(viewport.resolution),
                    "fps": viewport.fps if hasattr(viewport, "fps") else None,
                }
            else:
                info["viewport"] = {"status": "no active viewport"}
        except Exception as exc:
            info["viewport_error"] = str(exc)

        try:
            import omni.usd

            ctx = omni.usd.get_context()
            stage = ctx.get_stage()
            # Counting prims walks the entire stage on Kit's main thread.
            # Callers that only need the diagnostics can opt out; the key
            # stays present so the response shape does not change. Older
            # clients omit the flag and keep the previous behaviour.
            include_prim_count = request.payload.get("include_prim_count", True)
            if stage:
                info["stage"] = {
                    "url": ctx.get_stage_url(),
                    "prim_count": (
                        sum(1 for _ in stage.Traverse()) if include_prim_count else None
                    ),
                }
            else:
                info["stage"] = {"status": "no stage open"}
        except Exception as exc:
            info["stage_error"] = str(exc)

        try:
            import omni.kit.app

            manager = omni.kit.app.get_app().get_extension_manager()
            extensions = manager.get_extensions()
            info["extensions"] = {
                "total": len(extensions),
                "enabled": len([item for item in extensions if item.get("enabled")]),
            }
        except Exception as exc:
            info["extensions_error"] = str(exc)

        return BridgeResponse.success(request.request_id, info)

    def _handle_get_simulation_state(
        self, request: BridgeRequest
    ) -> BridgeResponse:
        """Return current simulation state."""
        import omni.timeline

        timeline = omni.timeline.get_timeline_interface()
        is_playing = timeline.is_playing()
        is_stopped = timeline.is_stopped()
        if is_playing:
            state = "playing"
        elif is_stopped:
            state = "stopped"
        else:
            state = "paused"
        return BridgeResponse.success(
            request.request_id,
            {
                "transport": "simul_bridge",
                "state": state,
                "current_time": timeline.get_current_time(),
                "time_codes_per_second": timeline.get_time_codes_per_seconds(),
                "is_playing": is_playing,
                "is_stopped": is_stopped,
            },
        )

    async def _handle_simulation_control(
        self, request: BridgeRequest
    ) -> BridgeResponse:
        """Control the simulation state."""
        import omni.kit.app
        import omni.timeline

        timeline = omni.timeline.get_timeline_interface()
        command = str(request.payload.get("command", "")).lower()

        if command == "start":
            timeline.play()
            return BridgeResponse.success(
                request.request_id,
                {"transport": "simul_bridge", "state": "playing", "started": True},
            )
        if command == "stop":
            timeline.stop()
            return BridgeResponse.success(
                request.request_id,
                {"transport": "simul_bridge", "state": "stopped", "stopped": True},
            )
        if command == "pause":
            timeline.pause()
            return BridgeResponse.success(
                request.request_id,
                {"transport": "simul_bridge", "state": "paused", "paused": True},
            )
        if command == "reset":
            timeline.stop()
            timeline.set_current_time(0)
            return BridgeResponse.success(
                request.request_id,
                {"transport": "simul_bridge", "state": "stopped", "reset": True},
            )
        if command == "step":
            num_steps = max(1, min(int(request.payload.get("num_steps", 1)), 1000))
            budget: float | None = None
            try:
                budget = float(request.payload["timeout"]) if "timeout" in request.payload else None
            except (TypeError, ValueError):
                budget = None
            ceiling = self._script_timeout_ceiling
            if ceiling is not None and (budget is None or budget <= 0 or budget > ceiling):
                # The lifecycle abandons the request at its own timeout; stop
                # stepping first so the reply says how far it got.
                budget = ceiling
            result = await self._step_timeline(
                timeline, omni.kit.app.get_app(), num_steps, budget_seconds=budget
            )
            result["transport"] = "simul_bridge"
            # An incomplete step keeps its counts beside ``error``; the client
            # envelope turns that key into success=False.
            return BridgeResponse.success(request.request_id, result)
        return BridgeResponse.failure(
            request.request_id,
            "InvalidSimulationCommand",
            f"Unsupported simulation command: {command}",
        )

    @staticmethod
    async def _step_timeline(
        timeline: Any,
        app: Any,
        num_steps: int,
        budget_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Advance the timeline by exactly ``num_steps`` frames and leave it paused.

        A paused timeline does not move on app updates, and a stopped one
        spends its first updates after ``play`` initialising physics without
        advancing time. So the timeline is played and only updates that
        actually moved the current time are counted; once ``num_steps`` of
        them have run it is paused again. The update budget bounds the loop
        when time cannot advance (end of a non-looping range), and
        ``budget_seconds`` bounds it in wall-clock time: Isaac Sim's full app
        is rate-limited (60 Hz by default), so a long step can outlast any
        fixed timeout.

        One frame is not one physics step: PhysX consumes the timeline's time
        in steps of the physics scene's dt, so a frame runs
        ``frame_dt / physics_dt`` of them (Isaac Sim's SimulationContext
        derives its substeps the same way). Both rates are reported.

        Args:
            timeline: The ``omni.timeline`` interface.
            app: The Kit application, for ``next_update_async``.
            num_steps: Frames to advance, already clamped by the caller.
            budget_seconds: Wall-clock limit; stepping stops and reports when
                it runs out. None means no limit beyond the update budget.

        Returns:
            ``steps`` (frames advanced), ``steps_requested``, ``start_time``,
            ``current_time``, ``time_delta``, ``app_updates``, ``state``,
            ``frame_dt``, ``physics`` (rates and estimated physics steps) and
            ``elapsed_seconds``; plus ``error`` when fewer frames advanced
            than requested.
        """
        started = time.monotonic()
        start_time = timeline.get_current_time()
        if not timeline.is_playing():
            timeline.play()
            commit = getattr(timeline, "commit", None)
            if callable(commit):
                commit()
        advanced = 0
        updates = 0
        last = start_time
        max_updates = num_steps + 60
        out_of_time = False
        try:
            while advanced < num_steps and updates < max_updates:
                if budget_seconds is not None and time.monotonic() - started >= budget_seconds:
                    out_of_time = True
                    break
                await app.next_update_async()
                updates += 1
                now = timeline.get_current_time()
                if now != last:
                    advanced += 1
                    last = now
        finally:
            timeline.pause()
            commit = getattr(timeline, "commit", None)
            if callable(commit):
                commit()
        current_time = timeline.get_current_time()
        time_delta = current_time - start_time
        frame_dt = time_delta / advanced if advanced else None
        elapsed = time.monotonic() - started
        result: dict[str, Any] = {
            "steps": advanced,
            "steps_requested": num_steps,
            "start_time": start_time,
            "current_time": current_time,
            "time_delta": time_delta,
            "app_updates": updates,
            "state": "paused",
            "frame_dt": frame_dt,
            "physics": _physics_rates(timeline, frame_dt, time_delta),
            "elapsed_seconds": round(elapsed, 3),
        }
        if out_of_time:
            result["error"] = (
                f"Stepped {advanced} of {num_steps} frames before the {budget_seconds:g}s "
                f"budget ran out ({advanced / elapsed if elapsed else 0:.1f} frames/s; the "
                "app is rate-limited). Step fewer frames per call."
            )
        elif advanced < num_steps:
            result["error"] = (
                f"Timeline advanced {advanced} of {num_steps} steps in {updates} "
                "app updates; it may be at the end of a non-looping time range."
            )
        return result

    @staticmethod
    def _page_bounds(payload: dict[str, Any], default_limit: int) -> tuple[int, int]:
        """Read the page size and offset of a listing request.

        ``max_items`` is the name clients used before ``max_results`` and is
        honoured when the new name is absent.

        Args:
            payload: The request payload.
            default_limit: Page size when the payload names none.

        Returns:
            ``(applied_limit, offset)`` with the limit clamped to [1, 10000]
            and the offset to a non-negative integer.
        """
        requested = payload.get("max_results", payload.get("max_items", default_limit))
        applied_limit = max(1, min(int(requested), 10000))
        offset = max(0, int(payload.get("offset", 0)))
        return applied_limit, offset
