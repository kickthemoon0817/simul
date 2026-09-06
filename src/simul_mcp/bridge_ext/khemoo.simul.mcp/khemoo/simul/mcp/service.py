"""Typed command service for the Simul Isaac bridge."""

from __future__ import annotations

import time
from typing import Any, Optional

from .executor import ScriptExecutor
from .protocol import BridgeRequest, BridgeResponse


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
        "list_prims",
        "get_runtime_info",
    }
)

# Actions dispatched without the request lock. interrupt changes nothing in
# the scene but exists to reach a script that is holding the lock, so it must
# not queue behind it.
LOCK_FREE_ACTIONS = READ_ONLY_ACTIONS | frozenset({"interrupt"})

# Array attributes big enough that pulling their value is the cost being
# avoided. Gating on names rather than on isArray keeps small arrays such as
# xformOpOrder and primvars:displayColor readable; skipping those would lose
# information the caller needs and save nothing.
BULK_GEOMETRY_ATTRIBUTES = frozenset(
    {
        "points", "normals", "velocities", "accelerations",
        "faceVertexIndices", "faceVertexCounts", "holeIndices",
        "cornerIndices", "cornerSharpnesses", "creaseIndices",
        "creaseLengths", "creaseSharpnesses", "curveVertexCounts", "widths",
        "primvars:st", "primvars:normals",
        "positions", "orientations", "scales", "protoIndices",
        "invisibleIds", "ids",
    }
)


class BridgeCommandService:
    """Dispatch typed bridge actions inside Isaac Sim."""

    def __init__(
        self,
        executor: ScriptExecutor,
        allow_unsafe_execution: bool,
    ) -> None:
        self._executor = executor
        self._allow_unsafe_execution = allow_unsafe_execution
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
        max_depth = int(request.payload.get("max_depth", -1))
        max_items = max(1, min(int(request.payload.get("max_items", 500)), 10000))

        root = stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            return BridgeResponse.failure(
                request.request_id,
                "InvalidRootPath",
                f"Invalid root path: {root_path}",
            )

        root_depth = len(str(root.GetPath()).rstrip("/").split("/"))
        prims = []
        # continue skips emitting a prim but still descends its subtree, so the
        # depth limit bounded the response and not the work. Prune instead —
        # and the max_items break below is unreachable while we continue past
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
            prims.append(
                {
                    "path": path_str,
                    "type": prim_type_name,
                    "name": prim.GetName(),
                    "active": prim.IsActive(),
                }
            )
            if len(prims) >= max_items:
                break

        return BridgeResponse.success(
            request.request_id,
            {
                "transport": "simul_bridge",
                "root_path": root_path,
                "type_filter": prim_type,
                "count": len(prims),
                "truncated": len(prims) >= max_items,
                "prims": prims,
            },
        )

    def _handle_get_prim_info(self, request: BridgeRequest) -> BridgeResponse:
        """Return detailed information about a single prim."""
        import omni.usd
        from pxr import Gf, UsdGeom, UsdShade

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

        child_types: dict[str, int] = {}
        children = [str(child.GetPath()) for child in prim.GetChildren()]
        for child in prim.GetChildren():
            child_type = child.GetTypeName() or "Typeless"
            child_types[child_type] = child_types.get(child_type, 0) + 1

        attrs: dict[str, Any] = {}
        for attr in prim.GetAttributes():
            try:
                # Bulk geometry only: Get() would decompress the whole array
                # out of the crate layer just for _serialize_value to replace
                # it with an element count. Small arrays such as xformOpOrder
                # still come back in full — they carry information the caller
                # needs and cost nothing to read.
                attr_name = attr.GetName()
                if attr_name in BULK_GEOMETRY_ATTRIBUTES:
                    type_name = attr.GetTypeName()
                    if getattr(type_name, "isArray", False):
                        attrs[attr_name] = f"<array {type_name}>"
                        continue
                value = attr.Get()
                if value is not None:
                    attrs[attr.GetName()] = self._serialize_value(value)
            except Exception:
                attrs[attr.GetName()] = "<unreadable>"

        transform = None
        if prim.IsA(UsdGeom.Xformable):
            xformable = UsdGeom.Xformable(prim)
            matrix = xformable.ComputeLocalToWorldTransform(
                self._default_time_code()
            )
            quat = matrix.ExtractRotation().GetQuat()
            scale = [1.0, 1.0, 1.0]
            for op in xformable.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                    value = op.Get()
                    if value is not None:
                        scale = list(Gf.Vec3d(value))
                    break
            transform = {
                "translation": list(matrix.ExtractTranslation()),
                "rotation_quat": [quat.GetReal()] + list(quat.GetImaginary()),
                "scale": scale,
            }

        material_bindings = []
        try:
            bindings = UsdShade.MaterialBindingAPI(prim)
            material, _ = bindings.ComputeBoundMaterial()
            if material:
                material_bindings.append(str(material.GetPath()))
        except Exception:
            pass

        purpose = None
        visibility = None
        if prim.IsA(UsdGeom.Imageable):
            imageable = UsdGeom.Imageable(prim)
            purpose = imageable.ComputePurpose()
            visibility = imageable.ComputeVisibility()

        return BridgeResponse.success(
            request.request_id,
            {
                "transport": "simul_bridge",
                "path": prim_path,
                "name": prim.GetName(),
                "type": prim.GetTypeName(),
                "is_active": prim.IsActive(),
                "is_defined": prim.IsDefined(),
                "is_instance": prim.IsInstance(),
                "purpose": purpose,
                "visibility": visibility,
                "children_count": len(children),
                "children_types": child_types,
                "children": children[:50],
                "material_bindings": material_bindings,
                "transform": transform,
                "attributes": attrs,
            },
        )

    def _handle_get_prim_transform(self, request: BridgeRequest) -> BridgeResponse:
        """Return local or world transform for a prim."""
        import omni.usd
        from pxr import Gf, UsdGeom

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
        if not prim.IsA(UsdGeom.Xformable):
            return BridgeResponse.failure(
                request.request_id,
                "PrimNotXformable",
                f"Prim is not Xformable: {prim_path}",
            )

        xformable = UsdGeom.Xformable(prim)
        if world_space:
            matrix = xformable.ComputeLocalToWorldTransform(self._default_time_code())
        else:
            matrix = xformable.GetLocalTransformation(self._default_time_code())

        quat = matrix.ExtractRotation().GetQuat()
        scale = [1.0, 1.0, 1.0]
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                value = op.Get()
                if value is not None:
                    scale = list(Gf.Vec3d(value))
                break

        return BridgeResponse.success(
            request.request_id,
            {
                "transport": "simul_bridge",
                "prim_path": prim_path,
                "space": "world" if world_space else "local",
                "translation": list(matrix.ExtractTranslation()),
                "rotation_quat": [quat.GetReal()] + list(quat.GetImaginary()),
                "scale": scale,
            },
        )

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
        query = str(request.payload.get("query", "Mesh"))
        root_path = str(request.payload.get("root_path", "/"))
        max_results = max(1, min(int(request.payload.get("max_results", 100)), 10000))

        root = stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            return BridgeResponse.failure(
                request.request_id,
                "InvalidRootPath",
                f"Invalid root path: {root_path}",
            )

        matches = []
        for prim in Usd.PrimRange(root):
            if search_type == "type":
                match = prim.GetTypeName() == query
            elif search_type == "name":
                match = query.lower() in prim.GetName().lower()
            else:
                return BridgeResponse.failure(
                    request.request_id,
                    "InvalidSearchType",
                    f"Unsupported search_type: {search_type}",
                )
            if match:
                matches.append(
                    {
                        "path": str(prim.GetPath()),
                        "type": prim.GetTypeName(),
                        "name": prim.GetName(),
                    }
                )
            if len(matches) >= max_results:
                break

        return BridgeResponse.success(
            request.request_id,
            {
                "transport": "simul_bridge",
                "search_type": search_type,
                "query": query,
                "root_path": root_path,
                "count": len(matches),
                "truncated": len(matches) >= max_results,
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
            import omni.physx

            physx = omni.physx.get_physx_interface()
            # get_physics_stats and is_cuda_lib_present live on PhysXUnitTests,
            # not on the PhysX object acquire_physx_interface returns, in both
            # Kit 107 (Isaac Sim 5.1) and Kit 110 (6.0). Probe rather than call
            # blind, and say so when neither is reachable — an empty dict alone
            # reads like an empty physics scene.
            physics: dict[str, Any] = {}
            available = False
            if hasattr(physx, "get_physics_stats"):
                stats = physx.get_physics_stats()
                if isinstance(stats, dict):
                    physics.update(stats)
                available = True
            if hasattr(physx, "is_cuda_lib_present"):
                physics["cuda_available"] = physx.is_cuda_lib_present()
                available = True
            if not available:
                physics["stats_unavailable"] = (
                    "omni.physx interface exposes no get_physics_stats or "
                    "is_cuda_lib_present on this build"
                )
            info["physics"] = physics
        except Exception as exc:
            info["physics_error"] = str(exc)

        try:
            import carb.settings

            settings = carb.settings.get_settings()
            info["physics_config"] = {
                "gpu_dynamics_enabled": settings.get("/physics/gpuDynamicsEnabled"),
                "physics_dt": settings.get("/persistent/simulation/defaultPhysicsDt"),
                "solver_type": settings.get("/persistent/physics/solverType"),
            }
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
            if stage:
                info["stage"] = {
                    "url": ctx.get_stage_url(),
                    "prim_count": sum(1 for _ in stage.Traverse()),
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
            if timeline.is_stopped():
                timeline.play()
                for _ in range(3):
                    await omni.kit.app.get_app().next_update_async()
            for _ in range(num_steps):
                await omni.kit.app.get_app().next_update_async()
            return BridgeResponse.success(
                request.request_id,
                {
                    "transport": "simul_bridge",
                    "steps": num_steps,
                    "current_time": timeline.get_current_time(),
                    "state": "playing" if timeline.is_playing() else "paused",
                },
            )
        return BridgeResponse.failure(
            request.request_id,
            "InvalidSimulationCommand",
            f"Unsupported simulation command: {command}",
        )

    @staticmethod
    def _default_time_code() -> Any:
        """Return the USD default time code."""
        from pxr import Usd

        return Usd.TimeCode.Default()

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        """Serialize common USD values into JSON-safe Python objects."""
        from pxr import Gf

        if value is None:
            return None
        if isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(
            value,
            (
                Gf.Vec2f,
                Gf.Vec2d,
                Gf.Vec2h,
                Gf.Vec2i,
                Gf.Vec3f,
                Gf.Vec3d,
                Gf.Vec3h,
                Gf.Vec3i,
                Gf.Vec4f,
                Gf.Vec4d,
                Gf.Vec4h,
                Gf.Vec4i,
            ),
        ):
            return [float(item) for item in value]
        if isinstance(value, (Gf.Quatf, Gf.Quatd, Gf.Quath)):
            return [float(value.GetReal())] + [
                float(item) for item in value.GetImaginary()
            ]
        if isinstance(value, (Gf.Matrix4d, Gf.Matrix4f, Gf.Matrix3d, Gf.Matrix3f)):
            return str(type(value).__name__)
        try:
            if hasattr(value, "__len__") and not isinstance(value, str):
                if len(value) > 16:
                    return f"[{len(value)} elements]"
                return [BridgeCommandService._serialize_value(item) for item in value]
        except Exception:
            pass
        try:
            return float(value)
        except Exception:
            return str(value)
