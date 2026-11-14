"""Runtime diagnostics tools for Isaac Sim."""

import textwrap
from typing import Any, Dict, Optional

from ... import error_types
from ...schemas.common import ErrorResponse
from ._shared import (
    PROTECTED_EXTENSIONS,
    LOG_SCAN_WINDOW_BYTES,
    _pyval,
)
from .._meta import tool_meta


class DiagnosticsMixin:
    # ------------------------------------------------------------------
    # Runtime diagnostics
    # ------------------------------------------------------------------

    @tool_meta(
        name="get_isaac_runtime_info",
        description=(
            "Get consolidated runtime diagnostics from the running Isaac Sim instance: Kit "
            "app version, timeline state, where physics writes (CUDA device, "
            "updateToUsd, omni.physx.fabric, resetOnStop), the physics scene's rate, GPU "
            "dynamics and solver, renderer settings, viewport state, stage summary, and "
            "extension counts. Use this for a quick health check or to understand the "
            "current state of the simulation environment. Pass include_prim_count=true to "
            "also count every prim on the stage. While the instance is held after a timed-out "
            "call it answers at once with busy and held_after_timeout instead."
        ),
        read_only=True,
        idempotent=True,
        reports_hold=True,
    )
    async def get_runtime_info(
        self, include_prim_count: bool = False
    ) -> Dict[str, Any]:
        """
        Get consolidated runtime diagnostics from the running Isaac Sim instance.

        Collects Kit app info, timeline state, the physics write path and
        scene settings, renderer settings and viewport state in a single
        call. ``physics`` holds the Carb settings Isaac Sim's
        SimulationManager reads and sets; ``physics_config`` the physics
        scene's PhysxSceneAPI rate, GPU dynamics, solver and broadphase.

        Args:
            include_prim_count: Also count every prim on the stage for
                ``stage.prim_count``. The count walks the whole stage on Kit's
                main thread, so it is off unless asked for; the key is then
                ``None``.

        Returns:
            Dict with app, timeline, physics, renderer, and viewport sections,
            plus ``client`` describing this side of the wire. Over the bridge
            the payload also carries ``bridge`` (``busy``, ``busy_since``,
            ``busy_for_seconds``, ``current_action``), which is how a caller
            tells an instance that is working from one that is hung. While
            the instance is held after a timed-out call (by this server or
            another on the machine), the MCP tool answers at once from the
            server's own state instead: ``busy``, ``held_after_timeout`` and
            ``client``, with the engine sections left out.
        """
        bridge_result = await self._execute_bridge_action(
            "get_runtime_info", {"include_prim_count": include_prim_count}
        )
        if bridge_result is not None:
            bridge_result["client"] = self._client_state()
            return bridge_result

        _include_prim_count = _pyval(bool(include_prim_count))
        script = textwrap.dedent(f"""\
            import json
            import sys
            import time as _time

            info = {{}}

            # Kit app info
            try:
                import omni.kit.app
                app = omni.kit.app.get_app()
                info["app"] = {{
                    "version": str(app.get_build_version()),
                    "python_version": sys.version.split()[0],
                    "update_number": int(app.get_update_number()),
                }}
                # The Kit build string above is not the Isaac Sim release, and
                # the release is what callers key API namespaces off. The
                # bridge reports the same field; agents must not have to know
                # which transport answered.
                if hasattr(app, "get_app_version"):
                    info["app"]["isaac_version"] = str(app.get_app_version())
            except Exception as e:
                info["app_error"] = str(e)

            # Timeline state
            try:
                import omni.timeline
                tl = omni.timeline.get_timeline_interface()
                info["timeline"] = {{
                    "is_playing": tl.is_playing(),
                    "is_stopped": tl.is_stopped(),
                    "current_time": tl.get_current_time(),
                    "start_time": tl.get_start_time(),
                    "end_time": tl.get_end_time(),
                    "fps": tl.get_time_codes_per_second(),
                }}
            except Exception as e:
                info["timeline_error"] = str(e)

            # Where physics writes: the Carb settings and extension Isaac Sim's
            # SimulationManager reads and sets, plus the app's resetOnStop.
            # The omni.physx interface has no physics statistics to report.
            try:
                import carb.settings
                settings = carb.settings.get_settings()
                physics = {{
                    "cuda_device": settings.get("/physics/cudaDevice"),
                    "suppress_readback": settings.get("/physics/suppressReadback"),
                    "update_to_usd": settings.get("/physics/updateToUsd"),
                    "update_velocities_to_usd": settings.get("/physics/updateVelocitiesToUsd"),
                    "reset_on_stop": settings.get("/physics/resetOnStop"),
                    "physx_fabric_enabled": None,
                }}
                try:
                    import omni.kit.app
                    manager = omni.kit.app.get_app().get_extension_manager()
                    physics["physx_fabric_enabled"] = bool(
                        manager.is_extension_enabled("omni.physx.fabric")
                    )
                except Exception:
                    pass
                info["physics"] = physics
            except Exception as e:
                info["physics_error"] = str(e)

            # The physics scene's own settings, on its PhysxSceneAPI; searched
            # in the top two levels only (/physicsScene, /World/physicsScene).
            try:
                import omni.usd
                from pxr import PhysxSchema, Usd, UsdPhysics
                config = {{
                    "scene_path": None,
                    "time_steps_per_second": None,
                    "physics_dt": None,
                    "gpu_dynamics_enabled": None,
                    "solver_type": None,
                    "broadphase_type": None,
                }}
                stage = omni.usd.get_context().get_stage()
                if stage is not None:
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
                info["physics_config"] = config
            except Exception as e:
                info["physics_config_error"] = str(e)

            # Renderer info
            try:
                import carb.settings
                settings = carb.settings.get_settings()
                info["renderer"] = {{
                    "active_gpu": settings.get("/renderer/activeGpu"),
                    "gpu_name": settings.get("/renderer/gpuName"),
                    "hgi_driver": settings.get("/renderer/hgi/driver"),
                    "raytracing_mode": settings.get("/rtx/rendermode"),
                    "realtime_mode": settings.get("/rtx/ecoMode/enabled"),
                }}
            except Exception as e:
                info["renderer_error"] = str(e)

            # Viewport info
            try:
                from omni.kit.viewport.utility import get_active_viewport
                viewport = get_active_viewport()
                if viewport:
                    info["viewport"] = {{
                        "camera_path": str(viewport.camera_path),
                        "resolution": list(viewport.resolution),
                        "fps": viewport.fps if hasattr(viewport, "fps") else None,
                    }}
                else:
                    info["viewport"] = {{"status": "no active viewport"}}
            except Exception as e:
                info["viewport_error"] = str(e)

            # Stage summary
            try:
                import omni.usd
                ctx = omni.usd.get_context()
                stage = ctx.get_stage()
                if stage:
                    info["stage"] = {{
                        "url": ctx.get_stage_url(),
                        "prim_count": (
                            sum(1 for _ in stage.Traverse())
                            if {_include_prim_count} else None
                        ),
                    }}
                else:
                    info["stage"] = {{"status": "no stage open"}}
            except Exception as e:
                info["stage_error"] = str(e)

            # Extensions summary
            try:
                import omni.kit.app
                ext_mgr = omni.kit.app.get_app().get_extension_manager()
                all_exts = ext_mgr.get_extensions()
                enabled = [e for e in all_exts if e.get("enabled")]
                info["extensions"] = {{
                    "total": len(all_exts),
                    "enabled": len(enabled),
                }}
            except Exception as e:
                info["extensions_error"] = str(e)

            print(json.dumps(info))
        """)
        mode = self._raw_script_transport_mode
        result = await self._execute_json_script(script, transport_mode=mode)
        if result.get("success", True):
            result["client"] = self._client_state()
        return result

    @tool_meta(
        name="interrupt_isaac_script",
        description=(
            "Stop the script the Isaac Sim bridge is currently running, so a runaway "
            "script does not hold the instance for every agent. Reaches a "
            "coroutine script or one suspended at an await; a synchronous script that never "
            "yields also blocks the bridge's event loop, so that one is stopped by the "
            "per-request timeout sent with every script instead. Requires the bridge "
            "extension; the stock Python socket has no interrupt path. Check "
            "get_isaac_runtime_info's bridge section (busy, busy_since, current_action) "
            "first to tell a busy instance from a hung one."
        ),
        read_only=False,
        idempotent=True,
        bypasses_instance_lock=True,
    )
    async def interrupt_script(self) -> Dict[str, Any]:
        """
        Stop the script the bridge is currently running.

        Only the bridge has an interrupt path. The stock python_server socket
        honours a per-request timeout only by replying with a TimeoutError;
        the script itself keeps running inside Kit, and nothing can stop it
        early. The request skips the client's request lock and circuit
        breaker so it reaches the bridge while a runaway call still holds
        both.

        Returns:
            Dict with ``interrupted``, ``was_busy``, ``phase``,
            ``current_action`` and ``busy_for_seconds`` from the bridge, or an
            ErrorResponse dict when no bridge is configured or reachable.
        """
        client = self._client
        if not client.bridge_enabled:
            return ErrorResponse(
                error=(
                    "interrupt needs the simul bridge extension; this instance "
                    "is configured for the stock Python socket only, which "
                    "cannot stop a running script. Scripts sent to it carry a "
                    f"{client.script_timeout_seconds}s timeout, but on 6.0 that "
                    "only makes python_server reply early (the 5.x socket has "
                    "none): an overrunning script keeps running inside Kit."
                ),
                error_type=error_types.BRIDGE_UNAVAILABLE,
            ).model_dump()
        try:
            response = await client.interrupt_bridge_script()
        except (ConnectionRefusedError, TimeoutError, OSError, ValueError) as exc:
            return ErrorResponse(
                error=(
                    f"Could not reach the bridge at {client.bridge_endpoint} to "
                    f"interrupt: {exc}. A synchronous script that never yields "
                    "also blocks the bridge's event loop, so it cannot take this "
                    "request; its per-request timeout is what stops it."
                ),
                **error_types.exception_fields(exc),
            ).model_dump()
        return self._bridge_response_envelope(response)

    def _client_state(self) -> Dict[str, Any]:
        """Describe this client's transport state for diagnostics payloads."""
        client = self._client
        return {
            "bridge_enabled": client.bridge_enabled,
            "bridge_address": client.bridge_address,
            "bridge_circuit_open": client.bridge_circuit_open,
            "bridge_consecutive_failures": client.bridge_consecutive_failures,
            "vscode_address": client.vscode_address,
            "socket_protocol": client.socket_protocol,
            "script_timeout_seconds": client.script_timeout_seconds,
        }

    @tool_meta(
        name="get_isaac_logs",
        description=(
            "Read recent log entries from this Isaac Sim instance's Kit log (the file "
            "Kit names in its /log/file setting). Returns entries filtered by minimum level "
            "(verbose/info/warn/error), optional source module filter, and optional text "
            "search, with per-level counts, the log file path and log_file_source; when "
            "Kit does not name its file, the newest Kit log is read and a warning says it "
            "may belong to another process. Use this to diagnose errors, warnings, or check "
            "what happened during a simulation."
        ),
        read_only=True,
        idempotent=True,
    )
    async def get_isaac_logs(
        self,
        level: str = "warn",
        last_n: int = 50,
        source_filter: Optional[str] = None,
        search: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Read recent log entries from the running Isaac Sim instance.

        Reads this instance's Kit log, the path Kit keeps in the
        ``/log/file`` Carb setting, and returns the last N entries,
        optionally filtered by level and source module. When the setting
        is unavailable the newest log under ``~/.nvidia-omniverse/logs/Kit``
        is read instead, with ``log_file_source`` "newest_kit_log" and a
        ``warning``: with several Kit processes it may be another's.

        Args:
            level: Minimum log level to return: "verbose", "info", "warn", "error".
                   Defaults to "warn" (warnings and errors only).
            last_n: Number of most recent matching entries to return. Max 500.
            source_filter: Optional source module substring filter (e.g. "physx", "omni.usd").
            search: Optional text search within log messages.

        Returns:
            Dict with log entries, counts per level, ``log_file`` and
            ``log_file_source`` ("carb:/log/file" or "newest_kit_log").
        """
        last_n = max(1, min(last_n, 500))
        _level = _pyval(level.lower())
        _last_n = last_n
        _source_filter = _pyval(source_filter)
        _search = _pyval(search)
        script = textwrap.dedent(f"""\
            import collections
            import json
            import re
            import os

            # Counts and totals below describe this window, not the whole file.
            SCAN_WINDOW_BYTES = {LOG_SCAN_WINDOW_BYTES}

            level_filter = {_level}
            last_n = {_last_n}
            source_filter = {_source_filter}
            search_text = {_search}

            level_priority = {{"verbose": 0, "info": 1, "warn": 2, "warning": 2, "error": 3, "fatal": 4}}
            min_priority = level_priority.get(level_filter, 2)

            # This instance's log: Kit keeps the path of the file it writes in
            # the /log/file Carb setting (Isaac Sim's own console_log skill
            # reads it the same way). Only when that is unavailable fall back
            # to the newest log under ~/.nvidia-omniverse/logs/Kit, which can
            # belong to another Kit process, and say so.
            log_path = None
            log_source = None
            source_warning = None
            try:
                import carb.settings
                configured = carb.settings.get_settings().get("/log/file")
                if configured and os.path.isfile(str(configured)):
                    log_path = str(configured)
                    log_source = "carb:/log/file"
            except Exception:
                pass
            if log_path is None:
                log_dir = os.path.expanduser("~/.nvidia-omniverse/logs/Kit")
                newest_mtime = -1.0
                for root, dirs, files in os.walk(log_dir):
                    for f in files:
                        if not f.endswith(".log"):
                            continue
                        full = os.path.join(root, f)
                        try:
                            mtime = os.path.getmtime(full)
                        except OSError:
                            continue
                        if mtime > newest_mtime:
                            newest_mtime = mtime
                            log_path = full
                if log_path:
                    log_source = "newest_kit_log"
                    source_warning = (
                        "Kit did not expose this instance's log file (/log/file), so this "
                        "is the newest log under ~/.nvidia-omniverse/logs/Kit, which may "
                        "belong to another Kit process."
                    )

            # Transport tokens stay out of the result. Isaac Sim 6.0.1's
            # python_server logs "Python server authentication token: <token>"
            # at Info level when it generates one. Redaction runs before the
            # search filter, so a search cannot probe the token either.
            secrets = []
            try:
                import carb.settings
                _settings = carb.settings.get_settings()
                for _key in (
                    "/exts/isaacsim.code_editor.python_server/auth_token",
                    "/exts/khemoo.simul/auth_token",
                ):
                    _value = _settings.get(_key)
                    if _value is not None and not isinstance(_value, bool) and len(str(_value).strip()) >= 4:
                        secrets.append(str(_value).strip())
            except Exception:
                pass
            token_pattern = re.compile(
                r'((?:authentication|auth)[ _-]?token["\\']?\\s*[:=]\\s*["\\']?)[^\\s"\\',]+',
                re.IGNORECASE,
            )
            redacted = [0]

            def _redact(text):
                original = text
                for secret in secrets:
                    text = text.replace(secret, "***")
                text = token_pattern.sub(lambda match: match.group(1) + "***", text)
                if text != original:
                    redacted[0] += 1
                return text

            if not log_path:
                print(json.dumps({{"error": "No Kit log file found"}}))
            else:
                pattern = re.compile(
                    r'\\[(Info|Warning|Warn|Error|Fatal|Verbose)\\]\\s*\\[([^\\]]+)\\]\\s*(.*)',
                    re.IGNORECASE
                )
                # Only the tail is ever returned, so only the tail is read.
                # Kit logs reach multiple gigabytes, and this runs on the main
                # thread — reading one to hand back 50 lines freezes the sim.
                entries = collections.deque(maxlen=last_n)
                matched = 0
                counts = {{"verbose": 0, "info": 0, "warn": 0, "error": 0}}
                log_size = os.path.getsize(log_path)
                scan_start = max(0, log_size - SCAN_WINDOW_BYTES)

                with open(log_path, "r", errors="replace") as f:
                    if scan_start:
                        f.seek(scan_start)
                        f.readline()  # drop the partial line we landed inside
                        scan_start = f.tell()
                    for line in f:
                        m = pattern.search(line)
                        if not m:
                            continue
                        raw_level = m.group(1).lower()
                        if raw_level == "warning":
                            raw_level = "warn"
                        source = m.group(2)
                        message = _redact(m.group(3).strip())

                        if raw_level in counts:
                            counts[raw_level] += 1

                        priority = level_priority.get(raw_level, 0)
                        if priority < min_priority:
                            continue
                        if source_filter and source_filter.lower() not in source.lower():
                            continue
                        if search_text and search_text.lower() not in message.lower():
                            continue

                        timestamp = ""
                        ts_match = re.match(r'(\\d{{4}}-\\d{{2}}-\\d{{2}}T[\\d:]+Z)', line)
                        if ts_match:
                            timestamp = ts_match.group(1)

                        matched += 1
                        entries.append({{
                            "timestamp": timestamp,
                            "level": raw_level,
                            "source": source,
                            "message": message[:500],
                        }})

                tail = list(entries)
                payload = {{
                    "log_file": log_path,
                    "log_file_source": log_source,
                    "log_size_bytes": log_size,
                    "scanned_bytes": log_size - scan_start,
                    "truncated_scan": scan_start > 0,
                    "total_matching": matched,
                    "returned": len(tail),
                    "counts": counts,
                    "entries": tail,
                }}
                if source_warning:
                    payload["warning"] = source_warning
                if redacted[0]:
                    payload["redacted_lines"] = redacted[0]
                print(json.dumps(payload))
        """)
        return await self._execute_json_script(script)

    @tool_meta(
        name="set_isaac_log_level",
        description=(
            "Set the Carbonite logging threshold for the running Isaac Sim. Levels: "
            "verbose, info, warn, error. Lower levels include all higher levels. Use "
            "'verbose' for maximum detail when debugging."
        ),
        read_only=False,
        destructive=True,
        idempotent=True,
    )
    async def set_isaac_log_level(self, level: str) -> Dict[str, Any]:
        """
        Set the Carbonite logging threshold level for the running Isaac Sim instance.

        Args:
            level: Log level: "verbose", "info", "warn", "error".

        Returns:
            Dict with the previous and new log level.
        """
        _level = _pyval(level.lower())
        script = textwrap.dedent(f"""\
            import json
            import carb.logging

            logging = carb.logging.acquire_logging()
            level_map = {{
                "verbose": carb.logging.LEVEL_VERBOSE,
                "info": carb.logging.LEVEL_INFO,
                "warn": carb.logging.LEVEL_WARN,
                "error": carb.logging.LEVEL_ERROR,
            }}
            level_names = {{v: k for k, v in level_map.items()}}

            requested = {_level}
            if requested not in level_map:
                print(json.dumps({{"error": f"Invalid level: {{requested}}. Use: verbose, info, warn, error"}}))
            else:
                old_level = logging.get_level_threshold()
                old_name = level_names.get(old_level, str(old_level))
                logging.set_level_threshold(level_map[requested])
                new_level = logging.get_level_threshold()
                new_name = level_names.get(new_level, str(new_level))
                print(json.dumps({{
                    "previous_level": old_name,
                    "current_level": new_name,
                }}))
        """)
        return await self._execute_json_script(script)

    @tool_meta(
        name="disable_isaac_extension",
        description=(
            "Disable an extension immediately in the running Isaac Sim instance, and fail when "
            "it is still enabled afterwards (an enabled extension depending on it keeps it "
            "loaded). Refuses the transport "
            "extensions simul speaks through (khemoo.simul, "
            "isaacsim.code_editor.python_server, isaacsim.code_editor.vscode)."
        ),
        read_only=False,
        destructive=True,
        idempotent=True,
    )
    async def disable_isaac_extension(self, extension_id: str) -> Dict[str, Any]:
        """
        Disable an extension by its ID in the running Isaac Sim instance.

        Args:
            extension_id: Bare Kit extension name (preferred, e.g. "omni.physx")
                or the version-suffixed ID from list_isaac_extensions
                (e.g. "omni.physx-107.3.7").

        Returns:
            Dict with success status and extension info after disabling, or a
            RefusedOperation error for the extensions the MCP transport runs on.
        """
        # Names hold dots, never hyphens; a version may hold hyphens
        # (1.0.0-rc1), so the name ends at the first one.
        bare_name = extension_id.strip().split("-", 1)[0]
        if bare_name in PROTECTED_EXTENSIONS:
            return self._refusal(
                f"Refusing to disable {extension_id!r}: simul reaches Isaac Sim "
                "through it, and disabling it would leave this instance unreachable.",
                extension_id=extension_id,
                protected_extensions=sorted(PROTECTED_EXTENSIONS),
            )
        _ext_id = _pyval(extension_id)
        script = textwrap.dedent(f"""\
            import json
            import omni.kit.app

            ext_id = {_ext_id}
            ext_manager = omni.kit.app.get_app().get_extension_manager()
            ext_manager.set_extension_enabled_immediate(ext_id, False)

            # Verify by matching on bare name or fully qualified ID — both are
            # accepted by the Kit extension manager, but get_extensions() always
            # returns the version-suffixed form in 'id' and the bare form in 'name'.
            bare_query = ext_id.split("-", 1)[0]
            found = False
            for ext in ext_manager.get_extensions():
                eid = ext.get("id", "")
                ename = ext.get("name", "")
                if ext_id in (eid, ename) or bare_query == ename:
                    found = True
                    enabled = bool(ext.get("enabled", False))
                    payload = {{
                        "extension_id": eid or ename,
                        "name": ename,
                        "enabled": enabled,
                        "version": ext.get("version", ""),
                    }}
                    if enabled:
                        # The manager keeps an extension that another enabled
                        # one depends on; without an explicit failure the
                        # envelope would report success for a no-op.
                        payload["success"] = False
                        payload["error"] = (
                            "Extension is still enabled: " + (eid or ename)
                            + " (an enabled extension may depend on it; check the Kit log)"
                        )
                    print(json.dumps(payload))
                    break

            if not found:
                print(json.dumps({{
                    "success": False,
                    "error": "Extension not found: " + ext_id,
                }}))
        """)
        return await self._execute_json_script(script)
