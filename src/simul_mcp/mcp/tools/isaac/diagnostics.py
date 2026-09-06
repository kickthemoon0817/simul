"""Runtime diagnostics tools for Isaac Sim."""

import json
import textwrap
from typing import Any, Callable, Dict, List, Optional

from ....adapters import IsaacSocketClient, ScriptResult
from ...schemas.common import ErrorResponse
from ._shared import (
    PROTECTED_EXTENSIONS,
    BULK_GEOMETRY_ATTRIBUTES,
    LOG_SCAN_WINDOW_BYTES,
    MAX_CAPTURE_DIMENSION,
    MAX_INLINE_CAPTURE_BYTES,
    MAX_RETAINED_CAPTURES,
    MAX_SCRIPT_BYTES,
    PRIM_DETAIL_ASPECTS,
    FloatList,
    _pyval,
    logger,
)


class DiagnosticsMixin:
    # ------------------------------------------------------------------
    # Runtime diagnostics
    # ------------------------------------------------------------------

    async def get_runtime_info(
        self, include_prim_count: bool = False
    ) -> Dict[str, Any]:
        """
        Get consolidated runtime diagnostics from the running Isaac Sim instance.

        Collects Kit app info, timeline state, physics stats, GPU/renderer
        info, and viewport state in a single call.

        Args:
            include_prim_count: Also count every prim on the stage for
                ``stage.prim_count``. The count walks the whole stage on Kit's
                main thread, so it is off unless asked for; the key is then
                ``None``.

        Returns:
            Dict with app, timeline, physics, renderer, and viewport sections.
        """
        bridge_result = await self._execute_bridge_action(
            "get_runtime_info", {"include_prim_count": include_prim_count}
        )
        if bridge_result is not None:
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

            # Physics stats
            try:
                import omni.physx
                physx = omni.physx.get_physx_interface()
                # These live on PhysXUnitTests, not on the PhysX object this
                # returns, in both Kit 107 (5.1) and Kit 110 (6.0). Probe
                # rather than call blind, and say when neither is reachable.
                physics = {{}}
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
            except Exception as e:
                info["physics_error"] = str(e)

            # Physics scene settings
            try:
                import carb.settings
                settings = carb.settings.get_settings()
                gpu_dynamics = settings.get("/physics/gpuDynamicsEnabled")
                physics_dt = settings.get("/persistent/simulation/defaultPhysicsDt")
                solver_type = settings.get("/persistent/physics/solverType")
                info["physics_config"] = {{
                    "gpu_dynamics_enabled": gpu_dynamics,
                    "physics_dt": physics_dt,
                    "solver_type": solver_type,
                }}
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
        return await self._execute_json_script(script, transport_mode=mode)

    async def get_isaac_logs(
        self,
        level: str = "warn",
        last_n: int = 50,
        source_filter: Optional[str] = None,
        search: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Read recent log entries from the running Isaac Sim instance.

        Reads the Kit log file and returns the last N entries, optionally
        filtered by level and source module.

        Args:
            level: Minimum log level to return: "verbose", "info", "warn", "error".
                   Defaults to "warn" (warnings and errors only).
            last_n: Number of most recent matching entries to return. Max 500.
            source_filter: Optional source module substring filter (e.g. "physx", "omni.usd").
            search: Optional text search within log messages.

        Returns:
            Dict with log entries, counts per level, and log file path.
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

            # Find the most recent Kit log file
            log_dir = os.path.expanduser("~/.nvidia-omniverse/logs/Kit")
            log_path = None
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
                        message = m.group(3).strip()

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
                print(json.dumps({{
                    "log_file": log_path,
                    "log_size_bytes": log_size,
                    "scanned_bytes": log_size - scan_start,
                    "truncated_scan": scan_start > 0,
                    "total_matching": matched,
                    "returned": len(tail),
                    "counts": counts,
                    "entries": tail,
                }}))
        """)
        return await self._execute_json_script(script)

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

    async def disable_isaac_extension(self, extension_id: str) -> Dict[str, Any]:
        """
        Disable an extension by its ID in the running Isaac Sim instance.

        Accepts either the bare canonical extension name (e.g. "worv.env.sun")
        — which is what ``omni.kit.app.IExtensionManager.set_extension_enabled_immediate``
        natively takes — or the fully version-suffixed ID returned by
        ``list_isaac_extensions`` (e.g. "worv.env.sun-0.3.0").

        Args:
            extension_id: The extension name or fully qualified ID
                (e.g. "isaacsim.core.utils", "omni.physx", "worv.env.sun-0.3.0").

        Returns:
            Dict with success status and extension info after disabling, or a
            RefusedOperation error for the extensions the MCP transport runs on.
        """
        bare_name = extension_id.rsplit("-", 1)[0] if "-" in extension_id else extension_id
        if bare_name in PROTECTED_EXTENSIONS:
            return self._refusal(
                f"Refusing to disable {extension_id!r}: simul-mcp reaches Isaac Sim "
                "through it, and disabling it would leave this instance unreachable.",
                extension_id=extension_id,
                protected_extensions=sorted(PROTECTED_EXTENSIONS),
            )
        _ext_id = repr(extension_id)
        script = textwrap.dedent(f"""\
            import json
            import omni.kit.app

            ext_id = {_ext_id}
            ext_manager = omni.kit.app.get_app().get_extension_manager()
            ext_manager.set_extension_enabled_immediate(ext_id, False)

            # Verify by matching on bare name or fully qualified ID — both are
            # accepted by the Kit extension manager, but get_extensions() always
            # returns the version-suffixed form in 'id' and the bare form in 'name'.
            bare_query = ext_id.rsplit("-", 1)[0] if "-" in ext_id else ext_id
            found = False
            for ext in ext_manager.get_extensions():
                eid = ext.get("id", "")
                ename = ext.get("name", "")
                if ext_id in (eid, ename) or bare_query == ename:
                    found = True
                    print(json.dumps({{
                        "extension_id": eid or ename,
                        "name": ename,
                        "enabled": ext.get("enabled", False),
                        "version": ext.get("version", ""),
                    }}))
                    break

            if not found:
                print(json.dumps({{
                    "success": False,
                    "error": "Extension not found: " + ext_id,
                }}))
        """)
        return await self._execute_json_script(script)
