"""GUI/state observability tools for Isaac Sim.

Both tools degrade gracefully on headless runs (``--no-window``): sections
that need ``omni.ui`` report ``available: False`` with the reason instead of
failing the whole call.
"""

import textwrap
from typing import Any, Dict

from ._shared import _pyval


class UiObservabilityMixin:
    # ------------------------------------------------------------------
    # GUI / application state observability
    # ------------------------------------------------------------------

    async def get_isaac_ui_state(self) -> Dict[str, Any]:
        """
        Get a consolidated snapshot of the Isaac Sim GUI and app state.

        Collects app-window config, the omni.ui workspace (windows,
        visibility, focus, docking), active viewport, current selection,
        timeline state, and open-stage status in one call.

        Returns:
            Dict with app_window, ui, viewport, selection, timeline, and
            stage sections; failed sections carry an error note instead.
        """
        script = textwrap.dedent("""\
            import json

            state = {}

            try:
                import carb.settings
                settings = carb.settings.get_settings()
                state["app_window"] = {
                    "window_enabled": bool(settings.get("/app/window/enabled")),
                    "width": settings.get("/app/window/width"),
                    "height": settings.get("/app/window/height"),
                }
            except Exception as e:
                state["app_window_error"] = str(e)

            try:
                import omni.ui as ui
                windows = []
                focused = None
                for win in ui.Workspace.get_windows():
                    entry = {
                        "title": win.title,
                        "visible": bool(win.visible),
                        "focused": bool(getattr(win, "focused", False)),
                        "docked": bool(getattr(win, "docked", False)),
                        "width": getattr(win, "width", None),
                        "height": getattr(win, "height", None),
                    }
                    if entry["focused"]:
                        focused = entry["title"]
                    windows.append(entry)
                windows.sort(key=lambda w: (w["title"] is None, w["title"]))
                state["ui"] = {
                    "available": True,
                    "window_count": len(windows),
                    "focused_window": focused,
                    "windows": windows,
                }
            except Exception as e:
                state["ui"] = {"available": False, "reason": str(e)}

            try:
                from omni.kit.viewport.utility import get_active_viewport
                viewport = get_active_viewport()
                if viewport:
                    state["viewport"] = {
                        "camera_path": str(viewport.camera_path),
                        "resolution": list(viewport.resolution),
                    }
                else:
                    state["viewport"] = {"status": "no active viewport"}
            except Exception as e:
                state["viewport_error"] = str(e)

            try:
                import omni.usd
                ctx = omni.usd.get_context()
                paths = list(ctx.get_selection().get_selected_prim_paths())
                state["selection"] = {"count": len(paths), "paths": paths}
            except Exception as e:
                state["selection_error"] = str(e)

            try:
                import omni.timeline
                tl = omni.timeline.get_timeline_interface()
                if tl.is_playing():
                    playback = "playing"
                elif tl.is_stopped():
                    playback = "stopped"
                else:
                    playback = "paused"
                state["timeline"] = {
                    "state": playback,
                    "current_time": tl.get_current_time(),
                    "fps": tl.get_time_codes_per_seconds(),
                }
            except Exception as e:
                state["timeline_error"] = str(e)

            try:
                import omni.usd
                ctx = omni.usd.get_context()
                if ctx.get_stage() is not None:
                    state["stage"] = {
                        "url": ctx.get_stage_url(),
                        "pending_edits": bool(ctx.has_pending_edit()),
                    }
                else:
                    state["stage"] = {"status": "no stage open"}
            except Exception as e:
                state["stage_error"] = str(e)

            print(json.dumps(state))
        """)
        mode = self._raw_script_transport_mode
        return await self._execute_json_script(script, transport_mode=mode)

    async def get_isaac_ui_window(
        self,
        window_title: str,
        max_depth: int = 4,
        max_widgets: int = 400,
    ) -> Dict[str, Any]:
        """
        Inspect one omni.ui window as a widget tree.

        Walks the window's frame via ``omni.ui.Inspector`` and reports each
        widget's type, identifier/name, label text, basic state flags, and
        model value where one is readable.

        Args:
            window_title: Exact title of the window to inspect (as listed
                by get_isaac_ui_state).
            max_depth: Maximum tree depth to descend.
            max_widgets: Maximum number of widgets to describe before
                truncating.

        Returns:
            Dict with window metadata and the (possibly truncated) widget
            tree, or an error listing the available window titles.
        """
        _window_title = _pyval(window_title)
        _max_depth = _pyval(max_depth)
        _max_widgets = _pyval(max_widgets)
        prelude = textwrap.dedent(f"""\
            _TARGET = {_window_title}
            _MAX_DEPTH = {_max_depth}
            _MAX_WIDGETS = {_max_widgets}
        """)
        script = prelude + textwrap.dedent("""\
            import json

            try:
                import omni.ui as ui
            except Exception as e:
                print(json.dumps({
                    "error": "omni.ui is not available: " + str(e),
                    "error_type": "UiUnavailable",
                }))
            else:
                window = ui.Workspace.get_window(_TARGET)
                if window is None:
                    titles = sorted(
                        str(w.title) for w in ui.Workspace.get_windows()
                    )
                    print(json.dumps({
                        "error": "Window not found: " + _TARGET,
                        "available_windows": titles,
                    }))
                else:
                    budget = {"left": _MAX_WIDGETS, "truncated": False}

                    def describe(widget, depth):
                        if budget["left"] <= 0:
                            budget["truncated"] = True
                            return None
                        budget["left"] -= 1
                        node = {"type": type(widget).__name__}
                        for key in ("identifier", "name"):
                            value = getattr(widget, key, "")
                            if value:
                                node[key] = value
                        text = getattr(widget, "text", None)
                        if isinstance(text, str) and text:
                            node["text"] = text
                        for flag in ("visible", "enabled", "checked",
                                     "selected"):
                            try:
                                value = getattr(widget, flag)
                            except Exception:
                                continue
                            # Default-true flags are only worth reporting
                            # when off; checked/selected always carry state.
                            if isinstance(value, bool) and (
                                not value or flag in ("checked", "selected")
                            ):
                                node[flag] = value
                        model = getattr(widget, "model", None)
                        if model is not None:
                            for getter in ("get_value_as_string",
                                           "get_value_as_float",
                                           "get_value_as_bool",
                                           "get_value_as_int"):
                                read = getattr(model, getter, None)
                                if read is None:
                                    continue
                                try:
                                    node["value"] = read()
                                except Exception:
                                    continue
                                break
                        try:
                            raw_children = ui.Inspector.get_children(widget)
                        except Exception:
                            raw_children = []
                        if raw_children:
                            if depth >= _MAX_DEPTH:
                                node["children_omitted"] = len(raw_children)
                            else:
                                children = []
                                for child in raw_children:
                                    described = describe(child, depth + 1)
                                    if described is not None:
                                        children.append(described)
                                if children:
                                    node["children"] = children
                        return node

                    frame = getattr(window, "frame", None)
                    tree = describe(frame, 0) if frame is not None else None
                    print(json.dumps({
                        "window": _TARGET,
                        "visible": bool(window.visible),
                        "docked": bool(getattr(window, "docked", False)),
                        "width": getattr(window, "width", None),
                        "height": getattr(window, "height", None),
                        "frame_available": frame is not None,
                        "widget_count": _MAX_WIDGETS - budget["left"],
                        "truncated": budget["truncated"],
                        "widget_tree": tree,
                    }))
        """)
        mode = self._raw_script_transport_mode
        return await self._execute_json_script(script, transport_mode=mode)
