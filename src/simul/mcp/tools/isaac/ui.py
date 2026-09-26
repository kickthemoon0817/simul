"""GUI/state observability tools for Isaac Sim.

Sections degrade independently: a run without ``omni.ui`` (or with a
destroyed window in the workspace) reports what it can and marks what it
cannot, instead of failing the whole call. Note that Kit ``--no-window``
usually still loads ``omni.ui`` — the common headless shape is
``ui_available: true`` with an empty window list, not an import failure.
"""

import textwrap
from typing import Any, Dict

from ._shared import _pyval
from .._meta import tool_meta

# Caps for the widget-tree walk, mirroring the max(1, min(x, N)) clamp the
# other bounded tools use. The depth cap doubles as the recursion bound.
MAX_UI_TREE_DEPTH = 32
MAX_UI_TREE_WIDGETS = 2000


class UiObservabilityMixin:
    # ------------------------------------------------------------------
    # GUI / application state observability
    # ------------------------------------------------------------------

    @tool_meta(
        name="get_isaac_ui_state",
        description=(
            "Get a consolidated snapshot of the Isaac Sim GUI and app state: omni.ui "
            "windows (visibility, focus, docking), active viewport, selection, timeline, "
            "and open-stage status. Degrades gracefully on headless runs."
        ),
        read_only=True,
        idempotent=True,
    )
    async def get_isaac_ui_state(self) -> Dict[str, Any]:
        """
        Get a consolidated snapshot of the Isaac Sim GUI and app state.

        Collects app-window config, the omni.ui workspace (windows,
        visibility, focus, docking), active viewport, current selection,
        timeline state, and open-stage status in one call.

        Returns:
            Dict with app_window, windows, viewport, selection_paths,
            timeline, and stage sections; failed sections carry an error
            note instead. ``windows`` and ``selection_paths`` sit at the
            top level so the result budget can trim them.
        """
        script = textwrap.dedent("""\
            import json
            import math

            state = {}

            def _finite(value):
                if isinstance(value, float) and not math.isfinite(value):
                    return str(value)
                return value

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
                    # A destroyed window can still hand out a live handle
                    # whose properties raise; report it, keep the rest.
                    try:
                        entry = {
                            "title": str(win.title),
                            "visible": bool(win.visible),
                            "focused": bool(getattr(win, "focused", False)),
                            "docked": bool(getattr(win, "docked", False)),
                            "width": _finite(getattr(win, "width", None)),
                            "height": _finite(getattr(win, "height", None)),
                        }
                    except Exception as we:
                        windows.append({"error": str(we)})
                        continue
                    if entry["focused"]:
                        focused = entry["title"]
                    windows.append(entry)
                windows.sort(key=lambda w: w.get("title") or "")
                state["ui_available"] = True
                state["window_count"] = len(windows)
                state["focused_window"] = focused
                state["windows"] = windows
            except Exception as e:
                state["ui_available"] = False
                state["ui_error"] = str(e)
                state["windows"] = []

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
                state["selection_count"] = len(paths)
                state["selection_paths"] = paths
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

    @tool_meta(
        name="get_isaac_ui_window",
        description=(
            "Inspect one omni.ui window as a widget tree: widget types, identifiers, label "
            "text, state flags, and model values, depth- and count-limited. Window titles "
            "come from get_isaac_ui_state."
        ),
        read_only=True,
        idempotent=True,
    )
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
            max_depth: Maximum tree depth to descend (clamped to 1..32; the
                clamp also bounds recursion).
            max_widgets: Maximum number of widgets to describe before
                truncating (clamped to 1..2000).

        Returns:
            Dict with window metadata and the (possibly truncated) widget
            tree, or an error listing the available window titles.
        """
        max_depth = max(1, min(max_depth, MAX_UI_TREE_DEPTH))
        max_widgets = max(1, min(max_widgets, MAX_UI_TREE_WIDGETS))
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
            import math

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

                    def _finite(value):
                        if isinstance(value, float) and not math.isfinite(value):
                            return str(value)
                        return value

                    def describe(widget, depth):
                        if budget["left"] <= 0:
                            budget["truncated"] = True
                            return None
                        budget["left"] -= 1
                        # A single widget with a raising property must cost
                        # one node, not the whole tree.
                        try:
                            return _describe(widget, depth)
                        except Exception as e:
                            return {"type": "<unreadable>", "error": str(e)}

                    def _describe(widget, depth):
                        node = {"type": type(widget).__name__}
                        for key in ("identifier", "name"):
                            try:
                                value = getattr(widget, key, "")
                            except Exception:
                                continue
                            if value:
                                node[key] = value
                        try:
                            text = getattr(widget, "text", None)
                        except Exception:
                            text = None
                        if isinstance(text, str) and text:
                            node["text"] = text
                        for flag in ("visible", "enabled", "checked",
                                     "selected"):
                            try:
                                value = getattr(widget, flag)
                            except Exception:
                                continue
                            # Report only the informative state: default-true
                            # flags when off, default-false flags when on.
                            if isinstance(value, bool) and (
                                (flag in ("visible", "enabled") and not value)
                                or (flag in ("checked", "selected") and value)
                            ):
                                node[flag] = value
                        try:
                            model = getattr(widget, "model", None)
                        except Exception:
                            model = None
                        if model is not None:
                            for getter in ("get_value_as_string",
                                           "get_value_as_float",
                                           "get_value_as_bool",
                                           "get_value_as_int"):
                                read = getattr(model, getter, None)
                                if read is None:
                                    continue
                                try:
                                    node["value"] = _finite(read())
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
                        "width": _finite(getattr(window, "width", None)),
                        "height": _finite(getattr(window, "height", None)),
                        "frame_available": frame is not None,
                        "widget_count": _MAX_WIDGETS - budget["left"],
                        "truncated": budget["truncated"],
                        "widget_tree": tree,
                    }))
        """)
        mode = self._raw_script_transport_mode
        return await self._execute_json_script(script, transport_mode=mode)
