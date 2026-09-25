"""Bounded named UI actions and visible agent pointers in an attached window."""

from __future__ import annotations

import math
from typing import Any

import bpy

from .agent_cursor import cursors, observations

MENUS = {"add": "VIEW3D_MT_add", "object": "VIEW3D_MT_object", "view": "VIEW3D_MT_view"}
TOOLS = {name: f"builtin.{name}" for name in ("select_box", "move", "rotate", "scale")}
ACTIONS = (
    "inspect",
    "observe",
    "move_cursor",
    "clear_cursor",
    "open_menu",
    "select_object",
    "set_tool",
    "show_properties",
    "set_property",
    "isolate_workspace",
)
PROPERTIES = ("location", "rotation_euler", "scale")
_pending_isolations: dict[str, tuple[str, str]] = {}


def reset_ui() -> None:
    """Forget transient visual and workspace-request state on load or shutdown."""
    cursors.stop()
    observations.stop()
    _pending_isolations.clear()


def _workspace_info(window: Any) -> dict[str, Any]:
    return {
        "workspace_id": str(window.workspace.as_pointer()),
        "workspace_name": window.workspace.name,
        "shared_window_ids": [
            str(w.as_pointer())
            for w in bpy.context.window_manager.windows
            if w != window and w.workspace == window.workspace
        ],
        "tool_scope": "workspace_and_mode",
    }


def _isolate_workspace(window: Any) -> dict[str, Any]:
    current = {
        str(w.as_pointer()): str(w.workspace.as_pointer())
        for w in bpy.context.window_manager.windows
    }
    for key, (old, _) in list(_pending_isolations.items()):
        if current.get(key) != old:
            del _pending_isolations[key]
    info = _workspace_info(window)
    if not info["shared_window_ids"]:
        return {**info, "completion": "already_isolated", "inspect_required": False}
    if window.parent is not None or any(
        w.parent == window for w in bpy.context.window_manager.windows
    ):
        raise ValueError(
            "Linked child windows follow their parent's workspace. "
            "Use a separate main Blender window before isolation."
        )
    key = str(window.as_pointer())
    pending = _pending_isolations.get(key)
    if (
        pending
        and pending[0] == info["workspace_id"]
        and any(str(w.as_pointer()) == pending[1] for w in bpy.data.workspaces)
    ):
        return {
            **info,
            "completion": "workspace_copy_requested",
            "requested_workspace_id": pending[1],
            "inspect_required": True,
        }
    before = {w.as_pointer() for w in bpy.data.workspaces}
    status = bpy.ops.workspace.duplicate()
    created = [w for w in bpy.data.workspaces if w.as_pointer() not in before]
    if "FINISHED" not in status or len(created) != 1:
        raise RuntimeError(
            "Blender did not create the separate workspace; inspect before retrying"
        )
    _pending_isolations[key] = (info["workspace_id"], str(created[0].as_pointer()))
    # Blender applies the workspace switch after this timer returns. Old area
    # IDs are deliberately not reused for subsequent actions.
    return {
        **info,
        "completion": "workspace_copy_requested",
        "requested_workspace_id": str(created[0].as_pointer()),
        "inspect_required": True,
    }


def _object_action(
    action: str, target: str | None, value: list[float] | None
) -> dict[str, Any]:
    if bpy.context.mode != "OBJECT":
        raise ValueError(f"{action} requires Object Mode")
    if action == "select_object":
        obj = bpy.context.view_layer.objects.get(target or "")
        if obj is None or obj.hide_select or not obj.visible_get():
            raise ValueError(
                "Target object is absent, hidden, or not selectable in this window's view layer"
            )
        for selected in bpy.context.selected_objects:
            selected.select_set(False)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        return {"active_object": obj.name}
    if target not in PROPERTIES:
        raise ValueError(f"Supported active-object properties: {PROPERTIES}")
    if value is None or len(value) != 3 or not all(math.isfinite(n) for n in value):
        raise ValueError(
            "value must contain three finite numbers; rotation_euler uses radians"
        )
    obj = bpy.context.active_object
    if obj is None or not obj.is_editable:
        raise ValueError(
            "set_property requires an editable active object in Object Mode"
        )
    setattr(obj, target, value)
    return {"object_name": obj.name, "value": list(getattr(obj, target))}


def control_ui(
    agent_control: str,
    target: str | None = None,
    area_id: str | None = None,
    position: list[float] | None = None,
    value: list[float] | None = None,
    agent_id: str = "agent",
) -> dict[str, Any]:
    """Resolve live targets and annotate actions without moving the user's cursor."""
    window = bpy.context.window
    if bpy.app.background or window is None:
        raise RuntimeError("Agent control requires an attached GUI Blender window")
    if (
        not isinstance(agent_id, str)
        or not 1 <= len(agent_id) <= 64
        or not agent_id.isprintable()
        or not agent_id.strip()
    ):
        raise ValueError("agent_id must be a printable label of 1 to 64 characters")
    if agent_control not in ACTIONS:
        raise ValueError(f"Unsupported agent_control; choose from {ACTIONS}")
    if position is not None and agent_control != "move_cursor":
        raise ValueError("position is only supported by move_cursor")
    if value is not None and agent_control != "set_property":
        raise ValueError("value is only supported by set_property")
    result: dict[str, Any] = {
        "success": True,
        "agent_control": agent_control,
        "execution_method": "blender_ui_api",
        "window_id": str(window.as_pointer()),
        "agent_id": agent_id,
        "target": target,
    }
    if agent_control in {"inspect", "clear_cursor", "isolate_workspace"}:
        if target is not None or area_id is not None:
            raise ValueError(f"{agent_control} does not accept target or area_id")
        if agent_control == "clear_cursor":
            return {**result, "removed": cursors.clear(window, agent_id)}
        if agent_control == "isolate_workspace":
            return {**result, **_isolate_workspace(window)}
        tool = window.workspace.tools.from_space_view3d_mode(
            bpy.context.mode, create=False
        )
        return {
            **result,
            **_workspace_info(window),
            "mode": bpy.context.mode,
            "active_object": (
                bpy.context.active_object.name if bpy.context.active_object else None
            ),
            "active_tool": tool.idname if tool else None,
            "actions": list(ACTIONS),
            "menus": list(MENUS),
            "tools": list(TOOLS),
            "properties": list(PROPERTIES),
            "agent_cursors": cursors.inspect(window),
            "agent_observations": observations.inspect(window),
            "cursor_kind": "agent_overlay",
            "areas": [
                {
                    "area_id": str(a.as_pointer()),
                    "type": a.type,
                    "width": a.width,
                    "height": a.height,
                    "properties_context": (
                        a.spaces.active.context if a.type == "PROPERTIES" else None
                    ),
                }
                for a in window.screen.areas
            ],
        }
    # Object data does not depend on which of several 3D Views was chosen.
    if agent_control in {"select_object", "set_property"}:
        areas = [a for a in window.screen.areas if a.type == "VIEW_3D"]
        if area_id is not None:
            areas = [a for a in areas if str(a.as_pointer()) == area_id]
            if not areas:
                raise ValueError("Stale VIEW_3D area_id; inspect current editors")
        result.update(_object_action(agent_control, target, value))
        if areas:
            area = max(areas, key=lambda a: a.width * a.height)
            label = (
                f"{agent_control}: {target}"
                + (f" = {value}" if value is not None else "")
                + " (done)"
            )
            result["agent_cursor"] = cursors.update(
                window, area, agent_id, [0.5, 0.5], label
            )
        return result
    area_type = "PROPERTIES" if agent_control == "show_properties" else "VIEW_3D"
    areas = [a for a in window.screen.areas if a.type == area_type]
    if area_id is not None:
        areas = [a for a in areas if str(a.as_pointer()) == area_id]
    if len(areas) != 1:
        raise ValueError(
            f"No unique {area_type} editor; use inspect and supply a current area_id"
        )
    area = areas[0]
    region = next((r for r in area.regions if r.type == "WINDOW"), None)
    if region is None or region.width < 2 or region.height < 2:
        raise ValueError("The selected editor has no usable window region")
    result["area_id"] = str(area.as_pointer())
    if agent_control == "observe":
        if target is not None:
            raise ValueError("observe does not accept target")
        return {
            **result,
            "agent_observation": observations.show(window, area, agent_id),
        }
    point = [0.5, 0.5]
    if agent_control != "move_cursor":
        existing = next(
            (
                m
                for m in cursors.inspect(window)
                if m["agent_id"] == agent_id and m["area_id"] == str(area.as_pointer())
            ),
            None,
        )
        if existing is not None:
            point = existing["position"]
    with bpy.context.temp_override(window=window, area=area, region=region):
        if agent_control == "move_cursor":
            if target not in (None, "viewport"):
                raise ValueError(
                    "move_cursor target must be viewport; arbitrary button lookup is unsupported"
                )
            point = position if position is not None else point
            if len(point) != 2 or not all(
                math.isfinite(n) and 0 <= n <= 1 for n in point
            ):
                raise ValueError(
                    "position must contain two finite coordinates in [0, 1], from bottom-left"
                )
            result.update(
                position=point, cursor_kind="agent_overlay", system_cursor_moved=False
            )
        elif agent_control == "open_menu":
            if target not in MENUS:
                raise ValueError(f"Supported menus: {list(MENUS)}")
            if bpy.context.mode != "OBJECT":
                raise ValueError("Supported menus require Object Mode")
            status = bpy.ops.wm.call_menu("INVOKE_DEFAULT", name=MENUS[target])
            if "CANCELLED" in status:
                raise RuntimeError("Blender cancelled the menu request")
            result.update(operator_status=sorted(status), completion="menu_requested")
        elif agent_control == "set_tool":
            if target not in TOOLS:
                raise ValueError(f"Supported tools: {list(TOOLS)}")
            scope = _workspace_info(window)
            if scope["shared_window_ids"]:
                raise ValueError(
                    "Tool changes affect a shared workspace. Run isolate_workspace, "
                    "then inspect for new area IDs before set_tool."
                )
            status = bpy.ops.wm.tool_set_by_id(name=TOOLS[target])
            if "FINISHED" not in status:
                raise RuntimeError(
                    f"Blender did not activate the tool: {sorted(status)}"
                )
            tool = window.workspace.tools.from_space_view3d_mode(
                bpy.context.mode, create=False
            )
            if tool is None or tool.idname != TOOLS[target]:
                raise RuntimeError("Blender did not report the requested active tool")
            result.update(tool_id=tool.idname, **scope)
        elif agent_control == "show_properties":
            if not target:
                raise ValueError(
                    "target must be a Properties tab identifier, for example OBJECT or RENDER"
                )
            area.spaces.active.context = target
            result["context"] = area.spaces.active.context
        # This is a visual annotation of the last action, not a synthetic click
        # location. Menu/tool operations still use their semantic Blender APIs.
        label = (
            "Agent pointer"
            if agent_control == "move_cursor"
            else f"{agent_control}: {target} ({'requested' if agent_control == 'open_menu' else 'done'})"
        )
        result["agent_cursor"] = cursors.update(window, area, agent_id, point, label)
        area.tag_redraw()
    return result
