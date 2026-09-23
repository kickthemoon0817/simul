"""Bounded, named UI actions in the attached window; no generated Python."""

from __future__ import annotations

import math
from typing import Any

import bpy

MENUS = {"add": "VIEW3D_MT_add", "object": "VIEW3D_MT_object", "view": "VIEW3D_MT_view"}
TOOLS = {name: f"builtin.{name}" for name in ("select_box", "move", "rotate", "scale")}
ACTIONS = (
    "inspect",
    "move_cursor",
    "open_menu",
    "select_object",
    "set_tool",
    "show_properties",
    "set_property",
)
PROPERTIES = ("location", "rotation_euler", "scale")


def control_ui(
    agent_control: str,
    target: str | None = None,
    area_id: str | None = None,
    position: list[float] | None = None,
    value: list[float] | None = None,
) -> dict[str, Any]:
    """Resolve targets from the current window, failing on stale or ambiguous IDs."""
    window = bpy.context.window
    if bpy.app.background or window is None:
        raise RuntimeError("Agent control requires an attached GUI Blender window")
    if agent_control not in ACTIONS:
        raise ValueError(f"Unsupported agent_control; choose from {ACTIONS}")
    if position is not None and agent_control != "move_cursor":
        raise ValueError("position is only supported by move_cursor")
    if value is not None and agent_control != "set_property":
        raise ValueError("value is only supported by set_property")
    if agent_control == "inspect":
        if target is not None or area_id is not None:
            raise ValueError("inspect does not accept target or area_id")
        return {
            "success": True,
            "agent_control": agent_control,
            "execution_method": "blender_ui_api",
            "window_id": str(window.as_pointer()),
            "mode": bpy.context.mode,
            "active_object": (
                bpy.context.active_object.name if bpy.context.active_object else None
            ),
            "actions": list(ACTIONS),
            "menus": list(MENUS),
            "tools": list(TOOLS),
            "properties": list(PROPERTIES),
            "areas": [
                {
                    "area_id": str(a.as_pointer()),
                    "type": a.type,
                    "width": a.width,
                    "height": a.height,
                }
                for a in window.screen.areas
            ],
        }
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
    result: dict[str, Any] = {
        "success": True,
        "agent_control": agent_control,
        "execution_method": "blender_ui_api",
        "window_id": str(window.as_pointer()),
        "area_id": str(area.as_pointer()),
        "target": target,
    }
    with bpy.context.temp_override(window=window, area=area, region=region):
        if agent_control == "move_cursor":
            if target not in (None, "viewport"):
                raise ValueError(
                    "move_cursor target must be viewport; arbitrary button lookup is unsupported"
                )
            point = position if position is not None else [0.5, 0.5]
            if len(point) != 2 or not all(
                math.isfinite(n) and 0 <= n <= 1 for n in point
            ):
                raise ValueError(
                    "position must contain two finite coordinates in [0, 1], from bottom-left"
                )
            x = region.x + round(point[0] * (region.width - 1))
            y = region.y + round(point[1] * (region.height - 1))
            window.cursor_warp(x, y)
            result.update(cursor_window=[x, y], position=point)
        elif agent_control == "open_menu":
            if target not in MENUS:
                raise ValueError(f"Supported menus: {list(MENUS)}")
            if bpy.context.mode != "OBJECT":
                raise ValueError("Supported menus require Object Mode")
            # Invoke the named menu directly; do not guess a header button position.
            status = bpy.ops.wm.call_menu("INVOKE_DEFAULT", name=MENUS[target])
            if "CANCELLED" in status:
                raise RuntimeError("Blender cancelled the menu request")
            result.update(operator_status=sorted(status), completion="menu_requested")
        elif agent_control == "set_tool":
            if target not in TOOLS:
                raise ValueError(f"Supported tools: {list(TOOLS)}")
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
            result["tool_id"] = tool.idname
        elif agent_control == "show_properties":
            if not target:
                raise ValueError(
                    "target must be a Properties tab identifier, for example OBJECT or RENDER"
                )
            area.spaces.active.context = target
            result["context"] = area.spaces.active.context
        elif agent_control == "select_object":
            if bpy.context.mode != "OBJECT":
                raise ValueError("select_object requires Object Mode")
            obj = bpy.context.view_layer.objects.get(target or "")
            if obj is None or obj.hide_select or not obj.visible_get():
                raise ValueError(
                    "Target object is absent, hidden, or not selectable in this window's view layer"
                )
            for selected in bpy.context.selected_objects:
                selected.select_set(False)
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            result["active_object"] = obj.name
        elif agent_control == "set_property":
            if target not in PROPERTIES:
                raise ValueError(f"Supported active-object properties: {PROPERTIES}")
            if (
                value is None
                or len(value) != 3
                or not all(math.isfinite(n) for n in value)
            ):
                raise ValueError(
                    "value must contain three finite numbers; rotation_euler uses radians"
                )
            obj = bpy.context.active_object
            if obj is None or bpy.context.mode != "OBJECT" or not obj.is_editable:
                raise ValueError(
                    "set_property requires an editable active object in Object Mode"
                )
            setattr(obj, target, value)
            result.update(object_name=obj.name, value=list(getattr(obj, target)))
        area.tag_redraw()
    return result
