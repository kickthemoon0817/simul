"""UI target resolution must never guess another editor or accept arbitrary code."""

from __future__ import annotations

import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from simul_mcp.mcp.schemas.blender_ui import BlenderUIRequest


@pytest.fixture
def ui(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any, Any]:
    region = SimpleNamespace(type="WINDOW", x=100, y=50, width=801, height=601)
    area = SimpleNamespace(
        type="VIEW_3D",
        as_pointer=lambda: 7,
        width=801,
        height=601,
        regions=[region],
        tag_redraw=Mock(),
    )
    window = SimpleNamespace(
        screen=SimpleNamespace(areas=[area]),
        as_pointer=lambda: 4,
        cursor_warp=Mock(),
        workspace=SimpleNamespace(as_pointer=lambda: 10, name="Layout"),
        parent=None,
    )
    context = SimpleNamespace(
        window=window,
        mode="OBJECT",
        active_object=None,
        temp_override=lambda **kw: nullcontext(),
        window_manager=SimpleNamespace(windows=[window]),
    )
    bpy = SimpleNamespace(context=context, app=SimpleNamespace(background=False))
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    cursors = Mock()
    cursors.inspect.return_value = []
    cursors.update.side_effect = lambda window, area, agent_id, position, label: {
        "agent_id": agent_id,
        "area_id": str(area.as_pointer()),
        "position": position,
        "label": label,
    }
    monkeypatch.setitem(
        sys.modules,
        "simul_mcp.blender_bridge.agent_cursor",
        SimpleNamespace(cursors=cursors, observations=Mock()),
    )
    path = Path(__file__).parents[2] / "src/simul_mcp/blender_bridge/agent_control.py"
    spec = importlib.util.spec_from_file_location(
        "simul_mcp.blender_bridge.agent_control_under_test", path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, bpy, area


def test_agent_cursor_never_moves_os_pointer_and_refuses_ambiguity(ui: tuple) -> None:
    module, bpy, area = ui
    result = module.control_ui("move_cursor", agent_id="planner")
    assert result["agent_cursor"]["position"] == [0.5, 0.5]
    assert result["agent_cursor"]["agent_id"] == "planner"
    assert result["system_cursor_moved"] is False
    # Resize/move the editor: no cached absolute display coordinates.
    area.regions[0].x = 200
    result = module.control_ui("move_cursor", position=[1, 0])
    assert result["agent_cursor"]["position"] == [1, 0]
    bpy.context.window.screen.areas.append(
        SimpleNamespace(type="VIEW_3D", as_pointer=lambda: 8)
    )
    with pytest.raises(ValueError, match="unique"):
        module.control_ui("move_cursor")
    module.control_ui("move_cursor", area_id="7")
    with pytest.raises(ValueError, match="unique"):
        module.control_ui("move_cursor", area_id="closed")
    bpy.context.window.cursor_warp.assert_not_called()


def test_shared_workspace_refuses_tool_change_before_operator(ui: tuple) -> None:
    module, bpy, _ = ui
    window = bpy.context.window
    other = SimpleNamespace(as_pointer=lambda: 5, workspace=window.workspace)
    bpy.context.window_manager.windows.append(other)
    bpy.ops = SimpleNamespace(wm=SimpleNamespace(tool_set_by_id=Mock()))
    with pytest.raises(ValueError, match="isolate_workspace"):
        module.control_ui("set_tool", target="move")
    bpy.ops.wm.tool_set_by_id.assert_not_called()


def test_menu_preserves_agent_pointer_position(ui: tuple) -> None:
    module, bpy, _ = ui
    module.cursors.inspect.return_value = [
        {"agent_id": "builder", "area_id": "7", "position": [0.3, 0.7]}
    ]
    bpy.ops = SimpleNamespace(
        wm=SimpleNamespace(call_menu=Mock(return_value={"INTERFACE"}))
    )
    result = module.control_ui("open_menu", target="add", agent_id="builder")
    assert result["agent_cursor"]["position"] == [0.3, 0.7]
    bpy.context.window.cursor_warp.assert_not_called()


def test_isolation_refuses_linked_child_windows(ui: tuple) -> None:
    module, bpy, _ = ui
    window = bpy.context.window
    other = SimpleNamespace(
        as_pointer=lambda: 5, workspace=window.workspace, parent=window
    )
    bpy.context.window_manager.windows.append(other)
    with pytest.raises(ValueError, match="Linked child"):
        module.control_ui("isolate_workspace")


def test_property_write_does_not_require_unique_editor(ui: tuple) -> None:
    module, bpy, area = ui
    bpy.context.active_object = SimpleNamespace(
        name="Cube", is_editable=True, location=[0, 0, 0]
    )
    bpy.context.window.screen.areas.append(area)
    result = module.control_ui("set_property", target="location", value=[1, 2, 3])
    assert result["value"] == [1, 2, 3]
    with pytest.raises(ValueError, match="Stale"):
        module.control_ui(
            "set_property", target="location", value=[4, 5, 6], area_id="old"
        )
    assert bpy.context.active_object.location == [1, 2, 3]


@pytest.mark.parametrize(
    "position", [[-0.1, 0], [0, 1.1], [float("nan"), 0], [float("inf"), 0], [0]]
)
def test_invalid_positions_never_move_cursor(ui: tuple, position: list) -> None:
    module, bpy, _ = ui
    with pytest.raises(ValueError, match="position"):
        module.control_ui("move_cursor", position=position)
    bpy.context.window.cursor_warp.assert_not_called()


@pytest.mark.parametrize(
    "action,target",
    [
        ("execute_script", "print('no')"),
        ("open_menu", "CONSOLE_MT_console"),
        ("set_tool", "python.exec"),
        ("set_property", "__class__"),
        ("move_cursor", "Save button"),
    ],
)
def test_unsupported_actions_and_targets_fail(
    ui: tuple, action: str, target: str
) -> None:
    module, bpy, _ = ui
    with pytest.raises(ValueError):
        module.control_ui(action, target=target)
    bpy.context.window.cursor_warp.assert_not_called()


def test_headless_refused_and_schema_rejects_unknown_action(ui: tuple) -> None:
    module, bpy, _ = ui
    bpy.app.background = True
    with pytest.raises(RuntimeError, match="GUI"):
        module.control_ui("inspect")
    with pytest.raises(ValueError):
        BlenderUIRequest(agent_control="execute_script")


def test_observe_uses_selected_editor_without_replacing_pointer(ui: tuple) -> None:
    module, bpy, area = ui
    module.control_ui("observe", agent_id="reviewer", area_id="7")
    module.observations.show.assert_called_once_with(
        bpy.context.window, area, "reviewer"
    )
    module.cursors.update.assert_not_called()
    bpy.context.window.cursor_warp.assert_not_called()
    with pytest.raises(ValueError, match="does not accept target"):
        module.control_ui("observe", target="Cube")
    with pytest.raises(ValueError, match="unique"):
        module.control_ui("observe", area_id="stale")
