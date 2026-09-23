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
        screen=SimpleNamespace(areas=[area]), as_pointer=lambda: 4, cursor_warp=Mock()
    )
    context = SimpleNamespace(
        window=window,
        mode="OBJECT",
        active_object=None,
        temp_override=lambda **kw: nullcontext(),
    )
    bpy = SimpleNamespace(context=context, app=SimpleNamespace(background=False))
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    path = Path(__file__).parents[2] / "src/simul_mcp/blender_bridge/agent_control.py"
    spec = importlib.util.spec_from_file_location("agent_control_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, bpy, area


def test_cursor_uses_live_region_geometry_and_refuses_ambiguity(ui: tuple) -> None:
    module, bpy, area = ui
    module.control_ui("move_cursor")
    bpy.context.window.cursor_warp.assert_called_once_with(500, 350)
    # Resize/move the editor: no cached absolute display coordinates.
    area.regions[0].x = 200
    module.control_ui("move_cursor", position=[1, 0])
    bpy.context.window.cursor_warp.assert_called_with(1000, 50)
    bpy.context.window.screen.areas.append(
        SimpleNamespace(type="VIEW_3D", as_pointer=lambda: 8)
    )
    with pytest.raises(ValueError, match="unique"):
        module.control_ui("move_cursor")
    module.control_ui("move_cursor", area_id="7")
    with pytest.raises(ValueError, match="unique"):
        module.control_ui("move_cursor", area_id="closed")


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
