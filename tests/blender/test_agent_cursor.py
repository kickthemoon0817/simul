"""Agent annotations are scoped by window/agent and never invoke OS input."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def overlay(monkeypatch):
    area = SimpleNamespace(as_pointer=lambda: 7, type="VIEW_3D", tag_redraw=Mock())
    workspace = SimpleNamespace(as_pointer=lambda: 8)
    scene = SimpleNamespace(as_pointer=lambda: 9)
    one = SimpleNamespace(
        as_pointer=lambda: 1,
        workspace=workspace,
        scene=scene,
        screen=SimpleNamespace(areas=[area]),
    )
    two = SimpleNamespace(
        as_pointer=lambda: 2,
        workspace=workspace,
        scene=scene,
        screen=SimpleNamespace(areas=[area]),
    )
    space = SimpleNamespace(
        draw_handler_add=Mock(return_value="handler"), draw_handler_remove=Mock()
    )
    bpy = SimpleNamespace(
        types=SimpleNamespace(SpaceView3D=space, SpaceProperties=space),
        context=SimpleNamespace(
            window_manager=SimpleNamespace(windows=[one, two]),
            window=two,
            area=area,
            region=SimpleNamespace(),
        ),
    )
    monkeypatch.setitem(sys.modules, "bpy", bpy)
    path = Path(__file__).parents[2] / "src/simul_mcp/blender_bridge/agent_cursor.py"
    spec = importlib.util.spec_from_file_location("agent_cursor_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AgentCursors(), one, two, area, space


def test_two_agents_and_two_windows_do_not_replace_each_other(overlay):
    cursors, one, two, area, _ = overlay
    cursors.update(one, area, "builder", [0.2, 0.3], "Move tool (done)")
    cursors.update(one, area, "reviewer", [0.7, 0.8], "Inspecting")
    cursors.update(two, area, "builder", [0.1, 0.9], "Other window")
    assert len(cursors.inspect(one)) == 2
    assert cursors.inspect(two)[0]["label"] == "Other window"
    cursors.clear(one, "builder")
    assert [m["agent_id"] for m in cursors.inspect(one)] == ["reviewer"]
    assert len(cursors.inspect(two)) == 1


def test_overlay_is_not_drawn_in_another_window_even_for_same_area(
    overlay, monkeypatch
):
    cursors, one, _, area, _ = overlay
    cursors.update(one, area, "builder", [0.2, 0.3], "Move tool (done)")
    # The draw context is window two. No GPU or text operations may run there.
    monkeypatch.setitem(sys.modules, "gpu", None)
    monkeypatch.setitem(sys.modules, "blf", None)
    cursors.draw()


def test_expiry_workspace_changes_and_stop_remove_annotations(overlay):
    cursors, one, _, area, space = overlay
    cursors.update(one, area, "builder", [0.2, 0.3], "Move tool (done)")
    cursors.markers[("1", "builder")]["updated_at"] -= 121
    assert cursors.inspect(one) == []
    cursors.update(one, area, "builder", [0.2, 0.3], "Move tool (done)")
    one.workspace = SimpleNamespace(as_pointer=lambda: 10)
    assert cursors.inspect(one) == []
    cursors.stop()
    assert not cursors.handlers and not cursors.markers
    assert space.draw_handler_remove.call_count == 2
