"""Pinned tool annotations, so hint drift is a test failure rather than a surprise.

A harness that auto-approves read-only tools trusts these hints. The rule is
the same on every backend: ``readOnlyHint`` is true only when the tool changes
nothing; ``destructiveHint`` is true when the tool overwrites or deletes
existing state or files, or runs arbitrary code. Additive tools (create, add,
spawn, import) and runtime control (play, pause, step, focus, enable) are not
destructive. Hints matching the client's assumed default are omitted.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pytest

src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_path))

from simul_mcp.config import Settings  # noqa: E402
from simul_mcp.mcp import server as server_module  # noqa: E402
from tests.mcp.test_discoverability import FakeFastMCP, _AvailableAdapter  # noqa: E402

RO, RW, DESTRUCTIVE = "read_only", "additive", "destructive"

# tool name -> (class, idempotent, open_world)
EXPECTED: Dict[str, tuple[str, bool, bool]] = {
    # Isaac Sim
    "ping_isaac": (RO, True, True),
    "get_isaac_stage_info": (RO, True, True),
    "read_isaac_aovs": (RW, False, True),  # creates render products
    "capture_isaac_viewport": (DESTRUCTIVE, True, True),  # reclaims old captures
    "create_isaac_prim": (RW, False, True),
    "duplicate_isaac_prim": (RW, False, True),
    "import_isaac_asset": (RW, False, True),
    "delete_isaac_prim": (DESTRUCTIVE, True, True),
    "set_isaac_prim_attribute": (DESTRUCTIVE, True, True),
    "set_isaac_prim_transform": (DESTRUCTIVE, True, True),
    "set_isaac_prim_visibility": (DESTRUCTIVE, True, True),
    "reparent_isaac_prim": (DESTRUCTIVE, True, True),
    "assign_isaac_material": (DESTRUCTIVE, True, True),
    "set_isaac_carb_settings": (DESTRUCTIVE, True, True),
    "set_isaac_log_level": (DESTRUCTIVE, True, True),
    "disable_isaac_extension": (DESTRUCTIVE, True, True),
    "enable_isaac_extension": (RW, True, True),
    "save_isaac_stage": (DESTRUCTIVE, True, True),
    "open_isaac_stage": (DESTRUCTIVE, True, True),
    "new_isaac_stage": (DESTRUCTIVE, True, True),
    "execute_isaac_script": (DESTRUCTIVE, False, True),
    "start_isaac_simulation": (RW, True, True),
    "step_isaac_simulation": (RW, False, True),
    "set_isaac_camera": (RW, True, True),
    "focus_isaac_viewport": (RW, True, True),
    "set_active_isaac_instance": (RW, True, False),
    # Headless USD
    "load_usd_file": (RO, True, True),
    "get_prim_info": (RO, True, False),
    "create_prim": (RW, False, False),
    "update_prim_attributes": (DESTRUCTIVE, False, False),
    "delete_prim": (DESTRUCTIVE, False, False),
    # Unreal
    "unreal_health_check": (RO, True, True),
    "capture_unreal_viewport": (RW, False, True),  # writes a screenshot file
    "get_unreal_actor_thumbnail": (RO, False, True),
    "spawn_unreal_actor": (RW, False, True),
    "delete_unreal_actor": (DESTRUCTIVE, False, True),
    "set_unreal_actor_transform": (DESTRUCTIVE, True, True),
    "set_unreal_actor_property": (DESTRUCTIVE, True, True),
    "assign_unreal_material": (DESTRUCTIVE, True, True),
    "export_unreal_usd": (DESTRUCTIVE, False, True),
    "import_unreal_usd": (RW, False, True),
    "execute_unreal_script": (DESTRUCTIVE, False, True),
    "control_unreal_simulation": (RW, False, True),
    "set_unreal_camera_view": (RW, True, True),
    "simplify_unreal_mesh": (DESTRUCTIVE, False, True),
    "compute_unreal_convex_hull": (RW, False, True),
    # Blender
    "get_blender_info": (RO, True, True),
    "capture_blender_viewport": (RO, True, True),  # base64 only, no file
    "create_blender_object": (RW, False, True),
    "delete_blender_object": (DESTRUCTIVE, False, True),
    "set_blender_object_transform": (DESTRUCTIVE, True, True),
    "assign_blender_material": (DESTRUCTIVE, True, True),
    "save_blender_file": (DESTRUCTIVE, True, True),
    "export_blender_file": (DESTRUCTIVE, False, True),
    "open_blender_file": (DESTRUCTIVE, False, True),
    "execute_blender_script": (DESTRUCTIVE, False, True),
    "insert_blender_keyframe": (RW, False, True),
    "delete_blender_keyframe": (DESTRUCTIVE, False, True),
    "free_blender_bake": (DESTRUCTIVE, True, True),
    "set_blender_frame": (RW, True, True),
    # Meta
    "get_tool_usage_stats": (RO, True, False),
}


def _expected_hints(kind: str, idempotent: bool, open_world: bool) -> Dict[str, Any]:
    hints: Dict[str, Any] = {"readOnlyHint": kind == RO}
    if kind == DESTRUCTIVE:
        hints["destructiveHint"] = True
    if idempotent:
        hints["idempotentHint"] = True
    if not open_world:
        hints["openWorldHint"] = False
    return hints


@pytest.fixture(scope="module")
def registered() -> Dict[str, Dict[str, Any]]:
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
        monkeypatch.setattr(server_module, "TaskConfig", None)
        monkeypatch.setattr(server_module, "is_headless_available", lambda: True)
        monkeypatch.setattr(server_module, "is_blender_available", lambda: True)
        monkeypatch.setattr(server_module, "BlenderRuntimeAdapter", _AvailableAdapter)
        monkeypatch.setattr(server_module, "UnrealRuntimeAdapter", _AvailableAdapter)
        base = Settings()
        settings = base.model_copy(
            update={"unreal": base.unreal.model_copy(update={"tool_surface": "full"})}
        )
        instance = server_module.SimulMCPServer(settings=settings)
    return {
        tool.name: tool.kwargs["annotations"].model_dump(exclude_none=True)
        for tool in instance.mcp.tools
    }


@pytest.mark.parametrize(("tool_name", "expected"), sorted(EXPECTED.items()))
def test_hints_match_the_table(
    registered: Dict[str, Dict[str, Any]], tool_name: str, expected: tuple[str, bool, bool]
) -> None:
    assert registered[tool_name] == _expected_hints(*expected)


def test_no_read_only_tool_is_destructive(registered: Dict[str, Dict[str, Any]]) -> None:
    contradictions = [
        name for name, hints in registered.items() if hints["readOnlyHint"] and hints.get("destructiveHint")
    ]
    assert contradictions == []


def test_reset_tool_usage_stats_is_not_on_the_agent_surface(
    registered: Dict[str, Dict[str, Any]]
) -> None:
    assert "reset_tool_usage_stats" not in registered
    assert "get_tool_usage_stats" in registered
