"""Pinned tool annotations for every tool, so hint drift is a test failure rather than a surprise.

A harness that auto-approves read-only tools trusts these hints, so they are
pinned for all of them: every tool the server registers with the full Unreal
and Blender surfaces and script execution on (over 200 tools) must be in the table,
and the table names no tool that is not registered. The rule is the same on
every backend:

- ``readOnlyHint`` is true only when the tool changes nothing (a capture that
  returns the image in memory is read-only; one that writes a file is not).
- ``destructiveHint`` is true when the tool overwrites or deletes existing
  state or files, or runs arbitrary code. Additive tools (create, add, spawn,
  import) and runtime control (play, pause, step, focus, enable) are not
  destructive.
- ``idempotentHint`` is true when repeating the call with the same arguments
  leaves things as the first call left them: a set, a delete, a save or an
  export to the same path, a focus. A create, a step, a capture that writes a
  new file, or an edit that compounds (simplify, subdivide) is not.

The three viewport captures differ because they do different things:
``capture_blender_viewport`` returns the image only (read-only),
``capture_unreal_viewport`` writes a new screenshot file (additive), and
``capture_isaac_viewport`` writes a new PNG and deletes the oldest captures in
its directory (destructive). None of them is idempotent except the read-only
one. Hints matching the client's assumed default are omitted.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest


from simul.config import Settings
from simul.mcp import backends as backends_module
from simul.mcp import server as server_module
from tests.fakes import AvailableAdapter, FakeFastMCP

RO, RW, DESTRUCTIVE = "read_only", "additive", "destructive"

# tool name -> (class, idempotent, open_world)
EXPECTED: Dict[str, tuple[str, bool, bool]] = {
    # Isaac Sim
    "add_isaac_collision": (RW, False, True),
    "add_isaac_reference": (RW, False, True),
    "add_isaac_rigid_body": (RW, False, True),
    "assign_isaac_material": (DESTRUCTIVE, True, True),
    "capture_isaac_viewport": (DESTRUCTIVE, False, True),  # writes a new PNG and deletes the oldest captures
    "claim_isaac_instance": (RW, True, False),
    "configure_isaac_physics_scene": (DESTRUCTIVE, True, True),
    "connect_isaac_graph_nodes": (RW, True, True),
    "create_isaac_graph_node": (RW, False, True),
    "create_isaac_light": (RW, False, True),
    "create_isaac_material": (RW, False, True),
    "create_isaac_object": (RW, False, True),
    "create_isaac_physics_scene": (RW, True, True),
    "create_isaac_prim": (RW, False, True),
    "delete_isaac_graph_node": (DESTRUCTIVE, True, True),
    "delete_isaac_prim": (DESTRUCTIVE, True, True),
    "disable_isaac_extension": (DESTRUCTIVE, True, True),
    "duplicate_isaac_prim": (RW, False, True),
    "enable_isaac_extension": (RW, True, True),
    "execute_isaac_script": (DESTRUCTIVE, False, True),
    "find_isaac_prims_in_area": (RO, True, True),
    "focus_isaac_viewport": (RW, True, True),
    "get_isaac_camera_info": (RO, True, True),
    "get_isaac_carb_settings": (RO, True, True),
    "get_isaac_graph_nodes": (RO, True, True),
    "get_isaac_layer_info": (RO, True, True),
    "get_isaac_logs": (RO, True, True),
    "get_isaac_physics_scene": (RO, True, True),
    "get_isaac_prim_detail": (RO, True, True),
    "get_isaac_runtime_info": (RO, True, True),
    "get_isaac_scene_stats": (RO, True, True),
    "get_isaac_scene_summary": (RO, True, True),
    "get_isaac_selection": (RO, True, True),
    "get_isaac_simulation_state": (RO, True, True),
    "get_isaac_simulation_time": (RO, True, True),
    "get_isaac_stage_info": (RO, True, True),
    "get_isaac_subtree": (RO, True, True),
    "get_isaac_texture_dependencies": (RO, True, True),
    "get_isaac_ui_state": (RO, True, True),
    "get_isaac_ui_window": (RO, True, True),
    "get_isaac_viewport_info": (RO, True, True),
    "import_isaac_asset": (RW, False, True),
    "interrupt_isaac_script": (RW, True, True),  # stops the running script
    "list_isaac_aovs": (RO, True, True),
    "list_isaac_articulations": (RO, True, True),
    "list_isaac_cameras": (RO, True, True),
    "list_isaac_extensions": (RO, True, True),
    "list_isaac_graph_node_types": (RO, True, True),
    "list_isaac_graphs": (RO, True, True),
    "list_isaac_instances": (RO, True, True),
    "list_isaac_lights": (RO, True, True),
    "list_isaac_materials": (RO, True, True),
    "list_isaac_physics_objects": (RO, True, True),
    "list_isaac_prims": (RO, True, True),
    "list_isaac_render_vars": (RO, True, True),
    "new_isaac_stage": (DESTRUCTIVE, True, True),
    "open_isaac_stage": (DESTRUCTIVE, True, True),  # discards unsaved work
    "pause_isaac_simulation": (RW, True, True),
    "ping_isaac": (RO, True, True),
    "query_isaac_typed_prims": (RO, True, True),
    "raycast_isaac_scene": (RO, True, True),
    "read_isaac_aovs": (RW, False, True),  # creates render products
    "release_isaac_instance": (RW, True, False),
    "reparent_isaac_prim": (DESTRUCTIVE, True, True),
    "reset_isaac_simulation": (RW, True, True),
    "save_isaac_stage": (DESTRUCTIVE, True, True),
    "search_isaac_prims": (RO, True, True),
    "set_active_isaac_instance": (RW, True, False),  # routes this session; the instance is untouched
    "set_isaac_camera": (RW, True, True),
    "set_isaac_carb_settings": (DESTRUCTIVE, True, True),
    "set_isaac_graph_node_values": (DESTRUCTIVE, True, True),
    "set_isaac_joint_drive": (DESTRUCTIVE, True, True),
    "set_isaac_joint_limits": (DESTRUCTIVE, True, True),
    "set_isaac_log_level": (DESTRUCTIVE, True, True),
    "set_isaac_mass_properties": (DESTRUCTIVE, True, True),
    "set_isaac_material_property": (DESTRUCTIVE, True, True),
    "set_isaac_physics_material": (DESTRUCTIVE, True, True),
    "set_isaac_prim_attribute": (DESTRUCTIVE, True, True),
    "set_isaac_prim_transform": (DESTRUCTIVE, True, True),
    "set_isaac_prim_visibility": (DESTRUCTIVE, True, True),
    "start_isaac_simulation": (RW, True, True),
    "step_isaac_simulation": (RW, False, True),
    "stop_isaac_simulation": (RW, True, True),
    # Headless USD
    "create_prim": (DESTRUCTIVE, False, False),  # redefine_existing re-types a prim
    "delete_prim": (DESTRUCTIVE, True, False),
    "get_bounding_box": (RO, True, False),
    "get_mesh_info": (RO, True, False),
    "get_prim_info": (RO, True, False),
    "get_prim_subtree": (RO, True, False),
    "load_usd_file": (RO, True, False),
    "search_prims": (RO, True, False),
    "summarize_scene": (RO, True, False),
    "update_prim_attributes": (DESTRUCTIVE, True, False),
    "validate_usd_file": (RO, True, False),
    "list_usd_stages": (RO, True, False),
    "save_usd_stage": (DESTRUCTIVE, True, False),  # overwrites the stage's own file
    "export_usd_stage": (DESTRUCTIVE, False, False),  # overwrite=true replaces a file
    "reload_usd_stage": (DESTRUCTIVE, True, False),  # discard_unsaved drops edits
    "unload_usd_stage": (DESTRUCTIVE, False, False),
    # Unreal (full surface)
    "add_unreal_component": (RW, False, True),
    "analyze_unreal_scene_for_robotics": (RO, True, True),
    "apply_unreal_force": (RW, False, True),
    "apply_unreal_mesh_boolean": (DESTRUCTIVE, False, True),
    "assign_unreal_material": (DESTRUCTIVE, True, True),
    "batch_unreal_operations": (DESTRUCTIVE, False, True),
    "call_unreal_actor_function": (DESTRUCTIVE, False, True),
    "capture_unreal_camera": (RW, False, True),  # writes a new image file
    "capture_unreal_viewport": (RW, False, True),  # writes a new screenshot file
    # output collision replaces the source's simple collision.
    "compute_unreal_convex_hull": (DESTRUCTIVE, False, True),
    "compute_unreal_mesh_uv": (DESTRUCTIVE, False, True),
    "control_unreal_simulation": (RW, False, True),
    "control_unreal_ui": (RW, False, True),
    "convert_to_simready": (DESTRUCTIVE, False, True),
    "convert_unreal_mesh_format": (DESTRUCTIVE, False, True),
    "create_unreal_material_instance": (RW, False, True),
    "cut_unreal_mesh_plane": (DESTRUCTIVE, False, True),
    "decompose_unreal_convex_hull": (DESTRUCTIVE, False, True),
    "delete_unreal_actor": (DESTRUCTIVE, True, True),
    "describe_unreal_object": (RO, True, True),
    "edit_unreal_mesh_topology": (DESTRUCTIVE, False, True),
    "enable_unreal_physics": (DESTRUCTIVE, True, True),
    "execute_unreal_script": (DESTRUCTIVE, False, True),
    "export_unreal_usd": (DESTRUCTIVE, True, True),
    "focus_unreal_on_actor": (RW, True, True),  # moves the view camera and selects
    "generate_unreal_mesh_primitive": (RW, False, True),
    "generate_unreal_procedural_scene": (RW, False, True),
    "get_unreal_actor_by_semantic_label": (RO, True, True),
    "get_unreal_actor_info": (RO, True, True),
    "get_unreal_actor_thumbnail": (RO, True, True),  # renders in memory
    "get_unreal_engine_info": (RO, True, True),
    "get_unreal_interchange_info": (RO, True, True),
    "get_unreal_loaded_map": (RO, True, True),
    "get_unreal_material_info": (RO, True, True),
    "get_unreal_physics_state": (RO, True, True),
    "get_unreal_simulation_status": (RO, True, True),
    "get_unreal_viewport_info": (RO, True, True),
    "import_unreal_usd": (DESTRUCTIVE, False, True),  # replace_existing replaces assets and actors
    "import_unreal_asset": (DESTRUCTIVE, False, True),
    "list_unreal_actor_components": (RO, True, True),
    "list_unreal_actors": (RO, True, True),
    "list_unreal_instances": (RO, True, True),
    "list_unreal_simulation_actors": (RO, True, True),
    # Loads actors into the editor, not the map; loading a loaded region again changes nothing.
    "load_unreal_world_partition_actors": (RW, True, True),
    "ping_unreal": (RO, True, True),
    "query_unreal_scene_graph": (RO, True, True),
    "remesh_unreal_mesh": (DESTRUCTIVE, False, True),
    "search_unreal_assets": (RO, True, True),
    "set_unreal_actor_parent": (DESTRUCTIVE, True, True),
    "save_unreal_assets": (DESTRUCTIVE, True, True),
    "set_unreal_actor_property": (DESTRUCTIVE, True, True),
    "set_unreal_actor_transform": (DESTRUCTIVE, True, True),
    "set_unreal_actor_visibility": (DESTRUCTIVE, True, True),
    "set_unreal_camera_view": (RW, True, True),
    "set_unreal_collision": (DESTRUCTIVE, True, True),
    "set_unreal_light_params": (DESTRUCTIVE, True, True),
    "set_unreal_material_params": (DESTRUCTIVE, True, True),
    "set_unreal_physics_params": (DESTRUCTIVE, True, True),
    "set_unreal_render_settings": (DESTRUCTIVE, True, True),
    "setup_unreal_physics_bulk": (DESTRUCTIVE, True, True),
    "simplify_unreal_mesh": (DESTRUCTIVE, False, True),
    "spawn_unreal_actor": (RW, False, True),
    "subdivide_unreal_mesh": (DESTRUCTIVE, False, True),
    "summarize_unreal_scene": (RO, True, True),
    "unreal_health_check": (RO, True, True),
    "validate_simready_asset": (RO, True, True),
    "validate_unreal_mesh": (RO, True, True),
    # Blender (full surface)
    "add_blender_force_field": (RW, False, True),
    "add_blender_modifier": (RW, False, True),
    "add_blender_rigid_body_constraint": (RW, False, True),
    "apply_blender_modifier": (DESTRUCTIVE, False, True),
    "apply_simready_metadata": (DESTRUCTIVE, True, True),
    "assign_blender_material": (DESTRUCTIVE, True, True),
    "attach_blender_window": (RW, True, True),  # selects the target; Blender untouched
    "bake_blender_simulation": (DESTRUCTIVE, True, True),  # replaces the bake cache of the range
    "capture_blender_viewport": (RO, True, True),  # base64 only, no file
    "capture_blender_viewport_sequence": (RO, True, True),  # base64 only, frame restored
    "check_blender_object_bounds": (RO, True, True),
    "clean_blender_mesh": (DESTRUCTIVE, True, True),
    "clear_blender_object_parent": (DESTRUCTIVE, True, True),
    "control_blender_ui": (RW, False, True),
    "create_blender_mesh_from_data": (RW, False, True),
    "create_blender_object": (RW, False, True),
    "delete_blender_keyframe": (DESTRUCTIVE, True, True),
    "delete_blender_object": (DESTRUCTIVE, False, True),
    "execute_blender_script": (DESTRUCTIVE, False, True),
    "export_blender_file": (DESTRUCTIVE, True, True),
    "export_simready_usd": (DESTRUCTIVE, True, True),
    "focus_blender_on_object": (RW, True, True),
    "free_blender_bake": (DESTRUCTIVE, True, True),
    "get_blender_bounding_box": (RO, True, True),
    "get_blender_camera_info": (RO, True, True),
    "get_blender_constraint_info": (RO, True, True),
    "get_blender_distance_between": (RO, True, True),
    "get_blender_file_info": (RO, True, True),
    "get_blender_force_field_info": (RO, True, True),
    "get_blender_frame": (RO, True, True),
    "get_blender_info": (RO, True, True),
    "get_blender_keyframes": (RO, True, True),
    "get_blender_material_info": (RO, True, True),
    "get_blender_mesh_info": (RO, True, True),
    "get_blender_object_info": (RO, True, True),
    "get_blender_object_trajectory": (RW, True, True),  # steps the scene, restores the frame
    "get_blender_physics_state": (RO, True, True),
    "get_blender_rigid_body_world": (RO, True, True),
    "get_blender_trajectories": (RW, True, True),
    "get_blender_viewport_info": (RO, True, True),
    "get_simready_metadata": (RO, True, True),
    "import_blender_file": (RW, False, True),
    "insert_blender_keyframe": (RW, False, True),
    "insert_blender_keyframes": (RW, False, True),
    "list_blender_scene_objects": (RO, True, True),
    "open_blender_file": (DESTRUCTIVE, True, True),  # discards unsaved work
    "play_blender_animation": (RW, True, True),
    "remove_blender_modifier": (DESTRUCTIVE, False, True),
    "save_blender_file": (DESTRUCTIVE, True, True),
    "search_blender_objects": (RO, True, True),
    "set_blender_camera_view": (RW, True, True),
    "set_blender_frame": (RW, True, True),
    "set_blender_frame_range": (DESTRUCTIVE, True, True),
    "set_blender_light_params": (DESTRUCTIVE, True, True),
    "set_blender_object_mode": (RW, True, True),
    "set_blender_object_parent": (DESTRUCTIVE, True, True),
    "set_blender_object_transform": (DESTRUCTIVE, True, True),
    "set_blender_pose_bone_transform": (DESTRUCTIVE, True, True),
    "set_blender_rigid_body_world": (DESTRUCTIVE, True, True),
    "setup_blender_rigid_body": (DESTRUCTIVE, True, True),
    "setup_simready_hierarchy": (RW, False, True),
    "summarize_blender_scene": (RO, True, True),
    "uv_unwrap_blender_mesh": (DESTRUCTIVE, True, True),
    "validate_simready_compliance": (RO, True, True),
    # Server
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
        monkeypatch.setattr(backends_module, "is_headless_available", lambda: True)
        monkeypatch.setattr(backends_module, "is_blender_available", lambda: True)
        monkeypatch.setattr(backends_module, "BlenderRuntimeAdapter", AvailableAdapter)
        monkeypatch.setattr(backends_module, "UnrealRuntimeAdapter", AvailableAdapter)
        base = Settings()
        settings = base.model_copy(
            update={
                "unreal": base.unreal.model_copy(update={"tool_surface": "full"}),
                "blender": base.blender.model_copy(update={"tool_surface": "full"}),
            }
        )
        instance = server_module.SimulMCPServer(settings=settings)
    return {
        tool.name: tool.kwargs["annotations"].model_dump(by_alias=True, exclude_none=True)
        for tool in instance.mcp.tools
    }


def test_every_registered_tool_is_pinned(registered: Dict[str, Dict[str, Any]]) -> None:
    assert sorted(set(registered) - set(EXPECTED)) == [], "add the new tools to EXPECTED"
    assert sorted(set(EXPECTED) - set(registered)) == [], "EXPECTED names a tool that is gone"
    assert len(registered) >= 200


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


def test_every_read_only_tool_is_idempotent(registered: Dict[str, Dict[str, Any]]) -> None:
    """A call that changes nothing has nothing to repeat; the hint says so uniformly."""
    assert [
        name for name, hints in registered.items() if hints["readOnlyHint"] and not hints.get("idempotentHint")
    ] == []


def test_reset_tool_usage_stats_is_not_on_the_agent_surface(
    registered: Dict[str, Dict[str, Any]]
) -> None:
    assert "reset_tool_usage_stats" not in registered
    assert "get_tool_usage_stats" in registered
