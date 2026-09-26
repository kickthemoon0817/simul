# MCP tool catalog

The tools a server registers depend on which backends are available and
selected (`--backends`), on `--unreal-tools` / `--blender-tools`, and on
`security.allow_script_execution`. To see exactly what your server exposes:

```bash
simul tools            # add --json for descriptions and input schemas
```

Every tool returns a JSON object. Failures carry `success: false` and an
error type such as `NotRunning`, `RefusedOperation` or
`UnsupportedOperation`.

## Headless USD

No running application needed. File arguments must be inside the sandbox
(see [configuration.md](configuration.md#file-sandbox)).

| Group | Tools |
|---|---|
| Load and validate | `load_usd_file`, `validate_usd_file` |
| Query | `get_prim_info`, `search_prims`, `summarize_scene`, `get_mesh_info`, `get_bounding_box` |
| Edit | `create_prim`, `delete_prim`, `update_prim_attributes` |

## Isaac Sim

Requires a running Isaac Sim 5.1.0, 6.0.0 or 6.0.1 reachable over the bridge
(8229) or the stock Python socket (8226). See [isaac-sim.md](isaac-sim.md).

| Group | Tools |
|---|---|
| Connection and instances | `ping_isaac`, `get_isaac_runtime_info`, `list_isaac_instances`, `set_active_isaac_instance`, `claim_isaac_instance`, `release_isaac_instance` |
| Scene inspection | `get_isaac_stage_info`, `get_isaac_layer_info`, `list_isaac_prims`, `get_isaac_prim_detail`, `search_isaac_prims`, `get_isaac_scene_summary`, `get_isaac_subtree`, `get_isaac_scene_stats`, `get_isaac_selection`, `query_isaac_typed_prims` |
| Prim editing | `create_isaac_object`, `create_isaac_prim`, `delete_isaac_prim`, `set_isaac_prim_transform`, `set_isaac_prim_visibility`, `set_isaac_prim_attribute`, `duplicate_isaac_prim`, `reparent_isaac_prim` |
| Lights | `list_isaac_lights`, `create_isaac_light` |
| Cameras and viewport | `list_isaac_cameras`, `get_isaac_camera_info`, `set_isaac_camera`, `capture_isaac_viewport`, `focus_isaac_viewport`, `get_isaac_viewport_info` |
| Physics | `get_isaac_physics_scene`, `create_isaac_physics_scene`, `add_isaac_rigid_body`, `add_isaac_collision`, `set_isaac_mass_properties`, `set_isaac_physics_material`, `list_isaac_physics_objects` |
| Simulation control | `get_isaac_simulation_state`, `start_isaac_simulation`, `pause_isaac_simulation`, `stop_isaac_simulation`, `step_isaac_simulation`, `reset_isaac_simulation`, `get_isaac_simulation_time` |
| Materials | `list_isaac_materials`, `create_isaac_material`, `assign_isaac_material`, `set_isaac_material_property` |
| Rendering and settings | `read_isaac_aovs`, `list_isaac_aovs`, `list_isaac_render_vars`, `get_isaac_carb_settings`, `set_isaac_carb_settings` |
| OmniGraph | `list_isaac_graphs`, `get_isaac_graph_nodes`, `list_isaac_graph_node_types`, `create_isaac_graph_node`, `delete_isaac_graph_node`, `connect_isaac_graph_nodes`, `set_isaac_graph_node_values` |
| Stage and assets | `new_isaac_stage`, `open_isaac_stage`, `save_isaac_stage`, `import_isaac_asset`, `add_isaac_reference`, `get_isaac_texture_dependencies` |
| Spatial queries | `raycast_isaac_scene`, `find_isaac_prims_in_area` |
| Extensions | `list_isaac_extensions`, `enable_isaac_extension`, `disable_isaac_extension` |
| UI and logs | `get_isaac_ui_state`, `get_isaac_ui_window`, `get_isaac_logs`, `set_isaac_log_level` |
| Scripting | `execute_isaac_script`, `interrupt_isaac_script` |

Notes:

- `get_isaac_prim_detail` reads one aspect of a prim per call: `info`,
  `transform`, `ancestors`, `relationships`, `variants`, `bounding_box`,
  `mesh`, `light`, `material`, `rigid_body`, `collision`, `joint`, `mass`,
  `animation`.
- `create_isaac_object` creates a prim with transform, rigid body, collider,
  mass and material in one call.
- `get_isaac_ui_state` is a single snapshot of windows, focus, viewport,
  selection, timeline and stage; `get_isaac_ui_window` walks one window's
  widget tree.
- `execute_isaac_script` scripts should follow the scripting reference served
  as the `simul://isaac-sim/skills` resource. Isaac Sim 6.0 deprecates
  `isaacsim.core.{api,prims,utils}` and removes the `omni.isaac.*` shims.

## Unreal Engine

Requires an editor with Remote Control and Python enabled; run
`simul unreal setup <project>.uproject --yes`
([unreal-setup.md](unreal-setup.md)).

### Thin surface (default)

| Tool | Purpose |
|---|---|
| `unreal_health_check` | Connectivity, engine version, project name |
| `ping_unreal` | Reachability and latency |
| `list_unreal_instances` | Editors reachable on the scanned ports |
| `control_unreal_ui` | Named editor controls and per-agent viewport pointers in attached mode; works with script execution disabled. See [unreal-attachment.md](unreal-attachment.md) |
| `capture_unreal_viewport` | Viewport screenshot (PNG) |
| `execute_unreal_script` | Run Python inside the editor |

The rest of the operation set is available from the CLI
(`simul unreal ...`, see [cli.md](cli.md#simul-unreal)).

### Full surface (`--unreal-tools full`)

Also set with `unreal.tool_surface: full` or `UNREAL__TOOL_SURFACE=full`.

| Group | Tools |
|---|---|
| Engine and scene | `get_unreal_engine_info`, `get_unreal_loaded_map`, `summarize_unreal_scene`, `query_unreal_scene_graph`, `analyze_unreal_scene_for_robotics`, `get_unreal_viewport_info` |
| Actors | `list_unreal_actors`, `get_unreal_actor_info`, `describe_unreal_object`, `get_unreal_actor_by_semantic_label`, `get_unreal_actor_thumbnail`, `spawn_unreal_actor`, `delete_unreal_actor`, `set_unreal_actor_transform`, `set_unreal_actor_property`, `set_unreal_actor_visibility`, `set_unreal_actor_parent`, `add_unreal_component`, `focus_unreal_on_actor` |
| Assets | `search_unreal_assets` |
| Camera, lights, rendering | `set_unreal_camera_view`, `set_unreal_light_params`, `set_unreal_render_settings` |
| Materials | `get_unreal_material_info`, `assign_unreal_material`, `create_unreal_material_instance`, `set_unreal_material_params` |
| Physics and simulation | `enable_unreal_physics`, `set_unreal_physics_params`, `set_unreal_collision`, `apply_unreal_force`, `control_unreal_simulation`, `get_unreal_simulation_status` |
| Mesh operations | `generate_unreal_mesh_primitive`, `generate_unreal_procedural_scene`, `validate_unreal_mesh`, `simplify_unreal_mesh`, `remesh_unreal_mesh`, `subdivide_unreal_mesh`, `edit_unreal_mesh_topology`, `apply_unreal_mesh_boolean`, `cut_unreal_mesh_plane`, `compute_unreal_convex_hull`, `decompose_unreal_convex_hull`, `compute_unreal_mesh_uv`, `convert_unreal_mesh_format` |
| USD interchange | `get_unreal_interchange_info`, `import_unreal_usd`, `export_unreal_usd` |
| SimReady | `validate_simready_asset`, `convert_to_simready` |
| Generic dispatch | `call_unreal_actor_function`, `batch_unreal_operations` |

Behaviour worth knowing:

- Actor inspection, physics, materials and PIE start/stop/pause/resume use
  editor Python APIs. Single-frame stepping is not supported by the editor and
  returns `UnsupportedOperation`; simulation `frame_count` is `null` when the
  engine cannot report a simulation-specific count.
- USD import and export need the optional **USDImporter** plugin enabled when
  the editor starts. They use `AssetImportTask` and `LevelExporterUSD`;
  `get_unreal_interchange_info` reports whether the plugin is available.
  Import options accept booleans `import_actors`, `import_geometry`,
  `import_materials`, `import_lights`, `import_cameras`; export options accept
  booleans `export_actor_folders`, `export_sublayers`.
- SimReady conversion and validation are not implemented for Unreal. Those
  tools return `UnsupportedOperation` without changing files.
- Screenshots need console-command execution enabled in Remote Control. When
  upgrading an older project, re-run `simul unreal setup` and restart the
  editor.

## Blender

Registered when Blender is available: either `bpy` is importable by the
server (embedded mode) or a window is attached through the bridge add-on
(attached mode, [blender-attachment.md](blender-attachment.md)). `--blender-tools thin`
registers only `THIN_BLENDER_TOOLS` from `src/simul/tool_surfaces.py`.

| Group | Tools |
|---|---|
| Scene and files | `get_blender_info`, `get_blender_file_info`, `summarize_blender_scene`, `list_blender_scene_objects`, `search_blender_objects`, `open_blender_file`, `save_blender_file`, `import_blender_file`, `export_blender_file` |
| Objects | `create_blender_object`, `delete_blender_object`, `get_blender_object_info`, `set_blender_object_transform`, `set_blender_object_parent`, `clear_blender_object_parent`, `create_blender_mesh_from_data`, `get_blender_mesh_info`, `add_blender_modifier` |
| Measurement | `get_blender_bounding_box`, `check_blender_object_bounds`, `get_blender_distance_between` |
| Materials and lights | `get_blender_material_info`, `assign_blender_material`, `set_blender_light_params` |
| Physics | `setup_blender_rigid_body`, `add_blender_rigid_body_constraint`, `get_blender_constraint_info`, `add_blender_force_field`, `get_blender_force_field_info`, `get_blender_physics_state`, `bake_blender_simulation`, `free_blender_bake` |
| Animation | `get_blender_frame`, `set_blender_frame`, `set_blender_frame_range`, `play_blender_animation`, `insert_blender_keyframe`, `delete_blender_keyframe`, `get_blender_keyframes`, `get_blender_object_trajectory` |
| Camera and viewport | `get_blender_camera_info`, `set_blender_camera_view`, `get_blender_viewport_info`, `focus_blender_on_object`, `capture_blender_viewport`, `capture_blender_viewport_sequence` |
| UI and attachment | `control_blender_ui`, `attach_blender_window` |
| SimReady | `get_simready_metadata`, `apply_simready_metadata`, `setup_simready_hierarchy`, `validate_simready_compliance`, `export_simready_usd` |
| Scripting | `execute_blender_script` |

## Server

| Tool | Purpose |
|---|---|
| `get_tool_usage_stats` | Per-tool call counts, success rates and durations from the persistent usage log. Clearing it is operator-only: `simul stats --reset` |

## MCP resources

| URI | Contents |
|---|---|
| `simul://isaac-sim/skills` | Isaac Sim 5.1 / 6.0 scripting patterns for `execute_isaac_script` |
| `simul://isaac-sim/api/{core,sensors,physics,replicator,robots,rendering,assets}` | Isaac Sim API references by area |
