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

- **Bounds.** `get_bounding_box`, and the bounds in `get_prim_info` and
  `summarize_scene`, are USD's own `UsdGeom.BBoxCache` bounds: every
  transform applies and instances are followed. Only geometry whose purpose
  is in `include_purposes` counts; the default, `default` and `render`, is
  what renderers and simulators see, so `proxy` and `guide` geometry (often
  collision shapes) needs `include_purposes` to be bounded. A prim with
  nothing boundable is a successful result with `has_geometry: false` and
  `bbox: null`, never a guessed box; `excluded_purposes_with_geometry` then
  names the purposes left out that do hold geometry. `world_space: false`
  bounds the prim in its own space (its descendants' transforms apply, its
  own does not), and `time_code` evaluates animated transforms.
- **Lookups.** A `stage_id` that is not loaded, or a prim path with no prim,
  is a `NotFoundError`; a relative, malformed or property path is a
  `ValidationError`.

## Isaac Sim

Requires a running Isaac Sim 5.1.0, 6.0.0 or 6.0.1 reachable over the bridge
(8229) or the stock Python socket (8226). See [isaac-sim.md](isaac-sim.md).
Registered by default only on Linux and Windows, where Isaac Sim runs
([Platforms](isaac-sim.md#platforms)).

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
| `list_unreal_instances` | Editors on the scanned ports: identity, viewports, active and attached flags. Same payload as `simul unreal instances`; see [unreal-attachment.md](unreal-attachment.md#discover-and-select) |
| `control_unreal_ui` | Named editor controls and per-agent viewport pointers in attached mode; works with script execution disabled. See [unreal-attachment.md](unreal-attachment.md) |
| `capture_unreal_viewport` | Viewport screenshot (PNG or JPEG): the level editor viewport, or the player's view during Play In Editor |
| `execute_unreal_script` | Run Python inside the editor |

The rest of the operation set is available from the CLI
(`simul unreal ...`, see [cli.md](cli.md#simul-unreal)).

### Full surface (`--unreal-tools full`)

Also set with `unreal.tool_surface: full` or `UNREAL__TOOL_SURFACE=full`.

| Group | Tools |
|---|---|
| Engine and scene | `get_unreal_engine_info`, `get_unreal_loaded_map`, `summarize_unreal_scene`, `load_unreal_world_partition_actors`, `query_unreal_scene_graph`, `analyze_unreal_scene_for_robotics`, `get_unreal_viewport_info` |
| Actors | `list_unreal_actors`, `get_unreal_actor_info`, `list_unreal_actor_components`, `describe_unreal_object`, `get_unreal_actor_by_semantic_label`, `get_unreal_actor_thumbnail`, `spawn_unreal_actor`, `delete_unreal_actor`, `set_unreal_actor_transform`, `set_unreal_actor_property`, `set_unreal_actor_visibility`, `set_unreal_actor_parent`, `add_unreal_component`, `focus_unreal_on_actor` |
| Assets | `search_unreal_assets` |
| Camera, lights, rendering | `capture_unreal_camera`, `set_unreal_camera_view`, `set_unreal_light_params`, `set_unreal_render_settings` |
| Materials | `get_unreal_material_info`, `assign_unreal_material`, `create_unreal_material_instance`, `set_unreal_material_params` |
| Physics and simulation | `enable_unreal_physics`, `set_unreal_physics_params`, `set_unreal_collision`, `apply_unreal_force`, `control_unreal_simulation`, `get_unreal_simulation_status` |
| Mesh operations | `generate_unreal_mesh_primitive`, `generate_unreal_procedural_scene`, `validate_unreal_mesh`, `simplify_unreal_mesh`, `remesh_unreal_mesh`, `subdivide_unreal_mesh`, `edit_unreal_mesh_topology`, `apply_unreal_mesh_boolean`, `cut_unreal_mesh_plane`, `compute_unreal_convex_hull`, `decompose_unreal_convex_hull`, `compute_unreal_mesh_uv`, `convert_unreal_mesh_format` |
| USD interchange | `get_unreal_interchange_info`, `import_unreal_usd`, `export_unreal_usd` |
| SimReady | `validate_simready_asset`, `convert_to_simready` |
| Generic dispatch | `call_unreal_actor_function`, `batch_unreal_operations` |

Behaviour worth knowing:

- The scene reads go through the editor world, one page at a time.
  `list_unreal_actors`, `get_unreal_actor_by_semantic_label`,
  `list_unreal_actor_components` and `query_unreal_scene_graph` count every
  match (`total`; `total_matches` for the label search) and return the page
  from `offset`: up to `max_results` entries, stopping earlier rather than go
  past 30,000 characters, so a reply always fits the 40 KB result budget
  untrimmed. `count` is what came back, `next_offset` is where the next page
  starts (`null` at the end), and `truncated` and `hint` say when more is left.
  At 5,000 actors a page of actors holds about 80, in about 0.2 s.
  `list_unreal_actors` filters by class (name, as in `StaticMeshActor` or
  `BP_Door_C`, or path; `include_subclasses` also keeps derived classes, so
  `Light` keeps point, spot and Blueprint lights), tag and label (text the
  label contains, ignoring case). A class `include_subclasses` cannot resolve
  (a Blueprint class is found by its path, or by its name when an actor of
  that class is in the level) is matched by name only, with a `warning`.
  Actor entries carry the attachment hierarchy: `parent` (the actor the root
  component is attached to), `children` (the first 20 attached actors;
  `get_unreal_actor_info` names 200) and `child_count`.
- `query_unreal_scene_graph` returns the attachment hierarchy as a tree:
  the level, whose children are the actors attached to nothing, or the actor
  at `root_path`, down `max_depth` levels. The page holds the root's
  children from `offset`, each with its subtree; an actor whose children lie
  deeper has `children_truncated`, and a subtree too large for one page is
  drawn shallower and named in `depth_limited` (query it with its actor as
  `root_path`). `total_actors` and `total_depth` describe the whole tree. A
  `class_filter` keeps the matching actors and the actors above them, which
  carry `matches_filter: false`; `include_components` adds each actor's
  components (name, class, root flag, the component it is attached to).
- In a World Partition map the editor loads only some regions' actors, and
  the scene reads see only loaded actors. `summarize_unreal_scene`,
  `list_unreal_actors`, `get_unreal_actor_by_semantic_label` and
  `query_unreal_scene_graph` report `world_partition` and `unloaded_actors`
  (the map's saved actors, from its World Partition actor descriptors, that
  are not loaded) and warn when some are unloaded; a saved map opened
  normally can have nearly all of them unloaded. `load_unreal_world_partition_actors`
  loads the actors whose bounds meet a box (`bounds_min`, `bounds_max`), or
  every actor, through `WorldPartitionBlueprintLibrary.load_actors`, which
  loads them at once and leaves the map unmodified; it refuses, changing
  nothing, when more than `max_actors` (default 500) would load. `action`
  `unload` releases the actors it loaded (the editor's own loading regions
  keep theirs: `not_changed`). Loaded actors stay loaded until unloaded or
  another map opens. During Play In Editor Unreal's World Partition library
  finds no editor world, so `world_partition` and `unloaded_actors` are null
  and loading is refused (`PIEActive`). CLI: `simul unreal wp-load`.
- `list_unreal_actor_components` lists an actor's components, editor-only
  ones included: class, root and editor-only flags, and for scene components
  the component they are attached to (its path when it belongs to another
  actor), world location, mobility and visibility; primitive components add
  their world bounds (box center and half-size, sphere radius, in cm).
- Actor inspection, physics, materials and PIE start/stop/pause/resume use
  editor Python APIs. Single-frame stepping is not supported by the editor and
  returns `UnsupportedOperation`; simulation `frame_count` is `null` when the
  engine cannot report a simulation-specific count.
- `set_unreal_material_params` writes a Material Instance Constant through
  Unreal's MaterialEditingLibrary in one undoable transaction, in or out of
  Play In Editor, and reads every value back, so `success` alone tells
  whether every value applied. It sets the material's global parameters.
  The request is checked first: a name the material has no global parameter
  of that kind for (names match ignoring case, so two names that differ only
  in case are one parameter and may not both be given), a texture that does
  not load or a malformed value fails the call with `ValidationError`, and
  nothing is written; `not_applied` gives each parameter's reason. That
  includes a name only the material's Material Layers have (a `layer` or
  `blend` parameter in `get_unreal_material_info`; the reason names each layer
  and blend): Unreal's Python API writes parameters by name, and the global
  override it would add reads back as set while the layer renders as it was.
  A name both the material and a layer have sets the material's own
  parameter. After the write, a value that reads back different fails the
  call with `MaterialParamsNotApplied`, naming it in `not_applied`; the
  values that did apply stay written, as with lights. `applied` holds the
  values read back (scalars and colors as the 32-bit floats Unreal stores)
  and `params_set` counts them. The instance is left unsaved.
- USD import and export need the optional **USDImporter** plugin enabled when
  the editor starts. They use `AssetImportTask` and `LevelExporterUSD`;
  `get_unreal_interchange_info` reports whether the plugin is available.
  Import options accept booleans `import_actors`, `import_geometry`,
  `import_materials`, `import_lights`, `import_cameras`; export options accept
  booleans `export_actor_folders`, `export_sublayers`.
- SimReady conversion and validation are not implemented for Unreal. Those
  tools return `UnsupportedOperation` without changing files.
- `analyze_unreal_scene_for_robotics` classifies the actors loaded in the
  editor world by the bounding box of their collision-enabled components:
  traversable floor-like surfaces (physics collision, at least 1 m² in plan, no
  thicker than a quarter of the shorter side, tilted at most 15°), graspable
  objects (Movable, physics collision, fitting in a 60 cm cube) and the
  collision setup of their static meshes (simple shapes, convex hulls, render
  triangles used as simple collision, physics collision without any simple
  shape). A landscape section counts whole, hills included. The thresholds
  come back as `criteria`.
- `generate_unreal_procedural_scene` builds a room, corridor, warehouse or
  outdoor scene from Engine basic shapes inside the bounds. The layout is
  planned before anything is spawned, so one that does not fit fails without
  touching the level, and the same arguments give the same layout. The seed is
  an integer from 0 to 4294967295 (2³² − 1), and each seed gives its own
  layout. The actors are spawned in one undoable transaction. They are labelled
  `<Scene>_<Kind>_<n>`, tagged `simul_procedural` and
  `simul_<scene>_<seed>_<generation>`, and filed in the outliner folder
  `Simul/<Scene>_<seed>_<generation>`, where `<generation>` is 8 hex digits
  drawn fresh for every call: generating the same scene twice gives two groups
  with their own tag and folder, and the tag is what `list_unreal_actors`
  filters on. If a spawn fails, every actor already spawned is removed (one
  Unreal refuses to remove is listed in `details.actors_left`) and the
  transaction is cancelled. The actors are filed in their folder only once all
  have spawned, because Unreal keeps an outliner folder, empty, after its
  actors go and its Python API cannot delete one; a failed call therefore
  leaves no folder behind. Props are Movable boxes; everything else is Static.
- `get_unreal_material_info` reads a Material or Material Instance through
  editor scripting, in or out of Play In Editor: each scalar, vector and
  texture parameter with its value (a Material's default; for an instance,
  the value it renders with and whether the instance overrides it), the
  parent and base material, the material domain and the blend mode (an
  instance's override, else its parent's), named as Unreal names them
  (`MD_SURFACE`, `BLEND_OPAQUE`; an instance without a parent renders
  `BLEND_OPAQUE`). An override an instance keeps for a name that is no
  longer a parameter of its material is not listed, as the Material Instance
  editor does not show it. Each parameter has an `association`: `global` for
  the material's own, `layer` or `blend` for one inside its Material Layers,
  with the layer or blend `index` (layer 0 is the bottom layer, and blend i
  blends layer i + 1 onto the layers below; `null` for a global parameter).
  A layer or blend parameter has that layer's value and override, and a name
  the material and its layers share is listed once for each. Unreal's Python
  API lists and reads parameters by name only, so the tool reads them through
  a transient Material Instance Dynamic that copies the material's
  parameters with their layers.
- `focus_unreal_on_actor` points the level viewport camera at the actor and
  selects it, reading the selection back: `selected` is false, with a
  `warning`, while PIE runs (below) and for an actor Unreal does not select:
  one hidden in the editor or in a hidden or locked level, a child actor or an
  actor inside a level instance (Unreal selects those only through their
  parent), or any actor while the editor mode or a selection lock prevents
  selection. The camera stands `distance` behind the point it looks at and
  half as far above it; the reply's `distance` is the one used. A positive
  `distance` looks at the actor's location. With `distance` 0 (the default) or
  less the camera looks at the center of the actor's bounds and stands back until
  their bounding sphere fills the narrower side of the view with 10% to spare,
  and at least until the sphere, with that margin, is beyond Unreal's 10 cm
  near clip plane, which a small actor in a wide or square view needs (the
  engine's default near clip; a project that changes it is not detected).
  The view is `fov` degrees across (default 90, UE's default; at least 0.001,
  the narrowest Unreal renders, and less than 180) and `aspect_ratio` wide per
  unit of height (default 16:9, the default 1920x1080 capture's; from 1/16384
  to 16384, the shapes a capture can have), and keeps its field of view across
  the width, as Unreal's viewports and captures do. Editor Python cannot read
  the level viewport's field of view, so pass `fov` when it was changed, and
  `aspect_ratio` for a capture of another shape: otherwise a narrower field of
  view, or a view wider than about 2:1, can crop the actor. Both are checked
  even when a positive `distance` makes them unused. An actor whose
  bounds have no size, such as a StaticMeshActor without a mesh, gets 500 cm
  from its location instead, with a `fit_warning`; an empty actor or a light
  is framed by its editor icon. The editor-only components the editor's own
  Focus leaves out are left out of the bounds too, since theirs can reach far
  beyond the actor, so a camera is framed by its editor mesh. They include the
  frustum the editor draws for a camera or a scene capture (10 m on every side
  of it), a CineCamera's debug focus plane (100 m across at the focus
  distance, 1 km by default; turning the plane off only hides it) and a spare
  icon attached to nothing, such as the one a SkyLight leaves at the world
  origin. A shape (box, sphere or capsule component) whose collision is turned
  off is left out too, as the editor's own Focus leaves out every shape: it
  collides with nothing, like a SphereReflectionCapture's influence sphere
  (30 m in radius by default) or a BoxReflectionCapture's boxes, so a
  reflection capture is framed by its icon, as a light is. Any other component
  that is not editor-only counts, such as a trigger box's collision shape;
  with the actor's own collision turned off (Actor Enable Collision) its
  shapes collide with nothing either, so such a trigger box is framed by its
  icon. An
  actor whose bounds are not the box around its components, such as a level
  instance or a NavLinkProxy, is framed by the bounds Unreal reports.
- While Play In Editor (or Simulate In Editor) runs, Unreal's editor refuses
  changes to the editor world: it spawns and destroys no actors, lists and
  selects none, and hands out no editor world. The tools detect it:
  - `spawn_unreal_actor`, `delete_unreal_actor`,
    `generate_unreal_procedural_scene`, the mesh tools that create an actor
    (`generate_unreal_mesh_primitive`, `compute_unreal_convex_hull`,
    `decompose_unreal_convex_hull`, `convert_unreal_mesh_format` to
    `dynamic_mesh`), `import_unreal_usd`, `export_unreal_usd` and, in attached
    mode, every `control_unreal_ui` action except `inspect` and `clear_cursor`
    return `error_type: "PIEActive"`, with `details.hint` naming the way out:
    `control_unreal_simulation` with action `stop` (CLI: `simul unreal sim stop`).
  - `create_unreal_material_instance` returns the same `PIEActive` error: it
    loads the parent material and saves the new instance through Unreal's
    editor asset library, which loads, finds and saves no assets during PIE.
  - `list_unreal_actors`, `get_unreal_actor_by_semantic_label`,
    `query_unreal_scene_graph`, `list_unreal_actor_components`,
    `summarize_unreal_scene`, `analyze_unreal_scene_for_robotics`,
    `get_unreal_actor_info`, `get_unreal_loaded_map` and
    `get_unreal_engine_info` report `pie_active: true` and read the editor
    world, not the running game. Because Unreal's level listing is withheld,
    the listings enumerate the editor world directly and can include the
    level's builder brush and engine helper actors the Outliner hides; the
    robotics analysis reports its static mesh collision counts, and the list
    of actors missing simple collision, as `null`, since Unreal does not
    report mesh collision during PIE.
  - Editing actors already in the level by path (transform, lights, collision,
    physics settings, visibility, materials) keeps working and changes the
    editor world.
  - `set_unreal_material_params` edits the material instance asset, which the
    running game uses too: an edit made during PIE shows in the game and stays
    after PIE stops.
  - `focus_unreal_on_actor` still moves the viewport camera but selects
    nothing, since Unreal reads back no selection during PIE and clears it
    when PIE stops: the reply has `selected: false`, `pie_active: true` and a
    `warning`. After Simulate In Editor (what `control_unreal_simulation`
    starts) the camera stays where it was put; when a Play session in the
    level viewport stops, Unreal replaces the editor camera, by default with
    the game camera's last view.
- `capture_unreal_viewport` writes the image into the editor's
  `Saved/Screenshots` and returns its path on the editor host; `inline` also
  sends it as an image block, up to 256 KB. `format: "jpeg"` converts the PNG
  HighResShot writes into a JPEG beside it in the editor (about a tenth of the
  size: a 640x360 capture is about 27 KB), and `inline_max_size` inlines a copy
  scaled to that many pixels on its longer side while the file keeps full
  size. `view` picks what is captured, and the reply says which (`view`,
  `pie_active`): `editor` is the level editor viewport, whose camera
  `set_unreal_camera_view` moves; `game` is the player's view during a Play In
  Editor session (HighResShot run in the game world goes to its game
  viewport); `auto` (the default) takes the game view when a Play session's
  game viewport draws. Simulate In Editor (what `control_unreal_simulation`
  starts) shows the simulation in the level viewport and its game viewport
  never draws, and Unreal's Python cannot tell it from Play, so `auto` tries
  the game view for 2 s and then captures the level viewport, with a
  warning; `game` fails with `InvalidState`. An editor HighResShot goes to
  every level viewport pane that draws, while Unreal's camera calls
  (`set_unreal_camera_view`, `focus_unreal_on_actor`,
  `get_unreal_viewport_info`) read and move only the first perspective pane:
  when several panes write a screenshot the newest is returned with a
  warning that it may not be that pane, so maximize the pane or use a
  one-pane layout. (The headless editor `simul unreal setup` starts has a
  four-pane layout in which only the perspective pane draws.)
- `get_unreal_viewport_info` returns the first perspective pane's camera, so
  `projection_type` is `Perspective`. Editor Python cannot read a level
  viewport's field of view (UE 5.7 keeps it in config-only settings), so `fov`
  is null and `notes` says so; pass the real value to `focus_unreal_on_actor`.
  The viewport's size is not readable either: `measure_size: true` measures
  it with a HighResShot at the viewport's own size (the file is deleted).
  During Play In Editor `game_view` is the player camera (location, rotation,
  field of view) and the game viewport's size, which is also `viewport_size`;
  `viewports` and `active_viewport` name the layout's panes.
- `capture_unreal_camera` renders what a camera or sensor actor sees: a
  CameraActor, CineCameraActor, SceneCapture2D or any actor with a
  CameraComponent or SceneCaptureComponent2D (that one first). A transient
  SceneCapture2D stands at that component with its projection, field of view
  across the width (a CineCamera's from its focal length and filmback; its
  `field_of_view` property can be stale), ortho width and post-process
  settings, hides the actor itself, and captures final color once; the
  image is written as PNG or JPEG into the editor's screenshot directory, and
  the capture actor is destroyed (it is transient, so the map stays
  unmodified). The reply has the camera's pose, fields of view and lens, and
  `intrinsics`: the pinhole model of this image (square pixels, `cx`, `cy` at
  the image center in OpenCV's pixel-center convention; Unreal's camera +X is
  the optical axis, +Y image right, -Z image down; null for an orthographic
  camera). A capture whose shape differs from a camera that constrains its
  aspect ratio comes with a warning. During Play In Editor the actor's
  counterpart in the running game is captured (`world: "game"`); an actor
  with none there is refused (`PIEActive`). A one-shot scene capture has no
  auto-exposure history, so its exposure can differ from the viewport's.
  RGB only: depth, segmentation and other render passes are not captured.
  CLI: `simul unreal capture-camera`.
- Screenshots need console-command execution enabled in Remote Control. When
  upgrading an older project, re-run `simul unreal setup` and restart the
  editor.
- `execute_unreal_script` runs the code you send in the editor's persistent
  Python console namespace through `PythonScriptLibrary.ExecutePythonCommandEx`,
  in or out of PIE. On UE 5.7 avoid one engine hazard in that code: Unreal wraps
  each static-library function
  (`unreal.MaterialEditingLibrary.get_scalar_parameter_names`,
  `unreal.EditorAssetLibrary.*`, `unreal.SystemLibrary.*`, every `*Library` /
  `*Statics` call) and each struct method (`unreal.Vector.length`) in a
  `builtin_function_or_method_with_closure` object whose weak-reference slot is
  laid out one pointer past the end of the object. If such an object is left in
  a reference cycle that becomes unreachable, the first cyclic garbage
  collection that sweeps it crashes the editor in `_PyWeakref_ClearRef`. That
  collection is often the editor's own, on autosave or level load, so the script
  that built the cycle can return cleanly and the crash land minutes later. Call
  these functions inline (`unreal.MaterialEditingLibrary.get_scalar_parameter_names(mat)`)
  and don't stash them in a dict, list, closure or object that forms a cycle; a
  reference that is dropped or stays reachable is safe. Module functions
  (`unreal.load_asset`) and the common `unreal.Object` accessors
  (`actor.get_name`, `actor.get_editor_property`) are a different, in-bounds
  wrapper and are safe, and simul's own tools call the library functions inline,
  so only `execute_unreal_script` payloads are exposed.

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

Behaviour worth knowing:

- `capture_blender_viewport` captures the 3D View the user sees in attached
  mode and needs no camera; `method: "render"` renders through the scene camera,
  or a temporary one when the scene has none. Embedded (headless) mode has no 3D
  View, so its captures are renders. The per-mode table is in
  [blender-attachment.md](blender-attachment.md#capture-feedback).
- `set_blender_camera_view` and `get_blender_camera_info` work in world space:
  `location` and an XYZ Euler `rotation_euler` in radians, whatever the
  camera's parent and rotation mode (glTF imports store quaternions).
  `focus_blender_on_object` aims the camera from the front and above at the
  object's bounding-box center, at the distance where the bounding sphere fits
  the frame through the camera's field of view, the render aspect ratio, sensor
  fit and lens shift (`distance_factor: 1.0`, the default; 2.0 shows the object
  at half that size). An orthographic camera gets the `ortho_scale` that fits
  instead. A clip range that would cut the object away is widened, and the
  reply reports `distance`, `clip_start`, `clip_end` and `clip_adjusted`.
- Both placing tools write the camera's own transform; Blender then applies
  the camera's constraints (Track To, Copy Location, Child Of, ...) and drivers
  on top, so success does not mean the camera ends up at the placement. The
  reply reports where Blender evaluates the camera (`location` and
  `rotation_euler`, or `camera_location` and `camera_rotation_euler`). When
  that is not the placement, `placement_overridden` is `true`,
  `constraints_applied` lists the active constraints (not muted, influence
  above 0, valid, target not deleted: Blender keeps calling such a constraint
  valid) and `warning` names the cause. Rounding is not reported as an
  override, so the reported pose can miss the placement by what single
  precision explains: up to a few milliradians where Blender stores the
  rotation near gimbal lock of the camera's Euler order or as an axis-angle
  near zero (not under a parent with non-uniform scale, where any such miss
  is reported), and more where a constraint aims the camera at a target
  close to it far from the origin. Only the aim's own two locations count,
  so a far target of another constraint changes nothing. Looking almost
  straight up or down at its Track To's target, the camera's roll is
  checked unless the target stands beside the camera's axis by no more
  than single precision resolves at those coordinates. Aimed exactly
  straight up or down (as single precision stores both locations), or
  standing at its target, a Track To gives the camera a fixed pose, world
  +Y up, so a placement off that pose is reported. For constraints
  it says how to disable them: mute them or set their influence to 0, then
  place the camera again. A driver on the camera's transform, or a parent with
  non-uniform scale, is named instead when it moves the camera before the
  constraints apply, and also when they leave that pose alone. Simul never
  mutes or removes a constraint itself. `get_blender_camera_info` reports the
  evaluated pose and `constraints_applied` too. Keys on the camera's transform
  take over again at the next frame change. `override_check` is `"done"` when
  Blender evaluated the camera after the placement. Blender's viewport skips a
  camera that is disabled in viewports (the monitor icon on the camera or on a
  collection holding it, or its collection excluded from the view layer)
  unless it is the scene camera, a timeline marker binds it, or something
  visible depends on its transform (a child, or a constraint, driver or
  modifier that reads it); its constraints then do not run there, although a
  render through it applies them. The scene camera and a marker's camera
  lose that exception
  while their `hide_viewport` is animated (keyframes, a driver, or a strip in
  an NLA track that plays: not muted, or the solo track when there is one)
  and a collection not excluded from the view layer holds them: Blender then
  skips such a camera while its
  `hide_viewport` is on at the current frame, or while each of those
  collections is disabled in viewports. Such a disabled camera is checked
  when Blender holds an evaluated copy of it whose pose changed with the
  placement. Otherwise
  `override_check` is `"skipped"`, `placement_overridden` is `null`, the
  reported pose is the requested placement, unchecked, `constraints_applied`
  lists the constraints that apply when Blender evaluates it, and `warning`
  names the setting that disables it. That covers a camera Blender evaluates
  only in part (a driver reads only one of its properties, or its visibility
  is animated), and also a disabled camera Blender does evaluate when the
  placement leaves its pose as it was: placed where it already is, or under
  constraints that set its whole transform. Neither the depsgraph's member
  IDs nor `is_evaluated` tells these cases apart. A camera hidden with the eye
  icon is still evaluated and checked. An add-on built before this report
  answers without these fields, and the server returns them as `null`.
- `get_blender_camera_info` reports where Blender evaluates the camera
  (`pose_source: "evaluated"`). A camera the viewport skips that way keeps the
  `matrix_world` of an earlier evaluation, so once it moves, that matrix no
  longer says where it is: on Blender 5.0.1 a camera moved after it was disabled
  was reported 11 m from where a render through it put it. For such a camera
  the reply composes the pose from its own location, rotation and scale through
  its parent (`pose_source: "transform"`). That pose leaves out its constraints,
  drivers and keys, which a render through it applies, and `warning` names the
  settings that disable it and the constraints left out. A parent disabled too
  may not have been evaluated since it moved, and `warning` then says the pose
  was composed through that parent's last evaluated matrix. Parent types
  ARMATURE and LATTICE give the camera its parent's `matrix_world`, as an
  object parent does (their deform moves only geometry), so the pose is
  composed through them too. A parent whose matrix only Blender computes (a
  bone, vertices or a curve, for instance) leaves the camera's last evaluated
  pose, with a warning that it may be stale. A disabled camera that an object
  Blender evaluates needs, also through other disabled objects, is evaluated
  in full and keeps its evaluated pose. That object needs it through a child,
  a constraint, or a modifier enabled in viewports that reads its transform:
  a Hook, an Array's offset object, a UV Project projector, a texture mapped
  through it, and so on. Before reading that pose the reply tags the camera
  and has Blender evaluate again, which evaluates what depends on it again
  too: Blender 5.0.1 kept the stale `matrix_world` of a camera it came to
  need without rebuilding relations (a modifier's `show_viewport` switched
  on or keyed, or the keyed `hide_viewport` of the object that needs it)
  through later evaluations, until the camera was tagged. A camera the
  depsgraph does not hold is composed. A Boolean's object operand and a Data
  Transfer's object transform are not followed, as Blender rebuilds no
  relations when `operand_type` or `use_object_transform` changes. Geometry
  Nodes object inputs are not followed either: Blender relates an input that
  a script sets (`modifier[key] = object`) only at its next relations update,
  as such a write tags nothing, so a camera that only a Geometry Nodes input
  needs is composed. A constraint on a bone, and a modifier,
  count while their armature or object is enabled: through a disabled one,
  Blender 5.0.1 evaluated the camera or left it stale depending on the order
  of the view layer's objects alone. A child parented to the bone whose
  constraint targets the camera (parent type BONE), or to a bone below it,
  counts whatever its armature: Blender relates the child to its bone and
  each bone to the one above it, and 5.0.1 evaluated the camera for such a
  child in all 24 orders of the child, an object reading it, the armature
  and the camera, but for a child of the armature itself, of a bone above
  the constrained one or of another bone only in the 12 where the child came
  before its reader. An IK constraint counts only on a bone that has a parent
  or uses its tail, and a Spline IK only on a bone: Blender relates their
  targets to an armature's IK solver, so to nothing on an object or for an
  IK without a chain, and 5.0.1 left the camera stale for those. A particle
  system's boid rule that avoids or follows the camera (Avoid, Follow
  Leader), and a texture its settings map through the camera (OBJECT
  coordinates), count while the depsgraph holds the
  object they name for something else, the particle system's modifier
  enabled in viewports or not: Blender builds that object for neither, so
  without another reference it relates them to nothing, and 5.0.1 left the
  camera stale then. Its instance
  object and instance collection are not followed: Blender relates them
  only while `render_type` shows them and rebuilds no relations when
  `render_type` changes, so on 5.0.1 a camera named while `render_type` was
  HALO, switched to OBJECT by a script afterwards, stayed 11 m from a render
  while a muted Hook held it. Nor are keyed targets, which Blender 5.0.1
  evaluated only when the mesh came before an object reading it in the view
  layer, or the camera's force field: Blender 5.0.1 took a camera disabled
  in viewports for no effector, except one only excluded from the view
  layer that an effector collection named, and evaluated that one only
  while something else held it. One that only a driver, or a constraint
  that is muted or has no influence, depends on is composed: Blender 5.0.1
  evaluated it for those too, but such a constraint needs nothing of its
  target, and a driver that reads only a custom property of the camera left
  its matrix stale. Objects named inside a Geometry Nodes group, or through a
  collection, are not followed, so a camera that only they need is composed
  as well. An add-on built before this report answers without
  `pose_source` and `warning`, and the server returns them as `null`.
- `search_blender_objects` refuses a `name_pattern` that repeats a group holding
  a quantifier or alternatives, such as `(\w+)+`: Python's regex search
  backtracks, and on a long object name such a pattern runs for hours on
  Blender's main thread. `get_unreal_actor_by_semantic_label` refuses the same
  patterns in regex mode.
- `execute_blender_script` returns the first `max_output_chars` characters of
  printed output (default 32 000, at most 100 000). A longer output sets
  `output_truncated: true`, and `output_total_chars` gives its full length, so a
  cut JSON dump is reported rather than failing to parse later. The default
  keeps a typical result inside the server's 40 KB result budget. Above its
  400 KB hard limit the server replaces the whole field instead of returning a
  prefix, and it counts JSON bytes: at most 2 per printable ASCII character, 6
  per control or other non-ASCII character and 12 per emoji. So output is also
  cut at 300 000 bytes of JSON, which printable ASCII never reaches but heavily
  non-ASCII output can, with the same report. Write larger results to a file
  and return its path.

## Server

| Tool | Purpose |
|---|---|
| `get_tool_usage_stats` | Per-tool call counts, success rates and durations from the persistent usage log. Clearing it is operator-only: `simul stats --reset` |

## MCP resources

| URI | Contents |
|---|---|
| `simul://isaac-sim/skills` | Isaac Sim 5.1 / 6.0 scripting patterns for `execute_isaac_script` |
| `simul://isaac-sim/api/{core,sensors,physics,replicator,robots,rendering,assets}` | Isaac Sim API references by area |
