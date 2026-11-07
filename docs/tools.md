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

Results are strict JSON: a non-finite float is sent as the string
`"Infinity"`, `"-Infinity"` or `"NaN"`, never as a bare token. Each result
is held to a 40 KB budget by trimming its largest top-level list, with
`truncated` and a `truncation` notice. When that list is a page, its
`next_offset` (and a `count` or `returned` that counted the page) are
rewritten to match the items kept, so following `next_offset` skips
nothing. Above a 400 KB hard limit the field carrying the excess is cut at
any depth: a string becomes a `truncated_field` marker, and a list or
mapping keeps its prefix and is named in `truncated_fields`.

## Headless USD

No running application needed. File arguments must be inside the sandbox
(see [configuration.md](configuration.md#file-sandbox)).

| Group | Tools |
|---|---|
| Load and validate | `load_usd_file`, `validate_usd_file` |
| Query | `get_prim_info`, `get_prim_subtree`, `search_prims`, `summarize_scene`, `get_mesh_info`, `get_bounding_box` |
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
- **Prim info.** `get_prim_info` returns each attribute as
  `{type, value, authored}` (`authored: false` is a schema fallback), each
  relationship as `{targets, authored}`, applied API schemas, and metadata:
  specifier, instanceable, composition arcs with `resolved`, variant sets
  and selections, the layers holding opinions, and other authored metadata.
  An array attribute carries its `length` and first `array_preview`
  elements (default 16, up to 10000), so a million-point mesh costs a few
  KB; `attribute_filter` takes glob patterns, `time_code` reads animated
  values.
- **Traversal and counts.** Every listing walks the composed stage with one
  rule: active, defined, loaded, non-abstract prims, plus the instance
  proxies inside instanceable prims. So `load_usd_file`'s `total_prims`,
  `summarize_scene`'s `total_prims` and a match-everything `search_prims`
  agree, and an instance's children are counted.
- **Search.** `search_prims` matches `by_name`, `by_path`, `by_type` or
  `by_api_schema`; `match` is `auto`, `exact`, `substring`, `glob` or `regex`,
  and names compare case-insensitively unless `case_sensitive`. With `auto`,
  a registered type matches its schema family (`PhysicsJoint` finds every
  joint, `Gprim` every geometric primitive), and an API schema matches its
  multiple-apply instances (`PhysicsDriveAPI` finds `PhysicsDriveAPI:angular`).
  `root_path` limits the search and `include_instance_proxies: false` skips
  instances. Results are pages: `offset`/`limit` (default 200, up to 5000),
  with `total_matches`, `truncated` and `next_offset`.
- **Subtrees.** `get_prim_subtree` lists a prim and its descendants
  depth-first (path, type, depth, child count, API schemas), paged with
  `max_results`/`offset`. `include_properties` adds each prim's authored
  attributes and relationship targets, so a whole robot reads in one call.
- **JSON values.** Vectors are lists, quaternions `[w, x, y, z]`, matrices
  lists of rows, asset paths their authored string.
- **Lookups.** A `stage_id` that is not loaded, or a prim path with no prim,
  is a `NotFoundError`; a relative, malformed or property path is a
  `ValidationError`.

`validate_usd_file` opens the file with pxr, as `load_usd_file` does, and
keeps no stage. `can_read` says whether pxr opened it (so whether
`load_usd_file` can); `is_valid` additionally needs no issue of severity
`error`. Each issue carries `severity`, a stable `code`, pxr's `message`, its
`source` (`simul` or the pxr validator) and a `location` (`layer`,
`prim_path`, `property_path`, `asset_path`, or `line` and `column` for a
parse error). The checks, in order:

- Access, shared with the loader: the file exists, is a regular file, has an
  extension from `usd.allowed_extensions`, is readable by the server process,
  is not empty and fits `usd.max_file_size_mb` (`FileNotFound`, `NotAFile`,
  `UnsupportedExtension`, `PermissionDenied`, `EmptyFile`, `FileTooLarge`).
- Parsing: pxr's own reason (`ParseError` with line and column,
  `CorruptFile` for crate or usdz damage, `UnknownFormat`), with a hint when
  the bytes do not match the extension (text in a `.usdc`).
- Composition: `MissingSublayer`, `MissingReference`, `MissingPayload`,
  `UnresolvedPrimPath`, and pxr's `UnresolvableDependency` for other asset
  paths such as textures.
- Stage metadata: a dangling defaultPrim is an error; an unset defaultPrim,
  upAxis or metersPerUnit is a warning, since consumers fall back to Y up and
  centimeters.
- Physics: `NestedRigidBody` (without `resetXformStack`) and
  `InvertedJointLimits`, plus pxr's UsdPhysics validators (dangling joint
  bodies, nested articulation roots, invalid mass and inertia).
- pxr's UsdValidation validators, listed in `validators`; the usdz package
  rules (alignment, compression, file types) run only for a `.usdz`.

`load_usd_file` fails with `error_type: LoadError` and `details.reason` set
to the same access or parsing code, with pxr's message in `error`. A stage
that opens with missing sublayers, references or payloads loads, and lists
them in `warnings`. `simul usd validate` runs the same checks and exits 1
when the file is invalid.

### Editing

Edits go to the stage's edit target, its root layer, and are all or nothing:
each call checks the whole request first, then writes, and if a write still
fails it restores the specs it touched. A call that reports failure has
changed nothing.

Edit errors use the same types as the lookups above. Invalid input is a
`ValidationError` whose `details.code` says which check failed
(`InvalidPrimPath`, `UnknownPrimType`, `InvalidAttribute`,
`InvalidTimeCode`). A missing stage or prim is a `NotFoundError`. A write
that fails after the checks passed is an
`EngineError` with `details.rolled_back: true`. A refusal that an explicit flag
overrides is a `RefusedOperation` naming the flag in `details.override`, and
an edit USD cannot express is an `UnsupportedOperation`.

- **Paths** must be absolute prim paths (`/World/Box`). A relative path, a
  property path (`/World/Box.size`), a trailing `/` or an invalid prim name
  fails with `details.code: InvalidPrimPath`. The message names the bad
  element and suggests the fix (`details.suggestion`). A missing prim names
  the nearest prim that exists.
- **`create_prim`** takes only concrete types registered with USD (`Xform`,
  `Scope`, `Mesh`, `Cube`, `PhysicsRevoluteJoint`, ...). An unknown, abstract
  or API schema name fails with `details.code: UnknownPrimType` and close
  matches in `details.suggestions`. A path that already holds a prim is
  refused unless `redefine_existing=true`, which changes its type and keeps
  its children, properties and composition arcs.
- **Attributes** are checked against the prim's existing property or its
  schema definition: the value must fit the attribute's type, and a token
  must be one of its `allowedTokens`. An existing relationship takes a target
  path or a list of them. The keys `active`, `instanceable` and `kind` set
  prim metadata through USD's API; other metadata keys are refused rather
  than written as attributes. One error (`details.code: InvalidAttribute`)
  lists every rejected key with its reason in `details.problems`.
- **Values** are converted to the attribute's type: JSON numbers widen
  (`45` into a `float`, ints into a `double3`), an integer type refuses a
  fraction, vectors are lists (`[x, y, z]`), quaternions are `[w, x, y, z]`
  with the real part first, matrices are a list of rows or the flat
  row-major list, assets are path strings, and arrays are lists of their
  element's form.
- **Transform ops**: a new `xformOp:<type>[:<suffix>]` (`translate`,
  `scale`, `rotateX`..`rotateZYX`, `orient`, `transform`) is created with the
  op's value type (`double3` translate, `float3` scale, `quatf` orient,
  `matrix4d` transform) and added to `xformOpOrder`, so it takes effect. An
  existing op that is missing from the order is added to it. Pass
  `xformOpOrder` in the same call to set the order yourself; every entry
  must name an op the prim has or the call creates.
- **New custom attributes** take their type from `attribute_types`
  (`{"custom:offset": "double3"}`, which also picks a new op's precision), or
  else from the value: `bool`, `int`, `float`, `string`, `float2`..`float4`
  for 2 to 4 numbers, `int[]` / `float[]`, `float3[]` for a list of
  3-number lists, `string[]`.
- **Time samples**: `time_code` writes the values as samples at that time.
  An attribute that already has time samples is refused without it, since a
  default value is hidden by the samples at every frame. Uniform attributes
  (such as `xformOpOrder`), relationships and metadata cannot take a
  `time_code`. A sample is also refused when the attribute's animation comes
  from a weaker layer (a sublayer or a referenced asset): USD takes all of an
  attribute's samples from the strongest layer that has any, so one sample
  in the root layer would replace that whole animation.
- **`delete_prim`** refuses `/`, and the stage's default prim unless
  `allow_root_delete=true`. It also refuses a prim the delete cannot remove
  because opinions outside the root layer keep it alive: a sublayer defines
  it, or it comes from a reference or payload on an ancestor. Deactivate such
  a prim instead with `update_prim_attributes {"active": false}`.
- Prims inside an instanceable prim (instance proxies) cannot be authored;
  edits to them fail with `UnsupportedOperation`. Set `instanceable` to
  `false` on the instance first.

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
| Physics | `get_isaac_physics_scene`, `create_isaac_physics_scene`, `configure_isaac_physics_scene`, `add_isaac_rigid_body`, `add_isaac_collision`, `set_isaac_mass_properties`, `set_isaac_physics_material`, `list_isaac_physics_objects` |
| Joints and articulations | `list_isaac_articulations`, `set_isaac_joint_drive`, `set_isaac_joint_limits` |
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

- `get_isaac_prim_detail` reads one or more aspects of a prim per call:
  `info`, `transform`, `ancestors`, `relationships`, `variants`,
  `bounding_box`, `mesh`, `light`, `material`, `rigid_body`, `collision`,
  `joint`, `mass`, `animation`.
- Poses (the `transform` aspect, `info`'s transform, `get_isaac_camera_info`
  and the reply of `set_isaac_prim_transform`) are factored out of the
  composed matrix, so rotation and scale stay right under scaled parents and
  for matrix ops. Each carries `space` plus `local` and `world` blocks with
  `translation`, `rotation_quat` (`[w, x, y, z]`), `rotation_euler_xyz`
  (degrees, the order `set_isaac_prim_transform` takes) and `scale`;
  `sheared: true` marks a rotated prim under a non-uniform scale, where no
  rotation and per-axis scale reproduce the matrix exactly.
- `set_isaac_prim_transform` takes local values (relative to the parent). A
  prim whose stack holds a matrix op (`xformOp:transform`) is converted to
  the equivalent translate/orient/scale stack first; a sheared matrix is
  refused. An op with time samples is refused unless `time_code` names the
  sample to write, and an animated rotate op the XYZ rotation would replace
  is refused outright.
- `set_isaac_prim_attribute` converts the JSON value to the attribute's type:
  quaternions (`xformOp:orient`, `physics:localRot0`) as `[w, x, y, z]`,
  matrices as rows or a flat row-major list, arrays as lists.
- `query_isaac_typed_prims` resolves `type_name` through the schema registry
  (`UsdPhysics.RevoluteJoint`, `PhysicsRevoluteJoint`,
  `UsdPhysics.RigidBodyAPI`, any `pxr` module) and refuses a name that
  resolves to nothing. Values come back as JSON; arrays longer than
  `max_array_elements` (default 16) come back as a length summary, and
  non-finite floats as `null`.
- The `mass` and `rigid_body` aspects report an unset centre of mass (the
  schema's `-inf` fallback) as `null` with `center_of_mass_note`.
- `delete_isaac_prim`, `duplicate_isaac_prim` and `reparent_isaac_prim` edit
  the stage's edit target, not the root layer, and say what they did. A prim
  that comes from a referenced asset (every link of an imported robot) or is
  also defined in another layer can't be removed or moved from this stage:
  - `delete_isaac_prim` with `mode="auto"` (default) deactivates it instead
    (`action: "deactivated"`, with the `reason`; local overrides are kept).
    `mode="remove"` refuses, and `mode="deactivate"` always deactivates.
  - `duplicate_isaac_prim` copies the spec, arcs included, when the edit
    target defines the prim. Otherwise it copies the flattened composed
    result (`method: "flattened_copy"`), and instanceable prims inside stay
    instanced.
  - `reparent_isaac_prim` refuses unchanged.
  
  Instance proxies are refused by all three. `reparent_isaac_prim` keeps the
  world pose (`keep_world_transform=true`), retargets relationships and
  connections that point into the moved subtree, and uses Kit's undoable
  `MovePrim` when available. `undoable` reports whether the whole edit is on
  Kit's undo stack.
- The listings (`list_isaac_prims`, `search_isaac_prims`, `get_isaac_subtree`,
  `query_isaac_typed_prims`) descend into instanceable prims by default,
  where Isaac assets keep their meshes and colliders; entries there carry
  `instance_proxy: true` and are read-only. Pass
  `include_instance_proxies=false` to stop at the instance. Pass
  `include_inactive=true` to also see deactivated prims (`active: false`).
  `list_isaac_physics_objects` counts instanced colliders too, and
  `get_isaac_subtree` takes `max_depth=-1` as unlimited.
- `create_isaac_prim` refuses a type that isn't a concrete registered schema
  (`cube` gets a suggestion, `Cube`) and attributes the type's schema doesn't
  define; a value that can't be set removes the new prim again.
  `create_isaac_object` checks its vectors before creating anything. When a
  step fails, `left_on_stage` lists what the earlier steps created.
- With `security.allow_script_execution` off, `set_isaac_prim_attribute` and
  `create_isaac_prim` refuse to write an OmniGraph script node's
  `inputs:script*` or a script-capable `node:type`
  (`ScriptExecutionDisabled`), since that would run Python in Kit.
- `create_isaac_object` creates a prim with transform, rigid body, collider,
  mass and material in one call.
- `create_isaac_physics_scene` derives a new scene's gravity from the stage:
  Earth gravity along the negative up axis, in stage units (981 on a
  centimetre stage); `gravity_magnitude` is taken in m/s^2. It refuses a prim
  of another type at `prim_path` rather than retyping it.
  `get_isaac_physics_scene` reports gravity resolved the way the simulation
  reads it plus the PhysxSceneAPI settings authored on the scene (physics
  rate, solver type, iteration clamps, GPU dynamics, broadphase, CCD,
  stabilization, enhanced determinism); `configure_isaac_physics_scene`
  writes them.
- `add_isaac_collision` applies CollisionAPI to the geometry at and under
  `prim_path` (an asset root's meshes, not the root Xform, which would make no
  shape), puts `approximation` on meshes only, and lists the shapes.
  `set_isaac_physics_material` binds a MaterialAPI material to the colliders
  at or under `prim_path` with purpose `physics`, leaving the visual binding
  alone, and lists them in `bound_to`.
- The joint tools are USD-level (UsdPhysics DriveAPI, LimitAPI,
  ArticulationRootAPI). `list_isaac_articulations` orders each
  articulation's joints depth first from the root body; PhysX's DOF indices
  need not follow that order. Angular drive targets and revolute limits are in
  degrees. In the joint aspect, an infinite value ("no limit",
  "unbreakable") is null with a `<key>_note`, since JSON has no infinity.
- `get_isaac_ui_state` is a single snapshot of windows, focus, viewport,
  selection, timeline and stage; `get_isaac_ui_window` walks one window's
  widget tree.
- `execute_isaac_script` scripts should follow the scripting reference served
  as the `simul://isaac-sim/skills` resource. Isaac Sim 6.0 deprecates
  `isaacsim.core.{api,prims,utils}` and removes the `omni.isaac.*` shims.
  On Isaac Sim 5.1 with the bridge down, scripts over 32 KiB are refused
  (`ScriptTooLargeForSocket`): that socket runs each read as its own script.
- `open_isaac_stage`, `new_isaac_stage`, `read_isaac_aovs`,
  `capture_isaac_viewport` and `enable_isaac_extension` run with a budget of
  at least 110 s instead of the default 29 s. Any timeout reply says that the
  Kit work may still complete; check the state before retrying.
- A tool whose script shared Kit's stdout with another callback returns its
  result with the other text under `foreign_output` and a note in `warnings`.
- `claim_isaac_instance` and `set_active_isaac_instance` (with a `purpose`)
  return `ClaimStoreError` when the claim could not be saved. Calling
  `claim_isaac_instance` again with the same `agent_id` after a reconnect or
  a server restart takes the orphaned claim over; `release_isaac_instance`
  with that `agent_id` releases it. See
  [configuration.md](configuration.md#multi-agent-claims).

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

- A failed call names its cause with one `error_type` from the set the
  backends share:
  - `ConnectionError`: nothing answered at the Remote Control endpoint, or
    the editor closed the connection without answering (it may have crashed).
  - `TimeoutError`: no answer within `UNREAL__TIMEOUT`. A script or a change
    may still be running in the editor, and Remote Control cannot cancel it,
    so check the editor's state before retrying.
  - `EngineError`: Remote Control answered with HTTP 5xx.
  - `ValidationError`: Remote Control refused the request as made (HTTP 400),
    for example a read-only, `EditConst` or protected property, or a value of
    the wrong type.
  - `NotFoundError`: the object, property, function or console variable
    named does not exist, whether Remote Control (HTTP 404, or a 400 saying
    so) or a tool's editor script reports it.
  - `InstanceBusy`: the editor was saving or collecting garbage.
  - `AuthenticationError`: Remote Control refused this client (HTTP 401).

  `error` carries Remote Control's own message, and `details` carries the
  `status`, the `route` and `remote_control_message`. The domain types
  (`PIEActive`, `TransformNotApplied` and the like) are unchanged.
- `batch_unreal_operations` (registered with script execution allowed) sends
  each operation as `RequestId`, `URL`, `Verb`, `Body`, in that order, with a
  unique `RequestId`. Operations without one are numbered. Remote Control
  hands each `Body` to the last `RequestId` before it, so an operation
  without one, a repeated one or one written after its `Body` would run with
  another operation's body. A malformed batch is refused with
  `ValidationError` before anything is sent: at most 100 operations, `/remote/`
  routes only and no batch inside a batch, with `Verb` one of GET, PUT, POST,
  PATCH or DELETE. Every batch carries a `Passphrase` header, a placeholder
  when `UNREAL__PASSPHRASE` is unset. UE 5.7's batch route reads that header
  unchecked and aborts the editor without it; Remote Control ignores the value
  unless it enforces passphrases. A failed operation's result also has
  `error` and `error_type`.
- `set_unreal_render_settings` sets one existing numeric `r.*` or `sg.*`
  console variable and reads it back: `applied` is true only when it holds the
  value afterwards, with `value` and `previous_value` as Unreal formats them
  (`true`/`false` for a bool). A name Unreal has no variable for fails with
  `NotFoundError`, and nothing is run. A value Unreal did not take fails with
  `RenderSettingNotApplied`. That covers a read-only variable, such as
  `r.RayTracing`, which only startup configuration sets, and one Unreal limits
  or rounds. Post-process settings live on PostProcessVolume actors, not in
  console variables, and this tool does not change them.
- The actor tools take an actor's full object path, as `list_unreal_actors`
  reports it, in the level open in the editor. A path that names no live actor
  there fails with `NotFoundError`, `Actor not found: <path>` and why: nothing
  is at that path, the actor was deleted (Unreal keeps a deleted actor in memory, empty,
  until garbage collection), or it belongs to another map, loaded or not, or to
  the running game during PIE. The actor is looked up among the objects already
  loaded, so simul never loads a map that is not open: a lookup that loaded one
  used to edit that map unseen and leave it dirty for the next Save All.
- `set_unreal_actor_parent` attaches the actor to `parent_path`, or detaches it
  with `detach: true` and an empty `parent_path`; one of the two is required.
  The actor keeps its world location, rotation and scale, the change is one
  undoable transaction, and the parent is read back: `parent_path` is the
  parent afterwards (null once detached), with `previous_parent_path` and
  `changed`. Unreal refuses, without a reason, to attach an actor to itself or
  under one of its own children, or a Static actor under a parent that is not
  Static; those come back as `ValidationError` (itself, a cycle) or
  `InvalidState` (mobility, an actor without a root component) with the reason,
  before anything changes. An attachment Unreal still refuses is an
  `EngineError`, and its transaction is cancelled.
- `add_unreal_component` adds an instance component the way the editor's Add
  Component button does (`SubobjectDataSubsystem`): Remote Control refuses
  `AddComponentByClass` on UE 5.7. `component_class` is a native class name
  (`PointLightComponent`) or a class path (`/Script/Engine.PointLightComponent`,
  a Blueprint component's `/Game/BP_Lamp.BP_Lamp_C`); a path into a map's
  package is refused before anything loads. A scene component attaches under
  `parent_component` (by name) or the root component, and becomes the root of
  an actor without one; other components take no parent. The add and the name
  are one undoable transaction, and the reply is read back from the actor:
  `component_path`, `component_name`, `component_class`, `parent_component` and
  `is_root`. An abstract class (Unreal adds one and fires an engine ensure), a
  class that is not a component and a name already used inside the actor are
  refused first. Unreal adds its usual helpers too, such as a light's editor
  icon (a BillboardComponent).
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
- `enable_unreal_physics`, `set_unreal_physics_params` and
  `set_unreal_collision` edit the actor's primitive component (its root, else
  its only one) the way the Details panel does: in one undoable editor
  transaction that marks the map dirty, so Save All writes the edit and one
  Undo reverts that call alone. Each reply is the body read back:
  `physics_mode` (`simulated`, `kinematic`, `static` for a body that is not
  Movable, `none` without physics collision), mobility, collision mode and
  preset, `mass_kg` (the mass Unreal computes) and the mass override, gravity
  and damping, the attach parent and a `warning`.
  - `enable_unreal_physics` makes the body simulate (`enable` true), makes it
    kinematic (`kinematic` true: physics collision on, so simulated bodies hit
    it, but moved only by its transform) or turns physics off (`enable`
    false; collision and mobility kept). Unreal accepts the simulate flag on
    any body and reads it back as simulating even when the body never moves,
    so the tool checks first and refuses, with nothing changed, a body that
    cannot take the mode: `error_type: "IncompatibleBody"`, each reason in
    `details.problems` (`NOT_MOVABLE`, `NO_PHYSICS_COLLISION` for NoCollision
    or a query-only preset such as OverlapAll, `NO_GEOMETRY` for a
    StaticMeshActor without a mesh, `NO_SIMPLE_COLLISION`,
    `COMPLEX_AS_SIMPLE` for a simulated mesh that uses its render triangles
    as simple collision, `ATTACHED`). A Static or Stationary body is made
    Movable (`mobility_changed_from`) unless `make_movable` is false.
  - Unreal detaches a body from its parent when it starts simulating, keeping
    its world transform, and turning physics off does not reattach it. An
    attached body is refused unless `allow_detach` is true; it is then
    detached in the same transaction, so Undo reattaches it, and the reply
    names the parent in `detached_from`.
  - `set_unreal_physics_params` takes a mass from 0.001 kg (Unreal's least)
    to 1e9 kg and damping from 0 to 1e6, and reads every value back
    (`applied`, `params_set`; `PhysicsParamsNotApplied` names one that reads
    back different). A Static body keeps the values but has no mass
    (`mass_kg` 0) and never simulates until it is Movable; the `warning`
    says so. With no parameter it only reads the body.
  - `set_unreal_collision` checks a preset before writing (an unknown one
    changes nothing) and, when it sets `collision_enabled` on a static mesh,
    clears the mesh's Use Default Collision flag as a preset does; with the
    flag left set, Unreal put the mesh's own collision back in the running
    simulation and after the map reloaded.
  - A path in a running simulation's world (`UEDPIE_<n>_<map>`) edits that
    play copy, which is neither undoable nor saved (`world: "simulation"`).
- USD import and export need the optional **USDImporter** plugin enabled when
  the editor starts. They use `AssetImportTask` and `LevelExporterUSD`;
  `get_unreal_interchange_info` reports whether the plugin is available.
  Import options accept booleans `import_actors`, `import_geometry`,
  `import_materials`, `import_lights`, `import_cameras`; export options accept
  booleans `export_actor_folders`, `export_sublayers`.
  - `import_unreal_usd` files the assets under
    `<destination_path>/<file name>` (the name as Unreal names assets: `my
    robot.usda` becomes `my_robot`) and spawns the actors in the level being
    edited. Without `replace_existing` (the default) an import into a folder
    that already holds assets is refused with `InvalidState` and nothing is
    imported, and existing actors are kept: the new ones are added beside
    them. Unreal's USD importer otherwise replaces both, whatever
    `AssetImportTask.replace_existing` says: a re-import replaced a mesh
    customized since (its collision and lightmap resolution were lost) and
    deleted the earlier import's actors. With `replace_existing` true the
    folder's assets and the level's actors from an earlier import of the file
    (the top-level actor named after the file) are replaced;
    `removed_actor_paths` lists the actors that went.
  - The import options are set without saving them. Unreal's
    UsdStageImportOptions saves its settings to the project's
    `Saved/Config/<Platform>Editor/EditorPerProjectUserSettings.ini` whenever
    a change is notified, and earlier simul versions notified it, so one
    import with `import_actors` false left `bImportActors=False` there for
    every later import and the editor's own **Import into Level** dialog.
    simul now sets `import_actors`, `import_geometry` and `import_materials`
    on every import (true unless given) and, when the project's saved
    settings differ from Unreal's defaults, says so in `warnings` and leaves
    them alone: nothing tells a value simul wrote from one the user chose.
    Reset them in the editor's USD import dialog.
  - `export_unreal_usd` exports actors of the level being edited; a path in
    another map is refused without loading it. `actors_exported` counts the
    actors the written stage has a prim for, read back from the file
    (`exported_prims` gives each one's prim), since Unreal's exporter leaves
    out editor helpers such as WorldSettings or PlayerStart without an error;
    they are listed in `skipped_actors`, and an export that wrote none of the
    actors fails with `EngineError`. Mesh and material files are written next
    to the output file.
  - Both wait `unreal.import_export_timeout` (default 300 s,
    `UNREAL__IMPORT_EXPORT_TIMEOUT`) instead of `unreal.timeout`: the first
    USD export of an editor session loads Unreal's USD Python modules and
    takes 25-40 s on UE 5.7. When the time runs out the error is
    `TimeoutError` and says the import or export may still complete, since
    Remote Control cannot stop a script it has stopped waiting for; check
    before retrying.
  - Neither leaves an editor object behind in the Python namespace Remote
    Control runs its scripts in. The export's AssetExportTask used to keep
    the editor world referenced there, and the next level load failed the
    editor's world leak check and took the editor down.
- `assign_unreal_material` assigns the material in one undoable transaction
  that records the component first, which marks the level dirty (in a World
  Partition level, the actor's package), so saving keeps the assignment and
  Undo reverts it; the slot is read back, and `previous_material_path` is what
  it rendered with before. The actor must be in the level being edited: a path
  in another map is refused without loading that map, as is a map path given
  as the material.
- `create_unreal_material_instance` writes under the project's `/Game` folder:
  `save_path`, else next to a parent in `/Game`, else `/Game/Materials`. It
  used to save an instance of an `/Engine` parent into the engine install
  (`Engine/Content/BasicShapes`), which every project using that engine
  shares. A folder outside `/Game` (`/Engine`, a plugin's content) is refused
  with `SandboxError` unless `allow_engine_content` is true, and
  `set_unreal_material_params` refuses an instance outside `/Game` the same
  way. A folder with an empty, `.` or `..` name, or with characters Unreal
  does not allow, is a `ValidationError`: Unreal does not resolve `..` and
  turned `/Game/../Engine` into a folder named `__`.
- `search_unreal_assets` returns at most `max_results` assets, from 1 to 1000:
  Remote Control sizes its result array to the limit it is sent, and -1
  crashed the editor.
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
    physics settings, visibility, materials, parenting, components) keeps
    working and changes the editor world, which the running simulation does
    not see: it takes effect the next time the simulation starts, and a
    physics or collision edit says so in its `warning`. A physics or collision
    edit given the actor's path in the simulation's world (`UEDPIE_<n>_<map>`)
    changes the running copy instead, which is neither undoable nor saved;
    every other actor tool finds no actor at such a path.
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
| Objects | `create_blender_object`, `delete_blender_object`, `get_blender_object_info`, `set_blender_object_transform`, `set_blender_object_parent`, `clear_blender_object_parent`, `create_blender_mesh_from_data`, `clean_blender_mesh`, `get_blender_mesh_info`, `add_blender_modifier`, `apply_blender_modifier`, `remove_blender_modifier` |
| Measurement | `get_blender_bounding_box`, `check_blender_object_bounds`, `get_blender_distance_between` |
| Materials and lights | `get_blender_material_info`, `assign_blender_material`, `set_blender_light_params` |
| Physics | `setup_blender_rigid_body`, `add_blender_rigid_body_constraint`, `get_blender_constraint_info`, `add_blender_force_field`, `get_blender_force_field_info`, `get_blender_physics_state`, `bake_blender_simulation`, `free_blender_bake`, `get_blender_rigid_body_world`, `set_blender_rigid_body_world` |
| Animation | `get_blender_frame`, `set_blender_frame`, `set_blender_frame_range`, `play_blender_animation`, `insert_blender_keyframe`, `delete_blender_keyframe`, `get_blender_keyframes`, `get_blender_object_trajectory` |
| Camera and viewport | `get_blender_camera_info`, `set_blender_camera_view`, `get_blender_viewport_info`, `focus_blender_on_object`, `capture_blender_viewport`, `capture_blender_viewport_sequence` |
| UI and attachment | `control_blender_ui`, `attach_blender_window` |
| SimReady | `get_simready_metadata`, `apply_simready_metadata`, `setup_simready_hierarchy`, `validate_simready_compliance`, `export_simready_usd` |
| Scripting | `execute_blender_script` |

Behaviour worth knowing:

- In attached mode each call that changes the document is one Blender undo
  step named `simul: <tool>`, and marks the file modified: the user's Ctrl+Z
  takes back the last agent operation only, and quitting asks to save. Reads,
  captures, saves, exports, playback and editor-only `control_blender_ui`
  actions record no step. Details:
  [blender-attachment.md](blender-attachment.md#undo-and-unsaved-changes).
- `open_blender_file` refuses with `UnsavedChangesError` while the open file has
  unsaved changes; save them with `save_blender_file`, or pass
  `discard_unsaved: true`. Headless (embedded) Blender does not track unsaved
  changes, so there it opens without the check.
- `get_blender_object_info` reports an object's own transform (`location`,
  `rotation_euler`, `scale`: relative to its parent and before constraints, as
  Blender's Location, Rotation and Scale fields show them) and where it is in
  the scene (`world_location`, `world_rotation_euler`,
  `world_rotation_quaternion`, `world_scale`, and the exact `matrix_world`).
  A turned child of a parent with non-uniform scale has a sheared world
  matrix that no rotation and scale rebuild; the world rotation is then the
  rotation nearest it (its polar decomposition), the parent's rotation times
  the child's, where Blender's `decompose()` and `to_euler()` were 0.19 and
  0.15 rad off for a glTF link under a parent scaled 0.5 on Z. It also
  reports `rotation_mode`, the rotation as Blender stores it in that mode
  (`rotation_in_mode`) and as a quaternion (`rotation_quaternion`). Euler
  angles are in XYZ order throughout, whatever the object's rotation mode.
- `set_blender_object_transform` writes the object's own transform:
  `location` relative to its parent, and `rotation_euler` (XYZ) or
  `rotation_quaternion` (`[w, x, y, z]`, normalized), converted to the
  object's rotation mode, which is kept. Blender reads only the property of
  that mode, so a plain `rotation_euler` write left a glTF import's
  quaternion nodes unturned. `control_blender_ui`'s `set_property` of
  `rotation_euler` converts the same way.
- `set_blender_object_parent` keeps the child's world transform, as Blender's
  Parent > Object (Keep Transform) does: the parent inverse cancels the
  parent's transform, and a child without a previous parent keeps its own
  values. `keep_transform: false` clears the parent inverse instead, so the
  child's values become relative to the parent and it moves. A parent that is
  the child itself or one of its descendants is refused (Blender ignores such
  a parent without an error).
- Reads of world-space transforms (object info, distance, bounds, bounding
  box, parenting) evaluate the depsgraph first: in embedded (bpy module) mode
  Blender leaves a transform written by an earlier call out of `matrix_world`
  until then. An add-on built before `rotation_quaternion` and
  `keep_transform` still answers calls that leave them at their defaults, with
  its own behaviour; the newer reply fields then come back `null`.
- `create_blender_mesh_from_data` checks everything before it creates
  anything: each vertex is exactly `[x, y, z]` finite numbers, each index is
  an integer within the vertex list, no face repeats a vertex or the vertex
  set of another face (in any order), no edge joins a vertex to itself or
  repeats another edge, and `collection_name` names a collection in the
  current scene that is not linked from a library. Blender's `from_pydata`
  flattens the vertex list, so a vertex of 2 or 4 numbers used to shift every
  later vertex, and faces such as `[0,1,2,1]` with duplicates froze Blender
  5.0.1's Decimate. The new mesh is then validated (`mesh.validate()`), and
  when any step fails the mesh and object made so far are removed: an orphan
  object, or one in another scene, made whole-scene exports fail. `warnings`
  reports a renamed object (`probe.001`), a collection the view layer
  excludes, and any change validation made.
- `clean_blender_mesh` repairs a mesh made elsewhere (an import, a script).
  It always validates the mesh, removing faces that repeat a vertex and
  duplicate faces and edges, then optionally merges vertices within
  `merge_distance`, dissolves edges and faces smaller than
  `dissolve_degenerate_distance`, and recalculates normals. It reports the
  counts before and after; mesh data shared by several objects changes for
  all of them.
- `add_blender_modifier` checks each `params` key against the modifier
  type's RNA: an unknown name is refused with the closest names, an enum takes
  one of its items (matched case-insensitively; an enum flag such as
  Decimate's `delimit` takes a list), a number must lie within the setting's
  hard limits, which Blender would otherwise clamp silently, and an object,
  collection or other datablock setting takes its name, such as
  `{"object": "Cutter"}` for BOOLEAN. Blender silently drops a datablock its
  poll refuses (a camera for a Boolean), which is reported as an error. The
  reply's `params` echo the stored settings. A refused call removes the
  modifier again, and an object without a modifier stack (camera, light,
  empty) is refused up front. DECIMATE is refused on a mesh whose validation
  would change it, as Decimate's collapse can loop forever there; the check
  validates a throwaway copy (0.16 s for a million faces).
- `apply_blender_modifier` is Blender's Apply: the modifier's result replaces
  the mesh data. A modifier below others is applied to the base mesh without
  them, with a warning; mesh data shared with other objects is copied first
  (`made_single_user`). A modifier disabled in the viewport is refused, as
  Blender 5.0.1 drops it unapplied. `remove_blender_modifier` removes one
  without applying it. Both, and `clean_blender_mesh`, need the object in
  Object Mode.
- `get_blender_mesh_info` reports the stored mesh's counts and, as
  `evaluated_*`, the counts after the modifiers enabled in the viewport, which
  is the geometry an export writes (94 triangles where the base cube has 6
  faces after Bevel, Decimate, Triangulate and Weld). A mesh in Edit Mode is
  written back from the edit mesh first, so the counts include unsaved edits.
  USD export evaluates modifiers for render, so `warnings` names modifiers
  whose render settings differ (`show_render`, Subdivision `render_levels`).
  An object Blender does not evaluate (outside the view layer) has null
  evaluated counts.
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
- `export_blender_file` writes `file_path` exactly. The path must end in the
  format's extension (`.obj`, `.fbx`, `.glb`/`.gltf`, `.usd`/`.usda`/`.usdc`/`.usdz`,
  `.stl`, `.ply`); anything else is refused before Blender runs, as Blender's glTF
  exporter would otherwise swap the extension (a `model.gltf` exported as GLB
  used to overwrite `model.glb`). For GLTF the extension picks the layout: `.glb`
  is GLB, `.gltf` is `GLTF_SEPARATE` (a `.bin` and textures beside it);
  `gltf_format: "GLTF_EMBEDDED"` writes one `.gltf` with its data inline. The
  reply's `files_written` lists every file the export created or changed in
  that folder (and its `textures/` or newly created subfolders), `file_path`
  first. An existing `.bin` or `.mtl` beside the file that the file being
  replaced does not refer to is not overwritten unless
  `overwrite_companion_files: true`; texture files a `GLTF_SEPARATE` export
  writes replace files of the same name. Missing folders are created for every
  format.
- What to export: the whole scene by default, `selected_only` (an error when
  nothing is selected, instead of an empty file), `object_names` or
  `collection_name` (selected for the export, the previous selection restored
  afterwards; a hidden object or one outside the view layer is refused, since
  the exporter would leave it out). Children are not added to `object_names`.
- Defaults that changed: `apply_modifiers` is on for every format (glTF used to
  export the cage without its modifiers; USD always applies them, so `false` is
  an error there), and `export_animation` is on for glTF, FBX and USD, also for
  USD, whose exporter leaves animation out by default. `frame_start`/`frame_end`
  set the scene's frame range for the export and restore it. FBX, OBJ, STL and
  PLY take `axis_forward`/`axis_up` (together, on different axes) and
  `global_scale`; USD takes `usd_up_axis` (`Y` rotates the scene to Y-up),
  `meters_per_unit` (Blender 4.4+; the root is scaled so objects keep their
  size) and `root_prim_path` (`""` for none). An option that does not apply to
  the format, or that this Blender's exporter lacks, is an error rather than
  ignored. `exporter_options` reports the operator properties that were set.
  Edit, Pose and paint modes are left for the export, so unsaved mesh edits are
  included, and entered again afterwards.
- `import_blender_file` takes `scale`, about the world origin: OBJ, FBX, STL
  and PLY pass it to Blender's importer; glTF (no importer option) and USD
  (Blender 5.0.1's importer ignored it for USD files Blender wrote) scale the
  imported root objects, and `warning` names roots whose transform is animated.
  `save_blender_file` accepts only `.blend` paths, since the saved file becomes
  the open document.
- Rigid bodies simulate only inside the rigid-body world's cache range
  (`rigid_body_cache` in the replies of `get_blender_rigid_body_world`,
  `get_blender_physics_state`, `bake_blender_simulation`, `free_blender_bake`
  and `set_blender_frame_range`): past its end Blender holds the bodies at the
  end frame's pose. Blender makes a new world's cache 1-250 whatever the scene
  range, so a world simul adds covers the scene's frame range instead;
  `set_blender_frame_range` moves an unbaked cache with the scene range and
  warns when a baked one leaves part of it out; and `free_blender_bake` gives a
  freed cache the scene range back. `set_blender_rigid_body_world` sets the
  range explicitly, along with `enabled`, the scene gravity (shared with cloth,
  soft bodies and particles), substeps, solver iterations, time scale and split
  impulse, adding the world when there is none. A baked cache's range changes
  only after it is freed, and changed settings reach a baked simulation only
  after a new bake; the reply's `warnings` says so.
- `bake_blender_simulation` replaces an earlier bake. Blender skips a cache that
  is already baked, so a second bake used to report the new range over the old
  frames; the bake is freed first and the simulation restarts at `frame_start`
  from the bodies' current transforms. `frame_start` must come before
  `frame_end`: an inverted range baked nothing and left new bodies unable to
  move. `scope: "rigid_body"` (the default) bakes only the rigid-body cache, by
  stepping each frame and then marking that cache baked, because Blender 5.0.1's
  `ptcache.bake` given that cache baked every cache in the scene (a cloth over
  its own 250 frames) while `ptcache.free_bake` freed only the one.
  `scope: "all"` frees and bakes every point cache, as Bake All Dynamics does:
  rigid bodies, cloth, soft bodies and dynamic paint over the requested range,
  particle systems over the range their emission settings give.
  `free_blender_bake` takes the same scope and lists what it `freed`. A disabled
  rigid body world is refused rather than marked baked with no frames. The bake
  runs on Blender's main thread, stepping every frame, and the scene returns to
  its current frame afterwards.
- `get_blender_physics_state` steps an unbaked simulation from the cache start
  to the current frame before reading the pose (`step_simulation`, default
  true, at most 400 000 frames x bodies), as playback would: an unbaked
  simulation advances one frame at a time, so after `set_blender_frame` jumped
  from 1 to 48 a falling body still sat at its starting pose. Frames Blender has
  cached are read, not simulated again. `pose_source` says where the pose comes
  from: `simulation` or `baked`; `before_cache` (the starting pose);
  `past_cache_end` (held at the cache's last frame); `world_disabled`; `unknown`
  when it was not stepped; or `evaluated` for an object nothing simulates (no
  rigid body, passive, animated or disabled). `simulated_for_frame` says whether
  the pose is the simulation's result for the frame, and `warning` explains
  when it is not. An add-on built before this report answers without these
  fields, and the server returns them as `null`.
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
  and return its path. Anything the script raises, `SystemExit` from
  `sys.exit()` and `KeyboardInterrupt` included, comes back in `error`; it
  never ends Blender.

## Server

| Tool | Purpose |
|---|---|
| `get_tool_usage_stats` | Per-tool call counts, success rates and durations from the persistent usage log. Clearing it is operator-only: `simul stats --reset` |

## MCP resources

| URI | Contents |
|---|---|
| `simul://isaac-sim/skills` | Isaac Sim 5.1 / 6.0 scripting patterns for `execute_isaac_script` |
| `simul://isaac-sim/api/{core,sensors,physics,replicator,robots,rendering,assets}` | Isaac Sim API references by area |
