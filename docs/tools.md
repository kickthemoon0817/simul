# MCP tool catalog

The tools a server registers depend on which backends are available and
selected (`--backends`), on `--unreal-tools` / `--blender-tools`, and on
`security.allow_script_execution`. To see exactly what your server exposes:

```bash
simul tools            # add --json for descriptions and input schemas
```

Every tool returns a JSON object. A failure is sent with MCP `isError: true`
and carries `success: false`, an `error_type` and an `error` message; see
[Errors](#errors) for the types. Parameter bounds and choices are in each
tool's input schema (`minimum`, `maximum`, `enum`), and a value outside them
is refused with `ValidationError` before the tool runs.

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

No running application needed. File arguments must be inside the sandbox,
and so must every sublayer, reference, payload, clip and asset path the file
draws on: one outside it fails the call with `SandboxError` listing them (see
[configuration.md](configuration.md#file-sandbox)).

| Group | Tools |
|---|---|
| Load and validate | `load_usd_file`, `validate_usd_file` |
| Query | `get_prim_info`, `get_prim_subtree`, `search_prims`, `summarize_scene`, `get_mesh_info`, `get_bounding_box` |
| Edit | `create_prim`, `delete_prim`, `update_prim_attributes` |
| Stages | `list_usd_stages`, `save_usd_stage`, `export_usd_stage`, `reload_usd_stage`, `unload_usd_stage` |

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
- **Meshes.** `get_mesh_info` is computed with numpy over whole arrays (a
  million-vertex mesh in about 0.3 s). It reports normals from
  `primvars:normals` or `normals` (`normals_source`), `uv_sets`, and the
  materials `UsdShade.MaterialBindingAPI` resolves: the mesh's, then each
  `materialBind` subset's (also on the subset entry), plus a
  `physics_material` that carries `PhysicsMaterialAPI`. `surface_area`, `volume` and
  `bbox` are in the mesh's own space; `volume` is null unless the mesh is
  closed, `surface_area` null for invalid topology, and `bbox` null without
  points. Face areas use Newell's method, exact for planar faces, concave
  ones included. Faces listed in `holeIndices` add no area and leave their
  edges open. `time_code` reads a deforming mesh, and `points_time_samples`
  says whether it deforms. A mesh whose points or topology exist only as
  time samples has nothing to measure at the default time: it comes back
  `measured: false`, with null topology and geometry (not invalid), its
  `sampled_time_range` and a `note`. Subset indices are summarised
  (`index_count` plus the first 16).
- **Summaries.** `summarize_scene` reports:
  - `scene_size_meters` next to `scene_size`;
  - `has_animation` from an authored time range or time samples spanning
    more than one time (`time_sample_range`), with `duration_seconds` (time
    codes divided by `timeCodesPerSecond`);
  - `warnings` for layers whose authored `upAxis` or `metersPerUnit` differs
    from the stage's (USD rescales and reorients nothing), one entry per
    mismatch and value with `layer_count` and up to 5 example `layers`;
    at most 10 groups, then a `MoreUnitMismatches` count;
  - `materials`, `root_prims` and their counts;
  - `(typeless)` as the type key of untyped prims.

  Instanced meshes count once per instance and are measured once. Mesh
  `total_surface_area` and `total_volume` sum each mesh in its own space,
  before transforms. A deforming mesh with time-sampled points is measured
  at its first sample and counted in `meshes_measured_at_first_sample`.
  `layer_count` (here and in `load_usd_file`) counts the file's layer stack,
  not the session layer, and `load_usd_file`'s `has_animation` follows the
  same rule as the summary.
- **JSON values.** Vectors are lists, quaternions `[w, x, y, z]`, matrices
  lists of rows, asset paths their authored string.
- **Lookups.** A `stage_id` that is not loaded, or a prim path with no prim,
  is a `NotFoundError`; a relative, malformed or property path is a
  `ValidationError` with `details.code` `InvalidPrimPath`, as the edit tools
  report it.

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
when the file is invalid. A relative `file_path` resolves against
`security.relative_path_base` (default: the server's working directory), and
the result's `file_path` is the resolved path. A URL needs an asset resolver
for its scheme, which usd-core does not ship, so `omniverse://` fails with
`UnsupportedURL` (see [configuration.md](configuration.md#file-sandbox)).

Every successful response carries `success: true`, like the other backends.
The pxr work of each call runs on a worker thread, so a slow call does not
hold up the server. Reads run concurrently, one at a time per stage, and an
edit (`create_prim`, `update_prim_attributes`, `delete_prim`), a
`load_usd_file` and a `reload_usd_stage` run alone, because pxr does not
allow a write alongside reads of the same layers (a load or reload may reread
a shared file in place). `save_usd_stage`, `export_usd_stage` and
`unload_usd_stage` hold their stage like a read. A
call that cannot start within `server.timeout` seconds (default 30) fails
with `StageBusy`; retry it. Python cannot stop pxr mid-call, so a call the
client cancels or times out still finishes in the background and holds its
stage until it does. Calls on other stages and `validate_usd_file` carry on
meanwhile.

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

### Stages, saving and sessions

Edits live in memory until they are written; nothing is saved on its own,
and unsaved edits are lost when the server exits.

- **Each load is its own stage.** `load_usd_file` opens the file's root
  layer as an instance no other stage shares, and every edit goes to that
  root layer. Sublayers, references and payloads are shared, read-only
  layers, so an unsaved edit in one stage never shows up in another stage
  that loads or references the same file. Loading a file again returns the
  same `stage_id` while that stage has no unsaved edits and none of its
  files changed on disk. Otherwise the load opens a fresh stage from disk
  (rereading changed sublayers and references first), and the older stage
  keeps its id and its edits.
- **Rereads stay in the sandbox.** A changed sublayer or referenced file is
  shared, and rereading it updates every stage that composes it, including
  other sessions' stages. So its new content is read into a throwaway layer
  and held to the composition sandbox first. If it now reaches outside the
  sandbox, nothing is reread, the stages keep the old content, and the load
  or reload fails with `SandboxError` naming the file. A stage that reaches
  outside after a reread anyway (a file swapped between the check and the
  reread) is unloaded. A load or reload runs alone, after the running calls
  finish, since rereading a shared file changes stages other calls may be
  reading.
- **Stages belong to the MCP session that loaded them.** Another session
  gets `NotFoundError` for the id and cannot read or edit the stage. Stage
  ids are random, not derived from the path. A stdio server has one
  session. Over HTTP, each initialize-handshake connection is its own
  session, and unauthenticated sessionless (2026-07-28) requests all share
  one anonymous scope (see AGENTS.md, Runtime Notes). A session can hold
  up to 32 stages; unload stages to load more.
- **When a session ends, its stages are unloaded.** An initialize-handshake
  connection that closes takes its stages with it, unsaved edits included
  (the server logs a warning naming the stages that had unsaved edits).
  Save or export before disconnecting. Stages of a session whose end the
  server cannot observe (stdio's single session, the anonymous scope) stay
  until they are unloaded or the server exits.
- **`list_usd_stages`** shows the session's stages, whether each has
  unsaved edits, and which of its files changed on disk since it read them.
- **`save_usd_stage`** writes the root layer back to its own file, which
  must be writable inside the sandbox. It refuses a `.usdz` root, which is a
  read-only package; export it instead. It also refuses a file that changed
  on disk after the stage read it, unless `overwrite_external_changes=true`.
  `files_written` lists what was written; a stage with no unsaved edits
  writes nothing.
- **`export_usd_stage`** writes a new `.usda`, `.usdc`, `.usd` or `.usdz`
  file (sandboxed, and refused if it exists unless `overwrite=true`). By
  default it writes the root layer with its composition arcs, and
  re-anchors relative asset paths (sublayers, references, textures) so they
  resolve from the new location. `flatten=true` writes the composed stage
  as one layer. A `.usdz` is packaged with the files it depends on. The
  stage stays bound to its own file and keeps its unsaved state.
- **`reload_usd_stage`** rereads the root file and any changed sublayer or
  referenced file, each held to the sandbox first as above.
  **`unload_usd_stage`** closes the stage. Both refuse a stage with unsaved
  edits unless `discard_unsaved=true`.

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
  
  Instance proxies are refused by all three. `delete_isaac_prim` also
  refuses prims marked `no_delete` and prims defined only in the session
  layer (Kit's `OmniverseKit_*` viewport cameras), which a deactivation would
  hide by writing into the scene's layer. `reparent_isaac_prim` keeps the
  world pose (`keep_world_transform=true`) and retargets relationships and
  connections that point into the moved subtree. Removal and moves go
  through Kit's undoable `DeletePrims` / `MovePrim` (`destructive=false`)
  when Kit is there; when Kit declines one, the tool reports
  `KIT_DECLINED` and changes nothing rather than editing the layer itself.
  `undoable` reports whether the whole edit is on Kit's undo stack.
  Refusals are typed: `error_type` is `NotFoundError`, `ValidationError` or
  `InvalidState`, and `details.code` names the case (`INSTANCE_PROXY`,
  `DEFINED_ELSEWHERE`, `DESTINATION_EXISTS`, `UNKNOWN_PRIM_TYPE`, ...).
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
  define; a value that can't be set removes the new prim again, with any
  ancestors the call created (`ancestors_created` lists them on success).
  `create_isaac_object` checks its vectors before creating anything. When a
  step fails, `left_on_stage` lists what the earlier steps created.
- With `security.allow_script_execution` off, the prim tools refuse what
  would plant code Kit runs (`ScriptExecutionDisabled`): code-carrying
  attributes (`inputs:script*`, `inputs:codeStr`, Replicator's
  `inputs:augmentationScript` / `inputs:conditionScript`), a `node:type` that
  runs code (ScriptNode, omni.warp's WarpKernel, Replicator's OgnAugment and
  condition-script nodes), any write to such a node, creating one, and
  duplicating a subtree that holds one. They use the same predicates as the
  graph tools (`graph.py`).
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
- `set_isaac_material_property` types a new input the way the shader declares
  it (texture paths become assets, flags bools, enums ints) and refuses a
  name that is not an OmniPBR or UsdPreviewSurface input, suggesting the
  closest ones (`allow_unknown_input=true` overrides).
- `import_isaac_asset` and `add_isaac_reference` reference USD layers only
  (convert OBJ, FBX and glTF first). They reference the asset's first root
  prim when it has no defaultPrim, and undo a reference that composes
  nothing. An import never merges into an existing prim (it goes to
  `<target_path>_N`). It corrects an authored metersPerUnit / upAxis
  mismatch by making `target_path` an Xform with
  `xformOp:scale:unitsResolve` and `xformOp:rotateX:unitsResolve` and
  referencing the asset on its child `<target_path>/<asset root name>`
  (`asset_root_path`), so the asset root's own transform is converted too.
  A failed import removes every prim it created.
- `save_isaac_stage` checks an in-place save's files against the write
  sandbox, needs `overwrite=true` for URL targets, and refuses while the
  timeline is playing or paused unless `allow_while_playing=true`.
- `set_isaac_graph_node_values` fails when any attribute fails, still
  reporting what was set. With script execution disabled, the graph, import
  and extension tools refuse code-running nodes (ScriptNode,
  omni.warp.WarpKernel, Replicator script nodes), code attributes,
  connections into them and the script extensions
  ([configuration.md](configuration.md#script-execution)).
  `disable_isaac_extension` fails when the extension stays enabled.
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
- Pose and velocity readers (`get_isaac_prim_detail` transform, info,
  bounding_box and rigid_body aspects, `find_isaac_prims_in_area`,
  `get_isaac_camera_info`) read USD. Under GPU physics Isaac Sim's
  SimulationManager stops PhysX writing to USD (`/physics/updateToUsd` and
  `/physics/updateVelocitiesToUsd` false, `omni.physx.fabric` on). While the
  simulation runs, those results then carry `stale_under_fabric: true`,
  `usd_readback` and a warning: the values are from before play. Read the
  simulated state with `execute_isaac_script` through a physics view (see
  [isaac-sim.md](isaac-sim.md#reading-simulated-state)).
- `step_isaac_simulation` advances N timeline frames. Each frame runs
  `frame_dt / physics_dt` physics steps, reported under `physics`. The call's
  budget grows with N up to 110 s, and a step that runs out stops, pauses
  and says how far it got. The timeline stays paused, and with
  `/physics/resetOnStop` on (Isaac Sim's default) the next stop or reset
  restores simulated prims to their pre-play state. Edits made in between
  (`set_isaac_prim_transform`, `set_isaac_prim_attribute`,
  `create_isaac_object`) are reverted with them, so those tools add a
  warning while paused.
- `raycast_isaac_scene` normalizes the direction and returns both the
  collision shape hit (`collision_path`) and its body (`rigid_body_path`).
  PhysX holds colliders only once the simulation has started, so a miss on a
  stopped timeline is `SimulationNotRunning`, not "no hit".
- `focus_isaac_viewport` frames the prim the way Edit > Focus Prim does and
  restores the selection. `focused` is true only when the camera moved or
  the framing call said so.
- `read_isaac_aovs` handles annotators that return `{data, info}`
  (segmentation, bounding boxes, pointcloud): it reports the array's stats,
  plus per-field ranges and counts for bounding boxes, and an `info`
  summary such as `idToLabels`.
- `get_isaac_logs` reads this instance's log (Kit's `/log/file`). When Kit
  does not name it, the tool falls back to the newest Kit log and warns that
  it may be another process's.
- `set_isaac_carb_settings` refuses the transport namespaces (bridge,
  python_server, 5.x VS Code). It refuses `/persistent/` keys, which Kit saves
  for later sessions, unless `allow_persistent=true`. `get_isaac_carb_settings`
  reads `auth_token` values back as `***`.
- `ping_isaac` does not wait for a call already running on the instance and
  reports `busy` when one is. A call the client cancels keeps the instance
  until Isaac Sim has finished it (the bridge is asked to interrupt it first);
  see [isaac-sim.md](isaac-sim.md#transports).
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
| Actors | `list_unreal_actors`, `get_unreal_actor_info`, `list_unreal_actor_components`, `describe_unreal_object`, `get_unreal_actor_by_semantic_label`, `get_unreal_actor_thumbnail`, `spawn_unreal_actor`, `delete_unreal_actor`, `set_unreal_actor_transform`, `set_unreal_actor_property`, `set_unreal_actor_visibility`, `set_unreal_actor_parent`, `add_unreal_component`, `focus_unreal_on_actor` |
| Assets | `search_unreal_assets`, `import_unreal_asset`, `save_unreal_assets` |
| Camera, lights, rendering | `set_unreal_camera_view`, `set_unreal_light_params`, `set_unreal_render_settings` |
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
- `list_unreal_actor_components` lists an actor's components, editor-only
  ones included: class, root and editor-only flags, and for scene components
  the component they are attached to (its path when it belongs to another
  actor), world location, mobility and visibility; primitive components add
  their world bounds (box center and half-size, sphere radius, in cm).
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
- A change Unreal took without an error, but which the read-back shows it did
  not make, fails with an `error_type` ending in `NotApplied`, so a client can
  match the family by that suffix: `TransformNotApplied`,
  `RenderSettingNotApplied`, `MaterialParamsNotApplied`, `PhysicsNotApplied`,
  `PhysicsParamsNotApplied`, and for the actor tools `AttachmentNotApplied`
  (parenting), `ComponentNotApplied` (a component not added),
  `RenameNotApplied` (a component name or an actor label not taken),
  `DeleteNotApplied`, `VisibilityNotApplied` and `CameraNotApplied` (the
  viewport camera). The actor tools' ones carry `details.expected`, what was
  asked, and `details.actual`, what Unreal reads back.
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
  `AttachmentNotApplied`, and its transaction is cancelled.
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
- Every actor edit is one undoable editor transaction, with the actors it
  changes recorded first, so one editor undo reverts that edit and nothing
  before it: spawn, delete, transform, visibility, parenting, components and
  `call_unreal_actor_function` (Remote Control's `generateTransaction`, which
  records the actor and its root component; a function that changes another
  object records only what it records itself). An edit outside a transaction
  used to leave the undo history alone, so the next undo reverted the edit
  before it. An edit that changes nothing, such as showing a visible actor or
  calling a getter, leaves no undo step: Unreal drops a transaction that
  records no change.
- `set_unreal_actor_visibility` sets the visibility `scope` names, on the actor
  and, with `propagate` (the default), on every actor attached under it, since
  a hidden parent hides none of its children:
  - `editor`: the editor's hide, as the Outliner's eye. The level viewport and
    `capture_unreal_viewport` leave the actor out. It lasts the editor session;
    with `persistent: true` the hidden-at-editor-startup flag, saved with the
    level and applied when it loads, is set too (or cleared, to show it). Unreal
    sets that flag only through its startup commands, which act on the
    selection, so the actors are selected for it and the selection restored;
    `persistent` is refused while PIE runs, when the editor selects nothing.
  - `game`: hidden in game, saved with the level, which Play In Editor and the
    packaged game honour. The level viewport still draws the actor outside
    Game View.
  - `all` (the default): both.

  The reply's `actors` lists each actor's `hidden_in_editor`,
  `hidden_in_editor_at_startup` and `hidden_in_game` read back, the actor
  first; one that reads back other than asked fails the call with
  `VisibilityNotApplied`.
- `spawn_unreal_actor` refuses a class Unreal does not let you place, and a
  Blueprint of one: the level's per-level singletons (WorldSettings,
  LevelScriptActor, WorldDataLayers, the World Partition minimap, the default
  physics volume, the particle event manager, navigation data) and the
  game-framework classes the game mode spawns when play starts (GameModeBase,
  GameStateBase, GameSession, GameNetworkManager, PlayerState, Controller,
  HUD, PlayerCameraManager), all declared NotPlaceable, which the editor's
  spawn functions ignore (a second WorldSettings landed in a saved level). It
  refuses the bare `Brush` class too (volumes spawn), an abstract class, and,
  before anything loads, a map or a path into one (an actor in a level). A
  label with leading or trailing whitespace, over 1023 characters or `None` is
  refused before anything spawns, since Unreal trims or drops it without a
  word, and the label is read back: one Unreal still does not take removes the
  actor again (`RenameNotApplied`).
- `delete_unreal_actor` takes the actor's full path; a label, a missing path or
  an actor already deleted is `NotFoundError`. The level's WorldSettings, its
  builder brush and the other actors the editor keeps out of the Outliner are
  refused with `ValidationError`, as the editor does not let you delete them.
  The deletion is read back (`DeleteNotApplied` when the actor is still
  there).
- `set_unreal_camera_view` takes each axis of `location` within
  4398046511104 cm (2^42) of the origin and `pitch` from -90 to 90 degrees;
  Unreal clamps anything beyond without a word, so a value outside is a
  `ValidationError` and the camera does not move. The reply is the camera
  read back, location and rotation, also for a group left unchanged; a camera
  that reads back other than what was set is a `CameraNotApplied`.
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
  Export options accept booleans `export_actor_folders`, `export_sublayers`.
  - `import_unreal_usd` options are Unreal 5.7's USD importer settings:
    booleans `import_actors`, `import_geometry`, `import_materials`,
    `import_skeletal_animations`, `import_level_sequences`,
    `import_groom_assets`, `import_sparse_volume_textures`, `import_sounds`,
    `import_only_used_materials`, `use_prim_kinds_for_collapsing`,
    `merge_identical_material_slots`, `share_assets_for_identical_prims`,
    `prim_path_folder_structure`, `interpret_lods`; `kinds_to_collapse` (a
    list of USD kinds: `model`, `component`, `group`, `assembly`,
    `subcomponent`); `fallback_collision_type` (`none`, `convex_decomposition`,
    `convex_hull`, `sphere`, `cube`, `mesh_simplification`, `capsule`,
    `custom_mesh`); `nanite_triangle_threshold`; `prims_to_import` (prim
    paths); and `meters_per_unit` with `up_axis` (`y` or `z`), given together,
    to override the stage's own units and up axis. An option left out takes
    Unreal's default, whatever the project's saved settings say. By default
    the importer merges the meshes under a prim of a collapsing kind into one
    (three exported StaticMeshActors came back as one `SM_Root`); pass
    `use_prim_kinds_for_collapsing` false for a mesh per prim. The default
    fallback collision, `none`, uses the render triangles as collision, so a
    mesh has no simple shapes; `convex_hull` gives each a hull. Earlier simul
    versions documented `import_lights` and `import_cameras`, which 5.7 does
    not have ("Failed to find property"): lights and cameras are imported as
    actors with `import_actors`. Any other name or a malformed value is a
    `ValidationError` before anything runs.
  - `import_unreal_usd` files the assets under
    `<destination_path>/<file name>` (the name as Unreal names assets: `my
    robot.usda` becomes `my_robot`) and spawns the actors in the level being
    edited. `destination_path` is a folder under `/Game`, `/Game` itself
    included; another content root is a `SandboxError`. Without
    `replace_existing` (the default) an import into a folder
    that already holds assets is refused with `InvalidState` and nothing is
    imported, and existing actors are kept: the new ones are added beside
    them. Unreal's USD importer otherwise replaces both, whatever
    `AssetImportTask.replace_existing` says: a re-import replaced a mesh
    customized since (its collision and lightmap resolution were lost) and
    deleted the earlier import's actors. With `replace_existing` true the
    folder's assets and the level's actors from an earlier import of the file
    (the top-level actor named after the file) are replaced;
    `removed_actor_paths` lists the actors that went, and `actor_paths` lists
    each new actor once. `warnings` holds the warnings and errors the editor
    logged during the import (read back from its log file; they can include a
    line from other editor activity at the same time), and a file the
    importer cannot read fails with `EngineError` carrying those lines. The
    log file is the one this editor writes, found by a marker line the script
    logs first: a second editor of the same project writes
    `<Project>_2.log`, and the lines used to come from `<Project>.log`, the
    other editor's.
  - The USD file's composition must stay inside the sandbox, as in the
    headless USD tools: before the file reaches the editor, simul walks its
    layers with `simul.usd.sandbox.CompositionSandbox` (every sublayer,
    reference, payload, value clip and asset-valued attribute such as a
    texture, in every variant). A stage inside the sandbox whose
    `references = @../../outside/ref.usda@` names a file outside it used to
    import that file's prims; it is now refused with `SandboxError`, whose
    `details.outside_sandbox` names each path, its kind and the layer that
    authors it, and `details.references_outside` the locations. A file pxr
    cannot open on the simul host is refused too, since nothing then vouches
    for what it names. An Unreal content path that Unreal's own exporter
    writes into an asset-valued attribute
    (`info:unreal:sourceAsset = @/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial@`)
    is not a file: the importer loads that object from the editor's content.
    Only that shape passes: an asset-valued attribute (not a sublayer,
    reference, payload or clip) whose value is under `/Engine`, `/Game` or
    `/Script` and ends in `<Name>.<Name>` with no file extension, naming no
    file on disk. `@/Engine/x.usda@`, a relative path and another content
    root (a plugin's) are refused like any path outside the sandbox.
  - An import can leave engine or plugin packages modified in memory: on UE
    5.7.4 Interchange set `bUsedWithNanite` on its shared parent material
    `/InterchangeAssets/Materials/PBRSurfaceMaterial_MR`. Unreal does that on
    a later editor frame, up to several imports after the one whose mesh
    needed it, so each import lists in `dirty_engine_packages`, with a
    warning, every package outside `/Game` the editor holds modified when it
    finishes, whichever call left it so. Python can neither clear that flag
    nor save one package of it safely: simul saves none of them
    (`save_unreal_assets` skips them, and its `dry_run` lists them too), but
    the editor's **Save All** would write them into the engine install.
    Discard them when the editor asks, or restart it. `import_unreal_asset`
    reports the same.
  - The import options are set without saving them. Unreal's
    UsdStageImportOptions saves its settings to the project's
    `Saved/Config/<Platform>Editor/EditorPerProjectUserSettings.ini` whenever
    a change is notified, and earlier simul versions notified it, so one
    import with `import_actors` false left `bImportActors=False` there for
    every later import and the editor's own **Import into Level** dialog.
    simul now sets every option above on every import and, when the
    project's saved settings differ from Unreal's defaults, says so in
    `warnings` and leaves them alone: nothing tells a value simul wrote from
    one the user chose. Reset them in the editor's USD import dialog.
  - `export_unreal_usd` exports actors of the level being edited; a path in
    another map is refused without loading it. `actors_exported` counts the
    actors the written stage has a prim for, read back from the file
    (`exported_prims` gives each one's prim), since Unreal's exporter leaves
    out editor helpers such as WorldSettings or PlayerStart without an error;
    they are listed in `skipped_actors`, and an export that wrote none of the
    actors fails with `EngineError`. Unreal writes a file per mesh and
    material into the export's asset folder, `<output name>_assets` beside
    the output, so two exports into one folder keep their meshes apart; with
    `export_sublayers` it also writes a layer per sublevel and
    `MaterialOverrides` beside the output. Without `overwrite` the export is
    refused with `InvalidState`, listing `details.existing_files`, when any
    of them exists: the output, a file in its asset folder, or a sublevel
    layer or `MaterialOverrides`. The gate used to cover the output alone, and
    the mesh files beside it were replaced. `written_files` lists every file
    the export created or changed under the output's folder and
    `overwritten_files` those that existed before. With `overwrite`, a file
    whose content would not change is left as it is (Unreal skips identical
    files), so it is not in `written_files`. Unreal's exporter writes a
    DynamicMeshActor as an empty Xform, without its mesh; `warnings` names
    them, and `convert_unreal_mesh_format` with `target_format`
    `static_mesh` and `replace_actor` true makes them StaticMeshActors, which
    export with their geometry.
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
- `import_unreal_asset` imports an FBX, glTF, GLB or OBJ file through
  Unreal's Interchange importer with the project's default content pipelines,
  as the Content Browser's **Import** does: the meshes, materials and textures
  are created and saved under `destination_path` (a `/Game` folder; glTF files
  get a folder of their own), and no actor is placed. The file must be inside
  the sandbox, like a USD file, and so must every file it names, which
  Interchange reads too: a glTF's buffers and images, an OBJ's MTL libraries
  and their textures, an FBX's textures, and the other UDIM tiles of a
  texture named like one (`wood.1001.png`). Each is looked for where Unreal
  5.7's translators look (`simul.utils.asset_references`, shared with the
  Blender import): an OBJ's textures are joined to the OBJ's folder even when
  the MTL is elsewhere, glTF URIs are URL-decoded, an FBX texture is tried as
  its stored name (a relative one where the editor opens it) and joined to
  the FBX's folder, and `..` is judged both as Unreal collapses it and as the
  file system follows it. A referenced file outside the sandbox that exists
  refuses the import with `SandboxError`, naming it in
  `details.references_outside` (and the model in `referenced_by`), before
  the editor reads anything; one that does not exist reads nothing and
  passes. A binary FBX simul cannot parse is refused, since nothing then
  vouches for its textures. Without `replace_existing` an asset that already
  exists is kept (`kept_assets`), and a file whose every asset exists imports
  nothing (`InvalidState`, listing the meshes, materials and textures in
  `details.existing_assets`). With it, Unreal reimports the asset named after
  the file and keeps the other assets an earlier import of the file made, as
  they were, even when their source changed (a 32 px texture redrawn at 48
  px stayed 32 px): the reimported asset is in `replaced_assets`, the ones
  kept in `kept_assets`, with a warning. Delete them, or import into another
  folder, to import them anew. An earlier import's assets are told apart by
  the files their import data names. `warnings` holds what the editor logged,
  `dirty_engine_packages` the engine or plugin packages the import left
  modified, and a file Unreal cannot read fails with `EngineError` and its
  logged reasons. It waits `unreal.import_export_timeout`. USD files go
  through `import_unreal_usd`.
- `save_unreal_assets` saves modified packages to disk: the packages of
  `asset_paths`, the level being edited with `save_level` (its map package
  and, in a World Partition level, its actors' and objects' packages), or
  every modified package with `all_dirty`. Only modified packages are written.
  Packages outside `/Game` (engine or plugin content, which importers and edits
  can leave modified) are skipped unless `allow_engine_content`, as are ones
  never saved (under `/Temp`); `skipped` says why for each. `dry_run` lists
  what would be saved. A package still modified afterwards is in `failed`,
  and the call fails with `EngineError`. A level never saved is
  `InvalidState`: save it once in the editor to give it a file. In attached
  mode it runs, like every tool, only while the attached map is open.
- `assign_unreal_material` assigns the material in one undoable transaction
  that records the component first, which marks the level dirty (in a World
  Partition level, the actor's package), so saving keeps the assignment and
  Undo reverts it; the slot is read back, and `previous_material_path` is what
  it rendered with before. The actor must be in the level being edited: a path
  in another map is refused without loading that map. A material path that
  names something other than a material (a level, which is never loaded, or
  a mesh) is an `InvalidAssetType`, as in `create_unreal_material_instance`,
  and one that names nothing a `NotFoundError`. `component_name` picks the mesh component by name (or
  path); without it the actor's mesh root component is used, else its only
  mesh component, and an actor with several is a `ValidationError` naming
  them. `slot_name` picks the material slot by the name the Details panel
  shows; an unknown component or slot is a `ValidationError` listing the
  actor's components or the component's slots.
- `create_unreal_material_instance` writes under the project's `/Game` folder:
  `save_path`, else next to a parent in `/Game`, else `/Game/Materials`. It
  used to save an instance of an `/Engine` parent into the engine install
  (`Engine/Content/BasicShapes`), which every project using that engine
  shares. A folder outside `/Game` (`/Engine`, a plugin's content) is refused
  with `SandboxError` unless `allow_engine_content` is true, and
  `set_unreal_material_params` refuses an instance outside `/Game` the same
  way. A folder with an empty, `.` or `..` name, or with characters Unreal
  does not allow, is a `ValidationError`: Unreal does not resolve `..` and
  turned `/Game/../Engine` into a folder named `__`. A missing parent is a
  `NotFoundError`, a parent that is no material (a StaticMesh, a level, which
  is never loaded) an `InvalidAssetType`, a name already taken an
  `InvalidState`, and an instance Unreal created but did not save an
  `EngineError` naming it.
- `search_unreal_assets` reads the editor's asset registry, loading nothing,
  in or out of Play In Editor. Without `package_paths` it searches every
  mounted content root (`/Game`, `/Engine` and each plugin's); with them, their
  subfolders too unless `recursive_paths` is false. `class_names` take a
  class's name (`StaticMesh`, `Material`, `Texture2D`) or its path
  (`/Script/Engine.StaticMesh`), with subclasses unless `include_subclasses`
  is false; an unknown name is a `ValidationError`. The query matches part of
  the asset name, ignoring case, and redirectors are left out. Matches are
  sorted by object path; `total` counts them, `truncated` says more follow and
  `next_offset` is the `offset` of the next page. Each asset has its
  `class_path`, `package_name`, `package_path` and `metadata`: the registry
  tags Unreal records for its kind, as the strings it stores (`Triangles`,
  `Vertices`, `LODs`, `NaniteEnabled`, `CollisionPrims` for a mesh;
  `Dimensions` and `Format` for a texture; `MaterialDomain`, `BlendMode` or
  `Parent` for a material). Remote Control's search, used before, matched no
  short class name, searched only `/Game` and no subfolders, and could not
  tell whether more matched. `max_results` is 1 to 1000: Remote Control sized
  its result array to the limit it was sent, and -1 crashed the editor.
- `get_unreal_interchange_info` lists Interchange's formats for each kind of
  translator (`supported_formats_by_kind`: assets, scenes, actors) and the
  project's pipeline stacks, content and scene, with each stack's pipeline
  assets (`pipelines`, `default_pipeline_stacks`).
- In attached mode a tool run after the attached editor, map or viewport
  changed fails with `InvalidState` and `details.reason`
  `"attachment_changed"`, having run nothing; run `simul unreal attach` again.
  It used to come back as the tool's own traceback.
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
  makes it the whole editor selection, reading the selection back: `selected` is false, with a
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
    `generate_unreal_procedural_scene`, every mesh tool except
    `validate_unreal_mesh` (they create actors or replace meshes, see below),
    `import_unreal_usd`, `export_unreal_usd`, `import_unreal_asset`,
    `save_unreal_assets` (the editor saves nothing during PIE) and, in attached
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
    physics settings, visibility, materials, parenting, components) keeps
    working and changes the editor world, which the running simulation does
    not see: it takes effect the next time the simulation starts, and a
    physics or collision edit says so in its `warning`. A physics or collision
    edit given the actor's path in the simulation's world (`UEDPIE_<n>_<map>`)
    changes the running copy instead, which is neither undoable nor saved;
    every other actor tool finds no actor at such a path, and its error names
    the editor's path. `set_unreal_actor_visibility` with `persistent` is
    refused with `PIEActive`, as the editor selects nothing then.
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
- The mesh tools (`generate_unreal_mesh_primitive`, `validate_unreal_mesh`
  and the other `*_unreal_mesh*` and `*_unreal_convex_hull` tools) run
  GeometryScript in the editor:
  - They need the GeometryScripting plugin, which `simul unreal setup` enables
    (restart an editor that was already running). Without it every mesh tool
    returns `error_type: "UnsupportedOperation"` with `details.code:
    "GEOMETRY_SCRIPTING_DISABLED"` and a hint naming setup.
  - They take the path of an actor in the level open in the editor, found
    without loading a map, or of an asset. A path that names nothing is
    `NotFoundError`. `validate_unreal_mesh`, `compute_unreal_convex_hull` and
    `decompose_unreal_convex_hull` read a DynamicMeshActor, a StaticMeshActor
    or a StaticMesh asset (its first LOD, copied into a transient
    DynamicMesh; the reply's `source_kind` says which). The tools that edit a
    mesh in place take a DynamicMeshActor only: a StaticMeshActor
    (`details.code: "STATIC_MESH_ACTOR"`) or a StaticMesh asset path
    (`"STATIC_MESH_ASSET"`) is a `ValidationError` whose `details.hint` names
    `convert_unreal_mesh_format`; any other actor is `"NO_DYNAMIC_MESH"`.
  - `convert_unreal_mesh_format` moves meshes between the two kinds.
    `dynamic_mesh` places a DynamicMeshActor copy of a StaticMesh asset (at
    `location`) or of a StaticMeshActor (at its transform), with its
    materials; `replace_actor` removes the StaticMeshActor in the same
    transaction. `static_mesh` writes a DynamicMeshActor's mesh into a
    StaticMesh asset: `asset_path` (default `/Game/Generated/SM_<label>`, the
    label's ASCII letters and digits joined by underscores) must
    be under `/Game` (anything else, `/Engine` or a plugin's content, is
    `SandboxError` `"OUTSIDE_GAME_CONTENT"`); an existing StaticMesh is
    refused (`"ASSET_EXISTS"`) unless `overwrite`, which replaces its mesh and
    keeps its references; the component's materials go to its slots; its
    simple collision is the component's, a convex hull or none (`collision`,
    `auto` by default); `nanite` enables Nanite; `save` (default true) saves
    that asset's package and nothing else, and the reply names the `.uasset`
    written. With Nanite the asset's `triangle_count` and `vertex_count` are the
    mesh written, which Nanite renders, and `fallback_triangle_count` and
    `fallback_vertex_count` its fallback mesh (LOD 0: 316 triangles for a
    1,920-triangle sphere). `replace_actor` swaps the DynamicMeshActor for a
    StaticMeshActor at its transform, in one undoable transaction. Writing the
    asset, new or overwritten, with its materials and collision, is not
    undoable: Unreal records no undo step for a new asset, so simul records
    none for the rest of the write either (its collision used to be an Undo
    step of its own, and one Undo dropped it from the asset). Unreal's USD exporter
    writes a DynamicMeshActor as an empty Xform, without its mesh, so convert
    with `replace_actor` before exporting a mesh made with these tools; the
    reply's `hint` says so when the DynamicMeshActor was left in place.
  - A hull or decomposition with `output` `collision` or `both` on a
    StaticMesh source sets the StaticMesh asset's simple collision (every
    actor using it shares it), only for an asset under `/Game`, in one
    undoable transaction; `save` saves that asset. Undo restores the asset's
    collision, but Unreal does not rebuild the physics of the actors already
    placed with it: until the level is reloaded they keep colliding with the
    hull (a line trace still hits it). Save before such an edit, or set the
    collision again rather than undo it.
  - Every GeometryScript call is checked after it returns. GeometryScript
    reports most failures that way and otherwise leaves the mesh as it was
    (Catmull-Clark on a mesh without quad polygroups returned with the
    triangle count unchanged). An error fails the tool with `EngineError`,
    `details.code: "GEOMETRY_SCRIPT_FAILED"` and GeometryScript's messages in
    `details.messages`, and nothing is changed; warnings come back in
    `warnings`.
  - An edit runs on a copy of the mesh, and the copy replaces the component's
    mesh in one transaction, so a failed edit changes nothing, one Undo
    reverts one edit and the level is marked dirty for Save. A tool that
    creates an actor spawns it, with its mesh, in one transaction too.
  - Undo and Redo of a mesh edit restore which mesh the DynamicMeshComponent
    holds, not what Unreal does when a mesh is set (UE 5.7.4): the component
    stops listening to the mesh it got back, so an in-place edit of that mesh
    outside simul (a Blueprint, Python GeometryScript, Modeling Mode) no longer
    updates its bounds or what it draws, and its physics keeps the simple
    collision it had before the Undo or Redo (a trace against a hull that Redo
    put back misses). Every mesh tool repairs both on each DynamicMeshActor it
    touches, `validate_unreal_mesh` included: it rebuilds the collision, and
    where an Undo or Redo left the actor's package dirty (or in a level never
    saved) it sets the component's mesh again, which binds it (and renames the
    mesh object). A clean level is left alone, so a level saved right after an
    Undo keeps the stale binding until it is reloaded or edited.
  - Each in-place edit keeps the mesh it replaced alive for Undo: a full copy
    that the editor's undo buffer limit (256 MB by default) does not count, so
    it is freed only when the edit leaves the undo history. A hundred edits of
    a 32,256-triangle mesh added about 370 MB to the editor; one UV unwrap of a
    1M-triangle mesh about 760 MB.
  - A mesh tool leaves at most 2,000,000 triangles in a mesh it makes or
    edits. A larger result is refused with `details.code: "RESULT_TOO_LARGE"`
    (`details.predicted` says whether it was predicted or measured) and
    nothing is changed: before anything runs when the size is known up front
    (a subdivision, a remesh, which makes up to 1.6 times its target, a UV
    unwrap or topology edit of a mesh already over the cap, a primitive with
    more than 512 segments), otherwise when the operation has run and before
    its result replaces anything.
  - `compute_unreal_mesh_uv` unwraps with XAtlas only up to 20,000 triangles:
    XAtlas crashed the editor on a 130,048- and a 318,400-triangle mesh, and
    took 28.6 s for 32,256 (UE 5.7.4). Method `auto`, the default, uses
    PatchBuilder above that size (29 s for 318,400 triangles) and the reply's
    `method` names the one that ran; `xatlas` on a larger mesh is refused with
    `details.code: "MESH_TOO_LARGE_FOR_XATLAS"`. A `uv_channel` the mesh does
    not have is refused (`"UV_CHANNEL_MISSING"`).
  - Positions are where the actors are in the level. `apply_unreal_mesh_boolean`
    combines both meshes at their actors' world transforms (location,
    rotation, scale) and writes the result in the target's local space.
    `cut_unreal_mesh_plane` takes its plane in world space unless `space` is
    `local`, removes the half the normal points into (or keeps it as a second
    actor with `keep_both_sides`), reports the plane in the mesh's space, and
    refuses a plane that misses the mesh (`"PLANE_MISSES_MESH"`). The hull,
    decomposition and cut actors take the source's whole transform. Face
    selections of `edit_unreal_mesh_topology` (`normal:x,y,z[,angle]`,
    `box:...`, `all`) are in the mesh's local space.
  - `compute_unreal_convex_hull` and `decompose_unreal_convex_hull` produce
    simple collision: `output` `actor` (default) spawns a DynamicMeshActor
    whose mesh and simple collision are the hull or pieces, `collision`
    replaces the source's simple collision, `both` does both, in one
    transaction (Undo restores the collision Unreal uses, the component's
    BodySetup included). Unreal rebuilds the physics of neither: after an Undo
    or Redo a line trace still meets the shapes from before it (Redo of a hull
    put the shape back, and traces missed it) until a mesh tool next touches
    the actor, which rebuilds its collision. The hull is simplified to `max_faces` triangles
    (250; 0 keeps the exact hull, 84,064 triangles for a 130,048-triangle
    sphere). The decomposition is Unreal's own (GeometryScript's collision
    generation, not V-HACD), with its sphere, box and capsule detection off:
    left on, a sphere or capsule source came back as "Hull Computation Failed"
    and a 100 cm cube. `max_vertices_per_hull` limits each piece (as 2V - 4
    triangles); `error_tolerance` and `min_part_thickness` are passed on;
    `volume_ratio` says how much of a closed source the pieces cover.
  - `generate_unreal_mesh_primitive` takes each type's dimensions by name in
    Unreal's axes (box `width` X, `depth` Y, `height` Z; capsule `radius` and
    `length`, its straight part; torus `major_radius`, `minor_radius`), refuses
    unknown names and sizes that are not above 0, and gives primitives quad
    polygroups, so Catmull-Clark subdivides them. `segments` is 3 to 512.
  - Options are implemented or refused, never ignored: `subdivide_unreal_mesh`
    runs Catmull-Clark, Loop or uniform tessellation (levels 1 to 4; a result
    over 2,000,000 triangles is refused); `simplify_unreal_mesh` takes one of a
    triangle count, a fraction below 1 or `max_error` (a tolerance in cm);
    `remesh_unreal_mesh` is uniform only and keeps the triangle count by
    default; `compute_unreal_mesh_uv` packs islands `island_padding` texels
    apart at `texture_resolution`; `edit_unreal_mesh_topology` needs a
    selection and reports `faces_affected` and `edges_affected`.
  - `validate_unreal_mesh` runs the checks named in `checks` (all by
    default): `closed`, `manifold` (bowtie vertices), `degenerate_triangles`,
    `orientation` (a closed mesh encloses a positive volume), `compact`,
    `normals` and `uvs`, each with `passed`, `count` and `detail`. It does not
    look for self-intersections.
  - A mesh script runs under an HTTP timeout of at least 300 s, whatever
    `UNREAL__TIMEOUT` says, since a GeometryScript call cannot be cancelled.
    The slowest operations on a 2M-triangle mesh took 83 to 109 s (a
    PatchBuilder unwrap, a convex hull, a simplification; UE 5.7.4, Apple
    M-series). A script commits its result only within that timeout less 10
    s: one that finishes later discards its result, so a `TimeoutError`
    (`details.code: "MESH_TIME_BUDGET"`) leaves the level as it was once the
    editor finishes, though the editor stays busy until then. For an editor on
    this machine the limit runs on this machine's clock, so it holds however
    long the call waited in the editor's queue; for a remote editor it is the
    script's own run time, and a call that first waited behind another script
    may still commit late.
  - In attached mode, a call the attachment guard refuses (the editor shows
    another map or instance) is `error_type: "InvalidState"`, `details.code:
    "ATTACHMENT_CHANGED"`.
- Screenshots need console-command execution enabled in Remote Control. When
  upgrading an older project, re-run `simul unreal setup` and restart the
  editor.
- Every Python call simul makes runs in a namespace of its own, and that
  namespace is cleared when the call ends, whether it returns or raises. This
  covers its own tools, `execute_unreal_script`, `simul unreal exec`, and
  `batch_unreal_operations` calls to `ExecutePythonCommandEx` or
  `ExecutePythonCommand`. (`call_unreal_actor_function` takes only actors of
  the level open in the editor, so it never reaches the Python library.)
  - **Why it matters.** UE 5.7 runs all literal Python in one dict that every
    call and the editor's own Python console share, and `FileExecutionScope`
    applies only to `.py` files. A name a script leaves there keeps its object
    alive for Unreal's garbage collector.
  - **The crash it prevents.** When the editor changes maps it releases
    Python's hold on the old world and the objects inside it, but not on the
    world's package or on objects outside it. A kept
    `pkg = world.get_outermost()` therefore made the next `new_blank_map` or
    map load abort the editor with "World Memory Leaks: Old Package ... not
    cleaned up by GC", losing unsaved work.
  - **`py` console commands are refused.** An `ExecuteConsoleCommand` whose
    command is `py <code>`, or has a line or `|`-separated part that is one,
    would run the code in that shared dict, past the per-call namespace. simul
    refuses it with a `ValidationError` before anything is sent; run the code
    with `execute_unreal_script` instead.
  - **What remains.** Anything a script stores outside its namespace stays
    alive for the editor's lifetime. A package or world kept that way, or an
    object that refers to one, still aborts the editor at the next map change
    as above. Places that keep things include a module attribute, an entry in
    `sys.modules`, a registered callback or what its closure holds, and a
    `unreal.uclass` type. An editor that ran an older simul keeps what those
    calls bound until it restarts.
- `execute_unreal_script` runs the code you send through
  `PythonScriptLibrary.ExecutePythonCommandEx`, in or out of PIE.
  - **A fresh namespace per call.** Each call starts with what a fresh
    `__main__` holds (`__builtins__`, `__doc__`, `__spec__` and the rest) and
    `unreal` imported. Nothing carries over between calls, so pass values back
    in as code. A callback the script registers must import what it uses
    inside itself, because the script's names are gone once it returns.
  - **Mode behaviour is unchanged.** `ExecuteStatement` prints an expression's
    value as the console does, but no longer keeps it in `builtins._`. A
    `sys.displayhook` the script installs itself stays installed.
  - **Code mentioning `.py` now runs.** Code that names a `.py` file followed
    by a space (`print('see my.py first')`) runs as code; Unreal used to take
    it for the path of a file to run.
  - **A byte-order mark is skipped**, as Unreal skipped it, so a file saved as
    "UTF-8 with BOM" runs.
  - **A traceback starts with two frames of simul's.** The command's
    (`File "<string>", line 1`) and the runner's (`File "<simul runner>"`)
    come before the script's own frames, which keep the script's line numbers.

  On UE 5.7 avoid one engine hazard in that code: Unreal wraps
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
  reference that is dropped or stays reachable is safe.
  - **Top-level aliases are safe.** An alias bound at the top level
    (`names = unreal.MaterialEditingLibrary.get_scalar_parameter_names`) is
    dropped when the call's namespace is cleared, even alongside functions
    that use it.
  - **Classes the call leaves are kept safe.** A class is always in a cycle,
    so a class the script defines becomes garbage when the call ends. simul
    keeps the library functions held by the classes the call's namespace
    still reaches when it ends: bound to a name or not (appended to a list in
    a loop, made with `type()`, the class of an object there). That covers
    their attributes, the containers there, and their methods' default
    arguments and closures. The class is collected without them, and the
    next call lets go of each one nothing else holds any more.
  - **Other cycles are not.** A cycle holding such a function that the
    namespace no longer reaches when the call ends still crashes the editor's
    next garbage collection: a class defined and dropped inside a function,
    or one a loop rebinds the same name over without keeping it.

  Module functions
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
| Objects | `create_blender_object`, `delete_blender_object`, `set_blender_object_mode`, `get_blender_object_info`, `set_blender_object_transform`, `set_blender_object_parent`, `clear_blender_object_parent`, `create_blender_mesh_from_data`, `clean_blender_mesh`, `get_blender_mesh_info`, `add_blender_modifier`, `apply_blender_modifier`, `remove_blender_modifier` |
| Measurement | `get_blender_bounding_box`, `check_blender_object_bounds`, `get_blender_distance_between` |
| Materials and lights | `get_blender_material_info`, `assign_blender_material`, `uv_unwrap_blender_mesh`, `set_blender_light_params` |
| Physics | `setup_blender_rigid_body`, `add_blender_rigid_body_constraint`, `get_blender_constraint_info`, `add_blender_force_field`, `get_blender_force_field_info`, `get_blender_physics_state`, `bake_blender_simulation`, `free_blender_bake`, `get_blender_rigid_body_world`, `set_blender_rigid_body_world` |
| Animation | `get_blender_frame`, `set_blender_frame`, `set_blender_frame_range`, `play_blender_animation`, `insert_blender_keyframe`, `insert_blender_keyframes`, `delete_blender_keyframe`, `get_blender_keyframes`, `set_blender_pose_bone_transform`, `get_blender_object_trajectory`, `get_blender_trajectories` |
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
- `create_blender_object` puts the object in the active collection, as
  Blender's Add menu does. When that collection, or one holding it, is
  excluded from the view layer, hidden or disabled in viewports, the object
  goes in the scene's own collection instead, and `warning` says so; the
  user's active collection stays as it was. The new object is the one the add
  operator created (an object hidden in the 3D View is not its
  `context.active_object`), and a failed call removes it again. It stays in
  Object Mode whatever Blender's "Enter Edit Mode" preference says, and
  outside Object Mode it is refused before anything is added (embedded mode
  too, where an add operator would otherwise add to the edited mesh).
  `set_blender_object_mode` switches an object (the active one by default)
  into a mode, `OBJECT` by default: the way back for a window left in Edit or
  Pose Mode, where creating, deleting, importing and the other mode-sensitive
  tools refuse to run.
- `delete_blender_object` runs Blender's own Delete on the object alone, the
  selection untouched: the object leaves the current scene, and stays in the
  file while another scene links it (`remaining_scenes`, `removed_from_file`).
  `use_global: true` deletes it from every scene (Delete Globally). The
  operator also resets a deleted rigid body's cache: removing the object
  directly left the cache, so after a frame jump another body took the
  deleted one's cached positions.
- `list_blender_scene_objects` and `search_blender_objects` return a page:
  `count` objects from `offset`, `total` matching, and `next_offset` for the
  next page (`null` after the last; `truncated` says whether more follow). A
  page stops before its entries pass 30 000 bytes of JSON, so it can hold
  fewer than `max_items`/`max_results`: the server's 40 KB result budget used
  to cut a longer list below the count it reported. Each entry's
  `collection` names a collection holding the object, the scene's own being
  "Scene Collection". `object_type` matches in any case and must be one of
  the running Blender's object types.
- `check_blender_object_bounds` tests the object's world-space bounding box,
  turned and scaled with the object, against the axis-aligned bounds:
  `relation` is `inside` (`within_bounds`), `intersecting` (touching counts)
  or `outside`, and `origin_within_bounds` tests the origin alone, which is
  all the tool used to test. `bounds_min` must not exceed `bounds_max`.
- `summarize_blender_scene`'s `collection_names` are the scene's top-level
  collections. `get_blender_file_info`'s `object_count` counts the current
  scene's objects, and `file_object_count` the file's.
- Reads of world-space transforms (object info, distance, bounds, bounding
  box, parenting) evaluate the depsgraph first: in embedded (bpy module) mode
  Blender leaves a transform written by an earlier call out of `matrix_world`
  until then. An add-on built before `rotation_quaternion`, `keep_transform`,
  `offset` and `use_global` still answers calls that leave them at their
  defaults, with its own behaviour; the newer reply fields then come back
  `null`.
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
- `assign_blender_material` reuses the material named `material_name` when it
  exists, so repeating a call changes nothing more (it used to create
  `name.001` each time); `reuse_existing: false` creates a new one. Shading
  arguments left out keep the material's own values, so an existing or
  imported material can be assigned as it is. By default the material replaces
  the object's: it goes into slot 0 and every face uses slot 0 (a second
  material used to land in a slot no face used, so the USD kept the first).
  `slot_index` sets one slot (the slot count appends one) and leaves the faces
  alone; `face_indices` assigns the material to those faces through its slot,
  or a new one. `link: "OBJECT"` puts the material in this object's slot
  instead of the mesh's, but which faces use which slot, and how many slots
  there are, belong to the mesh. So on mesh data that other objects share,
  link OBJECT may change only an existing slot's material: assigning faces or
  adding a slot is refused (`details.code: "shared_data"`, naming the other
  objects) unless `single_user: true` gives this object its own copy of the
  mesh first, which the reply reports. That copy needs the object, and its
  data, in Object Mode: Blender 5.0.1 ignores a new data block for an object in
  Edit Mode without an error. With link DATA the change is meant for every
  object sharing the mesh, and the reply names them. `single_user` also makes
  library-linked data local; a library-linked object is refused.
  `base_color` is RGB or RGBA in 0-1: the alpha sets the Principled BSDF's
  Alpha (exported as USD `opacity`) and, below 1,
  `surface_render_method: BLENDED`. Image files can drive Base Color, Metallic
  and Roughness, and Normal through a Normal Map node; a new texture replaces
  the nodes of the old one. Data maps are read Non-Color; a base colour keeps
  the colour space Blender loads the file in (sRGB for an 8-bit image, linear
  for an EXR or other float image). An image of the same file already read
  the way the input needs is reused, so repeating a call adds no datablock.
  An image already in the file is never changed, since other materials may
  read it: when every datablock of the file reads the other way, the input
  gets one of its own (`albedo.png.001`). A texture replaced behind Reroute
  nodes takes the Reroutes and its old nodes with it. Textures pass the filesystem sandbox; a missing file is
  a `NotFoundError` and a file Blender cannot read as an image a
  `ValidationError`. Textures need UV coordinates (`uv_unwrap_blender_mesh`);
  a mesh without a UV map gets a warning, and so does a replaced material left
  with no users (Blender discards it when the file is saved and reopened).
  Everything is checked before the scene changes, and a failure removes the
  material, slot, mesh copy and images the call made. Writing face indices
  needs Object Mode, also of any other object editing the same mesh. On
  Blender 5.0 the tool never touches the deprecated `Material.use_nodes`.
  Blender's USD exporter writes UsdPreviewSurface materials only: SimReady's
  `shader_type: OmniPBR` is recorded as metadata, not produced.
- `get_blender_material_info` reports the Principled BSDF that feeds the
  active Material Output. An input a link drives reads `null` in
  `principled_params` (its own value is unused) and appears in
  `linked_inputs` with the source node and socket and, for an image texture,
  also behind a Normal Map node, the image's path, colour space and whether it
  is packed. Reroute nodes are followed to the node behind them; a muted link
  drives nothing, so its input reports its own value. `nodes` lists at most
  `max_nodes` (default 50), those feeding the output first; `node_count`
  counts the whole tree.
- `uv_unwrap_blender_mesh` generates UVs for every face, hidden ones
  included: `SMART_PROJECT` (islands split where faces meet at more than
  `angle_limit`, 66 degrees by default, packed with `island_margin`) or
  `CUBE_PROJECT` (`cube_size`). It writes `uv_map_name`, created when missing,
  else the active UV map, else a new `UVMap`. The active UV map stays what it
  was (a mesh's first UV map becomes the active and render one, as Blender
  makes it); the reply's `render_uv_map` is the map image textures read and USD
  export names `st`, with a warning when it is not the one written. The UV
  operators need Edit Mode, so the object briefly becomes the active, only
  selected one and its hidden faces are shown; the mesh's selection and hidden
  faces, the object selection and the active object are restored, and a
  failure removes a UV map the call created. The object and Blender must be in
  Object Mode; a hidden object and library-linked data are refused.
- The material, light and UV tools refuse bad arguments with `error_type`
  `ValidationError` (or `NotFoundError` for a missing material or texture
  file) and a stable `details.code`, such as `shared_data`, `out_of_range`,
  `setting_not_for_light_type`, `wrong_mode` or `linked_from_library`.
- These calls need a Blender add-on built for bridge protocol 11 or later:
  assignment now replaces where it appended, so an older add-on is refused
  with the rebuild instructions rather than silently appending a slot no face
  uses.
- `set_blender_light_params` checks each setting against the light's type
  (after `light_type`, when given): `shadow_soft_size` is the radius of POINT
  and SPOT lights, `spot_size` and `spot_blend` are SPOT's, `angle` is a sun's
  softness, and `shape`, `size` and `size_y` (RECTANGLE and ELLIPSE) are an
  area light's. A setting of another type is refused instead of ignored, a
  value outside Blender's limits is refused instead of clamped, and energy
  may not be negative. Nothing changes unless every value is valid, and the
  reply reports every setting of the light's type as stored. Light data shared
  by several objects is reported. Blender 5.0.1's USD exporter writes a point
  or spot light as a SphereLight whose `radius` is `shadow_soft_size` (a
  spot's cone as `shaping:cone:angle`, half of `spot_size` in degrees, and
  `spot_blend` as its softness), a sun as a DistantLight whose `angle` is half
  of the sun's `angle` in degrees, and an area light as a RectLight of
  `size` x `size_y` (`size` x `size` for SQUARE) or a DiskLight of radius
  `size` / 2.
- Not covered by granular tools yet (use `execute_blender_script`): smooth
  or flat shading and custom normals, and UV coordinates given with
  `create_blender_mesh_from_data`.
- `capture_blender_viewport` captures the 3D View the user sees in attached
  mode and needs no camera; `method: "render"` renders through the scene camera,
  or a temporary one when the scene has none. Embedded (headless) mode has no 3D
  View, so its captures are renders. The per-mode table is in
  [blender-attachment.md](blender-attachment.md#capture-feedback). `camera_name`
  captures through another camera without changing the scene camera or the
  view. `capture_blender_viewport_sequence` writes JPEG files and returns their
  paths, at most `max_frames` (50, at most 100) per call and no new frame after
  10 s, with `next_start_frame` to continue.
- `get_blender_viewport_info` reports, besides the render settings, `view3d`:
  the attached window's largest 3D View (the one captures use) with its
  `view_matrix` (world to view), `eye_location`, `view_perspective`, `lens`,
  `clip_start`/`clip_end`, `shading_type` and, for a camera view,
  `view_camera`. Without an editor window (embedded `bpy`, `blender -b`) or a 3D
  View it is `null`, and `view3d_unavailable_reason` says which.
- `import_blender_file` checks the files an import would read besides the one
  named against `security.allowed_paths`, where Blender's importer looks for
  them: a glTF's buffers and images (also in a GLB's JSON), an OBJ's MTL
  libraries and their texture maps, an FBX's textures, and a USD file's
  sublayers, references, payloads and asset paths such as textures, following
  the layers inside the sandbox. An existing file outside the sandbox is refused
  with `SandboxError` before anything is imported; `details.references_outside`
  lists those files and `details.referenced_by` the imported file. A reference to
  a missing file reads nothing and passes. The check follows the importers'
  fallbacks: an FBX texture's case-insensitive match (Blender's
  `resolve_ncase`) and its name when it stores no path, an MTL texture's base
  name beside the MTL, and USD's resolver search paths. The FBX importer's
  recursive image search (`use_image_search`), which walked the texture's
  folder and found a same-named image outside the sandbox, is switched off
  while the sandbox is on, so an FBX texture must be at the path it stores or
  beside the FBX. USD files are read through pxr, which Blender bundles; without
  pxr a binary USD file, or a text one that composes other layers, is refused.
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
- `play_blender_animation` takes `play`, `reverse`, `pause` or `stop`, and
  repeating a call changes nothing more (`screen.animation_play` on its own
  toggles playback, so a second `play` used to stop it). `play` and `reverse`
  start playback in that direction or turn it around; `pause` stops on the
  current frame; `stop` stops and returns to the frame playback started from,
  as Esc does in Blender, also after the tool turned it around. The reply
  reports `is_playing`, `direction`, `changed` (false when playback already ran
  that way, or was already stopped) and the current `frame`. Blender does not
  report which way playback runs, so playback that the user or a script
  started is restarted in the requested direction. Playback needs Blender's
  event loop: in background mode (`blender -b`, the `bpy` module) `play` and
  `reverse` are refused rather than reported as playing on a frame that never
  moves; step frames with `set_blender_frame` or sample them with
  `get_blender_object_trajectory`.
- `get_blender_frame` reports `fps` as Blender plays it, `render.fps /
  render.fps_base` (29.97 for NTSC's 30 / 1.001), and `fps_base` itself.
  `set_blender_frame_range` takes a one-frame range (start equal to end).
  Frames Blender would clamp are refused instead: a scene range outside
  0..1048574, or a current frame outside -1048574..1048574.
- `insert_blender_keyframe` keys the property's current value, as Blender's I
  key does, unless `value` gives one: one number for one channel (`index` 0 or
  more, or a property that is not an array), else one per element. A given
  value is keyed through the property, so NLA remapping applies, and each
  keyed channel then shows what its F-curve gives at the current frame (the
  keyed value on that frame, or where the key is the channel's only one).
  Left to itself, a GUI Blender 5.0.1 showed a first key at once and a later
  one only at the next frame change, and background Blender neither. Only the
  keyed channels are written: every other channel keeps its value, an
  unkeyed edit included, as Blender's own keying leaves it. On an object with
  NLA tracks, which Blender blends with the action only on a frame change,
  the keyed channels keep their values until then and `warning` says so; a
  muted or driven channel keeps its value too. `interpolation`
  sets the key's interpolation (otherwise Blender's choice: the previous key's
  on a curve with two keys or more, else its preference, Bezier by default)
  and `keyframe_type` its key type. The reply's `keyframes` give each keyed
  channel's value, interpolation and type as Blender stored them, and
  `value_source` says which value was keyed.
- `insert_blender_keyframes` keys many frames in one call: per entry of
  `channels` a `data_path`, an `index` (-1 for every element), `frames` and a
  value per frame, up to 10 000 keys (frames times elements) a call. Each value
  is keyed as `insert_blender_keyframe` keys a given one: the keyed channels
  then show what their F-curves give at the current frame, and every other
  channel keeps its value, an unkeyed edit included (on an object with NLA
  tracks the keyed channels keep theirs too, and `warning` says so). The same
  `interpolation` and `keyframe_type` go to every key. Every entry is checked
  before anything is keyed: a channel keyed twice on one frame, or on two
  frames within 0.01 of each other, which Blender keeps as one key (on Blender
  5.0.1 a key at 11.005 replaced the one at 11), is refused, and so is an
  F-curve Blender keys nothing on: locked, sampled, or under an F-modifier, not
  muted or invalid, that overrides its keys (all but Cycles, Stepped, Noise and
  an additive Generator; on Blender 5.0.1 a LIMITS modifier did). Should Blender
  still refuse a key part way (an NLA setup it cannot remap a key onto), the
  keys that went in stay, the reply is a `KeyframesPartlyInserted` error whose
  `details` give the keys inserted and the one refused, and in attached mode
  the call is one undo step. On Blender 5.0.1 it inserted 6 000 keys (a 6-DOF
  arm over 1 000 frames) in 0.09 s, which took some 13 000 calls of 55 ms one
  key at a time.
- `get_blender_keyframes` lists the keyed channels a page at a time: each
  channel's data path, array index, the pose bone it animates (`bone_name`),
  its key count and `frame_range` (in float frames, since a key can sit between
  frames). `include_keys` adds every key: frame, value, interpolation and key
  type, the handles and their types for a Bezier key, and the easing of an
  eased one. `bone_name` keeps one pose bone's channels, and `data_path` one
  property's (the bone's, with `bone_name`: `rotation_quaternion`, or its full
  path, `pose.bones["Bone"].rotation_quaternion`). A page holds what fits in one
  reply, up to `limit` channels; `next_offset` (and, with `include_keys`,
  `next_key_offset`, as a long channel can run over several pages) is what the
  next call passes as `offset` and `key_offset`, and `truncation` says so. They
  are null on the last page. An 80-bone rig's 800 channels came back as 301,
  with a hint naming parameters the tool did not have; they now come in four
  pages.
- `set_blender_pose_bone_transform` sets a pose bone's own location, rotation
  and scale: its pose relative to the rest position, before constraints, as the
  bone's Transform panel in Pose Mode shows it. The rotation is written in the
  bone's `rotation_mode`, which is kept: `rotation_euler` takes XYZ Euler
  angles and `rotation_quaternion` a quaternion (w, x, y, z), either converted
  to that mode (a new bone's is QUATERNION, where Blender ignores
  `rotation_euler`). The reply gives the transform as stored, the rotation as
  XYZ Euler angles and as a quaternion, and `world_head` and `world_tail`, where
  Blender evaluates the bone afterwards. Channels the bone locks
  (`lock_location`, `lock_rotation`, `lock_scale`) are written all the same, as
  `set_blender_object_transform` writes an object's: the locks only stop
  Blender's interactive transform tools. Key it with `insert_blender_keyframe`
  or `insert_blender_keyframes` on `pose.bones["Bone"].<property>`.
- `get_blender_object_trajectory` is not read-only: each sample is a frame
  change, so frame-change handlers run, and an unbaked rigid-body world is
  stepped on from the last frame its memory cache holds, which fills the
  cache. The frame and subframe current before the call are set again
  afterwards. `step` is at least 1 and `start_frame` not after `end_frame`.
  `time` counts seconds from the scene's `frame_start` (`time_origin_frame`) at
  `fps / fps_base`, and `velocity` is the world location change since the
  sample before, per second (null for the first sample of the request). Each
  point gives the world `location`, `rotation_euler` in `euler_order` (the
  target's Euler order, XYZ for a quaternion or axis-angle one), each sample
  the equivalent nearest the previous one, so it stays continuous while the
  target turns less than half a turn between samples, `rotation_quaternion`
  (w, x, y, z, its sign kept continuous) and `local_rotation`, the target's own
  rotation channels as its `rotation_mode` stores them, unwrapped: a joint
  keyed from 0 to 6 pi reads 6 pi there. The world rotation is the one nearest the world matrix, as
  `get_blender_object_info` reports it: a turned child of a parent with
  non-uniform scale has a sheared matrix, from which a plain Euler or
  quaternion read takes another rotation (0.190 rad for a child turned 30
  degrees under a parent scaled 3 on X). `fields` picks which of these a sample
  reports (fewer fit more samples a page). Values are the float32 Blender holds,
  written in their shortest form (0.1, not 0.10000000149011612).
- `bone_name` samples a pose bone of the armature instead: its world matrix is
  the armature's times the bone's pose matrix, and its location the bone's head.
  A bone's name passed as `object_name` fails with the armature to pass.
  `get_blender_trajectories` samples up to 32 objects and bones in one pass,
  setting each frame once for all of them, with the same samples, paging and
  side effects.
- A trajectory comes a page at a time. A page holds the samples that fit in
  one reply, measured as they are written, up to `limit`; `next_offset` (null on
  the last page) is the next call's `offset`, and the pages join into exactly
  what one long reply would hold, velocities at their edges included. A
  1 000-frame trajectory came back as frames 1-199 before; it is now 9 pages of
  about 120 samples with every field, or 2 with `fields: ["location"]` (596 a
  page). 32 targets with every field fit 3 frames a page. A page holds at
  least one sample of every target, so targets whose names and one sample
  each are more than a reply holds are refused: before any frame is set when
  even the narrowest samples would not fit, else once the first sample is
  measured. `details.targets_that_fit` counts how many of them, in order, fit
  with samples as wide as the requested fields make them, so that many fit
  every page. The server writes each non-ASCII character as a 6-byte escape:
  32 bones of an armature named with 127 accented letters made a page the
  server cut, dropping a target and skipping samples.
- For an unbaked rigid-body world, frames its memory cache already holds are
  read, not simulated again: the walk jumps to the cache's last frame and steps
  on from there (`simulated_frames` counts the frames it stepped). What the
  cache holds is read from its English info text; anything else (outdated,
  frames skipped, a translated interface) walks from the cache start. A page
  ends before a sample whose simulation would pass 200 000 bodies times frames
  (or `max_simulated_frames` when given), so each call does at most that much
  and the next page goes on. A request whose first sample alone needs more is
  refused before any frame is set, with what it would cost: bake first
  (`bake_blender_simulation`), sample fewer frames, or pass
  `max_simulated_frames` to go ahead. On Blender 5.0.1 on an M-series Mac a
  body-frame took about 35 microseconds headless and 61 in a GUI Blender, so the
  bound is some 12 s there, inside the 30 s an attached call waits by default
  (`BLENDER__CONNECTION_TIMEOUT`); a confirmed walk that takes longer keeps
  running in Blender after the call times out, and a later call reads what it
  cached. A 2-point request on 1 000 unbaked bodies over 1 000 frames took 29 s
  unasked; it is now refused at once, takes 35 s headless once confirmed, and
  0.01 s when repeated. Samples past the cache end come with a `warning`:
  Blender simulates nothing later and holds the bodies at their last cached
  pose.
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
- `export_simready_usd` writes an asset a simulator reads, not just Blender's
  geometry. After Blender's USD exporter runs, the export authors from each
  object's `simready_` metadata (`apply_simready_metadata`):
  - `is_rigid_body`: `UsdPhysics.RigidBodyAPI` on the object's prim;
    `mass_kg` and `density`: `MassAPI` (density converted from kg/m³ to the
    stage's units);
  - `collider_type`: `CollisionAPI` plus a `MeshCollisionAPI` approximation
    on the object's meshes (`convexHull`, `convexDecomposition`,
    `meshSimplification`, `none` for the triangle mesh, `boundingCube`,
    `boundingSphere`; the older `mesh`, `box`, `sphere` and `capsule` are
    written as `none`, `boundingCube`, `boundingSphere` and `convexHull`, as
    UsdPhysics has no capsule approximation of a mesh);
  - `static_friction`, `dynamic_friction`, `restitution`: a physics material
    (`UsdPhysics.MaterialAPI` on a `UsdShade.Material` under
    `<asset root>/PhysicsMaterials`) bound to those colliders with the
    `physics` purpose;
  - the semantic fields: `UsdSemantics.LabelsAPI` labels in SimReady's
    taxonomies `class`, `hierarchy` and `wikidata_qcode`, and each additional
    label's key (files from an earlier Simul carry the Q-code under `qcode`, so
    a reader should accept both). A Blender
    whose bundled USD has no UsdSemantics module gets Isaac Sim's
    older `SemanticsAPI:Semantics_<taxonomy>` attributes instead;
    `semantics.labels_schema` in the result says which;
  - each rigid body constraint between two exported objects
    (`add_blender_rigid_body_constraint`): a UsdPhysics joint under
    `<asset root>/Joints`. FIXED becomes a `FixedJoint`, POINT a
    `SphericalJoint`, HINGE a `RevoluteJoint` about the constraint's local Z,
    SLIDER a `PrismaticJoint` along its local X, and PISTON, GENERIC and
    GENERIC_SPRING a D6 `PhysicsJoint` with a `LimitAPI` per locked or limited
    axis (an axis Blender leaves unlimited stays free). `body0` is object1's
    rigid body (its own, the nearest exported one above it, or the object as a
    static body) and `body1` object2's; `localPos`/`localRot` place the
    constraint empty's frame in each body, `collisionEnabled` is the opposite
    of `disable_collisions` and `jointEnabled` follows `enabled`. Limits are
    written as UsdPhysics reads them, body1 relative to body0, in degrees:
    Blender measures a HINGE's and a GENERIC's angular limits the other way
    round (object1 relative to object2, measured on 5.0.1), so their range is
    negated; linear limits and a PISTON's angle keep their sign. The
    constraint empty itself is not written as a prim, so it neither adds a
    stray Xform nor forces a wrapper root. A MOTOR has no UsdPhysics joint
    (Blender's motor is a free joint whose drive is capped by an impulse per
    step), and springs and breaking thresholds have no exact equivalent: each
    is reported in `issues` and `warnings`, and `physics.joints` lists the
    joints written.

  `collider_type` and the friction values also apply to the meshes of
  descendants that set none of their own, so metadata on an asset root makes
  all its meshes collide; rigid body and mass stay on the object that sets
  them. The result's `warnings` name a rigid body without a collider, nested
  rigid bodies with no joint between them, a triangle-mesh collider on a
  dynamic body, and stored values that are not valid numbers, which are left
  out. The asset root is the default
  prim (kind `component`): the one top-level object, or an Xform named after
  the file when several objects share the top level. Blender's `/root`
  container goes, and the render materials move to `<asset root>/Looks`.
  `metersPerUnit` is the scene's Unit Scale, with the coordinates left in
  Blender units, `upAxis` is Z and `kilogramsPerUnit` 1. The result lists each
  written object's prim path in `prims`; `object_count` counts the objects
  written.

  Listed objects are exported with their descendants (`include_children`),
  also when they are hidden: the export shows them for Blender's exporter and
  hides them again. A descendant, or with `object_names` null a scene object,
  that is hidden in renders is left out and named in `skipped_objects`, as are
  lights and cameras; `include_lights_and_cameras: true` keeps those and the
  World's dome light. An empty `object_names` is refused. When validation finds
  an error, nothing is written and the call fails with `SimReadyValidationFailed`
  and the issues, unless `force: true`. Like `export_blender_file`, it writes
  `file_path` exactly (a missing folder is created) and lists `files_written`,
  applies modifiers, leaves Edit, Pose and paint modes for the export (so
  unsaved mesh edits are written) and enters them again, and restores the
  selection; a SimReady asset is static, so animation is left out. Blender
  writes next to the destination and the authored layer replaces the file only
  at the end; a `.usdz` is packaged from a `.usdc` with its textures. With
  `embed_metadata` (the default) the
  `simready_` properties stay as `userProperties` attributes too, and Blender's
  USD import reads them back as custom properties. The importer names an object
  merged with its single mesh after the mesh, and imports the `Looks` and
  `PhysicsMaterials` scopes as empties.
- `apply_simready_metadata` advertises the real fields of `metadata`
  (`semantic`, `physics`, `material`) in its input schema, `collider_type` as
  an enum. An unknown group or field is refused with the closest valid name
  (`physics.mass`: did you mean `mass_kg`?), and so is an invalid value; the
  call then writes nothing. A field left out or null keeps its stored value,
  `is_rigid_body: false` clears an earlier true, and `clear` removes stored
  values first: a field, a group, `additional_labels` or `all`. Label keys
  become semantics taxonomies, so they must be identifiers, and
  `semantic_qcode` must look like `Q1420`. An additional label named after a
  field's taxonomy (`class`, `hierarchy`, `wikidata_qcode`, or the older
  `qcode`) sets that field, and one that disagrees with the field given in the
  same call is refused, as it would overwrite the field's label in the USD.
  `get_simready_metadata` returns the stored values that meet those rules as
  `metadata`. The ones that do not (set by a script, an import or an older
  Simul) are listed in `invalid_values` with the problem, so one bad value does
  not fail the call.
- `apply_simready_metadata` and `setup_simready_hierarchy` need Object Mode:
  in Edit Mode Blender records their undo step on the mesh undo stack, which
  holds no object changes, so the first Ctrl+Z did nothing and the second also
  left Edit Mode. They also refuse objects linked from a library, whose changes
  are lost when the file is reopened. Their refusals, and those of the other
  SimReady tools, are `ValidationError`s whose `details.code` names the reason
  (`InvalidMetadata`, `ObjectModeRequired`, `LinkedData`, `RootExists`,
  `RootNotInvertible`, `EmptyObjectList` and so on); an unknown object is a
  `NotFoundError` with `details.code` `object_not_found`, as in every Blender
  tool, and a list naming several unknown objects lists them all in
  `details.objects`.
- `setup_simready_hierarchy` keeps every object where it is: it parents the
  listed objects that have no listed ancestor to the root, and a listed object
  under another listed object keeps its parent, which may be an unlisted
  object between them (`reparented` and `kept_parent` in the result). It keeps
  the object's matrix before constraints, and it computes it from the former
  parent's world matrix and the parent inverse matrix: an object Blender does
  not evaluate (in an excluded collection, or a collection or object disabled
  in viewports) has a stale `matrix_world`, which moved such objects to the
  origin. The transform moves into the object's own location, rotation and
  scale; where the former parent's non-uniform scale skews it, which those
  cannot express, the skew stays in its parent inverse matrix and `warnings`
  says so. An object that object constraints act on keeps its location,
  rotation and scale, and its former parent space goes into the parent inverse
  matrix instead: Blender's local space for a constraint is relative to parent
  world × parent inverse, so a local-space Limit Location (which moved such a
  part 1.5 m) sees the same values and gives the same pose. A post-check still
  warns if an evaluated pose moved. A bone or vertex parent becomes an object
  parent in place (one with constraints is refused). It checks every name, the
  root and the semantic labels before changing anything, and a failure
  part-way puts the parents back and removes the root it created. An existing
  object with the root's name is refused unless `reuse_root: true`, which
  needs an empty that can be inverted (not zero-scale), so a retry never makes
  `robot.001`. Root names are USD identifiers; camelCase is fine.
  `hierarchy_path` and `prim_paths` are the prim paths `export_simready_usd`
  writes; when two objects become the same prim name (`arm.L` and `arm_L`),
  `warnings` says so and the export renames one.
- `validate_simready_compliance` checks what `export_simready_usd` would
  export: the listed objects with their descendants (`include_children`), or
  with `object_names` null the scene's objects visible in renders, without
  lights and cameras; `skipped_objects` names the rest. It reports
  `compliant: false` for any issue of severity `error`; warnings never change
  it. The rules follow NVIDIA's SimReady Robotics-Prop profile
  (simready-foundation `fd677ad`) and Isaac Sim 5.1's asset validation; codes
  in brackets:

  | Check | Error | Warning |
  |---|---|---|
  | naming | | an object, material, light or camera name that is not a USD identifier (camelCase is fine); a mesh data name likewise; two siblings that become one prim name |
  | scale | | a largest real-world size (world-space size × Unit Scale) above 100 m or below 1 mm; a Unit Scale other than 1 [UN.007] |
  | transforms | the asset root, the prim that becomes the default prim, not the identity [VG.025]; a rigid body whose world transform is skewed, as a rotated part under a non-uniformly scaled group is [RB.009] | a scaled or mirrored part; a part skewed relative to its parent |
  | materials | no material | an empty slot that faces use |
  | hierarchy | | a mesh without a parent, or a root that is not an empty |
  | metadata | no semantic label of any taxonomy on the mesh or an exported object above it [SL.001]; a stored value that breaks the rules | no collider_type on the mesh or an exported object above it |
  | physics | a rigid body without mass or density [RB.007], without a collider, or nested in another rigid body with no joint the export writes between them [RB.012]; a rigid body constraint between exported objects that the export would drop: a MOTOR, or two objects that are parts of one rigid body | a triangle-mesh collider on a dynamic body; no rigid body at all [RB.001]; a constraint to an object the export leaves out, or between two static objects; springs, a breaking threshold, or several rotation axes limited to ranges (Blender reads them as Euler angles, PhysX as twist and swing) |
  | geometry | degenerate (zero-area) faces [VG.014] | no UV map; n-gons |

  Labels, colliders and physics count only from exported objects: an
  unexported ancestor is written as a bare Xform without them. The default
  prim is the top-most ancestor of the exported objects when they share one
  (Blender writes unexported ancestors too), so a lone off-origin mesh is a root
  error, and top-level objects that the export wraps in a new root are parts.
  Transforms are read from the world matrices the exporter writes, so scale in
  a parent inverse matrix counts; part rotation is not checked. Meshes are read
  with modifiers applied and with Edit Mode's edits written back
  (`update_from_editmode`). `severity` returns only errors or warnings, and
  `offset` / `limit` (50 by default, at most 500) page through the issues:
  `next_offset` is the next page's offset, null on the last. A page also stops
  before 30 KB of issues so that it fits the result budget; `issue_count`,
  `error_count` and `warning_count` count them all. `export_simready_usd` runs
  every check on the objects it exports; `skip_checks` leaves checks out, so
  that one that does not apply needs no `force`, which lets every error through.
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
- `setup_blender_rigid_body` checks every argument before it changes anything:
  Blender's body types and collision shapes (COMPOUND included), and its ranges
  (mass at least 0.001, friction 0 or more, restitution and damping 0-1).
  Blender itself stored a mass of -5, and the body fell through the ground. It
  needs a mesh object, where Blender's operator only said `poll() failed`. A
  failure removes the rigid body or world it added, or restores an existing
  body's settings, and the user's selection never changes: the operator runs
  under a context override. `kinematic: true` makes an animated body that
  follows its keys and pushes simulated ones. `collision_margin`,
  `collision_collections`, `use_deactivation` and `start_deactivated` change
  only when given.
- `add_blender_rigid_body_constraint` needs two different rigid bodies and
  checks the type and its settings first; a failure leaves no empty behind. The
  constraint empty is made directly rather than with `empty_add`, which follows
  the editor's "Align: View" preference and selects the empty. It sits at
  `location`, midway between the two objects' origins by default. Its local axes
  are the constraint's: a HINGE turns about local Z, a SLIDER or PISTON moves
  along local X, a MOTOR drives along and about local X, and GENERIC limits apply
  per local axis. `axis` turns the empty so that axis points along a world
  direction. Without it the empty is unrotated, so a HINGE turns about world Z,
  like a door; a pendulum swinging in the XZ plane needs `axis: [0, 1, 0]`.
  `limits`, `springs` (GENERIC_SPRING), the `motor_*` settings (MOTOR) and
  `breaking_threshold` configure it. A MOTOR drives object1 relative to
  object2: with object1 the anchor, a positive velocity turns or moves object2
  the negative way along the axis (on Blender 5.0.1, -1.91 rad in 23 frames at
  +2 rad/s, and -0.91 m at +1 m/s); swap the objects or the sign to reverse it.
  A MOTOR without a velocity drives nothing, and the reply warns about it. The
  empty, like a force field, goes where `create_blender_object` puts objects:
  the active collection, or the scene's collection when the active one (or one
  holding it) is excluded, hidden or disabled in viewports, or is linked from a
  library; `warnings` (`warning` for a field) then says so. The reply and `get_blender_constraint_info`
  report `pivot`, `axis`, `local_axes` (the directions the limits refer to) and
  the settings.
- `add_blender_force_field` adds the field with `align: WORLD` and an explicit
  rotation, so the editor's Align preference no longer turns it toward the 3D
  View. `direction` points the field's local +Z (where WIND blows, the axis
  VORTEX turns about, a PLANE's normal), and the reply reports it with
  `rotation_euler`. The types are Blender's: MAGNET (MAGNETIC is accepted for
  it), TEXTURE, GUIDE, BOID and FLUID (which Blender then reports as FLUID_FLOW)
  included. Shape, flow, falloff, maximum distance and noise can be set too.
  `get_blender_force_field_info` refuses an object whose field type is NONE,
  such as a collider, whose collision settings live in the same place.
- `bake_blender_simulation` refuses more than 20 000 frames or 400 000
  body-frames (frames x simulated objects) unless `allow_long_bake: true`. That
  is up to about 16 s on Blender 5.0.1, where 1000 bodies x 1000 frames took
  29.4 s and 401 cubes x 900 frames 14.8 s headless (1000 frames took 8.7 s in
  the GUI).
  A bake runs on Blender's main thread and cannot be stopped once it starts.
  Past the attached-mode client timeout (`BLENDER__CONNECTION_TIMEOUT`, 30 s by
  default), the call reports a timeout while Blender keeps baking, and the next
  calls wait behind it and can time out too. Raise the timeout before allowing
  a long bake; `get_blender_rigid_body_world` shows when it has finished
  (`rigid_body_cache.is_baked`).
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
| `get_tool_usage_stats` | Per-tool call counts, success rates and durations for the calling MCP session; `scope="server"` counts every session's calls. The recent call log (`include_recent`, `limit` 1 to 200) is always the caller's own, because records carry parameters, file paths and agent labels. Operators read every session with `simul stats`; clearing it is operator-only: `simul stats --reset` |

## Errors

Every failed call has the same shape, on every backend and whichever layer
refused it (the MCP input schema, a request model, the tool, the engine):

```json
{"success": false, "error_type": "ValidationError", "error": "width: Input should be less than or equal to 3840",
 "details": {"errors": [{"field": "width", "message": "...", "type": "less_than_equal"}], "input": {"width": 5000}}}
```

and it is sent with `isError: true`. `error_type` is always one of the names
below (`simul.mcp.error_types`): seven classes that mean the same on every
backend, and domain types where a backend says more than a class would.
Nothing else reaches a client. A library's or Python's exception name
(`ClientConnectorError`, `ConnectionResetError`, `KeyError`) is mapped onto
its family and kept in `details.exception_type`: a name with `Timeout` in it
is a `TimeoutError`; `Connection*` and `BrokenPipeError` a `ConnectionError`;
`*NotFound*`, `FileNotFoundError`, `KeyError` and `LookupError` a
`NotFoundError`; `ValueError`, `TypeError`, `JSONDecodeError`, `Invalid*` and
`*ValidationError` a `ValidationError`; `PermissionError` a
`PermissionDenied`; anything else (`OSError`, `RuntimeError`, `ImportError`,
`ModuleNotFoundError`) an `EngineError`.

An exception raised by code running inside Isaac Sim is the engine's,
whatever its name: a tool's own script that raises (a `TypeError` from a
changed Kit API, an `OSError` from a full disk, a `ModuleNotFoundError` for a
removed `omni.isaac.*` module) is an `EngineError`, and an agent's
`execute_isaac_script` that raises is a `ScriptError`, both naming it in
`details.exception_type`. Only the transports' own failures keep a class of
their own: the bridge's refusals (`StageUnavailable` is `InvalidState`,
`UnsafeExecutionDisabled` is `ScriptExecutionDisabled`, `PrimNotFound` is
`NotFoundError`, `RequestTimeout` is `TimeoutError`), an empty reply
(`ConnectionError`), a reply that is not JSON (`BridgeProtocolError`) and a
script the bridge stopped at its timeout (`TimeoutError`).

| `error_type` | Meaning | What to do |
|---|---|---|
| `ValidationError` | The arguments do not fit the tool: wrong type, missing or unexpected argument, out of bounds, not among the choices. `details.errors` lists each field; `details.code` names the check when a tool has several (USD's `InvalidPrimPath`, `InvalidAttribute`) | Fix the call |
| `NotFoundError` | What the call names (prim, actor, object, material, stage, instance, tool, file) does not exist | Check the name |
| `ConnectionError` | The engine could not be reached, or the connection broke before it answered; the operation may not have run | Check the engine is running; retry |
| `TimeoutError` | The engine did not answer in time; the operation may still be running | Check state before retrying |
| `EngineError` | The engine received the request and reported a failure the tool does not name more precisely (a non-domain exception, a reply that does not parse or does not match the tool's response schema, a partial failure under `*_error` keys) | Read `error` |
| `InstanceBusy` | The Isaac instance is serving another call | Retry shortly |
| `SandboxError` | A path or URL is outside the file sandbox; `details` names the allowed roots | Use an allowed path |

Domain types of the server:

| `error_type` | Meaning |
|---|---|
| `RateLimitError` | The agent spent its rate-limit budget; `retry_after_seconds` says when to retry. `release_isaac_instance` is never refused |
| `InstanceClaimed` | Another agent holds a live claim on the Isaac instance (claims enforced) |
| `ClaimStoreError` | The claim records could not be read or saved (check `$SIMUL_HOME/sessions`) |
| `SessionRequired` | The tool keeps per-client state and the request has no client identity |
| `PermissionDenied` | The caller may not act on what it named: another agent's claim, a file the process may not read or write |
| `ScriptExecutionDisabled` | `security.allow_script_execution` is off; the arbitrary-code tools (`execute_isaac_script`, ...) answer this instead of an unknown-tool error |
| `PayloadTooLarge` | A script, request or reply is larger than the transport accepts |

Domain types of the engines:

| `error_type` | Meaning |
|---|---|
| `RefusedOperation` | The tool refuses an operation that would damage the application or unsaved work without an override flag |
| `AuthenticationError` | The engine refused the credentials (python_server token, Remote Control passphrase) |
| `UnsupportedOperation` | The backend or its mode cannot do this (`attach_blender_window` outside attached mode) |
| `InvalidState` | Not possible in the current state (nothing claimed to release, a folder that already holds the import) |
| `ScriptError` | An agent's script raised inside the engine, whatever it raised (an `OSError` or a `ConnectionRefusedError` of its own included); `details.exception_type` names it, `details.traceback` says where |
| `EmptyOutput` | The engine's script printed nothing where the tool expected its reply |
| `InvalidAssetType` | The asset is not of the kind the tool needs |
| `UnsupportedFormat` | The file format is not one the tool reads or writes |
| `CaptureError` | A viewport capture failed |
| `ViewportUnavailable` | There is no viewport to read or drive (a headless editor) |
| `UiUnavailable` | The editor UI the tool drives is not available |
| `SimulationNotRunning` | The answer needs a running simulation (play or step once) |
| `ValidationFailed` | An asset check ran and found errors |
| `SimReadyValidationFailed` | A SimReady export was stopped by the validation it runs first |
| `PIEActive` | Unreal is in Play-In-Editor |
| `SpawnError` | Unreal could not spawn the actor from the class or asset given |
| `PluginUnavailable` | The optional Unreal plugin the tool needs is not enabled |
| `IncompatibleBody` | The Unreal body cannot simulate as asked; `details.problems` says why |
| `EditorExited` | The Unreal editor exited while `simul unreal setup` waited for it |
| `BridgeUnavailable` | The Isaac Sim bridge extension the tool needs is not configured |
| `BridgeProtocolError` | A transport (Isaac Sim's bridge or stock socket, Blender's bridge) answered outside its protocol: another version, a reply that is not JSON, malformed or unmatched; the request may have run |
| `ResponseEncodingError` | Blender's bridge could not encode the result; the operation may have run |
| `AttachmentError`, `AttachmentStale`, `AttachmentTargetChanged` | The attached Blender window could not be used, its process went away, or it loaded another file |
| `UnsavedChangesError` | Blender refuses to replace a file with unsaved changes without the flag |
| `KeyframesPartlyInserted` | Blender refused a key part way through a batch; the keys before it stay, and `details` gives how many went in and the one refused |
| `LoadError` | A headless USD file could not be opened; `details.reason` says why |
| `StageBusy` | A headless USD stage is serving a call that cannot run beside this one; retry shortly |
| `ComputationError` | A headless USD computation failed |
| `LogFileNotFound` | `simul logs` found no log file |
| `*NotApplied` | The read-back family (`TransformNotApplied`, `MaterialNotApplied`, `RenderSettingNotApplied`, ...): the engine took the request, but what it reads back afterwards is not what was asked. The payload keeps the values read back, with `details.expected` and `details.actual` where the tool reports them. Match the family by the suffix |

Each entry of a `ValidationError`'s `details.errors` has the `field` (a dotted
path for a nested model's field, `metadata.physics.mass`), the `message` and
pydantic's `type`, and when they apply the `constraint` that failed
(`{"le": 3840}`, `{"min_length": 3}`, `{"expected": "'DomeLight', ... or
'CylinderLight'"}`), the `input` given and a `suggestion`: the closest valid
choice, or the closest argument name for an unexpected one. A bpy enum
refusal (`enum "BOXX" not found in ('BOX', 'SPHERE')`) gets the same
`constraint`, `input` and `suggestion`.

`details.input` (and each error's `input`) echoes the arguments bounded: a
string longer than 256 characters by its size, a list or mapping by its first
16 entries and a count of the rest (`"<19984 more items, 20000 in all>"`), at
most 64 values per argument. A value whose name has a word that names a
secret is echoed as `***`: an argument, a mapping key or a settings path, at
any depth (`token`, `password`, `passphrase`, `passwd`, `pwd`, `secret`, `key`,
`api_key`, `auth`, `authorization`, `credential`, `cookie`, `bearer`, `jwt`,
`pin`; `access_key`, `apiKey`, `/persistent/auth_token`, but not `author` or
`keyframe`). `error` and each `details.errors` message are cut at 2 000
characters (`"... <38042 more characters>"`), so a message that quotes a long
argument stays small too.

Arguments that do not fit the input schema are refused before the tool runs,
but through the same envelope as every other failure: the call spends a
rate-limit token and is recorded in the usage log and the audit log. A call to
a tool that is not registered is a `NotFoundError`, recorded and rate limited
under `<unknown tool>` with the name it gave in the record's parameters. The
audit row of every call states the envelope's outcome, `error_type` included
(see [Audit log](configuration.md#audit-log)).

A call the client cancels may still have run: cancelling the request does not
stop the engine. Its usage and audit rows say `cancelled` and `executed`:
`false` (it never started), `true` (Isaac: the server waited for the engine,
and the row keeps the engine's outcome) or `"unknown"` (Unreal, Blender: the
request had been sent). Check the scene before repeating a cancelled call.

## Annotations

Every tool's `readOnlyHint`, `destructiveHint` and `idempotentHint` follow one
rule on every backend, pinned for all tools by
`tests/mcp/test_tool_annotation_table.py`:

- **read-only**: changes nothing, a capture that only returns the image
  included (`capture_blender_viewport`);
- **destructive**: overwrites or deletes existing state or files, or runs
  arbitrary code (`capture_isaac_viewport` deletes the oldest captures in its
  directory, so it is destructive; `capture_unreal_viewport` writes a new
  screenshot and is only additive);
- **idempotent**: repeating the call with the same arguments changes nothing
  more (a set, a delete, a save or export to the same path, a focus); a
  create, a step, a capture that writes a new file or an edit that compounds
  (simplify, subdivide) is not.

## MCP resources

| URI | Contents |
|---|---|
| `simul://isaac-sim/skills` | Isaac Sim 5.1 / 6.0 scripting patterns for `execute_isaac_script` |
| `simul://isaac-sim/api/{core,sensors,physics,replicator,robots,rendering,assets}` | Isaac Sim API references by area |
