---
name: usd-operations
description: Use when the user asks to "analyze a USD file", "load a USD file", "inspect USD", "headless USD", "prim info", "search prims", "mesh info", "scene summary", "bounding box", or needs to work with USD files without a running Isaac Sim instance.
version: 0.1.0
---

# Headless USD File Operations

This skill teaches you how to load, inspect, analyze, and modify USD files using the headless USD tools. These tools operate on **local files** without requiring a running Isaac Sim instance. They use a `stage_id` handle returned by `load_usd_file`.

## Critical Distinction

| Tool set | When to use | Identifier |
|---|---|---|
| `load_usd_file`, `get_prim_info`, `get_prim_subtree`, `search_prims`, `get_mesh_info`, `get_bounding_box`, `summarize_scene`, `create_prim`, `update_prim_attributes`, `delete_prim`, `list_usd_stages`, `save_usd_stage`, `export_usd_stage`, `reload_usd_stage`, `unload_usd_stage` | Offline/headless USD file analysis, editing and saving | `stage_id` |
| `get_isaac_prim_detail` (aspects: info, mesh, ...), `search_isaac_prims`, etc. | Live Isaac Sim scene (running instance) | prim path only |

**Never mix these two sets.** If Isaac Sim is not running, use the headless tools with `stage_id`. If Isaac Sim is running with a stage open, use the Isaac tools (names containing `isaac`, such as `get_isaac_prim_detail`).

## Tool Chain Overview

```
validate_usd_file             (quick check before loading)
    → load_usd_file           (open file, returns stage_id)
        → get_prim_info       (inspect a specific prim)
        → search_prims        (find prims by type or name)
        → get_mesh_info       (vertex/face/UV analysis)
        → get_bounding_box    (spatial extents)
        → summarize_scene     (full scene overview)
        → create_prim         (add new prim)
        → update_prim_attributes  (modify attributes)
        → delete_prim         (remove prim)
```

## Step-by-Step Workflow

### 1. Validate the File (Optional Fast Check)

Before loading, validate the file for USD errors without fully opening the stage.

```
mcp__simul__validate_usd_file
  file_path: "/home/user/assets/scene.usd"
```

Returns a list of errors and warnings. If the file has critical errors, loading may fail or produce incomplete results.

### 2. Load the File

Open the USD file and receive a `stage_id` handle. All subsequent headless operations require this ID.

```
mcp__simul__load_usd_file
  file_path: "/home/user/assets/scene.usd"
```

Returns: `{ "stage_id": "abc123", "prim_count": 142, "root_prims": [...] }`

**Important:** The `stage_id` is a temporary handle for this session. Save it — you will pass it to every subsequent call.

### 3. Inspect a Specific Prim

Get the properties, attributes, and metadata of a single prim by path.

```
mcp__simul__get_prim_info
  stage_id: "abc123"
  prim_path: "/World/Robot/base_link"
```

Returns type, applied API schemas, bounding box, transform, children, attributes, relationships and metadata.

- Each attribute is `{ "type": "point3f", "value": [0, 0, 0.2], "authored": true }`. `authored: false` means the value is the schema fallback, not something the file sets.
- Values are plain JSON: vectors are lists, quaternions `[w, x, y, z]`, matrices lists of rows, and an infinite or NaN float the string `"Infinity"`, `"-Infinity"` or `"NaN"`.
- Arrays carry `length` and only their first `array_preview` elements (default 16), with `truncated: true` when cut.
- Narrow a large prim with `attribute_filter: ["physics:*"]`, and read animated values with `time_code`.
- `relationships` holds `{targets, authored}` per relationship: joint `physics:body0`/`body1`, `material:binding`, `physics:filteredPairs`.
- `metadata` holds `specifier`, `instanceable`, `composition` (references, payloads, inherits and specializes, each with `resolved`), `variant_sets`, `opinion_layers` and other authored metadata.

### 4. Search for Prims

Find prims by type or name pattern across the entire stage.

```
mcp__simul__search_prims
  stage_id: "abc123"
  search_type: "by_type"
  query: "Mesh"
```

```
mcp__simul__search_prims
  stage_id: "abc123"
  search_type: "by_name"
  query: "wheel"
```

**`search_type` options:**

| Value | Behavior (`match: "auto"`) |
|---|---|
| `by_type` | A registered type matches its whole schema family: `PhysicsJoint` finds every joint, `Gprim` every geometric primitive, `Boundable` everything with extent. `match: "exact"` for the exact type only |
| `by_api_schema` | Applied API schemas: `PhysicsRigidBodyAPI`, `PhysicsCollisionAPI`; `PhysicsDriveAPI` also finds `PhysicsDriveAPI:angular` |
| `by_name` | Substring of the prim name, case-insensitive; an empty query lists every prim |
| `by_path` | Substring of the full path; use `match: "glob"` for `/World/*/joints/*` |

`match` can also be `exact`, `substring`, `glob` or `regex`; pass `case_sensitive: true` to respect case. Instances are searched; `root_path` limits the search to a subtree. Results are paged: `limit` (default 200), `total_matches`, and `next_offset` to pass as the next `offset`.

### 4b. List a Subtree

Read a robot or asset in one call instead of one `get_prim_info` per prim:

```
mcp__simul__get_prim_subtree
  stage_id: "abc123"
  root_path: "/World/Robot"
  include_properties: true
```

Each prim has `path`, `type`, `depth`, `child_count` and `api_schemas`. With `include_properties`, each prim also carries its authored attributes and relationship targets, such as joint `physics:body0`/`body1`. Paged with `max_results`/`offset`.

### 5. Analyze Mesh Geometry

Get detailed mesh topology for a Mesh prim: vertex count, face count, normals, UV sets, and bounds.

```
mcp__simul__get_mesh_info
  stage_id: "abc123"
  prim_path: "/World/Robot/base_link/visuals/mesh"
```

Returns: `{ "vertex_count": 2048, "face_count": 1024, "has_normals": true, "normals_source": "primvars:normals", "uv_sets": ["st"], "is_closed": true, "surface_area": ..., "volume": ..., "bbox": {...}, "materials": [...], "physics_material": ..., "subsets": [...] }`

- `materials` lists the material bound to the mesh, then those bound to its `materialBind` geometry subsets; each such subset also carries its `material`.
- `volume` is null unless the mesh is closed. Area, volume and bounds are in the mesh's own space, before transforms.
- Pass `time_code` for a deforming mesh: `points_time_samples > 0` means its points are animated, and at the default time it may have none.

Use this to assess mesh complexity before simulation (high-poly meshes should use convex decomposition for collision).

### 6. Get Bounding Box

Get the spatial extents of a prim in world or local space.

```
mcp__simul__get_bounding_box
  stage_id: "abc123"
  prim_path: "/World/Robot"
  world_space: true
```

Returns: `{ "bbox": { "min": [x, y, z], "max": [x, y, z], "center": [x, y, z], "size": [w, h, d], "volume": v }, "has_geometry": true, "included_purposes": ["default", "render"] }`

Set `world_space: false` to get local (object-space) extents. Only geometry whose purpose is in `include_purposes` counts (default: `default` and `render`); add `"guide"` to bound collision shapes authored as guides. A prim with no such geometry returns `has_geometry: false` with `bbox: null`, and `excluded_purposes_with_geometry` says which purposes would find some. Pass `time_code` to bound an animated prim at a given time.

### 7. Summarize the Scene

Get a full overview of the scene contents without inspecting prim by prim.

```
mcp__simul__summarize_scene
  stage_id: "abc123"
  include_meshes: true
  format: "text"
```

Use `format: "json"` for structured output when parsing programmatically. Use `format: "text"` for human-readable summaries.

`include_meshes: true` adds mesh totals (vertices, faces, normals, UVs, closed meshes, area, volume) to the summary; it is fast even on large scenes.

- `scene_size_meters` converts the bounds with the stage's `metersPerUnit`.
- `animation_info.duration_seconds` is the animated span in seconds. Animation is detected from time samples too, when no time range is authored.
- `warnings` flags referenced or sublayered files whose `upAxis` or `metersPerUnit` differs from the stage's: such an asset shows up scaled or rotated, because USD does not convert it.

### 8. Modify the USD File

The headless tools support basic creation, modification, and deletion.

**Create a prim:**
```
mcp__simul__create_prim
  stage_id: "abc123"
  prim_path: "/World/NewXform"
  prim_type: "Xform"
```

**Update attributes:**
```
mcp__simul__update_prim_attributes
  stage_id: "abc123"
  prim_path: "/World/NewXform"
  attributes: {"xformOp:translate": [1.0, 2.0, 0.5]}
```

**Delete a prim:**
```
mcp__simul__delete_prim
  stage_id: "abc123"
  prim_path: "/World/UnwantedPrim"
```

Every edit is all or nothing: a call that fails changes nothing, and its
error names the prim, each rejected attribute and the reason.

- Paths are absolute (`/World/Box`, never `World/Box`).
- `create_prim` takes a concrete registered type (`Xform`, `Cube`, ...) and
  refuses an existing path unless `redefine_existing: true`.
- `active`, `instanceable` and `kind` in `attributes` set prim metadata.
- Values take the attribute's type: quaternions are `[w, x, y, z]`,
  matrices are lists of rows, and a new `xformOp:translate` is a `double3`
  op added to `xformOpOrder`. Name a new custom attribute's type with
  `attribute_types: {"custom:offset": "double3"}`.
- An animated attribute needs `time_code: <frame>`; without it the call is
  refused, because a default value would be hidden by the time samples.
- `delete_prim` refuses `/`, and the default prim unless
  `allow_root_delete: true`. A prim that also comes from a sublayer or a
  reference cannot be deleted from the root layer; deactivate it with
  `attributes: {"active": false}` instead.

**Saving:** edits stay in memory until you write them, and are lost when the server exits.

- `save_usd_stage` (`stage_id`) writes the edits back to the stage's own file. It refuses a `.usdz`, and a file that changed on disk since it was loaded (including another session's save of it) unless `overwrite_external_changes: true`.
- `export_usd_stage` (`stage_id`, `file_path`, optional `flatten`, `overwrite`) writes a new `.usda`/`.usdc`/`.usd`/`.usdz`.
- `list_usd_stages` shows which stages have unsaved edits.
- `reload_usd_stage` / `unload_usd_stage` take `discard_unsaved: true` to drop edits. A reload lists a changed sublayer that no longer parses in `failed_files`; fix the file and reload again.
- A file that references itself by file name (`@./self.usda@</Proto>`) loads with a `SelfReference` warning: that arc shows the file on disk, so unsaved edits appear through it only after a save and a reload.

## Common Analysis Workflows

### "Analyze this USD file and tell me what's in it"
1. `validate_usd_file` — check for errors
2. `load_usd_file` → get `stage_id`
3. `summarize_scene` with `format: "text"` — full overview
4. `search_prims` with `by_type: "Mesh"` — list all meshes
5. Report findings

### "How many polygons does this robot have?"
1. `load_usd_file` → `stage_id`
2. `search_prims` with `by_type: "Mesh"` — find all mesh prims
3. `get_mesh_info` for each mesh — accumulate vertex/face counts
4. Sum and report

### "What is the size of this asset?"
1. `load_usd_file` → `stage_id`
2. `get_bounding_box` on the root prim with `world_space: true`
3. Report `size` field

## Decision Guide

| User Intent | Tool to Use |
|---|---|
| "Is this USD file valid?" | `validate_usd_file` |
| "Load and inspect a USD file" | `load_usd_file` → `summarize_scene` |
| "Find all meshes in the file" | `search_prims` with `by_type: "Mesh"` |
| "How many vertices in this mesh?" | `get_mesh_info` |
| "What are the dimensions?" | `get_bounding_box` |
| "Inspect prim at path X" | `get_prim_info` |
| "Add a new prim to the file" | `create_prim` |
| "Change an attribute value" | `update_prim_attributes` |

## Common Pitfalls

- **Using headless tools on a live stage**: Do not use `get_prim_info` (headless) when Isaac Sim is running and you want live data — use `get_isaac_prim_detail` with `aspects=["info"]` instead.
- **Forgetting `stage_id`**: Every headless call requires the `stage_id` from `load_usd_file`. Calling `get_prim_info` without it will fail.
- **Stage ID is session-scoped**: A `stage_id` belongs to the MCP session that loaded it (other sessions get `NotFoundError`). When an HTTP session ends, its stages are unloaded with their unsaved edits, and no stage survives a server restart. Save edits you need with `save_usd_stage` or `export_usd_stage` before the session ends; after a reset, call `load_usd_file` again.
- **Large scenes**: `summarize_scene` with `include_meshes: true` on a scene with thousands of meshes can be slow. Use `include_meshes: false` for a quick overview first.
- **`by_name` is a case-insensitive substring match** by default: `"link"` matches `base_link`, `ARM_LINK_1`, etc. Use `match: "exact"`, `"glob"` or `"regex"` (and `case_sensitive`) to narrow it.
