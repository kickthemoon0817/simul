---
name: usd-operations
description: This skill should be used when the user asks to "analyze a USD file", "load a USD file", "inspect USD", "headless USD", "prim info", "search prims", "mesh info", "scene summary", "bounding box", or needs to work with USD files without a running Isaac Sim instance.
version: 0.1.0
---

# Headless USD File Operations

This skill teaches you how to load, inspect, analyze, and modify USD files using the headless USD tools. These tools operate on **local files** without requiring a running Isaac Sim instance. They use a `stage_id` handle returned by `load_usd_file`.

## Critical Distinction

| Tool set | When to use | Identifier |
|---|---|---|
| `load_usd_file`, `get_prim_info`, `search_prims`, `get_mesh_info`, `get_bounding_box`, `summarize_scene`, `create_prim`, `update_prim_attributes`, `delete_prim` | Offline/headless USD file analysis and editing | `stage_id` |
| `get_isaac_prim_detail` (aspects: info, mesh, ...), `search_isaac_prims`, etc. | Live Isaac Sim scene (running instance) | prim path only |

**Never mix these two sets.** If Isaac Sim is not running, use the headless tools with `stage_id`. If Isaac Sim is running with a stage open, use the `isaac_*` tools.

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

Returns type, attributes (with values), children, and applied schemas.

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

| Value | Behavior |
|---|---|
| `by_type` | Match USD schema type: `Mesh`, `Xform`, `Scope`, `Material`, `Camera`, `DistantLight`, `SphereLight`, `RectLight` |
| `by_name` | Substring match on prim name (case-insensitive) |

### 5. Analyze Mesh Geometry

Get detailed mesh topology for a Mesh prim: vertex count, face count, normals, UV sets, and bounds.

```
mcp__simul__get_mesh_info
  stage_id: "abc123"
  prim_path: "/World/Robot/base_link/visuals/mesh"
```

Returns: `{ "vertex_count": 2048, "face_count": 1024, "has_normals": true, "uv_sets": ["st"], "extent": [...] }`

Use this to assess mesh complexity before simulation (high-poly meshes should use convex decomposition for collision).

### 6. Get Bounding Box

Get the spatial extents of a prim in world or local space.

```
mcp__simul__get_bounding_box
  stage_id: "abc123"
  prim_path: "/World/Robot"
  world_space: true
```

Returns: `{ "min": [x, y, z], "max": [x, y, z], "center": [x, y, z], "size": [w, h, d] }`

Set `world_space: false` to get local (object-space) extents.

### 7. Summarize the Scene

Get a full overview of the scene contents without inspecting prim by prim.

```
mcp__simul__summarize_scene
  stage_id: "abc123"
  include_meshes: true
  format: "text"
```

Use `format: "json"` for structured output when parsing programmatically. Use `format: "text"` for human-readable summaries.

`include_meshes: true` adds per-mesh vertex/face counts to the summary. Set to `false` for faster results on large scenes.

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

**Note:** Modifications via these tools write back to the loaded stage. To persist changes to disk, the stage must be saved — use `execute_isaac_script` with `stage.Export(path)` if Isaac Sim is running, or be aware that headless modifications may require a separate save step depending on server configuration.

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
- **Stage ID is session-scoped**: The `stage_id` is not persistent across tool sessions. If the session resets, call `load_usd_file` again.
- **Large scenes**: `summarize_scene` with `include_meshes: true` on a scene with thousands of meshes can be slow. Use `include_meshes: false` for a quick overview first.
- **`by_name` is substring match**: `search_prims` with `by_name: "link"` will match `base_link`, `arm_link_1`, etc. Use more specific queries to narrow results.
