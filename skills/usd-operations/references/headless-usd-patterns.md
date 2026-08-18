# Headless USD Patterns Reference

## Stage ID Workflow

The `stage_id` is the central identifier for all headless USD operations. It is returned by `load_usd_file` and must be passed to every subsequent call.

```
load_usd_file(file_path) → { stage_id: "abc123", ... }
    ↓
get_prim_info(stage_id="abc123", prim_path="/World/Robot")
search_prims(stage_id="abc123", search_type="by_type", query="Mesh")
get_mesh_info(stage_id="abc123", prim_path="/World/Robot/mesh")
get_bounding_box(stage_id="abc123", prim_path="/World/Robot")
summarize_scene(stage_id="abc123")
create_prim(stage_id="abc123", prim_path="/World/NewPrim", prim_type="Xform")
update_prim_attributes(stage_id="abc123", prim_path="...", attributes={...})
delete_prim(stage_id="abc123", prim_path="/World/OldPrim")
```

**Session scope:** The `stage_id` is valid for the duration of the current tool session. If the session resets, call `load_usd_file` again to get a new handle.

## Prim Type Filters for search_prims

Use these exact strings with `search_type: "by_type"`:

| Type String | USD Schema | Common Use |
|---|---|---|
| `"Mesh"` | `UsdGeom.Mesh` | All polygon geometry |
| `"Xform"` | `UsdGeom.Xform` | Transform groups, robot links |
| `"Scope"` | `UsdGeom.Scope` | Organizational containers |
| `"Material"` | `UsdShade.Material` | Material definitions |
| `"Shader"` | `UsdShade.Shader` | Shader nodes inside materials |
| `"Camera"` | `UsdGeom.Camera` | Camera prims |
| `"DistantLight"` | `UsdLux.DistantLight` | Sun/directional lights |
| `"SphereLight"` | `UsdLux.SphereLight` | Point/sphere lights |
| `"RectLight"` | `UsdLux.RectLight` | Area lights |
| `"DiskLight"` | `UsdLux.DiskLight` | Disk-shaped area lights |
| `"Cylinder"` | `UsdGeom.Cylinder` | Procedural cylinders |
| `"Cube"` | `UsdGeom.Cube` | Procedural cubes |
| `"Sphere"` | `UsdGeom.Sphere` | Procedural spheres |
| `"PhysicsScene"` | `UsdPhysics.Scene` | Physics scene descriptor |
| `"ArticulationRoot"` | `UsdPhysics.ArticulationRootAPI` | Robot articulation roots |

## Mesh Topology Analysis Patterns

### Count total polygons across all meshes

1. `search_prims(stage_id, "by_type", "Mesh")` — get all mesh paths
2. For each path, call `get_mesh_info(stage_id, path)`
3. Sum `vertex_count` and `face_count` fields

### Identify high-poly meshes

After getting all mesh paths, call `get_mesh_info` on each and filter for `face_count > 10000`. These are candidates for LOD reduction or convex hull approximation before physics simulation.

### Check UV coverage

`get_mesh_info` returns a `uv_sets` list. If the list is empty, the mesh has no UV coordinates — texture mapping will not work on it.

### Assess normals

If `has_normals: false`, the mesh will render with flat shading. This is common for collision meshes that are not meant to be visible.

## Bounding Box Patterns

### World space vs local space

- `world_space: true` — extents after applying all parent transforms. Use this for placement decisions ("will this fit in the shelf?").
- `world_space: false` — extents in the prim's own coordinate frame. Use this for the asset's intrinsic size.

### Size calculation

The `size` field in the bounding box response is `max - min` per axis: `[width, depth, height]` (X, Y, Z). For a Z-up stage, Z is height.

### Centering an asset at origin

After getting the bounding box center, offset the translation so the center lands at [0, 0, 0]:
```
translation = [-center[0], -center[1], -center[2]]
update_prim_attributes(stage_id, root_path, {"xformOp:translate": translation})
```

## search_type Options

| Value | Behavior | Notes |
|---|---|---|
| `"by_type"` | Match USD schema type exactly | Use type strings from table above |
| `"by_name"` | Substring match on prim name | Case-insensitive; matches the prim's own name, not full path |

For `by_name`, the prim name is the last component of the path. Searching for `"link"` will match prims named `base_link`, `arm_link_1`, but not a prim at `/World/links/part`.

## Common Multi-Step Analysis Script

"Summarize this USD file: count meshes, total polygons, bounding box, and list cameras"

```
1. validate_usd_file(file_path)
   → check for errors

2. load_usd_file(file_path)
   → stage_id = "abc123"

3. summarize_scene(stage_id="abc123", include_meshes=true, format="json")
   → prim_count, hierarchy overview

4. search_prims(stage_id="abc123", search_type="by_type", query="Mesh")
   → mesh_paths = ["/World/Mesh0", "/World/Mesh1", ...]

5. For each path in mesh_paths (up to 20):
   get_mesh_info(stage_id="abc123", prim_path=path)
   → accumulate vertex_count, face_count

6. get_bounding_box(stage_id="abc123", prim_path="/", world_space=true)
   → overall scene dimensions

7. search_prims(stage_id="abc123", search_type="by_type", query="Camera")
   → camera_paths

8. Report: mesh count, total vertices, total faces, scene dimensions, camera list
```

## Headless vs Live Isaac Sim — Tool Mapping

| Operation | Headless (file) | Live Isaac Sim |
|---|---|---|
| Load/open | `load_usd_file` | `open_isaac_stage` |
| Prim info | `get_prim_info(stage_id, path)` | `get_isaac_prim_detail(path, aspects=["info"])` |
| Search | `search_prims(stage_id, ...)` | `search_isaac_prims(...)` |
| Mesh info | `get_mesh_info(stage_id, path)` | `get_isaac_prim_detail(path, aspects=["mesh"])` |
| Bounding box | `get_bounding_box(stage_id, path)` | `get_isaac_prim_detail(path, aspects=["bounding_box"])` |
| Scene summary | `summarize_scene(stage_id)` | `get_isaac_scene_summary()` |
| Create prim | `create_prim(stage_id, path, type)` | `create_isaac_prim(path, type)` |
| Delete prim | `delete_prim(stage_id, path)` | `delete_isaac_prim(path)` |
