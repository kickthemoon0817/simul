# Isaac Sim 5.1.0 Namespace Migration

Isaac Sim 5.1.0 migrated the core simulation package from `omni.isaac.*` to `isaacsim.*`. The old names raise `ModuleNotFoundError`. Always use the new names.

## Package Renames

| Old import (pre-5.1) | New import (5.1.0+) |
|----------------------|----------------------|
| `omni.isaac.core` | `isaacsim.core.api` |
| `omni.isaac.core.utils` | `isaacsim.core.utils` |
| `omni.isaac.core.utils.prims` | `isaacsim.core.utils.prims` |
| `omni.isaac.core.utils.stage` | `isaacsim.core.utils.stage` |
| `omni.isaac.core.utils.xforms` | `isaacsim.core.utils.xforms` |
| `omni.isaac.core.utils.bounds` | `isaacsim.core.utils.bounds` |
| `omni.isaac.core.utils.rotations` | `isaacsim.core.utils.rotations` |
| `omni.isaac.core.utils.collisions` | `isaacsim.core.utils.collisions` |
| `omni.isaac.core.utils.viewports` | `isaacsim.core.utils.viewports` |
| `omni.isaac.core.world` | `isaacsim.core.api` (World class) |

## Unchanged Kit-Level APIs

These never moved — use them exactly as before:

| Module | Purpose |
|--------|---------|
| `omni.usd` | Stage context, selection |
| `omni.timeline` | Playback control |
| `pxr.Usd` | Core USD (stage, prims, layers) |
| `pxr.UsdGeom` | Geometry schemas (Mesh, Cube, Sphere, Camera, XformOp) |
| `pxr.UsdPhysics` | Physics schemas (RigidBodyAPI, CollisionAPI, Scene, joints) |
| `pxr.UsdShade` | Shading (Material, Shader, MaterialBindingAPI) |
| `pxr.UsdLux` | Lights (DistantLight, DomeLight, SphereLight) |
| `pxr.Gf` | Math types (Vec3d, Vec3f, Quatf, Matrix4d) |
| `pxr.Sdf` | Schema definition (Path, ValueTypeNames) |
| `pxr.PhysxSchema` | NVIDIA PhysX extensions (PhysxRigidBodyAPI, etc.) |

## Class Renames

| Old class | New location |
|-----------|-------------|
| `omni.isaac.core.World` | `isaacsim.core.api.World` |
| `omni.isaac.core.objects.DynamicCuboid` | Use `create_prim` + `add_isaac_rigid_body` MCP tools instead |
| `omni.isaac.core.objects.GroundPlane` | Use `create_prim` with type `"Plane"` instead |

## Correct Import Examples

```python
# World singleton
from isaacsim.core.api import World
world = World.instance()

# Prim utilities
from isaacsim.core.utils.prims import create_prim, delete_prim, get_prim_at_path
from isaacsim.core.utils.prims import get_prim_attribute_value, set_prim_attribute_value

# Stage utilities
from isaacsim.core.utils.stage import open_stage, save_stage, create_new_stage
from isaacsim.core.utils.stage import add_reference_to_stage

# Transform utilities
from isaacsim.core.utils.xforms import get_world_pose, get_local_pose

# Bounds
from isaacsim.core.utils.bounds import create_bbox_cache, compute_aabb

# Rotations
from isaacsim.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles

# Viewport
from isaacsim.core.utils.viewports import set_camera_view, set_active_viewport_camera

# Collisions / raycast
from isaacsim.core.utils.collisions import ray_cast
```
