# Isaac Sim Namespace Migration (5.1 → 6.0)

Isaac Sim 5.1.0 migrated the core simulation package from `omni.isaac.*` to `isaacsim.*`. On 5.1 the old names still resolve through deprecation shims; on 6.0 the shims are gone and the old names raise `ModuleNotFoundError`. Always use the new names.

Isaac Sim 6.0.0 / 6.0.1 (Kit 110.1, Python 3.12) then deprecated the 5.1 core layer itself. Check the running version first:

```python
import json, omni.kit.app
print(json.dumps({"isaac": omni.kit.app.get_app().get_app_version()}))   # "6.0.1"
```

## 6.0 deprecations (still importable, but scheduled for removal)

| Deprecated in 6.0 | Use instead |
|-------------------|-------------|
| `isaacsim.core.api` (`World`, `SimulationContext`) | `isaacsim.core.simulation_manager`, `isaacsim.core.experimental.*` |
| `isaacsim.core.prims` (`XFormPrim`, `RigidPrim`, `Articulation`) | `isaacsim.core.experimental.prims` |
| `isaacsim.core.utils.stage` | `isaacsim.core.experimental.utils.stage` (`create_new_stage_async(template="empty")`, `open_stage`, `add_reference_to_stage`) |
| `isaacsim.core.utils.prims` | `isaacsim.core.experimental.utils.prim` or plain `pxr.Usd` |
| `isaacsim.core.utils.xforms`, `.rotations`, `.bounds` | `isaacsim.core.experimental.utils.xform` / `.ops`, or `pxr.Gf` / `UsdGeom.BBoxCache` |
| `isaacsim.sensors.camera`, `isaacsim.sensors.rtx` | `isaacsim.sensors.experimental.rtx` |
| `isaacsim.sensors.physics`, `isaacsim.sensors.physx` | `isaacsim.sensors.experimental.physics` |
| `isaacsim.robot.wheeled_robots`, `isaacsim.robot.manipulators` | `isaacsim.robot.experimental.wheeled_robots`, manipulator experimental APIs |
| `isaacsim.replicator.domain_randomization`, `isaacsim.replicator.mobility_gen` | `isaacsim.replicator.experimental.*` |
| `isaacsim.util.merge_mesh` | `omni.scene.optimizer.core` |
| `omni.physx.get_physx_interface().get_physics_stats()` / `.is_cuda_lib_present()` | removed; read `/physics/*` Carb settings instead |

Removed outright in 6.0: every `omni.isaac.*` shim, `isaacsim.asset.browser` (use `omni.simready.content.browser`), `isaacsim.replicator.scene_blox`, `isaacsim.app.selector`.

## Version-independent choices

`pxr.*`, `omni.usd`, `omni.timeline`, `omni.kit.app`, `omni.kit.commands`, `omni.kit.viewport.utility`, and `omni.replicator.core` behave the same on 5.1 and 6.0. Prefer them over `isaacsim.core.*` when a script must run on both; simul's granular tools are written that way.

Do not use `asyncio.wait_for` or `asyncio.timeout` inside a script: the 6.0 `isaacsim.code_editor.python_server` drives top-level `await` outside an asyncio Task and both raise `RuntimeError: Timeout should be used inside a task`. Bound waits with a counted `await omni.kit.app.get_app().next_update_async()` loop.

## Package Renames

| Old import (pre-5.1) | New import (5.1.0+, deprecated again in 6.0 — see above) |
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
