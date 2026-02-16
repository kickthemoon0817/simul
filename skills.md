# Isaac Sim 5.1.0 — Agent Skills Reference

> **Purpose**: This document teaches AI agents how to write Python scripts for
> execution inside a running Isaac Sim 5.1.0 instance via the
> `execute_isaac_script` MCP tool.

## Execution Model

Code is sent over TCP to the stock `isaacsim.code_editor.vscode` extension
(port 8226). The executor runs your code with `compile() + eval()` inside
Kit's Python process, with **full access to the global namespace** including
`omni.*`, `pxr.*`, and `isaacsim.*`.

**Rules:**
- `stdout` is captured and returned — use `print()` for output
- For structured results, **always** `print(json.dumps({...}))` as the last line
- Top-level `await` is supported (the executor detects coroutines)
- Each execution is independent — use `import` at the top of every script
- Avoid long-running loops that block the Kit main thread

---

## 1. Simulation Control (Timeline)

The timeline is the lowest-level playback API. Prefer it over World for simple
play/pause/stop — it doesn't require constructing a World instance.

```python
import json
import omni.timeline

tl = omni.timeline.get_timeline_interface()

# Play / Pause / Stop
tl.play()
tl.pause()
tl.stop()

# Query state
print(json.dumps({
    "is_playing": tl.is_playing(),
    "is_stopped": tl.is_stopped(),
    "current_time": tl.get_current_time(),
    "fps": tl.get_time_codes_per_second(),
}))
```

### Stepping with World (advanced)

```python
from isaacsim.core.api import World
import json

world = World.instance()          # get existing singleton, or None
if world is None:
    world = World(physics_dt=1.0/60.0, rendering_dt=1.0/60.0)

world.reset()                     # first-time init (does 1 internal step)
for _ in range(10):
    world.step(render=True)       # step physics + rendering
print(json.dumps({"stepped": 10}))
```

---

## 2. USD Stage Operations

```python
import json
import omni.usd
from pxr import Usd, UsdGeom

# Get the current stage
ctx = omni.usd.get_context()
stage = ctx.get_stage()

# Stage metadata
print(json.dumps({
    "url": ctx.get_stage_url(),
    "up_axis": UsdGeom.GetStageUpAxis(stage),
    "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
    "prim_count": len(list(stage.Traverse())),
}))
```

### Open / Save / Close

```python
from isaacsim.core.utils.stage import open_stage, save_stage, close_stage, create_new_stage

open_stage("/path/to/scene.usd")          # replaces current stage
save_stage("/path/to/output.usd")         # saves current stage
create_new_stage()                        # blank stage
```

### Traverse Prims

```python
import json, omni.usd
from pxr import Usd

stage = omni.usd.get_context().get_stage()
prims = [
    {"path": str(p.GetPath()), "type": p.GetTypeName()}
    for p in stage.Traverse()
]
print(json.dumps({"prims": prims}))
```

---

## 3. Prim CRUD

```python
import json
from isaacsim.core.utils.prims import (
    create_prim, delete_prim, get_prim_at_path,
    get_prim_attribute_names, get_prim_attribute_value,
    set_prim_attribute_value, is_prim_path_valid,
    set_prim_visibility, define_prim,
)

# Create with position + scale
prim = create_prim(
    prim_path="/World/MyCube",
    prim_type="Cube",
    position=[0.0, 0.0, 1.0],       # Gf.Vec3d
    scale=[0.5, 0.5, 0.5],
    attributes={"size": 1.0},
)
print(json.dumps({"created": str(prim.GetPath())}))

# Read attributes
names = get_prim_attribute_names("/World/MyCube")
size = get_prim_attribute_value("/World/MyCube", "size")

# Update attribute
set_prim_attribute_value("/World/MyCube", "size", 2.0)

# Delete
delete_prim("/World/MyCube")
```

### Transform Operations

```python
from isaacsim.core.utils.xforms import get_world_pose, get_local_pose
import json, numpy as np

pos, quat = get_world_pose("/World/Robot")
print(json.dumps({
    "position": pos.tolist(),
    "orientation_wxyz": quat.tolist(),
}))
```

---

## 4. Viewport & Camera

```python
from isaacsim.core.utils.viewports import set_camera_view, set_active_viewport_camera
import numpy as np

# Position the default perspective camera
set_camera_view(
    eye=np.array([5.0, 5.0, 3.0]),
    target=np.array([0.0, 0.0, 0.0]),
    camera_prim_path="/OmniverseKit_Persp",
)
```

### Create a Camera Prim

```python
import json
from isaacsim.core.utils.prims import create_prim

cam = create_prim(
    prim_path="/World/MyCamera",
    prim_type="Camera",
    attributes={
        "focalLength": 24.0,
        "horizontalAperture": 20.955,
        "verticalAperture": 15.2908,
        "clippingRange": (0.1, 10000.0),
    },
)
print(json.dumps({"camera": str(cam.GetPath())}))
```

---

## 5. Physics — Rigid Bodies

```python
import json
from pxr import UsdPhysics, UsdGeom, Gf
import omni.usd

stage = omni.usd.get_context().get_stage()
prim = stage.GetPrimAtPath("/World/Box")

# Apply rigid body + collision APIs
UsdPhysics.RigidBodyAPI.Apply(prim)
UsdPhysics.CollisionAPI.Apply(prim)

# Optionally set mass
mass_api = UsdPhysics.MassAPI.Apply(prim)
mass_api.CreateMassAttr(2.0)

print(json.dumps({"rigid_body_enabled": True, "prim": str(prim.GetPath())}))
```

### Query Rigid Body State

```python
import json
from pxr import UsdPhysics, PhysxSchema
import omni.usd

stage = omni.usd.get_context().get_stage()
prim = stage.GetPrimAtPath("/World/Box")

has_rb = prim.HasAPI(UsdPhysics.RigidBodyAPI)
vel = [0, 0, 0]
ang = [0, 0, 0]
if has_rb:
    rb = PhysxSchema.PhysxRigidBodyAPI(prim)
    # Velocities are only populated during simulation
    v = prim.GetAttribute("physics:velocity")
    a = prim.GetAttribute("physics:angularVelocity")
    if v and v.Get():
        vel = list(v.Get())
    if a and a.Get():
        ang = list(a.Get())

print(json.dumps({
    "has_rigid_body": has_rb,
    "linear_velocity": vel,
    "angular_velocity": ang,
}))
```

---

## 6. Physics Scene

```python
import json
from pxr import UsdPhysics, PhysxSchema
import omni.usd

stage = omni.usd.get_context().get_stage()
scene_prim = stage.GetPrimAtPath("/physicsScene")

if not scene_prim.IsValid():
    # Create physics scene
    scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
    scene.CreateGravityDirectionAttr().Set((0, 0, -1))
    scene.CreateGravityMagnitudeAttr().Set(9.81)
    print(json.dumps({"created": True}))
else:
    scene = UsdPhysics.Scene(scene_prim)
    grav_dir = scene.GetGravityDirectionAttr().Get()
    grav_mag = scene.GetGravityMagnitudeAttr().Get()
    print(json.dumps({
        "gravity_direction": list(grav_dir),
        "gravity_magnitude": grav_mag,
    }))
```

---

## 7. Materials & Appearance

### Apply a Preview Surface Material

```python
import json
from pxr import UsdShade, Sdf, Gf
import omni.usd

stage = omni.usd.get_context().get_stage()

# Create material
mat_path = "/World/Materials/RedMaterial"
mat = UsdShade.Material.Define(stage, mat_path)
shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
shader.CreateIdAttr("UsdPreviewSurface")
shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1, 0, 0))
shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

# Bind to prim
target_prim = stage.GetPrimAtPath("/World/MyCube")
UsdShade.MaterialBindingAPI(target_prim).Bind(mat)

print(json.dumps({"material": mat_path, "bound_to": "/World/MyCube"}))
```

---

## 8. Lights

```python
import json
from pxr import UsdLux, Gf
import omni.usd

stage = omni.usd.get_context().get_stage()

# Distant light (sun)
light = UsdLux.DistantLight.Define(stage, "/World/Sun")
light.CreateIntensityAttr(3000)
light.CreateAngleAttr(0.53)
light.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.85))

# Dome light (environment)
dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome.CreateIntensityAttr(1000)
# dome.CreateTextureFileAttr("path/to/hdr.hdr")

# Sphere light (point)
sphere = UsdLux.SphereLight.Define(stage, "/World/PointLight")
sphere.CreateIntensityAttr(5000)
sphere.CreateRadiusAttr(0.1)

print(json.dumps({"lights_created": 3}))
```

---

## 9. Bounding Boxes

```python
import json
from isaacsim.core.utils.bounds import create_bbox_cache, compute_aabb, compute_obb
import numpy as np

cache = create_bbox_cache()
aabb = compute_aabb(cache, "/World/Robot")   # [xmin, ymin, zmin, xmax, ymax, zmax]
center = ((aabb[:3] + aabb[3:]) / 2).tolist()
size = (aabb[3:] - aabb[:3]).tolist()

print(json.dumps({"center": center, "size": size, "aabb": aabb.tolist()}))
```

---

## 10. Rotations & Transforms

```python
import json
import numpy as np
from isaacsim.core.utils.rotations import (
    euler_angles_to_quat,
    quat_to_euler_angles,
)

# Euler (radians, XYZ extrinsic) → quaternion (w,x,y,z)
quat = euler_angles_to_quat(np.array([0, 0, np.pi/4]))
# Quaternion → Euler
euler = quat_to_euler_angles(quat, degrees=True)

print(json.dumps({
    "quat_wxyz": quat.tolist(),
    "euler_deg": euler.tolist(),
}))
```

---

## 11. Raycasting

```python
import json
import numpy as np
from isaacsim.core.utils.collisions import ray_cast

# Cast ray from position, along direction derived from orientation
hit_path, distance = ray_cast(
    position=np.array([0, 0, 1]),
    orientation=np.array([1, 0, 0, 0]),  # wxyz quaternion
    offset=np.array([0, 0, 0]),
    max_dist=100.0,
)
print(json.dumps({"hit": hit_path, "distance": distance}))
```

---

## 12. Asset Loading (References)

```python
import json
from isaacsim.core.utils.stage import add_reference_to_stage

# Load a USD asset as a reference at a specific prim path
prim = add_reference_to_stage(
    usd_path="/path/to/robot.usd",           # local or Nucleus path
    prim_path="/World/MyRobot",
)
print(json.dumps({"loaded": str(prim.GetPath())}))
```

### Nucleus Paths

Isaac Sim supports Nucleus server paths:
```
omniverse://localhost/NVIDIA/Assets/Isaac/4.2/Isaac/Robots/Franka/franka.usd
```

---

## 13. Selection

```python
import json
import omni.usd

ctx = omni.usd.get_context()
selection = ctx.get_selection()

# Get currently selected prims
selected = selection.get_selected_prim_paths()

# Select a prim
selection.set_prim_path_selected("/World/Robot", True, True, True, True)

print(json.dumps({"selected": list(selected)}))
```

---

## Common Patterns

### Error-Safe Template

Every script should follow this pattern for reliable JSON output:

```python
import json, traceback
try:
    # ... your logic here ...
    result = {"success": True, "data": "..."}
except Exception as e:
    result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}
print(json.dumps(result))
```

### Checking if Simulation is Running

```python
import json, omni.timeline
tl = omni.timeline.get_timeline_interface()
print(json.dumps({"playing": tl.is_playing(), "stopped": tl.is_stopped()}))
```

### List All Mesh Prims

```python
import json, omni.usd
stage = omni.usd.get_context().get_stage()
meshes = [
    str(p.GetPath()) for p in stage.Traverse()
    if p.GetTypeName() == "Mesh"
]
print(json.dumps({"meshes": meshes, "count": len(meshes)}))
```

### Set Prim Transform

```python
from isaacsim.core.utils.prims import set_prim_attribute_value
from pxr import Gf

set_prim_attribute_value("/World/Box", "xformOp:translate", Gf.Vec3d(1.0, 2.0, 0.5))
set_prim_attribute_value("/World/Box", "xformOp:scale", Gf.Vec3d(2.0, 2.0, 2.0))
```

---

## API Quick Reference

| Task | Module | Key Functions |
|------|--------|---------------|
| Play/Pause/Stop | `omni.timeline` | `get_timeline_interface().play/pause/stop()` |
| Step physics | `isaacsim.core.api.World` | `World().step(render=True)` |
| Get stage | `omni.usd` | `get_context().get_stage()` |
| Open/Save stage | `isaacsim.core.utils.stage` | `open_stage()`, `save_stage()` |
| Traverse prims | `pxr.Usd` | `stage.Traverse()` |
| Create prim | `isaacsim.core.utils.prims` | `create_prim(path, type, position=, ...)` |
| Delete prim | `isaacsim.core.utils.prims` | `delete_prim(path)` |
| Get/Set attrs | `isaacsim.core.utils.prims` | `get/set_prim_attribute_value()` |
| World pose | `isaacsim.core.utils.xforms` | `get_world_pose(path)` |
| Camera view | `isaacsim.core.utils.viewports` | `set_camera_view(eye, target)` |
| Rigid body | `pxr.UsdPhysics` | `RigidBodyAPI.Apply(prim)` |
| Materials | `pxr.UsdShade` | `Material.Define()`, `Shader.Define()` |
| Lights | `pxr.UsdLux` | `DistantLight/DomeLight/SphereLight.Define()` |
| Bounding box | `isaacsim.core.utils.bounds` | `create_bbox_cache()`, `compute_aabb()` |
| Rotations | `isaacsim.core.utils.rotations` | `euler_angles_to_quat()`, etc. |
| Raycast | `isaacsim.core.utils.collisions` | `ray_cast(pos, orient, offset)` |
| Load asset | `isaacsim.core.utils.stage` | `add_reference_to_stage(usd, path)` |
| Selection | `omni.usd` | `get_context().get_selection()` |

---

## Namespace Migration Note

Isaac Sim 5.1.0 migrated from `omni.isaac.*` to `isaacsim.*`:
- `omni.isaac.core` → `isaacsim.core.api`
- `omni.isaac.core.utils` → `isaacsim.core.utils`
- Kit-level APIs (`omni.usd`, `omni.timeline`, `pxr.*`) remain unchanged
