# Script Templates

Copy-paste templates for common `execute_isaac_script` patterns. Always adapt the prim paths and parameters for your specific scene.

## Error-Safe Base Template

Every script should use this structure. Never omit the try/except — it ensures the tool always returns parseable JSON:

```python
import json
import traceback

try:
    # --- your logic here ---
    result = {"success": True}
except Exception as e:
    result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}

print(json.dumps(result))
```

---

## Stage Info

```python
import json, traceback
try:
    import omni.usd
    from pxr import UsdGeom
    ctx = omni.usd.get_context()
    stage = ctx.get_stage()
    result = {
        "success": True,
        "url": ctx.get_stage_url(),
        "up_axis": UsdGeom.GetStageUpAxis(stage),
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "prim_count": len(list(stage.Traverse())),
    }
except Exception as e:
    result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}
print(json.dumps(result))
```

---

## Traverse All Prims by Type

```python
import json, traceback
try:
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    prims = [
        {"path": str(p.GetPath()), "type": p.GetTypeName()}
        for p in stage.Traverse()
        if p.GetTypeName()
    ]
    result = {"success": True, "prims": prims, "count": len(prims)}
except Exception as e:
    result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}
print(json.dumps(result))
```

---

## List All Mesh Prims

```python
import json, traceback
try:
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    meshes = [str(p.GetPath()) for p in stage.Traverse() if p.GetTypeName() == "Mesh"]
    result = {"success": True, "meshes": meshes, "count": len(meshes)}
except Exception as e:
    result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}
print(json.dumps(result))
```

---

## Set Prim Transform

```python
import json, traceback
try:
    from isaacsim.core.utils.prims import set_prim_attribute_value
    from pxr import Gf
    prim_path = "/World/Box"
    set_prim_attribute_value(prim_path, "xformOp:translate", Gf.Vec3d(1.0, 2.0, 0.5))
    set_prim_attribute_value(prim_path, "xformOp:scale", Gf.Vec3d(2.0, 2.0, 2.0))
    result = {"success": True, "prim": prim_path}
except Exception as e:
    result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}
print(json.dumps(result))
```

---

## Get World Pose

```python
import json, traceback
try:
    from isaacsim.core.utils.xforms import get_world_pose
    pos, quat = get_world_pose("/World/Robot")
    result = {
        "success": True,
        "position": pos.tolist(),
        "orientation_wxyz": quat.tolist(),
    }
except Exception as e:
    result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}
print(json.dumps(result))
```

---

## Check Simulation State

```python
import json, traceback
try:
    import omni.timeline
    tl = omni.timeline.get_timeline_interface()
    result = {
        "success": True,
        "playing": tl.is_playing(),
        "stopped": tl.is_stopped(),
        "current_time": tl.get_current_time(),
        "fps": tl.get_time_codes_per_second(),
    }
except Exception as e:
    result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}
print(json.dumps(result))
```

---

## Step Physics with World API

```python
import json, traceback
try:
    from isaacsim.core.api import World
    world = World.instance()
    if world is None:
        world = World(physics_dt=1.0/60.0, rendering_dt=1.0/60.0)
        world.reset()
    n_steps = 10
    for _ in range(n_steps):
        world.step(render=True)
    result = {"success": True, "steps_taken": n_steps}
except Exception as e:
    result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}
print(json.dumps(result))
```

---

## Compute Bounding Box

```python
import json, traceback
try:
    from isaacsim.core.utils.bounds import create_bbox_cache, compute_aabb
    import numpy as np
    cache = create_bbox_cache()
    aabb = compute_aabb(cache, "/World/Robot")
    center = ((aabb[:3] + aabb[3:]) / 2).tolist()
    size = (aabb[3:] - aabb[:3]).tolist()
    result = {"success": True, "center": center, "size": size, "aabb": aabb.tolist()}
except Exception as e:
    result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}
print(json.dumps(result))
```

---

## Raycast

```python
import json, traceback
try:
    import numpy as np
    from isaacsim.core.utils.collisions import ray_cast
    hit_path, distance = ray_cast(
        position=np.array([0.0, 0.0, 2.0]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),  # wxyz, looking down -Z
        offset=np.array([0.0, 0.0, 0.0]),
        max_dist=100.0,
    )
    result = {"success": True, "hit": hit_path, "distance": float(distance)}
except Exception as e:
    result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}
print(json.dumps(result))
```

---

## Create and Bind a Material

```python
import json, traceback
try:
    from pxr import UsdShade, Sdf, Gf
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    mat_path = "/World/Materials/RedMaterial"
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1, 0, 0))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    target = stage.GetPrimAtPath("/World/MyCube")
    UsdShade.MaterialBindingAPI(target).Bind(mat)
    result = {"success": True, "material": mat_path}
except Exception as e:
    result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}
print(json.dumps(result))
```

---

## Query Rigid Body Velocities

```python
import json, traceback
try:
    from pxr import UsdPhysics
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath("/World/Box")
    if not prim.IsValid():
        raise ValueError("Prim not found: /World/Box")
    has_rb = prim.HasAPI(UsdPhysics.RigidBodyAPI)
    lin_vel = [0.0, 0.0, 0.0]
    ang_vel = [0.0, 0.0, 0.0]
    if has_rb:
        v = prim.GetAttribute("physics:velocity")
        a = prim.GetAttribute("physics:angularVelocity")
        if v and v.Get():
            lin_vel = list(v.Get())
        if a and a.Get():
            ang_vel = list(a.Get())
    result = {
        "success": True,
        "has_rigid_body": has_rb,
        "linear_velocity": lin_vel,
        "angular_velocity": ang_vel,
    }
except Exception as e:
    result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}
print(json.dumps(result))
```
