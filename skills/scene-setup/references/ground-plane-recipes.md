# Ground Plane Recipes

## Option 1: Plane Prim (Infinite Physics, Finite Visual)

The simplest ground. Uses `create_isaac_prim` with type `"Plane"`. The physics collision is an infinite flat plane — objects never fall through the edges regardless of the visual mesh scale. Scale controls only the visible mesh extent.

```
mcp__simul__create_isaac_prim
  prim_path: "/World/GroundPlane"
  prim_type: "Plane"
  scale: [50, 50, 1]
  position: [0, 0, 0]
```

Then add collision using the analytic shape (no mesh approximation needed for a flat plane):

```
mcp__simul__add_isaac_collision
  prim_path: "/World/GroundPlane"
  collision_type: "none"
```

`collision_type: "none"` tells the physics engine to use the prim's analytic shape (infinite plane) rather than generating a mesh collider. This is the correct and most performant choice for Plane prims.

---

## Option 2: Thin Cube as Finite Ground

When you need a bounded ground surface (e.g. a platform with defined edges that objects can fall off):

```
mcp__simul__create_isaac_prim
  prim_path: "/World/Ground"
  prim_type: "Cube"
  position: [0, 0, -0.05]
  scale: [50, 50, 0.1]
```

Add collision with convexHull (correct for a box):

```
mcp__simul__add_isaac_collision
  prim_path: "/World/Ground"
  collision_type: "convexHull"
```

Optionally set a high mass so it acts as a static body that is never pushed by colliding objects:

```
mcp__simul__set_isaac_mass_properties
  prim_path: "/World/Ground"
  mass: 1000000
```

Or make it truly static by not adding a RigidBodyAPI at all — static objects with only CollisionAPI do not move.

---

## Option 3: Physics Ground Plane via USD (Static Collider Only)

For the lowest-overhead ground: create a physics plane directly in USD without any geometry. This is a zero-thickness infinite plane collider with no visual mesh:

```python
import json, traceback
try:
    from pxr import UsdGeom, UsdPhysics, Gf
    import omni.usd
    stage = omni.usd.get_context().get_stage()

    # Create an Xform to hold the plane
    xform = UsdGeom.Xform.Define(stage, "/World/PhysicsGround")
    plane = UsdGeom.Plane.Define(stage, "/World/PhysicsGround/CollisionPlane")
    plane.CreateAxisAttr("Z")  # Z-up normal

    # Apply collision only — no RigidBodyAPI means it is static
    UsdPhysics.CollisionAPI.Apply(plane.GetPrim())

    result = {"success": True, "path": "/World/PhysicsGround"}
except Exception as e:
    result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}
print(json.dumps(result))
```

---

## Physics Scene Creation

Every dynamic scene needs a `UsdPhysics.Scene` prim. Without it, simulation starts but gravity and contacts do not apply.

```python
import json, traceback
try:
    from pxr import UsdPhysics
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    scene_prim = stage.GetPrimAtPath("/physicsScene")
    if not scene_prim.IsValid():
        scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
        scene.CreateGravityDirectionAttr().Set((0, 0, -1))   # -Z is down (Z-up convention)
        scene.CreateGravityMagnitudeAttr().Set(9.81)          # m/s^2
        result = {"success": True, "created": True, "path": "/physicsScene"}
    else:
        result = {"success": True, "created": False, "note": "already exists"}
except Exception as e:
    result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}
print(json.dumps(result))
```

**Gravity direction reference:**
- Isaac Sim default (Z-up): direction `(0, 0, -1)`, magnitude `9.81`
- Y-up convention: direction `(0, -1, 0)`, magnitude `9.81`
- Moon gravity: direction `(0, 0, -1)`, magnitude `1.62`
- Mars gravity: direction `(0, 0, -1)`, magnitude `3.72`

Check the existing physics scene with:

```
mcp__simul__get_isaac_physics_scene
```
