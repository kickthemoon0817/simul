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
```

A Plane always collides with its analytic shape (an infinite plane) rather than a mesh collider, so it takes no `approximation`. This is the correct and most performant choice for Plane prims.

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

Add collision; a Cube collides as an exact box:

```
mcp__simul__add_isaac_collision
  prim_path: "/World/Ground"
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

```
mcp__simul__create_isaac_physics_scene
  prim_path: "/World/PhysicsScene"
```

With no arguments the scene gets Earth gravity along the stage's negative up axis, in stage units: UsdPhysics stores the magnitude in stage units per s^2, so `create_isaac_physics_scene` divides the m/s^2 value by the stage's metersPerUnit.

**Gravity reference** (`gravity_magnitude` in m/s^2, `gravity_direction` in stage axes):
- Isaac Sim default (Z-up, metres): direction `[0, 0, -1]`, magnitude `9.81`
- Y-up stage: direction `[0, -1, 0]` (the default there); a centimetre stage stores 981
- Moon gravity: magnitude `1.62`
- Mars gravity: magnitude `3.72`

In a raw script, author the magnitude in stage units: `9.81 / UsdGeom.GetStageMetersPerUnit(stage)`.

Check the existing physics scene with:

```
mcp__simul__get_isaac_physics_scene
```
