---
name: physics-configuration
description: This skill should be used when the user asks to "add physics", "make it a rigid body", "add a collider", "add collision", "set physics material", "configure physics", "add gravity", "set mass", "create a joint", "add articulation", or needs to configure physics properties on scene objects in Isaac Sim.
version: 0.1.0
---

# Physics Configuration Workflow

Isaac Sim physics is built on top of USD physics schemas (UsdPhysics) with NVIDIA PhysX as the backend (PhysxSchema). The MCP tools wrap the most common operations. Use `execute_isaac_script` for joint creation, articulation setup, and advanced PhysxSchema properties.

## Step 1: Verify Physics Scene Exists

Physics simulation requires a `UsdPhysics.Scene` prim on the stage. Without it, simulation starts but no physics forces are applied.

```
mcp__simul__get_isaac_physics_scene
```

If the result shows no physics scene, create one:

```
mcp__simul__execute_isaac_script
  script: |
    import json, traceback
    try:
        from pxr import UsdPhysics
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        scene_prim = stage.GetPrimAtPath("/physicsScene")
        if not scene_prim.IsValid():
            scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
            scene.CreateGravityDirectionAttr().Set((0, 0, -1))
            scene.CreateGravityMagnitudeAttr().Set(9.81)
            result = {"success": True, "created": True}
        else:
            result = {"success": True, "created": False, "note": "already exists"}
    except Exception as e:
        import traceback
        result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}
    print(json.dumps(result))
```

## Step 2: Add a Rigid Body

A rigid body makes an object participate in physics simulation — it will respond to gravity, collisions, and applied forces.

```
mcp__simul__add_isaac_rigid_body
  prim_path: "/World/Box"
```

This applies `UsdPhysics.RigidBodyAPI` to the prim. The prim must already exist. After adding, verify:

```
mcp__simul__get_isaac_prim_detail  aspects: ["rigid_body"]
  prim_path: "/World/Box"
```

Returns: enabled state, linear velocity, angular velocity, and whether the body is kinematic.

**Kinematic vs dynamic:** A dynamic rigid body is fully simulated (gravity, forces, contacts). A kinematic body is driven by animation or script — physics will react to it but it ignores forces. To make a body kinematic, use `execute_isaac_script` to set the `physics:kinematicEnabled` attribute to `True` via `UsdPhysics.RigidBodyAPI`.

## Step 3: Add Collision

A rigid body without a collision shape passes through other objects. Add a collider:

```
mcp__simul__add_isaac_collision
  prim_path: "/World/Box"
  collision_type: "convexHull"
```

**Collision type selection:**

| collision_type | Use when |
|----------------|----------|
| `"convexHull"` | Simple convex objects (boxes, capsules, most single-part assets). Fast and stable. |
| `"meshSimplification"` | Complex concave meshes that need moderate accuracy. Slower than convexHull. |
| `"none"` | Planes and analytic primitives (Sphere, Cylinder, Cube) — they use their exact shape automatically. |

For articulated robots with per-link geometry, apply collision to each link prim individually using the appropriate type for that link's shape.

Verify collision was applied:

```
mcp__simul__get_isaac_prim_detail  aspects: ["collision"]
  prim_path: "/World/Box"
```

## Step 4: Set Mass Properties

By default, PhysX estimates mass from geometry and density. Override explicitly when you know the true mass:

```
mcp__simul__set_isaac_mass_properties
  prim_path: "/World/Box"
  mass: 2.5
  center_of_mass: [0, 0, 0]
```

`mass` is in kilograms. `center_of_mass` is in local prim coordinates (meters). Omit `center_of_mass` to let PhysX estimate it from geometry.

Read back the applied values:

```
mcp__simul__get_isaac_prim_detail  aspects: ["mass"]
  prim_path: "/World/Box"
```

For inertia tensor control (advanced), use `execute_isaac_script` with `UsdPhysics.MassAPI`:

```python
import json, traceback
try:
    from pxr import UsdPhysics, Gf
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath("/World/Box")
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(2.5)
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
    # Diagonal inertia tensor (Ixx, Iyy, Izz)
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(0.1, 0.1, 0.1))
    result = {"success": True}
except Exception as e:
    import traceback
    result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}
print(json.dumps(result))
```

## Step 5: Physics Material

Physics materials control surface friction and restitution (bounciness). Without a physics material, PhysX uses defaults (moderate friction, no bounce).

```
mcp__simul__set_isaac_physics_material
  prim_path: "/World/Box"
  static_friction: 0.5
  dynamic_friction: 0.3
  restitution: 0.1
```

This creates a `UsdPhysics.MaterialAPI` on the prim. For shared materials applied to multiple objects, create the material prim once and bind it:

```
mcp__simul__set_isaac_material_property
  prim_path: "/World/Materials/IceMaterial"
  property_name: "physics:staticFriction"
  value: 0.05
```

Read current material settings:

```
mcp__simul__get_isaac_prim_detail  aspects: ["material"]
  prim_path: "/World/Box"
```

## Step 6: Query All Physics Objects

List everything in the scene with physics APIs applied:

```
mcp__simul__list_isaac_physics_objects
```

Returns paths of all prims with RigidBodyAPI, CollisionAPI, or JointAPI. Useful for auditing a scene before running simulation.

## Step 7: Joint Configuration (via execute_isaac_script)

Joints are not covered by granular tools — create them via `execute_isaac_script`. All joint types live under `pxr.UsdPhysics`:

**Fixed joint** (weld two bodies together):

```python
import json, traceback
try:
    from pxr import UsdPhysics, Sdf
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    joint = UsdPhysics.FixedJoint.Define(stage, "/World/Joints/BaseJoint")
    joint.CreateBody0Rel().SetTargets([Sdf.Path("/World/Base")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/World/Link1")])
    result = {"success": True, "joint": "/World/Joints/BaseJoint"}
except Exception as e:
    import traceback
    result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}
print(json.dumps(result))
```

**Revolute joint** (hinge, rotates around one axis):

```python
import json, traceback
try:
    from pxr import UsdPhysics, Gf, Sdf
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/Joints/HingeJoint")
    joint.CreateBody0Rel().SetTargets([Sdf.Path("/World/Link1")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/World/Link2")])
    joint.CreateAxisAttr("Z")  # rotation axis in local frame
    joint.CreateLowerLimitAttr(-180.0)   # degrees
    joint.CreateUpperLimitAttr(180.0)
    result = {"success": True, "joint": "/World/Joints/HingeJoint"}
except Exception as e:
    import traceback
    result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}
print(json.dumps(result))
```

**Prismatic joint** (slides along one axis):

```python
import json, traceback
try:
    from pxr import UsdPhysics, Sdf
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    joint = UsdPhysics.PrismaticJoint.Define(stage, "/World/Joints/SliderJoint")
    joint.CreateBody0Rel().SetTargets([Sdf.Path("/World/Base")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/World/Slider")])
    joint.CreateAxisAttr("X")
    joint.CreateLowerLimitAttr(-0.5)   # meters
    joint.CreateUpperLimitAttr(0.5)
    result = {"success": True, "joint": "/World/Joints/SliderJoint"}
except Exception as e:
    import traceback
    result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}
print(json.dumps(result))
```

Query joint state after creation:

```
mcp__simul__get_isaac_prim_detail  aspects: ["joint"]
  prim_path: "/World/Joints/HingeJoint"
```

## Typical Physics Setup Sequence for a Dynamic Object

1. `mcp__simul__create_isaac_prim` — create the geometry (e.g. Cube at `/World/Box`)
2. `mcp__simul__get_isaac_physics_scene` — verify physics scene exists; create if missing
3. `mcp__simul__add_isaac_rigid_body` — enable physics simulation on `/World/Box`
4. `mcp__simul__add_isaac_collision` — add `convexHull` collider to `/World/Box`
5. `mcp__simul__set_isaac_mass_properties` — set mass to known value
6. `mcp__simul__set_isaac_physics_material` — set friction and restitution
7. `mcp__simul__get_isaac_prim_detail` with `aspects=["rigid_body"]` — verify everything applied
8. `mcp__simul__start_isaac_simulation` — run and observe

## Reference Files

- `references/rigid-body-patterns.md` — UsdPhysics API patterns: RigidBodyAPI, CollisionAPI, MassAPI, PhysxSchema
- `references/joint-types.md` — joint configuration patterns for articulated mechanisms
