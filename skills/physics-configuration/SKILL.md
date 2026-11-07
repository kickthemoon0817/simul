---
name: physics-configuration
description: Use when the user asks to "add physics", "make it a rigid body", "add a collider", "add collision", "set physics material", "configure physics", "add gravity", "set mass", "create a joint", "add articulation", or needs to configure physics properties on scene objects in Isaac Sim.
version: 0.1.0
---

# Physics Configuration Workflow

Isaac Sim physics is built on top of USD physics schemas (UsdPhysics) with NVIDIA PhysX as the backend (PhysxSchema). The MCP tools wrap the most common operations, including joint drives, joint limits and articulation discovery. Use `execute_isaac_script` for joint creation and the PhysxSchema properties no tool covers.

## Step 1: Verify Physics Scene Exists

Physics simulation requires a `UsdPhysics.Scene` prim on the stage. Without it, simulation starts but no physics forces are applied.

```
mcp__simul__get_isaac_physics_scene
```

It reports each scene's gravity (resolved as the simulation reads it, in stage units and m/s^2), the stage's up axis and meters per unit, and the PhysX solver settings authored on the scene. If there is no scene, create one:

```
mcp__simul__create_isaac_physics_scene
```

A new scene gets Earth gravity along the stage's negative up axis, scaled to its units: a Y-up centimetre stage gets `[0, -1, 0]` at 981 units/s^2. Pass `gravity_magnitude` in m/s^2 and `gravity_direction` in stage axes to override. A prim of another type at `prim_path` is refused, not retyped.

Solver settings (physics rate, TGS/PGS, iteration clamps, GPU dynamics, broadphase, CCD, enhanced determinism):

```
mcp__simul__configure_isaac_physics_scene
  time_steps_per_second: 120
  solver_type: "TGS"
  enable_enhanced_determinism: true
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
  approximation: "convexHull"
```

Collision goes on geometry (Mesh, Cube, Sphere, Capsule, Cylinder, Cone, Plane, Points): pointed at an asset root Xform, the tool applies it to the geometry under it (`recursive`, on by default) and lists every shape in `shapes`. A CollisionAPI on the Xform itself would make no collision shape. It fails when there is no geometry.

**Approximation selection** (meshes only; Cube, Sphere and the other implicit shapes always collide with their exact shape):

| approximation | Use when |
|---------------|----------|
| `"convexHull"` | Simple convex objects (boxes, capsules, most single-part assets). Fast and stable. |
| `"convexDecomposition"` | Concave meshes on dynamic bodies. |
| `"meshSimplification"` | Complex concave meshes that need moderate accuracy. Slower than convexHull. |
| `"none"` | The triangle mesh itself; static or kinematic bodies only (PhysX simulates it as convexHull on a dynamic body, and the tool warns). |

Omit `approximation` to keep each mesh's authored one.

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

`prim_path` is the collider, or an asset root whose colliders should share the material. The tool creates (or reuses) a Material prim with `UsdPhysics.MaterialAPI`, by default at `/World/PhysicsMaterials/Box_PhysicsMaterial`, and binds it with purpose `physics` to every collision shape at or under `prim_path`; the visual material binding is untouched. `bound_to` lists the colliders. Add collision first: with no collision shapes the call fails.

For one material shared by several objects, pass the same `material_path` on each call:

```
mcp__simul__set_isaac_physics_material
  prim_path: "/World/Crate"
  material_path: "/World/PhysicsMaterials/Ice"
  static_friction: 0.05
```

Don't bind a physics material with `assign_isaac_material`: its all-purpose binding replaces the visual material and is not the binding the physics parser reads.

## Step 6: Query All Physics Objects

List everything in the scene with physics APIs applied:

```
mcp__simul__list_isaac_physics_objects
```

Returns paths of all prims with RigidBodyAPI, CollisionAPI, or JointAPI. Useful for auditing a scene before running simulation.

## Step 7: Joint Configuration (via execute_isaac_script)

Creating joints goes through `execute_isaac_script`; drives, limits and articulation discovery have granular tools (below). All joint types live under `pxr.UsdPhysics`:

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

Query joint state after creation (bodies, limits, drives; infinite values such as "no limit" read as null with a `*_note`):

```
mcp__simul__get_isaac_prim_detail  aspects: ["joint"]
  prim_path: "/World/Joints/HingeJoint"
```

Drive a joint (revolute joints default to the `angular` drive, prismatic to `linear`; angular targets are degrees):

```
mcp__simul__set_isaac_joint_drive
  joint_path: "/World/Joints/HingeJoint"
  stiffness: 1000
  damping: 50
  target_position: 45
```

Limit it (degrees for revolute joints, stage units for prismatic; a D6 joint takes `limit_axis`):

```
mcp__simul__set_isaac_joint_limits
  joint_path: "/World/Joints/HingeJoint"
  lower: -90
  upper: 90
```

List a robot's joints in kinematic order, with body0/body1, limits and drives:

```
mcp__simul__list_isaac_articulations
  root_path: "/World/Robot"
```

## Typical Physics Setup Sequence for a Dynamic Object

1. `mcp__simul__create_isaac_prim` — create the geometry (e.g. Cube at `/World/Box`)
2. `mcp__simul__get_isaac_physics_scene` — verify physics scene exists; create it with `create_isaac_physics_scene` if missing
3. `mcp__simul__add_isaac_rigid_body` — enable physics simulation on `/World/Box`
4. `mcp__simul__add_isaac_collision` — add a collider to `/World/Box` (a Cube collides as a box; meshes take `approximation`)
5. `mcp__simul__set_isaac_mass_properties` — set mass to known value
6. `mcp__simul__set_isaac_physics_material` — set friction and restitution on the collider
7. `mcp__simul__get_isaac_prim_detail` with `aspects=["rigid_body"]` — verify everything applied
8. `mcp__simul__start_isaac_simulation` — run and observe

## Reference Files

- `references/rigid-body-patterns.md` — UsdPhysics API patterns: RigidBodyAPI, CollisionAPI, MassAPI, PhysxSchema
- `references/joint-types.md` — joint configuration patterns for articulated mechanisms
