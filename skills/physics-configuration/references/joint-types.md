# Joint Types for Articulated Mechanisms

All joint types in Isaac Sim are defined via `pxr.UsdPhysics`. Joints connect two bodies (body0 and body1) and constrain their relative motion. Create joints via `execute_isaac_script` — there is no dedicated granular MCP tool for joint creation.

## General Joint Concepts

- **body0** and **body1**: the two prims the joint connects. Use `CreateBody0Rel` and `CreateBody1Rel` with `Sdf.Path` targets.
- **Local frames**: joints have a local coordinate frame on each body side. The joint axis and limits are defined in these local frames.
- **Limits**: lower and upper limit attributes constrain the range of motion. Units are degrees for revolute joints, meters for prismatic joints.
- **Drives**: `UsdPhysics.DriveAPI` adds motor/actuator behaviour (position or velocity targets with stiffness and damping).

---

## Fixed Joint

Welds two bodies together rigidly. No relative motion allowed. Use for bolted assemblies, gripper fingertips, or anchoring a robot base to the ground.

```python
import json, traceback
try:
    from pxr import UsdPhysics, Sdf
    import omni.usd
    stage = omni.usd.get_context().get_stage()

    joint = UsdPhysics.FixedJoint.Define(stage, "/World/Joints/BaseWeld")
    joint.CreateBody0Rel().SetTargets([Sdf.Path("/World/Base")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/World/Link1")])

    result = {"success": True, "joint": "/World/Joints/BaseWeld", "type": "fixed"}
except Exception as e:
    result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}
print(json.dumps(result))
```

---

## Revolute Joint (Hinge)

Allows rotation around one axis. Used for arm elbow joints, wheel axles, door hinges.

```python
import json, traceback
try:
    from pxr import UsdPhysics, Sdf
    import omni.usd
    stage = omni.usd.get_context().get_stage()

    joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/Joints/ElbowJoint")
    joint.CreateBody0Rel().SetTargets([Sdf.Path("/World/Link1")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/World/Link2")])
    joint.CreateAxisAttr("Z")          # rotation axis in the joint's local frame
    joint.CreateLowerLimitAttr(-90.0)  # degrees
    joint.CreateUpperLimitAttr(90.0)

    result = {"success": True, "joint": "/World/Joints/ElbowJoint", "type": "revolute"}
except Exception as e:
    result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}
print(json.dumps(result))
```

**Axis values:** `"X"`, `"Y"`, or `"Z"` in the joint's local frame. For robot arms in Z-up convention, the rotation axis is usually `"Z"` (yaw) or `"X"` (pitch/roll) depending on the joint orientation.

---

## Prismatic Joint (Slider)

Allows translation along one axis. Used for linear actuators, pistons, and telescoping mechanisms.

```python
import json, traceback
try:
    from pxr import UsdPhysics, Sdf
    import omni.usd
    stage = omni.usd.get_context().get_stage()

    joint = UsdPhysics.PrismaticJoint.Define(stage, "/World/Joints/ExtendJoint")
    joint.CreateBody0Rel().SetTargets([Sdf.Path("/World/Carriage")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/World/Rod")])
    joint.CreateAxisAttr("X")
    joint.CreateLowerLimitAttr(-0.5)   # meters
    joint.CreateUpperLimitAttr(0.5)

    result = {"success": True, "joint": "/World/Joints/ExtendJoint", "type": "prismatic"}
except Exception as e:
    result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}
print(json.dumps(result))
```

---

## Spherical Joint (Ball-and-Socket)

Allows rotation around all three axes. No translational freedom. Used for shoulder joints, universal mounts.

```python
import json, traceback
try:
    from pxr import UsdPhysics, Sdf
    import omni.usd
    stage = omni.usd.get_context().get_stage()

    joint = UsdPhysics.SphericalJoint.Define(stage, "/World/Joints/ShoulderJoint")
    joint.CreateBody0Rel().SetTargets([Sdf.Path("/World/Torso")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/World/UpperArm")])
    # Cone limits (degrees from rest pose)
    joint.CreateConeAngle0LimitAttr(60.0)  # azimuth cone half-angle
    joint.CreateConeAngle1LimitAttr(60.0)  # elevation cone half-angle

    result = {"success": True, "joint": "/World/Joints/ShoulderJoint", "type": "spherical"}
except Exception as e:
    result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}
print(json.dumps(result))
```

---

## Adding a Drive (Actuator) to a Joint

`UsdPhysics.DriveAPI` adds position or velocity targeting with stiffness and damping. Apply it to any joint that needs motor control.

```python
import json, traceback
try:
    from pxr import UsdPhysics, Sdf
    import omni.usd
    stage = omni.usd.get_context().get_stage()

    # Assume revolute joint already exists
    joint_prim = stage.GetPrimAtPath("/World/Joints/ElbowJoint")

    # Apply drive on the rotation axis
    drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")   # "angular" for revolute, "linear" for prismatic
    drive.CreateTypeAttr("force")           # "force" or "acceleration"
    drive.CreateStiffnessAttr(1000.0)       # position gain (spring)
    drive.CreateDampingAttr(100.0)          # velocity gain (damper)
    drive.CreateMaxForceAttr(1000.0)        # torque cap in N·m
    drive.CreateTargetPositionAttr(45.0)    # target angle in degrees

    result = {"success": True, "drive_applied": True}
except Exception as e:
    result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}
print(json.dumps(result))
```

**Drive type guidance:**
- `"force"` type: stiffness and damping are in force/torque units. Physically realistic.
- `"acceleration"` type: stiffness and damping are mass-normalised. Easier tuning, less physically accurate.
- For velocity control: set `targetVelocity` and `stiffness=0`, `damping>0`.
- For position control: set `targetPosition` and `stiffness>0`, `damping>0`.

---

## Querying Joints

```
mcp__simul__get_isaac_prim_detail  aspects: ["joint"]
  prim_path: "/World/Joints/ElbowJoint"
```

Returns joint type, body0/body1 paths, axis, limits, and drive state.

---

## Robot Articulation Root

For multi-link robots, mark the root link with `UsdPhysics.ArticulationRootAPI` to enable PhysX's articulation solver (more stable and efficient than individual joints):

```python
import json, traceback
try:
    from pxr import UsdPhysics
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath("/World/Robot/BaseLink")
    UsdPhysics.ArticulationRootAPI.Apply(root_prim)
    result = {"success": True, "articulation_root": str(root_prim.GetPath())}
except Exception as e:
    result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}
print(json.dumps(result))
```

Apply `ArticulationRootAPI` to exactly one prim in the robot hierarchy (the root link). All child links connected via joints will be included in the articulation automatically.
