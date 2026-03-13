# Rigid Body Patterns

Physics in Isaac Sim is layered: `UsdPhysics` schemas define the interface, `PhysxSchema` provides NVIDIA PhysX-specific extensions. Apply schemas in the correct order.

## Standard Dynamic Rigid Body

Apply both RigidBodyAPI and CollisionAPI for an object that moves under physics forces:

```python
import json, traceback
try:
    from pxr import UsdPhysics
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath("/World/Box")
    if not prim.IsValid():
        raise ValueError("Prim not found at /World/Box")

    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)

    result = {"success": True, "path": str(prim.GetPath()), "dynamic": True}
except Exception as e:
    result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}
print(json.dumps(result))
```

**Rule:** RigidBodyAPI controls dynamics (mass, velocity, forces). CollisionAPI controls shape-based collision detection. Both are needed for a standard dynamic object.

---

## Static Collider (CollisionAPI Only)

An object that other bodies collide with but does not move itself. Do not apply RigidBodyAPI:

```python
import json, traceback
try:
    from pxr import UsdPhysics
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath("/World/Wall")
    UsdPhysics.CollisionAPI.Apply(prim)
    result = {"success": True, "path": str(prim.GetPath()), "static_collider": True}
except Exception as e:
    result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}
print(json.dumps(result))
```

This is the correct pattern for floors, walls, and fixed obstacles.

---

## Kinematic Rigid Body

A body that is animated or script-driven. Physics reacts to it (other objects bounce off), but it ignores gravity and applied forces. Apply RigidBodyAPI then set kinematic flag:

```python
import json, traceback
try:
    from pxr import UsdPhysics
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath("/World/AnimatedPlatform")
    rb_api = UsdPhysics.RigidBodyAPI.Apply(prim)
    rb_api.CreateKinematicEnabledAttr(True)
    UsdPhysics.CollisionAPI.Apply(prim)
    result = {"success": True, "path": str(prim.GetPath()), "kinematic": True}
except Exception as e:
    result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}
print(json.dumps(result))
```

---

## Mass Properties

Override estimated mass, center of mass, and inertia tensor:

```python
import json, traceback
try:
    from pxr import UsdPhysics, Gf
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath("/World/Box")
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(2.5)                             # kg
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))      # local coords
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(0.1, 0.1, 0.1))  # Ixx, Iyy, Izz
    result = {"success": True}
except Exception as e:
    result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}
print(json.dumps(result))
```

If you do not set mass explicitly, PhysX estimates it from volume × density. Default density is 1000 kg/m³.

---

## PhysxSchema Extensions

PhysxSchema provides per-body solver settings beyond the UsdPhysics baseline:

```python
import json, traceback
try:
    from pxr import PhysxSchema, UsdPhysics
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath("/World/Box")

    # Apply PhysX rigid body extension
    physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    physx_rb.CreateLinearDampingAttr(0.05)      # drag on linear velocity
    physx_rb.CreateAngularDampingAttr(0.05)     # drag on angular velocity
    physx_rb.CreateMaxLinearVelocityAttr(100.0) # cap speed in m/s
    physx_rb.CreateSolverPositionIterationCountAttr(8)   # solver accuracy
    physx_rb.CreateSolverVelocityIterationCountAttr(0)

    result = {"success": True}
except Exception as e:
    result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}
print(json.dumps(result))
```

**Useful PhysxRigidBodyAPI attributes:**
- `linearDamping` / `angularDamping`: velocity damping (0 = no drag, higher = more drag)
- `maxLinearVelocity`: cap on translational speed
- `maxAngularVelocity`: cap on rotational speed
- `solverPositionIterationCount`: higher = more accurate but slower (default 4, use 8-16 for stiff contacts)
- `enableCCD`: continuous collision detection — prevents tunnelling at high speeds

---

## Mesh Collision Approximations

For complex geometry, PhysX cannot use the raw mesh directly. Choose the approximation that matches the shape:

```python
import json, traceback
try:
    from pxr import UsdPhysics, PhysxSchema
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath("/World/ComplexMesh")

    UsdPhysics.CollisionAPI.Apply(prim)

    # Apply mesh collision API and set approximation
    mesh_coll = UsdPhysics.MeshCollisionAPI.Apply(prim)
    # Options: "none", "convexHull", "convexDecomposition",
    #          "meshSimplification", "boundingSphere", "boundingCube"
    mesh_coll.CreateApproximationAttr("convexHull")

    result = {"success": True}
except Exception as e:
    result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}
print(json.dumps(result))
```

| Approximation | Use case |
|---------------|----------|
| `"none"` | Planes and analytic primitives (Sphere, Cylinder, Cube) |
| `"convexHull"` | Simple convex objects — fastest, most stable |
| `"convexDecomposition"` | Concave shapes decomposed into multiple convex parts |
| `"meshSimplification"` | Complex meshes with moderate accuracy |
| `"boundingSphere"` | Rough proxy — very fast, low accuracy |
| `"boundingCube"` | Box-shaped proxy |
