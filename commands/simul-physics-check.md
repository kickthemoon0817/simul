---
description: Audit physics configuration of the current scene
argument-hint: "[prim-path]"
allowed-tools: mcp__simul__get_isaac_physics_scene, mcp__simul__list_isaac_physics_objects, mcp__simul__get_isaac_rigid_body_info, mcp__simul__get_isaac_collision_info, mcp__simul__get_isaac_mass_properties
---

Audit physics configuration of the current Isaac Sim scene, with optional deep inspection of a specific prim.

1. Call `get_isaac_physics_scene` to check whether a physics scene is configured.
   - If none found: print a WARNING block — "No physics scene detected. Simulation will not run correctly. Create one via `UsdPhysics.Scene.Define(stage, '/World/PhysicsScene')`."
   - If found: note gravity vector, simulation rate, and any solver settings.

2. Call `list_isaac_physics_objects` to enumerate all physics-enabled prims:
   - Rigid bodies (dynamic)
   - Colliders
   - Joints and their types

3. If a `[prim-path]` argument was provided (starts with `/`), deep-inspect that prim by calling these three tools in parallel:
   - `get_isaac_rigid_body_info` — rigid body enabled, kinematic flag, linear/angular velocity
   - `get_isaac_collision_info` — collision API enabled, approximation shape (convex hull, mesh, box, etc.)
   - `get_isaac_mass_properties` — mass value, center of mass position, diagonal inertia

4. Cross-reference the full object list and emit warnings for any of these issues:
   - Rigid body prims that have no corresponding collider (will fall through the floor)
   - Collider prims with no mass or rigid body (static is fine — note it explicitly as "static collider")
   - Objects whose mass is exactly 1.0 kg and not set intentionally (likely the USD default — flag for review)
   - Missing physics scene (already covered in step 1)

5. Present findings in sections: **Physics Scene**, **Object Inventory**, **Deep Inspection** (if prim-path given), **Warnings & Recommendations**.
