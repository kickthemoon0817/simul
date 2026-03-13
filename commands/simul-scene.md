---
description: Get a comprehensive overview of the current Isaac Sim scene
allowed-tools: mcp__simul__get_isaac_stage_info, mcp__simul__get_isaac_scene_summary, mcp__simul__get_isaac_scene_stats, mcp__simul__get_isaac_simulation_state, mcp__simul__list_isaac_cameras, mcp__simul__list_isaac_lights, mcp__simul__list_isaac_materials, mcp__simul__list_isaac_physics_objects
---

Produce a comprehensive structured overview of the current Isaac Sim scene.

1. Call these four tools in parallel (they are independent):
   - `get_isaac_stage_info` — stage URL, up axis, meters per unit
   - `get_isaac_scene_summary` — prim counts by type, hierarchy overview
   - `get_isaac_scene_stats` — total prims, meshes, vertices
   - `get_isaac_simulation_state` — playing/paused/stopped, current simulation time

2. Once those complete, call these four tools in parallel:
   - `list_isaac_cameras` — all camera prims and their paths
   - `list_isaac_lights` — all light prims and their types (Distant, Dome, Sphere, Rect, etc.)
   - `list_isaac_materials` — all materials bound in the stage
   - `list_isaac_physics_objects` — rigid bodies, colliders, joints

3. Present the results in clearly labelled sections:

   **Stage**
   - URL, up axis, meters per unit, default prim

   **Simulation State**
   - Status (playing/paused/stopped), current time, time step

   **Scene Statistics**
   - Total prims, mesh count, vertex count, prim breakdown by type

   **Cameras** (count + list of paths)

   **Lights** (count + list showing path and light type)

   **Materials** (count + list of material paths)

   **Physics Objects**
   - Rigid bodies count + paths
   - Colliders count
   - Joints count + types

4. End with a one-line health note, e.g.:
   `Scene looks well-formed: physics scene present, 3 rigid bodies, 3 colliders, 2 lights, 1 camera.`
   Or flag obvious issues: missing physics scene, no lights, zero cameras, etc.
