---
name: scene-setup
description: This skill should be used when the user asks to "set up a scene", "create a new scene", "add a ground plane", "add lighting", "create a basic scene", "scene from scratch", "empty scene", "new stage", or needs to build a complete 3D scene in Isaac Sim from the ground up.
version: 0.1.0
---

# Scene Setup Workflow

Building a complete Isaac Sim scene from scratch follows a deterministic sequence of MCP tool calls. Each step depends on the previous one completing successfully. Work through the steps in order and verify each result before proceeding.

## Step 0: Verify Connectivity

Before any scene work, confirm Isaac Sim is reachable:

```
mcp__simul__ping_isaac
```

If ping fails, Isaac Sim is not running or the MCP server is not connected. Do not proceed until ping succeeds.

## Step 1: Create a Blank Stage

```
mcp__simul__new_isaac_stage
```

This creates a fresh USD stage, clearing any existing scene. Use `mcp__simul__open_isaac_stage` instead if you want to load an existing `.usd` file as the starting point.

After this call, the stage contains only the default prim (`/World` or pseudoroot). Verify with `mcp__simul__get_isaac_stage_info`.

## Step 2: Add a Ground Plane

A ground plane gives physics objects a surface to rest on and provides visual grounding:

```
mcp__simul__create_isaac_prim
  prim_path: "/World/GroundPlane"
  prim_type: "Plane"
  scale: [50, 50, 1]
  position: [0, 0, 0]
```

The `Plane` type in Isaac Sim creates a flat infinite collision surface. The scale controls the visible mesh extent but not the physics extent (the physics plane is infinite by default). For a finite collision surface, use a thin `Cube` instead.

Then add collision to the ground plane so physics objects interact with it:

```
mcp__simul__add_isaac_collision
  prim_path: "/World/GroundPlane"
  collision_type: "none"
```

`collision_type: "none"` on a Plane means the plane uses its analytic shape (the correct choice for flat planes — mesh-based collision is unnecessary and less accurate).

## Step 3: Create a Physics Scene

The physics scene defines global simulation parameters including gravity. Create it via `execute_isaac_script` since there is no dedicated granular tool for this:

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

The gravity direction `(0, 0, -1)` is -Z (the Isaac Sim convention with Z-up). Gravity magnitude is in m/s². Check the existing physics scene at any time with `mcp__simul__get_isaac_physics_scene`.

## Step 4: Add Lighting

A scene without lights is black in rendered viewports. Create at minimum a distant light (sun) and a dome light (ambient fill):

```
mcp__simul__execute_isaac_script
  script: |
    import json, traceback
    try:
        from pxr import UsdLux, Gf, UsdGeom
        import omni.usd
        stage = omni.usd.get_context().get_stage()

        # Distant light — directional sun
        sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
        sun.CreateIntensityAttr(3000)
        sun.CreateAngleAttr(0.53)
        sun.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.85))

        # Dome light — ambient sky fill
        dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome.CreateIntensityAttr(1000)

        result = {"success": True, "lights": ["/World/Sun", "/World/DomeLight"]}
    except Exception as e:
        import traceback
        result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}
    print(json.dumps(result))
```

For full lighting reference including SphereLight (point light) and HDRI dome textures, see `references/lighting-patterns.md`.

## Step 5: Position the Camera

Set a good default viewpoint so the scene is immediately visible:

```
mcp__simul__set_isaac_camera
  eye: [5, 5, 3]
  target: [0, 0, 0]
```

This looks at the origin from a 45-degree angle at moderate distance — appropriate for a scene centred at the origin. Adjust `eye` for larger or smaller scenes. The camera prim used is the default perspective camera (`/OmniverseKit_Persp`).

To list available cameras first: `mcp__simul__list_isaac_cameras`.

## Step 6: Verify the Scene

Confirm everything is in place:

```
mcp__simul__get_isaac_scene_summary
```

This returns a high-level overview: prim count, physics objects, lights, cameras. Cross-check that the ground plane, physics scene, and lights all appear.

For a more detailed breakdown: `mcp__simul__get_isaac_scene_stats`.

## Step 7: Save the Stage

Persist the scene so it survives a restart:

```
mcp__simul__save_isaac_stage
  file_path: "/path/to/your_scene.usd"
```

If you omit `file_path`, the current stage URL is used. Always save before running long simulations.

## Adding Objects to the Scene

Once the base scene is ready, populate it with objects using `mcp__simul__create_isaac_prim` for built-in geometry types:

| Type | prim_type value |
|------|----------------|
| Box | `"Cube"` |
| Sphere | `"Sphere"` |
| Cylinder | `"Cylinder"` |
| Cone | `"Cone"` |
| Flat plane | `"Plane"` |
| Camera | `"Camera"` |

To load an external USD asset (robot, furniture, etc.): `mcp__simul__import_isaac_asset` or `mcp__simul__add_isaac_reference`.

To configure physics on any object after creation: use `mcp__simul__add_isaac_rigid_body` and `mcp__simul__add_isaac_collision` — see the `physics-configuration` skill.

## Minimal Complete Example (Tool Call Sequence)

For a quick working scene with ground, gravity, and lights in the fewest steps:

1. `mcp__simul__ping_isaac` — confirm connection
2. `mcp__simul__new_isaac_stage` — blank stage
3. `mcp__simul__create_isaac_prim` — ground plane at `/World/GroundPlane`, type `"Plane"`, scale `[50,50,1]`
4. `mcp__simul__add_isaac_collision` — collision on `/World/GroundPlane`, type `"none"`
5. `mcp__simul__execute_isaac_script` — create `/physicsScene` with gravity
6. `mcp__simul__execute_isaac_script` — create `DistantLight` + `DomeLight`
7. `mcp__simul__set_isaac_camera` — eye `[5,5,3]`, target `[0,0,0]`
8. `mcp__simul__get_isaac_scene_summary` — verify
9. `mcp__simul__save_isaac_stage` — persist

## Reference Files

- `references/lighting-patterns.md` — DistantLight, DomeLight, SphereLight creation patterns with parameters
- `references/ground-plane-recipes.md` — Ground plane variants and physics scene creation patterns
