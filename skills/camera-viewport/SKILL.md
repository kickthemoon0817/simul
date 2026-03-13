---
name: camera-viewport
description: This skill should be used when the user asks to "set the camera", "change camera position", "capture the viewport", "take a screenshot", "render the scene", "focus on a prim", "look at", "create a camera", "camera view", or needs to control cameras and capture images in Isaac Sim.
version: 0.1.0
---

# Camera and Viewport Control

This skill teaches you how to discover, configure, and capture cameras in a live Isaac Sim 5.1.0 scene using the `mcp__simul__*` tools. Use this workflow any time the user wants to frame a shot, take a screenshot, or create a new camera.

## Tool Chain Overview

```
list_isaac_cameras
    → get_isaac_camera_info       (inspect existing camera)
    → set_isaac_camera            (reposition camera)
    → focus_isaac_viewport        (auto-frame a prim)
    → capture_isaac_viewport      (export image)

create_isaac_prim (type="Camera")
    → set_isaac_prim_attribute    (focal length, aperture, clipping)
    → set_isaac_camera            (move into position)
    → capture_isaac_viewport
```

## Step-by-Step Workflow

### 1. Discover Existing Cameras

Always start by listing what cameras are already in the stage. Do not assume a camera exists.

```
mcp__simul__list_isaac_cameras
```

Returns a list of camera prim paths (e.g. `/OmniverseKit_Persp`, `/World/Camera`).

### 2. Inspect a Camera

Before repositioning, inspect the camera to understand its current state.

```
mcp__simul__get_isaac_camera_info
  prim_path: "/OmniverseKit_Persp"
```

Returns focal length, aperture, clipping range, current eye/target position, and resolution.

### 3. Reposition a Camera

Set the camera's eye point, look-at target, and up vector. All coordinates are in world space (meters).

```
mcp__simul__set_isaac_camera
  camera_path: "/OmniverseKit_Persp"
  eye: [5.0, 5.0, 3.0]
  target: [0.0, 0.0, 0.0]
  up: [0.0, 0.0, 1.0]
```

**Common Presets:**

| View        | eye             | target        |
|-------------|-----------------|---------------|
| Perspective | [5, 5, 3]       | [0, 0, 0]     |
| Top-down    | [0, 0, 10]      | [0, 0, 0]     |
| Front       | [0, -10, 2]     | [0, 0, 0]     |
| Side        | [10, 0, 2]      | [0, 0, 0]     |
| Close-up    | [1.5, 1.5, 1.0] | [0, 0, 0.3]   |

Always set `up: [0, 0, 1]` unless the stage uses a Y-up convention, in which case use `[0, 1, 0]`.

### 4. Auto-Frame a Prim

When the user says "focus on X" or "look at the robot", use `focus_isaac_viewport` instead of manually computing eye/target. It fits the prim's bounding box into the viewport automatically.

```
mcp__simul__focus_isaac_viewport
  prim_path: "/World/Robot"
```

This is the fastest path when the user just wants to see a specific object.

### 5. Capture the Viewport

After positioning the camera, capture the image. Resolution and format are optional.

```
mcp__simul__capture_isaac_viewport
  width: 1920
  height: 1080
  format: "png"
```

**Capture tips:**
- Default resolution is the current viewport size if width/height are omitted.
- Use `format: "png"` for lossless images (recommended for analysis).
- Use `format: "jpg"` for smaller file size when quality is less critical.
- Capture returns the file path where the image was saved.
- If the scene has physics running, pause it before capture for a clean frame:
  `mcp__simul__pause_isaac_simulation` → capture → `mcp__simul__start_isaac_simulation`.

### 6. Create a New Camera Prim

When the user wants a persistent camera (not just repositioning the default viewport camera), create a Camera prim and configure its optics.

**Step 6a — Create the prim:**
```
mcp__simul__create_isaac_prim
  prim_path: "/World/MyCamera"
  prim_type: "Camera"
```

**Step 6b — Set optics attributes:**
```
mcp__simul__set_isaac_prim_attribute
  prim_path: "/World/MyCamera"
  attribute_name: "focalLength"
  value: 24.0

mcp__simul__set_isaac_prim_attribute
  prim_path: "/World/MyCamera"
  attribute_name: "horizontalAperture"
  value: 20.955

mcp__simul__set_isaac_prim_attribute
  prim_path: "/World/MyCamera"
  attribute_name: "verticalAperture"
  value: 15.2908

mcp__simul__set_isaac_prim_attribute
  prim_path: "/World/MyCamera"
  attribute_name: "clippingRange"
  value: [0.1, 10000.0]
```

**Step 6c — Position it:**
```
mcp__simul__set_isaac_camera
  camera_path: "/World/MyCamera"
  eye: [5.0, 5.0, 3.0]
  target: [0.0, 0.0, 0.0]
  up: [0.0, 0.0, 1.0]
```

See `references/camera-recipes.md` for the equivalent `execute_isaac_script` pattern using `isaacsim.core.utils.viewports`.

## Decision Guide

| User Intent | Tool to Use |
|---|---|
| "Show me the scene" | `set_isaac_camera` with perspective preset |
| "Focus on the robot" | `focus_isaac_viewport` with robot prim path |
| "Take a screenshot" | `capture_isaac_viewport` |
| "Create a camera named X" | `create_isaac_prim` → `set_isaac_prim_attribute` |
| "What cameras exist?" | `list_isaac_cameras` |
| "What is the focal length?" | `get_isaac_camera_info` |

## Common Pitfalls

- **Eye equals target**: If `eye` and `target` are the same point the camera has no direction. Always use distinct values.
- **Wrong up vector**: In a Z-up stage (Isaac Sim default), `up` must be `[0, 0, 1]`. Using `[0, 1, 0]` will roll the camera sideways.
- **Capturing before the scene loads**: If the stage was just opened, wait for it to settle (`get_isaac_stage_info` to confirm load state) before capturing.
- **Missing camera path**: `set_isaac_camera` requires an existing prim path. If the default viewport camera path is unknown, call `list_isaac_cameras` first.
