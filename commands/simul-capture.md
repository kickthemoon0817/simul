---
description: Capture the current viewport with optional camera positioning
argument-hint: "[camera-path-or-preset]"
allowed-tools: mcp__simul__list_isaac_cameras, mcp__simul__set_isaac_camera, mcp__simul__focus_isaac_viewport, mcp__simul__capture_isaac_viewport
---

Capture a 1920×1080 PNG screenshot of the Isaac Sim viewport, with optional camera repositioning before capture.

1. Inspect the argument to decide which branch to take:

   **Preset name** (`top`, `front`, `side`, `perspective`):
   Call `set_isaac_camera` with the corresponding values:
   - `top`:         eye=[0, 0, 10],  target=[0, 0, 0], up=[0, 1, 0]
   - `front`:       eye=[0, -10, 2], target=[0, 0, 0], up=[0, 0, 1]
   - `side`:        eye=[10, 0, 2],  target=[0, 0, 0], up=[0, 0, 1]
   - `perspective`: eye=[5, 5, 3],   target=[0, 0, 0], up=[0, 0, 1]

   **Prim path** (argument starts with `/`):
   Call `focus_isaac_viewport` with that prim path to auto-frame the object.

   **No argument**:
   Skip camera setup and capture the current view as-is.

2. Call `capture_isaac_viewport` with:
   - `width=1920`
   - `height=1080`
   - `format="png"`

3. Present the captured image inline to the user.

4. After the image, print a one-line caption showing:
   - Camera mode used (preset name, prim path, or "current view")
   - Resolution
   - Example: `Captured — perspective preset, 1920×1080`

If the capture fails, call `list_isaac_cameras` to check available cameras and suggest a valid path or preset.
