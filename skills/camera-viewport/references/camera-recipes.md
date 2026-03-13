# Camera Recipes Reference

## Camera Prim Attributes

When creating a Camera prim via `create_isaac_prim` + `set_isaac_prim_attribute`, these are the standard USD Camera schema attributes:

| Attribute | Type | Default | Description |
|---|---|---|---|
| `focalLength` | float | 24.0 | Focal length in mm. Shorter = wider FOV. |
| `horizontalAperture` | float | 20.955 | Sensor width in mm (matches 35mm film standard). |
| `verticalAperture` | float | 15.2908 | Sensor height in mm. |
| `clippingRange` | float2 | [0.1, 10000.0] | Near/far clip planes in cm (USD convention). |
| `fStop` | float | 0.0 | Aperture f-stop. 0.0 = no depth of field. |
| `focusDistance` | float | 400.0 | Focus distance in cm when fStop > 0. |

### Field of View vs Focal Length

Horizontal FOV (degrees) ≈ 2 × atan(horizontalAperture / (2 × focalLength)) × (180/π)

Common focal lengths:
- 14mm → ~90° FOV (ultra-wide)
- 24mm → ~65° FOV (wide)
- 50mm → ~40° FOV (standard)
- 85mm → ~24° FOV (telephoto)

## Viewport Capture Best Practices

### Resolution
- Standard preview: 1280×720
- Full HD: 1920×1080
- 4K: 3840×2160
- Square (for top-down analysis): 1024×1024

### Format
- `"png"` — lossless, best for analysis or images that will be processed further
- `"jpg"` — smaller file size, acceptable for display/reporting

### Before Capturing
1. Pause physics if running: `pause_isaac_simulation`
2. Ensure the camera is positioned (call `set_isaac_camera` or `focus_isaac_viewport`)
3. Let the renderer settle for one frame if ray-tracing is enabled

## set_camera_view Script Pattern

When you need to set the viewport camera programmatically via `execute_isaac_script`, use `isaacsim.core.utils.viewports`:

```python
import json
from isaacsim.core.utils.viewports import set_camera_view

# Set the default viewport camera
set_camera_view(
    eye=[5.0, 5.0, 3.0],
    target=[0.0, 0.0, 0.0],
    camera_prim_path="/OmniverseKit_Persp"
)

print(json.dumps({"status": "camera set"}))
```

This is equivalent to calling `mcp__simul__set_isaac_camera` but useful inside a larger script that also does other operations.

## Creating a Camera with Full Optics via Script

```python
import json
import omni.usd
from pxr import UsdGeom, Gf

ctx = omni.usd.get_context()
stage = ctx.get_stage()

cam_path = "/World/MyCamera"
cam_prim = UsdGeom.Camera.Define(stage, cam_path)

# Optics
cam_prim.GetFocalLengthAttr().Set(24.0)
cam_prim.GetHorizontalApertureAttr().Set(20.955)
cam_prim.GetVerticalApertureAttr().Set(15.2908)
cam_prim.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 10000.0))

print(json.dumps({"created": cam_path}))
```

Send via `execute_isaac_script` when you need to set multiple attributes atomically.
