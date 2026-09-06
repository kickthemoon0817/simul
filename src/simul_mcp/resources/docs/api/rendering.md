# Rendering API Reference — Isaac Sim 5.1.0

## ViewportAPI

Primary interface for controlling and querying a viewport window.

```python
import omni.kit.viewport.utility as vu

viewport_api = vu.get_active_viewport()
```

### Properties

| Property | Type | Description |
|---|---|---|
| `camera_path` | `str` | USD path to the active camera prim |
| `resolution` | `tuple[int, int]` | Viewport render resolution `(width, height)` |
| `hydra_engine` | `str` | Active Hydra render delegate (e.g. `"rtx"`, `"iray"`) |
| `render_mode` | `str` | Render mode string (e.g. `"PathTracing"`, `"RayTracing"`) |
| `fps` | `float` | Measured rendering frame rate |
| `frame_info` | `dict` | Metadata for the most recently rendered frame |

### View and Projection Matrices

```python
view_matrix       = viewport_api.view             # np.ndarray (4x4) world-to-camera
projection_matrix = viewport_api.projection       # np.ndarray (4x4) camera-to-clip
```

### Setting Camera and Resolution

```python
viewport_api.camera_path = "/World/Camera"
viewport_api.resolution  = (1920, 1080)
```

---

## Viewport Utility Functions

### get_active_viewport

```python
from omni.kit.viewport.utility import get_active_viewport

viewport_api = get_active_viewport()  # ViewportAPI | None
```

Returns the currently active `ViewportAPI`, or `None` if no viewport exists.

### capture_viewport_to_file

```python
from omni.kit.viewport.utility import capture_viewport_to_file

capture_viewport_to_file(
    viewport_api,
    file_path="/tmp/frame.png",
    resolution=(1920, 1080),   # optional override
)
```

| Parameter | Type | Description |
|---|---|---|
| `viewport_api` | `ViewportAPI` | Target viewport |
| `file_path` | `str` | Output image path (PNG, EXR, JPG) |
| `resolution` | `tuple[int, int] \| None` | Optional capture resolution override |

### frame_viewport_prims

```python
from omni.kit.viewport.utility import frame_viewport_prims

frame_viewport_prims(
    viewport_api=viewport_api,
    prim_paths=["/World/Robot", "/World/Table"],
)
```

Adjusts the camera so that all listed prims fit within the viewport frustum.

| Parameter | Type | Description |
|---|---|---|
| `viewport_api` | `ViewportAPI` | Target viewport |
| `prim_paths` | `list[str]` | Prims to frame |

---

## HydraTexture

Off-screen render target backed by a Hydra render product.

### Factory Function

```python
from omni.syntheticdata import sensors

hydra_texture = sensors.create_hydra_texture(
    name="my_sensor",
    width=1280,
    height=720,
    context=0,                         # GPU device index
    camera="/World/Camera",            # camera prim path
    renderer="RayTracedLighting",      # hydra delegate string
    is_async=True,                     # non-blocking capture
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | — | Unique texture name |
| `width` | `int` | — | Render width in pixels |
| `height` | `int` | — | Render height in pixels |
| `context` | `int` | `0` | CUDA device index |
| `camera` | `str` | — | USD path to camera prim |
| `renderer` | `str` | `"RayTracedLighting"` | Hydra render delegate |
| `is_async` | `bool` | `False` | Enable async (non-blocking) frame capture |

### Reading Back Pixels

```python
rgba = hydra_texture.get_rgba()    # np.ndarray shape (H, W, 4) uint8
depth = hydra_texture.get_depth()  # np.ndarray shape (H, W) float32
```

---

## RTX Post-Processing Settings

All RTX post-processing is controlled via USD settings under `/rtx/post/`.
Use `carb.settings.get_settings()` to read/write at runtime.

```python
import carb

s = carb.settings.get_settings()
s.set("/rtx/post/tonemap/op", 6)
s.set("/rtx/post/dof/enabled", True)
```

### Tone Mapping — `/rtx/post/tonemap/`

| Setting Key | Type | Description |
|---|---|---|
| `op` | `int` | Tone-map operator: `0`=Linear, `3`=Reinhard, `6`=ACES Film |
| `exposureTime` | `float` | Camera exposure time (seconds) |
| `filmIso` | `float` | Film sensitivity (ISO value) |
| `fNumber` | `float` | Aperture f-number |
| `whitepoint` | `float` | Scene white-point luminance |

### Depth of Field — `/rtx/post/dof/`

| Setting Key | Type | Description |
|---|---|---|
| `enabled` | `bool` | Enable depth-of-field effect |
| `focusDistance` | `float` | Distance to focal plane (cm) |
| `fNumber` | `float` | Aperture size (smaller = more blur) |
| `focalLength` | `float` | Lens focal length (mm) |

### Motion Blur — `/rtx/post/motionblur/`

| Setting Key | Type | Description |
|---|---|---|
| `enabled` | `bool` | Enable motion blur |
| `maxBlurDiameterFraction` | `float` | Max blur radius as fraction of image width |
| `exposureFraction` | `float` | Fraction of frame time used for blur |
| `numSamples` | `int` | Samples per blurred pixel |

### Atmospheric Fog — `/rtx/post/fog/`

| Setting Key | Type | Description |
|---|---|---|
| `enabled` | `bool` | Enable volumetric fog |
| `fogColor` | `list[float]` | RGB colour of the fog `[r, g, b]` |
| `fogColorIntensity` | `float` | Fog colour intensity |
| `fogStartDistance` | `float` | Distance fog begins (cm) |
| `fogEndDistance` | `float` | Distance fog reaches full density (cm) |

### Lens Flares — `/rtx/post/lensFlares/`

| Setting Key | Type | Description |
|---|---|---|
| `enabled` | `bool` | Enable lens flare simulation |
| `flareScale` | `float` | Global flare intensity |
| `cutoffPoint` | `float` | Luminance threshold before flares appear |
| `numBlades` | `int` | Aperture blade count (affects flare shape) |

### Chromatic Aberration — `/rtx/post/chromaticAberration/`

| Setting Key | Type | Description |
|---|---|---|
| `enabled` | `bool` | Enable chromatic aberration |
| `strengthR` | `float` | Red channel shift strength |
| `strengthG` | `float` | Green channel shift strength |
| `strengthB` | `float` | Blue channel shift strength |
| `lanczos` | `bool` | Use Lanczos reconstruction filter |

### Color Correction — `/rtx/post/colorCorrection/`

| Setting Key | Type | Description |
|---|---|---|
| `enabled` | `bool` | Enable colour correction pass |
| `brightness` | `float` | Brightness multiplier |
| `contrast` | `float` | Contrast multiplier |
| `saturation` | `float` | Colour saturation multiplier |

### Color Grading — `/rtx/post/colorGrading/`

| Setting Key | Type | Description |
|---|---|---|
| `enabled` | `bool` | Enable LUT-based colour grading |
| `lutFile` | `str` | Path to a `.cube` or `.png` LUT file |
| `lutStrength` | `float` | Blend strength of LUT (0.0–1.0) |

---

## CaptureOptions and Render Presets

### Render Presets

| Constant | String Value | Description |
|---|---|---|
| `RenderPreset.PATH_TRACE` | `"PathTracing"` | Full unidirectional path tracing |
| `RenderPreset.RAY_TRACE` | `"RayTracedLighting"` | Hybrid ray-traced lighting |
| `RenderPreset.IRAY` | `"iray"` | MDL-accurate physically based renderer |

Set the active render mode:

```python
import carb

carb.settings.get_settings().set("/rtx/rendermode", "PathTracing")
```

### Capture to File with Preset Override

```python
from omni.replicator.core import orchestrator, settings as rep_settings

# Set path-trace quality for capture
rep_settings.carb_settings().set("/rtx/pathtracing/spp", 64)
rep_settings.carb_settings().set("/rtx/pathtracing/totalSpp", 256)

capture_viewport_to_file(viewport_api, "/tmp/render.exr")
```

### EXR Compression Options

Set via `/rtx/exrCompression`:

| Value | Description |
|---|---|
| `"none"` | Uncompressed EXR |
| `"zip"` | ZIP lossless compression |
| `"zips"` | ZIP per-scanline compression |
| `"piz"` | PIZ wavelet (good for noisy data) |
| `"rle"` | Run-length encoding |
| `"b44"` | B44 lossy half-float compression |
| `"b44a"` | B44 with alpha masking |
| `"dwaa"` | DWA lossy, scanline blocks |
| `"dwab"` | DWA lossy, larger tile blocks |

```python
carb.settings.get_settings().set("/rtx/exrCompression", "dwab")
```

### Path-Tracing Quality Settings

| Setting Path | Type | Description |
|---|---|---|
| `/rtx/pathtracing/spp` | `int` | Samples per pixel per frame (interactive) |
| `/rtx/pathtracing/totalSpp` | `int` | Total SPP for converged offline capture |
| `/rtx/pathtracing/maxBounces` | `int` | Maximum ray bounce depth |
| `/rtx/pathtracing/maxSpecularAndTransmissionBounces` | `int` | Specular/transmission bounce cap |
| `/rtx/pathtracing/clampSpp` | `int` | SPP accumulation clamp |
