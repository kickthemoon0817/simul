# Isaac Sim 5.1.0 Replicator / Synthetic Data API Reference

Module: `omni.replicator.core` (aliased as `rep`)

---

## Render Product

A render product binds a camera to a resolution and is the attachment point for annotators.

```python
import omni.replicator.core as rep

rp = rep.create.render_product("/World/Camera", resolution=(1920, 1080))
rp.destroy()
```

**Lifecycle**

```
create -> attach annotators -> step/update -> get_data -> detach -> destroy
```

`rep.create.render_product(camera_path: str, resolution: tuple[int, int]) -> RenderProduct`

| Method | Description |
|---|---|
| `rp.destroy()` | Release GPU/CPU resources; call after all annotators are detached |

---

## Annotator System

### AnnotatorRegistry

```python
from omni.replicator.core import AnnotatorRegistry

ann = AnnotatorRegistry.get_annotator("rgb")
ann = AnnotatorRegistry.get_annotator("normals", device="cuda")
```

`AnnotatorRegistry.get_annotator(name: str, device: str = "cpu") -> Annotator`

- `device`: `"cpu"` (default) or `"cuda"` — where the output tensor is placed.
- Returns a shared annotator instance; multiple calls with the same name return the same object.

### Annotator

```python
ann.attach([rp])          # begin receiving data from render products
ann.attach([rp1, rp2])    # multiple render products

ann.detach([rp])          # stop receiving; call before rp.destroy()
data = ann.get_data()     # returns dict or ndarray depending on annotator
data = ann.get_data(device="cuda")  # override device at read time
```

| Method | Signature | Notes |
|---|---|---|
| `attach` | `attach(render_products: list[RenderProduct]) -> None` | Idempotent; safe to call repeatedly |
| `detach` | `detach(render_products: list[RenderProduct]) -> None` | Must precede `rp.destroy()` |
| `get_data` | `get_data(device: str \| None = None) -> dict \| ndarray` | Returns last rendered frame; blocks until data is available |

---

## Annotator Reference

### Color / Lighting

| Name | Shape / Type | Notes |
|---|---|---|
| `rgb` | `(H, W, 4)` uint8 | RGBA; alpha channel is always 255 |
| `HdrColor` | `(H, W, 4)` float32 | Linear HDR RGBA before tone-mapping |
| `DirectDiffuse` | `(H, W, 4)` float32 | Direct diffuse lighting contribution |
| `DirectSpecular` | `(H, W, 4)` float32 | Direct specular lighting contribution |
| `IndirectDiffuse` | `(H, W, 4)` float32 | Indirect (GI) diffuse contribution |
| `Reflections` | `(H, W, 4)` float32 | Reflection pass |
| `AmbientOcclusion` | `(H, W, 1)` float32 | Screen-space AO, range [0, 1] |

### Geometry / Depth

| Name | Shape / Type | Notes |
|---|---|---|
| `normals` | `(H, W, 3)` float32 | World-space normals, range [-1, 1] |
| `distance_to_camera` | `(H, W)` float32 | Depth in metres along camera ray |
| `distance_to_image_plane` | `(H, W)` float32 | Orthographic depth (Z-buffer) |
| `pointcloud` | `(N, 3)` float32 | 3-D points in world space; N = visible pixels |
| `motion_vectors` | `(H, W, 4)` float32 | 2-D screen-space motion (xy) + depth motion (z), w unused |

### Segmentation

| Name | Shape / Type | Notes |
|---|---|---|
| `semantic_segmentation` | `(H, W)` uint32 | Per-pixel semantic class ID |
| `instance_segmentation` | `(H, W)` uint32 | Per-pixel instance ID |
| `instance_id_segmentation` | `(H, W)` uint32 | Raw instance ID without semantic mapping |

`get_data()` for segmentation annotators returns a dict:

```python
data = ann.get_data()
# data["data"]       -> (H, W) uint32 ndarray
# data["info"]       -> {"idToLabels": {id: label, ...}}
```

### Bounding Boxes

#### 2-D Bounding Box

```python
ann = AnnotatorRegistry.get_annotator("bounding_box_2d_tight")   # occluded pixels excluded
ann = AnnotatorRegistry.get_annotator("bounding_box_2d_loose")   # full projected extent
data = ann.get_data()
# data["data"] -> structured array with fields:
#   semanticId  uint32
#   x_min       int32
#   y_min       int32
#   x_max       int32
#   y_max       int32
#   occlusionRatio float32
# data["info"]["idToLabels"] -> {id: label}
```

#### 3-D Bounding Box

```python
ann = AnnotatorRegistry.get_annotator("bounding_box_3d")
data = ann.get_data()
# data["data"] -> structured array with fields:
#   semanticId    uint32
#   occlusionRatio float32
#   transform     float32[4][4]  world-space transform of box center
#   x_min, y_min, z_min  float32  extents in local space
#   x_max, y_max, z_max  float32
# data["info"]["idToLabels"] -> {id: label}
```

### Camera Parameters

```python
ann = AnnotatorRegistry.get_annotator("camera_params")
data = ann.get_data()
```

`data` dict keys:

| Key | Type | Description |
|---|---|---|
| `cameraViewTransform` | float32[4][4] | View matrix (world → camera) |
| `cameraProjection` | float32[4][4] | Projection matrix |
| `cameraFocalLength` | float32 | Focal length in mm |
| `cameraHorizontalAperture` | float32 | Sensor width in mm |
| `cameraVerticalAperture` | float32 | Sensor height in mm |
| `cameraNearFar` | float32[2] | Near/far clip planes in m |
| `cameraAspectRatio` | float32 | width / height |
| `renderProductResolution` | int32[2] | (width, height) in pixels |

---

## Orchestrator

Controls the rendering/stepping loop.

```python
rep.orchestrator.step()                    # render exactly one frame (synchronous)
rep.orchestrator.step(rt_subframes=4)      # render with 4 ray-tracing sub-frames for motion blur
rep.orchestrator.start(num_frames=100)     # non-blocking; fires 100 frames then stops
rep.orchestrator.stop()                    # stop an ongoing run
rep.orchestrator.pause()                   # pause a running sequence
rep.orchestrator.resume()                  # resume after pause
```

**Status enum** (`rep.orchestrator.Status`)

| Value | Meaning |
|---|---|
| `STOPPED` | No active run |
| `STARTED` | Actively stepping |
| `PAUSED` | Run suspended, can resume |
| `STEPPING` | Mid single-step execution |

```python
status = rep.orchestrator.get_status()
if status == rep.orchestrator.Status.STOPPED:
    rep.orchestrator.start(num_frames=50)
```

**App-update mode** (headless / extension loop)

```python
import omni.kit.app
app = omni.kit.app.get_app()

rep.orchestrator.step()    # queue the render
app.update()               # pump the Kit event loop to flush GPU commands
data = ann.get_data()
```

---

## Writer System

Writers consume annotator data and serialize it to disk or cloud storage.

### WriterRegistry

```python
from omni.replicator.core import WriterRegistry

writer = WriterRegistry.get("BasicWriter",
    output_dir="/tmp/replicator_out",
    rgb=True,
    bounding_box_2d_tight=True,
    semantic_segmentation=True,
)
writer.attach([rp])
```

`WriterRegistry.get(name: str, **kwargs) -> Writer`

### Built-in Writers

| Writer | Key kwargs | Output format |
|---|---|---|
| `BasicWriter` | `output_dir`, per-annotator bool flags | PNG / NPY per frame |
| `CocoWriter` | `output_dir`, `semantic_segmentation`, `instance_segmentation` | COCO JSON + PNG |
| `KittiWriter` | `output_dir`, `bounding_box_2d_tight`, `bounding_box_3d` | KITTI `.txt` labels |
| `PoseWriter` | `output_dir`, `format` | JSON pose per frame |
| `PytorchWriter` | `output_dir` | `.pt` tensor files |

### Writer Methods

```python
writer.attach(render_products: list[RenderProduct]) -> None
writer.detach()                                         # flush and close
writer.initialize(output_dir: str, **kwargs) -> None    # called internally by get()
```

### BackendDispatch

Controls where data is written. Injected via `output_dir` scheme or explicit construction.

```python
from omni.replicator.core.backends import DiskBackend, S3Backend

# DiskBackend is used automatically when output_dir is a local path.
# S3Backend activates when output_dir starts with "s3://".

disk = DiskBackend(output_dir="/data/frames")
s3   = S3Backend(output_dir="s3://my-bucket/frames",
                 region="us-east-1")       # boto3 credentials from environment
```

---

## Domain Randomization

Module: `isaacsim.replicator.behavior`

```python
from isaacsim.replicator.behavior import (
    LightRandomizer,
    TextureRandomizer,
    LocationRandomizer,
    RotationRandomizer,
)
```

### LightRandomizer

```python
randomizer = LightRandomizer(
    prim_paths=["/World/Lights/DomeLight"],
    intensity_range=(500.0, 5000.0),
    color_temperature_range=(3000, 7000),
)
randomizer.randomize()   # apply one random sample
```

### TextureRandomizer

```python
randomizer = TextureRandomizer(
    prim_paths=["/World/Floor"],
    texture_list=["path/to/tex1.png", "path/to/tex2.png"],
)
randomizer.randomize()
```

### LocationRandomizer

```python
randomizer = LocationRandomizer(
    prim_paths=["/World/Objects/Box"],
    min_range=(-2.0, -2.0, 0.0),   # (x, y, z) metres
    max_range=( 2.0,  2.0, 0.0),
)
randomizer.randomize()
```

### RotationRandomizer

```python
randomizer = RotationRandomizer(
    prim_paths=["/World/Objects/Box"],
    min_range=(0.0, 0.0, 0.0),     # (roll, pitch, yaw) degrees
    max_range=(0.0, 0.0, 360.0),
)
randomizer.randomize()
```

---

## Complete Example

```python
import omni.replicator.core as rep
from omni.replicator.core import AnnotatorRegistry, WriterRegistry

# 1. Create render product
rp = rep.create.render_product("/World/Camera", resolution=(1280, 720))

# 2. Attach annotators
ann_rgb   = AnnotatorRegistry.get_annotator("rgb")
ann_depth = AnnotatorRegistry.get_annotator("distance_to_camera")
ann_bbox  = AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
ann_seg   = AnnotatorRegistry.get_annotator("semantic_segmentation")

for ann in (ann_rgb, ann_depth, ann_bbox, ann_seg):
    ann.attach([rp])

# 3. Optional: attach writer for automatic serialisation
writer = WriterRegistry.get("BasicWriter",
    output_dir="/tmp/synth_data",
    rgb=True,
    distance_to_camera=True,
    bounding_box_2d_tight=True,
    semantic_segmentation=True,
)
writer.attach([rp])

# 4. Render loop
for frame_idx in range(50):
    rep.orchestrator.step()

    rgb_data   = ann_rgb.get_data()           # (720, 1280, 4) uint8
    depth_data = ann_depth.get_data()         # (720, 1280)    float32
    bbox_data  = ann_bbox.get_data()["data"]  # structured array
    seg_data   = ann_seg.get_data()["data"]   # (720, 1280)    uint32

# 5. Clean up
for ann in (ann_rgb, ann_depth, ann_bbox, ann_seg):
    ann.detach([rp])

writer.detach()
rp.destroy()
```

---

## Notes

- Always call `ann.detach([rp])` before `rp.destroy()` to avoid dangling GPU handles.
- `get_data()` is synchronous with respect to the last `step()` call; no polling required.
- `device="cuda"` on `get_annotator` avoids a CPU round-trip for downstream GPU pipelines (PyTorch, Warp).
- For real-time operation avoid calling `WriterRegistry.get` inside the render loop; construct writers once and reuse.
- `rep.orchestrator.start(num_frames=N)` is non-blocking; pair with `rep.orchestrator.wait_until_complete()` to synchronise.
- Semantic labels are registered via `rep.settings.set_stage_up_axis` and USD `semanticType`/`semanticData` attributes on prims.
