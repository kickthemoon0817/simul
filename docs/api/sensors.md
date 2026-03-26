# Isaac Sim 5.1.0 Sensor API Reference

Practical reference for the sensor APIs shipped with Isaac Sim 5.1.0.
All sensors follow the same lifecycle: construct → configure → attach to stage → call `get_current_frame()` (or a typed getter) each physics/render step.

---

## Table of Contents

1. [Camera](#camera)
2. [Contact Sensor](#contact-sensor)
3. [IMU Sensor](#imu-sensor)
4. [Effort Sensor](#effort-sensor)
5. [PhysX LiDAR (CPU)](#physx-lidar-cpu)
6. [RTX LiDAR (GPU)](#rtx-lidar-gpu)
7. [Proximity Sensor](#proximity-sensor)
8. [Common Notes](#common-notes)

---

## Camera

**Module:** `isaacsim.sensors.camera`

### Initialization

```python
from isaacsim.sensors.camera import Camera

cam = Camera(
    prim_path="/World/robot/camera_link/Camera",
    name="front_camera",
    frequency=30,                    # Hz; 0 = every render frame
    resolution=(1280, 720),          # (width, height) in pixels
    translation=None,                # Gf.Vec3d, optional
    orientation=None,                # Gf.Quatd (xyzw), optional
    render_product_path=None,        # re-use an existing RenderProduct prim
)
cam.initialize()                     # must be called after stage is live
```

### Annotator management

```python
cam.add_distance_to_image_plane_to_frame()   # adds "distance_to_image_plane"
cam.add_distance_to_camera_to_frame()        # adds "distance_to_camera"
cam.add_pointcloud_to_frame()                # adds "pointcloud"
cam.add_motion_vectors_to_frame()            # adds "motion_vectors"
cam.add_normals_to_frame()                   # adds "normals"
cam.add_occlusion_to_frame()                 # adds "occlusion"
cam.add_semantic_segmentation_to_frame()     # adds "semantic_segmentation"
cam.add_instance_segmentation_to_frame()     # adds "instance_segmentation"
cam.add_instance_id_segmentation_to_frame()  # adds "instance_id_segmentation"
cam.add_bounding_box_2d_tight_to_frame()     # adds "bounding_box_2d_tight"
cam.add_bounding_box_2d_loose_to_frame()     # adds "bounding_box_2d_loose"
cam.add_bounding_box_3d_to_frame()           # adds "bounding_box_3d"
```

### Data accessors

```python
# Grab all annotator outputs for this step
frame: dict = cam.get_current_frame(clone: bool = False)
# Keys present depend on which annotators were added, e.g.:
#   frame["rgba"]                    ndarray (H, W, 4)  uint8
#   frame["distance_to_image_plane"] ndarray (H, W)     float32  [metres]
#   frame["normals"]                 ndarray (H, W, 4)  float32
#   frame["pointcloud"]              ndarray (N, 3)      float32  [metres]
#   frame["semantic_segmentation"]   ndarray (H, W)      uint32
#   frame["bounding_box_3d"]         structured array

# Typed convenience getters (thin wrappers over get_current_frame)
rgb:   np.ndarray = cam.get_rgb(device: str | None = None)
# -> (H, W, 3) uint8

depth: np.ndarray = cam.get_depth(device: str | None = None)
# -> (H, W) float32, metres, inf where no return

pc:    np.ndarray = cam.get_pointcloud(
    device: str | None = None,
    world_frame: bool = True,
)
# -> (H, W, 3) float32; reshape to (-1, 3) for a flat list of XYZ points

K:     np.ndarray = cam.get_intrinsics_matrix(device: str | None = None)
# -> (3, 3) float64
# [[fx,  0, cx],
#  [ 0, fy, cy],
#  [ 0,  0,  1]]

T_view: np.ndarray = cam.get_view_matrix_ros(device: str | None = None)
# -> (4, 4) float64, camera-from-world transform in ROS (Z-forward) convention
```

`device` accepts `"cpu"`, `"cuda"`, or `None` (returns a plain NumPy array on CPU).

### Frame dict annotator keys

| Key | Shape | dtype | Notes |
|-----|-------|-------|-------|
| `rgba` | (H, W, 4) | uint8 | RGBA colour |
| `distance_to_image_plane` | (H, W) | float32 | Depth along optical axis |
| `distance_to_camera` | (H, W) | float32 | Euclidean distance |
| `normals` | (H, W, 4) | float32 | World-space normals + alpha |
| `motion_vectors` | (H, W, 4) | float32 | Screen-space motion |
| `occlusion` | structured | — | Per-instance occlusion ratios |
| `pointcloud` | (N, 3) | float32 | World or camera frame XYZ |
| `semantic_segmentation` | (H, W) | uint32 | Semantic label IDs |
| `instance_segmentation` | (H, W) | uint32 | Instance IDs |
| `instance_id_segmentation` | (H, W) | uint32 | Unique prim IDs |
| `bounding_box_2d_tight` | structured | — | Pixel-tight 2-D boxes |
| `bounding_box_2d_loose` | structured | — | Pixel-loose 2-D boxes |
| `bounding_box_3d` | structured | — | 3-D OBBs in world space |

---

## Contact Sensor

**Module:** `isaacsim.sensors.physics`

### Initialization

```python
from isaacsim.sensors.physics import ContactSensor

cs = ContactSensor(
    prim_path="/World/robot/base_link/ContactSensor",
    name="base_contact",
    min_threshold: float = 0.0,    # N; readings below this are suppressed
    max_threshold: float = 1e8,    # N; readings above this are clamped
    radius: float = -1.0,          # m; -1 = use the prim's own collision shape
    dt: float = 0.0,               # integration interval; 0 = one physics step
    translation=None,              # Gf.Vec3d, optional offset
    orientation=None,              # Gf.Quatd (xyzw), optional offset
)
cs.initialize()
```

### Data accessor

```python
frame: dict = cs.get_current_frame()
```

Returned keys:

| Key | Type | Description |
|-----|------|-------------|
| `in_contact` | `bool` | `True` when any contact exceeds `min_threshold` |
| `force` | `np.ndarray (3,)` | Net contact force vector in the sensor frame [N] |
| `number_of_contacts` | `int` | Count of active contact pairs |
| `contacts` | `list[dict]` | Per-pair details (see below) |
| `time` | `float` | Simulation time [s] |
| `physics_step` | `int` | Physics step counter |

Each entry in `contacts`:

```python
{
    "body0":  str,               # prim path of the first body
    "body1":  str,               # prim path of the second body
    "position": np.ndarray(3,),  # contact point in world frame [m]
    "normal":   np.ndarray(3,),  # contact normal (points from body1 to body0)
    "impulse":  np.ndarray(3,),  # impulse [N·s] over the integration interval
}
```

---

## IMU Sensor

**Module:** `isaacsim.sensors.physics`

### Initialization

```python
from isaacsim.sensors.physics import IMUSensor

imu = IMUSensor(
    prim_path="/World/robot/imu_link/IMU",
    name="base_imu",
    dt: float = 0.0,                        # 0 = one physics step
    linear_acceleration_filter_size: int = 1,
    angular_velocity_filter_size: int = 1,
    orientation_filter_size: int = 1,
    translation=None,                        # Gf.Vec3d offset
    orientation=None,                        # Gf.Quatd (xyzw) offset
)
imu.initialize()
```

Filter sizes are simple moving-average window lengths applied to each signal independently.

### Data accessor

```python
frame: dict = imu.get_current_frame(read_gravity: bool = True)
```

Returned keys:

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `lin_acc` | `np.ndarray` | (3,) | Linear acceleration [m/s²] in sensor frame; includes gravity when `read_gravity=True` |
| `ang_vel` | `np.ndarray` | (3,) | Angular velocity [rad/s] in sensor frame |
| `orientation` | `np.ndarray` | (4,) | Quaternion `[x, y, z, w]` in world frame |
| `time` | `float` | — | Simulation time [s] |
| `physics_step` | `int` | — | Physics step counter |

---

## Effort Sensor

**Module:** `isaacsim.sensors.physics`

Measures joint torque / force at a single articulation joint DOF.

### Initialization

```python
from isaacsim.sensors.physics import EffortSensor

es = EffortSensor(
    prim_path="/World/robot/joint1",   # must be a PhysicsJoint prim
    sensor_period: float = -1.0,       # s; -1 = every physics step
    use_latest_data: bool = False,
    enabled: bool = True,
)
es.initialize()
```

### Data accessor

```python
from isaacsim.sensors.physics import EsSensorReading

reading: EsSensorReading = es.get_sensor_reading(
    interpolation_function=None,   # optional callable(t0, t1, readings) -> float
    use_latest_data: bool = False,
)
```

`EsSensorReading` fields:

| Field | Type | Description |
|-------|------|-------------|
| `is_valid` | `bool` | `False` until the first physics step has run |
| `time` | `float` | Simulation time of the reading [s] |
| `value` | `float` | Joint effort [N] (linear) or [N·m] (revolute) |

---

## PhysX LiDAR (CPU)

**Module:** `isaacsim.sensors.physx`

CPU-based rotating LiDAR using PhysX raycasting. Suitable for moderate beam counts and deterministic replay.

### Initialization

```python
from isaacsim.sensors.physx import RotatingLidarPhysX

lidar = RotatingLidarPhysX(
    prim_path="/World/robot/lidar_link/LidarPhysX",
    name="front_lidar",
    translation=None,     # Gf.Vec3d
    orientation=None,     # Gf.Quatd (xyzw)
)
lidar.initialize()
```

### Configuration

Call `set_*` helpers **before** `initialize()`, or use the USD attribute path directly.

```python
lidar.set_fov((360.0, 30.0))           # (horizontal °, vertical °)
lidar.set_resolution((0.4, 1.0))       # (horizontal °/beam, vertical °/beam)
lidar.set_rotation_frequency(10.0)     # Hz
lidar.set_valid_range((0.1, 100.0))    # (min m, max m)
lidar.enable_semantics(True)           # attach semantic labels
```

### Enabling data channels

```python
lidar.add_depth_data_to_frame()
lidar.add_point_cloud_data_to_frame()
lidar.add_intensity_data_to_frame()
lidar.add_azimuth_data_to_frame()
lidar.add_zenith_data_to_frame()
lidar.add_linear_depth_data_to_frame()
lidar.add_object_id_data_to_frame()     # requires enable_semantics(True)
lidar.add_semantic_data_to_frame()      # requires enable_semantics(True)
```

### Data accessor

```python
frame: dict = lidar.get_current_frame()
```

| Key | Shape | dtype | Description |
|-----|-------|-------|-------------|
| `depth` | (V, H) | float32 | Depth along beam axis [m] |
| `point_cloud` | (N, 3) | float32 | XYZ in sensor or world frame |
| `intensity` | (V, H) | float32 | Normalised return intensity [0, 1] |
| `azimuth` | (V, H) | float32 | Azimuth angle per beam [rad] |
| `zenith` | (V, H) | float32 | Elevation angle per beam [rad] |
| `linear_depth` | (V, H) | float32 | Euclidean distance to hit [m] |
| `object_id` | (V, H) | uint32 | Per-beam prim instance ID |
| `semantic` | (V, H) | uint32 | Per-beam semantic label ID |

V = vertical beam count, H = horizontal beam count per full rotation.

---

## RTX LiDAR (GPU)

**Module:** `isaacsim.sensors.rtx`

GPU ray-traced LiDAR with physically accurate sensor simulation. Required for high beam counts (512-channel spinning sensors, solid-state patterns).

### Initialization

```python
from isaacsim.sensors.rtx import LidarRtx

lidar = LidarRtx(
    prim_path="/World/robot/lidar_link/LidarRtx",
    name="top_lidar",
    config_file_name="Example_Rotary",  # sensor config in nucleus / local JSON
    translation=None,                   # Gf.Vec3d
    orientation=None,                   # Gf.Quatd (xyzw)
)
lidar.initialize()
```

Sensor parameters (FOV, beam pattern, range, pulse shape) are fully defined inside the JSON config file; no runtime setters exist for RTX sensors.

### Enabling data channels

```python
lidar.add_point_cloud_data_to_frame()
lidar.add_linear_depth_data_to_frame()
lidar.add_intensity_data_to_frame()
lidar.add_azimuth_data_to_frame()
lidar.add_zenith_data_to_frame()
lidar.add_range_data_to_frame()
lidar.add_object_id_data_to_frame()
lidar.add_semantic_data_to_frame()
```

### Annotator-based pipeline (low latency)

For tighter integration with the render pipeline, attach annotators directly:

```python
import omni.replicator.core as rep

render_product = lidar.get_render_product_path()

# Flat scan (single-return 2-D array)
flat_scan = rep.AnnotatorRegistry.get_annotator("IsaacComputeRTXLidarFlatScan")
flat_scan.attach(render_product)

# Full 3-D point cloud
pc_anno = rep.AnnotatorRegistry.get_annotator("RtxSensorCpuIsaacCreateRTXLidarScanBuffer")
pc_anno.attach(render_product)

# Per-step readout
data: dict = flat_scan.get_data()   # dict with "data", "info" keys
```

### Data accessor

```python
frame: dict = lidar.get_current_frame()
```

| Key | Shape | dtype | Description |
|-----|-------|-------|-------------|
| `point_cloud` | (N, 3) | float32 | XYZ in world frame [m] |
| `linear_depth` | (N,) | float32 | Euclidean distance [m] |
| `intensity` | (N,) | float32 | Normalised intensity [0, 1] |
| `azimuth` | (N,) | float32 | Azimuth per return [rad] |
| `zenith` | (N,) | float32 | Elevation per return [rad] |
| `range` | (N,) | float32 | Same as `linear_depth`, alias |
| `object_id` | (N,) | uint32 | Prim instance ID per return |
| `semantic` | (N,) | uint32 | Semantic label ID per return |

N = number of valid returns in this frame (varies with occlusion and sensor pattern).

---

## Proximity Sensor

**Module:** `isaacsim.sensors.physx`

Overlap-based trigger sensor; reports which prims enter, remain in, or exit the detection volume each step.

### Initialization

```python
from isaacsim.sensors.physx import ProximitySensor

prox = ProximitySensor(
    prim_path="/World/robot/hand/ProximitySensor",
    name="hand_proximity",
    translation=None,    # Gf.Vec3d
    orientation=None,    # Gf.Quatd (xyzw)
)
prox.initialize()
```

The detection volume shape is defined by the collision geometry attached to the prim.

### Callbacks

```python
def on_enter(event) -> None:
    print("entered:", event.prim_path)

def on_stay(event) -> None:
    pass

def on_exit(event) -> None:
    print("exited:", event.prim_path)

prox.add_on_enter_callback(on_enter)
prox.add_on_stay_callback(on_stay)
prox.add_on_exit_callback(on_exit)
```

Callbacks fire synchronously on the physics thread; keep them short.

### Data accessor

```python
data: dict = prox.get_data()
```

| Key | Type | Description |
|-----|------|-------------|
| `overlapped_prims` | `list[str]` | Prim paths currently inside the volume |
| `num_overlapping` | `int` | Count of overlapping prims |
| `time` | `float` | Simulation time [s] |

---

## Common Notes

### Sensor lifecycle

```python
# 1. Construct  — before simulation starts
sensor = SensorClass(prim_path=..., name=..., **kwargs)

# 2. Initialize — after the USD stage is open and the simulation world exists
sensor.initialize()

# 3. Per-step readout — inside the simulation loop
world.step(render=True)
frame = sensor.get_current_frame()
```

### Pause / resume

```python
sensor.pause()    # suspend data collection
sensor.resume()   # restart data collection
sensor.is_paused() -> bool
```

### Enabled flag

```python
sensor.set_enabled(True)
sensor.is_enabled() -> bool
```

### GPU / CPU data paths

Where `device` parameters are accepted, use:

- `None` — NumPy array on CPU (default, safest).
- `"cpu"` — explicit CPU, same as `None`.
- `"cuda"` — returns a `torch.Tensor` on the default CUDA device; avoids a GPU→CPU copy when the consumer (e.g. a neural network) already lives on GPU.

Data transferred with `device="cuda"` is only valid until the next render step; copy it if you need to hold it longer.

### Thread safety

All `get_current_frame()` calls must be made from the main simulation thread (the same thread that drives `world.step()`). Do not call sensor getters from background threads or `asyncio` coroutines that run off the main loop.
