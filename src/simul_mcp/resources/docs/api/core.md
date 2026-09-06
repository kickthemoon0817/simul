# Isaac Sim 5.1.0 — Core API Reference

**Scope:** `isaacsim.core.api`, `isaacsim.core.prims`, `isaacsim.core.utils`
**Source:**
- `~/isaac-sim-5.1.0/exts/isaacsim.core.api/isaacsim/core/api/`
- `~/isaac-sim-5.1.0/exts/isaacsim.core.prims/isaacsim/core/prims/impl/`
- `~/isaac-sim-5.1.0/exts/isaacsim.core.utils/`

---

## Imports

```python
from isaacsim.core.api import SimulationContext, PhysicsContext
from isaacsim.core.prims import Articulation, RigidPrim, XFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
```

---

## Array Shape Conventions

| Data | Shape | Notes |
|------|-------|-------|
| Positions | `(N, 3)` | meters, world frame |
| Orientations (quaternion) | `(N, 4)` | scalar-first `[w, x, y, z]`, normalized |
| Linear / angular velocity | `(N, 3)` | m/s, rad/s |
| Joint positions / velocities / efforts | `(N, num_dof)` | N = num articulations |
| DOF limits | `(N, num_dof, 2)` | `[lower, upper]` |
| Jacobian | `(N, rows, cols)` | see `get_jacobian_shape()` |
| Mass matrix | `(N, num_dof, num_dof)` | symmetric positive definite |
| Contact force (net) | `(N, 3)` | summed over all contacts |
| Contact force matrix | `(N, K, 3)` | K = number of filtered bodies |

**Common parameter semantics:**

| Param | Meaning |
|-------|---------|
| `indices` | Which prims to operate on; `None` = all |
| `joint_indices` | Which joints; mutually exclusive with `joint_names` |
| `clone=True` | Returns a copy; `False` returns a buffer reference (no alloc) |
| `usd=True` | Read/write USD stage; `False` = read/write Fabric (faster after init) |
| `is_global=True` | World frame; `False` = body frame |

---

## Backends

| Backend | Return type | Device | Use when |
|---------|-------------|--------|----------|
| `"numpy"` (default) | `np.ndarray` | CPU | Prototyping, small scenes |
| `"torch"` | `torch.Tensor` | CPU or `cuda:X` | Mid-scale, GPU tensor math |
| `"warp"` | `wp.indexedarray` | GPU | Large-scale, >100 prims, custom kernels |

- Backend is set once at `SimulationContext` construction and propagates to all views.
- `simulation_context.backend_utils` exposes backend-matched helpers (`isaacsim.core.utils.{numpy,torch,warp}`).

---

## SimulationContext

**Location:** `isaacsim.core.api.simulation_context.SimulationContext`
**Pattern:** Singleton — use `SimulationContext.instance()` to retrieve; call `clear_instance()` at shutdown.

### Constructor

```python
SimulationContext(
    physics_dt: float | None = None,        # time per physics step (default 1/60)
    rendering_dt: float | None = None,      # time per render frame
    stage_units_in_meters: float | None = None,
    physics_prim_path: str = "/physicsScene",
    sim_params: dict | None = None,
    set_defaults: bool = True,
    backend: str = "numpy",                 # "numpy" | "torch" | "warp"
    device: str | None = None,              # None = CPU, "cuda:0" = GPU
    stage: Usd.Stage | None = None,
) -> None
```

### Properties (read-only)

| Property | Type | Description |
|----------|------|-------------|
| `app` | `omni.kit.app.IApp` | Kit application handle |
| `current_time_step_index` | `int` | Physics steps since last `play()` |
| `current_time` | `float` | Simulated physical time (seconds) |
| `stage` | `Usd.Stage` | Active USD stage |
| `backend` | `str` | Active backend name |
| `device` | `str` | Active device string |
| `backend_utils` | module | Backend utility module |
| `physics_sim_view` | `omni.physics.tensors.SimulationView` | Available after `initialize_physics()` |

### Time Management

```python
set_simulation_dt(physics_dt: float | None, rendering_dt: float | None) -> None
get_physics_dt() -> float
get_rendering_dt() -> float
set_block_on_render(block: bool) -> None   # True = 1-frame lag; False = min latency
get_block_on_render() -> bool
```

> `rendering_dt >= physics_dt`. Both should be integer multiples of each other.
> Typical robotics: `physics_dt=1/120`, `rendering_dt=1/60` (2 physics steps per frame).

### Simulation Control — Sync (standalone scripts only)

```python
step(render: bool = True, update_fabric: bool = False) -> None
    # render=False: physics only (UI frozen); set update_fabric=True to sync Fabric
render() -> None
reset(soft: bool = False) -> None          # soft=True: reset objects only, no stop/play
initialize_physics() -> None               # builds physics_sim_view
play() -> None
pause() -> None
stop() -> None
clear() -> None                            # wipes stage, leaves PhysicsScene + /World
```

### Simulation Control — Async (Extensions / Kit-managed timing)

```python
async initialize_simulation_context_async() -> None
async reset_async(soft: bool = False) -> None
async play_async() -> None
async pause_async() -> None
async stop_async() -> None
async render_async() -> None
```

### State Queries

```python
is_playing() -> bool
is_paused() -> bool
is_stopped() -> bool
is_simulating() -> bool
```

### Physics Context Access

```python
get_physics_context() -> PhysicsContext
```

### Callbacks

```python
# Physics step — callback_fn(current_time: float) -> None
add_physics_callback(callback_name: str, callback_fn: Callable[[float], None]) -> None
remove_physics_callback(callback_name: str) -> None
physics_callback_exists(callback_name: str) -> bool
clear_physics_callbacks() -> None

# Stage events (prim add/remove)
add_stage_callback(callback_name: str, callback_fn: Callable) -> None
remove_stage_callback(callback_name: str) -> None
stage_callback_exists(callback_name: str) -> bool
clear_stage_callbacks() -> None

# Timeline events (play/pause/stop)
add_timeline_callback(callback_name: str, callback_fn: Callable) -> None
remove_timeline_callback(callback_name: str) -> None
timeline_callback_exists(callback_name: str) -> bool
clear_timeline_callbacks() -> None

# Render events
add_render_callback(callback_name: str, callback_fn: Callable) -> None
remove_render_callback(callback_name: str) -> None
render_callback_exists(callback_name: str) -> bool
clear_render_callbacks() -> None

clear_all_callbacks() -> None
```

---

## PhysicsContext

**Location:** `isaacsim.core.api.physics_context.PhysicsContext`
**Access:** Prefer `sim.get_physics_context()` over constructing directly.

### Constructor

```python
PhysicsContext(
    physics_dt: float | None = None,
    prim_path: str = "/physicsScene",
    sim_params: dict | None = None,
    set_defaults: bool = True,
) -> None
```

### Properties

| Property | Type |
|----------|------|
| `prim_path` | `str` |
| `device` | `str` |
| `use_gpu_pipeline` | `bool` |
| `use_gpu_sim` | `bool` |
| `use_fabric` | `bool` |

### Physics DT & Solver

```python
set_physics_dt(dt: float, substeps: int | None = None) -> None
get_physics_dt() -> float

set_solver_type(solver_type: str) -> None   # "TGS" (stable) | "PGS" (fast)
get_solver_type() -> str

set_gravity(value: float) -> None           # negative = downward
get_gravity() -> float
```

### Collision

```python
enable_ccd(flag: bool) -> None              # Continuous Collision Detection
is_ccd_enabled() -> bool

set_broadphase_type(broadcast_type: str) -> None  # "MBP" (CPU) | "GPU" (requires GPU sim)
get_broadphase_type() -> str

set_bounce_threshold(value: float) -> None
get_bounce_threshold() -> float

set_friction_offset_threshold(value: float) -> None
get_friction_offset_threshold() -> float

set_friction_correlation_distance(value: float) -> None
get_friction_correlation_distance() -> float
```

### Stabilization

```python
enable_stablization(flag: bool) -> None
is_stablization_enabled() -> bool
```

### GPU Dynamics

```python
enable_gpu_dynamics(flag: bool) -> None     # requires device="cuda:X"
is_gpu_dynamics_enabled() -> bool
```

### GPU Buffer Sizing

```python
set_gpu_max_rigid_contact_count(value: int) -> None
set_gpu_max_rigid_patch_count(value: int) -> None
set_gpu_found_lost_pairs_capacity(value: int) -> None
set_gpu_found_lost_aggregate_pairs_capacity(value: int) -> None
set_gpu_total_aggregate_pairs_capacity(value: int) -> None
set_gpu_max_soft_body_contacts(value: int) -> None
set_gpu_max_particle_contacts(value: int) -> None
set_gpu_heap_capacity(value: int) -> None
set_gpu_temp_buffer_capacity(value: int) -> None
set_gpu_max_num_partitions(value: int) -> None
set_gpu_collision_stack_size(value: int) -> None
# Each has a matching get_* counterpart
```

### Fabric & Advanced

```python
enable_fabric(enable: bool) -> None
set_enable_scene_query_support(enable_scene_query_support: bool) -> None
get_enable_scene_query_support() -> bool

enable_residual_reporting(flag: bool) -> None
get_solver_position_residual(report_max: bool = False) -> np.ndarray | torch.Tensor
get_solver_velocity_residual(report_max: bool = False) -> np.ndarray | torch.Tensor

set_solve_articulation_contact_last(solve_articulation_contact_last: bool) -> None
get_solve_articulation_contact_last() -> bool

set_invert_collision_group_filter(invert_collision_group_filter: bool) -> None
get_invert_collision_group_filter() -> bool

set_physx_update_transformations_settings(
    update_to_usd: bool,
    update_velocities_to_usd: bool,
    output_velocities_local_space: bool,
) -> None
get_physx_update_transformations_settings() -> tuple[bool, bool, bool]
```

---

## Articulation

**Location:** `isaacsim.core.prims.impl.articulation.Articulation`
**Inherits:** `XFormPrim`

### Constructor

```python
Articulation(
    prim_paths_expr: str | list[str],
    name: str = "articulation_prim_view",
    positions: array_like | None = None,
    translations: array_like | None = None,
    orientations: array_like | None = None,
    scales: array_like | None = None,
    visibilities: array_like | None = None,
    reset_xform_properties: bool = True,
    enable_residual_reports: bool = False,
) -> None
```

### Properties

| Property | Type |
|----------|------|
| `num_dof` | `int` |
| `num_bodies` | `int` |
| `num_shapes` | `int` |
| `num_joints` | `int` |
| `num_fixed_tendons` | `int` |
| `body_names` | `list[str]` |
| `dof_names` | `list[str]` |
| `joint_names` | `list[str]` |

### Initialization & State

```python
initialize(physics_sim_view: omni.physics.tensors.SimulationView) -> None
    # REQUIRED before any physics query

is_physics_handle_valid() -> bool

get_joints_default_state() -> JointsState           # positions, velocities, efforts
set_joints_default_state(
    positions: array_like | None = None,
    velocities: array_like | None = None,
    efforts: array_like | None = None,
) -> None

get_joints_state() -> JointsState                   # current state
```

### Joint Positions

```python
get_joint_positions(
    indices: array_like | None = None,
    joint_indices: array_like | None = None,
    joint_names: list[str] | None = None,
    clone: bool = True,
) -> np.ndarray | torch.Tensor | wp.indexedarray    # shape (N, num_dof)

set_joint_positions(positions, indices=None, joint_indices=None, joint_names=None) -> None
    # Teleports joints — bypasses dynamics

set_joint_position_targets(positions, indices=None, joint_indices=None, joint_names=None) -> None
    # PD controller targets — respects gains
```

### Joint Velocities

```python
get_joint_velocities(indices=None, joint_indices=None, joint_names=None, clone=True)
    -> np.ndarray | torch.Tensor | wp.indexedarray  # shape (N, num_dof)

set_joint_velocities(velocities, ...)               # teleports velocities
set_joint_velocity_targets(velocities, ...)         # velocity controller targets
```

### Joint Efforts

```python
get_applied_joint_efforts(indices=None, joint_indices=None, joint_names=None, clone=True)
get_measured_joint_efforts(indices=None, joint_indices=None, joint_names=None, clone=True)
get_measured_joint_forces(indices=None, joint_indices=None, joint_names=None, clone=True)

set_joint_efforts(efforts, indices=None, joint_indices=None, joint_names=None) -> None
    # Direct torque / force command
```

### High-level Action API

```python
apply_action(control_actions: ArticulationActions, indices=None) -> None
    # Unified: position targets + velocity targets + efforts in one call

get_applied_actions(clone: bool = True) -> ArticulationActions
```

### Control Mode

```python
switch_control_mode(
    mode: str,          # "position" | "velocity" | "effort"
    indices=None,
    joint_indices=None,
    joint_names=None,
) -> None

switch_dof_control_mode(mode: str, dof_index: int, indices=None) -> None
```

### Gains & Joint Properties

```python
set_gains(kps, kds, indices=None, joint_indices=None, joint_names=None, save_to_usd=False) -> None
get_gains(indices=None, joint_indices=None, joint_names=None, clone=True)
    -> tuple[array, array]                          # (kps, kds)

set_friction_coefficients(values, ...) -> None
get_friction_coefficients(...) -> array

set_armatures(values, ...) -> None
get_armatures(...) -> array

set_max_efforts(values, ...) -> None
get_max_efforts(...) -> array
```

### DOF Metadata

```python
get_dof_limits() -> array                  # shape (N, num_dof, 2)  [lower, upper]
get_dof_types(dof_names=None) -> list[str] # "Rotation" | "Translation"
get_drive_types() -> array                 # shape (N, num_dof)
get_dof_index(dof_name: str) -> int
```

### Body Properties

```python
get_body_index(body_name: str) -> int
get_link_index(link_name: str) -> int

get_body_masses(indices=None, body_indices=None, clone=True) -> array
set_body_masses(values, indices=None, body_indices=None) -> None

get_body_inertias(indices=None, body_indices=None, clone=True) -> array
set_body_inertias(values, ...) -> None

get_body_coms(indices=None, body_indices=None, clone=True) -> array
set_body_coms(positions, orientations, indices=None, body_indices=None) -> None
```

### Root Pose & Velocity

```python
get_world_poses(indices=None, clone=True, usd=True) -> tuple[array, array]  # (pos (N,3), quat (N,4))
set_world_poses(positions=None, orientations=None, indices=None, usd=True) -> None

get_local_poses(indices=None) -> tuple[array, array]
set_local_poses(translations=None, orientations=None, indices=None) -> None

get_linear_velocities(indices=None, clone=True) -> array   # root body, shape (N, 3)
set_linear_velocities(velocities, indices=None) -> None

get_angular_velocities(indices=None, clone=True) -> array  # root body, shape (N, 3)
set_angular_velocities(velocities, indices=None) -> None

get_velocities(indices=None, clone=True) -> tuple[array, array]   # (linear, angular)
set_velocities(velocities: tuple[array, array], indices=None) -> None
```

### Dynamics (Jacobian, Mass Matrix, Gravity, Coriolis)

```python
get_jacobian_shape() -> tuple[int, int]
get_jacobians(indices=None, clone=True) -> array     # shape (N, rows, cols)

get_mass_matrix_shape() -> tuple[int, int]
get_mass_matrices(indices=None, clone=True) -> array # shape (N, num_dof, num_dof)

get_generalized_gravity_forces(indices=None, joint_indices=None, joint_names=None, clone=True) -> array
get_coriolis_and_centrifugal_forces(indices=None, joint_indices=None, joint_names=None, clone=True) -> array
```

### Solver & Residuals

```python
set_solver_position_iteration_counts(counts, indices=None) -> None
get_solver_position_iteration_counts(indices=None) -> array

set_solver_velocity_iteration_counts(counts, indices=None) -> None
get_solver_velocity_iteration_counts(indices=None) -> array

set_sleep_thresholds(thresholds, indices=None) -> None
get_sleep_thresholds(indices=None) -> array

get_position_residuals(indices=None, report_max=False) -> array  # requires enable_residual_reports=True
get_velocity_residuals(indices=None, report_max=False) -> array
```

### Motion Control

```python
pause_motion() -> None
resume_motion() -> None
```

---

## RigidPrim

**Location:** `isaacsim.core.prims.impl.rigid_prim.RigidPrim`
**Inherits:** `XFormPrim`

### Constructor

```python
RigidPrim(
    prim_paths_expr: str | list[str],
    name: str = "rigid_prim_view",
    positions: array_like | None = None,
    translations: array_like | None = None,
    orientations: array_like | None = None,
    scales: array_like | None = None,
    visibilities: array_like | None = None,
    reset_xform_properties: bool = True,
    masses: array_like | None = None,
    densities: array_like | None = None,
    linear_velocities: array_like | None = None,
    angular_velocities: array_like | None = None,
    track_contact_forces: bool = False,         # required for contact queries
    prepare_contact_sensors: bool = True,
    disable_stablization: bool = True,
    contact_filter_prim_paths_expr: list[str] | None = [],  # required for contact matrix
    max_contact_count: int = 0,
) -> None
```

### Initialization & State

```python
initialize(physics_sim_view: omni.physics.tensors.SimulationView) -> None
is_physics_handle_valid() -> bool

get_default_state() -> DynamicsViewState            # pos, orient, linear_vel, angular_vel
set_default_state(
    positions=None, orientations=None,
    linear_velocities=None, angular_velocities=None,
    indices=None,
) -> None

get_current_dynamic_state() -> DynamicsViewState
```

### Poses

```python
get_world_poses(indices=None, clone=True, usd=True) -> tuple[array, array]  # (pos (N,3), quat (N,4))
set_world_poses(positions=None, orientations=None, indices=None, usd=True) -> None
    # Teleports — does not respect dynamics

get_local_poses(indices=None) -> tuple[array, array]
set_local_poses(translations=None, orientations=None, indices=None) -> None
```

### Velocities

```python
get_linear_velocities(indices=None, clone=True) -> array   # shape (N, 3)
set_linear_velocities(velocities, indices=None) -> None

get_angular_velocities(indices=None, clone=True) -> array  # shape (N, 3)
set_angular_velocities(velocities, indices=None) -> None

get_velocities(indices=None, clone=True) -> tuple[array, array]
set_velocities(velocities: tuple[array, array], indices=None) -> None
```

### Mass & Inertia

```python
get_masses(indices=None, clone=True) -> array
set_masses(masses, indices=None) -> None

get_densities(indices=None) -> array
set_densities(densities, indices=None) -> None

get_inertias(indices=None, clone=True) -> array
set_inertias(values, indices=None) -> None

get_inv_masses(indices=None, clone=True) -> array
get_inv_inertias(indices=None, clone=True) -> array

get_coms(indices=None, clone=True) -> array
set_coms(positions, orientations, indices=None) -> None
```

### Forces & Torques

```python
apply_forces(
    forces: array,                  # shape (N, 3)
    indices=None,
    is_global: bool = True,         # True = world frame; False = body frame
) -> None

apply_forces_and_torques_at_pos(
    forces: array,                  # shape (N, 3)
    torques: array,                 # shape (N, 3)
    positions: array,               # shape (N, 3) — application point
    indices=None,
    is_global: bool = True,
) -> None
```

### Gravity & Physics Enable

```python
enable_gravities(indices=None) -> None
disable_gravities(indices=None) -> None

enable_rigid_body_physics(indices=None) -> None
disable_rigid_body_physics(indices=None) -> None

set_sleep_thresholds(thresholds, indices=None) -> None
get_sleep_thresholds(indices=None) -> array
```

### Contact Queries

```python
get_net_contact_forces(
    indices=None, clone=True, dt: float = 1.0,
) -> array                          # shape (N, 3); requires track_contact_forces=True

get_contact_force_data(
    indices=None, clone=True, dt: float = 1.0,
) -> dict                           # keys: positions, normals, impulses, distances
                                    # requires track_contact_forces=True

get_contact_force_matrix(
    indices=None, clone=True, dt: float = 1.0,
) -> array                          # shape (N, K, 3); requires contact_filter_prim_paths_expr

get_friction_data(
    indices=None, clone=True, dt: float = 1.0,
) -> dict
```

---

## XFormPrim

**Location:** `isaacsim.core.prims.impl.xform_prim.XFormPrim`
**Role:** Base class for all prim views. Use directly for non-physics transform hierarchy.

### Constructor

```python
XFormPrim(
    prim_paths_expr: str | list[str],
    name: str = "xform_prim_view",
    positions: array_like | None = None,
    translations: array_like | None = None,
    orientations: array_like | None = None,
    scales: array_like | None = None,
    visibilities: array_like | None = None,
    reset_xform_properties: bool = True,
    usd: bool = True,
) -> None
```

### Poses

```python
get_world_poses(indices=None, usd: bool = True) -> tuple[array, array]   # (pos (N,3), quat (N,4))
    # usd=False reads from Fabric — faster after initialization

set_world_poses(positions=None, orientations=None, indices=None, usd=True) -> None

get_local_poses(indices=None) -> tuple[array, array]
set_local_poses(translations=None, orientations=None, indices=None) -> None
```

### Scales

```python
get_world_scales(indices=None) -> array
get_local_scales(indices=None) -> array
set_local_scales(scales, indices=None) -> None
```

### Visibility

```python
get_visibilities(indices=None) -> array      # boolean
set_visibilities(visibilities, indices=None) -> None
```

### Visual Materials

```python
apply_visual_materials(
    visual_materials: VisualMaterial | list[VisualMaterial],
    weaker_than_descendants: bool = False,
    indices=None,
) -> None

get_applied_visual_materials(indices=None) -> VisualMaterial | list[VisualMaterial]
is_visual_material_applied(indices=None) -> array  # boolean
```

### Default State

```python
get_default_state() -> XFormPrimViewState    # positions, orientations
set_default_state(positions=None, orientations=None, indices=None) -> None
```

---

## Common Patterns

### Standalone Script — Sync Control Loop

```python
from isaacsim.core.api import SimulationContext
from isaacsim.core.prims import Articulation, RigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage

sim = SimulationContext(backend="torch", device="cuda:0")
physics = sim.get_physics_context()
physics.set_physics_dt(1 / 120)
physics.enable_gpu_dynamics(True)

add_reference_to_stage("/World/Robot", "path/to/robot.usd")
robot = Articulation("/World/Robot")
box = RigidPrim("/World/Box", track_contact_forces=True)

sim.reset()
robot.initialize(sim.physics_sim_view)
box.initialize(sim.physics_sim_view)

for _ in range(1000):
    q = robot.get_joint_positions()            # (1, num_dof)
    F_contact = box.get_net_contact_forces()   # (1, 3)

    tau = my_controller(q)
    robot.set_joint_efforts(tau)

    sim.step(render=False)                     # physics only; faster

SimulationContext.clear_instance()
```

### Extension (Kit-managed timing) — Async Workflow

```python
async def on_startup():
    sim = SimulationContext(backend="torch", device="cuda:0")
    await sim.initialize_simulation_context_async()
    await sim.reset_async()

    robot = Articulation("/World/Robot")
    robot.initialize(sim.physics_sim_view)

    for _ in range(1000):
        q = robot.get_joint_positions()
        robot.set_joint_position_targets(my_targets(q))
        await sim.render_async()               # yields control to Kit each frame

    SimulationContext.clear_instance()
```

> Never mix sync and async calls. Sync methods (`step`, `reset`, `render`) are forbidden in Extensions; async methods are forbidden in standalone scripts.

### Physics Callback Pattern

```python
sim = SimulationContext()

def on_physics_step(current_time: float) -> None:
    q = robot.get_joint_positions()
    robot.set_joint_efforts(controller(q))
    # Keep lightweight — heavy compute here can cause deadlocks

sim.add_physics_callback("robot_control", on_physics_step)
sim.reset()
sim.play()
# Kit / app loop drives stepping from here
```

### PD Position Control

```python
robot.switch_control_mode("position")
robot.set_gains(
    kps=torch.tensor([[500.0] * robot.num_dof]),
    kds=torch.tensor([[50.0]  * robot.num_dof]),
)
robot.set_joint_position_targets(target_positions)  # respects gains each step
```

### Batched Multi-Robot Control (N robots via glob expression)

```python
robots = Articulation("/World/Robot_*")    # matches Robot_0 … Robot_N
sim.reset()
robots.initialize(sim.physics_sim_view)

# All operations are batched over N robots automatically
q = robots.get_joint_positions()           # shape (N, num_dof)
targets = compute_targets(q)               # shape (N, num_dof)
robots.set_joint_position_targets(targets)
```

### Contact Detection

```python
obj = RigidPrim(
    "/World/Object",
    track_contact_forces=True,
    contact_filter_prim_paths_expr=["/World/Floor"],  # optional: per-body matrix
)
obj.initialize(sim.physics_sim_view)

# Per step:
F_net = obj.get_net_contact_forces()       # (N, 3) — always available
F_mat = obj.get_contact_force_matrix()     # (N, K, 3) — needs filter expr
detail = obj.get_contact_force_data()      # dict with normals, positions, impulses
```

### Solver Residual Monitoring

```python
physics.enable_residual_reporting(True)
# or per-articulation:
robot = Articulation("/World/Robot", enable_residual_reports=True)

# Per step:
pos_res = physics.get_solver_position_residual()
vel_res = physics.get_solver_velocity_residual()
pos_res_art = robot.get_position_residuals()
```

---

## Performance Tips

| Tip | Detail |
|-----|--------|
| Use `warp` backend | Fastest for large batches; `wp.indexedarray` stays on GPU |
| Enable GPU dynamics | `physics.enable_gpu_dynamics(True)` — required for >100 parallel prims |
| Use `clone=False` | Avoids allocation; safe when you consume data immediately |
| Use `usd=False` | Reads from Fabric instead of USD — significantly faster in hot loops |
| `step(render=False)` | Skips renderer; 2-10x faster for headless training |
| `update_fabric=True` | Required with `render=False` if downstream code reads Fabric |
| Avoid Python loops | All prim APIs are batch-first; never iterate over individual prims |
| Pre-allocate tensors | Reuse tensor buffers rather than allocating new ones each step |
| Use `indices` param | Query/set only the prims you need to reduce GPU data transfer |
| Enable Fabric | `physics.enable_fabric(True)` for GPU-pipeline data synchronization |
| TGS solver | Better stability for articulations vs PGS; set via `set_solver_type("TGS")` |
| Physics DT `1/120` | Standard for robotics; `rendering_dt=1/60` gives 2 physics steps/frame |
| GPU buffer sizing | Increase `set_gpu_found_lost_pairs_capacity` if sim crashes with many contacts |

---

## Initialization Sequence

```
1. SimulationContext(backend=..., device=...)
2. physics = sim.get_physics_context()
3. physics.set_physics_dt(1/120)
4. physics.enable_gpu_dynamics(True)          # if GPU
5. add_reference_to_stage(...)                # load USD assets
6. robot = Articulation(prim_path_expr)
7. obj   = RigidPrim(prim_path_expr, track_contact_forces=True)
8. sim.reset()                                # triggers physics initialization
9. robot.initialize(sim.physics_sim_view)     # REQUIRED
10. obj.initialize(sim.physics_sim_view)      # REQUIRED
11. robot.set_joints_default_state(...)       # optional: set initial state
12. sim.reset(soft=True)                      # apply default states
# -> enter control loop
```

> Steps 9 and 10 are mandatory. Skipping them will raise errors or return zeros silently on physics queries.

---

## Source File Locations

| Class | File |
|-------|------|
| `SimulationContext` | `~/isaac-sim-5.1.0/exts/isaacsim.core.api/isaacsim/core/api/simulation_context/simulation_context.py` |
| `PhysicsContext` | `~/isaac-sim-5.1.0/exts/isaacsim.core.api/isaacsim/core/api/physics_context/physics_context.py` |
| `Articulation` | `~/isaac-sim-5.1.0/exts/isaacsim.core.prims/isaacsim/core/prims/impl/articulation.py` |
| `RigidPrim` | `~/isaac-sim-5.1.0/exts/isaacsim.core.prims/isaacsim/core/prims/impl/rigid_prim.py` |
| `XFormPrim` | `~/isaac-sim-5.1.0/exts/isaacsim.core.prims/isaacsim/core/prims/impl/xform_prim.py` |
| Backend utils | `~/isaac-sim-5.1.0/exts/isaacsim.core.utils/isaacsim/core/utils/{numpy,torch,warp}.py` |
| Fabric utils | `~/isaac-sim-5.1.0/exts/isaacsim.core.utils/isaacsim/core/utils/fabric.py` |
| Stage utils | `~/isaac-sim-5.1.0/exts/isaacsim.core.utils/isaacsim/core/utils/stage.py` |
