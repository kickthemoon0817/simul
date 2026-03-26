# Isaac Sim 5.1.0 — Physics API Reference

Practical reference for physics interfaces available in Isaac Sim 5.1.0. All APIs assume an active simulation stage.

---

## Table of Contents

1. [PhysX Interface](#physx-interface-omniphysx)
2. [Physics Tensor API](#physics-tensor-api-omniphysicstensors)
3. [Collision / Scene Query](#collision--scene-query-physxscenequery)
4. [Character Controller](#character-controller-omniphysxcct)
5. [Vehicle Physics](#vehicle-physics-omniphysxvehicle)
6. [Simulation Events](#simulation-events)

---

## PhysX Interface (`omni.physx`)

Entry points for the three core PhysX interfaces.

```python
import omni.physx
physx          = omni.physx.get_physx_interface()
scene_query    = omni.physx.get_physx_scene_query_interface()
simulation     = omni.physx.get_physx_simulation_interface()
```

### `get_physx_interface()` — rigid body control

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_rigidbody_transformation` | `(prim_path: str) -> dict` | Returns `position` (Gf.Vec3d), `rotation` (Gf.Quatd) for the body |
| `apply_force_at_pos` | `(prim_path: str, force: carb.Float3, pos: carb.Float3, mode: str)` | Apply world-space force at world-space position. `mode`: `"Force"`, `"Impulse"`, `"VelocityChange"`, `"Acceleration"` |
| `apply_torque` | `(prim_path: str, torque: carb.Float3)` | Apply world-space torque to a rigid body |

```python
import carb

physx.apply_force_at_pos(
    "/World/Robot/base",
    carb.Float3(0.0, 0.0, 100.0),
    carb.Float3(0.0, 0.0, 0.5),
    "Force",
)
physx.apply_torque("/World/Robot/base", carb.Float3(0.0, 10.0, 0.0))
```

---

## Physics Tensor API (`omni.physics.tensors`)

GPU-batched access to articulation and rigid body state. Operates on entire **views** rather than individual prims — designed for RL and batch simulation workloads.

### Creating a Simulation View

```python
from omni.physics.tensors import create_simulation_view

view = create_simulation_view("torch")   # "torch" | "numpy" | "warp"
view.set_subspace_roots("/World")        # required before creating sub-views
```

Backend selection:

| Backend | Returns | Use case |
|---------|---------|---------|
| `"torch"` | `torch.Tensor` (GPU) | RL training, CUDA pipelines |
| `"numpy"` | `np.ndarray` (CPU) | Scripting, debugging |
| `"warp"` | `wp.array` (GPU) | Warp kernels, differentiable sim |

### ArticulationView

```python
art_view = view.create_articulation_view("/World/Robot_*/base")
```

**DOF state** — positions and velocities for all joints across all instances:

| Method | Returns shape | Description |
|--------|--------------|-------------|
| `get_dof_positions()` | `[N, num_dof]` | Current joint positions (rad / m) |
| `set_dof_positions(positions)` | — | Teleport joints; bypasses dynamics |
| `get_dof_velocities()` | `[N, num_dof]` | Current joint velocities |
| `set_dof_velocities(velocities)` | — | Set joint velocities directly |
| `get_dof_position_targets()` | `[N, num_dof]` | PD controller position targets |
| `set_dof_position_targets(targets)` | — | Write PD position targets |

**Dynamics quantities** — all returned per-instance:

| Method | Returns shape | Description |
|--------|--------------|-------------|
| `get_jacobians()` | `[N, num_bodies*6, num_dof+6]` | Full body Jacobian |
| `get_mass_matrices()` | `[N, num_dof, num_dof]` | Generalized mass (inertia) matrix |
| `get_coriolis_and_centrifugal_forces()` | `[N, num_dof]` | Coriolis + centrifugal vector |
| `get_generalized_gravity_forces()` | `[N, num_dof]` | Gravity compensation torques |

```python
# Zero-gravity compensation torque example
positions = art_view.get_dof_positions()
gravity   = art_view.get_generalized_gravity_forces()
art_view.set_dof_position_targets(positions - gravity * dt)
```

### RigidBodyView

```python
rb_view = view.create_rigid_body_view("/World/Boxes/box_*")
```

| Method | Returns shape | Description |
|--------|--------------|-------------|
| `get_transforms()` | `[N, 7]` | Position (xyz) + quaternion (wxyz) |
| `set_transforms(transforms)` | — | Teleport bodies |
| `get_velocities()` | `[N, 6]` | Linear (xyz) + angular (xyz) velocity |
| `set_velocities(velocities)` | — | Set linear + angular velocity |
| `apply_forces(forces, positions, indices)` | — | Apply batched forces; `positions` in world space, `indices` selects instances |

```python
import torch

# Apply upward impulse to all boxes
N = rb_view.count
forces = torch.zeros((N, 3), device="cuda")
forces[:, 2] = 500.0
rb_view.apply_forces(forces, torch.zeros((N, 3), device="cuda"), indices=None)
```

### RigidContactView

```python
contact_view = view.create_rigid_contact_view(
    "/World/Robot_*/.*",        # sensors (bodies that report contacts)
    filter_patterns=["/World/Ground"],  # optional: only contacts with these paths
)
```

| Method | Returns shape | Description |
|--------|--------------|-------------|
| `get_net_contact_forces()` | `[N, 3]` | Net force (xyz) summed over all contacts per sensor |
| `get_contact_force_matrix()` | `[N_sensors, N_filters, 3]` | Per-(sensor, filter) contact force |
| `get_contact_data()` | structured | Raw contact points, normals, impulses |

```python
net_forces = contact_view.get_net_contact_forces()   # [N, 3]
in_contact = net_forces.norm(dim=-1) > 0.1           # [N] bool mask
```

---

## Collision / Scene Query (PhysXSceneQuery)

Low-level spatial queries against the PhysX scene. Returns results immediately (synchronous).

```python
scene_query = omni.physx.get_physx_scene_query_interface()
```

### Overlap Queries

Find all shapes that overlap a given volume. `_any` variants return on first hit (faster).

| Method | Key args | Returns |
|--------|----------|---------|
| `overlap_sphere(radius, origin, reportFn, ...)` | `radius: float`, `origin: carb.Double3` | calls `reportFn` per hit |
| `overlap_sphere_any(radius, origin, ...)` | same | `bool` |
| `overlap_box(halfExtent, pos, rot, reportFn, ...)` | `halfExtent: carb.Float3`, `rot: carb.Float4` (quat) | calls `reportFn` per hit |
| `overlap_box_any(halfExtent, pos, rot, ...)` | same | `bool` |
| `overlap_shape(prim_path, reportFn, ...)` | `prim_path: str` (existing collision shape) | calls `reportFn` per hit |
| `overlap_mesh(mesh_path, pos, rot, reportFn, ...)` | custom mesh asset | calls `reportFn` per hit |

```python
hits = []
def on_hit(hit):
    hits.append(hit.rigid_body)
    return True  # return False to stop early

scene_query.overlap_sphere(0.5, carb.Double3(0.0, 0.0, 1.0), on_hit)
```

### Raycast Queries

| Method | Key args | Returns |
|--------|----------|---------|
| `raycast_closest(origin, dir, distance)` | all `carb.Float3` | `dict` with `position`, `normal`, `distance`, `rigid_body`, `hit` |
| `raycast_any(origin, dir, distance)` | same | `bool` |
| `raycast_all(origin, dir, distance, reportFn)` | `reportFn` called per hit | — |

```python
hit = scene_query.raycast_closest(
    carb.Float3(0.0, 0.0, 5.0),
    carb.Float3(0.0, 0.0, -1.0),
    10.0,
)
if hit["hit"]:
    print(hit["rigid_body"], hit["distance"])
```

### Sweep Queries

Sweep a shape along a direction. Variants follow `_closest` / `_any` / `_all` pattern.

| Method | Notes |
|--------|-------|
| `sweep_sphere_closest(radius, origin, dir, distance, ...)` | sphere cast |
| `sweep_sphere_any(...)` | bool fast-path |
| `sweep_sphere_all(...)` | all hits via callback |
| `sweep_box_closest(halfExtent, pos, rot, dir, distance, ...)` | box cast |
| `sweep_box_any(...)` | |
| `sweep_box_all(...)` | |
| `sweep_shape_closest(prim_path, pos, rot, dir, distance, ...)` | existing shape cast |
| `sweep_mesh_closest(mesh_path, pos, rot, dir, distance, ...)` | mesh cast |

---

## Character Controller (`omni.physxcct`)

Kinematic capsule controller for bipeds and non-physics-driven agents.

```python
import omni.physxcct
cct = omni.physxcct.get_physx_cct_interface()
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `activate_cct` | `(prim_path: str)` | Activate the CCT component on the given prim |
| `set_move` | `(prim_path: str, displacement: carb.Float3, min_dist: float, elapsed_time: float)` | Move by `displacement` (world-space). Resolves collisions internally |
| `set_position` | `(prim_path: str, position: carb.Float3)` | Teleport the controller |
| `enable_gravity` | `(prim_path: str, enabled: bool)` | Toggle built-in gravity application |

```python
cct.activate_cct("/World/Character")
cct.enable_gravity("/World/Character", True)

# In physics step callback:
cct.set_move("/World/Character", carb.Float3(0.0, 0.05, 0.0), 0.001, dt)
```

---

## Vehicle Physics (`omni.physxvehicle`)

Read-only telemetry for PhysX vehicle drive components.

```python
import omni.physxvehicle
vehicle_iface = omni.physxvehicle.get_physx_vehicle_interface()
```

| Method | Signature | Returns |
|--------|-----------|---------|
| `get_vehicle_drive_state` | `(prim_path: str)` | `dict` — `engine_rotation_speed`, `gear`, `clutch`, `throttle`, `brake`, `steer`, `is_moving` |
| `get_wheel_state` | `(prim_path: str, wheel_index: int)` | `dict` — `rotation_speed`, `steer_angle`, `suspension_jounce`, `ground_plane`, `ground_actor`, `tire_friction`, `tire_longitudinal_slip`, `tire_lateral_slip` |

```python
drive = vehicle_iface.get_vehicle_drive_state("/World/Car")
print(drive["engine_rotation_speed"], drive["gear"])

wheel = vehicle_iface.get_wheel_state("/World/Car", 0)
print(wheel["tire_longitudinal_slip"])
```

---

## Simulation Events

Subscribe to physics callbacks. All subscriptions return a token; store it to keep the subscription alive (garbage-collected when token goes out of scope).

### Contact Reports

```python
physx_sim = omni.physx.get_physx_simulation_interface()

def on_contact(header, data):
    # header.actor0 / header.actor1 — prim paths of colliding bodies
    # data — list of contact points (position, normal, impulse, separation)
    pass

token = physx_sim.subscribe_contact_report_events(on_contact)
```

### Trigger Reports

```python
def on_trigger(trigger_data):
    # trigger_data.collider      — path of the trigger collider
    # trigger_data.other_collider — path of the entering/leaving body
    # trigger_data.type          — "enter" | "leave"
    pass

token = physx_sim.subscribe_physics_trigger_report_events(on_trigger)
```

### Per-Step Callback

Runs each physics sub-step (not per-render frame). Use for control loops that must be tightly coupled to the physics clock.

```python
def on_physics_step(dt: float):
    pass

token = physx_sim.subscribe_physics_on_step_events(on_physics_step)
```

> **Lifetime note**: assign the returned subscription token to a long-lived variable (e.g., an instance attribute). When the token is garbage-collected, the callback is automatically unsubscribed.

---

## Quick-Reference: Return Type Shapes

| View method | Shape | Dtype |
|-------------|-------|-------|
| `ArticulationView.get_dof_positions` | `[N, D]` | float32 |
| `ArticulationView.get_dof_velocities` | `[N, D]` | float32 |
| `ArticulationView.get_jacobians` | `[N, B*6, D+6]` | float32 |
| `ArticulationView.get_mass_matrices` | `[N, D, D]` | float32 |
| `ArticulationView.get_generalized_gravity_forces` | `[N, D]` | float32 |
| `RigidBodyView.get_transforms` | `[N, 7]` | float32 |
| `RigidBodyView.get_velocities` | `[N, 6]` | float32 |
| `RigidContactView.get_net_contact_forces` | `[N, 3]` | float32 |
| `RigidContactView.get_contact_force_matrix` | `[Ns, Nf, 3]` | float32 |

N = instances, D = DOFs, B = bodies, Ns = sensors, Nf = filters.
