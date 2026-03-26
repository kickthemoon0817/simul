# Robots API Reference — Isaac Sim 5.1.0

## ArticulationAction

Dataclass representing a command sent to an articulation.

```python
from isaacsim.robot.manipulators.grippers import ArticulationAction

action = ArticulationAction(
    joint_positions=np.array([...]),    # target joint positions (rad or m)
    joint_velocities=np.array([...]),   # target joint velocities (rad/s or m/s)
    joint_efforts=np.array([...]),      # target joint torques/forces (N·m or N)
    joint_indices=np.array([...]),      # optional: subset of joint indices
)
```

| Field | Type | Description |
|---|---|---|
| `joint_positions` | `np.ndarray \| None` | Target positions for each joint |
| `joint_velocities` | `np.ndarray \| None` | Target velocities for each joint |
| `joint_efforts` | `np.ndarray \| None` | Target efforts (torque/force) for each joint |
| `joint_indices` | `np.ndarray \| None` | Indices of joints this action applies to; `None` = all joints |

---

## SingleManipulator

Extends `SingleArticulation`. High-level wrapper for a robot arm with optional gripper.

```python
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper

gripper = ParallelGripper(
    end_effector_prim_path="/World/robot/ee_link",
    joint_prim_names=["finger_joint_1", "finger_joint_2"],
    joint_opened_positions=np.array([0.04, 0.04]),
    joint_closed_positions=np.array([0.0, 0.0]),
    action_deltas=np.array([0.04, 0.04]),
)

robot = SingleManipulator(
    prim_path="/World/robot",
    name="my_robot",
    end_effector_prim_name="ee_link",
    gripper=gripper,
)
robot.initialize()
```

### Properties

| Property | Type | Description |
|---|---|---|
| `end_effector` | `RigidPrim` | Prim at the end-effector link |
| `gripper` | `GripperBase` | Attached gripper instance |
| `end_effector_prim_path` | `str` | USD path to the end-effector prim |

---

## ParallelGripper

Parallel-jaw gripper controlled by mirrored joint commands.

```python
from isaacsim.robot.manipulators.grippers import ParallelGripper

gripper = ParallelGripper(
    end_effector_prim_path="/World/robot/ee_link",
    joint_prim_names=["left_finger_joint", "right_finger_joint"],
    joint_opened_positions=np.array([0.04, 0.04]),
    joint_closed_positions=np.array([0.0, 0.0]),
    action_deltas=np.array([0.04, 0.04]),
)
```

### Constructor Parameters

| Parameter | Type | Description |
|---|---|---|
| `end_effector_prim_path` | `str` | USD path to end-effector prim |
| `joint_prim_names` | `list[str]` | Names of the two finger joints |
| `joint_opened_positions` | `np.ndarray` | Joint positions when fully open |
| `joint_closed_positions` | `np.ndarray` | Joint positions when fully closed |
| `action_deltas` | `np.ndarray` | Per-step position delta when driving open/close |

### Methods

```python
gripper.open()                          # drive to joint_opened_positions
gripper.close()                         # drive to joint_closed_positions
action = gripper.forward(action_index)  # 0 = open, 1 = close → ArticulationAction
```

| Method | Returns | Description |
|---|---|---|
| `open()` | `None` | Apply open position command |
| `close()` | `None` | Apply close position command |
| `forward(action_index: int)` | `ArticulationAction` | Build an action for the given index |

### Properties

| Property | Type | Description |
|---|---|---|
| `joint_opened_positions` | `np.ndarray` | Read/write open positions |
| `joint_closed_positions` | `np.ndarray` | Read/write closed positions |

---

## SurfaceGripper

Suction-cup style gripper using contact constraints.

```python
from isaacsim.robot.manipulators.grippers import SurfaceGripper

gripper = SurfaceGripper(
    end_effector_prim_path="/World/robot/suction_cup",
    translate=0.0,
    direction="x",
)
gripper.initialize(physics_sim_view=sim_view, articulation_num_dofs=7)
```

### Constructor Parameters

| Parameter | Type | Description |
|---|---|---|
| `end_effector_prim_path` | `str` | USD path to suction prim |
| `translate` | `float` | Offset along the contact normal |
| `direction` | `str` | Contact normal axis: `"x"`, `"y"`, or `"z"` |

### Methods

```python
gripper.open()             # release suction
gripper.close()            # engage suction
open_state  = gripper.is_open()     # bool
close_state = gripper.is_closed()   # bool
```

| Method | Returns | Description |
|---|---|---|
| `open()` | `None` | Disengage the suction constraint |
| `close()` | `None` | Engage the suction constraint |
| `is_open()` | `bool` | `True` when suction is released |
| `is_closed()` | `bool` | `True` when suction is engaged |

---

## WheeledRobot

Extends `SingleArticulation` for mobile wheeled platforms.

```python
from isaacsim.robot.wheeled_robots import WheeledRobot

robot = WheeledRobot(
    prim_path="/World/jetbot",
    name="jetbot",
    wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
    create_robot=True,
    usd_path="/Isaac/Robots/Jetbot/jetbot.usd",
)
robot.initialize()
```

### Wheel Methods

```python
positions  = robot.get_wheel_positions()                   # np.ndarray (rad)
velocities = robot.get_wheel_velocities()                  # np.ndarray (rad/s)
robot.set_wheel_positions(np.array([1.0, 1.0]))            # rad
robot.set_wheel_velocities(np.array([5.0, 5.0]))           # rad/s
robot.apply_wheel_actions(ArticulationAction(...))         # direct action
```

| Method | Signature | Description |
|---|---|---|
| `get_wheel_positions` | `() → np.ndarray` | Current wheel joint angles (rad) |
| `get_wheel_velocities` | `() → np.ndarray` | Current wheel angular velocities (rad/s) |
| `set_wheel_positions` | `(positions: np.ndarray) → None` | Command wheel positions |
| `set_wheel_velocities` | `(velocities: np.ndarray) → None` | Command wheel velocities |
| `apply_wheel_actions` | `(action: ArticulationAction) → None` | Apply a full wheel action |

---

## DifferentialController

Converts `[linear_speed, angular_speed]` to left/right wheel velocities.

```python
from isaacsim.robot.wheeled_robots.controllers import DifferentialController

ctrl = DifferentialController(name="diff_ctrl", wheel_radius=0.035, wheel_base=0.1125)
action = ctrl.forward(command=np.array([0.3, 0.5]))  # ArticulationAction
robot.apply_wheel_actions(action)
```

### Constructor Parameters

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Controller name |
| `wheel_radius` | `float` | Wheel radius in metres |
| `wheel_base` | `float` | Distance between wheel centres in metres |

### Methods

| Method | Signature | Returns | Description |
|---|---|---|---|
| `forward` | `(command: np.ndarray)` | `ArticulationAction` | `command[0]` = linear (m/s), `command[1]` = angular (rad/s) |
| `reset` | `() → None` | `None` | Clear internal state |

---

## HolonomicController

Converts `[forward, lateral, yaw]` commands for omnidirectional platforms.

```python
from isaacsim.robot.wheeled_robots.controllers import HolonomicController

ctrl = HolonomicController(
    name="holo_ctrl",
    omnidirectional_robot=robot,
)
action = ctrl.forward(command=np.array([0.5, 0.0, 0.3]))  # ArticulationAction
```

### Methods

| Method | Signature | Returns | Description |
|---|---|---|---|
| `forward` | `(command: np.ndarray)` | `ArticulationAction` | `[forward_m_s, lateral_m_s, yaw_rad_s]` |
| `reset` | `() → None` | `None` | Clear internal state |

---

## AckermannController

Bicycle/Ackermann steering geometry controller.

```python
from isaacsim.robot.wheeled_robots.controllers import AckermannController

ctrl = AckermannController(
    name="ack_ctrl",
    wheel_base=2.5,
    track_width=1.5,
    max_wheel_velocity=10.0,
)
action = ctrl.forward(command=np.array([steering_angle, velocity, dt]))
```

### Constructor Parameters

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Controller name |
| `wheel_base` | `float` | Longitudinal axle separation (m) |
| `track_width` | `float` | Lateral wheel separation (m) |
| `max_wheel_velocity` | `float` | Maximum wheel angular velocity (rad/s) |

### Methods

| Method | Signature | Returns | Description |
|---|---|---|---|
| `forward` | `(command: np.ndarray)` | `ArticulationAction` | `[steering_angle_rad, velocity_m_s, dt_s]` |
| `reset` | `() → None` | `None` | Clear internal state |

---

## PickPlaceController

10-phase state machine for pick-and-place operations.

```python
from isaacsim.robot.manipulators.controllers import PickPlaceController

ctrl = PickPlaceController(
    name="pick_place",
    gripper=robot.gripper,
    robot_articulation=robot,
)

# Called every physics step
action = ctrl.forward(
    picking_position=np.array([0.4, 0.0, 0.1]),
    placing_position=np.array([-0.4, 0.0, 0.05]),
    current_joint_positions=robot.get_joint_positions(),
    end_effector_offset=np.array([0.0, 0.0, 0.02]),
)
robot.apply_action(action)
is_done = ctrl.is_done()
```

### Phase State Machine

| Phase | Index | Description |
|---|---|---|
| `APPROACH_PICK_ABOVE` | 0 | Move above pick position |
| `APPROACH_PICK` | 1 | Lower to pick position |
| `CLOSE_GRIPPER` | 2 | Close gripper on object |
| `LIFT` | 3 | Lift object vertically |
| `APPROACH_PLACE_ABOVE` | 4 | Move above place position |
| `APPROACH_PLACE` | 5 | Lower to place position |
| `OPEN_GRIPPER` | 6 | Release object |
| `RETREAT` | 7 | Lift away from object |
| `DONE` | 8 | Task complete |
| `PAUSED` | 9 | Controller paused |

### Methods

| Method | Signature | Returns | Description |
|---|---|---|---|
| `forward` | `(picking_position, placing_position, current_joint_positions, end_effector_offset, ...)` | `ArticulationAction` | Step the state machine |
| `is_done` | `() → bool` | `bool` | `True` when phase is `DONE` |
| `reset` | `() → None` | `None` | Reset to phase 0 |
| `get_current_event` | `() → int` | `int` | Current phase index |
| `pause` | `() → None` | `None` | Pause at current phase |
| `resume` | `() → None` | `None` | Resume from paused phase |

---

## ArticulationKinematicsSolver

Generic IK/FK solver interface bound to an articulation.

```python
from isaacsim.robot.manipulators.kinematics import ArticulationKinematicsSolver

solver = ArticulationKinematicsSolver(
    robot_articulation=robot,
    end_effector_frame_name="ee_link",
)

action, success = solver.compute_inverse_kinematics(
    target_position=np.array([0.5, 0.0, 0.3]),
    target_orientation=np.array([1.0, 0.0, 0.0, 0.0]),  # wxyz quaternion
)
if success:
    robot.apply_action(action)

ee_pos, ee_rot = solver.compute_end_effector_position()
```

### Methods

| Method | Signature | Returns | Description |
|---|---|---|---|
| `compute_inverse_kinematics` | `(target_position, target_orientation) → (ArticulationAction, bool)` | `(action, success)` | IK for given pose |
| `compute_end_effector_position` | `() → (np.ndarray, np.ndarray)` | `(pos_xyz, rot_wxyz)` | Forward kinematics |
| `set_robot_base_pose` | `(pos, rot) → None` | `None` | Update world-frame base transform |

---

## LulaKinematicsSolver

LULA-based IK solver with configurable solver parameters.

```python
from isaacsim.robot.manipulators.kinematics import LulaKinematicsSolver

solver = LulaKinematicsSolver(
    robot_description_path="/path/to/robot_descriptor.yaml",
    urdf_path="/path/to/robot.urdf",
)

action, success = solver.compute_inverse_kinematics(
    frame_name="ee_link",
    target_position=np.array([0.5, 0.0, 0.3]),
    target_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
    warm_start=robot.get_joint_positions(),
)
```

### Constructor Parameters

| Parameter | Type | Description |
|---|---|---|
| `robot_description_path` | `str` | Path to LULA robot descriptor YAML |
| `urdf_path` | `str` | Path to robot URDF |

### Solver Configuration

```python
solver.set_solver_param("max_iterations", 500)
solver.set_solver_param("position_tolerance", 1e-4)
solver.set_solver_param("orientation_tolerance", 1e-3)
```

| Parameter Key | Default | Description |
|---|---|---|
| `max_iterations` | `150` | Maximum IK iterations |
| `position_tolerance` | `1e-4` | Position convergence threshold (m) |
| `orientation_tolerance` | `1e-3` | Orientation convergence threshold (rad) |

### Methods

| Method | Signature | Returns | Description |
|---|---|---|---|
| `compute_inverse_kinematics` | `(frame_name, target_position, target_orientation, warm_start)` | `(ArticulationAction, bool)` | IK with optional warm start |
| `compute_forward_kinematics` | `(frame_name, joint_positions)` | `(pos, rot)` | FK for given joint config |
| `get_all_frame_names` | `() → list[str]` | `list[str]` | All kinematic frames in the chain |

---

## RmpFlow

Reactive Motion Policy: real-time, collision-aware motion generation.

```python
from isaacsim.robot.motion_generation import RmpFlow

rmpflow = RmpFlow(
    robot_description_path="/path/to/robot_descriptor.yaml",
    urdf_path="/path/to/robot.urdf",
    rmpflow_config_path="/path/to/rmpflow_config.yaml",
    end_effector_frame_name="ee_link",
    maximum_substep_size=0.00334,
    ignore_robot_state_updates=False,
)
rmpflow.set_robot_base_pose(robot_pos, robot_rot)
rmpflow.update_world(obstacles)

rmpflow.set_end_effector_target(
    target_position=np.array([0.5, 0.0, 0.3]),
    target_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
)
```

### Methods

| Method | Signature | Description |
|---|---|---|
| `set_end_effector_target` | `(target_position, target_orientation) → None` | Set goal pose |
| `update_world` | `(obstacles: list) → None` | Update collision world |
| `set_robot_base_pose` | `(pos, rot) → None` | Update robot base in world frame |
| `add_obstacle` | `(obstacle_prim, static: bool) → None` | Add a USD prim as an obstacle |
| `remove_obstacle` | `(obstacle_prim) → None` | Remove a previously added obstacle |
| `reset` | `() → None` | Reset internal policy state |

---

## ArticulationMotionPolicy

Adapter that connects a motion policy (e.g. RmpFlow) to an articulation.

```python
from isaacsim.robot.motion_generation import ArticulationMotionPolicy

motion_policy = ArticulationMotionPolicy(
    robot_articulation=robot,
    motion_policy=rmpflow,
    default_physics_dt=1 / 60.0,
)

# Per physics step:
motion_policy.move()
action = motion_policy.get_next_articulation_action()
robot.apply_action(action)
```

### Methods

| Method | Signature | Returns | Description |
|---|---|---|---|
| `move` | `() → None` | `None` | Advance the motion policy by one physics step |
| `get_next_articulation_action` | `() → ArticulationAction` | `ArticulationAction` | Retrieve the computed action |
| `set_robot_base_pose` | `(pos, rot) → None` | `None` | Propagates new base pose to underlying policy |
| `reset` | `() → None` | `None` | Reset policy and adapter state |
