# Assets API Reference — Isaac Sim 5.1.0

## URDF Import

### URDFCreateImportConfig

Creates a configuration object for URDF import. All setter methods return `None` and mutate
the config in place.

```python
from isaacsim.asset.importer.urdf import _urdf

config = _urdf.URDFCreateImportConfig()
```

### Configuration Setters

| Method | Signature | Default | Description |
|---|---|---|---|
| `set_merge_fixed_joints` | `(value: bool) → None` | `False` | Collapse links connected by fixed joints into their parent |
| `set_fix_base` | `(value: bool) → None` | `False` | Fix the root link to the world (no free-floating base) |
| `set_self_collision` | `(value: bool) → None` | `False` | Enable self-collision between links in the same robot |
| `set_default_drive_type` | `(value: int) → None` | `1` | Joint drive mode: `0`=None, `1`=Position, `2`=Velocity |
| `set_default_drive_strength` | `(value: float) → None` | `1e4` | Default joint drive stiffness |
| `set_default_position_drive_damping` | `(value: float) → None` | `1e3` | Damping for position-drive joints |
| `set_distance_scale` | `(value: float) → None` | `1.0` | Unit scale multiplier (1.0 = metres) |
| `set_density` | `(value: float) → None` | `0.0` | Default link density (kg/m³); 0 = use URDF values |
| `set_import_inertia_tensor` | `(value: bool) → None` | `True` | Import inertia tensor from URDF |
| `set_convex_decomp` | `(value: bool) → None` | `False` | Decompose collision meshes into convex hulls |
| `set_make_default_prim` | `(value: bool) → None` | `True` | Set the imported robot as the default stage prim |
| `set_create_physics_scene` | `(value: bool) → None` | `True` | Auto-create a physics scene if none exists |
| `set_up_vector` | `(x, y, z: float) → None` | `(0, 0, 1)` | World up-vector for the import coordinate frame |

### URDFParseAndImportFile

Parses a URDF file and imports it into the active USD stage.

```python
from isaacsim.asset.importer.urdf import _urdf

prim_path = _urdf.URDFParseAndImportFile(
    urdf_path="/path/to/robot.urdf",
    import_config=config,
    dest_path="/World/robot",   # optional target prim path
)
```

| Parameter | Type | Description |
|---|---|---|
| `urdf_path` | `str` | Absolute path to the `.urdf` file |
| `import_config` | `URDFImportConfig` | Config object from `URDFCreateImportConfig()` |
| `dest_path` | `str \| None` | Target USD prim path; auto-derived from robot name if omitted |

Returns `str`: the USD prim path where the robot was created.

### Full Example

```python
from isaacsim.asset.importer.urdf import _urdf

config = _urdf.URDFCreateImportConfig()
config.set_merge_fixed_joints(False)
config.set_fix_base(True)
config.set_self_collision(False)
config.set_default_drive_type(1)           # position drive
config.set_default_drive_strength(1e5)
config.set_default_position_drive_damping(1e4)
config.set_distance_scale(1.0)
config.set_convex_decomp(False)

prim_path = _urdf.URDFParseAndImportFile(
    "/home/user/robots/my_arm.urdf",
    config,
)
print(f"Imported to: {prim_path}")
```

---

## MJCF Import

### MJCFCreateImportConfig

Creates a configuration object for MJCF (MuJoCo XML) import.

```python
from isaacsim.asset.importer.mjcf import _mjcf

config = _mjcf.MJCFCreateImportConfig()
```

### Configuration Setters

| Method | Signature | Default | Description |
|---|---|---|---|
| `set_fix_base` | `(value: bool) → None` | `False` | Fix the root body to the world |
| `set_self_collision` | `(value: bool) → None` | `False` | Enable intra-robot collision |
| `set_import_inertia_tensor` | `(value: bool) → None` | `True` | Import inertia from MJCF |
| `set_distance_scale` | `(value: float) → None` | `1.0` | Unit scale multiplier |
| `set_density` | `(value: float) → None` | `0.0` | Default density (kg/m³); 0 = use MJCF values |
| `set_default_drive_type` | `(value: int) → None` | `1` | Drive mode: `0`=None, `1`=Position, `2`=Velocity |
| `set_convex_decomp` | `(value: bool) → None` | `False` | Convex hull decomposition for collision |
| `set_make_default_prim` | `(value: bool) → None` | `True` | Make imported robot the default prim |
| `set_create_physics_scene` | `(value: bool) → None` | `True` | Auto-create physics scene if absent |

### MJCFCreateAsset

Parses and imports an MJCF file into the active USD stage.

```python
from isaacsim.asset.importer.mjcf import _mjcf

result = _mjcf.MJCFCreateAsset(
    mjcf_path="/path/to/model.xml",
    import_config=config,
    prim_path="/World/mujoco_robot",
)
```

| Parameter | Type | Description |
|---|---|---|
| `mjcf_path` | `str` | Absolute path to the `.xml` MJCF file |
| `import_config` | `MJCFImportConfig` | Config from `MJCFCreateImportConfig()` |
| `prim_path` | `str` | Target USD prim path |

Returns `str`: the USD prim path of the imported asset.

---

## Cloner

Base cloner that duplicates a source prim to an arbitrary set of target paths.

```python
from isaacsim.core.cloner import Cloner

cloner = Cloner()
cloner.define_base_env(num_envs=4, env_ns="/World/envs")

paths = cloner.generate_paths(root="/World/envs/env", num_paths=4)
# → ["/World/envs/env_0", "/World/envs/env_1", ...]

cloner.clone(
    source_prim_path="/World/envs/env_0",
    prim_paths=paths,
    positions=np.array([[0, 0, 0], [2, 0, 0], [4, 0, 0], [6, 0, 0]]),
    orientations=None,              # np.ndarray (N, 4) wxyz; None = identity
    replicate_physics=True,         # share physics scene graph across clones
)
```

### Methods

| Method | Signature | Returns | Description |
|---|---|---|---|
| `define_base_env` | `(num_envs: int, env_ns: str) → None` | `None` | Create the namespace and base env prim |
| `generate_paths` | `(root: str, num_paths: int) → list[str]` | `list[str]` | Produce `"{root}_{i}"` paths for `i` in `range(num_paths)` |
| `clone` | `(source_prim_path, prim_paths, positions, orientations, replicate_physics)` | `None` | Duplicate source to all target paths |
| `filter_collisions` | `(physicsscene_path, collision_root, prim_paths, global_paths)` | `None` | Disable intra-environment collisions |

### `clone` Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `source_prim_path` | `str` | — | USD path of the prim to duplicate |
| `prim_paths` | `list[str]` | — | Destination paths for each clone |
| `positions` | `np.ndarray (N, 3)` | `None` | Translation of each clone in world space |
| `orientations` | `np.ndarray (N, 4)` | `None` | Quaternion `wxyz` rotation; `None` = identity |
| `replicate_physics` | `bool` | `False` | Share physics graph across all clones (faster) |

---

## GridCloner

Convenience cloner that arranges clones in a regular grid.

```python
from isaacsim.core.cloner import GridCloner

cloner = GridCloner(spacing=2.0)
# or with explicit row width:
cloner = GridCloner(spacing=2.0, num_per_row=4)
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `spacing` | `float` | — | Distance between adjacent clone origins (m) |
| `num_per_row` | `int` | `None` | Clones per row; square grid if `None` |

### Properties

| Property | Type | Description |
|---|---|---|
| `spacing` | `float` | Grid spacing |
| `num_per_row` | `int \| None` | Row width, or `None` for auto-square |

### Methods

```python
transforms = cloner.get_clone_transforms(num_clones=16)
# returns: list[tuple[np.ndarray, np.ndarray]]
#   each entry: (translation_xyz, rotation_wxyz)

paths = cloner.generate_paths("/World/envs/env", num_clones)

cloner.clone(
    source_prim_path="/World/envs/env_0",
    prim_paths=paths,
    replicate_physics=True,
)
```

| Method | Signature | Returns | Description |
|---|---|---|---|
| `get_clone_transforms` | `(num_clones: int)` | `list[tuple[ndarray, ndarray]]` | Compute grid positions and orientations |
| `generate_paths` | `(root: str, num_paths: int)` | `list[str]` | Same as base `Cloner.generate_paths` |
| `clone` | `(source_prim_path, prim_paths, replicate_physics)` | `None` | Clone using auto-computed grid positions |

### Grid Layout Example

```python
from isaacsim.core.cloner import GridCloner
import numpy as np

cloner = GridCloner(spacing=3.0, num_per_row=4)
num_envs = 16
paths = cloner.generate_paths("/World/envs/env", num_envs)

cloner.define_base_env(num_envs=num_envs, env_ns="/World/envs")
# Build env_0 contents here, then:
cloner.clone(
    source_prim_path="/World/envs/env_0",
    prim_paths=paths,
    replicate_physics=True,
)
```

---

## OmniGraph Nodes

OmniGraph nodes are used in Action Graph pipelines. They are referenced by their
`node_type` string and configured via attribute paths.

### IsaacArticulationController

Drives an articulation's joints by receiving position, velocity, or effort arrays each frame.

| Attribute | Type | Description |
|---|---|---|
| `inputs:robotPath` | `token` | USD path to the root articulation prim |
| `inputs:jointNames` | `token[]` | Joint names to drive (subset or full set) |
| `inputs:positionCommand` | `float[]` | Target joint positions (rad or m) |
| `inputs:velocityCommand` | `float[]` | Target joint velocities (rad/s or m/s) |
| `inputs:effortCommand` | `float[]` | Target joint efforts (N·m or N) |
| `inputs:execIn` | `execution` | Execution trigger |
| `outputs:execOut` | `execution` | Passes execution downstream |

Node type path: `omni.isaac.core_nodes.IsaacArticulationController`

### IsaacArticulationState

Reads current joint state from an articulation each physics step.

| Attribute | Type | Description |
|---|---|---|
| `inputs:robotPath` | `token` | USD path to the root articulation prim |
| `inputs:jointNames` | `token[]` | Names of joints to read (empty = all) |
| `inputs:execIn` | `execution` | Execution trigger |
| `outputs:jointNames` | `token[]` | Names of the returned joints |
| `outputs:positionState` | `float[]` | Current joint positions (rad or m) |
| `outputs:velocityState` | `float[]` | Current joint velocities (rad/s or m/s) |
| `outputs:effortState` | `float[]` | Current joint efforts (N·m or N) |
| `outputs:execOut` | `execution` | Passes execution downstream |

Node type path: `omni.isaac.core_nodes.IsaacArticulationState`

### IsaacCreateRenderProduct

Creates a render product (camera + resolution) used by synthetic-data sensors.

| Attribute | Type | Description |
|---|---|---|
| `inputs:cameraPrim` | `token` | USD path to the camera prim |
| `inputs:width` | `uint` | Render product width in pixels |
| `inputs:height` | `uint` | Render product height in pixels |
| `inputs:enabled` | `bool` | Whether the render product is active |
| `inputs:execIn` | `execution` | Execution trigger |
| `outputs:renderProductPath` | `token` | USD path to the created render product |
| `outputs:execOut` | `execution` | Passes execution downstream |

Node type path: `omni.isaac.core_nodes.IsaacCreateRenderProduct`

### Wiring OmniGraph Nodes in Python

```python
import omni.graph.core as og

keys = og.Controller.Keys

(graph, nodes, _, _) = og.Controller.edit(
    {"graph_path": "/World/ActionGraph", "evaluator_name": "execution"},
    {
        keys.CREATE_NODES: [
            ("OnPlaybackTick",    "omni.graph.action.OnPlaybackTick"),
            ("ArtState",          "omni.isaac.core_nodes.IsaacArticulationState"),
            ("ArtController",     "omni.isaac.core_nodes.IsaacArticulationController"),
            ("CreateRenderProd",  "omni.isaac.core_nodes.IsaacCreateRenderProduct"),
        ],
        keys.SET_VALUES: [
            ("ArtState.inputs:robotPath",         "/World/robot"),
            ("ArtController.inputs:robotPath",    "/World/robot"),
            ("CreateRenderProd.inputs:cameraPrim","/World/Camera"),
            ("CreateRenderProd.inputs:width",     1280),
            ("CreateRenderProd.inputs:height",    720),
        ],
        keys.CONNECT: [
            ("OnPlaybackTick.outputs:tick",       "ArtState.inputs:execIn"),
            ("ArtState.outputs:execOut",          "ArtController.inputs:execIn"),
            ("ArtController.outputs:execOut",     "CreateRenderProd.inputs:execIn"),
        ],
    },
)
```
