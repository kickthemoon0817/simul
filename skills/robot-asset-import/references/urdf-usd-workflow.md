# URDF and USD Import Workflow Reference

## URDF Import Notes

### What `import_isaac_asset` Does for URDF

When `asset_type: "urdf"` is used, the Isaac Sim URDF importer:
1. Parses the URDF XML and resolves mesh paths relative to the URDF file location
2. Creates a USD stage with Xform prims for each link
3. Creates UsdPhysics joints between links
4. Adds an ArticulationRoot schema to the robot root prim
5. Imports mesh files referenced in `<visual>` and `<collision>` elements

### Scale Correction

URDF files are specified in meters, but some robot exporters (especially those from CAD tools) output in centimeters or millimeters. If the robot appears oversized:

| Observed size | Likely unit mismatch | Correction scale |
|---|---|---|
| 100× too large | Exporter used cm | [0.01, 0.01, 0.01] |
| 1000× too large | Exporter used mm | [0.001, 0.001, 0.001] |

Apply via `set_isaac_prim_transform` after import:
```
mcp__simul__set_isaac_prim_transform
  prim_path: "/World/MyRobot"
  translation: [0.0, 0.0, 0.0]
  rotation: [0.0, 0.0, 0.0]
  scale: [0.01, 0.01, 0.01]
```

### Adding ArticulationRoot Manually

If `get_isaac_joint_info` returns an empty joint list, the ArticulationRoot schema may be missing. Add it via `execute_isaac_script`:

```python
import json
import omni.usd
from pxr import UsdPhysics

ctx = omni.usd.get_context()
stage = ctx.get_stage()

robot_root_path = "/World/MyRobot"
prim = stage.GetPrimAtPath(robot_root_path)
UsdPhysics.ArticulationRootAPI.Apply(prim)

print(json.dumps({"articulation_root_added": robot_root_path}))
```

## USD Reference Workflow

### Difference Between import_isaac_asset and add_isaac_reference

`import_isaac_asset`:
- Creates the prim and populates it in one step
- Best for first-time placement
- Handles URDF conversion internally

`add_isaac_reference`:
- Attaches a USD file as a composition arc (reference) on an existing prim
- The referenced file's namespace is overlaid onto the prim
- Allows multiple references on one prim (layering)
- Best for swapping assets or adding USD overlays

### Typical add_isaac_reference Workflow

```
# 1. Create an empty Xform as the container
mcp__simul__create_isaac_prim
  prim_path: "/World/Robot"
  prim_type: "Xform"

# 2. Attach the USD file as a reference
mcp__simul__add_isaac_reference
  prim_path: "/World/Robot"
  reference_path: "omniverse://localhost/NVIDIA/Assets/Isaac/4.2/Isaac/Robots/Franka/franka.usd"

# 3. Verify
mcp__simul__get_isaac_prim_info
  prim_path: "/World/Robot"
```

## Converting URDF to USD (Offline)

If you want a reusable USD file from a URDF, use the Isaac Sim URDF converter via `execute_isaac_script`:

```python
import json
from isaacsim.asset.importer.urdf import _urdf

urdf_path = "/home/user/robot/urdf/my_robot.urdf"
output_dir = "/home/user/robot/usd/"

importer = _urdf.acquire_urdf_interface()
config = _urdf.ImportConfig()
config.merge_fixed_joints = False
config.convex_decomp = False
config.import_inertia_tensor = True
config.distance_scale = 1.0
config.make_default_prim = True
config.create_physics_scene = False

result, prim_path = importer.import_robot_to_stage(urdf_path, output_dir + "my_robot.usd", config)
print(json.dumps({"success": result, "prim_path": prim_path}))
```

After conversion, use `import_isaac_asset` with `asset_type: "usd"` on the generated USD file for subsequent imports.

## Post-Import Checklist

After any robot import, verify these items in order:

1. `get_isaac_prim_info` — confirm prim exists and type is correct
2. `get_isaac_subtree` — inspect link/joint hierarchy, note exact prim paths
3. `get_isaac_joint_info` — verify joints were created with correct types and limits
4. `get_isaac_rigid_body_info` — confirm physics bodies on links
5. `set_isaac_prim_transform` — position the robot above the ground plane (z > 0)
6. Start simulation briefly (`start_isaac_simulation` → `step_isaac_simulation` × 5 → `pause_isaac_simulation`) and check robot does not explode or fall through the floor
