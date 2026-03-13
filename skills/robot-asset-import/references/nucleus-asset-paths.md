# Isaac Sim Nucleus Asset Paths

## Base Path

All built-in Isaac Sim assets are served from:
```
omniverse://localhost/NVIDIA/Assets/Isaac/4.2/Isaac/
```

The Nucleus service must be running locally. Verify connectivity by checking if Isaac Sim can reach `omniverse://localhost` before using these paths.

## Robots

```
omniverse://localhost/NVIDIA/Assets/Isaac/4.2/Isaac/Robots/
```

| Robot | Path |
|---|---|
| Franka Emika Panda | `.../Robots/Franka/franka.usd` |
| Universal Robots UR10 | `.../Robots/UniversalRobots/ur10/ur10.usd` |
| Universal Robots UR3 | `.../Robots/UniversalRobots/ur3/ur3.usd` |
| Universal Robots UR5 | `.../Robots/UniversalRobots/ur5/ur5.usd` |
| Kuka iiwa | `.../Robots/KUKA/KukaIiwa7/kuka_iiwa.usd` |
| Carter V1 (mobile) | `.../Robots/Carter/carter_v1.usd` |
| Carter V2 (mobile) | `.../Robots/Carter/nova_carter.usd` |
| Jetbot | `.../Robots/Jetbot/jetbot.usd` |
| Ant (MuJoCo) | `.../Robots/Ant/ant.usd` |
| Humanoid (MuJoCo) | `.../Robots/Humanoid/humanoid.usd` |
| Spot (Boston Dynamics) | `.../Robots/BostonDynamics/spot/spot.usd` |
| Go1 (Unitree) | `.../Robots/Unitree/go1.usd` |
| H1 (Unitree) | `.../Robots/Unitree/h1.usd` |

## Environments

```
omniverse://localhost/NVIDIA/Assets/Isaac/4.2/Isaac/Environments/
```

| Environment | Path |
|---|---|
| Simple room | `.../Environments/Simple_Room/simple_room.usd` |
| Simple warehouse | `.../Environments/Simple_Warehouse/warehouse.usd` |
| Full warehouse | `.../Environments/Full_Warehouse/full_warehouse.usd` |
| Grid room | `.../Environments/Grid_Room/gridroom_curved.usd` |
| Hospital | `.../Environments/Hospital/hospital.usd` |
| Office | `.../Environments/Office/office.usd` |
| Outdoor flat | `.../Environments/Outdoor/flat_plane.usd` |

## Props

```
omniverse://localhost/NVIDIA/Assets/Isaac/4.2/Isaac/Props/
```

| Prop | Path |
|---|---|
| YCB objects (set) | `.../Props/YCB/` (directory, many items) |
| Cardboard box | `.../Props/KLT_Bin/large_KLT.usd` |
| Pallet | `.../Props/Pallet/pallet.usd` |
| Cylinder (sample) | `.../Props/Shapes/cylinder.usd` |

## Sensors

```
omniverse://localhost/NVIDIA/Assets/Isaac/4.2/Isaac/Sensors/
```

| Sensor | Path |
|---|---|
| Intel RealSense D435 | `.../Sensors/Intel/RealSense/realsense_d435.usd` |
| Velodyne VLP-16 | `.../Sensors/Velodyne/VLP_16/vlp16.usd` |
| Ouster OS0-128 | `.../Sensors/Ouster/OS0_128ch20hz512/ouster_OS0_128.usd` |
| Generic LIDAR | `.../Sensors/Generic/GenericRotatingLidar.usd` |

## Usage Example

```
mcp__simul__import_isaac_asset
  file_path: "omniverse://localhost/NVIDIA/Assets/Isaac/4.2/Isaac/Robots/Franka/franka.usd"
  prim_path: "/World/Franka"
  asset_type: "usd"
```

## Troubleshooting Nucleus Access

If the import fails with a connection error:
1. Check that Isaac Sim's Nucleus launcher is running (system tray icon on Windows, or the Nucleus service on Linux)
2. Try opening the path in Isaac Sim's Content Browser to confirm it resolves
3. As a fallback, locate the assets in the Isaac Sim installation directory: `~/.local/share/ov/pkg/isaac-sim-*/` on Linux
