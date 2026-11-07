---
name: robot-asset-import
description: Use when the user asks to "import a robot", "load a URDF", "import a USD asset", "add a robot to the scene", "load an asset", "add a reference", "import from Nucleus", or needs to bring external 3D assets and robots into an Isaac Sim scene.
version: 0.1.0
---

# Robot and Asset Import

This skill teaches you how to bring external 3D assets — robots, environments, props — into a live Isaac Sim scene. The import tools reference USD layers (`.usd`, `.usda`, `.usdc`, `.usdz`); convert URDF, OBJ, FBX and glTF to USD first (URDF: `references/urdf-usd-workflow.md`). Assets can come from local disk, Nucleus server, or the Isaac Sim built-in asset library.

## Tool Chain Overview

```
import_isaac_asset            (USD from a file path or URL, onto a new prim)
    OR
add_isaac_reference           (attach USD as a reference on an existing prim)

→ get_isaac_prim_detail (aspects=["info"])         (confirm prim exists)
→ get_isaac_subtree           (inspect imported hierarchy)
→ set_isaac_prim_transform    (position, rotate, scale)
→ get_isaac_prim_detail (aspects=["joint"])        (verify robot articulation)
```

## Step-by-Step Workflow

### 1. Import an Asset

Use `import_isaac_asset` when you have a file path (local or Nucleus) and want to place the asset at a specific prim path in the stage.

```
mcp__simul__import_isaac_asset
  asset_path: "/home/user/assets/franka.usd"
  target_path: "/World/Robot"
```

What the result tells you:

- `target_path` is where the asset landed. An import never merges into an existing prim: when `/World/Robot` is taken it goes to `/World/Robot_1` (`renamed: true`).
- `referenced_prim` is the asset prim that was referenced: its defaultPrim, or its first root prim when it has none (with a `warning`). Pass `asset_prim_path` to pick another. A reference that composes nothing fails and leaves nothing behind.
- `units` compares the asset's metersPerUnit and upAxis with the stage's. When the asset authors different ones, the import adds `xformOp:scale:unitsResolve` / `xformOp:rotateX:unitsResolve` to the new prim (`correct_units: false` only reports it). An asset that does not author its units is not guessed at.

For URDF files, convert to USD first (`references/urdf-usd-workflow.md`), then import the USD.

**Asset path formats:**

| Source | Example Path |
|---|---|
| Local file | `/home/user/assets/robot.usd` |
| Nucleus (local) | `omniverse://localhost/NVIDIA/Assets/Isaac/4.2/Isaac/Robots/Franka/franka.usd` |
| Nucleus (cloud) | `omniverse://my-server/Projects/my_asset.usd` |

See `references/nucleus-asset-paths.md` for the full Isaac Sim built-in asset library paths.

### 2. Add a USD Reference (Alternative)

Use `add_isaac_reference` when you want to attach a USD file as a reference on a prim that already exists in the stage. This is the standard USD composition arc pattern — the referenced file's content is overlaid onto the prim.

```
mcp__simul__add_isaac_reference
  prim_path: "/World/Robot"
  reference_path: "omniverse://localhost/NVIDIA/Assets/Isaac/4.2/Isaac/Robots/Franka/franka.usd"
```

**When to use `add_isaac_reference` vs `import_isaac_asset`:**
- Use `import_isaac_asset` for first-time placement — it creates the prim and imports in one step.
- Use `add_isaac_reference` when the prim already exists and you want to overlay or swap the referenced asset.

### 3. Verify the Import

Confirm the prim was created and has the expected type.

```
mcp__simul__get_isaac_prim_detail  aspects: ["info"]
  prim_path: "/World/Robot"
```

Check that `is_valid: true` and the prim type matches (e.g. `Xform` for a robot root).

### 4. Inspect the Hierarchy

After import, inspect the subtree to understand the structure — joint names, link paths, mesh locations.

```
mcp__simul__get_isaac_subtree
  prim_path: "/World/Robot"
```

This is especially important for robots: you need the exact joint and link prim paths before you can manipulate the articulation.

### 5. Position the Asset

Place the asset at the desired location and orientation in the world.

```
mcp__simul__set_isaac_prim_transform
  prim_path: "/World/Robot"
  translation: [0.0, 0.0, 0.0]
  rotation: [0.0, 0.0, 0.0]
  scale: [1.0, 1.0, 1.0]
```

- `translation`: [x, y, z] in meters (world space)
- `rotation`: [x, y, z] Euler angles in degrees
- `scale`: [x, y, z] uniform scale — use [1, 1, 1] unless you need to resize

**Scale correction for URDF imports**: URDF files are in meters but some exporters use centimeters. If the robot appears 100× too large, set `scale: [0.01, 0.01, 0.01]`.

### 6. Verify Robot Articulation

For robots, confirm joints were imported correctly.

```
mcp__simul__get_isaac_prim_detail  aspects: ["joint"]
  prim_path: "/World/Robot"
```

Returns joint names, types (revolute/prismatic/fixed), limits, and current positions. If the joint list is empty, the articulation root may not have been created — see `references/urdf-usd-workflow.md` for articulation root setup.

## Common Import Scenarios

### Franka Panda from Nucleus
```
mcp__simul__import_isaac_asset
  asset_path: "omniverse://localhost/NVIDIA/Assets/Isaac/4.2/Isaac/Robots/Franka/franka.usd"
  target_path: "/World/Franka"
```

### Warehouse Environment
```
mcp__simul__import_isaac_asset
  asset_path: "omniverse://localhost/NVIDIA/Assets/Isaac/4.2/Isaac/Environments/Simple_Warehouse/warehouse.usd"
  target_path: "/World/Environment"
```

### Local URDF Robot

Convert it to USD (`references/urdf-usd-workflow.md`), then:
```
mcp__simul__import_isaac_asset
  asset_path: "/home/user/my_robot/usd/my_robot.usd"
  target_path: "/World/MyRobot"
```

Then position it above the ground plane:
```
mcp__simul__set_isaac_prim_transform
  prim_path: "/World/MyRobot"
  translation: [0.0, 0.0, 0.05]
  rotation: [0.0, 0.0, 0.0]
  scale: [1.0, 1.0, 1.0]
```

## Decision Guide

| User Intent | Tool to Use |
|---|---|
| "Import robot from file" | `import_isaac_asset` with a USD `asset_path` (convert URDF first) |
| "Add Nucleus asset" | `import_isaac_asset` with Nucleus path |
| "Attach USD as reference" | `add_isaac_reference` |
| "Did the import work?" | `get_isaac_prim_detail` with `aspects=["info"]` |
| "Show me the robot structure" | `get_isaac_subtree` |
| "Move the robot to position X" | `set_isaac_prim_transform` |
| "What joints does it have?" | `get_isaac_prim_detail` with `aspects=["joint"]` |

## Common Pitfalls

- **Prim path already exists**: `import_isaac_asset` never merges into an occupied prim; it imports to the first free `<target_path>_N` and reports the path it used. Read `target_path` from the result before addressing the asset.
- **Nucleus server not running**: Nucleus paths (`omniverse://localhost/...`) require the Nucleus service to be active. Local file paths (`/home/...`) work without Nucleus.
- **Scale or up-axis mismatch**: check `units` in the import result. A USD asset that authors centimetres or Y-up is corrected automatically; one that authors nothing is left as is. If it still appears 100x too large, its file is in centimetres without saying so: scale the prim by 0.01 with `set_isaac_prim_transform`.
- **No articulation joints found**: The URDF import may not have created an ArticulationRoot automatically. See `references/urdf-usd-workflow.md` for the script to add one.
- **Checking the wrong prim**: `get_isaac_prim_detail` with `aspects=["joint"]` on a non-root prim may return incomplete results. Use the root prim path returned by `get_isaac_subtree`.
