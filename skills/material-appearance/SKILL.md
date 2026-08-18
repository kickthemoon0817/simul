---
name: material-appearance
description: This skill should be used when the user asks to "create a material", "assign a material", "change the color", "set material properties", "add a texture", "make it red/blue/green", "PBR material", "preview surface", or needs to create and manage materials and appearance in Isaac Sim.
version: 0.1.0
---

# Material and Appearance Management

This skill teaches you how to create, inspect, assign, and modify materials in a live Isaac Sim 5.1.0 scene. There is no dedicated "create material" MCP tool — material creation is done via `execute_isaac_script` with the UsdPreviewSurface API. All other material operations have dedicated tools.

## Tool Chain Overview

```
list_isaac_materials
    → get_isaac_prim_detail (aspects=["material"])         (inspect existing material)
    → set_isaac_material_property     (modify a property)
    → assign_isaac_material           (bind to a prim)

execute_isaac_script (UsdPreviewSurface)
    → assign_isaac_material           (bind new material to prim)
```

## Step-by-Step Workflow

### 1. List Existing Materials

Before creating a new material, check whether a suitable one already exists.

```
mcp__simul__list_isaac_materials
```

Returns material prim paths (e.g. `/World/Looks/RedPlastic`, `/World/Looks/Metal`).

### 2. Inspect a Material

Get the current properties of an existing material.

```
mcp__simul__get_isaac_prim_detail  aspects: ["material"]
  material_path: "/World/Looks/RedPlastic"
```

Returns shader type, diffuseColor, roughness, metallic, opacity, and other inputs.

### 3. Create a Material via Script

Use `execute_isaac_script` with the UsdPreviewSurface pattern. This is the only way to create a new material — there is no dedicated create-material tool.

**Solid color material (red):**
```python
import json
import omni.usd
from pxr import UsdShade, Sdf, Gf

ctx = omni.usd.get_context()
stage = ctx.get_stage()

mat_path = "/World/Looks/RedMaterial"
mat = UsdShade.Material.Define(stage, mat_path)
shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
shader.CreateIdAttr("UsdPreviewSurface")
shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.0, 0.0))
shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

print(json.dumps({"created": mat_path}))
```

Send this via:
```
mcp__simul__execute_isaac_script
  script: "<above python>"
```

See `references/preview-surface-patterns.md` for complete presets including metallic, glass, and rubber.

### 4. Assign a Material

Bind an existing material to a prim. Call this after creating a material or to reassign an existing one.

```
mcp__simul__assign_isaac_material
  prim_path: "/World/Cube"
  material_path: "/World/Looks/RedMaterial"
```

To assign to multiple prims, call `assign_isaac_material` once per prim.

### 5. Modify Material Properties

Change individual shader inputs on an existing material without recreating it.

```
mcp__simul__set_isaac_material_property
  material_path: "/World/Looks/RedMaterial"
  property_name: "inputs:diffuseColor"
  value: [0.0, 0.5, 1.0]
```

**Common property names:**

| Property | Value Type | Example |
|---|---|---|
| `inputs:diffuseColor` | [R, G, B] floats 0–1 | [1.0, 0.0, 0.0] |
| `inputs:roughness` | float 0–1 | 0.3 |
| `inputs:metallic` | float 0–1 | 0.8 |
| `inputs:opacity` | float 0–1 | 0.5 |
| `inputs:ior` | float | 1.5 |
| `inputs:emissiveColor` | [R, G, B] | [1.0, 0.8, 0.0] |

**Note:** Property names use the `inputs:` prefix because they are UsdShade shader inputs. Omitting the prefix will silently fail.

## Material Presets

Use these script bodies inside `execute_isaac_script`. Replace `mat_path` and `target_prim_path` as needed.

### Metallic (chrome-like)
```python
shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.8, 0.8, 0.8))
shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(1.0)
shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.2)
```

### Glass / Transparent
```python
shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.9, 0.95, 1.0))
shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.3)
shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.5)
shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.0)
```

### Rubber / Matte
```python
shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.1, 0.1, 0.1))
shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
```

### Emissive (glowing)
```python
shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.6, 0.0))
shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
```

## Decision Guide

| User Intent | Tool to Use |
|---|---|
| "What materials are in the scene?" | `list_isaac_materials` |
| "What color is that material?" | `get_isaac_prim_detail` with `aspects=["material"]` |
| "Create a red material" | `execute_isaac_script` with UsdPreviewSurface |
| "Make this object red" | `execute_isaac_script` to create + `assign_isaac_material` |
| "Change roughness of existing material" | `set_isaac_material_property` |
| "Apply material X to object Y" | `assign_isaac_material` |

## Common Pitfalls

- **No dedicated create-material tool**: You must use `execute_isaac_script`. Do not look for a `create_isaac_material` tool — it does not exist.
- **Property name prefix**: Always include `inputs:` when calling `set_isaac_material_property` (e.g. `inputs:roughness`, not `roughness`).
- **Material path vs shader path**: The material lives at `/World/Looks/Mat`. The shader is at `/World/Looks/Mat/Shader`. Use the material path for `assign_isaac_material` and `get_isaac_prim_detail` with `aspects=["material"]`. Use the shader path only in scripts.
- **Opacity requires translucency enabled**: Setting `inputs:opacity` below 1.0 via `set_isaac_material_property` may not render correctly unless the material was created with opacity support. Create the material with the opacity input defined from the start.
- **Color values are 0–1 floats**: Not 0–255. `[1.0, 0.0, 0.0]` is red.
