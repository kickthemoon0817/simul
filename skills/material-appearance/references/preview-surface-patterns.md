# UsdPreviewSurface Patterns Reference

## Base Script Template

All material creation scripts follow this structure. Replace the shader input lines with the preset you need.

```python
import json
import omni.usd
from pxr import UsdShade, Sdf, Gf

ctx = omni.usd.get_context()
stage = ctx.get_stage()

mat_path = "/World/Looks/MyMaterial"
mat = UsdShade.Material.Define(stage, mat_path)
shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
shader.CreateIdAttr("UsdPreviewSurface")

# --- INSERT SHADER INPUTS HERE ---

mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
print(json.dumps({"created": mat_path}))
```

Send the full script via `mcp__simul__execute_isaac_script`.

## Preset: Solid Color

```python
r, g, b = 1.0, 0.0, 0.0  # red
shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(r, g, b))
shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
```

Common color values (R, G, B):
- Red: (1.0, 0.0, 0.0)
- Green: (0.0, 0.8, 0.0)
- Blue: (0.0, 0.2, 1.0)
- White: (1.0, 1.0, 1.0)
- Black: (0.02, 0.02, 0.02)
- Yellow: (1.0, 0.9, 0.0)
- Orange: (1.0, 0.4, 0.0)

## Preset: Metallic (Chrome / Steel)

```python
shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.8, 0.8, 0.8))
shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(1.0)
shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.2)
```

For brushed steel, increase roughness to 0.5. For gold, use diffuseColor (1.0, 0.78, 0.34).

## Preset: Glass / Transparent

```python
shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.9, 0.95, 1.0))
shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.3)
shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.5)
shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.0)
shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
```

IOR reference values: air=1.0, water=1.33, glass=1.5, diamond=2.42.

## Preset: Rubber / Matte Plastic

```python
shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.1, 0.1, 0.1))
shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
```

For colored rubber (e.g. red robot gripper), change diffuseColor and keep high roughness.

## Preset: Emissive (Glowing / LED)

```python
shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.6, 0.0))
shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
```

## Create + Assign in One Script

To create a material and immediately bind it to a prim in one script call:

```python
import json
import omni.usd
from pxr import UsdShade, Sdf, Gf, Usd

ctx = omni.usd.get_context()
stage = ctx.get_stage()

target_prim_path = "/World/Cube"
mat_path = "/World/Looks/BluePlastic"

mat = UsdShade.Material.Define(stage, mat_path)
shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
shader.CreateIdAttr("UsdPreviewSurface")
shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.2, 1.0))
shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

target_prim = stage.GetPrimAtPath(target_prim_path)
UsdShade.MaterialBindingAPI(target_prim).Bind(mat)

print(json.dumps({"created": mat_path, "bound_to": target_prim_path}))
```

## UsdPreviewSurface Input Reference

| Input Name | Sdf Type | Range | Description |
|---|---|---|---|
| `diffuseColor` | `Color3f` | [0,1] per channel | Base albedo color |
| `roughness` | `Float` | 0.0–1.0 | 0=mirror, 1=fully diffuse |
| `metallic` | `Float` | 0.0–1.0 | 0=dielectric, 1=conductor |
| `opacity` | `Float` | 0.0–1.0 | 1=opaque, 0=invisible |
| `ior` | `Float` | >1.0 | Index of refraction |
| `emissiveColor` | `Color3f` | [0,∞) | Self-illumination color |
| `specularColor` | `Color3f` | [0,1] | Specular tint (non-metallic) |
| `occlusion` | `Float` | 0.0–1.0 | Ambient occlusion factor |
| `normal` | `Normal3f` | unit vector | Tangent-space normal map value |
