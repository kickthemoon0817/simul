# Lighting Patterns

All lights are defined via `pxr.UsdLux`. Use `execute_isaac_script` to create lights since there is no dedicated granular MCP tool for light creation.

## Distant Light (Directional Sun)

Simulates a light source at infinite distance (sun, moon). Intensity in nits. Angle controls the angular diameter of the light source — 0.53 degrees matches the real sun.

```python
import json, traceback
try:
    from pxr import UsdLux, Gf
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    light = UsdLux.DistantLight.Define(stage, "/World/Sun")
    light.CreateIntensityAttr(3000)
    light.CreateAngleAttr(0.53)
    light.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.85))  # warm white
    result = {"success": True, "path": "/World/Sun"}
except Exception as e:
    result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}
print(json.dumps(result))
```

**Typical values:**
- Outdoor sun: intensity 3000, angle 0.53, color (1.0, 0.95, 0.85)
- Overcast sky: intensity 800, angle 5.0, color (0.85, 0.90, 1.0)
- Night moon: intensity 50, angle 0.5, color (0.8, 0.85, 1.0)

---

## Dome Light (Environment / Ambient)

Provides uniform ambient illumination from all directions. Can use an HDRI texture for realistic environment reflections.

```python
import json, traceback
try:
    from pxr import UsdLux
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(1000)
    # Optional: HDRI texture for realistic environment
    # dome.CreateTextureFileAttr("/path/to/environment.hdr")
    result = {"success": True, "path": "/World/DomeLight"}
except Exception as e:
    result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}
print(json.dumps(result))
```

**Typical values:**
- Neutral indoor: intensity 500-1000
- Bright outdoor: intensity 2000-5000
- Subtle fill only: intensity 200-400

To use an HDRI file from Nucleus: `dome.CreateTextureFileAttr("omniverse://localhost/NVIDIA/Assets/Skies/Clear/kloofendal_43d_clear.hdr")`

---

## Sphere Light (Point Light)

Emits light from a spherical source of given radius. Simulates bulbs, LEDs, or any localised light source.

```python
import json, traceback
try:
    from pxr import UsdLux, UsdGeom, Gf
    import omni.usd
    from isaacsim.core.utils.prims import set_prim_attribute_value
    stage = omni.usd.get_context().get_stage()
    sphere = UsdLux.SphereLight.Define(stage, "/World/PointLight")
    sphere.CreateIntensityAttr(5000)
    sphere.CreateRadiusAttr(0.1)
    sphere.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    # Position via xformOp
    xform = UsdGeom.Xformable(sphere.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 3.0))
    result = {"success": True, "path": "/World/PointLight"}
except Exception as e:
    result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}
print(json.dumps(result))
```

**Typical values:**
- Desk lamp: intensity 2000, radius 0.05
- Overhead light: intensity 5000, radius 0.1
- Industrial fixture: intensity 20000, radius 0.5

---

## Recommended Minimum Lighting Setup

For any general-purpose scene, create a DistantLight + DomeLight. This gives directional shadows from the sun and soft ambient fill from the dome:

```python
import json, traceback
try:
    from pxr import UsdLux, Gf
    import omni.usd
    stage = omni.usd.get_context().get_stage()

    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Sun")
    sun.CreateIntensityAttr(3000)
    sun.CreateAngleAttr(0.53)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.85))

    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Ambient")
    dome.CreateIntensityAttr(1000)

    result = {"success": True, "lights": ["/World/Lights/Sun", "/World/Lights/Ambient"]}
except Exception as e:
    result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}
print(json.dumps(result))
```

Use `/World/Lights/` as a parent prim to keep lights organised in the stage hierarchy.

---

## Querying Existing Lights

```
mcp__simul__list_isaac_lights
mcp__simul__get_isaac_prim_detail  prim_path: "/World/Lights/Sun"  aspects: ["light"]
```
