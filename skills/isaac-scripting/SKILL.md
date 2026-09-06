---
name: isaac-scripting
description: This skill should be used when the user asks to "execute a script in Isaac Sim", "run Python in Isaac", "write a custom Isaac Sim script", "use the omni API", "use the pxr API", "use the isaacsim API", "execute_isaac_script", or needs to perform operations not covered by the granular MCP tools.
version: 0.1.0
---

# Isaac Sim Scripting via execute_isaac_script

## When to Use This Tool vs Granular MCP Tools

**Prefer granular MCP tools first.** The simul plugin exposes purpose-built tools for the most common operations:

| Need | Use Instead |
|------|-------------|
| Build an object (prim + transform + physics + material) | `create_isaac_object` |
| Create a prim | `create_isaac_prim` |
| Move/rotate/scale | `set_isaac_prim_transform` |
| Set an attribute | `set_isaac_prim_attribute` |
| Add rigid body | `add_isaac_rigid_body` |
| Add collision | `add_isaac_collision` |
| Physics material | `set_isaac_physics_material` |
| Start/stop sim | `start_isaac_simulation`, `stop_isaac_simulation` |
| Query scene | `get_isaac_scene_summary`, `get_isaac_prim_detail` with `aspects=["info"]` |

**Use `execute_isaac_script` when:**
- The required API is not exposed by any granular tool (e.g. creating lights, querying rigid body velocities, custom material shaders, bounding box computations, raycasting)
- You need to batch multiple low-level USD operations in a single round-trip to reduce latency
- You need full `pxr.*`, `omni.*`, or `isaacsim.*` API access
- You are iterating over many prims (stage traversal, bulk attribute reads)

## Execution Model

Scripts are sent over TCP to port **8226** (the stock `isaacsim.code_editor.vscode` extension). The executor runs your code with `compile()` inside Kit's Python process, giving full access to the global namespace including `omni.*`, `pxr.*`, and `isaacsim.*`.

Key rules:
- `stdout` is captured and returned — use `print()` for all output
- For structured results, always end with `print(json.dumps({...}))` as the final line
- Top-level `await` is supported — the executor detects coroutines automatically
- Each execution is **independent**: always `import` everything at the top of each script; do not rely on state from previous executions
- Avoid long-running loops or `time.sleep()` calls that block the Kit main thread

## Script Structure

Every script should follow the **error-safe template**:

```python
import json
import traceback

try:
    # --- your logic here ---
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    prim_count = len(list(stage.Traverse()))
    result = {"success": True, "prim_count": prim_count}
except Exception as e:
    result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}

print(json.dumps(result))
```

Always wrap in try/except. Always end with `print(json.dumps(result))`. This ensures the tool always returns parseable JSON even when the script raises an exception.

## Namespace: Isaac Sim 5.1 and 6.0

Isaac Sim 5.1.0 migrated the core package namespace. Use the new names — the old `omni.isaac.*` imports will fail (5.1 keeps shims, 6.0 removes them):

| Old (pre-5.1) | New (5.1.0+) |
|---------------|--------------|
| `omni.isaac.core` | `isaacsim.core.api` |
| `omni.isaac.core.utils` | `isaacsim.core.utils` |
| `omni.isaac.core.utils.prims` | `isaacsim.core.utils.prims` |
| `omni.isaac.core.utils.stage` | `isaacsim.core.utils.stage` |
| `omni.isaac.core.utils.xforms` | `isaacsim.core.utils.xforms` |
| `omni.isaac.core.utils.bounds` | `isaacsim.core.utils.bounds` |
| `omni.isaac.core.utils.rotations` | `isaacsim.core.utils.rotations` |
| `omni.isaac.core.utils.collisions` | `isaacsim.core.utils.collisions` |
| `omni.isaac.core.utils.viewports` | `isaacsim.core.utils.viewports` |

Kit-level APIs are **unchanged**: `omni.usd`, `omni.timeline`, `pxr.*` (Usd, UsdGeom, UsdPhysics, UsdShade, UsdLux, Gf, Sdf, etc.) all import normally.

On **Isaac Sim 6.0** (`get_isaac_runtime_info` → `app.isaac_version` starts with `6.`), `isaacsim.core.api`, `isaacsim.core.prims`, and `isaacsim.core.utils` are deprecated; prefer `isaacsim.core.experimental.*` and `isaacsim.core.simulation_manager`, or plain `pxr` + `omni.usd`, which work identically on both versions. Never use `asyncio.wait_for` in a script — the 6.0 socket server runs top-level `await` outside a Task and it raises. See `references/namespace-migration.md` for the full table.

## Common Script Patterns

### Check simulation state

```python
import json, omni.timeline
tl = omni.timeline.get_timeline_interface()
print(json.dumps({
    "playing": tl.is_playing(),
    "stopped": tl.is_stopped(),
    "current_time": tl.get_current_time(),
}))
```

### Traverse and list all prims by type

```python
import json, omni.usd
stage = omni.usd.get_context().get_stage()
prims = [
    {"path": str(p.GetPath()), "type": p.GetTypeName()}
    for p in stage.Traverse()
    if p.GetTypeName()  # skip pseudoroot and untyped
]
print(json.dumps({"prims": prims, "count": len(prims)}))
```

### List all Mesh prims

```python
import json, omni.usd
stage = omni.usd.get_context().get_stage()
meshes = [str(p.GetPath()) for p in stage.Traverse() if p.GetTypeName() == "Mesh"]
print(json.dumps({"meshes": meshes, "count": len(meshes)}))
```

### Set prim transform via USD attribute

```python
import json, traceback
try:
    from isaacsim.core.utils.prims import set_prim_attribute_value
    from pxr import Gf
    set_prim_attribute_value("/World/Box", "xformOp:translate", Gf.Vec3d(1.0, 2.0, 0.5))
    set_prim_attribute_value("/World/Box", "xformOp:scale", Gf.Vec3d(2.0, 2.0, 2.0))
    result = {"success": True}
except Exception as e:
    import traceback
    result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}
print(json.dumps(result))
```

### Get world pose

```python
import json, traceback
try:
    from isaacsim.core.utils.xforms import get_world_pose
    pos, quat = get_world_pose("/World/Robot")
    result = {"success": True, "position": pos.tolist(), "orientation_wxyz": quat.tolist()}
except Exception as e:
    result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}
print(json.dumps(result))
```

### Compute bounding box

```python
import json, traceback
try:
    from isaacsim.core.utils.bounds import create_bbox_cache, compute_aabb
    import numpy as np
    cache = create_bbox_cache()
    aabb = compute_aabb(cache, "/World/Robot")
    center = ((aabb[:3] + aabb[3:]) / 2).tolist()
    size = (aabb[3:] - aabb[:3]).tolist()
    result = {"success": True, "center": center, "size": size, "aabb": aabb.tolist()}
except Exception as e:
    result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}
print(json.dumps(result))
```

## Common Pitfalls

**Do not reuse state between executions.** Each `execute_isaac_script` call is isolated. Variables, imports, and objects from a previous call are not available. Always re-import.

**Do not block the main thread.** Never use `while True`, `time.sleep()`, or any blocking wait inside a script. For physics stepping, use `world.step(render=True)` in a bounded loop (e.g. `for _ in range(10)`), not an infinite one.

**Use `World.instance()` not `World()` when the world already exists.** Calling `World()` a second time raises an error. Always check:

```python
from isaacsim.core.api import World
world = World.instance()
if world is None:
    world = World(physics_dt=1/60, rendering_dt=1/60)
```

**JSON-serialise everything.** `pxr.Gf.Vec3d`, numpy arrays, and USD types are not JSON-serialisable. Convert with `.tolist()` or `list(v)` before passing to `json.dumps`.

**Attribute access requires the prim to be valid.** Always check `prim.IsValid()` before calling methods on it. A prim retrieved from a path that does not exist returns an invalid prim that will raise on method calls.

## Reference Files

- `references/api-quick-reference.md` — full API task-to-module lookup table
- `references/namespace-migration.md` — complete old-to-new namespace mapping
- `references/script-templates.md` — copy-paste templates for common patterns
