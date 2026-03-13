# Timeline API vs World API

Isaac Sim exposes two distinct APIs for simulation control. Choose based on your control requirements.

## omni.timeline — Playback Control

The timeline is the lowest-level playback interface. It maps directly to the Play/Pause/Stop buttons in the Isaac Sim UI. Use it when you need simple playback control or need to query playback metadata.

**When to use:**
- Starting, pausing, or stopping simulation playback
- Querying current simulation time, FPS, or time range
- Simple open-loop runs where you do not need to read state between steps

**Key functions:**

```python
import omni.timeline

tl = omni.timeline.get_timeline_interface()

tl.play()                          # start playback
tl.pause()                         # pause at current time
tl.stop()                          # stop and rewind to 0

tl.is_playing()                    # bool
tl.is_stopped()                    # bool
tl.get_current_time()              # float, seconds
tl.get_time_codes_per_second()     # float, FPS
tl.get_start_time()                # float, timeline start
tl.get_end_time()                  # float, timeline end
tl.set_current_time(t)             # seek to time t
```

**Notes:**
- Does not require constructing any object — call `get_timeline_interface()` directly
- Works even without a World instance
- `stop()` rewinds time to 0 but does not reset prim transforms (use `World.reset()` for that)

---

## isaacsim.core.api.World — Physics Stepping

The World class provides a higher-level physics loop interface with explicit control over timestep sizes. Use it when you need deterministic, step-by-step physics with observations between steps.

**When to use:**
- Stepping physics by a precise number of frames
- Reading sensor state, joint positions, or rigid body velocities between steps
- Reinforcement learning or scripted control loops
- Resetting the scene (transforms + physics state) between episodes

**Key functions:**

```python
from isaacsim.core.api import World

# Always get the singleton — never call World() if one already exists
world = World.instance()
if world is None:
    world = World(physics_dt=1/60, rendering_dt=1/60)

world.reset()              # reset physics state + prim transforms to authored values
world.step(render=True)    # advance one physics timestep (+ render if True)
world.is_playing()         # bool
world.is_stopped()         # bool
```

**Stepping pattern:**

```python
import json, traceback
try:
    from isaacsim.core.api import World
    world = World.instance()
    if world is None:
        world = World(physics_dt=1.0/60.0, rendering_dt=1.0/60.0)
        world.reset()

    observations = []
    for i in range(60):  # 1 second at 60 Hz
        world.step(render=True)
        # read state here if needed

    result = {"success": True, "steps": 60}
except Exception as e:
    result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}
print(json.dumps(result))
```

**Notes:**
- `World()` raises if called when an instance already exists. Always use `World.instance()` first.
- `world.reset()` calls `tl.stop()` + `tl.play()` + one internal step internally — this is the correct way to reinitialise prim transforms.
- `render=True` in `world.step()` updates the viewport. For headless/server runs where rendering is not needed, pass `render=False` for better performance.
- `physics_dt` and `rendering_dt` set the simulation and render timestep sizes at construction time. These cannot be changed after construction without destroying and recreating the World.

---

## Quick Comparison

| Concern | omni.timeline | isaacsim.core.api.World |
|---------|--------------|------------------------|
| Simple play/pause/stop | Yes — preferred | Works but overkill |
| Query current time | Yes | Yes (via timeline internally) |
| Bounded step loop | No | Yes — `world.step()` in a for loop |
| Reset prim transforms | No | Yes — `world.reset()` |
| Set physics_dt | No | Yes — at construction |
| Requires object creation | No | Yes — singleton |
| Works without a stage | Yes | Requires a valid stage |

---

## Preferred Tool Mapping

For standard simulation control, use the granular MCP tools — they wrap the timeline internally:

| Action | MCP Tool |
|--------|----------|
| Start | `mcp__simul__start_isaac_simulation` |
| Pause | `mcp__simul__pause_isaac_simulation` |
| Stop | `mcp__simul__stop_isaac_simulation` |
| Reset | `mcp__simul__reset_isaac_simulation` |
| Step N frames | `mcp__simul__step_isaac_simulation num_steps: N` |
| Get state | `mcp__simul__get_isaac_simulation_state` |
| Get time | `mcp__simul__get_isaac_simulation_time` |

Use `execute_isaac_script` with `omni.timeline` or `World` only when you need capabilities not exposed by those tools (e.g. setting timeline FPS, custom physics_dt, or reading observations inside a step loop).
