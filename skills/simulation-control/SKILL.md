---
name: simulation-control
description: This skill should be used when the user asks to "start the simulation", "stop the simulation", "pause simulation", "step the simulation", "reset simulation", "play simulation", "check simulation state", "run the simulation", or needs to control simulation playback in Isaac Sim.
version: 0.1.0
---

# Simulation Control Workflow

Isaac Sim simulation control maps to a state machine: stopped → playing → paused → stopped. The granular MCP tools cover the full lifecycle. Use `execute_isaac_script` only when you need the lower-level `omni.timeline` or `isaacsim.core.api.World` APIs directly.

## Simulation Lifecycle

```
[stopped] --start--> [playing] --pause--> [paused] --start--> [playing]
    ^                    |                    |
    |                    v                    v
    +<------stop---------+<-------stop--------+
    |
  reset (clears physics state, returns to time 0)
```

## Step 1: Check Current State

Always check state before issuing a command to avoid sending `start` to an already-playing simulation:

```
mcp__simul__get_isaac_simulation_state
```

Returns: `{ "playing": bool, "paused": bool, "stopped": bool }`.

Also useful for monitoring elapsed time during a run:

```
mcp__simul__get_isaac_simulation_time
```

Returns current simulation time in seconds (0.0 when stopped or just reset).

## Step 2: Start the Simulation

```
mcp__simul__start_isaac_simulation
```

Equivalent to pressing the Play button in the Isaac Sim UI. Physics begins advancing, rigid bodies start responding to gravity and contacts, and sensors start producing data.

**Precondition:** A physics scene must exist (`/physicsScene` or equivalent). If none exists, simulation will start but physics will not run. Create one first — see the `scene-setup` skill or `mcp__simul__get_isaac_physics_scene` to check.

## Step 3: Step the Simulation

For controlled stepping (instead of free-running playback), use:

```
mcp__simul__step_isaac_simulation
  num_steps: 1
```

Each step advances physics by one timestep (default 1/60 s). This is useful when you need to:
- Advance physics a precise number of frames
- Read sensor state after each step
- Run a scripted sequence with intermediate observations

For a burst of steps without reading between them, pass a larger `num_steps` (e.g. 60 for 1 second at 60 Hz). The tool blocks until all steps complete.

**Note:** Stepping only works when the simulation is already running (started). Call `start_isaac_simulation` first, then `step_isaac_simulation`.

## Step 4: Pause the Simulation

```
mcp__simul__pause_isaac_simulation
```

Freezes physics and rendering at the current time. Objects hold their positions and velocities. Resume with `start_isaac_simulation` — playback continues from where it paused.

## Step 5: Stop the Simulation

```
mcp__simul__stop_isaac_simulation
```

Stops playback and rewinds to time 0. Physics state (velocities, contacts) is cleared, but prim positions are **not** reset to their pre-simulation positions unless you also call reset.

## Step 6: Reset the Simulation

```
mcp__simul__reset_isaac_simulation
```

Resets the full simulation: clears physics state, returns timeline to 0, and restores all prim transforms to their authored USD values (the positions they had before simulation started). Use this to run the same scenario again from scratch without reloading the stage.

**Typical pattern for repeated runs:**
1. `mcp__simul__stop_isaac_simulation`
2. `mcp__simul__reset_isaac_simulation`
3. (make any scene changes)
4. `mcp__simul__start_isaac_simulation`

## Monitoring During Simulation

Poll state while a simulation is running to track progress or wait for a condition:

```
mcp__simul__get_isaac_simulation_state   # check playing/paused/stopped
mcp__simul__get_isaac_simulation_time    # check elapsed time
mcp__simul__get_isaac_prim_transform     # read object positions
mcp__simul__get_isaac_rigid_body_info    # read velocities
```

For reading physics quantities mid-simulation, prefer `execute_isaac_script` with direct USD attribute access (e.g. `physics:velocity`) as it is lower-latency than chained tool calls.

## When to Use execute_isaac_script Instead

The granular tools cover all standard lifecycle operations. Use `execute_isaac_script` when you need:

**Timeline API** — for querying `fps`, `time_codes_per_second`, or time range bounds:

```python
import json, omni.timeline
tl = omni.timeline.get_timeline_interface()
print(json.dumps({
    "playing": tl.is_playing(),
    "stopped": tl.is_stopped(),
    "current_time": tl.get_current_time(),
    "fps": tl.get_time_codes_per_second(),
    "start_time": tl.get_start_time(),
    "end_time": tl.get_end_time(),
}))
```

**World API** — for physics stepping with explicit dt control and reading observations between steps:

```python
import json, traceback
try:
    from isaacsim.core.api import World
    world = World.instance()
    if world is None:
        world = World(physics_dt=1/60, rendering_dt=1/60)
        world.reset()
    for _ in range(10):
        world.step(render=True)
    result = {"success": True, "stepped": 10}
except Exception as e:
    result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}
print(json.dumps(result))
```

**Key difference:** `omni.timeline` is for playback control (play/pause/stop/query). `isaacsim.core.api.World` is for programmatic physics stepping with explicit control over `physics_dt` and `rendering_dt`. For simple playback, always use the granular MCP tools. For custom physics loops, use `World` via `execute_isaac_script`.

See `references/timeline-vs-world.md` for detailed comparison.

## Common Patterns

### Run for N seconds then stop

```
1. mcp__simul__start_isaac_simulation
2. mcp__simul__step_isaac_simulation  num_steps: 300   # 5 sec at 60 Hz
3. mcp__simul__pause_isaac_simulation
4. mcp__simul__get_isaac_prim_transform  prim_path: "/World/MyRobot"
```

### Check state before starting (safe start)

```
1. mcp__simul__get_isaac_simulation_state
   → if playing: skip start
   → if paused: call start to resume
   → if stopped: call start
```

### Full reset cycle

```
1. mcp__simul__stop_isaac_simulation
2. mcp__simul__reset_isaac_simulation
3. mcp__simul__get_isaac_simulation_state   # confirm stopped, time=0
4. mcp__simul__start_isaac_simulation
```

## Reference Files

- `references/timeline-vs-world.md` — when to use `omni.timeline` vs `isaacsim.core.api.World`, with code examples
