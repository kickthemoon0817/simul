# CLI reference

The `simul-toolkit` package installs one command, `simul`.
Every subcommand accepts `--help`. For a machine-readable list of all
commands and their parameters, run `simul commands` (always JSON).

## Output

- `--json` (a top-level option, placed before the subcommand) prints
  structured JSON. It is switched on automatically when stdout is not a TTY,
  so piping into `jq` or another program needs no flag.
- The command exits non-zero when the result has `success: false`. That is a
  domain result, not a crash: read the JSON payload.

```bash
simul --json info
simul isaac scene | jq .
```

## Server and general commands

| Command | Purpose |
|---|---|
| `simul server` | Start the MCP server |
| `simul info [--config FILE]` | Backends reachable from this machine and the tools registered for them |
| `simul commands` | Every CLI command with its parameters, as JSON |
| `simul validate-config FILE` | Validate a YAML configuration file |
| `simul version` | Version information |
| `simul stats [--tool NAME] [--recent N] [--reset]` | Tool usage statistics from the persistent log; `--reset` clears it |
| `simul logs paths` | Resolved log file paths for the current settings |
| `simul logs tail [-n N] [--follow] [--audit] [--json-log] [--tool NAME] [--file PATH]` | Pretty-print the end of a log file |

### `simul server`

```bash
simul server                                   # stdio, every available backend
simul server --backends unreal                 # only Unreal tools (plus usage stats)
simul server --backends isaac,usd
simul server --backends unreal --unreal-tools full
simul server --backends blender --blender-tools thin
simul server --backends unreal --unreal-mode attached
simul server --backends blender --blender-mode attached
simul server --transport http                  # streamable HTTP on server.host:server.port
simul server --config /abs/path/config.yaml --log-level DEBUG
```

| Option | Meaning |
|---|---|
| `-c, --config FILE` | YAML configuration file (default: the packaged `default.yaml`) |
| `-t, --transport` | `stdio` (default; the client spawns the server), `http` (streamable HTTP) or `sse` (legacy) |
| `-b, --backends` | Comma-separated subset of `isaac,unreal,usd,blender`. Default: every available backend |
| `--unreal-tools` | `thin` (default, six tools) or `full` (every granular Unreal tool). Overrides `unreal.tool_surface` |
| `--blender-tools` | `full` (default) or `thin` (the essentials in `THIN_BLENDER_TOOLS`). Overrides `blender.tool_surface` |
| `--unreal-mode` | `endpoint` (default) or `attached` (the editor chosen with `simul unreal attach`) |
| `--blender-mode` | `embedded` (default, local `bpy`) or `attached` (the window chosen with `simul blender attach`) |
| `-l, --log-level` | `DEBUG`, `INFO`, `WARNING` or `ERROR` |
| `-v, --verbose` | Verbose logging |

`--backends` also limits the startup probes: backends you did not select are
neither probed nor built.

### `simul tools`

Lists the MCP tools the server would register, grouped by backend, without
connecting to any engine. It takes the same `--backends`, `--unreal-tools`,
`--blender-tools`, `--unreal-mode` and `--blender-mode` options as `server`.

```bash
simul tools                                    # names, grouped by backend
simul --json tools --backends blender --blender-tools thin   # with descriptions and schemas
```

## `simul isaac`

Commands that talk to a running Isaac Sim accept `-H/--host` and `-p/--port`
to override the configured socket. See [isaac-sim.md](isaac-sim.md) for the
lifecycle.

**Install and transport**

| Command | Purpose |
|---|---|
| `install-bridge [--isaac-root PATH] [--symlink] [--force] [--source PATH]` | Publish the bundled `khemoo.simul` extension into `<isaac-root>/extsUser/` |
| `launch [--isaac-root PATH] [--no-headless] [--dry-run] ...` | Start Isaac Sim with the Python socket and the bridge enabled, and wait for the ports |
| `bridge-up` | On a running 5.x editor, enable the bridge through the Python socket and wait for port 8229 |
| `ping` | Check connectivity |
| `runtime-info` | Transport, bridge state (`busy`, `current_action`) and runtime diagnostics |
| `interrupt` | Stop the script the bridge is running |
| `bridge-capabilities` / `bridge-config` | Read the bridge's capability envelope or settings |
| `bridge-set-unsafe --enable/--disable [--restart]` | Toggle the bridge's raw `execute_script` action (not a security boundary; see [configuration.md](configuration.md#security)) |

`launch` flags: `--socket-port`, `--bridge-port`, `--auth-token TOKEN` or
`--generate-auth-token [--print-token]` (6.0+ only), `--wait-timeout`
(default 180 s), `--poll-interval`, `--kit-arg` (repeatable), `--log-file`.

**Scene and simulation**

| Command | Purpose |
|---|---|
| `status`, `scene` | Stage, prim counts and simulation state; scene overview |
| `list-prims`, `prim-info`, `search-prims`, `query-typed-prims` | Read prims |
| `create-prim`, `delete-prim`, `set-transform` | Edit prims |
| `create-light`, `create-material`, `create-physics-scene` | Create common prims |
| `open-stage`, `save-stage`, `new-stage` | Stage files |
| `start`, `stop`, `pause`, `step [COUNT]`, `reset` | Simulation control |
| `capture [OUTPUT] [--width] [--height] [--eye x,y,z] [--target x,y,z]` | Viewport to an image file |
| `viewport-info`, `read-aovs`, `list-aovs`, `list-render-vars` | Viewport and render data |
| `get-carb-settings`, `set-carb-settings key=value ...` | Carbonite settings |
| `list-extensions`, `enable-extension`, `disable-extension` | Extension manager |
| `logs [--level] [-n N] [--source] [--search]`, `set-log-level` | Isaac Sim console log |
| `exec CODE_OR_FILE [--raw]` | Run Python inside Isaac Sim |

```bash
simul isaac exec "print('hello')"
simul isaac step 120
simul isaac capture shot.png --eye 3,3,2 --target 0,0,0
```

## `simul unreal`

Commands accept `-H/--host` and `-p/--port` for the Remote Control endpoint
(default `localhost:30010`).

| Command | Purpose |
|---|---|
| `setup UPROJECT [--yes] ...` | Enable the plugins, write the Remote Control ini, launch the editor and wait for it. See [unreal-setup.md](unreal-setup.md) |
| `instances`, `attach [--viewport KEY]`, `status`, `detach` | Select an existing editor, map and level viewport. See [unreal-attachment.md](unreal-attachment.md) |
| `control ACTION [...]` | Named editor controls on the attached editor (`inspect`, `select_actor`, `set_property`, `move_cursor`, ...) |
| `health`, `info`, `scene`, `map` | Inspection |
| `list-actors [--class] [--tag] [-n]`, `actor-info`, `search`, `scene-graph` | Scene queries |
| `spawn CLASS [--location x,y,z] [--rotation p,y,r] [--label]`, `delete`, `set-transform`, `set-property`, `set-visibility` | Edit actors |
| `sim ACTION`, `sim-status` | Play-In-Editor: `start`, `stop`, `pause`, `resume`, `step` |
| `capture [OUTPUT] [--width] [--height] [--format png\|jpeg]` | Viewport screenshot |
| `materials` | Material information for an actor |
| `exec CODE` | Run Python inside the editor |

`setup` flags: `--port` (default 30010), `--engine-path`, `--no-launch`,
`--no-headless`, `--agent-overlay`, `--wait-timeout`, `--poll-interval`,
and the cross-host set `--bind`, `--allow-public`, `--passphrase`,
`--websocket-port`.

```bash
simul unreal setup /abs/path/MyProject.uproject --yes
simul unreal list-actors --class StaticMeshActor
simul unreal spawn StaticMeshActor --location 0,0,100
simul unreal exec "import unreal; print(unreal.EditorLevelLibrary.get_all_level_actors())"
simul unreal capture viewport.png --width 1920
```

Captures move from the editor to disk in bounded chunks, so files larger than
the MCP inline limit still arrive. `--format jpeg` converts the downloaded PNG
locally.

## `simul blender`

| Command | Purpose |
|---|---|
| `install-bridge [--output PATH]` | Build the add-on ZIP (default `~/.simul/blender/simul_blender_bridge.zip`) |
| `instances` | Bridge-enabled Blender processes, window IDs, files, scenes and unsaved state |
| `attach [--instance ID] [--window ID]` | Select a window; IDs are required only when the choice is ambiguous |
| `status` | Verify the saved attachment against the running process |
| `detach` | Forget the selection; Blender and its scene are untouched |

See [blender-attachment.md](blender-attachment.md).

## `simul usd`

Headless commands; no application needs to be running.

| Command | Purpose |
|---|---|
| `info FILE` | Stage metadata, prim counts and mesh statistics |
| `validate FILE` | Validate a USD file without fully loading it |
| `summary FILE [--format text\|json]` | Scene summary |

The CLI reads any path you give it. The equivalent MCP tools only accept
files inside the sandbox's allowed paths; see
[configuration.md](configuration.md#security).
