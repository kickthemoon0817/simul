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
- Exit 1 means the operation failed; exit 2 means a bad invocation, refused
  before anything is contacted. Every port option (`-p/--port`,
  `--socket-port`, `--bridge-port`, `--websocket-port`) takes an integer from
  1 to 65535, and every timeout option (`-t/--timeout`, `--wait-timeout`) a
  finite number of seconds greater than 0. Any other value exits 2 with
  `error_type: InvalidArgument`: the JSON envelope under `--json`, a message
  on stderr otherwise. So does an unknown `--transport`, `--backends`,
  `--log-level`, `--unreal-tools`, `--blender-tools`, `--unreal-mode` or
  `--blender-mode` value.
- Settings that cannot be used (a config file that is missing or not valid
  YAML, an invalid value such as `UNREAL__PORT=70000`, an unknown security
  key) exit 2 with `error_type: ConfigError` from every command, naming the
  setting, its source and the reason in `details.errors`. See
  [configuration.md](configuration.md#when-the-configuration-is-wrong).

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
| `simul validate-config FILE` | Validate a YAML configuration file strictly: YAML errors, unknown keys (with suggestions) and invalid values are errors (exit 1); retired keys and unknown non-security `SECTION__` variables are warnings |
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
| `-c, --config FILE` | YAML configuration file (default: `$CONFIG_FILE`, else the packaged `default.yaml`); missing or invalid exits 2 with `ConfigError` |
| `-t, --transport` | `stdio` (default; the client spawns the server), `http` (streamable HTTP) or `sse` (legacy) |
| `-b, --backends` | Comma-separated subset of `isaac,unreal,usd,blender`. Default: every available backend, Isaac Sim only on Linux and Windows; naming `isaac` registers it on any OS ([isaac-sim.md](isaac-sim.md#platforms)) |
| `--unreal-tools` | `thin` (default, six tools) or `full` (every granular Unreal tool). Overrides `unreal.tool_surface` |
| `--blender-tools` | `full` (default) or `thin` (the essentials in `THIN_BLENDER_TOOLS`). Overrides `blender.tool_surface` |
| `--unreal-mode` | `endpoint` (default) or `attached` (the editor chosen with `simul unreal attach`) |
| `--blender-mode` | `embedded` (default, local `bpy`) or `attached` (the window chosen with `simul blender attach`) |
| `-l, --log-level` | `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL` (any case); anything else exits 2 |
| `-v, --verbose` | Verbose logging |
| `--allow-public` | Let `http`/`sse` bind a non-loopback `server.host` (`SERVER__ALLOW_PUBLIC=true`) |

`--backends` also limits the startup probes: backends you did not select are
neither probed nor built. Isaac Sim is not probed either when its tools are
skipped (on macOS, or with `ISAAC_SIM__ENABLED=false`); the banner says so.

The server prints simul's banner on stderr; FastMCP's own banner, and the
PyPI update check that came with it, are off. MCP clients see simul's version
in `serverInfo.version`, the same one `simul version` prints. SIGTERM stops
the server the way Ctrl-C does, over stdio and HTTP alike: the adapters are
closed (`Simul 3D MCP Server shut down` in the log) and the process exits 0.
State (logs, usage log, sessions) goes under `$SIMUL_HOME` (default
`~/.simul`), created on first write; see
[configuration.md](configuration.md#state-directory-simul_home).

With `--transport http` or `sse` the server checks every request's `Host`
and `Origin` headers against loopback, the bound address and
`server.allowed_hosts` / `allowed_origins` (the DNS-rebinding defense), and
requires `Authorization: Bearer <token>` when `SERVER__AUTH_TOKEN_FILE` is
set. A non-loopback `server.host` without `--allow-public` exits 2 with
`error_type: InsecureTransport` before anything listens. The banner prints the
listener URL, the guard's host list and whether a token is required. See
[configuration.md](configuration.md#network-transports-http-and-sse).

```bash
SERVER__AUTH_TOKEN_FILE=~/.config/simul/server-token simul server --transport http
SERVER__HOST=0.0.0.0 SERVER__AUTH_TOKEN_FILE=~/.config/simul/server-token \
  simul server --transport http --allow-public
```

### `simul tools`

Lists the MCP tools the server would register, grouped by backend, without
connecting to any engine. It takes the same `--backends`, `--unreal-tools`,
`--blender-tools`, `--unreal-mode` and `--blender-mode` options as `server`.
A backend the server would skip whatever is installed (Isaac Sim on macOS)
lists no tools, and its `skipped` field gives the reason.

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
| `claims [--port N] [--agent ID] [--clear]` | List the instance claims the MCP servers on this machine hold, or clear them ([configuration.md](configuration.md#multi-agent-claims)) |
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

`exec` takes the code itself, a path ending in `.py`, or, with no argument,
code piped on stdin (`simul isaac exec < job.py`). A redirected stdin is the
script even when it is empty, so `< /dev/null` sends an empty script. A `.py`
argument that names nothing on disk is taken as code. The command exits 2, a
bad invocation, and sends nothing to Isaac Sim when:

- there is no argument and stdin is a terminal or closed
  (`error_type: InputError`);
- the `.py` path can't be read: a directory, a broken symlink, a file without
  read permission, or a path under a directory that can't be searched
  (`ReadError`, naming the path and the reason);
- the script is not UTF-8, however it arrives (`InvalidEncoding`, naming the
  first bad byte). Under a locale whose encoding is not UTF-8, an argument
  that encoding can't decode is refused the same way, and the error names the
  encoding. Pass such a script in a `.py` file or on stdin, which are always
  read as UTF-8.

## `simul unreal`

Commands accept `-H/--host` and `-p/--port` for the Remote Control endpoint
(default `localhost:30010`). With `UNREAL__MODE=attached`, the commands that talk
to the editor act on the attached one, and `--host`/`--port` exit 2 with
`InvalidArgument`; `instances`, `attach` and `setup --port` take them in either
mode.

| Command | Purpose |
|---|---|
| `setup UPROJECT [--yes] ...` | Enable the plugins, write the Remote Control ini, launch the editor and wait for it. See [unreal-setup.md](unreal-setup.md) |
| `instances [--no-scan]`, `attach [--viewport KEY]`, `status`, `detach` | Discover editors (the `list_unreal_instances` payload) and select an existing editor, map and level viewport. See [unreal-attachment.md](unreal-attachment.md) |
| `control ACTION [...]` | Named editor controls on the attached editor (`inspect`, `select_actor`, `set_property`, `move_cursor`, ...) |
| `health`, `info`, `scene`, `map` | Inspection |
| `list-actors [--class] [--tag] [-n]`, `actor-info`, `search`, `scene-graph` | Scene queries |
| `spawn PATH [--location x,y,z] [--rotation p,y,r] [--label]` (an Actor class path or an asset path), `delete`, `set-transform ACTOR [--location x,y,z] [--rotation p,y,r] [--scale x,y,z]` (an option left out keeps its value), `set-property`, `set-visibility` | Edit actors |
| `sim ACTION`, `sim-status` | Play-In-Editor: `start`, `stop`, `pause`, `resume`, `step` |
| `capture [OUTPUT] [--width] [--height] [--format png\|jpeg]` | Viewport screenshot |
| `materials MATERIAL_PATH` | Parameters (with values and, for an instance, overrides), parent, domain and blend mode of a material or material instance |
| `exec CODE_OR_FILE [--mode] [--raw]` | Run Python inside the editor; the script arrives as for [`simul isaac exec`](#simul-isaac). `--mode` is `ExecuteFile` (the default), `ExecuteStatement` or `EvaluateStatement`; any other value exits 2 with `InvalidArgument` |

`setup` flags: `--port` (default 30010), `--engine-path`, `--no-launch`,
`--no-headless`, `--agent-overlay`, `--wait-timeout`, `--poll-interval`,
and the cross-host set `--bind`, `--allow-public`, `--passphrase`,
`--websocket-port`.

```bash
simul unreal setup /abs/path/MyProject.uproject --yes
simul unreal list-actors --class StaticMeshActor
simul unreal spawn /Engine/BasicShapes/Cube.Cube --location 0,0,100 --label Crate
simul unreal spawn /Script/Engine.PointLight --location 0,0,300
simul unreal exec "import unreal; print(unreal.EditorLevelLibrary.get_all_level_actors())"
simul unreal capture viewport.png --width 1920
```

Captures move from the editor to disk in bounded chunks, so files larger than
the MCP inline limit still arrive. `--format jpeg` converts the downloaded PNG
locally.

## `simul blender`

| Command | Purpose |
|---|---|
| `install-bridge [--output PATH]` | Build the add-on ZIP (default `$SIMUL_HOME/blender/simul_blender_bridge.zip`, `SIMUL_HOME` defaulting to `~/.simul`) |
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
