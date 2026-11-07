# Isaac Sim

simul supports NVIDIA Isaac Sim **5.1.0, 6.0.0 and 6.0.1**. The MCP server
runs in its own Python interpreter (3.11 to 3.13) and reaches Isaac Sim over
TCP, so its Python version does not need to match the one Isaac ships (3.11
for 5.1, 3.12 for 6.0). Do not run the server inside Isaac's `python.sh`; that
interpreter lacks the server's dependencies.

Set `ISAAC_SIM_PATH` to the install root (for example `~/isaac-sim-6.0.1`),
ideally in your shell rc. The `isaac` commands also accept `--isaac-root`.

## Platforms

NVIDIA ships Isaac Sim for Linux and Windows only. On those platforms the MCP
server registers the Isaac tools, the instance routing tools, the
`simul://isaac-sim/*` resources and the Isaac part of its instructions. On any
other OS (macOS) it skips all of them by default: no Isaac Sim can run on that
machine, and the tool schemas would cost every agent session tens of
kilobytes of context. `simul server`, `simul info` and `simul tools` report
the skip and how to override it, and `get_capabilities` gives the reason under
`isaac.skipped`.

| `isaac_sim.enabled` (`ISAAC_SIM__ENABLED`) | Isaac tools are registered |
|---|---|
| unset or `auto` (default) | On Linux and Windows; elsewhere only when `--backends` names `isaac` |
| `true` | On any OS |
| `false` | Never, even with `--backends isaac` |

The overrides exist for setups simul does not support, such as an SSH tunnel
from a Mac to a Linux Isaac Sim host (see [Remote hosts](#remote-hosts)):

```bash
ISAAC_SIM__ENABLED=true simul server       # or: simul server --backends isaac,usd
```

The `simul isaac` commands (`launch`, `ping`, `exec`, ...) are not affected by
any of this.

## Lifecycle

### 1. Publish the bridge extension (once per Isaac install)

The `khemoo.simul` Kit extension (shown in the Extension Manager as
**Simul Bridge**) ships inside the `simul-toolkit` package (under `simul/bridge_ext/`). Isaac loads it
only from `<isaac-root>/extsUser/`, and does not pick up a newer copy until
you publish it again.

```bash
ISAAC_SIM_PATH=~/isaac-sim-6.0.1 simul isaac install-bridge            # copy
ISAAC_SIM_PATH=~/isaac-sim-6.0.1 simul isaac install-bridge --symlink  # track a repo checkout
```

- `--symlink` makes later `git pull`s take effect without re-running the
  command. Use the default copy mode if Isaac runs as a different user from
  the checkout's owner.
- Without `--force` the command does nothing when the installed version
  already matches; `--force` replaces it (for example to switch between copy
  and symlink).
- Always publish through this command rather than copying the folder by hand:
  it verifies the version after writing.

After upgrading simul in copy mode, run `install-bridge` again.

### 2. Start Isaac Sim with the transports enabled (every launch)

A plain `isaac-sim.sh` start leaves the bridge registered but disabled, so
port 8229 never opens. On 6.0 the Python socket on 8226 is off as well.

```bash
ISAAC_SIM_PATH=~/isaac-sim-6.0.1 simul isaac launch               # headless, waits for the ports
simul isaac launch --isaac-root ~/isaac-sim-5.1.0 --no-headless    # with the GUI
simul isaac launch --dry-run                                       # print the command only
```

`launch` reads `<isaac-root>/VERSION`, starts `isaac-sim.sh` detached with
`--enable <python socket extension> --enable khemoo.simul` (and
`--no-window` unless `--no-headless`), then polls both ports until they answer
or `--wait-timeout` (default 180 s) expires. Its JSON output includes `pid`,
`log_file`, `version`, `transport_extension`, `socket_reachable`,
`bridge_reachable` and `socket_protocol`. If the bridge has not been
published it starts Isaac with the Python socket only and prints the
`install-bridge` hint.

Other flags: `--socket-port`, `--bridge-port`, `--kit-arg` (repeatable
passthrough to Kit), `--log-file`, `--poll-interval`, and the 6.0 token
options below.

**Editor already running (5.x only).** Isaac Sim 5.1 enables the VS Code
socket on 8226 at startup, so simul can switch the bridge on afterwards:

```bash
simul isaac bridge-up
```

It reports `action: "already-up"` or `"auto-enabled"` with `success` and
`bridge_reachable`, and is safe to repeat. On a 6.0 editor started without
the `--enable` flags there is nothing to talk to: it reports `NotRunning`, and
you should restart Isaac through `launch`.

### 3. Verify

```bash
simul isaac ping
simul isaac runtime-info
simul isaac scene
```

From an agent, `ping_isaac` does the same check.

## Isaac Sim 6.0 differences

- The Python socket on 8226 moved from `isaacsim.code_editor.vscode` to
  `isaacsim.code_editor.python_server`, and it executes only after the client
  half-closes the connection. simul detects which flavour it is talking to
  with one probe; set `ISAAC_SIM__SOCKET_PROTOCOL=python_server` or `vscode`
  only to skip the probe.
- Neither the Python socket nor the bridge is enabled at startup. Use
  `simul isaac launch`. The equivalent manual command is
  `isaac-sim.sh --enable isaacsim.code_editor.python_server --enable khemoo.simul --no-window`.
- The python_server can require a token, and with it the bridge. Start Isaac
  with `simul isaac launch --generate-auth-token`, which lets Kit create the
  token and writes it to the discovery directory, where the server finds it
  for the newest running editor. `--auth-token <secret>` with
  `ISAAC_SIM__SOCKET_AUTH_TOKEN=<secret>` for the MCP server also works, but
  puts the secret on the editor's command line. See
  [Security model](#security-model).
- `isaacsim.core.api`, `isaacsim.core.prims` and `isaacsim.core.utils` are
  deprecated in favour of `isaacsim.core.experimental.*`, and the
  `omni.isaac.*` shims are gone. Read
  [`skills/isaac-scripting/references/namespace-migration.md`](../skills/isaac-scripting/references/namespace-migration.md)
  before writing `execute_isaac_script` code. The `simul://isaac-sim/api/core`
  resource documents the 5.1 API and lists the 6.0 equivalents.

## Reading simulated state

simul's granular readers (transforms, bounding boxes, rigid-body velocities,
camera poses) read USD. That is the simulated state only while PhysX writes
its results back to USD. For GPU physics, Isaac Sim's `SimulationManager`
(`set_physics_sim_device` on 5.1, `set_device` on 6.0) calls `enable_fabric`.
That enables `omni.physx.fabric` and sets `/physics/updateToUsd` and
`/physics/updateVelocitiesToUsd` to false. From then on USD keeps the
pre-play values while the robot moves.

The readers detect this. While the timeline is not stopped and either setting
is false, the result carries `stale_under_fabric: true`, a `usd_readback`
object (the two settings and whether `omni.physx.fabric` is enabled) and a
warning. `get_isaac_runtime_info` reports the same settings under `physics`.
simul does not read Fabric itself. To get simulated poses and velocities,
use `execute_isaac_script` with a physics view:
- on 5.1, the `isaacsim.core.prims` views in `simul://isaac-sim/api/core`;
- on 6.0, `isaacsim.core.experimental.prims.RigidPrim(...).get_world_poses()`
  and `get_velocities()`.

A stopped timeline is not flagged: USD then holds the authored state, which
`/physics/resetOnStop` (on in Isaac Sim's apps) restores on every stop.

## Transports

| Port | Transport | Notes |
|---|---|---|
| 8229 | `khemoo.simul` bridge | Preferred: typed protocol, fewer round trips, script interruption, busy state |
| 8226 | Stock Python socket | `isaacsim.code_editor.vscode` on 5.x, `isaacsim.code_editor.python_server` on 6.0; used when the bridge is unavailable |

- **Timeouts.** Every script carries a budget derived from
  `ISAAC_SIM__SOCKET_TIMEOUT` (one second under it, minimum 1 s). Tools that
  routinely take longer (`open_isaac_stage`, `new_isaac_stage`,
  `read_isaac_aovs`, `capture_isaac_viewport`, `enable_isaac_extension`)
  send at least 110 s; a longer configured timeout is kept. What happens
  when a budget runs out depends on the transport:
  - the bridge interrupts the script inside Kit, but Kit work the script had
    started (a stage load, an extension enable, a render) is not undone and
    may still complete;
  - the 6.0 python_server only replies with a `TimeoutError`; the script
    itself keeps running inside Kit and its effects may still land;
  - the 5.x VS Code socket has no timeout; the client stops waiting and the
    script runs on.

  Every timeout reply says so. Check the state (`get_isaac_stage_info`,
  `get_isaac_runtime_info`) before retrying a timed-out call.
- **5.x stock socket size limit.** Isaac Sim 5.1's `isaacsim.code_editor.vscode`
  has no framing: it runs every chunk of bytes it receives as a separate
  script, replies to the first and closes the connection. A script that
  arrives in two reads would execute as two fragments. simul therefore sends
  that socket at most 32 KiB per script (one loopback read) and refuses larger
  ones with `ScriptTooLargeForSocket`; every generated tool script fits well
  under it. Send bigger `execute_isaac_script` code through the bridge
  (`simul isaac bridge-up`), which reads a length-prefixed frame. Over a
  container network or an SSH tunnel even smaller scripts can be split, so use
  the bridge there. The 6.0 python_server buffers until the client half-closes
  and has no such limit.
- **Shared stdout.** Kit's `sys.stdout` is one stream for every Python
  callback. The bridge captures a script's output only while the script's own
  code runs; the stock sockets capture it across the script's awaits, so a
  physics callback or script node that prints during those frames lands in the
  reply. Tools take the JSON object their script printed last and report any
  other text under `foreign_output`, with a note in `warnings`, instead of
  failing to parse.
- **Interrupting.** `interrupt_isaac_script` / `simul isaac interrupt`
  stops a running script when it is a coroutine or suspended at an `await`. A
  synchronous loop that never yields also blocks the bridge's event loop, so
  only the timeout reaches it; a blocking C call is interrupted when it
  returns to Python.
- **Cancelled calls.** When an MCP client cancels a call (or times out on
  it), Isaac Sim keeps running the script. The server asks the bridge to
  interrupt it and keeps the instance until the engine has finished, so the
  next call is not let in while the first is still being served (it waits,
  or gets `InstanceBusy` after `server.timeout`). The cancelled call is
  recorded with `error: "cancelled"`.
- **Ping while busy.** `ping_isaac` does not wait for a call already running
  on the instance; its `busy` field says one is. A synchronous script blocks
  the stock socket and the bridge alike, so a ping during one can fail even
  though the instance is up.
- **Busy or hung?** `get_isaac_runtime_info` reports `bridge.busy`,
  `bridge.busy_since` and `bridge.current_action`.
- **Circuit breaker.** A filtered or half-open bridge port would cost the full
  timeout on every call. After `ISAAC_SIM__BRIDGE_FAILURE_THRESHOLD`
  consecutive connection failures (default 3) the client skips the bridge for
  `ISAAC_SIM__BRIDGE_COOLDOWN_SECONDS` (default 30) and uses the stock socket.
  Only failures to reach the bridge count: a request the bridge accepted and
  has not answered yet (a slow script) does not, because sending the next
  scripts to the stock socket would run them beside the one still going.
  `ping_isaac`, `list_isaac_instances` and `get_isaac_runtime_info` report
  this as `bridge_circuit_open`.
- **Raw scripts over the bridge.** The bridge's `allow_unsafe_execution`
  setting (`simul isaac bridge-set-unsafe`) only gates raw scripts on
  8229. With it off the bridge replies `UnsafeExecutionDisabled` without
  running anything, and the client sends the script (a generated tool script
  or `execute_isaac_script` code) to the stock socket on 8226 instead, when
  fallback is on and that socket is enabled. It is not a security boundary;
  to remove scripting from agents use `SECURITY__ALLOW_SCRIPT_EXECUTION=false`
  ([configuration.md](configuration.md#script-execution)).

## Security model

Both transports run arbitrary Python inside Isaac Sim, with the editor's
privileges, for whoever can connect. They bind to `127.0.0.1` (and the bridge
optionally to a Unix socket), so the network is not the exposure; **other
local users and processes are**.

| | Isaac Sim 5.1 | Isaac Sim 6.0 |
|---|---|---|
| Stock socket on 8226 | `isaacsim.code_editor.vscode`: no authentication at all | `isaacsim.code_editor.python_server`: optional token (`require_auth`) |
| Bridge on 8229 | Optional token (`/exts/khemoo.simul/require_auth`, `auth_token`) | Requires python_server's token whenever python_server does, and accepts it |
| `simul isaac launch --generate-auth-token` | Not available (no authenticated stock socket) | Both ports require the Kit-generated token; nothing secret on argv |

- **Default: no token.** Any local process can run Python through either
  port, as with Isaac Sim's own VS Code socket. The bridge logs a warning at
  startup saying so. This is acceptable on a single-user workstation and not
  on a shared host.
- **With a token.** `launch --generate-auth-token` (6.0+) turns on
  python_server's `require_auth` and the bridge's `require_auth`. Kit
  generates the token at startup; `launch` reads it from the launch log and
  stores it `0600` in the discovery directory, where the MCP server picks it
  up and sends it to both transports. The bridge checks it on every request,
  framed or raw, before anything runs, compares it in constant time, and
  rejects everything when auth is required but no token is configured. A
  client without the token gets `AuthenticationError`.
- **The bridge's own token.** `--/exts/khemoo.simul/auth_token=<token>`
  (with `ISAAC_SIM__SOCKET_AUTH_TOKEN=<token>` for the server) protects the
  bridge on its own, for example on 5.1. There the VS Code socket on 8226
  stays unauthenticated, so it only helps if that socket is disabled or
  firewalled. Kit settings on argv are visible to other local users through
  the process list.
- **Unix socket.** Created `0600` (owner only; `connect(2)` needs write
  permission). The Compose file opts into `socket_mode=0666` because its
  bridge runs as container root while the host user connects; there the
  discovery directory's mode and the token are the boundary.
- **Tokens stay out of results.** `get_isaac_carb_settings` reads any
  `auth_token` key back as `***`, the launch output redacts an `--auth-token`
  value, and the token never travels inside an action's payload.
- **Not a boundary.** `allow_unsafe_execution=false` on the bridge (see
  above). The control for agent-authored code is
  `SECURITY__ALLOW_SCRIPT_EXECUTION=false`.

The bridge extension's version moves in lockstep with simul's
([AGENTS.md](../AGENTS.md#versioning)); after upgrading, run `install-bridge`
again (copy mode) so Isaac loads a bridge that checks tokens.

## Multiple instances

Each bridge writes a discovery file to `isaac_sim.discovery_dir` (default
`/tmp/simul`) with its bridge port and Python socket port. Agents list
them with `list_isaac_instances` and switch with `set_active_isaac_instance`.
`claim_isaac_instance` / `release_isaac_instance` mark an instance as in use;
claims are advisory unless `ISAAC_SIM__ENFORCE_CLAIMS=true`. The switch and
the claim belong to the caller's MCP session, which an HTTP client on the
sessionless 2026-07-28 protocol does not have
([configuration.md](configuration.md#multi-agent-claims)). After a reconnect
or a server restart, claiming again with the same `agent_id` takes the
orphaned claim over; `simul isaac claims` lists the claims on this machine and
`--clear` removes them.

## Remote hosts

Cross-host Isaac Sim is not supported. The server trusts only loopback
addresses in discovery files, `launch` binds the transports on the local
machine, and neither transport authenticates beyond the optional token
([Security model](#security-model)), which travels in plaintext: anything
that can reach the ports can run Python inside Isaac Sim. Run `simul` on the same host as Isaac, or for a container,
publish the ports to the host's loopback as the Compose file does. Unreal's
`--bind` / `--allow-public` / `--passphrase` flow has no Isaac counterpart.

A server on macOS skips the Isaac tools (see [Platforms](#platforms)). If you
forward the ports from a Linux host anyway, for example over an SSH tunnel,
you are outside what simul supports: register the tools with
`ISAAC_SIM__ENABLED=true` or `--backends isaac` and keep the forwarded ports on
loopback.

## Containerized Isaac Sim (Linux)

[`compose.isaac-sim.yml`](../compose.isaac-sim.yml) runs the official
`nvcr.io/nvidia/isaac-sim` image (5.1.0 by default) with the bridge mounted
from this checkout:

```bash
docker compose -f compose.isaac-sim.yml up -d
docker compose -f compose.isaac-sim.yml down
```

The Compose file:

- mounts `./src/simul/bridge_ext/khemoo.simul` read-only into
  `/tmp/extsUser/khemoo.simul` and starts
  `isaac-sim.sh --allow-root --no-window` with the bridge and the Python socket
  extension enabled;
- binds both transports to `0.0.0.0` *inside* the container (a
  container-loopback bind cannot receive a published port: the host would
  connect and then be closed on with no data) and publishes them to
  `127.0.0.1` on the host, which is where the loopback restriction belongs;
- also exposes the bridge as a Unix socket on the shared discovery volume
  (`SIMUL_DISCOVERY_DIR`, default `/tmp/simul`), created with
  `socket_mode=0666` so the host user can connect to a socket container
  root owns;
- enables the bridge's raw `execute_script` action, which only gates the
  bridge transport and is not a security boundary;
- keeps the container stateless, so each run starts clean.

simul connects to the **host-published** ports. Override them with
`ISAAC_BRIDGE_PORT` and `ISAAC_VSCODE_PORT` and point the MCP server at the
same values:

```bash
ISAAC_BRIDGE_PORT=8829 ISAAC_VSCODE_PORT=8826 docker compose -f compose.isaac-sim.yml up -d
export ISAAC_SIM__BRIDGE_PORT=8829 ISAAC_SIM__SOCKET_PORT=8826
```

For an Isaac Sim 6.x image, also name the extension that serves the Python
socket:

```bash
ISAAC_SIM_IMAGE=nvcr.io/nvidia/isaac-sim:6.0.1 \
ISAAC_PYTHON_SERVER_EXT=isaacsim.code_editor.python_server \
docker compose -f compose.isaac-sim.yml up -d
```

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| The agent has no Isaac tools; `simul tools` lists Isaac Sim as skipped | The server runs on an OS Isaac Sim does not (macOS), or `ISAAC_SIM__ENABLED=false`; see [Platforms](#platforms) |
| `ping` fails on 6.0 after a plain `isaac-sim.sh` start | Nothing is enabled; restart through `simul isaac launch` |
| Bridge port 8229 never opens | Bridge not published (`install-bridge`) or not enabled (`launch` / `bridge-up`) |
| Old behaviour after upgrading simul | Stale copy in `extsUser`; run `install-bridge --force` |
| Every call is slow, then falls back | Bridge port filtered or half-open; the circuit breaker will route to 8226 |
| `AuthenticationError` from the bridge or 8226 | Isaac was launched with a token the server does not have; set `ISAAC_SIM__SOCKET_AUTH_TOKEN` or let the server read the token file `launch --generate-auth-token` wrote |
| `ScriptTooLargeForSocket` on 5.1 | Script over 32 KiB with the bridge down; enable it with `simul isaac bridge-up` |
| A timeout, yet the stage opened anyway | The budget ran out while Kit kept working; see Timeouts under [Transports](#transports) |
| `foreign_output` in a tool result | Another callback printed while the tool's script waited on frames; the result itself is intact |
| `ModuleNotFoundError: omni.isaac...` in a script on 6.0 | Shims removed; see the namespace migration reference |
