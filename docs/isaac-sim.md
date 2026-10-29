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
- The python_server can require a token. Start Isaac with
  `simul isaac launch --auth-token <secret>` and give the MCP server
  `ISAAC_SIM__SOCKET_AUTH_TOKEN=<secret>`. Alternatively
  `--generate-auth-token` creates one and writes it to the discovery
  directory, where the server finds it for the newest running editor.
- `isaacsim.core.api`, `isaacsim.core.prims` and `isaacsim.core.utils` are
  deprecated in favour of `isaacsim.core.experimental.*`, and the
  `omni.isaac.*` shims are gone. Read
  [`skills/isaac-scripting/references/namespace-migration.md`](../skills/isaac-scripting/references/namespace-migration.md)
  before writing `execute_isaac_script` code.

## Transports

| Port | Transport | Notes |
|---|---|---|
| 8229 | `khemoo.simul` bridge | Preferred: typed protocol, fewer round trips, script interruption, busy state |
| 8226 | Stock Python socket | `isaacsim.code_editor.vscode` on 5.x, `isaacsim.code_editor.python_server` on 6.0; used when the bridge is unavailable |

- **Timeouts.** Every script carries a server-side execution timeout derived
  from `ISAAC_SIM__SOCKET_TIMEOUT` (one second under it, minimum 1 s). The
  bridge interrupts an overrunning script inside Kit; the 6.0 python_server
  reports the overrun in its response.
- **Interrupting.** `interrupt_isaac_script` / `simul isaac interrupt`
  stops a running script when it is a coroutine or suspended at an `await`. A
  synchronous loop that never yields also blocks the bridge's event loop, so
  only the timeout reaches it; a blocking C call is interrupted when it
  returns to Python.
- **Busy or hung?** `get_isaac_runtime_info` reports `bridge.busy`,
  `bridge.busy_since` and `bridge.current_action`.
- **Circuit breaker.** A filtered or half-open bridge port would cost the full
  timeout on every call. After `ISAAC_SIM__BRIDGE_FAILURE_THRESHOLD`
  consecutive failures (default 3) the client skips the bridge for
  `ISAAC_SIM__BRIDGE_COOLDOWN_SECONDS` (default 30) and uses the stock socket.
  `ping_isaac`, `list_isaac_instances` and `get_isaac_runtime_info` report
  this as `bridge_circuit_open`.
- **Raw scripts over the bridge.** The bridge's `allow_unsafe_execution`
  setting (`simul isaac bridge-set-unsafe`) only gates raw scripts on
  8229. It is not a security boundary; to remove scripting from agents use
  `SECURITY__ALLOW_SCRIPT_EXECUTION=false`
  ([configuration.md](configuration.md#script-execution)).

## Multiple instances

Each bridge writes a discovery file to `isaac_sim.discovery_dir` (default
`/tmp/simul`) with its bridge port and Python socket port. Agents list
them with `list_isaac_instances` and switch with `set_active_isaac_instance`.
`claim_isaac_instance` / `release_isaac_instance` mark an instance as in use;
claims are advisory unless `ISAAC_SIM__ENFORCE_CLAIMS=true`. The switch and
the claim belong to the caller's MCP session, which an HTTP client on the
sessionless 2026-07-28 protocol does not have
([configuration.md](configuration.md#multi-agent-claims)).

## Remote hosts

Cross-host Isaac Sim is not supported. The server trusts only loopback
addresses in discovery files, `launch` binds the transports on the local
machine, and neither transport authenticates beyond the optional
python_server token: anything that can reach the ports can run Python inside
Isaac Sim. Run `simul` on the same host as Isaac, or for a container,
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
  (`SIMUL_DISCOVERY_DIR`, default `/tmp/simul`);
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
| `ModuleNotFoundError: omni.isaac...` in a script on 6.0 | Shims removed; see the namespace migration reference |
