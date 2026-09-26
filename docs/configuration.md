# Configuration

simul reads one YAML file and then applies environment overrides on top of
it. The environment always wins.

## Where settings come from

1. **YAML file.** By default the `default.yaml` shipped inside the package,
   [`src/simul/resources/config/default.yaml`](../src/simul/resources/config/default.yaml).
   Point at your own copy with `simul server --config FILE` or the
   `CONFIG_FILE` environment variable. Copy the packaged file as a starting
   point; `simul validate-config FILE` checks it.
2. **`.env` in the working directory**, then the process environment.

The packaged YAML is the reference for every key and its default. This page
covers the model and the settings people change most; it does not repeat the
whole file.

## Environment variables: `SECTION__KEY`

Each setting is named `<SECTION>__<FIELD>` with a double underscore, for
example `LOGGING__LEVEL=DEBUG` or `ISAAC_SIM__BRIDGE_PORT=8829`.

Nested YAML blocks flatten into a single field name, so `usd.cache.enabled`
in YAML is `USD__CACHE_ENABLED` in the environment, and `isaac_sim.bridge.port`
is `ISAAC_SIM__BRIDGE_PORT`. [`.env.example`](../.env.example) lists every
variable by its exact name and is the quickest way to find one.

Sections: `SERVER`, `ISAAC_SIM`, `BLENDER`, `UNREAL`, `USD`, `VIEWPORT`,
`LOGGING`, `SECURITY`.

`ISAAC_SIM_PATH` (single underscore, no section) is the Isaac Sim install root
used by `simul isaac launch` and `install-bridge`; the YAML default for
`isaac_sim.path` reads it.

## Common settings

| Variable | Default | Purpose |
|---|---|---|
| `LOGGING__LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `LOGGING__FILE_PATH` | `~/.simul/logs/simul.log` | Text log (rotated daily, 14 days kept) |
| `LOGGING__AUDIT_PATH` | `~/.simul/logs/audit.jsonl` | One JSON line per tool call |
| `ISAAC_SIM__SOCKET_PORT` / `ISAAC_SIM__BRIDGE_PORT` | `8226` / `8229` | Isaac Sim transports |
| `ISAAC_SIM__SOCKET_PROTOCOL` | `auto` | `python_server` (6.0+) or `vscode` (5.x) skips the one-time probe |
| `ISAAC_SIM__SOCKET_AUTH_TOKEN` | unset | Token for a 6.0 python_server started with `--auth-token` |
| `ISAAC_SIM__BRIDGE_FAILURE_THRESHOLD` / `ISAAC_SIM__BRIDGE_COOLDOWN_SECONDS` | `3` / `30` | Bridge circuit breaker; see [isaac-sim.md](isaac-sim.md#transports) |
| `UNREAL__HOST` / `UNREAL__PORT` | `localhost` / `30010` | Remote Control endpoint |
| `UNREAL__TOOL_SURFACE` | `thin` | `full` registers every granular Unreal tool (`--unreal-tools` overrides) |
| `BLENDER__TOOL_SURFACE` | `full` | `thin` registers only the essentials in `THIN_BLENDER_TOOLS` (`--blender-tools` overrides) |
| `UNREAL__MODE` | `endpoint` | `attached` uses the editor chosen by `simul unreal attach` |
| `UNREAL__PASSPHRASE` | unset | Plaintext or MD5 hex for an editor set up with `--passphrase` |
| `BLENDER__MODE` | `embedded` | `attached` uses the window chosen by `simul blender attach` |
| `VIEWPORT__MAX_SIZE` | `2048` | Largest capture dimension |

`simul logs paths` prints the log locations the current settings resolve
to, and `simul logs tail --follow` streams them.

## Security

### Script execution

`SECURITY__ALLOW_SCRIPT_EXECUTION` (YAML `security.allow_script_execution`,
default `true`). When `false`, the server does not register
`execute_isaac_script`, `execute_unreal_script`, `execute_blender_script`, or
Unreal's `call_unreal_actor_function` and `batch_unreal_operations`
dispatchers, and `simul isaac exec` / `simul unreal exec` return a
`ScriptExecutionDisabled` error. The fixed granular tools, including
`control_unreal_ui`, keep working.

This is the only switch that removes the agent-authored code surface. The
Isaac bridge extension's `allow_unsafe_execution` setting (and
`simul isaac bridge-set-unsafe`) only gates raw scripts sent over the
bridge on port 8229. Raw scripts and every generated tool script still run
over the stock Python socket on 8226, so that flag is not a security
boundary.

### File sandbox

MCP tools that read or write files only accept paths under
`security.sandbox.allowed_paths` (default `examples`, `tests/data`,
`/tmp/simul-work`; relative entries resolve against a source checkout and are
dropped for a wheel install). URL-shaped paths are allowed only for the
schemes in `allowed_url_schemes` (reads, default `omniverse`) and
`allowed_write_url_schemes` (writes, default none). Set
`SECURITY__ALLOWED_PATHS='["/abs/project","/tmp/simul-work"]'` to widen it.
The `simul` CLI is operated by you and is not sandboxed.

### Refusals

Some operations refuse by default to protect the running application and
your data. Each refusal is a structured `RefusedOperation` error:

- `delete_isaac_prim` refuses `/`, and refuses `/World` unless
  `allow_root_delete=true`.
- `disable_isaac_extension` refuses the transport extensions
  (`khemoo.simul`, `isaacsim.code_editor.python_server`,
  `isaacsim.code_editor.vscode`).
- `set_isaac_carb_settings` refuses keys under `/exts/khemoo.simul/` and
  `/exts/isaacsim.code_editor.python_server/`.
- `save_isaac_stage` needs `overwrite=true` to replace an existing file.
- `new_isaac_stage` and `open_isaac_stage` need `discard_unsaved=true` when
  the current stage has unsaved edits.

### Rate limiting

`security.rate_limiting` limits calls per agent session and per tool
(`SECURITY__REQUESTS_PER_MINUTE`, `SECURITY__BURST_SIZE`), with an overall
per-agent cap across all tools.

### Multi-agent claims

`ISAAC_SIM__ENFORCE_CLAIMS=true` makes `claim_isaac_instance` binding: while
one agent holds a live claim, mutating Isaac tools from other agents fail
with `InstanceClaimed` until the claim is released or expires after 120 s
without activity. The default keeps claims advisory.

## Usage statistics

Every tool call is recorded in a persistent JSONL log. Agents read it with
`get_tool_usage_stats`; operators read it with `simul stats` and clear it
with `simul stats --reset`.
