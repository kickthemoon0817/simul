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
| `ISAAC_SIM__ENABLED` | `auto` | Register the Isaac Sim tools: `auto` on Linux and Windows only (elsewhere when `--backends` names `isaac`), `true` on any OS, `false` never; see [isaac-sim.md](isaac-sim.md#platforms) |
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
| `USD__ALLOWED_EXTENSIONS` | `[".usd",".usda",".usdc",".usdz"]` | Extensions the headless USD tools and `simul usd` commands open; load and validate share the list |
| `USD__MAX_FILE_SIZE_MB` | `500` | Largest file they open; a bigger one fails with `FileTooLarge` |
| `SECURITY__RELATIVE_PATH_BASE` | unset | Directory a relative tool path resolves against; unset uses the server's working directory ([File sandbox](#file-sandbox)) |

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

A relative path passed to a tool resolves against
`security.sandbox.relative_path_base` (`SECURITY__RELATIVE_PATH_BASE`). When
it is unset, the path resolves against the directory the server was started
in. Over stdio that is the directory the client launched it from, so
`scenes/robot.usda` names the client's file. This replaces the source
checkout, which an editable install used to resolve against. Relative
`allowed_paths` entries still resolve against the checkout, because they name
its directories. A `SandboxError` reports the path as given (`file_path`),
where it led (`resolved_path`, with symlinks followed) and, for a relative
path, the base it resolved against (`relative_to`).

For the headless USD tools the sandbox also covers everything a file draws
on, not just the file named in the call. Before `load_usd_file` or
`validate_usd_file` composes a stage, it walks the layers the file reaches and
checks every asset path they author against the same roots and URL schemes:
sublayers, references, payloads, value clips and clip manifests, and
asset-valued attributes and metadata such as textures. Paths in unselected
variants count too, and expression variables are evaluated. Each path is
anchored to the layer that names it and resolved the way composition
resolves it. A symlink is followed, and a search-path reference that pxr
would find in the server's working directory is checked there. A path
outside the sandbox fails the call with `SandboxError`;
`details.outside_sandbox` lists each offender's `kind` (`sublayer`,
`reference`, `payload` or `asset`, or `layer` when the post-composition
check caught it), its authored `asset_path`, the
`location` it names, and the `layer` that names it, and `outside_count`
gives the total. Only layers that pass are opened, so a refused file is never
read. After composition, the layers the stage really uses are checked again
as a backstop.

An asset path outside the sandbox is refused even when nothing exists there,
so the error never reveals whether an outside file exists. Textures are held
to the same rule although the headless tools never open them, because pxr
checks that each one exists when it resolves it. A search-path asset that
anchors nowhere, such as `OmniPBR.mdl`, names no location and passes. A
URL passes when its scheme is in `allowed_url_schemes`. To use a file whose
references or textures live elsewhere, add that directory to
`security.allowed_paths`.

The headless USD tools open a URL only through an asset resolver registered
for its scheme. usd-core installs none, so `load_usd_file` refuses an
`omniverse://` URL with `LoadError`, `details.reason: UnsupportedURL` (and
`validate_usd_file` reports that code), although the scheme passes the
sandbox. Copy the asset into the sandbox and pass its local path. Isaac Sim
tools hand such URLs to Isaac Sim, which can resolve them.

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
(`SECURITY__REQUESTS_PER_MINUTE`, default 60, `SECURITY__BURST_SIZE`,
default 10), with an overall per-agent cap across all tools
(`SECURITY__GLOBAL_REQUESTS_PER_MINUTE`, default 600, burst a sixth of it).
The headless USD per-prim queries (`get_prim_info`, `get_mesh_info`,
`get_bounding_box`, `search_prims`) count only against that per-agent cap.
They are local reads an agent issues once per prim while it inspects an
asset, and the per-tool burst refused most of a prim-by-prim pass over a
small robot. Every other tool, including `load_usd_file`,
`validate_usd_file`, `summarize_scene` and the USD edit tools, spends from
both buckets. Raise `SECURITY__REQUESTS_PER_MINUTE` and
`SECURITY__BURST_SIZE`, or set `SECURITY__RATE_LIMITING_ENABLED=false` for a
trusted single-agent setup, if a workflow needs more. An Unreal tool's own argument checks (such as
`capture_unreal_viewport`'s format, `execute_unreal_script`'s mode, the
all-or-none camera groups of `set_unreal_camera_view` and `control_unreal_ui`'s
control and property names) run inside the shared envelope, so a call they
refuse spends a token and is recorded in the usage log like any other failed
call. Arguments that do not match a tool's input schema (a wrong type, a
missing required argument) are refused by the MCP layer before the call
reaches the envelope: no token is spent and nothing is recorded.

### Multi-agent claims

`ISAAC_SIM__ENFORCE_CLAIMS=true` makes `claim_isaac_instance` binding: while
one agent holds a live claim, mutating Isaac tools from other agents fail
with `InstanceClaimed` until the claim is released or expires after 120 s
without activity. The default keeps claims advisory.

A claim, an instance switch and a rate-limit budget belong to an MCP
session. Over stdio the server has one client. Over `--transport http`, a
client that connects with the initialize handshake (MCP protocol 2025-11-25
or earlier) is its own session. A client on the sessionless 2026-07-28
protocol has no session there, and the server cannot tell one such client
from another. Every such client acts as the agent `anonymous`: it uses the
default Isaac instance and finds every live claim held by someone else.
`set_active_isaac_instance`, `claim_isaac_instance` and
`release_isaac_instance` refuse it with `SessionRequired`, and all such
clients share one rate-limit budget. Connect agents that need their own
instance or claim with the handshake (for example `fastmcp.Client(url,
mode="legacy")`) or over stdio.

## Usage statistics

Every tool call is recorded in a persistent JSONL log. Agents read it with
`get_tool_usage_stats`; operators read it with `simul stats` and clear it
with `simul stats --reset`.
