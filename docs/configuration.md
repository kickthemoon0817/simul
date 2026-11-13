# Configuration

simul reads one YAML file and then applies environment overrides on top of
it. The environment always wins.

## Where settings come from

1. **YAML file.** By default the `default.yaml` shipped inside the package,
   [`src/simul/resources/config/default.yaml`](../src/simul/resources/config/default.yaml).
   Point at your own copy with `simul server --config FILE` or the
   `CONFIG_FILE` environment variable. Copy the packaged file as a starting
   point; `simul validate-config FILE` checks it.
2. **The `.env` file `SIMUL_ENV_FILE` names**, if any, then the process
   environment.

simul never reads a `.env` from the working directory. MCP clients start the
server in your project, so a cloned repository's `.env` could otherwise widen
the file sandbox, re-enable script execution or move the HTTP bind. To keep
settings in a file, name it explicitly, for example
`SIMUL_ENV_FILE=~/.config/simul/simul.env` in the MCP client's server entry;
a name that points at no file stops the command. When the working directory
has a `.env` that sets simul variables, simul logs a warning naming them and
ignores the file.

The packaged YAML is the reference for every key and its default. This page
covers the model and the settings people change most; it does not repeat the
whole file.

## When the configuration is wrong

A configuration that cannot be used as written stops every command with
`error_type: ConfigError` and exit code 2, before anything is started or
contacted. The message names each setting, the value, where the value came
from (`UNREAL__PORT`, the env file or the YAML file) and why it was refused.
Under `--json` the envelope's `details.errors` carries one
`{setting, source, value, reason}` entry per problem. `simul server` writes
the message to stderr only, because over stdio its stdout is the MCP channel.
The command stops when:

- the YAML does not parse, is not a mapping of sections, or a section or group
  has the wrong shape (`security: strict`);
- `--config FILE` or `CONFIG_FILE` names a file that does not exist (only the
  packaged default is allowed to be missing, with a warning);
- a value fails validation, from any source: `UNREAL__PORT=70000`, or
  `SECURITY__ALLOWED_PATHS=/a,/b` (list and mapping settings take JSON:
  `'["/a","/b"]'`);
- a key or variable under `security` names no setting, such as
  `allow_scripts_execution` or `SECURITY__RATE_LIMITING__REQUESTS_PER_MINUTE`.
  A typo there would otherwise leave the protection it names off, so it is an
  error and the message suggests the setting it most likely meant.

An unknown key or variable in any other section only warns, once, and is
ignored. The warning says what it most likely meant (`unreal.portt` → did you
mean `unreal.port`?). Keys that earlier releases shipped and nothing reads
any more (`server.cors_origins`, `isaac_sim.kit`, the old `mesh` and
`performance` sections and so on) load without a warning, so a file copied
from an old `default.yaml` keeps working.

`simul validate-config FILE` is stricter. Every unknown key in the file is an
error, with a suggestion; a value the settings refuse is an error; retired
keys and unknown non-security `SECTION__` variables in the environment are
warnings. It exits 0 when the file is valid and 1 when it is not, with
`valid`, `errors` and `warnings` in the JSON. A file that does not exist exits
2.

## Environment variables: `SECTION__KEY`

Each setting is named `<SECTION>__<FIELD>` with a double underscore, for
example `LOGGING__LEVEL=DEBUG` or `ISAAC_SIM__BRIDGE_PORT=8829`.

Nested YAML blocks flatten into a single field name, so `usd.cache.enabled`
in YAML is `USD__CACHE_ENABLED` in the environment, and `isaac_sim.bridge.port`
is `ISAAC_SIM__BRIDGE_PORT`. Spelling the nesting out
(`ISAAC_SIM__BRIDGE__PORT`) names no setting; simul warns and suggests the
flat name. [`.env.example`](../.env.example) lists every variable by its
exact name and is the quickest way to find one.

Sections: `SERVER`, `ISAAC_SIM`, `BLENDER`, `UNREAL`, `USD`, `VIEWPORT`,
`LOGGING`, `SECURITY`. A variable named after a bare section (`SERVER=prod`,
`LOGGING=1`, ordinary names in many shells and CI systems) is ignored unless
its value is a JSON object, which then sets that section's fields
(`SECURITY='{"allow_script_execution": false}'`).

`ISAAC_SIM_PATH` (single underscore, no section) is the Isaac Sim install root
used by `simul isaac launch` and `install-bridge`; the YAML default for
`isaac_sim.path` reads it.

## State directory: `SIMUL_HOME`

simul keeps its state in one directory: `$SIMUL_HOME` when set, otherwise
`~/.simul`. Each of these defaults to a path under it and moves with it,
unless you configure that path yourself:

| What | Path under the state directory | Setting |
|---|---|---|
| Text log, error and debug logs | `logs/simul.log`, `logs/simul_errors.log`, ... | `LOGGING__FILE_PATH` |
| Audit stream | `logs/audit.jsonl` | `LOGGING__AUDIT_PATH` |
| Tool usage log (`simul stats`, `get_tool_usage_stats`) | `logs/tool_usage.jsonl` | none |
| Isaac Sim session and claim records | `sessions/<port>.json` | none |
| Blender discovery files and the attachment | `blender/`, `blender/attachment.json` | `BLENDER__DISCOVERY_DIR`, `BLENDER__ATTACHMENT_PATH` |
| `simul blender install-bridge` ZIP | `blender/simul_blender_bridge.zip` | `--output` |
| Unreal attachment | `unreal/attachment.json` | `UNREAL__ATTACHMENT_PATH` |
| FastMCP's data directory | `fastmcp/` | `FASTMCP_HOME` |

Directories are created when something is first written there, with mode
`0700`; a directory that already exists keeps its mode (it may be one you
share, such as `SIMUL_HOME=$HOME`). Commands that only read (`simul tools`, `info`, `validate-config`,
`logs paths`) create nothing, and a server whose home directory is read-only
still starts: point `SIMUL_HOME` at a writable directory and everything lands
there. The variable is read when a process starts, so set it in the MCP
client's server entry or your shell, not in a config file. A YAML value may
use `${SIMUL_HOME}`, which expands to the state directory whether or not the
variable is set. The Blender add-on finds discovery files through the same
variable, so Blender must see the same `SIMUL_HOME` as the server (or set
`SIMUL_BLENDER_DISCOVERY_DIR` in Blender's environment and
`BLENDER__DISCOVERY_DIR` for the server).

## Common settings

| Variable | Default | Purpose |
|---|---|---|
| `LOGGING__LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` for simul's loggers and the console; `--log-level` sets it per run. Third-party loggers keep their quieter level unless this one is stricter |
| `LOGGING__FORMAT` | `detailed` | Console and text log format: `detailed`, `simple` (`LEVEL - logger - message`) or `json` (one object per line: `ts`, `level`, `logger`, `request_id`, `tool`, `msg`, ...) |
| `LOGGING__FILE_PATH` | `$SIMUL_HOME/logs/simul.log` (`~/.simul/...`) | Text log (rotated daily, `LOGGING__RETENTION_DAYS` days kept; shared by every server on the machine) |
| `LOGGING__AUDIT_PATH` | `$SIMUL_HOME/logs/audit.jsonl` (`~/.simul/...`) | One JSON line per tool call (see [Audit log](#audit-log)) |
| `ISAAC_SIM__ENABLED` | `auto` | Register the Isaac Sim tools: `auto` on Linux and Windows only (elsewhere when `--backends` names `isaac`), `true` on any OS, `false` never; see [isaac-sim.md](isaac-sim.md#platforms) |
| `ISAAC_SIM__SOCKET_PORT` / `ISAAC_SIM__BRIDGE_PORT` | `8226` / `8229` | Isaac Sim transports |
| `ISAAC_SIM__SOCKET_PROTOCOL` | `auto` | `python_server` (6.0+) or `vscode` (5.x) skips the one-time probe |
| `ISAAC_SIM__SOCKET_AUTH_TOKEN` | unset | Token for a 6.0 python_server started with `--auth-token`; also sent to the bridge, which then requires it ([security model](isaac-sim.md#security-model)) |
| `ISAAC_SIM__BRIDGE_FAILURE_THRESHOLD` / `ISAAC_SIM__BRIDGE_COOLDOWN_SECONDS` | `3` / `30` | Bridge circuit breaker; see [isaac-sim.md](isaac-sim.md#transports) |
| `UNREAL__HOST` / `UNREAL__PORT` | `localhost` / `30010` | Remote Control endpoint |
| `UNREAL__TOOL_SURFACE` | `thin` | `full` registers every granular Unreal tool (`--unreal-tools` overrides) |
| `BLENDER__TOOL_SURFACE` | `full` | `thin` registers only the essentials in `THIN_BLENDER_TOOLS` (`--blender-tools` overrides) |
| `UNREAL__MODE` | `endpoint` | `attached` uses the editor chosen by `simul unreal attach` |
| `UNREAL__PASSPHRASE` | unset | Plaintext or MD5 hex for an editor set up with `--passphrase` |
| `UNREAL__MAX_ACTORS` | `200` | Default `max_results` of `list_unreal_actors` and `simul unreal list-actors` |
| `BLENDER__MODE` | `embedded` | `attached` uses the window chosen by `simul blender attach` |
| `VIEWPORT__MAX_SIZE` | `2048` | Largest capture dimension |
| `USD__ALLOWED_EXTENSIONS` | `[".usd",".usda",".usdc",".usdz"]` | Extensions the headless USD tools and `simul usd` commands open; load and validate share the list |
| `USD__MAX_FILE_SIZE_MB` | `500` | Largest file they open; a bigger one fails with `FileTooLarge` |
| `USD__VALIDATION_TIMEOUT_SECONDS` | `300` | Longest `validate_usd_file` may take when it checks a file in a separate process (a loaded stage holds an older copy of a file it uses); past it the call fails with `TimeoutError` |
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
`ScriptExecutionDisabled` error. The server's instructions and the other
tools' descriptions then no longer point at the script tools. The fixed
granular tools, including `control_unreal_ui`, keep working. The Isaac prim
tools refuse the writes that would plant code instead
(`ScriptExecutionDisabled`): code-carrying attributes such as
`inputs:script*` and `inputs:codeStr`, a `node:type` that runs code
(ScriptNode, omni.warp's WarpKernel, Replicator's OgnAugment), any write to
such a node, creating one and duplicating a subtree that holds one.

Isaac Sim also runs Python that an OmniGraph node carries as data:
`omni.graph.scriptnode.ScriptNode` runs its `inputs:script`,
`omni.warp.WarpKernel` imports its `inputs:codeStr`, and Replicator's
augmentation and condition nodes run their `*Script` inputs. Both
`omni.graph.scriptnode` and `omni.warp` are enabled in Isaac Sim's own
`.kit`. With the switch off, these tools refuse with `ScriptExecutionDisabled`:

- `create_isaac_graph_node` refuses those node types and any type whose
  name has a word starting with `script`.
- `set_isaac_graph_node_values` refuses code attributes (a name with a word
  starting with `script` or `code`) and any write to a script-capable node.
- `connect_isaac_graph_nodes` refuses a connection into a code attribute or
  into a script-capable node.
- `import_isaac_asset` and `add_isaac_reference` refuse an asset whose
  composed content holds such a node or a Python Scripting Component (below),
  or holds payloads that are not loaded, since the check cannot see inside
  them.
- `enable_isaac_extension` refuses `omni.graph.scriptnode`,
  `omni.kit.scripting`, `omni.warp` and any extension whose name has a word
  starting with `script` (so 6.0's `omni.behavior.scripting.*` too).

Isaac Sim's **Python Scripting Component** runs Python files attached to
prims: a prim with the `OmniScriptingAPI` schema lists them in
`omni:scripting:scripts`. Isaac Sim 5.1's full app enables
`omni.kit.scripting`, and 6.0 enables `omni.behavior.scripting.bundle`,
which reads the same attribute. Once a stage without scripts has been opened,
the script manager allows scripts. Every later USD change that adds such a
prim then loads its scripts with no prompt. Setting
`/app/scripting/ignoreWarningDialog` also skips the prompt shown when a stage
that has scripts is opened. With the switch off:

- `import_isaac_asset`, `add_isaac_reference` and `duplicate_isaac_prim`
  refuse content that holds a prim with `OmniScriptingAPI` or any
  `omni:scripting:*` property;
- `set_isaac_prim_attribute`, `set_isaac_graph_node_values` and
  `create_isaac_prim` refuse `omni:scripting:*` attributes and the
  `OmniScriptingAPI` name;
- `set_isaac_carb_settings` refuses `/app/scripting` and its ancestors.

Names are split into words at `.`, `:`, `_`, `-` and camelCase boundaries,
so `inputs:codeStr` and `inputs:augmentationScript` count as code
attributes while `inputs:usdTimecode` does not, and
`omni.isaac.robot_description_editor` is not a script extension. A word that
contains `script` anywhere (an all-lowercase third-party name such as
`acme.nodes.runpythonscript` or `inputs:pythonscript`) also counts, unless
the `script` belongs to an English word such as `description` or `transcript`.
The other `omni.warp` nodes, which run fixed kernels, stay usable.

The prim-editing tools apply the same check, since OmniGraph nodes are prims:
`create_isaac_prim` refuses a code-running type and a code attribute or
code-running `node:type` among its attributes, `set_isaac_prim_attribute`
refuses a code attribute, a code-running `node:type` value and any write to
a code-running node, and `duplicate_isaac_prim` refuses a subtree that holds
one.

Routes this does not close:
- A stage opened with `open_isaac_stage` that already contains code nodes or
  scripting components. The Kit prompt guards those scripts, but a headless
  editor has no one to answer it, and `ignoreWarningDialog` set at launch
  removes it.
- Nodes and scripts already on the stage, which run when the graph evaluates
  or when Kit allows scripts.
- An all-lowercase name that hides `code` rather than `script`
  (`inputs:pythoncode`): matching that would also hit `timecode`, `barcode`
  and `encode`.
- 6.0's `omni.behavior.scripting` trust logic, whose source is not public;
  the guard relies on the attribute and setting its Isaac Sim 6.0.1 callers
  use.

Treat the switch as removing the agent's direct code tools, not as a sandbox
for an untrusted stage.

The switch is the one that removes the agent-authored code tools. The
Isaac bridge extension's `allow_unsafe_execution` setting (and
`simul isaac bridge-set-unsafe`) only gates raw scripts sent over the
bridge on port 8229. The bridge refuses such a script without running it,
and simul then sends it, like every generated tool script, to the stock
Python socket on 8226 when that socket is enabled, so that flag is not a
security boundary.

### Network transports (HTTP and SSE)

Over stdio (the default) only the MCP client that started the server can talk
to it. `simul server --transport http` (or `sse`) opens a listener on
`server.host`:`server.port`, and every tool, `execute_*_script` included, is
then one request away from whoever can reach it. Three callers are
considered:

- **A web page you open.** A page can send requests to
  `http://127.0.0.1:8765`, and through DNS rebinding (its own host name
  re-pointed at `127.0.0.1`) read the answers. The Host/Origin guard refuses
  a request whose `Host` header is not `localhost`, `127.0.0.1`, `::1`, the
  bound address or an entry of `server.allowed_hosts` (HTTP 421), and a
  browser request whose `Origin` is not same-origin, loopback or an entry of
  `server.allowed_origins` (HTTP 403). MCP clients send no `Origin` and are
  unaffected. The guard is on by default; `SERVER__HOST_ORIGIN_PROTECTION=false`
  removes it and the server warns at startup.
- **Another machine.** A `server.host` that is not loopback (`0.0.0.0`, a LAN
  address, a host name) needs `simul server --allow-public` or
  `SERVER__ALLOW_PUBLIC=true`; without it the server exits 2 with
  `InsecureTransport` before it listens. A public bind without a bearer token
  starts with a red warning: the guard checks headers, it does not
  authenticate anyone. Clients that reach the server by a DNS name need that
  name in `SERVER__ALLOWED_HOSTS='["simul.lan"]'`.
- **Another local user.** Loopback is shared by every account on the machine.
  A bearer token makes every request carry `Authorization: Bearer <token>`
  (HTTP 401 otherwise).

| Variable | Default | Purpose |
|---|---|---|
| `SERVER__HOST` / `SERVER__PORT` | `localhost` / `8765` | Listener address |
| `SERVER__ALLOW_PUBLIC` | `false` | Permit a non-loopback `SERVER__HOST` (`--allow-public`) |
| `SERVER__HOST_ORIGIN_PROTECTION` | `true` | The DNS-rebinding guard described above |
| `SERVER__ALLOWED_HOSTS` | `[]` | Extra `Host` names (port ignored, `*` wildcards) |
| `SERVER__ALLOWED_ORIGINS` | `[]` | Extra browser origins, `scheme://host[:port]` |
| `SERVER__AUTH_TOKEN_FILE` | unset | File whose first line is the bearer token; must be `chmod 600` |
| `SERVER__AUTH_TOKEN` | unset | The token itself; prefer the file, which keeps it out of the environment |

A token has at least 16 characters. Create one and point the server at it:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))" > ~/.config/simul/server-token
chmod 600 ~/.config/simul/server-token
SERVER__AUTH_TOKEN_FILE=~/.config/simul/server-token simul server --transport http
```

Clients then send the header, for example
`claude mcp add --transport http simul http://localhost:8765/mcp --header "Authorization: Bearer <token>"`.
Every client presenting the token is the same authenticated user, so clients
on the sessionless protocol share one session (see
[Multi-agent claims](#multi-agent-claims)). The token, the bind host and the
other settings here cannot come from a working-directory `.env`; see
[Where settings come from](#where-settings-come-from).

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
made absolute without following symlinks (`absolute_path`), and for a
relative path the base it resolved against (`relative_to`). When the path
sits inside an allowed root, it also reports where it leads with symlinks
followed (`resolved_path`), so a link inside the sandbox that points out of
it is visible. A path outside the sandbox is never followed, so a refusal
cannot reveal where symlinks elsewhere on disk point.

An allowed root inside a world-writable directory, such as the default
`/tmp/simul-work`, must belong to you. Any local user can create a directory
in `/tmp` first and read or plant what agents write there, so before the
first write simul creates a missing root with mode `0700`, and it refuses a
root there (and every path under it) that another user owns, that group or
others can write, or whose world-writable parent lacks the sticky bit. The
refusal is a `SandboxError` whose `details` carry `untrusted_root` and
`reason`. Roots elsewhere are not inspected. If Isaac Sim or another engine
runs as a different user and has to write into the root, configure a
directory you share with it on purpose instead.

Every allowed root admits reads and writes alike, so in a source checkout
`examples/` and `tests/data/` are writable by the tools too.

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
`details.outside_sandbox` lists each offender's `kind`, its authored
`asset_path`, the `location` it names, and the `layer` that names it, and
`outside_count` gives the total. The kinds are `sublayer`, `reference`,
`payload`, `clip` (a value clip, clip manifest or clip template a clip set
authors), `asset` (a texture, an `assetInfo` identifier or any other
asset-valued attribute or metadata), and `layer` when the post-composition
check caught it. Only layers that pass are opened, so a refused file is never read.
After composition, the layers the stage uses by then are checked again as a
backstop, judged afresh so a file swapped on disk between the two passes is
caught. That check cannot see value clips: when a clip set authors a
`manifestAssetPath`, pxr opens its clip layers only on the first read of a
time-varying value, after the load. Clips therefore rely on the first pass,
which checks every clip and manifest path before anything opens. Each load
resolves a path named from many layers once and checks a shared root's
trust once, so the walk adds a fraction of the stack's own load time.

An asset path outside the sandbox is refused even when nothing exists there,
so the error never reveals whether an outside file exists. Textures are held
to the same rule although the headless tools never open them, because pxr
checks that each one exists when it resolves it. A search-path asset that
anchors nowhere, such as `OmniPBR.mdl`, names no location and passes. A
URL passes when its scheme is in `allowed_url_schemes`. To use a file whose
references or textures live elsewhere, add that directory to
`security.allowed_paths`.

An Unreal content path is not a file, so it passes. Unreal's USD exporter
writes each mesh's source material as
`info:unreal:sourceAsset = @/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial@`,
an object in the editor's content that Unreal's importer loads by name. A
value passes this way only when all of these hold:

- it is an asset value (kind `asset`), never a sublayer, reference,
  payload or clip;
- it sits under `/Engine`, `/Game` or `/Script` and ends in
  `<Name>.<Name>`, where the suffix is no file extension;
- nothing exists at that path on disk.

`load_usd_file` lists each such path in `warnings` as `UnrealContentPath`,
with the layer that names it. With the sandbox off nothing is checked, so
nothing is listed. `validate_usd_file` reports pxr's
`UnresolvableDependency` and `MissingSourceTypeInRegistry` about exactly
those paths as warnings rather than errors, so an Unreal export inside the
sandbox loads and validates. `@/Engine/x.usda@`, a relative path and another
content root (a plugin's) are held to the sandbox like any other path. So is
an `assetInfo` identifier, which in a copied export still names the folder
Unreal wrote it to. Unreal's own USD import (`import_unreal_usd`) applies
the same rule.

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
  `allow_root_delete=true`. A prim from a referenced asset is deactivated
  rather than removed (`mode="remove"` refuses it instead).
- `disable_isaac_extension` refuses the transport extensions
  (`khemoo.simul`, `isaacsim.code_editor.python_server`,
  `isaacsim.code_editor.vscode`).
- `set_isaac_carb_settings` refuses keys under `/exts/khemoo.simul/` and
  `/exts/isaacsim.code_editor.python_server/`.
- `save_isaac_stage` needs `overwrite=true` to replace an existing file, and
  for any URL target (a remote file's existence cannot be checked first). It
  refuses while the timeline is playing or paused unless
  `allow_while_playing=true`, since the stage then holds simulated poses. An
  in-place save checks the stage's file and every dirty layer it writes
  against the write sandbox.
- `new_isaac_stage` and `open_isaac_stage` need `discard_unsaved=true` when
  the current stage has unsaved edits.
- `delete_prim` (headless USD) refuses `/`, and refuses the stage's default
  prim unless `allow_root_delete=true`.
- `create_prim` (headless USD) refuses a path that already holds a prim
  unless `redefine_existing=true`.
- `save_usd_stage` refuses a file that changed on disk after the stage read
  it unless `overwrite_external_changes=true`; `export_usd_stage` needs
  `overwrite=true` to replace an existing file.
- `reload_usd_stage` and `unload_usd_stage` need `discard_unsaved=true` when
  the stage has unsaved edits.

### Rate limiting

`security.rate_limiting` limits calls per agent session and per tool
(`SECURITY__REQUESTS_PER_MINUTE`, default 60, `SECURITY__BURST_SIZE`,
default 10), with an overall per-agent cap across all tools
(`SECURITY__GLOBAL_REQUESTS_PER_MINUTE`, default 600, burst a sixth of it).
The headless USD per-prim queries (`get_prim_info`, `get_mesh_info`,
`get_bounding_box`, `search_prims`) count only against that per-agent cap.
They are local reads an agent issues once per prim while it inspects an
asset, and the per-tool burst refused most of a prim-by-prim pass over a
small robot. `get_prim_subtree` spends from its own bucket for a first page
only: each later page (a nonzero `offset`) counts only against the
per-agent cap, since a walk over a large stage is one request in pages, and
25 pages used to draw 15 refusals. Every other tool, including
`load_usd_file`, `validate_usd_file`, `summarize_scene` and the USD edit
tools, spends from both buckets.

The defaults are meant to stop a runaway loop, not parallel work: one tool
takes `burst_size` (10) calls at once and then `requests_per_minute` (60, one
a second), and the agent's calls across every tool take
`global_requests_per_minute` (600) with a burst of ten seconds' worth (100).
An agent that fans one tool out in parallel (`get_isaac_prim_detail` on 50
prims at once) gets 10 answers and 40 `RateLimitError`s, each with
`retry_after_seconds`. For such agents raise the per-tool burst, for example
`SECURITY__BURST_SIZE=50` (the per-agent ceiling still applies), or turn the
limit off with `SECURITY__RATE_LIMITING_ENABLED=false` when the server has a
single trusted client. Over HTTP every sessionless client has its own budget
(see [Multi-agent claims](#multi-agent-claims)). An Unreal tool's own argument checks (such as
`capture_unreal_viewport`'s format, `execute_unreal_script`'s mode, the
all-or-none camera groups of `set_unreal_camera_view` and `control_unreal_ui`'s
control and property names) run inside the shared envelope, so a call they
refuse spends a token and is recorded in the usage log like any other failed
call. So do arguments that do not match a tool's input schema (a wrong type,
a missing or unexpected argument, a value outside the schema's
`minimum`/`maximum` or `enum`): FastMCP refuses them before the tool runs, and
the server answers with the same `ValidationError` envelope, sent with
`isError: true`, as every other refusal ([tools.md](tools.md#errors)), after
spending a token and recording the call. The server's own tools (the Isaac
instance and claim tools, `get_tool_usage_stats`) and an Isaac path the
sandbox refuses go through the envelope too.

### Multi-agent claims

`ISAAC_SIM__ENFORCE_CLAIMS=true` makes `claim_isaac_instance` binding: while
one agent holds a live claim, mutating Isaac tools from other agents fail
with `InstanceClaimed` until the claim is released or expires after 120 s
without activity. The default keeps claims advisory.

Claims are stored in `~/.simul/sessions/<port>.json`, one file per instance
port, shared by every simul server on the machine. Each change holds a
cross-process lock (`<port>.lock`) and replaces the file atomically, so
concurrent claims from several servers are all kept, and a claim check never
sees a half-written file.

- **A claim that cannot be saved fails.** `claim_isaac_instance` and
  `set_active_isaac_instance` return `ClaimStoreError` when the session
  directory cannot be written; they never report a claim that was not saved.
- **Enforcement fails closed.** With claims enforced, records that cannot be
  read refuse mutating Isaac tools with `ClaimStoreError`. A record file that
  does not parse is repaired, with a warning in the server log: the first
  complete set of claims it holds is kept, or the file is moved aside to
  `<port>.json.corrupt-<ms>`. For the next 120 s mutating tools are refused
  with `InstanceClaimed` for every agent that cannot show a claim of its own,
  because a lost claim may still be live; an agent whose session holds a
  claim on the instance re-asserts it on its next call. Read-only tools are
  never blocked.
- **Reclaiming after a reconnect or a restart.** A claim is bound to the MCP
  session that made it. When that session ends, or the server process that
  wrote the claim exits, the claim is *orphaned*: it still blocks other agents
  until it expires, so the agent that is reconnecting keeps its instance, but
  `claim_isaac_instance` with the same `agent_id` takes it over at once, and
  `release_isaac_instance` with that `agent_id` releases it. `InstanceClaimed`
  says so (`details.holder_orphaned`), and `list_isaac_instances` marks such
  records `orphaned`. Use an explicit `agent_id` if you want to reclaim: the
  default label is derived from the session and changes with it.
- **Operator view.** `simul isaac claims` lists the live claims (`--port`,
  `--agent` filter them), and `simul isaac claims --clear` removes them, for
  example a claim left by a server that crashed. It reports a damaged or
  unreadable record file too.

A claim, an instance switch and a rate-limit budget belong to an MCP
session. Over stdio the server has one client. Over `--transport http`, a
client that connects with the initialize handshake (MCP protocol 2025-11-25
or earlier) is its own session. A client on the sessionless 2026-07-28
protocol has no session there, and the server cannot tell one such client
from another. Every such client acts as the agent `anonymous`: it uses the
default Isaac instance and finds every live claim held by someone else.
`set_active_isaac_instance`, `claim_isaac_instance` and
`release_isaac_instance` refuse it with `SessionRequired`. For its rate-limit
budget and in the usage and audit logs it is named by its TCP peer,
`anonymous@<host>:<port>`, which the network assigns and a client keeps while
its connection stays open, so each client has its own per-tool budget (it
used to be one budget for all of them). The ceiling across every tool
(`global_requests_per_minute`) is kept per host, `anonymous@<host>`, so a
client that reconnects for every request is throttled all the same. This is
not a security boundary. A client
may name itself for the logs with an `X-Simul-Agent` header, recorded as
`anonymous:<label>`; the label decides nothing. Connect agents that need their own
instance or claim with the handshake (for example `fastmcp.Client(url,
mode="legacy")`) or over stdio. With a [bearer token](#network-transports-http-and-sse)
configured, sessionless clients are authenticated, and since they all present
the same token they share one session instead of having none.

When a handshake session ends, the server drops what it kept for it (its
instance switch, bindings and rate-limit buckets) and marks its claims
orphaned. Rate-limit buckets of identities whose end the server cannot see
(stdio's single client, authenticated sessionless users) are swept once they
have refilled, which never changes a verdict.

Usage records (`get_tool_usage_stats`, `simul stats`) name the agent by its
claim's `agent_id` when its session holds one, as `list_isaac_instances`
shows it, and by the session's default label otherwise.

## Usage statistics

Every tool call is recorded in a persistent JSONL log
(`~/.simul/logs/tool_usage.jsonl`), each row with the agent label, the
session label it came in on and the `request_id` its audit row carries. A
call the client cancels is recorded too, with `cancelled: true` and
`executed`, whether the engine ran it all the same: cancelling the request
does not stop the engine. `executed` is `false` for a call still waiting for
the instance (`error: "cancelled_before_start"`); `true` for an Isaac call,
which keeps the instance until the engine is done, and then the row keeps
the engine's own `success` and `error`; and `"unknown"` for a call whose
request was already sent and that nothing waited for (Unreal, Blender:
`error: "cancelled"`), so check the scene before retrying. Every field of a
row is bounded (names and agent labels at 128 characters, the error at
2 000, a string parameter at 256, a list or mapping parameter by its length
past 1 KB), so a call with a huge argument writes a small row. The log is
rotated at 10 MB into `tool_usage.jsonl.1` ... `.5` (the oldest is dropped),
and `simul stats` streams the files, keeping only the records it shows. A
`claim_isaac_instance` or `set_active_isaac_instance` `agent_id` is at most
128 characters and a `purpose` at most 1 000. Agents read their own session's calls with `get_tool_usage_stats`
(`scope="server"` adds per-tool counts across every session, never other
sessions' records); operators read everything with `simul stats` and clear
it, rotated copies included, with `simul stats --reset`.

## Audit log

`audit.jsonl` has one row per tool call, written as the call ends:

```json
{"ts": "...", "request_id": "3c34508323c0", "tool": "load_usd_file", "duration_ms": 1.2,
 "status": "error", "error_type": "SandboxError", "agent_id": "robot-1", "session": "session-3b00cb8d6c8f"}
```

`status` is `ok`, `error` (with the envelope's `error_type`, and `reason`
`rate_limited`, `instance_busy` or `instance_claimed` for a call refused
before it ran) or `cancelled`, with `executed` as in the usage row; an
exception that escaped the tool adds `error_class`. `request_id` matches the
call's usage row and the `[request_id tool]` prefix of its lines in
`simul.log`. Each field is cut at 128 characters.

Every server on the machine writes the same `audit.jsonl` and `simul.log`.
They rotate together: each record is written under a lock on
`<file>.lock`, to the file at its path opened afresh, and a file last
written on an earlier day is renamed to `<file>.<YYYY-MM-DD>` by whichever
process writes first that day. (Python's timed rotation, used before, kept
each process writing into the file it had open, so after the first midnight
the other servers' rows went into the previous day's file and retention
deleted them.) Dated copies older than `LOGGING__RETENTION_DAYS` days are
deleted. `audit.jsonl` is also rotated at 10 MB within a day, and at most 30
copies are kept. The lock and the rotation never cost a record: when the lock
cannot be had (an NFS home without a lock daemon, a log directory where the
lock file cannot be created) or another process holds it for more than
50 ms, or a rotation fails, the record is appended to the file as it is,
locking is skipped for 30 s, and one warning goes to stderr.

Log files are private to the user: directories simul creates for them are
0700 and every log, audit and usage file, rotated copies, lock files and
`simul.json` included, is created 0600. Existing files the user owns,
`tool_usage.jsonl` and its rotated copies included, are tightened to 0600; an
existing directory keeps its mode.
