# CLAUDE.md — Simul MCP

Coding standards, repo layout, and runtime notes live in `AGENTS.md`. This file
holds behavioral instructions for Claude Code when working with simul.

## Unreal workflow — use `simul unreal setup`, don't hand-configure

Using simul with Unreal requires UE5's Remote Control plugin + Python execution.
Users should **not** be asked to flip these switches by hand every session.
The `simul unreal setup` CLI does the full flow in one call: idempotent
`.uproject` + `DefaultRemoteControl.ini` patching, platform-aware editor
launch, and polling `unreal_health_check` until the port is live.

### Whenever the user asks to use simul with Unreal

Phrases like "use unreal", "open my UE project", "run simul on unreal",
"connect to unreal" → follow this flow:

1. **Ask the user** for the absolute path to their `.uproject` file (skip if
   unambiguous from CWD / conversation). Confirm before touching their project.
   Ask for `--engine-path` only if auto-detection is likely to fail (custom UE
   build, non-standard install, Linux where `UnrealEditor` isn't on `PATH`).

2. **Run `simul unreal setup <.uproject> --yes`** (from the repo root or
   wherever the `simul` entrypoint is installed). The CLI launches UE
   **headless by default** — no window opens, no focus dependency, and
   screenshots / scene captures still work because the render pipeline
   is live. Pass `--no-headless` only when the user explicitly wants the
   GUI editor open (e.g. to hand-edit a scene while simul is running).

   The CLI handles everything:

   - Patches the `.uproject` to enable `RemoteControl` and `PythonScriptPlugin`
     (idempotent — only writes when something is missing or different).
   - Patches `Config/DefaultRemoteControl.ini` under
     `[/Script/RemoteControlCommon.RemoteControlSettings]` with `bAutoStartWebServer`,
     `bAutoStartWebSocketServer`, `RemoteControlHttpServerPort`,
     `bRestrictServerAccess=True`, `bEnableRemotePythonExecution=True`.
   - Resolves the launcher per-OS (macOS: direct `UnrealEditor.app` binary
     when headless or `--engine-path` given, else LaunchServices `open -a`;
     Linux: `Engine/Binaries/Linux/UnrealEditor`). Picks the highest
     installed UE version when multiple are present (UE_5.7 over UE_5.6).
   - When headless (the default), appends `-RenderOffScreen -unattended
     -nopause -nosplash -nosound -stdout -FullStdOutLogOutput` so no
     window opens and the editor doesn't need OS-level focus to render
     screenshots.
   - Spawns the editor detached, then polls `unreal_health_check` up to
     `--wait-timeout` (default 90 s).

   Useful flags: `--port` (HTTP port, default 30010), `--engine-path`,
   `--no-launch` (config only — user already has the editor running),
   `--wait-timeout`, `--poll-interval`, `--bind`, `--websocket-port`,
   `--allow-public`, `--passphrase` (the last four are for cross-host
   scenarios — see below; ignore them for the common single-machine
   case).

3. **If the editor was already running**, call `simul unreal setup <.uproject>
   --no-launch --yes` — this still patches config on disk (harmless when keys
   already match) and just polls for readiness.

4. **Then use simul's Unreal tools** (`capture_unreal_viewport`,
   `execute_unreal_script`, the granular set, etc.).

### Hard rules

- Prefer `simul unreal setup` over hand-editing the `.uproject` or ini.
- Don't drop `--yes` unless you want the user to confirm interactively —
  in automated flows, ask the user once, then pass `--yes`.
- Don't hardcode macOS paths on Linux or vice versa — the CLI detects the OS
  and the user can override with `--engine-path` when needed.
- If `simul unreal setup` exits non-zero, surface the actual error (plugins
  not enabled, Python execution disabled, port conflict, editor crashed).
  Don't pretend it worked.

### Cross-host / remote access — only when the user actually needs it

The default `simul unreal setup` configures Remote Control to bind to UE's
default hostname (typically loopback), which is what you want when both
the editor and `simul-mcp` run on the same machine. Three opt-in flags
exist for cross-host workflows; **don't use them speculatively** — they
widen the trust radius.

- `--bind <host>` — sets the HTTP listener bind address in
  `Config/DefaultEngine.ini` under `[HTTPServer.Listeners]` as
  `DefaultBindAddress=<host>`, and the WebSocket bind address in
  `Config/DefaultRemoteControl.ini` as
  `RemoteControlWebsocketServerBindAddress=<host>`. (UE's HTTP bind
  is NOT a `URemoteControlSettings` field — earlier versions wrote
  the bogus `RemoteControlHttpServerHostname` key, which UE silently
  ignored.) Pass
  `0.0.0.0` to accept connections from anywhere, or a specific interface
  IP to bind to one network. Omit for the safe loopback default.
- `--websocket-port <int>` — overrides `RemoteControlWebSocketServerPort`
  (UE default 30020). Use only when running multiple UE editors on the
  same host so their WebSocket endpoints don't collide. The HTTP-side
  collision is already solved by `--port`.
- `--allow-public` — required acknowledgment whenever `--bind` is
  non-loopback. UE Remote Control runs **without authentication** and we
  enable `bEnableRemotePythonExecution=True`, so a public bind exposes
  arbitrary Python execution to anything on the network. The CLI refuses
  a non-loopback `--bind` without this flag.
- `--passphrase <plaintext>` — layer-2 hardening on top of the IP
  allowlist when `--bind` is non-loopback. simul MD5-hashes the
  plaintext (UE 5.x's `FMD5::HashAnsiString`) and writes the entry to
  the ini's `+Passphrases` array, plus pins
  `bEnforcePassphraseForRemoteClients=True`. The CLI refuses
  `--passphrase` without a non-loopback `--bind` because over loopback
  the IP allowlist already blocks remote access — turning passphrase on
  there would only break clients with no security gain. To make
  simul-mcp's own Remote Control calls work against a passphrase-enabled
  editor, set the matching plaintext (or pre-computed MD5 hex) on the
  client side via the `UNREAL__PASSPHRASE` environment variable
  (or in `.env`). simul-mcp's `UnrealRuntimeSession` then attaches
  `Passphrase: <md5>` to every Remote Control request automatically.

When the user asks for cross-host UE control, ask first: *"Is the
network behind a firewall, or are you OK exposing arbitrary Python
execution to it?"* Only pass `--allow-public` after they confirm.
Suggested invocation for trusted-LAN remote access:

```
simul unreal setup <.uproject> --bind 0.0.0.0 --allow-public --yes
```

For multi-instance on one host (no remote exposure), separate the ports:

```
simul unreal setup <.uproject> --port 30011 --websocket-port 30021 --yes
```

## Isaac workflow — `install-bridge` once per Isaac install, then `launch` (or `bridge-up`) per launch

Supported: Isaac Sim 5.1.0, 6.0.0, 6.0.1. Isaac Sim 5.1 ships with the
VS Code transport (`isaacsim.code_editor.vscode`) auto-enabled on port
8226; Isaac Sim 6.0 moved that socket into
`isaacsim.code_editor.python_server` and enables **nothing** at startup.
simul's preferred transport on every version is the `khemoo.simul.mcp`
bridge extension on port 8229 (typed protocol, faster, fewer
round-trips). Two post-clone steps wire it up cleanly:

### 1. One-time per Isaac install: `simul-mcp isaac install-bridge`

The bridge extension ships **inside** the `simul-mcp` Python package
at `src/simul_mcp/bridge_ext/khemoo.simul.mcp/` (so pip installs and
editable installs both have it on disk), and must be physically
present at `<isaac-root>/extsUser/khemoo.simul.mcp/` for the editor
to load it. Repo bumps and pip upgrades don't propagate by themselves
— Isaac keeps loading whatever stale copy is in `extsUser` until you
publish.

`simul-mcp isaac install-bridge` does the publish. Recommended for repo
workflows:

```
ISAAC_SIM_PATH=~/isaac-sim-5.1.0 simul-mcp isaac install-bridge --symlink
```

`--symlink` (vs the default copy) means future `git pull`s on the repo
propagate to Isaac without re-running the command. Use plain copy mode
if Isaac runs as a different user from the repo owner.

`--force` replaces the dest even when versions match (useful for
switching copy ↔ symlink mode). Without `--force` the command no-ops
when the dest version already matches the source.

### 2. Per Isaac launch: `simul-mcp isaac launch` (all versions) or `bridge-up` (5.x, editor already running)

A fresh `isaac-sim.sh` start leaves the bridge extension
**registered but disabled** — port 8229 silently doesn't bind, even
though the ext is present in `extsUser`. On 6.0 the Python socket on
8226 is disabled too, so nothing can enable the bridge after the fact.

`simul-mcp isaac launch` is the version-agnostic answer: it reads
`<isaac-root>/VERSION`, starts `isaac-sim.sh` detached with
`--enable <python socket ext> --enable khemoo.simul.mcp` (plus
`--no-window` unless `--no-headless`), and polls both ports until they
answer or `--wait-timeout` (default 180 s) expires. Its JSON output
carries `pid`, `log_file`, `version`, `transport_extension`,
`socket_reachable`, `bridge_reachable`, and `socket_protocol`.

```
ISAAC_SIM_PATH=~/isaac-sim-6.0.1 simul-mcp isaac launch
simul-mcp isaac launch --isaac-root ~/isaac-sim-5.1.0 --no-headless
simul-mcp isaac launch --dry-run        # show the command, start nothing
```

Useful flags: `--socket-port`, `--bridge-port`, `--auth-token` (6.0+
only, turns on python_server `require_auth`; pair it with
`ISAAC_SIM__SOCKET_AUTH_TOKEN` on the MCP side), `--kit-arg` (repeatable
passthrough), `--log-file`. When the bridge ext isn't published under
`extsUser/`, the command still starts Isaac with the Python socket only
and prints the `install-bridge` hint.

When the user already has a **5.x** editor running, `simul-mcp isaac
bridge-up` auto-enables the bridge via the VS Code transport (8226),
then re-probes the bridge with a 6×0.5s retry loop (Kit needs a frame
to bind the socket).

```
simul-mcp isaac bridge-up
```

The command's JSON output reports `action: "already-up" | "auto-enabled"`,
`success: bool`, `bridge_reachable: bool`, etc. Idempotent — re-running
when the bridge is already up returns `already-up` instantly. On a 6.0
editor started without `--enable` flags it reports `NotRunning` with the
launch hint; restart the editor through `launch` instead.

### Hard rules

- Never publish the bridge ext by hand-copying or symlinking — always
  go through `install-bridge` so the version is read + verified at the
  end (the command catches a partial extraction or a wrong-version
  source up front).
- After updating the repo (e.g. via `git pull`), re-run `launch` /
  `bridge-up` but NOT `install-bridge` if you used `--symlink` last
  time. The symlink already tracks the repo; only the per-launch enable
  step needs to repeat.
- Don't hand-assemble `--enable` flags for `isaac-sim.sh` when
  `launch` can do it; the socket extension name differs between 5.x
  and 6.0 and `launch` picks it from the install's `VERSION` file.
- The stock socket client auto-detects 5.x vs 6.0 wire behaviour
  (6.0's python_server only executes after a TCP half-close, 5.x's VS
  Code ext closes the connection on half-close). Set
  `ISAAC_SIM__SOCKET_PROTOCOL=python_server|vscode` only to skip the
  probe; never send EOF unconditionally.
- Never tell the user to manually run
  `simul-mcp isaac enable-extension khemoo.simul.mcp` — that was the
  pre-iter10 workaround. `bridge-up` does the same thing plus the
  retry loop and structured payload.

## Project scope

Simul is a Model Context Protocol (MCP) server that gives Claude Code and
other MCP clients live control over 3D simulation and DCC backends:

- **Isaac Sim 5.1.0 / 6.0.0 / 6.0.1** — granular tools over a TCP socket
  bridge: scene/prim inspection, physics, materials, viewport/camera,
  simulation control, rendering, asset/stage ops, extension management.
  Falls back to `execute_isaac_script` for anything not covered by a
  granular tool. Default ports: 8226 (stock Python socket:
  `isaacsim.code_editor.vscode` on 5.x, `isaacsim.code_editor.python_server`
  on 6.0), 8229 (optional bridge extension). Set `ISAAC_SIM_PATH` to the
  Isaac install root. 6.0 deprecates `isaacsim.core.{api,prims,utils}` in
  favour of `isaacsim.core.experimental.*` and drops the `omni.isaac.*`
  shims — scripts must not rely on either.
- **Unreal Engine 5** — Remote Control HTTP/WebSocket plus
  `PythonScriptPlugin`; always set up via `simul unreal setup` (see above).
- **Blender** — connected adapter when a Blender runtime is up.
- **USD (headless)** — file-level operations that don't need a running engine.

The Python package is `simul-mcp`, installed editable from this repo
(`pip install -e .`). The editor-facing skills, slash commands, and the
`.claude-plugin/plugin.json` manifest live in this same repo.

## Versioning

Version is tracked in **four places that must always move together**:

- `pyproject.toml` → `[project] version = "X.Y.Z"`
- `.claude-plugin/plugin.json` → `"version": "X.Y.Z"`
- `src/simul_mcp/__init__.py` → `__version__ = "X.Y.Z"`
- `src/simul_mcp/bridge_ext/khemoo.simul.mcp/config/extension.toml` →
  `[package] version = "X.Y.Z"` (the Isaac Sim bridge extension
  shipped *inside the wheel* as of iter14 — its version-suffixed Kit
  ID, e.g. `khemoo.simul.mcp-X.Y.Z`, must match the parent so callers
  reading either side see consistent metadata. The bridge ext was
  added to the lockstep in iter11 after drifting from 0.0.19 → 0.0.32
  across 13 patch tags went unnoticed; iter14 moved its canonical
  location into the package so pip-installed users no longer need a
  repo checkout for `simul-mcp isaac install-bridge`.)

### Release sequence — do not deviate

1. Bump all four constants in a single commit on a
   `chore/bump-version-X.Y.Z` branch. Commit subject: `chore: bump version
   to X.Y.Z` — no parentheses, no body needed.
2. Open a PR against `main`. Branch protection rejects direct pushes to
   `main`, so the PR is mandatory. Merge with a merge commit — project
   convention, and it preserves the bump commit's SHA so a tag can point at
   it.
3. After the PR merges, fetch and tag the **bump commit itself** (not the
   merge commit) with an annotated tag:
   `git tag -a vX.Y.Z -m "Release vX.Y.Z" <bump-sha>`.
4. Push the tag: `git push origin vX.Y.Z`. The tag must be reachable from
   `main` after this — verify with `git branch --contains vX.Y.Z main`.

### Hard rules

- Never tag a commit whose three version constants don't match the tag
  string. The bump commit and the tag must agree byte-for-byte. Past
  mistake to avoid: `v0.0.20` was tagged on a commit that still said
  `0.0.19` in source. The protocol above exists to prevent that recurring.
- Never reuse a version number whose tag is already on the remote, even
  if that tag points at the wrong commit. Skip to the next number — that's
  why this repo went `v0.0.19` → `v0.0.21`, with no `v0.0.20` bump commit
  ever made.
- Never force-move or delete a published tag. It breaks anyone who already
  fetched it.
- Pre-1.0 version semantics here are loose, but bias toward a minor bump
  (`0.X.0`) when meaningful new features land and a patch bump (`0.X.Y`)
  for fixes only. Audit `git log v<last-tag>..HEAD --oneline` before
  picking the next number.

## Marketplace integration

This plugin is published via the marketplace at
`https://github.com/kickthemoon0817/khemoo-claude-plugins`. The marketplace
manifest (`.claude-plugin/marketplace.json` in that repo) lists `simul` with
`source: { source: "github", repo: "kickthemoon0817/simul" }` and **no
version pin** — every user installing or updating the simul plugin always
gets whatever is on `main` of this repo.

Implications:

- The version constants in this repo are advisory metadata for humans and
  tooling; they do **not** gate what users receive. Pushes to `main` are
  effectively releases. Treat that with the same care.
- Stale version constants on `main` — where the manifest claims an old
  version while users are running newer code — are a confusing user-facing
  bug. Keep the versioning protocol above tight to avoid it.
- For a normal code release, nothing in `khemoo-claude-plugins` needs to
  change. Only edit `khemoo-claude-plugins/.claude-plugin/marketplace.json`
  when changing this plugin's marketplace metadata (description, keywords,
  category, source repo, or owner). Such changes go through that repo's
  own PR flow.

## Workflow gotchas

- `main` is branch-protected — direct push is rejected. All changes go
  through PRs (`gh pr create … && gh pr merge <N> --merge --delete-branch`).
- `gh issue view <N>` errors on this repo with the Projects-classic
  deprecation; use `gh api repos/kickthemoon0817/simul/issues/<N>` instead.
- `pytest tests/` should be 100% green on `main`. Reference
  numbers from iter20 baseline: `491 passed, 6 skipped, 3
  deselected, 0 failed`. The 6 skipped are `@pytest.mark.isaac` /
  `@pytest.mark.unreal_live` tests that need a running engine;
  the 3 deselected are the `packaging` marker tests
  (`tests/packaging/test_wheel_contents.py`) that addopts skips
  by default. The historical 14-16 pre-existing FakeFastMCP
  `add_middleware` failures were eliminated in iter17 (Blender
  file) + iter20 (the remaining 3: `tests/mcp/test_discoverability.py`,
  `tests/mcp/test_isaac_bridge_server.py`, `tests/mcp/test_isaac_session_routing.py`).
  All 5 FakeFastMCP doubles now stub `add_middleware` and
  `resource`. New failures are real regressions; debug them.
- The simul MCP server in a running Claude Code session does **not**
  hot-reload — Python loads source at process start, edits don't
  propagate. To live-verify a `simul_mcp` source change, run the
  editable-installed `simul-mcp` CLI as a fresh subprocess (it picks up
  edits via `pip install -e .`).
- `_execute_json_script` in `src/simul_mcp/mcp/tools/isaac_tools.py` wraps
  script JSON with `data.setdefault("success", True)`. To surface a domain
  failure, the inner script must explicitly emit
  `{"success": false, "error": "..."}` — otherwise the wrapper marks it
  true.
- The `simul-mcp` CLI exits non-zero when the parsed result has
  `success: false`. That's not an infrastructure failure — read the JSON
  payload.
- Strings shaped like `/exts/khemoo.simul.mcp/<key>` (in
  `compose.isaac-sim.yml`'s kit args, the bridge ext's `extension.py`,
  and the CLI's `settings.set(...)` calls) are **Carb settings keys**,
  not filesystem paths. The `/exts/<ext_name>/` namespace is Carb's
  convention; it doesn't move when the bridge ext directory moves on
  disk. Don't refactor them as part of a path rename.
- The dev test runner is `~/pt/simul/.venv/bin/python` (and
  `~/pt/simul/.venv/bin/simul-mcp` for the fresh-subprocess live
  verification pattern). Most pytest invocations in the loop use this
  venv; calling system `python` won't have the editable simul-mcp
  install on path.
- Don't add `__init__.py` to a `tests/<name>/` directory whose name
  collides with an installed pip package (e.g. `packaging`,
  `requests`, `setuptools`). Pytest treats the test dir as that
  Python module and fails collection with `ModuleNotFoundError`.
  Leave the test directory as a plain dir — pytest's rootdir-based
  discovery handles it without an init.
- Long-running build tests live under the `packaging` pytest marker
  (e.g. `tests/packaging/test_wheel_contents.py` — runs `uv build`
  then inspects the wheel zipfile). The bare `pytest tests/`
  invocation skips them automatically because `-m "not packaging"`
  is baked into `pyproject.toml addopts`; `pytest -m packaging`
  runs them as a pre-publish gate (the LAST `-m` wins, so the
  explicit override beats the default).

## Live-driven testing — when the machine has the runtime, use it

simul has unit tests *and* a live tier (`@pytest.mark.isaac`,
`@pytest.mark.unreal_live`, the Blender adapter, headless USD). The unit
tier runs everywhere; the live tier is the only thing that actually
proves the wire. **Whenever the machine has the corresponding runtime
available, run the live tier — don't stop at unit-tests-passed.**

How to detect "the machine has it":

- **Isaac Sim** — local installs at `~/isaac-sim-5.1.0/`,
  `~/isaac-sim-6.0.0/`, `~/isaac-sim-6.0.1/` (or `$ISAAC_SIM_PATH`).
  Launcher is `isaac-sim.sh`. Live socket on 8226 (stock Python socket)
  and optional 8229 (bridge). If the binary exists but isn't running,
  **start it with `ISAAC_SIM_PATH=<root> simul-mcp isaac launch`** (it
  enables the right transports per version and waits for the ports),
  then run the live test. A transport change must be verified on both a
  5.1 and a 6.0 install. Don't claim a fix works without it.
- **Unreal Engine** — installed engines under `~/apps/unreal-*/` or
  `~/UnrealEngine/`, with `Engine/Binaries/<OS>/UnrealEditor`. For
  per-OS detection, also check `which UnrealEditor`. The live test
  pattern is `simul unreal setup <copy of a template>.uproject --yes`
  followed by the C1–C5 probes in `docs/unreal-e2e-checklist.md`.
- **Blender** — `which blender` plus a connected runtime, otherwise
  fall back to the headless USD path.

Hard rules:

- If the runtime is up (or can be brought up) and the change touches the
  corresponding adapter or tools layer, the live test is mandatory before
  reporting the change as working. Code-trace verification is a fallback
  for "runtime not available", not a substitute.
- For test targets that mutate project files (e.g. `simul unreal setup`
  patches a `.uproject`), copy the project to `/tmp/<scratch>` first and
  test against the copy. Never mutate user work as a side effect of
  verification.
- Always shut down anything you started: kill background Isaac Sim and
  UE editors at the end of the session unless the user asked you to
  leave them running.

## When simul MCP misbehaves — propose filing an issue

When you're using simul (`mcp__simul__*` tools or the `simul-mcp` CLI)
and hit something that looks like a real defect — wrong result,
unhelpful error, missing capability, behavior that contradicts the docs
or tool description, undocumented gotcha that cost the user time —
**don't silently route around it**. Surface it.

The right move:

1. Diagnose enough to be confident it's a defect, not a misuse on your
   end. Read the relevant code in `~/pt/simul/src/simul_mcp/` and write
   down what you observed vs. what the docs / tool description claim.
2. Tell the user: *"This looks like a bug in simul. Want me to file an
   issue at https://github.com/kickthemoon0817/simul/issues?"* Wait for
   explicit permission. Don't bundle the question with anything else.
3. Only after the user says yes, file via `gh api repos/kickthemoon0817/simul/issues`
   (because `gh issue create` and `gh issue view` error on this repo
   with the Projects-classic deprecation). Use this body shape:
   - **What** — one-sentence description, neutral wording
   - **Reproduction** — the exact command or tool call with full inputs,
     copy-pasteable
   - **Expected** — what should have happened, citing the doc/tool
     description if applicable
   - **Observed** — what actually happened, including the verbatim error
     payload
   - **Environment** — `simul-mcp` version, backend (Isaac Sim/UE/Blender)
     version, OS, anything that scopes the regression
   - **What works** — bullets of related-but-functioning behavior so the
     maintainer can localize the defect
4. Reference the issue number in any subsequent workaround so the user
   knows it's a tracked compromise, not a silent one.

Hard rules:

- Never file an issue without explicit user permission for *that* issue.
  A blanket "yes file issues when you find them" earlier in the session
  doesn't carry forward — re-ask each time.
- Don't file for something you only suspect. Verify by reading the code
  or running a minimal reproduction first.
- Don't file for questions. Questions go directly to the user.
- Never close, edit, or comment on an existing issue without permission
  either — same rule, same reasoning.
