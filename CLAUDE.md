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
     `[/Script/RemoteControl.RemoteControlSettings]` with `bAutoStartWebServer`,
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

   Useful flags: `--port` (default 30010), `--engine-path`, `--no-launch`
   (config only — user already has the editor running), `--wait-timeout`,
   `--poll-interval`.

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

## Project scope

Simul is a Model Context Protocol (MCP) server that gives Claude Code and
other MCP clients live control over 3D simulation and DCC backends:

- **Isaac Sim 5.1.0** — granular tools over a TCP socket bridge: scene/prim
  inspection, physics, materials, viewport/camera, simulation control,
  rendering, asset/stage ops, extension management. Falls back to
  `execute_isaac_script` for anything not covered by a granular tool.
  Default ports: 8226 (VS Code plugin transport), 8229 (optional bridge
  extension). Set `ISAAC_SIM_PATH` to the Isaac install root.
- **Unreal Engine 5** — Remote Control HTTP/WebSocket plus
  `PythonScriptPlugin`; always set up via `simul unreal setup` (see above).
- **Blender** — connected adapter when a Blender runtime is up.
- **USD (headless)** — file-level operations that don't need a running engine.

The Python package is `simul-mcp`, installed editable from this repo
(`pip install -e .`). The editor-facing skills, slash commands, and the
`.claude-plugin/plugin.json` manifest live in this same repo.

## Versioning

Version is tracked in **three places that must always move together**:

- `pyproject.toml` → `[project] version = "X.Y.Z"`
- `.claude-plugin/plugin.json` → `"version": "X.Y.Z"`
- `src/simul_mcp/__init__.py` → `__version__ = "X.Y.Z"`

### Release sequence — do not deviate

1. Bump all three constants in a single commit on a
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
