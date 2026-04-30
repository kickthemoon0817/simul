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

2. **Run `simul unreal setup <.uproject> --headless --yes`** (from the repo
   root or wherever the `simul` entrypoint is installed). **Default to
   headless** — it's faster, has no focus dependency, and screenshots /
   scene captures still work because the render pipeline is live. Only
   drop `--headless` if the user explicitly wants the GUI editor open
   (e.g. for hand-editing a scene at the same time as running simul).

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
   - With `--headless`, appends `-RenderOffScreen -unattended -nopause
     -nosplash -nosound -stdout -FullStdOutLogOutput` so no window opens
     and the editor doesn't need OS-level focus to render screenshots.
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
