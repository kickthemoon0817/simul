# Unreal Setup — what `simul unreal setup` actually configures

When you run `simul unreal setup <project>.uproject`, the CLI patches **two
plugins** and a single `.ini` section. "Remote Control" is overloaded in
the UE ecosystem, so this doc names exactly what is required, what is not,
and what is sometimes confused for the same thing.

## What gets enabled (the only two plugins required)

Both are flipped to `Enabled: true` inside the `.uproject`'s `Plugins`
list, idempotently — the patcher only writes when something is missing or
different.

| Plugin name (`.uproject` key) | Ships with UE | Why simul needs it |
|---|---|---|
| `RemoteControl` | Yes — `Engine/Plugins/Runtime/RemoteControl/` since UE 4.27 | Hosts the HTTP server (default port 30010) and WebSocket server (default 30020) that simul talks to. Settings live under `[/Script/RemoteControlCommon.RemoteControlSettings]` — the class is in the `RemoteControlCommon` module, and UE looks the section up by exact module-path match. |
| `PythonScriptPlugin` | Yes — `Engine/Plugins/Experimental/PythonScriptPlugin/` | Provides the embedded Python interpreter that the `exec` endpoint dispatches to when `bEnableRemotePythonExecution=True`. Without it, `mcp__simul__execute_unreal_script` (and the entire scripting layer) cannot run. |

## What gets written into `Config/DefaultRemoteControl.ini`

Section: `[/Script/RemoteControlCommon.RemoteControlSettings]`. Other sections
and comments in the file are preserved verbatim.

| Key | Value | Purpose |
|---|---|---|
| `bAutoStartWebServer` | `True` | HTTP API comes up automatically with the editor |
| `bAutoStartWebSocketServer` | `True` | WebSocket API comes up automatically |
| `RemoteControlHttpServerPort` | `--port` (default `30010`) | HTTP listen port |
| `bRestrictServerAccess` | `True` | Hardcoded by simul; safe default |
| `bEnableRemotePythonExecution` | `True` | Allows the Python script endpoint |
| `RemoteControlHttpServerHostname` | `--bind` (only when set) | Cross-host access; default loopback |
| `RemoteControlWebSocketServerPort` | `--websocket-port` (only when set) | Multi-instance disambiguation; default 30020 |

## What is NOT installed and is NOT required

These are sometimes mistaken for "the same thing as RemoteControl". Don't
enable them just because you're running simul.

| Thing | What it actually is | Should I install it for simul? |
|---|---|---|
| `RemoteControlWebInterface` | Optional Marketplace plugin: a browser dashboard built on top of the RemoteControl API for humans to poke at presets in a browser. | No. simul calls the API directly, not the dashboard. Install only if a human wants the browser UI for preset editing — orthogonal to simul. |
| "Web Remote Control" | Older docs term for the WebSocket transport of the `RemoteControl` plugin. Not a separate plugin in modern UE. | N/A — already covered by `RemoteControl`. |
| `RemoteControlComponent` / `URemoteControlPreset` (preset assets) | Saved presets that expose a curated subset of properties/functions through the API. | No. simul drives the API generically without presets. Use them if a non-simul caller wants a stable preset surface. |
| PixelStreaming | A WebRTC viewport-streaming plugin (humans watching a remote render in a browser). | No. Different problem domain. |
| Concert / `UnrealMultiUserServer` | Multi-user collaborative editing across multiple editors. | No. |
| Movie Render Queue remote / Switchboard | Distributed render orchestration. | No. |

## Common gotchas

- **Don't manually toggle `bRestrictServerAccess` to `False`.** simul pins
  it `True`. The right knob for cross-host scenarios is `--bind`
  combined with `--allow-public`, not loosening this flag.
- **Don't expect setup to work without `PythonScriptPlugin`.** The plugin
  ships with UE but is `Experimental` and disabled by default; simul
  enables it for you. If a UE update reverts it, re-run setup.
- **Two UE editors on one host need both `--port` and `--websocket-port`.**
  `--port` only separates HTTP. The WebSocket endpoint defaults to
  30020 and will collide unless you also pass `--websocket-port`.
- **A non-loopback `--bind` is refused without `--allow-public`.** UE
  Remote Control runs without authentication and we enable Python
  execution by default — a public bind without acknowledgment would
  silently expose arbitrary Python execution to the network.

## Quick reference

```sh
# Common single-machine setup (loopback, default ports)
simul unreal setup MyProject.uproject --yes

# Multi-instance on one host
simul unreal setup A.uproject --port 30011 --websocket-port 30021 --yes
simul unreal setup B.uproject --port 30012 --websocket-port 30022 --yes

# Trusted-LAN remote access (read the safety note above first)
simul unreal setup MyProject.uproject --bind 0.0.0.0 --allow-public --yes

# Patch config only — editor already running
simul unreal setup MyProject.uproject --no-launch --yes
```

See `simul unreal setup --help` for the full flag list and
`docs/unreal-e2e-checklist.md` for end-to-end verification probes.
