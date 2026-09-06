---
name: setup
description: Bootstrap the simul MCP server (clone source + global install) and configure simulation backends. Run this once after installing the simul plugin from the khemoo marketplace.
---

# /simul:setup

End-to-end bootstrap for the simul plugin. Clones the source, installs the
`simul-mcp` Python package globally, registers it under
`~/.claude.json → mcpServers.simul`, and walks the user through backend
selection.

## Why this command exists

The simul plugin ships a Claude Code skills + commands surface, but the
heavy lifting (HTTP adapters for Isaac Sim / Unreal / Blender, USD
operations, viewport capture) lives in the `simul-mcp` Python package.
The plugin does not auto-register that MCP server — this command does
the full bootstrap: clone the source, install `simul-mcp` globally,
write the MCP server entry into `~/.claude.json`, and verify.

Doing it from `/simul:setup` (rather than auto-registering via a
plugin-shipped `.mcp.json`) keeps Claude Code from logging "failed to
spawn simul" warnings before `simul-mcp` is on `PATH`, and gives the
user a single source of truth for which `simul-mcp` binary their
Claude Code is talking to.

This is the only manual bootstrap step. Everything else (per-backend
config, project-specific `.uproject` patching for Unreal, etc.) is
covered by per-engine commands like `simul unreal setup <.uproject>`.

## Workflow

Follow these steps in order. Each step has explicit shell commands and
clear pass/fail criteria so a fresh Claude session can execute it
autonomously.

### Step 1 — Is `simul-mcp` already installed?

```bash
which simul-mcp || echo "not installed"
```

If a path prints (e.g. `~/.local/bin/simul-mcp`), skip to Step 5.

### Step 2 — Pick a source location

Default: `~/.simul/source/`. Ask the user if they have a preference;
otherwise use the default. State the location explicitly before
proceeding so they can back out.

### Step 3 — Clone the simul repo

```bash
mkdir -p ~/.simul
[ -d ~/.simul/source/.git ] || git clone https://github.com/kickthemoon0817/simul.git ~/.simul/source
git -C ~/.simul/source pull --ff-only
```

If the user already has a working clone elsewhere (e.g. they were
hacking on simul before installing the plugin), ask if they want to
use that path instead — `uv tool install` accepts any local source
tree.

### Step 4 — Install `simul-mcp` globally

Try installers in this order; stop at the first one that succeeds:

```bash
# Preferred — uv (fastest, isolated)
uv tool install ~/.simul/source && which simul-mcp

# Fallback 1 — pipx
pipx install ~/.simul/source && which simul-mcp

# Fallback 2 — pip --user
python3 -m pip install --user ~/.simul/source && which simul-mcp
```

If `which simul-mcp` doesn't print after install, the install
location's `bin/` directory isn't on `PATH`. Common cases:
- `~/.local/bin` (pip --user / pipx default) → user needs to add it
  to `PATH` (echo the right shell-rc append for their shell).
- `~/.cargo/bin` or `~/.local/share/uv/tools/.../bin` (uv tool) → `uv tool update-shell` fixes it.

Surface the actual missing-path issue rather than silently retrying.

### Step 5 — Verify the binary works

```bash
simul-mcp --version || simul-mcp --help | head -5
```

Anything other than a clean exit means the install is broken; do not
proceed past this step.

### Step 6 — Register the MCP server in `~/.claude.json`

Edit the user's Claude Code config so the next session spawns the
just-installed binary. Resolve the absolute path first
(`which simul-mcp` from Step 4) and write it explicitly — relying on
`PATH` at MCP-spawn time is fragile because Claude Code's spawn
environment may not include shell rc additions.

Patch JSON in place; preserve everything else in the file:

```python
# Run via: python3 -c "<this script>"
import json
import shutil
from pathlib import Path

cfg_path = Path.home() / ".claude.json"
binary = shutil.which("simul-mcp")
assert binary, "simul-mcp not on PATH after Step 4"

cfg = json.loads(cfg_path.read_text())
servers = cfg.setdefault("mcpServers", {})

# Default to all backends; let the user pin in Step 7 if they want fewer.
servers["simul"] = {
    "type": "stdio",
    "command": binary,
    "args": ["server"],
}
cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
print(f"registered simul -> {binary}")
```

Always make a backup of `~/.claude.json` first
(`cp ~/.claude.json ~/.claude.json.bak.$(date +%Y%m%d-%H%M%S)`) so the
user can roll back if the JSON edit corrupts anything.

If the user previously had a `simul` entry under `projects.<path>`,
leave it alone — the user-level `mcpServers.simul` takes precedence
for the global install case.

### Step 7 — Backend configuration

#### 7.0 — Compatibility check

Before showing the backend menu, detect what the user's machine can
actually run. Don't let them pick a backend their hardware or OS
won't support — surface the limitation up front rather than letting
runtime warnings explain it later.

```bash
uname -s                        # Darwin / Linux / (use $OS / wmic on Windows via WSL)
uname -m                        # arm64 / x86_64 / aarch64
nvidia-smi 2>&1 | head -3       # check for NVIDIA driver / GPU
nvidia-smi -L 2>/dev/null       # GPU model — Isaac needs RTX (Turing+)
```

Build a per-backend availability matrix and present only the
supported choices in the menu:

| Backend | Linux | macOS | Windows | Hard requirement |
|---|---|---|---|---|
| **Isaac Sim** | ✅ | ❌ | ✅ | NVIDIA RTX GPU (Turing 20xx or newer) |
| **Unreal Engine** | ✅ | ✅ | ✅ | none beyond the engine install |
| **Blender** | ✅ | ✅ | ✅ | none |
| **USD-only** | ✅ | ✅ | ✅ | none (pure CPU pxr) |

Specifically:

- **macOS** → Isaac is **not** an option. Don't even offer it. If
  the user explicitly asks for Isaac, explain that NVIDIA does not
  ship Isaac Sim for macOS and recommend the Unreal/Blender/USD
  paths instead.
- **No NVIDIA GPU** (nvidia-smi missing or non-zero exit) → Isaac
  is not an option, regardless of OS. Same explanation.
- **NVIDIA GPU but pre-Turing** (e.g. GTX 10-series, P-series) →
  warn that Isaac requires RTX-class hardware for the RTX renderer
  to function and ask the user to confirm before proceeding.
- **Apple Silicon macOS** → Unreal works (UE 5.5+ ships native
  arm64), Blender works (universal binary), USD-only works.
- **All OS / GPU combos** → USD-only is always available.

Print the resolved menu so the user sees what's filtered and why,
then proceed.

#### 7.1 — Per-backend setup

For each picked backend, walk the env-var setup **now** rather than
letting the user discover the requirement at runtime. simul's
templates expand env vars at server start, and a missing
`ISAAC_SIM_PATH` or `UE_ENGINE_PATH` produces a "warn-and-continue
with degraded behavior" path that's hard to debug after the fact.

Detailed flow per backend:

#### Isaac Sim (selected)

Pre-req: passed the compatibility check in 7.0 (Linux or Windows
with NVIDIA RTX GPU).

1. Probe: `echo $ISAAC_SIM_PATH`. If non-empty and
   `python.sh` (Linux) / `python.bat` (Windows) exists under it,
   confirm and skip to step 4.

2. **Auto-detect** by globbing the official install locations
   before asking. Stop at the first match that contains
   `python.sh` / `python.bat`:

   **Linux** (NVIDIA Omniverse Launcher default + manual installs):
   ```bash
   ls -d ~/.local/share/ov/pkg/isaac-sim-*/    \
         /opt/isaac-sim/*/                      \
         /opt/nvidia/isaac-sim/                 \
         ~/isaac-sim/                           \
         /workspace/isaac-sim/ 2>/dev/null
   ```

   **Windows** (PowerShell / WSL paths):
   ```bash
   ls -d "$LOCALAPPDATA/ov/pkg/isaac-sim-*/"                \
         "/mnt/c/Users/$USER/AppData/Local/ov/pkg/isaac-sim-*/" \
         "/c/Program Files/NVIDIA/IsaacSim/*/" 2>/dev/null
   ```

   Prefer the highest-versioned match (sort descending).

3. If auto-detection finds nothing, ask: "Where is your Isaac Sim
   install? (the directory that contains `python.sh` /
   `python.bat`)". Confirm the path exists and the launcher
   script is present.

4. Persist:
   - Detect shell from `$SHELL` (bash → `~/.bashrc`,
     zsh → `~/.zshrc`, fish → `~/.config/fish/config.fish`).
   - Append `export ISAAC_SIM_PATH="<path>"` (or `set -gx` for fish)
     to the shell rc, only if not already present.
   - `export ISAAC_SIM_PATH=...` in the current process so the rest
     of this command can use it.

5. Verify: `ls "$ISAAC_SIM_PATH/python.sh"` (or `.bat`) succeeds.

6. Tell the user the bridge socket is `localhost:8226`; if they
   need to change ports, set `ISAAC_SIM__SOCKET_PORT` / `ISAAC_SIM__BRIDGE_PORT`
   in the environment (the environment overrides the packaged default config
   at `src/simul_mcp/resources/config/default.yaml`).

#### Unreal Engine (selected)

1. Probe: `echo $UE_ENGINE_PATH` (and `$UNREAL_ENGINE_PATH` as a
   fallback name). If set and `Engine/Binaries/{Mac,Linux,Win64}/`
   exists under it, confirm and skip to step 4.

2. **Auto-detect** at the official install locations per OS.
   Prefer the highest version when multiple are installed.

   **macOS** (Epic Games Launcher default + alternates):
   ```bash
   ls -d "/Users/Shared/Epic Games/UE_"*/            \
         "/Applications/Epic Games/UE_"*/            \
         "$HOME/Applications/Epic Games/UE_"*/ 2>/dev/null
   ```
   Also probe LaunchServices: `open -Ra UnrealEditor`. If it
   succeeds, `simul unreal setup` will route through `open -a` at
   runtime — env var is optional. Skip the persist step in that
   case.

   **Linux** (Epic's binary release defaults + common manual
   install paths):
   ```bash
   ls -d ~/UnrealEngine/                         \
         "/opt/unreal-engine/UE_"*/              \
         /opt/UnrealEngine/                       \
         "$HOME/Epic Games/UE_"*/                 \
         /usr/local/UnrealEngine/ 2>/dev/null
   which UnrealEditor   # Epic ships a launcher script in some installs
   ```

   **Windows** (Epic Games Launcher + manual installs):
   ```bash
   ls -d "/c/Program Files/Epic Games/UE_"*/    \
         "/c/Epic Games/UE_"*/                  \
         "/d/Epic Games/UE_"*/                  \
         "$HOME/Epic Games/UE_"*/ 2>/dev/null
   ```

   Validate each candidate by checking
   `Engine/Binaries/Mac/UnrealEditor.app` (mac) /
   `Engine/Binaries/Linux/UnrealEditor` (linux) /
   `Engine/Binaries/Win64/UnrealEditor.exe` (win) actually exists.

3. If neither the env var nor any default location resolves, ask:
   "Where is your Unreal Engine install root (the directory that
   contains `Engine/`)?". Validate the binary file under that root.

4. Persist `UE_ENGINE_PATH` to the shell rc as in the Isaac flow.
   Skip if LaunchServices on macOS already resolves UnrealEditor —
   the env var is optional in that case.

5. Tell the user: project-level setup is `simul unreal setup
   <.uproject> --yes` (headless by default; cf. CLAUDE.md). They
   can run that against a `.uproject` whenever they want to bring
   simul up against a specific UE project.

#### Blender (selected)

1. Probe: `echo $BLENDER_PATH`. If set and the binary exists,
   confirm and skip to step 3.

2. **Auto-detect** at default install locations. Prefer the
   highest-versioned match.

   **macOS** (Blender Foundation `.app` + Steam variant):
   ```bash
   ls -d "/Applications/Blender.app/Contents/MacOS/Blender" \
         "/Applications/Blender "*"/Blender.app/Contents/MacOS/Blender" \
         "$HOME/Applications/Blender.app/Contents/MacOS/Blender" 2>/dev/null
   ```

   **Linux** (apt / snap / Flatpak / manual tarball):
   ```bash
   which blender                                      # /usr/bin/blender etc.
   ls -d /usr/local/bin/blender                       \
         /snap/bin/blender                            \
         /var/lib/flatpak/app/org.blender.Blender/*   \
         "$HOME/blender-"*"/blender" 2>/dev/null
   ```

   **Windows** (Blender Foundation default + Steam):
   ```bash
   ls -d "/c/Program Files/Blender Foundation/Blender "*"/blender.exe" \
         "/c/Program Files (x86)/Steam/steamapps/common/Blender/blender.exe" 2>/dev/null
   ```

3. If auto-detection finds nothing, ask for the Blender binary
   path and validate it's executable.

4. Persist `BLENDER_PATH` to the shell rc as in the Isaac flow.

5. Reference `src/simul_mcp/adapters/blender_runtime.py` for the
   runtime adapter; the user does not need to manually register
   anything beyond the env var.

#### USD-only (selected)

No env vars or runtimes required. Nothing further.

Always tell the user the persisted env vars only take effect in
**new** shell sessions — current Claude Code session needs to
quit/reopen (Step 8) AND any new terminals they open need to source
the rc again.

### Step 8 — Restart Claude Code

`~/.claude.json` is read at session start; the `simul` entry written
in Step 6 is not picked up by the current session. Tell the user to:

> Quit Claude Code and reopen — the simul MCP server will spawn on
> the next start.

After restart, sanity-check:
- `mcp__simul__unreal_health_check` (or any `mcp__simul__*` tool) is
  available.
- The `simul` MCP server shows as connected (`/mcp` slash command).

## Hard rules

- **Always confirm before cloning** — do not silently create
  `~/.simul/source/` without telling the user.
- **Always back up `~/.claude.json`** before the Step 6 edit
  (`cp ~/.claude.json ~/.claude.json.bak.<timestamp>`).
- **Never skip Step 5.** If verification fails, surface the underlying
  install error rather than writing a broken entry into
  `~/.claude.json`.
- **Don't use `pip install simul-mcp`** as a global step. The package
  is not on PyPI yet; that command will pull a name-squatted package
  if anything resolves at all.
- **Write the absolute path** of the resolved `simul-mcp` binary into
  `~/.claude.json`, not the bare name. Claude Code's MCP spawn
  environment may not include the user's shell rc, so a `PATH`-
  relative `command: "simul-mcp"` can fail at spawn time even when
  the binary works in their terminal.

## When to use this command

- Right after `/plugin install simul@khemoo`.
- After deleting `~/.simul/source/` to force a clean reinstall.
- When `mcp__simul__*` tools stop appearing — usually means the
  global `simul-mcp` was uninstalled or moved.

## When NOT to use this command

- For per-project setup (a specific `.uproject` for Unreal, a
  `~/.isaac/` config bump). Those flows are owned by per-engine
  commands like `simul unreal setup <.uproject>`. This command is
  the global "is simul itself installed and working" bootstrap.
