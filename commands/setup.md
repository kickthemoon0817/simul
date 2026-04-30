---
name: setup
description: Bootstrap the simul MCP server (clone source + global install) and configure simulation backends. Run this once after installing the simul plugin from the khemoo marketplace.
---

# /simul:setup

End-to-end bootstrap for the simul plugin. Installs the `simul-mcp` Python
package globally so the plugin's `.mcp.json` can spawn it, then walks the
user through backend selection.

## Why this command exists

The simul plugin ships a Claude Code skills + commands surface, but the
heavy lifting (HTTP adapters for Isaac Sim / Unreal / Blender, USD
operations, viewport capture) lives in the `simul-mcp` Python package.
The plugin's `.mcp.json` calls `simul-mcp server` — that command needs
to exist on the user's `PATH`, which it doesn't until this command
clones the repo and installs it.

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
proceed to Step 6.

### Step 6 — Backend configuration

Delegate to the existing `simul-setup` skill workflow at
`skills/simul-setup/SKILL.md`. Ask the user which simulation engines
they run (Isaac Sim / Unreal / Blender / USD-only) and only register
those backends so the agent's tool list stays small.

For each picked backend, point the user at the matching follow-up
flow:
- Unreal → `simul unreal setup <.uproject> --yes` (headless by default,
  cf. CLAUDE.md).
- Isaac → verify the bridge socket on port 8226 is reachable; reference
  `config/isaac/default.yaml` if they need to adjust ports.
- Blender → register the runtime adapter; reference
  `src/simul_mcp/adapters/blender_runtime.py`.
- USD-only → no runtime needed; nothing further.

### Step 7 — Restart Claude Code

The plugin's `.mcp.json` was already loaded when the plugin was
installed; until Claude Code restarts, the failed-to-spawn `simul`
MCP server stays in its broken state. Tell the user to:

> Quit Claude Code and reopen — the simul MCP server will spawn
> successfully on the next start now that `simul-mcp` is on `PATH`.

After restart, sanity-check:
- `mcp__simul__unreal_health_check` (or any `mcp__simul__*` tool) is
  available.
- The `simul` MCP server shows as connected (`/mcp` slash command).

## Hard rules

- **Always confirm before cloning** — do not silently create
  `~/.simul/source/` without telling the user.
- **Never skip Step 5.** If verification fails, surface the underlying
  install error rather than continuing to Step 6.
- **Don't use `pip install simul-mcp`** as a global step. The package
  is not on PyPI yet; that command will pull a name-squatted package
  if anything resolves at all.
- **Don't write to `~/.claude.json` directly.** The plugin's own
  `.mcp.json` registers the simul MCP server when the plugin is
  enabled — stacking another `simul` server in user config double-
  registers and breaks the tools.

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
