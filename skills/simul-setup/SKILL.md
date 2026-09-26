---
name: simul-setup
description: Use when the user asks to "set up simul", "install simul", "configure simul", "get started", "which backend", "how to install", or needs help choosing and installing the right backends for their simulation workflow.
version: 0.1.0
---

# Simul Setup Workflow

Guide the user through installing simul-mcp with the right backends for their workflow. Ask what simulation engines they use, then provide the exact install and configuration steps.

## Step 1: Identify the User's Backends

Ask the user which simulation engines they plan to use:

- **Isaac Sim** (NVIDIA Omniverse) — robotics, synthetic data, physics simulation
- **Unreal Engine** (Epic Games) — real-time visualization, virtual production, automotive
- **Blender** — open-source 3D modeling, animation, rendering
- **USD only** — headless scene analysis, asset validation, no runtime needed
- **Unity** — not yet supported, planned for a future release

Multiple backends can be used simultaneously. Isaac Sim and Unreal are the most common pairing.

## Step 2: Check Python Version

Simul requires Python 3.11, 3.12, or 3.13 (`python3 --version`). Embedded
Blender (`bpy`) additionally needs 3.11 or 3.13; see the README's
requirements section for the per-backend matrix.

## Step 3: Install

```bash
git clone https://github.com/kickthemoon0817/simul.git
cd simul
```

Then install based on the chosen backends:

| Backends | Command |
|----------|---------|
| USD only | `uv sync` |
| Isaac Sim | `uv sync` |
| Unreal Engine | `uv sync` |
| Blender | `uv sync --extra blender` |
| Isaac + Unreal + dev tools | `uv sync --extra dev` |
| Everything (including Blender) | `uv sync --extra dev --extra blender` |

Or with pip:

```bash
pip install -e .                    # Core (USD + Isaac + Unreal)
pip install -e ".[blender]"         # + Blender
pip install -e ".[dev]"             # + dev tools
pip install -e ".[dev,blender]"     # Everything
```

## Step 4: Configure Each Backend

### Isaac Sim

Isaac Sim provides its own `pxr` and `omni` Python modules — no extra pip packages needed.

1. Install the `khemoo.simul.mcp` bridge extension into Isaac Sim:
   - Run `simul-mcp isaac install-bridge` (uses bundled
     `src/simul_mcp/bridge_ext/khemoo.simul.mcp/`, works from a pip
     install or repo checkout). Add `--symlink` for editable workflows.
   - Or use Docker Compose: `docker compose -f compose.isaac-sim.yml up`

2. Start Isaac Sim with the bridge enabled:
   - `simul-mcp isaac launch` (any version; enables the Python socket and
     the bridge, then waits for both ports), or
   - `simul-mcp isaac bridge-up` when a 5.x editor is already running.

3. Verify connectivity:
   ```bash
   simul-mcp isaac ping
   ```

Ports are fixed, not auto-allocated: bridge 8229, stock Python socket 8226.
For a second instance pass `--socket-port` / `--bridge-port` to `launch`.

### Unreal Engine

Simul communicates with Unreal via the built-in Remote Control HTTP API. No extra Python packages needed.

1. Run setup against the project (ask for the `.uproject` path first):
   ```bash
   simul unreal setup /path/to/Project.uproject --yes
   ```
   It enables `RemoteControl` + `PythonScriptPlugin`, writes
   `Config/DefaultRemoteControl.ini`, launches the editor headless, and
   waits until Remote Control answers. Add `--no-launch` when the editor
   is already running. Don't hand-edit the `.uproject` or ini; see
   `docs/unreal-setup.md` for exactly what gets written.

2. Verify connectivity:
   ```bash
   simul-mcp unreal health
   ```

### Blender

Two modes:

- **Embedded** (default): `bpy` runs inside the MCP server. Install with
  `--extra blender`; no Blender editor needed.
- **Attached**: drive an already-open Blender window. Run
  `simul blender install-bridge`, enable the add-on ZIP in Blender, then
  `simul blender attach` and start the server with
  `--backends blender --blender-mode attached`. See
  `docs/blender-attachment.md`.

### USD Only (Headless)

No runtime setup needed. USD tools work immediately after install:

```bash
simul-mcp usd info /path/to/scene.usd
simul-mcp usd validate /path/to/scene.usd
simul-mcp usd summary /path/to/scene.usd
```

## Step 5: Register with AI Agent

### Claude Code

```bash
# All backends
claude mcp add simul -- uv --directory /path/to/simul run simul-mcp server

# Unreal only (minimal context)
claude mcp add simul -- uv --directory /path/to/simul run simul-mcp server --backends unreal

# Isaac Sim only
claude mcp add simul -- uv --directory /path/to/simul run simul-mcp server --backends isaac
```

### Codex (OpenAI)

Add to `.codex/config.json`:
```json
{
  "mcpServers": {
    "simul": {
      "command": "uv",
      "args": ["--directory", "/path/to/simul", "run", "simul-mcp", "server"]
    }
  }
}
```

## Step 6: Verify

Run a quick health check for each configured backend:

```bash
# Isaac Sim
simul-mcp isaac ping

# Unreal Engine
simul-mcp unreal health

# USD (always available)
simul-mcp usd info /path/to/any/scene.usd

# Show all registered tools
simul-mcp info
```

If any backend fails, re-check the setup steps above. Common issues:
- **Isaac Sim**: Isaac Sim not running, or started without `simul-mcp isaac launch` (bridge not enabled)
- **Unreal**: setup not run against this project; re-run `simul unreal setup <.uproject> --no-launch --yes`
- **Blender**: embedded mode on the wrong Python version (need 3.11 or 3.13)
