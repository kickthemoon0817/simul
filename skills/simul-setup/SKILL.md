---
name: simul-setup
description: This skill should be used when the user asks to "set up simul", "install simul", "configure simul", "get started", "which backend", "how to install", or needs help choosing and installing the right backends for their simulation workflow.
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

Simul requires Python 3.11, 3.12, or 3.13. Check the user's version:

```bash
python3 --version
```

**Blender users must use Python 3.11 or 3.13** — the `bpy` package has no 3.12 wheels.

| Python | USD | Isaac Sim | Unreal | Blender |
|--------|-----|-----------|--------|---------|
| 3.11   | Yes | Yes       | Yes    | Yes (bpy 4.x/5.0) |
| 3.12   | Yes | Yes       | Yes    | No      |
| 3.13   | Yes | TBD       | Yes    | Yes (bpy 5.1) |

**Recommended:** Python 3.11 for maximum compatibility across all backends.

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

2. Enable the extension in Isaac Sim:
   - Window > Extensions > search "simul" > Enable

3. Verify connectivity:
   ```bash
   simul-mcp isaac ping
   ```

The bridge extension auto-allocates ports for multi-instance support. No manual port configuration needed.

### Unreal Engine

Simul communicates with Unreal via the built-in Remote Control HTTP API. No extra Python packages needed.

1. Enable plugins in your `.uproject`:
   ```json
   {
     "Plugins": [
       {"Name": "RemoteControl", "Enabled": true},
       {"Name": "PythonScriptPlugin", "Enabled": true}
     ]
   }
   ```

2. Create `Config/DefaultRemoteControl.ini`:
   ```ini
   [/Script/RemoteControlCommon.RemoteControlSettings]
   bAutoStartWebServer=True
   bAutoStartWebSocketServer=True
   RemoteControlHttpServerPort=30010
   RemoteControlWebSocketServerPort=30020
   bRestrictServerAccess=True
   bEnableRemotePythonExecution=True
   bAllowConsoleCommandRemoteExecution=True
   ```

   > **Important:** `bRestrictServerAccess=True` is required. Without it, Python execution silently remains disabled.

3. Restart the Unreal Editor.

4. Verify connectivity:
   ```bash
   simul-mcp unreal health
   ```

### Blender

Blender integration uses the `bpy` pip package.

1. Ensure you installed with `--extra blender`
2. Blender runs in-process — no external editor needed for headless operations
3. For live editor integration, start Blender with the MCP addon (future release)

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
- **Isaac Sim**: Bridge extension not enabled, or Isaac Sim not running
- **Unreal**: Remote Control plugin not enabled, or `bRestrictServerAccess` not set to `True`
- **Blender**: Wrong Python version (need 3.11 or 3.13)
