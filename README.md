# Simul MCP Server

MCP (Model Context Protocol) server for simulation and DCC tools with USD operations and simulation control.

## Overview

This project provides a comprehensive MCP server implementation for Simul-MCP, enabling AI models to interact with USD scenes, perform simulation control, and capture viewport data. The server supports both headless USD operations and full Isaac Sim runtime integration.

Simul-MCP is designed for multi-engine workflows. Isaac Sim is the primary runtime, with planned adapters for Blender, Unreal, Maya, and 3ds Max. The goal is to expose consistent tools across engines for scale correction, SimReady asset formats, and engine-specific simulation features.

## Features

- **USD Operations**: Load, analyze, and manipulate USD files
- **Scene Analysis**: Extract scene information, prim details, and mesh statistics
- **Bounding Box Computation**: Calculate world and local space bounding boxes
- **Mesh Operations**: Analyze mesh topology, materials, and geometry
- **Isaac Sim Integration**: Viewport capture, simulation control, camera management
- **Unreal Engine Integration**: Scene control, actor manipulation, viewport capture, Python execution via Remote Control HTTP API
- **Blender Integration**: Scene and object manipulation via bpy
- **Backend Selection**: `--backends` flag to register only the engines you need, minimizing AI agent context overhead
- **Flexible Architecture**: Works in both headless and runtime environments
- **Comprehensive Logging**: Structured logging with multiple output formats
- **Configuration Management**: YAML-based configuration with environment variable support

## Requirements

- Python 3.11, 3.12, or 3.13
- USD Python bindings (`usd-core`) — installed automatically
- NVIDIA Isaac Sim 5.1.0+ (optional — for live simulation control)
- Unreal Engine 5.x with Remote Control plugin (optional)
- Blender via `bpy` package (optional — Python 3.11 or 3.13 only)

## Install via Claude Code Marketplace

simul ships through the `khemoo` Claude Code marketplace at <https://github.com/kickthemoon0817/khemoo-claude-plugins>, which also hosts other khemoo plugins (e.g. `khemoo-skills` for the `khemoo-vc` version-control workflow):

```text
/plugin marketplace add kickthemoon0817/khemoo-claude-plugins
/plugin install simul@khemoo
/simul:setup
```

That's the whole install. `/simul:setup` clones this repo to `~/.simul/source/`, installs the `simul-mcp` Python package globally (via `uv tool install`, with `pipx` / `pip --user` fallbacks), walks you through backend selection (Isaac Sim / Unreal / Blender / USD-only), and tells you when to restart Claude Code so the plugin's bundled MCP server can spawn cleanly.

No `pip install` needed — `simul-mcp` is not on PyPI yet, and `/simul:setup` handles the from-source install for you.

For a hacking-on-simul setup (you're working on the plugin itself, not just using it), use the source install below.

## Installation

**Requirements:** Python 3.11, 3.12, or 3.13

```bash
git clone https://github.com/kickthemoon0817/simul.git
cd simul
```

### Choose Your Backends

Install only what you need. USD support is included by default — pick the simulation engines you use:

| I want to use... | Install command |
|-------------------|----------------|
| **USD only** (headless scene analysis) | `uv sync` |
| **Isaac Sim** (NVIDIA Omniverse) | `uv sync` — then launch Isaac Sim with the bridge extension |
| **Unreal Engine** (5.x) | `uv sync` — then enable Remote Control in your UE project |
| **Blender** (Python 3.11 or 3.13) | `uv sync --extra blender` |
| **All backends + dev tools** | `uv sync --extra dev --extra blender` |

Or with pip:

```bash
# Core (USD + Isaac Sim + Unreal support)
pip install -e .

# With Blender
pip install -e ".[blender]"

# With dev tools
pip install -e ".[dev]"
```

### Backend-Specific Setup

**Isaac Sim** — No extra Python packages needed. Isaac Sim provides its own `pxr` and `omni` modules. Install the `khemoo.simul.mcp` bridge extension into Isaac Sim (see [Isaac Sim Extension](#isaac-sim-extension) below), or use the Docker Compose setup.

**Unreal Engine** — No extra Python packages needed. Simul communicates via UE5's built-in Remote Control HTTP API. Enable the `RemoteControl` and `PythonScriptPlugin` plugins in your `.uproject` and configure `DefaultRemoteControl.ini` (see [Unreal Engine Setup](#unreal-engine-setup) below).

**Blender** — Requires the `bpy` pip package which has strict Python version locks:
- Python 3.11 → `bpy 4.2–5.0.x` (Blender 4.x)
- Python 3.12 → Not supported by `bpy`
- Python 3.13 → `bpy 5.1.0` (Blender 5.1)

**Unity** — Planned for a future release. The architecture supports adding new backends via the adapter pattern.

### Python Version Guide

| Python | USD | Isaac Sim | Unreal | Blender |
|--------|-----|-----------|--------|---------|
| 3.11 | Yes | Yes (5.1) | Yes | Yes (bpy 4.x/5.0) |
| 3.12 | Yes | Yes | Yes | **No** (no bpy wheels) |
| 3.13 | Yes | TBD | Yes | Yes (bpy 5.1) |

**Recommended:** Python 3.11 for maximum compatibility across all backends.

## Agent Integration

Simul MCP works with any MCP-compatible AI coding agent. The most reliable
enrollment path from a local checkout is to point the agent at `uv run
simul-mcp server` inside this repository, so it uses the project-managed
virtual environment instead of assuming a global install.

Recommended repo-local command:

```bash
uv --directory /abs/path/to/simul run simul-mcp server
```

If you prefer a globally installed entrypoint, `simul-mcp server` also works
after installing the package into your environment.

The `skills.md` file in this repo is exposed automatically by the MCP server as
the `simul://isaac-sim/skills` resource. You do not need to install a separate
agent-side skill package to use it.

### Claude Code

Add to `~/.claude/settings.json` under `mcpServers`:

```json
{
  "mcpServers": {
    "simul": {
      "command": "uv",
      "args": ["--directory", "/abs/path/to/simul", "run", "simul-mcp", "server"]
    }
  }
}
```

Or use the CLI:

```bash
claude mcp add simul -- uv --directory /abs/path/to/simul run simul-mcp server
```

### Codex (OpenAI)

Add to `~/.codex/config.json`:

```json
{
  "mcpServers": {
    "simul": {
      "command": "uv",
      "args": ["--directory", "/abs/path/to/simul", "run", "simul-mcp", "server"]
    }
  }
}
```

### OpenCode

Add to `~/.config/opencode/config.json` under `mcpServers`:

```json
{
  "mcpServers": {
    "simul": {
      "command": "uv",
      "args": ["--directory", "/abs/path/to/simul", "run", "simul-mcp", "server"]
    }
  }
}
```

If you need a custom config file, append the extra server args after `server`,
for example:

```json
{
  "command": "uv",
  "args": [
    "--directory", "/abs/path/to/simul",
    "run", "simul-mcp", "server",
    "--config", "config/isaac/default.yaml"
  ]
}
```

### Prerequisites for Isaac Sim Tools

Isaac Sim tools prefer a running Isaac Sim instance with the repo-owned `khemoo.simul.mcp` bridge extension enabled on TCP port 8229. When that bridge is unavailable, the client falls back to `isaacsim.code_editor.vscode` on TCP port 8226. Use `ping_isaac` to verify connectivity.

When Isaac Sim runs in Docker, Simul MCP connects to the host-published ports, not the container-internal ports. With the included Compose file, the host-facing ports are controlled by `ISAAC_BRIDGE_PORT` and `ISAAC_VSCODE_PORT`, so the MCP server and agents should target those host values.

## Quick Start

### 1. Start the MCP Server

```bash
# Basic startup
uv run simul-mcp server

# With custom configuration
uv run simul-mcp server --config config/custom.yaml --verbose
```

### 2. Test USD Operations

```bash
# Analyze a USD file
simul-mcp usd info /path/to/scene.usd

# Check server capabilities
simul-mcp info

# View tool usage statistics
simul-mcp stats
```

### 3. Use with Isaac Sim

1. Launch Isaac Sim with the `khemoo.simul.mcp` bridge extension enabled
2. Keep `isaacsim.code_editor.vscode` enabled if you want transport fallback
3. Start your AI agent (Claude Code, Codex, or OpenCode) with simul MCP configured
4. The agent can now use 75+ Isaac Sim tools for scene control, rendering, physics, and more

### Containerized Isaac Sim 5.1.0

For Linux hosts, the repo includes a Docker Compose file that runs the official
`nvcr.io/nvidia/isaac-sim:5.1.0` image with the bridge extension mounted from
this checkout:

```bash
docker compose -f compose.isaac-sim.yml up -d
```

This Compose file:
- publishes the bridge and VS Code sockets from the container to the host
- mounts `./src/simul_mcp/bridge_ext/khemoo.simul.mcp` into `/tmp/extsUser/khemoo.simul.mcp`
- starts `/isaac-sim/isaac-sim.sh --allow-root --no-window`
- enables both `khemoo.simul.mcp` and `isaacsim.code_editor.vscode`
- binds the bridge inside the container on `0.0.0.0:${ISAAC_BRIDGE_PORT:-8229}`
- enables bridge `execute_script` by default for easy local use
- binds the VS Code fallback inside the container on `0.0.0.0:${ISAAC_VSCODE_PORT:-8226}`
- publishes those ports back to the host on the same numbers
- keeps the container stateless by default so validation runs start cleanly

Port-forwarding note:
- Simul MCP always talks to the host-visible ports.
- If the container publishes `127.0.0.1:9229` and `127.0.0.1:9226`, the MCP server must use `9229` / `9226`.
- Discovery files now include both the bridge port and the forwarded VS Code fallback port so multi-instance routing can distinguish local and containerized Isaac apps correctly.

To stop it:

```bash
docker compose -f compose.isaac-sim.yml down
```

Override ports or the image tag with standard Compose environment variables, for example:

```bash
ISAAC_BRIDGE_PORT=8829 ISAAC_VSCODE_PORT=8826 docker compose -f compose.isaac-sim.yml up -d
```

## Usage

### Command Line Interface

```bash
# Start the MCP server (all backends)
simul-mcp server

# Start with only specific backends (reduces agent context)
simul-mcp server --backends unreal
simul-mcp server --backends isaac,usd

# Start with custom configuration
simul-mcp server --config config/isaac/default.yaml

# Start with verbose logging
simul-mcp server --verbose

# Show server information and capabilities
simul-mcp info

# Test USD file loading and analysis
simul-mcp usd info /path/to/scene.usd

# Validate configuration file
simul-mcp validate-config config/isaac/default.yaml

# Show version information
simul-mcp version

# Isaac Sim commands
simul-mcp isaac ping
simul-mcp isaac status
simul-mcp isaac scene
simul-mcp isaac exec "print('hello')"

# Unreal Engine commands
simul-mcp unreal health
simul-mcp unreal list-actors --class StaticMeshActor
simul-mcp unreal spawn StaticMeshActor --location 0,0,100
simul-mcp unreal exec "print(unreal.EditorLevelLibrary.get_all_level_actors())"
simul-mcp unreal capture viewport.png --width 1920

# USD commands
simul-mcp usd info scene.usd
simul-mcp usd validate scene.usd
simul-mcp usd summary scene.usd
```

### Isaac Sim Extension

1. Open Isaac Sim
2. Go to Window → Extensions
3. Search for "Isaac Sim MCP Server"
4. Enable the extension
5. Use the MCP Server panel to start/stop the server

The extension provides:
- Server start/stop controls
- Configuration options (transport, log level)
- Real-time status monitoring
- Tool availability display
- Log viewer

### Python API

```python
from simul_mcp.mcp.server import SimulMCPServer
from simul_mcp.config import get_settings

# Create and run server
settings = get_settings()
server = SimulMCPServer(settings)
await server.run("stdio")
```

### Headless USD Operations

```python
from simul_mcp.adapters import HeadlessUSDAdapter

adapter = HeadlessUSDAdapter()
with adapter.create_session() as session:
    # Load USD file
    stage_id = session.load_stage("/path/to/scene.usd")

    # Get stage information
    stage_info = session.get_stage_info(stage_id)
    print(f"Stage has {stage_info.prim_count} prims")

    # Generate scene summary
    summary = session.summarize_stage(stage_id)
    print(f"Scene summary: {summary.total_prims} prims, {summary.hierarchy_depth} levels deep")

    # Find mesh prims
    meshes = session.find_prims_by_type(stage_id, "Mesh")
    print(f"Found {len(meshes)} mesh prims")

    # Analyze specific mesh
    if meshes:
        mesh_info = session.get_mesh_info(stage_id, meshes[0])
        print(f"Mesh has {mesh_info['vertex_count']} vertices, {mesh_info['face_count']} faces")
```

## Configuration

The server uses YAML configuration files. See `config/isaac/default.yaml` for all available options:

```yaml
server:
  name: "Simul - 3D Simulation & DCC Tools"

logging:
  level: "INFO"
  format: "detailed"
  file:
    enabled: true
    path: "logs/simul_mcp.log"
    max_size: "10MB"
    backup_count: 5
  console:
    enabled: true
    colored: true

usd:
  cache:
    enabled: true
    stage_cache_limit: 10
  files:
    max_file_size_mb: 500
    allowed_extensions: [".usd", ".usda", ".usdc", ".usdz"]

viewport:
  capture:
    width: 1920
    height: 1080
    max_size: 2048
    format: "png"

isaac_sim:
  path: "${ISAAC_SIM_PATH}"
  socket_host: "127.0.0.1"
  socket_port: 8226
  socket_timeout: 30.0
  bridge:
    enabled: true
    host: "127.0.0.1"
    port: 8229
    timeout: 30.0
    fallback_to_vscode: true
```

### Environment Variables

You can override configuration using environment variables:

```bash
export LOGGING__LEVEL=DEBUG
export USD__CACHE_ENABLED=false
export VIEWPORT__MAX_SIZE=4096
```

## MCP Tools

The server provides 75+ tools across multiple backends. Key tool categories:

### Headless USD (no runtime required)

`load_usd_file`, `validate_usd_file`, `get_prim_info`, `search_prims`, `summarize_scene`, `get_mesh_info`, `get_bounding_box`, `create_prim`, `delete_prim`, `update_prim_attributes`

### Isaac Sim — Scene Inspection

`get_isaac_stage_info`, `list_isaac_prims`, `get_isaac_prim_info`, `get_isaac_prim_transform`, `search_isaac_prims`, `get_isaac_scene_summary`, `get_isaac_subtree`, `get_isaac_prim_ancestors`, `get_isaac_prim_relationships`, `get_isaac_prim_variants`, `get_isaac_scene_stats`

### Isaac Sim — Prim Manipulation

`create_isaac_prim`, `delete_isaac_prim`, `set_isaac_prim_transform`, `set_isaac_prim_visibility`, `set_isaac_prim_attribute`, `duplicate_isaac_prim`, `reparent_isaac_prim`

### Isaac Sim — Viewport & Camera

`list_isaac_cameras`, `get_isaac_camera_info`, `set_isaac_camera`, `capture_isaac_viewport`, `focus_isaac_viewport`, `get_isaac_viewport_info`

### Isaac Sim — Physics

`get_isaac_physics_scene`, `create_isaac_physics_scene`, `get_isaac_rigid_body_info`, `add_isaac_rigid_body`, `add_isaac_collision`, `get_isaac_collision_info`, `get_isaac_joint_info`, `get_isaac_mass_properties`, `set_isaac_mass_properties`, `set_isaac_physics_material`, `list_isaac_physics_objects`

### Isaac Sim — Simulation Control

`get_isaac_simulation_state`, `start_isaac_simulation`, `pause_isaac_simulation`, `stop_isaac_simulation`, `step_isaac_simulation`, `reset_isaac_simulation`, `get_isaac_simulation_time`

### Isaac Sim — Materials

`get_isaac_material_info`, `list_isaac_materials`, `assign_isaac_material`, `set_isaac_material_property`, `create_isaac_material`

### Isaac Sim — Rendering & AOVs

`read_isaac_aovs`, `list_isaac_aovs`, `list_isaac_render_vars`, `get_isaac_carb_settings`, `set_isaac_carb_settings`

### Isaac Sim — USD Schema Queries

`query_isaac_typed_prims` — find prims by schema type (UsdLux, UsdGeom, UsdShade) and read attributes in one call

### Isaac Sim — Extensions & Assets

`list_isaac_extensions`, `enable_isaac_extension`, `disable_isaac_extension`, `open_isaac_stage`, `save_isaac_stage`, `new_isaac_stage`, `import_isaac_asset`, `add_isaac_reference`

### Isaac Sim — Advanced

`execute_isaac_script` (custom Python), `ping_isaac`, `raycast_isaac_scene`, `find_isaac_prims_in_area`, `get_isaac_texture_dependencies`, `list_isaac_instances`, `set_active_isaac_instance`

### Observability

`get_tool_usage_stats`, `reset_tool_usage_stats` — per-tool call counts, success rates, and durations via persistent JSONL log

### Blender (when runtime connected)

52 tools for scene objects, materials, rigid bodies, constraints, modifiers, mesh operations, animation, physics baking, viewport capture, and SimReady compliance.

### Unreal Engine Operations

Unreal Engine integration uses the built-in Remote Control HTTP API. The MCP server registers a thin tool set (3 tools) to minimize context overhead for AI agents; the full operation set is available via CLI.

**MCP Tools (always available):**
- `unreal_health_check`: Check connectivity to Unreal Engine
- `capture_unreal_viewport`: Capture viewport screenshot (returns image data)
- `execute_unreal_script`: Execute arbitrary Python inside the UE5 editor

**CLI Commands (`simul-mcp unreal ...`):**
- `health`, `info`, `scene`, `map` — inspection
- `list-actors`, `actor-info`, `search`, `scene-graph` — scene queries
- `spawn`, `delete`, `set-transform`, `set-property`, `set-visibility` — manipulation
- `sim`, `sim-status` — Play-In-Editor control
- `capture`, `exec`, `materials` — viewport, scripting, materials

#### Unreal Engine Setup

**Prerequisites:** Unreal Engine 5.x with a project open in the editor.

**Step 1 — Enable plugins** in your `.uproject` file:

```json
{
  "Plugins": [
    {"Name": "RemoteControl", "Enabled": true},
    {"Name": "PythonScriptPlugin", "Enabled": true}
  ]
}
```

**Step 2 — Configure Remote Control** in `Config/DefaultRemoteControl.ini`:

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

> **Important:** `bRestrictServerAccess=True` is required — the Python execution and
> console command settings are gated behind it. Without it, those features silently
> remain disabled even if set to `True`.

**Step 3 — Restart the Unreal Editor** to load the plugins and apply the config.

**Step 4 — Verify** the connection:

```bash
# Quick check
curl http://localhost:30010/remote/info

# Or via simul CLI
simul-mcp unreal health
```

#### Claude Code MCP Configuration

To use simul with Unreal Engine in Claude Code, add to your project's MCP config:

```bash
claude mcp add simul -- /path/to/.venv/bin/simul-mcp server --backends unreal
```

The `--backends unreal` flag registers only Unreal tools (3 MCP tools + 2 instance tools),
keeping agent context minimal. All other operations are available via `simul-mcp unreal <command>`.

## Examples

### Basic USD Analysis

```python
# examples/isaac/sample_usd_reader.py
python examples/isaac/sample_usd_reader.py /path/to/scene.usd --verbose
```

This example demonstrates:

- Loading USD files
- Extracting stage information
- Finding prims by type
- Computing bounding boxes
- Generating scene summaries
- Analyzing mesh statistics

### HTTP Client Example

```python
# examples/isaac/http_client_mcp.py
python examples/isaac/http_client_mcp.py --server http://localhost:8000 --usd-file /path/to/scene.usd
```

This example shows how to:

- Connect to MCP server via HTTP
- Call MCP tools programmatically
- Handle responses and errors
- Demonstrate both USD and Isaac Sim operations

### MCP Tool Usage

```python
import asyncio
from simul_mcp.mcp.server import SimulMCPServer

async def example():
    server = SimulMCPServer()

    # Load USD file
    result = await server.mcp.tools["load_usd_file"]("/path/to/scene.usd")
    if result.get("success", True):
        stage_id = result["stage_id"]

        # Get scene summary
        summary_result = await server.mcp.tools["summarize_scene"](
            stage_id=stage_id,
            include_meshes=True,
            format="text"
        )

        if summary_result.get("success", True):
            print(summary_result["digest"])

asyncio.run(example())
```

## Development

### Project Structure

```
simul-mcp/
├── pyproject.toml          # Project configuration
├── README.md              # This file
├── .env.example           # Environment variables template
├── .gitignore            # Git ignore rules
├── Makefile              # Development tasks
├── config/               # Configuration files
│   └── isaac/            # Isaac Sim configuration
│       ├── default.yaml  # Default configuration
│       ├── logging.yaml  # Logging configuration
│       └── kits/         # Isaac Sim Kit configurations
├── scripts/              # Shell scripts
│   └── isaac/            # Isaac Sim helpers
│       ├── run_kit_mcp.sh   # Linux/macOS launcher
│       ├── run_kit_mcp.ps1  # Windows launcher
│       └── dev_isort_black.sh # Code formatting
├── src/simul_mcp/        # Main source code
│   ├── __init__.py       # Package initialization
│   ├── config.py         # Configuration management
│   ├── logging.py        # Logging setup
│   ├── usd/             # USD operations
│   ├── adapters/        # Runtime adapters
│   ├── mcp/             # MCP server implementation
│   └── utils/           # Utility modules
├── src/simul_mcp/cli/    # Command-line interface
│   └── main.py          # CLI implementation
├── src/simul_mcp/bridge_ext/khemoo.simul.mcp/  # Isaac Sim extension
│                          # bundled in the wheel since v0.0.36;
│                          # publish via `simul-mcp isaac install-bridge`
├── tests/               # Test suite
│   └── isaac/           # Isaac Sim tests
├── examples/            # Example scripts
│   └── isaac/           # Isaac Sim examples
```

### Isaac Sim bridge extension setup

The `khemoo.simul.mcp` Kit extension that backs port 8229 is bundled
inside the `simul-mcp` Python wheel as of v0.0.36. After installing
the package (or pulling new repo commits) once per Isaac install:

```bash
# Publish the bundled bridge ext into Isaac's extsUser dir
ISAAC_SIM_PATH=~/isaac-sim-5.1.0 simul-mcp isaac install-bridge --symlink

# Then once per Isaac launch — auto-enables the ext + waits for port
simul-mcp isaac bridge-up
```

`--symlink` is recommended for editable / repo-checkout workflows so
future `git pull`s propagate without re-running `install-bridge`. See
`CLAUDE.md` for the full lifecycle and the `bridge-up` retry semantics.

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/isaac/test_reader.py

# Run with coverage
pytest --cov=simul_mcp tests/

# Run tests with verbose output
pytest -v tests/
```

### SimulationApp Smoke Test

```bash
# Run a minimal Isaac Sim smoke check
/isaac-sim/python.sh scripts/smoke_simulationapp.py
```

### Code Formatting

```bash
# Format code using the provided script
./scripts/isaac/dev_isort_black.sh

# Or directly
isort src/ tests/ examples/
black src/ tests/ examples/
```

### Development Tasks

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run linting + type checking
flake8 src/ tests/ && mypy src/

# Run tests
pytest tests/ -v

# Clean build artifacts
rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .coverage htmlcov/ .mypy_cache/
find . -type d -name __pycache__ -exec rm -rf {} +
```

## Architecture

The project follows a modular architecture with clear separation of concerns:

### Core Components

- **Configuration System**: Pydantic-based configuration with YAML support
- **Logging System**: Structured logging with multiple handlers and formatters
- **USD Operations**: Pure pxr-based USD file operations and analysis
- **Adapter Layer**: Abstraction between USD operations and runtime environments
- **MCP Server**: FastMCP-based server with tool registry and connection management

### Runtime Environments

1. **Headless Mode**: Uses pure pxr library for USD operations without GUI
2. **Isaac Sim Mode**: Full integration with Isaac Sim runtime for simulation and viewport operations

### Key Design Principles

- **Modularity**: Each component has a single responsibility
- **Extensibility**: Easy to add new tools and capabilities
- **Error Handling**: Comprehensive error handling with proper logging
- **Performance**: Caching and optimization for large USD files
- **Type Safety**: Full type hints and Pydantic validation

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite (`pytest`)
6. Run code formatting (`black src/ tests/ examples/ && isort src/ tests/ examples/`)
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write docstrings for all public functions and classes
- Add tests for new functionality
- Update documentation as needed
- Use meaningful commit messages

## Troubleshooting

### Common Issues

1. **USD Library Not Found**
   ```
   ImportError: pxr library not available
   ```
   Solution: Install USD Python bindings or run within Isaac Sim environment

2. **Isaac Sim Not Available**
   ```
   Isaac Sim runtime not available
   ```
   Solution: Run the server within Isaac Sim or use headless mode only

3. **Configuration Errors**
   ```
   ValidationError: Invalid configuration
   ```
Solution: Check configuration file syntax and validate with `simul-mcp validate-config`

4. **Port Already in Use**
   ```
   Address already in use
   ```
   Solution: Change the server port or stop the existing server

### Debug Mode

Enable debug logging for detailed troubleshooting:

```bash
simul-mcp server --log-level DEBUG
```

Or set environment variable:

```bash
export LOGGING__LEVEL=DEBUG
```

## License

This project is licensed under the MIT License. See LICENSE for details.

## Support

For issues and questions:

- GitHub Issues: https://github.com/kickthemoon0817/simul/issues
- Documentation: https://github.com/kickthemoon0817/simul/wiki
- Discussions: https://github.com/kickthemoon0817/simul/discussions

## Acknowledgments

- NVIDIA Isaac Sim team for the simulation platform
- Pixar for the USD format and libraries
- The MCP community for the protocol specification
- Contributors and users of this project
