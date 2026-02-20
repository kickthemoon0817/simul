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
- **Flexible Architecture**: Works in both headless and Isaac Sim runtime environments
- **Comprehensive Logging**: Structured logging with multiple output formats
- **Configuration Management**: YAML-based configuration with environment variable support

## Requirements

### Core Requirements
- Python 3.8+
- USD Python bindings (pxr)
- FastMCP
- Pydantic v2
- PyYAML
- NumPy

### Isaac Sim Requirements (Optional)
- NVIDIA Isaac Sim 2023.1.1+
- Omniverse Kit SDK
- Isaac Sim Python environment

## Installation

### Development Installation

```bash
# Clone the repository
git clone https://github.com/khemoo/simul-mcp.git
cd simul-mcp

# Install in development mode
pip install -e .

# Or install with all dependencies
pip install -e ".[dev,test]"
```

### Isaac Sim Installation

1. Install NVIDIA Isaac Sim
2. Copy the extension to Isaac Sim's extensions directory:
   ```bash
   cp -r exts/khemoo.simul.mcp $ISAAC_SIM_PATH/exts/
   ```
3. Enable the extension in Isaac Sim's Extension Manager

## Quick Start

### 1. Start the MCP Server

```bash
# Basic startup
simul-mcp server

# With custom configuration
simul-mcp server --config config/custom.yaml --verbose
```

### 2. Test USD Operations

```bash
# Analyze a USD file
simul-mcp test-usd examples/isaac/sample.usd --verbose

# Check server capabilities
simul-mcp info
```

### 3. Use in Isaac Sim

1. Launch Isaac Sim
2. Enable the "Isaac Sim MCP Server" extension
3. Use the MCP Server panel to control the server
4. Connect your AI model to the MCP server

## Usage

### Command Line Interface

```bash
# Start the MCP server
simul-mcp server

# Start with custom configuration
simul-mcp server --config config/custom.yaml

# Start with verbose logging
simul-mcp server --verbose

# Show server information and capabilities
simul-mcp info

# Test USD file loading and analysis
simul-mcp test-usd scene.usd --verbose

# Validate configuration file
simul-mcp validate-config config/isaac/default.yaml

# Show version information
simul-mcp version
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
from simul_mcp.mcp.server import IsaacMCPServer
from simul_mcp.config import get_settings

# Create and run server
settings = get_settings()
server = IsaacMCPServer(settings)
await server.run("stdio")
```

### Headless USD Operations

```python
from simul_mcp.adapters import HeadlessUSDAdapter

adapter = HeadlessUSDAdapter()
with adapter.create_session() as session:
    # Load USD file
    stage_id = session.load_stage("scene.usd")

    # Get stage information
    stage_info = session.get_stage_info(stage_id)
    print(f"Stage has {len(stage_info.all_prims)} prims")

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
  name: "simul-mcp"
  version: "1.0.0"
  description: "Isaac Sim MCP Server"

logging:
  level: "INFO"
  format: "detailed"
  handlers:
    - "console"
    - "file"
  file:
    path: "logs/simul_mcp.log"
    max_size_mb: 10
    backup_count: 5

usd:
  cache_enabled: true
  max_file_size_mb: 500
  allowed_extensions: [".usd", ".usda", ".usdc", ".usdz"]

viewport:
  max_size: 2048
  default_format: "png"

isaac_sim:
  auto_initialize_world: true
  default_physics_dt: 0.016667  # 60 FPS
  default_rendering_dt: 0.016667
```

### Environment Variables

You can override configuration using environment variables:

```bash
export SIMUL_MCP_LOG_LEVEL=DEBUG
export SIMUL_MCP_USD_CACHE_ENABLED=false
export SIMUL_MCP_VIEWPORT_MAX_SIZE=4096
```

## MCP Tools

The server provides the following MCP tools:

### USD File Operations

- `load_usd_file`: Load a USD file and return stage information
- `validate_usd_file`: Validate a USD file without loading it

### USD Scene Operations

- `get_prim_info`: Get detailed information about a USD prim
- `search_prims`: Search for prims by type or name pattern
- `summarize_scene`: Generate comprehensive scene summaries for LLM consumption

### USD Mesh Operations

- `get_mesh_info`: Get detailed mesh information and statistics including topology validation

### USD Bounding Box Operations

- `get_bounding_box`: Compute bounding boxes for prims or entire stages in world or local space

### Isaac Sim Operations (when available)

- `capture_viewport`: Capture the Isaac Sim viewport as an image
- `control_simulation`: Control simulation playback (play, pause, stop, reset, step)
- `set_camera_view`: Set the viewport camera position and orientation
- `focus_on_prim`: Focus the camera on a specific prim
- `get_simulation_status`: Get current simulation status
- `get_viewport_info`: Get viewport information and capabilities
- `get_camera_info`: Get camera information and capabilities

### Blender Operations (when available)

- `get_blender_info`: Get active Blender runtime details through `bpy`
- `list_blender_scene_objects`: List objects in the active Blender scene or collection

## Examples

### Basic USD Analysis

```python
# examples/isaac/sample_usd_reader.py
python examples/isaac/sample_usd_reader.py scene.usd --verbose
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
python examples/isaac/http_client_mcp.py --server http://localhost:8000 --usd-file scene.usd
```

This example shows how to:

- Connect to MCP server via HTTP
- Call MCP tools programmatically
- Handle responses and errors
- Demonstrate both USD and Isaac Sim operations

### MCP Tool Usage

```python
import asyncio
from simul_mcp.mcp.server import IsaacMCPServer

async def example():
    server = IsaacMCPServer()

    # Load USD file
    result = await server.mcp.tools["load_usd_file"]("scene.usd")
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
├── exts/                # Isaac Sim extension
│   └── khemoo.simul.mcp/ # Extension package
├── tests/               # Test suite
│   └── isaac/           # Isaac Sim tests
├── examples/            # Example scripts
│   └── isaac/           # Isaac Sim examples
```

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

# Or use make
make format

# Or manually
isort src/ tests/ examples/
black src/ tests/ examples/
```

### Development Tasks

```bash
# Install development dependencies
pip install -e ".[dev,test]"

# Run linting
make lint

# Run type checking
make typecheck

# Run all checks
make check

# Clean build artifacts
make clean
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
6. Run code formatting (`make format`)
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
export SIMUL_MCP_LOG_LEVEL=DEBUG
```

## License

This project is licensed under the MIT License. See LICENSE for details.

## Support

For issues and questions:

- GitHub Issues: https://github.com/khemoo/simul-mcp/issues
- Documentation: https://github.com/khemoo/simul-mcp/wiki
- Discussions: https://github.com/khemoo/simul-mcp/discussions

## Acknowledgments

- NVIDIA Isaac Sim team for the simulation platform
- Pixar for the USD format and libraries
- The MCP community for the protocol specification
- Contributors and users of this project
