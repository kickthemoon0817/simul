# Isaac Sim MCP Server Extension

This extension provides an integrated MCP (Model Context Protocol) server for NVIDIA Isaac Sim, enabling AI models to interact with USD scenes and simulation environments directly within Isaac Sim.

## Features

- **Integrated MCP Server**: Run MCP server directly within Isaac Sim
- **UI Control Panel**: Start/stop server with configuration options
- **Real-time Monitoring**: Server status and tool availability display
- **Log Viewer**: View server logs within Isaac Sim
- **Lifecycle Management**: Automatic server startup/shutdown with extension

## Installation

### Method 1: Copy Extension

1. Copy the extension directory to Isaac Sim's extensions folder:
   ```bash
cp -r exts/khemoo.simul.mcp $ISAAC_SIM_PATH/exts/
   ```

2. Launch Isaac Sim

3. Go to Window → Extensions

4. Search for "Isaac Sim MCP Server"

5. Enable the extension

### Method 2: Development Mode

1. Launch Isaac Sim

2. Go to Window → Extensions

3. Click the gear icon and add the extension path:
   ```
/path/to/simul-mcp/exts
   ```

4. Search for "Isaac Sim MCP Server" and enable it

## Usage

### UI Panel

Once enabled, the extension adds an "MCP Server" panel to Isaac Sim:

1. **Server Controls**:
   - Start/Stop button
   - Server status indicator
   - Configuration options

2. **Transport Selection**:
   - STDIO: Standard input/output (default)
   - SSE: Server-Sent Events (HTTP)

3. **Log Level**:
   - DEBUG: Detailed debugging information
   - INFO: General information (default)
   - WARNING: Warning messages only
   - ERROR: Error messages only

4. **Status Display**:
   - Server running status
   - Available tools count
   - Isaac Sim capabilities

5. **Log Viewer**:
   - Real-time server logs
   - Copy/clear functionality

### Starting the Server

1. Open the MCP Server panel
2. Select transport type (STDIO or SSE)
3. Choose log level
4. Click "Start Server"
5. Monitor status and logs

### Connecting AI Models

Once the server is running, AI models can connect using the selected transport:

#### STDIO Transport
Connect via standard input/output (most common for MCP):
```python
# AI model connects to Isaac Sim process stdio
```

#### SSE Transport
Connect via HTTP (useful for web-based AI models):
```python
import aiohttp

async with aiohttp.ClientSession() as session:
    async with session.post('http://localhost:8000/tools/load_usd_file',
                           json={'file_path': 'scene.usd'}) as response:
        result = await response.json()
```

## Available Tools

When running in Isaac Sim, the server provides both USD and Isaac Sim specific tools:

### USD Operations
- Load and analyze USD files
- Search for prims by type or name
- Get detailed prim information
- Compute bounding boxes
- Analyze mesh statistics
- Generate scene summaries

### Isaac Sim Operations
- Capture viewport images
- Control simulation (play/pause/stop/reset)
- Set camera positions
- Focus on specific prims
- Get simulation status
- Access viewport information

## Configuration

The extension uses the same configuration system as the standalone server. Configuration files are loaded from:

1. `config/isaac/default.yaml` (default settings)
2. Environment variables (overrides)
3. UI panel settings (runtime overrides)

### Key Settings

```yaml
server:
name: "simul-mcp"
  description: "Isaac Sim MCP Server Extension"

logging:
  level: "INFO"
  handlers: ["console", "file"]

isaac_sim:
  auto_initialize_world: true
  default_physics_dt: 0.016667
  default_rendering_dt: 0.016667

viewport:
  max_size: 2048
  default_format: "png"
```

## Development

### Extension Structure

```
exts/khemoo.simul.mcp/
├── config/extension.toml
├── docs/README.md
├── khemoo/simul/mcp/
│   ├── __init__.py
│   └── impl/
│       ├── extension.py
│       ├── ui.py
│       └── lifecycle.py
```

### Key Components

1. **IsaacMCPServerExtension**: Main extension class that manages UI and server lifecycle
2. **MCPServerPanel**: UI panel with controls and status display
3. **ServerLifecycleManager**: Handles server startup, shutdown, and monitoring
4. **UI Widgets**: Individual UI components for different functions

### Debugging

Enable debug logging to troubleshoot issues:

1. Set log level to DEBUG in the UI panel
2. Check the log viewer for detailed information
3. Monitor the Isaac Sim console for extension-specific messages

## Troubleshooting

### Extension Not Loading

1. Check Isaac Sim version compatibility
2. Verify extension path is correct
3. Check Isaac Sim console for error messages
4. Ensure all dependencies are available

### Server Not Starting

1. Check if MCP components are available
2. Verify configuration file syntax
3. Check for port conflicts (SSE transport)
4. Review server logs in the log viewer

### UI Not Responding

1. Check if extension is properly enabled
2. Restart Isaac Sim if UI becomes unresponsive
3. Check for Python errors in Isaac Sim console

## License

This extension is part of the Isaac Sim MCP Server project and is licensed under the MIT License.
