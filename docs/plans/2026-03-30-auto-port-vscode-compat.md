# Auto-Port Allocation & VS Code Protocol Compatibility

> **For Claude:** Use `${SUPERPOWERS_SKILLS_ROOT}/skills/collaboration/executing-plans/SKILL.md` to implement this plan task-by-task.

**Goal:** Enable zero-config multi-instance Isaac Sim by adding auto-port allocation to the bridge extension and integrating VS Code raw-script protocol compatibility, while keeping the stock VS Code extension as a last-resort fallback.

**Architecture:** The bridge extension tries its configured port, auto-increments on conflict, writes a discovery file, and accepts both bridge (length-prefixed JSON) and VS Code (raw Python) protocols on the same port. The MCP server reads discovery files for fast instance finding, falls back to port scanning. The socket client prefers bridge for all operations, falls back to VS Code extension only as emergency.

**Tech Stack:** Python 3.11, asyncio, Isaac Sim Kit API (carb.settings), JSON discovery files

---

## Agent 1: Extension — Auto-Port + VS Code Compat Protocol

### Files:
- Modify: `exts/khemoo.simul.mcp/khemoo/simul/mcp/lifecycle.py`
- Modify: `exts/khemoo.simul.mcp/khemoo/simul/mcp/extension.py`
- Modify: `exts/khemoo.simul.mcp/config/extension.toml`

### Changes:

**lifecycle.py:**
- Add auto-port retry loop in `start()` — try configured port, on OSError increment, up to `max_retries` attempts
- Expose `self._port` as the actual bound port (may differ from configured)
- Add VS Code protocol detection in `_handle_client()` — peek first 4 bytes, if they look like a valid length prefix (uint32 < max_request_bytes), use bridge protocol; otherwise treat as raw Python code
- Add `_handle_vscode_client()` for raw Python execution path — read all bytes, execute via a callback, return VS Code JSON format `{"status":"ok","output":"..."}`
- Add discovery file write after successful bind
- Add discovery file cleanup in `stop()`

**extension.py:**
- After `_server.start()`, read actual bound port from server, write back to Carb settings
- Add `_write_discovery_file()` and `_remove_discovery_file()` methods
- Pass `executor` callback to lifecycle for VS Code compat path
- Add `discovery_dir` and `max_port_retries` to settings reload
- Cleanup discovery file in `on_shutdown()`

**extension.toml:**
- Add `exts."khemoo.simul.mcp".max_port_retries = 10`
- Add `exts."khemoo.simul.mcp".discovery_dir = "/tmp/simul-mcp"`

## Agent 2: MCP Server — Discovery File Reader + Simplified Client

### Files:
- Modify: `src/simul_mcp/mcp/server.py`
- Modify: `src/simul_mcp/config.py`
- Modify: `config/isaac/default.yaml`

### Changes:

**server.py:**
- Add `_discover_from_files()` method that reads `/tmp/simul-mcp/*.json`, validates PID is alive, builds clients
- In `_scan_isaac_instances()`, call `_discover_from_files()` first, then fall back to port scanning for remaining
- Simplify `_get_instance_brief()` to prefer bridge-only path

**config.py:**
- Add `discovery_dir: str = "/tmp/simul-mcp"` to `IsaacSimConfig`

**default.yaml:**
- Add `discovery_dir` under isaac_sim section

## Agent 3: Socket Client — Unified Transport Priority

### Files:
- Modify: `src/simul_mcp/adapters/isaac_socket_client.py`

### Changes:
- Add `execute_via_bridge_raw()` method that sends code via bridge `execute_script` action (typed envelope, not raw TCP)
- Update `execute()` priority chain: (1) bridge typed execute_script, (2) VS Code extension fallback
- Keep `fallback_to_vscode` as emergency escape hatch
- The client no longer needs separate bridge vs vscode host/port for the same instance — bridge IS the primary port
