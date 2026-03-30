# Review Findings Resolution Plan

> **For Claude:** Use `${SUPERPOWERS_SKILLS_ROOT}/skills/collaboration/executing-plans/SKILL.md` to implement this plan task-by-task.

**Goal:** Resolve all 34 findings from the 8-agent code review — 5 critical, 10 high, 19 medium.

**Architecture:** Fixes grouped by file cluster for parallel execution. Each agent owns its files exclusively to avoid conflicts.

**Tech Stack:** Python 3.11, asyncio, Pydantic v2, FastMCP, Typer, Docker Compose

---

## Agent 1: Socket Client (C1, H1, H2, H3, M1-M5)

**Files:** `src/simul_mcp/adapters/isaac_socket_client.py`

- C1: Wrap `ping()` body in `async with self._lock:`
- H1: Catch `asyncio.IncompleteReadError` in `_bridge_request`, convert to `ConnectionError`
- H3: Lazy-init the lock via property for Python <3.10 safety
- M1: Add `logger.debug` in bridge cleanup `except` block (match vscode pattern)
- M2: Add `_bridge_configured` property, use in address/bridge_address/`_bridge_request`
- M3: Use `if x is not None` instead of `or` for `bridge_timeout_seconds`
- M4: Validate `request_id` in bridge response
- M5: Use deadline-based overall read timeout in vscode read loop
- H2: Document write_eof decision (VS Code extension reads until connection close, not EOF)

## Agent 2: Config (C3, H4, H5, H6, M6, M7, M8, M9)

**Files:** `src/simul_mcp/config.py`

- C3: Change `lru_cache(maxsize=None)` to `maxsize=1`
- H4: Replace `print()` with `logger.warning()` in `_load_yaml_settings`
- H5: In `IsaacInstanceConfig`, add note that `None` bridge fields inherit from global config (resolution happens in server.py)
- H6: Add `@model_validator(mode='after')` on `IsaacSimConfig` for `scan_port_start < scan_port_end`
- M7: Add `model_config = ConfigDict(frozen=True)` to key models
- M8: Validate `_normalise_isaac_instances` input types, raise clear ValueError
- M9: Remove duplicate Isaac path checks from `validate_settings`
- Fix lowercase `tuple[...]` to `Tuple[...]` for 3.8 compat

## Agent 3: Isaac Tools (C2, H7-partial, M10-partial, M11-partial)

**Files:** `src/simul_mcp/mcp/tools/isaac_tools.py`

- C2: Fix logic inversion — when `transport_mode="default"`, always call `self._client.execute(script)`
- Replace `getattr(self._client, "bridge_enabled", False)` with `self._client.bridge_enabled`
- Remove unused `Union` import
- Convert `_raw_script_transport_mode` to `@property`
- Update `_execute_json_script` docstring to document `transport_mode` parameter

## Agent 4: CLI (M12, M13, M14, M15, M16-partial)

**Files:** `src/simul_mcp/cli/isaac.py`

- M12: Fix unconditional `emit_error` at line 1021 — add `if is_json_mode()` guard
- M13: Combine `scene` command's 5 sequential `asyncio.run()` into single `asyncio.gather()`
- M13b: Combine `status` command's 2 sequential calls into `asyncio.gather()`
- M14: Remove unused `--type` option from `search-prims` command (not passed to tool)
- M15: Add vector length validation for `--eye`, `--target` etc.
- Fix `exec_script` timeout type annotation: `float` → `Optional[float]`
- Fix `_tools()` to use `if x is not None` instead of `or` for host/port/timeout
- Fix `-H` short flag clash: remove `-H` from `read-aovs --height`
- Fix `exec_script` to use `emit()` instead of raw `print(json.dumps(...))`

## Agent 5: MCP Server (H8, H9, M17)

**Files:** `src/simul_mcp/mcp/server.py`, `src/simul_mcp/mcp/registration/_reg_isaac.py`

- H8: Add `async def shutdown()` method + wire into `run()` finally block
- H9: Parallelize `_scan_isaac_instances` with `asyncio.gather()`
- M17: Guard vscode fallback in `_get_instance_brief` against `fallback_to_vscode` setting
- Add `ValueError` to `_get_instance_brief` exception handler
- Add bounds check in `_bridge_port_for_socket`

## Agent 6: Security/Docker (C4, C5)

**Files:** `compose.isaac-sim.yml`, `config/isaac/default.yaml`

- C5: Change Docker compose to bind `127.0.0.1` instead of `0.0.0.0`, use non-root user
- C4: Add `# SECURITY` comments documenting the auth-less trust model
- Tighten CORS defaults from `localhost:*` to specific port
