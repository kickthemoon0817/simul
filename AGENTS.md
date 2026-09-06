# AGENTS.md — Simul MCP Server

Coding standards and conventions are defined in CLAUDE.md. This file contains
repo-specific build commands, project layout, and runtime notes only.

## Repository Layout

- Primary package: `src/simul_mcp`
- CLI entrypoint: `src/simul_mcp/cli/main.py`
- MCP server: `src/simul_mcp/mcp/server.py`
- Backend registry: `src/simul_mcp/mcp/backends.py` (one `BackendSpec` per
  backend; the server iterates it for adapters, tool registration, the
  capability report and the ROUTING instructions)
- Adapter interface: `src/simul_mcp/adapters/base.py` (`BackendAdapter`);
  adapters: `src/simul_mcp/adapters/`
- Isaac Sim tools: `src/simul_mcp/mcp/tools/isaac/` (per-domain mixins;
  each tool method carries its MCP metadata as a `@tool_meta` decorator from
  `src/simul_mcp/mcp/tools/_meta.py`). `src/simul_mcp/mcp/tools/isaac_tools.py`
  is the compatibility shim that re-exports the package.
- Tool registration: `src/simul_mcp/mcp/registration/` (`_reg_isaac.py`
  iterates the decorated methods; the USD, Blender and Unreal modules
  register their wrappers by hand over the adapter sessions)
- USD tools: `src/simul_mcp/mcp/registration/_reg_usd.py` over
  `src/simul_mcp/adapters/headless_usd.py`
- Tests: `tests/` (`tests/conftest.py` puts `src` first on `sys.path` and
  provides the shared `FakeFastMCP` double from `tests/fakes.py`; the Isaac
  live tier is `tests/isaac/live/`)
- Config: `src/simul_mcp/resources/config/default.yaml` (shipped in the wheel; environment
  variables override it per key), `.env.example`
- Packaged data (skills document, API docs, default + logging YAML): `src/simul_mcp/resources/`

## Build / Run

```sh
pip install -e .                              # install package (editable)
pip install -e ".[dev]"                       # + dev deps
python -m build                               # build sdist + wheel
simul-mcp server                              # MCP server (dev, stdio)
simul-mcp server --transport http             # streamable HTTP on server.host:server.port
simul-mcp server --backends usd               # MCP server (headless USD only)
simul-mcp server --unreal-tools full          # MCP server with every granular Unreal tool
```

The server runs in its own interpreter, never inside Isaac Sim's
`python.sh` (that interpreter ships without `fastmcp`, `typer` and
`pydantic-settings`); it reaches Isaac Sim over the socket transports. Set
`ISAAC_SIM_PATH` to your Isaac Sim install root so the `simul-mcp isaac`
commands can find the install (export it from your shell rc; the server
warns at startup if it's expected but not set).

## Format / Lint / Test

```sh
black src/ tests/ examples/ && isort src/ tests/ examples/   # format
flake8 src/ tests/ && mypy src/                              # lint (119 cols, see .flake8) + types
pytest tests/ -v                                              # unit + live (live skips if engine down)
pytest tests/ -v --cov=simul_mcp                              # with coverage
pytest tests/isaac/live -v -m isaac                           # Isaac live (skips unless the socket answers)
pytest tests/ -v -m unreal_live                               # Unreal live (requires running editor)
pytest tests/packaging -m packaging                           # wheel build + install smoke (slow)
```

Run the suite with the checkout's source: `tests/conftest.py` puts `src`
first on `sys.path`, so `pytest tests/` from the repo root exercises the
files next to it whatever `simul-mcp` is installed in the interpreter.

## MCP Error Handling

- Tools return JSON-serializable dicts.
- On error, return `ErrorResponse(...).dict()`.
- Log errors with context before returning.

## Runtime Notes

- Isaac Sim tools require a running Isaac Sim instance (5.1.0, 6.0.0, or 6.0.1) with the bridge on port 8229 or the stock Python socket on port 8226. `simul-mcp isaac launch` starts one with both enabled.
- Headless USD tools work without Omniverse.
- `tests/isaac/live/` is the `@pytest.mark.isaac` tier: ping, stage info,
  create/delete prim, viewport capture, AOV reads and the bridge extension
  toggle against a running Isaac Sim. It skips unless the configured socket
  answers (the probe `simul-mcp isaac ping` runs) and never starts the engine.
- Adding a backend: implement `BackendAdapter` (`adapters/base.py`), write
  its registration module under `mcp/registration/`, and add one
  `BackendSpec` to `mcp/backends.py`. The server, the CLI's `info` grouping,
  `get_capabilities` and the ROUTING instructions pick it up from there.
- `src/simul_mcp/resources/skills.md` documents Isaac Sim 5.1 / 6.0 scripting patterns for
  `execute_isaac_script`; the server exposes it as the `simul://isaac-sim/skills` resource.
