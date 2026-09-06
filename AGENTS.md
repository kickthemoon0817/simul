# AGENTS.md — Simul MCP Server

Coding standards and conventions are defined in CLAUDE.md. This file contains
repo-specific build commands, project layout, and runtime notes only.

## Repository Layout

- Primary package: `src/simul_mcp`
- CLI entrypoint: `src/simul_mcp/cli/main.py`
- MCP server: `src/simul_mcp/mcp/server.py`
- Isaac Sim tools: `src/simul_mcp/mcp/tools/isaac_tools.py`
- USD tools: `src/simul_mcp/mcp/tools/usd_tools.py`
- Adapters: `src/simul_mcp/adapters/`
- Tests: `tests/`
- Config: `src/simul_mcp/resources/config/default.yaml` (shipped in the wheel; environment
  variables override it per key), `.env.example`
- Packaged data (skills document, API docs, default + logging YAML): `src/simul_mcp/resources/`

## Build / Run

```sh
pip install -e .                              # install package (editable)
pip install -e ".[dev]"                       # + dev deps
python -m build                               # build sdist + wheel
simul-mcp server                              # MCP server (dev)
simul-mcp server --backends usd               # MCP server (headless USD only)
$ISAAC_SIM_PATH/python.sh -m simul_mcp.cli.main server   # MCP server via Isaac Sim python.sh
```

Set `ISAAC_SIM_PATH` to your Isaac Sim install root before running the
Isaac variant (export it from your shell rc; the simul MCP server warns
at startup if it's expected but not set).

## Format / Lint / Test

```sh
black src/ tests/ examples/ && isort src/ tests/ examples/   # format
flake8 src/ tests/ && mypy src/                              # lint + types
pytest tests/ -v                                              # unit + live (live skips if engine down)
pytest tests/ -v --cov=simul_mcp                              # with coverage
pytest tests/ -v -m isaac                                     # Isaac live (requires runtime)
pytest tests/ -v -m unreal_live                               # Unreal live (requires running editor)
```

## MCP Error Handling

- Tools return JSON-serializable dicts.
- On error, return `ErrorResponse(...).dict()`.
- Log errors with context before returning.

## Runtime Notes

- Isaac Sim tools require a running Isaac Sim instance (5.1.0, 6.0.0, or 6.0.1) with the bridge on port 8229 or the stock Python socket on port 8226. `simul-mcp isaac launch` starts one with both enabled.
- Headless USD tools work without Omniverse.
- Some tests are marked `@pytest.mark.isaac` and need Isaac Sim runtime.
- `src/simul_mcp/resources/skills.md` documents Isaac Sim 5.1 / 6.0 scripting patterns for
  `execute_isaac_script`; the server exposes it as the `simul://isaac-sim/skills` resource.
