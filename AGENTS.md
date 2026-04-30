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
- Config: `config/isaac/default.yaml`, `.env.example`

## Build / Run

```sh
make install          # Install package
make install-dev      # Install dev deps
make build            # python -m build
make run-server       # MCP server (dev)
make run-headless     # MCP server (headless USD only)
make run-isaac        # MCP server via Isaac Sim python.sh
```

## Format / Lint / Test

```sh
make format           # black + isort
make lint             # flake8 + mypy
make test             # pytest tests/ -v
make test-cov         # pytest with coverage
make test-isaac       # pytest -m isaac (requires runtime)
make check            # format + lint + test
```

## MCP Error Handling

- Tools return JSON-serializable dicts.
- On error, return `ErrorResponse(...).dict()`.
- Log errors with context before returning.

## Runtime Notes

- Isaac Sim tools require a running Isaac Sim instance (TCP socket on port 8226).
- Headless USD tools work without Omniverse.
- Some tests are marked `@pytest.mark.isaac` and need Isaac Sim runtime.
- `skills.md` documents Isaac Sim 5.1.0 scripting patterns for `execute_isaac_script`.
