# AGENTS.md — Simul

This file is canonical for repo layout, build/test commands, coding
conventions and runtime notes. `CLAUDE.md` holds Claude Code behavioral rules
(engine setup flows, release protocol, issue filing).

## Repository Layout

- Primary package: `src/simul` (import `simul`; distribution `simul-toolkit`;
  console script `simul`)
- CLI entrypoint: `src/simul/cli/main.py`
- MCP server: `src/simul/mcp/server.py`
- Backend registry: `src/simul/mcp/backends.py` (one `BackendSpec` per
  backend; the server iterates it for adapters, tool registration, the
  capability report and the instructions, which cover only the backends
  whose tools registered)
- Adapter interface: `src/simul/adapters/base.py` (`BackendAdapter`);
  adapters: `src/simul/adapters/`
- Isaac Sim tools: `src/simul/mcp/tools/isaac/` (per-domain mixins;
  each tool method carries its MCP metadata as a `@tool_meta` decorator from
  `src/simul/mcp/tools/_meta.py`). `src/simul/mcp/tools/isaac_tools.py`
  is the compatibility shim that re-exports the package.
- Tool registration: `src/simul/mcp/registration/` (`_reg_isaac.py`
  iterates the decorated methods; `_reg_blender.py` builds each Blender tool
  from its `BLENDER_TOOLS` entry and the `BlenderRuntimeSession` operation it
  names; each Unreal tool is one `UnrealTool` declaration in
  `_unreal_tools.py`, from which `_reg_unreal.py` builds the wrapper, request
  model and session call; the USD module registers its wrappers by hand over
  the adapter session)
- USD tools: `src/simul/mcp/registration/_reg_usd.py` over
  `src/simul/adapters/headless_usd.py`
- Tests: `tests/` (`tests/conftest.py` puts `src` first on `sys.path` and
  provides the shared `FakeFastMCP` double from `tests/fakes.py`; the Isaac
  live tier is `tests/isaac/live/`)
- Config: `src/simul/resources/config/default.yaml` (shipped in the wheel; environment
  variables override it per key), `.env.example`
- Packaged data (skills document, API docs, default + logging YAML): `src/simul/resources/`

## Build / Run

```sh
uv venv .venv && uv pip install -e ".[dev]"   # dev venv (or: uv sync --extra dev)
pip install -e .                              # install package (editable)
python -m build                               # build sdist + wheel
simul server                                  # MCP server (dev, stdio)
simul server --transport http                 # streamable HTTP on server.host:server.port
simul server --backends usd                   # MCP server (headless USD only)
simul server --unreal-tools full              # MCP server with every granular Unreal tool
```

The server runs in its own interpreter, never inside Isaac Sim's
`python.sh` (that interpreter ships without `fastmcp`, `typer` and
`pydantic-settings`); it reaches Isaac Sim over the socket transports. Set
`ISAAC_SIM_PATH` to your Isaac Sim install root so the `simul isaac`
commands can find the install (export it from your shell rc; the server
warns at startup if it's expected but not set).

## Format / Lint / Test

```sh
black src/ tests/ examples/ && isort src/ tests/ examples/   # format
flake8 src/ tests/ && mypy src/                              # lint (119 cols, see .flake8) + types
pytest tests/ -v                                              # unit + live (live skips if engine down); coverage is on via addopts
pytest tests/ -q --no-cov                                     # fast loop without coverage
pytest tests/isaac/live -v -m isaac                           # Isaac live (skips unless the socket answers)
pytest tests/ -v -m unreal_live                               # Unreal live (requires running editor)
pytest tests/packaging -m packaging                           # wheel build + install smoke (slow)
```

Run the suite with the checkout's source: `tests/conftest.py` puts `src`
first on `sys.path`, so `pytest tests/` from the repo root exercises the
files next to it whatever `simul` is installed in the interpreter.

## Versioning

The version lives in four constants that move together (`pyproject.toml`,
`.claude-plugin/plugin.json`, `src/simul/__init__.py`, the bridge ext's
`config/extension.toml`), and `uv lock` records it in `uv.lock`, so a bump
commit includes the regenerated lock; `tests/test_version_lockstep.py`
enforces both. CI installs with `uv sync --locked`, so re-run `uv lock`
whenever dependencies change too.

- **Patch (`0.X.Y` → `0.X.Y+1`) — use actively.** Bump the patch version
  whenever changes land on `main` (a merged fix or feature PR, or a batch of
  them). No approval, no git tag and no GitHub Release: the bump PR alone.
- **Minor (`0.X.Y` → `0.X+1.0`) — developer approval only.** Never bump the
  minor version on your own initiative; propose it and wait for an explicit
  yes from a developer. Only minor versions are tagged (`v0.X.0`), and each
  one gets a GitHub Release with release notes.

Procedure for both: `docs/releasing.md`.

## MCP Error Handling

- Tools return JSON-serializable dicts.
- On error, return `ErrorResponse(...).model_dump()`.
- Log errors with context before returning.

## Runtime Notes

- Unreal attached mode: `simul unreal instances` / `attach` select an existing
  editor, map and level viewport. `simul server --backends unreal --unreal-mode attached`
  uses that selection; `control_unreal_ui` provides named editor controls even with
  scripting disabled. Optional `simul unreal setup <project> --agent-overlay --yes`
  builds the native Slate overlay with Unreal's C++ toolchain. Use unique `agent_id`
  labels and `move_cursor` for visible activity; never move the OS cursor.
  See `docs/unreal-attachment.md`.

- Blender attached mode: `simul blender install-bridge` builds an add-on ZIP to enable
  in the existing editor, then `simul blender instances` / `attach` select its window.
  Run `simul server --backends blender --blender-mode attached`; no server-side bpy is
  required. See `docs/blender-attachment.md`. `SIMUL_BLENDER_LIVE=1 pytest
  tests/blender/test_live_attach.py -m blender_live` launches and closes a disposable GUI.
- Isaac Sim tools require a running Isaac Sim instance (5.1.0, 6.0.0, or 6.0.1) with the bridge on port 8229 or the stock Python socket on port 8226. `simul isaac launch` starts one with both enabled.
- Isaac Sim runs only on Linux and Windows, so by default the server registers
  the Isaac tools (plus the instance tools, the `simul://isaac-sim/*` resources
  and the Isaac instructions) only there; on macOS `simul server`, `info` and
  `tools` report them as skipped. `ISAAC_SIM__ENABLED=true` or
  `--backends isaac` registers them on any OS (for unsupported setups such as
  an SSH tunnel to an Isaac host); `ISAAC_SIM__ENABLED=false` never does. The
  gate is `skip_check` on the Isaac `BackendSpec`, reading the
  `is_isaac_platform_supported` probe; the `simul isaac` commands ignore it.
- Headless USD tools work without Omniverse.
- `tests/isaac/live/` is the `@pytest.mark.isaac` tier: ping, stage info,
  create/delete prim, viewport capture, AOV reads and the bridge extension
  toggle against a running Isaac Sim. It skips unless the configured socket
  answers (the probe `simul isaac ping` runs) and never starts the engine.
- Adding a backend: implement `BackendAdapter` (`adapters/base.py`), write
  its registration module under `mcp/registration/`, and add one
  `BackendSpec` to `mcp/backends.py`. The server, the CLI's `info` grouping,
  `get_capabilities` and the ROUTING instructions pick it up from there.
- `src/simul/resources/skills.md` documents Isaac Sim 5.1 / 6.0 scripting patterns for
  `execute_isaac_script`; the server exposes it as the `simul://isaac-sim/skills` resource.
