# Development

How the code is laid out, how the pieces fit, and how to test a change. For
the contribution workflow (branches, commits, PRs) see
[CONTRIBUTING.md](../CONTRIBUTING.md).

## Setup

```bash
git clone https://github.com/kickthemoon0817/simul.git
cd simul
uv venv .venv && uv pip install -e ".[dev]"
# or reproduce CI exactly from uv.lock:
uv sync --extra dev
```

Add `--extra blender` (or `".[dev,blender]"`) to test embedded Blender; `bpy`
wheels exist for Python 3.11 and 3.13 only.

## Repository layout

```
.
├── src/simul/
│   ├── cli/                 # Typer CLI: main.py plus isaac, unreal, blender, usd subcommands
│   ├── config.py            # Pydantic settings: YAML + SECTION__KEY environment
│   ├── logging.py           # Log handlers, audit stream
│   ├── mcp/
│   │   ├── server.py        # SimulMCPServer (FastMCP), resources, routing instructions
│   │   ├── backends.py      # One BackendSpec per backend
│   │   ├── registration/    # _reg_isaac.py, _reg_unreal.py, _reg_blender.py, _reg_usd.py, ...
│   │   └── tools/isaac/     # Isaac tool mixins, each method decorated with @tool_meta
│   ├── adapters/            # BackendAdapter implementations and transport clients
│   ├── usd/                 # pxr-based reader, bounding boxes, mesh ops, summaries
│   ├── blender_bridge/      # Add-on built by `simul blender install-bridge`
│   ├── bridge_ext/khemoo.simul/   # Isaac Sim Kit extension, shipped in the wheel
│   ├── resources/           # skills.md, docs/api/*.md, config/default.yaml, Unreal overlay plugin
│   └── utils/
├── tests/                   # Unit tests and live tiers (see below)
├── skills/, commands/       # Claude Code plugin skills and /simul:setup
├── .claude-plugin/          # Plugin manifest
├── docs/                    # This documentation
├── examples/                # Sample scripts and example galleries
├── scripts/isaac/           # Kit launchers, SimulationApp smoke test, formatter helper
├── config/isaac/kits/       # Kit app configurations
└── compose.isaac-sim.yml    # Containerized Isaac Sim
```

## Architecture

- **Settings** (`config.py`): a single `Settings` object built from the YAML
  file and the environment; see [configuration.md](configuration.md).
- **Backend registry** (`mcp/backends.py`): each backend is one `BackendSpec`
  naming its adapter, availability probe and registration module. The server
  iterates the registry for adapters, tool registration, the capability report
  (`get_capabilities`, `simul info`) and the routing instructions it sends
  to clients.
- **Adapters** (`adapters/`): implement `BackendAdapter` from
  `adapters/base.py`. Isaac Sim uses a TCP client with bridge-then-socket
  fallback; Unreal uses Remote Control HTTP; Blender runs `bpy` in-process or
  talks to the add-on bridge; headless USD calls `pxr` directly.
- **Tools**: Isaac tools are methods on `IsaacTools` mixins carrying a
  `@tool_meta` decorator (name, description, hints); `_reg_isaac.py` builds
  the MCP wrapper from each method's signature and docstring. The USD,
  Blender and Unreal modules register their wrappers by hand over adapter
  sessions. Every call goes through a shared envelope that applies rate
  limiting, the file sandbox, usage recording and result-size budgets.
- **Results**: tools return JSON-serializable dicts; errors are
  `ErrorResponse(...).dict()` with `success: false` and an error type.

### Adding a backend

1. Implement `BackendAdapter` in `src/simul/adapters/`.
2. Write its registration module in `src/simul/mcp/registration/`.
3. Add one `BackendSpec` to `src/simul/mcp/backends.py`.

The server, `simul info`, `get_capabilities` and the routing
instructions pick it up from there.

### Python API

The headless USD adapter can be used directly:

```python
from simul.adapters import HeadlessUSDAdapter

adapter = HeadlessUSDAdapter()
with adapter.create_session() as session:
    stage_id = session.load_stage("tests/data/simple_scene.usda")
    info = session.get_stage_info(stage_id)
    meshes = session.find_prims_by_type(stage_id, "Mesh")
    summary = session.summarize_stage(stage_id)
    print(info.prim_count, len(meshes), summary.total_prims)
```

`examples/isaac/sample_usd_reader.py` is a runnable version of this
(`python examples/isaac/sample_usd_reader.py scene.usd --verbose`), and
`examples/isaac/http_client_mcp.py` calls tools on a server started with
`--transport http` (pass its URL with `--server`).

The server can be embedded with
`SimulMCPServer(settings, backends={"usd"})` from `simul.mcp.server` and
started with `await server.run("stdio")`.

## Tests

Run from the repository root; `tests/conftest.py` puts `src/` first on
`sys.path`, so the checkout's code is tested whatever is installed.

```bash
pytest tests/ --no-cov -q            # fast loop: unit tier, no coverage
pytest tests/                        # with coverage (as configured in pyproject)
pytest tests/cli/test_usd_cli.py     # one file
```

`pyproject.toml` bakes `-m "not packaging"` into `addopts`, so the default run
skips wheel builds. The last `-m` on the command line wins.

| Tier | Command | Needs |
|---|---|---|
| Unit | `pytest tests/ --no-cov` | Nothing; must be 100% green on `main` |
| Packaging | `pytest tests/packaging -m packaging --no-cov` | `uv` on `PATH`; builds and inspects the wheel |
| Isaac Sim live | `pytest tests/isaac/live -m isaac` | A running Isaac Sim (`simul isaac launch`); skips when the socket does not answer |
| Unreal live | `pytest tests/ -m unreal_live` | A running editor set up with `simul unreal setup`; see [unreal-e2e-checklist.md](unreal-e2e-checklist.md) |
| Blender live | `SIMUL_BLENDER_LIVE=1 pytest tests/blender/test_live_attach.py -m blender_live` | Launches and closes a disposable Blender GUI |

When your change touches an adapter or its tools and the runtime is available
on your machine, run the live tier too. Transport changes to Isaac Sim should
be checked on both a 5.1 and a 6.0 install. Copy any project a test mutates
(for example a `.uproject`) to a scratch directory first.

The FastMCP test double lives in `tests/fakes.py` and is installed by the
`fake_fastmcp` fixture. `tests/mcp/test_tool_meta_drift.py` fails when an
Isaac tool method lacks `@tool_meta` or its wrapper drifts from the
implementation.

Isaac Sim's own interpreter is used only for one smoke test:

```bash
$ISAAC_SIM_PATH/python.sh scripts/isaac/smoke_simulationapp.py
```

## Formatting and linting

```bash
black src/ tests/ examples/ && isort src/ tests/ examples/
flake8 src/ tests/          # max line length 119, see .flake8
mypy src/
```

CI runs these as informational checks; the unit and packaging jobs are the
gates. `scripts/isaac/dev_isort_black.sh` runs the formatters in one go.

## Live verification of the MCP server

An MCP server already running inside your agent session does not reload
edited source. To check a change end to end, run the editable-installed CLI
as a fresh process, for example `simul --json info` or
`simul isaac scene`.

## Versioning

The version lives in four places that must change together:
`pyproject.toml`, `.claude-plugin/plugin.json`,
`src/simul/__init__.py` and
`src/simul/bridge_ext/khemoo.simul/config/extension.toml`.
`tests/test_version_lockstep.py` enforces it. Releases are cut by
maintainers.
