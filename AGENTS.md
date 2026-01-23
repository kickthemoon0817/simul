# AGENTS.md — Isaac Sim MCP Server

This file guides agentic coding assistants working in this repository.

## Repository Scope
- Root: `/simul-mcp`
- Primary package: `src/simul_mcp`
- CLI: `src/simul_mcp/cli/main.py`
- Extension: `exts/khemoo.simul.mcp`

## Cursor/Copilot Rules
- No Cursor rules found (`.cursor/rules/` or `.cursorrules`).
- No Copilot rules found (`.github/copilot-instructions.md`).

## General Workflow
- Read relevant repo docs before coding; do not start implementation immediately.
- Prefer small, reviewable changes; explain rationale when behavior changes.
- Follow existing patterns in `src/`, `tests/`, `docs/`, and simulator-specific folders.
- Keep ASCII-only edits unless the file already uses Unicode.
- Avoid unrelated refactors while fixing a bug.

## Build / Lint / Test Commands (Makefile)
Run from `/simul-mcp`.
- `make help`             # List targets
- `make install`          # Install package
- `make install-dev`      # Install dev deps + pre-commit
- `make setup-isaac`      # Validate ISAAC_SIM_PATH

## Formatting
- `make format`            # black + isort
- `./scripts/isaac/dev_isort_black.sh`
- `./scripts/isaac/dev_isort_black.sh --check`
- `./scripts/isaac/dev_isort_black.sh --diff`
- `./scripts/isaac/dev_isort_black.sh --isort-only`
- `./scripts/isaac/dev_isort_black.sh --black-only`

## Lint / Type Check
- `make lint`              # flake8 + mypy
- `flake8 src/ tests/`
- `mypy src/`

## Tests
- `make test`              # pytest tests/ -v
- `make test-cov`          # pytest with coverage
- `make test-isaac`        # pytest -m isaac
- `make isaac-test`        # Isaac Sim python.sh + pytest -m isaac

## Single Test (Pytest)
- `pytest tests/isaac/test_reader.py`
- `pytest tests/isaac/test_reader.py::TestUSDReader::test_open_stage_success -v`
- `pytest tests/ -k "mesh" -v`

## Run Server (Dev)
- `make run-server`
- `make run-headless`
- `make run-isaac`          # uses ISAAC_SIM_PATH/python.sh
- `python -m simul_mcp.cli.main server --host localhost --port 8765`

## Debugging
- `make debug-server`
- `make debug-isaac`

## Extension Install/Uninstall
- `make ext-install`
- `make ext-uninstall`

## Isaac Sim Root Test Runner (Not MCP-specific)
Run from `/isaac-sim`.
- `python run_tests.py --suite alltests --bucket all`
- `./tests/tests-isaacsim.core.api.sh`
- `./tests/tests-nativepython-testing-isaacsim.core.api.hello_world.sh`

---

## Python Conventions
- Use type hints for public APIs.
- Add docstrings for public classes/functions.
- Keep modules small and focused; avoid circular imports.
- Use logging instead of print for runtime signals.
- Favor dataclasses when they improve clarity of data containers.
- Prefer explicit errors with clear messages over silent failures.

## Class Structure and Method Order
Use this exact order within classes:
1) `__init__`
2) Properties (`@property`, setters)
3) Core public methods
4) External-facing helpers
5) Internal helpers (prefixed `_`)

## Internal Helpers and Static Methods
- Do not define module-level helper functions for class internals.
- Internal helpers should be class methods, implemented as `@staticmethod` unless they require instance state.
- Do not call internal helpers from module-level code in the same file; only invoke them from class methods.

## Formatting and Imports
- Black line length: 88 (see `pyproject.toml`).
- isort profile: `black`, `known_first_party = ["simul_mcp"]`.
- Import order: standard library → third-party → local.
- Avoid unused imports; keep modules lean.

## Typing
- Mypy is strict (see `pyproject.toml`).
- Use `Optional[...]` explicitly when None is allowed.
- Prefer structured types or Pydantic models over `Dict[str, Any]`.

## Error Handling
- MCP tools should return JSON-serializable dicts.
- On error, return `ErrorResponse(...).dict()`.
- Log errors with context and helpful messages.

## Isaac Sim / Isaac Lab Workflow
- Before coding Isaac Sim/Lab features, review local docs:
  - `costnav_isaacsim/README.md`
  - `costnav_isaacsim/isaac_sim_teleop_ros2/README.md`
  - `costnav_isaaclab/README.md`
  - Relevant `docs/` references when needed
- Identify relevant extensions or features and cite them in the plan.
- Validate compatibility with simulator version and dependency setup.
- Ask clarifying questions if the request could conflict with simulator constraints.
- For a local sanity check, you can run Isaac Sim using:
```
/bin/bash -lc 'LD_LIBRARY_PATH=extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311/bin:$LD_LIBRARY_PATH ./python.sh - <<'"'"'PY'"'"'
import sys, os
sys.path.append('extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311')
from pxr import Usd
print('pxr import ok')
stage = Usd.Stage.Open('extsUser/khemoo.system.power/sources/openbot.usd')
print(stage)
for prim in stage.GetPseudoRoot().GetChildren():
    print('root child', prim.GetPath())
print('Prim count limit:')
for i, prim in enumerate(stage.Traverse()):
    print(prim.GetPath())
    if i >= 60:
        break
PY'
```

## Testing Expectations
- Add or update tests for new behavior when feasible.
- Keep tests deterministic; avoid simulator dependencies in unit tests unless requested.

## Documentation
- Update `README.md` or `docs/` when behavior or usage changes.
- Keep examples minimal, runnable, and aligned with actual APIs.

---

## Quick Reference (pyproject.toml)
- Python: >= 3.8
- Black: line length 88
- pytest markers: slow, integration, isaac
- Coverage: `--cov=simul_mcp`

## Notes for Agents
- Some tests require Isaac Sim runtime.
- Headless tools must work without Omniverse.
- Avoid changing unrelated caches or Isaac root files.
