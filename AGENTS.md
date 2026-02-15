# AGENTS.md — Isaac Sim MCP Server

This file guides agentic coding assistants working in this repository.

## Overall Coding Standards

1. Core Philosophy: "Performant & Explicit"
   - Hardware-Friendly Code: Python is interpreted, but our logic should be hardware-aware. Prefer vectorized operations (NumPy, PyTorch) or compiled extensions over pure Python loops for heavy computations. As a rule of thumb, if you are iterating over more than 10,000 elements, you should be reaching for a vectorized or compiled alternative. For smaller loops where clarity is obvious, plain Python is fine.
   - Explicit is Better than Implicit: Write code that clearly states its intent. Verbose variable names are encouraged if they add clarity (velocity_meters_per_second > v). If you find yourself writing a comment to explain what a variable holds, the variable is misnamed.
   - Type Safety (The Rust Influence): Treat Python as a strictly typed language. Use Type Hints everywhere — function signatures, return types, class attributes, and local variables where the type is not immediately obvious from assignment. Use full generics (dict[str, list[int]]), TypeAlias, Protocol, and TypeVar where appropriate. If the IDE cannot infer what an object is, the code is considered incomplete.
   - Prefer Immutability: Favor immutable data patterns where practical. Default to tuple over list, frozenset over set, and @dataclass(frozen=True) over mutable dataclasses when the data does not need to change after creation. If a function must mutate state, make that explicit in the function name (e.g., update_position_in_place(...)) rather than hiding side effects.

2. General Architecture & Design
   - Cohesion Over Fragmentation (Avoid "Micro-Methods"):
      - Rule: Do not break code into tiny 1-2 line helper methods unless they are reused in multiple places or encapsulate genuinely complex logic.
      - Why: Excessive method calls add stack overhead (performance) and force the developer to jump around the file to understand simple logic (readability).
      - Preference: It is better to have a slightly longer, linear function where the logic is visible in one place than ten tiny functions that obscure the flow. Aim for functions that read top-to-bottom like a narrative. If a function exceeds roughly 60-80 lines, consider extracting logical phases (setup, processing, teardown) — not micro-helpers.

   - Fail Fast & Fail Loudly:
     - Do not suppress errors with bare try: ... except: pass. Handle specific exceptions explicitly.
     - Catch errors at the boundary — I/O operations, API calls, external system interactions. Let pure internal logic propagate exceptions naturally; do not defensively wrap every function body.
     - If a function receives invalid input, raise immediately. Do not return None or a sentinel value to signal failure unless the function's contract explicitly defines that behavior (and documents it in the type signature, e.g., -> Result | None).

   - Dependencies:

     - Rule: Every third-party dependency is a liability — it is code you do not control, cannot always audit, and must maintain compatibility with.
     - Standard: Before adding a dependency, ask: "Is what I need from this library more than 50 lines of code to write myself?" If yes, use the library. If no, write it inline and own it.
     - Exceptions: Battle-tested, well-maintained ecosystem packages (NumPy, PyTorch, Pydantic, etc.) are always acceptable. The bar is for niche, single-purpose packages.

## Project Version Control

This repository uses Git for version control. Please follow these guidelines when contributing:

1. **Branching Strategy**: Use feature branches for new features and bug fixes. Name branches descriptively (e.g., `feature/add-user-authentication`, `bugfix/fix-login-error`). This is important to avoid conflicts between branches.
2. **Commit Messages**: Write clear and concise commit messages, e.g. "feat: add user authentication" or "docker: update Dockerfile for production". or so on. Never use parentheses in commit messages.
3. **Pull Requests**: Submit pull requests for code reviews before merging changes into the main branch. Ensure that your code passes all tests and adheres to coding standards.
4. **Code Reviews**: Participate in code reviews to maintain code quality and share knowledge among team members.

## Python Coding Conventions

All Python scripts in this repository must adhere to the following standards:

1.  **PEP 8 Compliance**: Follow standard PEP 8 style guidelines.
2.  **Object-Oriented Design**:
    -   Prefer Classes over standalone functions (except for simple utilities).
    -   **One Main Class**: Each script should contain only **one** main class. Additional classes are permitted only if they are for configuration or data modelling related to the main class.
3.  **Global Scope Hygiene**:
    -   Avoid exposing variables or internal functions in the global scope. Everything should be contained within the class namespace.
4.  **Static Methods**:
    -   If a function belongs conceptually to a class but does not require access to the instance (self) or class (cls), define it using the `@staticmethod` decorator inside the class rather than moving it outside.
5.  **Avoid Registry Patterns (DX First)**:
    -   Avoid dynamic registries or string-based dispatching (e.g., `func_map['name']()`).
    -   Use explicit imports and direct class/method references to ensure IDE navigation (F12 / Ctrl+Click) works for all code paths.
6.  **Order of Methods**:
    methods should be organized in the following order:
    1.  `__init__` and dunder methods
    2.  Properties (`@property`)
    3.  Main methods (explaining the class's role)
    4.  Public methods (Exposed methods, API)
    5.  Internal methods (helpers, prefixed with `_` if needed)
    6.  Static methods (`@staticmethod`)
7.  **Type Hints**: Use type hints for all function signatures to improve code clarity and facilitate static analysis. The type hints should provide the entry point, so developers can go to the definition of types using ctrl+click / F12 in IDEs.
8. Documentation Standards
    - Google Style Docstrings: Follow the Google Python Style Guide for docstrings.
    - Line Breaking: For multi-line docstrings, the summary text must start on the line after the opening triple quotes.
    - Required Sections: You must include Args: and Returns: sections for any method that takes parameters or returns a value.


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
