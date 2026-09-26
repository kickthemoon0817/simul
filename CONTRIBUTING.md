# Contributing to Simul

Thanks for helping. This page covers the workflow; the code layout,
architecture and test tiers are in [docs/development.md](docs/development.md).

## Reporting issues

File bugs and feature requests at
<https://github.com/kickthemoon0817/simul/issues>. A useful bug report has:

- **What** happened, in one sentence
- **Reproduction**: the exact command or MCP tool call with its inputs
- **Expected** and **observed** behaviour, including the JSON error payload
- **Environment**: `simul version`, the backend and its version (Isaac Sim,
  Unreal Engine, Blender), and your OS

For a security problem, open an issue asking for a private contact and leave
the details out of the public thread.

## Development setup

Python 3.11, 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/kickthemoon0817/simul.git
cd simul
uv venv .venv && uv pip install -e ".[dev]"
source .venv/bin/activate
```

`uv sync --extra dev` installs the exact versions from `uv.lock`, as CI does.

## Before you open a pull request

```bash
pytest tests/ --no-cov -q                        # must be green
pytest tests/packaging -m packaging --no-cov     # if you touched packaging or pyproject.toml
black src/ tests/ examples/ && isort src/ tests/ examples/
```

- Add or update tests for behaviour you change. Unit tests must not need a
  running engine; use the fakes in `tests/fakes.py`.
- If you changed an adapter or its tools and have the runtime installed, run
  the matching live tier (see
  [docs/development.md](docs/development.md#tests)) and say so in the PR.
- Update the docs that describe the behaviour: `README.md`, `docs/`, and the
  tool's description or docstring.
- Do not bump version numbers; maintainers do that when releasing.

## Pull request flow

1. Branch from `main` with a descriptive name, for example
   `fix/isaac-capture-timeout` or `docs/cli-reference`.
2. Write commits with [Conventional Commits](https://www.conventionalcommits.org/)
   subjects: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `perf:`,
   `chore:`. Keep the subject under about 72 characters and explain the why in
   the body.
3. Push and open a pull request against `main`, filling in the template
   (summary, changes, testing). `main` is protected, so every change goes
   through a PR.
4. CI runs the unit tests on Python 3.11 to 3.13 and the packaging gate;
   both must pass. Lint and type checks are informational.
5. PRs are merged with a merge commit (not squash or rebase), so keep your
   branch's commits meaningful.

## License

By contributing you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE).
