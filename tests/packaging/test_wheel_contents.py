"""Wheel-build regression test — locks the iter14 bundling invariant.

iter14 moved the Isaac Sim Kit bridge extension into the Python
wheel so pip-installed users can run ``simul-mcp isaac install-bridge``
without a repo checkout. The whole feature relies on a single
setuptools mechanism — a ``[tool.setuptools.package-data]`` glob
on a dotted-name subdirectory:

    [tool.setuptools.package-data]
    "simul_mcp.bridge_ext" = ["khemoo.simul.mcp/**/*"]

Setuptools' handling of dotted-name dirs in package-data globs has
had regressions in the past (for example, setuptools #3341, fixed
in 62.3 — the lower-bound this project pins). A future setuptools
release that quietly changes this behavior would silently produce
wheels with an empty ``bridge_ext/khemoo.simul.mcp/`` directory,
and ``install-bridge`` would fail at runtime for every pip user
without any signal at build time.

This test catches that class of regression by actually building a
wheel and inspecting its zipfile contents. It is marked
``packaging`` so the default ``pytest`` invocation (which has
``-m "not packaging"`` baked into ``addopts``) skips it — building
a wheel takes ~1 s on uv but several seconds on ``python -m build``,
too slow for the fast unit loop. Release CI should run
``pytest -m packaging`` before publishing.
"""

from __future__ import annotations

import functools
import json
import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

# Anchor at the repo root so the test runs regardless of cwd.
_REPO = Path(__file__).resolve().parents[2]


@functools.lru_cache(maxsize=None)
def _have_uv_or_build() -> tuple[str, list[str]] | None:
    """Return (label, build-cmd-prefix) for whichever wheel builder
    is available; ``None`` if neither is installed.

    Cached at module level so the three tests don't each spawn a
    fresh ``python -c "import build"`` probe. This also pins the
    answer for the whole pytest session — if the environment changes
    mid-run the same builder is used by every test, eliminating a
    cross-test inconsistency footgun.

    uv is preferred because it does not need pip in the calling venv
    (this project's dev venv is uv-managed and ships without pip),
    but falls back to ``python -m build`` for environments that have
    that installed instead.
    """
    if shutil.which("uv"):
        return "uv", ["uv", "build", "--wheel"]
    try:
        subprocess.run(
            ["python", "-c", "import build"], check=True, capture_output=True
        )
        return "build", ["python", "-m", "build", "--wheel"]
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _build_error_msg(label: str, proc: subprocess.CompletedProcess) -> str:
    """Consistent build-failure message — both stderr and stdout
    truncated identically. Some build tools route the actual failure
    detail to stdout, so showing only stderr would hide it.
    """
    return (
        f"{label} build failed (exit {proc.returncode}). "
        f"stderr:\n{proc.stderr[-2000:]}\n"
        f"stdout:\n{proc.stdout[-2000:]}"
    )


def _run_build(tmp_path: Path) -> Path:
    """Build a wheel into ``tmp_path/dist`` and return the wheel path.

    Returns the single produced wheel path. Skips the test if no
    builder is available; raises an ``AssertionError`` if the build
    fails (the message includes the truncated subprocess output).
    """
    builder = _have_uv_or_build()
    if builder is None:
        pytest.skip("Neither `uv` nor `python -m build` is available.")
    label, cmd = builder

    out_dir = tmp_path / "dist"
    out_dir.mkdir()
    if label == "uv":
        cmd = cmd + ["--out-dir", str(out_dir)]
    else:
        cmd = cmd + ["--outdir", str(out_dir)]

    proc = subprocess.run(
        cmd, cwd=_REPO, capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, _build_error_msg(label, proc)

    wheels = sorted(out_dir.glob("simul_mcp-*-py3-none-any.whl"))
    assert len(wheels) == 1, f"Expected exactly one wheel, got {wheels}"
    return wheels[0]


pytestmark = pytest.mark.packaging


def test_wheel_ships_bundled_bridge_ext(tmp_path: Path) -> None:
    """Build a wheel and assert the bundled bridge ext is present.

    Locks the iter14 contract: ``simul_mcp/bridge_ext/khemoo.simul.mcp/``
    must contain at minimum the extension manifest (config/extension.toml)
    and the entry-point Python module (khemoo/simul/mcp/extension.py).
    Failure means a future setuptools or pyproject change broke the
    package-data glob and no pip user can run ``install-bridge``.
    """
    wheel = _run_build(tmp_path)

    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()

    required = [
        "simul_mcp/bridge_ext/__init__.py",
        "simul_mcp/bridge_ext/khemoo.simul.mcp/config/extension.toml",
        "simul_mcp/bridge_ext/khemoo.simul.mcp/khemoo/simul/mcp/extension.py",
        "simul_mcp/bridge_ext/khemoo.simul.mcp/khemoo/simul/mcp/lifecycle.py",
        "simul_mcp/bridge_ext/khemoo.simul.mcp/khemoo/simul/mcp/protocol.py",
        "simul_mcp/bridge_ext/khemoo.simul.mcp/khemoo/simul/mcp/service.py",
        "simul_mcp/bridge_ext/khemoo.simul.mcp/khemoo/simul/mcp/executor.py",
        "simul_mcp/bridge_ext/khemoo.simul.mcp/khemoo/simul/mcp/ui_builder.py",
    ]
    missing = [r for r in required if r not in names]
    assert not missing, (
        f"Bundled bridge ext is incomplete in wheel {wheel.name}. "
        f"Missing entries: {missing}\n"
        f"Got bridge_ext entries: "
        f"{sorted(n for n in names if 'bridge_ext' in n)}"
    )


def test_wheel_excludes_pyc_bytecode(tmp_path: Path) -> None:
    """Pyc files must NOT ship in the wheel.

    The pyproject ``[tool.setuptools.exclude-package-data]`` rule
    drops ``khemoo.simul.mcp/**/__pycache__/*`` so a developer
    building from a dirty tree does not accidentally publish stale
    bytecode. This test runs after a build and asserts the exclude
    actually fired — silent inclusion would mean the rule was either
    unrecognized by the active setuptools or the glob path is wrong.

    Plants a sentinel ``.pyc`` in the source tree so the assertion
    isn't vacuous on a clean dev machine. Cleans up the sentinel
    AND the ``__pycache__`` dir if this test was the one that
    created it (otherwise leaves it alone — could be a real
    bytecode dir from a previous interpreter run that another
    process owns).
    """
    pycache = (
        _REPO / "src" / "simul_mcp" / "bridge_ext"
        / "khemoo.simul.mcp" / "khemoo" / "simul" / "mcp" / "__pycache__"
    )
    pycache_pre_existed = pycache.exists()
    pycache.mkdir(exist_ok=True)
    sentinel = pycache / "iter15_regression_sentinel.cpython-311.pyc"
    sentinel.write_bytes(b"\x00\x00\x00\x00")
    try:
        wheel = _run_build(tmp_path)

        with zipfile.ZipFile(wheel) as zf:
            names = zf.namelist()

        leaked = [n for n in names if n.endswith(".pyc") or "__pycache__" in n]
        assert not leaked, (
            f"Bytecode leaked into wheel — exclude-package-data is "
            f"not firing. Leaked entries: {leaked}"
        )
    finally:
        sentinel.unlink(missing_ok=True)
        # Only remove the dir if this test created it, AND only when
        # empty — a parallel test or import that wrote real bytecode
        # mid-flight should not be wiped.
        if not pycache_pre_existed and pycache.exists():
            try:
                pycache.rmdir()
            except OSError:
                pass  # Not empty — leave it.


def test_wheel_extension_toml_version_matches_package(tmp_path: Path) -> None:
    """The bundled extension.toml's version must equal the wheel version.

    The 4-file lockstep at source level is enforced by
    ``test_version_lockstep.py``. This test extends that guarantee
    into the *built artifact*: the bundled ``extension.toml`` shipped
    in the wheel must show the same version the wheel filename
    advertises. A drift here would mean ``install-bridge`` from a
    pip-installed copy publishes an extension whose Kit ID does not
    match the parent package — exactly the iter11 drift failure
    mode the lockstep was created to prevent.
    """
    wheel = _run_build(tmp_path)
    # Wheel filename: simul_mcp-X.Y.Z-py3-none-any.whl → "X.Y.Z"
    wheel_version = wheel.name.split("-")[1]

    member = "simul_mcp/bridge_ext/khemoo.simul.mcp/config/extension.toml"
    with zipfile.ZipFile(wheel) as zf:
        toml_text = zf.read(member).decode("utf-8")

    m = re.search(
        r'^\[package\]\s*\nversion\s*=\s*"([0-9]+\.[0-9]+\.[0-9]+)"',
        toml_text,
        flags=re.MULTILINE,
    )
    assert m, (
        f"Could not parse [package] version from bundled extension.toml. "
        f"Content:\n{toml_text}"
    )
    bundled_version = m.group(1)
    assert bundled_version == wheel_version, (
        f"Bundled bridge ext version drifted from wheel version. "
        f"wheel={wheel_version}, bundled extension.toml={bundled_version}. "
        f"This is exactly the iter11 lockstep failure mode the bundling "
        f"was supposed to make impossible."
    )


_API_DOCS = ("core", "sensors", "physics", "replicator", "robots", "rendering", "assets")


def test_wheel_ships_packaged_resources(tmp_path: Path) -> None:
    """The MCP resources and default config must travel inside the wheel.

    ``skills.md``, the ``docs/api`` references and the two YAML files are
    read through ``importlib.resources`` at runtime; a wheel missing any of
    them serves "not found" resources and default-only settings to every
    pip user without a build-time signal.
    """
    wheel = _run_build(tmp_path)

    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()

    required = [
        "simul_mcp/resources/__init__.py",
        "simul_mcp/resources/skills.md",
        "simul_mcp/resources/config/default.yaml",
        "simul_mcp/resources/config/logging.yaml",
    ] + [f"simul_mcp/resources/docs/api/{doc}.md" for doc in _API_DOCS]
    missing = [r for r in required if r not in names]
    assert not missing, (
        f"Packaged resources are incomplete in wheel {wheel.name}. "
        f"Missing entries: {missing}\n"
        f"Got resources entries: {sorted(n for n in names if '/resources/' in n)}"
    )


def _install_wheel_into_fresh_venv(tmp_path: Path, wheel: Path) -> Path:
    """Create a venv holding only the wheel and the imports ``simul_mcp`` needs.

    Returns the venv's python executable. Uses uv when present (no network
    once the cache is warm), otherwise stdlib venv plus pip.
    """
    venv_dir = tmp_path / "venv"
    python = venv_dir / "bin" / "python"
    runtime_deps = ["pydantic", "pydantic-settings", "pyyaml", "numpy"]
    if shutil.which("uv"):
        subprocess.run(["uv", "venv", str(venv_dir)], check=True, capture_output=True, text=True, timeout=120)
        pip_install = ["uv", "pip", "install", "--python", str(python)]
    else:
        subprocess.run(
            ["python", "-m", "venv", str(venv_dir)], check=True, capture_output=True, text=True, timeout=120
        )
        pip_install = [str(python), "-m", "pip", "install"]
    # The wheel goes in without its dependency closure (usd-core, fastmcp, ...)
    # so the venv stays small; only what ``import simul_mcp`` touches follows.
    for install_cmd in (pip_install + ["--no-deps", str(wheel)], pip_install + runtime_deps):
        proc = subprocess.run(install_cmd, capture_output=True, text=True, timeout=600)
        assert proc.returncode == 0, _build_error_msg("wheel install", proc)
    return python


_WHEEL_SMOKE_SCRIPT = """
import json
from simul_mcp.config import Settings, get_settings
from simul_mcp.resources import find_checkout_root, resource
from simul_mcp.utils.paths import PathPolicy

settings = get_settings()
checkout_root = find_checkout_root()
print(json.dumps({
    "checkout_root": None if checkout_root is None else str(checkout_root),
    "skills_head": resource("skills.md").read_text(encoding="utf-8")[:400],
    "api_core_is_file": resource("docs", "api", "core.md").is_file(),
    "socket_protocol": settings.isaac_sim.socket_protocol,
    "cors_origins": settings.server.cors_origins,
    "allowed_roots": [str(root) for root in PathPolicy.from_settings(settings).allowed_roots],
    "bare_settings_ok": isinstance(Settings(), Settings),
}))
"""


def test_wheel_install_serves_settings_and_resources(tmp_path: Path) -> None:
    """Install the wheel into an empty venv and use it away from the checkout.

    Runs from ``tmp_path`` so no repo file is reachable by accident: the
    packaged default YAML must load, the environment must still override
    it, ``skills.md`` must be readable through ``importlib.resources`` and
    the sandbox allowlist must not point at ``lib/python3.x``.
    """
    wheel = _run_build(tmp_path)
    python = _install_wheel_into_fresh_venv(tmp_path, wheel)

    env = {k: v for k, v in os.environ.items() if not k.startswith(("ISAAC_SIM", "CONFIG_FILE", "SERVER__"))}
    env["ISAAC_SIM__SOCKET_PROTOCOL"] = "vscode"
    proc = subprocess.run(
        [str(python), "-c", _WHEEL_SMOKE_SCRIPT],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"wheel smoke script failed:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    payload = json.loads(proc.stdout.strip().splitlines()[-1])

    assert payload["checkout_root"] is None
    assert "execute_isaac_script" in payload["skills_head"]
    assert payload["api_core_is_file"] is True
    assert payload["socket_protocol"] == "vscode"
    assert payload["cors_origins"][-1] == "http://localhost:8229", "packaged default.yaml was not loaded"
    assert payload["allowed_roots"] == ["/tmp/simul_mcp"]
    assert payload["bare_settings_ok"] is True
