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
``packaging`` so the standard
``pytest -m "not isaac and not unreal_live"`` invocation skips it
(building a wheel takes ~10s, too slow for the fast unit loop).
Release CI should run ``pytest -m packaging`` before publishing.
"""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

# Anchor at the repo root so the test runs regardless of cwd.
_REPO = Path(__file__).resolve().parents[2]


def _have_uv_or_build() -> tuple[str, list[str]] | None:
    """Return (label, build-cmd-prefix) for whichever wheel builder
    is available; ``None`` if neither is installed.

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


pytestmark = pytest.mark.packaging


def test_wheel_ships_bundled_bridge_ext(tmp_path: Path) -> None:
    """Build a wheel and assert the bundled bridge ext is present.

    Locks the iter14 contract: ``simul_mcp/bridge_ext/khemoo.simul.mcp/``
    must contain at minimum the extension manifest (config/extension.toml)
    and the entry-point Python module (khemoo/simul/mcp/extension.py).
    Failure means a future setuptools or pyproject change broke the
    package-data glob and no pip user can run ``install-bridge``.
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
    assert proc.returncode == 0, (
        f"{label} build failed (exit {proc.returncode}). "
        f"stderr:\n{proc.stderr[-2000:]}\nstdout:\n{proc.stdout[-2000:]}"
    )

    wheels = sorted(out_dir.glob("simul_mcp-*-py3-none-any.whl"))
    assert len(wheels) == 1, f"Expected exactly one wheel, got {wheels}"
    wheel = wheels[0]

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

    # Force-create a pycache file in the source tree so the exclude
    # actually has something to drop. Without this, the test passes
    # vacuously when the dev tree is already clean.
    pycache = (
        _REPO / "src" / "simul_mcp" / "bridge_ext"
        / "khemoo.simul.mcp" / "khemoo" / "simul" / "mcp" / "__pycache__"
    )
    pycache.mkdir(exist_ok=True)
    sentinel = pycache / "regression_sentinel.cpython-311.pyc"
    sentinel.write_bytes(b"\x00\x00\x00\x00")
    try:
        proc = subprocess.run(
            cmd, cwd=_REPO, capture_output=True, text=True, timeout=120
        )
        assert proc.returncode == 0, proc.stderr[-2000:]

        wheels = sorted(out_dir.glob("simul_mcp-*-py3-none-any.whl"))
        assert len(wheels) == 1, f"Expected exactly one wheel, got {wheels}"

        with zipfile.ZipFile(wheels[0]) as zf:
            names = zf.namelist()

        leaked = [n for n in names if n.endswith(".pyc") or "__pycache__" in n]
        assert not leaked, (
            f"Bytecode leaked into wheel — exclude-package-data is "
            f"not firing. Leaked entries: {leaked}"
        )
    finally:
        sentinel.unlink(missing_ok=True)
        # Don't remove the dir — other test runs may share it.


def test_wheel_extension_toml_version_matches_package(tmp_path: Path) -> None:
    """The bundled extension.toml's version must equal simul_mcp.__version__.

    The 4-file lockstep at source level is enforced by
    ``test_version_lockstep.py``. This test extends that guarantee
    into the *built artifact*: the bundled ``extension.toml`` shipped
    in the wheel must show the same version the wheel filename
    advertises. A drift here would mean ``install-bridge`` from a
    pip-installed copy publishes an extension whose Kit ID does not
    match the parent package — which is exactly the iter11 drift
    failure mode the lockstep was created to prevent.
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
    assert proc.returncode == 0, proc.stderr[-2000:]

    wheels = sorted(out_dir.glob("simul_mcp-*-py3-none-any.whl"))
    assert len(wheels) == 1, f"Expected exactly one wheel, got {wheels}"
    wheel = wheels[0]

    # Wheel filename: simul_mcp-X.Y.Z-py3-none-any.whl → "X.Y.Z"
    wheel_version = wheel.name.split("-")[1]

    member = "simul_mcp/bridge_ext/khemoo.simul.mcp/config/extension.toml"
    with zipfile.ZipFile(wheel) as zf:
        toml_text = zf.read(member).decode("utf-8")

    import re
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
