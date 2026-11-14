"""``python -m simul`` runs the ``simul`` command.

Before, the package had no ``__main__``, so ``python -m simul`` failed with
"No module named simul.__main__" and only the ``simul`` script (or
``python -m simul.cli.main``) reached the CLI.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import simul

SRC = str(Path(simul.__file__).resolve().parents[1])


def _run(*args: str) -> "subprocess.CompletedProcess[str]":
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [SRC, os.environ.get("PYTHONPATH")]))}
    return subprocess.run(
        [sys.executable, "-m", "simul", *args], capture_output=True, text=True, env=env, timeout=120
    )


def test_python_m_simul_prints_the_version() -> None:
    result = _run("--json", "version")

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["version"] == simul.__version__


def test_python_m_simul_names_itself_simul_in_its_help() -> None:
    result = _run("--help")

    assert result.returncode == 0, result.stderr
    assert "Usage: simul " in result.stdout
    assert "__main__" not in result.stdout
