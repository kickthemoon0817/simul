"""Regression: the four documented version constants must always agree.

CLAUDE.md's Versioning section pins this — any future bump that touches
fewer than the four files is a hygiene defect that will silently leave
either the Python package, the Claude Code plugin manifest, the Isaac
Sim bridge extension, or pyproject metadata behind. iter11 was created
specifically because the bridge ext drifted from 0.0.19 → 0.0.32 across
13 patch tags without anyone noticing; this test prevents a recurrence.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Anchor at the repo root so the test runs regardless of cwd.
_REPO = Path(__file__).resolve().parents[1]


def _read_pyproject_version() -> str:
    text = (_REPO / "pyproject.toml").read_text(encoding="utf-8")
    # Match the [project] section's version, not e.g. requires-python or
    # any other version-shaped string. Anchored to the second line of
    # the [project] block per the canonical layout.
    m = re.search(
        r'^\[project\][^\[]*?\nversion\s*=\s*"([0-9]+\.[0-9]+\.[0-9]+)"',
        text,
        flags=re.MULTILINE,
    )
    assert m, "pyproject.toml [project] version not found"
    return m.group(1)


def _read_plugin_json_version() -> str:
    return json.loads((_REPO / ".claude-plugin/plugin.json").read_text())["version"]


def _read_init_version() -> str:
    sys.path.insert(0, str(_REPO / "src"))
    try:
        import simul_mcp
        # Reload-safe — just read the module attribute.
        return simul_mcp.__version__
    finally:
        sys.path.pop(0)


def _read_bridge_ext_version() -> str:
    text = (_REPO / "exts/khemoo.simul.mcp/config/extension.toml").read_text(
        encoding="utf-8"
    )
    m = re.search(
        r'^\[package\]\s*\nversion\s*=\s*"([0-9]+\.[0-9]+\.[0-9]+)"',
        text,
        flags=re.MULTILINE,
    )
    assert m, "exts/khemoo.simul.mcp/config/extension.toml [package] version not found"
    return m.group(1)


def test_all_four_version_constants_agree() -> None:
    versions = {
        "pyproject.toml": _read_pyproject_version(),
        ".claude-plugin/plugin.json": _read_plugin_json_version(),
        "src/simul_mcp/__init__.py": _read_init_version(),
        "exts/khemoo.simul.mcp/config/extension.toml": _read_bridge_ext_version(),
    }
    distinct = set(versions.values())
    assert len(distinct) == 1, (
        "Version drift detected — bump skipped at least one file. "
        f"Per-file versions: {versions}"
    )
