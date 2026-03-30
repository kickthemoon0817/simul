"""Smoke tests for the Isaac Sim extension package layout."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def test_extension_package_imports_without_omni(monkeypatch) -> None:
    """The extension package should import from the repo roots outside Isaac Sim."""
    repo_root = Path(__file__).resolve().parents[2]
    extension_root = repo_root / "exts" / "khemoo.simul.mcp"
    source_root = repo_root / "src"

    monkeypatch.syspath_prepend(str(source_root))
    monkeypatch.syspath_prepend(str(extension_root))

    for module_name in (
        "khemoo.simul.mcp.extension",
        "khemoo.simul.mcp.lifecycle",
        "khemoo.simul.mcp.ui_builder",
    ):
        sys.modules.pop(module_name, None)

    module = importlib.import_module("khemoo.simul.mcp.extension")

    assert module.IsaacMCPServerExtension.__name__ == "IsaacMCPServerExtension"
