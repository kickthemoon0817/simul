"""Data files shipped inside the ``simul_mcp`` wheel.

The default settings YAML, the logging dictConfig, the Isaac Sim scripting
skills document and the API reference docs live next to this module so a
pip-installed copy of the package finds them without a repository checkout.
"""

from __future__ import annotations

from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Optional


def resource(*parts: str) -> Traversable:
    """Return the packaged data file at ``simul_mcp/resources/<parts...>``.

    Args:
        parts: Path segments below the resources directory.

    Returns:
        A Traversable handle; call ``read_text`` on it or ``is_file`` to probe.
    """
    node: Traversable = files(__name__)
    for part in parts:
        node = node.joinpath(part)
    return node


def resource_filesystem_path(*parts: str) -> Path:
    """Return the on-disk path of a packaged data file.

    The wheel is not zip-safe (the bridge extension is copied from disk by
    ``install-bridge``), so every resource has a concrete filesystem path.

    Args:
        parts: Path segments below the resources directory.

    Returns:
        Absolute path of the resource.
    """
    return Path(str(resource(*parts)))


def find_checkout_root() -> Optional[Path]:
    """Locate the repository root when the package runs from a source checkout.

    Returns:
        The directory holding ``pyproject.toml`` and ``src/simul_mcp`` for an
        editable install, or ``None`` for a wheel install where the package
        sits in ``site-packages`` and no such root exists.
    """
    candidate = Path(__file__).resolve().parents[3]
    if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "simul_mcp").is_dir():
        return candidate
    return None
