"""Isaac Sim install inspection: VERSION parsing and transport extension choice."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_path))

from simul_mcp.adapters.isaac_install import (  # noqa: E402
    PYTHON_SERVER_EXTENSION,
    VSCODE_EXTENSION,
    IsaacVersion,
    read_isaac_version,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("5.1.0-rc.19+release.26219.9c81211b.gl", IsaacVersion(5, 1, 0)),
        ("6.0.0-rc.59+release.41464.5f2772bc.gl\n", IsaacVersion(6, 0, 0)),
        ("6.0.1-rc.7+release.42383.32955d8d.gl", IsaacVersion(6, 0, 1)),
        ("4.5.0", IsaacVersion(4, 5, 0)),
    ],
)
def test_parse_reads_leading_semver(text: str, expected: IsaacVersion) -> None:
    assert IsaacVersion.parse(text) == expected


def test_parse_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        IsaacVersion.parse("isaac-sim")


def test_six_uses_python_server_and_five_uses_vscode() -> None:
    assert IsaacVersion(6, 0, 0).python_transport_extension == PYTHON_SERVER_EXTENSION
    assert IsaacVersion(6, 0, 1).python_transport_extension == PYTHON_SERVER_EXTENSION
    assert IsaacVersion(5, 1, 0).python_transport_extension == VSCODE_EXTENSION


def test_read_isaac_version_from_install_root(tmp_path: Path) -> None:
    (tmp_path / "VERSION").write_text("6.0.1-rc.7+release.42383.32955d8d.gl\n")
    assert read_isaac_version(tmp_path) == IsaacVersion(6, 0, 1)
    assert str(read_isaac_version(tmp_path)) == "6.0.1"


def test_read_isaac_version_missing_or_unparseable(tmp_path: Path) -> None:
    assert read_isaac_version(tmp_path) is None
    (tmp_path / "VERSION").write_text("not a version")
    assert read_isaac_version(tmp_path) is None


def test_year_scheme_versions_are_not_treated_as_six_or_newer() -> None:
    """Isaac Sim 2023.1.1 parses as major 2023; it must not select python_server."""
    legacy = IsaacVersion.parse("2023.1.1")
    assert legacy.is_supported is False
    with pytest.raises(ValueError):
        legacy.python_transport_extension


def test_supported_range_covers_five_and_six_only() -> None:
    assert IsaacVersion(5, 1, 0).is_supported is True
    assert IsaacVersion(6, 0, 1).is_supported is True
    assert IsaacVersion(4, 5, 0).is_supported is False
    assert IsaacVersion(7, 0, 0).is_supported is False
