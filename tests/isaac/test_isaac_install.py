"""Isaac Sim install inspection: VERSION parsing and transport extension choice."""

from __future__ import annotations

from pathlib import Path

import pytest


from simul_mcp.adapters.isaac_install import (
    NEWEST_KNOWN_MAJOR,
    PYTHON_SERVER_EXTENSION,
    PYTHON_SOCKET_PORT_SETTINGS,
    PYTHON_TRANSPORT_EXTENSIONS,
    VSCODE_EXTENSION,
    IsaacVersion,
    read_isaac_version,
)

extension_root = Path(__file__).resolve().parents[2] / "src" / "simul_mcp" / "bridge_ext" / "khemoo.simul.mcp"

from khemoo.simul.mcp import extension as bridge_extension  # noqa: E402


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
    assert legacy.support_level == "unsupported"
    with pytest.raises(ValueError):
        legacy.python_transport_extension


def test_support_levels_by_major() -> None:
    assert IsaacVersion(5, 1, 0).support_level == "supported"
    assert IsaacVersion(6, 0, 1).support_level == "supported"
    assert IsaacVersion(4, 5, 0).support_level == "unsupported"
    assert IsaacVersion(7, 0, 0).support_level == "assumed"
    assert IsaacVersion(12, 0, 0).support_level == "assumed"
    assert IsaacVersion(2022, 2, 1).support_level == "unsupported"


def test_newer_major_assumes_the_newest_known_extension() -> None:
    """Isaac Sim 7 is routed like 6, with the caller expected to warn."""
    assert NEWEST_KNOWN_MAJOR == 6
    assert IsaacVersion(7, 0, 0).python_transport_extension == PYTHON_TRANSPORT_EXTENSIONS[NEWEST_KNOWN_MAJOR]
    assert IsaacVersion(7, 0, 0).python_transport_extension == PYTHON_SERVER_EXTENSION


def test_older_major_has_no_transport_extension() -> None:
    with pytest.raises(ValueError):
        IsaacVersion(4, 5, 0).python_transport_extension


def test_port_setting_table_matches_the_bridge_extension() -> None:
    """The bridge runs inside Kit and cannot import simul_mcp, so it carries its own copy."""
    assert bridge_extension.PYTHON_SOCKET_PORT_SETTINGS == PYTHON_SOCKET_PORT_SETTINGS
    newest_first = [PYTHON_TRANSPORT_EXTENSIONS[major] for major in sorted(PYTHON_TRANSPORT_EXTENSIONS, reverse=True)]
    assert PYTHON_SOCKET_PORT_SETTINGS == tuple(f"/exts/{ext}/port" for ext in newest_first)
