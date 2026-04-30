"""Tests for the Unreal Remote Control auto-setup patchers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_path))

from simul_mcp.adapters.unreal_setup import (  # noqa: E402
    REMOTE_CONTROL_SECTION,
    ensure_remote_control_config,
    patch_remote_control_ini,
    patch_uproject,
)


# ---------------------------------------------------------------------------
# .uproject patching
# ---------------------------------------------------------------------------


def _write_uproject(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "Demo.uproject"
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p


def test_patch_uproject_adds_missing_plugins(tmp_path: Path) -> None:
    uproject = _write_uproject(tmp_path, {"FileVersion": 3, "EngineAssociation": "5.4"})

    result = patch_uproject(uproject)

    assert result.changed is True
    assert set(result.added) == {"RemoteControl", "PythonScriptPlugin"}
    data = json.loads(uproject.read_text())
    names = {p["Name"]: p for p in data["Plugins"]}
    assert names["RemoteControl"]["Enabled"] is True
    assert names["PythonScriptPlugin"]["Enabled"] is True


def test_patch_uproject_enables_present_but_disabled_plugins(tmp_path: Path) -> None:
    uproject = _write_uproject(
        tmp_path,
        {
            "Plugins": [
                {"Name": "RemoteControl", "Enabled": False},
                {"Name": "PythonScriptPlugin", "Enabled": True},
                {"Name": "SomeOther", "Enabled": True},
            ]
        },
    )

    result = patch_uproject(uproject)

    assert result.changed is True
    assert result.updated == ["RemoteControl"]
    assert result.already_ok == ["PythonScriptPlugin"]
    data = json.loads(uproject.read_text())
    # Unrelated plugin preserved.
    assert any(p["Name"] == "SomeOther" for p in data["Plugins"])


def test_patch_uproject_noop_when_already_configured(tmp_path: Path) -> None:
    uproject = _write_uproject(
        tmp_path,
        {
            "Plugins": [
                {"Name": "RemoteControl", "Enabled": True},
                {"Name": "PythonScriptPlugin", "Enabled": True},
            ]
        },
    )
    before = uproject.read_text()

    result = patch_uproject(uproject)

    assert result.changed is False
    assert result.added == [] and result.updated == []
    assert uproject.read_text() == before  # file untouched


def test_patch_uproject_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        patch_uproject(tmp_path / "missing.uproject")


# ---------------------------------------------------------------------------
# DefaultRemoteControl.ini patching
# ---------------------------------------------------------------------------


def test_patch_ini_creates_fresh_file(tmp_path: Path) -> None:
    result = patch_remote_control_ini(tmp_path, port=30010)

    assert result.changed is True
    assert set(result.added) >= {
        "bAutoStartWebServer",
        "bAutoStartWebSocketServer",
        "RemoteControlHttpServerPort",
        "bRestrictServerAccess",
        "bEnableRemotePythonExecution",
    }
    text = (tmp_path / "Config" / "DefaultRemoteControl.ini").read_text()
    assert f"[{REMOTE_CONTROL_SECTION}]" in text
    assert "RemoteControlHttpServerPort=30010" in text
    assert "bRestrictServerAccess=True" in text


def test_patch_ini_respects_custom_port(tmp_path: Path) -> None:
    patch_remote_control_ini(tmp_path, port=31111)
    text = (tmp_path / "Config" / "DefaultRemoteControl.ini").read_text()
    assert "RemoteControlHttpServerPort=31111" in text


def test_patch_ini_preserves_unrelated_sections(tmp_path: Path) -> None:
    config_dir = tmp_path / "Config"
    config_dir.mkdir()
    (config_dir / "DefaultRemoteControl.ini").write_text(
        "[/Script/Other.Thing]\n"
        "KeepMe=Yes\n"
        "\n"
        f"[{REMOTE_CONTROL_SECTION}]\n"
        "bAutoStartWebServer=False\n",
        encoding="utf-8",
    )

    result = patch_remote_control_ini(tmp_path, port=30010)

    text = (config_dir / "DefaultRemoteControl.ini").read_text()
    assert result.changed is True
    assert "bAutoStartWebServer" in result.updated
    assert "KeepMe=Yes" in text  # unrelated section preserved
    assert "bAutoStartWebServer=True" in text
    assert "bEnableRemotePythonExecution=True" in text


def test_patch_ini_noop_when_already_correct(tmp_path: Path) -> None:
    config_dir = tmp_path / "Config"
    config_dir.mkdir()
    target = config_dir / "DefaultRemoteControl.ini"
    target.write_text(
        f"[{REMOTE_CONTROL_SECTION}]\n"
        "bAutoStartWebServer=True\n"
        "bAutoStartWebSocketServer=True\n"
        "RemoteControlHttpServerPort=30010\n"
        "bRestrictServerAccess=True\n"
        "bEnableRemotePythonExecution=True\n",
        encoding="utf-8",
    )
    before = target.read_text()

    result = patch_remote_control_ini(tmp_path, port=30010)

    assert result.changed is False
    assert target.read_text() == before


# ---------------------------------------------------------------------------
# ensure_remote_control_config (aggregate)
# ---------------------------------------------------------------------------


def test_ensure_remote_control_config_runs_both_patches(tmp_path: Path) -> None:
    uproject = _write_uproject(tmp_path, {"FileVersion": 3})

    result = ensure_remote_control_config(uproject, port=30010)

    assert result.changed is True
    assert result.uproject.changed is True
    assert result.ini.changed is True
    assert (tmp_path / "Config" / "DefaultRemoteControl.ini").is_file()
