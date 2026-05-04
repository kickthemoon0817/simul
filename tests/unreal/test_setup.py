"""Tests for the Unreal Remote Control auto-setup patchers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_path))

from simul_mcp.adapters.unreal_setup import (  # noqa: E402
    HEADLESS_FLAGS,
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


def test_remote_control_section_targets_RemoteControlCommon_module() -> None:
    """Pinning regression for issue #44.

    URemoteControlSettings lives in module RemoteControlCommon, so UE's
    config loader expects the section under /Script/RemoteControlCommon.
    Earlier versions of this patcher used /Script/RemoteControl which
    UE silently ignored — keys appeared to work because their C++
    defaults already matched what we wanted, but bRestrictServerAccess
    and bEnableRemotePythonExecution (whose C++ defaults are False)
    never actually flipped to True, gating remote Python execution
    on UE 5.3+. Don't regress.
    """
    assert REMOTE_CONTROL_SECTION == "/Script/RemoteControlCommon.RemoteControlSettings"


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


def test_patch_ini_omits_bind_and_websocket_when_unset(tmp_path: Path) -> None:
    """Default behavior: leave UE's defaults alone for hostname + WS port."""
    patch_remote_control_ini(tmp_path, port=30010)
    text = (tmp_path / "Config" / "DefaultRemoteControl.ini").read_text()
    assert "RemoteControlHttpServerHostname" not in text
    assert "RemoteControlWebSocketServerPort" not in text


def test_patch_ini_writes_bind_when_set(tmp_path: Path) -> None:
    """Cross-host enable: --bind writes RemoteControlHttpServerHostname."""
    result = patch_remote_control_ini(tmp_path, port=30010, bind="0.0.0.0")
    text = (tmp_path / "Config" / "DefaultRemoteControl.ini").read_text()
    assert "RemoteControlHttpServerHostname=0.0.0.0" in text
    assert "RemoteControlHttpServerHostname" in result.added


def test_patch_ini_writes_websocket_port_when_set(tmp_path: Path) -> None:
    """Multi-instance enable: --websocket-port writes the WS port override."""
    result = patch_remote_control_ini(tmp_path, port=30010, websocket_port=30022)
    text = (tmp_path / "Config" / "DefaultRemoteControl.ini").read_text()
    assert "RemoteControlWebSocketServerPort=30022" in text
    assert "RemoteControlWebSocketServerPort" in result.added


def test_patch_ini_idempotent_with_bind_and_websocket_port(tmp_path: Path) -> None:
    """Re-running with the same overrides is a no-op."""
    patch_remote_control_ini(
        tmp_path, port=30010, bind="0.0.0.0", websocket_port=30022
    )
    before = (tmp_path / "Config" / "DefaultRemoteControl.ini").read_text()

    result = patch_remote_control_ini(
        tmp_path, port=30010, bind="0.0.0.0", websocket_port=30022
    )

    assert result.changed is False
    assert (tmp_path / "Config" / "DefaultRemoteControl.ini").read_text() == before


def test_patch_ini_updates_bind_when_value_differs(tmp_path: Path) -> None:
    """Changing --bind updates the in-place value, doesn't duplicate it."""
    patch_remote_control_ini(tmp_path, port=30010, bind="127.0.0.1")
    result = patch_remote_control_ini(tmp_path, port=30010, bind="0.0.0.0")
    text = (tmp_path / "Config" / "DefaultRemoteControl.ini").read_text()
    assert "RemoteControlHttpServerHostname=0.0.0.0" in text
    assert "RemoteControlHttpServerHostname=127.0.0.1" not in text
    assert "RemoteControlHttpServerHostname" in result.updated
    # Exactly one occurrence — no duplicate keys appended.
    assert text.count("RemoteControlHttpServerHostname=") == 1


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


def test_headless_flags_cover_window_focus_and_logging() -> None:
    """The HEADLESS_FLAGS tuple is what makes UE skip the GUI but keep
    rendering. Lock in the exact set so a refactor can't drop the
    -RenderOffScreen flag (the one that decouples capture from focus)."""
    expected = {
        "-RenderOffScreen",
        "-unattended",
        "-nopause",
        "-nosplash",
        "-nosound",
        "-stdout",
        "-FullStdOutLogOutput",
    }
    assert set(HEADLESS_FLAGS) == expected


def test_resolve_launch_argv_appends_headless_flags(tmp_path: Path, monkeypatch) -> None:
    """`--headless` must propagate all the way through to argv."""
    from simul_mcp.adapters import unreal_setup as us

    uproject = _write_uproject(tmp_path, {"FileVersion": 3})

    # Stub platform.system → 'Linux' and force a fake engine_path so neither
    # the host's actual UE install nor LaunchServices is consulted.
    monkeypatch.setattr(us.platform, "system", lambda: "Linux")
    fake_engine = tmp_path / "engine"
    binary = fake_engine / "Engine" / "Binaries" / "Linux" / "UnrealEditor"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)

    gui_argv = us.resolve_launch_argv(uproject, engine_path=fake_engine, headless=False)
    headless_argv = us.resolve_launch_argv(uproject, engine_path=fake_engine, headless=True)

    assert gui_argv == [str(binary), str(uproject)]
    assert headless_argv == [str(binary), str(uproject), *HEADLESS_FLAGS]
    # Default keeps GUI behavior — explicit opt-in is required.
    default_argv = us.resolve_launch_argv(uproject, engine_path=fake_engine)
    assert default_argv == gui_argv


def test_ensure_remote_control_config_runs_both_patches(tmp_path: Path) -> None:
    uproject = _write_uproject(tmp_path, {"FileVersion": 3})

    result = ensure_remote_control_config(uproject, port=30010)

    assert result.changed is True
    assert result.uproject.changed is True
    assert result.ini.changed is True
    assert (tmp_path / "Config" / "DefaultRemoteControl.ini").is_file()
