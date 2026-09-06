"""Tests for the Unreal Remote Control auto-setup patchers."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


from simul_mcp.adapters import unreal_setup
from simul_mcp.adapters.unreal_setup import (
    HEADLESS_FLAGS,
    HTTP_LISTENERS_SECTION,
    REMOTE_CONTROL_SECTION,
    ensure_remote_control_config,
    patch_default_engine_ini,
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


def test_patch_ini_writes_bind_to_websocket_address_when_set(tmp_path: Path) -> None:
    """--bind routes to RemoteControlWebsocketServerBindAddress (a real
    URemoteControlSettings field). HTTP bind is NOT in this ini — UE
    reads it from DefaultEngine.ini[HTTPServer.Listeners] instead, see
    test_patch_default_engine_ini_writes_default_bind_address. The earlier
    `RemoteControlHttpServerHostname` write was a silent no-op (field
    doesn't exist on the CDO) and was removed in iter6."""
    result = patch_remote_control_ini(tmp_path, port=30010, bind="0.0.0.0")
    text = (tmp_path / "Config" / "DefaultRemoteControl.ini").read_text()
    assert "RemoteControlWebsocketServerBindAddress=0.0.0.0" in text
    assert "RemoteControlHttpServerHostname" not in text
    assert "RemoteControlWebsocketServerBindAddress" in result.added


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


def test_patch_ini_writes_passphrase_array_and_enforce_flag(tmp_path: Path) -> None:
    """--passphrase appends a single +Passphrases array entry under the
    correct section AND pins bEnforcePassphraseForRemoteClients=True so
    the gate is explicit, not relying on UE's C++ default."""
    md5_hash = "5f4dcc3b5aa765d61d8327deb882cf99"  # md5("password")

    result = patch_remote_control_ini(
        tmp_path, port=30010, passphrase_md5=md5_hash
    )

    text = (tmp_path / "Config" / "DefaultRemoteControl.ini").read_text()
    assert (
        f'+Passphrases=(Identifier="simul",Passphrase="{md5_hash}")' in text
    )
    assert "bEnforcePassphraseForRemoteClients=True" in text
    assert "Passphrases" in result.added
    assert "bEnforcePassphraseForRemoteClients" in result.added


def test_patch_ini_passphrase_idempotent_on_same_hash(tmp_path: Path) -> None:
    """Re-running with the same passphrase hash does not duplicate the line."""
    md5_hash = "5f4dcc3b5aa765d61d8327deb882cf99"
    patch_remote_control_ini(tmp_path, port=30010, passphrase_md5=md5_hash)
    before = (tmp_path / "Config" / "DefaultRemoteControl.ini").read_text()

    result = patch_remote_control_ini(
        tmp_path, port=30010, passphrase_md5=md5_hash
    )

    assert result.changed is False
    assert (tmp_path / "Config" / "DefaultRemoteControl.ini").read_text() == before
    # Exactly one passphrase entry — no duplication.
    text = (tmp_path / "Config" / "DefaultRemoteControl.ini").read_text()
    assert text.count("+Passphrases=") == 1


def test_patch_ini_passphrase_omitted_does_not_touch_passphrase_keys(
    tmp_path: Path,
) -> None:
    """Default behavior: no passphrase, no +Passphrases line, no enforce key."""
    patch_remote_control_ini(tmp_path, port=30010)
    text = (tmp_path / "Config" / "DefaultRemoteControl.ini").read_text()
    assert "+Passphrases" not in text
    assert "bEnforcePassphraseForRemoteClients" not in text


def test_patch_ini_appends_second_passphrase_when_hash_differs(
    tmp_path: Path,
) -> None:
    """Documented behavior: a different passphrase hash on a subsequent
    run appends an additional +Passphrases entry rather than overwriting.
    UE accepts any matching entry per WebRemoteControlInternalUtils.cpp's
    CheckPassphrase, so this is non-destructive multi-tenancy. Pin the
    invariant so a future refactor can't silently break it."""
    first = "5f4dcc3b5aa765d61d8327deb882cf99"   # md5("password")
    second = "21232f297a57a5a743894a0e4a801fc3"  # md5("admin")
    patch_remote_control_ini(tmp_path, port=30010, passphrase_md5=first)
    patch_remote_control_ini(tmp_path, port=30010, passphrase_md5=second)
    text = (tmp_path / "Config" / "DefaultRemoteControl.ini").read_text()
    assert text.count("+Passphrases=") == 2
    assert f'Passphrase="{first}"' in text
    assert f'Passphrase="{second}"' in text


def test_patch_ini_updates_bind_when_value_differs(tmp_path: Path) -> None:
    """Changing --bind updates the in-place WS bind value (the real key
    that lives on URemoteControlSettings)."""
    patch_remote_control_ini(tmp_path, port=30010, bind="127.0.0.1")
    result = patch_remote_control_ini(tmp_path, port=30010, bind="0.0.0.0")
    text = (tmp_path / "Config" / "DefaultRemoteControl.ini").read_text()
    assert "RemoteControlWebsocketServerBindAddress=0.0.0.0" in text
    assert "RemoteControlWebsocketServerBindAddress=127.0.0.1" not in text
    assert "RemoteControlWebsocketServerBindAddress" in result.updated
    # Exactly one occurrence — no duplicate keys appended.
    assert text.count("RemoteControlWebsocketServerBindAddress=") == 1


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
    # No --bind → engine_ini patch is not applied; DefaultEngine.ini stays
    # untouched and the SetupResult reflects that.
    assert result.engine_ini is None
    assert not (tmp_path / "Config" / "DefaultEngine.ini").exists()


# ---------------------------------------------------------------------------
# patch_default_engine_ini — UE's HTTP listener reads the bind address from
# DefaultEngine.ini[HTTPServer.Listeners] DefaultBindAddress, NOT from any
# field on URemoteControlSettings (iter6 traced this through UE 5.x source
# at HttpServerConfig.cpp:11-17 and HttpListener.cpp:62-92).
# ---------------------------------------------------------------------------


def test_patch_default_engine_ini_writes_default_bind_address(tmp_path: Path) -> None:
    """Fresh-file write: creates Config/DefaultEngine.ini with the section
    and DefaultBindAddress key."""
    result = patch_default_engine_ini(tmp_path, bind="0.0.0.0")

    text = (tmp_path / "Config" / "DefaultEngine.ini").read_text()
    assert f"[{HTTP_LISTENERS_SECTION}]" in text
    assert "DefaultBindAddress=0.0.0.0" in text
    assert result.changed is True
    assert "DefaultBindAddress" in result.added


def test_patch_default_engine_ini_idempotent(tmp_path: Path) -> None:
    """Re-running with the same bind value is a no-op."""
    patch_default_engine_ini(tmp_path, bind="0.0.0.0")
    before = (tmp_path / "Config" / "DefaultEngine.ini").read_text()

    result = patch_default_engine_ini(tmp_path, bind="0.0.0.0")

    assert result.changed is False
    assert (tmp_path / "Config" / "DefaultEngine.ini").read_text() == before


def test_patch_default_engine_ini_updates_in_place(tmp_path: Path) -> None:
    """Changing bind updates the value, doesn't append a duplicate."""
    patch_default_engine_ini(tmp_path, bind="127.0.0.1")
    result = patch_default_engine_ini(tmp_path, bind="0.0.0.0")

    text = (tmp_path / "Config" / "DefaultEngine.ini").read_text()
    assert "DefaultBindAddress=0.0.0.0" in text
    assert "DefaultBindAddress=127.0.0.1" not in text
    assert "DefaultBindAddress" in result.updated
    assert text.count("DefaultBindAddress=") == 1


def test_patch_default_engine_ini_section_exists_key_absent(
    tmp_path: Path,
) -> None:
    """Realistic operator scenario: DefaultEngine.ini already has the
    [HTTPServer.Listeners] section with other keys (e.g. listener-port
    overrides set by another tool), but DefaultBindAddress is absent.
    The patcher must add the key inside the existing section, NOT
    duplicate the header."""
    config_dir = tmp_path / "Config"
    config_dir.mkdir()
    (config_dir / "DefaultEngine.ini").write_text(
        "[HTTPServer.Listeners]\n"
        "ListenerPort=8080\n",
        encoding="utf-8",
    )

    result = patch_default_engine_ini(tmp_path, bind="0.0.0.0")

    text = (config_dir / "DefaultEngine.ini").read_text()
    # Existing key preserved, new key added.
    assert "ListenerPort=8080" in text
    assert "DefaultBindAddress=0.0.0.0" in text
    # Section header appears exactly once — no duplication.
    assert text.count("[HTTPServer.Listeners]") == 1
    assert "DefaultBindAddress" in result.added


def test_patch_default_engine_ini_preserves_unrelated_sections(
    tmp_path: Path,
) -> None:
    """Existing DefaultEngine.ini content is preserved verbatim — UE
    projects routinely have many sections in this file (CoreRedirects,
    Renderer, etc.) and we must not nuke them."""
    config_dir = tmp_path / "Config"
    config_dir.mkdir()
    (config_dir / "DefaultEngine.ini").write_text(
        "[CoreRedirects]\n+ClassRedirects=(OldName=\"X\",NewName=\"Y\")\n\n"
        "[/Script/Engine.RendererSettings]\nr.Foo=42\n",
        encoding="utf-8",
    )

    patch_default_engine_ini(tmp_path, bind="0.0.0.0")

    text = (config_dir / "DefaultEngine.ini").read_text()
    assert "[CoreRedirects]" in text
    assert "+ClassRedirects=(OldName=\"X\",NewName=\"Y\")" in text
    assert "[/Script/Engine.RendererSettings]" in text
    assert "r.Foo=42" in text
    assert f"[{HTTP_LISTENERS_SECTION}]" in text
    assert "DefaultBindAddress=0.0.0.0" in text


def test_ensure_remote_control_config_writes_engine_ini_when_bind_supplied(
    tmp_path: Path,
) -> None:
    """ensure_remote_control_config wires the engine-ini patcher into the
    aggregate result whenever --bind is supplied."""
    uproject = _write_uproject(tmp_path, {"FileVersion": 3})

    result = ensure_remote_control_config(uproject, port=30010, bind="0.0.0.0")

    assert result.engine_ini is not None
    assert result.engine_ini.changed is True
    engine_text = (tmp_path / "Config" / "DefaultEngine.ini").read_text()
    assert "DefaultBindAddress=0.0.0.0" in engine_text
    # WS bind also routed through RC ini for symmetry.
    rc_text = (tmp_path / "Config" / "DefaultRemoteControl.ini").read_text()
    assert "RemoteControlWebsocketServerBindAddress=0.0.0.0" in rc_text
    # The bogus iter1 key must not appear — it doesn't exist on UE's
    # URemoteControlSettings CDO.
    assert "RemoteControlHttpServerHostname" not in rc_text


# ---------------------------------------------------------------------------
# Passphrase digest written into a git-visible ini
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=t", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


@needs_git
def test_passphrase_warns_when_ini_is_tracked(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    patch_remote_control_ini(tmp_path, port=30010)
    _git(tmp_path, "add", "Config/DefaultRemoteControl.ini")
    _git(tmp_path, "commit", "-q", "-m", "ini")

    result = patch_remote_control_ini(tmp_path, port=30010, passphrase_md5="d41d8cd98f00b204e9800998ecf8427e")

    assert len(result.warnings) == 1
    assert "tracked by git" in result.warnings[0]
    assert "passphrase" in result.warnings[0]


@needs_git
def test_passphrase_warns_when_ini_is_untracked_but_not_ignored(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")

    result = patch_remote_control_ini(tmp_path, port=30010, passphrase_md5="d41d8cd98f00b204e9800998ecf8427e")

    assert len(result.warnings) == 1
    assert "not ignored" in result.warnings[0]


@needs_git
def test_passphrase_is_quiet_when_ini_is_gitignored(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_text("Config/\n", encoding="utf-8")

    result = patch_remote_control_ini(tmp_path, port=30010, passphrase_md5="d41d8cd98f00b204e9800998ecf8427e")

    assert result.warnings == []


def test_passphrase_is_quiet_outside_a_repository(tmp_path: Path) -> None:
    result = patch_remote_control_ini(tmp_path, port=30010, passphrase_md5="d41d8cd98f00b204e9800998ecf8427e")

    assert result.warnings == []


def test_no_passphrase_never_consults_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*args, **kwargs):
        raise AssertionError("git must not be invoked without a passphrase")

    monkeypatch.setattr(unreal_setup.subprocess, "run", explode)
    result = patch_remote_control_ini(tmp_path, port=30010)

    assert result.warnings == []
