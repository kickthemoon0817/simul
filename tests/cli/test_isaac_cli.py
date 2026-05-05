"""CLI tests for Isaac bridge inspection and control commands."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from typer.testing import CliRunner

src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_path))

from simul_mcp.adapters.isaac_socket_client import ScriptResult  # noqa: E402
from simul_mcp.cli import isaac as isaac_cli  # noqa: E402
from simul_mcp.cli.main import app  # noqa: E402


runner = CliRunner()


def _make_tools() -> SimpleNamespace:
    """Create a mocked IsaacTools-style namespace for CLI tests."""
    client = SimpleNamespace(
        address="127.0.0.1:8229",
        bridge_address="127.0.0.1:8229",
        bridge_enabled=True,
        bridge_request=AsyncMock(),
        execute_vscode_only=AsyncMock(),
    )
    return SimpleNamespace(_client=client)


def test_bridge_capabilities_json(monkeypatch) -> None:
    """The CLI should expose bridge permission state from capabilities."""
    tools = _make_tools()
    tools._client.bridge_request.return_value = {
        "status": "ok",
        "protocol_version": "1.0",
        "payload": {
            "transport": "simul_bridge",
            "allow_unsafe_execution": True,
            "actions": ["ping", "capabilities", "execute_script"],
        },
    }
    monkeypatch.setattr(isaac_cli, "_tools", lambda *args, **kwargs: tools)

    result = runner.invoke(app, ["--json", "isaac", "bridge-capabilities"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["allow_unsafe_execution"] is True
    assert payload["bridge_address"] == "127.0.0.1:8229"
    assert "execute_script" in payload["actions"]


def test_bridge_config_json(monkeypatch) -> None:
    """The CLI should read the running bridge config through VS Code."""
    tools = _make_tools()
    tools._client.execute_vscode_only.return_value = ScriptResult(
        success=True,
        output=json.dumps(
            {
                "extension_enabled": True,
                "host": "127.0.0.1",
                "port": 8229,
                "allow_unsafe_execution": True,
                "max_request_bytes": 1048576,
                "max_response_bytes": 10485760,
            }
        ),
        transport="vscode",
    )
    monkeypatch.setattr(isaac_cli, "_tools", lambda *args, **kwargs: tools)

    result = runner.invoke(app, ["--json", "isaac", "bridge-config"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["extension_enabled"] is True
    assert payload["allow_unsafe_execution"] is True
    assert payload["port"] == 8229


def test_bridge_set_unsafe_json(monkeypatch) -> None:
    """The CLI should update unsafe execution and surface restart status."""
    tools = _make_tools()
    tools._client.execute_vscode_only.return_value = ScriptResult(
        success=True,
        output=json.dumps(
            {
                "allow_unsafe_execution": False,
                "restart_requested": True,
                "restarted": True,
            }
        ),
        transport="vscode",
    )
    monkeypatch.setattr(isaac_cli, "_tools", lambda *args, **kwargs: tools)

    result = runner.invoke(app, ["--json", "isaac", "bridge-set-unsafe", "--disable"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["allow_unsafe_execution"] is False
    assert payload["restarted"] is True


# ---------------------------------------------------------------------------
# bridge-up — closes the iter8 UX gap (manual enable-extension workaround).
# ---------------------------------------------------------------------------


def _make_bridge_up_tools() -> SimpleNamespace:
    """Mock tools with the extra fields/methods bridge-up needs."""
    tools = _make_tools()
    tools._client.vscode_address = "127.0.0.1:8226"
    tools.enable_isaac_extension = AsyncMock()
    return tools


def test_bridge_up_already_reachable(monkeypatch) -> None:
    """First branch: bridge already responds → action=already-up, no enable call."""
    tools = _make_bridge_up_tools()
    tools._client.bridge_request.return_value = {"status": "ok"}
    monkeypatch.setattr(isaac_cli, "_tools", lambda *args, **kwargs: tools)

    result = runner.invoke(app, ["--json", "isaac", "bridge-up"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["action"] == "already-up"
    assert payload["bridge_reachable"] is True
    assert payload["success"] is True
    tools.enable_isaac_extension.assert_not_called()


def test_bridge_up_isaac_not_running(monkeypatch) -> None:
    """Second branch: neither bridge nor VS Code reachable → exit non-zero
    with NotRunning error."""
    tools = _make_bridge_up_tools()
    tools._client.bridge_request.side_effect = ConnectionRefusedError("nope")
    tools._client.execute_vscode_only.side_effect = ConnectionRefusedError("nope")
    monkeypatch.setattr(isaac_cli, "_tools", lambda *args, **kwargs: tools)

    result = runner.invoke(app, ["--json", "isaac", "bridge-up"])

    assert result.exit_code != 0
    assert "NotRunning" in result.stdout
    tools.enable_isaac_extension.assert_not_called()


def test_bridge_up_extension_enable_fails(monkeypatch) -> None:
    """Third branch: bridge down, VS Code up, but enable-extension fails →
    exit non-zero with ExtensionNotRegistered."""
    tools = _make_bridge_up_tools()
    tools._client.bridge_request.side_effect = ConnectionRefusedError("nope")
    tools._client.execute_vscode_only.return_value = ScriptResult(
        success=True, output="pong\n", transport="vscode"
    )
    tools.enable_isaac_extension.return_value = {
        "success": False,
        "error": "Extension not found: khemoo.simul.mcp",
    }
    monkeypatch.setattr(isaac_cli, "_tools", lambda *args, **kwargs: tools)

    result = runner.invoke(app, ["--json", "isaac", "bridge-up"])

    assert result.exit_code != 0
    assert "ExtensionNotRegistered" in result.stdout
    tools.enable_isaac_extension.assert_awaited_once()


def test_bridge_up_auto_enables_then_reachable(monkeypatch) -> None:
    """Fourth branch (the win): bridge down, VS Code up, enable succeeds,
    bridge then becomes reachable on re-probe → action=auto-enabled,
    success=True. The whole point of the iter8 finding being closed.

    Iter10 review HIGH: the post-enable re-probe now retries up to 6
    times; this test gives it a couple of refusals before success to
    exercise the retry loop's middle case."""
    tools = _make_bridge_up_tools()
    tools._client.bridge_request.side_effect = [
        ConnectionRefusedError("initial probe — bridge not enabled"),
        ConnectionRefusedError("retry 1 — port not bound yet"),
        {"status": "ok"},  # retry 2 succeeds
    ]
    tools._client.execute_vscode_only.return_value = ScriptResult(
        success=True, output="pong\n", transport="vscode"
    )
    tools.enable_isaac_extension.return_value = {
        "success": True,
        "enabled": True,
        "extension_id": "khemoo.simul.mcp-0.0.31",
    }
    monkeypatch.setattr(isaac_cli, "_tools", lambda *args, **kwargs: tools)
    # Strip retry sleep for fast test — actual prod delay is 0.5 s.
    monkeypatch.setattr(
        isaac_cli.asyncio, "sleep", AsyncMock(return_value=None)
    )

    result = runner.invoke(app, ["--json", "isaac", "bridge-up"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["action"] == "auto-enabled"
    assert payload["bridge_reachable"] is True
    assert payload["extension_enabled"] is True
    assert payload["success"] is True
    tools.enable_isaac_extension.assert_awaited_once_with(
        extension_id="khemoo.simul.mcp"
    )


def _write_bridge_source(root: Path, version: str) -> Path:
    """Build a minimal bridge-ext source dir for install-bridge tests."""
    source = root / "exts" / "khemoo.simul.mcp"
    (source / "config").mkdir(parents=True, exist_ok=True)
    (source / "config" / "extension.toml").write_text(
        f'[package]\nversion = "{version}"\n', encoding="utf-8"
    )
    (source / "khemoo").mkdir(parents=True, exist_ok=True)
    (source / "khemoo" / "__init__.py").write_text("", encoding="utf-8")
    return source


def _make_isaac_root(root: Path) -> Path:
    isaac_root = root / "isaac-sim"
    (isaac_root / "extsUser").mkdir(parents=True, exist_ok=True)
    return isaac_root


# ---------------------------------------------------------------------------
# install-bridge — closes the iter11 publish gap (Isaac loads from extsUser,
# repo bumps don't propagate without an explicit copy/symlink).
# ---------------------------------------------------------------------------


def test_install_bridge_refuses_without_isaac_root(tmp_path: Path, monkeypatch) -> None:
    """No --isaac-root and no $ISAAC_SIM_PATH → InvalidArgument exit."""
    source = _write_bridge_source(tmp_path, "0.0.33")
    monkeypatch.delenv("ISAAC_SIM_PATH", raising=False)

    result = runner.invoke(
        app,
        ["--json", "isaac", "install-bridge", "--source", str(source)],
    )

    assert result.exit_code != 0
    assert "InvalidArgument" in result.stdout
    assert "ISAAC_SIM_PATH" in result.stdout


def test_install_bridge_refuses_when_extsUser_missing(tmp_path: Path) -> None:
    """isaac-root that's not actually an Isaac install (no extsUser dir) →
    refuses BEFORE writing anything."""
    source = _write_bridge_source(tmp_path, "0.0.33")
    fake_isaac = tmp_path / "not-actually-isaac"
    fake_isaac.mkdir()

    result = runner.invoke(
        app,
        [
            "--json", "isaac", "install-bridge",
            "--isaac-root", str(fake_isaac),
            "--source", str(source),
        ],
    )

    assert result.exit_code != 0
    assert "extsUser" in result.stdout


def test_install_bridge_copies_into_extsUser(tmp_path: Path) -> None:
    """Fresh install: dest doesn't exist → action=copied, version matches."""
    source = _write_bridge_source(tmp_path, "0.0.33")
    isaac_root = _make_isaac_root(tmp_path)

    result = runner.invoke(
        app,
        [
            "--json", "isaac", "install-bridge",
            "--isaac-root", str(isaac_root),
            "--source", str(source),
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["action"] == "copied"
    assert payload["version"] == "0.0.33"
    assert payload["previous_version"] is None
    assert payload["success"] is True
    # File system: dest exists with correct version.
    dest = isaac_root / "extsUser" / "khemoo.simul.mcp"
    assert (dest / "config" / "extension.toml").is_file()
    assert (dest / "khemoo" / "__init__.py").is_file()


def test_install_bridge_already_current_no_op(tmp_path: Path) -> None:
    """Re-running with the same version is a clean no-op (action=already-current).
    No --force, no copy, no error."""
    source = _write_bridge_source(tmp_path, "0.0.33")
    isaac_root = _make_isaac_root(tmp_path)

    # First install
    runner.invoke(app, [
        "--json", "isaac", "install-bridge",
        "--isaac-root", str(isaac_root),
        "--source", str(source),
    ])
    dest_toml = isaac_root / "extsUser" / "khemoo.simul.mcp" / "config" / "extension.toml"
    mtime_before = dest_toml.stat().st_mtime

    # Re-run
    result = runner.invoke(app, [
        "--json", "isaac", "install-bridge",
        "--isaac-root", str(isaac_root),
        "--source", str(source),
    ])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["action"] == "already-current"
    assert payload["version"] == "0.0.33"
    # File untouched.
    assert dest_toml.stat().st_mtime == mtime_before


def test_install_bridge_replaces_stale_dest(tmp_path: Path) -> None:
    """Pre-existing stale dest (e.g. iter11's verifier scenario: dest at
    0.0.13, source at 0.0.33) → action=copied, previous_version captured,
    new version verified."""
    isaac_root = _make_isaac_root(tmp_path)
    # Pre-populate stale dest at 0.0.13.
    stale_dest = isaac_root / "extsUser" / "khemoo.simul.mcp"
    (stale_dest / "config").mkdir(parents=True)
    (stale_dest / "config" / "extension.toml").write_text(
        '[package]\nversion = "0.0.13"\n', encoding="utf-8"
    )
    # New source at 0.0.33.
    source = _write_bridge_source(tmp_path, "0.0.33")

    result = runner.invoke(app, [
        "--json", "isaac", "install-bridge",
        "--isaac-root", str(isaac_root),
        "--source", str(source),
    ])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["action"] == "copied"
    assert payload["previous_version"] == "0.0.13"
    assert payload["version"] == "0.0.33"
    assert payload["success"] is True


def test_install_bridge_symlink_mode(tmp_path: Path) -> None:
    """--symlink uses ln -s instead of copy. Repo edits then propagate
    automatically (no re-install needed)."""
    source = _write_bridge_source(tmp_path, "0.0.33")
    isaac_root = _make_isaac_root(tmp_path)

    result = runner.invoke(app, [
        "--json", "isaac", "install-bridge",
        "--isaac-root", str(isaac_root),
        "--source", str(source),
        "--symlink",
    ])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["action"] == "symlinked"
    assert payload["version"] == "0.0.33"
    dest = isaac_root / "extsUser" / "khemoo.simul.mcp"
    assert dest.is_symlink()
    assert dest.resolve() == source


def test_install_bridge_force_replaces_matching_version(tmp_path: Path) -> None:
    """--force replaces dest even when versions match (e.g. mode change
    from copy to symlink, or stale-but-same-version content)."""
    source = _write_bridge_source(tmp_path, "0.0.33")
    isaac_root = _make_isaac_root(tmp_path)
    runner.invoke(app, [
        "--json", "isaac", "install-bridge",
        "--isaac-root", str(isaac_root),
        "--source", str(source),
    ])

    # Force re-install as symlink.
    result = runner.invoke(app, [
        "--json", "isaac", "install-bridge",
        "--isaac-root", str(isaac_root),
        "--source", str(source),
        "--force", "--symlink",
    ])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["action"] == "symlinked"
    dest = isaac_root / "extsUser" / "khemoo.simul.mcp"
    assert dest.is_symlink()


def test_bridge_up_enable_succeeds_but_bridge_stays_down(monkeypatch) -> None:
    """Fifth branch (test-engineer's flagged gap): enable succeeded but
    bridge port still doesn't bind even after the retry loop
    exhausts. action=auto-enabled but success=False, exit non-zero."""
    tools = _make_bridge_up_tools()
    # All 7 probes (1 initial + 6 retries) refuse.
    tools._client.bridge_request.side_effect = ConnectionRefusedError(
        "still not up"
    )
    tools._client.execute_vscode_only.return_value = ScriptResult(
        success=True, output="pong\n", transport="vscode"
    )
    tools.enable_isaac_extension.return_value = {
        "success": True,
        "enabled": True,
    }
    monkeypatch.setattr(isaac_cli, "_tools", lambda *args, **kwargs: tools)
    monkeypatch.setattr(
        isaac_cli.asyncio, "sleep", AsyncMock(return_value=None)
    )

    result = runner.invoke(app, ["--json", "isaac", "bridge-up"])

    assert result.exit_code != 0
    payload = json.loads(result.stdout)
    assert payload["action"] == "auto-enabled"
    assert payload["bridge_reachable"] is False
    assert payload["success"] is False
    assert payload["extension_enabled"] is True
