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
