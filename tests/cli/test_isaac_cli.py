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
