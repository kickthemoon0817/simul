"""Unit tests for IsaacSocketClient bridge preference and fallback behavior."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock

src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_path))

from simul_mcp.adapters.isaac_socket_client import IsaacSocketClient, ScriptResult


def test_execute_prefers_bridge_when_configured() -> None:
    """Bridge transport should be attempted first when enabled."""
    client = IsaacSocketClient(
        bridge_host="127.0.0.1",
        bridge_port=8229,
        prefer_bridge=True,
    )
    client._execute_bridge_script = AsyncMock(  # type: ignore[attr-defined]
        return_value=ScriptResult(success=True, output="ok", transport="bridge")
    )
    client._execute_vscode_unlocked = AsyncMock(  # type: ignore[attr-defined]
        return_value=ScriptResult(success=True, output="legacy", transport="vscode")
    )

    result = asyncio.run(client.execute("print('hi')"))

    assert result.transport == "bridge"
    client._execute_bridge_script.assert_awaited_once_with("print('hi')")  # type: ignore[attr-defined]
    client._execute_vscode_unlocked.assert_not_called()  # type: ignore[attr-defined]


def test_execute_falls_back_to_vscode_when_bridge_is_unavailable() -> None:
    """Bridge errors should fall back to the VS Code transport when enabled."""
    client = IsaacSocketClient(
        host="127.0.0.1",
        port=8226,
        bridge_host="127.0.0.1",
        bridge_port=8229,
        prefer_bridge=True,
        fallback_to_vscode=True,
    )
    client._execute_bridge_script = AsyncMock(  # type: ignore[attr-defined]
        side_effect=ConnectionRefusedError("bridge down")
    )
    client._execute_vscode_unlocked = AsyncMock(  # type: ignore[attr-defined]
        return_value=ScriptResult(success=True, output="legacy", transport="vscode")
    )

    result = asyncio.run(client.execute("print('hi')"))

    assert result.transport == "vscode"
    client._execute_bridge_script.assert_awaited_once_with("print('hi')")  # type: ignore[attr-defined]
    client._execute_vscode_unlocked.assert_awaited_once_with("print('hi')")  # type: ignore[attr-defined]


def test_ping_prefers_bridge_when_available() -> None:
    """Ping should succeed directly through the bridge when it is reachable."""
    client = IsaacSocketClient(
        host="127.0.0.1",
        port=8226,
        bridge_host="127.0.0.1",
        bridge_port=8229,
        prefer_bridge=True,
        fallback_to_vscode=True,
    )
    client._bridge_request = AsyncMock(  # type: ignore[attr-defined]
        return_value={"status": "ok", "payload": {"reachable": True}}
    )
    client._execute_vscode_unlocked = AsyncMock(  # type: ignore[attr-defined]
        return_value=ScriptResult(success=True, output="pong", transport="vscode")
    )

    reachable = asyncio.run(client.ping())

    assert reachable is True
    client._bridge_request.assert_awaited_once_with("ping", {})  # type: ignore[attr-defined]
    client._execute_vscode_unlocked.assert_not_called()  # type: ignore[attr-defined]
