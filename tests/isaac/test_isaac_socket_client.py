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


# ---------------------------------------------------------------------------
# Stock socket flavours: Isaac Sim 6.0 python_server vs 5.x VS Code extension
# ---------------------------------------------------------------------------
import json  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402


@dataclass
class _FakeSocketServer:
    """In-process stand-in for the stock Isaac Sim Python socket.

    ``python_server`` buffers until EOF and answers the introspection
    envelope; ``vscode`` executes on the first packet and never sees EOF
    before replying, as the 5.x extension does.
    """

    flavor: str
    requests: list[str] = field(default_factory=list)
    saw_eof_before_reply: list[bool] = field(default_factory=list)
    port: int = 0
    _server: asyncio.AbstractServer | None = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        assert self._server is not None
        self._server.close()
        await self._server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if self.flavor == "python_server":
            raw = (await reader.read()).decode()
            self.saw_eof_before_reply.append(True)
        else:
            raw = (await reader.read(65536)).decode()
            self.saw_eof_before_reply.append(reader.at_eof())
        self.requests.append(raw)
        writer.write(self._reply(raw).encode())
        await writer.drain()
        writer.close()

    def _reply(self, raw: str) -> str:
        if self.flavor == "python_server" and raw.lstrip().startswith("{"):
            envelope = json.loads(raw)
            if envelope.get("introspect") == "status":
                return json.dumps({"status": "ok", "result": {"uptime_seconds": 1.5}})
        # Both flavours evaluate plain source; a dict literal prints nothing.
        output = "pong" if "pong" in raw else ""
        return json.dumps({"status": "ok", "output": output})


def _run_against(flavor: str, **client_kwargs: object) -> tuple[_FakeSocketServer, ScriptResult, IsaacSocketClient]:
    async def scenario() -> tuple[_FakeSocketServer, ScriptResult, IsaacSocketClient]:
        server = _FakeSocketServer(flavor=flavor)
        await server.start()
        try:
            client = IsaacSocketClient(host="127.0.0.1", port=server.port, timeout_seconds=5.0, **client_kwargs)
            result = await client.execute_vscode_only("print('pong')")
        finally:
            await server.stop()
        return server, result, client

    return asyncio.run(scenario())


def test_auto_detects_python_server_and_half_closes() -> None:
    """Isaac Sim 6.0 executes only after EOF, so the client must send it."""
    server, result, client = _run_against("python_server")

    assert result.success and result.output == "pong"
    assert client.socket_protocol == "python_server"
    assert json.loads(server.requests[0]) == {"introspect": "status"}
    assert server.requests[1] == "print('pong')"
    assert server.saw_eof_before_reply == [True, True]


def test_auto_detects_vscode_and_keeps_socket_open() -> None:
    """The 5.x extension closes the transport on EOF, so the client must not send it."""
    server, result, client = _run_against("vscode")

    assert result.success and result.output == "pong"
    assert client.socket_protocol == "vscode"
    assert server.requests[1] == "print('pong')"
    assert server.saw_eof_before_reply[1] is False


def test_explicit_protocol_skips_the_probe() -> None:
    server, result, _ = _run_against("python_server", socket_protocol="python_server")

    assert result.success
    assert server.requests == ["print('pong')"]


def test_auth_token_is_sent_as_python_server_header() -> None:
    server, result, _ = _run_against("python_server", auth_token="s3cret")

    assert result.success
    assert json.loads(server.requests[0]) == {"introspect": "status", "auth_token": "s3cret"}
    assert server.requests[1] == "# isaacsim-python-server-token: s3cret\nprint('pong')"


def test_invalid_socket_protocol_is_rejected() -> None:
    import pytest

    with pytest.raises(ValueError):
        IsaacSocketClient(socket_protocol="telnet")  # type: ignore[arg-type]


def test_timeout_resets_detected_protocol() -> None:
    """Isaac restarted on the other major version times out, it does not refuse.

    6.0 buffers forever when no EOF arrives, so a stale "vscode" flavour hangs
    until the read deadline. Caching that would break every later call too.
    """

    async def scenario() -> IsaacSocketClient:
        client = IsaacSocketClient(host="127.0.0.1", port=1, timeout_seconds=5.0)
        client._detected_socket_protocol = "vscode"
        client._socket_round_trip = AsyncMock(  # type: ignore[attr-defined]
            side_effect=TimeoutError("read timed out")
        )
        try:
            await client.execute_vscode_only("print('pong')")
        except TimeoutError:
            pass
        return client

    client = asyncio.run(scenario())
    assert client.socket_protocol == "auto"


def test_empty_reply_resets_detected_protocol() -> None:
    """A 5.x server that receives EOF closes without replying; re-probe on that."""

    async def scenario() -> tuple[IsaacSocketClient, ScriptResult]:
        client = IsaacSocketClient(host="127.0.0.1", port=1, timeout_seconds=5.0)
        client._detected_socket_protocol = "python_server"
        client._socket_round_trip = AsyncMock(return_value="")  # type: ignore[attr-defined]
        result = await client.execute_vscode_only("print('pong')")
        return client, result

    client, result = asyncio.run(scenario())
    assert result.error_name == "EmptyResponse"
    assert client.socket_protocol == "auto"


def test_pinned_protocol_survives_a_failure() -> None:
    """An explicit protocol is the operator's choice; a failure must not undo it."""

    async def scenario() -> IsaacSocketClient:
        client = IsaacSocketClient(
            host="127.0.0.1", port=1, timeout_seconds=5.0, socket_protocol="python_server"
        )
        client._socket_round_trip = AsyncMock(  # type: ignore[attr-defined]
            side_effect=TimeoutError("read timed out")
        )
        try:
            await client.execute_vscode_only("print('pong')")
        except TimeoutError:
            pass
        return client

    client = asyncio.run(scenario())
    assert client.socket_protocol == "python_server"


def test_connection_refused_resets_detected_protocol() -> None:
    """An Isaac Sim restarted on another version must be re-probed."""

    async def scenario() -> IsaacSocketClient:
        server = _FakeSocketServer(flavor="python_server")
        await server.start()
        client = IsaacSocketClient(host="127.0.0.1", port=server.port, timeout_seconds=5.0)
        await client.execute_vscode_only("print('pong')")
        assert client.socket_protocol == "python_server"
        await server.stop()
        try:
            await client.execute_vscode_only("print('pong')")
        except ConnectionRefusedError:
            pass
        return client

    client = asyncio.run(scenario())
    assert client.socket_protocol == "auto"
