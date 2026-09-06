"""Unit tests for IsaacSocketClient bridge preference and fallback behavior."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock

import pytest


from simul_mcp.adapters.isaac_socket_client import (
    BridgeCircuitOpenError,
    IsaacSocketClient,
    ScriptResult,
)


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
    #: Parsed python_server envelopes, one per executed request.
    envelopes: list[dict[str, object]] = field(default_factory=list)
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
        code = raw
        if self.flavor == "python_server" and raw.lstrip().startswith("{"):
            envelope = json.loads(raw)
            if envelope.get("introspect") == "status":
                return json.dumps({"status": "ok", "result": {"uptime_seconds": 1.5}})
            # The real server executes envelope["code"] under envelope["timeout"].
            self.envelopes.append(envelope)
            code = str(envelope.get("code", ""))
        # Both flavours evaluate plain source; a dict literal prints nothing.
        output = "pong" if "pong" in code else ""
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
    # python_server gets the JSON envelope so it enforces the timeout itself;
    # the default budget is one second under the client's read deadline.
    assert server.envelopes == [{"code": "print('pong')", "timeout": 4.0}]
    assert client.script_timeout_seconds == 4.0
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
    assert len(server.requests) == 1
    assert server.envelopes[0]["code"] == "print('pong')"


def test_auth_token_is_sent_in_the_python_server_envelope() -> None:
    server, result, _ = _run_against("python_server", auth_token="s3cret")

    assert result.success
    assert json.loads(server.requests[0]) == {"introspect": "status", "auth_token": "s3cret"}
    assert server.envelopes == [
        {"code": "print('pong')", "timeout": 4.0, "auth_token": "s3cret"}
    ]


def test_auth_token_is_sent_as_a_header_on_the_vscode_flavour() -> None:
    """5.x has no envelope; the header is the only place the token can go."""
    server, result, _ = _run_against("vscode", auth_token="s3cret")

    assert result.success
    assert server.requests[1] == "# isaacsim-python-server-token: s3cret\nprint('pong')"


def test_script_timeout_is_floored_and_configurable() -> None:
    assert IsaacSocketClient(timeout_seconds=0.5).script_timeout_seconds == 1.0
    assert IsaacSocketClient(timeout_seconds=30.0).script_timeout_seconds == 29.0
    assert (
        IsaacSocketClient(timeout_seconds=30.0, script_timeout_seconds=7.0).script_timeout_seconds
        == 7.0
    )


def test_bridge_execute_script_carries_the_timeout() -> None:
    client = IsaacSocketClient(
        bridge_host="127.0.0.1", bridge_port=8229, prefer_bridge=True, timeout_seconds=10.0
    )
    client._bridge_request = AsyncMock(  # type: ignore[attr-defined]
        return_value={"status": "ok", "payload": {"output": "hi"}}
    )

    result = asyncio.run(client.execute("print('hi')"))

    assert result.success and result.transport == "bridge"
    client._bridge_request.assert_awaited_once_with(  # type: ignore[attr-defined]
        "execute_script", {"code": "print('hi')", "timeout": 9.0}
    )


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


def test_bridge_read_timeout_names_endpoint_and_timeout() -> None:
    """A bridge that accepts but never answers must not surface as an empty TimeoutError."""

    async def scenario() -> None:
        release = asyncio.Event()

        async def _hang(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            await reader.read()
            # Hold the connection open with no reply until the test releases it.
            # The server side must close its own writer: since Python 3.12
            # ``Server.wait_closed`` waits for every connection to finish, and a
            # half-closed peer alone never ends one.
            try:
                await release.wait()
            finally:
                writer.close()

        server = await asyncio.start_server(_hang, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        client = IsaacSocketClient(
            bridge_host="127.0.0.1",
            bridge_port=port,
            prefer_bridge=True,
            fallback_to_vscode=False,
            bridge_timeout_seconds=0.2,
        )
        try:
            with pytest.raises(TimeoutError) as excinfo:
                await client.bridge_request("ping", {})
        finally:
            release.set()
            server.close()
            await asyncio.wait_for(server.wait_closed(), timeout=5)
        message = str(excinfo.value)
        assert f"127.0.0.1:{port}" in message
        assert "0.2s" in message
        assert "'ping'" in message

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Circuit breaker: a hanging bridge must not tax every call forever
# ---------------------------------------------------------------------------


def _breaker_client(**kwargs: object) -> IsaacSocketClient:
    options: dict[str, object] = {
        "host": "127.0.0.1",
        "port": 8226,
        "bridge_host": "127.0.0.1",
        "bridge_port": 8229,
        "prefer_bridge": True,
        "fallback_to_vscode": True,
        "bridge_failure_threshold": 3,
        "bridge_cooldown_seconds": 30.0,
    }
    options.update(kwargs)
    return IsaacSocketClient(**options)  # type: ignore[arg-type]


def test_circuit_opens_after_threshold_failures_and_skips_the_bridge() -> None:
    client = _breaker_client()
    client._dial_bridge = AsyncMock(  # type: ignore[attr-defined]
        side_effect=TimeoutError("bridge silent")
    )
    client._execute_vscode_unlocked = AsyncMock(  # type: ignore[attr-defined]
        return_value=ScriptResult(success=True, output="legacy", transport="vscode")
    )

    async def scenario() -> list[str]:
        transports = []
        for _ in range(5):
            transports.append((await client.execute("print(1)")).transport)
        return transports

    transports = asyncio.run(scenario())

    assert transports == ["vscode"] * 5
    assert client.bridge_circuit_open is True
    assert client.bridge_consecutive_failures == 3
    # Three dials opened the circuit; the last two calls never touched the wire.
    assert client._dial_bridge.await_count == 3  # type: ignore[attr-defined]


def test_open_circuit_raises_a_connection_refused_subclass_without_fallback() -> None:
    client = _breaker_client(fallback_to_vscode=False)
    client._dial_bridge = AsyncMock(  # type: ignore[attr-defined]
        side_effect=ConnectionRefusedError("down")
    )

    async def scenario() -> Exception:
        for _ in range(3):
            with pytest.raises(ConnectionRefusedError):
                await client.bridge_request("ping", {})
        with pytest.raises(BridgeCircuitOpenError) as excinfo:
            await client.bridge_request("ping", {})
        return excinfo.value

    error = asyncio.run(scenario())
    assert "127.0.0.1:8229" in str(error)
    assert "3 consecutive failures" in str(error)
    assert isinstance(error, ConnectionRefusedError)


def test_circuit_closes_again_after_cooldown_and_a_successful_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _breaker_client()
    now = [1000.0]
    monkeypatch.setattr(
        "simul_mcp.adapters.isaac_socket_client.time.monotonic", lambda: now[0]
    )
    client._dial_bridge = AsyncMock(  # type: ignore[attr-defined]
        side_effect=[
            TimeoutError("a"),
            TimeoutError("b"),
            TimeoutError("c"),
            {"status": "ok", "payload": {"reachable": True}},
        ]
    )

    async def scenario() -> tuple[bool, bool, bool]:
        for _ in range(3):
            with pytest.raises(TimeoutError):
                await client.bridge_request("ping", {})
        opened = client.bridge_circuit_open
        now[0] += 31.0
        half_open = client.bridge_circuit_open
        await client.bridge_request("ping", {})
        return opened, half_open, client.bridge_circuit_open

    opened, half_open, closed = asyncio.run(scenario())
    assert opened is True
    assert half_open is False
    assert closed is False
    assert client.bridge_consecutive_failures == 0


def test_failed_probe_after_cooldown_reopens_the_circuit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _breaker_client()
    now = [1000.0]
    monkeypatch.setattr(
        "simul_mcp.adapters.isaac_socket_client.time.monotonic", lambda: now[0]
    )
    client._dial_bridge = AsyncMock(  # type: ignore[attr-defined]
        side_effect=TimeoutError("still silent")
    )

    async def scenario() -> int:
        for _ in range(3):
            with pytest.raises(TimeoutError):
                await client.bridge_request("ping", {})
        now[0] += 31.0
        with pytest.raises(TimeoutError):
            await client.bridge_request("ping", {})
        # Open again: no wire attempt until the next cooldown elapses.
        with pytest.raises(BridgeCircuitOpenError):
            await client.bridge_request("ping", {})
        return client._dial_bridge.await_count  # type: ignore[attr-defined]

    assert asyncio.run(scenario()) == 4
    assert client.bridge_circuit_open is True


def test_a_success_resets_the_failure_count_before_the_threshold() -> None:
    client = _breaker_client()
    client._dial_bridge = AsyncMock(  # type: ignore[attr-defined]
        side_effect=[
            TimeoutError("a"),
            TimeoutError("b"),
            {"status": "ok", "payload": {}},
            TimeoutError("c"),
        ]
    )

    async def scenario() -> None:
        for _ in range(2):
            with pytest.raises(TimeoutError):
                await client.bridge_request("ping", {})
        await client.bridge_request("ping", {})
        with pytest.raises(TimeoutError):
            await client.bridge_request("ping", {})

    asyncio.run(scenario())
    assert client.bridge_consecutive_failures == 1
    assert client.bridge_circuit_open is False


def test_protocol_errors_do_not_trip_the_breaker() -> None:
    """An oversized request is the caller's fault, not a sign the bridge is down."""
    client = _breaker_client(max_request_bytes=64)

    async def scenario() -> None:
        for _ in range(4):
            with pytest.raises(ValueError):
                await client.bridge_request("execute_script", {"code": "x" * 200})

    asyncio.run(scenario())
    assert client.bridge_consecutive_failures == 0
    assert client.bridge_circuit_open is False


def test_interrupt_bypasses_the_open_circuit_and_the_lock() -> None:
    client = _breaker_client()
    client._bridge_circuit_opened_at = time.monotonic()
    client._bridge_consecutive_failures = 3
    client._dial_bridge = AsyncMock(  # type: ignore[attr-defined]
        return_value={"status": "ok", "payload": {"interrupted": True}}
    )

    async def scenario() -> dict[str, object]:
        async with client._lock:
            # In real use the runaway execute() call holds this lock; the
            # interrupt must not queue behind it.
            return await asyncio.wait_for(client.interrupt_bridge_script(), timeout=1.0)

    response = asyncio.run(scenario())
    assert response["payload"] == {"interrupted": True}
    client._dial_bridge.assert_awaited_once_with("interrupt", {})  # type: ignore[attr-defined]


def test_breaker_settings_are_validated() -> None:
    with pytest.raises(ValueError):
        IsaacSocketClient(bridge_failure_threshold=0)
    with pytest.raises(ValueError):
        IsaacSocketClient(bridge_cooldown_seconds=-1.0)
