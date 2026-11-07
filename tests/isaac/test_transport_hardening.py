"""Client-side transport hardening: framing, breaker, refusals, budgets, auth.

Each test reproduces a defect against Isaac Sim's own transport behaviour, read
from its source because no Isaac Sim runs on the hosts this suite targets:

* 5.1's ``isaacsim.code_editor.vscode`` runs every chunk ``data_received``
  hands it as a separate script and closes after the first reply, so a
  script that arrives in two reads executes as two fragments.
* A reply timeout after the bridge holds the request is not a connection
  failure; counting it opened the breaker and sent new scripts to the stock
  socket while the bridge still ran the old one.
* A bridge with ``allow_unsafe_execution=false`` refuses a script without
  running it, so the client may run it on the stock socket instead.
* 6.0.1's python_server timeout only queues a reply; the script runs on.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

from simul.adapters.isaac_socket_client import (
    MAX_VSCODE_SCRIPT_BYTES,
    TIMEOUT_MAY_STILL_COMPLETE,
    BridgeResponseConnectionError,
    BridgeResponseTimeoutError,
    IsaacSocketClient,
    ScriptResult,
)

# ---------------------------------------------------------------------------
# F21: 5.x VS Code socket executes each read separately
# ---------------------------------------------------------------------------


class _ChunkExecutingSocket:
    """Stand-in for Isaac Sim 5.1's VS Code socket on a link that splits writes.

    The real extension (``isaacsim/code_editor/vscode/extension.py``,
    ``ServerProtocol.data_received``) schedules ``_process_code`` for every
    chunk it receives, replies to whichever finishes first and closes the
    transport. ``read_size`` models how much one read delivers.
    """

    def __init__(self, read_size: int) -> None:
        self.read_size = read_size
        self.fragments: List[str] = []
        self.port = 0
        self._server: Optional[asyncio.AbstractServer] = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        assert self._server is not None
        self._server.close()
        await self._server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        replied = False
        while True:
            try:
                chunk = await asyncio.wait_for(reader.read(self.read_size), timeout=0.2)
            except asyncio.TimeoutError:
                break
            if not chunk:
                break
            # Each read is "executed" as a script of its own.
            self.fragments.append(chunk.decode("utf-8", errors="replace"))
            if not replied:
                writer.write(json.dumps({"status": "ok", "output": ""}).encode())
                await writer.drain()
                replied = True
        writer.close()


def _script_of(size: int) -> str:
    line = "x = 1  # padding padding padding padding padding padding padding\n"
    return (line * (size // len(line) + 1))[:size]


def _send_to_vscode_socket(script: str, read_size: int, **client_kwargs: Any) -> tuple[_ChunkExecutingSocket, Any]:
    async def scenario() -> tuple[_ChunkExecutingSocket, Any]:
        server = _ChunkExecutingSocket(read_size)
        await server.start()
        try:
            client = IsaacSocketClient(
                host="127.0.0.1",
                port=server.port,
                timeout_seconds=5.0,
                socket_protocol="vscode",
                **client_kwargs,
            )
            outcome: Any = await client.execute_vscode_only(script)
        finally:
            await server.stop()
        return server, outcome

    return asyncio.run(scenario())


def test_a_script_past_one_read_is_refused_before_anything_reaches_5x() -> None:
    """Sent whole, it would run as fragments; refusing it runs nothing."""
    script = _script_of(MAX_VSCODE_SCRIPT_BYTES + 1)

    server, result = _send_to_vscode_socket(script, read_size=4096)

    assert server.fragments == []  # nothing executed, not even a prefix
    assert isinstance(result, ScriptResult)
    assert result.success is False
    assert result.error_name == "ScriptTooLargeForSocket"
    assert "bridge" in result.error_value
    assert str(MAX_VSCODE_SCRIPT_BYTES) in result.error_value


def test_a_script_within_one_read_still_reaches_5x_whole() -> None:
    script = _script_of(2000)

    server, result = _send_to_vscode_socket(script, read_size=MAX_VSCODE_SCRIPT_BYTES)

    assert result.success is True
    assert server.fragments == [script]


def test_the_limit_counts_the_token_header_and_utf8_bytes() -> None:
    # 3 bytes per character in UTF-8: under the limit in characters, over it in bytes.
    script = "#" + "€" * (MAX_VSCODE_SCRIPT_BYTES // 3)

    server, result = _send_to_vscode_socket(script, read_size=4096, auth_token="tok")

    assert server.fragments == []
    assert result.error_name == "ScriptTooLargeForSocket"


def test_python_server_buffers_until_eof_and_is_not_limited() -> None:
    """6.0 reassembles the payload before executing, so a big script is fine there."""
    client = IsaacSocketClient(socket_protocol="python_server", timeout_seconds=5.0)
    client._socket_round_trip = AsyncMock(  # type: ignore[attr-defined]
        return_value=json.dumps({"status": "ok", "output": "done"})
    )

    result = asyncio.run(client.execute_vscode_only(_script_of(MAX_VSCODE_SCRIPT_BYTES * 3)))

    assert result.success is True
    assert client._socket_round_trip.await_args.kwargs["half_close"] is True  # type: ignore[attr-defined]


def test_the_vscode_flavour_never_half_closes() -> None:
    """CLAUDE.md: never send EOF unconditionally; 5.x closes on it."""
    client = IsaacSocketClient(socket_protocol="vscode", timeout_seconds=5.0)
    client._socket_round_trip = AsyncMock(  # type: ignore[attr-defined]
        return_value=json.dumps({"status": "ok", "output": ""})
    )

    asyncio.run(client.execute_vscode_only("print('hi')"))

    assert client._socket_round_trip.await_args.kwargs["half_close"] is False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# F40: a delivered request that is slow is not a connection failure
# ---------------------------------------------------------------------------


def _breaker_client(**kwargs: Any) -> IsaacSocketClient:
    options: Dict[str, Any] = {
        "bridge_host": "127.0.0.1",
        "bridge_port": 8229,
        "prefer_bridge": True,
        "fallback_to_vscode": True,
        "bridge_failure_threshold": 3,
        "bridge_cooldown_seconds": 30.0,
    }
    options.update(kwargs)
    return IsaacSocketClient(**options)


def test_slow_delivered_scripts_do_not_open_the_circuit() -> None:
    client = _breaker_client()
    client._dial_bridge = AsyncMock(  # type: ignore[attr-defined]
        side_effect=BridgeResponseTimeoutError("no reply yet; still running")
    )
    client._execute_vscode_unlocked = AsyncMock()  # type: ignore[attr-defined]

    for _ in range(5):
        with pytest.raises(BridgeResponseTimeoutError):
            asyncio.run(client.execute("long_job()"))

    assert client.bridge_circuit_open is False
    assert client.bridge_consecutive_failures == 0
    # Every call still went to the bridge, behind the script it is running.
    assert client._dial_bridge.await_count == 5  # type: ignore[attr-defined]
    client._execute_vscode_unlocked.assert_not_called()  # type: ignore[attr-defined]


def test_a_peer_that_drops_the_connection_unanswered_still_counts() -> None:
    """A port proxy with nothing behind it closes without replying; keep skipping it."""
    client = _breaker_client()
    client._dial_bridge = AsyncMock(  # type: ignore[attr-defined]
        side_effect=BridgeResponseConnectionError("closed before sending a response")
    )

    for _ in range(3):
        with pytest.raises(BridgeResponseConnectionError):
            asyncio.run(client.execute("job()"))

    assert client.bridge_circuit_open is True


# ---------------------------------------------------------------------------
# F36: a bridge refusal ran nothing, so the stock socket may run the script
# ---------------------------------------------------------------------------


_REFUSAL = {
    "status": "error",
    "error": {
        "name": "UnsafeExecutionDisabled",
        "message": "execute_script is disabled in bridge settings.",
    },
}


def test_refused_script_runs_on_the_stock_socket_instead() -> None:
    client = _breaker_client()
    client._bridge_request = AsyncMock(return_value=_REFUSAL)  # type: ignore[attr-defined]
    client._execute_vscode_unlocked = AsyncMock(  # type: ignore[attr-defined]
        return_value=ScriptResult(success=True, output='{"ok": true}', transport="vscode")
    )

    result = asyncio.run(client.execute("print('x')"))

    assert result.success is True and result.transport == "vscode"
    client._execute_vscode_unlocked.assert_awaited_once()  # type: ignore[attr-defined]


def test_refusal_is_reported_when_fallback_is_off() -> None:
    client = _breaker_client(fallback_to_vscode=False)
    client._bridge_request = AsyncMock(return_value=_REFUSAL)  # type: ignore[attr-defined]
    client._execute_vscode_unlocked = AsyncMock()  # type: ignore[attr-defined]

    result = asyncio.run(client.execute("print('x')"))

    assert result.success is False
    assert result.error_name == "UnsafeExecutionDisabled"
    client._execute_vscode_unlocked.assert_not_called()  # type: ignore[attr-defined]


def test_bridge_only_execution_never_falls_back() -> None:
    client = _breaker_client()
    client._bridge_request = AsyncMock(return_value=_REFUSAL)  # type: ignore[attr-defined]
    client._execute_vscode_unlocked = AsyncMock()  # type: ignore[attr-defined]

    result = asyncio.run(client.execute_bridge_script_only("print('x')"))

    assert result.error_name == "UnsafeExecutionDisabled"
    client._execute_vscode_unlocked.assert_not_called()  # type: ignore[attr-defined]


def test_a_script_that_ran_and_failed_is_not_rerun() -> None:
    client = _breaker_client()
    client._bridge_request = AsyncMock(  # type: ignore[attr-defined]
        return_value={"status": "error", "error": {"name": "ValueError", "message": "boom"}}
    )
    client._execute_vscode_unlocked = AsyncMock()  # type: ignore[attr-defined]

    result = asyncio.run(client.execute("raise ValueError('boom')"))

    assert result.error_name == "ValueError"
    client._execute_vscode_unlocked.assert_not_called()  # type: ignore[attr-defined]


def test_bridge_compat_path_names_its_refusal() -> None:
    """The raw-code path's refusal carries the same name, so a 5.x-style client can tell."""
    from khemoo.simul.extension import IsaacMCPServerExtension

    extension = IsaacMCPServerExtension()
    extension._executor = object()  # type: ignore[assignment]
    extension._allow_unsafe_execution = False

    reply = asyncio.run(extension._handle_vscode_script("print('x')"))

    assert reply["status"] == "error"
    assert reply["ename"] == "UnsafeExecutionDisabled"


# ---------------------------------------------------------------------------
# F20: per-call budgets, and timeouts that say the work may still complete
# ---------------------------------------------------------------------------


def test_a_per_call_budget_reaches_the_bridge_and_its_read_deadline() -> None:
    client = _breaker_client(timeout_seconds=30.0)
    client._bridge_request = AsyncMock(  # type: ignore[attr-defined]
        return_value={"status": "ok", "payload": {"output": "{}"}}
    )

    asyncio.run(client.execute("open()", timeout_seconds=110.0))

    client._bridge_request.assert_awaited_once_with(  # type: ignore[attr-defined]
        "execute_script", {"code": "open()", "timeout": 110.0}, read_timeout=115.0
    )


def test_a_per_call_budget_reaches_the_python_server_envelope() -> None:
    client = IsaacSocketClient(socket_protocol="python_server", timeout_seconds=30.0)
    client._socket_round_trip = AsyncMock(  # type: ignore[attr-defined]
        return_value=json.dumps({"status": "ok", "output": ""})
    )

    asyncio.run(client.execute_vscode_only("open()", timeout_seconds=110.0))

    sent = json.loads(client._socket_round_trip.await_args.args[0])  # type: ignore[attr-defined]
    assert sent["timeout"] == 110.0
    assert client._socket_round_trip.await_args.kwargs["read_timeout"] == 111.0  # type: ignore[attr-defined]


def test_python_server_timeout_says_the_script_keeps_running() -> None:
    """v6.0.1's ``_timeout_watchdog`` only sends this reply; the code runs on."""
    client = IsaacSocketClient(socket_protocol="python_server", timeout_seconds=5.0)
    client._socket_round_trip = AsyncMock(  # type: ignore[attr-defined]
        return_value=json.dumps(
            {
                "status": "error",
                "output": "",
                "ename": "TimeoutError",
                "evalue": "Execution timed out after 4.0s",
                "traceback": [],
            }
        )
    )

    result = asyncio.run(client.execute_vscode_only("slow()"))

    assert result.error_name == "TimeoutError"
    assert "keeps running" in result.error_value
    assert TIMEOUT_MAY_STILL_COMPLETE in result.error_value


def test_bridge_timeout_says_kit_work_may_still_complete() -> None:
    client = _breaker_client()
    client._bridge_request = AsyncMock(  # type: ignore[attr-defined]
        return_value={
            "status": "error",
            "error": {"name": "ScriptInterrupted", "message": "timed out after 29s"},
        }
    )

    result = asyncio.run(client.execute("await open_stage()"))

    assert result.error_name == "ScriptInterrupted"
    assert result.error_value.startswith("Script timed out after 29s")
    assert TIMEOUT_MAY_STILL_COMPLETE in result.error_value


def test_an_interrupt_by_request_is_not_dressed_as_a_timeout() -> None:
    client = _breaker_client()
    client._bridge_request = AsyncMock(  # type: ignore[attr-defined]
        return_value={
            "status": "error",
            "error": {"name": "ScriptInterrupted", "message": "interrupted by request"},
        }
    )

    result = asyncio.run(client.execute("loop()"))

    assert result.error_value == "interrupted by request"


def _ok_frame_with_a_list_payload(request: Dict[str, Any]) -> bytes:
    body = json.dumps({"request_id": request["request_id"], "status": "ok", "payload": [1]})
    return len(body).to_bytes(4, "big") + body.encode()


@pytest.mark.parametrize(
    ("bridge", "reply"),
    [
        (False, b"\xff\xfe not utf-8"),
        (True, (9).to_bytes(4, "big") + b"\xffgarbage!"),
        (True, _ok_frame_with_a_list_payload),
    ],
    ids=["stock-socket-bytes", "bridge-bytes", "bridge-payload-not-an-object"],
)
def test_ping_reads_a_garbage_reply_as_unreachable(bridge: bool, reply: Any) -> None:
    """Something else on the port: ping answers False instead of raising."""

    async def scenario() -> bool:
        async def _garbage(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            if callable(reply):
                size = int.from_bytes(await reader.readexactly(4), "big")
                writer.write(reply(json.loads(await reader.readexactly(size))))
            else:
                writer.write(reply)
            await writer.drain()
            writer.close()

        server = await asyncio.start_server(_garbage, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        options: Dict[str, Any] = {"timeout_seconds": 2.0, "socket_protocol": "vscode"}
        if bridge:
            options.update(prefer_bridge=True, bridge_port=port, fallback_to_vscode=False)
        else:
            options["port"] = port
        client = IsaacSocketClient(**options)
        try:
            return await client.ping()
        finally:
            server.close()
            await server.wait_closed()

    assert asyncio.run(scenario()) is False


def test_a_stock_socket_read_timeout_says_the_script_may_still_run() -> None:
    async def scenario() -> BaseException:
        async def _never_reply(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            await asyncio.sleep(5)

        server = await asyncio.start_server(_never_reply, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        client = IsaacSocketClient(port=port, timeout_seconds=0.3, socket_protocol="vscode")
        try:
            await client.execute_vscode_only("slow()")
        except TimeoutError as exc:
            return exc
        finally:
            server.close()
        raise AssertionError("expected a timeout")

    error = asyncio.run(scenario())

    assert "may still be running" in str(error)


# ---------------------------------------------------------------------------
# F17: the token reaches the bridge
# ---------------------------------------------------------------------------


def _framed_request(client: IsaacSocketClient) -> Dict[str, Any]:
    captured: Dict[str, Any] = {}

    async def scenario() -> None:
        async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            size = int.from_bytes(await reader.readexactly(4), "big")
            captured.update(json.loads(await reader.readexactly(size)))
            body = json.dumps(
                {"request_id": captured["request_id"], "status": "ok", "payload": {}}
            ).encode()
            writer.write(len(body).to_bytes(4, "big") + body)
            await writer.drain()
            writer.close()

        server = await asyncio.start_server(_handle, "127.0.0.1", 0)
        client._bridge_port = server.sockets[0].getsockname()[1]
        try:
            await client.bridge_request("ping", {})
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(scenario())
    return captured


def test_the_token_travels_beside_the_payload_in_every_bridge_frame() -> None:
    request = _framed_request(_breaker_client(auth_token="s3cret"))

    assert request["auth_token"] == "s3cret"
    assert "auth_token" not in request["payload"]


def test_no_token_field_without_a_token() -> None:
    request = _framed_request(_breaker_client())

    assert "auth_token" not in request
