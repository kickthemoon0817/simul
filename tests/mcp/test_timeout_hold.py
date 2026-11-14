"""After an Isaac call times out, the instance stays held until the engine has finished it.

Before (round-2 audit M-2): a call to the stock socket that timed out released
the instance at once. Kit runs scripts one at a time on its main thread and
goes on running a script after its reply times out, so the next call was sent
to wait behind it: it timed out too, then ran after being reported failed
(``storm.py cascade``: a marker script reported ``TimeoutError`` at 10 s and
wrote its file at 12 s). The cancel path already held the instance (#341).

Now the client records a delivered request whose reply timed out
(``unfinished_request``), the server hands the instance lock to a task that
waits for the engine (``wait_until_idle``, a probe that queues behind the
request) and answers other calls ``InstanceBusy`` meanwhile, bounded by
``isaac_sim.hold_after_timeout``. A stopping server gives up on Isaac Sim
after ``server.shutdown_timeout``.
"""

from __future__ import annotations

import asyncio
import json
import re
import struct
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from simul.adapters.isaac_socket_client import (
    IDLE_PROBE_SCRIPT,
    IsaacSocketClient,
    SocketReplyTimeoutError,
    UnfinishedRequest,
)
from simul.config import Settings
from simul.mcp import backends as backends_module
from simul.mcp import server as server_module
from simul.mcp.usage_tracker import ToolUsageTracker
from tests.fakes import FakeFastMCP


class _SerialKit:
    """Kit's 5.x stock socket in miniature: scripts run one at a time, as on its main thread.

    ``sleep(N)`` in a script keeps the main thread N seconds; ``print('x')``
    answers ``x``. Every script that ran is logged with when it started and ended.
    """

    def __init__(self) -> None:
        self.ran: List[Tuple[str, float, float]] = []
        self.port = 0
        self._main_thread = asyncio.Lock()
        self._server: Optional[asyncio.AbstractServer] = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        assert self._server is not None
        self._server.close()
        await self._server.wait_closed()

    def scripts(self) -> List[str]:
        return [code for code, _start, _end in self.ran]

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        code = (await reader.read(65536)).decode()
        async with self._main_thread:
            started = time.monotonic()
            pause = re.search(r"sleep\(([\d.]+)\)", code)
            if pause:
                await asyncio.sleep(float(pause.group(1)))
            self.ran.append((code, started, time.monotonic()))
        printed = re.search(r"print\('([^']*)'\)", code)
        reply = {"status": "ok", "output": (printed.group(1) + "\n") if printed else ""}
        try:
            writer.write(json.dumps(reply).encode())
            await writer.drain()
            writer.close()
        except (ConnectionError, OSError):
            pass  # the client gave up on this reply


def _client(port: int, timeout: float = 0.5) -> IsaacSocketClient:
    return IsaacSocketClient(port=port, timeout_seconds=timeout, socket_protocol="vscode")


# ---------------------------------------------------------------------------
# The client: which requests it tracks, and how it learns the engine is done
# ---------------------------------------------------------------------------


def test_a_script_whose_reply_timed_out_is_tracked_until_the_engine_finishes() -> None:
    async def scenario() -> Tuple[_SerialKit, IsaacSocketClient, float, float, bool]:
        kit = _SerialKit()
        await kit.start()
        client = _client(kit.port)
        try:
            started = time.monotonic()
            with pytest.raises(SocketReplyTimeoutError):
                # A short read deadline; execute() never waits less than 2 s.
                await client._execute_vscode_unlocked("import time; time.sleep(1.2)", read_timeout=0.3)
            tracked = client.unfinished_request
            assert isinstance(tracked, UnfinishedRequest)
            assert (tracked.transport, tracked.action) == ("socket", "script")
            idle = await client.wait_until_idle(5.0)
            return kit, client, started, time.monotonic(), idle
        finally:
            await kit.stop()

    kit, client, started, idle_at, idle = asyncio.run(scenario())

    assert idle and client.unfinished_request is None
    (slow, slow_start, slow_end), (probe, probe_start, _end) = kit.ran
    assert probe == IDLE_PROBE_SCRIPT
    assert probe_start >= slow_end, "the probe ran only after the slow script"
    assert idle_at - started >= 1.1


def test_a_ping_or_a_quick_script_leaves_nothing_tracked() -> None:
    async def scenario() -> Tuple[bool, Optional[UnfinishedRequest], Optional[UnfinishedRequest]]:
        kit = _SerialKit()
        await kit.start()
        client = _client(kit.port, timeout=0.3)
        try:
            await client.execute("print('quick')")
            after_quick = client.unfinished_request
            blocker = asyncio.ensure_future(_client(kit.port, timeout=5).execute("sleep(2.3)"))
            await asyncio.sleep(0.05)
            reachable = await client.ping()  # queued behind the blocker, times out after 2 s
            after_ping = client.unfinished_request
            await blocker
            return reachable, after_quick, after_ping
        finally:
            await kit.stop()

    reachable, after_quick, after_ping = asyncio.run(scenario())

    assert reachable is False
    assert after_quick is None and after_ping is None


def test_waiting_for_the_engine_is_bounded_and_an_engine_gone_counts_as_idle() -> None:
    async def scenario() -> Tuple[bool, bool, float]:
        kit = _SerialKit()
        await kit.start()
        client = _client(kit.port)
        with pytest.raises(TimeoutError):
            await client._execute_vscode_unlocked("sleep(30)", read_timeout=0.2)
        started = time.monotonic()
        bounded = await client.wait_until_idle(0.4)
        waited = time.monotonic() - started
        assert client.unfinished_request is not None, "still tracked after giving up"
        await kit.stop()
        gone = await client.wait_until_idle(2.0)
        return bounded, gone, waited

    bounded, gone, waited = asyncio.run(scenario())

    assert bounded is False and 0.35 <= waited < 1.5
    assert gone is True


def test_python_server_s_timeout_reply_is_tracked() -> None:
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        envelope = json.loads((await reader.read()).decode())
        reply = (
            {"status": "ok", "output": "simul-idle\n"}
            if "simul-idle" in envelope.get("code", "")
            else {"status": "error", "ename": "TimeoutError", "evalue": "Execution timed out"}
        )
        writer.write(json.dumps(reply).encode())
        await writer.drain()
        writer.close()

    async def scenario() -> Tuple[Any, Optional[UnfinishedRequest], bool]:
        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        client = IsaacSocketClient(port=port, timeout_seconds=2.0, socket_protocol="python_server")
        try:
            result = await client.execute("busy()")
            tracked = client.unfinished_request
            return result, tracked, await client.wait_until_idle(2.0)
        finally:
            server.close()
            await server.wait_closed()

    result, tracked, idle = asyncio.run(scenario())

    assert result.error_name == "TimeoutError"
    assert tracked is not None and tracked.transport == "socket"
    assert idle is True


class _Bridge:
    """A bridge that never answers ``execute_script`` and reports busy for a few polls."""

    def __init__(self, busy_polls: int, request_timeout_reply: bool = False) -> None:
        self.busy_polls = busy_polls
        self.request_timeout_reply = request_timeout_reply
        self.actions: List[str] = []
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
        (size,) = struct.unpack(">I", await reader.readexactly(4))
        request = json.loads(await reader.readexactly(size))
        action = request["action"]
        self.actions.append(action)
        if action == "get_runtime_info":
            busy = self.busy_polls > 0
            self.busy_polls -= 1
            response: Dict[str, Any] = {
                "request_id": request["request_id"],
                "status": "ok",
                "payload": {"bridge": {"busy": busy, "current_action": "execute_script" if busy else None}},
            }
        elif self.request_timeout_reply:
            response = {
                "request_id": request["request_id"],
                "status": "error",
                "error": {"name": "RequestTimeout", "message": "Handler exceeded 1s."},
            }
        else:
            await asyncio.sleep(10)  # the reply never comes in time
            return
        body = json.dumps(response).encode()
        writer.write(struct.pack(">I", len(body)) + body)
        await writer.drain()
        writer.close()


@pytest.mark.parametrize("request_timeout_reply", [False, True])
def test_a_bridge_request_that_timed_out_is_tracked_until_the_bridge_is_idle(
    request_timeout_reply: bool,
) -> None:
    async def scenario() -> Tuple[_Bridge, Optional[UnfinishedRequest], bool]:
        bridge = _Bridge(busy_polls=2, request_timeout_reply=request_timeout_reply)
        await bridge.start()
        client = IsaacSocketClient(
            port=1,
            timeout_seconds=0.3,
            bridge_host="127.0.0.1",
            bridge_port=bridge.port,
            prefer_bridge=True,
            fallback_to_vscode=False,
        )
        try:
            try:
                await client.bridge_request("simulation_control", {"action": "step"}, read_timeout=0.3)
            except TimeoutError:
                pass
            tracked = client.unfinished_request
            return bridge, tracked, await client.wait_until_idle(5.0)
        finally:
            await bridge.stop()

    bridge, tracked, idle = asyncio.run(scenario())

    assert tracked is not None and (tracked.transport, tracked.action) == ("bridge", "simulation_control")
    assert idle is True
    assert bridge.actions.count("get_runtime_info") == 3, "polled until no longer busy"


def test_a_bridge_ping_or_interrupt_that_timed_out_leaves_nothing_tracked() -> None:
    async def scenario() -> Optional[UnfinishedRequest]:
        bridge = _Bridge(busy_polls=0)
        await bridge.start()
        client = IsaacSocketClient(
            port=1,
            timeout_seconds=0.2,
            bridge_host="127.0.0.1",
            bridge_port=bridge.port,
            prefer_bridge=True,
            fallback_to_vscode=False,
        )
        try:
            for action in ("ping", "interrupt"):
                with pytest.raises(TimeoutError):
                    await client._bridge_request(action, {}, ignore_circuit=True)
            return client.unfinished_request
        finally:
            await bridge.stop()

    assert asyncio.run(scenario()) is None


# ---------------------------------------------------------------------------
# The server: the hold, the refusal meanwhile, and shutdown
# ---------------------------------------------------------------------------


def _server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **isaac_sim: Any
) -> server_module.SimulMCPServer:
    monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
    monkeypatch.setattr(server_module, "TaskConfig", None)
    monkeypatch.setattr(backends_module, "is_headless_available", lambda: False)
    monkeypatch.setattr(backends_module, "is_blender_available", lambda: False)
    monkeypatch.setattr(backends_module, "UnrealRuntimeAdapter", None)
    settings = Settings(
        server={"timeout": 1, "shutdown_timeout": 0.5},
        isaac_sim={"bridge_enabled": False, "socket_protocol": "vscode", **isaac_sim},
    )
    instance = server_module.SimulMCPServer(settings=settings)
    instance.usage_tracker = ToolUsageTracker(log_dir=tmp_path)
    return instance


def _payload(result: Any) -> Dict[str, Any]:
    return json.loads(result.content[-1].text)


def test_a_timed_out_script_holds_the_instance_until_the_engine_is_done(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """storm.py's cascade, against the real tools and client: the marker never reaches the engine."""

    async def scenario() -> Tuple[_SerialKit, List[Dict[str, Any]], List[float], Dict[str, Any]]:
        kit = _SerialKit()
        await kit.start()
        instance = _server(monkeypatch, tmp_path, socket_port=kit.port, socket_timeout=1.0)
        tools = instance._isaac_tools
        t0 = time.monotonic()
        answers: List[Dict[str, Any]] = []
        times: List[float] = []
        try:
            for code in ("import time; time.sleep(3.5)", "print('marker')"):
                result = await instance._exec_isaac("execute_isaac_script", tools.execute_script(code=code))
                answers.append(_payload(result))
                times.append(time.monotonic() - t0)
            ping = _payload(await instance._exec_isaac("ping_isaac", _held(instance), bypass_instance_lock=True))
            while instance._isaac_holds:
                await asyncio.sleep(0.05)
            result = await instance._exec_isaac("execute_isaac_script", tools.execute_script(code="print('third')"))
            answers.append(_payload(result))
            times.append(time.monotonic() - t0)
            return kit, answers, times, ping
        finally:
            await instance.shutdown()
            await kit.stop()

    async def _held(instance: server_module.SimulMCPServer) -> Dict[str, Any]:
        return {"success": True, "held": instance.isaac_hold_info("default")}

    kit, answers, times, ping = asyncio.run(scenario())

    timed_out, busy, third = answers
    assert timed_out["error_type"] == "TimeoutError"
    assert busy["error_type"] == "InstanceBusy"
    assert busy["details"]["reason"] == "finishing_timed_out_call"
    assert busy["details"]["timed_out_tool"] == "execute_isaac_script"
    assert "interrupt_isaac_script" in busy["error"]
    assert ping["held"]["timed_out_tool"] == "execute_isaac_script"
    assert third["success"] is True and third["output"].strip() == "third"
    scripts = kit.scripts()
    assert not any("marker" in code for code in scripts), "the refused call never reached the engine"
    assert scripts[1] == IDLE_PROBE_SCRIPT
    assert kit.ran[1][1] >= kit.ran[0][2], "the probe waited for the slow script"
    assert times[1] < 3.5, "the refusal came within the lock wait, before the engine finished"


class _FakeClient:
    """An Isaac client whose engine stays busy after a timeout until told otherwise."""

    bridge_enabled = False

    def __init__(self) -> None:
        self.unfinished_request: Optional[UnfinishedRequest] = None
        self.idle = asyncio.Event()

    def time_out(self) -> Dict[str, Any]:
        self.unfinished_request = UnfinishedRequest("socket", "script", time.monotonic())
        return {"success": False, "error": "timed out", "error_type": "TimeoutError"}

    async def wait_until_idle(self, timeout: float) -> bool:
        try:
            await asyncio.wait_for(self.idle.wait(), timeout)
        except asyncio.TimeoutError:
            return False
        self.unfinished_request = None
        return True

    def forget_unfinished_request(self) -> None:
        self.unfinished_request = None


def _with_fake_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **isaac_sim: Any
) -> Tuple[server_module.SimulMCPServer, _FakeClient]:
    instance = _server(monkeypatch, tmp_path, **isaac_sim)
    fake = _FakeClient()
    monkeypatch.setattr(instance, "_isaac_clients", {"default": fake})
    monkeypatch.setattr(instance, "_get_request_isaac_client", lambda: fake)
    return instance, fake


def test_the_hold_ends_when_the_engine_is_idle_and_lets_the_next_call_in(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    instance, fake = _with_fake_client(monkeypatch, tmp_path)
    lock = instance._get_instance_lock("default")

    async def scenario() -> List[str]:
        events: List[str] = []

        async def times_out() -> Dict[str, Any]:
            return fake.time_out()

        async def follower() -> Dict[str, Any]:
            events.append("follower ran")
            return {"success": True}

        await instance._exec_isaac("isaac_slow", times_out())
        assert lock.locked() and "default" in instance._isaac_holds
        waiting = asyncio.ensure_future(instance._exec_isaac("isaac_next", follower()))
        await asyncio.sleep(0.2)
        events.append("engine idle")
        fake.idle.set()
        await waiting
        return events

    assert asyncio.run(scenario()) == ["engine idle", "follower ran"]
    assert not lock.locked() and instance._isaac_holds == {}


def test_hold_after_timeout_zero_releases_at_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    instance, fake = _with_fake_client(monkeypatch, tmp_path, hold_after_timeout=0)

    async def times_out() -> Dict[str, Any]:
        return fake.time_out()

    asyncio.run(instance._exec_isaac("isaac_slow", times_out()))

    assert not instance._get_instance_lock("default").locked()
    assert fake.unfinished_request is None


def test_a_hold_past_its_limit_is_released_with_a_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    instance, fake = _with_fake_client(monkeypatch, tmp_path, hold_after_timeout=0.3)

    async def scenario() -> None:
        async def times_out() -> Dict[str, Any]:
            return fake.time_out()

        await instance._exec_isaac("isaac_slow", times_out())
        await asyncio.wait_for(instance._isaac_holds["default"].task, 2.0)

    with caplog.at_level("WARNING"):
        asyncio.run(scenario())

    assert not instance._get_instance_lock("default").locked()
    assert fake.unfinished_request is None
    assert "isaac_sim.hold_after_timeout" in caplog.text


def test_shutdown_releases_a_held_instance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    instance, fake = _with_fake_client(monkeypatch, tmp_path)

    async def scenario() -> float:
        async def times_out() -> Dict[str, Any]:
            return fake.time_out()

        await instance._exec_isaac("isaac_slow", times_out())
        started = time.monotonic()
        await instance.shutdown()
        return time.monotonic() - started

    with caplog.at_level("WARNING"):
        took = asyncio.run(scenario())

    assert took < 1.0
    assert not instance._get_instance_lock("default").locked()
    assert "Released Isaac instance 'default' on shutdown" in caplog.text


def test_a_stopping_server_gives_up_on_a_cancelled_call_after_shutdown_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A cancelled call used to hold shutdown for as long as the transport's timeout (up to 110 s)."""
    instance, _fake = _with_fake_client(monkeypatch, tmp_path)

    async def scenario() -> float:
        started = asyncio.Event()

        async def engine() -> Dict[str, Any]:
            started.set()
            await asyncio.sleep(60)
            return {"success": True}

        task = asyncio.ensure_future(instance._exec_isaac("isaac_long", engine()))
        await started.wait()
        task.cancel()
        await asyncio.sleep(0.05)
        assert instance._draining_calls
        t0 = time.monotonic()
        instance.begin_shutdown()
        await instance.wait_for_draining_calls()
        with pytest.raises(asyncio.CancelledError):
            await task
        return time.monotonic() - t0

    with caplog.at_level("WARNING"):
        took = asyncio.run(scenario())

    assert 0.4 <= took < 2.0, took
    assert "stopped waiting for Isaac Sim to finish isaac_long" in caplog.text
    (row,) = instance.usage_tracker.get_recent(tool_name="isaac_long")
    assert row["cancelled"] is True and row["executed"] == "unknown"
