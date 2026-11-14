"""After an Unreal request times out, the editor is held until it has finished it.

Before (round-3 audit R3-M4): Remote Control serves requests on the editor's
game thread, one at a time and in arrival order, and goes on running a
request after simul stopped waiting. With ``UNREAL__TIMEOUT=5``, a call sent
right after a 12 s script timed out timed out too and then wrote its marker
file about a second after it was reported failed; an agent retrying it ran it
twice.

Now the session records the timed-out request in holds shared by every
session of the adapter (``_unreal_hold``), probes the editor with one
``GET /remote/info`` that queues behind it, and refuses every other request
with ``InstanceBusy`` (nothing sent) until that probe is answered, bounded by
``unreal.hold_after_timeout``. ``ping_unreal`` and ``unreal_health_check`` are
still sent, report the hold, and end it when answered.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from simul.adapters import _unreal_hold
from simul.adapters.unreal_runtime import UnrealError, UnrealRuntimeAdapter, UnrealRuntimeSession, UnrealTimeoutError
from simul.config import Settings
from simul.mcp import backends as backends_module
from simul.mcp import server as server_module
from simul.mcp.usage_tracker import ToolUsageTracker
from simul.utils.engine_holds import SharedHolds
from tests.fakes import FakeFastMCP
from tests.tcp_fakes import FakeTcpServer


class _GameThreadEditor:
    """Remote Control in miniature: requests run one at a time, in arrival order, as on the game thread.

    ``sleep(N)`` in a Python command keeps the game thread N seconds. Every
    request that ran is logged with its route, its Python command, its
    headers, and when it started and ended.
    """

    def __init__(self) -> None:
        self.ran: List[Tuple[str, str, Dict[str, str], float, float]] = []
        self.port = 0
        self._game_thread = asyncio.Lock()
        self._server = FakeTcpServer(self._handle)

    async def start(self) -> None:
        self.port = await self._server.start()

    async def stop(self) -> None:
        await self._server.stop()

    def routes(self) -> List[str]:
        return [route for route, *_ in self.ran]

    def commands(self) -> List[str]:
        return [command for _route, command, *_ in self.ran]

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request_line = await reader.readline()
            headers: Dict[str, str] = {}
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break
                name, _, value = line.decode("latin-1").partition(":")
                headers[name.strip().lower()] = value.strip()
            raw = await reader.readexactly(int(headers.get("content-length", "0")))
        except (ConnectionError, asyncio.IncompleteReadError):
            return
        route = request_line.decode("latin-1").split(" ")[1] if request_line else ""
        body = json.loads(raw) if raw else {}
        command = _script(str((body.get("parameters") or {}).get("PythonCommand", "")))
        async with self._game_thread:
            started = time.monotonic()
            pause = re.search(r"sleep\(([\d.]+)\)", command)
            if pause:
                await asyncio.sleep(float(pause.group(1)))
            self.ran.append((route, command, headers, started, time.monotonic()))
        if route == "/remote/object/call":
            reply: Dict[str, Any] = {"ReturnValue": True, "LogOutput": [{"Type": "Info", "Output": "ran"}]}
        elif route == "/remote/object/describe":
            reply = {"Name": "Cube", "Class": "/Script/Engine.StaticMeshActor", "Properties": [], "Functions": []}
        else:
            reply = {"HttpRoutes": []}
        data = json.dumps(reply).encode()
        try:
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n"
                + f"Content-Length: {len(data)}\r\n\r\n".encode()
                + data
            )
            await writer.drain()
            writer.close()
        except (ConnectionError, OSError):
            pass  # the client gave up on this answer


def _script(command: str) -> str:
    """The source a runner command carries (``_unreal_python.runner_command`` encodes it)."""
    found = re.search(r"\.run\('\w+', '([A-Za-z0-9+/=]*)'", command)
    return base64.b64decode(found.group(1)).decode() if found else command


def _settings(port: int, **unreal: Any) -> Settings:
    base = Settings()
    return base.model_copy(
        update={
            # model_copy skips validation, so a timeout below the 1 s minimum
            # keeps these tests short.
            "unreal": base.unreal.model_copy(
                update={"host": "127.0.0.1", "port": port, "timeout": 0.3, "ping_timeout": 0.2,
                        "retry_base_delay": 0.001, "max_retries": 0, **unreal}
            ),
            "security": base.security.model_copy(update={"allow_script_execution": True}),
        }
    )


async def _released(holds: _unreal_hold.EditorHolds, endpoint: str, within: float = 5.0) -> None:
    deadline = time.monotonic() + within
    while holds.get(endpoint) is not None:
        assert time.monotonic() < deadline, "the hold did not end"
        await asyncio.sleep(0.02)


# ---------------------------------------------------------------------------
# The session: what starts a hold, what it refuses, how it ends
# ---------------------------------------------------------------------------


def test_a_request_after_a_timeout_is_refused_unsent_until_the_editor_is_done() -> None:
    """p36.py's cascade on the session layer: the marker never reaches the editor."""

    async def scenario() -> Tuple[_GameThreadEditor, Dict[str, Any], UnrealError, Dict[str, Any], float]:
        editor = _GameThreadEditor()
        await editor.start()
        adapter = UnrealRuntimeAdapter(_settings(editor.port))
        endpoint = f"127.0.0.1:{editor.port}"
        try:
            with adapter.create_session() as first:
                with pytest.raises(UnrealTimeoutError) as timed_out:
                    await first.execute_script("import time; time.sleep(1.0)")
                await first.close()
            # The next call opens a session of its own, as the MCP server does.
            with adapter.create_session() as second:
                with pytest.raises(UnrealError) as refused:
                    await second.execute_script("marker_b = 1")
                await second.close()
            await _released(adapter.holds, endpoint)
            released_at = time.monotonic()
            with adapter.create_session() as third:
                after = await third.execute_script("after = 1")
                await third.close()
            return editor, {"message": str(timed_out.value)}, refused.value, after, released_at
        finally:
            adapter.close()
            await editor.stop()

    editor, timed_out, refused, after, released_at = asyncio.run(scenario())

    assert "other Unreal calls get InstanceBusy" in timed_out["message"]
    assert refused.error_type == "InstanceBusy"
    assert refused.details["reason"] == "finishing_timed_out_call"
    assert refused.details["timed_out_tool"] == "PUT /remote/object/call"
    assert refused.details["changes_state"] is True
    assert "nothing was sent" in str(refused)
    assert after == {"success": True, "output": "ran"}
    commands = editor.commands()
    assert not any("marker_b" in command for command in commands), "the refused call never reached the editor"
    assert editor.routes() == ["/remote/object/call", "/remote/info", "/remote/object/call"]
    (_, _, _, _, slow_end), (_, _, _, probe_start, probe_end), _ = editor.ran
    assert probe_start >= slow_end, "the probe waited for the slow script"
    assert released_at >= probe_end


def test_a_read_only_tool_s_timeout_holds_the_editor_and_says_it_only_reads() -> None:
    async def scenario() -> UnrealError:
        editor = _GameThreadEditor()
        await editor.start()
        session = UnrealRuntimeSession(_settings(editor.port))
        session.tool_name, session.tool_read_only = "list_unreal_actors", True
        try:
            with pytest.raises(UnrealTimeoutError):
                await session.execute_script("import time; time.sleep(0.8)")
            with pytest.raises(UnrealError) as refused:
                await session.describe_object("/Game/Maps/Main.Main:PersistentLevel.Cube")
            return refused.value
        finally:
            await session.close()
            await editor.stop()

    refused = asyncio.run(scenario())

    assert refused.details["timed_out_tool"] == "list_unreal_actors"
    assert refused.details["changes_state"] is False
    assert "it only reads" in str(refused)


def test_hold_after_timeout_zero_holds_nothing() -> None:
    async def scenario() -> _GameThreadEditor:
        editor = _GameThreadEditor()
        await editor.start()
        session = UnrealRuntimeSession(_settings(editor.port, hold_after_timeout=0))
        try:
            with pytest.raises(UnrealTimeoutError) as timed_out:
                await session.execute_script("import time; time.sleep(0.5)")
            assert "InstanceBusy" not in str(timed_out.value)
            assert session._holds.get(session._endpoint) is None
            assert await session.execute_script("queued = 1") == {"success": True, "output": "ran"}
            return editor
        finally:
            await session.close()
            await editor.stop()

    editor = asyncio.run(scenario())

    assert "/remote/info" not in editor.routes()
    assert any("queued" in command for command in editor.commands()), "without a hold it queued behind"


def test_a_hold_past_its_limit_is_released_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    async def scenario() -> float:
        editor = _GameThreadEditor()
        await editor.start()
        adapter = UnrealRuntimeAdapter(_settings(editor.port, hold_after_timeout=0.6))
        endpoint = f"127.0.0.1:{editor.port}"
        try:
            with adapter.create_session() as session:
                with pytest.raises(UnrealTimeoutError):
                    await session.execute_script("import time; time.sleep(2.0)")
                await session.close()
            started = time.monotonic()
            await _released(adapter.holds, endpoint)
            return time.monotonic() - started
        finally:
            adapter.close()
            await editor.stop()

    with caplog.at_level("WARNING"):
        held_for = asyncio.run(scenario())

    assert held_for < 1.0, "released at the limit, not when the editor finished"
    assert "unreal.hold_after_timeout" in caplog.text
    assert "still had not finished PUT /remote/object/call" in caplog.text


def test_an_editor_that_went_away_ends_the_hold() -> None:
    async def scenario() -> float:
        editor = _GameThreadEditor()
        await editor.start()
        adapter = UnrealRuntimeAdapter(_settings(editor.port))
        endpoint = f"127.0.0.1:{editor.port}"
        try:
            with adapter.create_session() as session:
                with pytest.raises(UnrealTimeoutError):
                    await session.execute_script("import time; time.sleep(30)")
                await session.close()
            assert adapter.holds.get(endpoint) is not None
            started = time.monotonic()
            await editor.stop()  # the editor quits (or crashes) mid-script
            await _released(adapter.holds, endpoint)
            return time.monotonic() - started
        finally:
            adapter.close()
            await editor.stop()

    assert asyncio.run(scenario()) < 2.0


def test_ping_and_health_are_sent_during_the_hold_and_report_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """A probe the editor does not answer reports the hold; one it answers ends it."""

    async def never_answers(*_args: Any) -> str:
        await asyncio.Event().wait()
        return "answered"

    # Only the pings can end this hold.
    monkeypatch.setattr(_unreal_hold, "editor_answer", never_answers)

    async def scenario() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], _GameThreadEditor]:
        editor = _GameThreadEditor()
        await editor.start()
        adapter = UnrealRuntimeAdapter(_settings(editor.port))
        endpoint = f"127.0.0.1:{editor.port}"
        try:
            with adapter.create_session() as session:
                session.tool_name = "execute_unreal_script"
                with pytest.raises(UnrealTimeoutError):
                    await session.execute_script("import time; time.sleep(1.0)")
                busy_ping = await session.ping()
                busy_health = await session.health_check()
                await session.close()
            assert adapter.holds.get(endpoint) is not None
            await asyncio.sleep(1.0)  # the editor finishes the script meanwhile
            with adapter.create_session() as session:
                idle_ping = await session.ping()
                await session.close()
            assert adapter.holds.get(endpoint) is None
            return busy_ping, busy_health, idle_ping, editor
        finally:
            adapter.close()
            await editor.stop()

    busy_ping, busy_health, idle_ping, editor = asyncio.run(scenario())

    for busy in (busy_ping, busy_health):
        assert busy["busy"] is True and busy["error_type"] == "InstanceBusy"
        assert busy["held_after_timeout"]["timed_out_tool"] == "execute_unreal_script"
        assert busy["held_after_timeout"]["changes_state"] is True
        assert "still running execute_unreal_script" in busy["error"]
    assert idle_ping["reachable"] is True and "held_after_timeout" not in idle_ping
    assert editor.routes().count("/remote/info") == 3


def test_an_attached_ping_during_the_hold_reports_the_editor_busy(monkeypatch: pytest.MonkeyPatch) -> None:
    """In attached mode a ping first checks the attachment with Python, which the hold keeps from the editor."""

    async def never_answers(*_args: Any) -> str:
        await asyncio.Event().wait()
        return "answered"

    monkeypatch.setattr(_unreal_hold, "editor_answer", never_answers)

    async def scenario() -> Dict[str, Any]:
        session = UnrealRuntimeSession(_settings(1, mode="attached"))
        monkeypatch.setattr(session, "_load_attachment", lambda: {"viewport": "Viewport0"})
        monkeypatch.setattr(session, "_attachment_guard", lambda: "")
        request = _unreal_hold.TimedOutRequest("spawn_unreal_actor", "/remote/object/call", True, time.monotonic())
        session._holds.hold(session.host, session.port, request, {})
        try:
            return await session.ping()
        finally:
            await session.close()

    ping = asyncio.run(scenario())

    assert ping["busy"] is True and ping["error_type"] == "InstanceBusy" and ping["reachable"] is False
    assert ping["held_after_timeout"]["timed_out_tool"] == "spawn_unreal_actor"


def test_an_answer_to_a_probe_sent_after_the_timeout_ends_the_hold() -> None:
    """The editor answers in arrival order; an answer to an older request proves nothing."""

    async def scenario() -> Tuple[Optional[Any], Optional[Any]]:
        holds = _unreal_hold.EditorHolds(60)
        never = asyncio.Event()

        async def waits(*_args: Any) -> str:
            await never.wait()
            return "answered"

        request = _unreal_hold.TimedOutRequest("spawn_unreal_actor", "/remote/object/call", True, time.monotonic())
        original = _unreal_hold.editor_answer
        _unreal_hold.editor_answer = waits  # type: ignore[assignment]
        try:
            holds.hold("127.0.0.1", 1, request, {})
        finally:
            _unreal_hold.editor_answer = original  # type: ignore[assignment]
        holds.editor_answered("127.0.0.1:1", sent_at=request.since - 1.0)
        before = holds.get("127.0.0.1:1")
        holds.editor_answered("127.0.0.1:1", sent_at=request.since + 0.1)
        await asyncio.sleep(0)
        return before, holds.get("127.0.0.1:1")

    before, after = asyncio.run(scenario())

    assert before is not None, "an answer to a request sent before the timeout keeps the hold"
    assert after is None


def test_the_probe_carries_the_passphrase() -> None:
    async def scenario() -> _GameThreadEditor:
        editor = _GameThreadEditor()
        await editor.start()
        adapter = UnrealRuntimeAdapter(_settings(editor.port, passphrase="secret"))
        try:
            with adapter.create_session() as session:
                with pytest.raises(UnrealTimeoutError):
                    await session.execute_script("import time; time.sleep(0.5)")
                await session.close()
            await _released(adapter.holds, f"127.0.0.1:{editor.port}")
            return editor
        finally:
            adapter.close()
            await editor.stop()

    editor = asyncio.run(scenario())

    (_, _, script_headers, _, _), (route, _, probe_headers, _, _) = editor.ran
    assert route == "/remote/info"
    assert probe_headers["passphrase"] == script_headers["passphrase"]


def test_a_property_write_to_a_held_editor_is_refused_before_anything_is_sent() -> None:

    async def scenario() -> Tuple[UnrealError, _GameThreadEditor]:
        editor = _GameThreadEditor()
        await editor.start()
        session = UnrealRuntimeSession(_settings(editor.port))
        try:
            with pytest.raises(UnrealTimeoutError):
                await session.execute_script("import time; time.sleep(0.8)")
            with pytest.raises(UnrealError) as refused:
                await session.set_actor_property(
                    "/Game/Maps/Main.Main:PersistentLevel.Cube", "CustomTimeDilation", "2.0"
                )
            await _released(session._holds, session._endpoint)
            return refused.value, editor
        finally:
            await session.close()
            await editor.stop()

    refused, editor = asyncio.run(scenario())

    assert refused.error_type == "InstanceBusy"
    # Neither the write's batch nor a second decision nor a read-back was sent.
    assert editor.routes() == ["/remote/object/call", "/remote/info"]


def test_a_write_whose_batch_timed_out_leaves_its_verdict_to_the_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deciding the write again in a request of its own is refused; the batch's own verdict will run.

    It used to say the write's undo transaction may still be open.
    """

    async def never_answers(*_args: Any) -> str:
        await asyncio.Event().wait()
        return "answered"

    monkeypatch.setattr(_unreal_hold, "editor_answer", never_answers)

    async def scenario() -> Dict[str, Any]:
        session = UnrealRuntimeSession(_settings(1))
        try:
            request = _unreal_hold.TimedOutRequest(
                "set_unreal_actor_property", "/remote/batch", True, time.monotonic()
            )
            session._holds.hold(session.host, session.port, request, {})
            return await session._decide_property_write("token", 2.0, "the batch timed out")
        finally:
            await session.close()

    decided = asyncio.run(scenario())

    assert decided["applied"] is None and decided["restored"] is False
    assert "still running the write's batch" in decided["late"]


# ---------------------------------------------------------------------------
# The MCP server: the tools, ping_unreal, shutdown
# ---------------------------------------------------------------------------


def _server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, port: int, **unreal: Any
) -> server_module.SimulMCPServer:
    monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
    monkeypatch.setattr(server_module, "TaskConfig", None)
    monkeypatch.setattr(backends_module, "is_headless_available", lambda: False)
    monkeypatch.setattr(backends_module, "is_blender_available", lambda: False)
    settings = _settings(port, **unreal)
    settings = settings.model_copy(
        update={"isaac_sim": settings.isaac_sim.model_copy(update={"enabled": False})}
    )
    instance = server_module.SimulMCPServer(settings=settings)
    instance.usage_tracker = ToolUsageTracker(log_dir=tmp_path)
    return instance


def _payload(result: Any) -> Dict[str, Any]:
    return json.loads(result.content[-1].text)


def test_the_cascade_through_the_tools(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """p36.py against the real tools and session: B is refused unsent, a call once idle succeeds."""

    async def scenario() -> Tuple[_GameThreadEditor, List[Dict[str, Any]], List[float], Dict[str, Any]]:
        editor = _GameThreadEditor()
        await editor.start()
        instance = _server(monkeypatch, tmp_path, editor.port)
        tools = instance.mcp.by_name
        adapter = instance.unreal_adapter
        assert isinstance(adapter, UnrealRuntimeAdapter)
        t0 = time.monotonic()
        answers: List[Dict[str, Any]] = []
        times: List[float] = []
        try:
            for code in ("import time; time.sleep(1.2)", "marker_b = 1", "marker_c = 1"):
                answers.append(_payload(await tools["execute_unreal_script"](code=code)))
                times.append(time.monotonic() - t0)
            ping = _payload(await tools["ping_unreal"]())
            await _released(adapter.holds, f"127.0.0.1:{editor.port}")
            answers.append(_payload(await tools["execute_unreal_script"](code="after = 1")))
            return editor, answers, times, ping
        finally:
            await instance.shutdown()
            await editor.stop()

    editor, answers, times, ping = asyncio.run(scenario())

    timed_out, busy_b, busy_c, after = answers
    assert timed_out["error_type"] == "TimeoutError"
    for busy in (busy_b, busy_c):
        assert busy["error_type"] == "InstanceBusy"
        assert busy["details"]["reason"] == "finishing_timed_out_call"
        assert busy["details"]["timed_out_tool"] == "execute_unreal_script"
        assert busy["details"]["changes_state"] is True
    assert times[2] - times[0] < 0.2, "the refusals are immediate"
    assert ping["busy"] is True
    assert ping["held_after_timeout"]["timed_out_tool"] == "execute_unreal_script"
    assert after["success"] is True
    commands = editor.commands()
    assert not any("marker_b" in command or "marker_c" in command for command in commands)
    assert any("after" in command for command in commands)


def test_shutdown_releases_a_held_editor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    async def scenario() -> Tuple[float, Optional[Any]]:
        editor = _GameThreadEditor()
        await editor.start()
        instance = _server(monkeypatch, tmp_path, editor.port)
        adapter = instance.unreal_adapter
        assert isinstance(adapter, UnrealRuntimeAdapter)
        try:
            await instance.mcp.by_name["execute_unreal_script"](code="import time; time.sleep(5)")
            hold = adapter.holds.get(f"127.0.0.1:{editor.port}")
            assert hold is not None
            started = time.monotonic()
            await instance.shutdown()
            took = time.monotonic() - started
            await asyncio.sleep(0)
            return took, hold.task
        finally:
            await editor.stop()

    with caplog.at_level("WARNING"):
        took, task = asyncio.run(scenario())

    assert took < 1.0
    assert task is not None and task.cancelled()
    assert "Released the Unreal editor" in caplog.text and "execute_unreal_script" in caplog.text


# ---------------------------------------------------------------------------
# Another simul server on the same editor (round-3 audit M3-2)
# ---------------------------------------------------------------------------


def test_another_server_s_hold_refuses_requests_to_the_same_editor_unsent(tmp_path: Path) -> None:
    """Each agent's client spawns its own server: the hold covered only the server that saw the timeout.

    Two adapters stand for two servers; their holds meet in the shared records
    (``simul.utils.engine_holds``), here a directory of the test's own.
    """

    async def scenario() -> Tuple[_GameThreadEditor, UnrealError, Dict[str, Any], List[str], Dict[str, Any]]:
        editor = _GameThreadEditor()
        await editor.start()
        records = tmp_path / "holds"
        settings = _settings(editor.port)
        server_a, server_b = UnrealRuntimeAdapter(settings), UnrealRuntimeAdapter(settings)
        server_a.holds = _unreal_hold.EditorHolds(1.0e3, SharedHolds("unreal", records))
        server_b.holds = _unreal_hold.EditorHolds(1.0e3, SharedHolds("unreal", records))
        endpoint = f"127.0.0.1:{editor.port}"
        try:
            with server_a.create_session() as a:
                with pytest.raises(UnrealTimeoutError):
                    await a.execute_script("import time; time.sleep(1.0)")
                await a.close()
            with server_b.create_session() as b:
                with pytest.raises(UnrealError) as refused:
                    await b.execute_script("marker_b = 1")
                ping = await b.ping()
                await b.close()
            await _released(server_a.holds, endpoint)
            left = sorted(path.name for path in records.glob("*.json"))
            with server_b.create_session() as b:
                after = await b.execute_script("after_b = 1")
                await b.close()
            return editor, refused.value, ping, left, after
        finally:
            server_a.close()
            server_b.close()
            await editor.stop()

    editor, refused, ping, left, after = asyncio.run(scenario())

    assert refused.error_type == "InstanceBusy"
    assert refused.details["reason"] == "finishing_timed_out_call"
    assert refused.details["held_by"] == {"pid": os.getpid()}
    assert refused.details["timed_out_route"] == "/remote/object/call"
    assert refused.details["changes_state"] is True
    assert "another simul server" in str(refused) and "nothing was sent" in str(refused)
    assert ping["held_after_timeout"]["held_by"] == {"pid": os.getpid()}
    assert left == [], "server A removed its record once the editor answered"
    assert after == {"success": True, "output": "ran"}
    assert not any("marker_b" in command for command in editor.commands())
