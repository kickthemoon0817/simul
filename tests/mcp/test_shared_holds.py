"""A hold after a timed-out Isaac call covers every simul server on the machine.

Before (round-3 audit M3-2): the hold lived in one server's memory. In the
usual multi-agent setup each agent's client spawns its own stdio server, so a
second server on the same Isaac Sim knew nothing of it: its script queued
inside Kit behind the timed-out one, timed out too and ran after it was
reported failed (``r3_hold.py h10``: B's marker reported ``TimeoutError`` at
6.2 s and was written at 8.1 s; over the bridge, ``r3_h10b.py``, at 14.2 s
and 20.1 s).

Now each hold is also a record under ``$SIMUL_HOME/holds``
(``simul.utils.engine_holds``), which the other servers read before they send
anything: they answer ``InstanceBusy`` until the holder sees the engine idle
and removes it. Records past their expiry are ignored. During a hold the
listing and status tools report it from the servers' own state, without a
probe that would wait behind it (m3-1).

Review follow-ups of #370: a hold ended with its server, so after the holder
stopped (its record withdrawn) or was killed (its record ignored) the other
servers' calls queued behind the engine again (B's marker ran at 9.1 s). Now
a stopping holder leaves its record ``released``, a record whose process is
gone (by pid and start time, so a reused pid does not keep it alive) is an
orphan, and the next server about to send that engine something takes the
hold over and runs its own idle probe first.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

import simul
from simul.adapters.isaac_socket_client import IDLE_PROBE_SCRIPT
from simul.config import Settings
from simul.mcp import server as server_module
from simul.mcp.session_manager import SessionManager
from simul.mcp.usage_tracker import ToolUsageTracker
from simul.utils import engine_holds, process_identity
from simul.utils.engine_holds import SharedHolds, endpoint_key
from tests.tcp_fakes import FakeTcpServer

SRC = str(Path(simul.__file__).resolve().parents[1])
posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX process checks and signals")


class _Kit:
    """Kit's 5.x stock socket in miniature: one script at a time; ``sleep(N)`` holds the main thread."""

    def __init__(self) -> None:
        self.ran: List[Tuple[str, float, float]] = []
        self.port = 0
        self._main_thread = asyncio.Lock()
        self._server = FakeTcpServer(self._handle)

    async def start(self) -> None:
        self.port = await self._server.start()

    async def stop(self) -> None:
        await self._server.stop()

    def printed(self) -> List[str]:
        found = (re.search(r"print\('([^']*)'\)", code) for code, _start, _end in self.ran)
        # The hold's idle probe prints simul-idle.
        return [match.group(1) for match in found if match and match.group(1) != "simul-idle"]

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        code = (await reader.read(65536)).decode()
        if not code:
            return  # a reachability probe that only connects
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


# ---------------------------------------------------------------------------
# The records
# ---------------------------------------------------------------------------


def _live() -> bool:
    return True


def test_another_keeper_sees_a_hold_and_its_own_keeper_does_not(tmp_path: Path) -> None:
    mine, theirs = SharedHolds("isaac", tmp_path), SharedHolds("isaac", tmp_path)
    record = mine.publish("127.0.0.1:8226", "execute_isaac_script", time.time() - 2.0, 300.0, _live)

    seen = theirs.foreign("127.0.0.1:8226")

    assert record is not None and record.exists()
    assert oct(record.stat().st_mode & 0o777) == "0o600"
    assert mine.foreign("127.0.0.1:8226") is None
    assert seen is not None and seen.tool == "execute_isaac_script" and seen.pid == os.getpid()
    info = seen.info()
    assert info["timed_out_tool"] == "execute_isaac_script"
    assert info["busy_for_seconds"] >= 2.0 and info["hold_limit_seconds"] == 300.0
    assert info["held_by"] == {"pid": os.getpid()}
    assert theirs.foreign("127.0.0.1:8227") is None, "another engine is not held"
    assert SharedHolds("unreal", tmp_path).foreign("127.0.0.1:8226") is None, "another backend's"

    mine.withdraw(record)

    assert not record.exists()
    assert theirs.foreign("127.0.0.1:8226") is None


def test_loopback_spellings_name_one_endpoint() -> None:
    assert endpoint_key("localhost", 8226) == endpoint_key("127.0.0.1", 8226) == "127.0.0.1:8226"
    assert endpoint_key("[::1]", 30010) == endpoint_key("LocalHost", 30010) == "127.0.0.1:30010"
    assert endpoint_key("Isaac-Box", 8226) == "isaac-box:8226"


def test_records_past_their_expiry_are_removed_and_an_exited_holder_s_is_an_orphan(tmp_path: Path) -> None:
    exited = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True)
    dead_pid = int(exited.stdout)
    reader = SharedHolds("isaac", tmp_path)
    tmp_path.mkdir(exist_ok=True)
    base = {
        "backend": "isaac", "endpoint": "127.0.0.1:8226", "tool": "x", "since": time.time(),
        "host": engine_holds._HOST,
    }
    files = {
        "dead": (tmp_path / f"isaac-127_0_0_1_8226.{dead_pid}.aaaaaaaaaaaa.1.json",
                 {**base, "pid": dead_pid, "expires_at": time.time() + 300}),
        "expired": (tmp_path / f"isaac-127_0_0_1_8226.{os.getppid()}.bbbbbbbbbbbb.1.json",
                    {**base, "pid": os.getppid(), "expires_at": time.time() - 1}),
        # This process, but no running hold wrote it (a hold whose loop ended).
        "stale": (tmp_path / f"isaac-127_0_0_1_8226.{os.getpid()}.cccccccccccc.1.json",
                  {**base, "pid": os.getpid(), "expires_at": time.time() + 300}),
    }
    for path, record in files.values():
        engine_holds.BridgeFiles.write(path, record)

    assert reader.foreign("127.0.0.1:8226") is None
    orphan = reader.orphan("127.0.0.1:8226")
    assert orphan is not None and orphan.pid == dead_pid and orphan.orphaned
    assert orphan.info()["held_by"] == {"pid": dead_pid, "stopped": True}
    # The exited holder's record stays for the next server to take over.
    assert [path.exists() for path, _ in files.values()] == [True, False, False]


@posix_only
def test_a_live_process_s_record_counts_until_it_expires(tmp_path: Path) -> None:
    reader = SharedHolds("isaac", tmp_path)
    tmp_path.mkdir(exist_ok=True)
    record = {
        "backend": "isaac", "endpoint": "127.0.0.1:8226", "tool": "step_isaac_simulation",
        "since": time.time(), "expires_at": time.time() + 300, "pid": os.getppid(), "host": engine_holds._HOST,
    }
    engine_holds.BridgeFiles.write(tmp_path / f"isaac-127_0_0_1_8226.{os.getppid()}.dddddddddddd.1.json", record)

    seen = reader.foreign("127.0.0.1:8226")

    assert seen is not None and seen.tool == "step_isaac_simulation" and seen.pid == os.getppid()


def test_a_hold_that_cannot_be_recorded_covers_its_own_server(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("")
    holds = SharedHolds("isaac", blocked)

    with caplog.at_level("WARNING"):
        record = holds.publish("127.0.0.1:8226", "x", time.time(), 300.0, _live)

    assert record is None
    assert "covers this server only" in caplog.text
    assert SharedHolds("isaac", blocked).foreign("127.0.0.1:8226") is None


_ENDPOINT = "127.0.0.1:8226"


def _record(directory: Path, pid: int, **fields: Any) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"isaac-127_0_0_1_8226.{pid}.eeeeeeeeeeee.1.json"
    engine_holds.BridgeFiles.write(path, {
        "backend": "isaac", "endpoint": _ENDPOINT, "tool": "execute_isaac_script", "since": time.time(),
        "expires_at": time.time() + 300, "pid": pid, "host": engine_holds._HOST, **fields,
    })
    return path


def test_a_record_carries_its_holder_s_start_time(tmp_path: Path) -> None:
    record = SharedHolds("isaac", tmp_path).publish(_ENDPOINT, "x", time.time(), 300.0, _live)

    assert record is not None
    started = json.loads(record.read_text()).get("pid_started")
    assert started == process_identity.process_started(os.getpid())
    if sys.platform in ("darwin", "linux"):
        assert started is not None


def test_an_exited_process_has_no_start_time_and_is_not_alive() -> None:
    exited = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True)
    dead_pid = int(exited.stdout)

    assert process_identity.pid_alive(os.getpid()) is True
    assert process_identity.pid_alive(dead_pid) is False
    assert process_identity.pid_alive(0) is False
    assert process_identity.process_started(dead_pid) is None
    assert process_identity.own_start() == process_identity.process_started(os.getpid())


@posix_only
def test_a_reused_pid_does_not_keep_an_exited_holder_s_record_live(tmp_path: Path) -> None:
    """Review of #370: a record whose pid a live process had taken blocked B until its expiry (300 s)."""
    live_pid = os.getppid()
    started = process_identity.process_started(live_pid)
    if started is None:
        pytest.skip("no process start time on this platform")
    reader = SharedHolds("isaac", tmp_path)
    path = _record(tmp_path, live_pid, pid_started=f"not-{started}")

    assert reader.foreign(_ENDPOINT) is None
    orphan = reader.orphan(_ENDPOINT)
    assert orphan is not None and orphan.pid == live_pid

    path.unlink()
    _record(tmp_path, live_pid, pid_started=started)
    assert reader.foreign(_ENDPOINT) is not None, "the same pid and start time: the holder runs"


def test_where_a_pid_cannot_be_probed_the_start_time_or_the_expiry_ends_a_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows: os.kill(pid, 0) would end the process, so every crashed holder blocked the others for 300 s."""
    monkeypatch.setattr(engine_holds, "pid_alive", lambda pid: True)
    monkeypatch.setattr(engine_holds, "process_started", lambda pid: "a later process")
    reader = SharedHolds("isaac", tmp_path)
    _record(tmp_path, 4242, pid_started="the holder")

    assert reader.foreign(_ENDPOINT) is None and reader.orphan(_ENDPOINT) is not None

    monkeypatch.setattr(engine_holds, "process_started", lambda pid: None)  # cannot be told either
    assert reader.foreign(_ENDPOINT) is not None, "live until it expires"


def test_a_temporary_file_a_killed_writer_left_is_removed_once_stale(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    abandoned, writing = tmp_path / ".simul-abandoned", tmp_path / ".simul-writing"
    for path in (abandoned, writing):
        path.write_text("{")
    stale = time.time() - engine_holds.STALE_TEMP_SECONDS - 5
    os.utime(abandoned, (stale, stale))

    SharedHolds("isaac", tmp_path).foreign(_ENDPOINT)

    assert not abandoned.exists()
    assert writing.exists(), "a write in progress is left alone"


def test_exactly_one_keeper_takes_a_released_hold_over(tmp_path: Path) -> None:
    holder = SharedHolds("isaac", tmp_path)
    record = holder.publish(
        _ENDPOINT, "execute_isaac_script", time.time() - 1.0, 300.0, _live, {"transport": "bridge"}
    )
    holder.release(record)
    b, c = SharedHolds("isaac", tmp_path), SharedHolds("isaac", tmp_path)
    orphan_b, orphan_c = b.orphan(_ENDPOINT), c.orphan(_ENDPOINT)
    assert orphan_b is not None and orphan_b == orphan_c and orphan_b.orphaned
    assert b.foreign(_ENDPOINT) is None, "a released record is not a live hold"

    taken_b, mine_b = b.adopt(orphan_b, _live)
    taken_c, mine_c = c.adopt(orphan_c, _live)

    assert (taken_b, taken_c) == (True, False) and mine_c is None
    assert record is not None and not record.exists()
    seen = c.foreign(_ENDPOINT)
    assert seen is not None and seen.detail("adopted_from") == os.getpid()
    assert seen.detail("transport") == "bridge"
    assert (seen.since, seen.expires_at) == (orphan_b.since, orphan_b.expires_at), "the hold is not extended"
    assert c.orphan(_ENDPOINT) is None
    b.withdraw(mine_b)
    assert list(tmp_path.glob("*.json")) == []


# ---------------------------------------------------------------------------
# Two servers in one process, one Kit
# ---------------------------------------------------------------------------


def _server(tmp_path: Path, name: str, kit: _Kit, **isaac: Any) -> server_module.SimulMCPServer:
    settings = Settings(
        server={"timeout": 5, "shutdown_timeout": 0.5},
        isaac_sim={
            "socket_port": kit.port,
            "socket_timeout": 1.0,
            "socket_protocol": "vscode",
            "bridge_enabled": False,
            "discovery_dir": str(tmp_path / "discovery"),
            **isaac,
        },
        security={"rate_limiting_enabled": False},
    )
    instance = server_module.SimulMCPServer(settings, backends={"isaac"})
    instance.session_manager = SessionManager(tmp_path / name / "sessions")
    instance.usage_tracker = ToolUsageTracker(log_dir=tmp_path / name / "usage")
    return instance


def _payload(result: Any) -> Dict[str, Any]:
    return json.loads(result.content[-1].text)


async def _call(client: Client, tool: str, **arguments: Any) -> Dict[str, Any]:
    return _payload(await client.call_tool(tool, arguments, raise_on_error=False))


async def _until(predicate: Any, within: float = 10.0) -> None:
    deadline = time.monotonic() + within
    while not predicate():
        assert time.monotonic() < deadline, "timed out waiting"
        await asyncio.sleep(0.05)


def test_another_server_answers_instance_busy_until_the_holder_sees_the_engine_idle(tmp_path: Path) -> None:
    async def scenario() -> Dict[str, Any]:
        kit = _Kit()
        await kit.start()
        a, b = _server(tmp_path, "a", kit), _server(tmp_path, "b", kit)
        out: Dict[str, Any] = {}
        try:
            async with Client(a.mcp, mode="legacy") as agent_a, Client(b.mcp, mode="legacy") as agent_b:
                t0 = time.monotonic()
                out["timed_out"] = await _call(agent_a, "execute_isaac_script", code="sleep(3.5); print('slow')")
                out["a_held"] = "default" in a._isaac_holds
                started = time.monotonic()
                out["refused"] = await _call(agent_b, "execute_isaac_script", code="print('b-marker')")
                out["refused_in"] = time.monotonic() - started
                out["refused_read"] = await _call(agent_b, "get_isaac_stage_info")
                out["ping"] = await _call(agent_b, "ping_isaac")
                out["runtime"] = await _call(agent_b, "get_isaac_runtime_info")
                out["listing"] = await _call(agent_b, "list_isaac_instances", scan=False)
                out["status_in"] = time.monotonic() - started
                await _until(lambda: not a._isaac_holds)
                out["released_after"] = time.monotonic() - t0
                out["records_left"] = sorted(p.name for p in engine_holds._DEFAULT_HOLD_DIR.glob("isaac-*.json"))
                out["after"] = await _call(agent_b, "execute_isaac_script", code="print('b-after')")
        finally:
            await a.shutdown()
            await b.shutdown()
            await kit.stop()
        out["printed"] = kit.printed()
        out["probe_ran"] = any(code == IDLE_PROBE_SCRIPT for code, _s, _e in kit.ran)
        return out

    out = asyncio.run(scenario())

    assert out["timed_out"]["error_type"] == "TimeoutError" and out["a_held"]
    refused = out["refused"]
    assert refused["error_type"] == "InstanceBusy"
    assert refused["details"]["reason"] == "finishing_timed_out_call"
    assert refused["details"]["timed_out_tool"] == "execute_isaac_script"
    assert refused["details"]["held_by"] == {"pid": os.getpid()}
    assert "another simul server" in refused["error"] and "nothing was sent" in refused["error"]
    assert out["refused_in"] < 1.0, "refused at once, not after server.timeout"
    assert out["refused_read"]["error_type"] == "InstanceBusy", "a read would wait behind it too"
    # The status tools answer from the servers' state, at once (m3-1).
    ping = out["ping"]
    assert ping["success"] is True and ping["busy"] is True and ping["probed"] is False
    assert ping["held_after_timeout"]["held_by"] == {"pid": os.getpid()}
    runtime = out["runtime"]
    assert runtime["success"] is True and runtime["busy"] is True
    assert runtime["held_after_timeout"]["timed_out_tool"] == "execute_isaac_script"
    (listed,) = out["listing"]["instances"]
    assert listed["instance_status"] == "busy" and listed["reachable"] is True and listed["busy"] is True
    assert listed["compatibility"] == "blocked"
    assert out["status_in"] < 1.5, "no status tool waited for a probe"
    # The holder removed its record once Kit had finished the slow script.
    assert out["records_left"] == []
    assert out["after"]["success"] is True
    assert out["printed"] == ["slow", "b-after"], "B's marker never reached Kit"
    assert out["probe_ran"]


def test_the_holding_server_reports_its_own_hold_from_its_state(tmp_path: Path) -> None:
    """m3-1: the listing waited for its probe and said unreachable; ping said ConnectionError."""

    async def scenario() -> Dict[str, Any]:
        kit = _Kit()
        await kit.start()
        a = _server(tmp_path, "a", kit)
        out: Dict[str, Any] = {}
        try:
            async with Client(a.mcp, mode="legacy") as agent:
                out["timed_out"] = await _call(agent, "execute_isaac_script", code="sleep(3); print('slow')")
                started = time.monotonic()
                out["ping"] = await _call(agent, "ping_isaac")
                out["runtime"] = await _call(agent, "get_isaac_runtime_info")
                out["listing"] = await _call(agent, "list_isaac_instances", scan=False)
                out["took"] = time.monotonic() - started
        finally:
            await a.shutdown()
            await kit.stop()
        return out

    out = asyncio.run(scenario())

    assert out["timed_out"]["error_type"] == "TimeoutError"
    assert out["ping"]["success"] is True and out["ping"]["probed"] is False
    assert "held_by" not in out["ping"]["held_after_timeout"]
    assert "execute_isaac_script" in out["ping"]["note"]
    assert out["runtime"]["busy"] is True and "client" in out["runtime"]
    (listed,) = out["listing"]["instances"]
    assert listed["instance_status"] == "busy"
    assert "execute_isaac_script" in listed["compatibility_reason"]
    assert out["took"] < 1.5


def test_shutdown_leaves_the_record_released_for_the_other_servers(tmp_path: Path) -> None:
    async def scenario() -> Tuple[List[str], List[Dict[str, Any]], Any]:
        kit = _Kit()
        await kit.start()
        a = _server(tmp_path, "a", kit)
        try:
            async with Client(a.mcp, mode="legacy") as agent:
                await _call(agent, "execute_isaac_script", code="sleep(3); print('slow')")
                during = sorted(p.name for p in engine_holds._DEFAULT_HOLD_DIR.glob("isaac-*.json"))
                await a.shutdown()
                after = [json.loads(p.read_text()) for p in engine_holds._DEFAULT_HOLD_DIR.glob("isaac-*.json")]
                orphan = SharedHolds("isaac").orphan(endpoint_key("127.0.0.1", kit.port))
        finally:
            await kit.stop()
        return during, after, orphan

    during, after, orphan = asyncio.run(scenario())

    assert len(during) == 1 and during[0].startswith("isaac-127_0_0_1_")
    (record,) = after
    assert record["released"] is True and record["tool"] == "execute_isaac_script"
    assert record["details"] == {"transport": "socket"}
    assert orphan is not None and orphan.pid == os.getpid()


def _records() -> List[str]:
    return sorted(p.name for p in engine_holds._DEFAULT_HOLD_DIR.glob("isaac-*.json"))


def test_a_hold_a_stopped_server_left_is_taken_over_by_the_next_call(tmp_path: Path) -> None:
    """Review of #370: A stopped mid-hold, and B's next call queued behind A's script (marker at 9.1 s)."""

    async def scenario() -> Dict[str, Any]:
        kit = _Kit()
        await kit.start()
        a, b = _server(tmp_path, "a", kit), _server(tmp_path, "b", kit)
        out: Dict[str, Any] = {}
        try:
            async with Client(a.mcp, mode="legacy") as agent_a, Client(b.mcp, mode="legacy") as agent_b:
                t0 = time.monotonic()
                out["timed_out"] = await _call(agent_a, "execute_isaac_script", code="sleep(4); print('slow')")
                await a.shutdown()
                out["ping"] = await _call(agent_b, "ping_isaac")
                started = time.monotonic()
                out["refused"] = await _call(agent_b, "execute_isaac_script", code="print('b-marker')")
                out["refused_in"] = time.monotonic() - started
                out["b_held"] = "default" in b._isaac_holds
                out["records_during"] = [json.loads((engine_holds._DEFAULT_HOLD_DIR / name).read_text())
                                         for name in _records()]
                await _until(lambda: not b._isaac_holds)
                out["released_after"] = time.monotonic() - t0
                out["records_left"] = _records()
                out["after"] = await _call(agent_b, "execute_isaac_script", code="print('b-after')")
        finally:
            await a.shutdown()
            await b.shutdown()
            await kit.stop()
        out["printed"] = kit.printed()
        return out

    out = asyncio.run(scenario())

    assert out["timed_out"]["error_type"] == "TimeoutError"
    ping = out["ping"]
    assert ping["busy"] is True and ping["probed"] is False
    assert ping["held_after_timeout"]["held_by"] == {"pid": os.getpid(), "stopped": True}
    refused = out["refused"]
    assert refused["error_type"] == "InstanceBusy", refused
    assert refused["details"]["reason"] == "finishing_timed_out_call"
    assert refused["details"]["adopted_from"] == os.getpid()
    assert refused["details"]["hold_limit_seconds"] == 300.0
    assert "stopped holding it" in refused["error"]
    assert out["refused_in"] < 2.0, "refused after one probe window, not after a timeout"
    assert out["b_held"]
    (record,) = out["records_during"]
    assert record["details"]["adopted_from"] == os.getpid() and "released" not in record
    assert out["released_after"] >= 3.5, "B held the engine until Kit had finished A's script"
    assert out["records_left"] == []
    assert out["after"]["success"] is True
    assert out["printed"] == ["slow", "b-after"], "B's marker never reached Kit"


def test_an_orphan_on_an_idle_engine_costs_the_call_one_probe(tmp_path: Path) -> None:
    exited = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True)
    dead_pid = int(exited.stdout)

    async def scenario() -> Dict[str, Any]:
        kit = _Kit()
        await kit.start()
        b = _server(tmp_path, "b", kit)
        endpoint = endpoint_key("127.0.0.1", kit.port)
        engine_holds.BridgeFiles.write(
            engine_holds._DEFAULT_HOLD_DIR
            / f"isaac-{endpoint.replace('.', '_').replace(':', '_')}.{dead_pid}.ffffffffffff.1.json",
            {"backend": "isaac", "endpoint": endpoint, "tool": "step_isaac_simulation", "since": time.time() - 5,
             "expires_at": time.time() + 295, "pid": dead_pid, "host": engine_holds._HOST},
        )
        out: Dict[str, Any] = {}
        try:
            async with Client(b.mcp, mode="legacy") as agent:
                started = time.monotonic()
                out["call"] = await _call(agent, "execute_isaac_script", code="print('b-marker')")
                out["took"] = time.monotonic() - started
                out["left"] = _records()
        finally:
            await b.shutdown()
            await kit.stop()
        out["ran"] = [code for code, _s, _e in kit.ran]
        out["printed"] = kit.printed()
        return out

    out = asyncio.run(scenario())

    assert out["call"]["success"] is True, out["call"]
    assert out["took"] < 1.0
    assert out["left"] == []
    assert out["ran"][0] == IDLE_PROBE_SCRIPT, "the engine was checked before the call was sent"
    assert out["printed"] == ["b-marker"]


def test_a_hold_is_not_taken_over_while_holds_are_off(tmp_path: Path) -> None:
    async def scenario() -> Tuple[Dict[str, Any], List[str]]:
        kit = _Kit()
        await kit.start()
        b = _server(tmp_path, "b", kit, hold_after_timeout=0)
        holder = SharedHolds("isaac")
        holder.release(holder.publish(endpoint_key("127.0.0.1", kit.port), "x", time.time(), 300.0, _live))
        try:
            async with Client(b.mcp, mode="legacy") as agent:
                call = await _call(agent, "execute_isaac_script", code="print('b-marker')")
        finally:
            await b.shutdown()
            await kit.stop()
        return call, [code for code, _s, _e in kit.ran]

    call, ran = asyncio.run(scenario())

    assert call["success"] is True
    assert IDLE_PROBE_SCRIPT not in ran


# ---------------------------------------------------------------------------
# Two real server processes, one Kit (the auditor's h10)
# ---------------------------------------------------------------------------


def _server_env(home: Path, kit: _Kit) -> Dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("ISAAC_SIM__", "SERVER__", "SECURITY__", "UNREAL__", "BLENDER__", "USD__"))
        and key not in ("SIMUL_HOME", "LOGGING__FILE_PATH", "LOGGING__AUDIT_PATH", "CONFIG_FILE")
    }
    env.update(
        HOME=str(home),
        SIMUL_HOME=str(home / "state"),
        PYTHONPATH=os.pathsep.join(filter(None, [SRC, os.environ.get("PYTHONPATH")])),
        ISAAC_SIM__ENABLED="true",
        ISAAC_SIM__SOCKET_PORT=str(kit.port),
        ISAAC_SIM__SOCKET_PROTOCOL="vscode",
        ISAAC_SIM__BRIDGE_ENABLED="false",
        ISAAC_SIM__SOCKET_TIMEOUT="2",
        ISAAC_SIM__DISCOVERY_DIR=str(home / "discovery"),
        SECURITY__RATE_LIMITING_ENABLED="false",
    )
    return env


def _stdio_server(home: Path, kit: _Kit, name: str) -> Client:
    transport = StdioTransport(
        command=sys.executable,
        args=["-m", "simul.cli.main", "server", "--backends", "isaac"],
        env=_server_env(home, kit),
        cwd=str(home),
        keep_alive=False,
        log_file=home / f"{name}.stderr",
    )
    return Client(transport, mode="legacy", timeout=60)


@posix_only
def test_two_server_processes_on_one_engine_do_not_cascade(tmp_path: Path) -> None:
    """r3_hold.py h10 with the shared SIMUL_HOME two agents' stdio servers have."""

    async def scenario() -> Dict[str, Any]:
        kit = _Kit()
        await kit.start()
        out: Dict[str, Any] = {}
        try:
            async with _stdio_server(tmp_path, kit, "a") as a, _stdio_server(tmp_path, kit, "b") as b:
                t0 = time.monotonic()
                out["a"] = await _call(a, "execute_isaac_script", code="sleep(5); print('a-slow')")
                out["a_at"] = time.monotonic() - t0
                out["records"] = [
                    json.loads(p.read_text()) for p in (tmp_path / "state" / "holds").glob("isaac-*.json")
                ]
                out["b"] = await _call(b, "execute_isaac_script", code="print('b-marker')")
                out["b_at"] = time.monotonic() - t0
                out["printed_when_b_answered"] = kit.printed()
                deadline = time.monotonic() + 15
                while list((tmp_path / "state" / "holds").glob("isaac-*.json")):
                    assert time.monotonic() < deadline, "the holder never removed its record"
                    await asyncio.sleep(0.1)
                out["released_at"] = time.monotonic() - t0
                out["b_after"] = await _call(b, "execute_isaac_script", code="print('b-after')")
        finally:
            await kit.stop()
        out["printed"] = kit.printed()
        return out

    out = asyncio.run(scenario())

    assert out["a"]["error_type"] == "TimeoutError"
    (record,) = out["records"]
    assert record["tool"] == "execute_isaac_script" and record["endpoint"].startswith("127.0.0.1:")
    assert record["expires_at"] - record["since"] == pytest.approx(300.0)
    b = out["b"]
    assert b["error_type"] == "InstanceBusy", b
    assert b["details"]["held_by"]["pid"] == record["pid"]
    assert out["b_at"] - out["a_at"] < 2.0, "B was refused at once, not after its own timeout"
    assert out["released_at"] >= 4.5, "the record stayed until Kit had finished A's script"
    assert out["b_after"]["success"] is True
    assert out["printed"] == ["a-slow", "b-after"], "B's marker never reached Kit"


@posix_only
def test_a_hold_a_killed_server_left_is_taken_over_by_another_process(tmp_path: Path) -> None:
    """Review of #370: with A SIGKILLed mid-hold, B's next call timed out at 6.2 s and its marker ran at 9.1 s."""

    async def rpc(proc: "subprocess.Popen[bytes]", message: Dict[str, Any], reply: bool = True) -> Any:
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(json.dumps(message).encode() + b"\n")
        proc.stdin.flush()
        return json.loads(await asyncio.to_thread(proc.stdout.readline)) if reply else None

    async def scenario() -> Dict[str, Any]:
        kit = _Kit()
        await kit.start()
        out: Dict[str, Any] = {}
        with open(tmp_path / "a.stderr", "wb") as errors:
            a = subprocess.Popen(
                [sys.executable, "-m", "simul.cli.main", "server", "--backends", "isaac"],
                env=_server_env(tmp_path, kit), cwd=str(tmp_path),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors,
            )
        try:
            await rpc(a, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "a", "version": "0"}}})
            await rpc(a, {"jsonrpc": "2.0", "method": "notifications/initialized"}, reply=False)
            async with _stdio_server(tmp_path, kit, "b") as b:
                t0 = time.monotonic()
                timed_out = await rpc(a, {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
                    "name": "execute_isaac_script", "arguments": {"code": "sleep(6); print('a-slow')"}}})
                out["a"] = json.loads(timed_out["result"]["content"][-1]["text"])
                a.kill()
                a.wait()
                out["refused"] = await _call(b, "execute_isaac_script", code="print('b-marker')")
                out["refused_at"] = time.monotonic() - t0
                deadline = time.monotonic() + 15
                while list((tmp_path / "state" / "holds").glob("isaac-*.json")):
                    assert time.monotonic() < deadline, "B never ended the hold it took over"
                    await asyncio.sleep(0.1)
                out["released_at"] = time.monotonic() - t0
                out["after"] = await _call(b, "execute_isaac_script", code="print('b-after')")
        finally:
            if a.poll() is None:
                a.kill()
                a.wait()
            await kit.stop()
        out["a_pid"] = a.pid
        out["printed"] = kit.printed()
        return out

    out = asyncio.run(scenario())

    assert out["a"]["error_type"] == "TimeoutError"
    refused = out["refused"]
    assert refused["error_type"] == "InstanceBusy", refused
    assert refused["details"]["adopted_from"] == out["a_pid"]
    assert out["refused_at"] < 5.0, "refused before Kit had finished A's script"
    assert out["released_at"] >= 5.5, "held until Kit had finished A's script"
    assert out["after"]["success"] is True
    assert out["printed"] == ["a-slow", "b-after"], "B's marker never reached Kit"
