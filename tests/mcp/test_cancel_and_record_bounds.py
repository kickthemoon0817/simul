"""Cancelled calls say whether they ran; usage and audit rows stay small; logs rotate across processes.

Before:

* every cancelled call was recorded ``success: false, error: "cancelled"``,
  though the engine went on and ran it (an Isaac script finished, an Unreal
  delete landed 2 s later, a Blender script created its object), and Isaac's
  drain had the engine's answer in hand and dropped it;
* an unknown tool with a 200 KB name wrote a 200 KB audit row and usage row,
  a 100 KB ``agent_id`` rode along in every later row of its session, and a
  message repeating a 40 KB argument wrote a 140 KB usage row, so some 500
  calls rotated the whole usage history away; the audit log had no size cap;
* the audit log and simul.log rotated with Python's TimedRotatingFileHandler,
  which is not safe for several processes: after the first midnight every
  process but the one that rotated wrote into the previous day's file, and
  retention deleted those rows (795 of 956 lost in a two-process run);
* an exception from the audit or usage writer failed the call it described.
"""

from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pytest
from fastmcp import Client

from simul import logging as simul_logging
from simul.config import Settings
from simul.mcp import backends as backends_module
from simul.mcp import server as server_module
from simul.mcp.session_manager import SessionManager
from simul.mcp.usage_tracker import ToolUsageTracker
from tests.fakes import FakeFastMCP


@pytest.fixture
def audit_rows() -> Iterator[List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if hasattr(record, "audit"):
                rows.append(record.audit)

    audit_logger = logging.getLogger("simul.audit")
    handler = _Collect(level=logging.INFO)
    previous = audit_logger.level
    audit_logger.addHandler(handler)
    audit_logger.setLevel(logging.INFO)
    try:
        yield rows
    finally:
        audit_logger.removeHandler(handler)
        audit_logger.setLevel(previous)


def _server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> server_module.SimulMCPServer:
    monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
    monkeypatch.setattr(server_module, "TaskConfig", None)
    monkeypatch.setattr(backends_module, "is_headless_available", lambda: False)
    monkeypatch.setattr(backends_module, "is_blender_available", lambda: False)
    monkeypatch.setattr(backends_module, "UnrealRuntimeAdapter", None)
    instance = server_module.SimulMCPServer(settings=Settings())
    instance.usage_tracker = ToolUsageTracker(log_dir=tmp_path)
    return instance


def _cancel_inside_a_call(tool: str, start: Any) -> None:
    """Run ``start()`` inside the audit middleware's call scope, cancel it once it runs."""
    middleware = simul_logging.build_request_context_middleware()

    class _Context:
        message = type("M", (), {"name": tool})()

    async def run() -> None:
        started = asyncio.Event()

        async def call_next(context: Any) -> Any:
            return await start(started)

        task = asyncio.ensure_future(middleware.on_call_tool(_Context(), call_next))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())


def _row(instance: server_module.SimulMCPServer, tool: str) -> Dict[str, Any]:
    (row,) = instance.usage_tracker.get_recent(tool_name=tool)
    return row


# ---------------------------------------------------------------------------
# Major 1: a cancelled call says whether the engine ran it
# ---------------------------------------------------------------------------


def test_a_cancelled_isaac_call_keeps_the_engine_s_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, audit_rows: List[Dict[str, Any]]
) -> None:
    instance = _server(monkeypatch, tmp_path)

    async def start(started: asyncio.Event) -> Any:
        async def engine() -> Dict[str, Any]:
            started.set()
            await asyncio.sleep(0.2)
            return {"error": "Prim /World/X is not a mesh", "error_type": "ValidationError"}

        return await instance._exec_isaac("isaac_step", engine())

    _cancel_inside_a_call("isaac_step", start)

    row = _row(instance, "isaac_step")
    assert (row["success"], row["error"], row["cancelled"], row["executed"]) == (
        False,
        "Prim /World/X is not a mesh",
        True,
        True,
    )
    (audit,) = audit_rows
    assert audit["status"] == "cancelled" and audit["executed"] is True
    assert audit["error_type"] == "ValidationError"


def test_a_cancelled_isaac_call_that_lost_its_connection_ran_or_not(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    instance = _server(monkeypatch, tmp_path)

    async def start(started: asyncio.Event) -> Any:
        async def engine() -> Dict[str, Any]:
            started.set()
            await asyncio.sleep(0.05)
            raise ConnectionResetError(54, "reset")

        return await instance._exec_isaac("isaac_step", engine())

    _cancel_inside_a_call("isaac_step", start)

    row = _row(instance, "isaac_step")
    assert (row["success"], row["error"], row["executed"]) == (False, "cancelled", "unknown")


def test_a_cancelled_backend_call_is_marked_executed_unknown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, audit_rows: List[Dict[str, Any]]
) -> None:
    """Unreal's request was sent, and Blender's thread runs on: nothing says whether it ran."""
    instance = _server(monkeypatch, tmp_path)

    class _Adapter:
        def is_available(self) -> bool:
            return True

        def create_session(self) -> Any:
            import contextlib

            return contextlib.nullcontext(object())

    async def start(started: asyncio.Event) -> Any:
        async def delete(session: Any) -> Dict[str, Any]:
            started.set()
            await asyncio.sleep(10)
            return {"success": True}

        return await instance._exec_backend("delete_unreal_actor", _Adapter(), "Unreal", None, delete)

    _cancel_inside_a_call("delete_unreal_actor", start)

    row = _row(instance, "delete_unreal_actor")
    assert (row["success"], row["error"], row["cancelled"], row["executed"]) == (
        False,
        "cancelled",
        True,
        "unknown",
    )
    (audit,) = audit_rows
    assert audit["status"] == "cancelled" and audit["executed"] == "unknown"


def test_a_call_cancelled_before_it_started_did_not_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    instance = _server(monkeypatch, tmp_path)
    lock = instance._get_instance_lock(instance._get_effective_instance_name())

    async def run() -> None:
        await lock.acquire()

        async def never() -> Dict[str, Any]:
            return {"success": True}

        task = asyncio.ensure_future(instance._exec_isaac("isaac_tool", never()))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        lock.release()

    asyncio.run(run())

    row = _row(instance, "isaac_tool")
    assert (row["error"], row["executed"]) == ("cancelled_before_start", False)


# ---------------------------------------------------------------------------
# Major 2: rows are bounded
# ---------------------------------------------------------------------------


def _isaac_server(tmp_path: Path) -> server_module.SimulMCPServer:
    instance = server_module.SimulMCPServer(
        Settings(
            isaac_sim={"discovery_dir": str(tmp_path / "disc")},
            security={"allowed_paths": [str(tmp_path)], "rate_limiting_enabled": False},
        ),
        backends={"isaac"},
    )
    instance.session_manager = SessionManager(tmp_path / "sessions")
    instance.usage_tracker = ToolUsageTracker(log_dir=tmp_path / "usage")
    return instance


def test_huge_names_and_messages_write_small_rows(
    tmp_path: Path, audit_rows: List[Dict[str, Any]]
) -> None:
    instance = _isaac_server(tmp_path)

    async def run() -> List[Any]:
        async with Client(instance.mcp, mode="legacy") as client:
            return [
                await client.call_tool("x" * 200_000, {}, raise_on_error=False),
                await client.call_tool(
                    "claim_isaac_instance", {"purpose": "p", "agent_id": "a" * 100_000}, raise_on_error=False
                ),
                await client.call_tool(
                    "configure_isaac_physics_scene", {"solver_type": "Q" * 40_000}, raise_on_error=False
                ),
            ]

    unknown, claim, message = asyncio.run(run())

    assert json.loads(claim.content[-1].text)["error_type"] == "ValidationError"
    assert json.loads(message.content[-1].text)["error_type"] == "ValidationError"
    rows = (tmp_path / "usage" / "tool_usage.jsonl").read_text().splitlines()
    assert len(rows) == 3
    assert max(len(row) for row in rows) < 4096
    assert max(len(json.dumps(row)) for row in audit_rows) < 1024
    # The claim was refused at the schema, so no record carries the label.
    assert all("a" * 200 not in row for row in rows)


def test_the_record_bounds_every_field(tmp_path: Path) -> None:
    tracker = ToolUsageTracker(log_dir=tmp_path)

    tracker.record(
        "t" * 1000,
        1.0,
        False,
        params={"path": "p" * 10_000, "points": list(range(10_000)), **{f"k{i}": i for i in range(100)}},
        error="e" * 100_000,
        agent_id="a" * 10_000,
        session="s" * 10_000,
    )

    (line,) = (tmp_path / "tool_usage.jsonl").read_text().splitlines()
    assert len(line) < 6000
    row = json.loads(line)
    assert row["params"]["path"] == "<10000 characters>"
    assert row["params"]["points"].startswith("<list of 10000")
    assert row["params"]["..."].endswith("more parameters>")


# ---------------------------------------------------------------------------
# Minor: bookkeeping never fails the call
# ---------------------------------------------------------------------------


def test_a_failing_audit_or_usage_writer_does_not_fail_the_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    instance = _isaac_server(tmp_path)

    def broken_audit(**_fields: Any) -> None:
        raise RuntimeError("audit disk gone")

    def broken_log(_rec: Any) -> None:
        raise TypeError("not serialisable")

    monkeypatch.setattr(simul_logging, "emit_audit", broken_audit)
    monkeypatch.setattr(instance.usage_tracker, "_append_log", broken_log)

    async def run() -> Any:
        async with Client(instance.mcp, mode="legacy") as client:
            return await client.call_tool("list_isaac_instances", {"scan": False}, raise_on_error=False)

    result = asyncio.run(run())

    assert result.is_error is False
    assert json.loads(result.content[-1].text)["success"] is True


# ---------------------------------------------------------------------------
# Major 3: several processes share one log and its rotation
# ---------------------------------------------------------------------------


class _Day:
    """A ``SharedRotatingFileHandler`` whose 'today' moves on after half the records."""

    def __init__(self, handler: simul_logging.SharedRotatingFileHandler, start: date) -> None:
        self.handler = handler
        self.today = start
        handler._today = lambda: self.today  # type: ignore[method-assign]


def _writer(path: str, worker: int, records: int, start_ordinal: int) -> None:
    handler = simul_logging.SharedRotatingFileHandler(
        path, when="midnight", backupCount=0, maxBytes=4096, encoding="utf8"
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    clock = _Day(handler, date.fromordinal(start_ordinal))
    for index in range(records):
        if index == records // 2:
            # Midnight, for every process at about the same time.
            clock.today = date.fromordinal(start_ordinal + 1)
            time.sleep(0.01)
        handler.emit(logging.LogRecord("x", logging.INFO, "x.py", 1, f"w{worker}-r{index}", (), None))


@pytest.mark.skipif(sys.platform == "win32", reason="os.replace over an open file")
def test_processes_sharing_a_log_lose_no_record_across_rotations(tmp_path: Path) -> None:
    path = str(tmp_path / "audit.jsonl")
    workers, records = 4, 300
    start = date.today().toordinal()
    processes = [
        multiprocessing.Process(target=_writer, args=(path, worker, records, start))
        for worker in range(workers)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(60)
        assert process.exitcode == 0

    files = [p for p in tmp_path.iterdir() if p.name.startswith("audit.jsonl") and not p.name.endswith(".lock")]
    lines = [line for p in files for line in p.read_text().splitlines()]
    assert len(lines) == workers * records
    assert len(set(lines)) == workers * records
    assert len(files) > 2, "it rotated by size and by day"


def test_a_file_written_yesterday_is_rotated_to_its_day(tmp_path: Path) -> None:
    path = tmp_path / "simul.log"
    path.write_text("yesterday's record\n")
    yesterday = time.time() - 86400
    os.utime(path, (yesterday, yesterday))
    old = tmp_path / f"simul.log.{date.fromtimestamp(yesterday):%Y-%m-%d}"
    stale = tmp_path / f"simul.log.{date.today() - timedelta(days=30):%Y-%m-%d}"
    stale.write_text("a month old\n")
    handler = simul_logging.SharedRotatingFileHandler(str(path), when="midnight", backupCount=7)
    handler.setFormatter(logging.Formatter("%(message)s"))

    handler.emit(logging.LogRecord("x", logging.INFO, "x.py", 1, "today's record", (), None))

    assert old.read_text() == "yesterday's record\n"
    assert path.read_text() == "today's record\n"
    assert not stale.exists(), "copies past the retention are deleted"
    if sys.platform != "win32":
        assert oct(path.stat().st_mode & 0o777) == oct(0o600)


def test_the_audit_log_has_a_size_cap(tmp_path: Path) -> None:
    handler = simul_logging.SharedRotatingFileHandler(
        str(tmp_path / "audit.jsonl"), backupCount=30, maxBytes=1000, maxBackups=3
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    for index in range(200):
        handler.emit(logging.LogRecord("x", logging.INFO, "x.py", 1, f"row {index:04d} " + "x" * 40, (), None))

    files = sorted(p for p in tmp_path.iterdir() if not p.name.endswith(".lock"))
    assert len(files) == 4
    assert all(p.stat().st_size <= 1000 for p in files)
    assert simul_logging.AUDIT_MAX_BYTES and simul_logging.AUDIT_MAX_BACKUPS
