"""The audit log tells how each call ended; log level, format and file modes do what they say.

* Every audit row said ``"status": "ok"`` unless an exception escaped: a
  timeout, an InstanceBusy refusal, a reset connection were all "ok", and no
  row named the agent, the session or the error type.
* ``LOGGING__LEVEL`` and ``--log-level`` never reached the YAML's loggers
  (ERROR still wrote INFO lines, DEBUG wrote none) and ``LOGGING__FORMAT`` did
  nothing.
* Log directories and files were created with the umask's modes (0755/0644),
  so any local user could read the tool calls and their arguments.
"""

from __future__ import annotations

import asyncio
import json
import logging
import logging.handlers
import os
import stat
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pytest
from fastmcp import Client

from simul import logging as simul_logging
from simul.config import Settings
from simul.mcp import server as server_module
from simul.mcp.usage_tracker import ToolUsageTracker

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@pytest.fixture
def audit_rows() -> Iterator[List[Dict[str, Any]]]:
    """Collect the audit rows emitted while the test runs."""
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


# ---------------------------------------------------------------------------
# M7: the audit row is the envelope's verdict
# ---------------------------------------------------------------------------


def test_audit_rows_record_the_real_outcome(
    tmp_path: Path, audit_rows: List[Dict[str, Any]]
) -> None:
    (tmp_path / "scene.usda").write_text('#usda 1.0\ndef Xform "World" {}\n')
    instance = server_module.SimulMCPServer(
        Settings(security={"allowed_paths": [str(tmp_path)], "burst_size": 3}), backends={"usd"}
    )
    instance.usage_tracker = ToolUsageTracker(log_dir=tmp_path / "usage")

    async def run() -> None:
        async with Client(instance.mcp, mode="legacy") as client:
            for arguments in (
                {"file_path": str(tmp_path / "scene.usda")},
                {"file_path": "/etc/passwd"},
                {"file_path": 123},
                {"file_path": str(tmp_path / "scene.usda")},
            ):
                await client.call_tool("load_usd_file", arguments, raise_on_error=False)
            await client.call_tool("no_such_tool", {}, raise_on_error=False)

    asyncio.run(run())

    summary = [
        (row["tool"], row["status"], row.get("error_type"), row.get("reason")) for row in audit_rows
    ]
    assert summary == [
        ("load_usd_file", "ok", None, None),
        ("load_usd_file", "error", "SandboxError", None),
        ("load_usd_file", "error", "ValidationError", None),
        ("load_usd_file", "error", "RateLimitError", "rate_limited"),
        # Named as the usage row names it, the name given kept aside (round-2 m-9).
        ("<unknown tool>", "error", "NotFoundError", None),
    ]
    assert audit_rows[-1]["extra"] == {"requested_tool": "no_such_tool"}
    assert all("extra" not in row for row in audit_rows[:4])
    tool_rows = audit_rows[:4]
    assert {row["session"] for row in tool_rows} == {tool_rows[0]["agent_id"]}
    assert tool_rows[0]["session"].startswith("session-")
    usage = [
        json.loads(line)
        for line in (tmp_path / "usage" / "tool_usage.jsonl").read_text().splitlines()
    ]
    # One id per call ties its usage row to its audit row; the unknown tool's
    # call is recorded too, under one name for every unknown tool.
    assert [row["request_id"] for row in usage] == [row["request_id"] for row in audit_rows]
    assert usage[-1]["tool"] == "<unknown tool>"


def test_a_cancelled_call_is_audited_as_cancelled(audit_rows: List[Dict[str, Any]]) -> None:
    middleware = simul_logging.build_request_context_middleware()

    class _Context:
        message = type("M", (), {"name": "slow_tool"})()

    async def call_next(context: Any) -> Any:
        await asyncio.sleep(10)

    async def run() -> None:
        task = asyncio.ensure_future(middleware.on_call_tool(_Context(), call_next))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())

    (row,) = audit_rows
    assert row["status"] == "cancelled"


def test_the_outcome_is_reported_only_inside_a_call() -> None:
    assert simul_logging.current_request_id() is None
    simul_logging.note_call_outcome(status="error")  # a no-op, not an error


# ---------------------------------------------------------------------------
# M8: logging.level and logging.format apply
# ---------------------------------------------------------------------------


@pytest.fixture
def restore_logging() -> Iterator[None]:
    """Stop the audit listener setup_logging starts (the conftest restores the loggers)."""
    try:
        yield
    finally:
        simul_logging._stop_audit_listener()


def _settings(tmp_path: Path, **logging_fields: Any) -> Settings:
    return Settings(
        logging={
            "file_path": str(tmp_path / "logs" / "simul.log"),
            "audit_path": str(tmp_path / "logs" / "audit.jsonl"),
            "console_enabled": False,
            **logging_fields,
        }
    )


def _flush() -> None:
    for name in ("", "simul"):
        for handler in logging.getLogger(name).handlers:
            handler.flush()


@pytest.mark.parametrize(
    ("level", "debug", "info", "error"),
    [("ERROR", False, False, True), ("INFO", False, True, True), ("DEBUG", True, True, True)],
)
def test_the_log_level_setting_reaches_simuls_loggers(
    tmp_path: Path, restore_logging: None, level: str, debug: bool, info: bool, error: bool
) -> None:
    simul_logging.setup_logging(_settings(tmp_path, level=level))
    probe = logging.getLogger("simul.probe")
    probe.debug("probe-debug")
    probe.info("probe-info")
    probe.error("probe-error")
    _flush()

    text = (tmp_path / "logs" / "simul.log").read_text()
    assert ("probe-debug" in text, "probe-info" in text, "probe-error" in text) == (debug, info, error)


def test_debug_for_simul_leaves_libraries_quiet(tmp_path: Path, restore_logging: None) -> None:
    simul_logging.setup_logging(_settings(tmp_path, level="DEBUG"))

    assert logging.getLogger("simul").getEffectiveLevel() == logging.DEBUG
    assert logging.getLogger("fastmcp").getEffectiveLevel() == logging.WARNING


def test_the_json_format_writes_one_object_per_line(tmp_path: Path, restore_logging: None) -> None:
    simul_logging.setup_logging(_settings(tmp_path, format="json"))
    logging.getLogger("simul.probe").warning("probe-json")
    _flush()

    lines = [json.loads(line) for line in (tmp_path / "logs" / "simul.log").read_text().splitlines()]
    probe = next(line for line in lines if line["msg"] == "probe-json")
    assert probe["level"] == "WARNING" and probe["logger"] == "simul.probe"


def test_an_unknown_format_is_reported(
    tmp_path: Path, restore_logging: None, capsys: pytest.CaptureFixture[str]
) -> None:
    simul_logging.setup_logging(_settings(tmp_path, format="xml"))

    assert "unknown logging.format 'xml'" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Log files are private to the user
# ---------------------------------------------------------------------------


@posix_only
def test_log_directories_are_0700_and_files_0600(tmp_path: Path, restore_logging: None) -> None:
    simul_logging.setup_logging(_settings(tmp_path / "deep" / "state", structured_enabled=True))
    logging.getLogger("simul.probe").error("probe")
    simul_logging.emit_audit(tool="t", request_id="r", duration_ms=1.0, status="ok")
    _flush()
    simul_logging._stop_audit_listener()

    logs = tmp_path / "deep" / "state" / "logs"
    for directory in (tmp_path / "deep", tmp_path / "deep" / "state", logs):
        assert _mode(directory) == 0o700, directory
    files = sorted(p for p in logs.iterdir() if p.is_file())
    # Each file is written by every server process under its own lock file
    # (created 0600 too); a file nothing has written yet does not exist.
    assert {p.name for p in files} >= {"simul.log", "simul.json", "audit.jsonl", "simul.log.lock"}
    assert {p.name: _mode(p) for p in files} == {p.name: 0o600 for p in files}


@posix_only
def test_a_rotated_log_file_stays_0600(tmp_path: Path) -> None:
    handler = simul_logging.PrivateTimedRotatingFileHandler(
        str(tmp_path / "simul.log"), when="midnight", backupCount=3, encoding="utf8"
    )
    try:
        handler.emit(logging.LogRecord("x", logging.INFO, "x.py", 1, "before", (), None))
        handler.doRollover()
        handler.emit(logging.LogRecord("x", logging.INFO, "x.py", 1, "after", (), None))
    finally:
        handler.close()

    files = sorted(tmp_path.iterdir())
    assert len(files) == 2
    assert all(_mode(path) == 0o600 for path in files)


@posix_only
def test_existing_simul_files_are_tightened_but_existing_directories_are_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore_logging: None
) -> None:
    """A directory simul did not create keeps its mode: SIMUL_HOME may be a shared one (the home)."""
    state = tmp_path / "state"
    (state / "logs").mkdir(parents=True)
    os.chmod(state / "logs", 0o755)
    (state / "logs" / "audit.jsonl").write_text("")
    os.chmod(state / "logs" / "audit.jsonl", 0o644)
    shared = tmp_path / "shared"
    shared.mkdir()
    os.chmod(shared, 0o755)
    monkeypatch.setenv("SIMUL_HOME", str(state))

    simul_logging.setup_logging(
        Settings(
            logging={
                "file_path": str(shared / "simul.log"),
                "audit_path": str(state / "logs" / "audit.jsonl"),
                "console_enabled": False,
            }
        )
    )
    simul_logging._stop_audit_listener()

    assert _mode(state / "logs") == 0o755, "an existing directory keeps its mode"
    assert _mode(state / "logs" / "audit.jsonl") == 0o600
    assert _mode(shared) == 0o755, "a directory outside the state root keeps its mode"
    assert _mode(shared / "simul.log") == 0o600


@posix_only
def test_an_old_world_readable_usage_log_is_tightened(tmp_path: Path) -> None:
    for name in ("tool_usage.jsonl", "tool_usage.jsonl.1"):
        (tmp_path / name).write_text("{}\n")
        os.chmod(tmp_path / name, 0o644)

    ToolUsageTracker(log_dir=tmp_path).record("tool", 1.0, True)

    assert _mode(tmp_path / "tool_usage.jsonl") == 0o600
    assert _mode(tmp_path / "tool_usage.jsonl.1") == 0o600


# ---------------------------------------------------------------------------
# m7: the usage log rotates, and reads stream it
# ---------------------------------------------------------------------------


def test_the_usage_log_rotates_and_keeps_its_backups(tmp_path: Path) -> None:
    tracker = ToolUsageTracker(log_dir=tmp_path, max_bytes=2_000, backups=3)
    for index in range(200):
        tracker.record("tool", 1.0, index % 2 == 0, params={"i": index})

    names = sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("tool_usage.jsonl"))
    assert names == ["tool_usage.jsonl", "tool_usage.jsonl.1", "tool_usage.jsonl.2", "tool_usage.jsonl.3"]
    assert all(p.stat().st_size < 2_500 for p in tmp_path.glob("tool_usage.jsonl*"))
    if sys.platform != "win32":
        assert all(_mode(p) == 0o600 for p in tmp_path.glob("tool_usage.jsonl*"))

    loaded = ToolUsageTracker.load_from_log(log_dir=tmp_path, limit=3)
    kept = sum(1 for p in tmp_path.glob("tool_usage.jsonl*") for _ in p.read_text().splitlines())
    assert loaded["total_calls"] == kept, "the rotated copies are read too"
    assert [row["params"]["i"] for row in loaded["recent"]] == [199, 198, 197]

    tracker.reset()
    assert sorted(p.name for p in tmp_path.glob("tool_usage.jsonl*")) == ["tool_usage.jsonl"]


def test_reading_recent_records_keeps_only_those(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``simul stats --recent 5`` held every record of the log in memory (1.38 GB for 1M rows)."""
    log = tmp_path / "tool_usage.jsonl"
    with open(log, "w") as handle:
        for index in range(20_000):
            handle.write(json.dumps({"tool": "t", "success": True, "duration_ms": 1, "timestamp": index}) + "\n")
    appended: List[int] = []
    original = __import__("collections").deque

    class _Watched(original):  # type: ignore[misc, valid-type]
        def append(self, item: Any) -> None:
            super().append(item)
            appended.append(len(self))

    monkeypatch.setattr("simul.mcp.usage_tracker.deque", _Watched)
    started = time.monotonic()
    loaded = ToolUsageTracker.load_from_log(log_dir=tmp_path, limit=5)

    assert loaded["total_calls"] == 20_000
    assert [row["timestamp"] for row in loaded["recent"]] == [19_999, 19_998, 19_997, 19_996, 19_995]
    assert max(appended) == 5
    assert time.monotonic() - started < 5
