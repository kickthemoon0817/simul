"""
Tool usage tracker for the Simul MCP server.

Tracks per-tool call counts, success/failure rates, durations.
Keeps both in-memory stats and a persistent JSONL log file
so the CLI ``simul stats`` command can read historical data.

The in-memory view is kept per MCP session too, because the agent-facing
``get_tool_usage_stats`` shows a caller its own session's calls: the records
carry parameters, file paths and agent labels of whoever made them. The JSONL
log, which only the operator reads, keeps every record with its session
and the request id its audit row carries.

The log is rotated by size (``tool_usage.jsonl.1`` ... ``.N``, oldest
dropped), so it no longer grows without bound, and reads stream it: the CLI
keeps only the records it will show. Every field of a row is bounded
(``_bounded_text``, ``_bounded_params``), so one call with a huge argument
cannot write a huge row and rotate the history away.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Iterator, List, Optional, Union

from ..utils.filelock import exclusive_file_lock
from ..utils.paths import state_dir

logger = logging.getLogger(__name__)

# Under $SIMUL_HOME (default ~/.simul); created on the first write, not here.
_DEFAULT_LOG_DIR = state_dir() / "logs"
#: Recent records kept in memory (and the largest ``limit`` the stats tool takes).
MAX_RECENT = 200
#: Sessions whose per-session stats are kept; the least recently active is
#: dropped first. The server also drops a session's stats when it ends.
MAX_SESSIONS = 1024
#: Size at which ``tool_usage.jsonl`` is rotated, and the rotated copies kept
#: (about 50 000 records each, so ``simul stats`` reads at most ~300 000).
MAX_LOG_BYTES = 10 * 1024 * 1024
LOG_BACKUPS = 5
_LOG_NAME = "tool_usage.jsonl"
#: Longest tool name, agent id or session label a row keeps, the longest
#: error message, and the longest string parameter; longer ones are cut or
#: shown by their size.
MAX_NAME_CHARS = 128
MAX_ERROR_CHARS = 2000
MAX_PARAM_CHARS = 256
# Most parameters a row keeps, and the JSON size a list or mapping may take.
_MAX_PARAMS = 32
_MAX_PARAM_JSON = 1024


def _bounded_text(value: Optional[str], limit: int) -> Optional[str]:
    """``value`` cut to ``limit`` characters, saying how much was left out."""
    if value is None or len(value) <= limit:
        return value
    return f"{value[:limit]}... <{len(value) - limit} more characters>"


def _bounded_param(value: Any) -> Any:
    """A parameter as a row keeps it: a long string by its size, a large list or mapping by its length."""
    if isinstance(value, str):
        return value if len(value) <= MAX_PARAM_CHARS else f"<{len(value)} characters>"
    if isinstance(value, (list, tuple, dict)):
        try:
            size = len(json.dumps(value, default=str))
        except (TypeError, ValueError):
            size = _MAX_PARAM_JSON + 1
        if size > _MAX_PARAM_JSON:
            return f"<{type(value).__name__} of {len(value)}, {size} bytes>"
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _bounded_text(str(value), MAX_PARAM_CHARS)


def _bounded_params(params: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The parameters a row keeps: each bounded, at most ``_MAX_PARAMS`` of them."""
    if not params:
        return params
    items = list(params.items())
    kept = {
        str(_bounded_text(str(key), MAX_NAME_CHARS)): _bounded_param(value)
        for key, value in items[:_MAX_PARAMS]
    }
    if len(items) > _MAX_PARAMS:
        kept["..."] = f"<{len(items) - _MAX_PARAMS} more parameters>"
    return kept


@dataclass
class ToolStats:
    """Aggregated statistics for a single tool."""

    total_calls: int = 0
    successes: int = 0
    failures: int = 0
    total_duration_ms: float = 0.0
    last_called: float = 0.0

    @property
    def avg_duration_ms(self) -> float:
        return self.total_duration_ms / self.total_calls if self.total_calls else 0.0

    def add(self, duration_ms: float, success: bool, now: float) -> None:
        """Count one call."""
        self.total_calls += 1
        self.total_duration_ms += duration_ms
        self.last_called = now
        if success:
            self.successes += 1
        else:
            self.failures += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "successes": self.successes,
            "failures": self.failures,
            "avg_duration_ms": round(self.avg_duration_ms, 2),
            "total_duration_ms": round(self.total_duration_ms, 2),
            "last_called": self.last_called,
        }


@dataclass
class CallRecord:
    """Single tool call record.

    ``session`` is the published label of the MCP session the call came in on
    (never the transport's session id). It scopes what the stats tool shows
    and is written to the JSONL log, but left out of ``to_dict``, the
    agent-facing form, whose records are the caller's own anyway.
    """

    tool_name: str
    timestamp: float
    duration_ms: float
    success: bool
    params: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    agent_id: Optional[str] = None
    script_sha256: Optional[str] = None
    session: Optional[str] = None
    request_id: Optional[str] = None
    cancelled: bool = False
    executed: Optional[Union[bool, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "tool": self.tool_name,
            "timestamp": self.timestamp,
            "duration_ms": round(self.duration_ms, 2),
            "success": self.success,
        }
        if self.params:
            out["params"] = self.params
        if self.error:
            out["error"] = self.error
        if self.agent_id:
            out["agent_id"] = self.agent_id
        if self.script_sha256:
            out["script_sha256"] = self.script_sha256
        if self.cancelled:
            # The client gave up on it; whether the engine ran it anyway.
            out["cancelled"] = True
            out["executed"] = "unknown" if self.executed is None else self.executed
        return out

    def to_log_dict(self) -> Dict[str, Any]:
        """The JSONL row: ``to_dict`` plus the session and the request id."""
        out = self.to_dict()
        if self.session:
            out["session"] = self.session
        if self.request_id:
            out["request_id"] = self.request_id
        return out


class ToolUsageTracker:
    """
    Tool usage tracker with in-memory aggregation and JSONL persistence.

    The JSONL file at ``~/.simul/logs/tool_usage.jsonl`` is append-only
    and human-readable.  The CLI reads it to produce historical stats.
    """

    def __init__(
        self,
        max_recent: int = MAX_RECENT,
        log_dir: Optional[Path] = None,
        max_bytes: int = MAX_LOG_BYTES,
        backups: int = LOG_BACKUPS,
    ) -> None:
        self._max_bytes = max_bytes
        self._backups = max(1, backups)
        self._stats: Dict[str, ToolStats] = {}
        self._session_stats: "OrderedDict[str, Dict[str, ToolStats]]" = OrderedDict()
        self._recent: Deque[CallRecord] = deque(maxlen=max_recent)

        self._log_dir = log_dir or _DEFAULT_LOG_DIR
        self._log_file = self._log_dir / _LOG_NAME
        self._tightened = False

    def record(
        self,
        tool_name: str,
        duration_ms: float,
        success: bool,
        params: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        agent_id: Optional[str] = None,
        script_sha256: Optional[str] = None,
        session: Optional[str] = None,
        request_id: Optional[str] = None,
        cancelled: bool = False,
        executed: Optional[Union[bool, str]] = None,
    ) -> None:
        """Record a tool call (in-memory + JSONL); never raises.

        Args:
            tool_name: Registered tool name.
            duration_ms: Wall-clock duration of the call.
            success: Whether the call succeeded.
            params: Call parameters worth keeping in the log.
            error: Error message for a failed call.
            agent_id: Who made the call, as resolved from the MCP session, so
                an operator can attribute a damaged scene to one agent.
            script_sha256: Hex digest of the source for script-execution
                tools, so identical scripts can be matched across sessions
                without storing the code itself.
            session: Published label of the MCP session the call came in on,
                which scopes the agent-facing stats.
            request_id: The call's correlation id, which its audit row and
                log lines carry too.
            cancelled: The client cancelled the call.
            executed: For a cancelled call, whether the engine ran the
                operation anyway: True, False, or ``"unknown"``. A cancelled
                call that ran keeps the engine's ``success`` and ``error``.
        """
        try:
            self._record(
                tool_name,
                duration_ms,
                success,
                params=params,
                error=error,
                agent_id=agent_id,
                script_sha256=script_sha256,
                session=session,
                request_id=request_id,
                cancelled=cancelled,
                executed=executed,
            )
        except Exception as exc:  # pragma: no cover - defensive
            # Bookkeeping never fails the call it describes.
            logger.warning("Could not record the %s call: %s", tool_name, exc)

    def _record(
        self,
        tool_name: str,
        duration_ms: float,
        success: bool,
        *,
        params: Optional[Dict[str, Any]],
        error: Optional[str],
        agent_id: Optional[str],
        script_sha256: Optional[str],
        session: Optional[str],
        request_id: Optional[str],
        cancelled: bool,
        executed: Optional[Union[bool, str]],
    ) -> None:
        """``record``, which may raise."""
        now = time.time()
        tool_name = _bounded_text(tool_name, MAX_NAME_CHARS) or tool_name
        agent_id = _bounded_text(agent_id, MAX_NAME_CHARS)
        session = _bounded_text(session, MAX_NAME_CHARS)
        error = _bounded_text(error, MAX_ERROR_CHARS)
        params = _bounded_params(params)

        self._stats.setdefault(tool_name, ToolStats()).add(duration_ms, success, now)
        if session:
            per_tool = self._session_stats.get(session)
            if per_tool is None:
                per_tool = self._session_stats[session] = {}
                while len(self._session_stats) > MAX_SESSIONS:
                    self._session_stats.popitem(last=False)
            else:
                self._session_stats.move_to_end(session)
            per_tool.setdefault(tool_name, ToolStats()).add(duration_ms, success, now)

        rec = CallRecord(
            tool_name=tool_name,
            timestamp=now,
            duration_ms=duration_ms,
            success=success,
            params=params,
            error=error,
            agent_id=agent_id,
            script_sha256=script_sha256,
            session=session,
            request_id=request_id,
            cancelled=cancelled,
            executed=executed,
        )
        self._recent.append(rec)
        self._append_log(rec)

    def _append_log(self, rec: CallRecord) -> None:
        """Append a single record to the JSONL log file, rotating it when it is full.

        The directory is created (0700) on the first write, and the file 0600.
        """
        try:
            from ..logging import make_private_dir, tighten_log_files

            make_private_dir(self._log_dir)
            if not self._tightened:
                # A log an earlier version wrote world-readable, and its
                # rotated copies, are simul's own: make them 0600 once.
                tighten_log_files(self._log_file)
                self._tightened = True
            self._rotate_if_full()
            line = json.dumps(rec.to_log_dict(), default=str) + "\n"
            with open(
                os.open(self._log_file, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(line)
        except Exception as exc:
            # A full disk, an unwritable directory or an unencodable value:
            # the call it describes still stands.
            logger.debug("Failed to write usage log: %s", exc)

    def _rotate_if_full(self) -> None:
        """Shift ``tool_usage.jsonl`` to ``.1`` (``.1`` to ``.2`` ...) once it reaches ``max_bytes``.

        Every server on the machine appends to the same file, so the shift
        holds a lock and re-checks the size under it: two servers that both
        see a full file rotate it once.
        """
        try:
            if self._log_file.stat().st_size < self._max_bytes:
                return
        except OSError:
            return
        try:
            with exclusive_file_lock(self._log_dir / "tool_usage.lock", timeout=2.0):
                try:
                    if self._log_file.stat().st_size < self._max_bytes:
                        return
                except FileNotFoundError:
                    return
                for index in range(self._backups - 1, 0, -1):
                    older = _rotated(self._log_file, index)
                    if older.exists():
                        os.replace(older, _rotated(self._log_file, index + 1))
                os.replace(self._log_file, _rotated(self._log_file, 1))
        except (OSError, TimeoutError) as exc:
            logger.debug("Could not rotate the usage log: %s", exc)

    def forget_session(self, session: str) -> None:
        """Drop the per-session stats of an MCP session that has ended."""
        self._session_stats.pop(session, None)

    # ------------------------------------------------------------------
    # In-memory queries (MCP tool)
    # ------------------------------------------------------------------

    def get_stats(
        self, tool_name: Optional[str] = None, session: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get in-memory usage statistics.

        Args:
            tool_name: If provided, return stats for that tool only.
            session: If provided, count only that session's calls; otherwise
                every call this server has served.

        Returns:
            Dict with per-tool stats sorted by total_calls descending.
        """
        source = self._stats if session is None else self._session_stats.get(session, {})
        if tool_name:
            stats = source.get(tool_name)
            if not stats:
                return {"tool": tool_name, "stats": None}
            return {"tool": tool_name, "stats": stats.to_dict()}

        ranked = sorted(
            source.items(),
            key=lambda kv: kv[1].total_calls,
            reverse=True,
        )
        return {
            "total_tools_used": len(ranked),
            "total_calls": sum(s.total_calls for _, s in ranked),
            "tools": {name: s.to_dict() for name, s in ranked},
        }

    def get_recent(
        self,
        limit: int = 50,
        tool_name: Optional[str] = None,
        session: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get recent call log from memory, newest first.

        Args:
            limit: Most records to return; a value below 1 returns none.
            tool_name: Only this tool's records.
            session: Only this session's records.

        Returns:
            The records, newest first.
        """
        if limit < 1:
            return []
        records = list(self._recent)
        if tool_name:
            records = [r for r in records if r.tool_name == tool_name]
        if session is not None:
            records = [r for r in records if r.session == session]
        records.reverse()
        return [r.to_dict() for r in records[:limit]]

    def reset(self) -> None:
        """Clear in-memory data and empty the JSONL log, rotated copies included."""
        self._stats.clear()
        self._session_stats.clear()
        self._recent.clear()
        for rotated in _log_files(self._log_file):
            if rotated == self._log_file:
                continue
            try:
                rotated.unlink()
            except OSError as exc:
                logger.debug("Failed to remove %s: %s", rotated, exc)
        if not self._log_file.exists():
            return
        try:
            self._log_file.write_text("", encoding="utf-8")
        except OSError as exc:
            logger.debug("Failed to truncate usage log: %s", exc)

    # ------------------------------------------------------------------
    # Persistent log queries (CLI)
    # ------------------------------------------------------------------

    @staticmethod
    def load_from_log(
        log_dir: Optional[Path] = None,
        tool_name: Optional[str] = None,
        limit: int = 0,
    ) -> Dict[str, Any]:
        """
        Build aggregated stats from the persistent JSONL log.

        This is a static method so the CLI can call it without a running
        server instance.

        Args:
            log_dir: Override log directory (default ~/.simul/logs).
            tool_name: Filter to a single tool.
            limit: If > 0, also return the last N records.

        Returns:
            Dict with per-tool stats and optional recent records.
        """
        log_file = (log_dir or _DEFAULT_LOG_DIR) / _LOG_NAME
        files = _log_files(log_file)
        if not files:
            return {"total_tools_used": 0, "total_calls": 0, "tools": {}}

        stats: Dict[str, ToolStats] = {}
        # Only the newest `limit` records are kept while the files stream past.
        recent: Deque[Dict[str, Any]] = deque(maxlen=max(limit, 0) or None)

        for rec in _read_records(files):
            name = rec.get("tool", "")
            if tool_name and name != tool_name:
                continue

            if name not in stats:
                stats[name] = ToolStats()

            s = stats[name]
            s.total_calls += 1
            s.total_duration_ms += rec.get("duration_ms", 0.0)
            s.last_called = max(s.last_called, rec.get("timestamp", 0.0))
            if rec.get("success"):
                s.successes += 1
            else:
                s.failures += 1

            if limit > 0:
                recent.append(rec)

        ranked = sorted(stats.items(), key=lambda kv: kv[1].total_calls, reverse=True)
        result: Dict[str, Any] = {
            "total_tools_used": len(ranked),
            "total_calls": sum(s.total_calls for _, s in ranked),
            "tools": {name: s.to_dict() for name, s in ranked},
        }
        if limit > 0:
            result["recent"] = list(reversed(recent))
        return result


def _rotated(log_file: Path, index: int) -> Path:
    """The ``index``-th rotated copy of ``log_file`` (``tool_usage.jsonl.1`` is the newest)."""
    return log_file.with_name(f"{log_file.name}.{index}")


def _log_files(log_file: Path) -> List[Path]:
    """The rotated copies, oldest first, then the current file; only those that exist."""
    rotated = []
    try:
        for candidate in log_file.parent.glob(f"{log_file.name}.*"):
            suffix = candidate.name[len(log_file.name) + 1:]
            if suffix.isdigit():
                rotated.append((int(suffix), candidate))
    except OSError:
        pass
    files = [candidate for _, candidate in sorted(rotated, reverse=True)]
    if log_file.exists():
        files.append(log_file)
    return files


def _read_records(files: List[Path]) -> Iterator[Dict[str, Any]]:
    """Yield each JSON record of ``files`` in order, skipping blank and damaged lines."""
    for path in files:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(record, dict):
                        yield record
        except OSError as exc:
            logger.debug("Could not read %s: %s", path, exc)
