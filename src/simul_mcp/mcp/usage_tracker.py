"""
Tool usage tracker for Simul MCP Server.

Tracks per-tool call counts, success/failure rates, durations.
Keeps both in-memory stats and a persistent JSONL log file
so the CLI ``simul stats`` command can read historical data.
"""

from __future__ import annotations

import json
import logging
import os
import time

logger = logging.getLogger(__name__)
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional


_DEFAULT_LOG_DIR = Path.home() / ".simul" / "logs"


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
    """Single tool call record."""

    tool_name: str
    timestamp: float
    duration_ms: float
    success: bool
    params: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    agent_id: Optional[str] = None

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
        return out


class ToolUsageTracker:
    """
    Tool usage tracker with in-memory aggregation and JSONL persistence.

    The JSONL file at ``~/.simul/logs/tool_usage.jsonl`` is append-only
    and human-readable.  The CLI reads it to produce historical stats.
    """

    def __init__(
        self,
        max_recent: int = 200,
        log_dir: Optional[Path] = None,
    ) -> None:
        self._stats: Dict[str, ToolStats] = {}
        self._recent: Deque[CallRecord] = deque(maxlen=max_recent)

        self._log_dir = log_dir or _DEFAULT_LOG_DIR
        self._log_file = self._log_dir / "tool_usage.jsonl"
        self._log_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    def record(
        self,
        tool_name: str,
        duration_ms: float,
        success: bool,
        params: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> None:
        """Record a tool call (in-memory + JSONL).

        Args:
            tool_name: Registered tool name.
            duration_ms: Wall time of the call.
            success: Whether the call produced a non-error payload.
            params: Call parameters worth keeping in the log.
            error: Error text when the call failed.
            agent_id: Calling agent, so an operator can tell who ran what.
        """
        now = time.time()

        if tool_name not in self._stats:
            self._stats[tool_name] = ToolStats()

        stats = self._stats[tool_name]
        stats.total_calls += 1
        stats.total_duration_ms += duration_ms
        stats.last_called = now
        if success:
            stats.successes += 1
        else:
            stats.failures += 1

        rec = CallRecord(
            tool_name=tool_name,
            timestamp=now,
            duration_ms=duration_ms,
            success=success,
            params=params,
            error=error,
            agent_id=agent_id,
        )
        self._recent.append(rec)
        self._append_log(rec)

    def _append_log(self, rec: CallRecord) -> None:
        """Append a single record to the JSONL log file."""
        try:
            with open(
                os.open(self._log_file, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(json.dumps(rec.to_dict()) + "\n")
        except OSError as exc:
            logger.debug("Failed to write usage log: %s", exc)

    # ------------------------------------------------------------------
    # In-memory queries (MCP tool)
    # ------------------------------------------------------------------

    def get_stats(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get in-memory usage statistics.

        Args:
            tool_name: If provided, return stats for that tool only.

        Returns:
            Dict with per-tool stats sorted by total_calls descending.
        """
        if tool_name:
            stats = self._stats.get(tool_name)
            if not stats:
                return {"tool": tool_name, "stats": None}
            return {"tool": tool_name, "stats": stats.to_dict()}

        ranked = sorted(
            self._stats.items(),
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
    ) -> List[Dict[str, Any]]:
        """Get recent call log from memory, newest first."""
        records = list(self._recent)
        if tool_name:
            records = [r for r in records if r.tool_name == tool_name]
        records.reverse()
        return [r.to_dict() for r in records[:limit]]

    def reset(self) -> None:
        """Clear in-memory data and truncate the JSONL log."""
        self._stats.clear()
        self._recent.clear()
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
        log_file = (log_dir or _DEFAULT_LOG_DIR) / "tool_usage.jsonl"
        if not log_file.exists():
            return {"total_tools_used": 0, "total_calls": 0, "tools": {}}

        stats: Dict[str, ToolStats] = {}
        recent: List[Dict[str, Any]] = []

        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

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
            recent.reverse()
            result["recent"] = recent[:limit]
        return result
