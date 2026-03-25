"""Unit tests for ToolUsageTracker.

Covers: record(), get_stats(), get_recent(), reset(), load_from_log(),
ToolStats.avg_duration_ms, CallRecord.to_dict(), and OSError suppression paths.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_path))

from simul_mcp.mcp.usage_tracker import CallRecord, ToolStats, ToolUsageTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tracker(max_recent: int = 200) -> tuple[ToolUsageTracker, Path]:
    """Return a tracker wired to an isolated temp directory."""
    d = Path(tempfile.mkdtemp())
    return ToolUsageTracker(log_dir=d, max_recent=max_recent), d


# ---------------------------------------------------------------------------
# ToolStats
# ---------------------------------------------------------------------------


class TestToolStats:
    """ToolStats dataclass contract."""

    def test_avg_duration_ms_returns_zero_when_no_calls(self) -> None:
        """avg_duration_ms does not raise ZeroDivisionError on fresh instance."""
        ts = ToolStats()
        assert ts.avg_duration_ms == 0.0

    def test_avg_duration_ms_returns_mean_of_recorded_durations(self) -> None:
        """avg_duration_ms is total_duration / total_calls."""
        ts = ToolStats(total_calls=3, total_duration_ms=300.0)
        assert ts.avg_duration_ms == 100.0

    def test_to_dict_includes_all_required_keys(self) -> None:
        """to_dict returns the six expected summary keys."""
        ts = ToolStats(total_calls=1, successes=1, failures=0, total_duration_ms=50.0)
        d = ts.to_dict()
        for key in ("total_calls", "successes", "failures", "avg_duration_ms", "total_duration_ms", "last_called"):
            assert key in d, f"Missing key: {key}"

    def test_to_dict_rounds_durations_to_two_decimal_places(self) -> None:
        """Floating point values in to_dict are rounded to 2 dp."""
        ts = ToolStats(total_calls=3, total_duration_ms=100.0)
        d = ts.to_dict()
        # avg is 33.333... - must be 33.33
        assert d["avg_duration_ms"] == round(100.0 / 3, 2)


# ---------------------------------------------------------------------------
# CallRecord
# ---------------------------------------------------------------------------


class TestCallRecord:
    """CallRecord.to_dict serialisation contract."""

    def test_to_dict_omits_params_when_none(self) -> None:
        """Absent params are not included in serialised output."""
        rec = CallRecord(tool_name="t", timestamp=1.0, duration_ms=10.0, success=True)
        d = rec.to_dict()
        assert "params" not in d

    def test_to_dict_omits_error_when_none(self) -> None:
        """Absent error is not included in serialised output."""
        rec = CallRecord(tool_name="t", timestamp=1.0, duration_ms=10.0, success=True)
        d = rec.to_dict()
        assert "error" not in d

    def test_to_dict_includes_params_when_provided(self) -> None:
        """params dict is serialised when present."""
        rec = CallRecord(
            tool_name="t", timestamp=1.0, duration_ms=10.0, success=True,
            params={"code_bytes": 42},
        )
        assert rec.to_dict()["params"] == {"code_bytes": 42}

    def test_to_dict_includes_error_when_provided(self) -> None:
        """error string is serialised when present."""
        rec = CallRecord(
            tool_name="t", timestamp=1.0, duration_ms=10.0, success=False,
            error="TimeoutError",
        )
        assert rec.to_dict()["error"] == "TimeoutError"

    def test_to_dict_rounds_duration_to_two_decimal_places(self) -> None:
        """duration_ms is rounded to 2 dp in serialised form."""
        rec = CallRecord(tool_name="t", timestamp=1.0, duration_ms=10.3333, success=True)
        assert rec.to_dict()["duration_ms"] == round(10.3333, 2)


# ---------------------------------------------------------------------------
# ToolUsageTracker.record()
# ---------------------------------------------------------------------------


class TestRecord:
    """record() in-memory accounting."""

    def test_record_increments_total_calls(self) -> None:
        """Each record() call increments total_calls by one."""
        tracker, _ = _tracker()
        tracker.record("ping", 10.0, True)
        tracker.record("ping", 20.0, False)
        assert tracker.get_stats()["tools"]["ping"]["total_calls"] == 2

    def test_record_increments_successes_on_success(self) -> None:
        """success=True increments the successes counter."""
        tracker, _ = _tracker()
        tracker.record("ping", 10.0, True)
        assert tracker.get_stats()["tools"]["ping"]["successes"] == 1

    def test_record_increments_failures_on_failure(self) -> None:
        """success=False increments the failures counter."""
        tracker, _ = _tracker()
        tracker.record("ping", 10.0, False, error="boom")
        assert tracker.get_stats()["tools"]["ping"]["failures"] == 1

    def test_record_accumulates_duration_correctly(self) -> None:
        """avg_duration_ms reflects the true mean of recorded durations."""
        tracker, _ = _tracker()
        tracker.record("ping", 100.0, True)
        tracker.record("ping", 200.0, True)
        avg = tracker.get_stats()["tools"]["ping"]["avg_duration_ms"]
        assert abs(avg - 150.0) < 0.01

    def test_record_tracks_multiple_different_tools(self) -> None:
        """Independent tools are tracked independently."""
        tracker, _ = _tracker()
        tracker.record("tool_a", 10.0, True)
        tracker.record("tool_b", 20.0, False)
        s = tracker.get_stats()
        assert "tool_a" in s["tools"]
        assert "tool_b" in s["tools"]
        assert s["total_calls"] == 2

    def test_record_appends_line_to_jsonl_file(self) -> None:
        """Each record() writes exactly one JSON line to the log file."""
        tracker, d = _tracker()
        tracker.record("ping", 10.0, True, params={"code_bytes": 5})
        tracker.record("ping", 20.0, False, error="err")
        lines = (d / "tool_usage.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2

    def test_record_jsonl_line_contains_correct_fields(self) -> None:
        """The JSONL line serialises tool name, success flag, and error."""
        tracker, d = _tracker()
        tracker.record("execute_isaac_script", 55.5, False, error="ConnectionError")
        line = (d / "tool_usage.jsonl").read_text().strip()
        rec: Dict[str, Any] = json.loads(line)
        assert rec["tool"] == "execute_isaac_script"
        assert rec["success"] is False
        assert rec["error"] == "ConnectionError"


# ---------------------------------------------------------------------------
# ToolUsageTracker._append_log() — OSError suppression
# ---------------------------------------------------------------------------


class TestAppendLogOSErrorSuppression:
    """_append_log silently ignores write failures."""

    def test_record_does_not_raise_when_log_file_is_read_only(self) -> None:
        """OSError from an unwritable log file is suppressed gracefully."""
        tracker, d = _tracker()
        log = d / "tool_usage.jsonl"
        log.touch()
        log.chmod(0o444)
        try:
            tracker.record("tool", 1.0, True)  # must not raise
        finally:
            log.chmod(0o644)


# ---------------------------------------------------------------------------
# ToolUsageTracker.get_stats()
# ---------------------------------------------------------------------------


class TestGetStats:
    """get_stats() return contract."""

    def test_get_stats_returns_all_tools_when_no_filter(self) -> None:
        """All tracked tools appear in the result when tool_name is not set."""
        tracker, _ = _tracker()
        tracker.record("a", 1.0, True)
        tracker.record("b", 2.0, True)
        s = tracker.get_stats()
        assert set(s["tools"].keys()) == {"a", "b"}

    def test_get_stats_sorted_by_total_calls_descending(self) -> None:
        """Tools are ordered most-called first."""
        tracker, _ = _tracker()
        tracker.record("rare", 1.0, True)
        tracker.record("frequent", 1.0, True)
        tracker.record("frequent", 1.0, True)
        names = list(tracker.get_stats()["tools"].keys())
        assert names[0] == "frequent"

    def test_get_stats_single_tool_returns_correct_shape(self) -> None:
        """Filtering by tool_name returns {'tool': name, 'stats': {...}}."""
        tracker, _ = _tracker()
        tracker.record("ping", 10.0, True)
        result = tracker.get_stats(tool_name="ping")
        assert result["tool"] == "ping"
        assert result["stats"]["total_calls"] == 1

    def test_get_stats_unknown_tool_returns_none_stats(self) -> None:
        """Requesting stats for a never-called tool returns stats=None."""
        tracker, _ = _tracker()
        result = tracker.get_stats(tool_name="unknown_tool")
        assert result["stats"] is None

    def test_get_stats_total_calls_sums_across_all_tools(self) -> None:
        """total_calls at the top level is the sum over all tools."""
        tracker, _ = _tracker()
        tracker.record("a", 1.0, True)
        tracker.record("a", 1.0, True)
        tracker.record("b", 1.0, False)
        assert tracker.get_stats()["total_calls"] == 3

    def test_get_stats_on_empty_tracker_returns_zero_totals(self) -> None:
        """A fresh tracker returns zero total_calls and an empty tools dict."""
        tracker, _ = _tracker()
        s = tracker.get_stats()
        assert s["total_calls"] == 0
        assert s["tools"] == {}


# ---------------------------------------------------------------------------
# ToolUsageTracker.get_recent()
# ---------------------------------------------------------------------------


class TestGetRecent:
    """get_recent() in-memory log contract."""

    def test_get_recent_returns_newest_first(self) -> None:
        """Results are ordered newest-first (reverse insertion order)."""
        tracker, _ = _tracker()
        tracker.record("first", 1.0, True)
        tracker.record("second", 2.0, True)
        recent = tracker.get_recent()
        assert recent[0]["tool"] == "second"
        assert recent[1]["tool"] == "first"

    def test_get_recent_respects_limit(self) -> None:
        """At most limit records are returned."""
        tracker, _ = _tracker()
        for i in range(10):
            tracker.record(f"tool_{i}", 1.0, True)
        assert len(tracker.get_recent(limit=3)) == 3

    def test_get_recent_filters_by_tool_name(self) -> None:
        """Only records matching tool_name are returned when filter is set."""
        tracker, _ = _tracker()
        tracker.record("ping", 1.0, True)
        tracker.record("execute_isaac_script", 2.0, True)
        tracker.record("ping", 3.0, False)
        results = tracker.get_recent(tool_name="ping")
        assert all(r["tool"] == "ping" for r in results)
        assert len(results) == 2

    def test_get_recent_returns_empty_list_for_unknown_tool(self) -> None:
        """Filtering by a never-called tool returns an empty list."""
        tracker, _ = _tracker()
        tracker.record("ping", 1.0, True)
        assert tracker.get_recent(tool_name="nonexistent") == []

    def test_get_recent_respects_max_recent_deque_capacity(self) -> None:
        """Records beyond max_recent are evicted from the in-memory deque."""
        tracker, _ = _tracker(max_recent=3)
        for i in range(5):
            tracker.record("tool", float(i), True)
        # Only last 3 survive in memory
        assert len(tracker.get_recent()) == 3


# ---------------------------------------------------------------------------
# ToolUsageTracker.reset()
# ---------------------------------------------------------------------------


class TestReset:
    """reset() clears in-memory state and truncates the JSONL file."""

    def test_reset_clears_in_memory_stats(self) -> None:
        """After reset(), get_stats() reports zero calls."""
        tracker, _ = _tracker()
        tracker.record("ping", 1.0, True)
        tracker.reset()
        assert tracker.get_stats()["total_calls"] == 0

    def test_reset_clears_in_memory_recent_log(self) -> None:
        """After reset(), get_recent() returns an empty list."""
        tracker, _ = _tracker()
        tracker.record("ping", 1.0, True)
        tracker.reset()
        assert tracker.get_recent() == []

    def test_reset_truncates_jsonl_file_to_empty(self) -> None:
        """After reset(), the JSONL log file is empty."""
        tracker, d = _tracker()
        tracker.record("ping", 1.0, True)
        tracker.reset()
        assert (d / "tool_usage.jsonl").read_text() == ""

    def test_reset_suppresses_oserror_on_read_only_file(self) -> None:
        """OSError from write_text on a read-only log is suppressed."""
        tracker, d = _tracker()
        tracker.record("ping", 1.0, True)
        log = d / "tool_usage.jsonl"
        log.chmod(0o444)
        try:
            tracker.reset()  # must not raise
        finally:
            log.chmod(0o644)


# ---------------------------------------------------------------------------
# ToolUsageTracker.load_from_log()
# ---------------------------------------------------------------------------


class TestLoadFromLog:
    """load_from_log() builds stats from the persistent JSONL file."""

    def test_load_from_log_returns_empty_result_when_file_missing(self) -> None:
        """A non-existent log directory returns zero totals."""
        result = ToolUsageTracker.load_from_log(log_dir=Path("/tmp/no_such_simul_dir_xyz"))
        assert result["total_calls"] == 0
        assert result["tools"] == {}

    def test_load_from_log_aggregates_calls_from_file(self) -> None:
        """Stats computed from disk match the records written."""
        tracker, d = _tracker()
        tracker.record("ping", 10.0, True)
        tracker.record("ping", 20.0, False)
        tracker.record("exec", 50.0, True)
        result = ToolUsageTracker.load_from_log(log_dir=d)
        assert result["total_calls"] == 3
        assert result["tools"]["ping"]["total_calls"] == 2
        assert result["tools"]["ping"]["failures"] == 1
        assert result["tools"]["exec"]["successes"] == 1

    def test_load_from_log_filters_by_tool_name(self) -> None:
        """tool_name filter excludes records from other tools."""
        tracker, d = _tracker()
        tracker.record("ping", 10.0, True)
        tracker.record("exec", 20.0, True)
        result = ToolUsageTracker.load_from_log(log_dir=d, tool_name="ping")
        assert "exec" not in result["tools"]
        assert result["tools"]["ping"]["total_calls"] == 1

    def test_load_from_log_with_limit_returns_recent_key(self) -> None:
        """limit > 0 adds a 'recent' list to the result."""
        tracker, d = _tracker()
        tracker.record("ping", 10.0, True)
        tracker.record("ping", 20.0, False)
        result = ToolUsageTracker.load_from_log(log_dir=d, limit=5)
        assert "recent" in result

    def test_load_from_log_limit_returns_at_most_n_records(self) -> None:
        """recent list is truncated to at most limit entries."""
        tracker, d = _tracker()
        for i in range(10):
            tracker.record("tool", float(i), True)
        result = ToolUsageTracker.load_from_log(log_dir=d, limit=3)
        assert len(result["recent"]) == 3

    def test_load_from_log_recent_is_newest_first(self) -> None:
        """Recent records are returned newest-first."""
        tracker, d = _tracker()
        tracker.record("a", 1.0, True)
        tracker.record("b", 2.0, True)
        result = ToolUsageTracker.load_from_log(log_dir=d, limit=2)
        assert result["recent"][0]["tool"] == "b"
        assert result["recent"][1]["tool"] == "a"

    def test_load_from_log_without_limit_omits_recent_key(self) -> None:
        """When limit=0, the 'recent' key is absent from the result."""
        tracker, d = _tracker()
        tracker.record("ping", 10.0, True)
        result = ToolUsageTracker.load_from_log(log_dir=d, limit=0)
        assert "recent" not in result

    def test_load_from_log_skips_corrupt_json_lines(self) -> None:
        """Malformed JSONL lines are skipped without raising an exception."""
        _, d = _tracker()
        log = d / "tool_usage.jsonl"
        log.write_text(
            json.dumps({"tool": "ping", "timestamp": 1.0, "duration_ms": 10.0, "success": True}) + "\n"
            + "NOT VALID JSON\n"
            + json.dumps({"tool": "ping", "timestamp": 2.0, "duration_ms": 20.0, "success": False}) + "\n"
        )
        result = ToolUsageTracker.load_from_log(log_dir=d)
        # Corrupt line is skipped; 2 valid lines processed
        assert result["total_calls"] == 2

    def test_load_from_log_skips_empty_lines_in_file(self) -> None:
        """Empty lines in the JSONL file are skipped without raising."""
        _, d = _tracker()
        log = d / "tool_usage.jsonl"
        log.write_text(
            "\n"
            + json.dumps({"tool": "ping", "timestamp": 1.0, "duration_ms": 5.0, "success": True}) + "\n"
            + "\n"
        )
        result = ToolUsageTracker.load_from_log(log_dir=d)
        assert result["total_calls"] == 1

    def test_load_from_log_nonexistent_tool_filter_returns_empty(self) -> None:
        """Filtering by a tool with no records returns zero totals."""
        tracker, d = _tracker()
        tracker.record("ping", 10.0, True)
        result = ToolUsageTracker.load_from_log(log_dir=d, tool_name="nonexistent")
        assert result["total_calls"] == 0
        assert result["tools"] == {}

    def test_load_from_log_survives_reset_between_sessions(self) -> None:
        """After reset() the file is empty; load_from_log returns zero totals."""
        tracker, d = _tracker()
        tracker.record("ping", 10.0, True)
        tracker.reset()
        result = ToolUsageTracker.load_from_log(log_dir=d)
        assert result["total_calls"] == 0
