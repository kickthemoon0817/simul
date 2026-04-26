"""Unit tests for the structured-logging machinery introduced for the MCP server.

Covers the high-impact regression surfaces flagged in PR #23 review:

* ContextVar reset on success and exception (sync + async)
* JsonFormatter stable schema (load-bearing for downstream tooling)
* AuditFormatter strict-mode (only the audit payload is emitted)
* _setup_audit_handler idempotency
* _configure_file_handler_paths PID suffix position
* simul logs tail tool filter (matches both 'tool' and 'tool_name' keys)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import pytest

src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_path))

from simul_mcp.config import Settings
from simul_mcp.logging import (  # noqa: E402  (sys.path manipulation above)
    AuditFormatter,
    JsonFormatter,
    _configure_file_handler_paths,
    _REQUEST_ID_VAR,
    _setup_audit_handler,
    _stop_audit_listener,
    _TOOL_NAME_VAR,
    wrap_tool_with_context,
)


# ---------------------------------------------------------------------------
# wrap_tool_with_context: ContextVar reset on success and on exception
# ---------------------------------------------------------------------------


class TestWrapToolContextVarReset:
    """Stale request_id leaking into subsequent records is a silent corruption.

    The ``finally`` blocks must reset the ContextVars on every path including
    BaseException, otherwise every later log line in the process inherits a
    dead correlation id.
    """

    def test_sync_resets_on_success(self) -> None:
        wrapped = wrap_tool_with_context(lambda x: x + 1, "sync_tool")
        assert wrapped(41) == 42
        assert _REQUEST_ID_VAR.get() == "-"
        assert _TOOL_NAME_VAR.get() == "-"

    def test_sync_resets_after_exception(self) -> None:
        def boom() -> None:
            raise RuntimeError("oops")

        wrapped = wrap_tool_with_context(boom, "boom_tool")
        with pytest.raises(RuntimeError):
            wrapped()
        assert _REQUEST_ID_VAR.get() == "-"
        assert _TOOL_NAME_VAR.get() == "-"

    def test_async_resets_on_success(self) -> None:
        async def add(x: int) -> int:
            return x + 1

        wrapped = wrap_tool_with_context(add, "async_tool")
        assert asyncio.run(wrapped(41)) == 42
        assert _REQUEST_ID_VAR.get() == "-"
        assert _TOOL_NAME_VAR.get() == "-"

    def test_async_resets_after_exception(self) -> None:
        async def async_boom() -> None:
            raise ValueError("async oops")

        wrapped = wrap_tool_with_context(async_boom, "async_boom")
        with pytest.raises(ValueError):
            asyncio.run(wrapped())
        assert _REQUEST_ID_VAR.get() == "-"
        assert _TOOL_NAME_VAR.get() == "-"

    def test_rejects_async_generator(self) -> None:
        async def streaming():  # type: ignore[no-untyped-def]
            yield 1

        with pytest.raises(TypeError):
            wrap_tool_with_context(streaming, "streamer")


# ---------------------------------------------------------------------------
# JsonFormatter: stable schema, exception serialisation
# ---------------------------------------------------------------------------


class TestJsonFormatter:
    """Downstream tooling indexes off these keys; renaming any breaks them."""

    @staticmethod
    def _record(msg: str = "hello", exc_info=None) -> logging.LogRecord:
        record = logging.LogRecord(
            name="simul_mcp.test",
            level=logging.INFO,
            pathname="x.py",
            lineno=7,
            msg=msg,
            args=(),
            exc_info=exc_info,
        )
        record.request_id = "abc123"
        record.tool_name = "fake_tool"
        return record

    def test_stable_keys_present(self) -> None:
        result = json.loads(JsonFormatter().format(self._record()))
        for key in ("ts", "level", "logger", "request_id", "tool", "msg"):
            assert key in result, f"missing key: {key}"

    def test_exception_serialisation(self) -> None:
        try:
            raise TypeError("bad type")
        except TypeError:
            exc_info = sys.exc_info()

        result = json.loads(JsonFormatter().format(self._record(exc_info=exc_info)))
        assert result["exc_class"] == "TypeError"
        assert "exc" in result and "TypeError" in result["exc"]

    def test_no_double_newline(self) -> None:
        """Each emitted line is a single JSON object terminated by exactly one newline."""
        out = JsonFormatter().format(self._record(msg="line1\nline2"))
        assert "\n" not in out  # embedded newline must be escaped, not literal
        json.loads(out)  # round-trips


# ---------------------------------------------------------------------------
# AuditFormatter: strict — emits only the audit payload
# ---------------------------------------------------------------------------


class TestAuditFormatter:
    def test_emits_only_audit_payload(self) -> None:
        record = logging.LogRecord(
            name="simul_mcp.audit",
            level=logging.INFO,
            pathname="x.py",
            lineno=1,
            msg="ignored",
            args=(),
            exc_info=None,
        )
        record.audit = {"tool": "t", "request_id": "r", "duration_ms": 1.0, "status": "ok"}
        out = json.loads(AuditFormatter().format(record))
        assert out["tool"] == "t"
        assert out["status"] == "ok"
        # The freeform message is NOT in the audit row
        assert "msg" not in out


# ---------------------------------------------------------------------------
# _setup_audit_handler idempotency: re-entry must not duplicate or leak threads
# ---------------------------------------------------------------------------


class TestSetupAuditHandlerIdempotency:
    def test_calling_twice_results_in_single_handler_entry(self, tmp_path: Path) -> None:
        settings = Settings()
        # Settings is frozen; rebuild the logging block via copy/update.
        settings = settings.model_copy(
            update={
                "logging": settings.logging.model_copy(
                    update={"audit_path": str(tmp_path / "audit.jsonl")}
                )
            }
        )

        config: dict = {"formatters": {}, "handlers": {}, "loggers": {}}
        _setup_audit_handler(config, settings)
        _setup_audit_handler(config, settings)

        assert list(config["handlers"]).count("audit_file") == 1
        assert config["loggers"]["simul_mcp.audit"]["handlers"] == ["audit_file"]
        # Cleanup the listener thread the second call started.
        _stop_audit_listener()


# ---------------------------------------------------------------------------
# _configure_file_handler_paths: PID suffix must come BEFORE the extension
# ---------------------------------------------------------------------------


class TestConfigureFileHandlerPaths:
    def test_per_instance_inserts_pid_before_extension(self, tmp_path: Path) -> None:
        handlers = {"file": {"filename": ""}, "file_json": {"filename": ""}}
        _configure_file_handler_paths(
            handlers, str(tmp_path / "simul_mcp.log"), per_instance=True
        )

        pid = str(os.getpid())
        for handler_name, expected_suffix in (("file", ".log"), ("file_json", ".json")):
            name = Path(handlers[handler_name]["filename"]).name
            assert name.endswith(expected_suffix), name
            assert pid in name
            assert name.index(pid) < name.index(expected_suffix)

    def test_per_instance_false_yields_stable_filename(self, tmp_path: Path) -> None:
        handlers = {"file": {"filename": ""}}
        _configure_file_handler_paths(
            handlers, str(tmp_path / "simul_mcp.log"), per_instance=False
        )
        assert Path(handlers["file"]["filename"]).name == "simul_mcp.log"


# ---------------------------------------------------------------------------
# simul logs tail tool filter: must accept both 'tool' and 'tool_name' keys
# ---------------------------------------------------------------------------


class TestLogsTailToolFilter:
    def test_filters_out_non_matching_tool(self) -> None:
        from simul_mcp.cli.main import _format_jsonl_line

        line = json.dumps({"tool": "other", "msg": "x", "level": "INFO"})
        assert _format_jsonl_line(line, tool_filter="my_tool") is None

    def test_passes_matching_tool_name_alias(self) -> None:
        from simul_mcp.cli.main import _format_jsonl_line

        line = json.dumps({"tool_name": "my_tool", "msg": "x", "level": "INFO"})
        rendered = _format_jsonl_line(line, tool_filter="my_tool")
        assert rendered is not None and "my_tool" in rendered

    def test_plain_text_line_returned_as_is(self) -> None:
        from simul_mcp.cli.main import _format_jsonl_line

        plain = "2026-04-26 12:00:00 - simul_mcp - INFO - bare text"
        assert _format_jsonl_line(plain, tool_filter=None) == plain
