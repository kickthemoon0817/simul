"""Unit tests for the structured-logging machinery introduced for the MCP server.

Covers the high-impact regression surfaces flagged in PR #23 review:

* ContextVar reset on success and exception (request-context middleware)
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


from simul_mcp.config import Settings
from simul_mcp.logging import (  # noqa: E402  (sys.path manipulation above)
    AuditFormatter,
    JsonFormatter,
    _configure_file_handler_paths,
    _REQUEST_ID_VAR,
    _setup_audit_handler,
    _stop_audit_listener,
    _TOOL_NAME_VAR,
    build_request_context_middleware,
)


# ---------------------------------------------------------------------------
# RequestContextMiddleware: ContextVar reset on success and on exception
# ---------------------------------------------------------------------------


class _FakeMessage:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeContext:
    def __init__(self, name: str) -> None:
        self.message = _FakeMessage(name)


class TestRequestContextMiddlewareContextVarReset:
    """Stale request_id leaking into subsequent records is a silent corruption.

    The ``finally`` blocks must reset the ContextVars on every path including
    BaseException, otherwise every later log line in the process inherits a
    dead correlation id.
    """

    def test_sets_context_during_call_and_resets_on_success(self) -> None:
        seen: dict = {}

        async def call_next(context):  # type: ignore[no-untyped-def]
            seen["request_id"] = _REQUEST_ID_VAR.get()
            seen["tool_name"] = _TOOL_NAME_VAR.get()
            return 42

        middleware = build_request_context_middleware()

        async def run() -> tuple:
            # asyncio.run gives the task its own context copy, so the reset
            # must be observed from inside the same task, not after run().
            result = await middleware.on_call_tool(_FakeContext("my_tool"), call_next)
            return result, _REQUEST_ID_VAR.get(), _TOOL_NAME_VAR.get()

        result, rid_after, tool_after = asyncio.run(run())
        assert result == 42
        assert seen["tool_name"] == "my_tool"
        assert len(seen["request_id"]) == 12
        assert rid_after == "-"
        assert tool_after == "-"

    def test_resets_after_exception(self) -> None:
        async def call_next(context):  # type: ignore[no-untyped-def]
            raise ValueError("async oops")

        middleware = build_request_context_middleware()

        async def run() -> tuple:
            with pytest.raises(ValueError):
                await middleware.on_call_tool(_FakeContext("boom_tool"), call_next)
            return _REQUEST_ID_VAR.get(), _TOOL_NAME_VAR.get()

        assert asyncio.run(run()) == ("-", "-")

    def test_emits_one_audit_record_per_call(self, caplog: pytest.LogCaptureFixture) -> None:
        async def ok(context):  # type: ignore[no-untyped-def]
            return None

        async def boom(context):  # type: ignore[no-untyped-def]
            raise RuntimeError("x")

        middleware = build_request_context_middleware()
        with caplog.at_level(logging.INFO, logger="simul_mcp.audit"):
            asyncio.run(middleware.on_call_tool(_FakeContext("ok_tool"), ok))
            with pytest.raises(RuntimeError):
                asyncio.run(middleware.on_call_tool(_FakeContext("bad_tool"), boom))
        audits = [r.audit for r in caplog.records if hasattr(r, "audit")]
        assert [a["tool"] for a in audits] == ["ok_tool", "bad_tool"]
        assert audits[0]["status"] == "ok"
        assert audits[1]["status"] == "error"
        assert audits[1]["error_class"] == "RuntimeError"


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


# ---------------------------------------------------------------------------
# Setup warnings must stay off stdout, which carries stdio MCP JSON-RPC
# ---------------------------------------------------------------------------


class TestSetupWarningsGoToStderr:
    def test_unloadable_config_warning_is_on_stderr(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        import simul_mcp.logging as logging_module

        fallback_calls: list = []
        monkeypatch.setattr(
            logging_module,
            "_setup_fallback_logging",
            lambda settings, level: fallback_calls.append(level),
        )
        bad = tmp_path / "logging.yaml"
        bad.write_text("version: 1\nhandlers: [unclosed\n", encoding="utf-8")

        logging_module.setup_logging(settings=Settings(), config_file=bad)

        captured = capsys.readouterr()
        assert fallback_calls, "an unloadable config must fall back"
        assert "Failed to load logging config" in captured.err
        assert captured.out == ""

    def test_uncreatable_log_directory_warning_is_on_stderr(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from simul_mcp.logging import _ensure_log_directories

        def refuse(self: Path, *args: object, **kwargs: object) -> None:
            raise PermissionError("read-only filesystem")

        monkeypatch.setattr(Path, "mkdir", refuse)
        config = {"handlers": {"file": {"filename": str(tmp_path / "missing" / "simul.log")}}}

        _ensure_log_directories(config)

        captured = capsys.readouterr()
        assert "Could not create log directory" in captured.err
        assert captured.out == ""
