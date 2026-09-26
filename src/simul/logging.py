"""
Logging configuration and utilities for Isaac Sim MCP Server.

This module provides centralized logging configuration using Python's dictConfig
with support for multiple handlers, formatters, and log levels.

It also wires four observability features that compose with the YAML config:
    * Per-tool ``request_id`` and ``tool_name`` propagation via ``ContextVar``
      so deep log records remain correlatable to the originating MCP call.
    * A ``RequestContextFilter`` that injects those values into every record.
    * A ``JsonFormatter`` that renders structured records (used by ``file_json``).
    * An audit channel (``simul.audit``) that writes one JSON line per
      tool-call boundary into ``settings.logging.audit_path``.
"""

import atexit
import contextlib
import contextvars
import importlib.util
import json
import logging
import logging.config
import logging.handlers
import os
import queue
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union

import yaml

from .config import Settings, get_settings
from .resources import find_checkout_root, resource_filesystem_path


_CHECKOUT_ROOT: Optional[Path] = find_checkout_root()
_DEFAULT_LOGGING_CONFIG: Path = resource_filesystem_path("config", "logging.yaml")

# Context variables that carry per-request correlation across logging records.
# Defaults of ``-`` keep formatters readable when no MCP call is active
# (startup, shutdown, background housekeeping).
_REQUEST_ID_VAR: contextvars.ContextVar[str] = contextvars.ContextVar(
    "simul_request_id", default="-"
)
_TOOL_NAME_VAR: contextvars.ContextVar[str] = contextvars.ContextVar(
    "simul_tool_name", default="-"
)

_AUDIT_LOGGER_NAME = "simul.audit"


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with microseconds."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def setup_logging(
    settings: Optional[Settings] = None,
    config_file: Optional[Union[str, Path]] = None,
    log_level: Optional[str] = None,
    profile: Optional[str] = None,
) -> None:
    """
    Setup logging configuration.

    Args:
        settings: Settings instance. If None, uses get_settings()
        config_file: Path to logging config file. If None, uses the logging.yaml
            shipped inside the package.
        log_level: Override log level
        profile: Logging profile to use (development, production, testing, json_logging)
    """
    if settings is None:
        settings = get_settings()

    config_file = _resolve_logging_config_path(config_file or _DEFAULT_LOGGING_CONFIG)

    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            _apply_settings_to_dictconfig(config, settings)

            if profile and "profiles" in config and profile in config["profiles"]:
                profile_config = config["profiles"][profile]
                if "loggers" in profile_config:
                    config.setdefault("loggers", {}).update(profile_config["loggers"])
                if "handlers" in profile_config:
                    config.setdefault("handlers", {}).update(profile_config["handlers"])

            if log_level:
                for logger_config in config.get("loggers", {}).values():
                    logger_config["level"] = log_level.upper()
                if "root" in config:
                    config["root"]["level"] = log_level.upper()

            # Ensure log directories exist
            _ensure_log_directories(config)

            # Apply the configuration
            logging.config.dictConfig(config)

            # Attach the request-context filter to every handler that doesn't
            # already have one. Done after dictConfig so it covers handlers
            # synthesised by ``_setup_audit_handler`` as well.
            _install_request_context_filter()

        except Exception as e:
            # Fallback to basic configuration
            print(f"Warning: Failed to load logging config from {config_file}: {e}", file=sys.stderr)
            _setup_fallback_logging(settings, log_level)
    else:
        # Use fallback configuration
        _setup_fallback_logging(settings, log_level)

    logger = logging.getLogger("simul.logging")
    logger.info("Logging system initialized")
    if profile:
        logger.info(f"Using logging profile: {profile}")


def _ensure_log_directories(config: Dict[str, Any]) -> None:
    """Ensure all log directories exist; expand ~ in handler filenames.

    Python's RotatingFileHandler does not call ``expanduser`` on the path it
    receives, so a filename like ``~/.simul/logs/simul.log`` would create
    a literal ``~`` directory in the CWD. We expand here and write the
    resolved absolute path back into the dictConfig payload.
    """
    handlers = config.get("handlers", {})

    for handler_config in handlers.values():
        if "filename" not in handler_config:
            continue
        log_file = Path(handler_config["filename"]).expanduser()
        handler_config["filename"] = str(log_file)
        log_dir = log_file.parent
        if not log_dir.exists():
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"Warning: Could not create log directory {log_dir}: {e}", file=sys.stderr)


def _resolve_logging_config_path(config_file: Union[str, Path]) -> Path:
    """Resolve logging config relative to cwd first, then the source checkout if any."""
    candidate = Path(config_file).expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.exists() or _CHECKOUT_ROOT is None:
        return candidate.resolve()
    return (_CHECKOUT_ROOT / candidate).resolve()


def _remove_handlers(config: Dict[str, Any], disabled: set[str]) -> None:
    """Strip disabled handlers from root/logger handler lists."""
    handlers = config.get("handlers", {})
    for handler_name in disabled:
        handlers.pop(handler_name, None)

    root = config.get("root")
    if isinstance(root, dict):
        root["handlers"] = [
            name for name in root.get("handlers", []) if name not in disabled
        ]

    for logger_config in config.get("loggers", {}).values():
        if isinstance(logger_config, dict) and "handlers" in logger_config:
            logger_config["handlers"] = [
                name
                for name in logger_config.get("handlers", [])
                if name not in disabled
            ]


def _colorlog_available() -> bool:
    """Return True when colorlog is importable."""
    return importlib.util.find_spec("colorlog") is not None


def _configure_file_handler_paths(
    handlers: Dict[str, Any], base_path: str, per_instance: bool = False
) -> None:
    """Update file handler paths to match ``Settings.logging.file_path``.

    When ``per_instance`` is True, the process PID is interleaved into each
    filename (``simul.<pid>.log``) so concurrently running servers do not
    interleave records into the same file.
    """
    target = Path(base_path).expanduser()
    suffix = target.suffix or ".log"
    stem = target.stem if target.suffix else target.name
    parent = target.parent
    pid_part = f".{os.getpid()}" if per_instance else ""

    if "file" in handlers:
        handlers["file"]["filename"] = str(parent / f"{stem}{pid_part}{suffix}")
    if "file_json" in handlers:
        handlers["file_json"]["filename"] = str(parent / f"{stem}{pid_part}.json")
    if "error_file" in handlers:
        handlers["error_file"]["filename"] = str(
            parent / f"{stem}{pid_part}_errors{suffix}"
        )
    if "debug_file" in handlers:
        handlers["debug_file"]["filename"] = str(
            parent / f"{stem}{pid_part}_debug{suffix}"
        )


def _apply_retention_to_timed_handlers(
    handlers: Dict[str, Any], retention_days: int
) -> None:
    """Set ``backupCount`` on every TimedRotatingFileHandler we control.

    The YAML carries a default; this lets ``Settings.logging.retention_days``
    override it without editing the YAML on each deployment.
    """
    for name in ("file", "file_json", "error_file", "debug_file", "audit_file"):
        cfg = handlers.get(name)
        if not cfg:
            continue
        if cfg.get("class") == "logging.handlers.TimedRotatingFileHandler":
            cfg["backupCount"] = retention_days


# Module-level state for the audit queue + listener pair. Created lazily by
# _start_audit_listener() and torn down via atexit.
_audit_queue: Optional["queue.Queue[Any]"] = None
_audit_listener: Optional[logging.handlers.QueueListener] = None


def _audit_filename_for(settings: Settings) -> Path:
    """Resolve the audit log path with ``per_instance`` PID interleaving applied."""
    target = Path(settings.logging.audit_path).expanduser()
    if not settings.logging.per_instance:
        return target
    suffix = target.suffix or ".jsonl"
    stem = target.stem if target.suffix else target.name
    return target.parent / f"{stem}.{os.getpid()}{suffix}"


def _start_audit_listener(audit_path: Path, retention_days: int) -> None:
    """Run the audit ``TimedRotatingFileHandler`` on a background thread.

    Producers (``emit_audit``) write to a queue via ``QueueHandler``; the
    ``QueueListener`` drains it on its own thread, so synchronous file I/O
    (and any midnight rollover stall) never blocks the asyncio event loop
    that handles MCP tool calls.

    Re-entrant: if a listener is already running we tear it down and start a
    new one (handles ``setup_logging()`` being called more than once).
    """
    global _audit_queue, _audit_listener

    if _audit_listener is not None:
        try:
            _audit_listener.stop()
        except Exception:  # pragma: no cover - defensive
            pass
        _audit_listener = None
    _audit_queue = queue.Queue(-1)

    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=str(audit_path),
        when="midnight",
        backupCount=retention_days,
        encoding="utf8",
    )
    file_handler.setFormatter(AuditFormatter())
    file_handler.setLevel(logging.INFO)

    _audit_listener = logging.handlers.QueueListener(
        _audit_queue, file_handler, respect_handler_level=True
    )
    _audit_listener.start()
    atexit.register(_stop_audit_listener)


def _stop_audit_listener() -> None:
    """Stop the audit listener thread; safe to call multiple times."""
    global _audit_listener
    if _audit_listener is None:
        return
    try:
        _audit_listener.stop()
    except Exception:  # pragma: no cover - defensive
        pass
    _audit_listener = None


def _setup_audit_handler(config: Dict[str, Any], settings: Settings) -> None:
    """Wire the ``simul.audit`` logger through a non-blocking queue.

    The actual file I/O happens on a background ``QueueListener`` thread (see
    ``_start_audit_listener``); the logger sees only a ``QueueHandler``,
    keeping the producer side free of blocking writes.

    Idempotent: re-running ``setup_logging`` rebuilds the listener and replaces
    the handler entry without leaving duplicates behind.
    """
    if not settings.logging.audit_enabled:
        return

    audit_path = _audit_filename_for(settings)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    _start_audit_listener(audit_path, settings.logging.retention_days)
    assert _audit_queue is not None  # set by _start_audit_listener

    formatters = config.setdefault("formatters", {})
    formatters.setdefault(
        "audit_json",
        {"()": "simul.logging.AuditFormatter"},
    )

    handlers = config.setdefault("handlers", {})
    handlers["audit_file"] = {
        "class": "logging.handlers.QueueHandler",
        "queue": _audit_queue,
        "level": "INFO",
    }

    loggers = config.setdefault("loggers", {})
    loggers["simul.audit"] = {
        "level": "INFO",
        "handlers": ["audit_file"],
        "propagate": False,
    }


def _apply_settings_to_dictconfig(
    config: Dict[str, Any], settings: Settings
) -> None:
    """Apply Settings-level toggles to the YAML dictConfig payload."""
    handlers = config.setdefault("handlers", {})
    disabled_handlers: set[str] = set()

    if not settings.logging.console_enabled:
        disabled_handlers.update({"console", "console_simple"})
    if not settings.logging.file_enabled:
        disabled_handlers.update({"file", "file_json", "error_file", "debug_file"})
    else:
        _configure_file_handler_paths(
            handlers,
            settings.logging.file_path,
            per_instance=settings.logging.per_instance,
        )

    # Synthesise the audit handler before applying retention so the audit
    # backupCount goes through the same uniform code path as the other
    # TimedRotatingFileHandlers.
    _setup_audit_handler(config, settings)

    if settings.logging.file_enabled:
        _apply_retention_to_timed_handlers(
            handlers, settings.logging.retention_days
        )

    if not settings.logging.structured_enabled:
        # Default: only the audit JSONL is structured. Removing ``file_json``
        # from every logger's handler list keeps per-record formatting cost at
        # one ``json.dumps`` (audit only) rather than two on the root path.
        disabled_handlers.add("file_json")

    if disabled_handlers:
        _remove_handlers(config, disabled_handlers)

    if "colored" in config.get("formatters", {}) and (
        not settings.logging.console_colored or not _colorlog_available()
    ):
        config["formatters"].pop("colored", None)
        for handler_name in ("console", "console_simple"):
            handler = handlers.get(handler_name)
            if handler and handler.get("formatter") == "colored":
                handler["formatter"] = "simple"

    component_levels = settings.logging.components or {}
    for logger_name, level in component_levels.items():
        config.setdefault("loggers", {}).setdefault(logger_name, {})["level"] = level


def _setup_fallback_logging(
    settings: Settings, log_level: Optional[str] = None
) -> None:
    """Setup fallback logging configuration when YAML config fails."""
    level = log_level or settings.logging.level

    handlers: list[logging.Handler] = []
    if settings.logging.console_enabled:
        handlers.append(logging.StreamHandler(sys.stderr))

    if settings.logging.file_enabled:
        log_path = Path(settings.logging.file_path).expanduser()
        log_dir = log_path.parent
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(log_path), encoding="utf-8"))

    if not handlers:
        handlers.append(logging.NullHandler())

    # Basic logging configuration
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )

    for logger_name, logger_level in (settings.logging.components or {}).items():
        logging.getLogger(logger_name).setLevel(logger_level)

    _install_request_context_filter()


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name, typically __name__ of the calling module

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return get_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")


# ---------------------------------------------------------------------------
# Structured logging: request context, JSON formatters, audit emission
# ---------------------------------------------------------------------------


class RequestContextFilter(logging.Filter):
    """Inject ``request_id`` and ``tool_name`` from contextvars into every record.

    Without this filter, format strings that reference ``%(request_id)s`` would
    raise ``KeyError`` for any record emitted outside an active MCP call.
    Defaults of ``-`` keep the formatted output legible.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = _REQUEST_ID_VAR.get()
        if not hasattr(record, "tool_name"):
            record.tool_name = _TOOL_NAME_VAR.get()
        return True


def _install_request_context_filter() -> None:
    """Attach a ``RequestContextFilter`` to every active handler exactly once."""
    sentinel_attr = "_simul_context_filter_attached"
    for handler in logging.getLogger().handlers:
        if getattr(handler, sentinel_attr, False):
            continue
        handler.addFilter(RequestContextFilter())
        setattr(handler, sentinel_attr, True)
    # Also attach to handlers on named loggers we own, to cover cases where the
    # YAML routes records through a non-root logger first.
    for logger_name in (
        "simul",
        "simul.audit",
        "fastmcp",
        "mcp",
    ):
        for handler in logging.getLogger(logger_name).handlers:
            if getattr(handler, sentinel_attr, False):
                continue
            handler.addFilter(RequestContextFilter())
            setattr(handler, sentinel_attr, True)


class JsonFormatter(logging.Formatter):
    """Render every log record as a single JSON line with full context.

    Output keys are stable so downstream tooling can rely on them: ``ts``,
    ``level``, ``logger``, ``request_id``, ``tool``, ``file``, ``line``,
    ``func``, ``msg``. Exceptions add ``exc_class`` and ``exc``.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": _utc_now_iso(),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "tool": getattr(record, "tool_name", "-"),
            "file": record.filename,
            "line": record.lineno,
            "func": record.funcName,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            payload["exc_class"] = record.exc_info[0].__name__
            payload["exc"] = self.formatException(record.exc_info)
        elif record.exc_text:
            payload["exc"] = record.exc_text
        return json.dumps(payload, ensure_ascii=False, default=str)


class AuditFormatter(logging.Formatter):
    """Render audit records (``logger.info('', extra={'audit': {...}})``).

    Only the structured ``audit`` payload is emitted; the human-readable
    ``message`` is ignored on purpose so the JSONL stays a strict tool-call
    record.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = getattr(record, "audit", None)
        if not isinstance(payload, dict):
            payload = {"msg": record.getMessage()}
        payload.setdefault("ts", _utc_now_iso())
        return json.dumps(payload, ensure_ascii=False, default=str)


def emit_audit(
    *,
    tool: str,
    request_id: str,
    duration_ms: float,
    status: str,
    error_class: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit one audit JSONL record describing a completed tool-call boundary."""
    audit_logger = logging.getLogger(_AUDIT_LOGGER_NAME)
    payload: Dict[str, Any] = {
        "ts": _utc_now_iso(),
        "request_id": request_id,
        "tool": tool,
        "duration_ms": round(duration_ms, 3),
        "status": status,
    }
    if error_class:
        payload["error_class"] = error_class
    if extra:
        payload["extra"] = extra
    audit_logger.info("", extra={"audit": payload})


@contextlib.contextmanager
def _tool_call_context(tool_name: str) -> Iterator[str]:
    """Scope one tool call: fresh request id, ContextVars, timing and audit row.

    Generates a 12-char hex correlation id, sets the ``request_id`` and
    ``tool_name`` ContextVars for the duration of the block, measures
    wall-time and emits one audit record on exit (success or failure). The
    ContextVars are reset on every path, including ``BaseException``, so a
    stale id never leaks into later log records.
    """
    request_id = uuid.uuid4().hex[:12]
    rid_token = _REQUEST_ID_VAR.set(request_id)
    tn_token = _TOOL_NAME_VAR.set(tool_name)
    start = time.monotonic()
    error_class: Optional[str] = None
    try:
        yield request_id
    except BaseException as exc:
        error_class = type(exc).__name__
        raise
    finally:
        duration_ms = (time.monotonic() - start) * 1000.0
        try:
            emit_audit(
                tool=tool_name,
                request_id=request_id,
                duration_ms=duration_ms,
                status="error" if error_class else "ok",
                error_class=error_class,
            )
        finally:
            _REQUEST_ID_VAR.reset(rid_token)
            _TOOL_NAME_VAR.reset(tn_token)


def build_request_context_middleware() -> Any:
    """Construct a FastMCP middleware that owns request-context + audit emission.

    Implemented as a factory (rather than a top-level class) so the import of
    ``fastmcp`` stays optional: this function is only called from the MCP
    server bootstrap, where ``fastmcp`` is guaranteed to be importable.

    The middleware wraps every ``CallTool`` request in ``_tool_call_context``
    at the FastMCP dispatch boundary, which means:
        * every ``@server.mcp.tool(...)`` decorator pattern is covered
        * audit emission happens once per ``CallTool`` request, not once per
          internal tool invocation
        * sync and async tools both flow through ``call_next`` uniformly
    """
    from fastmcp.server.middleware import Middleware

    class RequestContextMiddleware(Middleware):
        """Tag every ``CallTool`` request with a fresh correlation id + audit row."""

        async def on_call_tool(self, context: Any, call_next: Any) -> Any:
            tool_name = getattr(context.message, "name", "tool")
            with _tool_call_context(tool_name):
                return await call_next(context)

    return RequestContextMiddleware()


# Module-level logger for this module
_logger = get_logger(__name__)
