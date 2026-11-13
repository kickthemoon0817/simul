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
import re
import sys
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import yaml

from .config import Settings, get_settings
from .resources import find_checkout_root, resource_filesystem_path
from .utils.filelock import exclusive_file_lock


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


#: Mode of the log and audit files simul writes. They record tool calls with
#: their arguments (scripts, file paths), so on a shared machine no other local
#: user may read them.
LOG_FILE_MODE = 0o600
#: Mode of the directories simul creates for them.
LOG_DIR_MODE = 0o700


def _simul_state_root() -> Path:
    """The directory simul owns its state under (``$SIMUL_HOME``, default ``~/.simul``)."""
    from .utils.paths import state_dir

    return state_dir()


def make_private_dir(path: Path) -> None:
    """Create ``path`` and any missing parents with mode 0700.

    ``Path.mkdir(mode=..., parents=True)`` applies the mode only to the last
    directory, so each missing level is created on its own. An existing
    directory keeps its mode: simul did not create it, and a user may have
    pointed ``SIMUL_HOME`` or the log paths at a directory they share (their
    home). The files simul writes there are 0600 all the same.

    Raises:
        OSError: A missing directory could not be created.
    """
    path = Path(path).expanduser()
    missing: List[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        if current.parent == current:
            break
        current = current.parent
    for directory in reversed(missing):
        try:
            os.mkdir(directory, LOG_DIR_MODE)
        except FileExistsError:
            pass
        else:
            # mkdir's mode passes through the umask, which may only narrow it.
            _tighten(directory, LOG_DIR_MODE)


def _tighten(path: Path, mode: int) -> None:
    """Drop group and other permissions from a file or directory this user owns."""
    if os.name == "nt":  # pragma: no cover - POSIX modes only
        return
    try:
        info = path.stat()
    except OSError:
        return
    if info.st_uid != os.getuid() or (info.st_mode & 0o077) == 0:
        return
    try:
        os.chmod(path, (info.st_mode & 0o7777) & mode)
    except OSError:
        pass


def tighten_log_files(base: Path) -> None:
    """Make ``base`` and its rotated copies (``base.*``) 0600 when this user owns them."""
    base = Path(base).expanduser()
    candidates = [base]
    try:
        candidates += [p for p in base.parent.glob(f"{base.name}.*") if p.is_file()]
    except OSError:
        pass
    for candidate in candidates:
        _tighten(candidate, LOG_FILE_MODE)


def _open_private(filename: str, mode: str, encoding: Optional[str], errors: Optional[str]) -> Any:
    """Open a log file the way logging's handlers do, creating it with mode 0600."""
    flags = os.O_WRONLY | os.O_CREAT | (os.O_APPEND if "a" in mode else os.O_TRUNC)
    fd = os.open(filename, flags, LOG_FILE_MODE)
    return open(fd, mode, encoding=encoding, errors=errors)


class PrivateFileHandler(logging.FileHandler):
    """A FileHandler whose file is created with mode 0600."""

    def _open(self) -> Any:
        return _open_private(self.baseFilename, self.mode, self.encoding, self.errors)


class PrivateTimedRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    """A TimedRotatingFileHandler whose files, rotated ones included, are mode 0600.

    Rotation renames the current file (keeping its mode) and opens a new one
    through ``_open``.
    """

    def _open(self) -> Any:
        return _open_private(self.baseFilename, self.mode, self.encoding, self.errors)


class SharedRotatingFileHandler(logging.Handler):
    """A log file every simul process on the machine appends to and rotates together.

    Each server process writes the same ``simul.log`` and ``audit.jsonl``.
    Python's TimedRotatingFileHandler keeps its file open and rotates on its
    own schedule, so after the first midnight every process but the one that
    rotated went on writing into the previous day's file, and the retention
    of the next rotation deleted those rows. This handler holds a lock on
    ``<file>.lock`` for each record, rotates under it, and appends to the file
    at its path opened afresh (``O_APPEND``, mode 0600), so no process writes
    to a file another one has rotated away. Under the lock, before writing:

    * a file last written on an earlier day is renamed to
      ``<file>.<YYYY-MM-DD>`` (the day it was written; the names
      TimedRotatingFileHandler used);
    * a file that has reached ``maxBytes`` (when set) is renamed to
      ``<file>.<YYYY-MM-DD>`` or, when that is taken, ``<file>.<YYYY-MM-DD>.<n>``;

    and copies older than ``backupCount`` days are deleted, as are the oldest
    beyond ``maxBackups`` copies (when set).

    Nothing about the lock or the rotation can lose a record: it is appended
    unlocked and unrotated instead, with one warning on stderr, when the lock
    cannot be had (``ENOLCK`` on an NFS home whose lock daemon is gone, a log
    directory where the lock file cannot be created) or another process holds
    it for longer than ``_LOCK_TIMEOUT_SECONDS`` (a stopped or debugged
    process; the handlers of simul.log run on the server's event loop, so a
    long wait would stall it). After either, locking is skipped for
    ``_LOCK_BACKOFF_SECONDS``. A rotation that fails (a copy another program
    holds open on Windows) is skipped the same way.

    Args:
        filename: The log file.
        when: ``"midnight"`` (daily rotation) or anything else (none).
        backupCount: Days of rotated copies kept; 0 keeps them all.
        maxBytes: Size at which the file is rotated; 0 never.
        maxBackups: Most rotated copies kept; 0 no limit.
        encoding: Text encoding of the records.
    """

    _LOCK_TIMEOUT_SECONDS = 0.05
    _LOCK_BACKOFF_SECONDS = 30.0

    def __init__(
        self,
        filename: str,
        when: str = "midnight",
        backupCount: int = 14,
        maxBytes: int = 0,
        maxBackups: int = 0,
        encoding: Optional[str] = "utf-8",
        errors: Optional[str] = None,
        **_unused: Any,
    ) -> None:
        super().__init__()
        self.baseFilename = os.path.abspath(os.fspath(Path(filename).expanduser()))
        self.daily = str(when).lower() == "midnight"
        self.backupCount = max(0, int(backupCount))
        self.maxBytes = max(0, int(maxBytes))
        self.maxBackups = max(0, int(maxBackups))
        self.encoding = encoding or "utf-8"
        self.errors = errors
        self._unlocked_until = 0.0
        self._warned = False
        self._rotated_name = re.compile(
            re.escape(os.path.basename(self.baseFilename))
            + r"\.(\d{4}-\d{2}-\d{2})(?:\.(\d+))?$"
        )

    @staticmethod
    def _day(timestamp: float) -> date:
        """The local day a timestamp falls on (a test may make days shorter)."""
        return date.fromtimestamp(timestamp)

    def _today(self) -> date:
        return self._day(time.time())

    def emit(self, record: logging.LogRecord) -> None:
        try:
            data = (self.format(record) + "\n").encode(self.encoding, self.errors or "strict")
            with contextlib.ExitStack() as held:
                if self._lock(held):
                    try:
                        self._rotate(len(data))
                    except OSError as exc:
                        self._warn_once(f"could not rotate it ({exc}); appending to it as it is")
                self._append(data)
        except Exception:
            self.handleError(record)

    def _lock(self, held: contextlib.ExitStack) -> bool:
        """Take the cross-process lock into ``held``; False when it cannot be had now."""
        if time.monotonic() < self._unlocked_until:
            return False
        try:
            held.enter_context(
                exclusive_file_lock(
                    Path(self.baseFilename + ".lock"), timeout=self._LOCK_TIMEOUT_SECONDS
                )
            )
            return True
        except TimeoutError:
            reason = "another process holds its lock"
        except OSError as exc:
            reason = f"its lock is not available ({exc})"
        self._unlocked_until = time.monotonic() + self._LOCK_BACKOFF_SECONDS
        self._warn_once(f"{reason}; appending to it without the lock or rotation")
        return False

    def _warn_once(self, message: str) -> None:
        """Say once on stderr that the file is written without its lock or rotation."""
        if self._warned:
            return
        self._warned = True
        try:
            sys.stderr.write(f"simul: {self.baseFilename}: {message}\n")
        except Exception:  # pragma: no cover - stderr closed
            pass

    def _append(self, data: bytes) -> None:
        fd = os.open(self.baseFilename, os.O_WRONLY | os.O_CREAT | os.O_APPEND, LOG_FILE_MODE)
        try:
            os.write(fd, data)
        finally:
            os.close(fd)

    def _rotate(self, incoming: int) -> None:
        try:
            info = os.stat(self.baseFilename)
        except FileNotFoundError:
            return
        today = self._today()
        written = self._day(info.st_mtime)
        if self.daily and written < today:
            self._move(written)
        elif self.maxBytes and info.st_size and info.st_size + incoming > self.maxBytes:
            self._move(today)
        else:
            return
        self._prune(today)

    def _move(self, day: date) -> None:
        target = f"{self.baseFilename}.{day:%Y-%m-%d}"
        index = 0
        while os.path.exists(target if index == 0 else f"{target}.{index}"):
            index += 1
        os.replace(self.baseFilename, target if index == 0 else f"{target}.{index}")

    def _prune(self, today: date) -> None:
        directory = os.path.dirname(self.baseFilename)
        copies = []
        for name in os.listdir(directory):
            match = self._rotated_name.fullmatch(name)
            if match is None:
                continue
            try:
                day = date.fromisoformat(match.group(1))
            except ValueError:
                continue
            copies.append((day, int(match.group(2) or 0), os.path.join(directory, name)))
        copies.sort()
        if self.backupCount:
            oldest = today - timedelta(days=self.backupCount)
            for day, _index, path in [copy for copy in copies if copy[0] < oldest]:
                _remove_quietly(path)
            copies = [copy for copy in copies if copy[0] >= oldest]
        if self.maxBackups and len(copies) > self.maxBackups:
            for _day, _index, path in copies[: len(copies) - self.maxBackups]:
                _remove_quietly(path)


def _remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


#: Size at which ``audit.jsonl`` is rotated within a day, and the most
#: rotated copies kept (with ``retention_days`` of daily ones).
AUDIT_MAX_BYTES = 10 * 1024 * 1024
AUDIT_MAX_BACKUPS = 30

# The handler classes the YAML names and their private counterparts.
_PRIVATE_HANDLER_CLASSES = {
    "logging.FileHandler": "simul.logging.PrivateFileHandler",
    "logging.handlers.TimedRotatingFileHandler": "simul.logging.SharedRotatingFileHandler",
    "simul.logging.PrivateTimedRotatingFileHandler": "simul.logging.SharedRotatingFileHandler",
}


def _ensure_log_directories(config: Dict[str, Any]) -> None:
    """Ensure all log directories exist; expand ~ in handler filenames.

    Python's RotatingFileHandler does not call ``expanduser`` on the path it
    receives, so a filename like ``~/.simul/logs/simul.log`` would create
    a literal ``~`` directory in the CWD. We expand here and write the
    resolved absolute path back into the dictConfig payload. Directories are
    created 0700 and the handlers create their files 0600 (see
    ``make_private_dir`` and ``PrivateTimedRotatingFileHandler``); existing
    log files simul writes are tightened to 0600.
    """
    handlers = config.get("handlers", {})

    for handler_config in handlers.values():
        if "filename" not in handler_config:
            continue
        handler_class = handler_config.get("class")
        if handler_class in _PRIVATE_HANDLER_CLASSES:
            handler_config["class"] = _PRIVATE_HANDLER_CLASSES[handler_class]
        log_file = Path(handler_config["filename"]).expanduser()
        handler_config["filename"] = str(log_file)
        try:
            make_private_dir(log_file.parent)
        except Exception as e:
            print(f"Warning: Could not create log directory {log_file.parent}: {e}", file=sys.stderr)
            continue
        tighten_log_files(log_file)


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
        if cfg.get("class") in (
            "logging.handlers.TimedRotatingFileHandler",
            "simul.logging.PrivateTimedRotatingFileHandler",
            "simul.logging.SharedRotatingFileHandler",
        ):
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

    file_handler = SharedRotatingFileHandler(
        filename=str(audit_path),
        when="midnight",
        backupCount=retention_days,
        maxBytes=AUDIT_MAX_BYTES,
        maxBackups=AUDIT_MAX_BACKUPS,
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
    make_private_dir(audit_path.parent)
    tighten_log_files(audit_path)

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

    _apply_level(config, settings.logging.level)
    _apply_format(config, settings.logging.format)

    component_levels = settings.logging.components or {}
    for logger_name, level in component_levels.items():
        config.setdefault("loggers", {}).setdefault(logger_name, {})["level"] = level


#: Values ``logging.format`` takes, and the formatter each puts on the console
#: and the text log file (``detailed`` keeps the YAML's own).
LOG_FORMATS: Dict[str, Optional[Dict[str, str]]] = {
    "detailed": None,
    "simple": {"console": "simple", "console_simple": "simple", "file": "simple"},
    "json": {"console": "json", "console_simple": "json", "file": "json"},
}

_CONSOLE_HANDLERS = ("console", "console_simple")


def _numeric_level(level: Any, default: int = logging.INFO) -> int:
    value = logging.getLevelName(str(level).upper())
    return value if isinstance(value, int) else default


def _apply_level(config: Dict[str, Any], level: str) -> None:
    """Make ``logging.level`` (``LOGGING__LEVEL``, ``--log-level``) the level simul logs at.

    The YAML pins its loggers at INFO, so the setting never reached them:
    ERROR still wrote INFO lines and DEBUG wrote none. Now simul's loggers and
    the console handlers take the level. The root logger and third-party
    loggers (fastmcp, mcp, asyncio, ...) keep their quieter YAML level unless
    the setting is stricter, so DEBUG for simul does not turn on every
    library's debug output, while ERROR silences them too. The audit logger
    is not a log level matter and stays at INFO.
    """
    numeric = _numeric_level(level)
    name = logging.getLevelName(numeric)
    loggers = config.setdefault("loggers", {})
    scopes = list(loggers.items())
    if isinstance(config.get("root"), dict):
        scopes.append(("<root>", config["root"]))
    for logger_name, logger_config in scopes:
        if not isinstance(logger_config, dict) or logger_name == _AUDIT_LOGGER_NAME:
            continue
        if logger_name == "simul" or logger_name.startswith("simul."):
            logger_config["level"] = name
        elif numeric > _numeric_level(logger_config.get("level", "WARNING"), logging.WARNING):
            logger_config["level"] = name
    for handler_name in _CONSOLE_HANDLERS:
        handler = config.get("handlers", {}).get(handler_name)
        if isinstance(handler, dict):
            handler["level"] = name


def _apply_format(config: Dict[str, Any], log_format: str) -> None:
    """Apply ``logging.format`` (``LOGGING__FORMAT``): ``detailed``, ``simple`` or ``json``.

    ``json`` writes the console and the text log as one JSON object per line
    (the ``JsonFormatter`` schema), ``simple`` as ``LEVEL - logger - message``.
    An unknown value is reported on stderr and leaves the YAML's formatters.
    """
    key = str(log_format or "detailed").strip().lower()
    if key not in LOG_FORMATS:
        print(
            f"Warning: unknown logging.format {log_format!r}; "
            f"use one of {', '.join(LOG_FORMATS)}. Keeping the default.",
            file=sys.stderr,
        )
        return
    choice = LOG_FORMATS[key]
    if choice is None:
        return
    formatters = config.setdefault("formatters", {})
    formatters.setdefault("json", {"()": "simul.logging.JsonFormatter"})
    formatters.setdefault("simple", {"format": "%(levelname)s - %(name)s - %(message)s"})
    handlers = config.get("handlers", {})
    for handler_name, formatter in choice.items():
        handler = handlers.get(handler_name)
        if isinstance(handler, dict):
            handler["formatter"] = formatter


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
        make_private_dir(log_dir)
        tighten_log_files(log_path)
        handlers.append(PrivateFileHandler(str(log_path), encoding="utf-8"))

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
    error_type: Optional[str] = None,
    agent_id: Optional[str] = None,
    session: Optional[str] = None,
    reason: Optional[str] = None,
    executed: Optional[Union[bool, str]] = None,
) -> None:
    """Emit one audit JSONL record describing a completed tool-call boundary.

    Every text field is cut at ``AUDIT_MAX_FIELD_CHARS``.

    Args:
        tool: The tool called.
        request_id: The call's correlation id, shared with its usage record.
        duration_ms: Wall time of the whole call, refusals included.
        status: ``ok``, ``error`` (the call failed, ``error_type`` says how),
            or ``cancelled`` (the client gave up).
        error_class: The exception class, when one escaped the tool.
        extra: Free-form context.
        error_type: The failed envelope's ``error_type``.
        agent_id: The agent the call acted as (its claim's label, else the
            session's).
        session: The session label the call came in on.
        reason: Why the envelope refused the call before it ran
            (``rate_limited``, ``instance_busy``, ``instance_claimed``).
        executed: For a cancelled call, whether the engine ran it anyway
            (True, False or ``"unknown"``).
    """
    audit_logger = logging.getLogger(_AUDIT_LOGGER_NAME)
    payload: Dict[str, Any] = {
        "ts": _utc_now_iso(),
        "request_id": _bounded(request_id),
        "tool": _bounded(tool),
        "duration_ms": round(duration_ms, 3),
        "status": _bounded(status),
    }
    for key, value in (
        ("error_type", error_type),
        ("error_class", error_class),
        ("reason", reason),
        ("agent_id", agent_id),
        ("session", session),
    ):
        if value:
            payload[key] = _bounded(str(value))
    if executed is not None:
        payload["executed"] = executed if isinstance(executed, bool) else _bounded(str(executed))
    if extra:
        payload["extra"] = extra
    audit_logger.info("", extra={"audit": payload})


#: Longest value an audit row keeps in a field (a tool name a client made
#: up, an agent label); longer ones are cut, saying how much was left out.
AUDIT_MAX_FIELD_CHARS = 128


def _bounded(value: str) -> str:
    if len(value) <= AUDIT_MAX_FIELD_CHARS:
        return value
    return f"{value[:AUDIT_MAX_FIELD_CHARS]}... <{len(value) - AUDIT_MAX_FIELD_CHARS} more characters>"


# What the envelope reports about the call in progress, for the audit row:
# a dict the middleware sets per call and ``note_call_outcome`` fills in.
_CALL_OUTCOME_VAR: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    "simul_call_outcome", default=None
)


def current_request_id() -> Optional[str]:
    """Return the correlation id of the tool call in progress, or None outside one."""
    request_id = _REQUEST_ID_VAR.get()
    return None if request_id == "-" else request_id


def note_call_outcome(**fields: Any) -> None:
    """Tell the audit row of the call in progress how it ended.

    Called by the MCP server's envelope with ``status``, ``error_type``,
    ``agent_id``, ``session`` and ``reason``; a no-op outside a tool call.
    """
    outcome = _CALL_OUTCOME_VAR.get()
    if outcome is not None:
        outcome.update({key: value for key, value in fields.items() if value is not None})


def _outcome_of_result(result: Any) -> Dict[str, Any]:
    """Read the status of a tool result the envelope did not describe (an unknown tool)."""
    if not getattr(result, "is_error", False):
        return {"status": "ok"}
    outcome: Dict[str, Any] = {"status": "error"}
    for block in reversed(list(getattr(result, "content", None) or [])):
        text = getattr(block, "text", None)
        if not isinstance(text, str):
            continue
        try:
            payload = json.loads(text)
        except ValueError:
            break
        if isinstance(payload, dict) and isinstance(payload.get("error_type"), str):
            outcome["error_type"] = payload["error_type"]
        break
    return outcome


@contextlib.contextmanager
def _tool_call_context(tool_name: str) -> Iterator[Dict[str, Any]]:
    """Scope one tool call: fresh request id, ContextVars, timing and audit row.

    Generates a 12-char hex correlation id, sets the ``request_id`` and
    ``tool_name`` ContextVars for the duration of the block, measures
    wall-time and emits one audit record on exit (success or failure). The
    ContextVars are reset on every path, including ``BaseException``, so a
    stale id never leaks into later log records.

    Yields the call's outcome dict: the envelope fills it through
    ``note_call_outcome`` (``status``, ``error_type``, ``agent_id``,
    ``session``, ``reason``) and the caller may add the ``result`` to read a
    status from when the envelope never ran. ``request_id`` is in it too.
    """
    request_id = uuid.uuid4().hex[:12]
    outcome: Dict[str, Any] = {"request_id": request_id}
    # A made-up tool name can be any length; records and log lines keep a cut one.
    tool_name = _bounded(str(tool_name))
    rid_token = _REQUEST_ID_VAR.set(request_id)
    tn_token = _TOOL_NAME_VAR.set(tool_name)
    outcome_token = _CALL_OUTCOME_VAR.set(outcome)
    start = time.monotonic()
    error_class: Optional[str] = None
    try:
        yield outcome
    except BaseException as exc:
        error_class = type(exc).__name__
        raise
    finally:
        duration_ms = (time.monotonic() - start) * 1000.0
        try:
            if error_class == "CancelledError":
                status = "cancelled"
            elif error_class is not None:
                status = "error"
            elif "status" in outcome:
                status = str(outcome["status"])
            else:
                derived = _outcome_of_result(outcome.get("result"))
                status = derived["status"]
                outcome.setdefault("error_type", derived.get("error_type"))
            if status == "cancelled" and "executed" not in outcome:
                outcome["executed"] = "unknown"
            try:
                emit_audit(
                    tool=tool_name,
                    request_id=request_id,
                    duration_ms=duration_ms,
                    status=status,
                    error_class=error_class,
                    error_type=outcome.get("error_type"),
                    agent_id=outcome.get("agent_id"),
                    session=outcome.get("session"),
                    reason=outcome.get("reason"),
                    executed=outcome.get("executed"),
                )
            except Exception as exc:  # pragma: no cover - defensive
                # The audit row never fails the call it describes.
                _logger.warning("Could not write the audit row of %s: %s", _bounded(tool_name), exc)
        finally:
            _CALL_OUTCOME_VAR.reset(outcome_token)
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
        * the row records how the call really ended: the envelope reports
          its status, ``error_type`` and agent through ``note_call_outcome``,
          and a result the envelope never saw is read for ``isError``
    """
    from fastmcp.server.middleware import Middleware

    class RequestContextMiddleware(Middleware):
        """Tag every ``CallTool`` request with a fresh correlation id + audit row."""

        async def on_call_tool(self, context: Any, call_next: Any) -> Any:
            tool_name = getattr(context.message, "name", "tool")
            with _tool_call_context(tool_name) as outcome:
                result = await call_next(context)
                outcome["result"] = result
                return result

    return RequestContextMiddleware()


# Module-level logger for this module
_logger = get_logger(__name__)
