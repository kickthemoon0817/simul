"""
CLI interface for Simul.

Top-level app ``simul`` with backend-specific sub-commands::

    simul server          # start MCP server
    simul info            # show capabilities
    simul version         # show version
    simul validate-config # validate YAML config
    simul commands        # list all CLI commands (JSON manifest)
    simul tools           # list the MCP tools the server registers
    simul isaac ping      # check Isaac Sim connectivity
    simul isaac status    # instance status
    simul isaac scene     # scene overview
    simul isaac exec ...  # execute Python script
    simul isaac start     # play simulation
    simul usd info <f>    # analyse USD file
    simul usd validate <f>
    simul usd summary <f>

Global ``--json`` flag forces structured JSON output on stdout for all
commands.  When stdout is not a TTY the CLI auto-detects JSON mode.
"""

import asyncio
import errno
import faulthandler
import json
import os
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, NoReturn, Optional, Set, Tuple

import click
import typer
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from typer.core import TyperGroup

from simul.cli.output import emit, emit_error, fail, is_json_mode, set_json_mode
from simul.config import ConfigError, Settings, get_settings, inspect_config_file, load_settings
from simul.tool_surfaces import THIN_BLENDER_TOOLS, THIN_UNREAL_TOOLS, TOOL_SURFACES
from simul.logging import flush_logs, get_logger, setup_logging
from simul.utils.paths import state_dir

from simul.cli.isaac import app as isaac_app
from simul.cli.usd_cli import app as usd_app
from simul.cli.unreal_cli import app as unreal_app
from simul.cli.blender_cli import app as blender_app

# The MCP server (fastmcp + every backend adapter) and the adapter availability
# probes (pxr, bpy) are imported inside the commands that use them, so
# ``simul --help`` and the light sub-commands don't pay for them at startup.


def start_mcp_server(*args, **kwargs):
    """Lazily forward to :func:`simul.mcp.server.start_mcp_server`.

    A module-level name (rather than an import inside ``server``) so tests can
    monkeypatch ``simul.cli.main.start_mcp_server``.
    """
    from simul.mcp.server import start_mcp_server as _start_mcp_server

    return _start_mcp_server(*args, **kwargs)


def _is_isaac_reachable(host: str, port: int, timeout: float = 1.0) -> bool:
    """
    Check if Isaac Sim is reachable on the TCP socket.

    Args:
        host: Isaac Sim host address.
        port: TCP port for the VS Code extension socket.
        timeout: Connection timeout in seconds.

    Returns:
        True if a TCP connection succeeds.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (ConnectionRefusedError, OSError, TimeoutError):
        return False


# ---------------------------------------------------------------------------
# Top-level app with global --json callback
# ---------------------------------------------------------------------------
def _global_callback(
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output structured JSON to stdout (auto-enabled when stdout is not a TTY)",
    ),
) -> None:
    """Global options applied before any sub-command."""
    if json_output:
        set_json_mode(True)


def config_error_exit(exc: ConfigError) -> NoReturn:
    """Report settings that cannot be used and exit 2.

    The message names each bad setting, where its value came from and why it
    was refused; under ``--json`` the envelope carries them as
    ``details.errors``.
    """
    fail(exc.message, "ConfigError", {"errors": exc.errors}, exit_code=2)


class _SimulGroup(TyperGroup):
    """The root command group: a ``ConfigError`` from any command exits 2.

    Every command loads the settings, and one bad value (``UNREAL__PORT=70000``)
    used to end each of them in a Rich traceback with exit 1, even under
    ``--json``. Commands raise ``ConfigError`` and this reports it once, the
    same way for all of them; ``--help`` loads no settings and keeps working.
    """

    def invoke(self, ctx: click.Context) -> Any:
        try:
            return super().invoke(ctx)
        except ConfigError as exc:
            config_error_exit(exc)


app = typer.Typer(
    name="simul",
    help="Simul -- agent toolkit for 3D simulation (MCP server, engine setup, bridges)",
    add_completion=False,
    callback=_global_callback,
    cls=_SimulGroup,
)
console = Console(stderr=True)

# Register sub-apps
app.add_typer(isaac_app, name="isaac", help="Isaac Sim commands")
app.add_typer(usd_app, name="usd", help="USD file commands (headless)")
app.add_typer(blender_app, name="blender", help="Attach to existing Blender windows")
app.add_typer(
    unreal_app, name="unreal", help="Unreal Engine commands (Remote Control API)"
)


# ---------------------------------------------------------------------------
# logs sub-app: pretty-print structured / audit log streams
# ---------------------------------------------------------------------------
logs_app = typer.Typer(name="logs", help="Inspect simul log files")
app.add_typer(logs_app, name="logs")


def _resolve_log_path(explicit: Optional[Path], audit: bool, structured: bool) -> Path:
    """Pick the right default log path when the user did not pass one.

    ``audit`` -> logging.audit_path ($SIMUL_HOME/logs/audit.jsonl by default)
    ``structured`` -> the structured JSON file (newest matching basename)
    otherwise -> human-readable simul.log (newest matching basename)
    """
    if explicit is not None:
        return explicit.expanduser()

    settings = get_settings()
    if audit:
        return Path(settings.logging.audit_path).expanduser()

    base = Path(settings.logging.file_path).expanduser()
    suffix = ".json" if structured else (base.suffix or ".log")
    parent = base.parent
    stem = base.stem if base.suffix else base.name

    # Newest matching file (per_instance produces ``<stem>.<pid><suffix>``).
    candidates = sorted(
        parent.glob(f"{stem}*{suffix}"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    return candidates[0] if candidates else parent / f"{stem}{suffix}"


def _format_jsonl_line(raw: str, tool_filter: Optional[str]) -> Optional[str]:
    """Render one JSONL line as a coloured human-readable string.

    Returns ``None`` when the row is filtered out or the line is not JSON
    (in which case the caller should print the raw line as-is).
    """
    raw = raw.rstrip("\n")
    if not raw:
        return ""
    try:
        record = json.loads(raw)
    except (ValueError, TypeError):
        return raw  # Plain text line — already human readable.

    tool = record.get("tool") or record.get("tool_name") or "-"
    if tool_filter and tool != tool_filter:
        return None

    ts = record.get("ts") or record.get("timestamp") or "-"
    level = record.get("level") or ("AUDIT" if "duration_ms" in record else "INFO")
    request_id = record.get("request_id") or "-"

    # Audit row shape: tool + duration + status
    if "duration_ms" in record:
        status = record.get("status", "?")
        ms = record["duration_ms"]
        err = record.get("error_class")
        colour = "red" if status == "error" else "green"
        msg = f"[{colour}]{status:5s}[/{colour}] {ms:>8.2f}ms"
        if err:
            msg += f"  [red]{err}[/red]"
        return f"[dim]{ts}[/dim] [bold]{request_id}[/bold] [cyan]{tool}[/cyan]  {msg}"

    # General log row
    level_colour = {
        "ERROR": "red",
        "CRITICAL": "red",
        "WARNING": "yellow",
        "INFO": "green",
        "DEBUG": "blue",
    }.get(level, "white")
    msg = record.get("msg") or record.get("message") or ""
    return (
        f"[dim]{ts}[/dim] [{level_colour}]{level:<7}[/{level_colour}] "
        f"[bold]{request_id}[/bold] [cyan]{tool}[/cyan] {msg}"
    )


@logs_app.command("paths")
def logs_paths() -> None:
    """Print the resolved log file paths for the current settings."""
    settings = get_settings()
    paths = {
        "main": _resolve_log_path(None, audit=False, structured=False),
        "structured_json": _resolve_log_path(None, audit=False, structured=True),
        "audit": _resolve_log_path(None, audit=True, structured=False),
        "errors": Path(settings.logging.file_path)
        .expanduser()
        .with_name(Path(settings.logging.file_path).stem + "_errors.log"),
    }
    if is_json_mode():
        emit({k: str(v) for k, v in paths.items()})
        return
    for label, path in paths.items():
        exists = "[green]exists[/green]" if path.exists() else "[dim]missing[/dim]"
        console.print(f"[bold]{label:16s}[/bold] {path}  {exists}")


@logs_app.command("tail")
def logs_tail(
    file: Optional[Path] = typer.Option(
        None, "--file", "-f", help="Log file path (default: latest matching log)"
    ),
    lines: int = typer.Option(
        50, "--lines", "-n", help="Number of trailing lines to show"
    ),
    follow: bool = typer.Option(
        False, "--follow", help="Stream new lines as they arrive (Ctrl-C to stop)"
    ),
    audit: bool = typer.Option(False, "--audit", help="Tail the audit JSONL stream"),
    structured: bool = typer.Option(
        False, "--json-log", help="Tail the structured JSON file (vs. text log)"
    ),
    tool: Optional[str] = typer.Option(
        None, "--tool", help="Only show rows for this tool name"
    ),
) -> None:
    """Pretty-print the trailing portion of a simul log file."""
    path = _resolve_log_path(file, audit=audit, structured=structured or audit)
    if not path.exists():
        emit_error(f"Log file not found: {path}", error_type="LogFileNotFound")
        return

    # Show the last N lines first (efficient for typical log sizes).
    with path.open("r", encoding="utf-8", errors="replace") as fp:
        tail = fp.readlines()[-lines:]
        for raw in tail:
            rendered = _format_jsonl_line(raw, tool)
            if rendered is None:
                continue
            console.print(rendered)

        if not follow:
            return

        # Streaming mode: keep reading appended bytes.
        try:
            while True:
                line = fp.readline()
                if not line:
                    time.sleep(0.25)
                    continue
                rendered = _format_jsonl_line(line, tool)
                if rendered is None:
                    continue
                console.print(rendered)
        except KeyboardInterrupt:
            return


# ---------------------------------------------------------------------------
# Backend / tool-surface options shared by server, info and tools
# ---------------------------------------------------------------------------
def _backends_option() -> Any:
    return typer.Option(
        None,
        "--backends",
        "-b",
        help=(
            "Comma-separated backends to enable (isaac,unreal,usd,blender). Default: all available, "
            "Isaac Sim only on Linux and Windows; naming isaac registers it on any OS."
        ),
    )


def _unreal_tools_option() -> Any:
    return typer.Option(
        None,
        "--unreal-tools",
        help=(
            f"Unreal MCP tool surface: 'thin' (default: {', '.join(THIN_UNREAL_TOOLS)}) or 'full' "
            "(every granular tool). Overrides unreal.tool_surface / UNREAL__TOOL_SURFACE."
        ),
    )


def _blender_tools_option() -> Any:
    return typer.Option(
        None,
        "--blender-tools",
        help=(
            "Blender MCP tool surface: 'full' (default: every Blender and SimReady tool) or 'thin' "
            f"({', '.join(THIN_BLENDER_TOOLS)}). Overrides blender.tool_surface / BLENDER__TOOL_SURFACE."
        ),
    )


def _unreal_mode_option() -> Any:
    return typer.Option(
        None, "--unreal-mode", help="Unreal connection: endpoint or attached (explicitly selected editor)"
    )


def _blender_mode_option() -> Any:
    return typer.Option(
        None,
        "--blender-mode",
        help="Blender connection: embedded (local bpy) or attached (selected existing window)",
    )


def _option_error(message: str) -> NoReturn:
    """Report an invalid option value: exit 2 with ``InvalidArgument``, as docs/cli.md says.

    Under ``--json`` the envelope goes to stdout, otherwise the message to stderr.
    """
    fail(message, "InvalidArgument", exit_code=2)


# The levels LoggingConfig accepts, for --log-level.
_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def _with_section(settings: Settings, section: str, *, flag: Optional[str] = None, **update: Any) -> Settings:
    """Return ``settings`` with fields of one frozen section replaced and re-validated.

    ``model_copy(update=...)`` skips validation, so a flag could put a value
    into the settings that the same value from the environment is refused for.

    Args:
        settings: The loaded settings.
        section: The section to change (``logging``).
        flag: The option the values came from, for the error message.
        **update: The fields to replace.

    Returns:
        The new settings.
    """
    current = getattr(settings, section)
    try:
        replaced = type(current).model_validate({**current.model_dump(), **update})
    except ValidationError as exc:
        reasons = "; ".join(problem.get("msg", "invalid") for problem in exc.errors())
        _option_error(f"{flag or section}: {reasons}")
    return settings.model_copy(update={section: replaced})


def _apply_backend_options(
    settings: Settings,
    backends: Optional[str] = None,
    unreal_tools: Optional[str] = None,
    blender_tools: Optional[str] = None,
    unreal_mode: Optional[str] = None,
    blender_mode: Optional[str] = None,
) -> Tuple[Settings, Optional[Set[str]]]:
    """Validate the backend options and fold them into the settings.

    Args:
        settings: Settings loaded from config and the environment.
        backends: ``--backends`` value, comma-separated.
        unreal_tools: ``--unreal-tools`` value.
        blender_tools: ``--blender-tools`` value.
        unreal_mode: ``--unreal-mode`` value.
        blender_mode: ``--blender-mode`` value.

    Returns:
        The updated settings and the selected backend names (None for all).
    """
    choices = (
        ("--unreal-mode", "unreal", "mode", unreal_mode, ("endpoint", "attached")),
        ("--blender-mode", "blender", "mode", blender_mode, ("embedded", "attached")),
        ("--unreal-tools", "unreal", "tool_surface", unreal_tools, TOOL_SURFACES),
        ("--blender-tools", "blender", "tool_surface", blender_tools, TOOL_SURFACES),
    )
    for flag, section, field_name, raw, valid in choices:
        if raw is None:
            continue
        value = raw.strip().lower()
        if value not in valid:
            _option_error(f"Unknown {flag} value: {raw!r}. Valid: {', '.join(valid)}")
        settings = _with_section(settings, section, flag=flag, **{field_name: value})

    backend_set: Optional[Set[str]] = None
    if backends is not None:
        from simul.mcp.backends import ALL_BACKEND_NAMES

        backend_set = {b.strip().lower() for b in backends.split(",") if b.strip()}
        if not backend_set:
            # An empty selection must not silently mean "every backend".
            _option_error(
                f"--backends names no backend: {backends!r}. "
                f"Valid: {', '.join(sorted(ALL_BACKEND_NAMES))}"
            )
        unknown = backend_set - ALL_BACKEND_NAMES
        if unknown:
            _option_error(
                f"Unknown backends: {', '.join(sorted(unknown))}. "
                f"Valid: {', '.join(sorted(ALL_BACKEND_NAMES))}"
            )
    return settings, backend_set


def _is_enabled(backend_set: Optional[Set[str]], name: str) -> bool:
    """Return True when ``name`` is selected (every backend is when none were named)."""
    return backend_set is None or name in backend_set


# ---------------------------------------------------------------------------
# server
# ---------------------------------------------------------------------------
def _config_option() -> Any:
    """``--config FILE``; a file that is missing or does not load exits 2 with ``ConfigError``."""
    return typer.Option(
        None,
        "--config",
        "-c",
        help=(
            "YAML configuration file (default: $CONFIG_FILE, else the packaged default.yaml). "
            "A missing file, invalid YAML or an invalid value exits 2."
        ),
        dir_okay=False,
    )


def _load(config: Optional[Path]) -> Settings:
    """The settings from ``--config`` when given, otherwise from ``get_settings``."""
    return load_settings(config) if config else get_settings()


def _keep_fastmcp_home_in_state_dir() -> None:
    """Point FastMCP's data directory at ``<state dir>/fastmcp`` unless ``FASTMCP_HOME`` names one.

    It defaults to the platform's user data directory (``~/Library/Application
    Support/fastmcp`` on macOS). Nothing simul runs writes there now that the
    banner and its update check are off, but whatever FastMCP keeps follows
    ``SIMUL_HOME`` with the rest of simul's state.
    """
    if os.environ.get("FASTMCP_HOME"):
        return
    import fastmcp

    fastmcp.settings.home = state_dir() / "fastmcp"


#: The signals that stop ``simul server`` cleanly.
_STOP_SIGNALS: Tuple[int, ...] = (signal.SIGTERM, signal.SIGINT)
#: How long a stopping stdio server waits for its task to end before it
#: concludes that the thread reading stdin is what holds it: that thread
#: returns only when the client closes stdin.
_STDIN_RELEASE_SECONDS = 0.25
#: Past ``server.shutdown_timeout``, how long a stopping HTTP/SSE server is
#: given before it is cancelled: uvicorn's own graceful window (FastMCP's
#: ``timeout_graceful_shutdown``, 2 s) and a margin.
_NETWORK_STOP_MARGIN_SECONDS = 5.0
#: Past the shutdown deadline, how long a worker thread may keep the
#: interpreter lock before the process is ended without it (exit 1).
_WORKER_LOCK_GRACE_SECONDS = 1.0


class _StopSignals:
    """Record SIGTERM/SIGINT from the start of ``simul server`` until its loop handles them.

    Before this, a signal in the second or so between the command starting and
    the server's loop installing its handlers ended the process the default
    way (exit -15, or a KeyboardInterrupt traceback). Installed first thing in
    ``server``; ``_serve`` then takes the signals over and stops at once when
    one already came. A signal that arrives while Python is still importing
    the CLI, before ``server`` runs, ends it the default way: nothing has been
    opened by then.

    Once ``_serve`` has stopped the server (``stopped``), the handler is
    back in place until the command returns, and a signal then ends the
    process at once instead of being recorded: exit 0 when it is the first
    (shutdown has run), ``128 + signal`` when a stop signal came before it.
    ``server`` is the stopped ``SimulMCPServer``, whose calls still in
    flight a forced exit records.
    """

    def __init__(self) -> None:
        self.received: List[int] = []
        self.originals: Dict[int, Any] = {}
        self.stopped = False
        self.signalled = False
        self.server: Any = None

    def _record(self, signum: int, _frame: Any) -> None:
        if self.stopped:
            if self.signalled or self.received:
                _force_exit(signum, self.server)
            _exit_now(signum, self.server)
        self.received.append(signum)

    def __enter__(self) -> "_StopSignals":
        for signum in _STOP_SIGNALS:
            try:
                self.originals[signum] = signal.signal(signum, self._record)
            except (ValueError, OSError):
                pass  # not the main thread
        return self

    def __exit__(self, *_exc: Any) -> None:
        for signum, original in self.originals.items():
            try:
                signal.signal(signum, original)
            except (ValueError, OSError, TypeError):
                pass


async def _finished(task: "asyncio.Future[Any]", timeout: float) -> bool:
    """Whether ``task`` ends within ``timeout`` seconds (it is not cancelled)."""
    done, _ = await asyncio.wait({task}, timeout=max(0.0, timeout))
    return bool(done)


def _signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return f"signal {signum}"


def _report_stopped(signum: int) -> None:
    """Say on stderr and in the log which signal stopped the server."""
    name = _signal_name(signum)
    get_logger(__name__).info("Stopped on %s", name)
    console.print(f"[yellow]Server stopped ({name})[/yellow]")


def _flush_everything() -> None:
    flush_logs()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # pragma: no cover - a closed stream
            pass


def _exit_now(signum: Optional[int], server: Any = None) -> NoReturn:
    """End the process with exit 0 once a stopped server has shut down.

    For a stdio server whose client keeps stdin open: the thread FastMCP reads
    stdin with returns only when stdin closes, and ``asyncio.run`` would wait
    for it. For a call still running on a worker thread past the shutdown
    deadline: ``asyncio.run`` would wait for that thread. Everything else has
    finished by now (``SimulMCPServer.shutdown`` ran), so the logs are
    flushed and the process exits without those threads.

    Args:
        signum: The signal that stopped the server, reported; None when it
            stopped on its own (stdin closed).
        server: The ``SimulMCPServer``, whose calls not recorded yet are
            recorded ``server_stopped``; None when it was never built.
    """
    if signum is not None:
        _report_stopped(signum)
    # Calls a stop recorded as it cancelled them are not recorded again; one
    # still unrecorded (an Isaac call still waiting on the engine past the
    # deadline) would otherwise never be.
    _record_calls_in_flight(server, "server_stopped")
    _flush_everything()
    os._exit(0)


def _record_calls_in_flight(server: Any, reason: str) -> int:
    """Record the calls still in flight before the process exits without them, ``cancelled`` with ``reason``.

    Returns:
        How many were recorded; 0 when nothing could be.
    """
    try:
        if server is not None:
            return int(server.record_calls_in_flight(reason))
        from simul.logging import record_calls_in_flight

        return record_calls_in_flight(reason)
    except Exception:  # pragma: no cover - the exit must happen whatever this does
        return 0


def _force_exit(signum: int, server: Any = None) -> NoReturn:
    """End the process at once on a second SIGTERM or SIGINT while the server stops.

    Nothing more is waited for: not Isaac Sim, not a worker thread, not the
    transport. Each call still in flight is recorded first, in the usage log
    and the audit log, as ``cancelled`` with ``executed: "unknown"`` and
    reason ``forced_exit``, since nothing after this would write its rows.
    Then the logs are flushed and the exit status is ``128 + signum`` (143
    for SIGTERM, 130 for SIGINT), as for a process the signal ended.

    Args:
        signum: The second signal.
        server: The ``SimulMCPServer``, or None when it was never built.
    """
    name = _signal_name(signum)
    recorded = _record_calls_in_flight(server, "forced_exit")
    get_logger(__name__).warning(
        "%s again while stopping: exiting at once without waiting for the calls still running "
        "(%d recorded cancelled, reason forced_exit)",
        name,
        recorded,
    )
    console.print(f"[yellow]Server stopped at once ({name} again)[/yellow]")
    _flush_everything()
    os._exit(128 + signum)


async def _let_worker_calls_finish(server: Any, signum: Optional[int]) -> None:
    """Wait for calls still running on worker threads, until the shutdown deadline; then exit without them.

    A USD or attached-Blender call runs on a worker thread
    (``simul.mcp.worker_threads``), which nothing can stop, and
    ``asyncio.run`` would wait for it before the process could exit, however
    long it took. It is waited for until ``server.shutdown_timeout`` has
    passed since the stop began; then the log names it and the process exits
    0 without it (``_exit_now``). Its request was cancelled when the server
    stopped, so it is recorded with ``executed: "unknown"``.

    A worker thread inside one long C call (pxr opening a large stage) keeps
    the interpreter lock, and nothing on this thread runs until the call
    returns: the round-3 probe's 900k-prim validation overran the deadline
    by 2 s that way. So a timer that needs no interpreter lock
    (``faulthandler.dump_traceback_later``) ends the process
    ``_WORKER_LOCK_GRACE_SECONDS`` after the deadline if it is still there,
    with exit 1 and every thread's traceback in the errors log
    (``simul_errors.log``; stderr when there is none), after a line saying
    what it is. The log line that starts the wait says so, since nothing
    can be logged once the timer fires.

    Args:
        server: The stopped ``SimulMCPServer``, or None when it was never built.
        signum: The signal that stopped it, or None.
    """
    from simul.mcp import worker_threads

    if not worker_threads.running():
        return
    timeout = float(server.settings.server.shutdown_timeout) if server is not None else 0.0
    deadline = server._shutdown_deadline() if server is not None else time.monotonic()
    remaining = max(0.0, deadline - time.monotonic())
    calls = ", ".join(worker_threads.running())
    dump, where = _thread_dump_target(server, calls, remaining)
    get_logger(__name__).info(
        "Shutdown: waiting up to %.1fs for %s, running on a worker thread. If it still holds Python's "
        "interpreter lock %gs after that (one long pxr call on a large stage does), the process exits 1 "
        "and writes every thread's traceback to %s",
        remaining,
        calls,
        _WORKER_LOCK_GRACE_SECONDS,
        where,
    )
    flush_logs()
    try:
        faulthandler.dump_traceback_later(remaining + _WORKER_LOCK_GRACE_SECONDS, exit=True, file=dump)
        armed = True
    except (AttributeError, OSError, RuntimeError, ValueError):  # no usable file
        armed = False
    try:
        while True:
            names = worker_threads.running()
            if not names:
                return
            if time.monotonic() >= deadline:
                get_logger(__name__).warning(
                    "Shutdown: stopped waiting for %s, still running on a worker thread, after %gs "
                    "(server.shutdown_timeout); exiting without it",
                    ", ".join(names),
                    timeout,
                )
                _exit_now(signum, server)
            await asyncio.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    finally:
        if armed:
            faulthandler.cancel_dump_traceback_later()
        if dump is not sys.stderr:
            try:
                dump.close()
            except OSError:  # pragma: no cover - defensive
                pass


def _thread_dump_target(server: Any, calls: str, remaining: float) -> Tuple[Any, str]:
    """The file the worker-lock backstop writes its thread dump to, and what to call it in the log.

    The errors log (``simul_errors.log``), opened for appending, with a line
    in its format first saying what a dump right after it means; stderr when
    file logging is off or the file cannot be opened. The dump is written
    without the interpreter lock (``faulthandler``), so the line has to be
    written now, before it is known whether the dump comes.

    Args:
        server: The stopped ``SimulMCPServer``, whose settings name the log;
            None when it was never built.
        calls: The calls running on worker threads.
        remaining: Seconds until the shutdown deadline.
    """
    from simul.logging import LOG_FILE_MODE, error_log_path, make_private_dir

    path = error_log_path(server.settings) if server is not None else None
    if path is None:
        return sys.stderr, "stderr"
    line = (
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} [- -] {__name__} - WARNING - main.py - "
        f"_let_worker_calls_finish() - Shutdown: waiting up to {remaining:.1f}s for {calls} on a worker "
        "thread. A thread dump right after this line means it still held Python's interpreter lock "
        f"{_WORKER_LOCK_GRACE_SECONDS:g}s past server.shutdown_timeout, and the process exited 1 without it\n"
    )
    try:
        make_private_dir(path.parent)
        stream = os.fdopen(os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, LOG_FILE_MODE), "a", encoding="utf-8")
    except (OSError, ValueError):
        return sys.stderr, "stderr"
    try:
        stream.write(line)
        stream.flush()
    except OSError:
        stream.close()
        return sys.stderr, "stderr"
    return stream, str(path)


async def _stop_serving(
    serving: "asyncio.Future[Any]", server: Any, transport: str, signum: int, handler: Any
) -> None:
    """Stop a running server after ``signum``; returns once it ended, or exits the process.

    Every wait on Isaac Sim gives up ``server.shutdown_timeout`` after the
    signal (``begin_shutdown``), so the whole stop stays bounded.

    Args:
        serving: The task running the server.
        server: The ``SimulMCPServer``, or None when it is not built yet.
        transport: The server's transport.
        signum: The signal that stopped it.
        handler: The Python-level handler the loop installed for ``signum``;
            another one in its place means uvicorn took the signal over.
    """
    grace = 0.0
    if server is not None:
        server.begin_shutdown()
        grace = float(server.settings.server.shutdown_timeout)
    if transport == "stdio":
        serving.cancel()
        if await _finished(serving, _STDIN_RELEASE_SECONDS):
            return
        if server is not None:
            # Cancelled Isaac calls wait for the engine, bounded by the grace,
            # and record themselves as they end.
            await server.wait_for_draining_calls()
            if await _finished(serving, _STDIN_RELEASE_SECONDS):
                return
            await server.shutdown()
        # Only the thread blocked reading stdin is left.
        _exit_now(signum, server)
    # HTTP/SSE: uvicorn got the same signal (its handler replaced the loop's
    # while it serves) and stops gracefully. A signal that came before it
    # took the signals over (still starting) reached only this loop.
    if signal.getsignal(signum) is handler:
        serving.cancel()
    if await _finished(serving, grace + _NETWORK_STOP_MARGIN_SECONDS):
        return
    serving.cancel()
    if await _finished(serving, 1.0):
        return
    if server is not None:
        await server.shutdown()
    _exit_now(signum, server)


async def _serve(
    settings: Settings,
    transport: str,
    backends: Optional[Set[str]],
    stop_signals: Optional[_StopSignals] = None,
) -> Optional[int]:
    """Run the MCP server until it stops; SIGTERM and SIGINT stop it cleanly.

    Either signal starts the server's shutdown (``begin_shutdown``: waits on
    Isaac Sim end ``server.shutdown_timeout`` later), stops the transport,
    runs ``SimulMCPServer.shutdown`` and exits 0. Over stdio the server task
    is cancelled; when the client keeps stdin open, the thread reading it
    would hold ``asyncio.run`` until stdin closes, so the process exits
    without it (``_exit_now``) once the server has shut down. Over HTTP/SSE
    uvicorn stops gracefully on the same signal; it is cancelled if it is
    still running past the grace.

    The loop's handlers also see a signal uvicorn handles while it serves
    (asyncio's signal wakeup fd stays in place under uvicorn's Python-level
    handler), and the one uvicorn re-raises when it is done.

    Args:
        settings: The server's settings.
        transport: ``stdio``, ``http`` or ``sse``.
        backends: The selected backends, None for every available one.
        stop_signals: The signals ``server`` recorded before the loop ran.

    Returns:
        The signal that stopped the server, or None when it stopped on its own
        (stdin closed).
    """
    loop = asyncio.get_running_loop()
    received: List[int] = []
    signalled = asyncio.Event()
    # uvicorn records each signal its own handler takes while it serves and
    # raises it again once it is done (Server.capture_signals). The loop sees
    # both, and the second is not a new signal: ``taken`` counts the ones
    # uvicorn took, ``repeats`` the ones it has raised again whose loop
    # callback is still to come.
    taken: Dict[int, int] = {}
    repeats: Dict[int, int] = {}
    # Set once the server has shut down and only worker-thread calls are left.
    shut_down: List[bool] = []
    # The server, once start_mcp_server has built it.
    servers: List[Any] = []

    def _python_handler(signum: int, _frame: Any) -> None:
        # The Python-level handler under the loop's (in place of asyncio's
        # no-op). uvicorn puts it back before it raises a signal again, so a
        # repeat reaches it at once, before the loop's callback for it runs.
        if taken.get(signum):
            taken[signum] -= 1
            repeats[signum] = repeats.get(signum, 0) + 1

    def _on_signal(signum: int) -> None:
        if repeats.get(signum):
            repeats[signum] -= 1
            return
        if signal.getsignal(signum) is not _python_handler:
            taken[signum] = taken.get(signum, 0) + 1
        if received:
            # A second signal while stopping ends the process at once.
            _force_exit(signum, servers[0] if servers else None)
        if shut_down:
            _exit_now(signum, servers[0] if servers else None)
        received.append(signum)
        signalled.set()

    handlers: Dict[int, Any] = {}
    previous: Dict[int, Any] = {}
    for signum in _STOP_SIGNALS:
        before = signal.getsignal(signum)
        try:
            loop.add_signal_handler(signum, _on_signal, signum)
        except (NotImplementedError, RuntimeError, ValueError):
            # No loop signal handlers here (Windows, a non-main thread). The
            # recorder keeps SIGTERM from killing an HTTP server uvicorn
            # re-raises it to; SIGINT goes back to KeyboardInterrupt.
            if signum == signal.SIGINT and stop_signals is not None:
                original = stop_signals.originals.get(signum)
                if original is not None:
                    try:
                        signal.signal(signum, original)
                    except (ValueError, OSError, TypeError):
                        pass
            continue
        previous[signum] = before
        try:
            signal.signal(signum, _python_handler)
            # As asyncio leaves it: system calls resume after the signal.
            signal.siginterrupt(signum, False)
        except (ValueError, OSError, AttributeError):
            pass
        handlers[signum] = signal.getsignal(signum)
    # Read only now that the loop has the signals: one that came before is in
    # the recorder, one after reaches _on_signal, and none falls between.
    if stop_signals is not None and stop_signals.received:
        received[:0] = stop_signals.received
        signalled.set()
    try:
        if received:
            # Stopped before the server was built: nothing to shut down.
            return received[0]
        serving = asyncio.ensure_future(
            start_mcp_server(settings, transport, backends=backends, on_server=servers.append)
        )
        stop = asyncio.ensure_future(signalled.wait())
        try:
            await asyncio.wait({serving, stop}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            stop.cancel()
        if not serving.done():
            signum = received[0]
            await _stop_serving(
                serving, servers[0] if servers else None, transport, signum, handlers.get(signum)
            )
        if serving.cancelled() or serving.exception() is None:
            shut_down.append(True)
            await _let_worker_calls_finish(
                servers[0] if servers else None, received[0] if received else None
            )
        if serving.cancelled():
            if not received:
                raise asyncio.CancelledError()
        else:
            serving.result()
    finally:
        if stop_signals is not None:
            # A signal from here on ends the process at once (_StopSignals).
            stop_signals.signalled = bool(received)
            stop_signals.server = servers[0] if servers else None
            stop_signals.stopped = True
        for signum, before in previous.items():
            try:
                loop.remove_signal_handler(signum)
                signal.signal(signum, before)
            except (NotImplementedError, RuntimeError, ValueError, OSError, TypeError):
                pass
    if not received and stop_signals is not None and stop_signals.received:
        received.extend(stop_signals.received)
    return received[0] if received else None


@app.command()
def server(
    config: Optional[Path] = _config_option(),
    transport: str = typer.Option(
        "stdio",
        "--transport",
        "-t",
        help=(
            "Transport: stdio (the client spawns the server), http (streamable HTTP on "
            "server.host:server.port) or sse (legacy network transport)."
        ),
    ),
    backends: Optional[str] = _backends_option(),
    unreal_tools: Optional[str] = _unreal_tools_option(),
    blender_tools: Optional[str] = _blender_tools_option(),
    unreal_mode: Optional[str] = _unreal_mode_option(),
    blender_mode: Optional[str] = _blender_mode_option(),
    log_level: Optional[str] = typer.Option(
        None, "--log-level", "-l", help="Log level: DEBUG, INFO, WARNING, ERROR or CRITICAL"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
    allow_public: bool = typer.Option(
        False,
        "--allow-public",
        help=(
            "Let --transport http/sse bind a non-loopback server.host (0.0.0.0, a LAN "
            "address). Set SERVER__AUTH_TOKEN_FILE too: without a token anyone who can "
            "reach the port can run every tool. Same as SERVER__ALLOW_PUBLIC=true."
        ),
    ),
) -> None:
    """Start the Simul MCP server.

    Only the backends named by --backends are probed at startup; the others
    get no connection attempt and no banner line. Isaac Sim tools register
    only on Linux and Windows unless --backends names isaac or
    ISAAC_SIM__ENABLED=true; the banner says when they are skipped.

    The network transports refuse a non-loopback server.host unless
    --allow-public is passed, check every request's Host and Origin headers
    (the DNS-rebinding defense), and require a bearer token when
    SERVER__AUTH_TOKEN_FILE or SERVER__AUTH_TOKEN is set.

    SIGTERM and SIGINT stop it cleanly over every transport, stdin open or
    not: shutdown runs, the logs are flushed and the command exits 0.
    """
    # First: a signal during startup is kept for _serve rather than killing
    # the process before anything can shut down.
    stop_signals = _StopSignals().__enter__()
    try:
        from simul.mcp.backends import backend_skip_reason
        from simul.mcp.server import NETWORK_TRANSPORTS, TRANSPORTS

        # Flags first: a bad invocation is reported before the settings load.
        transport = transport.strip().lower()
        if transport not in TRANSPORTS:
            _option_error(f"Unknown --transport value: {transport!r}. Valid: {', '.join(TRANSPORTS)}")
        resolved_level = "DEBUG" if verbose else (log_level.strip().upper() if log_level else None)
        if resolved_level is not None and resolved_level not in _LOG_LEVELS:
            _option_error(f"Unknown --log-level value: {log_level!r}. Valid: {', '.join(_LOG_LEVELS)}")

        try:
            settings = _load(config)
        except ConfigError as exc:
            # stdout is the MCP channel over stdio; the error goes to stderr only.
            console.print(f"[red]ConfigError: {escape(exc.message)}[/red]")
            raise typer.Exit(2)

        settings, backend_set = _apply_backend_options(
            settings, backends, unreal_tools, blender_tools, unreal_mode, blender_mode
        )

        # LoggingConfig is frozen on purpose, so rebuild the section rather
        # than assigning into it — assignment raises ValidationError and took
        # both of these flags down with it.
        if resolved_level:
            settings = _with_section(settings, "logging", flag="--log-level", level=resolved_level)

        if allow_public:
            settings = _with_section(settings, "server", flag="--allow-public", allow_public=True)
        http_security = None
        if transport in NETWORK_TRANSPORTS:
            from simul.mcp.http_security import HttpSecurityError, resolve_http_security

            try:
                http_security = resolve_http_security(settings.server)
            except HttpSecurityError as exc:
                fail(str(exc), "InsecureTransport", exit_code=2)

        setup_logging(settings)
        logger = get_logger(__name__)

        backends_label = (
            ", ".join(sorted(backend_set)) if backend_set else "all available"
        )
        lines = [
            "[bold blue]Simul -- agent toolkit for 3D simulation[/bold blue]",
            f"Transport: {transport}",
            f"Backends: {backends_label}",
        ]
        if http_security is not None:
            path = "sse" if transport == "sse" else "mcp"
            shown_host = settings.server.host
            if ":" in shown_host and not shown_host.startswith("["):
                shown_host = f"[{shown_host}]"
            lines.append(f"Listening: http://{escape(shown_host)}:{settings.server.port}/{path}")
            lines += [escape(line) for line in http_security.describe()]
            lines += [f"[bold red]WARNING: {escape(warning)}[/bold red]" for warning in http_security.warnings]
        # Probe only what was selected: each probe is a connection attempt or
        # a runtime import that a per-session stdio spawn pays every time.
        if _is_enabled(backend_set, "isaac"):
            isaac_skipped = backend_skip_reason("isaac", settings, backend_set)
            if isaac_skipped is not None:
                lines.append(f"Isaac Sim: skipped — {escape(isaac_skipped)}")
            else:
                isaac_port = settings.isaac_sim.socket_port
                isaac_reachable = _is_isaac_reachable(
                    settings.isaac_sim.socket_host, isaac_port
                )
                lines.append(
                    f"Isaac Sim (TCP :{isaac_port}): "
                    f"{'reachable' if isaac_reachable else 'not reachable (tools will retry at call time)'}"
                )
        if _is_enabled(backend_set, "unreal"):
            lines.append(f"Unreal tools: {settings.unreal.tool_surface}")
        if _is_enabled(backend_set, "blender"):
            from simul.adapters import is_blender_available

            blender_available = settings.blender.mode == "attached" or is_blender_available()
            lines.append(
                f"Blender ({settings.blender.mode}): "
                f"{'available' if blender_available else 'unavailable'}, "
                f"tools: {settings.blender.tool_surface}"
            )
        if _is_enabled(backend_set, "usd"):
            from simul.adapters import is_headless_available

            lines.append(
                f"USD Headless: {'available' if is_headless_available() else 'unavailable'}"
            )
        lines += [f"Config: {config or 'default'}", f"Log Level: {settings.logging.level}"]
        console.print(Panel.fit("\n".join(lines), title="Starting Server"))

        logger.info(f"Starting Simul 3D MCP Server with {transport} transport")
        _keep_fastmcp_home_in_state_dir()
        try:
            stopped = asyncio.run(_serve(settings, transport, backend_set, stop_signals))
        except SystemExit as exc:
            # uvicorn exits 3 (its STARTUP_FAILURE) when it cannot listen,
            # having only logged why; report it as every command reports a
            # failure instead: exit 1, with the JSON envelope under --json.
            if transport in NETWORK_TRANSPORTS and exc.code not in (0, None):
                _fail_to_listen(settings.server.host, settings.server.port, transport)
            raise
        if stopped is not None:
            _report_stopped(stopped)

    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped by user[/yellow]")
    except typer.Exit:
        raise
    except ConfigError as exc:
        console.print(f"[red]ConfigError: {escape(exc.message)}[/red]")
        raise typer.Exit(2)
    except Exception as e:
        fail(f"Error starting server: {e}", "StartupError")
    finally:
        stop_signals.__exit__()


def _address_in_use(host: str, port: int) -> bool:
    """Whether another socket holds ``host:port``, probed by binding it as uvicorn does."""
    try:
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError:
        return False
    for family, kind, proto, _name, address in addresses:
        with socket.socket(family, kind, proto) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(address)
            except OSError as exc:
                if exc.errno == errno.EADDRINUSE:
                    return True
    return False


def _fail_to_listen(host: str, port: int, transport: str) -> NoReturn:
    """Exit 1 because the HTTP/SSE server could not listen, saying why when the port is taken."""
    details = {"host": host, "port": port, "transport": transport}
    if _address_in_use(host, port):
        fail(
            f"Cannot listen on {host}:{port}: the address is already in use. Stop what holds it, "
            "or pick another port with SERVER__PORT.",
            "AddressInUse",
            details,
        )
    fail(
        f"The {transport} server could not start on {host}:{port}; the log above says why.",
        "StartupError",
        details,
    )


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------
@app.command()
def info(
    config: Optional[Path] = _config_option(),
    backends: Optional[str] = _backends_option(),
    unreal_tools: Optional[str] = _unreal_tools_option(),
    blender_tools: Optional[str] = _blender_tools_option(),
    unreal_mode: Optional[str] = _unreal_mode_option(),
    blender_mode: Optional[str] = _blender_mode_option(),
) -> None:
    """Show server information and capabilities.

    Backends left out of --backends are not probed; they report null status.
    Isaac Sim is not probed either where its tools are skipped (an OS it does
    not run on, or ISAAC_SIM__ENABLED=false); the row says why.
    """
    try:
        from simul import adapters
        from simul.mcp.backends import NOT_SELECTED, backend_skip_reason
        from simul.mcp.server import SimulMCPServer

        settings = _load(config)
        settings, backend_set = _apply_backend_options(
            settings, backends, unreal_tools, blender_tools, unreal_mode, blender_mode
        )

        isaac_host = settings.isaac_sim.socket_host
        isaac_port = settings.isaac_sim.socket_port
        isaac_skipped = backend_skip_reason("isaac", settings, backend_set)
        isaac_reachable: Optional[bool] = (
            _is_isaac_reachable(isaac_host, isaac_port)
            if isaac_skipped is None
            else None
        )
        # A probe imports its runtime (bpy, pxr) on first access, so only a
        # selected backend's probe is looked up.
        blender_available: Optional[bool] = (
            settings.blender.mode == "attached" or adapters.is_blender_available()
            if _is_enabled(backend_set, "blender") else None
        )
        usd_available: Optional[bool] = (
            adapters.is_headless_available() if _is_enabled(backend_set, "usd") else None
        )

        # Instantiate the server so the listing is what actually registered,
        # grouped by the backend registry entry that registered each tool.
        server_instance = SimulMCPServer(settings, backends=backend_set)
        categories: dict[str, list[str]] = server_instance.tools_by_backend()
        tool_names: list[str] = sorted(
            name for names in categories.values() for name in names
        )

        isaac_install_path = settings.isaac_sim.path
        isaac_install_error = settings.isaac_sim.path_error

        if is_json_mode():
            data = {
                "backends": {
                    "isaac_sim": {
                        "reachable": isaac_reachable,
                        "port": isaac_port,
                        "install_path": isaac_install_path,
                        "install_path_error": isaac_install_error,
                        "skipped": isaac_skipped,
                    },
                    "blender": {"available": blender_available, "mode": settings.blender.mode},
                    "usd_headless": {"available": usd_available},
                },
                "tool_count": len(tool_names),
                "categories": categories,
                "capabilities": server_instance.get_capabilities(),
            }
            emit(data)
            return

        system_table = Table(title="System Information")
        system_table.add_column("Component", style="cyan")
        system_table.add_column("Status", style="green")
        system_table.add_column("Details")
        not_selected = ("Not Enabled", "Not probed: left out of --backends")
        if isaac_skipped == NOT_SELECTED:
            isaac_row: Tuple[str, str] = not_selected
        elif isaac_skipped is not None:
            isaac_row = ("[yellow]Skipped[/yellow]", escape(isaac_skipped))
        elif isaac_reachable:
            isaac_row = ("Reachable", "Simulation & viewport via TCP socket")
        else:
            isaac_row = ("Not Reachable", "Isaac Sim tools will retry at call time")
        system_table.add_row(f"Isaac Sim (TCP :{isaac_port})", *isaac_row)
        if isaac_install_path is not None:
            system_table.add_row(
                "Isaac Sim install",
                "Unusable" if isaac_install_error else "Found",
                isaac_install_error or isaac_install_path,
            )
        system_table.add_row(
            "Blender Runtime",
            *(
                not_selected if blender_available is None else (
                    "Available" if blender_available else "Not Available",
                    (
                        "Selected window bridge; use simul blender status to check liveness"
                        if settings.blender.mode == "attached"
                        else (
                            "Blender scene tools through local bpy"
                            if blender_available else "Install bpy or use attached mode"
                        )
                    ),
                )
            ),
        )
        system_table.add_row(
            "USD Headless",
            *(
                not_selected if usd_available is None else (
                    "Available" if usd_available else "Not Available",
                    "pxr library for USD file operations" if usd_available else "No USD support",
                )
            ),
        )
        console.print(system_table)
        console.print()

        categories_table = Table(title="Tool Categories")
        categories_table.add_column("Category", style="cyan")
        categories_table.add_column("Tools", justify="center")
        for cat_name, cat_tools in categories.items():
            categories_table.add_row(cat_name, str(len(cat_tools)))
        console.print(categories_table)
        console.print()

        tools_table = Table(title="Registered Tools")
        tools_table.add_column("Tool Name", style="cyan")
        tools_table.add_column("Category")
        for cat_name, cat_tools in categories.items():
            for t in cat_tools:
                tools_table.add_row(t, cat_name)
        console.print(tools_table)

        console.print(
            Panel.fit(
                f"[bold]Summary[/bold]\n" f"Total Registered Tools: {len(tool_names)}",
                title="Tool Summary",
            )
        )

    except (typer.Exit, ConfigError):
        raise
    except Exception as e:
        if is_json_mode():
            emit_error(str(e), "InfoError")
        console.print(f"[red]Error getting server info: {e}[/red]")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# commands  (P3: machine-readable introspection)
# ---------------------------------------------------------------------------
def _collect_commands(
    typer_app: typer.Typer,
    prefix: str = "",
) -> list[dict[str, object]]:
    """Recursively collect command metadata from a Typer app tree."""
    result: list[dict[str, object]] = []
    click_app = typer.main.get_command(typer_app)

    if hasattr(click_app, "list_commands"):
        ctx = typer.Context(click_app)
        for name in click_app.list_commands(ctx):
            full_name = f"{prefix} {name}".strip() if prefix else name
            cmd = click_app.get_command(ctx, name)
            if cmd is None:
                continue
            if hasattr(cmd, "list_commands"):
                # It's a group — recurse
                sub_typer = None
                for group in getattr(typer_app, "registered_groups", []):
                    if getattr(group, "name", None) == name:
                        sub_typer = getattr(group, "typer_instance", None)
                        break
                if sub_typer is not None:
                    result.extend(_collect_commands(sub_typer, full_name))
                else:
                    # Fallback: list sub-commands from the click group
                    sub_ctx = typer.Context(cmd, parent=ctx)
                    for sub_name in cmd.list_commands(sub_ctx):
                        sub_full = f"{full_name} {sub_name}"
                        sub_cmd = cmd.get_command(sub_ctx, sub_name)
                        if sub_cmd is None:
                            continue
                        params = []
                        for p in sub_cmd.params:
                            params.append(
                                {
                                    "name": p.name,
                                    "type": (
                                        p.type.name
                                        if hasattr(p.type, "name")
                                        else str(p.type)
                                    ),
                                    "required": p.required,
                                    "default": (
                                        str(p.default)
                                        if p.default is not None
                                        else None
                                    ),
                                    "help": getattr(p, "help", None),
                                }
                            )
                        result.append(
                            {
                                "command": sub_full,
                                "description": sub_cmd.help or "",
                                "params": params,
                            }
                        )
            else:
                params = []
                for p in cmd.params:
                    params.append(
                        {
                            "name": p.name,
                            "type": (
                                p.type.name if hasattr(p.type, "name") else str(p.type)
                            ),
                            "required": p.required,
                            "default": (
                                str(p.default) if p.default is not None else None
                            ),
                            "help": getattr(p, "help", None),
                        }
                    )
                result.append(
                    {
                        "command": full_name,
                        "description": cmd.help or "",
                        "params": params,
                    }
                )
    return result


@app.command()
def commands() -> None:
    """List all CLI commands with parameters (always JSON). For the MCP tools, see `tools`."""
    cmds = _collect_commands(app)
    print(
        json.dumps(
            {
                "commands": cmds,
                "count": len(cmds),
                "mcp_tools": "These are CLI commands; `simul tools` lists the MCP tools the server registers.",
            },
            indent=2,
        )
    )


# ---------------------------------------------------------------------------
# tools  (the MCP tool surface, without a live session)
# ---------------------------------------------------------------------------
async def _list_mcp_tools(server_instance: Any) -> List[Dict[str, Any]]:
    """Return each registered tool as the MCP ``tools/list`` response states it."""
    listed = await server_instance.mcp.list_tools()
    result: List[Dict[str, Any]] = []
    for tool in listed:
        mcp_tool = tool.to_mcp_tool() if hasattr(tool, "to_mcp_tool") else tool
        dumped = mcp_tool.model_dump(exclude_none=True, by_alias=True)
        entry: Dict[str, Any] = {
            "name": dumped["name"],
            "description": dumped.get("description", ""),
            "input_schema": dumped.get("inputSchema", {}),
        }
        if dumped.get("annotations"):
            entry["annotations"] = dumped["annotations"]
        result.append(entry)
    return result


@app.command()
def tools(
    config: Optional[Path] = _config_option(),
    backends: Optional[str] = _backends_option(),
    unreal_tools: Optional[str] = _unreal_tools_option(),
    blender_tools: Optional[str] = _blender_tools_option(),
    unreal_mode: Optional[str] = _unreal_mode_option(),
    blender_mode: Optional[str] = _blender_mode_option(),
) -> None:
    """List the MCP tools the server registers, grouped by backend.

    JSON mode gives each tool's name, description, input schema and
    annotations exactly as an MCP client's tools/list receives them, plus the
    serialized size per backend. Each selected backend's tool surface is
    listed whether or not its runtime is installed here ("available" says
    which); nothing is probed or connected. A backend the server would skip
    whatever its runtime (Isaac Sim on an OS it does not run on) lists no
    tools and says why in "skipped". `commands` lists CLI commands.
    """
    try:
        from simul.mcp.backends import BACKENDS
        from simul.mcp.server import SimulMCPServer

        settings = _load(config)
        settings, backend_set = _apply_backend_options(
            settings, backends, unreal_tools, blender_tools, unreal_mode, blender_mode
        )
        server_instance = SimulMCPServer(
            settings, backends=backend_set, require_available=False
        )
        listed = asyncio.run(_list_mcp_tools(server_instance))
        capabilities = server_instance.get_capabilities()
        owner = server_instance._tool_backends

        groups: Dict[str, Dict[str, Any]] = {}
        for spec in BACKENDS:
            if not _is_enabled(backend_set, spec.name):
                continue
            section = getattr(settings, spec.settings_attribute)
            group: Dict[str, Any] = {
                "label": spec.label,
                "available": capabilities[spec.name]["available"],
                "skipped": capabilities[spec.name]["skipped"],
                "tools": [t for t in listed if owner.get(t["name"]) == spec.name],
            }
            surface = getattr(section, "tool_surface", None)
            if surface is not None:
                group["tool_surface"] = surface
            groups[spec.name] = group
        groups["server"] = {
            "label": "Server",
            "available": True,
            "skipped": None,
            "tools": [t for t in listed if t["name"] not in owner],
        }
        for group in groups.values():
            group["tool_count"] = len(group["tools"])
            group["bytes"] = len(json.dumps(group["tools"]))

        if is_json_mode():
            emit(
                {
                    "tool_count": len(listed),
                    "bytes": sum(g["bytes"] for g in groups.values()),
                    "backends": groups,
                }
            )
            return

        table = Table(title=f"MCP Tools ({len(listed)})")
        table.add_column("Backend", style="cyan")
        table.add_column("Tool")
        for group in groups.values():
            heading = group["label"]
            if "tool_surface" in group:
                heading += f" ({group['tool_surface']})"
            if group["skipped"] is not None:
                table.add_row(
                    heading, f"[yellow]skipped:[/yellow] {escape(group['skipped'])}"
                )
                table.add_section()
                continue
            if not group["available"]:
                heading += " [dim]runtime unavailable here[/dim]"
            for index, name in enumerate(sorted(t["name"] for t in group["tools"])):
                table.add_row(heading if index == 0 else "", name)
            table.add_section()
        console.print(table)

    except (typer.Exit, ConfigError):
        raise
    except Exception as e:
        if is_json_mode():
            emit_error(str(e), "ToolsError")
        console.print(f"[red]Error listing MCP tools: {e}[/red]")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# validate-config
# ---------------------------------------------------------------------------
@app.command()
def validate_config(
    config: Path = typer.Argument(
        ...,
        help="Path to configuration file",
        dir_okay=False,
    ),
) -> None:
    """Validate a configuration file strictly.

    Errors (exit 1): YAML that does not parse, a key no setting reads (with the
    setting it most likely meant), a value the settings refuse, from the file
    or from the environment, and an unknown SECURITY__ variable. Warnings:
    retired keys and other unknown SECTION__ variables. A file that does not
    exist exits 2 with ConfigError.
    """
    report = inspect_config_file(config)
    if not report.path.is_file():
        config_error_exit(ConfigError(report.errors[0], [{"source": str(report.path), "reason": "no such file"}]))
    settings = report.settings

    if is_json_mode():
        emit(
            {
                "success": report.valid,
                "valid": report.valid,
                "config_path": str(config),
                "errors": report.errors,
                "warnings": report.warnings,
                "server_name": settings.server.name if settings else None,
                "log_level": settings.logging.level if settings else None,
                "usd_cache_enabled": settings.usd.cache_enabled if settings else None,
                "max_file_size_mb": settings.usd.max_file_size_mb if settings else None,
                "viewport_max_size": settings.viewport.max_size if settings else None,
                "allow_script_execution": settings.security.allow_script_execution if settings else None,
            }
        )
        if not report.valid:
            raise typer.Exit(1)
        return

    console.print(f"[cyan]Validating configuration file: {escape(str(config))}[/cyan]")
    if report.valid:
        console.print("[green]Configuration file is valid[/green]")
    else:
        console.print("[red]Configuration file is invalid[/red]")

    if settings is not None:
        settings_table = Table(title="Configuration Summary")
        settings_table.add_column("Setting", style="cyan")
        settings_table.add_column("Value")
        settings_table.add_row("Server Name", settings.server.name)
        settings_table.add_row("Log Level", settings.logging.level)
        settings_table.add_row("USD Cache Enabled", str(settings.usd.cache_enabled))
        settings_table.add_row("Max File Size (MB)", str(settings.usd.max_file_size_mb))
        settings_table.add_row("Viewport Max Size", str(settings.viewport.max_size))
        settings_table.add_row("Script Execution", str(settings.security.allow_script_execution))
        console.print(settings_table)
    for title, entries, style in (
        ("Validation Errors", report.errors, "red"),
        ("Warnings", report.warnings, "yellow"),
    ):
        if entries:
            table = Table(title=title)
            table.add_column(title[:-1] if title.endswith("s") else title, style=style)
            for entry in entries:
                table.add_row(escape(entry))
            console.print(table)
    if not report.valid:
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------
@app.command()
def version() -> None:
    """Show version information."""
    try:
        from simul import __version__

        version_str = __version__
    except ImportError:
        version_str = "unknown"

    from simul.adapters import is_blender_available, is_headless_available

    settings = get_settings()
    isaac_port = settings.isaac_sim.socket_port
    isaac_reachable = _is_isaac_reachable(settings.isaac_sim.socket_host, isaac_port)
    blender_available = settings.blender.mode == "attached" or is_blender_available()
    usd_available = is_headless_available()

    if is_json_mode():
        emit(
            {
                "version": version_str,
                "isaac_sim": {"reachable": isaac_reachable, "port": isaac_port},
                "blender": {"available": blender_available},
                "usd_headless": {"available": usd_available},
            }
        )
        return

    console.print(
        Panel.fit(
            f"[bold blue]Simul -- agent toolkit for 3D simulation[/bold blue]\n"
            f"Version: {version_str}\n"
            f"Isaac Sim (TCP :{isaac_port}): {'reachable' if isaac_reachable else 'not reachable'}\n"
            f"Blender: {'available' if blender_available else 'unavailable'}\n"
            f"USD Headless: {'available' if usd_available else 'unavailable'}",
            title="Version Information",
        )
    )


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------
@app.command()
def stats(
    tool_name: Optional[str] = typer.Option(
        None, "--tool", "-t", help="Filter to a specific tool"
    ),
    recent: int = typer.Option(0, "--recent", "-r", help="Show last N call records"),
    reset: bool = typer.Option(False, "--reset", help="Clear all usage stats"),
) -> None:
    """Show tool usage statistics from the persistent log."""
    from simul.mcp.usage_tracker import ToolUsageTracker

    if reset:
        tracker = ToolUsageTracker()
        tracker.reset()
        if is_json_mode():
            emit({"success": True, "message": "Stats cleared"})
        else:
            console.print("[green]Usage stats cleared[/green]")
        return

    data = ToolUsageTracker.load_from_log(tool_name=tool_name, limit=recent)

    if is_json_mode():
        emit(data)
        return

    tools = data.get("tools", {})
    if not tools:
        console.print("[dim]No tool usage recorded yet.[/dim]")
        return

    table = Table(title=f"Tool Usage ({data.get('total_calls', 0)} total calls)")
    table.add_column("Tool", style="cyan", no_wrap=True)
    table.add_column("Calls", justify="right")
    table.add_column("OK", justify="right", style="green")
    table.add_column("Fail", justify="right", style="red")
    table.add_column("Avg (ms)", justify="right", style="dim")

    for name, s in tools.items():
        table.add_row(
            name,
            str(s["total_calls"]),
            str(s["successes"]),
            str(s["failures"]),
            f"{s['avg_duration_ms']:.1f}",
        )
    console.print(table)

    recent_records = data.get("recent", [])
    if recent_records:
        console.print()
        log_table = Table(title=f"Recent Calls (last {len(recent_records)})")
        log_table.add_column("Tool", style="cyan")
        log_table.add_column("Duration", justify="right")
        log_table.add_column("Status", justify="center")
        log_table.add_column("Params", style="dim")

        for rec in recent_records:
            status = (
                "[green]OK[/green]"
                if rec.get("success")
                else f"[red]{rec.get('error', 'FAIL')}[/red]"
            )
            params_str = (
                json.dumps(rec.get("params", {}), separators=(",", ":"))
                if rec.get("params")
                else ""
            )
            log_table.add_row(
                rec.get("tool", "?"),
                f"{rec.get('duration_ms', 0):.1f}ms",
                status,
                params_str[:80],
            )
        console.print(log_table)


if __name__ == "__main__":
    app()
