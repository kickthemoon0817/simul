"""
Dual-mode CLI output: structured JSON for agents, Rich tables for humans.

The output mode is determined by:
1. Explicit ``--json`` flag (highest priority)
2. TTY detection: non-TTY stdout defaults to JSON

All structured data goes to **stdout** (via ``print``).
All human-readable chrome goes to **stderr** (via the Rich console).
"""

import asyncio
import codecs
import json
import math
import sys
from contextvars import ContextVar
from pathlib import Path
from typing import Any, BinaryIO, Coroutine, Dict, NoReturn, Optional, TypeVar, Union

import click
import typer
from rich.console import Console
from rich.markup import escape as rich_escape
from rich.panel import Panel

# Global context variable — set by the top-level Typer callback.
_json_mode: ContextVar[Optional[bool]] = ContextVar("_json_mode", default=None)

#: Human-readable chrome goes to stderr. The backend CLI modules re-export
#: this instance as their ``console`` so the helpers below and the commands
#: print through one object.
console = Console(stderr=True)

T = TypeVar("T")

_SCRIPT_INPUT_HINT = "Provide a script string, .py file path, or pipe code via stdin."


def set_json_mode(enabled: bool) -> None:
    """Set the global JSON output mode."""
    _json_mode.set(enabled)


def is_json_mode() -> bool:
    """
    Return True when output should be structured JSON.

    Priority:
        1. Explicit ``--json`` flag  (ContextVar)
        2. stdout is not a TTY       (auto-detect)
    """
    explicit = _json_mode.get()
    if explicit is not None:
        return explicit
    return not sys.stdout.isatty()


def emit(data: Dict[str, Any]) -> None:
    """
    Write *data* as a single JSON line to **stdout**.

    Adds ``"success": true`` if not already present and no ``"error"`` key.
    """
    out = data.copy()  # shallow — emit only touches top-level keys
    if "success" not in out and "error" not in out:
        out["success"] = True
    print(json.dumps(out))


def emit_error(
    error: str,
    error_type: str = "Error",
    details: Optional[Dict[str, Any]] = None,
    *,
    exit_code: int = 1,
) -> NoReturn:
    """Write a structured error envelope to **stdout** and exit ``exit_code``."""
    envelope: Dict[str, Any] = {
        "success": False,
        "error": error,
        "error_type": error_type,
    }
    if details:
        envelope["details"] = details
    print(json.dumps(envelope))
    raise typer.Exit(exit_code)


def fail(
    message: str,
    error_type: str = "Error",
    details: Optional[Dict[str, Any]] = None,
    *,
    exit_code: int = 1,
) -> NoReturn:
    """Report a CLI failure and exit ``exit_code``.

    JSON mode writes the ``emit_error`` envelope to stdout; human mode prints
    ``message`` (markup-escaped) in red on the stderr console. Both exit with the same code,
    so a usage error (``exit_code=2``) reads the same to a script either way.
    """
    if is_json_mode():
        emit_error(message, error_type, details, exit_code=exit_code)
    console.print(f"[red]{rich_escape(message)}[/red]")
    raise typer.Exit(exit_code)


def _bad_option_value(
    value: Any, param: Optional[click.Parameter], ctx: Optional[click.Context], expected: str
) -> NoReturn:
    """Exit 2 with ``InvalidArgument`` because an option's value is not ``expected``.

    Shell completion parses the command line resiliently and drops a value
    its type refuses, so there the refusal is a quiet ``click.BadParameter``,
    as Click's ``INT`` raises: what the CLI prints while it completes is what
    the shell reads as the candidates.
    """
    opts = list(getattr(param, "opts", None) or ["option"])
    name = next((opt for opt in opts if opt.startswith("--")), opts[0])
    message = f"{name} takes {expected}, got {value!r}"
    if ctx is not None and ctx.resilient_parsing:
        raise click.BadParameter(message, ctx=ctx, param=param)
    fail(message, "InvalidArgument", exit_code=2)


class _PortNumber(click.types.IntParamType):
    """Click type for a port option: an integer from 1 to 65535.

    Any other value, an integer out of range or not an integer at all, exits
    2 with ``InvalidArgument`` through ``fail`` while the command line is
    parsed, before the command runs. A ``click.BadParameter`` would exit 2
    too, but as a usage message on stderr, without the JSON envelope even
    under ``--json``. Valid values parse as Click's ``INT`` parses them, and
    the type keeps its name, ``integer``, in ``--help`` and ``simul commands``.
    """

    def convert(self, value: Any, param: Optional[click.Parameter], ctx: Optional[click.Context]) -> int:
        try:
            port = int(super().convert(value, param, ctx))
        except click.BadParameter:
            port = 0  # not an integer: refused below like any port out of range
        if not 1 <= port <= 65535:
            _bad_option_value(value, param, ctx, "a port number from 1 to 65535")
        return port


class _TimeoutSeconds(click.types.FloatParamType):
    """Click type for a timeout option: a finite number of seconds greater than 0.

    Zero, a negative number, ``nan``, ``inf`` or a value that is not a number
    exits 2 with ``InvalidArgument`` as ``_PortNumber`` does, instead of
    reaching a transport that reports an immediate "timed out after nans".
    The type keeps the name ``float``.
    """

    def convert(self, value: Any, param: Optional[click.Parameter], ctx: Optional[click.Context]) -> float:
        try:
            seconds = float(super().convert(value, param, ctx))
        except click.BadParameter:
            seconds = math.nan  # not a number: refused below like nan itself
        if not (math.isfinite(seconds) and seconds > 0):
            _bad_option_value(value, param, ctx, "a finite number of seconds greater than 0")
        return seconds


#: ``click_type`` of every port option (``--port``, ``--socket-port``, ...).
PORT_NUMBER = _PortNumber()
#: ``click_type`` of every timeout option (``--timeout``, ``--wait-timeout``).
TIMEOUT_SECONDS = _TimeoutSeconds()


def _failure_message(result: Dict[str, Any]) -> str:
    """Name what went wrong in a ``success: false`` payload without ``error``."""
    parts = [
        f"{key}: {value}"
        for key, value in result.items()
        if key.endswith("_error") and value is not None
    ]
    return "; ".join(parts) or "operation reported success: false"


def run_or_exit(
    coro: Coroutine[Any, Any, T],
    *,
    catch_exceptions: bool = False,
    show_traceback: bool = False,
    allow_partial: bool = False,
) -> T:
    """Run a backend coroutine and exit non-zero when its payload failed.

    A dict payload goes through ``apply_success_from_error`` first — the
    rule the MCP envelope applies — so a payload that carries ``error`` or
    any non-null ``*_error`` key without an explicit ``success`` counts as a
    failure, and so does an explicit ``success: false``.

    On failure, JSON mode writes the whole payload with ``success: false``
    (and ``error_type``, default ``Error``, beside a top-level ``error``),
    so data a failure carries reaches the caller as the MCP envelope passes
    it on: ``set-transform``'s read-back transform when Unreal did not apply
    it, for one. Then it exits 1. Human mode prints the error in red.

    Args:
        coro: The coroutine to run.
        catch_exceptions: Turn an exception raised by the coroutine into a
            CLI failure instead of letting it propagate.
        show_traceback: In human mode, print ``details.traceback`` from a
            failed payload in a panel.
        allow_partial: Hand a payload that failed without a top-level
            ``error`` back to the caller instead of exiting, so a command
            that renders per-section errors (``isaac runtime-info``) still
            shows the sections that worked. The caller must then call
            ``exit_if_failed`` after rendering.

    Returns:
        The payload, when it did not fail (or failed only partially and
        ``allow_partial`` is set).
    """
    # Deferred: simul.mcp pulls in every schema and FastMCP, which the
    # lightweight commands (usd, --help) should not pay for.
    from simul.mcp.registration._helpers import apply_success_from_error

    try:
        result = asyncio.run(coro)
    except Exception as exc:
        if not catch_exceptions:
            raise
        if is_json_mode():
            emit_error(str(exc), type(exc).__name__)
        console.print(f"[red]{type(exc).__name__}: {rich_escape(str(exc))}[/red]")
        raise typer.Exit(1)

    if not isinstance(result, dict):
        return result
    apply_success_from_error(result)
    error = result.get("error")
    if not error and (allow_partial or result.get("success") is not False):
        return result

    if error:
        error_type = result.get("error_type", "Error")
        if is_json_mode():
            print(
                json.dumps(
                    {**result, "success": False, "error": error, "error_type": error_type},
                    default=str,
                )
            )
            raise typer.Exit(1)
        console.print(f"[red]{error_type}: {rich_escape(str(error))}[/red]")
        details = result.get("details")
        if show_traceback and isinstance(details, dict) and details.get("traceback"):
            console.print(Panel(details["traceback"], title="Traceback", border_style="red"))
        raise typer.Exit(1)

    if is_json_mode():
        print(json.dumps(result, default=str))
        raise typer.Exit(1)
    console.print(f"[red]Error: {rich_escape(_failure_message(result))}[/red]")
    raise typer.Exit(1)


def exit_if_failed(result: Any) -> None:
    """Exit 1 after rendering a payload ``run_or_exit(allow_partial=True)`` returned failed."""
    if isinstance(result, dict) and result.get("success") is False:
        raise typer.Exit(1)


def _fail_not_utf8(
    source: str,
    exc: Union[UnicodeDecodeError, UnicodeEncodeError],
    *,
    escaped_bytes: bool = True,
) -> NoReturn:
    """Exit 2 because script input is not UTF-8, naming the first byte that is not.

    A ``UnicodeDecodeError`` (a file or stdin) points into the bytes read. A
    ``UnicodeEncodeError`` (an inline argument, or a stdin that hands over
    text) points at a lone surrogate. Text decoded from UTF-8 with
    surrogateescape, as POSIX argv is under a UTF-8 filesystem encoding,
    carries each byte that is not UTF-8 as U+DC80..U+DCFF, so with
    ``escaped_bytes`` such a surrogate is named as the byte 0x80..0xFF it
    stands for. Any other surrogate is named as itself.

    Args:
        source: Where the script came from, completing "Script ...":
            ``"file job.py"``, ``"from stdin"`` or ``"argument"``.
        exc: The error strict UTF-8 raised on it.
        escaped_bytes: Whether U+DC80..U+DCFF in the text stands for a byte
            that is not UTF-8.
    """
    if isinstance(exc, UnicodeDecodeError):
        offset = exc.start
        problem = f"byte 0x{exc.object[offset]:02x} ({exc.reason})"
    else:
        point = ord(exc.object[exc.start])
        offset = len(exc.object[: exc.start].encode("utf-8"))
        if escaped_bytes and 0xDC80 <= point <= 0xDCFF:
            problem = f"byte 0x{point - 0xDC00:02x}"
        else:
            problem = f"lone surrogate U+{point:04X}"
    fail(f"Script {source} is not valid UTF-8: {problem} at offset {offset}.", "InvalidEncoding", exit_code=2)


def _require_utf8_text(source: str, text: str) -> str:
    """Return ``text`` when it encodes as UTF-8, else exit 2 naming its first lone surrogate."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as exc:
        _fail_not_utf8(source, exc)
    return text


def _require_utf8_argument(script: str) -> str:
    """Return inline code that encodes as UTF-8, else exit 2 saying why it does not.

    POSIX hands argv to Python as bytes, decoded with the filesystem encoding
    and surrogateescape, so U+DC80..U+DCFF stands for a byte that encoding
    could not decode. Under UTF-8 that is the first byte that is not UTF-8,
    at the UTF-8 length of what precedes it. Under any other encoding the
    bytes may well be valid UTF-8 and the offset is unknown, so the error
    names the encoding instead. Windows hands argv over as UTF-16, where a
    surrogate stands for no byte and is named as itself.
    """
    try:
        script.encode("utf-8")
    except UnicodeEncodeError as exc:
        from_bytes = sys.platform != "win32" and 0xDC80 <= ord(script[exc.start]) <= 0xDCFF
        encoding = sys.getfilesystemencoding()
        if from_bytes and codecs.lookup(encoding).name != "utf-8":
            fail(
                f"Script argument could not be decoded in the filesystem encoding ({encoding}); "
                "pass the script in a .py file or on stdin, which are read as UTF-8.",
                "InvalidEncoding",
                exit_code=2,
            )
        _fail_not_utf8("argument", exc, escaped_bytes=from_bytes)
    return script


def _names_a_path(script: str) -> bool:
    """Whether a ``.py`` argument names something on disk, so it is a path and not code.

    It does when ``lstat`` finds it, whatever it is, or is refused permission
    to look, as under a directory that can't be searched. ``lstat`` does not
    follow a symlink, so a broken one or a loop counts as a path too, and
    then fails to read. A named pipe is read as ``python`` would read it,
    waiting for its writer. Any other failure leaves it code: no such file,
    or no possible file name, such as a long inline script ending in ``.py``
    (``OSError``, file name too long).
    """
    try:
        Path(script).lstat()
    except PermissionError:
        return True
    except (OSError, ValueError):
        return False
    return True


def read_script_arg(script: Optional[str]) -> str:
    """Resolve a script argument that may be code, a ``.py`` path, or absent.

    ``None`` reads stdin when it is piped. With nothing to read there (a
    terminal, or no stdin at all), it exits 2 with ``InputError``, since that
    is a bad invocation. A value ending in ``.py`` that names something on
    disk is read as a file. When the read fails (a directory, a broken
    symlink, no read permission), it exits 2 with ``ReadError``, naming the
    path and the OS reason, instead of sending the path to the engine as
    code. Anything else is the code itself, including a ``.py`` value that
    names nothing. The suffix is checked before touching the filesystem, and
    a long inline script that does end in ``.py`` stays code: its ``lstat``
    fails with file name too long.

    The script must be UTF-8 however it arrives, and stdin is read as bytes
    so that neither the locale nor UTF-8 mode decides how it decodes. Input
    that is not valid UTF-8 exits 2 naming its source and the first bad
    byte, instead of raising a ``UnicodeDecodeError`` traceback or slipping
    through as lone surrogates, which the script size limit counts at 3
    bytes each. Valid input reads as before: a BOM stays a leading U+FEFF,
    a file's line endings go through universal newlines, and stdin's are
    kept as sent. A stdin replaced by a text stream with no ``buffer`` (an
    ``io.StringIO``, which ``TextIOBase`` allows) is read as text and must
    hold no lone surrogate either. How an argument's lone surrogate is
    reported depends on how argv was decoded (see ``_require_utf8_argument``).
    """
    if script is None:
        if sys.stdin is None or sys.stdin.isatty():
            fail(_SCRIPT_INPUT_HINT, "InputError", exit_code=2)
        buffer: Optional[BinaryIO] = getattr(sys.stdin, "buffer", None)
        if buffer is None:
            return _require_utf8_text("from stdin", sys.stdin.read())
        try:
            return buffer.read().decode("utf-8")
        except UnicodeDecodeError as exc:
            _fail_not_utf8("from stdin", exc)
    if script.endswith(".py") and _names_a_path(script):
        try:
            return Path(script).read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            _fail_not_utf8(f"file {script}", exc)
        except OSError as exc:
            fail(f"Script file {script} could not be read: {exc.strerror or exc}.", "ReadError", exit_code=2)
    return _require_utf8_argument(script)
