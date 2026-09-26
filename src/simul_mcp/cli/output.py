"""
Dual-mode CLI output: structured JSON for agents, Rich tables for humans.

The output mode is determined by:
1. Explicit ``--json`` flag (highest priority)
2. TTY detection: non-TTY stdout defaults to JSON

All structured data goes to **stdout** (via ``print``).
All human-readable chrome goes to **stderr** (via the Rich console).
"""

import asyncio
import json
import sys
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Coroutine, Dict, NoReturn, Optional, TypeVar

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

    On failure, JSON mode writes the ``emit_error`` envelope when the
    payload has a top-level ``error``, otherwise the payload itself (it
    already says ``success: false`` and keeps whatever partial data it
    carries), then exits 1. Human mode prints the error in red.

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
    # Deferred: simul_mcp.mcp pulls in every schema and FastMCP, which the
    # lightweight commands (usd, --help) should not pay for.
    from simul_mcp.mcp.registration._helpers import apply_success_from_error

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
            emit_error(error, error_type, result.get("details"))
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


def read_script_arg(script: Optional[str]) -> str:
    """Resolve a script argument that may be code, a ``.py`` path, or absent.

    ``None`` reads stdin when it is piped and fails otherwise. A value ending
    in ``.py`` that names an existing file is read from disk; anything else
    is the code itself. The suffix is checked before touching the
    filesystem, since ``Path(code).is_file()`` on a long inline script
    raises ``OSError`` (file name too long) instead of returning False.
    """
    if script is None:
        if sys.stdin.isatty():
            fail(_SCRIPT_INPUT_HINT, "InputError")
        return sys.stdin.read()
    if script.endswith(".py"):
        path = Path(script)
        try:
            if path.is_file():
                return path.read_text(encoding="utf-8")
        except OSError:
            pass
    return script
