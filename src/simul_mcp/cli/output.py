"""
Dual-mode CLI output: structured JSON for agents, Rich tables for humans.

The output mode is determined by:
1. Explicit ``--json`` flag (highest priority)
2. TTY detection: non-TTY stdout defaults to JSON

All structured data goes to **stdout** (via ``print``).
All human-readable chrome goes to **stderr** (via the Rich console).
"""

import json
import sys
from contextvars import ContextVar
from typing import Any, Dict, Optional

import typer

# Global context variable — set by the top-level Typer callback.
_json_mode: ContextVar[Optional[bool]] = ContextVar("_json_mode", default=None)


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
    if "success" not in data and "error" not in data:
        data["success"] = True
    print(json.dumps(data))


def emit_error(
    error: str,
    error_type: str = "Error",
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a structured error envelope to **stdout** and exit 1."""
    envelope: Dict[str, Any] = {
        "success": False,
        "error": error,
        "error_type": error_type,
    }
    if details:
        envelope["details"] = details
    print(json.dumps(envelope))
    raise typer.Exit(1)
