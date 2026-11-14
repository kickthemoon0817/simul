"""How the JSON object an agent's script prints becomes a tool result.

simul's own granular scripts report a failure under ``error`` or a
``*_error`` key, and ``apply_success_from_error`` reads those keys. An
agent's script (``execute_isaac_script``, ``execute_unreal_script``,
``execute_blender_script`` and the ``exec`` commands) is different: what it
prints is its data. ``{"position_error": 0.002}`` is a tracking metric, and
``{"error": 0.0}`` a residual, not a failure. So the call succeeded when the
script ran without raising, unless the script itself says otherwise by
printing ``"success": false``.

One rule for the failure's message and type on every backend
(``reported_failure_message``, ``reported_error_type``), shared by the
adapters, the MCP layer and the CLI.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from ..mcp.error_types import SCRIPT_ERROR


def script_reported_failure(payload: Dict[str, Any]) -> bool:
    """Whether a script's printed object says the script failed: ``"success": false``.

    Args:
        payload: The JSON object the script printed.

    Returns:
        True only for ``success`` that is exactly ``False``.
    """
    return payload.get("success") is False


def apply_script_success(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Set ``success`` on the JSON object an agent's script printed (in place).

    The script ran without raising, so every key it printed is its data,
    ``error`` and ``*_error`` included. It failed only when it printed
    ``"success": false``; that payload's ``error`` becomes its message and
    its ``error_type`` its type (``reported_failure_message``,
    ``reported_error_type``). Any other ``success`` value, or none, reads as
    success and is set to ``True``: ``success`` is the envelope's flag.

    Args:
        payload: The JSON object the script printed.

    Returns:
        The same dict, for chaining.
    """
    failed = script_reported_failure(payload)
    payload["success"] = not failed
    if failed:
        payload["error"] = reported_failure_message(payload)
        payload["error_type"] = reported_error_type(payload)
    return payload


def printed_json_object(output: Optional[str]) -> Optional[Dict[str, Any]]:
    """The JSON object a script's whole printed output is, or None.

    Args:
        output: Everything the script printed.

    Returns:
        The object when the output, stripped of surrounding whitespace, is one
        JSON object; None for anything else (text, a bare value, several lines).
    """
    text = (output or "").strip()
    if not text.startswith("{"):
        return None
    try:
        parsed = json.loads(text)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def reported_failure_message(payload: Dict[str, Any]) -> str:
    """The message of a failure a script reported with ``"success": false``.

    The same on every backend: the printed ``error`` as text (a string as is,
    an object or a number as its JSON), else the non-null ``*_error`` keys as
    ``key: value; ...``, else a sentence saying the script gave no message.

    Args:
        payload: The JSON object the script printed.

    Returns:
        The message.
    """
    error = payload.get("error")
    if isinstance(error, str):
        if error.strip():
            return error
    elif error is not None:
        return json.dumps(error, default=str)
    parts = [
        f"{key}: {json.dumps(value, default=str) if not isinstance(value, str) else value}"
        for key, value in payload.items()
        if key.endswith("_error") and key != "error" and value is not None
    ]
    return (
        "; ".join(parts)
        or "The script printed success: false without an error message."
    )


def reported_error_type(payload: Dict[str, Any]) -> str:
    """The ``error_type`` of a failure a script reported: the one it printed, else ``ScriptError``.

    Args:
        payload: The JSON object the script printed.

    Returns:
        A non-empty type name. The MCP envelope maps a name outside its
        vocabulary onto that name's family.
    """
    printed = payload.get("error_type")
    return printed if isinstance(printed, str) and printed else SCRIPT_ERROR
