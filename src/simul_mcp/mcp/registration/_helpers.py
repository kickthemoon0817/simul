"""Shared helpers for tool-registration modules.

Keep this module narrow — only utility functions used by 2+ register
modules belong here. One-off helpers stay inline at their use site.
"""

from __future__ import annotations

from typing import Any, Dict


def apply_success_from_error(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Mark ``payload['success']`` based on the presence of ``error``.

    Mirrors the iter8 fix in ``isaac_tools._execute_json_script`` and
    the iter9 fix in ``unreal_runtime`` input-validation guards: an
    adapter call that returns a payload shaped like
    ``{"connected": false, "error": "..."}`` (e.g. ``ping()`` failing
    on an unreachable port, ``health_check()`` failing on connection
    refused) must surface as ``success=False`` to the MCP client. The
    long-standing pattern in ``_reg_unreal.py`` was a blanket
    ``payload["success"] = True`` immediately after the adapter call,
    which silently masked these "called fine, returned an error
    payload" failures across 56 sites.

    The rule:

      - If the payload already has an explicit ``success`` field
        (some adapters set it themselves), leave it alone.
      - Otherwise, set ``success`` to ``True`` iff ``error`` is
        absent or ``None``.

    Returns the same dict for chaining; mutation is in place because
    every caller already has the dict in hand.
    """
    if "success" in payload:
        return payload
    payload["success"] = payload.get("error") is None
    return payload
