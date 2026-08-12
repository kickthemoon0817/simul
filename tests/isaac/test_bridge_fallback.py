"""Regression: an unreachable bridge must fall through to the script transport.

``_execute_bridge_action`` returned an error envelope on a connection failure
where it should return ``None``. Callers treat any non-``None`` result as final,
so the VS Code-socket script path below it was never reached: tools failed
against a closed bridge port while ``ping``, which does fall back, reported the
same instance as reachable.
"""


from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_path))

from simul_mcp.adapters.isaac_socket_client import ScriptResult
from simul_mcp.mcp.tools.isaac_tools import IsaacTools


# ---------------------------------------------------------------------------
# #88 — bridge connection failure must fall through to the script path
# ---------------------------------------------------------------------------


def _client_with_unreachable_bridge(script_payload: Dict[str, Any]) -> MagicMock:
    """A client whose bridge port is closed but whose script transport works."""
    client = MagicMock()
    client.address = "127.0.0.1:8226"
    client.bridge_address = "127.0.0.1:8229"
    client.timeout_seconds = 30.0
    client.bridge_enabled = True
    client.fallback_to_vscode = True
    client.bridge_request = AsyncMock(
        side_effect=ConnectionRefusedError(
            "Cannot connect to Isaac bridge at 127.0.0.1:8229."
        )
    )
    result = ScriptResult(success=True, output=json.dumps(script_payload))
    client.execute = AsyncMock(return_value=result)
    client.execute_vscode_only = AsyncMock(return_value=result)
    client.execute_bridge_script_only = AsyncMock(return_value=result)
    return client


def test_unreachable_bridge_falls_back_to_script_transport() -> None:
    """A closed bridge port must not surface as a failed tool call."""
    client = _client_with_unreachable_bridge({"up_axis": "Z", "total_prims": 754})
    tools = IsaacTools(client)

    result = asyncio.run(tools.get_isaac_stage_info())

    assert result.get("up_axis") == "Z"
    assert result.get("total_prims") == 754
    assert result.get("error_type") != "ConnectionRefusedError"
    assert client.execute_vscode_only.await_count == 1


def test_bridge_action_returns_none_when_connection_refused() -> None:
    """_execute_bridge_action signals 'try the script path' via None."""
    client = _client_with_unreachable_bridge({"ok": True})
    tools = IsaacTools(client)

    outcome = asyncio.run(tools._execute_bridge_action("get_stage_info"))

    assert outcome is None


def test_bridge_action_still_reports_errors_when_fallback_disabled() -> None:
    """Without a fallback there is nothing below us, so the error must surface."""
    client = _client_with_unreachable_bridge({"ok": True})
    client.fallback_to_vscode = False
    tools = IsaacTools(client)

    outcome = asyncio.run(tools._execute_bridge_action("get_stage_info"))

    assert outcome is not None
    assert outcome["error_type"] == "ConnectionRefusedError"
