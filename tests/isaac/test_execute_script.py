"""Regression: raw script execution must have one implementation.

There was no ``IsaacTools.execute_script``, so the same logic existed three
times: 139 lines of business logic inside ``_reg_isaac.py`` (transport choice,
JSON unwrap, error envelopes, plus hand-rolled rate limiting, locking, usage
tracking and heartbeat that ``_exec_isaac`` already provides), a partial copy in
``cli/isaac.py`` reaching through ``tools._client`` into the adapter, and a
fourth unwrap helper beside it.

These tests pin the behaviour the consolidated method has to keep.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_path))

from simul_mcp.adapters.isaac_socket_client import ScriptResult
from simul_mcp.config import Settings
from simul_mcp.mcp.tools.isaac_tools import MAX_SCRIPT_BYTES, IsaacTools


def _tools(
    result: ScriptResult | None = None,
    *,
    side_effect: Exception | None = None,
    bridge_enabled: bool = False,
) -> tuple[IsaacTools, MagicMock]:
    client = MagicMock()
    client.address = "127.0.0.1:8226"
    client.timeout_seconds = 30.0
    client.bridge_enabled = bridge_enabled
    client.fallback_to_vscode = True
    client.execute = AsyncMock(return_value=result, side_effect=side_effect)
    client.execute_vscode_only = AsyncMock(return_value=result, side_effect=side_effect)
    client.execute_bridge_script_only = AsyncMock(
        return_value=result, side_effect=side_effect
    )
    client.bridge_request = AsyncMock(return_value=None)
    return IsaacTools(client, settings=Settings()), client


def test_json_object_output_is_returned_directly() -> None:
    tools, _ = _tools(ScriptResult(success=True, output=json.dumps({"prims": 12})))

    result = asyncio.run(tools.execute_script("print('x')"))

    assert result["prims"] == 12
    assert result["success"] is True


def test_non_json_output_is_returned_as_text() -> None:
    tools, _ = _tools(ScriptResult(success=True, output="hello from isaac"))

    result = asyncio.run(tools.execute_script("print('hello from isaac')"))

    assert result == {"success": True, "output": "hello from isaac"}


def test_json_scalar_output_is_kept_alongside_the_text() -> None:
    """A bare JSON value is not a payload, so it must not replace one."""
    tools, _ = _tools(ScriptResult(success=True, output="[1, 2, 3]"))

    result = asyncio.run(tools.execute_script("print([1,2,3])"))

    assert result["success"] is True
    assert result["parsed"] == [1, 2, 3]
    assert result["output"] == "[1, 2, 3]"


def test_oversize_payload_is_refused_before_execution() -> None:
    tools, client = _tools(ScriptResult(success=True, output="{}"))

    result = asyncio.run(tools.execute_script("x" * (MAX_SCRIPT_BYTES + 1)))

    assert result["error_type"] == "PayloadTooLarge"
    assert client.execute.await_count == 0


def test_failed_script_reports_the_traceback() -> None:
    tools, _ = _tools(
        ScriptResult(
            success=False,
            output="",
            error_name="NameError",
            error_value="name 'foo' is not defined",
            traceback="Traceback...",
        )
    )

    result = asyncio.run(tools.execute_script("foo"))

    assert result["error_type"] == "NameError"
    assert result["details"]["traceback"] == "Traceback..."


def test_connection_failure_names_the_address_and_the_remedy() -> None:
    tools, _ = _tools(side_effect=ConnectionRefusedError("refused"))

    result = asyncio.run(tools.execute_script("print(1)"))

    assert result["error_type"] == "ConnectionError"
    assert "127.0.0.1:8226" in result["error"]
    assert "ping_isaac" in result["error"]


def test_timeout_names_the_configured_timeout() -> None:
    tools, _ = _tools(side_effect=TimeoutError("slow"))

    result = asyncio.run(tools.execute_script("while True: pass"))

    assert result["error_type"] == "TimeoutError"
    assert "30.0" in result["error"]


def test_bridge_enabled_keeps_raw_scripts_on_the_vscode_socket() -> None:
    """The typed bridge stays on the typed control path."""
    tools, client = _tools(ScriptResult(success=True, output="{}"), bridge_enabled=True)

    asyncio.run(tools.execute_script("print(1)"))

    assert client.execute_vscode_only.await_count == 1
    assert client.execute.await_count == 0


def test_a_script_error_payload_is_not_stamped_successful() -> None:
    """Generated scripts report failure as {"error": ...}; do not contradict it."""
    tools, _ = _tools(
        ScriptResult(success=True, output=json.dumps({"error": "No stage is currently open"}))
    )

    result = asyncio.run(tools.execute_script("print(...)"))

    assert result["error"] == "No stage is currently open"
    assert result.get("success") is not True, "stamped success onto an error payload"


def test_raw_stdout_is_preserved_for_json_emitting_scripts() -> None:
    """`--raw` exists to hand back stdout; a JSON script must not yield nothing."""
    tools, _ = _tools(ScriptResult(success=True, output='{"prims": [1, 2]}'))

    plain = asyncio.run(tools.execute_script("print(...)"))
    assert "output" not in plain, "stdout duplicated the payload for MCP callers"

    tools2, _ = _tools(ScriptResult(success=True, output='{"prims": [1, 2]}'))
    raw = asyncio.run(tools2.execute_script("print(...)", keep_raw_output=True))

    assert raw["prims"] == [1, 2]
    assert raw["output"] == '{"prims": [1, 2]}', "raw stdout was discarded"
