"""A partial failure is not a success on either Isaac transport.

The script path parses one JSON object from stdout; the bridge path takes a
typed payload. Both run it through ``apply_success_from_error`` so a
``syntheticdata_error`` beside a valid listing, or a ``timeline_error`` in a
runtime report, comes back ``success=False``. ``ping_isaac`` follows the same
rule: its ``success`` is its reachability.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_path))

from simul_mcp.adapters.isaac_socket_client import ScriptResult  # noqa: E402
from simul_mcp.config import Settings  # noqa: E402
from simul_mcp.mcp import server as server_module  # noqa: E402
from simul_mcp.mcp.tools.isaac_tools import IsaacTools  # noqa: E402
from tests.mcp.test_discoverability import FakeFastMCP  # noqa: E402


def _tools_with_output(payload: Dict[str, Any]) -> IsaacTools:
    client = MagicMock()
    client.bridge_enabled = False
    client.fallback_to_vscode = True
    client.execute = AsyncMock(
        return_value=ScriptResult(success=True, output=json.dumps(payload))
    )
    return IsaacTools(client, settings=Settings())


def test_script_path_demotes_success_on_a_suffixed_error_key() -> None:
    tools = _tools_with_output(
        {
            "render_var_templates": ["LdrColor"],
            "syntheticdata_error": "no SyntheticData",
        }
    )
    result = asyncio.run(tools.list_render_vars())
    assert result["success"] is False
    assert result["render_var_templates"] == ["LdrColor"]


def test_script_path_keeps_success_without_error_keys() -> None:
    tools = _tools_with_output({"render_var_templates": [], "attach_errors": {}})
    assert asyncio.run(tools.list_render_vars())["success"] is True


def test_script_path_honours_an_explicit_success() -> None:
    tools = _tools_with_output({"success": True, "note_error": "informational"})
    assert asyncio.run(tools.list_render_vars())["success"] is True


def _tools_with_bridge_payload(payload: Dict[str, Any]) -> IsaacTools:
    client = MagicMock()
    client.bridge_enabled = True
    client.fallback_to_vscode = False
    client.bridge_request = AsyncMock(return_value={"status": "ok", "payload": payload})
    return IsaacTools(client, settings=Settings())


def test_bridge_path_demotes_success_on_a_suffixed_error_key() -> None:
    tools = _tools_with_bridge_payload(
        {
            "transport": "simul_bridge",
            "app": {"version": "107.3"},
            "timeline_error": "no timeline",
        }
    )
    result = asyncio.run(tools._execute_bridge_action("get_runtime_info"))
    assert result is not None
    assert result["success"] is False
    assert result["app"] == {"version": "107.3"}


def test_bridge_path_keeps_success_for_a_clean_payload() -> None:
    tools = _tools_with_bridge_payload({"transport": "simul_bridge", "reachable": True})
    result = asyncio.run(tools._execute_bridge_action("ping"))
    assert result is not None
    assert result["success"] is True


# ---------------------------------------------------------------------------
# ping_isaac
# ---------------------------------------------------------------------------


class _PingClient:
    def __init__(self, reachable: bool) -> None:
        self._reachable = reachable
        self.address = "127.0.0.1:8229"
        self.bridge_address = "127.0.0.1:8229"
        self.bridge_circuit_open = False
        self.vscode_address = "127.0.0.1:8226"
        self.timeout_seconds = 5.0

    async def ping(self) -> bool:
        return self._reachable


def _ping_payload(monkeypatch: pytest.MonkeyPatch, reachable: bool) -> Dict[str, Any]:
    monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
    monkeypatch.setattr(server_module, "TaskConfig", None)
    monkeypatch.setattr(server_module, "is_headless_available", lambda: False)
    monkeypatch.setattr(server_module, "is_blender_available", lambda: False)
    monkeypatch.setattr(server_module, "UnrealRuntimeAdapter", None)
    instance = server_module.SimulMCPServer(settings=Settings())
    monkeypatch.setattr(
        instance, "_get_request_isaac_client", lambda: _PingClient(reachable)
    )
    ping = next(tool.func for tool in instance.mcp.tools if tool.name == "ping_isaac")
    result = asyncio.run(ping())
    return json.loads(result.content[0].text)


def test_ping_success_tracks_reachability(monkeypatch: pytest.MonkeyPatch) -> None:
    down = _ping_payload(monkeypatch, reachable=False)
    assert down["success"] is False
    assert down["reachable"] is False
    assert "127.0.0.1:8229" in down["error"]

    up = _ping_payload(monkeypatch, reachable=True)
    assert up["success"] is True
    assert up["reachable"] is True
    assert "error" not in up
