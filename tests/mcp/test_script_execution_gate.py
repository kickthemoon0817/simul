"""``security.allow_script_execution`` is the switch for agent-authored code.

The bridge extension's ``allow_unsafe_execution`` flag only gates raw scripts
sent over the bridge transport; raw scripts and every generated tool script
still reach the stock Kit socket, so an operator who set it believed arbitrary
code was off when it was not. The server-side setting removes the three
``execute_*_script`` tools from the surface and makes the CLI path refuse,
while the granular tools keep working.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock

import pytest
from typer.testing import CliRunner

src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_path))

from simul_mcp.adapters.isaac_socket_client import ScriptResult  # noqa: E402
from simul_mcp.cli import isaac as isaac_cli  # noqa: E402
from simul_mcp.cli.main import app  # noqa: E402
from simul_mcp.config import Settings  # noqa: E402
from simul_mcp.mcp import backends as backends_module  # noqa: E402
from simul_mcp.mcp import server as server_module  # noqa: E402
from simul_mcp.mcp.tools.isaac_tools import IsaacTools  # noqa: E402
from tests.mcp.test_discoverability import FakeFastMCP, _AvailableAdapter  # noqa: E402

SCRIPT_TOOLS = ("execute_isaac_script", "execute_unreal_script", "execute_blender_script")


def _settings(allow_script_execution: bool) -> Settings:
    base = Settings()
    return base.model_copy(
        update={
            "security": base.security.model_copy(
                update={"allow_script_execution": allow_script_execution}
            ),
            "unreal": base.unreal.model_copy(update={"tool_surface": "full"}),
        }
    )


def _tool_names(monkeypatch: pytest.MonkeyPatch, allow_script_execution: bool) -> List[str]:
    monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
    monkeypatch.setattr(server_module, "TaskConfig", None)
    monkeypatch.setattr(backends_module, "is_headless_available", lambda: True)
    monkeypatch.setattr(backends_module, "is_blender_available", lambda: True)
    monkeypatch.setattr(backends_module, "BlenderRuntimeAdapter", _AvailableAdapter)
    monkeypatch.setattr(backends_module, "UnrealRuntimeAdapter", _AvailableAdapter)
    instance = server_module.SimulMCPServer(settings=_settings(allow_script_execution))
    return [tool.name for tool in instance.mcp.tools]


def _isaac_tools(allow_script_execution: bool) -> tuple[IsaacTools, MagicMock]:
    client = MagicMock()
    client.address = "127.0.0.1:8226"
    client.timeout_seconds = 30.0
    client.bridge_enabled = False
    client.fallback_to_vscode = True
    result = ScriptResult(success=True, output="ran")
    client.execute = AsyncMock(return_value=result)
    client.execute_vscode_only = AsyncMock(return_value=result)
    return IsaacTools(client, settings=_settings(allow_script_execution)), client


def test_setting_defaults_to_enabled() -> None:
    assert Settings().security.allow_script_execution is True


def test_disabled_setting_drops_only_the_script_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    names = _tool_names(monkeypatch, allow_script_execution=False)
    assert not set(SCRIPT_TOOLS) & set(names)
    # The granular surface is untouched on every backend.
    for granular in ("create_isaac_prim", "ping_isaac", "spawn_unreal_actor", "create_blender_object"):
        assert granular in names


def test_enabled_setting_registers_every_script_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    names = _tool_names(monkeypatch, allow_script_execution=True)
    assert set(SCRIPT_TOOLS) <= set(names)


def test_tools_layer_refuses_when_disabled_without_touching_the_socket() -> None:
    tools, client = _isaac_tools(allow_script_execution=False)

    payload = asyncio.run(tools.execute_script("print('still here')"))

    assert payload["success"] is False
    assert payload["error_type"] == "ScriptExecutionDisabled"
    assert "SECURITY__ALLOW_SCRIPT_EXECUTION" in payload["error"]
    client.execute.assert_not_called()
    client.execute_vscode_only.assert_not_called()


def test_tools_layer_runs_scripts_when_enabled() -> None:
    tools, client = _isaac_tools(allow_script_execution=True)

    payload = asyncio.run(tools.execute_script("print('still here')"))

    assert payload == {"success": True, "output": "ran"}
    client.execute.assert_awaited_once()


def test_cli_exec_reports_the_disabled_error(monkeypatch: pytest.MonkeyPatch) -> None:
    tools, _client = _isaac_tools(allow_script_execution=False)
    monkeypatch.setattr(isaac_cli, "_tools", lambda *args, **kwargs: tools)

    result = CliRunner().invoke(app, ["--json", "isaac", "exec", "print('x')"])

    assert result.exit_code == 1
    payload: dict[str, Any] = json.loads(result.stdout)
    assert payload["error_type"] == "ScriptExecutionDisabled"
