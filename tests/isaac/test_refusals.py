"""Known-fatal requests are refused in the tools layer with a structured error.

An agent told to clean up must not be able to empty the scene, saw off the
transport it speaks through, or discard unsaved work by accident. Each refusal
is a ``RefusedOperation`` payload that names the override where one exists,
and none of them sends anything to Isaac Sim. The stage refusals depend on
Isaac's own file system and dirty flag, so those are checked inside the
generated script; the tests pin that the script carries the check and that the
override flag reaches it.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest


from simul_mcp.adapters.isaac_socket_client import ScriptResult
from simul_mcp.config import Settings
from simul_mcp.mcp import backends as backends_module
from simul_mcp.mcp import server as server_module
from simul_mcp.mcp.tools.isaac._shared import (
    PROTECTED_CARB_SETTING_PREFIXES,
    PROTECTED_EXTENSIONS,
)
from simul_mcp.mcp.tools.isaac_tools import IsaacTools
from tests.fakes import FakeFastMCP

SANDBOX_USD = "/tmp/simul_mcp/scene.usd"


def _tools(output: Dict[str, Any] | None = None) -> tuple[IsaacTools, AsyncMock]:
    client = MagicMock()
    client.address = "127.0.0.1:8226"
    client.timeout_seconds = 30.0
    client.bridge_enabled = False
    client.fallback_to_vscode = True
    execute = AsyncMock(
        return_value=ScriptResult(success=True, output=json.dumps(output or {"ok": True}))
    )
    client.execute = execute
    client.execute_vscode_only = execute
    return IsaacTools(client, settings=Settings()), execute


def _sent_script(execute: AsyncMock) -> str:
    execute.assert_awaited_once()
    script: str = execute.await_args.args[0]
    ast.parse(script)  # the generated script must at least be valid Python
    return script


def _assert_refused(payload: Dict[str, Any], execute: AsyncMock, override: str | None = None) -> None:
    assert payload["success"] is False
    assert payload["error_type"] == "RefusedOperation"
    if override is not None:
        assert override in payload["error"]
        assert payload["details"]["override"] == override
    execute.assert_not_called()


# -- delete_isaac_prim -------------------------------------------------------


@pytest.mark.parametrize("prim_path", ["/", "//"])
def test_delete_refuses_the_pseudo_root_even_with_override(prim_path: str) -> None:
    tools, execute = _tools()
    payload = asyncio.run(tools.delete_isaac_prim(prim_path, allow_root_delete=True))
    _assert_refused(payload, execute)


@pytest.mark.parametrize("prim_path", ["/World", "/World/"])
def test_delete_refuses_world_by_default(prim_path: str) -> None:
    tools, execute = _tools()
    payload = asyncio.run(tools.delete_isaac_prim(prim_path))
    _assert_refused(payload, execute, override="allow_root_delete")
    assert payload["details"]["prim_path"] == prim_path


def test_delete_world_runs_when_overridden() -> None:
    tools, execute = _tools({"prim_path": "/World", "deleted": True})
    payload = asyncio.run(tools.delete_isaac_prim("/World", allow_root_delete=True))
    assert payload["deleted"] is True
    assert "'/World'" in _sent_script(execute)


def test_delete_of_an_ordinary_prim_is_not_gated() -> None:
    tools, execute = _tools({"prim_path": "/World/Cube", "deleted": True})
    payload = asyncio.run(tools.delete_isaac_prim("/World/Cube"))
    assert payload["deleted"] is True
    _sent_script(execute)


# -- disable_isaac_extension -------------------------------------------------


@pytest.mark.parametrize("extension_id", sorted(PROTECTED_EXTENSIONS))
def test_disable_refuses_each_transport_extension(extension_id: str) -> None:
    tools, execute = _tools()
    payload = asyncio.run(tools.disable_isaac_extension(extension_id))
    _assert_refused(payload, execute)
    assert payload["details"]["extension_id"] == extension_id


def test_disable_refuses_the_version_suffixed_form_too() -> None:
    tools, execute = _tools()
    payload = asyncio.run(tools.disable_isaac_extension("khemoo.simul.mcp-0.1.0"))
    _assert_refused(payload, execute)


def test_disable_of_another_extension_goes_through() -> None:
    tools, execute = _tools({"extension_id": "worv.env.sun-0.3.0", "enabled": False})
    payload = asyncio.run(tools.disable_isaac_extension("worv.env.sun"))
    assert payload["enabled"] is False
    _sent_script(execute)


# -- set_isaac_carb_settings -------------------------------------------------


@pytest.mark.parametrize("prefix", PROTECTED_CARB_SETTING_PREFIXES)
def test_carb_write_under_a_transport_prefix_is_refused_whole(prefix: str) -> None:
    tools, execute = _tools()
    settings = {"/rtx/fog/enabled": True, prefix + "port": 1}
    payload = asyncio.run(tools.set_carb_settings(settings))
    _assert_refused(payload, execute)
    assert payload["details"]["refused_keys"] == [prefix + "port"]
    assert "No settings were applied" in payload["error"]


def test_carb_prefix_check_tolerates_a_missing_leading_slash() -> None:
    tools, execute = _tools()
    payload = asyncio.run(
        tools.set_carb_settings({"exts/khemoo.simul.mcp/allow_unsafe_execution": False})
    )
    _assert_refused(payload, execute)


def test_carb_write_elsewhere_is_applied() -> None:
    tools, execute = _tools({"applied": {"/rtx/fog/enabled": True}, "count": 1})
    payload = asyncio.run(tools.set_carb_settings({"/rtx/fog/enabled": True}))
    assert payload["count"] == 1
    _sent_script(execute)


# -- stage operations --------------------------------------------------------


def test_save_as_script_refuses_an_existing_target_unless_overwrite() -> None:
    tools, execute = _tools({"file_path": SANDBOX_USD, "saved": True})
    asyncio.run(tools.save_isaac_stage(SANDBOX_USD))
    script = _sent_script(execute)
    assert "os.path.exists(target)" in script
    assert "not False" in script  # overwrite=False reaches the script
    assert '"error_type": "RefusedOperation"' in script
    assert "overwrite=true" in script


def test_save_as_script_carries_the_overwrite_override() -> None:
    tools, execute = _tools({"file_path": SANDBOX_USD, "saved": True})
    asyncio.run(tools.save_isaac_stage(SANDBOX_USD, overwrite=True))
    assert "not True" in _sent_script(execute)


def test_save_in_place_has_no_overwrite_gate() -> None:
    tools, execute = _tools({"file_path": SANDBOX_USD, "saved": True})
    asyncio.run(tools.save_isaac_stage())
    script = _sent_script(execute)
    assert "save_stage_async()" in script
    assert "RefusedOperation" not in script


@pytest.mark.parametrize("discard", [False, True])
def test_new_stage_script_checks_for_unsaved_edits(discard: bool) -> None:
    tools, execute = _tools({"created": True, "new_stage": True})
    asyncio.run(tools.new_isaac_stage(discard_unsaved=discard))
    script = _sent_script(execute)
    assert f"ctx.has_pending_edit() and not {discard}" in script
    assert '"override": "discard_unsaved"' in script


@pytest.mark.parametrize("discard", [False, True])
def test_open_stage_script_checks_for_unsaved_edits(discard: bool) -> None:
    tools, execute = _tools({"file_path": SANDBOX_USD, "opened": True})
    asyncio.run(tools.open_isaac_stage(SANDBOX_USD, discard_unsaved=discard))
    script = _sent_script(execute)
    assert f"ctx.has_pending_edit() and not {discard}" in script
    assert "open_stage_async" in script


def test_refusal_from_a_generated_script_keeps_its_error_type() -> None:
    """The JSON wrapper must not stamp success onto a script-side refusal."""
    tools, _execute = _tools(
        {"error": "unsaved edits", "error_type": "RefusedOperation", "override": "discard_unsaved"}
    )
    payload = asyncio.run(tools.new_isaac_stage())
    assert payload["success"] is False
    assert payload["error_type"] == "RefusedOperation"


# -- MCP schema --------------------------------------------------------------


def _registered(monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
    monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
    monkeypatch.setattr(server_module, "TaskConfig", None)
    monkeypatch.setattr(backends_module, "is_blender_available", lambda: False)
    monkeypatch.setattr(backends_module, "UnrealRuntimeAdapter", None)
    instance = server_module.SimulMCPServer(settings=Settings())
    return {tool.name: tool for tool in instance.mcp.tools}


@pytest.mark.parametrize(
    ("tool_name", "override", "phrase"),
    [
        ("delete_isaac_prim", "allow_root_delete", "/World"),
        ("save_isaac_stage", "overwrite", "overwrite=true"),
        ("new_isaac_stage", "discard_unsaved", "unsaved edits"),
        ("open_isaac_stage", "discard_unsaved", "unsaved edits"),
    ],
)
def test_overrides_are_in_the_mcp_schema_and_description(
    monkeypatch: pytest.MonkeyPatch, tool_name: str, override: str, phrase: str
) -> None:
    tool = _registered(monkeypatch)[tool_name]
    parameter = inspect.signature(tool.func).parameters[override]
    assert parameter.default is False
    assert phrase in tool.kwargs["description"]


def test_refusing_tools_say_so_in_their_descriptions(monkeypatch: pytest.MonkeyPatch) -> None:
    tools = _registered(monkeypatch)
    assert "khemoo.simul.mcp" in tools["disable_isaac_extension"].kwargs["description"]
    assert "/exts/khemoo.simul.mcp/" in tools["set_isaac_carb_settings"].kwargs["description"]
