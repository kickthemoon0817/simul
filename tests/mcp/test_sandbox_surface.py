"""The MCP surface must tell the truth about the sandbox and about side effects.

Two things a harness reads before it runs a tool: the description, which is
where an agent learns that a path has to sit inside ``security.allowed_paths``,
and the annotations, which an auto-approval policy may trust. A sandbox denial
that only echoes the rejected path back leaves the agent guessing and reaching
for ``execute_isaac_script``, where nothing checks paths at all. A capture tool
annotated read-only that writes files and moves the viewport resolution
invites a harness to auto-approve a write.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest


from simul_mcp.config import Settings
from simul_mcp.mcp import backends as backends_module
from simul_mcp.mcp import server as server_module
from simul_mcp.mcp.schemas.common import ErrorResponse
from simul_mcp.utils.paths import SandboxDenied
from tests.fakes import FakeFastMCP

OUTSIDE_SANDBOX = "/etc/shadow"

# Tools that take a filesystem path or URL and must say so.
FILE_TAKING_TOOLS = (
    "open_isaac_stage",
    "save_isaac_stage",
    "import_isaac_asset",
    "add_isaac_reference",
    "capture_isaac_viewport",
    "load_usd_file",
    "validate_usd_file",
)


def _make_server(monkeypatch: pytest.MonkeyPatch) -> server_module.SimulMCPServer:
    monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
    monkeypatch.setattr(server_module, "TaskConfig", None)
    # USD tools only register when the headless adapter exists, and these
    # tests exercise the USD sandbox surface, so keep the real adapter.
    monkeypatch.setattr(backends_module, "is_headless_available", lambda: True)
    monkeypatch.setattr(backends_module, "is_blender_available", lambda: False)
    monkeypatch.setattr(backends_module, "UnrealRuntimeAdapter", None)
    return server_module.SimulMCPServer(settings=Settings())


def _tool(instance: server_module.SimulMCPServer, name: str) -> SimpleNamespace:
    matches = [tool for tool in instance.mcp.tools if tool.name == name]
    assert len(matches) == 1, f"expected one tool named {name}, found {len(matches)}"
    return matches[0]


def _hints(instance: server_module.SimulMCPServer, name: str) -> Dict[str, Any]:
    return _tool(instance, name).kwargs["annotations"].model_dump(exclude_none=True)


def _payload(result: Any) -> Dict[str, Any]:
    if isinstance(result, dict):
        return result
    return json.loads(result.content[0].text)


@pytest.mark.parametrize("name", FILE_TAKING_TOOLS)
def test_file_taking_tools_name_the_sandbox(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    instance = _make_server(monkeypatch)
    description = _tool(instance, name).kwargs["description"]
    assert "sandbox" in description
    assert "security.allowed_paths" in description


def test_capture_is_not_read_only_and_says_it_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = _make_server(monkeypatch)
    assert _hints(instance, "capture_isaac_viewport")["readOnlyHint"] is False
    description = _tool(instance, "capture_isaac_viewport").kwargs["description"]
    assert "PNG" in description and "capture" in description.lower()


def test_overwriting_tools_are_destructive_on_both_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    """One rule — overwrites existing state means destructive — applied identically."""
    instance = _make_server(monkeypatch)
    assert _hints(instance, "save_isaac_stage").get("destructiveHint") is True
    assert _hints(instance, "set_isaac_prim_attribute").get("destructiveHint") is True
    assert _hints(instance, "update_prim_attributes").get("destructiveHint") is True


def test_creating_a_prim_is_not_destructive(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = _make_server(monkeypatch)
    assert "destructiveHint" not in _hints(instance, "create_prim")
    assert "destructiveHint" not in _hints(instance, "create_isaac_prim")


def _assert_actionable(payload: Dict[str, Any], path: str, access: str) -> None:
    assert payload["error_type"] == "SandboxError", payload
    details = payload["details"]
    assert details["file_path"] == path
    assert details["access"] == access
    assert isinstance(details["allowed_roots"], list) and details["allowed_roots"]
    assert all(Path(root).is_absolute() for root in details["allowed_roots"])
    assert isinstance(details["allowed_url_schemes"], list)
    assert "allowed_paths" in details["hint"]


def test_isaac_registration_denials_are_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = _make_server(monkeypatch)

    result = asyncio.run(_tool(instance, "open_isaac_stage").func(file_path=OUTSIDE_SANDBOX))
    _assert_actionable(_payload(result), OUTSIDE_SANDBOX, "read")

    result = asyncio.run(_tool(instance, "import_isaac_asset").func(asset_path=OUTSIDE_SANDBOX))
    _assert_actionable(_payload(result), OUTSIDE_SANDBOX, "read")

    result = asyncio.run(
        _tool(instance, "add_isaac_reference").func(
            prim_path="/World/Ref", reference_path=OUTSIDE_SANDBOX
        )
    )
    _assert_actionable(_payload(result), OUTSIDE_SANDBOX, "read")


def test_save_denial_reports_write_access_and_write_schemes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = _make_server(monkeypatch)
    url = "omniverse://nucleus/Projects/scene.usd"

    result = asyncio.run(_tool(instance, "save_isaac_stage").func(file_path=url))

    payload = _payload(result)
    _assert_actionable(payload, url, "write")
    assert payload["details"]["allowed_url_schemes"] == []


def test_usd_registration_denials_are_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = _make_server(monkeypatch)

    result = asyncio.run(_tool(instance, "load_usd_file").func(file_path=OUTSIDE_SANDBOX))
    _assert_actionable(_payload(result), OUTSIDE_SANDBOX, "read")

    result = asyncio.run(_tool(instance, "validate_usd_file").func(file_path=OUTSIDE_SANDBOX))
    _assert_actionable(_payload(result), OUTSIDE_SANDBOX, "read")


def test_backend_envelope_turns_sandbox_denied_into_sandbox_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Blender and Unreal sessions raise; the shared envelope must keep the details."""
    instance = _make_server(monkeypatch)
    denial = SandboxDenied(
        OUTSIDE_SANDBOX, instance._path_policy.denial_details(OUTSIDE_SANDBOX, write=True)
    )

    class _Session:
        def __enter__(self) -> "_Session":
            return self

        def __exit__(self, *exc: Any) -> None:
            return None

    adapter = SimpleNamespace(is_available=lambda: True, create_session=lambda: _Session())

    def _call(session: Any) -> Dict[str, Any]:
        raise denial

    result = asyncio.run(
        instance._exec_backend("export_unreal_usd", adapter, "Unreal", ErrorResponse, _call)
    )

    _assert_actionable(_payload(result), OUTSIDE_SANDBOX, "write")
