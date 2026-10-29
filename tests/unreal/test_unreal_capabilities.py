"""The Unreal adapter's capabilities are the tools the server registers.

``UnrealRuntimeAdapter.get_capabilities`` used to hard-code 53 tool names and
missed six registered tools (``ping_unreal``, ``list_unreal_instances``,
``execute_unreal_script``, ``control_unreal_ui``, ``convert_to_simready``,
``validate_simready_asset``). It is now read from the declarations, so these
tests compare it with what a server actually registers in every surface and
script-execution configuration.
"""

from __future__ import annotations

from typing import List

import pytest

from simul.adapters import unreal_runtime
from simul.config import Settings
from simul.mcp import backends as backends_module
from simul.mcp import server as server_module
from simul.mcp.registration._unreal_tools import UNREAL_TOOLS, registered_tool_names, surface_tools
from simul.tool_surfaces import THIN_UNREAL_TOOLS
from tests.fakes import FakeFastMCP

CONFIGS = [(surface, scripts) for surface in ("full", "thin") for scripts in (True, False)]
SCRIPT_TOOLS = {"execute_unreal_script", "call_unreal_actor_function", "batch_unreal_operations"}


def _settings(surface: str, scripts: bool) -> Settings:
    return Settings(unreal={"tool_surface": surface}, security={"allow_script_execution": scripts})


def _server(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> server_module.SimulMCPServer:
    """A server with the real Unreal adapter, registering on the recording double."""
    monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
    monkeypatch.setattr(server_module, "TaskConfig", None)
    monkeypatch.setattr(backends_module, "is_unreal_available", lambda: True)
    monkeypatch.setattr(unreal_runtime, "UNREAL_AVAILABLE", True)
    return server_module.SimulMCPServer(settings, backends={"unreal"})


def _registered(instance: server_module.SimulMCPServer) -> List[str]:
    return [tool.name for tool in instance.mcp.tools if tool.name != "get_tool_usage_stats"]


@pytest.mark.parametrize("surface, scripts", CONFIGS)
def test_capabilities_are_the_registered_tools(
    monkeypatch: pytest.MonkeyPatch, surface: str, scripts: bool
) -> None:
    instance = _server(monkeypatch, _settings(surface, scripts))
    assert isinstance(instance.unreal_adapter, unreal_runtime.UnrealRuntimeAdapter)

    registered = _registered(instance)

    assert instance.unreal_adapter.get_capabilities() == registered
    assert instance.get_capabilities()["unreal"]["capabilities"] == sorted(registered)


def test_full_surface_reports_every_declared_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(unreal_runtime, "UNREAL_AVAILABLE", True)
    adapter = unreal_runtime.UnrealRuntimeAdapter(_settings("full", True))

    capabilities = adapter.get_capabilities()

    assert capabilities == [spec.name for spec in UNREAL_TOOLS]
    # The six the hard-coded list left out.
    assert {
        "ping_unreal",
        "list_unreal_instances",
        "execute_unreal_script",
        "control_unreal_ui",
        "convert_to_simready",
        "validate_simready_asset",
    } <= set(capabilities)


def test_script_tools_follow_the_script_execution_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(unreal_runtime, "UNREAL_AVAILABLE", True)

    allowed = unreal_runtime.UnrealRuntimeAdapter(_settings("full", True)).get_capabilities()
    denied = unreal_runtime.UnrealRuntimeAdapter(_settings("full", False)).get_capabilities()

    assert set(allowed) - set(denied) == SCRIPT_TOOLS
    assert denied == [name for name in allowed if name not in SCRIPT_TOOLS]


def test_thin_surface_reports_the_thin_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(unreal_runtime, "UNREAL_AVAILABLE", True)

    thin = unreal_runtime.UnrealRuntimeAdapter(_settings("thin", True)).get_capabilities()
    thin_no_scripts = unreal_runtime.UnrealRuntimeAdapter(_settings("thin", False)).get_capabilities()

    assert set(thin) == set(THIN_UNREAL_TOOLS)
    assert thin_no_scripts == [name for name in thin if name != "execute_unreal_script"]


def test_an_unavailable_adapter_reports_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(unreal_runtime, "UNREAL_AVAILABLE", False)

    assert unreal_runtime.UnrealRuntimeAdapter(_settings("full", True)).get_capabilities() == []


def test_surface_tools_keep_registration_order() -> None:
    assert surface_tools(False) == UNREAL_TOOLS
    assert [spec.name for spec in surface_tools(True)] == [
        spec.name for spec in UNREAL_TOOLS if spec.name in THIN_UNREAL_TOOLS
    ]
    assert registered_tool_names(_settings("full", True)) == tuple(spec.name for spec in UNREAL_TOOLS)
