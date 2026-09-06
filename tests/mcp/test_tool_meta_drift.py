"""Every Isaac tool is declared once, on its implementation, and registered once.

``IsaacTools`` methods carry their MCP metadata as a ``@tool_meta`` decorator
and ``register_isaac_tools`` iterates those decorators. A public method that
forgets the decorator silently vanishes from the tool listing, and a wrapper
that drifts from its implementation advertises parameters the method does not
take; both are failures here rather than surprises in an agent session.
"""

from __future__ import annotations

import asyncio
import inspect
import typing
from collections import Counter
from types import SimpleNamespace
from typing import Any, Dict, List, Set, Type

import pytest

from simul_mcp.config import Settings
from simul_mcp.mcp import server as server_module
from simul_mcp.mcp.tools._meta import (
    DeprecatedAlias,
    SandboxedPath,
    ToolMeta,
    get_tool_meta,
    iter_tool_methods,
    tool_meta,
)
from simul_mcp.mcp.tools.isaac_tools import PRIM_DETAIL_ASPECTS, IsaacTools
from tests.fakes import FakeFastMCP

# Registered on every server but implemented outside IsaacTools: the Isaac
# instance routing tools and the usage statistics tool.
SERVER_TOOLS: Set[str] = {
    "list_isaac_instances",
    "set_active_isaac_instance",
    "claim_isaac_instance",
    "release_isaac_instance",
    "get_tool_usage_stats",
}


def _public_async_methods() -> Set[str]:
    return {
        name
        for name, member in inspect.getmembers(IsaacTools, inspect.iscoroutinefunction)
        if not name.startswith("_")
    }


def _decorated() -> Dict[str, ToolMeta]:
    return {name: meta for name, _function, meta in iter_tool_methods(IsaacTools)}


@pytest.fixture(scope="module")
def isaac_server() -> server_module.SimulMCPServer:
    """A server with only the Isaac backend registered on the recording double."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
        monkeypatch.setattr(server_module, "TaskConfig", None)
        return server_module.SimulMCPServer(Settings(), backends={"isaac"})


def test_every_public_method_is_a_tool_or_a_prim_detail_aspect() -> None:
    """The prim-detail aspect readers are folded into get_isaac_prim_detail; everything else is a tool."""
    aspect_readers = set(PRIM_DETAIL_ASPECTS.values())
    decorated = set(_decorated())
    undeclared = _public_async_methods() - aspect_readers - decorated
    assert not undeclared, f"public IsaacTools methods without @tool_meta: {sorted(undeclared)}"
    assert not (decorated & aspect_readers), "an aspect reader must not also be a tool"
    assert decorated <= _public_async_methods()


def test_tool_names_are_unique() -> None:
    names = Counter(meta.name for meta in _decorated().values())
    assert not [name for name, count in names.items() if count > 1]


def test_compound_and_prim_detail_tools_are_declared() -> None:
    assert get_tool_meta(IsaacTools.create_isaac_object) is not None
    assert get_tool_meta(IsaacTools.get_isaac_prim_detail) is not None


def test_each_declared_tool_registers_exactly_once(isaac_server: server_module.SimulMCPServer) -> None:
    registered = Counter(tool.name for tool in isaac_server.mcp.tools)
    expected = {meta.name for meta in _decorated().values()} | {"ping_isaac"}
    duplicates = sorted(name for name, count in registered.items() if count > 1)
    assert duplicates == []
    assert expected <= set(registered), f"declared but unregistered: {sorted(expected - set(registered))}"
    unexpected = set(registered) - expected - SERVER_TOOLS
    assert not unexpected, f"registered without a declaration: {sorted(unexpected)}"


def test_registered_metadata_comes_from_the_decorator(isaac_server: server_module.SimulMCPServer) -> None:
    by_name = {tool.name: tool for tool in isaac_server.mcp.tools}
    for meta in _decorated().values():
        tool = by_name[meta.name]
        assert tool.kwargs["description"] == meta.description
        hints = tool.kwargs["annotations"].model_dump(exclude_none=True)
        assert hints["readOnlyHint"] is meta.read_only, meta.name
        assert hints.get("destructiveHint", False) is meta.destructive, meta.name
        assert hints.get("idempotentHint", False) is meta.idempotent, meta.name
        assert hints.get("openWorldHint", True) is meta.open_world, meta.name


def test_wrapper_signature_mirrors_the_implementation(isaac_server: server_module.SimulMCPServer) -> None:
    """Parameters and defaults come from the method; aliases are added, hidden ones removed."""
    for method_name, implementation, meta in iter_tool_methods(IsaacTools):
        wrapper = isaac_server.mcp.by_name[meta.name]
        wrapper_parameters = inspect.signature(wrapper).parameters
        implementation_parameters = {
            name: parameter
            for name, parameter in inspect.signature(implementation).parameters.items()
            if name != "self" and name not in meta.hidden_parameters
        }
        expected_names = list(implementation_parameters) + [alias.alias for alias in meta.deprecated_aliases]
        assert list(wrapper_parameters) == expected_names, meta.name
        for name, parameter in implementation_parameters.items():
            assert wrapper_parameters[name].default == parameter.default, f"{meta.name}.{name}"
        hints = typing.get_type_hints(wrapper)
        for alias in meta.deprecated_aliases:
            assert wrapper_parameters[alias.alias].default is None
            assert alias.target in implementation_parameters
        assert hints["return"] is server_module.ToolResult


def test_deprecated_alias_is_forwarded_to_its_target(
    isaac_server: server_module.SimulMCPServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    received: List[Dict[str, Any]] = []

    async def _list_isaac_prims(**kwargs: Any) -> Dict[str, Any]:
        received.append(kwargs)
        return {"success": True}

    async def _exec(name: str, coro: Any, **kwargs: Any) -> Any:
        return await coro

    monkeypatch.setattr(isaac_server, "_isaac_tools", SimpleNamespace(list_isaac_prims=_list_isaac_prims))
    monkeypatch.setattr(isaac_server, "_exec_isaac", _exec)
    tool = isaac_server.mcp.by_name["list_isaac_prims"]

    asyncio.run(tool(max_items=7))
    assert received[-1]["max_results"] == 7
    assert "max_items" not in received[-1]

    asyncio.run(tool(max_results=3, max_items=7))
    assert received[-1]["max_results"] == 3


def test_hidden_parameter_is_not_advertised(isaac_server: server_module.SimulMCPServer) -> None:
    signature = inspect.signature(isaac_server.mcp.by_name["execute_isaac_script"])
    assert list(signature.parameters) == ["code"]


def test_sandboxed_paths_and_aliases_name_real_parameters() -> None:
    for _name, implementation, meta in iter_tool_methods(IsaacTools):
        parameters = set(inspect.signature(implementation).parameters) - {"self"}
        for guard in meta.sandboxed_paths:
            assert guard.parameter in parameters, meta.name
        for alias in meta.deprecated_aliases:
            assert alias.target in parameters and alias.alias not in parameters, meta.name


def test_decorator_rejects_metadata_that_names_a_missing_parameter() -> None:
    with pytest.raises(TypeError, match="sandboxed path"):

        @tool_meta(name="x", description="x", read_only=True, sandboxed_paths=(SandboxedPath("nope"),))
        async def _tool(self: Any, prim_path: str) -> Dict[str, Any]:
            return {}

    with pytest.raises(TypeError, match="alias"):

        @tool_meta(
            name="y", description="y", read_only=True, deprecated_aliases=(DeprecatedAlias("old", "nope"),)
        )
        async def _other(self: Any, prim_path: str) -> Dict[str, Any]:
            return {}

    with pytest.raises(TypeError, match="script"):

        @tool_meta(name="z", description="z", read_only=False, script=True)
        async def _script(self: Any, prim_path: str) -> Dict[str, Any]:
            return {}


def test_iter_tool_methods_prefers_a_subclass_override() -> None:
    class Base:
        @tool_meta(name="base_tool", description="base", read_only=True)
        async def act(self) -> Dict[str, Any]:
            return {}

    class Child(Base):
        @tool_meta(name="child_tool", description="child", read_only=True)
        async def act(self) -> Dict[str, Any]:
            return {}

    tools: Type[Base] = Child
    assert [meta.name for _name, _function, meta in iter_tool_methods(tools)] == ["child_tool"]
