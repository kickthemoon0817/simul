"""Every paged tool declares its page, and every declaration matches its tool.

The result budget restates ``next_offset`` only for the lists a tool declares
in ``PAGED_RESULTS``. A declaration that names a renamed tool, a parameter the
tool does not take or a list its result does not carry would leave that tool's
pages unprotected; a paged tool missing from the table would get the
cut-for-good treatment on every budget cut. Both drift silently, so this test
reads each backend's tool definitions:

- Isaac: the ``IsaacTools`` method (its script builds the result);
- Unreal and Blender: the tool's declared response model;
- USD: the response model its wrapper validates against.
"""

from __future__ import annotations

import asyncio
import inspect
import re
from typing import Any, Dict, Optional, Set

import pytest
from fastmcp import Client

from simul.config import Settings
from simul.mcp import backends as backends_module
from simul.mcp import server as server_module
from simul.mcp.paged_results import PAGED_RESULTS, PagedList
from simul.mcp.registration import _reg_usd
from simul.mcp.registration._reg_blender import BLENDER_TOOLS
from simul.mcp.registration._unreal_tools import UNREAL_TOOLS
from simul.mcp.tools._meta import TOOL_META_ATTRIBUTE
from simul.mcp.tools.isaac_tools import IsaacTools


@pytest.fixture(scope="module")
def input_schemas() -> Dict[str, Dict[str, Any]]:
    """Every tool on every backend, as a real MCP client lists it."""
    settings = Settings().model_copy(
        update={"unreal": Settings().unreal.model_copy(update={"tool_surface": "full"})}
    )
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(backends_module, "is_blender_available", lambda: True)
        patcher.setattr(backends_module.BlenderRuntimeAdapter, "is_available", lambda self: True)
        instance = server_module.SimulMCPServer(settings, backends={"isaac", "usd", "unreal", "blender"})

        async def _run() -> Dict[str, Dict[str, Any]]:
            async with Client(instance.mcp) as client:
                return {tool.name: tool.input_schema for tool in await client.list_tools()}

        return asyncio.run(_run())


def _isaac_sources() -> Dict[str, str]:
    sources: Dict[str, str] = {}
    for _name, member in inspect.getmembers(IsaacTools, predicate=inspect.isfunction):
        meta = getattr(member, TOOL_META_ATTRIBUTE, None)
        if meta is not None:
            sources[meta.name] = inspect.getsource(member)
    return sources


def _usd_response_models() -> Dict[str, Any]:
    source = inspect.getsource(_reg_usd)
    pairs = re.findall(r'_exec_backend\(\s*"(\w+)",\s*server\.headless_adapter,\s*"USD",\s*(\w+),', source)
    return {tool: getattr(_reg_usd, model) for tool, model in pairs}


def _response_models() -> Dict[str, Any]:
    models: Dict[str, Any] = {spec.name: spec.response for spec in UNREAL_TOOLS}
    models.update({spec.name: spec.response for spec in BLENDER_TOOLS})
    models.update(_usd_response_models())
    return models


def _result_fields(tool: str) -> Optional[Set[str]]:
    """The top-level result keys a tool can produce, or None when unknown."""
    models = _response_models()
    if tool in models:
        return set(models[tool].model_fields)
    source = _isaac_sources().get(tool)
    if source is not None:
        return set(re.findall(r'"(\w+)":', source))
    return None


def _is_paged(tool: str) -> bool:
    fields = _result_fields(tool)
    return bool(fields) and "next_offset" in fields


@pytest.mark.parametrize("tool, paging", sorted(PAGED_RESULTS.items()))
def test_declaration_matches_its_tool(input_schemas: Dict[str, Dict[str, Any]], tool: str, paging: PagedList) -> None:
    assert tool in input_schemas, f"{tool} is declared as paged but no backend registers it"
    params = input_schemas[tool].get("properties", {})
    assert paging.offset_param in params, f"{tool} takes no {paging.offset_param!r}"
    if paging.limit_param is not None:
        assert paging.limit_param in params, f"{tool} takes no {paging.limit_param!r}"
    fields = _result_fields(tool)
    assert fields is not None, f"cannot read {tool}'s result fields"
    assert "next_offset" in fields, f"{tool} is declared as paged but its result has no next_offset"
    for field in paging.fields:
        root = field.split(".")[0].removesuffix("[*]")
        assert root in fields, f"{tool} declares {field!r} but its result has no {root!r}"


def test_every_paged_tool_is_declared(input_schemas: Dict[str, Dict[str, Any]]) -> None:
    undeclared = sorted(tool for tool in input_schemas if _is_paged(tool) and tool not in PAGED_RESULTS)

    assert not undeclared, f"paged tools missing from PAGED_RESULTS: {undeclared}"


def test_the_drift_check_sees_each_backend() -> None:
    """The reverse check is only as good as what it reads: each backend must contribute."""
    assert _is_paged("list_isaac_prims")
    assert _is_paged("list_unreal_actors")
    assert _is_paged("list_blender_scene_objects")
    assert _is_paged("search_prims")
    assert not _is_paged("get_prim_info")
