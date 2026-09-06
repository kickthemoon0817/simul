"""Regression: a tool result must cross the wire once, not twice.

Every tool was annotated ``-> Dict[str, Any]``, so FastMCP derived an output
schema and emitted *both* a JSON text block and an identical ``structuredContent``
dict. Every payload was therefore paid for twice in the caller's context window.

The declared schemas were not buying anything in exchange: ``_tool_output_schema``
returns a real pydantic schema only when handed exactly one model, and all 119
call sites pass ``(Model, ErrorResponse)`` — so the single-model branch was dead
and all tools shipped the same permissive ``{"type": "object",
"additionalProperties": true}`` stub.

Dropping both together is what keeps this coherent: a tool that declares no
output schema is not expected to return structured content.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest


from fastmcp.tools.tool import ToolResult

from simul_mcp.config import Settings
from simul_mcp.mcp import backends as backends_module
from simul_mcp.mcp import server as server_module
from tests.fakes import FakeFastMCP


def _make_server(monkeypatch: pytest.MonkeyPatch) -> server_module.SimulMCPServer:
    monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
    monkeypatch.setattr(server_module, "TaskConfig", None)
    monkeypatch.setattr(backends_module, "is_headless_available", lambda: False)
    monkeypatch.setattr(backends_module, "is_blender_available", lambda: False)
    monkeypatch.setattr(backends_module, "UnrealRuntimeAdapter", None)
    return server_module.SimulMCPServer(settings=Settings())


def _payload(result: Any) -> Dict[str, Any]:
    """Read a tool result's single JSON content block."""
    return json.loads(result.content[0].text)


def test_exec_isaac_emits_content_without_structured_duplicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = _make_server(monkeypatch)

    async def _payload_coro() -> Dict[str, Any]:
        return {"success": True, "up_axis": "Z", "total_prims": 754}

    result = asyncio.run(instance._exec_isaac("get_isaac_stage_info", _payload_coro()))

    # One JSON text block carrying the whole payload...
    assert len(result.content) == 1
    assert _payload(result) == {"success": True, "up_axis": "Z", "total_prims": 754}
    # ...and no second copy of it.
    assert result.structured_content is None


def test_error_payloads_are_also_sent_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """The error path must not keep the duplicate the success path lost."""
    instance = _make_server(monkeypatch)

    async def _boom() -> Dict[str, Any]:
        raise RuntimeError("isaac exploded")

    result = asyncio.run(instance._exec_isaac("get_isaac_stage_info", _boom()))

    assert result.structured_content is None
    assert _payload(result)["error_type"] == "RuntimeError"


def test_no_tool_declares_the_dead_permissive_output_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The declared schemas were an identical stub on every tool."""
    instance = _make_server(monkeypatch)

    # An explicit ``output_schema=None`` is how a dict-returning tool suppresses
    # the schema FastMCP would otherwise derive, so the key may be present — what
    # must not survive is a declared schema object.
    declaring = [
        tool.name
        for tool in instance.mcp.tools
        if tool.kwargs.get("output_schema") is not None
    ]

    assert declaring == [], (
        f"{len(declaring)} tools still declare an output schema, e.g. {declaring[:5]}"
    )


def test_annotations_drop_constant_noise(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hints equal to a client's assumed default are omitted, never emitted.

    destructiveHint=false rode on 84 tools while saying nothing; the same goes
    for idempotentHint=false and openWorldHint=true.
    """
    instance = _make_server(monkeypatch)

    annotated = [
        tool.kwargs["annotations"]
        for tool in instance.mcp.tools
        if tool.kwargs.get("annotations") is not None
    ]
    assert annotated, "expected tools to carry annotations"

    for annotations in annotated:
        dumped = annotations.model_dump(exclude_none=True)
        assert "readOnlyHint" in dumped
        assert dumped.get("idempotentHint") is not False
        assert dumped.get("openWorldHint") is not True
        assert dumped.get("destructiveHint") is not False


class _StubAdapter:
    """Reports available so the backend's tools register."""

    def __init__(self, settings: Any) -> None:
        pass

    def is_available(self) -> bool:
        return True


def _argument_for(parameter: inspect.Parameter) -> Any:
    """Pick a value of the right shape for a required parameter."""
    annotation = str(parameter.annotation)
    if "Optional" in annotation:
        return None
    if "List" in annotation or "list" in annotation:
        return [0.0, 0.0, 0.0]
    if "Dict" in annotation or "dict" in annotation:
        return {}
    if "bool" in annotation:
        return False
    if "int" in annotation:
        return 1
    if "float" in annotation:
        return 1.0
    return "x"


def test_every_registered_tool_is_single_transmission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every tool on the surface, called once, returns a content-only ToolResult.

    A tool that returns a plain dict is emitted twice by FastMCP. Backends are
    stubbed absent so each envelope short-circuits before any session opens;
    the Isaac envelope is replaced so no socket is touched.
    """
    monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
    monkeypatch.setattr(server_module, "TaskConfig", None)
    monkeypatch.setattr(backends_module, "is_blender_available", lambda: True)
    monkeypatch.setattr(backends_module, "BlenderRuntimeAdapter", _StubAdapter)
    monkeypatch.setattr(backends_module, "UnrealRuntimeAdapter", _StubAdapter)
    instance = server_module.SimulMCPServer(
        settings=Settings(
            security={"rate_limiting_enabled": False}, unreal={"tool_surface": "full"}
        )
    )
    instance.blender_adapter = SimpleNamespace(is_available=lambda: False)
    instance.unreal_adapter = SimpleNamespace(is_available=lambda: False)

    async def _exec_isaac(name: str, coro: Any, **kwargs: Any) -> ToolResult:
        coro.close()
        return instance._as_text_result({"success": True, "tool": name})

    async def _ping(self: Any) -> bool:
        return False

    async def _scan() -> Dict[str, Any]:
        return {}

    async def _brief(name: str, client: Any) -> Dict[str, Any]:
        return {"name": name, "port": client._port, "reachable": False}

    monkeypatch.setattr(instance, "_exec_isaac", _exec_isaac)
    monkeypatch.setattr(type(instance.client), "ping", _ping)
    monkeypatch.setattr(instance, "_scan_isaac_instances", _scan)
    monkeypatch.setattr(instance, "_get_instance_brief", _brief)

    assert len(instance.mcp.tools) > 150
    offenders: List[str] = []
    for tool in instance.mcp.tools:
        kwargs = {
            name: _argument_for(parameter)
            for name, parameter in inspect.signature(tool.func).parameters.items()
            if parameter.default is inspect.Parameter.empty
        }
        result = asyncio.run(tool.func(**kwargs))
        if not isinstance(result, ToolResult) or result.structured_content is not None:
            offenders.append(tool.name)

    assert offenders == [], f"{len(offenders)} tools still send their payload twice: {offenders}"


def test_every_isaac_tool_is_single_transmission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tools that bypass _exec_isaac must convert too, or they keep the duplicate.

    execute_isaac_script and ping_isaac build their own responses rather than
    going through _exec_isaac, so the central fix does not reach them.
    execute_isaac_script returns arbitrary script output — the largest payload
    on the surface — which makes it the worst one to leave duplicating.
    """
    instance = _make_server(monkeypatch)

    dict_returning = [
        tool.name
        for tool in instance.mcp.tools
        if tool.name.endswith("_isaac") or tool.name.startswith(("isaac_", "execute_isaac", "ping_isaac"))
    ]
    assert "ping_isaac" in dict_returning

    for tool in instance.mcp.tools:
        if tool.name not in {"execute_isaac_script", "ping_isaac"}:
            continue
        annotation = inspect.signature(tool.func).return_annotation
        assert "ToolResult" in str(annotation), (
            f"{tool.name} still returns a plain dict, so its payload is sent twice"
        )
