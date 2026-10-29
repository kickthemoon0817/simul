"""add_blender_force_field without a location, through the real FastMCP protocol.

The tool lists ``location`` as optional and passes None when a caller omits
it, but the request model typed it ``List[float]`` (with a default list that
only applies when the key is absent), so every call without a location failed
validation before reaching Blender. The session places such a field at the
world origin.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List

import pytest
from fastmcp import Client
from mcp.types import TextContent

from simul.adapters import blender_runtime
from simul.config import Settings
from simul.mcp import server as server_module
from simul.mcp.registration import register_blender_tools
from tests.fakes import AvailableAdapter


class _Adapter(AvailableAdapter):
    def __init__(self, settings: Settings, session: Any) -> None:
        super().__init__(settings)
        self._session = session

    @contextmanager
    def create_session(self) -> Iterator[Any]:
        yield self._session


@pytest.fixture
def fake_bpy(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A bpy whose effector_add records its arguments and makes the field active."""
    calls: List[Dict[str, Any]] = []
    context = SimpleNamespace(active_object=None)

    def effector_add(**kwargs: Any) -> set:
        calls.append(kwargs)
        context.active_object = SimpleNamespace(
            name="Field",
            location=list(kwargs["location"]),
            field=SimpleNamespace(type=kwargs["type"], strength=1.0),
        )
        return {"FINISHED"}

    fake = SimpleNamespace(
        app=SimpleNamespace(version=(5, 0, 1)),
        context=context,
        ops=SimpleNamespace(object=SimpleNamespace(effector_add=effector_add)),
        effector_calls=calls,
    )
    monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
    monkeypatch.setattr(blender_runtime, "bpy", fake)
    return fake


def _server() -> server_module.SimulMCPServer:
    settings = Settings(security={"rate_limiting_enabled": False})
    instance = server_module.SimulMCPServer(settings, backends={"isaac"})
    session = blender_runtime.BlenderRuntimeSession(settings=settings)
    instance.blender_adapter = _Adapter(settings, session)  # type: ignore[assignment]
    register_blender_tools(instance)
    return instance


def _call(
    instance: server_module.SimulMCPServer, arguments: Dict[str, Any]
) -> Dict[str, Any]:
    async def _run() -> Any:
        async with Client(instance.mcp) as client:
            return await client.call_tool("add_blender_force_field", arguments)

    result = asyncio.run(_run())
    texts = [block for block in result.content if isinstance(block, TextContent)]
    return json.loads(texts[-1].text)


def test_location_is_listed_as_optional(fake_bpy: SimpleNamespace) -> None:
    async def _run() -> Any:
        async with Client(_server().mcp) as client:
            tools = {t.name: t for t in await client.list_tools()}
            return tools["add_blender_force_field"].inputSchema

    schema = asyncio.run(_run())
    assert schema["required"] == ["field_type"]


def test_a_field_without_location_is_added_at_the_origin(
    fake_bpy: SimpleNamespace,
) -> None:
    result = _call(_server(), {"field_type": "wind", "strength": 2.5})

    assert result["success"] is True, result
    assert result["location"] == [0.0, 0.0, 0.0]
    assert fake_bpy.effector_calls == [{"type": "WIND", "location": (0.0, 0.0, 0.0)}]
    assert fake_bpy.context.active_object.field.strength == 2.5


def test_an_explicit_location_is_used_and_still_validated(
    fake_bpy: SimpleNamespace,
) -> None:
    instance = _server()

    placed = _call(instance, {"field_type": "FORCE", "location": [1, 2, 3]})
    refused = _call(instance, {"field_type": "FORCE", "location": [1, 2]})

    assert placed["success"] is True, placed
    assert placed["location"] == [1.0, 2.0, 3.0]
    assert refused["success"] is False
    assert refused["error_type"] == "ValidationError"
    assert fake_bpy.effector_calls == [{"type": "FORCE", "location": (1.0, 2.0, 3.0)}]
