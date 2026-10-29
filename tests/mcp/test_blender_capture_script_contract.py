"""MCP contract of capture_blender_viewport (#228) and execute_blender_script (#231).

The tools are called through the real FastMCP protocol: what an agent can
send (the method enum, the bounded max_output_chars) and what it gets back
(an explicit truncation report, never a silent cut).
"""

from __future__ import annotations

import asyncio
import base64
import json
import typing
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
from simul.mcp.result_budget import (
    DEFAULT_RESULT_BUDGET_BYTES,
    HARD_RESULT_LIMIT_BYTES,
    encode_result_budget,
)
from simul.mcp.schemas import blender as blender_schemas
from tests.fakes import AvailableAdapter

JPEG_B64 = base64.b64encode(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00").decode("ascii")


class _CaptureSession:
    """Records the capture calls the tools make; the add-on side has its own tests."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def capture_viewport(
        self, width: int, height: int, jpeg_quality: int = 85, **kwargs: Any
    ) -> Dict[str, Any]:
        self.calls.append(kwargs)
        return {
            "image_base64": JPEG_B64,
            "width": width,
            "height": height // 2,
            "engine": "BLENDER_EEVEE",
            "capture_method": "viewport",
            "format": "jpeg",
        }

    def capture_viewport_sequence(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(kwargs)
        return {"frames": [], "frame_count": 0, "capture_method": kwargs["method"]}


class _Adapter(AvailableAdapter):
    def __init__(self, settings: Settings, session: Any) -> None:
        super().__init__(settings)
        self._session = session

    @contextmanager
    def create_session(self) -> Iterator[Any]:
        yield self._session


def _settings() -> Settings:
    return Settings(
        security={"allow_script_execution": True, "rate_limiting_enabled": False}
    )


def _server(session: Any) -> server_module.SimulMCPServer:
    settings = _settings()
    instance = server_module.SimulMCPServer(settings, backends={"isaac"})
    instance.blender_adapter = _Adapter(settings, session)  # type: ignore[assignment]
    register_blender_tools(instance)
    return instance


def _call(
    instance: server_module.SimulMCPServer, tool: str, arguments: Dict[str, Any]
) -> Dict[str, Any]:
    async def _run() -> Any:
        async with Client(instance.mcp) as client:
            return await client.call_tool(tool, arguments)

    result = asyncio.run(_run())
    texts = [block for block in result.content if isinstance(block, TextContent)]
    return json.loads(texts[-1].text)


def _listed(instance: server_module.SimulMCPServer, tool: str) -> Any:
    """Return the tool as a client lists it: name, description and inputSchema."""

    async def _run() -> Any:
        async with Client(instance.mcp) as client:
            return {t.name: t for t in await client.list_tools()}[tool]

    return asyncio.run(_run())


def _input_schema(instance: server_module.SimulMCPServer, tool: str) -> Dict[str, Any]:
    return _listed(instance, tool).inputSchema


@pytest.fixture
def embedded_bpy(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A bpy stand-in for running execute_script for real in this process."""
    fake = SimpleNamespace(
        app=SimpleNamespace(version=(5, 0, 1)),
        context=SimpleNamespace(scene=SimpleNamespace(name="Scene")),
    )
    monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
    monkeypatch.setattr(blender_runtime, "bpy", fake)
    return fake


# -- #231 ---------------------------------------------------------------------


def test_schema_mirrors_the_add_on_constants() -> None:
    """The add-on cannot import the schema, so the numbers are kept in step here."""
    assert (
        blender_schemas.SCRIPT_OUTPUT_CHARS_DEFAULT
        == blender_runtime.DEFAULT_SCRIPT_OUTPUT_CHARS
    )
    assert (
        blender_schemas.SCRIPT_OUTPUT_CHARS_MAX
        == blender_runtime.MAX_SCRIPT_OUTPUT_CHARS
    )
    assert (
        typing.get_args(blender_schemas.BlenderCaptureMethod)
        == blender_runtime.CAPTURE_METHODS
    )


def test_output_caps_fit_what_the_server_delivers() -> None:
    """A cap above the server's result limits would only move the cut elsewhere.

    The default keeps a typical printed JSON dump inside the 40 KB budget, so it
    arrives unflagged. The maximum keeps even output made only of characters JSON
    escapes (quotes, newlines) under the 400 KB hard limit, above which the server
    replaces the whole field with a marker instead of passing a prefix through.
    """
    rows = [
        {"name": f"Cube.{i:03d}", "location": [1.25, -2.5, 0.125]} for i in range(2000)
    ]
    printed = json.dumps(rows)[: blender_runtime.DEFAULT_SCRIPT_OUTPUT_CHARS]
    typical = {
        "success": True,
        "output": printed,
        "output_truncated": True,
        "output_total_chars": 10**6,
    }
    assert len(encode_result_budget(typical)) <= DEFAULT_RESULT_BUDGET_BYTES
    assert "oversized_bytes" not in json.loads(encode_result_budget(typical))

    escaped = '"\n' * (blender_runtime.MAX_SCRIPT_OUTPUT_CHARS // 2)
    worst = {
        "success": True,
        "output": escaped,
        "output_truncated": True,
        "output_total_chars": 10**6,
    }
    delivered = json.loads(encode_result_budget(worst))
    assert len(json.dumps(worst)) <= HARD_RESULT_LIMIT_BYTES
    assert delivered["output"] == escaped, "the prefix is delivered, not replaced"


def test_the_cap_and_how_to_get_more_are_advertised() -> None:
    tool = _listed(_server(None), "execute_blender_script")
    schema = tool.inputSchema["properties"]["max_output_chars"]
    assert schema["default"] == 32_000
    assert "1 to 100000" in schema["description"]
    assert "output_truncated" in schema["description"]
    assert "default 32000, at most 100000" in tool.description
    assert "output_truncated=true" in tool.description
    assert "Write larger results to a file" in tool.description


def test_6000_printed_chars_come_back_whole(embedded_bpy: SimpleNamespace) -> None:
    """Issue #231's reproduction, through the MCP tool and a real session."""
    session = blender_runtime.BlenderRuntimeSession(settings=_settings())
    result = _call(
        _server(session), "execute_blender_script", {"script": "print('x' * 6000)"}
    )

    assert result["success"] is True
    assert result["output"] == "x" * 6000 + "\n"
    assert result["output_truncated"] is False
    assert result["output_total_chars"] == 6001


def test_a_cut_is_reported_explicitly(embedded_bpy: SimpleNamespace) -> None:
    session = blender_runtime.BlenderRuntimeSession(settings=_settings())
    result = _call(
        _server(session),
        "execute_blender_script",
        {"script": "print('x' * 6000)", "max_output_chars": 4096},
    )

    assert result["success"] is True
    assert result["output"] == "x" * 4096
    assert result["output_truncated"] is True
    assert result["output_total_chars"] == 6001


@pytest.mark.parametrize("bad", [0, 100_001])
def test_out_of_range_caps_are_refused_before_blender_runs_anything(
    embedded_bpy: SimpleNamespace, bad: int
) -> None:
    ran: List[str] = []
    embedded_bpy.ran = ran
    session = blender_runtime.BlenderRuntimeSession(settings=_settings())
    result = _call(
        _server(session),
        "execute_blender_script",
        {"script": "bpy.ran.append('ran')", "max_output_chars": bad},
    )
    assert result["success"] is False
    assert ran == []


# -- #228 ---------------------------------------------------------------------


def test_capture_advertises_the_method_choice() -> None:
    properties = _input_schema(_server(_CaptureSession()), "capture_blender_viewport")[
        "properties"
    ]
    method = properties["method"]
    assert method["default"] == "auto"
    assert set(method["enum"]) == {"auto", "viewport", "render"}
    assert "Deprecated" in properties["use_render_fallback"]["description"]
    sequence = _input_schema(
        _server(_CaptureSession()), "capture_blender_viewport_sequence"
    )
    assert set(sequence["properties"]["method"]["enum"]) == {
        "auto",
        "viewport",
        "render",
    }


@pytest.mark.parametrize(
    ("arguments", "method"),
    [
        ({}, "auto"),
        ({"method": "viewport"}, "viewport"),
        ({"method": "render"}, "render"),
        ({"use_render_fallback": True}, "render"),
    ],
)
def test_capture_forwards_the_method(arguments: Dict[str, Any], method: str) -> None:
    session = _CaptureSession()
    result = _call(
        _server(session),
        "capture_blender_viewport",
        {"width": 256, "height": 256, **arguments},
    )

    assert result["success"] is True
    assert (result["width"], result["height"]) == (256, 128)
    assert result["capture_method"] == "viewport"
    # The deprecated flag is folded into method; only method reaches the add-on.
    (call,) = session.calls
    assert call["method"] == method
    assert set(call) == {"agent_id", "method"}


def test_capture_refuses_contradicting_arguments_without_calling_blender() -> None:
    session = _CaptureSession()
    result = _call(
        _server(session),
        "capture_blender_viewport",
        {"method": "viewport", "use_render_fallback": True},
    )
    assert result["success"] is False
    assert "contradicts" in result["error"]
    assert session.calls == []


def test_sequence_forwards_the_method() -> None:
    session = _CaptureSession()
    result = _call(
        _server(session),
        "capture_blender_viewport_sequence",
        {"start_frame": 1, "end_frame": 2, "method": "render"},
    )
    assert result["success"] is True and result["capture_method"] == "render"
    assert session.calls[0]["method"] == "render"
