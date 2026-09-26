"""Capture tools deliver what their schema and description say (issue #221).

The server lifts ``image_base64`` out of a capture payload into an MCP
``ImageContent`` block and marks the JSON with ``image_attached``. A response
model that still declared ``image_base64`` told clients to look for the image
in the one place it never is. These tests call each capture tool through the
real FastMCP protocol with a fake session and check the blocks the client gets
against the response model's published JSON schema.
"""

from __future__ import annotations

import asyncio
import base64
import json
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Tuple, Type

import jsonschema
from fastmcp import Client
from mcp.types import ImageContent, TextContent
from pydantic import BaseModel

from simul.config import Settings
from simul.mcp import server as server_module
from simul.mcp.registration import register_blender_tools, register_unreal_tools
from simul.mcp.schemas.blender import (
    BlenderCaptureSequenceResponse,
    BlenderCaptureViewportResponse,
)
from simul.mcp.schemas.unreal import (
    UnrealCaptureViewportResponse,
    UnrealGetThumbnailResponse,
)
from tests.fakes import AvailableAdapter

# The start of a JFIF JPEG: SOI marker then the APP0 segment.
JPEG_B64 = base64.b64encode(
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
).decode("ascii")
# A 1x1 PNG.
PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


class _FakeBlenderSession:
    def capture_viewport(self, width: int, height: int, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {
            "image_base64": JPEG_B64,
            "width": width,
            "height": height,
            "engine": "BLENDER_EEVEE",
            "capture_method": "render_fallback",
            "format": "jpeg",
        }

    def capture_viewport_sequence(self, start: int, end: int, step: int, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        frames = [{"frame": f, "image_base64": JPEG_B64} for f in range(start, end + 1, step)]
        return {"frames": frames, "frame_count": len(frames), "capture_method": "gpu_offscreen"}


class _FakeUnrealSession:
    async def capture_viewport(self, **kwargs: Any) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "path": "/Project/Saved/Screenshots/shot.png",
            "size_bytes": 70,
            "resolution_x": kwargs["resolution_x"],
            "resolution_y": kwargs["resolution_y"],
            "format": kwargs["format"],
        }
        if kwargs["inline"]:
            payload["image_base64"] = PNG_B64
            payload["encoding"] = "base64"
        return payload

    async def get_actor_thumbnail(self, asset_path: str, width: int, height: int) -> Dict[str, Any]:
        return {
            "asset_path": asset_path,
            "image_base64": PNG_B64,
            "format": "png",
            "width": width,
            "height": height,
        }


class _Adapter(AvailableAdapter):
    def __init__(self, session: Any) -> None:
        super().__init__(Settings())
        self._session = session

    @contextmanager
    def create_session(self) -> Iterator[Any]:
        yield self._session


def _server() -> server_module.SimulMCPServer:
    instance = server_module.SimulMCPServer(Settings(), backends={"isaac"})
    instance.blender_adapter = _Adapter(_FakeBlenderSession())  # type: ignore[assignment]
    instance.unreal_adapter = _Adapter(_FakeUnrealSession())  # type: ignore[assignment]
    register_blender_tools(instance)
    register_unreal_tools(instance, thin=False)
    return instance


def _call(tool: str, arguments: Dict[str, Any]) -> Any:
    async def _run() -> Any:
        async with Client(_server().mcp) as client:
            return await client.call_tool(tool, arguments)

    return asyncio.run(_run())


def _blocks(result: Any) -> Tuple[List[ImageContent], Dict[str, Any]]:
    assert result.structured_content is None
    images = [block for block in result.content if isinstance(block, ImageContent)]
    texts = [block for block in result.content if isinstance(block, TextContent)]
    assert len(texts) == 1
    if images:
        assert result.content[0] is images[0], "the image block comes first"
    return images, json.loads(texts[0].text)


def _assert_matches_schema(record: Dict[str, Any], model: Type[BaseModel]) -> None:
    """The JSON block holds exactly what the model publishes, nothing more."""
    schema = model.model_json_schema()
    jsonschema.validate(record, schema)
    assert set(record) <= set(schema["properties"]), set(record) - set(schema["properties"])


def test_blender_capture_is_an_image_block_plus_the_documented_record() -> None:
    images, record = _blocks(
        _call("capture_blender_viewport", {"width": 64, "height": 96, "agent_id": "tester"})
    )

    assert len(images) == 1
    assert images[0].mimeType == "image/jpeg"
    assert images[0].data == JPEG_B64
    assert record["image_attached"] is True
    assert record["format"] == "jpeg"
    assert (record["width"], record["height"]) == (64, 96)
    _assert_matches_schema(record, BlenderCaptureViewportResponse)


def test_capture_schemas_do_not_publish_the_lifted_bytes() -> None:
    for model in (
        BlenderCaptureViewportResponse,
        UnrealCaptureViewportResponse,
        UnrealGetThumbnailResponse,
    ):
        properties = model.model_json_schema()["properties"]
        assert "image_base64" not in properties, model.__name__
        assert "encoding" not in properties, model.__name__
        assert "image_attached" in properties, model.__name__


def test_blender_capture_description_names_the_image_block() -> None:
    async def _describe() -> str:
        async with Client(_server().mcp) as client:
            tools = {tool.name: tool for tool in await client.list_tools()}
        return tools["capture_blender_viewport"].description or ""

    description = asyncio.run(_describe())
    assert "image content block" in description
    assert "base64" not in description


def test_blender_sequence_keeps_frames_inline_as_its_schema_says() -> None:
    images, record = _blocks(
        _call("capture_blender_viewport_sequence", {"start_frame": 1, "end_frame": 3})
    )

    assert images == []
    assert record["frame_count"] == 3
    assert all(frame["image_base64"] == JPEG_B64 for frame in record["frames"])
    _assert_matches_schema(record, BlenderCaptureSequenceResponse)


def test_unreal_inline_capture_matches_its_schema() -> None:
    images, record = _blocks(
        _call("capture_unreal_viewport", {"resolution_x": 32, "resolution_y": 16, "inline": True})
    )

    assert images[0].mimeType == "image/png"
    assert record["image_attached"] is True
    _assert_matches_schema(record, UnrealCaptureViewportResponse)


def test_unreal_path_only_capture_carries_no_transport_keys() -> None:
    images, record = _blocks(
        _call("capture_unreal_viewport", {"resolution_x": 32, "resolution_y": 16})
    )

    assert images == []
    assert record["image_attached"] is False
    assert "image_base64" not in record and "encoding" not in record
    _assert_matches_schema(record, UnrealCaptureViewportResponse)


def test_unreal_thumbnail_matches_its_schema() -> None:
    images, record = _blocks(
        _call("get_unreal_actor_thumbnail", {"asset_path": "/Game/Meshes/SM_Cube"})
    )

    assert images[0].data == PNG_B64
    assert record["image_attached"] is True
    _assert_matches_schema(record, UnrealGetThumbnailResponse)
