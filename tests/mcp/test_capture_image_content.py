"""Captures arrive as an image the client can render, not as base64 in JSON.

A 1280x720 capture returned inline was 49k characters of base64 inside a JSON
text block: roughly 12k tokens the model could not look at. As an MCP
``ImageContent`` block the same bytes are an image, and the JSON that travels
with it is the small path/size record it should have been all along.
"""

from __future__ import annotations

import asyncio
import base64
import json
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List

import pytest
from fastmcp import Client
from mcp.types import ImageContent, TextContent


from simul_mcp.config import Settings
from simul_mcp.mcp import backends as backends_module
from simul_mcp.mcp import server as server_module
from simul_mcp.mcp.schemas.blender import (
    BlenderCaptureViewportRequest,
    BlenderCaptureViewportResponse,
)
from simul_mcp.mcp.schemas.unreal import UnrealCaptureViewportResponse
from tests.fakes import FakeFastMCP

# A 1x1 PNG.
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)
PNG_B64 = base64.b64encode(PNG_BYTES).decode("ascii")
# The start of a JFIF JPEG: SOI marker then the APP0 segment.
JPEG_BYTES = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
JPEG_B64 = base64.b64encode(JPEG_BYTES).decode("ascii")


def _make_server(monkeypatch: pytest.MonkeyPatch) -> server_module.SimulMCPServer:
    monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
    monkeypatch.setattr(server_module, "TaskConfig", None)
    monkeypatch.setattr(backends_module, "is_headless_available", lambda: False)
    monkeypatch.setattr(backends_module, "is_blender_available", lambda: False)
    monkeypatch.setattr(backends_module, "UnrealRuntimeAdapter", None)
    return server_module.SimulMCPServer(settings=Settings())


def _capture_payload(image: str = PNG_B64, **extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "success": True,
        "path": "/tmp/simul_mcp/captures/viewport_1.png",
        "width": 640,
        "height": 360,
        "format": "png",
        "size_bytes": len(PNG_BYTES),
    }
    if image:
        payload["image_base64"] = image
        payload["encoding"] = "base64"
    payload.update(extra)
    return payload


def _split(result: Any) -> tuple[List[ImageContent], Dict[str, Any]]:
    assert result.structured_content is None
    images = [block for block in result.content if isinstance(block, ImageContent)]
    texts = [block for block in result.content if isinstance(block, TextContent)]
    assert len(texts) == 1, "exactly one JSON block accompanies the image"
    return images, json.loads(texts[0].text)


async def _coro(payload: Dict[str, Any]) -> Dict[str, Any]:
    return payload


def test_inline_isaac_capture_is_an_image_block_plus_a_small_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = _make_server(monkeypatch)

    result = asyncio.run(
        instance._exec_isaac("capture_isaac_viewport", _coro(_capture_payload()))
    )

    images, record = _split(result)
    assert len(images) == 1
    assert images[0].mimeType == "image/png"
    assert images[0].data == PNG_B64
    assert "image_base64" not in record
    assert "encoding" not in record
    assert record["image_attached"] is True
    assert record["path"].endswith("viewport_1.png")
    assert record["size_bytes"] == len(PNG_BYTES)


def test_capture_without_inline_is_text_only(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = _make_server(monkeypatch)

    result = asyncio.run(
        instance._exec_isaac("capture_isaac_viewport", _coro(_capture_payload(image="")))
    )

    images, record = _split(result)
    assert images == []
    assert "image_attached" not in record
    assert record["path"]


def test_large_inline_image_does_not_count_against_the_text_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lift happens before the budget, so a real capture is not flagged."""
    instance = _make_server(monkeypatch)
    big_image = base64.b64encode(b"\x89" * 250_000).decode("ascii")

    result = asyncio.run(
        instance._exec_isaac("capture_isaac_viewport", _coro(_capture_payload(image=big_image)))
    )

    images, record = _split(result)
    assert images[0].data == big_image
    assert "oversized_bytes" not in record
    assert len(json.dumps(record)) < 1_000


class _CaptureSession:
    async def capture_viewport(self, **kwargs: Any) -> Dict[str, Any]:
        return {
            "path": "C:/Project/Saved/Screenshots/WindowsEditor/shot.jpeg",
            "size_bytes": len(PNG_BYTES),
            "resolution_x": kwargs["resolution_x"],
            "resolution_y": kwargs["resolution_y"],
            "format": kwargs["format"],
            "image_base64": JPEG_B64,
            "encoding": "base64",
        }


class _CaptureAdapter:
    def is_available(self) -> bool:
        return True

    @contextmanager
    def create_session(self) -> Iterator[_CaptureSession]:
        yield _CaptureSession()


def test_unreal_capture_carries_the_declared_image_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = _make_server(monkeypatch)

    result = asyncio.run(
        instance._exec_backend(
            "capture_unreal_viewport",
            _CaptureAdapter(),
            "Unreal",
            UnrealCaptureViewportResponse,
            lambda session: session.capture_viewport(
                resolution_x=320, resolution_y=180, format="jpeg", inline=True
            ),
        )
    )

    images, record = _split(result)
    assert len(images) == 1
    assert images[0].mimeType == "image/jpeg"
    assert record["success"] is True
    assert record["resolution_x"] == 320
    assert "image_base64" not in record


class _BlenderCaptureSession:
    def capture_viewport(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        # An add-on built before the fix returns JPEG bytes with no ``format`` key.
        return {
            "image_base64": JPEG_B64,
            "width": 64,
            "height": 64,
            "engine": "BLENDER_EEVEE_NEXT",
            "capture_method": "gpu_offscreen",
        }


class _BlenderCaptureAdapter:
    def is_available(self) -> bool:
        return True

    @contextmanager
    def create_session(self) -> Iterator[_BlenderCaptureSession]:
        yield _BlenderCaptureSession()


def test_blender_jpeg_capture_is_labelled_jpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    """Blender encodes JPEG; the image block must not claim image/png."""
    instance = _make_server(monkeypatch)

    result = asyncio.run(
        instance._exec_backend(
            "capture_blender_viewport",
            _BlenderCaptureAdapter(),
            "Blender",
            BlenderCaptureViewportResponse,
            lambda session: session.capture_viewport(64, 64, 85, False, agent_id="agent"),
        )
    )

    images, record = _split(result)
    assert images[0].mimeType == "image/jpeg"
    assert record["format"] == "jpeg"


@pytest.mark.parametrize(
    ("image", "declared", "expected"),
    [
        (JPEG_B64, None, "image/jpeg"),
        (JPEG_B64, "png", "image/jpeg"),  # the bytes beat a stale label
        (PNG_B64, "jpeg", "image/png"),
        (base64.b64encode(b"\x00" * 32).decode("ascii"), "webp", "image/webp"),
    ],
)
def test_image_mime_type_follows_the_bytes(
    monkeypatch: pytest.MonkeyPatch, image: str, declared: Any, expected: str
) -> None:
    instance = _make_server(monkeypatch)
    payload = _capture_payload(image=image)
    payload.pop("format")
    if declared is not None:
        payload["format"] = declared

    images, _record = _split(instance._as_text_result(payload))

    assert images[0].mimeType == expected


@pytest.mark.parametrize("label", ["   ", "agent\x00", "tab\there", "x" * 65, ""])
def test_blender_capture_rejects_labels_the_overlay_cannot_draw(label: str) -> None:
    """Checked in the request model, so attached and embedded modes agree."""
    with pytest.raises(ValueError):
        BlenderCaptureViewportRequest(agent_id=label)


def test_real_fastmcp_delivers_the_image_block_to_the_client() -> None:
    """Through the real protocol the client sees an image, and no duplicate."""
    instance = server_module.SimulMCPServer(Settings(), backends={"isaac"})

    async def _capture(**kwargs: Any) -> Dict[str, Any]:
        return _capture_payload(width=kwargs["width"], height=kwargs["height"])

    instance._isaac_tools.capture_isaac_viewport = _capture  # type: ignore[method-assign]

    async def _run() -> Any:
        async with Client(instance.mcp) as client:
            return await client.call_tool(
                "capture_isaac_viewport", {"width": 320, "height": 180, "inline": True}
            )

    result = asyncio.run(_run())

    assert result.structured_content is None
    images, record = _split(result)
    assert images[0].data == PNG_B64
    assert record["width"] == 320
