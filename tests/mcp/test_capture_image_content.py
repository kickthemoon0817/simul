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
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List

import pytest
from fastmcp import Client
from mcp.types import ImageContent, TextContent

src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_path))

from simul_mcp.config import Settings  # noqa: E402
from simul_mcp.mcp import backends as backends_module  # noqa: E402
from simul_mcp.mcp import server as server_module  # noqa: E402
from simul_mcp.mcp.schemas.unreal import UnrealCaptureViewportResponse  # noqa: E402

# A 1x1 PNG.
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)
PNG_B64 = base64.b64encode(PNG_BYTES).decode("ascii")


class FakeFastMCP:
    def __init__(self, name: str, version: str, **kwargs: Any):
        self.tools: List[SimpleNamespace] = []

    def tool(self, name: str, **kwargs: Any) -> Callable[..., Any]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.tools.append(SimpleNamespace(name=name, func=func))
            return func

        return decorator

    def resource(self, *args: Any, **kwargs: Any) -> Callable[..., Any]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator

    def add_middleware(self, middleware: Any) -> None:
        return


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
            "image_base64": PNG_B64,
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
