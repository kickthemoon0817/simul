"""Regression: a viewport capture must not inline a megabyte of base64.

``capture_isaac_viewport`` encoded the whole PNG into the JSON response. A
1280x720 capture is roughly 300 KB to 1.2 MB encoded, which overruns a client's
per-result budget — and in Claude Code that reroutes the result to a file pointer,
which the persistent-mode hooks then misread as "the agent paused" (#87).

Width and height were clamped only to 7680 each, so 7680x7680 was a legal
request: tens of megabytes, encoded through several buffers, only to fail at the
transport's 10 MB cap *after* all the work was done.

Captures now return a path by default. Inline base64 is opt-in and refuses
politely above a cap rather than emitting a payload nothing can accept.
"""

from __future__ import annotations

import ast
import asyncio
import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest


from simul_mcp.adapters.isaac_socket_client import ScriptResult
from simul_mcp.config import Settings
from simul_mcp.mcp.tools.isaac_tools import MAX_INLINE_CAPTURE_BYTES, IsaacTools


def _tools(response: Dict[str, Any]) -> tuple[IsaacTools, List[str]]:
    captured: List[str] = []

    def _record(code: str) -> ScriptResult:
        captured.append(code)
        return ScriptResult(success=True, output=json.dumps(response))

    client = MagicMock()
    client.address = "127.0.0.1:8226"
    client.timeout_seconds = 30.0
    client.bridge_enabled = False
    client.fallback_to_vscode = True
    client.execute = AsyncMock(side_effect=_record)
    client.bridge_request = AsyncMock(return_value=None)
    return IsaacTools(client, settings=Settings()), captured


@pytest.mark.parametrize("inline", [False, True])
def test_generated_capture_script_is_valid_python(inline: bool) -> None:
    """The script is assembled from an interpolated block, so check it parses.

    Substring assertions happily pass on a script whose indentation is wrong;
    only parsing catches that. The script uses top-level await, which Kit's
    async exec allows, so it is wrapped in a coroutine before parsing.
    """
    tools, captured = _tools({"path": "/tmp/capture.png"})
    asyncio.run(tools.capture_isaac_viewport(inline=inline))

    body = "\n".join("    " + line for line in captured[0].splitlines())
    ast.parse(f"async def _capture():\n{body}")


def test_default_capture_never_encodes_base64() -> None:
    """The default path must not build a payload the client cannot accept."""
    tools, captured = _tools({"path": "/tmp/capture.png", "size_bytes": 421_000})

    result = asyncio.run(tools.capture_isaac_viewport())

    script = captured[0]
    assert "b64encode" not in script, "default capture still encodes the image inline"
    assert result["path"] == "/tmp/capture.png"
    assert "image_base64" not in result


def test_default_capture_keeps_the_file_for_the_caller() -> None:
    """Returning a path is useless if the script deletes the file behind it."""
    tools, captured = _tools({"path": "/tmp/capture.png"})
    asyncio.run(tools.capture_isaac_viewport())

    assert "os.remove(tmp_path)" not in captured[0]


def test_successive_captures_do_not_overwrite_each_other() -> None:
    """A/B comparison needs two files, not one path written twice."""
    tools, captured = _tools({"path": "/tmp/capture.png"})
    asyncio.run(tools.capture_isaac_viewport())

    script = captured[0]
    assert "_simul_mcp_capture.png" not in script, "capture path is still a fixed name"
    assert "uuid" in script


def test_inline_capture_is_guarded_by_a_size_cap() -> None:
    """Opting into base64 must still refuse a payload nothing can accept."""
    tools, captured = _tools({"image_base64": "iVBOR", "format": "png"})
    asyncio.run(tools.capture_isaac_viewport(inline=True))

    script = captured[0]
    assert "b64encode" in script, "inline capture should encode"
    assert str(MAX_INLINE_CAPTURE_BYTES) in script, "inline capture has no size cap"


def test_capture_resolution_is_clamped_below_the_old_ceiling() -> None:
    """7680x7680 was legal and is ~59 megapixels."""
    tools, captured = _tools({"path": "/tmp/capture.png"})
    asyncio.run(tools.capture_isaac_viewport(width=99999, height=99999))

    script = captured[0]
    assert "99999" not in script
    assert "7680" not in script, "resolution ceiling is still 7680"
