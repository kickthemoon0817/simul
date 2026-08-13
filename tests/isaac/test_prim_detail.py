"""Regression: one prim-detail tool instead of fifteen near-identical ones.

Fifteen tools took exactly one ``prim_path`` and differed only in which aspect of
the prim they read. Measured against the real ``tools/list`` payload they cost
~1,779 tokens of every session's listing, and an agent had to discriminate
between fifteen entries whose descriptions are one clause apart.

``get_isaac_prim_detail`` replaces them. The originals stay registered as
deprecated aliases for one release, so nothing breaks on upgrade.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_path))

from simul_mcp.adapters.isaac_socket_client import ScriptResult
from simul_mcp.config import Settings
from simul_mcp.mcp.tools.isaac_tools import PRIM_DETAIL_ASPECTS, IsaacTools


def _tools(payload: Dict[str, Any] | None = None) -> IsaacTools:
    client = MagicMock()
    client.address = "127.0.0.1:8226"
    client.timeout_seconds = 30.0
    client.bridge_enabled = False
    client.fallback_to_vscode = True
    client.execute = AsyncMock(
        return_value=ScriptResult(
            success=True, output=json.dumps(payload or {"ok": True})
        )
    )
    client.bridge_request = AsyncMock(return_value=None)
    return IsaacTools(client, settings=Settings())


def test_every_aspect_maps_to_a_real_method() -> None:
    """The map is the contract; a typo in it would fail only at call time."""
    tools = _tools()
    for aspect, method_name in PRIM_DETAIL_ASPECTS.items():
        assert hasattr(tools, method_name), f"{aspect} -> missing {method_name}"


def test_default_returns_the_cheap_aspect() -> None:
    tools = _tools({"type": "Mesh"})

    result = asyncio.run(tools.get_isaac_prim_detail(prim_path="/World/Mesh"))

    assert result["prim_path"] == "/World/Mesh"
    assert result["aspects"] == ["info"]
    assert result["info"]["type"] == "Mesh"


def test_requested_aspects_are_merged_under_their_names() -> None:
    tools = _tools({"value": 1})

    result = asyncio.run(
        tools.get_isaac_prim_detail(
            prim_path="/World/Mesh", aspects=["info", "transform", "bounding_box"]
        )
    )

    assert result["aspects"] == ["info", "transform", "bounding_box"]
    for aspect in ("info", "transform", "bounding_box"):
        assert result[aspect]["value"] == 1


def test_unknown_aspect_names_the_valid_ones() -> None:
    tools = _tools()

    result = asyncio.run(
        tools.get_isaac_prim_detail(prim_path="/World/Mesh", aspects=["nonsense"])
    )

    assert result["error_type"] == "ValueError"
    assert "nonsense" in result["error"]
    assert "transform" in result["error"]


def test_a_failing_aspect_does_not_lose_the_others() -> None:
    """One unreadable aspect must not discard the rest of the answer."""
    tools = _tools()
    calls: List[str] = []

    async def _ok(path: str, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        calls.append("info")
        return {"success": True, "type": "Mesh"}

    async def _fails(path: str, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        calls.append("mesh")
        return {"success": False, "error": "not a mesh", "error_type": "TypeError"}

    tools.get_isaac_prim_info = _ok  # type: ignore[assignment]
    tools.get_isaac_mesh_info = _fails  # type: ignore[assignment]

    result = asyncio.run(
        tools.get_isaac_prim_detail(prim_path="/World/X", aspects=["info", "mesh"])
    )

    assert calls == ["info", "mesh"]
    assert result["info"]["type"] == "Mesh"
    assert result["mesh"]["error"] == "not a mesh"
