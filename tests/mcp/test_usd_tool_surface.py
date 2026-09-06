"""Headless USD tools must work end to end and stay confined to their backend.

These tests deliberately use the real FastMCP rather than a test double: the
failure they guard against was an adapter import error that every FakeFastMCP
test stubbed away, leaving nine of ten USD tools registered but erroring on
every call.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastmcp import Client


from simul_mcp.adapters import is_headless_available
from simul_mcp.config import Settings
from simul_mcp.mcp.server import SimulMCPServer

FIXTURE_SCENE: Path = Path(__file__).resolve().parents[1] / "data" / "simple_scene.usda"


class UsdToolDriver:
    """Drive one server's tools through the real MCP protocol."""

    def __init__(self, server: SimulMCPServer) -> None:
        self._server = server

    def list_tool_names(self) -> List[str]:
        """Return the tool names the FastMCP instance advertises."""

        async def _run() -> List[str]:
            async with Client(self._server.mcp) as client:
                return sorted(tool.name for tool in await client.list_tools())

        return asyncio.run(_run())

    def call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke one tool and decode its JSON payload.

        Args:
            name: Registered tool name.
            arguments: Tool arguments as the MCP client would send them.

        Returns:
            The tool's dict payload.
        """

        async def _run() -> Dict[str, Any]:
            async with Client(self._server.mcp) as client:
                result = await client.call_tool(name, arguments)
                # The payload must cross the wire once: as JSON text, with no
                # structured_content duplicate riding along.
                assert result.structured_content is None, name
                assert len(result.content) == 1, name
                return json.loads(result.content[0].text)

        return asyncio.run(_run())


@pytest.fixture
def usd_driver() -> UsdToolDriver:
    return UsdToolDriver(SimulMCPServer(Settings(), backends={"usd"}))


def _assert_ok(payload: Dict[str, Any]) -> None:
    """USD responses carry ``success`` only on some schemas, but every error does."""
    assert payload.get("success", True) is True, payload
    assert "error_type" not in payload, payload


def test_headless_usd_adapter_is_available() -> None:
    """usd-core is a hard dependency, so the headless adapter must import."""
    assert is_headless_available() is True


def test_load_usd_file_end_to_end(usd_driver: UsdToolDriver) -> None:
    """A USD-only server loads a fixture stage through the real MCP stack."""
    payload = usd_driver.call("load_usd_file", {"file_path": str(FIXTURE_SCENE)})

    _assert_ok(payload)
    assert payload["stage_id"]
    assert payload["up_axis"] == "Z"
    assert payload["default_prim"] == "/World"
    assert payload["total_prims"] == 3


def test_stage_id_survives_across_tool_calls(usd_driver: UsdToolDriver) -> None:
    """Every stage-bound tool must find the stage that load_usd_file returned.

    Each tool call opens its own adapter session, so the stage registry has to
    outlive a single call or the ``stage_id`` workflow can never work.
    """
    stage_id = usd_driver.call("load_usd_file", {"file_path": str(FIXTURE_SCENE)})[
        "stage_id"
    ]
    cube = {"stage_id": stage_id, "prim_path": "/World/Cube"}

    prim_info = usd_driver.call("get_prim_info", cube)
    _assert_ok(prim_info)
    assert prim_info["type"] == "Mesh"
    assert prim_info["transform"]["translation"] == [0.0, 0.0, 0.5]

    mesh_info = usd_driver.call("get_mesh_info", cube)
    _assert_ok(mesh_info)
    assert mesh_info["vertex_count"] == 8
    assert mesh_info["face_count"] == 6
    assert mesh_info["bbox"]["min"] == [-0.5, -0.5, -0.5]

    search = usd_driver.call(
        "search_prims",
        {"stage_id": stage_id, "search_type": "by_type", "query": "Mesh"},
    )
    _assert_ok(search)
    assert search["results"] == ["/World/Cube"]

    bbox = usd_driver.call("get_bounding_box", cube)
    _assert_ok(bbox)
    assert bbox["bbox"]["max"] == [0.5, 0.5, 1.0]

    summary = usd_driver.call("summarize_scene", {"stage_id": stage_id})
    _assert_ok(summary)
    assert summary["summary"]["total_prims"] == 3


def test_edit_tools_round_trip(usd_driver: UsdToolDriver) -> None:
    """create, update (with a JSON list for a vector attribute) and delete."""
    stage_id = usd_driver.call("load_usd_file", {"file_path": str(FIXTURE_SCENE)})[
        "stage_id"
    ]
    extra = {"stage_id": stage_id, "prim_path": "/World/Extra"}

    _assert_ok(usd_driver.call("create_prim", {**extra, "prim_type": "Xform"}))
    _assert_ok(
        usd_driver.call(
            "update_prim_attributes",
            {
                **extra,
                "attributes": {
                    "xformOp:translate": [1.0, 2.0, 3.0],
                    "custom:mass": 2.5,
                },
            },
        )
    )
    updated = usd_driver.call("get_prim_info", extra)
    _assert_ok(updated)
    assert updated["attributes"]["custom:mass"] == 2.5
    assert "xformOp:translate" in updated["attributes"]

    _assert_ok(usd_driver.call("delete_prim", extra))
    gone = usd_driver.call("get_prim_info", extra)
    assert gone["success"] is False


def test_stats_tool_is_single_transmission(usd_driver: UsdToolDriver) -> None:
    """The server-metadata tools convert too, or they keep the duplicate."""
    payload = usd_driver.call("get_tool_usage_stats", {"include_recent": True})

    assert "total_calls" in payload
    assert isinstance(payload["recent"], list)


def test_usd_tool_calls_are_recorded_with_the_agent(usd_driver: UsdToolDriver) -> None:
    """Every USD call goes through the envelope, so it lands in the audit log."""
    usd_driver.call("validate_usd_file", {"file_path": str(FIXTURE_SCENE)})

    recent = usd_driver.call("get_tool_usage_stats", {"include_recent": True})["recent"]
    validate = next(r for r in recent if r["tool"] == "validate_usd_file")
    assert validate["success"] is True
    assert validate["params"] == {"file_path": str(FIXTURE_SCENE)}
    assert validate["agent_id"]


def test_usd_only_server_registers_no_isaac_tools(usd_driver: UsdToolDriver) -> None:
    """``--backends usd`` must not leak the Isaac instance routing tools."""
    tool_names = usd_driver.list_tool_names()

    leaked = [name for name in tool_names if "isaac" in name]
    assert not leaked, f"Isaac tools registered for a USD-only server: {leaked}"
    assert "load_usd_file" in tool_names
    assert "get_tool_usage_stats" in tool_names


def test_isaac_server_keeps_instance_tools() -> None:
    """Selecting Isaac still registers the instance discovery and routing tools."""
    driver = UsdToolDriver(SimulMCPServer(Settings(), backends={"isaac"}))

    tool_names = set(driver.list_tool_names())

    assert {
        "list_isaac_instances",
        "set_active_isaac_instance",
        "claim_isaac_instance",
        "release_isaac_instance",
    } <= tool_names
    assert "load_usd_file" not in tool_names
