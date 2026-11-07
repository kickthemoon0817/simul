"""The USD edit tools, driven through the real MCP stack, refuse precisely and change nothing on failure.

The behaviour itself is pinned against pxr in
``tests/headless_usd/test_edit_safety.py``; these tests check that the MCP
wrappers carry it to the wire: the new arguments, the structured error
envelope (``error_type`` and ``details``), and that a failed call leaves the
stage as it was. Stage state is read through pxr, not ``get_prim_info``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict

import pytest
from fastmcp import Client

pytest.importorskip("pxr", reason="pxr library not available")

from pxr import Sdf, Usd, UsdGeom  # noqa: E402

from simul.config import Settings  # noqa: E402
from simul.mcp.server import SimulMCPServer  # noqa: E402


class _Driver:
    def __init__(self, server: SimulMCPServer) -> None:
        self.server = server

    def call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        async def _run() -> Dict[str, Any]:
            async with Client(self.server.mcp) as client:
                result = await client.call_tool(name, arguments, raise_on_error=False)
                return json.loads(result.content[0].text)

        return asyncio.run(_run())

    def stage(self, stage_id: str) -> Usd.Stage:
        with self.server.headless_adapter.create_session() as session:
            return session.get_stage(stage_id)


@pytest.fixture
def scene_file(tmp_path: Path) -> Path:
    sub = Usd.Stage.CreateNew(str(tmp_path / "sublayer.usda"))
    UsdGeom.Xform.Define(sub, "/World")
    UsdGeom.Xform.Define(sub, "/World/FromSublayer")
    sub.GetRootLayer().Save()
    stage = Usd.Stage.CreateNew(str(tmp_path / "scene.usda"))
    stage.GetRootLayer().subLayerPaths.append("./sublayer.usda")
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Cube.Define(stage, "/World/Box").GetSizeAttr().Set(2.0)
    stage.OverridePrim("/World/FromSublayer").CreateAttribute(
        "custom:note", Sdf.ValueTypeNames.String
    ).Set("override")
    stage.GetRootLayer().Save()
    return tmp_path / "scene.usda"


@pytest.fixture
def driver(tmp_path: Path) -> _Driver:
    base = Settings()
    settings = base.model_copy(
        update={"security": base.security.model_copy(update={"allowed_paths": [str(tmp_path)]})}
    )
    return _Driver(SimulMCPServer(settings, backends={"usd"}))


@pytest.fixture
def loaded(driver: _Driver, scene_file: Path) -> Dict[str, Any]:
    stage_id = driver.call("load_usd_file", {"file_path": str(scene_file)})["stage_id"]
    stage = driver.stage(stage_id)
    return {"id": stage_id, "stage": stage, "before": stage.GetRootLayer().ExportToString()}


def _unchanged(loaded: Dict[str, Any]) -> bool:
    return loaded["stage"].GetRootLayer().ExportToString() == loaded["before"]


def test_failed_create_creates_nothing(driver: _Driver, loaded: Dict[str, Any]) -> None:
    payload = driver.call(
        "create_prim",
        {
            "stage_id": loaded["id"],
            "prim_path": "/World/Partial",
            "prim_type": "Cube",
            "attributes": {"size": 2.0, "bad": {"nested": 1}},
        },
    )

    assert payload["success"] is False
    assert (payload["error_type"], payload["details"]["code"]) == ("ValidationError", "InvalidAttribute")
    assert payload["details"]["problems"][0]["attribute"] == "bad"
    assert _unchanged(loaded)


def test_failed_update_changes_nothing(driver: _Driver, loaded: Dict[str, Any]) -> None:
    payload = driver.call(
        "update_prim_attributes",
        {
            "stage_id": loaded["id"],
            "prim_path": "/World/Box",
            "attributes": {"size": 7.0, "custom:a": 1, "zz": {"x": 1}},
        },
    )

    assert (payload["error_type"], payload["details"]["code"]) == ("ValidationError", "InvalidAttribute")
    assert "'zz'" in payload["error"]
    assert _unchanged(loaded)


def test_successful_update_reports_what_changed(driver: _Driver, loaded: Dict[str, Any]) -> None:
    payload = driver.call(
        "update_prim_attributes",
        {
            "stage_id": loaded["id"],
            "prim_path": "/World/Box",
            "attributes": {"size": 4.0, "kind": "component"},
        },
    )

    assert payload["success"] is True
    assert payload["changed"] == ["size", "kind (metadata)"]
    box = loaded["stage"].GetPrimAtPath("/World/Box")
    assert box.GetAttribute("size").Get() == 4.0
    assert Usd.ModelAPI(box).GetKind() == "component"


@pytest.mark.parametrize(
    "arguments, error_type",
    [
        ({"prim_path": "World"}, "ValidationError"),
        ({"prim_path": "/World/DoesNotExist"}, "NotFoundError"),
        ({"prim_path": "/"}, "RefusedOperation"),
        ({"prim_path": "/World"}, "RefusedOperation"),
        ({"prim_path": "/World/FromSublayer"}, "UnsupportedOperation"),
    ],
)
def test_delete_refusals(
    driver: _Driver, loaded: Dict[str, Any], arguments: Dict[str, Any], error_type: str
) -> None:
    payload = driver.call("delete_prim", {"stage_id": loaded["id"], **arguments})

    assert payload["success"] is False
    assert payload["error_type"] == error_type
    assert _unchanged(loaded)


def test_create_on_an_existing_prim_needs_the_flag(driver: _Driver, loaded: Dict[str, Any]) -> None:
    arguments = {"stage_id": loaded["id"], "prim_path": "/World/Box", "prim_type": "Sphere"}

    refused = driver.call("create_prim", arguments)

    assert refused["error_type"] == "RefusedOperation"
    assert refused["details"]["override"] == "redefine_existing"
    assert _unchanged(loaded)

    redefined = driver.call("create_prim", {**arguments, "redefine_existing": True})

    assert redefined["success"] is True
    assert loaded["stage"].GetPrimAtPath("/World/Box").GetTypeName() == "Sphere"


def test_unknown_prim_type(driver: _Driver, loaded: Dict[str, Any]) -> None:
    payload = driver.call(
        "create_prim", {"stage_id": loaded["id"], "prim_path": "/World/Typo", "prim_type": "Cub"}
    )

    assert (payload["error_type"], payload["details"]["code"]) == ("ValidationError", "UnknownPrimType")
    assert payload["details"]["suggestions"][0] == "Cube"
    assert _unchanged(loaded)


def test_refusals_are_recorded_as_failed_calls(driver: _Driver, loaded: Dict[str, Any]) -> None:
    driver.call("delete_prim", {"stage_id": loaded["id"], "prim_path": "World"})

    recent = driver.call("get_tool_usage_stats", {"include_recent": True})["recent"]
    record = next(r for r in recent if r["tool"] == "delete_prim")
    assert record["success"] is False
    assert record["params"]["allow_root_delete"] is False


@pytest.mark.parametrize(
    "stage_id, prim_path, error_type, prefix",
    [
        (None, "World/Box", "ValidationError", "Invalid prim path 'World/Box': "),
        (None, "/World/Nope", "NotFoundError", "Prim not found: /World/Nope"),
        ("not_loaded_123", "/World/Box", "NotFoundError", "Stage not loaded: 'not_loaded_123'."),
    ],
)
def test_read_and_edit_tools_report_lookup_mistakes_alike(
    driver: _Driver,
    loaded: Dict[str, Any],
    stage_id: Any,
    prim_path: str,
    error_type: str,
    prefix: str,
) -> None:
    """The same bad stage id or prim path gets the same error type and wording from both kinds of tool."""
    target = {"stage_id": stage_id or loaded["id"], "prim_path": prim_path}

    read = driver.call("get_bounding_box", target)
    edit = driver.call("update_prim_attributes", {**target, "attributes": {"custom:a": 1}})

    assert read["error_type"] == edit["error_type"] == error_type
    assert read["error"].startswith(prefix)
    assert edit["error"].startswith(prefix)
