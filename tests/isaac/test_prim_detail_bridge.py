"""get_prim_detail: every aspect of a prim in one bridge round trip.

The per-aspect loop in ``get_isaac_prim_detail`` opened one connection per
aspect (14 aspects, 182 ms on loopback) while the tool description promised a
single call. The bridge action reads them all inside Kit and answers once; the
loop stays as the fallback for a bridge that is off or too old.
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
from typing import Any, Dict, List, Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

from khemoo.simul.mcp import prim_detail as bridge_prim_detail
from khemoo.simul.mcp.protocol import BridgeRequest, BridgeResponse
from khemoo.simul.mcp.service import READ_ONLY_ACTIONS, BridgeCommandService
from simul_mcp.adapters.isaac_socket_client import ScriptResult
from simul_mcp.config import Settings
from simul_mcp.mcp.tools.isaac_tools import PRIM_DETAIL_ASPECTS, IsaacTools

ALL_ASPECTS: List[str] = list(PRIM_DETAIL_ASPECTS)

# The prim each aspect applies to on the fixture stage.
ASPECT_PRIMS: Dict[str, str] = {
    "info": "/World/Cube",
    "transform": "/World/Cube",
    "ancestors": "/World/Cube",
    "relationships": "/World/Cube",
    "variants": "/World/Cube",
    "bounding_box": "/World/Cube",
    "mesh": "/World/Mesh",
    "light": "/World/Light",
    "material": "/World/Looks/Red",
    "rigid_body": "/World/Cube",
    "collision": "/World/Cube",
    "joint": "/World/Joint",
    "mass": "/World/Cube",
    "animation": "/World/Cube",
}


@pytest.fixture
def detail_stage() -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World")
    world.AddTranslateOp().Set(Gf.Vec3d(10, 0, 0))

    cube = UsdGeom.Cube.Define(stage, "/World/Cube")
    cube.AddTranslateOp().Set(Gf.Vec3d(1, 2, 3))
    cube.AddScaleOp().Set(Gf.Vec3f(2, 2, 2))
    cube.GetSizeAttr().Set(1.0, Usd.TimeCode(1))
    cube.GetSizeAttr().Set(2.0, Usd.TimeCode(10))
    prim = cube.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.MeshCollisionAPI.Apply(prim).GetApproximationAttr().Set("convexHull")
    UsdPhysics.MassAPI.Apply(prim).GetMassAttr().Set(2.5)
    look = prim.GetVariantSets().AddVariantSet("look")
    look.AddVariant("red")
    look.AddVariant("blue")
    look.SetVariantSelection("red")

    material = UsdShade.Material.Define(stage, "/World/Looks/Red")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/Red/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1, 0, 0))
    material.CreateSurfaceOutput().ConnectToSource(
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)

    UsdLux.SphereLight.Define(stage, "/World/Light").CreateIntensityAttr(500.0)

    mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray).Set(
        [(0, 0), (1, 0), (1, 1), (0, 1)]
    )

    joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/Joint")
    joint.CreateBody0Rel().SetTargets(["/World/Cube"])
    joint.CreateAxisAttr("Z")
    joint.CreateLowerLimitAttr(-90.0)
    joint.CreateUpperLimitAttr(90.0)
    return stage


@pytest.fixture
def bridge_service(detail_stage: Usd.Stage, monkeypatch: pytest.MonkeyPatch) -> BridgeCommandService:
    """A BridgeCommandService whose omni.usd hands back the fixture stage."""
    context = types.SimpleNamespace(
        get_stage=lambda: detail_stage, get_stage_url=lambda: "anon:memory.usda"
    )
    fake_usd = types.ModuleType("omni.usd")
    fake_usd.get_context = lambda: context  # type: ignore[attr-defined]
    fake_omni = types.ModuleType("omni")
    fake_omni.usd = fake_usd  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "omni", fake_omni)
    monkeypatch.setitem(sys.modules, "omni.usd", fake_usd)
    return BridgeCommandService(executor=MagicMock(), allow_unsafe_execution=False)


def _dispatch(service: BridgeCommandService, action: str, payload: Dict[str, Any]) -> BridgeResponse:
    return asyncio.run(service.dispatch(BridgeRequest(request_id="r1", action=action, payload=payload)))


# ---------------------------------------------------------------------------
# Bridge side
# ---------------------------------------------------------------------------


def test_action_is_advertised_and_read_only(bridge_service: BridgeCommandService) -> None:
    assert "get_prim_detail" in bridge_service.capabilities["actions"]
    assert "get_prim_detail" in READ_ONLY_ACTIONS


def test_bridge_and_mcp_side_agree_on_the_aspect_names() -> None:
    assert set(bridge_prim_detail.PRIM_DETAIL_ASPECTS) == set(PRIM_DETAIL_ASPECTS)


def test_every_aspect_arrives_in_one_response(bridge_service: BridgeCommandService) -> None:
    response = _dispatch(
        bridge_service, "get_prim_detail", {"prim_path": "/World/Cube", "aspects": ALL_ASPECTS}
    )

    assert response.status == "ok", response.error
    payload = response.payload
    assert payload["aspects"] == ALL_ASPECTS
    assert payload["info"]["type"] == "Cube"
    assert payload["transform"]["translation"] == [11.0, 2.0, 3.0]
    assert [a["path"] for a in payload["ancestors"]["ancestors"]] == ["/", "/World", "/World/Cube"]
    assert payload["relationships"]["material_binding"] == "/World/Looks/Red"
    assert payload["variants"]["variant_sets"]["look"]["selection"] == "red"
    assert payload["bounding_box"]["center"] == [11.0, 2.0, 3.0]
    assert payload["rigid_body"]["mass"] == 2.5
    assert payload["collision"]["approximation"] == "convexHull"
    assert payload["mass"]["mass"] == 2.5
    assert payload["animation"]["animated_attributes"][0]["name"] == "size"
    for aspect in ("mesh", "light", "material", "joint"):
        # Aspects that do not apply are reported in place, not fatal.
        assert payload[aspect]["success"] is False
        assert "error" in payload[aspect]
    for aspect in ALL_ASPECTS:
        assert payload[aspect]["success"] == ("error" not in payload[aspect])


def test_aspects_on_the_prims_they_apply_to(bridge_service: BridgeCommandService) -> None:
    mesh = _dispatch(bridge_service, "get_prim_detail", {"prim_path": "/World/Mesh", "aspects": ["mesh"]})
    light = _dispatch(bridge_service, "get_prim_detail", {"prim_path": "/World/Light", "aspects": ["light"]})
    material = _dispatch(
        bridge_service, "get_prim_detail", {"prim_path": "/World/Looks/Red", "aspects": ["material"]}
    )
    joint = _dispatch(bridge_service, "get_prim_detail", {"prim_path": "/World/Joint", "aspects": ["joint"]})

    assert mesh.payload["mesh"]["vertex_count"] == 4
    assert mesh.payload["mesh"]["face_count"] == 1
    assert mesh.payload["mesh"]["has_uvs"] is True
    assert light.payload["light"]["intensity"] == 500.0
    assert material.payload["material"]["shader_type"] == "UsdPreviewSurface"
    assert material.payload["material"]["inputs"]["diffuseColor"] == [1.0, 0.0, 0.0]
    assert joint.payload["joint"]["limits"] == {"type": "revolute", "axis": "Z", "lower": -90.0, "upper": 90.0}
    assert joint.payload["joint"]["body0"] == ["/World/Cube"]


def test_unknown_aspects_and_missing_prims_are_rejected(bridge_service: BridgeCommandService) -> None:
    unknown = _dispatch(bridge_service, "get_prim_detail", {"prim_path": "/World/Cube", "aspects": ["nonsense"]})
    missing = _dispatch(bridge_service, "get_prim_detail", {"prim_path": "/World/Nope", "aspects": ["info"]})

    assert unknown.status == "error" and unknown.error is not None
    assert unknown.error.name == "InvalidAspect"
    assert "transform" in unknown.error.message
    assert missing.status == "error" and missing.error is not None
    assert missing.error.name == "PrimNotFound"


def test_get_prim_info_and_transform_keep_their_shape(bridge_service: BridgeCommandService) -> None:
    """The older single-aspect actions now delegate to the same reader."""
    info = _dispatch(bridge_service, "get_prim_info", {"prim_path": "/World/Cube"})
    world = _dispatch(bridge_service, "get_prim_transform", {"prim_path": "/World/Cube"})
    local = _dispatch(bridge_service, "get_prim_transform", {"prim_path": "/World/Cube", "world_space": False})
    flat = _dispatch(bridge_service, "get_prim_transform", {"prim_path": "/World/Looks/Red"})

    assert info.payload["transport"] == "simul_bridge"
    assert info.payload["path"] == "/World/Cube"
    assert info.payload["material_bindings"] == ["/World/Looks/Red"]
    assert info.payload["attributes"]["xformOpOrder"] == ["xformOp:translate", "xformOp:scale"]
    assert world.payload["translation"] == [11.0, 2.0, 3.0]
    assert world.payload["scale"] == [2.0, 2.0, 2.0]
    assert local.payload["translation"] == [1.0, 2.0, 3.0]
    assert local.payload["space"] == "local"
    assert flat.status == "error" and flat.error is not None
    assert flat.error.name == "PrimNotXformable"


@pytest.mark.parametrize("aspect", ALL_ASPECTS)
def test_bridge_reader_matches_the_script_path(
    aspect: str,
    detail_stage: Usd.Stage,
    capturing_tools: Tuple[IsaacTools, List[str]],
    run_on_stage: Any,
) -> None:
    """Whichever transport answers, the caller must see the same payload."""
    tools, captured = capturing_tools
    prim_path = ASPECT_PRIMS[aspect]
    asyncio.run(getattr(tools, PRIM_DETAIL_ASPECTS[aspect])(prim_path))
    from_script = run_on_stage(captured[-1], detail_stage)

    reader = bridge_prim_detail.PrimDetailReader(detail_stage)
    from_bridge = json.loads(json.dumps(reader.read(detail_stage.GetPrimAtPath(prim_path), aspect)))

    assert from_bridge == from_script


# ---------------------------------------------------------------------------
# MCP side
# ---------------------------------------------------------------------------


def _bridged_tools(bridge_response: Dict[str, Any]) -> IsaacTools:
    client = MagicMock()
    client.address = "127.0.0.1:8226"
    client.timeout_seconds = 30.0
    client.bridge_enabled = True
    client.fallback_to_vscode = True
    client.bridge_request = AsyncMock(return_value=bridge_response)
    script_result = ScriptResult(success=True, output=json.dumps({"from": "script"}))
    client.execute = AsyncMock(return_value=script_result)
    client.execute_vscode_only = AsyncMock(return_value=script_result)
    client.execute_bridge_script_only = AsyncMock(return_value=script_result)
    return IsaacTools(client, settings=Settings())


def test_prim_detail_uses_one_bridge_request_for_many_aspects() -> None:
    aspects = ["info", "transform", "bounding_box", "mass"]
    payload = {"prim_path": "/World/Cube", "aspects": aspects, "transport": "simul_bridge"}
    payload.update({aspect: {"success": True, "aspect": aspect} for aspect in aspects})
    tools = _bridged_tools({"status": "ok", "payload": payload})

    result = asyncio.run(tools.get_isaac_prim_detail("/World/Cube", aspects=aspects))

    assert result["success"] is True
    assert result["aspects"] == aspects
    assert result["mass"] == {"success": True, "aspect": "mass"}
    tools._client.bridge_request.assert_awaited_once_with(
        "get_prim_detail", {"prim_path": "/World/Cube", "aspects": aspects}
    )
    tools._client.execute.assert_not_called()
    tools._client.execute_vscode_only.assert_not_called()


def test_prim_detail_falls_back_to_the_per_aspect_loop_without_the_action() -> None:
    """A bridge that predates the action answers UnknownAction; the loop still works."""
    aspects = ["info", "transform", "bounding_box"]
    tools = _bridged_tools(
        {"status": "error", "error": {"name": "UnknownAction", "message": "no such action"}}
    )

    result = asyncio.run(tools.get_isaac_prim_detail("/World/Cube", aspects=aspects))

    assert result["aspects"] == aspects
    assert result["bounding_box"]["from"] == "script"
    # info and transform have their own typed actions, which also answer
    # UnknownAction here, so every aspect ends on a script transport: one
    # script per aspect, which is the cost the bridge action removes.
    script_calls = tools._client.execute.await_count + tools._client.execute_vscode_only.await_count
    assert script_calls == len(aspects)
    assert tools._client.bridge_request.await_args_list[0].args == (
        "get_prim_detail",
        {"prim_path": "/World/Cube", "aspects": aspects},
    )


def test_unknown_aspects_are_rejected_before_any_transport() -> None:
    tools = _bridged_tools({"status": "ok", "payload": {}})

    result = asyncio.run(tools.get_isaac_prim_detail("/World/Cube", aspects=["nonsense"]))

    assert result["error_type"] == "ValueError"
    tools._client.bridge_request.assert_not_called()
