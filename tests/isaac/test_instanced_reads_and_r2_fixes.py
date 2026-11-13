"""Round-2 Isaac fixes, run on real ``pxr`` through ``stage_tools``.

- find_isaac_prims_in_area, the scene stats and summary, the stage info count
  and the light, camera, material and texture listings walked with the
  default predicate, which stops at instanceable prims: a scene built from
  instanced assets showed no geometry to the area search and no meshes to the
  stats, while search_isaac_prims found them.
- get_isaac_prim_detail's script path answered a missing prim with
  ``success: true`` and the error inside every aspect.
- set_isaac_prim_attribute wrote asset paths without the file sandbox.
- reparent_isaac_prim's pxr fallback left internal references, inherits and
  specializes on the old path.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

from simul.config import Settings
from tests.isaac.conftest import StageTools


def _prop_asset(directory: Path) -> str:
    """An asset with a mesh, a light, a camera and a textured material, as props carry."""
    path = str(directory / "prop.usda")
    stage = Usd.Stage.CreateNew(path)
    root = UsdGeom.Xform.Define(stage, "/Prop").GetPrim()
    stage.SetDefaultPrim(root)
    mesh = UsdGeom.Mesh.Define(stage, "/Prop/visual")
    mesh.CreatePointsAttr([(0, 0, 0), (0.2, 0, 0), (0, 0.2, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    mesh.CreateExtentAttr([(0, 0, 0), (0.2, 0.2, 0)])
    UsdLux.SphereLight.Define(stage, "/Prop/lamp")
    UsdGeom.Camera.Define(stage, "/Prop/sensor")
    material = UsdShade.Material.Define(stage, "/Prop/Looks/Paint")
    shader = UsdShade.Shader.Define(stage, "/Prop/Looks/Paint/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./paint.png")
    )
    material.CreateSurfaceOutput().ConnectToSource(
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )
    stage.GetRootLayer().Save()
    return path


@pytest.fixture
def instanced_stage(tmp_path: Path) -> Usd.Stage:
    asset = _prop_asset(tmp_path)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    for i in range(3):
        prim = UsdGeom.Xform.Define(stage, "/World/prop_%d" % i).GetPrim()
        prim.GetReferences().AddReference(asset)
        prim.SetInstanceable(True)
        UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(i * 0.5, 0, 0))
    UsdGeom.Cube.Define(stage, "/World/plain_cube").CreateSizeAttr(0.1)
    return stage


def _run(coro: Any) -> Dict[str, Any]:
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# M5: instanced geometry is seen
# ---------------------------------------------------------------------------


def test_area_search_finds_instanced_geometry(
    stage_tools: StageTools, instanced_stage: Usd.Stage
) -> None:
    """It used to return only the plain cube."""
    tools, _ = stage_tools(instanced_stage)

    result = _run(tools.find_isaac_prims_in_area(center=[0.5, 0, 0], radius=2.0))
    without = _run(
        tools.find_isaac_prims_in_area(
            center=[0.5, 0, 0], radius=2.0, include_instance_proxies=False
        )
    )

    paths = {m["path"] for m in result["matches"]}
    assert {
        "/World/plain_cube",
        "/World/prop_0/visual",
        "/World/prop_1/visual",
        "/World/prop_2/visual",
    } <= paths
    assert all(
        m.get("instance_proxy") for m in result["matches"] if "prop_" in m["path"]
    )
    assert [m["path"] for m in without["matches"]] == ["/World/plain_cube"]


def test_counts_agree_across_stats_summary_stage_info_and_listing(
    stage_tools: StageTools, instanced_stage: Usd.Stage
) -> None:
    """Stats said total_meshes 0 and total_prims 5 against 9 from list_isaac_prims."""
    tools, _ = stage_tools(instanced_stage)

    stats = _run(tools.get_isaac_scene_stats())
    summary = _run(tools.get_isaac_scene_summary())
    info = _run(tools.get_isaac_stage_info(include_prim_count=True))
    listing = _run(tools.list_isaac_prims(max_depth=-1, max_results=1000))

    assert stats["total_meshes"] == 3
    assert stats["total_vertices"] == 9
    assert stats["total_lights"] == 3
    assert (
        stats["total_prims"]
        == summary["total_prims"]
        == info["total_prims"]
        == listing["count"]
    )
    assert all(
        entry["path"] != "/" for entry in listing["prims"]
    ), "the pseudo-root is not a scene prim"
    flat = _run(tools.get_isaac_scene_stats(include_instance_proxies=False))
    assert flat["total_meshes"] == 0


@pytest.mark.parametrize(
    ("method", "entries", "expected"),
    [
        (
            "list_isaac_lights",
            "lights",
            ["/World/prop_0/lamp", "/World/prop_1/lamp", "/World/prop_2/lamp"],
        ),
        (
            "list_isaac_cameras",
            "cameras",
            ["/World/prop_0/sensor", "/World/prop_1/sensor", "/World/prop_2/sensor"],
        ),
        (
            "list_isaac_materials",
            "materials",
            [
                "/World/prop_0/Looks/Paint",
                "/World/prop_1/Looks/Paint",
                "/World/prop_2/Looks/Paint",
            ],
        ),
    ],
)
def test_listings_see_inside_instances(
    method: str,
    entries: str,
    expected: List[str],
    stage_tools: StageTools,
    instanced_stage: Usd.Stage,
) -> None:
    tools, _ = stage_tools(instanced_stage)

    result = _run(getattr(tools, method)())
    without = _run(getattr(tools, method)(include_instance_proxies=False))

    assert sorted(entry["path"] for entry in result[entries]) == expected
    assert all(entry["instance_proxy"] is True for entry in result[entries])
    assert without[entries] == []


def test_texture_dependencies_include_instanced_materials(
    stage_tools: StageTools, instanced_stage: Usd.Stage
) -> None:
    tools, _ = stage_tools(instanced_stage)

    result = _run(tools.get_isaac_texture_dependencies())

    assert result["unique_textures"] == 1
    assert result["textures"][0]["path"].endswith("paint.png")


# ---------------------------------------------------------------------------
# get_isaac_prim_detail: a missing prim fails the call, as on the bridge
# ---------------------------------------------------------------------------


def test_prim_detail_for_a_missing_prim_is_not_found(stage_tools: StageTools) -> None:
    stage = Usd.Stage.CreateInMemory()
    tools, sent = stage_tools(stage)

    result = _run(
        tools.get_isaac_prim_detail("/World/Nope", ["info", "transform", "mesh"])
    )

    assert result["success"] is False
    assert result["error"] == "Prim not found: /World/Nope"
    # What the bridge's PrimNotFound becomes in the shared vocabulary.
    assert result["error_type"] == "NotFoundError"
    assert (
        len(sent) == 1
    ), "the other aspects are not read once the prim is known missing"


# ---------------------------------------------------------------------------
# set_isaac_prim_attribute: asset values go through the file sandbox
# ---------------------------------------------------------------------------


def _sandboxed(stage_tools: StageTools, stage: Usd.Stage, sandbox: Path) -> Any:
    settings = Settings()
    security = settings.security.model_copy(update={"allowed_paths": [str(sandbox)]})
    return stage_tools(
        stage, settings=settings.model_copy(update={"security": security})
    )


def _asset_stage() -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    prim = UsdGeom.Xform.Define(stage, "/World/Dome").GetPrim()
    prim.CreateAttribute("inputs:texture:file", Sdf.ValueTypeNames.Asset)
    prim.CreateAttribute("textures", Sdf.ValueTypeNames.AssetArray)
    prim.CreateAttribute("label", Sdf.ValueTypeNames.String)
    return stage


def test_an_asset_path_outside_the_sandbox_is_refused(
    stage_tools: StageTools, tmp_path: Path
) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    stage = _asset_stage()
    tools, _ = _sandboxed(stage_tools, stage, sandbox)

    single = _run(
        tools.set_isaac_prim_attribute(
            "/World/Dome", "inputs:texture:file", "/etc/shadow"
        )
    )
    listed = _run(
        tools.set_isaac_prim_attribute(
            "/World/Dome", "textures", [str(sandbox / "a.png"), "/etc/shadow"]
        )
    )

    assert single["error_type"] == "SandboxError"
    assert listed["error_type"] == "SandboxError"
    prim = stage.GetPrimAtPath("/World/Dome")
    assert prim.GetAttribute("inputs:texture:file").Get() is None
    assert not prim.GetAttribute("textures").Get()


def test_an_asset_path_inside_the_sandbox_is_written_resolved(
    stage_tools: StageTools, tmp_path: Path
) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    stage = _asset_stage()
    tools, _ = _sandboxed(stage_tools, stage, sandbox)
    inside = sandbox / "sub" / ".." / "hdri.exr"

    result = _run(
        tools.set_isaac_prim_attribute(
            "/World/Dome", "inputs:texture:file", str(inside)
        )
    )

    assert result["success"] is True
    written = (
        stage.GetPrimAtPath("/World/Dome")
        .GetAttribute("inputs:texture:file")
        .Get()
        .path
    )
    assert written == str((sandbox / "hdri.exr").resolve())


def test_a_string_attribute_is_not_a_path(
    stage_tools: StageTools, tmp_path: Path
) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    stage = _asset_stage()
    tools, _ = _sandboxed(stage_tools, stage, sandbox)

    result = _run(tools.set_isaac_prim_attribute("/World/Dome", "label", "/etc/shadow"))

    assert result["success"] is True
    assert (
        stage.GetPrimAtPath("/World/Dome").GetAttribute("label").Get() == "/etc/shadow"
    )


# ---------------------------------------------------------------------------
# reparent_isaac_prim's pxr fallback retargets arcs
# ---------------------------------------------------------------------------


def test_reparent_fallback_retargets_internal_arcs(stage_tools: StageTools) -> None:
    """Kit's MovePrim rewrites these; the bare namespace edit left them stale."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/Classes/Base")
    UsdGeom.Xform.Define(stage, "/World/Lib")
    user = stage.DefinePrim("/World/User", "Xform")
    user.GetInherits().AddInherit("/World/Classes/Base")
    user.GetSpecializes().AddSpecialize("/World/Classes/Base")
    user.GetReferences().AddInternalReference("/World/Classes/Base")
    tools, _ = stage_tools(stage)

    result = _run(tools.reparent_isaac_prim("/World/Classes/Base", "/World/Lib"))

    assert result["method"] == "pxr Sdf namespace edit on the edit target"
    spec = stage.GetRootLayer().GetPrimAtPath("/World/User")
    new = Sdf.Path("/World/Lib/Base")
    assert list(spec.inheritPathList.GetAddedOrExplicitItems()) == [new]
    assert list(spec.specializesList.GetAddedOrExplicitItems()) == [new]
    assert [ref.primPath for ref in spec.referenceList.GetAddedOrExplicitItems()] == [
        new
    ]
    assert {
        "/World/User inherits",
        "/World/User specializes",
        "/World/User references",
    } <= set(result["retargeted"])
