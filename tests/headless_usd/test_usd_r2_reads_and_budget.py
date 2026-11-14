"""Round-2 headless USD audit: reads, the result budget and error typing.

Each test names the audit finding it pins (M1, M2, M4, M9, m1, m4, m6, m7, m8).
The tool-level tests go through the real FastMCP stack (``usd_tools``).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

pytest.importorskip("pxr", reason="pxr library not available")

from fastmcp import Client  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom  # noqa: E402

from simul.adapters.headless_usd import HeadlessUSDSession  # noqa: E402
from simul.config import Settings  # noqa: E402
from simul.mcp.paged_results import PAGED_RESULTS, BULK_FIELDS  # noqa: E402
from simul.mcp.result_budget import DEFAULT_RESULT_BUDGET_BYTES, apply_result_budget  # noqa: E402
from simul.mcp.server import SimulMCPServer  # noqa: E402
from simul.usd.mesh_ops import MeshArrays, geometry_statistics  # noqa: E402
from simul.usd import reader  # noqa: E402
from simul.usd.reader import animation_range  # noqa: E402
from simul.usd.stage_files import public_identifier  # noqa: E402

import numpy as np  # noqa: E402

CUBE_POINTS = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
CUBE_FACES = [0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7]
FLIPPED_FACES = CUBE_FACES[:4] + CUBE_FACES[4:8][::-1] + CUBE_FACES[8:]


def _size(payload: Any) -> int:
    return len(json.dumps(payload, default=str))


# ---------------------------------------------------------------------------
# M1: the budget cuts get_prim_info's attributes, not its API schemas
# ---------------------------------------------------------------------------


def _prim_info_payload(schemas: int = 20, attributes: int = 778) -> Dict[str, Any]:
    return {
        "success": True,
        "path": "/Robot/Base",
        "name": "Base",
        "type": "Xform",
        "api_schemas": [f"PhysicsSomeAPI:{index:02d}" for index in range(schemas)],
        "children": [],
        "attribute_count": attributes,
        "attributes": {
            f"physics:attr_{index:04d}": {"type": "float", "value": float(index), "authored": True,
                                          "doc": "x" * 40}
            for index in range(attributes)
        },
        "metadata": {"specifier": "def"},
    }


def test_m1_get_prim_info_bulk_is_its_attributes() -> None:
    assert BULK_FIELDS["get_prim_info"].field == "attributes"


def test_m1_budget_cuts_attributes_and_keeps_api_schemas() -> None:
    payload = _prim_info_payload()
    assert _size(payload) > 60_000

    result = apply_result_budget(payload, tool_name="get_prim_info")

    assert _size(result) <= DEFAULT_RESULT_BUDGET_BYTES
    assert result["api_schemas"] == payload["api_schemas"], "small identity list was emptied"
    assert result["path"] == "/Robot/Base" and result["attribute_count"] == 778
    kept = list(result["attributes"])
    assert 0 < len(kept) < 778
    assert kept == list(payload["attributes"])[: len(kept)]
    record = result["truncated_fields"][0]
    assert record["field"] == "attributes"
    assert record["returned"] == len(kept) and record["total"] == 778
    assert record["first_omitted"] == list(payload["attributes"])[len(kept)]
    assert "attribute_filter" in record["hint"]
    assert result["truncated"] is True
    assert "oversized_bytes" not in result


def test_m1_a_payload_that_fits_is_untouched() -> None:
    payload = _prim_info_payload(attributes=10)

    assert apply_result_budget(payload, tool_name="get_prim_info") == payload


def test_m1_a_larger_list_is_still_cut_first() -> None:
    """The declared mapping goes first only when it is the bulk."""
    payload = _prim_info_payload(attributes=40)
    payload["children"] = [f"/Robot/Base/child_{index:05d}_{'y' * 40}" for index in range(2000)]

    result = apply_result_budget(payload, tool_name="get_prim_info")

    assert len(result["attributes"]) == 40
    assert result["truncation"]["field"] == "children"


def test_m1_get_prim_info_through_the_server(usd_tools, tmp_path: Path) -> None:
    stage = Usd.Stage.CreateNew(str(tmp_path / "wide.usda"))
    prim = UsdGeom.Xform.Define(stage, "/Robot").GetPrim()
    schemas = [f"CollectionAPI:set{index:02d}" for index in range(20)]
    for schema in schemas:
        prim.AddAppliedSchema(schema)
    for index in range(778):
        prim.CreateAttribute(f"custom:attr_{index:04d}", Sdf.ValueTypeNames.String).Set("v" * 60)
    stage.GetRootLayer().Save()
    stage_id = usd_tools.load(tmp_path / "wide.usda")

    text = usd_tools.text("get_prim_info", {"stage_id": stage_id, "prim_path": "/Robot"})
    payload = json.loads(text)

    assert len(text) <= DEFAULT_RESULT_BUDGET_BYTES
    assert set(schemas) <= set(payload["api_schemas"])
    assert payload["truncated_fields"][0]["field"] == "attributes"
    assert payload["truncated_fields"][0]["total"] == len(prim.GetAttributes())


# ---------------------------------------------------------------------------
# M2: time_sample_range in stage time, through offsets, scale and clips
# ---------------------------------------------------------------------------


def _animated(path: Path, prim: str = "/A", times: List[float] = (0.0, 10.0)) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    op = UsdGeom.Xform.Define(stage, prim).AddTranslateOp()
    for time in times:
        op.Set(Gf.Vec3d(time, 0, 0), time)
    stage.SetDefaultPrim(stage.GetPrimAtPath(prim))
    stage.GetRootLayer().Save()
    return path


def _pxr_range(stage: Usd.Stage) -> List[float]:
    times = [t for prim in stage.Traverse() for attr in prim.GetAttributes() for t in attr.GetTimeSamples()]
    return [min(times), max(times)]


def test_m2_reference_offset_and_scale(tmp_path: Path) -> None:
    _animated(tmp_path / "asset.usda")
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/Shifted").GetReferences().AddReference(
        Sdf.Reference(str(tmp_path / "asset.usda"), layerOffset=Sdf.LayerOffset(100, 2))
    )

    assert animation_range(stage)["time_sample_range"] == [100.0, 120.0] == _pxr_range(stage)


def test_m2_sublayer_nested_and_internal_reference_offsets(tmp_path: Path) -> None:
    _animated(tmp_path / "asset.usda")
    _animated(tmp_path / "sub.usda", "/S", [1.0, 2.0])
    mid = Usd.Stage.CreateNew(str(tmp_path / "mid.usda"))
    mid.DefinePrim("/M").GetReferences().AddReference(
        Sdf.Reference("./asset.usda", layerOffset=Sdf.LayerOffset(5, 1))
    )
    mid.SetDefaultPrim(mid.GetPrimAtPath("/M"))
    mid.GetRootLayer().Save()
    root = Usd.Stage.CreateNew(str(tmp_path / "root.usda"))
    root.GetRootLayer().subLayerPaths.append("./sub.usda")
    root.GetRootLayer().subLayerOffsets[0] = Sdf.LayerOffset(1000, 1)
    root.DefinePrim("/R").GetReferences().AddReference(
        Sdf.Reference("./mid.usda", layerOffset=Sdf.LayerOffset(100, 2))
    )
    proto = UsdGeom.Xform.Define(root, "/Proto").AddTranslateOp()
    for time in (3.0, 4.0):
        proto.Set(Gf.Vec3d(time, 0, 0), time)
    root.DefinePrim("/Internal").GetReferences().AddInternalReference("/Proto", Sdf.LayerOffset(-50, 1))
    root.GetRootLayer().Save()
    stage = Usd.Stage.Open(str(tmp_path / "root.usda"))

    # Internal reference: only /Proto's samples shift, to -47..-46; the sublayer stays at 1001..1002.
    assert animation_range(stage)["time_sample_range"] == [-47.0, 1002.0] == _pxr_range(stage)


def _clipped(tmp_path: Path) -> Path:
    for index in range(2):
        clip = Sdf.Layer.CreateNew(str(tmp_path / f"clip_{index}.usda"))
        spec = Sdf.CreatePrimInLayer(clip, "/Model")
        spec.specifier = Sdf.SpecifierDef
        attribute = Sdf.AttributeSpec(spec, "xformOp:translate", Sdf.ValueTypeNames.Double3)
        for time in range(11):
            clip.SetTimeSample(attribute.path, time, Gf.Vec3d(index * 100 + time, 0, 0))
        clip.Save()
    stage = Usd.Stage.CreateNew(str(tmp_path / "clips.usda"))
    model = UsdGeom.Xform.Define(stage, "/Model").GetPrim()
    stage.SetDefaultPrim(model)
    model.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3)
    clips = Usd.ClipsAPI(model)
    clips.SetClipAssetPaths([Sdf.AssetPath("./clip_0.usda"), Sdf.AssetPath("./clip_1.usda")])
    clips.SetClipPrimPath("/Model")
    clips.SetClipActive([(100.0, 0), (110.0, 1)])
    clips.SetClipTimes([(100.0, 0.0), (110.0, 10.0), (110.0, 0.0), (120.0, 10.0)])
    stage.GetRootLayer().Save()
    return tmp_path / "clips.usda"


def test_m2_value_clips_report_stage_time(tmp_path: Path) -> None:
    stage = Usd.Stage.Open(str(_clipped(tmp_path)))

    assert animation_range(stage)["time_sample_range"] == [100.0, 120.0] == _pxr_range(stage)


def test_m2_a_referenced_clipped_asset_is_offset_too(tmp_path: Path) -> None:
    _clipped(tmp_path)
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/Late").GetReferences().AddReference(
        Sdf.Reference(str(tmp_path / "clips.usda"), layerOffset=Sdf.LayerOffset(1000, 1))
    )

    assert animation_range(stage)["time_sample_range"] == [1100.0, 1120.0] == _pxr_range(stage)


CLIPS_TEXT = """#usda 1.0
def "Model" (
    clips = {
        dictionary default = {
            asset[] assetPaths = [@./clip_0.usda@, @./clip_1.usda@]
            string primPath = "/Model"
            double2[] active = [(100, 0), (110, 1)]
            double2[] times = [(100, 0), (110, 10), (110, 0), (120, 10)]
            asset manifestAssetPath = @./manifest.usda@
        }
    }
)
{
    double v
}
"""


def test_m2_clips_with_a_manifest_are_found_before_any_read(tmp_path: Path) -> None:
    """With a manifest pxr opens clip layers only on a first read; the range must not wait for one (D3)."""
    for index in range(2):
        (tmp_path / f"clip_{index}.usda").write_text(
            '#usda 1.0\nover "Model"\n{\n    double v.timeSamples = { 0: %d, 10: %d }\n}\n' % (index, index + 1)
        )
    (tmp_path / "manifest.usda").write_text('#usda 1.0\nover "Model"\n{\n    double v\n}\n')
    (tmp_path / "clips.usda").write_text(CLIPS_TEXT)
    stage = Usd.Stage.Open(str(tmp_path / "clips.usda"))

    result = animation_range(stage)

    assert result["time_sample_range"] == [100.0, 120.0] == _pxr_range(stage)
    assert result["has_animation"] is True


def _clipped_world(tmp_path: Path, prims: int = 300) -> Path:
    """A clip set on /World above ``prims`` prims, animating one attribute of one of them (no manifest)."""
    for index in range(2):
        clip = Sdf.Layer.CreateNew(str(tmp_path / f"world_clip_{index}.usda"))
        spec = Sdf.CreatePrimInLayer(clip, "/World/p0000")
        attribute = Sdf.AttributeSpec(spec, "xformOp:translate", Sdf.ValueTypeNames.Double3)
        for time in range(11):
            clip.SetTimeSample(attribute.path, time, Gf.Vec3d(index * 100 + time, 0, 0))
        clip.Save()
    stage = Usd.Stage.CreateNew(str(tmp_path / "world.usda"))
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    for index in range(prims):
        prim = UsdGeom.Xform.Define(stage, f"/World/p{index:04d}")
        op = prim.AddTranslateOp()
        if index:
            op.Set(Gf.Vec3d(index, 0, 0))  # a default would hide p0000's clip values
        prim.GetPrim().CreateAttribute("custom:weight", Sdf.ValueTypeNames.Float).Set(1.0)
    clips = Usd.ClipsAPI(world)
    clips.SetClipAssetPaths([Sdf.AssetPath("./world_clip_0.usda"), Sdf.AssetPath("./world_clip_1.usda")])
    clips.SetClipPrimPath("/World")
    clips.SetClipActive([(100.0, 0), (110.0, 1)])
    clips.SetClipTimes([(100.0, 0.0), (110.0, 10.0), (110.0, 0.0), (120.0, 10.0)])
    stage.GetRootLayer().Save()
    return tmp_path / "world.usda"


def test_r3_m4_clips_on_a_large_prim_ask_only_the_attributes_they_animate(tmp_path: Path) -> None:
    """USD round-3 m4: every attribute under the clipped prim used to be asked for its samples."""
    stage = Usd.Stage.Open(str(_clipped_world(tmp_path)))

    asked = reader._clip_attributes(stage.GetPrimAtPath("/World"))

    assert [attribute.GetPath().pathString for attribute in asked] == ["/World/p0000.xformOp:translate"]
    assert animation_range(stage)["time_sample_range"] == [100.0, 120.0] == _pxr_range(stage)


def test_r3_m4_an_authored_manifest_names_the_attributes(tmp_path: Path) -> None:
    for index in range(2):
        (tmp_path / f"clip_{index}.usda").write_text(
            '#usda 1.0\nover "Model"\n{\n    double v.timeSamples = { 0: %d, 10: %d }\n}\n' % (index, index + 1)
        )
    (tmp_path / "manifest.usda").write_text('#usda 1.0\nover "Model"\n{\n    double v\n}\n')
    (tmp_path / "clips.usda").write_text(CLIPS_TEXT)
    stage = Usd.Stage.Open(str(tmp_path / "clips.usda"))

    asked = reader._clip_attributes(stage.GetPrimAtPath("/Model"))

    assert [attribute.GetPath().pathString for attribute in asked] == ["/Model.v"]


def test_r3_m4_without_a_manifest_every_attribute_is_asked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stage = Usd.Stage.Open(str(_clipped_world(tmp_path, prims=20)))

    def broken(*_args: Any) -> Any:
        raise RuntimeError("no manifest")

    monkeypatch.setattr(Usd.ClipsAPI, "GenerateClipManifest", broken)

    assert reader._clip_attributes(stage.GetPrimAtPath("/World")) is None
    assert animation_range(stage)["time_sample_range"] == [100.0, 120.0]


def test_m2_falls_back_to_each_layers_own_times_without_the_pcp_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The mapping reads a private pxr binding; if it breaks, the load still answers (D5)."""
    _animated(tmp_path / "asset.usda")
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/Shifted").GetReferences().AddReference(
        Sdf.Reference(str(tmp_path / "asset.usda"), layerOffset=Sdf.LayerOffset(100, 2))
    )

    def broken(_stage: Any) -> Any:
        raise RuntimeError("no Pcp cache binding")

    monkeypatch.setattr(Usd.Stage, "_GetPcpCache", broken)
    with caplog.at_level("WARNING"):
        result = animation_range(stage)

    assert result["time_sample_range"] == [0.0, 10.0]
    assert "no Pcp cache binding" in caplog.text


def test_m2_summary_and_load_use_stage_time(usd_tools, tmp_path: Path) -> None:
    _animated(tmp_path / "asset.usda")
    scene = Usd.Stage.CreateNew(str(tmp_path / "scene.usda"))
    scene.GetRootLayer().timeCodesPerSecond = 24
    scene.DefinePrim("/Shifted").GetReferences().AddReference(
        Sdf.Reference("./asset.usda", layerOffset=Sdf.LayerOffset(100, 2))
    )
    scene.GetRootLayer().Save()
    stage_id = usd_tools.load(tmp_path / "scene.usda")

    animation = usd_tools.call("summarize_scene", stage_id=stage_id)["summary"]["animation_info"]

    assert animation["time_sample_range"] == [100.0, 120.0]
    assert animation["duration_time_codes"] == 20.0
    assert animation["duration_seconds"] == pytest.approx(20.0 / 24)


# ---------------------------------------------------------------------------
# M4: summarize_scene on an unknown stage
# ---------------------------------------------------------------------------


def test_m4_summarize_unknown_stage_is_not_found(usd_tools) -> None:
    payload = usd_tools.call("summarize_scene", stage_id="nope_0123")

    assert payload["error_type"] == "NotFoundError"
    assert "nope_0123" in payload["error"]


# ---------------------------------------------------------------------------
# M9: a closed mesh with a flipped face
# ---------------------------------------------------------------------------


def _cube(faces: List[int]) -> MeshArrays:
    return MeshArrays(np.array(CUBE_POINTS, float), np.full(6, 4), np.array(faces))


def test_m9_flipped_face_is_closed_but_not_oriented_and_has_no_volume() -> None:
    good, flipped = geometry_statistics([_cube(CUBE_FACES)])[0], geometry_statistics([_cube(FLIPPED_FACES)])[0]

    assert (good.is_closed, good.consistently_oriented, good.volume) == (True, True, pytest.approx(1.0))
    assert (flipped.is_closed, flipped.consistently_oriented, flipped.volume) == (True, False, None)
    assert flipped.surface_area == pytest.approx(6.0)


def test_m9_the_batch_path_agrees() -> None:
    batch = geometry_statistics([_cube(CUBE_FACES), _cube(FLIPPED_FACES), _cube(CUBE_FACES[:20])])

    assert [(s.is_closed, s.consistently_oriented) for s in batch] == [(True, True), (True, False), (False, None)]
    assert [s.volume for s in batch] == [pytest.approx(1.0), None, None]


def _mesh_file(path: Path, faces: List[int]) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.Xform.Define(stage, "/World")
    mesh = UsdGeom.Mesh.Define(stage, "/World/Box")
    mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in CUBE_POINTS])
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr(faces)
    UsdGeom.Mesh.Define(stage, "/World/Empty")
    stage.GetRootLayer().Save()
    return path


def test_m9_get_mesh_info_says_why_there_is_no_volume(usd_tools, tmp_path: Path) -> None:
    stage_id = usd_tools.load(_mesh_file(tmp_path / "flipped.usda", FLIPPED_FACES))

    info = usd_tools.call("get_mesh_info", stage_id=stage_id, prim_path="/World/Box")
    summary = usd_tools.call("summarize_scene", stage_id=stage_id)["summary"]["mesh_statistics"]

    assert info["is_closed"] is True
    assert info["consistently_oriented"] is False
    assert info["volume"] is None
    assert "wound inconsistently" in info["note"]
    assert summary["inconsistently_wound_meshes"] == 1
    assert summary["total_volume"] == 0.0


# ---------------------------------------------------------------------------
# m1: the private root identifier never reaches a caller
# ---------------------------------------------------------------------------


def test_m1_public_identifier_strips_only_the_session_arguments() -> None:
    assert public_identifier("/a/b.usda:SDF_FORMAT_ARGS:simul_stage=00ff") == "/a/b.usda"
    assert public_identifier("/a/b.usda:SDF_FORMAT_ARGS:a=1&simul_stage=00ff") == "/a/b.usda:SDF_FORMAT_ARGS:a=1"
    assert public_identifier("/a/b.usda:SDF_FORMAT_ARGS:simul_check=0f&z=2") == "/a/b.usda:SDF_FORMAT_ARGS:z=2"
    assert public_identifier("in @/a/b.usda:SDF_FORMAT_ARGS:simul_stage=0f@.") == "in @/a/b.usda@."


def test_m1_no_tool_output_names_the_private_root(usd_tools, tmp_path: Path) -> None:
    _animated(tmp_path / "asset.usda", "/Asset")
    scene = Usd.Stage.CreateNew(str(tmp_path / "scene.usda"))
    scene.DefinePrim("/World/Ok").GetReferences().AddReference("./asset.usda")
    scene.DefinePrim("/World/Missing").GetReferences().AddReference("./missing.usda")
    scene.DefinePrim("/World/BadPath").GetReferences().AddReference("./asset.usda", "/Nope")
    scene.SetDefaultPrim(scene.GetPrimAtPath("/World"))
    scene.GetRootLayer().Save()
    path = str(tmp_path / "scene.usda")

    loaded = usd_tools.call("load_usd_file", file_path=path)
    info = usd_tools.call("get_prim_info", stage_id=loaded["stage_id"], prim_path="/World/Ok")
    summary = usd_tools.call("summarize_scene", stage_id=loaded["stage_id"], format="text")

    seen = json.dumps([loaded, info, summary])
    assert "SDF_FORMAT_ARGS" not in seen and "simul_stage" not in seen
    assert loaded["warnings"] and all(w["location"]["layer"] == path for w in loaded["warnings"])
    assert info["metadata"]["opinion_layers"][0] == path
    assert info["metadata"]["composition"]["references"][0]["layer"] == path
    assert summary["summary"]["file_path"] == path
    assert path in summary["digest"]


# ---------------------------------------------------------------------------
# m4: paging get_prim_subtree is not throttled per page
# ---------------------------------------------------------------------------


def _rate_limited_server(root: Path) -> SimulMCPServer:
    base = Settings()
    security = base.security.model_copy(
        update={
            "allowed_paths": [str(root)],
            "rate_limiting_enabled": True,
            "requests_per_minute": 2,
            "burst_size": 2,
            "global_requests_per_minute": 6000,
        }
    )
    return SimulMCPServer(base.model_copy(update={"security": security}), backends={"usd"})


def test_m4_subtree_continuations_skip_the_per_tool_bucket(tmp_path: Path) -> None:
    layer = Sdf.Layer.CreateNew(str(tmp_path / "wide.usda"))
    Sdf.CreatePrimInLayer(layer, "/World").specifier = Sdf.SpecifierDef
    for index in range(30):
        Sdf.CreatePrimInLayer(layer, f"/World/p{index:02d}").specifier = Sdf.SpecifierDef
    layer.Save()
    server = _rate_limited_server(tmp_path)

    async def walk() -> List[Dict[str, Any]]:
        async with Client(server.mcp) as client:
            async def call(name: str, **arguments: Any) -> Dict[str, Any]:
                result = await client.call_tool(name, arguments, raise_on_error=False)
                return json.loads(result.content[-1].text)

            stage_id = (await call("load_usd_file", file_path=str(tmp_path / "wide.usda")))["stage_id"]
            pages, offset = [], 0
            while offset is not None:
                page = await call("get_prim_subtree", stage_id=stage_id, root_path="/World", max_results=2,
                                  offset=offset)
                pages.append(page)
                offset = page.get("next_offset")
            first_pages = [await call("get_prim_subtree", stage_id=stage_id, root_path="/World") for _ in range(3)]
            return pages + first_pages

    results = asyncio.run(walk())
    walked, first_pages = results[:-3], results[-3:]

    assert all(page["success"] for page in walked), [p for p in walked if not p["success"]][:1]
    assert sum(page["count"] for page in walked) == 31
    # A first page still spends the tool's own bucket.
    assert [page.get("error_type") for page in first_pages].count("RateLimitError") >= 1


def test_m4_only_declared_paged_tools_can_be_exempted(tmp_path: Path) -> None:
    server = _rate_limited_server(tmp_path)

    with pytest.raises(ValueError, match="declares no page"):
        server.exempt_paging_continuations({"get_prim_info"})
    assert "get_prim_subtree" in PAGED_RESULTS


# ---------------------------------------------------------------------------
# m6: a compacted last page has no next page
# ---------------------------------------------------------------------------


def test_m6_a_compacted_last_page_keeps_next_offset_none() -> None:
    big = {"path": "/World/Mesh", "type": "Mesh", "attributes": {"points": {"value": [[1.0, 2.0, 3.0]] * 6000}}}
    payload = {"success": True, "prims": [big], "count": 1, "total": 9, "offset": 8, "next_offset": None}

    result = apply_result_budget(payload, tool_name="get_prim_subtree")

    assert result["prims"][0]["truncated_item"] is True
    assert result["next_offset"] is None
    assert result["count"] == 1
    assert result["truncation"]["next_offset"] is None
    assert "last page" in result["truncation"]["hint"]


def test_m6_a_compacted_page_with_more_after_it_keeps_the_tools_next_offset() -> None:
    big = {"path": "/World/Mesh", "type": "Mesh", "attributes": {"points": {"value": [[1.0, 2.0, 3.0]] * 6000}}}
    payload = {"success": True, "prims": [big], "count": 1, "total": 9, "offset": 3, "next_offset": 4}

    result = apply_result_budget(payload, tool_name="get_prim_subtree")

    assert result["next_offset"] == 4
    assert "offset=4" in result["truncation"]["hint"]


def test_m6_a_cut_page_still_moves_next_offset_back() -> None:
    """#344's invariant: a page cut short continues right after what was kept."""
    prims = [{"path": f"/World/p{index}", "blob": "z" * 2000} for index in range(50)]
    payload = {"success": True, "prims": prims, "count": 50, "total": 120, "offset": 10, "next_offset": 60}

    result = apply_result_budget(payload, tool_name="get_prim_subtree")

    kept = len(result["prims"])
    assert 0 < kept < 50
    assert result["next_offset"] == 10 + kept


# ---------------------------------------------------------------------------
# m7: error typing
# ---------------------------------------------------------------------------


def test_m7_invalid_regex_has_a_code(usd_tools, tmp_path: Path) -> None:
    stage_id = usd_tools.load(_mesh_file(tmp_path / "box.usda", CUBE_FACES))

    payload = usd_tools.call("search_prims", stage_id=stage_id, query="(", search_type="by_name", match="regex")

    assert payload["error_type"] == "ValidationError"
    assert payload["details"]["code"] == "InvalidRegex"
    assert payload["details"]["query"] == "("


def test_m7_a_reload_failure_is_not_a_save_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _mesh_file(tmp_path / "box.usda", CUBE_FACES)
    session = HeadlessUSDSession()
    try:
        stage_id = session.load_stage_or_raise(path)
        root = session.get_stage(stage_id).GetRootLayer()

        def broken(*_args: Any) -> bool:
            raise RuntimeError("disk went away")

        monkeypatch.setattr(type(root), "Reload", broken, raising=False)
        from simul.usd.editing import USDEditError

        with pytest.raises(USDEditError) as caught:
            session.reload_stage(stage_id)
    finally:
        monkeypatch.undo()
        session.cleanup()

    assert caught.value.error_type == "EngineError"
    assert caught.value.details["code"] == "ReloadFailed"


# ---------------------------------------------------------------------------
# m8
# ---------------------------------------------------------------------------


def test_m8_legacy_get_mesh_info_handles_a_mesh_without_points(tmp_path: Path) -> None:
    path = _mesh_file(tmp_path / "box.usda", CUBE_FACES)
    session = HeadlessUSDSession()
    try:
        stage_id = session.load_stage_or_raise(path)
        empty = session.get_mesh_info(stage_id, "/World/Empty")
        box = session.get_mesh_info(stage_id, "/World/Box")
    finally:
        session.cleanup()

    assert empty is not None and empty["bbox"] is None and empty["vertex_count"] == 0
    assert box["bbox"] == {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]}


def test_m8_get_mesh_info_description_names_measured_false(usd_tools) -> None:
    async def describe() -> str:
        async with Client(usd_tools.server.mcp) as client:
            tools = {tool.name: tool for tool in await client.list_tools()}
            return tools["get_mesh_info"].description

    description = asyncio.run(describe())

    assert "measured: false" in description and "sampled_time_range" in description
