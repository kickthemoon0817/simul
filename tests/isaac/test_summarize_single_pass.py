"""``SceneSummarizer.summarize_stage`` walks the stage once.

It used to traverse the whole stage five times per summary: once in
``USDReader.get_stage_info`` (which also materialised every prim into a list
just to count them), then again each for the prim-type counts, the material
list, the mesh statistics and the hierarchy depth. One pass now fills every
accumulator and hands the prim count to ``get_stage_info``.

The reference below is the previous multi-pass implementation, kept verbatim
in spirit, so the single pass is checked to produce the same summary on a
non-trivial stage — same counts, same material order, same float sums.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

from simul_mcp.usd.reader import USDReader
from simul_mcp.usd.summarize import MaterialSummary, SceneSummarizer


def _quad_mesh(stage: Usd.Stage, path: str, offset: float, *, normals: bool, uvs: bool, colors: bool) -> None:
    mesh = UsdGeom.Mesh.Define(stage, path)
    points = [
        Gf.Vec3f(offset, 0, 0), Gf.Vec3f(offset + 1, 0, 0),
        Gf.Vec3f(offset + 1, 1, 0), Gf.Vec3f(offset, 1, 0),
        Gf.Vec3f(offset, 0, 1), Gf.Vec3f(offset + 1, 0, 1),
        Gf.Vec3f(offset + 1, 1, 1), Gf.Vec3f(offset, 1, 1),
    ]
    mesh.CreatePointsAttr(Vt.Vec3fArray(points))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4] * 6))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([
        0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4,
        1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7,
    ]))
    mesh.CreateExtentAttr([Gf.Vec3f(offset, 0, 0), Gf.Vec3f(offset + 1, 1, 1)])
    if normals:
        mesh.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(0, 0, 1)] * 8))
    if uvs:
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
        ).Set(Vt.Vec2fArray([Gf.Vec2f(0, 0)] * 8))
    if colors:
        mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(1, 0, 0)]))


@pytest.fixture
def stage() -> Usd.Stage:
    """Nested Xforms, meshes with mixed primvars, gprims, lights, materials."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(48)
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Scope.Define(stage, "/Looks")
    for i in range(3):
        UsdShade.Material.Define(stage, f"/Looks/Mat{i}")
        UsdShade.Shader.Define(stage, f"/Looks/Mat{i}/Shader")
    parent = "/World"
    for depth in range(6):
        parent = f"{parent}/Level{depth}"
        UsdGeom.Xform.Define(stage, parent)
        _quad_mesh(
            stage, f"{parent}/Mesh", float(depth),
            normals=depth % 2 == 0, uvs=depth % 3 == 0, colors=depth % 4 == 0,
        )
        cube = UsdGeom.Cube.Define(stage, f"{parent}/Cube")
        cube.AddTranslateOp().Set(Gf.Vec3d(depth, depth, 0))
    UsdShade.Material.Define(stage, "/World/Level0/LocalMat")
    stage.DefinePrim("/World/Light", "SphereLight")
    stage.DefinePrim("/Untyped")
    return stage


def _reference_passes(summarizer: SceneSummarizer, stage: Usd.Stage) -> Dict[str, Any]:
    """The multi-pass computation summarize_stage used before the single pass."""
    prim_type_counts: Dict[str, int] = {}
    total_prims = 0
    for prim in stage.Traverse():
        total_prims += 1
        prim_type = prim.GetTypeName()
        prim_type_counts[prim_type] = prim_type_counts.get(prim_type, 0) + 1

    materials: List[MaterialSummary] = []
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Material":
            materials.append(MaterialSummary(
                path=str(prim.GetPath()), name=prim.GetName(),
                shader_type="Unknown", parameters={}, textures=[],
            ))

    stats = {
        'total_meshes': 0, 'total_vertices': 0, 'total_faces': 0,
        'meshes_with_normals': 0, 'meshes_with_uvs': 0, 'meshes_with_colors': 0,
        'closed_meshes': 0, 'total_surface_area': 0.0, 'total_volume': 0.0,
    }
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            info = summarizer.mesh_ops.get_mesh_statistics(prim)
            stats['total_meshes'] += 1
            stats['total_vertices'] += info.vertex_count
            stats['total_faces'] += info.face_count
            stats['meshes_with_normals'] += int(bool(info.has_normals))
            stats['meshes_with_uvs'] += int(bool(info.has_uvs))
            stats['meshes_with_colors'] += int(bool(info.has_colors))
            stats['closed_meshes'] += int(bool(info.is_closed))
            stats['total_surface_area'] += info.surface_area
            stats['total_volume'] += info.volume

    depth = 0
    for prim in stage.Traverse():
        depth = max(depth, str(prim.GetPath()).count('/'))

    return {
        'total_prims': total_prims,
        'prim_type_counts': prim_type_counts,
        'materials': materials,
        'mesh_statistics': stats,
        'hierarchy_depth': depth,
    }


def test_single_pass_matches_multi_pass_reference(stage: Usd.Stage) -> None:
    summarizer = SceneSummarizer()
    reference = _reference_passes(summarizer, stage)
    summary = summarizer.summarize_stage(stage, file_path="mem.usda")

    assert summary.total_prims == reference['total_prims']
    assert summary.prim_type_counts == reference['prim_type_counts']
    assert list(summary.prim_type_counts) == list(reference['prim_type_counts'])
    assert summary.materials == reference['materials']
    assert summary.mesh_statistics == reference['mesh_statistics']
    assert summary.hierarchy_depth == reference['hierarchy_depth']

    # The stage is non-trivial in every accumulator the pass fills.
    assert summary.total_prims == 29
    assert len(summary.materials) == 4
    assert summary.mesh_statistics['total_meshes'] == 6
    assert 0 < summary.mesh_statistics['meshes_with_uvs'] < 6
    assert summary.hierarchy_depth == 8


def test_summary_is_identical_to_repeated_run(stage: Usd.Stage) -> None:
    """Every field — root prim summaries, bounds, stage info — is deterministic."""
    first = dataclasses.asdict(SceneSummarizer().summarize_stage(stage, "mem.usda"))
    second = dataclasses.asdict(SceneSummarizer().summarize_stage(stage, "mem.usda"))
    assert first == second


def test_include_flags_still_gate_materials_and_mesh_statistics(stage: Usd.Stage) -> None:
    summary = SceneSummarizer().summarize_stage(stage, include_meshes=False, include_materials=False)
    assert summary.materials == []
    assert summary.mesh_statistics == {}
    assert summary.total_prims == sum(1 for _ in stage.Traverse())


def test_get_stage_info_accepts_a_precomputed_prim_count(stage: Usd.Stage) -> None:
    reader = USDReader()
    counted = reader.get_stage_info(stage)
    assert counted.prim_count == sum(1 for _ in stage.Traverse())

    supplied = reader.get_stage_info(stage, prim_count=counted.prim_count)
    assert dataclasses.asdict(supplied) == dataclasses.asdict(counted)
