"""get_mesh_info and summarize_scene: right materials, normals, time, units and speed.

The audit found get_mesh_info's materials always [] (a call to
UsdGeom.MaterialBindingAPI, which pxr does not have), primvars:normals not
counted as normals, reads that ignored time, 20 s for a million-vertex mesh
(a Python loop per face), and a summary that missed time-sampled animation,
called 100 time codes at 24 per second "100.0s", ignored metersPerUnit, never
warned about referenced assets in other units, counted the session layer and
keyed typeless prims as "".
"""

from __future__ import annotations

import itertools
import json
import math
import random
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pytest

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

CUBE_POINTS = [(-1, -1, -1), (1, -1, -1), (-1, 1, -1), (1, 1, -1), (-1, -1, 1), (1, -1, 1), (-1, 1, 1), (1, 1, 1)]
CUBE_FACES = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]]


def _mesh(stage: Usd.Stage, path: str, points, faces: Sequence[Sequence[int]]) -> UsdGeom.Mesh:
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.GetPointsAttr().Set([Gf.Vec3f(*p) for p in points])
    mesh.GetFaceVertexCountsAttr().Set([len(face) for face in faces])
    mesh.GetFaceVertexIndicesAttr().Set([index for face in faces for index in face])
    return mesh


def _mesh_stage(path: Path) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.Xform.Define(stage, "/World")
    red = UsdShade.Material.Define(stage, "/World/Looks/Red")
    blue = UsdShade.Material.Define(stage, "/World/Looks/Blue")
    rubber = UsdShade.Material.Define(stage, "/World/Looks/Rubber")
    UsdPhysics.MaterialAPI.Apply(rubber.GetPrim()).CreateStaticFrictionAttr(0.9)

    box = _mesh(stage, "/World/Box", CUBE_POINTS, CUBE_FACES)
    UsdShade.MaterialBindingAPI.Apply(box.GetPrim()).Bind(red)
    UsdShade.MaterialBindingAPI.Apply(box.GetPrim()).Bind(rubber, UsdShade.Tokens.weakerThanDescendants, "physics")
    top = UsdGeom.Subset.CreateGeomSubset(box, "top", UsdGeom.Tokens.face, Vt.IntArray([1]), "materialBind")
    UsdShade.MaterialBindingAPI.Apply(top.GetPrim()).Bind(blue)
    primvars = UsdGeom.PrimvarsAPI(box.GetPrim())
    primvars.CreatePrimvar("normals", Sdf.ValueTypeNames.Normal3fArray, UsdGeom.Tokens.faceVarying).Set(
        [Gf.Vec3f(0, 0, 1)] * 24
    )
    primvars.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying).Set(
        [Gf.Vec2f(0, 0)] * 24
    )

    animated = UsdGeom.Mesh.Define(stage, "/World/Deforming")
    animated.GetFaceVertexCountsAttr().Set([3])
    animated.GetFaceVertexIndicesAttr().Set([0, 1, 2])
    animated.GetPointsAttr().Set([Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(0, 1, 0)], 0)
    animated.GetPointsAttr().Set([Gf.Vec3f(0, 0, 0), Gf.Vec3f(10, 0, 0), Gf.Vec3f(0, 10, 0)], 100)

    big = _mesh(stage, "/World/Striped", CUBE_POINTS, CUBE_FACES)
    UsdGeom.Subset.CreateGeomSubset(big, "many", UsdGeom.Tokens.face, Vt.IntArray(list(range(6)) * 50), "partition")
    UsdGeom.Cube.Define(stage, "/World/NotAMesh")
    stage.GetRootLayer().Save()
    return path


@pytest.fixture
def mesh_stage(usd_tools, tmp_path: Path) -> str:
    return usd_tools.load(_mesh_stage(tmp_path / "meshes.usda"))


def _mesh_info(usd_tools, stage_id: str, prim_path: str, **arguments):
    payload = usd_tools.call("get_mesh_info", stage_id=stage_id, prim_path=prim_path, **arguments)
    assert payload.get("success", True) is True, payload
    return payload


class TestMeshInfo:
    def test_materials_resolve_per_mesh_and_per_subset(self, usd_tools, mesh_stage: str) -> None:
        info = _mesh_info(usd_tools, mesh_stage, "/World/Box")

        assert info["materials"] == ["/World/Looks/Red", "/World/Looks/Blue"]
        assert info["physics_material"] == "/World/Looks/Rubber"
        top = next(subset for subset in info["subsets"] if subset["name"] == "top")
        assert top["material"] == "/World/Looks/Blue"
        assert top["family_name"] == "materialBind"

    def test_primvar_normals_and_uv_sets(self, usd_tools, mesh_stage: str) -> None:
        info = _mesh_info(usd_tools, mesh_stage, "/World/Box")

        assert info["has_normals"] is True
        assert info["normals_source"] == "primvars:normals"
        assert info["has_uvs"] is True
        assert info["uv_sets"] == ["st"]

    def test_geometry_of_a_closed_cube(self, usd_tools, mesh_stage: str) -> None:
        info = _mesh_info(usd_tools, mesh_stage, "/World/Box")

        assert info["topology_valid"] is True
        assert info["is_closed"] is True
        assert info["surface_area"] == pytest.approx(24.0)
        assert info["volume"] == pytest.approx(8.0)
        assert info["bbox"]["min"] == [-1.0, -1.0, -1.0]

    def test_time_code_reads_a_deforming_mesh(self, usd_tools, mesh_stage: str) -> None:
        unanimated = _mesh_info(usd_tools, mesh_stage, "/World/Deforming")
        late = _mesh_info(usd_tools, mesh_stage, "/World/Deforming", time_code=100)

        assert unanimated["vertex_count"] == 0
        assert unanimated["bbox"] is None
        assert unanimated["points_time_samples"] == 2
        assert late["vertex_count"] == 3
        assert late["bbox"]["max"] == [10.0, 10.0, 0.0]
        assert late["surface_area"] == pytest.approx(50.0)
        assert late["volume"] is None
        assert late["time_code"] == 100.0

    def test_subset_indices_are_summarised(self, usd_tools, mesh_stage: str) -> None:
        info = _mesh_info(usd_tools, mesh_stage, "/World/Striped")

        subset = info["subsets"][0]
        assert subset["index_count"] == 300
        assert len(subset["indices"]) == 16
        assert subset["indices_truncated"] is True

    def test_deforming_mesh_is_not_measured_rather_than_invalid(self, usd_tools, mesh_stage: str) -> None:
        """Point-cached points have no value at the default time; that is not bad topology."""
        info = _mesh_info(usd_tools, mesh_stage, "/World/Deforming")

        assert info["measured"] is False
        assert info["topology_valid"] is None
        assert info["is_closed"] is None
        assert info["surface_area"] is None
        assert info["sampled_time_range"] == [0.0, 100.0]
        assert "time_code" in info["note"]

    def test_hole_faces_are_open_and_add_no_area(self, usd_tools, tmp_path: Path) -> None:
        stage = Usd.Stage.CreateNew(str(tmp_path / "hole.usda"))
        cube = _mesh(stage, "/Cube", CUBE_POINTS, CUBE_FACES)
        cube.GetHoleIndicesAttr().Set([0])
        stage.GetRootLayer().Save()
        stage_id = usd_tools.load(tmp_path / "hole.usda")

        info = _mesh_info(usd_tools, stage_id, "/Cube")

        assert info["is_closed"] is False
        assert info["surface_area"] == pytest.approx(20.0)
        assert info["volume"] is None

    def test_non_mesh_is_named(self, usd_tools, mesh_stage: str) -> None:
        payload = usd_tools.call("get_mesh_info", stage_id=mesh_stage, prim_path="/World/NotAMesh")

        assert payload["error_type"] == "NotFoundError"
        assert "Cube" in payload["error"]

    def test_million_vertex_mesh_is_fast(self, usd_tools, tmp_path: Path) -> None:
        side = 1000
        xs, ys = np.meshgrid(np.arange(side, dtype=np.float32), np.arange(side, dtype=np.float32))
        points = np.stack([xs.ravel(), ys.ravel(), np.zeros(side * side, dtype=np.float32)], axis=1)
        quad = np.arange((side - 1) * (side - 1))
        row, column = quad // (side - 1), quad % (side - 1)
        corner = row * side + column
        indices = np.stack([corner, corner + 1, corner + side + 1, corner + side], axis=1).ravel().astype(np.int32)
        stage = Usd.Stage.CreateNew(str(tmp_path / "terrain.usdc"))
        mesh = UsdGeom.Mesh.Define(stage, "/Terrain")
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
        mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(quad), 4, dtype=np.int32)))
        mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(indices))
        stage.GetRootLayer().Save()
        stage_id = usd_tools.load(tmp_path / "terrain.usdc")

        started = time.perf_counter()
        info = _mesh_info(usd_tools, stage_id, "/Terrain")
        elapsed = time.perf_counter() - started

        assert elapsed < 5.0, f"{elapsed:.1f}s for a million vertices"
        assert info["vertex_count"] == 1_000_000
        assert info["surface_area"] == pytest.approx(float(len(quad)))
        assert info["is_closed"] is False


def _brute_force(points, counts, indices) -> Tuple[bool, float, float]:
    """The per-face reference: closedness, Newell area and fan signed-volume sum."""
    edges = {}
    area = volume = 0.0
    offset = 0
    pts = [np.array(p, dtype=float) for p in points]
    for count in counts:
        face = indices[offset:offset + count]
        offset += count
        vector = np.zeros(3)
        for i, a in enumerate(face):
            b = face[(i + 1) % count]
            vector += np.cross(pts[a], pts[b])
            if a != b:
                key = (min(a, b), max(a, b))
                edges[key] = edges.get(key, 0) + 1
        area += 0.5 * np.linalg.norm(vector)
        for i in range(1, count - 1):
            v0, v1, v2 = pts[face[0]], pts[face[i]], pts[face[i + 1]]
            volume += np.dot(v0, np.cross(v1, v2)) / 6.0
    closed = bool(edges) and all(uses == 2 for uses in edges.values())
    return closed, area, abs(volume)


def _random_mesh(rng: random.Random) -> Tuple[List[Tuple[float, float, float]], List[int], List[int]]:
    vertices = rng.randint(5, 30)
    points = [(rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-5, 5)) for _ in range(vertices)]
    counts, indices = [], []
    for _ in range(rng.randint(1, 20)):
        size = rng.choice([3, 3, 4, 5])
        counts.append(size)
        indices.extend(rng.sample(range(vertices), size))
    return points, counts, indices


def geometry_statistics(meshes):
    # Imported here so the rest of the module still runs against code without it.
    from simul.usd.mesh_ops import geometry_statistics as measure

    return measure(meshes)


def _arrays(points, counts, indices):
    from simul.usd.mesh_ops import MeshArrays

    return MeshArrays(
        np.asarray(points, dtype=np.float64).reshape(-1, 3),
        np.asarray(counts, dtype=np.int32),
        np.asarray(indices, dtype=np.int32),
    )


def _holed_arrays(points, counts, indices, holes):
    arrays = _arrays(points, counts, indices)
    arrays.holes = np.asarray(holes, dtype=np.int64)
    return arrays


class TestVectorisedGeometry:
    def test_matches_the_per_face_reference(self) -> None:
        rng = random.Random(7)
        meshes = [_random_mesh(rng) for _ in range(60)]
        meshes.append((CUBE_POINTS, [4] * 6, [i for face in CUBE_FACES for i in face]))

        batch = geometry_statistics([_arrays(*mesh) for mesh in meshes])

        for (points, counts, indices), stats in zip(meshes, batch):
            closed, area, volume = _brute_force(points, counts, indices)
            single = geometry_statistics([_arrays(points, counts, indices)])[0]
            assert stats.topology_valid is True
            assert stats.is_closed is closed is single.is_closed
            assert stats.surface_area == pytest.approx(area, rel=1e-9)
            assert single.surface_area == pytest.approx(area, rel=1e-9)
            if closed:
                assert stats.volume == pytest.approx(volume, rel=1e-6, abs=1e-9)
            else:
                assert stats.volume is None and single.volume is None

    def test_invalid_and_empty_meshes(self) -> None:
        cube = _arrays(CUBE_POINTS, [4] * 6, [i for face in CUBE_FACES for i in face])
        out_of_range = _arrays(CUBE_POINTS[:3], [3], [0, 1, 7])
        short_face = _arrays(CUBE_POINTS[:3], [2, 3], [0, 1, 0, 1, 2])
        mismatched = _arrays(CUBE_POINTS[:3], [3, 3], [0, 1, 2, 0, 1])
        empty = _arrays([], [], [])

        stats = geometry_statistics([out_of_range, cube, short_face, empty, mismatched])

        assert [s.topology_valid for s in stats] == [False, True, False, True, False]
        assert stats[0].surface_area is None and stats[0].bbox is not None
        assert stats[1].volume == pytest.approx(8.0)
        assert stats[3].bbox is None and stats[3].surface_area == 0.0 and stats[3].is_closed is False

    def test_concave_polygon_area_is_exact(self) -> None:
        """An L-shaped hexagon has area 3; summing its fan triangles' magnitudes gives 4."""
        l_shape = [(0, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0), (1, 2, 0), (0, 2, 0)]
        regular = [(math.cos(k * math.pi / 3), math.sin(k * math.pi / 3), 0) for k in range(6)]

        stats = geometry_statistics([_arrays(l_shape, [6], list(range(6))), _arrays(regular, [6], list(range(6)))])
        alone = geometry_statistics([_arrays(l_shape, [6], list(range(6)))])[0]

        assert stats[0].surface_area == pytest.approx(3.0)
        assert alone.surface_area == pytest.approx(3.0)
        assert stats[1].surface_area == pytest.approx(3 * math.sqrt(3) / 2)

    def test_hole_faces_are_left_out(self) -> None:
        cube = [i for face in CUBE_FACES for i in face]
        holed = _holed_arrays(CUBE_POINTS, [4] * 6, cube, [0])

        alone = geometry_statistics([holed])[0]
        batch = geometry_statistics([holed, _arrays(CUBE_POINTS, [4] * 6, cube)])
        bad = geometry_statistics([_holed_arrays(CUBE_POINTS, [4] * 6, cube, [6])])[0]

        assert alone.is_closed is False and batch[0].is_closed is False
        assert alone.surface_area == pytest.approx(20.0)
        assert batch[0].surface_area == pytest.approx(20.0)
        assert alone.volume is None
        assert batch[1].is_closed is True and batch[1].surface_area == pytest.approx(24.0)
        assert bad.topology_valid is False

    def test_volume_far_from_the_origin(self) -> None:
        shifted = [(x + 1e5, y + 1e5, z + 1e5) for x, y, z in CUBE_POINTS]

        stats = geometry_statistics([_arrays(shifted, [4] * 6, [i for face in CUBE_FACES for i in face])])[0]

        assert stats.volume == pytest.approx(8.0, rel=1e-9)

    @pytest.mark.parametrize("count", [1, 2, 5])
    def test_batch_equals_one_at_a_time(self, count: int) -> None:
        rng = random.Random(count)
        meshes = [_arrays(*_random_mesh(rng)) for _ in range(count)]

        together = geometry_statistics(meshes)
        alone = list(itertools.chain.from_iterable(geometry_statistics([mesh]) for mesh in meshes))

        for joint, single in zip(together, alone):
            assert joint.is_closed == single.is_closed
            assert joint.surface_area == pytest.approx(single.surface_area)
            assert joint.bbox == single.bbox


def _scene(directory: Path) -> Path:
    asset = Usd.Stage.CreateNew(str(directory / "asset.usda"))
    UsdGeom.SetStageUpAxis(asset, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(asset, 1.0)
    root = UsdGeom.Xform.Define(asset, "/Asset")
    asset.SetDefaultPrim(root.GetPrim())
    _mesh(asset, "/Asset/geo", CUBE_POINTS, CUBE_FACES)
    asset.GetRootLayer().Save()

    path = directory / "scene.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    stage.SetTimeCodesPerSecond(24)
    stage.SetFramesPerSecond(24)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    ground = _mesh(stage, "/World/Ground", CUBE_POINTS, CUBE_FACES)
    ground.AddScaleOp().Set(Gf.Vec3f(500, 1, 500))
    for index in range(3):
        inst = UsdGeom.Xform.Define(stage, f"/World/Asset_{index}").GetPrim()
        inst.GetReferences().AddReference("./asset.usda")
        inst.SetInstanceable(True)
    UsdShade.Material.Define(stage, "/World/Looks/Paint")
    stage.DefinePrim("/World/Untyped")
    mover = UsdGeom.Xform.Define(stage, "/World/Mover")
    op = mover.AddTranslateOp()
    op.Set(Gf.Vec3d(0, 0, 0), 0)
    op.Set(Gf.Vec3d(10, 0, 0), 100)
    stage.GetRootLayer().Save()
    return path


class TestSummary:
    @pytest.fixture
    def summary(self, usd_tools, tmp_path: Path):
        stage_id = usd_tools.load(_scene(tmp_path))
        return usd_tools.call("summarize_scene", stage_id=stage_id, format="text")

    def test_animation_from_time_samples(self, usd_tools, tmp_path: Path, summary) -> None:
        animation = summary["summary"]["animation_info"]
        loaded = usd_tools.call("load_usd_file", file_path=str(tmp_path / "scene.usda"))

        assert animation["has_animation"] is True
        assert animation["authored_time_range"] is False
        assert animation["source"] == "time_samples"
        assert animation["time_sample_range"] == [0.0, 100.0]
        assert animation["duration_time_codes"] == 100.0
        assert animation["duration_seconds"] == pytest.approx(100 / 24)
        assert loaded["has_animation"] is True
        assert "Animation: 4.17 s (100 time codes at 24 per second" in summary["digest"]

    def test_scene_size_in_metres(self, summary) -> None:
        body = summary["summary"]

        assert body["scene_size"][0] == pytest.approx(1000.0)
        assert body["scene_size_meters"][0] == pytest.approx(10.0)
        assert "(10 x" in summary["digest"]

    def test_referenced_units_are_flagged(self, summary) -> None:
        warnings = summary["summary"]["warnings"]

        codes = sorted(warning["code"] for warning in warnings)
        assert codes == ["MetersPerUnitMismatch", "UpAxisMismatch"]
        mpu = next(w for w in warnings if w["code"] == "MetersPerUnitMismatch")
        assert mpu["layer_count"] == 1
        assert mpu["layers"][0].endswith("asset.usda")
        assert "100x smaller" in mpu["message"]
        assert "Warning: asset.usda authors metersPerUnit 1" in summary["digest"]

    def test_counts_types_materials_and_instances(self, summary) -> None:
        body = summary["summary"]

        assert "" not in body["prim_type_counts"]
        assert body["prim_type_counts"]["(typeless)"] == 2  # /World/Untyped and the implicit /World/Looks
        assert body["materials"] == ["/World/Looks/Paint"]
        assert body["material_count"] == 1
        assert body["root_prims"] == ["/World"]
        assert body["root_prim_count"] == 1
        meshes = body["mesh_statistics"]
        assert meshes["total_meshes"] == 4
        assert meshes["instanced_meshes"] == 3
        assert meshes["closed_meshes"] == 4
        assert meshes["total_volume"] == pytest.approx(32.0)

    def test_layer_count_leaves_out_the_session_layer(self, usd_tools, tmp_path: Path, summary) -> None:
        loaded = usd_tools.call("load_usd_file", file_path=str(tmp_path / "scene.usda"))

        assert loaded["layer_count"] == 1
        assert summary["summary"]["stage_info"]["layer_count"] == 1

    def test_authored_range_wins(self, usd_tools, tmp_path: Path) -> None:
        stage = Usd.Stage.CreateNew(str(tmp_path / "ranged.usda"))
        stage.SetStartTimeCode(10)
        stage.SetEndTimeCode(58)
        stage.SetTimeCodesPerSecond(24)
        UsdGeom.Xform.Define(stage, "/World")
        stage.GetRootLayer().Save()
        stage_id = usd_tools.load(tmp_path / "ranged.usda")

        animation = usd_tools.call("summarize_scene", stage_id=stage_id)["summary"]["animation_info"]

        assert animation["source"] == "authored_range"
        assert animation["duration_seconds"] == pytest.approx(2.0)

    def test_a_single_sample_is_not_animation(self, usd_tools, tmp_path: Path) -> None:
        stage = Usd.Stage.CreateNew(str(tmp_path / "pose.usda"))
        UsdGeom.Xform.Define(stage, "/World").AddTranslateOp().Set(Gf.Vec3d(1, 0, 0), 1)
        stage.GetRootLayer().Save()
        loaded = usd_tools.call("load_usd_file", file_path=str(tmp_path / "pose.usda"))

        summary = usd_tools.call("summarize_scene", stage_id=loaded["stage_id"], format="text")

        assert loaded["has_animation"] is False
        assert summary["summary"]["animation_info"]["has_animation"] is False
        assert summary["summary"]["animation_info"]["time_sample_range"] == [1.0, 1.0]
        assert "Animation" not in summary["digest"]

    def test_deforming_meshes_are_measured_at_their_first_sample(self, usd_tools, tmp_path: Path) -> None:
        stage = Usd.Stage.CreateNew(str(tmp_path / "cloth.usda"))
        for index in range(3):
            cloth = UsdGeom.Mesh.Define(stage, f"/Cloth_{index}")
            cloth.GetFaceVertexCountsAttr().Set([4])
            cloth.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
            square = [Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(1, 1, 0), Gf.Vec3f(0, 1, 0)]
            cloth.GetPointsAttr().Set(square, 1)
            cloth.GetPointsAttr().Set([p * 2 for p in square], 48)
        stage.GetRootLayer().Save()
        stage_id = usd_tools.load(tmp_path / "cloth.usda")

        summary = usd_tools.call("summarize_scene", stage_id=stage_id, format="text")
        meshes = summary["summary"]["mesh_statistics"]

        assert meshes["invalid_topology_meshes"] == 0
        assert meshes["meshes_measured_at_first_sample"] == 3
        assert meshes["total_vertices"] == 12
        assert meshes["total_surface_area"] == pytest.approx(3.0)
        assert "3 deforming, measured at their first time sample" in summary["digest"]

    def test_many_mismatched_layers_make_one_warning_per_group(self, usd_tools, tmp_path: Path) -> None:
        """2000 metre assets in a centimetre stage used to cost a warning (and digest line) each."""
        assets = tmp_path / "assets"
        assets.mkdir()
        stage = Usd.Stage.CreateNew(str(tmp_path / "many.usda"))
        UsdGeom.SetStageMetersPerUnit(stage, 0.01)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        UsdGeom.Xform.Define(stage, "/World")
        for index in range(300):
            layer = Sdf.Layer.CreateNew(str(assets / f"a{index:03d}.usda"))
            layer.pseudoRoot.SetInfo("metersPerUnit", 1.0)
            layer.pseudoRoot.SetInfo("upAxis", "Z")
            spec = Sdf.CreatePrimInLayer(layer, "/A")
            spec.specifier = Sdf.SpecifierDef
            layer.defaultPrim = "A"
            layer.Save()
            stage.DefinePrim(f"/World/R{index}").GetReferences().AddReference(f"./assets/a{index:03d}.usda")
        stage.GetRootLayer().Save()
        stage_id = usd_tools.load(tmp_path / "many.usda")

        for fmt in ("json", "text"):
            text = usd_tools.text("summarize_scene", {"stage_id": stage_id, "format": fmt})
            payload = json.loads(text)
            warnings = payload["summary"]["warnings"]

            assert len(text) < 20_000, (fmt, len(text))
            assert "truncated" not in payload and "oversized_bytes" not in payload
            assert [w["code"] for w in warnings] == ["MetersPerUnitMismatch", "UpAxisMismatch"]
            assert all(w["layer_count"] == 300 and len(w["layers"]) == 5 for w in warnings)
            assert "300 layers (a000.usda, a001.usda" in warnings[0]["message"]
            assert "and 295 more" in warnings[0]["message"]
        assert payload["digest"].count("Warning:") == 2

    def test_still_stage_has_no_animation(self, usd_tools, tmp_path: Path) -> None:
        stage = Usd.Stage.CreateNew(str(tmp_path / "still.usda"))
        UsdGeom.Xform.Define(stage, "/World")
        stage.GetRootLayer().Save()
        stage_id = usd_tools.load(tmp_path / "still.usda")

        animation = usd_tools.call("summarize_scene", stage_id=stage_id)["summary"]["animation_info"]

        assert animation["has_animation"] is False
        assert animation["duration_seconds"] == 0.0
        assert math.isclose(animation["time_codes_per_second"], 24.0)
