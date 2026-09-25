"""Correctness of headless USD bounds, cache invalidation, reload and mesh closure.

These run against real ``pxr`` stages (usd-core) rather than mocks: each test
pins a defect where the headless session returned a plausible-looking but
wrong answer.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import pytest

pytest.importorskip("pxr", reason="pxr library not available")

from pxr import Usd, UsdGeom  # noqa: E402

from simul_mcp.adapters.headless_usd import HeadlessUSDSession  # noqa: E402
from simul_mcp.usd.bbox import BBoxCache  # noqa: E402
from simul_mcp.usd.mesh_ops import MeshOperations  # noqa: E402

CUBE_POINTS: List[Tuple[float, float, float]] = [
    (-1, -1, -1),
    (1, -1, -1),
    (-1, 1, -1),
    (1, 1, -1),
    (-1, -1, 1),
    (1, -1, 1),
    (-1, 1, 1),
    (1, 1, 1),
]
CUBE_FACES: List[List[int]] = [
    [0, 2, 3, 1],
    [4, 5, 7, 6],
    [0, 1, 5, 4],
    [2, 6, 7, 3],
    [0, 4, 6, 2],
    [1, 3, 7, 5],
]


def _write_stage(path: Path, prims: Sequence[Tuple[str, str]]) -> str:
    """Create a USD file on disk holding the given (path, type) prims."""
    stage = Usd.Stage.CreateNew(str(path))
    for prim_path, prim_type in prims:
        stage.DefinePrim(prim_path, prim_type)
    stage.GetRootLayer().Save()
    return str(path)


@pytest.fixture
def session() -> HeadlessUSDSession:
    """A fresh headless session, cleaned up after the test."""
    usd_session = HeadlessUSDSession()
    yield usd_session
    usd_session.cleanup()


class TestNoFabricatedBounds:
    """Prims and stages without geometry have no bounding box."""

    @pytest.mark.parametrize("prim_type", ["Camera", "Scope", "Xform"])
    def test_non_geometry_prim_has_no_bbox(
        self, tmp_path: Path, session: HeadlessUSDSession, prim_type: str
    ) -> None:
        stage_id = session.load_stage(_write_stage(tmp_path / "s.usda", [("/P", prim_type)]))

        assert session.get_prim_bbox(stage_id, "/P", world_space=True) is None
        assert session.get_prim_bbox(stage_id, "/P", world_space=False) is None

    def test_geometry_less_stage_has_no_bbox(self, tmp_path: Path, session: HeadlessUSDSession) -> None:
        stage_id = session.load_stage(
            _write_stage(tmp_path / "s.usda", [("/Cam", "Camera"), ("/Grp", "Scope"), ("/X", "Xform")])
        )

        assert session.get_stage_bbox(stage_id) is None

    def test_geometry_still_bounded(self, tmp_path: Path, session: HeadlessUSDSession) -> None:
        stage_id = session.load_stage(_write_stage(tmp_path / "s.usda", [("/Cam", "Camera")]))
        assert session.create_prim(stage_id, "/World/Box", "Cube", {"size": 4.0})

        assert session.get_stage_bbox(stage_id) == {"min": [-2.0, -2.0, -2.0], "max": [2.0, 2.0, 2.0]}


class TestTransformedBounds:
    """Bounds come from USD's BBoxCache, so transforms are honoured."""

    def test_translated_child(self, tmp_path: Path, session: HeadlessUSDSession) -> None:
        path = tmp_path / "s.usda"
        stage = Usd.Stage.CreateNew(str(path))
        parent = UsdGeom.Xform.Define(stage, "/World")
        parent.AddTranslateOp().Set((100.0, 0.0, 0.0))
        child = UsdGeom.Cube.Define(stage, "/World/Box")
        child.GetSizeAttr().Set(2.0)
        child.AddTranslateOp().Set((0.0, 10.0, 0.0))
        stage.GetRootLayer().Save()
        del stage
        stage_id = session.load_stage(str(path))

        assert session.get_prim_bbox(stage_id, "/World/Box") == {"min": [99.0, 9.0, -1.0], "max": [101.0, 11.0, 1.0]}
        # Local (object) space: the child's own translate applies, the parent's does not.
        assert session.get_prim_bbox(stage_id, "/World", world_space=False) == {
            "min": [-1.0, 9.0, -1.0],
            "max": [1.0, 11.0, 1.0],
        }
        assert session.get_prim_bbox(stage_id, "/World/Box", world_space=False) == {
            "min": [-1.0, -1.0, -1.0],
            "max": [1.0, 1.0, 1.0],
        }


class TestCylinderManualBounds:
    """The manual fallback honours the Cylinder ``axis`` attribute."""

    @pytest.mark.parametrize(
        "axis, expected",
        [
            (None, ([-1.0, -1.0, -3.0], [1.0, 1.0, 3.0])),  # schema default is Z
            ("X", ([-3.0, -1.0, -1.0], [3.0, 1.0, 1.0])),
            ("Y", ([-1.0, -3.0, -1.0], [1.0, 3.0, 1.0])),
            ("Z", ([-1.0, -1.0, -3.0], [1.0, 1.0, 3.0])),
        ],
    )
    def test_axis(self, axis, expected) -> None:
        stage = Usd.Stage.CreateInMemory()
        cylinder = UsdGeom.Cylinder.Define(stage, "/C")
        cylinder.GetRadiusAttr().Set(1.0)
        cylinder.GetHeightAttr().Set(6.0)
        if axis is not None:
            cylinder.GetAxisAttr().Set(axis)

        bbox = BBoxCache(stage)._compute_local_bbox_manual(cylinder.GetPrim())

        assert bbox is not None
        assert [list(bbox[0]), list(bbox[1])] == [expected[0], expected[1]]


class TestBBoxInvalidation:
    """Mutations through the session invalidate cached bounds."""

    def test_update_prim_attributes_refreshes_bbox(self, tmp_path: Path, session: HeadlessUSDSession) -> None:
        stage_id = session.load_stage(_write_stage(tmp_path / "s.usda", []))
        assert session.create_prim(stage_id, "/Box", "Cube", {"size": 2.0})
        assert session.get_prim_bbox(stage_id, "/Box")["max"] == [1.0, 1.0, 1.0]
        assert session.get_prim_bbox(stage_id, "/Box", world_space=False)["max"] == [1.0, 1.0, 1.0]

        assert session.update_prim_attributes(stage_id, "/Box", {"size": 10.0})

        assert session.get_prim_bbox(stage_id, "/Box")["max"] == [5.0, 5.0, 5.0]
        assert session.get_prim_bbox(stage_id, "/Box", world_space=False)["max"] == [5.0, 5.0, 5.0]

    def test_create_prim_refreshes_stage_bbox(self, tmp_path: Path, session: HeadlessUSDSession) -> None:
        stage_id = session.load_stage(_write_stage(tmp_path / "s.usda", []))
        assert session.create_prim(stage_id, "/World/A", "Cube", {"size": 2.0})
        assert session.get_prim_bbox(stage_id, "/World")["max"] == [1.0, 1.0, 1.0]

        assert session.create_prim(stage_id, "/World/B", "Sphere", {"radius": 3.0})

        assert session.get_prim_bbox(stage_id, "/World") == {"min": [-3.0, -3.0, -3.0], "max": [3.0, 3.0, 3.0]}

    def test_delete_prim_refreshes_bbox(self, tmp_path: Path, session: HeadlessUSDSession) -> None:
        stage_id = session.load_stage(_write_stage(tmp_path / "s.usda", []))
        assert session.create_prim(stage_id, "/World/Small", "Cube", {"size": 2.0})
        assert session.create_prim(stage_id, "/World/Big", "Cube", {"size": 8.0})
        assert session.get_prim_bbox(stage_id, "/World")["max"] == [4.0, 4.0, 4.0]

        assert session.delete_prim(stage_id, "/World/Big")

        assert session.get_prim_bbox(stage_id, "/World")["max"] == [1.0, 1.0, 1.0]
        assert session.get_stage_bbox(stage_id)["max"] == [1.0, 1.0, 1.0]


class TestUnloadReload:
    """Unloading a stage discards its in-memory state."""

    def test_reload_reads_file_not_unsaved_stage(self, tmp_path: Path, session: HeadlessUSDSession) -> None:
        file_path = _write_stage(tmp_path / "s.usda", [("/Saved", "Xform")])
        stage_id = session.load_stage(file_path)
        assert session.create_prim(stage_id, "/Unsaved", "Xform")

        assert session.unload_stage(stage_id)
        reloaded_id = session.load_stage(file_path)

        reloaded = session.get_stage(reloaded_id)
        assert reloaded.GetPrimAtPath("/Saved").IsValid()
        assert not reloaded.GetPrimAtPath("/Unsaved").IsValid()

    def test_load_twice_without_unload_shares_stage(self, tmp_path: Path, session: HeadlessUSDSession) -> None:
        file_path = _write_stage(tmp_path / "s.usda", [("/Saved", "Xform")])
        first = session.get_stage(session.load_stage(file_path))

        second = session.get_stage(session.load_stage(file_path))

        assert first is second


def _mesh(stage: Usd.Stage, path: str, points, faces: List[List[int]]) -> Usd.Prim:
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.GetPointsAttr().Set(points)
    mesh.GetFaceVertexCountsAttr().Set([len(face) for face in faces])
    mesh.GetFaceVertexIndicesAttr().Set([index for face in faces for index in face])
    return mesh.GetPrim()


class TestMeshClosed:
    """``is_closed`` means no boundary edges, and volume follows it."""

    def test_closed_cube(self) -> None:
        stage = Usd.Stage.CreateInMemory()
        info = MeshOperations().get_mesh_statistics(_mesh(stage, "/Cube", CUBE_POINTS, CUBE_FACES))

        assert info.is_closed is True
        assert info.volume == pytest.approx(8.0)

    def test_open_box_missing_one_face(self) -> None:
        stage = Usd.Stage.CreateInMemory()
        info = MeshOperations().get_mesh_statistics(_mesh(stage, "/Box", CUBE_POINTS, CUBE_FACES[:1] + CUBE_FACES[2:]))

        assert info.is_closed is False
        assert info.volume == 0.0

    def test_open_grid(self) -> None:
        n = 4
        points = [(x, y, 0.5 * x * y) for y in range(n) for x in range(n)]
        faces = [
            [y * n + x, y * n + x + 1, (y + 1) * n + x + 1, (y + 1) * n + x]
            for y in range(n - 1)
            for x in range(n - 1)
        ]
        stage = Usd.Stage.CreateInMemory()
        info = MeshOperations().get_mesh_statistics(_mesh(stage, "/Grid", points, faces))

        assert info.is_closed is False
        assert info.volume == 0.0

    def test_closed_tetrahedron(self) -> None:
        points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        faces = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
        stage = Usd.Stage.CreateInMemory()
        info = MeshOperations().get_mesh_statistics(_mesh(stage, "/Tet", points, faces))

        assert info.is_closed is True
        assert info.volume == pytest.approx(1.0 / 6.0)
