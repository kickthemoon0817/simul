"""Bounds follow USD's BBoxCache: every transform applies, one purpose rule everywhere.

The headless tools used to union a parent's children without their transforms
whenever ``UsdGeom.BBoxCache`` (restricted to the ``default`` purpose) came up
empty, so a render-purpose cube translated to z=50 under a link translated to
x=100 was reported at the origin, and a guide-only group bounded its guides
while the stage around it did not. These tests pin pxr's own answer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pxr import Gf, Usd, UsdGeom

from simul.adapters.headless_usd import HeadlessUSDSession
from simul.usd.bbox import BBoxCache

RENDER_CUBE = {"min": [95.0, -5.0, 45.0], "max": [105.0, 5.0, 55.0]}


def _purpose_stage(path: Path) -> Path:
    """The audit's purpose_xf layout: a render cube two transforms down, plus a guide group."""
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/Robot")
    link = UsdGeom.Xform.Define(stage, "/World/Robot/link")
    link.AddTranslateOp().Set(Gf.Vec3d(100, 0, 0))
    visual = UsdGeom.Cube.Define(stage, "/World/Robot/link/visual")
    visual.GetSizeAttr().Set(10.0)
    visual.GetPurposeAttr().Set(UsdGeom.Tokens.render)
    visual.AddTranslateOp().Set(Gf.Vec3d(0, 0, 50))
    guides = UsdGeom.Xform.Define(stage, "/World/Guides")
    guides.AddTranslateOp().Set(Gf.Vec3d(0, 800, 0))
    shape = UsdGeom.Cube.Define(stage, "/World/Guides/shape")
    shape.GetSizeAttr().Set(2.0)
    shape.GetPurposeAttr().Set(UsdGeom.Tokens.guide)
    UsdGeom.Scope.Define(stage, "/World/Empty")
    mover = UsdGeom.Xform.Define(stage, "/World/Mover")
    op = mover.AddTranslateOp()
    op.Set(Gf.Vec3d(0, 0, 0), 0)
    op.Set(Gf.Vec3d(1000, 0, 0), 100)
    UsdGeom.Cube.Define(stage, "/World/Mover/box").GetSizeAttr().Set(10.0)
    stage.GetRootLayer().Save()
    return path


@pytest.fixture
def stage_path(tmp_path: Path) -> Path:
    return _purpose_stage(tmp_path / "purpose_xf.usda")


class TestTransformsApply:
    """Ancestors' and descendants' transforms are part of every bound."""

    def test_parent_bound_keeps_child_transforms(self, stage_path: Path) -> None:
        session = HeadlessUSDSession()
        stage_id = session.load_stage(str(stage_path))

        assert session.get_prim_bbox(stage_id, "/World/Robot") == RENDER_CUBE
        assert session.get_prim_bbox(stage_id, "/World/Robot/link") == RENDER_CUBE
        session.cleanup()

    def test_local_space_drops_only_the_prims_own_transform(self, usd_tools, stage_path: Path) -> None:
        stage_id = usd_tools.load(stage_path)

        local = usd_tools.call(
            "get_bounding_box", stage_id=stage_id, prim_path="/World/Robot/link", world_space=False
        )

        assert local["bbox"]["min"] == [-5.0, -5.0, 45.0]
        assert local["bbox"]["max"] == [5.0, 5.0, 55.0]
        assert local["world_space"] is False


class TestOnePurposeRule:
    """The prim, its parent and the stage all count the same purposes."""

    def test_default_is_default_plus_render(self, usd_tools, stage_path: Path) -> None:
        stage_id = usd_tools.load(stage_path)

        robot = usd_tools.call("get_bounding_box", stage_id=stage_id, prim_path="/World/Robot")

        assert robot["success"] is True
        assert robot["has_geometry"] is True
        assert robot["included_purposes"] == ["default", "render"]
        assert {"min": robot["bbox"]["min"], "max": robot["bbox"]["max"]} == RENDER_CUBE
        assert robot["bbox"]["center"] == [100.0, 0.0, 50.0]
        assert robot["bbox"]["size"] == [10.0, 10.0, 10.0]

    def test_guide_group_is_empty_like_the_stage_says(self, usd_tools, stage_path: Path) -> None:
        stage_id = usd_tools.load(stage_path)

        guides = usd_tools.call("get_bounding_box", stage_id=stage_id, prim_path="/World/Guides")
        stage = usd_tools.call("get_bounding_box", stage_id=stage_id)

        assert guides["success"] is True
        assert guides["has_geometry"] is False
        assert guides["bbox"] is None
        assert guides["excluded_purposes_with_geometry"] == ["guide"]
        assert "include_purposes" in guides["message"]
        # The stage bound leaves the guide out too: nothing reaches y=800.
        assert stage["bbox"]["max"][1] == 5.0

    def test_include_purposes_widens_every_level(self, usd_tools, stage_path: Path) -> None:
        stage_id = usd_tools.load(stage_path)
        everything = ["default", "render", "proxy", "guide"]

        guides = usd_tools.call(
            "get_bounding_box", stage_id=stage_id, prim_path="/World/Guides", include_purposes=everything
        )
        stage = usd_tools.call("get_bounding_box", stage_id=stage_id, include_purposes=everything)

        assert guides["bbox"]["min"] == [-1.0, 799.0, -1.0]
        assert stage["bbox"]["max"][1] == 801.0
        assert stage["included_purposes"] == everything

    def test_default_purpose_only_is_an_explicit_empty_result(self, usd_tools, stage_path: Path) -> None:
        stage_id = usd_tools.load(stage_path)

        robot = usd_tools.call(
            "get_bounding_box", stage_id=stage_id, prim_path="/World/Robot", include_purposes=["default"]
        )

        assert robot["success"] is True
        assert robot["bbox"] is None
        assert robot["has_geometry"] is False
        assert robot["excluded_purposes_with_geometry"] == ["render"]

    def test_no_geometry_is_not_an_error(self, usd_tools, stage_path: Path) -> None:
        stage_id = usd_tools.load(stage_path)

        empty = usd_tools.call("get_bounding_box", stage_id=stage_id, prim_path="/World/Empty")

        assert empty["success"] is True
        assert empty["has_geometry"] is False
        assert empty["excluded_purposes_with_geometry"] == []

    def test_prim_info_bbox_uses_the_same_rule(self, usd_tools, stage_path: Path) -> None:
        stage_id = usd_tools.load(stage_path)

        robot = usd_tools.call("get_prim_info", stage_id=stage_id, prim_path="/World/Robot")
        guides = usd_tools.call("get_prim_info", stage_id=stage_id, prim_path="/World/Guides")

        assert robot["bbox"]["min"] == RENDER_CUBE["min"]
        assert guides["bbox"] is None

    def test_summary_scene_bbox_matches_the_stage_bound(self, usd_tools, stage_path: Path) -> None:
        stage_id = usd_tools.load(stage_path)

        summary = usd_tools.call("summarize_scene", stage_id=stage_id)["summary"]
        stage = usd_tools.call("get_bounding_box", stage_id=stage_id)

        assert summary["scene_bbox"] == {"min": stage["bbox"]["min"], "max": stage["bbox"]["max"]}


class TestTimeCode:
    """``time_code`` evaluates animated transforms."""

    def test_bound_follows_the_animation(self, usd_tools, stage_path: Path) -> None:
        stage_id = usd_tools.load(stage_path)

        at_end = usd_tools.call("get_bounding_box", stage_id=stage_id, prim_path="/World/Mover", time_code=100)
        at_start = usd_tools.call("get_bounding_box", stage_id=stage_id, prim_path="/World/Mover", time_code=0)

        assert at_end["bbox"]["min"] == [995.0, -5.0, -5.0]
        assert at_end["time_code"] == 100.0
        assert at_start["bbox"]["min"] == [-5.0, -5.0, -5.0]


class TestLookupErrors:
    """A missing stage or prim, or a malformed path, is named as such."""

    def test_unknown_stage(self, usd_tools) -> None:
        payload = usd_tools.call("get_bounding_box", stage_id="nope")

        assert payload["success"] is False
        assert payload["error_type"] == "NotFoundError"
        assert "load_usd_file" in payload["error"]

    @pytest.mark.parametrize(
        "prim_path, error_type",
        [("/World/Nope", "NotFoundError"), ("World", "ValidationError"), ("/World/", "ValidationError"),
         ("/World/Robot.size", "ValidationError")],
    )
    def test_bad_prim_path(self, usd_tools, stage_path: Path, prim_path: str, error_type: str) -> None:
        stage_id = usd_tools.load(stage_path)

        payload = usd_tools.call("get_bounding_box", stage_id=stage_id, prim_path=prim_path)

        assert payload["success"] is False
        assert payload["error_type"] == error_type
        if error_type == "ValidationError":
            # The code the edit tools report for the same check, beside the
            # errors list every ValidationError carries.
            details = dict(payload["details"])
            assert [error["message"] for error in details.pop("errors")] == [payload["error"]]
            assert details == {"code": "InvalidPrimPath", "prim_path": prim_path}
        else:
            assert payload["details"] == {"prim_path": prim_path}

    def test_unknown_purpose_is_rejected(self, usd_tools, stage_path: Path) -> None:
        stage_id = usd_tools.load(stage_path)

        result = usd_tools.raw(
            "get_bounding_box", {"stage_id": stage_id, "include_purposes": ["collision"]}
        )

        assert result.is_error


class TestLibrary:
    """The library calls the tools sit on."""

    def test_describe_bounds_of_the_stage(self, stage_path: Path) -> None:
        from simul.usd.bbox import describe_bounds

        stage = Usd.Stage.Open(str(stage_path))

        bounds = describe_bounds(stage)

        assert bounds["has_geometry"] is True
        assert bounds["time_code"] is None
        assert bounds["bbox"]["max"] == [105.0, 5.0, 55.0]

    @pytest.mark.parametrize(
        "axis, expected",
        [
            (None, ([-1.0, -1.0, -3.0], [1.0, 1.0, 3.0])),  # schema default is Z
            ("X", ([-3.0, -1.0, -1.0], [3.0, 1.0, 1.0])),
            ("Y", ([-1.0, -3.0, -1.0], [1.0, 3.0, 1.0])),
        ],
    )
    def test_cylinder_axis(self, axis, expected) -> None:
        stage = Usd.Stage.CreateInMemory()
        cylinder = UsdGeom.Cylinder.Define(stage, "/C")
        cylinder.GetRadiusAttr().Set(1.0)
        cylinder.GetHeightAttr().Set(6.0)
        if axis is not None:
            cylinder.GetAxisAttr().Set(axis)

        bbox = BBoxCache(stage).compute_local_bbox(cylinder.GetPrim())

        assert [list(bbox[0]), list(bbox[1])] == [expected[0], expected[1]]

    def test_unknown_purpose_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown purpose"):
            BBoxCache(Usd.Stage.CreateInMemory(), purposes=["collision"])
