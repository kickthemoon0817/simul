"""Edit values get the USD type the attribute needs, and writes honour time samples.

JSON has no USD types, so ``[100, 200, 300]`` used to become an ``int[]``
attribute named ``xformOp:translate`` that no transform read, and a
quaternion could not be written at all. These tests pin the typed path
against real ``pxr`` stages: transform ops, quaternions, matrices, assets,
vector arrays, explicit ``attribute_types`` and ``time_code`` writes.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterator

import pytest

pytest.importorskip("pxr", reason="pxr library not available")

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt  # noqa: E402

from simul.adapters.headless_usd import HeadlessUSDSession  # noqa: E402
from simul.usd.editing import USDEditError  # noqa: E402


def _build(path: Path) -> str:
    anim = Usd.Stage.CreateNew(str(path.parent / "anim.usda"))
    anim_root = UsdGeom.Xform.Define(anim, "/Anim")
    anim.SetDefaultPrim(anim_root.GetPrim())
    anim_op = anim_root.AddTranslateOp()
    anim_op.Set(Gf.Vec3d(0, 0, 0), 0)
    anim_op.Set(Gf.Vec3d(0, 5, 0), 10)
    anim.GetRootLayer().Save()

    stage = Usd.Stage.CreateNew(str(path))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Cube.Define(stage, "/World/Box").GetSizeAttr().Set(2.0)
    floaty = UsdGeom.Xform.Define(stage, "/World/FloatOps")
    floaty.AddTranslateOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(0, 0, 0))
    mover = UsdGeom.Xform.Define(stage, "/World/Mover")
    op = mover.AddTranslateOp()
    op.Set(Gf.Vec3d(0, 0, 0), 0)
    op.Set(Gf.Vec3d(10, 0, 0), 24)
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/Joint")
    joint.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    joint.CreateUpperLimitAttr(90.0)
    joint.CreateBody0Rel().SetTargets(["/World/Box"])
    mesh = UsdGeom.Mesh.Define(stage, "/World/Tri")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    UsdGeom.Scope.Define(stage, "/World/Looks")
    stray = UsdGeom.Xform.Define(stage, "/World/Stray").GetPrim()
    stray.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.IntArray).Set([1, 2, 3])
    loose = UsdGeom.Xform.Define(stage, "/World/Loose").GetPrim()
    loose.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(0, 0, 0))
    ordered = UsdGeom.Xform.Define(stage, "/World/Ordered")
    ordered.GetXformOpOrderAttr().Set(["xformOp:translate"])
    UsdGeom.Xform.Define(stage, "/World/RefAnim").GetPrim().GetReferences().AddReference("./anim.usda")
    stage.GetRootLayer().Save()
    return str(path)


@pytest.fixture
def session() -> Iterator[HeadlessUSDSession]:
    usd_session = HeadlessUSDSession()
    yield usd_session
    usd_session.cleanup()


@pytest.fixture
def scene(tmp_path: Path, session: HeadlessUSDSession) -> Dict[str, Any]:
    stage_id = session.load_stage(_build(tmp_path / "typed.usda"))
    stage = session.get_stage(stage_id)
    return {"id": stage_id, "stage": stage, "layer": stage.GetRootLayer()}


def _prim(scene: Dict[str, Any], path: str) -> Usd.Prim:
    return scene["stage"].GetPrimAtPath(path)


def _problems(error: USDEditError) -> Dict[str, str]:
    return {problem["attribute"]: problem["reason"] for problem in error.details["problems"]}


def _refusal(call, *args: Any, **kwargs: Any) -> USDEditError:
    with pytest.raises(USDEditError) as caught:
        call(*args, **kwargs)
    return caught.value


def _translation(prim: Usd.Prim) -> Gf.Vec3d:
    return UsdGeom.Xformable(prim).GetLocalTransformation().ExtractTranslation()


class TestTransformOps:
    def test_new_translate_from_ints_is_a_double3_op_in_the_order(self, session, scene) -> None:
        result = session.update_prim_attributes(scene["id"], "/World/Box", {"xformOp:translate": [100, 200, 300]})

        box = _prim(scene, "/World/Box")
        attr = box.GetAttribute("xformOp:translate")
        assert attr.GetTypeName() == Sdf.ValueTypeNames.Double3
        assert attr.Get() == Gf.Vec3d(100, 200, 300)
        assert list(UsdGeom.Xformable(box).GetXformOpOrderAttr().Get()) == ["xformOp:translate"]
        assert _translation(box) == Gf.Vec3d(100, 200, 300)
        assert result["created_attributes"] == {"xformOp:translate": "double3"}
        assert result["changed"] == ["xformOp:translate", "xformOpOrder"]

    def test_existing_float_op_takes_ints(self, session, scene) -> None:
        session.update_prim_attributes(scene["id"], "/World/FloatOps", {"xformOp:translate": [1, 2, 3]})

        prim = _prim(scene, "/World/FloatOps")
        assert prim.GetAttribute("xformOp:translate").Get() == Gf.Vec3f(1, 2, 3)
        assert _translation(prim) == Gf.Vec3d(1, 2, 3)

    def test_create_with_translate_and_explicit_order(self, session, scene) -> None:
        """The audit's p14 case: ints plus an explicit order used to break get_prim_info."""
        session.create_prim(
            scene["id"],
            "/World/Marker",
            "Cube",
            {"xformOp:translate": [1, 2, 3], "xformOpOrder": ["xformOp:translate"]},
        )

        marker = _prim(scene, "/World/Marker")
        assert marker.GetAttribute("xformOp:translate").GetTypeName() == Sdf.ValueTypeNames.Double3
        assert list(UsdGeom.Xformable(marker).GetXformOpOrderAttr().Get()) == ["xformOp:translate"]
        assert session.get_prim_transform(scene["id"], "/World/Marker")["translation"] == [1.0, 2.0, 3.0]
        assert session.get_prim_info(scene["id"], "/World/Marker") is not None

    def test_new_orient_is_a_quatf_op(self, session, scene) -> None:
        session.update_prim_attributes(scene["id"], "/World/Box", {"xformOp:orient": [0.7071, 0, 0, 0.7071]})

        attr = _prim(scene, "/World/Box").GetAttribute("xformOp:orient")
        assert attr.GetTypeName() == Sdf.ValueTypeNames.Quatf
        assert attr.Get().GetReal() == pytest.approx(0.7071, abs=1e-4)
        assert attr.Get().GetImaginary()[2] == pytest.approx(0.7071, abs=1e-4)
        assert "xformOp:orient" in UsdGeom.Xformable(_prim(scene, "/World/Box")).GetXformOpOrderAttr().Get()

    def test_attribute_types_picks_the_precision(self, session, scene) -> None:
        session.update_prim_attributes(
            scene["id"],
            "/World/Box",
            {"xformOp:translate:offset": [1, 2, 3]},
            attribute_types={"xformOp:translate:offset": "float3"},
        )

        attr = _prim(scene, "/World/Box").GetAttribute("xformOp:translate:offset")
        assert attr.GetTypeName() == Sdf.ValueTypeNames.Float3

        error = _refusal(
            session.update_prim_attributes,
            scene["id"],
            "/World/Box",
            {"xformOp:scale": [1, 1, 1]},
            attribute_types={"xformOp:scale": "quatf"},
        )
        assert _problems(error)["xformOp:scale"] == "a scale op holds double3, float3, half3, not quatf"

    def test_matrix_op_from_rows_or_flat(self, session, scene) -> None:
        rows = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [5, 6, 7, 1]]
        session.update_prim_attributes(scene["id"], "/World/Box", {"xformOp:transform": rows})
        session.update_prim_attributes(
            scene["id"], "/World/FloatOps", {"xformOp:transform:flat": [v for row in rows for v in row]}
        )

        box = _prim(scene, "/World/Box")
        assert box.GetAttribute("xformOp:transform").GetTypeName() == Sdf.ValueTypeNames.Matrix4d
        assert _translation(box) == Gf.Vec3d(5, 6, 7)
        flat = _prim(scene, "/World/FloatOps").GetAttribute("xformOp:transform:flat").Get()
        assert flat.ExtractTranslation() == Gf.Vec3d(5, 6, 7)

    @pytest.mark.parametrize(
        "path, attributes, fragment",
        [
            ("/World/Box", {"xformOp:translat": [1, 2, 3]}, "is not a transform op name"),
            ("/World/Looks", {"xformOp:translate": [1, 2, 3]}, "not xformable"),
            ("/World/Box", {"xformOpOrder": ["xformOp:rotateZ"]}, "does not create one"),
            ("/World/Stray", {"xformOp:translate": [4, 5, 6]}, "cannot drive the transform"),
            ("/World/Box", {"xformOp:translate": [1, 2]}, "double3 takes 3 numbers"),
        ],
    )
    def test_bad_transform_edits_are_refused(self, session, scene, path, attributes, fragment) -> None:
        before = scene["layer"].ExportToString()

        error = _refusal(session.update_prim_attributes, scene["id"], path, attributes)

        assert fragment in next(iter(_problems(error).values()))
        assert scene["layer"].ExportToString() == before

    def test_op_already_in_the_order_is_not_listed_twice(self, session, scene) -> None:
        session.update_prim_attributes(scene["id"], "/World/Ordered", {"xformOp:translate": [1, 2, 3]})

        ordered = _prim(scene, "/World/Ordered")
        assert list(UsdGeom.Xformable(ordered).GetXformOpOrderAttr().Get()) == ["xformOp:translate"]
        assert _translation(ordered) == Gf.Vec3d(1, 2, 3)

    def test_existing_op_outside_the_order_is_added(self, session, scene) -> None:
        result = session.update_prim_attributes(scene["id"], "/World/Loose", {"xformOp:translate": [1, 1, 1]})

        loose = _prim(scene, "/World/Loose")
        assert list(UsdGeom.Xformable(loose).GetXformOpOrderAttr().Get()) == ["xformOp:translate"]
        assert _translation(loose) == Gf.Vec3d(1, 1, 1)
        assert "xformOpOrder" in result["changed"]


class TestValueTypes:
    def test_quaternion_is_w_first(self, session, scene) -> None:
        session.update_prim_attributes(scene["id"], "/World/Joint", {"physics:localRot0": [0.7071, 0, 0, 0.7071]})

        value = _prim(scene, "/World/Joint").GetAttribute("physics:localRot0").Get()
        assert isinstance(value, Gf.Quatf)
        assert value.GetReal() == pytest.approx(0.7071, abs=1e-4)
        assert list(value.GetImaginary()) == pytest.approx([0, 0, 0.7071], abs=1e-4)

    def test_integer_widens_to_float(self, session, scene) -> None:
        session.update_prim_attributes(scene["id"], "/World/Joint", {"physics:upperLimit": 45})

        assert _prim(scene, "/World/Joint").GetAttribute("physics:upperLimit").Get() == 45.0

    def test_integer_attribute_refuses_a_fraction(self, session, scene) -> None:
        session.update_prim_attributes(scene["id"], "/World/Box", {"custom:count": 3})

        error = _refusal(session.update_prim_attributes, scene["id"], "/World/Box", {"custom:count": 1.5})

        assert _problems(error)["custom:count"] == "int takes an integer, not 1.5"

    def test_asset_paths(self, session, scene) -> None:
        result = session.update_prim_attributes(
            scene["id"], "/World/Box", {"custom:tex": "./tex.png"}, attribute_types={"custom:tex": "asset"}
        )
        session.update_prim_attributes(scene["id"], "/World/Box", {"custom:tex": "./other.png"})

        value = _prim(scene, "/World/Box").GetAttribute("custom:tex").Get()
        assert isinstance(value, Sdf.AssetPath)
        assert value.path == "./other.png"
        assert result["created_attributes"] == {"custom:tex": "asset"}

    def test_vector_arrays(self, session, scene) -> None:
        session.update_prim_attributes(
            scene["id"],
            "/World/Tri",
            {"points": [[0, 0, 0], [2, 0, 0], [0, 2, 0]], "primvars:displayColor": [[1, 0, 0]]},
        )
        result = session.update_prim_attributes(scene["id"], "/World/Tri", {"custom:pts": [[1, 0, 0], [0, 1, 0]]})

        tri = _prim(scene, "/World/Tri")
        assert tri.GetAttribute("points").Get() == Vt.Vec3fArray([(0, 0, 0), (2, 0, 0), (0, 2, 0)])
        assert tri.GetAttribute("primvars:displayColor").GetTypeName() == Sdf.ValueTypeNames.Color3fArray
        assert result["created_attributes"] == {"custom:pts": "float3[]"}

    def test_array_errors_name_the_element(self, session, scene) -> None:
        error = _refusal(session.update_prim_attributes, scene["id"], "/World/Tri", {"points": [[0, 0, 0], [1, 0]]})

        assert _problems(error)["points"] == "element 1: point3f takes 3 numbers [x, y, z], not a list of 2 numbers"

    def test_matrix_and_quaternion_custom_attributes(self, session, scene) -> None:
        session.update_prim_attributes(
            scene["id"],
            "/World/Box",
            {"custom:m": [[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1]], "custom:q": [1, 0, 0, 0]},
            attribute_types={"custom:m": "matrix4d", "custom:q": "quatd"},
        )

        box = _prim(scene, "/World/Box")
        assert box.GetAttribute("custom:m").Get() == Gf.Matrix4d(2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1)
        assert box.GetAttribute("custom:q").Get() == Gf.Quatd(1, 0, 0, 0)

    @pytest.mark.parametrize(
        "attributes, attribute_types, fragment",
        [
            ({"size": 3}, {"size": "float"}, "a double attribute; attribute_types cannot change it to float"),
            ({"custom:x": 1}, {"custom:x": "flaot"}, "is not a USD value type"),
            ({"custom:x": 1}, {"custom:y": "float"}, "not in attributes"),
            ({"custom:e": []}, {}, "name one with attribute_types"),
        ],
    )
    def test_type_problems(self, session, scene, attributes, attribute_types, fragment) -> None:
        error = _refusal(
            session.update_prim_attributes, scene["id"], "/World/Box", attributes, attribute_types=attribute_types
        )

        assert any(fragment in reason for reason in _problems(error).values())


class TestTimeCodes:
    def test_animated_attribute_needs_a_time_code(self, session, scene) -> None:
        before = scene["layer"].ExportToString()

        error = _refusal(session.update_prim_attributes, scene["id"], "/World/Mover", {"xformOp:translate": [7, 7, 7]})

        assert _problems(error)["xformOp:translate"] == (
            "animated (2 time sample(s) at 0 to 24), so a default value would be "
            "hidden at every frame; pass time_code to write the sample at one time"
        )
        assert scene["layer"].ExportToString() == before

    def test_time_code_writes_one_sample(self, session, scene) -> None:
        result = session.update_prim_attributes(
            scene["id"], "/World/Mover", {"xformOp:translate": [7, 7, 7]}, time_code=12
        )

        attr = _prim(scene, "/World/Mover").GetAttribute("xformOp:translate")
        assert attr.GetTimeSamples() == [0.0, 12.0, 24.0]
        assert attr.Get(12) == Gf.Vec3d(7, 7, 7)
        assert attr.Get(0) == Gf.Vec3d(0, 0, 0)
        assert result["time_code"] == 12

    def test_sample_over_a_weaker_layers_animation_is_refused(self, session, scene) -> None:
        """One root-layer sample would hide every sample the referenced asset authors."""
        before = scene["layer"].ExportToString()

        error = _refusal(
            session.update_prim_attributes,
            scene["id"],
            "/World/RefAnim",
            {"xformOp:translate": [1, 1, 1]},
            time_code=5,
        )

        reason = _problems(error)["xformOp:translate"]
        assert "come from anim.usda" in reason and "replace that whole animation" in reason
        assert scene["layer"].ExportToString() == before

    @pytest.mark.parametrize(
        "attributes, fragment",
        [
            ({"xformOpOrder": ["xformOp:translate"]}, "uniform, so it cannot have time samples"),
            ({"active": False}, "prim metadata, which has no time samples"),
            ({"physics:body1": "/World/Box"}, "relationship, which has no time samples"),
        ],
    )
    def test_time_code_refused_where_samples_cannot_exist(self, session, scene, attributes, fragment) -> None:
        path = "/World/Joint" if "physics:body1" in attributes else "/World/Mover"

        error = _refusal(session.update_prim_attributes, scene["id"], path, attributes, time_code=5)

        assert fragment in next(iter(_problems(error).values()))

    def test_create_with_a_time_code(self, session, scene) -> None:
        session.create_prim(scene["id"], "/World/Pulse", "Cube", {"size": 2}, time_code=5)

        size = _prim(scene, "/World/Pulse").GetAttribute("size")
        assert size.GetTimeSamples() == [5.0]
        assert not scene["layer"].GetAttributeAtPath("/World/Pulse.size").HasDefaultValue()

    def test_non_finite_time_code(self, session, scene) -> None:
        error = _refusal(
            session.update_prim_attributes, scene["id"], "/World/Box", {"size": 1}, time_code=math.nan
        )

        assert error.error_type == "InvalidTimeCode"
