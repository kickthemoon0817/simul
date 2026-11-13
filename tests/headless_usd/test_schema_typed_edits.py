"""Edits follow the schema registry: relationships stay relationships, schema attributes keep their type.

Round-2 audit findings, pinned against real ``pxr`` stages:

- B1: a relationship name (``material:binding``, ``physics:body0``,
  ``skel:skeleton``) used to be written as an attribute, after which pxr
  refused to read the prim's bindings ("is not a relationship"). Such names
  are refused now, and the reads report a file that already holds one instead
  of failing.
- M3: an attribute of an API schema the prim lacks (``physics:mass``,
  ``drive:angular:physics:stiffness``) used to get a type guessed from JSON.
- B2: a time sample written over value clips replaced the whole clip
  animation.
- M10: non-finite floats as the reads spell them, and finite values a float
  cannot hold.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterator

import pytest

pytest.importorskip("pxr", reason="pxr library not available")

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

from simul.adapters.headless_usd import HeadlessUSDSession  # noqa: E402
from simul.usd import coerce, editing, schema_properties  # noqa: E402
from simul.usd.editing import USDEditError  # noqa: E402
from simul.usd.mesh_ops import get_mesh_statistics  # noqa: E402
from simul.usd.reader import describe_prim  # noqa: E402
from simul.usd.search import list_subtree  # noqa: E402


def _build(directory: Path) -> Path:
    stage = Usd.Stage.CreateNew(str(directory / "scene.usda"))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdShade.Material.Define(stage, "/World/Looks/Steel")
    UsdGeom.Cube.Define(stage, "/World/Box")
    UsdGeom.Scope.Define(stage, "/World/Group")
    rig = UsdGeom.Xform.Define(stage, "/World/Rig").GetPrim()
    rig.CreateAttribute("camera", Sdf.ValueTypeNames.String).Set("front")
    rig.CreateAttribute("skel:skeleton", Sdf.ValueTypeNames.String).Set("/World/Skel")
    UsdShade.Shader.Define(stage, "/World/Looks/Steel/Surface")
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/Joint")
    joint.CreateBody0Rel().SetTargets(["/World/Box"])
    joint.CreateUpperLimitAttr(90.0)
    UsdPhysics.FixedJoint.Define(stage, "/World/Fixed")
    free = UsdPhysics.RevoluteJoint.Define(stage, "/World/Free")
    free.CreateLowerLimitAttr(float("-inf"))
    free.CreateUpperLimitAttr(float("inf"))
    stage.GetRootLayer().Save()
    return directory / "scene.usda"


def _build_malformed(directory: Path) -> Path:
    """A file whose relationships are authored as attributes, as the old edit tools wrote them."""
    stage = Usd.Stage.CreateNew(str(directory / "malformed.usda"))
    world = UsdGeom.Xform.Define(stage, "/W")
    stage.SetDefaultPrim(world.GetPrim())
    UsdShade.Material.Define(stage, "/W/Looks/Mat")
    string = Sdf.ValueTypeNames.String
    box = UsdGeom.Cube.Define(stage, "/W/Box").GetPrim()
    box.CreateAttribute("material:binding", string).Set("/W/Looks/Mat")
    group = UsdGeom.Xform.Define(stage, "/W/Group").GetPrim()
    group.CreateAttribute("material:binding", string).Set("/W/Looks/Mat")
    UsdGeom.Cube.Define(stage, "/W/Group/Child")
    mesh = UsdGeom.Mesh.Define(stage, "/W/Mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    mesh.GetPrim().CreateAttribute("material:binding:physics", string).Set("/W/Looks/Mat")
    subset = UsdGeom.Subset.Define(stage, "/W/Mesh/Sub")
    subset.CreateElementTypeAttr("face")
    subset.CreateFamilyNameAttr("materialBind")
    subset.CreateIndicesAttr([0])
    subset.GetPrim().CreateAttribute("material:binding", string).Set("/W/Looks/Mat")
    joint = stage.DefinePrim("/W/Joint", "PhysicsRevoluteJoint")
    joint.CreateAttribute("physics:body0", string).Set("/W/Box")
    joint.CreateRelationship("physics:body1").SetTargets(["/W/Box"])
    stage.GetRootLayer().Save()
    return directory / "malformed.usda"


def _build_clips(directory: Path) -> Path:
    for index, offset in enumerate((0.0, 100.0)):
        clip = Usd.Stage.CreateNew(str(directory / f"clip_{index}.usda"))
        op = UsdGeom.Xform.Define(clip, "/Model").AddTranslateOp()
        for frame in range(11):
            op.Set(Gf.Vec3d(offset + frame, 0, 0), float(frame))
        clip.GetRootLayer().Save()
    stage = Usd.Stage.CreateNew(str(directory / "clips.usda"))
    model = UsdGeom.Xform.Define(stage, "/Model")
    stage.SetDefaultPrim(model.GetPrim())
    model.AddTranslateOp()
    model.GetPrim().CreateAttribute("custom:still", Sdf.ValueTypeNames.Double).Set(1.0)
    clips = Usd.ClipsAPI(model.GetPrim())
    clips.SetClipAssetPaths([Sdf.AssetPath("./clip_0.usda"), Sdf.AssetPath("./clip_1.usda")])
    clips.SetClipPrimPath("/Model")
    clips.SetClipActive([(100.0, 0), (110.0, 1)])
    clips.SetClipTimes([(100.0, 0.0), (110.0, 10.0), (110.0, 0.0), (120.0, 10.0)])
    stage.GetRootLayer().Save()
    return directory / "clips.usda"


@pytest.fixture
def session() -> Iterator[HeadlessUSDSession]:
    usd_session = HeadlessUSDSession()
    yield usd_session
    usd_session.cleanup()


@pytest.fixture
def scene(tmp_path: Path, session: HeadlessUSDSession) -> Dict[str, Any]:
    stage_id = session.load_stage(str(_build(tmp_path)))
    stage = session.get_stage(stage_id)
    return {"id": stage_id, "stage": stage, "layer": stage.GetRootLayer()}


def _refusal(call: Any, *args: Any, **kwargs: Any) -> USDEditError:
    with pytest.raises(USDEditError) as caught:
        call(*args, **kwargs)
    return caught.value


class TestRelationshipNamesAreRefused:
    """B1: a relationship is never authored as an attribute."""

    @pytest.mark.parametrize(
        "name, value, declared_by",
        [
            ("material:binding", "/World/Looks/Steel", "material:binding namespace"),
            ("material:binding:physics", "/World/Looks/Steel", "material:binding namespace"),
            ("material:binding:collection:parts", "/World/Looks/Steel", "material:binding namespace"),
            ("physics:body0", "/World/Box", "PhysicsJoint"),
            ("skel:skeleton", "/World/Skel", "SkelBindingAPI"),
            ("physics:filteredPairs", ["/World/Box"], "PhysicsFilteredPairsAPI"),
            ("collection:parts:includes", ["/World/Box"], "CollectionAPI:parts"),
        ],
    )
    def test_update_refuses_and_writes_nothing(self, session, scene, name, value, declared_by) -> None:
        before = scene["layer"].ExportToString()

        error = _refusal(session.update_prim_attributes, scene["id"], "/World/Box", {name: value})

        assert error.error_type == "UnsupportedOperation"
        assert error.details["code"] == editing.RELATIONSHIP_NOT_EDITABLE
        (problem,) = error.details["problems"]
        assert problem["attribute"] == name
        assert problem["code"] == editing.RELATIONSHIP_NOT_EDITABLE
        assert declared_by in problem["reason"]
        assert "relationships are not editable here" in problem["reason"]
        assert scene["layer"].ExportToString() == before
        box = scene["stage"].GetPrimAtPath("/World/Box")
        assert not box.GetAttribute(name)
        assert UsdShade.MaterialBindingAPI(box).ComputeBoundMaterial()[0].GetPrim().IsValid() is False

    def test_create_prim_refuses_and_creates_nothing(self, session, scene) -> None:
        error = _refusal(
            session.create_prim,
            scene["id"],
            "/World/Bound",
            "Cube",
            {"material:binding": "/World/Looks/Steel"},
        )

        assert error.details["code"] == editing.RELATIONSHIP_NOT_EDITABLE
        assert not scene["stage"].GetPrimAtPath("/World/Bound")

    def test_mixed_with_other_problems_is_a_validation_error_with_per_problem_codes(self, session, scene) -> None:
        error = _refusal(
            session.update_prim_attributes,
            scene["id"],
            "/World/Box",
            {"material:binding": "/World/Looks/Steel", "size": "big"},
        )

        assert error.error_type == "ValidationError"
        assert error.details["code"] == editing.INVALID_ATTRIBUTE
        codes = {problem["attribute"]: problem.get("code") for problem in error.details["problems"]}
        assert codes == {"material:binding": editing.RELATIONSHIP_NOT_EDITABLE, "size": None}

    def test_relationship_the_prim_has_is_still_retargeted(self, session, scene) -> None:
        result = session.update_prim_attributes(scene["id"], "/World/Joint", {"physics:body1": "/World/Box"})

        assert result["changed"] == ["physics:body1"]
        rel = scene["stage"].GetPrimAtPath("/World/Joint").GetRelationship("physics:body1")
        assert rel.GetTargets() == [Sdf.Path("/World/Box")]

    def test_relationship_authored_as_an_attribute_is_refused(self, tmp_path, session) -> None:
        stage_id = session.load_stage(str(_build_malformed(tmp_path)))

        error = _refusal(
            session.update_prim_attributes, stage_id, "/W/Box", {"material:binding": "/W/Looks/Other"}
        )

        (problem,) = error.details["problems"]
        assert "malformed.usda authors as an attribute" in problem["reason"]
        assert "correct it in malformed.usda" in problem["reason"]
        assert session.get_stage(stage_id).GetPrimAtPath("/W/Box").GetAttribute("material:binding").Get() == (
            "/W/Looks/Mat"
        )

    def test_schema_relationship_authored_as_an_attribute_is_refused(self, tmp_path, session) -> None:
        stage_id = session.load_stage(str(_build_malformed(tmp_path)))
        before = session.get_stage(stage_id).GetRootLayer().ExportToString()

        error = _refusal(session.update_prim_attributes, stage_id, "/W/Joint", {"physics:body0": "/W/Box"})

        assert error.details["code"] == editing.RELATIONSHIP_NOT_EDITABLE
        assert "this prim's type, PhysicsRevoluteJoint" in error.details["problems"][0]["reason"]
        assert "correct it in malformed.usda" in error.details["problems"][0]["reason"]
        assert session.get_stage(stage_id).GetRootLayer().ExportToString() == before


class TestReadsSurviveMalformedProperties:
    """B1: a file that already holds such a property is reported, not a failed read."""

    @pytest.fixture
    def stage(self, tmp_path: Path) -> Usd.Stage:
        return Usd.Stage.Open(str(_build_malformed(tmp_path)))

    @pytest.mark.parametrize(
        "path, culprit",
        [
            ("/W/Box", "/W/Box.material:binding"),
            ("/W/Group/Child", "/W/Group.material:binding"),
            ("/W/Mesh/Sub", "/W/Mesh/Sub.material:binding"),
        ],
    )
    def test_get_prim_info_reports_the_unreadable_binding(self, stage, path, culprit) -> None:
        info = describe_prim(stage.GetPrimAtPath(path))

        assert info["material_bindings"] == []
        (warning,) = info["warnings"]
        assert warning["code"] == "UnreadableProperty"
        assert warning["severity"] == "warning"
        assert f"{culprit} is not a relationship" in warning["message"]
        assert "pxrInternal" not in warning["message"] and "SDF_FORMAT_ARGS" not in warning["message"]
        assert warning["location"]["prim_path"] == path
        assert warning["location"]["layer"].endswith("malformed.usda")
        assert "SDF_FORMAT_ARGS" not in warning["location"]["layer"]

    def test_own_binding_survives_a_malformed_ancestor(self, stage) -> None:
        own = UsdGeom.Cube.Define(stage, "/W/Group/Bound").GetPrim()
        UsdShade.MaterialBindingAPI.Apply(own).Bind(UsdShade.Material(stage.GetPrimAtPath("/W/Looks/Mat")))
        UsdGeom.Cube.Define(stage, "/W/Group/Bound/Inner")

        info = describe_prim(own)
        inner = describe_prim(stage.GetPrimAtPath("/W/Group/Bound/Inner"))

        assert info["material_bindings"] == inner["material_bindings"] == ["/W/Looks/Mat"]
        assert "/W/Group.material:binding is not a relationship" in info["warnings"][0]["message"]
        assert "the nearest binding pxr can read, /W/Looks/Mat, is reported" in info["warnings"][0]["message"]

    def test_get_prim_info_leaves_out_a_relationship_authored_as_an_attribute(self, stage) -> None:
        info = describe_prim(stage.GetPrimAtPath("/W/Joint"))

        assert "physics:body0" not in info["relationships"]
        assert info["relationships"]["physics:body1"] == {"targets": ["/W/Box"], "authored": True}
        (warning,) = info["warnings"]
        assert warning["location"]["property_path"] == "/W/Joint.physics:body0"

    def test_get_mesh_info_reports_subset_and_physics_bindings(self, stage) -> None:
        info = get_mesh_statistics(stage.GetPrimAtPath("/W/Mesh"))

        assert info.materials == [] and info.physics_material is None
        assert info.subsets[0]["material"] is None
        messages = [warning["message"] for warning in info.warnings]
        assert any("/W/Mesh/Sub.material:binding is not a relationship" in m for m in messages)
        assert any("physics material binding" in m and "material:binding:physics" in m for m in messages)
        assert info.face_count == 1

    def test_subtree_with_properties_reports_per_prim(self, stage) -> None:
        page = list_subtree(stage, stage.GetPrimAtPath("/W"), include_properties=True)

        records = {record["path"]: record for record in page["prims"]}
        assert records["/W/Joint"]["relationships"] == {"physics:body1": ["/W/Box"]}
        assert records["/W/Joint"]["warnings"][0]["location"]["property_path"] == "/W/Joint.physics:body0"
        assert "warnings" not in records["/W/Box"]
        assert page["total"] == 9


class TestSchemaTypedAttributes:
    """M3: an attribute another schema declares takes that schema's type, with a warning."""

    def test_api_attribute_takes_the_schema_type_and_warns(self, session, scene) -> None:
        result = session.update_prim_attributes(scene["id"], "/World/Group", {"physics:mass": 5})

        assert result["created_attributes"] == {"physics:mass": "float"}
        (warning,) = result["warnings"]
        assert warning["code"] == editing.UNAPPLIED_SCHEMA_ATTRIBUTE
        assert "PhysicsMassAPI" in warning["message"]
        assert warning["location"]["property_path"] == "/World/Group.physics:mass"
        prim = scene["stage"].GetPrimAtPath("/World/Group")
        attr = prim.GetAttribute("physics:mass")
        assert attr.GetTypeName() == Sdf.ValueTypeNames.Float and not attr.IsCustom()
        prim.ApplyAPI(UsdPhysics.MassAPI)
        assert UsdPhysics.MassAPI(prim).GetMassAttr().Get() == 5.0

    def test_multiple_apply_instance_takes_the_template_type(self, session, scene) -> None:
        result = session.update_prim_attributes(
            scene["id"],
            "/World/Fixed",
            {"drive:angular:physics:stiffness": 10, "drive:angular:physics:type": "force"},
        )

        assert result["created_attributes"] == {
            "drive:angular:physics:stiffness": "float",
            "drive:angular:physics:type": "token",
        }
        assert all("PhysicsDriveAPI:angular" in warning["message"] for warning in result["warnings"])
        prim = scene["stage"].GetPrimAtPath("/World/Fixed")
        assert prim.GetAttribute("drive:angular:physics:type").GetVariability() == Sdf.VariabilityUniform
        prim.ApplyAPI(UsdPhysics.DriveAPI, "angular")
        assert UsdPhysics.DriveAPI(prim, "angular").GetStiffnessAttr().Get() == 10.0

    def test_schema_tokens_and_variability_are_checked(self, session, scene) -> None:
        bad_token = _refusal(
            session.update_prim_attributes, scene["id"], "/World/Fixed", {"drive:rotX:physics:type": "bogus"}
        )
        sampled = _refusal(
            session.update_prim_attributes,
            scene["id"],
            "/World/Fixed",
            {"drive:rotX:physics:type": "force"},
            time_code=1,
        )

        assert "allowed: force, acceleration" in bad_token.details["problems"][0]["reason"]
        assert "uniform" in sampled.details["problems"][0]["reason"]

    def test_attribute_types_must_match_the_schema(self, session, scene) -> None:
        error = _refusal(
            session.update_prim_attributes,
            scene["id"],
            "/World/Group",
            {"physics:mass": 5},
            attribute_types={"physics:mass": "int"},
        )

        assert "PhysicsMassAPI" in error.details["problems"][0]["reason"]
        assert not scene["stage"].GetPrimAtPath("/World/Group").GetAttribute("physics:mass")

    def test_api_declarations_win_over_prim_types(self, session, scene) -> None:
        # scales: float3[] in ParticleFieldScaleAttributeAPI, float3[] / half3[] in prim types.
        result = session.update_prim_attributes(scene["id"], "/World/Group", {"scales": [[1, 1, 1]]})

        assert result["created_attributes"] == {"scales": "float3[]"}
        assert "ParticleFieldScaleAttributeAPI" in result["warnings"][0]["message"]

    def test_conflicting_types_need_attribute_types_and_warn_for_the_chosen_one(
        self, session, scene, monkeypatch
    ) -> None:
        declared = schema_properties.SchemaProperty(
            "custom:pair",
            (
                schema_properties.Declaration("SomeAPI", True, False, Sdf.ValueTypeNames.Float3Array),
                schema_properties.Declaration("SomePrimType", False, False, Sdf.ValueTypeNames.Half3Array),
            ),
        )
        monkeypatch.setattr(schema_properties, "lookup", lambda name: declared if name == "custom:pair" else None)

        error = _refusal(session.update_prim_attributes, scene["id"], "/World/Group", {"custom:pair": [[1, 1, 1]]})
        result = session.update_prim_attributes(
            scene["id"], "/World/Group", {"custom:pair": [[1, 1, 1]]}, attribute_types={"custom:pair": "half3[]"}
        )

        assert "attribute_types" in error.details["problems"][0]["reason"]
        assert result["created_attributes"] == {"custom:pair": "half3[]"}
        (warning,) = result["warnings"]
        assert "SomePrimType" in warning["message"] and "SomeAPI" not in warning["message"]

    @pytest.mark.parametrize(
        "attributes, attribute_types, created",
        [
            ({"axis": [0.0, 0.0, 1.0]}, {}, {"axis": "float3"}),
            ({"axis": [0, 0, 1]}, {"axis": "double3"}, {"axis": "double3"}),
            ({"type": "gripper"}, {}, {"type": "string"}),
            ({"size": "large"}, {}, {"size": "string"}),
            ({"camera": "wrist_cam"}, {}, {"camera": "string"}),
            ({"geometry": "arm"}, {}, {"geometry": "string"}),
        ],
    )
    def test_plain_names_stay_custom_attributes(self, session, scene, attributes, attribute_types, created) -> None:
        result = session.update_prim_attributes(
            scene["id"], "/World/Group", attributes, attribute_types=attribute_types
        )

        assert result["created_attributes"] == created
        assert result["warnings"] == []

    def test_existing_custom_attributes_named_like_relationships_stay_editable(self, session, scene) -> None:
        result = session.update_prim_attributes(
            scene["id"], "/World/Rig", {"camera": "back", "skel:skeleton": "/World/Other"}
        )

        assert result["changed"] == ["camera", "skel:skeleton"]
        rig = scene["stage"].GetPrimAtPath("/World/Rig")
        assert rig.GetAttribute("camera").Get() == "back"
        assert rig.GetAttribute("skel:skeleton").Get() == "/World/Other"

    def test_primvars_take_the_prim_type_type_without_a_warning(self, session, scene) -> None:
        result = session.update_prim_attributes(
            scene["id"], "/World/Group", {"primvars:displayColor": [[1, 0, 0]]}
        )

        assert result["created_attributes"] == {"primvars:displayColor": "color3f[]"}
        assert result["warnings"] == []

    def test_shader_inputs_are_not_typed_by_light_schemas(self, session, scene) -> None:
        result = session.update_prim_attributes(
            scene["id"], "/World/Looks/Steel/Surface", {"inputs:radius": 2}
        )

        assert result["created_attributes"] == {"inputs:radius": "int"}
        assert result["warnings"] == []

    def test_prim_type_attribute_warns_without_an_api(self, session, scene) -> None:
        result = session.update_prim_attributes(scene["id"], "/World/Group", {"physics:axis": "Y"})

        assert result["created_attributes"] == {"physics:axis": "token"}
        assert "prim type" in result["warnings"][0]["message"]

    def test_create_prim_types_schema_attributes_too(self, session, scene) -> None:
        result = session.create_prim(scene["id"], "/World/Heavy", "Cube", {"physics:mass": 2, "size": 3})

        assert result["created_attributes"] == {"physics:mass": "float"}
        assert [warning["code"] for warning in result["warnings"]] == [editing.UNAPPLIED_SCHEMA_ATTRIBUTE]

    def test_custom_attributes_are_still_inferred_without_warnings(self, session, scene) -> None:
        result = session.update_prim_attributes(scene["id"], "/World/Group", {"custom:count": 5})

        assert result["created_attributes"] == {"custom:count": "int"}
        assert result["warnings"] == []

    def test_lookup_reports_declarations(self) -> None:
        mass = schema_properties.lookup("physics:mass")
        drive = schema_properties.lookup("drive:linear:physics:maxForce")
        includes = schema_properties.lookup("collection:things:includes")

        assert mass.type_names == (Sdf.ValueTypeNames.Float,) and mass.api_schemas == ("PhysicsMassAPI",)
        assert drive.api_schemas == ("PhysicsDriveAPI:linear",)
        assert includes.relationship and includes.api_schemas == ("CollectionAPI:things",)
        assert schema_properties.lookup("custom:anything") is None


class TestValueClips:
    """B2: a time sample over value clips is refused; the clip animation survives."""

    @pytest.fixture
    def clipped(self, tmp_path: Path, session: HeadlessUSDSession) -> Dict[str, Any]:
        stage_id = session.load_stage(str(_build_clips(tmp_path)))
        return {"id": stage_id, "stage": session.get_stage(stage_id)}

    @pytest.mark.parametrize("time_code", [100, 105, 115, 50])
    def test_sample_over_clips_is_refused(self, session, clipped, time_code) -> None:
        attr = clipped["stage"].GetPrimAtPath("/Model").GetAttribute("xformOp:translate")
        before = [attr.Get(t) for t in (100, 105, 115)]

        error = _refusal(
            session.update_prim_attributes,
            clipped["id"],
            "/Model",
            {"xformOp:translate": [9, 9, 9]},
            time_code=time_code,
        )

        assert "value clips (authored on /Model)" in error.details["problems"][0]["reason"]
        assert [attr.Get(t) for t in (100, 105, 115)] == before == [
            Gf.Vec3d(0, 0, 0),
            Gf.Vec3d(5, 0, 0),
            Gf.Vec3d(105, 0, 0),
        ]

    def test_default_over_clips_names_the_clips(self, session, clipped) -> None:
        error = _refusal(session.update_prim_attributes, clipped["id"], "/Model", {"xformOp:translate": [9, 9, 9]})

        assert "from value clips on /Model" in error.details["problems"][0]["reason"]

    def test_attribute_outside_the_clips_still_takes_samples(self, session, clipped) -> None:
        session.update_prim_attributes(clipped["id"], "/Model", {"custom:still": 2.0}, time_code=105)

        assert clipped["stage"].GetPrimAtPath("/Model").GetAttribute("custom:still").Get(105) == 2.0


class TestNonFiniteAndOverflow:
    """M10: non-finite strings round-trip; overflow is refused, not stored as infinity."""

    @pytest.mark.parametrize(
        "type_name, value, expected",
        [
            ("float", "Infinity", math.inf),
            ("double", "-Infinity", -math.inf),
            ("half", "Infinity", math.inf),
            ("float[]", [1, "-Infinity"], [1.0, -math.inf]),
            ("float3", ["Infinity", 0, -1], [math.inf, 0.0, -1.0]),
            ("double3", [0, "-Infinity", 2], [0.0, -math.inf, 2.0]),
            ("quatf", [1, 0, 0, "Infinity"], [1.0, 0.0, 0.0, math.inf]),
            ("float", 3.4028234663852886e38, 3.4028234663852886e38),
            ("double", 1e39, 1e39),
            ("half", 65504, 65504.0),
        ],
    )
    def test_accepted(self, type_name, value, expected) -> None:
        converted = coerce.coerce(value, coerce.parse_type_name(type_name))
        if isinstance(converted, Gf.Quatf):
            converted = [converted.GetReal(), *converted.GetImaginary()]

        assert list(converted) == expected if isinstance(expected, list) else converted == expected

    @pytest.mark.parametrize("value", ["Infinity", "-Infinity", "NaN"])
    def test_time_codes_stay_finite(self, value) -> None:
        with pytest.raises(coerce.ValueProblem, match="finite number"):
            coerce.coerce(value, Sdf.ValueTypeNames.TimeCode)

        assert coerce.coerce(12, Sdf.ValueTypeNames.TimeCode) == Sdf.TimeCode(12)

    def test_nan_is_accepted(self) -> None:
        assert math.isnan(coerce.coerce("NaN", Sdf.ValueTypeNames.Float))
        assert math.isnan(coerce.coerce([0, "NaN", 0], Sdf.ValueTypeNames.Float3)[1])

    @pytest.mark.parametrize(
        "type_name, value, fragment",
        [
            ("float", 1e39, "out of range for float"),
            ("float", -1e39, "out of range for float"),
            ("float3", [0, 1e39, 0], "out of range for float3"),
            ("half", 70000, "out of range for half"),
            ("matrix4d", [[10**400 if i == j == 0 else 0 for j in range(4)] for i in range(4)], "too large"),
            ("double", 10**400, "too large"),
            ("int3", [0, 2**40, 0], "out of range for int3"),
            ("float", "inf", 'exactly "Infinity"'),
            ("float", "big", "float takes a number, not the string 'big'"),
        ],
    )
    def test_refused(self, type_name, value, fragment) -> None:
        with pytest.raises(coerce.ValueProblem) as caught:
            coerce.coerce(value, coerce.parse_type_name(type_name))

        assert fragment in str(caught.value)

    def test_unlimited_joint_limits_round_trip(self, session, scene) -> None:
        free = describe_prim(scene["stage"].GetPrimAtPath("/World/Free"))["attributes"]
        lower, upper = free["physics:lowerLimit"]["value"], free["physics:upperLimit"]["value"]

        session.update_prim_attributes(
            scene["id"], "/World/Joint", {"physics:lowerLimit": lower, "physics:upperLimit": upper}
        )

        joint = scene["stage"].GetPrimAtPath("/World/Joint")
        assert (lower, upper) == ("-Infinity", "Infinity")
        assert joint.GetAttribute("physics:lowerLimit").Get() == -math.inf
        assert joint.GetAttribute("physics:upperLimit").Get() == math.inf

    def test_overflow_is_refused_and_writes_nothing(self, session, scene) -> None:
        error = _refusal(session.update_prim_attributes, scene["id"], "/World/Joint", {"physics:upperLimit": 1e39})

        assert "out of range for float" in error.details["problems"][0]["reason"]
        assert scene["stage"].GetPrimAtPath("/World/Joint").GetAttribute("physics:upperLimit").Get() == 90.0
