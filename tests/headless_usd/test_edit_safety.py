"""Headless USD edits are all-or-nothing, refuse what they cannot do, and say why.

Every test runs against a real ``pxr`` stage built on disk: a scene whose
``/World`` is defined in a sublayer and overridden in the root layer, with a
referenced robot, an instanceable copy of it and a few local prims. The
check that matters for atomicity is the root layer's text before and after a
failed call: it must not change by a byte.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator

import pytest

pytest.importorskip("pxr", reason="pxr library not available")

from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

from simul.adapters.headless_usd import HeadlessUSDSession  # noqa: E402
from simul.usd import editing  # noqa: E402
from simul.usd.editing import USDEditError  # noqa: E402


def _build_robot(path: Path) -> None:
    stage = Usd.Stage.CreateNew(str(path))
    robot = UsdGeom.Xform.Define(stage, "/Robot")
    stage.SetDefaultPrim(robot.GetPrim())
    base = UsdGeom.Xform.Define(stage, "/Robot/base_link")
    link1 = UsdGeom.Xform.Define(stage, "/Robot/link1")
    link1.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.2))
    UsdGeom.Cube.Define(stage, "/Robot/link1/visuals").GetSizeAttr().Set(0.5)
    UsdGeom.Scope.Define(stage, "/Robot/joints")
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/Robot/joints/shoulder")
    joint.CreateAxisAttr(UsdPhysics.Tokens.z)
    joint.CreateUpperLimitAttr(90.0)
    joint.CreateBody0Rel().SetTargets([base.GetPath()])
    joint.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    stage.GetRootLayer().Save()


def _build_scene(directory: Path) -> Path:
    """Write robot.usda, sublayer.usda and scene.usda; return scene.usda."""
    _build_robot(directory / "robot.usda")

    sub = Usd.Stage.CreateNew(str(directory / "sublayer.usda"))
    UsdGeom.Xform.Define(sub, "/World")
    UsdGeom.Xform.Define(sub, "/World/SharedDef")
    UsdGeom.Xform.Define(sub, "/World/FromSublayer")
    sub.GetRootLayer().Save()

    scene_path = directory / "scene.usda"
    stage = Usd.Stage.CreateNew(str(scene_path))
    stage.GetRootLayer().subLayerPaths.append("./sublayer.usda")
    world = stage.OverridePrim("/World")
    stage.SetDefaultPrim(world)
    shared = stage.OverridePrim("/World/SharedDef")
    shared.CreateAttribute("custom:note", Sdf.ValueTypeNames.String).Set("root override")
    UsdGeom.Cube.Define(stage, "/World/Box").GetSizeAttr().Set(2.0)
    local = UsdGeom.Xform.Define(stage, "/World/Local")
    UsdGeom.Cube.Define(stage, "/World/Local/child")
    local.GetPrim().CreateRelationship("custom:inner").SetTargets(["/World/Local/child"])
    UsdGeom.Xform.Define(stage, "/World/Last")
    ref = UsdGeom.Xform.Define(stage, "/World/Robot_ref")
    ref.GetPrim().GetReferences().AddReference("./robot.usda")
    inst = UsdGeom.Xform.Define(stage, "/World/Robot_inst")
    inst.GetPrim().GetReferences().AddReference("./robot.usda")
    inst.GetPrim().SetInstanceable(True)
    stage.GetRootLayer().Save()
    return scene_path


@pytest.fixture
def session() -> Iterator[HeadlessUSDSession]:
    usd_session = HeadlessUSDSession()
    yield usd_session
    usd_session.cleanup()


@pytest.fixture
def scene(tmp_path: Path, session: HeadlessUSDSession) -> Dict[str, Any]:
    """The loaded scene: its stage id, stage and root layer."""
    stage_id = session.load_stage(_build_scene(tmp_path))
    stage = session.get_stage(stage_id)
    return {"id": stage_id, "stage": stage, "layer": stage.GetRootLayer()}


def _refusal(call, *args: Any, **kwargs: Any) -> USDEditError:
    with pytest.raises(USDEditError) as caught:
        call(*args, **kwargs)
    return caught.value


def _assert_unchanged(scene: Dict[str, Any], before: str) -> None:
    assert scene["layer"].ExportToString() == before


class TestAllOrNothing:
    """A call that reports failure leaves the edit target exactly as it was."""

    def test_create_with_a_bad_attribute_creates_nothing(self, session, scene) -> None:
        before = scene["layer"].ExportToString()

        error = _refusal(
            session.create_prim,
            scene["id"],
            "/World/Partial",
            "Cube",
            {"size": 2.0, "bad": {"nested": 1}},
        )

        assert (error.error_type, error.details["code"]) == ("ValidationError", "InvalidAttribute")
        assert error.details["problems"] == [
            {"attribute": "bad", "reason": "a JSON object is not an attribute value"}
        ]
        assert not scene["stage"].GetPrimAtPath("/World/Partial")
        _assert_unchanged(scene, before)

    def test_update_with_one_bad_attribute_changes_nothing(self, session, scene) -> None:
        before = scene["layer"].ExportToString()

        error = _refusal(
            session.update_prim_attributes,
            scene["id"],
            "/World/Box",
            {"size": 7.0, "custom:a": 1, "zz": {"x": 1}},
        )

        assert (error.error_type, error.details["code"]) == ("ValidationError", "InvalidAttribute")
        assert [problem["attribute"] for problem in error.details["problems"]] == ["zz"]
        box = scene["stage"].GetPrimAtPath("/World/Box")
        assert box.GetAttribute("size").Get() == 2.0
        assert not box.GetAttribute("custom:a")
        _assert_unchanged(scene, before)

    def test_every_rejected_attribute_is_reported(self, session, scene) -> None:
        error = _refusal(
            session.update_prim_attributes,
            scene["id"],
            "/World/Box",
            {"size": "big", "custom:none": None, "custom:ok": 1, "bad name": 2},
        )

        reasons = {problem["attribute"]: problem["reason"] for problem in error.details["problems"]}
        assert set(reasons) == {"size", "custom:none", "bad name"}
        assert reasons["size"] == "a string cannot be stored in a double attribute"
        assert "null is not a value" in reasons["custom:none"]
        assert "not a valid attribute name" in reasons["bad name"]
        assert "'size'" in error.message and "nothing was changed" in error.message

    def test_delete_of_a_sublayer_prim_keeps_the_root_override(self, session, scene) -> None:
        before = scene["layer"].ExportToString()

        error = _refusal(session.delete_prim, scene["id"], "/World/SharedDef")

        assert error.error_type == "UnsupportedOperation"
        assert error.details["surviving_opinions"] == [{"layer": "sublayer.usda", "via": "a sublayer"}]
        assert '{"active": false}' in error.message
        shared = scene["stage"].GetPrimAtPath("/World/SharedDef")
        assert shared.GetAttribute("custom:note").Get() == "root override"
        _assert_unchanged(scene, before)

    def test_delete_inside_a_reference_is_refused(self, session, scene) -> None:
        before = scene["layer"].ExportToString()

        error = _refusal(session.delete_prim, scene["id"], "/World/Robot_ref/link1")

        assert error.error_type == "UnsupportedOperation"
        assert error.details["surviving_opinions"] == [
            {"layer": "robot.usda", "via": "the reference on /World/Robot_ref"}
        ]
        _assert_unchanged(scene, before)

    def test_delete_of_a_referencing_prim_removes_it(self, session, scene) -> None:
        """The reference is authored on the prim itself, so removing the prim removes it."""
        result = session.delete_prim(scene["id"], "/World/Robot_ref")

        assert result["removed_prims"] > 1
        assert not scene["stage"].GetPrimAtPath("/World/Robot_ref")


class _FailOn:
    """Make ``editing._apply_write`` fail on the write that sets ``name``."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
        real = editing._apply_write

        def apply(prim: Usd.Prim, write: editing.PlannedWrite) -> None:
            if write.name == name:
                raise editing._WriteFailure(write.name, "injected failure")
            real(prim, write)

        monkeypatch.setattr(editing, "_apply_write", apply)


class TestRollback:
    """When a write fails after others succeeded, the journal restores the layer."""

    def test_create_with_new_ancestors(self, session, scene, monkeypatch) -> None:
        before = scene["layer"].ExportToString()
        _FailOn(monkeypatch, "custom:b")

        error = _refusal(
            session.create_prim,
            scene["id"],
            "/World/New/Deep/Box",
            "Cube",
            {"size": 3.0, "custom:b": 1, "kind": "component"},
        )

        assert (error.error_type, error.details["code"]) == ("EngineError", "EditFailed")
        assert error.details["rolled_back"] is True
        assert error.details["attribute"] == "custom:b"
        assert not scene["stage"].GetPrimAtPath("/World/New")
        _assert_unchanged(scene, before)

    def test_update_of_a_root_layer_prim(self, session, scene, monkeypatch) -> None:
        before = scene["layer"].ExportToString()
        _FailOn(monkeypatch, "active")

        error = _refusal(
            session.update_prim_attributes,
            scene["id"],
            "/World/Box",
            {"size": 9.0, "custom:c": 2.5, "kind": "component", "active": False},
        )

        assert error.details["attribute"] == "active"
        assert scene["stage"].GetPrimAtPath("/World/Box").GetAttribute("size").Get() == 2.0
        _assert_unchanged(scene, before)

    def test_update_of_a_referenced_prim_drops_the_new_override(self, session, scene, monkeypatch) -> None:
        """/World/Robot_ref/link1 has no spec in scene.usda until the edit authors an over."""
        before = scene["layer"].ExportToString()
        _FailOn(monkeypatch, "custom:second")

        _refusal(
            session.update_prim_attributes,
            scene["id"],
            "/World/Robot_ref/link1",
            {"custom:first": 1, "custom:second": 2},
        )

        assert scene["layer"].GetPrimAtPath("/World/Robot_ref/link1") is None
        _assert_unchanged(scene, before)

    def test_redefine_restores_the_original_type(self, session, scene, monkeypatch) -> None:
        before = scene["layer"].ExportToString()
        _FailOn(monkeypatch, "radius")

        _refusal(
            session.create_prim,
            scene["id"],
            "/World/Box",
            "Sphere",
            {"radius": 4.0},
            redefine_existing=True,
        )

        assert scene["stage"].GetPrimAtPath("/World/Box").GetTypeName() == "Cube"
        _assert_unchanged(scene, before)

    def test_delete_restores_the_subtree_in_place(self, session, scene, monkeypatch) -> None:
        """If the prim survived its removal, the subtree comes back where it was.

        /World/Local is fully in the root layer, so the delete succeeds; force
        the post-check to fail and the journal must put it back between its
        siblings with its internal relationship target intact.
        """
        before = scene["layer"].ExportToString()
        monkeypatch.setattr(editing, "_still_composed", lambda stage, path: True)

        error = _refusal(session.delete_prim, scene["id"], "/World/Local")

        assert (error.error_type, error.details["code"]) == ("EngineError", "EditFailed")
        assert error.details["rolled_back"] is True
        _assert_unchanged(scene, before)
        assert scene["stage"].GetPrimAtPath("/World/Local").GetRelationship("custom:inner").GetTargets() == [
            Sdf.Path("/World/Local/child")
        ]


class TestPaths:
    """Paths must be absolute prim paths naming the right prim; errors say what is wrong."""

    @pytest.mark.parametrize(
        "call, args",
        [
            ("create_prim", ("World/Rel", "Xform")),
            ("update_prim_attributes", ("World/Box", {"size": 1.0})),
            ("delete_prim", ("World",)),
        ],
    )
    def test_relative_paths_are_refused(self, session, scene, call, args) -> None:
        before = scene["layer"].ExportToString()

        error = _refusal(getattr(session, call), scene["id"], *args)

        assert (error.error_type, error.details["code"]) == ("ValidationError", "InvalidPrimPath")
        assert "relative" in error.message
        assert error.details["suggestion"] == "/" + args[0]
        _assert_unchanged(scene, before)

    def test_invalid_prim_name_is_named(self, session, scene) -> None:
        error = _refusal(session.create_prim, scene["id"], "/World/bad name", "Xform")

        assert (error.error_type, error.details["code"]) == ("ValidationError", "InvalidPrimPath")
        assert error.details["element"] == "bad name"
        assert "absolute" not in error.message

    def test_property_path_points_at_the_prim(self, session, scene) -> None:
        error = _refusal(session.update_prim_attributes, scene["id"], "/World/Box.size", {"size": 1.0})

        assert (error.error_type, error.details["code"]) == ("ValidationError", "InvalidPrimPath")
        assert error.details["suggestion"] == "/World/Box"

    def test_trailing_slash(self, session, scene) -> None:
        error = _refusal(session.delete_prim, scene["id"], "/World/Box/")

        assert error.details["suggestion"] == "/World/Box"

    def test_deleting_a_missing_prim_fails(self, session, scene) -> None:
        before = scene["layer"].ExportToString()

        error = _refusal(session.delete_prim, scene["id"], "/World/DoesNotExist")

        assert error.error_type == "NotFoundError"
        assert error.details["nearest_existing_prim"] == "/World"
        _assert_unchanged(scene, before)

    def test_unknown_stage(self, session) -> None:
        error = _refusal(session.update_prim_attributes, "nope_123", "/World", {"a": 1})

        assert error.error_type == "NotFoundError"
        assert "load_usd_file" in error.message

    def test_pseudo_root_is_never_deleted(self, session, scene) -> None:
        error = _refusal(session.delete_prim, scene["id"], "/", allow_root_delete=True)

        assert error.error_type == "RefusedOperation"

    def test_default_prim_needs_confirmation(self, tmp_path, session) -> None:
        stage = Usd.Stage.CreateNew(str(tmp_path / "simple.usda"))
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        UsdGeom.Cube.Define(stage, "/World/Box")
        stage.GetRootLayer().Save()
        stage_id = session.load_stage(str(tmp_path / "simple.usda"))

        error = _refusal(session.delete_prim, stage_id, "/World")

        assert error.error_type == "RefusedOperation"
        assert error.details["override"] == "allow_root_delete"
        assert session.get_stage(stage_id).GetPrimAtPath("/World/Box")

        result = session.delete_prim(stage_id, "/World", allow_root_delete=True)

        assert result["removed_prims"] == 2
        assert not session.get_stage(stage_id).GetPrimAtPath("/World")


class TestPrimTypes:
    """create_prim takes only registered concrete types and never re-types silently."""

    def test_typo_is_refused_with_a_suggestion(self, session, scene) -> None:
        error = _refusal(session.create_prim, scene["id"], "/World/Typo", "Cub")

        assert (error.error_type, error.details["code"]) == ("ValidationError", "UnknownPrimType")
        assert error.details["suggestions"][0] == "Cube"
        assert not scene["stage"].GetPrimAtPath("/World/Typo")

    @pytest.mark.parametrize("type_name, phrase", [("Xformable", "abstract"), ("PhysicsRigidBodyAPI", "API schema")])
    def test_non_concrete_types_are_refused(self, session, scene, type_name, phrase) -> None:
        error = _refusal(session.create_prim, scene["id"], "/World/X", type_name)

        assert (error.error_type, error.details["code"]) == ("ValidationError", "UnknownPrimType")
        assert phrase in error.message

    def test_existing_prim_is_not_retyped(self, session, scene) -> None:
        before = scene["layer"].ExportToString()

        error = _refusal(session.create_prim, scene["id"], "/World/Robot_ref", "Cube")

        assert error.error_type == "RefusedOperation"
        assert error.details["override"] == "redefine_existing"
        assert error.details["existing_type"] == "Xform"
        _assert_unchanged(scene, before)

    def test_redefine_existing_retypes_and_keeps_children(self, session, scene) -> None:
        result = session.create_prim(scene["id"], "/World/Robot_ref", "Scope", redefine_existing=True)

        prim = scene["stage"].GetPrimAtPath("/World/Robot_ref")
        assert result["redefined"] is True
        assert prim.GetTypeName() == "Scope"
        assert prim.HasAuthoredReferences()
        assert scene["stage"].GetPrimAtPath("/World/Robot_ref/link1")

    def test_new_prim_gets_schema_typed_attributes(self, session, scene) -> None:
        result = session.create_prim(scene["id"], "/World/NewBox", "Cube", {"size": 3})

        assert result["changed"] == ["size"]
        size = scene["stage"].GetPrimAtPath("/World/NewBox").GetAttribute("size")
        assert size.GetTypeName() == Sdf.ValueTypeNames.Double
        assert size.Get() == 3.0


class TestMetadataAndProperties:
    """active / instanceable / kind are metadata; tokens and relationships are checked."""

    def test_metadata_keys_use_the_metadata_api(self, session, scene) -> None:
        result = session.update_prim_attributes(
            scene["id"], "/World/Box", {"active": False, "instanceable": True, "kind": "assembly"}
        )

        spec = scene["layer"].GetPrimAtPath("/World/Box")
        assert spec.active is False
        assert spec.instanceable is True
        assert spec.kind == Kind.Tokens.assembly
        assert not {"active", "instanceable", "kind"} & set(spec.properties.keys())
        assert not scene["stage"].GetPrimAtPath("/World/Box").IsActive()
        assert result["changed"] == ["active (metadata)", "instanceable (metadata)", "kind (metadata)"]

    @pytest.mark.parametrize(
        "attributes, fragment",
        [
            ({"active": 0}, "takes true or false"),
            ({"instanceable": "yes"}, "takes true or false"),
            ({"kind": "building"}, "registered kind"),
            ({"documentation": "text"}, "is prim metadata, not an attribute"),
            ({"typeName": "Sphere"}, "is prim metadata, not an attribute"),
        ],
    )
    def test_bad_metadata_is_refused(self, session, scene, attributes, fragment) -> None:
        before = scene["layer"].ExportToString()

        error = _refusal(session.update_prim_attributes, scene["id"], "/World/Box", attributes)

        assert fragment in error.details["problems"][0]["reason"]
        _assert_unchanged(scene, before)

    def test_token_outside_allowed_tokens_is_refused(self, session, scene) -> None:
        joint = "/World/Robot_ref/joints/shoulder"
        error = _refusal(session.update_prim_attributes, scene["id"], joint, {"physics:axis": "W"})

        assert error.details["problems"] == [
            {"attribute": "physics:axis", "reason": "'W' is not an allowed token; allowed: X, Y, Z"}
        ]

        session.update_prim_attributes(scene["id"], joint, {"physics:axis": "X"})

        assert scene["stage"].GetPrimAtPath(joint).GetAttribute("physics:axis").Get() == "X"

    def test_relationship_targets(self, session, scene) -> None:
        joint = "/World/Robot_ref/joints/shoulder"

        session.update_prim_attributes(scene["id"], joint, {"physics:body1": "/World/Robot_ref/link1"})

        rel = scene["stage"].GetPrimAtPath(joint).GetRelationship("physics:body1")
        assert rel.GetTargets() == [Sdf.Path("/World/Robot_ref/link1")]

        error = _refusal(session.update_prim_attributes, scene["id"], joint, {"physics:body0": "link1"})
        assert "not an absolute USD path" in error.details["problems"][0]["reason"]


class TestInstanceProxies:
    """Edits below an instanceable prim are refused before USD raises."""

    def test_update_and_delete_of_a_proxy(self, session, scene) -> None:
        for error in (
            _refusal(session.update_prim_attributes, scene["id"], "/World/Robot_inst/link1", {"custom:a": 1}),
            _refusal(session.delete_prim, scene["id"], "/World/Robot_inst/link1"),
        ):
            assert error.error_type == "UnsupportedOperation"
            assert error.details["instance"] == "/World/Robot_inst"

    def test_create_under_an_instance(self, session, scene) -> None:
        error = _refusal(session.create_prim, scene["id"], "/World/Robot_inst/extra", "Xform")

        assert error.error_type == "UnsupportedOperation"
        assert '"instanceable": false' in error.message


def test_no_pxr_internals_leak(session, scene) -> None:
    """Error text names the prim and the reason, never pxr's C++ frames."""
    errors = [
        _refusal(session.create_prim, scene["id"], "World/Rel", "Xform"),
        _refusal(session.create_prim, scene["id"], "/World/bad name", "Xform"),
        _refusal(session.create_prim, scene["id"], "/World/Robot_inst/x", "Xform"),
        _refusal(session.update_prim_attributes, scene["id"], "/World/Box", {"size": [1, 2, 3]}),
        _refusal(
            session.update_prim_attributes,
            scene["id"],
            "/World/Robot_ref/joints/shoulder",
            {"physics:upperLimit": "abc"},
        ),
    ]
    for error in errors:
        assert "pxrInternal" not in error.message
        assert ".cpp" not in error.message
