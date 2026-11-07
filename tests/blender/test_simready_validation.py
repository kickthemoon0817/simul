"""validate_simready_compliance: the SimReady Robotics-Prop rules and Isaac Sim's checks.

Errors flip ``compliant``; warnings do not. The stand-in objects carry world
matrices directly, as validation reads them (the stand-in bpy has no depsgraph,
so every ``matrix_world`` is taken as current).
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pytest

from simul.adapters import blender_runtime
from tests.fakes import FakeMatrix


class Polygons(list):
    """mesh.polygons: ``foreach_get`` of loop_total, area and material_index, as Blender offers it."""

    def foreach_get(self, attribute: str, out: Any) -> None:
        for index, polygon in enumerate(self):
            out[index] = polygon[attribute]


def _mesh(
    name: str,
    corners: Sequence[int] = (4, 4, 4, 4, 4, 4),
    areas: Optional[Sequence[float]] = None,
    uv: bool = True,
    materials: Sequence[Any] = ("opaque_metal",),
    slots: Optional[Sequence[int]] = None,
) -> SimpleNamespace:
    areas = list(areas) if areas is not None else [1.0] * len(corners)
    slots = list(slots) if slots is not None else [0] * len(corners)
    return SimpleNamespace(
        name=name,
        materials=list(materials),
        polygons=Polygons(
            {"loop_total": c, "area": a, "material_index": m}
            for c, a, m in zip(corners, areas, slots)
        ),
        uv_layers=["UVMap"] if uv else [],
    )


class Obj:
    """A Blender object with the matrix, material and custom-property surface validation reads."""

    def __init__(
        self,
        name: str,
        kind: str = "MESH",
        parent: Optional["Obj"] = None,
        data: Optional[Any] = None,
        world: Optional[FakeMatrix] = None,
    ) -> None:
        self.name = name
        self.type = kind
        self.parent = parent
        self.children: List[Obj] = []
        if parent is not None:
            parent.children.append(self)
        self.data = data if data is not None or kind != "MESH" else _mesh(name)
        self.dimensions = (1.0, 1.0, 1.0) if kind == "MESH" else (0.0, 0.0, 0.0)
        self.matrix_world = world or (parent.matrix_world if parent else FakeMatrix.Identity(4))
        self.modifiers: List[Any] = []
        self.props: Dict[str, Any] = {}
        self.mode = "OBJECT"
        self.hide_render = False

    def __setitem__(self, key: str, value: Any) -> None:
        self.props[key] = value

    def __getitem__(self, key: str) -> Any:
        return self.props[key]

    def __delitem__(self, key: str) -> None:
        del self.props[key]

    def keys(self) -> List[str]:
        return list(self.props)


class Objects(list):
    def get(self, name: str) -> Optional[Obj]:
        return next((o for o in self if o.name == name), None)


def _asset() -> List[Obj]:
    """A compliant crate: root (class, rigid body, mass, collider) > body and lid."""
    root = Obj("crate", "EMPTY")
    root.props.update(
        {
            "simready_semantic_class": "crate",
            "simready_is_rigid_body": 1,
            "simready_mass_kg": 4.0,
            "simready_collider_type": "convexHull",
        }
    )
    body = Obj("body", parent=root, world=FakeMatrix.Translation((0.0, 0.0, 0.5)))
    lid = Obj("lid", parent=root, world=FakeMatrix.Translation((0.0, 0.0, 1.0)))
    return [root, body, lid]


@pytest.fixture
def scene(monkeypatch: pytest.MonkeyPatch) -> Any:
    objects = Objects(_asset())
    fake = SimpleNamespace(
        data=SimpleNamespace(objects=objects),
        context=SimpleNamespace(
            scene=SimpleNamespace(
                name="Scene", objects=objects, unit_settings=SimpleNamespace(scale_length=1.0)
            ),
            view_layer=SimpleNamespace(objects=objects),
            mode="OBJECT",
        ),
        app=SimpleNamespace(version=(5, 0, 1), version_string="5.0.1", background=True),
    )
    monkeypatch.setattr(blender_runtime, "bpy", fake)
    monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
    session = blender_runtime.BlenderRuntimeSession()

    def add(obj: Obj) -> Obj:
        objects.append(obj)
        return obj

    return SimpleNamespace(session=session, objects=objects, get=objects.get, add=add, fake=fake)


def _issues(result: Dict[str, Any], check: Optional[str] = None) -> List[tuple]:
    return [
        (i["object_name"], i["check"], i["severity"])
        for i in result["issues"]
        if check is None or i["check"] == check
    ]


def _scaled(sx: float, sy: float, sz: float, at: Sequence[float] = (0.0, 0.0, 0.0)) -> FakeMatrix:
    return FakeMatrix.Translation(at) @ FakeMatrix(np.diag([sx, sy, sz, 1.0]))


def _rotated_z(degrees: float, at: Sequence[float] = (0.0, 0.0, 0.0)) -> FakeMatrix:
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return FakeMatrix.Translation(at) @ FakeMatrix(
        [(c, -s, 0, 0), (s, c, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]
    )


# ---------------------------------------------------------------------------
# A clean asset, and the asset root [VG.025]
# ---------------------------------------------------------------------------


def test_a_clean_asset_is_compliant_and_the_root_is_checked_by_default(scene: Any) -> None:
    result = scene.session.validate_simready_compliance()

    assert result["compliant"] is True, result["issues"]
    assert result["object_count"] == 3  # the root empty and both meshes
    assert result["issues"] == []


@pytest.mark.parametrize(
    ("world", "said"),
    [
        (_scaled(0.01, 0.01, 0.01), "scaled"),
        (FakeMatrix.Translation((1.0, 0.0, 0.0)), "at (1.0, 0.0, 0.0)"),
        (_rotated_z(90.0), "rotated"),
    ],
)
def test_an_asset_root_that_is_not_the_identity_is_an_error(
    scene: Any, world: FakeMatrix, said: str
) -> None:
    scene.get("crate").matrix_world = world

    result = scene.session.validate_simready_compliance()

    [issue] = [i for i in result["issues"] if i["object_name"] == "crate"]
    assert (issue["check"], issue["severity"]) == ("transforms", "error")
    assert said in issue["message"] and "default prim" in issue["message"]
    assert result["compliant"] is False


def test_a_lone_mesh_off_the_origin_is_the_default_prim_and_an_error(scene: Any) -> None:
    lone = scene.add(Obj("lone_part", world=_rotated_z(90.0, at=(5.0, 0.0, 0.0))))
    lone.props.update({"simready_semantic_class": "part"})

    result = scene.session.validate_simready_compliance(["lone_part"])

    assert ("lone_part", "transforms", "error") in _issues(result)


def test_top_level_objects_wrapped_in_a_new_root_are_not_roots(scene: Any) -> None:
    """With several top-level objects the export adds an identity root above them."""
    other = scene.add(Obj("other_part", world=FakeMatrix.Translation((3.0, 0.0, 0.0))))
    other.props.update({"simready_semantic_class": "part"})
    scene.get("crate").matrix_world = FakeMatrix.Translation((-3.0, 0.0, 0.0))

    result = scene.session.validate_simready_compliance()

    assert not [i for i in result["issues"] if i["severity"] == "error"]


# ---------------------------------------------------------------------------
# Parts: scale and mirroring are warnings; skew is an error on a rigid body [RB.009]
# ---------------------------------------------------------------------------


def test_a_scaled_or_mirrored_part_is_a_warning(scene: Any) -> None:
    scene.get("body").matrix_world = _scaled(1.2, 0.6, 0.04, at=(0.0, 0.0, 0.5))
    scene.get("lid").matrix_world = _scaled(-1.0, 1.0, 1.0, at=(0.0, 0.0, 1.0))

    result = scene.session.validate_simready_compliance()

    assert ("body", "transforms", "warning") in _issues(result)
    [mirrored] = [i for i in result["issues"] if i["object_name"] == "lid"]
    assert "mirrored" in mirrored["message"] and mirrored["severity"] == "warning"
    assert result["compliant"] is True


def test_scale_in_the_parent_inverse_matrix_counts(scene: Any) -> None:
    """The exporter writes the world-relative transform, not loc/rot/scale."""
    scene.get("body").matrix_world = _scaled(0.5, 0.5, 0.5, at=(0.0, 0.0, 0.5))

    result = scene.session.validate_simready_compliance()

    assert ("body", "transforms", "warning") in _issues(result)


def test_a_skewed_rigid_body_is_an_error_and_a_skewed_part_a_warning(scene: Any) -> None:
    skew = _scaled(1.0, 3.0, 1.0) @ _rotated_z(45.0)
    scene.get("body").matrix_world = skew
    scene.get("lid").matrix_world = skew
    scene.get("lid").props.update({"simready_is_rigid_body": 1, "simready_mass_kg": 1.0})

    result = scene.session.validate_simready_compliance()

    body = [i for i in result["issues"] if i["object_name"] == "body" and "skewed" in i["message"]]
    lid = [i for i in result["issues"] if i["object_name"] == "lid" and "skewed" in i["message"]]
    assert body[0]["severity"] == "warning"
    assert lid[0]["severity"] == "error"


@pytest.mark.parametrize("rigid", [True, False])
def test_rb009_reads_a_rigid_bodys_world_skew_not_its_local_one(scene: Any, rigid: bool) -> None:
    """A rotated part under a non-uniformly scaled group: its own transform is clean."""
    crate = scene.get("crate")
    stretch = scene.add(Obj("stretch", "EMPTY", parent=crate, world=_scaled(1.0, 3.0, 1.0)))
    tilt = scene.add(Obj("tilt_part", parent=stretch, world=_scaled(1.0, 3.0, 1.0) @ _rotated_z(45.0)))
    if rigid:
        crate.props.pop("simready_is_rigid_body")
        tilt.props.update({"simready_is_rigid_body": 1, "simready_mass_kg": 2.0})

    result = scene.session.validate_simready_compliance()

    skewed = [i for i in result["issues"] if i["object_name"] == "tilt_part" and "skewed" in i["message"]]
    if rigid:
        [issue] = skewed
        assert issue["severity"] == "error" and "world transform" in issue["message"]
        assert "36.9 degrees" in issue["message"]  # axes at cos 0.8, not perpendicular
        assert result["compliant"] is False
    else:
        assert skewed == []  # a static part keeps its shape in the file


def test_a_rotated_part_is_fine(scene: Any) -> None:
    scene.get("body").matrix_world = _rotated_z(30.0, at=(0.0, 0.0, 0.5))

    assert scene.session.validate_simready_compliance()["issues"] == []


# ---------------------------------------------------------------------------
# Names: warnings; camelCase is fine
# ---------------------------------------------------------------------------


def test_names_that_are_not_identifiers_are_warnings_and_camel_case_is_fine(scene: Any) -> None:
    body, lid = scene.get("body"), scene.get("lid")
    body.name = "lidPanel"
    lid.name = "Lid Panel.001"
    lid.data = _mesh("Cube.001", materials=[SimpleNamespace(name="Bad Material.001")])

    result = scene.session.validate_simready_compliance()

    naming = [i for i in result["issues"] if i["check"] == "naming"]
    assert {i["severity"] for i in naming} == {"warning"}
    assert not [i for i in naming if i["object_name"] == "lidPanel"]
    messages = " ".join(i["message"] for i in naming)
    assert "Lid_Panel_001" in messages and "Bad_Material_001" in messages and "Cube_001" in messages
    assert result["compliant"] is True


def test_two_siblings_that_become_one_prim_name_are_reported(scene: Any) -> None:
    root = scene.get("crate")
    scene.add(Obj("arm.L", parent=root))
    scene.add(Obj("arm_L", parent=root))

    result = scene.session.validate_simready_compliance()

    assert any("also becomes the prim name 'arm_L'" in i["message"] for i in result["issues"])


# ---------------------------------------------------------------------------
# Semantic labels [SL.001], from exported objects only (M6)
# ---------------------------------------------------------------------------


def test_a_label_of_any_taxonomy_counts(scene: Any) -> None:
    root = scene.get("crate")
    del root.props["simready_semantic_class"]
    root.props["simready_semantic_qcode"] = "Q2413"

    assert scene.session.validate_simready_compliance()["compliant"] is True
    del root.props["simready_semantic_qcode"]
    root.props["simready_label_product_id"] = "C-1"
    assert scene.session.validate_simready_compliance()["compliant"] is True


def test_a_mesh_without_a_label_is_an_error(scene: Any) -> None:
    scene.add(Obj("bare_part")).props["simready_collider_type"] = "convexHull"

    result = scene.session.validate_simready_compliance(["bare_part"])

    assert ("bare_part", "metadata", "error") in _issues(result)


def test_an_unexported_ancestor_passes_nothing_on(scene: Any) -> None:
    """The crate is not exported: its label, body and collider are not in the file."""
    result = scene.session.validate_simready_compliance(["body"])

    assert ("body", "metadata", "error") in _issues(result)
    assert ("body", "metadata", "warning") in _issues(result)  # no collider either
    assert result["compliant"] is False


def test_invalid_stored_values_are_errors(scene: Any) -> None:
    scene.get("crate")["simready_restitution"] = 1.5

    result = scene.session.validate_simready_compliance(["crate"])

    [issue] = [i for i in result["issues"] if i["check"] == "metadata"]
    assert issue["severity"] == "error"
    assert "simready_restitution=1.5 must be at most 1" in issue["message"]


# ---------------------------------------------------------------------------
# Rigid bodies (M5): mass [RB.007], collider, nesting [RB.012], triangle meshes, none [RB.001]
# ---------------------------------------------------------------------------


def test_a_rigid_body_without_mass_or_density_is_an_error(scene: Any) -> None:
    del scene.get("crate").props["simready_mass_kg"]

    result = scene.session.validate_simready_compliance()

    assert ("crate", "physics", "error") in _issues(result)
    scene.get("crate")["simready_density"] = 500.0
    assert scene.session.validate_simready_compliance()["compliant"] is True


def test_a_rigid_body_without_a_collider_is_an_error(scene: Any) -> None:
    del scene.get("crate").props["simready_collider_type"]

    result = scene.session.validate_simready_compliance()

    assert any(
        i["object_name"] == "crate" and "without a collider" in i["message"]
        for i in result["issues"]
    )
    assert result["compliant"] is False


def test_nested_rigid_bodies_are_an_error(scene: Any) -> None:
    scene.get("lid").props.update({"simready_is_rigid_body": 1, "simready_mass_kg": 0.5})

    result = scene.session.validate_simready_compliance()

    [issue] = [i for i in result["issues"] if i["object_name"] == "lid" and i["check"] == "physics"]
    assert issue["severity"] == "error" and "nested in the rigid body crate" in issue["message"]
    assert "no joint between them that the export writes" in issue["message"]


# ---------------------------------------------------------------------------
# Joints: rigid body constraints between exported objects [RB.012]
# ---------------------------------------------------------------------------

_BODY = {"simready_is_rigid_body": 1, "simready_mass_kg": 1.0, "simready_collider_type": "convexHull"}


def _links(scene: Any, nested: bool = True, rigid: bool = True) -> SimpleNamespace:
    """robot (empty, class) > base_link > arm_link (or both under robot), rigid bodies."""
    robot = scene.add(Obj("robot", "EMPTY"))
    robot.props["simready_semantic_class"] = "robot"
    base = scene.add(Obj("base_link", parent=robot, world=FakeMatrix.Translation((0.0, 0.0, 0.1))))
    arm = scene.add(
        Obj("arm_link", parent=base if nested else robot, world=FakeMatrix.Translation((0.0, 0.0, 0.35)))
    )
    if rigid:
        base.props.update(_BODY)
        arm.props.update(_BODY)
    return SimpleNamespace(robot=robot, base=base, arm=arm)


def _constraint(scene: Any, kind: str, first: Obj, second: Obj, **settings: Any) -> Obj:
    """A constraint empty, as add_rigid_body_constraint makes one: top level, at the pivot."""
    carrier = scene.add(
        Obj(f"RBC_{first.name}_{second.name}", "EMPTY", world=FakeMatrix.Translation((0.0, 0.0, 0.2)))
    )
    carrier.rigid_body_constraint = SimpleNamespace(
        type=kind,
        object1=first,
        object2=second,
        enabled=True,
        disable_collisions=True,
        use_breaking=False,
        **settings,
    )
    return carrier


def _physics(result: Dict[str, Any]) -> List[tuple]:
    return [
        (i["object_name"], i["severity"], i["message"])
        for i in result["issues"]
        if i["check"] == "physics"
    ]


def test_nested_rigid_bodies_joined_by_a_constraint_are_valid(scene: Any) -> None:
    links = _links(scene)
    _constraint(scene, "HINGE", links.base, links.arm, use_limit_ang_z=True,
                limit_ang_z_lower=-1.5, limit_ang_z_upper=1.5)

    result = scene.session.validate_simready_compliance(["robot"])

    assert _physics(result) == []
    assert result["compliant"] is True, result["issues"]


def test_a_motor_constraint_is_an_error_naming_the_joint_it_would_lose(scene: Any) -> None:
    links = _links(scene, nested=False)
    _constraint(scene, "MOTOR", links.base, links.arm, use_motor_ang=True)

    result = scene.session.validate_simready_compliance(["robot"])

    [(name, severity, message)] = _physics(result)
    assert (name, severity) == ("RBC_base_link_arm_link", "error")
    assert "MOTOR constraint RBC_base_link_arm_link" in message and "drop it" in message


def test_a_motor_does_not_join_nested_bodies(scene: Any) -> None:
    links = _links(scene)
    _constraint(scene, "MOTOR", links.base, links.arm)

    result = scene.session.validate_simready_compliance(["robot"])

    errors = {(name, message.split(" ")[0]) for name, severity, message in _physics(result)
              if severity == "error"}
    assert ("arm_link", "Rigid") in errors and ("RBC_base_link_arm_link", "The") in errors


def test_a_constraint_to_an_object_left_out_is_a_warning(scene: Any) -> None:
    links = _links(scene, nested=False)
    outsider = scene.add(Obj("outsider", world=FakeMatrix.Translation((5.0, 0.0, 0.0))))
    _constraint(scene, "HINGE", links.arm, outsider)

    result = scene.session.validate_simready_compliance(["robot"])

    [(name, severity, message)] = _physics(result)
    assert (name, severity) == ("RBC_arm_link_outsider", "warning")
    assert "joins arm_link to outsider, which the export leaves out" in message


def test_a_constraint_inside_one_rigid_body_is_an_error(scene: Any) -> None:
    links = _links(scene, rigid=False)
    links.robot.props.update(_BODY)
    _constraint(scene, "FIXED", links.base, links.arm)

    result = scene.session.validate_simready_compliance(["robot"])

    [(name, severity, message)] = _physics(result)
    assert severity == "error" and "parts of the same rigid body robot" in message


def test_a_constraint_between_static_objects_is_a_warning(scene: Any) -> None:
    links = _links(scene, nested=False, rigid=False)
    _constraint(scene, "FIXED", links.base, links.arm)

    result = scene.session.validate_simready_compliance(["robot"])

    messages = [message for _, _, message in _physics(result)]
    assert any("neither of which is a rigid body" in m for m in messages)


def test_what_a_joint_leaves_out_is_a_warning(scene: Any) -> None:
    links = _links(scene, nested=False)
    _constraint(
        scene, "GENERIC_SPRING", links.base, links.arm,
        use_spring_ang_x=True,
        use_limit_ang_x=True, limit_ang_x_lower=-0.1, limit_ang_x_upper=0.1,
        use_limit_ang_z=True, limit_ang_z_lower=-0.3, limit_ang_z_upper=0.3,
    )
    links_carrier = scene.get("RBC_base_link_arm_link")
    links_carrier.rigid_body_constraint.use_breaking = True

    result = scene.session.validate_simready_compliance(["robot"])

    messages = [message for _, severity, message in _physics(result) if severity == "warning"]
    assert any("springs of the GENERIC_SPRING constraint" in m for m in messages)
    assert any("breaking threshold" in m for m in messages)
    assert any("limits ang_x, ang_z to ranges" in m for m in messages)
    assert result["compliant"] is True


def test_constraint_empties_are_joints_not_exported_objects(scene: Any) -> None:
    links = _links(scene, nested=False)
    carrier = _constraint(scene, "FIXED", links.base, links.arm)
    carrier.parent = links.robot  # under the asset: still a joint, not a part
    links.robot.children.append(carrier)

    result = scene.session.validate_simready_compliance(["robot"])

    assert result["object_count"] == 3  # robot, base_link, arm_link
    [skipped] = result["skipped_objects"]
    assert skipped["object_name"] == carrier.name and "joint" in skipped["reason"]


def test_a_triangle_mesh_collider_on_a_dynamic_body_is_a_warning(scene: Any) -> None:
    scene.get("crate")["simready_collider_type"] = "none"

    result = scene.session.validate_simready_compliance()

    assert ("body", "physics", "warning") in _issues(result)
    assert result["compliant"] is True


def test_an_asset_without_a_rigid_body_is_a_warning(scene: Any) -> None:
    del scene.get("crate").props["simready_is_rigid_body"]

    result = scene.session.validate_simready_compliance()

    assert _issues(result, "physics") == [("crate", "physics", "warning")]
    assert result["compliant"] is True


# ---------------------------------------------------------------------------
# Materials, geometry, units
# ---------------------------------------------------------------------------


def test_an_empty_slot_is_a_warning_only_when_faces_use_it(scene: Any) -> None:
    scene.get("body").data = _mesh("body", materials=["opaque_metal", None])
    assert _issues(scene.session.validate_simready_compliance(), "materials") == []

    scene.get("body").data = _mesh("body", materials=["opaque_metal", None], slots=[0, 1, 0, 0, 0, 0])
    result = scene.session.validate_simready_compliance()

    [issue] = [i for i in result["issues"] if i["check"] == "materials"]
    assert issue["severity"] == "warning" and "slot(s) 1" in issue["message"]


def test_no_material_at_all_is_an_error(scene: Any) -> None:
    scene.get("body").data = _mesh("body", materials=[])

    assert ("body", "materials", "error") in _issues(scene.session.validate_simready_compliance())


def test_uvs_ngons_and_degenerate_faces_are_checked(scene: Any) -> None:
    scene.get("body").data = _mesh(
        "body", corners=(4, 6, 3, 3), areas=(1.0, 1.0, 0.0, 1.0), uv=False
    )

    result = scene.session.validate_simready_compliance(["body"], check_metadata=False)

    geometry = {i["message"]: i["severity"] for i in result["issues"] if i["check"] == "geometry"}
    assert geometry == {
        "Mesh has no UV map; textures cannot be mapped": "warning",
        "1 of 4 faces are n-gons (more than 4 corners)": "warning",
        "1 of 4 faces are degenerate (zero area)": "error",
    }


def test_geometry_is_read_from_the_mesh_with_modifiers_applied(scene: Any) -> None:
    body = scene.get("body")
    body.data = _mesh("body", corners=(6, 6))
    triangulated = _mesh("body", corners=(3, 3, 3, 3, 3, 3, 3, 3))
    released: List[bool] = []
    body.modifiers = ["TRIANGULATE"]
    body.evaluated_get = lambda graph: SimpleNamespace(
        to_mesh=lambda: triangulated, to_mesh_clear=lambda: released.append(True)
    )
    scene.fake.context.evaluated_depsgraph_get = lambda: None  # no depsgraph for the worlds

    result = scene.session.validate_simready_compliance(["body"], check_metadata=False)

    assert _issues(result, "geometry") == []
    assert released == [True]


def test_edit_mode_edits_are_written_back_before_the_checks(scene: Any) -> None:
    body = scene.get("body")
    body.mode = "EDIT"
    flushed: List[str] = []
    body.update_from_editmode = lambda: flushed.append("body")

    scene.session.validate_simready_compliance()

    assert flushed == ["body"]


def test_a_size_implausible_in_the_declared_units_is_a_warning(scene: Any) -> None:
    scene.get("body").dimensions = (500.0, 500.0, 500.0)

    result = scene.session.validate_simready_compliance()

    assert ("body", "scale", "warning") in _issues(result)
    assert result["compliant"] is True


def test_a_unit_scale_other_than_one_is_a_warning(scene: Any) -> None:
    scene.fake.context.scene.unit_settings.scale_length = 0.01

    result = scene.session.validate_simready_compliance()

    assert any("Unit Scale is 0.01" in i["message"] for i in result["issues"])
    assert result["compliant"] is True


# ---------------------------------------------------------------------------
# Scope: what the export exports
# ---------------------------------------------------------------------------


def test_the_scope_is_the_exports(scene: Any) -> None:
    """Render-hidden helpers and lights, which the export leaves out, are not checked."""
    helper = scene.add(Obj("Reference.Scan", data=_mesh("scan", materials=[])))
    helper.hide_render = True
    scene.add(Obj("Light", "LIGHT"))

    result = scene.session.validate_simready_compliance()

    assert result["compliant"] is True
    assert {s["object_name"] for s in result["skipped_objects"]} == {"Reference.Scan", "Light"}


def test_listed_objects_are_checked_with_their_descendants(scene: Any) -> None:
    assert scene.session.validate_simready_compliance(["crate"])["object_count"] == 3
    only = scene.session.validate_simready_compliance(["crate"], include_children=False)
    assert only["object_count"] == 1


def test_an_empty_object_list_is_refused(scene: Any) -> None:
    with pytest.raises(blender_runtime.ValidationError) as excinfo:
        scene.session.validate_simready_compliance([])
    assert excinfo.value.details["code"] == "EmptyObjectList"


# ---------------------------------------------------------------------------
# Paging
# ---------------------------------------------------------------------------


def test_issues_are_filtered_by_severity_and_paged(scene: Any) -> None:
    for index in range(10):
        scene.add(Obj(f"Part.{index:03d}"))  # each: name, label, collider, hierarchy issues

    everything = scene.session.validate_simready_compliance(limit=500)
    errors = scene.session.validate_simready_compliance(severity="error", limit=500)
    first = scene.session.validate_simready_compliance(limit=10)
    second = scene.session.validate_simready_compliance(offset=first["next_offset"], limit=10)

    assert everything["issue_count"] == errors["issue_count"] == len(everything["issues"])
    assert {i["severity"] for i in errors["issues"]} == {"error"}
    assert len(errors["issues"]) == everything["error_count"]
    assert errors["compliant"] is everything["compliant"] is False
    assert first["next_offset"] == 10 and len(first["issues"]) == 10
    assert second["offset"] == 10
    assert second["issues"] == everything["issues"][10:20]
    last = scene.session.validate_simready_compliance(offset=everything["issue_count"] - 1)
    assert last["next_offset"] is None


def test_a_page_stays_inside_the_result_budget(scene: Any) -> None:
    for index in range(400):
        scene.add(Obj(f"Part.{index:03d}"))

    page = scene.session.validate_simready_compliance(limit=500)

    assert len(page["issues"]) < 500
    assert page["next_offset"] == len(page["issues"])
    assert len(json.dumps(page["issues"])) <= 30_000 + 1_000


@pytest.mark.parametrize(
    ("severity", "offset", "limit"), [("fatal", 0, 10), ("all", -1, 10), ("all", 0, 0), ("all", 0, 501)]
)
def test_bad_paging_arguments_are_refused(scene: Any, severity: str, offset: int, limit: int) -> None:
    with pytest.raises(blender_runtime.ValidationError) as excinfo:
        scene.session.validate_simready_compliance(severity=severity, offset=offset, limit=limit)
    assert excinfo.value.details["code"] == "InvalidPaging"


def test_the_validate_response_model_accepts_a_page(scene: Any) -> None:
    from simul.mcp.schemas.simready import SimReadyValidateResponse

    scene.add(Obj("Light", "LIGHT"))
    payload = scene.session.validate_simready_compliance(limit=5)

    response = SimReadyValidateResponse(success=True, **payload)
    assert response.next_offset is None
    assert response.skipped_objects[0].object_name == "Light"
