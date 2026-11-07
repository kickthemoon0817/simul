"""validate_simready_compliance: real checks with severities, where errors flip compliant."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import pytest

from simul.adapters import blender_runtime


class Polygons(list):
    """mesh.polygons: ``foreach_get`` of loop_total and area, as Blender offers it."""

    def foreach_get(self, attribute: str, out: Any) -> None:
        for index, polygon in enumerate(self):
            out[index] = polygon[attribute]


def _mesh(
    name: str,
    corners: Sequence[int] = (4, 4, 4, 4, 4, 4),
    areas: Optional[Sequence[float]] = None,
    uv: bool = True,
    materials: Sequence[Any] = ("opaque_metal",),
) -> SimpleNamespace:
    areas = list(areas) if areas is not None else [1.0] * len(corners)
    return SimpleNamespace(
        name=name,
        materials=list(materials),
        polygons=Polygons({"loop_total": c, "area": a} for c, a in zip(corners, areas)),
        uv_layers=["UVMap"] if uv else [],
    )


class Obj:
    """A Blender object with the transform, material and custom-property surface validation reads."""

    def __init__(
        self,
        name: str,
        kind: str = "MESH",
        parent: Optional["Obj"] = None,
        data: Optional[Any] = None,
        **transform: Any,
    ) -> None:
        self.name = name
        self.type = kind
        self.parent = parent
        self.children: List[Obj] = []
        if parent is not None:
            parent.children.append(self)
        self.data = data if data is not None or kind != "MESH" else _mesh(name)
        self.dimensions = (1.0, 1.0, 1.0) if kind == "MESH" else (0.0, 0.0, 0.0)
        self.location = (0.0, 0.0, 0.0)
        self.delta_location = (0.0, 0.0, 0.0)
        self.rotation_mode = "XYZ"
        self.rotation_euler = (0.0, 0.0, 0.0)
        self.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        self.rotation_axis_angle = (0.0, 0.0, 1.0, 0.0)
        self.delta_rotation_euler = (0.0, 0.0, 0.0)
        self.delta_rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        self.scale = (1.0, 1.0, 1.0)
        self.delta_scale = (1.0, 1.0, 1.0)
        self.modifiers: List[Any] = []
        self.props: Dict[str, Any] = {}
        for key, value in transform.items():
            setattr(self, key, value)

    def __setitem__(self, key: str, value: Any) -> None:
        self.props[key] = value

    def __getitem__(self, key: str) -> Any:
        return self.props[key]

    def keys(self) -> List[str]:
        return list(self.props)


class Objects(list):
    def get(self, name: str) -> Optional[Obj]:
        return next((o for o in self if o.name == name), None)


def _asset() -> List[Obj]:
    """A compliant crate: root (class, rigid body) > body (collider) and lid."""
    root = Obj("crate", "EMPTY")
    root.props.update(
        {
            "simready_semantic_class": "crate",
            "simready_is_rigid_body": 1,
            "simready_mass_kg": 4.0,
            "simready_collider_type": "convexHull",
        }
    )
    body = Obj("body", parent=root, location=(0.0, 0.0, 0.5))
    lid = Obj("lid", parent=root, location=(0.0, 0.0, 1.0))
    return [root, body, lid]


@pytest.fixture
def scene(monkeypatch: pytest.MonkeyPatch) -> Any:
    objects = Objects(_asset())
    fake = SimpleNamespace(
        data=SimpleNamespace(objects=objects),
        context=SimpleNamespace(
            scene=SimpleNamespace(objects=objects, unit_settings=SimpleNamespace(scale_length=1.0)),
            evaluated_depsgraph_get=lambda: "depsgraph",
        ),
        app=SimpleNamespace(version=(5, 0, 1), version_string="5.0.1", background=True),
    )
    monkeypatch.setattr(blender_runtime, "bpy", fake)
    monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
    session = blender_runtime.BlenderRuntimeSession()

    def add(obj: Obj) -> Obj:
        objects.append(obj)
        return obj

    return SimpleNamespace(session=session, objects=objects, get=objects.get, add=add)


def _issues(result: Dict[str, Any], check: Optional[str] = None) -> List[tuple]:
    return [
        (i["object_name"], i["check"], i["severity"])
        for i in result["issues"]
        if check is None or i["check"] == check
    ]


def test_a_clean_asset_is_compliant_and_the_root_is_checked_by_default(scene: Any) -> None:
    result = scene.session.validate_simready_compliance()

    assert result["compliant"] is True, result["issues"]
    assert result["object_count"] == 3  # the root empty and both meshes
    assert result["issues"] == []


def test_a_scaled_root_is_an_error(scene: Any) -> None:
    scene.get("crate").scale = (0.01, 0.01, 0.01)

    result = scene.session.validate_simready_compliance()

    assert result["compliant"] is False
    assert ("crate", "transforms", "error") in _issues(result)


def test_a_root_away_from_the_origin_or_rotated_is_an_error(scene: Any) -> None:
    root = scene.get("crate")
    root.location = (1.0, 0.0, 0.0)
    root.rotation_mode = "QUATERNION"
    root.rotation_quaternion = (0.7071068, 0.7071068, 0.0, 0.0)  # glTF's Y-up turn

    result = scene.session.validate_simready_compliance(["crate"])

    messages = [i["message"] for i in result["issues"] if i["check"] == "transforms"]
    assert any("not the origin" in m for m in messages)
    assert any("quaternion" in m for m in messages)
    assert result["error_count"] == 2


def test_quaternion_axis_angle_and_delta_transforms_count(scene: Any) -> None:
    body = scene.get("body")
    body.rotation_mode = "QUATERNION"
    body.rotation_quaternion = (0.9238795, 0.0, 0.0, 0.3826834)
    lid = scene.get("lid")
    lid.delta_scale = (100.0, 100.0, 100.0)
    lid.delta_rotation_euler = (1.0, 0.0, 0.0)

    result = scene.session.validate_simready_compliance(["body", "lid"])

    assert ("body", "transforms", "warning") in _issues(result)  # a rotated part
    assert ("lid", "transforms", "error") in _issues(result)  # a delta scale
    assert ("lid", "transforms", "warning") in _issues(result)  # a delta rotation
    assert result["compliant"] is False


def test_a_mirrored_part_is_an_error(scene: Any) -> None:
    scene.get("body").scale = (-1.0, 1.0, 1.0)

    result = scene.session.validate_simready_compliance(["body"])

    [issue] = [i for i in result["issues"] if i["check"] == "transforms"]
    assert issue["severity"] == "error" and "mirrors" in issue["message"]


def test_a_mesh_without_metadata_is_not_compliant(scene: Any) -> None:
    bare = scene.add(Obj("bare_part", parent=None))

    result = scene.session.validate_simready_compliance(["bare_part"])

    assert ("bare_part", "metadata", "error") in _issues(result)  # no semantic class
    assert ("bare_part", "metadata", "warning") in _issues(result)  # no collider
    assert result["compliant"] is False
    bare["simready_semantic_class"] = "part"
    assert scene.session.validate_simready_compliance(["bare_part"])["compliant"] is True


def test_invalid_stored_values_are_errors(scene: Any) -> None:
    scene.get("crate")["simready_restitution"] = 1.5

    result = scene.session.validate_simready_compliance(["crate"])

    [issue] = [i for i in result["issues"] if i["check"] == "metadata"]
    assert issue["severity"] == "error"
    assert "simready_restitution=1.5 must be at most 1" in issue["message"]


def test_a_rigid_body_without_mass_is_a_warning(scene: Any) -> None:
    del scene.get("crate").props["simready_mass_kg"]

    result = scene.session.validate_simready_compliance(["crate"])

    assert _issues(result, "metadata") == [("crate", "metadata", "warning")]
    assert result["compliant"] is True


def test_material_and_mesh_data_names_are_checked(scene: Any) -> None:
    scene.get("body").data = _mesh("Cube.001", materials=[SimpleNamespace(name="Bad Material.001")])

    result = scene.session.validate_simready_compliance(["body"])

    naming = {i["message"]: i["severity"] for i in result["issues"] if i["check"] == "naming"}
    assert any("Material name 'Bad Material.001'" in m and s == "error" for m, s in naming.items())
    assert any("Mesh data name 'Cube.001'" in m and s == "warning" for m, s in naming.items())


def test_an_empty_material_slot_is_an_error(scene: Any) -> None:
    scene.get("body").data = _mesh("body", materials=["opaque_metal", None])

    result = scene.session.validate_simready_compliance(["body"])

    [issue] = [i for i in result["issues"] if i["check"] == "materials"]
    assert issue["severity"] == "error" and "slot(s) 1" in issue["message"]


def test_uvs_ngons_and_degenerate_faces_are_checked(scene: Any) -> None:
    scene.get("body").data = _mesh(
        "body", corners=(4, 6, 3, 3), areas=(1.0, 1.0, 0.0, 1.0), uv=False
    )

    result = scene.session.validate_simready_compliance(["body"])

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

    result = scene.session.validate_simready_compliance(["body"])

    assert _issues(result, "geometry") == []
    assert released == [True]


def test_an_empty_object_list_is_refused(scene: Any) -> None:
    with pytest.raises(ValueError, match="object_names is empty"):
        scene.session.validate_simready_compliance([])


def test_issues_are_filtered_by_severity_and_paged(scene: Any) -> None:
    for index in range(10):
        scene.add(Obj(f"Part.{index:03d}"))  # five issues each, errors and warnings

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
    import json

    assert len(json.dumps(page["issues"])) <= 30_000 + 1_000


@pytest.mark.parametrize(
    ("severity", "offset", "limit"), [("fatal", 0, 10), ("all", -1, 10), ("all", 0, 0), ("all", 0, 501)]
)
def test_bad_paging_arguments_are_refused(scene: Any, severity: str, offset: int, limit: int) -> None:
    with pytest.raises(ValueError):
        scene.session.validate_simready_compliance(severity=severity, offset=offset, limit=limit)


def test_the_validate_response_model_accepts_a_page(scene: Any) -> None:
    from simul.mcp.schemas.simready import SimReadyValidateResponse

    payload = scene.session.validate_simready_compliance(limit=5)

    assert SimReadyValidateResponse(success=True, **payload).next_offset is None
