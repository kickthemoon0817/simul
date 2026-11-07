"""author_simready_layer on synthetic layers shaped like Blender's USD export, read back with pxr."""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from simul.adapters import simready_usd
from simul.adapters.simready_usd import (
    LEGACY_SEMANTICS_SCHEMA,
    author_simready_layer,
    usd_identifier,
)

_BLENDER_LAYER = '''#usda 1.0
(
    defaultPrim = "{container}"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{container}"
{{
    def Xform "crate"
    {{
        custom string userProperties:blender:object_name = "crate"

        def Xform "body"
        {{
            custom string userProperties:blender:object_name = "body"

            def Mesh "body_mesh" (
                prepend apiSchemas = ["MaterialBindingAPI"]
            )
            {{
                point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
                int[] faceVertexCounts = [3]
                int[] faceVertexIndices = [0, 1, 2]
                rel material:binding = </{container}/_materials/wood>
            }}
        }}

        def Xform "lid"
        {{
            custom string userProperties:blender:object_name = "lid"

            def Mesh "lid_mesh"
            {{
                point3f[] points = [(0, 0, 1), (1, 0, 1), (0, 1, 1)]
                int[] faceVertexCounts = [3]
                int[] faceVertexIndices = [0, 1, 2]
            }}
        }}
    }}

    def Scope "_materials"
    {{
        def Material "wood"
        {{
            token outputs:surface.connect = </{container}/_materials/wood/Principled_BSDF.outputs:surface>

            def Shader "Principled_BSDF"
            {{
                uniform token info:id = "UsdPreviewSurface"
                token outputs:surface
            }}
        }}
    }}
}}
'''


def _layer(tmp_path: Path, container: str = "root", text: Optional[str] = None) -> str:
    path = tmp_path / "blender.usda"
    body = text if text is not None else _BLENDER_LAYER.format(container=container)
    path.write_text(body, encoding="utf-8")
    return str(path)


def _plan(objects: List[Dict[str, Any]], **extra: Any) -> Dict[str, Any]:
    plan: Dict[str, Any] = {"objects": objects, "asset_name": "crate_asset", "meters_per_unit": 1.0}
    plan.update(extra)
    return plan


def _crate(
    crate: Optional[Dict[str, Any]] = None,
    body: Optional[Dict[str, Any]] = None,
    lid: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    return [
        {"name": "crate", "parent": None, "physics": crate or {}, "semantic": {}},
        {"name": "body", "parent": "crate", "physics": body or {}, "semantic": {}},
        {"name": "lid", "parent": "crate", "physics": lid or {}, "semantic": {}},
    ]


def _stage(layer: Any) -> Any:
    return Usd.Stage.Open(layer)


def test_a_root_level_collider_and_material_apply_to_every_mesh_below(tmp_path: Path) -> None:
    plan = _plan(
        _crate(
            crate={
                "is_rigid_body": True,
                "mass_kg": 4.0,
                "collider_type": "convexHull",
                "static_friction": 0.8,
            }
        )
    )

    layer, report = author_simready_layer(_layer(tmp_path), plan)

    stage = _stage(layer)
    for mesh in ("/crate/body/body_mesh", "/crate/lid/lid_mesh"):
        prim = stage.GetPrimAtPath(mesh)
        assert prim.HasAPI(UsdPhysics.CollisionAPI), mesh
        bound = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")[0]
        assert bound.GetPath() == Sdf.Path("/crate/PhysicsMaterials/crate")
    assert report["rigid_bodies"] == ["/crate"]
    assert report["warnings"] == []


def test_blender_root_container_is_replaced_by_the_asset_root(tmp_path: Path) -> None:
    layer, report = author_simready_layer(_layer(tmp_path), _plan(_crate()))

    stage = _stage(layer)
    assert [str(p.GetPath()) for p in stage.GetPseudoRoot().GetChildren()] == ["/crate"]
    assert layer.defaultPrim == "crate"
    mesh = stage.GetPrimAtPath("/crate/body/body_mesh")
    assert UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()[0].GetPath() == Sdf.Path(
        "/crate/Looks/wood"
    )
    assert report["prims"] == {"crate": "/crate", "body": "/crate/body", "lid": "/crate/lid"}


def test_a_layer_without_a_root_container_keeps_its_root_and_takes_the_materials(
    tmp_path: Path,
) -> None:
    # Blender with root_prim_path "" writes the objects and materials at the top level.
    text = _BLENDER_LAYER.format(container="root").replace("</root/", "</")
    flat = Sdf.Layer.CreateAnonymous("flat.usda")
    flat.ImportFromString(text)
    for name in ("crate", "_materials"):
        Sdf.CopySpec(flat, f"/root/{name}", flat, f"/{name}")
    edit = Sdf.BatchNamespaceEdit()
    edit.Add(Sdf.NamespaceEdit.Remove("/root"))
    flat.Apply(edit)
    path = tmp_path / "flat.usda"
    flat.Export(str(path))

    layer, _ = author_simready_layer(str(path), _plan(_crate()))

    stage = _stage(layer)
    assert [str(p.GetPath()) for p in stage.GetPseudoRoot().GetChildren()] == ["/crate"]
    mesh = stage.GetPrimAtPath("/crate/body/body_mesh")
    assert UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()[0].GetPath() == Sdf.Path(
        "/crate/Looks/wood"
    )


def test_semantics_use_isaac_semantics_api_without_usd_semantics(tmp_path: Path) -> None:
    objects = _crate()
    objects[0]["semantic"] = {"semantic_class": "crate", "semantic_hierarchy": "container/crate"}

    layer, report = author_simready_layer(_layer(tmp_path), _plan(objects), use_labels_api=False)

    stage = _stage(layer)
    prim = stage.GetPrimAtPath("/crate")
    # pxr lists only registered schemas in GetAppliedSchemas; read the metadata.
    assert "SemanticsAPI:Semantics_class" in prim.GetMetadata("apiSchemas").GetAddedOrExplicitItems()
    assert prim.GetAttribute("semantic:Semantics_class:params:semanticType").Get() == "class"
    assert prim.GetAttribute("semantic:Semantics_class:params:semanticData").Get() == "crate"
    assert (
        prim.GetAttribute("semantic:Semantics_hierarchy:params:semanticData").Get()
        == "container/crate"
    )
    assert report["semantics_schema"] == LEGACY_SEMANTICS_SCHEMA


@pytest.mark.parametrize(
    ("collider_type", "approximation"),
    [
        ("convexDecomposition", "convexDecomposition"),
        ("meshSimplification", "meshSimplification"),
        ("none", "none"),
        ("mesh", "none"),
        ("box", "boundingCube"),
        ("sphere", "boundingSphere"),
        ("capsule", "convexHull"),
    ],
)
def test_collider_types_map_to_usd_physics_approximations(
    tmp_path: Path, collider_type: str, approximation: str
) -> None:
    layer, report = author_simready_layer(
        _layer(tmp_path), _plan(_crate(body={"collider_type": collider_type}))
    )

    stage = _stage(layer)
    mesh = stage.GetPrimAtPath("/crate/body/body_mesh")
    assert UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Get() == approximation
    assert any("capsule" in w for w in report["warnings"]) is (collider_type == "capsule")


def test_physics_problems_are_reported_not_hidden(tmp_path: Path) -> None:
    plan = _plan(
        _crate(
            crate={"is_rigid_body": True},
            body={"collider_type": "none", "mass_kg": "heavy", "restitution": 1.5},
            lid={"collider_type": "cylinder"},
        )
    )

    layer, report = author_simready_layer(_layer(tmp_path), plan)

    warnings = "\n".join(report["warnings"])
    assert "mass_kg='heavy'" in warnings
    assert "restitution=1.5" in warnings
    assert "'cylinder'" in warnings
    assert "triangle-mesh collider" in warnings
    stage = _stage(layer)
    body = stage.GetPrimAtPath("/crate/body")
    assert not body.HasAPI(UsdPhysics.MassAPI)


def test_a_rigid_body_without_a_collider_is_reported(tmp_path: Path) -> None:
    _, report = author_simready_layer(
        _layer(tmp_path), _plan(_crate(crate={"is_rigid_body": True}))
    )

    assert any("without a collider" in w for w in report["warnings"])


def test_objects_missing_from_the_layer_are_reported(tmp_path: Path) -> None:
    objects = _crate() + [{"name": "handle", "parent": "crate", "physics": {}, "semantic": {}}]

    _, report = author_simready_layer(_layer(tmp_path), _plan(objects))

    assert report["missing_objects"] == ["handle"]


def test_an_exporter_without_blender_names_is_matched_by_prim_name(tmp_path: Path) -> None:
    text = "\n".join(
        line
        for line in _BLENDER_LAYER.format(container="root").splitlines()
        if "blender:object_name" not in line
    )

    _, report = author_simready_layer(_layer(tmp_path, text=text), _plan(_crate()))

    assert report["prims"] == {"crate": "/crate", "body": "/crate/body", "lid": "/crate/lid"}


def test_stage_metrics_follow_the_plan(tmp_path: Path) -> None:
    layer, report = author_simready_layer(_layer(tmp_path), _plan(_crate(), meters_per_unit=0.001))

    stage = _stage(layer)
    assert UsdGeom.GetStageMetersPerUnit(stage) == pytest.approx(0.001)
    assert report["meters_per_unit"] == 0.001


@pytest.mark.parametrize(
    ("name", "identifier"),
    [("crate", "crate"), ("staged scene", "staged_scene"), ("3d_part", "_3d_part"), ("", "asset")],
)
def test_usd_identifier(name: str, identifier: str) -> None:
    assert usd_identifier(name) == identifier


def test_a_qcode_goes_under_simreadys_wikidata_qcode_taxonomy(tmp_path: Path) -> None:
    objects = _crate()
    objects[0]["semantic"] = {
        "semantic_class": "crate",
        "semantic_qcode": "Q1420",
        # Stored by an earlier Simul: it would overwrite the class label.
        "additional_labels": {"class": "vehicle", "sku": "C-1"},
    }

    layer, report = author_simready_layer(_layer(tmp_path), _plan(objects))

    stage = _stage(layer)
    prim = stage.GetPrimAtPath("/crate")
    assert list(prim.GetAttribute("semantics:labels:wikidata_qcode").Get()) == ["Q1420"]
    assert list(prim.GetAttribute("semantics:labels:class").Get()) == ["crate"]
    assert list(prim.GetAttribute("semantics:labels:sku").Get()) == ["C-1"]
    assert not prim.GetAttribute("semantics:labels:qcode")
    assert any("'class' is a semantic field's taxonomy" in w for w in report["warnings"])


# ---------------------------------------------------------------------------
# Joints
# ---------------------------------------------------------------------------


def _translate(x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def _rotate(axis: str, degrees: float) -> np.ndarray:
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    m = np.eye(4)
    i, j = {"x": (1, 2), "y": (2, 0), "z": (0, 1)}[axis]
    m[i, i], m[i, j], m[j, i], m[j, j] = c, -s, s, c
    return m


def _scale(factor: float) -> np.ndarray:
    return np.diag([factor, factor, factor, 1.0])


# The links: base_link at (0, 0, 0.1); arm_link turned 90 degrees about Z and
# scaled 2x at (0, 0, 0.35). The hinge pivot is at (0, 0, 0.2), its Z axis
# along world +Y (as add_rigid_body_constraint's axis=[0, 1, 0] turns it).
_BASE = _translate(0.0, 0.0, 0.1)
_ARM = _translate(0.0, 0.0, 0.35) @ _rotate("z", 90.0) @ _scale(2.0)
_PIVOT = _translate(0.0, 0.0, 0.2) @ _rotate("x", -90.0)


def _articulated_layer(tmp_path: Path, nested: bool = False) -> str:
    """robot > base_link and arm_link (or base_link > arm_link), as Blender writes them."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/root")

    def xform(path: str, name: str, local: np.ndarray) -> None:
        prim = UsdGeom.Xform.Define(stage, path)
        prim.GetPrim().CreateAttribute(
            "userProperties:blender:object_name", Sdf.ValueTypeNames.String
        ).Set(name)
        if not np.allclose(local, np.eye(4)):
            prim.AddTransformOp().Set(Gf.Matrix4d(local.T.tolist()))
        mesh = UsdGeom.Mesh.Define(stage, f"{path}/{name}_mesh")
        mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])

    xform("/root/robot", "robot", np.eye(4))
    xform("/root/robot/base_link", "base_link", _BASE)
    if nested:
        xform("/root/robot/base_link/arm_link", "arm_link", np.linalg.inv(_BASE) @ _ARM)
    else:
        xform("/root/robot/arm_link", "arm_link", _ARM)
    path = tmp_path / "articulated.usda"
    stage.GetRootLayer().Export(str(path))
    return str(path)


def _links(nested: bool = False) -> List[Dict[str, Any]]:
    body = {"is_rigid_body": True, "mass_kg": 1.0, "collider_type": "convexHull"}
    return [
        {"name": "robot", "parent": None, "physics": {}, "semantic": {"semantic_class": "robot"}},
        {"name": "base_link", "parent": "robot", "physics": dict(body), "semantic": {}},
        {
            "name": "arm_link",
            "parent": "base_link" if nested else "robot",
            "physics": dict(body),
            "semantic": {},
        },
    ]


def _joint(**extra: Any) -> Dict[str, Any]:
    joint: Dict[str, Any] = {
        "name": "RBC_base_link_arm_link",
        "type": "revolute",
        "body0": "base_link",
        "body1": "arm_link",
        "frame": _PIVOT.tolist(),
        "body0_world": _BASE.tolist(),
        "body1_world": _ARM.tolist(),
        "enabled": True,
        "collision_enabled": False,
        "axis": "Z",
        "limit": [-28.6, 0.0],
    }
    joint.update(extra)
    return joint


def _parsed_joints(stage: Any) -> Dict[str, Any]:
    """UsdPhysics' parse of the stage's joints: prim path -> joint description."""
    loader = getattr(UsdPhysics, "LoadUsdPhysicsFromRange", None)
    if loader is None:  # pragma: no cover - pxr without the parser
        pytest.skip("this pxr has no UsdPhysics parser")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        parsed = loader(stage, ["/"])
    return {
        str(path): desc
        for kind, (paths, descs) in parsed.items()
        if "Joint" in str(kind)
        for path, desc in zip(paths, descs)
    }


def _world_frame(stage: Any, body: Any, position: Any, orientation: Any) -> Any:
    """A joint frame in the world from a body, as a simulator composes it (body scale is in position)."""
    transform = Gf.Transform(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(body)))
    rotation = transform.GetRotation()
    world = Gf.Matrix4d().SetRotate(
        rotation.GetQuat() * Gf.Quatd(orientation.GetReal(), Gf.Vec3d(orientation.GetImaginary()))
    )
    world.SetTranslateOnly(rotation.TransformDir(Gf.Vec3d(position)) + transform.GetTranslation())
    return np.asarray(world).T  # column vectors, as the plan's matrices


def test_a_hinge_becomes_a_revolute_joint_both_bodies_place_at_the_pivot(tmp_path: Path) -> None:
    layer, report = author_simready_layer(
        _articulated_layer(tmp_path), _plan(_links(), joints=[_joint()])
    )

    stage = _stage(layer)
    path = "/robot/Joints/RBC_base_link_arm_link"
    assert report["joints"] == [path]
    joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(path))
    assert joint and joint.GetAxisAttr().Get() == "Z"
    assert joint.GetBody0Rel().GetTargets() == [Sdf.Path("/robot/base_link")]
    assert joint.GetBody1Rel().GetTargets() == [Sdf.Path("/robot/arm_link")]
    assert (joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()) == pytest.approx(
        (-28.6, 0.0)
    )
    assert joint.GetCollisionEnabledAttr().Get() is False
    assert joint.GetJointEnabledAttr().Get() is True
    # The pivot in the 2x arm's own coordinates: the parser scales it again.
    expected = np.linalg.inv(_ARM) @ np.array([0.0, 0.0, 0.2, 1.0])
    assert list(joint.GetLocalPos1Attr().Get()) == pytest.approx(expected[:3].tolist(), abs=1e-6)

    desc = _parsed_joints(stage)[path]
    for body, position, orientation in (
        (desc.body0, desc.localPose0Position, desc.localPose0Orientation),
        (desc.body1, desc.localPose1Position, desc.localPose1Orientation),
    ):
        world = _world_frame(stage, body, position, orientation)
        assert world == pytest.approx(_PIVOT, abs=1e-5)
    # The hinge turns about world +Y.
    assert (_PIVOT[:3, :3] @ [0.0, 0.0, 1.0]) == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)


def test_a_d6_joint_locks_and_limits_each_axis_with_limit_api(tmp_path: Path) -> None:
    joint = _joint(type="d6", limits={"transX": [1.0, -1.0], "rotY": [-22.9, 11.5]})
    del joint["axis"], joint["limit"]

    layer, report = author_simready_layer(
        _articulated_layer(tmp_path), _plan(_links(), joints=[joint])
    )

    stage = _stage(layer)
    prim = stage.GetPrimAtPath(report["joints"][0])
    assert prim.GetTypeName() == "PhysicsJoint"
    assert sorted(prim.GetAppliedSchemas()) == ["PhysicsLimitAPI:rotY", "PhysicsLimitAPI:transX"]
    assert prim.GetAttribute("limit:transX:physics:low").Get() == 1.0
    assert prim.GetAttribute("limit:transX:physics:high").Get() == -1.0
    assert prim.GetAttribute("limit:rotY:physics:low").Get() == pytest.approx(-22.9)
    desc = _parsed_joints(stage)[report["joints"][0]]
    assert str(desc.type).endswith("D6Joint")


@pytest.mark.parametrize(
    ("kind", "type_name"),
    [
        ("fixed", "PhysicsFixedJoint"),
        ("prismatic", "PhysicsPrismaticJoint"),
        ("spherical", "PhysicsSphericalJoint"),
    ],
)
def test_each_joint_type_has_its_usd_physics_schema(
    tmp_path: Path, kind: str, type_name: str
) -> None:
    joint = _joint(type=kind, axis="X", limit=[0.0, 0.3] if kind == "prismatic" else None)

    layer, report = author_simready_layer(
        _articulated_layer(tmp_path), _plan(_links(), joints=[joint])
    )

    stage = _stage(layer)
    prim = stage.GetPrimAtPath(report["joints"][0])
    assert prim.GetTypeName() == type_name
    if kind == "prismatic":
        assert prim.GetAttribute("physics:axis").Get() == "X"
        assert prim.GetAttribute("physics:upperLimit").Get() == pytest.approx(0.3)


def test_a_joint_between_nested_bodies_quiets_the_nesting_warning(tmp_path: Path) -> None:
    nested = _joint(body1_world=_ARM.tolist())

    _, without = author_simready_layer(_articulated_layer(tmp_path, nested=True), _plan(_links(True)))
    layer, joined = author_simready_layer(
        _articulated_layer(tmp_path, nested=True), _plan(_links(True), joints=[nested])
    )

    assert any("nested under the rigid body" in w for w in without["warnings"])
    assert not any("nested under the rigid body" in w for w in joined["warnings"])
    stage = _stage(layer)
    joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(joined["joints"][0]))
    assert joint.GetBody1Rel().GetTargets() == [Sdf.Path("/robot/base_link/arm_link")]
    desc = _parsed_joints(stage)[joined["joints"][0]]
    world = _world_frame(stage, desc.body1, desc.localPose1Position, desc.localPose1Orientation)
    assert world == pytest.approx(_PIVOT, abs=1e-5)


def test_a_joint_whose_body_has_no_prim_is_reported_and_not_written(tmp_path: Path) -> None:
    layer, report = author_simready_layer(
        _articulated_layer(tmp_path), _plan(_links(), joints=[_joint(body1="ghost")])
    )

    assert report["joints"] == []
    assert not _stage(layer).GetPrimAtPath("/robot/Joints")
    assert any("no prim for ghost" in w for w in report["warnings"])


def test_a_joint_frame_with_zero_scale_is_reported(tmp_path: Path) -> None:
    flat = np.diag([0.0, 0.0, 0.0, 1.0]).tolist()

    _, report = author_simready_layer(
        _articulated_layer(tmp_path), _plan(_links(), joints=[_joint(frame=flat)])
    )

    assert report["joints"] == []
    assert any("zero scale" in w for w in report["warnings"])


@pytest.mark.parametrize(
    "turns",
    [("x", 0.0), ("x", -90.0), ("y", 180.0), ("z", 179.0), ("x", 37.0), ("y", -123.0)],
)
def test_joint_rotations_match_pxr(turns: tuple) -> None:
    rotation = _rotate(*turns) @ _rotate("y", 25.0)

    w, x, y, z = simready_usd._quaternion(rotation[:3, :3].tolist())

    expected = Gf.Matrix4d(rotation.T.tolist()).ExtractRotationQuat()
    sign = 1.0 if expected.GetReal() >= 0 else -1.0
    assert [w, x, y, z] == pytest.approx(
        [sign * expected.GetReal(), *(sign * v for v in expected.GetImaginary())], abs=1e-9
    )
