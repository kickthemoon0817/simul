"""author_simready_layer on synthetic layers shaped like Blender's USD export, read back with pxr."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

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
