"""get_prim_info reports what the prim is made of, in valid, bounded JSON.

The audit found four gaps, each pinned here against real pxr stages driven
through the MCP server:

- ``metadata`` was always ``{}`` (a call to a method pxr does not have,
  swallowed by a bare except): no API schemas, composition arcs, variants,
  instancing, specifier or authored metadata;
- relationships (joint bodies, material bindings, collision filters) were
  dropped from the response;
- a joint's default ``physics:breakForce`` went out as the bare token
  ``Infinity`` (invalid JSON), vectors and quaternions as strings, and a
  fallback value looked the same as an authored one;
- a 100k-vertex mesh produced a 4.7 MB response (51 MB at 1M vertices),
  because every array was expanded whole.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt


def _asset(path: Path) -> Path:
    """A two-link robot asset with a joint, a physics material and a collision filter."""
    stage = Usd.Stage.CreateNew(str(path))
    robot = UsdGeom.Xform.Define(stage, "/Robot")
    stage.SetDefaultPrim(robot.GetPrim())
    Usd.ModelAPI(robot).SetKind(Kind.Tokens.component)
    UsdPhysics.ArticulationRootAPI.Apply(robot.GetPrim())
    rubber = UsdShade.Material.Define(stage, "/Robot/Looks/Rubber")
    for name in ("base", "arm"):
        link = UsdGeom.Xform.Define(stage, f"/Robot/{name}")
        UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
        collider = UsdGeom.Cube.Define(stage, f"/Robot/{name}/collider")
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(collider.GetPrim()).Bind(
            rubber, UsdShade.Tokens.weakerThanDescendants, "physics"
        )
    filtered = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath("/Robot/base/collider"))
    filtered.CreateFilteredPairsRel().SetTargets([Sdf.Path("/Robot/arm/collider")])
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/Robot/shoulder")
    joint.CreateBody0Rel().SetTargets([Sdf.Path("/Robot/base")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/Robot/arm")])
    joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0.2))
    joint.CreateLocalRot0Attr(Gf.Quatf(0.7071068, 0, 0, 0.7071068))
    joint.CreateLowerLimitAttr(-90.0)
    stage.GetRootLayer().Save()
    return path


def _scene(directory: Path) -> Path:
    """A scene that references, pays-for, instances and varies the asset."""
    _asset(directory / "robot.usda")
    path = directory / "scene.usda"
    stage = Usd.Stage.CreateNew(str(path))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    stage.CreateClassPrim("/_class_Robot")
    UsdGeom.Xform.Define(stage, "/_spec_Robot").GetPrim().SetSpecifier(Sdf.SpecifierClass)

    ref = UsdGeom.Xform.Define(stage, "/World/Robot_ref").GetPrim()
    ref.GetReferences().AddReference("./robot.usda")
    ref.GetInherits().AddInherit("/_class_Robot")
    ref.GetSpecializes().AddSpecialize("/_spec_Robot")
    ref.SetDocumentation("the reference under test")
    ref.SetCustomDataByKey("owner", "lane-usd-2")

    UsdGeom.Xform.Define(stage, "/World/Robot_payload").GetPrim().GetPayloads().AddPayload("./robot.usda")
    UsdGeom.Xform.Define(stage, "/World/Missing").GetPrim().GetReferences().AddReference("./nowhere.usda")

    for index in range(2):
        inst = UsdGeom.Xform.Define(stage, f"/World/Robot_inst_{index}").GetPrim()
        inst.GetReferences().AddReference("./robot.usda")
        inst.SetInstanceable(True)

    prop = UsdGeom.Xform.Define(stage, "/World/Prop").GetPrim()
    lod = prop.GetVariantSets().AddVariantSet("lod")
    for name in ("high", "low"):
        lod.AddVariant(name)
    lod.SetVariantSelection("low")

    mover = UsdGeom.Xform.Define(stage, "/World/Mover")
    op = mover.AddTranslateOp()
    op.Set(Gf.Vec3d(0, 0, 0), 0)
    op.Set(Gf.Vec3d(10, 0, 0), 10)
    matrix = UsdGeom.Xform.Define(stage, "/World/Matrix")
    matrix.AddTransformOp().Set(Gf.Matrix4d(1).SetTranslate(Gf.Vec3d(1, 2, 3)))
    stage.GetRootLayer().Save()
    return path


def _heavy_mesh(path: Path, vertices: int) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.Xform.Define(stage, "/World")
    mesh = UsdGeom.Mesh.Define(stage, "/World/Terrain")
    points = np.zeros((vertices, 3), dtype=np.float32)
    points[:, 0] = np.arange(vertices, dtype=np.float32)
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
    triangles = vertices // 3
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(triangles, 3, dtype=np.int32)))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(np.arange(triangles * 3, dtype=np.int32)))
    stage.GetRootLayer().Save()
    return path


@pytest.fixture
def scene(usd_tools, tmp_path: Path) -> str:
    return usd_tools.load(_scene(tmp_path))


def _info(usd_tools, stage_id: str, prim_path: str, **arguments):
    payload = usd_tools.call("get_prim_info", stage_id=stage_id, prim_path=prim_path, **arguments)
    assert payload.get("success", True) is True, payload
    return payload


class TestMetadata:
    def test_composition_arcs_and_authored_metadata(self, usd_tools, scene: str) -> None:
        info = _info(usd_tools, scene, "/World/Robot_ref")
        metadata = info["metadata"]

        assert metadata["specifier"] == "def"
        assert metadata["instanceable"] is False
        composition = metadata["composition"]
        assert [(arc["asset_path"], arc["prim_path"], arc["resolved"]) for arc in composition["references"]] == [
            ("./robot.usda", None, True)
        ]
        assert [arc["prim_path"] for arc in composition["inherits"]] == ["/_class_Robot"]
        assert [arc["prim_path"] for arc in composition["specializes"]] == ["/_spec_Robot"]
        assert metadata["authored"]["documentation"] == "the reference under test"
        assert metadata["authored"]["customData"] == {"owner": "lane-usd-2"}
        assert any(layer.endswith("robot.usda") for layer in metadata["opinion_layers"])
        assert info["kind"] == "component"
        assert info["api_schemas"] == ["PhysicsArticulationRootAPI"]

    def test_payload_and_unresolved_reference(self, usd_tools, scene: str) -> None:
        payload = _info(usd_tools, scene, "/World/Robot_payload")["metadata"]["composition"]
        missing = _info(usd_tools, scene, "/World/Missing")["metadata"]["composition"]

        assert payload["payloads"][0]["asset_path"] == "./robot.usda"
        assert payload["payloads"][0]["resolved"] is True
        assert missing["references"][0]["asset_path"] == "./nowhere.usda"
        assert missing["references"][0]["resolved"] is False

    def test_variant_sets_and_selection(self, usd_tools, scene: str) -> None:
        variants = _info(usd_tools, scene, "/World/Prop")["metadata"]["variant_sets"]

        assert variants == {"lod": {"selection": "low", "variants": ["high", "low"]}}

    def test_class_specifier(self, usd_tools, scene: str) -> None:
        assert _info(usd_tools, scene, "/_class_Robot")["metadata"]["specifier"] == "class"

    def test_instance_and_instance_proxy(self, usd_tools, scene: str) -> None:
        instance = _info(usd_tools, scene, "/World/Robot_inst_0")
        proxy = _info(usd_tools, scene, "/World/Robot_inst_0/shoulder")

        assert instance["is_instance"] is True
        assert instance["metadata"]["instanceable"] is True
        assert instance["prototype"].startswith("/__Prototype")
        # The default child predicate skips instance proxies; an instance has children.
        assert instance["children_count"] == 4
        assert instance["children_types"] == {"(typeless)": 1, "Xform": 2, "PhysicsRevoluteJoint": 1}
        assert proxy["is_instance_proxy"] is True
        assert proxy["prim_in_prototype"].endswith("/shoulder")


class TestRelationships:
    def test_joint_bodies(self, usd_tools, scene: str) -> None:
        relationships = _info(usd_tools, scene, "/World/Robot_ref/shoulder")["relationships"]

        assert relationships["physics:body0"] == {"targets": ["/World/Robot_ref/base"], "authored": True}
        assert relationships["physics:body1"] == {"targets": ["/World/Robot_ref/arm"], "authored": True}

    def test_material_binding_and_collision_filter(self, usd_tools, scene: str) -> None:
        relationships = _info(usd_tools, scene, "/World/Robot_ref/base/collider")["relationships"]

        assert relationships["material:binding:physics"]["targets"] == ["/World/Robot_ref/Looks/Rubber"]
        assert relationships["physics:filteredPairs"]["targets"] == ["/World/Robot_ref/arm/collider"]


class TestJsonValues:
    def test_payload_is_strict_json(self, usd_tools, scene: str) -> None:
        # usd_tools.call parses with a parser that rejects Infinity and NaN.
        attributes = _info(usd_tools, scene, "/World/Robot_ref/shoulder")["attributes"]

        assert attributes["physics:breakForce"] == {"type": "float", "value": "Infinity", "authored": False}
        assert attributes["physics:lowerLimit"] == {"type": "float", "value": -90.0, "authored": True}

    def test_vectors_quaternions_and_matrices_are_lists(self, usd_tools, scene: str) -> None:
        joint = _info(usd_tools, scene, "/World/Robot_ref/shoulder")["attributes"]
        matrix = _info(usd_tools, scene, "/World/Matrix")

        assert joint["physics:localPos0"]["value"] == pytest.approx([0.0, 0.0, 0.2])
        assert joint["physics:localPos0"]["type"] == "point3f"
        assert joint["physics:localRot0"]["value"] == pytest.approx([0.7071068, 0.0, 0.0, 0.7071068])
        assert matrix["attributes"]["xformOp:transform"]["value"] == [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 2.0, 3.0, 1.0],
        ]
        assert matrix["attributes"]["xformOpOrder"]["value"] == ["xformOp:transform"]
        assert matrix["transform"]["translation"] == [1.0, 2.0, 3.0]

    def test_fallback_values_are_marked_unauthored(self, usd_tools, scene: str) -> None:
        attributes = _info(usd_tools, scene, "/World/Robot_ref/base/collider")["attributes"]

        assert attributes["size"] == {"type": "double", "value": 2.0, "authored": False}
        assert attributes["purpose"]["authored"] is False


class TestTime:
    def test_time_code_reads_samples_and_the_transform(self, usd_tools, scene: str) -> None:
        default = _info(usd_tools, scene, "/World/Mover")
        late = _info(usd_tools, scene, "/World/Mover", time_code=10)

        translate = late["attributes"]["xformOp:translate"]
        assert translate["value"] == [10.0, 0.0, 0.0]
        assert translate["time_samples"] == 2
        assert late["transform"]["translation"] == [10.0, 0.0, 0.0]
        assert late["time_code"] == 10.0
        # No default value is authored: the default time has none.
        assert default["attributes"]["xformOp:translate"]["value"] is None
        assert default["time_code"] is None


class TestBoundedArrays:
    def test_large_mesh_is_summarised(self, usd_tools, tmp_path: Path) -> None:
        stage_id = usd_tools.load(_heavy_mesh(tmp_path / "heavy.usdc", 100_000))

        started = time.perf_counter()
        text = usd_tools.text("get_prim_info", {"stage_id": stage_id, "prim_path": "/World/Terrain"})
        elapsed = time.perf_counter() - started
        points = json.loads(text)["attributes"]["points"]

        assert len(text) < 20_000, len(text)
        assert elapsed < 5.0
        assert points["type"] == "point3f[]"
        assert points["length"] == 100_000
        assert points["truncated"] is True
        assert points["value"][:2] == [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        assert len(points["value"]) == 16

    def test_array_preview_and_attribute_filter(self, usd_tools, tmp_path: Path) -> None:
        stage_id = usd_tools.load(_heavy_mesh(tmp_path / "heavy.usdc", 3_000))

        info = _info(
            usd_tools, stage_id, "/World/Terrain", array_preview=100, attribute_filter=["points", "faceVertex*"]
        )

        assert sorted(info["attributes"]) == ["faceVertexCounts", "faceVertexIndices", "points"]
        assert len(info["attributes"]["points"]["value"]) == 100
        assert info["attribute_count"] > 3

    def test_short_arrays_are_whole(self, usd_tools, scene: str) -> None:
        order = _info(usd_tools, scene, "/World/Mover")["attributes"]["xformOpOrder"]
        only_length = _info(usd_tools, scene, "/World/Mover", array_preview=0)["attributes"]["xformOpOrder"]

        assert order == {"type": "token[]", "length": 1, "value": ["xformOp:translate"], "authored": True}
        assert only_length == {"type": "token[]", "length": 1, "value": [], "truncated": True, "authored": True}


class TestLookup:
    def test_missing_prim_and_stage(self, usd_tools, scene: str) -> None:
        missing = usd_tools.call("get_prim_info", stage_id=scene, prim_path="/World/Nope")
        unloaded = usd_tools.call("get_prim_info", stage_id="nope", prim_path="/World")

        assert missing["error_type"] == "NotFoundError"
        assert unloaded["error_type"] == "NotFoundError"
