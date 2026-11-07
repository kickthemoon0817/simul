"""search_prims and get_prim_subtree see the composed stage, page it and count it once.

The audit found search blind to instances (2 revolute joints where pxr finds 5),
``total_prims`` at 65 for a stage that composes 116, a 50k-prim search that said
50501 matches but returned 1272 with no way to reach the rest, case-sensitive
substring matching only, exact type names only (``PhysicsJoint`` and ``Gprim``
found nothing), no way to search applied API schemas, and an 18-prim robot that
took 18 get_prim_info calls (and a rate limit) to read.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

INSTANCES = 3


def _robot(path: Path) -> Path:
    stage = Usd.Stage.CreateNew(str(path))
    robot = UsdGeom.Xform.Define(stage, "/Robot")
    stage.SetDefaultPrim(robot.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(robot.GetPrim())
    for index, name in enumerate(("base_link", "link1", "link2")):
        link = UsdGeom.Xform.Define(stage, f"/Robot/{name}")
        link.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.2 * index))
        UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
        UsdPhysics.MassAPI.Apply(link.GetPrim()).CreateMassAttr(5.0 - index)
        UsdGeom.Mesh.Define(stage, f"/Robot/{name}/visuals")
        UsdPhysics.CollisionAPI.Apply(UsdGeom.Cube.Define(stage, f"/Robot/{name}/collisions").GetPrim())
    UsdGeom.Scope.Define(stage, "/Robot/joints")
    fixed = UsdPhysics.FixedJoint.Define(stage, "/Robot/joints/root_fixed")
    fixed.CreateBody1Rel().SetTargets([Sdf.Path("/Robot/base_link")])
    shoulder = UsdPhysics.RevoluteJoint.Define(stage, "/Robot/joints/shoulder")
    shoulder.CreateBody0Rel().SetTargets([Sdf.Path("/Robot/base_link")])
    shoulder.CreateBody1Rel().SetTargets([Sdf.Path("/Robot/link1")])
    UsdPhysics.DriveAPI.Apply(shoulder.GetPrim(), "angular")
    slider = UsdPhysics.PrismaticJoint.Define(stage, "/Robot/joints/slider")
    slider.CreateBody0Rel().SetTargets([Sdf.Path("/Robot/link1")])
    slider.CreateBody1Rel().SetTargets([Sdf.Path("/Robot/link2")])
    stage.GetRootLayer().Save()
    return path


def _scene(directory: Path) -> Path:
    _robot(directory / "robot.usda")
    path = directory / "scene.usda"
    stage = Usd.Stage.CreateNew(str(path))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Xform.Define(stage, "/World/Robot_ref").GetPrim().GetReferences().AddReference("./robot.usda")
    for index in range(INSTANCES):
        inst = UsdGeom.Xform.Define(stage, f"/World/Robot_inst_{index}").GetPrim()
        inst.GetReferences().AddReference("./robot.usda")
        inst.SetInstanceable(True)
    UsdGeom.Xform.Define(stage, "/World/LINK_upper")
    stage.GetRootLayer().Save()
    return path


def _wide(path: Path, groups: int, per_group: int) -> Path:
    layer = Sdf.Layer.CreateNew(str(path))
    with Sdf.ChangeBlock():
        for name, type_name in (("World", "Xform"),):
            spec = Sdf.CreatePrimInLayer(layer, f"/{name}")
            spec.specifier = Sdf.SpecifierDef
            spec.typeName = type_name
        for group in range(groups):
            spec = Sdf.CreatePrimInLayer(layer, f"/World/g{group:03d}")
            spec.specifier = Sdf.SpecifierDef
            spec.typeName = "Xform"
            for item in range(per_group):
                cube = Sdf.CreatePrimInLayer(layer, f"/World/g{group:03d}/cube_{item:03d}")
                cube.specifier = Sdf.SpecifierDef
                cube.typeName = "Cube"
    layer.Save()
    return path


def _composed_count(path: Path) -> int:
    stage = Usd.Stage.Open(str(path))
    predicate = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
    return sum(1 for _ in stage.Traverse(predicate))


@pytest.fixture
def scene_path(tmp_path: Path) -> Path:
    return _scene(tmp_path)


@pytest.fixture
def scene(usd_tools, scene_path: Path) -> str:
    return usd_tools.load(scene_path)


def _search(usd_tools, stage_id: str, search_type: str, query: str, **arguments):
    payload = usd_tools.call(
        "search_prims", stage_id=stage_id, search_type=search_type, query=query, **arguments
    )
    assert payload["success"] is True, payload
    return payload


class TestInstances:
    def test_search_finds_joints_inside_instances(self, usd_tools, scene: str) -> None:
        found = _search(usd_tools, scene, "by_type", "PhysicsRevoluteJoint")

        assert found["total_matches"] == 1 + INSTANCES
        assert "/World/Robot_inst_2/joints/shoulder" in found["results"]

    def test_instances_can_be_left_out(self, usd_tools, scene: str) -> None:
        found = _search(usd_tools, scene, "by_type", "PhysicsRevoluteJoint", include_instance_proxies=False)

        assert found["results"] == ["/World/Robot_ref/joints/shoulder"]

    def test_counts_agree_everywhere(self, usd_tools, scene_path: Path) -> None:
        loaded = usd_tools.call("load_usd_file", file_path=str(scene_path))
        everything = _search(usd_tools, loaded["stage_id"], "by_name", "", limit=5000)
        summary = usd_tools.call("summarize_scene", stage_id=loaded["stage_id"], include_meshes=False)

        expected = _composed_count(scene_path)
        assert loaded["total_prims"] == expected
        assert everything["total_matches"] == expected
        assert everything["searched"] == expected
        assert summary["summary"]["total_prims"] == expected

    def test_instance_children_are_counted(self, usd_tools, scene: str) -> None:
        info = usd_tools.call("get_prim_info", stage_id=scene, prim_path="/World/Robot_inst_0")

        assert info["children_count"] == 4


class TestMatching:
    def test_type_family(self, usd_tools, scene: str) -> None:
        joints = _search(usd_tools, scene, "by_type", "PhysicsJoint", root_path="/World/Robot_ref")
        gprims = _search(usd_tools, scene, "by_type", "Gprim", root_path="/World/Robot_ref")

        assert sorted(path.rsplit("/", 1)[1] for path in joints["results"]) == ["root_fixed", "shoulder", "slider"]
        assert gprims["total_matches"] == 6  # three visuals meshes, three collision cubes

    def test_exact_type_and_unregistered_type(self, usd_tools, scene: str) -> None:
        exact = _search(usd_tools, scene, "by_type", "PhysicsJoint", match="exact")
        custom = _search(usd_tools, scene, "by_type", "xform", root_path="/World/Robot_ref")

        assert exact["total_matches"] == 0
        assert custom["total_matches"] == 4  # Robot_ref and its three links, case-insensitively
        assert custom["notes"]

    def test_api_schemas(self, usd_tools, scene: str) -> None:
        bodies = _search(usd_tools, scene, "by_api_schema", "PhysicsRigidBodyAPI", root_path="/World/Robot_ref")
        drives = _search(usd_tools, scene, "by_api_schema", "PhysicsDriveAPI")

        assert bodies["total_matches"] == 3
        assert drives["total_matches"] == 1 + INSTANCES

    def test_names_ignore_case_unless_asked(self, usd_tools, scene: str) -> None:
        loose = _search(usd_tools, scene, "by_name", "LINK", root_path="/World/Robot_ref")
        strict = _search(usd_tools, scene, "by_name", "LINK", case_sensitive=True)

        assert loose["total_matches"] == 3
        assert strict["results"] == ["/World/LINK_upper"]

    @pytest.mark.parametrize(
        "match, query, expected",
        [
            ("glob", "link?", ["link1", "link2"]),
            ("regex", r"^(base_)?link\d?$", ["base_link", "link1", "link2"]),
            ("exact", "link1", ["link1"]),
            ("substring", "ink", ["base_link", "link1", "link2"]),
        ],
    )
    def test_match_modes(self, usd_tools, scene: str, match: str, query: str, expected: List[str]) -> None:
        found = _search(usd_tools, scene, "by_name", query, match=match, root_path="/World/Robot_ref")

        assert [path.rsplit("/", 1)[1] for path in found["results"]] == expected

    def test_legacy_exact_match(self, usd_tools, scene: str) -> None:
        found = _search(usd_tools, scene, "by_name", "link1", exact_match=True, root_path="/World/Robot_ref")

        assert found["match"] == "exact"
        assert found["results"] == ["/World/Robot_ref/link1"]

    def test_path_glob(self, usd_tools, scene: str) -> None:
        found = _search(usd_tools, scene, "by_path", "/World/Robot_inst_*/joints/*", match="glob")

        assert found["total_matches"] == 3 * INSTANCES

    def test_invalid_regex_is_a_validation_error(self, usd_tools, scene: str) -> None:
        payload = usd_tools.call("search_prims", stage_id=scene, search_type="by_name", query="(", match="regex")

        assert payload["success"] is False
        assert payload["error_type"] == "ValidationError"

    def test_unknown_root_is_not_found(self, usd_tools, scene: str) -> None:
        payload = usd_tools.call(
            "search_prims", stage_id=scene, search_type="by_name", query="", root_path="/World/Nope"
        )

        assert payload["error_type"] == "NotFoundError"


class TestPaging:
    def test_every_match_is_reachable(self, usd_tools, tmp_path: Path) -> None:
        stage_id = usd_tools.load(_wide(tmp_path / "wide.usdc", 30, 100))
        seen: List[str] = []
        offset = 0
        pages = 0
        while offset is not None:
            page = _search(usd_tools, stage_id, "by_type", "Cube", offset=offset, limit=1000)
            assert page["count"] == len(page["results"])
            assert page["total_matches"] == 3000
            seen.extend(page["results"])
            offset = page["next_offset"]
            pages += 1

        assert len(seen) == len(set(seen)) == 3000
        assert pages >= 3

    def test_default_page_fits_the_budget(self, usd_tools, tmp_path: Path) -> None:
        stage_id = usd_tools.load(_wide(tmp_path / "wide.usdc", 30, 100))

        page = _search(usd_tools, stage_id, "by_name", "")

        assert page["count"] == 200
        assert page["total_matches"] == 3031
        assert page["truncated"] is True
        assert page["next_offset"] == 200
        assert "truncation" not in page


class TestSubtree:
    def test_robot_in_one_call(self, usd_tools, scene: str) -> None:
        tree = usd_tools.call(
            "get_prim_subtree", stage_id=scene, root_path="/World/Robot_inst_1", include_properties=True
        )

        assert tree["success"] is True
        prims = {record["path"].rsplit("/World/Robot_inst_1", 1)[1] or "/": record for record in tree["prims"]}
        assert tree["total"] == len(tree["prims"]) == 14
        assert prims["/"]["is_instance"] is True
        assert prims["/"]["depth"] == 0
        assert prims["/joints/shoulder"]["is_instance_proxy"] is True
        assert prims["/joints/shoulder"]["relationships"] == {
            "physics:body0": ["/World/Robot_inst_1/base_link"],
            "physics:body1": ["/World/Robot_inst_1/link1"],
        }
        assert prims["/joints/shoulder"]["api_schemas"] == ["PhysicsDriveAPI:angular"]
        assert prims["/link1"]["attributes"]["physics:mass"] == {"type": "float", "value": 4.0, "authored": True}
        assert prims["/link1"]["child_count"] == 2

    def test_depth_and_paging(self, usd_tools, scene: str) -> None:
        shallow = usd_tools.call("get_prim_subtree", stage_id=scene, root_path="/World", max_depth=1)
        first = usd_tools.call("get_prim_subtree", stage_id=scene, max_results=10)
        second = usd_tools.call("get_prim_subtree", stage_id=scene, max_results=10, offset=first["next_offset"])

        assert [record["depth"] for record in shallow["prims"]] == [0] + [1] * (INSTANCES + 2)
        assert first["count"] == 10 and first["truncated"] is True
        assert second["offset"] == 10
        assert first["prims"][-1]["path"] != second["prims"][0]["path"]
        assert first["total"] == second["total"]


class TestSchemaEnums:
    def test_enums_are_advertised(self, usd_tools) -> None:
        import asyncio

        from fastmcp import Client

        async def _schemas():
            async with Client(usd_tools.server.mcp) as client:
                return {tool.name: tool.input_schema for tool in await client.list_tools()}

        schemas = asyncio.run(_schemas())
        search = schemas["search_prims"]["properties"]

        assert search["search_type"]["enum"] == ["by_name", "by_path", "by_type", "by_api_schema"]
        assert search["match"]["enum"] == ["auto", "exact", "substring", "glob", "regex"]
        assert schemas["summarize_scene"]["properties"]["format"]["enum"] == ["json", "text"]
