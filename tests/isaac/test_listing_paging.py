"""Listing tools page and report the cap they applied.

Every generated listing script is run against a fake stage so the paging
arithmetic is checked on the code Isaac Sim will execute, not on a mock of
it: the page honours ``offset``, ``applied_limit`` is the clamped cap,
``truncated`` is true only when another item exists past the page, and
``next_offset`` continues from where the page ended.
"""

from __future__ import annotations

import asyncio
import types
from typing import Any, Dict, List
from unittest.mock import MagicMock


from simul_mcp.config import Settings
from simul_mcp.mcp.tools.isaac_tools import IsaacTools
from tests.isaac.fake_usd import (
    FakePrim,
    FakeStage,
    capture_script,
    run_script,
    usd_modules,
)


def _stage_with_meshes(count: int) -> FakeStage:
    modules = usd_modules(FakeStage([]))
    mesh = modules["pxr"].UsdGeom.Mesh
    prims = [FakePrim("/World", "Xform")]
    prims.extend(
        FakePrim(f"/World/Mesh{i:03d}", "Mesh", kinds=(mesh,)) for i in range(count)
    )
    return FakeStage(prims)


def _run(method: str, stage: FakeStage, **kwargs: Any) -> Dict[str, Any]:
    return run_script(capture_script(method, **kwargs), usd_modules(stage))


def _paths(entries: List[Dict[str, Any]]) -> List[str]:
    return [entry["path"] for entry in entries]


# ---------------------------------------------------------------------------
# Streaming traversals: list, search, subtree, typed query
# ---------------------------------------------------------------------------


def test_list_prims_pages_through_matches() -> None:
    stage = _stage_with_meshes(7)

    first = _run(
        "list_isaac_prims", stage, root_path="/World", prim_type="Mesh", max_results=3
    )
    assert _paths(first["prims"]) == [
        "/World/Mesh000",
        "/World/Mesh001",
        "/World/Mesh002",
    ]
    assert first["applied_limit"] == 3
    assert first["offset"] == 0
    assert first["truncated"] is True
    assert first["next_offset"] == 3

    second = _run(
        "list_isaac_prims",
        stage,
        root_path="/World",
        prim_type="Mesh",
        max_results=3,
        offset=first["next_offset"],
    )
    assert _paths(second["prims"]) == [
        "/World/Mesh003",
        "/World/Mesh004",
        "/World/Mesh005",
    ]
    assert second["truncated"] is True

    last = _run(
        "list_isaac_prims",
        stage,
        root_path="/World",
        prim_type="Mesh",
        max_results=3,
        offset=6,
    )
    assert _paths(last["prims"]) == ["/World/Mesh006"]
    assert last["truncated"] is False
    assert last["next_offset"] is None


def test_list_prims_reports_the_clamped_limit() -> None:
    payload = _run("list_isaac_prims", _stage_with_meshes(2), max_results=10_000)
    assert payload["applied_limit"] == 1000
    assert payload["max_depth"] == 5


def test_list_prims_page_that_exactly_fills_is_not_truncated() -> None:
    payload = _run(
        "list_isaac_prims",
        _stage_with_meshes(3),
        root_path="/World",
        prim_type="Mesh",
        max_results=3,
    )
    assert payload["count"] == 3
    assert payload["truncated"] is False


def test_search_prims_pages_and_requires_a_query() -> None:
    stage = _stage_with_meshes(5)
    page = _run(
        "search_isaac_prims",
        stage,
        query="mesh",
        search_type="name",
        max_results=2,
        offset=2,
    )
    assert _paths(page["matches"]) == ["/World/Mesh002", "/World/Mesh003"]
    assert page["applied_limit"] == 2
    assert page["truncated"] is True
    assert page["next_offset"] == 4

    refused = asyncio.run(
        IsaacTools(MagicMock(), settings=Settings()).search_isaac_prims(query="  ")
    )
    assert refused["success"] is False
    assert refused["error_type"] == "ValueError"


def test_subtree_pages_in_traversal_order() -> None:
    stage = _stage_with_meshes(4)
    page = _run("get_isaac_subtree", stage, root_path="/World", max_results=2, offset=1)
    assert _paths(page["prims"]) == ["/World/Mesh000", "/World/Mesh001"]
    assert page["offset"] == 1
    assert page["applied_limit"] == 2
    assert page["truncated"] is True
    assert page["next_offset"] == 3


def test_typed_query_pages_and_clamps() -> None:
    stage = _stage_with_meshes(5)
    page = _run(
        "query_usd_typed_prims",
        stage,
        type_name="Mesh",
        root_path="/World",
        max_results=5000,
        offset=4,
    )
    assert _paths(page["prims"]) == ["/World/Mesh004"]
    assert page["applied_limit"] == 2000
    assert page["truncated"] is False
    assert page["next_offset"] is None


# ---------------------------------------------------------------------------
# Collected listings: materials, lights, cameras, physics objects
# ---------------------------------------------------------------------------


def test_materials_lights_cameras_report_totals_and_pages() -> None:
    modules = usd_modules(FakeStage([]))
    pxr = modules["pxr"]
    prims = [FakePrim("/World", "Xform"), FakePrim("/World/Looks", "Scope")]
    prims.extend(
        FakePrim(f"/World/Looks/Mat{i}", "Material", kinds=(pxr.UsdShade.Material,))
        for i in range(4)
    )
    prims.extend(
        FakePrim(
            f"/World/Light{i}",
            "SphereLight",
            apis=(pxr.UsdLux.LightAPI,),
            attributes={"inputs:intensity": 500.0 + i},
        )
        for i in range(3)
    )
    prims.extend(
        FakePrim(f"/World/Cam{i}", "Camera", kinds=(pxr.UsdGeom.Camera,))
        for i in range(3)
    )
    stage = FakeStage(prims)

    materials = _run("list_isaac_materials", stage, max_results=3, offset=1)
    assert _paths(materials["materials"]) == [
        "/World/Looks/Mat1",
        "/World/Looks/Mat2",
        "/World/Looks/Mat3",
    ]
    assert materials["total"] == 4
    assert materials["applied_limit"] == 3
    assert materials["truncated"] is False

    lights = _run("list_isaac_lights", stage, root_path="/World", max_results=2)
    assert lights["count"] == 2
    assert lights["total"] == 3
    assert lights["truncated"] is True
    assert lights["next_offset"] == 2
    assert lights["lights"][0]["intensity"] == 500.0

    cameras = _run("list_isaac_cameras", stage, max_results=1, offset=2)
    assert _paths(cameras["cameras"]) == ["/World/Cam2"]
    assert cameras["total"] == 3
    assert cameras["truncated"] is False


def test_physics_objects_page_each_list_and_report_full_counts() -> None:
    modules = usd_modules(FakeStage([]))
    physics = modules["pxr"].UsdPhysics
    prims = [FakePrim("/World", "Xform")]
    prims.extend(
        FakePrim(
            f"/World/Body{i}", "Cube", apis=(physics.RigidBodyAPI, physics.CollisionAPI)
        )
        for i in range(5)
    )
    prims.append(
        FakePrim("/World/Joint0", "PhysicsRevoluteJoint", kinds=(physics.Joint,))
    )
    stage = FakeStage(prims)

    page = _run(
        "list_isaac_physics_objects", stage, root_path="/World", max_results=2, offset=2
    )
    assert page["rigid_body_count"] == 5
    assert page["collider_count"] == 5
    assert page["joint_count"] == 1
    assert _paths(page["rigid_bodies"]) == ["/World/Body2", "/World/Body3"]
    assert page["joints"] == []
    assert page["applied_limit"] == 2
    assert page["truncated"] is True
    assert page["next_offset"] == 4


# ---------------------------------------------------------------------------
# Extensions: limit, offset, total, and a string version
# ---------------------------------------------------------------------------


def _kit_modules(extensions: List[Dict[str, Any]]) -> Dict[str, Any]:
    manager = types.SimpleNamespace(get_extensions=lambda: extensions)
    app = types.SimpleNamespace(get_extension_manager=lambda: manager)
    omni = types.ModuleType("omni")
    omni_kit = types.ModuleType("omni.kit")
    omni_kit_app = types.ModuleType("omni.kit.app")
    omni_kit_app.get_app = lambda: app  # type: ignore[attr-defined]
    omni.kit = omni_kit  # type: ignore[attr-defined]
    omni_kit.app = omni_kit_app  # type: ignore[attr-defined]
    return {"omni": omni, "omni.kit": omni_kit, "omni.kit.app": omni_kit_app}


def test_extensions_page_and_render_version_as_a_string() -> None:
    extensions = [
        {
            "id": f"omni.ext{i:02d}-1.{i}.0",
            "enabled": True,
            "version": [1, i, 0, "", ""],
        }
        for i in range(6)
    ]
    extensions.append(
        {
            "id": "omni.physx-107.3.3-rc.1",
            "enabled": True,
            "version": [107, 3, 3, "rc.1", ""],
        }
    )
    modules = _kit_modules(extensions)

    page = run_script(
        capture_script("list_isaac_extensions", limit=3, offset=4), modules
    )
    assert [ext["id"] for ext in page["extensions"]] == [
        "omni.ext04-1.4.0",
        "omni.ext05-1.5.0",
        "omni.physx-107.3.3-rc.1",
    ]
    assert page["total"] == 7
    assert page["applied_limit"] == 3
    assert page["truncated"] is False
    assert [ext["version"] for ext in page["extensions"]] == [
        "1.4.0",
        "1.5.0",
        "107.3.3-rc.1",
    ]

    clamped = run_script(capture_script("list_isaac_extensions", limit=99_999), modules)
    assert clamped["applied_limit"] == 1000
    assert clamped["count"] == 7


# ---------------------------------------------------------------------------
# Scene summary and scene stats count the same prims
# ---------------------------------------------------------------------------


def test_summary_and_stats_agree_on_prim_counts() -> None:
    stage = _stage_with_meshes(3)
    stage.by_path["/World"].children.append(FakePrim("/World/Untyped", ""))
    stage.by_path["/World/Untyped"] = stage.by_path["/World"].children[-1]

    summary = _run("get_isaac_scene_summary", stage)
    stats = _run("get_isaac_scene_stats", stage)

    assert summary["total_prims"] == stats["total_prims"] == 5
    assert summary["type_counts"]["Typeless"] == stats["type_counts"]["Typeless"] == 1
    assert summary["type_counts"]["Mesh"] == stats["type_counts"]["Mesh"] == 3


# ---------------------------------------------------------------------------
# Physics scene: existing scenes take the requested gravity
# ---------------------------------------------------------------------------


def test_physics_scene_updates_gravity_on_an_existing_scene() -> None:
    modules = usd_modules(FakeStage([]))
    scene_kind = modules["pxr"].UsdPhysics.Scene
    stage = FakeStage(
        [
            FakePrim("/World", "Xform"),
            FakePrim("/World/PhysicsScene", "PhysicsScene", kinds=(scene_kind,)),
        ]
    )
    modules = usd_modules(stage)

    updated = run_script(
        capture_script(
            "create_isaac_physics_scene",
            gravity_magnitude=3.7,
            gravity_direction=[0, 0, -1],
        ),
        modules,
    )
    assert updated["already_existed"] is True
    assert updated["updated"] is True
    assert updated["created"] is False
    assert updated["gravity_magnitude"] == 3.7
    assert updated["gravity_direction"] == [0.0, 0.0, -1.0]

    untouched = run_script(capture_script("create_isaac_physics_scene"), modules)
    assert untouched["updated"] is False
    assert untouched["gravity_magnitude"] == 3.7


def test_physics_scene_creates_with_z_up_defaults() -> None:
    stage = FakeStage([FakePrim("/World", "Xform")])
    created = run_script(
        capture_script("create_isaac_physics_scene"), usd_modules(stage)
    )
    assert created["created"] is True
    assert created["updated"] is False
    assert created["gravity_direction"] == [0.0, 0.0, -1.0]
    assert created["gravity_magnitude"] == 9.81
