"""Physics tools author what the physics parser reads, and say what they did.

Each test runs a generated script against a real ``Usd.Stage`` (see
``conftest.run_on_stage``) and, where the physics meaning is the point, reads
the result back through ``UsdPhysics.LoadUsdPhysicsFromRange``, the reference
UsdPhysics parser, rather than through the attributes the script wrote. The
defects pinned here all reported success:

- ``create_isaac_physics_scene`` wrote Z-up metre gravity on every stage (on a
  Y-up centimetre stage: sideways and 100x too weak) and retyped an existing
  Xform into a PhysicsScene;
- ``set_isaac_physics_material`` put MaterialAPI on the collider itself, or
  created a material nothing bound, so no shape had a material;
- ``add_isaac_collision`` put CollisionAPI on an asset-root Xform, which yields
  no collision shape, and accepted any approximation token;
- ``get_isaac_physics_scene`` promised solver settings and returned gravity;
- joint data carried ``Infinity``, which is not JSON.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Tuple

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from khemoo.simul import prim_detail as bridge_prim_detail
from simul.mcp.registration._helpers import apply_success_from_error
from simul.mcp.tools.isaac_tools import IsaacTools

Captured = Tuple[IsaacTools, List[str]]


def _stage(up_axis: str = "Z", meters_per_unit: float = 1.0) -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, up_axis)
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    UsdGeom.Xform.Define(stage, "/World")
    return stage


def _call(
    capturing_tools: Captured, run_on_stage: Any, stage: Usd.Stage, method: str, **kwargs: Any
) -> Dict[str, Any]:
    """Run one tool against ``stage``; a call refused before sending returns as is."""
    tools, captured = capturing_tools
    sent = len(captured)
    payload = asyncio.run(getattr(tools, method)(**kwargs))
    if len(captured) == sent:
        return payload
    return apply_success_from_error(run_on_stage(captured[-1], stage))


def _parsed(stage: Usd.Stage) -> Dict[str, Tuple[List[Sdf.Path], List[Any]]]:
    """The reference parser's view of the stage, keyed by object type name."""
    # OpenUSD 26 renamed LoadUsdPhysicsFromRange and deprecated the old name.
    load = getattr(UsdPhysics, "UsdPhysicsLoadStageFromPrimRange", None) or UsdPhysics.LoadUsdPhysicsFromRange
    parsed = load(stage, ["/"])
    return {str(kind): (list(paths), list(descs)) for kind, (paths, descs) in parsed.items()}


def _shapes(stage: Usd.Stage) -> Dict[str, Any]:
    """Collision shape descriptors by prim path."""
    shapes: Dict[str, Any] = {}
    for kind, (paths, descs) in _parsed(stage).items():
        if kind.endswith("Shape"):
            for path, desc in zip(paths, descs):
                shapes[str(path)] = desc
    return shapes


def _strict_json(payload: Dict[str, Any]) -> None:
    """Fail when the payload holds a value JSON cannot carry (inf, nan)."""
    json.dumps(payload, allow_nan=False)


def _mesh(stage: Usd.Stage, path: str) -> UsdGeom.Mesh:
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
    mesh.CreateFaceVertexCountsAttr([3, 3, 3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3])
    return mesh


# ---------------------------------------------------------------------------
# F04: create_isaac_physics_scene follows the stage's up axis and units
# ---------------------------------------------------------------------------


def test_new_scene_gravity_follows_a_y_up_centimetre_stage(
    capturing_tools: Captured, run_on_stage: Any
) -> None:
    stage = _stage("Y", 0.01)

    result = _call(capturing_tools, run_on_stage, stage, "create_isaac_physics_scene")

    assert result["success"] is True and result["created"] is True
    assert result["gravity_direction"] == [0.0, -1.0, 0.0]
    assert result["gravity_magnitude"] == pytest.approx(981.0)
    assert result["gravity_magnitude_m_per_s2"] == pytest.approx(9.81)
    assert result["stage_up_axis"] == "Y"
    (_paths, (scene,)) = _parsed(stage)["Scene"]
    assert tuple(scene.gravityDirection) == pytest.approx((0.0, -1.0, 0.0))
    assert scene.gravityMagnitude == pytest.approx(981.0)


def test_new_scene_on_a_z_up_metre_stage_keeps_earth_gravity_down_z(
    capturing_tools: Captured, run_on_stage: Any
) -> None:
    stage = _stage()

    result = _call(capturing_tools, run_on_stage, stage, "create_isaac_physics_scene")

    assert result["gravity_direction"] == [0.0, 0.0, -1.0]
    assert result["gravity_magnitude"] == pytest.approx(9.81)
    (_paths, (scene,)) = _parsed(stage)["Scene"]
    assert scene.gravityMagnitude == pytest.approx(9.81)


def test_requested_gravity_is_m_per_s2_converted_and_normalised(
    capturing_tools: Captured, run_on_stage: Any
) -> None:
    stage = _stage("Z", 0.01)

    result = _call(
        capturing_tools, run_on_stage, stage, "create_isaac_physics_scene",
        gravity_direction=[0, 0, -2], gravity_magnitude=3.7,
    )

    assert result["gravity_direction"] == [0.0, 0.0, -1.0]
    assert result["gravity_magnitude"] == pytest.approx(370.0)
    assert result["gravity_magnitude_m_per_s2"] == pytest.approx(3.7)
    scene = UsdPhysics.Scene(stage.GetPrimAtPath("/World/PhysicsScene"))
    assert scene.GetGravityMagnitudeAttr().Get() == pytest.approx(370.0)


def test_existing_scene_takes_requested_gravity_and_keeps_the_rest(
    capturing_tools: Captured, run_on_stage: Any
) -> None:
    stage = _stage()
    UsdPhysics.Scene.Define(stage, "/World/PhysicsScene").CreateGravityMagnitudeAttr(5.0)

    updated = _call(
        capturing_tools, run_on_stage, stage, "create_isaac_physics_scene", gravity_magnitude=3.7
    )
    untouched = _call(capturing_tools, run_on_stage, stage, "create_isaac_physics_scene")

    assert updated["already_existed"] is True and updated["updated"] is True
    assert updated["gravity_magnitude"] == pytest.approx(3.7)
    # Unauthored, the direction is UsdPhysics' "negative up axis".
    assert updated["gravity_direction"] == [0.0, 0.0, -1.0]
    assert untouched["updated"] is False
    assert untouched["gravity_magnitude"] == pytest.approx(3.7)


def test_an_existing_xform_is_refused_not_retyped(
    capturing_tools: Captured, run_on_stage: Any
) -> None:
    stage = _stage()
    robot = UsdGeom.Xform.Define(stage, "/World/Robot")
    robot.AddTranslateOp().Set(Gf.Vec3d(1, 2, 3))
    UsdGeom.Cube.Define(stage, "/World/Robot/body")

    result = _call(
        capturing_tools, run_on_stage, stage, "create_isaac_physics_scene", prim_path="/World/Robot"
    )

    assert result["success"] is False
    assert result["error_type"] == "RefusedOperation"
    assert result["existing_type"] == "Xform"
    prim = stage.GetPrimAtPath("/World/Robot")
    assert prim.GetTypeName() == "Xform"
    assert prim.IsA(UsdGeom.Xformable)
    assert "Scene" not in _parsed(stage)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gravity_direction": [0, 0, 0]},
        {"gravity_direction": [0, -1]},
        {"gravity_magnitude": -9.81},
        {"gravity_magnitude": float("inf")},
    ],
)
def test_unusable_gravity_is_refused_before_sending(
    capturing_tools: Captured, kwargs: Dict[str, Any]
) -> None:
    tools, captured = capturing_tools
    result = asyncio.run(tools.create_isaac_physics_scene(**kwargs))
    assert result["success"] is False and result["error_type"] == "ValidationError"
    assert captured == []


# ---------------------------------------------------------------------------
# F18: the PhysX scene settings are read and written
# ---------------------------------------------------------------------------


def test_physics_scene_report_reads_authored_physx_settings(
    capturing_tools: Captured, run_on_stage: Any
) -> None:
    stage = _stage("Y", 0.01)
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene").GetPrim()
    scene.AddAppliedSchema("PhysxSceneAPI")
    scene.CreateAttribute("physxScene:timeStepsPerSecond", Sdf.ValueTypeNames.UInt, False).Set(120)
    scene.CreateAttribute("physxScene:solverType", Sdf.ValueTypeNames.Token, False).Set("PGS")
    scene.CreateAttribute("physxScene:enableGPUDynamics", Sdf.ValueTypeNames.Bool, False).Set(False)

    report = _call(capturing_tools, run_on_stage, stage, "get_isaac_physics_scene")

    assert report["stage_up_axis"] == "Y"
    assert report["stage_meters_per_unit"] == pytest.approx(0.01)
    (entry,) = report["scenes"]
    assert entry["physx_scene_api"] is True
    assert entry["time_steps_per_second"] == 120
    assert entry["timestep_seconds"] == pytest.approx(1 / 120)
    assert entry["solver_type"] == "PGS"
    assert entry["enable_gpu_dynamics"] is False
    # usd-core has no PhysxSchema, so what is not authored reads as unknown.
    assert entry["broadphase_type"] is None
    assert sorted(entry["authored_settings"]) == [
        "enable_gpu_dynamics", "solver_type", "time_steps_per_second",
    ]
    # Gravity is unauthored: reported as the parser resolves it.
    assert entry["gravity_direction"] == [0.0, -1.0, 0.0]
    assert entry["gravity_magnitude"] == pytest.approx(981.0)
    assert entry["gravity_from_stage_defaults"] is True
    _strict_json(report)


def test_configure_authors_physx_settings_with_the_schema_types(
    capturing_tools: Captured, run_on_stage: Any
) -> None:
    stage = _stage()
    UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")

    result = _call(
        capturing_tools, run_on_stage, stage, "configure_isaac_physics_scene",
        time_steps_per_second=240, solver_type="pgs", enable_gpu_dynamics=False,
        broadphase_type="mbp", enable_enhanced_determinism=True, enable_ccd=True,
        min_position_iterations=4, max_position_iterations=32, max_velocity_iterations=4,
    )

    assert result["success"] is True, result
    assert result["prim_path"] == "/World/PhysicsScene"
    assert result["time_steps_per_second"] == 240
    assert result["timestep_seconds"] == pytest.approx(1 / 240)
    assert result["solver_type"] == "PGS"
    assert result["broadphase_type"] == "MBP"
    assert result["enable_enhanced_determinism"] is True
    assert result["min_position_iterations"] == 4
    prim = stage.GetPrimAtPath("/World/PhysicsScene")
    # usd-core has no PhysxSchema, so GetAppliedSchemas() filters the name out.
    assert "PhysxSceneAPI" in prim.GetPrimTypeInfo().GetAppliedAPISchemas()
    assert result["physx_scene_api"] is True
    types = {
        name: str(prim.GetAttribute(name).GetTypeName())
        for name in (
            "physxScene:timeStepsPerSecond", "physxScene:solverType",
            "physxScene:enableGPUDynamics", "physxScene:maxPositionIterationCount",
        )
    }
    assert types == {
        "physxScene:timeStepsPerSecond": "uint",
        "physxScene:solverType": "token",
        "physxScene:enableGPUDynamics": "bool",
        "physxScene:maxPositionIterationCount": "uint",
    }
    assert not prim.GetAttribute("physxScene:timeStepsPerSecond").IsCustom()


def test_configure_picks_the_only_scene_and_asks_when_there_are_several(
    capturing_tools: Captured, run_on_stage: Any
) -> None:
    stage = _stage()
    none = _call(
        capturing_tools, run_on_stage, stage, "configure_isaac_physics_scene", time_steps_per_second=90
    )
    UsdPhysics.Scene.Define(stage, "/World/A")
    UsdPhysics.Scene.Define(stage, "/World/B")
    several = _call(
        capturing_tools, run_on_stage, stage, "configure_isaac_physics_scene", time_steps_per_second=90
    )
    chosen = _call(
        capturing_tools, run_on_stage, stage, "configure_isaac_physics_scene",
        prim_path="/World/B", time_steps_per_second=90,
    )
    not_a_scene = _call(
        capturing_tools, run_on_stage, stage, "configure_isaac_physics_scene",
        prim_path="/World", time_steps_per_second=90,
    )

    assert none["success"] is False and "create_isaac_physics_scene" in none["error"]
    assert several["success"] is False and several["scenes"] == ["/World/A", "/World/B"]
    assert chosen["success"] is True and chosen["time_steps_per_second"] == 90
    assert not stage.GetPrimAtPath("/World/A").GetAttribute("physxScene:timeStepsPerSecond").IsValid()
    assert not_a_scene["success"] is False


def test_configure_refuses_a_minimum_above_the_authored_maximum(
    capturing_tools: Captured, run_on_stage: Any
) -> None:
    stage = _stage()
    UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    _call(
        capturing_tools, run_on_stage, stage, "configure_isaac_physics_scene", max_position_iterations=8
    )

    result = _call(
        capturing_tools, run_on_stage, stage, "configure_isaac_physics_scene", min_position_iterations=16
    )

    assert result["success"] is False and "nothing was changed" in result["error"]
    prim = stage.GetPrimAtPath("/World/PhysicsScene")
    assert not prim.GetAttribute("physxScene:minPositionIterationCount").IsValid()


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"solver_type": "Jacobi"},
        {"broadphase_type": "octree"},
        {"time_steps_per_second": 0},
        {"max_position_iterations": 300},
        {"min_position_iterations": 0},
        {"min_velocity_iterations": 9, "max_velocity_iterations": 2},
    ],
)
def test_configure_refuses_bad_settings_before_sending(
    capturing_tools: Captured, kwargs: Dict[str, Any]
) -> None:
    tools, captured = capturing_tools
    result = asyncio.run(tools.configure_isaac_physics_scene(**kwargs))
    assert result["success"] is False and result["error_type"] == "ValidationError"
    assert captured == []


# ---------------------------------------------------------------------------
# F05: physics materials are bound to colliders with purpose physics
# ---------------------------------------------------------------------------


def _box_with_visual_material(stage: Usd.Stage, path: str = "/World/Box") -> Usd.Prim:
    box = UsdGeom.Cube.Define(stage, path).GetPrim()
    UsdPhysics.CollisionAPI.Apply(box)
    look = UsdShade.Material.Define(stage, "/World/Looks/Red")
    UsdShade.MaterialBindingAPI.Apply(box).Bind(look)
    return box


def test_physics_material_on_a_collider_is_bound_for_physics_only(
    capturing_tools: Captured, run_on_stage: Any
) -> None:
    stage = _stage()
    box = _box_with_visual_material(stage)

    result = _call(
        capturing_tools, run_on_stage, stage, "set_isaac_physics_material",
        prim_path="/World/Box", static_friction=1.2, dynamic_friction=1.0,
    )

    material_path = "/World/PhysicsMaterials/Box_PhysicsMaterial"
    assert result["success"] is True, result
    assert result["material_path"] == material_path
    assert result["material_created"] is True
    assert result["bound_to"] == ["/World/Box"]
    assert result["static_friction"] == pytest.approx(1.2)
    assert result["restitution"] == 0.0
    assert _shapes(stage)["/World/Box"].materials == [Sdf.Path(material_path)]
    (_paths, (material,)) = _parsed(stage)["RigidBodyMaterial"]
    assert material.staticFriction == pytest.approx(1.2)
    assert material.dynamicFriction == pytest.approx(1.0)
    # The collider is not itself turned into a material, and its look stays.
    assert not box.HasAPI(UsdPhysics.MaterialAPI)
    visual, _ = UsdShade.MaterialBindingAPI(box).ComputeBoundMaterial()
    assert str(visual.GetPath()) == "/World/Looks/Red"


def test_physics_material_on_an_asset_root_binds_each_collider(
    capturing_tools: Captured, run_on_stage: Any
) -> None:
    stage = _stage()
    UsdGeom.Xform.Define(stage, "/World/Prop")
    UsdPhysics.CollisionAPI.Apply(_mesh(stage, "/World/Prop/hull").GetPrim())
    UsdPhysics.CollisionAPI.Apply(UsdGeom.Cube.Define(stage, "/World/Prop/base").GetPrim())
    _mesh(stage, "/World/Prop/visual")  # renders only; nothing to bind

    result = _call(
        capturing_tools, run_on_stage, stage, "set_isaac_physics_material",
        prim_path="/World/Prop", static_friction=0.1, material_path="/World/Materials/Ice",
    )

    assert result["bound_to"] == ["/World/Prop/hull", "/World/Prop/base"]
    shapes = _shapes(stage)
    assert shapes["/World/Prop/hull"].materials == [Sdf.Path("/World/Materials/Ice")]
    assert shapes["/World/Prop/base"].materials == [Sdf.Path("/World/Materials/Ice")]
    assert stage.GetPrimAtPath("/World/Materials").GetTypeName() == "Scope"


def test_physics_material_without_colliders_fails_and_creates_nothing(
    capturing_tools: Captured, run_on_stage: Any
) -> None:
    stage = _stage()
    UsdGeom.Cube.Define(stage, "/World/Box")

    result = _call(
        capturing_tools, run_on_stage, stage, "set_isaac_physics_material", prim_path="/World/Box"
    )

    assert result["success"] is False and "add_isaac_collision" in result["error"]
    assert not stage.GetPrimAtPath("/World/PhysicsMaterials").IsValid()


def test_a_material_path_alone_creates_an_unbound_material_and_says_so(
    capturing_tools: Captured, run_on_stage: Any
) -> None:
    stage = _stage()

    result = _call(
        capturing_tools, run_on_stage, stage, "set_isaac_physics_material",
        prim_path="/World/PhysMat", static_friction=1.2,
    )

    assert result["success"] is True
    assert result["bound_to"] == [] and result["bound_count"] == 0
    assert "No collider was bound" in result["warning"]
    prim = stage.GetPrimAtPath("/World/PhysMat")
    assert prim.IsA(UsdShade.Material) and prim.HasAPI(UsdPhysics.MaterialAPI)
    assert result["dynamic_friction"] == 0.5


def test_reusing_a_material_keeps_the_values_not_passed(
    capturing_tools: Captured, run_on_stage: Any
) -> None:
    stage = _stage()
    _box_with_visual_material(stage)
    material = UsdShade.Material.Define(stage, "/World/Rubber").GetPrim()
    api = UsdPhysics.MaterialAPI.Apply(material)
    api.CreateStaticFrictionAttr(0.9)
    api.CreateDynamicFrictionAttr(0.8)

    result = _call(
        capturing_tools, run_on_stage, stage, "set_isaac_physics_material",
        prim_path="/World/Box", material_path="/World/Rubber", restitution=0.6,
    )

    assert result["material_created"] is False
    assert result["static_friction"] == pytest.approx(0.9)
    assert result["dynamic_friction"] == pytest.approx(0.8)
    assert result["restitution"] == pytest.approx(0.6)


def test_rebinding_reports_the_physics_material_it_replaced(
    capturing_tools: Captured, run_on_stage: Any
) -> None:
    stage = _stage()
    box = _box_with_visual_material(stage)
    old = UsdShade.Material.Define(stage, "/World/Old")
    UsdPhysics.MaterialAPI.Apply(old.GetPrim())
    UsdShade.MaterialBindingAPI(box).Bind(old, UsdShade.Tokens.weakerThanDescendants, "physics")

    result = _call(
        capturing_tools, run_on_stage, stage, "set_isaac_physics_material",
        prim_path="/World/Box", material_path="/World/New",
    )

    assert result["replaced_bindings"] == {"/World/Box": "/World/Old"}
    assert _shapes(stage)["/World/Box"].materials == [Sdf.Path("/World/New")]


def test_a_non_material_at_material_path_is_refused(
    capturing_tools: Captured, run_on_stage: Any
) -> None:
    stage = _stage()
    _box_with_visual_material(stage)
    UsdGeom.Xform.Define(stage, "/World/NotAMaterial")

    result = _call(
        capturing_tools, run_on_stage, stage, "set_isaac_physics_material",
        prim_path="/World/Box", material_path="/World/NotAMaterial",
    )

    assert result["success"] is False and "not a Material" in result["error"]


@pytest.mark.parametrize(
    "kwargs",
    [{"static_friction": -0.1}, {"dynamic_friction": float("nan")}, {"restitution": 1.5}],
)
def test_impossible_material_values_are_refused_before_sending(
    capturing_tools: Captured, kwargs: Dict[str, Any]
) -> None:
    tools, captured = capturing_tools
    result = asyncio.run(tools.set_isaac_physics_material("/World/Box", **kwargs))
    assert result["success"] is False and result["error_type"] == "ValidationError"
    assert captured == []


# ---------------------------------------------------------------------------
# F06: collision goes on geometry, approximations are validated
# ---------------------------------------------------------------------------


def test_collision_on_an_asset_root_lands_on_its_geometry(
    capturing_tools: Captured, run_on_stage: Any
) -> None:
    stage = _stage()
    UsdGeom.Xform.Define(stage, "/World/Prop")
    _mesh(stage, "/World/Prop/hull")
    UsdGeom.Cube.Define(stage, "/World/Prop/base")
    UsdGeom.Xform.Define(stage, "/World/Prop/handle")
    UsdGeom.Sphere.Define(stage, "/World/Prop/handle/knob")

    result = _call(
        capturing_tools, run_on_stage, stage, "add_isaac_collision",
        prim_path="/World/Prop", approximation="convexHull",
    )

    assert result["success"] is True and result["shape_count"] == 3
    assert [shape["path"] for shape in result["shapes"]] == [
        "/World/Prop/hull", "/World/Prop/base", "/World/Prop/handle/knob",
    ]
    assert not stage.GetPrimAtPath("/World/Prop").HasAPI(UsdPhysics.CollisionAPI)
    shapes = _shapes(stage)
    assert set(shapes) == {"/World/Prop/hull", "/World/Prop/base", "/World/Prop/handle/knob"}
    assert shapes["/World/Prop/hull"].approximation == "convexHull"
    assert not stage.GetPrimAtPath("/World/Prop/base").HasAPI(UsdPhysics.MeshCollisionAPI)


def test_collision_on_geometry_free_prims_is_an_error(
    capturing_tools: Captured, run_on_stage: Any
) -> None:
    stage = _stage()
    UsdGeom.Xform.Define(stage, "/World/Empty")
    UsdGeom.Xform.Define(stage, "/World/Prop")
    UsdGeom.Cube.Define(stage, "/World/Prop/base")

    empty = _call(capturing_tools, run_on_stage, stage, "add_isaac_collision", prim_path="/World/Empty")
    flat = _call(
        capturing_tools, run_on_stage, stage, "add_isaac_collision",
        prim_path="/World/Prop", recursive=False,
    )

    assert empty["success"] is False and "No collision geometry" in empty["error"]
    assert flat["success"] is False and "recursive=true" in flat["error"]
    assert _shapes(stage) == {}


def test_an_unknown_approximation_is_refused_before_sending(capturing_tools: Captured) -> None:
    tools, captured = capturing_tools
    result = asyncio.run(tools.add_isaac_collision("/World/Mesh", approximation="convex"))
    assert result["success"] is False and "convexHull" in result["error"]
    assert captured == []


def test_a_triangle_mesh_on_a_dynamic_body_is_warned_about(
    capturing_tools: Captured, run_on_stage: Any
) -> None:
    stage = _stage()
    body = UsdGeom.Xform.Define(stage, "/World/Crate").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    _mesh(stage, "/World/Crate/mesh")

    result = _call(capturing_tools, run_on_stage, stage, "add_isaac_collision", prim_path="/World/Crate")

    assert result["success"] is True
    assert result["shapes"][0]["approximation"] == "none"
    assert "convexHull" in result["warnings"][0]


def test_create_isaac_object_reports_a_collision_step_with_no_geometry(
    capturing_tools: Captured, run_on_stage: Any
) -> None:
    stage = _stage()

    result = _call(
        capturing_tools, run_on_stage, stage, "create_isaac_object",
        prim_path="/World/Group", prim_type="Xform", collision="convexHull",
    )

    assert result["success"] is False
    assert result["failed_step"] == "collision"
    assert list(result["steps"]) == ["prim"]


# ---------------------------------------------------------------------------
# Joints: drives, limits, articulations, and JSON-safe infinities
# ---------------------------------------------------------------------------


def _robot(stage: Usd.Stage, fixed: bool = True) -> None:
    """A five-link arm whose joints are defined out of kinematic order."""
    robot = UsdGeom.Xform.Define(stage, "/World/Robot").GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(robot)
    for name in ("base", "link1", "link2", "link3", "link1b"):
        link = UsdGeom.Xform.Define(stage, f"/World/Robot/{name}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(link)
        UsdPhysics.CollisionAPI.Apply(UsdGeom.Cube.Define(stage, f"/World/Robot/{name}/geom").GetPrim())
    UsdGeom.Scope.Define(stage, "/World/Robot/joints")

    def joint(name: str, body0: str | None, body1: str, kind: Any = UsdPhysics.RevoluteJoint) -> Any:
        defined = kind.Define(stage, f"/World/Robot/joints/{name}")
        if body0:
            defined.CreateBody0Rel().SetTargets([f"/World/Robot/{body0}"])
        defined.CreateBody1Rel().SetTargets([f"/World/Robot/{body1}"])
        return defined

    joint("a_wrist", "link2", "link3")
    if fixed:
        joint("z_root", None, "base", UsdPhysics.FixedJoint)
    joint("m_shoulder", "base", "link1")
    joint("b_elbow", "link1", "link2", UsdPhysics.PrismaticJoint)
    joint("c_branch", "base", "link1b")


def test_articulation_joints_come_in_kinematic_order(
    capturing_tools: Captured, run_on_stage: Any
) -> None:
    stage = _stage()
    UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    _robot(stage)

    result = _call(capturing_tools, run_on_stage, stage, "list_isaac_articulations")

    assert result["success"] is True and result["articulation_count"] == 1
    (articulation,) = result["articulations"]
    names = [joint["prim_path"].rsplit("/", 1)[1] for joint in articulation["joints"]]
    assert names == ["z_root", "m_shoulder", "b_elbow", "a_wrist", "c_branch"]
    assert [joint["index"] for joint in articulation["joints"]] == [0, 1, 2, 3, 4]
    assert articulation["fixed_base"] is True
    assert articulation["root_body"] == "/World/Robot/base"
    assert articulation["bodies"] == [
        "/World/Robot/base", "/World/Robot/link1", "/World/Robot/link2",
        "/World/Robot/link3", "/World/Robot/link1b",
    ]
    elbow = articulation["joints"][2]
    assert (elbow["parent_body"], elbow["child_body"]) == ("/World/Robot/link1", "/World/Robot/link2")
    assert elbow["body0"] == ["/World/Robot/link1"] and elbow["joint_type"] == "PhysicsPrismaticJoint"
    (_paths, (parsed,)) = _parsed(stage)["Articulation"]
    assert {str(path) for path in parsed.articulatedJoints} == {
        joint["prim_path"] for joint in articulation["joints"]
    }
    _strict_json(result)


def test_a_floating_base_starts_at_the_body_no_joint_leads_to(
    capturing_tools: Captured, run_on_stage: Any
) -> None:
    stage = _stage()
    _robot(stage, fixed=False)
    loop = UsdPhysics.RevoluteJoint.Define(stage, "/World/Robot/joints/x_loop")
    loop.CreateBody0Rel().SetTargets(["/World/Robot/link3"])
    loop.CreateBody1Rel().SetTargets(["/World/Robot/link1"])  # reached before link3
    excluded = UsdPhysics.RevoluteJoint.Define(stage, "/World/Robot/joints/y_excluded")
    excluded.CreateBody0Rel().SetTargets(["/World/Robot/link3"])
    excluded.CreateBody1Rel().SetTargets(["/World/Robot/base"])
    excluded.CreateExcludeFromArticulationAttr(True)

    (articulation,) = _call(
        capturing_tools, run_on_stage, stage, "list_isaac_articulations", root_path="/World"
    )["articulations"]

    assert articulation["fixed_base"] is False
    assert articulation["root_body"] == "/World/Robot/base"
    assert articulation["joints"][0]["prim_path"] == "/World/Robot/joints/m_shoulder"
    assert articulation["loop_closure_joints"] == ["/World/Robot/joints/x_loop"]
    assert articulation["excluded_joints"] == ["/World/Robot/joints/y_excluded"]


def test_joint_aspect_reports_infinite_bounds_as_null_with_a_note(
    capturing_tools: Captured, run_on_stage: Any
) -> None:
    stage = _stage()
    UsdPhysics.RevoluteJoint.Define(stage, "/World/Hinge").CreateAxisAttr("Y")

    joint = _call(capturing_tools, run_on_stage, stage, "get_isaac_joint_info", prim_path="/World/Hinge")

    _strict_json(joint)
    assert joint["break_force"] is None and joint["break_force_note"] == "unbreakable (inf)"
    assert joint["limits"] == {
        "type": "revolute", "axis": "Y",
        "lower": None, "lower_note": "unlimited (-inf)",
        "upper": None, "upper_note": "unlimited (inf)",
    }
    assert joint["drives"] == {} and joint["limit_axes"] == {}


def test_bridge_and_script_joint_readers_agree_on_drives_and_limit_axes(
    capturing_tools: Captured, run_on_stage: Any
) -> None:
    stage = _stage()
    joint = UsdPhysics.Joint.Define(stage, "/World/D6").GetPrim()
    UsdPhysics.LimitAPI.Apply(joint, "rotX").CreateLowAttr(10.0)
    UsdPhysics.LimitAPI(joint, "rotX").CreateHighAttr(-10.0)
    drive = UsdPhysics.DriveAPI.Apply(joint, "rotY")
    drive.CreateStiffnessAttr(50.0)
    drive.CreateTypeAttr("acceleration")

    from_script = _call(capturing_tools, run_on_stage, stage, "get_isaac_joint_info", prim_path="/World/D6")
    from_bridge = bridge_prim_detail.PrimDetailReader(stage).read(joint, "joint")

    from_script.pop("success")
    assert json.loads(json.dumps(from_bridge, allow_nan=False)) == from_script
    assert from_script["limit_axes"]["rotX"]["locked"] is True
    assert from_script["drives"]["rotY"]["type"] == "acceleration"
    assert from_script["drives"]["rotY"]["max_force"] is None
    assert from_script["drives"]["rotY"]["max_force_note"] == "unlimited (inf)"


def test_set_joint_drive_is_what_the_parser_reads(
    capturing_tools: Captured, run_on_stage: Any
) -> None:
    stage = _stage()
    _robot(stage)

    result = _call(
        capturing_tools, run_on_stage, stage, "set_isaac_joint_drive",
        joint_path="/World/Robot/joints/m_shoulder",
        stiffness=100.0, damping=5.0, target_position=30.0, max_force=50.0,
    )

    assert result["success"] is True, result
    assert result["drive_axis"] == "angular"
    assert result["drives"]["angular"] == {
        "type": "force", "stiffness": 100.0, "damping": 5.0,
        "target_position": 30.0, "target_velocity": 0.0, "max_force": 50.0,
    }
    paths, descs = _parsed(stage)["RevoluteJoint"]
    drive = dict(zip((str(p) for p in paths), descs))["/World/Robot/joints/m_shoulder"].drive
    assert drive.enabled is True
    assert (drive.stiffness, drive.damping, drive.targetPosition, drive.forceLimit) == (100.0, 5.0, 30.0, 50.0)


def test_set_joint_drive_picks_linear_for_a_prismatic_joint(
    capturing_tools: Captured, run_on_stage: Any
) -> None:
    stage = _stage()
    _robot(stage)

    result = _call(
        capturing_tools, run_on_stage, stage, "set_isaac_joint_drive",
        joint_path="/World/Robot/joints/b_elbow", target_velocity=0.2, drive_type="acceleration",
    )

    assert result["drive_axis"] == "linear"
    assert result["drives"]["linear"]["type"] == "acceleration"
    assert result["drives"]["linear"]["target_velocity"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("joint_path", "axis", "message"),
    [
        ("/World/Robot/joints/z_root", None, "no degree of freedom"),
        ("/World/Robot/joints/m_shoulder", "linear", "does not apply"),
        ("/World/D6", None, "needs drive_axis"),
        ("/World/Robot", None, "not a Joint"),
    ],
)
def test_set_joint_drive_refuses_an_axis_the_joint_lacks(
    capturing_tools: Captured, run_on_stage: Any, joint_path: str, axis: str | None, message: str
) -> None:
    stage = _stage()
    _robot(stage)
    UsdPhysics.Joint.Define(stage, "/World/D6")

    result = _call(
        capturing_tools, run_on_stage, stage, "set_isaac_joint_drive",
        joint_path=joint_path, drive_axis=axis, stiffness=1.0,
    )

    assert result["success"] is False and message in result["error"]
    assert not any(
        str(schema).startswith("PhysicsDriveAPI")
        for schema in stage.GetPrimAtPath(joint_path).GetAppliedSchemas()
    )


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"stiffness": -1.0}, {"max_force": float("inf")}, {"drive_axis": "twist", "damping": 1.0},
     {"drive_type": "velocity", "damping": 1.0}],
)
def test_set_joint_drive_refuses_bad_values_before_sending(
    capturing_tools: Captured, kwargs: Dict[str, Any]
) -> None:
    tools, captured = capturing_tools
    result = asyncio.run(tools.set_isaac_joint_drive("/World/J", **kwargs))
    assert result["success"] is False and result["error_type"] == "ValidationError"
    assert captured == []


def test_set_joint_limits_is_what_the_parser_reads(
    capturing_tools: Captured, run_on_stage: Any
) -> None:
    stage = _stage()
    _robot(stage)

    result = _call(
        capturing_tools, run_on_stage, stage, "set_isaac_joint_limits",
        joint_path="/World/Robot/joints/m_shoulder", lower=-45.0, upper=60.0,
    )

    assert result["success"] is True
    assert result["limits"] == {"type": "revolute", "axis": "X", "lower": -45.0, "upper": 60.0}
    paths, descs = _parsed(stage)["RevoluteJoint"]
    limit = dict(zip((str(p) for p in paths), descs))["/World/Robot/joints/m_shoulder"].limit
    assert limit.enabled is True and (limit.lower, limit.upper) == (-45.0, 60.0)


def test_set_joint_limits_refuses_crossed_bounds_and_changes_nothing(
    capturing_tools: Captured, run_on_stage: Any
) -> None:
    stage = _stage()
    _robot(stage)
    _call(
        capturing_tools, run_on_stage, stage, "set_isaac_joint_limits",
        joint_path="/World/Robot/joints/m_shoulder", lower=-45.0, upper=60.0,
    )

    crossed = _call(
        capturing_tools, run_on_stage, stage, "set_isaac_joint_limits",
        joint_path="/World/Robot/joints/m_shoulder", lower=90.0,
    )

    assert crossed["success"] is False and "nothing was changed" in crossed["error"]
    joint = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath("/World/Robot/joints/m_shoulder"))
    assert joint.GetLowerLimitAttr().Get() == -45.0


def test_set_joint_limits_on_a_d6_axis_uses_limit_api_and_may_lock_it(
    capturing_tools: Captured, run_on_stage: Any
) -> None:
    stage = _stage()
    UsdPhysics.Joint.Define(stage, "/World/D6")

    missing_axis = _call(
        capturing_tools, run_on_stage, stage, "set_isaac_joint_limits", joint_path="/World/D6", lower=0.0
    )
    locked = _call(
        capturing_tools, run_on_stage, stage, "set_isaac_joint_limits",
        joint_path="/World/D6", limit_axis="transX", lower=1.0, upper=-1.0,
    )

    assert missing_axis["success"] is False and "limit_axis" in missing_axis["error"]
    assert locked["success"] is True
    assert locked["limit_axes"]["transX"] == {"locked": True, "low": 1.0, "high": -1.0}
    assert "PhysicsLimitAPI:transX" in stage.GetPrimAtPath("/World/D6").GetAppliedSchemas()
