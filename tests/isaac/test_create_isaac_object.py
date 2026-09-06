"""create_isaac_object: one generated script runs every build step.

Building one object used to take six tool calls (prim, transform, rigid body,
collision, material, binding), each a round trip and an agent turn. The
compound tool embeds the same script cores the single-step tools use and runs
them in one script, so the logic exists once and the object costs one call.
"""

from __future__ import annotations

import ast
import asyncio
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_path))

from simul_mcp.mcp.tools.isaac._shared import (  # noqa: E402
    APPLY_COLLISION_CORE,
    APPLY_RIGID_BODY_CORE,
    BIND_MATERIAL_CORE,
    DEFINE_MATERIAL_CORE,
    DEFINE_PRIM_CORE,
    SET_MASS_PROPERTIES_CORE,
    SET_PRIM_TRANSFORM_CORE,
)
from simul_mcp.mcp.tools.isaac_tools import IsaacTools  # noqa: E402

STEP_FUNCTIONS: Dict[str, str] = {
    "prim": "_define_prim",
    "transform": "_set_prim_transform",
    "rigid_body": "_apply_rigid_body",
    "collision": "_apply_collision",
    "mass": "_set_mass_properties",
    "material": "_ensure_material",
    "bind_material": "_bind_material",
}


def _full_build(tools: IsaacTools) -> Dict[str, Any]:
    return asyncio.run(
        tools.create_isaac_object(
            prim_path="/World/Box",
            translation=[1, 2, 3],
            rotation_euler=[0, 0, 90],
            scale=[2, 2, 2],
            rigid_body=True,
            collision="convexHull",
            mass=2.5,
            diffuse_color=[1, 0, 0],
        )
    )


def _world_stage() -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    return stage


def test_full_build_is_one_script_defining_every_step(
    capturing_tools: Tuple[IsaacTools, List[str]],
) -> None:
    tools, captured = capturing_tools

    _full_build(tools)

    assert len(captured) == 1, "the compound tool must cost one round trip"
    tree = ast.parse(captured[0])
    defined = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert set(STEP_FUNCTIONS.values()) <= defined
    for step in STEP_FUNCTIONS:
        assert f"({step!r}, lambda:" in captured[0], f"step {step} missing from the plan"


def test_default_call_plans_only_the_prim_step(
    capturing_tools: Tuple[IsaacTools, List[str]],
) -> None:
    tools, captured = capturing_tools

    asyncio.run(tools.create_isaac_object(prim_path="/World/Box"))

    script = captured[0]
    assert "('prim', lambda:" in script
    for step in STEP_FUNCTIONS:
        if step != "prim":
            assert f"({step!r}, lambda:" not in script, f"{step} planned without being asked"


def test_full_build_creates_the_object_on_a_stage(
    capturing_tools: Tuple[IsaacTools, List[str]], run_on_stage: Any
) -> None:
    tools, captured = capturing_tools
    stage = _world_stage()

    _full_build(tools)
    result = run_on_stage(captured[0], stage)

    assert "error" not in result, result
    assert list(result["steps"]) == list(STEP_FUNCTIONS)
    prim = stage.GetPrimAtPath("/World/Box")
    assert prim.GetTypeName() == "Cube"
    assert prim.HasAPI(UsdPhysics.RigidBodyAPI)
    assert prim.HasAPI(UsdPhysics.CollisionAPI)
    assert UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == "convexHull"
    assert UsdPhysics.MassAPI(prim).GetMassAttr().Get() == 2.5
    op_types = [op.GetOpType() for op in UsdGeom.Xformable(prim).GetOrderedXformOps()]
    assert op_types == [
        UsdGeom.XformOp.TypeTranslate,
        UsdGeom.XformOp.TypeRotateXYZ,
        UsdGeom.XformOp.TypeScale,
    ]
    assert result["steps"]["transform"]["translation"] == [1.0, 2.0, 3.0]
    # The schema fallback centre of mass is -inf; that is "unset", not a value.
    assert result["steps"]["mass"]["center_of_mass"] is None
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    assert str(material.GetPath()) == "/World/Looks/Box_Material"
    shader = UsdShade.Shader(stage.GetPrimAtPath("/World/Looks/Box_Material/Shader"))
    assert list(shader.GetInput("diffuseColor").Get()) == [1.0, 0.0, 0.0]


def test_existing_material_is_reused_not_recreated(
    capturing_tools: Tuple[IsaacTools, List[str]], run_on_stage: Any
) -> None:
    tools, captured = capturing_tools
    stage = _world_stage()
    asyncio.run(tools.create_isaac_material("/World/Looks/Red", diffuse_color=[1, 0, 0]))
    run_on_stage(captured[-1], stage)

    asyncio.run(
        tools.create_isaac_object(
            prim_path="/World/Box", material_path="/World/Looks/Red", diffuse_color=[0, 1, 0]
        )
    )
    result = run_on_stage(captured[-1], stage)

    assert "error" not in result, result
    assert result["steps"]["material"] == {"material_path": "/World/Looks/Red", "created": False}
    assert result["steps"]["bind_material"]["bound"] is True
    shader = UsdShade.Shader(stage.GetPrimAtPath("/World/Looks/Red/Shader"))
    assert list(shader.GetInput("diffuseColor").Get()) == [1.0, 0.0, 0.0], "existing look was changed"


def test_failure_stops_at_the_failing_step_and_reports_progress(
    capturing_tools: Tuple[IsaacTools, List[str]], run_on_stage: Any
) -> None:
    tools, captured = capturing_tools
    stage = _world_stage()
    UsdGeom.Cube.Define(stage, "/World/Box")

    _full_build(tools)
    occupied = run_on_stage(captured[-1], stage)

    assert "already exists" in occupied["error"]
    assert occupied["failed_step"] == "prim"
    assert occupied["steps"] == {}

    # A Scope is not Xformable, so the transform step is the first to fail.
    asyncio.run(
        tools.create_isaac_object(prim_path="/World/Group", prim_type="Scope", translation=[1, 0, 0])
    )
    not_xformable = run_on_stage(captured[-1], stage)

    assert not_xformable["failed_step"] == "transform"
    assert list(not_xformable["steps"]) == ["prim"]
    assert stage.GetPrimAtPath("/World/Group").IsValid(), "completed steps are kept"


def test_invalid_collision_is_refused_before_sending(
    capturing_tools: Tuple[IsaacTools, List[str]],
) -> None:
    tools, captured = capturing_tools

    result = asyncio.run(tools.create_isaac_object(prim_path="/World/Box", collision="sphere"))

    assert result["error_type"] == "ValueError"
    assert "convexHull" in result["error"]
    assert captured == []


@pytest.mark.parametrize(
    ("method", "kwargs", "core"),
    [
        ("create_isaac_prim", {"prim_path": "/World/A"}, DEFINE_PRIM_CORE),
        ("set_isaac_prim_transform", {"prim_path": "/World/A"}, SET_PRIM_TRANSFORM_CORE),
        ("add_isaac_rigid_body", {"prim_path": "/World/A"}, APPLY_RIGID_BODY_CORE),
        ("add_isaac_collision", {"prim_path": "/World/A"}, APPLY_COLLISION_CORE),
        ("set_isaac_mass_properties", {"prim_path": "/World/A"}, SET_MASS_PROPERTIES_CORE),
        ("create_isaac_material", {"material_path": "/World/Looks/A"}, DEFINE_MATERIAL_CORE),
        (
            "assign_isaac_material",
            {"prim_path": "/World/A", "material_path": "/World/Looks/A"},
            BIND_MATERIAL_CORE,
        ),
    ],
)
def test_single_step_tools_and_the_compound_tool_share_one_core(
    capturing_tools: Tuple[IsaacTools, List[str]],
    method: str,
    kwargs: Dict[str, Any],
    core: str,
) -> None:
    """The step logic lives once; both tools embed the identical source."""
    tools, captured = capturing_tools

    asyncio.run(getattr(tools, method)(**kwargs))
    _full_build(tools)

    assert textwrap.dedent(core) in captured[0]
    assert textwrap.dedent(core) in captured[1]


def test_single_step_tools_still_work_on_a_stage(
    capturing_tools: Tuple[IsaacTools, List[str]], run_on_stage: Any
) -> None:
    """The refactor into shared cores must not change what each tool does."""
    tools, captured = capturing_tools
    stage = _world_stage()

    def run(coro: Any) -> Dict[str, Any]:
        asyncio.run(coro)
        return run_on_stage(captured[-1], stage)

    created = run(tools.create_isaac_prim("/World/Ball", "Sphere", {"radius": 0.25}))
    assert created == {"prim_path": "/World/Ball", "prim_type": "Sphere", "created": True}
    assert UsdGeom.Sphere(stage.GetPrimAtPath("/World/Ball")).GetRadiusAttr().Get() == 0.25

    moved = run(tools.set_isaac_prim_transform("/World/Ball", translation=[0, 0, 5]))
    assert moved["translation"] == [0.0, 0.0, 5.0]

    body = run(tools.add_isaac_rigid_body("/World/Ball", kinematic=True))
    assert body["kinematic"] is True
    prim = stage.GetPrimAtPath("/World/Ball")
    assert UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get() is True

    again = run(tools.add_isaac_rigid_body("/World/Ball"))
    assert "already applied" in again["error"]

    collider = run(tools.add_isaac_collision("/World/Ball"))
    assert collider["approximation"] == "none"
    assert prim.HasAPI(UsdPhysics.CollisionAPI)
    assert not prim.HasAPI(UsdPhysics.MeshCollisionAPI)

    mass = run(tools.set_isaac_mass_properties("/World/Ball", mass=1.5, center_of_mass=[0, 0, 1]))
    assert mass["mass"] == 1.5
    assert mass["center_of_mass"] == [0.0, 0.0, 1.0]

    material = run(tools.create_isaac_material("/World/Looks/Blue", diffuse_color=[0, 0, 1]))
    assert material["created"] is True
    assert stage.GetPrimAtPath("/World/Looks/Blue/Shader").IsValid()

    bound = run(tools.assign_isaac_material("/World/Ball", "/World/Looks/Blue"))
    assert bound["bound"] is True
    found, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    assert str(found.GetPath()) == "/World/Looks/Blue"
