"""Round-2 Isaac fixes: physics values follow an import's unit correction, DOF order matches Isaac.

- ``import_isaac_asset`` scaled a centimetre asset's transform into a metre
  stage but left its physics values (prismatic limits, drive targets and
  gains, density, inertia, contact offsets) in centimetres, which PhysX reads
  as metres. Kit's metrics assembler rescales those with the exponents in
  ``pxr.PhysicsSchemaTools.units_db``; it only runs for Kit's reference
  commands, never for simul's raw pxr reference, so simul now does the same.
- ``list_isaac_articulations`` reported a depth-first walk index that counted
  fixed joints; Isaac Sim 6.0.1's USD articulation backend orders DOFs
  breadth first from the root link and drops fixed joints.
- Minors: ``create_isaac_material`` shader types, collision on visual
  geometry, the save budget, the material asset sandbox, articulation paging
  and wrong names in tool text and resources.
"""

from __future__ import annotations

import asyncio
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from simul.adapters.isaac_socket_client import ScriptResult
from simul.config import Settings
from simul.mcp.tools.isaac._physics_units import PHYSICS_UNIT_EXPONENTS
from simul.mcp.tools.isaac_tools import IsaacTools
from tests.isaac.conftest import StageTools
from tests.isaac.test_materials_import_save_graph import _call, _recording_tools, _settings, _world_stage
from tests.isaac.test_script_sizes import SCRIPT_BUDGET

RESOURCES = Path(__file__).resolve().parents[2] / "src" / "simul" / "resources" / "docs" / "api"


# ---------------------------------------------------------------------------
# B1: physics values in asset units are rescaled with the transform
# ---------------------------------------------------------------------------


def _cm_robot(path: Path, *, kilograms_per_unit: Optional[float] = None, instanced_collider: bool = False) -> str:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    UsdGeom.SetStageUpAxis(stage, "Z")
    if kilograms_per_unit is not None:
        UsdPhysics.SetStageKilogramsPerUnit(stage, kilograms_per_unit)
    root = UsdGeom.Xform.Define(stage, "/Robot").GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    base = UsdGeom.Cube.Define(stage, "/Robot/base").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(base)
    UsdPhysics.CollisionAPI.Apply(base)
    base.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float, False).Set(2.0)
    mass = UsdPhysics.MassAPI.Apply(base)
    mass.CreateDensityAttr(0.0078)
    mass.CreateMassAttr(5.0)
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(100.0, 100.0, 100.0))
    slider = UsdGeom.Cube.Define(stage, "/Robot/slider").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(slider)
    slide = UsdPhysics.PrismaticJoint.Define(stage, "/Robot/slide")
    slide.CreateBody0Rel().SetTargets(["/Robot/base"])
    slide.CreateBody1Rel().SetTargets(["/Robot/slider"])
    slide.CreateUpperLimitAttr(4.0)
    linear = UsdPhysics.DriveAPI.Apply(slide.GetPrim(), "linear")
    linear.CreateTargetPositionAttr(2.0)
    linear.CreateStiffnessAttr(50.0)
    arm = UsdGeom.Cube.Define(stage, "/Robot/arm").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(arm)
    hinge = UsdPhysics.RevoluteJoint.Define(stage, "/Robot/hinge")
    hinge.CreateBody0Rel().SetTargets(["/Robot/base"])
    hinge.CreateBody1Rel().SetTargets(["/Robot/arm"])
    hinge.CreateUpperLimitAttr(90.0)
    angular = UsdPhysics.DriveAPI.Apply(hinge.GetPrim(), "angular")
    angular.CreateStiffnessAttr(1000.0)
    angular.CreateTargetPositionAttr(45.0)
    if instanced_collider:
        part = stage.DefinePrim("/Robot/arm/shell")
        part.SetInstanceable(True)
        inner = UsdGeom.Cube.Define(stage, "/Prototypes/shell/geom").GetPrim()
        UsdPhysics.CollisionAPI.Apply(inner)
        inner.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float, False).Set(1.0)
        part.GetReferences().AddInternalReference("/Prototypes/shell")
    stage.Save()
    return str(path)


def _run_tool(stage_tools: StageTools, stage: Usd.Stage, method: str, **kwargs: Any) -> Dict[str, Any]:
    """Run a tool with the sandbox off, every script it sends running on ``stage``.

    Import and reference send the physics-units pass as a second script.
    """
    tools, sent = stage_tools(stage, _settings(sandbox=False))
    result = asyncio.run(getattr(tools, method)(**kwargs))
    assert all(len(script.encode()) < SCRIPT_BUDGET for script in sent)
    return result


def test_a_failed_physics_pass_is_reported_not_dropped() -> None:
    pending = {"pending": "/World/A/Robot", "applied": True, "length_ratio": 0.01, "mass_ratio": 1.0}
    tools, captured = _recording_tools(
        _settings(sandbox=False),
        [{"imported": True, "units": {"notes": [], "physics": pending}}, {"error": "socket closed"}],
    )

    result = asyncio.run(tools.import_isaac_asset(asset_path="/tmp/simul-work/robot.usda"))

    assert len(captured) == 2 and "'/World/A/Robot'" in captured[1]
    assert result["units"]["physics"]["applied"] is False
    assert result["units"]["physics"]["failure"] == "socket closed"
    assert "not rescaled (socket closed)" in result["warning"]


def _value(stage: Usd.Stage, path: str, name: str) -> Any:
    return stage.GetPrimAtPath(path).GetAttribute(name).Get()


def test_a_centimetre_robot_gets_its_physics_values_in_metres(
    stage_tools: StageTools, tmp_path: Path
) -> None:
    asset = _cm_robot(tmp_path / "robot.usda")
    stage = _world_stage()

    result = _run_tool(stage_tools, stage, "import_isaac_asset", asset_path=asset)

    root = result["asset_root_path"]
    physics = result["units"]["physics"]
    assert physics["applied"] is True and physics["uncorrected_count"] == 0
    assert _value(stage, root + "/slide", "physics:upperLimit") == pytest.approx(0.04)
    assert _value(stage, root + "/slide", "drive:linear:physics:targetPosition") == pytest.approx(0.02)
    assert _value(stage, root + "/base", "physics:density") == pytest.approx(7800.0, rel=1e-5)
    assert list(_value(stage, root + "/base", "physics:diagonalInertia")) == pytest.approx([0.01] * 3)
    assert _value(stage, root + "/base", "physxCollision:contactOffset") == pytest.approx(0.02)
    # Angles do not change between stages; N/m drive stiffness has no length.
    assert _value(stage, root + "/hinge", "physics:upperLimit") == pytest.approx(90.0)
    assert _value(stage, root + "/hinge", "drive:angular:physics:targetPosition") == pytest.approx(45.0)
    assert _value(stage, root + "/slide", "drive:linear:physics:stiffness") == pytest.approx(50.0)
    # Torque per degree is kg*m^2/s^2/deg: length squared.
    assert _value(stage, root + "/hinge", "drive:angular:physics:stiffness") == pytest.approx(0.1)
    assert _value(stage, root + "/base", "physics:mass") == pytest.approx(5.0)
    # The overrides live in the importing layer; the asset file is untouched.
    assert Usd.Stage.Open(asset).GetPrimAtPath("/Robot/slide").GetAttribute("physics:upperLimit").Get() == 4.0


def test_an_asset_in_grams_gets_its_masses_in_kilograms(stage_tools: StageTools, tmp_path: Path) -> None:
    asset = _cm_robot(tmp_path / "robot.usda", kilograms_per_unit=0.001)
    stage = _world_stage()

    result = _run_tool(stage_tools, stage, "import_isaac_asset", asset_path=asset)

    root = result["asset_root_path"]
    assert result["units"]["mass_ratio"] == pytest.approx(0.001)
    assert _value(stage, root + "/base", "physics:mass") == pytest.approx(0.005)
    assert _value(stage, root + "/base", "physics:density") == pytest.approx(7.8, rel=1e-5)


def test_without_correction_the_physics_values_are_only_counted(
    stage_tools: StageTools, tmp_path: Path
) -> None:
    asset = _cm_robot(tmp_path / "robot.usda")
    stage = _world_stage()

    result = _run_tool(stage_tools, stage, "import_isaac_asset", asset_path=asset, correct_units=False)

    physics = result["units"]["physics"]
    assert physics["applied"] is False and physics["attribute_count"] >= 6
    assert any("not rescaled" in note for note in result["units"]["notes"])
    assert _value(stage, "/World/ImportedAsset/slide", "physics:upperLimit") == pytest.approx(4.0)


def test_values_inside_instances_are_reported_not_silently_left(
    stage_tools: StageTools, tmp_path: Path
) -> None:
    asset = _cm_robot(tmp_path / "robot.usda", instanced_collider=True)
    stage = _world_stage()

    result = _run_tool(stage_tools, stage, "import_isaac_asset", asset_path=asset)

    physics = result["units"]["physics"]
    assert physics["uncorrected_count"] == 1
    assert physics["uncorrected"][0]["attribute"] == "physxCollision:restOffset"
    assert "could not be rescaled" in result["warning"]


def test_add_reference_rescales_physics_with_an_in_place_correction(
    stage_tools: StageTools, tmp_path: Path
) -> None:
    asset = _cm_robot(tmp_path / "robot.usda")
    stage = _world_stage()

    result = _run_tool(
        stage_tools, stage, "add_isaac_reference",
        prim_path="/World/Ref", reference_path=asset, correct_units=True,
    )

    assert result["units"]["applied"] is True and result["units"]["physics"]["applied"] is True
    assert _value(stage, "/World/Ref/slide", "physics:upperLimit") == pytest.approx(0.04)


def test_add_reference_leaves_a_prims_own_children_in_stage_units(
    stage_tools: StageTools, tmp_path: Path
) -> None:
    asset = _cm_robot(tmp_path / "robot.usda")
    stage = _world_stage()
    UsdGeom.Xform.Define(stage, "/World/Rig")
    UsdPhysics.PrismaticJoint.Define(stage, "/World/Rig/LocalSlider").CreateLowerLimitAttr(-1.0)

    result = _run_tool(
        stage_tools, stage, "add_isaac_reference",
        prim_path="/World/Rig", reference_path=asset, correct_units=True,
    )

    # Converting the asset's values would convert LocalSlider's metres too,
    # and the scale op would move it, so neither is applied.
    assert _value(stage, "/World/Rig/LocalSlider", "physics:lowerLimit") == pytest.approx(-1.0)
    assert result["units"]["applied"] is False and result["units"]["physics"]["applied"] is False
    assert not UsdGeom.Xformable(stage.GetPrimAtPath("/World/Rig")).GetOrderedXformOps()
    assert "LocalSlider" in result["warning"]
    assert any("already had children" in note for note in result["units"]["notes"])


def test_add_reference_leaves_a_prims_own_physics_values_in_stage_units(
    stage_tools: StageTools, tmp_path: Path
) -> None:
    asset = _cm_robot(tmp_path / "robot.usda")
    stage = _world_stage()
    UsdPhysics.MassAPI.Apply(UsdGeom.Xform.Define(stage, "/World/Rig").GetPrim()).CreateDensityAttr(1000.0)

    result = _run_tool(
        stage_tools, stage, "add_isaac_reference",
        prim_path="/World/Rig", reference_path=asset, correct_units=True,
    )

    assert _value(stage, "/World/Rig", "physics:density") == pytest.approx(1000.0)
    assert result["units"]["physics"]["applied"] is False
    assert "physics:density" in result["warning"]


def test_the_built_in_table_carries_the_units_db_exponents() -> None:
    """Spot checks against pxr/PhysicsSchemaTools/units_db.py (Isaac Sim 5.0 tree)."""
    assert PHYSICS_UNIT_EXPONENTS["physics:density"] == (1, -3)
    assert PHYSICS_UNIT_EXPONENTS["physics:diagonalInertia"] == (1, 2)
    assert PHYSICS_UNIT_EXPONENTS["physics:upperLimit"] == {"RevoluteJoint": (0, 0), "PrismaticJoint": (0, 1)}
    assert PHYSICS_UNIT_EXPONENTS["drive:angular:physics:stiffness"] == {"Force": (1, 2), "Acceleration": (0, 2)}
    assert PHYSICS_UNIT_EXPONENTS["drive:linear:physics:targetVelocity"] == (0, 1)
    assert PHYSICS_UNIT_EXPONENTS["limit:transX:physics:low"] == (0, 1)
    assert "limit:rotX:physics:low" not in PHYSICS_UNIT_EXPONENTS


# ---------------------------------------------------------------------------
# M2: dof_index is Isaac's DOF order
# ---------------------------------------------------------------------------


def _body(stage: Usd.Stage, path: str) -> None:
    UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, path).GetPrim())


def _joint(stage: Usd.Stage, path: str, kind: Any, body0: Optional[str], body1: str) -> None:
    joint = kind.Define(stage, path)
    if body0:
        joint.CreateBody0Rel().SetTargets([body0])
    joint.CreateBody1Rel().SetTargets([body1])


def _franka(stage: Usd.Stage) -> None:
    root = "/World/franka"
    UsdPhysics.ArticulationRootAPI.Apply(UsdGeom.Xform.Define(stage, root).GetPrim())
    for name in ["panda_link%d" % i for i in range(8)] + ["panda_hand", "panda_leftfinger", "panda_rightfinger"]:
        _body(stage, root + "/" + name)
    _joint(stage, root + "/root_joint", UsdPhysics.FixedJoint, None, root + "/panda_link0")
    for i in range(1, 8):
        _joint(stage, "%s/panda_link%d/panda_joint%d" % (root, i - 1, i), UsdPhysics.RevoluteJoint,
               "%s/panda_link%d" % (root, i - 1), "%s/panda_link%d" % (root, i))
    _joint(stage, root + "/panda_link7/panda_hand_joint", UsdPhysics.FixedJoint,
           root + "/panda_link7", root + "/panda_hand")
    for side, finger in ((1, "leftfinger"), (2, "rightfinger")):
        _joint(stage, "%s/panda_hand/panda_finger_joint%d" % (root, side), UsdPhysics.PrismaticJoint,
               root + "/panda_hand", root + "/panda_" + finger)


def _anymal(stage: Usd.Stage, root: str = "/World/anymal") -> None:
    UsdPhysics.ArticulationRootAPI.Apply(UsdGeom.Xform.Define(stage, root).GetPrim())
    _body(stage, root + "/base")
    for leg in ("LF", "LH", "RF", "RH"):
        previous = root + "/base"
        for segment, joint in (("HIP", "HAA"), ("THIGH", "HFE"), ("SHANK", "KFE")):
            link = "%s/%s_%s" % (root, leg, segment)
            _body(stage, link)
            _joint(stage, "%s/joints/%s_%s" % (root, leg, joint), UsdPhysics.RevoluteJoint, previous, link)
            previous = link


def _isaac_bfs_dofs(stage: Usd.Stage, root_link: str) -> List[str]:
    """The reviewer's reference: _walk_connected_articulation + _discover_dofs."""
    by_body: Dict[str, List[Any]] = {}
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            for slot, rel in enumerate((joint.GetBody0Rel(), joint.GetBody1Rel())):
                targets = rel.GetTargets()
                if targets:
                    by_body.setdefault(str(targets[0]), []).append((prim, slot))
    order: List[Any] = []
    seen_links: set = set()
    seen_joints: set = set()
    queue = deque([root_link])
    while queue:
        link = queue.popleft()
        if link in seen_links:
            continue
        seen_links.add(link)
        for prim, slot in by_body.get(link, []):
            if str(prim.GetPath()) not in seen_joints:
                seen_joints.add(str(prim.GetPath()))
                order.append(prim)
            joint = UsdPhysics.Joint(prim)
            other = (joint.GetBody0Rel(), joint.GetBody1Rel())[1 - slot].GetTargets()
            if other and str(other[0]) not in seen_links:
                queue.append(str(other[0]))
    return [str(prim.GetPath()) for prim in order if not prim.IsA(UsdPhysics.FixedJoint)]


def test_dof_index_follows_isaacs_breadth_first_order(capturing_tools: Any, run_on_stage: Any) -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    _franka(stage)
    _anymal(stage)

    result = _call(capturing_tools, run_on_stage, stage, "list_isaac_articulations")

    for articulation in result["articulations"]:
        expected = _isaac_bfs_dofs(stage, articulation["root_body"])
        assert articulation["dof_paths"] == expected
        for joint in articulation["joints"]:
            path = joint["prim_path"]
            assert joint["dof_index"] == (expected.index(path) if path in expected else None), path
    franka = next(a for a in result["articulations"] if a["root_path"] == "/World/franka")
    by_name = {j["prim_path"].rsplit("/", 1)[1]: j for j in franka["joints"]}
    assert [by_name["panda_joint%d" % i]["dof_index"] for i in range(1, 8)] == list(range(7))
    assert by_name["root_joint"]["dof_index"] is None and by_name["panda_hand_joint"]["dof_index"] is None
    anymal = next(a for a in result["articulations"] if a["root_path"] == "/World/anymal")
    assert [p.rsplit("/", 1)[1] for p in anymal["dof_paths"][:4]] == ["LF_HAA", "LH_HAA", "RF_HAA", "RH_HAA"]


def test_articulations_page_with_an_offset(capturing_tools: Any, run_on_stage: Any) -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    for index in range(3):
        _anymal(stage, "/World/anymal_%d" % index)

    first = _call(capturing_tools, run_on_stage, stage, "list_isaac_articulations", max_results=2)
    second = _call(
        capturing_tools, run_on_stage, stage, "list_isaac_articulations", max_results=2, offset=first["next_offset"]
    )

    assert first["articulation_count"] == 3 and first["truncated"] is True and first["next_offset"] == 2
    assert [a["root_path"] for a in second["articulations"]] == ["/World/anymal_2"]
    assert second["truncated"] is False and second["next_offset"] is None


# ---------------------------------------------------------------------------
# Minors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("requested", "built"), [("omnipbr", "OmniPBR"), ("USDPREVIEWSURFACE", "UsdPreviewSurface")])
def test_material_shader_types_are_matched_without_case(
    capturing_tools: Any, run_on_stage: Any, requested: str, built: str
) -> None:
    stage = Usd.Stage.CreateInMemory()

    result = _call(capturing_tools, run_on_stage, stage, "create_isaac_material",
                   material_path="/Looks/M", shader_type=requested)

    assert result["success"] is True and result["shader_type"] == built


def test_an_unsupported_shader_type_is_refused_before_sending(capturing_tools: Any) -> None:
    tools, captured = capturing_tools
    result = asyncio.run(tools.create_isaac_material("/Looks/M", shader_type="OmniGlass"))
    assert result["error_type"] == "ValidationError" and result["details"]["code"] == "UNSUPPORTED_SHADER_TYPE"
    assert captured == []


def test_a_texture_outside_the_sandbox_is_refused_and_nothing_is_created(
    capturing_tools: Any, run_on_stage: Any
) -> None:
    stage = Usd.Stage.CreateInMemory()
    _call(capturing_tools, run_on_stage, stage, "create_isaac_material", material_path="/Looks/M",
          shader_type="OmniPBR")

    result = _call(capturing_tools, run_on_stage, stage, "set_isaac_material_property",
                   material_path="/Looks/M", property_name="diffuse_texture", value="/etc/passwd")

    assert result["success"] is False and result["error_type"] == "SandboxError"
    assert not stage.GetPrimAtPath("/Looks/M/Shader").GetAttribute("inputs:diffuse_texture").IsValid()


def test_collision_skips_visual_and_guide_geometry_unless_asked(capturing_tools: Any, run_on_stage: Any) -> None:
    def robot() -> Usd.Stage:
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, "/Robot")
        UsdGeom.Cube.Define(stage, "/Robot/collision_box")
        visual = UsdGeom.Mesh.Define(stage, "/Robot/visual_mesh")
        visual.CreatePurposeAttr(UsdGeom.Tokens.render)
        UsdGeom.Sphere.Define(stage, "/Robot/marker").CreatePurposeAttr(UsdGeom.Tokens.guide)
        return stage

    stage = robot()
    default = _call(capturing_tools, run_on_stage, stage, "add_isaac_collision", prim_path="/Robot")
    stage_all = robot()
    everything = _call(capturing_tools, run_on_stage, stage_all, "add_isaac_collision",
                       prim_path="/Robot", all_purposes=True)

    assert [shape["path"] for shape in default["shapes"]] == ["/Robot/collision_box"]
    assert default["skipped_purpose_count"] == 2
    assert not stage.GetPrimAtPath("/Robot/visual_mesh").HasAPI(UsdPhysics.CollisionAPI)
    assert everything["shape_count"] == 3


def test_save_gets_the_heavy_script_budget() -> None:
    calls: List[Dict[str, Any]] = []

    async def _execute(code: str, **kwargs: Any) -> ScriptResult:
        calls.append(kwargs)
        return ScriptResult(success=True, output='{"file_path": "/tmp/simul-work/a.usd", "saved": true}')

    client = MagicMock()
    client.bridge_enabled = False
    client.fallback_to_vscode = True
    client.script_timeout_seconds = 29.0
    client.execute = AsyncMock(side_effect=_execute)
    tools = IsaacTools(client, settings=Settings())

    asyncio.run(tools.save_isaac_stage(file_path="/tmp/simul-work/a.usd"))

    assert calls and calls[-1].get("timeout_seconds", 0) > 29.0


def test_wrong_names_are_gone_from_tool_text_and_resources() -> None:
    tools = IsaacTools(MagicMock(), settings=Settings())
    graph_text = type(tools).create_isaac_graph_node.__tool_meta__.description
    aov_text = type(tools).read_aovs.__tool_meta__.description
    assert "omni.graph.action.OnPlaybackTick" in graph_text and "omni.graph.nodes.OnPlaybackTick" not in graph_text
    assert "Depth," not in aov_text and "randomizers" in aov_text
    assert "omni.isaac.core_nodes" not in (RESOURCES / "assets.md").read_text()
    for page in ("sensors.md", "robots.md"):
        assert "Isaac Sim 6.0" in (RESOURCES / page).read_text()
