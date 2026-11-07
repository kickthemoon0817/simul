"""Prim edits that respect composition, and traversals that see instances.

Run on real ``pxr`` against a robot-like asset on disk: a referenced robot
whose link keeps its collider under an instanceable prim, with a local
override authored on the referenced link. Each test reproduces a defect:

- ``delete_isaac_prim`` / ``duplicate_isaac_prim`` / ``reparent_isaac_prim``
  edited the root layer only. Deleting the referenced link destroyed the
  user's override, left the link in place and reported ``deleted: true``;
  duplicating it copied an undefined ``over`` no traversal sees.
- ``reparent_isaac_prim`` left relationships on the old path (an unbound
  cube, a joint whose body0 named nothing), reported ``//Box`` for a move to
  ``/`` and dropped the world pose.
- The listings stopped at instanceable prims (``collider_count`` 0 for a
  robot, no Mesh found) and never showed a deactivated prim.
- ``get_isaac_subtree(max_depth=-1)`` returned only the root.
- ``create_isaac_prim`` created an untyped prim for ``cube`` and skipped
  attributes the schema doesn't define; ``create_isaac_object`` lost its
  steps report when a step raised.
- With script execution disabled, a script could still be planted through
  an OmniGraph script node's attributes.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import sys
import types
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest
from khemoo.simul.protocol import BridgeRequest
from khemoo.simul.service import BridgeCommandService
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from simul.adapters.isaac_socket_client import ScriptResult
from simul.config import Settings
from simul.mcp.tools.isaac_tools import IsaacTools

Tools = Tuple[IsaacTools, List[str]]

LINK = "/World/Franka/link1"
COLLIDER = "/World/Franka/link1/collisions/mesh"


def _call(
    tools: Tools,
    run_on_stage: Any,
    stage: Usd.Stage,
    method: str,
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    instance, captured = tools
    result = asyncio.run(getattr(instance, method)(*args, **kwargs))
    if not captured or result.get("error_type") in (
        "ValueError",
        "ScriptExecutionDisabled",
    ):
        return result
    return run_on_stage(captured[-1], stage)


@pytest.fixture
def robot_asset(tmp_path: Path) -> str:
    """robot.usda: a link whose collider sits under an instanceable prim, as Isaac assets do."""
    collision_path = str(tmp_path / "collision.usda")
    collision = Usd.Stage.CreateNew(collision_path)
    UsdGeom.Xform.Define(collision, "/C")
    mesh = UsdGeom.Mesh.Define(collision, "/C/mesh")
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    collision.SetDefaultPrim(collision.GetPrimAtPath("/C"))
    collision.Save()

    robot_path = str(tmp_path / "robot.usda")
    robot = Usd.Stage.CreateNew(robot_path)
    UsdGeom.Xform.Define(robot, "/Robot")
    base = UsdGeom.Xform.Define(robot, "/Robot/base_link")
    UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
    link = UsdGeom.Xform.Define(robot, "/Robot/link1")
    link.AddTranslateOp().Set(Gf.Vec3d(0, 0, 1))
    UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
    instance = robot.DefinePrim("/Robot/link1/collisions", "Xform")
    instance.GetReferences().AddReference("./collision.usda")
    instance.SetInstanceable(True)
    joint = UsdPhysics.RevoluteJoint.Define(robot, "/Robot/joints/joint1")
    joint.CreateBody0Rel().SetTargets(["/Robot/base_link"])
    joint.CreateBody1Rel().SetTargets(["/Robot/link1"])
    robot.SetDefaultPrim(robot.GetPrimAtPath("/Robot"))
    robot.Save()
    return robot_path


@pytest.fixture
def robot_stage(robot_asset: str) -> Usd.Stage:
    """/World/Franka references the robot; the user tuned link1 with a local override."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    stage.DefinePrim("/World/Franka", "Xform").GetReferences().AddReference(robot_asset)
    stage.OverridePrim(LINK).CreateAttribute("userTuned", Sdf.ValueTypeNames.Float).Set(
        42.0
    )
    return stage


def _local_stage() -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    return stage


# ---------------------------------------------------------------------------
# Kit commands: run a script with a stand-in omni.kit.commands
# ---------------------------------------------------------------------------


def _run_with_kit(
    script: str,
    stage: Usd.Stage,
    commands: Dict[str, Callable[..., Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> Dict[str, Any]:
    calls: List[Tuple[str, Dict[str, Any]]] = []

    def _execute(name: str, **kwargs: Any) -> Any:
        calls.append((name, kwargs))
        return commands[name](**kwargs)

    context = types.SimpleNamespace(
        get_stage=lambda: stage, get_stage_url=lambda: "anon:memory.usda"
    )
    omni = types.ModuleType("omni")
    omni.__path__ = []  # type: ignore[attr-defined]
    omni_usd = types.ModuleType("omni.usd")
    omni_usd.get_context = lambda: context  # type: ignore[attr-defined]
    kit = types.ModuleType("omni.kit")
    kit.__path__ = []  # type: ignore[attr-defined]
    kit_commands = types.ModuleType("omni.kit.commands")
    kit_commands.execute = _execute  # type: ignore[attr-defined]
    omni.usd = omni_usd  # type: ignore[attr-defined]
    omni.kit = kit  # type: ignore[attr-defined]
    kit.commands = kit_commands  # type: ignore[attr-defined]
    for name, module in {
        "omni": omni,
        "omni.usd": omni_usd,
        "omni.kit": kit,
        "omni.kit.commands": kit_commands,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(
            compile(script, "<isaac-script>", "exec"), {"__name__": "__isaac_script__"}
        )
    result = json.loads(stdout.getvalue())
    result["_kit_calls"] = [name for name, _ in calls]
    result["_kit_kwargs"] = [kwargs for _, kwargs in calls]
    return result


# ---------------------------------------------------------------------------
# delete_isaac_prim
# ---------------------------------------------------------------------------


def test_deleting_a_referenced_link_deactivates_it_and_keeps_the_override(
    capturing_tools: Tools, run_on_stage: Any, robot_stage: Usd.Stage
) -> None:
    """It used to report deleted:true, leave the link and destroy the user's override."""
    result = _call(
        capturing_tools, run_on_stage, robot_stage, "delete_isaac_prim", LINK
    )

    assert result["action"] == "deactivated"
    assert result["deleted"] is False
    assert "reference authored on /World/Franka" in result["reason"]
    prim = robot_stage.GetPrimAtPath(LINK)
    assert prim.IsValid() and not prim.IsActive()
    assert prim.GetAttribute("userTuned").Get() == 42.0, "the local override survives"


def test_remove_mode_refuses_referenced_content_unchanged(
    capturing_tools: Tools, run_on_stage: Any, robot_stage: Usd.Stage
) -> None:
    result = _call(
        capturing_tools,
        run_on_stage,
        robot_stage,
        "delete_isaac_prim",
        LINK,
        mode="remove",
    )

    assert "mode='deactivate'" in result["error"]
    prim = robot_stage.GetPrimAtPath(LINK)
    assert prim.IsActive()
    assert prim.GetAttribute("userTuned").Get() == 42.0


def test_deleting_a_local_prim_removes_it(
    capturing_tools: Tools, run_on_stage: Any
) -> None:
    stage = _local_stage()
    UsdGeom.Cube.Define(stage, "/World/Box")

    result = _call(
        capturing_tools, run_on_stage, stage, "delete_isaac_prim", "/World/Box"
    )

    assert result["action"] == "removed"
    assert result["deleted"] is True
    assert not stage.GetPrimAtPath("/World/Box").IsValid()


def test_a_prim_added_under_a_referenced_link_is_removed(
    capturing_tools: Tools, run_on_stage: Any, robot_stage: Usd.Stage
) -> None:
    """Only the reference's own content is out of reach; a local addition is not."""
    UsdGeom.Xform.Define(robot_stage, LINK + "/sensor")

    result = _call(
        capturing_tools,
        run_on_stage,
        robot_stage,
        "delete_isaac_prim",
        LINK + "/sensor",
    )

    assert result["action"] == "removed"
    assert not robot_stage.GetPrimAtPath(LINK + "/sensor").IsValid()
    assert robot_stage.GetPrimAtPath(LINK).IsActive()


def test_delete_goes_through_the_edit_target(
    capturing_tools: Tools, run_on_stage: Any
) -> None:
    """A prim defined in a sublayer: the root layer has nothing to remove."""
    stage = _local_stage()
    sublayer = Sdf.Layer.CreateAnonymous("props.usda")
    stage.GetRootLayer().subLayerPaths.append(sublayer.identifier)
    stage.SetEditTarget(Usd.EditTarget(sublayer))
    UsdGeom.Cube.Define(stage, "/World/Prop")
    stage.SetEditTarget(Usd.EditTarget(stage.GetRootLayer()))

    from_root = _call(
        capturing_tools,
        run_on_stage,
        stage,
        "delete_isaac_prim",
        "/World/Prop",
        mode="remove",
    )
    assert "also defined in" in from_root["error"]
    assert stage.GetPrimAtPath("/World/Prop").IsActive()

    stage.SetEditTarget(Usd.EditTarget(sublayer))
    from_sublayer = _call(
        capturing_tools, run_on_stage, stage, "delete_isaac_prim", "/World/Prop"
    )
    assert from_sublayer["action"] == "removed"
    assert from_sublayer["edit_target"] == sublayer.identifier
    assert not stage.GetPrimAtPath("/World/Prop").IsValid()


def test_instance_proxies_are_refused(
    capturing_tools: Tools, run_on_stage: Any, robot_stage: Usd.Stage
) -> None:
    for method, args in (
        ("delete_isaac_prim", (COLLIDER,)),
        ("duplicate_isaac_prim", (COLLIDER, "/World/Copy")),
        ("reparent_isaac_prim", (COLLIDER, "/World")),
    ):
        result = _call(capturing_tools, run_on_stage, robot_stage, method, *args)
        assert result["error_type"] == "InvalidState", method
        assert result["details"]["code"] == "INSTANCE_PROXY", method
        assert result["details"]["instance"] == "/World/Franka/link1/collisions", method
    assert robot_stage.GetPrimAtPath(COLLIDER).IsValid()


def test_delete_uses_kits_undoable_command_when_present(
    capturing_tools: Tools, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = _local_stage()
    UsdGeom.Cube.Define(stage, "/World/Box")
    tools, captured = capturing_tools
    asyncio.run(tools.delete_isaac_prim("/World/Box"))

    result = _run_with_kit(
        captured[-1],
        stage,
        {
            "DeletePrims": lambda paths, destructive=True: [
                stage.RemovePrim(p) for p in paths
            ]
        },
        monkeypatch,
    )

    assert result["_kit_calls"] == ["DeletePrims"]
    assert result["_kit_kwargs"] == [{"paths": ["/World/Box"], "destructive": False}]
    assert result["method"] == "omni.kit.commands DeletePrims"
    assert result["undoable"] is True
    assert not stage.GetPrimAtPath("/World/Box").IsValid()


def test_a_kit_that_declines_the_delete_is_respected(
    capturing_tools: Tools, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kit refuses prims it protects; the delete used to fall back to RemovePrim."""
    stage = _local_stage()
    UsdGeom.Cube.Define(stage, "/World/Box")
    tools, captured = capturing_tools
    asyncio.run(tools.delete_isaac_prim("/World/Box"))

    result = _run_with_kit(
        captured[-1],
        stage,
        {"DeletePrims": lambda paths, destructive=True: None},
        monkeypatch,
    )

    assert result["error_type"] == "InvalidState"
    assert result["details"]["code"] == "KIT_DECLINED"
    assert stage.GetPrimAtPath("/World/Box").IsDefined(), "pxr did not override Kit"


def test_an_older_kit_without_destructive_is_called_again(
    capturing_tools: Tools, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = _local_stage()
    UsdGeom.Cube.Define(stage, "/World/Box")
    tools, captured = capturing_tools
    asyncio.run(tools.delete_isaac_prim("/World/Box"))

    result = _run_with_kit(
        captured[-1],
        stage,
        {"DeletePrims": lambda paths: [stage.RemovePrim(p) for p in paths]},
        monkeypatch,
    )

    assert result["_kit_kwargs"] == [
        {"paths": ["/World/Box"], "destructive": False},
        {"paths": ["/World/Box"]},
    ]
    assert result["method"] == "omni.kit.commands DeletePrims"
    assert "destructive" in result["kit_note"]


def test_session_layer_prims_are_not_deleted(
    capturing_tools: Tools, run_on_stage: Any
) -> None:
    """Kit's viewport cameras live in the session layer; an over hid them from the saved scene."""
    stage = _local_stage()
    stage.SetEditTarget(Usd.EditTarget(stage.GetSessionLayer()))
    UsdGeom.Camera.Define(stage, "/OmniverseKit_Persp")
    stage.SetEditTarget(Usd.EditTarget(stage.GetRootLayer()))

    result = _call(
        capturing_tools, run_on_stage, stage, "delete_isaac_prim", "/OmniverseKit_Persp"
    )

    assert result["details"]["code"] == "SESSION_LAYER_PRIM"
    assert stage.GetPrimAtPath("/OmniverseKit_Persp").IsActive()
    assert not stage.GetRootLayer().GetPrimAtPath("/OmniverseKit_Persp")


def test_refusals_are_typed(
    capturing_tools: Tools, run_on_stage: Any, robot_stage: Usd.Stage
) -> None:
    UsdGeom.Cube.Define(robot_stage, "/World/Box")
    UsdGeom.Cube.Define(robot_stage, "/World/Other/Box")
    cases = [
        (
            "delete_isaac_prim",
            (LINK,),
            {"mode": "remove"},
            "InvalidState",
            "DEFINED_ELSEWHERE",
        ),
        (
            "reparent_isaac_prim",
            (LINK, "/World"),
            {},
            "InvalidState",
            "DEFINED_ELSEWHERE",
        ),
        (
            "reparent_isaac_prim",
            ("/World/Box", "/World/Other"),
            {},
            "ValidationError",
            "DESTINATION_EXISTS",
        ),
        (
            "duplicate_isaac_prim",
            ("/World/Box", "/World/Other/Box"),
            {},
            "ValidationError",
            "DESTINATION_EXISTS",
        ),
        ("delete_isaac_prim", ("/World/Nope",), {}, "NotFoundError", "PRIM_NOT_FOUND"),
        (
            "create_isaac_prim",
            ("/World/C", "cube"),
            {},
            "ValidationError",
            "UNKNOWN_PRIM_TYPE",
        ),
        (
            "create_isaac_prim",
            ("/World/Box", "Cube"),
            {},
            "ValidationError",
            "PRIM_EXISTS",
        ),
    ]
    for method, args, kwargs, error_type, code in cases:
        result = _call(
            capturing_tools, run_on_stage, robot_stage, method, *args, **kwargs
        )
        assert (result.get("error_type"), result.get("details", {}).get("code")) == (
            error_type,
            code,
        ), method


def test_invalid_delete_mode_is_refused_before_sending(capturing_tools: Tools) -> None:
    tools, captured = capturing_tools

    result = asyncio.run(tools.delete_isaac_prim("/World/Box", mode="purge"))

    assert result["error_type"] == "ValueError"
    assert captured == []


# ---------------------------------------------------------------------------
# duplicate_isaac_prim
# ---------------------------------------------------------------------------


def test_duplicating_a_referenced_link_makes_a_defined_flattened_copy(
    capturing_tools: Tools, run_on_stage: Any, robot_stage: Usd.Stage
) -> None:
    """It used to copy the root layer's undefined over, which no traversal sees."""
    result = _call(
        capturing_tools,
        run_on_stage,
        robot_stage,
        "duplicate_isaac_prim",
        LINK,
        "/World/link1_copy",
    )

    assert result["duplicated"] is True
    assert result["method"] == "flattened_copy"
    copy = robot_stage.GetPrimAtPath("/World/link1_copy")
    assert copy.IsDefined()
    assert copy in list(robot_stage.Traverse())
    assert (
        copy.GetAttribute("userTuned").Get() == 42.0
    ), "the override is part of the copy"
    assert copy.HasAPI(UsdPhysics.RigidBodyAPI)
    assert list(
        UsdGeom.Xformable(copy).GetLocalTransformation().ExtractTranslation()
    ) == [0.0, 0.0, 1.0]
    # The instanced collider stays instanced and keeps its mesh.
    collisions = robot_stage.GetPrimAtPath("/World/link1_copy/collisions")
    assert collisions.IsInstance()
    assert result["instances_restored"] == ["/World/link1_copy/collisions"]
    mesh = robot_stage.GetPrimAtPath("/World/link1_copy/collisions/mesh")
    assert mesh.IsValid() and mesh.HasAPI(UsdPhysics.CollisionAPI)


def test_duplicating_a_referenced_robot_root_keeps_the_reference(
    capturing_tools: Tools, run_on_stage: Any, robot_stage: Usd.Stage
) -> None:
    result = _call(
        capturing_tools,
        run_on_stage,
        robot_stage,
        "duplicate_isaac_prim",
        "/World/Franka",
        "/World/Franka2",
    )

    assert result["method"] == "spec_copy"
    copy = robot_stage.GetPrimAtPath("/World/Franka2")
    assert copy.HasAuthoredReferences()
    assert (
        robot_stage.GetPrimAtPath("/World/Franka2/link1")
        .GetAttribute("userTuned")
        .Get()
        == 42.0
    )
    body0 = (
        UsdPhysics.Joint(robot_stage.GetPrimAtPath("/World/Franka2/joints/joint1"))
        .GetBody0Rel()
        .GetTargets()
    )
    assert body0 == [Sdf.Path("/World/Franka2/base_link")]


def test_duplicate_into_a_missing_parent_is_refused(
    capturing_tools: Tools, run_on_stage: Any
) -> None:
    stage = _local_stage()
    UsdGeom.Cube.Define(stage, "/World/Box")

    result = _call(
        capturing_tools,
        run_on_stage,
        stage,
        "duplicate_isaac_prim",
        "/World/Box",
        "/Nowhere/Box",
    )

    assert "Parent of the destination not found" in result["error"]


# ---------------------------------------------------------------------------
# reparent_isaac_prim
# ---------------------------------------------------------------------------


def test_moving_a_material_keeps_its_binding(
    capturing_tools: Tools, run_on_stage: Any
) -> None:
    """The cube used to be left bound to the old path, i.e. to nothing."""
    stage = _local_stage()
    UsdGeom.Scope.Define(stage, "/World/Looks")
    UsdGeom.Scope.Define(stage, "/World/Materials")
    material = UsdShade.Material.Define(stage, "/World/Looks/Red")
    cube = UsdGeom.Cube.Define(stage, "/World/Box")
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)

    result = _call(
        capturing_tools,
        run_on_stage,
        stage,
        "reparent_isaac_prim",
        "/World/Looks/Red",
        "/World/Materials",
    )

    assert result["reparented"] is True
    assert "/World/Box.material:binding" in result["retargeted"]
    bound, _ = UsdShade.MaterialBindingAPI(cube.GetPrim()).ComputeBoundMaterial()
    assert bound.GetPath() == Sdf.Path("/World/Materials/Red")


def test_moving_a_link_retargets_its_joint(
    capturing_tools: Tools, run_on_stage: Any
) -> None:
    stage = _local_stage()
    UsdGeom.Xform.Define(stage, "/World/Bot/a")
    UsdGeom.Xform.Define(stage, "/World/Bot/b")
    UsdGeom.Xform.Define(stage, "/World/Other")
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/Bot/j")
    joint.CreateBody0Rel().SetTargets(["/World/Bot/a"])
    joint.CreateBody1Rel().SetTargets(["/World/Bot/b"])

    _call(
        capturing_tools,
        run_on_stage,
        stage,
        "reparent_isaac_prim",
        "/World/Bot/a",
        "/World/Other",
    )

    assert joint.GetBody0Rel().GetTargets() == [Sdf.Path("/World/Other/a")]
    assert joint.GetBody1Rel().GetTargets() == [Sdf.Path("/World/Bot/b")]


def test_relationships_inside_the_moved_subtree_follow_it(
    capturing_tools: Tools, run_on_stage: Any
) -> None:
    stage = _local_stage()
    UsdGeom.Xform.Define(stage, "/World/Bot/a")
    UsdGeom.Xform.Define(stage, "/World/Group")
    UsdPhysics.FixedJoint.Define(stage, "/World/Bot/j").CreateBody0Rel().SetTargets(
        ["/World/Bot/a"]
    )

    _call(
        capturing_tools,
        run_on_stage,
        stage,
        "reparent_isaac_prim",
        "/World/Bot",
        "/World/Group",
    )

    moved = UsdPhysics.Joint(stage.GetPrimAtPath("/World/Group/Bot/j"))
    assert moved.GetBody0Rel().GetTargets() == [Sdf.Path("/World/Group/Bot/a")]


def test_moving_to_the_stage_root_joins_the_path(
    capturing_tools: Tools, run_on_stage: Any
) -> None:
    """The reply used to say //Box."""
    stage = _local_stage()
    UsdGeom.Cube.Define(stage, "/World/Box")

    result = _call(
        capturing_tools, run_on_stage, stage, "reparent_isaac_prim", "/World/Box", "/"
    )

    assert result["new_path"] == "/Box"
    assert stage.GetPrimAtPath("/Box").IsValid()
    assert not stage.GetPrimAtPath("/World/Box").IsValid()


@pytest.mark.parametrize("keep", [True, False])
def test_world_transform_is_kept_by_default(
    keep: bool, capturing_tools: Tools, run_on_stage: Any
) -> None:
    stage = _local_stage()
    UsdGeom.Xform.Define(stage, "/World/A").AddTranslateOp().Set(Gf.Vec3d(10, 0, 0))
    b = UsdGeom.Xform.Define(stage, "/World/B")
    b.AddTranslateOp().Set(Gf.Vec3d(0, 5, 0))
    b.AddRotateZOp().Set(90.0)
    box = UsdGeom.Cube.Define(stage, "/World/A/Box")
    box.AddTranslateOp().Set(Gf.Vec3d(1, 0, 0))
    before = box.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    result = _call(
        capturing_tools,
        run_on_stage,
        stage,
        "reparent_isaac_prim",
        "/World/A/Box",
        "/World/B",
        keep_world_transform=keep,
    )

    moved = UsdGeom.Xformable(stage.GetPrimAtPath("/World/B/Box"))
    after = moved.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    if keep:
        assert result["world_transform_kept"] is True
        assert Gf.IsClose(after, before, 1e-6)
    else:
        assert result["world_transform_kept"] is None
        assert list(moved.GetLocalTransformation().ExtractTranslation()) == [
            1.0,
            0.0,
            0.0,
        ]


def test_world_transform_that_needs_shear_is_refused_unchanged(
    capturing_tools: Tools, run_on_stage: Any
) -> None:
    stage = _local_stage()
    UsdGeom.Xform.Define(stage, "/World/Squash").AddScaleOp().Set(Gf.Vec3f(1, 1, 3))
    UsdGeom.Xform.Define(stage, "/World/Spun").AddRotateXOp().Set(30.0)
    UsdGeom.Cube.Define(stage, "/World/Spun/Box")

    result = _call(
        capturing_tools,
        run_on_stage,
        stage,
        "reparent_isaac_prim",
        "/World/Spun/Box",
        "/World/Squash",
    )

    assert "keep_world_transform=false" in result["error"]
    assert stage.GetPrimAtPath("/World/Spun/Box").IsValid()


def test_moving_a_referenced_link_is_refused_unchanged(
    capturing_tools: Tools, run_on_stage: Any, robot_stage: Usd.Stage
) -> None:
    result = _call(
        capturing_tools,
        run_on_stage,
        robot_stage,
        "reparent_isaac_prim",
        LINK,
        "/World",
    )

    assert "duplicate_isaac_prim" in result["error"]
    assert robot_stage.GetPrimAtPath(LINK).IsActive()
    assert not robot_stage.GetPrimAtPath("/World/link1").IsValid()


def test_moving_a_prim_under_itself_is_refused(
    capturing_tools: Tools, run_on_stage: Any
) -> None:
    stage = _local_stage()
    UsdGeom.Xform.Define(stage, "/World/A/B")

    result = _call(
        capturing_tools,
        run_on_stage,
        stage,
        "reparent_isaac_prim",
        "/World/A",
        "/World/A/B",
    )

    assert "under itself" in result["error"]


def test_kit_move_is_followed_by_retargeting_and_says_what_undo_covers(
    capturing_tools: Tools, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = _local_stage()
    UsdGeom.Scope.Define(stage, "/World/Looks")
    UsdGeom.Scope.Define(stage, "/World/Materials")
    material = UsdShade.Material.Define(stage, "/World/Looks/Red")
    cube = UsdGeom.Cube.Define(stage, "/World/Box")
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)

    def _move_prim(
        path_from: str,
        path_to: str,
        keep_world_transform: bool = False,
        destructive: bool = True,
    ) -> None:
        # A move that, like a bare namespace edit, leaves the targets alone.
        edit = Sdf.BatchNamespaceEdit()
        edit.Add(
            Sdf.NamespaceEdit.Reparent(path_from, Sdf.Path(path_to).GetParentPath(), -1)
        )
        stage.GetRootLayer().Apply(edit)

    tools, captured = capturing_tools
    asyncio.run(tools.reparent_isaac_prim("/World/Looks/Red", "/World/Materials"))
    result = _run_with_kit(captured[-1], stage, {"MovePrim": _move_prim}, monkeypatch)

    assert result["method"] == "omni.kit.commands MovePrim"
    assert result["_kit_kwargs"] == [
        {
            "path_from": "/World/Looks/Red",
            "path_to": "/World/Materials/Red",
            "keep_world_transform": True,
            "destructive": False,
        }
    ]
    assert result["retargeted"] == ["/World/Box.material:binding"]
    assert result["undoable"] is False
    assert "undo" in result["undo_note"]
    bound, _ = UsdShade.MaterialBindingAPI(cube.GetPrim()).ComputeBoundMaterial()
    assert bound.GetPath() == Sdf.Path("/World/Materials/Red")


def test_a_kit_that_declines_the_move_is_respected(
    capturing_tools: Tools, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The move used to fall back to an Sdf namespace edit behind Kit's back."""
    stage = _local_stage()
    UsdGeom.Cube.Define(stage, "/World/Box")
    UsdGeom.Xform.Define(stage, "/World/Group")
    tools, captured = capturing_tools
    asyncio.run(tools.reparent_isaac_prim("/World/Box", "/World/Group"))

    result = _run_with_kit(
        captured[-1], stage, {"MovePrim": lambda **kwargs: None}, monkeypatch
    )

    assert result["details"]["code"] == "KIT_DECLINED"
    assert stage.GetPrimAtPath("/World/Box").IsValid()
    assert not stage.GetPrimAtPath("/World/Group/Box").IsValid()


# ---------------------------------------------------------------------------
# Traversal: instance proxies and inactive prims
# ---------------------------------------------------------------------------


def test_physics_listing_counts_instanced_colliders(
    capturing_tools: Tools, run_on_stage: Any, robot_stage: Usd.Stage
) -> None:
    """An instanceable collider used to give collider_count 0."""
    result = _call(
        capturing_tools,
        run_on_stage,
        robot_stage,
        "list_isaac_physics_objects",
        "/World",
    )

    assert result["collider_count"] == 1
    assert result["colliders"][0]["path"] == COLLIDER
    assert result["rigid_body_count"] == 2


@pytest.mark.parametrize("proxies", [True, False])
def test_mesh_search_reaches_inside_instances(
    proxies: bool, capturing_tools: Tools, run_on_stage: Any, robot_stage: Usd.Stage
) -> None:
    result = _call(
        capturing_tools,
        run_on_stage,
        robot_stage,
        "search_isaac_prims",
        "Mesh",
        root_path="/World",
        include_instance_proxies=proxies,
    )

    if proxies:
        assert result["matches"] == [
            {"path": COLLIDER, "type": "Mesh", "name": "mesh", "instance_proxy": True}
        ]
    else:
        assert result["matches"] == []


def test_listing_shows_deactivated_prims_on_request(
    capturing_tools: Tools, run_on_stage: Any
) -> None:
    stage = _local_stage()
    UsdGeom.Xform.Define(stage, "/World/DisabledSensor").GetPrim().SetActive(False)

    default = _call(capturing_tools, run_on_stage, stage, "list_isaac_prims", "/World")
    with_inactive = _call(
        capturing_tools,
        run_on_stage,
        stage,
        "list_isaac_prims",
        "/World",
        include_inactive=True,
    )

    assert [p["path"] for p in default["prims"]] == ["/World"]
    sensor = [p for p in with_inactive["prims"] if p["path"] == "/World/DisabledSensor"]
    assert sensor == [
        {
            "path": "/World/DisabledSensor",
            "type": "Xform",
            "name": "DisabledSensor",
            "active": False,
        }
    ]


def test_typed_query_reaches_instanced_colliders(
    capturing_tools: Tools, run_on_stage: Any, robot_stage: Usd.Stage
) -> None:
    result = _call(
        capturing_tools,
        run_on_stage,
        robot_stage,
        "query_usd_typed_prims",
        "UsdPhysics.CollisionAPI",
    )

    assert [(p["path"], p.get("instance_proxy")) for p in result["prims"]] == [
        (COLLIDER, True)
    ]


def test_subtree_unlimited_depth_and_instance_child_counts(
    capturing_tools: Tools, run_on_stage: Any, robot_stage: Usd.Stage
) -> None:
    """max_depth=-1 used to return only the root; an instance's children counted 0."""
    result = _call(
        capturing_tools,
        run_on_stage,
        robot_stage,
        "get_isaac_subtree",
        "/World/Franka",
        max_depth=-1,
    )

    by_path = {p["path"]: p for p in result["prims"]}
    assert COLLIDER in by_path
    assert by_path[COLLIDER]["instance_proxy"] is True
    assert by_path["/World/Franka/link1/collisions"]["child_count"] == 1
    assert by_path["/World/Franka/joints/joint1"]["depth"] == 2


def _bridge(stage: Usd.Stage, monkeypatch: pytest.MonkeyPatch) -> BridgeCommandService:
    context = types.SimpleNamespace(
        get_stage=lambda: stage, get_stage_url=lambda: "anon:memory.usda"
    )
    fake_usd = types.ModuleType("omni.usd")
    fake_usd.get_context = lambda: context  # type: ignore[attr-defined]
    fake_omni = types.ModuleType("omni")
    fake_omni.usd = fake_usd  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "omni", fake_omni)
    monkeypatch.setitem(sys.modules, "omni.usd", fake_usd)
    return BridgeCommandService(executor=MagicMock(), allow_unsafe_execution=False)


def _dispatch(
    service: BridgeCommandService, action: str, payload: Dict[str, Any]
) -> Dict[str, Any]:
    response = asyncio.run(
        service.dispatch(BridgeRequest(request_id="r1", action=action, payload=payload))
    )
    return dict(response.payload)


def test_bridge_listings_match_the_script_path(
    capturing_tools: Tools,
    run_on_stage: Any,
    robot_stage: Usd.Stage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot_stage.DefinePrim("/World/Off", "Xform").SetActive(False)
    service = _bridge(robot_stage, monkeypatch)

    for flags in ({}, {"include_instance_proxies": False}, {"include_inactive": True}):
        bridged = _dispatch(
            service,
            "search_prims",
            dict({"query": "Mesh", "root_path": "/World"}, **flags),
        )
        scripted = _call(
            capturing_tools,
            run_on_stage,
            robot_stage,
            "search_isaac_prims",
            "Mesh",
            root_path="/World",
            **flags,
        )
        assert bridged["matches"] == scripted["matches"], flags
        listed = _dispatch(
            service,
            "list_prims",
            dict({"root_path": "/World", "max_depth": -1}, **flags),
        )
        script_list = _call(
            capturing_tools,
            run_on_stage,
            robot_stage,
            "list_isaac_prims",
            "/World",
            max_depth=-1,
            **flags,
        )
        assert listed["prims"] == script_list["prims"], flags
        assert listed["include_instance_proxies"] == flags.get(
            "include_instance_proxies", True
        )


def test_a_bridge_that_ignores_the_flags_falls_back_to_the_script() -> None:
    """A bridge published before the flags walks with the default predicate."""
    stale = {"status": "ok", "payload": {"prims": [], "count": 0, "offset": 0}}
    client = MagicMock()
    client.address = "127.0.0.1:8226"
    client.timeout_seconds = 30.0
    client.bridge_enabled = True
    client.fallback_to_vscode = True
    client.bridge_request = AsyncMock(return_value=stale)
    script_result = ScriptResult(success=True, output=json.dumps({"from": "script"}))
    client.execute = AsyncMock(return_value=script_result)
    client.execute_vscode_only = AsyncMock(return_value=script_result)
    tools = IsaacTools(client, settings=Settings())

    listed = asyncio.run(tools.list_isaac_prims("/World"))
    searched = asyncio.run(tools.search_isaac_prims("Mesh"))

    assert listed["from"] == "script"
    assert searched["from"] == "script"


# ---------------------------------------------------------------------------
# create_isaac_prim and create_isaac_object
# ---------------------------------------------------------------------------


def test_mistyped_prim_type_is_refused_with_a_suggestion(
    capturing_tools: Tools, run_on_stage: Any
) -> None:
    """'cube' used to create an untyped prim reported as created."""
    stage = _local_stage()

    result = _call(
        capturing_tools, run_on_stage, stage, "create_isaac_prim", "/World/Box", "cube"
    )

    assert result["suggestion"] == "Cube"
    assert not stage.GetPrimAtPath("/World/Box").IsValid()


def test_abstract_prim_type_is_refused(
    capturing_tools: Tools, run_on_stage: Any
) -> None:
    stage = _local_stage()

    result = _call(
        capturing_tools, run_on_stage, stage, "create_isaac_prim", "/World/G", "Gprim"
    )

    assert "abstract" in result["error"]
    assert not stage.GetPrimAtPath("/World/G").IsValid()


def test_attributes_the_schema_lacks_are_refused(
    capturing_tools: Tools, run_on_stage: Any
) -> None:
    """'radius' on a Cube used to be skipped without a word."""
    stage = _local_stage()

    result = _call(
        capturing_tools,
        run_on_stage,
        stage,
        "create_isaac_prim",
        "/World/Box",
        "Cube",
        {"size": 2.0, "radius": 1.0},
    )

    assert result["unknown_attributes"] == ["radius"]
    assert not stage.GetPrimAtPath("/World/Box").IsValid()


def test_known_attributes_are_set_with_conversion(
    capturing_tools: Tools, run_on_stage: Any
) -> None:
    stage = _local_stage()

    result = _call(
        capturing_tools,
        run_on_stage,
        stage,
        "create_isaac_prim",
        "/World/Box",
        "Cube",
        {"size": 2.0, "primvars:displayColor": [[1, 0, 0]]},
    )

    assert result["created"] is True
    cube = UsdGeom.Cube(stage.GetPrimAtPath("/World/Box"))
    assert cube.GetSizeAttr().Get() == 2.0
    assert list(cube.GetDisplayColorAttr().Get()) == [Gf.Vec3f(1, 0, 0)]


def test_a_value_that_does_not_fit_leaves_nothing_behind(
    capturing_tools: Tools, run_on_stage: Any
) -> None:
    stage = _local_stage()

    result = _call(
        capturing_tools,
        run_on_stage,
        stage,
        "create_isaac_prim",
        "/World/Box",
        "Cube",
        {"size": "big"},
    )

    assert "the prim was not created" in result["error"]
    assert not stage.GetPrimAtPath("/World/Box").IsValid()


def test_create_object_checks_vectors_before_sending(capturing_tools: Tools) -> None:
    """A two-element translation used to raise after the prim existed."""
    tools, captured = capturing_tools

    result = asyncio.run(
        tools.create_isaac_object("/World/Obj", translation=[1.0, 2.0], rigid_body=True)
    )

    assert result["error_type"] == "ValueError"
    assert "3 components" in result["error"]
    assert captured == []


def test_create_object_step_that_raises_keeps_the_report(
    capturing_tools: Tools, run_on_stage: Any
) -> None:
    stage = _local_stage()

    result = _call(
        capturing_tools,
        run_on_stage,
        stage,
        "create_isaac_object",
        "/World/Obj",
        rigid_body=True,
        material_path="/World/Looks/bad name",
    )

    assert result["failed_step"] == "material"
    assert list(result["steps"]) == ["prim", "rigid_body"]
    assert result["left_on_stage"] == ["/World/Obj"]
    assert stage.GetPrimAtPath("/World/Obj").IsValid()


# ---------------------------------------------------------------------------
# With script execution off, the prim tools can't plant a script
# ---------------------------------------------------------------------------


def _locked_tools() -> Tools:
    captured: List[str] = []

    def _record(code: str) -> ScriptResult:
        captured.append(code)
        return ScriptResult(success=True, output=json.dumps({"ok": True}))

    base = Settings()
    settings = base.model_copy(
        update={
            "security": base.security.model_copy(
                update={"allow_script_execution": False}
            )
        }
    )
    client = MagicMock()
    client.address = "127.0.0.1:8226"
    client.timeout_seconds = 30.0
    client.bridge_enabled = False
    client.fallback_to_vscode = True
    client.execute = AsyncMock(side_effect=_record)
    client.execute_vscode_only = AsyncMock(side_effect=_record)
    return IsaacTools(client, settings=settings), captured


@pytest.mark.parametrize(
    ("method", "args"),
    [
        (
            "set_isaac_prim_attribute",
            ("/World/Graph/node", "inputs:script", "import os"),
        ),
        (
            "set_isaac_prim_attribute",
            ("/World/Graph/node", "inputs:scriptPath", "/tmp/evil.py"),
        ),
        (
            "set_isaac_prim_attribute",
            ("/World/Graph/node", "node:type", "omni.graph.scriptnode.ScriptNode"),
        ),
        (
            "create_isaac_prim",
            (
                "/World/Graph/node",
                "OmniGraphNode",
                {"node:type": "omni.graph.scriptnode.ScriptNode"},
            ),
        ),
        # omni.warp's kernel node compiles the code in inputs:codeStr.
        (
            "set_isaac_prim_attribute",
            ("/World/Graph/kernel", "inputs:codeStr", "import os"),
        ),
        (
            "set_isaac_prim_attribute",
            ("/World/Graph/node", "node:type", "omni.warp.WarpKernel"),
        ),
        # Replicator nodes run these.
        (
            "set_isaac_prim_attribute",
            ("/World/Graph/aug", "inputs:augmentationScript", "x"),
        ),
        (
            "set_isaac_prim_attribute",
            ("/World/Graph/aug", "inputs:conditionScript", "x"),
        ),
        ("set_isaac_prim_attribute", ("/World/Graph/node", "inputs:SCRIPT", "x")),
        ("create_isaac_object", ("/World/Graph/node", "ScriptNode")),
    ],
)
def test_script_writes_are_refused_when_scripts_are_disabled(
    method: str, args: Tuple[Any, ...]
) -> None:
    tools, captured = _locked_tools()

    result = asyncio.run(getattr(tools, method)(*args))

    assert result["error_type"] == "ScriptExecutionDisabled"
    assert captured == []


def test_ordinary_writes_pass_when_scripts_are_disabled() -> None:
    tools, captured = _locked_tools()

    asyncio.run(
        tools.set_isaac_prim_attribute(
            "/World/Graph/node", "node:type", "omni.graph.nodes.Add"
        )
    )
    asyncio.run(tools.set_isaac_prim_attribute("/World/Box", "size", 2.0))

    assert len(captured) == 2


def test_script_writes_pass_when_scripts_are_enabled(capturing_tools: Tools) -> None:
    tools, captured = capturing_tools

    asyncio.run(
        tools.set_isaac_prim_attribute("/World/Graph/node", "inputs:script", "print(1)")
    )

    assert len(captured) == 1


def _graph_stage() -> Usd.Stage:
    stage = _local_stage()
    stage.DefinePrim("/World/Graph", "OmniGraph")
    node = stage.DefinePrim("/World/Graph/script", "OmniGraphNode")
    node.CreateAttribute("node:type", Sdf.ValueTypeNames.Token).Set(
        "omni.graph.scriptnode.ScriptNode"
    )
    node.CreateAttribute("inputs:usePath", Sdf.ValueTypeNames.Bool).Set(False)
    kernel = stage.DefinePrim("/World/Graph/kernel", "OmniGraphNode")
    kernel.CreateAttribute("node:type", Sdf.ValueTypeNames.Token).Set(
        "omni.warp.WarpKernel"
    )
    kernel.CreateAttribute("inputs:dim1", Sdf.ValueTypeNames.Int).Set(1)
    const = stage.DefinePrim("/World/Graph/const", "OmniGraphNode")
    const.CreateAttribute("node:type", Sdf.ValueTypeNames.Token).Set(
        "omni.graph.nodes.ConstantString"
    )
    const.CreateAttribute("inputs:value", Sdf.ValueTypeNames.String).Set("")
    return stage


@pytest.mark.parametrize(
    ("node", "attribute", "value"),
    [
        ("/World/Graph/script", "inputs:usePath", True),
        ("/World/Graph/kernel", "inputs:dim1", 4),
    ],
)
def test_no_attribute_of_a_code_running_node_is_written(
    node: str, attribute: str, value: Any, run_on_stage: Any
) -> None:
    """The target's own node:type is read in Kit: inputs:usePath points a ScriptNode at a file."""
    stage = _graph_stage()
    tools, captured = _locked_tools()

    asyncio.run(tools.set_isaac_prim_attribute(node, attribute, value))
    result = run_on_stage(captured[-1], stage)

    assert result["error_type"] == "ScriptExecutionDisabled"
    assert stage.GetPrimAtPath(node).GetAttribute(attribute).Get() != value


def test_ordinary_nodes_stay_writable_when_scripts_are_disabled(
    run_on_stage: Any,
) -> None:
    stage = _graph_stage()
    tools, captured = _locked_tools()

    asyncio.run(
        tools.set_isaac_prim_attribute("/World/Graph/const", "inputs:value", "hello")
    )
    result = run_on_stage(captured[-1], stage)

    assert "error" not in result
    value = stage.GetPrimAtPath("/World/Graph/const").GetAttribute("inputs:value").Get()
    assert value == "hello"


def test_duplicating_a_subtree_with_a_script_node_is_refused(run_on_stage: Any) -> None:
    stage = _graph_stage()
    tools, captured = _locked_tools()

    asyncio.run(tools.duplicate_isaac_prim("/World/Graph", "/World/GraphCopy"))
    result = run_on_stage(captured[-1], stage)

    assert result["error_type"] == "ScriptExecutionDisabled"
    assert [node["path"] for node in result["details"]["nodes"]] == [
        "/World/Graph/script",
        "/World/Graph/kernel",
    ]
    assert not stage.GetPrimAtPath("/World/GraphCopy").IsValid()


def test_duplicating_a_script_node_works_when_scripts_are_enabled(
    capturing_tools: Tools, run_on_stage: Any
) -> None:
    stage = _graph_stage()

    result = _call(
        capturing_tools,
        run_on_stage,
        stage,
        "duplicate_isaac_prim",
        "/World/Graph",
        "/World/GraphCopy",
    )

    assert result["duplicated"] is True


def test_a_failed_create_removes_the_ancestors_it_made(
    capturing_tools: Tools, run_on_stage: Any
) -> None:
    """DefinePrim made /World/A and /World/A/B; only the leaf used to be removed."""
    stage = _local_stage()

    result = _call(
        capturing_tools,
        run_on_stage,
        stage,
        "create_isaac_prim",
        "/World/A/B/Box",
        "Cube",
        {"size": "x"},
    )

    assert result["details"]["code"] == "VALUE_REJECTED"
    assert result["details"]["removed"] == ["/World/A", "/World/A/B", "/World/A/B/Box"]
    assert not stage.GetPrimAtPath("/World/A").IsValid()
    assert stage.GetPrimAtPath("/World").IsValid()


def test_a_create_reports_the_ancestors_it_made(
    capturing_tools: Tools, run_on_stage: Any
) -> None:
    stage = _local_stage()

    result = _call(
        capturing_tools,
        run_on_stage,
        stage,
        "create_isaac_prim",
        "/World/A/Box",
        "Cube",
    )

    assert result["created"] is True
    assert result["ancestors_created"] == ["/World/A"]
