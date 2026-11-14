"""The physics tools check that a body really collides, and pick one body of a multi-part actor.

The round-2 audit on UE 5.7.4 found that:

* ``set_unreal_collision {collision_enabled: true}`` on an OverlapAll,
  OverlapAllDynamic or Trigger body reported QUERY_AND_PHYSICS, and
  ``enable_unreal_physics`` then reported it simulated, but the cube fell
  through the floor at about 2000 cm/s: a preset keeps its overlap responses
  when its mode is changed, and Unreal makes two bodies collide only when each
  blocks the other. That was the very fix the NO_PHYSICS_COLLISION refusal
  recommended (R2B-01).
* the physics tools answered a multi-part actor (a SceneComponent root and two
  meshes) with a ScriptError traceback, had no ``component_name``, and
  ``get_unreal_physics_state`` read its fields as null (R2B-03).
* a generated DynamicMeshActor without simple collision was refused only after
  it failed to simulate, with no cause (R2B-06).

The scripts run against ``physics_editor``'s fake editor.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest

from simul.adapters._unreal_scripts import INCOMPATIBLE_BODY
from tests.unreal.physics_editor import Editor, Enum, physics_session


def run(coroutine: Any) -> Dict[str, Any]:
    return asyncio.run(coroutine)


def codes(result: Dict[str, Any]) -> List[str]:
    return [problem["code"] for problem in result["details"]["problems"]]


EVERY_TYPE = ["WorldStatic", "WorldDynamic", "Pawn", "PhysicsBody", "Vehicle", "Destructible"]


# ---------------------------------------------------------------------------
# R2B-01: a body must block what it should land on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset", ["OverlapAll", "Trigger"])
def test_collision_enabled_on_an_overlap_preset_warns_and_physics_is_refused(
    monkeypatch: pytest.MonkeyPatch, preset: str
) -> None:
    editor = Editor()
    cube = editor.cube("Sensor", mobility="MOVABLE", preset=preset)
    session = physics_session(monkeypatch, editor)

    enabled = run(session.set_collision(cube.get_path_name(), collision_enabled=True))
    writes = len(editor.writes)
    refused = run(session.enable_physics(cube.get_path_name()))

    assert (enabled["collision_mode"], enabled["collision_preset"]) == ("QUERY_AND_PHYSICS", "Custom")
    assert enabled["blocked_object_types"] == []
    assert "nothing collides with this body" in enabled["warning"]
    # A preset for each kind of body: simulated, Movable and moved by its transform, static.
    assert all(preset in enabled["warning"] for preset in ("PhysicsActor", "BlockAllDynamic", "BlockAll "))
    assert refused["error_type"] == INCOMPATIBLE_BODY
    assert codes(refused) == ["NO_BLOCKING_RESPONSE"]
    assert "fall through the floor" in refused["error"]
    assert "collision_preset PhysicsActor" in refused["error"]
    # Refused before anything was written.
    assert len(editor.writes) == writes
    assert cube.root_component.body["simulate_physics"] is False


@pytest.mark.parametrize("preset", ["OverlapAll", "Trigger"])
def test_the_fix_the_refusal_names_gives_a_body_that_blocks(monkeypatch: pytest.MonkeyPatch, preset: str) -> None:
    editor = Editor()
    cube = editor.cube("Sensor", mobility="MOVABLE", preset=preset)
    session = physics_session(monkeypatch, editor)

    refused = run(session.enable_physics(cube.get_path_name()))
    # The refusal no longer offers collision_enabled true, which kept the overlaps.
    message = refused["details"]["problems"][0]["message"]
    assert codes(refused) == ["NO_PHYSICS_COLLISION"]
    assert "collision_preset PhysicsActor" in message
    assert "collision_enabled true alone keeps" in message

    fixed = run(session.set_collision(cube.get_path_name(), collision_preset="PhysicsActor"))
    enabled = run(session.enable_physics(cube.get_path_name()))

    assert fixed["blocked_object_types"] == EVERY_TYPE and fixed["warning"] is None
    assert enabled["physics_mode"] == "simulated" and enabled["warning"] is None
    assert enabled["blocked_object_types"] == EVERY_TYPE


def test_collision_enabled_is_the_fix_when_the_responses_already_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """NoCollision blocks every object type (UE 5.7.4), so turning the mode on is enough."""
    editor = Editor()
    cube = editor.cube("Crate", mobility="MOVABLE", preset="NoCollision")
    session = physics_session(monkeypatch, editor)

    refused = run(session.enable_physics(cube.get_path_name()))
    enabled_collision = run(session.set_collision(cube.get_path_name(), collision_enabled=True))
    enabled = run(session.enable_physics(cube.get_path_name()))

    assert "collision_enabled true (its responses block WorldStatic" in refused["error"]
    assert enabled_collision["warning"] is None
    assert enabled["physics_mode"] == "simulated"


def test_a_kinematic_body_that_blocks_nothing_is_refused_with_a_kinematic_fix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    editor = Editor()
    cube = editor.cube("Door", mobility="MOVABLE", preset="OverlapAll")
    cube.root_component.collision = "QUERY_AND_PHYSICS"
    session = physics_session(monkeypatch, editor)

    refused = run(session.enable_physics(cube.get_path_name(), kinematic=True))

    assert codes(refused) == ["NO_BLOCKING_RESPONSE"]
    assert "simulated bodies would pass through it" in refused["error"]
    assert "BlockAllDynamic" in refused["error"]


def test_a_body_that_does_not_block_world_static_simulates_with_a_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    editor = Editor()
    cube = editor.cube("Ghost", mobility="MOVABLE", preset="PhysicsActor")
    cube.root_component.responses["ECC_WORLD_STATIC"] = "ECR_IGNORE"
    session = physics_session(monkeypatch, editor)

    result = run(session.enable_physics(cube.get_path_name()))

    assert result["physics_mode"] == "simulated"
    assert "WorldStatic" not in result["blocked_object_types"]
    assert "not WorldStatic" in result["warning"] and "passes through them" in result["warning"]


def test_bulk_setup_checks_the_responses_too(monkeypatch: pytest.MonkeyPatch) -> None:
    editor = Editor()
    sensor = editor.cube("Room_Box_Sensor", mobility="MOVABLE", preset="OverlapAll")
    sensor.root_component.collision = "QUERY_AND_PHYSICS"  # collision_enabled true on top of OverlapAll
    crate = editor.cube("Room_Box_Crate", mobility="MOVABLE")
    for actor in (sensor, crate):
        actor.tags.append("props")
    session = physics_session(monkeypatch, editor)

    refused = run(session.setup_physics_bulk(tag="props"))
    with_preset = run(session.setup_physics_bulk(tag="props", collision_preset="PhysicsActor"))

    assert refused["error_type"] == INCOMPATIBLE_BODY
    assert refused["details"]["failed"][0]["problems"] == ["NO_BLOCKING_RESPONSE"]
    assert with_preset["configured_count"] == 2
    assert sensor.root_component.responses["ECC_WORLD_STATIC"] == "ECR_BLOCK"


def test_bulk_setup_judges_a_preset_by_its_own_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    editor = Editor()
    crate = editor.cube("Room_Box_Crate", mobility="MOVABLE")
    crate.tags.append("props")
    session = physics_session(monkeypatch, editor)

    refused = run(session.setup_physics_bulk(tag="props", collision_preset="Trigger"))

    assert refused["error_type"] == INCOMPATIBLE_BODY
    assert refused["details"]["failed"][0]["problems"] == ["NO_PHYSICS_COLLISION"]
    assert crate.root_component.preset == "BlockAll"


def test_a_failed_bulk_setup_puts_the_responses_back(monkeypatch: pytest.MonkeyPatch) -> None:
    editor = Editor()
    good = editor.cube("Room_Box_Good", mobility="MOVABLE", preset="OverlapAll")
    bad = editor.cube("Room_Box_Bad", mobility="MOVABLE", preset="OverlapAll")
    bad.root_component.refuses_simulation = True
    for actor in (good, bad):
        actor.tags.append("props")
    session = physics_session(monkeypatch, editor)

    result = run(session.setup_physics_bulk(tag="props", collision_preset="PhysicsActor"))

    assert result["error_type"] == "PhysicsNotApplied"
    for actor in (good, bad):
        assert actor.root_component.preset == "OverlapAll"
        assert set(actor.root_component.responses.values()) == {"ECR_OVERLAP"}


# ---------------------------------------------------------------------------
# R2B-03: multi-part actors
# ---------------------------------------------------------------------------


EDITS = {
    # Each part is attached to the actor's root; simulating one keeps it there (R3-B1).
    "enable_physics": {},
    "set_collision": {"collision_preset": "BlockAll"},
    "set_physics_params": {"mass": 5.0},
}


@pytest.mark.parametrize("method", sorted(EDITS))
def test_a_multi_part_actor_needs_component_name(monkeypatch: pytest.MonkeyPatch, method: str) -> None:
    editor = Editor()
    robot = editor.multi_part_actor("Robot")
    session = physics_session(monkeypatch, editor)

    result = run(getattr(session, method)(robot.get_path_name(), **EDITS[method]))

    assert result["success"] is False
    assert result["error_type"] == "ValidationError"
    assert result["details"]["code"] == "AMBIGUOUS_COMPONENT"
    assert [c["component_name"] for c in result["details"]["components"]] == ["Part0", "Part1"]
    assert "pass component_name" in result["error"]
    assert editor.transactions == [] and editor.writes == []


@pytest.mark.parametrize("method", sorted(EDITS))
def test_component_name_picks_the_body(monkeypatch: pytest.MonkeyPatch, method: str) -> None:
    editor = Editor()
    robot = editor.multi_part_actor("Robot")
    session = physics_session(monkeypatch, editor)

    result = run(getattr(session, method)(robot.get_path_name(), component_name="Part1", **EDITS[method]))

    assert "error" not in result, result
    assert result["component_path"] == robot.get_path_name() + ".Part1"
    assert {path.rpartition(".")[2] for path, _, _ in editor.writes} == {"Part1"}
    assert editor.unrecorded_writes() == []


@pytest.mark.parametrize("method", sorted(EDITS))
def test_an_unknown_component_name_is_refused(monkeypatch: pytest.MonkeyPatch, method: str) -> None:
    editor = Editor()
    robot = editor.multi_part_actor("Robot")
    session = physics_session(monkeypatch, editor)

    result = run(getattr(session, method)(robot.get_path_name(), component_name="Wheel", **EDITS[method]))

    assert result["error_type"] == "ValidationError"
    assert result["details"]["code"] == "COMPONENT_NOT_FOUND"
    assert "Part0 (StaticMeshComponent), Part1 (StaticMeshComponent)" in result["error"]
    assert editor.writes == []


def test_a_refusal_names_the_part(monkeypatch: pytest.MonkeyPatch) -> None:
    """Review of #359: the refusal names the part, not only the actor."""
    editor = Editor()
    robot = editor.multi_part_actor("Robot")
    robot.components[-1]._mobility = Enum("STATIC")
    session = physics_session(monkeypatch, editor)

    refused = run(session.enable_physics(robot.get_path_name(), component_name="Part1", make_movable=False))

    assert codes(refused) == ["NOT_MOVABLE"]
    assert refused["error"].startswith("Cannot simulate physics on Robot (component Part1): ")


def test_the_editor_only_sprite_is_no_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """An Actor spawned from the class carries an editor-only BillboardComponent
    (UE 5.7.4): one mesh beside it is the actor's body, and it is never listed."""
    editor = Editor()
    lamp = editor.multi_part_actor("Lamp", parts=1)
    session = physics_session(monkeypatch, editor)

    result = run(session.set_collision(lamp.get_path_name(), collision_preset="BlockAll"))
    state = run(session.get_physics_state([lamp.get_path_name()]))

    assert result["component_path"] == lamp.get_path_name() + ".Part0"
    assert state["bodies"][0]["component_path"] == lamp.get_path_name() + ".Part0"


def test_bulk_setup_points_a_multi_part_actor_to_the_per_part_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Review of #359: bulk setup told a multi-part actor to pass component_name,
    which it does not take, and the failed entry carried no problem code."""
    editor = Editor()
    robot = editor.multi_part_actor("Robot")
    robot.tags.append("props")
    empty = editor.multi_part_actor("Empty", parts=0)
    empty.tags.append("props")
    session = physics_session(monkeypatch, editor)

    result = run(session.setup_physics_bulk(tag="props"))

    by_path = {entry["actor_path"]: entry for entry in result["details"]["failed"]}
    ambiguous, bare = by_path[robot.get_path_name()], by_path[empty.get_path_name()]
    assert ambiguous["problems"] == ["AMBIGUOUS_COMPONENT"]
    assert "enable_unreal_physics and component_name" in ambiguous["error"]
    assert ambiguous["error"].endswith("its parts: Part0, Part1")
    assert "pass component_name" not in ambiguous["error"]
    assert bare["problems"] == ["NO_PRIMITIVE_COMPONENT"]


def test_an_actor_without_a_primitive_component_is_an_incompatible_body(monkeypatch: pytest.MonkeyPatch) -> None:
    editor = Editor()
    empty = editor.multi_part_actor("Empty", parts=0)
    session = physics_session(monkeypatch, editor)

    result = run(session.enable_physics(empty.get_path_name()))

    assert result["error_type"] == INCOMPATIBLE_BODY
    assert codes(result) == ["NO_PRIMITIVE_COMPONENT"]


# ---------------------------------------------------------------------------
# R2B-06: generated meshes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kinematic", [False, True])
def test_a_dynamic_mesh_without_simple_collision_is_refused_with_the_fix(
    monkeypatch: pytest.MonkeyPatch, kinematic: bool
) -> None:
    editor = Editor()
    box = editor.dynamic_mesh_actor("Generated")
    session = physics_session(monkeypatch, editor)

    result = run(session.enable_physics(box.get_path_name(), kinematic=kinematic))

    assert result["error_type"] == INCOMPATIBLE_BODY
    assert codes(result) == ["NO_SIMPLE_COLLISION"]
    assert "compute_unreal_convex_hull" in result["error"]
    assert editor.writes == []


def test_a_dynamic_mesh_with_simple_collision_simulates(monkeypatch: pytest.MonkeyPatch) -> None:
    editor = Editor()
    box = editor.dynamic_mesh_actor("Generated", simple=1)
    session = physics_session(monkeypatch, editor)

    result = run(session.enable_physics(box.get_path_name()))

    assert result["physics_mode"] == "simulated"


def test_a_dynamic_mesh_using_its_triangles_as_simple_collision_can_only_be_kinematic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    editor = Editor()
    box = editor.dynamic_mesh_actor("Generated")
    box.root_component.collision_type = "CTF_USE_COMPLEX_AS_SIMPLE"
    session = physics_session(monkeypatch, editor)

    simulated = run(session.enable_physics(box.get_path_name()))
    kinematic = run(session.enable_physics(box.get_path_name(), kinematic=True))

    assert codes(simulated) == ["COMPLEX_AS_SIMPLE"]
    assert kinematic["physics_mode"] == "kinematic"
