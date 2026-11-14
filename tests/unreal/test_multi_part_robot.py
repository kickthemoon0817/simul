"""A robot built from parts keeps its hierarchy, and its welded parts read as one body.

The round-3 audit on UE 5.7.4 found that:

* ``enable_unreal_physics`` with ``component_name`` and ``allow_detach`` on a
  part of a multi-part actor called ``set_simulate_physics`` in the editor
  world, which detached the part inside the editor actor itself: moving the
  actor left the parts behind, ``export_unreal_usd`` wrote an empty Xform with
  ``success: true`` and no warning, and no tool could reattach the part
  (R3-B1). Unreal's own Details panel writes the body's flag as a property:
  the editor actor keeps its hierarchy, and only the simulation's copy
  detaches, welding the parts attached under it into its body.
* the parts welded into that body read as independent ``kinematic`` bodies,
  a force on one was refused with a hint that would have split the robot, and
  a mass set on one became the weld root's (R3-M1).

The scripts run against ``physics_editor``'s fake editor, which welds as UE
5.7.4 did.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict

import pytest

from simul.adapters._unreal_scripts import INCOMPATIBLE_BODY
from tests.unreal.physics_editor import CUBE_MASS, PIE_LEVEL, Editor, physics_session
from tests.unreal.test_simulation_tools import _call_tool, _server


def run(coroutine: Any) -> Dict[str, Any]:
    return asyncio.run(coroutine)


def parts_of(actor: Any) -> Dict[str, Any]:
    return {c.name: c for c in actor.components if hasattr(c, "body")}


# ---------------------------------------------------------------------------
# R3-B1: the editor actor keeps its hierarchy
# ---------------------------------------------------------------------------


def test_simulating_a_part_keeps_the_editor_actor_whole(monkeypatch: pytest.MonkeyPatch) -> None:
    editor = Editor()
    robot = editor.robot()
    parts = parts_of(robot)
    session = physics_session(monkeypatch, editor)

    result = run(session.enable_physics(robot.get_path_name(), component_name="Chassis"))

    assert result.get("success", True) is True, result
    assert result["physics_mode"] == "simulated" and result["physics_enabled"] is True
    # Nothing was detached in the editor: the flag went in as a property, as the Details panel writes it.
    assert editor.detached == []
    assert parts["Chassis"].parent is robot.root_component
    assert parts["Arm"].parent is parts["Chassis"] and parts["Wheel"].parent is parts["Chassis"]
    assert parts["Chassis"].body["simulate_physics"] is True
    assert result["attach_parent"] == robot.root_component.get_path_name()
    assert result["detached_from"] is None
    assert result["simulation_detaches_from"] == robot.root_component.get_path_name()
    assert result["welded_parts"] == ["Arm", "Wheel"]
    assert "the editor actor keeps its hierarchy" in result["warning"]
    assert "Arm, Wheel are attached under it without simulating" in result["warning"]
    # One undoable transaction that recorded every write.
    assert [t["description"] for t in editor.transactions] == ["simul: enable physics"]
    assert editor.unrecorded_writes() == []


def test_turning_a_part_off_keeps_it_attached(monkeypatch: pytest.MonkeyPatch) -> None:
    editor = Editor()
    robot = editor.robot()
    parts = parts_of(robot)
    session = physics_session(monkeypatch, editor)

    run(session.enable_physics(robot.get_path_name(), component_name="Chassis"))
    result = run(session.enable_physics(robot.get_path_name(), component_name="Chassis", enable=False))

    assert result["physics_mode"] == "kinematic"
    assert parts["Chassis"].parent is robot.root_component and editor.detached == []
    assert result["simulation_detaches_from"] is None and result["welded_parts"] == []


def test_allow_detach_is_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    """The option that broke the editor actor's hierarchy is no longer offered."""
    editor = Editor()
    robot = editor.robot()
    session = physics_session(monkeypatch, editor)

    with pytest.raises(TypeError):
        run(session.enable_physics(robot.get_path_name(), component_name="Chassis", allow_detach=True))
    with pytest.raises(TypeError):
        run(session.setup_physics_bulk(actor_paths=[robot.get_path_name()], allow_detach=True))


def test_simulating_a_welded_part_says_it_leaves_its_parent_in_the_simulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    editor = Editor()
    robot = editor.robot()
    parts = parts_of(robot)
    session = physics_session(monkeypatch, editor)
    run(session.enable_physics(robot.get_path_name(), component_name="Chassis"))

    result = run(session.enable_physics(robot.get_path_name(), component_name="Arm"))

    assert parts["Arm"].parent is parts["Chassis"] and editor.detached == []
    assert result["simulation_detaches_from"] == parts["Chassis"].get_path_name()
    assert "becomes a body of its own and no longer moves with Chassis" in result["warning"]
    assert "with its physics off it stays welded into Chassis's body" in result["warning"]


def test_the_play_copy_is_detached_by_the_runtime_setter(monkeypatch: pytest.MonkeyPatch) -> None:
    """A simulation path edits the transient copy the way Unreal does, and says so."""
    editor = Editor()
    robot = editor.robot()
    editor.start()
    copy = editor.actors[PIE_LEVEL + robot.name]
    session = physics_session(monkeypatch, editor)

    result = run(session.enable_physics(copy.get_path_name(), component_name="Chassis"))

    assert result["world"] == "simulation"
    assert result["detached_from"] == copy.root_component.get_path_name()
    assert result["simulation_detaches_from"] is None
    assert "the editor actor is unchanged" in result["warning"]
    assert parts_of(robot)["Chassis"].parent is robot.root_component


def test_bulk_setup_keeps_attachments_and_names_what_the_simulation_detaches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    editor = Editor()
    table = editor.cube("Table", mobility="MOVABLE")
    cup = editor.cube("Cup", mobility="MOVABLE")
    cup.root_component.parent = table.root_component
    session = physics_session(monkeypatch, editor)

    result = run(session.setup_physics_bulk(actor_paths=[cup.get_path_name()], collision_preset="PhysicsActor"))

    assert result["configured_count"] == 1, result
    assert cup.root_component.parent is table.root_component and editor.detached == []
    assert result["configured"][0]["simulation_detaches_from"] == table.root_component.get_path_name()
    assert "detached_from" not in result["configured"][0]
    assert "the editor actors keep their attachment" in result["warning"]


def test_bulk_refusal_names_the_preset_it_applies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Round-3 minor: the refusal paired the preset's collision mode with the
    component's current preset name ("QUERY_ONLY (preset Custom)")."""
    editor = Editor()
    cube = editor.cube("Crate", mobility="MOVABLE")
    cube.root_component.preset = "Custom"
    session = physics_session(monkeypatch, editor)

    result = run(session.setup_physics_bulk(actor_paths=[cube.get_path_name()], collision_preset="OverlapAll"))

    assert result["error_type"] == INCOMPATIBLE_BODY
    assert "(preset OverlapAll)" in result["details"]["failed"][0]["error"]
    assert "preset Custom" not in result["details"]["failed"][0]["error"]


def test_the_ambiguity_hint_no_longer_says_each_part_simulates_on_its_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    editor = Editor()
    robot = editor.robot()
    session = physics_session(monkeypatch, editor)

    result = run(session.enable_physics(robot.get_path_name()))

    assert result["details"]["code"] == "AMBIGUOUS_COMPONENT"
    hint = result["details"]["hint"]
    assert "each simulates on its own" not in hint
    assert "welded into its body" in hint and "the part the others are attached under" in hint


# ---------------------------------------------------------------------------
# R3-M1: welded parts are one body
# ---------------------------------------------------------------------------


def test_the_editor_state_reads_the_parts_as_welded(monkeypatch: pytest.MonkeyPatch) -> None:
    editor = Editor()
    robot = editor.robot()
    session = physics_session(monkeypatch, editor)
    run(session.enable_physics(robot.get_path_name(), component_name="Chassis"))

    state = run(session.get_physics_state([robot.get_path_name()]))

    body = state["bodies"][0]
    modes = {part["component_name"]: (part["physics_mode"], part["welded_to"]) for part in body["components"]}
    chassis = robot.get_path_name() + ".Chassis"
    assert modes == {"Chassis": ("simulated", None), "Arm": ("welded", chassis), "Wheel": ("welded", chassis)}
    assert "each moving on its own" not in body["note"]
    assert "welded into its body" in body["note"]


def test_the_simulation_state_reads_one_welded_body(monkeypatch: pytest.MonkeyPatch) -> None:
    editor = Editor()
    robot = editor.robot()
    session = physics_session(monkeypatch, editor)
    run(session.enable_physics(robot.get_path_name(), component_name="Chassis"))
    editor.start()

    state = run(session.get_physics_state([robot.get_path_name()]))

    assert state["world"] == "simulation"
    parts = {part["component_name"]: part for part in state["bodies"][0]["components"]}
    copy_chassis = PIE_LEVEL + robot.name + ".Chassis"
    assert parts["Arm"]["physics_mode"] == "welded" and parts["Arm"]["welded_to"] == copy_chassis
    assert {part["mass_kg"] for part in parts.values()} == {3 * CUBE_MASS}


def test_a_force_on_a_welded_part_pushes_the_whole_body(monkeypatch: pytest.MonkeyPatch) -> None:
    editor = Editor()
    robot = editor.robot()
    session = physics_session(monkeypatch, editor)
    run(session.enable_physics(robot.get_path_name(), component_name="Chassis"))
    editor.start()
    copy = parts_of(editor.actors[PIE_LEVEL + robot.name])

    result = run(session.apply_force(robot.get_path_name(), force_z=3000.0, is_impulse=True, component_name="Arm"))

    assert result.get("success", True) is True, result
    assert result["component_path"] == copy["Arm"].get_path_name()
    assert result["welded_to"] == copy["Chassis"].get_path_name()
    assert result["mass_kg"] == pytest.approx(3 * CUBE_MASS)
    assert result["expected_delta_v"] == pytest.approx([0.0, 0.0, 3000.0 / (3 * CUBE_MASS)])
    assert "welded into Chassis's body, so the force acts on that whole body" in result["warning"]
    # Unreal's AddImpulse on the welded Arm moved the Chassis's body.
    assert copy["Chassis"].velocity.z == pytest.approx(3000.0 / (3 * CUBE_MASS))


def test_a_part_that_does_not_simulate_is_pointed_at_the_part_that_carries_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refusal's hint used to say to enable physics on the part itself, which
    would have detached it from the robot in the simulation."""
    editor = Editor()
    robot = editor.robot()
    editor.start()
    session = physics_session(monkeypatch, editor)

    result = run(session.apply_force(robot.get_path_name(), force_z=5.0, is_impulse=True, component_name="Arm"))

    assert result["error_type"] == INCOMPATIBLE_BODY
    assert "enable_unreal_physics with component_name Chassis on the editor actor" in result["error"]
    assert "enabling physics on Arm itself would detach it from Chassis" in result["error"]


def test_params_on_a_welded_part_in_the_simulation_are_refused_with_nothing_written(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unreal's setters on a welded part write the weld root's body: a mass of 5 kg
    set on Arm became the Chassis's override (UE 5.7.4)."""
    editor = Editor()
    robot = editor.robot()
    session = physics_session(monkeypatch, editor)
    run(session.enable_physics(robot.get_path_name(), component_name="Chassis"))
    editor.start()
    copy = editor.actors[PIE_LEVEL + robot.name]
    before = {name: dict(part.body) for name, part in parts_of(copy).items()}

    result = run(session.set_physics_params(copy.get_path_name(), mass=5.0, component_name="Arm"))

    assert result["error_type"] == INCOMPATIBLE_BODY
    assert result["details"]["code"] == "WELDED"
    assert result["details"]["welded_to"] == copy.get_path_name() + ".Chassis"
    assert "set them on Chassis (component_name Chassis)" in result["error"]
    assert {name: part.body for name, part in parts_of(copy).items()} == before


def test_params_on_a_part_that_will_weld_are_written_to_it_with_a_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    editor = Editor()
    robot = editor.robot()
    parts = parts_of(robot)
    session = physics_session(monkeypatch, editor)
    run(session.enable_physics(robot.get_path_name(), component_name="Chassis"))

    result = run(session.set_physics_params(robot.get_path_name(), mass=5.0, component_name="Arm"))

    assert result.get("success", True) is True, result
    assert parts["Arm"].body["mass_in_kg_override"] == 5.0
    assert parts["Chassis"].body["override_mass"] is False
    assert result["physics_mode"] == "welded"
    assert "whose mass is the parts' masses added up" in result["warning"]


def test_the_welded_force_reply_keeps_welded_to_through_the_tool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    editor = Editor()
    robot = editor.robot()
    run(physics_session(monkeypatch, editor).enable_physics(robot.get_path_name(), component_name="Chassis"))
    editor.start()
    instance = _server(monkeypatch, tmp_path, editor)

    pushed = _call_tool(
        instance,
        "apply_unreal_force",
        {"actor_path": robot.get_path_name(), "force_x": 10, "is_impulse": True, "component_name": "Wheel"},
    )
    enabled = _call_tool(
        instance, "enable_unreal_physics", {"actor_path": robot.get_path_name(), "component_name": "Chassis"}
    )

    assert pushed["welded_to"] == PIE_LEVEL + robot.name + ".Chassis", json.dumps(pushed)
    # The editor path edits the editor world; its new fields reach the client too.
    assert enabled["world"] == "editor"
    assert enabled["simulation_detaches_from"] == robot.root_component.get_path_name()
    assert enabled["welded_parts"] == ["Arm", "Wheel"]
