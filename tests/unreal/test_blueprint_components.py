"""Physics, collision and light edits on a Blueprint actor's construction-script components.

The round-4 audit on UE 5.7.4 found that ``enable_unreal_physics`` and
``setup_unreal_physics_bulk`` on a component added in the Blueprint editor
reported ``PhysicsNotApplied`` and "nothing was changed" while the flag was in
fact set, and cancelled the undo record, so the next Undo reverted the user's
edit before it (R4-M1). Writing the simulate flag as the Details panel writes
it (``set_editor_property("body_instance", ...)``) re-runs the Blueprint's
construction script, which replaces every component it makes with a new one
of the same name and renames the old one ``TRASH_<class>_<n>``; the scripts
went on checking, restoring and reporting the trashed one.

``physics_editor``'s fake models that rerun (``Actor.rerun_construction``):
the scripts now find each part again by its actor and name after such a write.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict

import pytest

from tests.unreal.physics_editor import Editor, physics_session


def run(coroutine: Any) -> Dict[str, Any]:
    return asyncio.run(coroutine)


def part(actor: Any, name: str) -> Any:
    """The component the actor holds now under ``name``."""
    return next(c for c in actor.components if c.get_name() == name)


def test_enabling_physics_on_a_blueprint_crate_reports_the_live_component(monkeypatch: pytest.MonkeyPatch) -> None:
    editor = Editor()
    crate = editor.blueprint_crate()
    original = part(crate, "Mesh")
    session = physics_session(monkeypatch, editor)

    result = run(session.enable_physics(crate.get_path_name()))

    assert result.get("success", True) is True, result
    assert editor.reruns >= 1 and original.get_name().startswith("TRASH_")
    mesh = part(crate, "Mesh")
    assert mesh is not original and mesh.body["simulate_physics"] is True
    assert result["component_path"] == crate.get_path_name() + ".Mesh"
    assert result["physics_mode"] == "simulated" and result["mass_kg"] > 0
    assert result["simulation_detaches_from"] == crate.root_component.get_path_name()
    assert "TRASH_" not in str(result)
    # One undo step, kept: cancelling it would leave the next Undo to revert the edit before it.
    assert [(t["description"], t["cancelled"]) for t in editor.transactions] == [("simul: enable physics", False)]
    assert editor.unrecorded_writes() == []


def test_turning_a_blueprint_crate_off_reads_the_live_component(monkeypatch: pytest.MonkeyPatch) -> None:
    editor = Editor()
    crate = editor.blueprint_crate()
    session = physics_session(monkeypatch, editor)
    run(session.enable_physics(crate.get_path_name()))

    result = run(session.enable_physics(crate.get_path_name(), enable=False))

    assert result.get("success", True) is True, result
    assert part(crate, "Mesh").body["simulate_physics"] is False
    # The trashed component read NO_COLLISION and 0 kg for a body with QUERY_AND_PHYSICS collision.
    assert result["physics_mode"] == "kinematic" and result["collision_mode"] == "QUERY_AND_PHYSICS"
    assert result["mass_kg"] > 0 and result["component_path"].endswith(".Mesh")


def test_a_blueprint_robot_s_chassis_simulates_with_its_parts_welded(monkeypatch: pytest.MonkeyPatch) -> None:
    editor = Editor()
    robot = editor.blueprint_robot()
    session = physics_session(monkeypatch, editor)

    result = run(session.enable_physics(robot.get_path_name(), component_name="Chassis"))

    assert result.get("success", True) is True, result
    chassis = part(robot, "Chassis")
    assert chassis.body["simulate_physics"] is True
    assert part(robot, "Arm").parent is chassis and part(robot, "Wheel").parent is chassis
    assert result["welded_parts"] == ["Arm", "Wheel"]
    assert result["simulation_detaches_from"] == robot.get_path_name() + ".DefaultSceneRoot"
    assert "TRASH_" not in str(result)
    assert editor.transactions[-1]["cancelled"] is False and editor.unrecorded_writes() == []


def test_a_blueprint_body_that_does_not_simulate_is_put_back_on_its_live_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    editor = Editor()
    crate = editor.blueprint_crate()
    part(crate, "Mesh").refuses_simulation = True
    session = physics_session(monkeypatch, editor)

    result = run(session.enable_physics(crate.get_path_name()))

    assert result["success"] is False and result["error_type"] == "PhysicsNotApplied"
    assert "nothing was changed" in result["error"]
    assert result["details"]["component_path"].endswith(".Mesh") and result["details"]["undo_kept"] is False
    assert part(crate, "Mesh").body["simulate_physics"] is False
    assert editor.transactions[0]["cancelled"] is True


def test_a_body_that_cannot_be_put_back_keeps_its_undo_step(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancelling drops the record and keeps the change, so a change still there keeps its undo step."""
    editor = Editor()
    crate = editor.blueprint_crate()
    mesh = part(crate, "Mesh")
    mesh.base_mass, mesh.sticky_simulation = 0.0, True  # the flag takes, Unreal gives no mass, and it sticks
    session = physics_session(monkeypatch, editor)

    result = run(session.enable_physics(crate.get_path_name()))

    assert result["success"] is False and result["error_type"] == "PhysicsNotApplied"
    assert "nothing was changed" not in result["error"]
    assert "kept as one undo step" in result["error"] and result["details"]["undo_kept"] is True
    assert editor.transactions[0]["cancelled"] is False


def test_bulk_setup_on_a_blueprint_crate_and_a_plain_cube(monkeypatch: pytest.MonkeyPatch) -> None:
    editor = Editor()
    crate = editor.blueprint_crate()
    cube = editor.cube("PlainCube", mobility="MOVABLE", preset="PhysicsActor")
    session = physics_session(monkeypatch, editor)

    result = run(session.setup_physics_bulk(actor_paths=[crate.get_path_name(), cube.get_path_name()], mass=7.0))

    assert result.get("success", True) is True, result
    assert (result["configured_count"], result["failed_count"]) == (2, 0)
    mesh = part(crate, "Mesh")
    # The mass went to the live component, after the flag's write replaced it.
    assert mesh.body["simulate_physics"] is True and mesh.get_mass() == 7.0
    assert {entry["physics_mode"] for entry in result["configured"]} == {"simulated"}
    assert [(t["description"], t["cancelled"]) for t in editor.transactions] == [("simul: set up physics", False)]
    assert editor.unrecorded_writes() == []


def test_bulk_setup_puts_a_blueprint_crate_back_when_another_actor_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    editor = Editor()
    crate = editor.blueprint_crate()
    odd = editor.cube("Odd", mobility="MOVABLE")
    odd.root_component.refuses_simulation = True
    session = physics_session(monkeypatch, editor)

    result = run(session.setup_physics_bulk(actor_paths=[crate.get_path_name(), odd.get_path_name()]))

    assert result["success"] is False and result["error_type"] == "PhysicsNotApplied"
    assert "nothing was changed" in result["error"] and result["details"]["undo_kept"] is False
    assert part(crate, "Mesh").body["simulate_physics"] is False
    assert editor.transactions[0]["cancelled"] is True


def test_bulk_setup_keeps_its_undo_step_when_a_blueprint_crate_cannot_be_put_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    editor = Editor()
    crate = editor.blueprint_crate()
    part(crate, "Mesh").sticky_simulation = True
    odd = editor.cube("Odd", mobility="MOVABLE")
    odd.root_component.refuses_simulation = True
    session = physics_session(monkeypatch, editor)

    result = run(session.setup_physics_bulk(actor_paths=[crate.get_path_name(), odd.get_path_name()]))

    assert result["success"] is False and result["details"]["undo_kept"] is True
    assert "Crate's its simulate flag stayed changed" in result["error"]
    assert editor.transactions[0]["cancelled"] is False


def test_collision_off_on_a_blueprint_crate_reaches_the_live_component(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clearing Use Default Collision is a property write, so it re-runs the construction script too."""
    editor = Editor()
    crate = editor.blueprint_crate()
    part(crate, "Mesh").use_default_collision = True
    session = physics_session(monkeypatch, editor)

    result = run(session.set_collision(crate.get_path_name(), collision_enabled=False))

    assert result.get("success", True) is True, result
    mesh = part(crate, "Mesh")
    assert mesh.collision == "NO_COLLISION" and mesh.use_default_collision is False
    assert result["collision_mode"] == "NO_COLLISION" and result["component_path"].endswith(".Mesh")
    assert editor.unrecorded_writes() == []
