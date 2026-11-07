"""``enable_unreal_physics``, ``set_unreal_physics_params`` and ``set_unreal_collision``.

A production-readiness audit on UE 5.7.4 found that:

* the three tools wrote the body with the runtime setters alone, so an edit
  was no undo step and left the map clean: Save All found nothing to save, the
  edit was lost on reload, and one Undo after spawn plus the three tools
  removed the cube (U2-02);
* ``set_unreal_collision(collision_enabled=false)`` left a spawned mesh's Use
  Default Collision flag set, so the running simulation and the reloaded map
  had BlockAll again (U2-03);
* ``enable_unreal_physics`` reported ``physics_enabled: true`` for bodies that
  never simulate: a Static body, one without physics collision (NoCollision,
  OverlapAll, the PlayerStart capsule) and one whose mesh has no simple
  collision, since its only check was ``is_simulating_physics()``, which
  reads True for all of them (U2-04);
* a failed enable left the simulate flag set (U2-11), ``set_unreal_physics_params``
  read nothing back and stored ``inf`` for mass 1e40 (U2-12), ``enable=true,
  simulate_physics=false`` turned physics off (U2-14), and enabling physics on
  an attached actor detached it silently (U2-08).

The scripts run here against ``physics_editor``'s fake editor, which answers
them as UE 5.7.4 did.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict

import pytest

from simul.adapters._unreal_scripts import INCOMPATIBLE_BODY
from tests.unreal.physics_editor import (
    CUBE_MASS,
    PIE_LEVEL,
    Editor,
    Mesh,
    f32,
    physics_session,
)


def run(coroutine: Any) -> Dict[str, Any]:
    return asyncio.run(coroutine)


def codes(result: Dict[str, Any]) -> list:
    return [problem["code"] for problem in result["details"]["problems"]]


# ---------------------------------------------------------------------------
# enable_physics
# ---------------------------------------------------------------------------


def test_enable_makes_a_static_body_movable_in_one_undoable_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    editor = Editor()
    cube = editor.cube("Crate")
    session = physics_session(monkeypatch, editor)

    result = run(session.enable_physics(cube.get_path_name()))

    assert "error" not in result, result
    assert (result["physics_enabled"], result["physics_mode"]) == (True, "simulated")
    assert (result["mobility"], result["mobility_changed_from"]) == (
        "Movable",
        "Static",
    )
    assert result["mass_kg"] == pytest.approx(CUBE_MASS)
    assert result["world"] == "editor"
    # One transaction, the component modified before anything was written, and the map dirty.
    assert [t["description"] for t in editor.transactions] == ["simul: enable physics"]
    assert editor.transactions[0]["cancelled"] is False
    assert editor.unrecorded_writes() == []
    assert editor.dirty is True


@pytest.mark.parametrize(
    "preset, mode",
    [("NoCollision", "NO_COLLISION"), ("OverlapAll", "QUERY_ONLY")],
)
def test_enable_refuses_a_body_without_physics_collision(
    monkeypatch: pytest.MonkeyPatch, preset: str, mode: str
) -> None:
    editor = Editor()
    cube = editor.cube("Sensor", mobility="MOVABLE", preset=preset)
    session = physics_session(monkeypatch, editor)

    result = run(session.enable_physics(cube.get_path_name()))

    assert result["success"] is False
    assert result["error_type"] == INCOMPATIBLE_BODY
    assert codes(result) == ["NO_PHYSICS_COLLISION"]
    assert mode in result["error"] and "set_unreal_collision" in result["error"]
    # Nothing written: no transaction, the flag not set, the map clean.
    assert editor.transactions == [] and editor.writes == []
    assert cube.root_component.body["simulate_physics"] is False
    assert editor.dirty is False


def test_enable_refuses_the_player_start_capsule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    editor = Editor()
    start = editor.capsule_actor("PlayerStart")
    session = physics_session(monkeypatch, editor)

    result = run(session.enable_physics(start.get_path_name()))

    assert result["error_type"] == INCOMPATIBLE_BODY
    assert codes(result) == ["NO_PHYSICS_COLLISION"]
    assert result["details"]["component_path"].endswith(".CollisionCapsule")


def test_enable_refuses_a_static_body_when_make_movable_is_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    editor = Editor()
    cube = editor.cube("Pinned")
    session = physics_session(monkeypatch, editor)

    result = run(session.enable_physics(cube.get_path_name(), make_movable=False))

    assert codes(result) == ["NOT_MOVABLE"]
    assert "make_movable" in result["error"]
    assert cube.root_component.mobility.name == "STATIC"
    assert editor.writes == []


def test_enable_refuses_a_static_mesh_actor_without_a_mesh_and_leaves_no_flag_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U2-11: the failed enable used to leave simulate_physics set."""
    editor = Editor()
    empty = editor.cube("Empty", mobility="MOVABLE", no_mesh=True)
    session = physics_session(monkeypatch, editor)

    result = run(session.enable_physics(empty.get_path_name()))

    assert codes(result) == ["NO_GEOMETRY"]
    assert empty.root_component.body["simulate_physics"] is False
    assert editor.writes == []


def test_a_mesh_without_simple_collision_is_refused_for_both_physical_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    editor = Editor()
    statue = editor.cube(
        "Statue", mobility="MOVABLE", mesh=Mesh("/Game/SM_Statue.SM_Statue", simple=0)
    )
    session = physics_session(monkeypatch, editor)

    for kinematic in (False, True):
        result = run(
            session.enable_physics(statue.get_path_name(), kinematic=kinematic)
        )
        assert codes(result) == ["NO_SIMPLE_COLLISION"], result
        assert "/Game/SM_Statue.SM_Statue" in result["error"]
    assert editor.writes == []


def test_complex_as_simple_collision_can_be_kinematic_but_not_simulated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    editor = Editor()
    mesh = Mesh(
        "/Game/SM_Cliff.SM_Cliff", simple=0, complexity="CTF_USE_COMPLEX_AS_SIMPLE"
    )
    cliff = editor.cube("Cliff", mobility="MOVABLE", mesh=mesh)
    session = physics_session(monkeypatch, editor)

    simulated = run(session.enable_physics(cliff.get_path_name()))
    kinematic = run(session.enable_physics(cliff.get_path_name(), kinematic=True))

    assert codes(simulated) == ["COMPLEX_AS_SIMPLE"]
    assert "error" not in kinematic, kinematic
    assert kinematic["physics_mode"] == "kinematic"


def test_kinematic_keeps_physics_collision_without_simulating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U2-14: enable with simulation off used to turn physics off instead."""
    editor = Editor()
    platform = editor.cube("Platform")
    session = physics_session(monkeypatch, editor)

    result = run(session.enable_physics(platform.get_path_name(), kinematic=True))

    assert (result["physics_mode"], result["physics_enabled"]) == ("kinematic", False)
    assert result["simulate_physics"] is False
    assert (result["mobility"], result["collision_mode"]) == (
        "Movable",
        "QUERY_AND_PHYSICS",
    )
    assert editor.unrecorded_writes() == []


def test_turning_physics_off_keeps_collision_and_mobility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    editor = Editor()
    cube = editor.cube("Crate", mobility="MOVABLE")
    cube.root_component.body["simulate_physics"] = True
    session = physics_session(monkeypatch, editor)

    result = run(session.enable_physics(cube.get_path_name(), enable=False))

    assert (result["physics_enabled"], result["simulate_physics"]) == (False, False)
    assert (result["mobility"], result["collision_mode"]) == (
        "Movable",
        "QUERY_AND_PHYSICS",
    )
    assert [t["description"] for t in editor.transactions] == ["simul: disable physics"]
    assert editor.unrecorded_writes() == []


def test_an_attached_body_is_refused_unless_allow_detach(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U2-08: Unreal detaches a body as it starts simulating; the tool did it silently."""
    editor = Editor()
    parent = editor.cube("Table", mobility="MOVABLE")
    child = editor.cube("Cup", mobility="MOVABLE")
    child.root_component.parent = parent.root_component
    session = physics_session(monkeypatch, editor)

    refused = run(session.enable_physics(child.get_path_name()))
    assert codes(refused) == ["ATTACHED"]
    assert "allow_detach" in refused["error"]
    assert child.root_component.parent is parent.root_component

    result = run(session.enable_physics(child.get_path_name(), allow_detach=True))

    assert result["physics_enabled"] is True
    assert result["detached_from"] == parent.root_component.get_path_name()
    assert result["attach_parent"] is None
    assert "Undo reattaches" in result["warning"]
    # The parent was modified in the same transaction, so Undo reattaches the body.
    assert parent.root_component.get_path_name() in editor.transactions[0]["modified"]
    assert editor.unrecorded_writes() == []


def test_a_body_that_does_not_start_simulating_is_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    editor = Editor()
    cube = editor.cube("Odd")
    cube.root_component.refuses_simulation = True
    session = physics_session(monkeypatch, editor)

    result = run(session.enable_physics(cube.get_path_name()))

    assert result["success"] is False
    assert result["error_type"] == "PhysicsNotApplied"
    assert "nothing was changed" in result["error"]
    assert cube.root_component.mobility.name == "STATIC"
    assert editor.transactions[0]["cancelled"] is True


def test_the_simulation_copy_is_edited_without_a_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    editor = Editor()
    editor.pie = True
    cube = editor.cube("Crate", mobility="MOVABLE")
    copy = editor.simulation_copy(cube)
    session = physics_session(monkeypatch, editor)

    result = run(session.enable_physics(copy.get_path_name()))

    assert result["world"] == "simulation"
    assert result["actor_path"].startswith(PIE_LEVEL)
    assert "running simulation's copy" in result["warning"]
    assert editor.transactions == [] and editor.dirty is False
    assert copy.root_component.body["simulate_physics"] is True
    assert cube.root_component.body["simulate_physics"] is False


def test_an_editor_edit_during_pie_says_the_simulation_does_not_see_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    editor = Editor()
    editor.pie = True
    cube = editor.cube("Crate", mobility="MOVABLE")
    session = physics_session(monkeypatch, editor)

    result = run(session.enable_physics(cube.get_path_name()))

    assert result["world"] == "editor"
    assert "does not see" in result["warning"]
    assert editor.transactions and editor.unrecorded_writes() == []


# ---------------------------------------------------------------------------
# set_physics_params
# ---------------------------------------------------------------------------


def test_physics_params_are_written_in_one_transaction_and_read_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    editor = Editor()
    cube = editor.cube("Crate", mobility="MOVABLE")
    session = physics_session(monkeypatch, editor)

    result = run(
        session.set_physics_params(
            cube.get_path_name(),
            mass=5.3,
            linear_damping=0.7,
            angular_damping=0.4,
            enable_gravity=False,
        )
    )

    assert "error" not in result, result
    assert result["params_set"] == 4
    assert result["applied"] == {
        "mass": f32(5.3),
        "linear_damping": f32(0.7),
        "angular_damping": f32(0.4),
        "enable_gravity": False,
    }
    assert result["mass_kg"] == pytest.approx(5.3)
    assert result["warning"] is None
    assert [t["description"] for t in editor.transactions] == [
        "simul: set physics params"
    ]
    assert editor.unrecorded_writes() == [] and editor.dirty is True


def test_physics_params_on_a_static_body_warn_that_it_has_no_mass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U2-04: params_set 4 on a Static cube while get_mass() stayed 0."""
    editor = Editor()
    cube = editor.cube("Pinned")
    session = physics_session(monkeypatch, editor)

    result = run(session.set_physics_params(cube.get_path_name(), mass=5.0))

    assert result["applied"] == {"mass": 5.0} and result["mass_kg"] == 0.0
    assert "Static" in result["warning"] and "Movable" in result["warning"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mass": 1e40},
        {"mass": float("inf")},
        {"mass": float("nan")},
        {"mass": 0.0},
        {"mass": 0.0005},
        {"linear_damping": -1.0},
        {"angular_damping": float("inf")},
        {"linear_damping": 1e7},
    ],
)
def test_out_of_range_physics_params_are_refused_before_the_editor_is_asked(
    monkeypatch: pytest.MonkeyPatch, kwargs: Dict[str, float]
) -> None:
    """U2-12: mass 1e40 stored inf."""
    editor = Editor()
    cube = editor.cube("Crate", mobility="MOVABLE")
    session = physics_session(monkeypatch, editor)

    with pytest.raises(ValueError, match="must be from"):
        run(session.set_physics_params(cube.get_path_name(), **kwargs))

    assert editor.scripts == []


def test_physics_params_with_nothing_to_set_read_the_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    editor = Editor()
    cube = editor.cube("Crate", mobility="MOVABLE")
    session = physics_session(monkeypatch, editor)

    result = run(session.set_physics_params(cube.get_path_name()))
    missing = run(session.set_physics_params(cube.get_path_name() + "_Gone"))

    assert (result["params_set"], result["applied"], result["mass_kg"]) == (
        0,
        {},
        pytest.approx(CUBE_MASS),
    )
    assert editor.transactions == []
    assert missing["success"] is False and "Actor not found" in missing["error"]


def test_a_param_that_reads_back_different_fails_the_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    editor = Editor()
    cube = editor.cube("Crate", mobility="MOVABLE")
    component = cube.root_component
    component.set_linear_damping = lambda value: component._write("linear_damping")  # type: ignore[method-assign]
    session = physics_session(monkeypatch, editor)

    result = run(
        session.set_physics_params(cube.get_path_name(), mass=2.0, linear_damping=0.5)
    )

    assert result["success"] is False
    assert result["error_type"] == "PhysicsParamsNotApplied"
    assert "linear_damping" in result["error"]
    assert result["params_set"] == 1


# ---------------------------------------------------------------------------
# set_collision
# ---------------------------------------------------------------------------


def test_turning_collision_off_clears_use_default_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U2-03: with the flag left set, Unreal put the mesh's BlockAll back in the
    simulation and after reload."""
    editor = Editor()
    cube = editor.cube("Ghost", mobility="MOVABLE")
    session = physics_session(monkeypatch, editor)

    result = run(session.set_collision(cube.get_path_name(), collision_enabled=False))

    assert (
        result["collision_enabled"],
        result["collision_mode"],
        result["collision_preset"],
    ) == (False, "NO_COLLISION", "Custom")
    assert result["use_default_collision"] is False
    assert cube.root_component.use_default_collision is False
    assert [t["description"] for t in editor.transactions] == ["simul: set collision"]
    assert editor.unrecorded_writes() == [] and editor.dirty is True


def test_a_preset_is_set_in_one_transaction(monkeypatch: pytest.MonkeyPatch) -> None:
    editor = Editor()
    cube = editor.cube("Crate", mobility="MOVABLE")
    session = physics_session(monkeypatch, editor)

    result = run(session.set_collision(cube.get_path_name(), "PhysicsActor"))

    assert (result["collision_preset"], result["collision_mode"]) == (
        "PhysicsActor",
        "QUERY_AND_PHYSICS",
    )
    assert editor.unrecorded_writes() == [] and editor.dirty is True


def test_an_unknown_preset_changes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    editor = Editor()
    cube = editor.cube("Crate", mobility="MOVABLE")
    session = physics_session(monkeypatch, editor)

    result = run(session.set_collision(cube.get_path_name(), "Bogus"))

    assert result["success"] is False
    assert result["error_type"] == "ValidationError"
    assert "Unknown collision preset: Bogus" in result["error"]
    assert cube.root_component.get_collision_profile_name() == "BlockAll"
    assert editor.transactions == [] and editor.writes == [] and editor.dirty is False


def test_a_preset_alone_keeps_the_preset_s_collision_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """collision_enabled used to default to True and override every preset:
    OverlapAll became Custom with query-and-physics collision while the tool
    reported OverlapAll (reproduced live on UE 5.7)."""
    editor = Editor()
    cube = editor.cube("Sensor", mobility="MOVABLE")
    session = physics_session(monkeypatch, editor)

    result = run(session.set_collision(cube.get_path_name(), "OverlapAll"))

    assert (
        result["collision_preset"],
        result["collision_mode"],
        result["collision_enabled"],
    ) == ("OverlapAll", "QUERY_ONLY", True)
    assert not any(what == "collision_enabled" for _, what, _ in editor.writes)


def test_collision_reports_what_the_component_reads_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    editor = Editor()
    cube = editor.cube("Crate", mobility="MOVABLE")
    session = physics_session(monkeypatch, editor)
    path = cube.get_path_name()

    disabled = run(session.set_collision(path, collision_enabled=False))
    unchanged = run(session.set_collision(path))
    restored = run(session.set_collision(path, "BlockAll"))

    assert (disabled["collision_preset"], disabled["collision_enabled"]) == (
        "Custom",
        False,
    )
    assert (unchanged["collision_preset"], unchanged["collision_enabled"]) == (
        "Custom",
        False,
    )
    assert (restored["collision_preset"], restored["collision_enabled"]) == (
        "BlockAll",
        True,
    )
    # The read with nothing to set opened no transaction.
    assert len(editor.transactions) == 2


def test_an_unknown_preset_with_a_mode_writes_neither(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    editor = Editor()
    cube = editor.cube("Sensor", mobility="MOVABLE", preset="OverlapAll")
    session = physics_session(monkeypatch, editor)

    result = run(
        session.set_collision(
            cube.get_path_name(), "NotAPreset", collision_enabled=False
        )
    )

    assert (
        result["success"] is False
        and "Unknown collision preset: NotAPreset" in result["error"]
    )
    assert (cube.root_component.preset, cube.root_component.collision) == (
        "OverlapAll",
        "QUERY_ONLY",
    )


def test_turning_off_a_simulated_body_s_collision_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    editor = Editor()
    cube = editor.cube("Crate", mobility="MOVABLE")
    cube.root_component.body["simulate_physics"] = True
    session = physics_session(monkeypatch, editor)

    query_only = run(session.set_collision(cube.get_path_name(), "OverlapAll"))
    none = run(session.set_collision(cube.get_path_name(), "NoCollision"))

    # As UE 5.7.4 simulated them: a query-only body falls through the floor, one
    # without collision gets no physics body and stays where it is.
    assert "falls through everything" in query_only["warning"]
    assert "does not move at all" in none["warning"]


# ---------------------------------------------------------------------------
# setup_physics_bulk
# ---------------------------------------------------------------------------


def _props(editor: Editor, count: int, tag: str = "simul_room_5") -> list:
    actors = []
    for n in range(count):
        actor = editor.cube(f"Room_Box_{n}", mobility="MOVABLE")
        actor.tags.append(tag)
        actor.folder = "Simul/Room_5"
        actors.append(actor)
    return actors


def test_bulk_setup_sets_up_a_tag_in_one_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U2-10: one call per prop ran into the rate limit (126 bodies took 116 s)."""
    editor = Editor()
    props = _props(editor, 6)
    floor = editor.cube("Room_Floor_0")
    session = physics_session(monkeypatch, editor)

    result = run(
        session.setup_physics_bulk(
            tag="simul_room_5", collision_preset="PhysicsActor", mass=2.0
        )
    )

    assert "error" not in result, result
    assert (result["matched"], result["configured_count"], result["failed_count"]) == (
        6,
        6,
        0,
    )
    assert {entry["actor_path"] for entry in result["configured"]} == {
        p.get_path_name() for p in props
    }
    assert all(entry["physics_mode"] == "simulated" for entry in result["configured"])
    assert all(p.root_component.preset == "PhysicsActor" for p in props)
    assert all(p.root_component.get_mass() == 2.0 for p in props)
    assert floor.root_component.body["simulate_physics"] is False
    assert [t["description"] for t in editor.transactions] == ["simul: set up physics"]
    assert editor.unrecorded_writes() == [] and editor.dirty is True


def test_bulk_setup_refuses_everything_when_one_actor_cannot_take_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    editor = Editor()
    props = _props(editor, 3)
    props[1].root_component.collision = "QUERY_ONLY"
    props[1].root_component.preset = "OverlapAll"
    session = physics_session(monkeypatch, editor)

    refused = run(session.setup_physics_bulk(tag="simul_room_5"))
    with_preset = run(
        session.setup_physics_bulk(tag="simul_room_5", collision_preset="PhysicsActor")
    )

    assert refused["error_type"] == INCOMPATIBLE_BODY
    assert (
        "none was changed" in refused["error"] and "skip_failures" in refused["error"]
    )
    assert refused["details"]["failed"][0]["problems"] == ["NO_PHYSICS_COLLISION"]
    assert refused["details"]["failed_count"] == 1
    # The refusal wrote nothing; the preset makes the query-only box physical, so the second call sets up all.
    assert [t["description"] for t in editor.transactions] == ["simul: set up physics"]
    assert with_preset["configured_count"] == 3


def test_bulk_setup_with_skip_failures_sets_up_the_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    editor = Editor()
    props = _props(editor, 3)
    empty = editor.cube("Room_Box_Empty", mobility="MOVABLE", no_mesh=True)
    empty.tags.append("simul_room_5")
    session = physics_session(monkeypatch, editor)

    result = run(
        session.setup_physics_bulk(
            tag="simul_room_5",
            actor_paths=["/Game/Maps/Lab.Lab:PersistentLevel.Gone"],
            skip_failures=True,
        )
    )

    assert (result["configured_count"], result["failed_count"]) == (3, 2)
    failures = {entry["actor_path"]: entry for entry in result["failed"]}
    gone = failures["/Game/Maps/Lab.Lab:PersistentLevel.Gone"]["error"]
    assert gone.startswith("Actor not found: /Game/Maps/Lab.Lab:PersistentLevel.Gone")
    assert failures[empty.get_path_name()]["problems"] == ["NO_GEOMETRY"]
    assert all(p.root_component.body["simulate_physics"] for p in props)


def test_bulk_setup_selects_by_folder_and_tag_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    editor = Editor()
    props = _props(editor, 2)
    props[1].folder = "Simul/Room_5/Shelves"  # a subfolder counts
    stray = editor.cube("Stray", mobility="MOVABLE")
    stray.folder = "Simul/Room_5"  # in the folder, without the tag
    other = editor.cube("Other", mobility="MOVABLE")
    other.tags.append("simul_room_5")  # with the tag, in no folder
    session = physics_session(monkeypatch, editor)

    by_folder = run(session.setup_physics_bulk(folder="Simul/Room_5", kinematic=True))
    both = run(
        session.setup_physics_bulk(
            folder="/Simul/Room_5/", tag="simul_room_5", enable=False
        )
    )

    assert {entry["actor_path"] for entry in by_folder["configured"]} == {
        props[0].get_path_name(),
        props[1].get_path_name(),
        stray.get_path_name(),
    }
    assert by_folder["mode"] == "kinematic"
    assert {entry["actor_path"] for entry in both["configured"]} == {
        p.get_path_name() for p in props
    }


def test_bulk_setup_actor_filter_picks_a_scene_s_props(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A procedural scene tags its floor and walls with its props: selecting by the tag
    alone made the floor Movable and simulated, and everything fell through (live, UE 5.7.4).
    """
    editor = Editor()
    props = _props(editor, 3)
    floor = editor.cube("Room_Floor_0")
    floor.tags.append("simul_room_5")
    session = physics_session(monkeypatch, editor)

    result = run(session.setup_physics_bulk(tag="simul_room_5", actor_filter="_box_"))

    assert {entry["actor_path"] for entry in result["configured"]} == {
        p.get_path_name() for p in props
    }
    assert floor.root_component.mobility.name == "STATIC"


def test_bulk_setup_refuses_simulation_paths_and_too_many_actors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    editor = Editor()
    props = _props(editor, 4)
    editor.pie = True
    copy = editor.simulation_copy(props[0])
    session = physics_session(monkeypatch, editor)
    monkeypatch.setattr("simul.adapters.unreal_runtime.PHYSICS_BULK_MAX_ACTORS", 3)

    simulation = run(session.setup_physics_bulk(actor_paths=[copy.get_path_name()]))
    too_many = run(session.setup_physics_bulk(tag="simul_room_5"))

    assert simulation["error_type"] == INCOMPATIBLE_BODY
    assert "pass the editor path" in simulation["details"]["failed"][0]["error"]
    assert (
        too_many["error_type"] == "ValidationError"
        and "4 actors matched" in too_many["error"]
    )
    assert editor.transactions == []


def test_bulk_setup_checks_its_arguments_first(monkeypatch: pytest.MonkeyPatch) -> None:
    editor = Editor()
    _props(editor, 2)
    session = physics_session(monkeypatch, editor)

    with pytest.raises(ValueError, match="actor_paths, tag, folder or actor_filter"):
        run(session.setup_physics_bulk())
    with pytest.raises(ValueError, match="mass must be from"):
        run(session.setup_physics_bulk(tag="simul_room_5", mass=1e40))
    unknown = run(
        session.setup_physics_bulk(tag="simul_room_5", collision_preset="Bogus")
    )
    nothing = run(session.setup_physics_bulk(tag="no_such_tag"))

    assert unknown["error_type"] == "ValidationError" and "Bogus" in unknown["error"]
    assert (
        nothing["error_type"] == "ValidationError"
        and "No actor matched" in nothing["error"]
    )
    assert editor.transactions == []


def _state(actor: Any) -> Any:
    component = actor.root_component
    return (component.mobility.name, component.preset, component.collision, dict(component.body),
            getattr(component, "use_default_collision", None))


def test_bulk_setup_puts_every_actor_back_when_one_does_not_take_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """Review of #334: a body that did not take the setup after the write left the
    actors before it changed, and itself with the new preset and mass, while the
    reply said it was put back."""
    editor = Editor()
    props = _props(editor, 3)
    props[0].root_component._mobility = props[0].root_component._mobility.__class__("STATIC")
    props[1].root_component.refuses_simulation = True
    before = [_state(prop) for prop in props]
    session = physics_session(monkeypatch, editor)

    result = run(session.setup_physics_bulk(tag="simul_room_5", collision_preset="PhysicsActor", mass=2.0))

    assert result["success"] is False and result["error_type"] == "PhysicsNotApplied"
    assert "nothing was changed" in result["error"]
    assert result["details"]["code"] == "NOT_APPLIED"
    assert result["details"]["expected"] == {
        "physics_mode": "simulated", "collision_preset": "PhysicsActor", "mass": 2.0
    }
    assert result["details"]["actual"][props[1].get_path_name()]["simulate_physics"] is False
    assert [_state(prop) for prop in props] == before  # all three, the first's mobility included
    assert editor.transactions[-1]["cancelled"] is True


def test_bulk_setup_with_skip_failures_puts_the_failed_actor_back_whole(monkeypatch: pytest.MonkeyPatch) -> None:
    editor = Editor()
    props = _props(editor, 3)
    props[1].root_component.refuses_simulation = True
    failed_before = _state(props[1])
    session = physics_session(monkeypatch, editor)

    result = run(
        session.setup_physics_bulk(tag="simul_room_5", collision_preset="PhysicsActor", mass=2.0, skip_failures=True)
    )

    assert (result["configured_count"], result["failed_count"]) == (2, 1)
    assert "put back as it was" in result["failed"][0]["error"]
    assert _state(props[1]) == failed_before  # not left with PhysicsActor and 2 kg
    assert props[0].root_component.preset == props[2].root_component.preset == "PhysicsActor"


def test_bulk_setup_that_sets_up_nothing_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Review of #334: skip_failures returned success with 0 actors configured."""
    editor = Editor()
    empty = editor.cube("Room_Box_Empty", mobility="MOVABLE", no_mesh=True)
    empty.tags.append("simul_room_5")
    session = physics_session(monkeypatch, editor)

    result = run(session.setup_physics_bulk(tag="simul_room_5", skip_failures=True))

    assert result["success"] is False and result["error_type"] == INCOMPATIBLE_BODY
    assert result["details"]["code"] == "NOTHING_SET_UP"
    assert result["failed_count"] == 1


def test_editor_side_refusals_carry_a_code(monkeypatch: pytest.MonkeyPatch) -> None:
    editor = Editor()
    props = _props(editor, 1)
    session = physics_session(monkeypatch, editor)

    preset = run(session.set_collision(props[0].get_path_name(), "Bogus"))
    bulk_preset = run(session.setup_physics_bulk(tag="simul_room_5", collision_preset="Bogus"))
    nothing = run(session.setup_physics_bulk(tag="no_such_tag"))

    assert (preset["details"]["code"], bulk_preset["details"]["code"], nothing["details"]["code"]) == (
        "UNKNOWN_PRESET", "UNKNOWN_PRESET", "NO_MATCH"
    )
