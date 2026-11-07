"""Acting on and reading the running simulation: forces, physics state, simulation control.

A production-readiness audit on UE 5.7.4 found that:

* ``apply_unreal_force`` never moved the simulated body. ``actor_at`` loaded the
  editor world's actor, whose body reads as simulating but never steps: during
  Simulate a 100000 impulse on the editor path returned ``force_applied: true``
  while the play copy stayed at z 49.5, and the same impulse on the
  ``UEDPIE_0_`` path gave 554 cm/s (562 expected). Outside PIE a 500000
  impulse returned success and was lost (U2-01).
* There was no way to read the running simulation: the reads returned the
  editor world, so a body resting at z 49.5 read z 300.
* A "continuous" force acted for one frame (one call at twice the body's
  weight gave a 19 cm/s blip) and its units were undocumented: kg cm/s^2 and
  kg cm/s, so newtons acted 100 times too weak (U2-07).
* ``control_unreal_simulation`` said Play-In-Editor but started Simulate, had
  no Play mode, and took ``action`` as a free string (U2-15).

The scripts run against ``physics_editor``'s fake editor, whose play world
steps when the test advances a frame.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from typing import Any, Dict, Iterator

import pytest
from fastmcp import Client

from simul.adapters import _unreal_scripts as scripts
from simul.adapters import unreal_runtime
from simul.config import Settings
from simul.mcp import server as server_module
from simul.mcp.usage_tracker import ToolUsageTracker
from tests.fakes import AvailableAdapter
from tests.unreal.physics_editor import (
    CUBE_MASS,
    GRAVITY,
    LEVEL,
    PIE_LEVEL,
    Editor,
    physics_session,
)


def run(coroutine: Any) -> Dict[str, Any]:
    return asyncio.run(coroutine)


def simulated_cube(editor: Editor, label: str = "Crate", gravity: bool = False) -> Any:
    """A Movable cube that simulates physics, gravity off so only the force moves it."""
    cube = editor.cube(label, mobility="MOVABLE")
    cube.root_component.body["simulate_physics"] = True
    cube.root_component.body["enable_gravity"] = gravity
    return cube


# ---------------------------------------------------------------------------
# apply_force
# ---------------------------------------------------------------------------


def test_a_force_outside_the_simulation_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U2-01: outside PIE a 500000 impulse returned success and was lost."""
    editor = Editor()
    cube = simulated_cube(editor)
    session = physics_session(monkeypatch, editor)

    result = run(
        session.apply_force(cube.get_path_name(), force_z=500000, is_impulse=True)
    )

    assert result["success"] is False
    assert result["error_type"] == scripts.SIMULATION_NOT_RUNNING
    assert "control_unreal_simulation" in result["details"]["hint"]
    assert cube.root_component.pushes == []


def test_an_impulse_on_the_editor_path_moves_the_simulation_s_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U2-01: the impulse went to the editor body, which never steps."""
    editor = Editor()
    cube = simulated_cube(editor)
    editor.start()
    copy = editor.actors[PIE_LEVEL + cube.name]
    session = physics_session(monkeypatch, editor)

    result = run(
        session.apply_force(cube.get_path_name(), force_z=100000, is_impulse=True)
    )
    editor.frame()

    assert "error" not in result, result
    assert result["world"] == "simulation"
    assert (result["actor_path"], result["simulation_actor_path"]) == (
        cube.get_path_name(),
        copy.get_path_name(),
    )
    assert result["application"] == "impulse"
    assert result["mass_kg"] == pytest.approx(CUBE_MASS)
    assert result["expected_delta_v"] == pytest.approx([0, 0, 100000 / CUBE_MASS])
    assert cube.root_component.pushes == []
    assert copy.root_component.velocity.z == pytest.approx(100000 / CUBE_MASS)
    assert copy.location.z > cube.location.z


def test_a_simulation_path_is_still_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    editor = Editor()
    cube = simulated_cube(editor)
    editor.start()
    copy = editor.actors[PIE_LEVEL + cube.name]
    session = physics_session(monkeypatch, editor)

    result = run(
        session.apply_force(
            copy.get_path_name(),
            force_x=10,
            is_impulse=True,
            location_x=1,
            location_y=2,
            location_z=3,
        )
    )

    assert result["simulation_actor_path"] == copy.get_path_name()
    assert copy.root_component.pushes == [
        ("impulse", [10.0, 0.0, 0.0], [1.0, 2.0, 3.0])
    ]


def test_a_copy_world_partition_streamed_elsewhere_is_found_by_its_actor_guid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    editor = Editor()
    cube = simulated_cube(editor)
    editor.pie = True
    copy = editor.simulation_copy(
        cube, level="/Memory/UEDPIE_0_Cell_12.Lab:PersistentLevel."
    )
    session = physics_session(monkeypatch, editor)

    result = run(session.apply_force(cube.get_path_name(), force_z=5, is_impulse=True))

    assert result["simulation_actor_path"] == copy.get_path_name()


def test_a_body_that_does_not_simulate_in_the_simulation_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    editor = Editor()
    cube = editor.cube("Pinned")  # Static, not simulating
    editor.start()
    session = physics_session(monkeypatch, editor)

    result = run(session.apply_force(cube.get_path_name(), force_z=5, is_impulse=True))

    assert result["error_type"] == scripts.INCOMPATIBLE_BODY
    assert "physics_mode static" in result["error"]
    assert "restart the simulation" in result["error"]


def test_an_actor_without_a_copy_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    editor = Editor()
    editor.pie = True
    cube = simulated_cube(editor)  # placed after the simulation started: no copy
    session = physics_session(monkeypatch, editor)

    result = run(session.apply_force(cube.get_path_name(), force_z=5, is_impulse=True))

    assert result["error_type"] == scripts.NOT_IN_SIMULATION
    assert "list_unreal_simulation_actors" in result["error"]


def test_si_units_are_newtons(monkeypatch: pytest.MonkeyPatch) -> None:
    """U2-07: a force in newtons acted 100 times too weak."""
    editor = Editor()
    cube = simulated_cube(editor)
    editor.start()
    copy = editor.actors[PIE_LEVEL + cube.name]
    session = physics_session(monkeypatch, editor)

    result = run(
        session.apply_force(
            cube.get_path_name(), force_z=50, is_impulse=True, units="si"
        )
    )

    assert (result["force_vector"], result["units"]) == ([0.0, 0.0, 50.0], "si")
    assert result["force_vector_unreal"] == [0.0, 0.0, 5000.0]
    assert copy.root_component.pushes == [("impulse", [0.0, 0.0, 5000.0], None)]


def test_a_force_without_duration_acts_for_one_frame_and_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    editor = Editor()
    cube = simulated_cube(editor)
    editor.start()
    copy = editor.actors[PIE_LEVEL + cube.name]
    session = physics_session(monkeypatch, editor)

    result = run(
        session.apply_force(cube.get_path_name(), force_x=2 * CUBE_MASS * GRAVITY)
    )
    editor.frame(120)

    assert result["application"] == "one_frame" and result["expected_delta_v"] is None
    assert "one frame" in result["warning"] and "duration" in result["warning"]
    # Twice the weight for one 1/120 s frame: 2 g / 120, about 16 cm/s, then nothing.
    assert copy.root_component.velocity.x == pytest.approx(2 * GRAVITY / 120)
    assert editor.tickers == {}


def test_a_force_with_a_duration_acts_every_frame_for_that_long(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """U2-07: one call at twice the weight gave a 19 cm/s blip."""
    editor = Editor()
    cube = simulated_cube(editor)
    editor.start()
    editor.frame(10)
    copy = editor.actors[PIE_LEVEL + cube.name]
    session = physics_session(monkeypatch, editor)
    force = 2 * CUBE_MASS * GRAVITY

    result = run(session.apply_force(cube.get_path_name(), force_x=force, duration=1.0))
    assert result["application"] == "continuous"
    assert result["ends_at_sim_time"] == pytest.approx(result["sim_time"] + 1.0)
    assert result["expected_delta_v"] == pytest.approx([2 * GRAVITY, 0, 0])
    assert len(editor.tickers) == 1

    editor.frame(240)  # two seconds

    # F t / m within a frame, and the ticker removed itself once the force ended.
    assert copy.root_component.velocity.x == pytest.approx(2 * GRAVITY, rel=0.02)
    assert editor.tickers == {}


def test_a_continuous_force_waits_while_paused_and_ends_with_the_simulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    editor = Editor()
    cube = simulated_cube(editor)
    editor.start()
    copy = editor.actors[PIE_LEVEL + cube.name]
    session = physics_session(monkeypatch, editor)

    run(
        session.apply_force(
            cube.get_path_name(), force_x=CUBE_MASS * 100, duration=10.0
        )
    )
    editor.frame(12)
    editor.paused = True
    editor.frame(120)
    # 100 cm/s^2 for the 12 frames of game time that passed, then nothing while paused.
    assert copy.root_component.velocity.x == pytest.approx(100 * 12 / 120)
    assert len(editor.tickers) == 1
    editor.paused = False
    editor.stop()
    editor.frame()
    assert editor.tickers == {}


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"is_impulse": True, "duration": 1.0}, "impulse acts at once"),
        ({"duration": -1.0}, "duration must be from 0"),
        ({"duration": 601.0}, "duration must be from 0"),
        ({"force_z": float("nan")}, "finite"),
        ({"units": "newton"}, "units must be one of"),
        ({"location_x": 1.0}, "all three"),
    ],
)
def test_bad_force_arguments_are_refused_before_the_editor_is_asked(
    monkeypatch: pytest.MonkeyPatch, kwargs: Dict[str, Any], message: str
) -> None:
    editor = Editor()
    cube = simulated_cube(editor)
    session = physics_session(monkeypatch, editor)

    with pytest.raises(ValueError, match=message):
        run(session.apply_force(cube.get_path_name(), **{"force_x": 1.0, **kwargs}))
    assert editor.scripts == []


# ---------------------------------------------------------------------------
# get_physics_state and list_simulation_actors
# ---------------------------------------------------------------------------


def test_physics_state_reads_the_simulation_while_it_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reads returned the editor world, so a resting body read z 300."""
    editor = Editor()
    cube = simulated_cube(editor, gravity=True)
    editor.start()
    editor.frame(30)
    copy = editor.actors[PIE_LEVEL + cube.name]
    session = physics_session(monkeypatch, editor)

    live = run(session.get_physics_state([cube.get_path_name()]))
    placed = run(session.get_physics_state([copy.get_path_name()], world="editor"))

    assert (live["world"], live["pie_active"], live["paused"]) == (
        "simulation",
        True,
        False,
    )
    assert live["sim_time"] == pytest.approx(30 / 120)
    body = live["bodies"][0]
    assert (body["editor_path"], body["simulation_path"]) == (
        cube.get_path_name(),
        copy.get_path_name(),
    )
    assert (
        body["location"][2] == pytest.approx(copy.location.z)
        and body["location"][2] < 300
    )
    assert body["linear_velocity"][2] == pytest.approx(-GRAVITY * 30 / 120)
    assert body["speed"] == pytest.approx(GRAVITY * 30 / 120)
    assert (body["physics_mode"], body["awake"], body["mass_kg"]) == (
        "simulated",
        True,
        pytest.approx(CUBE_MASS),
    )
    assert body["center_of_mass"] == pytest.approx(copy.location.as_list())
    # world "editor" maps a simulation path back to the editor actor, where it was placed.
    assert placed["world"] == "editor"
    assert placed["bodies"][0]["editor_path"] == cube.get_path_name()
    assert placed["bodies"][0]["location"][2] == 300


def test_physics_state_reads_the_editor_world_when_no_simulation_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    editor = Editor()
    cube = simulated_cube(editor)
    session = physics_session(monkeypatch, editor)

    result = run(session.get_physics_state([cube.get_path_name(), LEVEL + "Gone"]))
    refused = run(session.get_physics_state([cube.get_path_name()], world="simulation"))

    assert (result["world"], result["sim_time"], result["count"]) == ("editor", None, 1)
    assert result["not_found"] == [LEVEL + "Gone"]
    assert result["bodies"][0]["simulation_path"] is None
    assert refused["error_type"] == scripts.SIMULATION_NOT_RUNNING


def test_physics_state_by_tag_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    editor = Editor()
    for n in range(5):
        simulated_cube(editor, f"Box{n}").tags.append("props")
    editor.cube("Floor")
    editor.start()
    session = physics_session(monkeypatch, editor)

    result = run(session.get_physics_state(tag_filter="props", max_results=3))

    assert (result["count"], result["total"], result["truncated"]) == (3, 5, True)
    assert all(
        body["simulation_path"].startswith(PIE_LEVEL) for body in result["bodies"]
    )


def test_physics_state_needs_paths_or_a_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    session = physics_session(monkeypatch, Editor())
    with pytest.raises(ValueError, match="actor_paths, tag_filter"):
        run(session.get_physics_state())
    with pytest.raises(ValueError, match="world must be one of"):
        run(session.get_physics_state(["/Game/X.X:PersistentLevel.A"], world="game"))


def test_simulation_actors_map_to_their_editor_actors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    editor = Editor()
    cube = simulated_cube(editor)
    floor = editor.cube("Floor")
    editor.start()
    pawn = editor.spawned_in_simulation("DefaultPawn")
    session = physics_session(monkeypatch, editor)

    listed = run(session.list_simulation_actors())
    moving = run(session.list_simulation_actors(simulating_only=True))

    by_name = {entry["name"]: entry for entry in listed["actors"]}
    assert by_name["Crate"]["editor_path"] == cube.get_path_name()
    assert by_name["Crate"]["simulation_path"] == PIE_LEVEL + cube.name
    assert by_name["Floor"]["editor_path"] == floor.get_path_name()
    assert (
        by_name["DefaultPawn"]["editor_path"],
        by_name["DefaultPawn"]["simulation_path"],
    ) == (None, pawn.get_path_name())
    assert [entry["name"] for entry in moving["actors"]] == ["Crate"]
    assert listed["world"] == editor.pie_world.get_path_name()


def test_listing_simulation_actors_needs_a_simulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = physics_session(monkeypatch, Editor())

    result = run(session.list_simulation_actors())

    assert result["error_type"] == scripts.SIMULATION_NOT_RUNNING


# ---------------------------------------------------------------------------
# control_simulation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode, called", [("simulate", "simulate"), ("play", "play")])
def test_start_begins_simulate_or_play(
    monkeypatch: pytest.MonkeyPatch, mode: str, called: str
) -> None:
    """U2-15: there was no Play mode."""
    editor = Editor()
    session = physics_session(monkeypatch, editor)

    started = run(session.control_simulation("start", mode=mode))
    again = run(session.control_simulation("start", mode="play"))

    assert started == {"action": "start", "state": "playing", "mode": mode}
    assert again == {"action": "start", "state": "playing", "mode": None}
    assert editor.started_modes == [called]
    assert run(session.control_simulation("stop")) == {
        "action": "stop",
        "state": "stopped",
    }


def test_an_unknown_simulation_mode_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    session = physics_session(monkeypatch, Editor())
    with pytest.raises(ValueError, match="Invalid simulation mode"):
        run(session.control_simulation("start", mode="game"))


class _ScriptAdapter(AvailableAdapter):
    """Hands the server a session whose editor Python runs against the fake editor."""

    def __init__(self, session: unreal_runtime.UnrealRuntimeSession) -> None:
        super().__init__(session.settings)
        self.session = session

    @contextmanager
    def create_session(self) -> Iterator[unreal_runtime.UnrealRuntimeSession]:
        yield self.session


def _server(monkeypatch: pytest.MonkeyPatch, tmp_path: Any, editor: Editor) -> Any:
    settings = Settings(
        unreal={
            "tool_surface": "full",
            "attachment_path": str(tmp_path / "attachment.json"),
        },
        security={"rate_limiting_enabled": False},
    )
    instance = server_module.SimulMCPServer(settings, backends={"unreal"})
    instance.usage_tracker = ToolUsageTracker(log_dir=tmp_path / "usage")
    instance.unreal_adapter = _ScriptAdapter(physics_session(monkeypatch, editor))
    return instance


def test_the_force_reply_keeps_its_fields_through_the_tool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    editor = Editor()
    cube = simulated_cube(editor)
    editor.start()
    instance = _server(monkeypatch, tmp_path, editor)

    async def call() -> Dict[str, Any]:
        async with Client(instance.mcp) as client:
            result = await client.call_tool(
                "apply_unreal_force",
                {
                    "actor_path": cube.get_path_name(),
                    "force_z": 3,
                    "is_impulse": True,
                    "units": "si",
                },
                raise_on_error=False,
            )
        return json.loads(
            [
                block.text
                for block in result.content
                if getattr(block, "type", None) == "text"
            ][-1]
        )

    payload = run(call())

    assert payload["success"] is True
    assert payload["simulation_actor_path"] == PIE_LEVEL + cube.name
    assert (payload["world"], payload["units"], payload["force_vector_unreal"]) == (
        "simulation",
        "si",
        [0, 0, 300],
    )
    assert payload["expected_delta_v"] == pytest.approx([0, 0, 300 / CUBE_MASS])


def test_the_simulation_tool_lists_its_actions_and_modes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """action used to be a free string; the schema now lists the actions and modes."""
    instance = _server(monkeypatch, tmp_path, Editor())

    async def listing() -> Dict[str, Any]:
        async with Client(instance.mcp) as client:
            return {tool.name: tool.input_schema for tool in await client.list_tools()}

    schemas = run(listing())

    control = schemas["control_unreal_simulation"]["properties"]
    assert control["action"]["enum"] == list(unreal_runtime.SIMULATION_ACTIONS)
    assert control["mode"]["enum"] == list(unreal_runtime.SIMULATION_MODES)
    force = schemas["apply_unreal_force"]["properties"]
    assert force["units"]["enum"] == list(unreal_runtime.FORCE_UNITS)
    assert "kg cm/s^2" in force["units"]["description"]
    assert {"get_unreal_physics_state", "list_unreal_simulation_actors"} <= set(schemas)
