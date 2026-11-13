"""Full-surface regressions against an explicitly opted-in disposable editor.

Set SIMUL_UNREAL_REVIEW_LIVE=1 and UNREAL__PORT for a scratch project set up
by simul unreal setup. USD round-trip additionally requires USDImporter enabled.
Never point this opt-in tier at an editor with user work.

A test saves only the packages it made, never Save All: a broad save writes
any engine or plugin package dirty in the session into the shared engine
install (tests/unreal/test_no_broad_editor_saves.py enforces it). A test that
opens a scratch level reopens the map it started on and deletes the level.
"""

import asyncio
import base64
import io
import itertools
import math
import os
import uuid
from pathlib import Path

import pytest
from PIL import Image

from simul.adapters.unreal_runtime import UnrealRuntimeSession
from simul.config import Settings

pytestmark = pytest.mark.unreal_live


@pytest.fixture(autouse=True)
def require_disposable_editor():
    if os.environ.get("SIMUL_UNREAL_REVIEW_LIVE") != "1":
        pytest.skip("Set SIMUL_UNREAL_REVIEW_LIVE=1 only for a disposable editor")


def session():
    base = Settings()
    return UnrealRuntimeSession(
        base.model_copy(
            update={
                "unreal": base.unreal.model_copy(update={"timeout": 120}),
            }
        )
    )


async def spawn_cube(client):
    result = await client._execute_json_script("""
import unreal, json
subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
actor = subsystem.spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector(100, 200, 300))
actor.static_mesh_component.set_static_mesh(unreal.load_asset('/Engine/BasicShapes/Cube'))
actor.static_mesh_component.set_mobility(unreal.ComponentMobility.MOVABLE)
actor.set_editor_property('tags', [unreal.Name('simul_review')])
actor.set_actor_hidden_in_game(True)
print(json.dumps({'path': actor.get_path_name()}))
""")
    assert not result.get("error"), result
    return result["path"]


async def destroy_actor(client, path):
    await client._execute_python(
        "import unreal\n"
        f"a = unreal.load_object(None, {path!r})\n"
        "if a: unreal.get_editor_subsystem(unreal.EditorActorSubsystem).destroy_actor(a)"
    )


def test_actor_reads_and_component_operations():
    async def run():
        client = session()
        path = await spawn_cube(client)
        try:
            info = await client.get_actor_info(path)
            assert info["location"] == [100, 200, 300], info
            assert info["tags"] == ["simul_review"]
            assert info["mobility"] == "Movable"
            assert info["is_hidden"] is True
            assert info["components"]
            for cls in ("StaticMeshActor", "/Script/Engine.StaticMeshActor"):
                listing = await client.list_actors(
                    class_filter=cls, tag_filter="simul_review"
                )
                assert path in [a["path"] for a in listing["actors"]], listing
            assert (await client.enable_physics(path))["physics_enabled"] is True
            assert (await client.set_collision(path, "BlockAll"))[
                "collision_enabled"
            ] is True
            result = await client.set_physics_params(
                path, mass=5, linear_damping=1, angular_damping=2, enable_gravity=False
            )
            assert result["params_set"] == 4, result
            # Outside the simulation a force is refused: the editor's bodies never step,
            # so the force_applied reply this used to assert was a force lost.
            result = await client.apply_force(path, force_z=100, is_impulse=True)
            assert result["error_type"] == "SimulationNotRunning", result
            resting = await client.get_physics_state([path])
            assert resting["world"] == "editor" and resting["bodies"][0]["speed"] == 0, resting
            result = await client.assign_material(
                path, "/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial"
            )
            assert not result.get("error"), result
            readback = await client._execute_json_script(f"""
import unreal, json
c = unreal.load_object(None, {path!r}).static_mesh_component
print(json.dumps({{'mass': c.get_mass(), 'gravity': c.is_gravity_enabled(),
                  'material': c.get_material(0).get_path_name()}}))
""")
            assert readback["mass"] == pytest.approx(5)
            assert readback["gravity"] is False
            assert "BasicShapeMaterial" in readback["material"]
        finally:
            await destroy_actor(client, path)
            await client.close()

    asyncio.run(run())


PHYSICS_STATE = """
import unreal, json
def read(path):
    actor = unreal.find_object(None, path)
    if actor is None:
        return None
    c = actor.root_component
    body = c.get_editor_property('body_instance')
    parent = c.get_attach_parent()
    override = body.get_editor_property('override_mass')
    return {'z': actor.get_actor_location().z, 'vz': c.get_physics_linear_velocity().z, 'mobility': c.mobility.name,
            'simulate': body.get_editor_property('simulate_physics'), 'mass': c.get_mass(),
            'override': body.get_editor_property('mass_in_kg_override') if override else None,
            'gravity': body.get_editor_property('enable_gravity'), 'collision': c.get_collision_enabled().name,
            'use_default_collision': c.get_editor_property('use_default_collision'),
            'parent': parent.get_path_name() if parent else None}
world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_game_world()
print(json.dumps({'actors': {path: read(path) for path in args['paths']},
                  'dirty': [p.get_name() for p in unreal.EditorLoadingAndSavingUtils.get_dirty_map_packages()],
                  'time': unreal.GameplayStatics.get_time_seconds(world) if world else None}))
"""

TRANSACTION = """
import unreal, json
world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
unreal.SystemLibrary.execute_console_command(world, 'TRANSACTION ' + args['command'])
print(json.dumps({'done': True}))
"""


async def physics_state(client, *paths):
    result = await client._execute_json_script(f"args = {{'paths': {list(paths)!r}}}\n" + PHYSICS_STATE)
    assert not result.get("error"), result
    return result


async def transaction(client, command):
    result = await client._execute_json_script(f"args = {{'command': {command!r}}}\n" + TRANSACTION)
    assert not result.get("error"), result


async def current_map(client):
    """The editor world's package, to come back to after a test opens a scratch level."""
    result = await client._execute_json_script(
        "import unreal, json\nprint(json.dumps({'map': unreal.get_editor_subsystem("
        "unreal.UnrealEditorSubsystem).get_editor_world().get_outermost().get_name()}))"
    )
    assert not result.get("error"), result
    return result["map"]


async def leave_scratch_level(client, started_on, scratch):
    """Reopen the map the test started on, then delete the scratch level asset.

    A map never saved (``/Temp/Untitled_N``, a disposable editor's startup map)
    cannot be reopened, so a new blank map takes its place. Unsaved changes in
    the scratch level are discarded, and nothing else is saved.
    """
    await client._execute_json_script(f"""
import unreal, json
utils = unreal.EditorLoadingAndSavingUtils
if {started_on!r}.startswith("/Temp/"):
    utils.new_blank_map(False)
else:
    utils.load_map({started_on!r})
deleted = unreal.EditorAssetLibrary.delete_asset({scratch!r})
print(json.dumps({{'deleted': bool(deleted)}}))
""")


def simulation_path(path):
    """The actor's path in the running simulation's world (UEDPIE_0_<map>)."""
    package, _, rest = path.partition(".")
    folder, _, name = package.rpartition("/")
    return f"{folder}/UEDPIE_0_{name}.{rest}"


def test_physics_edits_are_undoable_saved_and_simulate():
    """The physics and collision tools wrote the runtime setters alone: an edit
    was no undo step and left the map clean, so Save All found nothing and the
    edit was gone after reload, and one Undo after spawn plus the three tools
    removed the cube (U2-02). Collision turned off came back as BlockAll in the
    simulation and after reload (U2-03). enable reported physics_enabled for a
    Static cube that never moved, and for a query-only one that fell through
    the floor (U2-04). This checks the engine, not the replies: the map's dirty
    flag, Undo and Redo, a save and reload, and a free fall in Simulate."""

    async def run():
        client = session()
        level = "/Game/SimulReview_Physics_" + uuid.uuid4().hex[:8]
        started_on = await current_map(client)
        try:
            # A session another test (or run) left going would make start a resume and refuse new_level.
            assert (await client.control_simulation("stop"))["state"] == "stopped"
            made = await client._execute_json_script(f"""
import unreal, json
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
assert les.new_level({level!r})
print(json.dumps({{'saved': bool(les.save_current_level())}}))
""")
            assert made == {"saved": True}, made
            name = level.rsplit("/", 1)[1]
            # A Static cube, as spawn_unreal_actor places a mesh, high above an empty level.
            cube = (await client.spawn_actor("/Engine/BasicShapes/Cube.Cube", (0, 0, 20000)))["actor_path"]
            ghost = (await client.spawn_actor("/Engine/BasicShapes/Cube.Cube", (300, 0, 20000)))["actor_path"]
            sensor = (await client.spawn_actor("/Engine/BasicShapes/Cube.Cube", (600, 0, 20000)))["actor_path"]
            saved = await client._execute_json_script(
                "import unreal, json\nprint(json.dumps({'saved': bool(unreal.get_editor_subsystem("
                "unreal.LevelEditorSubsystem).save_current_level())}))"
            )
            assert saved == {"saved": True}, saved
            assert (await physics_state(client, cube))["dirty"] == []

            enabled = await client.enable_physics(cube)
            assert (enabled["physics_mode"], enabled["mobility_changed_from"]) == ("simulated", "Static"), enabled
            state = await physics_state(client, cube)
            assert state["dirty"] == [level], state  # the edit dirtied the map
            assert (state["actors"][cube]["mobility"], state["actors"][cube]["simulate"]) == ("MOVABLE", True)
            assert state["actors"][cube]["mass"] > 0, state

            params = await client.set_physics_params(cube, mass=5, linear_damping=0, enable_gravity=True)
            assert params["applied"]["mass"] == 5 and params["params_set"] == 3, params
            assert (await physics_state(client, cube))["actors"][cube]["override"] == 5

            # One Undo reverts the last call alone: the cube, its mobility and physics stay.
            await transaction(client, "UNDO")
            undone = (await physics_state(client, cube))["actors"][cube]
            assert (undone["override"], undone["simulate"], undone["mobility"]) == (None, True, "MOVABLE"), undone
            await transaction(client, "REDO")
            assert (await physics_state(client, cube))["actors"][cube]["override"] == 5

            # Collision turned off stays off; a query-only body is refused, its flag untouched.
            off = await client.set_collision(ghost, collision_enabled=False)
            assert (off["collision_mode"], off["use_default_collision"]) == ("NO_COLLISION", False), off
            overlap = await client.set_collision(sensor, "OverlapAll")
            assert overlap["collision_mode"] == "QUERY_ONLY", overlap
            refused = await client.enable_physics(sensor)
            assert refused["error_type"] == "IncompatibleBody", refused
            assert [p["code"] for p in refused["details"]["problems"]] == ["NO_PHYSICS_COLLISION"], refused
            assert (await physics_state(client, sensor))["actors"][sensor]["simulate"] is False

            # Saving the map writes the edits, and they survive a reload. Only
            # the test's own map is saved: Save All (save_dirty_packages) also
            # writes any engine or plugin package dirty in the session into the
            # shared engine install.
            reloaded = await client._execute_json_script(f"""
import unreal, json
world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
assert world.get_outermost().get_name() == {level!r}, world.get_outermost().get_name()
ok = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).save_current_level()
unreal.EditorLoadingAndSavingUtils.load_map({level!r})
print(json.dumps({{'saved': bool(ok)}}))
""")
            assert reloaded == {"saved": True}, reloaded
            state = await physics_state(client, cube, ghost)
            assert state["dirty"] == [], state
            kept = state["actors"][cube]
            assert (kept["mobility"], kept["simulate"], kept["override"]) == ("MOVABLE", True, 5), kept
            assert (state["actors"][ghost]["collision"], state["actors"][ghost]["use_default_collision"]) == (
                "NO_COLLISION", False
            ), state

            # In Simulate the cube falls freely (no floor, no damping): it accelerates at g.
            started = await client.control_simulation("start")
            assert started.get("state") == "playing", started
            copies = (simulation_path(cube), simulation_path(ghost))
            assert copies[0].startswith(level.rsplit("/", 1)[0] + "/UEDPIE_0_" + name)
            first = await physics_state(client, *copies)
            await asyncio.sleep(0.5)
            second = await physics_state(client, *copies)
            dt = second["time"] - first["time"]
            before, after = first["actors"][copies[0]], second["actors"][copies[0]]
            assert (after["vz"] - before["vz"]) / dt == pytest.approx(-980, rel=0.05), (first, second)
            assert after["z"] < before["z"] - 100, (first, second)
            assert second["actors"][copies[1]]["collision"] == "NO_COLLISION", second
        finally:
            await client.control_simulation("stop")
            await leave_scratch_level(client, started_on, level)
            await client.close()

    asyncio.run(run())


def test_forces_act_on_the_running_simulation_and_the_reads_follow_it():
    """An impulse on the editor path returned force_applied while the simulated
    body stayed put, the reads returned the editor world (z 300 for a body at
    49.5), and a "continuous" force acted for one frame (U2-01, U2-07). This
    measures the body in the simulation: impulse / mass, force x time / mass."""

    async def run():
        client = session()
        level = "/Game/SimulReview_Forces_" + uuid.uuid4().hex[:8]
        try:
            assert (await client.control_simulation("stop"))["state"] == "stopped"
            made = await client._execute_json_script(f"""
import unreal, json
print(json.dumps({{'new': bool(unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).new_level({level!r}))}}))
""")
            assert made == {"new": True}, made
            # Gravity and damping off: only the forces move the cube, high above an empty level.
            cube = (await client.spawn_actor("/Engine/BasicShapes/Cube.Cube", (0, 0, 20000)))["actor_path"]
            assert (await client.enable_physics(cube))["physics_mode"] == "simulated"
            params = await client.set_physics_params(cube, mass=10, linear_damping=0, angular_damping=0,
                                                     enable_gravity=False)
            assert params["params_set"] == 4, params

            started = await client.control_simulation("start")
            assert (started["state"], started["mode"]) == ("playing", "simulate"), started

            async def velocity():
                state = await client.get_physics_state([cube])
                assert state["world"] == "simulation", state
                return state["bodies"][0]["linear_velocity"]

            await asyncio.sleep(0.3)
            assert await velocity() == pytest.approx([0, 0, 0], abs=1e-3)

            impulse = await client.apply_force(cube, force_x=5000, is_impulse=True)
            assert impulse["simulation_actor_path"] == simulation_path(cube), impulse
            assert impulse["expected_delta_v"] == pytest.approx([500, 0, 0]), impulse
            await asyncio.sleep(0.2)
            assert (await velocity())[0] == pytest.approx(500, rel=0.02)

            si = await client.apply_force(cube, force_y=20, is_impulse=True, units="si")
            assert si["force_vector_unreal"] == [0, 2000, 0], si
            await asyncio.sleep(0.2)
            assert (await velocity())[1] == pytest.approx(200, rel=0.02)

            pushed = await client.apply_force(cube, force_z=2000, duration=0.5)
            assert (pushed["application"], pushed["expected_delta_v"]) == ("continuous", [0, 0, 100]), pushed
            await asyncio.sleep(1.2)
            assert (await velocity())[2] == pytest.approx(100, rel=0.05)

            listed = await client.list_simulation_actors(simulating_only=True)
            entry = [a for a in listed["actors"] if a["simulation_path"] == simulation_path(cube)]
            assert entry and entry[0]["editor_path"] == cube, listed
            placed = await client.get_physics_state([cube], world="editor")
            assert placed["bodies"][0]["location"][2] == pytest.approx(20000), placed

            stopped = await client.control_simulation("stop")
            assert stopped["state"] == "stopped", stopped
            played = await client.control_simulation("start", mode="play")
            assert (played["state"], played["mode"]) == ("playing", "play"), played
            pawns = await client.list_simulation_actors(class_filter="DefaultPawn")
            assert pawns["count"] == 1 and pawns["actors"][0]["editor_path"] is None, pawns
        finally:
            await client.control_simulation("stop")
            await client.close()

    asyncio.run(run())


_MAX_FPS = """
import unreal, json
world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_game_world()
unreal.SystemLibrary.execute_console_command(world, 't.MaxFPS ' + str(args['fps']))
print(json.dumps({'fps': args['fps']}))
"""


def test_a_force_keeps_pace_with_gravity_at_ten_frames_a_second():
    """Review of #320: the force was counted in game time while Unreal's physics,
    gravity included, steps at most MaxPhysicsDeltaTime (1/30 s) a frame, so a 10 kg
    cube held up by exactly m g gained +1216 cm/s at 10 fps. Now it keeps the speed
    it fell at for the whole force, which lasts 2 s of physics time."""

    async def run():
        client = session()
        level = "/Game/SimulReview_Pace_" + uuid.uuid4().hex[:8]
        try:
            assert (await client.control_simulation("stop"))["state"] == "stopped"
            made = await client._execute_json_script(
                "import unreal, json\nprint(json.dumps({'new': bool(unreal.get_editor_subsystem("
                f"unreal.LevelEditorSubsystem).new_level({level!r}))}}))"
            )
            assert made == {"new": True}, made
            cube = (await client.spawn_actor("/Engine/BasicShapes/Cube.Cube", (0, 0, 50000)))["actor_path"]
            assert (await client.enable_physics(cube))["physics_mode"] == "simulated"
            await client.set_physics_params(cube, mass=10, linear_damping=0, angular_damping=0, enable_gravity=True)
            assert (await client.control_simulation("start"))["state"] == "playing"
            await client._execute_json_script("args = {'fps': 10}\n" + _MAX_FPS)
            await asyncio.sleep(0.5)

            async def vz():
                state = await client.get_physics_state([cube])
                return state["sim_time"], state["bodies"][0]["linear_velocity"][2]

            pushed = await client.apply_force(cube, force_z=10 * 980.0, duration=2.0)
            assert pushed["physics_step_limit_s"] == pytest.approx(1 / 30, rel=1e-4), pushed
            held = [await vz()]
            for _ in range(4):
                await asyncio.sleep(1.0)
                held.append(await vz())
            # Within the force (6 s of game time at 10 fps), the speed does not change.
            speeds = [speed for _, speed in held]
            assert max(speeds) - min(speeds) <= 980.0 / 30 + 1e-3, held
        finally:
            await client._execute_json_script("args = {'fps': 0}\n" + _MAX_FPS)
            await client.control_simulation("stop")
            await client.close()

    asyncio.run(run())


def test_transform_light_and_collision_keep_what_is_not_given():
    """A component or parameter left out keeps its value. The transform tool
    used to reset omitted axes to 0 / 1, a lone color channel forced the
    others to 255, and a preset alone was overridden to QueryAndPhysics."""

    async def run():
        client = session()
        path = await spawn_cube(client)
        light = None
        try:
            full = await client.set_actor_transform(
                path, location=[10, 20, 30], rotation=[10, 20, 30], scale=[2, 3, 4]
            )
            assert not full.get("error"), full
            moved = await client.set_actor_transform(path, location=[500, 20, 30])
            assert moved["location"] == pytest.approx([500, 20, 30])
            assert moved["rotation"] == pytest.approx([10, 20, 30])
            assert moved["scale"] == pytest.approx([2, 3, 4])
            turned = await client.set_actor_transform(path, rotation=[0, 270, 0])
            assert not turned.get("error"), turned
            assert turned["rotation"] == pytest.approx([0, -90, 0], abs=1e-6)
            assert turned["location"] == pytest.approx([500, 20, 30])
            # Unreal reads a pitch this close to the pole back snapped to 90.
            near_pole = await client.set_actor_transform(path, rotation=[89.95, 0, 0])
            assert not near_pole.get("error"), near_pole

            overlap = await client.set_collision(path, "OverlapAll")
            assert overlap["collision_preset"] == "OverlapAll", overlap
            mode = await client._execute_json_script(f"""
import unreal, json
c = unreal.load_object(None, {path!r}).static_mesh_component
print(json.dumps({{'mode': c.get_collision_enabled().name}}))
""")
            assert mode["mode"] == "QUERY_ONLY", mode
            unknown = await client.set_collision(path, "NotAPreset")
            assert unknown["success"] is False, unknown
            untouched = await client._execute_json_script(f"""
import unreal, json
c = unreal.load_object(None, {path!r}).static_mesh_component
print(json.dumps({{'profile': str(c.get_collision_profile_name()),
                   'mode': c.get_collision_enabled().name}}))
""")
            assert untouched == {"profile": "OverlapAll", "mode": "QUERY_ONLY"}, untouched

            spawned = await client.spawn_actor("/Script/Engine.PointLight")
            light = spawned["actor_path"]
            colored = await client.set_light_params(light, color=[0.2, 0.4, 0.6])
            assert colored["applied"]["light_color"] == [51, 102, 153, 255], colored
            brighter = await client.set_light_params(light, intensity=1234.0)
            assert not brighter.get("error"), brighter
            kept = await client._execute_json_script(f"""
import unreal, json
c = unreal.load_object(None, {light!r}).light_component
color = c.get_editor_property('light_color')
print(json.dumps({{'color': [color.r, color.g, color.b]}}))
""")
            assert kept["color"] == [51, 102, 153], kept
        finally:
            await destroy_actor(client, path)
            if light:
                await destroy_actor(client, light)
            await client.close()

    asyncio.run(run())


def test_material_params_are_written_and_read_back_in_and_out_of_play_in_editor():
    """set_unreal_material_params reported params_set 1 while the instance
    kept no override, Roughness reading its 0.6407 default, in and out of PIE:
    the Remote Control write it sent changed nothing. A request naming a
    parameter the material lacks is refused with nothing written, and an edit
    made during PIE is the asset the game renders with and stays after PIE."""

    async def run():
        client = session()
        folder = "/Game/SimulReview_" + uuid.uuid4().hex[:8]
        readback = """
import unreal, json
m = unreal.load_asset(args['path'])
print(json.dumps({'scalar': {str(v.parameter_info.name): v.parameter_value
                             for v in m.get_editor_property('scalar_parameter_values')},
                  'vector': {str(v.parameter_info.name): [v.parameter_value.r, v.parameter_value.g,
                                                          v.parameter_value.b, v.parameter_value.a]
                             for v in m.get_editor_property('vector_parameter_values')},
                  'texture': {str(v.parameter_info.name): v.parameter_value.get_path_name()
                              for v in m.get_editor_property('texture_parameter_values')}}))
"""
        game_material = """
import unreal, json
world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_game_world()
copy = [a for a in unreal.GameplayStatics.get_all_actors_of_class(world, unreal.StaticMeshActor)
        if a.get_name() == args['name']]
material = copy[0].static_mesh_component.get_material(0) if copy else None
print(json.dumps({'material': material.get_path_name() if material else None,
                  'roughness': unreal.MaterialEditingLibrary.get_material_instance_scalar_parameter_value(
                      material, 'Roughness') if material else None}))
"""
        cube = None
        try:
            created = await client.create_material_instance(
                "/Engine/BasicShapes/BasicShapeMaterial", "MI_Params", folder
            )
            assert not created.get("error"), created
            widget = await client.create_material_instance(
                "/Engine/EngineMaterials/Widget3DPassThrough", "MI_Widget", folder
            )
            assert not widget.get("error"), widget
            path, widget_path = created["instance_path"], widget["instance_path"]
            # A cube using the instance; always loaded, so World Partition puts
            # its copy in the PIE world without a streaming source.
            cube = await client._execute_json_script(f"""
import unreal, json
actor = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).spawn_actor_from_class(
    unreal.StaticMeshActor, unreal.Vector(0, 0, 100))
actor.static_mesh_component.set_static_mesh(unreal.load_asset('/Engine/BasicShapes/Cube'))
actor.static_mesh_component.set_material(0, unreal.load_asset({path!r}))
actor.set_editor_property('is_spatially_loaded', False)
print(json.dumps({{'path': actor.get_path_name(), 'name': actor.get_name()}}))
""")
            assert not cube.get("error"), cube

            async def read(asset):
                return await client._execute_json_script(f"args = {{'path': {asset!r}}}\n" + readback)

            for roughness, pie in ((0.25, False), (0.5, True)):
                if pie:
                    started = await client.control_simulation("start")
                    assert started.get("state") == "playing", started
                before = await read(path)
                refused = await client.set_material_params(
                    path,
                    scalar_params={"Roughness": 0.9, "Bogus": 1.0},
                    vector_params={"Color": [0.9, 0.1, 0.1]},
                )
                assert refused["error_type"] == "ValidationError", refused
                assert (refused["applied"], refused["params_set"]) == ({}, 0), refused
                assert list(refused["not_applied"]["scalar"]) == ["Bogus"], refused
                # Nothing written: no Bogus override, and not the valid Roughness and Color either.
                assert await read(path) == before
                result = await client.set_material_params(
                    path,
                    scalar_params={"Roughness": roughness},
                    vector_params={"Color": [0.1, 0.2, 0.3]},
                )
                assert "error" not in result, result
                assert result["params_set"] == 2, result
                assert result["applied"]["scalar"] == {"Roughness": roughness}, result
                assert result["applied"]["vector"]["Color"] == pytest.approx([0.1, 0.2, 0.3, 1.0]), result
                assert result["not_applied"] == {}, result
                assert (await read(path))["scalar"] == {"Roughness": roughness}
            # The game renders with the asset itself, so it has the PIE-time value.
            game = await client._execute_json_script(f"args = {{'name': {cube['name']!r}}}\n" + game_material)
            assert game == {"material": path, "roughness": 0.5}, game
            stopped = await client.control_simulation("stop")
            assert stopped.get("state") == "stopped", stopped
            assert (await read(path))["scalar"] == {"Roughness": 0.5}
            textured = await client.set_material_params(
                widget_path, texture_params={"SlateUI": "/Engine/EngineResources/DefaultTexture"}
            )
            assert textured["applied"] == {
                "texture": {"SlateUI": "/Engine/EngineResources/DefaultTexture.DefaultTexture"}
            }, textured
            assert (await read(widget_path))["texture"] == textured["applied"]["texture"]
            missing = await client.set_material_params(
                widget_path,
                texture_params={"SlateUI": "/Game/NoSuchTexture"},
                scalar_params={"OpacityFromTexture": 0.5},
            )
            assert missing["error_type"] == "ValidationError", missing
            assert "OpacityFromTexture" not in (await read(widget_path))["scalar"]
        finally:
            await client.control_simulation("stop")
            if cube and cube.get("path"):
                await destroy_actor(client, cube["path"])
            await client._execute_python(f"import unreal\nunreal.EditorAssetLibrary.delete_directory({folder!r})")
            await client.close()

    asyncio.run(run())


def test_spawn_from_asset_and_class_paths_then_find_by_semantic_label():
    """An asset path used to go to SpawnActorFromClass and come back as a
    successful spawn with an empty actor path; the semantic label lookup
    called an EditorLevelLibrary object UE 5.7 does not have (HTTP 400)."""

    async def run():
        client = session()
        tag = f"simul_review_{uuid.uuid4().hex}"
        spawned = []
        try:
            cube = await client.spawn_actor(
                "/Engine/BasicShapes/Cube.Cube",
                location=[100, 200, 300],
                rotation=[0, 45, 0],
                label="SimulReviewCube",
            )
            assert not cube.get("error"), cube
            spawned.append(cube["actor_path"])
            assert cube["actor_class"] == "StaticMeshActor", cube
            assert cube["label"] == "SimulReviewCube", cube
            assert cube["location"] == pytest.approx([100, 200, 300])
            assert cube["rotation"] == pytest.approx([0, 45, 0], abs=1e-6)
            light = await client.spawn_actor("/Script/Engine.PointLight")
            assert not light.get("error"), light
            spawned.append(light["actor_path"])
            assert light["actor_class"] == "PointLight", light
            readback = await client._execute_json_script(f"""
import unreal, json
c = unreal.load_object(None, {cube['actor_path']!r}).static_mesh_component
print(json.dumps({{'mesh': c.static_mesh.get_path_name()}}))
""")
            assert readback == {"mesh": "/Engine/BasicShapes/Cube.Cube"}, readback
            missing = await client.spawn_actor("/Game/SimulReview/Missing.Missing")
            assert missing["success"] is False, missing
            assert missing["error_type"] == "NotFoundError", missing

            for path, label in (
                (cube["actor_path"], tag),
                (light["actor_path"], tag.upper() + "_light"),
            ):
                await client._execute_python(
                    "import unreal\n"
                    f"a = unreal.load_object(None, {path!r})\n"
                    f"a.set_editor_property('tags', [unreal.Name({label!r})])"
                )
            exact = await client.get_actor_by_semantic_label(tag)
            assert [a["path"] for a in exact["actors"]] == [cube["actor_path"]], exact
            assert exact["actors"][0]["tags"] == [tag], exact
            contains = await client.get_actor_by_semantic_label(
                tag, match_mode="contains"
            )
            assert sorted(a["path"] for a in contains["actors"]) == sorted(
                spawned
            ), contains
            regex = await client.get_actor_by_semantic_label(
                f"^{tag}_LIGHT$", match_mode="regex"
            )
            assert [a["path"] for a in regex["actors"]] == [light["actor_path"]], regex
        finally:
            for path in spawned:
                await destroy_actor(client, path)
            await client.close()

    asyncio.run(run())


def test_material_info_reports_parameters_and_overrides_in_and_out_of_play_in_editor():
    """get_unreal_material_info listed the instance's object properties
    (ParameterStateId, PhysMaterialMask, ...) without values, and a null parent."""

    async def run():
        client = session()
        folder = "/Game/SimulReview_" + uuid.uuid4().hex[:8]
        parent = "/Engine/BasicShapes/BasicShapeMaterial"
        try:
            created = await client.create_material_instance(parent, "MI_Info", folder)
            assert not created.get("error"), created
            path = created["instance_path"]
            overridden = await client._execute_json_script(f"""
import unreal, json
MEL = unreal.MaterialEditingLibrary
m = unreal.load_asset({path!r})
MEL.set_material_instance_scalar_parameter_value(m, 'Roughness', 0.25)
MEL.update_material_instance(m)
print(json.dumps({{'roughness': MEL.get_material_instance_scalar_parameter_value(m, 'Roughness')}}))
""")
            assert overridden == {"roughness": 0.25}, overridden
            for pie in (False, True):
                if pie:
                    started = await client.control_simulation("start")
                    assert started.get("state") == "playing", started
                info = await client.get_material_info(path)
                assert info["parent_path"] == parent + ".BasicShapeMaterial", info
                roughness, color = info["parameters"]
                assert roughness == {"name": "Roughness", "param_type": "scalar", "association": "global",
                                     "index": None, "value": 0.25, "overridden": True}
                assert (color["name"], color["param_type"], color["association"], color["overridden"]) == (
                    "Color", "vector", "global", False)
                assert color["value"] == pytest.approx([0.9, 0.9, 0.9, 1.0]), color
                assert info["material_path"] == path, info
                assert info["class_name"] == "MaterialInstanceConstant", info
                assert info["base_material_path"] == parent + ".BasicShapeMaterial", info
                assert (info["material_domain"], info["blend_mode"]) == ("MD_SURFACE", "BLEND_OPAQUE"), info
            base = await client.get_material_info(parent)
            assert base["parent_path"] is None, base
            assert base["parameters"][0]["value"] == pytest.approx(0.6407), base
            assert base["parameters"][0]["overridden"] is None, base
            missing = await client.get_material_info(folder + "/MI_Missing")
            assert missing["error_type"] == "NotFoundError", missing
            mesh = await client.get_material_info("/Engine/BasicShapes/Cube")
            assert mesh["error_type"] == "InvalidAssetType", mesh
        finally:
            await client.control_simulation("stop")
            await client._execute_python(f"import unreal\nunreal.EditorAssetLibrary.delete_directory({folder!r})")
            await client.close()

    asyncio.run(run())


# Builds args["folder"]/M_Layered: GlobalRough and Shared of its own, and a
# layer stack of ML_Base (LayerRough 0.25, Shared 0.7), ML_Top (LayerRough
# 0.75, TopOnly 0.5, and NestedParam 0.33 in a function it calls) and MLB_Mix
# (BlendAlpha 0.5); and MI_Layered overriding GlobalRough, LayerRough on layer
# 0, TopOnly on layer 1 and BlendAlpha on blend 0. The layer stack is built as
# a standalone struct and assigned once: the struct get_editor_property returns
# is bound to the expression, so setting its fields one at a time validates
# each half-built stack, and Runtime.Layers.Num() == EditorOnly.LayerStates.Num()
# asserted and took the editor down (UE 5.7.4). MaterialEditingLibrary writes a
# layer parameter on layer 0 only, so the layer 1 and blend overrides go into
# the instance's parameter array, also written once.
LAYERED_MATERIAL = """
import json, unreal
MEL = unreal.MaterialEditingLibrary
A = unreal.MaterialParameterAssociation
tools = unreal.AssetToolsHelpers.get_asset_tools()
folder = args["folder"]

def asset(name, cls, factory):
    return tools.create_asset(name, folder, cls, factory)

def node(function, cls, **properties):
    expression = MEL.create_material_expression_in_function(function, cls, 0, 0)
    for key, value in properties.items():
        expression.set_editor_property(key, value)
    return expression

def scalar(function, name, value):
    return node(function, unreal.MaterialExpressionScalarParameter, parameter_name=name, default_value=value)

def link(source, target, pin, output=""):
    assert MEL.connect_material_expressions(source, output, target, pin), (pin, output)

helper = asset("MF_Helper", unreal.MaterialFunction, unreal.MaterialFunctionFactoryNew())
result = node(helper, unreal.MaterialExpressionFunctionOutput, output_name="Result")
link(scalar(helper, "NestedParam", 0.33), result, "")
MEL.update_material_function(helper)

layers = []
for name, rough, extra in (("ML_Base", 0.25, ("Shared", 0.7)), ("ML_Top", 0.75, ("TopOnly", 0.5))):
    layer = asset(name, unreal.MaterialFunctionMaterialLayer, unreal.MaterialFunctionMaterialLayerFactory())
    attributes = node(layer, unreal.MaterialExpressionMakeMaterialAttributes)
    link(scalar(layer, "LayerRough", rough), attributes, "Roughness")
    link(scalar(layer, *extra), attributes, "Specular")
    if name == "ML_Top":
        link(node(layer, unreal.MaterialExpressionMaterialFunctionCall, material_function=helper), attributes,
             "Metallic", "Result")
    link(attributes, node(layer, unreal.MaterialExpressionFunctionOutput), "")
    MEL.update_material_function(layer)
    layers.append(layer)

blend = asset("MLB_Mix", unreal.MaterialFunctionMaterialLayerBlend, unreal.MaterialFunctionMaterialLayerBlendFactory())
mix = node(blend, unreal.MaterialExpressionBlendMaterialAttributes)
for pin, (input_name, priority) in zip(("A", "B"), (("Bottom Layer", 0), ("Top Layer", 1))):
    link(node(blend, unreal.MaterialExpressionFunctionInput, input_name=input_name, sort_priority=priority,
              input_type=unreal.FunctionInputType.FUNCTION_INPUT_MATERIAL_ATTRIBUTES), mix, pin)
link(scalar(blend, "BlendAlpha", 0.5), mix, "Alpha")
link(mix, node(blend, unreal.MaterialExpressionFunctionOutput), "")
MEL.update_material_function(blend)

material = asset("M_Layered", unreal.Material, unreal.MaterialFactoryNew())
material.set_editor_property("use_material_attributes", True)
for name, value in (("GlobalRough", 0.4), ("Shared", 0.1)):
    own = MEL.create_material_expression(material, unreal.MaterialExpressionScalarParameter, -600, 0)
    own.set_editor_property("parameter_name", name)
    own.set_editor_property("default_value", value)
stack = MEL.create_material_expression(material, unreal.MaterialExpressionMaterialAttributeLayers, -300, 0)
background = stack.get_editor_property("default_layers").get_editor_property("editor_only")
background_guid = unreal.GuidLibrary.parse_string_to_guid(
    unreal.GuidLibrary.conv_guid_to_string(background.get_editor_property("layer_guids")[0]))[0]
editor_only = unreal.MaterialLayersFunctionsEditorOnlyData()
for key, value in (("layer_states", [True, True]),
                   ("layer_names", [unreal.Text("Background"), unreal.Text("Layer 1")]),
                   ("restrict_to_layer_relatives", [False, False]), ("restrict_to_blend_relatives", [False]),
                   ("layer_guids", [background_guid, unreal.GuidLibrary.new_guid()]),
                   ("layer_link_states", [unreal.MaterialLayerLinkState.NOT_FROM_PARENT] * 2)):
    editor_only.set_editor_property(key, value)
functions = unreal.MaterialLayersFunctions()
functions.set_editor_property("layers", layers)
functions.set_editor_property("blends", [blend])
functions.set_editor_property("editor_only", editor_only)
stack.set_editor_property("default_layers", functions)
assert MEL.connect_material_property(stack, "", unreal.MaterialProperty.MP_MATERIAL_ATTRIBUTES)
MEL.recompile_material(material)

instance = asset("MI_Layered", unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew())
MEL.set_material_instance_parent(instance, material)
MEL.set_material_instance_scalar_parameter_value(instance, "GlobalRough", 0.9)
MEL.set_material_instance_scalar_parameter_value(instance, "LayerRough", 0.05, A.LAYER_PARAMETER)
overrides = []
for name, association, index, value in [(str(e.parameter_info.name), e.parameter_info.association,
                                         e.parameter_info.index, e.parameter_value)
                                        for e in instance.get_editor_property("scalar_parameter_values")] + [
        ("TopOnly", A.LAYER_PARAMETER, 1, 0.2), ("BlendAlpha", A.BLEND_PARAMETER, 0, 0.8)]:
    entry = unreal.ScalarParameterValue()
    entry.set_editor_property("parameter_info", unreal.MaterialParameterInfo(name, association, index))
    entry.set_editor_property("parameter_value", value)
    overrides.append(entry)
instance.set_editor_property("scalar_parameter_values", overrides)
MEL.update_material_instance(instance)
print(json.dumps({"material": material.get_path_name(), "instance": instance.get_path_name()}))
"""

# The scalar overrides an instance holds: [name, association, index, value].
INSTANCE_OVERRIDES = """
import json, unreal
m = unreal.load_asset(args['path'])
print(json.dumps({'overrides': sorted([str(v.parameter_info.name), v.parameter_info.association.name,
                                       v.parameter_info.index, round(v.parameter_value, 4)]
                                      for v in m.get_editor_property('scalar_parameter_values'))}))
"""


def test_material_layer_parameters_are_reported_per_layer_and_refused_by_set_in_and_out_of_play_in_editor():
    """A parameter inside a Material Layer was read and written by name, which
    addresses the material's global parameter: TopOnly (layer 1 only) and
    BlendAlpha (blend 0 only) read 0 with overridden false despite the
    instance's 0.2 and 0.8, and setting TopOnly reported it applied while
    adding a global override that left layer 1 rendering 0.5."""

    async def run():
        client = session()
        folder = "/Game/SimulReview_" + uuid.uuid4().hex[:8]
        try:
            built = await client._execute_json_script(f"args = {{'folder': {folder!r}}}\n" + LAYERED_MATERIAL)
            assert not built.get("error"), built
            material, instance = built["material"], built["instance"]

            async def overrides():
                return await client._execute_json_script(f"args = {{'path': {instance!r}}}\n" + INSTANCE_OVERRIDES)

            before = await overrides()
            assert before["overrides"] == [
                ["BlendAlpha", "BLEND_PARAMETER", 0, 0.8],
                ["GlobalRough", "GLOBAL_PARAMETER", -1, 0.9],
                ["LayerRough", "LAYER_PARAMETER", 0, 0.05],
                ["TopOnly", "LAYER_PARAMETER", 1, 0.2],
            ], before
            defaults = await client.get_material_info(material)
            assert [(p["name"], p["association"], p["index"], round(p["value"], 4), p["overridden"])
                    for p in defaults["parameters"]] == [
                ("GlobalRough", "global", None, 0.4, None),
                ("Shared", "global", None, 0.1, None),
                ("LayerRough", "layer", 0, 0.25, None),
                ("Shared", "layer", 0, 0.7, None),
                ("LayerRough", "layer", 1, 0.75, None),
                ("NestedParam", "layer", 1, 0.33, None),
                ("TopOnly", "layer", 1, 0.5, None),
                ("BlendAlpha", "blend", 0, 0.5, None),
            ], defaults
            for pie in (False, True):
                if pie:
                    started = await client.control_simulation("start")
                    assert started.get("state") == "playing", started
                info = await client.get_material_info(instance)
                assert [(p["name"], p["association"], p["index"], round(p["value"], 4), p["overridden"])
                        for p in info["parameters"]] == [
                    ("GlobalRough", "global", None, 0.9, True),
                    ("Shared", "global", None, 0.1, False),
                    ("LayerRough", "layer", 0, 0.05, True),
                    ("Shared", "layer", 0, 0.7, False),
                    ("LayerRough", "layer", 1, 0.75, False),
                    ("NestedParam", "layer", 1, 0.33, False),
                    ("TopOnly", "layer", 1, 0.2, True),
                    ("BlendAlpha", "blend", 0, 0.8, True),
                ], info
                refused = await client.set_material_params(
                    instance, scalar_params={"TopOnly": 0.3, "BlendAlpha": 0.3, "GlobalRough": 0.5}
                )
                assert refused["error_type"] == "ValidationError", refused
                assert (refused["params_set"], refused["applied"]) == (0, {}), refused
                assert sorted(refused["not_applied"]["scalar"]) == ["BlendAlpha", "TopOnly"], refused
                assert "(layer 1)" in refused["not_applied"]["scalar"]["TopOnly"], refused
                assert "(blend 0)" in refused["not_applied"]["scalar"]["BlendAlpha"], refused
                # Nothing written: no global TopOnly or BlendAlpha, and GlobalRough kept.
                assert await overrides() == before
            stopped = await client.control_simulation("stop")
            assert stopped.get("state") == "stopped", stopped
            # A name the material shares with layer 0 sets the material's own.
            shared = await client.set_material_params(instance, scalar_params={"Shared": 0.55})
            assert shared["applied"] == {"scalar": {"Shared": pytest.approx(0.55)}}, shared
            expected = sorted(before["overrides"] + [["Shared", "GLOBAL_PARAMETER", -1, 0.55]])
            assert (await overrides())["overrides"] == expected
            info = await client.get_material_info(instance)
            shared_entries = [(p["association"], round(p["value"], 4), p["overridden"])
                              for p in info["parameters"] if p["name"] == "Shared"]
            assert shared_entries == [("global", 0.55, True), ("layer", 0.7, False)], info
        finally:
            await client.control_simulation("stop")
            await client._execute_python(f"import unreal\nunreal.EditorAssetLibrary.delete_directory({folder!r})")
            await client.close()

    asyncio.run(run())


def test_scene_graph_listing_pages_and_components_read_the_level():
    """The scene graph read the Asset Registry and found no level actors, and
    the listings had no paging; both now read the editor world's attachment
    hierarchy, one page at a time."""

    async def run():
        client = session()
        tag = "simul_graph_" + uuid.uuid4().hex[:8]
        made = await client._execute_json_script(f"""
import unreal, json
sub = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
cube = unreal.load_asset('/Engine/BasicShapes/Cube')
paths = []
for i in range(3):
    actor = sub.spawn_actor_from_object(cube, unreal.Vector(5000 + i * 200, 5000, 100))
    actor.set_actor_label({tag!r} + '_' + str(i))
    actor.set_editor_property('tags', [unreal.Name({tag!r})])
    if paths:
        actor.attach_to_actor(unreal.load_object(None, paths[-1]), unreal.Name('None'),
                              unreal.AttachmentRule.KEEP_WORLD, unreal.AttachmentRule.KEEP_WORLD,
                              unreal.AttachmentRule.KEEP_WORLD, False)
    paths.append(actor.get_path_name())
light = sub.spawn_actor_from_class(unreal.PointLight, unreal.Vector(5000, 5400, 300))
light.set_actor_label({tag!r} + '_light')
light.set_editor_property('tags', [unreal.Name({tag!r})])
paths.append(light.get_path_name())
print(json.dumps({{'paths': paths}}))
""")
        assert not made.get("error"), made
        top, middle, leaf, light = made["paths"]
        try:
            graph = await client.query_scene_graph(root_path=top, include_components=True)
            assert not graph.get("error"), graph
            assert graph["root"]["path"] == top
            assert [child["path"] for child in graph["root"]["children"]] == [middle]
            assert graph["root"]["children"][0]["children"][0]["path"] == leaf
            assert (graph["total_actors"], graph["total_depth"]) == (2, 2)
            assert graph["root"]["children"][0]["components"][0]["is_root"] is True
            shallow = await client.query_scene_graph(root_path=top, max_depth=1)
            assert shallow["root"]["children"][0]["children_truncated"] is True
            level = await client.query_scene_graph(class_filter="StaticMeshActor", max_results=5000)
            assert level["total_actors"] >= 3 and level["root"]["class_name"] == "World"
            first = await client.get_actor_by_semantic_label(tag, max_results=2)
            assert (first["count"], first["total_matches"], first["next_offset"]) == (2, 4, 2), first
            rest = await client.get_actor_by_semantic_label(tag, max_results=2, offset=2)
            assert [a["path"] for a in first["actors"] + rest["actors"]] == made["paths"]
            listing = await client.list_actors(label_filter=tag, class_filter="Light", include_subclasses=True)
            assert [a["path"] for a in listing["actors"]] == [light]
            listing = await client.list_actors(label_filter=tag + "_1")
            entry = listing["actors"][0]
            assert (entry["parent"], entry["children"], entry["child_count"]) == (top, [leaf], 1)
            info = await client.get_actor_info(leaf)
            assert (info["parent"], info["child_count"]) == (middle, 0)
            components = await client.list_actor_components(light)
            assert components["total"] >= 1 and components["components"][0]["mobility"] in (
                "Static", "Stationary", "Movable",
            )
            primitives = await client.list_actor_components(top, class_filter="PrimitiveComponent")
            assert primitives["components"][0]["bounds"]["extent"] == pytest.approx([50, 50, 50], abs=1e-3)
        finally:
            for path in reversed(made["paths"]):
                await destroy_actor(client, path)
            await client.close()

    asyncio.run(run())


def test_simulation_lifecycle():
    async def run():
        client = session()
        try:
            assert not (await client.get_simulation_status())["is_playing"]
            for action, state in [
                ("start", "playing"),
                ("start", "playing"),
                ("pause", "paused"),
                ("pause", "paused"),
                ("start", "playing"),
                ("pause", "paused"),
                ("resume", "playing"),
                ("resume", "playing"),
                ("stop", "stopped"),
                ("stop", "stopped"),
            ]:
                result = await client.control_simulation(action)
                assert result.get("state") == state, result
            assert (await client.control_simulation("step"))[
                "error_type"
            ] == "UnsupportedOperation"
        finally:
            await client.control_simulation("stop")
            await client.close()

    asyncio.run(run())


def test_thumbnail_and_capture_file(tmp_path):
    async def run():
        client = session()
        try:
            result = await client.get_actor_thumbnail("/Engine/BasicShapes/Cube.Cube")
            with Image.open(
                io.BytesIO(base64.b64decode(result["image_base64"]))
            ) as image:
                assert image.format.lower() == result["format"]
            for fmt in ("png", "jpeg"):
                target = tmp_path / f"capture.{fmt}"
                result = await client.capture_to_file(target, 256, 256, fmt)
                assert not result.get("error"), result
                with Image.open(target) as image:
                    assert image.format.lower() == fmt
                    assert image.size == (256, 256)
        finally:
            await client.close()

    asyncio.run(run())


def test_usd_roundtrip_and_selection_restoration():
    async def run():
        client = session()
        path = None
        imported = []
        suffix = uuid.uuid4().hex
        destination = f"/Game/SimulReview_{suffix}"
        target = Path("/tmp/simul-work") / f"review-{suffix}" / "export.usda"
        try:
            info = await client.get_interchange_info()
            if not info.get("usd_import_available") or not info.get(
                "usd_export_available"
            ):
                pytest.skip("USDImporter must be enabled in the disposable project")
            path = await spawn_cube(client)
            export_state = """
import unreal, json
subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
inner = unreal.get_default_object(unreal.LevelExporterUSDOptions).inner
print(json.dumps({'selection': [a.get_path_name() for a in subsystem.get_selected_level_actors()],
                  'selection_only': inner.selection_only, 'asset_folder': inner.asset_folder.path,
                  'export_sublayers': inner.export_sublayers}))
"""
            await client._execute_python(
                "import unreal\n"
                "unreal.get_editor_subsystem(unreal.EditorActorSubsystem).set_selected_level_actors([])"
            )
            previous_state = await client._execute_json_script(export_state)
            assert not previous_state.get("error"), previous_state
            result = await client.export_usd([path], str(target))
            assert not result.get("error"), result
            assert result["file_size_bytes"] > 0
            assert await client._execute_json_script(export_state) == previous_state
            # The integration runner and editor share this scratch filesystem.
            from pxr import Usd

            stage = Usd.Stage.Open(str(target))
            assert stage is not None
            assert any(p.GetTypeName() == "Mesh" for p in stage.Traverse())
            result = await client.import_usd(str(target), destination)
            assert not result.get("error"), result
            imported = result["actor_paths"]
            assert imported or result["imported_assets"]
        finally:
            for actor_path in ([path] if path else []) + imported:
                await destroy_actor(client, actor_path)
            await client._execute_python(
                "import unreal\n"
                f"if unreal.EditorAssetLibrary.does_directory_exist({destination!r}):\n"
                f"    unreal.EditorAssetLibrary.delete_directory({destination!r})"
            )
            await client.close()

    asyncio.run(run())


def _remove_tagged(prefix):
    """Editor Python removing every level actor with a tag starting with ``prefix``."""
    return (
        "import unreal\n"
        "subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)\n"
        "for actor in subsystem.get_all_level_actors():\n"
        f"    if any(str(t).startswith({prefix!r}) for t in actor.tags):\n"
        "        subsystem.destroy_actor(actor)"
    )


def test_generate_and_analyze_a_procedural_scene():
    """Both tools called /Script/UnrealEd.Default__EditorLevelLibrary, which
    UE 5.7 does not have (HTTP 400), and the generator a function no Unreal
    version has. The scene is spawned high above the level and removed."""
    from simul.adapters._unreal_procedural import plan_scene

    async def run():
        client = session()
        seed = uuid.uuid4().int % 100000
        arguments = ("warehouse", {"seed": seed, "rows": 2, "props": 4}, [0, 0, 20000], [1500, 1200, 20400])
        plan = plan_scene(*arguments)
        prefix = f"simul_warehouse_{seed}_"
        try:
            scene = await client.generate_procedural_scene(*arguments)
            assert not scene.get("error"), scene
            tag = scene["tag"]
            generation = tag[len(prefix):]
            assert tag.startswith(prefix) and len(generation) == 8, scene
            assert scene["folder"] == f"Simul/Warehouse_{seed}_{generation}", scene
            paths = scene["actors_spawned"]
            assert len(paths) == len(plan.pieces) == 1 + 4 + 2 + 4
            listed = await client.list_actors(tag_filter=tag, max_results=50)
            by_path = {actor["path"]: actor for actor in listed["actors"]}
            assert set(by_path) == set(paths), listed
            for piece, path in zip(plan.pieces, paths):
                assert by_path[path]["name"] == piece.label
                assert by_path[path]["location"] == pytest.approx(list(piece.center), abs=0.01)
                assert by_path[path]["scale"] == pytest.approx([s / 100 for s in piece.size], abs=1e-4)
                assert by_path[path]["mobility"] == ("Movable" if piece.movable else "Static")

            analysis = await client.analyze_scene_for_robotics(actor_filter="Warehouse_")
            assert not analysis.get("error"), analysis
            ours = set(paths)
            floor = [entry for entry in analysis["traversable_surfaces"] if entry["path"] in ours]
            assert [entry["path"] for entry in floor] == [paths[0]]
            assert floor[0]["size"] == pytest.approx([1500, 1200, 10], abs=0.01)
            assert floor[0]["top_z"] == pytest.approx(20010, abs=0.01)
            graspable = {entry["path"]: entry for entry in analysis["graspable_objects"] if entry["path"] in ours}
            expected = {
                path for piece, path in zip(plan.pieces, paths) if piece.movable and max(piece.size) <= 60
            }
            assert set(graspable) == expected
            assert all(entry["mass_kg"] > 0 for entry in graspable.values())
            summary = analysis["collision_summary"]
            assert summary["missing_simple_collision"] == 0
            assert summary["static_mesh_components"] >= len(paths)

            level = await client.analyze_scene_for_robotics()
            assert not level.get("error"), level
            assert level["total_actors_analyzed"] >= len(paths)
        finally:
            await client._execute_python(_remove_tagged(prefix))
            await client.close()

    asyncio.run(run())


_ROTATED_PIECES = """
import unreal, json
sub = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
cube = unreal.load_asset('/Engine/BasicShapes/Cube.Cube')
paths = {}
for label, loc, (pitch, yaw, roll), scale, movable in args['pieces']:
    actor = sub.spawn_actor_from_object(cube, unreal.Vector(*loc), unreal.Rotator(roll=roll, pitch=pitch, yaw=yaw))
    actor.set_actor_scale3d(unreal.Vector(*scale))
    actor.set_actor_label(label)
    actor.set_editor_property('tags', [unreal.Name(args['tag'])])
    if movable:
        actor.root_component.set_mobility(unreal.ComponentMobility.MOVABLE)
    paths[label] = actor.get_path_name()
print(json.dumps(paths))
"""


def test_robotics_analysis_measures_rotated_pieces_by_their_own_box_and_pages():
    """The world-axis box read a 20 m x 0.5 m plank yawed 45 degrees as 1449 x
    1449 cm (210 m2), rejected a slab pitched 14 degrees and one rolled 180
    (U2-06), and a large level's result was cut by the server with every surface
    lost (U2-05)."""

    async def run():
        client = session()
        tag = "simul_review_obb_" + uuid.uuid4().hex[:8]
        pieces = [
            ("ObbPlank", (0, 0, 30005), (0, 45, 0), (20, 0.5, 0.1), False),
            ("ObbSlope", (0, 3000, 30005), (14, 0, 0), (10, 10, 0.1), False),
            ("ObbSteep", (0, 6000, 30005), (16, 0, 0), (10, 10, 0.1), False),
            ("ObbFlipped", (0, 9000, 30005), (0, 0, 180), (10, 10, 0.1), False),
            ("ObbTurnedBox", (0, 12000, 30025), (0, 45, 0), (0.5, 0.5, 0.5), True),
        ]
        try:
            made = await client._execute_json_script(
                f"args = {{'tag': {tag!r}, 'pieces': {pieces!r}}}\n" + _ROTATED_PIECES
            )
            assert not made.get("error"), made
            analysis = await client.analyze_scene_for_robotics(actor_filter="Obb", max_results=10)
            ours = set(made.values())
            surfaces = {entry["name"]: entry for entry in analysis["traversable_surfaces"] if entry["path"] in ours}
            assert set(surfaces) == {"ObbPlank", "ObbSlope", "ObbFlipped"}, analysis
            assert surfaces["ObbPlank"]["size"] == pytest.approx([2000, 50, 10], abs=0.01)
            assert surfaces["ObbPlank"]["area_m2"] == pytest.approx(10.0)
            assert surfaces["ObbSlope"]["tilt_degrees"] == pytest.approx(14.0, abs=0.01)
            assert surfaces["ObbFlipped"]["tilt_degrees"] == pytest.approx(0.0, abs=0.01)
            assert surfaces["ObbFlipped"]["top_z"] == pytest.approx(30010, abs=0.01)
            grasped = [entry for entry in analysis["graspable_objects"] if entry["path"] in made.values()]
            assert [entry["name"] for entry in grasped] == ["ObbTurnedBox"], analysis
            assert grasped[0]["size"] == pytest.approx([50, 50, 50], abs=0.01)

            paged = await client.analyze_scene_for_robotics(actor_filter="Obb", max_results=1)
            assert paged["returned"] == {"traversable_surfaces": 1, "graspable_objects": 1}, paged
            assert paged["totals"]["traversable_surfaces"] >= 3 and paged["next_offset"] == 1, paged
            assert "offset 1" in paged["hint"], paged
            following = await client.analyze_scene_for_robotics(actor_filter="Obb", max_results=1, offset=1)
            assert following["traversable_surfaces"][0]["path"] != paged["traversable_surfaces"][0]["path"]
        finally:
            await client._execute_python(_remove_tagged(tag))
            await client.close()

    asyncio.run(run())


_LIGHT_STATE = """
import unreal, json
light = unreal.load_object(None, args['path']).get_components_by_class(unreal.LightComponentBase)[0]
print(json.dumps({'intensity': light.get_editor_property('intensity'),
                  'color': [light.get_editor_property('light_color').r, light.get_editor_property('light_color').g]}))
"""


def test_lights_are_checked_before_writing_and_sky_lights_are_set():
    """intensity -5 turned the light to 0 while the reply said it was not
    applied (U2-09); a SkyLight was refused, and a missing actor with nothing to
    set returned success (U2-13)."""

    async def run():
        client = session()
        made = []
        try:
            for cls in ("PointLight", "DirectionalLight", "SkyLight"):
                spawned = await client.spawn_actor(f"/Script/Engine.{cls}", (0, 0, 31000))
                assert not spawned.get("error"), spawned
                made.append(spawned["actor_path"])
            point, sun, sky = made

            async def state(path):
                return await client._execute_json_script(f"args = {{'path': {path!r}}}\n" + _LIGHT_STATE)

            before = await state(point)
            with pytest.raises(ValueError, match="intensity"):
                await client.set_light_params(point, intensity=-5.0)
            with pytest.raises(ValueError, match="temperature"):
                await client.set_light_params(point, temperature=100.0)
            assert await state(point) == before

            set_point = await client.set_light_params(point, intensity=5000.0)
            assert (set_point["intensity_unit"], set_point["params_set"]) == ("candelas", 1), set_point
            assert (await state(point))["intensity"] == pytest.approx(5000.0)
            refused = await client.set_light_params(sun, intensity=9.0, attenuation_radius=500.0)
            assert refused["error_type"] == "ValidationError", refused
            assert (await state(sun))["intensity"] != pytest.approx(9.0)
            assert (await client.set_light_params(sun))["intensity_unit"] == "lux"

            set_sky = await client.set_light_params(sky, intensity=2.0, color=[1.0, 0.5, 0.25])
            assert (set_sky["light_class"], set_sky["intensity_unit"]) == ("SkyLightComponent", "scale"), set_sky
            assert await state(sky) == {"intensity": pytest.approx(2.0), "color": [255, 128]}

            missing = await client.set_light_params(point + "_Gone")
            assert missing["success"] is False and "Actor not found" in missing["error"], missing
        finally:
            for path in made:
                await destroy_actor(client, path)
            await client.close()

    asyncio.run(run())


_TAG_BODIES = """
import unreal, json
out = {'n': 0, 'simulating': 0, 'physics_actor': 0}
for actor in unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors():
    if unreal.Name(args['tag']) in actor.tags and '_Box_' in actor.get_actor_label():
        c = actor.root_component
        out['n'] += 1
        out['simulating'] += bool(c.get_editor_property('body_instance').get_editor_property('simulate_physics'))
        out['physics_actor'] += str(c.get_collision_profile_name()) == 'PhysicsActor'
print(json.dumps(out))
"""


def test_bulk_physics_sets_up_a_scene_s_props_in_one_undoable_call():
    """One call per prop ran into the rate limit: 126 bodies took 116 s (U2-10);
    and the generator's props came out not simulating (U2-16)."""

    async def run():
        client = session()
        seed = uuid.uuid4().int % 100000
        prefix = f"simul_room_{seed}_"
        try:
            scene = await client.generate_procedural_scene(
                "room", {"seed": seed, "props": 30, "door_width": 0}, [0, 0, 32000], [1500, 1200, 32400]
            )
            assert not scene.get("error"), scene
            tag = scene["tag"]

            async def bodies():
                return await client._execute_json_script(f"args = {{'tag': {tag!r}}}\n" + _TAG_BODIES)

            assert await bodies() == {"n": 30, "simulating": 0, "physics_actor": 0}
            bulk = await client.setup_physics_bulk(tag=tag, actor_filter="_Box_", collision_preset="PhysicsActor")
            assert (bulk["matched"], bulk["configured_count"], bulk["failed_count"]) == (30, 30, 0), bulk
            assert await bodies() == {"n": 30, "simulating": 30, "physics_actor": 30}
            await transaction(client, "UNDO")  # one Undo reverts the whole call
            assert await bodies() == {"n": 30, "simulating": 0, "physics_actor": 0}
            await transaction(client, "REDO")
            assert (await bodies())["simulating"] == 30

            ready = await client.generate_procedural_scene(
                "room", {"seed": seed, "props": 5, "door_width": 0}, [3000, 0, 32000], [4500, 1200, 32400],
                physics_props=True,
            )
            assert ready["physics_props"] == 5, ready
            ready_bodies = await client._execute_json_script(f"args = {{'tag': {ready['tag']!r}}}\n" + _TAG_BODIES)
            assert ready_bodies == {"n": 5, "simulating": 5, "physics_actor": 5}, ready_bodies
        finally:
            await client._execute_python(_remove_tagged(prefix))
            await client.close()

    asyncio.run(run())


# Spawns actors that each hold one instanced static mesh component, added the
# way the editor's Add Component button does, and the instances given.
_INSTANCED_ACTORS = """
import json, unreal
def build():
    sub = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    data = unreal.get_engine_subsystem(unreal.SubobjectDataSubsystem)
    lib = unreal.SubobjectDataBlueprintFunctionLibrary
    made = {}
    for label, cls, mesh, z, instances in args['actors']:
        actor = sub.spawn_actor_from_class(unreal.Actor, unreal.Vector(0, 0, z), unreal.Rotator(0, 0, 0))
        actor.set_actor_label(label)
        actor.set_editor_property('tags', [unreal.Name(args['tag'])])
        root = data.k2_gather_subobject_data_for_instance(actor)[0]
        handle, reason = data.add_new_subobject(unreal.AddNewSubobjectParams(
            parent_handle=root, new_class=getattr(unreal, cls), blueprint_context=None))
        component = lib.get_object(lib.get_data(handle))
        component.set_editor_property('static_mesh', unreal.load_asset(mesh))
        for location, rotation, scale in instances:
            component.add_instance(unreal.Transform(location=unreal.Vector(*location),
                                                    rotation=unreal.Rotator(**rotation),
                                                    scale=unreal.Vector(*scale)), True)
        made[label] = {'path': actor.get_path_name(), 'component': component.get_path_name(),
                       'count': component.get_instance_count()}
    return made
print(json.dumps(build()))
"""


def test_instanced_meshes_are_measured_one_instance_at_a_time_and_later_pages_read_the_scan():
    """An instanced mesh component was measured as one mesh at its own
    transform, so a floor of instanced tiles read as a single tile; and every
    page scanned the level again (a 5,011-actor level took 13 to 99 s a page)."""

    async def run():
        client = session()
        tag = "simul_review_ism_" + uuid.uuid4().hex[:8]
        prefix = "Ism" + tag[-8:]
        flat, small = {"roll": 0, "pitch": 0, "yaw": 0}, (0.3, 0.3, 0.3)
        tiles = [((i * 200.0, j * 200.0, 0.0), dict(flat, pitch=10) if i * 5 + j == 7 else flat, (2, 2, 1))
                 for i in range(4) for j in range(5)]
        boxes = [((i * 100.0, 0.0, 0.0), dict(flat, yaw=45) if i == 5 else flat, small) for i in range(12)]
        many = [(((i % 50) * 30.0, (i // 50) * 30.0, 0.0), flat, (0.2, 0.2, 0.2)) for i in range(1001)]
        plane, cube = "/Engine/BasicShapes/Plane.Plane", "/Engine/BasicShapes/Cube.Cube"
        actors = [
            (prefix + "Floor", "HierarchicalInstancedStaticMeshComponent", plane, 33000, tiles),
            (prefix + "Shelf", "InstancedStaticMeshComponent", cube, 34000, boxes),
            (prefix + "Gravel", "InstancedStaticMeshComponent", cube, 35000, many),
        ]
        try:
            made = await client._execute_json_script(f"args = {{'tag': {tag!r}, 'actors': {actors!r}}}\n"
                                                     + _INSTANCED_ACTORS)
            assert not made.get("error"), made
            assert [made[label]["count"] for label, *_ in actors] == [20, 12, 1001], made
            floor, shelf, gravel = (made[label] for label, *_ in actors)

            analysis = await client.analyze_scene_for_robotics(actor_filter=prefix, max_results=50)
            assert analysis["totals"] == {"traversable_surfaces": 21, "graspable_objects": 12}, analysis
            per_tile = {entry["instance_index"]: entry for entry in analysis["traversable_surfaces"]
                        if entry.get("component_path") == floor["component"]}
            assert sorted(per_tile) == list(range(20)), analysis
            assert all(entry["size"] == pytest.approx([200, 200, 0], abs=0.01) for entry in per_tile.values())
            assert per_tile[7]["tilt_degrees"] == pytest.approx(10.0, abs=0.01)
            assert per_tile[0]["tilt_degrees"] == pytest.approx(0.0, abs=0.01)
            assert per_tile[0]["area_m2"] == pytest.approx(4.0)
            whole = [entry for entry in analysis["traversable_surfaces"] if entry["path"] == gravel["path"]]
            assert len(whole) == 1 and whole[0]["instances"] == 1001, whole
            assert "instance_index" not in whole[0]
            grasped = {entry["instance_index"]: entry for entry in analysis["graspable_objects"]}
            assert sorted(grasped) == list(range(12)), analysis
            assert all(entry["path"] == shelf["path"] and entry["mass_kg"] is None for entry in grasped.values())
            assert all(entry["size"] == pytest.approx([30, 30, 30], abs=0.01) for entry in grasped.values())

            first = await client.analyze_scene_for_robotics(actor_filter=prefix, max_results=5)
            await destroy_actor(client, floor["path"])
            later = await client.analyze_scene_for_robotics(actor_filter=prefix, max_results=5, offset=5)
            assert later["scanned_at"] == first["scanned_at"] and later["snapshot_age_s"] > 0, later
            assert later["totals"] == first["totals"] == analysis["totals"], later  # the scan, not the level
            rescanned = await client.analyze_scene_for_robotics(actor_filter=prefix, max_results=5)
            assert rescanned["scanned_at"] > first["scanned_at"], rescanned
            assert rescanned["totals"] == {"traversable_surfaces": 1, "graspable_objects": 12}, rescanned
        finally:
            await client._execute_python(_remove_tagged(tag))
            await client.close()

    asyncio.run(run())


_BODY_STATE = """
import unreal, json
def read(path):
    c = unreal.load_object(None, path).root_component
    body = c.get_editor_property('body_instance')
    return {'mobility': c.mobility.name, 'profile': str(c.get_collision_profile_name()),
            'collision': c.get_collision_enabled().name,
            'simulate': body.get_editor_property('simulate_physics'),
            'override_mass': body.get_editor_property('override_mass'),
            'mass_override': round(body.get_editor_property('mass_in_kg_override'), 3),
            'linear_damping': round(body.get_editor_property('linear_damping'), 3),
            'use_default_collision': (c.get_editor_property('use_default_collision')
                                      if isinstance(c, unreal.StaticMeshComponent) else None)}
print(json.dumps({path: read(path) for path in args['paths']}))
"""

_BULK_ACTORS = """
import unreal, json
def build():
    sub = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    cube = unreal.load_asset('/Engine/BasicShapes/Cube.Cube')
    paths = []
    for i in range(2):
        actor = sub.spawn_actor_from_object(cube, unreal.Vector(0, i * 400.0, 36000), unreal.Rotator(0, 0, 0))
        actor.set_editor_property('tags', [unreal.Name(args['tag'])])
        paths.append(actor.get_path_name())
    skeletal = sub.spawn_actor_from_class(unreal.SkeletalMeshActor, unreal.Vector(0, 800, 36000),
                                          unreal.Rotator(0, 0, 0))
    skeletal.set_editor_property('tags', [unreal.Name(args['tag'])])
    return paths[:1] + [skeletal.get_path_name()] + paths[1:]
print(json.dumps({'paths': build()}))
"""


def test_bulk_setup_puts_every_actor_back_when_one_does_not_simulate():
    """A body the checks pass that Unreal does not simulate (a skeletal mesh
    actor without a mesh) left the actors before it changed, and the reply said
    it was put back while its body kept the new mass: the struct
    get_editor_property returns refers to the component's own."""

    async def run():
        client = session()
        tag = "simul_review_bulk_" + uuid.uuid4().hex[:8]
        try:
            made = await client._execute_json_script(f"args = {{'tag': {tag!r}}}\n" + _BULK_ACTORS)
            assert not made.get("error"), made
            paths = made["paths"]
            cube_a, skeletal, cube_b = paths

            async def bodies():
                return await client._execute_json_script(f"args = {{'paths': {paths!r}}}\n" + _BODY_STATE)

            before = await bodies()
            assert before[cube_a]["mobility"] == "STATIC" and before[cube_a]["override_mass"] is False, before
            refused = await client.setup_physics_bulk(
                actor_paths=paths, collision_preset="PhysicsActor", mass=2.0, linear_damping=0.5
            )
            assert refused["error_type"] == "PhysicsNotApplied", refused
            assert refused["details"]["code"] == "NOT_APPLIED"
            assert refused["details"]["expected"]["mass"] == 2.0
            assert list(refused["details"]["actual"]) == [skeletal], refused
            assert await bodies() == before  # the first cube's mobility, preset and mass included

            kept = await client.setup_physics_bulk(
                actor_paths=paths, collision_preset="PhysicsActor", mass=2.0, linear_damping=0.5, skip_failures=True
            )
            assert (kept["configured_count"], kept["failed_count"]) == (2, 1), kept
            assert "put back as it was" in kept["failed"][0]["error"]
            after = await bodies()
            assert after[skeletal] == before[skeletal], after
            for cube in (cube_a, cube_b):
                assert (after[cube]["profile"], after[cube]["simulate"], after[cube]["mass_override"]) == (
                    "PhysicsActor", True, 2.0
                ), after
            await transaction(client, "UNDO")
            assert await bodies() == before

            nothing = await client.setup_physics_bulk(actor_paths=[skeletal], skip_failures=True)
            assert nothing["error_type"] == "IncompatibleBody", nothing
            assert nothing["details"]["code"] == "NOTHING_SET_UP", nothing
            assert (await bodies())[skeletal] == before[skeletal]
        finally:
            await client._execute_python(_remove_tagged(tag))
            await client.close()

    asyncio.run(run())


_LIGHT_VALUES = """
import unreal, json
light = unreal.load_object(None, args['path']).get_components_by_class(unreal.LightComponentBase)[0]
color = light.get_editor_property('light_color')
print(json.dumps({'intensity': light.get_editor_property('intensity'),
                  'cast_shadows': light.get_editor_property('cast_shadows'),
                  'color': [color.r, color.g, color.b, color.a]}))
"""


def test_a_light_value_that_does_not_take_puts_every_value_back():
    """The value Unreal stored differently was put back but the others in the
    call stayed written; a non-light actor failed with a traceback and an
    intensity of 1e30 was stored."""
    from simul.adapters._unreal_scripts import SET_LIGHT_PARAMS

    async def run():
        client = session()
        made = []
        try:
            for spawn in (("/Script/Engine.PointLight", (0, 0, 37000)),
                          ("/Engine/BasicShapes/Cube.Cube", (0, 400, 37000))):
                spawned = await client.spawn_actor(*spawn)
                assert not spawned.get("error"), spawned
                made.append(spawned["actor_path"])
            point, cube = made

            async def values():
                return await client._execute_json_script(f"args = {{'path': {point!r}}}\n" + _LIGHT_VALUES)

            before = await values()
            # UE 5.7.4 stores every in-range value the session lets through as given, so a
            # stand-in rejects the intensity on read-back; the put-back and cancel are Unreal's.
            header = "def differs(name, wanted, value):\n"
            stand_in = "    if name == 'intensity':\n        return True\n"
            rejecting = SET_LIGHT_PARAMS.replace(header, header + stand_in, 1)
            assert rejecting != SET_LIGHT_PARAMS
            args = {"actor_path": point, "values": {"intensity": 5000.0, "cast_shadows": False,
                                                    "light_color": [10, 20, 30, 255]}}
            result = await client._execute_json_script(f"args = {args!r}\n" + rejecting)
            assert result["error_type"] == "LightParamsNotApplied", result
            assert result["details"]["code"] == "NOT_APPLIED", result
            assert (result["applied"], result["params_set"]) == ({}, 0), result
            assert await values() == before

            not_a_light = await client.set_light_params(cube, intensity=1.0)
            assert (not_a_light["error_type"], not_a_light["details"]["code"]) == ("ValidationError", "NOT_A_LIGHT")
            assert "Traceback" not in not_a_light["error"], not_a_light
            with pytest.raises(ValueError, match="intensity must be from 0 to 1e"):
                await client.set_light_params(point, intensity=1e30)
            assert await values() == before
        finally:
            for path in made:
                await destroy_actor(client, path)
            await client.close()

    asyncio.run(run())


# Editor-side counts of outliner folder objects (World Partition levels file
# folders as ActorFolder objects; others have none, and count 0) and of the
# actors carrying a tag prefix.
_LEVEL_STATE = """
import json, unreal
folders = 0
for obj in unreal.ObjectIterator(unreal.Object):
    try:
        full = unreal.Object.get_full_name(obj)
    except Exception:
        continue
    if full.startswith("ActorFolder ") and "Default__" not in full:
        folders += 1
actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
tagged = [a for a in actors if any(str(t).startswith(args["prefix"]) for t in a.tags)]
print(json.dumps({"folders": folders, "tagged": len(tagged),
                  "in_folder": sorted({str(a.get_folder_path()) for a in tagged})}))
"""


def test_generating_a_scene_twice_files_each_call_apart():
    """The same scene type and seed used to share one tag and outliner folder."""

    async def run():
        client = session()
        seed = uuid.uuid4().int % 100000
        prefix = f"simul_room_{seed}_"
        arguments = ("room", {"seed": seed, "props": 1}, [0, 0, 21000], [600, 600, 21300])
        try:
            first = await client.generate_procedural_scene(*arguments)
            second = await client.generate_procedural_scene(*arguments)
            assert not first.get("error") and not second.get("error"), (first, second)
            assert first["tag"] != second["tag"] and first["folder"] != second["folder"]
            for scene in (first, second):
                listed = await client.list_actors(tag_filter=scene["tag"], max_results=50)
                assert sorted(a["path"] for a in listed["actors"]) == sorted(scene["actors_spawned"]), listed
                state = await client._execute_json_script(f"args = {{'prefix': {scene['tag']!r}}}\n" + _LEVEL_STATE)
                assert state["in_folder"] == [scene["folder"]], state
        finally:
            await client._execute_python(_remove_tagged(prefix))
            await client.close()

    asyncio.run(run())


def test_a_failed_procedural_scene_leaves_no_actor_or_folder(monkeypatch):
    """Rollback used to leave the scene's outliner folders behind, empty."""
    import dataclasses

    from simul.adapters import unreal_runtime

    real_plan = unreal_runtime.plan_scene

    def broken_plan(*args, **kwargs):
        plan = real_plan(*args, **kwargs)
        pieces = list(plan.pieces)
        # Three pieces spawn before the fourth names a mesh that does not exist.
        pieces[3] = dataclasses.replace(pieces[3], mesh="/Engine/BasicShapes/DoesNotExist.DoesNotExist")
        return dataclasses.replace(plan, pieces=tuple(pieces))

    monkeypatch.setattr(unreal_runtime, "plan_scene", broken_plan)

    async def run():
        client = session()
        seed = uuid.uuid4().int % 100000
        state_script = f"args = {{'prefix': 'simul_room_{seed}_'}}\n" + _LEVEL_STATE
        try:
            before = await client._execute_json_script(state_script)
            result = await client.generate_procedural_scene(
                "room", {"seed": seed, "props": 2}, [0, 0, 22000], [600, 600, 22300]
            )
            after = await client._execute_json_script(state_script)
            assert result["error_type"] == "SpawnError", result
            missing = "/Engine/BasicShapes/DoesNotExist.DoesNotExist"
            assert result["error"] == "Spawned nothing: no static mesh at " + missing
            assert after == before == {"folders": before["folders"], "tagged": 0, "in_folder": []}, (before, after)
        finally:
            await client._execute_python(_remove_tagged(f"simul_room_{seed}_"))
            await client.close()

    asyncio.run(run())


def test_a_material_instance_is_refused_during_play_in_editor_and_created_after():
    """While PIE ran, EditorAssetLibrary found no assets, so the tool reported
    the parent material, which exists, as not found."""

    async def run():
        client = session()
        folder = "/Game/SimulReview_" + uuid.uuid4().hex[:8]
        parent = "/Engine/BasicShapes/BasicShapeMaterial"
        try:
            started = await client.control_simulation("start")
            assert started.get("state") == "playing", started
            refused = await client.create_material_instance(parent, "MI_DuringPIE", folder)
            assert refused["error_type"] == "PIEActive", refused
            assert "control_unreal_simulation" in refused["details"]["hint"], refused

            stopped = await client.control_simulation("stop")
            assert stopped.get("state") == "stopped", stopped
            leftover = await client._execute_json_script(
                "import json, unreal\n"
                f"path = {folder + '/MI_DuringPIE'!r}\n"
                "print(json.dumps({'exists': unreal.EditorAssetLibrary.does_asset_exist(path)}))"
            )
            assert leftover == {"exists": False}, leftover
            created = await client.create_material_instance(parent, "MI_AfterPIE", folder)
            assert created["instance_path"] == f"{folder}/MI_AfterPIE.MI_AfterPIE", created
            assert created["parent_path"] == parent + ".BasicShapeMaterial", created
        finally:
            await client.control_simulation("stop")
            await client._execute_python(f"import unreal\nunreal.EditorAssetLibrary.delete_directory({folder!r})")
            await client.close()

    asyncio.run(run())


def test_play_in_editor_refuses_editor_changes_and_flags_reads():
    """While PIE ran, spawn told the agent to pass another class or asset,
    delete reported deleted false, the reads returned 0 actors and the
    loaded map raised on a None editor world."""

    async def run():
        client = session()
        path = await spawn_cube(client)
        try:
            map_before = (await client.get_loaded_map())["map_path"]
            started = await client.control_simulation("start")
            assert started.get("state") == "playing", started

            spawned = await client.spawn_actor("/Engine/BasicShapes/Cube.Cube")
            assert spawned["error_type"] == "PIEActive", spawned
            assert "control_unreal_simulation" in spawned["details"]["hint"], spawned
            scene = await client.generate_procedural_scene("room", {"props": 0}, [0, 0, 23000], [600, 600, 23300])
            assert scene["error_type"] == "PIEActive", scene
            deleted = await client.delete_actor(path)
            assert deleted["error_type"] == "PIEActive", deleted

            listed = await client.list_actors(tag_filter="simul_review")
            assert listed["pie_active"] is True and path in [a["path"] for a in listed["actors"]], listed
            found = await client.get_actor_by_semantic_label("simul_review")
            assert found["pie_active"] is True and path in [a["path"] for a in found["actors"]], found
            loaded = await client.get_loaded_map()
            assert loaded == {"map_path": map_before, "pie_active": True}, loaded
            summary = await client.summarize_scene()
            assert summary["pie_active"] is True and summary["map_path"] == map_before, summary
            assert summary["total_actors"] > 0, summary
            info = await client.get_actor_info(path)
            assert info["pie_active"] is True and info["tags"] == ["simul_review"], info

            stopped = await client.control_simulation("stop")
            assert stopped.get("state") == "stopped", stopped
            assert (await client.get_actor_info(path))["pie_active"] is False
            assert path in [a["path"] for a in (await client.list_actors(tag_filter="simul_review"))["actors"]]
        finally:
            await client.control_simulation("stop")
            await destroy_actor(client, path)
            await client.close()

    asyncio.run(run())


SELECTION = """
import unreal, json
subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
print(json.dumps({'selection': [a.get_path_name() for a in subsystem.get_selected_level_actors()]}))
"""


def test_focus_selects_outside_play_in_editor_and_says_why_not_during_it():
    """While PIE ran, focus reported success as if it had selected the actor;
    Unreal reads back no selection made then and clears it when PIE stops. The
    camera move works in and out of PIE."""

    async def run():
        client = session()
        path = await spawn_cube(client)
        try:
            outside = await client.focus_on_actor(path)
            assert (outside["selected"], outside["pie_active"]) == (True, False), outside
            # Fitted to the 1 m cube centered at (100, 200, 300): behind its
            # center and half as far above it.
            distance = outside["distance"]
            framed = pytest.approx((100.0 - distance, 200.0, 300.0 + distance / 2))
            assert "warning" not in outside and outside["camera_location"] == framed, outside
            assert path in (await client._execute_json_script(SELECTION))["selection"]

            await client._execute_python(
                "import unreal\nunreal.get_editor_subsystem(unreal.EditorActorSubsystem).select_nothing()"
            )
            await client.set_camera_view(location=(0.0, 0.0, 1000.0), rotation=(0.0, 0.0, 0.0))
            started = await client.control_simulation("start")
            assert started.get("state") == "playing", started

            during = await client.focus_on_actor(path)
            assert (during["selected"], during["pie_active"]) == (False, True), during
            assert "control_unreal_simulation" in during["warning"], during
            assert during["distance"] == pytest.approx(distance), during
            assert during["camera_location"] == framed, during

            stopped = await client.control_simulation("stop")
            assert stopped.get("state") == "stopped", stopped
            assert path not in (await client._execute_json_script(SELECTION))["selection"]
            assert (await client.get_viewport_info())["camera_location"] == framed
        finally:
            await client.control_simulation("stop")
            await destroy_actor(client, path)
            await client.close()

    asyncio.run(run())


SPAWN_FOCUS_ACTORS = """
import unreal, json
subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
cube = unreal.load_asset('/Engine/BasicShapes/Cube')
def tagged(actor, scale=None):
    # A BoxReflectionCapture's scale is the size of its influence box, so the
    # spawn scale is kept unless one is given.
    if scale is not None:
        actor.set_actor_scale3d(unreal.Vector(scale, scale, scale))
    actor.set_editor_property('tags', [unreal.Name('simul_review_focus')])
    return actor.get_path_name()
def cine_camera(y, *focus_plane):
    actor = subsystem.spawn_actor_from_class(unreal.CineCameraActor, unreal.Vector(0, y, 200))
    component = actor.get_cine_camera_component()
    for drawn in focus_plane:
        settings = component.get_editor_property('focus_settings')
        settings.set_editor_property('draw_debug_focus_plane', drawn)
        component.set_editor_property('focus_settings', settings)
    return actor
print(json.dumps({
    # A 20 m cube and a 10 cm one, centered on their locations.
    'large': tagged(subsystem.spawn_actor_from_object(cube, unreal.Vector(0, 3000, 1000)), 20.0),
    'small': tagged(subsystem.spawn_actor_from_object(cube, unreal.Vector(0, 6000, 5)), 0.1),
    # A mesh standing on its location, the center of its bounds 128 cm above it.
    'standing': tagged(subsystem.spawn_actor_from_object(
        unreal.load_asset('/Engine/EngineMeshes/SM_MatPreviewMesh_01'), unreal.Vector(0, 9000, 0))),
    # A StaticMeshActor without a mesh, whose bounds have no size.
    'no_mesh': tagged(subsystem.spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector(0, 12000, 100))),
    # A camera, whose editor-only frustum get_actor_bounds includes.
    'camera': tagged(subsystem.spawn_actor_from_class(unreal.CameraActor, unreal.Vector(0, 15000, 200))),
    # CineCameras: plain, with the debug focus plane drawn, and with it turned off
    # again, which only hides it.
    'cine': tagged(cine_camera(18000)),
    'cine_plane': tagged(cine_camera(21000, True)),
    'cine_plane_off': tagged(cine_camera(24000, True, False)),
    # A SkyLight, which leaves a spare editor icon at the world origin.
    'sky_light': tagged(subsystem.spawn_actor_from_class(unreal.SkyLight, unreal.Vector(0, 27000, 200))),
    # Reflection captures, whose influence shapes collide with nothing.
    'sphere_capture': tagged(
        subsystem.spawn_actor_from_class(unreal.SphereReflectionCapture, unreal.Vector(0, 30000, 200))),
    'box_capture': tagged(subsystem.spawn_actor_from_class(unreal.BoxReflectionCapture, unreal.Vector(0, 33000, 200))),
}))
"""

ACTOR_BOUNDS = """
import unreal, json
origin, extent = unreal.load_object(None, args['path']).get_actor_bounds(False, True)
print(json.dumps({'origin': [origin.x, origin.y, origin.z], 'extent': [extent.x, extent.y, extent.z]}))
"""


def _in_default_view(focused, bounds, fov=90.0, aspect=16.0 / 9.0):
    """The nearest depth of the box's corners seen from the camera read back, and
    the largest share of the view they reach: by default UE's default view, 90
    degrees across, on 16:9."""
    half_width = math.tan(math.radians(fov) / 2)
    pitch, yaw = (math.radians(angle) for angle in focused["camera_rotation"][:2])
    axes = (
        (math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)),
        (-math.sin(yaw), math.cos(yaw), 0.0),
        (-math.sin(pitch) * math.cos(yaw), -math.sin(pitch) * math.sin(yaw), math.cos(pitch)),
    )
    nearest, reach = math.inf, 0.0
    for signs in itertools.product((-1.0, 1.0), repeat=3):
        offset = [
            o + s * e - c
            for o, s, e, c in zip(bounds["origin"], signs, bounds["extent"], focused["camera_location"])
        ]
        depth, across, up = (sum(a * b for a, b in zip(offset, axis)) for axis in axes)
        nearest = min(nearest, depth)
        if depth <= 0:
            reach = math.inf
        else:
            reach = max(reach, abs(across) / depth / half_width, abs(up) / depth / (half_width / aspect))
    return nearest, reach


def test_focus_distance_0_fits_the_actor_and_an_explicit_distance_looks_at_its_location():
    """Distance 0 put the camera a fixed 500 cm from the actor: inside a 20 m
    cube, and a 10 cm cube was a speck. Now it fits the actor's bounds in UE's
    default view, or in the view given, looking at their center, and frames a
    camera by its mesh, not by the frustum the editor draws for it or a
    CineCamera's debug focus plane, a SkyLight by its icon, not by the spare
    one it leaves at the world origin, and a reflection capture by its icon,
    not by its influence shapes; an actor without bounds keeps the 500 cm and
    says so, and an explicit distance looks at the actor's location as before."""

    async def run():
        client = session()
        spawned = await client._execute_json_script(SPAWN_FOCUS_ACTORS)
        assert not spawned.get("error"), spawned
        try:
            for name in ("large", "small", "standing"):
                bounds = await client._execute_json_script(f"args = {{'path': {spawned[name]!r}}}\n" + ACTOR_BOUNDS)
                focused = await client.focus_on_actor(spawned[name])
                nearest, reach = _in_default_view(focused, bounds)
                # In the view, beyond the 10 cm near clip plane, not a speck.
                assert nearest > 10.0 and 0.4 < reach <= 1.0, (name, focused, bounds)
                center = bounds["origin"]
                d = focused["distance"]
                assert focused["camera_location"] == pytest.approx((center[0] - d, center[1], center[2] + d / 2))
                assert "fit_warning" not in focused, focused

            # get_actor_bounds includes the frustum the editor draws for a camera, a cube
            # reaching 10 m on every side of it, which put the camera 35 m away; the fit
            # leaves the frustum out and frames the camera's mesh.
            frustum = await client._execute_json_script(f"args = {{'path': {spawned['camera']!r}}}\n" + ACTOR_BOUNDS)
            assert min(frustum["extent"]) >= 1000.0, frustum
            camera = await client.focus_on_actor(spawned["camera"])
            assert camera["distance"] < 500.0 and "fit_warning" not in camera, camera

            # A CineCamera's debug focus plane, 100 m across at the focus distance (1 km
            # by default), put the camera 1 km away, also after the plane was turned off,
            # which only hides it. The fit leaves the plane out, as the editor's own Focus
            # does, and frames the CineCamera by its mesh, as without the plane.
            cine = await client.focus_on_actor(spawned["cine"])
            assert cine["distance"] < 500.0 and "fit_warning" not in cine, cine
            for name in ("cine_plane", "cine_plane_off"):
                plane = await client._execute_json_script(f"args = {{'path': {spawned[name]!r}}}\n" + ACTOR_BOUNDS)
                assert max(plane["extent"]) >= 5000.0, (name, plane)
                focused = await client.focus_on_actor(spawned[name])
                assert focused["distance"] == pytest.approx(cine["distance"]), (name, focused)
                assert "fit_warning" not in focused, (name, focused)

            # The spare icon a SkyLight leaves at the world origin stretched its bounds to
            # the origin; the fit leaves it out, and the SkyLight is framed by its icon.
            sky = await client._execute_json_script(f"args = {{'path': {spawned['sky_light']!r}}}\n" + ACTOR_BOUNDS)
            assert sky["origin"][1] < 27000.0 - 5000.0, sky
            sky_light = await client.focus_on_actor(spawned["sky_light"])
            d = sky_light["distance"]
            assert d < 1000.0 and "fit_warning" not in sky_light, sky_light
            assert sky_light["camera_location"] == pytest.approx((-d, 27000.0, 200.0 + d / 2)), sky_light

            # A reflection capture's influence shapes, a 30 m sphere around a
            # SphereReflectionCapture, collide with nothing; they stretched its bounds,
            # and the fit now leaves them out and frames the capture by its icon.
            for name, y, influence in (("sphere_capture", 30000.0, 3000.0), ("box_capture", 33000.0, 1000.0)):
                shapes = await client._execute_json_script(f"args = {{'path': {spawned[name]!r}}}\n" + ACTOR_BOUNDS)
                assert max(shapes["extent"]) >= influence, (name, shapes)
                focused = await client.focus_on_actor(spawned[name])
                d = focused["distance"]
                assert d < 500.0 and "fit_warning" not in focused, (name, focused)
                assert focused["camera_location"] == pytest.approx((-d, y, 200.0 + d / 2)), (name, focused)

            # A view of another field of view and shape, 60 degrees across and 2.4 times
            # as wide as tall, crops the 20 m cube framed for the default view; given
            # that view, the fit backs off until the cube fits it.
            bounds = await client._execute_json_script(f"args = {{'path': {spawned['large']!r}}}\n" + ACTOR_BOUNDS)
            default = await client.focus_on_actor(spawned["large"])
            assert _in_default_view(default, bounds, 60.0, 2.4)[1] > 1.0, (default, bounds)
            given = await client.focus_on_actor(spawned["large"], fov=60.0, aspect_ratio=2.4)
            nearest, reach = _in_default_view(given, bounds, 60.0, 2.4)
            assert nearest > 10.0 and 0.4 < reach <= 1.0, (given, bounds)

            no_mesh = await client.focus_on_actor(spawned["no_mesh"])
            assert no_mesh["distance"] == 500.0 and "no bounds" in no_mesh["fit_warning"], no_mesh
            assert no_mesh["camera_location"] == pytest.approx((-500.0, 12000.0, 350.0)), no_mesh

            explicit = await client.focus_on_actor(spawned["standing"], distance=250.0)
            assert explicit["distance"] == 250.0, explicit
            assert explicit["camera_location"] == pytest.approx((-250.0, 9000.0, 125.0)), explicit
        finally:
            await client._execute_python(_remove_tagged("simul_review_focus"))
            await client.close()

    asyncio.run(run())
