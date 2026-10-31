"""Full-surface regressions against an explicitly opted-in disposable editor.

Set SIMUL_UNREAL_REVIEW_LIVE=1 and UNREAL__PORT for a scratch project set up
by simul unreal setup. USD round-trip additionally requires USDImporter enabled.
Never point this opt-in tier at an editor with user work.
"""

import asyncio
import base64
import io
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
            result = await client.apply_force(path, force_z=100, is_impulse=True)
            assert result["force_applied"] is True, result
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
                assert roughness == {"name": "Roughness", "param_type": "scalar", "value": 0.25, "overridden": True}
                assert (color["name"], color["param_type"], color["overridden"]) == ("Color", "vector", False)
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
        # 500 cm behind the cube at (100, 200, 300) and 250 cm above it.
        framed = pytest.approx((-400.0, 200.0, 550.0))
        try:
            outside = await client.focus_on_actor(path)
            assert (outside["selected"], outside["pie_active"]) == (True, False), outside
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
