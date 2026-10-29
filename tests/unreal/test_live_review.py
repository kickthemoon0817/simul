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
        tag = plan.tag
        try:
            scene = await client.generate_procedural_scene(*arguments)
            assert not scene.get("error"), scene
            assert (scene["tag"], scene["folder"]) == (tag, f"Simul/Warehouse_{seed}")
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
            await client._execute_python(
                "import unreal\n"
                "subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)\n"
                "for actor in subsystem.get_all_level_actors():\n"
                f"    if {tag!r} in [str(t) for t in actor.tags]:\n"
                "        subsystem.destroy_actor(actor)"
            )
            await client.close()

    asyncio.run(run())
