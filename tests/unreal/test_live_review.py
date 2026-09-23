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

from simul_mcp.adapters.unreal_runtime import UnrealRuntimeSession
from simul_mcp.config import Settings

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
        target = Path("/tmp/simul_mcp") / f"review-{suffix}" / "export.usda"
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
