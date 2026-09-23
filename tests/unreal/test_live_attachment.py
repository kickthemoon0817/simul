"""Opt-in checks for attachment against a disposable Unreal editor only.

Use simul unreal setup on a copied project, set UNREAL__PORT, and set
SIMUL_UNREAL_ATTACH_LIVE=1. This test deliberately replaces the scratch map.
"""

import asyncio
import json
import os

import pytest

from simul_mcp.adapters.unreal_connection import UnrealAttachments
from simul_mcp.adapters.unreal_runtime import UnrealRuntimeSession
from simul_mcp.config import Settings
from simul_mcp.mcp.server import SimulMCPServer

pytestmark = pytest.mark.unreal_live


def test_attach_named_controls_and_reject_replaced_map(tmp_path, fake_fastmcp):
    if os.environ.get("SIMUL_UNREAL_ATTACH_LIVE") != "1":
        pytest.skip("SIMUL_UNREAL_ATTACH_LIVE=1 requires a disposable editor")

    async def run():
        base = Settings()
        endpoint = UnrealRuntimeSession(
            base.model_copy(
                update={
                    "unreal": base.unreal.model_copy(update={"mode": "endpoint"}),
                }
            )
        )
        settings = base.model_copy(
            update={
                "unreal": base.unreal.model_copy(
                    update={
                        "mode": "attached",
                        "attachment_path": str(tmp_path / "attachment.json"),
                    }
                ),
                "security": base.security.model_copy(
                    update={
                        "allow_script_execution": False,
                        "rate_limiting_enabled": False,
                    }
                ),
            }
        )
        manager = UnrealAttachments(settings)
        attached = UnrealRuntimeSession(settings)
        path = None
        map_replaced = False
        try:
            created = await endpoint._execute_json_script("""
import unreal, json
sub = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
a = sub.spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector(100,200,300))
a.set_actor_label('Simul attachment sentinel')
print(json.dumps({'path': a.get_path_name()}))
""")
            assert not created.get("error"), created
            path = created["path"]
            info = await manager.describe(base.unreal.host, base.unreal.port)
            assert info["dirty_map_packages"]
            viewports = info["viewports"]
            assert viewports
            if len(viewports) > 1:
                with pytest.raises(ValueError, match="unique level viewport"):
                    await manager.attach(port=base.unreal.port)
            selected = await manager.attach(
                port=base.unreal.port, viewport=viewports[0]
            )
            assert selected["document_id"] == info["document_id"]
            assert selected["dirty_map_packages"] == info["dirty_map_packages"]
            status = await attached.attachment_status()
            assert status["instance_id"] == info["instance_id"], status
            assert (await attached.get_actor_info(path))["location"] == [100, 200, 300]
            assert (await attached.ping())["address"].endswith(
                ":" + str(base.unreal.port)
            )
            server = SimulMCPServer(settings, backends={"unreal"})
            tools = server.mcp.by_name
            assert "execute_unreal_script" not in tools

            async def control(action, **kwargs):
                response = await tools["control_unreal_ui"](
                    agent_control=action, **kwargs
                )
                result = json.loads(response.content[0].text)
                assert result.get("success") is True, result
                return result

            overlay_required = os.environ.get("SIMUL_UNREAL_OVERLAY_LIVE") == "1"
            if overlay_required:
                assert (await control("inspect"))["overlay_available"]
                before = await endpoint._execute_json_script("""
import unreal, json
actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
print(json.dumps({'actors': [a.get_path_name() for a in actors],
                  'dirty': [p.get_path_name() for p in unreal.EditorLoadingAndSavingUtils.get_dirty_map_packages()]}))
""")
                for agent, point in [("builder", [0.3, 0.6]), ("reviewer", [0.7, 0.4])]:
                    marker = (
                        await control(
                            "move_cursor",
                            agent_id=agent,
                            position=point,
                            activity="Inspecting scene",
                        )
                    )["agent_cursor"]
                    assert marker["agent_id"] == agent and marker["position"] == point
                    assert marker["system_cursor_moved"] is False
                cursors = (await control("inspect"))["agent_cursors"]
                assert {c["agent_id"] for c in cursors} == {"builder", "reviewer"}
                after = await endpoint._execute_json_script("""
import unreal, json
actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
print(json.dumps({'actors': [a.get_path_name() for a in actors],
                  'dirty': [p.get_path_name() for p in unreal.EditorLoadingAndSavingUtils.get_dirty_map_packages()]}))
""")
                assert before == after  # overlays never add actors or dirty a package
                result = await control("select_actor", target=path, agent_id="builder")
                assert result["agent_cursor"]["position"] == [0.3, 0.6]
                assert "select_actor" in result["agent_cursor"]["label"]
                assert (await control("clear_cursor", agent_id="builder"))["removed"]
                assert not (await control("clear_cursor", agent_id="builder"))[
                    "removed"
                ]
                assert [
                    c["agent_id"] for c in (await control("inspect"))["agent_cursors"]
                ] == ["reviewer"]
            await control("select_actor", target=path)
            inspected = await control("inspect")
            assert inspected["selected_actors"][0]["path"] == path
            for prop, value in [
                ("location", [110, 220, 330]),
                ("rotation", [5, 10, 15]),
                ("scale", [1, 2, 3]),
            ]:
                result = await control("set_property", property_name=prop, value=value)
                assert result["value"] == pytest.approx(value)
            await control("pilot_actor", target=path)
            assert (await control("inspect"))["pilot_actor"] == path
            await control("eject_actor")
            initial_game_view = inspected["game_view"]
            result = await control("set_game_view", enabled=not initial_game_view)
            assert result["enabled"] is not initial_game_view
            await control("set_game_view", enabled=initial_game_view)
            await control("clear_selection")
            assert (await control("inspect"))["selected_actors"] == []
            # Unknown actions cannot become console commands, even with scripts off.
            rejected = await tools["control_unreal_ui"](agent_control="py print(1)")
            assert json.loads(rejected.content[0].text)["success"] is False
            # New map events invalidate even a map with the same display name.
            result = await endpoint._execute_json_script("""
import unreal, json
unreal.EditorLoadingAndSavingUtils.new_blank_map(False)
print(json.dumps({'changed': True}))
""")
            assert not result.get("error"), result
            map_replaced = True
            stale = await attached.control_ui("inspect")
            assert (
                stale.get("success") is False and "attachment changed" in stale["error"]
            ), stale
            with pytest.raises(RuntimeError, match="attachment changed"):
                await attached._call_function(
                    "/Script/UnrealEd.Default__EditorActorSubsystem",
                    "SetSelectedLevelActors",
                    {"ActorsToSelect": []},
                )
            current = await manager.describe(base.unreal.host, base.unreal.port)
            assert current["document_id"] != info["document_id"]
            await manager.attach(
                port=base.unreal.port, viewport=current["viewports"][0]
            )
            fresh = UnrealRuntimeSession(settings)
            try:
                inspection = await fresh.control_ui("inspect")
                assert not inspection.get("error")
                if overlay_required:
                    assert inspection["agent_cursors"] == []
            finally:
                await fresh.close()
            manager.detach()
            with pytest.raises(RuntimeError, match="No Unreal editor attached"):
                await UnrealRuntimeSession(settings).control_ui("inspect")
        finally:
            if path and not map_replaced:
                await endpoint._execute_python(
                    f"import unreal\na = unreal.load_object(None, {path!r})\n"
                    "if a: unreal.get_editor_subsystem(unreal.EditorActorSubsystem).destroy_actor(a)"
                )
            await attached.close()
            await endpoint.close()

    asyncio.run(run())
