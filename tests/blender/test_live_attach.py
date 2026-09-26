"""Opt-in GUI attachment test using only a new, disposable Blender process."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import pytest

from simul.adapters.blender_connection import (
    AttachmentStale,
    AttachmentTargetChanged,
    BlenderAttachments,
    BlenderConnection,
)
from simul.blender_bridge.protocol import BridgeFiles, BridgeWire
from simul.config import Settings
from simul.mcp.server import SimulMCPServer

pytestmark = pytest.mark.blender_live


def test_attach_existing_gui_and_refuse_changed_targets(
    tmp_path: Path, fake_fastmcp: Any
) -> None:
    if os.environ.get("SIMUL_BLENDER_LIVE") != "1":
        pytest.skip("Set SIMUL_BLENDER_LIVE=1 to launch a disposable GUI Blender")
    binary = shutil.which("blender")
    if (
        sys.platform == "darwin"
        and Path("/Applications/Blender.app/Contents/MacOS/Blender").is_file()
    ):
        binary = "/Applications/Blender.app/Contents/MacOS/Blender"
    if binary is None:
        pytest.skip("Blender executable unavailable")
    addon = BlenderAttachments.build_addon(tmp_path / "addon.zip")
    with zipfile.ZipFile(addon) as archive:
        archive.extractall(tmp_path / "addons")
    startup = tmp_path / "startup.py"
    startup.write_text(
        "import sys, bpy\n"
        f"sys.path.insert(0, {str(tmp_path / 'addons')!r})\n"
        "import simul_blender_bridge\n"
        "simul_blender_bridge.register()\n"
        "bpy.context.scene['attachment_sentinel'] = 'unsaved user work'\n"
    )
    settings = Settings(
        blender={
            "mode": "attached",
            "discovery_dir": str(tmp_path),
            "attachment_path": str(tmp_path / "attachment.json"),
        },
        security={"allowed_paths": [str(tmp_path)], "rate_limiting_enabled": False},
    )
    environment = {**os.environ, "SIMUL_BLENDER_DISCOVERY_DIR": str(tmp_path)}
    with (tmp_path / "blender.log").open("w") as log:
        process = subprocess.Popen(
            [
                binary,
                "--factory-startup",
                "--no-window-focus",
                "--python",
                str(startup),
            ],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            manager = BlenderAttachments(settings)
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    pytest.fail((tmp_path / "blender.log").read_text())
                if any(item["reachable"] for item in manager.instances()):
                    break
                time.sleep(0.2)
            else:
                pytest.fail("Blender bridge did not start")
            # Blender is already running; attachment must not reload its scene.
            attached = manager.attach()
            client = BlenderConnection(settings)
            info = client.get_runtime_info()
            assert info["pid"] == process.pid and info["background"] is False
            assert info["window_id"] == attached["window"]["window_id"]
            endpoint = BridgeFiles.read(Path(settings.blender.attachment_path))
            request = {
                "request_id": "refusal-probe",
                "instance_id": endpoint["instance_id"],
                "target": endpoint["target"],
                "deadline": time.time() + 5,
                "method": "create_object",
            }
            with pytest.raises(RuntimeError, match="authentication failed"):
                BridgeWire.request({**endpoint, "token": "incorrect"}, request, 2)
            with pytest.raises(RuntimeError, match="expired before execution"):
                BridgeWire.request(
                    endpoint, {**request, "deadline": time.time() - 1}, 2
                )
            with pytest.raises(RuntimeError, match="window closed"):
                BridgeWire.request(
                    endpoint,
                    {
                        **request,
                        "target": {**endpoint["target"], "window_id": "closed"},
                    },
                    2,
                )
            sentinel = client.execute_script(
                "__result__ = bpy.context.scene['attachment_sentinel']"
            )
            assert "unsaved user work" in sentinel["return_value"]
            main_thread = client.execute_script(
                "import threading\n__result__ = threading.current_thread() is threading.main_thread()"
            )
            assert main_thread["return_value"] == "True"
            server = SimulMCPServer(settings, backends={"blender"})
            ui_settings = settings.model_copy(
                update={
                    "security": settings.security.model_copy(
                        update={"allow_script_execution": False}
                    )
                }
            )
            ui_server = SimulMCPServer(ui_settings, backends={"blender"})
            ui_tool = next(
                t for t in ui_server.mcp.tools if t.name == "control_blender_ui"
            )

            def ui(action: str, **kwargs: Any) -> dict[str, Any]:
                response = asyncio.run(ui_tool.func(agent_control=action, **kwargs))
                return json.loads(response.content[0].text)

            # Named UI actions still work when arbitrary scripts are disabled.
            layout = ui("inspect")
            assert layout["success"] is True
            assert layout["window_id"] == info["window_id"]
            view = next(a for a in layout["areas"] if a["type"] == "VIEW_3D")
            assert ui("select_object", target="Cube")["active_object"] == "Cube"
            assert ui("set_tool", target="move")["tool_id"] == "builtin.move"
            assert ui("show_properties", target="OBJECT")["context"] == "OBJECT"
            assert ui("set_property", target="location", value=[0, 0, 1])["value"] == [
                0,
                0,
                1,
            ]
            assert (
                ui("set_property", target="location", value=[0, 0, 0])["success"]
                is True
            )
            assert ui("move_cursor", area_id=view["area_id"])["position"] == [0.5, 0.5]
            assert ui("move_cursor", area_id="stale")["success"] is False
            assert ui("open_menu", target="python.console")["success"] is False
            assert (
                ui("set_property", target="__class__", value=[0, 0, 0])["success"]
                is False
            )
            create_tool = next(
                t for t in server.mcp.tools if t.name == "create_blender_object"
            )
            created = json.loads(
                asyncio.run(
                    create_tool.func(object_type="CUBE", name="AttachmentProbe")
                )
                .content[0]
                .text
            )
            assert created["success"] is True
            assert created["name"] == "AttachmentProbe"
            transform_tool = next(
                t for t in server.mcp.tools if t.name == "set_blender_object_transform"
            )
            moved = json.loads(
                asyncio.run(
                    transform_tool.func(
                        object_name="AttachmentProbe", location=[1, 2, 3]
                    )
                )
                .content[0]
                .text
            )
            assert moved["success"] is True and moved["location"] == [1, 2, 3]
            client.delete_object("AttachmentProbe")
            client.execute_script(
                "bpy.context.view_layer.objects.active = bpy.data.objects['Cube']\n"
                "bpy.data.objects['Cube'].select_set(True)\n"
                "bpy.ops.object.mode_set(mode='EDIT')"
            )
            with pytest.raises(RuntimeError, match="Object Mode"):
                client.create_object("CUBE", name="MustNotModifyActiveMesh")
            client.execute_script("bpy.ops.object.mode_set(mode='OBJECT')")
            capture = client.capture_viewport(
                width=64, height=64, use_render_fallback=True, agent_id="viewer"
            )
            assert base64.b64decode(capture["image_base64"]).startswith(b"\xff\xd8")
            assert capture["format"] == "jpeg"
            # Through MCP the JPEG is labelled as one, and a label the overlay
            # cannot draw is refused before Blender is touched.
            capture_tool = next(
                t for t in server.mcp.tools if t.name == "capture_blender_viewport"
            )
            mcp_capture = asyncio.run(
                capture_tool.func(
                    width=64, height=64, use_render_fallback=True, agent_id="viewer"
                )
            )
            assert mcp_capture.content[0].mimeType == "image/jpeg"
            refused = json.loads(
                asyncio.run(capture_tool.func(width=64, height=64, agent_id="   "))
                .content[0]
                .text
            )
            assert refused["success"] is False
            # A server newer than the installed add-on names the fix.
            with pytest.raises(ValueError, match="protocol mismatch"):
                BridgeWire.request({**endpoint, "protocol": 1}, request, 2)
            watching = ui("inspect")["agent_observations"]
            assert [m["agent_id"] for m in watching] == ["viewer"]
            assert watching[0]["area_id"] == view["area_id"]
            assert ui("observe", agent_id="reviewer")["success"] is True
            assert {m["agent_id"] for m in ui("inspect")["agent_observations"]} == {
                "viewer",
                "reviewer",
            }
            time.sleep(2.6)
            assert ui("inspect")["agent_observations"] == []
            # A failed capture must not announce that the agent saw an image.
            client.execute_script("bpy.context.scene.camera = None")
            try:
                with pytest.raises(RuntimeError, match="camera"):
                    client.capture_viewport(width=64, height=64, agent_id="failed")
                assert ui("inspect")["agent_observations"] == []
            finally:
                client.execute_script(
                    "bpy.context.scene.camera = bpy.data.objects['Camera']"
                )
            sequence = client.capture_viewport_sequence(
                start_frame=1,
                end_frame=2,
                width=64,
                height=64,
                agent_id="sequence-viewer",
            )
            assert sequence["frame_count"] == 2
            assert [m["agent_id"] for m in ui("inspect")["agent_observations"]] == [
                "sequence-viewer"
            ]

            # Virtual cursors belong to their agent and window, independently.
            ui("move_cursor", agent_id="planner", position=[0.25, 0.5])
            ui("move_cursor", agent_id="builder", position=[0.75, 0.5])
            markers = ui("inspect")["agent_cursors"]
            assert {m["agent_id"] for m in markers} >= {"planner", "builder"}
            assert ui("clear_cursor", agent_id="planner")["removed"] is True
            assert "builder" in {m["agent_id"] for m in ui("inspect")["agent_cursors"]}

            # A different main window may share workspace tools even when its
            # scene differs. Refuse that mutation and explicitly isolate first.
            client.execute_script("bpy.ops.wm.window_new_main()")
            shared = ui("inspect")
            assert shared["shared_window_ids"]
            assert ui("set_tool", target="scale")["success"] is False
            isolation = ui("isolate_workspace")
            assert isolation["completion"] == "workspace_copy_requested"
            isolated = ui("inspect")
            assert isolated["shared_window_ids"] == []
            assert isolated["workspace_id"] != shared["workspace_id"]
            assert isolated["agent_cursors"] == []  # old editor markers are stale
            assert ui("set_tool", target="scale")["tool_id"] == "builtin.scale"
            other_tool = client.execute_script(
                "__result__ = [w.workspace.tools.from_space_view3d_mode('OBJECT', create=False).idname "
                "for w in bpy.context.window_manager.windows if w != bpy.context.window]"
            )
            assert other_tool["return_value"] == "['builtin.move']"
            assert ui("isolate_workspace")["completion"] == "already_isolated"

            # A second window requires explicit selection; changing the original
            # window's scene invalidates the existing attachment.
            client.execute_script("bpy.ops.wm.window_new()")
            assert ui("isolate_workspace")["success"] is False
            with pytest.raises(ValueError, match="--window"):
                manager.attach()
            manager.attach(window_id=info["window_id"])
            client.execute_script(
                "bpy.context.window.scene = bpy.data.scenes.new('Changed scene')"
            )
            with pytest.raises(RuntimeError, match="changed scene"):
                client.get_runtime_info()
            assert ui("inspect")["success"] is False
            manager.attach(window_id=info["window_id"])
            assert client.get_runtime_info()["scene_name"] == "Changed scene"
            assert client.list_scene_objects()["count"] == 0
            with pytest.raises(RuntimeError, match="Object not found"):
                client.delete_object("Cube")
            assert (
                client.create_object("CUBE", name="ScopedCube")["name"] == "ScopedCube"
            )
            assert client.list_scene_objects()["count"] == 1

            # File loading invalidates every old attachment even in the same PID.
            saved = tmp_path / "scratch.blend"
            client.save_blend_file(str(saved))
            # Several windows: open_blend_file cannot pick one, so it reports
            # that instead of guessing and the old pin keeps refusing.
            opened = client.open_blend_file(str(saved))
            assert opened["reattached"] is False
            assert "attach_blender_window" in opened["reattach_error"]
            with pytest.raises(AttachmentTargetChanged, match="loaded another file"):
                client.get_runtime_info()
            live = manager.instances()[0]
            manager.attach(window_id=live["windows"][0]["window_id"])
            assert client.get_runtime_info()["pid"] == process.pid
            client.execute_script(
                "bpy.ops.screen.area_split(direction='VERTICAL', factor=0.5)"
            )
            assert ui("move_cursor")["success"] is False
            layout = ui("inspect")
            views = [a for a in layout["areas"] if a["type"] == "VIEW_3D"]
            assert len(views) == 2
            ui(
                "move_cursor",
                area_id=views[0]["area_id"],
                position=[0.3, 0.7],
                agent_id="builder",
            )
            assert (
                ui(
                    "open_menu",
                    target="add",
                    area_id=views[0]["area_id"],
                    agent_id="builder",
                )["completion"]
                == "menu_requested"
            )
            assert next(
                m for m in ui("inspect")["agent_cursors"] if m["agent_id"] == "builder"
            )["position"] == [0.3, 0.7]
            manager.detach()
            assert process.poll() is None
            with pytest.raises(RuntimeError, match="No Blender window attached"):
                client.get_runtime_info()
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def test_recover_from_file_loads_context_loss_and_exit(
    tmp_path: Path, fake_fastmcp: Any
) -> None:
    """Issues #215, #216 and #217 through the MCP tools, one disposable window."""
    if os.environ.get("SIMUL_BLENDER_LIVE") != "1":
        pytest.skip("Set SIMUL_BLENDER_LIVE=1 to launch a disposable GUI Blender")
    binary = shutil.which("blender")
    if (
        sys.platform == "darwin"
        and Path("/Applications/Blender.app/Contents/MacOS/Blender").is_file()
    ):
        binary = "/Applications/Blender.app/Contents/MacOS/Blender"
    if binary is None:
        pytest.skip("Blender executable unavailable")
    addon = BlenderAttachments.build_addon(tmp_path / "addon.zip")
    with zipfile.ZipFile(addon) as archive:
        archive.extractall(tmp_path / "addons")
    startup = tmp_path / "startup.py"
    startup.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(tmp_path / 'addons')!r})\n"
        "import simul_blender_bridge\n"
        "simul_blender_bridge.register()\n"
    )
    settings = Settings(
        blender={
            "mode": "attached",
            "discovery_dir": str(tmp_path),
            "attachment_path": str(tmp_path / "attachment.json"),
        },
        security={"allowed_paths": [str(tmp_path)], "rate_limiting_enabled": False},
    )
    environment = {**os.environ, "SIMUL_BLENDER_DISCOVERY_DIR": str(tmp_path)}
    with (tmp_path / "blender.log").open("w") as log:
        process = subprocess.Popen(
            [binary, "--factory-startup", "--no-window-focus", "--python", str(startup)],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            manager = BlenderAttachments(settings)
            deadline = time.monotonic() + 45
            while not any(item["reachable"] for item in manager.instances()):
                if process.poll() is not None or time.monotonic() > deadline:
                    pytest.fail((tmp_path / "blender.log").read_text())
                time.sleep(0.2)
            server = SimulMCPServer(settings, backends={"blender"})
            tools = {t.name: t for t in server.mcp.tools}

            def call(name: str, **kwargs: Any) -> dict[str, Any]:
                return json.loads(asyncio.run(tools[name].func(**kwargs)).content[0].text)

            # Recovery never needs a shell: the MCP tool selects the only window.
            attached = call("attach_blender_window")
            assert attached["success"] is True and attached["pid"] == process.pid

            # #215: exporter and render from a script in the attached GUI.
            glb, still = tmp_path / "cube.glb", tmp_path / "render.png"
            exported = call(
                "execute_blender_script",
                script=(
                    "bpy.ops.mesh.primitive_cube_add()\n"
                    f"bpy.ops.export_scene.gltf(filepath={str(glb)!r}, export_format='GLB')\n"
                    "scene = bpy.context.scene\n"
                    "scene.render.resolution_x = scene.render.resolution_y = 32\n"
                    f"scene.render.filepath = {str(still)!r}\n"
                    "bpy.ops.render.render(write_still=True)\n"
                ),
            )
            assert exported["success"] is True, exported
            assert glb.stat().st_size > 0 and still.stat().st_size > 0

            # #215: loading a file inside a script replaces the window the call
            # was bound to; the error now carries the recipe, and the recipe works.
            lost = call(
                "execute_blender_script",
                script=(
                    "bpy.ops.wm.read_homefile(use_empty=True)\n"
                    f"bpy.ops.export_scene.gltf(filepath={str(glb)!r}, export_format='GLB')\n"
                ),
            )
            assert lost["success"] is False
            assert "'Context' object has no attribute 'active_object'" in lost["error"]
            assert "temp_override(window=win" in lost["error"]

            # #216: the strict pin refuses, names the new document and recovers in-session.
            refused = call("get_blender_info")
            assert refused["error_type"] == "AttachmentTargetChanged"
            new_document = refused["details"]["document_id"]
            assert new_document in refused["error"]
            assert "attach_blender_window" in refused["error"]
            recovered = call("attach_blender_window")
            assert recovered["success"] is True
            assert recovered["document_id"] == new_document
            glb.unlink()
            recipe = call(
                "execute_blender_script",
                script=(
                    "bpy.ops.wm.read_homefile(use_empty=True)\n"
                    "win = bpy.context.window_manager.windows[0]\n"
                    "area = next(a for a in win.screen.areas if a.type == 'VIEW_3D')\n"
                    "region = next(r for r in area.regions if r.type == 'WINDOW')\n"
                    "with bpy.context.temp_override(window=win, screen=win.screen, scene=win.scene,\n"
                    "        view_layer=win.view_layer, area=area, region=region):\n"
                    "    bpy.ops.mesh.primitive_cube_add()\n"
                    f"    bpy.ops.export_scene.gltf(filepath={str(glb)!r}, export_format='GLB')\n"
                ),
            )
            assert recipe["success"] is True, recipe
            assert glb.stat().st_size > 0
            assert call("attach_blender_window")["success"] is True

            # #216: open_blender_file follows the document it opened itself.
            saved = tmp_path / "saved.blend"
            assert call("save_blender_file", file_path=str(saved))["success"] is True
            opened = call("open_blender_file", file_path=str(saved))
            assert opened["success"] is True and opened["reattached"] is True, opened
            info = call("get_blender_info")
            assert info["success"] is True
            assert info["document_id"] == opened["document_id"]
            assert info["blend_file_path"] == str(saved)

            # #217: an exited process is named, typed and never a raw Errno 61.
            process.kill()
            process.wait(timeout=10)
            stale = call("get_blender_info")
            assert stale["error_type"] == "AttachmentStale"
            assert f"pid {process.pid}" in stale["error"]
            assert "not running" in stale["error"]
            with pytest.raises(AttachmentStale):
                BlenderConnection(settings).get_runtime_info()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
