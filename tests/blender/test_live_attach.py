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

from simul_mcp.adapters.blender_connection import BlenderAttachments, BlenderConnection
from simul_mcp.blender_bridge.protocol import BridgeFiles, BridgeWire
from simul_mcp.config import Settings
from simul_mcp.mcp.server import SimulMCPServer

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
                width=64, height=64, use_render_fallback=True
            )
            assert base64.b64decode(capture["image_base64"]).startswith(b"\xff\xd8")

            # A second window requires explicit selection; changing the original
            # window's scene invalidates the existing attachment.
            client.execute_script("bpy.ops.wm.window_new()")
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
            client.open_blend_file(str(saved))
            with pytest.raises(RuntimeError, match="loaded another file"):
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
            assert (
                ui("open_menu", target="add", area_id=views[0]["area_id"])["completion"]
                == "menu_requested"
            )
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
