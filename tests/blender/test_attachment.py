"""Target identity and transport regressions for existing-window attachment."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from simul_mcp.adapters.blender_connection import BlenderAttachments, BlenderConnection
from simul_mcp.blender_bridge.protocol import PROTOCOL_VERSION, BridgeFiles, BridgeWire
from simul_mcp.cli import blender_cli
from simul_mcp.cli.main import app
from simul_mcp.config import Settings
from simul_mcp.mcp.backends import backend_spec
from simul_mcp.mcp.server import SimulMCPServer


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        blender={
            "mode": "attached",
            "discovery_dir": str(tmp_path),
            "attachment_path": str(tmp_path / "attachment.json"),
            "connection_timeout": 0.5,
        }
    )


@pytest.fixture
def advertised(settings: Settings) -> tuple[dict[str, Any], dict[str, Any]]:
    endpoint = {
        "instance_id": "one",
        "protocol": PROTOCOL_VERSION,
        "host": "127.0.0.1",
        "port": 12345,
        "token": "secret",
    }
    BridgeFiles.write(
        Path(settings.blender.discovery_dir) / "instance-one.json", endpoint
    )
    info = {
        "instance_id": "one",
        "document_id": "doc",
        "pid": 123,
        "background": False,
        "blend_file_path": None,
        "is_dirty": True,
        "windows": [
            {
                "window_id": "window",
                "scene_id": "scene",
                "scene_name": "Unsaved",
                "workspace": "Layout",
            }
        ],
    }
    return endpoint, info


def test_attach_preserves_unsaved_identity_without_exposing_credentials(
    settings: Settings,
    advertised: tuple,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint, info = advertised
    monkeypatch.setattr(BlenderAttachments, "_hello", lambda self, target: info)
    result = BlenderAttachments(settings).attach()
    assert result["is_dirty"] and result["blend_file_path"] is None
    assert "token" not in json.dumps(result)
    stored = BridgeFiles.read(Path(settings.blender.attachment_path))
    assert stored["token"] == endpoint["token"]
    assert stored["target"]["scene_id"] == "scene"
    if os.name != "nt":
        assert Path(settings.blender.attachment_path).stat().st_mode & 0o777 == 0o600


def test_multiple_windows_require_selection(
    settings: Settings, advertised: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, info = advertised
    info["windows"].append(
        {
            "window_id": "second",
            "scene_id": "other",
            "scene_name": "Other",
            "workspace": "Layout",
        }
    )
    monkeypatch.setattr(BlenderAttachments, "_hello", lambda self, target: info)
    with pytest.raises(ValueError, match="--window"):
        BlenderAttachments(settings).attach()
    assert not Path(settings.blender.attachment_path).exists()
    assert (
        BlenderAttachments(settings).attach(window_id="second")["window"]["scene_name"]
        == "Other"
    )


def test_multiple_processes_require_selection(
    settings: Settings, advertised: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    endpoint, info = advertised
    BridgeFiles.write(
        Path(settings.blender.discovery_dir) / "instance-two.json",
        {**endpoint, "instance_id": "two"},
    )
    monkeypatch.setattr(
        BlenderAttachments,
        "_hello",
        lambda self, target: {**info, "instance_id": target["instance_id"]},
    )
    with pytest.raises(ValueError, match="--instance"):
        BlenderAttachments(settings).attach()
    assert (
        BlenderAttachments(settings).attach(instance_id="two")["instance_id"] == "two"
    )


def test_document_change_during_attach_does_not_publish(
    settings: Settings, advertised: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, info = advertised
    monkeypatch.setattr(
        BlenderAttachments,
        "_hello",
        Mock(side_effect=[info, {**info, "document_id": "changed"}]),
    )
    with pytest.raises(RuntimeError, match="changed while attaching"):
        BlenderAttachments(settings).attach()
    assert not Path(settings.blender.attachment_path).exists()


def test_no_bridge_never_creates_an_embedded_scene(settings: Settings) -> None:
    with pytest.raises(RuntimeError, match="No matching GUI"):
        BlenderAttachments(settings).attach()
    with pytest.raises(RuntimeError, match="No Blender window attached"):
        BlenderConnection(settings).create_object("CUBE")


def test_attached_tools_register_without_local_bpy(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, fake_fastmcp: Any
) -> None:
    monkeypatch.setattr("simul_mcp.mcp.backends.is_blender_available", lambda: False)
    server = SimulMCPServer(settings, backends={"blender"})
    assert "create_blender_object" in {tool.name for tool in server.mcp.tools}
    assert backend_spec("blender").adapter_factory(settings).is_available()


def test_timeout_never_retries_and_script_gate_precedes_transport(
    settings: Settings,
    advertised: tuple,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, info = advertised
    monkeypatch.setattr(BlenderAttachments, "_hello", lambda self, target: info)
    BlenderAttachments(settings).attach()
    request = Mock(side_effect=TimeoutError("timed out"))
    monkeypatch.setattr(BridgeWire, "request", request)
    with pytest.raises(TimeoutError, match="outcome is unknown"):
        BlenderConnection(settings).execute_script("pass", timeout=0.1)
    assert request.call_count == 1
    assert request.call_args.args[2] == 0.1
    disabled = settings.model_copy(
        update={
            "security": settings.security.model_copy(
                update={"allow_script_execution": False}
            )
        }
    )
    with pytest.raises(PermissionError, match="disabled"):
        BlenderConnection(disabled).execute_script("pass")
    assert request.call_count == 1


def test_proxy_sends_pinned_identity_and_resolved_policy(
    settings: Settings, advertised: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, info = advertised
    monkeypatch.setattr(BlenderAttachments, "_hello", lambda self, target: info)
    BlenderAttachments(settings).attach()
    request = Mock(return_value={"name": "Cube"})
    monkeypatch.setattr(BridgeWire, "request", request)
    assert BlenderConnection(settings).get_object_info("Cube") == {"name": "Cube"}
    payload = request.call_args.args[1]
    assert payload["target"]["document_id"] == "doc"
    assert payload["target"]["window_id"] == "window"
    assert all(Path(p).is_absolute() for p in payload["path_policy"]["allowed_paths"])
    with pytest.raises(AttributeError):
        BlenderConnection(settings)._get_object_or_raise("Cube")


def test_stale_records_are_reported_and_detach_only_removes_attachment(
    settings: Settings,
    advertised: tuple,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        BlenderAttachments, "_hello", Mock(side_effect=ConnectionRefusedError("closed"))
    )
    manager = BlenderAttachments(settings)
    assert manager.instances()[0]["reachable"] is False
    manager.detach()
    assert (Path(settings.blender.discovery_dir) / "instance-one.json").exists()


def test_wire_reassembles_fragmented_response() -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        endpoint = {
            "protocol": PROTOCOL_VERSION,
            "host": "127.0.0.1",
            "port": listener.getsockname()[1],
            "token": "secret",
        }

        def respond() -> None:
            with listener.accept()[0] as connection:
                payload = json.loads(connection.makefile("rb").readline())
                assert payload["token"] == "secret"
                encoded = BridgeWire.encode(
                    {"request_id": "id", "success": True, "result": {"value": "안녕"}}
                )
                connection.sendall(encoded[:17])
                time.sleep(0.02)
                connection.sendall(encoded[17:])

        worker = threading.Thread(target=respond)
        worker.start()
        try:
            assert BridgeWire.request(endpoint, {"request_id": "id"}, 1) == {
                "value": "안녕"
            }
        finally:
            worker.join(timeout=2)


@pytest.mark.parametrize(
    "override", [{"host": "0.0.0.0"}, {"port": None}, {"port": 0}, {"port": 65536}]
)
def test_invalid_endpoints_never_open_a_socket(
    override: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    connect = Mock()
    monkeypatch.setattr(socket, "create_connection", connect)
    endpoint = {
        "protocol": PROTOCOL_VERSION,
        "host": "127.0.0.1",
        "port": 12345,
        "token": "secret",
        **override,
    }
    with pytest.raises(ValueError):
        BridgeWire.request(endpoint, {"request_id": "id"}, 1)
    connect.assert_not_called()


def test_cli_missing_bridge_is_actionable(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(blender_cli, "get_settings", lambda: settings)
    result = CliRunner().invoke(app, ["--json", "blender", "attach"])
    assert result.exit_code == 1
    assert "Enable the Simul Blender Bridge" in json.loads(result.stdout)["error"]


def test_server_mode_flag_rejects_unknown_values() -> None:
    result = CliRunner().invoke(app, ["server", "--blender-mode", "guess"])
    assert result.exit_code == 1


def test_mcp_transform_forwards_validated_values_over_attachment(
    settings: Settings,
    advertised: tuple,
    monkeypatch: pytest.MonkeyPatch,
    fake_fastmcp: Any,
) -> None:
    _, info = advertised
    monkeypatch.setattr(BlenderAttachments, "_hello", lambda self, target: info)
    BlenderAttachments(settings).attach()

    def respond(endpoint: dict, payload: dict, timeout: float) -> dict:
        assert threading.current_thread() is not threading.main_thread()
        assert payload["method"] == "set_object_transform"
        assert payload["args"] == ("Cube", [1.0, 2.0, 3.0], None, None)
        return {
            "object_name": "Cube",
            "location": [1, 2, 3],
            "rotation_euler": [0, 0, 0],
            "scale": [1, 1, 1],
        }

    monkeypatch.setattr(BridgeWire, "request", respond)
    server = SimulMCPServer(settings, backends={"blender"})
    tool = next(t for t in server.mcp.tools if t.name == "set_blender_object_transform")
    result = asyncio.run(tool.func(object_name="Cube", location=[1, 2, 3]))
    assert json.loads(result.content[0].text)["success"] is True


def test_private_files_reject_symlinks_and_world_readable_credentials(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.json"
    BridgeFiles.write(target, {"token": "secret"})
    if os.name == "nt":
        pytest.skip("POSIX file permissions")
    target.chmod(0o644)
    with pytest.raises(PermissionError):
        BridgeFiles.read(target)
    target.chmod(0o600)
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(OSError):
        BridgeFiles.read(link)


def test_addon_bundles_the_private_file_helpers(tmp_path: Path) -> None:
    """The add-on imports BridgeFiles from utils/, so the ZIP must carry it."""
    import subprocess
    import sys
    import zipfile

    addon = BlenderAttachments.build_addon(tmp_path / "addon.zip")
    with zipfile.ZipFile(addon) as archive:
        assert "simul_blender_bridge/utils/private_files.py" in archive.namelist()
        archive.extractall(tmp_path / "addons")
    probe = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from simul_blender_bridge.blender_bridge.protocol import BridgeFiles; "
        "print(BridgeFiles.__module__)"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(tmp_path / "addons")],
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip() == "simul_blender_bridge.utils.private_files"
