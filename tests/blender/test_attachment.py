"""Target identity and transport regressions for existing-window attachment."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import socket
import threading
import time
from pathlib import Path
from typing import Any, Optional
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from simul.adapters.blender_connection import BlenderAttachments, BlenderConnection
from simul.adapters.blender_runtime import BlenderRuntimeSession
from simul.blender_bridge.protocol import PROTOCOL_VERSION, BridgeFiles, BridgeWire
from simul.cli import blender_cli
from simul.cli.main import app
from simul.config import Settings
from simul.mcp.backends import backend_spec
from simul.mcp.server import SimulMCPServer


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
    # Nothing to connect to: a ConnectionError, not a RuntimeError.
    with pytest.raises(ConnectionError, match="No Blender window attached"):
        BlenderConnection(settings).create_object("CUBE")


def test_attached_tools_register_without_local_bpy(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, fake_fastmcp: Any
) -> None:
    monkeypatch.setattr("simul.mcp.backends.is_blender_available", lambda: False)
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
        BlenderConnection(settings).get_frame()
    assert request.call_count == 1
    # B1-23: a script that may still run reports timed_out, as in embedded mode.
    timed_out = BlenderConnection(settings).execute_script("pass", timeout=0.1)
    assert timed_out["timed_out"] is True
    assert timed_out["error"].startswith("Script did not finish within 0.1s.")
    assert timed_out["details"] == {"executed": None}
    assert request.call_count == 2
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
    assert request.call_count == 2


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
    assert result.exit_code == 2  # a bad invocation, as docs/cli.md says


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
        # The add-on binds args and kwargs to the operation's signature, so the
        # value each parameter receives is what reaches Blender.
        bound = inspect.signature(BlenderRuntimeSession.set_object_transform).bind(
            None, *payload["args"], **payload["kwargs"]
        )
        assert dict(list(bound.arguments.items())[1:]) == {
            "object_name": "Cube",
            "location": [1.0, 2.0, 3.0],
            "rotation_euler": None,
            "scale": None,
        }
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


def _exited_pid() -> int:
    import subprocess
    import sys

    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


def _attach_with_pid(
    settings: Settings, advertised: tuple, monkeypatch: pytest.MonkeyPatch, pid: int
) -> None:
    endpoint, info = advertised
    BridgeFiles.write(
        Path(settings.blender.discovery_dir) / "instance-one.json", {**endpoint, "pid": pid}
    )
    monkeypatch.setattr(BlenderAttachments, "_hello", lambda self, target: info)
    BlenderAttachments(settings).attach()


def test_dead_process_is_reported_as_stale_without_connecting(
    settings: Settings, advertised: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #217: a dead pid names itself and the recovery commands, not Errno 61."""
    from simul.adapters.blender_connection import AttachmentStale

    pid = _exited_pid()
    _attach_with_pid(settings, advertised, monkeypatch, pid)
    request = Mock()
    monkeypatch.setattr(BridgeWire, "request", request)
    with pytest.raises(AttachmentStale) as raised:
        BlenderConnection(settings).get_runtime_info()
    request.assert_not_called()
    assert f"pid {pid}" in str(raised.value) and "not running" in str(raised.value)
    assert "simul blender attach" in str(raised.value)
    assert "attach_blender_window" in str(raised.value)
    assert raised.value.details["process_alive"] is False

    monkeypatch.setattr(blender_cli, "get_settings", lambda: settings)
    result = CliRunner().invoke(app, ["--json", "blender", "status"])
    assert result.exit_code == 1
    envelope = json.loads(result.stdout)
    assert envelope["error_type"] == "AttachmentStale"
    assert envelope["details"]["pid"] == pid


def test_refused_bridge_of_a_live_process_is_stale(
    settings: Settings, advertised: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    from simul.adapters.blender_connection import AttachmentStale

    _attach_with_pid(settings, advertised, monkeypatch, os.getpid())
    monkeypatch.setattr(
        BridgeWire, "request", Mock(side_effect=ConnectionRefusedError(61, "Connection refused"))
    )
    with pytest.raises(AttachmentStale, match="not accepting bridge connections") as raised:
        BlenderConnection(settings).create_object("CUBE")
    assert raised.value.details["process_alive"] is True


def test_mcp_tools_surface_the_stale_error_type(
    settings: Settings,
    advertised: tuple,
    monkeypatch: pytest.MonkeyPatch,
    fake_fastmcp: Any,
) -> None:
    _attach_with_pid(settings, advertised, monkeypatch, _exited_pid())
    server = SimulMCPServer(settings, backends={"blender"})
    tool = next(t for t in server.mcp.tools if t.name == "get_blender_info")
    payload = json.loads(asyncio.run(tool.func()).content[0].text)
    assert payload["success"] is False
    assert payload["error_type"] == "AttachmentStale"
    assert "not running" in payload["error"]


@pytest.mark.parametrize(
    ("remote_type", "message"),
    [
        ("AttachmentTargetChanged", "Blender loaded another file (attached document doc, now new)"),
        # An add-on built before the typed error still gets the structured path.
        ("ValueError", "ValueError: Blender loaded another file; run simul blender attach again"),
    ],
)
def test_document_change_names_the_new_document_and_recovery(
    settings: Settings,
    advertised: tuple,
    monkeypatch: pytest.MonkeyPatch,
    remote_type: str,
    message: str,
) -> None:
    """Issue #216: the pin stays strict but the error says what changed and how to re-attach."""
    from simul.adapters.blender_connection import AttachmentTargetChanged
    from simul.blender_bridge.protocol import BridgeRemoteError

    _, info = advertised
    monkeypatch.setattr(BlenderAttachments, "_hello", lambda self, target: info)
    BlenderAttachments(settings).attach()
    fresh = {
        **info,
        "document_id": "new",
        "windows": [{**info["windows"][0], "window_id": "w2", "scene_id": "s2"}],
    }
    monkeypatch.setattr(BlenderAttachments, "_hello", lambda self, target: fresh)
    monkeypatch.setattr(
        BridgeWire, "request", Mock(side_effect=BridgeRemoteError(message, remote_type, {}))
    )
    with pytest.raises(AttachmentTargetChanged) as raised:
        BlenderConnection(settings).get_runtime_info()
    assert "now new" in str(raised.value)
    assert "attach_blender_window" in str(raised.value)
    assert raised.value.details["document_id"] == "new"
    assert raised.value.details["attached_document_id"] == "doc"
    assert raised.value.details["windows"][0]["window_id"] == "w2"
    # Strict pin: the attachment was not silently moved.
    stored = BridgeFiles.read(Path(settings.blender.attachment_path))
    assert stored["target"]["document_id"] == "doc"


def test_open_blend_file_follows_its_own_document_change(
    settings: Settings, advertised: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, info = advertised
    monkeypatch.setattr(BlenderAttachments, "_hello", lambda self, target: info)
    BlenderAttachments(settings).attach()
    new_window = {**info["windows"][0], "window_id": "w2", "scene_id": "s2", "scene_name": "Loaded"}
    fresh = {**info, "document_id": "new", "windows": [new_window]}
    monkeypatch.setattr(BlenderAttachments, "_hello", lambda self, target: fresh)
    monkeypatch.setattr(
        BridgeWire, "request", Mock(return_value={"file_path": "/x.blend", "object_count": 2})
    )
    result = BlenderConnection(settings).open_blend_file("/x.blend")
    assert result["reattached"] is True
    assert result["document_id"] == "new" and result["window_id"] == "w2"
    stored = BridgeFiles.read(Path(settings.blender.attachment_path))
    assert stored["target"] == {"document_id": "new", **new_window}


def test_open_blend_file_reports_an_ambiguous_follow_instead_of_guessing(
    settings: Settings, advertised: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, info = advertised
    monkeypatch.setattr(BlenderAttachments, "_hello", lambda self, target: info)
    BlenderAttachments(settings).attach()
    windows = [
        {**info["windows"][0], "window_id": "a", "scene_id": "s"},
        {**info["windows"][0], "window_id": "b", "scene_id": "s"},
    ]
    monkeypatch.setattr(
        BlenderAttachments, "_hello", lambda self, target: {**info, "document_id": "new", "windows": windows}
    )
    monkeypatch.setattr(
        BridgeWire, "request", Mock(return_value={"file_path": "/x.blend", "object_count": 2})
    )
    result = BlenderConnection(settings).open_blend_file("/x.blend")
    assert result["reattached"] is False
    assert "attach_blender_window" in result["reattach_error"]
    stored = BridgeFiles.read(Path(settings.blender.attachment_path))
    assert stored["target"]["document_id"] == "doc"


def test_attach_blender_window_tool_attaches_and_lists_candidates_when_ambiguous(
    settings: Settings,
    advertised: tuple,
    monkeypatch: pytest.MonkeyPatch,
    fake_fastmcp: Any,
) -> None:
    _, info = advertised
    monkeypatch.setattr(BlenderAttachments, "_hello", lambda self, target: info)
    server = SimulMCPServer(settings, backends={"blender"})
    tool = next(t for t in server.mcp.tools if t.name == "attach_blender_window")
    payload = json.loads(asyncio.run(tool.func()).content[0].text)
    assert payload["success"] is True, payload
    assert payload["window"]["window_id"] == "window"
    assert "token" not in json.dumps(payload)
    assert BridgeFiles.read(Path(settings.blender.attachment_path))["target"]["window_id"] == "window"

    info["windows"].append({**info["windows"][0], "window_id": "second"})
    payload = json.loads(asyncio.run(tool.func()).content[0].text)
    assert payload["success"] is False
    # Raised as ValueError by the attachment; ValidationError in the shared taxonomy.
    assert payload["error_type"] == "ValidationError"
    assert payload["details"]["exception_type"] == "ValueError"
    assert {w["window_id"] for w in payload["details"]["instances"][0]["windows"]} == {"window", "second"}
    payload = json.loads(asyncio.run(tool.func(window_id="second")).content[0].text)
    assert payload["success"] is True and payload["window"]["window_id"] == "second"


def test_attach_blender_window_with_no_bridge_is_a_connection_error(
    settings: Settings, fake_fastmcp: Any
) -> None:
    """A bare RuntimeError read as EngineError; nothing answered, so it is a ConnectionError."""
    server = SimulMCPServer(settings, backends={"blender"})
    tool = next(t for t in server.mcp.tools if t.name == "attach_blender_window")

    payload = json.loads(asyncio.run(tool.func()).content[0].text)

    assert payload["success"] is False
    assert payload["error_type"] == "ConnectionError"
    assert payload["details"] == {"instances": []}
    assert "No matching GUI Blender bridge" in payload["error"]


def test_attach_blender_window_tool_requires_attached_mode(
    monkeypatch: pytest.MonkeyPatch, fake_fastmcp: Any, tmp_path: Path
) -> None:
    from contextlib import nullcontext

    from tests.fakes import AvailableAdapter

    monkeypatch.setattr("simul.mcp.backends.is_blender_available", lambda: True)
    monkeypatch.setattr("simul.mcp.backends.BlenderRuntimeAdapter", AvailableAdapter)
    monkeypatch.setattr(AvailableAdapter, "create_session", lambda self: nullcontext(object()))
    embedded = Settings(blender={"mode": "embedded", "attachment_path": str(tmp_path / "a.json")})
    server = SimulMCPServer(embedded, backends={"blender"})
    tool = next(t for t in server.mcp.tools if t.name == "attach_blender_window")
    payload = json.loads(asyncio.run(tool.func()).content[0].text)
    assert payload["success"] is False and "attached" in payload["error"]
    assert not (tmp_path / "a.json").exists()


def test_script_context_errors_get_the_override_recipe(
    settings: Settings, advertised: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #215: an older add-on's bare AttributeError still gets the recipe client-side."""
    _, info = advertised
    monkeypatch.setattr(BlenderAttachments, "_hello", lambda self, target: info)
    BlenderAttachments(settings).attach()
    error = "AttributeError: 'Context' object has no attribute 'active_object'"
    monkeypatch.setattr(
        BridgeWire,
        "request",
        Mock(return_value={"output": None, "return_value": None, "duration_seconds": 0.1, "error": error}),
    )
    result = BlenderConnection(settings).execute_script("import bpy")
    assert result["error"].startswith(error)
    assert "temp_override(window=win, screen=win.screen" in result["error"]


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


def test_addon_bundles_the_error_message_helper(tmp_path: Path) -> None:
    """The bridge renders exception notes with utils/errors.py, so the ZIP must carry it."""
    import subprocess
    import sys
    import zipfile

    addon = BlenderAttachments.build_addon(tmp_path / "addon.zip")
    with zipfile.ZipFile(addon) as archive:
        assert "simul_blender_bridge/utils/errors.py" in archive.namelist()
        archive.extractall(tmp_path / "addons")
    probe = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from simul_blender_bridge.utils.errors import exception_message; "
        "error = RuntimeError('poll() failed'); error.add_note('Failed to add rigid body'); "
        "print(exception_message(error))"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(tmp_path / "addons")],
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip() == "Failed to add rigid body: poll() failed"


def test_addon_bundles_the_shared_import_reference_readers(tmp_path: Path) -> None:
    """blender_runtime's import sandbox reads references with utils/asset_references.py, shared with Unreal.

    An interpreter without simul (and without bpy) imports the bundled
    runtime, so the readers come from the add-on's own copy and refuse an
    OBJ whose MTL names a file outside the sandbox.
    """
    import subprocess
    import sys
    import zipfile

    addon = BlenderAttachments.build_addon(tmp_path / "addon.zip")
    with zipfile.ZipFile(addon) as archive:
        assert "simul_blender_bridge/utils/asset_references.py" in archive.namelist()
        archive.extractall(tmp_path / "addons")
    sbx, outside = tmp_path / "sbx", tmp_path / "outside"
    sbx.mkdir()
    outside.mkdir()
    (outside / "secret.png").write_bytes(b"\x89PNG")
    (sbx / "cube.obj").write_text("mtllib cube.mtl\n")
    (sbx / "cube.mtl").write_text("newmtl M\nmap_Kd ../outside/secret.png\n")
    probe = (
        "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
        "from simul_blender_bridge.adapters import blender_runtime as runtime; "
        "inside = lambda path: Path(path).resolve().is_relative_to(Path(sys.argv[2]).resolve()); "
        "reads = runtime._obj_reads(Path(sys.argv[2]) / 'cube.obj'); "
        "print(runtime.outside_references.__module__); "
        "print(runtime.outside_references(reads, type('P', (), {'is_allowed': staticmethod(inside), "
        "'resolve': staticmethod(lambda path: Path(path).resolve())})()))"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(tmp_path / "addons"), str(sbx)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.splitlines() == [
        "simul_blender_bridge.utils.asset_references",
        str([str((outside / "secret.png").resolve())]),
    ]


def test_addon_bundles_the_regex_guard(tmp_path: Path) -> None:
    """blender_runtime imports utils/regex_guard.py, so the ZIP must carry it.

    An interpreter without simul (and without bpy) imports the bundled
    runtime, so its guard comes from the add-on's own copy.
    """
    import subprocess
    import sys
    import zipfile

    addon = BlenderAttachments.build_addon(tmp_path / "addon.zip")
    with zipfile.ZipFile(addon) as archive:
        assert "simul_blender_bridge/utils/regex_guard.py" in archive.namelist()
        archive.extractall(tmp_path / "addons")
    probe = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from simul_blender_bridge.adapters import blender_runtime; "
        "search = blender_runtime.name_searcher; "
        "from simul_blender_bridge.utils.regex_guard import exponential_regex_construct as guard; "
        "print(search.__module__); print(guard(r'^(\\w+)+_x$')); "
        "print(guard('.*.*.*x').split(',')[0]); print(search('.*.*.*x')('a' * 255))"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(tmp_path / "addons")],
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.splitlines() == [
        "simul_blender_bridge.utils.regex_guard",
        "a quantifier inside a repeated group, as in (\\w+)+",
        "3 quantifiers that can divide the same characters between them",
        "False",
    ]


# ---------------------------------------------------------------------------
# Whether an unanswered request ran, and retries only when it did not (B1-09)
# ---------------------------------------------------------------------------


class _FakeBridge:
    """A loopback listener that answers each connection with a scripted behaviour.

    Each behaviour receives the accepted connection and the request it read
    (None when it reads nothing), and runs on a background thread.
    """

    def __init__(self, *behaviours: Any) -> None:
        self.behaviours = list(behaviours)
        self.requests: list = []
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen()
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def endpoint(self, acknowledging: bool = True) -> dict:
        return {
            "protocol": PROTOCOL_VERSION,
            "host": "127.0.0.1",
            "port": self.listener.getsockname()[1],
            "token": "secret",
            **({"capabilities": ["start_ack"]} if acknowledging else {}),
        }

    def _serve(self) -> None:
        for behaviour in self.behaviours:
            connection, _ = self.listener.accept()
            with connection:
                if behaviour is _close_unread:
                    continue
                with connection.makefile("rb") as reader:
                    request = json.loads(reader.readline())
                self.requests.append(request)
                behaviour(connection, request)

    def close(self) -> None:
        self.thread.join(timeout=5)
        self.listener.close()


def _close_unread(connection: socket.socket, request: Any) -> None:
    """What a bridge holding too many connections does: close before reading."""


def _ack(connection: socket.socket, request: dict) -> None:
    connection.sendall(
        BridgeWire.encode({"request_id": request["request_id"], "started": True})
    )


def _answer(connection: socket.socket, request: dict) -> None:
    _ack(connection, request)
    connection.sendall(
        BridgeWire.encode(
            {"request_id": request["request_id"], "success": True, "result": {"ok": 1}}
        )
    )


def _close_after_reading(connection: socket.socket, request: dict) -> None:
    pass


def _ack_then_close(connection: socket.socket, request: dict) -> None:
    _ack(connection, request)


def _silent(connection: socket.socket, request: dict) -> None:
    time.sleep(1.2)


def _ack_then_silent(connection: socket.socket, request: dict) -> None:
    _ack(connection, request)
    time.sleep(1.2)


def test_a_connection_dropped_before_the_start_ack_is_sent_again() -> None:
    """The add-on runs nothing it has not acknowledged, so this request did not run."""
    bridge = _FakeBridge(_close_unread, _answer)
    try:
        result = BridgeWire.request(bridge.endpoint(), {"request_id": "r"}, 5)
    finally:
        bridge.close()
    assert result == {"ok": 1}
    assert [request["ack"] for request in bridge.requests] == [True]


@pytest.mark.parametrize(
    ("behaviour", "acknowledging", "executed"),
    [
        (_ack_then_close, True, True),
        # An add-on without the acknowledgement may have run it.
        (_close_after_reading, False, None),
    ],
)
def test_a_connection_dropped_after_a_possible_start_is_not_sent_again(
    behaviour: Any, acknowledging: bool, executed: Optional[bool]
) -> None:
    from simul.blender_bridge.protocol import BridgeConnectionLost

    bridge = _FakeBridge(behaviour)
    try:
        with pytest.raises(BridgeConnectionLost) as lost:
            BridgeWire.request(bridge.endpoint(acknowledging), {"request_id": "r"}, 5)
    finally:
        bridge.close()
    assert lost.value.executed is executed
    assert len(bridge.requests) == 1
    assert ("ack" in bridge.requests[0]) is acknowledging


@pytest.mark.parametrize(
    ("behaviour", "acknowledging", "executed"),
    [(_silent, True, False), (_ack_then_silent, True, True), (_silent, False, None)],
    ids=["not-started", "started", "without-acknowledgements"],
)
def test_a_timeout_says_whether_the_request_ran(
    behaviour: Any, acknowledging: bool, executed: Optional[bool]
) -> None:
    from simul.blender_bridge.protocol import BridgeTimeout

    bridge = _FakeBridge(behaviour)
    try:
        with pytest.raises(BridgeTimeout) as timed_out:
            BridgeWire.request(bridge.endpoint(acknowledging), {"request_id": "r"}, 0.2)
    finally:
        bridge.close()
    assert timed_out.value.executed is executed
    assert len(bridge.requests) == 1


def test_a_request_over_the_cap_is_refused_before_sending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from simul.blender_bridge import protocol

    connect = Mock()
    monkeypatch.setattr(socket, "create_connection", connect)
    monkeypatch.setattr(protocol, "MAX_MESSAGE_BYTES", 100)
    endpoint = {"protocol": PROTOCOL_VERSION, "host": "127.0.0.1", "port": 1, "token": "t"}
    with pytest.raises(protocol.MessageTooLarge, match="per call"):
        BridgeWire.request(endpoint, {"request_id": "r", "data": "x" * 200}, 1)
    connect.assert_not_called()


def _attached(settings: Settings, advertised: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    _, info = advertised
    monkeypatch.setattr(BlenderAttachments, "_hello", lambda self, target: info)
    BlenderAttachments(settings).attach()


@pytest.mark.parametrize(
    ("executed", "says"),
    [
        (False, "did not start get_frame within the 0.5s timeout"),
        (True, "started get_frame but did not finish"),
        (None, "outcome is unknown"),
    ],
)
def test_an_unanswered_call_says_whether_it_ran(
    settings: Settings,
    advertised: tuple,
    monkeypatch: pytest.MonkeyPatch,
    executed: Optional[bool],
    says: str,
) -> None:
    """B1-09: the error was a raw ConnectionResetError that said nothing about it."""
    from simul.blender_bridge.protocol import BridgeTimeout

    _attached(settings, advertised, monkeypatch)
    monkeypatch.setattr(
        BridgeWire, "request", Mock(side_effect=BridgeTimeout("deadline", executed))
    )
    with pytest.raises(TimeoutError, match=says) as raised:
        BlenderConnection(settings).get_frame()
    assert raised.value.details == {  # type: ignore[attr-defined]
        "executed": executed,
        "method": "get_frame",
        "timeout_seconds": 0.5,
    }
    if executed is False:
        assert "retrying is safe" in str(raised.value)
    else:
        assert "inspect Blender before retrying" in str(raised.value).replace("Inspect", "inspect")


@pytest.mark.parametrize(
    ("executed", "says"),
    [
        (False, "was not executed, so retrying is safe"),
        (True, "Blender started create_object, but the connection closed"),
        (None, "may or may not have run"),
    ],
)
def test_a_lost_connection_says_whether_the_call_ran(
    settings: Settings,
    advertised: tuple,
    monkeypatch: pytest.MonkeyPatch,
    fake_fastmcp: Any,
    executed: Optional[bool],
    says: str,
) -> None:
    from simul.blender_bridge.protocol import BridgeConnectionLost

    _attached(settings, advertised, monkeypatch)
    monkeypatch.setattr(
        BridgeWire,
        "request",
        Mock(side_effect=BridgeConnectionLost("[Errno 54] Connection reset by peer", executed)),
    )
    server = SimulMCPServer(settings, backends={"blender"})
    tool = next(t for t in server.mcp.tools if t.name == "create_blender_object")
    payload = json.loads(asyncio.run(tool.func(object_type="CUBE")).content[0].text)

    assert payload["success"] is False
    assert payload["error_type"] == "ConnectionError"
    assert says in payload["error"]
    assert payload["details"]["executed"] is executed


@pytest.mark.parametrize("executed", [True, None])
def test_a_script_past_its_timeout_reports_timed_out_as_in_embedded_mode(
    settings: Settings,
    advertised: tuple,
    monkeypatch: pytest.MonkeyPatch,
    fake_fastmcp: Any,
    executed: Optional[bool],
) -> None:
    """B1-23: attached mode answered TimeoutError and never set timed_out."""
    from simul.blender_bridge.protocol import BridgeTimeout

    _attached(settings, advertised, monkeypatch)
    monkeypatch.setattr(
        BridgeWire, "request", Mock(side_effect=BridgeTimeout("deadline", executed))
    )
    server = SimulMCPServer(settings, backends={"blender"})
    tool = next(t for t in server.mcp.tools if t.name == "execute_blender_script")
    payload = json.loads(
        asyncio.run(tool.func(script="import time; time.sleep(9)", timeout=0.2)).content[0].text
    )

    assert payload["success"] is False
    assert payload["timed_out"] is True
    assert payload["error"].startswith("Script did not finish within 0.2s.")
    assert payload["error_type"] == "TimeoutError"
    assert payload["duration_seconds"] == 0.2
    assert payload["details"] == {"executed": executed}


def test_a_script_that_never_started_is_not_reported_as_timed_out(
    settings: Settings, advertised: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    from simul.blender_bridge.protocol import BridgeTimeout

    _attached(settings, advertised, monkeypatch)
    monkeypatch.setattr(
        BridgeWire, "request", Mock(side_effect=BridgeTimeout("deadline", False))
    )
    with pytest.raises(TimeoutError, match="was not executed") as raised:
        BlenderConnection(settings).execute_script("pass", timeout=0.2)
    assert raised.value.details["executed"] is False  # type: ignore[attr-defined]


def test_attaching_an_unknown_instance_names_the_known_ones(
    settings: Settings, advertised: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B1-22: the error said to enable the add-on, which was enabled."""
    _, info = advertised
    monkeypatch.setattr(BlenderAttachments, "_hello", lambda self, target: info)
    with pytest.raises(RuntimeError) as raised:
        BlenderAttachments(settings).attach(instance_id="deadbeef")
    message = str(raised.value)
    assert "No Blender instance has the id 'deadbeef'" in message
    assert "known instances: one (pid 123)" in message
    assert "Enable the Simul Blender Bridge" not in message


def test_attaching_an_instance_that_stopped_answering_says_so(
    settings: Settings, advertised: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        BlenderAttachments, "_hello", Mock(side_effect=ConnectionRefusedError("refused"))
    )
    with pytest.raises(RuntimeError, match="instance one does not answer"):
        BlenderAttachments(settings).attach(instance_id="one")


# ---------------------------------------------------------------------------
# Review of #333: framing, busy refusals, expiry, attach error types
# ---------------------------------------------------------------------------


def _busy(connection: socket.socket, request: dict) -> None:
    connection.sendall(
        BridgeWire.encode(
            {
                "request_id": None,
                "success": False,
                "error": "InstanceBusy: holding too much; not executed",
                "error_type": "InstanceBusy",
                "details": {"code": "bridge_busy", "executed": False},
            }
        )
    )


def _late_ack_then_answer(connection: socket.socket, request: dict) -> None:
    time.sleep(0.3)
    _answer(connection, request)


def test_a_request_starts_with_its_protocol_and_token() -> None:
    """The add-on authenticates on the first bytes, before reading the rest."""
    from simul.blender_bridge.protocol import request_prefix

    bridge = _FakeBridge(_answer)
    raw: list = []
    original = BridgeWire.encode

    def recorded(payload: dict) -> bytes:
        encoded = original(payload)
        raw.append(encoded)
        return encoded

    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(BridgeWire, "encode", staticmethod(recorded))
            BridgeWire.request(bridge.endpoint(), {"request_id": "r", "method": "hello"}, 5)
    finally:
        bridge.close()
    assert raw[0].startswith(request_prefix("secret"))


def test_a_busy_refusal_is_sent_again_while_time_remains() -> None:
    """The bridge refuses as busy before reading, so the request did not run."""
    bridge = _FakeBridge(_busy, _answer)
    try:
        result = BridgeWire.request(bridge.endpoint(), {"request_id": "r"}, 5)
    finally:
        bridge.close()
    assert result == {"ok": 1}
    assert len(bridge.requests) == 2


def test_a_reply_right_after_a_late_acknowledgement_is_kept() -> None:
    """Review: the grace read dropped a reply that followed the acknowledgement."""
    bridge = _FakeBridge(_late_ack_then_answer)
    try:
        result = BridgeWire.request(bridge.endpoint(), {"request_id": "r"}, 0.2)
    finally:
        bridge.close()
    assert result == {"ok": 1}


def test_a_client_stalled_past_the_grace_still_sees_the_acknowledgement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-review: the grace ended at a fixed time, so a stalled client never looked.

    Another thread holding the GIL (json.dumps of a large mesh) kept the
    client from reading for longer than the grace; the acknowledgement had
    come meanwhile and the call ran, but it was reported as not executed.
    """
    from simul.blender_bridge import protocol

    original = protocol._Lines.next
    stalled: list = []

    def stalls_once(self: Any, deadline: float) -> Optional[bytes]:
        line = original(self, deadline)
        if line is None and not stalled:
            stalled.append(True)
            time.sleep(protocol.ACK_GRACE_SECONDS + 0.4)
        return line

    monkeypatch.setattr(protocol._Lines, "next", stalls_once)
    bridge = _FakeBridge(_late_ack_then_answer)
    try:
        result = BridgeWire.request(bridge.endpoint(), {"request_id": "r"}, 0.2)
    finally:
        bridge.close()
    assert stalled
    assert result == {"ok": 1}


def test_a_line_that_arrived_is_read_even_past_the_deadline() -> None:
    from simul.blender_bridge.protocol import _Lines

    ours, theirs = socket.socketpair()
    with ours, theirs:
        theirs.sendall(b'{"request_id": "r", "started": true}\n')
        time.sleep(0.05)
        lines = _Lines(ours)
        assert lines.next(time.monotonic() - 1) == b'{"request_id": "r", "started": true}'
        assert lines.next(time.monotonic() - 1) is None


def test_a_request_expired_in_the_queue_says_it_did_not_run(
    settings: Settings, advertised: tuple, monkeypatch: pytest.MonkeyPatch, fake_fastmcp: Any
) -> None:
    """Review: its deadline passed while it waited, and the reply had details null."""
    from simul.blender_bridge.protocol import BridgeRemoteError

    _attached(settings, advertised, monkeypatch)
    monkeypatch.setattr(
        BridgeWire,
        "request",
        Mock(
            side_effect=BridgeRemoteError(
                "TimeoutError: Request expired before execution; no operation was performed",
                "TimeoutError",
                {"executed": False},
            )
        ),
    )
    server = SimulMCPServer(settings, backends={"blender"})
    tool = next(t for t in server.mcp.tools if t.name == "execute_blender_script")
    payload = json.loads(
        asyncio.run(tool.func(script="bpy.data.objects.new('G', None)", timeout=1)).content[0].text
    )

    assert payload["error_type"] == "TimeoutError"
    assert payload["details"]["executed"] is False
    assert "not executed" in payload["error"]


@pytest.mark.parametrize(
    "sent",
    [
        "Cannot parent 'P' to 'C': 'C' is a descendant of 'P'",
        # An add-on up to 0.1.12 put the class in front of the message.
        "ValidationError: Cannot parent 'P' to 'C': 'C' is a descendant of 'P'",
    ],
)
def test_a_refusal_inside_blender_names_its_class_once(
    settings: Settings,
    advertised: tuple,
    monkeypatch: pytest.MonkeyPatch,
    fake_fastmcp: Any,
    caplog: pytest.LogCaptureFixture,
    sent: str,
) -> None:
    """R2-BLA-02/03: the error read "ValidationError: …" and the log "ValueError: ValueError: …"."""
    from simul.blender_bridge.protocol import BridgeRemoteError

    _attached(settings, advertised, monkeypatch)
    details = {"code": "parent_loop", "field": "parent_name"}
    monkeypatch.setattr(
        BridgeWire,
        "request",
        Mock(side_effect=BridgeRemoteError(sent, "ValidationError", dict(details))),
    )
    server = SimulMCPServer(settings, backends={"blender"})
    tool = next(t for t in server.mcp.tools if t.name == "set_blender_object_parent")
    with caplog.at_level("ERROR"):
        payload = json.loads(
            asyncio.run(tool.func(child_name="P", parent_name="C")).content[0].text
        )

    message = "Cannot parent 'P' to 'C': 'C' is a descendant of 'P'"
    assert payload["error"] == message
    assert payload["error_type"] == "ValidationError"
    assert payload["details"]["code"] == "parent_loop"
    logged = [r.getMessage() for r in caplog.records if "set_blender_object_parent" in r.getMessage()]
    assert logged == [f"Tool set_blender_object_parent failed: ValidationError: {message}"]


@pytest.mark.parametrize(("attempts", "says"), [(1, "before starting get_frame ("), (4, "on each of 4 attempts")])
def test_a_lost_connection_counts_only_the_retries_it_made(
    settings: Settings, advertised: tuple, monkeypatch: pytest.MonkeyPatch, attempts: int, says: str
) -> None:
    """Review: the message said "again on each retry" when none was sent."""
    from simul.blender_bridge.protocol import BridgeConnectionLost

    _attached(settings, advertised, monkeypatch)
    lost = BridgeConnectionLost("reset", False, attempts)
    monkeypatch.setattr(BridgeWire, "request", Mock(side_effect=lost))
    with pytest.raises(ConnectionError, match=re.escape(says)) as raised:
        BlenderConnection(settings).get_frame()
    assert raised.value.details["attempts"] == attempts  # type: ignore[attr-defined]
    if attempts == 1:
        assert "retry" not in str(raised.value).split("so retrying is safe")[0]


def test_a_timed_out_script_reports_how_long_it_ran(
    settings: Settings, advertised: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review: duration_seconds counted the time the script waited in the queue."""
    from simul.blender_bridge.protocol import BridgeTimeout

    _attached(settings, advertised, monkeypatch)
    monkeypatch.setattr(
        BridgeWire, "request", Mock(side_effect=BridgeTimeout("deadline", True, running_seconds=0.125))
    )
    result = BlenderConnection(settings).execute_script("pass", timeout=0.5)
    assert result["timed_out"] is True
    assert result["duration_seconds"] == 0.125


def test_an_oversized_request_says_it_was_not_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    from simul.blender_bridge import protocol

    monkeypatch.setattr(protocol, "MAX_MESSAGE_BYTES", 100)
    endpoint = {"protocol": PROTOCOL_VERSION, "host": "127.0.0.1", "port": 1, "token": "t"}
    with pytest.raises(protocol.MessageTooLarge) as raised:
        BridgeWire.request(endpoint, {"request_id": "r", "data": "x" * 200}, 1)
    assert raised.value.details["code"] == "request_too_large"
    assert raised.value.details["executed"] is False
    assert raised.value.details["limit_bytes"] == 100


@pytest.mark.parametrize(
    ("hello", "instance_id", "error_type", "says"),
    [
        (None, "deadbeef", "NotFoundError", "No Blender instance has the id 'deadbeef'"),
        ("busy", "one", "InstanceBusy", "is busy"),
        ("busy", None, "InstanceBusy", "is busy"),
    ],
)
def test_attach_blender_window_names_why_an_instance_cannot_be_attached(
    settings: Settings,
    advertised: tuple,
    monkeypatch: pytest.MonkeyPatch,
    fake_fastmcp: Any,
    hello: Optional[str],
    instance_id: Optional[str],
    error_type: str,
    says: str,
) -> None:
    """Review: a busy instance was reported as exited, and an unknown id as a RuntimeError."""
    from simul.blender_bridge.protocol import BridgeTimeout

    _, info = advertised
    answer = Mock(side_effect=BridgeTimeout("deadline", False)) if hello == "busy" else Mock(return_value=info)
    monkeypatch.setattr(BlenderAttachments, "_hello", lambda self, target: answer())
    server = SimulMCPServer(settings, backends={"blender"})
    tool = next(t for t in server.mcp.tools if t.name == "attach_blender_window")
    payload = json.loads(asyncio.run(tool.func(instance_id=instance_id)).content[0].text)

    assert payload["success"] is False
    assert payload["error_type"] == error_type, payload
    assert says in payload["error"]
    assert "exited" not in payload["error"]
    assert [entry["instance_id"] for entry in payload["details"]["instances"]] == ["one"]


@pytest.mark.parametrize("instance_id", ["one", None])
def test_attach_names_an_add_on_built_for_another_protocol(
    settings: Settings,
    advertised: tuple,
    monkeypatch: pytest.MonkeyPatch,
    fake_fastmcp: Any,
    instance_id: Optional[str],
) -> None:
    """Live review: an add-on of the previous protocol was told to "Enable the add-on"."""
    endpoint, _ = advertised
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        BridgeFiles.write(
            Path(settings.blender.discovery_dir) / "instance-one.json",
            {**endpoint, "port": listener.getsockname()[1], "protocol": PROTOCOL_VERSION - 1},
        )
        server = SimulMCPServer(settings, backends={"blender"})
        tool = next(t for t in server.mcp.tools if t.name == "attach_blender_window")
        payload = json.loads(asyncio.run(tool.func(instance_id=instance_id)).content[0].text)
        # The add-on was only seen to listen: it was sent nothing.
        accepted, _ = listener.accept()
        with accepted:
            accepted.settimeout(1)
            assert accepted.recv(1) == b""

    assert payload["success"] is False
    assert payload["error_type"] == "BridgeProtocolError", payload
    assert f"speaks protocol {PROTOCOL_VERSION - 1}" in payload["error"]
    assert "simul blender install-bridge" in payload["error"]
    assert "Enable the Simul Blender Bridge add-on" not in payload["error"]
    assert payload["details"]["instances"][0]["protocol_mismatch"] is True


def test_a_stale_record_of_another_protocol_is_not_called_a_mismatch(
    settings: Settings, advertised: tuple
) -> None:
    """A Blender that exited leaves its record; the rebuild advice would not apply to it."""
    endpoint, _ = advertised
    with socket.socket() as closed:
        closed.bind(("127.0.0.1", 0))
        port = closed.getsockname()[1]
    BridgeFiles.write(
        Path(settings.blender.discovery_dir) / "instance-one.json",
        {**endpoint, "port": port, "protocol": PROTOCOL_VERSION - 1},
    )
    attachments = BlenderAttachments(settings)
    [entry] = attachments.instances()
    assert entry["reachable"] is False
    assert entry["protocol_mismatch"] is False
    with pytest.raises(RuntimeError, match="No matching GUI Blender bridge"):
        attachments.attach()
