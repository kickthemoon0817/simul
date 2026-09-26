"""Attachment identity, guarded dispatch, and explicit named-control validation."""

import io
import json
import sys
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from typer.testing import CliRunner

from simul.adapters.unreal_connection import UnrealAttachments, read_attachment
from simul.adapters.unreal_runtime import UnrealRuntimeSession
from simul.cli.main import app
from simul.config import Settings


@pytest.fixture
def settings(tmp_path):
    base = Settings()
    return base.model_copy(
        update={
            "unreal": base.unreal.model_copy(
                update={
                    "mode": "attached",
                    "attachment_path": str(tmp_path / "attachment.json"),
                    "host": "127.0.0.1",
                    "ping_timeout": 1,
                }
            )
        }
    )


@pytest.fixture
def editor(monkeypatch):
    callbacks, events = [], []
    level = SimpleNamespace(
        on_map_changed=SimpleNamespace(add_callable=callbacks.append),
        get_viewport_config_keys=lambda: ["viewport-a"],
        get_active_viewport_config_key=lambda: "viewport-a",
    )
    world = SimpleNamespace(get_path_name=lambda: "/Temp/Untitled.Untitled")
    unreal = SimpleNamespace(
        LevelEditorSubsystem="level",
        UnrealEditorSubsystem="editor",
        get_editor_subsystem=lambda name: (
            level
            if name == "level"
            else SimpleNamespace(get_editor_world=lambda: world)
        ),
        Paths=SimpleNamespace(get_project_file_path=lambda: "/project/test.uproject"),
        SystemLibrary=SimpleNamespace(get_engine_version=lambda: "5.7"),
        EditorLoadingAndSavingUtils=SimpleNamespace(get_dirty_map_packages=lambda: []),
        events=events,
    )
    monkeypatch.setitem(sys.modules, "unreal", unreal)
    monkeypatch.delitem(sys.modules, "_simul_unreal_attachment_v1", raising=False)
    yield unreal, callbacks, events
    sys.modules.pop("_simul_unreal_attachment_v1", None)


async def serve_editor():
    calls = []

    async def handle(request):
        body = await request.json()
        calls.append(body)
        if body.get("functionName") != "ExecutePythonCommandEx":
            return web.json_response({"ack": True})
        output = io.StringIO()
        params = body["parameters"]
        try:
            with redirect_stdout(output):
                if params["ExecutionMode"] == "EvaluateStatement":
                    value = eval(params["PythonCommand"], {})
                else:
                    exec(params["PythonCommand"], {})
                    value = None
            return web.json_response(
                {
                    "ReturnValue": True,
                    "CommandResult": repr(value),
                    "LogOutput": [
                        {"Type": "Info", "Output": line}
                        for line in output.getvalue().splitlines()
                    ],
                }
            )
        except Exception as exc:
            return web.json_response(
                {"ReturnValue": False, "CommandResult": str(exc), "LogOutput": []}
            )

    application = web.Application()
    application.router.add_put("/remote/object/call", handle)
    runner = web.AppRunner(application)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    return runner, site._server.sockets[0].getsockname()[1], calls


@pytest.mark.asyncio
async def test_guards_execute_on_editor_before_scripts_and_legacy_mutations(
    settings, editor
):
    unreal, callbacks, events = editor
    runner, port, calls = await serve_editor()
    session = UnrealRuntimeSession(settings)
    try:
        manager = UnrealAttachments(settings)
        record = await manager.attach(port=port)
        assert not events
        assert record["viewport"] == "viewport-a"
        assert read_attachment(manager.path)["port"] == port
        assert manager.path.stat().st_mode & 0o077 == 0
        result = await session._execute_python(
            "import unreal; unreal.events.append('first')"
        )
        assert result["ReturnValue"] is True
        assert events == ["first"]
        assert (await session._execute_python("40 + 2", mode="EvaluateStatement"))[
            "CommandResult"
        ] == "42"
        assert (
            await session._execute_python(
                "import unreal; unreal.events.append('statement')",
                mode="ExecuteStatement",
            )
        )["ReturnValue"] is True
        assert await session._call_function("/Actor", "Move") == {"ack": True}
        level = unreal.get_editor_subsystem("level")
        level.get_active_viewport_config_key = lambda: "different-viewport"
        before = len(calls)
        with pytest.raises(RuntimeError, match="activate the attached viewport"):
            await session._call_function("/Editor", "SetLevelViewportCameraInfo")
        assert len(calls) == before + 1  # guard only; other viewport never moved
        level.get_active_viewport_config_key = lambda: "viewport-a"
        # Same map path, but a fresh load generation: old attachment must fail.
        for callback in callbacks:
            callback(1)
        result = await session._execute_python(
            "import unreal; unreal.events.append('wrong map')"
        )
        assert (
            result["ReturnValue"] is False
            and "attachment changed" in result["CommandResult"]
        )
        before = len(calls)
        with pytest.raises(RuntimeError, match="attachment changed"):
            await session._call_function("/Actor", "Move")
        assert len(calls) == before + 1  # verification only; mutation never dispatched
        assert events == ["first", "statement"]
        manager.detach()
        fresh = UnrealRuntimeSession(settings)
        with pytest.raises(RuntimeError, match="No Unreal editor attached"):
            await fresh._call_function("/Actor", "Move")
    finally:
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_attach_requires_unambiguous_editor_and_viewport(settings, monkeypatch):
    manager = UnrealAttachments(settings)
    info = {
        "instance_id": "one",
        "document_id": "doc",
        "project_path": "/project.uproject",
        "map_path": "/Game/Map",
        "host": "localhost",
        "port": 30010,
        "viewports": ["a", "b"],
        "reachable": True,
    }
    monkeypatch.setattr(manager, "instances", AsyncMock(return_value=[info, info]))
    with pytest.raises(ValueError, match="unique Unreal"):
        await manager.attach()
    monkeypatch.setattr(manager, "instances", AsyncMock(return_value=[info]))
    with pytest.raises(ValueError, match="unique level viewport"):
        await manager.attach()
    monkeypatch.setattr(
        manager, "describe", AsyncMock(return_value={**info, "document_id": "changed"})
    )
    with pytest.raises(RuntimeError, match="changed while attaching"):
        await manager.attach(viewport="a")
    assert not manager.path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [
        {"agent_control": "execute_script"},
        {
            "agent_control": "set_property",
            "property_name": "__class__",
            "value": [1, 2, 3],
        },
        {
            "agent_control": "set_property",
            "property_name": "location",
            "value": [float("nan"), 0, 0],
        },
        {"agent_control": "set_game_view"},
        {"agent_control": "inspect", "target": "Cube"},
        {"agent_control": "move_cursor", "agent_id": "\n"},
        {"agent_control": "move_cursor", "agent_id": "a" * 65},
        {"agent_control": "move_cursor", "position": [float("nan"), 0]},
        {"agent_control": "move_cursor", "position": [-1, 0]},
        {"agent_control": "move_cursor", "position": [0.5]},
        {"agent_control": "inspect", "position": [0.5, 0.5]},
        {"agent_control": "move_cursor", "activity": "a" * 257},
        {"agent_control": "select_actor", "activity": "custom"},
    ],
)
async def test_invalid_controls_fail_before_any_connection(settings, kwargs):
    session = UnrealRuntimeSession(settings)
    session._ensure_http_session = AsyncMock(
        side_effect=AssertionError("must not connect")
    )
    with pytest.raises(ValueError):
        await session.control_ui(**kwargs)
    session._ensure_http_session.assert_not_called()


def _valid_record():
    return {
        "version": 1,
        "host": "127.0.0.1",
        "port": 30010,
        "target": {
            "instance_id": "i",
            "document_id": "d",
            "project_path": "/p.uproject",
            "map_path": "/Game/Map",
            "viewport": "viewport-a",
        },
    }


def test_attachment_record_is_private_and_round_trips(tmp_path):
    from simul.utils.private_files import BridgeFiles

    path = tmp_path / "nested" / "attachment.json"
    BridgeFiles.write(path, _valid_record())
    assert path.stat().st_mode & 0o777 == 0o600
    assert read_attachment(path) == _valid_record()


def test_attachment_refuses_symlink_loose_mode_and_oversize(tmp_path):
    from simul.adapters.unreal_connection import MAX_ATTACHMENT_BYTES
    from simul.utils.private_files import BridgeFiles

    real = tmp_path / "attachment.json"
    BridgeFiles.write(real, _valid_record())
    link = tmp_path / "link.json"
    link.symlink_to(real)
    with pytest.raises(OSError):
        read_attachment(link)

    real.chmod(0o644)
    with pytest.raises(PermissionError):
        read_attachment(real)

    big = tmp_path / "big.json"
    BridgeFiles.write(big, {**_valid_record(), "pad": "x" * MAX_ATTACHMENT_BYTES})
    with pytest.raises(ValueError):
        read_attachment(big)


def test_attachment_keeps_its_schema_checks(tmp_path):
    from simul.utils.private_files import BridgeFiles

    path = tmp_path / "attachment.json"
    BridgeFiles.write(path, {**_valid_record(), "port": 80})
    with pytest.raises(ValueError, match="Invalid Unreal attachment port"):
        read_attachment(path)


def test_cli_status_refuses_missing_attachment(monkeypatch, settings):
    monkeypatch.setattr("simul.cli.unreal_cli.get_settings", lambda: settings)
    result = CliRunner().invoke(app, ["--json", "unreal", "status"])
    assert result.exit_code == 1
    assert "No Unreal editor attached" in json.loads(result.stdout)["error"]


@pytest.mark.parametrize("option,value", [("--host", "localhost"), ("--port", "30019")])
def test_cli_attached_endpoint_override_returns_json_error(
    monkeypatch, settings, option, value
):
    monkeypatch.setattr("simul.cli.unreal_cli.get_settings", lambda: settings)
    result = CliRunner().invoke(app, ["--json", "unreal", "health", option, value])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["success"] is False
    assert payload["error_type"] == "ValueError"
    assert "Host/port overrides cannot replace an attached editor" in payload["error"]
