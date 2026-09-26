"""Unreal runtime calls report editor-side failures instead of passing them off as values.

UE's ``ExecutePythonCommandEx`` puts an expression's value *and* a raised
traceback in the same ``CommandResult`` field, telling them apart only by
``ReturnValue``. These tests pin that every reader checks it, and that the
viewport and generic-call tools stop inventing results.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

import pytest

from simul.adapters import unreal_runtime
from simul.config import Settings

TRACEBACK = (
    "Traceback (most recent call last):\n  File \"<string>\", line 1, in <module>\n"
    "AttributeError: 'NoneType' object has no attribute 'get_path_name'\n"
)


class _Response:
    def __init__(self, payload: Dict[str, Any], status: int = 200) -> None:
        self._payload = payload
        self.status = status

    async def json(self) -> Dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise Exception(f"HTTP {self.status}")

    async def __aenter__(self) -> "_Response":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass


class _Client:
    """Routes each PUT through ``handler`` and records the bodies."""

    def __init__(self, handler: Any) -> None:
        self.handler = handler
        self.bodies: List[Dict[str, Any]] = []
        self.closed = False

    def get(self, path: str) -> _Response:
        return _Response({})

    def put(self, path: str, json: Any = None) -> _Response:
        self.bodies.append({"path": path, **(json or {})})
        return self.handler(path, json or {})

    def post(self, path: str, json: Any = None) -> _Response:
        return _Response({}, 404)

    async def close(self) -> None:
        self.closed = True


def _scripting_settings() -> Settings:
    base = Settings()
    return base.model_copy(
        update={"security": base.security.model_copy(update={"allow_script_execution": True})}
    )


def _session(monkeypatch: pytest.MonkeyPatch, handler: Any) -> unreal_runtime.UnrealRuntimeSession:
    monkeypatch.setattr(unreal_runtime, "UNREAL_AVAILABLE", True)
    monkeypatch.setattr(unreal_runtime, "AIOHTTP_AVAILABLE", True)
    session = unreal_runtime.UnrealRuntimeSession(settings=_scripting_settings())
    session._session = _Client(handler)
    return session


def _python(payload: Dict[str, Any]) -> Any:
    """Answer every Python call with ``payload`` and engine-version calls with 5.7."""

    def handler(path: str, body: Dict[str, Any]) -> _Response:
        if body.get("functionName") == "GetEngineVersion":
            return _Response({"ReturnValue": "5.7.4"})
        if body.get("functionName") == "ExecutePythonCommandEx":
            return _Response(payload)
        return _Response({}, 404)

    return handler


FAILED = {"ReturnValue": False, "CommandResult": TRACEBACK, "LogOutput": []}


# -- execute_script (execute_unreal_script) ----------------------------------


def test_evaluate_statement_returns_the_command_result(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _session(monkeypatch, _python({"ReturnValue": True, "CommandResult": "3", "LogOutput": []}))

    result = asyncio.run(session.execute_script("1+2", mode="EvaluateStatement"))

    assert result == {"success": True, "result": "3", "output": ""}


def test_script_without_json_succeeds_with_its_output(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _session(monkeypatch, _python({
        "ReturnValue": True,
        "CommandResult": "None",
        "LogOutput": [{"Type": "Info", "Output": "hello\n"}, {"Type": "Warning", "Output": "careful\n"}],
    }))

    result = asyncio.run(session.execute_script("print('hello')"))

    assert result == {"success": True, "output": "hello\n"}


def test_script_json_is_returned_as_printed(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _session(monkeypatch, _python({
        "ReturnValue": True,
        "LogOutput": [{"Type": "Info", "Output": "noise\n"}, {"Type": "Info", "Output": '{"count": 4}\n'}],
    }))

    assert asyncio.run(session.execute_script("...")) == {"count": 4}


def test_script_json_error_stays_a_script_error(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _session(monkeypatch, _python({
        "ReturnValue": True,
        "LogOutput": [{"Type": "Info", "Output": '{"error": "bad input"}'}],
    }))

    result = asyncio.run(session.execute_script("..."))

    assert result == {"success": False, "error_type": "ScriptError", "error": "bad input"}


@pytest.mark.parametrize("mode", ["ExecuteFile", "EvaluateStatement", "ExecuteStatement"])
def test_raised_script_is_a_script_error_in_every_mode(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    session = _session(monkeypatch, _python(FAILED))

    result = asyncio.run(session.execute_script("boom", mode=mode))

    assert result["success"] is False
    assert result["error_type"] == "ScriptError"
    assert "AttributeError" in result["error"]


# -- engine info / loaded map / scene summary / health ------------------------


def test_get_engine_info_reports_a_raised_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _session(monkeypatch, _python(FAILED))

    result = asyncio.run(session.get_engine_info())

    assert result["success"] is False
    assert result["error_type"] == "ScriptError"
    assert "project_name" not in result and "loaded_map" not in result


def test_get_loaded_map_reports_a_raised_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _session(monkeypatch, _python(FAILED))

    result = asyncio.run(session.get_loaded_map())

    assert result["success"] is False
    assert "map_path" not in result
    assert "AttributeError" in result["error"]


def test_summarize_scene_reports_a_raised_map_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _session(monkeypatch, _python(FAILED))

    result = asyncio.run(session.summarize_scene())

    assert result["success"] is False
    assert "map_path" not in result
    # It stops before walking the actors.
    assert not any(b.get("functionName") == "GetAllLevelActors" for b in session._session.bodies)


def test_health_check_turns_a_raised_name_probe_into_a_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _session(monkeypatch, _python(FAILED))

    result = asyncio.run(session.health_check())

    assert result["reachable"] is True
    assert result["project_name"] == ""
    assert any("project_name unavailable" in w for w in result["warnings"])


# -- viewport ------------------------------------------------------------------


def test_get_viewport_info_propagates_a_failed_read(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _session(monkeypatch, lambda path, body: _Response({}, 400))

    with pytest.raises(Exception, match="400"):
        asyncio.run(session.get_viewport_info())


def test_get_viewport_info_without_a_camera_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _session(monkeypatch, lambda path, body: _Response({"ReturnValue": False}))

    result = asyncio.run(session.get_viewport_info())

    assert result["success"] is False
    assert result["error_type"] == "ViewportUnavailable"
    assert "camera_location" not in result


def test_set_camera_view_keeps_the_unset_rotation(monkeypatch: pytest.MonkeyPatch) -> None:
    current = {"Pitch": -10.0, "Yaw": 30.0, "Roll": 0.0}

    def handler(path: str, body: Dict[str, Any]) -> _Response:
        if body.get("functionName") == "GetLevelViewportCameraInfo":
            return _Response({"CameraLocation": {"X": 1, "Y": 2, "Z": 3}, "CameraRotation": current,
                              "ReturnValue": True})
        return _Response({})

    session = _session(monkeypatch, handler)

    result = asyncio.run(session.set_camera_view(location=(10.0, 20.0, 30.0)))

    assert result == {"location": (10.0, 20.0, 30.0), "rotation": None}
    sent = next(b for b in session._session.bodies if b.get("functionName") == "SetLevelViewportCameraInfo")
    assert sent["parameters"]["CameraLocation"] == {"X": 10.0, "Y": 20.0, "Z": 30.0}
    assert sent["parameters"]["CameraRotation"] == current


def test_set_camera_view_has_no_fov_parameter() -> None:
    import inspect

    assert "fov" not in inspect.signature(unreal_runtime.UnrealRuntimeSession.set_camera_view).parameters


# -- call_actor_function -------------------------------------------------------


@pytest.mark.parametrize("parameters", ["{not json", "[1, 2]", "42"])
def test_call_actor_function_rejects_bad_parameters_without_calling(
    monkeypatch: pytest.MonkeyPatch, parameters: str
) -> None:
    session = _session(monkeypatch, lambda path, body: _Response({"ReturnValue": 1}))

    result = asyncio.run(session.call_actor_function(
        actor_path="/Game/Maps/T.T:PersistentLevel.A", function_name="Jump", parameters=parameters,
    ))

    assert result["success"] is False
    assert result["error_type"] == "ValidationError"
    assert session._session.bodies == []


def test_call_actor_function_passes_a_json_object(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _session(monkeypatch, lambda path, body: _Response({"ReturnValue": 1}))

    asyncio.run(session.call_actor_function(
        actor_path="/Game/Maps/T.T:PersistentLevel.A", function_name="SetSpeed", parameters='{"Speed": 2}',
    ))

    assert session._session.bodies[0]["parameters"] == {"Speed": 2}


# -- MCP wrappers ----------------------------------------------------------------


@pytest.fixture
def unreal_server(monkeypatch: pytest.MonkeyPatch) -> Any:
    from simul.mcp import backends as backends_module
    from simul.mcp import server as server_module
    from simul.mcp.registration import register_unreal_tools
    from tests.fakes import FakeFastMCP

    monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
    monkeypatch.setattr(server_module, "TaskConfig", None)
    monkeypatch.setattr(backends_module, "is_headless_available", lambda: False)
    monkeypatch.setattr(backends_module, "is_blender_available", lambda: False)
    monkeypatch.setattr(backends_module, "UnrealRuntimeAdapter", None)
    instance = server_module.SimulMCPServer(settings=_scripting_settings(), backends={"isaac"})
    register_unreal_tools(instance, thin=False)
    return instance


def _payload(result: Any) -> Dict[str, Any]:
    return json.loads(result.content[-1].text)


def test_set_camera_view_tool_rejects_a_partial_location(unreal_server: Any) -> None:
    tool = unreal_server.mcp.by_name["set_unreal_camera_view"]

    payload = _payload(asyncio.run(tool(location_x=1.0, location_y=2.0)))

    assert payload["success"] is False
    assert payload["error_type"] == "ValidationError"


def test_execute_unreal_script_tool_returns_evaluate_statement_value(
    unreal_server: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contextlib import contextmanager

    from tests.fakes import AvailableAdapter

    session = _session(monkeypatch, _python({"ReturnValue": True, "CommandResult": "'TP_BlankBP'"}))

    class _Adapter(AvailableAdapter):
        @contextmanager
        def create_session(self) -> Any:
            yield session

    monkeypatch.setattr(unreal_server, "unreal_adapter", _Adapter(Settings()))
    tool = unreal_server.mcp.by_name["execute_unreal_script"]

    payload = _payload(asyncio.run(tool(code="unreal.SystemLibrary.get_game_name()", mode="EvaluateStatement")))

    assert payload["success"] is True
    assert payload["result"] == "'TP_BlankBP'"
