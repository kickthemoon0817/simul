"""Round-3 Unreal audit minors: list parameters, the attached-mode refusal, the exec CLI's
error type and a raising script's output.

* ``actor_paths`` given as a JSON array was split on its commas into paths such
  as ``["/Game/...``, reported as "Actor not found", while vector parameters
  took JSON arrays.
* ``control_unreal_ui`` outside attached mode came back as EngineError
  (a RuntimeError), though it is a precondition.
* ``simul unreal exec`` reported a timeout as ``UnrealTimeoutError``, where the
  MCP tools report the shared ``TimeoutError``.
* ``execute_unreal_script`` lost what a script printed before it raised.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List
from unittest.mock import patch

import pytest
from fastmcp import Client
from typer.testing import CliRunner

from simul.adapters.unreal_runtime import UnrealRuntimeSession, UnrealTimeoutError
from simul.config import Settings
from simul.mcp import server as server_module
from simul.mcp.registration._unreal_tools import _name_list, _path_list
from tests.fakes import AvailableAdapter

A = "/Game/Map.Map:PersistentLevel.A"
B = "/Game/Map.Map:PersistentLevel.B"


@pytest.mark.parametrize(
    "value",
    [f"{A},{B}", f" {A} , {B} ", json.dumps([A, B]), f'  ["{A}", "{B}"]', [A, B], [f" {A}", B]],
)
def test_lists_take_a_json_array_or_comma_separated_text(value: Any) -> None:
    assert _path_list(value) == [A, B]
    assert _name_list(value) == [A, B]


def test_an_empty_list_is_none_and_blanks_are_dropped() -> None:
    assert _name_list("") is None and _name_list([]) is None
    assert _name_list(f"{A},,") == [A] and _name_list(json.dumps([A, ""])) == [A]


@pytest.mark.parametrize("value", ['["/Game/a", ', "[1, 2]", [1, 2], {"a": 1}])
def test_a_malformed_array_is_a_value_error(value: Any) -> None:
    with pytest.raises(ValueError):
        _path_list(value)


class _Recording:
    def __init__(self) -> None:
        self.calls: List[tuple] = []

    async def get_physics_state(self, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(("get_physics_state", kwargs))
        return {"world": "editor", "bodies": [], "count": 0, "total": 0}


class _Adapter(AvailableAdapter):
    def __init__(self, session: _Recording) -> None:
        super().__init__(Settings())
        self.session = session

    @contextmanager
    def create_session(self) -> Iterator[_Recording]:
        yield self.session


@pytest.mark.parametrize("given", [json.dumps([A, B]), [A, B], f"{A},{B}"])
def test_actor_paths_reach_the_session_whole_through_the_tool(monkeypatch: pytest.MonkeyPatch, given: Any) -> None:
    base = Settings()
    settings = base.model_copy(update={"unreal": base.unreal.model_copy(update={"tool_surface": "full"})})
    instance = server_module.SimulMCPServer(settings, backends={"unreal"})
    session = _Recording()
    monkeypatch.setattr(instance, "unreal_adapter", _Adapter(session))

    async def call() -> Any:
        async with Client(instance.mcp) as client:
            schema = {tool.name: tool.input_schema for tool in await client.list_tools()}
            result = await client.call_tool("get_unreal_physics_state", {"actor_paths": given},
                                            raise_on_error=False)
            return schema["get_unreal_physics_state"]["properties"]["actor_paths"], result

    prop, result = asyncio.run(call())

    assert not result.is_error, result
    assert session.calls[0][1]["actor_paths"] == [A, B]
    assert {branch.get("type") for branch in prop["anyOf"]} == {"string", "array"}
    assert "JSON array" in prop["description"]


def test_named_editor_control_outside_attached_mode_is_invalid_state() -> None:
    session = UnrealRuntimeSession(Settings())

    with pytest.raises(Exception) as raised:
        asyncio.run(session.control_ui("select_actor", target="Cube"))

    assert getattr(raised.value, "error_type", None) == "InvalidState"
    assert raised.value.details == {"code": "NOT_ATTACHED", "mode": session.settings.unreal.mode}
    assert "UNREAL__MODE=attached" in str(raised.value)


def test_a_script_that_prints_then_raises_keeps_its_output() -> None:
    session = UnrealRuntimeSession(Settings())

    async def execute(code: str, mode: str = "ExecuteFile") -> Dict[str, Any]:
        return {
            "ReturnValue": False,
            "CommandResult": "Traceback (most recent call last):\nValueError: boom",
            "LogOutput": [{"Type": "Info", "Output": "step 1\n"}, {"Type": "Info", "Output": "step 2\n"},
                          {"Type": "Error", "Output": "ValueError: boom"}],
        }

    session._execute_python = execute  # type: ignore[method-assign]
    result = asyncio.run(session.execute_script("print('step 1'); print('step 2'); raise ValueError('boom')"))

    assert (result["success"], result["error_type"]) == (False, "ScriptError")
    assert "ValueError: boom" in result["error"]
    assert result["output"] == "step 1\nstep 2\n"


def test_exec_reports_a_timeout_by_the_shared_error_type() -> None:
    from simul.cli.main import app

    async def times_out(self: Any, code: str, mode: str = "ExecuteFile", **_: Any) -> Dict[str, Any]:
        raise UnrealTimeoutError("did not finish the script within 5s", "TimeoutError")

    with patch.object(UnrealRuntimeSession, "_execute_python", times_out):
        result = CliRunner().invoke(app, ["--json", "unreal", "exec", "print(1)"])

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert payload["error_type"] == "TimeoutError"
