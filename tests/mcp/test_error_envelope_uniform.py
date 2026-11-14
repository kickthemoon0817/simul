"""One error envelope, sent as an MCP error, whatever layer or backend refused the call.

Pinned through real FastMCP clients:

* every failed call was sent with ``isError: false``; only FastMCP's own
  schema refusals set it, and those carried pydantic's raw text, no JSON;
* the same input mistake got a different envelope per layer (schema: raw
  text; request model: a multi-line pydantic message) and per backend;
* transport failures leaked library names (``ClientConnectorError``,
  ``ConnectionResetError``, ``Exception``) or no type at all;
* the instance tools, ``get_tool_usage_stats`` and Isaac's sandbox denials
  skipped the envelope, so they were neither rate limited nor recorded;
* with scripts disabled the instructions still pointed at execute_isaac_script;
* ``get_tool_usage_stats`` showed every session's records to any client, and
  ``limit=-3`` returned all but three.
"""

from __future__ import annotations

import ast
import asyncio
import errno
import json
import re
import time
from pathlib import Path
from typing import Annotated, Any, Dict, List

import pytest
from fastmcp import Client
from pydantic import BaseModel, Field

from simul.config import Settings
from simul.mcp import error_types
from simul.mcp import server as server_module
from simul.mcp.error_types import classify_exception, normalize_error_payload
from simul.mcp.session_manager import SessionManager
from simul.mcp.usage_tracker import ToolUsageTracker

DOCS = Path(__file__).resolve().parents[2] / "docs"


class _IsaacClient:
    """Reachable Isaac client double."""

    _host = "127.0.0.1"
    _port = 8226
    _bridge_host = "127.0.0.1"
    _bridge_port = 8229
    _bridge_configured = True
    address = bridge_address = "127.0.0.1:8229"
    vscode_address = "127.0.0.1:8226"
    bridge_circuit_open = False
    bridge_enabled = False
    fallback_to_vscode = False
    timeout_seconds = 5.0

    async def ping(self) -> bool:
        return True


def _usd_server(tmp_path: Path, **security: Any) -> server_module.SimulMCPServer:
    instance = server_module.SimulMCPServer(
        Settings(security={"allowed_paths": [str(tmp_path)], **security}), backends={"usd"}
    )
    instance.usage_tracker = ToolUsageTracker(log_dir=tmp_path / "usage")
    return instance


def _isaac_server(tmp_path: Path, **settings: Any) -> server_module.SimulMCPServer:
    instance = server_module.SimulMCPServer(
        Settings(
            isaac_sim={"discovery_dir": str(tmp_path / "disc")},
            security={"allowed_paths": [str(tmp_path)], **settings.pop("security", {})},
            **settings,
        ),
        backends={"isaac"},
    )
    instance.session_manager = SessionManager(tmp_path / "sessions")
    instance.usage_tracker = ToolUsageTracker(log_dir=tmp_path / "usage")
    instance._isaac_clients = {"default": _IsaacClient()}
    instance.client = instance._isaac_clients["default"]
    return instance


def _run(instance: server_module.SimulMCPServer, calls: List[tuple], mode: str = "legacy") -> List[Any]:
    async def run() -> List[Any]:
        async with Client(instance.mcp, mode=mode) as client:
            return [await client.call_tool(name, args, raise_on_error=False) for name, args in calls]

    return asyncio.run(run())


def _json(result: Any) -> Dict[str, Any]:
    return json.loads(result.content[-1].text)


# ---------------------------------------------------------------------------
# M5: a failure is an MCP error
# ---------------------------------------------------------------------------


def test_every_failed_call_is_sent_as_an_mcp_error(tmp_path: Path) -> None:
    (tmp_path / "scene.usda").write_text('#usda 1.0\ndef Xform "World" {}\n')
    instance = _usd_server(tmp_path)

    loaded, denied, missing = _run(
        instance,
        [
            ("load_usd_file", {"file_path": str(tmp_path / "scene.usda")}),
            ("load_usd_file", {"file_path": "/etc/passwd"}),
            ("get_prim_info", {"stage_id": "nope", "prim_path": "/World"}),
        ],
    )

    assert loaded.is_error is False
    assert [r.is_error for r in (denied, missing)] == [True, True]
    assert _json(denied)["error_type"] == "SandboxError"
    assert _json(missing)["error_type"] == "NotFoundError"


# ---------------------------------------------------------------------------
# M6: one ValidationError envelope from the schema and from a request model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("arguments", "field", "kind"),
    [
        ({"file_path": 123}, "file_path", "string_type"),
        ({}, "file_path", "missing_argument"),
        ({"file_path": "x.usda", "bogus": 1}, "bogus", "unexpected_keyword_argument"),
    ],
)
def test_a_schema_refusal_is_the_json_validation_envelope(
    tmp_path: Path, arguments: Dict[str, Any], field: str, kind: str
) -> None:
    (result,) = _run(_usd_server(tmp_path), [("load_usd_file", arguments)])

    assert result.is_error is True
    payload = _json(result)
    assert payload["success"] is False
    assert payload["error_type"] == "ValidationError"
    assert payload["error"].startswith(f"{field}: ")
    assert "pydantic" not in payload["error"] and "\n" not in payload["error"]
    (error,) = payload["details"]["errors"]
    assert error["field"] == field and error["type"] == kind
    assert payload["details"]["input"] == arguments


def test_schema_and_model_refusals_have_the_same_shape(tmp_path: Path) -> None:
    """search_type is now an enum in the schema; a request-model check reports alike."""
    schema, model = _run(
        _usd_server(tmp_path),
        [
            ("search_prims", {"stage_id": "s", "search_type": "by_magic", "query": "x"}),
            ("get_prim_info", {"stage_id": "s", "prim_path": "/World", "unknown": True}),
        ],
    )

    for result in (schema, model):
        payload = _json(result)
        assert result.is_error is True
        assert set(payload) == {"success", "error", "error_type", "details"}
        assert payload["error_type"] == "ValidationError"
        assert set(payload["details"]) == {"errors", "input"}
        keys = {"field", "message", "type"}
        optional = {"constraint", "input", "suggestion"}
        assert all(keys <= set(e) <= keys | optional for e in payload["details"]["errors"])


def test_an_unknown_tool_is_a_json_not_found_error(tmp_path: Path) -> None:
    instance = _usd_server(tmp_path, burst_size=1)

    result, limited = _run(instance, [("no_such_tool", {}), ("another_made_up_tool", {})])

    assert result.is_error is True
    assert _json(result) == {
        "success": False,
        "error": "Unknown tool: 'no_such_tool'",
        "error_type": "NotFoundError",
        "details": {"tool": "no_such_tool"},
    }
    # Recorded and rate limited like any call, under one name whatever was
    # asked for, so made-up names add no records or buckets of their own.
    assert _json(limited)["error_type"] == "RateLimitError"
    records = list(reversed(instance.usage_tracker.get_recent(tool_name="<unknown tool>")))
    assert [(r["success"], r["error"], r["params"]) for r in records] == [
        (False, "Unknown tool: 'no_such_tool'", {"name": "no_such_tool"}),
        (False, "rate_limited", {"name": "another_made_up_tool"}),
    ]


def test_a_script_tool_that_policy_hides_answers_script_execution_disabled(
    tmp_path: Path,
) -> None:
    instance = _isaac_server(tmp_path, security={"allow_script_execution": False})

    (result,) = _run(instance, [("execute_isaac_script", {"code": "print(1)"})])

    payload = _json(result)
    assert result.is_error is True
    assert payload["error_type"] == "ScriptExecutionDisabled"
    assert payload["details"] == {"tool": "execute_isaac_script"}
    assert "allow_script_execution" in payload["error"]
    assert [r["tool"] for r in instance.usage_tracker.get_recent()] == ["execute_isaac_script"]


def test_a_long_argument_is_echoed_by_size_at_the_schema_layer(tmp_path: Path) -> None:
    (result,) = _run(_usd_server(tmp_path), [("load_usd_file", {"file_path": 1, "x": "y" * 5000})])

    assert _json(result)["details"]["input"] == {"file_path": 1, "x": "<5000 characters>"}


def test_a_message_quoting_a_long_argument_is_cut(tmp_path: Path) -> None:
    """The tool's own refusal quotes the value; only details.input was bounded."""
    instance = _isaac_server(tmp_path)

    (result,) = _run(instance, [("configure_isaac_physics_scene", {"solver_type": "Q" * 40_000})])

    payload = _json(result)
    limit = error_types.MAX_MESSAGE_CHARS
    assert payload["error"].endswith("more characters>") and len(payload["error"]) < limit + 40
    (error,) = payload["details"]["errors"]
    assert len(error["message"]) < limit + 40
    assert len(result.content[-1].text) < 3 * limit
    record = instance.usage_tracker.get_recent(tool_name="configure_isaac_physics_scene")[0]
    assert len(record["error"]) < limit + 40


def test_a_long_argument_name_is_echoed_by_its_size(tmp_path: Path) -> None:
    (result,) = _run(_usd_server(tmp_path), [("load_usd_file", {"file_path": 1, "k" * 5000: 1})])

    payload = _json(result)
    assert payload["details"]["input"] == {"file_path": 1, "<name 1 of 5000 characters>": 1}
    assert len(result.content[-1].text) < 2000


def test_a_large_list_or_mapping_is_echoed_by_its_first_entries_and_a_count(
    tmp_path: Path,
) -> None:
    """A 20 000-vertex mesh refused for one stray argument used to come back whole."""
    vertices = [[i * 0.5, 1.0, 2.0] for i in range(20_000)]
    settings = {f"/k{i}": i for i in range(100)}

    (result,) = _run(
        _usd_server(tmp_path),
        [("load_usd_file", {"file_path": 1, "vertices": vertices, "settings": settings})],
    )

    echoed = _json(result)["details"]["input"]
    assert echoed["vertices"][:2] == [[0.0, 1.0, 2.0], [0.5, 1.0, 2.0]]
    assert echoed["vertices"][-1] == "<19984 more items, 20000 in all>"
    assert len(echoed["vertices"]) == 17
    assert list(echoed["settings"])[:2] == ["/k0", "/k1"]
    assert echoed["settings"]["..."] == "<84 more keys, 100 in all>"
    assert len(result.content[-1].text) < 4000


def test_a_schema_refusal_names_the_constraint_and_the_value(tmp_path: Path) -> None:
    """A bounded int: the field, the bound and the value given, in the envelope."""
    instance = _isaac_server(tmp_path)

    (result,) = _run(instance, [("capture_isaac_viewport", {"width": 5000})])

    (error,) = _json(result)["details"]["errors"]
    assert error == {
        "field": "width",
        "message": "Input should be less than or equal to 3840",
        "type": "less_than_equal",
        "constraint": {"le": 3840},
        "input": 5000,
    }


def test_a_misspelt_choice_or_argument_gets_a_suggestion(tmp_path: Path) -> None:
    light, extra = _run(
        _isaac_server(tmp_path),
        [
            ("create_isaac_light", {"prim_path": "/World/L", "light_type": "DomeLigth"}),
            ("create_isaac_prim", {"prim_path": "/World/X", "prim_typ": "Xform"}),
        ],
    )

    (choice,) = _json(light)["details"]["errors"]
    assert choice["field"] == "light_type" and choice["input"] == "DomeLigth"
    assert "DomeLight" in choice["constraint"]["expected"]
    assert choice["suggestion"] == "DomeLight"
    (argument,) = _json(extra)["details"]["errors"]
    assert argument["field"] == "prim_typ" and argument["suggestion"] == "prim_type"


class _Physics(BaseModel):
    mass: float = Field(gt=0)


class _Metadata(BaseModel):
    physics: _Physics


def test_a_nested_model_field_and_a_secret_argument(tmp_path: Path) -> None:
    """The dotted path of a nested model's field; a secret's value is never echoed."""
    instance = _usd_server(tmp_path)

    @instance.mcp.tool(name="probe_nested_tool")
    async def probe_nested_tool(metadata: _Metadata, auth_token: int = 0) -> str:
        return "ran"

    (result,) = _run(
        instance,
        [("probe_nested_tool", {"metadata": {"physics": {"mass": -1}}, "auth_token": "hunter2"})],
    )

    payload = _json(result)
    errors = {error["field"]: error for error in payload["details"]["errors"]}
    assert errors["metadata.physics.mass"]["constraint"] == {"gt": 0}
    assert errors["metadata.physics.mass"]["input"] == -1
    assert errors["auth_token"]["input"] == "***"
    assert payload["details"]["input"]["auth_token"] == "***"
    assert "hunter2" not in result.content[-1].text


_SECRET_ARGUMENTS = {
    "passphrase": "S1", "auth": "S2", "key": "S3", "password": "S4", "access_key": "S5",
    "private_key": "S6", "passwd": "S7", "pwd": "S8", "bearer": "S9", "cookie": "S10",
    "Authorization": "S11", "x_api_key": "S12", "client_secret": "S13", "session_token": "S14",
    "credentials": "S15", "jwt": "S16", "pin": "S17", "apiKey": "S18", "X-Auth-Token": "S19",
}


def test_secrets_are_masked_by_name_word_and_nested(tmp_path: Path) -> None:
    """Matched by whole words, so author is shown; under any mapping key or settings path."""
    (unexpected, nested) = _run(
        _usd_server(tmp_path),
        [
            ("load_usd_file", {"file_path": 1, **_SECRET_ARGUMENTS, "author": "Ada", "keyframe": 3}),
            (
                "load_usd_file",
                {
                    "file_path": 1,
                    "attributes": {"api_key": "NESTED-SECRET", "color": "red"},
                    "settings": {"/persistent/auth_token": "CARB-SECRET", "/app/fps": 60},
                    "layers": [{"token": "LIST-SECRET"}],
                },
            ),
        ],
    )

    echoed = _json(unexpected)["details"]["input"]
    assert {name: echoed[name] for name in _SECRET_ARGUMENTS} == dict.fromkeys(_SECRET_ARGUMENTS, "***")
    assert echoed["author"] == "Ada" and echoed["keyframe"] == 3
    text = nested.content[-1].text
    for secret in ("NESTED-SECRET", "CARB-SECRET", "LIST-SECRET"):
        assert secret not in text
    nested_input = _json(nested)["details"]["input"]
    assert nested_input["attributes"] == {"api_key": "***", "color": "red"}
    assert nested_input["settings"] == {"/persistent/auth_token": "***", "/app/fps": 60}
    assert nested_input["layers"] == [{"token": "***"}]


def test_a_constraint_names_only_the_bound_that_failed(tmp_path: Path) -> None:
    """pydantic's context also describes the input (field_type, actual_length); not a constraint."""
    instance = _usd_server(tmp_path)

    @instance.mcp.tool(name="probe_list_tool")
    async def probe_list_tool(
        position: Annotated[List[float], Field(min_length=3, max_length=3)],
    ) -> str:
        return "ran"

    (result,) = _run(instance, [("probe_list_tool", {"position": [1.0]})])

    (error,) = _json(result)["details"]["errors"]
    assert error["field"] == "position" and error["type"] == "too_short"
    assert error["constraint"] == {"min_length": 3}


def test_release_is_never_rate_limited(tmp_path: Path) -> None:
    """An agent that spent its budget must still be able to let its claim go."""
    instance = _isaac_server(
        tmp_path, security={"burst_size": 1, "global_requests_per_minute": 1}
    )

    claimed, refused, released, again = _run(
        instance,
        [
            ("claim_isaac_instance", {"purpose": "probe"}),
            ("ping_isaac", {}),
            ("release_isaac_instance", {}),
            ("release_isaac_instance", {}),
        ],
    )

    # The one token across every tool went to the claim.
    assert _json(claimed)["success"] is True
    assert _json(refused)["error_type"] == "RateLimitError"
    assert _json(refused)["details"]["scope"] == "agent"
    assert _json(released)["success"] is True
    assert _json(again)["error_type"] == "InvalidState"


def test_a_schema_refusal_is_rate_limited_and_recorded(tmp_path: Path) -> None:
    """It used to leave no usage record and spend no token."""
    instance = _isaac_server(tmp_path, security={"burst_size": 1})

    invalid, valid = _run(
        instance,
        [
            ("capture_isaac_viewport", {"width": 0}),
            ("capture_isaac_viewport", {"width": 640}),
        ],
    )

    assert _json(invalid)["error_type"] == "ValidationError"
    assert _json(valid)["error_type"] == "RateLimitError"
    records = list(reversed(instance.usage_tracker.get_recent(tool_name="capture_isaac_viewport")))
    assert [(r["success"], r["error"]) for r in records] == [
        (False, "width: Input should be greater than or equal to 1"),
        (False, "rate_limited"),
    ]


# ---------------------------------------------------------------------------
# m2: a stable taxonomy for transport failures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (ConnectionResetError(54, "reset"), "ConnectionError"),
        (ConnectionRefusedError(61, "refused"), "ConnectionError"),
        (BrokenPipeError(32, "pipe"), "ConnectionError"),
        (TimeoutError(), "TimeoutError"),
        (asyncio.TimeoutError(), "TimeoutError"),
        (OSError(errno.EHOSTUNREACH, "No route to host"), "ConnectionError"),
        (OSError("closed"), "ConnectionError"),
        (OSError(errno.ENOSPC, "No space left on device"), "EngineError"),
        (ModuleNotFoundError("No module named 'omni.isaac.core'"), "EngineError"),
        (ImportError("cannot import name 'World'"), "EngineError"),
        (asyncio.InvalidStateError("Result is not set."), "EngineError"),
        (ValueError("seed must be an integer"), "ValidationError"),
        (TypeError("expected str"), "ValidationError"),
        (json.JSONDecodeError("bad", "{", 0), "ValidationError"),
        (UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"), "EngineError"),
        (KeyError("x"), "NotFoundError"),
        (IndexError("x"), "NotFoundError"),
        (FileNotFoundError(2, "missing"), "NotFoundError"),
        (PermissionError(13, "denied"), "PermissionDenied"),
        (NotImplementedError(), "UnsupportedOperation"),
        (RuntimeError("poll() failed"), "EngineError"),
        (AttributeError("x"), "EngineError"),
        (Exception("bare"), "EngineError"),
    ],
)
def test_exceptions_map_onto_the_shared_classes(exc: BaseException, expected: Any) -> None:
    assert classify_exception(exc) == expected
    assert classify_exception(exc) in error_types.VOCABULARY


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        # Relayed from an engine, OSError is its failure (a full disk inside
        # Kit or Blender), not a connection that broke.
        ("OSError", "EngineError"),
        ("ModuleNotFoundError", "EngineError"),
        ("ImportError", "EngineError"),
        ("InvalidStateError", "EngineError"),
        ("ClientOSError", "ConnectionError"),
        ("ConnectionRefusedError", "ConnectionError"),
        ("InvalidRequest", "ValidationError"),
    ],
)
def test_names_map_onto_their_family(name: str, expected: str) -> None:
    assert error_types.family_of(name) == expected


@pytest.mark.parametrize(
    ("name", "transport_failure", "raised", "expected"),
    [
        # The bridge's and the client's own failures keep their meaning.
        ("StageUnavailable", False, "EngineError", "InvalidState"),
        ("UnsafeExecutionDisabled", False, "ScriptError", "ScriptExecutionDisabled"),
        ("PrimNotFound", False, "EngineError", "NotFoundError"),
        ("RequestTimeout", False, "ScriptError", "TimeoutError"),
        ("EmptyResponse", False, "ScriptError", "ConnectionError"),
        ("ScriptTooLargeForSocket", False, "ScriptError", "PayloadTooLarge"),
        ("JSONDecodeError", True, "ScriptError", "BridgeProtocolError"),
        ("ScriptInterrupted", True, "ScriptError", "TimeoutError"),
        ("TimeoutError", True, "ScriptError", "TimeoutError"),
        # Anything the engine's code raised is the engine's, or the agent's script's.
        ("OSError", False, "EngineError", "EngineError"),
        ("TimeoutError", False, "ScriptError", "ScriptError"),
        ("ConnectionRefusedError", False, "ScriptError", "ScriptError"),
        ("JSONDecodeError", False, "ScriptError", "ScriptError"),
        ("ModuleNotFoundError", False, "EngineError", "EngineError"),
        ("TypeError", False, "EngineError", "EngineError"),
        ("ScriptInterrupted", False, "ScriptError", "ScriptError"),
    ],
)
def test_isaac_failures_are_read_by_where_they_came_from(
    name: str, transport_failure: bool, raised: str, expected: str
) -> None:
    assert (
        error_types.engine_failure_type(name, transport_failure=transport_failure, raised=raised)
        == expected
    )


def test_a_long_message_parses_in_linear_time() -> None:
    """A bpy-enum pattern with an optional [\\w.]* group backtracked quadratically.

    "solver_type must be one of TGS, PGS, got 'Z9Z9...'" took 4.6 s at 40 KB and
    froze every other client meanwhile.
    """
    for message in (
        "solver_type must be one of TGS, PGS, got '" + "Z9" * 50_000 + "'",
        "x" * 100_000,
        ("a" * 50 + " = ") * 2000,
        'enum "' * 20_000,
        'enum "x" not found in (' * 5000,
    ):
        started = time.perf_counter()
        errors = error_types.errors_from_message(message)
        payload = error_types.normalize_error_payload(
            {"success": False, "error": message, "error_type": "ValueError"}
        )
        elapsed = time.perf_counter() - started
        assert elapsed < 0.05, (message[:40], elapsed)
        assert len(errors[0]["message"]) <= error_types.MAX_MESSAGE_CHARS + 40
        assert len(payload["error"]) <= error_types.MAX_MESSAGE_CHARS + 40


def test_the_bpy_enum_parse_still_names_the_field_and_the_choices() -> None:
    (error,) = error_types.errors_from_message(
        "TypeError: bpy_struct: item.type = val: enum \"BOXX\" not found in ('BOX', 'SPHERE')"
    )

    assert error["field"] == "type"
    assert error["constraint"] == {"expected": ["BOX", "SPHERE"]}
    assert error["input"] == "BOXX" and error["suggestion"] == "BOX"


def test_a_huge_argument_does_not_hold_up_another_client(tmp_path: Path) -> None:
    """The tool's own refusal quotes the value; the envelope reads a bounded copy."""
    instance = _isaac_server(tmp_path, security={"rate_limiting_enabled": False})

    async def run() -> tuple:
        async with Client(instance.mcp, mode="legacy") as big, Client(instance.mcp, mode="legacy") as other:
            started = time.perf_counter()
            slow = asyncio.create_task(
                big.call_tool(
                    "configure_isaac_physics_scene",
                    {"solver_type": "Z9" * 50_000},
                    raise_on_error=False,
                )
            )
            await asyncio.sleep(0)
            await other.call_tool("ping_isaac", {}, raise_on_error=False)
            other_done = time.perf_counter() - started
            result = await slow
            return result, other_done, time.perf_counter() - started

    result, other_done, total = asyncio.run(run())

    payload = _json(result)
    assert payload["error_type"] == "ValidationError"
    assert "solver_type" in payload["error"]
    assert len(result.content[-1].text) < 8000
    assert total < 2.0 and other_done < 2.0


def test_the_isaac_bridge_protocol_error_is_not_a_validation_error() -> None:
    """A ValueError subclass, but a reply the bridge sent: not the caller's input."""
    from simul.adapters.isaac_socket_client import BridgeResponseProtocolError

    assert classify_exception(BridgeResponseProtocolError("garbled")) == "BridgeProtocolError"


def test_library_exception_names_map_by_class_name() -> None:
    class ClientConnectorError(OSError):
        pass

    class ServerTimeoutError(Exception):
        pass

    assert classify_exception(ClientConnectorError()) == "ConnectionError"
    assert classify_exception(ServerTimeoutError()) == "TimeoutError"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            {"success": False, "error": "No response from Isaac Sim", "error_type": "EmptyResponse"},
            {"error_type": "ConnectionError", "details": {"exception_type": "EmptyResponse"}},
        ),
        (
            {"success": False, "error": "", "error_type": "TimeoutError"},
            {"error_type": "TimeoutError", "error": error_types._EMPTY_MESSAGES["TimeoutError"]},
        ),
        (
            {"success": False, "connected": False, "error": "Cannot connect"},
            {"error_type": "ConnectionError"},
        ),
        (
            {"success": False, "reachable": False, "error": "not reachable"},
            {"error_type": "ConnectionError"},
        ),
        ({"success": False, "error": "Prim not found"}, {"error_type": "EngineError"}),
        (
            {"success": False, "error": "Invalid light_type", "error_type": "ValueError"},
            {"error_type": "ValidationError", "details": {"exception_type": "ValueError", "errors": [
                {"field": None, "message": "Invalid light_type", "type": "value_error"}
            ]}},
        ),
        (
            {"success": False, "error": "boom", "error_type": "AttributeError"},
            {"error_type": "EngineError", "details": {"exception_type": "AttributeError"}},
        ),
        (
            {"success": False, "error": "gone", "error_type": "SimulNotFound"},
            {"error_type": "NotFoundError", "details": {"exception_type": "SimulNotFound"}},
        ),
        (
            {"success": False, "error": "x", "error_type": "TransformNotApplied"},
            {"error_type": "TransformNotApplied", "details": None},
        ),
        (
            {"success": False, "error": "disk full", "error_type": "OSError"},
            {"error_type": "EngineError", "details": {"exception_type": "OSError"}},
        ),
        (
            {"success": False, "error_type": "RefusedOperation", "error": "kept"},
            {"error_type": "RefusedOperation"},
        ),
    ],
)
def test_failed_payloads_get_a_stable_type(payload: Dict[str, Any], expected: Dict[str, Any]) -> None:
    normalized = normalize_error_payload(dict(payload))
    assert {key: normalized.get(key) for key in expected} == expected


def test_release_without_a_claim_names_the_state(tmp_path: Path) -> None:
    (result,) = _run(_isaac_server(tmp_path), [("release_isaac_instance", {})])

    payload = _json(result)
    assert result.is_error is True
    assert payload["error_type"] == "InvalidState"
    assert "agent_id" in payload["details"]["hint"]


def test_every_error_type_is_documented() -> None:
    documented = (DOCS / "tools.md").read_text()
    names = [
        value
        for key, value in vars(error_types).items()
        if key.isupper() and isinstance(value, str) and not key.startswith("_")
    ]
    assert error_types.NOT_APPLIED_SUFFIX in names
    names.remove(error_types.NOT_APPLIED_SUFFIX)
    assert set(names) == error_types.VOCABULARY
    assert [name for name in names if f"`{name}`" not in documented] == []
    assert "`*NotApplied`" in documented


# Where a tool's error_type is spelled: ``error_type="X"``, ``"error_type": "X"``
# (a payload, or a generated script's print), ``error_type = "X"`` (a class).
_ERROR_TYPE_LITERAL = re.compile(
    r"""["']?error_type["']?\s*[:=]\s*["']([A-Za-z_]+)["']"""
)
_SRC = Path(__file__).resolve().parents[2] / "src" / "simul"


def test_no_tool_source_spells_a_type_outside_the_vocabulary() -> None:
    """Every error_type literal a tool can return is in the vocabulary.

    The envelope maps anything else onto its family at run time
    (``normalize_error_payload``); this catches the literal at its source, so
    a new domain type is added to ``error_types`` and documented instead of
    silently becoming an EngineError. The CLI's own JSON is not a tool's.
    """
    found: Dict[str, List[str]] = {}
    for source in sorted(_SRC.rglob("*.py")):
        if "cli" in source.relative_to(_SRC).parts:
            continue
        where = str(source.relative_to(_SRC))
        text = source.read_text()
        for name in _ERROR_TYPE_LITERAL.findall(text):
            found.setdefault(name, []).append(where)
        for name in _positional_failure_names(ast.parse(text)):
            found.setdefault(name, []).append(where)
    assert found
    # In the vocabulary, or mapped on purpose (the transports' own names);
    # never left to fall through to EngineError unnoticed.
    outside = {name: files for name, files in found.items() if not error_types.is_mapped(name)}
    assert outside == {}
    bridge = {
        name for name, files in found.items() if any(f.startswith("bridge_ext") for f in files)
    }
    assert {"StageUnavailable", "UnsafeExecutionDisabled", "PrimNotFound"} <= bridge


# Where a tool names its error_type by a constant, as the Unreal editor scripts
# do (``"error_type": _NOT_IN_SIMULATION`` in a script, which an f-string sets
# from the module's NOT_IN_SIMULATION).
_ERROR_TYPE_CONSTANT = re.compile(r"""["']?error_type["']?\s*[:=]\s*_?([A-Z][A-Z0-9_]+)\b""")


def test_no_tool_names_a_type_outside_the_vocabulary_by_a_constant() -> None:
    """Round-2 audit R2B-04: apply_unreal_force's NotInSimulation was named by a
    constant the literal scan above does not read, was outside the vocabulary, and
    reached clients as EngineError."""
    from simul.adapters import _unreal_errors, _unreal_mesh_scripts, _unreal_scripts, unreal_runtime

    modules = (error_types, _unreal_scripts, _unreal_mesh_scripts, _unreal_errors, unreal_runtime)
    found: Dict[str, List[str]] = {}
    for source in sorted((_SRC / "adapters").glob("*unreal*.py")):
        for constant in _ERROR_TYPE_CONSTANT.findall(source.read_text()):
            found.setdefault(constant, []).append(str(source.relative_to(_SRC)))
    assert "NOT_IN_SIMULATION" in found
    values = {}
    for constant in found:
        owners = [getattr(module, constant) for module in modules if isinstance(getattr(module, constant, None), str)]
        assert owners, f"{constant} ({found[constant]}) is not a constant of {[m.__name__ for m in modules]}"
        values[constant] = owners[0]
    assert {name: value for name, value in values.items() if not error_types.is_mapped(value)} == {}


# Calls that name a failure by position: the bridge's BridgeResponse.failure(
# request_id, name, message), UnrealError(message, error_type), and the
# keyword forms ScriptResult(error_name=...) / failure(name=...).
_POSITIONAL = {"failure": 1, "UnrealError": 1, "UnrealTimeoutError": 1, "UnrealConnectionError": 1}
_NAME_KEYWORDS = {"error_type", "error_name", "name"}


def _positional_failure_names(tree: ast.AST) -> List[str]:
    names: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        index = _POSITIONAL.get(callee)
        if index is not None and len(node.args) > index:
            argument = node.args[index]
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                names.append(argument.value)
        for keyword in node.keywords:
            if keyword.arg in _NAME_KEYWORDS and (callee in _POSITIONAL or keyword.arg != "name"):
                value = keyword.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    names.append(value.value)
    return names


# ---------------------------------------------------------------------------
# m3: every call goes through the envelope
# ---------------------------------------------------------------------------


def test_instance_and_stats_tools_and_isaac_sandbox_denials_are_recorded(tmp_path: Path) -> None:
    instance = _isaac_server(tmp_path)

    _run(
        instance,
        [
            ("list_isaac_instances", {"scan": False}),
            ("set_active_isaac_instance", {"instance_name": "default"}),
            ("claim_isaac_instance", {"purpose": "test"}),
            ("release_isaac_instance", {}),
            ("open_isaac_stage", {"file_path": "/etc/passwd"}),
            ("get_tool_usage_stats", {}),
            ("get_tool_usage_stats", {}),
        ],
    )

    recorded = [row["tool"] for row in reversed(instance.usage_tracker.get_recent(limit=200))]
    assert recorded == [
        "list_isaac_instances",
        "set_active_isaac_instance",
        "claim_isaac_instance",
        "release_isaac_instance",
        "open_isaac_stage",
        "get_tool_usage_stats",
        "get_tool_usage_stats",
    ]
    denial = instance.usage_tracker.get_recent(tool_name="open_isaac_stage")[0]
    assert denial["success"] is False


def test_isaac_sandbox_denials_are_rate_limited(tmp_path: Path) -> None:
    instance = _isaac_server(tmp_path, security={"burst_size": 2})

    results = _run(instance, [("open_isaac_stage", {"file_path": "/etc/passwd"})] * 3)

    assert [_json(r)["error_type"] for r in results] == ["SandboxError", "SandboxError", "RateLimitError"]


# ---------------------------------------------------------------------------
# m14: with scripts off, nothing points at the script tool
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("allowed", [True, False])
def test_instructions_and_descriptions_mention_the_script_tool_only_when_registered(
    tmp_path: Path, allowed: bool
) -> None:
    instance = _isaac_server(tmp_path, security={"allow_script_execution": allowed})

    async def listing() -> Dict[str, str]:
        async with Client(instance.mcp) as client:
            return {tool.name: tool.description or "" for tool in await client.list_tools()}

    descriptions = asyncio.run(listing())
    assert ("execute_isaac_script" in descriptions) is allowed
    mentions = [
        name for name, text in descriptions.items()
        if "execute_isaac_script" in text and name != "execute_isaac_script"
    ]
    assert mentions == []
    assert ("execute_isaac_script" in (instance.mcp.instructions or "")) is allowed
    assert "get_isaac_ui_window" in (instance.mcp.instructions or "")


# ---------------------------------------------------------------------------
# m15: usage stats are the caller's own
# ---------------------------------------------------------------------------


def test_usage_stats_show_the_caller_its_own_session(tmp_path: Path) -> None:
    (tmp_path / "secret.usda").write_text('#usda 1.0\ndef Xform "World" {}\n')
    instance = _usd_server(tmp_path)

    async def run() -> List[Dict[str, Any]]:
        async with Client(instance.mcp, mode="legacy") as alice, Client(
            instance.mcp, mode="legacy"
        ) as bob:
            await alice.call_tool("validate_usd_file", {"file_path": str(tmp_path / "secret.usda")})
            await alice.call_tool("validate_usd_file", {"file_path": str(tmp_path / "secret.usda")})
            own = _json(await bob.call_tool("get_tool_usage_stats", {"include_recent": True}))
            server = _json(
                await bob.call_tool(
                    "get_tool_usage_stats", {"include_recent": True, "scope": "server"}
                )
            )
            alices = _json(await alice.call_tool("get_tool_usage_stats", {"include_recent": True}))
            return [own, server, alices]

    own, server, alices = asyncio.run(run())

    assert own["scope"] == "session"
    assert own["total_calls"] == 0 and own["recent"] == []
    assert "secret.usda" not in json.dumps(own)
    # The whole server's counts, but still only bob's records.
    assert server["tools"]["validate_usd_file"]["total_calls"] == 2
    assert [r["tool"] for r in server["recent"]] == ["get_tool_usage_stats"]
    assert "secret.usda" not in json.dumps(server)
    assert alices["tools"]["validate_usd_file"]["total_calls"] == 2
    assert [r["tool"] for r in alices["recent"]] == ["validate_usd_file", "validate_usd_file"]


@pytest.mark.parametrize("limit", [-3, 0, 201])
def test_usage_stats_refuse_a_limit_out_of_range(tmp_path: Path, limit: int) -> None:
    (result,) = _run(_usd_server(tmp_path), [("get_tool_usage_stats", {"include_recent": True, "limit": limit})])

    payload = _json(result)
    assert result.is_error is True
    assert payload["error_type"] == "ValidationError"
    assert payload["details"]["errors"][0]["field"] == "limit"


def test_the_usage_log_row_carries_the_session_label(tmp_path: Path) -> None:
    instance = _usd_server(tmp_path)
    _run(instance, [("get_tool_usage_stats", {})])

    (row,) = [json.loads(line) for line in (tmp_path / "usage" / "tool_usage.jsonl").read_text().splitlines()]
    assert re.fullmatch(r"session-[0-9a-f]{12}", row["session"])
    assert "session" not in instance.usage_tracker.get_recent()[0]


@pytest.mark.parametrize(
    ("message", "field"),
    [
        ("stiffness must be >= 0, got -1.0", "stiffness"),
        ("time_steps_per_second must be >= 1, got 0", "time_steps_per_second"),
        ("scene_type must be one of room, corridor, warehouse, outdoor, got 'cave'", "scene_type"),
        ("Invalid mode 'Bogus'. Must be one of ['EvaluateStatement']", "mode"),
        ("min_position_iterations (5) exceeds max_position_iterations (2)", "min_position_iterations"),
        ("aov_names must not exceed 16 entries", "aov_names"),
        ("Pass at least one setting to configure.", None),
        ("Invalid light_type", None),
        ("ValueError: Attached window is not in Object Mode", None),
        ('bpy_struct: attribute "x" is read-only', None),
    ],
)
def test_a_tool_check_names_the_field_it_refused(message: str, field: Any) -> None:
    """details.errors[].field names the parameter, as docs/tools.md promises (round-2 m-4)."""
    payload = normalize_error_payload({"success": False, "error": message, "error_type": "ValidationError"})
    assert payload["details"]["errors"][0]["field"] == field
    assert error_types.field_named_by(message) == field


def test_a_tool_check_over_mcp_names_its_field(tmp_path: Path) -> None:
    """A check only the tool makes (a min above its max) reaches the client with the field."""
    (result,) = _run(
        _isaac_server(tmp_path),
        [("configure_isaac_physics_scene", {"min_position_iterations": 9, "max_position_iterations": 2})],
    )

    (error,) = _json(result)["details"]["errors"]
    assert error["field"] == "min_position_iterations"
