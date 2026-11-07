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

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastmcp import Client

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
        assert all(set(e) == {"field", "message", "type"} for e in payload["details"]["errors"])


def test_an_unknown_tool_is_a_json_not_found_error(tmp_path: Path) -> None:
    (result,) = _run(_usd_server(tmp_path), [("no_such_tool", {})])

    assert result.is_error is True
    assert _json(result) == {
        "success": False,
        "error": "Unknown tool: 'no_such_tool'",
        "error_type": "NotFoundError",
        "details": {"tool": "no_such_tool"},
    }


def test_a_long_argument_is_echoed_by_size_at_the_schema_layer(tmp_path: Path) -> None:
    (result,) = _run(_usd_server(tmp_path), [("load_usd_file", {"file_path": 1, "x": "y" * 5000})])

    assert _json(result)["details"]["input"] == {"file_path": 1, "x": "<5000 characters>"}


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
        (OSError(65, "No route to host"), "ConnectionError"),
        (ValueError("seed must be an integer"), "ValidationError"),
        (KeyError("x"), None),
        (FileNotFoundError(2, "missing"), None),
        (json.JSONDecodeError("bad", "{", 0), None),
    ],
)
def test_exceptions_map_onto_the_shared_classes(exc: BaseException, expected: Any) -> None:
    assert classify_exception(exc) == expected


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
            {"error_type": "ConnectionError", "details": {"cause": "EmptyResponse"}},
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
            {"error_type": "ValidationError", "details": {"cause": "ValueError", "errors": [
                {"field": None, "message": "Invalid light_type", "type": "value_error"}
            ]}},
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
    assert names and [name for name in names if f"`{name}`" not in documented] == []


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
