"""Snapshots of the error payload each tool envelope sends, and the usage it records.

One scenario per representative failure, driven through the registered tool
the way an MCP client calls it: an Isaac instance that is not running, a
script that raised inside Isaac Sim, a ``bpy`` operator failure in embedded
and attached Blender, an Unreal editor that refuses the connection, invalid
input, a rate-limited call, a claimed instance and a busy one. Each pins the
exact JSON the caller receives and the usage records the call leaves behind,
one per call. Fields that depend on the clock or the host (retry delays,
claim expiry, a pydantic help URL, an ephemeral port) are normalised or
derived from the same source rather than written out.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import re
import socket
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import pytest

from simul.adapters import blender_runtime
from simul.adapters.blender_connection import BlenderAttachments
from simul.adapters.isaac_socket_client import ScriptResult
from simul.blender_bridge.protocol import (
    PROTOCOL_VERSION,
    BridgeFiles,
    BridgeRemoteError,
    BridgeWire,
)
from simul.config import Settings
from simul.mcp import backends as backends_module
from simul.mcp import server as server_module
from simul.mcp.schemas.blender import BlenderSceneObjectsRequest
from simul.mcp.session_manager import SessionManager
from simul.mcp.usage_tracker import ToolUsageTracker
from tests.fakes import FakeFastMCP

ISAAC_ADDRESS = "127.0.0.1:8229"
CODE = "print(1)"
CODE_SHA256 = hashlib.sha256(CODE.encode("utf-8")).hexdigest()
POLL_FAILED = (
    "Operator bpy.ops.rigidbody.object_add.poll() failed, context is incorrect"
)
ENUM_MISSING = (
    'Converting py args to operator properties: enum "BOGUS" not found in '
    "('FORCE', 'WIND', 'VORTEX')"
)
NOT_REACHABLE = (
    f"Isaac Sim is not reachable at {ISAAC_ADDRESS}. Ensure Isaac Sim is running with the "
    "isaacsim.code_editor.vscode extension enabled. Use ping_isaac to verify connectivity."
)
TRACEBACK = (
    'Traceback (most recent call last):\n  File "<string>", line 1, in <module>\n'
    "NameError: name 'stage' is not defined\n"
)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _server(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    settings: Optional[Settings] = None,
    *,
    blender: bool = False,
    unreal: bool = False,
) -> server_module.SimulMCPServer:
    """Build a server on the FastMCP double with a private usage log."""
    monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
    monkeypatch.setattr(server_module, "TaskConfig", None)
    monkeypatch.setattr(backends_module, "is_headless_available", lambda: False)
    monkeypatch.setattr(backends_module, "is_blender_available", lambda: blender)
    if not unreal:
        monkeypatch.setattr(backends_module, "UnrealRuntimeAdapter", None)
    instance = server_module.SimulMCPServer(settings=settings or Settings())
    instance.usage_tracker = ToolUsageTracker(log_dir=tmp_path / "usage")
    return instance


def _call(
    instance: server_module.SimulMCPServer, tool: str, **kwargs: Any
) -> Dict[str, Any]:
    """Call a registered tool and return the JSON block it sends."""
    result = asyncio.run(instance.mcp.by_name[tool](**kwargs))
    assert result.structured_content is None
    return json.loads(result.content[-1].text)


def _snapshot(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Replace the clock-dependent fields of an error payload with placeholders."""
    payload = copy.deepcopy(payload)
    if "retry_after_seconds" in payload:
        assert 0 < payload["retry_after_seconds"] <= 60
        payload["retry_after_seconds"] = "<seconds>"
        payload["error"] = re.sub(
            r"retry in [0-9.]+ s", "retry in <seconds> s", payload["error"]
        )
    details = payload.get("details") or {}
    if "claim_expires_in_seconds" in details:
        assert (
            0 < details["claim_expires_in_seconds"] <= server_module.CLAIM_TTL_SECONDS
        )
        details["claim_expires_in_seconds"] = "<seconds>"
    return payload


def _records(instance: server_module.SimulMCPServer, tool: str) -> List[Dict[str, Any]]:
    """Return the tool's usage records, oldest first, without timing fields."""
    rows = list(reversed(instance.usage_tracker.get_recent(tool_name=tool)))
    for row in rows:
        assert row.pop("agent_id"), "every record names its agent"
        row.pop("timestamp")
        row.pop("duration_ms")
        row.pop("tool")
    return rows


def _validation_message(model: Any, **kwargs: Any) -> str:
    """The pydantic message for invalid input; it embeds a version-specific URL."""
    with pytest.raises(Exception) as raised:
        model(**kwargs)
    return str(raised.value)


# ---------------------------------------------------------------------------
# Isaac Sim
# ---------------------------------------------------------------------------


class _IsaacClient:
    """Socket client double; ``outcome`` is the ScriptResult or exception ``execute`` yields."""

    address = ISAAC_ADDRESS
    bridge_address = ISAAC_ADDRESS
    vscode_address = "127.0.0.1:8226"
    timeout_seconds = 30.0
    bridge_enabled = False
    fallback_to_vscode = True

    def __init__(self, outcome: Any = None) -> None:
        self._host = "127.0.0.1"
        self._port = 8226
        self._bridge_port = 8229
        self._bridge_configured = True
        self._outcome = outcome

    async def execute(self, code: str) -> ScriptResult:
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        assert isinstance(self._outcome, ScriptResult)
        return self._outcome


def _isaac_server(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    outcome: Any = None,
    settings: Optional[Settings] = None,
) -> server_module.SimulMCPServer:
    instance = _server(monkeypatch, tmp_path, settings)
    client = _IsaacClient(outcome)
    instance._isaac_clients = {"default": client}  # type: ignore[dict-item]
    instance.client = client  # type: ignore[assignment]
    return instance


def test_isaac_not_running(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    instance = _isaac_server(
        monkeypatch, tmp_path, ConnectionRefusedError(61, "Connection refused")
    )

    payload = _call(instance, "execute_isaac_script", code=CODE)

    assert payload == {
        "success": False,
        "error": NOT_REACHABLE,
        "error_type": "ConnectionError",
        "details": None,
    }
    assert _records(instance, "execute_isaac_script") == [
        {
            "success": False,
            "error": NOT_REACHABLE,
            "params": {"code_bytes": len(CODE)},
            "script_sha256": CODE_SHA256,
        }
    ]


def test_isaac_script_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    raised = ScriptResult(
        success=False,
        output="",
        error_name="NameError",
        error_value="name 'stage' is not defined",
        traceback=TRACEBACK,
    )
    instance = _isaac_server(monkeypatch, tmp_path, raised)

    payload = _call(instance, "execute_isaac_script", code=CODE)

    assert payload == {
        "success": False,
        "error": "name 'stage' is not defined",
        "error_type": "NameError",
        "details": {"traceback": TRACEBACK},
    }
    assert _records(instance, "execute_isaac_script") == [
        {
            "success": False,
            "error": "name 'stage' is not defined",
            "params": {"code_bytes": len(CODE)},
            "script_sha256": CODE_SHA256,
        }
    ]


def test_isaac_tool_that_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    instance = _isaac_server(monkeypatch, tmp_path)

    async def reset(include_prim_count: bool = False) -> Dict[str, Any]:
        raise ConnectionResetError(54, "Connection reset by peer")

    monkeypatch.setattr(instance._isaac_tools, "get_isaac_stage_info", reset)

    payload = _call(instance, "get_isaac_stage_info")

    assert payload == {
        "success": False,
        "error": "[Errno 54] Connection reset by peer",
        "error_type": "ConnectionResetError",
        "details": None,
    }
    assert _records(instance, "get_isaac_stage_info") == [
        {"success": False, "error": "[Errno 54] Connection reset by peer"}
    ]


def test_isaac_partial_failure_without_a_success_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A hand-built payload naming a failure under ``*_error`` and no ``success``.

    Isaac payloads now pass through ``apply_success_from_error`` like every
    other backend's; the Isaac envelope used to forward them untouched and
    record the call as a success.
    """
    instance = _isaac_server(monkeypatch, tmp_path)

    async def partial(include_prim_count: bool = False) -> Dict[str, Any]:
        return {"stage_url": "anon:stage", "timeline_error": "timeline unavailable"}

    monkeypatch.setattr(instance._isaac_tools, "get_isaac_stage_info", partial)

    payload = _call(instance, "get_isaac_stage_info")

    assert payload == {
        "stage_url": "anon:stage",
        "timeline_error": "timeline unavailable",
        "success": False,
    }
    assert _records(instance, "get_isaac_stage_info") == [{"success": False}]


def test_isaac_rate_limited(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    instance = _isaac_server(
        monkeypatch, tmp_path, settings=Settings(security={"burst_size": 1})
    )

    async def stage_info(include_prim_count: bool = False) -> Dict[str, Any]:
        return {"success": True, "stage_url": "anon:stage"}

    monkeypatch.setattr(instance._isaac_tools, "get_isaac_stage_info", stage_info)
    agent = instance._resolve_agent_id(None)

    first = _call(instance, "get_isaac_stage_info")
    refused = _call(instance, "get_isaac_stage_info")

    assert first == {"success": True, "stage_url": "anon:stage"}
    assert _snapshot(refused) == {
        "success": False,
        "error": "Rate limit exceeded; retry in <seconds> s",
        "error_type": "RateLimitError",
        "details": {
            "tool": "get_isaac_stage_info",
            "agent_id": agent,
            "scope": "tool",
            "limit_per_minute": 60,
        },
        "retry_after_seconds": "<seconds>",
    }
    assert _records(instance, "get_isaac_stage_info") == [
        {"success": True},
        {"success": False, "error": "rate_limited"},
    ]


def test_isaac_claimed_instance_refusal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = Settings(
        isaac_sim={"enforce_claims": True, "discovery_dir": str(tmp_path / "disc")}
    )
    instance = _isaac_server(
        monkeypatch, tmp_path, ScriptResult(success=True, output="{}"), settings
    )
    instance.session_manager = SessionManager(tmp_path / "sessions")
    context: ContextVar[Any] = ContextVar("context", default=None)
    monkeypatch.setattr(instance, "_get_request_context", lambda: context.get())

    def as_agent(session_id: str, tool: str, **kwargs: Any) -> Dict[str, Any]:
        token = context.set(SimpleNamespace(session_id=session_id))
        try:
            return _call(instance, tool, **kwargs)
        finally:
            context.reset(token)

    assert (
        as_agent("agent-a", "claim_isaac_instance", purpose="RL training run")[
            "success"
        ]
        is True
    )
    refused = as_agent("agent-b", "execute_isaac_script", code=CODE)

    assert _snapshot(refused) == {
        "success": False,
        "error": (
            "Isaac instance 'default' is claimed by agent 'agent-a' for 'RL training run'; "
            "isaac_sim.enforce_claims refuses mutating tools from other agents while that claim is live."
        ),
        "error_type": "InstanceClaimed",
        "details": {
            "instance": "default",
            "holder_agent_id": "agent-a",
            "holder_purpose": "RL training run",
            "claim_expires_in_seconds": "<seconds>",
            "hint": (
                "Read-only tools still work. Ask the holder to call release_isaac_instance, or "
                "wait for the claim to expire (120 s without activity), then call "
                "claim_isaac_instance yourself. list_isaac_instances shows other instances."
            ),
        },
    }
    assert _records(instance, "execute_isaac_script") == [
        {
            "success": False,
            "error": "instance_claimed",
            "params": {"code_bytes": len(CODE)},
            "script_sha256": CODE_SHA256,
        }
    ]


def test_isaac_busy_instance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    instance = _isaac_server(monkeypatch, tmp_path)
    instance._instance_lock_timeout = 0.05

    async def held() -> Any:
        async with instance._get_instance_lock("default"):
            return await instance.mcp.by_name["get_isaac_stage_info"]()

    payload = json.loads(asyncio.run(held()).content[0].text)

    assert payload == {
        "success": False,
        "error": (
            "Isaac instance 'default' is busy with another call. It is running, "
            "not unreachable — retry shortly."
        ),
        "error_type": "InstanceBusy",
        "details": None,
    }
    assert _records(instance, "get_isaac_stage_info") == [
        {"success": False, "error": "instance_busy"}
    ]


# ---------------------------------------------------------------------------
# Blender
# ---------------------------------------------------------------------------


def _raising(exc: BaseException) -> Any:
    def operator(*args: Any, **kwargs: Any) -> Any:
        raise exc

    return operator


def _stand_in_bpy() -> SimpleNamespace:
    """Just enough of bpy for the physics tools, with operators that fail like Blender 5.0's."""
    cube = SimpleNamespace(
        name="Cube", type="MESH", rigid_body=None, select_set=lambda value: None
    )
    scene = SimpleNamespace(rigidbody_world=SimpleNamespace(), objects={"Cube": cube})
    return SimpleNamespace(
        app=SimpleNamespace(version=(5, 0, 1), version_string="5.0.1", background=True),
        data=SimpleNamespace(objects={"Cube": cube}),
        context=SimpleNamespace(
            scene=scene,
            view_layer=SimpleNamespace(objects=SimpleNamespace(active=None)),
        ),
        ops=SimpleNamespace(
            rigidbody=SimpleNamespace(object_add=_raising(RuntimeError(POLL_FAILED))),
            object=SimpleNamespace(effector_add=_raising(TypeError(ENUM_MISSING))),
        ),
    )


def _embedded_blender(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **security: Any
) -> Any:
    """A server whose real embedded Blender adapter runs real sessions over a stand-in bpy."""
    monkeypatch.setattr(blender_runtime, "bpy", _stand_in_bpy())
    monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
    settings = Settings(blender={"mode": "embedded"}, security=security)
    return _server(monkeypatch, tmp_path, settings, blender=True)


def test_blender_bpy_failure_with_object_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """bpy's RuntimeError keeps its class; the object it concerned rides as a note."""
    instance = _embedded_blender(monkeypatch, tmp_path)

    payload = _call(instance, "setup_blender_rigid_body", object_name="Cube")

    message = f"Failed to add rigid body to 'Cube': {POLL_FAILED}"
    assert payload == {
        "success": False,
        "error": message,
        "error_type": "RuntimeError",
        "details": None,
    }
    assert _records(instance, "setup_blender_rigid_body") == [
        {"success": False, "error": message}
    ]


def test_blender_bpy_failure_without_added_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A bad enum is a TypeError from bpy, whose message already lists the valid values.

    The wrapper that restated the field type, and re-raised the TypeError as a
    RuntimeError, is gone.
    """
    instance = _embedded_blender(monkeypatch, tmp_path)

    payload = _call(
        instance,
        "add_blender_force_field",
        field_type="BOGUS",
        location=[0.0, 0.0, 0.0],
    )

    message = ENUM_MISSING
    assert payload == {
        "success": False,
        "error": message,
        "error_type": "TypeError",
        "details": None,
    }
    assert _records(instance, "add_blender_force_field") == [
        {"success": False, "error": message}
    ]


def test_blender_attached_bpy_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The add-on replies with the failure; the client raises it as a BridgeRemoteError.

    The envelope reports the class the add-on raised, which the reply names.
    """
    settings = Settings(
        blender={
            "mode": "attached",
            "discovery_dir": str(tmp_path / "blender"),
            "attachment_path": str(tmp_path / "blender" / "attachment.json"),
        }
    )
    BridgeFiles.write(
        tmp_path / "blender" / "instance-one.json",
        {
            "instance_id": "one",
            "protocol": PROTOCOL_VERSION,
            "host": "127.0.0.1",
            "port": 12345,
            "token": "secret",
            "pid": os.getpid(),
        },
    )
    window = {
        "window_id": "w",
        "scene_id": "s",
        "scene_name": "Scene",
        "workspace": "Layout",
    }
    hello = {
        "instance_id": "one",
        "document_id": "doc",
        "pid": os.getpid(),
        "background": False,
        "blend_file_path": None,
        "is_dirty": False,
        "windows": [window],
    }
    monkeypatch.setattr(BlenderAttachments, "_hello", lambda self, target: hello)
    BlenderAttachments(settings).attach()
    remote = f"RuntimeError: Failed to add rigid body to 'Cube': {POLL_FAILED}"

    def reply(endpoint: Any, payload: Any, timeout: float) -> Dict[str, Any]:
        raise BridgeRemoteError(remote, "RuntimeError", {})

    monkeypatch.setattr(BridgeWire, "request", reply)
    instance = _server(monkeypatch, tmp_path, settings)

    payload = _call(instance, "setup_blender_rigid_body", object_name="Cube")

    assert payload == {
        "success": False,
        "error": remote,
        "error_type": "RuntimeError",
        "details": None,
    }
    assert _records(instance, "setup_blender_rigid_body") == [
        {"success": False, "error": remote}
    ]


def test_blender_validation_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Validated inside the envelope now, so the failure is recorded like any other."""
    instance = _embedded_blender(monkeypatch, tmp_path)

    payload = _call(instance, "list_blender_scene_objects", max_items=0)

    arguments = {"collection_name": None, "include_hidden": False, "max_items": 0}
    message = _validation_message(BlenderSceneObjectsRequest, **arguments)
    assert payload == {
        "success": False,
        "error": message,
        "error_type": "ValidationError",
        "details": {"input": arguments},
    }
    assert _records(instance, "list_blender_scene_objects") == [
        {"success": False, "error": message}
    ]


def test_blender_validation_error_and_the_rate_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With one token, an invalid call followed by a valid one.

    The invalid call spends the token, so the valid one is refused; before,
    invalid input never reached the rate limit.
    """
    instance = _embedded_blender(monkeypatch, tmp_path, burst_size=1)
    agent = instance._resolve_agent_id(None)

    invalid = _call(instance, "get_blender_object_info", object_name=None)
    payload = _call(instance, "get_blender_object_info", object_name="Missing")

    assert invalid["error_type"] == "ValidationError"
    assert _snapshot(payload) == {
        "success": False,
        "error": "Rate limit exceeded; retry in <seconds> s",
        "error_type": "RateLimitError",
        "details": {
            "tool": "get_blender_object_info",
            "agent_id": agent,
            "scope": "tool",
            "limit_per_minute": 60,
        },
        "retry_after_seconds": "<seconds>",
    }
    assert _records(instance, "get_blender_object_info") == [
        {"success": False, "error": invalid["error"]},
        {"success": False, "error": "rate_limited"},
    ]


def test_blender_runtime_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    instance = _embedded_blender(monkeypatch, tmp_path)
    monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", False)

    payload = _call(instance, "get_blender_info")

    assert payload == {
        "success": False,
        "error": "Blender runtime not available",
        "error_type": "RuntimeError",
        "details": None,
    }
    assert _records(instance, "get_blender_info") == [
        {"success": False, "error": "Blender runtime not available"}
    ]


# ---------------------------------------------------------------------------
# Unreal Engine
# ---------------------------------------------------------------------------


def _refused_port() -> int:
    """A loopback port that was just free, so connecting to it is refused."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _aiohttp_refusal(port: int) -> Tuple[str, str]:
    """The message and class aiohttp raises for a refused Remote Control call on ``port``."""
    import aiohttp

    async def attempt() -> Tuple[str, str]:
        async with aiohttp.ClientSession() as session:
            try:
                await session.put(
                    f"http://127.0.0.1:{port}/remote/object/call", json={}
                )
            except aiohttp.ClientError as exc:
                return str(exc), type(exc).__name__
        raise AssertionError(f"port {port} accepted a connection")

    return asyncio.run(attempt())


def _unreal_server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, port: int) -> Any:
    pytest.importorskip("aiohttp")
    settings = Settings(unreal={"host": "127.0.0.1", "port": port, "max_retries": 0})
    return _server(monkeypatch, tmp_path, settings, unreal=True)


def test_unreal_editor_refuses_the_connection(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """aiohttp's exception escapes the session and is named by its class."""
    port = _refused_port()
    instance = _unreal_server(monkeypatch, tmp_path, port)
    message, raised = _aiohttp_refusal(port)
    assert raised == "ClientConnectorError"

    payload = _call(instance, "execute_unreal_script", code=CODE)

    assert payload == {
        "success": False,
        "error": message,
        "error_type": "ClientConnectorError",
        "details": None,
    }
    assert _records(instance, "execute_unreal_script") == [
        {
            "success": False,
            "error": message,
            "params": {"code_bytes": len(CODE), "mode": "ExecuteFile"},
        }
    ]


def test_unreal_health_check_reports_the_refusal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The session catches this one itself; the envelope derives ``success`` from ``error``."""
    port = _refused_port()
    instance = _unreal_server(monkeypatch, tmp_path, port)

    payload = _call(instance, "unreal_health_check")

    assert payload["error"].startswith(f"Cannot connect to host 127.0.0.1:{port}")
    assert payload == {
        "reachable": False,
        "connected": False,
        "error": payload["error"],
        "success": False,
    }
    assert _records(instance, "unreal_health_check") == [
        {"success": False, "error": payload["error"]}
    ]
