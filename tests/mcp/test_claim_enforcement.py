"""Instance claims: compatibility scoring, enforcement on/off, expiry and release ownership."""

from __future__ import annotations

import json
import time
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest


from simul_mcp.config import Settings
from simul_mcp.mcp import backends as backends_module
from simul_mcp.mcp import server as server_module
from simul_mcp.mcp import session_manager as session_manager_module
from simul_mcp.mcp.session_manager import SessionManager
from tests.fakes import FakeFastMCP


class FakeIsaacClient:
    def __init__(self, *, socket_port: int, bridge_port: int) -> None:
        self._host = "127.0.0.1"
        self._port = socket_port
        self._bridge_host = "127.0.0.1"
        self._bridge_port = bridge_port
        self._bridge_configured = True
        self._prefer_bridge = True
        self._fallback_to_vscode = True
        self._timeout_seconds = 5.0

    @property
    def address(self) -> str:
        return f"{self._bridge_host}:{self._bridge_port}"

    bridge_address = address

    @property
    def vscode_address(self) -> str:
        return f"{self._host}:{self._port}"

    async def ping(self) -> bool:
        return True


def _make_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, enforce: bool
) -> tuple[server_module.SimulMCPServer, ContextVar[Any]]:
    monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
    monkeypatch.setattr(server_module, "TaskConfig", None)
    monkeypatch.setattr(backends_module, "is_headless_available", lambda: False)
    monkeypatch.setattr(backends_module, "is_blender_available", lambda: False)
    monkeypatch.setattr(backends_module, "UnrealRuntimeAdapter", None)
    settings = Settings(isaac_sim={"enforce_claims": enforce, "discovery_dir": str(tmp_path / "disc")})
    instance = server_module.SimulMCPServer(settings=settings)
    instance.session_manager = SessionManager(tmp_path / "sessions")
    instance._isaac_clients = {"default": FakeIsaacClient(socket_port=8226, bridge_port=8229)}
    instance.client = instance._isaac_clients["default"]
    ctx_var: ContextVar[Any] = ContextVar("ctx", default=None)
    monkeypatch.setattr(instance, "_get_request_context", lambda: ctx_var.get())

    async def fake_execute_script(code: str) -> dict[str, Any]:
        return {"success": True, "output": "ran"}

    async def fake_stage_info(include_prim_count: bool = False) -> dict[str, Any]:
        return {"success": True, "stage": "/World"}

    monkeypatch.setattr(instance._isaac_tools, "execute_script", fake_execute_script)
    monkeypatch.setattr(instance._isaac_tools, "get_isaac_stage_info", fake_stage_info)
    return instance, ctx_var


def _tool(instance: server_module.SimulMCPServer, name: str) -> SimpleNamespace:
    for tool in instance.mcp.tools:
        if tool.name == name:
            return tool
    raise AssertionError(f"Tool {name!r} not registered")


def _payload(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    return json.loads(result.content[0].text)


async def _as(ctx_var: ContextVar[Any], session_id: str, call: Callable[[], Any]) -> Any:
    token = ctx_var.set(SimpleNamespace(session_id=session_id))
    try:
        return await call()
    finally:
        ctx_var.reset(token)


# ---------------------------------------------------------------------------
# Compatibility scoring
# ---------------------------------------------------------------------------


def test_zero_overlap_reason_names_the_existing_purpose(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    manager.get_instance_session(8226).register("agent-a", "RL training run, do not touch")

    compat = manager.score_compatibility("delete every prim under /World", 8226)

    assert compat["compatibility"] == "caution"
    assert compat["score"] == 0.0
    assert "RL training run, do not touch" in compat["reason"]


def test_overlap_still_picks_the_closest_purpose(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    session = manager.get_instance_session(8226)
    session.register("agent-a", "warehouse lighting pass")
    session.register("agent-b", "robot arm grasp training")

    compat = manager.score_compatibility("robot grasp benchmark", 8226)

    assert compat["compatibility"] in {"likely_safe", "compatible"}
    assert "robot arm grasp training" in compat["reason"]


# ---------------------------------------------------------------------------
# Enforcement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_advisory_mode_lets_a_foreign_agent_mutate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    instance, ctx = _make_server(monkeypatch, tmp_path, enforce=False)
    claim = _tool(instance, "claim_isaac_instance").func
    execute = _tool(instance, "execute_isaac_script").func

    await _as(ctx, "agent-a", lambda: claim(purpose="RL training run, do not touch"))
    result = _payload(await _as(ctx, "agent-b", lambda: execute(code="print(1)")))

    assert result["success"] is True
    assert "ADVISORY" in _tool(instance, "claim_isaac_instance").kwargs["description"]
    assert "ADVISORY" in _tool(instance, "release_isaac_instance").kwargs["description"]
    assert "ADVISORY" in _tool(instance, "list_isaac_instances").kwargs["description"]


@pytest.mark.asyncio
async def test_enforced_mode_refuses_foreign_mutation_but_not_reads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    instance, ctx = _make_server(monkeypatch, tmp_path, enforce=True)
    claim = _tool(instance, "claim_isaac_instance").func
    execute = _tool(instance, "execute_isaac_script").func
    stage_info = _tool(instance, "get_isaac_stage_info").func

    holder = _payload(await _as(ctx, "agent-a", lambda: claim(purpose="RL training run, do not touch")))
    assert holder["success"] is True

    refused = _payload(await _as(ctx, "agent-b", lambda: execute(code="print(1)")))
    assert refused["success"] is False
    assert refused["error_type"] == "InstanceClaimed"
    assert refused["details"]["holder_agent_id"] == "agent-a"
    assert refused["details"]["holder_purpose"] == "RL training run, do not touch"
    assert "claim_isaac_instance" in refused["details"]["hint"]
    assert "release_isaac_instance" in refused["details"]["hint"]

    read = _payload(await _as(ctx, "agent-b", lambda: stage_info()))
    assert read["success"] is True

    own = _payload(await _as(ctx, "agent-a", lambda: execute(code="print(2)")))
    assert own["success"] is True

    assert "ENFORCED" in _tool(instance, "claim_isaac_instance").kwargs["description"]
    assert "ENFORCED" in _tool(instance, "release_isaac_instance").kwargs["description"]
    assert "ENFORCED" in _tool(instance, "list_isaac_instances").kwargs["description"]


@pytest.mark.asyncio
async def test_enforced_mode_refuses_a_second_claim_until_release(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    instance, ctx = _make_server(monkeypatch, tmp_path, enforce=True)
    claim = _tool(instance, "claim_isaac_instance").func
    release = _tool(instance, "release_isaac_instance").func
    set_active = _tool(instance, "set_active_isaac_instance").func
    execute = _tool(instance, "execute_isaac_script").func

    await _as(ctx, "agent-a", lambda: claim(purpose="RL training run"))

    second = _payload(await _as(ctx, "agent-b", lambda: claim(purpose="scene cleanup")))
    assert second["error_type"] == "InstanceClaimed"
    via_switch = _payload(
        await _as(ctx, "agent-b", lambda: set_active(instance_name="default", purpose="scene cleanup"))
    )
    assert via_switch["error_type"] == "InstanceClaimed"

    released = _payload(await _as(ctx, "agent-a", lambda: release()))
    assert released["status"] == "released"

    after = _payload(await _as(ctx, "agent-b", lambda: claim(purpose="scene cleanup")))
    assert after["success"] is True
    assert _payload(await _as(ctx, "agent-b", lambda: execute(code="print(1)")))["success"] is True


@pytest.mark.asyncio
async def test_expired_claim_no_longer_blocks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    instance, ctx = _make_server(monkeypatch, tmp_path, enforce=True)
    claim = _tool(instance, "claim_isaac_instance").func
    execute = _tool(instance, "execute_isaac_script").func
    monkeypatch.setattr(session_manager_module, "CLAIM_TTL_SECONDS", 0.05)

    await _as(ctx, "agent-a", lambda: claim(purpose="RL training run"))
    assert _payload(await _as(ctx, "agent-b", lambda: execute(code="print(1)")))["error_type"] == "InstanceClaimed"

    time.sleep(0.1)

    assert _payload(await _as(ctx, "agent-b", lambda: execute(code="print(1)")))["success"] is True


@pytest.mark.asyncio
async def test_release_refuses_someone_elses_claim(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    instance, ctx = _make_server(monkeypatch, tmp_path, enforce=False)
    claim = _tool(instance, "claim_isaac_instance").func
    release = _tool(instance, "release_isaac_instance").func

    await _as(ctx, "agent-a", lambda: claim(purpose="RL training run"))
    await _as(ctx, "agent-b", lambda: claim(purpose="lighting pass"))

    refused = _payload(await _as(ctx, "agent-b", lambda: release(agent_id="agent-a")))
    assert refused["success"] is False
    assert refused["error_type"] == "PermissionError"

    sessions = instance.session_manager.get_instance_session(8226).get_status()["sessions"]
    assert sorted(s["agent_id"] for s in sessions) == ["agent-a", "agent-b"]
