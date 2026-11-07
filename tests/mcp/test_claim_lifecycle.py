"""Claims through the server: failures reported, fail-closed enforcement, reclaim, eviction.

Pinned through real FastMCP clients:

* a claim the store could not save was reported ``success: true``;
* a damaged or unreadable store read as "no claims", so ``enforce_claims``
  let every agent through;
* an agent was locked out by its own claim for up to 120 s after a reconnect
  or a server restart;
* one discovery file that is valid JSON but not an object broke
  ``list_isaac_instances``;
* the rate-limit buckets and Isaac routes of every handshake-era session
  stayed in memory for the life of the server;
* usage records named the session, not the agent label its claim shows.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastmcp import Client

from simul.config import Settings
from simul.mcp import server as server_module
from simul.mcp import session_manager as session_manager_module
from simul.mcp.session_manager import SessionManager
from simul.mcp.usage_tracker import ToolUsageTracker


class _IsaacClient:
    """Reachable Isaac client double."""

    def __init__(self, socket_port: int = 8226, bridge_port: int = 8229) -> None:
        self._host = "127.0.0.1"
        self._port = socket_port
        self._bridge_host = "127.0.0.1"
        self._bridge_port = bridge_port
        self._bridge_configured = True

    @property
    def address(self) -> str:
        return f"{self._bridge_host}:{self._bridge_port}"

    bridge_address = address

    @property
    def vscode_address(self) -> str:
        return f"{self._host}:{self._port}"

    @property
    def bridge_circuit_open(self) -> bool:
        return False

    @property
    def bridge_enabled(self) -> bool:
        return False

    @property
    def fallback_to_vscode(self) -> bool:
        return False

    @property
    def timeout_seconds(self) -> float:
        return 5.0

    async def ping(self) -> bool:
        return True


def _server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, enforce: bool = True
) -> server_module.SimulMCPServer:
    instance = server_module.SimulMCPServer(
        Settings(
            isaac_sim={"discovery_dir": str(tmp_path / "discovery"), "enforce_claims": enforce}
        ),
        backends={"isaac"},
    )
    instance.session_manager = SessionManager(tmp_path / "sessions")
    instance.usage_tracker = ToolUsageTracker(log_dir=tmp_path / "usage")
    instance._isaac_clients = {"default": _IsaacClient()}
    instance.client = instance._isaac_clients["default"]

    async def execute_script(code: str) -> Dict[str, Any]:
        return {"success": True, "output": "ran"}

    async def stage_info(include_prim_count: bool = False) -> Dict[str, Any]:
        return {"success": True, "stage": "/World"}

    monkeypatch.setattr(instance._isaac_tools, "execute_script", execute_script)
    monkeypatch.setattr(instance._isaac_tools, "get_isaac_stage_info", stage_info)
    return instance


def _payload(result: Any) -> Dict[str, Any]:
    return json.loads(result.content[-1].text)


async def _call(client: Client, tool: str, **arguments: Any) -> Dict[str, Any]:
    return _payload(await client.call_tool(tool, arguments, raise_on_error=False))


def _seed_claim(server: server_module.SimulMCPServer, agent_id: str, **fields: Any) -> None:
    """Write a claim by another process straight into the store."""
    store = server.session_manager.get_instance_session(8226)
    store.register(agent_id, "RL training run", owner="another-session")
    path = server.session_manager.session_dir / "8226.json"
    document = json.loads(path.read_text())
    document["sessions"][0].update(fields)
    path.write_text(json.dumps(document))


# ---------------------------------------------------------------------------
# A claim that was not saved is not reported as made (M12)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores modes")
@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("claim_isaac_instance", {"purpose": "x"}),
        ("set_active_isaac_instance", {"instance_name": "default", "purpose": "x"}),
    ],
)
def test_a_claim_the_store_cannot_save_is_reported_as_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool: str, arguments: Dict[str, Any]
) -> None:
    server = _server(tmp_path, monkeypatch, enforce=False)
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    sessions.chmod(0o500)

    async def run() -> List[Dict[str, Any]]:
        async with Client(server.mcp, mode="legacy") as client:
            return [
                await _call(client, tool, **arguments),
                await _call(client, "list_isaac_instances", scan=False),
            ]

    try:
        result, listing = asyncio.run(run())
    finally:
        sessions.chmod(0o700)

    assert result["success"] is False
    assert result["error_type"] == "ClaimStoreError"
    assert "not saved" in result["error"]
    assert "binding" not in result
    assert listing["instances"][0]["sessions"] == []


# ---------------------------------------------------------------------------
# Enforcement fails closed (B2)
# ---------------------------------------------------------------------------


def test_a_damaged_store_refuses_mutating_calls_until_the_holder_reasserts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A corrupt file used to read as "no claims": every agent could mutate."""
    server = _server(tmp_path, monkeypatch)

    async def run() -> Dict[str, Any]:
        async with Client(server.mcp, mode="legacy") as holder, Client(
            server.mcp, mode="legacy"
        ) as other:
            out: Dict[str, Any] = {}
            out["claim"] = await _call(holder, "claim_isaac_instance", purpose="RL", agent_id="trainer")
            (tmp_path / "sessions" / "8226.json").write_text('{"port": 8226, "sess')
            with caplog.at_level(logging.WARNING):
                out["other"] = await _call(other, "execute_isaac_script", code="print(1)")
            out["other_read"] = await _call(other, "get_isaac_stage_info")
            out["other_claim"] = await _call(other, "claim_isaac_instance", purpose="cleanup")
            out["holder"] = await _call(holder, "execute_isaac_script", code="print(2)")
            out["records"] = server.session_manager.get_instance_session(8226).view().sessions
            out["other_after"] = await _call(other, "execute_isaac_script", code="print(3)")
            return out

    out = asyncio.run(run())

    assert out["claim"]["success"] is True
    assert out["other"]["error_type"] == "InstanceClaimed"
    assert out["other"]["details"]["holder_agent_id"] is None
    assert "repaired" in out["other"]["error"]
    assert "moved to" in caplog.text
    assert out["other_read"]["success"] is True, "read-only tools are never blocked"
    assert out["other_claim"]["error_type"] == "InstanceClaimed"
    # The holder's binding shows it held the claim; its record is re-asserted.
    assert out["holder"]["success"] is True
    assert [r["agent_id"] for r in out["records"]] == ["trainer"]
    assert out["other_after"]["error_type"] == "InstanceClaimed"
    assert out["other_after"]["details"]["holder_agent_id"] == "trainer"


def test_the_repair_window_closes_after_one_ttl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _server(tmp_path, monkeypatch)
    (tmp_path / "sessions").mkdir()
    (tmp_path / "sessions" / "8226.json").write_text("garbage")
    monkeypatch.setattr(session_manager_module, "CLAIM_TTL_SECONDS", 0.2)

    async def run() -> List[Dict[str, Any]]:
        async with Client(server.mcp, mode="legacy") as client:
            refused = await _call(client, "execute_isaac_script", code="print(1)")
            await asyncio.sleep(0.3)
            return [refused, await _call(client, "execute_isaac_script", code="print(1)")]

    refused, admitted = asyncio.run(run())

    assert refused["error_type"] == "InstanceClaimed"
    assert admitted["success"] is True


def test_an_unreadable_store_refuses_mutating_calls_with_claim_store_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _server(tmp_path, monkeypatch)
    (tmp_path / "sessions" / "8226.json").mkdir(parents=True)  # a directory cannot be read as a file

    async def run() -> List[Dict[str, Any]]:
        async with Client(server.mcp, mode="legacy") as client:
            return [
                await _call(client, "execute_isaac_script", code="print(1)"),
                await _call(client, "get_isaac_stage_info"),
                await _call(client, "list_isaac_instances", scan=False),
            ]

    mutating, read, listing = asyncio.run(run())

    assert mutating["success"] is False
    assert mutating["error_type"] == "ClaimStoreError"
    assert read["success"] is True
    (entry,) = listing["instances"]
    assert entry["claims_error"]
    assert entry["compatibility"] == "blocked"


def test_advisory_mode_is_not_blocked_by_a_damaged_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _server(tmp_path, monkeypatch, enforce=False)
    (tmp_path / "sessions").mkdir()
    (tmp_path / "sessions" / "8226.json").write_text("garbage")

    async def run() -> Dict[str, Any]:
        async with Client(server.mcp, mode="legacy") as client:
            return await _call(client, "execute_isaac_script", code="print(1)")

    assert asyncio.run(run())["success"] is True


# ---------------------------------------------------------------------------
# Reclaiming one's own claim after a reconnect or a restart (m4)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="no side-effect-free pid probe")
def test_a_claim_whose_server_exited_can_be_reclaimed_by_its_agent_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _server(tmp_path, monkeypatch)
    _seed_claim(server, "robot-1", pid=2**22 + 12345)  # no such process

    async def run() -> List[Dict[str, Any]]:
        async with Client(server.mcp, mode="legacy") as client:
            return [
                await _call(client, "execute_isaac_script", code="print(1)"),
                await _call(client, "claim_isaac_instance", purpose="cleanup", agent_id="robot-2"),
                await _call(client, "claim_isaac_instance", purpose="RL", agent_id="robot-1"),
                await _call(client, "execute_isaac_script", code="print(1)"),
            ]

    other, impostor, reclaimed, ran = asyncio.run(run())

    # Another agent still waits: the reconnecting agent keeps its instance.
    assert other["error_type"] == "InstanceClaimed"
    assert other["details"]["holder_orphaned"] is True
    assert "takes the claim over now" in other["details"]["hint"]
    assert impostor["error_type"] == "InstanceClaimed"
    assert reclaimed["success"] is True
    assert reclaimed["session"]["status"] == "reclaimed"
    assert ran["success"] is True


def test_a_live_claim_of_another_session_is_still_not_reclaimable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _server(tmp_path, monkeypatch)
    _seed_claim(server, "robot-1")  # written by this (live) process

    async def run() -> List[Dict[str, Any]]:
        async with Client(server.mcp, mode="legacy") as client:
            return [
                await _call(client, "claim_isaac_instance", purpose="RL", agent_id="robot-1"),
                await _call(client, "release_isaac_instance", agent_id="robot-1"),
            ]

    claimed, released = asyncio.run(run())

    assert claimed["error_type"] == "InstanceClaimed"
    assert "bound to a different MCP session" in claimed["error"]
    assert released["error_type"] == "PermissionError"
    assert len(server.session_manager.get_instance_session(8226).view().sessions) == 1


def test_an_ended_session_leaves_an_orphaned_claim_its_agent_reclaims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _server(tmp_path, monkeypatch)

    async def run() -> List[Dict[str, Any]]:
        async with Client(server.mcp, mode="legacy") as first:
            claimed = await _call(first, "claim_isaac_instance", purpose="RL", agent_id="robot-1")
        records = server.session_manager.get_instance_session(8226).view().sessions
        async with Client(server.mcp, mode="legacy") as second:
            other = await _call(second, "execute_isaac_script", code="print(1)")
            reclaimed = await _call(second, "claim_isaac_instance", purpose="RL", agent_id="robot-1")
            ran = await _call(second, "execute_isaac_script", code="print(1)")
        return [claimed, {"records": records}, other, reclaimed, ran]

    claimed, seen, other, reclaimed, ran = asyncio.run(run())

    assert claimed["success"] is True
    assert seen["records"][0]["orphaned"] is True
    assert other["error_type"] == "InstanceClaimed"
    assert reclaimed["session"]["status"] == "reclaimed"
    assert ran["success"] is True


def test_an_orphaned_claim_can_be_released_by_its_agent_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _server(tmp_path, monkeypatch)
    _seed_claim(server, "robot-1", orphaned=True)

    async def run() -> Dict[str, Any]:
        async with Client(server.mcp, mode="legacy") as client:
            return await _call(client, "release_isaac_instance", agent_id="robot-1")

    released = asyncio.run(run())

    assert released["success"] is True
    assert released["status"] == "released"
    assert server.session_manager.get_instance_session(8226).view().sessions == []


# ---------------------------------------------------------------------------
# A discovery file that is not an object (m5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("content", ["[1, 2, 3]", '"text"', "42", "null"])
def test_a_discovery_file_that_is_not_an_object_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, content: str
) -> None:
    server = _server(tmp_path, monkeypatch)
    discovery = tmp_path / "discovery"
    discovery.mkdir(mode=0o700)
    (discovery / "evil.json").write_text(content)

    async def run() -> Dict[str, Any]:
        return await server._discover_from_files()

    with caplog.at_level(logging.WARNING):
        assert asyncio.run(run()) == {}
    assert "expected a JSON object" in caplog.text


# ---------------------------------------------------------------------------
# Per-session state is dropped when the session ends (m6)
# ---------------------------------------------------------------------------


def test_handshake_sessions_leave_no_state_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _server(tmp_path, monkeypatch, enforce=False)

    async def run() -> None:
        for _ in range(30):
            async with Client(server.mcp, mode="legacy") as client:
                await _call(client, "get_isaac_stage_info")
                await _call(client, "set_active_isaac_instance", instance_name="default")

    asyncio.run(run())

    assert server._rate_limiters == {}
    assert server._session_routes == {}
    assert server._watched_sessions == set()


def test_idle_rate_limit_buckets_are_swept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _server(tmp_path, monkeypatch, enforce=False)
    monkeypatch.setattr(server_module, "_RATE_LIMITER_SWEEP_SIZE", 10)
    for index in range(10):
        server._get_rate_limiter("tool", f"agent-{index}")
    spent = server._get_rate_limiter("tool", "agent-0")
    spent.acquire()

    server._get_rate_limiter("tool", "new-agent")

    assert set(server._rate_limiters) == {("agent-0", "tool"), ("new-agent", "tool")}


# ---------------------------------------------------------------------------
# Usage records name the claim's agent label (m16)
# ---------------------------------------------------------------------------


def test_usage_records_name_the_claims_agent_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _server(tmp_path, monkeypatch, enforce=False)

    async def run() -> Dict[str, Any]:
        async with Client(server.mcp, mode="legacy") as client:
            await _call(client, "get_isaac_stage_info")
            await _call(client, "claim_isaac_instance", purpose="x", agent_id="robot-1")
            await _call(client, "get_isaac_stage_info")
            return await _call(client, "list_isaac_instances", scan=False)

    listing = asyncio.run(run())

    holders = [s["agent_id"] for s in listing["instances"][0]["sessions"]]
    agents = [r["agent_id"] for r in reversed(server.usage_tracker.get_recent(tool_name="get_isaac_stage_info"))]
    assert holders == ["robot-1"]
    assert agents[0].startswith("session-")
    assert agents[1] == "robot-1"
    published = listing["instances"][0]["sessions"][0]
    assert "pid" not in published and "host" not in published
    assert published["orphaned"] is False


def test_claim_expiry_still_applies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = _server(tmp_path, monkeypatch)
    monkeypatch.setattr(session_manager_module, "CLAIM_TTL_SECONDS", 0.1)
    _seed_claim(server, "robot-1")
    time.sleep(0.15)

    async def run() -> Dict[str, Any]:
        async with Client(server.mcp, mode="legacy") as client:
            return await _call(client, "execute_isaac_script", code="print(1)")

    assert asyncio.run(run())["success"] is True
