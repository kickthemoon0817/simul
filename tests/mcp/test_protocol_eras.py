"""Per-session state on both MCP protocol eras a FastMCP 4 server speaks.

A client that connects with the initialize handshake (protocol 2025-11-25 and
earlier) gets one session per connection. The 2026-07-28 protocol, which
``fastmcp.Client`` negotiates by default, has no sessions, and FastMCP hands
each of its requests a fresh session id. Keyed on that id, every call would be
a new agent: a new rate-limit bucket pair and Isaac route that are never used
again, and an Isaac instance switch or claim forgotten by the next call. Those
requests share one session per server instead, which these tests pin through
real FastMCP clients of both eras.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
from fastmcp import Client

from simul.config import Settings
from simul.mcp import server as server_module
from simul.mcp.session_manager import SessionManager
from simul.mcp.usage_tracker import ToolUsageTracker

SESSIONLESS_PROTOCOL = "2026-07-28"


class _IsaacClient:
    """Reachable Isaac client double; ``ping_isaac`` reports its address."""

    def __init__(self, socket_port: int, bridge_port: int) -> None:
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
    def timeout_seconds(self) -> float:
        return 5.0

    async def ping(self) -> bool:
        return True


DEFAULT = _IsaacClient(8226, 8229)
SECOND = _IsaacClient(8326, 8329)


@pytest.fixture
def server(tmp_path: Path) -> server_module.SimulMCPServer:
    """A real FastMCP server with the Isaac tools over two reachable instances."""
    instance = server_module.SimulMCPServer(
        Settings(isaac_sim={"discovery_dir": str(tmp_path / "discovery")}),
        backends={"isaac"},
    )
    instance.session_manager = SessionManager(tmp_path / "sessions")
    instance.usage_tracker = ToolUsageTracker(log_dir=tmp_path / "usage")
    instance._isaac_clients = {"default": DEFAULT, "second": SECOND}
    instance.client = DEFAULT
    return instance


Call = Tuple[str, Dict[str, Any]]


def _connect(
    instance: server_module.SimulMCPServer, mode: str, *connections: List[Call]
) -> Tuple[List[str], List[List[Dict[str, Any]]]]:
    """Run each list of calls on its own client connection, one connection after another.

    Returns:
        The protocol each connection negotiated, and each connection's payloads.
    """

    async def run() -> Tuple[List[str], List[List[Dict[str, Any]]]]:
        protocols: List[str] = []
        payloads: List[List[Dict[str, Any]]] = []
        for calls in connections:
            async with Client(instance.mcp, mode=mode) as client:
                protocols.append(str(client.protocol_version))
                results = [
                    await client.call_tool(name, arguments, raise_on_error=False)
                    for name, arguments in calls
                ]
                payloads.append(
                    [
                        (
                            {"is_error": True}
                            if result.is_error
                            else json.loads(result.content[-1].text)
                        )
                        for result in results
                    ]
                )
        return protocols, payloads

    return asyncio.run(run())


def _agents(instance: server_module.SimulMCPServer) -> List[str]:
    """The agent of each recorded call, oldest first."""
    return [
        row["agent_id"]
        for row in reversed(instance.usage_tracker.get_recent(limit=1000))
    ]


PING: Call = ("ping_isaac", {})


def test_each_handshake_connection_is_its_own_session(
    server: server_module.SimulMCPServer,
) -> None:
    protocols, payloads = _connect(
        server,
        "legacy",
        [("set_active_isaac_instance", {"instance_name": "second"}), PING, PING],
        [PING],
    )

    assert SESSIONLESS_PROTOCOL not in protocols
    first, second, other = _agents(server)
    assert first == second != other
    assert server._sessionless_session_id not in (first, other)
    # The switch belongs to the session that made it.
    assert [payload["address"] for payload in payloads[0][1:]] == [SECOND.address] * 2
    assert payloads[1][0]["address"] == DEFAULT.address


def test_sessionless_requests_share_one_session_per_server(
    server: server_module.SimulMCPServer,
) -> None:
    protocols, _ = _connect(server, "auto", [PING, PING], [PING])

    assert protocols == [SESSIONLESS_PROTOCOL] * 2
    assert _agents(server) == [server._sessionless_session_id] * 3


def test_an_instance_switch_and_claim_last_across_sessionless_calls(
    server: server_module.SimulMCPServer,
) -> None:
    _, payloads = _connect(
        server,
        "auto",
        [
            (
                "set_active_isaac_instance",
                {"instance_name": "second", "purpose": "probe"},
            ),
            PING,
            ("release_isaac_instance", {}),
        ],
    )

    switched, pinged, released = payloads[0]
    assert switched["binding"]["agent_id"] == server._sessionless_session_id
    assert pinged["address"] == SECOND.address
    assert released["success"] is True
    assert released["binding_released"] == switched["binding"]["binding_id"]


def test_sessionless_calls_add_no_state_per_call(
    server: server_module.SimulMCPServer,
) -> None:
    _connect(server, "auto", [PING] * 3, [PING] * 3)

    session = server._sessionless_session_id
    assert set(server._rate_limiters) == {(session, "ping_isaac"), (session, "*")}
    assert set(server._session_routes) == {session}


@pytest.mark.parametrize("mode", ["legacy", "auto"])
def test_each_tool_call_leaves_one_audit_record(
    server: server_module.SimulMCPServer, mode: str
) -> None:
    """FastMCP 4 runs middleware for every inbound message; the audit hook counts calls only."""
    records: List[logging.LogRecord] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    audit_logger = logging.getLogger("simul.audit")
    handler = _Collect(level=logging.INFO)
    previous_level = audit_logger.level
    audit_logger.addHandler(handler)
    audit_logger.setLevel(logging.INFO)
    try:
        _connect(server, mode, [PING, ("no_such_tool", {}), PING])
    finally:
        audit_logger.removeHandler(handler)
        audit_logger.setLevel(previous_level)

    audits = [record.audit for record in records if hasattr(record, "audit")]
    assert [audit["tool"] for audit in audits] == [
        "ping_isaac",
        "no_such_tool",
        "ping_isaac",
    ]
