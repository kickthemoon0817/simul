"""Per-session state on both MCP protocol eras a FastMCP 4 server speaks.

A client that connects with the initialize handshake (protocol 2025-11-25 and
earlier) gets one session per connection. The 2026-07-28 protocol, which
``fastmcp.Client`` negotiates by default, has no sessions, and FastMCP hands
each of its requests a fresh session id. Keyed on that id, every call would be
a new agent: a new rate-limit bucket pair and Isaac route that are never used
again, and an Isaac instance switch or claim forgotten by the next call.

Those requests are scoped the way FastMCP scopes its own per-client state: an
authenticated request to its user; over stdio (in memory here), where the
server serves one client, to one session per server; and over unauthenticated
HTTP to no session at all, because nothing tells one such client from another
and sharing a session would let one client switch, claim or release for the
others. These tests pin all of it through real FastMCP clients of both eras.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest
from fastmcp import Client
from fastmcp.utilities.tests import asgi_server

from simul.config import Settings
from simul.mcp import server as server_module
from simul.mcp.session_manager import SessionManager
from simul.mcp.usage_tracker import ToolUsageTracker

SESSIONLESS_PROTOCOL = "2026-07-28"
HANDSHAKE_PROTOCOL = "2025-11-25"


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


def _isaac_server(
    tmp_path: Path, enforce_claims: bool = False
) -> server_module.SimulMCPServer:
    """A real FastMCP server with the Isaac tools over two reachable instances."""
    instance = server_module.SimulMCPServer(
        Settings(
            isaac_sim={
                "discovery_dir": str(tmp_path / "discovery"),
                "enforce_claims": enforce_claims,
            }
        ),
        backends={"isaac"},
    )
    instance.session_manager = SessionManager(tmp_path / "sessions")
    instance.usage_tracker = ToolUsageTracker(log_dir=tmp_path / "usage")
    instance._isaac_clients = {"default": DEFAULT, "second": SECOND}
    instance.client = DEFAULT
    return instance


@pytest.fixture
def server(tmp_path: Path) -> server_module.SimulMCPServer:
    return _isaac_server(tmp_path)


@pytest.fixture
def claim_server(tmp_path: Path) -> server_module.SimulMCPServer:
    """The same server with isaac_sim.enforce_claims on."""
    return _isaac_server(tmp_path, enforce_claims=True)


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


def test_sessionless_requests_share_one_session_per_stdio_server(
    server: server_module.SimulMCPServer,
) -> None:
    protocols, _ = _connect(server, "auto", [PING, PING], [PING])

    assert protocols == [SESSIONLESS_PROTOCOL] * 2
    assert _agents(server) == [
        server_module.default_agent_label(server._sessionless_session_id)
    ] * 3


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
    assert switched["binding"]["agent_id"] == server_module.default_agent_label(
        server._sessionless_session_id
    )
    assert pinged["address"] == SECOND.address
    assert released["success"] is True
    assert released["binding_released"] == switched["binding"]["binding_id"]


def test_sessionless_calls_add_no_state_per_call(
    server: server_module.SimulMCPServer,
) -> None:
    _connect(server, "auto", [PING] * 3, [PING] * 3)

    session = server._sessionless_session_id
    label = server_module.default_agent_label(session)
    assert set(server._rate_limiters) == {(label, "ping_isaac"), (label, "*")}
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


# -- Over HTTP ---------------------------------------------------------------
#
# The server's real streamable HTTP app, served in-process by FastMCP's test
# helper: the same session manager and request handling as `--transport http`.


def _payload(result: Any) -> Dict[str, Any]:
    return json.loads(result.content[-1].text)


SESSION_TOOLS: List[Call] = [
    ("set_active_isaac_instance", {"instance_name": "second"}),
    ("claim_isaac_instance", {"purpose": "probe"}),
    ("release_isaac_instance", {}),
]


def test_sessionless_http_clients_get_no_session_to_share(
    server: server_module.SimulMCPServer,
) -> None:
    """Nothing tells unauthenticated sessionless HTTP clients apart, so none keeps per-client state."""

    async def run() -> Tuple[List[str], List[List[Dict[str, Any]]]]:
        async with asgi_server(server.mcp) as http:
            async with (
                http.client(mode="auto") as alice,
                http.client(mode="auto") as bob,
            ):
                refused = [
                    _payload(
                        await alice.call_tool(name, arguments, raise_on_error=False)
                    )
                    for name, arguments in SESSION_TOOLS
                ]
                pings = [
                    _payload(await client.call_tool(*PING))
                    for client in (alice, bob, alice)
                ]
                protocols = [str(alice.protocol_version), str(bob.protocol_version)]
                return protocols, [refused, pings]

    protocols, (refused, pings) = asyncio.run(run())

    assert protocols == [SESSIONLESS_PROTOCOL] * 2
    assert [payload["error_type"] for payload in refused] == ["SessionRequired"] * 3
    assert refused[0]["details"]["active_instance"] == "default"
    assert 'mode="legacy"' in refused[0]["details"]["hint"]
    # Every call reaches the server's default instance as the one anonymous agent.
    assert [payload["address"] for payload in pings] == [DEFAULT.address] * 3
    assert _agents(server) == [server_module._ANONYMOUS_AGENT] * 3
    assert server._session_routes == {}
    assert set(server._rate_limiters) == {
        (server_module._ANONYMOUS_AGENT, "ping_isaac"),
        (server_module._ANONYMOUS_AGENT, "*"),
    }
    assert server.session_manager.get_all_sessions() == []


def test_handshake_http_clients_keep_their_own_sessions(
    server: server_module.SimulMCPServer,
) -> None:
    async def run() -> List[Dict[str, Any]]:
        async with asgi_server(server.mcp) as http:
            async with (
                http.client(mode="legacy") as alice,
                http.client(mode="legacy") as bob,
            ):
                assert (
                    alice.protocol_version == bob.protocol_version == HANDSHAKE_PROTOCOL
                )
                await alice.call_tool(
                    "set_active_isaac_instance", {"instance_name": "second"}
                )
                return [
                    _payload(await client.call_tool(*PING)) for client in (alice, bob)
                ]

    alice_ping, bob_ping = asyncio.run(run())

    assert alice_ping["address"] == SECOND.address
    assert bob_ping["address"] == DEFAULT.address
    alice_agent, bob_agent = _agents(server)
    assert alice_agent != bob_agent


def test_a_sessionless_http_client_cannot_take_a_handshake_clients_claim(
    claim_server: server_module.SimulMCPServer,
) -> None:
    """The #202 guarantee holds across eras: a label is not an identity."""

    async def run() -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        async with asgi_server(claim_server.mcp) as http:
            async with (
                http.client(mode="legacy") as alice,
                http.client(mode="auto") as bob,
            ):
                claimed = _payload(
                    await alice.call_tool(
                        "set_active_isaac_instance",
                        {
                            "instance_name": "default",
                            "purpose": "alice builds",
                            "agent_id": "alice",
                        },
                    )
                )
                attempts = [
                    _payload(
                        await bob.call_tool(name, {**arguments, "agent_id": "alice"})
                    )
                    for name, arguments in SESSION_TOOLS
                ]
                attempts.append(
                    _payload(
                        await bob.call_tool(
                            "create_isaac_object",
                            {"prim_path": "/World/Bob"},
                            raise_on_error=False,
                        )
                    )
                )
                return claimed, attempts

    claimed, attempts = asyncio.run(run())

    assert claimed["success"] is True
    assert [payload["error_type"] for payload in attempts] == [
        "SessionRequired",
        "SessionRequired",
        "SessionRequired",
        "InstanceClaimed",
    ]
    records = claim_server.session_manager.get_instance_session(
        DEFAULT._port
    ).get_status()
    assert [
        (record["agent_id"], record["purpose"]) for record in records["sessions"]
    ] == [("alice", "alice builds")]


def test_a_session_header_gives_a_sessionless_request_no_session(
    server: server_module.SimulMCPServer,
) -> None:
    """FastMCP reads a sessionless request's session id from Mcp-Session-Id; simul never uses it."""

    async def run() -> Tuple[Dict[str, Any], Dict[str, Any]]:
        async with asgi_server(server.mcp) as http:
            async with http.client(mode="legacy") as alice:
                await alice.call_tool(
                    "set_active_isaac_instance", {"instance_name": "second"}
                )
                (alice_session,) = server._session_routes
                async with http.client(
                    mode="auto", headers={"Mcp-Session-Id": alice_session}
                ) as bob:
                    assert bob.protocol_version == SESSIONLESS_PROTOCOL
                    ping = _payload(await bob.call_tool(*PING))
                    switch = _payload(await bob.call_tool(*SESSION_TOOLS[0]))
                    return ping, switch

    ping, switch = asyncio.run(run())

    assert ping["address"] == DEFAULT.address
    assert switch["error_type"] == "SessionRequired"


def test_authenticated_sessionless_requests_are_scoped_to_their_user(
    server: server_module.SimulMCPServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With auth, FastMCP's per-user identity tells sessionless HTTP clients apart."""
    principal: Dict[str, Optional[str]] = {"current": None}
    monkeypatch.setattr(
        server_module, "_request_principal", lambda: principal["current"]
    )

    async def run() -> List[Dict[str, Any]]:
        async with asgi_server(server.mcp) as http:
            async with http.client(mode="auto") as client:
                principal["current"] = '["alice-client","issuer","alice"]'
                await client.call_tool(
                    "set_active_isaac_instance", {"instance_name": "second"}
                )
                alice = _payload(await client.call_tool(*PING))
                principal["current"] = '["bob-client","issuer","bob"]'
                bob = _payload(await client.call_tool(*PING))
                return [alice, bob]

    alice, bob = asyncio.run(run())

    assert alice["address"] == SECOND.address
    assert bob["address"] == DEFAULT.address
    alice_agent, bob_agent = _agents(server)
    assert alice_agent != bob_agent
    # The principal itself never becomes a label, a session record or a key.
    for agent in (alice_agent, bob_agent):
        assert re.fullmatch(r"session-[0-9a-f]{12}", agent), agent


def test_the_default_agent_label_does_not_reveal_the_session_id(
    claim_server: server_module.SimulMCPServer,
) -> None:
    """Over HTTP the session id is what a client presents to act in that session.

    list_isaac_instances publishes every instance's session records, claim
    holders included, so the default agent label must not be the session id.
    """

    async def run() -> Tuple[str, Dict[str, Any]]:
        async with asgi_server(claim_server.mcp) as http:
            async with http.client(mode="legacy") as alice:
                await alice.call_tool(
                    "set_active_isaac_instance", {"instance_name": "second"}
                )
                (alice_session,) = claim_server._session_routes
                claimed = _payload(
                    await alice.call_tool("claim_isaac_instance", {"purpose": "label test"})
                )
                return alice_session, claimed

    alice_session, claimed = asyncio.run(run())

    published = claim_server.session_manager.get_instance_session(SECOND._port).get_status()
    label = claimed["binding"]["agent_id"]
    assert re.fullmatch(r"session-[0-9a-f]{12}", label), label
    assert label in json.dumps(published)
    assert alice_session not in json.dumps(published)
