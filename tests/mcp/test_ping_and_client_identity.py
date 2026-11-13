"""ping_isaac answers during a running call; anonymous HTTP clients get their own budget.

* ``IsaacSocketClient.ping`` took the request lock, so a ping waited behind a
  running script for as long as it ran (7.8 s in the audit), although
  ping_isaac skips the instance lock precisely so it can answer while busy.
* Every unauthenticated sessionless HTTP client was the one agent
  ``anonymous`` with one rate-limit budget: three clients making 8 calls each
  had half their calls refused.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from simul.adapters.isaac_socket_client import IsaacSocketClient
from simul.config import Settings
from simul.mcp import server as server_module


async def _pong_server() -> Any:
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.read(65536)
        writer.write(json.dumps({"status": "ok", "output": "pong\n"}).encode())
        await writer.drain()
        writer.close()

    return await asyncio.start_server(handle, "127.0.0.1", 0)


def test_ping_does_not_wait_for_a_call_in_flight() -> None:
    async def scenario() -> float:
        server = await _pong_server()
        port = server.sockets[0].getsockname()[1]
        client = IsaacSocketClient(port=port, timeout_seconds=5, socket_protocol="vscode")
        async with server:
            async with client._lock:  # a script is running on this client
                assert client.busy
                started = asyncio.get_running_loop().time()
                assert await asyncio.wait_for(client.ping(), timeout=2.0) is True
                return asyncio.get_running_loop().time() - started

    assert asyncio.run(scenario()) < 1.0


def test_ping_isaac_reports_a_busy_instance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    instance = server_module.SimulMCPServer(
        Settings(isaac_sim={"discovery_dir": str(tmp_path / "disc")}), backends={"isaac"}
    )

    async def reachable() -> bool:
        return False

    monkeypatch.setattr(instance.client, "ping", reachable)

    async def scenario() -> Dict[str, Any]:
        async with instance._get_instance_lock("default"):
            result = await instance._exec_isaac(
                "ping_isaac",
                __import__("simul.mcp.registration._reg_isaac", fromlist=["_ping"])._ping(instance),
                bypass_instance_lock=True,
            )
        return json.loads(result.content[-1].text)

    payload = asyncio.run(scenario())

    assert payload["busy"] is True
    assert payload["error_type"] == "ConnectionError"
    assert "synchronous script" in payload["error"]


# ---------------------------------------------------------------------------
# M14: each anonymous HTTP client is named by its TCP peer
# ---------------------------------------------------------------------------


def _anonymous_request(
    monkeypatch: pytest.MonkeyPatch,
    instance: server_module.SimulMCPServer,
    peer: Optional[tuple],
    headers: Optional[Dict[str, str]] = None,
) -> None:
    """Make the server see an unauthenticated sessionless HTTP request from ``peer``."""
    from fastmcp.server import dependencies

    monkeypatch.setattr(instance, "_get_request_context", lambda: SimpleNamespace())
    monkeypatch.setattr(instance, "_get_request_session_id", lambda: None)
    client = SimpleNamespace(host=peer[0], port=peer[1]) if peer else None
    request = SimpleNamespace(client=client, headers=headers or {})
    monkeypatch.setattr(dependencies, "get_http_request", lambda: request)


def test_each_anonymous_peer_has_its_own_agent_and_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    instance = server_module.SimulMCPServer(
        Settings(security={"burst_size": 2}, isaac_sim={"discovery_dir": str(tmp_path)}),
        backends={"usd"},
    )
    verdicts: Dict[str, List[bool]] = {}
    for port in (50001, 50002, 50003):
        _anonymous_request(monkeypatch, instance, ("127.0.0.1", port))
        agent = instance._resolve_agent_id(None)
        verdicts[agent] = [instance._check_rate_limit("tool") is None for _ in range(3)]

    assert verdicts == {
        f"anonymous@127.0.0.1:{port}": [True, True, False] for port in (50001, 50002, 50003)
    }


def test_the_client_label_names_the_agent_in_records_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    instance = server_module.SimulMCPServer(
        Settings(isaac_sim={"discovery_dir": str(tmp_path)}), backends={"usd"}
    )
    _anonymous_request(
        monkeypatch, instance, ("10.0.0.5", 40000), headers={"x-simul-agent": "robot 1\x00"}
    )

    session = instance._resolve_agent_id(None)
    assert session == "anonymous@10.0.0.5:40000"
    assert instance._recorded_agent_id(session) == "anonymous:robot1"
    # Still anonymous for everything that keeps per-client state.
    assert instance._request_is_anonymous()
    assert instance._session_required_error("claim_isaac_instance")["error_type"] == "SessionRequired"


def test_without_an_http_request_the_agent_is_plain_anonymous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    instance = server_module.SimulMCPServer(
        Settings(isaac_sim={"discovery_dir": str(tmp_path)}), backends={"usd"}
    )
    from fastmcp.server import dependencies

    def no_request() -> Any:
        raise RuntimeError("no active HTTP request")

    monkeypatch.setattr(instance, "_get_request_context", lambda: SimpleNamespace())
    monkeypatch.setattr(instance, "_get_request_session_id", lambda: None)
    monkeypatch.setattr(dependencies, "get_http_request", no_request)

    assert instance._resolve_agent_id(None) == "anonymous"
