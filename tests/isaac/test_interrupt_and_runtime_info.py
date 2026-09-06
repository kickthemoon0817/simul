"""interrupt_isaac_script and the busy/circuit fields on runtime info and ping."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_path))

from simul_mcp.adapters.isaac_socket_client import ScriptResult  # noqa: E402
from simul_mcp.config import Settings  # noqa: E402
from simul_mcp.mcp import backends as backends_module  # noqa: E402
from simul_mcp.mcp import server as server_module  # noqa: E402
from simul_mcp.mcp.tools.isaac_tools import IsaacTools  # noqa: E402


def _client(*, bridge_enabled: bool = True, circuit_open: bool = False) -> MagicMock:
    client = MagicMock()
    client.address = "127.0.0.1:8229"
    client.bridge_address = "127.0.0.1:8229"
    client.bridge_endpoint = "127.0.0.1:8229"
    client.vscode_address = "127.0.0.1:8226"
    client.timeout_seconds = 30.0
    client.script_timeout_seconds = 29.0
    client.socket_protocol = "python_server"
    client.bridge_enabled = bridge_enabled
    client.fallback_to_vscode = True
    client.bridge_circuit_open = circuit_open
    client.bridge_consecutive_failures = 3 if circuit_open else 0
    return client


def test_interrupt_forwards_the_bridge_payload() -> None:
    client = _client()
    client.interrupt_bridge_script = AsyncMock(
        return_value={
            "status": "ok",
            "payload": {
                "interrupted": True,
                "was_busy": True,
                "phase": "async",
                "current_action": "execute_script",
                "busy_for_seconds": 12.5,
            },
        }
    )

    result = asyncio.run(IsaacTools(client).interrupt_script())

    assert result["success"] is True
    assert result["interrupted"] is True
    assert result["current_action"] == "execute_script"
    client.interrupt_bridge_script.assert_awaited_once()
    # Never through the locked, circuit-guarded request path.
    client.bridge_request.assert_not_called()


def test_interrupt_without_a_bridge_explains_the_timeout_that_applies_instead() -> None:
    client = _client(bridge_enabled=False)

    result = asyncio.run(IsaacTools(client).interrupt_script())

    assert result["success"] is False
    assert result["error_type"] == "BridgeUnavailable"
    assert "29.0s" in result["error"]


def test_interrupt_reports_an_unreachable_bridge() -> None:
    client = _client()
    client.interrupt_bridge_script = AsyncMock(side_effect=ConnectionRefusedError("refused"))

    result = asyncio.run(IsaacTools(client).interrupt_script())

    assert result["success"] is False
    assert result["error_type"] == "ConnectionRefusedError"
    assert "127.0.0.1:8229" in result["error"]
    assert "per-request timeout" in result["error"]


def test_interrupt_surfaces_a_bridge_error_envelope() -> None:
    client = _client()
    client.interrupt_bridge_script = AsyncMock(
        return_value={
            "status": "error",
            "error": {"name": "UnknownAction", "message": "Unsupported bridge action: interrupt"},
        }
    )

    result = asyncio.run(IsaacTools(client).interrupt_script())

    assert result["success"] is False
    assert result["error_type"] == "UnknownAction"


def test_runtime_info_over_the_bridge_carries_bridge_and_client_sections() -> None:
    client = _client(circuit_open=False)
    client.bridge_request = AsyncMock(
        return_value={
            "status": "ok",
            "payload": {
                "transport": "simul_bridge",
                "bridge": {
                    "busy": True,
                    "busy_since": 1700000000.0,
                    "busy_for_seconds": 4.2,
                    "current_action": "simulation_control",
                },
                "app": {"version": "110.1.2"},
            },
        }
    )

    result = asyncio.run(IsaacTools(client).get_runtime_info())

    assert result["success"] is True
    assert result["bridge"]["busy"] is True
    assert result["bridge"]["current_action"] == "simulation_control"
    assert result["client"]["bridge_circuit_open"] is False
    assert result["client"]["bridge_consecutive_failures"] == 0
    assert result["client"]["script_timeout_seconds"] == 29.0


def test_runtime_info_over_the_script_path_reports_the_open_circuit() -> None:
    client = _client(circuit_open=True)
    client.bridge_request = AsyncMock(side_effect=ConnectionRefusedError("circuit open"))
    client.execute = AsyncMock(
        return_value=ScriptResult(success=True, output=json.dumps({"app": {"version": "x"}}))
    )

    result = asyncio.run(IsaacTools(client).get_runtime_info())

    assert result["success"] is True
    assert "bridge" not in result
    assert result["client"]["bridge_circuit_open"] is True
    assert result["client"]["bridge_consecutive_failures"] == 3


# ---------------------------------------------------------------------------
# MCP registration: ping_isaac success tracks reachable, interrupt tool exists
# ---------------------------------------------------------------------------


class FakeFastMCP:
    def __init__(self, name: str, version: str, **kwargs: Any):
        self.name = name
        self.version = version
        self.tools: List[SimpleNamespace] = []

    def tool(self, name: str, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.tools.append(SimpleNamespace(name=name, func=func, kwargs=kwargs))
            return func

        return decorator

    def resource(self, *args: Any, **kwargs: Any) -> Callable[..., Any]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator

    def add_middleware(self, middleware: Any) -> None:
        return


def _make_server(monkeypatch: pytest.MonkeyPatch) -> server_module.SimulMCPServer:
    monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
    monkeypatch.setattr(server_module, "TaskConfig", None)
    monkeypatch.setattr(backends_module, "is_headless_available", lambda: False)
    monkeypatch.setattr(backends_module, "is_blender_available", lambda: False)
    monkeypatch.setattr(backends_module, "UnrealRuntimeAdapter", None)
    return server_module.SimulMCPServer(settings=Settings())


def _tool(instance: server_module.SimulMCPServer, name: str) -> Callable[..., Any]:
    for tool in instance.mcp.tools:
        if tool.name == name:
            return tool.func
    raise AssertionError(f"Tool {name!r} not registered")


def _payload(result: Any) -> Dict[str, Any]:
    return json.loads(result.content[0].text)


@pytest.mark.parametrize("reachable", [True, False])
def test_ping_isaac_success_tracks_reachable(
    monkeypatch: pytest.MonkeyPatch, reachable: bool
) -> None:
    instance = _make_server(monkeypatch)
    instance.client.ping = AsyncMock(return_value=reachable)  # type: ignore[method-assign]

    payload = _payload(asyncio.run(_tool(instance, "ping_isaac")()))

    assert payload["reachable"] is reachable
    assert payload["success"] is reachable
    assert payload["bridge_circuit_open"] is False


def test_interrupt_tool_skips_the_instance_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """The lock is held by the runaway call; the interrupt must not queue behind it."""
    instance = _make_server(monkeypatch)
    instance._instance_lock_timeout = 5.0
    instance.client.interrupt_bridge_script = AsyncMock(  # type: ignore[method-assign]
        return_value={"status": "ok", "payload": {"interrupted": True, "phase": "async"}}
    )

    async def scenario() -> Dict[str, Any]:
        lock = instance._get_instance_lock(instance._get_effective_instance_name())
        async with lock:
            result = await asyncio.wait_for(_tool(instance, "interrupt_isaac_script")(), timeout=1.0)
        return _payload(result)

    payload = asyncio.run(scenario())
    assert payload["success"] is True
    assert payload["interrupted"] is True


def test_list_isaac_instances_reports_the_circuit_state(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = _make_server(monkeypatch)
    instance.client._bridge_circuit_opened_at = 10**12  # far in the future: stays open
    instance.client._bridge_consecutive_failures = 3

    async def _brief(name: str, client: Any) -> Dict[str, Any]:
        return {"name": name, "host": "127.0.0.1", "port": client._port, "reachable": True}

    monkeypatch.setattr(instance, "_get_instance_brief", _brief)

    result = _payload(asyncio.run(_tool(instance, "list_isaac_instances")(scan=False)))

    assert result["instances"][0]["bridge_circuit_open"] is True
