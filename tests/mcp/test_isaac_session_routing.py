"""Tests for per-session Isaac routing and same-instance serialization."""

from __future__ import annotations

import asyncio
import json
import sys
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, List

import pytest

src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_path))

from simul_mcp.config import Settings  # noqa: E402
from simul_mcp.mcp import server as server_module  # noqa: E402


class FakeFastMCP:
    """Minimal FastMCP double that records registered tools."""

    def __init__(self, name: str, version: str, **kwargs: Any):
        self.name = name
        self.version = version
        self.instructions = kwargs.get("instructions")
        self.tools: List[SimpleNamespace] = []

    def tool(
        self, name: str, **kwargs: Any
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.tools.append(SimpleNamespace(name=name, func=func, kwargs=kwargs))
            return func

        return decorator

    def resource(self, *args, **kwargs):
        """Stub for resource registration."""
        def decorator(func):
            return func
        return decorator

    def add_middleware(self, middleware: Any) -> None:
        """Stub for FastMCP middleware registration.

        SimulMCPServer adds a request-context middleware (PR #23)
        before any tools register. The stub only needs to not raise.
        """
        return


class FakeIsaacClient:
    """Small Isaac client double used for routing tests."""

    def __init__(
        self,
        *,
        host: str,
        socket_port: int,
        bridge_port: int,
        timeout_seconds: float = 5.0,
    ) -> None:
        self._host = host
        self._port = socket_port
        self._bridge_host = host
        self._bridge_port = bridge_port
        self._bridge_configured = True
        self._prefer_bridge = True
        self._fallback_to_vscode = True
        self._timeout_seconds = timeout_seconds

    @property
    def address(self) -> str:
        return self.bridge_address

    @property
    def bridge_address(self) -> str:
        return f"{self._bridge_host}:{self._bridge_port}"

    @property
    def vscode_address(self) -> str:
        return f"{self._host}:{self._port}"

    @property
    def bridge_enabled(self) -> bool:
        return self._prefer_bridge

    @property
    def fallback_to_vscode(self) -> bool:
        return self._fallback_to_vscode

    @property
    def timeout_seconds(self) -> float:
        return self._timeout_seconds

    async def ping(self) -> bool:
        return True


def _make_server(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> server_module.SimulMCPServer:
    monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
    monkeypatch.setattr(server_module, "TaskConfig", None)
    monkeypatch.setattr(server_module, "is_headless_available", lambda: False)
    monkeypatch.setattr(server_module, "is_blender_available", lambda: False)
    monkeypatch.setattr(server_module, "UnrealRuntimeAdapter", None)
    return server_module.SimulMCPServer(settings=settings)


def _payload(result: Any) -> dict[str, Any]:
    """Read a tool result's JSON content block.

    Isaac tools return a content-only ToolResult so the payload is not also sent
    as structuredContent.
    """
    return json.loads(result.content[0].text)


def _get_tool(instance: server_module.SimulMCPServer, name: str) -> Callable[..., Any]:
    for tool in instance.mcp.tools:
        if tool.name == name:
            return tool.func
    raise AssertionError(f"Tool {name!r} not found")


@pytest.mark.asyncio
async def test_three_agents_route_to_three_distinct_isaac_apps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-session routing should isolate three agents across three Isaac apps."""
    settings = Settings(
        isaac_sim={
            "instances": [
                {"name": "local-b", "host": "127.0.0.1", "port": 8326},
                {"name": "container", "host": "127.0.0.1", "port": 9226},
            ]
        }
    )
    instance = _make_server(monkeypatch, settings)
    instance._isaac_clients = {
        "default": FakeIsaacClient(host="127.0.0.1", socket_port=8226, bridge_port=8229),
        "local-b": FakeIsaacClient(host="127.0.0.1", socket_port=8326, bridge_port=8329),
        "container": FakeIsaacClient(host="127.0.0.1", socket_port=9226, bridge_port=9229),
    }
    instance.client = instance._isaac_clients["default"]

    ctx_var: ContextVar[Any | None] = ContextVar("ctx", default=None)
    monkeypatch.setattr(instance, "_get_request_context", lambda: ctx_var.get())

    set_active = _get_tool(instance, "set_active_isaac_instance")
    ping = _get_tool(instance, "ping_isaac")

    async def worker(session_id: str, instance_name: str) -> dict[str, Any]:
        token = ctx_var.set(SimpleNamespace(session_id=session_id))
        try:
            await set_active(instance_name=instance_name, purpose=f"route-{session_id}")
            return await ping()
        finally:
            ctx_var.reset(token)

    results = await asyncio.gather(
        worker("agent-a", "default"),
        worker("agent-b", "local-b"),
        worker("agent-c", "container"),
    )

    assert [_payload(result)["address"] for result in results] == [
        "127.0.0.1:8229",
        "127.0.0.1:8329",
        "127.0.0.1:9229",
    ]
    assert [_payload(result)["vscode_address"] for result in results] == [
        "127.0.0.1:8226",
        "127.0.0.1:8326",
        "127.0.0.1:9226",
    ]
    assert instance._active_instance == "default"
    assert sorted(instance._session_routes) == ["agent-a", "agent-b", "agent-c"]


@pytest.mark.asyncio
async def test_same_instance_calls_are_serialized_per_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent sessions targeting the same Isaac app should not interleave."""
    instance = _make_server(monkeypatch, Settings())
    instance._isaac_clients = {
        "default": FakeIsaacClient(host="127.0.0.1", socket_port=8226, bridge_port=8229),
    }
    instance.client = instance._isaac_clients["default"]

    ctx_var: ContextVar[Any | None] = ContextVar("ctx", default=None)
    monkeypatch.setattr(instance, "_get_request_context", lambda: ctx_var.get())

    set_active = _get_tool(instance, "set_active_isaac_instance")
    get_stage_info = _get_tool(instance, "get_isaac_stage_info")

    running = 0
    max_running = 0

    async def fake_stage_info(include_prim_count: bool = False) -> dict[str, Any]:
        nonlocal running, max_running
        running += 1
        max_running = max(max_running, running)
        await asyncio.sleep(0.02)
        running -= 1
        return {"success": True, "address": instance._isaac_tools._client.address}

    monkeypatch.setattr(instance._isaac_tools, "get_isaac_stage_info", fake_stage_info)

    async def worker(session_id: str) -> dict[str, Any]:
        token = ctx_var.set(SimpleNamespace(session_id=session_id))
        try:
            await set_active(instance_name="default", purpose=f"same-{session_id}")
            return await get_stage_info()
        finally:
            ctx_var.reset(token)

    results = await asyncio.gather(worker("agent-a"), worker("agent-b"))

    assert max_running == 1
    assert all(_payload(result)["address"] == "127.0.0.1:8229" for result in results)


@pytest.mark.asyncio
async def test_discovery_file_uses_forwarded_vscode_port(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Discovery files should preserve the host-visible VS Code fallback port."""
    settings = Settings(
        isaac_sim={
            "discovery_dir": str(tmp_path),
            "scan_port_start": 8226,
            "scan_port_end": 8227,
        }
    )
    instance = _make_server(monkeypatch, settings)

    discovery_file = tmp_path / "simul-mcp-123.json"
    discovery_file.write_text(
        json.dumps(
            {
                "pid": 999999999,  # ignored via monkeypatched os.kill
                "host": "127.0.0.1",
                "port": 9229,
                "vscode_port": 9226,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(server_module.os, "kill", lambda pid, sig: None)

    class FakeSocketClient:
        def __init__(self, **kwargs: Any) -> None:
            self._host = kwargs["host"]
            self._port = kwargs["port"]
            self._bridge_host = kwargs["bridge_host"]
            self._bridge_port = kwargs["bridge_port"]
            self._bridge_configured = True
            self._prefer_bridge = kwargs["prefer_bridge"]
            self._fallback_to_vscode = kwargs["fallback_to_vscode"]
            self._timeout_seconds = kwargs["timeout_seconds"]

        async def ping(self) -> bool:
            return True

        @property
        def bridge_enabled(self) -> bool:
            return self._prefer_bridge

        @property
        def bridge_address(self) -> str:
            return f"{self._bridge_host}:{self._bridge_port}"

        @property
        def vscode_address(self) -> str:
            return f"{self._host}:{self._port}"

    monkeypatch.setattr(server_module, "IsaacSocketClient", FakeSocketClient)

    discovered = await instance._discover_from_files()

    client = discovered["isaac-9229"]
    assert client._port == 9226
    assert client._bridge_port == 9229
