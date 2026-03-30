"""Regression tests for bridge-aware Isaac server construction."""

from __future__ import annotations

import asyncio
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_path))

from simul_mcp.config import Settings  # noqa: E402
from simul_mcp.mcp import server as server_module  # noqa: E402

extension_root = Path(__file__).resolve().parents[2] / "exts" / "khemoo.simul.mcp"
sys.path.insert(0, str(extension_root))

from khemoo.simul.mcp.lifecycle import BridgeServerLifecycle  # noqa: E402
from khemoo.simul.mcp.protocol import BridgeResponse  # noqa: E402


class FakeFastMCP:
    """Minimal FastMCP test double for server construction."""

    def __init__(self, name: str, version: str, **kwargs: Any):
        self.name = name
        self.version = version
        self.description = kwargs.get("description")
        self.instructions = kwargs.get("instructions")
        self.tools: list[SimpleNamespace] = []

    def tool(
        self, name: str, **kwargs: Any
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Record tool metadata and return the original function."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.tools.append(SimpleNamespace(name=name, func=func, kwargs=kwargs))
            return func

        return decorator


def _make_server(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> server_module.SimulMCPServer:
    """Instantiate the server with runtime adapters stubbed out."""
    monkeypatch.setattr(server_module, "FASTMCP_AVAILABLE", True)
    monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
    monkeypatch.setattr(server_module, "TaskConfig", None)
    monkeypatch.setattr(server_module, "is_headless_available", lambda: False)
    monkeypatch.setattr(server_module, "is_blender_available", lambda: False)
    monkeypatch.setattr(server_module, "is_unreal_available", lambda: False)
    return server_module.SimulMCPServer(settings=settings)


def test_server_builds_named_instance_clients_with_bridge_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Named Isaac instances should inherit or override bridge settings explicitly."""
    settings = Settings(
        isaac_sim={
            "socket_host": "127.0.0.1",
            "socket_port": 8226,
            "socket_timeout": 30.0,
            "bridge_enabled": True,
            "bridge_host": "127.0.0.1",
            "bridge_port": 8229,
            "bridge_timeout": 11.0,
            "bridge_fallback_to_vscode": True,
            "instances": [
                {
                    "name": "secondary",
                    "host": "127.0.0.1",
                    "port": 8326,
                    "timeout": 7.5,
                },
                {
                    "name": "strict",
                    "host": "192.168.0.10",
                    "port": 8426,
                    "timeout": 9.0,
                    "bridge_enabled": True,
                    "bridge_host": "192.168.0.10",
                    "bridge_port": 9029,
                    "bridge_timeout": 4.5,
                    "bridge_fallback_to_vscode": False,
                },
            ],
        }
    )

    instance = _make_server(monkeypatch, settings)

    secondary = instance._isaac_clients["secondary"]
    strict = instance._isaac_clients["strict"]

    assert secondary._host == "127.0.0.1"
    assert secondary._port == 8326
    assert secondary._bridge_host == "127.0.0.1"
    assert secondary._bridge_port == 8229 + (8326 - 8226)
    assert secondary._bridge_timeout_seconds == 11.0
    assert secondary.bridge_enabled is True
    assert secondary.fallback_to_vscode is True

    assert strict._host == "192.168.0.10"
    assert strict._port == 8426
    assert strict._bridge_host == "192.168.0.10"
    assert strict._bridge_port == 9029
    assert strict._bridge_timeout_seconds == 4.5
    assert strict.bridge_enabled is True
    assert strict.fallback_to_vscode is False


def test_scan_isaac_instances_builds_bridge_aware_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto-discovered instances should probe the matching derived bridge port."""
    settings = Settings(
        isaac_sim={
            "socket_host": "127.0.0.1",
            "socket_port": 8226,
            "socket_timeout": 30.0,
            "bridge_enabled": True,
            "bridge_host": "127.0.0.1",
            "bridge_port": 8229,
            "bridge_timeout": 5.0,
            "bridge_fallback_to_vscode": True,
            "scan_port_start": 8226,
            "scan_port_end": 8229,
        }
    )
    instance = _make_server(monkeypatch, settings)

    constructed: list[SimpleNamespace] = []

    class FakeSocketClient:
        """Capture discovery client construction and emulate reachability."""

        def __init__(self, **kwargs: Any) -> None:
            constructed.append(SimpleNamespace(**kwargs))
            self._host = kwargs["host"]
            self._port = kwargs["port"]
            self._bridge_host = kwargs.get("bridge_host")
            self._bridge_port = kwargs.get("bridge_port")

        async def ping(self) -> bool:
            return self._port == 8227

    monkeypatch.setattr(server_module, "IsaacSocketClient", FakeSocketClient)

    discovered = asyncio.run(instance._scan_isaac_instances())

    assert list(discovered) == ["isaac-8227"]
    assert len(constructed) == 2
    assert [item.port for item in constructed] == [8227, 8228]
    assert [item.bridge_port for item in constructed] == [8230, 8231]
    assert all(item.prefer_bridge is True for item in constructed)
    assert all(item.bridge_host == "127.0.0.1" for item in constructed)


def test_bridge_lifecycle_returns_structured_error_for_oversize_response() -> None:
    """Oversize bridge payloads should downgrade to a deterministic error envelope."""

    async def _exercise() -> dict[str, Any]:
        lifecycle = BridgeServerLifecycle(
            host="127.0.0.1",
            port=0,
            request_handler=lambda request: asyncio.sleep(
                0,
                result=BridgeResponse.success(
                    request.request_id,
                    {"blob": "x" * 1024},
                ),
            ),
            max_response_bytes=256,
        )
        await lifecycle.start()
        assert lifecycle._server is not None
        sock = next(iter(lifecycle._server.sockets))
        port = sock.getsockname()[1]

        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            request = json.dumps(
                {
                    "request_id": "req-1",
                    "action": "capabilities",
                    "payload": {},
                }
            ).encode("utf-8")
            writer.write(struct.pack(">I", len(request)) + request)
            await writer.drain()

            header = await reader.readexactly(4)
            size = struct.unpack(">I", header)[0]
            payload = await reader.readexactly(size)
            writer.close()
            await writer.wait_closed()
        finally:
            await lifecycle.stop()

        return json.loads(payload.decode("utf-8"))

    response = asyncio.run(_exercise())

    assert response["status"] == "error"
    assert response["request_id"] == "req-1"
    assert response["error"]["name"] == "ResponseTooLarge"
    assert "exceeded 256 bytes" in response["error"]["message"]
