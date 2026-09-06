"""The instance port scan must never probe a bridge port with the stock protocol.

The bridge reads a length prefix and waits for the body; a raw-source probe
never sends one, so the probe only returns when its read deadline expires.
With the default range 8226-8235 and the bridge on 8229 that added the
full cap to every ``list_isaac_instances`` call for nothing.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List

import pytest

src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_path))

from simul_mcp.config import IsaacInstanceConfig, IsaacSimConfig, Settings  # noqa: E402
from simul_mcp.mcp import server as server_module  # noqa: E402


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


def _make_server(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> server_module.SimulMCPServer:
    monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
    monkeypatch.setattr(server_module, "TaskConfig", None)
    monkeypatch.setattr(server_module, "is_headless_available", lambda: False)
    monkeypatch.setattr(server_module, "is_blender_available", lambda: False)
    monkeypatch.setattr(server_module, "UnrealRuntimeAdapter", None)
    return server_module.SimulMCPServer(settings=settings)


def _scan_with_recorded_clients(
    monkeypatch: pytest.MonkeyPatch, instance: server_module.SimulMCPServer
) -> List[int]:
    """Run the scan with every constructed client recorded and pinging False."""
    built_ports: List[int] = []
    original_build = instance._build_isaac_client

    def _recording_build(**kwargs: Any) -> Any:
        client = original_build(**kwargs)
        built_ports.append(kwargs["socket_port"])

        async def _never_reachable() -> bool:
            return False

        client.ping = _never_reachable  # type: ignore[method-assign]
        return client

    monkeypatch.setattr(instance, "_build_isaac_client", _recording_build)

    async def _no_files() -> Dict[str, Any]:
        return {}

    monkeypatch.setattr(instance, "_discover_from_files", _no_files)
    asyncio.run(instance._scan_isaac_instances())
    return built_ports


def test_scan_never_builds_a_client_for_the_configured_bridge_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        isaac_sim=IsaacSimConfig(
            socket_port=8226, bridge_port=8229, scan_port_start=8226, scan_port_end=8236
        )
    )
    instance = _make_server(monkeypatch, settings)

    scanned = _scan_with_recorded_clients(monkeypatch, instance)

    assert 8229 not in scanned
    # The default instance owns 8226; everything else in range is still probed.
    assert scanned == [8227, 8228, 8230, 8231, 8232, 8233, 8234, 8235]


def test_scan_skips_per_instance_bridge_ports_too(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(
        isaac_sim=IsaacSimConfig(
            socket_port=8226,
            bridge_port=8229,
            scan_port_start=8226,
            scan_port_end=8236,
            instances=[
                IsaacInstanceConfig(name="second", port=8227, bridge_port=8233),
            ],
        )
    )
    instance = _make_server(monkeypatch, settings)

    scanned = _scan_with_recorded_clients(monkeypatch, instance)

    assert 8229 not in scanned
    assert 8233 not in scanned
    assert 8227 not in scanned  # the named instance's own socket port
    assert scanned == [8228, 8230, 8231, 8232, 8234, 8235]


def test_scan_skips_bridge_ports_advertised_by_discovery_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        isaac_sim=IsaacSimConfig(
            socket_port=8226, bridge_port=8229, scan_port_start=8226, scan_port_end=8236
        )
    )
    instance = _make_server(monkeypatch, settings)
    # A bridge that bound a fallback port after 8229 was taken, discovered by file.
    discovered_client = instance._build_isaac_client(
        socket_host="127.0.0.1",
        socket_port=8231,
        socket_timeout=3.0,
        bridge_enabled=True,
        bridge_host="127.0.0.1",
        bridge_port=8234,
    )

    built_ports: List[int] = []
    original_build = instance._build_isaac_client

    def _recording_build(**kwargs: Any) -> Any:
        client = original_build(**kwargs)
        built_ports.append(kwargs["socket_port"])

        async def _never_reachable() -> bool:
            return False

        client.ping = _never_reachable  # type: ignore[method-assign]
        return client

    monkeypatch.setattr(instance, "_build_isaac_client", _recording_build)

    async def _from_files() -> Dict[str, Any]:
        return {"isaac-8234": discovered_client}

    monkeypatch.setattr(instance, "_discover_from_files", _from_files)
    result = asyncio.run(instance._scan_isaac_instances())

    assert "isaac-8234" in result
    assert 8234 not in built_ports
    assert 8231 not in built_ports
    assert 8229 not in built_ports
