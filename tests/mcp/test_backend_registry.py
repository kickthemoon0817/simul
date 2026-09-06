"""The backend registry is the one place the server learns about backends.

``BACKENDS`` drives adapter construction, tool registration, the capability
report, the CLI's tool grouping and the ROUTING instructions. These tests pin
that every one of those reads the registry, that each shipped adapter satisfies
``BackendAdapter``, and that the server and CLI accept the streamable HTTP
transport.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

import pytest
from typer.testing import CliRunner

from simul_mcp.adapters import HeadlessUSDAdapter, IsaacRuntimeAdapter, UnrealRuntimeAdapter
from simul_mcp.adapters.base import BackendAdapter
from simul_mcp.cli import main as cli_main
from simul_mcp.cli.main import app
from simul_mcp.config import Settings
from simul_mcp.mcp import backends as backends_module
from simul_mcp.mcp import server as server_module
from simul_mcp.mcp.backends import ALL_BACKEND_NAMES, BACKENDS, BackendSpec, backend_spec
from tests.fakes import AvailableAdapter, FakeFastMCP

runner = CliRunner()


def _full_settings() -> Settings:
    base = Settings()
    return base.model_copy(update={"unreal": base.unreal.model_copy(update={"tool_surface": "full"})})


@pytest.fixture
def full_server(fake_fastmcp: Any, monkeypatch: pytest.MonkeyPatch) -> server_module.SimulMCPServer:
    """Every backend available and registered, on the recording double."""
    monkeypatch.setattr(backends_module, "is_headless_available", lambda: True)
    monkeypatch.setattr(backends_module, "is_blender_available", lambda: True)
    monkeypatch.setattr(backends_module, "is_unreal_available", lambda: True)
    monkeypatch.setattr(backends_module, "BlenderRuntimeAdapter", AvailableAdapter)
    monkeypatch.setattr(backends_module, "UnrealRuntimeAdapter", AvailableAdapter)
    return server_module.SimulMCPServer(settings=_full_settings())


class TestRegistryShape:
    def test_names_are_unique_and_match_the_server_whitelist(self) -> None:
        names = [spec.name for spec in BACKENDS]
        assert len(names) == len(set(names))
        assert set(names) == ALL_BACKEND_NAMES == server_module.SimulMCPServer.ALL_BACKENDS
        assert {"isaac", "usd", "blender", "unreal"} <= set(names)

    def test_every_spec_names_a_settings_section(self) -> None:
        settings = Settings()
        for spec in BACKENDS:
            assert hasattr(settings, spec.settings_attribute), spec.name

    def test_backend_spec_lookup(self) -> None:
        assert backend_spec("isaac").label == "Isaac Sim"
        with pytest.raises(KeyError):
            backend_spec("houdini")

    def test_specs_are_immutable(self) -> None:
        with pytest.raises(AttributeError):
            BACKENDS[0].name = "other"  # type: ignore[misc]


class TestAdapterProtocol:
    @pytest.mark.parametrize("adapter_class", [IsaacRuntimeAdapter, HeadlessUSDAdapter, UnrealRuntimeAdapter])
    def test_shipped_adapters_satisfy_the_protocol(self, adapter_class: Any) -> None:
        if adapter_class is None:
            pytest.skip("adapter module did not import in this environment")
        adapter = adapter_class(Settings())
        assert isinstance(adapter, BackendAdapter)
        assert adapter.name in ALL_BACKEND_NAMES
        assert isinstance(adapter.is_available(), bool)
        assert isinstance(adapter.get_capabilities(), list)
        adapter.close()

    def test_blender_adapter_satisfies_the_protocol_when_importable(self) -> None:
        if backends_module.BlenderRuntimeAdapter is None:
            pytest.skip("bpy is not importable")
        adapter = backends_module.BlenderRuntimeAdapter(Settings())
        assert isinstance(adapter, BackendAdapter)
        assert adapter.name == "blender"

    def test_registry_adapter_names_match_their_spec(self, full_server: server_module.SimulMCPServer) -> None:
        for spec in BACKENDS:
            adapter = full_server._adapters[spec.name]
            assert adapter is not None, spec.name
            if not isinstance(adapter, AvailableAdapter):
                assert adapter.name == spec.name

    def test_isaac_adapter_owns_the_default_client(self, full_server: server_module.SimulMCPServer) -> None:
        assert full_server.client is full_server.isaac_adapter.client
        assert full_server.isaac_adapter.get_capabilities() == ["socket", "bridge"]
        with full_server.isaac_adapter.create_session() as session:
            assert session is full_server.client


class TestServerIteratesTheRegistry:
    def test_capability_report_covers_every_spec(self, full_server: server_module.SimulMCPServer) -> None:
        report = full_server.get_capabilities()
        assert list(report) == [spec.name for spec in BACKENDS]
        for entry in report.values():
            assert entry["enabled"] is True and entry["available"] is True

    def test_tools_are_grouped_by_the_backend_that_registered_them(
        self, full_server: server_module.SimulMCPServer
    ) -> None:
        groups = full_server.tools_by_backend()
        labels = [spec.label for spec in BACKENDS]
        assert [label for label in groups if label != "Server"] == labels
        registered = {tool.name for tool in full_server.mcp.tools}
        grouped = {name for names in groups.values() for name in names}
        assert grouped == registered
        assert "claim_isaac_instance" in groups["Isaac Sim"]
        assert "load_usd_file" in groups["USD / Headless"]
        assert "spawn_unreal_actor" in groups["Unreal"]
        assert groups["Server"] == ["get_tool_usage_stats"]

    def test_backend_filter_registers_only_the_selected_backends(
        self, fake_fastmcp: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(backends_module, "is_headless_available", lambda: True)
        instance = server_module.SimulMCPServer(settings=Settings(), backends={"usd"})
        groups = instance.tools_by_backend()
        assert set(groups) == {"USD / Headless", "Server"}
        assert instance.get_capabilities()["isaac"]["enabled"] is False

    def test_unavailable_adapter_registers_nothing(self, fake_fastmcp: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(backends_module, "is_headless_available", lambda: False)
        monkeypatch.setattr(backends_module, "is_blender_available", lambda: False)
        monkeypatch.setattr(backends_module, "UnrealRuntimeAdapter", None)
        instance = server_module.SimulMCPServer(settings=Settings())
        assert instance.headless_adapter is None
        assert instance.blender_adapter is None
        assert instance.unreal_adapter is None
        assert set(instance.tools_by_backend()) == {"Isaac Sim", "Server"}

    def test_instructions_state_every_routing_rule(self, full_server: server_module.SimulMCPServer) -> None:
        text = server_module._MCP_INSTRUCTIONS
        for spec in BACKENDS:
            assert spec.routing_rule in text, spec.name

    def test_shutdown_closes_every_adapter(self, full_server: server_module.SimulMCPServer) -> None:
        closed: List[str] = []

        class _Closing(AvailableAdapter):
            def __init__(self, label: str) -> None:
                super().__init__(Settings())
                self.name = label

            def close(self) -> None:
                closed.append(self.name)

        full_server.headless_adapter = _Closing("usd")
        full_server.blender_adapter = _Closing("blender")
        full_server.unreal_adapter = _Closing("unreal")

        asyncio.run(full_server.shutdown())

        assert sorted(closed) == ["blender", "unreal", "usd"]
        assert full_server._isaac_clients == {}


class TestTransports:
    def test_http_transport_listens_on_the_configured_address(self, fake_fastmcp: Any) -> None:
        instance = server_module.SimulMCPServer(settings=Settings(), backends={"isaac"})

        asyncio.run(instance.run("http"))

        assert instance.mcp.run_calls == [
            {"transport": "http", "host": instance.settings.server.host, "port": instance.settings.server.port}
        ]

    def test_stdio_transport_takes_no_address(self, fake_fastmcp: Any) -> None:
        instance = server_module.SimulMCPServer(settings=Settings(), backends={"isaac"})

        asyncio.run(instance.run("stdio"))

        assert instance.mcp.run_calls == [{"transport": "stdio"}]

    def test_unknown_transport_is_refused_before_serving(self, fake_fastmcp: Any) -> None:
        instance = server_module.SimulMCPServer(settings=Settings(), backends={"isaac"})

        with pytest.raises(ValueError, match="grpc"):
            asyncio.run(instance.run("grpc"))
        assert instance.mcp.run_calls == []

    def test_transport_whitelist(self) -> None:
        assert server_module.TRANSPORTS == ("stdio", "http", "sse")


class TestCli:
    @pytest.fixture
    def captured(self, monkeypatch: pytest.MonkeyPatch) -> List[Dict[str, Any]]:
        """Stop before the server starts, keeping what the CLI would have started it with."""
        seen: List[Dict[str, Any]] = []

        async def _capture(settings: Settings, transport: str, **kwargs: Any) -> None:
            seen.append({"transport": transport, **kwargs})

        monkeypatch.setattr(cli_main, "start_mcp_server", _capture)
        monkeypatch.setattr(cli_main, "_is_isaac_reachable", lambda *a, **k: False)
        return seen

    def test_server_accepts_http_transport(self, captured: List[Dict[str, Any]]) -> None:
        result = runner.invoke(app, ["server", "--transport", "HTTP"])
        assert result.exit_code == 0, result.output
        assert captured[-1]["transport"] == "http"

    def test_server_refuses_unknown_transport(self, captured: List[Dict[str, Any]]) -> None:
        result = runner.invoke(app, ["server", "--transport", "grpc"])
        assert result.exit_code == 1
        assert "stdio, http, sse" in result.output
        assert captured == []

    def test_backends_flag_filters_and_validates(self, captured: List[Dict[str, Any]]) -> None:
        result = runner.invoke(app, ["server", "--backends", "usd, Isaac"])
        assert result.exit_code == 0, result.output
        assert captured[-1]["backends"] == {"usd", "isaac"}

        result = runner.invoke(app, ["server", "--backends", "usd,houdini"])
        assert result.exit_code == 1
        assert "houdini" in result.output
        assert len(captured) == 1

    def test_info_groups_tools_by_registry_label(self) -> None:
        result = runner.invoke(app, ["--json", "info"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        labels = {spec.label for spec in BACKENDS} | {"Server"}
        assert set(payload["categories"]) <= labels
        assert "Isaac Sim" in payload["categories"]
        assert "ping_isaac" in payload["categories"]["Isaac Sim"]
        assert payload["categories"]["Server"] == ["get_tool_usage_stats"]
        assert payload["tool_count"] == sum(len(names) for names in payload["categories"].values())
        assert set(payload["capabilities"]) == ALL_BACKEND_NAMES


def test_spec_fields_are_complete() -> None:
    for spec in BACKENDS:
        assert isinstance(spec, BackendSpec)
        assert spec.label and spec.routing_rule.startswith("Tools containing")
        assert callable(spec.adapter_factory) and callable(spec.register_tools)
