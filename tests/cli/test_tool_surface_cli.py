"""Tool surfaces, backend-scoped startup probes and ``simul-mcp tools``.

Covers issues #218 (thin Blender surface, one definition of each thin list),
#219 (the server probes only the backends it was asked for) and #220 (the
``tools`` command lists the MCP tool surface without a live session).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest
from typer.testing import CliRunner

from simul_mcp import tool_surfaces
from simul_mcp.cli import main as cli_main
from simul_mcp.cli.main import app
from simul_mcp.config import BlenderConfig, Settings, UnrealConfig
from simul_mcp.mcp import backends as backends_module
from simul_mcp.mcp import server as server_module
from tests.fakes import AvailableAdapter

runner = CliRunner()


def _surface(section: str, surface: str) -> Settings:
    base = Settings()
    return base.model_copy(
        update={section: getattr(base, section).model_copy(update={"tool_surface": surface})}
    )


@pytest.fixture
def runtimes_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the Blender and Unreal adapters look importable and available."""
    monkeypatch.setattr(backends_module, "is_blender_available", lambda: True)
    monkeypatch.setattr(backends_module, "is_unreal_available", lambda: True)
    monkeypatch.setattr(backends_module, "BlenderRuntimeAdapter", AvailableAdapter)
    monkeypatch.setattr(backends_module, "UnrealRuntimeAdapter", AvailableAdapter)


def _registered(settings: Settings, backend: str) -> set:
    instance = server_module.SimulMCPServer(settings=settings, backends={backend})
    return set(instance.tools_by_backend().get(backends_module.backend_spec(backend).label, []))


# ---------------------------------------------------------------------------
# #218 — thin surfaces, defined once
# ---------------------------------------------------------------------------
class TestThinSurfaces:
    @pytest.mark.parametrize(
        "backend,thin",
        [
            ("unreal", tool_surfaces.THIN_UNREAL_TOOLS),
            ("blender", tool_surfaces.THIN_BLENDER_TOOLS),
        ],
    )
    def test_thin_registers_exactly_the_shared_tuple(
        self, fake_fastmcp: Any, runtimes_present: None, backend: str, thin: tuple
    ) -> None:
        assert len(set(thin)) == len(thin)
        assert _registered(_surface(backend, "thin"), backend) == set(thin)

    @pytest.mark.parametrize(
        "backend,thin",
        [
            ("unreal", tool_surfaces.THIN_UNREAL_TOOLS),
            ("blender", tool_surfaces.THIN_BLENDER_TOOLS),
        ],
    )
    def test_full_is_a_strict_superset(
        self, fake_fastmcp: Any, runtimes_present: None, backend: str, thin: tuple
    ) -> None:
        full = _registered(_surface(backend, "full"), backend)
        assert set(thin) < full

    def test_blender_defaults_to_full_and_unreal_to_thin(self) -> None:
        settings = Settings()
        assert settings.blender.tool_surface == "full"
        assert settings.unreal.tool_surface == "thin"

    def test_blender_surface_reads_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BLENDER__TOOL_SURFACE", "thin")
        assert Settings().blender.tool_surface == "thin"

    def test_config_descriptions_name_every_thin_tool(self) -> None:
        unreal = UnrealConfig.model_fields["tool_surface"].description or ""
        blender = BlenderConfig.model_fields["tool_surface"].description or ""
        assert all(name in unreal for name in tool_surfaces.THIN_UNREAL_TOOLS)
        assert all(name in blender for name in tool_surfaces.THIN_BLENDER_TOOLS)

    def test_cli_help_names_every_thin_tool(self) -> None:
        unreal = cli_main._unreal_tools_option().help
        blender = cli_main._blender_tools_option().help
        assert all(name in unreal for name in tool_surfaces.THIN_UNREAL_TOOLS)
        assert all(name in blender for name in tool_surfaces.THIN_BLENDER_TOOLS)

    def test_describe_tools(self) -> None:
        assert tool_surfaces.describe_tools(("a",)) == "a"
        assert tool_surfaces.describe_tools(("a", "b", "c")) == "a, b and c"


# ---------------------------------------------------------------------------
# Shared option parsing
# ---------------------------------------------------------------------------
class TestSharedOptions:
    @pytest.fixture
    def captured(self, monkeypatch: pytest.MonkeyPatch) -> List[Dict[str, Any]]:
        seen: List[Dict[str, Any]] = []

        async def _capture(settings: Settings, transport: str, **kwargs: Any) -> None:
            seen.append({"settings": settings, **kwargs})

        monkeypatch.setattr(cli_main, "start_mcp_server", _capture)
        monkeypatch.setattr(cli_main, "_is_isaac_reachable", lambda *a, **k: False)
        return seen

    def test_server_blender_tools_flag(self, captured: List[Dict[str, Any]]) -> None:
        result = runner.invoke(app, ["server", "--backends", "usd", "--blender-tools", "Thin"])
        assert result.exit_code == 0, result.output
        assert captured[-1]["settings"].blender.tool_surface == "thin"

    @pytest.mark.parametrize("command", ["server", "info", "tools"])
    def test_every_command_rejects_an_unknown_surface(
        self, captured: List[Dict[str, Any]], command: str
    ) -> None:
        result = runner.invoke(app, [command, "--blender-tools", "medium"])
        assert result.exit_code == 1
        assert "Unknown --blender-tools value" in result.output
        assert not captured

    @pytest.mark.parametrize("command", ["server", "info", "tools"])
    def test_every_command_rejects_an_unknown_backend(
        self, captured: List[Dict[str, Any]], command: str
    ) -> None:
        result = runner.invoke(app, [command, "--backends", "usd,houdini"])
        assert result.exit_code == 1
        assert "houdini" in result.output

    def test_apply_backend_options_parses_every_flag(self) -> None:
        settings, backend_set = cli_main._apply_backend_options(
            Settings(), " USD ,blender", "FULL", "thin", "attached", "attached"
        )
        assert backend_set == {"usd", "blender"}
        assert settings.unreal.tool_surface == "full"
        assert settings.blender.tool_surface == "thin"
        assert settings.unreal.mode == "attached"
        assert settings.blender.mode == "attached"
        assert cli_main._apply_backend_options(Settings())[1] is None


# ---------------------------------------------------------------------------
# #219 — only the selected backends are probed
# ---------------------------------------------------------------------------
class TestScopedStartup:
    def test_usd_only_server_probes_nothing_else(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import simul_mcp.adapters as adapters

        def _forbidden(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("probed a backend that was not selected")

        seen: List[Any] = []

        async def _capture(settings: Settings, transport: str, **kwargs: Any) -> None:
            seen.append(kwargs["backends"])

        monkeypatch.setattr(cli_main, "start_mcp_server", _capture)
        monkeypatch.setattr(cli_main, "_is_isaac_reachable", _forbidden)
        monkeypatch.setattr(adapters, "is_blender_available", _forbidden)
        monkeypatch.setattr(adapters, "is_headless_available", lambda: True)

        result = runner.invoke(app, ["server", "--backends", "usd"])

        assert result.exit_code == 0, result.output
        assert seen == [{"usd"}]
        assert "USD Headless: available" in result.output
        for absent in ("Isaac Sim", "Unreal tools", "Blender"):
            assert absent not in result.output

    def test_all_backends_banner_still_reports_each(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _noop(*args: Any, **kwargs: Any) -> None:
            return None

        monkeypatch.setattr(cli_main, "start_mcp_server", _noop)
        monkeypatch.setattr(cli_main, "_is_isaac_reachable", lambda *a, **k: False)
        result = runner.invoke(app, ["server"])
        assert result.exit_code == 0, result.output
        for present in ("Isaac Sim", "Unreal tools: thin", "Blender (", "USD Headless"):
            assert present in result.output

    def test_server_builds_adapters_only_for_selected_backends(
        self, fake_fastmcp: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _forbidden(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("built the adapter of a backend that was not selected")

        monkeypatch.setattr(backends_module, "is_headless_available", lambda: True)
        monkeypatch.setattr(backends_module, "is_blender_available", _forbidden)
        monkeypatch.setattr(backends_module, "is_unreal_available", _forbidden)

        instance = server_module.SimulMCPServer(settings=Settings(), backends={"usd"})

        assert instance.blender_adapter is None
        assert instance.unreal_adapter is None
        assert instance.headless_adapter is not None
        # Isaac's adapter is always built; it owns the default client.
        assert instance.client is instance.isaac_adapter.client
        report = instance.get_capabilities()
        assert report["blender"] == {"enabled": False, "available": False, "capabilities": []}

    def test_info_skips_probes_for_unselected_backends(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _forbidden(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("probed Isaac although only usd was selected")

        monkeypatch.setattr(cli_main, "_is_isaac_reachable", _forbidden)
        result = runner.invoke(app, ["--json", "info", "--backends", "usd"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert payload["backends"]["isaac_sim"]["reachable"] is None
        assert payload["backends"]["blender"]["available"] is None
        assert set(payload["categories"]) <= {"USD / Headless", "Server"}


# ---------------------------------------------------------------------------
# #220 — simul-mcp tools
# ---------------------------------------------------------------------------
class TestToolsCommand:
    def test_json_lists_usd_tools_with_schemas(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cli_main, "_is_isaac_reachable", lambda *a, **k: pytest.fail("probed"))
        result = runner.invoke(app, ["--json", "tools", "--backends", "usd"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert set(payload["backends"]) == {"usd", "server"}
        usd = payload["backends"]["usd"]
        names = {tool["name"] for tool in usd["tools"]}
        assert "load_usd_file" in names
        assert not any("isaac" in name or "blender" in name for name in names)
        load = next(tool for tool in usd["tools"] if tool["name"] == "load_usd_file")
        assert load["description"]
        assert load["input_schema"]["type"] == "object"
        assert "file_path" in load["input_schema"]["properties"]
        assert usd["tool_count"] == len(usd["tools"])
        assert payload["tool_count"] == sum(g["tool_count"] for g in payload["backends"].values())
        assert payload["bytes"] == sum(g["bytes"] for g in payload["backends"].values())
        assert [t["name"] for t in payload["backends"]["server"]["tools"]] == ["get_tool_usage_stats"]

    def test_thin_blender_lists_the_shared_tuple_without_a_runtime(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(backends_module, "is_blender_available", lambda: False)
        result = runner.invoke(
            app, ["--json", "tools", "--backends", "blender", "--blender-tools", "thin"]
        )
        assert result.exit_code == 0, result.output
        blender = json.loads(result.stdout)["backends"]["blender"]
        assert blender["tool_surface"] == "thin"
        assert blender["available"] is False
        assert {t["name"] for t in blender["tools"]} == set(tool_surfaces.THIN_BLENDER_TOOLS)

    def test_thin_is_smaller_than_full(self) -> None:
        sizes = {}
        for surface in ("thin", "full"):
            result = runner.invoke(
                app, ["--json", "tools", "--backends", "blender", "--blender-tools", surface]
            )
            assert result.exit_code == 0, result.output
            sizes[surface] = json.loads(result.stdout)["backends"]["blender"]["bytes"]
        assert sizes["thin"] * 3 < sizes["full"]

    def test_human_mode_groups_names_by_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cli_main, "is_json_mode", lambda: False)
        result = runner.invoke(app, ["tools", "--backends", "unreal"])
        assert result.exit_code == 0, result.output
        assert "Unreal (thin)" in result.output
        for name in tool_surfaces.THIN_UNREAL_TOOLS:
            assert name in result.output

    def test_commands_points_at_tools(self) -> None:
        result = runner.invoke(app, ["commands"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert "simul-mcp tools" in payload["mcp_tools"]
        assert "tools" in {entry["command"] for entry in payload["commands"]}


@pytest.mark.parametrize("value", [",", " , ", ""])
def test_empty_backends_selection_is_rejected(value: str) -> None:
    """A --backends value that names nothing must not enable every backend."""
    result = runner.invoke(app, ["--json", "tools", "--backends", value])
    assert result.exit_code != 0
    assert "names no backend" in result.stdout
