"""The Isaac Sim tools register only where Isaac Sim can run, unless the user says otherwise.

NVIDIA ships Isaac Sim for Linux and Windows. Elsewhere (macOS) the Isaac
tools cannot reach a local Isaac Sim, yet cost every session tens of
kilobytes of tool schemas, so ``isaac_sim.enabled`` (``ISAAC_SIM__ENABLED``)
is tri-state: unset registers them by platform, true on any OS, false never.
Naming ``isaac`` in ``--backends`` also registers them on any OS. The
suite-wide fixture in ``tests/conftest.py`` reports a supported platform;
these tests replace it with the real probe pinned to the platform they need.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List, Optional, Set

import pytest
from typer.testing import CliRunner

from simul import config as config_module
from simul.cli import isaac as isaac_cli
from simul.cli import main as cli_main
from simul.cli.main import app
from simul.config import IsaacSimConfig, Settings
from simul.mcp import backends as backends_module
from simul.mcp import server as server_module
from simul.mcp.backends import (
    ALL_BACKEND_NAMES,
    BACKENDS,
    ISAAC_DISABLED_SKIP,
    ISAAC_PLATFORM_SKIP,
    NOT_SELECTED,
    backend_skip_reason,
    is_isaac_platform_supported,
)

runner = CliRunner()

ISAAC_RESOURCE = "simul://isaac-sim/skills"


def _on(monkeypatch: pytest.MonkeyPatch, platform: str) -> None:
    """Run the real platform probe as if this interpreter were on ``platform``."""
    monkeypatch.setattr(
        backends_module,
        "is_isaac_platform_supported",
        lambda platform_name=None: is_isaac_platform_supported(platform),
    )


def _with_enabled(value: Optional[bool]) -> Settings:
    base = Settings()
    return base.model_copy(
        update={"isaac_sim": base.isaac_sim.model_copy(update={"enabled": value})}
    )


def _isaac_tools(instance: server_module.SimulMCPServer) -> List[str]:
    return [tool.name for tool in instance.mcp.tools if "isaac" in tool.name]


def _flat(output: str) -> str:
    """Join Rich's wrapped table and panel lines back into one line of words."""
    for border in "│┃":
        output = output.replace(border, " ")
    return " ".join(output.split())


@pytest.fixture
def make_server(
    fake_fastmcp: Any, monkeypatch: pytest.MonkeyPatch
) -> Callable[..., server_module.SimulMCPServer]:
    """Build servers on the recording double with headless USD present and instructions on."""
    monkeypatch.setattr(backends_module, "is_headless_available", lambda: True)
    monkeypatch.setattr(server_module, "_FASTMCP_SUPPORTS_INSTRUCTIONS", True)

    def _make(
        settings: Optional[Settings] = None, backends: Optional[Set[str]] = None
    ) -> server_module.SimulMCPServer:
        return server_module.SimulMCPServer(
            settings=settings or Settings(), backends=backends
        )

    return _make


@pytest.fixture
def fresh_settings() -> Iterator[None]:
    """Make the CLI re-read the environment: ``get_settings`` caches per config file."""
    config_module._get_cached_settings.cache_clear()
    yield
    config_module._get_cached_settings.cache_clear()


# ---------------------------------------------------------------------------
# The probe and the setting
# ---------------------------------------------------------------------------
class TestPlatformProbe:
    @pytest.mark.parametrize("platform", ["linux", "linux2", "win32", "cygwin"])
    def test_platforms_isaac_sim_runs_on(self, platform: str) -> None:
        assert is_isaac_platform_supported(platform) is True

    @pytest.mark.parametrize(
        "platform", ["darwin", "freebsd14", "openbsd7", "emscripten", "wasi", "aix"]
    )
    def test_other_platforms(self, platform: str) -> None:
        assert is_isaac_platform_supported(platform) is False

    def test_defaults_to_this_interpreter(self) -> None:
        assert is_isaac_platform_supported() is is_isaac_platform_supported(
            sys.platform
        )


class TestEnabledSetting:
    def test_defaults_to_auto(self) -> None:
        assert IsaacSimConfig().enabled is None
        assert Settings().isaac_sim.enabled is None

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("true", True),
            ("1", True),
            ("false", False),
            ("0", False),
            ("auto", None),
            ("AUTO", None),
            ("", None),
        ],
    )
    def test_environment(
        self, monkeypatch: pytest.MonkeyPatch, raw: str, expected: Optional[bool]
    ) -> None:
        monkeypatch.setenv("ISAAC_SIM__ENABLED", raw)
        assert Settings().isaac_sim.enabled is expected

    @pytest.mark.parametrize(
        "raw,expected",
        [("false", False), ("true", True), ("auto", None), ("null", None)],
    )
    def test_yaml(self, tmp_path: Path, raw: str, expected: Optional[bool]) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(f"isaac_sim:\n  enabled: {raw}\n", encoding="utf-8")
        assert Settings.from_yaml(config_file).isaac_sim.enabled is expected

    def test_environment_beats_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("isaac_sim:\n  enabled: false\n", encoding="utf-8")
        monkeypatch.setenv("ISAAC_SIM__ENABLED", "true")
        assert Settings.from_yaml(config_file).isaac_sim.enabled is True

    def test_rejects_other_values(self) -> None:
        with pytest.raises(ValueError):
            IsaacSimConfig(enabled="sometimes")  # type: ignore[arg-type]

    def test_packaged_yaml_leaves_it_on_auto(self) -> None:
        from simul.config import get_settings

        assert get_settings().isaac_sim.enabled is None


# ---------------------------------------------------------------------------
# The skip decision, shared by the server and the CLI banner
# ---------------------------------------------------------------------------
class TestSkipReason:
    @pytest.mark.parametrize("platform", ["linux", "win32"])
    def test_auto_registers_where_isaac_runs(
        self, monkeypatch: pytest.MonkeyPatch, platform: str
    ) -> None:
        _on(monkeypatch, platform)
        assert backend_skip_reason("isaac", Settings()) is None

    def test_auto_skips_elsewhere_and_says_how_to_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _on(monkeypatch, "darwin")
        reason = backend_skip_reason("isaac", Settings())
        assert reason == ISAAC_PLATFORM_SKIP
        assert "Linux or Windows" in reason
        assert "ISAAC_SIM__ENABLED=true" in reason
        assert "--backends isaac" in reason
        assert "unsupported" in reason

    def test_enabled_true_registers_on_any_os(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _on(monkeypatch, "darwin")
        assert backend_skip_reason("isaac", _with_enabled(True)) is None

    @pytest.mark.parametrize("platform", ["linux", "darwin"])
    def test_enabled_false_never_registers(
        self, monkeypatch: pytest.MonkeyPatch, platform: str
    ) -> None:
        _on(monkeypatch, platform)
        assert backend_skip_reason("isaac", _with_enabled(False)) == ISAAC_DISABLED_SKIP
        assert (
            backend_skip_reason("isaac", _with_enabled(False), {"isaac"})
            == ISAAC_DISABLED_SKIP
        )

    def test_naming_isaac_overrides_the_platform(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _on(monkeypatch, "darwin")
        assert backend_skip_reason("isaac", Settings(), {"isaac"}) is None
        assert backend_skip_reason("isaac", Settings(), {"isaac", "usd"}) is None

    def test_selection_without_isaac_leaves_it_out(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _on(monkeypatch, "linux")
        assert backend_skip_reason("isaac", Settings(), {"usd"}) == NOT_SELECTED

    def test_other_backends_have_no_platform_gate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _on(monkeypatch, "darwin")
        for name in ALL_BACKEND_NAMES - {"isaac"}:
            assert backend_skip_reason(name, Settings()) is None, name
            assert (
                backend_skip_reason(name, Settings(), {"isaac"}) == NOT_SELECTED
            ), name

    def test_unknown_backend(self) -> None:
        with pytest.raises(KeyError):
            backend_skip_reason("houdini", Settings())


# ---------------------------------------------------------------------------
# The server: tools, resources, instructions, capabilities
# ---------------------------------------------------------------------------
class TestServerRegistration:
    @pytest.mark.parametrize("platform", ["linux", "win32"])
    def test_default_on_where_isaac_runs(
        self,
        make_server: Callable[..., server_module.SimulMCPServer],
        monkeypatch: pytest.MonkeyPatch,
        platform: str,
    ) -> None:
        _on(monkeypatch, platform)
        instance = make_server()
        assert "ping_isaac" in _isaac_tools(instance)
        assert "list_isaac_instances" in _isaac_tools(instance)
        assert ISAAC_RESOURCE in instance.mcp.resources
        assert "ISAAC SIM —" in instance.mcp.instructions
        report = instance.get_capabilities()["isaac"]
        assert report["enabled"] is True and report["skipped"] is None

    def test_default_off_on_macos(
        self,
        make_server: Callable[..., server_module.SimulMCPServer],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        _on(monkeypatch, "darwin")
        # The packaged logging config stops propagation at simul.mcp.server,
        # so listen on the server's own logger rather than the root.
        server_logger = logging.getLogger(f"{server_module.__name__}.SimulMCPServer")
        previous_level = server_logger.level
        server_logger.addHandler(caplog.handler)
        server_logger.setLevel(logging.INFO)
        try:
            instance = make_server()
        finally:
            server_logger.removeHandler(caplog.handler)
            server_logger.setLevel(previous_level)
        assert _isaac_tools(instance) == []
        assert instance.mcp.resources == []
        assert "isaac" not in instance.mcp.instructions.lower()
        assert "load_usd_file" in instance.mcp.by_name
        assert "Isaac Sim" not in instance.tools_by_backend()
        report = instance.get_capabilities()["isaac"]
        assert report["enabled"] is False
        assert report["skipped"] == ISAAC_PLATFORM_SKIP
        assert f"Isaac Sim tools skipped: {ISAAC_PLATFORM_SKIP}" in caplog.text

    def test_enabled_true_on_macos(
        self,
        make_server: Callable[..., server_module.SimulMCPServer],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _on(monkeypatch, "darwin")
        monkeypatch.setenv("ISAAC_SIM__ENABLED", "true")
        instance = make_server(Settings())
        assert "ping_isaac" in _isaac_tools(instance)
        assert ISAAC_RESOURCE in instance.mcp.resources
        assert instance.get_capabilities()["isaac"]["skipped"] is None

    @pytest.mark.parametrize("backends", [None, {"isaac"}, {"isaac", "usd"}])
    def test_enabled_false_on_linux(
        self,
        make_server: Callable[..., server_module.SimulMCPServer],
        monkeypatch: pytest.MonkeyPatch,
        backends: Optional[Set[str]],
    ) -> None:
        _on(monkeypatch, "linux")
        monkeypatch.setenv("ISAAC_SIM__ENABLED", "false")
        instance = make_server(Settings(), backends)
        assert _isaac_tools(instance) == []
        assert instance.mcp.resources == []
        assert instance.get_capabilities()["isaac"]["skipped"] == ISAAC_DISABLED_SKIP

    @pytest.mark.parametrize("backends", [{"isaac"}, {"isaac", "usd"}])
    def test_naming_isaac_registers_on_macos(
        self,
        make_server: Callable[..., server_module.SimulMCPServer],
        monkeypatch: pytest.MonkeyPatch,
        backends: Set[str],
    ) -> None:
        _on(monkeypatch, "darwin")
        instance = make_server(backends=backends)
        assert "ping_isaac" in _isaac_tools(instance)
        assert "ISAAC SIM —" in instance.mcp.instructions
        assert instance.get_capabilities()["isaac"]["enabled"] is True

    def test_listing_mode_honours_the_gate(
        self, fake_fastmcp: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``simul tools`` lists missing runtimes, but not a backend the server would skip."""
        _on(monkeypatch, "darwin")
        instance = server_module.SimulMCPServer(
            settings=Settings(), require_available=False
        )
        assert _isaac_tools(instance) == []
        listed = server_module.SimulMCPServer(
            settings=Settings(), backends={"isaac"}, require_available=False
        )
        assert "ping_isaac" in _isaac_tools(listed)

    def test_capabilities_report_why_a_backend_is_off(
        self,
        make_server: Callable[..., server_module.SimulMCPServer],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _on(monkeypatch, "darwin")
        report = make_server(backends={"usd"}).get_capabilities()
        assert report["usd"]["skipped"] is None
        assert {report[name]["skipped"] for name in ("isaac", "blender", "unreal")} == {
            NOT_SELECTED
        }


class TestInstructions:
    def test_full_registry_states_every_backend(self) -> None:
        text = server_module.build_instructions(
            ALL_BACKEND_NAMES,
            {"apply_simready_metadata": "blender", "convert_to_simready": "unreal"},
        )
        for spec in BACKENDS:
            assert spec.summary in text, spec.name
            assert spec.routing_rule in text, spec.name
            assert spec.guidance in text, spec.name
        assert (
            "Tools containing 'simready' (SimReady asset helpers) belong to Blender or Unreal;"
            in text
        )

    def test_unregistered_backends_are_not_mentioned(self) -> None:
        text = server_module.build_instructions({"usd"})
        for token in ("isaac", "unreal", "blender", "simready"):
            assert token not in text.lower(), token
        assert backends_module.backend_spec("usd").routing_rule in text
        assert "get_tool_usage_stats" in text

    def test_without_isaac(self) -> None:
        text = server_module.build_instructions({"usd", "unreal"})
        assert "isaac" not in text.lower()
        assert "Tools containing 'unreal'" in text

    def test_the_simready_line_needs_a_registered_simready_tool(self) -> None:
        """A registered backend is not enough: its thin surface has no SimReady helper."""
        thin = server_module.build_instructions(
            {"usd", "unreal"}, {"capture_unreal_viewport": "unreal"}
        )
        assert "simready" not in thin.lower()
        full = server_module.build_instructions(
            {"usd", "unreal"},
            {"capture_unreal_viewport": "unreal", "convert_to_simready": "unreal"},
        )
        assert (
            "Tools containing 'simready' (SimReady asset helpers) belong to Unreal;"
            in full
        )
        assert "Blender" not in full


# ---------------------------------------------------------------------------
# The CLI: server banner, info, tools; the isaac sub-commands are untouched
# ---------------------------------------------------------------------------
class TestCli:
    @pytest.fixture
    def started(self, monkeypatch: pytest.MonkeyPatch) -> List[Dict[str, Any]]:
        """Stop before serving and record whether Isaac was probed."""
        seen: List[Dict[str, Any]] = []

        async def _capture(settings: Settings, transport: str, **kwargs: Any) -> None:
            seen.append({"settings": settings, **kwargs})

        def _probe(*args: Any, **kwargs: Any) -> bool:
            seen.append({"probed": args})
            return False

        monkeypatch.setattr(cli_main, "start_mcp_server", _capture)
        monkeypatch.setattr(cli_main, "_is_isaac_reachable", _probe)
        return seen

    def test_server_banner_on_macos_skips_isaac_without_probing(
        self, monkeypatch: pytest.MonkeyPatch, started: List[Dict[str, Any]]
    ) -> None:
        _on(monkeypatch, "darwin")
        result = runner.invoke(app, ["server"])
        assert result.exit_code == 0, result.output
        assert "Isaac Sim: skipped" in result.output
        assert "ISAAC_SIM__ENABLED=true" in _flat(result.output)
        assert [entry for entry in started if "probed" in entry] == []
        assert started[-1]["backends"] is None

    def test_server_banner_with_isaac_named_probes_it(
        self, monkeypatch: pytest.MonkeyPatch, started: List[Dict[str, Any]]
    ) -> None:
        _on(monkeypatch, "darwin")
        result = runner.invoke(app, ["server", "--backends", "isaac"])
        assert result.exit_code == 0, result.output
        assert "Isaac Sim (TCP :8226): not reachable" in result.output
        assert "skipped" not in result.output
        assert started[-1]["backends"] == {"isaac"}

    def test_server_banner_on_linux_is_unchanged(
        self, monkeypatch: pytest.MonkeyPatch, started: List[Dict[str, Any]]
    ) -> None:
        _on(monkeypatch, "linux")
        result = runner.invoke(app, ["server"])
        assert result.exit_code == 0, result.output
        assert "Isaac Sim (TCP :8226)" in result.output
        assert "skipped" not in result.output

    def test_info_json_on_macos(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _on(monkeypatch, "darwin")
        monkeypatch.setattr(
            cli_main, "_is_isaac_reachable", lambda *a, **k: pytest.fail("probed Isaac")
        )
        result = runner.invoke(app, ["--json", "info"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert payload["backends"]["isaac_sim"]["reachable"] is None
        assert payload["backends"]["isaac_sim"]["skipped"] == ISAAC_PLATFORM_SKIP
        assert payload["capabilities"]["isaac"]["skipped"] == ISAAC_PLATFORM_SKIP
        assert "Isaac Sim" not in payload["categories"]

    def test_info_table_on_macos(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _on(monkeypatch, "darwin")
        monkeypatch.setattr(cli_main, "is_json_mode", lambda: False)
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0, result.output
        flat = _flat(result.output)
        assert "Skipped" in flat
        assert "--backends isaac" in flat

    def test_info_with_isaac_named_on_macos(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _on(monkeypatch, "darwin")
        monkeypatch.setattr(cli_main, "_is_isaac_reachable", lambda *a, **k: False)
        result = runner.invoke(app, ["--json", "info", "--backends", "isaac"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert payload["backends"]["isaac_sim"]["reachable"] is False
        assert payload["backends"]["isaac_sim"]["skipped"] is None
        assert "ping_isaac" in payload["categories"]["Isaac Sim"]

    def test_tools_json_on_macos(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _on(monkeypatch, "darwin")
        result = runner.invoke(app, ["--json", "tools"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        isaac = payload["backends"]["isaac"]
        assert isaac["skipped"] == ISAAC_PLATFORM_SKIP
        assert isaac["tools"] == [] and isaac["tool_count"] == 0
        names = [
            tool["name"]
            for group in payload["backends"].values()
            for tool in group["tools"]
        ]
        assert names and not [name for name in names if "isaac" in name]
        assert payload["backends"]["usd"]["skipped"] is None

    @pytest.mark.parametrize(
        "argv", [["--backends", "isaac"], ["--backends", "usd,isaac"]]
    )
    def test_tools_json_with_isaac_named_on_macos(
        self, monkeypatch: pytest.MonkeyPatch, argv: List[str]
    ) -> None:
        _on(monkeypatch, "darwin")
        result = runner.invoke(app, ["--json", "tools", *argv])
        assert result.exit_code == 0, result.output
        isaac = json.loads(result.stdout)["backends"]["isaac"]
        assert isaac["skipped"] is None
        assert "ping_isaac" in {tool["name"] for tool in isaac["tools"]}

    def test_tools_json_with_isaac_enabled_on_macos(
        self, monkeypatch: pytest.MonkeyPatch, fresh_settings: None
    ) -> None:
        _on(monkeypatch, "darwin")
        monkeypatch.setenv("ISAAC_SIM__ENABLED", "true")
        result = runner.invoke(app, ["--json", "tools"])
        assert result.exit_code == 0, result.output
        isaac = json.loads(result.stdout)["backends"]["isaac"]
        assert isaac["skipped"] is None
        assert isaac["tool_count"] > 50

    def test_tools_json_with_isaac_disabled_on_linux(
        self, monkeypatch: pytest.MonkeyPatch, fresh_settings: None
    ) -> None:
        _on(monkeypatch, "linux")
        monkeypatch.setenv("ISAAC_SIM__ENABLED", "false")
        result = runner.invoke(app, ["--json", "tools", "--backends", "isaac"])
        assert result.exit_code == 0, result.output
        isaac = json.loads(result.stdout)["backends"]["isaac"]
        assert isaac["skipped"] == ISAAC_DISABLED_SKIP
        assert isaac["tools"] == []

    def test_tools_table_on_macos(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _on(monkeypatch, "darwin")
        monkeypatch.setattr(cli_main, "is_json_mode", lambda: False)
        result = runner.invoke(app, ["tools"])
        assert result.exit_code == 0, result.output
        flat = _flat(result.output)
        assert "skipped:" in flat
        assert "ISAAC_SIM__ENABLED=true" in flat
        assert "ping_isaac" not in flat

    def test_isaac_commands_ignore_the_gate(
        self, monkeypatch: pytest.MonkeyPatch, fresh_settings: None
    ) -> None:
        _on(monkeypatch, "darwin")
        monkeypatch.setenv("ISAAC_SIM__ENABLED", "false")

        async def _ping() -> bool:
            return True

        client = SimpleNamespace(address="10.0.0.5:8226", ping=_ping)
        monkeypatch.setattr(
            isaac_cli, "_tools", lambda *args, **kwargs: SimpleNamespace(_client=client)
        )
        result = runner.invoke(app, ["--json", "isaac", "ping"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)["reachable"] is True
