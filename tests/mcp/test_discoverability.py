"""End-to-end discoverability test for the MCP protocol chain.

Validates that an AI agent connecting via MCP can discover what this
server does without scanning individual tool signatures:

    server name → instructions field → tool listing → tool descriptions
"""

import asyncio
from typing import List, Optional, Tuple

import pytest
from fastmcp import Client

from simul.config import Settings
from simul.mcp import backends as backends_module
from simul.mcp import server as server_module
from tests.fakes import FakeFastMCP


def _make_server(
    monkeypatch: pytest.MonkeyPatch, *, headless: bool = False
) -> server_module.SimulMCPServer:
    """Instantiate SimulMCPServer with the engine adapters stubbed out.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        headless: Whether the headless USD adapter should report as available.
    """
    monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
    monkeypatch.setattr(server_module, "TaskConfig", None)

    monkeypatch.setattr(backends_module, "is_headless_available", lambda: headless)
    monkeypatch.setattr(backends_module, "is_blender_available", lambda: False)
    monkeypatch.setattr(backends_module, "is_unreal_available", lambda: False)

    return server_module.SimulMCPServer(settings=Settings())


class TestMCPDiscoverability:
    """Validate the full MCP discoverability chain."""

    # ------------------------------------------------------------------
    # 1. Server identity
    # ------------------------------------------------------------------
    def test_server_name_identifies_purpose(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The server identifies as ``simul``; its instructions carry the 3D scope."""
        monkeypatch.setattr(server_module, "_FASTMCP_SUPPORTS_INSTRUCTIONS", True)
        instance = _make_server(monkeypatch)
        assert instance.mcp.name == "simul"
        assert "3D simulation" in instance.mcp.instructions

    def test_server_version_is_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Server version must be a non-empty semver-like string."""
        instance = _make_server(monkeypatch)
        version: str = instance.mcp.version
        assert version
        parts = version.split(".")
        assert len(parts) == 3, f"Expected semver, got {version}"
        assert all(p.isdigit() for p in parts)

    # ------------------------------------------------------------------
    # 2. Instructions field (protocol-level discoverability)
    # ------------------------------------------------------------------
    def test_instructions_field_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Instructions must be provided so agents understand scope."""
        monkeypatch.setattr(server_module, "_FASTMCP_SUPPORTS_INSTRUCTIONS", True)
        instance = _make_server(monkeypatch)
        assert instance.mcp.instructions is not None
        assert len(instance.mcp.instructions) > 0

    def test_instructions_mention_key_domains(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With every backend registered the instructions cover Isaac Sim, USD, Blender, Unreal."""
        monkeypatch.setattr(server_module, "_FASTMCP_SUPPORTS_INSTRUCTIONS", True)
        instance = _make_full_server(monkeypatch)
        text: str = instance.mcp.instructions
        for keyword in ("Isaac Sim", "USD", "Blender", "Unreal"):
            assert keyword in text, f"Missing domain keyword: {keyword}"

    def test_instructions_leave_out_unregistered_backends(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only Isaac registers here, so the instructions name no other engine."""
        monkeypatch.setattr(server_module, "_FASTMCP_SUPPORTS_INSTRUCTIONS", True)
        instance = _make_server(monkeypatch)
        text: str = instance.mcp.instructions
        assert "Isaac Sim" in text
        for keyword in ("Blender", "Unreal", "blender", "unreal", "load_usd_file"):
            assert keyword not in text, f"Mentions an unregistered backend: {keyword}"

    # ------------------------------------------------------------------
    # 3. Core tool listing (USD gated on its adapter; Isaac registered on a supported
    #    platform, which tests/conftest.py reports for the whole session)
    # ------------------------------------------------------------------
    _CORE_USD_TOOLS: frozenset[str] = frozenset({
        "load_usd_file",
        "get_prim_info",
        "search_prims",
        "get_mesh_info",
        "get_bounding_box",
    })

    def test_core_usd_tools_registered_when_adapter_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """USD headless tools register when the headless adapter imports."""
        instance = _make_server(monkeypatch, headless=True)
        tool_names = {t.name for t in instance.mcp.tools}
        missing = self._CORE_USD_TOOLS - tool_names
        assert not missing, f"Missing core USD tools: {missing}"

    def test_usd_tools_skipped_without_adapter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without the adapter the USD tools would fail on every call, so
        they must not be advertised at all."""
        instance = _make_server(monkeypatch, headless=False)
        tool_names = {t.name for t in instance.mcp.tools}
        leaked = self._CORE_USD_TOOLS & tool_names
        assert not leaked, f"USD tools registered without an adapter: {leaked}"

    def test_isaac_tools_registered_on_supported_platform(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Isaac Sim tools register even when Isaac is unreachable."""
        instance = _make_server(monkeypatch)
        tool_names = {t.name for t in instance.mcp.tools}
        assert "execute_isaac_script" in tool_names
        assert "ping_isaac" in tool_names

    # ------------------------------------------------------------------
    # 4. Tool descriptions contain actionable guidance
    # ------------------------------------------------------------------
    def test_tool_descriptions_are_nonempty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every registered tool must have a non-empty description."""
        instance = _make_server(monkeypatch)
        for tool in instance.mcp.tools:
            desc = tool.kwargs.get("description", "")
            assert desc, f"Tool {tool.name!r} has empty description"

    def test_execute_isaac_script_suggests_ping(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """execute_isaac_script description must mention ping_isaac."""
        instance = _make_server(monkeypatch)
        tool = next(
            (t for t in instance.mcp.tools if t.name == "execute_isaac_script"), None
        )
        assert tool is not None, "Tool 'execute_isaac_script' not registered"
        desc: str = tool.kwargs.get("description", "")
        assert "ping_isaac" in desc

    def test_ping_isaac_describes_preflight(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ping_isaac description must indicate it is a pre-flight check."""
        instance = _make_server(monkeypatch)
        tool = next(
            (t for t in instance.mcp.tools if t.name == "ping_isaac"), None
        )
        assert tool is not None, "Tool 'ping_isaac' not registered"
        desc: str = tool.kwargs.get("description", "")
        assert "pre-flight" in desc.lower() or "verify" in desc.lower()


class AvailableAdapter:
    """Adapter stub that reports itself available so its tools register."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def is_available(self) -> bool:
        return True

    def get_capabilities(self) -> List[str]:
        return ["stub_capability"]


BACKEND_TOKENS: frozenset[str] = frozenset({"isaac", "unreal", "blender"})
HEADLESS_USD_TOOLS: frozenset[str] = frozenset({
    "load_usd_file",
    "validate_usd_file",
    "get_prim_info",
    "create_prim",
    "update_prim_attributes",
    "delete_prim",
    "get_mesh_info",
    "search_prims",
    "get_bounding_box",
    "summarize_scene",
})
META_TOOLS: frozenset[str] = frozenset({"get_tool_usage_stats"})


def _make_full_server(monkeypatch: pytest.MonkeyPatch) -> server_module.SimulMCPServer:
    """Register every backend, with the full Unreal surface, on a FakeFastMCP."""
    monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
    monkeypatch.setattr(server_module, "TaskConfig", None)
    monkeypatch.setattr(backends_module, "is_headless_available", lambda: True)
    monkeypatch.setattr(backends_module, "is_blender_available", lambda: True)
    monkeypatch.setattr(backends_module, "BlenderRuntimeAdapter", AvailableAdapter)
    monkeypatch.setattr(backends_module, "UnrealRuntimeAdapter", AvailableAdapter)
    settings = Settings().model_copy(
        update={"unreal": Settings().unreal.model_copy(update={"tool_surface": "full"})}
    )
    return server_module.SimulMCPServer(settings=settings)


class TestRoutingRuleMatchesRegistry:
    """The ROUTING paragraph in the instructions must be true of every tool."""

    def test_instructions_state_the_infix_rule(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(server_module, "_FASTMCP_SUPPORTS_INSTRUCTIONS", True)
        text: str = _make_full_server(monkeypatch).mcp.instructions
        assert "appears somewhere inside the tool name" in text
        assert "Tools containing 'isaac'" in text
        assert "isaac_*" not in text
        assert "blender_*" not in text
        assert "unreal_*" not in text

    def test_every_tool_is_classifiable_by_the_stated_rule(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each tool carries exactly one backend token, is a SimReady helper
        whose description names its runtime, or is a headless USD / meta tool."""
        instance = _make_full_server(monkeypatch)
        assert len(instance.mcp.tools) > 150, "full registry expected"

        unclassifiable: List[str] = []
        for tool in instance.mcp.tools:
            tokens = {token for token in BACKEND_TOKENS if token in tool.name}
            if len(tokens) == 1:
                continue
            if tokens:
                unclassifiable.append(f"{tool.name} (several backend tokens: {sorted(tokens)})")
                continue
            if "simready" in tool.name:
                description: str = tool.kwargs.get("description", "")
                runtimes = {name for name in ("Blender", "Unreal") if name in description}
                if len(runtimes) != 1:
                    unclassifiable.append(
                        f"{tool.name} (SimReady description names {sorted(runtimes)})"
                    )
                continue
            if tool.name not in HEADLESS_USD_TOOLS | META_TOOLS:
                unclassifiable.append(f"{tool.name} (no backend token, not USD or meta)")

        assert not unclassifiable, "\n".join(unclassifiable)

    def test_simready_tools_exist_on_both_sides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Documents why the instructions single out the 'simready' family."""
        instance = _make_full_server(monkeypatch)
        descriptions = [
            t.kwargs.get("description", "") for t in instance.mcp.tools if "simready" in t.name
        ]
        assert any("Blender" in desc for desc in descriptions)
        assert any("Unreal" in desc for desc in descriptions)


def _macos_default_server(
    monkeypatch: pytest.MonkeyPatch, unreal_surface: str, *, fake: bool = True
) -> server_module.SimulMCPServer:
    """The macOS default: no Isaac Sim (darwin), no bpy, headless USD, Unreal on ``unreal_surface``."""
    if fake:
        monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
        monkeypatch.setattr(server_module, "TaskConfig", None)
        monkeypatch.setattr(server_module, "_FASTMCP_SUPPORTS_INSTRUCTIONS", True)
    monkeypatch.setattr(
        backends_module, "is_isaac_platform_supported", lambda platform=None: False
    )
    monkeypatch.setattr(backends_module, "is_headless_available", lambda: True)
    monkeypatch.setattr(backends_module, "is_blender_available", lambda: False)
    monkeypatch.setattr(backends_module, "is_unreal_available", lambda: True)
    monkeypatch.setattr(backends_module, "UnrealRuntimeAdapter", AvailableAdapter)
    return server_module.SimulMCPServer(
        settings=Settings(unreal={"tool_surface": unreal_surface})
    )


class TestSimReadyLine:
    """The instructions route 'simready' tools only when one is in tools/list.

    The line used to follow the registered backends, so the macOS default
    (thin Unreal surface, no bpy) told agents about SimReady tools that the
    thin surface never registers.
    """

    def test_macos_default_has_no_simready_tool_and_no_line(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        instance = _macos_default_server(monkeypatch, "thin")

        assert "capture_unreal_viewport" in instance.mcp.by_name
        assert [t.name for t in instance.mcp.tools if "simready" in t.name] == []
        assert "simready" not in instance.mcp.instructions.lower()

    def test_the_full_unreal_surface_states_the_line_for_unreal_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        instance = _macos_default_server(monkeypatch, "full")

        assert "convert_to_simready" in instance.mcp.by_name
        assert (
            "Tools containing 'simready' (SimReady asset helpers) belong to Unreal;"
            in instance.mcp.instructions
        )
        assert "Blender" not in instance.mcp.instructions

    @pytest.mark.parametrize(("surface", "simready"), [("thin", False), ("full", True)])
    def test_the_client_receives_the_instructions_written_after_registration(
        self, monkeypatch: pytest.MonkeyPatch, surface: str, simready: bool
    ) -> None:
        """Over the real FastMCP protocol, initialize carries what tools/list matches."""
        instance = _macos_default_server(monkeypatch, surface, fake=False)

        async def handshake() -> Tuple[Optional[str], List[str]]:
            async with Client(instance.mcp) as client:
                names = [tool.name for tool in await client.list_tools()]
                assert client.initialize_result is not None
                return client.initialize_result.instructions, names

        instructions, names = asyncio.run(handshake())

        assert instructions is not None
        assert instructions == instance.mcp.instructions
        assert any("simready" in name for name in names) is simready
        assert ("simready" in instructions.lower()) is simready

    @pytest.mark.parametrize(
        ("blender_surface", "unreal_surface", "owners"),
        [
            ("thin", "full", ["Unreal"]),
            ("full", "thin", ["Blender"]),
            ("full", "full", ["Blender", "Unreal"]),
        ],
    )
    def test_only_the_simready_line_routes_simready_tools(
        self,
        monkeypatch: pytest.MonkeyPatch,
        blender_surface: str,
        unreal_surface: str,
        owners: List[str],
    ) -> None:
        """One ROUTING line claims the 'simready' tools, naming the backends that registered them.

        The Blender rule also claimed every tool containing 'simready', so with
        the thin Blender surface (no SimReady tools) and the full Unreal one it
        sent Unreal's convert_to_simready to Blender, contradicting the
        SimReady line below it.
        """
        monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
        monkeypatch.setattr(server_module, "TaskConfig", None)
        monkeypatch.setattr(server_module, "_FASTMCP_SUPPORTS_INSTRUCTIONS", True)
        monkeypatch.setattr(backends_module, "is_blender_available", lambda: True)
        monkeypatch.setattr(backends_module, "BlenderRuntimeAdapter", AvailableAdapter)
        monkeypatch.setattr(backends_module, "is_unreal_available", lambda: True)
        monkeypatch.setattr(backends_module, "UnrealRuntimeAdapter", AvailableAdapter)
        instance = server_module.SimulMCPServer(
            settings=Settings(
                blender={"tool_surface": blender_surface},
                unreal={"tool_surface": unreal_surface},
            ),
            backends={"blender", "unreal"},
        )

        routing = instance.mcp.instructions.split("ROUTING")[1]
        claims = [line.strip() for line in routing.splitlines() if "'simready'" in line]
        assert claims == [
            "Tools containing 'simready' (SimReady asset helpers) belong to "
            f"{' or '.join(owners)}; each one's description names the runtime it targets."
        ]


class TestCapabilitiesReport:
    """``get_capabilities`` must cover all four backends."""

    def test_reports_every_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        report = _make_full_server(monkeypatch).get_capabilities()

        assert set(report) == {"isaac", "usd", "blender", "unreal"}
        for backend, entry in report.items():
            assert set(entry) == {"enabled", "available", "capabilities", "skipped"}, backend
            assert entry["enabled"] is True, backend
            assert entry["available"] is True, backend
            assert entry["skipped"] is None, backend
        assert "load_usd_files" in report["usd"]["capabilities"]
        assert report["blender"]["capabilities"] == ["stub_capability"]
        assert report["unreal"]["capabilities"] == ["stub_capability"]
        assert "socket" in report["isaac"]["capabilities"]

    def test_unavailable_backends_report_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(backends_module, "UnrealRuntimeAdapter", None)
        instance = _make_server(monkeypatch)
        instance._backends = {"usd"}

        report = instance.get_capabilities()

        assert report["usd"] == {"enabled": True, "available": False, "capabilities": [], "skipped": None}
        assert report["blender"]["available"] is False
        assert report["unreal"] == {
            "enabled": False,
            "available": False,
            "capabilities": [],
            "skipped": "left out of --backends",
        }
        assert report["isaac"]["enabled"] is False
