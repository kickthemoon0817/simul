"""End-to-end discoverability test for the MCP protocol chain.

Validates that an AI agent connecting via MCP can discover what this
server does without scanning individual tool signatures:

    server name → instructions field → tool listing → tool descriptions
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, List

import pytest

src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_path))

from simul_mcp.config import Settings  # noqa: E402
from simul_mcp.mcp import backends as backends_module  # noqa: E402
from simul_mcp.mcp import server as server_module  # noqa: E402


class FakeFastMCP:
    """Minimal FastMCP test double that captures all registration metadata."""

    def __init__(self, name: str, version: str, **kwargs: Any):
        self.name = name
        self.version = version
        self.description = kwargs.get("description")
        self.instructions = kwargs.get("instructions")
        self.tools: List[SimpleNamespace] = []

    def tool(
        self, name: str, **kwargs: Any
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Return decorator that records tool metadata."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.tools.append(SimpleNamespace(name=name, func=func, kwargs=kwargs))
            return func

        return decorator

    def get_tools(self) -> List[SimpleNamespace]:
        """Mirror FastMCP get_tools API."""
        return self.tools

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
        """Server name must clearly indicate 3D simulation / DCC scope."""
        instance = _make_server(monkeypatch)
        name: str = instance.mcp.name
        assert "Simul" in name
        assert "3D" in name or "DCC" in name

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
        """Instructions must mention Isaac Sim, USD, Blender, Unreal."""
        monkeypatch.setattr(server_module, "_FASTMCP_SUPPORTS_INSTRUCTIONS", True)
        instance = _make_server(monkeypatch)
        text: str = instance.mcp.instructions
        for keyword in ("Isaac Sim", "USD", "Blender", "Unreal"):
            assert keyword in text, f"Missing domain keyword: {keyword}"

    # ------------------------------------------------------------------
    # 3. Core tool listing (USD gated on its adapter, Isaac always registered)
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

    def test_isaac_tools_always_registered(
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


class _AvailableAdapter:
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
    monkeypatch.setattr(backends_module, "BlenderRuntimeAdapter", _AvailableAdapter)
    monkeypatch.setattr(backends_module, "UnrealRuntimeAdapter", _AvailableAdapter)
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


class TestCapabilitiesReport:
    """``get_capabilities`` must cover all four backends."""

    def test_reports_every_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        report = _make_full_server(monkeypatch).get_capabilities()

        assert set(report) == {"isaac", "usd", "blender", "unreal"}
        for backend, entry in report.items():
            assert set(entry) == {"enabled", "available", "capabilities"}, backend
            assert entry["enabled"] is True, backend
            assert entry["available"] is True, backend
        assert "load_usd_files" in report["usd"]["capabilities"]
        assert report["blender"]["capabilities"] == ["stub_capability"]
        assert report["unreal"]["capabilities"] == ["stub_capability"]
        assert "socket" in report["isaac"]["capabilities"]

    def test_unavailable_backends_report_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(backends_module, "UnrealRuntimeAdapter", None)
        instance = _make_server(monkeypatch)
        instance._backends = {"usd"}

        report = instance.get_capabilities()

        assert report["usd"] == {"enabled": True, "available": False, "capabilities": []}
        assert report["blender"]["available"] is False
        assert report["unreal"] == {"enabled": False, "available": False, "capabilities": []}
        assert report["isaac"]["enabled"] is False
