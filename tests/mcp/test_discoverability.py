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


def _make_server(monkeypatch: pytest.MonkeyPatch) -> server_module.SimulMCPServer:
    """Instantiate SimulMCPServer with all adapters stubbed out."""
    monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
    monkeypatch.setattr(server_module, "TaskConfig", None)

    monkeypatch.setattr(server_module, "is_headless_available", lambda: False)
    monkeypatch.setattr(server_module, "is_blender_available", lambda: False)
    monkeypatch.setattr(server_module, "is_unreal_available", lambda: False)

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
    # 3. Core tool listing (USD + Isaac always registered)
    # ------------------------------------------------------------------
    def test_core_usd_tools_always_registered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """USD headless tools register regardless of adapter availability."""
        instance = _make_server(monkeypatch)
        tool_names = {t.name for t in instance.mcp.tools}
        expected_usd: frozenset[str] = frozenset({
            "load_usd_file",
            "get_prim_info",
            "search_prims",
            "get_mesh_info",
            "get_bounding_box",
        })
        missing = expected_usd - tool_names
        assert not missing, f"Missing core USD tools: {missing}"

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
