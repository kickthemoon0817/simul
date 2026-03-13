"""Tests for Blender tool registration inside FastMCP server."""

import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_path))

from simul_mcp.config import Settings  # noqa: E402
from simul_mcp.mcp import server as server_module  # noqa: E402


class FakeFastMCP:
    """Minimal FastMCP test double for tool registration."""

    def __init__(self, name: str, version: str, **kwargs: Any):
        self.name = name
        self.version = version
        self.description = kwargs.get("description")
        self.instructions = kwargs.get("instructions")
        self.tools: list[SimpleNamespace] = []

    def tool(self, name: str, **kwargs):
        """Return decorator that records tool metadata."""

        def decorator(func):
            self.tools.append(SimpleNamespace(name=name, func=func, kwargs=kwargs))
            return func

        return decorator

    def get_tools(self):
        """Mirror FastMCP get_tools API."""
        return self.tools


class FakeBlenderAdapter:
    """Minimal Blender adapter stub used during registration tests."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def is_available(self) -> bool:
        """Pretend Blender runtime is available."""
        return True

    def get_capabilities(self):
        """Return predictable Blender capability list."""
        return ["blender_runtime_info"]

    @contextmanager
    def create_session(self):
        """Yield minimal session object."""
        session = SimpleNamespace(
            get_runtime_info=lambda: {
                "version": [4, 1, 0],
                "version_string": "4.1.0",
                "binary_path": "/bin/blender",
                "background": True,
                "blend_file_path": None,
            },
            list_scene_objects=lambda **kwargs: {
                "collection": kwargs.get("collection_name"),
                "include_hidden": kwargs.get("include_hidden", False),
                "max_items": kwargs.get("max_items", 200),
                "count": 0,
                "objects": [],
                "truncated": False,
            },
        )
        yield session


class TestBlenderToolRegistration:
    """Test Blender tool registration behavior in server."""

    def test_server_registers_blender_tools_when_available(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Blender tools are present in FastMCP registration set."""
        monkeypatch.setattr(server_module, "FASTMCP_AVAILABLE", True)
        monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
        monkeypatch.setattr(server_module, "TaskConfig", None)

        monkeypatch.setattr(server_module, "is_headless_available", lambda: False)
        monkeypatch.setattr(server_module, "is_isaac_available", lambda: False)
        monkeypatch.setattr(server_module, "is_blender_available", lambda: True)
        monkeypatch.setattr(server_module, "BlenderRuntimeAdapter", FakeBlenderAdapter)

        instance = server_module.SimulMCPServer(settings=Settings())
        tool_names = {tool.name for tool in instance.mcp.tools}

        assert "get_blender_info" in tool_names
        assert "list_blender_scene_objects" in tool_names
        assert "capture_viewport" not in tool_names

    def test_server_skips_blender_tools_when_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Blender tools are not registered when runtime is unavailable."""
        monkeypatch.setattr(server_module, "FASTMCP_AVAILABLE", True)
        monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
        monkeypatch.setattr(server_module, "TaskConfig", None)

        monkeypatch.setattr(server_module, "is_headless_available", lambda: False)
        monkeypatch.setattr(server_module, "is_isaac_available", lambda: False)
        monkeypatch.setattr(server_module, "is_blender_available", lambda: False)

        instance = server_module.SimulMCPServer(settings=Settings())
        tool_names = {tool.name for tool in instance.mcp.tools}

        assert "get_blender_info" not in tool_names
        assert "list_blender_scene_objects" not in tool_names
