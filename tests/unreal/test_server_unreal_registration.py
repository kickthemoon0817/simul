"""Tests for Unreal Engine tool registration inside FastMCP server."""

import asyncio
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
import sys

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

    def resource(self, *args, **kwargs):
        """Stub for resource registration."""
        def decorator(func):
            return func
        return decorator

    def add_middleware(self, middleware: Any) -> None:
        """Stub for FastMCP middleware registration.

        SimulMCPServer adds a request-context middleware (PR #23) before any
        tools register. The double records the middleware so a future test
        can assert ordering, but registration tests only need it to not raise.
        """
        if not hasattr(self, "middlewares"):
            self.middlewares = []
        self.middlewares.append(middleware)


class FakeUnrealAdapter:
    """Minimal Unreal adapter stub for registration tests."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def is_available(self) -> bool:
        return True

    def get_capabilities(self):
        return ["unreal_health_check", "unreal_engine_info", "unreal_loaded_map"]

    @contextmanager
    def create_session(self):
        session = SimpleNamespace(
            health_check=_coro(
                {
                    "connected": True,
                    "engine_version": "5.4.0",
                    "project_name": "Test",
                    "is_editor": True,
                }
            ),
            get_engine_info=_coro(
                {
                    "engine_version": "5.4.0",
                    "project_name": "Test",
                    "loaded_map": "/Game/Maps/Test",
                    "is_editor": True,
                    "is_game": False,
                    "platform": "Win64",
                }
            ),
            get_loaded_map=_coro({"map_path": "/Game/Maps/Test"}),
        )
        yield session


def _coro(value: Dict[str, Any]):
    """Return an async callable that resolves to *value*."""

    async def _inner(*args, **kwargs):
        return value

    return _inner


class TestUnrealToolRegistration:
    """Test Unreal tool registration behavior in server."""

    def test_server_registers_thin_unreal_tools_by_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Default (thin) mode registers only essential Unreal MCP tools."""
        monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
        monkeypatch.setattr(server_module, "TaskConfig", None)

        monkeypatch.setattr(server_module, "is_headless_available", lambda: False)
        monkeypatch.setattr(server_module, "is_blender_available", lambda: False)
        monkeypatch.setattr(
            server_module, "UnrealRuntimeAdapter", FakeUnrealAdapter
        )

        instance = server_module.SimulMCPServer(settings=Settings())
        assert instance.unreal_adapter is not None
        tool_names = {tool.name for tool in instance.mcp.tools}

        # Thin MCP set: only 3 essential tools
        assert "unreal_health_check" in tool_names
        assert "capture_unreal_viewport" in tool_names
        assert "execute_unreal_script" in tool_names
        # Full tools should NOT be registered in thin mode
        assert "get_unreal_engine_info" not in tool_names
        assert "list_unreal_actors" not in tool_names
        assert "spawn_unreal_actor" not in tool_names
        # Blender tools should NOT be registered
        assert "get_blender_info" not in tool_names

    def test_server_skips_unreal_tools_when_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Unreal tools are not registered when runtime is unavailable."""
        monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
        monkeypatch.setattr(server_module, "TaskConfig", None)

        monkeypatch.setattr(server_module, "is_headless_available", lambda: False)
        monkeypatch.setattr(server_module, "is_blender_available", lambda: False)
        monkeypatch.setattr(server_module, "UnrealRuntimeAdapter", None)

        instance = server_module.SimulMCPServer(settings=Settings())
        assert instance.unreal_adapter is None
        tool_names = {tool.name for tool in instance.mcp.tools}

        assert "unreal_health_check" not in tool_names
        assert "capture_unreal_viewport" not in tool_names
        assert "execute_unreal_script" not in tool_names
        assert "list_unreal_actors" not in tool_names
        assert "spawn_unreal_actor" not in tool_names
        assert "delete_unreal_actor" not in tool_names
        assert "set_unreal_actor_transform" not in tool_names
