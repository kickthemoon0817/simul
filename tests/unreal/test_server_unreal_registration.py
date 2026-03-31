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

    def test_server_registers_unreal_tools_when_available(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Unreal tools are present in FastMCP registration set."""
        monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
        monkeypatch.setattr(server_module, "TaskConfig", None)

        monkeypatch.setattr(server_module, "is_headless_available", lambda: False)
        monkeypatch.setattr(server_module, "is_blender_available", lambda: False)
        monkeypatch.setattr(server_module, "is_unreal_available", lambda: True)
        monkeypatch.setattr(
            server_module, "UnrealRuntimeAdapter", FakeUnrealAdapter
        )

        instance = server_module.SimulMCPServer(settings=Settings())
        tool_names = {tool.name for tool in instance.mcp.tools}

        # Phase 0 tools
        assert "unreal_health_check" in tool_names
        assert "get_unreal_engine_info" in tool_names
        assert "get_unreal_loaded_map" in tool_names
        # Phase 1 tools
        assert "list_unreal_actors" in tool_names
        assert "get_unreal_actor_info" in tool_names
        assert "search_unreal_assets" in tool_names
        assert "describe_unreal_object" in tool_names
        assert "get_unreal_actor_thumbnail" in tool_names
        assert "summarize_unreal_scene" in tool_names
        # Phase 2 tools
        assert "capture_unreal_viewport" in tool_names
        assert "get_unreal_viewport_info" in tool_names
        assert "set_unreal_camera_view" in tool_names
        assert "focus_unreal_on_actor" in tool_names
        # Phase 3 tools
        assert "spawn_unreal_actor" in tool_names
        assert "delete_unreal_actor" in tool_names
        assert "set_unreal_actor_transform" in tool_names
        assert "set_unreal_actor_property" in tool_names
        assert "call_unreal_actor_function" in tool_names
        assert "set_unreal_actor_parent" in tool_names
        assert "add_unreal_component" in tool_names
        assert "set_unreal_actor_visibility" in tool_names
        # Phase 4 tools
        assert "get_unreal_material_info" in tool_names
        assert "set_unreal_material_params" in tool_names
        assert "create_unreal_material_instance" in tool_names
        assert "assign_unreal_material" in tool_names
        assert "set_unreal_light_params" in tool_names
        assert "set_unreal_render_settings" in tool_names
        # Phase 5 tools
        assert "control_unreal_simulation" in tool_names
        assert "get_unreal_simulation_status" in tool_names
        assert "enable_unreal_physics" in tool_names
        assert "set_unreal_collision" in tool_names
        assert "apply_unreal_force" in tool_names
        assert "set_unreal_physics_params" in tool_names
        # Phase 6 tools (USD / SimReady Bridge)
        assert "import_unreal_usd" in tool_names
        assert "export_unreal_usd" in tool_names
        assert "convert_to_simready" in tool_names
        assert "validate_simready_asset" in tool_names
        assert "get_unreal_interchange_info" in tool_names
        # Phase 7 tools (Advanced Agent Tools)
        assert "batch_unreal_operations" in tool_names
        assert "query_unreal_scene_graph" in tool_names
        assert "analyze_unreal_scene_for_robotics" in tool_names
        assert "generate_unreal_procedural_scene" in tool_names
        assert "get_unreal_actor_by_semantic_label" in tool_names
        # Phase 8 tools (Geometry & Modeling)
        assert "generate_unreal_mesh_primitive" in tool_names
        assert "apply_unreal_mesh_boolean" in tool_names
        assert "compute_unreal_convex_hull" in tool_names
        assert "decompose_unreal_convex_hull" in tool_names
        assert "edit_unreal_mesh_topology" in tool_names
        assert "subdivide_unreal_mesh" in tool_names
        assert "simplify_unreal_mesh" in tool_names
        assert "cut_unreal_mesh_plane" in tool_names
        assert "validate_unreal_mesh" in tool_names
        assert "convert_unreal_mesh_format" in tool_names
        assert "remesh_unreal_mesh" in tool_names
        assert "compute_unreal_mesh_uv" in tool_names
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
        monkeypatch.setattr(server_module, "is_unreal_available", lambda: False)

        instance = server_module.SimulMCPServer(settings=Settings())
        tool_names = {tool.name for tool in instance.mcp.tools}

        assert "unreal_health_check" not in tool_names
        assert "get_unreal_engine_info" not in tool_names
        assert "get_unreal_loaded_map" not in tool_names
        assert "list_unreal_actors" not in tool_names
        assert "summarize_unreal_scene" not in tool_names
        assert "capture_unreal_viewport" not in tool_names
        assert "spawn_unreal_actor" not in tool_names
        assert "delete_unreal_actor" not in tool_names
        assert "set_unreal_actor_transform" not in tool_names
