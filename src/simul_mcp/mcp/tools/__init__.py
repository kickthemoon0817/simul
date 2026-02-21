"""
MCP tools for Simul MCP Server.

This package provides individual MCP tool implementations for USD file operations.
Isaac Sim tools are registered directly in the server via TCP socket execution.
"""

from .usd_tools import (
    USDFileTools,
    USDSceneTools,
    USDMeshTools,
    USDBBoxTools,
)

from .blender_tools import BlenderTools
from .isaac_tools import IsaacTools
from .unreal_tools import UnrealTools

from .registry import (
    ToolRegistry,
    register_all_tools,
    get_tool_registry,
)

__all__ = [
    # USD tools
    "USDFileTools",
    "USDSceneTools",
    "USDMeshTools",
    "USDBBoxTools",
    # Blender tools
    "BlenderTools",
    # Isaac tools
    "IsaacTools",
    # Unreal tools
    "UnrealTools",
    # Registry
    "ToolRegistry",
    "register_all_tools",
    "get_tool_registry",
]
