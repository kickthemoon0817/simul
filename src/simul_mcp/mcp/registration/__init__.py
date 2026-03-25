"""
Tool registration modules for Simul MCP Server.

Each module registers tool closures on the FastMCP server instance
for a specific backend (USD, Isaac Sim, Blender, Unreal Engine).
"""

from ._reg_blender import register_blender_tools
from ._reg_instance import register_instance_tools
from ._reg_isaac import register_isaac_tools
from ._reg_stats import register_stats_tools
from ._reg_unreal import register_unreal_tools
from ._reg_usd import register_usd_tools

__all__ = [
    "register_instance_tools",
    "register_usd_tools",
    "register_isaac_tools",
    "register_blender_tools",
    "register_unreal_tools",
    "register_stats_tools",
]
