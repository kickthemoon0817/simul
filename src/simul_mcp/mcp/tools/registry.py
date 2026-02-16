"""
Tool registry for Simul MCP Server.

This module provides a registry system for organizing and managing
MCP tools with automatic discovery and registration.
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

from ...logging import get_logger, LoggerMixin
from ...config import Settings, get_settings
from ...adapters import (
    is_blender_available,
    is_headless_available,
    is_unreal_available,
)
from .usd_tools import USDFileTools, USDSceneTools, USDMeshTools, USDBBoxTools
from .blender_tools import BlenderTools

logger = get_logger(__name__)


class ToolCategory(str, Enum):
    """Tool categories."""

    USD_FILE = "usd_file"
    USD_SCENE = "usd_scene"
    USD_MESH = "usd_mesh"
    USD_BBOX = "usd_bbox"
    BLENDER = "blender"
    UNREAL_SCENE = "unreal_scene"
    UNREAL_VIEWPORT = "unreal_viewport"
    UNREAL_MANIPULATION = "unreal_manipulation"
    UNREAL_MATERIAL = "unreal_material"
    UNREAL_PHYSICS = "unreal_physics"
    UNREAL_USD = "unreal_usd"
    UNREAL_AGENT = "unreal_agent"
    UNREAL_GEOMETRY = "unreal_geometry"


@dataclass
class ToolInfo:
    """Information about a registered tool."""

    name: str
    category: ToolCategory
    description: str
    method: Callable
    requires_blender: bool = False
    requires_unreal: bool = False
    requires_usd: bool = True
    enabled: bool = True


class ToolRegistry(LoggerMixin):
    """
    Registry for MCP tools.

    Manages tool registration, discovery, and availability checking
    based on runtime environment capabilities.
    """

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize tool registry.

        Args:
            settings: Configuration settings
        """
        self.settings = settings or get_settings()
        self._tools: Dict[str, ToolInfo] = {}
        self._tool_instances: Dict[ToolCategory, Any] = {}

        # Check runtime capabilities
        self._blender_available = is_blender_available()
        self._unreal_available = is_unreal_available()
        self._usd_available = is_headless_available()

        self.logger.info(
            "Tool registry initialized - Blender: %s, Unreal: %s, USD: %s",
            self._blender_available,
            self._unreal_available,
            self._usd_available,
        )

    def register_tool(
        self,
        name: str,
        category: ToolCategory,
        method: Callable,
        description: str,
        requires_blender: bool = False,
        requires_unreal: bool = False,
        requires_usd: bool = True,
    ) -> None:
        """
        Register a tool.

        Args:
            name: Tool name
            category: Tool category
            method: Tool method
            description: Tool description
            requires_blender: Whether tool requires Blender runtime
            requires_unreal: Whether tool requires Unreal Engine runtime
            requires_usd: Whether tool requires USD support
        """
        # Check if tool can be enabled based on requirements
        enabled = True

        if requires_blender and not self._blender_available:
            enabled = False
            self.logger.debug(f"Tool {name} disabled - Blender runtime not available")

        if requires_unreal and not self._unreal_available:
            enabled = False
            self.logger.debug(f"Tool {name} disabled - Unreal runtime not available")

        if requires_usd and not self._usd_available:
            enabled = False
            self.logger.debug(f"Tool {name} disabled - USD support not available")

        tool_info = ToolInfo(
            name=name,
            category=category,
            description=description,
            method=method,
            requires_blender=requires_blender,
            requires_unreal=requires_unreal,
            requires_usd=requires_usd,
            enabled=enabled,
        )

        self._tools[name] = tool_info
        self.logger.debug(
            f"Registered tool: {name} ({'enabled' if enabled else 'disabled'})"
        )

    def get_tool(self, name: str) -> Optional[ToolInfo]:
        """
        Get a tool by name.

        Args:
            name: Tool name

        Returns:
            ToolInfo or None if not found
        """
        return self._tools.get(name)

    def get_tools_by_category(self, category: ToolCategory) -> List[ToolInfo]:
        """
        Get all tools in a category.

        Args:
            category: Tool category

        Returns:
            List of ToolInfo objects
        """
        return [tool for tool in self._tools.values() if tool.category == category]

    def get_enabled_tools(self) -> List[ToolInfo]:
        """
        Get all enabled tools.

        Returns:
            List of enabled ToolInfo objects
        """
        return [tool for tool in self._tools.values() if tool.enabled]

    def get_all_tools(self) -> List[ToolInfo]:
        """
        Get all registered tools.

        Returns:
            List of all ToolInfo objects
        """
        return list(self._tools.values())

    def is_tool_available(self, name: str) -> bool:
        """
        Check if a tool is available.

        Args:
            name: Tool name

        Returns:
            True if tool is available and enabled
        """
        tool = self.get_tool(name)
        return tool is not None and tool.enabled

    def get_tool_instance(self, category: ToolCategory) -> Optional[Any]:
        """
        Get tool instance for a category.

        Args:
            category: Tool category

        Returns:
            Tool instance or None
        """
        if category not in self._tool_instances:
            # Create tool instance on demand
            instance = self._create_tool_instance(category)
            if instance:
                self._tool_instances[category] = instance

        return self._tool_instances.get(category)

    def _create_tool_instance(self, category: ToolCategory) -> Optional[Any]:
        """Create a tool instance for a category."""
        try:
            if category == ToolCategory.USD_FILE:
                return USDFileTools(self.settings)
            elif category == ToolCategory.USD_SCENE:
                return USDSceneTools(self.settings)
            elif category == ToolCategory.USD_MESH:
                return USDMeshTools(self.settings)
            elif category == ToolCategory.USD_BBOX:
                return USDBBoxTools(self.settings)
            elif category == ToolCategory.BLENDER:
                if self._blender_available:
                    return BlenderTools(self.settings)
            elif category.value.startswith("unreal_"):
                if self._unreal_available:
                    from .unreal_tools import UnrealTools

                    return UnrealTools(self.settings)
        except Exception as e:
            self.logger.error(f"Error creating tool instance for {category}: {e}")

        return None

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get registry capabilities summary.

        Returns:
            Capabilities dictionary
        """
        enabled_tools = self.get_enabled_tools()

        capabilities = {
            "blender_available": self._blender_available,
            "unreal_available": self._unreal_available,
            "usd_available": self._usd_available,
            "total_tools": len(self._tools),
            "enabled_tools": len(enabled_tools),
            "categories": {},
            "tools": {},
        }

        # Count tools by category
        for category in ToolCategory:
            category_tools = self.get_tools_by_category(category)
            enabled_count = len([t for t in category_tools if t.enabled])
            capabilities["categories"][category.value] = {
                "total": len(category_tools),
                "enabled": enabled_count,
            }

        # List all tools with status
        for tool in self._tools.values():
            capabilities["tools"][tool.name] = {
                "category": tool.category.value,
                "enabled": tool.enabled,
                "requires_blender": tool.requires_blender,
                "requires_unreal": tool.requires_unreal,
                "requires_usd": tool.requires_usd,
                "description": tool.description,
            }

        return capabilities


# Global registry instance
_registry: Optional[ToolRegistry] = None


def get_tool_registry(settings: Optional[Settings] = None) -> ToolRegistry:
    """
    Get the global tool registry instance.

    Args:
        settings: Configuration settings

    Returns:
        ToolRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = ToolRegistry(settings)
        register_all_tools(_registry)
    return _registry


def register_all_tools(registry: ToolRegistry) -> None:
    """
    Register all available tools with the registry.

    Args:
        registry: ToolRegistry instance
    """
    # USD File Tools
    usd_file_tools = USDFileTools()
    registry.register_tool(
        "load_usd_file",
        ToolCategory.USD_FILE,
        usd_file_tools.load_usd_file,
        "Load a USD file and return stage information",
        requires_usd=True,
    )
    registry.register_tool(
        "validate_usd_file",
        ToolCategory.USD_FILE,
        usd_file_tools.validate_usd_file,
        "Validate a USD file without loading it",
        requires_usd=True,
    )

    # USD Scene Tools
    usd_scene_tools = USDSceneTools()
    registry.register_tool(
        "get_prim_info",
        ToolCategory.USD_SCENE,
        usd_scene_tools.get_prim_info,
        "Get information about a USD prim",
        requires_usd=True,
    )
    registry.register_tool(
        "create_prim",
        ToolCategory.USD_SCENE,
        usd_scene_tools.create_prim,
        "Create a prim in a USD stage",
        requires_usd=True,
    )
    registry.register_tool(
        "update_prim_attributes",
        ToolCategory.USD_SCENE,
        usd_scene_tools.update_prim_attributes,
        "Update attributes on a USD prim",
        requires_usd=True,
    )
    registry.register_tool(
        "delete_prim",
        ToolCategory.USD_SCENE,
        usd_scene_tools.delete_prim,
        "Delete a prim from a USD stage",
        requires_usd=True,
    )
    registry.register_tool(
        "search_prims",
        ToolCategory.USD_SCENE,
        usd_scene_tools.search_prims,
        "Search for prims in a USD stage",
        requires_usd=True,
    )
    registry.register_tool(
        "summarize_scene",
        ToolCategory.USD_SCENE,
        usd_scene_tools.summarize_scene,
        "Generate a summary of a USD scene",
        requires_usd=True,
    )

    # USD Mesh Tools
    usd_mesh_tools = USDMeshTools()
    registry.register_tool(
        "get_mesh_info",
        ToolCategory.USD_MESH,
        usd_mesh_tools.get_mesh_info,
        "Get mesh information for a mesh prim",
        requires_usd=True,
    )

    # USD BBox Tools
    usd_bbox_tools = USDBBoxTools()
    registry.register_tool(
        "get_bounding_box",
        ToolCategory.USD_BBOX,
        usd_bbox_tools.get_bounding_box,
        "Get bounding box for a prim or entire stage",
        requires_usd=True,
    )

    # Blender tools (only if Blender is available)
    if is_blender_available():
        blender_tools = BlenderTools()
        registry.register_tool(
            "get_blender_info",
            ToolCategory.BLENDER,
            blender_tools.get_blender_info,
            "Get information about Blender runtime",
            requires_blender=True,
            requires_usd=False,
        )
        registry.register_tool(
            "list_blender_scene_objects",
            ToolCategory.BLENDER,
            blender_tools.list_blender_scene_objects,
            "List objects from active Blender scene",
            requires_blender=True,
            requires_usd=False,
        )

    # Note: Isaac Sim tools (execute_isaac_script, ping_isaac) are registered
    # directly in server.py via TCP socket execution. They are not part of the
    # tool registry because they use IsaacSocketClient, not local adapters.

    logger.info(
        f"Registered {len(registry.get_all_tools())} tools, "
        f"{len(registry.get_enabled_tools())} enabled"
    )
