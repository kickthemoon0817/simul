"""
Tool registry for Isaac Sim MCP Server.

This module provides a registry system for organizing and managing
MCP tools with automatic discovery and registration.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from ...adapters import (
    is_blender_available,
    is_headless_available,
    is_isaac_available,
)
from ...config import Settings, get_settings
from ...logging import LoggerMixin, get_logger
from .blender_tools import BlenderTools
from .isaac_tools import CameraTools, RigidBodyTools, SimulationTools, ViewportTools
from .usd_tools import USDBBoxTools, USDFileTools, USDMeshTools, USDSceneTools

logger = get_logger(__name__)


class ToolCategory(str, Enum):
    """Tool categories."""

    USD_FILE = "usd_file"
    USD_SCENE = "usd_scene"
    USD_MESH = "usd_mesh"
    USD_BBOX = "usd_bbox"
    VIEWPORT = "viewport"
    SIMULATION = "simulation"
    RIGID_BODY = "rigid_body"
    CAMERA = "camera"
    BLENDER = "blender"
    SIMREADY = "simready"


@dataclass
class ToolInfo:
    """Information about a registered tool."""

    name: str
    category: ToolCategory
    description: str
    method: Callable
    requires_isaac: bool = False
    requires_blender: bool = False
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
        self._isaac_available = is_isaac_available()
        self._blender_available = is_blender_available()
        self._usd_available = is_headless_available() or self._isaac_available

        self.logger.info(
            "Tool registry initialized - Isaac: %s, Blender: %s, USD: %s",
            self._isaac_available,
            self._blender_available,
            self._usd_available,
        )

    def register_tool(
        self,
        name: str,
        category: ToolCategory,
        method: Callable,
        description: str,
        requires_isaac: bool = False,
        requires_blender: bool = False,
        requires_usd: bool = True,
    ) -> None:
        """
        Register a tool.

        Args:
            name: Tool name
            category: Tool category
            method: Tool method
            description: Tool description
            requires_isaac: Whether tool requires Isaac Sim
            requires_usd: Whether tool requires USD support
        """
        # Check if tool can be enabled based on requirements
        enabled = True
        if requires_isaac and not self._isaac_available:
            enabled = False
            self.logger.debug(f"Tool {name} disabled - Isaac Sim not available")

        if requires_blender and not self._blender_available:
            enabled = False
            self.logger.debug(f"Tool {name} disabled - Blender runtime not available")

        if requires_usd and not self._usd_available:
            enabled = False
            self.logger.debug(f"Tool {name} disabled - USD support not available")

        tool_info = ToolInfo(
            name=name,
            category=category,
            description=description,
            method=method,
            requires_isaac=requires_isaac,
            requires_blender=requires_blender,
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
            elif category == ToolCategory.VIEWPORT:
                if self._isaac_available:
                    return ViewportTools(self.settings)
            elif category == ToolCategory.SIMULATION:
                if self._isaac_available:
                    return SimulationTools(self.settings)
            elif category == ToolCategory.RIGID_BODY:
                if self._isaac_available:
                    return RigidBodyTools(self.settings)
            elif category == ToolCategory.CAMERA:
                if self._isaac_available:
                    return CameraTools(self.settings)
            elif category == ToolCategory.BLENDER:
                if self._blender_available:
                    return BlenderTools(self.settings)
            elif category == ToolCategory.SIMREADY:
                if self._blender_available:
                    return BlenderTools(self.settings)
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
            "isaac_available": self._isaac_available,
            "blender_available": self._blender_available,
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
                "requires_isaac": tool.requires_isaac,
                "requires_blender": tool.requires_blender,
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

    # Isaac Sim specific tools (only if Isaac is available)
    if is_isaac_available():
        # Viewport Tools
        viewport_tools = ViewportTools()
        registry.register_tool(
            "capture_viewport",
            ToolCategory.VIEWPORT,
            viewport_tools.capture_viewport,
            "Capture the Isaac Sim viewport",
            requires_isaac=True,
        )
        registry.register_tool(
            "get_viewport_info",
            ToolCategory.VIEWPORT,
            viewport_tools.get_viewport_info,
            "Get information about the current viewport",
            requires_isaac=True,
        )

        # Simulation Tools
        simulation_tools = SimulationTools()
        registry.register_tool(
            "control_simulation",
            ToolCategory.SIMULATION,
            simulation_tools.control_simulation,
            "Control Isaac Sim simulation (play, pause, stop, reset, step)",
            requires_isaac=True,
        )
        registry.register_tool(
            "get_simulation_status",
            ToolCategory.SIMULATION,
            simulation_tools.get_simulation_status,
            "Get current simulation status",
            requires_isaac=True,
        )

        rigid_body_tools = RigidBodyTools()
        registry.register_tool(
            "enable_rigid_body",
            ToolCategory.RIGID_BODY,
            rigid_body_tools.enable_rigid_body,
            "Enable rigid body physics on a prim",
            requires_isaac=True,
        )
        registry.register_tool(
            "set_rigid_body_velocity",
            ToolCategory.RIGID_BODY,
            rigid_body_tools.set_rigid_body_velocity,
            "Set rigid body linear or angular velocity",
            requires_isaac=True,
        )
        registry.register_tool(
            "get_rigid_body_state",
            ToolCategory.RIGID_BODY,
            rigid_body_tools.get_rigid_body_state,
            "Get rigid body physics state for a prim",
            requires_isaac=True,
        )

        # Camera Tools
        camera_tools = CameraTools()
        registry.register_tool(
            "set_camera_view",
            ToolCategory.CAMERA,
            camera_tools.set_camera_view,
            "Set camera view in the viewport",
            requires_isaac=True,
        )
        registry.register_tool(
            "get_camera_info",
            ToolCategory.CAMERA,
            camera_tools.get_camera_info,
            "Get information about the current camera",
            requires_isaac=True,
        )
        registry.register_tool(
            "focus_on_prim",
            ToolCategory.CAMERA,
            camera_tools.focus_on_prim,
            "Focus camera on a specific prim",
            requires_isaac=True,
        )

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
        registry.register_tool(
            "get_blender_object_info",
            ToolCategory.BLENDER,
            blender_tools.get_blender_object_info,
            "Get detailed information about a single Blender object",
            requires_blender=True,
            requires_usd=False,
        )
        registry.register_tool(
            "get_blender_mesh_info",
            ToolCategory.BLENDER,
            blender_tools.get_blender_mesh_info,
            "Get counts-only mesh geometry information",
            requires_blender=True,
            requires_usd=False,
        )
        registry.register_tool(
            "get_blender_bounding_box",
            ToolCategory.BLENDER,
            blender_tools.get_blender_bounding_box,
            "Get bounding box corners of a Blender object",
            requires_blender=True,
            requires_usd=False,
        )
        registry.register_tool(
            "search_blender_objects",
            ToolCategory.BLENDER,
            blender_tools.search_blender_objects,
            "Search for Blender objects by name pattern and type",
            requires_blender=True,
            requires_usd=False,
        )
        registry.register_tool(
            "summarize_blender_scene",
            ToolCategory.BLENDER,
            blender_tools.summarize_blender_scene,
            "Get high-level summary of the current Blender scene",
            requires_blender=True,
            requires_usd=False,
        )
        registry.register_tool(
            "get_blender_material_info",
            ToolCategory.BLENDER,
            blender_tools.get_blender_material_info,
            "Get material information with bounded node traversal",
            requires_blender=True,
            requires_usd=False,
        )
        registry.register_tool(
            "get_blender_distance_between",
            ToolCategory.BLENDER,
            blender_tools.get_blender_distance_between,
            "Compute Euclidean distance between two Blender objects",
            requires_blender=True,
            requires_usd=False,
        )
        registry.register_tool(
            "check_blender_object_bounds",
            ToolCategory.BLENDER,
            blender_tools.check_blender_object_bounds,
            "Check whether an object is within spatial bounds",
            requires_blender=True,
            requires_usd=False,
        )
        registry.register_tool(
            "capture_blender_viewport",
            ToolCategory.BLENDER,
            blender_tools.capture_blender_viewport,
            "Capture the Blender viewport as a base64 JPEG image",
            requires_blender=True,
            requires_usd=False,
        )
        registry.register_tool(
            "set_blender_camera_view",
            ToolCategory.BLENDER,
            blender_tools.set_blender_camera_view,
            "Set Blender camera location and rotation",
            requires_blender=True,
            requires_usd=False,
        )
        registry.register_tool(
            "get_blender_camera_info",
            ToolCategory.BLENDER,
            blender_tools.get_blender_camera_info,
            "Get Blender camera properties (lens, sensor, clip distances)",
            requires_blender=True,
            requires_usd=False,
        )
        registry.register_tool(
            "focus_blender_on_object",
            ToolCategory.BLENDER,
            blender_tools.focus_blender_on_object,
            "Focus the Blender camera on a specific object",
            requires_blender=True,
            requires_usd=False,
        )
        registry.register_tool(
            "get_blender_viewport_info",
            ToolCategory.BLENDER,
            blender_tools.get_blender_viewport_info,
            "Get Blender viewport and render settings summary",
            requires_blender=True,
            requires_usd=False,
        )
        registry.register_tool(
            "capture_blender_viewport_sequence",
            ToolCategory.BLENDER,
            blender_tools.capture_blender_viewport_sequence,
            "Capture Blender viewport at multiple frames",
            requires_blender=True,
            requires_usd=False,
        )

        # -- Phase 3: Scene manipulation tools ---
        registry.register_tool(
            "create_blender_object",
            ToolCategory.BLENDER,
            blender_tools.create_blender_object,
            "Create a new object in the Blender scene",
            requires_blender=True,
            requires_usd=False,
        )
        registry.register_tool(
            "delete_blender_object",
            ToolCategory.BLENDER,
            blender_tools.delete_blender_object,
            "Delete an object from the Blender scene",
            requires_blender=True,
            requires_usd=False,
        )
        registry.register_tool(
            "set_blender_object_transform",
            ToolCategory.BLENDER,
            blender_tools.set_blender_object_transform,
            "Set an object's location, rotation, and/or scale",
            requires_blender=True,
            requires_usd=False,
        )
        registry.register_tool(
            "set_blender_object_parent",
            ToolCategory.BLENDER,
            blender_tools.set_blender_object_parent,
            "Parent one object to another",
            requires_blender=True,
            requires_usd=False,
        )
        registry.register_tool(
            "clear_blender_object_parent",
            ToolCategory.BLENDER,
            blender_tools.clear_blender_object_parent,
            "Unparent an object",
            requires_blender=True,
            requires_usd=False,
        )
        registry.register_tool(
            "assign_blender_material",
            ToolCategory.BLENDER,
            blender_tools.assign_blender_material,
            "Create and assign a Principled BSDF material to an object",
            requires_blender=True,
            requires_usd=False,
        )
        registry.register_tool(
            "add_blender_modifier",
            ToolCategory.BLENDER,
            blender_tools.add_blender_modifier,
            "Add a modifier to an object",
            requires_blender=True,
            requires_usd=False,
        )
        registry.register_tool(
            "set_blender_light_params",
            ToolCategory.BLENDER,
            blender_tools.set_blender_light_params,
            "Set properties on a Blender light object",
            requires_blender=True,
            requires_usd=False,
        )

        # -- Phase 4: File I/O tools ------------------------------------------
        registry.register_tool(
            "open_blender_file",
            ToolCategory.BLENDER,
            blender_tools.open_blender_file,
            "Open a .blend file, replacing the current scene",
            requires_blender=True,
            requires_usd=False,
        )
        registry.register_tool(
            "save_blender_file",
            ToolCategory.BLENDER,
            blender_tools.save_blender_file,
            "Save the current scene to a .blend file",
            requires_blender=True,
            requires_usd=False,
        )
        registry.register_tool(
            "import_blender_file",
            ToolCategory.BLENDER,
            blender_tools.import_blender_file,
            "Import a file (OBJ/FBX/GLTF/USD/STL/PLY) into the scene",
            requires_blender=True,
            requires_usd=False,
        )
        registry.register_tool(
            "export_blender_file",
            ToolCategory.BLENDER,
            blender_tools.export_blender_file,
            "Export scene objects to a file (OBJ/FBX/GLTF/USD/STL/PLY)",
            requires_blender=True,
            requires_usd=False,
        )
        registry.register_tool(
            "get_blender_file_info",
            ToolCategory.BLENDER,
            blender_tools.get_blender_file_info,
            "Get information about the current .blend file",
            requires_blender=True,
            requires_usd=False,
        )

        # -- Phase 5: Animation & Timeline tools --------------------------------
        registry.register_tool(
            "set_blender_frame",
            ToolCategory.BLENDER,
            blender_tools.set_blender_frame,
            "Set the current animation frame",
            requires_blender=True,
            requires_usd=False,
        )
        registry.register_tool(
            "get_blender_frame",
            ToolCategory.BLENDER,
            blender_tools.get_blender_frame,
            "Get the current frame and animation range",
            requires_blender=True,
            requires_usd=False,
        )
        registry.register_tool(
            "set_blender_frame_range",
            ToolCategory.BLENDER,
            blender_tools.set_blender_frame_range,
            "Set the animation frame range",
            requires_blender=True,
            requires_usd=False,
        )
        registry.register_tool(
            "play_blender_animation",
            ToolCategory.BLENDER,
            blender_tools.play_blender_animation,
            "Control animation playback (play, stop, reverse)",
            requires_blender=True,
            requires_usd=False,
        )
        registry.register_tool(
            "insert_blender_keyframe",
            ToolCategory.BLENDER,
            blender_tools.insert_blender_keyframe,
            "Insert a keyframe on an object property",
            requires_blender=True,
            requires_usd=False,
        )
        registry.register_tool(
            "delete_blender_keyframe",
            ToolCategory.BLENDER,
            blender_tools.delete_blender_keyframe,
            "Delete a keyframe from an object property",
            requires_blender=True,
            requires_usd=False,
        )
        registry.register_tool(
            "get_blender_keyframes",
            ToolCategory.BLENDER,
            blender_tools.get_blender_keyframes,
            "Get keyframe summary for an object",
            requires_blender=True,
            requires_usd=False,
        )

        # -- Phase 6: Physics & Simulation tools --
        registry.register_tool(
            "setup_blender_rigid_body",
            ToolCategory.BLENDER,
            blender_tools.setup_blender_rigid_body,
            "Set up rigid body physics on an object",
            requires_blender=True,
            requires_usd=False,
        )
        registry.register_tool(
            "add_blender_force_field",
            ToolCategory.BLENDER,
            blender_tools.add_blender_force_field,
            "Add a force field to the scene",
            requires_blender=True,
            requires_usd=False,
        )
        registry.register_tool(
            "get_blender_force_field_info",
            ToolCategory.BLENDER,
            blender_tools.get_blender_force_field_info,
            "Get force field parameters for an object",
            requires_blender=True,
            requires_usd=False,
        )
        registry.register_tool(
            "add_blender_rigid_body_constraint",
            ToolCategory.BLENDER,
            blender_tools.add_blender_rigid_body_constraint,
            "Add a rigid body constraint between two objects",
            requires_blender=True,
            requires_usd=False,
        )
        registry.register_tool(
            "get_blender_constraint_info",
            ToolCategory.BLENDER,
            blender_tools.get_blender_constraint_info,
            "Get rigid body constraint info for an object",
            requires_blender=True,
            requires_usd=False,
        )
        registry.register_tool(
            "get_blender_physics_state",
            ToolCategory.BLENDER,
            blender_tools.get_blender_physics_state,
            "Get current physics state of an object",
            requires_blender=True,
            requires_usd=False,
        )
        registry.register_tool(
            "get_blender_object_trajectory",
            ToolCategory.BLENDER,
            blender_tools.get_blender_object_trajectory,
            "Sample object position over a frame range",
            requires_blender=True,
            requires_usd=False,
        )
        registry.register_tool(
            "bake_blender_simulation",
            ToolCategory.BLENDER,
            blender_tools.bake_blender_simulation,
            "Bake physics simulation for a frame range",
            requires_blender=True,
            requires_usd=False,
        )
        registry.register_tool(
            "free_blender_bake",
            ToolCategory.BLENDER,
            blender_tools.free_blender_bake,
            "Free baked physics simulation data",
            requires_blender=True,
            requires_usd=False,
        )

        # -- Scripting & mesh-from-data tools --
        registry.register_tool(
            "execute_blender_script",
            ToolCategory.BLENDER,
            blender_tools.execute_blender_script,
            "Execute arbitrary Python code inside Blender",
            requires_blender=True,
            requires_usd=False,
        )
        registry.register_tool(
            "create_blender_mesh_from_data",
            ToolCategory.BLENDER,
            blender_tools.create_blender_mesh_from_data,
            "Create a mesh from raw vertex/edge/face data",
            requires_blender=True,
            requires_usd=False,
        )

        # -- SimReady tools --
        registry.register_tool(
            "apply_simready_metadata",
            ToolCategory.SIMREADY,
            blender_tools.apply_simready_metadata,
            "Apply SimReady metadata to an object",
            requires_blender=True,
            requires_usd=False,
        )
        registry.register_tool(
            "get_simready_metadata",
            ToolCategory.SIMREADY,
            blender_tools.get_simready_metadata,
            "Get SimReady metadata from an object",
            requires_blender=True,
            requires_usd=False,
        )
        registry.register_tool(
            "validate_simready_compliance",
            ToolCategory.SIMREADY,
            blender_tools.validate_simready_compliance,
            "Validate objects against SimReady spec",
            requires_blender=True,
            requires_usd=False,
        )
        registry.register_tool(
            "export_simready_usd",
            ToolCategory.SIMREADY,
            blender_tools.export_simready_usd,
            "Export objects as SimReady USD",
            requires_blender=True,
            requires_usd=False,
        )
        registry.register_tool(
            "setup_simready_hierarchy",
            ToolCategory.SIMREADY,
            blender_tools.setup_simready_hierarchy,
            "Set up SimReady-compliant object hierarchy",
            requires_blender=True,
            requires_usd=False,
        )

    total = len(registry.get_all_tools())
    enabled = len(registry.get_enabled_tools())
    logger.info(f"Registered {total} tools, {enabled} enabled")
