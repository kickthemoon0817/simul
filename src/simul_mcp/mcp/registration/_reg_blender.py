"""
Blender runtime tool registration for Simul MCP Server.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from fastmcp.tools.tool import ToolResult

from ..schemas.blender import *
from ..schemas.common import ErrorResponse
from ..schemas.simready import *

if TYPE_CHECKING:
    from ..server import SimulMCPServer


def register_blender_tools(server: "SimulMCPServer") -> None:
    """Register Blender runtime specific tools."""

    @server.mcp.tool(
        name="get_blender_info",
        description="Get information about the active Blender runtime.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def get_blender_info() -> Dict[str, Any]:
        """
        Get information about the active Blender runtime.

        Returns:
            Blender runtime information or an error response.
        """
        return await server._exec_backend(
            "get_blender_info",
            server.blender_adapter,
            "Blender",
            BlenderInfoResponse,
            lambda session: session.get_runtime_info(),
        )

    @server.mcp.tool(
        name="list_blender_scene_objects",
        description="List objects from the active Blender scene.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def list_blender_scene_objects(
        collection_name: Optional[str] = None,
        include_hidden: bool = False,
        max_items: int = server.settings.blender.max_scene_objects,
    ) -> Dict[str, Any]:
        """
        List objects from the active Blender scene.

        Args:
            collection_name: Optional collection name filter.
            include_hidden: Include hidden objects when true.
            max_items: Maximum number of objects to return.

        Returns:
            Blender object listing response or error response.
        """
        input_data = server._validate_input(
            BlenderSceneObjectsRequest,
            collection_name=collection_name,
            include_hidden=include_hidden,
            max_items=max_items,
        )
        if isinstance(input_data, dict):
            return input_data

        return await server._exec_backend(
            "list_blender_scene_objects",
            server.blender_adapter,
            "Blender",
            BlenderSceneObjectsResponse,
            lambda session: session.list_scene_objects(
                collection_name=input_data.collection_name,
                include_hidden=input_data.include_hidden,
                max_items=input_data.max_items,
            ),
        )

    # -- Phase 1: Core Observation tools ----------------------------------

    @server.mcp.tool(
        name="get_blender_object_info",
        description="Get detailed information about a single Blender object.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def get_blender_object_info(
        object_name: str,
    ) -> Dict[str, Any]:
        input_data = server._validate_input(
            BlenderObjectInfoRequest,
            object_name=object_name,
        )
        if isinstance(input_data, dict):
            return input_data
        return await server._exec_backend(
            "get_blender_object_info",
            server.blender_adapter,
            "Blender",
            BlenderObjectInfoResponse,
            lambda session: session.get_object_info(input_data.object_name),
        )

    @server.mcp.tool(
        name="get_blender_mesh_info",
        description="Get mesh geometry counts for a Blender mesh object.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def get_blender_mesh_info(
        object_name: str,
    ) -> Dict[str, Any]:
        input_data = server._validate_input(
            BlenderMeshInfoRequest,
            object_name=object_name,
        )
        if isinstance(input_data, dict):
            return input_data
        return await server._exec_backend(
            "get_blender_mesh_info",
            server.blender_adapter,
            "Blender",
            BlenderMeshInfoResponse,
            lambda session: session.get_mesh_info(input_data.object_name),
        )

    @server.mcp.tool(
        name="get_blender_bounding_box",
        description="Get the bounding box of a Blender object.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def get_blender_bounding_box(
        object_name: str,
        world_space: bool = True,
    ) -> Dict[str, Any]:
        input_data = server._validate_input(
            BlenderBoundingBoxRequest,
            object_name=object_name,
            world_space=world_space,
        )
        if isinstance(input_data, dict):
            return input_data
        return await server._exec_backend(
            "get_blender_bounding_box",
            server.blender_adapter,
            "Blender",
            BlenderBoundingBoxResponse,
            lambda session: session.get_bounding_box(
                input_data.object_name,
                input_data.world_space,
            ),
        )

    @server.mcp.tool(
        name="search_blender_objects",
        description=(
            "Search for objects in the Blender scene " "by name pattern or type."
        ),
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def search_blender_objects(
        name_pattern: Optional[str] = None,
        object_type: Optional[str] = None,
        max_results: int = 50,
    ) -> Dict[str, Any]:
        input_data = server._validate_input(
            BlenderSearchObjectsRequest,
            name_pattern=name_pattern,
            object_type=object_type,
            max_results=max_results,
        )
        if isinstance(input_data, dict):
            return input_data
        return await server._exec_backend(
            "search_blender_objects",
            server.blender_adapter,
            "Blender",
            BlenderSearchObjectsResponse,
            lambda session: session.search_objects(
                input_data.name_pattern,
                input_data.object_type,
                input_data.max_results,
            ),
        )

    @server.mcp.tool(
        name="summarize_blender_scene",
        description="Get a high-level summary of the Blender scene.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def summarize_blender_scene() -> Dict[str, Any]:
        return await server._exec_backend(
            "summarize_blender_scene",
            server.blender_adapter,
            "Blender",
            BlenderSceneSummaryResponse,
            lambda session: session.summarize_scene(),
        )

    @server.mcp.tool(
        name="get_blender_material_info",
        description="Get material information with bounded node tree traversal.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def get_blender_material_info(
        material_name: str,
    ) -> Dict[str, Any]:
        input_data = server._validate_input(
            BlenderMaterialInfoRequest,
            material_name=material_name,
        )
        if isinstance(input_data, dict):
            return input_data
        return await server._exec_backend(
            "get_blender_material_info",
            server.blender_adapter,
            "Blender",
            BlenderMaterialInfoResponse,
            lambda session: session.get_material_info(
                input_data.material_name,
            ),
        )

    @server.mcp.tool(
        name="get_blender_distance_between",
        description="Measure the Euclidean distance between two Blender objects.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def get_blender_distance_between(
        object_name_a: str,
        object_name_b: str,
    ) -> Dict[str, Any]:
        input_data = server._validate_input(
            BlenderDistanceRequest,
            object_name_a=object_name_a,
            object_name_b=object_name_b,
        )
        if isinstance(input_data, dict):
            return input_data
        return await server._exec_backend(
            "get_blender_distance_between",
            server.blender_adapter,
            "Blender",
            BlenderDistanceResponse,
            lambda session: session.get_distance_between(
                input_data.object_name_a,
                input_data.object_name_b,
            ),
        )

    @server.mcp.tool(
        name="check_blender_object_bounds",
        description="Check if a Blender object is within spatial bounds.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def check_blender_object_bounds(
        object_name: str,
        bounds_min: List[float],
        bounds_max: List[float],
    ) -> Dict[str, Any]:
        input_data = server._validate_input(
            BlenderBoundsCheckRequest,
            object_name=object_name,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
        )
        if isinstance(input_data, dict):
            return input_data
        return await server._exec_backend(
            "check_blender_object_bounds",
            server.blender_adapter,
            "Blender",
            BlenderBoundsCheckResponse,
            lambda session: session.check_object_bounds(
                input_data.object_name,
                input_data.bounds_min,
                input_data.bounds_max,
            ),
        )

    # -- Phase 2: Visual Observation tools --------------------------------

    @server.mcp.tool(
        name="capture_blender_viewport",
        description="Capture the Blender viewport as a base64-encoded JPEG image.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def capture_blender_viewport(
        width: int = 512,
        height: int = 512,
        jpeg_quality: int = 85,
        use_render_fallback: bool = False,
    ) -> Dict[str, Any]:
        input_data = server._validate_input(
            BlenderCaptureViewportRequest,
            width=width,
            height=height,
            jpeg_quality=jpeg_quality,
            use_render_fallback=use_render_fallback,
        )
        if isinstance(input_data, dict):
            return input_data
        return await server._exec_backend(
            "capture_blender_viewport",
            server.blender_adapter,
            "Blender",
            BlenderCaptureViewportResponse,
            lambda session: session.capture_viewport(
                input_data.width,
                input_data.height,
                input_data.jpeg_quality,
                input_data.use_render_fallback,
            ),
        )

    @server.mcp.tool(
        name="set_blender_camera_view",
        description="Set the active camera's location and rotation.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def set_blender_camera_view(
        location: List[float],
        rotation_euler: List[float],
        camera_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        input_data = server._validate_input(
            BlenderSetCameraViewRequest,
            location=location,
            rotation_euler=rotation_euler,
            camera_name=camera_name,
        )
        if isinstance(input_data, dict):
            return input_data
        return await server._exec_backend(
            "set_blender_camera_view",
            server.blender_adapter,
            "Blender",
            BlenderSetCameraViewResponse,
            lambda session: session.set_camera_view(
                list(input_data.location),
                list(input_data.rotation_euler),
                input_data.camera_name,
            ),
        )

    @server.mcp.tool(
        name="get_blender_camera_info",
        description="Get information about the active Blender camera.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def get_blender_camera_info(
        camera_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await server._exec_backend(
            "get_blender_camera_info",
            server.blender_adapter,
            "Blender",
            BlenderCameraInfoResponse,
            lambda session: session.get_camera_info(camera_name),
        )

    @server.mcp.tool(
        name="focus_blender_on_object",
        description="Focus the camera on a specific Blender object.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def focus_blender_on_object(
        object_name: str,
        distance_factor: float = 2.0,
        camera_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        input_data = server._validate_input(
            BlenderFocusOnObjectRequest,
            object_name=object_name,
            distance_factor=distance_factor,
            camera_name=camera_name,
        )
        if isinstance(input_data, dict):
            return input_data
        return await server._exec_backend(
            "focus_blender_on_object",
            server.blender_adapter,
            "Blender",
            BlenderFocusOnObjectResponse,
            lambda session: session.focus_on_object(
                input_data.object_name,
                input_data.distance_factor,
                input_data.camera_name,
            ),
        )

    @server.mcp.tool(
        name="get_blender_viewport_info",
        description="Get active viewport and render settings.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def get_blender_viewport_info() -> Dict[str, Any]:
        return await server._exec_backend(
            "get_blender_viewport_info",
            server.blender_adapter,
            "Blender",
            BlenderViewportInfoResponse,
            lambda session: session.get_viewport_info(),
        )

    @server.mcp.tool(
        name="capture_blender_viewport_sequence",
        description=(
            "Capture a sequence of viewport frames " "as base64-encoded JPEGs."
        ),
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def capture_blender_viewport_sequence(
        start_frame: int,
        end_frame: int,
        step: int = 1,
        width: int = 512,
        height: int = 512,
        jpeg_quality: int = 85,
    ) -> Dict[str, Any]:
        input_data = server._validate_input(
            BlenderCaptureSequenceRequest,
            start_frame=start_frame,
            end_frame=end_frame,
            step=step,
            width=width,
            height=height,
            jpeg_quality=jpeg_quality,
        )
        if isinstance(input_data, dict):
            return input_data
        return await server._exec_backend(
            "capture_blender_viewport_sequence",
            server.blender_adapter,
            "Blender",
            BlenderCaptureSequenceResponse,
            lambda session: session.capture_viewport_sequence(
                input_data.start_frame,
                input_data.end_frame,
                input_data.step,
                input_data.width,
                input_data.height,
                input_data.jpeg_quality,
            ),
        )

    # -- Phase 3: Scene Manipulation tools --------------------------------

    @server.mcp.tool(
        name="create_blender_object",
        description="Create a new object in the Blender scene.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def create_blender_object(
        object_type: str,
        name: Optional[str] = None,
        location: List[float] = [0.0, 0.0, 0.0],
        rotation_euler: List[float] = [0.0, 0.0, 0.0],
        scale: List[float] = [1.0, 1.0, 1.0],
    ) -> Dict[str, Any]:
        input_data = server._validate_input(
            BlenderCreateObjectRequest,
            object_type=object_type,
            name=name,
            location=location,
            rotation_euler=rotation_euler,
            scale=scale,
        )
        if isinstance(input_data, dict):
            return input_data
        return await server._exec_backend(
            "create_blender_object",
            server.blender_adapter,
            "Blender",
            BlenderCreateObjectResponse,
            lambda session: session.create_object(
                input_data.object_type,
                input_data.name,
                list(input_data.location),
                list(input_data.rotation_euler),
                list(input_data.scale),
            ),
        )

    @server.mcp.tool(
        name="delete_blender_object",
        description="Delete an object from the Blender scene.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
            destructive=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def delete_blender_object(
        object_name: str,
    ) -> Dict[str, Any]:
        input_data = server._validate_input(
            BlenderDeleteObjectRequest,
            object_name=object_name,
        )
        if isinstance(input_data, dict):
            return input_data
        return await server._exec_backend(
            "delete_blender_object",
            server.blender_adapter,
            "Blender",
            BlenderDeleteObjectResponse,
            lambda session: session.delete_object(input_data.object_name),
        )

    @server.mcp.tool(
        name="set_blender_object_transform",
        description="Set location, rotation, and/or scale on a Blender object.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def set_blender_object_transform(
        object_name: str,
        location: Optional[List[float]] = None,
        rotation_euler: Optional[List[float]] = None,
        scale: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        input_data = server._validate_input(
            BlenderSetTransformRequest,
            object_name=object_name,
            location=location,
            rotation_euler=rotation_euler,
            scale=scale,
        )
        if isinstance(input_data, dict):
            return input_data
        return await server._exec_backend(
            "set_blender_object_transform",
            server.blender_adapter,
            "Blender",
            BlenderSetTransformResponse,
            lambda session: session.set_object_transform(
                input_data.object_name,
                loc,
                rot,
                sc,
            ),
        )

    @server.mcp.tool(
        name="set_blender_object_parent",
        description="Parent one Blender object to another.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def set_blender_object_parent(
        child_name: str,
        parent_name: str,
    ) -> Dict[str, Any]:
        input_data = server._validate_input(
            BlenderSetParentRequest,
            child_name=child_name,
            parent_name=parent_name,
        )
        if isinstance(input_data, dict):
            return input_data
        return await server._exec_backend(
            "set_blender_object_parent",
            server.blender_adapter,
            "Blender",
            BlenderSetParentResponse,
            lambda session: session.set_object_parent(
                input_data.child_name,
                input_data.parent_name,
            ),
        )

    @server.mcp.tool(
        name="clear_blender_object_parent",
        description="Remove parent from a Blender object.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def clear_blender_object_parent(
        object_name: str,
        keep_transform: bool = True,
    ) -> Dict[str, Any]:
        input_data = server._validate_input(
            BlenderClearParentRequest,
            object_name=object_name,
            keep_transform=keep_transform,
        )
        if isinstance(input_data, dict):
            return input_data
        return await server._exec_backend(
            "clear_blender_object_parent",
            server.blender_adapter,
            "Blender",
            BlenderClearParentResponse,
            lambda session: session.clear_object_parent(
                input_data.object_name,
                input_data.keep_transform,
            ),
        )

    @server.mcp.tool(
        name="assign_blender_material",
        description="Assign a Principled BSDF material to a Blender object.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def assign_blender_material(
        object_name: str,
        material_name: Optional[str] = None,
        base_color: List[float] = [0.8, 0.8, 0.8, 1.0],
        metallic: float = 0.0,
        roughness: float = 0.5,
    ) -> Dict[str, Any]:
        input_data = server._validate_input(
            BlenderAssignMaterialRequest,
            object_name=object_name,
            material_name=material_name,
            base_color=base_color,
            metallic=metallic,
            roughness=roughness,
        )
        if isinstance(input_data, dict):
            return input_data
        return await server._exec_backend(
            "assign_blender_material",
            server.blender_adapter,
            "Blender",
            BlenderAssignMaterialResponse,
            lambda session: session.assign_material(
                input_data.object_name,
                input_data.material_name,
                list(input_data.base_color),
                input_data.metallic,
                input_data.roughness,
            ),
        )

    @server.mcp.tool(
        name="add_blender_modifier",
        description="Add a modifier to a Blender object.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def add_blender_modifier(
        object_name: str,
        modifier_type: str,
        modifier_name: Optional[str] = None,
        params: Dict[str, Any] = {},
    ) -> Dict[str, Any]:
        input_data = server._validate_input(
            BlenderAddModifierRequest,
            object_name=object_name,
            modifier_type=modifier_type,
            modifier_name=modifier_name,
            params=params,
        )
        if isinstance(input_data, dict):
            return input_data
        return await server._exec_backend(
            "add_blender_modifier",
            server.blender_adapter,
            "Blender",
            BlenderAddModifierResponse,
            lambda session: session.add_modifier(
                input_data.object_name,
                input_data.modifier_type,
                input_data.modifier_name,
                dict(input_data.params),
            ),
        )

    @server.mcp.tool(
        name="set_blender_light_params",
        description="Set light parameters on a Blender light object.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def set_blender_light_params(
        light_name: str,
        energy: Optional[float] = None,
        color: Optional[List[float]] = None,
        use_shadow: Optional[bool] = None,
        spot_size: Optional[float] = None,
        spot_blend: Optional[float] = None,
        shadow_soft_size: Optional[float] = None,
    ) -> Dict[str, Any]:
        input_data = server._validate_input(
            BlenderSetLightParamsRequest,
            light_name=light_name,
            energy=energy,
            color=color,
            use_shadow=use_shadow,
            spot_size=spot_size,
            spot_blend=spot_blend,
            shadow_soft_size=shadow_soft_size,
        )
        if isinstance(input_data, dict):
            return input_data
        return await server._exec_backend(
            "set_blender_light_params",
            server.blender_adapter,
            "Blender",
            BlenderSetLightParamsResponse,
            lambda session: session.set_light_params(
                input_data.light_name,
                input_data.energy,
                col,
                input_data.use_shadow,
                input_data.spot_size,
                input_data.spot_blend,
                input_data.shadow_soft_size,
            ),
        )

    # -- Phase 4: File I/O tools ------------------------------------------

    @server.mcp.tool(
        name="open_blender_file",
        description="Open a .blend file, replacing the current scene.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
            destructive=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def open_blender_file(
        file_path: str,
    ) -> Dict[str, Any]:
        input_data = server._validate_input(
            BlenderOpenFileRequest,
            file_path=file_path,
        )
        if isinstance(input_data, dict):
            return input_data
        return await server._exec_backend(
            "open_blender_file",
            server.blender_adapter,
            "Blender",
            BlenderOpenFileResponse,
            lambda session: session.open_blend_file(input_data.file_path),
        )

    @server.mcp.tool(
        name="save_blender_file",
        description="Save the current .blend file.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def save_blender_file(
        file_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        input_data = server._validate_input(
            BlenderSaveFileRequest,
            file_path=file_path,
        )
        if isinstance(input_data, dict):
            return input_data
        return await server._exec_backend(
            "save_blender_file",
            server.blender_adapter,
            "Blender",
            BlenderSaveFileResponse,
            lambda session: session.save_blend_file(input_data.file_path),
        )

    @server.mcp.tool(
        name="import_blender_file",
        description="Import a file into the Blender scene.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def import_blender_file(
        file_path: str,
        file_format: str,
    ) -> Dict[str, Any]:
        input_data = server._validate_input(
            BlenderImportFileRequest,
            file_path=file_path,
            file_format=file_format,
        )
        if isinstance(input_data, dict):
            return input_data
        return await server._exec_backend(
            "import_blender_file",
            server.blender_adapter,
            "Blender",
            BlenderImportFileResponse,
            lambda session: session.import_file(
                input_data.file_path,
                input_data.file_format,
            ),
        )

    # -- File I/O tools (export + info) ----------------------------------

    @server.mcp.tool(
        name="export_blender_file",
        description="Export scene objects to a file.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def export_blender_file(
        file_path: str,
        file_format: str,
        selected_only: bool = False,
    ) -> Dict[str, Any]:
        input_data = server._validate_input(
            BlenderExportFileRequest,
            file_path=file_path,
            file_format=file_format,
            selected_only=selected_only,
        )
        if isinstance(input_data, dict):
            return input_data
        return await server._exec_backend(
            "export_blender_file",
            server.blender_adapter,
            "Blender",
            BlenderExportFileResponse,
            lambda session: session.export_file(
                file_path=input_data.file_path,
                file_format=input_data.file_format,
                selected_only=input_data.selected_only,
            ),
        )

    @server.mcp.tool(
        name="get_blender_file_info",
        description="Get information about the current .blend file.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def get_blender_file_info() -> Dict[str, Any]:
        return await server._exec_backend(
            "get_blender_file_info",
            server.blender_adapter,
            "Blender",
            BlenderFileInfoResponse,
            lambda session: session.get_file_info(),
        )

    # -- Animation & Timeline tools ---------------------------------------

    @server.mcp.tool(
        name="set_blender_frame",
        description="Set the current animation frame.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def set_blender_frame(frame: int) -> Dict[str, Any]:
        input_data = server._validate_input(
            BlenderSetFrameRequest,
            frame=frame,
        )
        if isinstance(input_data, dict):
            return input_data
        return await server._exec_backend(
            "set_blender_frame",
            server.blender_adapter,
            "Blender",
            BlenderSetFrameResponse,
            lambda session: session.set_frame(input_data.frame),
        )

    @server.mcp.tool(
        name="get_blender_frame",
        description="Get the current frame and animation range.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def get_blender_frame() -> Dict[str, Any]:
        return await server._exec_backend(
            "get_blender_frame",
            server.blender_adapter,
            "Blender",
            BlenderGetFrameResponse,
            lambda session: session.get_frame(),
        )

    @server.mcp.tool(
        name="set_blender_frame_range",
        description="Set the animation frame range.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def set_blender_frame_range(
        frame_start: int,
        frame_end: int,
    ) -> Dict[str, Any]:
        input_data = server._validate_input(
            BlenderSetFrameRangeRequest,
            frame_start=frame_start,
            frame_end=frame_end,
        )
        if isinstance(input_data, dict):
            return input_data
        return await server._exec_backend(
            "set_blender_frame_range",
            server.blender_adapter,
            "Blender",
            BlenderSetFrameRangeResponse,
            lambda session: session.set_frame_range(
                input_data.frame_start,
                input_data.frame_end,
            ),
        )

    @server.mcp.tool(
        name="play_blender_animation",
        description="Control animation playback (play, stop, reverse).",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def play_blender_animation(
        action: str = "play",
    ) -> Dict[str, Any]:
        input_data = server._validate_input(
            BlenderPlayAnimationRequest,
            action=action,
        )
        if isinstance(input_data, dict):
            return input_data
        return await server._exec_backend(
            "play_blender_animation",
            server.blender_adapter,
            "Blender",
            BlenderPlayAnimationResponse,
            lambda session: session.play_animation(input_data.action),
        )

    @server.mcp.tool(
        name="insert_blender_keyframe",
        description="Insert a keyframe on an object property.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def insert_blender_keyframe(
        object_name: str,
        data_path: str,
        frame: int,
        index: int = -1,
    ) -> Dict[str, Any]:
        input_data = server._validate_input(
            BlenderInsertKeyframeRequest,
            object_name=object_name,
            data_path=data_path,
            frame=frame,
            index=index,
        )
        if isinstance(input_data, dict):
            return input_data
        return await server._exec_backend(
            "insert_blender_keyframe",
            server.blender_adapter,
            "Blender",
            BlenderInsertKeyframeResponse,
            lambda session: session.insert_keyframe(
                input_data.object_name,
                input_data.data_path,
                input_data.frame,
                input_data.index,
            ),
        )

    @server.mcp.tool(
        name="delete_blender_keyframe",
        description="Delete a keyframe from an object property.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def delete_blender_keyframe(
        object_name: str,
        data_path: str,
        frame: int,
        index: int = -1,
    ) -> Dict[str, Any]:
        input_data = server._validate_input(
            BlenderDeleteKeyframeRequest,
            object_name=object_name,
            data_path=data_path,
            frame=frame,
            index=index,
        )
        if isinstance(input_data, dict):
            return input_data
        return await server._exec_backend(
            "delete_blender_keyframe",
            server.blender_adapter,
            "Blender",
            BlenderDeleteKeyframeResponse,
            lambda session: session.delete_keyframe(
                input_data.object_name,
                input_data.data_path,
                input_data.frame,
                input_data.index,
            ),
        )

    @server.mcp.tool(
        name="get_blender_keyframes",
        description="Get keyframe summary for an object.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def get_blender_keyframes(
        object_name: str,
    ) -> Dict[str, Any]:
        input_data = server._validate_input(
            BlenderGetKeyframesRequest,
            object_name=object_name,
        )
        if isinstance(input_data, dict):
            return input_data
        return await server._exec_backend(
            "get_blender_keyframes",
            server.blender_adapter,
            "Blender",
            BlenderGetKeyframesResponse,
            lambda session: session.get_keyframes(
                input_data.object_name,
            ),
        )

    # -- Physics & Simulation tools ---------------------------------------

    @server.mcp.tool(
        name="setup_blender_rigid_body",
        description="Set up rigid body physics on a Blender object.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def setup_blender_rigid_body(
        object_name: str,
        body_type: str = "ACTIVE",
        mass: float = 1.0,
        friction: float = 0.5,
        restitution: float = 0.0,
        collision_shape: str = "CONVEX_HULL",
        linear_damping: float = 0.04,
        angular_damping: float = 0.1,
    ) -> Dict[str, Any]:
        input_data = server._validate_input(
            BlenderSetupRigidBodyRequest,
            object_name=object_name,
            body_type=body_type,
            mass=mass,
            friction=friction,
            restitution=restitution,
            collision_shape=collision_shape,
            linear_damping=linear_damping,
            angular_damping=angular_damping,
        )
        if isinstance(input_data, dict):
            return input_data
        return await server._exec_backend(
            "setup_blender_rigid_body",
            server.blender_adapter,
            "Blender",
            BlenderSetupRigidBodyResponse,
            lambda session: session.setup_rigid_body(
                input_data.object_name,
                input_data.body_type,
                input_data.mass,
                input_data.friction,
                input_data.restitution,
                input_data.collision_shape,
                input_data.linear_damping,
                input_data.angular_damping,
            ),
        )

    @server.mcp.tool(
        name="add_blender_force_field",
        description="Add a force field to the scene.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def add_blender_force_field(
        field_type: str,
        strength: float = 1.0,
        location: Optional[List[float]] = None,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        input_data = server._validate_input(
            BlenderAddForceFieldRequest,
            field_type=field_type,
            strength=strength,
            location=location,
            name=name,
        )
        if isinstance(input_data, dict):
            return input_data
        return await server._exec_backend(
            "add_blender_force_field",
            server.blender_adapter,
            "Blender",
            BlenderAddForceFieldResponse,
            lambda session: session.add_force_field(
                input_data.field_type,
                input_data.strength,
                input_data.location,
                input_data.name,
            ),
        )

    @server.mcp.tool(
        name="get_blender_force_field_info",
        description="Get force field parameters for an object.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def get_blender_force_field_info(
        object_name: str,
    ) -> Dict[str, Any]:
        input_data = server._validate_input(
            BlenderGetForceFieldInfoRequest,
            object_name=object_name,
        )
        if isinstance(input_data, dict):
            return input_data
        return await server._exec_backend(
            "get_blender_force_field_info",
            server.blender_adapter,
            "Blender",
            BlenderGetForceFieldInfoResponse,
            lambda session: session.get_force_field_info(
                input_data.object_name,
            ),
        )

    @server.mcp.tool(
        name="add_blender_rigid_body_constraint",
        description="Add a rigid body constraint between two objects.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def add_blender_rigid_body_constraint(
        constraint_type: str,
        object1_name: str,
        object2_name: str,
        location: Optional[List[float]] = None,
        disable_collisions: bool = True,
    ) -> Dict[str, Any]:
        input_data = server._validate_input(
            BlenderAddConstraintRequest,
            constraint_type=constraint_type,
            object1_name=object1_name,
            object2_name=object2_name,
            location=location,
            disable_collisions=disable_collisions,
        )
        if isinstance(input_data, dict):
            return input_data
        return await server._exec_backend(
            "add_blender_rigid_body_constraint",
            server.blender_adapter,
            "Blender",
            BlenderAddConstraintResponse,
            lambda session: session.add_rigid_body_constraint(
                input_data.constraint_type,
                input_data.object1_name,
                input_data.object2_name,
                input_data.location,
                input_data.disable_collisions,
            ),
        )

    @server.mcp.tool(
        name="get_blender_constraint_info",
        description="Get rigid body constraint info for an object.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def get_blender_constraint_info(
        object_name: str,
    ) -> Dict[str, Any]:
        input_data = server._validate_input(
            BlenderGetConstraintInfoRequest,
            object_name=object_name,
        )
        if isinstance(input_data, dict):
            return input_data
        return await server._exec_backend(
            "get_blender_constraint_info",
            server.blender_adapter,
            "Blender",
            BlenderGetConstraintInfoResponse,
            lambda session: session.get_constraint_info(
                input_data.object_name,
            ),
        )

    @server.mcp.tool(
        name="get_blender_physics_state",
        description="Get current physics state of an object.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def get_blender_physics_state(
        object_name: str,
    ) -> Dict[str, Any]:
        input_data = server._validate_input(
            BlenderGetPhysicsStateRequest,
            object_name=object_name,
        )
        if isinstance(input_data, dict):
            return input_data
        return await server._exec_backend(
            "get_blender_physics_state",
            server.blender_adapter,
            "Blender",
            BlenderGetPhysicsStateResponse,
            lambda session: session.get_physics_state(
                input_data.object_name,
            ),
        )

    @server.mcp.tool(
        name="get_blender_object_trajectory",
        description="Sample object position over a frame range.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def get_blender_object_trajectory(
        object_name: str,
        start_frame: int,
        end_frame: int,
        step: int = 1,
    ) -> Dict[str, Any]:
        input_data = server._validate_input(
            BlenderGetTrajectoryRequest,
            object_name=object_name,
            start_frame=start_frame,
            end_frame=end_frame,
            step=step,
        )
        if isinstance(input_data, dict):
            return input_data
        return await server._exec_backend(
            "get_blender_object_trajectory",
            server.blender_adapter,
            "Blender",
            BlenderGetTrajectoryResponse,
            lambda session: session.get_object_trajectory(
                input_data.object_name,
                input_data.start_frame,
                input_data.end_frame,
                input_data.step,
            ),
        )

    @server.mcp.tool(
        name="bake_blender_simulation",
        description="Bake physics simulation for a frame range.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def bake_blender_simulation(
        frame_start: int,
        frame_end: int,
    ) -> Dict[str, Any]:
        input_data = server._validate_input(
            BlenderBakeSimulationRequest,
            frame_start=frame_start,
            frame_end=frame_end,
        )
        if isinstance(input_data, dict):
            return input_data
        return await server._exec_backend(
            "bake_blender_simulation",
            server.blender_adapter,
            "Blender",
            BlenderBakeSimulationResponse,
            lambda session: session.bake_simulation(
                input_data.frame_start,
                input_data.frame_end,
            ),
        )

    @server.mcp.tool(
        name="free_blender_bake",
        description="Free (delete) baked physics simulation data.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def free_blender_bake() -> ToolResult:
        def _free_bake(session: Any) -> Dict[str, Any]:
            # session.free_bake() returns None and raises on failure, so
            # reaching the return means the bake was actually freed.
            session.free_bake()
            return {}

        return await server._exec_backend(
            "free_blender_bake",
            server.blender_adapter,
            "Blender",
            BlenderFreeBakeResponse,
            _free_bake,
        )

    # -- Scripting & mesh-from-data tools --------------------------------

    @server.mcp.tool(
        name="execute_blender_script",
        description=(
            "Execute arbitrary Python code inside Blender with access "
            "to bpy. Assign to __result__ to return a value."
        ),
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def execute_blender_script(
        script: str,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        input_data = server._validate_input(
            BlenderExecuteScriptRequest,
            script=script,
            timeout=timeout,
        )
        if isinstance(input_data, dict):
            return input_data
        return await server._exec_backend(
            "execute_blender_script",
            server.blender_adapter,
            "Blender",
            BlenderExecuteScriptResponse,
            lambda session: session.execute_script(
                input_data.script,
                input_data.timeout,
            ),
        )

    @server.mcp.tool(
        name="create_blender_mesh_from_data",
        description=(
            "Create a mesh object from raw vertex, edge, and face "
            "data. Use for procedural geometry."
        ),
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def create_blender_mesh_from_data(
        name: str,
        vertices: List[List[float]],
        edges: List[List[int]] = [],
        faces: List[List[int]] = [],
        location: Optional[List[float]] = None,
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        input_data = server._validate_input(
            BlenderCreateMeshFromDataRequest,
            name=name,
            vertices=vertices,
            edges=edges,
            faces=faces,
            location=location,
            collection_name=collection_name,
        )
        if isinstance(input_data, dict):
            return input_data
        return await server._exec_backend(
            "create_blender_mesh_from_data",
            server.blender_adapter,
            "Blender",
            BlenderCreateMeshFromDataResponse,
            lambda session: session.create_mesh_from_data(
                name=input_data.name,
                vertices=[list(v) for v in input_data.vertices],
                edges=[list(e) for e in input_data.edges],
                faces=[list(f) for f in input_data.faces],
                location=(list(input_data.location) if input_data.location else None),
                collection_name=input_data.collection_name,
            ),
        )

    # -- SimReady Asset Format tools ---------------------------------------

    @server.mcp.tool(
        name="apply_simready_metadata",
        description=(
            "Apply NVIDIA SimReady metadata (semantic labels, physics "
            "properties, material info) to a Blender object as custom "
            "properties. Stored with simready_ prefix for USD export."
        ),
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def apply_simready_metadata(
        object_name: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        input_data = server._validate_input(
            SimReadyApplyMetadataRequest,
            object_name=object_name,
            metadata=metadata,
        )
        if isinstance(input_data, dict):
            return input_data
        return await server._exec_backend(
            "apply_simready_metadata",
            server.blender_adapter,
            "Blender",
            SimReadyApplyMetadataResponse,
            lambda session: session.apply_simready_metadata(
                object_name=input_data.object_name,
                metadata=input_data.metadata.model_dump(exclude_none=True),
            ),
        )

    @server.mcp.tool(
        name="get_simready_metadata",
        description=(
            "Read NVIDIA SimReady metadata from a Blender object. "
            "Returns semantic labels, physics properties, and material "
            "info stored as simready_ custom properties."
        ),
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def get_simready_metadata(
        object_name: str,
    ) -> Dict[str, Any]:
        input_data = server._validate_input(
            SimReadyGetMetadataRequest,
            object_name=object_name,
        )
        if isinstance(input_data, dict):
            return input_data
        return await server._exec_backend(
            "get_simready_metadata",
            server.blender_adapter,
            "Blender",
            SimReadyGetMetadataResponse,
            lambda session: session.get_simready_metadata(
                object_name=input_data.object_name,
            ),
        )

    @server.mcp.tool(
        name="validate_simready_compliance",
        description=(
            "Validate Blender objects against NVIDIA SimReady "
            "conventions: naming (lowercase_underscore), scale (meters), "
            "clean transforms, material segmentation, hierarchy."
        ),
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def validate_simready_compliance(
        object_names: Optional[List[str]] = None,
        check_naming: bool = True,
        check_scale: bool = True,
        check_transforms: bool = True,
        check_materials: bool = True,
        check_hierarchy: bool = True,
    ) -> Dict[str, Any]:
        input_data = server._validate_input(
            SimReadyValidateRequest,
            object_names=object_names,
            check_naming=check_naming,
            check_scale=check_scale,
            check_transforms=check_transforms,
            check_materials=check_materials,
            check_hierarchy=check_hierarchy,
        )
        if isinstance(input_data, dict):
            return input_data
        return await server._exec_backend(
            "validate_simready_compliance",
            server.blender_adapter,
            "Blender",
            SimReadyValidateResponse,
            lambda session: session.validate_simready_compliance(
                object_names=input_data.object_names,
                check_naming=input_data.check_naming,
                check_scale=input_data.check_scale,
                check_transforms=input_data.check_transforms,
                check_materials=input_data.check_materials,
                check_hierarchy=input_data.check_hierarchy,
            ),
        )

    @server.mcp.tool(
        name="export_simready_usd",
        description=(
            "Export a SimReady-compliant USD file from Blender. Validates "
            "objects before export, selects the requested objects, and "
            "carries simready_ custom properties into USD attributes."
        ),
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def export_simready_usd(
        file_path: str,
        object_names: Optional[List[str]] = None,
        embed_metadata: bool = True,
        validate_before_export: bool = True,
    ) -> Dict[str, Any]:
        input_data = server._validate_input(
            SimReadyExportRequest,
            file_path=file_path,
            object_names=object_names,
            embed_metadata=embed_metadata,
            validate_before_export=validate_before_export,
        )
        if isinstance(input_data, dict):
            return input_data
        return await server._exec_backend(
            "export_simready_usd",
            server.blender_adapter,
            "Blender",
            SimReadyExportResponse,
            lambda session: session.export_simready_usd(
                file_path=input_data.file_path,
                object_names=input_data.object_names,
                embed_metadata=input_data.embed_metadata,
                validate_before_export=input_data.validate_before_export,
            ),
        )

    @server.mcp.tool(
        name="setup_simready_hierarchy",
        description=(
            "Create a SimReady-compliant object hierarchy in Blender with "
            "a root empty (XForm equivalent) and parent the given children "
            "under it. Optionally applies semantic labels to the root."
        ),
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def setup_simready_hierarchy(
        root_name: str,
        child_names: List[str],
        semantic: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        input_data = server._validate_input(
            SimReadySetupHierarchyRequest,
            root_name=root_name,
            child_names=child_names,
            semantic=semantic,
        )
        if isinstance(input_data, dict):
            return input_data
        return await server._exec_backend(
            "setup_simready_hierarchy",
            server.blender_adapter,
            "Blender",
            SimReadySetupHierarchyResponse,
            lambda session: session.setup_simready_hierarchy(
                root_name=input_data.root_name,
                child_names=input_data.child_names,
                semantic=(
                    input_data.semantic.model_dump(exclude_none=True)
                    if input_data.semantic
                    else None
                ),
            ),
        )
