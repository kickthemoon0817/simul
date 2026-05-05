"""
Blender runtime tool registration for Simul MCP Server.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..schemas.blender import *
from ..schemas.common import ErrorResponse
from ..schemas.simready import *
from ._helpers import apply_success_from_error

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
        output_schema=server._tool_output_schema(BlenderInfoResponse, ErrorResponse),
        task=server._task_optional(),
    )
    async def get_blender_info() -> Dict[str, Any]:
        """
        Get information about the active Blender runtime.

        Returns:
            Blender runtime information or an error response.
        """
        rate_error = server._check_rate_limit("get_blender_info")
        if rate_error:
            return rate_error

        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()

            with server.blender_adapter.create_session() as session:
                runtime_info = session.get_runtime_info()
                apply_success_from_error(runtime_info)
                result = BlenderInfoResponse(**runtime_info).model_dump()
                return server._validate_output(
                    result,
                    (BlenderInfoResponse, ErrorResponse),
                    "get_blender_info",
                )

        except Exception as e:
            server.logger.error(f"Error getting Blender runtime info: {e}")
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (BlenderInfoResponse, ErrorResponse),
                "get_blender_info",
            )

    @server.mcp.tool(
        name="list_blender_scene_objects",
        description="List objects from the active Blender scene.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            BlenderSceneObjectsResponse,
            ErrorResponse,
        ),
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
        rate_error = server._check_rate_limit("list_blender_scene_objects")
        if rate_error:
            return rate_error

        input_data = server._validate_input(
            BlenderSceneObjectsRequest,
            collection_name=collection_name,
            include_hidden=include_hidden,
            max_items=max_items,
        )
        if isinstance(input_data, dict):
            return input_data

        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()

            with server.blender_adapter.create_session() as session:
                objects_payload = session.list_scene_objects(
                    collection_name=input_data.collection_name,
                    include_hidden=input_data.include_hidden,
                    max_items=input_data.max_items,
                )
                apply_success_from_error(objects_payload)
                result = BlenderSceneObjectsResponse(**objects_payload).model_dump()
                return server._validate_output(
                    result,
                    (BlenderSceneObjectsResponse, ErrorResponse),
                    "list_blender_scene_objects",
                )

        except Exception as e:
            server.logger.error(f"Error listing Blender scene objects: {e}")
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (BlenderSceneObjectsResponse, ErrorResponse),
                "list_blender_scene_objects",
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
        output_schema=server._tool_output_schema(
            BlenderObjectInfoResponse,
            ErrorResponse,
        ),
        task=server._task_optional(),
    )
    async def get_blender_object_info(
        object_name: str,
    ) -> Dict[str, Any]:
        rate_error = server._check_rate_limit("get_blender_object_info")
        if rate_error:
            return rate_error
        input_data = server._validate_input(
            BlenderObjectInfoRequest,
            object_name=object_name,
        )
        if isinstance(input_data, dict):
            return input_data
        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.blender_adapter.create_session() as session:
                payload = session.get_object_info(input_data.object_name)
                apply_success_from_error(payload)
                result = BlenderObjectInfoResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (BlenderObjectInfoResponse, ErrorResponse),
                    "get_blender_object_info",
                )
        except Exception as e:
            server.logger.error(f"Error getting Blender object info: {e}")
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (BlenderObjectInfoResponse, ErrorResponse),
                "get_blender_object_info",
            )

    @server.mcp.tool(
        name="get_blender_mesh_info",
        description="Get mesh geometry counts for a Blender mesh object.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            BlenderMeshInfoResponse,
            ErrorResponse,
        ),
        task=server._task_optional(),
    )
    async def get_blender_mesh_info(
        object_name: str,
    ) -> Dict[str, Any]:
        rate_error = server._check_rate_limit("get_blender_mesh_info")
        if rate_error:
            return rate_error
        input_data = server._validate_input(
            BlenderMeshInfoRequest,
            object_name=object_name,
        )
        if isinstance(input_data, dict):
            return input_data
        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.blender_adapter.create_session() as session:
                payload = session.get_mesh_info(input_data.object_name)
                apply_success_from_error(payload)
                result = BlenderMeshInfoResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (BlenderMeshInfoResponse, ErrorResponse),
                    "get_blender_mesh_info",
                )
        except Exception as e:
            server.logger.error(f"Error getting Blender mesh info: {e}")
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (BlenderMeshInfoResponse, ErrorResponse),
                "get_blender_mesh_info",
            )

    @server.mcp.tool(
        name="get_blender_bounding_box",
        description="Get the bounding box of a Blender object.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            BlenderBoundingBoxResponse,
            ErrorResponse,
        ),
        task=server._task_optional(),
    )
    async def get_blender_bounding_box(
        object_name: str,
        world_space: bool = True,
    ) -> Dict[str, Any]:
        rate_error = server._check_rate_limit("get_blender_bounding_box")
        if rate_error:
            return rate_error
        input_data = server._validate_input(
            BlenderBoundingBoxRequest,
            object_name=object_name,
            world_space=world_space,
        )
        if isinstance(input_data, dict):
            return input_data
        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.blender_adapter.create_session() as session:
                payload = session.get_bounding_box(
                    input_data.object_name,
                    input_data.world_space,
                )
                apply_success_from_error(payload)
                result = BlenderBoundingBoxResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (BlenderBoundingBoxResponse, ErrorResponse),
                    "get_blender_bounding_box",
                )
        except Exception as e:
            server.logger.error(f"Error getting Blender bounding box: {e}")
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (BlenderBoundingBoxResponse, ErrorResponse),
                "get_blender_bounding_box",
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
        output_schema=server._tool_output_schema(
            BlenderSearchObjectsResponse,
            ErrorResponse,
        ),
        task=server._task_optional(),
    )
    async def search_blender_objects(
        name_pattern: Optional[str] = None,
        object_type: Optional[str] = None,
        max_results: int = 50,
    ) -> Dict[str, Any]:
        rate_error = server._check_rate_limit("search_blender_objects")
        if rate_error:
            return rate_error
        input_data = server._validate_input(
            BlenderSearchObjectsRequest,
            name_pattern=name_pattern,
            object_type=object_type,
            max_results=max_results,
        )
        if isinstance(input_data, dict):
            return input_data
        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.blender_adapter.create_session() as session:
                payload = session.search_objects(
                    input_data.name_pattern,
                    input_data.object_type,
                    input_data.max_results,
                )
                apply_success_from_error(payload)
                result = BlenderSearchObjectsResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (BlenderSearchObjectsResponse, ErrorResponse),
                    "search_blender_objects",
                )
        except Exception as e:
            server.logger.error(f"Error searching Blender objects: {e}")
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (BlenderSearchObjectsResponse, ErrorResponse),
                "search_blender_objects",
            )

    @server.mcp.tool(
        name="summarize_blender_scene",
        description="Get a high-level summary of the Blender scene.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            BlenderSceneSummaryResponse,
            ErrorResponse,
        ),
        task=server._task_optional(),
    )
    async def summarize_blender_scene() -> Dict[str, Any]:
        rate_error = server._check_rate_limit("summarize_blender_scene")
        if rate_error:
            return rate_error
        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.blender_adapter.create_session() as session:
                payload = session.summarize_scene()
                apply_success_from_error(payload)
                result = BlenderSceneSummaryResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (BlenderSceneSummaryResponse, ErrorResponse),
                    "summarize_blender_scene",
                )
        except Exception as e:
            server.logger.error(f"Error summarizing Blender scene: {e}")
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (BlenderSceneSummaryResponse, ErrorResponse),
                "summarize_blender_scene",
            )

    @server.mcp.tool(
        name="get_blender_material_info",
        description="Get material information with bounded node tree traversal.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            BlenderMaterialInfoResponse,
            ErrorResponse,
        ),
        task=server._task_optional(),
    )
    async def get_blender_material_info(
        material_name: str,
    ) -> Dict[str, Any]:
        rate_error = server._check_rate_limit("get_blender_material_info")
        if rate_error:
            return rate_error
        input_data = server._validate_input(
            BlenderMaterialInfoRequest,
            material_name=material_name,
        )
        if isinstance(input_data, dict):
            return input_data
        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.blender_adapter.create_session() as session:
                payload = session.get_material_info(
                    input_data.material_name,
                )
                apply_success_from_error(payload)
                result = BlenderMaterialInfoResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (BlenderMaterialInfoResponse, ErrorResponse),
                    "get_blender_material_info",
                )
        except Exception as e:
            server.logger.error(f"Error getting Blender material info: {e}")
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (BlenderMaterialInfoResponse, ErrorResponse),
                "get_blender_material_info",
            )

    @server.mcp.tool(
        name="get_blender_distance_between",
        description="Measure the Euclidean distance between two Blender objects.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            BlenderDistanceResponse,
            ErrorResponse,
        ),
        task=server._task_optional(),
    )
    async def get_blender_distance_between(
        object_name_a: str,
        object_name_b: str,
    ) -> Dict[str, Any]:
        rate_error = server._check_rate_limit("get_blender_distance_between")
        if rate_error:
            return rate_error
        input_data = server._validate_input(
            BlenderDistanceRequest,
            object_name_a=object_name_a,
            object_name_b=object_name_b,
        )
        if isinstance(input_data, dict):
            return input_data
        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.blender_adapter.create_session() as session:
                payload = session.get_distance_between(
                    input_data.object_name_a,
                    input_data.object_name_b,
                )
                apply_success_from_error(payload)
                result = BlenderDistanceResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (BlenderDistanceResponse, ErrorResponse),
                    "get_blender_distance_between",
                )
        except Exception as e:
            server.logger.error(f"Error measuring Blender distance: {e}")
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (BlenderDistanceResponse, ErrorResponse),
                "get_blender_distance_between",
            )

    @server.mcp.tool(
        name="check_blender_object_bounds",
        description="Check if a Blender object is within spatial bounds.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            BlenderBoundsCheckResponse,
            ErrorResponse,
        ),
        task=server._task_optional(),
    )
    async def check_blender_object_bounds(
        object_name: str,
        bounds_min: List[float],
        bounds_max: List[float],
    ) -> Dict[str, Any]:
        rate_error = server._check_rate_limit("check_blender_object_bounds")
        if rate_error:
            return rate_error
        input_data = server._validate_input(
            BlenderBoundsCheckRequest,
            object_name=object_name,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
        )
        if isinstance(input_data, dict):
            return input_data
        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.blender_adapter.create_session() as session:
                payload = session.check_object_bounds(
                    input_data.object_name,
                    input_data.bounds_min,
                    input_data.bounds_max,
                )
                apply_success_from_error(payload)
                result = BlenderBoundsCheckResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (BlenderBoundsCheckResponse, ErrorResponse),
                    "check_blender_object_bounds",
                )
        except Exception as e:
            server.logger.error(f"Error checking Blender bounds: {e}")
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (BlenderBoundsCheckResponse, ErrorResponse),
                "check_blender_object_bounds",
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
        output_schema=server._tool_output_schema(
            BlenderCaptureViewportResponse,
            ErrorResponse,
        ),
        task=server._task_optional(),
    )
    async def capture_blender_viewport(
        width: int = 512,
        height: int = 512,
        jpeg_quality: int = 85,
        use_render_fallback: bool = False,
    ) -> Dict[str, Any]:
        rate_error = server._check_rate_limit("capture_blender_viewport")
        if rate_error:
            return rate_error
        input_data = server._validate_input(
            BlenderCaptureViewportRequest,
            width=width,
            height=height,
            jpeg_quality=jpeg_quality,
            use_render_fallback=use_render_fallback,
        )
        if isinstance(input_data, dict):
            return input_data
        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.blender_adapter.create_session() as session:
                payload = session.capture_viewport(
                    input_data.width,
                    input_data.height,
                    input_data.jpeg_quality,
                    input_data.use_render_fallback,
                )
                apply_success_from_error(payload)
                result = BlenderCaptureViewportResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (BlenderCaptureViewportResponse, ErrorResponse),
                    "capture_blender_viewport",
                )
        except Exception as e:
            server.logger.error(f"Error capturing Blender viewport: {e}")
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (BlenderCaptureViewportResponse, ErrorResponse),
                "capture_blender_viewport",
            )

    @server.mcp.tool(
        name="set_blender_camera_view",
        description="Set the active camera's location and rotation.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            BlenderSetCameraViewResponse,
            ErrorResponse,
        ),
        task=server._task_optional(),
    )
    async def set_blender_camera_view(
        location: List[float],
        rotation_euler: List[float],
        camera_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        rate_error = server._check_rate_limit("set_blender_camera_view")
        if rate_error:
            return rate_error
        input_data = server._validate_input(
            BlenderSetCameraViewRequest,
            location=location,
            rotation_euler=rotation_euler,
            camera_name=camera_name,
        )
        if isinstance(input_data, dict):
            return input_data
        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.blender_adapter.create_session() as session:
                payload = session.set_camera_view(
                    list(input_data.location),
                    list(input_data.rotation_euler),
                    input_data.camera_name,
                )
                apply_success_from_error(payload)
                result = BlenderSetCameraViewResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (BlenderSetCameraViewResponse, ErrorResponse),
                    "set_blender_camera_view",
                )
        except Exception as e:
            server.logger.error(f"Error setting Blender camera view: {e}")
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (BlenderSetCameraViewResponse, ErrorResponse),
                "set_blender_camera_view",
            )

    @server.mcp.tool(
        name="get_blender_camera_info",
        description="Get information about the active Blender camera.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            BlenderCameraInfoResponse,
            ErrorResponse,
        ),
        task=server._task_optional(),
    )
    async def get_blender_camera_info(
        camera_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        rate_error = server._check_rate_limit("get_blender_camera_info")
        if rate_error:
            return rate_error
        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.blender_adapter.create_session() as session:
                payload = session.get_camera_info(camera_name)
                apply_success_from_error(payload)
                result = BlenderCameraInfoResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (BlenderCameraInfoResponse, ErrorResponse),
                    "get_blender_camera_info",
                )
        except Exception as e:
            server.logger.error(f"Error getting Blender camera info: {e}")
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (BlenderCameraInfoResponse, ErrorResponse),
                "get_blender_camera_info",
            )

    @server.mcp.tool(
        name="focus_blender_on_object",
        description="Focus the camera on a specific Blender object.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            BlenderFocusOnObjectResponse,
            ErrorResponse,
        ),
        task=server._task_optional(),
    )
    async def focus_blender_on_object(
        object_name: str,
        distance_factor: float = 2.0,
        camera_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        rate_error = server._check_rate_limit("focus_blender_on_object")
        if rate_error:
            return rate_error
        input_data = server._validate_input(
            BlenderFocusOnObjectRequest,
            object_name=object_name,
            distance_factor=distance_factor,
            camera_name=camera_name,
        )
        if isinstance(input_data, dict):
            return input_data
        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.blender_adapter.create_session() as session:
                payload = session.focus_on_object(
                    input_data.object_name,
                    input_data.distance_factor,
                    input_data.camera_name,
                )
                apply_success_from_error(payload)
                result = BlenderFocusOnObjectResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (BlenderFocusOnObjectResponse, ErrorResponse),
                    "focus_blender_on_object",
                )
        except Exception as e:
            server.logger.error(f"Error focusing Blender camera: {e}")
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (BlenderFocusOnObjectResponse, ErrorResponse),
                "focus_blender_on_object",
            )

    @server.mcp.tool(
        name="get_blender_viewport_info",
        description="Get active viewport and render settings.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            BlenderViewportInfoResponse,
            ErrorResponse,
        ),
        task=server._task_optional(),
    )
    async def get_blender_viewport_info() -> Dict[str, Any]:
        rate_error = server._check_rate_limit("get_blender_viewport_info")
        if rate_error:
            return rate_error
        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.blender_adapter.create_session() as session:
                payload = session.get_viewport_info()
                apply_success_from_error(payload)
                result = BlenderViewportInfoResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (BlenderViewportInfoResponse, ErrorResponse),
                    "get_blender_viewport_info",
                )
        except Exception as e:
            server.logger.error(f"Error getting Blender viewport info: {e}")
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (BlenderViewportInfoResponse, ErrorResponse),
                "get_blender_viewport_info",
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
        output_schema=server._tool_output_schema(
            BlenderCaptureSequenceResponse,
            ErrorResponse,
        ),
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
        rate_error = server._check_rate_limit("capture_blender_viewport_sequence")
        if rate_error:
            return rate_error
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
        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.blender_adapter.create_session() as session:
                payload = session.capture_viewport_sequence(
                    input_data.start_frame,
                    input_data.end_frame,
                    input_data.step,
                    input_data.width,
                    input_data.height,
                    input_data.jpeg_quality,
                )
                apply_success_from_error(payload)
                result = BlenderCaptureSequenceResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (BlenderCaptureSequenceResponse, ErrorResponse),
                    "capture_blender_viewport_sequence",
                )
        except Exception as e:
            server.logger.error(f"Error capturing Blender viewport sequence: {e}")
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (BlenderCaptureSequenceResponse, ErrorResponse),
                "capture_blender_viewport_sequence",
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
        output_schema=server._tool_output_schema(
            BlenderCreateObjectResponse,
            ErrorResponse,
        ),
        task=server._task_optional(),
    )
    async def create_blender_object(
        object_type: str,
        name: Optional[str] = None,
        location: List[float] = [0.0, 0.0, 0.0],
        rotation_euler: List[float] = [0.0, 0.0, 0.0],
        scale: List[float] = [1.0, 1.0, 1.0],
    ) -> Dict[str, Any]:
        rate_error = server._check_rate_limit("create_blender_object")
        if rate_error:
            return rate_error
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
        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.blender_adapter.create_session() as session:
                payload = session.create_object(
                    input_data.object_type,
                    input_data.name,
                    list(input_data.location),
                    list(input_data.rotation_euler),
                    list(input_data.scale),
                )
                apply_success_from_error(payload)
                result = BlenderCreateObjectResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (BlenderCreateObjectResponse, ErrorResponse),
                    "create_blender_object",
                )
        except Exception as e:
            server.logger.error(f"Error creating Blender object: {e}")
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (BlenderCreateObjectResponse, ErrorResponse),
                "create_blender_object",
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
        output_schema=server._tool_output_schema(
            BlenderDeleteObjectResponse,
            ErrorResponse,
        ),
        task=server._task_optional(),
    )
    async def delete_blender_object(
        object_name: str,
    ) -> Dict[str, Any]:
        rate_error = server._check_rate_limit("delete_blender_object")
        if rate_error:
            return rate_error
        input_data = server._validate_input(
            BlenderDeleteObjectRequest,
            object_name=object_name,
        )
        if isinstance(input_data, dict):
            return input_data
        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.blender_adapter.create_session() as session:
                payload = session.delete_object(input_data.object_name)
                apply_success_from_error(payload)
                result = BlenderDeleteObjectResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (BlenderDeleteObjectResponse, ErrorResponse),
                    "delete_blender_object",
                )
        except Exception as e:
            server.logger.error(f"Error deleting Blender object: {e}")
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (BlenderDeleteObjectResponse, ErrorResponse),
                "delete_blender_object",
            )

    @server.mcp.tool(
        name="set_blender_object_transform",
        description="Set location, rotation, and/or scale on a Blender object.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            BlenderSetTransformResponse,
            ErrorResponse,
        ),
        task=server._task_optional(),
    )
    async def set_blender_object_transform(
        object_name: str,
        location: Optional[List[float]] = None,
        rotation_euler: Optional[List[float]] = None,
        scale: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        rate_error = server._check_rate_limit("set_blender_object_transform")
        if rate_error:
            return rate_error
        input_data = server._validate_input(
            BlenderSetTransformRequest,
            object_name=object_name,
            location=location,
            rotation_euler=rotation_euler,
            scale=scale,
        )
        if isinstance(input_data, dict):
            return input_data
        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.blender_adapter.create_session() as session:
                loc = list(input_data.location) if input_data.location else None
                rot = (
                    list(input_data.rotation_euler)
                    if input_data.rotation_euler
                    else None
                )
                sc = list(input_data.scale) if input_data.scale else None
                payload = session.set_object_transform(
                    input_data.object_name,
                    loc,
                    rot,
                    sc,
                )
                apply_success_from_error(payload)
                result = BlenderSetTransformResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (BlenderSetTransformResponse, ErrorResponse),
                    "set_blender_object_transform",
                )
        except Exception as e:
            server.logger.error(f"Error setting Blender transform: {e}")
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (BlenderSetTransformResponse, ErrorResponse),
                "set_blender_object_transform",
            )

    @server.mcp.tool(
        name="set_blender_object_parent",
        description="Parent one Blender object to another.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            BlenderSetParentResponse,
            ErrorResponse,
        ),
        task=server._task_optional(),
    )
    async def set_blender_object_parent(
        child_name: str,
        parent_name: str,
    ) -> Dict[str, Any]:
        rate_error = server._check_rate_limit("set_blender_object_parent")
        if rate_error:
            return rate_error
        input_data = server._validate_input(
            BlenderSetParentRequest,
            child_name=child_name,
            parent_name=parent_name,
        )
        if isinstance(input_data, dict):
            return input_data
        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.blender_adapter.create_session() as session:
                payload = session.set_object_parent(
                    input_data.child_name,
                    input_data.parent_name,
                )
                apply_success_from_error(payload)
                result = BlenderSetParentResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (BlenderSetParentResponse, ErrorResponse),
                    "set_blender_object_parent",
                )
        except Exception as e:
            server.logger.error(f"Error setting Blender parent: {e}")
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (BlenderSetParentResponse, ErrorResponse),
                "set_blender_object_parent",
            )

    @server.mcp.tool(
        name="clear_blender_object_parent",
        description="Remove parent from a Blender object.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            BlenderClearParentResponse,
            ErrorResponse,
        ),
        task=server._task_optional(),
    )
    async def clear_blender_object_parent(
        object_name: str,
        keep_transform: bool = True,
    ) -> Dict[str, Any]:
        rate_error = server._check_rate_limit("clear_blender_object_parent")
        if rate_error:
            return rate_error
        input_data = server._validate_input(
            BlenderClearParentRequest,
            object_name=object_name,
            keep_transform=keep_transform,
        )
        if isinstance(input_data, dict):
            return input_data
        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.blender_adapter.create_session() as session:
                payload = session.clear_object_parent(
                    input_data.object_name,
                    input_data.keep_transform,
                )
                apply_success_from_error(payload)
                result = BlenderClearParentResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (BlenderClearParentResponse, ErrorResponse),
                    "clear_blender_object_parent",
                )
        except Exception as e:
            server.logger.error(f"Error clearing Blender parent: {e}")
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (BlenderClearParentResponse, ErrorResponse),
                "clear_blender_object_parent",
            )

    @server.mcp.tool(
        name="assign_blender_material",
        description="Assign a Principled BSDF material to a Blender object.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            BlenderAssignMaterialResponse,
            ErrorResponse,
        ),
        task=server._task_optional(),
    )
    async def assign_blender_material(
        object_name: str,
        material_name: Optional[str] = None,
        base_color: List[float] = [0.8, 0.8, 0.8, 1.0],
        metallic: float = 0.0,
        roughness: float = 0.5,
    ) -> Dict[str, Any]:
        rate_error = server._check_rate_limit("assign_blender_material")
        if rate_error:
            return rate_error
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
        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.blender_adapter.create_session() as session:
                payload = session.assign_material(
                    input_data.object_name,
                    input_data.material_name,
                    list(input_data.base_color),
                    input_data.metallic,
                    input_data.roughness,
                )
                apply_success_from_error(payload)
                result = BlenderAssignMaterialResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (BlenderAssignMaterialResponse, ErrorResponse),
                    "assign_blender_material",
                )
        except Exception as e:
            server.logger.error(f"Error assigning Blender material: {e}")
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (BlenderAssignMaterialResponse, ErrorResponse),
                "assign_blender_material",
            )

    @server.mcp.tool(
        name="add_blender_modifier",
        description="Add a modifier to a Blender object.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            BlenderAddModifierResponse,
            ErrorResponse,
        ),
        task=server._task_optional(),
    )
    async def add_blender_modifier(
        object_name: str,
        modifier_type: str,
        modifier_name: Optional[str] = None,
        params: Dict[str, Any] = {},
    ) -> Dict[str, Any]:
        rate_error = server._check_rate_limit("add_blender_modifier")
        if rate_error:
            return rate_error
        input_data = server._validate_input(
            BlenderAddModifierRequest,
            object_name=object_name,
            modifier_type=modifier_type,
            modifier_name=modifier_name,
            params=params,
        )
        if isinstance(input_data, dict):
            return input_data
        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.blender_adapter.create_session() as session:
                payload = session.add_modifier(
                    input_data.object_name,
                    input_data.modifier_type,
                    input_data.modifier_name,
                    dict(input_data.params),
                )
                apply_success_from_error(payload)
                result = BlenderAddModifierResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (BlenderAddModifierResponse, ErrorResponse),
                    "add_blender_modifier",
                )
        except Exception as e:
            server.logger.error(f"Error adding Blender modifier: {e}")
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (BlenderAddModifierResponse, ErrorResponse),
                "add_blender_modifier",
            )

    @server.mcp.tool(
        name="set_blender_light_params",
        description="Set light parameters on a Blender light object.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            BlenderSetLightParamsResponse,
            ErrorResponse,
        ),
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
        rate_error = server._check_rate_limit("set_blender_light_params")
        if rate_error:
            return rate_error
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
        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.blender_adapter.create_session() as session:
                col = list(input_data.color) if input_data.color else None
                payload = session.set_light_params(
                    input_data.light_name,
                    input_data.energy,
                    col,
                    input_data.use_shadow,
                    input_data.spot_size,
                    input_data.spot_blend,
                    input_data.shadow_soft_size,
                )
                apply_success_from_error(payload)
                result = BlenderSetLightParamsResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (BlenderSetLightParamsResponse, ErrorResponse),
                    "set_blender_light_params",
                )
        except Exception as e:
            server.logger.error(f"Error setting Blender light params: {e}")
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (BlenderSetLightParamsResponse, ErrorResponse),
                "set_blender_light_params",
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
        output_schema=server._tool_output_schema(
            BlenderOpenFileResponse,
            ErrorResponse,
        ),
        task=server._task_optional(),
    )
    async def open_blender_file(
        file_path: str,
    ) -> Dict[str, Any]:
        rate_error = server._check_rate_limit("open_blender_file")
        if rate_error:
            return rate_error
        input_data = server._validate_input(
            BlenderOpenFileRequest,
            file_path=file_path,
        )
        if isinstance(input_data, dict):
            return input_data
        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.blender_adapter.create_session() as session:
                payload = session.open_blend_file(input_data.file_path)
                apply_success_from_error(payload)
                result = BlenderOpenFileResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (BlenderOpenFileResponse, ErrorResponse),
                    "open_blender_file",
                )
        except Exception as e:
            server.logger.error(f"Error opening Blender file: {e}")
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (BlenderOpenFileResponse, ErrorResponse),
                "open_blender_file",
            )

    @server.mcp.tool(
        name="save_blender_file",
        description="Save the current .blend file.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            BlenderSaveFileResponse,
            ErrorResponse,
        ),
        task=server._task_optional(),
    )
    async def save_blender_file(
        file_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        rate_error = server._check_rate_limit("save_blender_file")
        if rate_error:
            return rate_error
        input_data = server._validate_input(
            BlenderSaveFileRequest,
            file_path=file_path,
        )
        if isinstance(input_data, dict):
            return input_data
        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.blender_adapter.create_session() as session:
                payload = session.save_blend_file(input_data.file_path)
                apply_success_from_error(payload)
                result = BlenderSaveFileResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (BlenderSaveFileResponse, ErrorResponse),
                    "save_blender_file",
                )
        except Exception as e:
            server.logger.error(f"Error saving Blender file: {e}")
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (BlenderSaveFileResponse, ErrorResponse),
                "save_blender_file",
            )

    @server.mcp.tool(
        name="import_blender_file",
        description="Import a file into the Blender scene.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            BlenderImportFileResponse,
            ErrorResponse,
        ),
        task=server._task_optional(),
    )
    async def import_blender_file(
        file_path: str,
        file_format: str,
    ) -> Dict[str, Any]:
        rate_error = server._check_rate_limit("import_blender_file")
        if rate_error:
            return rate_error
        input_data = server._validate_input(
            BlenderImportFileRequest,
            file_path=file_path,
            file_format=file_format,
        )
        if isinstance(input_data, dict):
            return input_data
        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.blender_adapter.create_session() as session:
                payload = session.import_file(
                    input_data.file_path,
                    input_data.file_format,
                )
                apply_success_from_error(payload)
                result = BlenderImportFileResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (BlenderImportFileResponse, ErrorResponse),
                    "import_blender_file",
                )
        except Exception as e:
            server.logger.error(f"Error importing Blender file: {e}")
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (BlenderImportFileResponse, ErrorResponse),
                "import_blender_file",
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
        output_schema=server._tool_output_schema(
            BlenderExportFileResponse,
            ErrorResponse,
        ),
        task=server._task_optional(),
    )
    async def export_blender_file(
        file_path: str,
        file_format: str,
        selected_only: bool = False,
    ) -> Dict[str, Any]:
        rate_error = server._check_rate_limit("export_blender_file")
        if rate_error:
            return rate_error
        input_data = server._validate_input(
            BlenderExportFileRequest,
            file_path=file_path,
            file_format=file_format,
            selected_only=selected_only,
        )
        if isinstance(input_data, dict):
            return input_data
        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.blender_adapter.create_session() as session:
                payload = session.export_file(
                    file_path=input_data.file_path,
                    file_format=input_data.file_format,
                    selected_only=input_data.selected_only,
                )
                apply_success_from_error(payload)
                result = BlenderExportFileResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (BlenderExportFileResponse, ErrorResponse),
                    "export_blender_file",
                )
        except Exception as e:
            server.logger.error("Error exporting Blender file: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (BlenderExportFileResponse, ErrorResponse),
                "export_blender_file",
            )

    @server.mcp.tool(
        name="get_blender_file_info",
        description="Get information about the current .blend file.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            BlenderFileInfoResponse,
            ErrorResponse,
        ),
        task=server._task_optional(),
    )
    async def get_blender_file_info() -> Dict[str, Any]:
        rate_error = server._check_rate_limit("get_blender_file_info")
        if rate_error:
            return rate_error
        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.blender_adapter.create_session() as session:
                payload = session.get_file_info()
                apply_success_from_error(payload)
                result = BlenderFileInfoResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (BlenderFileInfoResponse, ErrorResponse),
                    "get_blender_file_info",
                )
        except Exception as e:
            server.logger.error("Error getting file info: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (BlenderFileInfoResponse, ErrorResponse),
                "get_blender_file_info",
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
        output_schema=server._tool_output_schema(
            BlenderSetFrameResponse,
            ErrorResponse,
        ),
        task=server._task_optional(),
    )
    async def set_blender_frame(frame: int) -> Dict[str, Any]:
        rate_error = server._check_rate_limit("set_blender_frame")
        if rate_error:
            return rate_error
        input_data = server._validate_input(
            BlenderSetFrameRequest,
            frame=frame,
        )
        if isinstance(input_data, dict):
            return input_data
        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.blender_adapter.create_session() as session:
                payload = session.set_frame(input_data.frame)
                apply_success_from_error(payload)
                result = BlenderSetFrameResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (BlenderSetFrameResponse, ErrorResponse),
                    "set_blender_frame",
                )
        except Exception as e:
            server.logger.error("Error setting frame: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (BlenderSetFrameResponse, ErrorResponse),
                "set_blender_frame",
            )

    @server.mcp.tool(
        name="get_blender_frame",
        description="Get the current frame and animation range.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            BlenderGetFrameResponse,
            ErrorResponse,
        ),
        task=server._task_optional(),
    )
    async def get_blender_frame() -> Dict[str, Any]:
        rate_error = server._check_rate_limit("get_blender_frame")
        if rate_error:
            return rate_error
        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.blender_adapter.create_session() as session:
                payload = session.get_frame()
                apply_success_from_error(payload)
                result = BlenderGetFrameResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (BlenderGetFrameResponse, ErrorResponse),
                    "get_blender_frame",
                )
        except Exception as e:
            server.logger.error("Error getting frame: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (BlenderGetFrameResponse, ErrorResponse),
                "get_blender_frame",
            )

    @server.mcp.tool(
        name="set_blender_frame_range",
        description="Set the animation frame range.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            BlenderSetFrameRangeResponse,
            ErrorResponse,
        ),
        task=server._task_optional(),
    )
    async def set_blender_frame_range(
        frame_start: int,
        frame_end: int,
    ) -> Dict[str, Any]:
        rate_error = server._check_rate_limit("set_blender_frame_range")
        if rate_error:
            return rate_error
        input_data = server._validate_input(
            BlenderSetFrameRangeRequest,
            frame_start=frame_start,
            frame_end=frame_end,
        )
        if isinstance(input_data, dict):
            return input_data
        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.blender_adapter.create_session() as session:
                payload = session.set_frame_range(
                    input_data.frame_start,
                    input_data.frame_end,
                )
                apply_success_from_error(payload)
                result = BlenderSetFrameRangeResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (BlenderSetFrameRangeResponse, ErrorResponse),
                    "set_blender_frame_range",
                )
        except Exception as e:
            server.logger.error("Error setting frame range: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (BlenderSetFrameRangeResponse, ErrorResponse),
                "set_blender_frame_range",
            )

    @server.mcp.tool(
        name="play_blender_animation",
        description="Control animation playback (play, stop, reverse).",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            BlenderPlayAnimationResponse,
            ErrorResponse,
        ),
        task=server._task_optional(),
    )
    async def play_blender_animation(
        action: str = "play",
    ) -> Dict[str, Any]:
        rate_error = server._check_rate_limit("play_blender_animation")
        if rate_error:
            return rate_error
        input_data = server._validate_input(
            BlenderPlayAnimationRequest,
            action=action,
        )
        if isinstance(input_data, dict):
            return input_data
        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.blender_adapter.create_session() as session:
                payload = session.play_animation(input_data.action)
                apply_success_from_error(payload)
                result = BlenderPlayAnimationResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (BlenderPlayAnimationResponse, ErrorResponse),
                    "play_blender_animation",
                )
        except Exception as e:
            server.logger.error("Error controlling animation: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (BlenderPlayAnimationResponse, ErrorResponse),
                "play_blender_animation",
            )

    @server.mcp.tool(
        name="insert_blender_keyframe",
        description="Insert a keyframe on an object property.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            BlenderInsertKeyframeResponse,
            ErrorResponse,
        ),
        task=server._task_optional(),
    )
    async def insert_blender_keyframe(
        object_name: str,
        data_path: str,
        frame: int,
        index: int = -1,
    ) -> Dict[str, Any]:
        rate_error = server._check_rate_limit("insert_blender_keyframe")
        if rate_error:
            return rate_error
        input_data = server._validate_input(
            BlenderInsertKeyframeRequest,
            object_name=object_name,
            data_path=data_path,
            frame=frame,
            index=index,
        )
        if isinstance(input_data, dict):
            return input_data
        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.blender_adapter.create_session() as session:
                payload = session.insert_keyframe(
                    input_data.object_name,
                    input_data.data_path,
                    input_data.frame,
                    input_data.index,
                )
                apply_success_from_error(payload)
                result = BlenderInsertKeyframeResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (
                        BlenderInsertKeyframeResponse,
                        ErrorResponse,
                    ),
                    "insert_blender_keyframe",
                )
        except Exception as e:
            server.logger.error("Error inserting keyframe: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (
                    BlenderInsertKeyframeResponse,
                    ErrorResponse,
                ),
                "insert_blender_keyframe",
            )

    @server.mcp.tool(
        name="delete_blender_keyframe",
        description="Delete a keyframe from an object property.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            BlenderDeleteKeyframeResponse,
            ErrorResponse,
        ),
        task=server._task_optional(),
    )
    async def delete_blender_keyframe(
        object_name: str,
        data_path: str,
        frame: int,
        index: int = -1,
    ) -> Dict[str, Any]:
        rate_error = server._check_rate_limit("delete_blender_keyframe")
        if rate_error:
            return rate_error
        input_data = server._validate_input(
            BlenderDeleteKeyframeRequest,
            object_name=object_name,
            data_path=data_path,
            frame=frame,
            index=index,
        )
        if isinstance(input_data, dict):
            return input_data
        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.blender_adapter.create_session() as session:
                payload = session.delete_keyframe(
                    input_data.object_name,
                    input_data.data_path,
                    input_data.frame,
                    input_data.index,
                )
                apply_success_from_error(payload)
                result = BlenderDeleteKeyframeResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (
                        BlenderDeleteKeyframeResponse,
                        ErrorResponse,
                    ),
                    "delete_blender_keyframe",
                )
        except Exception as e:
            server.logger.error("Error deleting keyframe: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (
                    BlenderDeleteKeyframeResponse,
                    ErrorResponse,
                ),
                "delete_blender_keyframe",
            )

    @server.mcp.tool(
        name="get_blender_keyframes",
        description="Get keyframe summary for an object.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            BlenderGetKeyframesResponse,
            ErrorResponse,
        ),
        task=server._task_optional(),
    )
    async def get_blender_keyframes(
        object_name: str,
    ) -> Dict[str, Any]:
        rate_error = server._check_rate_limit("get_blender_keyframes")
        if rate_error:
            return rate_error
        input_data = server._validate_input(
            BlenderGetKeyframesRequest,
            object_name=object_name,
        )
        if isinstance(input_data, dict):
            return input_data
        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.blender_adapter.create_session() as session:
                payload = session.get_keyframes(
                    input_data.object_name,
                )
                apply_success_from_error(payload)
                result = BlenderGetKeyframesResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (
                        BlenderGetKeyframesResponse,
                        ErrorResponse,
                    ),
                    "get_blender_keyframes",
                )
        except Exception as e:
            server.logger.error("Error getting keyframes: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (
                    BlenderGetKeyframesResponse,
                    ErrorResponse,
                ),
                "get_blender_keyframes",
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
        output_schema=server._tool_output_schema(
            BlenderSetupRigidBodyResponse,
            ErrorResponse,
        ),
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
        rate_error = server._check_rate_limit("setup_blender_rigid_body")
        if rate_error:
            return rate_error
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
        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.blender_adapter.create_session() as session:
                payload = session.setup_rigid_body(
                    input_data.object_name,
                    input_data.body_type,
                    input_data.mass,
                    input_data.friction,
                    input_data.restitution,
                    input_data.collision_shape,
                    input_data.linear_damping,
                    input_data.angular_damping,
                )
                apply_success_from_error(payload)
                result = BlenderSetupRigidBodyResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (
                        BlenderSetupRigidBodyResponse,
                        ErrorResponse,
                    ),
                    "setup_blender_rigid_body",
                )
        except Exception as e:
            server.logger.error("Error setting up rigid body: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (
                    BlenderSetupRigidBodyResponse,
                    ErrorResponse,
                ),
                "setup_blender_rigid_body",
            )

    @server.mcp.tool(
        name="add_blender_force_field",
        description="Add a force field to the scene.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            BlenderAddForceFieldResponse,
            ErrorResponse,
        ),
        task=server._task_optional(),
    )
    async def add_blender_force_field(
        field_type: str,
        strength: float = 1.0,
        location: Optional[List[float]] = None,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        rate_error = server._check_rate_limit("add_blender_force_field")
        if rate_error:
            return rate_error
        input_data = server._validate_input(
            BlenderAddForceFieldRequest,
            field_type=field_type,
            strength=strength,
            location=location,
            name=name,
        )
        if isinstance(input_data, dict):
            return input_data
        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.blender_adapter.create_session() as session:
                payload = session.add_force_field(
                    input_data.field_type,
                    input_data.strength,
                    input_data.location,
                    input_data.name,
                )
                apply_success_from_error(payload)
                result = BlenderAddForceFieldResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (
                        BlenderAddForceFieldResponse,
                        ErrorResponse,
                    ),
                    "add_blender_force_field",
                )
        except Exception as e:
            server.logger.error("Error adding force field: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (
                    BlenderAddForceFieldResponse,
                    ErrorResponse,
                ),
                "add_blender_force_field",
            )

    @server.mcp.tool(
        name="get_blender_force_field_info",
        description="Get force field parameters for an object.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            BlenderGetForceFieldInfoResponse,
            ErrorResponse,
        ),
        task=server._task_optional(),
    )
    async def get_blender_force_field_info(
        object_name: str,
    ) -> Dict[str, Any]:
        rate_error = server._check_rate_limit("get_blender_force_field_info")
        if rate_error:
            return rate_error
        input_data = server._validate_input(
            BlenderGetForceFieldInfoRequest,
            object_name=object_name,
        )
        if isinstance(input_data, dict):
            return input_data
        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.blender_adapter.create_session() as session:
                payload = session.get_force_field_info(
                    input_data.object_name,
                )
                apply_success_from_error(payload)
                result = BlenderGetForceFieldInfoResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (
                        BlenderGetForceFieldInfoResponse,
                        ErrorResponse,
                    ),
                    "get_blender_force_field_info",
                )
        except Exception as e:
            server.logger.error("Error getting force field info: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (
                    BlenderGetForceFieldInfoResponse,
                    ErrorResponse,
                ),
                "get_blender_force_field_info",
            )

    @server.mcp.tool(
        name="add_blender_rigid_body_constraint",
        description="Add a rigid body constraint between two objects.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            BlenderAddConstraintResponse,
            ErrorResponse,
        ),
        task=server._task_optional(),
    )
    async def add_blender_rigid_body_constraint(
        constraint_type: str,
        object1_name: str,
        object2_name: str,
        location: Optional[List[float]] = None,
        disable_collisions: bool = True,
    ) -> Dict[str, Any]:
        rate_error = server._check_rate_limit("add_blender_rigid_body_constraint")
        if rate_error:
            return rate_error
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
        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.blender_adapter.create_session() as session:
                payload = session.add_rigid_body_constraint(
                    input_data.constraint_type,
                    input_data.object1_name,
                    input_data.object2_name,
                    input_data.location,
                    input_data.disable_collisions,
                )
                apply_success_from_error(payload)
                result = BlenderAddConstraintResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (
                        BlenderAddConstraintResponse,
                        ErrorResponse,
                    ),
                    "add_blender_rigid_body_constraint",
                )
        except Exception as e:
            server.logger.error("Error adding rigid body constraint: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (
                    BlenderAddConstraintResponse,
                    ErrorResponse,
                ),
                "add_blender_rigid_body_constraint",
            )

    @server.mcp.tool(
        name="get_blender_constraint_info",
        description="Get rigid body constraint info for an object.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            BlenderGetConstraintInfoResponse,
            ErrorResponse,
        ),
        task=server._task_optional(),
    )
    async def get_blender_constraint_info(
        object_name: str,
    ) -> Dict[str, Any]:
        rate_error = server._check_rate_limit("get_blender_constraint_info")
        if rate_error:
            return rate_error
        input_data = server._validate_input(
            BlenderGetConstraintInfoRequest,
            object_name=object_name,
        )
        if isinstance(input_data, dict):
            return input_data
        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.blender_adapter.create_session() as session:
                payload = session.get_constraint_info(
                    input_data.object_name,
                )
                apply_success_from_error(payload)
                result = BlenderGetConstraintInfoResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (
                        BlenderGetConstraintInfoResponse,
                        ErrorResponse,
                    ),
                    "get_blender_constraint_info",
                )
        except Exception as e:
            server.logger.error("Error getting constraint info: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (
                    BlenderGetConstraintInfoResponse,
                    ErrorResponse,
                ),
                "get_blender_constraint_info",
            )

    @server.mcp.tool(
        name="get_blender_physics_state",
        description="Get current physics state of an object.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            BlenderGetPhysicsStateResponse,
            ErrorResponse,
        ),
        task=server._task_optional(),
    )
    async def get_blender_physics_state(
        object_name: str,
    ) -> Dict[str, Any]:
        rate_error = server._check_rate_limit("get_blender_physics_state")
        if rate_error:
            return rate_error
        input_data = server._validate_input(
            BlenderGetPhysicsStateRequest,
            object_name=object_name,
        )
        if isinstance(input_data, dict):
            return input_data
        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.blender_adapter.create_session() as session:
                payload = session.get_physics_state(
                    input_data.object_name,
                )
                apply_success_from_error(payload)
                result = BlenderGetPhysicsStateResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (
                        BlenderGetPhysicsStateResponse,
                        ErrorResponse,
                    ),
                    "get_blender_physics_state",
                )
        except Exception as e:
            server.logger.error("Error getting physics state: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (
                    BlenderGetPhysicsStateResponse,
                    ErrorResponse,
                ),
                "get_blender_physics_state",
            )

    @server.mcp.tool(
        name="get_blender_object_trajectory",
        description="Sample object position over a frame range.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            BlenderGetTrajectoryResponse,
            ErrorResponse,
        ),
        task=server._task_optional(),
    )
    async def get_blender_object_trajectory(
        object_name: str,
        start_frame: int,
        end_frame: int,
        step: int = 1,
    ) -> Dict[str, Any]:
        rate_error = server._check_rate_limit("get_blender_object_trajectory")
        if rate_error:
            return rate_error
        input_data = server._validate_input(
            BlenderGetTrajectoryRequest,
            object_name=object_name,
            start_frame=start_frame,
            end_frame=end_frame,
            step=step,
        )
        if isinstance(input_data, dict):
            return input_data
        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.blender_adapter.create_session() as session:
                payload = session.get_object_trajectory(
                    input_data.object_name,
                    input_data.start_frame,
                    input_data.end_frame,
                    input_data.step,
                )
                apply_success_from_error(payload)
                result = BlenderGetTrajectoryResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (
                        BlenderGetTrajectoryResponse,
                        ErrorResponse,
                    ),
                    "get_blender_object_trajectory",
                )
        except Exception as e:
            server.logger.error("Error getting trajectory: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (
                    BlenderGetTrajectoryResponse,
                    ErrorResponse,
                ),
                "get_blender_object_trajectory",
            )

    @server.mcp.tool(
        name="bake_blender_simulation",
        description="Bake physics simulation for a frame range.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            BlenderBakeSimulationResponse,
            ErrorResponse,
        ),
        task=server._task_optional(),
    )
    async def bake_blender_simulation(
        frame_start: int,
        frame_end: int,
    ) -> Dict[str, Any]:
        rate_error = server._check_rate_limit("bake_blender_simulation")
        if rate_error:
            return rate_error
        input_data = server._validate_input(
            BlenderBakeSimulationRequest,
            frame_start=frame_start,
            frame_end=frame_end,
        )
        if isinstance(input_data, dict):
            return input_data
        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.blender_adapter.create_session() as session:
                payload = session.bake_simulation(
                    input_data.frame_start,
                    input_data.frame_end,
                )
                apply_success_from_error(payload)
                result = BlenderBakeSimulationResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (
                        BlenderBakeSimulationResponse,
                        ErrorResponse,
                    ),
                    "bake_blender_simulation",
                )
        except Exception as e:
            server.logger.error("Error baking simulation: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (
                    BlenderBakeSimulationResponse,
                    ErrorResponse,
                ),
                "bake_blender_simulation",
            )

    @server.mcp.tool(
        name="free_blender_bake",
        description="Free (delete) baked physics simulation data.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            BlenderFreeBakeResponse,
            ErrorResponse,
        ),
        task=server._task_optional(),
    )
    async def free_blender_bake() -> Dict[str, Any]:
        rate_error = server._check_rate_limit("free_blender_bake")
        if rate_error:
            return rate_error
        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.blender_adapter.create_session() as session:
                # Void side-effect: session.free_bake() returns None,
                # so there is no payload to inspect for an `error` key.
                # If the call fails it raises, and the outer except
                # clause emits ErrorResponse. Reaching this line means
                # the bake was actually freed — success=True is genuine.
                session.free_bake()
                result = BlenderFreeBakeResponse(
                    success=True,
                ).model_dump()
                return server._validate_output(
                    result,
                    (
                        BlenderFreeBakeResponse,
                        ErrorResponse,
                    ),
                    "free_blender_bake",
                )
        except Exception as e:
            server.logger.error("Error freeing bake: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (
                    BlenderFreeBakeResponse,
                    ErrorResponse,
                ),
                "free_blender_bake",
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
        output_schema=server._tool_output_schema(
            BlenderExecuteScriptResponse,
            ErrorResponse,
        ),
        task=server._task_optional(),
    )
    async def execute_blender_script(
        script: str,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        rate_error = server._check_rate_limit("execute_blender_script")
        if rate_error:
            return rate_error
        input_data = server._validate_input(
            BlenderExecuteScriptRequest,
            script=script,
            timeout=timeout,
        )
        if isinstance(input_data, dict):
            return input_data
        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.blender_adapter.create_session() as session:
                payload = session.execute_script(
                    input_data.script,
                    input_data.timeout,
                )
                apply_success_from_error(payload)
                result = BlenderExecuteScriptResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (BlenderExecuteScriptResponse, ErrorResponse),
                    "execute_blender_script",
                )
        except Exception as e:
            server.logger.error("Error executing Blender script: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (BlenderExecuteScriptResponse, ErrorResponse),
                "execute_blender_script",
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
        output_schema=server._tool_output_schema(
            BlenderCreateMeshFromDataResponse,
            ErrorResponse,
        ),
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
        rate_error = server._check_rate_limit("create_blender_mesh_from_data")
        if rate_error:
            return rate_error
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
        try:
            if not server.blender_adapter or not server.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.blender_adapter.create_session() as session:
                payload = session.create_mesh_from_data(
                    name=input_data.name,
                    vertices=[list(v) for v in input_data.vertices],
                    edges=[list(e) for e in input_data.edges],
                    faces=[list(f) for f in input_data.faces],
                    location=(
                        list(input_data.location) if input_data.location else None
                    ),
                    collection_name=input_data.collection_name,
                )
                apply_success_from_error(payload)
                result = BlenderCreateMeshFromDataResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (
                        BlenderCreateMeshFromDataResponse,
                        ErrorResponse,
                    ),
                    "create_blender_mesh_from_data",
                )
        except Exception as e:
            server.logger.error("Error creating mesh from data: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (
                    BlenderCreateMeshFromDataResponse,
                    ErrorResponse,
                ),
                "create_blender_mesh_from_data",
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
        output_schema=server._tool_output_schema(
            SimReadyApplyMetadataResponse,
            ErrorResponse,
        ),
        task=server._task_optional(),
    )
    async def apply_simready_metadata(
        object_name: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        rate_error = server._check_rate_limit("apply_simready_metadata")
        if rate_error:
            return rate_error
        input_data = server._validate_input(
            SimReadyApplyMetadataRequest,
            object_name=object_name,
            metadata=metadata,
        )
        if isinstance(input_data, dict):
            return input_data
        try:
            with server.blender_adapter.create_session() as session:
                payload = session.apply_simready_metadata(
                    object_name=input_data.object_name,
                    metadata=input_data.metadata.model_dump(exclude_none=True),
                )
                apply_success_from_error(payload)
                result = SimReadyApplyMetadataResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (SimReadyApplyMetadataResponse, ErrorResponse),
                    "apply_simready_metadata",
                )
        except Exception as e:
            server.logger.error("Error applying SimReady metadata: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (SimReadyApplyMetadataResponse, ErrorResponse),
                "apply_simready_metadata",
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
        output_schema=server._tool_output_schema(
            SimReadyGetMetadataResponse,
            ErrorResponse,
        ),
        task=server._task_optional(),
    )
    async def get_simready_metadata(
        object_name: str,
    ) -> Dict[str, Any]:
        rate_error = server._check_rate_limit("get_simready_metadata")
        if rate_error:
            return rate_error
        input_data = server._validate_input(
            SimReadyGetMetadataRequest,
            object_name=object_name,
        )
        if isinstance(input_data, dict):
            return input_data
        try:
            with server.blender_adapter.create_session() as session:
                payload = session.get_simready_metadata(
                    object_name=input_data.object_name,
                )
                apply_success_from_error(payload)
                result = SimReadyGetMetadataResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (SimReadyGetMetadataResponse, ErrorResponse),
                    "get_simready_metadata",
                )
        except Exception as e:
            server.logger.error("Error getting SimReady metadata: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (SimReadyGetMetadataResponse, ErrorResponse),
                "get_simready_metadata",
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
        output_schema=server._tool_output_schema(
            SimReadyValidateResponse,
            ErrorResponse,
        ),
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
        rate_error = server._check_rate_limit("validate_simready_compliance")
        if rate_error:
            return rate_error
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
        try:
            with server.blender_adapter.create_session() as session:
                payload = session.validate_simready_compliance(
                    object_names=input_data.object_names,
                    check_naming=input_data.check_naming,
                    check_scale=input_data.check_scale,
                    check_transforms=input_data.check_transforms,
                    check_materials=input_data.check_materials,
                    check_hierarchy=input_data.check_hierarchy,
                )
                apply_success_from_error(payload)
                result = SimReadyValidateResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (SimReadyValidateResponse, ErrorResponse),
                    "validate_simready_compliance",
                )
        except Exception as e:
            server.logger.error("Error validating SimReady compliance: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (SimReadyValidateResponse, ErrorResponse),
                "validate_simready_compliance",
            )

    @server.mcp.tool(
        name="export_simready_usd",
        description=(
            "Export a SimReady-compliant USD file. Validates objects "
            "before export, selects the requested objects, and carries "
            "simready_ custom properties into USD attributes."
        ),
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            SimReadyExportResponse,
            ErrorResponse,
        ),
        task=server._task_optional(),
    )
    async def export_simready_usd(
        file_path: str,
        object_names: Optional[List[str]] = None,
        embed_metadata: bool = True,
        validate_before_export: bool = True,
    ) -> Dict[str, Any]:
        rate_error = server._check_rate_limit("export_simready_usd")
        if rate_error:
            return rate_error
        input_data = server._validate_input(
            SimReadyExportRequest,
            file_path=file_path,
            object_names=object_names,
            embed_metadata=embed_metadata,
            validate_before_export=validate_before_export,
        )
        if isinstance(input_data, dict):
            return input_data
        try:
            with server.blender_adapter.create_session() as session:
                payload = session.export_simready_usd(
                    file_path=input_data.file_path,
                    object_names=input_data.object_names,
                    embed_metadata=input_data.embed_metadata,
                    validate_before_export=input_data.validate_before_export,
                )
                apply_success_from_error(payload)
                result = SimReadyExportResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (SimReadyExportResponse, ErrorResponse),
                    "export_simready_usd",
                )
        except Exception as e:
            server.logger.error("Error exporting SimReady USD: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (SimReadyExportResponse, ErrorResponse),
                "export_simready_usd",
            )

    @server.mcp.tool(
        name="setup_simready_hierarchy",
        description=(
            "Create a SimReady-compliant object hierarchy with a root "
            "empty (XForm equivalent) and parent the given children "
            "under it. Optionally applies semantic labels to the root."
        ),
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            SimReadySetupHierarchyResponse,
            ErrorResponse,
        ),
        task=server._task_optional(),
    )
    async def setup_simready_hierarchy(
        root_name: str,
        child_names: List[str],
        semantic: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        rate_error = server._check_rate_limit("setup_simready_hierarchy")
        if rate_error:
            return rate_error
        input_data = server._validate_input(
            SimReadySetupHierarchyRequest,
            root_name=root_name,
            child_names=child_names,
            semantic=semantic,
        )
        if isinstance(input_data, dict):
            return input_data
        try:
            with server.blender_adapter.create_session() as session:
                payload = session.setup_simready_hierarchy(
                    root_name=input_data.root_name,
                    child_names=input_data.child_names,
                    semantic=(
                        input_data.semantic.model_dump(exclude_none=True)
                        if input_data.semantic
                        else None
                    ),
                )
                apply_success_from_error(payload)
                result = SimReadySetupHierarchyResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (
                        SimReadySetupHierarchyResponse,
                        ErrorResponse,
                    ),
                    "setup_simready_hierarchy",
                )
        except Exception as e:
            server.logger.error("Error setting up SimReady hierarchy: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (
                    SimReadySetupHierarchyResponse,
                    ErrorResponse,
                ),
                "setup_simready_hierarchy",
            )
