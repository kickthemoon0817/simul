"""
Unreal Engine runtime tool registration for Simul MCP Server.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..schemas.common import ErrorResponse
from ..schemas.unreal import *

if TYPE_CHECKING:
    from ..server import SimulMCPServer


def register_unreal_tools(server: "SimulMCPServer", thin: bool = False) -> None:
    """Register Unreal Engine runtime specific tools.

    Args:
        server: The MCP server instance.
        thin: When True, only register essential MCP tools
              (health_check, capture_viewport, execute_script).
              Full operations are available via CLI: ``simul unreal --help``.
    """

    @server.mcp.tool(
        name="unreal_health_check",
        description="Check connectivity to the Unreal Engine Remote Control API.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealHealthCheckResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def unreal_health_check() -> Dict[str, Any]:
        """
        Check connectivity to the Unreal Engine Remote Control API.

        Returns:
            Connection status or error response.
        """
        rate_error = server._check_rate_limit("unreal_health_check")
        if rate_error:
            return rate_error

        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()

            with server.unreal_adapter.create_session() as session:
                payload = await session.health_check()
                payload["success"] = True
                result = UnrealHealthCheckResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealHealthCheckResponse, ErrorResponse),
                    "unreal_health_check",
                )

        except Exception as e:
            server.logger.error("Error in Unreal health check: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealHealthCheckResponse, ErrorResponse),
                "unreal_health_check",
            )

    @server.mcp.tool(
        name="capture_unreal_viewport",
        description="Capture a viewport screenshot via HighResScreenshot.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=False,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealCaptureViewportResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def capture_unreal_viewport(
        resolution_x: int = 1920,
        resolution_y: int = 1080,
        format: str = "png",
    ) -> Dict[str, Any]:
        """
        Capture viewport screenshot.

        Args:
            resolution_x: Width in pixels.
            resolution_y: Height in pixels.
            format: Image format (png or jpeg).

        Returns:
            Capture result or error response.
        """
        _VALID_FORMATS = {"png", "jpeg", "jpg"}
        if format not in _VALID_FORMATS:
            return ErrorResponse(
                error=f"Invalid format '{format}'. Must be one of {sorted(_VALID_FORMATS)}",
                error_type="ValidationError",
            ).dict()

        rate_error = server._check_rate_limit("capture_unreal_viewport")
        if rate_error:
            return rate_error

        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with server.unreal_adapter.create_session() as session:
                payload = await session.capture_viewport(
                    resolution_x=resolution_x,
                    resolution_y=resolution_y,
                    format=format,
                )
                payload["success"] = True
                result = UnrealCaptureViewportResponse(**payload).dict()
                return server._validate_output(
                    result,
                    (UnrealCaptureViewportResponse, ErrorResponse),
                    "capture_unreal_viewport",
                )

        except Exception as e:
            server.logger.error("Error capturing Unreal viewport: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").dict()
            return server._validate_output(
                result,
                (UnrealCaptureViewportResponse, ErrorResponse),
                "capture_unreal_viewport",
            )

    @server.mcp.tool(
        name="execute_unreal_script",
        description=(
            "Execute arbitrary Python code inside the Unreal Engine editor. "
            "Use for operations not covered by other tools. The code runs via "
            "PythonScriptLibrary.ExecutePythonCommandEx. Print JSON to return "
            "structured data. For granular CLI operations see: simul unreal --help"
        ),
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
            destructive=True,
        ),
        task=server._task_optional(),
    )
    async def execute_unreal_script(
        code: str,
        mode: str = "ExecuteFile",
    ) -> Dict[str, Any]:
        """
        Execute Python code inside Unreal Engine.

        Args:
            code: Python source code to execute.
            mode: ExecuteFile (multi-line), EvaluateStatement (expression),
                  or ExecuteStatement (single statement).

        Returns:
            Execution result with CommandResult, LogOutput, ReturnValue.
        """
        _VALID_EXEC_MODES = {"ExecuteFile", "EvaluateStatement", "ExecuteStatement"}
        if mode not in _VALID_EXEC_MODES:
            return ErrorResponse(
                error=f"Invalid mode '{mode}'. Must be one of {sorted(_VALID_EXEC_MODES)}",
                error_type="ValidationError",
            ).model_dump()

        rate_error = server._check_rate_limit("execute_unreal_script")
        if rate_error:
            return rate_error

        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()

            with server.unreal_adapter.create_session() as session:
                raw = await session._execute_python(code, mode=mode)
                parsed = session._parse_python_json(raw)
                if parsed.get("error"):
                    return ErrorResponse(
                        error=parsed["error"],
                        error_type="ScriptError",
                    ).model_dump()
                parsed["success"] = True
                return parsed

        except Exception as e:
            server.logger.error("Error executing Unreal script: %s", e)
            return ErrorResponse(error=str(e), error_type="Exception").model_dump()

    # -- Thin mode: only health_check, capture_viewport, execute_script.
    #    Full operations available via CLI: simul unreal --help
    if thin:
        return

    # -- Full MCP tool set below -------------------------------------------

    @server.mcp.tool(
        name="get_unreal_engine_info",
        description="Get Unreal Engine runtime information.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealEngineInfoResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def get_unreal_engine_info() -> Dict[str, Any]:
        """
        Get Unreal Engine runtime information.

        Returns:
            Engine info or error response.
        """
        rate_error = server._check_rate_limit("get_unreal_engine_info")
        if rate_error:
            return rate_error

        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()

            with server.unreal_adapter.create_session() as session:
                payload = await session.get_engine_info()
                payload["success"] = True
                result = UnrealEngineInfoResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealEngineInfoResponse, ErrorResponse),
                    "get_unreal_engine_info",
                )

        except Exception as e:
            server.logger.error("Error getting Unreal engine info: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealEngineInfoResponse, ErrorResponse),
                "get_unreal_engine_info",
            )

    @server.mcp.tool(
        name="get_unreal_loaded_map",
        description="Get the currently loaded persistent level path.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealLoadedMapResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def get_unreal_loaded_map() -> Dict[str, Any]:
        """
        Get the currently loaded persistent level path.

        Returns:
            Loaded map path or error response.
        """
        rate_error = server._check_rate_limit("get_unreal_loaded_map")
        if rate_error:
            return rate_error

        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()

            with server.unreal_adapter.create_session() as session:
                payload = await session.get_loaded_map()
                payload["success"] = True
                result = UnrealLoadedMapResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealLoadedMapResponse, ErrorResponse),
                    "get_unreal_loaded_map",
                )

        except Exception as e:
            server.logger.error("Error getting Unreal loaded map: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealLoadedMapResponse, ErrorResponse),
                "get_unreal_loaded_map",
            )

    # ------------------------------------------------------------------
    # Phase 1: Scene Read Operations
    # ------------------------------------------------------------------

    @server.mcp.tool(
        name="list_unreal_actors",
        description="List actors in the current Unreal Engine level with optional class and tag filters.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealListActorsResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def list_unreal_actors(
        class_filter: str = "",
        tag_filter: str = "",
        max_results: int = 200,
    ) -> Dict[str, Any]:
        """
        List actors in the current Unreal Engine level.

        Args:
            class_filter: Filter by UClass name (empty string = no filter).
            tag_filter: Filter by actor tag (empty string = no filter).
            max_results: Maximum number of actors to return.

        Returns:
            Actor listing or error response.
        """
        rate_error = server._check_rate_limit("list_unreal_actors")
        if rate_error:
            return rate_error

        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()

            with server.unreal_adapter.create_session() as session:
                payload = await session.list_actors(
                    class_filter=class_filter or None,
                    tag_filter=tag_filter or None,
                    max_results=max_results,
                )
                payload["success"] = True
                result = UnrealListActorsResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealListActorsResponse, ErrorResponse),
                    "list_unreal_actors",
                )

        except Exception as e:
            server.logger.error("Error listing Unreal actors: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealListActorsResponse, ErrorResponse),
                "list_unreal_actors",
            )

    @server.mcp.tool(
        name="get_unreal_actor_info",
        description="Get detailed information about a specific actor including transform, components, and tags.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealGetActorInfoResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def get_unreal_actor_info(actor_path: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific actor.

        Args:
            actor_path: Full object path of the actor.

        Returns:
            Actor info or error response.
        """
        rate_error = server._check_rate_limit("get_unreal_actor_info")
        if rate_error:
            return rate_error

        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()

            with server.unreal_adapter.create_session() as session:
                payload = await session.get_actor_info(actor_path)
                payload["success"] = True
                result = UnrealGetActorInfoResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealGetActorInfoResponse, ErrorResponse),
                    "get_unreal_actor_info",
                )

        except Exception as e:
            server.logger.error("Error getting Unreal actor info: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealGetActorInfoResponse, ErrorResponse),
                "get_unreal_actor_info",
            )

    @server.mcp.tool(
        name="search_unreal_assets",
        description="Search the Unreal Asset Registry by name, class, or package path.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealSearchAssetsResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def search_unreal_assets(
        query: str = "",
        class_names: str = "",
        package_paths: str = "",
        max_results: int = 100,
    ) -> Dict[str, Any]:
        """
        Search the Unreal Asset Registry.

        Args:
            query: Search query string.
            class_names: Comma-separated UClass names to filter.
            package_paths: Comma-separated package paths to search.
            max_results: Maximum number of results.

        Returns:
            Asset search results or error response.
        """
        rate_error = server._check_rate_limit("search_unreal_assets")
        if rate_error:
            return rate_error

        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()

            parsed_classes = (
                [c.strip() for c in class_names.split(",") if c.strip()]
                if class_names
                else None
            )
            parsed_paths = (
                [p.strip() for p in package_paths.split(",") if p.strip()]
                if package_paths
                else None
            )

            with server.unreal_adapter.create_session() as session:
                payload = await session.search_assets(
                    query=query,
                    class_names=parsed_classes,
                    package_paths=parsed_paths,
                    max_results=max_results,
                )
                payload["success"] = True
                result = UnrealSearchAssetsResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealSearchAssetsResponse, ErrorResponse),
                    "search_unreal_assets",
                )

        except Exception as e:
            server.logger.error("Error searching Unreal assets: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealSearchAssetsResponse, ErrorResponse),
                "search_unreal_assets",
            )

    @server.mcp.tool(
        name="describe_unreal_object",
        description="Get full property and function metadata for any UObject by path.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealDescribeObjectResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def describe_unreal_object(object_path: str) -> Dict[str, Any]:
        """
        Describe a UObject's properties and functions.

        Args:
            object_path: Full object path.

        Returns:
            Object description or error response.
        """
        rate_error = server._check_rate_limit("describe_unreal_object")
        if rate_error:
            return rate_error

        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()

            with server.unreal_adapter.create_session() as session:
                payload = await session.describe_object(object_path)
                payload["success"] = True
                result = UnrealDescribeObjectResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealDescribeObjectResponse, ErrorResponse),
                    "describe_unreal_object",
                )

        except Exception as e:
            server.logger.error("Error describing Unreal object: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealDescribeObjectResponse, ErrorResponse),
                "describe_unreal_object",
            )

    @server.mcp.tool(
        name="get_unreal_actor_thumbnail",
        description="Get a thumbnail image for an Unreal asset.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=False,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealGetThumbnailResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def get_unreal_actor_thumbnail(
        asset_path: str,
        width: int = 256,
        height: int = 256,
    ) -> Dict[str, Any]:
        """
        Get a thumbnail image for an asset.

        Args:
            asset_path: Full asset path.
            width: Thumbnail width in pixels.
            height: Thumbnail height in pixels.

        Returns:
            Thumbnail data or error response.
        """
        rate_error = server._check_rate_limit("get_unreal_actor_thumbnail")
        if rate_error:
            return rate_error

        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()

            with server.unreal_adapter.create_session() as session:
                payload = await session.get_actor_thumbnail(
                    asset_path=asset_path, width=width, height=height
                )
                payload["success"] = True
                result = UnrealGetThumbnailResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealGetThumbnailResponse, ErrorResponse),
                    "get_unreal_actor_thumbnail",
                )

        except Exception as e:
            server.logger.error("Error getting Unreal thumbnail: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealGetThumbnailResponse, ErrorResponse),
                "get_unreal_actor_thumbnail",
            )

    @server.mcp.tool(
        name="summarize_unreal_scene",
        description="Generate an LLM-friendly digest of the current Unreal scene.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealSceneSummaryResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def summarize_unreal_scene() -> Dict[str, Any]:
        """
        Generate an LLM-friendly scene digest.

        Returns:
            Scene summary or error response.
        """
        rate_error = server._check_rate_limit("summarize_unreal_scene")
        if rate_error:
            return rate_error

        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()

            with server.unreal_adapter.create_session() as session:
                payload = await session.summarize_scene()
                payload["success"] = True
                result = UnrealSceneSummaryResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealSceneSummaryResponse, ErrorResponse),
                    "summarize_unreal_scene",
                )

        except Exception as e:
            server.logger.error("Error summarizing Unreal scene: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealSceneSummaryResponse, ErrorResponse),
                "summarize_unreal_scene",
            )

    # -- Phase 2: Viewport & Visual Observation --

    @server.mcp.tool(
        name="get_unreal_viewport_info",
        description="Get active viewport camera and render information.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealViewportInfoResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def get_unreal_viewport_info() -> Dict[str, Any]:
        """
        Get viewport camera and render settings.

        Returns:
            Viewport info or error response.
        """
        rate_error = server._check_rate_limit("get_unreal_viewport_info")
        if rate_error:
            return rate_error

        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()

            with server.unreal_adapter.create_session() as session:
                payload = await session.get_viewport_info()
                payload["success"] = True
                result = UnrealViewportInfoResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealViewportInfoResponse, ErrorResponse),
                    "get_unreal_viewport_info",
                )

        except Exception as e:
            server.logger.error("Error getting Unreal viewport info: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealViewportInfoResponse, ErrorResponse),
                "get_unreal_viewport_info",
            )

    @server.mcp.tool(
        name="set_unreal_camera_view",
        description="Set the editor viewport camera position and rotation.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealSetCameraViewResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def set_unreal_camera_view(
        location_x: float = 0.0,
        location_y: float = 0.0,
        location_z: float = 0.0,
        rotation_pitch: float = 0.0,
        rotation_yaw: float = 0.0,
        rotation_roll: float = 0.0,
        fov: float = 90.0,
    ) -> Dict[str, Any]:
        """
        Set the editor viewport camera.

        Args:
            location_x: Camera X position in cm.
            location_y: Camera Y position in cm.
            location_z: Camera Z position in cm.
            rotation_pitch: Camera pitch in degrees.
            rotation_yaw: Camera yaw in degrees.
            rotation_roll: Camera roll in degrees.
            fov: Field of view in degrees.

        Returns:
            Applied camera state or error response.
        """
        rate_error = server._check_rate_limit("set_unreal_camera_view")
        if rate_error:
            return rate_error

        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()

            with server.unreal_adapter.create_session() as session:
                payload = await session.set_camera_view(
                    location=(location_x, location_y, location_z),
                    rotation=(rotation_pitch, rotation_yaw, rotation_roll),
                    fov=fov,
                )
                payload["success"] = True
                result = UnrealSetCameraViewResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealSetCameraViewResponse, ErrorResponse),
                    "set_unreal_camera_view",
                )

        except Exception as e:
            server.logger.error("Error setting Unreal camera view: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealSetCameraViewResponse, ErrorResponse),
                "set_unreal_camera_view",
            )

    @server.mcp.tool(
        name="focus_unreal_on_actor",
        description="Focus the editor viewport camera on a specific actor.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealFocusActorResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def focus_unreal_on_actor(
        actor_path: str,
        distance: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Focus viewport camera on a specific actor.

        Args:
            actor_path: Full actor path to focus on.
            distance: Camera distance from actor (0 = auto-fit).

        Returns:
            Focus result or error response.
        """
        rate_error = server._check_rate_limit("focus_unreal_on_actor")
        if rate_error:
            return rate_error

        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()

            with server.unreal_adapter.create_session() as session:
                payload = await session.focus_on_actor(
                    actor_path=actor_path, distance=distance
                )
                payload["success"] = True
                result = UnrealFocusActorResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealFocusActorResponse, ErrorResponse),
                    "focus_unreal_on_actor",
                )

        except Exception as e:
            server.logger.error("Error focusing on Unreal actor: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealFocusActorResponse, ErrorResponse),
                "focus_unreal_on_actor",
            )

    # -- Phase 3: Scene Manipulation --

    @server.mcp.tool(
        name="spawn_unreal_actor",
        description="Spawn an actor from a class or asset path.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealSpawnActorResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def spawn_unreal_actor(
        asset_path: str,
        location_x: float = 0.0,
        location_y: float = 0.0,
        location_z: float = 0.0,
        rotation_pitch: float = 0.0,
        rotation_yaw: float = 0.0,
        rotation_roll: float = 0.0,
        label: str = "",
    ) -> Dict[str, Any]:
        """Spawn an actor from class or asset path."""
        rate_error = server._check_rate_limit("spawn_unreal_actor")
        if rate_error:
            return rate_error
        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.unreal_adapter.create_session() as session:
                payload = await session.spawn_actor(
                    asset_path=asset_path,
                    location=(location_x, location_y, location_z),
                    rotation=(rotation_pitch, rotation_yaw, rotation_roll),
                    label=label or None,
                )
                payload["success"] = True
                result = UnrealSpawnActorResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealSpawnActorResponse, ErrorResponse),
                    "spawn_unreal_actor",
                )
        except Exception as e:
            server.logger.error("Error spawning Unreal actor: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealSpawnActorResponse, ErrorResponse),
                "spawn_unreal_actor",
            )

    @server.mcp.tool(
        name="delete_unreal_actor",
        description="Delete an actor from the level. DESTRUCTIVE operation.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
            destructive=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealDeleteActorResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def delete_unreal_actor(
        actor_path: str,
    ) -> Dict[str, Any]:
        """Delete an actor from the level."""
        rate_error = server._check_rate_limit("delete_unreal_actor")
        if rate_error:
            return rate_error
        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.unreal_adapter.create_session() as session:
                payload = await session.delete_actor(actor_path=actor_path)
                payload["success"] = True
                result = UnrealDeleteActorResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealDeleteActorResponse, ErrorResponse),
                    "delete_unreal_actor",
                )
        except Exception as e:
            server.logger.error("Error deleting Unreal actor: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealDeleteActorResponse, ErrorResponse),
                "delete_unreal_actor",
            )

    @server.mcp.tool(
        name="set_unreal_actor_transform",
        description="Set an actor's location, rotation, and scale.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealSetActorTransformResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def set_unreal_actor_transform(
        actor_path: str,
        location_x: float = 0.0,
        location_y: float = 0.0,
        location_z: float = 0.0,
        rotation_pitch: float = 0.0,
        rotation_yaw: float = 0.0,
        rotation_roll: float = 0.0,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        scale_z: float = 1.0,
    ) -> Dict[str, Any]:
        """Set an actor's transform."""
        rate_error = server._check_rate_limit("set_unreal_actor_transform")
        if rate_error:
            return rate_error
        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.unreal_adapter.create_session() as session:
                payload = await session.set_actor_transform(
                    actor_path=actor_path,
                    location=(location_x, location_y, location_z),
                    rotation=(rotation_pitch, rotation_yaw, rotation_roll),
                    scale=(scale_x, scale_y, scale_z),
                )
                payload["success"] = True
                result = UnrealSetActorTransformResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealSetActorTransformResponse, ErrorResponse),
                    "set_unreal_actor_transform",
                )
        except Exception as e:
            server.logger.error("Error setting Unreal actor transform: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealSetActorTransformResponse, ErrorResponse),
                "set_unreal_actor_transform",
            )

    @server.mcp.tool(
        name="set_unreal_actor_property",
        description="Set a property on an Unreal actor by name and JSON value.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealSetActorPropertyResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def set_unreal_actor_property(
        actor_path: str,
        property_name: str,
        property_value: str,
        generate_transaction: bool = True,
    ) -> Dict[str, Any]:
        """Set a property on an actor."""
        rate_error = server._check_rate_limit("set_unreal_actor_property")
        if rate_error:
            return rate_error
        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.unreal_adapter.create_session() as session:
                payload = await session.set_actor_property(
                    actor_path=actor_path,
                    property_name=property_name,
                    property_value=property_value,
                    generate_transaction=generate_transaction,
                )
                payload["success"] = True
                result = UnrealSetActorPropertyResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealSetActorPropertyResponse, ErrorResponse),
                    "set_unreal_actor_property",
                )
        except Exception as e:
            server.logger.error("Error setting Unreal actor property: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealSetActorPropertyResponse, ErrorResponse),
                "set_unreal_actor_property",
            )

    @server.mcp.tool(
        name="call_unreal_actor_function",
        description="Call a BlueprintCallable UFUNCTION on an actor.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealCallActorFunctionResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def call_unreal_actor_function(
        actor_path: str,
        function_name: str,
        parameters: str = "",
    ) -> Dict[str, Any]:
        """Call a UFUNCTION on an actor."""
        rate_error = server._check_rate_limit("call_unreal_actor_function")
        if rate_error:
            return rate_error
        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.unreal_adapter.create_session() as session:
                payload = await session.call_actor_function(
                    actor_path=actor_path,
                    function_name=function_name,
                    parameters=parameters or None,
                )
                payload["success"] = True
                result = UnrealCallActorFunctionResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealCallActorFunctionResponse, ErrorResponse),
                    "call_unreal_actor_function",
                )
        except Exception as e:
            server.logger.error("Error calling Unreal actor function: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealCallActorFunctionResponse, ErrorResponse),
                "call_unreal_actor_function",
            )

    @server.mcp.tool(
        name="set_unreal_actor_parent",
        description="Attach an actor to a parent actor or detach it.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealSetActorParentResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def set_unreal_actor_parent(
        actor_path: str,
        parent_path: str = "",
    ) -> Dict[str, Any]:
        """Attach or detach an actor."""
        rate_error = server._check_rate_limit("set_unreal_actor_parent")
        if rate_error:
            return rate_error
        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.unreal_adapter.create_session() as session:
                payload = await session.set_actor_parent(
                    actor_path=actor_path,
                    parent_path=parent_path or None,
                )
                payload["success"] = True
                result = UnrealSetActorParentResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealSetActorParentResponse, ErrorResponse),
                    "set_unreal_actor_parent",
                )
        except Exception as e:
            server.logger.error("Error setting Unreal actor parent: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealSetActorParentResponse, ErrorResponse),
                "set_unreal_actor_parent",
            )

    @server.mcp.tool(
        name="add_unreal_component",
        description="Add a component to an Unreal actor.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealAddComponentResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def add_unreal_component(
        actor_path: str,
        component_class: str,
        component_name: str = "",
    ) -> Dict[str, Any]:
        """Add a component to an actor."""
        rate_error = server._check_rate_limit("add_unreal_component")
        if rate_error:
            return rate_error
        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.unreal_adapter.create_session() as session:
                payload = await session.add_component(
                    actor_path=actor_path,
                    component_class=component_class,
                    component_name=component_name or None,
                )
                payload["success"] = True
                result = UnrealAddComponentResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealAddComponentResponse, ErrorResponse),
                    "add_unreal_component",
                )
        except Exception as e:
            server.logger.error("Error adding Unreal component: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealAddComponentResponse, ErrorResponse),
                "add_unreal_component",
            )

    @server.mcp.tool(
        name="set_unreal_actor_visibility",
        description="Set actor visibility in the Unreal level.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealSetActorVisibilityResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def set_unreal_actor_visibility(
        actor_path: str,
        visible: bool = True,
        propagate: bool = True,
    ) -> Dict[str, Any]:
        """Set actor visibility."""
        rate_error = server._check_rate_limit("set_unreal_actor_visibility")
        if rate_error:
            return rate_error
        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.unreal_adapter.create_session() as session:
                payload = await session.set_actor_visibility(
                    actor_path=actor_path,
                    visible=visible,
                    propagate=propagate,
                )
                payload["success"] = True
                result = UnrealSetActorVisibilityResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealSetActorVisibilityResponse, ErrorResponse),
                    "set_unreal_actor_visibility",
                )
        except Exception as e:
            server.logger.error("Error setting Unreal actor visibility: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealSetActorVisibilityResponse, ErrorResponse),
                "set_unreal_actor_visibility",
            )

    # ---------------------------------------------------------------
    # Phase 4 — Materials, Lighting & Rendering
    # ---------------------------------------------------------------

    @server.mcp.tool(
        name="get_unreal_material_info",
        description="Get material instance parameters and metadata.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealGetMaterialInfoResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def get_unreal_material_info(
        material_path: str,
    ) -> Dict[str, Any]:
        """Get material info."""
        rate_error = server._check_rate_limit("get_unreal_material_info")
        if rate_error:
            return rate_error
        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.unreal_adapter.create_session() as session:
                payload = await session.get_material_info(
                    material_path=material_path,
                )
                payload["success"] = True
                result = UnrealGetMaterialInfoResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealGetMaterialInfoResponse, ErrorResponse),
                    "get_unreal_material_info",
                )
        except Exception as e:
            server.logger.error("Error getting Unreal material info: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealGetMaterialInfoResponse, ErrorResponse),
                "get_unreal_material_info",
            )

    @server.mcp.tool(
        name="set_unreal_material_params",
        description="Set scalar/vector/texture parameters on a Material Instance.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealSetMaterialParamsResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def set_unreal_material_params(
        material_path: str,
        scalar_params_json: str = "",
        vector_params_json: str = "",
        texture_params_json: str = "",
    ) -> Dict[str, Any]:
        """Set material instance parameters."""
        import json as json_lib

        rate_error = server._check_rate_limit("set_unreal_material_params")
        if rate_error:
            return rate_error
        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()

            scalar_params = (
                json_lib.loads(scalar_params_json) if scalar_params_json else None
            )
            vector_params = (
                json_lib.loads(vector_params_json) if vector_params_json else None
            )
            texture_params = (
                json_lib.loads(texture_params_json) if texture_params_json else None
            )

            with server.unreal_adapter.create_session() as session:
                payload = await session.set_material_params(
                    material_path=material_path,
                    scalar_params=scalar_params,
                    vector_params=vector_params,
                    texture_params=texture_params,
                )
                payload["success"] = True
                result = UnrealSetMaterialParamsResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealSetMaterialParamsResponse, ErrorResponse),
                    "set_unreal_material_params",
                )
        except Exception as e:
            server.logger.error("Error setting Unreal material params: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealSetMaterialParamsResponse, ErrorResponse),
                "set_unreal_material_params",
            )

    @server.mcp.tool(
        name="create_unreal_material_instance",
        description="Create a Material Instance Constant from a parent material.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealCreateMaterialInstanceResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def create_unreal_material_instance(
        parent_path: str,
        instance_name: str,
        save_path: str = "",
    ) -> Dict[str, Any]:
        """Create a material instance."""
        rate_error = server._check_rate_limit("create_unreal_material_instance")
        if rate_error:
            return rate_error
        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.unreal_adapter.create_session() as session:
                payload = await session.create_material_instance(
                    parent_path=parent_path,
                    instance_name=instance_name,
                    save_path=save_path,
                )
                payload["success"] = True
                result = UnrealCreateMaterialInstanceResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealCreateMaterialInstanceResponse, ErrorResponse),
                    "create_unreal_material_instance",
                )
        except Exception as e:
            server.logger.error("Error creating Unreal material instance: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealCreateMaterialInstanceResponse, ErrorResponse),
                "create_unreal_material_instance",
            )

    @server.mcp.tool(
        name="assign_unreal_material",
        description="Assign a material to a mesh component's material slot.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealAssignMaterialResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def assign_unreal_material(
        actor_path: str,
        material_path: str,
        slot_index: int = 0,
    ) -> Dict[str, Any]:
        """Assign material to actor."""
        rate_error = server._check_rate_limit("assign_unreal_material")
        if rate_error:
            return rate_error
        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.unreal_adapter.create_session() as session:
                payload = await session.assign_material(
                    actor_path=actor_path,
                    material_path=material_path,
                    slot_index=slot_index,
                )
                payload["success"] = True
                result = UnrealAssignMaterialResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealAssignMaterialResponse, ErrorResponse),
                    "assign_unreal_material",
                )
        except Exception as e:
            server.logger.error("Error assigning Unreal material: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealAssignMaterialResponse, ErrorResponse),
                "assign_unreal_material",
            )

    @server.mcp.tool(
        name="set_unreal_light_params",
        description="Set light component parameters (intensity, color, temperature, shadows).",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealSetLightParamsResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def set_unreal_light_params(
        actor_path: str,
        intensity: Optional[float] = None,
        color_r: Optional[float] = None,
        color_g: Optional[float] = None,
        color_b: Optional[float] = None,
        temperature: Optional[float] = None,
        use_temperature: Optional[bool] = None,
        attenuation_radius: Optional[float] = None,
        cast_shadows: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Set light parameters."""
        rate_error = server._check_rate_limit("set_unreal_light_params")
        if rate_error:
            return rate_error
        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.unreal_adapter.create_session() as session:
                payload = await session.set_light_params(
                    actor_path=actor_path,
                    intensity=intensity,
                    color_r=color_r,
                    color_g=color_g,
                    color_b=color_b,
                    temperature=temperature,
                    use_temperature=use_temperature,
                    attenuation_radius=attenuation_radius,
                    cast_shadows=cast_shadows,
                )
                payload["success"] = True
                result = UnrealSetLightParamsResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealSetLightParamsResponse, ErrorResponse),
                    "set_unreal_light_params",
                )
        except Exception as e:
            server.logger.error("Error setting Unreal light params: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealSetLightParamsResponse, ErrorResponse),
                "set_unreal_light_params",
            )

    @server.mcp.tool(
        name="set_unreal_render_settings",
        description="Set rendering or post-process settings via console command.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealSetRenderSettingsResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def set_unreal_render_settings(
        setting_name: str,
        setting_value: str,
    ) -> Dict[str, Any]:
        """Set render settings."""
        rate_error = server._check_rate_limit("set_unreal_render_settings")
        if rate_error:
            return rate_error
        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.unreal_adapter.create_session() as session:
                payload = await session.set_render_settings(
                    setting_name=setting_name,
                    setting_value=setting_value,
                )
                payload["success"] = True
                result = UnrealSetRenderSettingsResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealSetRenderSettingsResponse, ErrorResponse),
                    "set_unreal_render_settings",
                )
        except Exception as e:
            server.logger.error("Error setting Unreal render settings: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealSetRenderSettingsResponse, ErrorResponse),
                "set_unreal_render_settings",
            )

    # ---- Phase 5: Physics & Simulation Control ----

    @server.mcp.tool(
        name="control_unreal_simulation",
        description="Control Play-In-Editor (PIE) session: start, stop, pause, resume, step.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealControlSimulationResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def control_unreal_simulation(
        action: str,
    ) -> Dict[str, Any]:
        """Control PIE session."""
        rate_error = server._check_rate_limit("control_unreal_simulation")
        if rate_error:
            return rate_error
        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.unreal_adapter.create_session() as session:
                payload = await session.control_simulation(action=action)
                payload["success"] = True
                result = UnrealControlSimulationResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealControlSimulationResponse, ErrorResponse),
                    "control_unreal_simulation",
                )
        except Exception as e:
            server.logger.error("Error controlling Unreal simulation: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealControlSimulationResponse, ErrorResponse),
                "control_unreal_simulation",
            )

    @server.mcp.tool(
        name="get_unreal_simulation_status",
        description="Get current Play-In-Editor simulation status (playing, paused, stopped).",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealGetSimulationStatusResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def get_unreal_simulation_status() -> Dict[str, Any]:
        """Get PIE simulation status."""
        rate_error = server._check_rate_limit("get_unreal_simulation_status")
        if rate_error:
            return rate_error
        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.unreal_adapter.create_session() as session:
                payload = await session.get_simulation_status()
                payload["success"] = True
                result = UnrealGetSimulationStatusResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealGetSimulationStatusResponse, ErrorResponse),
                    "get_unreal_simulation_status",
                )
        except Exception as e:
            server.logger.error("Error getting Unreal simulation status: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealGetSimulationStatusResponse, ErrorResponse),
                "get_unreal_simulation_status",
            )

    @server.mcp.tool(
        name="enable_unreal_physics",
        description="Enable or disable physics simulation on an actor.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealEnablePhysicsResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def enable_unreal_physics(
        actor_path: str,
        enable: bool = True,
        simulate_physics: bool = True,
    ) -> Dict[str, Any]:
        """Enable physics on actor."""
        rate_error = server._check_rate_limit("enable_unreal_physics")
        if rate_error:
            return rate_error
        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.unreal_adapter.create_session() as session:
                payload = await session.enable_physics(
                    actor_path=actor_path,
                    enable=enable,
                    simulate_physics=simulate_physics,
                )
                payload["success"] = True
                result = UnrealEnablePhysicsResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealEnablePhysicsResponse, ErrorResponse),
                    "enable_unreal_physics",
                )
        except Exception as e:
            server.logger.error("Error enabling Unreal physics: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealEnablePhysicsResponse, ErrorResponse),
                "enable_unreal_physics",
            )

    @server.mcp.tool(
        name="set_unreal_collision",
        description="Set collision presets and enable/disable collision on an actor.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealSetCollisionResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def set_unreal_collision(
        actor_path: str,
        collision_preset: str = "",
        collision_enabled: bool = True,
    ) -> Dict[str, Any]:
        """Set collision configuration."""
        rate_error = server._check_rate_limit("set_unreal_collision")
        if rate_error:
            return rate_error
        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.unreal_adapter.create_session() as session:
                payload = await session.set_collision(
                    actor_path=actor_path,
                    collision_preset=collision_preset,
                    collision_enabled=collision_enabled,
                )
                payload["success"] = True
                result = UnrealSetCollisionResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealSetCollisionResponse, ErrorResponse),
                    "set_unreal_collision",
                )
        except Exception as e:
            server.logger.error("Error setting Unreal collision: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealSetCollisionResponse, ErrorResponse),
                "set_unreal_collision",
            )

    @server.mcp.tool(
        name="apply_unreal_force",
        description="Apply a force or impulse to an actor's physics body.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealApplyForceResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def apply_unreal_force(
        actor_path: str,
        force_x: float = 0.0,
        force_y: float = 0.0,
        force_z: float = 0.0,
        is_impulse: bool = False,
        location_x: Optional[float] = None,
        location_y: Optional[float] = None,
        location_z: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Apply force or impulse."""
        rate_error = server._check_rate_limit("apply_unreal_force")
        if rate_error:
            return rate_error
        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.unreal_adapter.create_session() as session:
                payload = await session.apply_force(
                    actor_path=actor_path,
                    force_x=force_x,
                    force_y=force_y,
                    force_z=force_z,
                    is_impulse=is_impulse,
                    location_x=location_x,
                    location_y=location_y,
                    location_z=location_z,
                )
                payload["success"] = True
                result = UnrealApplyForceResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealApplyForceResponse, ErrorResponse),
                    "apply_unreal_force",
                )
        except Exception as e:
            server.logger.error("Error applying Unreal force: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealApplyForceResponse, ErrorResponse),
                "apply_unreal_force",
            )

    @server.mcp.tool(
        name="set_unreal_physics_params",
        description="Set physics body parameters (mass, damping, gravity) on an actor.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealSetPhysicsParamsResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def set_unreal_physics_params(
        actor_path: str,
        mass: Optional[float] = None,
        linear_damping: Optional[float] = None,
        angular_damping: Optional[float] = None,
        enable_gravity: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Set physics parameters."""
        rate_error = server._check_rate_limit("set_unreal_physics_params")
        if rate_error:
            return rate_error
        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.unreal_adapter.create_session() as session:
                payload = await session.set_physics_params(
                    actor_path=actor_path,
                    mass=mass,
                    linear_damping=linear_damping,
                    angular_damping=angular_damping,
                    enable_gravity=enable_gravity,
                )
                payload["success"] = True
                result = UnrealSetPhysicsParamsResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealSetPhysicsParamsResponse, ErrorResponse),
                    "set_unreal_physics_params",
                )
        except Exception as e:
            server.logger.error("Error setting Unreal physics params: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealSetPhysicsParamsResponse, ErrorResponse),
                "set_unreal_physics_params",
            )

    # ----------------------------------------------------------
    # Phase 6: USD / SimReady Bridge
    # ----------------------------------------------------------

    @server.mcp.tool(
        name="import_unreal_usd",
        description="Import a USD file into Unreal via Interchange Framework.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealImportUsdResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def import_unreal_usd(
        usd_path: str,
        target_path: Optional[str] = None,
        import_animations: bool = True,
        import_materials: bool = True,
        scale_factor: float = 1.0,
    ) -> Dict[str, Any]:
        """Import USD file into Unreal."""
        rate_error = server._check_rate_limit("import_unreal_usd")
        if rate_error:
            return rate_error
        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.unreal_adapter.create_session() as session:
                payload = await session.import_usd(
                    usd_path=usd_path,
                    target_path=target_path,
                    import_animations=import_animations,
                    import_materials=import_materials,
                    scale_factor=scale_factor,
                )
                payload["success"] = True
                result = UnrealImportUsdResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealImportUsdResponse, ErrorResponse),
                    "import_unreal_usd",
                )
        except Exception as e:
            server.logger.error("Error importing USD to Unreal: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealImportUsdResponse, ErrorResponse),
                "import_unreal_usd",
            )

    @server.mcp.tool(
        name="export_unreal_usd",
        description="Export Unreal actors to USD via Interchange Framework.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealExportUsdResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def export_unreal_usd(
        actor_paths: str,
        output_path: str,
        export_materials: bool = True,
        export_animations: bool = True,
        convert_to_meters: bool = True,
    ) -> Dict[str, Any]:
        """Export actors to USD."""
        rate_error = server._check_rate_limit("export_unreal_usd")
        if rate_error:
            return rate_error
        try:
            paths = [p.strip() for p in actor_paths.split(",")]
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.unreal_adapter.create_session() as session:
                payload = await session.export_usd(
                    actor_paths=paths,
                    output_path=output_path,
                    export_materials=export_materials,
                    export_animations=export_animations,
                    convert_to_meters=convert_to_meters,
                )
                payload["success"] = True
                result = UnrealExportUsdResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealExportUsdResponse, ErrorResponse),
                    "export_unreal_usd",
                )
        except Exception as e:
            server.logger.error("Error exporting Unreal USD: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealExportUsdResponse, ErrorResponse),
                "export_unreal_usd",
            )

    @server.mcp.tool(
        name="convert_to_simready",
        description="Convert Unreal actors to NVIDIA SimReady asset format.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealConvertToSimreadyResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def convert_to_simready(
        actor_paths: str,
        output_directory: str,
        add_physics: bool = True,
        add_collision: bool = True,
        semantic_labels: str = "",
    ) -> Dict[str, Any]:
        """Convert actors to SimReady format."""
        rate_error = server._check_rate_limit("convert_to_simready")
        if rate_error:
            return rate_error
        try:
            paths = [p.strip() for p in actor_paths.split(",")]
            labels = None
            if semantic_labels:
                labels = dict(
                    pair.split("=")
                    for pair in semantic_labels.split(",")
                    if "=" in pair
                )
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.unreal_adapter.create_session() as session:
                payload = await session.convert_to_simready(
                    actor_paths=paths,
                    output_directory=output_directory,
                    add_physics=add_physics,
                    add_collision=add_collision,
                    semantic_labels=labels,
                )
                payload["success"] = True
                result = UnrealConvertToSimreadyResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealConvertToSimreadyResponse, ErrorResponse),
                    "convert_to_simready",
                )
        except Exception as e:
            server.logger.error("Error converting to SimReady: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealConvertToSimreadyResponse, ErrorResponse),
                "convert_to_simready",
            )

    @server.mcp.tool(
        name="validate_simready_asset",
        description="Validate an asset against NVIDIA SimReady requirements.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealValidateSimreadyResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def validate_simready_asset(
        asset_path: str,
        checks: str = "",
    ) -> Dict[str, Any]:
        """Validate asset against SimReady spec."""
        rate_error = server._check_rate_limit("validate_simready_asset")
        if rate_error:
            return rate_error
        try:
            check_list = (
                [c.strip() for c in checks.split(",") if c.strip()]
                if checks
                else None
            )
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.unreal_adapter.create_session() as session:
                payload = await session.validate_simready_asset(
                    asset_path=asset_path,
                    checks=check_list,
                )
                payload["success"] = True
                result = UnrealValidateSimreadyResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealValidateSimreadyResponse, ErrorResponse),
                    "validate_simready_asset",
                )
        except Exception as e:
            server.logger.error("Error validating SimReady asset: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealValidateSimreadyResponse, ErrorResponse),
                "validate_simready_asset",
            )

    @server.mcp.tool(
        name="get_unreal_interchange_info",
        description="Query available Interchange pipelines and supported formats.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealGetInterchangeInfoResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def get_unreal_interchange_info() -> Dict[str, Any]:
        """Get Interchange Framework info."""
        rate_error = server._check_rate_limit("get_unreal_interchange_info")
        if rate_error:
            return rate_error
        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.unreal_adapter.create_session() as session:
                payload = await session.get_interchange_info()
                payload["success"] = True
                result = UnrealGetInterchangeInfoResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealGetInterchangeInfoResponse, ErrorResponse),
                    "get_unreal_interchange_info",
                )
        except Exception as e:
            server.logger.error("Error getting interchange info: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealGetInterchangeInfoResponse, ErrorResponse),
                "get_unreal_interchange_info",
            )

    # ----------------------------------------------------------
    # Phase 7: Advanced Agent Tools
    # ----------------------------------------------------------

    @server.mcp.tool(
        name="batch_unreal_operations",
        description="Execute multiple Remote Control operations in one HTTP call.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealBatchOperationsResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def batch_unreal_operations(
        operations: str,
    ) -> Dict[str, Any]:
        """Batch multiple operations."""
        rate_error = server._check_rate_limit("batch_unreal_operations")
        if rate_error:
            return rate_error
        try:
            import json as _json

            ops = _json.loads(operations)
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.unreal_adapter.create_session() as session:
                payload = await session.batch_operations(operations=ops)
                payload["success"] = True
                result = UnrealBatchOperationsResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealBatchOperationsResponse, ErrorResponse),
                    "batch_unreal_operations",
                )
        except Exception as e:
            server.logger.error("Error in batch Unreal operations: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealBatchOperationsResponse, ErrorResponse),
                "batch_unreal_operations",
            )

    @server.mcp.tool(
        name="query_unreal_scene_graph",
        description="Query the Unreal scene graph hierarchy.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealQuerySceneGraphResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def query_unreal_scene_graph(
        root_path: Optional[str] = None,
        max_depth: int = 10,
        include_components: bool = False,
        class_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Query scene graph hierarchy."""
        rate_error = server._check_rate_limit("query_unreal_scene_graph")
        if rate_error:
            return rate_error
        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.unreal_adapter.create_session() as session:
                payload = await session.query_scene_graph(
                    root_path=root_path,
                    max_depth=max_depth,
                    include_components=include_components,
                    class_filter=class_filter,
                )
                payload["success"] = True
                result = UnrealQuerySceneGraphResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealQuerySceneGraphResponse, ErrorResponse),
                    "query_unreal_scene_graph",
                )
        except Exception as e:
            server.logger.error("Error querying scene graph: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealQuerySceneGraphResponse, ErrorResponse),
                "query_unreal_scene_graph",
            )

    @server.mcp.tool(
        name="analyze_unreal_scene_for_robotics",
        description="Analyze the scene for robotics use-cases (traversability, graspability, collision).",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealAnalyzeSceneForRoboticsResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def analyze_unreal_scene_for_robotics(
        analysis_types: str = "",
        actor_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyze scene for robotics."""
        rate_error = server._check_rate_limit("analyze_unreal_scene_for_robotics")
        if rate_error:
            return rate_error
        try:
            types_list = (
                [t.strip() for t in analysis_types.split(",") if t.strip()]
                if analysis_types
                else None
            )
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.unreal_adapter.create_session() as session:
                payload = await session.analyze_scene_for_robotics(
                    analysis_types=types_list,
                    actor_filter=actor_filter,
                )
                payload["success"] = True
                result = UnrealAnalyzeSceneForRoboticsResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealAnalyzeSceneForRoboticsResponse, ErrorResponse),
                    "analyze_unreal_scene_for_robotics",
                )
        except Exception as e:
            server.logger.error("Error analyzing scene for robotics: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealAnalyzeSceneForRoboticsResponse, ErrorResponse),
                "analyze_unreal_scene_for_robotics",
            )

    @server.mcp.tool(
        name="generate_unreal_procedural_scene",
        description="Generate a procedural scene (warehouse, outdoor, room, corridor).",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealGenerateProceduralSceneResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def generate_unreal_procedural_scene(
        scene_type: str,
        parameters: str = "{}",
        bounds_min: str = "",
        bounds_max: str = "",
    ) -> Dict[str, Any]:
        """Generate procedural scene."""
        rate_error = server._check_rate_limit("generate_unreal_procedural_scene")
        if rate_error:
            return rate_error
        try:
            import json as _json

            params = _json.loads(parameters) if parameters else None
            bmin = [float(v) for v in bounds_min.split(",")] if bounds_min else None
            bmax = [float(v) for v in bounds_max.split(",")] if bounds_max else None
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.unreal_adapter.create_session() as session:
                payload = await session.generate_procedural_scene(
                    scene_type=scene_type,
                    parameters=params,
                    bounds_min=bmin,
                    bounds_max=bmax,
                )
                payload["success"] = True
                result = UnrealGenerateProceduralSceneResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealGenerateProceduralSceneResponse, ErrorResponse),
                    "generate_unreal_procedural_scene",
                )
        except Exception as e:
            server.logger.error("Error generating procedural scene: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealGenerateProceduralSceneResponse, ErrorResponse),
                "generate_unreal_procedural_scene",
            )

    @server.mcp.tool(
        name="get_unreal_actor_by_semantic_label",
        description="Find actors by semantic tag or label.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealGetActorBySemanticLabelResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def get_unreal_actor_by_semantic_label(
        label: str,
        match_mode: str = "exact",
        max_results: int = 100,
    ) -> Dict[str, Any]:
        """Find actors by semantic label."""
        rate_error = server._check_rate_limit("get_unreal_actor_by_semantic_label")
        if rate_error:
            return rate_error
        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.unreal_adapter.create_session() as session:
                payload = await session.get_actor_by_semantic_label(
                    label=label,
                    match_mode=match_mode,
                    max_results=max_results,
                )
                payload["success"] = True
                result = UnrealGetActorBySemanticLabelResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealGetActorBySemanticLabelResponse, ErrorResponse),
                    "get_unreal_actor_by_semantic_label",
                )
        except Exception as e:
            server.logger.error("Error finding actors by label: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealGetActorBySemanticLabelResponse, ErrorResponse),
                "get_unreal_actor_by_semantic_label",
            )

    # ----------------------------------------------------------
    # Phase 8: Geometry & Modeling (GeometryScript)
    # ----------------------------------------------------------

    @server.mcp.tool(
        name="generate_unreal_mesh_primitive",
        description="Create a parametric mesh primitive (box, sphere, cylinder, cone, torus, capsule).",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealGenerateMeshPrimitiveResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def generate_unreal_mesh_primitive(
        primitive_type: str,
        dimensions: str = "{}",
        segments: int = 32,
        location: str = "",
        actor_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create mesh primitive."""
        rate_error = server._check_rate_limit("generate_unreal_mesh_primitive")
        if rate_error:
            return rate_error
        try:
            import json as _json

            dims = _json.loads(dimensions) if dimensions else None
            loc = [float(v) for v in location.split(",")] if location else None
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.unreal_adapter.create_session() as session:
                payload = await session.generate_mesh_primitive(
                    primitive_type=primitive_type,
                    dimensions=dims,
                    segments=segments,
                    location=loc,
                    actor_label=actor_label,
                )
                payload["success"] = True
                result = UnrealGenerateMeshPrimitiveResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealGenerateMeshPrimitiveResponse, ErrorResponse),
                    "generate_unreal_mesh_primitive",
                )
        except Exception as e:
            server.logger.error("Error generating mesh primitive: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealGenerateMeshPrimitiveResponse, ErrorResponse),
                "generate_unreal_mesh_primitive",
            )

    @server.mcp.tool(
        name="apply_unreal_mesh_boolean",
        description="Apply boolean operation (union, subtract, intersect) between two meshes.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealApplyMeshBooleanResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def apply_unreal_mesh_boolean(
        target_mesh_path: str,
        tool_mesh_path: str,
        operation: str,
    ) -> Dict[str, Any]:
        """Apply mesh boolean."""
        rate_error = server._check_rate_limit("apply_unreal_mesh_boolean")
        if rate_error:
            return rate_error
        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.unreal_adapter.create_session() as session:
                payload = await session.apply_mesh_boolean(
                    target_mesh_path=target_mesh_path,
                    tool_mesh_path=tool_mesh_path,
                    operation=operation,
                )
                payload["success"] = True
                result = UnrealApplyMeshBooleanResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealApplyMeshBooleanResponse, ErrorResponse),
                    "apply_unreal_mesh_boolean",
                )
        except Exception as e:
            server.logger.error("Error applying mesh boolean: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealApplyMeshBooleanResponse, ErrorResponse),
                "apply_unreal_mesh_boolean",
            )

    @server.mcp.tool(
        name="compute_unreal_convex_hull",
        description="Compute convex hull envelope of a mesh.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealComputeConvexHullResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def compute_unreal_convex_hull(
        mesh_path: str,
    ) -> Dict[str, Any]:
        """Compute convex hull."""
        rate_error = server._check_rate_limit("compute_unreal_convex_hull")
        if rate_error:
            return rate_error
        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.unreal_adapter.create_session() as session:
                payload = await session.compute_convex_hull(
                    mesh_path=mesh_path,
                )
                payload["success"] = True
                result = UnrealComputeConvexHullResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealComputeConvexHullResponse, ErrorResponse),
                    "compute_unreal_convex_hull",
                )
        except Exception as e:
            server.logger.error("Error computing convex hull: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealComputeConvexHullResponse, ErrorResponse),
                "compute_unreal_convex_hull",
            )

    @server.mcp.tool(
        name="decompose_unreal_convex_hull",
        description="V-HACD convex decomposition for collision geometry.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealDecomposeConvexHullResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def decompose_unreal_convex_hull(
        mesh_path: str,
        max_hulls: int = 16,
        max_vertices_per_hull: int = 32,
        min_cluster_size: int = 256,
        resolution: int = 100000,
    ) -> Dict[str, Any]:
        """V-HACD convex decomposition."""
        rate_error = server._check_rate_limit("decompose_unreal_convex_hull")
        if rate_error:
            return rate_error
        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.unreal_adapter.create_session() as session:
                payload = await session.decompose_convex_hull(
                    mesh_path=mesh_path,
                    max_hulls=max_hulls,
                    max_vertices_per_hull=max_vertices_per_hull,
                    min_cluster_size=min_cluster_size,
                    resolution=resolution,
                )
                payload["success"] = True
                result = UnrealDecomposeConvexHullResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealDecomposeConvexHullResponse, ErrorResponse),
                    "decompose_unreal_convex_hull",
                )
        except Exception as e:
            server.logger.error("Error decomposing convex hull: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealDecomposeConvexHullResponse, ErrorResponse),
                "decompose_unreal_convex_hull",
            )

    @server.mcp.tool(
        name="edit_unreal_mesh_topology",
        description="Edit mesh topology (extrude, bevel, inset, loop cut, scale_faces).",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealEditMeshTopologyResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def edit_unreal_mesh_topology(
        mesh_path: str,
        operation: str,
        face_selection: Optional[str] = None,
        edge_selection: Optional[str] = None,
        distance: Optional[float] = None,
        offset: Optional[float] = None,
        scale: str = "",
        count: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Edit mesh topology."""
        rate_error = server._check_rate_limit("edit_unreal_mesh_topology")
        if rate_error:
            return rate_error
        try:
            scale_list = [float(v) for v in scale.split(",")] if scale else None
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.unreal_adapter.create_session() as session:
                payload = await session.edit_mesh_topology(
                    mesh_path=mesh_path,
                    operation=operation,
                    face_selection=face_selection,
                    edge_selection=edge_selection,
                    distance=distance,
                    offset=offset,
                    scale=scale_list,
                    count=count,
                )
                payload["success"] = True
                result = UnrealEditMeshTopologyResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealEditMeshTopologyResponse, ErrorResponse),
                    "edit_unreal_mesh_topology",
                )
        except Exception as e:
            server.logger.error("Error editing mesh topology: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealEditMeshTopologyResponse, ErrorResponse),
                "edit_unreal_mesh_topology",
            )

    @server.mcp.tool(
        name="subdivide_unreal_mesh",
        description="Catmull-Clark / Loop / bilinear subdivision.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealSubdivideMeshResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def subdivide_unreal_mesh(
        mesh_path: str,
        level: int = 2,
        scheme: str = "catmull_clark",
    ) -> Dict[str, Any]:
        """Subdivide mesh."""
        rate_error = server._check_rate_limit("subdivide_unreal_mesh")
        if rate_error:
            return rate_error
        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.unreal_adapter.create_session() as session:
                payload = await session.subdivide_mesh(
                    mesh_path=mesh_path,
                    level=level,
                    scheme=scheme,
                )
                payload["success"] = True
                result = UnrealSubdivideMeshResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealSubdivideMeshResponse, ErrorResponse),
                    "subdivide_unreal_mesh",
                )
        except Exception as e:
            server.logger.error("Error subdividing mesh: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealSubdivideMeshResponse, ErrorResponse),
                "subdivide_unreal_mesh",
            )

    @server.mcp.tool(
        name="simplify_unreal_mesh",
        description="Simplify/decimate a mesh to reduce triangle count.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealSimplifyMeshResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def simplify_unreal_mesh(
        mesh_path: str,
        target_triangle_count: Optional[int] = None,
        target_percentage: Optional[float] = None,
        max_error: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Simplify mesh."""
        rate_error = server._check_rate_limit("simplify_unreal_mesh")
        if rate_error:
            return rate_error
        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.unreal_adapter.create_session() as session:
                payload = await session.simplify_mesh(
                    mesh_path=mesh_path,
                    target_triangle_count=target_triangle_count,
                    target_percentage=target_percentage,
                    max_error=max_error,
                )
                payload["success"] = True
                result = UnrealSimplifyMeshResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealSimplifyMeshResponse, ErrorResponse),
                    "simplify_unreal_mesh",
                )
        except Exception as e:
            server.logger.error("Error simplifying mesh: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealSimplifyMeshResponse, ErrorResponse),
                "simplify_unreal_mesh",
            )

    @server.mcp.tool(
        name="cut_unreal_mesh_plane",
        description="Cut/slice a mesh along an arbitrary plane.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealCutMeshPlaneResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def cut_unreal_mesh_plane(
        mesh_path: str,
        plane_origin: str,
        plane_normal: str,
        fill_holes: bool = True,
        keep_both_sides: bool = False,
    ) -> Dict[str, Any]:
        """Cut mesh with plane."""
        rate_error = server._check_rate_limit("cut_unreal_mesh_plane")
        if rate_error:
            return rate_error
        try:
            origin = [float(v) for v in plane_origin.split(",")]
            normal = [float(v) for v in plane_normal.split(",")]
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.unreal_adapter.create_session() as session:
                payload = await session.cut_mesh_plane(
                    mesh_path=mesh_path,
                    plane_origin=origin,
                    plane_normal=normal,
                    fill_holes=fill_holes,
                    keep_both_sides=keep_both_sides,
                )
                payload["success"] = True
                result = UnrealCutMeshPlaneResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealCutMeshPlaneResponse, ErrorResponse),
                    "cut_unreal_mesh_plane",
                )
        except Exception as e:
            server.logger.error("Error cutting mesh with plane: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealCutMeshPlaneResponse, ErrorResponse),
                "cut_unreal_mesh_plane",
            )

    @server.mcp.tool(
        name="validate_unreal_mesh",
        description="Validate mesh integrity (manifold, normals, degenerates, self-intersection).",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealValidateMeshResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def validate_unreal_mesh(
        mesh_path: str,
        checks: str = "",
    ) -> Dict[str, Any]:
        """Validate mesh integrity."""
        rate_error = server._check_rate_limit("validate_unreal_mesh")
        if rate_error:
            return rate_error
        try:
            check_list = (
                [c.strip() for c in checks.split(",") if c.strip()]
                if checks
                else None
            )
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.unreal_adapter.create_session() as session:
                payload = await session.validate_mesh(
                    mesh_path=mesh_path,
                    checks=check_list,
                )
                payload["success"] = True
                result = UnrealValidateMeshResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealValidateMeshResponse, ErrorResponse),
                    "validate_unreal_mesh",
                )
        except Exception as e:
            server.logger.error("Error validating mesh: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealValidateMeshResponse, ErrorResponse),
                "validate_unreal_mesh",
            )

    @server.mcp.tool(
        name="convert_unreal_mesh_format",
        description="Convert mesh between formats (static mesh, dynamic mesh, skeletal mesh).",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealConvertMeshFormatResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def convert_unreal_mesh_format(
        mesh_path: str,
        target_format: str,
        tessellation_options: str = "{}",
    ) -> Dict[str, Any]:
        """Convert mesh format."""
        rate_error = server._check_rate_limit("convert_unreal_mesh_format")
        if rate_error:
            return rate_error
        try:
            import json as _json

            tess_opts = (
                _json.loads(tessellation_options) if tessellation_options else None
            )
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.unreal_adapter.create_session() as session:
                payload = await session.convert_mesh_format(
                    mesh_path=mesh_path,
                    target_format=target_format,
                    tessellation_options=tess_opts,
                )
                payload["success"] = True
                result = UnrealConvertMeshFormatResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealConvertMeshFormatResponse, ErrorResponse),
                    "convert_unreal_mesh_format",
                )
        except Exception as e:
            server.logger.error("Error converting mesh format: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealConvertMeshFormatResponse, ErrorResponse),
                "convert_unreal_mesh_format",
            )

    @server.mcp.tool(
        name="remesh_unreal_mesh",
        description="Remesh a mesh (uniform, adaptive) to improve triangle quality.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealRemeshMeshResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def remesh_unreal_mesh(
        mesh_path: str,
        mode: str = "uniform",
        target_edge_length: Optional[float] = None,
        target_triangle_count: Optional[int] = None,
        smoothing_iterations: int = 3,
    ) -> Dict[str, Any]:
        """Remesh mesh."""
        rate_error = server._check_rate_limit("remesh_unreal_mesh")
        if rate_error:
            return rate_error
        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.unreal_adapter.create_session() as session:
                payload = await session.remesh_mesh(
                    mesh_path=mesh_path,
                    mode=mode,
                    target_edge_length=target_edge_length,
                    target_triangle_count=target_triangle_count,
                    smoothing_iterations=smoothing_iterations,
                )
                payload["success"] = True
                result = UnrealRemeshMeshResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealRemeshMeshResponse, ErrorResponse),
                    "remesh_unreal_mesh",
                )
        except Exception as e:
            server.logger.error("Error remeshing: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealRemeshMeshResponse, ErrorResponse),
                "remesh_unreal_mesh",
            )

    @server.mcp.tool(
        name="compute_unreal_mesh_uv",
        description="Generate or recompute UV coordinates for a mesh.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=server._tool_output_schema(
            UnrealComputeMeshUvResponse, ErrorResponse
        ),
        task=server._task_optional(),
    )
    async def compute_unreal_mesh_uv(
        mesh_path: str,
        method: str = "auto_uv",
        uv_channel: int = 0,
        island_padding: float = 2.0,
    ) -> Dict[str, Any]:
        """Compute mesh UVs."""
        rate_error = server._check_rate_limit("compute_unreal_mesh_uv")
        if rate_error:
            return rate_error
        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()
            with server.unreal_adapter.create_session() as session:
                payload = await session.compute_mesh_uv(
                    mesh_path=mesh_path,
                    method=method,
                    uv_channel=uv_channel,
                    island_padding=island_padding,
                )
                payload["success"] = True
                result = UnrealComputeMeshUvResponse(**payload).model_dump()
                return server._validate_output(
                    result,
                    (UnrealComputeMeshUvResponse, ErrorResponse),
                    "compute_unreal_mesh_uv",
                )
        except Exception as e:
            server.logger.error("Error computing mesh UVs: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealComputeMeshUvResponse, ErrorResponse),
                "compute_unreal_mesh_uv",
            )
