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
        thin: When True, only register the five essential MCP tools:
              ``unreal_health_check``, ``ping_unreal``,
              ``list_unreal_instances``, ``capture_unreal_viewport`` and
              ``execute_unreal_script``. Selected by
              ``unreal.tool_surface`` / ``simul-mcp server --unreal-tools``;
              the full set is also reachable via ``simul unreal --help``.
    """

    @server.mcp.tool(
        name="unreal_health_check",
        description="Check connectivity to the Unreal Engine Remote Control API.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def unreal_health_check() -> Dict[str, Any]:
        """
        Check connectivity to the Unreal Engine Remote Control API.

        Returns:
            Connection status or error response.
        """
        return await server._exec_backend(
            "unreal_health_check",
            server.unreal_adapter,
            "Unreal",
            UnrealHealthCheckResponse,
            lambda session: session.health_check(),
        )

    # ------------------------------------------------------------------
    # Ping / multi-instance discovery
    # ------------------------------------------------------------------

    @server.mcp.tool(
        name="ping_unreal",
        description=(
            "Pre-flight check: verify that a running Unreal Engine instance is "
            "reachable on the configured Remote Control API port. Call this before "
            "other Unreal tools to confirm connectivity and measure latency."
        ),
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def ping_unreal() -> Dict[str, Any]:
        """
        Lightweight reachability probe for the Unreal Remote Control API.

        Returns:
            Ping result with reachable status and latency, or error response.
        """
        return await server._exec_backend(
            "ping_unreal",
            server.unreal_adapter,
            "Unreal",
            UnrealPingResponse,
            lambda session: session.ping(),
        )

    @server.mcp.tool(
        name="list_unreal_instances",
        description=(
            "Discover all running Unreal Engine instances by scanning the configured "
            "port range for Remote Control API endpoints. Returns each instance's "
            "reachability, engine version, project name, and latency."
        ),
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def list_unreal_instances(
        scan: bool = True,
    ) -> Dict[str, Any]:
        """
        Scan configured port range for running Unreal Engine instances.

        Args:
            scan: When True, probe ports in the configured range concurrently.

        Returns:
            List of discovered instances or error response.
        """
        import asyncio as _asyncio

        rate_error = server._check_rate_limit("list_unreal_instances")
        if rate_error:
            return rate_error

        try:
            if not server.unreal_adapter or not server.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).model_dump()

            from ...adapters.unreal_runtime import (
                UnrealRuntimeSession,
                _passphrase_to_md5,
            )

            cfg = server.settings.unreal
            host = cfg.host
            active_port = cfg.port
            instances: List[Any] = []

            if scan:
                probe_tasks = []
                ports = list(range(cfg.scan_port_start, cfg.scan_port_end))
                # Always include the active port even if outside scan range
                if active_port not in ports:
                    ports.insert(0, active_port)
                # If a passphrase is configured, every probe must carry the
                # header — otherwise a passphrase-enforcing editor returns
                # 401 and discovery silently reports it as unreachable.
                discovery_md5 = _passphrase_to_md5(cfg.passphrase)
                for port in ports:
                    probe_tasks.append(
                        UnrealRuntimeSession.probe_port(
                            host,
                            port,
                            timeout=cfg.ping_timeout,
                            passphrase_md5=discovery_md5,
                        )
                    )
                results = await _asyncio.gather(*probe_tasks, return_exceptions=True)

                for port, probe_result in zip(ports, results):
                    if isinstance(probe_result, Exception):
                        continue
                    name = "default" if port == active_port else f"unreal-{port}"
                    instances.append(
                        UnrealInstanceInfo(
                            name=name,
                            host=host,
                            port=port,
                            reachable=probe_result.get("reachable", False),
                            active=port == active_port,
                            engine_version=probe_result.get("engine_version"),
                            project_name=probe_result.get("project_name"),
                            loaded_map=None,
                            latency_ms=probe_result.get("latency_ms"),
                        )
                    )
            else:
                with server.unreal_adapter.create_session() as session:
                    payload = await session.ping()
                    instances.append(
                        UnrealInstanceInfo(
                            name="default",
                            host=host,
                            port=active_port,
                            reachable=payload.get("reachable", False),
                            active=True,
                            latency_ms=payload.get("latency_ms"),
                        )
                    )

            reachable = [i for i in instances if i.reachable]
            active_name = next((i.name for i in instances if i.active), None)
            result = UnrealListInstancesResponse(
                success=True,
                instances=instances,
                active_instance=active_name,
                total_discovered=len(reachable),
            ).model_dump()
            return server._validate_output(
                result,
                (UnrealListInstancesResponse, ErrorResponse),
                "list_unreal_instances",
            )

        except Exception as e:
            server.logger.error("Error listing Unreal instances: %s", e)
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (UnrealListInstancesResponse, ErrorResponse),
                "list_unreal_instances",
            )

    # ------------------------------------------------------------------
    # Viewport capture
    # ------------------------------------------------------------------

    @server.mcp.tool(
        name="capture_unreal_viewport",
        description=(
            "Capture a viewport screenshot via HighResScreenshot and return "
            "its path on the editor host. Pass inline=true to also receive "
            "base64 image data, included only for small captures."
        ),
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def capture_unreal_viewport(
        resolution_x: int = 1920,
        resolution_y: int = 1080,
        format: str = "png",
        inline: bool = False,
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
            ).model_dump()

        return await server._exec_backend(
            "capture_unreal_viewport",
            server.unreal_adapter,
            "Unreal",
            UnrealCaptureViewportResponse,
            lambda session: session.capture_viewport(
                resolution_x=resolution_x,
                resolution_y=resolution_y,
                format=format,
                inline=inline,
            ),
        )

    @server._script_tool(
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

    # -- Thin mode ends here: health check, ping, instance listing,
    #    viewport capture and script execution are registered above.
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
        output_schema=None,
        task=server._task_optional(),
    )
    async def get_unreal_engine_info() -> Dict[str, Any]:
        """
        Get Unreal Engine runtime information.

        Returns:
            Engine info or error response.
        """
        return await server._exec_backend(
            "get_unreal_engine_info",
            server.unreal_adapter,
            "Unreal",
            UnrealEngineInfoResponse,
            lambda session: session.get_engine_info(),
        )

    @server.mcp.tool(
        name="get_unreal_loaded_map",
        description="Get the currently loaded persistent level path.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def get_unreal_loaded_map() -> Dict[str, Any]:
        """
        Get the currently loaded persistent level path.

        Returns:
            Loaded map path or error response.
        """
        return await server._exec_backend(
            "get_unreal_loaded_map",
            server.unreal_adapter,
            "Unreal",
            UnrealLoadedMapResponse,
            lambda session: session.get_loaded_map(),
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
        output_schema=None,
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
        return await server._exec_backend(
            "list_unreal_actors",
            server.unreal_adapter,
            "Unreal",
            UnrealListActorsResponse,
            lambda session: session.list_actors(
                class_filter=class_filter or None,
                tag_filter=tag_filter or None,
                max_results=max_results,
            ),
        )

    @server.mcp.tool(
        name="get_unreal_actor_info",
        description="Get detailed information about a specific actor including transform, components, and tags.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
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
        return await server._exec_backend(
            "get_unreal_actor_info",
            server.unreal_adapter,
            "Unreal",
            UnrealGetActorInfoResponse,
            lambda session: session.get_actor_info(actor_path),
        )

    @server.mcp.tool(
        name="search_unreal_assets",
        description="Search the Unreal Asset Registry by name, class, or package path.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
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

        def _call(session):
            # Parsing stays inside the envelope: malformed input must
            # return the error payload it always has, not escape as an
            # unhandled exception.
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
            return session.search_assets(
                query=query,
                class_names=parsed_classes,
                package_paths=parsed_paths,
                max_results=max_results,
            )

        return await server._exec_backend(
            "search_unreal_assets",
            server.unreal_adapter,
            "Unreal",
            UnrealSearchAssetsResponse,
            _call,
        )

    @server.mcp.tool(
        name="describe_unreal_object",
        description="Get full property and function metadata for any UObject by path.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
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
        return await server._exec_backend(
            "describe_unreal_object",
            server.unreal_adapter,
            "Unreal",
            UnrealDescribeObjectResponse,
            lambda session: session.describe_object(object_path),
        )

    @server.mcp.tool(
        name="get_unreal_actor_thumbnail",
        description="Get a thumbnail image for an Unreal asset.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=False,
            open_world=True,
        ),
        output_schema=None,
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
        return await server._exec_backend(
            "get_unreal_actor_thumbnail",
            server.unreal_adapter,
            "Unreal",
            UnrealGetThumbnailResponse,
            lambda session: session.get_actor_thumbnail(
                asset_path=asset_path, width=width, height=height
            ),
        )

    @server.mcp.tool(
        name="summarize_unreal_scene",
        description="Generate an LLM-friendly digest of the current Unreal scene.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def summarize_unreal_scene() -> Dict[str, Any]:
        """
        Generate an LLM-friendly scene digest.

        Returns:
            Scene summary or error response.
        """
        return await server._exec_backend(
            "summarize_unreal_scene",
            server.unreal_adapter,
            "Unreal",
            UnrealSceneSummaryResponse,
            lambda session: session.summarize_scene(),
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
        output_schema=None,
        task=server._task_optional(),
    )
    async def get_unreal_viewport_info() -> Dict[str, Any]:
        """
        Get viewport camera and render settings.

        Returns:
            Viewport info or error response.
        """
        return await server._exec_backend(
            "get_unreal_viewport_info",
            server.unreal_adapter,
            "Unreal",
            UnrealViewportInfoResponse,
            lambda session: session.get_viewport_info(),
        )

    @server.mcp.tool(
        name="set_unreal_camera_view",
        description="Set the editor viewport camera position and rotation.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
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
        return await server._exec_backend(
            "set_unreal_camera_view",
            server.unreal_adapter,
            "Unreal",
            UnrealSetCameraViewResponse,
            lambda session: session.set_camera_view(
                location=(location_x, location_y, location_z),
                rotation=(rotation_pitch, rotation_yaw, rotation_roll),
                fov=fov,
            ),
        )

    @server.mcp.tool(
        name="focus_unreal_on_actor",
        description="Focus the editor viewport camera on a specific actor.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=None,
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
        return await server._exec_backend(
            "focus_unreal_on_actor",
            server.unreal_adapter,
            "Unreal",
            UnrealFocusActorResponse,
            lambda session: session.focus_on_actor(
                actor_path=actor_path, distance=distance
            ),
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
        output_schema=None,
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
        return await server._exec_backend(
            "spawn_unreal_actor",
            server.unreal_adapter,
            "Unreal",
            UnrealSpawnActorResponse,
            lambda session: session.spawn_actor(
                asset_path=asset_path,
                location=(location_x, location_y, location_z),
                rotation=(rotation_pitch, rotation_yaw, rotation_roll),
                label=label or None,
            ),
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
        output_schema=None,
        task=server._task_optional(),
    )
    async def delete_unreal_actor(
        actor_path: str,
    ) -> Dict[str, Any]:
        """Delete an actor from the level."""
        return await server._exec_backend(
            "delete_unreal_actor",
            server.unreal_adapter,
            "Unreal",
            UnrealDeleteActorResponse,
            lambda session: session.delete_actor(actor_path=actor_path),
        )

    @server.mcp.tool(
        name="set_unreal_actor_transform",
        description="Set an actor's location, rotation, and scale.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
            destructive=True,
        ),
        output_schema=None,
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
        return await server._exec_backend(
            "set_unreal_actor_transform",
            server.unreal_adapter,
            "Unreal",
            UnrealSetActorTransformResponse,
            lambda session: session.set_actor_transform(
                actor_path=actor_path,
                location=(location_x, location_y, location_z),
                rotation=(rotation_pitch, rotation_yaw, rotation_roll),
                scale=(scale_x, scale_y, scale_z),
            ),
        )

    @server.mcp.tool(
        name="set_unreal_actor_property",
        description="Set a property on an Unreal actor by name and JSON value.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
            destructive=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def set_unreal_actor_property(
        actor_path: str,
        property_name: str,
        property_value: str,
        generate_transaction: bool = True,
    ) -> Dict[str, Any]:
        """Set a property on an actor."""
        return await server._exec_backend(
            "set_unreal_actor_property",
            server.unreal_adapter,
            "Unreal",
            UnrealSetActorPropertyResponse,
            lambda session: session.set_actor_property(
                actor_path=actor_path,
                property_name=property_name,
                property_value=property_value,
                generate_transaction=generate_transaction,
            ),
        )

    @server.mcp.tool(
        name="call_unreal_actor_function",
        description="Call a BlueprintCallable UFUNCTION on an actor.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
            destructive=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def call_unreal_actor_function(
        actor_path: str,
        function_name: str,
        parameters: str = "",
    ) -> Dict[str, Any]:
        """Call a UFUNCTION on an actor."""
        return await server._exec_backend(
            "call_unreal_actor_function",
            server.unreal_adapter,
            "Unreal",
            UnrealCallActorFunctionResponse,
            lambda session: session.call_actor_function(
                actor_path=actor_path,
                function_name=function_name,
                parameters=parameters or None,
            ),
        )

    @server.mcp.tool(
        name="set_unreal_actor_parent",
        description="Attach an actor to a parent actor or detach it.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
            destructive=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def set_unreal_actor_parent(
        actor_path: str,
        parent_path: str = "",
    ) -> Dict[str, Any]:
        """Attach or detach an actor."""
        return await server._exec_backend(
            "set_unreal_actor_parent",
            server.unreal_adapter,
            "Unreal",
            UnrealSetActorParentResponse,
            lambda session: session.set_actor_parent(
                actor_path=actor_path,
                parent_path=parent_path or None,
            ),
        )

    @server.mcp.tool(
        name="add_unreal_component",
        description="Add a component to an Unreal actor.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def add_unreal_component(
        actor_path: str,
        component_class: str,
        component_name: str = "",
    ) -> Dict[str, Any]:
        """Add a component to an actor."""
        return await server._exec_backend(
            "add_unreal_component",
            server.unreal_adapter,
            "Unreal",
            UnrealAddComponentResponse,
            lambda session: session.add_component(
                actor_path=actor_path,
                component_class=component_class,
                component_name=component_name or None,
            ),
        )

    @server.mcp.tool(
        name="set_unreal_actor_visibility",
        description="Set actor visibility in the Unreal level.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
            destructive=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def set_unreal_actor_visibility(
        actor_path: str,
        visible: bool = True,
        propagate: bool = True,
    ) -> Dict[str, Any]:
        """Set actor visibility."""
        return await server._exec_backend(
            "set_unreal_actor_visibility",
            server.unreal_adapter,
            "Unreal",
            UnrealSetActorVisibilityResponse,
            lambda session: session.set_actor_visibility(
                actor_path=actor_path,
                visible=visible,
                propagate=propagate,
            ),
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
        output_schema=None,
        task=server._task_optional(),
    )
    async def get_unreal_material_info(
        material_path: str,
    ) -> Dict[str, Any]:
        """Get material info."""
        return await server._exec_backend(
            "get_unreal_material_info",
            server.unreal_adapter,
            "Unreal",
            UnrealGetMaterialInfoResponse,
            lambda session: session.get_material_info(
                material_path=material_path,
            ),
        )

    @server.mcp.tool(
        name="set_unreal_material_params",
        description="Set scalar/vector/texture parameters on a Material Instance.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
            destructive=True,
        ),
        output_schema=None,
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

        def _call(session):
            # Parsing stays inside the envelope: malformed input must
            # return the error payload it always has, not escape as an
            # unhandled exception.
            scalar_params = (
                json_lib.loads(scalar_params_json) if scalar_params_json else None
            )
            vector_params = (
                json_lib.loads(vector_params_json) if vector_params_json else None
            )
            texture_params = (
                json_lib.loads(texture_params_json) if texture_params_json else None
            )
            return session.set_material_params(
                material_path=material_path,
                scalar_params=scalar_params,
                vector_params=vector_params,
                texture_params=texture_params,
            )

        return await server._exec_backend(
            "set_unreal_material_params",
            server.unreal_adapter,
            "Unreal",
            UnrealSetMaterialParamsResponse,
            _call,
        )

    @server.mcp.tool(
        name="create_unreal_material_instance",
        description="Create a Material Instance Constant from a parent material.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def create_unreal_material_instance(
        parent_path: str,
        instance_name: str,
        save_path: str = "",
    ) -> Dict[str, Any]:
        """Create a material instance."""
        return await server._exec_backend(
            "create_unreal_material_instance",
            server.unreal_adapter,
            "Unreal",
            UnrealCreateMaterialInstanceResponse,
            lambda session: session.create_material_instance(
                parent_path=parent_path,
                instance_name=instance_name,
                save_path=save_path,
            ),
        )

    @server.mcp.tool(
        name="assign_unreal_material",
        description="Assign a material to a mesh component's material slot.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
            destructive=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def assign_unreal_material(
        actor_path: str,
        material_path: str,
        slot_index: int = 0,
    ) -> Dict[str, Any]:
        """Assign material to actor."""
        return await server._exec_backend(
            "assign_unreal_material",
            server.unreal_adapter,
            "Unreal",
            UnrealAssignMaterialResponse,
            lambda session: session.assign_material(
                actor_path=actor_path,
                material_path=material_path,
                slot_index=slot_index,
            ),
        )

    @server.mcp.tool(
        name="set_unreal_light_params",
        description="Set light component parameters (intensity, color, temperature, shadows).",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
            destructive=True,
        ),
        output_schema=None,
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
        return await server._exec_backend(
            "set_unreal_light_params",
            server.unreal_adapter,
            "Unreal",
            UnrealSetLightParamsResponse,
            lambda session: session.set_light_params(
                actor_path=actor_path,
                intensity=intensity,
                color_r=color_r,
                color_g=color_g,
                color_b=color_b,
                temperature=temperature,
                use_temperature=use_temperature,
                attenuation_radius=attenuation_radius,
                cast_shadows=cast_shadows,
            ),
        )

    @server.mcp.tool(
        name="set_unreal_render_settings",
        description="Set rendering or post-process settings via console command.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
            destructive=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def set_unreal_render_settings(
        setting_name: str,
        setting_value: str,
    ) -> Dict[str, Any]:
        """Set render settings."""
        return await server._exec_backend(
            "set_unreal_render_settings",
            server.unreal_adapter,
            "Unreal",
            UnrealSetRenderSettingsResponse,
            lambda session: session.set_render_settings(
                setting_name=setting_name,
                setting_value=setting_value,
            ),
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
        output_schema=None,
        task=server._task_optional(),
    )
    async def control_unreal_simulation(
        action: str,
    ) -> Dict[str, Any]:
        """Control PIE session."""
        return await server._exec_backend(
            "control_unreal_simulation",
            server.unreal_adapter,
            "Unreal",
            UnrealControlSimulationResponse,
            lambda session: session.control_simulation(action=action),
        )

    @server.mcp.tool(
        name="get_unreal_simulation_status",
        description="Get current Play-In-Editor simulation status (playing, paused, stopped).",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def get_unreal_simulation_status() -> Dict[str, Any]:
        """Get PIE simulation status."""
        return await server._exec_backend(
            "get_unreal_simulation_status",
            server.unreal_adapter,
            "Unreal",
            UnrealGetSimulationStatusResponse,
            lambda session: session.get_simulation_status(),
        )

    @server.mcp.tool(
        name="enable_unreal_physics",
        description="Enable or disable physics simulation on an actor.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
            destructive=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def enable_unreal_physics(
        actor_path: str,
        enable: bool = True,
        simulate_physics: bool = True,
    ) -> Dict[str, Any]:
        """Enable physics on actor."""
        return await server._exec_backend(
            "enable_unreal_physics",
            server.unreal_adapter,
            "Unreal",
            UnrealEnablePhysicsResponse,
            lambda session: session.enable_physics(
                actor_path=actor_path,
                enable=enable,
                simulate_physics=simulate_physics,
            ),
        )

    @server.mcp.tool(
        name="set_unreal_collision",
        description="Set collision presets and enable/disable collision on an actor.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
            destructive=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def set_unreal_collision(
        actor_path: str,
        collision_preset: str = "",
        collision_enabled: bool = True,
    ) -> Dict[str, Any]:
        """Set collision configuration."""
        return await server._exec_backend(
            "set_unreal_collision",
            server.unreal_adapter,
            "Unreal",
            UnrealSetCollisionResponse,
            lambda session: session.set_collision(
                actor_path=actor_path,
                collision_preset=collision_preset,
                collision_enabled=collision_enabled,
            ),
        )

    @server.mcp.tool(
        name="apply_unreal_force",
        description="Apply a force or impulse to an actor's physics body.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=None,
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
        return await server._exec_backend(
            "apply_unreal_force",
            server.unreal_adapter,
            "Unreal",
            UnrealApplyForceResponse,
            lambda session: session.apply_force(
                actor_path=actor_path,
                force_x=force_x,
                force_y=force_y,
                force_z=force_z,
                is_impulse=is_impulse,
                location_x=location_x,
                location_y=location_y,
                location_z=location_z,
            ),
        )

    @server.mcp.tool(
        name="set_unreal_physics_params",
        description="Set physics body parameters (mass, damping, gravity) on an actor.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=True,
            open_world=True,
            destructive=True,
        ),
        output_schema=None,
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
        return await server._exec_backend(
            "set_unreal_physics_params",
            server.unreal_adapter,
            "Unreal",
            UnrealSetPhysicsParamsResponse,
            lambda session: session.set_physics_params(
                actor_path=actor_path,
                mass=mass,
                linear_damping=linear_damping,
                angular_damping=angular_damping,
                enable_gravity=enable_gravity,
            ),
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
        output_schema=None,
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
        return await server._exec_backend(
            "import_unreal_usd",
            server.unreal_adapter,
            "Unreal",
            UnrealImportUsdResponse,
            lambda session: session.import_usd(
                usd_path=usd_path,
                target_path=target_path,
                import_animations=import_animations,
                import_materials=import_materials,
                scale_factor=scale_factor,
            ),
        )

    @server.mcp.tool(
        name="export_unreal_usd",
        description="Export Unreal actors to USD via Interchange Framework.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
            destructive=True,
        ),
        output_schema=None,
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

        def _call(session):
            # Parsing stays inside the envelope: malformed input must
            # return the error payload it always has, not escape as an
            # unhandled exception.
            paths = [p.strip() for p in actor_paths.split(",")]
            return session.export_usd(
                actor_paths=paths,
                output_path=output_path,
                export_materials=export_materials,
                export_animations=export_animations,
                convert_to_meters=convert_to_meters,
            )

        return await server._exec_backend(
            "export_unreal_usd",
            server.unreal_adapter,
            "Unreal",
            UnrealExportUsdResponse,
            _call,
        )

    @server.mcp.tool(
        name="convert_to_simready",
        description="Convert Unreal actors to NVIDIA SimReady asset format.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
            destructive=True,
        ),
        output_schema=None,
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

        def _call(session):
            # Parsing stays inside the envelope: malformed input must
            # return the error payload it always has, not escape as an
            # unhandled exception.
            paths = [p.strip() for p in actor_paths.split(",")]
            labels = None
            if semantic_labels:
                labels = dict(
                    pair.split("=")
                    for pair in semantic_labels.split(",")
                    if "=" in pair
                )
            return session.convert_to_simready(
                actor_paths=paths,
                output_directory=output_directory,
                add_physics=add_physics,
                add_collision=add_collision,
                semantic_labels=labels,
            )

        return await server._exec_backend(
            "convert_to_simready",
            server.unreal_adapter,
            "Unreal",
            UnrealConvertToSimreadyResponse,
            _call,
        )

    @server.mcp.tool(
        name="validate_simready_asset",
        description="Validate an Unreal asset against NVIDIA SimReady requirements.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def validate_simready_asset(
        asset_path: str,
        checks: str = "",
    ) -> Dict[str, Any]:
        """Validate asset against SimReady spec."""

        def _call(session):
            # Parsing stays inside the envelope: malformed input must
            # return the error payload it always has, not escape as an
            # unhandled exception.
            check_list = (
                [c.strip() for c in checks.split(",") if c.strip()] if checks else None
            )
            return session.validate_simready_asset(
                asset_path=asset_path,
                checks=check_list,
            )

        return await server._exec_backend(
            "validate_simready_asset",
            server.unreal_adapter,
            "Unreal",
            UnrealValidateSimreadyResponse,
            _call,
        )

    @server.mcp.tool(
        name="get_unreal_interchange_info",
        description="Query available Interchange pipelines and supported formats.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def get_unreal_interchange_info() -> Dict[str, Any]:
        """Get Interchange Framework info."""
        return await server._exec_backend(
            "get_unreal_interchange_info",
            server.unreal_adapter,
            "Unreal",
            UnrealGetInterchangeInfoResponse,
            lambda session: session.get_interchange_info(),
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
            destructive=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def batch_unreal_operations(
        operations: str,
    ) -> Dict[str, Any]:
        """Batch multiple operations."""

        def _call(session):
            # Parsing stays inside the envelope: malformed input must
            # return the error payload it always has, not escape as an
            # unhandled exception.
            import json as _json

            ops = _json.loads(operations)
            return session.batch_operations(operations=ops)

        return await server._exec_backend(
            "batch_unreal_operations",
            server.unreal_adapter,
            "Unreal",
            UnrealBatchOperationsResponse,
            _call,
        )

    @server.mcp.tool(
        name="query_unreal_scene_graph",
        description="Query the Unreal scene graph hierarchy.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def query_unreal_scene_graph(
        root_path: Optional[str] = None,
        max_depth: int = 10,
        include_components: bool = False,
        class_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Query scene graph hierarchy."""
        return await server._exec_backend(
            "query_unreal_scene_graph",
            server.unreal_adapter,
            "Unreal",
            UnrealQuerySceneGraphResponse,
            lambda session: session.query_scene_graph(
                root_path=root_path,
                max_depth=max_depth,
                include_components=include_components,
                class_filter=class_filter,
            ),
        )

    @server.mcp.tool(
        name="analyze_unreal_scene_for_robotics",
        description="Analyze the scene for robotics use-cases (traversability, graspability, collision).",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def analyze_unreal_scene_for_robotics(
        analysis_types: str = "",
        actor_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyze scene for robotics."""

        def _call(session):
            # Parsing stays inside the envelope: malformed input must
            # return the error payload it always has, not escape as an
            # unhandled exception.
            types_list = (
                [t.strip() for t in analysis_types.split(",") if t.strip()]
                if analysis_types
                else None
            )
            return session.analyze_scene_for_robotics(
                analysis_types=types_list,
                actor_filter=actor_filter,
            )

        return await server._exec_backend(
            "analyze_unreal_scene_for_robotics",
            server.unreal_adapter,
            "Unreal",
            UnrealAnalyzeSceneForRoboticsResponse,
            _call,
        )

    @server.mcp.tool(
        name="generate_unreal_procedural_scene",
        description="Generate a procedural scene (warehouse, outdoor, room, corridor).",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def generate_unreal_procedural_scene(
        scene_type: str,
        parameters: str = "{}",
        bounds_min: str = "",
        bounds_max: str = "",
    ) -> Dict[str, Any]:
        """Generate procedural scene."""

        def _call(session):
            # Parsing stays inside the envelope: malformed input must
            # return the error payload it always has, not escape as an
            # unhandled exception.
            import json as _json

            params = _json.loads(parameters) if parameters else None
            bmin = [float(v) for v in bounds_min.split(",")] if bounds_min else None
            bmax = [float(v) for v in bounds_max.split(",")] if bounds_max else None
            return session.generate_procedural_scene(
                scene_type=scene_type,
                parameters=params,
                bounds_min=bmin,
                bounds_max=bmax,
            )

        return await server._exec_backend(
            "generate_unreal_procedural_scene",
            server.unreal_adapter,
            "Unreal",
            UnrealGenerateProceduralSceneResponse,
            _call,
        )

    @server.mcp.tool(
        name="get_unreal_actor_by_semantic_label",
        description="Find actors by semantic tag or label.",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def get_unreal_actor_by_semantic_label(
        label: str,
        match_mode: str = "exact",
        max_results: int = 100,
    ) -> Dict[str, Any]:
        """Find actors by semantic label."""
        return await server._exec_backend(
            "get_unreal_actor_by_semantic_label",
            server.unreal_adapter,
            "Unreal",
            UnrealGetActorBySemanticLabelResponse,
            lambda session: session.get_actor_by_semantic_label(
                label=label,
                match_mode=match_mode,
                max_results=max_results,
            ),
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
        output_schema=None,
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

        def _call(session):
            # Parsing stays inside the envelope: malformed input must
            # return the error payload it always has, not escape as an
            # unhandled exception.
            import json as _json

            dims = _json.loads(dimensions) if dimensions else None
            loc = [float(v) for v in location.split(",")] if location else None
            return session.generate_mesh_primitive(
                primitive_type=primitive_type,
                dimensions=dims,
                segments=segments,
                location=loc,
                actor_label=actor_label,
            )

        return await server._exec_backend(
            "generate_unreal_mesh_primitive",
            server.unreal_adapter,
            "Unreal",
            UnrealGenerateMeshPrimitiveResponse,
            _call,
        )

    @server.mcp.tool(
        name="apply_unreal_mesh_boolean",
        description="Apply boolean operation (union, subtract, intersect) between two meshes.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
            destructive=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def apply_unreal_mesh_boolean(
        target_mesh_path: str,
        tool_mesh_path: str,
        operation: str,
    ) -> Dict[str, Any]:
        """Apply mesh boolean."""
        return await server._exec_backend(
            "apply_unreal_mesh_boolean",
            server.unreal_adapter,
            "Unreal",
            UnrealApplyMeshBooleanResponse,
            lambda session: session.apply_mesh_boolean(
                target_mesh_path=target_mesh_path,
                tool_mesh_path=tool_mesh_path,
                operation=operation,
            ),
        )

    @server.mcp.tool(
        name="compute_unreal_convex_hull",
        description="Compute convex hull envelope of a mesh.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def compute_unreal_convex_hull(
        mesh_path: str,
    ) -> Dict[str, Any]:
        """Compute convex hull."""
        return await server._exec_backend(
            "compute_unreal_convex_hull",
            server.unreal_adapter,
            "Unreal",
            UnrealComputeConvexHullResponse,
            lambda session: session.compute_convex_hull(
                mesh_path=mesh_path,
            ),
        )

    @server.mcp.tool(
        name="decompose_unreal_convex_hull",
        description="V-HACD convex decomposition for collision geometry.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
            destructive=True,
        ),
        output_schema=None,
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
        return await server._exec_backend(
            "decompose_unreal_convex_hull",
            server.unreal_adapter,
            "Unreal",
            UnrealDecomposeConvexHullResponse,
            lambda session: session.decompose_convex_hull(
                mesh_path=mesh_path,
                max_hulls=max_hulls,
                max_vertices_per_hull=max_vertices_per_hull,
                min_cluster_size=min_cluster_size,
                resolution=resolution,
            ),
        )

    @server.mcp.tool(
        name="edit_unreal_mesh_topology",
        description="Edit mesh topology (extrude, bevel, inset, loop cut, scale_faces).",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
            destructive=True,
        ),
        output_schema=None,
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

        def _call(session):
            # Parsing stays inside the envelope: malformed input must
            # return the error payload it always has, not escape as an
            # unhandled exception.
            scale_list = [float(v) for v in scale.split(",")] if scale else None
            return session.edit_mesh_topology(
                mesh_path=mesh_path,
                operation=operation,
                face_selection=face_selection,
                edge_selection=edge_selection,
                distance=distance,
                offset=offset,
                scale=scale_list,
                count=count,
            )

        return await server._exec_backend(
            "edit_unreal_mesh_topology",
            server.unreal_adapter,
            "Unreal",
            UnrealEditMeshTopologyResponse,
            _call,
        )

    @server.mcp.tool(
        name="subdivide_unreal_mesh",
        description="Catmull-Clark / Loop / bilinear subdivision.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
            destructive=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def subdivide_unreal_mesh(
        mesh_path: str,
        level: int = 2,
        scheme: str = "catmull_clark",
    ) -> Dict[str, Any]:
        """Subdivide mesh."""
        return await server._exec_backend(
            "subdivide_unreal_mesh",
            server.unreal_adapter,
            "Unreal",
            UnrealSubdivideMeshResponse,
            lambda session: session.subdivide_mesh(
                mesh_path=mesh_path,
                level=level,
                scheme=scheme,
            ),
        )

    @server.mcp.tool(
        name="simplify_unreal_mesh",
        description="Simplify/decimate a mesh to reduce triangle count.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
            destructive=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def simplify_unreal_mesh(
        mesh_path: str,
        target_triangle_count: Optional[int] = None,
        target_percentage: Optional[float] = None,
        max_error: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Simplify mesh."""
        return await server._exec_backend(
            "simplify_unreal_mesh",
            server.unreal_adapter,
            "Unreal",
            UnrealSimplifyMeshResponse,
            lambda session: session.simplify_mesh(
                mesh_path=mesh_path,
                target_triangle_count=target_triangle_count,
                target_percentage=target_percentage,
                max_error=max_error,
            ),
        )

    @server.mcp.tool(
        name="cut_unreal_mesh_plane",
        description="Cut/slice a mesh along an arbitrary plane.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
            destructive=True,
        ),
        output_schema=None,
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

        def _call(session):
            # Parsing stays inside the envelope: malformed input must
            # return the error payload it always has, not escape as an
            # unhandled exception.
            origin = [float(v) for v in plane_origin.split(",")]
            normal = [float(v) for v in plane_normal.split(",")]
            return session.cut_mesh_plane(
                mesh_path=mesh_path,
                plane_origin=origin,
                plane_normal=normal,
                fill_holes=fill_holes,
                keep_both_sides=keep_both_sides,
            )

        return await server._exec_backend(
            "cut_unreal_mesh_plane",
            server.unreal_adapter,
            "Unreal",
            UnrealCutMeshPlaneResponse,
            _call,
        )

    @server.mcp.tool(
        name="validate_unreal_mesh",
        description="Validate mesh integrity (manifold, normals, degenerates, self-intersection).",
        annotations=server._tool_annotations(
            read_only=True,
            idempotent=True,
            open_world=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def validate_unreal_mesh(
        mesh_path: str,
        checks: str = "",
    ) -> Dict[str, Any]:
        """Validate mesh integrity."""

        def _call(session):
            # Parsing stays inside the envelope: malformed input must
            # return the error payload it always has, not escape as an
            # unhandled exception.
            check_list = (
                [c.strip() for c in checks.split(",") if c.strip()] if checks else None
            )
            return session.validate_mesh(
                mesh_path=mesh_path,
                checks=check_list,
            )

        return await server._exec_backend(
            "validate_unreal_mesh",
            server.unreal_adapter,
            "Unreal",
            UnrealValidateMeshResponse,
            _call,
        )

    @server.mcp.tool(
        name="convert_unreal_mesh_format",
        description="Convert mesh between formats (static mesh, dynamic mesh, skeletal mesh).",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
            destructive=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def convert_unreal_mesh_format(
        mesh_path: str,
        target_format: str,
        tessellation_options: str = "{}",
    ) -> Dict[str, Any]:
        """Convert mesh format."""

        def _call(session):
            # Parsing stays inside the envelope: malformed input must
            # return the error payload it always has, not escape as an
            # unhandled exception.
            import json as _json

            tess_opts = (
                _json.loads(tessellation_options) if tessellation_options else None
            )
            return session.convert_mesh_format(
                mesh_path=mesh_path,
                target_format=target_format,
                tessellation_options=tess_opts,
            )

        return await server._exec_backend(
            "convert_unreal_mesh_format",
            server.unreal_adapter,
            "Unreal",
            UnrealConvertMeshFormatResponse,
            _call,
        )

    @server.mcp.tool(
        name="remesh_unreal_mesh",
        description="Remesh a mesh (uniform, adaptive) to improve triangle quality.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
            destructive=True,
        ),
        output_schema=None,
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
        return await server._exec_backend(
            "remesh_unreal_mesh",
            server.unreal_adapter,
            "Unreal",
            UnrealRemeshMeshResponse,
            lambda session: session.remesh_mesh(
                mesh_path=mesh_path,
                mode=mode,
                target_edge_length=target_edge_length,
                target_triangle_count=target_triangle_count,
                smoothing_iterations=smoothing_iterations,
            ),
        )

    @server.mcp.tool(
        name="compute_unreal_mesh_uv",
        description="Generate or recompute UV coordinates for a mesh.",
        annotations=server._tool_annotations(
            read_only=False,
            idempotent=False,
            open_world=True,
            destructive=True,
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def compute_unreal_mesh_uv(
        mesh_path: str,
        method: str = "auto_uv",
        uv_channel: int = 0,
        island_padding: float = 2.0,
    ) -> Dict[str, Any]:
        """Compute mesh UVs."""
        return await server._exec_backend(
            "compute_unreal_mesh_uv",
            server.unreal_adapter,
            "Unreal",
            UnrealComputeMeshUvResponse,
            lambda session: session.compute_mesh_uv(
                mesh_path=mesh_path,
                method=method,
                uv_channel=uv_channel,
                island_padding=island_padding,
            ),
        )
