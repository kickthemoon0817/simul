"""
USD headless file operation tool registration for Simul MCP Server.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..result_budget import apply_result_budget
from ..schemas.common import BoundingBox, ErrorResponse, Transform
from ..schemas.usd import (
    BBoxRequest,
    BBoxResponse,
    MeshInfo,
    MeshInfoRequest,
    PrimActionResponse,
    PrimCreateRequest,
    PrimDeleteRequest,
    PrimInfo,
    PrimInfoRequest,
    PrimSearchRequest,
    PrimSearchResponse,
    PrimUpdateRequest,
    SceneSummaryRequest,
    SceneSummaryResponse,
    StageInfo,
    USDFileInfo,
    USDFileRequest,
    USDValidateRequest,
)

if TYPE_CHECKING:
    from ..server import SimulMCPServer


def register_usd_tools(server: "SimulMCPServer") -> None:
    """Register USD-related tools."""

    @server.mcp.tool(
        name="load_usd_file",
        description=(
            "Load a USD file and return stage information. "
            "Operates on local USD files via the headless adapter — "
            "no running application required. For live Isaac Sim stages, "
            "use the isaac_* prefixed tools instead."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=True
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def load_usd_file(file_path: str) -> Dict[str, Any]:
        """
        Load a USD file and return stage information.

        Args:
            file_path: Path to USD file

        Returns:
            Stage information or error
        """
        rate_error = server._check_rate_limit("load_usd_file")
        if rate_error:
            return rate_error

        input_data = server._validate_input(USDFileRequest, file_path=file_path)
        if isinstance(input_data, dict):
            return input_data

        if not server._is_path_allowed(input_data.file_path):
            return ErrorResponse(
                error="File path is not allowed by sandbox policy",
                error_type="SandboxError",
                details={"file_path": input_data.file_path},
            ).model_dump()

        try:
            adapter = server.headless_adapter
            if not adapter:
                return ErrorResponse(
                    error="No USD adapter available", error_type="AdapterError"
                ).model_dump()

            with adapter.create_session() as session:
                stage_id = session.load_stage(input_data.file_path)
                if stage_id:
                    stage_info = session.get_stage_info(stage_id)
                    if stage_info:
                        result = StageInfo(
                            stage_id=stage_id,
                            file_path=input_data.file_path,
                            up_axis=stage_info.up_axis,
                            meters_per_unit=stage_info.meters_per_unit,
                            time_codes_per_second=stage_info.time_codes_per_second,
                            start_time=stage_info.start_time_code,
                            end_time=stage_info.end_time_code,
                            frame_rate=stage_info.frame_rate,
                            total_prims=len(stage_info.all_prims),
                            root_prims=stage_info.root_prims,
                            has_animation=stage_info.start_time_code
                            != stage_info.end_time_code,
                            layer_count=len(stage_info.layers),
                            default_prim=stage_info.default_prim,
                        ).model_dump()
                        return server._validate_output(
                            result, (StageInfo, ErrorResponse), "load_usd_file"
                        )

            result = ErrorResponse(
                error=f"Failed to load USD file: {input_data.file_path}",
                error_type="LoadError",
            ).model_dump()
            return server._validate_output(
                result, (StageInfo, ErrorResponse), "load_usd_file"
            )

        except Exception as e:
            server.logger.error(f"Error loading USD file {input_data.file_path}: {e}")
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result, (StageInfo, ErrorResponse), "load_usd_file"
            )

    @server.mcp.tool(
        name="validate_usd_file",
        description=(
            "Validate a USD file without loading it. "
            "Operates on local USD files via the headless adapter — "
            "no running application required. For live Isaac Sim stages, "
            "use the isaac_* prefixed tools instead."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=True
        ),
        output_schema=None,
    )
    async def validate_usd_file(file_path: str) -> Dict[str, Any]:
        rate_error = server._check_rate_limit("validate_usd_file")
        if rate_error:
            return rate_error

        input_data = server._validate_input(USDValidateRequest, file_path=file_path)
        if isinstance(input_data, dict):
            return input_data

        if not server._is_path_allowed(input_data.file_path):
            return ErrorResponse(
                error="File path is not allowed by sandbox policy",
                error_type="SandboxError",
                details={"file_path": input_data.file_path},
            ).model_dump()

        try:
            path = Path(input_data.file_path)
            file_exists = path.exists()
            is_file = path.is_file() if file_exists else False
            file_size = path.stat().st_size if is_file else 0
            valid_extensions = [".usd", ".usda", ".usdc", ".usdz"]
            valid_extension = path.suffix.lower() in valid_extensions
            max_size_mb = server.settings.usd.max_file_size_mb
            size_ok = file_size <= (max_size_mb * 1024 * 1024)

            result = USDFileInfo(
                file_path=str(path.resolve()),
                file_size=file_size,
                format=path.suffix.lower().lstrip("."),
                is_valid=file_exists and is_file and valid_extension and size_ok,
                can_read=file_exists and is_file and valid_extension,
            ).model_dump()
            return server._validate_output(
                result, (USDFileInfo, ErrorResponse), "validate_usd_file"
            )
        except Exception as e:
            server.logger.error(
                f"Error validating USD file {input_data.file_path}: {e}"
            )
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result, (USDFileInfo, ErrorResponse), "validate_usd_file"
            )

    @server.mcp.tool(
        name="get_prim_info",
        description=(
            "Get information about a USD prim. "
            "Operates on local USD files via the headless adapter — "
            "no running application required. For live Isaac Sim stages, "
            "use the isaac_* prefixed tools instead."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=False
        ),
        output_schema=None,
    )
    async def get_prim_info(stage_id: str, prim_path: str) -> Dict[str, Any]:
        """
        Get information about a USD prim.

        Args:
            stage_id: Stage identifier
            prim_path: Path to the prim

        Returns:
            Prim information or error
        """
        rate_error = server._check_rate_limit("get_prim_info")
        if rate_error:
            return rate_error

        input_data = server._validate_input(
            PrimInfoRequest, stage_id=stage_id, prim_path=prim_path
        )
        if isinstance(input_data, dict):
            return input_data

        try:
            adapter = server.headless_adapter
            if not adapter:
                return ErrorResponse(
                    error="No USD adapter available", error_type="AdapterError"
                ).model_dump()

            with adapter.create_session() as session:
                prim_info = session.get_prim_info(
                    input_data.stage_id, input_data.prim_path
                )
                if prim_info:
                    # Convert to response format
                    bbox = None
                    if hasattr(session, "get_prim_bbox"):
                        bbox_dict = session.get_prim_bbox(
                            input_data.stage_id, input_data.prim_path
                        )
                        if bbox_dict:
                            bbox = BoundingBox(**bbox_dict)

                    transform = None
                    if hasattr(session, "get_prim_transform"):
                        transform_dict = session.get_prim_transform(
                            input_data.stage_id, input_data.prim_path
                        )
                        if transform_dict:
                            transform = Transform(**transform_dict)

                    children_types = {}
                    if hasattr(session, "get_children_type_counts"):
                        children_types = session.get_children_type_counts(
                            input_data.stage_id, input_data.prim_path
                        )

                    material_bindings = []
                    if hasattr(session, "get_material_bindings"):
                        material_bindings = session.get_material_bindings(
                            input_data.stage_id, input_data.prim_path
                        )

                    result = PrimInfo(
                        path=prim_info.path,
                        name=prim_info.name,
                        type=prim_info.type_name,
                        is_active=prim_info.is_active,
                        is_loaded=prim_info.is_loaded,
                        is_defined=prim_info.is_defined,
                        is_instance=prim_info.is_instance,
                        purpose=prim_info.purpose,
                        visibility=prim_info.visibility,
                        kind=prim_info.kind,
                        bbox=bbox,
                        transform=transform,
                        children_count=len(prim_info.children),
                        children_types=children_types,
                        material_bindings=material_bindings,
                        attributes=prim_info.attributes,
                        metadata=prim_info.metadata,
                    ).model_dump()
                    return server._validate_output(
                        result, (PrimInfo, ErrorResponse), "get_prim_info"
                    )

            result = ErrorResponse(
                error=f"Prim not found: {input_data.prim_path}",
                error_type="NotFoundError",
            ).model_dump()
            return server._validate_output(
                result, (PrimInfo, ErrorResponse), "get_prim_info"
            )

        except Exception as e:
            server.logger.error(
                "Error getting prim info "
                f"{input_data.stage_id}:{input_data.prim_path}: {e}"
            )
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result, (PrimInfo, ErrorResponse), "get_prim_info"
            )

    @server.mcp.tool(
        name="create_prim",
        description=(
            "Create a prim in a USD stage. "
            "Operates on local USD files via the headless adapter — "
            "no running application required. For live Isaac Sim stages, "
            "use the isaac_* prefixed tools instead."
        ),
        annotations=server._tool_annotations(
            read_only=False, idempotent=False, open_world=False, destructive=True
        ),
        output_schema=None,
    )
    async def create_prim(
        stage_id: str,
        prim_path: str,
        prim_type: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        rate_error = server._check_rate_limit("create_prim")
        if rate_error:
            return rate_error

        input_data = server._validate_input(
            PrimCreateRequest,
            stage_id=stage_id,
            prim_path=prim_path,
            prim_type=prim_type,
            attributes=attributes or {},
        )
        if isinstance(input_data, dict):
            return input_data

        try:
            adapter = server.headless_adapter
            if not adapter:
                return ErrorResponse(
                    error="No USD adapter available", error_type="AdapterError"
                ).model_dump()

            with adapter.create_session() as session:
                success = session.create_prim(
                    input_data.stage_id,
                    input_data.prim_path,
                    input_data.prim_type,
                    input_data.attributes,
                )
                if success:
                    result = PrimActionResponse(
                        success=True,
                        stage_id=input_data.stage_id,
                        prim_path=input_data.prim_path,
                        message=f"Created prim {input_data.prim_path}",
                    ).model_dump()
                    return server._validate_output(
                        result, (PrimActionResponse, ErrorResponse), "create_prim"
                    )

            result = ErrorResponse(
                error=f"Failed to create prim: {input_data.prim_path}",
                error_type="CreateError",
            ).model_dump()
            return server._validate_output(
                result, (PrimActionResponse, ErrorResponse), "create_prim"
            )

        except Exception as e:
            server.logger.error(
                "Error creating prim "
                f"{input_data.stage_id}:{input_data.prim_path}: {e}"
            )
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result, (PrimActionResponse, ErrorResponse), "create_prim"
            )

    @server.mcp.tool(
        name="update_prim_attributes",
        description=(
            "Update attributes on a USD prim. "
            "Operates on local USD files via the headless adapter — "
            "no running application required. For live Isaac Sim stages, "
            "use the isaac_* prefixed tools instead."
        ),
        annotations=server._tool_annotations(
            read_only=False, idempotent=False, open_world=False, destructive=True
        ),
        output_schema=None,
    )
    async def update_prim_attributes(
        stage_id: str,
        prim_path: str,
        attributes: Dict[str, Any],
    ) -> Dict[str, Any]:
        rate_error = server._check_rate_limit("update_prim_attributes")
        if rate_error:
            return rate_error

        input_data = server._validate_input(
            PrimUpdateRequest,
            stage_id=stage_id,
            prim_path=prim_path,
            attributes=attributes,
        )
        if isinstance(input_data, dict):
            return input_data

        try:
            adapter = server.headless_adapter
            if not adapter:
                return ErrorResponse(
                    error="No USD adapter available", error_type="AdapterError"
                ).model_dump()

            with adapter.create_session() as session:
                success = session.update_prim_attributes(
                    input_data.stage_id,
                    input_data.prim_path,
                    input_data.attributes,
                )
                if success:
                    result = PrimActionResponse(
                        success=True,
                        stage_id=input_data.stage_id,
                        prim_path=input_data.prim_path,
                        message=f"Updated prim {input_data.prim_path}",
                    ).model_dump()
                    return server._validate_output(
                        result,
                        (PrimActionResponse, ErrorResponse),
                        "update_prim_attributes",
                    )

            result = ErrorResponse(
                error=f"Failed to update prim: {input_data.prim_path}",
                error_type="UpdateError",
            ).model_dump()
            return server._validate_output(
                result,
                (PrimActionResponse, ErrorResponse),
                "update_prim_attributes",
            )

        except Exception as e:
            server.logger.error(
                "Error updating prim "
                f"{input_data.stage_id}:{input_data.prim_path}: {e}"
            )
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result,
                (PrimActionResponse, ErrorResponse),
                "update_prim_attributes",
            )

    @server.mcp.tool(
        name="delete_prim",
        description=(
            "Delete a prim from a USD stage. "
            "Operates on local USD files via the headless adapter — "
            "no running application required. For live Isaac Sim stages, "
            "use the isaac_* prefixed tools instead."
        ),
        annotations=server._tool_annotations(
            read_only=False, idempotent=False, open_world=False, destructive=True
        ),
        output_schema=None,
    )
    async def delete_prim(stage_id: str, prim_path: str) -> Dict[str, Any]:
        rate_error = server._check_rate_limit("delete_prim")
        if rate_error:
            return rate_error

        input_data = server._validate_input(
            PrimDeleteRequest,
            stage_id=stage_id,
            prim_path=prim_path,
        )
        if isinstance(input_data, dict):
            return input_data

        try:
            adapter = server.headless_adapter
            if not adapter:
                return ErrorResponse(
                    error="No USD adapter available", error_type="AdapterError"
                ).model_dump()

            with adapter.create_session() as session:
                success = session.delete_prim(
                    input_data.stage_id, input_data.prim_path
                )
                if success:
                    result = PrimActionResponse(
                        success=True,
                        stage_id=input_data.stage_id,
                        prim_path=input_data.prim_path,
                        message=f"Deleted prim {input_data.prim_path}",
                    ).model_dump()
                    return server._validate_output(
                        result, (PrimActionResponse, ErrorResponse), "delete_prim"
                    )

            result = ErrorResponse(
                error=f"Failed to delete prim: {input_data.prim_path}",
                error_type="DeleteError",
            ).model_dump()
            return server._validate_output(
                result, (PrimActionResponse, ErrorResponse), "delete_prim"
            )

        except Exception as e:
            server.logger.error(
                "Error deleting prim "
                f"{input_data.stage_id}:{input_data.prim_path}: {e}"
            )
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result, (PrimActionResponse, ErrorResponse), "delete_prim"
            )

    @server.mcp.tool(
        name="get_mesh_info",
        description=(
            "Get mesh information for a mesh prim. "
            "Operates on local USD files via the headless adapter — "
            "no running application required. For live Isaac Sim stages, "
            "use the isaac_* prefixed tools instead."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=False
        ),
        output_schema=None,
    )
    async def get_mesh_info(stage_id: str, prim_path: str) -> Dict[str, Any]:
        rate_error = server._check_rate_limit("get_mesh_info")
        if rate_error:
            return rate_error

        input_data = server._validate_input(
            MeshInfoRequest, stage_id=stage_id, prim_path=prim_path
        )
        if isinstance(input_data, dict):
            return input_data

        try:
            adapter = server.headless_adapter
            if not adapter:
                return ErrorResponse(
                    error="No USD adapter available", error_type="AdapterError"
                ).model_dump()

            with adapter.create_session() as session:
                mesh_info = session.get_mesh_info(
                    input_data.stage_id, input_data.prim_path
                )
                if mesh_info:
                    result = MeshInfo(**mesh_info).model_dump()
                    return server._validate_output(
                        result, (MeshInfo, ErrorResponse), "get_mesh_info"
                    )

            result = ErrorResponse(
                error=f"Mesh not found: {input_data.prim_path}",
                error_type="NotFoundError",
            ).model_dump()
            return server._validate_output(
                result, (MeshInfo, ErrorResponse), "get_mesh_info"
            )

        except Exception as e:
            server.logger.error(
                "Error getting mesh info "
                f"{input_data.stage_id}:{input_data.prim_path}: {e}"
            )
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result, (MeshInfo, ErrorResponse), "get_mesh_info"
            )

    @server.mcp.tool(
        name="search_prims",
        description=(
            "Search for prims in a USD stage. "
            "Operates on local USD files via the headless adapter — "
            "no running application required. For live Isaac Sim stages, "
            "use the isaac_* prefixed tools instead."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=False
        ),
        output_schema=None,
    )
    async def search_prims(
        stage_id: str, search_type: str, query: str, exact_match: bool = False
    ) -> Dict[str, Any]:
        """
        Search for prims in a USD stage.

        Args:
            stage_id: Stage identifier
            search_type: Search type (by_type, by_name)
            query: Search query
            exact_match: Use exact matching for name search

        Returns:
            Search results or error
        """
        rate_error = server._check_rate_limit("search_prims")
        if rate_error:
            return rate_error

        input_data = server._validate_input(
            PrimSearchRequest,
            stage_id=stage_id,
            search_type=search_type,
            query=query,
            exact_match=exact_match,
        )
        if isinstance(input_data, dict):
            return input_data

        try:
            adapter = server.headless_adapter
            if not adapter:
                return ErrorResponse(
                    error="No USD adapter available", error_type="AdapterError"
                ).model_dump()

            with adapter.create_session() as session:
                results: List[str] = []
                if input_data.search_type == "by_type":
                    results = session.find_prims_by_type(
                        input_data.stage_id, input_data.query
                    )
                elif input_data.search_type == "by_name":
                    results = session.find_prims_by_name(
                        input_data.stage_id,
                        input_data.query,
                        input_data.exact_match,
                    )

                result = PrimSearchResponse(
                    success=True,
                    stage_id=input_data.stage_id,
                    search_type=input_data.search_type,
                    query=input_data.query,
                    results=results,
                    count=len(results),
                ).model_dump()
                # search_type="by_type", query="Mesh" returns every mesh path in
                # the file with no cap. The USD tools have no shared chokepoint
                # to hang this on, so the budget is applied here directly;
                # `count` stays the true total.
                return apply_result_budget(
                    server._validate_output(
                        result, (PrimSearchResponse, ErrorResponse), "search_prims"
                    )
                )

        except Exception as e:
            server.logger.error(f"Error searching prims {input_data.stage_id}: {e}")
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result, (PrimSearchResponse, ErrorResponse), "search_prims"
            )

    @server.mcp.tool(
        name="get_bounding_box",
        description=(
            "Get bounding box for a prim or entire stage. "
            "Operates on local USD files via the headless adapter — "
            "no running application required. For live Isaac Sim stages, "
            "use the isaac_* prefixed tools instead."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=False
        ),
        output_schema=None,
    )
    async def get_bounding_box(
        stage_id: str, prim_path: Optional[str] = None, world_space: bool = True
    ) -> Dict[str, Any]:
        """
        Get bounding box for a prim or entire stage.

        Args:
            stage_id: Stage identifier
            prim_path: Prim path (None for stage bbox)
            world_space: Compute in world space

        Returns:
            Bounding box information or error
        """
        rate_error = server._check_rate_limit("get_bounding_box")
        if rate_error:
            return rate_error

        input_data = server._validate_input(
            BBoxRequest,
            stage_id=stage_id,
            prim_path=prim_path,
            world_space=world_space,
            time_code=None,
        )
        if isinstance(input_data, dict):
            return input_data

        try:
            adapter = server.headless_adapter
            if not adapter:
                return ErrorResponse(
                    error="No USD adapter available", error_type="AdapterError"
                ).model_dump()

            with adapter.create_session() as session:
                if input_data.prim_path:
                    bbox_dict = session.get_prim_bbox(
                        input_data.stage_id,
                        input_data.prim_path,
                        input_data.world_space,
                    )
                else:
                    bbox_dict = session.get_stage_bbox(input_data.stage_id)

                if bbox_dict:
                    bbox = BoundingBox(**bbox_dict)
                    result = BBoxResponse(
                        success=True,
                        stage_id=input_data.stage_id,
                        prim_path=input_data.prim_path,
                        bbox=bbox,
                        world_space=input_data.world_space,
                    ).model_dump()
                    return server._validate_output(
                        result, (BBoxResponse, ErrorResponse), "get_bounding_box"
                    )
                else:
                    result = ErrorResponse(
                        error="Could not compute bounding box",
                        error_type="ComputationError",
                    ).model_dump()
                    return server._validate_output(
                        result, (BBoxResponse, ErrorResponse), "get_bounding_box"
                    )

        except Exception as e:
            server.logger.error(
                "Error computing bounding box "
                f"{input_data.stage_id}:{input_data.prim_path}: {e}"
            )
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result, (BBoxResponse, ErrorResponse), "get_bounding_box"
            )

    @server.mcp.tool(
        name="summarize_scene",
        description=(
            "Generate a summary of a USD scene. "
            "Operates on local USD files via the headless adapter — "
            "no running application required. For live Isaac Sim stages, "
            "use the isaac_* prefixed tools instead."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=False
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def summarize_scene(
        stage_id: str, include_meshes: bool = True, format: str = "json"
    ) -> Dict[str, Any]:
        """
        Generate a summary of a USD scene.

        Args:
            stage_id: Stage identifier
            include_meshes: Include detailed mesh information
            format: Output format (json, text)

        Returns:
            Scene summary or error
        """
        rate_error = server._check_rate_limit("summarize_scene")
        if rate_error:
            return rate_error

        input_data = server._validate_input(
            SceneSummaryRequest,
            stage_id=stage_id,
            include_meshes=include_meshes,
            include_materials=True,
            max_depth=5,
            format=format,
        )
        if isinstance(input_data, dict):
            return input_data

        try:
            adapter = server.headless_adapter
            if not adapter:
                return ErrorResponse(
                    error="No USD adapter available", error_type="AdapterError"
                ).model_dump()

            with adapter.create_session() as session:
                summary = session.summarize_stage(
                    input_data.stage_id, input_data.include_meshes
                )
                if summary:
                    # Convert summary to dict
                    summary_dict = {
                        "file_path": summary.file_path,
                        "stage_info": summary.stage_info,
                        "total_prims": summary.total_prims,
                        "prim_type_counts": summary.prim_type_counts,
                        "scene_bbox": summary.scene_bbox,
                        "scene_center": summary.scene_center,
                        "scene_size": summary.scene_size,
                        "mesh_statistics": summary.mesh_statistics,
                        "hierarchy_depth": summary.hierarchy_depth,
                        "animation_info": summary.animation_info,
                    }

                    # Generate digest if text format requested
                    digest = None
                    if input_data.format == "text":
                        from ..usd.summarize import generate_scene_digest

                        digest = generate_scene_digest(summary)

                    result = SceneSummaryResponse(
                        success=True,
                        stage_id=input_data.stage_id,
                        summary=summary_dict,
                        digest=digest,
                    ).model_dump()
                    return server._validate_output(
                        result,
                        (SceneSummaryResponse, ErrorResponse),
                        "summarize_scene",
                    )
                else:
                    result = ErrorResponse(
                        error="Could not generate scene summary",
                        error_type="ComputationError",
                    ).model_dump()
                    return server._validate_output(
                        result,
                        (SceneSummaryResponse, ErrorResponse),
                        "summarize_scene",
                    )

        except Exception as e:
            server.logger.error(f"Error summarizing scene {input_data.stage_id}: {e}")
            result = ErrorResponse(error=str(e), error_type="Exception").model_dump()
            return server._validate_output(
                result, (SceneSummaryResponse, ErrorResponse), "summarize_scene"
            )
