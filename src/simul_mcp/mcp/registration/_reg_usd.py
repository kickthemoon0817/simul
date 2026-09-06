"""
USD headless file operation tool registration for Simul MCP Server.

Every tool here is a thin closure around ``server._exec_backend``: the
closure validates its input, performs the stage operation and builds the
response model; the envelope owns the rate limit, usage record, sandbox
and exception wrapping, result budget and single transmission.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from fastmcp.tools.tool import ToolResult

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
    from ..adapters.headless_usd import HeadlessUSDSession
    from ..server import SimulMCPServer


def register_usd_tools(server: "SimulMCPServer") -> None:
    """Register USD-related tools."""

    @server.mcp.tool(
        name="load_usd_file",
        description=(
            "Load a USD file and return stage information. Paths must be inside "
            "the configured sandbox (security.allowed_paths)."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=True
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def load_usd_file(file_path: str) -> ToolResult:
        """
        Load a USD file and return stage information.

        Args:
            file_path: Path to USD file

        Returns:
            Stage information or error
        """

        def _load(session: "HeadlessUSDSession") -> Dict[str, Any]:
            input_data = server._validate_input(USDFileRequest, file_path=file_path)
            if isinstance(input_data, dict):
                return input_data

            stage_id = session.load_stage(
                server._path_policy.authorize(input_data.file_path)
            )
            stage_info = session.get_stage_info(stage_id) if stage_id else None
            if not stage_id or stage_info is None:
                return ErrorResponse(
                    error=f"Failed to load USD file: {input_data.file_path}",
                    error_type="LoadError",
                ).model_dump()
            return StageInfo(
                stage_id=stage_id,
                file_path=input_data.file_path,
                up_axis=stage_info.up_axis,
                meters_per_unit=stage_info.meters_per_unit,
                time_codes_per_second=stage_info.time_codes_per_second,
                start_time=stage_info.start_time_code,
                end_time=stage_info.end_time_code,
                frame_rate=stage_info.frame_rate,
                total_prims=stage_info.prim_count,
                root_prims=stage_info.root_prims,
                has_animation=stage_info.start_time_code != stage_info.end_time_code,
                layer_count=len(stage_info.layers),
                default_prim=stage_info.default_prim_path,
            ).model_dump()

        return await server._exec_backend(
            "load_usd_file",
            server.headless_adapter,
            "USD",
            StageInfo,
            _load,
            params={"file_path": file_path},
        )

    @server.mcp.tool(
        name="validate_usd_file",
        description=(
            "Validate a USD file without loading it. Paths must be inside the "
            "configured sandbox (security.allowed_paths)."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=True
        ),
        output_schema=None,
    )
    async def validate_usd_file(file_path: str) -> ToolResult:
        """
        Check that a path names a readable USD file within the size limit.

        Args:
            file_path: Path to USD file

        Returns:
            File validity information or error
        """

        def _validate(session: "HeadlessUSDSession") -> Dict[str, Any]:
            input_data = server._validate_input(USDValidateRequest, file_path=file_path)
            if isinstance(input_data, dict):
                return input_data

            path = Path(server._path_policy.authorize(input_data.file_path))
            file_exists = path.exists()
            is_file = path.is_file() if file_exists else False
            file_size = path.stat().st_size if is_file else 0
            valid_extension = path.suffix.lower() in (".usd", ".usda", ".usdc", ".usdz")
            max_size_mb = server.settings.usd.max_file_size_mb
            size_ok = file_size <= (max_size_mb * 1024 * 1024)
            return USDFileInfo(
                file_path=str(path.resolve()),
                file_size=file_size,
                format=path.suffix.lower().lstrip("."),
                is_valid=file_exists and is_file and valid_extension and size_ok,
                can_read=file_exists and is_file and valid_extension,
            ).model_dump()

        return await server._exec_backend(
            "validate_usd_file",
            server.headless_adapter,
            "USD",
            USDFileInfo,
            _validate,
            params={"file_path": file_path},
        )

    @server.mcp.tool(
        name="get_prim_info",
        description="Get information about a USD prim.",
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=False
        ),
        output_schema=None,
    )
    async def get_prim_info(stage_id: str, prim_path: str) -> ToolResult:
        """
        Get information about a USD prim.

        Args:
            stage_id: Stage identifier
            prim_path: Path to the prim

        Returns:
            Prim information or error
        """

        def _prim_info(session: "HeadlessUSDSession") -> Dict[str, Any]:
            input_data = server._validate_input(
                PrimInfoRequest, stage_id=stage_id, prim_path=prim_path
            )
            if isinstance(input_data, dict):
                return input_data

            prim_info = session.get_prim_info(input_data.stage_id, input_data.prim_path)
            if not prim_info:
                return ErrorResponse(
                    error=f"Prim not found: {input_data.prim_path}",
                    error_type="NotFoundError",
                ).model_dump()

            bbox_dict = session.get_prim_bbox(input_data.stage_id, input_data.prim_path)
            transform_dict = session.get_prim_transform(
                input_data.stage_id, input_data.prim_path
            )
            return PrimInfo(
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
                bbox=BoundingBox(**bbox_dict) if bbox_dict else None,
                transform=Transform(**transform_dict) if transform_dict else None,
                children_count=len(prim_info.children),
                children_types=session.get_children_type_counts(
                    input_data.stage_id, input_data.prim_path
                ),
                material_bindings=session.get_material_bindings(
                    input_data.stage_id, input_data.prim_path
                ),
                attributes=prim_info.attributes,
                metadata=prim_info.metadata,
            ).model_dump()

        return await server._exec_backend(
            "get_prim_info",
            server.headless_adapter,
            "USD",
            PrimInfo,
            _prim_info,
            params={"stage_id": stage_id, "prim_path": prim_path},
        )

    @server.mcp.tool(
        name="create_prim",
        description="Create a prim in a USD stage.",
        annotations=server._tool_annotations(
            read_only=False, idempotent=False, open_world=False
        ),
        output_schema=None,
    )
    async def create_prim(
        stage_id: str,
        prim_path: str,
        prim_type: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """
        Create a prim in a USD stage.

        Args:
            stage_id: Stage identifier
            prim_path: Path of the prim to create
            prim_type: USD prim type name
            attributes: Initial attribute values

        Returns:
            Action confirmation or error
        """

        def _create(session: "HeadlessUSDSession") -> Dict[str, Any]:
            input_data = server._validate_input(
                PrimCreateRequest,
                stage_id=stage_id,
                prim_path=prim_path,
                prim_type=prim_type,
                attributes=attributes or {},
            )
            if isinstance(input_data, dict):
                return input_data

            created = session.create_prim(
                input_data.stage_id,
                input_data.prim_path,
                input_data.prim_type,
                input_data.attributes,
            )
            if not created:
                return ErrorResponse(
                    error=f"Failed to create prim: {input_data.prim_path}",
                    error_type="CreateError",
                ).model_dump()
            return PrimActionResponse(
                success=True,
                stage_id=input_data.stage_id,
                prim_path=input_data.prim_path,
                message=f"Created prim {input_data.prim_path}",
            ).model_dump()

        return await server._exec_backend(
            "create_prim",
            server.headless_adapter,
            "USD",
            PrimActionResponse,
            _create,
            params={"stage_id": stage_id, "prim_path": prim_path, "prim_type": prim_type},
        )

    @server.mcp.tool(
        name="update_prim_attributes",
        description="Update attributes on a USD prim.",
        annotations=server._tool_annotations(
            read_only=False, idempotent=False, open_world=False, destructive=True
        ),
        output_schema=None,
    )
    async def update_prim_attributes(
        stage_id: str,
        prim_path: str,
        attributes: Dict[str, Any],
    ) -> ToolResult:
        """
        Update attributes on a USD prim.

        Args:
            stage_id: Stage identifier
            prim_path: Path to the prim
            attributes: Attribute name to value mapping

        Returns:
            Action confirmation or error
        """

        def _update(session: "HeadlessUSDSession") -> Dict[str, Any]:
            input_data = server._validate_input(
                PrimUpdateRequest,
                stage_id=stage_id,
                prim_path=prim_path,
                attributes=attributes,
            )
            if isinstance(input_data, dict):
                return input_data

            updated = session.update_prim_attributes(
                input_data.stage_id, input_data.prim_path, input_data.attributes
            )
            if not updated:
                return ErrorResponse(
                    error=f"Failed to update prim: {input_data.prim_path}",
                    error_type="UpdateError",
                ).model_dump()
            return PrimActionResponse(
                success=True,
                stage_id=input_data.stage_id,
                prim_path=input_data.prim_path,
                message=f"Updated prim {input_data.prim_path}",
            ).model_dump()

        return await server._exec_backend(
            "update_prim_attributes",
            server.headless_adapter,
            "USD",
            PrimActionResponse,
            _update,
            params={
                "stage_id": stage_id,
                "prim_path": prim_path,
                "attributes": sorted(attributes),
            },
        )

    @server.mcp.tool(
        name="delete_prim",
        description="Delete a prim from a USD stage.",
        annotations=server._tool_annotations(
            read_only=False, idempotent=False, open_world=False, destructive=True
        ),
        output_schema=None,
    )
    async def delete_prim(stage_id: str, prim_path: str) -> ToolResult:
        """
        Delete a prim from a USD stage.

        Args:
            stage_id: Stage identifier
            prim_path: Path to the prim

        Returns:
            Action confirmation or error
        """

        def _delete(session: "HeadlessUSDSession") -> Dict[str, Any]:
            input_data = server._validate_input(
                PrimDeleteRequest, stage_id=stage_id, prim_path=prim_path
            )
            if isinstance(input_data, dict):
                return input_data

            if not session.delete_prim(input_data.stage_id, input_data.prim_path):
                return ErrorResponse(
                    error=f"Failed to delete prim: {input_data.prim_path}",
                    error_type="DeleteError",
                ).model_dump()
            return PrimActionResponse(
                success=True,
                stage_id=input_data.stage_id,
                prim_path=input_data.prim_path,
                message=f"Deleted prim {input_data.prim_path}",
            ).model_dump()

        return await server._exec_backend(
            "delete_prim",
            server.headless_adapter,
            "USD",
            PrimActionResponse,
            _delete,
            params={"stage_id": stage_id, "prim_path": prim_path},
        )

    @server.mcp.tool(
        name="get_mesh_info",
        description="Get mesh information for a mesh prim.",
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=False
        ),
        output_schema=None,
    )
    async def get_mesh_info(stage_id: str, prim_path: str) -> ToolResult:
        """
        Get mesh information for a mesh prim.

        Args:
            stage_id: Stage identifier
            prim_path: Path to the mesh prim

        Returns:
            Mesh statistics or error
        """

        def _mesh_info(session: "HeadlessUSDSession") -> Dict[str, Any]:
            input_data = server._validate_input(
                MeshInfoRequest, stage_id=stage_id, prim_path=prim_path
            )
            if isinstance(input_data, dict):
                return input_data

            mesh_info = session.get_mesh_info(input_data.stage_id, input_data.prim_path)
            if not mesh_info:
                return ErrorResponse(
                    error=f"Mesh not found: {input_data.prim_path}",
                    error_type="NotFoundError",
                ).model_dump()
            return MeshInfo(**mesh_info).model_dump()

        return await server._exec_backend(
            "get_mesh_info",
            server.headless_adapter,
            "USD",
            MeshInfo,
            _mesh_info,
            params={"stage_id": stage_id, "prim_path": prim_path},
        )

    @server.mcp.tool(
        name="search_prims",
        description="Search for prims in a USD stage.",
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=False
        ),
        output_schema=None,
    )
    async def search_prims(
        stage_id: str, search_type: str, query: str, exact_match: bool = False
    ) -> ToolResult:
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

        def _search(session: "HeadlessUSDSession") -> Dict[str, Any]:
            input_data = server._validate_input(
                PrimSearchRequest,
                stage_id=stage_id,
                search_type=search_type,
                query=query,
                exact_match=exact_match,
            )
            if isinstance(input_data, dict):
                return input_data

            results: List[str] = []
            if input_data.search_type == "by_type":
                results = session.find_prims_by_type(input_data.stage_id, input_data.query)
            elif input_data.search_type == "by_name":
                results = session.find_prims_by_name(
                    input_data.stage_id, input_data.query, input_data.exact_match
                )
            # The envelope's result budget trims ``results`` when a broad
            # query matches every mesh in the file; ``count`` stays the true total.
            return PrimSearchResponse(
                success=True,
                stage_id=input_data.stage_id,
                search_type=input_data.search_type,
                query=input_data.query,
                results=results,
                count=len(results),
            ).model_dump()

        return await server._exec_backend(
            "search_prims",
            server.headless_adapter,
            "USD",
            PrimSearchResponse,
            _search,
            params={"stage_id": stage_id, "search_type": search_type, "query": query},
        )

    @server.mcp.tool(
        name="get_bounding_box",
        description="Get bounding box for a prim or entire stage.",
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=False
        ),
        output_schema=None,
    )
    async def get_bounding_box(
        stage_id: str, prim_path: Optional[str] = None, world_space: bool = True
    ) -> ToolResult:
        """
        Get bounding box for a prim or entire stage.

        Args:
            stage_id: Stage identifier
            prim_path: Prim path (None for stage bbox)
            world_space: Compute in world space

        Returns:
            Bounding box information or error
        """

        def _bbox(session: "HeadlessUSDSession") -> Dict[str, Any]:
            input_data = server._validate_input(
                BBoxRequest,
                stage_id=stage_id,
                prim_path=prim_path,
                world_space=world_space,
                time_code=None,
            )
            if isinstance(input_data, dict):
                return input_data

            if input_data.prim_path:
                bbox_dict = session.get_prim_bbox(
                    input_data.stage_id, input_data.prim_path, input_data.world_space
                )
            else:
                bbox_dict = session.get_stage_bbox(input_data.stage_id)
            if not bbox_dict:
                return ErrorResponse(
                    error="Could not compute bounding box",
                    error_type="ComputationError",
                ).model_dump()
            return BBoxResponse(
                success=True,
                stage_id=input_data.stage_id,
                prim_path=input_data.prim_path,
                bbox=BoundingBox(**bbox_dict),
                world_space=input_data.world_space,
            ).model_dump()

        return await server._exec_backend(
            "get_bounding_box",
            server.headless_adapter,
            "USD",
            BBoxResponse,
            _bbox,
            params={"stage_id": stage_id, "prim_path": prim_path},
        )

    @server.mcp.tool(
        name="summarize_scene",
        description="Generate a summary of a USD scene.",
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=False
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    async def summarize_scene(
        stage_id: str, include_meshes: bool = True, format: str = "json"
    ) -> ToolResult:
        """
        Generate a summary of a USD scene.

        Args:
            stage_id: Stage identifier
            include_meshes: Include detailed mesh information
            format: Output format (json, text)

        Returns:
            Scene summary or error
        """

        def _summarize(session: "HeadlessUSDSession") -> Dict[str, Any]:
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

            summary = session.summarize_stage(input_data.stage_id, input_data.include_meshes)
            if not summary:
                return ErrorResponse(
                    error="Could not generate scene summary",
                    error_type="ComputationError",
                ).model_dump()

            digest: Optional[str] = None
            if input_data.format == "text":
                from ...usd.summarize import generate_scene_digest

                digest = generate_scene_digest(summary)
            return SceneSummaryResponse(
                success=True,
                stage_id=input_data.stage_id,
                summary={
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
                },
                digest=digest,
            ).model_dump()

        return await server._exec_backend(
            "summarize_scene",
            server.headless_adapter,
            "USD",
            SceneSummaryResponse,
            _summarize,
            params={"stage_id": stage_id, "include_meshes": include_meshes, "format": format},
        )
