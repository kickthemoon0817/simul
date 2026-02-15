"""
Isaac Sim MCP Server implementation.

This module provides the main MCP server class with tool registry,
connection management, and Isaac Sim integration based on FastMCP.
"""

import asyncio
import inspect
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable, Type, Tuple
from contextlib import asynccontextmanager

from pydantic import BaseModel

try:
    from fastmcp import FastMCP
    from mcp.types import (
        Tool,
        TextContent,
        ImageContent,
        EmbeddedResource,
        ToolAnnotations,
    )
    from fastmcp.server.tasks import TaskConfig

    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False
    FastMCP = None
    Tool = None
    TextContent = None
    ImageContent = None
    EmbeddedResource = None
    ToolAnnotations = None
    TaskConfig = None

from ..logging import get_logger, LoggerMixin
from ..config import Settings, get_settings
from ..adapters import (
    BlenderRuntimeAdapter,
    HeadlessUSDAdapter,
    IsaacRuntimeAdapter,
    is_blender_available,
    is_isaac_available,
    is_headless_available,
)
from ..utils.timing import RateLimiter
from .schemas import *

logger = get_logger(__name__)


class IsaacMCPServer(LoggerMixin):
    """
    Isaac Sim MCP Server.

    Provides MCP server functionality for Isaac Sim with USD operations,
    viewport capture, and simulation control.
    """

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize Isaac MCP Server.

        Args:
            settings: Configuration settings
        """
        if not FASTMCP_AVAILABLE:
            raise ImportError("FastMCP not available. Please install fastmcp package.")

        self.settings = settings or get_settings()
        self._project_root = Path(__file__).resolve().parents[3]
        self._allowed_paths = self._resolve_allowed_paths()

        self._rate_limiters: Dict[str, RateLimiter] = {}
        self._rate_limit_enabled = self.settings.security.rate_limiting_enabled
        self._rate_limit_rate = self.settings.security.requests_per_minute / 60.0
        self._rate_limit_burst = self.settings.security.burst_size
        self._tool_timeout = self.settings.server.timeout

        # Initialize adapters
        self.headless_adapter = (
            HeadlessUSDAdapter(self.settings) if is_headless_available() else None
        )
        self.isaac_adapter = (
            IsaacRuntimeAdapter(self.settings) if is_isaac_available() else None
        )
        self.blender_adapter = (
            BlenderRuntimeAdapter(self.settings) if is_blender_available() else None
        )

        # Initialize FastMCP server
        assert FastMCP is not None

        mcp_kwargs: Dict[str, Any] = {
            "name": "simul-mcp",
            "version": "0.1.7",
        }
        if "description" in inspect.signature(FastMCP).parameters:
            mcp_kwargs["description"] = (
                "MCP server for simulation and DCC tools with USD operations "
                "and simulation control"
            )

        self.mcp = FastMCP(
            **mcp_kwargs,
        )

        # Register tools
        self._register_tools()

        self.logger.info("Simul MCP Server initialized")

    def _get_rate_limiter(self, tool_name: str) -> Optional[RateLimiter]:
        if not self._rate_limit_enabled:
            return None

        if tool_name not in self._rate_limiters:
            self._rate_limiters[tool_name] = RateLimiter(
                self._rate_limit_rate,
                self._rate_limit_burst,
            )
        return self._rate_limiters[tool_name]

    def _check_rate_limit(self, tool_name: str) -> Optional[Dict[str, Any]]:
        limiter = self._get_rate_limiter(tool_name)
        if limiter and not limiter.acquire():
            return ErrorResponse(
                error="Rate limit exceeded",
                error_type="RateLimitError",
                details={"tool": tool_name},
            ).dict()
        return None

    def _resolve_allowed_paths(self) -> List[Path]:
        allowed_paths: List[Path] = []
        for path_str in self.settings.security.allowed_paths:
            expanded = os.path.expandvars(path_str)
            candidate = Path(expanded).expanduser()
            if not candidate.is_absolute():
                candidate = self._project_root / candidate
            try:
                candidate = candidate.resolve()
            except Exception:
                candidate = candidate.absolute()
            allowed_paths.append(candidate)
        return allowed_paths

    def _is_path_allowed(self, path_str: str) -> bool:
        if not self.settings.security.sandbox_enabled:
            return True
        if not path_str:
            return False
        expanded = os.path.expandvars(path_str)
        candidate = Path(expanded).expanduser()
        if not candidate.is_absolute():
            candidate = self._project_root / candidate
        try:
            candidate = candidate.resolve()
        except Exception:
            candidate = candidate.absolute()
        for allowed_path in self._allowed_paths:
            try:
                candidate.relative_to(allowed_path)
                return True
            except ValueError:
                continue
        return False

    def _validate_input(
        self, model: Type[BaseModel], **kwargs
    ) -> Union[BaseModel, Dict[str, Any]]:
        try:
            return model(**kwargs)
        except Exception as e:
            return ErrorResponse(
                error=str(e), error_type="ValidationError", details={"input": kwargs}
            ).dict()

    def _validate_output(
        self,
        result: Any,
        models: Tuple[Type[BaseModel], ...],
        tool_name: str,
    ) -> Dict[str, Any]:
        if isinstance(result, BaseModel):
            payload: Any = result.dict()
        else:
            payload = result

        if (
            isinstance(payload, dict)
            and payload.get("success") is False
            and payload.get("error")
        ):
            return payload

        if not isinstance(payload, dict):
            return ErrorResponse(
                error="Tool returned invalid response type",
                error_type="ValidationError",
                details={"type": type(payload).__name__},
            ).dict()

        for model in models:
            try:
                model(**payload)
                return payload
            except Exception:
                continue

        return ErrorResponse(
            error="Tool response failed schema validation",
            error_type="ValidationError",
            details={"tool": tool_name},
        ).dict()

    def _tool_annotations(
        self,
        read_only: bool,
        idempotent: bool,
        open_world: bool,
        destructive: bool = False,
    ) -> Optional[Any]:
        annotations = {
            "readOnlyHint": read_only,
            "idempotentHint": idempotent,
            "openWorldHint": open_world,
            "destructiveHint": destructive,
        }
        if ToolAnnotations:
            return ToolAnnotations(**annotations)
        return annotations

    def _tool_output_schema(self, *models: Type[BaseModel]) -> Dict[str, Any]:
        if len(models) == 1:
            model = models[0]
            if hasattr(model, "model_json_schema"):
                return model.model_json_schema()
            return model.schema()

        # FastMCP 2.x validates output schema as a single object and rejects
        # union schemas such as oneOf. Use a permissive object schema for
        # multi-response tools to stay compatible across FastMCP versions.
        return {
            "type": "object",
            "additionalProperties": True,
        }

    def _task_optional(self) -> Optional[Any]:
        if TaskConfig:
            return TaskConfig(mode="optional")
        return None

    def _register_tools(self) -> None:
        """Register all MCP tools."""

        # USD file operations
        self._register_usd_tools()

        # Isaac Sim specific tools (if available)
        if self.isaac_adapter and self.isaac_adapter.is_available():
            self._register_isaac_tools()

        # Blender specific tools (if available)
        if self.blender_adapter and self.blender_adapter.is_available():
            self._register_blender_tools()

        tool_count = len(getattr(self.mcp, "tools", []))
        if tool_count == 0 and hasattr(self.mcp, "_tool_manager"):
            tool_count = len(getattr(self.mcp._tool_manager, "_tools", {}))
        self.logger.info(f"Registered {tool_count} MCP tools")

    def _register_usd_tools(self) -> None:
        """Register USD-related tools."""

        @self.mcp.tool(
            name="load_usd_file",
            description="Load a USD file and return stage information.",
            annotations=self._tool_annotations(
                read_only=True, idempotent=True, open_world=True
            ),
            output_schema=self._tool_output_schema(StageInfo, ErrorResponse),
            task=self._task_optional(),
        )
        async def load_usd_file(file_path: str) -> Dict[str, Any]:
            """
            Load a USD file and return stage information.

            Args:
                file_path: Path to USD file

            Returns:
                Stage information or error
            """
            rate_error = self._check_rate_limit("load_usd_file")
            if rate_error:
                return rate_error

            input_data = self._validate_input(USDFileRequest, file_path=file_path)
            if isinstance(input_data, dict):
                return input_data

            if not self._is_path_allowed(input_data.file_path):
                return ErrorResponse(
                    error="File path is not allowed by sandbox policy",
                    error_type="SandboxError",
                    details={"file_path": input_data.file_path},
                ).dict()

            try:
                adapter = self.headless_adapter or self.isaac_adapter
                if not adapter:
                    return ErrorResponse(
                        error="No USD adapter available", error_type="AdapterError"
                    ).dict()

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
                            ).dict()
                            return self._validate_output(
                                result, (StageInfo, ErrorResponse), "load_usd_file"
                            )

                result = ErrorResponse(
                    error=f"Failed to load USD file: {input_data.file_path}",
                    error_type="LoadError",
                ).dict()
                return self._validate_output(
                    result, (StageInfo, ErrorResponse), "load_usd_file"
                )

            except Exception as e:
                self.logger.error(f"Error loading USD file {input_data.file_path}: {e}")
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result, (StageInfo, ErrorResponse), "load_usd_file"
                )

        @self.mcp.tool(
            name="validate_usd_file",
            description="Validate a USD file without loading it.",
            annotations=self._tool_annotations(
                read_only=True, idempotent=True, open_world=True
            ),
            output_schema=self._tool_output_schema(USDFileInfo, ErrorResponse),
        )
        async def validate_usd_file(file_path: str) -> Dict[str, Any]:
            rate_error = self._check_rate_limit("validate_usd_file")
            if rate_error:
                return rate_error

            input_data = self._validate_input(USDValidateRequest, file_path=file_path)
            if isinstance(input_data, dict):
                return input_data

            if not self._is_path_allowed(input_data.file_path):
                return ErrorResponse(
                    error="File path is not allowed by sandbox policy",
                    error_type="SandboxError",
                    details={"file_path": input_data.file_path},
                ).dict()

            try:
                path = Path(input_data.file_path)
                file_exists = path.exists()
                is_file = path.is_file() if file_exists else False
                file_size = path.stat().st_size if is_file else 0
                valid_extensions = [".usd", ".usda", ".usdc", ".usdz"]
                valid_extension = path.suffix.lower() in valid_extensions
                max_size_mb = self.settings.usd.max_file_size_mb
                size_ok = file_size <= (max_size_mb * 1024 * 1024)

                result = USDFileInfo(
                    file_path=str(path.resolve()),
                    file_size=file_size,
                    format=path.suffix.lower().lstrip("."),
                    is_valid=file_exists and is_file and valid_extension and size_ok,
                    can_read=file_exists and is_file and valid_extension,
                ).dict()
                return self._validate_output(
                    result, (USDFileInfo, ErrorResponse), "validate_usd_file"
                )
            except Exception as e:
                self.logger.error(
                    f"Error validating USD file {input_data.file_path}: {e}"
                )
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result, (USDFileInfo, ErrorResponse), "validate_usd_file"
                )

        @self.mcp.tool(
            name="get_prim_info",
            description="Get information about a USD prim.",
            annotations=self._tool_annotations(
                read_only=True, idempotent=True, open_world=False
            ),
            output_schema=self._tool_output_schema(PrimInfo, ErrorResponse),
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
            rate_error = self._check_rate_limit("get_prim_info")
            if rate_error:
                return rate_error

            input_data = self._validate_input(
                PrimInfoRequest, stage_id=stage_id, prim_path=prim_path
            )
            if isinstance(input_data, dict):
                return input_data

            try:
                adapter = self.headless_adapter or self.isaac_adapter
                if not adapter:
                    return ErrorResponse(
                        error="No USD adapter available", error_type="AdapterError"
                    ).dict()

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
                        ).dict()
                        return self._validate_output(
                            result, (PrimInfo, ErrorResponse), "get_prim_info"
                        )

                result = ErrorResponse(
                    error=f"Prim not found: {input_data.prim_path}",
                    error_type="NotFoundError",
                ).dict()
                return self._validate_output(
                    result, (PrimInfo, ErrorResponse), "get_prim_info"
                )

            except Exception as e:
                self.logger.error(
                    f"Error getting prim info {input_data.stage_id}:{input_data.prim_path}: {e}"
                )
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result, (PrimInfo, ErrorResponse), "get_prim_info"
                )

        @self.mcp.tool(
            name="create_prim",
            description="Create a prim in a USD stage.",
            annotations=self._tool_annotations(
                read_only=False, idempotent=False, open_world=False, destructive=True
            ),
            output_schema=self._tool_output_schema(PrimActionResponse, ErrorResponse),
        )
        async def create_prim(
            stage_id: str,
            prim_path: str,
            prim_type: str,
            attributes: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            rate_error = self._check_rate_limit("create_prim")
            if rate_error:
                return rate_error

            input_data = self._validate_input(
                PrimCreateRequest,
                stage_id=stage_id,
                prim_path=prim_path,
                prim_type=prim_type,
                attributes=attributes or {},
            )
            if isinstance(input_data, dict):
                return input_data

            try:
                adapter = self.headless_adapter or self.isaac_adapter
                if not adapter:
                    return ErrorResponse(
                        error="No USD adapter available", error_type="AdapterError"
                    ).dict()

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
                        ).dict()
                        return self._validate_output(
                            result, (PrimActionResponse, ErrorResponse), "create_prim"
                        )

                result = ErrorResponse(
                    error=f"Failed to create prim: {input_data.prim_path}",
                    error_type="CreateError",
                ).dict()
                return self._validate_output(
                    result, (PrimActionResponse, ErrorResponse), "create_prim"
                )

            except Exception as e:
                self.logger.error(
                    f"Error creating prim {input_data.stage_id}:{input_data.prim_path}: {e}"
                )
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result, (PrimActionResponse, ErrorResponse), "create_prim"
                )

        @self.mcp.tool(
            name="update_prim_attributes",
            description="Update attributes on a USD prim.",
            annotations=self._tool_annotations(
                read_only=False, idempotent=False, open_world=False, destructive=True
            ),
            output_schema=self._tool_output_schema(PrimActionResponse, ErrorResponse),
        )
        async def update_prim_attributes(
            stage_id: str,
            prim_path: str,
            attributes: Dict[str, Any],
        ) -> Dict[str, Any]:
            rate_error = self._check_rate_limit("update_prim_attributes")
            if rate_error:
                return rate_error

            input_data = self._validate_input(
                PrimUpdateRequest,
                stage_id=stage_id,
                prim_path=prim_path,
                attributes=attributes,
            )
            if isinstance(input_data, dict):
                return input_data

            try:
                adapter = self.headless_adapter or self.isaac_adapter
                if not adapter:
                    return ErrorResponse(
                        error="No USD adapter available", error_type="AdapterError"
                    ).dict()

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
                        ).dict()
                        return self._validate_output(
                            result,
                            (PrimActionResponse, ErrorResponse),
                            "update_prim_attributes",
                        )

                result = ErrorResponse(
                    error=f"Failed to update prim: {input_data.prim_path}",
                    error_type="UpdateError",
                ).dict()
                return self._validate_output(
                    result,
                    (PrimActionResponse, ErrorResponse),
                    "update_prim_attributes",
                )

            except Exception as e:
                self.logger.error(
                    f"Error updating prim {input_data.stage_id}:{input_data.prim_path}: {e}"
                )
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (PrimActionResponse, ErrorResponse),
                    "update_prim_attributes",
                )

        @self.mcp.tool(
            name="delete_prim",
            description="Delete a prim from a USD stage.",
            annotations=self._tool_annotations(
                read_only=False, idempotent=False, open_world=False, destructive=True
            ),
            output_schema=self._tool_output_schema(PrimActionResponse, ErrorResponse),
        )
        async def delete_prim(stage_id: str, prim_path: str) -> Dict[str, Any]:
            rate_error = self._check_rate_limit("delete_prim")
            if rate_error:
                return rate_error

            input_data = self._validate_input(
                PrimDeleteRequest,
                stage_id=stage_id,
                prim_path=prim_path,
            )
            if isinstance(input_data, dict):
                return input_data

            try:
                adapter = self.headless_adapter or self.isaac_adapter
                if not adapter:
                    return ErrorResponse(
                        error="No USD adapter available", error_type="AdapterError"
                    ).dict()

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
                        ).dict()
                        return self._validate_output(
                            result, (PrimActionResponse, ErrorResponse), "delete_prim"
                        )

                result = ErrorResponse(
                    error=f"Failed to delete prim: {input_data.prim_path}",
                    error_type="DeleteError",
                ).dict()
                return self._validate_output(
                    result, (PrimActionResponse, ErrorResponse), "delete_prim"
                )

            except Exception as e:
                self.logger.error(
                    f"Error deleting prim {input_data.stage_id}:{input_data.prim_path}: {e}"
                )
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result, (PrimActionResponse, ErrorResponse), "delete_prim"
                )

        @self.mcp.tool(
            name="get_mesh_info",
            description="Get mesh information for a mesh prim.",
            annotations=self._tool_annotations(
                read_only=True, idempotent=True, open_world=False
            ),
            output_schema=self._tool_output_schema(MeshInfo, ErrorResponse),
        )
        async def get_mesh_info(stage_id: str, prim_path: str) -> Dict[str, Any]:
            rate_error = self._check_rate_limit("get_mesh_info")
            if rate_error:
                return rate_error

            input_data = self._validate_input(
                MeshInfoRequest, stage_id=stage_id, prim_path=prim_path
            )
            if isinstance(input_data, dict):
                return input_data

            try:
                adapter = self.headless_adapter or self.isaac_adapter
                if not adapter:
                    return ErrorResponse(
                        error="No USD adapter available", error_type="AdapterError"
                    ).dict()

                with adapter.create_session() as session:
                    mesh_info = session.get_mesh_info(
                        input_data.stage_id, input_data.prim_path
                    )
                    if mesh_info:
                        result = MeshInfo(**mesh_info).dict()
                        return self._validate_output(
                            result, (MeshInfo, ErrorResponse), "get_mesh_info"
                        )

                result = ErrorResponse(
                    error=f"Mesh not found: {input_data.prim_path}",
                    error_type="NotFoundError",
                ).dict()
                return self._validate_output(
                    result, (MeshInfo, ErrorResponse), "get_mesh_info"
                )

            except Exception as e:
                self.logger.error(
                    f"Error getting mesh info {input_data.stage_id}:{input_data.prim_path}: {e}"
                )
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result, (MeshInfo, ErrorResponse), "get_mesh_info"
                )

        @self.mcp.tool(
            name="search_prims",
            description="Search for prims in a USD stage.",
            annotations=self._tool_annotations(
                read_only=True, idempotent=True, open_world=False
            ),
            output_schema=self._tool_output_schema(PrimSearchResponse, ErrorResponse),
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
            rate_error = self._check_rate_limit("search_prims")
            if rate_error:
                return rate_error

            input_data = self._validate_input(
                PrimSearchRequest,
                stage_id=stage_id,
                search_type=search_type,
                query=query,
                exact_match=exact_match,
            )
            if isinstance(input_data, dict):
                return input_data

            try:
                adapter = self.headless_adapter or self.isaac_adapter
                if not adapter:
                    return ErrorResponse(
                        error="No USD adapter available", error_type="AdapterError"
                    ).dict()

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
                    ).dict()
                    return self._validate_output(
                        result, (PrimSearchResponse, ErrorResponse), "search_prims"
                    )

            except Exception as e:
                self.logger.error(f"Error searching prims {input_data.stage_id}: {e}")
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result, (PrimSearchResponse, ErrorResponse), "search_prims"
                )

        @self.mcp.tool(
            name="get_bounding_box",
            description="Get bounding box for a prim or entire stage.",
            annotations=self._tool_annotations(
                read_only=True, idempotent=True, open_world=False
            ),
            output_schema=self._tool_output_schema(BBoxResponse, ErrorResponse),
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
            rate_error = self._check_rate_limit("get_bounding_box")
            if rate_error:
                return rate_error

            input_data = self._validate_input(
                BBoxRequest,
                stage_id=stage_id,
                prim_path=prim_path,
                world_space=world_space,
                time_code=None,
            )
            if isinstance(input_data, dict):
                return input_data

            try:
                adapter = self.headless_adapter or self.isaac_adapter
                if not adapter:
                    return ErrorResponse(
                        error="No USD adapter available", error_type="AdapterError"
                    ).dict()

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
                        ).dict()
                        return self._validate_output(
                            result, (BBoxResponse, ErrorResponse), "get_bounding_box"
                        )
                    else:
                        result = ErrorResponse(
                            error="Could not compute bounding box",
                            error_type="ComputationError",
                        ).dict()
                        return self._validate_output(
                            result, (BBoxResponse, ErrorResponse), "get_bounding_box"
                        )

            except Exception as e:
                self.logger.error(
                    f"Error computing bounding box {input_data.stage_id}:{input_data.prim_path}: {e}"
                )
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result, (BBoxResponse, ErrorResponse), "get_bounding_box"
                )

        @self.mcp.tool(
            name="summarize_scene",
            description="Generate a summary of a USD scene.",
            annotations=self._tool_annotations(
                read_only=True, idempotent=True, open_world=False
            ),
            output_schema=self._tool_output_schema(SceneSummaryResponse, ErrorResponse),
            task=self._task_optional(),
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
            rate_error = self._check_rate_limit("summarize_scene")
            if rate_error:
                return rate_error

            input_data = self._validate_input(
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
                adapter = self.headless_adapter or self.isaac_adapter
                if not adapter:
                    return ErrorResponse(
                        error="No USD adapter available", error_type="AdapterError"
                    ).dict()

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
                        ).dict()
                        return self._validate_output(
                            result,
                            (SceneSummaryResponse, ErrorResponse),
                            "summarize_scene",
                        )
                    else:
                        result = ErrorResponse(
                            error="Could not generate scene summary",
                            error_type="ComputationError",
                        ).dict()
                        return self._validate_output(
                            result,
                            (SceneSummaryResponse, ErrorResponse),
                            "summarize_scene",
                        )

            except Exception as e:
                self.logger.error(f"Error summarizing scene {input_data.stage_id}: {e}")
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result, (SceneSummaryResponse, ErrorResponse), "summarize_scene"
                )

    def _register_isaac_tools(self) -> None:
        """Register Isaac Sim specific tools."""

        @self.mcp.tool(
            name="capture_viewport",
            description="Capture the Isaac Sim viewport.",
            annotations=self._tool_annotations(
                read_only=True, idempotent=False, open_world=True
            ),
            output_schema=self._tool_output_schema(
                ViewportCaptureResponse, ErrorResponse
            ),
            task=self._task_optional(),
        )
        async def capture_viewport(
            width: Optional[int] = None,
            height: Optional[int] = None,
            format: str = "png",
            save_to_file: bool = False,
            file_path: Optional[str] = None,
        ) -> Dict[str, Any]:
            """
            Capture the Isaac Sim viewport.

            Args:
                width: Image width
                height: Image height
                format: Image format (png, jpg, exr)
                save_to_file: Save image to file
                file_path: File path for saved image

            Returns:
                Viewport capture response or error
            """
            rate_error = self._check_rate_limit("capture_viewport")
            if rate_error:
                return rate_error

            input_data = self._validate_input(
                ViewportCaptureRequest,
                width=width,
                height=height,
                format=format,
                save_to_file=save_to_file,
                file_path=file_path,
            )
            if isinstance(input_data, dict):
                return input_data

            if input_data.save_to_file and input_data.file_path:
                if not self._is_path_allowed(input_data.file_path):
                    return ErrorResponse(
                        error="File path is not allowed by sandbox policy",
                        error_type="SandboxError",
                        details={"file_path": input_data.file_path},
                    ).dict()

            try:
                if not self.isaac_adapter or not self.isaac_adapter.is_available():
                    return ErrorResponse(
                        error="Isaac Sim runtime not available",
                        error_type="RuntimeError",
                    ).dict()

                with self.isaac_adapter.create_session() as session:
                    capture = session.capture_viewport(
                        width=input_data.width,
                        height=input_data.height,
                        format=input_data.format.value,
                        save_to_file=input_data.save_to_file,
                        file_path=input_data.file_path,
                    )

                    if capture:
                        result = ViewportCaptureResponse(
                            success=True,
                            width=capture.width,
                            height=capture.height,
                            format=capture.format,
                            file_path=capture.file_path,
                        ).dict()
                        return self._validate_output(
                            result,
                            (ViewportCaptureResponse, ErrorResponse),
                            "capture_viewport",
                        )
                    else:
                        result = ErrorResponse(
                            error="Failed to capture viewport",
                            error_type="CaptureError",
                        ).dict()
                        return self._validate_output(
                            result,
                            (ViewportCaptureResponse, ErrorResponse),
                            "capture_viewport",
                        )

            except Exception as e:
                self.logger.error(f"Error capturing viewport: {e}")
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result, (ViewportCaptureResponse, ErrorResponse), "capture_viewport"
                )

        @self.mcp.tool(
            name="get_viewport_info",
            description="Get information about the current viewport.",
            annotations=self._tool_annotations(
                read_only=True, idempotent=True, open_world=True
            ),
            output_schema=self._tool_output_schema(ViewportInfoResponse, ErrorResponse),
        )
        async def get_viewport_info() -> Dict[str, Any]:
            rate_error = self._check_rate_limit("get_viewport_info")
            if rate_error:
                return rate_error

            try:
                if not self.isaac_adapter or not self.isaac_adapter.is_available():
                    return ErrorResponse(
                        error="Isaac Sim runtime not available",
                        error_type="RuntimeError",
                    ).dict()

                with self.isaac_adapter.create_session() as session:
                    info = session.get_viewport_info()
                    info["success"] = True
                    result = ViewportInfoResponse(**info).dict()
                    return self._validate_output(
                        result,
                        (ViewportInfoResponse, ErrorResponse),
                        "get_viewport_info",
                    )

            except Exception as e:
                self.logger.error(f"Error getting viewport info: {e}")
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result, (ViewportInfoResponse, ErrorResponse), "get_viewport_info"
                )

        @self.mcp.tool(
            name="control_simulation",
            description="Control Isaac Sim simulation.",
            annotations=self._tool_annotations(
                read_only=False, idempotent=False, open_world=True, destructive=True
            ),
            output_schema=self._tool_output_schema(
                SimulationControlResponse, ErrorResponse
            ),
        )
        async def control_simulation(action: str, steps: int = 1) -> Dict[str, Any]:
            """
            Control Isaac Sim simulation.

            Args:
                action: Action (play, pause, stop, reset, step)
                steps: Number of steps (for step action)

            Returns:
                Success status or error
            """
            rate_error = self._check_rate_limit("control_simulation")
            if rate_error:
                return rate_error

            input_data = self._validate_input(
                SimulationControlRequest,
                action=action,
                steps=steps,
            )
            if isinstance(input_data, dict):
                return input_data

            try:
                if not self.isaac_adapter or not self.isaac_adapter.is_available():
                    return ErrorResponse(
                        error="Isaac Sim runtime not available",
                        error_type="RuntimeError",
                    ).dict()

                with self.isaac_adapter.create_session() as session:
                    # Initialize world if needed
                    if not session.get_world():
                        session.initialize_world()

                    success = False
                    if input_data.action == "play":
                        success = session.play_simulation()
                    elif input_data.action == "pause":
                        success = session.pause_simulation()
                    elif input_data.action == "stop":
                        success = session.stop_simulation()
                    elif input_data.action == "reset":
                        success = session.reset_simulation()
                    elif input_data.action == "step":
                        success = session.step_simulation(input_data.steps)

                    result = SimulationControlResponse(
                        success=success,
                        action=input_data.action,
                        steps=input_data.steps if input_data.action == "step" else None,
                        message=f"Simulation {input_data.action} {'successful' if success else 'failed'}",
                    ).dict()
                    return self._validate_output(
                        result,
                        (SimulationControlResponse, ErrorResponse),
                        "control_simulation",
                    )

            except Exception as e:
                self.logger.error(f"Error controlling simulation: {e}")
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (SimulationControlResponse, ErrorResponse),
                    "control_simulation",
                )

        @self.mcp.tool(
            name="get_simulation_status",
            description="Get current simulation status.",
            annotations=self._tool_annotations(
                read_only=True, idempotent=True, open_world=True
            ),
            output_schema=self._tool_output_schema(
                SimulationStatusResponse, ErrorResponse
            ),
        )
        async def get_simulation_status() -> Dict[str, Any]:
            rate_error = self._check_rate_limit("get_simulation_status")
            if rate_error:
                return rate_error

            try:
                if not self.isaac_adapter or not self.isaac_adapter.is_available():
                    return ErrorResponse(
                        error="Isaac Sim runtime not available",
                        error_type="RuntimeError",
                    ).dict()

                with self.isaac_adapter.create_session() as session:
                    status = session.get_simulation_status()
                    status["success"] = True
                    result = SimulationStatusResponse(**status).dict()
                    return self._validate_output(
                        result,
                        (SimulationStatusResponse, ErrorResponse),
                        "get_simulation_status",
                    )

            except Exception as e:
                self.logger.error(f"Error getting simulation status: {e}")
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (SimulationStatusResponse, ErrorResponse),
                    "get_simulation_status",
                )

        @self.mcp.tool(
            name="enable_rigid_body",
            description="Enable rigid body physics on a prim.",
            annotations=self._tool_annotations(
                read_only=False, idempotent=False, open_world=True, destructive=True
            ),
            output_schema=self._tool_output_schema(
                RigidBodyActionResponse, ErrorResponse
            ),
        )
        async def enable_rigid_body(
            prim_path: str, mass: Optional[float] = None
        ) -> Dict[str, Any]:
            """
            Enable rigid body physics on a prim.

            Args:
                prim_path: Prim path
                mass: Mass in kilograms

            Returns:
                Action status or error
            """
            rate_error = self._check_rate_limit("enable_rigid_body")
            if rate_error:
                return rate_error

            input_data = self._validate_input(
                RigidBodyEnableRequest,
                prim_path=prim_path,
                mass=mass,
            )
            if isinstance(input_data, dict):
                return input_data

            try:
                if not self.isaac_adapter or not self.isaac_adapter.is_available():
                    return ErrorResponse(
                        error="Isaac Sim runtime not available",
                        error_type="RuntimeError",
                    ).dict()

                with self.isaac_adapter.create_session() as session:
                    success = session.enable_rigid_body(
                        input_data.prim_path, input_data.mass
                    )
                    result = RigidBodyActionResponse(
                        success=success,
                        prim_path=input_data.prim_path,
                        message=f"Rigid body {'enabled' if success else 'not enabled'} for {input_data.prim_path}",
                    ).dict()
                    return self._validate_output(
                        result,
                        (RigidBodyActionResponse, ErrorResponse),
                        "enable_rigid_body",
                    )

            except Exception as e:
                self.logger.error(f"Error enabling rigid body: {e}")
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (RigidBodyActionResponse, ErrorResponse),
                    "enable_rigid_body",
                )

        @self.mcp.tool(
            name="set_rigid_body_velocity",
            description="Set rigid body linear or angular velocity.",
            annotations=self._tool_annotations(
                read_only=False, idempotent=False, open_world=True, destructive=True
            ),
            output_schema=self._tool_output_schema(
                RigidBodyActionResponse, ErrorResponse
            ),
        )
        async def set_rigid_body_velocity(
            prim_path: str,
            linear_velocity: Optional[List[float]] = None,
            angular_velocity: Optional[List[float]] = None,
        ) -> Dict[str, Any]:
            """
            Set rigid body linear or angular velocity.

            Args:
                prim_path: Prim path
                linear_velocity: Linear velocity [x, y, z]
                angular_velocity: Angular velocity [x, y, z]

            Returns:
                Action status or error
            """
            rate_error = self._check_rate_limit("set_rigid_body_velocity")
            if rate_error:
                return rate_error

            input_data = self._validate_input(
                RigidBodyVelocityRequest,
                prim_path=prim_path,
                linear_velocity=linear_velocity,
                angular_velocity=angular_velocity,
            )
            if isinstance(input_data, dict):
                return input_data

            try:
                if not self.isaac_adapter or not self.isaac_adapter.is_available():
                    return ErrorResponse(
                        error="Isaac Sim runtime not available",
                        error_type="RuntimeError",
                    ).dict()

                with self.isaac_adapter.create_session() as session:
                    success = session.set_rigid_body_velocity(
                        input_data.prim_path,
                        input_data.linear_velocity,
                        input_data.angular_velocity,
                    )
                    result = RigidBodyActionResponse(
                        success=success,
                        prim_path=input_data.prim_path,
                        message=f"Rigid body velocity {'updated' if success else 'not updated'} for {input_data.prim_path}",
                    ).dict()
                    return self._validate_output(
                        result,
                        (RigidBodyActionResponse, ErrorResponse),
                        "set_rigid_body_velocity",
                    )

            except Exception as e:
                self.logger.error(f"Error setting rigid body velocity: {e}")
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (RigidBodyActionResponse, ErrorResponse),
                    "set_rigid_body_velocity",
                )

        @self.mcp.tool(
            name="get_rigid_body_state",
            description="Get rigid body physics state for a prim.",
            annotations=self._tool_annotations(
                read_only=True, idempotent=True, open_world=True
            ),
            output_schema=self._tool_output_schema(
                RigidBodyStateResponse, ErrorResponse
            ),
        )
        async def get_rigid_body_state(prim_path: str) -> Dict[str, Any]:
            """
            Get rigid body physics state for a prim.

            Args:
                prim_path: Prim path

            Returns:
                Rigid body state or error
            """
            rate_error = self._check_rate_limit("get_rigid_body_state")
            if rate_error:
                return rate_error

            try:
                if not self.isaac_adapter or not self.isaac_adapter.is_available():
                    return ErrorResponse(
                        error="Isaac Sim runtime not available",
                        error_type="RuntimeError",
                    ).dict()

                with self.isaac_adapter.create_session() as session:
                    state = session.get_rigid_body_state(prim_path)
                    if not state:
                        result = ErrorResponse(
                            error=f"Rigid body not found: {prim_path}",
                            error_type="NotFoundError",
                        ).dict()
                        return self._validate_output(
                            result,
                            (RigidBodyStateResponse, ErrorResponse),
                            "get_rigid_body_state",
                        )

                    result = RigidBodyStateResponse(
                        success=True,
                        prim_path=state["prim_path"],
                        enabled=state["enabled"],
                        mass=state.get("mass"),
                        linear_velocity=state.get("linear_velocity"),
                        angular_velocity=state.get("angular_velocity"),
                    ).dict()
                    return self._validate_output(
                        result,
                        (RigidBodyStateResponse, ErrorResponse),
                        "get_rigid_body_state",
                    )

            except Exception as e:
                self.logger.error(f"Error getting rigid body state: {e}")
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (RigidBodyStateResponse, ErrorResponse),
                    "get_rigid_body_state",
                )

        @self.mcp.tool(
            name="set_camera_view",
            description="Set camera view in the viewport.",
            annotations=self._tool_annotations(
                read_only=False, idempotent=False, open_world=True
            ),
            output_schema=self._tool_output_schema(CameraViewResponse, ErrorResponse),
        )
        async def set_camera_view(
            eye: List[float],
            target: List[float],
            up: List[float] = [0, 1, 0],
        ) -> Dict[str, Any]:
            rate_error = self._check_rate_limit("set_camera_view")
            if rate_error:
                return rate_error

            input_data = self._validate_input(
                CameraViewRequest, eye=eye, target=target, up=up
            )
            if isinstance(input_data, dict):
                return input_data

            try:
                if not self.isaac_adapter or not self.isaac_adapter.is_available():
                    return ErrorResponse(
                        error="Isaac Sim runtime not available",
                        error_type="RuntimeError",
                    ).dict()

                with self.isaac_adapter.create_session() as session:
                    success = session.set_camera_view(
                        eye=(input_data.eye[0], input_data.eye[1], input_data.eye[2]),
                        target=(
                            input_data.target[0],
                            input_data.target[1],
                            input_data.target[2],
                        ),
                        up=(input_data.up[0], input_data.up[1], input_data.up[2]),
                    )

                    result = CameraViewResponse(
                        success=success,
                        eye=input_data.eye,
                        target=input_data.target,
                        up=input_data.up,
                        message=f"Camera view {'set successfully' if success else 'failed to set'}",
                    ).dict()
                    return self._validate_output(
                        result, (CameraViewResponse, ErrorResponse), "set_camera_view"
                    )

            except Exception as e:
                self.logger.error(f"Error setting camera view: {e}")
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result, (CameraViewResponse, ErrorResponse), "set_camera_view"
                )

        @self.mcp.tool(
            name="get_camera_info",
            description="Get information about the current camera.",
            annotations=self._tool_annotations(
                read_only=True, idempotent=True, open_world=True
            ),
            output_schema=self._tool_output_schema(CameraInfoResponse, ErrorResponse),
        )
        async def get_camera_info() -> Dict[str, Any]:
            rate_error = self._check_rate_limit("get_camera_info")
            if rate_error:
                return rate_error

            try:
                if not self.isaac_adapter or not self.isaac_adapter.is_available():
                    return ErrorResponse(
                        error="Isaac Sim runtime not available",
                        error_type="RuntimeError",
                    ).dict()

                with self.isaac_adapter.create_session() as session:
                    info = session.get_camera_info()
                    info["success"] = True
                    info["can_control"] = info.get("camera_available", False)
                    result = CameraInfoResponse(**info).dict()
                    return self._validate_output(
                        result, (CameraInfoResponse, ErrorResponse), "get_camera_info"
                    )

            except Exception as e:
                self.logger.error(f"Error getting camera info: {e}")
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result, (CameraInfoResponse, ErrorResponse), "get_camera_info"
                )

        @self.mcp.tool(
            name="focus_on_prim",
            description="Focus camera on a specific prim.",
            annotations=self._tool_annotations(
                read_only=False, idempotent=False, open_world=True
            ),
            output_schema=self._tool_output_schema(FocusPrimResponse, ErrorResponse),
        )
        async def focus_on_prim(stage_id: str, prim_path: str) -> Dict[str, Any]:
            rate_error = self._check_rate_limit("focus_on_prim")
            if rate_error:
                return rate_error

            input_data = self._validate_input(
                FocusPrimRequest, stage_id=stage_id, prim_path=prim_path
            )
            if isinstance(input_data, dict):
                return input_data

            try:
                if not self.isaac_adapter or not self.isaac_adapter.is_available():
                    return ErrorResponse(
                        error="Isaac Sim runtime not available",
                        error_type="RuntimeError",
                    ).dict()

                with self.isaac_adapter.create_session() as session:
                    bbox_dict = session.get_prim_bbox(
                        input_data.stage_id, input_data.prim_path, world_space=True
                    )
                    if not bbox_dict:
                        result = ErrorResponse(
                            error=f"Could not get bounding box for prim: {input_data.prim_path}",
                            error_type="ComputationError",
                        ).dict()
                        return self._validate_output(
                            result, (FocusPrimResponse, ErrorResponse), "focus_on_prim"
                        )

                    min_point = bbox_dict["min"]
                    max_point = bbox_dict["max"]
                    center = [
                        (min_point[0] + max_point[0]) / 2,
                        (min_point[1] + max_point[1]) / 2,
                        (min_point[2] + max_point[2]) / 2,
                    ]
                    size = [
                        max_point[0] - min_point[0],
                        max_point[1] - min_point[1],
                        max_point[2] - min_point[2],
                    ]
                    max_size = max(size)
                    distance = max_size * 2.0
                    eye = [
                        center[0] + distance,
                        center[1] + distance,
                        center[2] + distance,
                    ]

                    success = session.set_camera_view(
                        eye=(eye[0], eye[1], eye[2]),
                        target=(center[0], center[1], center[2]),
                        up=(0, 1, 0),
                    )

                    result = FocusPrimResponse(
                        success=success,
                        stage_id=input_data.stage_id,
                        prim_path=input_data.prim_path,
                        focus_point=center,
                        camera_position=eye,
                        message=f"Camera {'focused on' if success else 'failed to focus on'} {input_data.prim_path}",
                    ).dict()
                    return self._validate_output(
                        result, (FocusPrimResponse, ErrorResponse), "focus_on_prim"
                    )

            except Exception as e:
                self.logger.error(
                    f"Error focusing on prim {input_data.stage_id}:{input_data.prim_path}: {e}"
                )
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result, (FocusPrimResponse, ErrorResponse), "focus_on_prim"
                )

    def _register_blender_tools(self) -> None:
        """Register Blender runtime specific tools."""

        @self.mcp.tool(
            name="get_blender_info",
            description="Get information about the active Blender runtime.",
            annotations=self._tool_annotations(
                read_only=True,
                idempotent=True,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(BlenderInfoResponse, ErrorResponse),
            task=self._task_optional(),
        )
        async def get_blender_info() -> Dict[str, Any]:
            """
            Get information about the active Blender runtime.

            Returns:
                Blender runtime information or an error response.
            """
            rate_error = self._check_rate_limit("get_blender_info")
            if rate_error:
                return rate_error

            try:
                if not self.blender_adapter or not self.blender_adapter.is_available():
                    return ErrorResponse(
                        error="Blender runtime not available",
                        error_type="RuntimeError",
                    ).dict()

                with self.blender_adapter.create_session() as session:
                    runtime_info = session.get_runtime_info()
                    runtime_info["success"] = True
                    result = BlenderInfoResponse(**runtime_info).dict()
                    return self._validate_output(
                        result,
                        (BlenderInfoResponse, ErrorResponse),
                        "get_blender_info",
                    )

            except Exception as e:
                self.logger.error(f"Error getting Blender runtime info: {e}")
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (BlenderInfoResponse, ErrorResponse),
                    "get_blender_info",
                )

        @self.mcp.tool(
            name="list_blender_scene_objects",
            description="List objects from the active Blender scene.",
            annotations=self._tool_annotations(
                read_only=True,
                idempotent=True,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                BlenderSceneObjectsResponse,
                ErrorResponse,
            ),
            task=self._task_optional(),
        )
        async def list_blender_scene_objects(
            collection_name: Optional[str] = None,
            include_hidden: bool = False,
            max_items: int = self.settings.blender.max_scene_objects,
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
            rate_error = self._check_rate_limit("list_blender_scene_objects")
            if rate_error:
                return rate_error

            input_data = self._validate_input(
                BlenderSceneObjectsRequest,
                collection_name=collection_name,
                include_hidden=include_hidden,
                max_items=max_items,
            )
            if isinstance(input_data, dict):
                return input_data

            try:
                if not self.blender_adapter or not self.blender_adapter.is_available():
                    return ErrorResponse(
                        error="Blender runtime not available",
                        error_type="RuntimeError",
                    ).dict()

                with self.blender_adapter.create_session() as session:
                    objects_payload = session.list_scene_objects(
                        collection_name=input_data.collection_name,
                        include_hidden=input_data.include_hidden,
                        max_items=input_data.max_items,
                    )
                    objects_payload["success"] = True
                    result = BlenderSceneObjectsResponse(**objects_payload).dict()
                    return self._validate_output(
                        result,
                        (BlenderSceneObjectsResponse, ErrorResponse),
                        "list_blender_scene_objects",
                    )

            except Exception as e:
                self.logger.error(f"Error listing Blender scene objects: {e}")
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (BlenderSceneObjectsResponse, ErrorResponse),
                    "list_blender_scene_objects",
                )

    async def run(self, transport: str = "stdio") -> None:
        """
        Run the MCP server.

        Args:
            transport: Transport type (stdio, sse)
        """
        try:
            self.logger.info(f"Starting Isaac MCP Server with {transport} transport")

            if transport == "stdio":
                await self.mcp.run()
            elif transport == "sse":
                run_sse = getattr(self.mcp, "run_sse", None)
                if callable(run_sse):
                    result = run_sse(
                        host=self.settings.server.host, port=self.settings.server.port
                    )
                    if inspect.isawaitable(result):
                        await result
                else:
                    raise ValueError("SSE transport not supported by FastMCP")
            else:
                raise ValueError(f"Unsupported transport: {transport}")

        except Exception as e:
            self.logger.error(f"Error running MCP server: {e}")
            raise

    def get_capabilities(self) -> List[str]:
        """Get list of server capabilities."""
        capabilities = []

        if self.headless_adapter and self.headless_adapter.is_available():
            capabilities.extend(self.headless_adapter.get_capabilities())

        if self.isaac_adapter and self.isaac_adapter.is_available():
            capabilities.extend(self.isaac_adapter.get_capabilities())

        if self.blender_adapter and self.blender_adapter.is_available():
            capabilities.extend(self.blender_adapter.get_capabilities())

        return list(set(capabilities))  # Remove duplicates


# Convenience functions
def create_server_instance(settings: Optional[Settings] = None) -> IsaacMCPServer:
    """
    Create an Isaac MCP Server instance.

    Args:
        settings: Configuration settings

    Returns:
        IsaacMCPServer instance
    """
    return IsaacMCPServer(settings)


async def start_mcp_server(
    settings: Optional[Settings] = None, transport: str = "stdio"
) -> None:
    """
    Start the Isaac MCP Server.

    Args:
        settings: Configuration settings
        transport: Transport type
    """
    server = create_server_instance(settings)
    await server.run(transport)
