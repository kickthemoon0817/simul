"""
Isaac Sim MCP Server implementation.

This module provides the main MCP server class with tool registry,
connection management, and Isaac Sim integration based on FastMCP.
"""

import asyncio
import inspect
import json
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
    UnrealRuntimeAdapter,
    is_blender_available,
    is_headless_available,
    is_unreal_available,
)
from ..adapters.isaac_socket_client import IsaacSocketClient, ScriptResult
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
        self.client = IsaacSocketClient(
            host="127.0.0.1",
            port=8226,
        )
        self.blender_adapter = (
            BlenderRuntimeAdapter(self.settings) if is_blender_available() else None
        )
        self.unreal_adapter = (
            UnrealRuntimeAdapter(self.settings) if is_unreal_available() else None
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

        # Isaac Sim tools — always registered; connection checked at runtime
        self._register_isaac_tools()

        # Blender specific tools (if available)
        if self.blender_adapter and self.blender_adapter.is_available():
            self._register_blender_tools()

        # Unreal Engine specific tools (if available)
        if self.unreal_adapter and self.unreal_adapter.is_available():
            self._register_unreal_tools()

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
                adapter = self.headless_adapter
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
                adapter = self.headless_adapter
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
                adapter = self.headless_adapter
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
                adapter = self.headless_adapter
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
                adapter = self.headless_adapter
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
                adapter = self.headless_adapter
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
                adapter = self.headless_adapter
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
                adapter = self.headless_adapter
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
                adapter = self.headless_adapter
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
        """Register Isaac Sim tools that execute via TCP socket on port 8226."""

        @self.mcp.tool(
            name="execute_isaac_script",
            description=(
                "Execute arbitrary Python code inside a running Isaac Sim application. "
                "The code runs in Kit's Python scope with full access to omni.*, pxr.*, "
                "and isaacsim.* APIs. stdout is captured and returned. For structured "
                "results, print JSON via json.dumps()."
            ),
            annotations=self._tool_annotations(
                read_only=False, idempotent=False, open_world=True, destructive=True
            ),
        )
        async def execute_isaac_script(code: str) -> Dict[str, Any]:
            """
            Execute Python code inside the running Isaac Sim process.

            The code is sent over TCP to the stock isaacsim.code_editor.vscode
            extension (port 8226). stdout is captured and returned.

            Args:
                code: Python source code to execute in Isaac Sim.

            Returns:
                Dict with success, output, and optional error info.
            """
            MAX_CODE_SIZE: int = 100_000  # 100 KB
            if len(code) > MAX_CODE_SIZE:
                return ErrorResponse(
                    error=f"Code payload too large ({len(code)} bytes, max {MAX_CODE_SIZE}).",
                    error_type="PayloadTooLarge",
                ).dict()

            rate_error = self._check_rate_limit("execute_isaac_script")
            if rate_error:
                return rate_error
            try:
                result: ScriptResult = await self.client.execute(code)
                if not result.success:
                    return ErrorResponse(
                        error=result.error_value or "Script execution failed",
                        error_type=result.error_name or "RuntimeError",
                        details={"traceback": result.traceback} if result.traceback else None,
                    ).dict()

                # If output is valid JSON, return it directly
                output = result.output.strip()
                if output:
                    try:
                        return json.loads(output)
                    except json.JSONDecodeError:
                        pass

                return {"success": True, "output": result.output}

            except ConnectionRefusedError:
                return ErrorResponse(
                    error="Isaac Sim is not reachable on 127.0.0.1:8226. Is it running?",
                    error_type="ConnectionError",
                ).dict()
            except TimeoutError:
                return ErrorResponse(
                    error="Script execution timed out.",
                    error_type="TimeoutError",
                ).dict()
            except Exception as exc:
                return ErrorResponse(error=str(exc), error_type="Exception").dict()

        @self.mcp.tool(
            name="ping_isaac",
            description="Check if a running Isaac Sim instance is reachable.",
            annotations=self._tool_annotations(
                read_only=True, idempotent=True, open_world=True
            ),
        )
        async def ping_isaac() -> Dict[str, Any]:
            """
            Ping Isaac Sim to verify connectivity.

            Returns:
                Dict with reachable status and address.
            """
            reachable = await self.client.ping()
            return {
                "reachable": reachable,
                "address": self.client.address,
            }



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

    def _register_unreal_tools(self) -> None:
        """Register Unreal Engine runtime specific tools."""

        @self.mcp.tool(
            name="unreal_health_check",
            description="Check connectivity to the Unreal Engine Remote Control API.",
            annotations=self._tool_annotations(
                read_only=True,
                idempotent=True,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealHealthCheckResponse, ErrorResponse
            ),
            task=self._task_optional(),
        )
        async def unreal_health_check() -> Dict[str, Any]:
            """
            Check connectivity to the Unreal Engine Remote Control API.

            Returns:
                Connection status or error response.
            """
            rate_error = self._check_rate_limit("unreal_health_check")
            if rate_error:
                return rate_error

            try:
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()

                with self.unreal_adapter.create_session() as session:
                    payload = await session.health_check()
                    payload["success"] = True
                    result = UnrealHealthCheckResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealHealthCheckResponse, ErrorResponse),
                        "unreal_health_check",
                    )

            except Exception as e:
                self.logger.error("Error in Unreal health check: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealHealthCheckResponse, ErrorResponse),
                    "unreal_health_check",
                )

        @self.mcp.tool(
            name="get_unreal_engine_info",
            description="Get Unreal Engine runtime information.",
            annotations=self._tool_annotations(
                read_only=True,
                idempotent=True,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealEngineInfoResponse, ErrorResponse
            ),
            task=self._task_optional(),
        )
        async def get_unreal_engine_info() -> Dict[str, Any]:
            """
            Get Unreal Engine runtime information.

            Returns:
                Engine info or error response.
            """
            rate_error = self._check_rate_limit("get_unreal_engine_info")
            if rate_error:
                return rate_error

            try:
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()

                with self.unreal_adapter.create_session() as session:
                    payload = await session.get_engine_info()
                    payload["success"] = True
                    result = UnrealEngineInfoResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealEngineInfoResponse, ErrorResponse),
                        "get_unreal_engine_info",
                    )

            except Exception as e:
                self.logger.error("Error getting Unreal engine info: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealEngineInfoResponse, ErrorResponse),
                    "get_unreal_engine_info",
                )

        @self.mcp.tool(
            name="get_unreal_loaded_map",
            description="Get the currently loaded persistent level path.",
            annotations=self._tool_annotations(
                read_only=True,
                idempotent=True,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealLoadedMapResponse, ErrorResponse
            ),
            task=self._task_optional(),
        )
        async def get_unreal_loaded_map() -> Dict[str, Any]:
            """
            Get the currently loaded persistent level path.

            Returns:
                Loaded map path or error response.
            """
            rate_error = self._check_rate_limit("get_unreal_loaded_map")
            if rate_error:
                return rate_error

            try:
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()

                with self.unreal_adapter.create_session() as session:
                    payload = await session.get_loaded_map()
                    payload["success"] = True
                    result = UnrealLoadedMapResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealLoadedMapResponse, ErrorResponse),
                        "get_unreal_loaded_map",
                    )

            except Exception as e:
                self.logger.error("Error getting Unreal loaded map: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealLoadedMapResponse, ErrorResponse),
                    "get_unreal_loaded_map",
                )

        # ------------------------------------------------------------------
        # Phase 1: Scene Read Operations
        # ------------------------------------------------------------------

        @self.mcp.tool(
            name="list_unreal_actors",
            description="List actors in the current Unreal Engine level with optional class and tag filters.",
            annotations=self._tool_annotations(
                read_only=True,
                idempotent=True,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealListActorsResponse, ErrorResponse
            ),
            task=self._task_optional(),
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
            rate_error = self._check_rate_limit("list_unreal_actors")
            if rate_error:
                return rate_error

            try:
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()

                with self.unreal_adapter.create_session() as session:
                    payload = await session.list_actors(
                        class_filter=class_filter or None,
                        tag_filter=tag_filter or None,
                        max_results=max_results,
                    )
                    payload["success"] = True
                    result = UnrealListActorsResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealListActorsResponse, ErrorResponse),
                        "list_unreal_actors",
                    )

            except Exception as e:
                self.logger.error("Error listing Unreal actors: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealListActorsResponse, ErrorResponse),
                    "list_unreal_actors",
                )

        @self.mcp.tool(
            name="get_unreal_actor_info",
            description="Get detailed information about a specific actor including transform, components, and tags.",
            annotations=self._tool_annotations(
                read_only=True,
                idempotent=True,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealGetActorInfoResponse, ErrorResponse
            ),
            task=self._task_optional(),
        )
        async def get_unreal_actor_info(actor_path: str) -> Dict[str, Any]:
            """
            Get detailed information about a specific actor.

            Args:
                actor_path: Full object path of the actor.

            Returns:
                Actor info or error response.
            """
            rate_error = self._check_rate_limit("get_unreal_actor_info")
            if rate_error:
                return rate_error

            try:
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()

                with self.unreal_adapter.create_session() as session:
                    payload = await session.get_actor_info(actor_path)
                    payload["success"] = True
                    result = UnrealGetActorInfoResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealGetActorInfoResponse, ErrorResponse),
                        "get_unreal_actor_info",
                    )

            except Exception as e:
                self.logger.error("Error getting Unreal actor info: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealGetActorInfoResponse, ErrorResponse),
                    "get_unreal_actor_info",
                )

        @self.mcp.tool(
            name="search_unreal_assets",
            description="Search the Unreal Asset Registry by name, class, or package path.",
            annotations=self._tool_annotations(
                read_only=True,
                idempotent=True,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealSearchAssetsResponse, ErrorResponse
            ),
            task=self._task_optional(),
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
            rate_error = self._check_rate_limit("search_unreal_assets")
            if rate_error:
                return rate_error

            try:
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()

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

                with self.unreal_adapter.create_session() as session:
                    payload = await session.search_assets(
                        query=query,
                        class_names=parsed_classes,
                        package_paths=parsed_paths,
                        max_results=max_results,
                    )
                    payload["success"] = True
                    result = UnrealSearchAssetsResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealSearchAssetsResponse, ErrorResponse),
                        "search_unreal_assets",
                    )

            except Exception as e:
                self.logger.error("Error searching Unreal assets: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealSearchAssetsResponse, ErrorResponse),
                    "search_unreal_assets",
                )

        @self.mcp.tool(
            name="describe_unreal_object",
            description="Get full property and function metadata for any UObject by path.",
            annotations=self._tool_annotations(
                read_only=True,
                idempotent=True,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealDescribeObjectResponse, ErrorResponse
            ),
            task=self._task_optional(),
        )
        async def describe_unreal_object(object_path: str) -> Dict[str, Any]:
            """
            Describe a UObject's properties and functions.

            Args:
                object_path: Full object path.

            Returns:
                Object description or error response.
            """
            rate_error = self._check_rate_limit("describe_unreal_object")
            if rate_error:
                return rate_error

            try:
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()

                with self.unreal_adapter.create_session() as session:
                    payload = await session.describe_object(object_path)
                    payload["success"] = True
                    result = UnrealDescribeObjectResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealDescribeObjectResponse, ErrorResponse),
                        "describe_unreal_object",
                    )

            except Exception as e:
                self.logger.error("Error describing Unreal object: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealDescribeObjectResponse, ErrorResponse),
                    "describe_unreal_object",
                )

        @self.mcp.tool(
            name="get_unreal_actor_thumbnail",
            description="Get a thumbnail image for an Unreal asset.",
            annotations=self._tool_annotations(
                read_only=True,
                idempotent=False,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealGetThumbnailResponse, ErrorResponse
            ),
            task=self._task_optional(),
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
            rate_error = self._check_rate_limit("get_unreal_actor_thumbnail")
            if rate_error:
                return rate_error

            try:
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()

                with self.unreal_adapter.create_session() as session:
                    payload = await session.get_actor_thumbnail(
                        asset_path=asset_path, width=width, height=height
                    )
                    payload["success"] = True
                    result = UnrealGetThumbnailResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealGetThumbnailResponse, ErrorResponse),
                        "get_unreal_actor_thumbnail",
                    )

            except Exception as e:
                self.logger.error("Error getting Unreal thumbnail: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealGetThumbnailResponse, ErrorResponse),
                    "get_unreal_actor_thumbnail",
                )

        @self.mcp.tool(
            name="summarize_unreal_scene",
            description="Generate an LLM-friendly digest of the current Unreal scene.",
            annotations=self._tool_annotations(
                read_only=True,
                idempotent=True,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealSceneSummaryResponse, ErrorResponse
            ),
            task=self._task_optional(),
        )
        async def summarize_unreal_scene() -> Dict[str, Any]:
            """
            Generate an LLM-friendly scene digest.

            Returns:
                Scene summary or error response.
            """
            rate_error = self._check_rate_limit("summarize_unreal_scene")
            if rate_error:
                return rate_error

            try:
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()

                with self.unreal_adapter.create_session() as session:
                    payload = await session.summarize_scene()
                    payload["success"] = True
                    result = UnrealSceneSummaryResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealSceneSummaryResponse, ErrorResponse),
                        "summarize_unreal_scene",
                    )

            except Exception as e:
                self.logger.error("Error summarizing Unreal scene: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealSceneSummaryResponse, ErrorResponse),
                    "summarize_unreal_scene",
                )

        # -- Phase 2: Viewport & Visual Observation --

        @self.mcp.tool(
            name="capture_unreal_viewport",
            description="Capture a viewport screenshot via HighResScreenshot.",
            annotations=self._tool_annotations(
                read_only=True,
                idempotent=False,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealCaptureViewportResponse, ErrorResponse
            ),
            task=self._task_optional(),
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
            rate_error = self._check_rate_limit("capture_unreal_viewport")
            if rate_error:
                return rate_error

            try:
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()

                with self.unreal_adapter.create_session() as session:
                    payload = await session.capture_viewport(
                        resolution_x=resolution_x,
                        resolution_y=resolution_y,
                        format=format,
                    )
                    payload["success"] = True
                    result = UnrealCaptureViewportResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealCaptureViewportResponse, ErrorResponse),
                        "capture_unreal_viewport",
                    )

            except Exception as e:
                self.logger.error("Error capturing Unreal viewport: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealCaptureViewportResponse, ErrorResponse),
                    "capture_unreal_viewport",
                )

        @self.mcp.tool(
            name="get_unreal_viewport_info",
            description="Get active viewport camera and render information.",
            annotations=self._tool_annotations(
                read_only=True,
                idempotent=True,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealViewportInfoResponse, ErrorResponse
            ),
            task=self._task_optional(),
        )
        async def get_unreal_viewport_info() -> Dict[str, Any]:
            """
            Get viewport camera and render settings.

            Returns:
                Viewport info or error response.
            """
            rate_error = self._check_rate_limit("get_unreal_viewport_info")
            if rate_error:
                return rate_error

            try:
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()

                with self.unreal_adapter.create_session() as session:
                    payload = await session.get_viewport_info()
                    payload["success"] = True
                    result = UnrealViewportInfoResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealViewportInfoResponse, ErrorResponse),
                        "get_unreal_viewport_info",
                    )

            except Exception as e:
                self.logger.error("Error getting Unreal viewport info: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealViewportInfoResponse, ErrorResponse),
                    "get_unreal_viewport_info",
                )

        @self.mcp.tool(
            name="set_unreal_camera_view",
            description="Set the editor viewport camera position and rotation.",
            annotations=self._tool_annotations(
                read_only=False,
                idempotent=True,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealSetCameraViewResponse, ErrorResponse
            ),
            task=self._task_optional(),
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
            rate_error = self._check_rate_limit("set_unreal_camera_view")
            if rate_error:
                return rate_error

            try:
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()

                with self.unreal_adapter.create_session() as session:
                    payload = await session.set_camera_view(
                        location=(location_x, location_y, location_z),
                        rotation=(rotation_pitch, rotation_yaw, rotation_roll),
                        fov=fov,
                    )
                    payload["success"] = True
                    result = UnrealSetCameraViewResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealSetCameraViewResponse, ErrorResponse),
                        "set_unreal_camera_view",
                    )

            except Exception as e:
                self.logger.error("Error setting Unreal camera view: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealSetCameraViewResponse, ErrorResponse),
                    "set_unreal_camera_view",
                )

        @self.mcp.tool(
            name="focus_unreal_on_actor",
            description="Focus the editor viewport camera on a specific actor.",
            annotations=self._tool_annotations(
                read_only=False,
                idempotent=False,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealFocusActorResponse, ErrorResponse
            ),
            task=self._task_optional(),
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
            rate_error = self._check_rate_limit("focus_unreal_on_actor")
            if rate_error:
                return rate_error

            try:
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()

                with self.unreal_adapter.create_session() as session:
                    payload = await session.focus_on_actor(
                        actor_path=actor_path, distance=distance
                    )
                    payload["success"] = True
                    result = UnrealFocusActorResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealFocusActorResponse, ErrorResponse),
                        "focus_unreal_on_actor",
                    )

            except Exception as e:
                self.logger.error("Error focusing on Unreal actor: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealFocusActorResponse, ErrorResponse),
                    "focus_unreal_on_actor",
                )

        # -- Phase 3: Scene Manipulation --

        @self.mcp.tool(
            name="spawn_unreal_actor",
            description="Spawn an actor from a class or asset path.",
            annotations=self._tool_annotations(
                read_only=False,
                idempotent=False,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealSpawnActorResponse, ErrorResponse
            ),
            task=self._task_optional(),
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
            rate_error = self._check_rate_limit("spawn_unreal_actor")
            if rate_error:
                return rate_error
            try:
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()
                with self.unreal_adapter.create_session() as session:
                    payload = await session.spawn_actor(
                        asset_path=asset_path,
                        location=(location_x, location_y, location_z),
                        rotation=(rotation_pitch, rotation_yaw, rotation_roll),
                        label=label or None,
                    )
                    payload["success"] = True
                    result = UnrealSpawnActorResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealSpawnActorResponse, ErrorResponse),
                        "spawn_unreal_actor",
                    )
            except Exception as e:
                self.logger.error("Error spawning Unreal actor: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealSpawnActorResponse, ErrorResponse),
                    "spawn_unreal_actor",
                )

        @self.mcp.tool(
            name="delete_unreal_actor",
            description="Delete an actor from the level. DESTRUCTIVE operation.",
            annotations=self._tool_annotations(
                read_only=False,
                idempotent=False,
                open_world=True,
                destructive=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealDeleteActorResponse, ErrorResponse
            ),
            task=self._task_optional(),
        )
        async def delete_unreal_actor(
            actor_path: str,
        ) -> Dict[str, Any]:
            """Delete an actor from the level."""
            rate_error = self._check_rate_limit("delete_unreal_actor")
            if rate_error:
                return rate_error
            try:
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()
                with self.unreal_adapter.create_session() as session:
                    payload = await session.delete_actor(actor_path=actor_path)
                    payload["success"] = True
                    result = UnrealDeleteActorResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealDeleteActorResponse, ErrorResponse),
                        "delete_unreal_actor",
                    )
            except Exception as e:
                self.logger.error("Error deleting Unreal actor: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealDeleteActorResponse, ErrorResponse),
                    "delete_unreal_actor",
                )

        @self.mcp.tool(
            name="set_unreal_actor_transform",
            description="Set an actor's location, rotation, and scale.",
            annotations=self._tool_annotations(
                read_only=False,
                idempotent=True,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealSetActorTransformResponse, ErrorResponse
            ),
            task=self._task_optional(),
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
            rate_error = self._check_rate_limit("set_unreal_actor_transform")
            if rate_error:
                return rate_error
            try:
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()
                with self.unreal_adapter.create_session() as session:
                    payload = await session.set_actor_transform(
                        actor_path=actor_path,
                        location=(location_x, location_y, location_z),
                        rotation=(rotation_pitch, rotation_yaw, rotation_roll),
                        scale=(scale_x, scale_y, scale_z),
                    )
                    payload["success"] = True
                    result = UnrealSetActorTransformResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealSetActorTransformResponse, ErrorResponse),
                        "set_unreal_actor_transform",
                    )
            except Exception as e:
                self.logger.error("Error setting Unreal actor transform: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealSetActorTransformResponse, ErrorResponse),
                    "set_unreal_actor_transform",
                )

        @self.mcp.tool(
            name="set_unreal_actor_property",
            description="Set a property on an Unreal actor by name and JSON value.",
            annotations=self._tool_annotations(
                read_only=False,
                idempotent=True,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealSetActorPropertyResponse, ErrorResponse
            ),
            task=self._task_optional(),
        )
        async def set_unreal_actor_property(
            actor_path: str,
            property_name: str,
            property_value: str,
            generate_transaction: bool = True,
        ) -> Dict[str, Any]:
            """Set a property on an actor."""
            rate_error = self._check_rate_limit("set_unreal_actor_property")
            if rate_error:
                return rate_error
            try:
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()
                with self.unreal_adapter.create_session() as session:
                    payload = await session.set_actor_property(
                        actor_path=actor_path,
                        property_name=property_name,
                        property_value=property_value,
                        generate_transaction=generate_transaction,
                    )
                    payload["success"] = True
                    result = UnrealSetActorPropertyResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealSetActorPropertyResponse, ErrorResponse),
                        "set_unreal_actor_property",
                    )
            except Exception as e:
                self.logger.error("Error setting Unreal actor property: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealSetActorPropertyResponse, ErrorResponse),
                    "set_unreal_actor_property",
                )

        @self.mcp.tool(
            name="call_unreal_actor_function",
            description="Call a BlueprintCallable UFUNCTION on an actor.",
            annotations=self._tool_annotations(
                read_only=False,
                idempotent=False,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealCallActorFunctionResponse, ErrorResponse
            ),
            task=self._task_optional(),
        )
        async def call_unreal_actor_function(
            actor_path: str,
            function_name: str,
            parameters: str = "",
        ) -> Dict[str, Any]:
            """Call a UFUNCTION on an actor."""
            rate_error = self._check_rate_limit("call_unreal_actor_function")
            if rate_error:
                return rate_error
            try:
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()
                with self.unreal_adapter.create_session() as session:
                    payload = await session.call_actor_function(
                        actor_path=actor_path,
                        function_name=function_name,
                        parameters=parameters or None,
                    )
                    payload["success"] = True
                    result = UnrealCallActorFunctionResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealCallActorFunctionResponse, ErrorResponse),
                        "call_unreal_actor_function",
                    )
            except Exception as e:
                self.logger.error("Error calling Unreal actor function: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealCallActorFunctionResponse, ErrorResponse),
                    "call_unreal_actor_function",
                )

        @self.mcp.tool(
            name="set_unreal_actor_parent",
            description="Attach an actor to a parent actor or detach it.",
            annotations=self._tool_annotations(
                read_only=False,
                idempotent=False,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealSetActorParentResponse, ErrorResponse
            ),
            task=self._task_optional(),
        )
        async def set_unreal_actor_parent(
            actor_path: str,
            parent_path: str = "",
        ) -> Dict[str, Any]:
            """Attach or detach an actor."""
            rate_error = self._check_rate_limit("set_unreal_actor_parent")
            if rate_error:
                return rate_error
            try:
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()
                with self.unreal_adapter.create_session() as session:
                    payload = await session.set_actor_parent(
                        actor_path=actor_path,
                        parent_path=parent_path or None,
                    )
                    payload["success"] = True
                    result = UnrealSetActorParentResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealSetActorParentResponse, ErrorResponse),
                        "set_unreal_actor_parent",
                    )
            except Exception as e:
                self.logger.error("Error setting Unreal actor parent: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealSetActorParentResponse, ErrorResponse),
                    "set_unreal_actor_parent",
                )

        @self.mcp.tool(
            name="add_unreal_component",
            description="Add a component to an Unreal actor.",
            annotations=self._tool_annotations(
                read_only=False,
                idempotent=False,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealAddComponentResponse, ErrorResponse
            ),
            task=self._task_optional(),
        )
        async def add_unreal_component(
            actor_path: str,
            component_class: str,
            component_name: str = "",
        ) -> Dict[str, Any]:
            """Add a component to an actor."""
            rate_error = self._check_rate_limit("add_unreal_component")
            if rate_error:
                return rate_error
            try:
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()
                with self.unreal_adapter.create_session() as session:
                    payload = await session.add_component(
                        actor_path=actor_path,
                        component_class=component_class,
                        component_name=component_name or None,
                    )
                    payload["success"] = True
                    result = UnrealAddComponentResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealAddComponentResponse, ErrorResponse),
                        "add_unreal_component",
                    )
            except Exception as e:
                self.logger.error("Error adding Unreal component: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealAddComponentResponse, ErrorResponse),
                    "add_unreal_component",
                )

        @self.mcp.tool(
            name="set_unreal_actor_visibility",
            description="Set actor visibility in the Unreal level.",
            annotations=self._tool_annotations(
                read_only=False,
                idempotent=True,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealSetActorVisibilityResponse, ErrorResponse
            ),
            task=self._task_optional(),
        )
        async def set_unreal_actor_visibility(
            actor_path: str,
            visible: bool = True,
            propagate: bool = True,
        ) -> Dict[str, Any]:
            """Set actor visibility."""
            rate_error = self._check_rate_limit("set_unreal_actor_visibility")
            if rate_error:
                return rate_error
            try:
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()
                with self.unreal_adapter.create_session() as session:
                    payload = await session.set_actor_visibility(
                        actor_path=actor_path,
                        visible=visible,
                        propagate=propagate,
                    )
                    payload["success"] = True
                    result = UnrealSetActorVisibilityResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealSetActorVisibilityResponse, ErrorResponse),
                        "set_unreal_actor_visibility",
                    )
            except Exception as e:
                self.logger.error("Error setting Unreal actor visibility: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealSetActorVisibilityResponse, ErrorResponse),
                    "set_unreal_actor_visibility",
                )

        # ---------------------------------------------------------------
        # Phase 4 — Materials, Lighting & Rendering
        # ---------------------------------------------------------------

        @self.mcp.tool(
            name="get_unreal_material_info",
            description="Get material instance parameters and metadata.",
            annotations=self._tool_annotations(
                read_only=True,
                idempotent=True,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealGetMaterialInfoResponse, ErrorResponse
            ),
            task=self._task_optional(),
        )
        async def get_unreal_material_info(
            material_path: str,
        ) -> Dict[str, Any]:
            """Get material info."""
            rate_error = self._check_rate_limit("get_unreal_material_info")
            if rate_error:
                return rate_error
            try:
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()
                with self.unreal_adapter.create_session() as session:
                    payload = await session.get_material_info(
                        material_path=material_path,
                    )
                    payload["success"] = True
                    result = UnrealGetMaterialInfoResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealGetMaterialInfoResponse, ErrorResponse),
                        "get_unreal_material_info",
                    )
            except Exception as e:
                self.logger.error("Error getting Unreal material info: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealGetMaterialInfoResponse, ErrorResponse),
                    "get_unreal_material_info",
                )

        @self.mcp.tool(
            name="set_unreal_material_params",
            description="Set scalar/vector/texture parameters on a Material Instance.",
            annotations=self._tool_annotations(
                read_only=False,
                idempotent=True,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealSetMaterialParamsResponse, ErrorResponse
            ),
            task=self._task_optional(),
        )
        async def set_unreal_material_params(
            material_path: str,
            scalar_params_json: str = "",
            vector_params_json: str = "",
            texture_params_json: str = "",
        ) -> Dict[str, Any]:
            """Set material instance parameters."""
            import json as json_lib

            rate_error = self._check_rate_limit("set_unreal_material_params")
            if rate_error:
                return rate_error
            try:
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()

                scalar_params = (
                    json_lib.loads(scalar_params_json) if scalar_params_json else None
                )
                vector_params = (
                    json_lib.loads(vector_params_json) if vector_params_json else None
                )
                texture_params = (
                    json_lib.loads(texture_params_json) if texture_params_json else None
                )

                with self.unreal_adapter.create_session() as session:
                    payload = await session.set_material_params(
                        material_path=material_path,
                        scalar_params=scalar_params,
                        vector_params=vector_params,
                        texture_params=texture_params,
                    )
                    payload["success"] = True
                    result = UnrealSetMaterialParamsResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealSetMaterialParamsResponse, ErrorResponse),
                        "set_unreal_material_params",
                    )
            except Exception as e:
                self.logger.error("Error setting Unreal material params: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealSetMaterialParamsResponse, ErrorResponse),
                    "set_unreal_material_params",
                )

        @self.mcp.tool(
            name="create_unreal_material_instance",
            description="Create a Material Instance Constant from a parent material.",
            annotations=self._tool_annotations(
                read_only=False,
                idempotent=False,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealCreateMaterialInstanceResponse, ErrorResponse
            ),
            task=self._task_optional(),
        )
        async def create_unreal_material_instance(
            parent_path: str,
            instance_name: str,
            save_path: str = "",
        ) -> Dict[str, Any]:
            """Create a material instance."""
            rate_error = self._check_rate_limit("create_unreal_material_instance")
            if rate_error:
                return rate_error
            try:
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()
                with self.unreal_adapter.create_session() as session:
                    payload = await session.create_material_instance(
                        parent_path=parent_path,
                        instance_name=instance_name,
                        save_path=save_path,
                    )
                    payload["success"] = True
                    result = UnrealCreateMaterialInstanceResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealCreateMaterialInstanceResponse, ErrorResponse),
                        "create_unreal_material_instance",
                    )
            except Exception as e:
                self.logger.error("Error creating Unreal material instance: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealCreateMaterialInstanceResponse, ErrorResponse),
                    "create_unreal_material_instance",
                )

        @self.mcp.tool(
            name="assign_unreal_material",
            description="Assign a material to a mesh component's material slot.",
            annotations=self._tool_annotations(
                read_only=False,
                idempotent=True,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealAssignMaterialResponse, ErrorResponse
            ),
            task=self._task_optional(),
        )
        async def assign_unreal_material(
            actor_path: str,
            material_path: str,
            slot_index: int = 0,
        ) -> Dict[str, Any]:
            """Assign material to actor."""
            rate_error = self._check_rate_limit("assign_unreal_material")
            if rate_error:
                return rate_error
            try:
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()
                with self.unreal_adapter.create_session() as session:
                    payload = await session.assign_material(
                        actor_path=actor_path,
                        material_path=material_path,
                        slot_index=slot_index,
                    )
                    payload["success"] = True
                    result = UnrealAssignMaterialResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealAssignMaterialResponse, ErrorResponse),
                        "assign_unreal_material",
                    )
            except Exception as e:
                self.logger.error("Error assigning Unreal material: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealAssignMaterialResponse, ErrorResponse),
                    "assign_unreal_material",
                )

        @self.mcp.tool(
            name="set_unreal_light_params",
            description="Set light component parameters (intensity, color, temperature, shadows).",
            annotations=self._tool_annotations(
                read_only=False,
                idempotent=True,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealSetLightParamsResponse, ErrorResponse
            ),
            task=self._task_optional(),
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
            rate_error = self._check_rate_limit("set_unreal_light_params")
            if rate_error:
                return rate_error
            try:
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()
                with self.unreal_adapter.create_session() as session:
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
                    result = UnrealSetLightParamsResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealSetLightParamsResponse, ErrorResponse),
                        "set_unreal_light_params",
                    )
            except Exception as e:
                self.logger.error("Error setting Unreal light params: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealSetLightParamsResponse, ErrorResponse),
                    "set_unreal_light_params",
                )

        @self.mcp.tool(
            name="set_unreal_render_settings",
            description="Set rendering or post-process settings via console command.",
            annotations=self._tool_annotations(
                read_only=False,
                idempotent=True,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealSetRenderSettingsResponse, ErrorResponse
            ),
            task=self._task_optional(),
        )
        async def set_unreal_render_settings(
            setting_name: str,
            setting_value: str,
        ) -> Dict[str, Any]:
            """Set render settings."""
            rate_error = self._check_rate_limit("set_unreal_render_settings")
            if rate_error:
                return rate_error
            try:
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()
                with self.unreal_adapter.create_session() as session:
                    payload = await session.set_render_settings(
                        setting_name=setting_name,
                        setting_value=setting_value,
                    )
                    payload["success"] = True
                    result = UnrealSetRenderSettingsResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealSetRenderSettingsResponse, ErrorResponse),
                        "set_unreal_render_settings",
                    )
            except Exception as e:
                self.logger.error("Error setting Unreal render settings: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealSetRenderSettingsResponse, ErrorResponse),
                    "set_unreal_render_settings",
                )

        # ---- Phase 5: Physics & Simulation Control ----

        @self.mcp.tool(
            name="control_unreal_simulation",
            description="Control Play-In-Editor (PIE) session: start, stop, pause, resume, step.",
            annotations=self._tool_annotations(
                read_only=False,
                idempotent=False,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealControlSimulationResponse, ErrorResponse
            ),
            task=self._task_optional(),
        )
        async def control_unreal_simulation(
            action: str,
        ) -> Dict[str, Any]:
            """Control PIE session."""
            rate_error = self._check_rate_limit("control_unreal_simulation")
            if rate_error:
                return rate_error
            try:
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()
                with self.unreal_adapter.create_session() as session:
                    payload = await session.control_simulation(action=action)
                    payload["success"] = True
                    result = UnrealControlSimulationResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealControlSimulationResponse, ErrorResponse),
                        "control_unreal_simulation",
                    )
            except Exception as e:
                self.logger.error("Error controlling Unreal simulation: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealControlSimulationResponse, ErrorResponse),
                    "control_unreal_simulation",
                )

        @self.mcp.tool(
            name="get_unreal_simulation_status",
            description="Get current Play-In-Editor simulation status (playing, paused, stopped).",
            annotations=self._tool_annotations(
                read_only=True,
                idempotent=True,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealGetSimulationStatusResponse, ErrorResponse
            ),
            task=self._task_optional(),
        )
        async def get_unreal_simulation_status() -> Dict[str, Any]:
            """Get PIE simulation status."""
            rate_error = self._check_rate_limit("get_unreal_simulation_status")
            if rate_error:
                return rate_error
            try:
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()
                with self.unreal_adapter.create_session() as session:
                    payload = await session.get_simulation_status()
                    payload["success"] = True
                    result = UnrealGetSimulationStatusResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealGetSimulationStatusResponse, ErrorResponse),
                        "get_unreal_simulation_status",
                    )
            except Exception as e:
                self.logger.error("Error getting Unreal simulation status: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealGetSimulationStatusResponse, ErrorResponse),
                    "get_unreal_simulation_status",
                )

        @self.mcp.tool(
            name="enable_unreal_physics",
            description="Enable or disable physics simulation on an actor.",
            annotations=self._tool_annotations(
                read_only=False,
                idempotent=True,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealEnablePhysicsResponse, ErrorResponse
            ),
            task=self._task_optional(),
        )
        async def enable_unreal_physics(
            actor_path: str,
            enable: bool = True,
            simulate_physics: bool = True,
        ) -> Dict[str, Any]:
            """Enable physics on actor."""
            rate_error = self._check_rate_limit("enable_unreal_physics")
            if rate_error:
                return rate_error
            try:
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()
                with self.unreal_adapter.create_session() as session:
                    payload = await session.enable_physics(
                        actor_path=actor_path,
                        enable=enable,
                        simulate_physics=simulate_physics,
                    )
                    payload["success"] = True
                    result = UnrealEnablePhysicsResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealEnablePhysicsResponse, ErrorResponse),
                        "enable_unreal_physics",
                    )
            except Exception as e:
                self.logger.error("Error enabling Unreal physics: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealEnablePhysicsResponse, ErrorResponse),
                    "enable_unreal_physics",
                )

        @self.mcp.tool(
            name="set_unreal_collision",
            description="Set collision presets and enable/disable collision on an actor.",
            annotations=self._tool_annotations(
                read_only=False,
                idempotent=True,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealSetCollisionResponse, ErrorResponse
            ),
            task=self._task_optional(),
        )
        async def set_unreal_collision(
            actor_path: str,
            collision_preset: str = "",
            collision_enabled: bool = True,
        ) -> Dict[str, Any]:
            """Set collision configuration."""
            rate_error = self._check_rate_limit("set_unreal_collision")
            if rate_error:
                return rate_error
            try:
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()
                with self.unreal_adapter.create_session() as session:
                    payload = await session.set_collision(
                        actor_path=actor_path,
                        collision_preset=collision_preset,
                        collision_enabled=collision_enabled,
                    )
                    payload["success"] = True
                    result = UnrealSetCollisionResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealSetCollisionResponse, ErrorResponse),
                        "set_unreal_collision",
                    )
            except Exception as e:
                self.logger.error("Error setting Unreal collision: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealSetCollisionResponse, ErrorResponse),
                    "set_unreal_collision",
                )

        @self.mcp.tool(
            name="apply_unreal_force",
            description="Apply a force or impulse to an actor's physics body.",
            annotations=self._tool_annotations(
                read_only=False,
                idempotent=False,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealApplyForceResponse, ErrorResponse
            ),
            task=self._task_optional(),
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
            rate_error = self._check_rate_limit("apply_unreal_force")
            if rate_error:
                return rate_error
            try:
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()
                with self.unreal_adapter.create_session() as session:
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
                    result = UnrealApplyForceResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealApplyForceResponse, ErrorResponse),
                        "apply_unreal_force",
                    )
            except Exception as e:
                self.logger.error("Error applying Unreal force: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealApplyForceResponse, ErrorResponse),
                    "apply_unreal_force",
                )

        @self.mcp.tool(
            name="set_unreal_physics_params",
            description="Set physics body parameters (mass, damping, gravity) on an actor.",
            annotations=self._tool_annotations(
                read_only=False,
                idempotent=True,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealSetPhysicsParamsResponse, ErrorResponse
            ),
            task=self._task_optional(),
        )
        async def set_unreal_physics_params(
            actor_path: str,
            mass: Optional[float] = None,
            linear_damping: Optional[float] = None,
            angular_damping: Optional[float] = None,
            enable_gravity: Optional[bool] = None,
        ) -> Dict[str, Any]:
            """Set physics parameters."""
            rate_error = self._check_rate_limit("set_unreal_physics_params")
            if rate_error:
                return rate_error
            try:
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()
                with self.unreal_adapter.create_session() as session:
                    payload = await session.set_physics_params(
                        actor_path=actor_path,
                        mass=mass,
                        linear_damping=linear_damping,
                        angular_damping=angular_damping,
                        enable_gravity=enable_gravity,
                    )
                    payload["success"] = True
                    result = UnrealSetPhysicsParamsResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealSetPhysicsParamsResponse, ErrorResponse),
                        "set_unreal_physics_params",
                    )
            except Exception as e:
                self.logger.error("Error setting Unreal physics params: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealSetPhysicsParamsResponse, ErrorResponse),
                    "set_unreal_physics_params",
                )

        # ----------------------------------------------------------
        # Phase 6: USD / SimReady Bridge
        # ----------------------------------------------------------

        @self.mcp.tool(
            name="import_unreal_usd",
            description="Import a USD file into Unreal via Interchange Framework.",
            annotations=self._tool_annotations(
                read_only=False,
                idempotent=False,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealImportUsdResponse, ErrorResponse
            ),
            task=self._task_optional(),
        )
        async def import_unreal_usd(
            usd_path: str,
            target_path: Optional[str] = None,
            import_animations: bool = True,
            import_materials: bool = True,
            scale_factor: float = 1.0,
        ) -> Dict[str, Any]:
            """Import USD file into Unreal."""
            rate_error = self._check_rate_limit("import_unreal_usd")
            if rate_error:
                return rate_error
            try:
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()
                with self.unreal_adapter.create_session() as session:
                    payload = await session.import_usd(
                        usd_path=usd_path,
                        target_path=target_path,
                        import_animations=import_animations,
                        import_materials=import_materials,
                        scale_factor=scale_factor,
                    )
                    payload["success"] = True
                    result = UnrealImportUsdResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealImportUsdResponse, ErrorResponse),
                        "import_unreal_usd",
                    )
            except Exception as e:
                self.logger.error("Error importing USD to Unreal: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealImportUsdResponse, ErrorResponse),
                    "import_unreal_usd",
                )

        @self.mcp.tool(
            name="export_unreal_usd",
            description="Export Unreal actors to USD via Interchange Framework.",
            annotations=self._tool_annotations(
                read_only=False,
                idempotent=False,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealExportUsdResponse, ErrorResponse
            ),
            task=self._task_optional(),
        )
        async def export_unreal_usd(
            actor_paths: str,
            output_path: str,
            export_materials: bool = True,
            export_animations: bool = True,
            convert_to_meters: bool = True,
        ) -> Dict[str, Any]:
            """Export actors to USD."""
            rate_error = self._check_rate_limit("export_unreal_usd")
            if rate_error:
                return rate_error
            try:
                paths = [p.strip() for p in actor_paths.split(",")]
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()
                with self.unreal_adapter.create_session() as session:
                    payload = await session.export_usd(
                        actor_paths=paths,
                        output_path=output_path,
                        export_materials=export_materials,
                        export_animations=export_animations,
                        convert_to_meters=convert_to_meters,
                    )
                    payload["success"] = True
                    result = UnrealExportUsdResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealExportUsdResponse, ErrorResponse),
                        "export_unreal_usd",
                    )
            except Exception as e:
                self.logger.error("Error exporting Unreal USD: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealExportUsdResponse, ErrorResponse),
                    "export_unreal_usd",
                )

        @self.mcp.tool(
            name="convert_to_simready",
            description="Convert Unreal actors to NVIDIA SimReady asset format.",
            annotations=self._tool_annotations(
                read_only=False,
                idempotent=False,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealConvertToSimreadyResponse, ErrorResponse
            ),
            task=self._task_optional(),
        )
        async def convert_to_simready(
            actor_paths: str,
            output_directory: str,
            add_physics: bool = True,
            add_collision: bool = True,
            semantic_labels: str = "",
        ) -> Dict[str, Any]:
            """Convert actors to SimReady format."""
            rate_error = self._check_rate_limit("convert_to_simready")
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
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()
                with self.unreal_adapter.create_session() as session:
                    payload = await session.convert_to_simready(
                        actor_paths=paths,
                        output_directory=output_directory,
                        add_physics=add_physics,
                        add_collision=add_collision,
                        semantic_labels=labels,
                    )
                    payload["success"] = True
                    result = UnrealConvertToSimreadyResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealConvertToSimreadyResponse, ErrorResponse),
                        "convert_to_simready",
                    )
            except Exception as e:
                self.logger.error("Error converting to SimReady: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealConvertToSimreadyResponse, ErrorResponse),
                    "convert_to_simready",
                )

        @self.mcp.tool(
            name="validate_simready_asset",
            description="Validate an asset against NVIDIA SimReady requirements.",
            annotations=self._tool_annotations(
                read_only=True,
                idempotent=True,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealValidateSimreadyResponse, ErrorResponse
            ),
            task=self._task_optional(),
        )
        async def validate_simready_asset(
            asset_path: str,
            checks: str = "",
        ) -> Dict[str, Any]:
            """Validate asset against SimReady spec."""
            rate_error = self._check_rate_limit("validate_simready_asset")
            if rate_error:
                return rate_error
            try:
                check_list = (
                    [c.strip() for c in checks.split(",") if c.strip()]
                    if checks
                    else None
                )
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()
                with self.unreal_adapter.create_session() as session:
                    payload = await session.validate_simready_asset(
                        asset_path=asset_path,
                        checks=check_list,
                    )
                    payload["success"] = True
                    result = UnrealValidateSimreadyResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealValidateSimreadyResponse, ErrorResponse),
                        "validate_simready_asset",
                    )
            except Exception as e:
                self.logger.error("Error validating SimReady asset: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealValidateSimreadyResponse, ErrorResponse),
                    "validate_simready_asset",
                )

        @self.mcp.tool(
            name="get_unreal_interchange_info",
            description="Query available Interchange pipelines and supported formats.",
            annotations=self._tool_annotations(
                read_only=True,
                idempotent=True,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealGetInterchangeInfoResponse, ErrorResponse
            ),
            task=self._task_optional(),
        )
        async def get_unreal_interchange_info() -> Dict[str, Any]:
            """Get Interchange Framework info."""
            rate_error = self._check_rate_limit("get_unreal_interchange_info")
            if rate_error:
                return rate_error
            try:
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()
                with self.unreal_adapter.create_session() as session:
                    payload = await session.get_interchange_info()
                    payload["success"] = True
                    result = UnrealGetInterchangeInfoResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealGetInterchangeInfoResponse, ErrorResponse),
                        "get_unreal_interchange_info",
                    )
            except Exception as e:
                self.logger.error("Error getting interchange info: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealGetInterchangeInfoResponse, ErrorResponse),
                    "get_unreal_interchange_info",
                )

        # ----------------------------------------------------------
        # Phase 7: Advanced Agent Tools
        # ----------------------------------------------------------

        @self.mcp.tool(
            name="batch_unreal_operations",
            description="Execute multiple Remote Control operations in one HTTP call.",
            annotations=self._tool_annotations(
                read_only=False,
                idempotent=False,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealBatchOperationsResponse, ErrorResponse
            ),
            task=self._task_optional(),
        )
        async def batch_unreal_operations(
            operations: str,
        ) -> Dict[str, Any]:
            """Batch multiple operations."""
            rate_error = self._check_rate_limit("batch_unreal_operations")
            if rate_error:
                return rate_error
            try:
                import json as _json

                ops = _json.loads(operations)
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()
                with self.unreal_adapter.create_session() as session:
                    payload = await session.batch_operations(operations=ops)
                    payload["success"] = True
                    result = UnrealBatchOperationsResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealBatchOperationsResponse, ErrorResponse),
                        "batch_unreal_operations",
                    )
            except Exception as e:
                self.logger.error("Error in batch Unreal operations: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealBatchOperationsResponse, ErrorResponse),
                    "batch_unreal_operations",
                )

        @self.mcp.tool(
            name="query_unreal_scene_graph",
            description="Query the Unreal scene graph hierarchy.",
            annotations=self._tool_annotations(
                read_only=True,
                idempotent=True,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealQuerySceneGraphResponse, ErrorResponse
            ),
            task=self._task_optional(),
        )
        async def query_unreal_scene_graph(
            root_path: Optional[str] = None,
            max_depth: int = 10,
            include_components: bool = False,
            class_filter: Optional[str] = None,
        ) -> Dict[str, Any]:
            """Query scene graph hierarchy."""
            rate_error = self._check_rate_limit("query_unreal_scene_graph")
            if rate_error:
                return rate_error
            try:
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()
                with self.unreal_adapter.create_session() as session:
                    payload = await session.query_scene_graph(
                        root_path=root_path,
                        max_depth=max_depth,
                        include_components=include_components,
                        class_filter=class_filter,
                    )
                    payload["success"] = True
                    result = UnrealQuerySceneGraphResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealQuerySceneGraphResponse, ErrorResponse),
                        "query_unreal_scene_graph",
                    )
            except Exception as e:
                self.logger.error("Error querying scene graph: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealQuerySceneGraphResponse, ErrorResponse),
                    "query_unreal_scene_graph",
                )

        @self.mcp.tool(
            name="analyze_unreal_scene_for_robotics",
            description="Analyze the scene for robotics use-cases (traversability, graspability, collision).",
            annotations=self._tool_annotations(
                read_only=True,
                idempotent=True,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealAnalyzeSceneForRoboticsResponse, ErrorResponse
            ),
            task=self._task_optional(),
        )
        async def analyze_unreal_scene_for_robotics(
            analysis_types: str = "",
            actor_filter: Optional[str] = None,
        ) -> Dict[str, Any]:
            """Analyze scene for robotics."""
            rate_error = self._check_rate_limit("analyze_unreal_scene_for_robotics")
            if rate_error:
                return rate_error
            try:
                types_list = (
                    [t.strip() for t in analysis_types.split(",") if t.strip()]
                    if analysis_types
                    else None
                )
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()
                with self.unreal_adapter.create_session() as session:
                    payload = await session.analyze_scene_for_robotics(
                        analysis_types=types_list,
                        actor_filter=actor_filter,
                    )
                    payload["success"] = True
                    result = UnrealAnalyzeSceneForRoboticsResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealAnalyzeSceneForRoboticsResponse, ErrorResponse),
                        "analyze_unreal_scene_for_robotics",
                    )
            except Exception as e:
                self.logger.error("Error analyzing scene for robotics: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealAnalyzeSceneForRoboticsResponse, ErrorResponse),
                    "analyze_unreal_scene_for_robotics",
                )

        @self.mcp.tool(
            name="generate_unreal_procedural_scene",
            description="Generate a procedural scene (warehouse, outdoor, room, corridor).",
            annotations=self._tool_annotations(
                read_only=False,
                idempotent=False,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealGenerateProceduralSceneResponse, ErrorResponse
            ),
            task=self._task_optional(),
        )
        async def generate_unreal_procedural_scene(
            scene_type: str,
            parameters: str = "{}",
            bounds_min: str = "",
            bounds_max: str = "",
        ) -> Dict[str, Any]:
            """Generate procedural scene."""
            rate_error = self._check_rate_limit("generate_unreal_procedural_scene")
            if rate_error:
                return rate_error
            try:
                import json as _json

                params = _json.loads(parameters) if parameters else None
                bmin = (
                    [float(v) for v in bounds_min.split(",")]
                    if bounds_min
                    else None
                )
                bmax = (
                    [float(v) for v in bounds_max.split(",")]
                    if bounds_max
                    else None
                )
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()
                with self.unreal_adapter.create_session() as session:
                    payload = await session.generate_procedural_scene(
                        scene_type=scene_type,
                        parameters=params,
                        bounds_min=bmin,
                        bounds_max=bmax,
                    )
                    payload["success"] = True
                    result = UnrealGenerateProceduralSceneResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealGenerateProceduralSceneResponse, ErrorResponse),
                        "generate_unreal_procedural_scene",
                    )
            except Exception as e:
                self.logger.error("Error generating procedural scene: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealGenerateProceduralSceneResponse, ErrorResponse),
                    "generate_unreal_procedural_scene",
                )

        @self.mcp.tool(
            name="get_unreal_actor_by_semantic_label",
            description="Find actors by semantic tag or label.",
            annotations=self._tool_annotations(
                read_only=True,
                idempotent=True,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealGetActorBySemanticLabelResponse, ErrorResponse
            ),
            task=self._task_optional(),
        )
        async def get_unreal_actor_by_semantic_label(
            label: str,
            match_mode: str = "exact",
            max_results: int = 100,
        ) -> Dict[str, Any]:
            """Find actors by semantic label."""
            rate_error = self._check_rate_limit("get_unreal_actor_by_semantic_label")
            if rate_error:
                return rate_error
            try:
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()
                with self.unreal_adapter.create_session() as session:
                    payload = await session.get_actor_by_semantic_label(
                        label=label,
                        match_mode=match_mode,
                        max_results=max_results,
                    )
                    payload["success"] = True
                    result = UnrealGetActorBySemanticLabelResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealGetActorBySemanticLabelResponse, ErrorResponse),
                        "get_unreal_actor_by_semantic_label",
                    )
            except Exception as e:
                self.logger.error("Error finding actors by label: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealGetActorBySemanticLabelResponse, ErrorResponse),
                    "get_unreal_actor_by_semantic_label",
                )

        # ----------------------------------------------------------
        # Phase 8: Geometry & Modeling (GeometryScript)
        # ----------------------------------------------------------

        @self.mcp.tool(
            name="generate_unreal_mesh_primitive",
            description="Create a parametric mesh primitive (box, sphere, cylinder, cone, torus, capsule).",
            annotations=self._tool_annotations(
                read_only=False,
                idempotent=False,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealGenerateMeshPrimitiveResponse, ErrorResponse
            ),
            task=self._task_optional(),
        )
        async def generate_unreal_mesh_primitive(
            primitive_type: str,
            dimensions: str = "{}",
            segments: int = 32,
            location: str = "",
            actor_label: Optional[str] = None,
        ) -> Dict[str, Any]:
            """Create mesh primitive."""
            rate_error = self._check_rate_limit("generate_unreal_mesh_primitive")
            if rate_error:
                return rate_error
            try:
                import json as _json

                dims = _json.loads(dimensions) if dimensions else None
                loc = (
                    [float(v) for v in location.split(",")]
                    if location
                    else None
                )
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()
                with self.unreal_adapter.create_session() as session:
                    payload = await session.generate_mesh_primitive(
                        primitive_type=primitive_type,
                        dimensions=dims,
                        segments=segments,
                        location=loc,
                        actor_label=actor_label,
                    )
                    payload["success"] = True
                    result = UnrealGenerateMeshPrimitiveResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealGenerateMeshPrimitiveResponse, ErrorResponse),
                        "generate_unreal_mesh_primitive",
                    )
            except Exception as e:
                self.logger.error("Error generating mesh primitive: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealGenerateMeshPrimitiveResponse, ErrorResponse),
                    "generate_unreal_mesh_primitive",
                )

        @self.mcp.tool(
            name="apply_unreal_mesh_boolean",
            description="Apply boolean operation (union, subtract, intersect) between two meshes.",
            annotations=self._tool_annotations(
                read_only=False,
                idempotent=False,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealApplyMeshBooleanResponse, ErrorResponse
            ),
            task=self._task_optional(),
        )
        async def apply_unreal_mesh_boolean(
            target_mesh_path: str,
            tool_mesh_path: str,
            operation: str,
        ) -> Dict[str, Any]:
            """Apply mesh boolean."""
            rate_error = self._check_rate_limit("apply_unreal_mesh_boolean")
            if rate_error:
                return rate_error
            try:
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()
                with self.unreal_adapter.create_session() as session:
                    payload = await session.apply_mesh_boolean(
                        target_mesh_path=target_mesh_path,
                        tool_mesh_path=tool_mesh_path,
                        operation=operation,
                    )
                    payload["success"] = True
                    result = UnrealApplyMeshBooleanResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealApplyMeshBooleanResponse, ErrorResponse),
                        "apply_unreal_mesh_boolean",
                    )
            except Exception as e:
                self.logger.error("Error applying mesh boolean: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealApplyMeshBooleanResponse, ErrorResponse),
                    "apply_unreal_mesh_boolean",
                )

        @self.mcp.tool(
            name="compute_unreal_convex_hull",
            description="Compute convex hull envelope of a mesh.",
            annotations=self._tool_annotations(
                read_only=False,
                idempotent=False,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealComputeConvexHullResponse, ErrorResponse
            ),
            task=self._task_optional(),
        )
        async def compute_unreal_convex_hull(
            mesh_path: str,
        ) -> Dict[str, Any]:
            """Compute convex hull."""
            rate_error = self._check_rate_limit("compute_unreal_convex_hull")
            if rate_error:
                return rate_error
            try:
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()
                with self.unreal_adapter.create_session() as session:
                    payload = await session.compute_convex_hull(
                        mesh_path=mesh_path,
                    )
                    payload["success"] = True
                    result = UnrealComputeConvexHullResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealComputeConvexHullResponse, ErrorResponse),
                        "compute_unreal_convex_hull",
                    )
            except Exception as e:
                self.logger.error("Error computing convex hull: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealComputeConvexHullResponse, ErrorResponse),
                    "compute_unreal_convex_hull",
                )

        @self.mcp.tool(
            name="decompose_unreal_convex_hull",
            description="V-HACD convex decomposition for collision geometry.",
            annotations=self._tool_annotations(
                read_only=False,
                idempotent=False,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealDecomposeConvexHullResponse, ErrorResponse
            ),
            task=self._task_optional(),
        )
        async def decompose_unreal_convex_hull(
            mesh_path: str,
            max_hulls: int = 16,
            max_vertices_per_hull: int = 32,
            min_cluster_size: int = 256,
            resolution: int = 100000,
        ) -> Dict[str, Any]:
            """V-HACD convex decomposition."""
            rate_error = self._check_rate_limit("decompose_unreal_convex_hull")
            if rate_error:
                return rate_error
            try:
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()
                with self.unreal_adapter.create_session() as session:
                    payload = await session.decompose_convex_hull(
                        mesh_path=mesh_path,
                        max_hulls=max_hulls,
                        max_vertices_per_hull=max_vertices_per_hull,
                        min_cluster_size=min_cluster_size,
                        resolution=resolution,
                    )
                    payload["success"] = True
                    result = UnrealDecomposeConvexHullResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealDecomposeConvexHullResponse, ErrorResponse),
                        "decompose_unreal_convex_hull",
                    )
            except Exception as e:
                self.logger.error("Error decomposing convex hull: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealDecomposeConvexHullResponse, ErrorResponse),
                    "decompose_unreal_convex_hull",
                )

        @self.mcp.tool(
            name="edit_unreal_mesh_topology",
            description="Edit mesh topology (extrude, bevel, inset, loop cut, scale_faces).",
            annotations=self._tool_annotations(
                read_only=False,
                idempotent=False,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealEditMeshTopologyResponse, ErrorResponse
            ),
            task=self._task_optional(),
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
            rate_error = self._check_rate_limit("edit_unreal_mesh_topology")
            if rate_error:
                return rate_error
            try:
                scale_list = (
                    [float(v) for v in scale.split(",")]
                    if scale
                    else None
                )
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()
                with self.unreal_adapter.create_session() as session:
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
                    result = UnrealEditMeshTopologyResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealEditMeshTopologyResponse, ErrorResponse),
                        "edit_unreal_mesh_topology",
                    )
            except Exception as e:
                self.logger.error("Error editing mesh topology: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealEditMeshTopologyResponse, ErrorResponse),
                    "edit_unreal_mesh_topology",
                )

        @self.mcp.tool(
            name="subdivide_unreal_mesh",
            description="Catmull-Clark / Loop / bilinear subdivision.",
            annotations=self._tool_annotations(
                read_only=False,
                idempotent=False,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealSubdivideMeshResponse, ErrorResponse
            ),
            task=self._task_optional(),
        )
        async def subdivide_unreal_mesh(
            mesh_path: str,
            level: int = 2,
            scheme: str = "catmull_clark",
        ) -> Dict[str, Any]:
            """Subdivide mesh."""
            rate_error = self._check_rate_limit("subdivide_unreal_mesh")
            if rate_error:
                return rate_error
            try:
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()
                with self.unreal_adapter.create_session() as session:
                    payload = await session.subdivide_mesh(
                        mesh_path=mesh_path,
                        level=level,
                        scheme=scheme,
                    )
                    payload["success"] = True
                    result = UnrealSubdivideMeshResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealSubdivideMeshResponse, ErrorResponse),
                        "subdivide_unreal_mesh",
                    )
            except Exception as e:
                self.logger.error("Error subdividing mesh: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealSubdivideMeshResponse, ErrorResponse),
                    "subdivide_unreal_mesh",
                )

        @self.mcp.tool(
            name="simplify_unreal_mesh",
            description="Simplify/decimate a mesh to reduce triangle count.",
            annotations=self._tool_annotations(
                read_only=False,
                idempotent=False,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealSimplifyMeshResponse, ErrorResponse
            ),
            task=self._task_optional(),
        )
        async def simplify_unreal_mesh(
            mesh_path: str,
            target_triangle_count: Optional[int] = None,
            target_percentage: Optional[float] = None,
            max_error: Optional[float] = None,
        ) -> Dict[str, Any]:
            """Simplify mesh."""
            rate_error = self._check_rate_limit("simplify_unreal_mesh")
            if rate_error:
                return rate_error
            try:
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()
                with self.unreal_adapter.create_session() as session:
                    payload = await session.simplify_mesh(
                        mesh_path=mesh_path,
                        target_triangle_count=target_triangle_count,
                        target_percentage=target_percentage,
                        max_error=max_error,
                    )
                    payload["success"] = True
                    result = UnrealSimplifyMeshResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealSimplifyMeshResponse, ErrorResponse),
                        "simplify_unreal_mesh",
                    )
            except Exception as e:
                self.logger.error("Error simplifying mesh: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealSimplifyMeshResponse, ErrorResponse),
                    "simplify_unreal_mesh",
                )

        @self.mcp.tool(
            name="cut_unreal_mesh_plane",
            description="Cut/slice a mesh along an arbitrary plane.",
            annotations=self._tool_annotations(
                read_only=False,
                idempotent=False,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealCutMeshPlaneResponse, ErrorResponse
            ),
            task=self._task_optional(),
        )
        async def cut_unreal_mesh_plane(
            mesh_path: str,
            plane_origin: str,
            plane_normal: str,
            fill_holes: bool = True,
            keep_both_sides: bool = False,
        ) -> Dict[str, Any]:
            """Cut mesh with plane."""
            rate_error = self._check_rate_limit("cut_unreal_mesh_plane")
            if rate_error:
                return rate_error
            try:
                origin = [float(v) for v in plane_origin.split(",")]
                normal = [float(v) for v in plane_normal.split(",")]
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()
                with self.unreal_adapter.create_session() as session:
                    payload = await session.cut_mesh_plane(
                        mesh_path=mesh_path,
                        plane_origin=origin,
                        plane_normal=normal,
                        fill_holes=fill_holes,
                        keep_both_sides=keep_both_sides,
                    )
                    payload["success"] = True
                    result = UnrealCutMeshPlaneResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealCutMeshPlaneResponse, ErrorResponse),
                        "cut_unreal_mesh_plane",
                    )
            except Exception as e:
                self.logger.error("Error cutting mesh with plane: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealCutMeshPlaneResponse, ErrorResponse),
                    "cut_unreal_mesh_plane",
                )

        @self.mcp.tool(
            name="validate_unreal_mesh",
            description="Validate mesh integrity (manifold, normals, degenerates, self-intersection).",
            annotations=self._tool_annotations(
                read_only=True,
                idempotent=True,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealValidateMeshResponse, ErrorResponse
            ),
            task=self._task_optional(),
        )
        async def validate_unreal_mesh(
            mesh_path: str,
            checks: str = "",
        ) -> Dict[str, Any]:
            """Validate mesh integrity."""
            rate_error = self._check_rate_limit("validate_unreal_mesh")
            if rate_error:
                return rate_error
            try:
                check_list = (
                    [c.strip() for c in checks.split(",") if c.strip()]
                    if checks
                    else None
                )
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()
                with self.unreal_adapter.create_session() as session:
                    payload = await session.validate_mesh(
                        mesh_path=mesh_path,
                        checks=check_list,
                    )
                    payload["success"] = True
                    result = UnrealValidateMeshResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealValidateMeshResponse, ErrorResponse),
                        "validate_unreal_mesh",
                    )
            except Exception as e:
                self.logger.error("Error validating mesh: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealValidateMeshResponse, ErrorResponse),
                    "validate_unreal_mesh",
                )

        @self.mcp.tool(
            name="convert_unreal_mesh_format",
            description="Convert mesh between formats (static mesh, dynamic mesh, skeletal mesh).",
            annotations=self._tool_annotations(
                read_only=False,
                idempotent=False,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealConvertMeshFormatResponse, ErrorResponse
            ),
            task=self._task_optional(),
        )
        async def convert_unreal_mesh_format(
            mesh_path: str,
            target_format: str,
            tessellation_options: str = "{}",
        ) -> Dict[str, Any]:
            """Convert mesh format."""
            rate_error = self._check_rate_limit("convert_unreal_mesh_format")
            if rate_error:
                return rate_error
            try:
                import json as _json

                tess_opts = (
                    _json.loads(tessellation_options)
                    if tessellation_options
                    else None
                )
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()
                with self.unreal_adapter.create_session() as session:
                    payload = await session.convert_mesh_format(
                        mesh_path=mesh_path,
                        target_format=target_format,
                        tessellation_options=tess_opts,
                    )
                    payload["success"] = True
                    result = UnrealConvertMeshFormatResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealConvertMeshFormatResponse, ErrorResponse),
                        "convert_unreal_mesh_format",
                    )
            except Exception as e:
                self.logger.error("Error converting mesh format: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealConvertMeshFormatResponse, ErrorResponse),
                    "convert_unreal_mesh_format",
                )

        @self.mcp.tool(
            name="remesh_unreal_mesh",
            description="Remesh a mesh (uniform, adaptive) to improve triangle quality.",
            annotations=self._tool_annotations(
                read_only=False,
                idempotent=False,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealRemeshMeshResponse, ErrorResponse
            ),
            task=self._task_optional(),
        )
        async def remesh_unreal_mesh(
            mesh_path: str,
            mode: str = "uniform",
            target_edge_length: Optional[float] = None,
            target_triangle_count: Optional[int] = None,
            smoothing_iterations: int = 3,
        ) -> Dict[str, Any]:
            """Remesh mesh."""
            rate_error = self._check_rate_limit("remesh_unreal_mesh")
            if rate_error:
                return rate_error
            try:
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()
                with self.unreal_adapter.create_session() as session:
                    payload = await session.remesh_mesh(
                        mesh_path=mesh_path,
                        mode=mode,
                        target_edge_length=target_edge_length,
                        target_triangle_count=target_triangle_count,
                        smoothing_iterations=smoothing_iterations,
                    )
                    payload["success"] = True
                    result = UnrealRemeshMeshResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealRemeshMeshResponse, ErrorResponse),
                        "remesh_unreal_mesh",
                    )
            except Exception as e:
                self.logger.error("Error remeshing: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealRemeshMeshResponse, ErrorResponse),
                    "remesh_unreal_mesh",
                )

        @self.mcp.tool(
            name="compute_unreal_mesh_uv",
            description="Generate or recompute UV coordinates for a mesh.",
            annotations=self._tool_annotations(
                read_only=False,
                idempotent=False,
                open_world=True,
            ),
            output_schema=self._tool_output_schema(
                UnrealComputeMeshUvResponse, ErrorResponse
            ),
            task=self._task_optional(),
        )
        async def compute_unreal_mesh_uv(
            mesh_path: str,
            method: str = "auto_uv",
            uv_channel: int = 0,
            island_padding: float = 2.0,
        ) -> Dict[str, Any]:
            """Compute mesh UVs."""
            rate_error = self._check_rate_limit("compute_unreal_mesh_uv")
            if rate_error:
                return rate_error
            try:
                if not self.unreal_adapter or not self.unreal_adapter.is_available():
                    return ErrorResponse(
                        error="Unreal runtime not available",
                        error_type="RuntimeError",
                    ).dict()
                with self.unreal_adapter.create_session() as session:
                    payload = await session.compute_mesh_uv(
                        mesh_path=mesh_path,
                        method=method,
                        uv_channel=uv_channel,
                        island_padding=island_padding,
                    )
                    payload["success"] = True
                    result = UnrealComputeMeshUvResponse(**payload).dict()
                    return self._validate_output(
                        result,
                        (UnrealComputeMeshUvResponse, ErrorResponse),
                        "compute_unreal_mesh_uv",
                    )
            except Exception as e:
                self.logger.error("Error computing mesh UVs: %s", e)
                result = ErrorResponse(error=str(e), error_type="Exception").dict()
                return self._validate_output(
                    result,
                    (UnrealComputeMeshUvResponse, ErrorResponse),
                    "compute_unreal_mesh_uv",
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
            self.logger.error("Error running MCP server: %s", e)
            raise

    def get_capabilities(self) -> List[str]:
        """Get list of server capabilities."""
        capabilities: list[str] = [
            "isaac_sim_script_execution",
            "isaac_sim_connectivity_check",
        ]

        if self.headless_adapter and self.headless_adapter.is_available():
            capabilities.extend(self.headless_adapter.get_capabilities())

        if self.blender_adapter and self.blender_adapter.is_available():
            capabilities.extend(self.blender_adapter.get_capabilities())

        if self.unreal_adapter and self.unreal_adapter.is_available():
            capabilities.extend(self.unreal_adapter.get_capabilities())

        return list(set(capabilities))  # Remove duplicates


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
