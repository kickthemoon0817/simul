"""
USD headless file operation tool registration for the Simul MCP server.

Every tool here is a thin closure around ``server._exec_backend``: the
closure validates its input, performs the stage operation and builds the
response model; the envelope owns the rate limit, usage record, sandbox
and exception wrapping, result budget and single transmission.

The pxr work of every tool runs on a worker thread through one
``USDCallScheduler`` per server, so a slow call no longer stalls the event
loop: reads run concurrently (one at a time per stage), and an edit runs
alone (see ``simul.mcp.usd_scheduler``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional

from fastmcp.tools import ToolResult

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
    Purpose,
    SceneSummaryRequest,
    SceneSummaryResponse,
    StageInfo,
    USDFileInfo,
    USDFileRequest,
    USDValidateRequest,
)
from ...usd.sandbox import CompositionSandbox
from ...usd.validation import WARNING, composition_issues, url_scheme
from ...usd.validation import validate_usd_file as run_validation
from ..usd_scheduler import USDCallScheduler
from ._helpers import with_param_descriptions

if TYPE_CHECKING:
    from ...adapters.headless_usd import HeadlessUSDSession
    from ..server import SimulMCPServer

SessionWork = Callable[["HeadlessUSDSession"], Dict[str, Any]]
SessionCall = Callable[["HeadlessUSDSession"], Awaitable[Dict[str, Any]]]

# Stage-bound queries an agent makes once per prim while it inspects an
# asset. They count only against the per-agent ceiling
# (security.global_requests_per_minute), not a per-tool bucket of their own.
PER_PRIM_QUERY_TOOLS = frozenset(
    {"get_prim_info", "get_mesh_info", "get_bounding_box", "search_prims"}
)


def _loaded_stage(session: "HeadlessUSDSession", stage_id: str) -> Any:
    """Return the stage ``load_usd_file`` registered as ``stage_id``.

    Raises:
        USDLookupError: No stage is loaded under that id.
    """
    from ...usd.lookup import USDLookupError

    stage = session.get_stage(stage_id)
    if stage is None:
        raise USDLookupError(
            f"Stage not loaded: {stage_id!r}. Load the file with load_usd_file "
            "and pass the stage_id it returns."
        )
    return stage


def register_usd_tools(server: "SimulMCPServer") -> None:
    """Register USD-related tools."""
    scheduler = USDCallScheduler(server.settings.server.timeout)
    server.exempt_from_per_tool_rate_limit(PER_PRIM_QUERY_TOOLS)

    def scheduled(
        tool_name: str, work: SessionWork, *, stage_id: Optional[str] = None, writes: bool = False
    ) -> SessionCall:
        """Run ``work`` on a worker thread once the scheduler admits it."""

        async def call(session: "HeadlessUSDSession") -> Dict[str, Any]:
            return await scheduler.run(
                tool_name, lambda: work(session), stage_key=stage_id, writes=writes
            )

        return call

    def usd_target(file_path: str) -> str:
        """The location a file argument names, checked against the sandbox.

        A local path comes back absolute: ``~`` and ``$VARS`` expanded and a
        relative path resolved against ``security.relative_path_base`` (the
        server's working directory by default), with the sandbox on or off. A
        URL comes back verbatim; the session refuses one no installed asset
        resolver handles.

        Raises:
            SandboxDenied: The location is outside the sandbox.
        """
        authorized = server._path_policy.authorize(file_path)
        if url_scheme(authorized) is not None:
            return authorized
        return str(server._path_policy.resolve(authorized))

    @server.mcp.tool(
        name="load_usd_file",
        description=(
            "Load a USD file and return stage information. A file pxr cannot open "
            "fails with LoadError, whose details.reason names the cause (ParseError "
            "with line and column, CorruptFile, EmptyFile, FileTooLarge, "
            "PermissionDenied, UnsupportedExtension, FileNotFound, NotAFile, "
            "UnsupportedURL for a URL no installed asset resolver opens, such as "
            "omniverse:// here). Missing sublayers, references or payloads do not fail "
            "the load; they are listed in warnings. A relative path resolves against "
            "security.relative_path_base (default: the server's working directory); "
            "file_path in the result is the resolved path. Paths: sandbox "
            "(security.allowed_paths), which also holds every sublayer, reference, "
            "payload, clip and asset path the file draws on: one outside it fails the "
            "load with SandboxError listing them."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=False
        ),
        output_schema=None,
        task=server._task_optional(),
    )
    @with_param_descriptions()
    async def load_usd_file(file_path: str) -> ToolResult:
        """
        Load a USD file and return stage information.

        Args:
            file_path: Path to USD file

        Returns:
            Stage information or error
        """

        def _load(session: "HeadlessUSDSession", target: str) -> Dict[str, Any]:
            # A USDLoadError carries pxr's reason; the envelope reports it as
            # LoadError with details.reason.
            stage_id = session.load_stage_or_raise(
                target, sandbox=CompositionSandbox(server._path_policy)
            )
            stage = session.get_stage(stage_id)
            stage_info = session.get_stage_info(stage_id)
            if stage is None or stage_info is None:
                return ErrorResponse(
                    error=f"Loaded {target} but could not read its stage information",
                    error_type="LoadError",
                ).model_dump()
            return StageInfo(
                stage_id=stage_id,
                file_path=target,
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
                warnings=[issue.to_dict() for issue in composition_issues(stage, WARNING)],
            ).model_dump()

        async def _load_call(session: "HeadlessUSDSession") -> Dict[str, Any]:
            input_data = server._validate_input(USDFileRequest, file_path=file_path)
            if isinstance(input_data, dict):
                return input_data
            target = usd_target(input_data.file_path)
            # Loading (re)registers the stage, so it waits for calls bound to it.
            return await scheduler.run(
                "load_usd_file",
                lambda: _load(session, target),
                stage_key=session.stage_id_for(target),
            )

        return await server._exec_backend(
            "load_usd_file",
            server.headless_adapter,
            "USD",
            StageInfo,
            _load_call,
            params={"file_path": file_path},
        )

    @server.mcp.tool(
        name="validate_usd_file",
        description=(
            "Open a USD file with pxr and report what is wrong with it; no stage is "
            "kept. Checks: access (exists, permissions, usd.allowed_extensions, "
            "usd.max_file_size_mb, empty), parse and crate errors with pxr's reason, "
            "missing sublayers, references, payloads and other asset paths, "
            "defaultPrim, upAxis and metersPerUnit, physics (joint body targets, "
            "nested rigid bodies or articulation roots, mass, inverted joint limits) "
            "and pxr's UsdValidation validators, with the usdz package rules for a "
            ".usdz. Returns is_valid (no error-severity issue), can_read (pxr opened "
            "it, so load_usd_file can) and issues with severity, code and location. "
            "Paths: sandbox (security.allowed_paths), applied to everything the file "
            "draws on as load_usd_file applies it."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=False
        ),
        output_schema=None,
    )
    @with_param_descriptions()
    async def validate_usd_file(file_path: str) -> ToolResult:
        """
        Open a USD file with pxr and report every problem found.

        Args:
            file_path: Path to USD file

        Returns:
            File validity information or error
        """

        def _validate(target: str) -> Dict[str, Any]:
            report = run_validation(
                target,
                server.settings.usd,
                sandbox=CompositionSandbox(server._path_policy),
            )
            return USDFileInfo(**report.to_dict()).model_dump()

        async def _validate_call(session: "HeadlessUSDSession") -> Dict[str, Any]:
            input_data = server._validate_input(USDValidateRequest, file_path=file_path)
            if isinstance(input_data, dict):
                return input_data
            target = usd_target(input_data.file_path)
            # Touches no loaded stage: it waits only for a running edit.
            return await scheduler.run("validate_usd_file", lambda: _validate(target))

        return await server._exec_backend(
            "validate_usd_file",
            server.headless_adapter,
            "USD",
            USDFileInfo,
            _validate_call,
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
    @with_param_descriptions()
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
            scheduled("get_prim_info", _prim_info, stage_id=stage_id),
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
    @with_param_descriptions()
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
            scheduled("create_prim", _create, writes=True),
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
    @with_param_descriptions()
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
            scheduled("update_prim_attributes", _update, writes=True),
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
    @with_param_descriptions()
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
            scheduled("delete_prim", _delete, writes=True),
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
    @with_param_descriptions()
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
            scheduled("get_mesh_info", _mesh_info, stage_id=stage_id),
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
    @with_param_descriptions()
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
            scheduled("search_prims", _search, stage_id=stage_id),
            params={"stage_id": stage_id, "search_type": search_type, "query": query},
        )

    @server.mcp.tool(
        name="get_bounding_box",
        description=(
            "Get the bounding box of a prim (with its descendants) or of the whole "
            "stage, as USD's BBoxCache computes it: every transform applies and "
            "instances are followed. Only geometry whose purpose is in "
            "include_purposes counts; the default, default and render, is what "
            "renderers and simulators see, so proxy and guide geometry (often "
            "collision shapes) is left out unless asked for. Nothing boundable is "
            "a successful result with has_geometry false and bbox null, never a "
            "guessed box; excluded_purposes_with_geometry then says whether "
            "widening include_purposes would find some."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=False
        ),
        output_schema=None,
    )
    @with_param_descriptions()
    async def get_bounding_box(
        stage_id: str,
        prim_path: Optional[str] = None,
        world_space: bool = True,
        include_purposes: Optional[List[Purpose]] = None,
        time_code: Optional[float] = None,
    ) -> ToolResult:
        """
        Get bounding box for a prim or entire stage.

        Args:
            stage_id: Stage identifier
            prim_path: Absolute prim path; None (or "/") bounds the whole stage
            world_space: True for world coordinates; False for the prim's own
                space, where its descendants' transforms apply but its own and
                its ancestors' do not
            include_purposes: Purposes whose geometry counts, any of default,
                render, proxy, guide; None for default and render
            time_code: Time code to evaluate animated transforms and points at;
                None for the default (unanimated) value

        Returns:
            Bounding box information or error
        """

        def _bbox(session: "HeadlessUSDSession") -> Dict[str, Any]:
            input_data = server._validate_input(
                BBoxRequest,
                stage_id=stage_id,
                prim_path=prim_path,
                world_space=world_space,
                time_code=time_code,
                include_purposes=include_purposes,
            )
            if isinstance(input_data, dict):
                return input_data

            from ...usd.bbox import describe_bounds
            from ...usd.lookup import lookup_prim

            stage = _loaded_stage(session, input_data.stage_id)
            prim = (
                lookup_prim(stage, input_data.prim_path) if input_data.prim_path else None
            )
            bounds = describe_bounds(
                stage,
                prim,
                world_space=input_data.world_space,
                purposes=input_data.include_purposes,
                time_code=input_data.time_code,
            )
            message = None
            if not bounds["has_geometry"]:
                target = input_data.prim_path or "the stage"
                message = (
                    f"No boundable geometry with purpose "
                    f"{'/'.join(bounds['included_purposes'])} at or below {target}."
                )
                if bounds["excluded_purposes_with_geometry"]:
                    message += (
                        " Geometry with purpose "
                        f"{'/'.join(bounds['excluded_purposes_with_geometry'])} is "
                        "there; add it to include_purposes to bound it."
                    )
            return BBoxResponse(
                success=True,
                stage_id=input_data.stage_id,
                prim_path=input_data.prim_path,
                world_space=input_data.world_space,
                message=message,
                **bounds,
            ).model_dump()

        return await server._exec_backend(
            "get_bounding_box",
            server.headless_adapter,
            "USD",
            BBoxResponse,
            scheduled("get_bounding_box", _bbox, stage_id=stage_id),
            params={
                "stage_id": stage_id,
                "prim_path": prim_path,
                "include_purposes": include_purposes,
                "time_code": time_code,
            },
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
    @with_param_descriptions()
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
            scheduled("summarize_scene", _summarize, stage_id=stage_id),
            params={"stage_id": stage_id, "include_meshes": include_meshes, "format": format},
        )
