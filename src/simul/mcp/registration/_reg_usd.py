"""
USD headless file operation tool registration for the Simul MCP server.

Every tool here is a thin closure around ``server._exec_backend``: the
closure validates its input, performs the stage operation and builds the
response model; the envelope owns the rate limit, usage record, sandbox
and exception wrapping, result budget and single transmission.

The pxr work of every tool runs on a worker thread through one
``USDCallScheduler`` per server, so a slow call no longer stalls the event
loop: reads run concurrently (one at a time per stage), and an edit, a load
or a reload runs alone (see ``simul.mcp.usd_scheduler``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional

from fastmcp.tools import ToolResult

from ..schemas.common import ErrorResponse
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
    MatchMode,
    PrimSearchRequest,
    PrimSearchResponse,
    PrimSubtreeRequest,
    PrimSubtreeResponse,
    PrimUpdateRequest,
    Purpose,
    SceneSummaryRequest,
    SceneSummaryResponse,
    SearchType,
    StageDiscardRequest,
    StageExportRequest,
    StageExportResponse,
    StageInfo,
    StageListResponse,
    StageReloadResponse,
    StageSaveRequest,
    StageSaveResponse,
    StageUnloadResponse,
    SummaryFormat,
    USDFileInfo,
    USDFileRequest,
    USDValidateRequest,
)
from ...logging import get_logger
from ...usd.sandbox import CompositionSandbox
from ...usd.validation import WARNING, composition_issues, unreal_content_issues, url_scheme
from ...usd.validation import validate_usd_file as run_validation
from ..usd_scheduler import USDCallScheduler
from ._helpers import with_param_descriptions

if TYPE_CHECKING:
    from ...adapters.headless_usd import HeadlessUSDSession
    from ...usd.editing import USDEditError
    from ..server import SimulMCPServer

SessionWork = Callable[["HeadlessUSDSession"], Dict[str, Any]]
SessionCall = Callable[["HeadlessUSDSession"], Awaitable[Dict[str, Any]]]

# Stage-bound queries an agent makes once per prim while it inspects an
# asset. They count only against the per-agent ceiling
# (security.global_requests_per_minute), not a per-tool bucket of their own.
PER_PRIM_QUERY_TOOLS = frozenset(
    {"get_prim_info", "get_mesh_info", "get_bounding_box", "search_prims"}
)

logger = get_logger(__name__)


def _edit_error(tool_name: str, exc: "USDEditError") -> Dict[str, Any]:
    """Return the error envelope for a refused or failed USD edit, logged with its context."""
    logger.warning("%s refused: %s: %s", tool_name, exc.error_type, exc.message)
    return ErrorResponse(
        error=exc.message, error_type=exc.error_type, details=exc.details or None
    ).model_dump()


def _at(result: Dict[str, Any]) -> str:
    """Name the time code a write targeted, for the confirmation message."""
    time_code = result.get("time_code")
    return "" if time_code is None else f" at time {time_code:g}"


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
            "and pass the stage_id it returns.",
            stage_id=stage_id,
        )
    return stage


def register_usd_tools(server: "SimulMCPServer") -> None:
    """Register USD-related tools."""
    scheduler = USDCallScheduler(server.settings.server.timeout)
    server.exempt_from_per_tool_rate_limit(PER_PRIM_QUERY_TOOLS)

    def request_owner() -> Optional[str]:
        """The MCP session of the current request, read on the event loop.

        Its end is hooked here, so the session's stages are released then.
        """
        session_id = server._get_request_session_id()
        if session_id is not None:
            server._watch_session_end(session_id)
        return session_id

    def scheduled(
        tool_name: str, work: SessionWork, *, stage_id: Optional[str] = None, writes: bool = False
    ) -> SessionCall:
        """Run ``work`` on a worker thread once the scheduler admits it.

        The worker thread cannot see the MCP request, so the calling session
        is read here, on the event loop, and the work runs as that session
        (``HeadlessUSDSession.acting_for``): a stage belongs to the session
        that loaded it.
        """

        async def call(session: "HeadlessUSDSession") -> Dict[str, Any]:
            owner = request_owner()
            acting_for = getattr(session, "acting_for", None)

            def run() -> Dict[str, Any]:
                if acting_for is None:
                    return work(session)
                with acting_for(owner):
                    return work(session)

            return await scheduler.run(tool_name, run, stage_key=stage_id, writes=writes)

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

    from ...usd.editing import USDEditError

    # Stages belong to the MCP session that loaded them (see scheduled above)
    # and are released when that session ends; files a load or reload rereads
    # are held to the composition sandbox.
    adapter = server.headless_adapter
    if adapter is not None and hasattr(adapter, "release_session"):
        server.add_session_end_hook(adapter.release_session)
        adapter.set_sandbox_factory(lambda: CompositionSandbox(server._path_policy))

    @server.mcp.tool(
        name="load_usd_file",
        description=(
            "Load a USD file into a stage private to this MCP session and return stage "
            "information. Loading the same file again returns the same stage while it has "
            "no unsaved edits and its files are unchanged on disk; otherwise a fresh stage "
            "is read from disk, rereading changed sublayers and references (each held to "
            "the sandbox first). Edits stay in memory until save_usd_stage or "
            "export_usd_stage writes them, and are dropped when the session ends. "
            "A file pxr cannot open fails with LoadError, whose details.reason names "
            "the cause (ParseError "
            "with line and column, CorruptFile, EmptyFile, FileTooLarge, "
            "PermissionDenied, UnsupportedExtension, FileNotFound, NotAFile, "
            "UnsupportedURL for a URL no installed asset resolver opens, such as "
            "omniverse:// here). Missing sublayers, references or payloads do not fail "
            "the load; they are listed in warnings. A relative path resolves against "
            "security.relative_path_base (default: the server's working directory); "
            "file_path in the result is the resolved path. Paths: sandbox "
            "(security.allowed_paths), which also holds every sublayer, reference, "
            "payload, clip and asset path the file draws on: one outside it fails the "
            "load with SandboxError listing them. An Unreal content path in an asset value "
            "(/Engine, /Game or /Script .../Name.Name, as Unreal's USD exporter writes "
            "info:unreal:sourceAsset) names an editor object, not a file: it passes and, "
            "with the sandbox on, is listed in warnings as UnrealContentPath."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=False
        ),
        output_schema=None,
        # Foreground only: a stage belongs to the MCP session read from the
        # request as the call runs, which a background task would run without.
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
            # LoadError with details.reason. A session at its stage limit is
            # refused as a RefusedOperation.
            try:
                stage_id = session.load_stage_or_raise(
                    target, sandbox=CompositionSandbox(server._path_policy)
                )
            except USDEditError as exc:
                return _edit_error("load_usd_file", exc)
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
                has_animation=stage_info.has_animation,
                layer_count=len(stage_info.layers),
                default_prim=stage_info.default_prim_path,
                warnings=[
                    issue.to_dict()
                    for issue in composition_issues(stage, WARNING)
                    + unreal_content_issues(session.unreal_content_paths(stage_id))
                ],
            ).model_dump()

        async def _load_call(session: "HeadlessUSDSession") -> Dict[str, Any]:
            input_data = server._validate_input(USDFileRequest, file_path=file_path)
            if isinstance(input_data, dict):
                return input_data
            target = usd_target(input_data.file_path)
            # Alone: a load may reread a changed sublayer or reference in place,
            # which recomposes every stage using it. Its stage id is random and
            # only known once it registers, so no read can be waiting on it.
            return await scheduled("load_usd_file", lambda s: _load(s, target), writes=True)(session)

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
            "draws on as load_usd_file applies it. pxr's UnresolvableDependency and "
            "MissingSourceTypeInRegistry about an Unreal content path (an editor object "
            "Unreal's USD exporter names, not a file) are warnings, not errors."
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
        description=(
            "Get one USD prim: state, applied API schemas, world bounding box, local "
            "transform, children, material bindings, attributes, relationships "
            "(joint bodies, material bindings, collision filters) and metadata "
            "(specifier, instanceable, references/payloads/inherits/specializes "
            "with whether each resolved, variant sets and selections, other "
            "authored metadata). Each attribute is {type, value, authored}; "
            "authored false means the value is the schema fallback. Values are "
            "JSON: vectors are lists, quaternions [w, x, y, z], matrices lists of "
            "rows, and a non-finite float the string \"Infinity\", "
            "\"-Infinity\" or \"NaN\". An array attribute carries its length "
            "and its first array_preview elements, so a large mesh stays small; "
            "use get_mesh_info for geometry statistics."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=False
        ),
        output_schema=None,
    )
    @with_param_descriptions()
    async def get_prim_info(
        stage_id: str,
        prim_path: str,
        time_code: Optional[float] = None,
        include_purposes: Optional[List[Purpose]] = None,
        array_preview: int = 16,
        attribute_filter: Optional[List[str]] = None,
    ) -> ToolResult:
        """
        Get information about a USD prim.

        Args:
            stage_id: Stage identifier
            prim_path: Absolute path to the prim, e.g. /World/Robot
            time_code: Time code to read animated values, the transform and the
                bounding box at; None for the default (unanimated) value
            include_purposes: Purposes the bounding box counts, any of
                default, render, proxy, guide; None for default and render
            array_preview: Leading elements of each array attribute to include,
                0 to 10000; longer arrays report length and truncated
            attribute_filter: Glob patterns such as physics:* or xformOp:*;
                only matching attributes are listed. None lists them all

        Returns:
            Prim information or error
        """

        def _prim_info(session: "HeadlessUSDSession") -> Dict[str, Any]:
            input_data = server._validate_input(
                PrimInfoRequest,
                stage_id=stage_id,
                prim_path=prim_path,
                time_code=time_code,
                include_purposes=include_purposes,
                array_preview=array_preview,
                attribute_filter=attribute_filter,
            )
            if isinstance(input_data, dict):
                return input_data

            from ...usd.lookup import lookup_prim
            from ...usd.reader import describe_prim

            stage = _loaded_stage(session, input_data.stage_id)
            prim = lookup_prim(stage, input_data.prim_path)
            return PrimInfo(
                **describe_prim(
                    prim,
                    time_code=input_data.time_code,
                    purposes=input_data.include_purposes,
                    array_preview=input_data.array_preview,
                    attribute_filter=input_data.attribute_filter,
                )
            ).model_dump()

        return await server._exec_backend(
            "get_prim_info",
            server.headless_adapter,
            "USD",
            PrimInfo,
            scheduled("get_prim_info", _prim_info, stage_id=stage_id),
            params={"stage_id": stage_id, "prim_path": prim_path, "time_code": time_code},
        )

    @server.mcp.tool(
        name="create_prim",
        description=(
            "Define a prim of a concrete USD type (Xform, Scope, Mesh, Cube, Sphere, "
            "PhysicsRevoluteJoint, ...) at an absolute path, with its initial attributes. "
            "Refuses a path that already holds a prim unless redefine_existing=true, which "
            "changes that prim's type and keeps its children, properties and composition "
            "arcs. All or nothing: if the type, the path or any attribute is invalid, "
            "nothing is created and the error names each rejected attribute and why. "
            "Values take the type of the schema attribute (JSON numbers widen, vectors "
            "are lists, quaternions [w, x, y, z], matrices lists of rows, assets path "
            "strings); xformOp:<type> entries create transform ops and add them to "
            "xformOpOrder; attribute_types names the type of a new custom attribute. "
            "The attribute keys active, instanceable and kind set prim metadata."
        ),
        annotations=server._tool_annotations(
            read_only=False, idempotent=False, open_world=False, destructive=True
        ),
        output_schema=None,
    )
    @with_param_descriptions()
    async def create_prim(
        stage_id: str,
        prim_path: str,
        prim_type: str,
        attributes: Optional[Dict[str, Any]] = None,
        redefine_existing: bool = False,
        attribute_types: Optional[Dict[str, str]] = None,
        time_code: Optional[float] = None,
    ) -> ToolResult:
        """
        Create a prim in a USD stage.

        Args:
            stage_id: Stage identifier returned by load_usd_file
            prim_path: Absolute path of the prim to create, e.g. /World/Box; missing
                ancestors are defined as typeless prims
            prim_type: Concrete USD prim type name registered with USD, e.g. Xform or
                Cube; unknown, abstract and API schema names are refused
            attributes: Initial values by attribute name; an existing relationship
                takes a target path or list of paths, and active (bool), instanceable
                (bool) and kind (registered kind name) set prim metadata
            redefine_existing: Allow prim_path to name an existing prim, whose type
                is changed to prim_type
            attribute_types: USD value type name per new attribute, e.g.
                {"custom:offset": "double3", "custom:rot": "quatf"}, or the precision
                of a new transform op ({"xformOp:translate": "float3"}); without it a
                new custom attribute's type is inferred from its value
            time_code: Write the attribute values as time samples at this time code
                instead of as default values

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
                redefine_existing=redefine_existing,
                attribute_types=attribute_types or {},
                time_code=time_code,
            )
            if isinstance(input_data, dict):
                return input_data

            try:
                result = session.create_prim(
                    input_data.stage_id,
                    input_data.prim_path,
                    input_data.prim_type,
                    input_data.attributes,
                    redefine_existing=input_data.redefine_existing,
                    attribute_types=input_data.attribute_types,
                    time_code=input_data.time_code,
                )
            except USDEditError as exc:
                return _edit_error("create_prim", exc)
            verb = "Redefined" if result["redefined"] else "Created"
            return PrimActionResponse(
                success=True,
                stage_id=input_data.stage_id,
                prim_path=result["prim_path"],
                message=f"{verb} {result['prim_type']} prim {result['prim_path']}{_at(result)}",
                changed=result["changed"],
                created_attributes=result["created_attributes"] or None,
                time_code=result["time_code"],
            ).model_dump()

        return await server._exec_backend(
            "create_prim",
            server.headless_adapter,
            "USD",
            PrimActionResponse,
            scheduled("create_prim", _create, writes=True),
            params={
                "stage_id": stage_id,
                "prim_path": prim_path,
                "prim_type": prim_type,
                "redefine_existing": redefine_existing,
                "time_code": time_code,
            },
        )

    @server.mcp.tool(
        name="update_prim_attributes",
        description=(
            "Set attribute values on an existing prim, plus relationship targets and the "
            "active / instanceable / kind prim metadata. All or nothing: every entry is "
            "checked first (name, type, allowed tokens), and if any is invalid nothing is "
            "changed and the error lists each rejected key with its reason. Values take "
            "the attribute's type (JSON numbers widen, vectors are lists, quaternions "
            "[w, x, y, z], matrices lists of rows, assets path strings); a new "
            "xformOp:<type> creates the transform op and adds it to xformOpOrder. An "
            "animated attribute needs time_code, since a default value is hidden by its "
            "time samples. Refuses instance proxies (prims inside an instanceable prim)."
        ),
        annotations=server._tool_annotations(
            read_only=False, idempotent=True, open_world=False, destructive=True
        ),
        output_schema=None,
    )
    @with_param_descriptions()
    async def update_prim_attributes(
        stage_id: str,
        prim_path: str,
        attributes: Dict[str, Any],
        attribute_types: Optional[Dict[str, str]] = None,
        time_code: Optional[float] = None,
    ) -> ToolResult:
        """
        Update attributes on a USD prim.

        Args:
            stage_id: Stage identifier returned by load_usd_file
            prim_path: Absolute path of an existing prim
            attributes: Values by attribute name; an existing relationship takes a
                target path or list of paths, and active (bool), instanceable (bool)
                and kind (registered kind name) set prim metadata
            attribute_types: USD value type name per new attribute, e.g.
                {"custom:offset": "double3"}, or the precision of a new transform op;
                an existing attribute keeps its type
            time_code: Write the values as time samples at this time code; required
                for attributes that already have time samples, refused for uniform
                attributes, relationships and metadata

        Returns:
            Action confirmation or error
        """

        def _update(session: "HeadlessUSDSession") -> Dict[str, Any]:
            input_data = server._validate_input(
                PrimUpdateRequest,
                stage_id=stage_id,
                prim_path=prim_path,
                attributes=attributes,
                attribute_types=attribute_types or {},
                time_code=time_code,
            )
            if isinstance(input_data, dict):
                return input_data

            try:
                result = session.update_prim_attributes(
                    input_data.stage_id,
                    input_data.prim_path,
                    input_data.attributes,
                    attribute_types=input_data.attribute_types,
                    time_code=input_data.time_code,
                )
            except USDEditError as exc:
                return _edit_error("update_prim_attributes", exc)
            return PrimActionResponse(
                success=True,
                stage_id=input_data.stage_id,
                prim_path=result["prim_path"],
                message=f"Updated prim {result['prim_path']}{_at(result)}",
                changed=result["changed"],
                created_attributes=result["created_attributes"] or None,
                time_code=result["time_code"],
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
                "time_code": time_code,
            },
        )

    @server.mcp.tool(
        name="delete_prim",
        description=(
            "Delete a prim and its subtree from the stage's edit target (the root layer). "
            "Needs an absolute path to an existing prim. Refuses the pseudo-root '/', and "
            "the stage's default prim unless allow_root_delete=true. Refuses a prim that "
            "would survive the delete because it also comes from a sublayer or from a "
            "reference or payload on an ancestor; deactivate it instead with "
            'update_prim_attributes {"active": false}. A refused delete changes nothing.'
        ),
        annotations=server._tool_annotations(
            read_only=False, idempotent=True, open_world=False, destructive=True
        ),
        output_schema=None,
    )
    @with_param_descriptions()
    async def delete_prim(
        stage_id: str, prim_path: str, allow_root_delete: bool = False
    ) -> ToolResult:
        """
        Delete a prim from a USD stage.

        Args:
            stage_id: Stage identifier returned by load_usd_file
            prim_path: Absolute path of an existing prim
            allow_root_delete: Permit deleting the stage's default prim (or an
                ancestor of it), which removes the whole scene

        Returns:
            Action confirmation or error
        """

        def _delete(session: "HeadlessUSDSession") -> Dict[str, Any]:
            input_data = server._validate_input(
                PrimDeleteRequest,
                stage_id=stage_id,
                prim_path=prim_path,
                allow_root_delete=allow_root_delete,
            )
            if isinstance(input_data, dict):
                return input_data

            try:
                result = session.delete_prim(
                    input_data.stage_id,
                    input_data.prim_path,
                    allow_root_delete=input_data.allow_root_delete,
                )
            except USDEditError as exc:
                return _edit_error("delete_prim", exc)
            removed = result["removed_prims"]
            below = f" and {removed - 1} descendant(s)" if removed > 1 else ""
            return PrimActionResponse(
                success=True,
                stage_id=input_data.stage_id,
                prim_path=result["prim_path"],
                message=f"Deleted prim {result['prim_path']}{below}",
            ).model_dump()

        return await server._exec_backend(
            "delete_prim",
            server.headless_adapter,
            "USD",
            PrimActionResponse,
            scheduled("delete_prim", _delete, writes=True),
            params={
                "stage_id": stage_id,
                "prim_path": prim_path,
                "allow_root_delete": allow_root_delete,
            },
        )

    @server.mcp.tool(
        name="get_mesh_info",
        description=(
            "Get statistics for a Mesh prim: vertex, face and face-vertex counts; "
            "normals (primvars:normals or normals), UV sets, colours, tangents; "
            "topology validity, watertightness, surface area and (closed meshes "
            "only) volume, and the points' bounds, all in the mesh's own "
            "space; the materials UsdShade resolves for the mesh and for each "
            "materialBind geometry subset, and its physics material. Computed with "
            "numpy, so a million-vertex mesh takes well under a second. time_code "
            "reads a deforming mesh's points; points_time_samples says whether it "
            "deforms."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=False
        ),
        output_schema=None,
    )
    @with_param_descriptions()
    async def get_mesh_info(
        stage_id: str, prim_path: str, time_code: Optional[float] = None
    ) -> ToolResult:
        """
        Get mesh information for a mesh prim.

        Args:
            stage_id: Stage identifier
            prim_path: Absolute path to the Mesh prim
            time_code: Time code to read points and topology at; None for the
                default (unanimated) value

        Returns:
            Mesh statistics or error
        """

        def _mesh_info(session: "HeadlessUSDSession") -> Dict[str, Any]:
            input_data = server._validate_input(
                MeshInfoRequest, stage_id=stage_id, prim_path=prim_path, time_code=time_code
            )
            if isinstance(input_data, dict):
                return input_data

            from ...usd._pxr import UsdGeom
            from ...usd.lookup import lookup_prim
            from ...usd.mesh_ops import MeshOperations, mesh_info_to_dict

            stage = _loaded_stage(session, input_data.stage_id)
            prim = lookup_prim(stage, input_data.prim_path)
            if not prim.IsA(UsdGeom.Mesh):
                return ErrorResponse(
                    error=(
                        f"Mesh not found: {input_data.prim_path} is a "
                        f"{prim.GetTypeName() or 'typeless prim'}, not a Mesh"
                    ),
                    error_type="NotFoundError",
                    details={"prim_path": input_data.prim_path, "prim_type": prim.GetTypeName()},
                ).model_dump()
            info = MeshOperations().get_mesh_statistics(prim, input_data.time_code)
            return MeshInfo(**mesh_info_to_dict(info)).model_dump()

        return await server._exec_backend(
            "get_mesh_info",
            server.headless_adapter,
            "USD",
            MeshInfo,
            scheduled("get_mesh_info", _mesh_info, stage_id=stage_id),
            params={"stage_id": stage_id, "prim_path": prim_path, "time_code": time_code},
        )

    @server.mcp.tool(
        name="search_prims",
        description=(
            "Search a USD stage for prims, one page at a time. search_type picks "
            "what the query is matched against: by_name (prim name), by_path "
            "(full path), by_type (prim type: with match auto a registered schema "
            "matches its whole family, so PhysicsJoint finds every joint and "
            "Gprim every geometric primitive) or by_api_schema (applied API "
            "schemas: PhysicsRigidBodyAPI; PhysicsDriveAPI also finds "
            "PhysicsDriveAPI:angular). match is auto, exact, substring, glob or "
            "regex; names compare case-insensitively unless case_sensitive. "
            "Instances are searched (include_instance_proxies), root_path limits "
            "the search to a subtree, and offset/limit page the matches: "
            "total_matches counts them all and next_offset starts the next page. "
            "An empty by_name query lists every prim."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=False
        ),
        output_schema=None,
    )
    @with_param_descriptions()
    async def search_prims(
        stage_id: str,
        search_type: SearchType,
        query: str,
        exact_match: bool = False,
        match: MatchMode = "auto",
        case_sensitive: bool = False,
        root_path: Optional[str] = None,
        include_instance_proxies: bool = True,
        offset: int = 0,
        limit: int = 200,
    ) -> ToolResult:
        """
        Search for prims in a USD stage.

        Args:
            stage_id: Stage identifier
            search_type: by_name, by_path, by_type or by_api_schema
            query: Name, path, type or API schema to find; may be empty for
                by_name and by_path (matches every prim)
            exact_match: Same as match="exact"; kept for older callers
            match: auto (substring for names and paths, the schema family for
                types, the schema and its instances for API schemas), exact,
                substring, glob (* and ?) or regex
            case_sensitive: Compare case-sensitively (default: ignore case)
            root_path: Absolute prim path to search under (the prim itself
                included); None searches the whole stage
            include_instance_proxies: Search the prims inside instanceable
                prims (default true)
            offset: Matches to skip, in traversal order; pass the previous
                page's next_offset to continue
            limit: Matches per page, 1 to 5000

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
                match=match,
                case_sensitive=case_sensitive,
                root_path=root_path,
                include_instance_proxies=include_instance_proxies,
                offset=offset,
                limit=limit,
            )
            if isinstance(input_data, dict):
                return input_data

            from ...usd.lookup import lookup_prim
            from ...usd.search import search_prims as run_search

            stage = _loaded_stage(session, input_data.stage_id)
            root = lookup_prim(stage, input_data.root_path) if input_data.root_path else None
            mode = "exact" if input_data.exact_match and input_data.match == "auto" else input_data.match
            page = run_search(
                stage,
                input_data.search_type,
                input_data.query,
                match=mode,
                case_sensitive=input_data.case_sensitive,
                root=root,
                include_instance_proxies=input_data.include_instance_proxies,
                offset=input_data.offset,
                limit=input_data.limit,
            )
            return PrimSearchResponse(
                success=True,
                stage_id=input_data.stage_id,
                search_type=input_data.search_type,
                query=input_data.query,
                match=mode,
                case_sensitive=input_data.case_sensitive,
                root_path=input_data.root_path,
                include_instance_proxies=input_data.include_instance_proxies,
                **page,
            ).model_dump()

        return await server._exec_backend(
            "search_prims",
            server.headless_adapter,
            "USD",
            PrimSearchResponse,
            scheduled("search_prims", _search, stage_id=stage_id),
            params={
                "stage_id": stage_id,
                "search_type": search_type,
                "query": query,
                "match": match,
                "root_path": root_path,
                "offset": offset,
            },
        )

    @server.mcp.tool(
        name="get_prim_subtree",
        description=(
            "List a prim and its descendants (a robot, an imported asset) in one "
            "call: a flat, depth-first page with path, type, depth, child count "
            "and applied API schemas per prim, instances included. With "
            "include_properties each prim also carries its authored attributes "
            "({type, value, authored}) and relationship targets (joint bodies, "
            "material bindings), so a whole articulation is read without one "
            "get_prim_info call per link. Paged with max_results/offset; total "
            "counts the subtree and next_offset starts the next page."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=False
        ),
        output_schema=None,
    )
    @with_param_descriptions()
    async def get_prim_subtree(
        stage_id: str,
        root_path: str = "/",
        max_depth: int = 5,
        max_results: int = 150,
        offset: int = 0,
        include_instance_proxies: bool = True,
        include_properties: bool = False,
        attribute_filter: Optional[List[str]] = None,
        array_preview: int = 4,
    ) -> ToolResult:
        """
        List the prims under one prim.

        Args:
            stage_id: Stage identifier
            root_path: Absolute path of the subtree root (listed at depth 0);
                "/" lists the whole stage
            max_depth: Deepest level below root_path to list, 0 to 64
            max_results: Prims per page, 1 to 1000
            offset: Prims to skip, in traversal order; pass the previous
                page's next_offset to continue
            include_instance_proxies: Descend into instanceable prims
                (default true)
            include_properties: Add each prim's authored attributes and
                authored relationship targets
            attribute_filter: Glob patterns (physics:*) limiting the
                properties include_properties adds
            array_preview: Leading elements of array attributes to include,
                0 to 10000

        Returns:
            One page of the subtree or error
        """

        def _subtree(session: "HeadlessUSDSession") -> Dict[str, Any]:
            input_data = server._validate_input(
                PrimSubtreeRequest,
                stage_id=stage_id,
                root_path=root_path,
                max_depth=max_depth,
                max_results=max_results,
                offset=offset,
                include_instance_proxies=include_instance_proxies,
                include_properties=include_properties,
                attribute_filter=attribute_filter,
                array_preview=array_preview,
            )
            if isinstance(input_data, dict):
                return input_data

            from ...usd.lookup import lookup_prim
            from ...usd.search import list_subtree

            stage = _loaded_stage(session, input_data.stage_id)
            root = lookup_prim(stage, input_data.root_path)
            page = list_subtree(
                stage,
                root,
                max_depth=input_data.max_depth,
                include_instance_proxies=input_data.include_instance_proxies,
                include_properties=input_data.include_properties,
                attribute_filter=input_data.attribute_filter,
                array_preview=input_data.array_preview,
                offset=input_data.offset,
                limit=input_data.max_results,
            )
            return PrimSubtreeResponse(
                success=True, stage_id=input_data.stage_id, **page
            ).model_dump()

        return await server._exec_backend(
            "get_prim_subtree",
            server.headless_adapter,
            "USD",
            PrimSubtreeResponse,
            scheduled("get_prim_subtree", _subtree, stage_id=stage_id),
            params={"stage_id": stage_id, "root_path": root_path, "offset": offset},
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
        description=(
            "Summarise a USD stage in one call: prim counts by type (instance "
            "proxies included; '(typeless)' for prims without a type), the scene "
            "bounds and size in stage units and in metres, mesh totals (vertices, "
            "faces, normals, UVs, closed meshes; surface area and volume summed in "
            "each mesh's own space, before transforms; instanced meshes counted per "
            "instance; a deforming mesh with time-sampled points measured at its "
            "first sample), materials, root prims, and animation: detected from "
            "time samples spanning more than one time even without an authored "
            "time range, with its duration in seconds. warnings groups the layers "
            "whose upAxis or metersPerUnit differs from the stage's (USD does not "
            "rescale them), with a count and example layers per group. format "
            "text adds a short digest."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=False
        ),
        output_schema=None,
        # Foreground only, like load_usd_file: the stage is looked up for the
        # MCP session of the request.
    )
    @with_param_descriptions()
    async def summarize_scene(
        stage_id: str, include_meshes: bool = True, format: SummaryFormat = "json"
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
                    "scene_size_meters": summary.scene_size_meters,
                    "mesh_statistics": summary.mesh_statistics,
                    "hierarchy_depth": summary.hierarchy_depth,
                    "animation_info": summary.animation_info,
                    "root_prim_count": summary.root_prim_count,
                    "root_prims": summary.root_prim_paths,
                    "material_count": len(summary.materials),
                    "materials": [material.path for material in summary.materials[:50]],
                    "warnings": summary.warnings,
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

    @server.mcp.tool(
        name="list_usd_stages",
        description=(
            "List the USD stages this MCP session has loaded: each one's stage_id, file, "
            "whether it has unsaved edits, and which of its files changed on disk since it "
            "was read. Stages are private to the session that loaded them, and edits live "
            "in memory until save_usd_stage or export_usd_stage writes them."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=False
        ),
        output_schema=None,
    )
    @with_param_descriptions()
    async def list_usd_stages() -> ToolResult:
        """
        List the loaded USD stages.

        Returns:
            The stages with their unsaved and on-disk state
        """

        def _list(session: "HeadlessUSDSession") -> Dict[str, Any]:
            stages = [session.stage_status(stage_id) for stage_id in session.list_stages()]
            return StageListResponse(success=True, stages=stages, count=len(stages)).model_dump()

        return await server._exec_backend(
            "list_usd_stages",
            server.headless_adapter,
            "USD",
            StageListResponse,
            scheduled("list_usd_stages", _list),
            params={},
        )

    @server.mcp.tool(
        name="save_usd_stage",
        description=(
            "Save a stage's edits to its own file: the root layer, which every edit "
            "targets. Paths: sandbox (security.allowed_paths); the file must be writable "
            "there. Refuses a .usdz root (use export_usd_stage) and a file that changed on "
            "disk after the stage read it, unless overwrite_external_changes=true. A stage "
            "with no unsaved edits writes nothing. Reports the files written."
        ),
        annotations=server._tool_annotations(
            read_only=False, idempotent=True, open_world=False, destructive=True
        ),
        output_schema=None,
    )
    @with_param_descriptions()
    async def save_usd_stage(
        stage_id: str, overwrite_external_changes: bool = False
    ) -> ToolResult:
        """
        Save a USD stage in place.

        Args:
            stage_id: Stage identifier returned by load_usd_file
            overwrite_external_changes: Save even though the file changed on disk after
                the stage read it, replacing those changes

        Returns:
            The files written, or error
        """

        def _save(session: "HeadlessUSDSession") -> Dict[str, Any]:
            input_data = server._validate_input(
                StageSaveRequest,
                stage_id=stage_id,
                overwrite_external_changes=overwrite_external_changes,
            )
            if isinstance(input_data, dict):
                return input_data
            try:
                server._path_policy.authorize(
                    session.stage_file_path(input_data.stage_id), write=True
                )
                result = session.save_stage(
                    input_data.stage_id,
                    overwrite_external_changes=input_data.overwrite_external_changes,
                )
            except USDEditError as exc:
                return _edit_error("save_usd_stage", exc)
            message = (
                f"Saved {result['file_path']}"
                if result["saved"]
                else f"No unsaved edits; {result['file_path']} was not written"
            )
            return StageSaveResponse(success=True, message=message, **result).model_dump()

        return await server._exec_backend(
            "save_usd_stage",
            server.headless_adapter,
            "USD",
            StageSaveResponse,
            scheduled("save_usd_stage", _save, stage_id=stage_id),
            params={
                "stage_id": stage_id,
                "overwrite_external_changes": overwrite_external_changes,
            },
        )

    @server.mcp.tool(
        name="export_usd_stage",
        description=(
            "Write a stage to a new file: .usda, .usdc, .usd, or .usdz (packaged with the "
            "files it depends on). flatten=true writes the composed stage as one layer; "
            "otherwise the root layer is written with its sublayers, references and "
            "payloads kept as composition arcs and relative asset paths re-anchored to the "
            "new location. Refuses an existing file unless overwrite=true. The stage stays "
            "bound to its own file and keeps its unsaved state. "
            "Paths: sandbox (security.allowed_paths)."
        ),
        annotations=server._tool_annotations(
            read_only=False, idempotent=False, open_world=False, destructive=True
        ),
        output_schema=None,
    )
    @with_param_descriptions()
    async def export_usd_stage(
        stage_id: str, file_path: str, flatten: bool = False, overwrite: bool = False
    ) -> ToolResult:
        """
        Export a USD stage to a new file.

        Args:
            stage_id: Stage identifier returned by load_usd_file
            file_path: Destination file; its extension (.usda, .usdc, .usd, .usdz) picks
                the format, and its directory must exist
            flatten: Write the composed stage (all layers and arcs) as a single layer
            overwrite: Replace the destination if it already exists

        Returns:
            The file written, or error
        """

        def _export(session: "HeadlessUSDSession") -> Dict[str, Any]:
            input_data = server._validate_input(
                StageExportRequest,
                stage_id=stage_id,
                file_path=file_path,
                flatten=flatten,
                overwrite=overwrite,
            )
            if isinstance(input_data, dict):
                return input_data
            destination = server._path_policy.authorize(input_data.file_path, write=True)
            try:
                result = session.export_stage(
                    input_data.stage_id,
                    destination,
                    flatten=input_data.flatten,
                    overwrite=input_data.overwrite,
                )
            except USDEditError as exc:
                return _edit_error("export_usd_stage", exc)
            how = "flattened " if result["flatten"] else ""
            message = f"Wrote {how}{result['format']} file {result['file_path']}"
            return StageExportResponse(success=True, message=message, **result).model_dump()

        return await server._exec_backend(
            "export_usd_stage",
            server.headless_adapter,
            "USD",
            StageExportResponse,
            scheduled("export_usd_stage", _export, stage_id=stage_id),
            params={
                "stage_id": stage_id,
                "file_path": file_path,
                "flatten": flatten,
                "overwrite": overwrite,
            },
        )

    @server.mcp.tool(
        name="reload_usd_stage",
        description=(
            "Reread a stage from disk: its root file, and any sublayer or referenced file "
            "that changed. Use it after the files were edited outside this server, or to "
            "drop edits. Refuses a stage with unsaved edits unless discard_unsaved=true."
        ),
        annotations=server._tool_annotations(
            read_only=False, idempotent=True, open_world=False, destructive=True
        ),
        output_schema=None,
    )
    @with_param_descriptions()
    async def reload_usd_stage(stage_id: str, discard_unsaved: bool = False) -> ToolResult:
        """
        Reload a USD stage from disk.

        Args:
            stage_id: Stage identifier returned by load_usd_file
            discard_unsaved: Reload even though the stage has unsaved edits, which are lost

        Returns:
            The files reread, or error
        """

        def _reload(session: "HeadlessUSDSession") -> Dict[str, Any]:
            input_data = server._validate_input(
                StageDiscardRequest, stage_id=stage_id, discard_unsaved=discard_unsaved
            )
            if isinstance(input_data, dict):
                return input_data
            try:
                result = session.reload_stage(
                    input_data.stage_id, discard_unsaved=input_data.discard_unsaved
                )
            except USDEditError as exc:
                return _edit_error("reload_usd_stage", exc)
            dropped = "; unsaved edits were discarded" if result["discarded_unsaved"] else ""
            message = f"Reloaded {result['file_path']}{dropped}"
            return StageReloadResponse(success=True, message=message, **result).model_dump()

        return await server._exec_backend(
            "reload_usd_stage",
            server.headless_adapter,
            "USD",
            StageReloadResponse,
            scheduled("reload_usd_stage", _reload, writes=True),
            params={"stage_id": stage_id, "discard_unsaved": discard_unsaved},
        )

    @server.mcp.tool(
        name="unload_usd_stage",
        description=(
            "Close a loaded stage and free its memory; its stage_id stops working. "
            "Refuses a stage with unsaved edits unless discard_unsaved=true."
        ),
        annotations=server._tool_annotations(
            read_only=False, idempotent=False, open_world=False, destructive=True
        ),
        output_schema=None,
    )
    @with_param_descriptions()
    async def unload_usd_stage(stage_id: str, discard_unsaved: bool = False) -> ToolResult:
        """
        Unload a USD stage.

        Args:
            stage_id: Stage identifier returned by load_usd_file
            discard_unsaved: Unload even though the stage has unsaved edits, which are lost

        Returns:
            Confirmation, or error
        """

        def _unload(session: "HeadlessUSDSession") -> Dict[str, Any]:
            input_data = server._validate_input(
                StageDiscardRequest, stage_id=stage_id, discard_unsaved=discard_unsaved
            )
            if isinstance(input_data, dict):
                return input_data
            try:
                dirty = session.stage_status(input_data.stage_id)["has_unsaved_changes"]
                session.unload_stage(
                    input_data.stage_id, discard_unsaved=input_data.discard_unsaved
                )
            except USDEditError as exc:
                return _edit_error("unload_usd_stage", exc)
            dropped = "; unsaved edits were discarded" if dirty else ""
            return StageUnloadResponse(
                success=True,
                stage_id=input_data.stage_id,
                discarded_unsaved=dirty,
                message=f"Unloaded stage {input_data.stage_id}{dropped}",
            ).model_dump()

        return await server._exec_backend(
            "unload_usd_stage",
            server.headless_adapter,
            "USD",
            StageUnloadResponse,
            scheduled("unload_usd_stage", _unload, stage_id=stage_id),
            params={"stage_id": stage_id, "discard_unsaved": discard_unsaved},
        )
