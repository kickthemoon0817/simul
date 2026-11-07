"""
USD headless file operation tool registration for the Simul MCP server.

Every tool here is a thin closure around ``server._exec_backend``: the
closure validates its input, performs the stage operation and builds the
response model; the envelope owns the rate limit, usage record, sandbox
and exception wrapping, result budget and single transmission.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

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
    SummaryFormat,
    StageInfo,
    USDFileInfo,
    USDFileRequest,
    USDValidateRequest,
)
from ...logging import get_logger
from ...usd.validation import WARNING, composition_issues
from ...usd.validation import validate_usd_file as run_validation
from ._helpers import with_param_descriptions

if TYPE_CHECKING:
    from ...adapters.headless_usd import HeadlessUSDSession
    from ...usd.editing import USDEditError
    from ..server import SimulMCPServer

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
            "and pass the stage_id it returns."
        )
    return stage


def register_usd_tools(server: "SimulMCPServer") -> None:
    """Register USD-related tools."""
    from ...usd.editing import USDEditError

    @server.mcp.tool(
        name="load_usd_file",
        description=(
            "Load a USD file and return stage information. A file pxr cannot open "
            "fails with LoadError, whose details.reason names the cause (ParseError "
            "with line and column, CorruptFile, EmptyFile, FileTooLarge, "
            "PermissionDenied, UnsupportedExtension, FileNotFound, NotAFile). Missing "
            "sublayers, references or payloads do not fail the load; they are listed "
            "in warnings. Paths: sandbox (security.allowed_paths)."
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

        def _load(session: "HeadlessUSDSession") -> Dict[str, Any]:
            input_data = server._validate_input(USDFileRequest, file_path=file_path)
            if isinstance(input_data, dict):
                return input_data

            # A USDLoadError carries pxr's reason; the envelope reports it as
            # LoadError with details.reason.
            stage_id = session.load_stage_or_raise(
                server._path_policy.authorize(input_data.file_path)
            )
            stage = session.get_stage(stage_id)
            stage_info = session.get_stage_info(stage_id)
            if stage is None or stage_info is None:
                return ErrorResponse(
                    error=f"Loaded {input_data.file_path} but could not read its stage information",
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
                has_animation=stage_info.has_animation,
                layer_count=len(stage_info.layers),
                default_prim=stage_info.default_prim_path,
                warnings=[issue.to_dict() for issue in composition_issues(stage, WARNING)],
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
            "Open a USD file with pxr and report what is wrong with it; no stage is "
            "kept. Checks: access (exists, permissions, usd.allowed_extensions, "
            "usd.max_file_size_mb, empty), parse and crate errors with pxr's reason, "
            "missing sublayers, references, payloads and other asset paths, "
            "defaultPrim, upAxis and metersPerUnit, physics (joint body targets, "
            "nested rigid bodies or articulation roots, mass, inverted joint limits) "
            "and pxr's UsdValidation validators, with the usdz package rules for a "
            ".usdz. Returns is_valid (no error-severity issue), can_read (pxr opened "
            "it, so load_usd_file can) and issues with severity, code and location. "
            "Paths: sandbox (security.allowed_paths)."
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

        def _validate(session: "HeadlessUSDSession") -> Dict[str, Any]:
            input_data = server._validate_input(USDValidateRequest, file_path=file_path)
            if isinstance(input_data, dict):
                return input_data

            path = Path(server._path_policy.authorize(input_data.file_path))
            report = run_validation(str(path.resolve()), server.settings.usd)
            return USDFileInfo(**report.to_dict()).model_dump()

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
            _prim_info,
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
            _create,
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
            _update,
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
            read_only=False, idempotent=False, open_world=False, destructive=True
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
            _delete,
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
                ).model_dump()
            info = MeshOperations().get_mesh_statistics(prim, input_data.time_code)
            return MeshInfo(**mesh_info_to_dict(info)).model_dump()

        return await server._exec_backend(
            "get_mesh_info",
            server.headless_adapter,
            "USD",
            MeshInfo,
            _mesh_info,
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
            _search,
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
            _subtree,
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
            _bbox,
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
            "faces, normals, UVs, closed meshes, surface area, volume; instanced "
            "meshes counted per instance), materials, root prims, and animation: "
            "detected from time samples even without an authored time range, with "
            "its duration in seconds. warnings flags layers whose upAxis or "
            "metersPerUnit differs from the stage's (USD does not rescale them). "
            "format text adds a short digest."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=False
        ),
        output_schema=None,
        task=server._task_optional(),
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
            _summarize,
            params={"stage_id": stage_id, "include_meshes": include_meshes, "format": format},
        )
