"""
Blender runtime tool registration for the Simul MCP server.

Each Blender tool is declared once, as a ``BlenderTool`` entry in
``BLENDER_TOOLS``: its name, description and safety hints, the
``BlenderRuntimeSession`` operation it calls, and its request and response
models. ``register_blender_tools`` turns the entries into FastMCP tools, in
table order:

- the parameters, their defaults and their descriptions come from the
  operation's signature and docstring, except where the entry advertises a
  parameter differently (``params``) or words it differently
  (``descriptions``);
- every call runs through the server's shared envelope, which validates the
  arguments against the request model (so invalid input is rate limited and
  recorded like any other failed call) and forwards the validated values to
  the operation by name.

A tool whose call is more than that forwarding writes its own wrapper
(``build``); its entry still supplies the name, description, hints and
models, so nothing about a tool is stated twice.

The table lives here rather than as decorators on ``BlenderRuntimeSession``:
``simul.adapters.blender_runtime`` ships inside the Blender add-on, which must
not import ``simul.mcp``.
"""

from __future__ import annotations

import inspect
import typing
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
)

from fastmcp.tools import ToolResult
from pydantic import BaseModel

from ...adapters.blender_runtime import BlenderRuntimeSession
from ...tool_surfaces import THIN_BLENDER_TOOLS
from ..schemas.blender import (
    LISTING_JSON_BYTES_MAX,
    SCRIPT_OUTPUT_CHARS_DEFAULT,
    SCRIPT_OUTPUT_CHARS_MAX,
    SCRIPT_OUTPUT_JSON_BYTES_MAX,
    SEQUENCE_FRAMES_DEFAULT,
    SEQUENCE_FRAMES_MAX,
    SEQUENCE_TIME_BUDGET_SECONDS,
    BlenderAddConstraintRequest,
    BlenderAddConstraintResponse,
    BlenderAddForceFieldRequest,
    BlenderAddForceFieldResponse,
    BlenderAddModifierRequest,
    BlenderAddModifierResponse,
    BlenderApplyModifierResponse,
    BlenderAssignMaterialRequest,
    BlenderAssignMaterialResponse,
    BlenderAttachWindowRequest,
    BlenderAttachWindowResponse,
    BlenderBakeScope,
    BlenderBakeSimulationRequest,
    BlenderBakeSimulationResponse,
    BlenderBoundingBoxRequest,
    BlenderBoundingBoxResponse,
    BlenderBoundsCheckRequest,
    BlenderBoundsCheckResponse,
    BlenderCameraInfoResponse,
    BlenderCaptureMethod,
    BlenderCaptureSequenceRequest,
    BlenderCaptureSequenceResponse,
    BlenderCaptureViewportRequest,
    BlenderCaptureViewportResponse,
    BlenderCleanMeshRequest,
    BlenderCleanMeshResponse,
    BlenderClearParentRequest,
    BlenderClearParentResponse,
    BlenderCreateMeshFromDataRequest,
    BlenderCreateMeshFromDataResponse,
    BlenderCreateObjectRequest,
    BlenderCreateObjectResponse,
    BlenderDeleteKeyframeRequest,
    BlenderDeleteKeyframeResponse,
    BlenderDeleteObjectRequest,
    BlenderDeleteObjectResponse,
    BlenderDistanceRequest,
    BlenderDistanceResponse,
    BlenderExecuteScriptRequest,
    BlenderExecuteScriptResponse,
    BlenderExportFileRequest,
    BlenderExportFileResponse,
    BlenderFileInfoResponse,
    BlenderFocusOnObjectRequest,
    BlenderFocusOnObjectResponse,
    BlenderFreeBakeRequest,
    BlenderFreeBakeResponse,
    BlenderGetConstraintInfoRequest,
    BlenderGetConstraintInfoResponse,
    BlenderGetForceFieldInfoRequest,
    BlenderGetForceFieldInfoResponse,
    BlenderGetFrameResponse,
    BlenderGetKeyframesRequest,
    BlenderGetKeyframesResponse,
    BlenderGetPhysicsStateRequest,
    BlenderGetPhysicsStateResponse,
    BlenderGetTrajectoryRequest,
    BlenderGetTrajectoryResponse,
    BlenderImportFileRequest,
    BlenderImportFileResponse,
    BlenderInfoResponse,
    BlenderInsertKeyframeRequest,
    BlenderInsertKeyframeResponse,
    BlenderMaterialInfoRequest,
    BlenderMaterialInfoResponse,
    BlenderMeshInfoRequest,
    BlenderMeshInfoResponse,
    BlenderModifierTargetRequest,
    BlenderObjectInfoRequest,
    BlenderObjectInfoResponse,
    BlenderOpenFileRequest,
    BlenderOpenFileResponse,
    BlenderPlayAnimationRequest,
    BlenderPlayAnimationResponse,
    BlenderRemoveModifierResponse,
    BlenderRigidBodyWorldResponse,
    BlenderSaveFileRequest,
    BlenderSaveFileResponse,
    BlenderSceneObjectsRequest,
    BlenderSceneObjectsResponse,
    BlenderSceneSummaryResponse,
    BlenderSearchObjectsRequest,
    BlenderSearchObjectsResponse,
    BlenderSetCameraViewRequest,
    BlenderSetCameraViewResponse,
    BlenderSetFrameRangeRequest,
    BlenderSetFrameRangeResponse,
    BlenderSetFrameRequest,
    BlenderSetFrameResponse,
    BlenderSetLightParamsRequest,
    BlenderSetLightParamsResponse,
    BlenderSetObjectModeRequest,
    BlenderSetObjectModeResponse,
    BlenderSetParentRequest,
    BlenderSetParentResponse,
    BlenderSetRigidBodyWorldRequest,
    BlenderSetTransformRequest,
    BlenderSetTransformResponse,
    BlenderSetupRigidBodyRequest,
    BlenderSetupRigidBodyResponse,
    BlenderViewportInfoResponse,
)
from ..schemas.blender_ui import AgentControl, BlenderUIRequest, BlenderUIResponse
from ..schemas.simready import (
    SimReadyApplyMetadataRequest,
    SimReadyApplyMetadataResponse,
    SimReadyCheckName,
    SimReadyClearName,
    SimReadyExportRequest,
    SimReadyExportResponse,
    SimReadyGetMetadataRequest,
    SimReadyGetMetadataResponse,
    SimReadyMetadata,
    SimReadySemanticLabels,
    SimReadySetupHierarchyRequest,
    SimReadySetupHierarchyResponse,
    SimReadySeverity,
    SimReadyValidateRequest,
    SimReadyValidateResponse,
)
from ._helpers import surface_tool, with_param_descriptions

if TYPE_CHECKING:
    from ...config import Settings
    from ..server import SimulMCPServer

ToolWrapper = Callable[..., Coroutine[Any, Any, ToolResult]]
# Called with the open session (and the validated request when the tool has a
# request model); returns the payload.
SessionCall = Callable[..., Dict[str, Any]]


@dataclass(frozen=True)
class Hints:
    """The MCP safety hints of a tool.

    Every Blender tool talks to a Blender process, so ``openWorldHint`` is
    always left at its default.
    """

    read_only: bool
    idempotent: bool
    destructive: bool = False


#: Reads Blender state and changes nothing.
READ_ONLY = Hints(read_only=True, idempotent=True)
#: Adds to the scene or starts something; repeating the call does it again.
ADDITIVE = Hints(read_only=False, idempotent=False)
#: Changes the view, frame or attachment target; repeating the call changes nothing more.
IDEMPOTENT = Hints(read_only=False, idempotent=True)
#: Overwrites state; repeating the call changes nothing more.
OVERWRITING = Hints(read_only=False, idempotent=True, destructive=True)
#: Deletes or replaces state or files, or runs arbitrary code.
DESTRUCTIVE = Hints(read_only=False, idempotent=False, destructive=True)

# Marks a Param field that keeps what the operation's signature declares.
_FROM_OPERATION: Any = object()


@dataclass(frozen=True)
class Param:
    """How a tool advertises one of its operation's parameters differently.

    Attributes:
        annotation: Type in the tool's input schema instead of the operation's.
        default: Default in the input schema instead of the operation's.
        default_from: Reads the default from the server settings when the tool
            is registered, instead of a fixed ``default``.
    """

    annotation: Any = _FROM_OPERATION
    default: Any = _FROM_OPERATION
    default_from: Optional[Callable[["Settings"], Any]] = None


@dataclass(frozen=True)
class BlenderTool:
    """One Blender MCP tool.

    Attributes:
        name: Registered tool name.
        description: Tool description shown in the listing.
        hints: MCP safety hints.
        response: Schema a successful payload is checked against.
        operation: The ``BlenderRuntimeSession`` operation the tool calls; its
            signature and docstring supply the tool's parameters. None for a
            tool that calls no operation, which then needs ``build``.
        request: Model the arguments are validated against inside the
            envelope; None for a tool with nothing to validate.
        params: Operation parameters the tool advertises differently, by name.
        descriptions: Parameter descriptions that replace the operation
            docstring's, by name.
        call: Runs instead of the operation: called with the session and the
            validated arguments by keyword, returns the payload.
        script: The tool runs agent-authored code, so it registers only while
            ``security.allow_script_execution`` is on.
        build: Writes the tool's wrapper, for a tool whose call is more than
            forwarding its validated arguments to ``operation``.
        added_params: Operation parameters that an add-on built before them
            does not take. Each is forwarded only when the call sets it away
            from its default, so an older add-on keeps answering the calls
            that leave it alone (with its own behavior) instead of refusing
            every call with an unexpected keyword argument.
    """

    name: str
    description: str
    hints: Hints
    response: Type[BaseModel]
    operation: Optional[str] = None
    request: Optional[Type[BaseModel]] = None
    params: Mapping[str, Param] = field(default_factory=dict)
    descriptions: Mapping[str, str] = field(default_factory=dict)
    call: Optional[SessionCall] = None
    script: bool = False
    build: Optional[Callable[["SimulMCPServer", "BlenderTool"], ToolWrapper]] = None
    added_params: Tuple[str, ...] = ()


def register_blender_tools(server: "SimulMCPServer", thin: bool = False) -> None:
    """Register the Blender tools, in ``BLENDER_TOOLS`` order.

    Args:
        server: The MCP server instance.
        thin: When True, register only ``THIN_BLENDER_TOOLS`` from
            ``simul.tool_surfaces``. Selected by ``blender.tool_surface`` /
            ``simul server --blender-tools``.
    """
    surface = THIN_BLENDER_TOOLS if thin else None
    tool = surface_tool(server.mcp.tool, surface)
    script_tool = surface_tool(server._script_tool, surface)
    for spec in BLENDER_TOOLS:
        register = script_tool if spec.script else tool
        register(
            name=spec.name,
            description=spec.description,
            annotations=server._tool_annotations(
                read_only=spec.hints.read_only,
                idempotent=spec.hints.idempotent,
                open_world=True,
                destructive=spec.hints.destructive,
            ),
            output_schema=None,
            task=server._task_optional(),
        )((spec.build or _forwarding_tool)(server, spec))


def session_operation(spec: BlenderTool) -> Callable[..., Dict[str, Any]]:
    """Return the ``BlenderRuntimeSession`` operation a tool calls.

    Args:
        spec: The tool's entry.

    Returns:
        The unbound session method.

    Raises:
        TypeError: When the entry names no public session operation.
    """
    operation = getattr(BlenderRuntimeSession, spec.operation or "", None)
    if (
        not spec.operation
        or spec.operation.startswith("_")
        or not inspect.isfunction(operation)
    ):
        raise TypeError(
            f"{spec.name}: {spec.operation!r} is not a BlenderRuntimeSession operation"
        )
    return operation


async def _exec(
    server: "SimulMCPServer",
    spec: BlenderTool,
    call: SessionCall,
    arguments: Optional[Dict[str, Any]] = None,
) -> ToolResult:
    """Run one call of the tool through the server's envelope.

    Args:
        server: The server whose envelope and Blender adapter are used.
        spec: The tool's entry, for its name and models.
        call: Receives the session, plus the validated request when the tool
            has a request model.
        arguments: The tool's arguments, validated against ``spec.request``.

    Returns:
        The tool result.
    """
    return await server._exec_backend(
        spec.name,
        server.blender_adapter,
        "Blender",
        spec.response,
        call,
        request_model=spec.request,
        request_kwargs=arguments if spec.request is not None else None,
    )


def _session_value(value: Any) -> Any:
    """What the session receives for a validated field: nested models as plain dicts."""
    if isinstance(value, BaseModel):
        return value.model_dump(exclude_none=True)
    return value


def _advertised_parameters(
    spec: BlenderTool, operation: Callable[..., Any], settings: "Settings"
) -> List[inspect.Parameter]:
    """The tool's parameters: the operation's, minus ``self``, with the entry's overrides."""
    hints = typing.get_type_hints(operation)
    parameters = []
    for name, parameter in list(inspect.signature(operation).parameters.items())[1:]:
        override = spec.params.get(name, Param())
        annotation = override.annotation
        if annotation is _FROM_OPERATION:
            annotation = hints.get(name, parameter.annotation)
        default = override.default
        if override.default_from is not None:
            default = override.default_from(settings)
        elif default is _FROM_OPERATION:
            default = parameter.default
        parameters.append(parameter.replace(annotation=annotation, default=default))
    return parameters


def _forwarding_tool(server: "SimulMCPServer", spec: BlenderTool) -> ToolWrapper:
    """Build the wrapper that validates a tool's arguments and forwards them to its operation.

    The wrapper advertises the operation's signature (with the entry's
    overrides) so FastMCP derives the input schema from it, and takes the
    parameter descriptions from the operation's docstring.

    Args:
        server: The server whose envelope and Blender adapter are used.
        spec: The tool's entry.

    Returns:
        An async callable named after the tool.
    """
    operation = session_operation(spec)
    parameters = _advertised_parameters(spec, operation, server.settings)
    advertised = inspect.Signature(parameters, return_annotation=ToolResult)
    method = operation.__name__
    added_defaults = {
        parameter.name: parameter.default
        for parameter in parameters
        if parameter.name in spec.added_params
    }

    def forward(session: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        arguments = {
            name: value
            for name, value in arguments.items()
            if name not in added_defaults or value != added_defaults[name]
        }
        if spec.call is not None:
            return spec.call(session, **arguments)
        payload: Dict[str, Any] = getattr(session, method)(**arguments)
        return payload

    async def tool(*args: Any, **kwargs: Any) -> ToolResult:
        bound = advertised.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        if spec.request is None:
            return await _exec(
                server, spec, lambda session: forward(session, arguments)
            )
        return await _exec(
            server,
            spec,
            lambda session, request: forward(
                session,
                {name: _session_value(getattr(request, name)) for name in arguments},
            ),
            arguments,
        )

    tool.__name__ = spec.name
    tool.__qualname__ = spec.name
    tool.__doc__ = operation.__doc__
    tool.__signature__ = advertised  # type: ignore[attr-defined]
    tool.__annotations__ = {
        parameter.name: parameter.annotation for parameter in parameters
    }
    tool.__annotations__["return"] = ToolResult
    return with_param_descriptions(operation, **spec.descriptions)(tool)


# ---------------------------------------------------------------------------
# Tools that write their own wrapper
# ---------------------------------------------------------------------------


def _control_ui_tool(server: "SimulMCPServer", spec: BlenderTool) -> ToolWrapper:
    """control_blender_ui: label the action with the caller's MCP session unless told otherwise.

    Written out because the agent label is resolved per request before
    validation, the action is advertised as the ``AgentControl`` enum where the
    operation takes a plain string, and the parameters are documented here:
    ``control_ui`` has no Args section of its own.
    """

    @with_param_descriptions()
    async def control_blender_ui(
        agent_control: AgentControl,
        target: Optional[str] = None,
        area_id: Optional[str] = None,
        position: Optional[List[float]] = None,
        value: Optional[List[float]] = None,
        agent_id: Optional[str] = None,
    ) -> ToolResult:
        """Control the selected Blender UI without generating a Python script.

        Args:
            agent_control: inspect, observe, move_cursor, clear_cursor, open_menu, select_object,
                set_tool, show_properties, set_property or isolate_workspace.
            target: Menu add/object/view, tool select_box/move/rotate/scale, object name,
                Properties tab such as OBJECT, or active-object property location/rotation_euler/scale.
            area_id: Editor ID from inspect; required for editor actions with multiple matches.
                Object selection/property writes do not require it. Re-inspect after workspace isolation.
            position: move_cursor only: agent overlay coordinates [x, y] in [0, 1], bottom-left origin; default center.
            value: set_property only: three finite values for the active object; rotation_euler is
                XYZ radians, written in the object's rotation mode (quaternion, axis-angle, other order).
            agent_id: Visible agent label, 1 to 64 printable characters; defaults to the MCP session identity.
                Use distinct stable labels for named agents across connections.
                clear_cursor removes only this agent's marker in the attached window; markers expire after 120 seconds.
        """
        return await _exec(
            server,
            spec,
            lambda session, request: session.control_ui(**request.model_dump()),
            {
                "agent_control": agent_control,
                "target": target,
                "area_id": area_id,
                "position": position,
                "value": value,
                "agent_id": server._resolve_agent_id(agent_id),
            },
        )

    return control_blender_ui


def _attach_window_tool(server: "SimulMCPServer", spec: BlenderTool) -> ToolWrapper:
    """attach_blender_window: select the attached target; no session operation is involved.

    Written out because it changes the attachment record through
    ``BlenderAttachments`` (refusing outside attached mode) instead of calling
    the Blender session.
    """

    @with_param_descriptions()
    async def attach_blender_window(
        instance_id: Optional[str] = None,
        window_id: Optional[str] = None,
    ) -> ToolResult:
        """Verify and persist the attached Blender target.

        Args:
            instance_id: Bridge instance ID; required when several Blender processes run.
            window_id: Window ID; required when the selected process has several windows.
        """
        return await _exec(
            server,
            spec,
            lambda _session, request: _attach(server, request),
            {"instance_id": instance_id, "window_id": window_id},
        )

    return attach_blender_window


def _attach(server: "SimulMCPServer", request: Any) -> Dict[str, Any]:
    """Attach to the requested Blender window, or explain why not."""
    if server.settings.blender.mode != "attached":
        return {
            "success": False,
            "error": "attach_blender_window requires --blender-mode attached (BLENDER__MODE=attached)",
            "error_type": "ValueError",
        }
    from ...adapters.blender_connection import BlenderAttachments

    attachments = BlenderAttachments(server.settings)
    try:
        return attachments.attach(request.instance_id, request.window_id)
    except (ValueError, RuntimeError) as exc:
        # List the candidates so the agent can choose without a shell.
        return {
            "success": False,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "details": {"instances": attachments.instances()},
        }


def _capture_viewport_tool(server: "SimulMCPServer", spec: BlenderTool) -> ToolWrapper:
    """capture_blender_viewport: one image, labelled with the caller's MCP session.

    Written out because the agent label is resolved per request before
    validation, the request model folds the deprecated ``use_render_fallback``
    into ``method`` so only ``method`` reaches the session, and the tool
    advertises ``method`` (as an enum) ahead of the deprecated flag, unlike the
    operation.
    """

    @with_param_descriptions(BlenderRuntimeSession.capture_viewport)
    async def capture_blender_viewport(
        width: int = 512,
        height: int = 512,
        jpeg_quality: int = 85,
        method: BlenderCaptureMethod = "auto",
        use_render_fallback: bool = False,
        agent_id: Optional[str] = None,
        camera_name: Optional[str] = None,
    ) -> ToolResult:
        return await _exec(
            server,
            spec,
            lambda session, request: session.capture_viewport(
                request.width,
                request.height,
                request.jpeg_quality,
                agent_id=request.agent_id,
                method=request.method,
                camera_name=request.camera_name,
            ),
            {
                "width": width,
                "height": height,
                "jpeg_quality": jpeg_quality,
                "method": method,
                "use_render_fallback": use_render_fallback,
                "agent_id": server._resolve_agent_id(agent_id),
                "camera_name": camera_name,
            },
        )

    return capture_blender_viewport


def _capture_sequence_tool(server: "SimulMCPServer", spec: BlenderTool) -> ToolWrapper:
    """capture_blender_viewport_sequence: a frame range, labelled like capture_blender_viewport.

    Written out for the same per-request agent label and ``method`` enum as
    capture_blender_viewport.
    """

    @with_param_descriptions(BlenderRuntimeSession.capture_viewport_sequence)
    async def capture_blender_viewport_sequence(
        start_frame: int,
        end_frame: int,
        step: int = 1,
        width: int = 512,
        height: int = 512,
        jpeg_quality: int = 85,
        agent_id: Optional[str] = None,
        method: BlenderCaptureMethod = "auto",
        camera_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        max_frames: int = SEQUENCE_FRAMES_DEFAULT,
    ) -> ToolResult:
        return await _exec(
            server,
            spec,
            lambda session, request: session.capture_viewport_sequence(
                request.start_frame,
                request.end_frame,
                request.step,
                request.width,
                request.height,
                request.jpeg_quality,
                agent_id=request.agent_id,
                method=request.method,
                camera_name=request.camera_name,
                output_dir=request.output_dir,
                max_frames=request.max_frames,
            ),
            {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "step": step,
                "width": width,
                "height": height,
                "jpeg_quality": jpeg_quality,
                "agent_id": server._resolve_agent_id(agent_id),
                "method": method,
                "camera_name": camera_name,
                "output_dir": output_dir,
                "max_frames": max_frames,
            },
        )

    return capture_blender_viewport_sequence


# ---------------------------------------------------------------------------
# The tools, in registration order
# ---------------------------------------------------------------------------

BLENDER_TOOLS: Tuple[BlenderTool, ...] = (
    BlenderTool(
        name="control_blender_ui",
        description=(
            "Perform a named agent_control action in the explicitly attached Blender window. "
            "Use inspect for editor IDs, active tool, Properties tabs, shared workspace and agent cursors. "
            "Actions display labelled agent pointers; move_cursor moves this overlay, never the OS cursor. "
            "Use observe after an external screenshot to show a brief eye badge and viewport pulse. "
            "Before set_tool on a shared workspace, use isolate_workspace then inspect for new editor IDs. "
            "Uses Blender UI APIs, not physical key/click events or arbitrary button lookup. "
            "Requires attached mode; does not launch Blender or execute arbitrary scripts."
        ),
        hints=ADDITIVE,
        operation="control_ui",
        request=BlenderUIRequest,
        response=BlenderUIResponse,
        build=_control_ui_tool,
    ),
    BlenderTool(
        name="get_blender_info",
        description="Get information about the active Blender runtime.",
        hints=READ_ONLY,
        operation="get_runtime_info",
        response=BlenderInfoResponse,
    ),
    BlenderTool(
        name="attach_blender_window",
        description=(
            "Attached mode: select the Blender process and window that Blender tools act on, "
            "the same as `simul blender attach`. Use it to recover after another file was loaded "
            "(read_homefile, open_mainfile) or Blender restarted. Succeeds only when the choice is "
            "unambiguous; pass instance_id/window_id from the error details or get_blender_info otherwise."
        ),
        hints=IDEMPOTENT,
        request=BlenderAttachWindowRequest,
        response=BlenderAttachWindowResponse,
        build=_attach_window_tool,
    ),
    BlenderTool(
        name="list_blender_scene_objects",
        description=(
            "List objects from the active Blender scene, a page at a time: count objects from "
            "offset, total matching, and next_offset for the next page (null after the last). "
            f"A page stops before its entries pass {LISTING_JSON_BYTES_MAX} bytes of JSON, so it "
            "can hold fewer than max_items. Each entry names a collection holding the object."
        ),
        hints=READ_ONLY,
        operation="list_scene_objects",
        request=BlenderSceneObjectsRequest,
        response=BlenderSceneObjectsResponse,
        params={
            "max_items": Param(
                default_from=lambda settings: settings.blender.max_scene_objects
            )
        },
        descriptions={
            "collection_name": "Optional collection name filter.",
            "include_hidden": "Include hidden objects when true.",
            "max_items": "Maximum number of objects to return.",
        },
        added_params=("offset",),
    ),
    # -- Core observation ----------------------------------------------------
    BlenderTool(
        name="get_blender_object_info",
        description=(
            "Get detailed information about a single Blender object. location, rotation_euler "
            "(XYZ radians) and scale are its own transform, relative to its parent as Blender's "
            "Location, Rotation and Scale fields show them; world_location, "
            "world_rotation_euler, world_rotation_quaternion, world_scale and matrix_world give "
            "where it is in the scene. rotation_mode says how Blender stores the rotation "
            "(rotation_in_mode); rotation_quaternion gives it as a quaternion."
        ),
        hints=READ_ONLY,
        operation="get_object_info",
        request=BlenderObjectInfoRequest,
        response=BlenderObjectInfoResponse,
    ),
    BlenderTool(
        name="get_blender_mesh_info",
        description=(
            "Get a Blender mesh object's vertex, edge, face and triangle counts: of the "
            "stored mesh, and evaluated with the modifiers enabled in the viewport, which is "
            "what an export writes. A mesh in Edit Mode is counted with its unsaved edits. "
            "warnings name modifiers set differently for render, which USD export evaluates."
        ),
        hints=READ_ONLY,
        operation="get_mesh_info",
        request=BlenderMeshInfoRequest,
        response=BlenderMeshInfoResponse,
    ),
    BlenderTool(
        name="get_blender_bounding_box",
        description="Get the bounding box of a Blender object.",
        hints=READ_ONLY,
        operation="get_bounding_box",
        request=BlenderBoundingBoxRequest,
        response=BlenderBoundingBoxResponse,
    ),
    BlenderTool(
        name="search_blender_objects",
        description=(
            "Search for objects in the Blender scene by name pattern or type (in any case), a "
            "page at a time as list_blender_scene_objects pages: count, total and next_offset."
        ),
        hints=READ_ONLY,
        operation="search_objects",
        request=BlenderSearchObjectsRequest,
        response=BlenderSearchObjectsResponse,
        added_params=("offset",),
    ),
    BlenderTool(
        name="summarize_blender_scene",
        description=(
            "Get a high-level summary of the Blender scene: object counts by type, its "
            "top-level collections, active camera and frame range."
        ),
        hints=READ_ONLY,
        operation="summarize_scene",
        response=BlenderSceneSummaryResponse,
    ),
    BlenderTool(
        name="get_blender_material_info",
        description="Get material information with bounded node tree traversal.",
        hints=READ_ONLY,
        operation="get_material_info",
        request=BlenderMaterialInfoRequest,
        response=BlenderMaterialInfoResponse,
    ),
    BlenderTool(
        name="get_blender_distance_between",
        description="Measure the Euclidean distance between two Blender objects.",
        hints=READ_ONLY,
        operation="get_distance_between",
        request=BlenderDistanceRequest,
        response=BlenderDistanceResponse,
    ),
    BlenderTool(
        name="check_blender_object_bounds",
        description=(
            "Check where a Blender object's world-space bounding box lies relative to an "
            "axis-aligned box: relation 'inside' (within_bounds), 'intersecting' or 'outside'. "
            "The bounding box turns and scales with the object; origin_within_bounds tests the "
            "origin alone. bounds_min must not exceed bounds_max."
        ),
        hints=READ_ONLY,
        operation="is_object_within_bounds",
        request=BlenderBoundsCheckRequest,
        response=BlenderBoundsCheckResponse,
        descriptions={
            "object_name": "Name of the Blender object to test.",
            "bounds_min": "Minimum corner of the box as [x, y, z] in scene units (Blender is Z-up).",
            "bounds_max": "Maximum corner of the box as [x, y, z] in scene units.",
        },
    ),
    # -- Visual observation --------------------------------------------------
    BlenderTool(
        name="capture_blender_viewport",
        description=(
            "Capture Blender as a JPEG. In attached mode the default (method='auto') is the 3D "
            "View the user sees: view angle, shading, overlays, Edit Mode and unsaved edits, "
            "with no camera needed, keeping the view's aspect ratio within width x height "
            "(capture_method viewport; an external engine's Rendered view, such as Cycles, is "
            "read from the screen as viewport_screen and includes overlapping UI). Embedded "
            "(headless bpy) mode has no 3D View, so 'auto' renders. method='render' renders "
            "through the scene camera, or through a temporary camera removed afterwards: "
            "aligned to the 3D View when attached, framing the visible objects when embedded "
            "or when the window shows no 3D View "
            "(render_temp_camera_view / render_temp_camera_framing). camera_name captures "
            "through that camera without changing the scene camera or the view: a render "
            "renders through it, a viewport capture draws the 3D View's shading from it "
            "(viewport_camera). The image arrives as an MCP image content block, followed by a "
            "JSON block with success, width, height, engine, capture_method, camera_name, format "
            "and image_attached=true. In attached mode, briefly show an agent eye badge and "
            "border pulse after success. Agent annotations are excluded from the captured image."
        ),
        hints=READ_ONLY,
        operation="capture_viewport",
        request=BlenderCaptureViewportRequest,
        response=BlenderCaptureViewportResponse,
        build=_capture_viewport_tool,
    ),
    BlenderTool(
        name="set_blender_camera_view",
        description=(
            "Place a Blender camera (the active one by default) at a world-space location "
            "and XYZ Euler rotation, whatever its parent and rotation mode. Success means "
            "the placement is in the camera's transform; its constraints (Track To, Copy "
            "Location, Child Of, ...) still apply and are never muted. location and "
            "rotation_euler report where Blender evaluates the camera; placement_overridden "
            "is true when that is not the request, with constraints_applied and a warning "
            "on how to disable them. Keys on the camera's transform take over at the next "
            "frame change. Blender's viewport skips a camera disabled in viewports unless it "
            "is needed by something visible, or is the scene camera or a timeline marker's "
            "camera and its hide_viewport is not animated: when Blender did not evaluate it "
            "after the placement (or its evaluated pose did not change), override_check is "
            "'skipped' and placement_overridden null."
        ),
        hints=IDEMPOTENT,
        operation="set_camera_view",
        request=BlenderSetCameraViewRequest,
        response=BlenderSetCameraViewResponse,
    ),
    BlenderTool(
        name="get_blender_camera_info",
        description=(
            "Get a Blender camera's world-space location and XYZ Euler rotation where "
            "Blender evaluates it (constraints included), the constraints that act on it "
            "(constraints_applied), lens and clip range (the active camera by default). "
            "Blender keeps a stale pose for a camera disabled in viewports unless it is "
            "needed by something visible through a child, a constraint or a modifier, "
            "or is the scene camera or a timeline marker's camera and its hide_viewport "
            "is not animated: its pose is then composed from its own transform "
            "(pose_source 'transform'), without its constraints, drivers and keys, and "
            "warning says what that leaves out."
        ),
        hints=READ_ONLY,
        operation="get_camera_info",
        response=BlenderCameraInfoResponse,
    ),
    BlenderTool(
        name="focus_blender_on_object",
        description=(
            "Frame a Blender object with a camera (the active one by default): aim it from "
            "the front and above at the object's bounding-box center, at the distance where "
            "the object fits the frame (an orthographic camera gets the ortho_scale that fits "
            "it), widening the clip range when it would cut the object away. As with "
            "set_blender_camera_view, success means the framing is in the camera's "
            "transform and its constraints still apply: camera_location and "
            "camera_rotation_euler report where Blender evaluates the camera, and "
            "placement_overridden is true when constraints or drivers moved it off the framing. "
            "For a camera disabled in viewports that Blender did not evaluate after the "
            "placement (or whose evaluated pose did not change), override_check is 'skipped' "
            "and placement_overridden null."
        ),
        hints=IDEMPOTENT,
        operation="focus_on_object",
        request=BlenderFocusOnObjectRequest,
        response=BlenderFocusOnObjectResponse,
    ),
    BlenderTool(
        name="get_blender_viewport_info",
        description=(
            "Get the render settings (engine, resolution, film, active camera) and, in attached "
            "mode, view3d: the 3D View captures use, with its view_matrix, eye_location, "
            "perspective, lens, clip range and shading. view3d is null without an editor window "
            "(embedded bpy, blender -b) or 3D View, and view3d_unavailable_reason says which."
        ),
        hints=READ_ONLY,
        operation="get_viewport_info",
        response=BlenderViewportInfoResponse,
    ),
    BlenderTool(
        name="capture_blender_viewport_sequence",
        description=(
            "Capture a frame range as JPEG files, each made like capture_blender_viewport "
            "with the same method and camera_name (the 3D View when attached, a camera render "
            "in embedded mode), and return their paths: frames [{frame, file_path}], "
            "output_dir (default a new folder in the sandbox's capture directory), width and "
            "height. end_frame must not be before start_frame. One call captures at most "
            f"max_frames (default {SEQUENCE_FRAMES_DEFAULT}, at most {SEQUENCE_FRAMES_MAX}) and "
            f"starts no frame after {SEQUENCE_TIME_BUDGET_SECONDS:g} s, since Blender is busy "
            "meanwhile; then complete is false and next_start_frame says where to continue. "
            "In attached mode, briefly show an agent eye badge and border pulse after success. "
            "Agent annotations are excluded from the captured images."
        ),
        hints=READ_ONLY,
        operation="capture_viewport_sequence",
        request=BlenderCaptureSequenceRequest,
        response=BlenderCaptureSequenceResponse,
        build=_capture_sequence_tool,
    ),
    # -- Scene manipulation --------------------------------------------------
    BlenderTool(
        name="create_blender_object",
        description=(
            "Create a new object in the Blender scene's active collection, or in the scene's own "
            "collection when the active one is excluded, hidden or disabled in viewports (warning "
            "says so). It stays in Object Mode whatever Blender's Enter Edit Mode preference."
        ),
        hints=ADDITIVE,
        operation="create_object",
        request=BlenderCreateObjectRequest,
        response=BlenderCreateObjectResponse,
        params={
            "location": Param(List[float], [0.0, 0.0, 0.0]),
            "rotation_euler": Param(List[float], [0.0, 0.0, 0.0]),
            "scale": Param(List[float], [1.0, 1.0, 1.0]),
        },
    ),
    BlenderTool(
        name="delete_blender_object",
        description=(
            "Delete an object from the current Blender scene with Blender's own Delete, which "
            "also resets a rigid body's cache. Another scene linking it keeps it "
            "(remaining_scenes); use_global=true deletes it from every scene and the file."
        ),
        hints=DESTRUCTIVE,
        operation="delete_object",
        request=BlenderDeleteObjectRequest,
        response=BlenderDeleteObjectResponse,
        added_params=("use_global",),
    ),
    BlenderTool(
        name="set_blender_object_mode",
        description=(
            "Switch an object (the active one by default) into a mode, as the 3D View's mode menu "
            "does: OBJECT by default, the way back for a window left in Edit or Pose Mode, where "
            "create, delete, import and the other mode-sensitive tools refuse to run."
        ),
        hints=IDEMPOTENT,
        operation="set_object_mode",
        request=BlenderSetObjectModeRequest,
        response=BlenderSetObjectModeResponse,
    ),
    BlenderTool(
        name="set_blender_object_transform",
        description=(
            "Set location, rotation, and/or scale of a Blender object's own transform, relative "
            "to its parent (Blender's Location, Rotation and Scale fields). rotation_euler is XYZ "
            "Euler radians, or pass rotation_quaternion [w, x, y, z] instead; either is converted "
            "to the object's rotation_mode (glTF imports store quaternions), which is kept. The "
            "reply reports the result, rotation_mode and world_location."
        ),
        hints=OVERWRITING,
        operation="set_object_transform",
        request=BlenderSetTransformRequest,
        response=BlenderSetTransformResponse,
        added_params=("rotation_quaternion",),
    ),
    BlenderTool(
        name="set_blender_object_parent",
        description=(
            "Parent one Blender object to another. By default the child keeps its world "
            "transform, as Blender's Parent > Object (Keep Transform); keep_transform=false "
            "makes its own values relative to the parent, so it moves. A parent that is the "
            "child itself or one of its descendants is refused."
        ),
        hints=OVERWRITING,
        operation="set_object_parent",
        request=BlenderSetParentRequest,
        response=BlenderSetParentResponse,
        added_params=("keep_transform",),
    ),
    BlenderTool(
        name="clear_blender_object_parent",
        description="Remove parent from a Blender object.",
        hints=OVERWRITING,
        operation="clear_object_parent",
        request=BlenderClearParentRequest,
        response=BlenderClearParentResponse,
    ),
    BlenderTool(
        name="assign_blender_material",
        description="Assign a Principled BSDF material to a Blender object.",
        hints=OVERWRITING,
        operation="assign_material",
        request=BlenderAssignMaterialRequest,
        response=BlenderAssignMaterialResponse,
        params={"base_color": Param(List[float], [0.8, 0.8, 0.8, 1.0])},
    ),
    BlenderTool(
        name="add_blender_modifier",
        description=(
            "Add a modifier to a Blender mesh, curve or other object with a modifier stack. "
            "params are the modifier's settings by Blender RNA name, checked before use: an "
            "unknown name is refused with the closest names, an enum value must be one of its "
            "items, a number must lie within the setting's limits (Blender would clamp it "
            "silently), and an object or collection setting takes its name, e.g. "
            "{'object': 'Cutter'} for BOOLEAN. The reply echoes the stored settings. A refused "
            "call leaves no modifier behind. DECIMATE is refused on a mesh with invalid geometry, "
            "which can freeze Blender; run clean_blender_mesh first."
        ),
        hints=ADDITIVE,
        operation="add_modifier",
        request=BlenderAddModifierRequest,
        response=BlenderAddModifierResponse,
        params={"params": Param(Dict[str, Any], {})},
    ),
    BlenderTool(
        name="apply_blender_modifier",
        description=(
            "Apply one modifier of a Blender mesh object, baking its result into the mesh data "
            "(for example a Decimate before export) and removing it from the stack. A modifier "
            "below others is applied to the base mesh without them, with a warning; shared mesh "
            "data is copied for this object first. A modifier disabled in the viewport is "
            "refused, as Blender would drop it unapplied. Requires Object Mode."
        ),
        hints=DESTRUCTIVE,
        operation="apply_modifier",
        request=BlenderModifierTargetRequest,
        response=BlenderApplyModifierResponse,
    ),
    BlenderTool(
        name="remove_blender_modifier",
        description="Remove one modifier from a Blender object's stack without applying it.",
        hints=DESTRUCTIVE,
        operation="remove_modifier",
        request=BlenderModifierTargetRequest,
        response=BlenderRemoveModifierResponse,
    ),
    BlenderTool(
        name="set_blender_light_params",
        description="Set light parameters on a Blender light object.",
        hints=OVERWRITING,
        operation="set_light_params",
        request=BlenderSetLightParamsRequest,
        response=BlenderSetLightParamsResponse,
    ),
    # -- File I/O ------------------------------------------------------------
    BlenderTool(
        name="open_blender_file",
        description=(
            "Open a .blend file, replacing the current scene. Refuses with "
            "UnsavedChangesError while the open file has unsaved changes (in attached mode "
            "every agent edit counts, as each is an undo step); save them first with "
            "save_blender_file, or pass discard_unsaved=true to discard them."
        ),
        hints=DESTRUCTIVE,
        operation="open_blend_file",
        request=BlenderOpenFileRequest,
        response=BlenderOpenFileResponse,
    ),
    BlenderTool(
        name="save_blender_file",
        description=(
            "Save the current .blend file, in place or under a new .blend path, which then "
            "becomes the open document. Other extensions are refused; use "
            "export_blender_file for other formats."
        ),
        hints=OVERWRITING,
        operation="save_blend_file",
        request=BlenderSaveFileRequest,
        response=BlenderSaveFileResponse,
    ),
    BlenderTool(
        name="import_blender_file",
        description=(
            "Import an OBJ, FBX, glTF, USD, STL or PLY file into the Blender scene, "
            "optionally scaled about the world origin (scale). The textures, buffers, MTL "
            "libraries and USD layers it would read must be inside the sandbox too, or it is "
            "refused with SandboxError; within the sandbox the FBX importer does not search "
            "folders for textures."
        ),
        hints=ADDITIVE,
        operation="import_file",
        request=BlenderImportFileRequest,
        response=BlenderImportFileResponse,
    ),
    BlenderTool(
        name="export_blender_file",
        description=(
            "Export the scene, the selection (selected_only), a list of objects "
            "(object_names) or a collection (collection_name) to OBJ, FBX, glTF, USD, STL "
            "or PLY. file_path must end in the format's extension and is written exactly: "
            "for GLTF, .glb writes a GLB and .gltf a .gltf with its .bin and textures "
            "(gltf_format to choose GLTF_EMBEDDED). Modifiers are applied and animation "
            "included by default (apply_modifiers, export_animation, frame_start/frame_end; "
            "USD animation is on here although Blender's exporter leaves it off). FBX, OBJ, "
            "STL and PLY take axis_forward/axis_up and global_scale; USD takes usd_up_axis, "
            "meters_per_unit and root_prim_path. Missing folders are created, Edit Mode is "
            "entered again afterwards, an unrelated existing .bin/.mtl beside the file is "
            "not replaced unless overwrite_companion_files, and files_written lists every "
            "file written."
        ),
        hints=DESTRUCTIVE,
        operation="export_file",
        request=BlenderExportFileRequest,
        response=BlenderExportFileResponse,
    ),
    BlenderTool(
        name="get_blender_file_info",
        description=(
            "Get information about the current .blend file: its path, saved and dirty state, the "
            "current scene's name and object_count, and file_object_count for the whole file."
        ),
        hints=READ_ONLY,
        operation="get_file_info",
        response=BlenderFileInfoResponse,
    ),
    # -- Animation and timeline ----------------------------------------------
    BlenderTool(
        name="set_blender_frame",
        description="Set the current animation frame.",
        hints=IDEMPOTENT,
        operation="set_frame",
        request=BlenderSetFrameRequest,
        response=BlenderSetFrameResponse,
    ),
    BlenderTool(
        name="get_blender_frame",
        description="Get the current frame and animation range.",
        hints=READ_ONLY,
        operation="get_frame",
        response=BlenderGetFrameResponse,
    ),
    BlenderTool(
        name="set_blender_frame_range",
        description=(
            "Set the animation frame range. An unbaked rigid-body cache follows it; "
            "a baked one keeps its range, with a warning."
        ),
        hints=OVERWRITING,
        operation="set_frame_range",
        request=BlenderSetFrameRangeRequest,
        response=BlenderSetFrameRangeResponse,
    ),
    BlenderTool(
        name="play_blender_animation",
        description="Control animation playback (play, stop, reverse).",
        hints=ADDITIVE,
        operation="play_animation",
        request=BlenderPlayAnimationRequest,
        response=BlenderPlayAnimationResponse,
    ),
    BlenderTool(
        name="insert_blender_keyframe",
        description="Insert a keyframe on an object property.",
        hints=ADDITIVE,
        operation="insert_keyframe",
        request=BlenderInsertKeyframeRequest,
        response=BlenderInsertKeyframeResponse,
    ),
    BlenderTool(
        name="delete_blender_keyframe",
        description="Delete a keyframe from an object property.",
        hints=DESTRUCTIVE,
        operation="delete_keyframe",
        request=BlenderDeleteKeyframeRequest,
        response=BlenderDeleteKeyframeResponse,
    ),
    BlenderTool(
        name="get_blender_keyframes",
        description="Get keyframe summary for an object.",
        hints=READ_ONLY,
        operation="get_keyframes",
        request=BlenderGetKeyframesRequest,
        response=BlenderGetKeyframesResponse,
    ),
    # -- Physics and simulation ----------------------------------------------
    BlenderTool(
        name="setup_blender_rigid_body",
        description="Set up rigid body physics on a Blender object.",
        hints=OVERWRITING,
        operation="setup_rigid_body",
        request=BlenderSetupRigidBodyRequest,
        response=BlenderSetupRigidBodyResponse,
    ),
    BlenderTool(
        name="add_blender_force_field",
        description="Add a force field to the scene.",
        hints=ADDITIVE,
        operation="add_force_field",
        request=BlenderAddForceFieldRequest,
        response=BlenderAddForceFieldResponse,
    ),
    BlenderTool(
        name="get_blender_force_field_info",
        description="Get force field parameters for an object.",
        hints=READ_ONLY,
        operation="get_force_field_info",
        request=BlenderGetForceFieldInfoRequest,
        response=BlenderGetForceFieldInfoResponse,
    ),
    BlenderTool(
        name="add_blender_rigid_body_constraint",
        description="Add a rigid body constraint between two objects.",
        hints=ADDITIVE,
        operation="add_rigid_body_constraint",
        request=BlenderAddConstraintRequest,
        response=BlenderAddConstraintResponse,
    ),
    BlenderTool(
        name="get_blender_constraint_info",
        description="Get rigid body constraint info for an object.",
        hints=READ_ONLY,
        operation="get_constraint_info",
        request=BlenderGetConstraintInfoRequest,
        response=BlenderGetConstraintInfoResponse,
    ),
    BlenderTool(
        name="get_blender_physics_state",
        description=(
            "Get an object's world pose and rigid-body state at the current frame. An "
            "unbaked simulation advances one frame at a time, so after a frame jump a "
            "body sits where it was last simulated: by default the scene first steps "
            "from the cache start to the current frame, as playback would (up to "
            "400 000 body-frames). pose_source and simulated_for_frame say whether the "
            "pose is the simulation's for this frame; frames past the cache end hold "
            "the bodies at its last frame (past_cache_end). rigid_body_cache reports "
            "the cache range and bake state."
        ),
        hints=READ_ONLY,
        operation="get_physics_state",
        request=BlenderGetPhysicsStateRequest,
        response=BlenderGetPhysicsStateResponse,
    ),
    BlenderTool(
        name="get_blender_object_trajectory",
        description="Sample object position over a frame range.",
        hints=READ_ONLY,
        operation="get_object_trajectory",
        request=BlenderGetTrajectoryRequest,
        response=BlenderGetTrajectoryResponse,
    ),
    BlenderTool(
        name="bake_blender_simulation",
        description=(
            "Bake the physics simulation for frame_start-frame_end (start before end), "
            "replacing an earlier bake: the simulation restarts from frame_start with "
            "the bodies' current transforms. scope 'rigid_body' (default) bakes only "
            "the rigid-body world's cache; 'all' bakes every point cache (rigid "
            "bodies, cloth, soft bodies, dynamic paint, particles). Refused for a "
            "disabled rigid body world. The reply gives the range the cache holds "
            "and warns about scene frames it leaves out, where bodies hold their "
            "pose. Runs on Blender's main thread: a long bake can outlast the "
            "client timeout."
        ),
        hints=OVERWRITING,
        operation="bake_simulation",
        request=BlenderBakeSimulationRequest,
        response=BlenderBakeSimulationResponse,
        params={"scope": Param(BlenderBakeScope)},
    ),
    BlenderTool(
        name="free_blender_bake",
        description=(
            "Free baked physics simulation data over the same scope as "
            "bake_blender_simulation ('rigid_body' by default, or 'all'). A freed "
            "cache gets the scene's frame range back, so the simulation is not cut "
            "off at the last bake's end frame. freed lists the caches that were baked."
        ),
        hints=OVERWRITING,
        operation="free_bake",
        request=BlenderFreeBakeRequest,
        response=BlenderFreeBakeResponse,
        params={"scope": Param(BlenderBakeScope)},
    ),
    BlenderTool(
        name="get_blender_rigid_body_world",
        description=(
            "Get the scene's rigid body world: enabled, scene gravity, substeps, "
            "solver iterations, time scale, split impulse, body and constraint "
            "counts, and the cache range and bake state (rigid_body_cache). Bodies "
            "hold their pose past the cache end; warnings name a disabled world or a "
            "cache that leaves out part of the scene range."
        ),
        hints=READ_ONLY,
        operation="get_rigid_body_world",
        response=BlenderRigidBodyWorldResponse,
    ),
    BlenderTool(
        name="set_blender_rigid_body_world",
        description=(
            "Change the scene's rigid body world, adding one when there is none; only "
            "the given settings change. gravity is the scene's, shared with cloth, "
            "soft bodies and particles. A baked cache keeps playing its bake (changed "
            "settings apply after free_blender_bake and a new bake), and its range "
            "cannot change until freed. Replies like get_blender_rigid_body_world."
        ),
        hints=OVERWRITING,
        operation="set_rigid_body_world",
        request=BlenderSetRigidBodyWorldRequest,
        response=BlenderRigidBodyWorldResponse,
    ),
    # -- Scripting and mesh from data ----------------------------------------
    BlenderTool(
        name="execute_blender_script",
        description=(
            "Execute arbitrary Python code inside Blender with access "
            "to bpy. Assign to __result__ to return a value. Printed output "
            "comes back in output, cut to its first max_output_chars characters "
            f"(default {SCRIPT_OUTPUT_CHARS_DEFAULT}, at most {SCRIPT_OUTPUT_CHARS_MAX}), "
            f"or earlier when they exceed {SCRIPT_OUTPUT_JSON_BYTES_MAX} bytes of JSON "
            "(heavily non-ASCII output); "
            "a cut sets output_truncated=true and output_total_chars to the full length. "
            "Write larger results to a file and return its path instead. Anything the script "
            "raises, including SystemExit from sys.exit(), is reported in error and never "
            "ends Blender. In attached mode each call is one undo step."
        ),
        hints=DESTRUCTIVE,
        operation="execute_script",
        request=BlenderExecuteScriptRequest,
        response=BlenderExecuteScriptResponse,
        script=True,
    ),
    BlenderTool(
        name="create_blender_mesh_from_data",
        description=(
            "Create a mesh object from raw vertex, edge, and face data. Use for procedural "
            "geometry. Each vertex is exactly [x, y, z] finite numbers; a face repeating a "
            "vertex or another face's vertex set, an edge joining a vertex to itself or "
            "repeating another, and a collection outside the current scene are refused before "
            "anything is created. The mesh is validated, and a failed call leaves no mesh or "
            "object behind."
        ),
        hints=ADDITIVE,
        operation="create_mesh_from_data",
        request=BlenderCreateMeshFromDataRequest,
        response=BlenderCreateMeshFromDataResponse,
        params={
            "edges": Param(List[List[int]], []),
            "faces": Param(List[List[int]], []),
        },
    ),
    BlenderTool(
        name="clean_blender_mesh",
        description=(
            "Repair a Blender mesh object's data. Always validates it, removing faces that "
            "repeat a vertex and duplicate faces or edges (geometry that can freeze Decimate); "
            "then optionally merges vertices within merge_distance, dissolves edges and faces "
            "smaller than dissolve_degenerate_distance, and recalculates normals to point "
            "outside. Reports the counts before and after. Shared mesh data changes for every "
            "object using it. Requires Object Mode."
        ),
        hints=OVERWRITING,
        operation="clean_mesh",
        request=BlenderCleanMeshRequest,
        response=BlenderCleanMeshResponse,
    ),
    # -- SimReady asset format -----------------------------------------------
    BlenderTool(
        name="apply_simready_metadata",
        description=(
            "Store NVIDIA SimReady metadata (semantic labels, physics properties, material info) "
            "on a Blender object as simready_ custom properties, which export_simready_usd writes "
            "as UsdPhysics and UsdSemantics schemas. Omitted fields keep their stored value; "
            "clear removes stored values. An additional label named after a field's taxonomy "
            "(class, hierarchy, wikidata_qcode) sets that field. Unknown fields, invalid values, "
            "Edit Mode and linked objects are refused, before anything changes."
        ),
        hints=OVERWRITING,
        operation="apply_simready_metadata",
        request=SimReadyApplyMetadataRequest,
        response=SimReadyApplyMetadataResponse,
        params={
            "metadata": Param(Optional[SimReadyMetadata]),
            "clear": Param(Optional[List[SimReadyClearName]]),
        },
    ),
    BlenderTool(
        name="get_simready_metadata",
        description=(
            "Read NVIDIA SimReady metadata from a Blender object: semantic labels, physics "
            "properties and material info stored as simready_ custom properties. Stored values "
            "that break the rules are listed in invalid_values instead of failing the call."
        ),
        hints=READ_ONLY,
        operation="get_simready_metadata",
        request=SimReadyGetMetadataRequest,
        response=SimReadyGetMetadataResponse,
    ),
    BlenderTool(
        name="validate_simready_compliance",
        description=(
            "Validate the Blender objects export_simready_usd would export against the SimReady "
            "Robotics-Prop rules and Isaac Sim's asset checks; any error makes them non-compliant. Errors: an "
            "asset root that is not the identity, a skewed rigid body, no material, no semantic "
            "label on a mesh or an exported object above it, invalid stored values, a rigid body "
            "without mass, without a collider or nested in another, degenerate faces. Warnings: "
            "names, sizes and units, scaled parts, an empty slot faces use, missing colliders, "
            "UVs, n-gons. Filter by severity and page with offset/limit (next_offset)."
        ),
        hints=READ_ONLY,
        operation="validate_simready_compliance",
        request=SimReadyValidateRequest,
        response=SimReadyValidateResponse,
        params={"severity": Param(SimReadySeverity)},
    ),
    BlenderTool(
        name="export_simready_usd",
        description=(
            "Export Blender objects as a SimReady USD asset a simulator reads: from the simready_ "
            "metadata it authors UsdPhysics rigid bodies, mass, mesh colliders and physics materials, "
            "and UsdSemantics labels. The asset root is the default prim and holds its materials; "
            "metersPerUnit follows the scene's Unit Scale. Listed objects export with their "
            "descendants, hidden ones too; lights, cameras and the World are left out by default. "
            "Refuses when validation finds errors; skip_checks leaves a check out, force=true "
            "ignores every error. Returns each object's prim path."
        ),
        hints=DESTRUCTIVE,
        operation="export_simready_usd",
        request=SimReadyExportRequest,
        response=SimReadyExportResponse,
        params={"skip_checks": Param(Optional[List[SimReadyCheckName]])},
    ),
    BlenderTool(
        name="setup_simready_hierarchy",
        description=(
            "Put Blender objects under a SimReady root empty (an Xform in the USD), keeping where "
            "they are, also for objects in excluded collections, skewed by a parent or under "
            "constraints. Only listed objects without a listed ancestor are re-parented; existing "
            "nesting stays. Needs Object Mode; refuses linked objects. Checks everything first "
            "and changes nothing on failure; an existing root name is refused unless "
            "reuse_root=true. Optionally stores semantic labels on the root."
        ),
        hints=ADDITIVE,
        operation="setup_simready_hierarchy",
        request=SimReadySetupHierarchyRequest,
        response=SimReadySetupHierarchyResponse,
        params={"semantic": Param(Optional[SimReadySemanticLabels])},
    ),
)
