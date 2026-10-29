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

from fastmcp.tools.tool import ToolResult
from pydantic import BaseModel

from ...adapters.blender_runtime import BlenderRuntimeSession
from ...tool_surfaces import THIN_BLENDER_TOOLS
from ..schemas.blender import (
    SCRIPT_OUTPUT_CHARS_DEFAULT,
    SCRIPT_OUTPUT_CHARS_MAX,
    SCRIPT_OUTPUT_JSON_BYTES_MAX,
    BlenderAddConstraintRequest,
    BlenderAddConstraintResponse,
    BlenderAddForceFieldRequest,
    BlenderAddForceFieldResponse,
    BlenderAddModifierRequest,
    BlenderAddModifierResponse,
    BlenderAssignMaterialRequest,
    BlenderAssignMaterialResponse,
    BlenderAttachWindowRequest,
    BlenderAttachWindowResponse,
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
    BlenderObjectInfoRequest,
    BlenderObjectInfoResponse,
    BlenderOpenFileRequest,
    BlenderOpenFileResponse,
    BlenderPlayAnimationRequest,
    BlenderPlayAnimationResponse,
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
    BlenderSetParentRequest,
    BlenderSetParentResponse,
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
    SimReadyExportRequest,
    SimReadyExportResponse,
    SimReadyGetMetadataRequest,
    SimReadyGetMetadataResponse,
    SimReadySetupHierarchyRequest,
    SimReadySetupHierarchyResponse,
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

    def forward(session: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
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
            value: set_property only: three finite values for the active object; rotations in radians.
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
            ),
            {
                "width": width,
                "height": height,
                "jpeg_quality": jpeg_quality,
                "method": method,
                "use_render_fallback": use_render_fallback,
                "agent_id": server._resolve_agent_id(agent_id),
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
            },
        )

    return capture_blender_viewport_sequence


def _free_bake(session: Any) -> Dict[str, Any]:
    """Free the bake; success is the absence of an exception, whatever the session returns."""
    session.free_bake()
    return {}


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
        description="List objects from the active Blender scene.",
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
    ),
    # -- Core observation ----------------------------------------------------
    BlenderTool(
        name="get_blender_object_info",
        description="Get detailed information about a single Blender object.",
        hints=READ_ONLY,
        operation="get_object_info",
        request=BlenderObjectInfoRequest,
        response=BlenderObjectInfoResponse,
    ),
    BlenderTool(
        name="get_blender_mesh_info",
        description="Get mesh geometry counts for a Blender mesh object.",
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
        description="Search for objects in the Blender scene by name pattern or type.",
        hints=READ_ONLY,
        operation="search_objects",
        request=BlenderSearchObjectsRequest,
        response=BlenderSearchObjectsResponse,
    ),
    BlenderTool(
        name="summarize_blender_scene",
        description="Get a high-level summary of the Blender scene.",
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
        description="Check if a Blender object is within spatial bounds.",
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
            "(render_temp_camera_view / render_temp_camera_framing). The image arrives as an MCP "
            "image content block, followed by a JSON block with success, width, height, engine, "
            "capture_method, format and image_attached=true. In attached mode, briefly show an "
            "agent eye badge and border pulse after success. Agent annotations are excluded "
            "from the captured image."
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
            "and XYZ Euler rotation, whatever its parent and rotation mode."
        ),
        hints=IDEMPOTENT,
        operation="set_camera_view",
        request=BlenderSetCameraViewRequest,
        response=BlenderSetCameraViewResponse,
    ),
    BlenderTool(
        name="get_blender_camera_info",
        description=(
            "Get a Blender camera's world-space location and XYZ Euler rotation, lens and "
            "clip range (the active camera by default)."
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
            "it), widening the clip range when it would cut the object away."
        ),
        hints=IDEMPOTENT,
        operation="focus_on_object",
        request=BlenderFocusOnObjectRequest,
        response=BlenderFocusOnObjectResponse,
    ),
    BlenderTool(
        name="get_blender_viewport_info",
        description="Get active viewport and render settings.",
        hints=READ_ONLY,
        operation="get_viewport_info",
        response=BlenderViewportInfoResponse,
    ),
    BlenderTool(
        name="capture_blender_viewport_sequence",
        description=(
            "Capture a frame range as base64-encoded JPEGs, each made like "
            "capture_blender_viewport with the same method (the 3D View when attached, a "
            "camera render in embedded mode). "
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
        description="Create a new object in the Blender scene.",
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
        description="Delete an object from the Blender scene.",
        hints=DESTRUCTIVE,
        operation="delete_object",
        request=BlenderDeleteObjectRequest,
        response=BlenderDeleteObjectResponse,
    ),
    BlenderTool(
        name="set_blender_object_transform",
        description="Set location, rotation, and/or scale on a Blender object.",
        hints=OVERWRITING,
        operation="set_object_transform",
        request=BlenderSetTransformRequest,
        response=BlenderSetTransformResponse,
    ),
    BlenderTool(
        name="set_blender_object_parent",
        description="Parent one Blender object to another.",
        hints=OVERWRITING,
        operation="set_object_parent",
        request=BlenderSetParentRequest,
        response=BlenderSetParentResponse,
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
        description="Add a modifier to a Blender object.",
        hints=ADDITIVE,
        operation="add_modifier",
        request=BlenderAddModifierRequest,
        response=BlenderAddModifierResponse,
        params={"params": Param(Dict[str, Any], {})},
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
        description="Open a .blend file, replacing the current scene.",
        hints=DESTRUCTIVE,
        operation="open_blend_file",
        request=BlenderOpenFileRequest,
        response=BlenderOpenFileResponse,
    ),
    BlenderTool(
        name="save_blender_file",
        description="Save the current .blend file.",
        hints=OVERWRITING,
        operation="save_blend_file",
        request=BlenderSaveFileRequest,
        response=BlenderSaveFileResponse,
    ),
    BlenderTool(
        name="import_blender_file",
        description="Import a file into the Blender scene.",
        hints=ADDITIVE,
        operation="import_file",
        request=BlenderImportFileRequest,
        response=BlenderImportFileResponse,
    ),
    BlenderTool(
        name="export_blender_file",
        description="Export scene objects to a file.",
        hints=DESTRUCTIVE,
        operation="export_file",
        request=BlenderExportFileRequest,
        response=BlenderExportFileResponse,
    ),
    BlenderTool(
        name="get_blender_file_info",
        description="Get information about the current .blend file.",
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
        description="Set the animation frame range.",
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
        description="Get current physics state of an object.",
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
        description="Bake physics simulation for a frame range.",
        hints=ADDITIVE,
        operation="bake_simulation",
        request=BlenderBakeSimulationRequest,
        response=BlenderBakeSimulationResponse,
    ),
    BlenderTool(
        name="free_blender_bake",
        description="Free (delete) baked physics simulation data.",
        hints=OVERWRITING,
        operation="free_bake",
        response=BlenderFreeBakeResponse,
        call=_free_bake,
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
            "Write larger results to a file and return its path instead."
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
            "Create a mesh object from raw vertex, edge, and face data. Use for procedural geometry."
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
    # -- SimReady asset format -----------------------------------------------
    BlenderTool(
        name="apply_simready_metadata",
        description=(
            "Apply NVIDIA SimReady metadata (semantic labels, physics properties, material info) "
            "to a Blender object as custom properties. Stored with simready_ prefix for USD export."
        ),
        hints=OVERWRITING,
        operation="apply_simready_metadata",
        request=SimReadyApplyMetadataRequest,
        response=SimReadyApplyMetadataResponse,
    ),
    BlenderTool(
        name="get_simready_metadata",
        description=(
            "Read NVIDIA SimReady metadata from a Blender object. Returns semantic labels, "
            "physics properties, and material info stored as simready_ custom properties."
        ),
        hints=READ_ONLY,
        operation="get_simready_metadata",
        request=SimReadyGetMetadataRequest,
        response=SimReadyGetMetadataResponse,
    ),
    BlenderTool(
        name="validate_simready_compliance",
        description=(
            "Validate Blender objects against NVIDIA SimReady conventions: naming "
            "(lowercase_underscore), scale (meters), clean transforms, material segmentation, hierarchy."
        ),
        hints=READ_ONLY,
        operation="validate_simready_compliance",
        request=SimReadyValidateRequest,
        response=SimReadyValidateResponse,
    ),
    BlenderTool(
        name="export_simready_usd",
        description=(
            "Export a SimReady-compliant USD file from Blender. Validates objects before export, "
            "selects the requested objects, and carries simready_ custom properties into USD attributes."
        ),
        hints=DESTRUCTIVE,
        operation="export_simready_usd",
        request=SimReadyExportRequest,
        response=SimReadyExportResponse,
    ),
    BlenderTool(
        name="setup_simready_hierarchy",
        description=(
            "Create a SimReady-compliant object hierarchy in Blender with a root empty (XForm "
            "equivalent) and parent the given children under it. Optionally applies semantic "
            "labels to the root."
        ),
        hints=ADDITIVE,
        operation="setup_simready_hierarchy",
        request=SimReadySetupHierarchyRequest,
        response=SimReadySetupHierarchyResponse,
    ),
)
