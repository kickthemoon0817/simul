"""Every Unreal MCP tool, declared once.

An ``UnrealTool`` entry is the whole declaration of one tool: the registered
name, the description and safety hints a client sees, the parameters with
their types, defaults and descriptions, the response model, and the
``UnrealRuntimeSession`` method the call runs. ``register_unreal_tools``
(``_reg_unreal.py``) builds the FastMCP wrapper, the request model the shared
envelope validates and the session call from the entry. The table is in
registration order, which is the order ``tools/list`` reports;
``simul.tool_surfaces.THIN_UNREAL_TOOLS`` picks the thin surface out of it
(``surface_tools``), and ``registered_tool_names`` is what a configuration
registers, which ``UnrealRuntimeAdapter.get_capabilities`` reports.

The metadata stays in the registration layer: the session implements the
operations and reads nothing here. A parameter without a description of its
own takes the ``Args:`` entry of its session method's docstring.

Session methods are named rather than imported, and the adapter is imported
only inside the functions that run a call, so this module loads where the
Unreal backend (aiohttp) is absent.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import KW_ONLY, dataclass
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Tuple, Type

from pydantic import BaseModel

from ...config import Settings
from ...tool_surfaces import THIN_UNREAL_TOOLS
from ..schemas.unreal import (
    ColorRGB,
    UnrealAddComponentResponse,
    UnrealAnalyzeSceneForRoboticsResponse,
    UnrealApplyForceResponse,
    UnrealApplyMeshBooleanResponse,
    UnrealAssignMaterialResponse,
    UnrealBatchOperationsResponse,
    UnrealCallActorFunctionResponse,
    UnrealCaptureViewportResponse,
    UnrealComputeConvexHullResponse,
    UnrealComputeMeshUvResponse,
    UnrealControlSimulationResponse,
    UnrealConvertMeshFormatResponse,
    UnrealConvertToSimreadyResponse,
    UnrealCreateMaterialInstanceResponse,
    UnrealCutMeshPlaneResponse,
    UnrealDecomposeConvexHullResponse,
    UnrealDeleteActorResponse,
    UnrealDescribeObjectResponse,
    UnrealEditMeshTopologyResponse,
    UnrealEnablePhysicsResponse,
    UnrealEngineInfoResponse,
    UnrealExecuteScriptResponse,
    UnrealExportUsdResponse,
    UnrealFocusActorResponse,
    UnrealGenerateMeshPrimitiveResponse,
    UnrealGenerateProceduralSceneResponse,
    UnrealGetActorBySemanticLabelResponse,
    UnrealGetActorInfoResponse,
    UnrealGetInterchangeInfoResponse,
    UnrealGetMaterialInfoResponse,
    UnrealGetSimulationStatusResponse,
    UnrealGetThumbnailResponse,
    UnrealHealthCheckResponse,
    UnrealImportUsdResponse,
    UnrealListActorsResponse,
    UnrealListInstancesResponse,
    UnrealLoadedMapResponse,
    UnrealPingResponse,
    UnrealQuerySceneGraphResponse,
    UnrealRemeshMeshResponse,
    UnrealSceneSummaryResponse,
    UnrealSearchAssetsResponse,
    UnrealSetActorParentResponse,
    UnrealSetActorPropertyResponse,
    UnrealSetActorTransformResponse,
    UnrealSetActorVisibilityResponse,
    UnrealSetCameraViewResponse,
    UnrealSetCollisionResponse,
    UnrealSetLightParamsResponse,
    UnrealSetMaterialParamsResponse,
    UnrealSetPhysicsParamsResponse,
    UnrealSetRenderSettingsResponse,
    UnrealSimplifyMeshResponse,
    UnrealSpawnActorResponse,
    UnrealSubdivideMeshResponse,
    UnrealValidateMeshResponse,
    UnrealValidateSimreadyResponse,
    UnrealViewportInfoResponse,
    Vector3,
)
from ..schemas.unreal_ui import UnrealUIRequest, UnrealUIResponse

#: Default of a parameter the caller must supply.
REQUIRED: Any = inspect.Parameter.empty


@dataclass(frozen=True)
class ToolParam:
    """One tool parameter as the client sees it, and how it reaches the session.

    Attributes:
        name: Parameter name in the input schema.
        annotation: Type the input schema and the request model are built from.
        default: Default value, or ``REQUIRED``.
        description: Schema description; None takes the session method's
            ``Args:`` entry for ``name``.
        target: Session keyword argument the value is passed as; None passes
            it as ``name``.
        convert: Turns the validated value into what the session method takes
            (an empty string into None, comma-separated text into a list, JSON
            text into an object). It runs inside the call, so a value it cannot
            parse fails the call like an error in the session would.
    """

    name: str
    annotation: Any
    default: Any = REQUIRED
    _: KW_ONLY
    description: Optional[str] = None
    target: Optional[str] = None
    convert: Optional[Callable[[Any], Any]] = None


@dataclass(frozen=True)
class UnrealTool:
    """One Unreal MCP tool.

    Attributes:
        name: Registered tool name.
        description: Tool description shown in the listing.
        response: Model a successful payload is validated against.
        read_only: The tool does not modify the editor.
        idempotent: Repeating the call with the same arguments has no further
            effect.
        destructive: The tool overwrites or deletes editor state, or runs
            arbitrary code.
        script: The tool runs agent-authored code or dispatches arbitrary
            editor functions, so it registers only when
            ``security.allow_script_execution`` is on.
        params: The parameters, in schema order.
        method: The ``UnrealRuntimeSession`` coroutine the call awaits. Each
            parameter is passed as a keyword argument (see ``ToolParam``)
            unless ``arguments`` builds them.
        arguments: Builds ``method``'s keyword arguments from the validated
            request, for parameters that do not map one to one onto it.
        sessionless: Runs instead of a session method, given the server
            settings and the validated request, for work that is not bound to
            the endpoint of the session the envelope opens.
        request: Model the arguments are validated against; None builds one
            from ``params``.
        check: Cross-parameter rule on the validated request, returning the
            message the call fails with as a ValidationError, or None. It is
            enforced by the model built from ``params``, so a tool with its
            own ``request`` model puts the rule in that model instead.
        usage: Picks what the usage log keeps from the arguments; None keeps
            nothing.
        echo: Picks what a call refused by validation echoes back as
            ``details.input``; None echoes every argument as given.
    """

    name: str
    description: str
    response: Type[BaseModel]
    read_only: bool
    idempotent: bool = False
    destructive: bool = False
    script: bool = False
    params: Tuple[ToolParam, ...] = ()
    method: Optional[str] = None
    arguments: Optional[Callable[[Any], Dict[str, Any]]] = None
    sessionless: Optional[Callable[[Settings, Any], Awaitable[Dict[str, Any]]]] = None
    request: Optional[Type[BaseModel]] = None
    check: Optional[Callable[[Any], Optional[str]]] = None
    usage: Optional[Callable[[Mapping[str, Any]], Dict[str, Any]]] = None
    echo: Optional[Callable[[Mapping[str, Any]], Dict[str, Any]]] = None


# ---------------------------------------------------------------------------
# Argument conversions
# ---------------------------------------------------------------------------


def _optional_text(value: str) -> Optional[str]:
    """An empty string leaves the session's default in place."""
    return value or None


def _name_list(value: str) -> Optional[List[str]]:
    """Comma-separated names, blanks dropped; an empty string means none."""
    return [item.strip() for item in value.split(",") if item.strip()] if value else None


def _path_list(value: str) -> List[str]:
    """Comma-separated object paths, each stripped."""
    return [item.strip() for item in value.split(",")]


def _floats(value: str) -> List[float]:
    """Comma-separated numbers."""
    return [float(item) for item in value.split(",")]


def _optional_floats(value: str) -> Optional[List[float]]:
    """Comma-separated numbers; an empty string means none."""
    return _floats(value) if value else None


def _optional_json(value: str) -> Any:
    """JSON text; an empty string means none."""
    return json.loads(value) if value else None


# ---------------------------------------------------------------------------
# Rules, argument builders and usage records of particular tools
# ---------------------------------------------------------------------------


def _png_format(request: Any) -> Optional[str]:
    """Viewport captures are written as PNG; ``simul unreal capture`` converts."""
    formats = {"png"}
    if request.format not in formats:
        return f"Invalid format '{request.format}'. Must be one of {sorted(formats)}"
    return None


def _exec_mode(request: Any) -> Optional[str]:
    """The ``ExecutionMode`` values ``ExecutePythonCommandEx`` accepts."""
    from ...adapters.unreal_runtime import UNREAL_EXEC_MODES

    if request.mode not in UNREAL_EXEC_MODES:
        return f"Invalid mode '{request.mode}'. Must be one of {sorted(UNREAL_EXEC_MODES)}"
    return None


_CAMERA_GROUPS: Dict[str, Tuple[str, str, str]] = {
    "location": ("location_x", "location_y", "location_z"),
    "rotation": ("rotation_pitch", "rotation_yaw", "rotation_roll"),
}


def _whole_camera_groups(request: Any) -> Optional[str]:
    """The camera location and rotation are each set whole or left unchanged."""
    for group, names in _CAMERA_GROUPS.items():
        given = [getattr(request, name) is not None for name in names]
        if any(given) and not all(given):
            return f"Give all three {group} components or none"
    return None


def _camera_arguments(request: Any) -> Dict[str, Any]:
    """The location and rotation triples; None for a group left unchanged."""
    arguments: Dict[str, Any] = {}
    for group, names in _CAMERA_GROUPS.items():
        values = tuple(getattr(request, name) for name in names)
        arguments[group] = values if None not in values else None
    return arguments


def _spawn_arguments(request: Any) -> Dict[str, Any]:
    """The location and rotation triples from their scalar components."""
    return {
        "asset_path": request.asset_path,
        "location": (request.location_x, request.location_y, request.location_z),
        "rotation": (request.rotation_pitch, request.rotation_yaw, request.rotation_roll),
        "label": request.label or None,
    }


def _discover_instances(settings: Settings, request: Any) -> Awaitable[Dict[str, Any]]:
    """``simul unreal instances``' discovery, which opens a session per probed port."""
    from ...adapters import unreal_connection

    return unreal_connection.discover_instances(settings, scan=request.scan)


def _all_arguments(arguments: Mapping[str, Any]) -> Dict[str, Any]:
    """Every argument, for tools whose arguments are small and say what was asked."""
    return dict(arguments)


def _script_usage(arguments: Mapping[str, Any]) -> Dict[str, Any]:
    """The script's size and mode; the source stays out of the log."""
    return {"code_bytes": len(arguments["code"]), "mode": arguments["mode"]}


#: Characters of a refused script echoed back in front of its size.
SCRIPT_ECHO_CHARS = 200


def _script_echo(arguments: Mapping[str, Any]) -> Dict[str, Any]:
    """A refused script's opening characters, size and mode, not the whole source."""
    code = str(arguments["code"])
    return {
        "code": code[:SCRIPT_ECHO_CHARS],
        **_script_usage(arguments),
        "code_truncated": len(code) > SCRIPT_ECHO_CHARS,
    }


# ---------------------------------------------------------------------------
# The tools, in registration order
# ---------------------------------------------------------------------------

UNREAL_TOOLS: Tuple[UnrealTool, ...] = (
    UnrealTool(
        name="unreal_health_check",
        description="Check connectivity to the Unreal Engine Remote Control API.",
        response=UnrealHealthCheckResponse,
        read_only=True,
        idempotent=True,
        method="health_check",
    ),
    UnrealTool(
        name="ping_unreal",
        description=(
            "Pre-flight check: verify that a running Unreal Engine instance is "
            "reachable on the configured Remote Control API port. Call this before "
            "other Unreal tools to confirm connectivity and measure latency."
        ),
        response=UnrealPingResponse,
        read_only=True,
        idempotent=True,
        method="ping",
    ),
    UnrealTool(
        name="list_unreal_instances",
        description=(
            "Discover running Unreal editors (same payload as `simul unreal "
            "instances`). Each instance: reachable, latency_ms, error, editor "
            "identity (pid, instance_id, project_name, project_path, map_path, "
            "viewports, dirty_map_packages; null when its Python does not "
            "answer), active (Unreal tools target it) and attached (the saved "
            "`simul unreal attach` selection). Ports that do not answer are "
            "omitted, except the active one."
        ),
        response=UnrealListInstancesResponse,
        read_only=True,
        idempotent=True,
        params=(
            ToolParam(
                "scan",
                bool,
                True,
                description=(
                    "When True, probe the configured port range plus the configured and attached "
                    "ports concurrently. When False, probe only the active endpoint."
                ),
            ),
        ),
        sessionless=_discover_instances,
        usage=_all_arguments,
    ),
    UnrealTool(
        name="capture_unreal_viewport",
        description=(
            "Capture a viewport screenshot via HighResScreenshot and return "
            "its path on the editor host. Pass inline=true to also receive "
            "the image itself as an image content block; captures above the "
            "inline size cap return the path alone."
        ),
        response=UnrealCaptureViewportResponse,
        read_only=False,
        method="capture_viewport",
        params=(
            ToolParam("resolution_x", int, 1920, description="Width in pixels."),
            ToolParam("resolution_y", int, 1080, description="Height in pixels."),
            ToolParam("format", str, "png", description="Image format (png). Use CLI capture to convert to JPEG."),
            ToolParam("inline", bool, False, description="Also return the image as an image content block."),
        ),
        check=_png_format,
        usage=_all_arguments,
    ),
    UnrealTool(
        name="execute_unreal_script",
        description=(
            "Execute arbitrary Python code inside the Unreal Engine editor. "
            "Use for operations not covered by other tools. The code runs via "
            "PythonScriptLibrary.ExecutePythonCommandEx. Print JSON to return "
            "structured data. For granular CLI operations see: simul unreal --help"
        ),
        response=UnrealExecuteScriptResponse,
        read_only=False,
        destructive=True,
        script=True,
        method="execute_script",
        params=(
            ToolParam("code", str, description="Python source code to execute."),
            ToolParam(
                "mode",
                str,
                "ExecuteFile",
                description=(
                    "ExecuteFile (multi-line), EvaluateStatement (expression), "
                    "or ExecuteStatement (single statement)."
                ),
            ),
        ),
        check=_exec_mode,
        usage=_script_usage,
        echo=_script_echo,
    ),
    UnrealTool(
        name="control_unreal_ui",
        description=(
            "Named controls in the explicitly attached Unreal editor: inspect, select actors, "
            "edit transforms, pilot/eject actors, set game view, and show per-agent overlay cursors. "
            "Requires simul unreal attach."
        ),
        response=UnrealUIResponse,
        read_only=False,
        method="control_ui",
        params=(
            ToolParam(
                "agent_control",
                str,
                description=(
                    "inspect, select_actor, clear_selection, set_property, pilot_actor, "
                    "eject_actor, set_game_view, move_cursor, or clear_cursor."
                ),
            ),
            ToolParam(
                "target",
                Optional[str],
                None,
                description="Actor full path or unique label; otherwise the sole selected actor.",
            ),
            ToolParam(
                "property_name",
                Optional[str],
                None,
                description="location, rotation (Pitch/Yaw/Roll degrees), or scale.",
            ),
            ToolParam(
                "value",
                Optional[List[float]],
                None,
                description="Three finite numbers for set_property; location is in cm.",
            ),
            ToolParam("enabled", Optional[bool], None, description="Required boolean for set_game_view."),
            ToolParam(
                "agent_id",
                str,
                "agent",
                description="Stable agent label; supply a unique value for each concurrent agent.",
            ),
            ToolParam(
                "position",
                Optional[List[float]],
                None,
                description="move_cursor only: normalized [x,y] from bottom-left; defaults to center.",
            ),
            ToolParam(
                "activity",
                Optional[str],
                None,
                description="move_cursor only: short label describing the current work.",
            ),
        ),
        request=UnrealUIRequest,
    ),
    # -- Granular tools (full surface only) --------------------------------
    UnrealTool(
        name="get_unreal_engine_info",
        description="Get Unreal Engine runtime information.",
        response=UnrealEngineInfoResponse,
        read_only=True,
        idempotent=True,
        method="get_engine_info",
    ),
    UnrealTool(
        name="get_unreal_loaded_map",
        description="Get the currently loaded persistent level path.",
        response=UnrealLoadedMapResponse,
        read_only=True,
        idempotent=True,
        method="get_loaded_map",
    ),
    # -- Scene read operations ----------------------------------------------
    UnrealTool(
        name="list_unreal_actors",
        description="List actors in the current Unreal Engine level with optional class and tag filters.",
        response=UnrealListActorsResponse,
        read_only=True,
        idempotent=True,
        method="list_actors",
        params=(
            ToolParam(
                "class_filter",
                str,
                "",
                description="Filter by UClass name (empty string = no filter).",
                convert=_optional_text,
            ),
            ToolParam(
                "tag_filter",
                str,
                "",
                description="Filter by actor tag (empty string = no filter).",
                convert=_optional_text,
            ),
            ToolParam("max_results", int, 200, description="Maximum number of actors to return."),
        ),
    ),
    UnrealTool(
        name="get_unreal_actor_info",
        description="Get detailed information about a specific actor including transform, components, and tags.",
        response=UnrealGetActorInfoResponse,
        read_only=True,
        idempotent=True,
        method="get_actor_info",
        params=(ToolParam("actor_path", str, description="Full object path of the actor."),),
    ),
    UnrealTool(
        name="search_unreal_assets",
        description="Search the Unreal Asset Registry by name, class, or package path.",
        response=UnrealSearchAssetsResponse,
        read_only=True,
        idempotent=True,
        method="search_assets",
        params=(
            ToolParam("query", str, "", description="Search query string."),
            ToolParam(
                "class_names",
                str,
                "",
                description="Comma-separated UClass names to filter.",
                convert=_name_list,
            ),
            ToolParam(
                "package_paths",
                str,
                "",
                description="Comma-separated package paths to search.",
                convert=_name_list,
            ),
            ToolParam("max_results", int, 100, description="Maximum number of results."),
        ),
    ),
    UnrealTool(
        name="describe_unreal_object",
        description="Get full property and function metadata for any UObject by path.",
        response=UnrealDescribeObjectResponse,
        read_only=True,
        idempotent=True,
        method="describe_object",
        params=(ToolParam("object_path", str, description="Full object path."),),
    ),
    UnrealTool(
        name="get_unreal_actor_thumbnail",
        description=(
            "Get a thumbnail image for an Unreal asset. The image arrives as an "
            "MCP image content block, followed by a JSON block with asset_path, "
            "format, width, height and image_attached=true."
        ),
        response=UnrealGetThumbnailResponse,
        read_only=True,
        method="get_actor_thumbnail",
        params=(
            ToolParam("asset_path", str, description="Full asset path."),
            ToolParam("width", int, 256, description="Thumbnail width in pixels."),
            ToolParam("height", int, 256, description="Thumbnail height in pixels."),
        ),
    ),
    UnrealTool(
        name="summarize_unreal_scene",
        description="Generate an LLM-friendly digest of the current Unreal scene.",
        response=UnrealSceneSummaryResponse,
        read_only=True,
        idempotent=True,
        method="summarize_scene",
    ),
    # -- Viewport and visual observation --------------------------------------
    UnrealTool(
        name="get_unreal_viewport_info",
        description="Get active viewport camera and render information.",
        response=UnrealViewportInfoResponse,
        read_only=True,
        idempotent=True,
        method="get_viewport_info",
    ),
    UnrealTool(
        name="set_unreal_camera_view",
        description="Set the editor viewport camera position and rotation.",
        response=UnrealSetCameraViewResponse,
        read_only=False,
        idempotent=True,
        method="set_camera_view",
        params=(
            ToolParam("location_x", Optional[float], None, description="Camera X position in cm."),
            ToolParam("location_y", Optional[float], None, description="Camera Y position in cm."),
            ToolParam("location_z", Optional[float], None, description="Camera Z position in cm."),
            ToolParam("rotation_pitch", Optional[float], None, description="Camera pitch in degrees."),
            ToolParam("rotation_yaw", Optional[float], None, description="Camera yaw in degrees."),
            ToolParam("rotation_roll", Optional[float], None, description="Camera roll in degrees."),
        ),
        arguments=_camera_arguments,
        check=_whole_camera_groups,
    ),
    UnrealTool(
        name="focus_unreal_on_actor",
        description="Focus the editor viewport camera on a specific actor.",
        response=UnrealFocusActorResponse,
        read_only=False,
        method="focus_on_actor",
        params=(
            ToolParam("actor_path", str, description="Full actor path to focus on."),
            ToolParam("distance", float, 0.0, description="Camera distance from actor (0 = auto-fit)."),
        ),
    ),
    # -- Scene manipulation ---------------------------------------------------
    UnrealTool(
        name="spawn_unreal_actor",
        description="Spawn an actor from a class or asset path.",
        response=UnrealSpawnActorResponse,
        read_only=False,
        method="spawn_actor",
        params=(
            ToolParam("asset_path", str, description="Asset or class path to spawn from."),
            ToolParam("location_x", float, 0.0, description="Spawn X position in Unreal units (cm)."),
            ToolParam("location_y", float, 0.0, description="Spawn Y position in Unreal units (cm)."),
            ToolParam("location_z", float, 0.0, description="Spawn Z position in Unreal units (cm); Unreal is Z-up."),
            ToolParam("rotation_pitch", float, 0.0, description="Spawn pitch in degrees (rotation about Y)."),
            ToolParam("rotation_yaw", float, 0.0, description="Spawn yaw in degrees (rotation about Z)."),
            ToolParam("rotation_roll", float, 0.0, description="Spawn roll in degrees (rotation about X)."),
            ToolParam("label", str, "", description="Optional actor label in the outliner."),
        ),
        arguments=_spawn_arguments,
    ),
    UnrealTool(
        name="delete_unreal_actor",
        description="Delete an actor from the level. DESTRUCTIVE operation.",
        response=UnrealDeleteActorResponse,
        read_only=False,
        destructive=True,
        method="delete_actor",
        params=(ToolParam("actor_path", str),),
    ),
    UnrealTool(
        name="set_unreal_actor_transform",
        description=(
            "Set an actor's world location, rotation and/or scale. Each is optional; "
            "one left out keeps its current value. Returns the transform read back."
        ),
        response=UnrealSetActorTransformResponse,
        read_only=False,
        idempotent=True,
        destructive=True,
        method="set_actor_transform",
        params=(
            ToolParam("actor_path", str, description="Actor object path."),
            ToolParam(
                "location",
                Optional[Vector3],
                None,
                description=(
                    "World position [x, y, z] in Unreal units (cm); Unreal is Z-up. "
                    "Omit to keep the current location."
                ),
            ),
            ToolParam(
                "rotation",
                Optional[Vector3],
                None,
                description=(
                    "World rotation [pitch, yaw, roll] in degrees: pitch about Y, yaw about Z, "
                    "roll about X. Omit to keep the current rotation."
                ),
            ),
            ToolParam(
                "scale",
                Optional[Vector3],
                None,
                description="Scale factors [x, y, z]; 1.0 is unscaled. Omit to keep the current scale.",
            ),
        ),
    ),
    UnrealTool(
        name="set_unreal_actor_property",
        description="Set a property on an Unreal actor by name and JSON value.",
        response=UnrealSetActorPropertyResponse,
        read_only=False,
        idempotent=True,
        destructive=True,
        method="set_actor_property",
        params=(
            ToolParam("actor_path", str),
            ToolParam("property_name", str),
            ToolParam("property_value", str),
            ToolParam("generate_transaction", bool, True),
        ),
    ),
    UnrealTool(
        name="call_unreal_actor_function",
        description="Call a BlueprintCallable UFUNCTION on an actor.",
        response=UnrealCallActorFunctionResponse,
        read_only=False,
        destructive=True,
        script=True,
        method="call_actor_function",
        params=(
            ToolParam("actor_path", str),
            ToolParam("function_name", str),
            ToolParam("parameters", str, "", convert=_optional_text),
        ),
    ),
    UnrealTool(
        name="set_unreal_actor_parent",
        description="Attach an actor to a parent actor or detach it.",
        response=UnrealSetActorParentResponse,
        read_only=False,
        destructive=True,
        method="set_actor_parent",
        params=(
            ToolParam("actor_path", str),
            ToolParam("parent_path", str, "", convert=_optional_text),
        ),
    ),
    UnrealTool(
        name="add_unreal_component",
        description="Add a component to an Unreal actor.",
        response=UnrealAddComponentResponse,
        read_only=False,
        method="add_component",
        params=(
            ToolParam("actor_path", str),
            ToolParam("component_class", str),
            ToolParam("component_name", str, "", convert=_optional_text),
        ),
    ),
    UnrealTool(
        name="set_unreal_actor_visibility",
        description="Set actor visibility in the Unreal level.",
        response=UnrealSetActorVisibilityResponse,
        read_only=False,
        idempotent=True,
        destructive=True,
        method="set_actor_visibility",
        params=(
            ToolParam("actor_path", str),
            ToolParam("visible", bool, True),
            ToolParam("propagate", bool, True),
        ),
    ),
    # -- Materials, lighting and rendering --------------------------------------
    UnrealTool(
        name="get_unreal_material_info",
        description="Get material instance parameters and metadata.",
        response=UnrealGetMaterialInfoResponse,
        read_only=True,
        idempotent=True,
        method="get_material_info",
        params=(ToolParam("material_path", str),),
    ),
    UnrealTool(
        name="set_unreal_material_params",
        description="Set scalar/vector/texture parameters on a Material Instance.",
        response=UnrealSetMaterialParamsResponse,
        read_only=False,
        idempotent=True,
        destructive=True,
        method="set_material_params",
        params=(
            ToolParam("material_path", str, description="Material Instance asset path."),
            ToolParam(
                "scalar_params_json",
                str,
                "",
                description=(
                    'JSON object of scalar parameter name to float value, e.g. \'{"Roughness": 0.4}\'. '
                    "Empty string sets none."
                ),
                target="scalar_params",
                convert=_optional_json,
            ),
            ToolParam(
                "vector_params_json",
                str,
                "",
                description=(
                    "JSON object of vector parameter name to an [R, G, B, A] list of floats. "
                    "Empty string sets none."
                ),
                target="vector_params",
                convert=_optional_json,
            ),
            ToolParam(
                "texture_params_json",
                str,
                "",
                description=(
                    "JSON object of texture parameter name to a texture asset path. Empty string sets none."
                ),
                target="texture_params",
                convert=_optional_json,
            ),
        ),
    ),
    UnrealTool(
        name="create_unreal_material_instance",
        description="Create a Material Instance Constant from a parent material.",
        response=UnrealCreateMaterialInstanceResponse,
        read_only=False,
        method="create_material_instance",
        params=(
            ToolParam("parent_path", str),
            ToolParam("instance_name", str),
            ToolParam("save_path", str, ""),
        ),
    ),
    UnrealTool(
        name="assign_unreal_material",
        description="Assign a material to a mesh component's material slot.",
        response=UnrealAssignMaterialResponse,
        read_only=False,
        idempotent=True,
        destructive=True,
        method="assign_material",
        params=(
            ToolParam("actor_path", str),
            ToolParam("material_path", str),
            ToolParam("slot_index", int, 0),
        ),
    ),
    UnrealTool(
        name="set_unreal_light_params",
        description=(
            "Set light component parameters (intensity, color, temperature, shadows); "
            "a parameter left out keeps its current value."
        ),
        response=UnrealSetLightParamsResponse,
        read_only=False,
        idempotent=True,
        destructive=True,
        method="set_light_params",
        params=(
            ToolParam("actor_path", str),
            ToolParam("intensity", Optional[float], None),
            ToolParam("color", Optional[ColorRGB], None),
            ToolParam("temperature", Optional[float], None),
            ToolParam("use_temperature", Optional[bool], None),
            ToolParam("attenuation_radius", Optional[float], None),
            ToolParam("cast_shadows", Optional[bool], None),
        ),
    ),
    UnrealTool(
        name="set_unreal_render_settings",
        description="Set rendering or post-process settings via console command.",
        response=UnrealSetRenderSettingsResponse,
        read_only=False,
        idempotent=True,
        destructive=True,
        method="set_render_settings",
        params=(
            ToolParam("setting_name", str),
            ToolParam("setting_value", str),
        ),
    ),
    # -- Physics and simulation control ------------------------------------------
    UnrealTool(
        name="control_unreal_simulation",
        description="Control Play-In-Editor: start, stop, pause or resume. Exact frame stepping is unsupported.",
        response=UnrealControlSimulationResponse,
        read_only=False,
        method="control_simulation",
        params=(ToolParam("action", str),),
    ),
    UnrealTool(
        name="get_unreal_simulation_status",
        description="Get current Play-In-Editor simulation status (playing, paused, stopped).",
        response=UnrealGetSimulationStatusResponse,
        read_only=True,
        idempotent=True,
        method="get_simulation_status",
    ),
    UnrealTool(
        name="enable_unreal_physics",
        description="Enable or disable physics simulation on an actor.",
        response=UnrealEnablePhysicsResponse,
        read_only=False,
        idempotent=True,
        destructive=True,
        method="enable_physics",
        params=(
            ToolParam("actor_path", str),
            ToolParam("enable", bool, True),
            ToolParam("simulate_physics", bool, True),
        ),
    ),
    UnrealTool(
        name="set_unreal_collision",
        description=(
            "Set an actor's collision preset and/or enable or disable its collision; "
            "what is left out is kept. Returns the state read back."
        ),
        response=UnrealSetCollisionResponse,
        read_only=False,
        idempotent=True,
        destructive=True,
        method="set_collision",
        params=(
            ToolParam("actor_path", str),
            ToolParam("collision_preset", str, ""),
            ToolParam("collision_enabled", Optional[bool], None),
        ),
    ),
    UnrealTool(
        name="apply_unreal_force",
        description="Apply a force or impulse to an actor's physics body.",
        response=UnrealApplyForceResponse,
        read_only=False,
        method="apply_force",
        params=(
            ToolParam("actor_path", str),
            ToolParam("force_x", float, 0.0),
            ToolParam("force_y", float, 0.0),
            ToolParam("force_z", float, 0.0),
            ToolParam("is_impulse", bool, False),
            ToolParam("location_x", Optional[float], None),
            ToolParam("location_y", Optional[float], None),
            ToolParam("location_z", Optional[float], None),
        ),
    ),
    UnrealTool(
        name="set_unreal_physics_params",
        description="Set physics body parameters (mass, damping, gravity) on an actor.",
        response=UnrealSetPhysicsParamsResponse,
        read_only=False,
        idempotent=True,
        destructive=True,
        method="set_physics_params",
        params=(
            ToolParam("actor_path", str),
            ToolParam("mass", Optional[float], None),
            ToolParam("linear_damping", Optional[float], None),
            ToolParam("angular_damping", Optional[float], None),
            ToolParam("enable_gravity", Optional[bool], None),
        ),
    ),
    # -- USD and SimReady ----------------------------------------------------------
    UnrealTool(
        name="import_unreal_usd",
        description="Import USD with Unreal's optional USDImporter plugin, enabled at editor startup.",
        response=UnrealImportUsdResponse,
        read_only=False,
        method="import_usd",
        params=(
            ToolParam("usd_path", str, description="Path to the USD file on disk; must be inside the sandbox."),
            ToolParam("destination_path", str, "/Game/Imports", description="Content browser destination path."),
            ToolParam(
                "import_options",
                Optional[Dict[str, Any]],
                None,
                description=(
                    "Boolean USD options: import_actors, import_geometry, import_materials, "
                    "import_lights, import_cameras."
                ),
            ),
        ),
    ),
    UnrealTool(
        name="export_unreal_usd",
        description="Export selected Unreal actors with the optional USDImporter plugin's LevelExporterUSD.",
        response=UnrealExportUsdResponse,
        read_only=False,
        destructive=True,
        method="export_usd",
        params=(
            ToolParam(
                "actor_paths",
                str,
                description="Comma-separated actor object paths to export.",
                convert=_path_list,
            ),
            ToolParam("output_path", str, description="Output USD file path; must be inside the sandbox."),
            ToolParam(
                "export_options",
                Optional[Dict[str, Any]],
                None,
                description="Boolean USD options: export_actor_folders, export_sublayers.",
            ),
        ),
    ),
    UnrealTool(
        name="convert_to_simready",
        description="Unavailable in Unreal: SimReady conversion returns UnsupportedOperation without writing files.",
        response=UnrealConvertToSimreadyResponse,
        read_only=False,
        destructive=True,
        method="convert_to_simready",
        params=(
            ToolParam("usd_path", str, description="Source USD file; must be inside the sandbox."),
            ToolParam("output_path", str, description="Output SimReady USD file path; must be inside the sandbox."),
            ToolParam("add_physics", bool, True, description="Add physics schemas (rigid body) to the output."),
            ToolParam("add_collision", bool, True, description="Generate collision geometry."),
            ToolParam("add_semantic_labels", bool, True, description="Attach semantic labels to the output prims."),
            ToolParam("target_up_axis", str, "Z", description="Up axis written to the output stage (Y or Z)."),
            ToolParam(
                "target_units",
                str,
                "meters",
                description="Unit system of the output stage (meters or centimeters).",
            ),
        ),
    ),
    UnrealTool(
        name="validate_simready_asset",
        description="Unavailable in Unreal: SimReady validation returns UnsupportedOperation.",
        response=UnrealValidateSimreadyResponse,
        read_only=True,
        idempotent=True,
        method="validate_simready_asset",
        params=(
            ToolParam("usd_path", str, description="USD file path to validate; must be inside the sandbox."),
            ToolParam(
                "checks",
                str,
                "",
                description=(
                    "Comma-separated validation checks to run (physics, collision, materials, scale, "
                    "up_axis, semantics). Empty string runs every check."
                ),
                convert=_name_list,
            ),
        ),
    ),
    UnrealTool(
        name="get_unreal_interchange_info",
        description="Query available Interchange pipelines and supported formats.",
        response=UnrealGetInterchangeInfoResponse,
        read_only=True,
        idempotent=True,
        method="get_interchange_info",
    ),
    # -- Advanced agent tools ------------------------------------------------------
    UnrealTool(
        name="batch_unreal_operations",
        description="Execute multiple Remote Control operations in one HTTP call.",
        response=UnrealBatchOperationsResponse,
        read_only=False,
        destructive=True,
        script=True,
        method="batch_operations",
        params=(ToolParam("operations", str, convert=json.loads),),
    ),
    UnrealTool(
        name="query_unreal_scene_graph",
        description="Query the Unreal scene graph hierarchy.",
        response=UnrealQuerySceneGraphResponse,
        read_only=True,
        idempotent=True,
        method="query_scene_graph",
        params=(
            ToolParam("root_path", Optional[str], None),
            ToolParam("max_depth", int, 10),
            ToolParam("include_components", bool, False),
            ToolParam("class_filter", Optional[str], None),
        ),
    ),
    UnrealTool(
        name="analyze_unreal_scene_for_robotics",
        description=(
            "Classify level actors for robotics from their collision setup: floor-like surfaces a "
            "robot can stand on, small Movable objects it can grasp, and collision counts and "
            "problems (static meshes with physics collision but no simple shapes). Uses the "
            "bounding box of each actor's collision-enabled components; the thresholds come "
            "back as criteria."
        ),
        response=UnrealAnalyzeSceneForRoboticsResponse,
        read_only=True,
        idempotent=True,
        method="analyze_scene_for_robotics",
        params=(
            ToolParam(
                "analysis_types",
                str,
                "",
                description=(
                    "Comma-separated analyses: traversability, graspability, collision_complexity. "
                    "Empty runs all three."
                ),
                convert=_name_list,
            ),
            ToolParam(
                "actor_filter",
                Optional[str],
                None,
                description=(
                    "Only analyze actors whose label or object path contains this text, ignoring case; "
                    "empty or null analyzes every actor."
                ),
            ),
        ),
    ),
    UnrealTool(
        name="generate_unreal_procedural_scene",
        description=(
            "Build a room, corridor, warehouse or outdoor test scene from Engine basic shapes inside "
            "the bounds, in one undoable transaction; the same arguments give the same scene. Returns "
            "the spawned actor paths and the tag and outliner folder that group them."
        ),
        response=UnrealGenerateProceduralSceneResponse,
        read_only=False,
        method="generate_procedural_scene",
        params=(
            ToolParam("scene_type", str, description="room, corridor, warehouse or outdoor."),
            ToolParam(
                "parameters",
                str,
                "{}",
                description=(
                    "JSON object; lengths in cm, counts 0 to 200. Any type: seed (integer, 42). "
                    "room: wall_thickness (20), wall_height (bounds top), door_width (100, capped by the "
                    "side walls; 0 for none), "
                    "props (one per 16 m2 of floor). corridor: wall_thickness, wall_height, props (one "
                    "per 3 m). warehouse: wall_thickness, wall_height, rows (as many as fit), "
                    "shelf_depth (100), shelf_height (300, capped by the bounds), aisle_width (250), "
                    "props (one per 16 m2 of free floor). outdoor: trees (one per 20 m2), rocks (one "
                    "per 10 m2). Props are Movable boxes; everything else is Static."
                ),
                convert=_optional_json,
            ),
            ToolParam(
                "bounds_min",
                str,
                "",
                description="Minimum corner as comma-separated x,y,z in cm; empty is 0,0,0.",
                convert=_optional_floats,
            ),
            ToolParam(
                "bounds_max",
                str,
                "",
                description="Maximum corner as comma-separated x,y,z in cm; empty is 1000,1000,500.",
                convert=_optional_floats,
            ),
        ),
    ),
    UnrealTool(
        name="get_unreal_actor_by_semantic_label",
        description="Find level actors whose actor tags match a label: exact, contains or regex, ignoring case.",
        response=UnrealGetActorBySemanticLabelResponse,
        read_only=True,
        idempotent=True,
        method="get_actor_by_semantic_label",
        params=(
            ToolParam("label", str),
            ToolParam("match_mode", str, "exact"),
            ToolParam("max_results", int, 100),
        ),
    ),
    # -- Geometry and modeling (GeometryScript) --------------------------------------
    UnrealTool(
        name="generate_unreal_mesh_primitive",
        description="Create a parametric mesh primitive (box, sphere, cylinder, cone, torus, capsule).",
        response=UnrealGenerateMeshPrimitiveResponse,
        read_only=False,
        method="generate_mesh_primitive",
        params=(
            ToolParam("primitive_type", str),
            ToolParam("dimensions", str, "{}", convert=_optional_json),
            ToolParam("segments", int, 32),
            ToolParam("location", str, "", convert=_optional_floats),
            ToolParam("actor_label", Optional[str], None),
        ),
    ),
    UnrealTool(
        name="apply_unreal_mesh_boolean",
        description="Apply boolean operation (union, subtract, intersect) between two meshes.",
        response=UnrealApplyMeshBooleanResponse,
        read_only=False,
        destructive=True,
        method="apply_mesh_boolean",
        params=(
            ToolParam("target_mesh_path", str),
            ToolParam("tool_mesh_path", str),
            ToolParam("operation", str),
        ),
    ),
    UnrealTool(
        name="compute_unreal_convex_hull",
        description="Compute convex hull envelope of a mesh.",
        response=UnrealComputeConvexHullResponse,
        read_only=False,
        method="compute_convex_hull",
        params=(ToolParam("mesh_path", str),),
    ),
    UnrealTool(
        name="decompose_unreal_convex_hull",
        description="V-HACD convex decomposition for collision geometry.",
        response=UnrealDecomposeConvexHullResponse,
        read_only=False,
        destructive=True,
        method="decompose_convex_hull",
        params=(
            ToolParam("mesh_path", str),
            ToolParam("max_hulls", int, 16),
            ToolParam("max_vertices_per_hull", int, 32),
            ToolParam("min_cluster_size", int, 256),
            ToolParam("resolution", int, 100000),
        ),
    ),
    UnrealTool(
        name="edit_unreal_mesh_topology",
        description="Edit mesh topology (extrude, bevel, inset, loop cut, scale_faces).",
        response=UnrealEditMeshTopologyResponse,
        read_only=False,
        destructive=True,
        method="edit_mesh_topology",
        params=(
            ToolParam("mesh_path", str),
            ToolParam("operation", str),
            ToolParam("face_selection", Optional[str], None),
            ToolParam("edge_selection", Optional[str], None),
            ToolParam("distance", Optional[float], None),
            ToolParam("offset", Optional[float], None),
            ToolParam("scale", str, "", convert=_optional_floats),
            ToolParam("count", Optional[int], None),
        ),
    ),
    UnrealTool(
        name="subdivide_unreal_mesh",
        description="Catmull-Clark / Loop / bilinear subdivision.",
        response=UnrealSubdivideMeshResponse,
        read_only=False,
        destructive=True,
        method="subdivide_mesh",
        params=(
            ToolParam("mesh_path", str),
            ToolParam("level", int, 2),
            ToolParam("scheme", str, "catmull_clark"),
        ),
    ),
    UnrealTool(
        name="simplify_unreal_mesh",
        description="Simplify/decimate a mesh to reduce triangle count.",
        response=UnrealSimplifyMeshResponse,
        read_only=False,
        destructive=True,
        method="simplify_mesh",
        params=(
            ToolParam("mesh_path", str),
            ToolParam("target_triangle_count", Optional[int], None),
            ToolParam("target_percentage", Optional[float], None),
            ToolParam("max_error", Optional[float], None),
        ),
    ),
    UnrealTool(
        name="cut_unreal_mesh_plane",
        description="Cut/slice a mesh along an arbitrary plane.",
        response=UnrealCutMeshPlaneResponse,
        read_only=False,
        destructive=True,
        method="cut_mesh_plane",
        params=(
            ToolParam("mesh_path", str),
            ToolParam("plane_origin", str, convert=_floats),
            ToolParam("plane_normal", str, convert=_floats),
            ToolParam("fill_holes", bool, True),
            ToolParam("keep_both_sides", bool, False),
        ),
    ),
    UnrealTool(
        name="validate_unreal_mesh",
        description="Validate mesh integrity (manifold, normals, degenerates, self-intersection).",
        response=UnrealValidateMeshResponse,
        read_only=True,
        idempotent=True,
        method="validate_mesh",
        params=(
            ToolParam("mesh_path", str),
            ToolParam("checks", str, "", convert=_name_list),
        ),
    ),
    UnrealTool(
        name="convert_unreal_mesh_format",
        description="Convert mesh between formats (static mesh, dynamic mesh, skeletal mesh).",
        response=UnrealConvertMeshFormatResponse,
        read_only=False,
        destructive=True,
        method="convert_mesh_format",
        params=(
            ToolParam("mesh_path", str),
            ToolParam("target_format", str),
            ToolParam("tessellation_options", str, "{}", convert=_optional_json),
        ),
    ),
    UnrealTool(
        name="remesh_unreal_mesh",
        description="Remesh a mesh (uniform, adaptive) to improve triangle quality.",
        response=UnrealRemeshMeshResponse,
        read_only=False,
        destructive=True,
        method="remesh_mesh",
        params=(
            ToolParam("mesh_path", str),
            ToolParam("mode", str, "uniform"),
            ToolParam("target_edge_length", Optional[float], None),
            ToolParam("target_triangle_count", Optional[int], None),
            ToolParam("smoothing_iterations", int, 3),
        ),
    ),
    UnrealTool(
        name="compute_unreal_mesh_uv",
        description="Generate or recompute UV coordinates for a mesh.",
        response=UnrealComputeMeshUvResponse,
        read_only=False,
        destructive=True,
        method="compute_mesh_uv",
        params=(
            ToolParam("mesh_path", str),
            ToolParam("method", str, "auto_uv"),
            ToolParam("uv_channel", int, 0),
            ToolParam("island_padding", float, 2.0),
        ),
    ),
)


def surface_tools(thin: bool) -> Tuple[UnrealTool, ...]:
    """The declarations a tool surface registers, in registration order.

    Args:
        thin: The thin surface, which keeps ``THIN_UNREAL_TOOLS``; otherwise
            the full one, which keeps every declaration.

    Returns:
        The declarations, script tools included.
    """
    return tuple(spec for spec in UNREAL_TOOLS if not thin or spec.name in THIN_UNREAL_TOOLS)


def registered_tool_names(settings: Settings) -> Tuple[str, ...]:
    """The Unreal tools a server built with ``settings`` registers, in registration order.

    ``unreal.tool_surface`` picks the surface (``surface_tools``). Script
    tools register only with ``security.allow_script_execution`` on: the
    server's ``_script_tool`` leaves them out otherwise.

    Args:
        settings: The server settings.

    Returns:
        The tool names.
    """
    allow_scripts = settings.security.allow_script_execution
    return tuple(
        spec.name
        for spec in surface_tools(settings.unreal.tool_surface == "thin")
        if allow_scripts or not spec.script
    )
