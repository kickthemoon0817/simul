"""Unreal Engine MCP schemas."""

import json
from typing import Annotated, Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator
from pydantic.json_schema import SkipJsonSchema

# ---------------------------------------------------------------------------
# Tool input types
# ---------------------------------------------------------------------------


def _list_from_json_string(value: Any) -> Any:
    """Accept ``'[1, 2, 3]'`` for a list: some MCP clients send lists as JSON text."""
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return value
        if isinstance(parsed, list):
            return parsed
    return value


#: Three finite numbers: [x, y, z] or [pitch, yaw, roll].
Vector3 = Annotated[
    List[Annotated[float, Field(allow_inf_nan=False)]],
    BeforeValidator(_list_from_json_string),
    Field(min_length=3, max_length=3),
]

#: An [r, g, b] color, each channel between 0 and 1.
ColorRGB = Annotated[
    List[Annotated[float, Field(ge=0.0, le=1.0)]],
    BeforeValidator(_list_from_json_string),
    Field(min_length=3, max_length=3),
]


def _pie_active_field() -> Any:
    """The ``pie_active`` field of the responses that read the editor world."""
    return Field(
        False,
        description=(
            "Whether Play In Editor (or Simulate) was running. The values are the editor "
            "world's, not the running game's; Unreal's editor lists no level actors then, so "
            "actor listings read the editor world directly"
        ),
    )

# ---------------------------------------------------------------------------
# Unreal Engine schemas -- Phase 0
# ---------------------------------------------------------------------------


class UnrealPingResponse(BaseModel):
    """Response for a lightweight Unreal Engine reachability probe."""

    success: bool = Field(..., description="Whether request was successful")
    reachable: bool = Field(
        ...,
        description=(
            "Whether Remote Control answered GET /remote/info. A refusal (HTTP "
            "401, error_type AuthenticationError) is an answer: reachable, with "
            "error saying to set UNREAL__PASSPHRASE. list_unreal_instances "
            "reports the same"
        ),
    )
    address: str = Field(..., description="host:port that was probed")
    latency_ms: Optional[float] = Field(None, description="Round-trip time in milliseconds")
    error: Optional[str] = Field(None, description="Error message when success is False")


class UnrealInstanceInfo(BaseModel):
    """One endpoint reported by Unreal discovery.

    ``list_unreal_instances`` and ``simul unreal instances`` return the same
    records (``simul.adapters.unreal_connection.discover_instances``). The
    identity fields are null, and the lists empty, when the editor's Python
    did not answer; ``error`` says why.
    """

    name: str = Field(
        ..., description="'default' for the active endpoint, otherwise 'unreal-<port>'"
    )
    host: str = Field(..., description="Remote Control API host")
    port: int = Field(..., description="Remote Control API port")
    reachable: bool = Field(
        ...,
        description=(
            "Whether Remote Control answered GET /remote/info, as ping_unreal "
            "decides it: a refusal (HTTP 401) is reachable, with error saying "
            "to set UNREAL__PASSPHRASE"
        ),
    )
    active: bool = Field(
        False,
        description=(
            "Whether Unreal tools target this endpoint: the configured "
            "host/port, or the attached editor in attached mode"
        ),
    )
    attached: bool = Field(
        False,
        description=(
            "Whether this editor, its loaded map and the selected viewport "
            "match the saved simul unreal attach selection"
        ),
    )
    latency_ms: Optional[float] = Field(
        None, description="GET /remote/info round trip in milliseconds"
    )
    error: Optional[str] = Field(
        None,
        description=(
            "Why the endpoint is unreachable, refused the request or has no "
            "identity available"
        ),
    )
    instance_id: Optional[str] = Field(
        None, description="Editor process identity; changes when the editor restarts"
    )
    document_id: Optional[str] = Field(
        None, description="Loaded-map generation; changes when a map is opened or replaced"
    )
    pid: Optional[int] = Field(None, description="Editor process ID")
    engine_version: Optional[str] = Field(None, description="Unreal Engine version")
    project_name: Optional[str] = Field(None, description="Project name")
    project_path: Optional[str] = Field(None, description="Absolute .uproject path")
    map_path: Optional[str] = Field(None, description="Loaded editor world path")
    active_viewport: Optional[str] = Field(
        None, description="Config key of the active level viewport"
    )
    viewports: List[str] = Field(
        default_factory=list,
        description="Level viewport config keys, for simul unreal attach --viewport",
    )
    dirty_map_packages: List[str] = Field(
        default_factory=list, description="Map packages with unsaved changes"
    )


class UnrealListInstancesResponse(BaseModel):
    """Response listing discovered Unreal editors."""

    success: bool = Field(..., description="Whether request was successful")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    instances: List[UnrealInstanceInfo] = Field(
        default_factory=list,
        description=(
            "Discovered instances by port; a scan omits ports where nothing "
            "answered, except the active endpoint"
        ),
    )
    active_instance: Optional[str] = Field(
        None, description="Name of the instance Unreal tools target"
    )
    total_discovered: int = Field(0, description="Instances whose Remote Control answered")
    warnings: List[str] = Field(
        default_factory=list,
        description=(
            "Attached-mode problems (missing, invalid or stale attachment), and "
            "ports whose probe never left this machine (out of file descriptors "
            "or local ports, or a host that does not resolve), where an editor "
            "would be missing from instances"
        ),
    )


class UnrealHealthCheckResponse(BaseModel):
    """Response for Unreal Engine Remote Control API health check."""

    success: bool = Field(..., description="Whether request was successful")
    reachable: bool = Field(
        ...,
        description=(
            "Whether the Remote Control API responded, as ping_unreal decides "
            "it: a refusal (HTTP 401) is reachable, with error saying to set "
            "UNREAL__PASSPHRASE"
        ),
    )
    connected: bool = Field(
        ...,
        description=(
            "Deprecated, kept for one release: reachable and not refused "
            "(false on HTTP 401). Read reachable and error instead"
        ),
    )
    engine_version: Optional[str] = Field(
        None, description="Unreal Engine version string"
    )
    project_name: Optional[str] = Field(None, description="Active project name")
    is_editor: Optional[bool] = Field(
        None, description="Whether the engine is running in editor mode"
    )
    error: Optional[str] = Field(None, description="Error message when success is False")

    @model_validator(mode="before")
    @classmethod
    def _mirror_reachable_and_connected(cls, data: Any) -> Any:
        """Fill whichever of ``reachable``/``connected`` a payload left out.

        The adapter emits both; a caller that built the payload from the
        older shape still validates, and the reply always carries both names
        for the alias period.
        """
        if isinstance(data, dict):
            if "reachable" not in data and "connected" in data:
                data = {**data, "reachable": data["connected"]}
            elif "connected" not in data and "reachable" in data:
                data = {**data, "connected": data["reachable"]}
        return data


class UnrealEngineInfoResponse(BaseModel):
    """Response with Unreal Engine runtime information."""

    success: bool = Field(..., description="Whether request was successful")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    engine_version: str = Field(..., description="Unreal Engine version string")
    project_name: str = Field(..., description="Active project name")
    project_dir: str = Field(
        ..., description="Absolute project directory on the editor host"
    )
    loaded_map: str = Field(..., description="Currently loaded persistent level path")
    is_editor: bool = Field(..., description="Whether engine is running in editor mode")
    is_game: bool = Field(..., description="Whether engine is running in game mode")
    platform: str = Field(
        ...,
        description="Platform the editor runs on, as UGameplayStatics::GetPlatformName "
        "reports it (Mac, Windows, Linux)",
    )
    pie_active: bool = _pie_active_field()


class UnrealLoadedMapResponse(BaseModel):
    """Response with the currently loaded persistent level path."""

    success: bool = Field(..., description="Whether request was successful")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    map_path: str = Field(..., description="Currently loaded persistent level path")
    pie_active: bool = _pie_active_field()


# ---------------------------------------------------------------------------
# Unreal Engine schemas -- Phase 1: Scene Read Operations
# ---------------------------------------------------------------------------


class UnrealActorEntry(BaseModel):
    """Single actor entry in a listing response."""

    name: str = Field(..., description="Actor label/name")
    path: str = Field(..., description="Full object path in the level")
    class_name: str = Field(..., description="UClass name")
    location: Tuple[float, float, float] = Field(
        ..., description="World location (X, Y, Z) in cm"
    )
    rotation: Tuple[float, float, float] = Field(
        ..., description="Rotation (Pitch, Yaw, Roll) in degrees"
    )
    scale: Tuple[float, float, float] = Field(..., description="3D scale")
    tags: List[str] = Field(default_factory=list, description="Actor tags")


class UnrealListActorsResponse(BaseModel):
    """Response for actor listing."""

    success: bool = Field(..., description="Whether request was successful")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    actors: List[UnrealActorEntry] = Field(
        default_factory=list, description="Actors in the level"
    )
    count: int = Field(..., description="Number of actors returned")
    truncated: bool = Field(
        False, description="Whether results were truncated by max_results"
    )
    pie_active: bool = _pie_active_field()


class UnrealActorComponentInfo(BaseModel):
    """Information about a single actor component."""

    name: str = Field(..., description="Component name")
    class_name: str = Field(..., description="UClass name of the component")
    is_root: bool = Field(False, description="Whether this is the root component")


class UnrealGetActorInfoResponse(BaseModel):
    """Response with detailed actor information."""

    success: bool = Field(..., description="Whether request was successful")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    name: str = Field(..., description="Actor label")
    path: str = Field(..., description="Full object path")
    class_name: str = Field(..., description="UClass name")
    location: Tuple[float, float, float] = Field(
        ..., description="World location (X, Y, Z) in cm"
    )
    rotation: Tuple[float, float, float] = Field(
        ..., description="Rotation (Pitch, Yaw, Roll) in degrees"
    )
    scale: Tuple[float, float, float] = Field(..., description="3D scale")
    components: List[UnrealActorComponentInfo] = Field(
        default_factory=list, description="Attached components"
    )
    tags: List[str] = Field(default_factory=list, description="Actor tags")
    mobility: str = Field("Static", description="Mobility (Static/Stationary/Movable)")
    is_hidden: bool = Field(False, description="Whether actor is hidden in game")
    pie_active: bool = _pie_active_field()


class UnrealAssetEntry(BaseModel):
    """Single asset entry from the Asset Registry."""

    name: str = Field(..., description="Asset name")
    path: str = Field(..., description="Full asset path")
    class_name: str = Field(..., description="UClass name")
    package_path: str = Field(..., description="Package path")


class UnrealSearchAssetsResponse(BaseModel):
    """Response from asset search."""

    success: bool = Field(..., description="Whether request was successful")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    assets: List[UnrealAssetEntry] = Field(
        default_factory=list, description="Matching assets"
    )
    count: int = Field(..., description="Number of assets returned")
    truncated: bool = Field(False, description="Whether results were truncated")


class UnrealPropertyInfo(BaseModel):
    """Single property descriptor from UObject description."""

    name: str = Field(..., description="Property name")
    type: str = Field(..., description="Property type string")
    value: Optional[Any] = Field(None, description="Current value (if readable)")


class UnrealDescribeObjectResponse(BaseModel):
    """Response with UObject metadata."""

    success: bool = Field(..., description="Whether request was successful")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    object_path: str = Field(..., description="Object path queried")
    class_name: str = Field(..., description="UClass name")
    properties: List[UnrealPropertyInfo] = Field(
        default_factory=list, description="Object properties"
    )
    functions: List[str] = Field(
        default_factory=list, description="Callable function names"
    )


class UnrealGetThumbnailResponse(BaseModel):
    """Thumbnail result: an image content block plus this JSON record.

    ``image_base64`` is what the Unreal session hands the server; the server
    lifts it into an MCP ``ImageContent`` block and sets ``image_attached``,
    so it is left out of the schema.
    """

    success: bool = Field(..., description="Whether request was successful")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    asset_path: str = Field(..., description="Asset path queried")
    image_base64: SkipJsonSchema[str] = Field(
        ..., description="Session-side image data, lifted into the image content block"
    )
    image_attached: bool = Field(
        False,
        description="True when the image was sent as an MCP image content block",
    )
    format: str = Field("png", description="Image format: png or jpeg")
    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")


class UnrealSceneSummaryResponse(BaseModel):
    """LLM-friendly scene digest response."""

    success: bool = Field(..., description="Whether request was successful")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    map_path: str = Field(..., description="Loaded map path")
    total_actors: int = Field(..., description="Total actor count in the level")
    actor_class_counts: Dict[str, int] = Field(
        default_factory=dict, description="Actor count per UClass"
    )
    static_meshes: int = Field(0, description="Number of StaticMeshActors")
    lights: int = Field(0, description="Number of light actors")
    cameras: int = Field(0, description="Number of camera actors")
    summary_text: str = Field(
        "", description="Human-readable scene summary for LLM consumption"
    )
    pie_active: bool = _pie_active_field()


# ---------------------------------------------------------------------------
# Unreal Phase 2 -- Viewport & Visual Observation
# ---------------------------------------------------------------------------


class UnrealExecuteScriptResponse(BaseModel):
    """Whatever JSON object a script printed, with the envelope's status fields.

    The script decides the shape, so every key it emitted is kept.
    """

    model_config = ConfigDict(extra="allow")

    success: bool = Field(True, description="Whether the script ran and printed JSON")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )


class UnrealCaptureViewportResponse(BaseModel):
    """Viewport capture result: a path record, plus an image block when inline.

    With ``inline`` and a small enough capture, the client receives the image
    as an MCP ``ImageContent`` block ahead of this JSON record, which then
    carries ``image_attached: true``. ``image_base64``/``encoding`` are what
    the Unreal session hands the server; the server lifts them into the image
    block, so they are left out of the schema.
    """

    success: bool = Field(..., description="Whether request was successful")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    path: str = Field("", description="Capture path on the Unreal editor host")
    size_bytes: int = Field(0, description="Size of the capture file in bytes")
    image_base64: SkipJsonSchema[Optional[str]] = Field(
        None, description="Session-side image data, lifted into the image content block"
    )
    encoding: SkipJsonSchema[Optional[str]] = Field(
        None, description="Encoding of image_base64; dropped with it"
    )
    image_attached: bool = Field(
        False,
        description="True when the image was sent as an MCP image content block",
    )
    inline_skipped: Optional[str] = Field(
        None, description="Why inline data was omitted, when it was"
    )
    resolution_x: int = Field(..., description="Actual capture width")
    resolution_y: int = Field(..., description="Actual capture height")
    format: str = Field(..., description="Image format used")


class UnrealViewportInfoResponse(BaseModel):
    """Viewport camera and render information."""

    success: bool = Field(..., description="Whether request was successful")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    camera_location: Tuple[float, float, float] = Field(
        ..., description="Camera position (X, Y, Z) in cm"
    )
    camera_rotation: Tuple[float, float, float] = Field(
        ..., description="Camera rotation (Pitch, Yaw, Roll) in degrees"
    )
    viewport_size: Optional[Tuple[int, int]] = Field(
        None,
        description="Viewport dimensions (width, height); None when Remote Control does not expose them",
    )
    fov: Optional[float] = Field(
        None, description="Field of view in degrees; None when Remote Control does not expose it"
    )
    projection_type: Optional[str] = Field(
        None,
        description="Perspective or Orthographic; None when Remote Control does not expose it",
    )


class UnrealSetCameraViewResponse(BaseModel):
    """Response after setting camera view."""

    success: bool = Field(..., description="Whether request was successful")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    location: Optional[Tuple[float, float, float]] = Field(
        None, description="Applied camera position; None when left unchanged"
    )
    rotation: Optional[Tuple[float, float, float]] = Field(
        None, description="Applied camera rotation; None when left unchanged"
    )


class UnrealFocusActorResponse(BaseModel):
    """Response after focusing on actor."""

    success: bool = Field(..., description="Whether request was successful")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    actor_path: str = Field(..., description="Actor that was focused on")
    camera_location: Tuple[float, float, float] = Field(
        ..., description="Resulting camera position"
    )
    camera_rotation: Tuple[float, float, float] = Field(
        ..., description="Resulting camera rotation"
    )
    distance: float = Field(
        ...,
        description=(
            "Distance used, in cm: the camera stands this far behind the point it looks at and half "
            "as far above it. The distance passed, or for distance 0 or less the one that fits the actor's "
            "bounds in the view (500 for an actor without bounds)"
        ),
    )
    fit_warning: Optional[str] = Field(
        None,
        description=(
            "Why auto-fit (distance 0 or less) did not fit the actor: it has no bounds (zero extent), so the camera "
            "stands the default 500 cm from its location; null otherwise"
        ),
    )
    selected: bool = Field(
        ...,
        description=(
            "Whether the actor is selected in the editor, as Unreal reads its selection back; "
            "false while Play In Editor runs, when Unreal's editor selects nothing"
        ),
    )
    pie_active: bool = Field(
        False,
        description=(
            "Whether Play In Editor (or Simulate) was running; the camera moves then, "
            "the selection does not"
        ),
    )
    warning: Optional[str] = Field(
        None, description="Why the actor is not selected; null when it is"
    )


# ---------------------------------------------------------------------------
# Unreal Phase 3 -- Scene Manipulation
# ---------------------------------------------------------------------------


class UnrealSpawnActorResponse(BaseModel):
    """Response after spawning an actor."""

    success: bool = Field(..., description="Whether request was successful")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    actor_path: str = Field(
        ..., min_length=1, description="Full path of the spawned actor"
    )
    actor_class: str = Field(
        ...,
        description="Class of the spawned actor, e.g. StaticMeshActor for a mesh asset",
    )
    label: Optional[str] = Field(None, description="Actor label in the outliner")
    location: Tuple[float, float, float] = Field(
        ..., description="World location [x, y, z] in cm, read back from the new actor"
    )
    rotation: Optional[Tuple[float, float, float]] = Field(
        None,
        description="World rotation [pitch, yaw, roll] in degrees, read back from the new actor",
    )


class UnrealDeleteActorResponse(BaseModel):
    """Response after deleting an actor."""

    success: bool = Field(..., description="Whether request was successful")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    actor_path: str = Field(..., description="Path of the deleted actor")
    deleted: bool = Field(..., description="Whether actor was actually deleted")


class UnrealSetActorTransformResponse(BaseModel):
    """Response after setting actor transform."""

    success: bool = Field(..., description="Whether request was successful")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    actor_path: str = Field(..., description="Actor whose transform was set")
    location: Tuple[float, float, float] = Field(
        ..., description="World location [x, y, z] in cm, read back after the write"
    )
    rotation: Tuple[float, float, float] = Field(
        ...,
        description="World rotation [pitch, yaw, roll] in degrees, read back after the write",
    )
    scale: Tuple[float, float, float] = Field(
        ..., description="Scale [x, y, z], read back after the write"
    )


class UnrealSetActorPropertyResponse(BaseModel):
    """Response after setting actor property."""

    success: bool = Field(..., description="Whether request was successful")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    actor_path: str = Field(..., description="Actor whose property was set")
    property_name: str = Field(..., description="Property that was set")


class UnrealCallActorFunctionResponse(BaseModel):
    """Response after calling an actor function."""

    success: bool = Field(..., description="Whether request was successful")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    actor_path: str = Field(..., description="Actor on which function was called")
    function_name: str = Field(..., description="Function that was called")
    return_value: Optional[str] = Field(None, description="Return value as JSON string")


class UnrealSetActorParentResponse(BaseModel):
    """Response after setting actor parent."""

    success: bool = Field(..., description="Whether request was successful")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    actor_path: str = Field(..., description="Child actor path")
    parent_path: Optional[str] = Field(
        None, description="Parent actor path (None if detached)"
    )


class UnrealAddComponentResponse(BaseModel):
    """Response after adding a component."""

    success: bool = Field(..., description="Whether request was successful")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    actor_path: str = Field(..., description="Actor that received the component")
    component_path: str = Field(..., description="Full path of the new component")
    component_class: str = Field(..., description="Class of the added component")


class UnrealSetActorVisibilityResponse(BaseModel):
    """Response after setting actor visibility."""

    success: bool = Field(..., description="Whether request was successful")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    actor_path: str = Field(..., description="Actor whose visibility was set")
    visible: bool = Field(..., description="Applied visibility state")


# ---------------------------------------------------------------
# Unreal Phase 4 -- Materials, Lighting & Rendering
# ---------------------------------------------------------------


class UnrealMaterialParameterInfo(BaseModel):
    """Single material parameter entry."""

    name: str = Field(..., description="Parameter name")
    param_type: str = Field(..., description="scalar | vector | texture")
    association: str = Field(
        "global",
        description=(
            "global (the material's own parameter), layer (inside one of its Material Layers) or blend "
            "(inside a layer blend)"
        ),
    )
    index: Optional[int] = Field(
        None,
        description=(
            "The Material Layer or layer blend the parameter belongs to, as Unreal numbers them: layer 0 "
            "is the bottom layer, and blend i blends layer i + 1 onto the layers below; null for a global "
            "parameter"
        ),
    )
    value: Any = Field(
        None,
        description=(
            "A float (scalar), [r, g, b, a] (vector) or a texture asset path (texture; null for none): "
            "a Material's default, or the value an instance renders with, its override or what it "
            "inherits; a layer or blend parameter's for that layer or blend"
        ),
    )
    overridden: Optional[bool] = Field(
        None, description="Whether the instance overrides this parameter; null for a Material"
    )


class UnrealGetMaterialInfoResponse(BaseModel):
    """Response with the parameters and metadata of a Material or Material Instance."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    material_path: str = Field(..., description="Object path of the material asset")
    class_name: Optional[str] = Field(
        None, description="Material, MaterialInstanceConstant or another material class"
    )
    parent_path: Optional[str] = Field(
        None, description="The instance's parent material; null for a Material"
    )
    base_material_path: Optional[str] = Field(
        None, description="The Material at the root of the parent chain (the material itself for a Material)"
    )
    material_domain: Optional[str] = Field(
        None, description="The base material's domain, as Unreal names it (MD_SURFACE, MD_POST_PROCESS, ...)"
    )
    blend_mode: Optional[str] = Field(
        None,
        description=(
            "Blend mode as Unreal names it (BLEND_OPAQUE, BLEND_MASKED, BLEND_TRANSLUCENT, ...): an "
            "instance's override, else its parent's; BLEND_OPAQUE for an instance without a parent"
        ),
    )
    parameters: List[UnrealMaterialParameterInfo] = Field(
        default_factory=list,
        description=(
            "Scalar, vector and texture parameters, by kind; within a kind the material's own (global) "
            "parameters, then each Material Layer's and each blend's in index order, each by name. A name "
            "the material and its layers share is listed once for each"
        ),
    )


class UnrealSetMaterialParamsResponse(BaseModel):
    """Response after setting material instance parameters."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    material_path: str = Field(..., description="Material Instance path")
    params_set: int = Field(0, description="Number of parameters that read back as requested")
    applied: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Values read back after the write, by kind (scalar, vector, texture) and parameter name: "
            "a float, [r, g, b, a], or a texture asset path"
        ),
    )
    not_applied: Dict[str, Dict[str, str]] = Field(
        default_factory=dict,
        description=(
            "By kind and parameter name, why a requested parameter was not set: refused before anything "
            "was written (the material has no global parameter of that kind by that name, or only its "
            "Material Layers have one; the texture does not load; a malformed value), or read back "
            "different after the write. Empty when the call succeeds"
        ),
    )


class UnrealCreateMaterialInstanceResponse(BaseModel):
    """Response with newly created material instance path."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    instance_path: str = Field(..., description="New MIC asset path")
    parent_path: str = Field(..., description="Parent material path")


class UnrealAssignMaterialResponse(BaseModel):
    """Response after assigning a material."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    actor_path: str = Field(..., description="Actor path")
    material_path: str = Field(..., description="Assigned material path")
    slot_index: int = Field(0, description="Slot index")


class UnrealSetLightParamsResponse(BaseModel):
    """Response after setting light parameters."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    actor_path: str = Field(..., description="Light actor path")
    component_path: Optional[str] = Field(
        None, description="Light component the parameters were written to"
    )
    params_set: int = Field(0, description="Number of params changed")
    applied: Optional[Dict[str, Any]] = Field(
        None, description="Values read back from the light component after the write"
    )


class UnrealSetRenderSettingsResponse(BaseModel):
    """Response after changing render settings."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    setting_name: str = Field(..., description="Setting name")
    applied: bool = Field(True, description="Whether the setting was applied")


# ------------------------------------------------------------------
# Unreal Phase 5: Physics & Simulation Control
# ------------------------------------------------------------------


class UnrealControlSimulationResponse(BaseModel):
    """Response after controlling a PIE session."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    action: str = Field(..., description="Action that was executed")
    state: str = Field(
        ...,
        description="Resulting PIE state: playing, paused, or stopped",
    )


class UnrealGetSimulationStatusResponse(BaseModel):
    """Response with current PIE simulation status."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    is_playing: bool = Field(False, description="Whether PIE is running")
    is_paused: bool = Field(False, description="Whether PIE is paused")
    frame_count: Optional[int] = Field(
        None, description="Simulated frame count, unavailable when null"
    )
    sim_time: float = Field(0.0, description="Elapsed simulation time in seconds")


class UnrealEnablePhysicsResponse(BaseModel):
    """Response after toggling physics on an actor."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    actor_path: str = Field(..., description="Actor that was modified")
    physics_enabled: bool = Field(..., description="Current physics state")


class UnrealSetCollisionResponse(BaseModel):
    """Response after setting collision configuration."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    actor_path: str = Field(..., description="Actor that was modified")
    collision_preset: str = Field(
        ..., description="Collision preset read back after the write (Custom once overridden)"
    )
    collision_enabled: bool = Field(
        ..., description="Whether the component has any collision, read back after the write"
    )


class UnrealApplyForceResponse(BaseModel):
    """Response after applying a force or impulse."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    actor_path: str = Field(..., description="Actor the force was applied to")
    force_applied: bool = Field(True, description="Whether force was applied")
    force_vector: List[float] = Field(..., description="Applied force vector [x, y, z]")
    is_impulse: bool = Field(..., description="Whether an impulse was applied")


class UnrealSetPhysicsParamsResponse(BaseModel):
    """Response after setting physics parameters."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    actor_path: str = Field(..., description="Actor that was modified")
    params_set: int = Field(..., description="Number of parameters set")


# ----------------------------------------------------------------------
# Phase 6: USD / SimReady Bridge
# ----------------------------------------------------------------------


class UnrealImportUsdResponse(BaseModel):
    """Response after importing a USD file."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    imported_assets: List[str] = Field(..., description="List of imported asset paths")
    actor_paths: List[str] = Field(
        default_factory=list, description="Spawned actor paths in the level"
    )
    warnings: List[str] = Field(default_factory=list, description="Import warnings")


class UnrealExportUsdResponse(BaseModel):
    """Response after exporting actors to USD."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    output_path: str = Field(..., description="Written USD file path")
    actors_exported: int = Field(..., description="Number of actors exported")
    file_size_bytes: int = Field(0, description="Output file size in bytes")


class UnrealConvertToSimreadyResponse(BaseModel):
    """Response after SimReady conversion."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    output_path: str = Field(..., description="SimReady USD path")
    conversions_applied: List[str] = Field(
        ..., description="List of conversions applied"
    )
    warnings: List[str] = Field(default_factory=list, description="Conversion warnings")


class UnrealValidateSimreadyResponse(BaseModel):
    """Response after SimReady validation."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    usd_path: str = Field(..., description="Validated USD path")
    is_valid: bool = Field(..., description="Overall validation result")
    checks: Dict[str, bool] = Field(..., description="Per-check pass/fail results")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    suggestions: List[str] = Field(default_factory=list, description="Fix suggestions")


class UnrealGetInterchangeInfoResponse(BaseModel):
    """Response with Interchange Framework info."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    pipelines: List[Dict[str, Any]] = Field(
        ..., description="Available import/export pipelines"
    )
    supported_formats: List[str] = Field(..., description="Supported file formats")
    interchange_version: str = Field(..., description="Interchange Framework version")
    pipeline_enumeration_available: bool = Field(
        False, description="Whether pipeline enumeration is available"
    )
    usd_import_available: bool = Field(
        False, description="USD import plugin available in the editor"
    )
    usd_export_available: bool = Field(
        False, description="USD exporter available in the editor"
    )
    simready_available: bool = Field(
        False, description="SimReady conversion/validation implemented"
    )


# ----------------------------------------------------------------------
# Phase 7: Advanced Agent Tools
# ----------------------------------------------------------------------


class UnrealBatchOperationsResponse(BaseModel):
    """Response after batch execution."""

    success: bool = Field(True, description="Overall success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    results: List[Dict[str, Any]] = Field(..., description="Per-operation results")
    total: int = Field(..., description="Total operations submitted")
    succeeded: int = Field(..., description="Number that succeeded")
    failed: int = Field(0, description="Number that failed")


class UnrealQuerySceneGraphResponse(BaseModel):
    """Response with scene graph tree."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    root: Dict[str, Any] = Field(
        ..., description="Scene graph tree (nested dicts with children)"
    )
    total_actors: int = Field(..., description="Total actors in graph")
    total_depth: int = Field(..., description="Deepest nesting level")


class UnrealAnalyzeSceneForRoboticsResponse(BaseModel):
    """Response with robotics scene analysis."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    traversable_surfaces: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Floor-like surfaces a robot can stand on: name, path, class_name, "
            "size [x, y, z] cm, top_z cm, area_m2, tilt_degrees"
        ),
    )
    graspable_objects: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Movable actors with physics collision small enough to grasp: name, "
            "path, class_name, size [x, y, z] cm, simulate_physics, mass_kg"
        ),
    )
    collision_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Collision counts and static mesh collision setup; empty unless collision_complexity ran. "
            "While Play In Editor runs the static mesh collision counts and "
            "missing_simple_collision_actors are null"
        ),
    )
    total_actors_analyzed: int = Field(0, description="Number of actors analyzed")
    analysis_types: List[str] = Field(
        default_factory=list, description="Analyses that ran"
    )
    criteria: Dict[str, float] = Field(
        default_factory=dict, description="Thresholds the analyses applied"
    )
    pie_active: bool = _pie_active_field()


class UnrealGenerateProceduralSceneResponse(BaseModel):
    """Response after procedural scene generation."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    actors_spawned: List[str] = Field(..., description="Paths of spawned actors")
    total_spawned: int = Field(..., description="Number of actors spawned")
    scene_type: str = Field(..., description="Generated scene type")
    seed: int = Field(..., description="Random seed used")
    tag: str = Field(
        ...,
        description="Actor tag on every actor this call spawned, simul_<scene_type>_<seed>_<generation>; "
        "unique to the call",
    )
    folder: str = Field(
        ...,
        description="Outliner folder this call filed its actors under, Simul/<Scene>_<seed>_<generation>; "
        "unique to the call",
    )


class UnrealGetActorBySemanticLabelResponse(BaseModel):
    """Response with actors matching semantic label."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    actors: List[UnrealActorEntry] = Field(
        ..., description="Matching actors with their paths, transforms and tags"
    )
    total_matches: int = Field(
        ..., description="Number of matching actors, including any past max_results"
    )
    truncated: bool = Field(
        False, description="Whether matches were left out by max_results"
    )
    label_searched: str = Field(..., description="Label that was searched")
    pie_active: bool = _pie_active_field()


# ----------------------------------------------------------------------
# Phase 8: Geometry & Modeling
# ----------------------------------------------------------------------


class UnrealGenerateMeshPrimitiveResponse(BaseModel):
    """Response after creating a mesh primitive."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    actor_path: str = Field(..., description="Created actor path")
    primitive_type: str = Field(..., description="Primitive type created")
    triangle_count: int = Field(..., description="Number of triangles")
    vertex_count: int = Field(..., description="Number of vertices")


class UnrealApplyMeshBooleanResponse(BaseModel):
    """Response after boolean operation."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    target_mesh_path: str = Field(..., description="Modified mesh path")
    operation: str = Field(..., description="Operation performed")
    result_triangle_count: int = Field(
        ..., description="Triangle count after operation"
    )
    result_vertex_count: int = Field(..., description="Vertex count after operation")


class UnrealComputeConvexHullResponse(BaseModel):
    """Response after convex hull computation."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    mesh_path: str = Field(..., description="Source mesh path")
    hull_actor_path: str = Field(..., description="Created convex hull actor path")
    hull_vertex_count: int = Field(..., description="Hull vertex count")
    hull_triangle_count: int = Field(..., description="Hull triangle count")
    volume_ratio: float = Field(..., description="Hull volume / original volume ratio")


class UnrealDecomposeConvexHullResponse(BaseModel):
    """Response after V-HACD decomposition."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    mesh_path: str = Field(..., description="Source mesh path")
    hull_count: int = Field(..., description="Number of convex hulls generated")
    hulls: List[Dict[str, Any]] = Field(
        ..., description="Per-hull info (path, vertex_count, volume)"
    )
    total_vertices: int = Field(..., description="Total vertices across all hulls")


class UnrealEditMeshTopologyResponse(BaseModel):
    """Response after topology edit."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    mesh_path: str = Field(..., description="Modified mesh path")
    operation: str = Field(..., description="Operation performed")
    faces_affected: int = Field(0, description="Faces affected")
    edges_affected: int = Field(0, description="Edges affected")
    result_triangle_count: int = Field(..., description="Triangle count after edit")


class UnrealSubdivideMeshResponse(BaseModel):
    """Response after mesh subdivision."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    mesh_path: str = Field(..., description="Subdivided mesh path")
    level: int = Field(..., description="Subdivision level applied")
    scheme: str = Field(..., description="Scheme used")
    result_triangle_count: int = Field(
        ..., description="Triangle count after subdivision"
    )
    result_vertex_count: int = Field(..., description="Vertex count after subdivision")


class UnrealSimplifyMeshResponse(BaseModel):
    """Response after mesh simplification."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    mesh_path: str = Field(..., description="Simplified mesh path")
    original_triangles: int = Field(..., description="Original triangle count")
    result_triangles: int = Field(
        ..., description="Triangle count after simplification"
    )
    reduction_ratio: float = Field(..., description="Reduction ratio achieved")


class UnrealCutMeshPlaneResponse(BaseModel):
    """Response after plane cut."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    mesh_path: str = Field(..., description="Cut mesh path")
    pieces: List[str] = Field(..., description="Resulting piece actor paths")
    cut_faces_added: int = Field(0, description="Number of fill faces added")


class UnrealValidateMeshResponse(BaseModel):
    """Response after mesh validation."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    mesh_path: str = Field(..., description="Validated mesh path")
    is_valid: bool = Field(..., description="Overall validation result")
    checks: Dict[str, bool] = Field(..., description="Per-check pass/fail results")
    issues: List[str] = Field(default_factory=list, description="Detected issues")
    triangle_count: int = Field(..., description="Current triangle count")
    vertex_count: int = Field(..., description="Current vertex count")


class UnrealConvertMeshFormatResponse(BaseModel):
    """Response after mesh format conversion."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    source_path: str = Field(..., description="Source mesh path")
    result_path: str = Field(..., description="Converted mesh path")
    source_format: str = Field(..., description="Original format")
    target_format: str = Field(..., description="Converted format")
    triangle_count: int = Field(..., description="Result triangle count")


class UnrealRemeshMeshResponse(BaseModel):
    """Response after remeshing."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    mesh_path: str = Field(..., description="Remeshed mesh path")
    mode: str = Field(..., description="Remesh mode used")
    original_triangles: int = Field(..., description="Original triangle count")
    result_triangles: int = Field(..., description="Triangle count after remesh")
    average_edge_length: float = Field(
        ..., description="Average edge length after remesh"
    )


class UnrealComputeMeshUvResponse(BaseModel):
    """Response after UV computation."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    mesh_path: str = Field(..., description="Mesh path")
    method: str = Field(..., description="UV method used")
    uv_channel: int = Field(..., description="UV channel written")
    island_count: int = Field(..., description="Number of UV islands")
    coverage_ratio: float = Field(..., description="UV space coverage (0.0-1.0)")
    overlap_detected: bool = Field(
        False, description="Whether UV overlaps were detected"
    )


__all__ = [
    "Vector3",
    "ColorRGB",
    "UnrealExecuteScriptResponse",
    "UnrealPingResponse",
    "UnrealInstanceInfo",
    "UnrealListInstancesResponse",
    "UnrealHealthCheckResponse",
    "UnrealEngineInfoResponse",
    "UnrealLoadedMapResponse",
    "UnrealActorEntry",
    "UnrealListActorsResponse",
    "UnrealActorComponentInfo",
    "UnrealGetActorInfoResponse",
    "UnrealAssetEntry",
    "UnrealSearchAssetsResponse",
    "UnrealPropertyInfo",
    "UnrealDescribeObjectResponse",
    "UnrealGetThumbnailResponse",
    "UnrealSceneSummaryResponse",
    "UnrealCaptureViewportResponse",
    "UnrealViewportInfoResponse",
    "UnrealSetCameraViewResponse",
    "UnrealFocusActorResponse",
    "UnrealSpawnActorResponse",
    "UnrealDeleteActorResponse",
    "UnrealSetActorTransformResponse",
    "UnrealSetActorPropertyResponse",
    "UnrealCallActorFunctionResponse",
    "UnrealSetActorParentResponse",
    "UnrealAddComponentResponse",
    "UnrealSetActorVisibilityResponse",
    "UnrealMaterialParameterInfo",
    "UnrealGetMaterialInfoResponse",
    "UnrealSetMaterialParamsResponse",
    "UnrealCreateMaterialInstanceResponse",
    "UnrealAssignMaterialResponse",
    "UnrealSetLightParamsResponse",
    "UnrealSetRenderSettingsResponse",
    "UnrealControlSimulationResponse",
    "UnrealGetSimulationStatusResponse",
    "UnrealEnablePhysicsResponse",
    "UnrealSetCollisionResponse",
    "UnrealApplyForceResponse",
    "UnrealSetPhysicsParamsResponse",
    "UnrealImportUsdResponse",
    "UnrealExportUsdResponse",
    "UnrealConvertToSimreadyResponse",
    "UnrealValidateSimreadyResponse",
    "UnrealGetInterchangeInfoResponse",
    "UnrealBatchOperationsResponse",
    "UnrealQuerySceneGraphResponse",
    "UnrealAnalyzeSceneForRoboticsResponse",
    "UnrealGenerateProceduralSceneResponse",
    "UnrealGetActorBySemanticLabelResponse",
    "UnrealGenerateMeshPrimitiveResponse",
    "UnrealApplyMeshBooleanResponse",
    "UnrealComputeConvexHullResponse",
    "UnrealDecomposeConvexHullResponse",
    "UnrealEditMeshTopologyResponse",
    "UnrealSubdivideMeshResponse",
    "UnrealSimplifyMeshResponse",
    "UnrealCutMeshPlaneResponse",
    "UnrealValidateMeshResponse",
    "UnrealConvertMeshFormatResponse",
    "UnrealRemeshMeshResponse",
    "UnrealComputeMeshUvResponse",
]
