"""Unreal Engine MCP schemas."""

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Unreal Engine schemas -- Phase 0
# ---------------------------------------------------------------------------


class UnrealPingResponse(BaseModel):
    """Response for a lightweight Unreal Engine reachability probe."""

    success: bool = Field(..., description="Whether request was successful")
    reachable: bool = Field(..., description="Whether the Remote Control API responded")
    address: str = Field(..., description="host:port that was probed")
    latency_ms: Optional[float] = Field(None, description="Round-trip time in milliseconds")
    error: Optional[str] = Field(None, description="Error message when success is False")


class UnrealInstanceInfo(BaseModel):
    """Information about a single discovered Unreal Engine instance."""

    name: str = Field(..., description="Instance identifier")
    host: str = Field(..., description="Remote Control API host")
    port: int = Field(..., description="Remote Control API port")
    reachable: bool = Field(..., description="Whether the instance responded to ping")
    active: bool = Field(False, description="Whether this is the currently active instance")
    engine_version: Optional[str] = Field(None, description="Unreal Engine version")
    project_name: Optional[str] = Field(None, description="Active project name")
    loaded_map: Optional[str] = Field(None, description="Currently loaded level path")
    latency_ms: Optional[float] = Field(None, description="Ping latency in milliseconds")


class UnrealListInstancesResponse(BaseModel):
    """Response listing all discovered Unreal Engine instances."""

    success: bool = Field(..., description="Whether request was successful")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    instances: List[UnrealInstanceInfo] = Field(
        default_factory=list, description="Discovered instances"
    )
    active_instance: Optional[str] = Field(
        None, description="Name of the currently active instance"
    )
    total_discovered: int = Field(0, description="Total instances found via port scan")


class UnrealHealthCheckResponse(BaseModel):
    """Response for Unreal Engine Remote Control API health check."""

    success: bool = Field(..., description="Whether request was successful")
    connected: bool = Field(..., description="Whether the engine is reachable")
    engine_version: Optional[str] = Field(
        None, description="Unreal Engine version string"
    )
    project_name: Optional[str] = Field(None, description="Active project name")
    is_editor: Optional[bool] = Field(
        None, description="Whether the engine is running in editor mode"
    )
    error: Optional[str] = Field(None, description="Error message when success is False")


class UnrealEngineInfoResponse(BaseModel):
    """Response with Unreal Engine runtime information."""

    success: bool = Field(..., description="Whether request was successful")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    engine_version: str = Field(..., description="Unreal Engine version string")
    project_name: str = Field(..., description="Active project name")
    loaded_map: str = Field(..., description="Currently loaded persistent level path")
    is_editor: bool = Field(..., description="Whether engine is running in editor mode")
    is_game: bool = Field(..., description="Whether engine is running in game mode")
    platform: str = Field(..., description="Platform identifier")


class UnrealLoadedMapResponse(BaseModel):
    """Response with the currently loaded persistent level path."""

    success: bool = Field(..., description="Whether request was successful")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    map_path: str = Field(..., description="Currently loaded persistent level path")


# ---------------------------------------------------------------------------
# Unreal Engine schemas -- Phase 1: Scene Read Operations
# ---------------------------------------------------------------------------


class UnrealListActorsRequest(BaseModel):
    """Request for listing actors in the current Unreal level."""

    class_filter: Optional[str] = Field(
        None, description="Filter actors by UClass name (e.g. 'StaticMeshActor')"
    )
    tag_filter: Optional[str] = Field(None, description="Filter actors by tag")
    max_results: int = Field(
        200, description="Maximum number of actors to return", ge=1, le=5000
    )


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


class UnrealGetActorInfoRequest(BaseModel):
    """Request for detailed actor information."""

    actor_path: str = Field(..., description="Full object path of the actor")


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


class UnrealSearchAssetsRequest(BaseModel):
    """Request to search the Unreal Asset Registry."""

    query: str = Field("", description="Search query string")
    class_names: Optional[List[str]] = Field(
        None, description="Filter by UClass names (e.g. ['StaticMesh', 'Material'])"
    )
    package_paths: Optional[List[str]] = Field(
        None, description="Package paths to search within (e.g. ['/Game/Meshes'])"
    )
    max_results: int = Field(
        100, description="Maximum number of results", ge=1, le=1000
    )


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


class UnrealDescribeObjectRequest(BaseModel):
    """Request to describe a UObject's properties and functions."""

    object_path: str = Field(
        ...,
        description="Full object path (e.g. '/Game/Maps/Test.Test:PersistentLevel.StaticMeshActor_0')",
    )


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


class UnrealGetThumbnailRequest(BaseModel):
    """Request for an asset thumbnail image."""

    asset_path: str = Field(
        ..., description="Full asset path (e.g. '/Game/Meshes/SM_Chair')"
    )
    width: int = Field(256, description="Thumbnail width in pixels", ge=32, le=1024)
    height: int = Field(256, description="Thumbnail height in pixels", ge=32, le=1024)


class UnrealGetThumbnailResponse(BaseModel):
    """Response with a base64-encoded thumbnail image."""

    success: bool = Field(..., description="Whether request was successful")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    asset_path: str = Field(..., description="Asset path queried")
    image_base64: str = Field(..., description="Base64-encoded PNG image")
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


# ---------------------------------------------------------------------------
# Unreal Phase 2 -- Viewport & Visual Observation
# ---------------------------------------------------------------------------


class UnrealCaptureViewportRequest(BaseModel):
    """Request to capture viewport screenshot."""

    resolution_x: int = Field(1920, description="Capture width in pixels")
    resolution_y: int = Field(1080, description="Capture height in pixels")
    format: str = Field("png", description="Image format: png or jpeg")


class UnrealCaptureViewportResponse(BaseModel):
    """Response with captured viewport image."""

    success: bool = Field(..., description="Whether request was successful")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    path: str = Field("", description="Capture path on the Unreal editor host")
    size_bytes: int = Field(0, description="Size of the capture file in bytes")
    image_base64: Optional[str] = Field(
        None, description="Base64 image data; only present for small inline captures"
    )
    encoding: Optional[str] = Field(None, description="Encoding of image_base64")
    inline_skipped: Optional[str] = Field(
        None, description="Why inline data was omitted, when it was"
    )
    resolution_x: int = Field(..., description="Actual capture width")
    resolution_y: int = Field(..., description="Actual capture height")
    format: str = Field(..., description="Image format used")


class UnrealViewportInfoRequest(BaseModel):
    """Request to get viewport information."""

    pass


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
    viewport_size: Tuple[int, int] = Field(
        ..., description="Viewport dimensions (width, height)"
    )
    fov: float = Field(90.0, description="Field of view in degrees")
    projection_type: str = Field(
        "Perspective", description="Perspective or Orthographic"
    )


class UnrealSetCameraViewRequest(BaseModel):
    """Request to set editor viewport camera."""

    location: Optional[Tuple[float, float, float]] = Field(
        None, description="Camera position (X, Y, Z) in cm"
    )
    rotation: Optional[Tuple[float, float, float]] = Field(
        None, description="Camera rotation (Pitch, Yaw, Roll) in degrees"
    )
    fov: Optional[float] = Field(None, description="Field of view in degrees")


class UnrealSetCameraViewResponse(BaseModel):
    """Response after setting camera view."""

    success: bool = Field(..., description="Whether request was successful")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    location: Tuple[float, float, float] = Field(
        ..., description="Applied camera position"
    )
    rotation: Tuple[float, float, float] = Field(
        ..., description="Applied camera rotation"
    )
    fov: float = Field(90.0, description="Applied field of view")


class UnrealFocusActorRequest(BaseModel):
    """Request to focus camera on an actor."""

    actor_path: str = Field(..., description="Full actor path to focus on")
    distance: float = Field(
        0.0, description="Camera distance from actor (0 = auto-fit)"
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


# ---------------------------------------------------------------------------
# Unreal Phase 3 -- Scene Manipulation
# ---------------------------------------------------------------------------


class UnrealSpawnActorRequest(BaseModel):
    """Request to spawn an actor from a class or asset path."""

    asset_path: str = Field(..., description="Asset or class path to spawn from")
    location: Tuple[float, float, float] = Field(
        (0.0, 0.0, 0.0), description="Spawn location (X, Y, Z) in cm"
    )
    rotation: Tuple[float, float, float] = Field(
        (0.0, 0.0, 0.0), description="Spawn rotation (Pitch, Yaw, Roll) in degrees"
    )
    label: Optional[str] = Field(None, description="Actor label in the outliner")


class UnrealSpawnActorResponse(BaseModel):
    """Response after spawning an actor."""

    success: bool = Field(..., description="Whether request was successful")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    actor_path: str = Field(..., description="Full path of the spawned actor")
    actor_class: str = Field(..., description="Class of the spawned actor")
    location: Tuple[float, float, float] = Field(
        ..., description="Actual spawn location"
    )


class UnrealDeleteActorRequest(BaseModel):
    """Request to delete an actor from the level."""

    actor_path: str = Field(..., description="Full actor path to delete")


class UnrealDeleteActorResponse(BaseModel):
    """Response after deleting an actor."""

    success: bool = Field(..., description="Whether request was successful")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    actor_path: str = Field(..., description="Path of the deleted actor")
    deleted: bool = Field(..., description="Whether actor was actually deleted")


class UnrealSetActorTransformRequest(BaseModel):
    """Request to set an actor's transform."""

    actor_path: str = Field(..., description="Full actor path")
    location: Optional[Tuple[float, float, float]] = Field(
        None, description="New location (X, Y, Z) in cm"
    )
    rotation: Optional[Tuple[float, float, float]] = Field(
        None, description="New rotation (Pitch, Yaw, Roll) in degrees"
    )
    scale: Optional[Tuple[float, float, float]] = Field(
        None, description="New scale (X, Y, Z)"
    )


class UnrealSetActorTransformResponse(BaseModel):
    """Response after setting actor transform."""

    success: bool = Field(..., description="Whether request was successful")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    actor_path: str = Field(..., description="Actor whose transform was set")
    location: Tuple[float, float, float] = Field(..., description="Applied location")
    rotation: Tuple[float, float, float] = Field(..., description="Applied rotation")
    scale: Tuple[float, float, float] = Field(..., description="Applied scale")


class UnrealSetActorPropertyRequest(BaseModel):
    """Request to set a property on an actor."""

    actor_path: str = Field(..., description="Full actor path")
    property_name: str = Field(..., description="Property name to set")
    property_value: str = Field(..., description="Property value as JSON string")
    generate_transaction: bool = Field(
        True, description="Whether to generate an undo transaction"
    )


class UnrealSetActorPropertyResponse(BaseModel):
    """Response after setting actor property."""

    success: bool = Field(..., description="Whether request was successful")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    actor_path: str = Field(..., description="Actor whose property was set")
    property_name: str = Field(..., description="Property that was set")


class UnrealCallActorFunctionRequest(BaseModel):
    """Request to call a BlueprintCallable function on an actor."""

    actor_path: str = Field(..., description="Full actor path")
    function_name: str = Field(..., description="Function name to call")
    parameters: Optional[str] = Field(None, description="Parameters as JSON string")


class UnrealCallActorFunctionResponse(BaseModel):
    """Response after calling an actor function."""

    success: bool = Field(..., description="Whether request was successful")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    actor_path: str = Field(..., description="Actor on which function was called")
    function_name: str = Field(..., description="Function that was called")
    return_value: Optional[str] = Field(None, description="Return value as JSON string")


class UnrealSetActorParentRequest(BaseModel):
    """Request to attach an actor to a parent actor."""

    actor_path: str = Field(..., description="Child actor path")
    parent_path: Optional[str] = Field(
        None, description="Parent actor path (None to detach)"
    )


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


class UnrealAddComponentRequest(BaseModel):
    """Request to add a component to an actor."""

    actor_path: str = Field(..., description="Full actor path")
    component_class: str = Field(
        ..., description="Component class name (e.g. StaticMeshComponent)"
    )
    component_name: Optional[str] = Field(
        None, description="Name for the new component"
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


class UnrealSetActorVisibilityRequest(BaseModel):
    """Request to set actor visibility."""

    actor_path: str = Field(..., description="Full actor path")
    visible: bool = Field(..., description="Whether the actor should be visible")
    propagate: bool = Field(True, description="Propagate to child components/actors")


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


class UnrealGetMaterialInfoRequest(BaseModel):
    """Request to get material instance info."""

    material_path: str = Field(..., description="Material or MIC asset path")


class UnrealMaterialParameterInfo(BaseModel):
    """Single material parameter entry."""

    name: str = Field(..., description="Parameter name")
    param_type: str = Field(..., description="scalar | vector | texture")
    value: Any = Field(None, description="Current value")


class UnrealGetMaterialInfoResponse(BaseModel):
    """Response with material instance parameters."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    material_path: str = Field(..., description="Material asset path")
    parent_path: Optional[str] = Field(None, description="Parent material path")
    parameters: List[UnrealMaterialParameterInfo] = Field(
        default_factory=list, description="Material parameters"
    )


class UnrealSetMaterialParamsRequest(BaseModel):
    """Request to set material instance parameters."""

    material_path: str = Field(..., description="Material Instance path")
    scalar_params: Optional[Dict[str, float]] = Field(
        None, description="Scalar parameter overrides"
    )
    vector_params: Optional[Dict[str, List[float]]] = Field(
        None, description="Vector parameter overrides (RGBA lists)"
    )
    texture_params: Optional[Dict[str, str]] = Field(
        None, description="Texture parameter overrides (asset paths)"
    )


class UnrealSetMaterialParamsResponse(BaseModel):
    """Response after setting material instance parameters."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    material_path: str = Field(..., description="Material Instance path")
    params_set: int = Field(0, description="Number of parameters set")


class UnrealCreateMaterialInstanceRequest(BaseModel):
    """Request to create a dynamic material instance."""

    parent_path: str = Field(..., description="Parent material asset path")
    instance_name: str = Field(..., description="New instance name")
    save_path: str = Field(
        "", description="Content-relative save path (auto-generated if empty)"
    )


class UnrealCreateMaterialInstanceResponse(BaseModel):
    """Response with newly created material instance path."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    instance_path: str = Field(..., description="New MIC asset path")
    parent_path: str = Field(..., description="Parent material path")


class UnrealAssignMaterialRequest(BaseModel):
    """Request to assign a material to a mesh component."""

    actor_path: str = Field(..., description="Target actor path")
    material_path: str = Field(..., description="Material asset path to assign")
    slot_index: int = Field(0, description="Material slot index")


class UnrealAssignMaterialResponse(BaseModel):
    """Response after assigning a material."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    actor_path: str = Field(..., description="Actor path")
    material_path: str = Field(..., description="Assigned material path")
    slot_index: int = Field(0, description="Slot index")


class UnrealSetLightParamsRequest(BaseModel):
    """Request to set light component parameters."""

    actor_path: str = Field(..., description="Light actor path")
    intensity: Optional[float] = Field(None, description="Light intensity")
    color_r: Optional[float] = Field(None, description="Color red (0-1)")
    color_g: Optional[float] = Field(None, description="Color green (0-1)")
    color_b: Optional[float] = Field(None, description="Color blue (0-1)")
    temperature: Optional[float] = Field(
        None, description="Color temperature in Kelvin"
    )
    use_temperature: Optional[bool] = Field(
        None, description="Use color temperature instead of color"
    )
    attenuation_radius: Optional[float] = Field(
        None, description="Attenuation radius in cm"
    )
    cast_shadows: Optional[bool] = Field(None, description="Enable shadow casting")


class UnrealSetLightParamsResponse(BaseModel):
    """Response after setting light parameters."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    actor_path: str = Field(..., description="Light actor path")
    params_set: int = Field(0, description="Number of params changed")


class UnrealSetRenderSettingsRequest(BaseModel):
    """Request to set rendering/post-process settings."""

    setting_name: str = Field(..., description="Render setting name")
    setting_value: str = Field(..., description="Value as JSON string")


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


class UnrealControlSimulationRequest(BaseModel):
    """Request to control a Play-In-Editor (PIE) session."""

    action: str = Field(
        ...,
        description="PIE action: start, stop, pause, resume, or step",
    )


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


class UnrealGetSimulationStatusRequest(BaseModel):
    """Request to query current PIE simulation status."""

    pass


class UnrealGetSimulationStatusResponse(BaseModel):
    """Response with current PIE simulation status."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    is_playing: bool = Field(False, description="Whether PIE is running")
    is_paused: bool = Field(False, description="Whether PIE is paused")
    frame_count: int = Field(0, description="Number of simulated frames")
    sim_time: float = Field(0.0, description="Elapsed simulation time in seconds")


class UnrealEnablePhysicsRequest(BaseModel):
    """Request to enable or disable physics simulation on an actor."""

    actor_path: str = Field(..., description="Full actor object path")
    enable: bool = Field(True, description="Enable or disable physics")
    simulate_physics: bool = Field(
        True, description="Whether the body should actively simulate"
    )


class UnrealEnablePhysicsResponse(BaseModel):
    """Response after toggling physics on an actor."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    actor_path: str = Field(..., description="Actor that was modified")
    physics_enabled: bool = Field(..., description="Current physics state")


class UnrealSetCollisionRequest(BaseModel):
    """Request to configure collision settings on an actor."""

    actor_path: str = Field(..., description="Full actor object path")
    collision_preset: str = Field(
        "",
        description="Named collision preset (e.g. BlockAll, NoCollision, PhysicsActor)",
    )
    collision_enabled: bool = Field(True, description="Enable collision")


class UnrealSetCollisionResponse(BaseModel):
    """Response after setting collision configuration."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    actor_path: str = Field(..., description="Actor that was modified")
    collision_preset: str = Field(..., description="Applied collision preset")
    collision_enabled: bool = Field(..., description="Current collision state")


class UnrealApplyForceRequest(BaseModel):
    """Request to apply a force or impulse to a physics body."""

    actor_path: str = Field(..., description="Full actor object path")
    force_x: float = Field(0.0, description="Force X component (Newtons or cm/s)")
    force_y: float = Field(0.0, description="Force Y component")
    force_z: float = Field(0.0, description="Force Z component")
    is_impulse: bool = Field(
        False,
        description="True for impulse (instant), False for continuous force",
    )
    location_x: Optional[float] = Field(
        None, description="Application point X (actor-local); None for center of mass"
    )
    location_y: Optional[float] = Field(None, description="Application point Y")
    location_z: Optional[float] = Field(None, description="Application point Z")


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


class UnrealSetPhysicsParamsRequest(BaseModel):
    """Request to set physics body parameters on an actor."""

    actor_path: str = Field(..., description="Full actor object path")
    mass: Optional[float] = Field(None, description="Mass in kg")
    linear_damping: Optional[float] = Field(None, description="Linear damping")
    angular_damping: Optional[float] = Field(None, description="Angular damping")
    enable_gravity: Optional[bool] = Field(None, description="Enable gravity")


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


class UnrealImportUsdRequest(BaseModel):
    """Request to import a USD file via Interchange Framework."""

    usd_path: str = Field(..., description="Path to USD file to import")
    destination_path: str = Field(
        "/Game/Imports",
        description="Content browser destination path",
    )
    import_options: Optional[Dict[str, Any]] = Field(
        None, description="Interchange pipeline options"
    )


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


class UnrealExportUsdRequest(BaseModel):
    """Request to export actors to USD."""

    actor_paths: List[str] = Field(..., description="Actor paths to export")
    output_path: str = Field(..., description="Output USD file path (.usd/.usda/.usdc)")
    export_options: Optional[Dict[str, Any]] = Field(
        None, description="Export pipeline options"
    )


class UnrealExportUsdResponse(BaseModel):
    """Response after exporting actors to USD."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    output_path: str = Field(..., description="Written USD file path")
    actors_exported: int = Field(..., description="Number of actors exported")
    file_size_bytes: int = Field(0, description="Output file size in bytes")


class UnrealConvertToSimreadyRequest(BaseModel):
    """Request to convert a USD asset to SimReady format."""

    usd_path: str = Field(..., description="Source USD file path")
    output_path: str = Field(..., description="Output SimReady USD path")
    add_physics: bool = Field(True, description="Add physics schema")
    add_collision: bool = Field(True, description="Generate collision geometry")
    add_semantic_labels: bool = Field(True, description="Add semantic label metadata")
    target_up_axis: str = Field("Z", description="Target up axis (Z)")
    target_units: str = Field("meters", description="Target units")


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


class UnrealValidateSimreadyRequest(BaseModel):
    """Request to validate asset against SimReady spec."""

    usd_path: str = Field(..., description="USD file path to validate")
    checks: List[str] = Field(
        default_factory=lambda: [
            "physics",
            "collision",
            "materials",
            "scale",
            "up_axis",
            "semantics",
        ],
        description="Validation checks to run",
    )


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


class UnrealGetInterchangeInfoRequest(BaseModel):
    """Request to query available Interchange pipelines."""

    pass


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


# ----------------------------------------------------------------------
# Phase 7: Advanced Agent Tools
# ----------------------------------------------------------------------


class UnrealBatchOperationsRequest(BaseModel):
    """Request to execute multiple operations atomically."""

    operations: List[Dict[str, Any]] = Field(
        ..., description="List of operation dicts with endpoint/body pairs"
    )


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


class UnrealQuerySceneGraphRequest(BaseModel):
    """Request to query the scene graph structure."""

    root_path: Optional[str] = Field(
        None, description="Root actor path to start from (None for level root)"
    )
    max_depth: int = Field(10, description="Maximum traversal depth")
    include_components: bool = Field(False, description="Include component hierarchy")
    class_filter: Optional[str] = Field(None, description="Filter by actor class")


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


class UnrealAnalyzeSceneForRoboticsRequest(BaseModel):
    """Request to analyze a scene for robotics use-cases."""

    analysis_types: List[str] = Field(
        default_factory=lambda: [
            "traversability",
            "graspability",
            "collision_complexity",
        ],
        description="Analyses to run",
    )
    actor_filter: Optional[str] = Field(
        None, description="Filter to specific actor subtree"
    )


class UnrealAnalyzeSceneForRoboticsResponse(BaseModel):
    """Response with robotics scene analysis."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    traversable_surfaces: List[Dict[str, Any]] = Field(
        default_factory=list, description="Surfaces a robot can traverse"
    )
    graspable_objects: List[Dict[str, Any]] = Field(
        default_factory=list, description="Objects suitable for grasping"
    )
    collision_summary: Dict[str, Any] = Field(
        default_factory=dict, description="Collision complexity summary"
    )
    total_actors_analyzed: int = Field(0, description="Number of actors analyzed")


class UnrealGenerateProceduralSceneRequest(BaseModel):
    """Request to generate a procedural scene via PCG."""

    scene_type: str = Field(
        ..., description="Scene type (warehouse, outdoor, room, corridor)"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Generation parameters (size, density, seed, etc.)",
    )
    bounds_min: List[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        description="Minimum bounds [x, y, z] in cm",
    )
    bounds_max: List[float] = Field(
        default_factory=lambda: [1000.0, 1000.0, 500.0],
        description="Maximum bounds [x, y, z] in cm",
    )


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


class UnrealGetActorBySemanticLabelRequest(BaseModel):
    """Request to find actors by semantic tag."""

    label: str = Field(..., description="Semantic label to search for")
    match_mode: str = Field("exact", description="Match mode: exact, contains, regex")
    max_results: int = Field(100, description="Maximum results to return")


class UnrealGetActorBySemanticLabelResponse(BaseModel):
    """Response with actors matching semantic label."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    actors: List[Dict[str, Any]] = Field(
        ..., description="Matching actors with paths and labels"
    )
    total_matches: int = Field(..., description="Number of matches found")
    label_searched: str = Field(..., description="Label that was searched")


# ----------------------------------------------------------------------
# Phase 8: Geometry & Modeling
# ----------------------------------------------------------------------


class UnrealGenerateMeshPrimitiveRequest(BaseModel):
    """Request to create a parametric mesh primitive."""

    primitive_type: str = Field(
        ...,
        description="Primitive type: box, sphere, cylinder, cone, torus, capsule",
    )
    dimensions: Dict[str, float] = Field(
        default_factory=dict,
        description="Type-specific dimensions (e.g. radius, height, width)",
    )
    segments: int = Field(32, description="Tessellation segments")
    location: List[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        description="Spawn location [x, y, z] in cm",
    )
    actor_label: Optional[str] = Field(None, description="Optional actor label")


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


class UnrealApplyMeshBooleanRequest(BaseModel):
    """Request to apply boolean operation between two meshes."""

    target_mesh_path: str = Field(
        ..., description="Target mesh actor path (will be modified)"
    )
    tool_mesh_path: str = Field(
        ..., description="Tool mesh actor path (used as operand)"
    )
    operation: str = Field(
        ..., description="Boolean operation: union, subtract, intersect"
    )


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


class UnrealComputeConvexHullRequest(BaseModel):
    """Request to compute a convex hull of a mesh."""

    mesh_path: str = Field(..., description="Source mesh actor path")


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


class UnrealDecomposeConvexHullRequest(BaseModel):
    """Request for V-HACD convex decomposition."""

    mesh_path: str = Field(..., description="Source mesh actor path")
    max_hulls: int = Field(16, description="Maximum number of convex pieces")
    max_vertices_per_hull: int = Field(32, description="Max vertices per hull")
    min_cluster_size: int = Field(256, description="Minimum cluster size")
    resolution: int = Field(100000, description="V-HACD voxelization resolution")


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


class UnrealEditMeshTopologyRequest(BaseModel):
    """Request to edit mesh topology (extrude, bevel, inset, loop cut)."""

    mesh_path: str = Field(..., description="Mesh actor path")
    operation: str = Field(
        ...,
        description="Operation: extrude_faces, bevel_edges, inset_faces, loop_cut",
    )
    face_selection: Optional[str] = Field(
        None, description="Face selection filter (e.g. top, front_hemisphere)"
    )
    edge_selection: Optional[str] = Field(
        None, description="Edge selection filter (e.g. all_boundary)"
    )
    distance: Optional[float] = Field(None, description="Extrude distance")
    offset: Optional[float] = Field(None, description="Bevel/inset offset")
    scale: Optional[List[float]] = Field(
        None, description="Scale factors [x, y, z] for scale_faces"
    )
    count: Optional[int] = Field(None, description="Loop cut count")


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


class UnrealSubdivideMeshRequest(BaseModel):
    """Request to subdivide a mesh (Catmull-Clark, Loop, or bilinear)."""

    mesh_path: str = Field(..., description="Mesh actor path")
    level: int = Field(2, description="Subdivision level (1-4)")
    scheme: str = Field(
        "catmull_clark",
        description="Scheme: catmull_clark, loop, bilinear",
    )


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


class UnrealSimplifyMeshRequest(BaseModel):
    """Request to simplify/decimate a mesh."""

    mesh_path: str = Field(..., description="Mesh actor path")
    target_triangle_count: Optional[int] = Field(
        None, description="Target triangle count"
    )
    target_percentage: Optional[float] = Field(
        None, description="Target percentage (0.0-1.0) of original triangles"
    )
    max_error: Optional[float] = Field(
        None, description="Maximum geometric error tolerance"
    )


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


class UnrealCutMeshPlaneRequest(BaseModel):
    """Request to cut/slice a mesh along a plane."""

    mesh_path: str = Field(..., description="Mesh actor path")
    plane_origin: List[float] = Field(..., description="Plane origin [x, y, z] in cm")
    plane_normal: List[float] = Field(..., description="Plane normal [x, y, z]")
    fill_holes: bool = Field(True, description="Fill cut holes with faces")
    keep_both_sides: bool = Field(
        False, description="Keep both sides as separate actors"
    )


class UnrealCutMeshPlaneResponse(BaseModel):
    """Response after plane cut."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    mesh_path: str = Field(..., description="Cut mesh path")
    pieces: List[str] = Field(..., description="Resulting piece actor paths")
    cut_faces_added: int = Field(0, description="Number of fill faces added")


class UnrealValidateMeshRequest(BaseModel):
    """Request to validate mesh geometry."""

    mesh_path: str = Field(..., description="Mesh actor path")
    checks: List[str] = Field(
        default_factory=lambda: [
            "watertight",
            "manifold",
            "normals",
            "self_intersection",
        ],
        description="Validation checks to run",
    )


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


class UnrealConvertMeshFormatRequest(BaseModel):
    """Request to convert between StaticMesh and DynamicMesh."""

    mesh_path: str = Field(..., description="Source mesh actor/asset path")
    target_format: str = Field(
        ...,
        description="Target: static_mesh, dynamic_mesh, or cad_tessellation",
    )
    tessellation_options: Optional[Dict[str, Any]] = Field(
        None,
        description="CAD tessellation params (chord_tolerance, angle_tolerance)",
    )


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


class UnrealRemeshMeshRequest(BaseModel):
    """Request to remesh for clean topology."""

    mesh_path: str = Field(..., description="Mesh actor path")
    mode: str = Field("uniform", description="Remesh mode: uniform, adaptive")
    target_edge_length: Optional[float] = Field(
        None, description="Target edge length in cm (uniform mode)"
    )
    target_triangle_count: Optional[int] = Field(
        None, description="Target triangle count (adaptive mode)"
    )
    smoothing_iterations: int = Field(3, description="Number of smoothing iterations")


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


class UnrealComputeMeshUvRequest(BaseModel):
    """Request to generate UV coordinates."""

    mesh_path: str = Field(..., description="Mesh actor path")
    method: str = Field(
        "auto_uv",
        description="Method: auto_uv, box, planar, cylindrical, atlas_pack",
    )
    uv_channel: int = Field(0, description="UV channel index")
    island_padding: float = Field(2.0, description="Padding between UV islands")


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
    "UnrealPingResponse",
    "UnrealInstanceInfo",
    "UnrealListInstancesResponse",
    "UnrealHealthCheckResponse",
    "UnrealEngineInfoResponse",
    "UnrealLoadedMapResponse",
    "UnrealListActorsRequest",
    "UnrealActorEntry",
    "UnrealListActorsResponse",
    "UnrealGetActorInfoRequest",
    "UnrealActorComponentInfo",
    "UnrealGetActorInfoResponse",
    "UnrealSearchAssetsRequest",
    "UnrealAssetEntry",
    "UnrealSearchAssetsResponse",
    "UnrealDescribeObjectRequest",
    "UnrealPropertyInfo",
    "UnrealDescribeObjectResponse",
    "UnrealGetThumbnailRequest",
    "UnrealGetThumbnailResponse",
    "UnrealSceneSummaryResponse",
    "UnrealCaptureViewportRequest",
    "UnrealCaptureViewportResponse",
    "UnrealViewportInfoRequest",
    "UnrealViewportInfoResponse",
    "UnrealSetCameraViewRequest",
    "UnrealSetCameraViewResponse",
    "UnrealFocusActorRequest",
    "UnrealFocusActorResponse",
    "UnrealSpawnActorRequest",
    "UnrealSpawnActorResponse",
    "UnrealDeleteActorRequest",
    "UnrealDeleteActorResponse",
    "UnrealSetActorTransformRequest",
    "UnrealSetActorTransformResponse",
    "UnrealSetActorPropertyRequest",
    "UnrealSetActorPropertyResponse",
    "UnrealCallActorFunctionRequest",
    "UnrealCallActorFunctionResponse",
    "UnrealSetActorParentRequest",
    "UnrealSetActorParentResponse",
    "UnrealAddComponentRequest",
    "UnrealAddComponentResponse",
    "UnrealSetActorVisibilityRequest",
    "UnrealSetActorVisibilityResponse",
    "UnrealGetMaterialInfoRequest",
    "UnrealMaterialParameterInfo",
    "UnrealGetMaterialInfoResponse",
    "UnrealSetMaterialParamsRequest",
    "UnrealSetMaterialParamsResponse",
    "UnrealCreateMaterialInstanceRequest",
    "UnrealCreateMaterialInstanceResponse",
    "UnrealAssignMaterialRequest",
    "UnrealAssignMaterialResponse",
    "UnrealSetLightParamsRequest",
    "UnrealSetLightParamsResponse",
    "UnrealSetRenderSettingsRequest",
    "UnrealSetRenderSettingsResponse",
    "UnrealControlSimulationRequest",
    "UnrealControlSimulationResponse",
    "UnrealGetSimulationStatusRequest",
    "UnrealGetSimulationStatusResponse",
    "UnrealEnablePhysicsRequest",
    "UnrealEnablePhysicsResponse",
    "UnrealSetCollisionRequest",
    "UnrealSetCollisionResponse",
    "UnrealApplyForceRequest",
    "UnrealApplyForceResponse",
    "UnrealSetPhysicsParamsRequest",
    "UnrealSetPhysicsParamsResponse",
    "UnrealImportUsdRequest",
    "UnrealImportUsdResponse",
    "UnrealExportUsdRequest",
    "UnrealExportUsdResponse",
    "UnrealConvertToSimreadyRequest",
    "UnrealConvertToSimreadyResponse",
    "UnrealValidateSimreadyRequest",
    "UnrealValidateSimreadyResponse",
    "UnrealGetInterchangeInfoRequest",
    "UnrealGetInterchangeInfoResponse",
    "UnrealBatchOperationsRequest",
    "UnrealBatchOperationsResponse",
    "UnrealQuerySceneGraphRequest",
    "UnrealQuerySceneGraphResponse",
    "UnrealAnalyzeSceneForRoboticsRequest",
    "UnrealAnalyzeSceneForRoboticsResponse",
    "UnrealGenerateProceduralSceneRequest",
    "UnrealGenerateProceduralSceneResponse",
    "UnrealGetActorBySemanticLabelRequest",
    "UnrealGetActorBySemanticLabelResponse",
    "UnrealGenerateMeshPrimitiveRequest",
    "UnrealGenerateMeshPrimitiveResponse",
    "UnrealApplyMeshBooleanRequest",
    "UnrealApplyMeshBooleanResponse",
    "UnrealComputeConvexHullRequest",
    "UnrealComputeConvexHullResponse",
    "UnrealDecomposeConvexHullRequest",
    "UnrealDecomposeConvexHullResponse",
    "UnrealEditMeshTopologyRequest",
    "UnrealEditMeshTopologyResponse",
    "UnrealSubdivideMeshRequest",
    "UnrealSubdivideMeshResponse",
    "UnrealSimplifyMeshRequest",
    "UnrealSimplifyMeshResponse",
    "UnrealCutMeshPlaneRequest",
    "UnrealCutMeshPlaneResponse",
    "UnrealValidateMeshRequest",
    "UnrealValidateMeshResponse",
    "UnrealConvertMeshFormatRequest",
    "UnrealConvertMeshFormatResponse",
    "UnrealRemeshMeshRequest",
    "UnrealRemeshMeshResponse",
    "UnrealComputeMeshUvRequest",
    "UnrealComputeMeshUvResponse",
]
