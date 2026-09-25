"""Unreal Engine MCP schemas."""

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    reachable: bool = Field(
        ..., description="Whether the Remote Control API responded"
    )
    connected: bool = Field(
        ...,
        description=(
            "Deprecated alias of reachable, kept for one release; read "
            "reachable instead"
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
    """Response with a base64-encoded thumbnail image."""

    success: bool = Field(..., description="Whether request was successful")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    asset_path: str = Field(..., description="Asset path queried")
    image_base64: str = Field(..., description="Base64-encoded thumbnail image")
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
    location: Tuple[float, float, float] = Field(..., description="Applied location")
    rotation: Tuple[float, float, float] = Field(..., description="Applied rotation")
    scale: Tuple[float, float, float] = Field(..., description="Applied scale")


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


class UnrealSetMaterialParamsResponse(BaseModel):
    """Response after setting material instance parameters."""

    success: bool = Field(True, description="Operation success")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    material_path: str = Field(..., description="Material Instance path")
    params_set: int = Field(0, description="Number of parameters set")


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
    params_set: int = Field(0, description="Number of params changed")


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
    collision_preset: str = Field(..., description="Applied collision preset")
    collision_enabled: bool = Field(..., description="Current collision state")


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
        default_factory=list, description="Surfaces a robot can traverse"
    )
    graspable_objects: List[Dict[str, Any]] = Field(
        default_factory=list, description="Objects suitable for grasping"
    )
    collision_summary: Dict[str, Any] = Field(
        default_factory=dict, description="Collision complexity summary"
    )
    total_actors_analyzed: int = Field(0, description="Number of actors analyzed")


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
