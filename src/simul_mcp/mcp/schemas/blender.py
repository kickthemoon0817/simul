"""Blender MCP schemas."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

class BlenderInfoResponse(BaseModel):
    """Response with Blender runtime information."""

    success: bool = Field(..., description="Whether request was successful")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    version: List[int] = Field(..., description="Blender version tuple values")
    version_string: str = Field(..., description="Human-readable Blender version")
    binary_path: Optional[str] = Field(None, description="Blender binary path")
    background: bool = Field(..., description="Whether Blender runs in background mode")
    blend_file_path: Optional[str] = Field(None, description="Current .blend file path")


class BlenderObjectInfo(BaseModel):
    """Serializable Blender object entry."""

    name: str = Field(..., description="Object name")
    object_type: str = Field(..., description="Blender object type")
    collection: Optional[str] = Field(None, description="Collection name filter")
    visible: bool = Field(..., description="Whether object is visible")


class BlenderSceneObjectsRequest(BaseModel):
    """Request for Blender scene object listing."""

    collection_name: Optional[str] = Field(
        None,
        description="Optional collection name to filter objects",
    )
    include_hidden: bool = Field(
        False,
        description="Include hidden objects when true",
    )
    max_items: int = Field(
        200,
        description="Maximum number of objects to return",
        ge=1,
        le=5000,
    )


class BlenderSceneObjectsResponse(BaseModel):
    """Response for Blender scene object listing."""

    success: bool = Field(..., description="Whether request was successful")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    collection: Optional[str] = Field(None, description="Collection used for filtering")
    include_hidden: bool = Field(
        ..., description="Whether hidden objects were included"
    )
    max_items: int = Field(..., description="Maximum number of requested objects")
    count: int = Field(..., description="Number of objects in response")
    objects: List[BlenderObjectInfo] = Field(
        ...,
        description="Listed Blender objects",
    )
    truncated: bool = Field(..., description="Whether output reached max_items limit")


# ---------------------------------------------------------------------------
# Blender observation schemas (Phase 1)
# ---------------------------------------------------------------------------


class BlenderObjectInfoRequest(BaseModel):
    """Request for detailed object information."""

    object_name: str = Field(..., description="Name of the Blender object")


class BlenderModifierEntry(BaseModel):
    """Single modifier on a Blender object."""

    name: str = Field(..., description="Modifier name")
    modifier_type: str = Field(..., description="Modifier type identifier")


class BlenderConstraintEntry(BaseModel):
    """Single constraint on a Blender object."""

    name: str = Field(..., description="Constraint name")
    constraint_type: str = Field(..., description="Constraint type identifier")


class BlenderMaterialSlotEntry(BaseModel):
    """Single material slot on a Blender object."""

    slot_index: int = Field(..., description="Material slot index")
    material_name: Optional[str] = Field(
        None, description="Assigned material name or None if empty"
    )


class BlenderObjectInfoResponse(BaseModel):
    """Detailed information about a single Blender object."""

    success: bool = Field(..., description="Whether request was successful")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    name: str = Field(..., description="Object name")
    object_type: str = Field(..., description="Blender object type")
    location: List[float] = Field(
        ..., description="World location [x, y, z]", min_length=3, max_length=3
    )
    rotation_euler: List[float] = Field(
        ..., description="Rotation in Euler radians [x, y, z]", min_length=3, max_length=3
    )
    scale: List[float] = Field(
        ..., description="Scale [x, y, z]", min_length=3, max_length=3
    )
    parent_name: Optional[str] = Field(None, description="Parent object name")
    children_names: List[str] = Field(
        default_factory=list, description="Direct child object names"
    )
    modifiers: List[BlenderModifierEntry] = Field(
        default_factory=list, description="Object modifiers"
    )
    constraints: List[BlenderConstraintEntry] = Field(
        default_factory=list, description="Object constraints"
    )
    material_slots: List[BlenderMaterialSlotEntry] = Field(
        default_factory=list, description="Material slot assignments"
    )
    visible: bool = Field(..., description="Whether object is visible in viewport")


class BlenderMeshInfoRequest(BaseModel):
    """Request for mesh geometry counts."""

    object_name: str = Field(..., description="Name of the mesh object")


class BlenderMeshInfoResponse(BaseModel):
    """Counts-only mesh geometry information (O(1) access)."""

    success: bool = Field(..., description="Whether request was successful")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    object_name: str = Field(..., description="Mesh object name")
    vertex_count: int = Field(..., description="Number of vertices")
    edge_count: int = Field(..., description="Number of edges")
    face_count: int = Field(..., description="Number of polygon faces")
    uv_layer_names: List[str] = Field(
        default_factory=list, description="UV layer names"
    )
    has_shape_keys: bool = Field(..., description="Whether mesh has shape keys")


class BlenderBoundingBoxRequest(BaseModel):
    """Request for object bounding box."""

    object_name: str = Field(..., description="Name of the Blender object")
    world_space: bool = Field(
        True, description="Return corners in world space when True"
    )


class BlenderBoundingBoxResponse(BaseModel):
    """Eight world-space bounding box corners."""

    success: bool = Field(..., description="Whether request was successful")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    object_name: str = Field(..., description="Object name")
    corners: List[List[float]] = Field(
        ..., description="Eight bounding box corners as [x, y, z] lists"
    )
    bbox_min: List[float] = Field(
        ..., description="Axis-aligned minimum [x, y, z]", min_length=3, max_length=3
    )
    bbox_max: List[float] = Field(
        ..., description="Axis-aligned maximum [x, y, z]", min_length=3, max_length=3
    )
    world_space: bool = Field(..., description="Whether corners are in world space")


class BlenderSearchObjectsRequest(BaseModel):
    """Request for searching objects by criteria."""

    name_pattern: Optional[str] = Field(
        None, description="Regex or glob pattern to match object names"
    )
    object_type: Optional[str] = Field(
        None, description="Filter by Blender object type (MESH, LIGHT, CAMERA, etc.)"
    )
    max_results: int = Field(50, description="Maximum number of results", ge=1, le=5000)


class BlenderSearchObjectsResponse(BaseModel):
    """Search results for object lookup."""

    success: bool = Field(..., description="Whether request was successful")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    pattern: Optional[str] = Field(None, description="Name pattern used")
    object_type: Optional[str] = Field(None, description="Type filter used")
    count: int = Field(..., description="Number of matching objects")
    objects: List[BlenderObjectInfo] = Field(..., description="Matching objects")
    truncated: bool = Field(..., description="Whether results were capped")


class BlenderSceneSummaryResponse(BaseModel):
    """High-level scene summary grouped by type."""

    success: bool = Field(..., description="Whether request was successful")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    total_objects: int = Field(..., description="Total number of objects")
    type_counts: Dict[str, int] = Field(
        ..., description="Object counts keyed by Blender type"
    )
    collection_names: List[str] = Field(
        default_factory=list, description="Top-level collection names"
    )
    active_camera: Optional[str] = Field(
        None, description="Name of the active scene camera"
    )
    frame_current: int = Field(..., description="Current frame number")
    frame_start: int = Field(..., description="Scene start frame")
    frame_end: int = Field(..., description="Scene end frame")


class BlenderMaterialInfoRequest(BaseModel):
    """Request for material information."""

    material_name: str = Field(..., description="Name of the Blender material")


class BlenderNodeEntry(BaseModel):
    """Summarised shader node."""

    name: str = Field(..., description="Node name")
    node_type: str = Field(..., description="Node bl_idname")
    label: str = Field("", description="Node label")


class BlenderMaterialInfoResponse(BaseModel):
    """Material information with bounded node tree traversal."""

    success: bool = Field(..., description="Whether request was successful")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    material_name: str = Field(..., description="Material name")
    use_nodes: bool = Field(..., description="Whether material uses node tree")
    nodes: List[BlenderNodeEntry] = Field(
        default_factory=list, description="Shader nodes (bounded traversal)"
    )
    principled_params: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Principled BSDF parameters if present "
            "(base_color, metallic, roughness, etc.)"
        ),
    )


class BlenderDistanceRequest(BaseModel):
    """Request for distance between two objects."""

    object_name_a: str = Field(..., description="First object name")
    object_name_b: str = Field(..., description="Second object name")


class BlenderDistanceResponse(BaseModel):
    """Distance measurement between two objects."""

    success: bool = Field(..., description="Whether request was successful")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    object_name_a: str = Field(..., description="First object name")
    object_name_b: str = Field(..., description="Second object name")
    distance: float = Field(..., description="Euclidean distance between objects")
    location_a: List[float] = Field(
        ...,
        description="World location of first object [x, y, z]",
        min_length=3,
        max_length=3,
    )
    location_b: List[float] = Field(
        ...,
        description="World location of second object [x, y, z]",
        min_length=3,
        max_length=3,
    )


class BlenderBoundsCheckRequest(BaseModel):
    """Request to check if an object is within spatial bounds."""

    object_name: str = Field(..., description="Object name to check")
    bounds_min: List[float] = Field(
        ..., description="Minimum bounds [x, y, z]", min_length=3, max_length=3
    )
    bounds_max: List[float] = Field(
        ..., description="Maximum bounds [x, y, z]", min_length=3, max_length=3
    )


class BlenderBoundsCheckResponse(BaseModel):
    """Result of a spatial bounds check."""

    success: bool = Field(..., description="Whether request was successful")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    object_name: str = Field(..., description="Object name checked")
    within_bounds: bool = Field(
        ..., description="Whether object location is within the bounds"
    )
    object_location: List[float] = Field(
        ...,
        description="Object world location [x, y, z]",
        min_length=3,
        max_length=3,
    )


# ---------------------------------------------------------------------------
# End Blender observation schemas
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Blender visual observation schemas (Phase 2)
# ---------------------------------------------------------------------------


class BlenderCaptureViewportRequest(BaseModel):
    """Request to capture the current viewport as a JPEG image."""

    width: int = Field(512, description="Output image width in pixels", ge=64, le=4096)
    height: int = Field(
        512, description="Output image height in pixels", ge=64, le=4096
    )
    jpeg_quality: int = Field(
        85, description="JPEG compression quality (1-100)", ge=1, le=100
    )
    use_render_fallback: bool = Field(
        False,
        description=(
            "Force bpy.ops.render.render path " "instead of GPUOffScreen fast path"
        ),
    )


class BlenderCaptureViewportResponse(BaseModel):
    """Base64-encoded JPEG viewport capture result."""

    success: bool = Field(..., description="Whether capture succeeded")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    image_base64: str = Field(..., description="Base64-encoded JPEG image data")
    width: int = Field(..., description="Captured image width")
    height: int = Field(..., description="Captured image height")
    engine: str = Field(
        ..., description="Render engine used (BLENDER_EEVEE, CYCLES, etc.)"
    )
    capture_method: str = Field(
        ..., description="Method used: gpu_offscreen or render_fallback"
    )


class BlenderSetCameraViewRequest(BaseModel):
    """Request to set the active camera's transform."""

    location: List[float] = Field(
        ..., description="Camera location [x, y, z]", min_length=3, max_length=3
    )
    rotation_euler: List[float] = Field(
        ...,
        description="Camera rotation [rx, ry, rz] in radians",
        min_length=3,
        max_length=3,
    )
    camera_name: Optional[str] = Field(
        None, description="Target camera name. Uses active camera when None."
    )


class BlenderSetCameraViewResponse(BaseModel):
    """Result of setting a camera view."""

    success: bool = Field(..., description="Whether operation succeeded")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    camera_name: str = Field(..., description="Name of the camera that was updated")
    location: List[float] = Field(
        ...,
        description="New camera location [x, y, z]",
        min_length=3,
        max_length=3,
    )
    rotation_euler: List[float] = Field(
        ...,
        description="New camera rotation [rx, ry, rz]",
        min_length=3,
        max_length=3,
    )


class BlenderCameraInfoResponse(BaseModel):
    """Active camera information."""

    success: bool = Field(..., description="Whether request succeeded")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    camera_name: str = Field(..., description="Active camera name")
    location: List[float] = Field(
        ...,
        description="Camera location [x, y, z]",
        min_length=3,
        max_length=3,
    )
    rotation_euler: List[float] = Field(
        ...,
        description="Camera rotation [rx, ry, rz]",
        min_length=3,
        max_length=3,
    )
    lens: float = Field(..., description="Focal length in mm")
    sensor_width: float = Field(..., description="Sensor width in mm")
    clip_start: float = Field(..., description="Near clip distance")
    clip_end: float = Field(..., description="Far clip distance")
    camera_type: str = Field(..., description="Camera type: PERSP, ORTHO, or PANO")


class BlenderFocusOnObjectRequest(BaseModel):
    """Request to focus the camera on a specific object."""

    object_name: str = Field(..., description="Object name to focus on")
    distance_factor: float = Field(
        2.0,
        description="Distance multiplier from bounding box diagonal",
        ge=0.5,
        le=20.0,
    )
    camera_name: Optional[str] = Field(
        None, description="Target camera name. Uses active camera when None."
    )


class BlenderFocusOnObjectResponse(BaseModel):
    """Result of focusing camera on an object."""

    success: bool = Field(..., description="Whether focus operation succeeded")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    camera_name: str = Field(..., description="Camera that was updated")
    object_name: str = Field(..., description="Object that was focused on")
    camera_location: List[float] = Field(
        ...,
        description="New camera location [x, y, z]",
        min_length=3,
        max_length=3,
    )
    look_at: List[float] = Field(
        ...,
        description="Point the camera is aimed at [x, y, z]",
        min_length=3,
        max_length=3,
    )


class BlenderViewportInfoResponse(BaseModel):
    """Active viewport / render settings summary."""

    success: bool = Field(..., description="Whether request succeeded")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    render_engine: str = Field(..., description="Active render engine identifier")
    resolution_x: int = Field(..., description="Render resolution X")
    resolution_y: int = Field(..., description="Render resolution Y")
    resolution_percentage: int = Field(..., description="Resolution percentage scale")
    film_transparent: bool = Field(
        ..., description="Whether film transparency is enabled"
    )
    active_camera: Optional[str] = Field(None, description="Active camera name or None")


class BlenderCaptureSequenceRequest(BaseModel):
    """Request for multi-frame viewport capture."""

    start_frame: int = Field(..., description="First frame to capture")
    end_frame: int = Field(..., description="Last frame to capture")
    step: int = Field(1, description="Frame step between captures", ge=1)
    width: int = Field(512, description="Output image width", ge=64, le=4096)
    height: int = Field(512, description="Output image height", ge=64, le=4096)
    jpeg_quality: int = Field(85, description="JPEG compression quality", ge=1, le=100)


class BlenderCaptureSequenceResponse(BaseModel):
    """Multi-frame capture result with per-frame base64 images."""

    success: bool = Field(..., description="Whether capture succeeded")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    frames: List[Dict[str, Any]] = Field(
        ..., description="Per-frame data: [{frame: int, image_base64: str}, ...]"
    )
    frame_count: int = Field(..., description="Number of frames captured")
    capture_method: str = Field(..., description="Method used for capture")


# ---------------------------------------------------------------------------
# End Blender visual observation schemas
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Blender scene manipulation schemas (Phase 3)
# ---------------------------------------------------------------------------


class BlenderCreateObjectRequest(BaseModel):
    """Request to create a new object in the Blender scene."""

    object_type: str = Field(
        ...,
        description=(
            "Object type: CUBE, SPHERE, CYLINDER, CONE, PLANE, TORUS, "
            "POINT_LIGHT, SUN_LIGHT, SPOT_LIGHT, AREA_LIGHT, CAMERA, EMPTY"
        ),
    )
    name: Optional[str] = Field(
        None, description="Object name (auto-generated if omitted)"
    )
    location: List[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        description="Initial location [x, y, z]",
        min_length=3,
        max_length=3,
    )
    rotation_euler: List[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        description="Initial rotation [rx, ry, rz] radians",
        min_length=3,
        max_length=3,
    )
    scale: List[float] = Field(
        default_factory=lambda: [1.0, 1.0, 1.0],
        description="Initial scale [sx, sy, sz]",
        min_length=3,
        max_length=3,
    )


class BlenderCreateObjectResponse(BaseModel):
    """Response after creating a Blender object."""

    success: bool = Field(..., description="Whether creation succeeded")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    name: str = Field(..., description="Resulting object name")
    object_type: str = Field(..., description="Created object type")
    location: List[float] = Field(
        ...,
        description="Object location [x, y, z]",
        min_length=3,
        max_length=3,
    )


class BlenderDeleteObjectRequest(BaseModel):
    """Request to delete an object from the scene."""

    object_name: str = Field(..., description="Name of object to delete")


class BlenderDeleteObjectResponse(BaseModel):
    """Response after deleting a Blender object."""

    success: bool = Field(..., description="Whether deletion succeeded")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    deleted_name: str = Field(..., description="Name of deleted object")


class BlenderSetTransformRequest(BaseModel):
    """Request to set an object's transform."""

    object_name: str = Field(..., description="Object to transform")
    location: Optional[List[float]] = Field(
        None,
        description="New location [x, y, z]",
        min_length=3,
        max_length=3,
    )
    rotation_euler: Optional[List[float]] = Field(
        None,
        description="New rotation [rx, ry, rz] radians",
        min_length=3,
        max_length=3,
    )
    scale: Optional[List[float]] = Field(
        None,
        description="New scale [sx, sy, sz]",
        min_length=3,
        max_length=3,
    )


class BlenderSetTransformResponse(BaseModel):
    """Response after setting an object's transform."""

    success: bool = Field(..., description="Whether transform succeeded")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    object_name: str = Field(..., description="Object name")
    location: List[float] = Field(
        ...,
        description="Final location [x, y, z]",
        min_length=3,
        max_length=3,
    )
    rotation_euler: List[float] = Field(
        ...,
        description="Final rotation [rx, ry, rz]",
        min_length=3,
        max_length=3,
    )
    scale: List[float] = Field(
        ...,
        description="Final scale [sx, sy, sz]",
        min_length=3,
        max_length=3,
    )


class BlenderSetParentRequest(BaseModel):
    """Request to parent one object to another."""

    child_name: str = Field(..., description="Object to parent")
    parent_name: str = Field(..., description="Object to be the parent")


class BlenderSetParentResponse(BaseModel):
    """Response after parenting objects."""

    success: bool = Field(..., description="Whether parenting succeeded")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    child_name: str = Field(..., description="Child object name")
    parent_name: str = Field(..., description="Parent object name")


class BlenderClearParentRequest(BaseModel):
    """Request to unparent an object."""

    object_name: str = Field(..., description="Object to unparent")
    keep_transform: bool = Field(
        True, description="Keep world transform after unparenting"
    )


class BlenderClearParentResponse(BaseModel):
    """Response after unparenting an object."""

    success: bool = Field(..., description="Whether unparenting succeeded")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    object_name: str = Field(..., description="Object name")
    previous_parent: Optional[str] = Field(None, description="Previous parent name")


class BlenderAssignMaterialRequest(BaseModel):
    """Request to assign a Principled BSDF material to an object."""

    object_name: str = Field(..., description="Object to assign material to")
    material_name: Optional[str] = Field(
        None,
        description="Material name (auto-generated if omitted)",
    )
    base_color: List[float] = Field(
        default_factory=lambda: [0.8, 0.8, 0.8, 1.0],
        description="Base color RGBA [r, g, b, a]",
    )
    metallic: float = Field(0.0, description="Metallic value (0-1)", ge=0.0, le=1.0)
    roughness: float = Field(0.5, description="Roughness value (0-1)", ge=0.0, le=1.0)


class BlenderAssignMaterialResponse(BaseModel):
    """Response after assigning a material."""

    success: bool = Field(..., description="Whether assignment succeeded")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    object_name: str = Field(..., description="Object name")
    material_name: str = Field(..., description="Material name assigned")


class BlenderAddModifierRequest(BaseModel):
    """Request to add a modifier to an object."""

    object_name: str = Field(..., description="Object to add modifier to")
    modifier_type: str = Field(
        ...,
        description="Modifier type: SUBSURF, MIRROR, ARRAY, BEVEL, SOLIDIFY, BOOLEAN",
    )
    modifier_name: Optional[str] = Field(None, description="Modifier name")
    params: Dict[str, Any] = Field(
        default_factory=dict, description="Modifier-specific parameters"
    )


class BlenderAddModifierResponse(BaseModel):
    """Response after adding a modifier."""

    success: bool = Field(..., description="Whether modifier was added")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    object_name: str = Field(..., description="Object name")
    modifier_name: str = Field(..., description="Modifier name")
    modifier_type: str = Field(..., description="Modifier type")


class BlenderSetLightParamsRequest(BaseModel):
    """Request to set light properties."""

    light_name: str = Field(..., description="Light object name")
    energy: Optional[float] = Field(None, description="Light energy/power")
    color: Optional[List[float]] = Field(None, description="Light color RGB [r, g, b]")
    use_shadow: Optional[bool] = Field(None, description="Enable shadow casting")
    spot_size: Optional[float] = Field(
        None, description="Spot cone angle radians (SPOT only)"
    )
    spot_blend: Optional[float] = Field(None, description="Spot edge blend (SPOT only)")
    shadow_soft_size: Optional[float] = Field(
        None, description="Shadow soft size / radius (POINT/SUN/AREA)"
    )


class BlenderSetLightParamsResponse(BaseModel):
    """Response after setting light parameters."""

    success: bool = Field(..., description="Whether light params were set")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    light_name: str = Field(..., description="Light object name")
    light_type: str = Field(..., description="Light type (POINT/SUN/SPOT/AREA)")
    energy: float = Field(..., description="Current energy")
    color: List[float] = Field(..., description="Current color RGB")


# ---------------------------------------------------------------------------
# End Blender scene manipulation schemas
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Blender file I/O schemas (Phase 4)
# ---------------------------------------------------------------------------


class BlenderOpenFileRequest(BaseModel):
    """Request to open a .blend file."""

    file_path: str = Field(..., description="Absolute path to the .blend file")


class BlenderOpenFileResponse(BaseModel):
    """Response after opening a .blend file."""

    success: bool = Field(..., description="Whether the file was opened")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    file_path: str = Field(..., description="Path that was opened")
    object_count: int = Field(..., description="Number of objects in the opened scene")


class BlenderSaveFileRequest(BaseModel):
    """Request to save the current .blend file."""

    file_path: Optional[str] = Field(
        None, description="Path to save to. None saves in place."
    )


class BlenderSaveFileResponse(BaseModel):
    """Response after saving a .blend file."""

    success: bool = Field(..., description="Whether the file was saved")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    file_path: str = Field(..., description="Path the file was saved to")


class BlenderImportFileRequest(BaseModel):
    """Request to import a file into the Blender scene."""

    file_path: str = Field(..., description="Absolute path to the file to import")
    file_format: str = Field(
        ..., description="File format: OBJ, FBX, GLTF, USD, STL, PLY"
    )


class BlenderImportFileResponse(BaseModel):
    """Response after importing a file."""

    success: bool = Field(..., description="Whether the import succeeded")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    file_path: str = Field(..., description="Path that was imported")
    file_format: str = Field(..., description="Format that was imported")
    imported_objects: List[str] = Field(
        default_factory=list, description="Names of newly imported objects"
    )


class BlenderExportFileRequest(BaseModel):
    """Request to export scene objects to a file."""

    file_path: str = Field(..., description="Absolute path to export to")
    file_format: str = Field(
        ..., description="File format: OBJ, FBX, GLTF, USD, STL, PLY"
    )
    selected_only: bool = Field(False, description="Export only selected objects")


class BlenderExportFileResponse(BaseModel):
    """Response after exporting a file."""

    success: bool = Field(..., description="Whether the export succeeded")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    file_path: str = Field(..., description="Path the file was exported to")
    file_format: str = Field(..., description="Format that was exported")


class BlenderFileInfoResponse(BaseModel):
    """Response with current file information."""

    success: bool = Field(..., description="Whether info retrieval succeeded")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    file_path: str = Field(..., description="Current .blend file path")
    is_saved: bool = Field(..., description="Whether the file has been saved to disk")
    is_dirty: bool = Field(..., description="Whether there are unsaved changes")
    object_count: int = Field(..., description="Number of objects in the scene")
    scene_name: str = Field(..., description="Active scene name")


# ---------------------------------------------------------------------------
# End Blender file I/O schemas
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Blender animation & timeline schemas (Phase 5)
# ---------------------------------------------------------------------------


class BlenderSetFrameRequest(BaseModel):
    """Request to set the current animation frame."""

    frame: int = Field(..., description="Frame number to set")


class BlenderSetFrameResponse(BaseModel):
    """Response after setting the animation frame."""

    success: bool = Field(..., description="Whether frame was set successfully")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    frame: int = Field(..., description="The frame that was set")


class BlenderGetFrameResponse(BaseModel):
    """Response with current frame and range information."""

    success: bool = Field(..., description="Whether retrieval succeeded")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    current_frame: int = Field(..., description="Current scene frame")
    frame_start: int = Field(..., description="Animation start frame")
    frame_end: int = Field(..., description="Animation end frame")
    fps: float = Field(..., description="Frames per second")


class BlenderSetFrameRangeRequest(BaseModel):
    """Request to set the animation frame range."""

    frame_start: int = Field(..., description="Start frame of the animation range")
    frame_end: int = Field(..., description="End frame of the animation range")


class BlenderSetFrameRangeResponse(BaseModel):
    """Response after setting the frame range."""

    success: bool = Field(..., description="Whether frame range was set")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    frame_start: int = Field(..., description="Start frame that was set")
    frame_end: int = Field(..., description="End frame that was set")


class BlenderPlayAnimationRequest(BaseModel):
    """Request to play or stop animation playback."""

    action: str = Field(
        ..., description="Playback action: 'play', 'stop', or 'reverse'"
    )


class BlenderPlayAnimationResponse(BaseModel):
    """Response after changing playback state."""

    success: bool = Field(..., description="Whether playback action succeeded")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    action: str = Field(..., description="The action that was performed")
    is_playing: bool = Field(..., description="Whether animation is currently playing")


class BlenderInsertKeyframeRequest(BaseModel):
    """Request to insert a keyframe on an object property."""

    object_name: str = Field(..., description="Name of the object")
    data_path: str = Field(
        ..., description="Property data path (e.g. 'location', 'rotation_euler')"
    )
    frame: int = Field(..., description="Frame number for the keyframe")
    index: int = Field(-1, description="Array index (-1 for all channels)")


class BlenderInsertKeyframeResponse(BaseModel):
    """Response after inserting a keyframe."""

    success: bool = Field(..., description="Whether keyframe was inserted")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    object_name: str = Field(..., description="Object name")
    data_path: str = Field(..., description="Property data path")
    frame: int = Field(..., description="Frame number of the keyframe")


class BlenderDeleteKeyframeRequest(BaseModel):
    """Request to delete a keyframe from an object property."""

    object_name: str = Field(..., description="Name of the object")
    data_path: str = Field(
        ..., description="Property data path (e.g. 'location', 'rotation_euler')"
    )
    frame: int = Field(..., description="Frame number of the keyframe to delete")
    index: int = Field(-1, description="Array index (-1 for all channels)")


class BlenderDeleteKeyframeResponse(BaseModel):
    """Response after deleting a keyframe."""

    success: bool = Field(..., description="Whether keyframe was deleted")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    object_name: str = Field(..., description="Object name")
    data_path: str = Field(..., description="Property data path")
    frame: int = Field(..., description="Frame number that was deleted")


class BlenderGetKeyframesRequest(BaseModel):
    """Request to get keyframe summary for an object."""

    object_name: str = Field(..., description="Name of the object")


class BlenderKeyframeSummaryEntry(BaseModel):
    """Summary of keyframes for a single FCurve or channel."""

    data_path: str = Field(..., description="Property data path")
    array_index: int = Field(..., description="Array index of the channel")
    keyframe_count: int = Field(..., description="Number of keyframes")
    frame_range: List[int] = Field(
        ...,
        description="[first_frame, last_frame] of keyframes",
        min_length=2,
        max_length=2,
    )


class BlenderGetKeyframesResponse(BaseModel):
    """Response with keyframe summary for an object."""

    success: bool = Field(..., description="Whether retrieval succeeded")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    object_name: str = Field(..., description="Object name")
    has_animation: bool = Field(
        ..., description="Whether the object has animation data"
    )
    channels: List[BlenderKeyframeSummaryEntry] = Field(
        default_factory=list, description="Keyframe summary per channel"
    )


# ---------------------------------------------------------------------------
# End Blender animation schemas
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Blender physics & simulation schemas (Phase 6)
# ---------------------------------------------------------------------------


class BlenderSetupRigidBodyRequest(BaseModel):
    """Request to set up rigid body physics on an object."""

    object_name: str = Field(..., description="Name of the target object")
    body_type: str = Field(
        "ACTIVE",
        description="Rigid body type: ACTIVE (dynamic) or PASSIVE (static)",
    )
    mass: float = Field(1.0, description="Mass in kilograms")
    friction: float = Field(0.5, description="Surface friction coefficient")
    restitution: float = Field(0.0, description="Bounciness (0-1)")
    collision_shape: str = Field(
        "CONVEX_HULL",
        description=(
            "Collision shape: BOX, SPHERE, CAPSULE, CYLINDER, "
            "CONE, CONVEX_HULL, MESH"
        ),
    )
    linear_damping: float = Field(0.04, description="Linear damping factor")
    angular_damping: float = Field(0.1, description="Angular damping factor")


class BlenderSetupRigidBodyResponse(BaseModel):
    """Response after setting up rigid body."""

    success: bool = Field(..., description="Whether setup succeeded")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    object_name: str = Field(..., description="Name of the object")
    body_type: str = Field(..., description="Rigid body type set")
    mass: float = Field(..., description="Mass set")
    collision_shape: str = Field(..., description="Collision shape set")


class BlenderAddForceFieldRequest(BaseModel):
    """Request to add a force field to the scene."""

    field_type: str = Field(
        ...,
        description=(
            "Force field type: FORCE, WIND, VORTEX, MAGNETIC, "
            "HARMONIC, CHARGE, LENNARDJ, TURBULENCE, DRAG"
        ),
    )
    strength: float = Field(1.0, description="Field strength")
    location: List[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        description="World-space XYZ location",
        min_length=3,
        max_length=3,
    )
    name: Optional[str] = Field(None, description="Optional name for the field")


class BlenderAddForceFieldResponse(BaseModel):
    """Response after adding a force field."""

    success: bool = Field(..., description="Whether creation succeeded")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    name: str = Field(..., description="Name of the created force field object")
    field_type: str = Field(..., description="Type of force field")
    strength: float = Field(..., description="Strength of the field")
    location: List[float] = Field(
        ...,
        description="Location of the field [x, y, z]",
        min_length=3,
        max_length=3,
    )


class BlenderGetForceFieldInfoRequest(BaseModel):
    """Request to get force field info."""

    object_name: str = Field(..., description="Name of the force field object")


class BlenderGetForceFieldInfoResponse(BaseModel):
    """Response with force field details."""

    success: bool = Field(..., description="Whether query succeeded")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    object_name: str = Field(..., description="Name of the force field object")
    field_type: str = Field(..., description="Force field type")
    strength: float = Field(..., description="Field strength")
    shape: str = Field(..., description="Field shape (POINT, PLANE, etc.)")
    flow: float = Field(..., description="Field flow value")
    location: List[float] = Field(
        ...,
        description="World-space location [x, y, z]",
        min_length=3,
        max_length=3,
    )


class BlenderAddConstraintRequest(BaseModel):
    """Request to add a rigid body constraint."""

    constraint_type: str = Field(
        ...,
        description=(
            "Constraint type: FIXED, POINT, HINGE, SLIDER, "
            "PISTON, GENERIC, GENERIC_SPRING, MOTOR"
        ),
    )
    object1_name: str = Field(..., description="First constrained object")
    object2_name: str = Field(..., description="Second constrained object")
    location: Optional[List[float]] = Field(
        None,
        description="Optional location for the constraint empty",
        min_length=3,
        max_length=3,
    )
    disable_collisions: bool = Field(
        True, description="Disable collisions between constrained objects"
    )


class BlenderAddConstraintResponse(BaseModel):
    """Response after adding a rigid body constraint."""

    success: bool = Field(..., description="Whether constraint was created")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    constraint_name: str = Field(..., description="Name of the constraint empty object")
    constraint_type: str = Field(..., description="Type of constraint")
    object1_name: str = Field(..., description="First constrained object")
    object2_name: str = Field(..., description="Second constrained object")


class BlenderGetConstraintInfoRequest(BaseModel):
    """Request to get constraint info."""

    object_name: str = Field(..., description="Name of the constraint object")


class BlenderGetConstraintInfoResponse(BaseModel):
    """Response with rigid body constraint details."""

    success: bool = Field(..., description="Whether query succeeded")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    object_name: str = Field(..., description="Name of the constraint object")
    constraint_type: str = Field(..., description="Constraint type")
    object1_name: Optional[str] = Field(
        None, description="First constrained object name"
    )
    object2_name: Optional[str] = Field(
        None, description="Second constrained object name"
    )
    enabled: bool = Field(..., description="Whether constraint is enabled")
    disable_collisions: bool = Field(..., description="Whether collisions are disabled")


class BlenderGetPhysicsStateRequest(BaseModel):
    """Request to get physics state of an object."""

    object_name: str = Field(..., description="Name of the object")


class BlenderGetPhysicsStateResponse(BaseModel):
    """Response with physics state readback."""

    success: bool = Field(..., description="Whether query succeeded")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    object_name: str = Field(..., description="Name of the object")
    location: List[float] = Field(
        ...,
        description="World-space XYZ position",
        min_length=3,
        max_length=3,
    )
    rotation_euler: List[float] = Field(
        ...,
        description="Rotation as Euler angles (radians)",
        min_length=3,
        max_length=3,
    )
    is_active: bool = Field(..., description="Whether object has active rigid body")
    has_rigid_body: bool = Field(
        ..., description="Whether object has rigid body physics"
    )
    mass: Optional[float] = Field(None, description="Mass if rigid body exists")
    collision_shape: Optional[str] = Field(
        None, description="Collision shape if rigid body exists"
    )


class BlenderTrajectoryPoint(BaseModel):
    """Single point in a trajectory."""

    frame: int = Field(..., description="Frame number")
    time: float = Field(..., description="Time in seconds")
    location: List[float] = Field(
        ...,
        description="XYZ position",
        min_length=3,
        max_length=3,
    )
    rotation_euler: List[float] = Field(
        ...,
        description="Euler rotation (radians)",
        min_length=3,
        max_length=3,
    )
    velocity: Optional[List[float]] = Field(
        None,
        description="Estimated velocity (computed from position delta)",
        min_length=3,
        max_length=3,
    )


class BlenderGetTrajectoryRequest(BaseModel):
    """Request to get object trajectory over a frame range."""

    object_name: str = Field(..., description="Name of the object to track")
    start_frame: int = Field(..., description="First frame of trajectory")
    end_frame: int = Field(..., description="Last frame of trajectory")
    step: int = Field(1, description="Frame step size")


class BlenderGetTrajectoryResponse(BaseModel):
    """Response with trajectory data."""

    success: bool = Field(..., description="Whether query succeeded")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    object_name: str = Field(..., description="Name of the tracked object")
    point_count: int = Field(..., description="Number of trajectory points")
    points: List[BlenderTrajectoryPoint] = Field(
        default_factory=list, description="Trajectory points"
    )


class BlenderBakeSimulationRequest(BaseModel):
    """Request to bake physics simulation."""

    frame_start: int = Field(..., description="First frame to bake")
    frame_end: int = Field(..., description="Last frame to bake")


class BlenderBakeSimulationResponse(BaseModel):
    """Response after baking simulation."""

    success: bool = Field(..., description="Whether bake succeeded")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    frame_start: int = Field(..., description="First baked frame")
    frame_end: int = Field(..., description="Last baked frame")


class BlenderFreeBakeResponse(BaseModel):
    """Response after freeing baked simulation data."""

    success: bool = Field(..., description="Whether free bake succeeded")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )


# ---------------------------------------------------------------------------
# End Blender physics schemas
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Blender scripting & mesh-from-data schemas
# ---------------------------------------------------------------------------


class BlenderExecuteScriptRequest(BaseModel):
    """Request to execute arbitrary bpy Python code in Blender."""

    script: str = Field(
        ...,
        description="Python script to execute inside Blender (has access to bpy)",
        min_length=1,
    )
    timeout: Optional[float] = Field(
        None,
        description=("Maximum execution time in seconds. " "None means no timeout."),
        ge=0.1,
    )


class BlenderExecuteScriptResponse(BaseModel):
    """Response after executing a Blender script."""

    success: bool = Field(..., description="Whether the script executed without error")
    output: Optional[str] = Field(None, description="Captured stdout from the script")
    return_value: Optional[str] = Field(
        None,
        description=(
            "String repr of the last expression value, if the script "
            "assigned to __result__"
        ),
    )
    error: Optional[str] = Field(None, description="Error message if execution failed")
    duration_seconds: float = Field(
        ..., description="Wall-clock execution time in seconds"
    )
    timed_out: bool = Field(
        False,
        description=(
            "True when the script exceeded its timeout; it keeps running on a "
            "background thread and its output is discarded"
        ),
    )


class BlenderCreateMeshFromDataRequest(BaseModel):
    """Request to create a mesh object from raw vertex/edge/face data."""

    name: str = Field(
        ...,
        description="Name for the new mesh object",
        min_length=1,
    )
    vertices: List[List[float]] = Field(
        ...,
        description="List of vertex positions, each [x, y, z]",
        min_length=1,
    )
    edges: List[List[int]] = Field(
        default_factory=list,
        description="List of edges, each [v_index_a, v_index_b]",
    )
    faces: List[List[int]] = Field(
        default_factory=list,
        description="List of faces, each a list of vertex indices",
    )
    location: Optional[List[float]] = Field(
        None,
        description="World-space location [x, y, z] for the object origin",
        min_length=3,
        max_length=3,
    )
    collection_name: Optional[str] = Field(
        None,
        description=(
            "Target collection name. Links to the active scene " "collection when None."
        ),
    )


class BlenderCreateMeshFromDataResponse(BaseModel):
    """Response after creating a mesh from vertex/edge/face data."""

    success: bool = Field(..., description="Whether the mesh was created")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    object_name: str = Field(..., description="Final Blender object name")
    mesh_name: str = Field(..., description="Final Blender mesh data-block name")
    vertex_count: int = Field(..., description="Number of vertices created")
    edge_count: int = Field(..., description="Number of edges created")
    face_count: int = Field(..., description="Number of faces created")

__all__ = [
    "BlenderInfoResponse",
    "BlenderObjectInfo",
    "BlenderSceneObjectsRequest",
    "BlenderSceneObjectsResponse",
    "BlenderObjectInfoRequest",
    "BlenderModifierEntry",
    "BlenderConstraintEntry",
    "BlenderMaterialSlotEntry",
    "BlenderObjectInfoResponse",
    "BlenderMeshInfoRequest",
    "BlenderMeshInfoResponse",
    "BlenderBoundingBoxRequest",
    "BlenderBoundingBoxResponse",
    "BlenderSearchObjectsRequest",
    "BlenderSearchObjectsResponse",
    "BlenderSceneSummaryResponse",
    "BlenderMaterialInfoRequest",
    "BlenderNodeEntry",
    "BlenderMaterialInfoResponse",
    "BlenderDistanceRequest",
    "BlenderDistanceResponse",
    "BlenderBoundsCheckRequest",
    "BlenderBoundsCheckResponse",
    "BlenderCaptureViewportRequest",
    "BlenderCaptureViewportResponse",
    "BlenderSetCameraViewRequest",
    "BlenderSetCameraViewResponse",
    "BlenderCameraInfoResponse",
    "BlenderFocusOnObjectRequest",
    "BlenderFocusOnObjectResponse",
    "BlenderViewportInfoResponse",
    "BlenderCaptureSequenceRequest",
    "BlenderCaptureSequenceResponse",
    "BlenderCreateObjectRequest",
    "BlenderCreateObjectResponse",
    "BlenderDeleteObjectRequest",
    "BlenderDeleteObjectResponse",
    "BlenderSetTransformRequest",
    "BlenderSetTransformResponse",
    "BlenderSetParentRequest",
    "BlenderSetParentResponse",
    "BlenderClearParentRequest",
    "BlenderClearParentResponse",
    "BlenderAssignMaterialRequest",
    "BlenderAssignMaterialResponse",
    "BlenderAddModifierRequest",
    "BlenderAddModifierResponse",
    "BlenderSetLightParamsRequest",
    "BlenderSetLightParamsResponse",
    "BlenderOpenFileRequest",
    "BlenderOpenFileResponse",
    "BlenderSaveFileRequest",
    "BlenderSaveFileResponse",
    "BlenderImportFileRequest",
    "BlenderImportFileResponse",
    "BlenderExportFileRequest",
    "BlenderExportFileResponse",
    "BlenderFileInfoResponse",
    "BlenderSetFrameRequest",
    "BlenderSetFrameResponse",
    "BlenderGetFrameResponse",
    "BlenderSetFrameRangeRequest",
    "BlenderSetFrameRangeResponse",
    "BlenderPlayAnimationRequest",
    "BlenderPlayAnimationResponse",
    "BlenderInsertKeyframeRequest",
    "BlenderInsertKeyframeResponse",
    "BlenderDeleteKeyframeRequest",
    "BlenderDeleteKeyframeResponse",
    "BlenderGetKeyframesRequest",
    "BlenderKeyframeSummaryEntry",
    "BlenderGetKeyframesResponse",
    "BlenderSetupRigidBodyRequest",
    "BlenderSetupRigidBodyResponse",
    "BlenderAddForceFieldRequest",
    "BlenderAddForceFieldResponse",
    "BlenderGetForceFieldInfoRequest",
    "BlenderGetForceFieldInfoResponse",
    "BlenderAddConstraintRequest",
    "BlenderAddConstraintResponse",
    "BlenderGetConstraintInfoRequest",
    "BlenderGetConstraintInfoResponse",
    "BlenderGetPhysicsStateRequest",
    "BlenderGetPhysicsStateResponse",
    "BlenderTrajectoryPoint",
    "BlenderGetTrajectoryRequest",
    "BlenderGetTrajectoryResponse",
    "BlenderBakeSimulationRequest",
    "BlenderBakeSimulationResponse",
    "BlenderFreeBakeResponse",
    "BlenderExecuteScriptRequest",
    "BlenderExecuteScriptResponse",
    "BlenderCreateMeshFromDataRequest",
    "BlenderCreateMeshFromDataResponse",
]
