"""
Pydantic schemas for Isaac Sim MCP Server tool inputs and outputs.

This module defines the data models used for MCP tool communication,
ensuring type safety and validation for all tool operations.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class PrimType(str, Enum):
    """Common USD prim types."""

    UNKNOWN = "unknown"
    XFORM = "Xform"
    MESH = "Mesh"
    SPHERE = "Sphere"
    CUBE = "Cube"
    CYLINDER = "Cylinder"
    CONE = "Cone"
    PLANE = "Plane"
    CAMERA = "Camera"
    LIGHT = "Light"
    MATERIAL = "Material"
    SHADER = "Shader"
    SCOPE = "Scope"


class BoundingBox(BaseModel):
    """Bounding box representation."""

    min: List[float] = Field(
        ..., description="Minimum point [x, y, z]", min_items=3, max_items=3
    )
    max: List[float] = Field(
        ..., description="Maximum point [x, y, z]", min_items=3, max_items=3
    )
    center: Optional[List[float]] = Field(
        None, description="Center point [x, y, z]", min_items=3, max_items=3
    )
    size: Optional[List[float]] = Field(
        None, description="Size [width, height, depth]", min_items=3, max_items=3
    )
    volume: Optional[float] = Field(None, description="Bounding box volume")


class Transform(BaseModel):
    """3D transformation representation."""

    translation: List[float] = Field(
        ..., description="Translation [x, y, z]", min_items=3, max_items=3
    )
    rotation: List[float] = Field(
        ..., description="Rotation quaternion [w, x, y, z]", min_items=4, max_items=4
    )
    scale: List[float] = Field(
        ..., description="Scale [x, y, z]", min_items=3, max_items=3
    )


class USDFileInfo(BaseModel):
    """USD file information."""

    file_path: str = Field(..., description="Path to USD file")
    file_size: int = Field(..., description="File size in bytes")
    format: str = Field(..., description="USD file format (usd, usda, usdc, usdz)")
    is_valid: bool = Field(..., description="Whether file is valid USD")
    can_read: bool = Field(..., description="Whether file is readable")


class USDFileRequest(BaseModel):
    """Request for USD file operations."""

    file_path: str = Field(..., description="Path to USD file", min_length=1)


class USDValidateRequest(BaseModel):
    """Request to validate a USD file."""

    file_path: str = Field(..., description="Path to USD file", min_length=1)


class StageInfo(BaseModel):
    """USD stage information."""

    stage_id: str = Field(..., description="Unique stage identifier")
    file_path: str = Field(..., description="Path to USD file")
    up_axis: str = Field(..., description="Stage up axis (Y or Z)")
    meters_per_unit: float = Field(..., description="Meters per unit scale")
    time_codes_per_second: float = Field(..., description="Time codes per second")
    start_time: float = Field(..., description="Start time code")
    end_time: float = Field(..., description="End time code")
    frame_rate: float = Field(..., description="Frame rate")
    total_prims: int = Field(..., description="Total number of prims")
    root_prims: List[str] = Field(..., description="Root prim paths")
    has_animation: bool = Field(..., description="Whether stage has animation")
    layer_count: int = Field(..., description="Number of layers")
    default_prim: Optional[str] = Field(None, description="Default prim path")


class PrimInfo(BaseModel):
    """USD prim information."""

    path: str = Field(..., description="Prim path")
    name: str = Field(..., description="Prim name")
    type: str = Field(..., description="Prim type")
    is_active: bool = Field(..., description="Whether prim is active")
    is_loaded: bool = Field(..., description="Whether prim is loaded")
    is_defined: bool = Field(..., description="Whether prim is defined")
    is_instance: bool = Field(..., description="Whether prim is an instance")
    purpose: Optional[str] = Field(None, description="Prim purpose")
    visibility: Optional[str] = Field(None, description="Prim visibility")
    kind: Optional[str] = Field(None, description="Prim kind")
    bbox: Optional[BoundingBox] = Field(None, description="Bounding box")
    transform: Optional[Transform] = Field(None, description="Local transformation")
    children_count: int = Field(..., description="Number of child prims")
    children_types: Dict[str, int] = Field(..., description="Child prim type counts")
    material_bindings: List[str] = Field(..., description="Bound material paths")
    attributes: Dict[str, Any] = Field(..., description="Important attributes")
    metadata: Dict[str, Any] = Field(..., description="Prim metadata")


class PrimInfoRequest(BaseModel):
    """Request for prim details."""

    stage_id: str = Field(..., description="Stage identifier", min_length=1)
    prim_path: str = Field(..., description="Prim path", min_length=1)


class MeshInfoRequest(BaseModel):
    """Request for mesh details."""

    stage_id: str = Field(..., description="Stage identifier", min_length=1)
    prim_path: str = Field(..., description="Mesh prim path", min_length=1)


class PrimCreateRequest(BaseModel):
    stage_id: str = Field(..., description="Stage identifier", min_length=1)
    prim_path: str = Field(..., description="Prim path", min_length=1)
    prim_type: str = Field(..., description="USD prim type", min_length=1)
    attributes: Dict[str, Any] = Field(
        default_factory=dict, description="Attribute values"
    )


class PrimUpdateRequest(BaseModel):
    stage_id: str = Field(..., description="Stage identifier", min_length=1)
    prim_path: str = Field(..., description="Prim path", min_length=1)
    attributes: Dict[str, Any] = Field(
        default_factory=dict, description="Attribute values"
    )


class PrimDeleteRequest(BaseModel):
    stage_id: str = Field(..., description="Stage identifier", min_length=1)
    prim_path: str = Field(..., description="Prim path", min_length=1)


class MeshInfo(BaseModel):
    """USD mesh information."""

    prim_path: str = Field(..., description="Mesh prim path")
    vertex_count: int = Field(..., description="Number of vertices")
    face_count: int = Field(..., description="Number of faces")
    point_count: int = Field(..., description="Number of face vertex indices")
    has_normals: bool = Field(..., description="Whether mesh has normals")
    has_uvs: bool = Field(..., description="Whether mesh has UV coordinates")
    has_colors: bool = Field(..., description="Whether mesh has vertex colors")
    has_tangents: bool = Field(..., description="Whether mesh has tangents")
    subdivision_scheme: str = Field(..., description="Subdivision scheme")
    topology_valid: bool = Field(..., description="Whether topology is valid")
    is_closed: bool = Field(..., description="Whether mesh is watertight")
    surface_area: float = Field(..., description="Surface area")
    volume: float = Field(..., description="Volume (if closed)")
    bbox: BoundingBox = Field(..., description="Mesh bounding box")
    materials: List[str] = Field(..., description="Material paths")
    subsets: List[Dict[str, Any]] = Field(..., description="Geometry subsets")


class BlenderInfoResponse(BaseModel):
    """Response with Blender runtime information."""

    success: bool = Field(..., description="Whether request was successful")
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
    name: str = Field(..., description="Object name")
    object_type: str = Field(..., description="Blender object type")
    location: List[float] = Field(
        ..., description="World location [x, y, z]", min_items=3, max_items=3
    )
    rotation_euler: List[float] = Field(
        ..., description="Rotation in Euler radians [x, y, z]", min_items=3, max_items=3
    )
    scale: List[float] = Field(
        ..., description="Scale [x, y, z]", min_items=3, max_items=3
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
    object_name: str = Field(..., description="Object name")
    corners: List[List[float]] = Field(
        ..., description="Eight bounding box corners as [x, y, z] lists"
    )
    bbox_min: List[float] = Field(
        ..., description="Axis-aligned minimum [x, y, z]", min_items=3, max_items=3
    )
    bbox_max: List[float] = Field(
        ..., description="Axis-aligned maximum [x, y, z]", min_items=3, max_items=3
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
    pattern: Optional[str] = Field(None, description="Name pattern used")
    object_type: Optional[str] = Field(None, description="Type filter used")
    count: int = Field(..., description="Number of matching objects")
    objects: List[BlenderObjectInfo] = Field(..., description="Matching objects")
    truncated: bool = Field(..., description="Whether results were capped")


class BlenderSceneSummaryResponse(BaseModel):
    """High-level scene summary grouped by type."""

    success: bool = Field(..., description="Whether request was successful")
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
    object_name_a: str = Field(..., description="First object name")
    object_name_b: str = Field(..., description="Second object name")
    distance: float = Field(..., description="Euclidean distance between objects")
    location_a: List[float] = Field(
        ...,
        description="World location of first object [x, y, z]",
        min_items=3,
        max_items=3,
    )
    location_b: List[float] = Field(
        ...,
        description="World location of second object [x, y, z]",
        min_items=3,
        max_items=3,
    )


class BlenderBoundsCheckRequest(BaseModel):
    """Request to check if an object is within spatial bounds."""

    object_name: str = Field(..., description="Object name to check")
    bounds_min: List[float] = Field(
        ..., description="Minimum bounds [x, y, z]", min_items=3, max_items=3
    )
    bounds_max: List[float] = Field(
        ..., description="Maximum bounds [x, y, z]", min_items=3, max_items=3
    )


class BlenderBoundsCheckResponse(BaseModel):
    """Result of a spatial bounds check."""

    success: bool = Field(..., description="Whether request was successful")
    object_name: str = Field(..., description="Object name checked")
    within_bounds: bool = Field(
        ..., description="Whether object location is within the bounds"
    )
    object_location: List[float] = Field(
        ...,
        description="Object world location [x, y, z]",
        min_items=3,
        max_items=3,
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
        ..., description="Camera location [x, y, z]", min_items=3, max_items=3
    )
    rotation_euler: List[float] = Field(
        ...,
        description="Camera rotation [rx, ry, rz] in radians",
        min_items=3,
        max_items=3,
    )
    camera_name: Optional[str] = Field(
        None, description="Target camera name. Uses active camera when None."
    )


class BlenderSetCameraViewResponse(BaseModel):
    """Result of setting a camera view."""

    success: bool = Field(..., description="Whether operation succeeded")
    camera_name: str = Field(..., description="Name of the camera that was updated")
    location: List[float] = Field(
        ...,
        description="New camera location [x, y, z]",
        min_items=3,
        max_items=3,
    )
    rotation_euler: List[float] = Field(
        ...,
        description="New camera rotation [rx, ry, rz]",
        min_items=3,
        max_items=3,
    )


class BlenderCameraInfoResponse(BaseModel):
    """Active camera information."""

    success: bool = Field(..., description="Whether request succeeded")
    camera_name: str = Field(..., description="Active camera name")
    location: List[float] = Field(
        ...,
        description="Camera location [x, y, z]",
        min_items=3,
        max_items=3,
    )
    rotation_euler: List[float] = Field(
        ...,
        description="Camera rotation [rx, ry, rz]",
        min_items=3,
        max_items=3,
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
    camera_name: str = Field(..., description="Camera that was updated")
    object_name: str = Field(..., description="Object that was focused on")
    camera_location: List[float] = Field(
        ...,
        description="New camera location [x, y, z]",
        min_items=3,
        max_items=3,
    )
    look_at: List[float] = Field(
        ...,
        description="Point the camera is aimed at [x, y, z]",
        min_items=3,
        max_items=3,
    )


class BlenderViewportInfoResponse(BaseModel):
    """Active viewport / render settings summary."""

    success: bool = Field(..., description="Whether request succeeded")
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
        min_items=3,
        max_items=3,
    )
    rotation_euler: List[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        description="Initial rotation [rx, ry, rz] radians",
        min_items=3,
        max_items=3,
    )
    scale: List[float] = Field(
        default_factory=lambda: [1.0, 1.0, 1.0],
        description="Initial scale [sx, sy, sz]",
        min_items=3,
        max_items=3,
    )


class BlenderCreateObjectResponse(BaseModel):
    """Response after creating a Blender object."""

    success: bool = Field(..., description="Whether creation succeeded")
    name: str = Field(..., description="Resulting object name")
    object_type: str = Field(..., description="Created object type")
    location: List[float] = Field(
        ...,
        description="Object location [x, y, z]",
        min_items=3,
        max_items=3,
    )


class BlenderDeleteObjectRequest(BaseModel):
    """Request to delete an object from the scene."""

    object_name: str = Field(..., description="Name of object to delete")


class BlenderDeleteObjectResponse(BaseModel):
    """Response after deleting a Blender object."""

    success: bool = Field(..., description="Whether deletion succeeded")
    deleted_name: str = Field(..., description="Name of deleted object")


class BlenderSetTransformRequest(BaseModel):
    """Request to set an object's transform."""

    object_name: str = Field(..., description="Object to transform")
    location: Optional[List[float]] = Field(
        None,
        description="New location [x, y, z]",
        min_items=3,
        max_items=3,
    )
    rotation_euler: Optional[List[float]] = Field(
        None,
        description="New rotation [rx, ry, rz] radians",
        min_items=3,
        max_items=3,
    )
    scale: Optional[List[float]] = Field(
        None,
        description="New scale [sx, sy, sz]",
        min_items=3,
        max_items=3,
    )


class BlenderSetTransformResponse(BaseModel):
    """Response after setting an object's transform."""

    success: bool = Field(..., description="Whether transform succeeded")
    object_name: str = Field(..., description="Object name")
    location: List[float] = Field(
        ...,
        description="Final location [x, y, z]",
        min_items=3,
        max_items=3,
    )
    rotation_euler: List[float] = Field(
        ...,
        description="Final rotation [rx, ry, rz]",
        min_items=3,
        max_items=3,
    )
    scale: List[float] = Field(
        ...,
        description="Final scale [sx, sy, sz]",
        min_items=3,
        max_items=3,
    )


class BlenderSetParentRequest(BaseModel):
    """Request to parent one object to another."""

    child_name: str = Field(..., description="Object to parent")
    parent_name: str = Field(..., description="Object to be the parent")


class BlenderSetParentResponse(BaseModel):
    """Response after parenting objects."""

    success: bool = Field(..., description="Whether parenting succeeded")
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
    file_path: str = Field(..., description="Path the file was exported to")
    file_format: str = Field(..., description="Format that was exported")


class BlenderFileInfoResponse(BaseModel):
    """Response with current file information."""

    success: bool = Field(..., description="Whether info retrieval succeeded")
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
    frame: int = Field(..., description="The frame that was set")


class BlenderGetFrameResponse(BaseModel):
    """Response with current frame and range information."""

    success: bool = Field(..., description="Whether retrieval succeeded")
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
        min_items=2,
        max_items=2,
    )


class BlenderGetKeyframesResponse(BaseModel):
    """Response with keyframe summary for an object."""

    success: bool = Field(..., description="Whether retrieval succeeded")
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
        min_items=3,
        max_items=3,
    )
    name: Optional[str] = Field(None, description="Optional name for the field")


class BlenderAddForceFieldResponse(BaseModel):
    """Response after adding a force field."""

    success: bool = Field(..., description="Whether creation succeeded")
    name: str = Field(..., description="Name of the created force field object")
    field_type: str = Field(..., description="Type of force field")
    strength: float = Field(..., description="Strength of the field")
    location: List[float] = Field(
        ...,
        description="Location of the field [x, y, z]",
        min_items=3,
        max_items=3,
    )


class BlenderGetForceFieldInfoRequest(BaseModel):
    """Request to get force field info."""

    object_name: str = Field(..., description="Name of the force field object")


class BlenderGetForceFieldInfoResponse(BaseModel):
    """Response with force field details."""

    success: bool = Field(..., description="Whether query succeeded")
    object_name: str = Field(..., description="Name of the force field object")
    field_type: str = Field(..., description="Force field type")
    strength: float = Field(..., description="Field strength")
    shape: str = Field(..., description="Field shape (POINT, PLANE, etc.)")
    flow: float = Field(..., description="Field flow value")
    location: List[float] = Field(
        ...,
        description="World-space location [x, y, z]",
        min_items=3,
        max_items=3,
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
        min_items=3,
        max_items=3,
    )
    disable_collisions: bool = Field(
        True, description="Disable collisions between constrained objects"
    )


class BlenderAddConstraintResponse(BaseModel):
    """Response after adding a rigid body constraint."""

    success: bool = Field(..., description="Whether constraint was created")
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
    object_name: str = Field(..., description="Name of the object")
    location: List[float] = Field(
        ...,
        description="World-space XYZ position",
        min_items=3,
        max_items=3,
    )
    rotation_euler: List[float] = Field(
        ...,
        description="Rotation as Euler angles (radians)",
        min_items=3,
        max_items=3,
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
        min_items=3,
        max_items=3,
    )
    rotation_euler: List[float] = Field(
        ...,
        description="Euler rotation (radians)",
        min_items=3,
        max_items=3,
    )
    velocity: Optional[List[float]] = Field(
        None,
        description="Estimated velocity (computed from position delta)",
        min_items=3,
        max_items=3,
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
    frame_start: int = Field(..., description="First baked frame")
    frame_end: int = Field(..., description="Last baked frame")


class BlenderFreeBakeResponse(BaseModel):
    """Response after freeing baked simulation data."""

    success: bool = Field(..., description="Whether free bake succeeded")


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
        min_items=1,
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
        min_items=3,
        max_items=3,
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
    object_name: str = Field(..., description="Final Blender object name")
    mesh_name: str = Field(..., description="Final Blender mesh data-block name")
    vertex_count: int = Field(..., description="Number of vertices created")
    edge_count: int = Field(..., description="Number of edges created")
    face_count: int = Field(..., description="Number of faces created")


# ── SimReady Asset Format Models ──────────────────────────────────────────────


class SimReadySemanticLabels(BaseModel):
    """
    Semantic labeling for SimReady assets.

    Follows the USDSemanticLabels API with class, hierarchy, and qcode fields.
    """

    semantic_class: str = Field(
        ..., description="Human-readable classification, e.g. 'cup', 'truck'"
    )
    semantic_hierarchy: Optional[str] = Field(
        None,
        description=(
            "Ordered relational hierarchy, "
            "e.g. 'machine/vehicle/emergency_vehicle/fire_engine'"
        ),
    )
    semantic_qcode: Optional[str] = Field(
        None, description="WikiData Q-Code identifier, e.g. 'Q1420'"
    )
    additional_labels: Optional[Dict[str, str]] = Field(
        None, description="Custom labels such as SKU or ProductID"
    )


class SimReadyPhysicsProperties(BaseModel):
    """
    Physics properties for SimReady assets.

    Maps to USDPhysics schema: collider, mass, physical material, rigid body.
    """

    mass_kg: Optional[float] = Field(None, ge=0.0, description="Mass in kilograms")
    collider_type: Optional[str] = Field(
        None,
        description="Collision shape: convexHull, mesh, box, sphere, capsule",
    )
    static_friction: Optional[float] = Field(
        None, ge=0.0, description="Static friction coefficient"
    )
    dynamic_friction: Optional[float] = Field(
        None, ge=0.0, description="Dynamic friction coefficient"
    )
    restitution: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Restitution (bounciness)"
    )
    density: Optional[float] = Field(
        None, ge=0.0, description="Material density in kg/m³"
    )
    is_rigid_body: bool = Field(
        False, description="Whether the root prim is a rigid body"
    )

    @field_validator("collider_type")
    @classmethod
    def validate_collider_type(cls, v: Optional[str]) -> Optional[str]:
        """Validate collider type against allowed values."""
        if v is not None:
            allowed = {"convexHull", "mesh", "box", "sphere", "capsule"}
            if v not in allowed:
                raise ValueError(
                    f"collider_type must be one of {sorted(allowed)}, got '{v}'"
                )
        return v


class SimReadyMaterialProperties(BaseModel):
    """
    Material properties for SimReady assets.

    Follows SimReady material naming and shader conventions.
    """

    substrate_type: Optional[str] = Field(
        None,
        description=(
            "Physical substrate: metal, wood, plastic, glass, "
            "rubber, fabric, ceramic, concrete, stone"
        ),
    )
    material_naming: Optional[str] = Field(
        None,
        description=(
            "SimReady material name: prefix_surfacetype_description, "
            "e.g. 'opaque_metal_brushed_aluminum'"
        ),
    )
    shader_type: Optional[str] = Field(
        None,
        description="Target shader: OmniPBR, OmniGlass, SimPBR",
    )
    texel_density: Optional[float] = Field(
        None, gt=0.0, description="Target texel density in pixels per meter"
    )


class SimReadyMetadata(BaseModel):
    """
    Combined SimReady metadata for an asset.

    Groups semantic, physics, and material properties into a single model
    that is stored as Blender custom properties with a ``simready_`` prefix
    and carried forward into USD export.
    """

    semantic: Optional[SimReadySemanticLabels] = Field(
        None, description="Semantic labeling data"
    )
    physics: Optional[SimReadyPhysicsProperties] = Field(
        None, description="Physics properties"
    )
    material: Optional[SimReadyMaterialProperties] = Field(
        None, description="Material properties"
    )


# ── SimReady Request / Response Schemas ──────────────────────────────────────


class SimReadyApplyMetadataRequest(BaseModel):
    """Request to apply SimReady metadata to a Blender object."""

    object_name: str = Field(..., description="Name of the Blender object")
    metadata: SimReadyMetadata = Field(..., description="SimReady metadata to apply")


class SimReadyApplyMetadataResponse(BaseModel):
    """Response after applying SimReady metadata."""

    success: bool = Field(..., description="Whether metadata was applied")
    object_name: str = Field(..., description="Blender object name")
    applied_properties: List[str] = Field(
        ..., description="List of custom property keys written"
    )


class SimReadyGetMetadataRequest(BaseModel):
    """Request to read SimReady metadata from a Blender object."""

    object_name: str = Field(..., description="Name of the Blender object")


class SimReadyGetMetadataResponse(BaseModel):
    """Response containing SimReady metadata for an object."""

    success: bool = Field(...)
    object_name: str = Field(...)
    metadata: Optional[SimReadyMetadata] = Field(
        None, description="SimReady metadata (null if none found)"
    )
    has_simready_data: bool = Field(
        ..., description="Whether any simready_ properties exist"
    )


class SimReadyValidationIssue(BaseModel):
    """A single compliance issue found during validation."""

    object_name: str = Field(..., description="Object with the issue")
    check: str = Field(
        ...,
        description="Check category: naming, scale, transforms, materials, hierarchy",
    )
    severity: str = Field(..., description="Issue severity: error or warning")
    message: str = Field(..., description="Human-readable description")
    suggestion: Optional[str] = Field(None, description="Suggested fix")


class SimReadyValidateRequest(BaseModel):
    """Request to validate objects against SimReady conventions."""

    object_names: Optional[List[str]] = Field(
        None, description="Objects to check (null = all scene objects)"
    )
    check_naming: bool = Field(True, description="Check lowercase_underscore naming")
    check_scale: bool = Field(True, description="Check real-world meter scale")
    check_transforms: bool = Field(True, description="Check for clean transforms")
    check_materials: bool = Field(True, description="Check material segmentation")
    check_hierarchy: bool = Field(True, description="Check hierarchy structure")


class SimReadyValidateResponse(BaseModel):
    """Response from SimReady compliance validation."""

    success: bool = Field(...)
    compliant: bool = Field(..., description="True when zero errors found")
    object_count: int = Field(..., description="Number of objects checked")
    issue_count: int = Field(..., description="Total issues found")
    issues: List[SimReadyValidationIssue] = Field(
        default_factory=list, description="Detailed issue list"
    )


class SimReadyExportRequest(BaseModel):
    """Request to export a SimReady-compliant USD file."""

    file_path: str = Field(..., description="Output .usd / .usda / .usdc file path")
    object_names: Optional[List[str]] = Field(
        None, description="Objects to export (null = all scene objects)"
    )
    embed_metadata: bool = Field(
        True, description="Embed simready_ custom properties in USD"
    )
    validate_before_export: bool = Field(
        True, description="Run validation before exporting"
    )


class SimReadyExportResponse(BaseModel):
    """Response after exporting SimReady USD."""

    success: bool = Field(...)
    file_path: str = Field(..., description="Written file path")
    object_count: int = Field(..., description="Number of objects exported")
    validation_passed: bool = Field(
        ..., description="Whether pre-export validation passed"
    )
    issues: Optional[List[SimReadyValidationIssue]] = Field(
        None, description="Validation issues (if validation was run)"
    )


class SimReadySetupHierarchyRequest(BaseModel):
    """Request to create a SimReady-compliant object hierarchy."""

    root_name: str = Field(
        ..., description="Name for the root empty (XForm equivalent)"
    )
    child_names: List[str] = Field(
        ..., description="Existing objects to parent under root"
    )
    semantic: Optional[SimReadySemanticLabels] = Field(
        None, description="Semantic labels for the root"
    )


class SimReadySetupHierarchyResponse(BaseModel):
    """Response after setting up SimReady hierarchy."""

    success: bool = Field(...)
    root_name: str = Field(..., description="Root empty name")
    children: List[str] = Field(..., description="Successfully parented children")
    hierarchy_path: str = Field(..., description="Logical hierarchy path from root")


class FocusPrimResponse(BaseModel):
    """Response from focusing camera on a prim."""

    success: bool = Field(..., description="Whether focus succeeded")
    stage_id: str = Field(..., description="Stage identifier")
    prim_path: str = Field(..., description="Prim path")
    focus_point: List[float] = Field(
        ..., description="Focus point [x, y, z]", min_items=3, max_items=3
    )
    camera_position: List[float] = Field(
        ..., description="Camera position [x, y, z]", min_items=3, max_items=3
    )
    message: str = Field(..., description="Status message")


class PrimActionResponse(BaseModel):
    success: bool = Field(..., description="Whether request was successful")
    stage_id: str = Field(..., description="Stage identifier")
    prim_path: str = Field(..., description="Prim path")
    message: str = Field(..., description="Status message")


class SceneSummaryRequest(BaseModel):
    """Request for scene summary."""

    stage_id: str = Field(..., description="Stage identifier")
    include_meshes: bool = Field(True, description="Include detailed mesh information")
    include_materials: bool = Field(True, description="Include material information")
    max_depth: int = Field(5, description="Maximum hierarchy depth", ge=1, le=10)
    format: str = Field("json", description="Output format (json, text)")

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        valid_formats = ["json", "text"]
        if v not in valid_formats:
            raise ValueError(f"Format must be one of {valid_formats}")
        return v


class SceneSummaryResponse(BaseModel):
    """Response from scene summary."""

    success: bool = Field(..., description="Whether summary was successful")
    stage_id: str = Field(..., description="Stage identifier")
    summary: Dict[str, Any] = Field(..., description="Scene summary data")
    digest: Optional[str] = Field(None, description="Human-readable digest")
    error: Optional[str] = Field(None, description="Error message if failed")


class PrimSearchRequest(BaseModel):
    """Request to search for prims."""

    stage_id: str = Field(..., description="Stage identifier")
    search_type: str = Field(..., description="Search type (by_type, by_name)")
    query: str = Field(..., description="Search query")
    exact_match: bool = Field(False, description="Use exact matching (for name search)")

    @field_validator("search_type")
    @classmethod
    def validate_search_type(cls, v: str) -> str:
        valid_types = ["by_type", "by_name"]
        if v not in valid_types:
            raise ValueError(f"Search type must be one of {valid_types}")
        return v


class PrimSearchResponse(BaseModel):
    """Response from prim search."""

    success: bool = Field(..., description="Whether search was successful")
    stage_id: str = Field(..., description="Stage identifier")
    search_type: str = Field(..., description="Search type used")
    query: str = Field(..., description="Search query")
    results: List[str] = Field(..., description="List of matching prim paths")
    count: int = Field(..., description="Number of results")
    error: Optional[str] = Field(None, description="Error message if failed")


class BBoxRequest(BaseModel):
    """Request for bounding box computation."""

    stage_id: str = Field(..., description="Stage identifier")
    prim_path: Optional[str] = Field(
        None, description="Prim path (None for stage bbox)"
    )
    world_space: bool = Field(True, description="Compute in world space")
    time_code: Optional[float] = Field(None, description="Time code for animated data")


class BBoxResponse(BaseModel):
    """Response from bounding box computation."""

    success: bool = Field(..., description="Whether computation was successful")
    stage_id: str = Field(..., description="Stage identifier")
    prim_path: Optional[str] = Field(None, description="Prim path")
    bbox: Optional[BoundingBox] = Field(None, description="Computed bounding box")
    world_space: bool = Field(..., description="Whether bbox is in world space")
    error: Optional[str] = Field(None, description="Error message if failed")


class ErrorResponse(BaseModel):
    """Generic error response."""

    success: bool = Field(False, description="Always false for error responses")
    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Error type")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional error details"
    )


# ---------------------------------------------------------------------------
# Isaac Sim instance discovery schemas
# ---------------------------------------------------------------------------


class IsaacInstanceInfo(BaseModel):
    """Information about a discovered Isaac Sim instance."""

    name: str = Field(..., description="Instance identifier")
    host: str = Field(..., description="TCP socket host")
    port: int = Field(..., description="TCP socket port")
    reachable: bool = Field(..., description="Whether the instance responded to ping")
    active: bool = Field(False, description="Whether this is the currently active instance")
    stage_url: Optional[str] = Field(None, description="URL of the loaded stage")
    up_axis: Optional[str] = Field(None, description="Stage up axis")
    prim_count: Optional[int] = Field(None, description="Total prim count on stage")
    is_playing: Optional[bool] = Field(None, description="Whether simulation is playing")


class ListIsaacInstancesResponse(BaseModel):
    """Response from list_isaac_instances tool."""

    success: bool = Field(True, description="Whether discovery completed")
    instances: List[IsaacInstanceInfo] = Field(default_factory=list, description="Discovered instances")
    active_instance: Optional[str] = Field(None, description="Name of the currently active instance")
    total_discovered: int = Field(0, description="Number of reachable instances found")


class SetActiveInstanceResponse(BaseModel):
    """Response from set_active_isaac_instance tool."""

    success: bool = Field(..., description="Whether the switch succeeded")
    active_instance: str = Field(..., description="Name of the now-active instance")
    address: str = Field(..., description="host:port of the now-active instance")
    message: str = Field("", description="Status message")


# ---------------------------------------------------------------------------
# Unreal Engine schemas — Phase 0
# ---------------------------------------------------------------------------


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
    error: Optional[str] = Field(None, description="Error message if not connected")


class UnrealEngineInfoResponse(BaseModel):
    """Response with Unreal Engine runtime information."""

    success: bool = Field(..., description="Whether request was successful")
    engine_version: str = Field(..., description="Unreal Engine version string")
    project_name: str = Field(..., description="Active project name")
    loaded_map: str = Field(..., description="Currently loaded persistent level path")
    is_editor: bool = Field(..., description="Whether engine is running in editor mode")
    is_game: bool = Field(..., description="Whether engine is running in game mode")
    platform: str = Field(..., description="Platform identifier")


class UnrealLoadedMapResponse(BaseModel):
    """Response with the currently loaded persistent level path."""

    success: bool = Field(..., description="Whether request was successful")
    map_path: str = Field(..., description="Currently loaded persistent level path")


# ---------------------------------------------------------------------------
# Unreal Engine schemas — Phase 1: Scene Read Operations
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
    asset_path: str = Field(..., description="Asset path queried")
    image_base64: str = Field(..., description="Base64-encoded PNG image")
    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")


class UnrealSceneSummaryResponse(BaseModel):
    """LLM-friendly scene digest response."""

    success: bool = Field(..., description="Whether request was successful")
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
# Unreal Phase 2 — Viewport & Visual Observation
# ---------------------------------------------------------------------------


class UnrealCaptureViewportRequest(BaseModel):
    """Request to capture viewport screenshot."""

    resolution_x: int = Field(1920, description="Capture width in pixels")
    resolution_y: int = Field(1080, description="Capture height in pixels")
    format: str = Field("png", description="Image format: png or jpeg")


class UnrealCaptureViewportResponse(BaseModel):
    """Response with captured viewport image."""

    success: bool = Field(..., description="Whether request was successful")
    image_base64: str = Field(..., description="Base64-encoded image data")
    resolution_x: int = Field(..., description="Actual capture width")
    resolution_y: int = Field(..., description="Actual capture height")
    format: str = Field(..., description="Image format used")


class UnrealViewportInfoRequest(BaseModel):
    """Request to get viewport information."""

    pass


class UnrealViewportInfoResponse(BaseModel):
    """Viewport camera and render information."""

    success: bool = Field(..., description="Whether request was successful")
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
    actor_path: str = Field(..., description="Actor that was focused on")
    camera_location: Tuple[float, float, float] = Field(
        ..., description="Resulting camera position"
    )
    camera_rotation: Tuple[float, float, float] = Field(
        ..., description="Resulting camera rotation"
    )


# ---------------------------------------------------------------------------
# Unreal Phase 3 — Scene Manipulation
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
    actor_path: str = Field(..., description="Actor whose visibility was set")
    visible: bool = Field(..., description="Applied visibility state")


# ---------------------------------------------------------------
# Unreal Phase 4 — Materials, Lighting & Rendering
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
    actor_path: str = Field(..., description="Light actor path")
    params_set: int = Field(0, description="Number of params changed")


class UnrealSetRenderSettingsRequest(BaseModel):
    """Request to set rendering/post-process settings."""

    setting_name: str = Field(..., description="Render setting name")
    setting_value: str = Field(..., description="Value as JSON string")


class UnrealSetRenderSettingsResponse(BaseModel):
    """Response after changing render settings."""

    success: bool = Field(True, description="Operation success")
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
    actor_path: str = Field(..., description="Actor that was modified")
    params_set: int = Field(..., description="Number of parameters set")


# ──────────────────────────────────────────────────────────────────────
# Phase 6: USD / SimReady Bridge
# ──────────────────────────────────────────────────────────────────────


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
    pipelines: List[Dict[str, Any]] = Field(
        ..., description="Available import/export pipelines"
    )
    supported_formats: List[str] = Field(..., description="Supported file formats")
    interchange_version: str = Field(..., description="Interchange Framework version")


# ──────────────────────────────────────────────────────────────────────
# Phase 7: Advanced Agent Tools
# ──────────────────────────────────────────────────────────────────────


class UnrealBatchOperationsRequest(BaseModel):
    """Request to execute multiple operations atomically."""

    operations: List[Dict[str, Any]] = Field(
        ..., description="List of operation dicts with endpoint/body pairs"
    )


class UnrealBatchOperationsResponse(BaseModel):
    """Response after batch execution."""

    success: bool = Field(True, description="Overall success")
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
    actors: List[Dict[str, Any]] = Field(
        ..., description="Matching actors with paths and labels"
    )
    total_matches: int = Field(..., description="Number of matches found")
    label_searched: str = Field(..., description="Label that was searched")


# ──────────────────────────────────────────────────────────────────────
# Phase 8: Geometry & Modeling
# ──────────────────────────────────────────────────────────────────────


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
    mesh_path: str = Field(..., description="Mesh path")
    method: str = Field(..., description="UV method used")
    uv_channel: int = Field(..., description="UV channel written")
    island_count: int = Field(..., description="Number of UV islands")
    coverage_ratio: float = Field(..., description="UV space coverage (0.0-1.0)")
    overlap_detected: bool = Field(
        False, description="Whether UV overlaps were detected"
    )


# Tool result types
ToolResult = Union[
    str,  # Simple string response
    Dict[str, Any],  # JSON response
    StageInfo,
    USDFileInfo,
    PrimInfo,
    MeshInfo,
    BlenderInfoResponse,
    BlenderSceneObjectsResponse,
    BlenderObjectInfoResponse,
    BlenderMeshInfoResponse,
    BlenderBoundingBoxResponse,
    BlenderSearchObjectsResponse,
    BlenderSceneSummaryResponse,
    BlenderMaterialInfoResponse,
    BlenderDistanceResponse,
    BlenderBoundsCheckResponse,
    BlenderCaptureViewportResponse,
    BlenderSetCameraViewResponse,
    BlenderCameraInfoResponse,
    BlenderFocusOnObjectResponse,
    BlenderViewportInfoResponse,
    BlenderCaptureSequenceResponse,
    BlenderCreateObjectResponse,
    BlenderDeleteObjectResponse,
    BlenderSetTransformResponse,
    BlenderSetParentResponse,
    BlenderClearParentResponse,
    BlenderAssignMaterialResponse,
    BlenderAddModifierResponse,
    BlenderSetLightParamsResponse,
    BlenderOpenFileResponse,
    BlenderSaveFileResponse,
    BlenderImportFileResponse,
    BlenderExportFileResponse,
    BlenderFileInfoResponse,
    BlenderSetFrameResponse,
    BlenderGetFrameResponse,
    BlenderSetFrameRangeResponse,
    BlenderPlayAnimationResponse,
    BlenderInsertKeyframeResponse,
    BlenderDeleteKeyframeResponse,
    BlenderGetKeyframesResponse,
    BlenderSetupRigidBodyResponse,
    BlenderAddForceFieldResponse,
    BlenderGetForceFieldInfoResponse,
    BlenderAddConstraintResponse,
    BlenderGetConstraintInfoResponse,
    BlenderGetPhysicsStateResponse,
    BlenderGetTrajectoryResponse,
    BlenderBakeSimulationResponse,
    BlenderFreeBakeResponse,
    SimReadyApplyMetadataResponse,
    SimReadyGetMetadataResponse,
    SimReadyValidateResponse,
    SimReadyExportResponse,
    SimReadySetupHierarchyResponse,
    UnrealHealthCheckResponse,
    UnrealEngineInfoResponse,
    UnrealLoadedMapResponse,
    UnrealListActorsResponse,
    UnrealGetActorInfoResponse,
    UnrealSearchAssetsResponse,
    UnrealDescribeObjectResponse,
    UnrealGetThumbnailResponse,
    UnrealSceneSummaryResponse,
    UnrealCaptureViewportResponse,
    UnrealViewportInfoResponse,
    UnrealSetCameraViewResponse,
    UnrealFocusActorResponse,
    UnrealSpawnActorResponse,
    UnrealDeleteActorResponse,
    UnrealSetActorTransformResponse,
    UnrealSetActorPropertyResponse,
    UnrealCallActorFunctionResponse,
    UnrealSetActorParentResponse,
    UnrealAddComponentResponse,
    UnrealSetActorVisibilityResponse,
    UnrealGetMaterialInfoResponse,
    UnrealSetMaterialParamsResponse,
    UnrealCreateMaterialInstanceResponse,
    UnrealAssignMaterialResponse,
    UnrealSetLightParamsResponse,
    UnrealSetRenderSettingsResponse,
    UnrealControlSimulationResponse,
    UnrealGetSimulationStatusResponse,
    UnrealEnablePhysicsResponse,
    UnrealSetCollisionResponse,
    UnrealApplyForceResponse,
    UnrealSetPhysicsParamsResponse,
    # Phase 6: USD/SimReady
    UnrealImportUsdResponse,
    UnrealExportUsdResponse,
    UnrealConvertToSimreadyResponse,
    UnrealValidateSimreadyResponse,
    UnrealGetInterchangeInfoResponse,
    # Phase 7: Agent Tools
    UnrealBatchOperationsResponse,
    UnrealQuerySceneGraphResponse,
    UnrealAnalyzeSceneForRoboticsResponse,
    UnrealGenerateProceduralSceneResponse,
    UnrealGetActorBySemanticLabelResponse,
    # Phase 8: Geometry & Modeling
    UnrealGenerateMeshPrimitiveResponse,
    UnrealApplyMeshBooleanResponse,
    UnrealComputeConvexHullResponse,
    UnrealDecomposeConvexHullResponse,
    UnrealEditMeshTopologyResponse,
    UnrealSubdivideMeshResponse,
    UnrealSimplifyMeshResponse,
    UnrealCutMeshPlaneResponse,
    UnrealValidateMeshResponse,
    UnrealConvertMeshFormatResponse,
    UnrealRemeshMeshResponse,
    UnrealComputeMeshUvResponse,
    FocusPrimResponse,
    PrimActionResponse,
    SceneSummaryResponse,
    PrimSearchResponse,
    BBoxResponse,
    ErrorResponse,
]
