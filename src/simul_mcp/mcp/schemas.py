"""
Pydantic schemas for Isaac Sim MCP Server tool inputs and outputs.

This module defines the data models used for MCP tool communication,
ensuring type safety and validation for all tool operations.
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


class ImageFormat(str, Enum):
    """Supported image formats for viewport capture."""
    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"
    EXR = "exr"


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
    min: List[float] = Field(..., description="Minimum point [x, y, z]", min_items=3, max_items=3)
    max: List[float] = Field(..., description="Maximum point [x, y, z]", min_items=3, max_items=3)
    center: Optional[List[float]] = Field(None, description="Center point [x, y, z]", min_items=3, max_items=3)
    size: Optional[List[float]] = Field(None, description="Size [width, height, depth]", min_items=3, max_items=3)
    volume: Optional[float] = Field(None, description="Bounding box volume")


class Transform(BaseModel):
    """3D transformation representation."""
    translation: List[float] = Field(..., description="Translation [x, y, z]", min_items=3, max_items=3)
    rotation: List[float] = Field(..., description="Rotation quaternion [w, x, y, z]", min_items=4, max_items=4)
    scale: List[float] = Field(..., description="Scale [x, y, z]", min_items=3, max_items=3)


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


class FocusPrimRequest(BaseModel):
    """Request to focus on a prim."""
    stage_id: str = Field(..., description="Stage identifier", min_length=1)
    prim_path: str = Field(..., description="Prim path", min_length=1)


class PrimCreateRequest(BaseModel):
    stage_id: str = Field(..., description="Stage identifier", min_length=1)
    prim_path: str = Field(..., description="Prim path", min_length=1)
    prim_type: str = Field(..., description="USD prim type", min_length=1)
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Attribute values")


class PrimUpdateRequest(BaseModel):
    stage_id: str = Field(..., description="Stage identifier", min_length=1)
    prim_path: str = Field(..., description="Prim path", min_length=1)
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Attribute values")


class PrimDeleteRequest(BaseModel):
    stage_id: str = Field(..., description="Stage identifier", min_length=1)
    prim_path: str = Field(..., description="Prim path", min_length=1)


class RigidBodyEnableRequest(BaseModel):
    """Request to enable rigid body physics on a prim."""
    prim_path: str = Field(..., description="Prim path", min_length=1)
    mass: Optional[float] = Field(None, description="Mass in kilograms", gt=0)


class RigidBodyVelocityRequest(BaseModel):
    """Request to set rigid body velocities."""
    prim_path: str = Field(..., description="Prim path", min_length=1)
    linear_velocity: Optional[List[float]] = Field(None, description="Linear velocity [x, y, z]", min_items=3, max_items=3)
    angular_velocity: Optional[List[float]] = Field(None, description="Angular velocity [x, y, z]", min_items=3, max_items=3)

    @validator('angular_velocity', always=True)
    def validate_velocity(cls, v, values):
        if v is None and values.get('linear_velocity') is None:
            raise ValueError("At least one of linear_velocity or angular_velocity must be provided")
        return v


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


class ViewportCaptureRequest(BaseModel):
    """Request for viewport capture."""
    width: Optional[int] = Field(None, description="Image width", gt=0, le=4096)
    height: Optional[int] = Field(None, description="Image height", gt=0, le=4096)
    format: ImageFormat = Field(ImageFormat.PNG, description="Image format")
    save_to_file: bool = Field(False, description="Save image to file")
    file_path: Optional[str] = Field(None, description="File path for saved image")
    
    @validator('width', 'height')
    def validate_dimensions(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Dimensions must be positive")
        return v


class ViewportCaptureResponse(BaseModel):
    """Response from viewport capture."""
    success: bool = Field(..., description="Whether capture was successful")
    width: int = Field(..., description="Actual image width")
    height: int = Field(..., description="Actual image height")
    format: str = Field(..., description="Image format")
    file_path: Optional[str] = Field(None, description="Path to saved file")
    error: Optional[str] = Field(None, description="Error message if failed")


class ViewportInfoResponse(BaseModel):
    """Response with viewport information."""
    success: bool = Field(..., description="Whether request was successful")
    viewport_available: bool = Field(..., description="Whether a viewport is available")
    width: Optional[int] = Field(None, description="Viewport width")
    height: Optional[int] = Field(None, description="Viewport height")
    camera_path: Optional[str] = Field(None, description="Active camera path")
    supported_formats: List[str] = Field(..., description="Supported capture formats")
    max_capture_size: int = Field(..., description="Maximum capture size")


class SimulationControlResponse(BaseModel):
    """Response from simulation control."""
    success: bool = Field(..., description="Whether the action succeeded")
    action: str = Field(..., description="Action performed")
    steps: Optional[int] = Field(None, description="Step count if action was step")
    message: str = Field(..., description="Status message")


class SimulationStatusResponse(BaseModel):
    """Response with simulation status."""
    success: bool = Field(..., description="Whether request was successful")
    world_initialized: bool = Field(..., description="Whether world is initialized")
    is_playing: bool = Field(..., description="Whether simulation is playing")
    current_time: float = Field(..., description="Current simulation time")
    physics_dt: Optional[float] = Field(None, description="Physics timestep")
    rendering_dt: Optional[float] = Field(None, description="Rendering timestep")


class RigidBodyActionResponse(BaseModel):
    """Response for rigid body actions."""
    success: bool = Field(..., description="Whether the action succeeded")
    prim_path: str = Field(..., description="Prim path")
    message: str = Field(..., description="Status message")


class RigidBodyStateResponse(BaseModel):
    """Response with rigid body state."""
    success: bool = Field(..., description="Whether request was successful")
    prim_path: str = Field(..., description="Prim path")
    enabled: bool = Field(..., description="Whether rigid body API is applied")
    mass: Optional[float] = Field(None, description="Mass in kilograms")
    linear_velocity: Optional[List[float]] = Field(None, description="Linear velocity [x, y, z]", min_items=3, max_items=3)
    angular_velocity: Optional[List[float]] = Field(None, description="Angular velocity [x, y, z]", min_items=3, max_items=3)


class CameraInfoResponse(BaseModel):
    """Response with camera information."""
    success: bool = Field(..., description="Whether request was successful")
    camera_available: bool = Field(..., description="Whether a camera is available")
    camera_path: Optional[str] = Field(None, description="Camera prim path")
    focal_length: Optional[float] = Field(None, description="Camera focal length")
    horizontal_aperture: Optional[float] = Field(None, description="Camera horizontal aperture")
    vertical_aperture: Optional[float] = Field(None, description="Camera vertical aperture")
    can_control: Optional[bool] = Field(None, description="Whether camera can be controlled")


class CameraViewResponse(BaseModel):
    success: bool = Field(..., description="Whether request was successful")
    eye: List[float] = Field(..., description="Camera position [x, y, z]", min_items=3, max_items=3)
    target: List[float] = Field(..., description="Camera target [x, y, z]", min_items=3, max_items=3)
    up: List[float] = Field(..., description="Up vector [x, y, z]", min_items=3, max_items=3)
    message: str = Field(..., description="Status message")


class FocusPrimResponse(BaseModel):
    """Response from focusing camera on a prim."""
    success: bool = Field(..., description="Whether focus succeeded")
    stage_id: str = Field(..., description="Stage identifier")
    prim_path: str = Field(..., description="Prim path")
    focus_point: List[float] = Field(..., description="Focus point [x, y, z]", min_items=3, max_items=3)
    camera_position: List[float] = Field(..., description="Camera position [x, y, z]", min_items=3, max_items=3)
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

    @validator('format')
    def validate_format(cls, v):
        valid_formats = ['json', 'text']
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


class CameraViewRequest(BaseModel):
    """Request to set camera view."""
    eye: List[float] = Field(..., description="Camera position [x, y, z]", min_items=3, max_items=3)
    target: List[float] = Field(..., description="Camera target [x, y, z]", min_items=3, max_items=3)
    up: List[float] = Field([0, 1, 0], description="Up vector [x, y, z]", min_items=3, max_items=3)


class SimulationControlRequest(BaseModel):
    """Request for simulation control."""
    action: str = Field(..., description="Action (play, pause, stop, reset, step)")
    steps: int = Field(1, description="Number of steps (for step action)", ge=1)
    
    @validator('action')
    def validate_action(cls, v):
        valid_actions = ['play', 'pause', 'stop', 'reset', 'step']
        if v not in valid_actions:
            raise ValueError(f"Action must be one of {valid_actions}")
        return v


class PrimSearchRequest(BaseModel):
    """Request to search for prims."""
    stage_id: str = Field(..., description="Stage identifier")
    search_type: str = Field(..., description="Search type (by_type, by_name)")
    query: str = Field(..., description="Search query")
    exact_match: bool = Field(False, description="Use exact matching (for name search)")
    
    @validator('search_type')
    def validate_search_type(cls, v):
        valid_types = ['by_type', 'by_name']
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
    prim_path: Optional[str] = Field(None, description="Prim path (None for stage bbox)")
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
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


# Tool result types
ToolResult = Union[
    str,  # Simple string response
    Dict[str, Any],  # JSON response
    StageInfo,
    USDFileInfo,
    PrimInfo,
    MeshInfo,
    ViewportCaptureResponse,
    ViewportInfoResponse,
    SimulationControlResponse,
    SimulationStatusResponse,
    RigidBodyActionResponse,
    RigidBodyStateResponse,
    CameraInfoResponse,
    CameraViewResponse,
    FocusPrimResponse,
    PrimActionResponse,
    SceneSummaryResponse,
    PrimSearchResponse,
    BBoxResponse,
    ErrorResponse,
]
