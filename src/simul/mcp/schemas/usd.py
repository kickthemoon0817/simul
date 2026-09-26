"""USD and generic scene schemas exposed through MCP."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from .common import BoundingBox, Transform


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


class FocusPrimResponse(BaseModel):
    """Response from focusing camera on a prim."""

    success: bool = Field(..., description="Whether focus succeeded")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    stage_id: str = Field(..., description="Stage identifier")
    prim_path: str = Field(..., description="Prim path")
    focus_point: List[float] = Field(
        ..., description="Focus point [x, y, z]", min_length=3, max_length=3
    )
    camera_position: List[float] = Field(
        ..., description="Camera position [x, y, z]", min_length=3, max_length=3
    )
    message: str = Field(..., description="Status message")


class PrimActionResponse(BaseModel):
    success: bool = Field(..., description="Whether request was successful")
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
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

__all__ = [
    "USDFileInfo",
    "USDFileRequest",
    "USDValidateRequest",
    "StageInfo",
    "PrimInfo",
    "PrimInfoRequest",
    "MeshInfoRequest",
    "PrimCreateRequest",
    "PrimUpdateRequest",
    "PrimDeleteRequest",
    "MeshInfo",
    "FocusPrimResponse",
    "PrimActionResponse",
    "SceneSummaryRequest",
    "SceneSummaryResponse",
    "PrimSearchRequest",
    "PrimSearchResponse",
    "BBoxRequest",
    "BBoxResponse",
]
