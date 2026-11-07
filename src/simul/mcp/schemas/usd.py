"""USD and generic scene schemas exposed through MCP."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .common import BoundingBox, Transform

Purpose = Literal["default", "render", "proxy", "guide"]

DEFAULT_BBOX_PURPOSES: List[Purpose] = ["default", "render"]

# Mirrors simul.usd.values; kept here so the schema module imports no pxr.
DEFAULT_ARRAY_PREVIEW = 16
MAX_ARRAY_PREVIEW = 10_000


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
    """USD prim information.

    Values are plain JSON: vectors are lists, quaternions ``[w, x, y, z]``
    lists, matrices lists of rows, and a non-finite float the string
    ``"Infinity"``, ``"-Infinity"`` or ``"NaN"``.
    """

    path: str = Field(..., description="Prim path")
    name: str = Field(..., description="Prim name")
    type: str = Field(..., description="Prim type")
    is_active: bool = Field(..., description="Whether prim is active")
    is_loaded: bool = Field(..., description="Whether prim is loaded")
    is_defined: bool = Field(..., description="Whether prim is defined")
    is_instance: bool = Field(..., description="Whether prim is an instance")
    is_instance_proxy: bool = Field(
        False, description="Whether prim lies under an instance (read-only proxy)"
    )
    prototype: Optional[str] = Field(None, description="Prototype path of an instance")
    prim_in_prototype: Optional[str] = Field(
        None, description="Prototype prim an instance proxy stands for"
    )
    purpose: Optional[str] = Field(None, description="Authored or fallback purpose")
    computed_purpose: Optional[str] = Field(
        None, description="Purpose after inheritance from ancestors"
    )
    visibility: Optional[str] = Field(None, description="Prim visibility")
    computed_visibility: Optional[str] = Field(
        None, description="Visibility after inheritance (invisible if an ancestor is)"
    )
    kind: Optional[str] = Field(None, description="Prim kind")
    api_schemas: List[str] = Field(
        default_factory=list, description="Applied API schemas, e.g. PhysicsRigidBodyAPI"
    )
    bbox: Optional[BoundingBox] = Field(
        None, description="World bounding box; None when nothing is boundable"
    )
    bbox_purposes: List[str] = Field(
        default_factory=list, description="Purposes the bounding box counts"
    )
    transform: Optional[Transform] = Field(None, description="Local transformation")
    children_count: int = Field(
        ..., description="Number of child prims, instance proxies included"
    )
    children_types: Dict[str, int] = Field(
        ..., description="Child prim type counts; '(typeless)' for prims without a type"
    )
    material_bindings: List[str] = Field(..., description="Bound material paths")
    attribute_count: int = Field(
        0, description="Attributes on the prim, before attribute_filter"
    )
    attributes: Dict[str, Any] = Field(
        ...,
        description=(
            "Attribute name to {type, value, authored}; arrays add length and "
            "truncated, animated attributes time_samples, connected ones connections"
        ),
    )
    relationships: Dict[str, Any] = Field(
        default_factory=dict,
        description="Relationship name to {targets, authored}",
    )
    metadata: Dict[str, Any] = Field(
        ...,
        description=(
            "specifier, instanceable, composition arcs, variant_sets, "
            "opinion_layers and other authored metadata"
        ),
    )
    time_code: Optional[float] = Field(
        None, description="Time code evaluated; None for the default time"
    )


class PrimInfoRequest(BaseModel):
    """Request for prim details."""

    stage_id: str = Field(..., description="Stage identifier", min_length=1)
    prim_path: str = Field(..., description="Prim path", min_length=1)
    time_code: Optional[float] = Field(None, description="Time code to evaluate at")
    include_purposes: Optional[List[Purpose]] = Field(
        None, description="Purposes the bounding box counts"
    )
    array_preview: int = Field(
        DEFAULT_ARRAY_PREVIEW,
        ge=0,
        le=MAX_ARRAY_PREVIEW,
        description="Leading elements of each array attribute to include",
    )
    attribute_filter: Optional[List[str]] = Field(
        None, description="Glob patterns selecting the attributes to list"
    )


class MeshInfoRequest(BaseModel):
    """Request for mesh details."""

    stage_id: str = Field(..., description="Stage identifier", min_length=1)
    prim_path: str = Field(..., description="Mesh prim path", min_length=1)


class PrimCreateRequest(BaseModel):
    stage_id: str = Field(..., description="Stage identifier", min_length=1)
    prim_path: str = Field(..., description="Absolute prim path", min_length=1)
    prim_type: str = Field(..., description="Concrete USD prim type", min_length=1)
    attributes: Dict[str, Any] = Field(
        default_factory=dict, description="Attribute values"
    )
    redefine_existing: bool = Field(
        False, description="Allow re-typing a prim that already exists"
    )
    attribute_types: Dict[str, str] = Field(
        default_factory=dict, description="USD value type per new attribute name"
    )
    time_code: Optional[float] = Field(
        None, description="Write values as time samples at this time", allow_inf_nan=False
    )


class PrimUpdateRequest(BaseModel):
    stage_id: str = Field(..., description="Stage identifier", min_length=1)
    prim_path: str = Field(..., description="Absolute prim path", min_length=1)
    attributes: Dict[str, Any] = Field(
        default_factory=dict, description="Attribute values"
    )
    attribute_types: Dict[str, str] = Field(
        default_factory=dict, description="USD value type per new attribute name"
    )
    time_code: Optional[float] = Field(
        None, description="Write values as time samples at this time", allow_inf_nan=False
    )


class PrimDeleteRequest(BaseModel):
    stage_id: str = Field(..., description="Stage identifier", min_length=1)
    prim_path: str = Field(..., description="Absolute prim path", min_length=1)
    allow_root_delete: bool = Field(
        False, description="Permit deleting the stage's default prim"
    )


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
    changed: Optional[List[str]] = Field(
        None, description="Attributes, relationships and metadata the edit wrote"
    )
    created_attributes: Optional[Dict[str, str]] = Field(
        None, description="New attributes the edit created, with their USD value types"
    )
    time_code: Optional[float] = Field(
        None, description="Time code the values were written at; null for defaults"
    )


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
    time_code: Optional[float] = Field(
        None, description="Time code to evaluate at; None for the default time"
    )
    include_purposes: Optional[List[Purpose]] = Field(
        None,
        description="Purposes whose geometry counts; None for default and render",
    )

    @field_validator("include_purposes")
    @classmethod
    def validate_include_purposes(
        cls, v: Optional[List[Purpose]]
    ) -> Optional[List[Purpose]]:
        if v is None:
            return v
        if not v:
            raise ValueError("include_purposes must name at least one purpose")
        return list(dict.fromkeys(v))


class BBoxResponse(BaseModel):
    """Response from bounding box computation."""

    success: bool = Field(..., description="Whether computation was successful")
    stage_id: str = Field(..., description="Stage identifier")
    prim_path: Optional[str] = Field(None, description="Prim path")
    bbox: Optional[BoundingBox] = Field(
        None, description="Computed bounding box; None when has_geometry is False"
    )
    world_space: bool = Field(..., description="Whether bbox is in world space")
    has_geometry: bool = Field(
        True,
        description=(
            "False when nothing boundable with an included purpose lies at or "
            "below the prim"
        ),
    )
    included_purposes: List[Purpose] = Field(
        default_factory=lambda: list(DEFAULT_BBOX_PURPOSES),
        description="Purposes whose geometry the bound covers",
    )
    time_code: Optional[float] = Field(
        None, description="Time code evaluated; None for the default time"
    )
    excluded_purposes_with_geometry: Optional[List[Purpose]] = Field(
        None,
        description=(
            "With no bound: the purposes left out that do hold geometry here; "
            "add them to include_purposes to bound it"
        ),
    )
    message: Optional[str] = Field(None, description="What an empty result means")
    error: Optional[str] = Field(None, description="Error message if failed")


class StageRequest(BaseModel):
    """Request naming one loaded stage."""

    stage_id: str = Field(..., description="Stage identifier", min_length=1)


class StageSaveRequest(StageRequest):
    """Request to save a stage's root layer in place."""

    overwrite_external_changes: bool = Field(
        False, description="Save over a file that changed on disk after the stage read it"
    )


class StageExportRequest(StageRequest):
    """Request to write a stage to a new file."""

    file_path: str = Field(..., description="Destination .usda, .usdc, .usd or .usdz", min_length=1)
    flatten: bool = Field(False, description="Write the composed stage as one layer")
    overwrite: bool = Field(False, description="Replace an existing file")


class StageDiscardRequest(StageRequest):
    """Request to reload or unload a stage."""

    discard_unsaved: bool = Field(False, description="Drop unsaved edits")


class FileWritten(BaseModel):
    """One file a save or export wrote."""

    path: str = Field(..., description="Absolute path of the file")
    bytes: int = Field(..., description="Size of the file written")


class StageStatus(BaseModel):
    """A loaded stage's file and unsaved state."""

    stage_id: str = Field(..., description="Stage identifier")
    file_path: str = Field(..., description="Root file the stage was loaded from")
    has_unsaved_changes: bool = Field(..., description="The root layer has edits not saved to disk")
    changed_on_disk: List[str] = Field(
        ..., description="Files the stage composes that changed on disk since it read them"
    )


class StageListResponse(BaseModel):
    """The stages the calling MCP session has loaded."""

    success: bool = Field(..., description="Whether request was successful")
    stages: List[StageStatus] = Field(..., description="Loaded stages")
    count: int = Field(..., description="Number of loaded stages")


class StageSaveResponse(BaseModel):
    """Result of saving a stage in place."""

    success: bool = Field(..., description="Whether request was successful")
    stage_id: str = Field(..., description="Stage identifier")
    file_path: str = Field(..., description="The stage's root file")
    saved: bool = Field(..., description="False when there were no unsaved edits to write")
    files_written: List[FileWritten] = Field(..., description="Files written")
    message: str = Field(..., description="Status message")


class StageExportResponse(BaseModel):
    """Result of writing a stage to a new file."""

    success: bool = Field(..., description="Whether request was successful")
    stage_id: str = Field(..., description="Stage identifier")
    file_path: str = Field(..., description="The file written")
    format: str = Field(..., description="usda, usdc, usd or usdz")
    flatten: bool = Field(..., description="Whether the composed stage was flattened")
    files_written: List[FileWritten] = Field(..., description="Files written")
    rewritten_asset_paths: int = Field(
        ..., description="Relative asset paths re-anchored to the new location"
    )
    message: str = Field(..., description="Status message")


class StageReloadResponse(BaseModel):
    """Result of rereading a stage from disk."""

    success: bool = Field(..., description="Whether request was successful")
    stage_id: str = Field(..., description="Stage identifier")
    file_path: str = Field(..., description="The stage's root file")
    discarded_unsaved: bool = Field(..., description="Unsaved edits were dropped")
    reloaded_files: List[str] = Field(..., description="Files reread from disk")
    message: str = Field(..., description="Status message")


class StageUnloadResponse(BaseModel):
    """Result of unloading a stage."""

    success: bool = Field(..., description="Whether request was successful")
    stage_id: str = Field(..., description="Stage identifier")
    discarded_unsaved: bool = Field(..., description="Unsaved edits were dropped")
    message: str = Field(..., description="Status message")


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
    "StageRequest",
    "StageSaveRequest",
    "StageExportRequest",
    "StageDiscardRequest",
    "FileWritten",
    "StageStatus",
    "StageListResponse",
    "StageSaveResponse",
    "StageExportResponse",
    "StageReloadResponse",
    "StageUnloadResponse",
]
