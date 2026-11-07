"""USD and generic scene schemas exposed through MCP."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .common import BoundingBox, Transform

Purpose = Literal["default", "render", "proxy", "guide"]

DEFAULT_BBOX_PURPOSES: List[Purpose] = ["default", "render"]

SearchType = Literal["by_name", "by_path", "by_type", "by_api_schema"]

MatchMode = Literal["auto", "exact", "substring", "glob", "regex"]

SummaryFormat = Literal["json", "text"]

# Mirror simul.usd.values and simul.usd.search; kept here so the schema
# module imports no pxr.
DEFAULT_ARRAY_PREVIEW = 16
MAX_ARRAY_PREVIEW = 10_000
DEFAULT_SEARCH_LIMIT = 200
MAX_SEARCH_LIMIT = 5000
DEFAULT_SUBTREE_LIMIT = 150
MAX_SUBTREE_LIMIT = 1000


class USDIssue(BaseModel):
    """One problem found in a USD file."""

    severity: str = Field(..., description="error, warning or info")
    code: str = Field(
        ..., description="Stable identifier, e.g. ParseError, MissingReference, NestedArticulation"
    )
    message: str = Field(..., description="What is wrong, in pxr's words where pxr reported it")
    source: str = Field(
        "simul", description="simul, or the pxr validator that reported the issue"
    )
    location: Dict[str, Any] = Field(
        default_factory=dict,
        description="Where it is, as known: layer, prim_path, property_path, asset_path, line, column",
    )


class USDFileInfo(BaseModel):
    """USD file information."""

    file_path: str = Field(..., description="Path to USD file")
    file_size: int = Field(..., description="File size in bytes")
    format: str = Field(..., description="File extension (usd, usda, usdc, usdz)")
    is_valid: bool = Field(
        ..., description="pxr opened the file and no issue has severity error"
    )
    can_read: bool = Field(
        ..., description="pxr opened the file, so load_usd_file can load it"
    )
    layer_format: Optional[str] = Field(
        None, description="Format pxr read the root layer as (usda, usdc, usdz)"
    )
    default_prim: Optional[str] = Field(None, description="Default prim path, when valid")
    up_axis: Optional[str] = Field(None, description="Stage up axis (authored or fallback)")
    meters_per_unit: Optional[float] = Field(
        None, description="Stage meters per unit (authored or fallback)"
    )
    error_count: int = Field(0, description="Issues with severity error")
    warning_count: int = Field(0, description="Issues with severity warning")
    info_count: int = Field(0, description="Issues with severity info")
    issues: List[USDIssue] = Field(
        default_factory=list, description="Problems found, errors first"
    )
    validators: List[str] = Field(
        default_factory=list, description="pxr UsdValidation validators that ran"
    )


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
    warnings: List[USDIssue] = Field(
        default_factory=list,
        description=(
            "Problems the stage loaded despite: missing sublayers, references or "
            "payloads, unresolved reference prim paths"
        ),
    )


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


class PrimUpdateRequest(BaseModel):
    stage_id: str = Field(..., description="Stage identifier", min_length=1)
    prim_path: str = Field(..., description="Prim path", min_length=1)
    attributes: Dict[str, Any] = Field(
        default_factory=dict, description="Attribute values"
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


class SceneSummaryRequest(BaseModel):
    """Request for scene summary."""

    stage_id: str = Field(..., description="Stage identifier")
    include_meshes: bool = Field(True, description="Include detailed mesh information")
    include_materials: bool = Field(True, description="Include material information")
    max_depth: int = Field(5, description="Maximum hierarchy depth", ge=1, le=10)
    format: SummaryFormat = Field("json", description="Output format (json, text)")


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
    search_type: SearchType = Field(..., description="What the query is matched against")
    query: str = Field(..., description="Search query")
    exact_match: bool = Field(
        False, description="Shorthand for match='exact' (kept for older callers)"
    )
    match: MatchMode = Field("auto", description="How the query is matched")
    case_sensitive: bool = Field(False, description="Compare case-sensitively")
    root_path: Optional[str] = Field(None, description="Search only under this prim")
    include_instance_proxies: bool = Field(True, description="Look inside instances")
    offset: int = Field(0, ge=0, description="Matches to skip")
    limit: int = Field(
        DEFAULT_SEARCH_LIMIT, ge=1, le=MAX_SEARCH_LIMIT, description="Matches to return"
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str, info: Any) -> str:
        if not v and info.data.get("search_type") in ("by_type", "by_api_schema"):
            raise ValueError("query must name a type or API schema")
        return v


class PrimSearchResponse(BaseModel):
    """Response from prim search: one page of the matches."""

    success: bool = Field(..., description="Whether search was successful")
    stage_id: str = Field(..., description="Stage identifier")
    search_type: str = Field(..., description="Search type used")
    query: str = Field(..., description="Search query")
    match: str = Field("auto", description="Match mode used")
    case_sensitive: bool = Field(False, description="Whether case mattered")
    root_path: Optional[str] = Field(None, description="Prim the search was limited to")
    include_instance_proxies: bool = Field(True, description="Whether instances were searched")
    results: List[str] = Field(..., description="Matching prim paths in this page, traversal order")
    count: int = Field(..., description="Matches in this page")
    total_matches: int = Field(0, description="Matches in the whole search")
    searched: int = Field(0, description="Prims the search visited")
    offset: int = Field(0, description="Matches skipped before this page")
    limit: int = Field(DEFAULT_SEARCH_LIMIT, description="Page size applied")
    truncated: bool = Field(False, description="More matches follow this page")
    next_offset: Optional[int] = Field(
        None, description="offset of the next page; None when this is the last"
    )
    notes: List[str] = Field(default_factory=list, description="How the query was interpreted")
    error: Optional[str] = Field(None, description="Error message if failed")


class PrimSubtreeRequest(BaseModel):
    """Request for a subtree listing."""

    stage_id: str = Field(..., description="Stage identifier", min_length=1)
    root_path: str = Field("/", description="Subtree root", min_length=1)
    max_depth: int = Field(5, ge=0, le=64, description="Deepest level listed below the root")
    max_results: int = Field(
        DEFAULT_SUBTREE_LIMIT, ge=1, le=MAX_SUBTREE_LIMIT, description="Prims per page"
    )
    offset: int = Field(0, ge=0, description="Prims to skip")
    include_instance_proxies: bool = Field(True, description="Descend into instances")
    include_properties: bool = Field(False, description="Add authored attributes and relationships")
    attribute_filter: Optional[List[str]] = Field(
        None, description="Glob patterns limiting the properties added"
    )
    array_preview: int = Field(
        4, ge=0, le=MAX_ARRAY_PREVIEW, description="Leading array elements to include"
    )


class PrimSubtreeResponse(BaseModel):
    """One page of a subtree listing."""

    success: bool = Field(True, description="Whether the listing succeeded")
    stage_id: str = Field(..., description="Stage identifier")
    root_path: str = Field(..., description="Subtree root")
    max_depth: int = Field(..., description="Deepest level listed")
    prims: List[Dict[str, Any]] = Field(
        ...,
        description=(
            "path, name, type, depth, child_count, api_schemas (kind, is_instance, "
            "is_instance_proxy when set; attributes and relationships when asked)"
        ),
    )
    count: int = Field(..., description="Prims in this page")
    total: int = Field(..., description="Prims in the subtree up to max_depth")
    offset: int = Field(0, description="Prims skipped before this page")
    limit: int = Field(DEFAULT_SUBTREE_LIMIT, description="Page size applied")
    truncated: bool = Field(False, description="More prims follow this page")
    next_offset: Optional[int] = Field(
        None, description="offset of the next page; None when this is the last"
    )
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


__all__ = [
    "USDIssue",
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
    "PrimSubtreeRequest",
    "PrimSubtreeResponse",
    "BBoxRequest",
    "BBoxResponse",
]
