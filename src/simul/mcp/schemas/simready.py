"""SimReady schemas for Blender and Unreal asset workflows."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class SimReadySemanticLabels(BaseModel):
    """
    Semantic labeling for SimReady assets.

    export_simready_usd writes each field as a UsdSemantics LabelsAPI taxonomy
    (``class``, ``hierarchy``, ``qcode``, and each additional label's key), or
    with Isaac Sim's SemanticsAPI convention where Blender's USD has no
    UsdSemantics.
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


#: collider_type values: the UsdPhysics MeshCollisionAPI approximations, then the
#: earlier aliases (mesh = none, box = boundingCube, sphere = boundingSphere,
#: capsule = convexHull, as UsdPhysics has no capsule approximation of a mesh).
COLLIDER_TYPES = (
    "convexHull",
    "convexDecomposition",
    "meshSimplification",
    "none",
    "boundingCube",
    "boundingSphere",
    "mesh",
    "box",
    "sphere",
    "capsule",
)


class SimReadyPhysicsProperties(BaseModel):
    """
    Physics properties for SimReady assets.

    export_simready_usd writes them as UsdPhysics schemas: RigidBodyAPI,
    MassAPI, CollisionAPI with a MeshCollisionAPI approximation on the meshes,
    and a physics material (UsdPhysics.MaterialAPI) bound with the physics
    purpose.
    """

    mass_kg: Optional[float] = Field(None, ge=0.0, description="Mass in kilograms")
    collider_type: Optional[str] = Field(
        None,
        description=(
            "UsdPhysics mesh collision approximation: convexHull, convexDecomposition, "
            "meshSimplification, none (the triangle mesh), boundingCube or boundingSphere; "
            "also mesh, box, sphere and capsule (written as none, boundingCube, "
            "boundingSphere and convexHull)"
        ),
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
        if v is not None and v not in COLLIDER_TYPES:
            raise ValueError(
                f"collider_type must be one of {list(COLLIDER_TYPES)}, got '{v}'"
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
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
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
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
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
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    compliant: bool = Field(..., description="True when zero errors found")
    object_count: int = Field(..., description="Number of objects checked")
    issue_count: int = Field(..., description="Total issues found")
    issues: List[SimReadyValidationIssue] = Field(
        default_factory=list, description="Detailed issue list"
    )
    meters_per_unit: Optional[float] = Field(
        None,
        description=(
            "Meters per Blender unit (the scene's Unit Scale) the sizes were judged in"
        ),
    )


class SimReadyExportRequest(BaseModel):
    """Request to export a SimReady-compliant USD file."""

    file_path: str = Field(
        ..., description="Output .usd, .usda, .usdc or .usdz file path"
    )
    object_names: Optional[List[str]] = Field(
        None,
        min_length=1,
        description=(
            "Objects to export, each with its descendants unless include_children is "
            "false; listed objects are exported even when hidden. Null exports the "
            "scene's objects that are visible in renders"
        ),
    )
    embed_metadata: bool = Field(
        True,
        description=(
            "Also keep the simready_ custom properties as userProperties attributes"
        ),
    )
    validate_before_export: bool = Field(
        True, description="Run validation before exporting"
    )
    include_children: bool = Field(
        True, description="Export the descendants of the listed objects"
    )
    include_lights_and_cameras: bool = Field(
        False,
        description="Also export lights, cameras and the World's dome light",
    )
    force: bool = Field(
        False, description="Export although validation found errors"
    )


class SimReadyExportedPrim(BaseModel):
    """Where one Blender object was written."""

    object_name: str = Field(..., description="Blender object name")
    prim_path: str = Field(..., description="Its prim path in the exported USD")


class SimReadySkippedObject(BaseModel):
    """An object the export left out, and why."""

    object_name: str = Field(..., description="Blender object name")
    reason: str = Field(..., description="Why it was not exported")


class SimReadyExportPhysics(BaseModel):
    """The UsdPhysics schemas the export authored (lists cut at 50 entries)."""

    rigid_body_count: int = Field(0, description="Prims with RigidBodyAPI")
    collider_count: int = Field(
        0, description="Meshes with CollisionAPI and MeshCollisionAPI"
    )
    mass_count: int = Field(0, description="Prims with MassAPI")
    physics_material_count: int = Field(
        0, description="Physics materials (UsdPhysics.MaterialAPI)"
    )
    rigid_bodies: List[str] = Field(default_factory=list, description="Rigid body prims")
    colliders: List[str] = Field(default_factory=list, description="Collider meshes")
    physics_materials: List[str] = Field(
        default_factory=list, description="Physics material prims"
    )


class SimReadyExportSemantics(BaseModel):
    """The semantic labels the export authored."""

    labels_schema: str = Field(
        ...,
        description=(
            "UsdSemantics.LabelsAPI, or Isaac SemanticsAPI where Blender's USD has "
            "no UsdSemantics"
        ),
    )
    labelled_prim_count: int = Field(0, description="Prims given labels")
    labelled_prims: List[str] = Field(
        default_factory=list, description="Labelled prims (cut at 50)"
    )


class SimReadyExportResponse(BaseModel):
    """Response after exporting SimReady USD."""

    success: bool = Field(...)
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    file_path: str = Field(..., description="Written file path")
    object_count: int = Field(..., description="Number of objects written")
    default_prim: Optional[str] = Field(
        None, description="The asset root, the stage's default prim"
    )
    prims: List[SimReadyExportedPrim] = Field(
        default_factory=list, description="Each written object's prim (cut at 50)"
    )
    meters_per_unit: Optional[float] = Field(
        None, description="Stage metersPerUnit, from the scene's Unit Scale"
    )
    up_axis: Optional[str] = Field(None, description="Stage upAxis")
    kilograms_per_unit: Optional[float] = Field(
        None, description="Stage kilogramsPerUnit"
    )
    physics: Optional[SimReadyExportPhysics] = Field(
        None, description="UsdPhysics schemas authored"
    )
    semantics: Optional[SimReadyExportSemantics] = Field(
        None, description="Semantic labels authored"
    )
    skipped_objects: List[SimReadySkippedObject] = Field(
        default_factory=list, description="Objects left out, and why (cut at 50)"
    )
    validation_passed: Optional[bool] = Field(
        None, description="Whether pre-export validation passed; null when not run"
    )
    forced: bool = Field(
        False, description="Exported with force although validation found errors"
    )
    issue_count: int = Field(0, description="Validation issues found")
    issues: Optional[List[SimReadyValidationIssue]] = Field(
        None, description="Validation issues (cut at 50; null when not run)"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Physics or layout problems worth fixing (cut at 50)",
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
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    root_name: str = Field(..., description="Root empty name")
    children: List[str] = Field(..., description="Successfully parented children")
    hierarchy_path: str = Field(..., description="Logical hierarchy path from root")


__all__ = [
    "SimReadySemanticLabels",
    "SimReadyPhysicsProperties",
    "SimReadyMaterialProperties",
    "SimReadyMetadata",
    "SimReadyApplyMetadataRequest",
    "SimReadyApplyMetadataResponse",
    "SimReadyGetMetadataRequest",
    "SimReadyGetMetadataResponse",
    "SimReadyValidationIssue",
    "SimReadyValidateRequest",
    "SimReadyValidateResponse",
    "COLLIDER_TYPES",
    "SimReadyExportRequest",
    "SimReadyExportedPrim",
    "SimReadySkippedObject",
    "SimReadyExportPhysics",
    "SimReadyExportSemantics",
    "SimReadyExportResponse",
    "SimReadySetupHierarchyRequest",
    "SimReadySetupHierarchyResponse",
]
