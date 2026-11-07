"""SimReady schemas for the Blender SimReady tools.

The metadata models refuse unknown fields, naming the closest valid one, so a
misspelled field fails instead of being dropped. The Blender runtime
(``simul.adapters.blender_runtime``) applies the same rules to stored values;
it ships inside the Blender add-on and cannot import these models, and
``tests/blender/test_simready_metadata.py`` keeps the two in step.
"""

import difflib
from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
ColliderType = Literal[
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
]

#: What apply_simready_metadata's ``clear`` accepts: a field, a group,
#: additional_labels, or all.
SimReadyClearName = Literal[
    "semantic_class",
    "semantic_hierarchy",
    "semantic_qcode",
    "mass_kg",
    "collider_type",
    "static_friction",
    "dynamic_friction",
    "restitution",
    "density",
    "is_rigid_body",
    "substrate_type",
    "material_naming",
    "shader_type",
    "texel_density",
    "semantic",
    "physics",
    "material",
    "additional_labels",
    "all",
]

SimReadySeverity = Literal["all", "error", "warning"]

#: A label key is a USD semantics taxonomy name, so it must be an identifier.
LabelKey = Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]
NonEmpty = Annotated[str, Field(min_length=1)]


class _StrictModel(BaseModel):
    """A model that refuses unknown fields and names the closest valid one."""

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _refuse_unknown_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            known = list(cls.model_fields)
            for key in data:
                if key not in known:
                    close = difflib.get_close_matches(str(key), known, n=1, cutoff=0.5)
                    hint = f" (did you mean '{close[0]}'?)" if close else ""
                    raise ValueError(
                        f"unknown field '{key}'{hint}; valid fields: {', '.join(known)}"
                    )
        return data


class SimReadySemanticLabels(_StrictModel):
    """
    Semantic labeling for SimReady assets.

    export_simready_usd writes each field as a UsdSemantics LabelsAPI taxonomy
    (``class``, ``hierarchy``, ``qcode``, and each additional label's key), or
    with Isaac Sim's SemanticsAPI convention where Blender's USD has no
    UsdSemantics. Validation requires a semantic class on each mesh or an
    object above it.
    """

    semantic_class: Optional[NonEmpty] = Field(
        None, description="Human-readable classification, e.g. 'cup', 'truck'"
    )
    semantic_hierarchy: Optional[NonEmpty] = Field(
        None,
        description=(
            "Ordered relational hierarchy, "
            "e.g. 'machine/vehicle/emergency_vehicle/fire_engine'"
        ),
    )
    semantic_qcode: Optional[str] = Field(
        None, pattern=r"^Q[1-9][0-9]*$", description="Wikidata Q-code, e.g. 'Q1420'"
    )
    additional_labels: Optional[Dict[LabelKey, NonEmpty]] = Field(
        None,
        description=(
            "Custom labels such as sku or product_id; each key becomes a semantics "
            "taxonomy, so it must be an identifier"
        ),
    )


class SimReadyPhysicsProperties(_StrictModel):
    """
    Physics properties for SimReady assets.

    export_simready_usd writes them as UsdPhysics schemas: RigidBodyAPI,
    MassAPI, CollisionAPI with a MeshCollisionAPI approximation on the meshes,
    and a physics material (UsdPhysics.MaterialAPI) bound with the physics
    purpose. collider_type and the friction values also apply to meshes below
    the object that set none of their own.
    """

    mass_kg: Optional[float] = Field(None, ge=0.0, description="Mass in kilograms")
    collider_type: Optional[ColliderType] = Field(
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
    is_rigid_body: Optional[bool] = Field(
        None,
        description="Whether the object is a rigid body; false clears an earlier true",
    )


class SimReadyMaterialProperties(_StrictModel):
    """
    Material properties for SimReady assets.

    Stored as metadata only: export_simready_usd keeps them as userProperties
    attributes; no USD schema carries them.
    """

    substrate_type: Optional[NonEmpty] = Field(
        None,
        description=(
            "Physical substrate, e.g. metal, wood, plastic, glass, "
            "rubber, fabric, ceramic, concrete, stone"
        ),
    )
    material_naming: Optional[NonEmpty] = Field(
        None,
        description=(
            "SimReady material name: prefix_surfacetype_description, "
            "e.g. 'opaque_metal_brushed_aluminum'"
        ),
    )
    shader_type: Optional[NonEmpty] = Field(
        None,
        description="Target shader, e.g. OmniPBR, OmniGlass, SimPBR",
    )
    texel_density: Optional[float] = Field(
        None, gt=0.0, description="Target texel density in pixels per meter"
    )


class SimReadyMetadata(_StrictModel):
    """
    Combined SimReady metadata for an asset.

    Groups semantic, physics, and material properties into a single model
    that is stored as Blender custom properties with a ``simready_`` prefix
    and written by export_simready_usd.
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
    metadata: Optional[SimReadyMetadata] = Field(
        None, description="SimReady metadata to store; omitted fields keep their value"
    )
    clear: Optional[List[SimReadyClearName]] = Field(
        None,
        description=(
            "Stored values to remove before storing metadata: a field, a group, "
            "additional_labels, or all"
        ),
    )


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
    cleared_properties: List[str] = Field(
        default_factory=list, description="Custom property keys removed"
    )


class SimReadyGetMetadataRequest(BaseModel):
    """Request to read SimReady metadata from a Blender object."""

    object_name: str = Field(..., description="Name of the Blender object")


class SimReadyInvalidValue(BaseModel):
    """A stored simready_ property that breaks the metadata rules."""

    property: str = Field(..., description="Custom property name")
    value: Any = Field(None, description="The stored value")
    problem: str = Field(..., description="Why it is invalid")


class SimReadyGetMetadataResponse(BaseModel):
    """Response containing SimReady metadata for an object."""

    success: bool = Field(...)
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    object_name: str = Field(...)
    metadata: Optional[SimReadyMetadata] = Field(
        None, description="The valid stored metadata (null if none)"
    )
    has_simready_data: bool = Field(
        ..., description="Whether any simready_ properties exist"
    )
    invalid_values: List[SimReadyInvalidValue] = Field(
        default_factory=list,
        description="Stored values left out of metadata because they break the rules",
    )


class SimReadyValidationIssue(BaseModel):
    """A single compliance issue found during validation."""

    object_name: str = Field(..., description="Object with the issue")
    check: str = Field(
        ...,
        description=(
            "Check category: naming, scale, transforms, materials, hierarchy, "
            "metadata or geometry"
        ),
    )
    severity: str = Field(
        ..., description="error (makes the asset non-compliant) or warning"
    )
    message: str = Field(..., description="Human-readable description")
    suggestion: Optional[str] = Field(None, description="Suggested fix")


class SimReadyValidateRequest(BaseModel):
    """Request to validate objects against SimReady conventions."""

    object_names: Optional[List[str]] = Field(
        None,
        min_length=1,
        description=(
            "Objects to check; null checks the scene's meshes and every object "
            "above them, asset roots included"
        ),
    )
    check_naming: bool = Field(
        True, description="Check object, material and mesh-data names"
    )
    check_scale: bool = Field(
        True, description="Check real-world size in the scene's units"
    )
    check_transforms: bool = Field(True, description="Check for applied transforms")
    check_materials: bool = Field(True, description="Check material slots")
    check_hierarchy: bool = Field(True, description="Check hierarchy structure")
    check_metadata: bool = Field(
        True, description="Check semantic class, collider and stored values"
    )
    check_geometry: bool = Field(
        True, description="Check UVs, n-gons and degenerate faces"
    )
    severity: SimReadySeverity = Field(
        "all", description="Return all issues, or only errors or warnings"
    )
    offset: int = Field(0, ge=0, description="Index of the first issue returned")
    limit: int = Field(
        50, ge=1, le=500, description="Most issues returned; next_offset pages on"
    )


class SimReadyValidateResponse(BaseModel):
    """Response from SimReady compliance validation."""

    success: bool = Field(...)
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    compliant: bool = Field(
        ..., description="True when no check found an error (whatever the page)"
    )
    object_count: int = Field(..., description="Number of objects checked")
    issue_count: int = Field(..., description="Total issues found")
    error_count: int = Field(0, description="Issues of severity error")
    warning_count: int = Field(0, description="Issues of severity warning")
    issues: List[SimReadyValidationIssue] = Field(
        default_factory=list, description="This page of the issues"
    )
    offset: int = Field(0, description="Index of the first issue in this page")
    next_offset: Optional[int] = Field(
        None, description="offset for the next page; null on the last page"
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
        ..., min_length=1, description="Existing objects to put under the root"
    )
    semantic: Optional[SimReadySemanticLabels] = Field(
        None, description="Semantic labels for the root"
    )
    reuse_root: bool = Field(
        False, description="Put the objects under an existing empty named root_name"
    )


class SimReadyKeptParent(BaseModel):
    """A listed object left under its listed parent."""

    object_name: str = Field(..., description="Blender object name")
    parent: str = Field(..., description="Its parent, also listed")


class SimReadySetupHierarchyResponse(BaseModel):
    """Response after setting up SimReady hierarchy."""

    success: bool = Field(...)
    error: Optional[str] = Field(
        None, description="Error message when success is False"
    )
    root_name: str = Field(..., description="Root empty name")
    root_created: bool = Field(
        True, description="False when an existing root was reused"
    )
    children: List[str] = Field(..., description="The listed objects, now under the root")
    reparented: List[str] = Field(
        default_factory=list,
        description="Objects parented to the root, their world transforms kept",
    )
    kept_parent: List[SimReadyKeptParent] = Field(
        default_factory=list, description="Listed objects left under their listed parent"
    )
    hierarchy_path: str = Field(
        ..., description="The root's prim path in export_simready_usd's USD"
    )
    prim_paths: Dict[str, str] = Field(
        default_factory=dict, description="Each listed object's prim path in that USD"
    )


__all__ = [
    "COLLIDER_TYPES",
    "ColliderType",
    "SimReadyClearName",
    "SimReadySeverity",
    "SimReadySemanticLabels",
    "SimReadyPhysicsProperties",
    "SimReadyMaterialProperties",
    "SimReadyMetadata",
    "SimReadyApplyMetadataRequest",
    "SimReadyApplyMetadataResponse",
    "SimReadyGetMetadataRequest",
    "SimReadyInvalidValue",
    "SimReadyGetMetadataResponse",
    "SimReadyValidationIssue",
    "SimReadyValidateRequest",
    "SimReadyValidateResponse",
    "SimReadyExportRequest",
    "SimReadyExportedPrim",
    "SimReadySkippedObject",
    "SimReadyExportPhysics",
    "SimReadyExportSemantics",
    "SimReadyExportResponse",
    "SimReadySetupHierarchyRequest",
    "SimReadyKeptParent",
    "SimReadySetupHierarchyResponse",
]
