"""SimReady schemas for Blender and Unreal asset workflows."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

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
    "SimReadyExportRequest",
    "SimReadyExportResponse",
    "SimReadySetupHierarchyRequest",
    "SimReadySetupHierarchyResponse",
]
