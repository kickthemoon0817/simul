"""Common MCP schemas shared across backends."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


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
        ..., description="Minimum point [x, y, z]", min_length=3, max_length=3
    )
    max: List[float] = Field(
        ..., description="Maximum point [x, y, z]", min_length=3, max_length=3
    )
    center: Optional[List[float]] = Field(
        None, description="Center point [x, y, z]", min_length=3, max_length=3
    )
    size: Optional[List[float]] = Field(
        None, description="Size [width, height, depth]", min_length=3, max_length=3
    )
    volume: Optional[float] = Field(None, description="Bounding box volume")


class Transform(BaseModel):
    """3D transformation representation."""

    translation: List[float] = Field(
        ..., description="Translation [x, y, z]", min_length=3, max_length=3
    )
    rotation: List[float] = Field(
        ..., description="Rotation quaternion [w, x, y, z]", min_length=4, max_length=4
    )
    scale: List[float] = Field(
        ..., description="Scale [x, y, z]", min_length=3, max_length=3
    )


class ErrorResponse(BaseModel):
    """Generic error response."""

    success: bool = Field(False, description="Always false for error responses")
    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Error type")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional error details"
    )

__all__ = [
    "PrimType",
    "BoundingBox",
    "Transform",
    "ErrorResponse",
]
