"""
MCP (Model Context Protocol) server for 3D simulation and DCC tools.

This package provides the MCP server implementation with tool registry,
connection management, and integration for Isaac Sim, Blender, Unreal Engine,
and headless USD operations.

Every export is resolved on first access (PEP 562). The server pulls in
fastmcp, and the schema models belong to one backend each, so importing a
light submodule such as ``simul.mcp.tools`` (which the ``simul isaac`` CLI
does) or a server for one backend does not pay for the rest.
"""

import importlib
from typing import Any, Dict, List, Tuple

# Submodule -> the names this package re-exports from it.
_EXPORTS: Dict[str, Tuple[str, ...]] = {
    "server": ("SimulMCPServer", "create_server_instance", "start_mcp_server"),
    "schemas.common": ("BoundingBox",),
    "schemas.usd": (
        "MeshInfo",
        "PrimInfo",
        "SceneSummaryRequest",
        "SceneSummaryResponse",
        "StageInfo",
        "USDFileInfo",
    ),
    "schemas.blender": (
        "BlenderAddConstraintResponse",
        "BlenderAddForceFieldResponse",
        "BlenderAddModifierResponse",
        "BlenderAssignMaterialResponse",
        "BlenderBakeSimulationResponse",
        "BlenderBoundingBoxResponse",
        "BlenderBoundsCheckResponse",
        "BlenderCameraInfoResponse",
        "BlenderCaptureSequenceResponse",
        "BlenderCaptureViewportResponse",
        "BlenderClearParentResponse",
        "BlenderCreateMeshFromDataResponse",
        "BlenderCreateObjectResponse",
        "BlenderDeleteKeyframeResponse",
        "BlenderDeleteObjectResponse",
        "BlenderDistanceResponse",
        "BlenderExecuteScriptResponse",
        "BlenderExportFileResponse",
        "BlenderFileInfoResponse",
        "BlenderFocusOnObjectResponse",
        "BlenderFreeBakeResponse",
        "BlenderGetConstraintInfoResponse",
        "BlenderGetForceFieldInfoResponse",
        "BlenderGetFrameResponse",
        "BlenderGetKeyframesResponse",
        "BlenderGetPhysicsStateResponse",
        "BlenderGetTrajectoryResponse",
        "BlenderImportFileResponse",
        "BlenderInfoResponse",
        "BlenderInsertKeyframeResponse",
        "BlenderMaterialInfoResponse",
        "BlenderMeshInfoResponse",
        "BlenderObjectInfoResponse",
        "BlenderOpenFileResponse",
        "BlenderPlayAnimationResponse",
        "BlenderSaveFileResponse",
        "BlenderSceneObjectsRequest",
        "BlenderSceneObjectsResponse",
        "BlenderSceneSummaryResponse",
        "BlenderSearchObjectsResponse",
        "BlenderSetCameraViewResponse",
        "BlenderSetFrameRangeResponse",
        "BlenderSetFrameResponse",
        "BlenderSetLightParamsResponse",
        "BlenderSetParentResponse",
        "BlenderSetTransformResponse",
        "BlenderSetupRigidBodyResponse",
        "BlenderViewportInfoResponse",
    ),
    "schemas.simready": (
        "SimReadyApplyMetadataResponse",
        "SimReadyExportResponse",
        "SimReadyGetMetadataResponse",
        "SimReadyMetadata",
        "SimReadySemanticLabels",
        "SimReadySetupHierarchyResponse",
        "SimReadyValidateResponse",
        "SimReadyValidationIssue",
    ),
}

_SOURCES: Dict[str, str] = {
    name: module for module, names in _EXPORTS.items() for name in names
}


def __getattr__(name: str) -> Any:
    """Import an export from its submodule on first access and keep it."""
    module_name = _SOURCES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    # Server
    "SimulMCPServer",
    "start_mcp_server",
    "create_server_instance",
    # USD Schemas
    "USDFileInfo",
    "StageInfo",
    "PrimInfo",
    "MeshInfo",
    "BoundingBox",
    "SceneSummaryRequest",
    "SceneSummaryResponse",
    # Blender Schemas
    "BlenderCreateMeshFromDataResponse",
    "BlenderExecuteScriptResponse",
    "BlenderInfoResponse",
    "BlenderSceneObjectsRequest",
    "BlenderSceneObjectsResponse",
    "BlenderObjectInfoResponse",
    "BlenderMeshInfoResponse",
    "BlenderBoundingBoxResponse",
    "BlenderSearchObjectsResponse",
    "BlenderSceneSummaryResponse",
    "BlenderMaterialInfoResponse",
    "BlenderDistanceResponse",
    "BlenderBoundsCheckResponse",
    "BlenderCameraInfoResponse",
    "BlenderCaptureViewportResponse",
    "BlenderCaptureSequenceResponse",
    "BlenderFocusOnObjectResponse",
    "BlenderSetCameraViewResponse",
    "BlenderViewportInfoResponse",
    "BlenderCreateObjectResponse",
    "BlenderDeleteObjectResponse",
    "BlenderSetTransformResponse",
    "BlenderSetParentResponse",
    "BlenderClearParentResponse",
    "BlenderAssignMaterialResponse",
    "BlenderAddModifierResponse",
    "BlenderSetLightParamsResponse",
    "BlenderOpenFileResponse",
    "BlenderSaveFileResponse",
    "BlenderFileInfoResponse",
    "BlenderImportFileResponse",
    "BlenderExportFileResponse",
    "BlenderGetFrameResponse",
    "BlenderSetFrameResponse",
    "BlenderSetFrameRangeResponse",
    "BlenderPlayAnimationResponse",
    "BlenderInsertKeyframeResponse",
    "BlenderDeleteKeyframeResponse",
    "BlenderGetKeyframesResponse",
    "BlenderSetupRigidBodyResponse",
    "BlenderAddConstraintResponse",
    "BlenderGetConstraintInfoResponse",
    "BlenderAddForceFieldResponse",
    "BlenderGetForceFieldInfoResponse",
    "BlenderBakeSimulationResponse",
    "BlenderFreeBakeResponse",
    "BlenderGetPhysicsStateResponse",
    "BlenderGetTrajectoryResponse",
    # SimReady Schemas
    "SimReadySemanticLabels",
    "SimReadyMetadata",
    "SimReadyValidationIssue",
    "SimReadyApplyMetadataResponse",
    "SimReadyGetMetadataResponse",
    "SimReadyValidateResponse",
    "SimReadyExportResponse",
    "SimReadySetupHierarchyResponse",
]
