"""
MCP (Model Context Protocol) server for 3D simulation and DCC tools.

This package provides the MCP server implementation with tool registry,
connection management, and integration for Isaac Sim, Blender, Unreal Engine,
and headless USD operations.
"""

from .schemas.blender import (
    BlenderAddConstraintResponse,
    BlenderAddForceFieldResponse,
    BlenderAddModifierResponse,
    BlenderAssignMaterialResponse,
    BlenderBakeSimulationResponse,
    BlenderBoundingBoxResponse,
    BlenderBoundsCheckResponse,
    BlenderCameraInfoResponse,
    BlenderCaptureSequenceResponse,
    BlenderCaptureViewportResponse,
    BlenderClearParentResponse,
    BlenderCreateMeshFromDataResponse,
    BlenderCreateObjectResponse,
    BlenderDeleteKeyframeResponse,
    BlenderDeleteObjectResponse,
    BlenderDistanceResponse,
    BlenderExecuteScriptResponse,
    BlenderExportFileResponse,
    BlenderFileInfoResponse,
    BlenderFocusOnObjectResponse,
    BlenderFreeBakeResponse,
    BlenderGetConstraintInfoResponse,
    BlenderGetForceFieldInfoResponse,
    BlenderGetFrameResponse,
    BlenderGetKeyframesResponse,
    BlenderGetPhysicsStateResponse,
    BlenderGetTrajectoryResponse,
    BlenderImportFileResponse,
    BlenderInfoResponse,
    BlenderInsertKeyframeResponse,
    BlenderMaterialInfoResponse,
    BlenderMeshInfoResponse,
    BlenderObjectInfoResponse,
    BlenderOpenFileResponse,
    BlenderPlayAnimationResponse,
    BlenderSaveFileResponse,
    BlenderSceneObjectsRequest,
    BlenderSceneObjectsResponse,
    BlenderSceneSummaryResponse,
    BlenderSearchObjectsResponse,
    BlenderSetCameraViewResponse,
    BlenderSetFrameRangeResponse,
    BlenderSetFrameResponse,
    BlenderSetLightParamsResponse,
    BlenderSetParentResponse,
    BlenderSetTransformResponse,
    BlenderSetupRigidBodyResponse,
    BlenderViewportInfoResponse,
)
from .schemas.common import BoundingBox
from .schemas.simready import (
    SimReadyApplyMetadataResponse,
    SimReadyExportResponse,
    SimReadyGetMetadataResponse,
    SimReadyMetadata,
    SimReadySemanticLabels,
    SimReadySetupHierarchyResponse,
    SimReadyValidateResponse,
    SimReadyValidationIssue,
)
from .schemas.usd import (
    MeshInfo,
    PrimInfo,
    SceneSummaryRequest,
    SceneSummaryResponse,
    StageInfo,
    USDFileInfo,
)
from typing import Any

# The server pulls in fastmcp and every backend adapter. Resolve its exports on
# first access (PEP 562) so importing a light submodule such as
# ``simul_mcp.mcp.tools`` — which the ``simul isaac`` CLI does — does not pay
# for the whole server.
_SERVER_EXPORTS = ("SimulMCPServer", "create_server_instance", "start_mcp_server")


def __getattr__(name: str) -> Any:
    if name in _SERVER_EXPORTS:
        from . import server

        value = getattr(server, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
