"""
MCP (Model Context Protocol) server implementation for Isaac Sim.

This package provides the MCP server implementation with tool registry,
connection management, and Isaac Sim integration.
"""

from .server import (
    IsaacMCPServer,
    start_mcp_server,
    create_server_instance,
)

from .schemas import (
    USDFileInfo,
    StageInfo,
    PrimInfo,
    MeshInfo,
    BoundingBox,
    SceneSummaryRequest,
    SceneSummaryResponse,
    BlenderInfoResponse,
    BlenderSceneObjectsRequest,
    BlenderSceneObjectsResponse,
)

__all__ = [
    # Server
    "IsaacMCPServer",
    "start_mcp_server",
    "create_server_instance",
    # Schemas
    "USDFileInfo",
    "StageInfo",
    "PrimInfo",
    "MeshInfo",
    "BoundingBox",
    "SceneSummaryRequest",
    "SceneSummaryResponse",
    "BlenderInfoResponse",
    "BlenderSceneObjectsRequest",
    "BlenderSceneObjectsResponse",
]
