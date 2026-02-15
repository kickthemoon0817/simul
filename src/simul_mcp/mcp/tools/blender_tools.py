"""
Blender-specific MCP tools for Simul MCP Server.

This module provides read-focused Blender runtime tools backed by the
optional bpy module.
"""

from typing import Any, Dict, Optional

from ...adapters import BlenderRuntimeAdapter, is_blender_available
from ...config import Settings, get_settings
from ...logging import LoggerMixin, get_logger
from ..schemas import (
    BlenderInfoResponse,
    BlenderSceneObjectsRequest,
    BlenderSceneObjectsResponse,
    ErrorResponse,
)

logger = get_logger(__name__)


class BlenderTools(LoggerMixin):
    """Tools for Blender runtime operations."""

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize Blender tools.

        Args:
            settings: Configuration settings.
        """
        self.settings = settings or get_settings()
        self.blender_adapter = (
            BlenderRuntimeAdapter(self.settings) if is_blender_available() else None
        )

    async def get_blender_info(self) -> Dict[str, Any]:
        """
        Get Blender runtime information.

        Returns:
            Runtime information dictionary or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                runtime_info = session.get_runtime_info()
                runtime_info["success"] = True
                return BlenderInfoResponse(**runtime_info).dict()

        except Exception as e:
            self.logger.error(f"Error getting Blender runtime info: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def list_blender_scene_objects(
        self,
        collection_name: Optional[str] = None,
        include_hidden: bool = False,
        max_items: int = 200,
    ) -> Dict[str, Any]:
        """
        List objects from the active Blender scene.

        Args:
            collection_name: Optional collection name filter.
            include_hidden: Include hidden objects when true.
            max_items: Maximum number of objects to return.

        Returns:
            Scene object listing dictionary or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            request = BlenderSceneObjectsRequest(
                collection_name=collection_name,
                include_hidden=include_hidden,
                max_items=max_items or self.settings.blender.max_scene_objects,
            )

            with self.blender_adapter.create_session() as session:
                objects_payload = session.list_scene_objects(
                    collection_name=request.collection_name,
                    include_hidden=request.include_hidden,
                    max_items=request.max_items,
                )
                objects_payload["success"] = True
                return BlenderSceneObjectsResponse(**objects_payload).dict()

        except Exception as e:
            self.logger.error(f"Error listing Blender scene objects: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()
