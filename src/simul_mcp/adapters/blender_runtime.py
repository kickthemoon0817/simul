"""
Blender runtime adapter for Simul MCP Server.

This module provides an adapter for Blender runtime operations through the
optional `bpy` Python module.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import bpy

    BLENDER_AVAILABLE = True
except ImportError:
    bpy = None
    BLENDER_AVAILABLE = False

from ..config import Settings, get_settings
from ..logging import LoggerMixin, get_logger

logger = get_logger(__name__)


@dataclass
class BlenderObjectEntry:
    """Serializable Blender object summary."""

    name: str
    object_type: str
    collection: Optional[str]
    visible: bool


class BlenderRuntimeSession(LoggerMixin):
    """
    Blender runtime session for scene inspection operations.

    This class provides read-focused operations that are safe for environments
    where Blender is running as a Python module.
    """

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize Blender runtime session.

        Args:
            settings: Configuration settings.
        """
        if not BLENDER_AVAILABLE:
            raise ImportError(
                "Blender runtime not available. Please run in an environment "
                "where the bpy module is installed."
            )

        self.settings = settings or get_settings()
        self.logger.info("Blender runtime session initialized")

    def get_runtime_info(self) -> Dict[str, Any]:
        """
        Get Blender runtime information.

        Returns:
            Dictionary containing Blender runtime metadata.
        """
        if bpy is None:
            raise RuntimeError("bpy module is unavailable during runtime info query")

        blender_module: Any = bpy
        blender_app = blender_module.app
        version_tuple = tuple(blender_app.version)
        version_string = blender_app.version_string
        executable_path = blender_app.binary_path
        blend_file_path = blender_module.data.filepath

        return {
            "version": list(version_tuple),
            "version_string": version_string,
            "binary_path": executable_path,
            "background": bool(blender_app.background),
            "blend_file_path": blend_file_path or None,
        }

    def list_scene_objects(
        self,
        collection_name: Optional[str] = None,
        include_hidden: bool = False,
        max_items: int = 200,
    ) -> Dict[str, Any]:
        """
        List scene objects from the active Blender data context.

        Args:
            collection_name: Optional collection name to filter objects.
            include_hidden: Include hidden objects when True.
            max_items: Maximum number of objects returned.

        Returns:
            Dictionary with object summaries and truncation metadata.
        """
        if max_items < 1:
            raise ValueError("max_items must be greater than zero")

        source_objects = self._resolve_object_source(collection_name)
        object_entries: List[BlenderObjectEntry] = []

        for scene_object in source_objects:
            object_visible = self._is_object_visible(scene_object)
            if not include_hidden and not object_visible:
                continue

            entry = BlenderObjectEntry(
                name=scene_object.name,
                object_type=scene_object.type,
                collection=collection_name,
                visible=object_visible,
            )
            object_entries.append(entry)

            if len(object_entries) >= max_items:
                break

        serialized_objects = [
            {
                "name": entry.name,
                "object_type": entry.object_type,
                "collection": entry.collection,
                "visible": entry.visible,
            }
            for entry in object_entries
        ]

        return {
            "collection": collection_name,
            "include_hidden": include_hidden,
            "max_items": max_items,
            "count": len(serialized_objects),
            "objects": serialized_objects,
            "truncated": len(serialized_objects) >= max_items,
        }

    def cleanup(self) -> None:
        """Clean up resources for the Blender session."""
        self.logger.debug("Blender runtime session cleaned up")

    @staticmethod
    def _is_object_visible(scene_object: Any) -> bool:
        """Determine object visibility in a version-compatible way."""
        if hasattr(scene_object, "visible_get"):
            try:
                return bool(scene_object.visible_get())
            except Exception:
                pass
        return not bool(getattr(scene_object, "hide_viewport", False))

    @staticmethod
    def _resolve_object_source(collection_name: Optional[str]) -> Any:
        """
        Resolve object iterable from collection or global object list.

        Args:
            collection_name: Optional collection to source objects from.

        Returns:
            Iterable of Blender objects.
        """
        if bpy is None:
            raise RuntimeError("bpy module is unavailable during object listing")

        blender_module: Any = bpy

        if not collection_name:
            return blender_module.data.objects

        collection = blender_module.data.collections.get(collection_name)
        if collection is None:
            raise ValueError(f"Collection not found: {collection_name}")
        return collection.objects


class BlenderRuntimeAdapter(LoggerMixin):
    """Adapter for Blender runtime operations."""

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize Blender runtime adapter.

        Args:
            settings: Configuration settings.
        """
        self.settings = settings or get_settings()
        self.logger.info("Blender runtime adapter initialized")

    @contextmanager
    def create_session(self) -> Any:
        """
        Create a Blender runtime session context manager.

        Yields:
            BlenderRuntimeSession instance.
        """
        session = BlenderRuntimeSession(self.settings)
        try:
            yield session
        finally:
            session.cleanup()

    def is_available(self) -> bool:
        """
        Check whether Blender runtime is available.

        Returns:
            True when bpy module is available.
        """
        return BLENDER_AVAILABLE and self.settings.blender.enabled

    def get_capabilities(self) -> List[str]:
        """
        Get list of Blender runtime capabilities.

        Returns:
            Capability list for Blender runtime adapter.
        """
        if not self.is_available():
            return []

        return [
            "blender_runtime_info",
            "blender_scene_listing",
        ]


def create_blender_session(
    settings: Optional[Settings] = None,
) -> BlenderRuntimeSession:
    """
    Create a Blender runtime session.

    Args:
        settings: Configuration settings.

    Returns:
        BlenderRuntimeSession instance.
    """
    return BlenderRuntimeSession(settings)


def is_blender_available() -> bool:
    """Check whether Blender runtime is available."""
    return BLENDER_AVAILABLE
