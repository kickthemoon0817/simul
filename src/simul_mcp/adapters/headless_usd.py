"""
Headless USD adapter for Isaac Sim MCP Server.

This module provides a headless USD adapter that works with pure pxr library
without requiring Isaac Sim runtime. Suitable for USD file analysis and
processing without GUI or simulation capabilities.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from contextlib import contextmanager

try:
    from pxr import Usd, UsdGeom, UsdShade, Sdf, Tf, Gf
    PXR_AVAILABLE = True
except ImportError:
    PXR_AVAILABLE = False
    Usd = None
    UsdGeom = None
    UsdShade = None
    Sdf = None
    Tf = None
    Gf = None

from ..logging import get_logger, LoggerMixin
from ..utils.timing import Timer, monitor_performance
from ..config import Settings, get_settings
from ..usd import USDReader, BBoxCache, MeshOperations, SceneSummarizer
from ..usd.reader import USDStageInfo, USDPrimInfo
from ..usd.summarize import SceneSummary

logger = get_logger(__name__)


class HeadlessUSDSession(LoggerMixin):
    """
    A headless USD session for file operations without Isaac Sim runtime.
    
    Provides USD file loading, analysis, and manipulation capabilities
    using only the pxr library.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize headless USD session.
        
        Args:
            settings: Configuration settings
        """
        if not PXR_AVAILABLE:
            raise ImportError("pxr library not available. Please install USD Python bindings.")
        
        self.settings = settings or get_settings()
        self.usd_reader = USDReader(enable_caching=self.settings.usd.cache_enabled)
        self.mesh_ops = MeshOperations()
        self.summarizer = SceneSummarizer()
        
        self._active_stages: Dict[str, Usd.Stage] = {}
        self._bbox_caches: Dict[str, BBoxCache] = {}
        
        self.logger.info("Headless USD session initialized")
    
    @monitor_performance("headless_session.load_stage")
    def load_stage(self, file_path: Union[str, Path]) -> Optional[str]:
        """
        Load a USD stage from file.
        
        Args:
            file_path: Path to USD file
            
        Returns:
            Stage identifier or None if failed
        """
        file_path = str(Path(file_path).resolve())
        
        # Validate file
        if not self._validate_usd_file(file_path):
            return None
        
        try:
            stage = self.usd_reader.open_stage(file_path)
            if stage:
                stage_id = self._generate_stage_id(file_path)
                self._active_stages[stage_id] = stage
                self._bbox_caches[stage_id] = BBoxCache(stage)
                
                self.logger.info(f"Loaded USD stage: {file_path} -> {stage_id}")
                return stage_id
            else:
                self.logger.error(f"Failed to load USD stage: {file_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error loading USD stage {file_path}: {e}")
            return None
    
    def unload_stage(self, stage_id: str) -> bool:
        """
        Unload a USD stage.
        
        Args:
            stage_id: Stage identifier
            
        Returns:
            True if successfully unloaded
        """
        try:
            if stage_id in self._active_stages:
                del self._active_stages[stage_id]
                if stage_id in self._bbox_caches:
                    del self._bbox_caches[stage_id]
                
                self.logger.info(f"Unloaded USD stage: {stage_id}")
                return True
            else:
                self.logger.warning(f"Stage not found: {stage_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error unloading stage {stage_id}: {e}")
            return False
    
    def get_stage(self, stage_id: str) -> Optional[Usd.Stage]:
        """
        Get a loaded USD stage.
        
        Args:
            stage_id: Stage identifier
            
        Returns:
            USD Stage or None if not found
        """
        return self._active_stages.get(stage_id)
    
    def list_stages(self) -> List[str]:
        """
        List all loaded stages.
        
        Returns:
            List of stage identifiers
        """
        return list(self._active_stages.keys())
    
    @monitor_performance("headless_session.get_stage_info")
    def get_stage_info(self, stage_id: str) -> Optional[USDStageInfo]:
        """
        Get information about a loaded stage.
        
        Args:
            stage_id: Stage identifier
            
        Returns:
            USDStageInfo or None if stage not found
        """
        stage = self.get_stage(stage_id)
        if stage:
            return self.usd_reader.get_stage_info(stage)
        return None
    
    @monitor_performance("headless_session.get_prim_info")
    def get_prim_info(self, stage_id: str, prim_path: str) -> Optional[USDPrimInfo]:
        """
        Get information about a prim in a stage.
        
        Args:
            stage_id: Stage identifier
            prim_path: Path to the prim
            
        Returns:
            USDPrimInfo or None if not found
        """
        stage = self.get_stage(stage_id)
        if stage:
            prim = stage.GetPrimAtPath(prim_path)
            if prim and prim.IsValid():
                return self.usd_reader.get_prim_info(prim)
        return None

    def get_prim_transform(self, stage_id: str, prim_path: str) -> Optional[Dict[str, List[float]]]:
        """Get local transform for a prim."""
        stage = self.get_stage(stage_id)
        if not stage:
            return None

        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Xformable):
            return None

        xformable = UsdGeom.Xformable(prim)
        matrix, _ = xformable.GetLocalTransformation()
        transform = Gf.Transform(matrix)

        translation = transform.GetTranslation()
        rotation = transform.GetRotation().GetQuat()
        scale = transform.GetScale()

        return {
            "translation": [translation[0], translation[1], translation[2]],
            "rotation": [rotation.GetReal(), rotation.GetImaginary()[0], rotation.GetImaginary()[1], rotation.GetImaginary()[2]],
            "scale": [scale[0], scale[1], scale[2]],
        }

    def get_children_type_counts(self, stage_id: str, prim_path: str) -> Dict[str, int]:
        """Get counts of child prim types."""
        stage = self.get_stage(stage_id)
        if not stage:
            return {}

        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return {}

        counts: Dict[str, int] = {}
        for child in prim.GetChildren():
            type_name = child.GetTypeName() or "Unknown"
            counts[type_name] = counts.get(type_name, 0) + 1
        return counts

    def get_material_bindings(self, stage_id: str, prim_path: str) -> List[str]:
        """Get bound material paths for a prim."""
        if not UsdShade:
            return []

        stage = self.get_stage(stage_id)
        if not stage:
            return []

        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return []

        binding_api = UsdShade.MaterialBindingAPI(prim)
        material, _ = binding_api.ComputeBoundMaterial()
        if material:
            return [str(material.GetPath())]
        return []

    def _infer_value_type(self, value: Any):
        if not Sdf:
            return None
        if isinstance(value, bool):
            return Sdf.ValueTypeNames.Bool
        if isinstance(value, int) and not isinstance(value, bool):
            return Sdf.ValueTypeNames.Int
        if isinstance(value, float):
            return Sdf.ValueTypeNames.Float
        if isinstance(value, str):
            return Sdf.ValueTypeNames.String
        if isinstance(value, (list, tuple)):
            if not value:
                return Sdf.ValueTypeNames.StringArray
            if all(isinstance(v, bool) for v in value):
                return Sdf.ValueTypeNames.BoolArray
            if all(isinstance(v, int) and not isinstance(v, bool) for v in value):
                return Sdf.ValueTypeNames.IntArray
            if all(isinstance(v, (int, float)) for v in value):
                if len(value) == 2:
                    return Sdf.ValueTypeNames.Float2
                if len(value) == 3:
                    return Sdf.ValueTypeNames.Float3
                if len(value) == 4:
                    return Sdf.ValueTypeNames.Float4
                return Sdf.ValueTypeNames.FloatArray
            if all(isinstance(v, str) for v in value):
                return Sdf.ValueTypeNames.StringArray
        return None

    def _set_attribute(self, prim: Usd.Prim, name: str, value: Any) -> bool:
        attr = prim.GetAttribute(name)
        if not attr:
            value_type = self._infer_value_type(value)
            if not value_type:
                return False
            attr = prim.CreateAttribute(name, value_type)
        try:
            attr.Set(value)
            return True
        except Exception:
            return False

    def create_prim(
        self,
        stage_id: str,
        prim_path: str,
        prim_type: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> bool:
        stage = self.get_stage(stage_id)
        if not stage:
            return False
        if not prim_path or not prim_type:
            return False

        prim = stage.DefinePrim(prim_path, prim_type)
        if not prim or not prim.IsValid():
            return False

        if attributes:
            for name, value in attributes.items():
                if not self._set_attribute(prim, name, value):
                    return False
        return True

    def update_prim_attributes(
        self,
        stage_id: str,
        prim_path: str,
        attributes: Dict[str, Any],
    ) -> bool:
        stage = self.get_stage(stage_id)
        if not stage:
            return False
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return False
        if not attributes:
            return True
        for name, value in attributes.items():
            if not self._set_attribute(prim, name, value):
                return False
        return True

    def delete_prim(self, stage_id: str, prim_path: str) -> bool:
        stage = self.get_stage(stage_id)
        if not stage:
            return False
        if not prim_path:
            return False
        stage.RemovePrim(prim_path)
        prim = stage.GetPrimAtPath(prim_path)
        return not prim or not prim.IsValid()
 
    @monitor_performance("headless_session.summarize_stage")


    def summarize_stage(self, stage_id: str, include_meshes: bool = True) -> Optional[SceneSummary]:
        """
        Generate a summary of a loaded stage.
        
        Args:
            stage_id: Stage identifier
            include_meshes: Include detailed mesh information
            
        Returns:
            SceneSummary or None if stage not found
        """
        stage = self.get_stage(stage_id)
        if stage:
            # Get the original file path if available
            file_path = ""
            if stage_id in self._active_stages:
                root_layer = stage.GetRootLayer()
                file_path = root_layer.identifier
            
            return self.summarizer.summarize_stage(
                stage, 
                file_path=file_path,
                include_meshes=include_meshes
            )
        return None
    
    def find_prims_by_type(self, stage_id: str, prim_type: str) -> List[str]:
        """
        Find prims of a specific type in a stage.
        
        Args:
            stage_id: Stage identifier
            prim_type: Prim type to search for
            
        Returns:
            List of prim paths
        """
        stage = self.get_stage(stage_id)
        if stage:
            prims = self.usd_reader.find_prims_by_type(stage, prim_type)
            return [str(prim.GetPath()) for prim in prims]
        return []
    
    def find_prims_by_name(self, stage_id: str, name_pattern: str, exact_match: bool = False) -> List[str]:
        """
        Find prims by name pattern in a stage.
        
        Args:
            stage_id: Stage identifier
            name_pattern: Name pattern to search for
            exact_match: Use exact matching
            
        Returns:
            List of prim paths
        """
        stage = self.get_stage(stage_id)
        if stage:
            prims = self.usd_reader.find_prims_by_name(stage, name_pattern, exact_match)
            return [str(prim.GetPath()) for prim in prims]
        return []
    
    def get_prim_bbox(self, stage_id: str, prim_path: str, world_space: bool = True) -> Optional[Dict[str, List[float]]]:
        """
        Get bounding box for a prim.
        
        Args:
            stage_id: Stage identifier
            prim_path: Path to the prim
            world_space: Return world-space bbox
            
        Returns:
            Bounding box dictionary or None
        """
        stage = self.get_stage(stage_id)
        bbox_cache = self._bbox_caches.get(stage_id)
        
        if stage and bbox_cache:
            prim = stage.GetPrimAtPath(prim_path)
            if prim and prim.IsValid():
                if world_space:
                    bbox = bbox_cache.compute_world_bbox(prim)
                else:
                    bbox = bbox_cache.compute_local_bbox(prim)
                
                if bbox:
                    from ..usd.bbox import bbox_to_dict
                    return bbox_to_dict(bbox)
        
        return None
    
    def get_stage_bbox(self, stage_id: str) -> Optional[Dict[str, List[float]]]:
        """
        Get bounding box for entire stage.
        
        Args:
            stage_id: Stage identifier
            
        Returns:
            Bounding box dictionary or None
        """
        bbox_cache = self._bbox_caches.get(stage_id)
        if bbox_cache:
            bbox = bbox_cache.get_stage_bbox()
            if bbox:
                from ..usd.bbox import bbox_to_dict
                return bbox_to_dict(bbox)
        return None
    
    def get_mesh_info(self, stage_id: str, prim_path: str) -> Optional[Dict[str, Any]]:
        """
        Get mesh information for a mesh prim.
        
        Args:
            stage_id: Stage identifier
            prim_path: Path to mesh prim
            
        Returns:
            Mesh information dictionary or None
        """
        stage = self.get_stage(stage_id)
        if stage:
            prim = stage.GetPrimAtPath(prim_path)
            if prim and prim.IsValid() and prim.IsA(UsdGeom.Mesh):
                try:
                    mesh_info = self.mesh_ops.get_mesh_statistics(prim)
                    return {
                        "prim_path": mesh_info.prim_path,
                        "vertex_count": mesh_info.vertex_count,
                        "face_count": mesh_info.face_count,
                        "point_count": mesh_info.point_count,
                        "has_normals": mesh_info.has_normals,
                        "has_uvs": mesh_info.has_uvs,
                        "has_colors": mesh_info.has_colors,
                        "has_tangents": mesh_info.has_tangents,
                        "subdivision_scheme": mesh_info.subdivision_scheme,
                        "topology_valid": mesh_info.topology_valid,
                        "is_closed": mesh_info.is_closed,
                        "surface_area": mesh_info.surface_area,
                        "volume": mesh_info.volume,
                        "materials": mesh_info.materials,
                        "subsets": mesh_info.subsets,
                        "bbox": mesh_info.bbox,
                    }
                except Exception as e:
                    self.logger.error(f"Error getting mesh info for {prim_path}: {e}")
        return None
    
    def _validate_usd_file(self, file_path: str) -> bool:
        """Validate USD file before loading."""
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            self.logger.error(f"USD file not found: {file_path}")
            return False
        
        # Check file extension
        if not any(file_path.lower().endswith(ext) for ext in self.settings.usd.allowed_extensions):
            self.logger.error(f"Invalid USD file extension: {file_path}")
            return False
        
        # Check file size
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.settings.usd.max_file_size_mb:
            self.logger.error(f"USD file too large: {file_size_mb:.1f}MB > {self.settings.usd.max_file_size_mb}MB")
            return False
        
        return True
    
    def _generate_stage_id(self, file_path: str) -> str:
        """Generate a unique stage identifier."""
        import hashlib
        path_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
        filename = Path(file_path).stem
        return f"{filename}_{path_hash}"
    
    def cleanup(self) -> None:
        """Clean up all resources."""
        try:
            self._active_stages.clear()
            self._bbox_caches.clear()
            self.usd_reader.clear_cache()
            self.logger.info("Headless USD session cleaned up")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


class HeadlessUSDAdapter(LoggerMixin):
    """
    Adapter for headless USD operations.
    
    Provides a high-level interface for USD operations without Isaac Sim runtime.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize headless USD adapter.
        
        Args:
            settings: Configuration settings
        """
        self.settings = settings or get_settings()
        self.logger.info("Headless USD adapter initialized")
    
    @contextmanager
    def create_session(self):
        """
        Create a headless USD session context manager.
        
        Yields:
            HeadlessUSDSession instance
        """
        session = HeadlessUSDSession(self.settings)
        try:
            yield session
        finally:
            session.cleanup()
    
    def is_available(self) -> bool:
        """
        Check if headless USD operations are available.
        
        Returns:
            True if pxr library is available
        """
        return PXR_AVAILABLE
    
    def get_capabilities(self) -> List[str]:
        """
        Get list of supported capabilities.
        
        Returns:
            List of capability names
        """
        capabilities = [
            "load_usd_files",
            "analyze_scene_structure", 
            "extract_mesh_data",
            "compute_bounding_boxes",
            "generate_scene_summaries",
            "search_prims",
            "validate_topology"
        ]
        
        if not PXR_AVAILABLE:
            capabilities = []
        
        return capabilities


# Convenience functions
def create_headless_session(settings: Optional[Settings] = None) -> HeadlessUSDSession:
    """
    Create a headless USD session.
    
    Args:
        settings: Configuration settings
        
    Returns:
        HeadlessUSDSession instance
    """
    return HeadlessUSDSession(settings)


def is_headless_available() -> bool:
    """Check if headless USD operations are available."""
    return PXR_AVAILABLE
