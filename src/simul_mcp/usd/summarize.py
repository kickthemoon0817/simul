"""
Scene summarization utilities for USD data in Isaac Sim MCP Server.

This module provides scene and prim summarization functionality to generate
concise, LLM-friendly descriptions of USD scenes and their contents.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import json

from ._pxr import Usd, UsdGeom

from ..logging import get_logger, LoggerMixin
from ..utils.timing import monitor_performance
from ..utils.math import BBox, bbox_center, bbox_size, bbox_volume
from .reader import USDReader, USDStageInfo, USDPrimInfo, PrimType
from .bbox import BBoxCache
from .mesh_ops import MeshOperations, MeshInfo

logger = get_logger(__name__)


@dataclass
class MaterialSummary:
    """Summary of a USD material."""
    path: str
    name: str
    shader_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    textures: List[str] = field(default_factory=list)


@dataclass
class PrimSummary:
    """Summary of a USD prim for LLM consumption."""
    path: str
    name: str
    type: str
    purpose: Optional[str]
    visibility: Optional[str]
    bbox: Optional[Dict[str, List[float]]]
    bbox_center: Optional[List[float]]
    bbox_size: Optional[List[float]]
    bbox_volume: Optional[float]
    transform: Optional[Dict[str, Any]]
    mesh_info: Optional[Dict[str, Any]]
    material_bindings: List[str] = field(default_factory=list)
    children_count: int = 0
    children_types: Dict[str, int] = field(default_factory=dict)
    attributes: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneSummary:
    """Summary of a USD scene for LLM consumption."""
    file_path: str
    stage_info: Dict[str, Any]
    total_prims: int
    prim_type_counts: Dict[str, int]
    scene_bbox: Optional[Dict[str, List[float]]]
    scene_center: Optional[List[float]]
    scene_size: Optional[List[float]]
    scene_volume: Optional[float]
    materials: List[MaterialSummary] = field(default_factory=list)
    root_prims: List[PrimSummary] = field(default_factory=list)
    mesh_statistics: Dict[str, Any] = field(default_factory=dict)
    hierarchy_depth: int = 0
    animation_info: Dict[str, Any] = field(default_factory=dict)


class SceneSummarizer(LoggerMixin):
    """
    USD scene summarizer for generating LLM-friendly scene descriptions.
    
    Provides functionality to analyze USD scenes and generate concise summaries
    suitable for consumption by language models.
    """
    
    def __init__(self):
        """Initialize scene summarizer."""
        self.usd_reader = USDReader()
        self.mesh_ops = MeshOperations()
        self.logger.info("Scene summarizer initialized")
    
    @monitor_performance("summarizer.summarize_stage")
    def summarize_stage(
        self, 
        stage: Usd.Stage, 
        file_path: str = "",
        include_meshes: bool = True,
        include_materials: bool = True,
        max_depth: int = 5
    ) -> SceneSummary:
        """
        Generate a comprehensive summary of a USD stage.
        
        Args:
            stage: USD Stage to summarize
            file_path: Path to the USD file
            include_meshes: Include detailed mesh information
            include_materials: Include material information
            max_depth: Maximum hierarchy depth to analyze
            
        Returns:
            SceneSummary object
        """
        try:
            # Get basic stage information
            stage_info = self.usd_reader.get_stage_info(stage)
            
            # Initialize bbox cache
            bbox_cache = BBoxCache(stage)
            
            # Count prims by type
            prim_type_counts = {}
            total_prims = 0
            
            for prim in stage.Traverse():
                total_prims += 1
                prim_type = prim.GetTypeName()
                prim_type_counts[prim_type] = prim_type_counts.get(prim_type, 0) + 1
            
            # Compute scene bounding box
            scene_bbox = bbox_cache.get_stage_bbox()
            scene_bbox_dict = None
            scene_center = None
            scene_size = None
            scene_volume = None
            
            if scene_bbox:
                from .bbox import bbox_to_dict
                scene_bbox_dict = bbox_to_dict(scene_bbox)
                scene_center = bbox_center(scene_bbox)
                scene_size = bbox_size(scene_bbox)
                scene_volume = bbox_volume(scene_bbox)
            
            # Summarize root prims
            root_prims = []
            for prim in stage.GetPseudoRoot().GetChildren():
                prim_summary = self.summarize_prim(
                    prim, 
                    bbox_cache=bbox_cache,
                    include_mesh_info=include_meshes,
                    max_depth=max_depth
                )
                root_prims.append(prim_summary)
            
            # Get materials
            materials = []
            if include_materials:
                materials = self._get_stage_materials(stage)
            
            # Get mesh statistics
            mesh_statistics = {}
            if include_meshes:
                mesh_statistics = self._get_mesh_statistics(stage)
            
            # Calculate hierarchy depth
            hierarchy_depth = self._calculate_hierarchy_depth(stage)
            
            # Get animation info
            animation_info = self._get_animation_info(stage_info)
            
            summary = SceneSummary(
                file_path=file_path,
                stage_info={
                    'up_axis': stage_info.up_axis,
                    'meters_per_unit': stage_info.meters_per_unit,
                    'time_codes_per_second': stage_info.time_codes_per_second,
                    'start_time': stage_info.start_time_code,
                    'end_time': stage_info.end_time_code,
                    'frame_rate': stage_info.frame_rate,
                    'layer_count': len(stage_info.layers)
                },
                total_prims=total_prims,
                prim_type_counts=prim_type_counts,
                scene_bbox=scene_bbox_dict,
                scene_center=scene_center,
                scene_size=scene_size,
                scene_volume=scene_volume,
                materials=materials,
                root_prims=root_prims,
                mesh_statistics=mesh_statistics,
                hierarchy_depth=hierarchy_depth,
                animation_info=animation_info
            )
            
            self.logger.info(f"Generated scene summary: {total_prims} prims, {len(root_prims)} root prims")
            return summary
            
        except Exception as e:
            self.logger.error(f"Error summarizing stage: {e}")
            raise
    
    @monitor_performance("summarizer.summarize_prim")
    def summarize_prim(
        self, 
        prim: Usd.Prim,
        bbox_cache: Optional[BBoxCache] = None,
        include_mesh_info: bool = True,
        max_depth: int = 5,
        current_depth: int = 0
    ) -> PrimSummary:
        """
        Generate a summary of a USD prim.
        
        Args:
            prim: USD Prim to summarize
            bbox_cache: Bounding box cache (created if None)
            include_mesh_info: Include detailed mesh information
            max_depth: Maximum recursion depth
            current_depth: Current recursion depth
            
        Returns:
            PrimSummary object
        """
        try:
            # Get basic prim info
            prim_info = self.usd_reader.get_prim_info(prim)
            
            # Initialize bbox cache if not provided
            if bbox_cache is None:
                bbox_cache = BBoxCache(prim.GetStage())
            
            # Compute bounding box
            bbox = bbox_cache.compute_world_bbox(prim)
            bbox_dict = None
            bbox_center_val = None
            bbox_size_val = None
            bbox_volume_val = None
            
            if bbox:
                from .bbox import bbox_to_dict
                bbox_dict = bbox_to_dict(bbox)
                bbox_center_val = bbox_center(bbox)
                bbox_size_val = bbox_size(bbox)
                bbox_volume_val = bbox_volume(bbox)
            
            # Get transform information
            transform_info = None
            if prim.IsA(UsdGeom.Xformable):
                transform_info = self._get_transform_info(prim)
            
            # Get mesh information
            mesh_info_dict = None
            if include_mesh_info and prim.IsA(UsdGeom.Mesh):
                try:
                    mesh_info = self.mesh_ops.get_mesh_statistics(prim)
                    mesh_info_dict = {
                        'vertex_count': mesh_info.vertex_count,
                        'face_count': mesh_info.face_count,
                        'has_normals': mesh_info.has_normals,
                        'has_uvs': mesh_info.has_uvs,
                        'has_colors': mesh_info.has_colors,
                        'surface_area': mesh_info.surface_area,
                        'volume': mesh_info.volume,
                        'is_closed': mesh_info.is_closed,
                        'topology_valid': mesh_info.topology_valid
                    }
                except Exception as e:
                    self.logger.debug(f"Could not get mesh info for {prim.GetPath()}: {e}")
            
            # Count children by type
            children_types = {}
            children_count = len(prim_info.children)
            
            if current_depth < max_depth:
                for child in prim.GetChildren():
                    child_type = child.GetTypeName()
                    children_types[child_type] = children_types.get(child_type, 0) + 1
            
            # Filter attributes for summary (keep only important ones)
            important_attributes = self._filter_important_attributes(prim_info.attributes)
            
            summary = PrimSummary(
                path=prim_info.path,
                name=prim_info.name,
                type=prim_info.type_name,
                purpose=prim_info.purpose,
                visibility=prim_info.visibility,
                bbox=bbox_dict,
                bbox_center=bbox_center_val,
                bbox_size=bbox_size_val,
                bbox_volume=bbox_volume_val,
                transform=transform_info,
                mesh_info=mesh_info_dict,
                material_bindings=self._get_material_bindings(prim),
                children_count=children_count,
                children_types=children_types,
                attributes=important_attributes,
                metadata=prim_info.metadata
            )
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error summarizing prim {prim.GetPath()}: {e}")
            raise
    
    def _get_stage_materials(self, stage: Usd.Stage) -> List[MaterialSummary]:
        """Get all materials in the stage."""
        materials = []
        
        try:
            for prim in stage.Traverse():
                if prim.GetTypeName() == "Material":
                    material_summary = MaterialSummary(
                        path=str(prim.GetPath()),
                        name=prim.GetName(),
                        shader_type="Unknown",  # Would need shader analysis
                        parameters={},
                        textures=[]
                    )
                    materials.append(material_summary)
        except Exception as e:
            self.logger.debug(f"Error getting stage materials: {e}")
        
        return materials
    
    def _get_mesh_statistics(self, stage: Usd.Stage) -> Dict[str, Any]:
        """Get overall mesh statistics for the stage."""
        stats = {
            'total_meshes': 0,
            'total_vertices': 0,
            'total_faces': 0,
            'meshes_with_normals': 0,
            'meshes_with_uvs': 0,
            'meshes_with_colors': 0,
            'closed_meshes': 0,
            'total_surface_area': 0.0,
            'total_volume': 0.0
        }
        
        try:
            for prim in stage.Traverse():
                if prim.IsA(UsdGeom.Mesh):
                    try:
                        mesh_info = self.mesh_ops.get_mesh_statistics(prim)
                        stats['total_meshes'] += 1
                        stats['total_vertices'] += mesh_info.vertex_count
                        stats['total_faces'] += mesh_info.face_count
                        if mesh_info.has_normals:
                            stats['meshes_with_normals'] += 1
                        if mesh_info.has_uvs:
                            stats['meshes_with_uvs'] += 1
                        if mesh_info.has_colors:
                            stats['meshes_with_colors'] += 1
                        if mesh_info.is_closed:
                            stats['closed_meshes'] += 1
                        stats['total_surface_area'] += mesh_info.surface_area
                        stats['total_volume'] += mesh_info.volume
                    except Exception as e:
                        self.logger.debug(f"Error getting mesh stats for {prim.GetPath()}: {e}")
        except Exception as e:
            self.logger.debug(f"Error calculating mesh statistics: {e}")
        
        return stats
    
    def _calculate_hierarchy_depth(self, stage: Usd.Stage) -> int:
        """Calculate the maximum hierarchy depth in the stage."""
        max_depth = 0
        
        try:
            for prim in stage.Traverse():
                path_depth = str(prim.GetPath()).count('/')
                max_depth = max(max_depth, path_depth)
        except Exception:
            pass
        
        return max_depth
    
    def _get_animation_info(self, stage_info: USDStageInfo) -> Dict[str, Any]:
        """Get animation information from stage info."""
        return {
            'has_animation': stage_info.start_time_code != stage_info.end_time_code,
            'start_time': stage_info.start_time_code,
            'end_time': stage_info.end_time_code,
            'duration': stage_info.end_time_code - stage_info.start_time_code,
            'frame_rate': stage_info.frame_rate,
            'time_codes_per_second': stage_info.time_codes_per_second
        }
    
    def _get_transform_info(self, prim: Usd.Prim) -> Dict[str, Any]:
        """Get transform information for a prim."""
        try:
            if prim.IsA(UsdGeom.Xformable):
                xformable = UsdGeom.Xformable(prim)
                transform = xformable.GetLocalTransformation()
                
                # Extract translation, rotation, scale
                translation = transform.ExtractTranslation()
                rotation = transform.ExtractRotation()
                scale = transform.ExtractScale()
                
                return {
                    'translation': [translation[0], translation[1], translation[2]],
                    'rotation': [rotation.GetReal(), rotation.GetImaginary()[0], 
                               rotation.GetImaginary()[1], rotation.GetImaginary()[2]],
                    'scale': [scale[0], scale[1], scale[2]]
                }
        except Exception as e:
            self.logger.debug(f"Could not get transform info for {prim.GetPath()}: {e}")
        
        return None
    
    def _get_material_bindings(self, prim: Usd.Prim) -> List[str]:
        """Get material bindings for a prim."""
        bindings = []
        
        try:
            if hasattr(UsdGeom, 'MaterialBindingAPI'):
                binding_api = UsdGeom.MaterialBindingAPI(prim)
                if binding_api:
                    material_rel = binding_api.GetDirectBindingRel()
                    if material_rel:
                        targets = material_rel.GetTargets()
                        bindings = [str(target) for target in targets]
        except Exception as e:
            self.logger.debug(f"Could not get material bindings for {prim.GetPath()}: {e}")
        
        return bindings
    
    def _filter_important_attributes(self, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Filter attributes to keep only important ones for summary."""
        important_attrs = {}
        
        # List of attribute names that are typically important
        important_names = [
            'points', 'normals', 'primvars:st', 'displayColor',
            'size', 'radius', 'height', 'width', 'depth',
            'focalLength', 'horizontalAperture', 'verticalAperture',
            'intensity', 'color', 'temperature', 'exposure'
        ]
        
        for name, value in attributes.items():
            if any(important in name for important in important_names):
                # Truncate large arrays for summary
                if isinstance(value, list) and len(value) > 10:
                    important_attrs[name] = f"Array[{len(value)}]"
                else:
                    important_attrs[name] = value
        
        return important_attrs


def generate_scene_digest(summary: SceneSummary) -> str:
    """
    Generate a concise text digest of the scene summary.
    
    Args:
        summary: SceneSummary object
        
    Returns:
        Human-readable scene digest
    """
    digest_parts = []
    
    # Basic scene info
    digest_parts.append(f"USD Scene: {summary.file_path}")
    digest_parts.append(f"Total prims: {summary.total_prims}")
    
    # Scene dimensions
    if summary.scene_size:
        size = summary.scene_size
        digest_parts.append(f"Scene size: {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} units")
    
    # Prim type breakdown
    if summary.prim_type_counts:
        top_types = sorted(summary.prim_type_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        type_str = ", ".join([f"{count} {ptype}" for ptype, count in top_types])
        digest_parts.append(f"Main content: {type_str}")
    
    # Mesh statistics
    if summary.mesh_statistics and summary.mesh_statistics.get('total_meshes', 0) > 0:
        mesh_stats = summary.mesh_statistics
        digest_parts.append(f"Meshes: {mesh_stats['total_meshes']} total, "
                          f"{mesh_stats['total_vertices']} vertices, "
                          f"{mesh_stats['total_faces']} faces")
    
    # Materials
    if summary.materials:
        digest_parts.append(f"Materials: {len(summary.materials)}")
    
    # Animation
    if summary.animation_info.get('has_animation', False):
        anim_info = summary.animation_info
        digest_parts.append(f"Animation: {anim_info['duration']:.1f}s at {anim_info['frame_rate']:.1f} FPS")
    
    return "\n".join(digest_parts)


def format_summary_for_llm(summary: SceneSummary) -> str:
    """
    Format scene summary for LLM consumption.
    
    Args:
        summary: SceneSummary object
        
    Returns:
        JSON string formatted for LLM
    """
    # Create a simplified version for LLM
    llm_summary = {
        'scene_overview': {
            'file_path': summary.file_path,
            'total_prims': summary.total_prims,
            'prim_types': summary.prim_type_counts,
            'scene_bounds': summary.scene_bbox,
            'scene_center': summary.scene_center,
            'scene_size': summary.scene_size
        },
        'content_summary': {
            'root_prims': len(summary.root_prims),
            'materials': len(summary.materials),
            'hierarchy_depth': summary.hierarchy_depth,
            'has_animation': summary.animation_info.get('has_animation', False)
        },
        'mesh_statistics': summary.mesh_statistics,
        'stage_settings': summary.stage_info
    }
    
    return json.dumps(llm_summary, indent=2)


# Convenience functions
def summarize_stage(stage: Usd.Stage, file_path: str = "") -> SceneSummary:
    """Summarize a USD stage."""
    summarizer = SceneSummarizer()
    return summarizer.summarize_stage(stage, file_path)


def summarize_prim(prim: Usd.Prim) -> PrimSummary:
    """Summarize a USD prim."""
    summarizer = SceneSummarizer()
    return summarizer.summarize_prim(prim)
