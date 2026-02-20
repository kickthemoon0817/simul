"""
Bounding box utilities for USD data in Isaac Sim MCP Server.

This module provides bounding box computation, caching, and utilities
for USD prims and stages using the pxr library.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np

try:
    from pxr import Usd, UsdGeom, Gf
    PXR_AVAILABLE = True
except ImportError:
    PXR_AVAILABLE = False
    # Mock classes for development
    Usd = None
    UsdGeom = None
    Gf = None

from ..logging import get_logger, LoggerMixin
from ..utils.timing import monitor_performance
from ..utils.math import BBox, bbox_union, bbox_from_points

logger = get_logger(__name__)


class BBoxCache(LoggerMixin):
    """
    Bounding box cache for USD prims.
    
    Provides efficient bounding box computation and caching for USD prims
    with support for world and local coordinate spaces.
    """
    
    def __init__(self, stage: Usd.Stage, time_code: float = Usd.TimeCode.Default()):
        """
        Initialize bounding box cache.
        
        Args:
            stage: USD Stage
            time_code: Time code for animated data
        """
        if not PXR_AVAILABLE:
            raise ImportError("pxr library not available. Please install USD Python bindings.")
        
        self.stage = stage
        self.time_code = time_code
        self._world_bbox_cache: Dict[str, BBox] = {}
        self._local_bbox_cache: Dict[str, BBox] = {}
        self._usd_bbox_cache: Optional[UsdGeom.BBoxCache] = None
        
        # Initialize USD's built-in bbox cache
        try:
            self._usd_bbox_cache = UsdGeom.BBoxCache(time_code, includedPurposes=[UsdGeom.Tokens.default_])
        except Exception as e:
            self.logger.warning(f"Could not initialize USD BBoxCache: {e}")
        
        self.logger.info(f"BBox cache initialized for stage at time {time_code}")
    
    def clear_cache(self) -> None:
        """Clear all cached bounding boxes."""
        self._world_bbox_cache.clear()
        self._local_bbox_cache.clear()
        if self._usd_bbox_cache:
            self._usd_bbox_cache.Clear()
        self.logger.debug("Cleared bounding box cache")
    
    @monitor_performance("bbox_cache.compute_world_bbox")
    def compute_world_bbox(self, prim: Usd.Prim, use_cache: bool = True) -> Optional[BBox]:
        """
        Compute world-space bounding box for a prim.
        
        Args:
            prim: USD Prim
            use_cache: Use cached result if available
            
        Returns:
            Bounding box in world coordinates or None if computation failed
        """
        prim_path = str(prim.GetPath())
        
        # Check cache first
        if use_cache and prim_path in self._world_bbox_cache:
            return self._world_bbox_cache[prim_path]
        
        try:
            bbox = None
            
            # Try using USD's built-in bbox cache first
            if self._usd_bbox_cache:
                try:
                    usd_bbox = self._usd_bbox_cache.ComputeWorldBound(prim)
                    if not usd_bbox.IsEmpty():
                        min_point = usd_bbox.GetMin()
                        max_point = usd_bbox.GetMax()
                        bbox = ([min_point[0], min_point[1], min_point[2]], 
                               [max_point[0], max_point[1], max_point[2]])
                except Exception as e:
                    self.logger.debug(f"USD BBoxCache failed for {prim_path}: {e}")
            
            # Fallback to manual computation
            if bbox is None:
                bbox = self._compute_world_bbox_manual(prim)
            
            # Cache the result
            if bbox and use_cache:
                self._world_bbox_cache[prim_path] = bbox
            
            return bbox
            
        except Exception as e:
            self.logger.error(f"Error computing world bbox for {prim_path}: {e}")
            return None
    
    @monitor_performance("bbox_cache.compute_local_bbox")
    def compute_local_bbox(self, prim: Usd.Prim, use_cache: bool = True) -> Optional[BBox]:
        """
        Compute local-space bounding box for a prim.
        
        Args:
            prim: USD Prim
            use_cache: Use cached result if available
            
        Returns:
            Bounding box in local coordinates or None if computation failed
        """
        prim_path = str(prim.GetPath())
        
        # Check cache first
        if use_cache and prim_path in self._local_bbox_cache:
            return self._local_bbox_cache[prim_path]
        
        try:
            bbox = None
            
            # Try using USD's built-in bbox cache first
            if self._usd_bbox_cache:
                try:
                    usd_bbox = self._usd_bbox_cache.ComputeLocalBound(prim)
                    if not usd_bbox.IsEmpty():
                        min_point = usd_bbox.GetMin()
                        max_point = usd_bbox.GetMax()
                        bbox = ([min_point[0], min_point[1], min_point[2]], 
                               [max_point[0], max_point[1], max_point[2]])
                except Exception as e:
                    self.logger.debug(f"USD BBoxCache failed for {prim_path}: {e}")
            
            # Fallback to manual computation
            if bbox is None:
                bbox = self._compute_local_bbox_manual(prim)
            
            # Cache the result
            if bbox and use_cache:
                self._local_bbox_cache[prim_path] = bbox
            
            return bbox
            
        except Exception as e:
            self.logger.error(f"Error computing local bbox for {prim_path}: {e}")
            return None
    
    def _compute_world_bbox_manual(self, prim: Usd.Prim) -> Optional[BBox]:
        """Manually compute world bounding box."""
        try:
            # Get local bbox first
            local_bbox = self._compute_local_bbox_manual(prim)
            if not local_bbox:
                return None
            
            # Get world transform
            if prim.IsA(UsdGeom.Imageable):
                imageable = UsdGeom.Imageable(prim)
                world_transform = imageable.ComputeLocalToWorldTransform(self.time_code)
                
                # Transform bbox corners
                min_point, max_point = local_bbox
                corners = [
                    [min_point[0], min_point[1], min_point[2]],
                    [max_point[0], min_point[1], min_point[2]],
                    [min_point[0], max_point[1], min_point[2]],
                    [max_point[0], max_point[1], min_point[2]],
                    [min_point[0], min_point[1], max_point[2]],
                    [max_point[0], min_point[1], max_point[2]],
                    [min_point[0], max_point[1], max_point[2]],
                    [max_point[0], max_point[1], max_point[2]],
                ]
                
                # Transform all corners
                transformed_corners = []
                for corner in corners:
                    point = Gf.Vec3f(corner[0], corner[1], corner[2])
                    transformed = world_transform.Transform(point)
                    transformed_corners.append([transformed[0], transformed[1], transformed[2]])
                
                # Compute bbox from transformed corners
                return bbox_from_points(transformed_corners)
            else:
                # For non-imageable prims, world bbox is same as local
                return local_bbox
                
        except Exception as e:
            self.logger.debug(f"Manual world bbox computation failed: {e}")
            return None
    
    def _compute_local_bbox_manual(self, prim: Usd.Prim) -> Optional[BBox]:
        """Manually compute local bounding box."""
        try:
            if prim.IsA(UsdGeom.Mesh):
                # For mesh prims, compute from vertices
                mesh = UsdGeom.Mesh(prim)
                points_attr = mesh.GetPointsAttr()
                if points_attr:
                    points = points_attr.Get(self.time_code)
                    if points:
                        points_list = [[p[0], p[1], p[2]] for p in points]
                        return bbox_from_points(points_list)
            
            elif prim.IsA(UsdGeom.Cube):
                # For cube prims, use size attribute
                cube = UsdGeom.Cube(prim)
                size_attr = cube.GetSizeAttr()
                size = size_attr.Get(self.time_code) if size_attr else 2.0
                half_size = size / 2.0
                return ([-half_size, -half_size, -half_size], [half_size, half_size, half_size])
            
            elif prim.IsA(UsdGeom.Sphere):
                # For sphere prims, use radius attribute
                sphere = UsdGeom.Sphere(prim)
                radius_attr = sphere.GetRadiusAttr()
                radius = radius_attr.Get(self.time_code) if radius_attr else 1.0
                return ([-radius, -radius, -radius], [radius, radius, radius])
            
            elif prim.IsA(UsdGeom.Cylinder):
                # For cylinder prims, use radius and height attributes
                cylinder = UsdGeom.Cylinder(prim)
                radius_attr = cylinder.GetRadiusAttr()
                height_attr = cylinder.GetHeightAttr()
                radius = radius_attr.Get(self.time_code) if radius_attr else 1.0
                height = height_attr.Get(self.time_code) if height_attr else 2.0
                half_height = height / 2.0
                return ([-radius, -half_height, -radius], [radius, half_height, radius])
            
            elif prim.IsA(UsdGeom.Xform):
                # For Xform prims, compute union of children bboxes
                child_bboxes = []
                for child in prim.GetChildren():
                    child_bbox = self.compute_local_bbox(child, use_cache=False)
                    if child_bbox:
                        child_bboxes.append(child_bbox)
                
                if child_bboxes:
                    # Compute union of all child bboxes
                    union_bbox = child_bboxes[0]
                    for bbox in child_bboxes[1:]:
                        union_bbox = bbox_union(union_bbox, bbox)
                    return union_bbox
            
            # Default fallback - return unit cube
            return ([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
            
        except Exception as e:
            self.logger.debug(f"Manual local bbox computation failed: {e}")
            return None
    
    def get_stage_bbox(self) -> Optional[BBox]:
        """
        Compute bounding box for the entire stage.
        
        Returns:
            Stage bounding box or None if computation failed
        """
        try:
            root_prims = [prim for prim in self.stage.GetPseudoRoot().GetChildren()]
            if not root_prims:
                return None
            
            # Compute union of all root prim bboxes
            stage_bbox = None
            for prim in root_prims:
                prim_bbox = self.compute_world_bbox(prim)
                if prim_bbox:
                    if stage_bbox is None:
                        stage_bbox = prim_bbox
                    else:
                        stage_bbox = bbox_union(stage_bbox, prim_bbox)
            
            return stage_bbox
            
        except Exception as e:
            self.logger.error(f"Error computing stage bbox: {e}")
            return None


def bbox_to_dict(bbox: BBox) -> Dict[str, List[float]]:
    """
    Convert bounding box to dictionary format.
    
    Args:
        bbox: Bounding box tuple
        
    Returns:
        Dictionary with min and max points
    """
    min_point, max_point = bbox
    return {
        'min': list(min_point),
        'max': list(max_point)
    }


def dict_to_bbox(bbox_dict: Dict[str, List[float]]) -> BBox:
    """
    Convert dictionary to bounding box format.
    
    Args:
        bbox_dict: Dictionary with min and max points
        
    Returns:
        Bounding box tuple
    """
    return (bbox_dict['min'], bbox_dict['max'])


# Convenience functions
def compute_world_bbox(prim: Usd.Prim, time_code: float = Usd.TimeCode.Default()) -> Optional[BBox]:
    """Compute world bounding box for a prim."""
    cache = BBoxCache(prim.GetStage(), time_code)
    return cache.compute_world_bbox(prim)


def compute_local_bbox(prim: Usd.Prim, time_code: float = Usd.TimeCode.Default()) -> Optional[BBox]:
    """Compute local bounding box for a prim."""
    cache = BBoxCache(prim.GetStage(), time_code)
    return cache.compute_local_bbox(prim)


def get_prim_bbox(prim: Usd.Prim, world_space: bool = True, time_code: float = Usd.TimeCode.Default()) -> Optional[BBox]:
    """Get bounding box for a prim in world or local space."""
    cache = BBoxCache(prim.GetStage(), time_code)
    if world_space:
        return cache.compute_world_bbox(prim)
    else:
        return cache.compute_local_bbox(prim)


def get_stage_bbox(stage: Usd.Stage, time_code: float = Usd.TimeCode.Default()) -> Optional[BBox]:
    """Get bounding box for an entire stage."""
    cache = BBoxCache(stage, time_code)
    return cache.get_stage_bbox()
