"""
Mathematical utilities for Isaac Sim MCP Server.

This module provides common mathematical operations for 3D graphics,
linear algebra, and geometric computations used in USD and mesh processing.
"""

from typing import List, Tuple, Union
import numpy as np

from ..logging import get_logger

logger = get_logger(__name__)

# Type aliases for clarity
Vector3 = Union[List[float], Tuple[float, float, float], np.ndarray]
BBox = Tuple[Vector3, Vector3]  # (min_point, max_point)


def bbox_from_points(points: List[Vector3]) -> BBox:
    """
    Calculate bounding box from a list of points.
    
    Args:
        points: List of 3D points
        
    Returns:
        Bounding box as (min_point, max_point)
        
    Raises:
        ValueError: If points list is empty
    """
    if not points:
        raise ValueError("Cannot calculate bounding box from empty points list")
    
    points_array = np.array(points, dtype=float)
    min_point = np.min(points_array, axis=0)
    max_point = np.max(points_array, axis=0)
    
    return (min_point.tolist(), max_point.tolist())


def bbox_union(bbox1: BBox, bbox2: BBox) -> BBox:
    """
    Calculate the union of two bounding boxes.
    
    Args:
        bbox1: First bounding box
        bbox2: Second bounding box
        
    Returns:
        Union bounding box
    """
    min1, max1 = bbox1
    min2, max2 = bbox2
    
    min_point = [
        min(min1[0], min2[0]),
        min(min1[1], min2[1]),
        min(min1[2], min2[2])
    ]
    
    max_point = [
        max(max1[0], max2[0]),
        max(max1[1], max2[1]),
        max(max1[2], max2[2])
    ]
    
    return (min_point, max_point)


def bbox_center(bbox: BBox) -> List[float]:
    """
    Calculate the center point of a bounding box.
    
    Args:
        bbox: Bounding box
        
    Returns:
        Center point coordinates
    """
    min_point, max_point = bbox
    
    return [
        (min_point[0] + max_point[0]) / 2,
        (min_point[1] + max_point[1]) / 2,
        (min_point[2] + max_point[2]) / 2
    ]


def bbox_size(bbox: BBox) -> List[float]:
    """
    Calculate the size (dimensions) of a bounding box.
    
    Args:
        bbox: Bounding box
        
    Returns:
        Size in each dimension [width, height, depth]
    """
    min_point, max_point = bbox
    
    return [
        max_point[0] - min_point[0],
        max_point[1] - min_point[1],
        max_point[2] - min_point[2]
    ]


def bbox_volume(bbox: BBox) -> float:
    """
    Calculate the volume of a bounding box.
    
    Args:
        bbox: Bounding box
        
    Returns:
        Bounding box volume
    """
    size = bbox_size(bbox)
    return size[0] * size[1] * size[2]
