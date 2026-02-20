"""
Mathematical utilities for Isaac Sim MCP Server.

This module provides common mathematical operations for 3D graphics,
linear algebra, and geometric computations used in USD and mesh processing.
"""

import math
from typing import List, Tuple, Union, Optional
import numpy as np

from ..logging import get_logger

logger = get_logger(__name__)

# Type aliases for clarity
Vector3 = Union[List[float], Tuple[float, float, float], np.ndarray]
Matrix4 = Union[List[List[float]], np.ndarray]
BBox = Tuple[Vector3, Vector3]  # (min_point, max_point)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp a value between minimum and maximum bounds.
    
    Args:
        value: Value to clamp
        min_val: Minimum bound
        max_val: Maximum bound
        
    Returns:
        Clamped value
    """
    return max(min_val, min(value, max_val))


def lerp(a: float, b: float, t: float) -> float:
    """
    Linear interpolation between two values.
    
    Args:
        a: Start value
        b: End value
        t: Interpolation parameter (0.0 to 1.0)
        
    Returns:
        Interpolated value
    """
    return a + t * (b - a)


def normalize_vector(vector: Vector3) -> np.ndarray:
    """
    Normalize a 3D vector to unit length.
    
    Args:
        vector: 3D vector to normalize
        
    Returns:
        Normalized vector as numpy array
        
    Raises:
        ValueError: If vector has zero length
    """
    vec = np.array(vector, dtype=float)
    length = np.linalg.norm(vec)
    
    if length == 0:
        raise ValueError("Cannot normalize zero-length vector")
    
    return vec / length


def vector_length(vector: Vector3) -> float:
    """
    Calculate the length (magnitude) of a 3D vector.
    
    Args:
        vector: 3D vector
        
    Returns:
        Vector length
    """
    vec = np.array(vector, dtype=float)
    return float(np.linalg.norm(vec))


def vector_distance(point1: Vector3, point2: Vector3) -> float:
    """
    Calculate the Euclidean distance between two 3D points.
    
    Args:
        point1: First point
        point2: Second point
        
    Returns:
        Distance between points
    """
    p1 = np.array(point1, dtype=float)
    p2 = np.array(point2, dtype=float)
    return float(np.linalg.norm(p2 - p1))


def dot_product(vector1: Vector3, vector2: Vector3) -> float:
    """
    Calculate dot product of two 3D vectors.
    
    Args:
        vector1: First vector
        vector2: Second vector
        
    Returns:
        Dot product
    """
    v1 = np.array(vector1, dtype=float)
    v2 = np.array(vector2, dtype=float)
    return float(np.dot(v1, v2))


def cross_product(vector1: Vector3, vector2: Vector3) -> np.ndarray:
    """
    Calculate cross product of two 3D vectors.
    
    Args:
        vector1: First vector
        vector2: Second vector
        
    Returns:
        Cross product as numpy array
    """
    v1 = np.array(vector1, dtype=float)
    v2 = np.array(vector2, dtype=float)
    return np.cross(v1, v2)


def matrix_multiply(matrix1: Matrix4, matrix2: Matrix4) -> np.ndarray:
    """
    Multiply two 4x4 matrices.
    
    Args:
        matrix1: First matrix
        matrix2: Second matrix
        
    Returns:
        Result matrix as numpy array
    """
    m1 = np.array(matrix1, dtype=float)
    m2 = np.array(matrix2, dtype=float)
    return np.dot(m1, m2)


def transform_point(point: Vector3, matrix: Matrix4) -> np.ndarray:
    """
    Transform a 3D point by a 4x4 transformation matrix.
    
    Args:
        point: 3D point to transform
        matrix: 4x4 transformation matrix
        
    Returns:
        Transformed point as numpy array
    """
    # Convert point to homogeneous coordinates
    p = np.array([point[0], point[1], point[2], 1.0], dtype=float)
    m = np.array(matrix, dtype=float)
    
    # Transform point
    transformed = np.dot(m, p)
    
    # Convert back to 3D coordinates (divide by w)
    if transformed[3] != 0:
        return transformed[:3] / transformed[3]
    else:
        return transformed[:3]


def create_translation_matrix(translation: Vector3) -> np.ndarray:
    """
    Create a 4x4 translation matrix.
    
    Args:
        translation: Translation vector
        
    Returns:
        4x4 translation matrix
    """
    matrix = np.eye(4, dtype=float)
    matrix[0, 3] = translation[0]
    matrix[1, 3] = translation[1]
    matrix[2, 3] = translation[2]
    return matrix


def create_scale_matrix(scale: Union[float, Vector3]) -> np.ndarray:
    """
    Create a 4x4 scale matrix.
    
    Args:
        scale: Uniform scale factor or per-axis scale vector
        
    Returns:
        4x4 scale matrix
    """
    matrix = np.eye(4, dtype=float)
    
    if isinstance(scale, (int, float)):
        # Uniform scaling
        matrix[0, 0] = scale
        matrix[1, 1] = scale
        matrix[2, 2] = scale
    else:
        # Per-axis scaling
        matrix[0, 0] = scale[0]
        matrix[1, 1] = scale[1]
        matrix[2, 2] = scale[2]
    
    return matrix


def create_rotation_matrix_x(angle_radians: float) -> np.ndarray:
    """
    Create a 4x4 rotation matrix around X-axis.
    
    Args:
        angle_radians: Rotation angle in radians
        
    Returns:
        4x4 rotation matrix
    """
    cos_a = math.cos(angle_radians)
    sin_a = math.sin(angle_radians)
    
    matrix = np.eye(4, dtype=float)
    matrix[1, 1] = cos_a
    matrix[1, 2] = -sin_a
    matrix[2, 1] = sin_a
    matrix[2, 2] = cos_a
    
    return matrix


def create_rotation_matrix_y(angle_radians: float) -> np.ndarray:
    """
    Create a 4x4 rotation matrix around Y-axis.
    
    Args:
        angle_radians: Rotation angle in radians
        
    Returns:
        4x4 rotation matrix
    """
    cos_a = math.cos(angle_radians)
    sin_a = math.sin(angle_radians)
    
    matrix = np.eye(4, dtype=float)
    matrix[0, 0] = cos_a
    matrix[0, 2] = sin_a
    matrix[2, 0] = -sin_a
    matrix[2, 2] = cos_a
    
    return matrix


def create_rotation_matrix_z(angle_radians: float) -> np.ndarray:
    """
    Create a 4x4 rotation matrix around Z-axis.
    
    Args:
        angle_radians: Rotation angle in radians
        
    Returns:
        4x4 rotation matrix
    """
    cos_a = math.cos(angle_radians)
    sin_a = math.sin(angle_radians)
    
    matrix = np.eye(4, dtype=float)
    matrix[0, 0] = cos_a
    matrix[0, 1] = -sin_a
    matrix[1, 0] = sin_a
    matrix[1, 1] = cos_a
    
    return matrix


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


def bbox_intersection(bbox1: BBox, bbox2: BBox) -> Optional[BBox]:
    """
    Calculate the intersection of two bounding boxes.
    
    Args:
        bbox1: First bounding box
        bbox2: Second bounding box
        
    Returns:
        Intersection bounding box, or None if no intersection
    """
    min1, max1 = bbox1
    min2, max2 = bbox2
    
    min_point = [
        max(min1[0], min2[0]),
        max(min1[1], min2[1]),
        max(min1[2], min2[2])
    ]
    
    max_point = [
        min(max1[0], max2[0]),
        min(max1[1], max2[1]),
        min(max1[2], max2[2])
    ]
    
    # Check if intersection is valid
    if (min_point[0] <= max_point[0] and 
        min_point[1] <= max_point[1] and 
        min_point[2] <= max_point[2]):
        return (min_point, max_point)
    else:
        return None


def bbox_contains_point(bbox: BBox, point: Vector3) -> bool:
    """
    Check if a bounding box contains a point.
    
    Args:
        bbox: Bounding box
        point: 3D point to test
        
    Returns:
        True if point is inside bounding box
    """
    min_point, max_point = bbox
    
    return (min_point[0] <= point[0] <= max_point[0] and
            min_point[1] <= point[1] <= max_point[1] and
            min_point[2] <= point[2] <= max_point[2])


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


def degrees_to_radians(degrees: float) -> float:
    """Convert degrees to radians."""
    return degrees * math.pi / 180.0


def radians_to_degrees(radians: float) -> float:
    """Convert radians to degrees."""
    return radians * 180.0 / math.pi


def is_approximately_equal(a: float, b: float, tolerance: float = 1e-6) -> bool:
    """
    Check if two floating point numbers are approximately equal.
    
    Args:
        a: First number
        b: Second number
        tolerance: Comparison tolerance
        
    Returns:
        True if numbers are approximately equal
    """
    return abs(a - b) <= tolerance
