"""
Utility modules for Isaac Sim MCP Server.

This package contains various utility functions and classes for I/O operations,
mathematical computations, timing, and other common tasks.
"""

from .io import (
    ensure_directory,
    safe_file_read,
    safe_file_write,
    get_file_size,
    is_file_readable,
    validate_file_extension,
    create_temp_file,
    cleanup_temp_files,
)

from .math import (
    clamp,
    lerp,
    normalize_vector,
    vector_length,
    vector_distance,
    matrix_multiply,
    transform_point,
    bbox_union,
    bbox_intersection,
    bbox_contains_point,
)

from .timing import (
    Timer,
    measure_time,
    timeout_after,
    rate_limiter,
    debounce,
)

__all__ = [
    # I/O utilities
    "ensure_directory",
    "safe_file_read", 
    "safe_file_write",
    "get_file_size",
    "is_file_readable",
    "validate_file_extension",
    "create_temp_file",
    "cleanup_temp_files",
    
    # Math utilities
    "clamp",
    "lerp",
    "normalize_vector",
    "vector_length", 
    "vector_distance",
    "matrix_multiply",
    "transform_point",
    "bbox_union",
    "bbox_intersection",
    "bbox_contains_point",
    
    # Timing utilities
    "Timer",
    "measure_time",
    "timeout_after",
    "rate_limiter",
    "debounce",
]
