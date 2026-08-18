"""Compatibility shim: the implementation lives in the ``isaac`` package.

Split into per-domain mixins; every name importable from this module before
the split is still importable from it.
"""

from .isaac import (
    BULK_GEOMETRY_ATTRIBUTES,
    FloatList,
    IsaacScriptBase,
    IsaacTools,
    LOG_SCAN_WINDOW_BYTES,
    MAX_CAPTURE_DIMENSION,
    MAX_INLINE_CAPTURE_BYTES,
    MAX_RETAINED_CAPTURES,
    MAX_SCRIPT_BYTES,
    PRIM_DETAIL_ASPECTS,
    _pyval,
)

__all__ = [
    "BULK_GEOMETRY_ATTRIBUTES",
    "FloatList",
    "IsaacScriptBase",
    "IsaacTools",
    "LOG_SCAN_WINDOW_BYTES",
    "MAX_CAPTURE_DIMENSION",
    "MAX_INLINE_CAPTURE_BYTES",
    "MAX_RETAINED_CAPTURES",
    "MAX_SCRIPT_BYTES",
    "PRIM_DETAIL_ASPECTS",
    "_pyval",
]
