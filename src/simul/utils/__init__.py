"""
Utility modules for Isaac Sim MCP Server.

Submodules are imported directly (``simul.utils.paths``,
``simul.utils.timing``, ``simul.utils.math``,
``simul.utils.discovery``); this package re-exports only the helpers
other code reaches through it.

The re-exports are resolved on first access (PEP 562): ``utils.math`` imports
numpy, which only the USD library needs, and every ``simul.utils.<module>``
import runs this file first, so an eager import here put numpy in every
server and CLI process. ``math`` and ``timing`` resolve as attributes too,
as when this file imported them. Type checkers read the ``TYPE_CHECKING``
imports instead of ``__getattr__``, so they see each helper's type and reject
a name the package does not export.
"""

import importlib
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from .math import (
        BBox,
        bbox_center,
        bbox_from_points,
        bbox_size,
        bbox_union,
        bbox_volume,
    )
    from .timing import RateLimiter, Timer, monitor_performance

# Re-exported name -> the submodule defining it.
_EXPORTS: Dict[str, str] = {
    "BBox": "math",
    "bbox_center": "math",
    "bbox_from_points": "math",
    "bbox_size": "math",
    "bbox_union": "math",
    "bbox_volume": "math",
    "RateLimiter": "timing",
    "Timer": "timing",
    "monitor_performance": "timing",
}


if not TYPE_CHECKING:

    def __getattr__(name: str) -> Any:
        """Import a re-exported helper, or the submodule defining one, on first access."""
        if name in _EXPORTS.values():
            return importlib.import_module(f"{__name__}.{name}")
        module_name = _EXPORTS.get(name)
        if module_name is None:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        value = getattr(importlib.import_module(f"{__name__}.{module_name}"), name)
        globals()[name] = value
        return value


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    # Math utilities
    "BBox",
    "bbox_center",
    "bbox_from_points",
    "bbox_size",
    "bbox_union",
    "bbox_volume",
    # Timing utilities
    "RateLimiter",
    "Timer",
    "monitor_performance",
]
