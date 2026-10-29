"""
Utility modules for Isaac Sim MCP Server.

Submodules are imported directly (``simul.utils.paths``,
``simul.utils.timing``, ``simul.utils.math``,
``simul.utils.discovery``); this package re-exports only the helpers
other code reaches through it.

The re-exports are resolved on first access (PEP 562): ``utils.math`` imports
numpy, which only the USD library needs, and every ``simul.utils.<module>``
import runs this file first, so an eager import here put numpy in every
server and CLI process.
"""

import importlib
from typing import Any, Dict, List

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


def __getattr__(name: str) -> Any:
    """Import a re-exported helper from its submodule on first access and keep it."""
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
