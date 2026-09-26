"""
Tool registration modules for Simul MCP Server.

Each module registers tool closures on the FastMCP server instance
for a specific backend (USD, Isaac Sim, Blender, Unreal Engine).

The ``register_*`` functions are resolved on first access (PEP 562). The
modules import fastmcp, and ``_helpers`` — which the Isaac tool classes use at
import time — lives in this package, so eager imports here made importing
``IsaacTools`` load fastmcp and every registration module, and formed an
import cycle (``tools.isaac`` -> ``registration`` -> ``_reg_isaac`` ->
``tools.isaac_tools``) that only resolved when the server happened to be
imported first.
"""

import importlib
from typing import Any

_EXPORTS = {
    "register_instance_tools": "_reg_instance",
    "register_usd_tools": "_reg_usd",
    "register_isaac_tools": "_reg_isaac",
    "register_blender_tools": "_reg_blender",
    "register_unreal_tools": "_reg_unreal",
    "register_stats_tools": "_reg_stats",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
