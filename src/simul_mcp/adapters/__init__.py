"""
Adapter layer for Isaac Sim MCP Server.

This package provides adapter classes that bridge between the core USD functionality
and different runtime environments (headless USD operations vs Isaac Sim runtime).
"""

from .headless_usd import (
    HeadlessUSDAdapter,
    HeadlessUSDSession,
    create_headless_session,
    is_headless_available,
)

from .isaac_runtime import (
    IsaacRuntimeAdapter,
    IsaacRuntimeSession,
    ViewportCapture,
    create_isaac_session,
    is_isaac_available,
)

__all__ = [
    # Headless USD adapter
    "HeadlessUSDAdapter",
    "HeadlessUSDSession", 
    "create_headless_session",
    "is_headless_available",

    # Isaac Sim runtime adapter
    "IsaacRuntimeAdapter",
    "IsaacRuntimeSession",
    "ViewportCapture",
    "create_isaac_session",
    "is_isaac_available",
]
