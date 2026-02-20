"""
Adapter layer for Simul MCP Server.

This package provides adapter classes that bridge between the MCP server
and different runtime environments (headless USD operations, Isaac Sim via TCP,
Blender, Unreal Engine).
"""

from typing import Any


def _raise_import_error(adapter_name: str) -> Any:
    """Raise consistent ImportError for unavailable optional adapters."""
    raise ImportError(f"{adapter_name} is not available in this environment")


# Isaac Sim TCP socket client (always available — no omni.* dependency)
from .isaac_socket_client import IsaacSocketClient, ScriptResult


try:
    from .headless_usd import (
        HeadlessUSDAdapter,
        HeadlessUSDSession,
        create_headless_session,
        is_headless_available,
    )
except Exception:
    HeadlessUSDAdapter = None
    HeadlessUSDSession = None

    def create_headless_session(*args: Any, **kwargs: Any) -> Any:
        """Fallback headless session creator when USD runtime is unavailable."""
        return _raise_import_error("HeadlessUSDAdapter")

    def is_headless_available() -> bool:
        """Fallback headless availability check."""
        return False


try:
    from .blender_runtime import (
        BlenderRuntimeAdapter,
        BlenderRuntimeSession,
        create_blender_session,
        is_blender_available,
    )
except Exception:
    BlenderRuntimeAdapter = None
    BlenderRuntimeSession = None

    def create_blender_session(*args: Any, **kwargs: Any) -> Any:
        """Fallback Blender session creator when Blender runtime is unavailable."""
        return _raise_import_error("BlenderRuntimeAdapter")

    def is_blender_available() -> bool:
        """Fallback Blender availability check."""
        return False


try:
    from .unreal_runtime import (
        UnrealRuntimeAdapter,
        UnrealRuntimeSession,
        create_unreal_session,
        is_unreal_available,
    )
except Exception:
    UnrealRuntimeAdapter = None
    UnrealRuntimeSession = None

    def create_unreal_session(*args: Any, **kwargs: Any) -> Any:
        """Fallback Unreal session creator when Unreal runtime is unavailable."""
        return _raise_import_error("UnrealRuntimeAdapter")

    def is_unreal_available() -> bool:
        """Fallback Unreal availability check."""
        return False


__all__ = [
    # Isaac Sim TCP adapter
    "IsaacSocketClient",
    "ScriptResult",
    # Headless USD adapter
    "HeadlessUSDAdapter",
    "HeadlessUSDSession",
    "create_headless_session",
    "is_headless_available",
    # Blender runtime adapter
    "BlenderRuntimeAdapter",
    "BlenderRuntimeSession",
    "create_blender_session",
    "is_blender_available",
    # Unreal Engine runtime adapter
    "UnrealRuntimeAdapter",
    "UnrealRuntimeSession",
    "create_unreal_session",
    "is_unreal_available",
]
