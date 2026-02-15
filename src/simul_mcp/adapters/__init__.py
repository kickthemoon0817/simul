"""Adapter layer for simulation runtime integrations."""

from typing import Any


def _raise_import_error(adapter_name: str) -> Any:
    """Raise consistent ImportError for unavailable optional adapters."""
    raise ImportError(f"{adapter_name} is not available in this environment")


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
    from .isaac_runtime import (
        IsaacRuntimeAdapter,
        IsaacRuntimeSession,
        ViewportCapture,
        create_isaac_session,
        is_isaac_available,
    )
except Exception:
    IsaacRuntimeAdapter = None
    IsaacRuntimeSession = None
    ViewportCapture = None

    def create_isaac_session(*args: Any, **kwargs: Any) -> Any:
        """Fallback Isaac session creator when Isaac runtime is unavailable."""
        return _raise_import_error("IsaacRuntimeAdapter")

    def is_isaac_available() -> bool:
        """Fallback Isaac availability check."""
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
    # Blender runtime adapter
    "BlenderRuntimeAdapter",
    "BlenderRuntimeSession",
    "create_blender_session",
    "is_blender_available",
]
