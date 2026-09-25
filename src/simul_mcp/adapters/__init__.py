"""
Adapter layer for Simul MCP Server.

This package provides adapter classes that bridge between the MCP server
and different runtime environments (headless USD operations, Isaac Sim via TCP,
Blender, Unreal Engine).

The exports are resolved lazily (PEP 562 module ``__getattr__``): importing
the package costs nothing, and a name is imported from its submodule on first
access. The optional runtimes pull in heavy dependencies (``pxr`` for headless
USD, ``bpy`` for Blender, ``aiohttp`` for Unreal), and every ``simul`` CLI
invocation imports this package, so eager imports made ``simul --help`` pay for
all of them. An optional runtime whose import fails resolves to the same
fallbacks as before: ``None`` for its classes, a raising ``create_*_session``
and an ``is_*_available`` that returns False.
"""

import importlib
from typing import Any, Callable, Dict, Tuple


def _raise_import_error(adapter_name: str) -> Any:
    """Raise consistent ImportError for unavailable optional adapters."""
    raise ImportError(f"{adapter_name} is not available in this environment")


# Always-importable exports: name -> submodule.
_REQUIRED: Dict[str, str] = {
    # Interface every backend adapter implements
    "BackendAdapter": "base",
    # Isaac Sim TCP socket client (no omni.* dependency)
    "IsaacRuntimeAdapter": "isaac_runtime",
    "IsaacSocketClient": "isaac_socket_client",
    "ScriptResult": "isaac_socket_client",
}


def _fallbacks(label: str, create_name: str, available_name: str) -> Dict[str, Any]:
    """Build the stand-ins used when an optional runtime cannot be imported."""

    def create_session(*args: Any, **kwargs: Any) -> Any:
        return _raise_import_error(label)

    def is_available() -> bool:
        return False

    create_session.__name__ = create_session.__qualname__ = create_name
    is_available.__name__ = is_available.__qualname__ = available_name
    return {create_name: create_session, available_name: is_available}


# Optional runtimes: submodule -> (exported names, fallback factory).
_OPTIONAL: Dict[str, Tuple[Tuple[str, ...], Callable[[], Dict[str, Any]]]] = {
    "headless_usd": (
        ("HeadlessUSDAdapter", "HeadlessUSDSession", "create_headless_session", "is_headless_available"),
        lambda: {
            "HeadlessUSDAdapter": None,
            "HeadlessUSDSession": None,
            **_fallbacks("HeadlessUSDAdapter", "create_headless_session", "is_headless_available"),
        },
    ),
    "blender_runtime": (
        ("BlenderRuntimeAdapter", "BlenderRuntimeSession", "create_blender_session", "is_blender_available"),
        lambda: {
            "BlenderRuntimeAdapter": None,
            "BlenderRuntimeSession": None,
            **_fallbacks("BlenderRuntimeAdapter", "create_blender_session", "is_blender_available"),
        },
    ),
    "unreal_runtime": (
        ("UnrealRuntimeAdapter", "UnrealRuntimeSession", "create_unreal_session", "is_unreal_available"),
        lambda: {
            "UnrealRuntimeAdapter": None,
            "UnrealRuntimeSession": None,
            **_fallbacks("UnrealRuntimeAdapter", "create_unreal_session", "is_unreal_available"),
        },
    ),
}

_OPTIONAL_BY_NAME: Dict[str, str] = {
    name: module for module, (names, _) in _OPTIONAL.items() for name in names
}


def __getattr__(name: str) -> Any:
    """Import an export from its submodule on first access and cache it."""
    module_name = _REQUIRED.get(name)
    if module_name is not None:
        value = getattr(importlib.import_module(f"{__name__}.{module_name}"), name)
        globals()[name] = value
        return value

    module_name = _OPTIONAL_BY_NAME.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    names, fallback = _OPTIONAL[module_name]
    try:
        module = importlib.import_module(f"{__name__}.{module_name}")
        resolved = {export: getattr(module, export) for export in names}
    except Exception:
        resolved = fallback()
    # Resolve the whole group at once so the four names always agree.
    for export, value in resolved.items():
        globals().setdefault(export, value)
    return globals()[name]


def __dir__() -> list:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    # Interface every backend adapter implements
    "BackendAdapter",
    # Isaac Sim TCP adapter
    "IsaacRuntimeAdapter",
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
