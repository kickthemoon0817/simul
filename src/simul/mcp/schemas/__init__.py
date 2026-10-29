"""MCP schema package grouped by backend/domain.

Import a backend's models from its own module (``simul.mcp.schemas.usd``).
The package-level names (``from simul.mcp.schemas import PrimInfo``) still
work, but are resolved on first use (PEP 562): any import of a
``simul.mcp.schemas.<module>`` runs this file first, so importing every
module here made a server load each backend's models whatever it served.
"""

import importlib
from typing import Any, List

# The modules whose names the package re-exports, in ``__all__`` order. They
# share no names, so each package-level name has one source.
_EXPORTING_MODULES = ("common", "usd", "instance", "simready", "blender", "unreal")
_SUBMODULES = (*_EXPORTING_MODULES, "blender_ui", "unreal_ui")


def _load_exports() -> List[str]:
    """Import the exporting modules and bind their names here, as the package once did on import.

    Returns:
        The package's ``__all__``.
    """
    namespace = globals()
    names: List[str] = []
    for module_name in _EXPORTING_MODULES:
        module = importlib.import_module(f"{__name__}.{module_name}")
        for name in module.__all__:
            namespace[name] = getattr(module, name)
            names.append(name)
    namespace["__all__"] = names
    return names


def __getattr__(name: str) -> Any:
    """Resolve a schema module, or a package-level name, on first use."""
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    namespace = globals()
    if "__all__" not in namespace and (name == "__all__" or not name.startswith("__")):
        _load_exports()
        if name in namespace:
            return namespace[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    names = globals().get("__all__")
    return sorted(set(globals()) | set(names if names is not None else _load_exports()))
