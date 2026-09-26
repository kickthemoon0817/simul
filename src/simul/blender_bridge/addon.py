"""Entry point copied to simul_blender_bridge/__init__.py in the add-on ZIP."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .bridge import BlenderBridge

bl_info = {
    "name": "Simul Blender Bridge",
    "author": "Simul",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "Preferences > Add-ons",
    "description": "Attach Simul to this running Blender process without replacing its scene",
    "category": "Interface",
}

_bridge: BlenderBridge | None = None


def register() -> None:
    """Start one bridge in the Blender process that enabled this add-on."""
    global _bridge
    if _bridge is None:
        # This entry point lives one level higher in the installed add-on.
        bridge_type = import_module(
            f"{__package__}.blender_bridge.bridge"
        ).BlenderBridge
        _bridge = bridge_type()
        assert _bridge is not None
        _bridge.start()


def unregister() -> None:
    """Stop accepting requests without closing Blender or saving its scene."""
    global _bridge
    if _bridge is not None:
        _bridge.stop()
        _bridge = None
