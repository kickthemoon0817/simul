"""The editor-world parts of a fake ``unreal`` module that ``actor_at`` reads.

``_unreal_scripts.ACTOR_HELPERS``' ``actor_at`` finds an actor among loaded
objects (``unreal.find_object``, which never loads a map, where
``load_object`` loaded the package of an actor in a map that is not open),
checks that it is live (``SystemLibrary.is_valid``, False for a destroyed
actor until garbage collection) and that it belongs to the editor world
(``actor.get_world()`` against ``editor_world()``, which asks
``UnrealEditorSubsystem.get_editor_world`` and, during PIE,
``LayersSubsystem.get_world``). The fake editors of the Unreal tests take
these members from here.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional


class World:
    """A ``unreal.World``, named by its object path (``/Game/Map.Map``)."""

    def __init__(self, path: str) -> None:
        self.package, _, self.name = path.rpartition(".")

    def get_path_name(self) -> str:
        return self.package + "." + self.name

    def get_name(self) -> str:
        return self.name

    def get_outermost(self) -> SimpleNamespace:
        return SimpleNamespace(get_name=lambda: self.package)


def is_valid(obj: Any) -> bool:
    """``SystemLibrary.is_valid``: an object a fake marks ``valid = False`` is destroyed."""
    return obj is not None and getattr(obj, "valid", True)


def editor_subsystems(
    world: World, pie: Callable[[], bool] = lambda: False, pie_world: Optional[World] = None
) -> Dict[str, Any]:
    """``get_editor_subsystem`` answers for the editor world, in and out of PIE.

    During PIE ``get_editor_world`` returns None, as on UE 5.7.4, and
    ``LayersSubsystem.get_world`` (GWorld while Remote Control runs a script)
    still returns the editor world.
    """
    play = pie_world or World("/Game/UEDPIE_0_Map.Map")
    return {
        "UnrealEditorSubsystem": SimpleNamespace(
            get_editor_world=lambda: None if pie() else world,
            get_game_world=lambda: play if pie() else None,
        ),
        "LayersSubsystem": SimpleNamespace(get_world=lambda: world),
    }


#: Module members naming those subsystems, as ``unreal`` has them.
SUBSYSTEM_NAMES = {"UnrealEditorSubsystem": "UnrealEditorSubsystem", "LayersSubsystem": "LayersSubsystem"}
