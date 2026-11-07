"""A fake Unreal editor for the physics, collision and simulation tools.

The ``unreal`` module it builds answers the editor scripts the way UE 5.7.4
did in a live editor (observed through Remote Control):

* the runtime setters (``set_simulate_physics``, ``set_mobility``,
  ``set_collision_enabled``, ``set_mass_override_in_kg`` and the like) change
  the body but record no undo step and leave the map clean; ``modify()``
  inside a ``ScopedEditorTransaction`` records the object and marks the map
  dirty, and a transaction's ``cancel()`` only drops its record;
* ``set_simulate_physics(True)`` is accepted on any body and detaches it from
  its parent; ``is_simulating_physics()`` then reads True for a Static body,
  one without physics collision and one whose mesh has no simple collision,
  though none of them ever moves (``get_mass()`` reads 0 for a Static body,
  1 for one without physics collision);
* ``set_collision_profile_name`` with an unknown name sets the profile to
  Custom; on a static mesh component it clears Use Default Collision, which
  ``set_collision_enabled`` does not;
* ``StaticMeshEditorSubsystem`` reports -1 for every mesh while Play In
  Editor runs;
* a running simulation has its own world, ``/Game/Maps/UEDPIE_0_Lab``, with
  a copy of each actor at the same path in that package.

Every write is logged in ``Editor.writes`` with whether the object had been
modified in the open transaction, so a test can tell an undoable edit from a
silent one.
"""

from __future__ import annotations

import io
import struct
import sys
import traceback
from contextlib import redirect_stdout
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest

from simul.adapters import unreal_runtime
from simul.config import Settings

MAP = "/Game/Maps/Lab"
PIE_MAP = "/Game/Maps/UEDPIE_0_Lab"
LEVEL = MAP + ".Lab:PersistentLevel."
PIE_LEVEL = PIE_MAP + ".Lab:PersistentLevel."
CUBE = "/Engine/BasicShapes/Cube.Cube"
CUBE_MASS = 177.8279266357422


def f32(value: float) -> float:
    """``value`` as the 32-bit float Unreal stores."""
    return struct.unpack("f", struct.pack("f", value))[0]


class Enum:
    """An Unreal enum value: compared by name, which ``.name`` gives."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Enum) and other.name == self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return f"<Enum {self.name}>"


MOBILITY = SimpleNamespace(
    **{name: Enum(name) for name in ("STATIC", "STATIONARY", "MOVABLE")}
)
COLLISION = SimpleNamespace(
    **{
        name: Enum(name)
        for name in ("NO_COLLISION", "QUERY_ONLY", "PHYSICS_ONLY", "QUERY_AND_PHYSICS")
    }
)
#: Collision presets and the collision mode each sets; Custom keeps the mode.
PRESETS: Dict[str, Optional[str]] = {
    "BlockAll": "QUERY_AND_PHYSICS",
    "BlockAllDynamic": "QUERY_AND_PHYSICS",
    "PhysicsActor": "QUERY_AND_PHYSICS",
    "OverlapAll": "QUERY_ONLY",
    "NoCollision": "NO_COLLISION",
    "Custom": None,
}


class Mesh:
    def __init__(
        self,
        path: str,
        simple: int = 1,
        convex: int = 0,
        complexity: str = "CTF_USE_DEFAULT",
    ) -> None:
        self.path, self.simple, self.convex, self.complexity = (
            path,
            simple,
            convex,
            complexity,
        )

    def get_path_name(self) -> str:
        return self.path


class World:
    def __init__(self, package: str) -> None:
        self.package = package

    def get_outermost(self) -> SimpleNamespace:
        return SimpleNamespace(get_name=lambda: self.package)

    def get_path_name(self) -> str:
        return self.package + ".Lab"


class Body:
    """What ``get_editor_property("body_instance")`` returns: a copy of the body's settings."""

    def __init__(self, values: Dict[str, Any]) -> None:
        self.values = dict(values)

    def get_editor_property(self, name: str) -> Any:
        return self.values[name]


class ActorComponent:
    pass


class SceneComponent(ActorComponent):
    pass


class PrimitiveComponent(SceneComponent):
    def __init__(
        self,
        editor: "Editor",
        name: str,
        mobility: str = "STATIC",
        preset: str = "BlockAll",
        mass: float = CUBE_MASS,
    ) -> None:
        self.editor, self.name = editor, name
        self.owner: Optional["Actor"] = None
        self._mobility = Enum(mobility)
        self.preset = preset
        self.collision = PRESETS[preset] or "QUERY_AND_PHYSICS"
        self.base_mass = mass
        self.parent: Optional["PrimitiveComponent"] = None
        self.socket = "None"
        self.body: Dict[str, Any] = {
            "simulate_physics": False,
            "override_mass": False,
            "mass_in_kg_override": 100.0,
            "enable_gravity": True,
            "linear_damping": f32(0.01),
            "angular_damping": 0.0,
        }
        # A body whose simulate flag never takes (the fake's stand-in for an
        # engine refusal the checks do not foresee).
        self.refuses_simulation = False

    # -- identity -----------------------------------------------------------
    def get_path_name(self) -> str:
        assert self.owner is not None
        return self.owner.get_path_name() + "." + self.name

    def get_name(self) -> str:
        return self.name

    def get_owner(self) -> Optional["Actor"]:
        return self.owner

    def get_class(self) -> SimpleNamespace:
        return SimpleNamespace(get_name=lambda: type(self).__name__)

    def in_simulation(self) -> bool:
        return self.owner is not None and self.owner.simulation

    # -- undo and dirty -------------------------------------------------------
    def modify(self, always_mark_dirty: bool = True) -> bool:
        self.editor.modified(self)
        return True

    def _write(self, what: str) -> None:
        self.editor.write(self, what)

    # -- mobility and attachment ---------------------------------------------
    @property
    def mobility(self) -> Enum:
        return self._mobility

    def set_mobility(self, mobility: Enum) -> None:
        self._write("mobility")
        self._mobility = mobility

    def get_attach_parent(self) -> Optional["PrimitiveComponent"]:
        return self.parent

    def get_attach_socket_name(self) -> str:
        return self.socket

    def attach_to_component(self, parent: Any, socket: str, *rules: Any) -> bool:
        self._write("attach")
        self.parent, self.socket = parent, socket
        return True

    # -- physics ----------------------------------------------------------------
    def has_body(self) -> bool:
        return True

    def set_simulate_physics(self, simulate: bool) -> None:
        self._write("simulate_physics")
        if simulate and self.parent is not None:
            # FBodyInstance::SetInstanceSimulatePhysics detaches a registered root body.
            self.parent._write("detach child")
            self.parent, self.socket = None, "None"
        if not self.refuses_simulation:
            self.body["simulate_physics"] = bool(simulate)

    def is_simulating_physics(self) -> bool:
        return bool(self.body["simulate_physics"]) and self.has_body()

    def get_mass(self) -> float:
        if not self.has_body() or self._mobility.name != "MOVABLE":
            return 0.0
        if self.collision not in ("QUERY_AND_PHYSICS", "PHYSICS_ONLY"):
            return 1.0
        return (
            self.body["mass_in_kg_override"]
            if self.body["override_mass"]
            else self.base_mass
        )

    def set_mass_override_in_kg(self, bone: str, mass: float, override: bool) -> None:
        self._write("mass")
        self.body["override_mass"] = bool(override)
        self.body["mass_in_kg_override"] = f32(mass)

    def set_linear_damping(self, value: float) -> None:
        self._write("linear_damping")
        self.body["linear_damping"] = f32(value)

    def set_angular_damping(self, value: float) -> None:
        self._write("angular_damping")
        self.body["angular_damping"] = f32(value)

    def set_enable_gravity(self, value: bool) -> None:
        self._write("enable_gravity")
        self.body["enable_gravity"] = bool(value)

    # -- collision ----------------------------------------------------------------
    def get_collision_enabled(self) -> Enum:
        return Enum(self.collision)

    def get_collision_profile_name(self) -> str:
        return self.preset

    def set_collision_profile_name(self, name: str) -> None:
        self._write("collision_profile")
        known = {preset.lower(): preset for preset in PRESETS}
        if name.lower() not in known:
            self.preset = "Custom"
            return
        self.preset = name
        mode = PRESETS[known[name.lower()]]
        if mode is not None:
            self.collision = mode

    def set_collision_enabled(self, mode: Enum) -> None:
        self._write("collision_enabled")
        self.collision = mode.name
        self.preset = "Custom"

    # -- properties ---------------------------------------------------------------
    def get_editor_property(self, name: str) -> Any:
        if name == "body_instance":
            return Body(self.body)
        raise AttributeError(f"{type(self).__name__} has no property {name!r}")

    def set_editor_property(self, name: str, value: Any) -> None:
        # set_editor_property runs PreEditChange, which modifies the object.
        self.modify()
        self._write(name)
        raise AttributeError(f"{type(self).__name__} has no property {name!r}")


class StaticMeshComponent(PrimitiveComponent):
    def __init__(
        self, editor: "Editor", name: str, mesh: Optional[Mesh], **kwargs: Any
    ) -> None:
        super().__init__(editor, name, **kwargs)
        self.mesh = mesh
        self.use_default_collision = True

    def has_body(self) -> bool:
        return (
            self.mesh is not None
            and self.mesh.complexity != "CTF_USE_COMPLEX_AS_SIMPLE"
        )

    def set_collision_profile_name(self, name: str) -> None:
        super().set_collision_profile_name(name)
        self.use_default_collision = False

    def get_editor_property(self, name: str) -> Any:
        if name == "static_mesh":
            return self.mesh
        if name == "use_default_collision":
            return self.use_default_collision
        return super().get_editor_property(name)

    def set_editor_property(self, name: str, value: Any) -> None:
        if name == "use_default_collision":
            self.modify()
            self._write(name)
            self.use_default_collision = bool(value)
            return
        super().set_editor_property(name, value)


class CapsuleComponent(PrimitiveComponent):
    pass


class Actor:
    def __init__(
        self,
        editor: "Editor",
        name: str,
        label: str,
        root: PrimitiveComponent,
        simulation: bool = False,
    ):
        self.editor, self.name, self.label, self.root_component = (
            editor,
            name,
            label,
            root,
        )
        self.simulation = simulation
        self.components: List[Any] = [root]
        root.owner = self

    def get_path_name(self) -> str:
        return (PIE_LEVEL if self.simulation else LEVEL) + self.name

    def get_name(self) -> str:
        return self.name

    def get_actor_label(self) -> str:
        return self.label

    def get_world(self) -> World:
        return self.editor.pie_world if self.simulation else self.editor.editor_world

    def get_components_by_class(self, cls: type) -> List[Any]:
        return [
            component for component in self.components if isinstance(component, cls)
        ]

    def get_class(self) -> SimpleNamespace:
        return SimpleNamespace(get_name=lambda: "StaticMeshActor")


class Transaction:
    def __init__(self, editor: "Editor", description: str) -> None:
        self.editor = editor
        self.record: Dict[str, Any] = {
            "description": description,
            "modified": [],
            "cancelled": False,
            "open": False,
        }
        editor.transactions.append(self.record)

    def __enter__(self) -> "Transaction":
        assert self.editor.open is None, "transactions do not nest"
        self.record["open"] = True
        self.editor.open = self.record
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.record["open"] = False
        self.editor.open = None

    def cancel(self) -> None:
        assert self.record["open"], "cancel inside the transaction"
        self.record["cancelled"] = True


class Editor:
    """The level: actors by path, the transaction log and the map's dirty flag."""

    def __init__(self) -> None:
        self.actors: Dict[str, Actor] = {}
        self.transactions: List[Dict[str, Any]] = []
        self.open: Optional[Dict[str, Any]] = None
        self.dirty = False
        self.writes: List[Tuple[str, str, bool]] = []
        self.pie = False
        self.editor_world = World(MAP)
        self.pie_world = World(PIE_MAP)
        self.scripts: List[str] = []

    # -- building the level ---------------------------------------------------------
    def cube(
        self,
        label: str,
        mobility: str = "STATIC",
        preset: str = "BlockAll",
        mesh: Optional[Mesh] = None,
        no_mesh: bool = False,
        simulation: bool = False,
    ) -> Actor:
        name = f"StaticMeshActor_{len(self.actors)}"
        component = StaticMeshComponent(
            self,
            "StaticMeshComponent0",
            None if no_mesh else (mesh or Mesh(CUBE)),
            mobility=mobility,
            preset=preset,
        )
        actor = Actor(self, name, label, component, simulation=simulation)
        self.actors[actor.get_path_name()] = actor
        return actor

    def capsule_actor(self, label: str, preset: str = "NoCollision") -> Actor:
        name = f"PlayerStart_{len(self.actors)}"
        actor = Actor(
            self, name, label, CapsuleComponent(self, "CollisionCapsule", preset=preset)
        )
        self.actors[actor.get_path_name()] = actor
        return actor

    def simulation_copy(self, actor: Actor) -> Actor:
        """The running simulation's copy of ``actor``, at its path in the play world."""
        source = actor.root_component
        component = StaticMeshComponent(
            self,
            source.name,
            getattr(source, "mesh", None),
            mobility=source.mobility.name,
            preset=source.preset,
        )
        component.body = dict(source.body)
        copy = Actor(self, actor.name, actor.label, component, simulation=True)
        self.actors[copy.get_path_name()] = copy
        return copy

    # -- the record ------------------------------------------------------------------
    def modified(self, component: PrimitiveComponent) -> None:
        if component.in_simulation():
            return
        if self.open is not None:
            self.open["modified"].append(component.get_path_name())
        self.dirty = True

    def write(self, component: PrimitiveComponent, what: str) -> None:
        recorded = (
            self.open is not None and component.get_path_name() in self.open["modified"]
        )
        self.writes.append((component.get_path_name(), what, recorded))

    def unrecorded_writes(self) -> List[Tuple[str, str]]:
        """Writes to editor-world objects that no open transaction had recorded."""
        return [
            (path, what)
            for path, what, recorded in self.writes
            if not recorded and not path.startswith(PIE_LEVEL)
        ]

    # -- the unreal module -------------------------------------------------------------
    def module(self) -> SimpleNamespace:
        editor = self

        def load_object(outer: Any, path: str) -> Any:
            return editor.actors.get(path)

        def mesh_count(kind: str) -> Callable[[Mesh], int]:
            return lambda mesh: -1 if editor.pie else getattr(mesh, kind)

        subsystems = {
            "StaticMeshEditorSubsystem": SimpleNamespace(
                get_simple_collision_count=mesh_count("simple"),
                get_convex_collision_count=mesh_count("convex"),
                get_collision_complexity=lambda mesh: Enum(
                    "CTF_USE_DEFAULT" if editor.pie else mesh.complexity
                ),
            ),
            "LevelEditorSubsystem": SimpleNamespace(
                is_in_play_in_editor=lambda: editor.pie
            ),
            "UnrealEditorSubsystem": SimpleNamespace(
                get_editor_world=lambda: None if editor.pie else editor.editor_world,
                get_game_world=lambda: editor.pie_world if editor.pie else None,
            ),
            # GWorld, the editor world wherever Remote Control runs a script,
            # which editor_world() falls back on during PIE.
            "LayersSubsystem": SimpleNamespace(get_world=lambda: editor.editor_world),
        }

        def new_object(cls: type) -> Any:
            assert cls is StaticMeshComponent
            probe = StaticMeshComponent(editor, "Probe", Mesh(CUBE))
            probe.owner = Actor(editor, "Transient", "Transient", probe)
            probe._write = lambda what: None  # type: ignore[method-assign]
            return probe

        return SimpleNamespace(
            Actor=Actor,
            ActorComponent=ActorComponent,
            SceneComponent=SceneComponent,
            PrimitiveComponent=PrimitiveComponent,
            StaticMeshComponent=StaticMeshComponent,
            ComponentMobility=MOBILITY,
            CollisionEnabled=COLLISION,
            AttachmentRule=SimpleNamespace(KEEP_WORLD=Enum("KEEP_WORLD")),
            ScopedEditorTransaction=lambda description: Transaction(
                editor, description
            ),
            StaticMeshEditorSubsystem="StaticMeshEditorSubsystem",
            LevelEditorSubsystem="LevelEditorSubsystem",
            UnrealEditorSubsystem="UnrealEditorSubsystem",
            LayersSubsystem="LayersSubsystem",
            SystemLibrary=SimpleNamespace(is_valid=lambda obj: obj is not None),
            get_editor_subsystem=subsystems.__getitem__,
            load_object=load_object,
            find_object=load_object,
            new_object=new_object,
        )

    def run_python(self, code: str) -> Dict[str, Any]:
        """Run an editor script as ExecutePythonCommandEx does."""
        self.scripts.append(code)
        output = io.StringIO()
        try:
            with redirect_stdout(output):
                exec(code, {})
        except Exception:
            return {
                "ReturnValue": False,
                "CommandResult": traceback.format_exc(),
                "LogOutput": [],
            }
        return {
            "ReturnValue": True,
            "LogOutput": [
                {"Type": "Info", "Output": line}
                for line in output.getvalue().splitlines()
            ],
        }


def physics_session(
    monkeypatch: pytest.MonkeyPatch, editor: Editor
) -> unreal_runtime.UnrealRuntimeSession:
    """A session whose editor Python runs here against ``editor``."""
    monkeypatch.setitem(sys.modules, "unreal", editor.module())
    monkeypatch.setattr(unreal_runtime, "UNREAL_AVAILABLE", True)
    monkeypatch.setattr(unreal_runtime, "AIOHTTP_AVAILABLE", True)
    session = unreal_runtime.UnrealRuntimeSession(Settings())

    async def execute(code: str, mode: str = "ExecuteFile") -> Dict[str, Any]:
        return editor.run_python(code)

    monkeypatch.setattr(session, "_execute_python", execute)
    return session
