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
* a preset sets the collision responses too (``PRESET_RESPONSES``: OverlapAll
  and Trigger overlap every object type, NoCollision's block them), and
  ``set_collision_enabled`` keeps them, turning the preset into Custom;
* ``StaticMeshEditorSubsystem`` reports -1 for every mesh while Play In
  Editor runs;
* a running simulation has its own world, ``/Game/Maps/UEDPIE_0_Lab``, with
  a copy of each actor at the same path in that package, keeping its label
  and ActorGuid. Only that world steps (``Editor.frame``): an editor body
  takes forces and impulses but never moves. Each frame the world ticks
  first, advancing game time by the frame and physics by the frame capped at
  PhysicsSettings' MaxPhysicsDeltaTime (1/30 s), as FChaosScene::SetUpForFrame
  does, then the ticker callbacks run (FEngineLoop::Tick ticks the core
  ticker after the engine); a callback that returns False is removed. A force
  acts for the one step after it is added; an impulse changes the velocity by
  impulse / mass at once.
* a destroyed actor stays findable (``find_object``) until garbage
  collection, and ``SystemLibrary.is_valid`` is False for it and its
  components.

Every write is logged in ``Editor.writes`` with whether the object had been
modified in the open transaction, so a test can tell an undoable edit from a
silent one.
"""

from __future__ import annotations

import copy
import io
import itertools
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
GRAVITY = 980.0


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
    "Trigger": "QUERY_ONLY",
    "Spectator": "QUERY_ONLY",
    "NoCollision": "NO_COLLISION",
    "Custom": None,
}
#: Unreal's object channels, which the physics checks read the responses to.
OBJECT_CHANNELS = (
    "ECC_WORLD_STATIC",
    "ECC_WORLD_DYNAMIC",
    "ECC_PAWN",
    "ECC_PHYSICS_BODY",
    "ECC_VEHICLE",
    "ECC_DESTRUCTIBLE",
)
CHANNELS = SimpleNamespace(
    **{name: Enum(name) for name in OBJECT_CHANNELS + ("ECC_VISIBILITY", "ECC_CAMERA")}
)


def _responses(default: str, **overrides: str) -> Dict[str, str]:
    responses = {name: default for name in OBJECT_CHANNELS}
    responses.update(overrides)
    return responses


#: Each preset's responses to the object channels, as UE 5.7.4 read them off a
#: transient component; Custom keeps the current ones.
PRESET_RESPONSES: Dict[str, Optional[Dict[str, str]]] = {
    "BlockAll": _responses("ECR_BLOCK"),
    "BlockAllDynamic": _responses("ECR_BLOCK"),
    "PhysicsActor": _responses("ECR_BLOCK"),
    "OverlapAll": _responses("ECR_OVERLAP"),
    "Trigger": _responses("ECR_OVERLAP"),
    "Spectator": _responses("ECR_IGNORE", ECC_WORLD_STATIC="ECR_BLOCK"),
    "NoCollision": _responses("ECR_BLOCK"),
    "Custom": None,
}


class Vector:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.x, self.y, self.z = float(x), float(y), float(z)

    def as_list(self) -> List[float]:
        return [self.x, self.y, self.z]


class Rotator:
    def __init__(self, pitch: float = 0.0, yaw: float = 0.0, roll: float = 0.0) -> None:
        self.pitch, self.yaw, self.roll = pitch, yaw, roll


class Guid:
    def __init__(self, value: int) -> None:
        self.value = value


class Class:
    """What ``get_class()`` returns: compared by name."""

    def __init__(self, name: str) -> None:
        self.name = name

    def get_name(self) -> str:
        return self.name

    def get_path_name(self) -> str:
        return "/Script/Engine." + self.name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Class) and other.name == self.name

    def __hash__(self) -> int:
        return hash(self.name)


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
    """What ``get_editor_property("body_instance")`` returns. As on UE 5.7.4 it
    refers to the component's own struct, so a later write shows through it;
    ``copy()`` takes the struct as it is now (collision preset and mode
    included)."""

    def __init__(self, component: Optional["PrimitiveComponent"] = None,
                 values: Optional[Dict[str, Any]] = None) -> None:
        self._component = component
        self._values = None if values is None else dict(values)
        # Fields set on this struct, which reach the component when it is
        # written back with set_editor_property("body_instance", ...).
        self._set: Dict[str, Any] = {}

    @property
    def values(self) -> Dict[str, Any]:
        if self._values is not None:
            return dict(self._values, **self._set)
        component = self._component
        assert component is not None
        return dict(component.body, collision_profile_name=component.preset,
                    collision_enabled=Enum(component.collision),
                    collision_responses=dict(component.responses), **self._set)

    def get_editor_property(self, name: str) -> Any:
        return self.values[name]

    def set_editor_property(self, name: str, value: Any) -> None:
        assert name in self.values, name
        self._set[name] = value
        if self._component is not None:
            # A field written on the component's own struct notifies the component as a property write
            # does, which re-runs a Blueprint actor's construction script (UE 5.7.4).
            self._component.set_editor_property("body_instance", Body(values=self.values))

    def copy(self) -> "Body":
        return Body(values=self.values)


class ActorComponent:
    pass


class SceneComponent(ActorComponent):
    """A component with a transform and no body, such as a multi-part actor's root."""

    def __init__(self, name: str = "DefaultSceneRoot") -> None:
        self.name = name
        self.owner: Optional["Actor"] = None

    def get_name(self) -> str:
        return self.name

    def get_path_name(self) -> str:
        assert self.owner is not None
        return self.owner.get_path_name() + "." + self.name

    def get_owner(self) -> Optional["Actor"]:
        return self.owner

    def get_attach_parent(self) -> Optional["SceneComponent"]:
        return None

    def get_class(self) -> "Class":
        return Class(type(self).__name__)

    def modify(self, always_mark_dirty: bool = True) -> bool:
        return True

    def _write(self, what: str) -> None:
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
        self.responses: Dict[str, str] = dict(PRESET_RESPONSES[preset] or _responses("ECR_BLOCK"))
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
        # A body whose simulate flag, once set, the Details panel write cannot clear (putting it back fails).
        self.sticky_simulation = False
        # Motion: velocity in cm/s, the force pending for the next step, and
        # every force and impulse call, as (kind, vector, location).
        self.velocity = Vector()
        self.angular = Vector()
        self.pending_force = Vector()
        self.pushes: List[Tuple[str, List[float], Optional[List[float]]]] = []

    # -- identity -----------------------------------------------------------
    def get_path_name(self) -> str:
        assert self.owner is not None
        return self.owner.get_path_name() + "." + self.name

    def get_name(self) -> str:
        return self.name

    def get_owner(self) -> Optional["Actor"]:
        return self.owner

    def get_class(self) -> Class:
        return Class(type(self).__name__)

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
            self.editor.detached.append(self.get_path_name())
        if not self.refuses_simulation:
            self.body["simulate_physics"] = bool(simulate)

    def weld_root(self) -> Optional["PrimitiveComponent"]:
        """In the play world, the simulating part above this one whose body Unreal welded it into."""
        if not self.in_simulation() or self.body["simulate_physics"]:
            return None
        parent = self.parent
        while isinstance(parent, PrimitiveComponent):
            if parent.body["simulate_physics"] and parent.has_body():
                return parent
            parent = parent.parent
        return None

    def welded_parts(self) -> List["PrimitiveComponent"]:
        owner = self.owner
        if owner is None:
            return []
        return [c for c in owner.components
                if isinstance(c, PrimitiveComponent) and c is not self and c.weld_root() is self]

    def is_simulating_physics(self) -> bool:
        # A welded part's body is its weld root's (GetBodyInstance follows WeldParent).
        root = self.weld_root()
        if root is not None:
            return root.is_simulating_physics()
        return bool(self.body["simulate_physics"]) and self.has_body()

    def moves(self) -> bool:
        """Whether the body really steps: simulating, Movable, with physics collision."""
        return (
            self.is_simulating_physics()
            and self._mobility.name == "MOVABLE"
            and self.collision in ("QUERY_AND_PHYSICS", "PHYSICS_ONLY")
        )

    def own_mass(self) -> float:
        if not self.has_body() or self._mobility.name != "MOVABLE":
            return 0.0
        if self.collision not in ("QUERY_AND_PHYSICS", "PHYSICS_ONLY"):
            return 1.0
        return (
            self.body["mass_in_kg_override"]
            if self.body["override_mass"]
            else self.base_mass
        )

    def get_mass(self) -> float:
        # A welded body's mass is its parts' masses added up (UE 5.7.4).
        root = self.weld_root() or self
        return root.own_mass() + sum(part.own_mass() for part in root.welded_parts())

    def _target(self) -> "PrimitiveComponent":
        # The setters and the force calls reach a welded part's weld root (GetBodyInstance).
        return self.weld_root() or self

    def set_mass_override_in_kg(self, bone: str, mass: float, override: bool) -> None:
        target = self._target()
        target._write("mass")
        target.body["override_mass"] = bool(override)
        target.body["mass_in_kg_override"] = f32(mass)

    def set_linear_damping(self, value: float) -> None:
        target = self._target()
        target._write("linear_damping")
        target.body["linear_damping"] = f32(value)

    def set_angular_damping(self, value: float) -> None:
        target = self._target()
        target._write("angular_damping")
        target.body["angular_damping"] = f32(value)

    def set_enable_gravity(self, value: bool) -> None:
        target = self._target()
        target._write("enable_gravity")
        target.body["enable_gravity"] = bool(value)

    def is_gravity_enabled(self) -> bool:
        return bool(self.body["enable_gravity"])

    # -- forces and motion ---------------------------------------------------------
    def add_force(
        self, force: Vector, bone: str = "None", accel_change: bool = False
    ) -> None:
        self.pushes.append(("force", force.as_list(), None))
        self.pending_force = Vector(
            self.pending_force.x + force.x,
            self.pending_force.y + force.y,
            self.pending_force.z + force.z,
        )

    def add_force_at_location(
        self, force: Vector, location: Vector, bone: str = "None"
    ) -> None:
        self.pushes.append(("force", force.as_list(), location.as_list()))
        self.pending_force = Vector(
            self.pending_force.x + force.x,
            self.pending_force.y + force.y,
            self.pending_force.z + force.z,
        )

    def add_impulse(
        self, impulse: Vector, bone: str = "None", vel_change: bool = False
    ) -> None:
        self.pushes.append(("impulse", impulse.as_list(), None))
        self._kick(impulse)

    def add_impulse_at_location(
        self, impulse: Vector, location: Vector, bone: str = "None"
    ) -> None:
        self.pushes.append(("impulse", impulse.as_list(), location.as_list()))
        self._kick(impulse)

    def _kick(self, impulse: Vector) -> None:
        root = self.weld_root()
        if root is not None:
            root._kick(impulse)
            return
        if self.moves():
            mass = self.get_mass()
            self.velocity = Vector(
                self.velocity.x + impulse.x / mass,
                self.velocity.y + impulse.y / mass,
                self.velocity.z + impulse.z / mass,
            )

    def step(self, dt: float) -> None:
        """One physics step of the play world: the pending force acts, then is cleared."""
        if self.moves():
            mass = self.get_mass()
            gravity = GRAVITY if self.body["enable_gravity"] else 0.0
            self.velocity = Vector(
                self.velocity.x + self.pending_force.x / mass * dt,
                self.velocity.y + self.pending_force.y / mass * dt,
                self.velocity.z + (self.pending_force.z / mass - gravity) * dt,
            )
            assert self.owner is not None
            self.owner.location = Vector(
                self.owner.location.x + self.velocity.x * dt,
                self.owner.location.y + self.velocity.y * dt,
                self.owner.location.z + self.velocity.z * dt,
            )
        self.pending_force = Vector()

    def get_physics_linear_velocity(self, bone: str = "None") -> Vector:
        return self._target().velocity

    def get_physics_angular_velocity_in_degrees(self, bone: str = "None") -> Vector:
        return self.angular

    def is_any_rigid_body_awake(self) -> bool:
        return self.moves()

    def get_center_of_mass(self, bone: str = "None") -> Vector:
        assert self.owner is not None
        return self.owner.location

    def get_world_location(self) -> Vector:
        assert self.owner is not None
        return self.owner.location

    def get_world_rotation(self) -> Rotator:
        return Rotator()

    # -- collision ----------------------------------------------------------------
    def get_collision_enabled(self) -> Enum:
        return Enum(self.collision)

    def get_collision_profile_name(self) -> str:
        return self.preset

    def get_collision_response_to_channel(self, channel: Enum) -> Enum:
        return Enum(self.responses.get(channel.name, "ECR_BLOCK"))

    def set_collision_profile_name(self, name: str) -> None:
        self._write("collision_profile")
        known = {preset.lower(): preset for preset in PRESETS}
        if name.lower() not in known:
            self.preset = "Custom"
            return
        self.preset = known[name.lower()]
        mode = PRESETS[self.preset]
        if mode is not None:
            self.collision = mode
        responses = PRESET_RESPONSES[self.preset]
        if responses is not None:
            self.responses = dict(responses)

    def set_collision_enabled(self, mode: Enum) -> None:
        # Keeps the preset's responses, as Unreal does.
        self._write("collision_enabled")
        self.collision = mode.name
        self.preset = "Custom"

    # -- properties ---------------------------------------------------------------
    def get_editor_property(self, name: str) -> Any:
        if name == "body_instance":
            return Body(self)
        raise AttributeError(f"{type(self).__name__} has no property {name!r}")

    def set_editor_property(self, name: str, value: Any) -> None:
        # set_editor_property runs PreEditChange, which modifies the object.
        self.modify()
        self._write(name)
        if name == "body_instance":
            # The Details panel's write: it keeps the attachment, unlike set_simulate_physics.
            values = dict(value.values)
            self.preset = values.pop("collision_profile_name")
            self.collision = values.pop("collision_enabled").name
            self.responses = values.pop("collision_responses")
            if self.refuses_simulation or (self.sticky_simulation and self.body["simulate_physics"]):
                values["simulate_physics"] = self.body["simulate_physics"]
            self.body = values
            self._edited()
            return
        raise AttributeError(f"{type(self).__name__} has no property {name!r}")

    def _edited(self) -> None:
        """PostEditChangeProperty: a Blueprint actor of the editor world re-runs its construction script."""
        owner = self.owner
        if owner is not None and owner.blueprint and not owner.simulation and self in owner.components:
            owner.rerun_construction()


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
            self._edited()
            return
        super().set_editor_property(name, value)


class CapsuleComponent(PrimitiveComponent):
    pass


class BillboardComponent(PrimitiveComponent):
    """The editor-only sprite an Actor spawned from the class carries (UE 5.7.4)."""

    def get_editor_property(self, name: str) -> Any:
        if name == "is_editor_only":
            return True
        return super().get_editor_property(name)


class DynamicMeshComponent(PrimitiveComponent):
    """A generated mesh's component: ``simple`` collision shapes, as GeometryScript reads them."""

    def __init__(self, editor: "Editor", name: str, simple: int = 0, **kwargs: Any) -> None:
        super().__init__(editor, name, **kwargs)
        self.simple = simple
        self.collision_type = "CTF_USE_DEFAULT"

    def get_editor_property(self, name: str) -> Any:
        if name == "collision_type":
            return Enum(self.collision_type)
        return super().get_editor_property(name)


_guids = itertools.count(1)
_trash = itertools.count(6)


class Actor:
    def __init__(
        self,
        editor: "Editor",
        name: str,
        label: str,
        root: Optional[PrimitiveComponent],
        simulation: bool = False,
        class_name: str = "StaticMeshActor",
        guid: Optional[int] = None,
    ):
        self.editor, self.name, self.label, self.root_component = (
            editor,
            name,
            label,
            root,
        )
        self.simulation = simulation
        self.components: List[Any] = [root] if root is not None else []
        self.class_name = class_name
        self.guid = Guid(next(_guids) if guid is None else guid)
        self.tags: List[str] = []
        self.folder = ""
        self.location = Vector(0.0, 0.0, 300.0)
        # The path's level, for an actor World Partition streamed into a cell.
        self.level: Optional[str] = None
        # Destroyed: still findable, but SystemLibrary.is_valid is False.
        self.destroyed = False
        # A Blueprint actor whose components its construction script makes (rerun_construction).
        self.blueprint = False
        if root is not None:
            root.owner = self

    def rerun_construction(self) -> None:
        """Re-run the construction script, as a component property write does on a Blueprint actor (UE 5.7.4).

        Every component is replaced by a new one of the same name that carries its state (Unreal's
        component instance data), attached as the old one was; the old one is renamed
        ``TRASH_<class>_<n>`` and left out of the actor, its owner unchanged.
        """
        made: Dict[int, Any] = {}
        for old in self.components:
            new = copy.copy(old)
            for field in ("body", "responses"):
                if hasattr(old, field):
                    setattr(new, field, dict(getattr(old, field)))
            made[id(old)] = new
            old.name = f"TRASH_{type(old).__name__}_{next(_trash)}"
        for new in made.values():
            parent = getattr(new, "parent", None)
            if parent is not None and id(parent) in made:
                new.parent = made[id(parent)]
        self.components = [made[id(old)] for old in self.components]
        if self.root_component is not None:
            self.root_component = made.get(id(self.root_component), self.root_component)
        self.editor.reruns += 1

    def get_path_name(self) -> str:
        if self.level is not None:
            return self.level + self.name
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

    def get_class(self) -> Class:
        return Class(self.class_name)

    def get_actor_location(self) -> Vector:
        return self.location

    def get_actor_rotation(self) -> Rotator:
        return Rotator()

    def actor_has_tag(self, tag: str) -> bool:
        return tag in self.tags

    def get_folder_path(self) -> str:
        return self.folder

    def is_editable(self) -> bool:
        return True

    def get_editor_property(self, name: str) -> Any:
        if name == "actor_guid":
            return self.guid
        raise AttributeError(name)


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
    """The level: actors by path, the transaction log, the map's dirty flag and the play world."""

    def __init__(self) -> None:
        self.actors: Dict[str, Actor] = {}
        self.transactions: List[Dict[str, Any]] = []
        self.open: Optional[Dict[str, Any]] = None
        self.dirty = False
        self.writes: List[Tuple[str, str, bool]] = []
        self.pie = False
        self.paused = False
        self.sim_time = 0.0
        self.frame_seconds = 1.0 / 120.0
        # PhysicsSettings: the longest physics step a frame takes (1/30 s, BaseEngine.ini).
        self.max_physics_delta_time = 1.0 / 30.0
        self.editor_world = World(MAP)
        self.pie_world = World(PIE_MAP)
        self.scripts: List[str] = []
        # Components set_simulate_physics detached from their parent, by path.
        self.detached: List[str] = []
        self.tickers: Dict[int, Callable[[float], Any]] = {}
        self.ticker_ids = itertools.count(1)
        self.started_modes: List[str] = []
        # Construction-script reruns of Blueprint actors (Actor.rerun_construction).
        self.reruns = 0
        # The level viewport camera: location, and rotation as (pitch, yaw, roll).
        self.camera: Tuple[List[float], List[float]] = ([-458.0, 1300.0, 440.0], [-20.0, 30.0, 0.0])

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

    def multi_part_actor(self, label: str, parts: int = 2, simulation: bool = False) -> Actor:
        """An Actor with a SceneComponent root, its editor-only sprite and ``parts`` cube StaticMeshComponents."""
        name = f"Actor_{len(self.actors)}"
        root = SceneComponent()
        actor = Actor(self, name, label, None, simulation=simulation, class_name="Actor")
        actor.root_component = root  # type: ignore[assignment]
        root.owner = actor
        sprite = BillboardComponent(self, "BillboardComponent_0", mobility="MOVABLE", preset="NoCollision")
        sprite.owner = actor
        actor.components.append(sprite)
        for index in range(parts):
            part = StaticMeshComponent(self, f"Part{index}", Mesh(CUBE), mobility="MOVABLE")
            part.owner = actor
            part.parent = root  # type: ignore[assignment]
            actor.components.append(part)
        actor.components.insert(0, root)
        self.actors[actor.get_path_name()] = actor
        return actor

    def robot(self, label: str = "Robot") -> Actor:
        """An Actor built as add_unreal_component builds one: a SceneComponent root, Chassis under it and
        Arm and Wheel under Chassis, each a Movable cube StaticMeshComponent."""
        actor = self.multi_part_actor(label, parts=0)
        root = actor.root_component
        parts: Dict[str, Any] = {}
        for name, parent in (("Chassis", root), ("Arm", "Chassis"), ("Wheel", "Chassis")):
            part = StaticMeshComponent(self, name, Mesh(CUBE), mobility="MOVABLE", preset="BlockAllDynamic")
            part.owner = actor
            part.parent = parts[parent] if isinstance(parent, str) else parent
            parts[name] = part
            actor.components.append(part)
        return actor

    def blueprint_crate(self, label: str = "Crate") -> Actor:
        """A Blueprint actor (BP_Crate_C) built in the Blueprint editor: DefaultSceneRoot and a Movable cube
        StaticMeshComponent "Mesh" under it, both made by its construction script."""
        actor = self.multi_part_actor(label, parts=0)
        actor.class_name, actor.blueprint = "BP_Crate_C", True
        mesh = StaticMeshComponent(self, "Mesh", Mesh(CUBE), mobility="MOVABLE", preset="BlockAllDynamic")
        mesh.owner, mesh.parent = actor, actor.root_component
        actor.components.append(mesh)
        return actor

    def blueprint_robot(self, label: str = "Robot") -> Actor:
        """``robot`` built as a Blueprint: Chassis, Arm and Wheel made by its construction script."""
        actor = self.robot(label)
        actor.class_name, actor.blueprint = "BP_Robot_C", True
        return actor

    def dynamic_mesh_actor(self, label: str, simple: int = 0) -> Actor:
        """A generated DynamicMeshActor, with ``simple`` simple collision shapes."""
        name = f"DynamicMeshActor_{len(self.actors)}"
        component = DynamicMeshComponent(self, "DynamicMeshComponent0", simple=simple, mobility="MOVABLE")
        actor = Actor(self, name, label, component, class_name="DynamicMeshActor")
        self.actors[actor.get_path_name()] = actor
        return actor

    def capsule_actor(self, label: str, preset: str = "NoCollision") -> Actor:
        name = f"PlayerStart_{len(self.actors)}"
        actor = Actor(
            self,
            name,
            label,
            CapsuleComponent(self, "CollisionCapsule", preset=preset),
            class_name="PlayerStart",
        )
        self.actors[actor.get_path_name()] = actor
        return actor

    def simulation_copy(self, actor: Actor, level: Optional[str] = None) -> Actor:
        """The running simulation's copy of ``actor``, at its path in the play world (or ``level``)."""
        if not isinstance(actor.root_component, PrimitiveComponent):
            return self._multi_part_copy(actor)
        source = actor.root_component
        assert isinstance(source, StaticMeshComponent)
        component = StaticMeshComponent(
            self,
            source.name,
            source.mesh,
            mobility=source.mobility.name,
            preset=source.preset,
        )
        component.collision = source.collision
        component.responses = dict(source.responses)
        component.body = dict(source.body)
        copy = Actor(
            self,
            actor.name,
            actor.label,
            component,
            simulation=True,
            class_name=actor.class_name,
            guid=actor.guid.value,
        )
        copy.tags = list(actor.tags)
        copy.location = Vector(*actor.location.as_list())
        copy.level = level
        self.actors[copy.get_path_name()] = copy
        return copy

    def _multi_part_copy(self, actor: Actor) -> Actor:
        parts = [c for c in actor.components if isinstance(c, StaticMeshComponent)]
        copy = self.multi_part_actor(actor.label, parts=0, simulation=True)
        del self.actors[copy.get_path_name()]
        copy.name, copy.guid = actor.name, Guid(actor.guid.value)
        made: Dict[str, Any] = {copy.root_component.get_name(): copy.root_component}
        for source in parts:
            part = StaticMeshComponent(self, source.name, source.mesh, mobility=source.mobility.name,
                                       preset=source.preset)
            part.collision, part.responses, part.body = source.collision, dict(source.responses), dict(source.body)
            part.base_mass = source.base_mass
            part.owner = copy
            made[part.name] = part
            copy.components.append(part)
        for source in parts:
            part = made[source.name]
            parent = source.parent.get_name() if source.parent is not None else None
            # As the simulation starts, Unreal detaches a simulating body from its parent (in the copy only).
            part.parent = None if parent is None or part.body["simulate_physics"] else made.get(parent)
        copy.location = Vector(*actor.location.as_list())
        self.actors[copy.get_path_name()] = copy
        return copy

    def spawned_in_simulation(
        self, label: str, class_name: str = "DefaultPawn"
    ) -> Actor:
        """An actor the simulation spawned: no editor counterpart, no valid ActorGuid."""
        actor = Actor(
            self,
            label + "_0",
            label,
            None,
            simulation=True,
            class_name=class_name,
            guid=0,
        )
        self.actors[actor.get_path_name()] = actor
        return actor

    def start(self) -> None:
        """Start the simulation: a play copy of every editor actor."""
        self.pie = True
        self.sim_time = 0.0
        for actor in [a for a in self.actors.values() if not a.simulation]:
            self.simulation_copy(actor)

    def begin(self, mode: str) -> None:
        """What LevelEditorSubsystem's play calls do: start, and note which was called."""
        self.started_modes.append(mode)
        self.start()

    def set_camera(self, location: Vector, rotation: Rotator) -> None:
        self.camera = (location.as_list(), [rotation.pitch, rotation.yaw, rotation.roll])

    def stop(self) -> None:
        if self.started_modes and self.started_modes[-1] == "play":
            # Unreal leaves the player camera's last view in the level viewport (UE 5.7.4).
            self.camera = ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        self.pie = False
        self.paused = False
        for path in [p for p, a in self.actors.items() if a.simulation]:
            del self.actors[path]

    def physics_step(self) -> float:
        """The physics time one frame simulates: the frame, capped at MaxPhysicsDeltaTime."""
        return min(self.frame_seconds, self.max_physics_delta_time)

    def frame(self, count: int = 1, frame_seconds: Optional[float] = None) -> None:
        """Advance the editor ``count`` frames: the play world ticks, then the tickers run."""
        if frame_seconds is not None:
            self.frame_seconds = frame_seconds
        for _ in range(count):
            if self.pie and not self.paused:
                self.sim_time += self.frame_seconds
                step = self.physics_step()
                for actor in list(self.actors.values()):
                    if (
                        actor.simulation
                        and isinstance(actor.root_component, PrimitiveComponent)
                        and not actor.destroyed
                    ):
                        actor.root_component.step(step)
            for handle, callback in list(self.tickers.items()):
                if not callback(self.frame_seconds):
                    self.tickers.pop(handle, None)

    def world_actors(self, world: World) -> List[Actor]:
        simulation = world is self.pie_world
        return [
            actor for actor in self.actors.values() if actor.simulation == simulation
        ]

    # -- the record ------------------------------------------------------------------
    def modified(self, component: PrimitiveComponent) -> None:
        if component.in_simulation():
            return
        if self.open is not None:
            self.open["modified"].append(component.get_path_name())
            # By object: a construction-script rerun's replacement has the path of the one it replaced.
            self.open.setdefault("objects", []).append(component)
        self.dirty = True

    def write(self, component: PrimitiveComponent, what: str) -> None:
        recorded = self.open is not None and any(
            obj is component for obj in self.open.get("objects", [])
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
            if path in editor.actors:
                return editor.actors[path]
            owner, _, name = path.rpartition(".")
            actor = editor.actors.get(owner)
            components = actor.components if actor is not None else []
            return next((c for c in components if c.get_name() == name), None)

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
                is_in_play_in_editor=lambda: editor.pie,
                editor_play_simulate=lambda: editor.begin("simulate"),
                editor_request_begin_play=lambda: editor.begin("play"),
                editor_request_end_play=editor.stop,
            ),
            "UnrealEditorSubsystem": SimpleNamespace(
                get_editor_world=lambda: None if editor.pie else editor.editor_world,
                get_game_world=lambda: editor.pie_world if editor.pie else None,
                get_level_viewport_camera_info=lambda: (Vector(*editor.camera[0]), Rotator(*editor.camera[1])),
                set_level_viewport_camera_info=editor.set_camera,
            ),
            # GWorld, the editor world wherever Remote Control runs a script,
            # which editor_world() falls back on during PIE.
            "LayersSubsystem": SimpleNamespace(get_world=lambda: editor.editor_world),
            "EditorActorSubsystem": SimpleNamespace(
                get_all_level_actors=lambda: (
                    [] if editor.pie else editor.world_actors(editor.editor_world)
                )
            ),
        }

        geometry_collision = SimpleNamespace(
            get_simple_collision_from_component=lambda component: component.simple,
            get_simple_collision_shape_count=lambda collision: collision,
        )

        def new_object(cls: type) -> Any:
            assert cls is StaticMeshComponent
            probe = StaticMeshComponent(editor, "Probe", Mesh(CUBE))
            probe.owner = Actor(editor, "Transient", "Transient", probe)
            probe._write = lambda what: None  # type: ignore[method-assign]
            return probe

        def all_actors_of_class(world: World, cls: Any) -> List[Actor]:
            return [
                actor
                for actor in editor.world_actors(world)
                if cls is Actor or actor.get_class() == cls
            ]

        def is_valid(obj: Any) -> bool:
            if obj is None:
                return False
            owner = obj if isinstance(obj, Actor) else getattr(obj, "owner", None)
            return not (owner is not None and owner.destroyed)

        settings = {
            "max_physics_delta_time": editor.max_physics_delta_time,
            "substepping": False,
            "max_substep_delta_time": 1.0 / 60.0,
            "max_substeps": 6,
        }
        physics_settings = SimpleNamespace(get_editor_property=settings.__getitem__)

        def register_ticker_callback(
            callback: Callable[[float], Any], delay: float = 0.0
        ) -> int:
            handle = next(editor.ticker_ids)
            editor.tickers[handle] = callback
            return handle

        def unregister_ticker_callback(handle: int) -> None:
            editor.tickers.pop(handle, None)

        def set_game_paused(world: World, paused: bool) -> bool:
            editor.paused = paused
            return True

        gameplay = SimpleNamespace(
            get_all_actors_of_class=all_actors_of_class,
            set_game_paused=set_game_paused,
            is_game_paused=lambda world: editor.paused,
            get_time_seconds=lambda world: editor.sim_time,
            get_world_delta_seconds=lambda world: editor.frame_seconds,
        )
        guids = SimpleNamespace(
            is_valid_guid=lambda guid: guid.value != 0,
            equal_equal_guid_guid=lambda a, b: a.value == b.value,
        )
        return SimpleNamespace(
            Actor=Actor,
            WorldSettings=type("WorldSettings", (Actor,), {}),
            ActorComponent=ActorComponent,
            SceneComponent=SceneComponent,
            PrimitiveComponent=PrimitiveComponent,
            StaticMeshComponent=StaticMeshComponent,
            ComponentMobility=MOBILITY,
            CollisionEnabled=COLLISION,
            CollisionChannel=CHANNELS,
            DynamicMeshComponent=DynamicMeshComponent,
            GeometryScript_Collision=geometry_collision,
            AttachmentRule=SimpleNamespace(KEEP_WORLD=Enum("KEEP_WORLD")),
            ScopedEditorTransaction=lambda description: Transaction(
                editor, description
            ),
            StaticMeshEditorSubsystem="StaticMeshEditorSubsystem",
            LevelEditorSubsystem="LevelEditorSubsystem",
            UnrealEditorSubsystem="UnrealEditorSubsystem",
            LayersSubsystem="LayersSubsystem",
            EditorActorSubsystem="EditorActorSubsystem",
            GameplayStatics=gameplay,
            GuidLibrary=guids,
            Vector=Vector,
            Rotator=Rotator,
            SystemLibrary=SimpleNamespace(is_valid=is_valid),
            PhysicsSettings=SimpleNamespace(
                get_default_object=lambda: physics_settings
            ),
            get_editor_subsystem=subsystems.__getitem__,
            load_object=load_object,
            find_object=load_object,
            new_object=new_object,
            register_ticker_callback=register_ticker_callback,
            unregister_ticker_callback=unregister_ticker_callback,
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
    # The editor-side modules a script keeps (the force ticker) live in this editor.
    from simul.adapters._unreal_scripts import EDITOR_CAMERA_MODULE, FORCE_MODULE

    monkeypatch.delitem(sys.modules, FORCE_MODULE, raising=False)
    monkeypatch.delitem(sys.modules, EDITOR_CAMERA_MODULE, raising=False)
    monkeypatch.setattr(unreal_runtime, "UNREAL_AVAILABLE", True)
    monkeypatch.setattr(unreal_runtime, "AIOHTTP_AVAILABLE", True)
    session = unreal_runtime.UnrealRuntimeSession(Settings())

    async def execute(code: str, mode: str = "ExecuteFile") -> Dict[str, Any]:
        return editor.run_python(code)

    monkeypatch.setattr(session, "_execute_python", execute)
    return session
