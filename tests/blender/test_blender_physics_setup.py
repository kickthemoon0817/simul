"""Rigid body, constraint and force-field setup against a stand-in Blender.

Each case pins a defect the Blender 5.0.1 audit found: a failed setup left a
rigid body, a world or an orphan constraint empty behind; values outside
Blender's ranges were stored as given (a mass of -5 fell through the ground);
fields and constraint empties followed the user's "Align: View" preference;
the type lists were wrong; and a bake had no bound on Blender's main thread.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional

import pytest
from pydantic import ValidationError

from simul.adapters import blender_runtime
from simul.mcp.schemas.blender import (
    BlenderAddConstraintRequest,
    BlenderAddConstraintResponse,
    BlenderAddForceFieldRequest,
    BlenderAddForceFieldResponse,
    BlenderBakeSimulationRequest,
    BlenderGetConstraintInfoResponse,
    BlenderGetForceFieldInfoResponse,
    BlenderSetupRigidBodyRequest,
    BlenderSetupRigidBodyResponse,
)

Session = blender_runtime.BlenderRuntimeSession


def _euler_matrix(rotation: List[float]) -> List[List[float]]:
    """Blender's XYZ Euler as a 3x3 matrix, R = Rz Ry Rx."""
    rx, ry, rz = rotation
    cx, sx, cy, sy, cz, sz = (
        math.cos(rx),
        math.sin(rx),
        math.cos(ry),
        math.sin(ry),
        math.cos(rz),
        math.sin(rz),
    )
    return [
        [cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz],
        [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz],
        [-sy, sx * cy, cx * cy],
    ]


class _Matrix:
    """A world matrix from a location and an XYZ Euler, indexed [row][column]."""

    def __init__(self, location: Any, rotation: Any) -> None:
        self.translation = [float(v) for v in location]
        self._rotation = [float(v) for v in rotation]
        self._rows = _euler_matrix(self._rotation)

    def __getitem__(self, row: int) -> List[float]:
        return self._rows[row]

    def to_euler(self, order: str) -> List[float]:
        return list(self._rotation)


class FakeObject(SimpleNamespace):
    def evaluated_get(self, depsgraph: Any) -> Any:
        return SimpleNamespace(matrix_world=_Matrix(self.location, self.rotation_euler))

    def select_set(self, value: bool) -> None:
        self.selected = value

    def as_pointer(self) -> int:
        return id(self)


def _object(
    name: str, kind: str = "MESH", location: Any = (0.0, 0.0, 0.0)
) -> FakeObject:
    return FakeObject(
        name=name,
        type=kind,
        location=tuple(location),
        rotation_euler=(0.0, 0.0, 0.0),
        rotation_mode="XYZ",
        rigid_body=None,
        rigid_body_constraint=None,
        field=None,
        selected=False,
        users_collection=[],
    )


class FakeObjects(dict):
    """bpy.data.objects: iterates over the objects, and makes and removes them."""

    def __init__(self, blender: "FakeBlender") -> None:
        super().__init__()
        self._blender = blender

    def __iter__(self) -> Iterator[Any]:  # type: ignore[override]
        return iter(list(self.values()))

    def new(self, name: str, data: Any) -> FakeObject:
        return self._blender._new_object(name, data)

    def remove(self, obj: FakeObject, do_unlink: bool = False) -> None:
        self._blender._remove_object(obj, do_unlink)


class FakeCollection(SimpleNamespace):
    """A collection that refuses links when it comes from a library, as Blender's does."""

    def __init__(self, name: str, blender: "FakeBlender", library: Any = None) -> None:
        super().__init__(
            name=name, library=library, is_editable=library is None, hide_viewport=False
        )
        self.objects = SimpleNamespace(link=lambda obj: self._link(blender, obj))

    def _link(self, blender: "FakeBlender", obj: FakeObject) -> None:
        if self.library is not None:
            raise RuntimeError(
                f"Error: Could not (un)link the object '{obj.name}' because the "
                f"collection '{self.name}' is linked"
            )
        obj.users_collection = [self]
        blender.orphans.pop(obj.name, None)
        blender.add(obj)


#: RigidBodyConstraint's limit and spring axes (Blender 5.0.1).
LIMIT_AXES = ("lin_x", "lin_y", "lin_z", "ang_x", "ang_y", "ang_z")
SPRING_AXES = ("x", "y", "z", "ang_x", "ang_y", "ang_z")

#: Blender 5.0.1's RigidBodyObject.collision_shape items.
BLENDER_SHAPES = (
    "BOX",
    "SPHERE",
    "CAPSULE",
    "CYLINDER",
    "CONE",
    "CONVEX_HULL",
    "MESH",
    "COMPOUND",
)


class FakeRigidBody(SimpleNamespace):
    """RigidBodyObject; an enum it does not know raises TypeError, as bpy's does."""

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "collision_shape" and value not in BLENDER_SHAPES:
            raise TypeError(f'enum "{value}" not found')
        super().__setattr__(name, value)


def _rigid_body(kind: str) -> FakeRigidBody:
    return FakeRigidBody(
        type=kind,
        mass=1.0,
        friction=0.5,
        restitution=0.0,
        collision_shape="CONVEX_HULL",
        linear_damping=0.04,
        angular_damping=0.1,
        kinematic=False,
        enabled=True,
        use_margin=False,
        collision_margin=0.04,
        collision_collections=[True] + [False] * 19,
        use_deactivation=False,
        use_start_deactivated=False,
    )


def _constraint(kind: str) -> SimpleNamespace:
    settings: Dict[str, Any] = {
        "type": kind,
        "object1": None,
        "object2": None,
        "enabled": True,
        "disable_collisions": True,
        "use_breaking": False,
        "breaking_threshold": 10.0,
        "use_motor_ang": False,
        "use_motor_lin": False,
        "motor_ang_target_velocity": 1.0,
        "motor_ang_max_impulse": 1.0,
        "motor_lin_target_velocity": 1.0,
        "motor_lin_max_impulse": 1.0,
    }
    for key in LIMIT_AXES:
        settings.update(
            {
                f"use_limit_{key}": False,
                f"limit_{key}_lower": -1.0,
                f"limit_{key}_upper": 1.0,
            }
        )
    for key in SPRING_AXES:
        settings.update(
            {
                f"use_spring_{key}": False,
                f"spring_stiffness_{key}": 10.0,
                f"spring_damping_{key}": 0.5,
            }
        )
    return SimpleNamespace(**settings)


class FakeBlender:
    """bpy for the physics setup operations, recording every operator call."""

    def __init__(self) -> None:
        self.calls: List[Any] = []
        self.objects = FakeObjects(self)
        self.world: Optional[SimpleNamespace] = None
        self.fail: Dict[str, BaseException] = {}
        #: The rotation an add operator gives an object without align='WORLD':
        #: the 3D View's, as with the "Align: View" preference.
        self.view_rotation = (1.1, 0.0, 0.8)
        #: Whether an add operator makes its object the context's active object;
        #: in the GUI it does not when the active collection is disabled in viewports.
        self.reports_active_object = True
        #: Objects effector_add makes, for an operator that misbehaves.
        self.effector_objects = 1
        self._overrides: List[Dict[str, Any]] = []
        self.scene_collection = FakeCollection("Scene Collection", self)
        self.view_layer = SimpleNamespace(
            objects=SimpleNamespace(active=None),
            active_layer_collection=SimpleNamespace(
                collection=self.scene_collection, exclude=False, hide_viewport=False
            ),
        )
        self.scene = SimpleNamespace(
            name="Scene",
            collection=self.scene_collection,
            frame_start=1,
            frame_end=250,
            frame_current=1,
            rigidbody_world=None,
            objects=[],
            frame_set=lambda frame: setattr(self.scene, "frame_current", frame),
        )
        self.context = SimpleNamespace(
            scene=self.scene,
            view_layer=self.view_layer,
            active_object=None,
            temp_override=self._temp_override,
            evaluated_depsgraph_get=lambda: None,
        )
        self.data = SimpleNamespace(objects=self.objects)
        self.app = SimpleNamespace(
            version=(5, 0, 1), version_string="5.0.1", background=True
        )
        self.ops = SimpleNamespace(
            rigidbody=SimpleNamespace(
                world_add=self._op("rigidbody.world_add", self._world_add),
                world_remove=self._op("rigidbody.world_remove", self._world_remove),
                object_add=self._op("rigidbody.object_add", self._object_add),
                object_remove=self._op("rigidbody.object_remove", self._object_remove),
                constraint_add=self._op(
                    "rigidbody.constraint_add", self._constraint_add
                ),
                constraint_remove=self._op(
                    "rigidbody.constraint_remove", self._constraint_remove
                ),
            ),
            object=SimpleNamespace(
                effector_add=self._op("object.effector_add", self._effector_add),
                empty_add=self._op("object.empty_add", self._empty_add),
            ),
            ptcache=SimpleNamespace(
                bake_from_cache=self._op(
                    "ptcache.bake_from_cache", self._bake_from_cache
                ),
                free_bake=self._op("ptcache.free_bake", lambda: None),
            ),
        )

    # -- scene content ---------------------------------------------------------

    def add(self, obj: FakeObject) -> FakeObject:
        self.objects[obj.name] = obj
        self.scene.objects.append(obj)
        return obj

    def selection(self) -> Any:
        """What the user would see selected, and the active object."""
        active = self.view_layer.objects.active
        return (
            sorted(o.name for o in self.objects.values() if o.selected),
            active.name if active is not None else None,
        )

    def body(
        self, name: str, kind: str = "ACTIVE", location: Any = (0.0, 0.0, 0.0)
    ) -> FakeObject:
        obj = self.add(_object(name, location=location))
        self._ensure_world()
        obj.rigid_body = _rigid_body(kind)
        self.world.collection.all_objects.append(obj)
        return obj

    def _ensure_world(self) -> None:
        if self.world is None:
            self._world_add()

    # -- operators ---------------------------------------------------------------

    def _op(self, name: str, run: Any) -> Any:
        def operator(**properties: Any) -> set:
            self.calls.append((name, properties))
            if name in self.fail:
                raise self.fail[name]
            run(**properties)
            return {"FINISHED"}

        return operator

    @contextmanager
    def _temp_override(self, **members: Any) -> Iterator[None]:
        self._overrides.append(members)
        try:
            yield
        finally:
            self._overrides.pop()

    def _override(self, member: str) -> Any:
        return self._overrides[-1][member]

    def _context_object(self) -> Any:
        """The operator's object: the override's, else the active object, as in Blender."""
        for members in reversed(self._overrides):
            if "object" in members:
                return members["object"]
        return self.view_layer.objects.active

    def _make_active(self, obj: FakeObject) -> None:
        """What an add operator does: put the new object in the active collection,
        select only it and make it active."""
        obj.users_collection = [self.view_layer.active_layer_collection.collection]
        for other in self.objects.values():
            other.selected = False
        obj.selected = True
        self.view_layer.objects.active = obj
        self.context.active_object = obj if self.reports_active_object else None

    def _world_add(self) -> None:
        self.world = SimpleNamespace(
            enabled=True,
            time_scale=1.0,
            collection=SimpleNamespace(name="RigidBodyWorld", all_objects=[]),
            constraints=SimpleNamespace(all_objects=[]),
            point_cache=SimpleNamespace(
                frame_start=1, frame_end=250, is_baked=False, is_outdated=True, info=""
            ),
        )
        self.scene.rigidbody_world = self.world

    def _world_remove(self) -> None:
        self.world = None
        self.scene.rigidbody_world = None

    def _object_add(self, type: str) -> None:
        obj = self._context_object()
        if obj.type != "MESH":
            raise RuntimeError(
                "Operator bpy.ops.rigidbody.object_add.poll() failed, context is incorrect"
            )
        if obj.rigid_body is None:
            obj.rigid_body = _rigid_body(type)
            self.world.collection.all_objects.append(obj)
        else:
            obj.rigid_body.type = type

    def _object_remove(self) -> None:
        obj = self._context_object()
        obj.rigid_body = None
        self.world.collection.all_objects.remove(obj)

    def _constraint_add(self, type: str) -> None:
        # Like Blender's, it acts on the view layer's active object.
        empty = self.view_layer.objects.active
        empty.rigid_body_constraint = _constraint(type)
        self.world.constraints.all_objects.append(empty)

    def _constraint_remove(self) -> None:
        empty = self.view_layer.objects.active
        empty.rigid_body_constraint = None
        self.world.constraints.all_objects.remove(empty)

    def _effector_add(
        self,
        type: str,
        location: Any = (0.0, 0.0, 0.0),
        align: str = "PREFERENCE",
        rotation: Any = (0.0, 0.0, 0.0),
    ) -> None:
        if self.effector_objects == 0:
            return
        obj = self.add(_object(self._free_name("Field"), "EMPTY", location))
        obj.rotation_euler = tuple(rotation) if align == "WORLD" else self.view_rotation
        obj.field = SimpleNamespace(
            type="FLUID_FLOW" if type == "FLUID" else type,
            strength=1.0,
            shape="POINT",
            flow=0.0,
            falloff_type="SPHERE",
            falloff_power=0.0,
            use_max_distance=False,
            distance_max=0.0,
            noise=0.0,
        )
        for _ in range(self.effector_objects - 1):
            self.add(_object(self._free_name("Extra"), "EMPTY"))
        self._make_active(obj)

    def _empty_add(
        self,
        location: Any = (0.0, 0.0, 0.0),
        align: str = "PREFERENCE",
        rotation: Any = (0.0, 0.0, 0.0),
    ) -> None:
        obj = self.add(_object(self._free_name("Empty"), "EMPTY", location))
        obj.rotation_euler = tuple(rotation) if align == "WORLD" else self.view_rotation
        self._make_active(obj)

    def _bake_from_cache(self) -> None:
        self._override("point_cache").is_baked = True

    # -- data ------------------------------------------------------------------

    def _free_name(self, name: str) -> str:
        # Blender numbers a name that is taken: RBC_A_B.001.
        taken, base, number = set(self.objects.keys()) | set(self.orphans), name, 0
        while name in taken:
            number += 1
            name = f"{base}.{number:03d}"
        return name

    #: Objects made in bpy.data but linked into no collection (0 users).
    orphans: Dict[str, FakeObject] = {}

    def _new_object(self, name: str, data: Any) -> FakeObject:
        obj = _object(self._free_name(name), "EMPTY")
        self.orphans[obj.name] = obj
        self.calls.append(("data.objects.new", {"name": obj.name}))
        return obj

    def _remove_object(self, obj: FakeObject, do_unlink: bool = False) -> None:
        self.calls.append(("data.objects.remove", {"name": obj.name}))
        self.orphans.pop(obj.name, None)
        if obj.name in self.objects:
            del self.objects[obj.name]
            self.scene.objects.remove(obj)


@pytest.fixture
def blender(monkeypatch: pytest.MonkeyPatch) -> Any:
    fake = FakeBlender()
    fake.orphans = {}
    monkeypatch.setattr(blender_runtime, "bpy", fake)
    monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
    session = Session(scene_scope=True, path_policy=object())

    def lookup(name: str) -> Any:
        found = [obj for obj in fake.objects.values() if obj.name == name]
        if not found:
            raise ValueError(f"Object not found: {name}")
        return found[0]

    session._get_object_or_raise = lookup  # type: ignore[method-assign]
    return fake, session


def names(fake: FakeBlender) -> List[str]:
    """The names of the objects in the scene, as they are now."""
    return sorted(obj.name for obj in fake.objects.values())


def named(fake: FakeBlender, name: str) -> FakeObject:
    (found,) = [obj for obj in fake.objects.values() if obj.name == name]
    return found


def _operators(fake: FakeBlender) -> List[str]:
    return [name for name, _ in fake.calls]


# -- B3-04: a failed setup changes nothing ------------------------------------


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"collision_shape": "BOXX"}, "Invalid collision_shape 'BOXX'"),
        ({"body_type": "DYNAMIC"}, "Invalid body_type 'DYNAMIC'"),
    ],
)
def test_a_bad_enum_is_refused_before_anything_changes(
    blender: Any, arguments: Dict[str, Any], message: str
) -> None:
    fake, session = blender
    cube = fake.add(_object("Cube"))

    with pytest.raises(ValueError, match=message):
        session.setup_rigid_body("Cube", **arguments)

    assert cube.rigid_body is None
    assert fake.scene.rigidbody_world is None
    assert fake.calls == []


def test_a_non_mesh_object_gets_a_clear_error(blender: Any) -> None:
    fake, session = blender
    fake.add(_object("Cam", "CAMERA"))

    with pytest.raises(ValueError, match="need a mesh object, and 'Cam' is a CAMERA"):
        session.setup_rigid_body("Cam")

    assert fake.scene.rigidbody_world is None
    assert fake.calls == []


def test_setup_acts_through_an_override_and_keeps_the_selection(blender: Any) -> None:
    fake, session = blender
    cube = fake.add(_object("Cube"))
    other = fake.add(_object("Other"))
    fake.view_layer.objects.active = other

    result = session.setup_rigid_body(
        "Cube", collision_shape="box", body_type="passive"
    )

    assert fake.view_layer.objects.active is other
    assert (cube.rigid_body.type, cube.rigid_body.collision_shape) == ("PASSIVE", "BOX")
    assert result["world_created"] is True and result["replaced"] is False
    assert fake.world.point_cache.frame_end == 250
    BlenderSetupRigidBodyResponse(success=True, **result)


def test_a_failure_after_the_operator_removes_the_new_body_and_world(
    blender: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, session = blender
    cube = fake.add(_object("Cube"))
    # Let a shape Blender refuses past the session's check, so the failure
    # comes after the operator added the body.
    monkeypatch.setattr(
        Session, "_COLLISION_SHAPES", BLENDER_SHAPES + ("BOXX",), raising=False
    )

    with pytest.raises(TypeError, match='enum "BOXX" not found'):
        session.setup_rigid_body("Cube", collision_shape="BOXX")

    assert cube.rigid_body is None
    assert fake.scene.rigidbody_world is None
    assert _operators(fake)[-2:] == [
        "rigidbody.object_remove",
        "rigidbody.world_remove",
    ]


def test_a_failed_resetup_restores_the_earlier_settings(
    blender: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, session = blender
    cube = fake.add(_object("Cube"))
    session.setup_rigid_body("Cube", mass=3.0, collision_shape="SPHERE")
    # Let a shape Blender refuses past the session's check, so the failure
    # comes after the operator added the body.
    monkeypatch.setattr(
        Session, "_COLLISION_SHAPES", BLENDER_SHAPES + ("BOXX",), raising=False
    )

    with pytest.raises(TypeError):
        session.setup_rigid_body(
            "Cube", body_type="PASSIVE", mass=9.0, collision_shape="BOXX"
        )

    assert (cube.rigid_body.type, cube.rigid_body.mass) == ("ACTIVE", 3.0)
    assert cube.rigid_body.collision_shape == "SPHERE"
    assert fake.scene.rigidbody_world is not None


def test_a_failed_operator_keeps_its_message_and_leaves_no_world(blender: Any) -> None:
    fake, session = blender
    fake.add(_object("Cube"))
    fake.fail["rigidbody.object_add"] = RuntimeError("poll() failed")

    with pytest.raises(RuntimeError, match="poll"):
        session.setup_rigid_body("Cube")

    assert fake.scene.rigidbody_world is None


# -- B3-11: Blender's ranges ---------------------------------------------------


@pytest.mark.parametrize(
    "arguments",
    [
        {"mass": -5.0},
        {"mass": 0.0},
        {"friction": -1.0},
        {"restitution": 7.0},
        {"linear_damping": 3.0},
        {"angular_damping": -2.0},
        {"mass": float("nan")},
        {"collision_margin": 2.0},
        {"collision_collections": [20]},
    ],
)
def test_values_outside_blenders_ranges_are_refused(
    blender: Any, arguments: Dict[str, Any]
) -> None:
    fake, session = blender
    fake.add(_object("Cube"))

    with pytest.raises(ValueError):
        session.setup_rigid_body("Cube", **arguments)
    with pytest.raises(ValidationError):
        BlenderSetupRigidBodyRequest(object_name="Cube", **arguments)

    assert fake.calls == []


def test_new_body_settings_are_written_and_reported(blender: Any) -> None:
    fake, session = blender
    cube = fake.add(_object("Cube"))

    result = session.setup_rigid_body(
        "Cube",
        kinematic=True,
        collision_margin=0.01,
        collision_collections=[3, 0, 3],
        use_deactivation=True,
        start_deactivated=True,
    )

    rb = cube.rigid_body
    assert (rb.kinematic, rb.use_margin, rb.collision_margin) == (True, True, 0.01)
    assert [i for i, member in enumerate(rb.collision_collections) if member] == [0, 3]
    assert (rb.use_deactivation, rb.use_start_deactivated) == (True, True)
    assert result["collision_collections"] == [0, 3]
    assert result["kinematic"] is True


def test_setting_up_again_leaves_optional_settings_alone(blender: Any) -> None:
    fake, session = blender
    cube = fake.add(_object("Cube"))
    session.setup_rigid_body("Cube", kinematic=True)

    result = session.setup_rigid_body("Cube", mass=2.0)

    assert cube.rigid_body.kinematic is True
    assert result["replaced"] is True and result["world_created"] is False


# -- B3-05 / B3-22: constraints -------------------------------------------------


def test_a_bad_constraint_type_leaves_nothing_behind(blender: Any) -> None:
    fake, session = blender
    fake.body("A")
    fake.body("B", location=(3.0, 0.0, 0.0))

    with pytest.raises(ValueError, match="Invalid constraint_type 'HINGEE'"):
        session.add_rigid_body_constraint("HINGEE", "A", "B")

    assert names(fake) == ["A", "B"]
    assert fake.calls == []


@pytest.mark.parametrize(
    ("first", "second", "message"),
    [
        ("A", "A", "two different objects"),
        ("A", "Plain", "'Plain' is not a rigid body"),
    ],
)
def test_a_constraint_needs_two_distinct_rigid_bodies(
    blender: Any, first: str, second: str, message: str
) -> None:
    fake, session = blender
    fake.body("A")
    fake.add(_object("Plain"))

    with pytest.raises(ValueError, match=message):
        session.add_rigid_body_constraint("FIXED", first, second)

    assert names(fake) == ["A", "Plain"]
    assert fake.calls == []


def test_a_failed_constraint_removes_its_empty_and_restores_the_active_object(
    blender: Any,
) -> None:
    fake, session = blender
    a = fake.body("A")
    fake.body("B", location=(2.0, 0.0, 0.0))
    fake.view_layer.objects.active = a
    fake.fail["rigidbody.constraint_add"] = RuntimeError("context is incorrect")

    with pytest.raises(RuntimeError, match="context is incorrect"):
        session.add_rigid_body_constraint("HINGE", "A", "B")

    assert names(fake) == ["A", "B"]
    assert fake.view_layer.objects.active is a
    assert "data.objects.remove" in _operators(fake)


def test_the_pivot_defaults_to_midway_and_the_axis_turns_the_empty(
    blender: Any,
) -> None:
    fake, session = blender
    a = fake.body("Pivot", "PASSIVE", location=(0.0, 0.0, 5.0))
    fake.body("Bob", location=(2.0, 0.0, 5.0))
    fake.view_layer.objects.active = a

    result = session.add_rigid_body_constraint(
        "HINGE", "Pivot", "Bob", axis=[0.0, 2.0, 0.0]
    )

    empty = named(fake, result["constraint_name"])
    assert empty.location == (1.0, 0.0, 5.0)
    assert result["pivot"] == [1.0, 0.0, 5.0]
    # The hinge turns about local Z, now world +Y.
    assert result["axis"] == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)
    assert result["local_axes"]["x"] == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)
    assert fake.view_layer.objects.active is a
    # Made from data, not empty_add: the user's Align preference cannot turn it.
    assert "data.objects.new" in _operators(fake)
    BlenderAddConstraintResponse(success=True, **result)


def test_limits_motor_and_breaking_are_written(blender: Any) -> None:
    fake, session = blender
    fake.body("Base", "PASSIVE")
    fake.body("Wheel", location=(0.0, 0.0, 2.0))

    hinge = session.add_rigid_body_constraint(
        "HINGE", "Base", "Wheel", limits={"ang_z": [-0.5, 0.5]}, breaking_threshold=20.0
    )
    motor = session.add_rigid_body_constraint(
        "MOTOR",
        "Base",
        "Wheel",
        motor_angular_velocity=2.0,
        motor_angular_max_impulse=5.0,
    )
    idle = session.add_rigid_body_constraint("MOTOR", "Base", "Wheel")

    assert hinge["limits"] == {"ang_z": [-0.5, 0.5]}
    assert hinge["breaking_threshold"] == 20.0
    assert motor["motor"] == {
        "angular_velocity": 2.0,
        "angular_max_impulse": 5.0,
        "linear_velocity": None,
        "linear_max_impulse": 1.0,
    }
    # A MOTOR turns about local X.
    assert motor["axis"] == pytest.approx([1.0, 0.0, 0.0])
    assert "drives nothing" in idle["warnings"][0]
    info = session.get_constraint_info(hinge["constraint_name"])
    BlenderGetConstraintInfoResponse(success=True, **info)
    assert info["limits"] == {"ang_z": [-0.5, 0.5]}


def test_springs_are_written_for_a_generic_spring(blender: Any) -> None:
    fake, session = blender
    fake.body("Ceiling", "PASSIVE")
    fake.body("Weight", location=(0.0, 0.0, -2.0))

    result = session.add_rigid_body_constraint(
        "GENERIC_SPRING",
        "Ceiling",
        "Weight",
        limits={"lin_z": [-1.0, 1.0]},
        springs={"z": {"stiffness": 50.0, "damping": 0.5}},
    )

    assert result["springs"] == {"z": {"stiffness": 50.0, "damping": 0.5}}
    assert result["limits"] == {"lin_z": [-1.0, 1.0]}


@pytest.mark.parametrize(
    ("kind", "arguments", "message"),
    [
        ("HINGE", {"limits": {"lin_x": [0.0, 1.0]}}, "no lin_x limit"),
        ("HINGE", {"limits": {"ang_z": [1.0, -1.0]}}, "lower <= upper"),
        ("HINGE", {"limits": {"ang_z": [-7.0, 0.0]}}, "2 pi"),
        (
            "FIXED",
            {"springs": {"z": {"stiffness": 1.0, "damping": 0.0}}},
            "GENERIC_SPRING",
        ),
        ("HINGE", {"motor_angular_velocity": 1.0}, "MOTOR constraints only"),
        ("FIXED", {"breaking_threshold": -1.0}, "at least 0"),
        ("HINGE", {"axis": [0.0, 0.0, 0.0]}, "zero vector"),
    ],
)
def test_constraint_settings_that_do_not_fit_the_type_are_refused(
    blender: Any, kind: str, arguments: Dict[str, Any], message: str
) -> None:
    fake, session = blender
    fake.body("A")
    fake.body("B", location=(1.0, 0.0, 0.0))

    with pytest.raises(ValueError, match=message):
        session.add_rigid_body_constraint(kind, "A", "B", **arguments)
    with pytest.raises(ValidationError):
        BlenderAddConstraintRequest(
            constraint_type=kind, object1_name="A", object2_name="B", **arguments
        )

    assert names(fake) == ["A", "B"]


def test_constraint_request_refuses_the_same_object_twice() -> None:
    with pytest.raises(ValidationError, match="two different objects"):
        BlenderAddConstraintRequest(
            constraint_type="fixed", object1_name="A", object2_name="A"
        )


# -- B3-10 / B3-18 / B3-19: force fields ----------------------------------------


def test_a_force_field_is_placed_in_world_space_along_its_direction(
    blender: Any,
) -> None:
    fake, session = blender

    result = session.add_force_field(
        "wind",
        strength=30.0,
        location=[0.0, 0.0, -2.0],
        name="Wind",
        direction=[1.0, 0.0, 0.0],
    )

    name, properties = fake.calls[-1]
    assert name == "object.effector_add"
    assert properties["align"] == "WORLD"
    assert result["direction"] == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)
    assert result["name"] == "Wind" and result["field_type"] == "WIND"
    BlenderAddForceFieldResponse(success=True, **result)


def test_force_field_settings_are_written(blender: Any) -> None:
    fake, session = blender

    result = session.add_force_field(
        "FORCE",
        shape="plane",
        falloff_type="tube",
        falloff_power=2.0,
        distance_max=5.0,
        noise=1.0,
        flow=0.5,
    )

    field = fake.context.active_object.field
    assert (field.shape, field.falloff_type, field.falloff_power) == (
        "PLANE",
        "TUBE",
        2.0,
    )
    assert (field.use_max_distance, field.distance_max, field.noise, field.flow) == (
        True,
        5.0,
        1.0,
        0.5,
    )
    assert result["distance_max"] == 5.0


@pytest.mark.parametrize(
    ("given", "added"),
    [("MAGNETIC", "MAGNET"), ("FLUID_FLOW", "FLUID"), ("Texture", "TEXTURE")],
)
def test_force_field_type_names(blender: Any, given: str, added: str) -> None:
    fake, session = blender

    session.add_force_field(given)

    assert fake.calls[-1][1]["type"] == added
    assert BlenderAddForceFieldRequest(field_type=given).field_type == added


def test_an_unknown_force_field_type_is_refused(blender: Any) -> None:
    fake, session = blender

    with pytest.raises(ValueError, match="Invalid field_type 'GRAVITY'"):
        session.add_force_field("GRAVITY")
    with pytest.raises(ValidationError):
        BlenderAddForceFieldRequest(field_type="GRAVITY")
    with pytest.raises(ValidationError):
        BlenderAddForceFieldRequest(field_type="FORCE", strength=float("nan"))

    assert fake.calls == []


def test_force_field_info_refuses_a_collider(blender: Any) -> None:
    fake, session = blender
    cube = fake.add(_object("Cube"))
    # What a Collision modifier leaves: field settings with no field type.
    cube.field = SimpleNamespace(
        type="NONE",
        strength=1.0,
        shape="POINT",
        flow=0.0,
        falloff_type="SPHERE",
        falloff_power=0.0,
        use_max_distance=False,
        distance_max=0.0,
        noise=0.0,
    )

    with pytest.raises(ValueError, match="no force field"):
        session.get_force_field_info("Cube")


def test_force_field_info_reports_the_direction(blender: Any) -> None:
    fake, session = blender
    session.add_force_field("WIND", direction=[0.0, 0.0, -1.0])

    info = session.get_force_field_info("Field")

    assert info["direction"] == pytest.approx([0.0, 0.0, -1.0], abs=1e-9)
    BlenderGetForceFieldInfoResponse(success=True, **info)


def test_compound_is_a_collision_shape() -> None:
    assert (
        BlenderSetupRigidBodyRequest(
            object_name="A", collision_shape="compound"
        ).collision_shape
        == "COMPOUND"
    )


# -- the axis rotation ------------------------------------------------------------


@pytest.mark.parametrize("primary", ["X", "Z"])
@pytest.mark.parametrize(
    "axis",
    [
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.3, -0.5, 0.8],
        [-2.0, 1.0, 0.1],
    ],
)
def test_the_axis_rotation_turns_the_primary_axis_onto_the_axis(
    primary: str, axis: List[float]
) -> None:
    rotation = Session._axis_rotation(axis, primary, "axis")

    matrix = _euler_matrix(list(rotation))
    column = 2 if primary == "Z" else 0
    turned = [matrix[row][column] for row in range(3)]
    length = math.sqrt(sum(v * v for v in axis))
    assert turned == pytest.approx([v / length for v in axis], abs=1e-9)


def test_no_axis_means_no_rotation() -> None:
    assert Session._axis_rotation(None, "Z", "axis") == (0.0, 0.0, 0.0)


# -- B3-12: the bake's work budget -----------------------------------------------


def test_a_bake_over_the_work_budget_needs_allow_long_bake(blender: Any) -> None:
    fake, session = blender
    for index in range(1000):
        fake.body(f"B{index}")

    with pytest.raises(ValueError, match="allow_long_bake") as refused:
        session.bake_simulation(1, 1000)
    session.bake_simulation(1, 1000, allow_long_bake=True)

    assert "1000000 body-frames" in str(refused.value)
    assert "BLENDER__CONNECTION_TIMEOUT" in str(refused.value)
    assert fake.world.point_cache.is_baked is True


def test_a_bake_over_the_frame_budget_needs_allow_long_bake(blender: Any) -> None:
    fake, session = blender
    fake.body("Drop")

    with pytest.raises(ValueError, match="allow_long_bake"):
        session.bake_simulation(1, 1_000_000)

    assert fake.calls == []


def test_a_small_bake_runs_without_confirmation(blender: Any) -> None:
    fake, session = blender
    for index in range(100):
        fake.body(f"B{index}")

    session.bake_simulation(1, 250)

    assert fake.world.point_cache.is_baked is True
    assert BlenderBakeSimulationRequest(
        frame_start=1, frame_end=2, allow_long_bake=True
    ).allow_long_bake


# -- review of #322: what the operators leave behind ----------------------------


def test_a_force_field_is_found_when_blender_reports_no_active_object(
    blender: Any,
) -> None:
    """In the GUI an add into a collection disabled in viewports leaves active_object None."""
    fake, session = blender
    fake.reports_active_object = False

    result = session.add_force_field("WIND", name="Gust")

    (field,) = fake.objects.values()
    assert result["name"] == field.name == "Gust"
    assert field.field.strength == 1.0


@pytest.mark.parametrize("made", [0, 2])
def test_an_effector_add_that_makes_no_single_object_leaves_nothing(
    blender: Any, made: int
) -> None:
    fake, session = blender
    fake.effector_objects = made

    with pytest.raises(RuntimeError, match=f"created {made} objects, not one"):
        session.add_force_field("WIND")

    assert names(fake) == []


def test_a_force_field_ignores_the_align_view_preference(blender: Any) -> None:
    fake, session = blender

    result = session.add_force_field("WIND")

    assert named(fake, result["name"]).rotation_euler == (0.0, 0.0, 0.0)
    assert result["direction"] == pytest.approx([0.0, 0.0, 1.0])
    assert fake.calls[-1][1].get("align") == "WORLD"


def test_a_constraint_empty_ignores_the_align_view_preference_and_the_selection(
    blender: Any,
) -> None:
    fake, session = blender
    a = fake.body("A")
    fake.body("B", location=(2.0, 0.0, 0.0))
    a.selected = True
    fake.view_layer.objects.active = a

    result = session.add_rigid_body_constraint("FIXED", "A", "B")

    empty = named(fake, result["constraint_name"])
    assert empty.rotation_euler == (0.0, 0.0, 0.0)
    assert empty.location == (1.0, 0.0, 0.0)
    assert fake.selection() == (["A"], "A")


def test_a_constraint_with_a_linked_active_collection_goes_to_the_scene_collection(
    blender: Any,
) -> None:
    """Blender refuses to link into a library collection; that used to orphan the empty."""
    fake, session = blender
    fake.body("A")
    fake.body("B", location=(2.0, 0.0, 0.0))
    linked = FakeCollection("LibCol", fake, library="lib.blend")
    fake.view_layer.active_layer_collection.collection = linked

    result = session.add_rigid_body_constraint("FIXED", "A", "B")

    empty = named(fake, result["constraint_name"])
    assert empty.users_collection == [fake.scene_collection]
    assert fake.orphans == {}


@pytest.mark.parametrize("setting", ["exclude", "hide_viewport"])
def test_a_constraint_skips_an_active_collection_the_view_layer_hides(
    blender: Any, setting: str
) -> None:
    fake, session = blender
    fake.body("A")
    fake.body("B", location=(2.0, 0.0, 0.0))
    hidden = FakeCollection("Hidden", fake)
    layer = fake.view_layer.active_layer_collection
    layer.collection = hidden
    setattr(layer, setting, True)

    result = session.add_rigid_body_constraint("FIXED", "A", "B")

    assert named(fake, result["constraint_name"]).users_collection == [
        fake.scene_collection
    ]


def test_a_failed_link_leaves_no_orphan_empty(blender: Any) -> None:
    fake, session = blender
    fake.body("A")
    fake.body("B", location=(2.0, 0.0, 0.0))
    fake.scene_collection.library = "lib.blend"  # nothing to link into
    fake.view_layer.active_layer_collection.collection = fake.scene_collection

    with pytest.raises(RuntimeError, match="is linked"):
        session.add_rigid_body_constraint("FIXED", "A", "B")

    assert fake.orphans == {}
    assert names(fake) == ["A", "B"]
