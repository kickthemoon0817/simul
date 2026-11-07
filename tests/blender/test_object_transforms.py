"""Object transforms: rotation in every rotation mode, local versus world, parenting.

Blender keeps an object's rotation only in the property its rotation_mode
names, so writing rotation_euler to a glTF node (QUATERNION) changed nothing
while set_blender_object_transform replied success. get_object_info labelled
the parent-relative location as world space. In background Blender and the
bpy module, matrix_world lags a transform written since the last depsgraph
evaluation. set_object_parent moved the child, and accepted a parent loop
that Blender silently refuses. These tests pin each of those.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Callable, List, Optional, Sequence

import numpy as np
import pytest

from simul.adapters import blender_runtime
from simul.adapters.blender_runtime import BlenderRuntimeSession
from tests.fakes import (
    FakeConstraint,
    FakeEuler,
    FakeMatrix,
    FakeTransformObject,
    FakeVector,
    euler_matrix,
    matrix_quaternion,
    quaternion_matrix,
)

ROTATION_MODES = ("XYZ", "ZXY", "YXZ", "QUATERNION", "AXIS_ANGLE")
TURN = (0.3, -0.4, 1.2)


def _bpy(objects: Sequence[FakeTransformObject]) -> SimpleNamespace:
    """A bpy whose evaluated_depsgraph_get evaluates every object, as Blender's does."""
    store = {obj.name: obj for obj in objects}

    class Objects:
        def get(self, name: str) -> Optional[FakeTransformObject]:
            return store.get(name)

        def __iter__(self):  # type: ignore[no-untyped-def]
            return iter(store.values())

        def __len__(self) -> int:
            return len(store)

    def evaluated_depsgraph_get() -> None:
        for obj in store.values():
            obj.evaluate()

    return SimpleNamespace(
        app=SimpleNamespace(version=(5, 0, 1)),
        data=SimpleNamespace(objects=Objects()),
        context=SimpleNamespace(
            scene=SimpleNamespace(objects=Objects()),
            evaluated_depsgraph_get=evaluated_depsgraph_get,
        ),
    )


@pytest.fixture
def make_session(
    monkeypatch: pytest.MonkeyPatch, mathutils: Any
) -> Callable[..., BlenderRuntimeSession]:
    """Return a factory: a session over the given objects."""

    def make(*objects: FakeTransformObject) -> BlenderRuntimeSession:
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", _bpy(objects))
        return BlenderRuntimeSession()

    return make


def _world(obj: FakeTransformObject) -> np.ndarray:
    return np.asarray(obj.matrix_world.rows)


def _turn(matrix: np.ndarray) -> np.ndarray:
    """The rotation of a world matrix, its axes' scale divided out."""
    linear = matrix[:3, :3]
    return linear / np.linalg.norm(linear, axis=0)


def _object(
    name: str, rotation_mode: str = "XYZ", **kwargs: Any
) -> FakeTransformObject:
    obj = FakeTransformObject(name, rotation_mode=rotation_mode, **kwargs)
    # A fresh axis-angle object stores no rotation about +Y, as Blender's does.
    obj.rotation_axis_angle = FakeVector((0.0, 0.0, 1.0, 0.0))
    return obj


# -- set_object_transform rotation (B1-01) -----------------------------------


@pytest.mark.parametrize("mode", ROTATION_MODES)
def test_set_object_transform_turns_the_object_whatever_its_rotation_mode(
    make_session: Callable[..., BlenderRuntimeSession], mode: str
) -> None:
    """rotation_euler is XYZ, converted to the object's own mode, which is kept.

    On a QUATERNION or AXIS_ANGLE object the write used to land in the unused
    rotation_euler and change nothing; a ZXY object read the angles in its own
    order and turned elsewhere.
    """
    robot = _object("Robot_Root", mode)
    session = make_session(robot)

    result = session.set_object_transform("Robot_Root", rotation_euler=list(TURN))

    assert np.allclose(_turn(_world(robot)), euler_matrix(TURN, "XYZ"), atol=1e-9)
    assert robot.rotation_mode == mode
    assert result["rotation_mode"] == mode
    assert result["rotation_euler"] == pytest.approx(TURN, abs=1e-9)
    assert np.allclose(
        quaternion_matrix(result["rotation_quaternion"]),
        euler_matrix(TURN, "XYZ"),
        atol=1e-9,
    )


@pytest.mark.parametrize("mode", ROTATION_MODES)
def test_set_object_transform_takes_a_quaternion(
    make_session: Callable[..., BlenderRuntimeSession], mode: str
) -> None:
    """A quaternion [w, x, y, z] is normalized and converted to the object's mode."""
    robot = _object("Robot_Root", mode)
    session = make_session(robot)
    wanted = matrix_quaternion(euler_matrix(TURN, "XYZ"))

    result = session.set_object_transform(
        "Robot_Root", rotation_quaternion=[2 * v for v in wanted]
    )

    assert np.allclose(_turn(_world(robot)), euler_matrix(TURN, "XYZ"), atol=1e-9)
    assert robot.rotation_mode == mode
    assert result["rotation_euler"] == pytest.approx(TURN, abs=1e-9)


def test_set_object_transform_keeps_a_quaternion_on_its_side(
    make_session: Callable[..., BlenderRuntimeSession],
) -> None:
    """Converted Euler angles keep the sign of the stored quaternion, so keys
    between the two stay a small turn instead of a flip to the opposite side."""
    robot = _object("Robot_Root", "QUATERNION")
    robot.rotation_quaternion = FakeVector((-1.0, 0.0, 0.0, 0.0))
    session = make_session(robot)

    session.set_object_transform("Robot_Root", rotation_euler=[0.0, 0.0, 0.2])

    assert np.allclose(_turn(_world(robot)), euler_matrix((0, 0, 0.2)), atol=1e-9)
    assert robot.rotation_quaternion[0] < 0


def test_set_object_transform_keeps_euler_angles_near_the_current_ones(
    make_session: Callable[..., BlenderRuntimeSession],
) -> None:
    """A converted Euler rotation does not jump by a whole turn."""
    robot = _object("Robot_Root", "ZXY", rotation_euler=(0.0, 0.0, 6.0))
    session = make_session(robot)

    session.set_object_transform(
        "Robot_Root",
        rotation_quaternion=list(matrix_quaternion(euler_matrix((0.0, 0.0, 6.1)))),
    )

    assert list(robot.rotation_euler) == pytest.approx([0.0, 0.0, 6.1], abs=1e-9)


@pytest.mark.parametrize(
    "arguments, message",
    [
        (
            {"rotation_euler": [0, 0, 1], "rotation_quaternion": [1, 0, 0, 0]},
            "not both",
        ),
        ({"rotation_quaternion": [0, 0, 0, 0]}, "zero length"),
        ({"rotation_euler": [0, float("nan"), 0]}, "finite"),
        ({"scale": [1, float("inf"), 1]}, "finite"),
    ],
)
def test_set_object_transform_writes_nothing_on_a_refused_call(
    make_session: Callable[..., BlenderRuntimeSession], arguments: dict, message: str
) -> None:
    robot = _object("Robot_Root", "QUATERNION", location=(1.0, 2.0, 3.0))
    session = make_session(robot)

    with pytest.raises(ValueError, match=message):
        session.set_object_transform(
            "Robot_Root", location=[9.0, 9.0, 9.0], **arguments
        )

    assert list(robot.location) == [1.0, 2.0, 3.0]


# -- get_object_info local and world transforms (B1-06) ----------------------


def _robot() -> tuple:
    """A glTF-like rig: quaternion nodes, parent inverses left at identity."""
    root = _object("Robot_Root", "QUATERNION", location=(1.0, 2.0, 0.0))
    root.rotation_quaternion = FakeVector(
        (math.cos(math.pi / 12), 0.0, 0.0, math.sin(math.pi / 12))
    )
    arm = _object("Link_Arm", "QUATERNION", location=(0.3, 0.0, 1.0), parent=root)
    arm.matrix_parent_inverse = FakeMatrix.Identity(4)
    arm.rotation_quaternion = FakeVector(
        (math.cos(math.pi / 8), 0.0, 0.0, math.sin(math.pi / 8))
    )
    return root, arm


def test_get_object_info_reports_the_own_and_the_world_transform(
    make_session: Callable[..., BlenderRuntimeSession],
) -> None:
    """location is parent-relative; world_location and matrix_world say where it is.

    It reported Link_Arm's parent-relative location as its world location, and
    rotation_euler [0, 0, 0] for every quaternion node.
    """
    root, arm = _robot()
    session = make_session(root, arm)

    info = session.get_object_info("Link_Arm")

    turned = (math.cos(math.pi / 6) * 0.3, math.sin(math.pi / 6) * 0.3)
    assert info["location"] == pytest.approx([0.3, 0.0, 1.0])
    assert info["world_location"] == pytest.approx(
        [1.0 + turned[0], 2.0 + turned[1], 1.0]
    )
    assert info["rotation_mode"] == "QUATERNION"
    assert info["rotation_euler"] == pytest.approx([0.0, 0.0, math.pi / 4])
    assert info["rotation_quaternion"] == pytest.approx(
        [math.cos(math.pi / 8), 0.0, 0.0, math.sin(math.pi / 8)]
    )
    assert info["rotation_in_mode"] == pytest.approx(list(arm.rotation_quaternion))
    assert info["world_rotation_euler"] == pytest.approx(
        [0.0, 0.0, math.pi / 6 + math.pi / 4]
    )
    assert info["world_rotation_quaternion"] == pytest.approx(
        [math.cos(5 * math.pi / 24), 0.0, 0.0, math.sin(5 * math.pi / 24)]
    )
    assert info["world_scale"] == pytest.approx([1.0, 1.0, 1.0])
    assert np.allclose(info["matrix_world"], _world(arm), atol=1e-12)


@pytest.mark.parametrize("mode", ["ZXY", "AXIS_ANGLE"])
def test_get_object_info_reads_the_rotation_in_the_objects_own_mode(
    make_session: Callable[..., BlenderRuntimeSession], mode: str
) -> None:
    obj = _object("Gear", mode)
    session = make_session(obj)
    session.set_object_transform("Gear", rotation_euler=list(TURN))

    info = session.get_object_info("Gear")

    assert info["rotation_euler"] == pytest.approx(TURN, abs=1e-9)
    stored = (
        list(obj.rotation_axis_angle)
        if mode == "AXIS_ANGLE"
        else list(obj.rotation_euler)
    )
    assert info["rotation_in_mode"] == pytest.approx(stored)
    assert info["world_rotation_euler"] == pytest.approx(TURN, abs=1e-9)


def test_get_object_info_gives_a_sheared_child_the_rotation_of_its_axes(
    make_session: Callable[..., BlenderRuntimeSession],
) -> None:
    """Under a parent scaled 0.5 on Z, a tilted child's world matrix shears.

    Blender's decompose() and to_euler() read the rotation from some of the
    entries and were 0.19 and 0.15 rad off on such a glTF link; the nearest
    rotation is the parent's rotation times the child's.
    """
    base = _object(
        "Link_Base", rotation_euler=(0.0, 0.0, 5 * math.pi / 12), scale=(1.0, 1.0, 0.5)
    )
    tilt = (0.349, 0.1745, 0.0)
    arm = _object("Link_Arm", "QUATERNION", parent=base)
    arm.matrix_parent_inverse = FakeMatrix.Identity(4)
    arm.rotation_quaternion = FakeVector(matrix_quaternion(euler_matrix(tilt)))
    session = make_session(base, arm)

    info = session.get_object_info("Link_Arm")

    expected = euler_matrix((0.0, 0.0, 5 * math.pi / 12)) @ euler_matrix(tilt)
    assert np.allclose(
        quaternion_matrix(info["world_rotation_quaternion"]), expected, atol=1e-9
    )
    assert np.allclose(euler_matrix(info["world_rotation_euler"]), expected, atol=1e-9)
    # A plain read of the matrix entries gives another rotation.
    assert not np.allclose(_turn(_world(arm)), expected, atol=1e-3)


def test_get_object_info_decomposes_a_mirrored_world_matrix_as_blender_does(
    make_session: Callable[..., BlenderRuntimeSession],
) -> None:
    """decompose() gives a mirror a scale negative on every axis; LocRotScale of
    the reported parts rebuilds matrix_world."""
    obj = _object("Mirror", scale=(-1.0, 2.0, 2.0), rotation_euler=(0.0, 0.0, 0.5))
    session = make_session(obj)

    info = session.get_object_info("Mirror")

    assert info["world_scale"] == pytest.approx([-1.0, -2.0, -2.0])
    rebuilt = quaternion_matrix(info["world_rotation_quaternion"]) * np.asarray(
        info["world_scale"]
    )
    assert np.allclose(rebuilt, _world(obj)[:3, :3], atol=1e-9)


# -- Reads of matrix_world after an unevaluated write (B1-16) ----------------


def test_reads_see_a_move_written_since_the_last_evaluation(
    make_session: Callable[..., BlenderRuntimeSession],
) -> None:
    """In embedded mode distance and bounds read matrix_world before the move."""
    cube = _object("Cube", stored=True)
    anchor = _object("Anchor", stored=True)
    session = make_session(cube, anchor)
    cube.location = FakeVector((10.0, 0.0, 0.0))  # an earlier call's write

    distance = session.get_distance_between("Cube", "Anchor")
    bounds = session.is_object_within_bounds("Cube", [9, -1, -1], [11, 1, 1])
    info = session.get_object_info("Cube")

    assert distance["location_a"] == pytest.approx([10.0, 0.0, 0.0])
    assert distance["distance"] == pytest.approx(10.0)
    assert bounds["within_bounds"] is True
    assert info["world_location"] == pytest.approx([10.0, 0.0, 0.0])


def test_clear_object_parent_keeps_the_transform_written_since_the_last_evaluation(
    make_session: Callable[..., BlenderRuntimeSession],
) -> None:
    """It kept the child where its parent's stale matrix put it."""
    parent = _object("P", location=(0.0, 0.0, 5.0), stored=True)
    child = _object("C", location=(1.0, 0.0, 0.0), parent=parent, stored=True)
    child.matrix_parent_inverse = FakeMatrix.Identity(4)
    child.evaluate()
    session = make_session(parent, child)
    parent.location = FakeVector((0.0, 0.0, 8.0))  # an earlier call's write

    session.clear_object_parent("C", keep_transform=True)
    child.evaluate()

    assert child.parent is None
    assert list(child.matrix_world.translation) == pytest.approx([1.0, 0.0, 8.0])


# -- set_object_parent (B1-07) -----------------------------------------------


def _parent_and_child() -> tuple:
    parent = _object(
        "P",
        location=(5.0, 0.0, 0.0),
        rotation_euler=(0, 0, math.pi / 2),
        scale=(2, 2, 2),
    )
    child = _object("C", location=(1.0, 1.0, 1.0))
    return parent, child


def test_set_object_parent_keeps_the_childs_world_transform(
    make_session: Callable[..., BlenderRuntimeSession],
) -> None:
    """C at (1, 1, 1) under P (90 degrees, scale 2) jumped to (3, 2, 2) with scale 2."""
    parent, child = _parent_and_child()
    before = _world(child)
    session = make_session(parent, child)

    result = session.set_object_parent("C", "P")

    assert child.parent is parent
    assert np.allclose(_world(child), before, atol=1e-9)
    # As Ctrl+P > Object: the child's own values stay; the inverse cancels P.
    assert list(child.location) == pytest.approx([1.0, 1.0, 1.0])
    assert np.allclose(
        np.asarray(child.matrix_parent_inverse.rows),
        np.linalg.inv(_world(parent)),
        atol=1e-9,
    )
    assert result["keep_transform"] is True
    assert result["previous_parent"] is None
    assert result["world_location"] == pytest.approx([1.0, 1.0, 1.0])


def test_reparenting_does_not_compound(
    make_session: Callable[..., BlenderRuntimeSession],
) -> None:
    """Each re-parenting moved the child by the new parent's transform again."""
    parent, child = _parent_and_child()
    other = _object(
        "Q", location=(0.0, 10.0, 0.0), rotation_euler=(0.3, 0, 0), scale=(1, 1, 3)
    )
    before = _world(child)
    session = make_session(parent, child, other)

    for target in ("P", "Q", "P", "P"):
        result = session.set_object_parent("C", target)
        assert np.allclose(_world(child), before, atol=1e-9), target
    assert result["previous_parent"] == "P"


def test_set_object_parent_leaves_the_constraints_out_of_the_kept_transform(
    make_session: Callable[..., BlenderRuntimeSession],
) -> None:
    """The constraints apply on top of the kept transform once, not twice."""
    parent, child = _parent_and_child()
    child.constraints.append(FakeConstraint("Lift", "LIMIT_LOCATION", _lift))
    first = _object("First", location=(0.0, -4.0, 2.0), rotation_euler=(0, 0, 1.0))
    session = make_session(parent, child, first)
    session.set_object_parent("C", "First")
    before = _world(child)

    session.set_object_parent("C", "P")
    session.clear_object_parent("C", keep_transform=True)

    assert np.allclose(_world(child), before, atol=1e-9)


def _lift(matrix: np.ndarray) -> np.ndarray:
    lifted = matrix.copy()
    lifted[2, 3] += 1.0
    return lifted


def test_set_object_parent_without_keep_transform_moves_the_child_with_the_parent(
    make_session: Callable[..., BlenderRuntimeSession],
) -> None:
    parent, child = _parent_and_child()
    child.matrix_parent_inverse = FakeMatrix.Translation((7.0, 7.0, 7.0))  # stale
    session = make_session(parent, child)

    result = session.set_object_parent("C", "P", keep_transform=False)

    assert list(child.location) == pytest.approx([1.0, 1.0, 1.0])
    assert list(child.matrix_world.translation) == pytest.approx([3.0, 2.0, 2.0])
    assert np.allclose(np.asarray(child.matrix_parent_inverse.rows), np.eye(4))
    assert result["keep_transform"] is False


@pytest.mark.parametrize(
    "child_name, parent_name, message",
    [
        ("P", "C", "loop"),
        ("P", "G", "loop"),
        ("C", "C", "its own parent"),
    ],
)
def test_set_object_parent_refuses_a_parent_loop(
    make_session: Callable[..., BlenderRuntimeSession],
    child_name: str,
    parent_name: str,
    message: str,
) -> None:
    """Blender refuses such a parent without an error; the tool replied success."""
    parent, child = _parent_and_child()
    child.parent = parent
    grandchild = _object("G", parent=child)
    session = make_session(parent, child, grandchild)
    parents: List[Any] = [obj.parent for obj in (parent, child, grandchild)]

    with pytest.raises(ValueError, match=message):
        session.set_object_parent(child_name, parent_name)

    assert [obj.parent for obj in (parent, child, grandchild)] == parents


@pytest.mark.parametrize(
    "rotation",
    [
        {"rotation_euler": [0, 0, 1], "rotation_quaternion": [1, 0, 0, 0]},
        {"rotation_quaternion": [0, 0, 0, 0]},
        {"rotation_quaternion": [float("nan"), 0, 0, 1]},
        {"rotation_quaternion": [1, 0, 0]},
    ],
)
def test_the_transform_request_refuses_a_rotation_it_cannot_apply(
    rotation: dict,
) -> None:
    from pydantic import ValidationError

    from simul.mcp.schemas.blender import BlenderSetTransformRequest

    assert BlenderSetTransformRequest(
        object_name="Cube", rotation_quaternion=[1, 0, 0, 0]
    ).rotation_quaternion == [1.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValidationError):
        BlenderSetTransformRequest(object_name="Cube", **rotation)


def test_fake_euler_round_trips_through_a_quaternion() -> None:
    """The stand-in's conversions agree, so the tests above check the runtime."""
    quaternion = FakeEuler(TURN, "ZXY").to_quaternion()
    assert np.allclose(
        quaternion.to_matrix().rows, euler_matrix(TURN, "ZXY"), atol=1e-12
    )
    axis, angle = quaternion.to_axis_angle()
    assert np.allclose(
        quaternion_matrix(
            (math.cos(angle / 2), *(np.asarray(list(axis)) * math.sin(angle / 2)))
        ),
        euler_matrix(TURN, "ZXY"),
        atol=1e-12,
    )
