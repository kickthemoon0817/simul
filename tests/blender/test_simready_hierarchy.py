"""setup_simready_hierarchy keeps world transforms and nesting, and changes nothing on failure.

The stand-in objects compose ``matrix_world`` as Blender does, parent world @
(bone matrix @) parent inverse @ basis, so a re-parent that moved an object
shows up here. An object marked ``excluded`` is one Blender's depsgraph leaves
out: its ``matrix_world`` is the stale one of its last evaluation and its
evaluated copy reports ``is_evaluated`` False, as Blender 5.0.1 does for an
object in an excluded collection.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from simul.adapters import blender_runtime
from tests.fakes import FakeMatrix


class Obj:
    """A Blender object with Blender's world-matrix composition and custom properties."""

    def __init__(
        self,
        name: str,
        kind: str = "MESH",
        parent: Optional["Obj"] = None,
        basis: Optional[FakeMatrix] = None,
    ) -> None:
        self.name = name
        self.type = kind
        self.parent = parent
        self.parent_type = "OBJECT"
        self.parent_bone = ""
        self.bone_matrix = FakeMatrix.Identity(4)
        self.matrix_basis = basis or FakeMatrix.Identity(4)
        self.matrix_parent_inverse = FakeMatrix.Identity(4)
        self.props: Dict[str, Any] = {}
        self.empty_display_type = ""
        self.library: Optional[str] = None
        self.constraints: List[Any] = []
        self.excluded = False
        self.stale_world: Optional[FakeMatrix] = None

    def composed_world(self) -> FakeMatrix:
        """Where Blender places the object once it evaluates it."""
        local = self.matrix_parent_inverse @ self.matrix_basis
        if self.parent is None:
            return local
        if self.parent_type == "BONE":
            local = self.bone_matrix @ local
        return self.parent.composed_world() @ local

    @property
    def matrix_world(self) -> FakeMatrix:
        if self.excluded:
            return self.stale_world or FakeMatrix.Identity(4)
        return self.composed_world()

    def evaluated_get(self, depsgraph: Any) -> Any:
        if self.excluded:
            return SimpleNamespace(is_evaluated=False, matrix_world=self.matrix_world)
        return SimpleNamespace(is_evaluated=True, matrix_world=self.matrix_world)

    @property
    def children(self) -> List["Obj"]:
        return [o for o in _SCENE if o.parent is self]

    def __setitem__(self, key: str, value: Any) -> None:
        self.props[key] = value

    def __getitem__(self, key: str) -> Any:
        return self.props[key]

    def keys(self) -> List[str]:
        return list(self.props)


_SCENE: List[Obj] = []


class Objects(list):
    def get(self, name: str) -> Optional[Obj]:
        return next((o for o in self if o.name == name), None)

    def new(self, name: str, data: Any) -> Obj:
        obj = Obj(name, "EMPTY" if data is None else "MESH")
        self.append(obj)
        return obj

    def remove(self, obj: Obj) -> None:
        super().remove(obj)

    def link(self, obj: Obj) -> None:
        if obj not in self:
            self.append(obj)


def _translation(x: float, y: float, z: float) -> FakeMatrix:
    return FakeMatrix.Translation((x, y, z))


def _rotation_z(degrees: float) -> FakeMatrix:
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return FakeMatrix([(c, -s, 0, 0), (s, c, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)])


def _world(obj: Obj) -> List[float]:
    return [round(v, 6) for v in obj.composed_world().translation]


def _rows(matrix: FakeMatrix) -> np.ndarray:
    return np.asarray(matrix.rows)


@pytest.fixture
def scene(monkeypatch: pytest.MonkeyPatch, mathutils: Any) -> Any:
    _SCENE.clear()
    objects = Objects(_SCENE)

    base = Obj("base_link", basis=_translation(1, 0, 0))
    arm = Obj("arm_link", parent=base, basis=_translation(0, 0, 1))
    rig = Obj("rig", "EMPTY", basis=FakeMatrix(np.diag([2.0, 2.0, 2.0, 1.0])) @ _translation(5, 5, 0))
    wheel = Obj("wheel", parent=rig)
    # Ctrl+P "Keep Transform": the wheel stays at the world origin under the rig.
    wheel.matrix_parent_inverse = rig.matrix_world.inverted()
    cube = Obj("Cube")
    objects.extend([base, arm, rig, wheel, cube])
    context = SimpleNamespace(
        scene=SimpleNamespace(objects=objects, collection=SimpleNamespace(objects=objects)),
        view_layer=SimpleNamespace(update=lambda: None),
        evaluated_depsgraph_get=lambda: "depsgraph",
        mode="OBJECT",
    )
    fake = SimpleNamespace(
        data=SimpleNamespace(objects=objects),
        context=context,
        app=SimpleNamespace(version=(5, 0, 1), version_string="5.0.1", background=True),
    )
    monkeypatch.setattr(blender_runtime, "bpy", fake)
    monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
    session = blender_runtime.BlenderRuntimeSession()

    def add(obj: Obj) -> Obj:
        objects.append(obj)
        return obj

    yield SimpleNamespace(session=session, objects=objects, get=objects.get, add=add, context=context)
    _SCENE.clear()


def _code(excinfo: Any) -> str:
    return excinfo.value.details["code"]


def test_nested_links_keep_their_parent_and_nothing_moves(scene: Any) -> None:
    before = {o.name: _world(o) for o in scene.objects}

    result = scene.session.setup_simready_hierarchy(
        "robot", ["base_link", "arm_link"], semantic={"semantic_class": "robot"}
    )

    assert scene.get("base_link").parent is scene.get("robot")
    assert scene.get("arm_link").parent is scene.get("base_link")
    assert {o.name: _world(o) for o in scene.objects if o.name != "robot"} == before
    assert _world(scene.get("arm_link")) == [1.0, 0.0, 1.0]
    assert result["reparented"] == ["base_link"]
    assert result["kept_parent"] == [{"object_name": "arm_link", "parent": "base_link"}]
    assert result["prim_paths"] == {
        "base_link": "/robot/base_link",
        "arm_link": "/robot/base_link/arm_link",
    }
    assert scene.get("robot")["simready_semantic_class"] == "robot"


def test_an_object_under_a_transformed_parent_keeps_its_world_transform(scene: Any) -> None:
    assert _world(scene.get("wheel")) == [0.0, 0.0, 0.0]

    scene.session.setup_simready_hierarchy("cart", ["wheel"])

    wheel = scene.get("wheel")
    assert wheel.parent is scene.get("cart")
    assert _world(wheel) == [0.0, 0.0, 0.0]
    # The root is at the origin, so the transform is the object's own: clean transforms.
    assert np.allclose(_rows(wheel.matrix_parent_inverse), np.eye(4))


def test_an_object_blender_does_not_evaluate_keeps_its_real_place(scene: Any) -> None:
    """M2: an excluded collection's object has a stale matrix_world; its parent moved since."""
    lift = scene.add(Obj("lift", "EMPTY", basis=_translation(0, 0, 5)))
    crate = scene.add(Obj("crate_part", parent=lift, basis=_translation(0, 2, 0)))
    crate.excluded = True
    crate.stale_world = _translation(0, 0, 0)  # evaluated before the lift moved, then excluded

    result = scene.session.setup_simready_hierarchy("props", ["crate_part"])

    assert result["reparented"] == ["crate_part"]
    assert _world(crate) == [0.0, 2.0, 5.0]


def test_a_child_skewed_by_its_parent_keeps_its_shape(scene: Any) -> None:
    """M3: location, rotation and scale cannot express the skew; the parent inverse keeps it."""
    stretched = scene.add(Obj("stretched", "EMPTY", basis=FakeMatrix(np.diag([1.0, 3.0, 1.0, 1.0]))))
    sheared = scene.add(
        Obj("sheared_part", parent=stretched, basis=_translation(1, 0, 0) @ _rotation_z(45))
    )
    before = _rows(sheared.composed_world())

    result = scene.session.setup_simready_hierarchy("stretch_asset", ["sheared_part"])

    assert np.allclose(_rows(sheared.composed_world()), before)
    assert not np.allclose(_rows(sheared.matrix_parent_inverse), np.eye(4))
    assert any("skews it" in w for w in result["warnings"])


def test_a_bone_parent_becomes_an_object_parent_in_place(scene: Any) -> None:
    rig = scene.add(Obj("armature", "ARMATURE", basis=_translation(0, 0, 1)))
    bolt = scene.add(Obj("bolt", parent=rig, basis=_translation(0.3, 0, 0)))
    bolt.parent_type = "BONE"
    bolt.parent_bone = "b1"
    bolt.bone_matrix = _translation(0, 2, 0)  # the bone's tail
    before = _rows(bolt.composed_world())

    scene.session.setup_simready_hierarchy("fixtures", ["bolt"])

    assert (bolt.parent_type, bolt.parent_bone) == ("OBJECT", "")
    assert np.allclose(_rows(bolt.composed_world()), before)


@pytest.mark.parametrize("muted", [False, True])
def test_a_constrained_object_keeps_its_basis_and_parent_space(scene: Any, muted: bool) -> None:
    """A Limit Location in local space sees the same values after re-parenting.

    Blender's local space for an object constraint is relative to parent world @
    parent inverse: kept, with the basis untouched, the constraint gives the same pose.
    """
    arm = scene.get("arm_link")
    arm.constraints = [SimpleNamespace(mute=muted, influence=1.0)]
    basis = _rows(arm.matrix_basis)
    space = _rows(arm.parent.matrix_world) @ _rows(arm.matrix_parent_inverse)

    scene.session.setup_simready_hierarchy("asset", ["arm_link"])

    assert arm.parent is scene.get("asset")
    assert _world(arm) == [1.0, 0.0, 1.0]
    if muted:  # nothing acts on it: the transform moves into its own basis
        assert _rows(arm.matrix_basis) == pytest.approx(space @ basis)
        assert _rows(arm.matrix_parent_inverse) == pytest.approx(np.eye(4))
    else:
        assert _rows(arm.matrix_basis) == pytest.approx(basis)
        assert _rows(arm.parent.matrix_world) @ _rows(arm.matrix_parent_inverse) == pytest.approx(
            space
        )


def test_a_constrained_bone_child_is_refused(scene: Any) -> None:
    rig = scene.add(Obj("armature", "ARMATURE"))
    bolt = scene.add(Obj("bolt", parent=rig))
    bolt.parent_type = "BONE"
    bolt.constraints = [SimpleNamespace(mute=False)]

    with pytest.raises(blender_runtime.ValidationError) as excinfo:
        scene.session.setup_simready_hierarchy("fixtures", ["bolt"])

    assert _code(excinfo) == "UnsupportedParent"
    assert scene.get("fixtures") is None and bolt.parent is rig


def test_a_missing_child_changes_nothing(scene: Any) -> None:
    names = [o.name for o in scene.objects]

    with pytest.raises(
        blender_runtime.NotFoundError,
        match="Objects not found: does_not_exist, ghost; nothing was changed",
    ) as excinfo:
        scene.session.setup_simready_hierarchy("crate_asset", ["Cube", "does_not_exist", "ghost"])

    assert _code(excinfo) == "object_not_found"
    assert excinfo.value.details["objects"] == ["does_not_exist", "ghost"]
    assert [o.name for o in scene.objects] == names
    assert scene.get("Cube").parent is None
    # A retry after fixing the list gets the requested name, not crate_asset.001.
    result = scene.session.setup_simready_hierarchy("crate_asset", ["Cube"])
    assert result["root_name"] == "crate_asset"


def test_an_existing_root_is_refused_unless_reused(scene: Any) -> None:
    scene.session.setup_simready_hierarchy("robot", ["base_link"])

    with pytest.raises(ValueError, match="already exists; pass reuse_root=true") as excinfo:
        scene.session.setup_simready_hierarchy("robot", ["Cube"])
    assert _code(excinfo) == "RootExists"
    assert scene.get("Cube").parent is None

    result = scene.session.setup_simready_hierarchy("robot", ["Cube"], reuse_root=True)

    assert result["root_created"] is False
    assert scene.get("Cube").parent is scene.get("robot")
    assert len([o for o in scene.objects if o.name.startswith("robot")]) == 1


def test_a_reused_root_with_zero_scale_is_refused(scene: Any) -> None:
    scene.add(Obj("flat_root", "EMPTY", basis=FakeMatrix(np.diag([0.0, 0.0, 0.0, 1.0]))))

    with pytest.raises(blender_runtime.ValidationError) as excinfo:
        scene.session.setup_simready_hierarchy("flat_root", ["Cube"], reuse_root=True)

    assert _code(excinfo) == "RootNotInvertible"
    assert scene.get("Cube").parent is None


def test_a_root_that_is_not_an_empty_is_refused(scene: Any) -> None:
    scene.get("Cube").name = "cube"

    with pytest.raises(ValueError, match="is a mesh, not an empty"):
        scene.session.setup_simready_hierarchy("cube", ["base_link"], reuse_root=True)
    assert scene.get("base_link").parent is None


def test_a_root_cannot_go_under_its_own_child(scene: Any) -> None:
    scene.session.setup_simready_hierarchy("robot", ["base_link"])
    scene.get("robot").parent = scene.get("rig")

    with pytest.raises(ValueError, match="is above 'robot'"):
        scene.session.setup_simready_hierarchy("robot", ["rig"], reuse_root=True)
    assert scene.get("rig").parent is None


def test_invalid_semantic_labels_are_refused_before_anything_changes(scene: Any) -> None:
    with pytest.raises(ValueError, match="semantic_qcode") as excinfo:
        scene.session.setup_simready_hierarchy(
            "crate_asset", ["Cube"], semantic={"semantic_qcode": "not-a-qcode"}
        )

    assert _code(excinfo) == "InvalidMetadata"
    assert scene.get("crate_asset") is None
    assert scene.get("Cube").parent is None


def test_a_linked_object_is_refused(scene: Any) -> None:
    """Its parent would be reset when the file is reopened."""
    scene.get("Cube").library = "//library_parts.blend"

    with pytest.raises(blender_runtime.ValidationError) as excinfo:
        scene.session.setup_simready_hierarchy("crate_asset", ["base_link", "Cube"])

    assert _code(excinfo) == "LinkedData"
    assert excinfo.value.details["objects"] == ["Cube"]
    assert scene.get("crate_asset") is None and scene.get("base_link").parent is None


def test_edit_mode_is_refused(scene: Any) -> None:
    """Edit Mode's undo cannot step back over an object change."""
    scene.context.mode = "EDIT_MESH"

    with pytest.raises(blender_runtime.ValidationError) as excinfo:
        scene.session.setup_simready_hierarchy("crate_asset", ["Cube"])

    assert _code(excinfo) == "ObjectModeRequired"
    assert scene.get("crate_asset") is None


def test_camel_case_root_names_are_accepted(scene: Any) -> None:
    assert scene.session.setup_simready_hierarchy("toolBox", ["Cube"])["root_name"] == "toolBox"

    with pytest.raises(blender_runtime.ValidationError) as excinfo:
        scene.session.setup_simready_hierarchy("Robot Root", ["base_link"])
    assert _code(excinfo) == "InvalidRootName"


def test_two_objects_that_become_one_prim_name_are_reported(scene: Any) -> None:
    scene.add(Obj("arm.L"))
    scene.add(Obj("arm_L"))

    result = scene.session.setup_simready_hierarchy("rig_asset", ["arm.L", "arm_L"])

    assert any("both become the prim /rig_asset/arm_L" in w for w in result["warnings"])


def test_a_failure_part_way_restores_the_parents_and_removes_the_root(scene: Any) -> None:
    class Stuck(Obj):
        def __setattr__(self, key: str, value: Any) -> None:
            if key == "parent" and getattr(self, "_armed", False):
                raise RuntimeError("cannot parent a linked object")
            super().__setattr__(key, value)

    stuck = Stuck("linked_part")
    scene.objects.append(stuck)
    stuck._armed = True
    cube = scene.get("Cube")
    cube_basis = cube.matrix_basis.rows

    with pytest.raises(RuntimeError, match="cannot parent"):
        scene.session.setup_simready_hierarchy("crate_asset", ["Cube", "linked_part"])

    assert scene.get("crate_asset") is None
    assert cube.parent is None
    assert cube.matrix_basis.rows == cube_basis


def test_an_empty_child_list_is_refused(scene: Any) -> None:
    with pytest.raises(ValueError, match="child_names is empty"):
        scene.session.setup_simready_hierarchy("crate_asset", [])
    assert scene.get("crate_asset") is None
