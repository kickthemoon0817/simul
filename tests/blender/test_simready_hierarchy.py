"""setup_simready_hierarchy keeps world transforms and nesting, and changes nothing on failure.

The stand-in objects compute ``matrix_world`` as Blender does, parent world @
parent inverse @ basis, so a re-parent that moved an object shows up here.
"""

from __future__ import annotations

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
        self.matrix_basis = basis or FakeMatrix.Identity(4)
        self.matrix_parent_inverse = FakeMatrix.Identity(4)
        self.props: Dict[str, Any] = {}
        self.empty_display_type = ""

    @property
    def matrix_world(self) -> FakeMatrix:
        local = self.matrix_parent_inverse @ self.matrix_basis
        return self.parent.matrix_world @ local if self.parent is not None else local

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


def _world(obj: Obj) -> List[float]:
    return [round(v, 6) for v in obj.matrix_world.translation]


@pytest.fixture
def scene(monkeypatch: pytest.MonkeyPatch) -> Any:
    _SCENE.clear()
    objects = Objects(_SCENE)
    _SCENE_REF = objects  # the same list: data.objects and scene.objects

    base = Obj("base_link", basis=_translation(1, 0, 0))
    arm = Obj("arm_link", parent=base, basis=_translation(0, 0, 1))
    rig = Obj("rig", "EMPTY", basis=FakeMatrix(np.diag([2.0, 2.0, 2.0, 1.0])) @ _translation(5, 5, 0))
    wheel = Obj("wheel", parent=rig)
    # Ctrl+P "Keep Transform": the wheel stays at the world origin under the rig.
    wheel.matrix_parent_inverse = rig.matrix_world.inverted()
    cube = Obj("Cube")
    objects.extend([base, arm, rig, wheel, cube])
    fake = SimpleNamespace(
        data=SimpleNamespace(objects=_SCENE_REF),
        context=SimpleNamespace(
            scene=SimpleNamespace(objects=_SCENE_REF, collection=SimpleNamespace(objects=_SCENE_REF)),
            view_layer=SimpleNamespace(update=lambda: None),
        ),
        app=SimpleNamespace(version=(5, 0, 1), version_string="5.0.1", background=True),
    )
    monkeypatch.setattr(blender_runtime, "bpy", fake)
    monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
    session = blender_runtime.BlenderRuntimeSession()
    yield SimpleNamespace(session=session, objects=objects, get=objects.get)
    _SCENE.clear()


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
    # The root is at the origin, so the world transform is the basis: clean transforms.
    assert np.allclose(np.asarray(wheel.matrix_parent_inverse.rows), np.eye(4))


def test_a_missing_child_changes_nothing(scene: Any) -> None:
    names = [o.name for o in scene.objects]

    with pytest.raises(ValueError, match="Objects not found: does_not_exist; nothing was changed"):
        scene.session.setup_simready_hierarchy("crate_asset", ["Cube", "does_not_exist"])

    assert [o.name for o in scene.objects] == names
    assert scene.get("Cube").parent is None
    # A retry after fixing the list gets the requested name, not crate_asset.001.
    result = scene.session.setup_simready_hierarchy("crate_asset", ["Cube"])
    assert result["root_name"] == "crate_asset"


def test_an_existing_root_is_refused_unless_reused(scene: Any) -> None:
    scene.session.setup_simready_hierarchy("robot", ["base_link"])

    with pytest.raises(ValueError, match="already exists; pass reuse_root=true"):
        scene.session.setup_simready_hierarchy("robot", ["Cube"])
    assert scene.get("Cube").parent is None

    result = scene.session.setup_simready_hierarchy("robot", ["Cube"], reuse_root=True)

    assert result["root_created"] is False
    assert scene.get("Cube").parent is scene.get("robot")
    assert len([o for o in scene.objects if o.name.startswith("robot")]) == 1


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
    with pytest.raises(ValueError, match="semantic_qcode"):
        scene.session.setup_simready_hierarchy(
            "crate_asset", ["Cube"], semantic={"semantic_qcode": "not-a-qcode"}
        )

    assert scene.get("crate_asset") is None
    assert scene.get("Cube").parent is None


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
