"""Object creation and deletion, modes, listing pages, bounds and scene summaries.

The audit of the Blender tools found, at 6,006 objects and in ordinary scenes:

- create_blender_object failed with AttributeError, leaving an unnamed cube,
  when the active collection was disabled in viewports, and left the window in
  Edit Mode under Blender's "Enter Edit Mode" preference;
- delete_blender_object removed the object from every scene, unlike Blender's
  Delete, and left a deleted rigid body's cache to other bodies;
- the listing tools reported a count the result budget then cut, had no
  offset, set truncated when exactly max_items existed, never named a
  collection and matched object types case-sensitively;
- check_blender_object_bounds tested the origin only and took min > max;
- summarize_blender_scene listed every collection in the file and
  get_blender_file_info counted every object in the file.

The fake below models what those depend on: collections and their view-layer
flags, scenes, the add operators (the 3D View's context.active_object is None
for an object hidden in it, as on Blender 5.0.1), modes and Blender's Delete.
"""

from __future__ import annotations

import json
import math
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import pytest

from simul.adapters import blender_runtime
from simul.adapters.blender_runtime import BlenderRuntimeSession
from simul.mcp.result_budget import DEFAULT_RESULT_BUDGET_BYTES
from tests.fakes import (
    BLENDER_5_OBJECT_TYPES,
    FakeCollection,
    FakeMatrix,
    FakePropArray,
    FakeTransformObject,
    FakeVector,
    fake_bpy_types,
)

MODES = (
    "OBJECT",
    "EDIT",
    "POSE",
    "SCULPT",
    "VERTEX_PAINT",
    "WEIGHT_PAINT",
    "TEXTURE_PAINT",
)


class FakeData:
    """Object data (a mesh, light or camera): its users are the objects that use it."""

    def __init__(self, world: "World", name: str) -> None:
        self._world = world
        self.name = name

    @property
    def users(self) -> int:
        return sum(obj.data is self for obj in self._world.objects.values())


class FakeObject:
    """A Blender object: its collections, mode, transform and visibility."""

    def __init__(self, world: "World", name: str, object_type: str = "MESH") -> None:
        self._world = world
        self._name = name
        self.type = object_type
        self.data: Any = FakeData(world, name) if object_type != "EMPTY" else None
        self.location: Any = FakeVector((0.0, 0.0, 0.0))
        self.scale: Any = FakeVector((1.0, 1.0, 1.0))
        self.users_collection: List[FakeCollection] = []
        self.mode = "OBJECT"
        self.rigid_body: Any = None
        self.library: Any = None
        self.hide = False
        self.refuse_name: Optional[str] = None

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        if value == self.refuse_name:
            raise ValueError(f"cannot name an object {value!r}")
        self._world.objects.pop(self._name, None)
        self._name = self._world.unique(value)
        self._world.objects[self._name] = self

    def visible_get(self) -> bool:
        """Hidden by its own flag or by every collection holding it being hidden."""
        return not self.hide and any(
            not self._world.collection_hidden(c) for c in self.users_collection
        )

    def as_pointer(self) -> int:
        return id(self)


class FakeLayer:
    """A layer collection: a collection's exclude and hide (eye) flags in the view layer."""

    def __init__(self, collection: FakeCollection) -> None:
        self.collection = collection
        self.children: List["FakeLayer"] = []
        self.exclude = False
        self.hide_viewport = False

    @property
    def name(self) -> str:
        return self.collection.name


class FakeStore:
    """A bpy collection of IDs by name: get, iteration, len, keys."""

    def __init__(self, items: Callable[[], List[Any]]) -> None:
        self._items = items

    def get(self, name: str) -> Any:
        return next((item for item in self._items() if item.name == name), None)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._items())

    def __len__(self) -> int:
        return len(self._items())

    def keys(self) -> List[str]:
        return [item.name for item in self._items()]


class FakeScene:
    """A scene: its own collection, and the objects its collections hold."""

    def __init__(self, world: "World", name: str) -> None:
        self.name = name
        self.collection = FakeCollection("Scene Collection")
        self.objects = FakeStore(
            lambda: [
                obj
                for obj in world.objects.values()
                if any(self.holds(c) for c in obj.users_collection)
            ]
        )

    def holds(self, collection: FakeCollection) -> bool:
        return collection is self.collection or any(
            collection is child for child in self.collection.children_recursive
        )


def _operator(
    function: Callable[..., Any], properties: Sequence[str], **enums: Sequence[str]
) -> Callable[..., Any]:
    """Give a fake operator Blender's get_rna_type().properties."""
    props: Dict[str, Any] = {name: SimpleNamespace() for name in properties}
    for name, items in enums.items():
        props[name] = SimpleNamespace(
            enum_items=[SimpleNamespace(identifier=item) for item in items]
        )
    function.get_rna_type = lambda: SimpleNamespace(properties=props)  # type: ignore[attr-defined]
    return function


class World:
    """The fake Blender: data, context and the operators the object tools run."""

    def __init__(self, enter_edit_mode: bool = False) -> None:
        self.objects: Dict[str, FakeObject] = {}
        self.scene = FakeScene(self, "Scene")
        self.scenes = [self.scene]
        self.layer_root = FakeLayer(self.scene.collection)
        self.view_layer = SimpleNamespace(
            layer_collection=self.layer_root,
            active_layer_collection=self.layer_root,
            objects=SimpleNamespace(active=None),
        )
        self.mode = "OBJECT"
        self.enter_edit_mode = enter_edit_mode
        self.overrides: List[Dict[str, Any]] = []
        self.calls: List[tuple] = []
        self.removed_directly: List[str] = []
        self.rigid_body_cache_resets = 0
        self.bpy = self._bpy()

    # -- scene building ----------------------------------------------------

    def unique(self, name: str) -> str:
        if name not in self.objects:
            return name
        number = 1
        while f"{name}.{number:03d}" in self.objects:
            number += 1
        return f"{name}.{number:03d}"

    def add(
        self,
        name: str,
        collection: Optional[FakeCollection] = None,
        object_type: str = "MESH",
    ) -> FakeObject:
        obj = FakeObject(self, self.unique(name), object_type)
        self.objects[obj.name] = obj
        (collection or self.scene.collection).link(obj)
        return obj

    def collection(self, name: str, parent: Optional[FakeLayer] = None) -> FakeLayer:
        """A collection under ``parent`` (the scene's own by default), and its layer."""
        parent = parent or self.layer_root
        child = FakeCollection(name)
        child.objects = self.members(child)  # type: ignore[attr-defined]
        parent.collection.children.append(child)
        layer = FakeLayer(child)
        parent.children.append(layer)
        return layer

    def members(self, collection: FakeCollection) -> FakeStore:
        """``Collection.objects``: the objects directly in a collection."""
        return FakeStore(
            lambda: [
                obj
                for obj in self.objects.values()
                if any(c is collection for c in obj.users_collection)
            ]
        )

    def layer_of(self, collection: FakeCollection) -> Optional[FakeLayer]:
        def find(layer: FakeLayer) -> Optional[FakeLayer]:
            if layer.collection is collection:
                return layer
            return next(
                (found for found in map(find, layer.children) if found is not None),
                None,
            )

        return find(self.layer_root)

    def collection_hidden(self, collection: FakeCollection) -> bool:
        """Whether a collection or one holding it is excluded, hidden or disabled."""

        def path(layer: FakeLayer) -> Optional[List[FakeLayer]]:
            if layer.collection is collection:
                return [layer]
            for child in layer.children:
                found = path(child)
                if found:
                    return [layer, *found]
            return None

        chain = path(self.layer_root)
        if chain is None:
            return False  # another scene's collection
        return any(
            layer.exclude or layer.hide_viewport or layer.collection.hide_viewport
            for layer in chain
        )

    # -- bpy ---------------------------------------------------------------

    def _bpy(self) -> SimpleNamespace:
        world = self

        class Context:
            scene = property(lambda _: world.scene)
            view_layer = property(lambda _: world.view_layer)
            mode = property(lambda _: world.mode)

            @property
            def active_object(self) -> Any:
                # The 3D View's context: Blender 5.0.1 gave None for the active
                # object while the view did not show it.
                active = world.view_layer.objects.active
                return active if active is not None and active.visible_get() else None

            @staticmethod
            def evaluated_depsgraph_get() -> None:
                return None

            @staticmethod
            @contextmanager
            def temp_override(**members: Any) -> Iterator[None]:
                world.overrides.append(members)
                try:
                    yield
                finally:
                    world.overrides.pop()

        def adder(operator: str, base: str, object_type: str, edits: bool) -> Callable:
            def add(**properties: Any) -> set:
                world.calls.append((operator, properties))
                target = world.view_layer.active_layer_collection.collection
                obj = world.add(base, target, object_type)
                obj.location = FakeVector(properties["location"])
                world.view_layer.objects.active = obj
                enters = properties.get("enter_editmode", world.enter_edit_mode)
                if object_type in ("MESH", "CAMERA") and enters:
                    obj.mode = "EDIT"
                    world.mode = "EDIT_MESH"
                return {"FINISHED"}

            names = ["location", "rotation", "align"] + (
                ["enter_editmode"] if edits else []
            )
            if operator == "object.light_add":
                names.append("type")
            return _operator(add, names)

        def mode_set(mode: str) -> set:
            world.calls.append(("object.mode_set", {"mode": mode}))
            override = world.overrides[-1] if world.overrides else {}
            obj = override.get("active_object") or world.view_layer.objects.active
            if obj.type == "EMPTY" and mode != "OBJECT":
                # Blender's poll: an empty has no Edit Mode.
                raise RuntimeError("Operator bpy.ops.object.mode_set.poll() failed")
            obj.mode = mode
            world.mode = "OBJECT" if mode == "OBJECT" else f"{mode}_MESH"
            return {"FINISHED"}

        def delete(use_global: bool = False, confirm: bool = True) -> set:
            members = world.overrides[-1]
            world.calls.append(
                (
                    "object.delete",
                    {
                        "use_global": use_global,
                        "confirm": confirm,
                        "selected_objects": [
                            o.name for o in members["selected_objects"]
                        ],
                    },
                )
            )
            for obj in members["selected_objects"]:
                if obj.library is not None:
                    continue  # Blender skips linked library data
                for collection in list(obj.users_collection):
                    if use_global or world.scene.holds(collection):
                        obj.users_collection.remove(collection)
                if obj.rigid_body is not None:
                    world.rigid_body_cache_resets += 1
                if not obj.users_collection:
                    del world.objects[obj.name]
            return {"FINISHED"}

        def remove(obj: FakeObject, do_unlink: bool = True) -> None:
            world.removed_directly.append(obj.name)
            obj.users_collection.clear()
            del world.objects[obj.name]

        objects = FakeStore(lambda: list(world.objects.values()))
        collections = FakeStore(
            lambda: [
                c for scene in world.scenes for c in scene.collection.children_recursive
            ]
        )
        objects.remove = remove  # type: ignore[attr-defined]
        meshes = FakeStore(lambda: [])
        meshes.remove = lambda data: None  # type: ignore[attr-defined]
        return SimpleNamespace(
            app=SimpleNamespace(version=(5, 0, 1)),
            context=Context(),
            data=SimpleNamespace(
                objects=objects,
                collections=collections,
                scenes=world.scenes,
                meshes=meshes,
                lights=meshes,
                cameras=meshes,
                filepath="",
                is_dirty=False,
            ),
            ops=SimpleNamespace(
                mesh=SimpleNamespace(
                    primitive_cube_add=adder(
                        "mesh.primitive_cube_add", "Cube", "MESH", True
                    ),
                    primitive_plane_add=adder(
                        "mesh.primitive_plane_add", "Plane", "MESH", True
                    ),
                ),
                object=SimpleNamespace(
                    light_add=adder("object.light_add", "Point", "LIGHT", False),
                    camera_add=adder("object.camera_add", "Camera", "CAMERA", True),
                    empty_add=adder("object.empty_add", "Empty", "EMPTY", False),
                    mode_set=_operator(mode_set, ["mode"], mode=MODES),
                    delete=_operator(delete, ["use_global", "confirm"]),
                ),
            ),
            types=fake_bpy_types(),
        )


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch, mathutils: Any) -> World:
    world = World()
    monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
    monkeypatch.setattr(blender_runtime, "bpy", world.bpy)
    return world


def _session(attached: bool = True) -> BlenderRuntimeSession:
    """An attached session looks objects up in the current scene, an embedded one in the file."""
    return BlenderRuntimeSession(scene_scope=attached)


# -- create_object (B1-12, B1-19) ---------------------------------------------


@pytest.mark.parametrize("hidden_by", ["monitor", "eye", "excluded parent"])
def test_create_object_puts_the_object_where_it_can_be_seen(
    world: World, hidden_by: str
) -> None:
    """It failed with AttributeError and left an unnamed 'Cube.001' when the
    active collection was disabled in viewports: the 3D View's active_object
    was None for the new, invisible cube."""
    work = world.collection("Work")
    active = world.collection("HiddenWork", parent=work)
    if hidden_by == "monitor":
        active.collection.hide_viewport = True
    elif hidden_by == "eye":
        active.hide_viewport = True
    else:
        work.exclude = True
    world.view_layer.active_layer_collection = active
    world.add("Cube")

    result = _session().create_object("CUBE", name="NewAgentBox", scale=[3, 3, 3])

    assert sorted(world.objects) == ["Cube", "NewAgentBox"]
    box = world.objects["NewAgentBox"]
    assert box.visible_get() and list(box.scale) == [3.0, 3.0, 3.0]
    assert box.users_collection == [world.scene.collection]
    assert (
        result["name"] == "NewAgentBox" and result["collection"] == "Scene Collection"
    )
    hidden = "Work" if hidden_by == "excluded parent" else "HiddenWork"
    assert f"collection '{hidden}'" in result["warning"]
    # The user's active collection is theirs to keep.
    assert world.view_layer.active_layer_collection is active


def test_create_object_uses_a_visible_active_collection(world: World) -> None:
    props = world.collection("Props")
    world.view_layer.active_layer_collection = props

    result = _session().create_object("POINT_LIGHT", location=[1, 2, 3])

    light = world.objects["Point"]
    assert light.users_collection == [props.collection]
    assert result == {
        "name": "Point",
        "object_type": "POINT_LIGHT",
        "location": [1.0, 2.0, 3.0],
        "collection": "Props",
        "mode": "OBJECT",
        "warning": None,
    }
    operator, properties = world.calls[0]
    assert operator == "object.light_add" and properties["type"] == "POINT"


@pytest.mark.parametrize("object_type", ["CUBE", "CAMERA"])
def test_create_object_stays_in_object_mode_under_the_enter_edit_mode_preference(
    world: World, object_type: str
) -> None:
    """The window was left in Edit Mode, where the next guarded call failed."""
    world.enter_edit_mode = True

    result = _session().create_object(object_type, name="PrefObject")

    assert world.mode == "OBJECT" and result["mode"] == "OBJECT"
    assert world.objects["PrefObject"].mode == "OBJECT"
    assert world.calls[0][1]["enter_editmode"] is False


def test_create_object_returns_to_object_mode_when_the_operator_entered_edit_mode(
    world: World,
) -> None:
    """An add operator without enter_editmode that still enters Edit Mode is undone."""
    plane = world.bpy.ops.mesh.primitive_plane_add
    world.bpy.ops.mesh.primitive_plane_add = _operator(
        lambda **properties: plane(**properties, enter_editmode=True),
        ["location", "rotation"],
    )

    result = _session().create_object("PLANE", name="Floor")

    assert world.mode == "OBJECT" and result["mode"] == "OBJECT"
    assert ("object.mode_set", {"mode": "OBJECT"}) in world.calls


def test_create_object_removes_the_object_when_naming_it_fails(world: World) -> None:
    """A failed call must not leave an unnamed object behind."""
    cube = world.bpy.ops.mesh.primitive_cube_add

    def refusing(**properties: Any) -> set:
        result = cube(**properties)
        world.view_layer.objects.active.refuse_name = "Bad"
        return result

    world.bpy.ops.mesh.primitive_cube_add = _operator(
        refusing, ["location", "rotation", "enter_editmode"]
    )

    with pytest.raises(ValueError, match="cannot name"):
        _session().create_object("CUBE", name="Bad")

    assert world.objects == {}


def test_create_object_refuses_values_it_cannot_write(world: World) -> None:
    with pytest.raises(ValueError, match="location must be 3 finite numbers"):
        _session().create_object("CUBE", location=[0, float("nan"), 0])
    assert world.objects == {} and world.calls == []


def test_create_object_refuses_edit_mode_before_adding_anything(world: World) -> None:
    """Embedded mode has no bridge guard: in Edit Mode the add operator edited
    the active mesh, and the call then failed with nothing created."""
    world.view_layer.objects.active = world.add("Edited")
    world.mode = "EDIT_MESH"

    with pytest.raises(ValueError, match="set_blender_object_mode"):
        _session(attached=False).create_object("CUBE")

    assert world.calls == [] and sorted(world.objects) == ["Edited"]


# -- delete_object (B1-11, B3-03) ---------------------------------------------


def _shared(world: World) -> FakeObject:
    """An object linked into the current scene and into a second one."""
    other = FakeScene(world, "VariantB")
    world.scenes.append(other)
    obj = world.add("SharedProp")
    other.collection.link(obj)
    return obj


def test_delete_object_leaves_the_object_in_other_scenes(world: World) -> None:
    """It removed the object from every scene, unlike Blender's Delete."""
    _shared(world)

    result = _session().delete_object("SharedProp")

    assert world.scene.objects.get("SharedProp") is None
    assert world.scenes[1].objects.get("SharedProp") is not None
    assert result == {
        "deleted_name": "SharedProp",
        "use_global": False,
        "removed_constraints": [],
        "warnings": [],
        "removed_from_file": False,
        "remaining_scenes": ["VariantB"],
    }


def _joint(world: World, scene: Any = None) -> Dict[str, Any]:
    """Two rigid bodies Link and Base joined by a constraint on the empty RBC_Base_Link.

    ``constraint_remove`` acts on the overridden active object, as Blender's
    does, and records the scene it ran in and whether both bodies still existed.
    """
    scene = scene or world.scene
    body, other = world.add("Link"), world.add("Base")
    body.rigid_body, other.rigid_body = SimpleNamespace(), SimpleNamespace()
    carrier = world.add("RBC_Base_Link", object_type="EMPTY")
    carrier.hide_viewport = False
    carrier.rigid_body_constraint = SimpleNamespace(object1=other, object2=body)
    scene.rigidbody_world = SimpleNamespace(constraints=SimpleNamespace(all_objects=[carrier]))
    log: List[Tuple[str, Any, bool]] = []

    def constraint_remove() -> set:
        members = world.overrides[-1]
        target = members["active_object"]
        log.append((target.name, members["scene"], carrier.hide_viewport))
        assert "Link" in world.objects and "Base" in world.objects, "both bodies must exist"
        target.rigid_body_constraint = None
        return {"FINISHED"}

    world.bpy.ops.rigidbody = SimpleNamespace(constraint_remove=constraint_remove)
    return {"body": body, "other": other, "carrier": carrier, "log": log}


def test_delete_object_removes_a_constraint_joining_it_first(world: World) -> None:
    """Blender's Delete left the constraint pointing at the freed body: the next restart of
    the simulation (a bake) crashed Blender 5.0.1 in btDiscreteDynamicsWorld::removeConstraint."""
    joint = _joint(world)
    unrelated = world.add("RBC_Base_Base2", object_type="EMPTY")
    unrelated.rigid_body_constraint = SimpleNamespace(object1=joint["other"], object2=None)
    world.scene.rigidbody_world.constraints.all_objects.append(unrelated)
    world.view_layer.objects.active = joint["other"]

    result = _session().delete_object("Link")

    assert joint["log"] == [("RBC_Base_Link", world.scene, False)]
    assert result["removed_constraints"] == ["RBC_Base_Link"] and result["warnings"] == []
    assert "Link" not in world.objects and "RBC_Base_Link" in world.objects
    assert unrelated.rigid_body_constraint is not None
    # The user's active object is restored.
    assert world.view_layer.objects.active is joint["other"]


@pytest.mark.parametrize("gone", ["deleted", "rigid_body_removed"])
def test_delete_object_leaves_a_constraint_whose_other_body_is_gone(
    world: World, gone: str
) -> None:
    """Review of #360: removing such a constraint reads the freed body and crashed Blender
    (MallocScribble), where Blender's own Delete did not."""
    joint = _joint(world)
    if gone == "deleted":
        joint["carrier"].rigid_body_constraint.object1 = None
    else:
        joint["other"].rigid_body = None

    result = _session().delete_object("Link")

    assert joint["log"] == []
    assert result["removed_constraints"] == []
    assert "already deleted or has no rigid body" in result["warnings"][0]
    assert "Link" not in world.objects


def test_delete_object_removes_a_constraint_whose_empty_is_disabled_in_viewports(
    world: World,
) -> None:
    """Blender's constraint_remove refuses an object disabled in viewports; it is shown for
    the call and hidden again."""
    joint = _joint(world)
    joint["carrier"].hide_viewport = True

    result = _session().delete_object("Link")

    assert joint["log"] == [("RBC_Base_Link", world.scene, False)]
    assert result["removed_constraints"] == ["RBC_Base_Link"]
    assert joint["carrier"].hide_viewport is True


def test_delete_object_links_an_excluded_empty_into_the_scene_for_the_removal(
    world: World,
) -> None:
    """Outside the view layer (an excluded collection) the empty cannot be made active, and
    Blender 5.0.1 ran constraint_remove on the layer's active object instead."""
    _joint(world)
    links: List[str] = []
    world.view_layer.objects = SimpleNamespace(
        active=None,
        get=lambda name: next((o for o in world.objects.values() if o.name in links and o.name == name), None),
    )
    world.scene.collection.objects = SimpleNamespace(  # type: ignore[attr-defined]
        link=lambda ob: links.append(ob.name),
        unlink=lambda ob: links.append(f"unlink {ob.name}"),
    )

    result = _session().delete_object("Link")

    assert links == ["RBC_Base_Link", "unlink RBC_Base_Link"]
    assert result["removed_constraints"] == ["RBC_Base_Link"]
    assert world.view_layer.objects.active is None


def test_delete_object_globally_removes_a_joint_of_another_scene(world: World) -> None:
    """With use_global the constraint is removed in the scene whose world holds it."""
    other_scene = FakeScene(world, "VariantB")
    other_scene.view_layers = [world.view_layer]  # type: ignore[attr-defined]
    world.scenes.append(other_scene)
    joint = _joint(world, scene=other_scene)
    world.scene.rigidbody_world = None

    result = _session().delete_object("Link", use_global=True)

    assert joint["log"] == [("RBC_Base_Link", other_scene, False)]
    assert result["removed_constraints"] == ["RBC_Base_Link"]


def test_delete_object_goes_ahead_when_a_constraint_cannot_be_removed(world: World) -> None:
    """Blender's Delete runs as before; the warning says the simulation stays refused."""
    joint = _joint(world)

    def refuse() -> set:
        raise RuntimeError("Operator bpy.ops.rigidbody.constraint_remove.poll() failed")

    world.bpy.ops.rigidbody = SimpleNamespace(constraint_remove=refuse)

    result = _session().delete_object("Link")

    assert result["removed_constraints"] == []
    assert "constraint_remove.poll() failed" in result["warnings"][0]
    assert "restarting the simulation is refused" in result["warnings"][0]
    assert "Link" not in world.objects
    assert joint["carrier"].rigid_body_constraint is not None


def test_delete_object_globally_deletes_it_from_every_scene(world: World) -> None:
    _shared(world)

    result = _session().delete_object("SharedProp", use_global=True)

    assert "SharedProp" not in world.objects
    assert result["removed_from_file"] is True and result["remaining_scenes"] == []


def test_delete_object_runs_blenders_delete_on_the_object_alone(world: World) -> None:
    """Removing the object directly left a deleted rigid body in the cache, so
    another body jumped to its cached positions after a frame change; Blender's
    Delete operator resets that cache."""
    body = world.add("A")
    body.rigid_body = SimpleNamespace(type="ACTIVE")
    world.add("Bystander")

    _session().delete_object("A")

    assert world.removed_directly == []
    assert world.rigid_body_cache_resets == 1
    assert world.calls == [
        (
            "object.delete",
            {"use_global": False, "confirm": False, "selected_objects": ["A"]},
        )
    ]
    assert sorted(world.objects) == ["Bystander"]


def test_delete_object_refuses_an_object_outside_the_current_scene(
    world: World,
) -> None:
    """Embedded mode looks objects up in the whole file."""
    other = FakeScene(world, "VariantB")
    world.scenes.append(other)
    elsewhere = world.add("OnlyInVariantB", other.collection)

    with pytest.raises(ValueError, match="use_global=true"):
        _session(attached=False).delete_object("OnlyInVariantB")

    assert world.objects["OnlyInVariantB"] is elsewhere
    _session(attached=False).delete_object("OnlyInVariantB", use_global=True)
    assert world.objects == {}


def test_delete_object_reports_an_object_blender_kept(world: World) -> None:
    world.add("Linked").library = SimpleNamespace(name="lib.blend")

    with pytest.raises(RuntimeError, match="did not delete 'Linked'"):
        _session().delete_object("Linked")


# -- set_object_mode (B1-19's way out) ----------------------------------------


def test_set_object_mode_returns_the_window_to_object_mode(world: World) -> None:
    obj = world.add("Sphere")
    world.view_layer.objects.active = obj
    obj.mode, world.mode = "EDIT", "EDIT_MESH"

    result = _session().set_object_mode()

    assert world.mode == "OBJECT" and obj.mode == "OBJECT"
    assert result == {
        "object_name": "Sphere",
        "mode": "OBJECT",
        "previous_mode": "EDIT",
        "context_mode": "OBJECT",
        "changed": True,
    }


def test_set_object_mode_activates_a_named_object(world: World) -> None:
    world.view_layer.objects.active = world.add("Other")
    world.add("Sphere")

    result = _session().set_object_mode("edit", object_name="Sphere")

    assert world.view_layer.objects.active is world.objects["Sphere"]
    assert result["mode"] == "EDIT" and result["context_mode"] == "EDIT_MESH"


def test_a_refused_mode_switch_keeps_the_active_object(world: World) -> None:
    """The named object became active even when Blender refused its mode."""
    sphere = world.add("Sphere")
    world.view_layer.objects.active = sphere
    world.add("Marker", object_type="EMPTY")

    with pytest.raises(RuntimeError, match="poll"):
        _session().set_object_mode("EDIT", object_name="Marker")

    assert world.view_layer.objects.active is sphere


def test_set_object_mode_without_an_object(world: World) -> None:
    assert _session().set_object_mode()["changed"] is False
    with pytest.raises(ValueError, match="pass object_name"):
        _session().set_object_mode("EDIT")
    with pytest.raises(ValueError, match="Blender's modes: OBJECT, EDIT"):
        _session().set_object_mode("FLY")


# -- listing and search (B1-13) -----------------------------------------------


def _props(world: World, zones: int = 6, per_zone: int = 100, loose: int = 6) -> None:
    """A scene like the audit's: props in zone collections, a few in the scene's own."""
    for zone in range(zones):
        layer = world.collection(f"Zone_{zone:02d}")
        for index in range(per_zone):
            world.add(f"Prop_{zone * per_zone + index:05d}", layer.collection)
    for index in range(loose):
        world.add(f"Loose_{index}", object_type="EMPTY")


def _all_pages(read: Callable[[int], Dict[str, Any]]) -> List[Dict[str, Any]]:
    pages, offset = [], 0
    while offset is not None:
        page = read(offset)
        pages.append(page)
        offset = page["next_offset"]
    return pages


def test_listing_pages_through_every_object_within_the_result_budget(
    world: World,
) -> None:
    """max_items 5000 reported count 5000 and returned 429 of 6006 objects."""
    _props(world, zones=60)
    session = _session()

    pages = _all_pages(
        lambda offset: session.list_scene_objects(max_items=5000, offset=offset)
    )

    names = [entry["name"] for page in pages for entry in page["objects"]]
    assert names == [obj.name for obj in world.scene.objects]
    assert len(names) == 6006
    for page in pages:
        assert page["count"] == len(page["objects"])
        assert page["total"] == 6006
        assert page["truncated"] is (page["next_offset"] is not None)
        # The server measures its budget as json.dumps does.
        assert len(json.dumps(page)) < DEFAULT_RESULT_BUDGET_BYTES
    assert len(pages) > 1 and pages[-1]["truncated"] is False


def test_listing_is_truncated_only_when_more_objects_follow(world: World) -> None:
    """truncated was true when exactly max_items objects existed."""
    _props(world, zones=1, per_zone=100, loose=0)
    session = _session()

    exact = session.list_scene_objects(max_items=100)
    short = session.list_scene_objects(max_items=99)
    beyond = session.list_scene_objects(max_items=10, offset=500)

    assert (exact["count"], exact["truncated"], exact["next_offset"]) == (
        100,
        False,
        None,
    )
    assert (short["count"], short["truncated"], short["next_offset"]) == (99, True, 99)
    assert (beyond["count"], beyond["truncated"], beyond["total"]) == (0, False, 100)


def test_listing_names_a_collection_holding_each_object(world: World) -> None:
    """collection was null for every entry."""
    _props(world, zones=2, per_zone=2, loose=1)
    other = FakeScene(world, "VariantB")
    world.scenes.append(other)
    world.add("Elsewhere", other.collection)
    session = _session(attached=False)

    listed = {
        e["name"]: e["collection"] for e in session.list_scene_objects()["objects"]
    }
    filtered = session.list_scene_objects(collection_name="Zone_01")

    assert listed["Prop_00000"] == "Zone_00" and listed["Prop_00003"] == "Zone_01"
    assert listed["Loose_0"] == "Scene Collection"
    # Embedded mode lists the whole file; another scene's object names its own.
    assert listed["Elsewhere"] == "Scene Collection"
    assert {e["collection"] for e in filtered["objects"]} == {"Zone_01"}


def test_a_page_of_long_names_stops_before_the_budget(world: World) -> None:
    """Names of 60 non-ASCII characters cost 360 bytes of JSON each."""
    for index in range(400):
        world.add(f"{index:03d}" + "é" * 60)
    session = _session()

    page = session.list_scene_objects(max_items=400)

    entries = len(json.dumps(page["objects"]))
    assert entries <= blender_runtime.LISTING_JSON_BYTES_MAX
    assert 0 < page["count"] < 400 and page["next_offset"] == page["count"]


def test_search_matches_object_types_in_any_case(world: World) -> None:
    """object_type 'mesh' silently matched nothing."""
    _props(world, zones=1, per_zone=30, loose=3)
    session = _session()

    page = session.search_objects(object_type=" mesh ", max_results=20)
    rest = session.search_objects(object_type="MESH", max_results=20, offset=20)

    assert page["object_type"] == "MESH"
    assert (page["count"], page["total"], page["next_offset"]) == (20, 30, 20)
    assert (rest["count"], rest["truncated"]) == (10, False)
    assert {e["collection"] for e in page["objects"]} == {"Zone_00"}


def test_search_refuses_an_object_type_blender_does_not_have(world: World) -> None:
    with pytest.raises(ValueError, match="Blender's object types: MESH, CURVE"):
        _session().search_objects(object_type="MESHES")


def test_the_listing_byte_cap_is_mirrored_and_fits_the_result_budget() -> None:
    from simul.mcp.schemas.blender import LISTING_JSON_BYTES_MAX

    assert LISTING_JSON_BYTES_MAX == blender_runtime.LISTING_JSON_BYTES_MAX
    assert LISTING_JSON_BYTES_MAX + 2_000 < DEFAULT_RESULT_BUDGET_BYTES
    assert "GREASEPENCIL" in BLENDER_5_OBJECT_TYPES


# -- summarize_scene and get_file_info (B1-18) --------------------------------


def test_summary_lists_the_scenes_top_level_collections(world: World) -> None:
    """It listed every collection in the file: nested, other scenes' and orphans."""
    zone = world.collection("Zone_00")
    world.collection("Zone_01")
    world.collection("Zone_00_Nested", parent=zone)
    world.bpy.data.collections = FakeStore(lambda: [FakeCollection("Orphan")])
    world.scene.camera = None
    world.scene.frame_current = world.scene.frame_start = 1
    world.scene.frame_end = 250

    summary = _session().summarize_scene()

    assert summary["collection_names"] == ["Zone_00", "Zone_01"]


def test_file_info_counts_the_scenes_objects_and_the_files(world: World) -> None:
    """object_count, described as the scene's, counted every object in the file."""
    _props(world, zones=1, per_zone=3, loose=0)
    other = FakeScene(world, "Other")
    world.scenes.append(other)
    world.add("Elsewhere", other.collection)

    info = _session().get_file_info()

    assert (info["object_count"], info["file_object_count"]) == (3, 4)


# -- is_object_within_bounds (B1-14) ------------------------------------------


def _box(low: float = -1.0, high: float = 1.0) -> List[FakePropArray]:
    return [
        FakePropArray((x, y, z))
        for x in (low, high)
        for y in (low, high)
        for z in (low, high)
    ]


@pytest.fixture
def placed(monkeypatch: pytest.MonkeyPatch, mathutils: Any) -> Callable[..., Any]:
    """Return a factory: a session over one object with a bounding box, placed."""

    def place(**transform: Any) -> BlenderRuntimeSession:
        obj = FakeTransformObject("Probe", "MESH", **transform)
        obj.bound_box = transform.pop("bound_box", None) or _box()
        store = {"Probe": obj}
        bpy = SimpleNamespace(
            app=SimpleNamespace(version=(5, 0, 1)),
            data=SimpleNamespace(objects=store),
            context=SimpleNamespace(evaluated_depsgraph_get=lambda: None),
        )
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", bpy)
        return BlenderRuntimeSession()

    return place


def test_bounds_test_the_bounding_box_not_the_origin(
    placed: Callable[..., Any],
) -> None:
    """A 10 m plane was within_bounds of a 2 m box around its origin."""
    session = placed(location=(0.0, 40.0, 0.0), scale=(5.0, 5.0, 0.0))

    result = session.is_object_within_bounds("Probe", [-1, 39, -1], [1, 41, 1])

    assert result["within_bounds"] is False
    assert result["relation"] == "intersecting" and result["intersects"] is True
    assert result["origin_within_bounds"] is True
    assert result["object_bbox_min"] == pytest.approx([-5.0, 35.0, 0.0])
    assert result["object_bbox_max"] == pytest.approx([5.0, 45.0, 0.0])


@pytest.mark.parametrize(
    "low, high, relation",
    [
        ([-6, 34, -1], [6, 46, 1], "inside"),
        ([5, 45, 0], [6, 46, 1], "intersecting"),  # touching a corner
        ([20, 20, 20], [21, 21, 21], "outside"),
    ],
)
def test_bounds_report_inside_intersecting_or_outside(
    placed: Callable[..., Any], low: list, high: list, relation: str
) -> None:
    session = placed(location=(0.0, 40.0, 0.0), scale=(5.0, 5.0, 0.0))

    result = session.is_object_within_bounds("Probe", low, high)

    assert result["relation"] == relation
    assert result["within_bounds"] is (relation == "inside")


def test_bounds_test_a_turned_box_as_turned(placed: Callable[..., Any]) -> None:
    """A box in the corner of a turned cube's axis-aligned extent, but off the cube."""
    session = placed(rotation_euler=(0.0, 0.0, math.pi / 4))

    corner = session.is_object_within_bounds("Probe", [1.2, 1.2, -1], [2, 2, 1])
    edge = session.is_object_within_bounds("Probe", [1.0, -0.1, -1], [2, 0.1, 1])

    assert corner["object_bbox_max"][0] == pytest.approx(math.sqrt(2))
    assert corner["relation"] == "outside"
    assert edge["relation"] == "intersecting"


def test_bounds_allow_for_single_precision_rounding(placed: Callable[..., Any]) -> None:
    """A cube turned a quarter turn fills its own box: inside, not intersecting."""
    session = placed(location=(5.0, 5.0, 5.0), rotation_euler=(0.0, 0.0, math.pi / 2))

    result = session.is_object_within_bounds("Probe", [4, 4, 4], [6, 6, 6])

    assert result["relation"] == "inside"


@pytest.mark.parametrize(
    "low, high",
    [([1, 0, 0], [-1, 1, 1]), ([0, 0, float("nan")], [1, 1, 1])],
)
def test_bounds_refuse_a_box_that_holds_nothing(
    placed: Callable[..., Any], low: list, high: list
) -> None:
    from pydantic import ValidationError

    from simul.mcp.schemas.blender import BlenderBoundsCheckRequest

    with pytest.raises(ValueError):
        placed().is_object_within_bounds("Probe", low, high)
    with pytest.raises(ValidationError):
        BlenderBoundsCheckRequest(object_name="Probe", bounds_min=low, bounds_max=high)


def test_box_relation_of_a_point_is_its_containment() -> None:
    """An empty's bounding box is its origin, eight times."""
    point = [[0.5, 0.5, 0.5]] * 8
    identity = [list(row) for row in FakeMatrix.Identity(4).rows][:3]

    inside = blender_runtime._box_relation(point, identity, [0, 0, 0], [1, 1, 1])
    outside = blender_runtime._box_relation(point, identity, [1, 1, 1], [2, 2, 2])

    assert (inside, outside) == ("inside", "outside")
