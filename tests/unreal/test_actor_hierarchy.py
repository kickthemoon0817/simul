"""Actor lookup, parenting and components against a UE-shaped editor.

Observed on UE 5.7.4 through the MCP server:

- After ``delete_unreal_actor`` reported ``deleted: true``,
  ``get_unreal_actor_info`` on the same path succeeded with an empty name, a
  zero scale and mobility "None", and focus succeeded with a false "no
  bounds" warning: ``actor_at`` loaded the path with ``load_object``, which
  still finds a destroyed actor until garbage collection.
- With MapB open, info, transform and focus calls on a path into MapA loaded
  MapA from disk and reported success; the edits left MapA dirty, so a later
  Save All wrote changes nobody saw.
- ``set_unreal_actor_parent`` reported success for the actor itself as its
  parent, for a cycle and for a parent that does not exist:
  ``K2_AttachToActor`` returned ``ReturnValue: false`` and the reply ignored
  it. The attachment was not undoable, and detaching was an empty parent.
- ``add_unreal_component`` failed for every input with ``400, message=''``:
  Remote Control refuses ``AddComponentByClass`` ("deprecated or unavailable
  remotely"), ``component_name`` was unused and the class was always
  ``/Script/Engine.<name>``.

Now ``actor_at`` finds the actor among loaded objects (``find_object``, which
never loads a map) and accepts only a live actor of the editor world;
parenting refuses those cases with their reason, attaches in one undoable
transaction and reads the parent back; and components are added through
``SubobjectDataSubsystem`` the way the editor's Add Component button adds
them. The fake editor below answers the scripts the way UE 5.7.4 did.
"""

from __future__ import annotations

import asyncio
import enum
import io
import json
import sys
from contextlib import contextmanager, redirect_stdout
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pytest
from fastmcp import Client

from simul.adapters import unreal_runtime
from simul.adapters._unreal_scripts import ACTOR_HELPERS, ACTOR_NOT_FOUND
from simul.config import Settings
from simul.mcp import server as server_module
from simul.mcp.usage_tracker import ToolUsageTracker
from tests.fakes import AvailableAdapter
from tests.unreal.editor_world import SUBSYSTEM_NAMES, World, editor_subsystems

MAP_B = "/Game/Maps/MapB.MapB"
MAP_A = "/Game/Maps/MapA.MapA"
LEVEL = MAP_B + ":PersistentLevel."


# ---------------------------------------------------------------------------
# The fake ``unreal`` module
# ---------------------------------------------------------------------------


class Class:
    """A ``unreal.Class``; ``pytype`` is the Python type its objects have."""

    def __init__(self, name: str, parent: Optional["Class"] = None, abstract: bool = False) -> None:
        self.name, self.parent, self.abstract = name, parent, abstract
        self.pytype: Any = None

    def get_name(self) -> str:
        return self.name

    def get_path_name(self) -> str:
        return "/Script/Engine." + self.name


def class_is_child_of(cls: Class, parent: Class) -> bool:
    node: Optional[Class] = cls
    while node is not None:
        if node is parent:
            return True
        node = node.parent
    return False


def _native(name: str, base: Any, abstract: bool = False) -> Any:
    """A Python wrapper type with its ``static_class``, as ``unreal`` exposes a native class."""
    cls = Class(name, base.CLASS if base is not None else None, abstract)
    pytype = type(name, (base,) if base is not None else (), {"CLASS": cls})
    cls.pytype = pytype
    return pytype


class Mobility(enum.Enum):
    STATIC = 0
    STATIONARY = 1
    MOVABLE = 2


class _Object:
    CLASS: Class

    @classmethod
    def static_class(cls) -> Class:
        return cls.CLASS

    def get_class(self) -> Class:
        return type(self).CLASS


ObjectBase = type("ObjectBase", (_Object,), {"CLASS": Class("Object")})
ActorComponent = _native("ActorComponent", ObjectBase, abstract=True)


def _component_init(self: Any, name: str, owner: "Actor") -> None:
    self.name, self.owner = name, owner
    self.attach_parent: Any = None
    self.mobility = Mobility.STATIC


ActorComponent.__init__ = _component_init
ActorComponent.get_name = lambda self: self.name
ActorComponent.get_path_name = lambda self: self.owner.get_path_name() + "." + self.name
ActorComponent.get_owner = lambda self: self.owner
SceneComponent = _native("SceneComponent", ActorComponent)
SceneComponent.get_attach_parent = lambda self: self.attach_parent
PrimitiveComponent = _native("PrimitiveComponent", SceneComponent, abstract=True)
StaticMeshComponent = _native("StaticMeshComponent", PrimitiveComponent)
PointLightComponent = _native("PointLightComponent", SceneComponent)
SpotLightComponent = _native("SpotLightComponent", SceneComponent)
ProjectileMovementComponent = _native("ProjectileMovementComponent", ActorComponent)


class Actor(_Object):
    CLASS = Class("Actor", ObjectBase.CLASS)

    def __init__(self, editor: "Editor", path: str, world: World, mobility: Mobility = Mobility.STATIC) -> None:
        self.editor, self.path, self.world = editor, path, world
        self.label = path.rsplit(".", 1)[-1]
        self.valid = True
        self.parent: Optional[Actor] = None
        self.root_component: Any = StaticMeshComponent("StaticMeshComponent0", self)
        self.root_component.mobility = mobility
        self.components: List[Any] = [self.root_component]

    def get_path_name(self) -> str:
        return self.path

    def get_actor_label(self) -> str:
        return self.label

    def get_world(self) -> World:
        return self.world

    def modify(self) -> bool:
        self.editor.events.append(("modify", self.label))
        return True

    def get_components_by_class(self, cls: Any) -> List[Any]:
        return [c for c in self.components if isinstance(c, cls)]

    def get_attach_parent_actor(self) -> Optional["Actor"]:
        return self.parent

    def attach_to_actor(
        self, parent: "Actor", socket: str, location: Any, rotation: Any, scale: Any, weld: bool
    ) -> bool:
        """``K2_AttachToActor``: False, changing nothing, for what Unreal refuses."""
        self.editor.events.append(("attach", self.label, parent.label, (location, rotation, scale)))
        node: Optional[Actor] = parent
        while node is not None:
            if node is self:
                return False
            node = node.parent
        if self.root_component.mobility is Mobility.STATIC and parent.root_component.mobility is not Mobility.STATIC:
            return False
        if self.editor.refuse_attach:
            return False
        self.parent = parent
        return True

    def detach_from_actor(self, location: Any, rotation: Any, scale: Any) -> None:
        self.editor.events.append(("detach", self.label, (location, rotation, scale)))
        self.parent = None


StaticMeshActor = type("StaticMeshActor", (Actor,), {"CLASS": Class("StaticMeshActor", Actor.CLASS)})


class Text:
    def __init__(self, text: str = "") -> None:
        self.text = text

    def __str__(self) -> str:
        return self.text


class Editor:
    """The open map (MapB), a map on disk (MapA) and the editor's scripting."""

    def __init__(self) -> None:
        self.world = World(MAP_B)
        self.map_a = World(MAP_A)
        # The running game's copy of MapB while Play In Editor runs.
        self.play_world = World("/Game/Maps/UEDPIE_0_MapB.MapB")
        # Loaded objects by path: what find_object finds, destroyed actors
        # included until garbage collection, as on UE 5.7.4.
        self.memory: Dict[str, Actor] = {}
        # Map packages on disk that are not loaded, and the actors they hold.
        self.disk: Dict[str, List[str]] = {"/Game/Maps/MapA": ["Crate"]}
        self.loads: List[str] = []
        self.events: List[Tuple[Any, ...]] = []
        self.refuse_attach = False
        self.rename_fails = False
        self.add_fails: Optional[str] = None
        self.pie = False

    def spawn(self, name: str, world: Optional[World] = None, mobility: Mobility = Mobility.STATIC) -> Actor:
        world = world or self.world
        actor = Actor(self, world.get_path_name() + ":PersistentLevel." + name, world, mobility)
        self.memory[actor.path] = actor
        return actor

    def destroy(self, actor: Actor) -> None:
        """EditorActorSubsystem.destroy_actor: the actor stays loaded, invalid."""
        actor.valid = False
        actor.label = ""

    # Object lookup.
    def find_object(self, outer: Any, name: str) -> Any:
        if outer is None:
            return self.memory.get(name)
        return next((c for c in getattr(outer, "components", []) if c.get_name() == name), None)

    def load_object(self, outer: Any, path: str) -> Any:
        """What load_object did with a path into a map on disk: load the map."""
        package = path.split(".", 1)[0]
        if package in self.disk:
            self.loads.append(package)
            for name in self.disk.pop(package):
                self.spawn(name, self.map_a)
        return self.memory.get(path)

    # SubobjectDataSubsystem, for an actor instance.
    def gather(self, actor: Actor) -> List[SimpleNamespace]:
        return [SimpleNamespace(obj=actor)] + [SimpleNamespace(obj=c) for c in actor.components]

    def add_new_subobject(self, params: SimpleNamespace) -> Tuple[SimpleNamespace, Text]:
        self.events.append(("add", params.new_class.get_name()))
        if self.add_fails is not None:
            return SimpleNamespace(obj=None), Text(self.add_fails)
        parent = params.parent_handle.obj
        actor = parent if isinstance(parent, Actor) else parent.owner
        stem = params.new_class.get_name().replace("Component", "")
        name, n = stem, 0
        while self.find_object(actor, name) is not None:
            n += 1
            name = f"{stem}_{n}"
        component = params.new_class.pytype(name, actor)
        actor.components.append(component)
        if isinstance(component, SceneComponent):
            if actor.root_component is None:
                actor.root_component = component
            else:
                component.attach_parent = parent if isinstance(parent, SceneComponent) else actor.root_component
        return SimpleNamespace(obj=component), Text()

    def rename_subobject(self, handle: SimpleNamespace, name: Text) -> bool:
        self.events.append(("rename", str(name)))
        if self.rename_fails or self.find_object(handle.obj.owner, str(name)) is not None:
            return False
        handle.obj.name = str(name)
        return True

    def delete_subobject(self, context: SimpleNamespace, handle: SimpleNamespace, blueprint: Any) -> int:
        self.events.append(("delete", handle.obj.get_name()))
        context.obj.components.remove(handle.obj)
        return 1

    @contextmanager
    def transaction(self, description: str) -> Iterator[SimpleNamespace]:
        self.events.append(("begin", description))
        try:
            yield SimpleNamespace(cancel=lambda: self.events.append(("cancel", description)))
        finally:
            self.events.append(("end", description))

    def new_object(self, cls: Class) -> Any:
        """unreal.new_object: refuses an abstract class before creating anything."""
        if cls.abstract:
            raise Exception(f"new_object: Class '{cls.get_name()}' is abstract")
        return object()

    def module(self) -> SimpleNamespace:
        editor = self
        library = SimpleNamespace(
            is_handle_valid=lambda handle: handle is not None and handle.obj is not None,
            get_data=lambda handle: handle,
            get_associated_object=lambda data: data.obj,
        )
        subsystem = SimpleNamespace(
            k2_gather_subobject_data_for_instance=self.gather,
            add_new_subobject=self.add_new_subobject,
            rename_subobject=self.rename_subobject,
            delete_subobject=self.delete_subobject,
        )
        registry = SimpleNamespace(
            get_assets_by_package_name=lambda package: [
                SimpleNamespace(asset_class_path=SimpleNamespace(asset_name="World"))
            ]
            if package in ("/Game/Maps/MapA", "/Game/Maps/MapB")
            else []
        )
        subsystems = {
            "LevelEditorSubsystem": SimpleNamespace(is_in_play_in_editor=lambda: editor.pie),
            **editor_subsystems(self.world, pie=lambda: editor.pie, pie_world=self.play_world),
        }
        return SimpleNamespace(
            Actor=Actor,
            StaticMeshActor=StaticMeshActor,
            ActorComponent=ActorComponent,
            SceneComponent=SceneComponent,
            PrimitiveComponent=PrimitiveComponent,
            StaticMeshComponent=StaticMeshComponent,
            PointLightComponent=PointLightComponent,
            SpotLightComponent=SpotLightComponent,
            ProjectileMovementComponent=ProjectileMovementComponent,
            Class=Class,
            Blueprint=type("Blueprint", (), {}),
            Text=Text,
            ComponentMobility=Mobility,
            AttachmentRule=SimpleNamespace(KEEP_WORLD="KeepWorld", KEEP_RELATIVE="KeepRelative"),
            DetachmentRule=SimpleNamespace(KEEP_WORLD="KeepWorld", KEEP_RELATIVE="KeepRelative"),
            AddNewSubobjectParams=lambda **kwargs: SimpleNamespace(**kwargs),
            SubobjectDataSubsystem="SubobjectDataSubsystem",
            SubobjectDataBlueprintFunctionLibrary=library,
            AssetRegistryHelpers=SimpleNamespace(get_asset_registry=lambda: registry),
            MathLibrary=SimpleNamespace(class_is_child_of=class_is_child_of),
            SystemLibrary=SimpleNamespace(is_valid=lambda obj: obj is not None and obj.valid),
            ScopedEditorTransaction=self.transaction,
            LevelEditorSubsystem="LevelEditorSubsystem",
            **SUBSYSTEM_NAMES,
            get_editor_subsystem=subsystems.__getitem__,
            get_engine_subsystem=lambda name: subsystem,
            find_object=self.find_object,
            load_object=self.load_object,
            new_object=self.new_object,
        )

    def run(self, code: str) -> Dict[str, Any]:
        """``ExecutePythonCommandEx`` in ExecuteFile mode."""
        out = io.StringIO()
        try:
            with redirect_stdout(out):
                exec(code, {})
        except Exception as exc:
            return {"ReturnValue": False, "CommandResult": f"{type(exc).__name__}: {exc}"}
        return {
            "ReturnValue": True,
            "LogOutput": [{"Type": "Info", "Output": line} for line in out.getvalue().splitlines()],
        }


@pytest.fixture
def editor(monkeypatch: pytest.MonkeyPatch) -> Editor:
    level = Editor()
    monkeypatch.setitem(sys.modules, "unreal", level.module())
    return level


def _session(editor: Editor, monkeypatch: pytest.MonkeyPatch, settings: Optional[Settings] = None) -> Any:
    session = unreal_runtime.UnrealRuntimeSession(settings or Settings())
    session.scripts = []  # type: ignore[attr-defined]
    session.calls = []  # type: ignore[attr-defined]

    async def execute(code: str, mode: str = "ExecuteFile") -> Dict[str, Any]:
        session.scripts.append(code)  # type: ignore[attr-defined]
        return editor.run(code)

    async def call_function(object_path: str, function_name: str, parameters: Any = None) -> Dict[str, Any]:
        session.calls.append(function_name)  # type: ignore[attr-defined]
        return {"ReturnValue": True}

    monkeypatch.setattr(session, "_execute_python", execute)
    monkeypatch.setattr(session, "_call_function", call_function)
    return session


def _run(editor: Editor, monkeypatch: pytest.MonkeyPatch, method: str, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    session = _session(editor, monkeypatch)
    return asyncio.run(getattr(session, method)(*args, **kwargs))


def _not_found(result: Dict[str, Any], path: str) -> bool:
    """The refusal ``actor_at`` raises for ``path``, as the session reports it: NotFoundError."""
    return (
        result.get("success") is False
        and result.get("error_type") == "NotFoundError"
        and result.get("error", "").startswith(ACTOR_NOT_FOUND.format(path=path) + " (")
    )


# ---------------------------------------------------------------------------
# actor_at: only a live actor of the editor world
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("get_actor_info", {}),
        ("focus_on_actor", {}),
        ("set_actor_transform", {"location": [1.0, 2.0, 3.0]}),
        ("add_component", {"component_class": "PointLightComponent"}),
    ],
)
def test_a_deleted_actor_is_not_found(
    editor: Editor, monkeypatch: pytest.MonkeyPatch, method: str, kwargs: Dict[str, Any]
) -> None:
    """It read back as a live actor with an empty name and a zero scale."""
    crate = editor.spawn("Crate")
    editor.destroy(crate)
    session = _session(editor, monkeypatch)

    result = asyncio.run(getattr(session, method)(crate.path, **kwargs))

    assert _not_found(result, crate.path), result
    assert "no live actor at that path in the level open in the editor, " + MAP_B in result["error"]
    # Nothing was written, and focus moved no camera.
    assert editor.events == [] and session.calls == []


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("get_actor_info", {}),
        ("focus_on_actor", {}),
        ("set_actor_transform", {"location": [999.0, 0.0, 0.0]}),
        ("add_component", {"component_class": "PointLightComponent"}),
    ],
)
def test_a_path_into_a_map_that_is_not_open_is_refused_without_loading_it(
    editor: Editor, monkeypatch: pytest.MonkeyPatch, method: str, kwargs: Dict[str, Any]
) -> None:
    """With MapB open, a MapA path loaded MapA from disk and edited it unseen."""
    path = MAP_A + ":PersistentLevel.Crate"
    session = _session(editor, monkeypatch)

    result = asyncio.run(getattr(session, method)(path, **kwargs))

    assert _not_found(result, path), result
    assert editor.loads == [] and "/Game/Maps/MapA" in editor.disk
    assert editor.events == [] and session.calls == []


def test_an_actor_of_a_map_loaded_but_not_open_is_refused(editor: Editor, monkeypatch: pytest.MonkeyPatch) -> None:
    """A map's actors can stay loaded after it closes; they are still not the open level's."""
    stale = editor.spawn("Crate", editor.map_a)

    result = _run(editor, monkeypatch, "set_actor_transform", stale.path, location=[999.0, 0.0, 0.0])

    assert _not_found(result, stale.path), result
    assert "it belongs to " + MAP_A + ", not to the level open in the editor, " + MAP_B in result["error"]
    assert editor.events == []


def test_the_running_games_copy_is_found_only_when_a_script_asks_for_it(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The physics and collision edits change the running copy on purpose
    (actor_at(path, play_world=True)); every other lookup refuses it."""
    editor.pie = True
    copy = editor.spawn("Crate", editor.play_world)
    session = _session(editor, monkeypatch)
    lookup = ACTOR_HELPERS + (
        f"\nprint(json.dumps({{'path': actor_at({copy.path!r}, play_world=True).get_path_name()}}))"
    )

    found = asyncio.run(session._execute_json_script(lookup))
    refused = asyncio.run(session.set_actor_transform(copy.path, location=[1.0, 2.0, 3.0]))

    assert found == {"path": copy.path}
    assert _not_found(refused, copy.path)


def test_the_editor_world_is_found_during_pie(editor: Editor, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unreal hands out no editor world during PIE; editor_world() finds it all the same."""
    crate = editor.spawn("Crate")
    editor.pie = True

    added = _run(editor, monkeypatch, "add_component", crate.path, "PointLightComponent")

    assert added["component_class"] == "PointLightComponent", added


# ---------------------------------------------------------------------------
# set_actor_parent
# ---------------------------------------------------------------------------


def test_attaching_keeps_the_world_transform_in_one_undoable_transaction(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    child, parent, old = editor.spawn("Child"), editor.spawn("Parent"), editor.spawn("Old")
    child.parent = old

    result = _run(editor, monkeypatch, "set_actor_parent", child.path, parent.path)

    assert result == {
        "actor_path": child.path,
        "parent_path": parent.path,
        "previous_parent_path": old.path,
        "changed": True,
    }
    assert child.parent is parent
    assert editor.events == [
        ("begin", "simul: attach actor"),
        ("modify", "Child"),
        ("modify", "Parent"),
        ("modify", "Old"),
        ("attach", "Child", "Parent", ("KeepWorld", "KeepWorld", "KeepWorld")),
        ("end", "simul: attach actor"),
    ]


def test_attaching_to_the_current_parent_changes_nothing(editor: Editor, monkeypatch: pytest.MonkeyPatch) -> None:
    child, parent = editor.spawn("Child"), editor.spawn("Parent")
    child.parent = parent

    result = _run(editor, monkeypatch, "set_actor_parent", child.path, parent.path)

    assert (result["parent_path"], result["previous_parent_path"], result["changed"]) == (
        parent.path,
        parent.path,
        False,
    )
    assert editor.events == []


def test_an_actor_cannot_be_its_own_parent(editor: Editor, monkeypatch: pytest.MonkeyPatch) -> None:
    """It used to report success while K2_AttachToActor returned false."""
    crate = editor.spawn("Crate")

    result = _run(editor, monkeypatch, "set_actor_parent", crate.path, crate.path)

    assert result["success"] is False and result["error_type"] == "ValidationError"
    assert result["error"] == "An actor cannot be attached to itself: " + crate.path
    assert editor.events == [] and crate.parent is None


def test_a_cycle_is_refused_with_its_reason(editor: Editor, monkeypatch: pytest.MonkeyPatch) -> None:
    top, middle, bottom = editor.spawn("Top"), editor.spawn("Middle"), editor.spawn("Bottom")
    middle.parent, bottom.parent = top, middle

    result = _run(editor, monkeypatch, "set_actor_parent", top.path, bottom.path)

    assert result["success"] is False and result["error_type"] == "ValidationError"
    assert result["error"] == (
        f"Attaching {top.path} to {bottom.path} would make a cycle: {bottom.path} is attached under "
        f"{top.path}; detach {bottom.path} first"
    )
    assert editor.events == [] and top.parent is None


def test_a_missing_parent_is_not_found_and_the_child_keeps_its_parent(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    child, parent = editor.spawn("Child"), editor.spawn("Parent")
    child.parent = parent
    missing = LEVEL + "Nope"

    result = _run(editor, monkeypatch, "set_actor_parent", child.path, missing)

    assert _not_found(result, missing), result
    assert child.parent is parent and editor.events == []


def test_a_static_actor_is_not_attached_under_a_movable_one(editor: Editor, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unreal refuses it and returns false without saying why; the reply says why."""
    child, parent = editor.spawn("Wall"), editor.spawn("Cart", mobility=Mobility.MOVABLE)

    result = _run(editor, monkeypatch, "set_actor_parent", child.path, parent.path)

    assert result["success"] is False and result["error_type"] == "InvalidState"
    assert f"{child.path} is Static and {parent.path} is Movable" in result["error"]
    assert "Movable first" in result["error"]
    assert editor.events == [] and child.parent is None


def test_an_attachment_unreal_refuses_is_an_error_and_its_transaction_is_cancelled(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    child, parent = editor.spawn("Child"), editor.spawn("Parent")
    editor.refuse_attach = True

    result = _run(editor, monkeypatch, "set_actor_parent", child.path, parent.path)

    assert result["success"] is False and result["error_type"] == "EngineError"
    assert result["error"] == f"Unreal did not attach {child.path} to {parent.path}; the editor log says why"
    assert ("cancel", "simul: attach actor") in editor.events and child.parent is None


def test_detaching_is_explicit_and_undoable(editor: Editor, monkeypatch: pytest.MonkeyPatch) -> None:
    child, parent = editor.spawn("Child"), editor.spawn("Parent")
    child.parent = parent

    detached = _run(editor, monkeypatch, "set_actor_parent", child.path, detach=True)
    again = _run(editor, monkeypatch, "set_actor_parent", child.path, detach=True)

    assert detached == {
        "actor_path": child.path,
        "parent_path": None,
        "previous_parent_path": parent.path,
        "changed": True,
    }
    assert again == {"actor_path": child.path, "parent_path": None, "previous_parent_path": None, "changed": False}
    assert editor.events == [
        ("begin", "simul: detach actor"),
        ("modify", "Child"),
        ("modify", "Parent"),
        ("detach", "Child", ("KeepWorld", "KeepWorld", "KeepWorld")),
        ("end", "simul: detach actor"),
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "Pass parent_path to attach the actor, or detach=True to detach it from its parent"),
        (
            {"parent_path": LEVEL + "Parent", "detach": True},
            "Pass parent_path to attach the actor, or detach=True to detach it from its parent, not both",
        ),
    ],
)
def test_the_session_needs_a_parent_or_detach(
    editor: Editor, monkeypatch: pytest.MonkeyPatch, kwargs: Dict[str, Any], message: str
) -> None:
    """An omitted parent used to detach silently."""
    child = editor.spawn("Child")
    session = _session(editor, monkeypatch)

    result = asyncio.run(session.set_actor_parent(child.path, **kwargs))

    assert result == {"success": False, "error_type": "ValidationError", "error": message}
    assert session.scripts == []


# ---------------------------------------------------------------------------
# add_component
# ---------------------------------------------------------------------------


def test_a_component_is_added_under_the_root_named_and_read_back(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every input used to fail: Remote Control refuses AddComponentByClass."""
    crate = editor.spawn("Crate")

    result = _run(editor, monkeypatch, "add_component", crate.path, "PointLightComponent", "Lamp")

    assert result == {
        "actor_path": crate.path,
        "component_path": crate.path + ".Lamp",
        "component_name": "Lamp",
        "component_class": "PointLightComponent",
        "parent_component": "StaticMeshComponent0",
        "is_root": False,
    }
    assert editor.events == [
        ("begin", "simul: add component"),
        ("modify", "Crate"),
        ("add", "PointLightComponent"),
        ("rename", "Lamp"),
        ("end", "simul: add component"),
    ]


def test_a_class_path_and_a_parent_component_are_honoured(editor: Editor, monkeypatch: pytest.MonkeyPatch) -> None:
    crate = editor.spawn("Crate")
    lamp = PointLightComponent("Lamp", crate)
    lamp.attach_parent = crate.root_component
    crate.components.append(lamp)
    editor.memory["/Script/Engine.SpotLightComponent"] = SpotLightComponent.CLASS  # type: ignore[assignment]

    result = _run(
        editor,
        monkeypatch,
        "add_component",
        crate.path,
        "/Script/Engine.SpotLightComponent",
        parent_component="Lamp",
    )

    assert (result["component_name"], result["component_class"], result["parent_component"]) == (
        "SpotLight",
        "SpotLightComponent",
        "Lamp",
    )


def test_a_component_that_is_not_a_scene_component_attaches_to_nothing(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    crate = editor.spawn("Crate")

    added = _run(editor, monkeypatch, "add_component", crate.path, "ProjectileMovementComponent", "Mover")
    refused = _run(
        editor, monkeypatch, "add_component", crate.path, "ProjectileMovementComponent", parent_component="Mover"
    )

    assert (added["parent_component"], added["is_root"]) == (None, False)
    assert refused["success"] is False and refused["error_type"] == "ValidationError"
    assert "not a scene component" in refused["error"]


@pytest.mark.parametrize(
    ("component_class", "error_type", "message"),
    [
        ("PrimitiveComponent", "ValidationError", "PrimitiveComponent is abstract; pass a concrete component class"),
        (
            "StaticMeshActor",
            "ValidationError",
            "StaticMeshActor is the class StaticMeshActor, not a component class (an ActorComponent)",
        ),
        ("NoSuchComponent", "NotFoundError", "No class named 'NoSuchComponent'"),
        ("/Game/Maps/MapA.MapA", "NotFoundError", "/Game/Maps/MapA.MapA is in a map package, not a class"),
        (MAP_A + ":PersistentLevel.Crate", "NotFoundError", "names an object inside a package, not a class"),
    ],
)
def test_a_class_that_cannot_be_added_is_refused_before_anything_is_added(
    editor: Editor, monkeypatch: pytest.MonkeyPatch, component_class: str, error_type: str, message: str
) -> None:
    """An abstract class added through the subsystem fires an engine ensure on UE 5.7.4."""
    crate = editor.spawn("Crate")

    result = _run(editor, monkeypatch, "add_component", crate.path, component_class)

    assert result["success"] is False and result["error_type"] == error_type, result
    assert message in result["error"]
    assert editor.events == [] and editor.loads == []


def test_a_name_already_used_in_the_actor_is_refused_before_adding(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    crate = editor.spawn("Crate")

    result = _run(editor, monkeypatch, "add_component", crate.path, "PointLightComponent", "StaticMeshComponent0")

    assert result["success"] is False and result["error_type"] == "ValidationError"
    assert "already has a component or subobject named 'StaticMeshComponent0'" in result["error"]
    assert editor.events == []


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("", "component_name must not be empty"),
        ("Desk Lamp", "component_name 'Desk Lamp' has characters Unreal refuses in an object name: ' '"),
        ("a.b", "has characters Unreal refuses in an object name: '.'"),
        ("None", "Unreal reserves for no name"),
        ("x" * 1024, "component_name is 1024 characters long; Unreal keeps at most 1023"),
    ],
)
def test_a_name_unreal_cannot_take_is_refused_before_the_editor_is_asked(
    editor: Editor, monkeypatch: pytest.MonkeyPatch, name: str, message: str
) -> None:
    crate = editor.spawn("Crate")
    session = _session(editor, monkeypatch)

    result = asyncio.run(session.add_component(crate.path, "PointLightComponent", name))

    assert result["success"] is False and result["error_type"] == "ValidationError"
    assert message in result["error"]
    assert session.scripts == []


def test_a_missing_parent_component_is_not_found(editor: Editor, monkeypatch: pytest.MonkeyPatch) -> None:
    crate = editor.spawn("Crate")

    result = _run(editor, monkeypatch, "add_component", crate.path, "PointLightComponent", parent_component="Nope")

    assert result["success"] is False and result["error_type"] == "NotFoundError"
    assert result["error"].startswith("Component not found: Nope")
    assert editor.events == []


def test_a_component_unreal_does_not_name_is_removed_and_its_transaction_cancelled(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    crate = editor.spawn("Crate")
    editor.rename_fails = True

    result = _run(editor, monkeypatch, "add_component", crate.path, "PointLightComponent", "Lamp")

    assert result["success"] is False and result["error_type"] == "EngineError"
    assert "did not name the new PointLightComponent 'Lamp'" in result["error"]
    assert ("delete", "PointLight") in editor.events and ("cancel", "simul: add component") in editor.events
    assert [c.get_name() for c in crate.components] == ["StaticMeshComponent0"]


def test_a_component_unreal_does_not_add_is_an_error(editor: Editor, monkeypatch: pytest.MonkeyPatch) -> None:
    crate = editor.spawn("Crate")
    editor.add_fails = "Cannot add a subobject of class 'Point Light Component'"

    result = _run(editor, monkeypatch, "add_component", crate.path, "PointLightComponent")

    assert result["success"] is False and result["error_type"] == "EngineError"
    assert result["error"].endswith(": Cannot add a subobject of class 'Point Light Component'")
    assert ("cancel", "simul: add component") in editor.events


# ---------------------------------------------------------------------------
# Through the MCP tools
# ---------------------------------------------------------------------------


class _Adapter(AvailableAdapter):
    def __init__(self, session: Any) -> None:
        super().__init__(session.settings)
        self.session = session

    @contextmanager
    def create_session(self) -> Iterator[Any]:
        yield self.session


def _call(editor: Editor, monkeypatch: pytest.MonkeyPatch, tmp_path: Any, tool: str, arguments: Dict[str, Any]) -> Any:
    settings = Settings(
        unreal={"tool_surface": "full", "attachment_path": str(tmp_path / "attachment.json")},
        security={"rate_limiting_enabled": False},
    )
    instance = server_module.SimulMCPServer(settings, backends={"unreal"})
    instance.usage_tracker = ToolUsageTracker(log_dir=tmp_path / "usage")
    instance.unreal_adapter = _Adapter(_session(editor, monkeypatch, settings))

    async def run() -> Any:
        async with Client(instance.mcp) as client:
            return await client.call_tool(tool, arguments, raise_on_error=False)

    result = asyncio.run(run())
    return json.loads([block.text for block in result.content if getattr(block, "type", None) == "text"][-1])


def test_the_parent_tool_detaches_only_when_asked(
    editor: Editor, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    child, parent = editor.spawn("Child"), editor.spawn("Parent")
    child.parent = parent

    unasked = _call(
        editor, monkeypatch, tmp_path, "set_unreal_actor_parent", {"actor_path": child.path, "parent_path": ""}
    )
    asked = _call(
        editor,
        monkeypatch,
        tmp_path,
        "set_unreal_actor_parent",
        {"actor_path": child.path, "parent_path": "", "detach": True},
    )

    assert unasked["success"] is False and unasked["error_type"] == "ValidationError"
    assert "detach=true" in unasked["error"]
    assert asked["success"] is True
    assert (asked["parent_path"], asked["previous_parent_path"], asked["changed"]) == (None, parent.path, True)
    assert child.parent is None


def test_the_component_tool_returns_what_was_read_back(
    editor: Editor, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    crate = editor.spawn("Crate")

    result = _call(
        editor,
        monkeypatch,
        tmp_path,
        "add_unreal_component",
        {"actor_path": crate.path, "component_class": "PointLightComponent", "component_name": "Lamp"},
    )

    assert result["success"] is True
    assert (result["component_path"], result["component_name"], result["parent_component"], result["is_root"]) == (
        crate.path + ".Lamp",
        "Lamp",
        "StaticMeshComponent0",
        False,
    )
