"""A UE-shaped fake editor for the actor tool scripts, as UE 5.7.4 answers them.

The scripts in ``_unreal_scripts`` (``ACTOR_HELPERS``' ``actor_at``, parenting,
components, visibility, delete, focus) run in-process against the fake
``unreal`` module ``Editor.module()`` builds:

- ``load_object`` on a path into a map on disk loads that map, as the old
  lookup did; ``find_object`` finds only loaded objects, destroyed actors
  included until garbage collection (``SystemLibrary.is_valid`` is False).
- The open map is MapB; MapA is on disk.
- Attachments follow Unreal's rules: no cycle, no Static actor under a parent
  that is not Static.
- Components are added through ``SubobjectDataSubsystem``.
- The editor hides an actor three ways (the editor's temporary hide, hidden at
  editor startup, hidden in game), selects no hidden actor, and runs the
  ``ACTOR HIDE / UNHIDE SELECTED STARTUP`` console commands on the selection.
- The level listing leaves out the WorldSettings and the builder brush.

Every change is recorded in ``Editor.events``, with the transactions and the
``modify`` calls around it, so a test can check that a change is undoable.
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
from simul.adapters._unreal_scripts import ACTOR_NOT_FOUND
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
        # The three ways the editor hides an actor (UE 5.7.4).
        self.hidden_in_editor = False
        self.hidden_at_startup = False
        self.hidden_in_game = False
        # False for an actor Unreal will not select for another reason (a
        # locked level, say); the editor never selects a hidden one.
        self.selectable = True
        # False for an actor the level listing leaves out (WorldSettings, the
        # builder brush, helpers the Outliner hides).
        self.listed = True

    def get_path_name(self) -> str:
        return self.path

    def get_editor_property(self, name: str) -> Any:
        return {"hidden": self.hidden_in_game}[name]

    def is_temporarily_hidden_in_editor(self) -> bool:
        return self.hidden_in_editor

    def is_hidden_ed_at_startup(self) -> bool:
        return self.hidden_at_startup

    def set_is_temporarily_hidden_in_editor(self, hidden: bool) -> None:
        self.editor.events.append(("hide in editor", self.label, hidden))
        self.hidden_in_editor = hidden

    def set_actor_hidden_in_game(self, hidden: bool) -> None:
        self.editor.events.append(("hide in game", self.label, hidden))
        self.hidden_in_game = hidden

    def get_attached_actors(
        self, reset_array: bool = True, recursively_include_attached_actors: bool = False
    ) -> List["Actor"]:
        found: List[Actor] = []
        for actor in self.editor.memory.values():
            if actor.parent is self and actor.valid:
                found.append(actor)
                if recursively_include_attached_actors:
                    found.extend(actor.get_attached_actors(True, True))
        return found

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
WorldSettings = type("WorldSettings", (Actor,), {"CLASS": Class("WorldSettings", Actor.CLASS)})
Brush = type("Brush", (Actor,), {"CLASS": Class("Brush", Actor.CLASS)})


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
        self.refuse_destroy = False
        self.selection: List[Actor] = []
        self.rename_fails = False
        self.add_fails: Optional[str] = None
        self.pie = False

    def spawn(
        self,
        name: str,
        world: Optional[World] = None,
        mobility: Mobility = Mobility.STATIC,
        cls: Any = None,
    ) -> Actor:
        world = world or self.world
        actor = (cls or Actor)(self, world.get_path_name() + ":PersistentLevel." + name, world, mobility)
        self.memory[actor.path] = actor
        return actor

    def destroy(self, actor: Actor) -> None:
        """EditorActorSubsystem.destroy_actor: the actor stays loaded, invalid."""
        actor.valid = False
        actor.label = ""

    # EditorActorSubsystem.
    def get_all_level_actors(self) -> List[Actor]:
        if self.pie:
            return []
        return [a for a in self.memory.values() if a.valid and a.world is self.world and a.listed]

    def destroy_actor(self, actor: Actor) -> bool:
        if self.pie or self.refuse_destroy:
            return False
        self.events.append(("destroy", actor.label))
        self.destroy(actor)
        return True

    def get_selected_level_actors(self) -> List[Actor]:
        return [] if self.pie else list(self.selection)

    def set_selected_level_actors(self, actors: List[Actor]) -> None:
        """Replaces the selection; the editor selects no hidden or unselectable actor."""
        self.events.append(("select", [a.label for a in actors]))
        self.selection = [a for a in actors if a.selectable and not a.hidden_in_editor]

    def execute_console_command(self, world: Any, command: str) -> None:
        """The level editor's ACTOR HIDE / UNHIDE SELECTED STARTUP: the selection's startup flag."""
        startup = {"ACTOR HIDE SELECTED STARTUP": True, "ACTOR UNHIDE SELECTED STARTUP": False}
        if command not in startup:
            raise AssertionError("Unexpected console command: " + command)
        for actor in self.selection:
            self.events.append(("hide at startup", actor.label, startup[command]))
            actor.hidden_at_startup = startup[command]

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
            "EditorActorSubsystem": self,
            "LevelEditorSubsystem": SimpleNamespace(is_in_play_in_editor=lambda: editor.pie),
            **editor_subsystems(self.world, pie=lambda: editor.pie, pie_world=self.play_world),
        }
        return SimpleNamespace(
            Actor=Actor,
            StaticMeshActor=StaticMeshActor,
            WorldSettings=WorldSettings,
            Brush=Brush,
            EditorActorSubsystem="EditorActorSubsystem",
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
            SystemLibrary=SimpleNamespace(
                is_valid=lambda obj: obj is not None and obj.valid,
                execute_console_command=self.execute_console_command,
            ),
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


def install_editor(monkeypatch: pytest.MonkeyPatch) -> Editor:
    """A fresh editor, installed as the ``unreal`` module for the test."""
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
