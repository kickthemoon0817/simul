"""set_unreal_actor_property against a simulated UE 5.7 editor: whole, finished, undoable, or rolled back.

The fake editor is a real HTTP server with Remote Control's property, call and
batch routes. Python requests run, as in the editor, in one shared console dict
against a fake ``unreal`` module. What it reproduces is what made the write go
wrong on UE 5.7.4:

* Remote Control writes an array by index and never shrinks it, and a
  property with a setter (MaterialBillboardComponent's Elements) gets the new
  elements appended to the old ones (RemoteControlModule.cpp,
  FindSetterFunctionInternal_TMP).
* A write through a setter (Elements, StaticMesh, Mobility, RelativeLocation,
  a light's Intensity) is a function call with no transaction of its own, so
  Remote Control never calls Modify: the object is not recorded for undo and
  the level is not marked changed (CreateRCCall, InvokeCall with
  ERCTransactionMode::NONE). Any other write runs PreEditChange, which calls
  Modify and marks the level changed.
* An error body is printed with its message unescaped, and a batch answer
  embeds it verbatim: a message with a line break makes the answer invalid
  JSON (CreateUTF8ErrorMessage, HandleBatchRequest).
* The write's PostEditChangeProperty runs only when Remote Control's next
  function call arrives (TestOrFinalizeOngoingChange, with
  RemoteControl.EnableOngoingChangeOptimization on), so a read straight after
  the write sees what Unreal has not yet corrected (OverrideMaterials past the
  mesh's slots, StaticMeshComponent::CleanUpOverrideMaterials).
* An object path that loads nothing is written as None.
* get_editor_property returns a container that is the property itself, and an
  edit made through it skips the undo snapshot.
* Undo restores each object to its state at its first Modify in the
  transaction; Cancel drops the record without reverting anything.
"""

from __future__ import annotations

import asyncio
import copy
import io
import json
import sys
import traceback
from contextlib import asynccontextmanager, redirect_stdout
from types import SimpleNamespace
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

import pytest
from aiohttp import web

from simul.adapters import unreal_runtime
from simul.adapters.unreal_runtime import PROPERTY_TRANSACTIONS, UnrealRuntimeSession
from simul.config import Settings

LEVEL = "/Game/Maps/Lab.Lab:PersistentLevel."
ACTOR = LEVEL + "Props_1"
MESH = ACTOR + ".StaticMeshComponent0"
BILLBOARD = ACTOR + ".MaterialBillboard"
LAMP = LEVEL + "PointLight_1"
LIGHT = LAMP + ".LightComponent0"
SPHERE, CUBE = "/Engine/BasicShapes/Sphere.Sphere", "/Engine/BasicShapes/Cube.Cube"
MATERIAL_A, MATERIAL_B = "/Engine/M_A.M_A", "/Engine/M_B.M_B"
ELEMENT = {"Material": "", "BaseSizeX": 32, "BaseSizeY": 32, "bSizeIsInScreenSpace": False}


# -- the fake ``unreal`` module --------------------------------------------------------


class FakeArray(list):
    def copy(self) -> "FakeArray":  # type: ignore[override]
        return FakeArray(copy.deepcopy(list(self)))


class FakeMap(dict):
    def copy(self) -> "FakeMap":  # type: ignore[override]
        return FakeMap(copy.deepcopy(dict(self)))


class FakeSet(set):
    pass


NEVER = "NEVER"


class Printed(str):
    """An error body as Remote Control prints it: raw text, maybe not JSON."""


class Obj:
    """An object of the fake editor; ``props`` are Remote Control's JSON shapes."""

    def __init__(
        self,
        editor: "Editor",
        path: str,
        cls: str,
        props: Dict[str, Any],
        meta: Dict[str, Dict[str, Any]],
        world: Any = None,
        owner: Optional["Obj"] = None,
    ) -> None:
        self.editor, self.path, self.cls, self.world, self.owner = editor, path, cls, world, owner
        self.props = {name: FakeArray(v) if isinstance(v, list) else v for name, v in props.items()}
        self.meta, self.valid = meta, True

    def get_name(self) -> str:
        return self.path.rsplit(".", 1)[-1].rsplit(":", 1)[-1]

    def get_path_name(self) -> str:
        return self.path

    def get_class(self) -> Any:
        return SimpleNamespace(get_name=lambda: self.cls)

    def get_world(self) -> Any:
        return self.world

    def get_outer(self) -> Any:
        return self.owner

    def get_editor_property(self, name: str) -> Any:
        if name not in self.props or self.meta.get(name, {}).get("python_hidden"):
            raise Exception(f"Failed to find property '{name}'")
        if self.meta.get(name, {}).get("object") and self.props[name] == "":
            return None  # Remote Control reads a null object as "", Python as None
        return self.props[name]  # the property itself, as UE returns a container

    def modify(self, always_mark_dirty: bool = True) -> bool:
        """UObject::Modify: recorded in an open transaction; the level marked changed only when asked."""
        self.editor.modify(self)
        if always_mark_dirty:
            self.editor.dirty = True
        return self.editor.record is not None

    def set_editor_property(self, name: str, value: Any, notify_mode: Any = None) -> None:
        if self.meta.get(name, {}).get("python_readonly"):
            raise Exception(f"Property '{name}' is read-only and cannot be set")
        if value == self.props.get(name):
            return  # identical: Unreal neither notifies nor records (an empty container emptied)
        if notify_mode != NEVER:
            self.modify()  # PreEditChange
        if isinstance(value, list):
            self.props[name] = FakeArray(copy.deepcopy(list(value)))
        elif isinstance(value, dict) and isinstance(self.props.get(name), FakeMap):
            self.props[name] = FakeMap(copy.deepcopy(dict(value)))
        else:
            self.props[name] = copy.deepcopy(value)
        if notify_mode != NEVER:
            self.editor.post_edit(self, name)


class Actor(Obj):
    pass


class ActorComponent(Obj):
    def get_owner(self) -> Optional[Obj]:
        return self.owner


class Editor:
    """Remote Control and the editor's Python, over objects, an undo buffer and a pending edit."""

    def __init__(self) -> None:
        self.world = SimpleNamespace(
            get_path_name=lambda: "/Game/Maps/Lab.Lab",
            get_outermost=lambda: SimpleNamespace(get_name=lambda: "/Game/Maps/Lab"),
        )
        self.pie_world = SimpleNamespace(
            get_path_name=lambda: "/Game/Maps/UEDPIE_0_Lab.Lab",
            get_outermost=lambda: SimpleNamespace(get_name=lambda: "/Game/Maps/UEDPIE_0_Lab"),
        )
        self.pie = False
        self.objects: Dict[str, Obj] = {}
        self.assets = {SPHERE, CUBE, MATERIAL_A, MATERIAL_B}
        self.depth, self.record, self.history = 0, None, []  # type: Tuple[int, Optional[Dict[str, Any]], List[Any]]
        self.pending: List[Tuple[Obj, str]] = []
        self.writes: List[str] = []
        self.console: Dict[str, Any] = {"__name__": "__main__"}
        self.drop_batch_answer = False
        self.dirty = False  # the level's package

    # -- undo -----------------------------------------------------------------------
    def modify(self, obj: Obj) -> None:
        if self.record is not None and obj.path not in self.record:
            self.record[obj.path] = copy.deepcopy(obj.props)

    def begin(self) -> int:
        index = self.depth
        self.depth += 1
        if index == 0:
            self.record = {}
        return index

    def end(self) -> int:
        active = self.depth
        if self.depth:
            self.depth -= 1
            if self.depth == 0:
                if self.record:
                    self.history.append(self.record)
                self.record = None
        return active

    def cancel(self, index: int) -> None:
        if self.depth:
            self.depth, self.record = 0, None

    def undo(self) -> None:
        for path, props in self.history.pop().items():
            self.objects[path].props = copy.deepcopy(props)

    def post_edit(self, obj: Obj, name: str) -> None:
        hook = obj.meta.get(name, {}).get("post_edit")
        if hook is not None:
            hook(obj, name)

    # -- the objects ----------------------------------------------------------------
    def add(self, obj: Obj) -> Obj:
        self.objects[obj.path] = obj
        return obj

    def module(self) -> SimpleNamespace:
        editor = self
        system = SimpleNamespace(
            is_valid=lambda obj: obj is not None and obj.valid,
            begin_transaction=lambda context, description, obj: editor.begin(),
            end_transaction=lambda: editor.end(),
            cancel_transaction=lambda index: editor.cancel(index),
        )
        subsystems = {
            "UnrealEditorSubsystem": SimpleNamespace(
                get_editor_world=lambda: None if editor.pie else editor.world,
                get_game_world=lambda: editor.pie_world if editor.pie else None,
            ),
            "LevelEditorSubsystem": SimpleNamespace(is_in_play_in_editor=lambda: editor.pie),
            "LayersSubsystem": SimpleNamespace(get_world=lambda: editor.world),
            "EditorActorSubsystem": SimpleNamespace(get_all_level_actors=lambda: []),
        }
        return SimpleNamespace(
            find_object=lambda outer, path: editor.objects.get(path),
            SystemLibrary=system,
            Actor=Actor,
            ActorComponent=ActorComponent,
            Array=FakeArray,
            Set=FakeSet,
            Map=FakeMap,
            PropertyAccessChangeNotifyMode=SimpleNamespace(NEVER=NEVER),
            get_editor_subsystem=subsystems.__getitem__,
            UnrealEditorSubsystem="UnrealEditorSubsystem",
            LevelEditorSubsystem="LevelEditorSubsystem",
            LayersSubsystem="LayersSubsystem",
            EditorActorSubsystem="EditorActorSubsystem",
        )

    # -- Remote Control ---------------------------------------------------------------
    def call(self, body: Dict[str, Any]) -> Tuple[int, Any]:
        # A function call first finishes the pending property write.
        for obj, name in self.pending:
            self.post_edit(obj, name)
        self.pending = []
        command = body["parameters"]["PythonCommand"]
        output = io.StringIO()
        sys.modules["unreal"] = self.module()  # type: ignore[assignment]
        try:
            with redirect_stdout(output):
                exec(compile(command, "<string>", "exec"), self.console)  # noqa: S102 - the editor's Python
        except Exception:
            return 200, {"ReturnValue": False, "CommandResult": traceback.format_exc(), "LogOutput": []}
        lines = [{"Type": "Info", "Output": line} for line in output.getvalue().splitlines()]
        return 200, {"ReturnValue": True, "CommandResult": "None", "LogOutput": lines}

    def property(self, body: Dict[str, Any]) -> Tuple[int, Any]:
        obj = self.objects.get(body["objectPath"])
        name = body.get("propertyName", "")
        if obj is None:
            return 400, {
                "errorMessage": f"Object: {body['objectPath']} does not exist when trying to resolve property: {name}"
            }
        if name not in obj.props:
            return 400, {"errorMessage": f"Object property: {name} could not be resolved on object: {obj.path}"}
        meta = obj.meta.get(name, {})
        if body["access"] == "READ_ACCESS":
            return 200, {
                name: copy.deepcopy(list(obj.props[name]) if isinstance(obj.props[name], list) else obj.props[name])
            }
        if meta.get("rc_refuses"):
            return 400, {
                "errorMessage": "Property is not writeable because it is protected without the AllowPrivateAccess "
                "specifier set to true or a setter assigned."
            }
        if meta.get("rc_refuses_printed"):
            return 400, Printed(
                '{ "errorMessage": "Property is not writeable because it is protected\nwithout the '
                'AllowPrivateAccess specifier set to true or a setter assigned." }'
            )
        self.writes.append(name)
        if not meta.get("setter"):
            obj.modify()  # Remote Control's PreEditChange; the setter path calls no Modify
        value = body["propertyValue"][name]
        current = obj.props[name]
        if isinstance(current, list):
            defaults = meta.get("element", {})
            new = [dict(defaults, **v) if isinstance(v, dict) else v for v in value]
            if meta.get("setter"):
                obj.props[name] = FakeArray(list(current) + new)
            else:
                merged = FakeArray(current)
                for index, element in enumerate(value):
                    if index < len(merged):
                        merged[index] = dict(merged[index], **element) if isinstance(element, dict) else element
                    else:
                        merged.append(new[index])
                obj.props[name] = merged
            if meta.get("objects"):
                obj.props[name] = FakeArray(v if v in self.assets else "" for v in obj.props[name])
        elif meta.get("object"):
            obj.props[name] = value if value in self.assets else ""
        elif meta.get("enum"):
            if value in meta["enum"]:
                obj.props[name] = value
        elif isinstance(current, dict):
            obj.props[name] = dict(current, **value)
        else:
            obj.props[name] = value
        self.pending.append((obj, name))
        return 200, {}

    def dispatch(self, route: str, body: Dict[str, Any]) -> Tuple[int, Any]:
        if route == "/remote/object/call":
            return self.call(body)
        if route == "/remote/object/property":
            return self.property(body)
        if route == "/remote/batch":
            # Each part's body goes in verbatim, as HandleBatchRequest writes it.
            parts = []
            for request in body["Requests"]:
                code, answer = self.dispatch(request["URL"], request["Body"])
                text = answer if isinstance(answer, Printed) else json.dumps(answer)
                parts.append(f'{{"RequestId":{request["RequestId"]},"ResponseCode":{code},"ResponseBody":{text}}}')
            return 200, Printed('{"Responses":[' + ",".join(parts) + "]}")
        return 404, {"errorMessage": "Route not found"}

    async def __call__(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        await asyncio.sleep(0)  # let a concurrent call's request in between, as the network does
        code, answer = self.dispatch(request.path, body)  # one request, a batch whole, at a time
        if request.path == "/remote/batch" and self.drop_batch_answer:
            self.drop_batch_answer = False
            request.transport.close()  # the batch ran; its answer is lost
            raise ConnectionResetError
        if isinstance(answer, Printed):
            return web.Response(text=str(answer), status=code, content_type="application/json")
        return web.json_response(answer, status=code)

    def open_transactions(self) -> Any:
        registry = sys.modules.get(PROPERTY_TRANSACTIONS)
        return getattr(registry, "open", {})


def _scene(editor: Editor) -> None:
    actor = editor.add(
        Actor(
            editor,
            ACTOR,
            "StaticMeshActor",
            {"Tags": ["seed"], "bHidden": False, "Settings": FakeMap({"k": 1}), "PrintedRefusal": ["a", "b"]},
            {"PrintedRefusal": {"rc_refuses_printed": True}},
            editor.world,
        )
    )

    def trim(obj: Obj, name: str) -> None:  # CleanUpOverrideMaterials: one slot on a sphere
        if len(obj.props[name]) > 1:
            obj.props[name] = FakeArray(obj.props[name][:1])

    editor.add(
        ActorComponent(
            editor,
            MESH,
            "StaticMeshComponent",
            {
                "StaticMesh": SPHERE,
                "OverrideMaterials": [],
                "Mobility": "Static",
                "RelativeLocation": {"X": 0, "Y": 0, "Z": 0},
                "ProtectedList": ["kept"],
            },
            {
                "StaticMesh": {"object": True, "setter": True},
                "OverrideMaterials": {"objects": True, "post_edit": trim},
                "Mobility": {"enum": ["Static", "Stationary", "Movable"], "setter": True},
                "RelativeLocation": {"setter": True},
                "ProtectedList": {"rc_refuses": True},
            },
            owner=actor,
        )
    )
    lamp = editor.add(Actor(editor, LAMP, "PointLight", {"bHidden": False}, {}, editor.world))
    editor.add(
        ActorComponent(
            editor, LIGHT, "PointLightComponent", {"Intensity": 5000.0}, {"Intensity": {"setter": True}}, owner=lamp
        )
    )
    editor.add(
        ActorComponent(
            editor,
            BILLBOARD,
            "MaterialBillboardComponent",
            {"Elements": []},
            {"Elements": {"setter": True, "element": ELEMENT}},
            owner=actor,
        )
    )


@asynccontextmanager
async def editor_session(**unreal: Any) -> AsyncIterator[Tuple[UnrealRuntimeSession, Editor]]:
    editor = Editor()
    _scene(editor)

    async def route(request: web.Request) -> web.StreamResponse:
        return await editor(request)

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", route)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    base = Settings()
    settings = base.model_copy(
        update={
            "unreal": base.unreal.model_copy(
                update={
                    "host": "127.0.0.1",
                    "port": site._server.sockets[0].getsockname()[1],
                    "timeout": 5.0,
                    "max_retries": 0,
                    **unreal,
                }
            )
        }
    )
    session = UnrealRuntimeSession(settings)
    try:
        yield session, editor
    finally:
        await session.close()
        await runner.cleanup()
        sys.modules.pop("unreal", None)
        sys.modules.pop(PROPERTY_TRANSACTIONS, None)


def elements(editor: Editor) -> List[Any]:
    return [e["BaseSizeX"] for e in editor.objects[BILLBOARD].props["Elements"]]


# -- replaced whole, one undo step each ------------------------------------------------


@pytest.mark.asyncio
async def test_tags_are_replaced_whole_and_each_call_is_one_undo_step() -> None:
    async with editor_session() as (session, editor):
        for value in ('["a","b","c"]', '["only"]', "[]"):
            result = await session.set_actor_property(ACTOR, "Tags", value)
            assert result["applied"] is True and result["value"] == json.loads(value), result
            assert editor.objects[ACTOR].props["Tags"] == json.loads(value)
        assert result["previous_value"] == ["only"]

        assert len(editor.history) == 3
        editor.undo()
        assert editor.objects[ACTOR].props["Tags"] == ["only"]
        editor.undo()
        assert editor.objects[ACTOR].props["Tags"] == ["a", "b", "c"]
        editor.undo()
        assert editor.objects[ACTOR].props["Tags"] == ["seed"]
        assert editor.depth == 0 and editor.open_transactions() == {}


@pytest.mark.asyncio
async def test_a_struct_array_with_a_setter_is_replaced_whole() -> None:
    """Remote Control's setter path appended: [1,2,3] then [4,5,6] held six, then [7] held [1]."""
    async with editor_session() as (session, editor):
        for value, expected in (([1, 2, 3], [1, 2, 3]), ([4, 5, 6], [4, 5, 6]), ([7], [7]), ([], [])):
            result = await session.set_actor_property(
                BILLBOARD, "Elements", json.dumps([{"BaseSizeX": v} for v in value])
            )
            assert result["applied"] is True, result
            assert elements(editor) == expected
        # The fields not given are the struct's defaults, as in a new element.
        await session.set_actor_property(BILLBOARD, "Elements", '[{"BaseSizeX": 9}]')
        assert editor.objects[BILLBOARD].props["Elements"] == [dict(ELEMENT, BaseSizeX=9)]

        editor.undo()
        assert elements(editor) == []
        editor.undo()
        assert elements(editor) == [7]
        editor.undo()
        assert elements(editor) == [4, 5, 6]


@pytest.mark.asyncio
async def test_without_a_transaction_the_write_is_whole_and_leaves_no_undo_step() -> None:
    async with editor_session() as (session, editor):
        result = await session.set_actor_property(ACTOR, "Tags", '["a"]', generate_transaction=False)

        assert result["applied"] is True and editor.objects[ACTOR].props["Tags"] == ["a"]
        assert editor.history == [] and editor.open_transactions() == {}


# -- the read-back sees what Unreal finished, and a refusal leaves the level intact ------


@pytest.mark.asyncio
async def test_the_read_back_sees_what_unreal_corrected_and_the_change_is_rolled_back() -> None:
    """Two materials on a one-slot mesh read back as two before Unreal finished the write."""
    async with editor_session() as (session, editor):
        result = await session.set_actor_property(MESH, "OverrideMaterials", json.dumps([MATERIAL_A, MATERIAL_B]))

    assert (result["success"], result["error_type"], result["applied"]) == (False, "PropertyNotApplied", False)
    assert result["details"] == {"expected": [MATERIAL_A, MATERIAL_B], "actual": [MATERIAL_A]}
    assert (result["restored"], result["previous_value"], result["value"]) == (True, [], [])
    assert editor.objects[MESH].props["OverrideMaterials"] == []
    # Rolled back by hand and cancelled: no undo step for a call that changed nothing.
    assert editor.history == [] and editor.depth == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,value,expected,actual",
    [
        ("StaticMesh", '"/Engine/BasicShapes/NoSuchMesh"', "/Engine/BasicShapes/NoSuchMesh", ""),
        ("Mobility", '"Bogus"', "Bogus", "Static"),
    ],
)
async def test_a_value_unreal_does_not_take_is_put_back(name: str, value: str, expected: str, actual: str) -> None:
    async with editor_session() as (session, editor):
        before = copy.deepcopy(editor.objects[MESH].props)
        result = await session.set_actor_property(MESH, name, value)

    assert (result["error_type"], result["restored"]) == ("PropertyNotApplied", True)
    assert result["details"] == {"expected": expected, "actual": actual}
    assert result["previous_value"] == before[name] and result["value"] == before[name]
    assert editor.objects[MESH].props == before and editor.history == []


@pytest.mark.asyncio
async def test_a_container_remote_control_refuses_to_write_gets_its_copy_back() -> None:
    """Python could empty it, Remote Control refused the write (a protected property)."""
    async with editor_session() as (session, editor):
        result = await session.set_actor_property(MESH, "ProtectedList", '["new"]')

    assert (result["success"], result["error_type"], result["restored"]) == (False, "ValidationError", True)
    assert result["value"] == ["kept"] and "put back" in result["error"]
    assert editor.objects[MESH].props["ProtectedList"] == ["kept"]
    assert editor.history == [] and editor.depth == 0 and editor.open_transactions() == {}


@pytest.mark.asyncio
async def test_a_refusal_whose_message_breaks_the_batch_json_is_put_back() -> None:
    """A protected property's refusal spans two lines, printed raw into the batch answer.

    The answer was invalid JSON: the call raised JSONDecodeError and kept the
    emptied container as an undo step ([1111, 2778] became []).
    """
    async with editor_session() as (session, editor):
        result = await session.set_actor_property(ACTOR, "PrintedRefusal", '["x"]')

    assert (result["success"], result["error_type"], result["restored"]) == (False, "ValidationError", True)
    assert "not writeable because it is protected" in result["error"]
    assert editor.objects[ACTOR].props["PrintedRefusal"] == ["a", "b"]
    assert editor.history == [] and editor.depth == 0 and editor.open_transactions() == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,name,value",
    [
        (LIGHT, "Intensity", "999"),
        (MESH, "Mobility", '"Movable"'),
        (MESH, "RelativeLocation", '{"X": 333, "Y": 0, "Z": 100}'),
        (MESH, "StaticMesh", json.dumps(CUBE)),
        (BILLBOARD, "Elements", '[{"BaseSizeX": 5}, {"BaseSizeX": 6}]'),
    ],
)
async def test_a_write_through_a_setter_is_one_undo_step_and_marks_the_level_changed(
    path: str, name: str, value: str
) -> None:
    """Remote Control's setter path calls no Modify: the write left no undo step and a clean level.

    The next undo then reverted the edit before it, and a save of the changed
    packages skipped the write.
    """
    async with editor_session() as (session, editor):
        # An earlier, unrelated undo step, and a saved level.
        editor.begin()
        editor.objects[ACTOR].modify()
        editor.objects[ACTOR].props["Tags"] = ["earlier"]
        editor.end()
        editor.dirty = False
        before = copy.deepcopy(editor.objects[path].props[name])

        result = await session.set_actor_property(path, name, value)

        assert result["applied"] is True, result
        assert editor.dirty is True
        assert len(editor.history) == 2 and editor.depth == 0
        editor.undo()
        assert editor.objects[path].props[name] == before
        assert editor.objects[ACTOR].props["Tags"] == ["earlier"]  # the earlier edit stays


@pytest.mark.asyncio
async def test_a_setter_records_the_components_attached_below_its_target() -> None:
    """SetMobility(Movable) moves the components attached below too; undo puts them all back."""
    async with editor_session() as (session, editor):
        child = editor.add(ActorComponent(editor, MESH + ".Child", "StaticMeshComponent", {"Mobility": "Static"}, {},
                                          owner=editor.objects[ACTOR]))
        mesh = editor.objects[MESH]
        mesh.get_children_components = lambda all_descendants: [child]  # type: ignore[attr-defined]

        def spread(obj: Obj, name: str) -> None:
            if obj.props[name] == "Movable":
                child.props["Mobility"] = "Movable"

        mesh.meta["Mobility"]["post_edit"] = spread
        result = await session.set_actor_property(MESH, "Mobility", '"Movable"')

        assert result["applied"] is True and child.props["Mobility"] == "Movable"
        editor.undo()
        assert (mesh.props["Mobility"], child.props["Mobility"]) == ("Static", "Static")


@pytest.mark.asyncio
async def test_a_setter_write_put_back_leaves_no_undo_step() -> None:
    async with editor_session() as (session, editor):
        result = await session.set_actor_property(MESH, "StaticMesh", '"/Engine/BasicShapes/NoSuchMesh"')

        assert (result["error_type"], result["restored"]) == ("PropertyNotApplied", True)
        assert editor.objects[MESH].props["StaticMesh"] == SPHERE and editor.history == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,value,code",
    [
        ("Tags", "5", "value_not_a_list"),
        ("Tags", '{"a": 1}', "value_not_a_list"),
        ("Tags", '"prop"', "value_not_a_list"),
        ("Settings", '["k"]', "value_not_an_object"),
    ],
)
async def test_a_value_of_another_shape_is_refused_before_the_container_is_emptied(
    name: str, value: str, code: str
) -> None:
    """5 into Tags held ['None'] and {"a": 1} held [], both reported as success."""
    async with editor_session() as (session, editor):
        before = copy.deepcopy(editor.objects[ACTOR].props[name])
        result = await session.set_actor_property(ACTOR, name, value)

    assert (result["success"], result["error_type"], result["details"]) == (False, "ValidationError", {"code": code})
    assert editor.objects[ACTOR].props[name] == before
    assert editor.writes == [] and editor.history == [] and editor.open_transactions() == {}


@pytest.mark.asyncio
async def test_a_map_is_replaced_whole() -> None:
    async with editor_session() as (session, editor):
        result = await session.set_actor_property(ACTOR, "Settings", '{"j": 2}')

    assert result["applied"] is True and editor.objects[ACTOR].props["Settings"] == {"j": 2}


@pytest.mark.asyncio
async def test_a_struct_written_in_part_keeps_its_other_fields() -> None:
    async with editor_session() as (session, editor):
        result = await session.set_actor_property(MESH, "RelativeLocation", '{"Z": 250}')

    assert result["applied"] is True and result["value"] == {"X": 0, "Y": 0, "Z": 250}
    assert len(editor.history) == 1


@pytest.mark.asyncio
async def test_a_lost_batch_answer_leaves_nothing_open_and_reports_what_unreal_holds() -> None:
    """The batch ran whole in the editor, its transaction closed inside it; only the answer was lost."""
    async with editor_session() as (session, editor):
        editor.drop_batch_answer = True
        result = await session.set_actor_property(ACTOR, "Tags", '["x"]')

        assert (result["success"], result["error_type"]) == (False, "ConnectionError")
        assert "whether the request ran is unknown" in result["error"]
        assert result["value"] == ["x"] and "restored" not in result
        assert editor.depth == 0 and editor.open_transactions() == {} and len(editor.history) == 1


@pytest.mark.asyncio
async def test_two_writes_at_once_are_two_undo_steps_and_one_undo_reverts_one() -> None:
    """A transaction used to span the write's requests, so a concurrent call's write landed in it.

    The second call's rollback was recorded inside the first call's
    transaction: one undo reverted Intensity and also put back the materials
    the rejected call had reported restored.
    """
    async with editor_session() as (session, editor):
        rejected, kept = await asyncio.gather(
            session.set_actor_property(MESH, "OverrideMaterials", json.dumps([MATERIAL_A, MATERIAL_B])),
            session.set_actor_property(LIGHT, "Intensity", "100"),
        )

        assert (rejected["error_type"], rejected["restored"]) == ("PropertyNotApplied", True)
        assert kept["applied"] is True
        assert editor.objects[MESH].props["OverrideMaterials"] == []
        assert len(editor.history) == 1 and editor.depth == 0 and editor.open_transactions() == {}
        editor.undo()
        assert editor.objects[LIGHT].props["Intensity"] == 5000.0
        assert editor.objects[MESH].props["OverrideMaterials"] == []
        assert editor.history == []


@pytest.mark.asyncio
async def test_many_writes_at_once_each_leave_their_own_undo_step() -> None:
    async with editor_session() as (session, editor):
        results = await asyncio.gather(
            *(session.set_actor_property(ACTOR, "Tags", json.dumps([f"t{n}"])) for n in range(4)),
            session.set_actor_property(LIGHT, "Intensity", "7"),
            session.set_actor_property(MESH, "StaticMesh", '"/Engine/BasicShapes/NoSuchMesh"'),
        )

        assert [r.get("applied") for r in results] == [True] * 5 + [False]
        assert len(editor.history) == 5 and editor.depth == 0 and editor.open_transactions() == {}
        for _ in range(5):
            editor.undo()
        assert editor.objects[ACTOR].props["Tags"] == ["seed"]
        assert editor.objects[LIGHT].props["Intensity"] == 5000.0
        assert editor.objects[MESH].props["StaticMesh"] == SPHERE


# -- the target: the level's actors and components, as actor_at resolves them ----------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "setup,path,error_type,code",
    [
        (
            lambda e: e.add(Obj(e, CUBE, "StaticMesh", {"LightMapResolution": 64}, {})),
            CUBE,
            "ValidationError",
            "not_an_actor_or_component",
        ),
        (
            lambda e: e.add(
                Actor(e, "/Script/Engine.Default__StaticMeshActor", "StaticMeshActor", {"bHidden": False}, {})
            ),
            "/Script/Engine.Default__StaticMeshActor",
            "ValidationError",
            "class_default_object",
        ),
        (
            lambda e: e.add(
                ActorComponent(e, "/Game/BP.BP_C:Light_GEN_VARIABLE", "PointLightComponent", {"Intensity": 1}, {})
            ),
            "/Game/BP.BP_C:Light_GEN_VARIABLE",
            "ValidationError",
            "component_template",
        ),
        (
            lambda e: e.add(
                ActorComponent(
                    e,
                    "/Script/Engine.Default__PointLight:LightComponent0",
                    "PointLightComponent",
                    {"Intensity": 1},
                    {},
                    owner=Actor(e, "/Script/Engine.Default__PointLight", "PointLight", {}, {}),
                )
            ),
            "/Script/Engine.Default__PointLight:LightComponent0",
            "ValidationError",
            "class_default_object",
        ),
        (lambda e: None, LEVEL + "Gone_1", "NotFoundError", None),
        (lambda e: None, "/Game/Maps/NotOpen.NotOpen:PersistentLevel.Lamp_1", "NotFoundError", None),
        (
            lambda e: e.add(
                Actor(
                    e,
                    "/Game/Maps/UEDPIE_0_Lab.Lab:PersistentLevel.Props_1",
                    "StaticMeshActor",
                    {"bHidden": False},
                    {},
                    e.pie_world,
                )
            ),
            "/Game/Maps/UEDPIE_0_Lab.Lab:PersistentLevel.Props_1",
            "NotFoundError",
            None,
        ),
    ],
)
async def test_anything_but_a_live_level_object_is_refused_before_a_write(
    setup: Callable[[Editor], Any], path: str, error_type: str, code: Optional[str]
) -> None:
    async with editor_session() as (session, editor):
        setup(editor)
        result = await session.set_actor_property(path, "bHidden", "true")

    assert (result["success"], result["error_type"]) == (False, error_type), result
    assert (result.get("details") or {}).get("code") == code
    assert editor.writes == [] and editor.history == []


@pytest.mark.asyncio
async def test_a_deleted_actor_and_its_component_are_not_found() -> None:
    async with editor_session() as (session, editor):
        editor.objects[ACTOR].valid = False
        editor.objects[MESH].valid = False
        actor = await session.set_actor_property(ACTOR, "bHidden", "true")
        component = await session.set_actor_property(MESH, "Mobility", '"Movable"')

    for result in (actor, component):
        assert result["error_type"] == "NotFoundError", result
        assert result["error"].startswith(f"Actor not found: {ACTOR}")
    assert editor.writes == []


@pytest.mark.asyncio
async def test_a_container_python_cannot_replace_is_refused_before_a_write() -> None:
    async with editor_session() as (session, editor):
        editor.objects[ACTOR].meta["Tags"] = {"python_readonly": True}
        result = await session.set_actor_property(ACTOR, "Tags", '["x"]')

    assert (result["error_type"], result["details"]) == ("ValidationError", {"code": "container_not_replaceable"})
    assert editor.objects[ACTOR].props["Tags"] == ["seed"] and editor.writes == []


@pytest.mark.asyncio
async def test_nothing_is_left_in_the_shared_console_namespace() -> None:
    async with editor_session() as (session, editor):
        await session.set_actor_property(BILLBOARD, "Elements", '[{"BaseSizeX": 1}]')
        await session.set_actor_property(MESH, "StaticMesh", '"/Engine/Nope"')

        # The batched parts run in dicts of their own; no object is kept anywhere.
        kept = [k for k, v in editor.console.items() if isinstance(v, Obj)]
        assert kept == []
        assert editor.open_transactions() == {}


def test_the_batched_parts_name_nothing_in_the_shared_dict() -> None:
    body = unreal_runtime._python_call("x = 1\n", {"a": 1})
    console: Dict[str, Any] = {}

    exec(body["parameters"]["PythonCommand"], console)  # noqa: S102 - runs as the editor would

    assert sorted(k for k in console if k != "__builtins__") == []
