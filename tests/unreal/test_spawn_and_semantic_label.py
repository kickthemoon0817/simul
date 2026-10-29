"""spawn_unreal_actor and get_unreal_actor_by_semantic_label against a UE-shaped editor.

Both failed live on UE 5.7.4. spawn_unreal_actor sent every path to
EditorActorSubsystem.SpawnActorFromClass, which answers HTTP 200 with an empty
ReturnValue for anything that is not an Actor class, so an asset path such as
/Engine/BasicShapes/Cube.Cube came back as a successful spawn with an empty
actor_path and nothing in the level. get_unreal_actor_by_semantic_label called
GetAllLevelActors on /Script/UnrealEd.Default__EditorLevelLibrary, an object
UE 5.7 does not have, so every lookup was an HTTP 400.

The double below answers Remote Control the way UE 5.7 does: an object it does
not have is a 400, SpawnActorFromClass returns "" for a path that is not an
Actor class, and a script sent to PythonScriptLibrary runs against a fake
``unreal`` module with the EditorActorSubsystem calls the real editor exposes.
The tools are called through a real MCP client, so each assertion is on the
payload an agent receives.
"""

from __future__ import annotations

import ast
import io
import json
import sys
import traceback
from contextlib import asynccontextmanager, contextmanager, redirect_stdout
from types import SimpleNamespace
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from fastmcp import Client

from simul.adapters.unreal_runtime import UnrealRuntimeSession
from simul.config import Settings
from simul.mcp.server import SimulMCPServer

PYTHON_LIBRARY = "/Script/PythonScriptPlugin.Default__PythonScriptLibrary"
ACTOR_SUBSYSTEM = "/Script/UnrealEd.Default__EditorActorSubsystem"
# Objects a UE 5.7 editor answers Remote Control calls on; any other is a 400.
EDITOR_OBJECTS = {
    PYTHON_LIBRARY,
    ACTOR_SUBSYSTEM,
    "/Script/UnrealEd.Default__UnrealEditorSubsystem",
    "/Script/Engine.Default__KismetSystemLibrary",
}


# ---------------------------------------------------------------------------
# The fake ``unreal`` module
# ---------------------------------------------------------------------------


class Vector:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.x, self.y, self.z = x, y, z


class Rotator:
    """Positional order is UE's (roll, pitch, yaw); scripts must pass keywords."""

    def __init__(self, roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0) -> None:
        self.roll, self.pitch, self.yaw = roll, pitch, yaw


class Class:
    """``unreal.Class``: what a /Script/ or Blueprint ``..._C`` path loads."""

    def __init__(self, path: str, parent: Optional["Class"] = None) -> None:
        self.path, self.parent = path, parent

    def get_name(self) -> str:
        return self.path.rsplit(".", 1)[-1]

    def get_path_name(self) -> str:
        return self.path

    def get_class(self) -> "Class":
        return UCLASS


def class_is_child_of(cls: Class, parent: Class) -> bool:
    """``unreal.MathLibrary.class_is_child_of``."""
    node: Optional[Class] = cls
    while node is not None:
        if node is parent:
            return True
        node = node.parent
    return False


UOBJECT = Class("/Script/CoreUObject.Object")
UCLASS = Class("/Script/CoreUObject.Class", UOBJECT)
ACTOR = Class("/Script/Engine.Actor", UOBJECT)
STATIC_MESH_ACTOR = Class("/Script/Engine.StaticMeshActor", ACTOR)
POINT_LIGHT = Class("/Script/Engine.PointLight", ACTOR)
STATIC_MESH = Class("/Script/Engine.StaticMesh", UOBJECT)
MATERIAL = Class("/Script/Engine.Material", UOBJECT)


class Asset:
    def __init__(self, path: str, cls: Class) -> None:
        self.path, self.cls = path, cls

    def get_path_name(self) -> str:
        return self.path

    def get_class(self) -> Class:
        return self.cls


CUBE = Asset("/Engine/BasicShapes/Cube.Cube", STATIC_MESH)
SHAPE_MATERIAL = Asset(
    "/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial", MATERIAL
)


class Component:
    mobility = SimpleNamespace(name="MOVABLE")

    def get_name(self) -> str:
        return "Root"

    def get_class(self) -> Class:
        return Class("/Script/Engine.SceneComponent")


class Actor:
    def __init__(
        self,
        path: str,
        cls: Class,
        location: Vector,
        rotation: Rotator,
        mesh: Optional[Asset] = None,
        tags: tuple = (),
    ) -> None:
        self.path, self.cls, self.mesh = path, cls, mesh
        self.location, self.rotation = location, rotation
        self.label = path.rsplit(".", 1)[-1]
        self.tags = list(tags)
        self.root_component = Component()

    @staticmethod
    def static_class() -> Class:
        return ACTOR

    def get_path_name(self) -> str:
        return self.path

    def get_class(self) -> Class:
        return self.cls

    def get_actor_label(self) -> str:
        return self.label

    def set_actor_label(self, label: str) -> None:
        self.label = label

    def get_actor_location(self) -> Vector:
        return self.location

    def get_actor_rotation(self) -> Rotator:
        return self.rotation

    def get_actor_scale3d(self) -> Vector:
        return Vector(1.0, 1.0, 1.0)

    def get_editor_property(self, name: str) -> Any:
        return {"tags": list(self.tags), "hidden": False}[name]

    def get_components_by_class(self, cls: Any) -> List[Component]:
        return [self.root_component]


class Editor:
    """The level, the loadable objects and the EditorActorSubsystem."""

    def __init__(self) -> None:
        self.objects: Dict[str, Any] = {
            obj.get_path_name(): obj
            for obj in (
                ACTOR,
                STATIC_MESH_ACTOR,
                POINT_LIGHT,
                STATIC_MESH,
                CUBE,
                SHAPE_MATERIAL,
            )
        }
        self.actors: List[Actor] = []
        self.spawn_calls: List[tuple] = []
        self.transactions: List[str] = []

    def load_object(self, outer: Any, path: str) -> Any:
        found = self.objects.get(path)
        return found or next((a for a in self.actors if a.path == path), None)

    def place(
        self, cls: Class, location: Vector, rotation: Rotator, **kwargs: Any
    ) -> Actor:
        actor = Actor(
            f"/Game/Map.Map:PersistentLevel.{cls.get_name()}_{len(self.actors)}",
            cls,
            location,
            rotation,
            **kwargs,
        )
        self.actors.append(actor)
        return actor

    def spawn_actor_from_class(
        self, actor_class: Class, location: Vector, rotation: Rotator
    ) -> Optional[Actor]:
        self.spawn_calls.append(("class", actor_class.get_path_name()))
        if not class_is_child_of(actor_class, ACTOR):
            return None
        return self.place(actor_class, location, rotation)

    def spawn_actor_from_object(
        self, source: Any, location: Vector, rotation: Rotator
    ) -> Optional[Actor]:
        """The editor's actor factories: an Actor class spawns, a StaticMesh
        becomes a StaticMeshActor showing it, and nothing else is placeable."""
        self.spawn_calls.append(("object", source.get_path_name()))
        if isinstance(source, Class):
            return (
                self.place(source, location, rotation)
                if class_is_child_of(source, ACTOR)
                else None
            )
        if source.get_class() is STATIC_MESH:
            return self.place(STATIC_MESH_ACTOR, location, rotation, mesh=source)
        return None

    def get_all_level_actors(self) -> List[Actor]:
        return list(self.actors)

    @contextmanager
    def transaction(self, description: str) -> Iterator[None]:
        self.transactions.append(description)
        yield

    def module(self) -> SimpleNamespace:
        return SimpleNamespace(
            Actor=Actor,
            ActorComponent=Component,
            Class=Class,
            Vector=Vector,
            Rotator=Rotator,
            EditorActorSubsystem="EditorActorSubsystem",
            get_editor_subsystem=lambda which: (
                self if which == "EditorActorSubsystem" else None
            ),
            load_object=self.load_object,
            MathLibrary=SimpleNamespace(class_is_child_of=class_is_child_of),
            ScopedEditorTransaction=self.transaction,
        )

    def run_python(self, code: str) -> Dict[str, Any]:
        """``ExecutePythonCommandEx`` in ExecuteFile mode."""
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
            "CommandResult": "None",
            "LogOutput": [
                {"Type": "Info", "Output": line + "\n"}
                for line in output.getvalue().splitlines()
            ],
        }


@pytest.fixture
def editor(monkeypatch: pytest.MonkeyPatch) -> Editor:
    level = Editor()
    monkeypatch.setitem(sys.modules, "unreal", level.module())
    return level


@asynccontextmanager
async def remote_control(editor: Editor) -> AsyncIterator[SimpleNamespace]:
    """Serve Remote Control for ``editor``; yields the settings and every request body."""
    requests: List[Dict[str, Any]] = []

    async def handler(request: web.Request) -> web.Response:
        body = await request.json()
        requests.append(body)
        target = body.get("objectPath")
        if request.path != "/remote/object/call" or target not in EDITOR_OBJECTS:
            return web.json_response(
                {"errorMessage": f"Object: {target} does not exist."}, status=400
            )
        parameters = body.get("parameters", {})
        if body.get("functionName") == "ExecutePythonCommandEx":
            return web.json_response(editor.run_python(parameters["PythonCommand"]))
        if body.get("functionName") == "SpawnActorFromClass":
            source = editor.load_object(None, parameters["ActorClass"])
            actor = (
                editor.spawn_actor_from_class(source, Vector(), Rotator())
                if isinstance(source, Class)
                else None
            )
            return web.json_response(
                {"ReturnValue": actor.get_path_name() if actor else ""}
            )
        return web.json_response(
            {"errorMessage": "Function is unavailable remotely"}, status=400
        )

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    base = Settings()
    unreal = base.unreal.model_copy(
        update={
            "host": "127.0.0.1",
            "port": site._server.sockets[0].getsockname()[1],
            "tool_surface": "full",
        }
    )
    try:
        yield SimpleNamespace(
            settings=base.model_copy(update={"unreal": unreal}), requests=requests
        )
    finally:
        await runner.cleanup()


async def call_tool(
    settings: Settings, name: str, arguments: Dict[str, Any]
) -> Dict[str, Any]:
    """Call one tool through a real MCP client; return the payload an agent sees."""
    server = SimulMCPServer(settings, backends={"unreal"})
    async with Client(server.mcp) as client:
        result = await client.call_tool(name, arguments, raise_on_error=False)
    return json.loads(result.content[0].text)


def objects_called(requests: List[Dict[str, Any]]) -> set:
    return {body.get("objectPath") for body in requests}


# ---------------------------------------------------------------------------
# spawn_unreal_actor
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_spawn_from_an_asset_path_places_a_static_mesh_actor_showing_it(
    editor: Editor,
) -> None:
    async with remote_control(editor) as rc:
        payload = await call_tool(
            rc.settings,
            "spawn_unreal_actor",
            {
                "asset_path": "/Engine/BasicShapes/Cube.Cube",
                "location_x": 100.0,
                "location_y": 50.0,
                "location_z": 25.0,
                "rotation_yaw": 30.0,
                "label": "Crate",
            },
        )

    (actor,) = editor.actors
    assert payload["success"] is True, payload
    assert payload["actor_path"] == actor.path
    assert payload["actor_class"] == "StaticMeshActor"
    assert payload["label"] == "Crate"
    assert payload["location"] == [100.0, 50.0, 25.0]
    assert payload["rotation"] == [0.0, 30.0, 0.0]
    assert actor.mesh is CUBE
    assert editor.spawn_calls == [("object", "/Engine/BasicShapes/Cube.Cube")]
    assert editor.transactions == ["simul: spawn actor"]
    assert objects_called(rc.requests) == {PYTHON_LIBRARY}


@pytest.mark.asyncio
async def test_spawn_from_a_class_path_spawns_that_class(editor: Editor) -> None:
    async with remote_control(editor) as rc:
        payload = await call_tool(
            rc.settings,
            "spawn_unreal_actor",
            {"asset_path": "/Script/Engine.PointLight"},
        )

    (actor,) = editor.actors
    assert payload["success"] is True, payload
    assert payload["actor_path"] == actor.path
    assert payload["actor_class"] == "PointLight"
    assert payload["label"] == actor.label
    assert editor.spawn_calls == [("class", "/Script/Engine.PointLight")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("asset_path", "error_type", "message"),
    [
        (
            "/Game/Missing/Nope.Nope",
            "NotFoundError",
            "No class or asset at '/Game/Missing/Nope.Nope'",
        ),
        (
            "/Script/Engine.StaticMesh",
            "SpawnError",
            "is a class but not an Actor class",
        ),
        (
            SHAPE_MATERIAL.path,
            "SpawnError",
            "Unreal placed no actor for " + SHAPE_MATERIAL.path,
        ),
    ],
)
async def test_a_spawn_that_places_nothing_is_an_error(
    editor: Editor, asset_path: str, error_type: str, message: str
) -> None:
    async with remote_control(editor) as rc:
        payload = await call_tool(
            rc.settings, "spawn_unreal_actor", {"asset_path": asset_path}
        )

    assert payload["success"] is False, payload
    assert payload["error_type"] == error_type
    assert message in payload["error"]
    assert not payload.get("actor_path")
    assert editor.actors == []


@pytest.mark.asyncio
async def test_spawning_from_an_actor_in_the_level_is_an_error(editor: Editor) -> None:
    """An actor's path is neither a class nor an asset; the editor would place
    a fresh default actor of its class, not a copy of it."""
    placed = editor.place(STATIC_MESH_ACTOR, Vector(), Rotator())
    async with remote_control(editor) as rc:
        payload = await call_tool(
            rc.settings, "spawn_unreal_actor", {"asset_path": placed.path}
        )

    assert payload["success"] is False, payload
    assert payload["error_type"] == "SpawnError"
    assert "is an actor in a level" in payload["error"]
    assert editor.actors == [placed]
    assert editor.spawn_calls == []


@pytest.mark.asyncio
async def test_a_spawn_reply_without_an_actor_path_is_an_error() -> None:
    session = UnrealRuntimeSession(Settings())
    session._execute_python = AsyncMock(
        return_value={
            "ReturnValue": True,
            "LogOutput": [
                {
                    "Type": "Info",
                    "Output": json.dumps({"actor_path": "", "actor_class": "Cube"}),
                }
            ],
        }
    )

    result = await session.spawn_actor("/Engine/BasicShapes/Cube.Cube")

    assert result["success"] is False
    assert "/Engine/BasicShapes/Cube.Cube" in result["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [{"location": (0.0, 0.0)}, {"rotation": (0.0, float("nan"), 0.0)}],
)
async def test_spawn_refuses_a_transform_that_is_not_three_finite_numbers(
    kwargs: Dict[str, Any],
) -> None:
    session = UnrealRuntimeSession(Settings())
    session._execute_python = AsyncMock()

    with pytest.raises(ValueError, match="three finite numbers"):
        await session.spawn_actor("/Script/Engine.PointLight", **kwargs)
    session._execute_python.assert_not_called()


# ---------------------------------------------------------------------------
# get_unreal_actor_by_semantic_label
# ---------------------------------------------------------------------------


def tag_level(editor: Editor) -> Dict[str, Actor]:
    actors = {
        "chair": editor.place(
            STATIC_MESH_ACTOR, Vector(1.0, 2.0, 3.0), Rotator(yaw=90.0), tags=("Chair",)
        ),
        "leg": editor.place(
            STATIC_MESH_ACTOR, Vector(), Rotator(), tags=("chair_leg", "wood")
        ),
        "table": editor.place(STATIC_MESH_ACTOR, Vector(), Rotator(), tags=("table",)),
        "light": editor.place(POINT_LIGHT, Vector(), Rotator()),
    }
    actors["chair"].label = "Chair_01"
    return actors


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("label", "match_mode", "expected"),
    [
        ("chair", "exact", ["chair"]),
        ("CHAIR", "contains", ["chair", "leg"]),
        ("^chair_", "regex", ["leg"]),
        ("^(table|wood)$", "regex", ["leg", "table"]),
    ],
)
async def test_semantic_label_lookup_matches_actor_tags(
    editor: Editor, label: str, match_mode: str, expected: List[str]
) -> None:
    """Tag matching ignores case, as Unreal's FName comparison does."""
    actors = tag_level(editor)
    async with remote_control(editor) as rc:
        payload = await call_tool(
            rc.settings,
            "get_unreal_actor_by_semantic_label",
            {"label": label, "match_mode": match_mode},
        )

    assert payload["success"] is True, payload
    assert [entry["path"] for entry in payload["actors"]] == [
        actors[key].path for key in expected
    ]
    assert payload["total_matches"] == len(expected)
    assert payload["truncated"] is False
    assert payload["label_searched"] == label
    # One editor script; never a Remote Control object UE 5.7 does not have.
    assert objects_called(rc.requests) == {PYTHON_LIBRARY}


@pytest.mark.asyncio
async def test_semantic_label_lookup_returns_list_actor_entries(editor: Editor) -> None:
    actors = tag_level(editor)
    async with remote_control(editor) as rc:
        payload = await call_tool(
            rc.settings, "get_unreal_actor_by_semantic_label", {"label": "chair"}
        )

    assert payload["actors"] == [
        {
            "name": "Chair_01",
            "path": actors["chair"].path,
            "class_name": "StaticMeshActor",
            "location": [1.0, 2.0, 3.0],
            "rotation": [0.0, 90.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
            "tags": ["Chair"],
        }
    ]


@pytest.mark.asyncio
async def test_semantic_label_lookup_counts_matches_past_max_results(
    editor: Editor,
) -> None:
    actors = tag_level(editor)
    async with remote_control(editor) as rc:
        payload = await call_tool(
            rc.settings,
            "get_unreal_actor_by_semantic_label",
            {"label": "chair", "match_mode": "contains", "max_results": 1},
        )

    assert [entry["path"] for entry in payload["actors"]] == [actors["chair"].path]
    assert payload["total_matches"] == 2
    assert payload["truncated"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"match_mode": "fuzzy"}, "match_mode must be one of exact, contains, regex"),
        ({"label": "chair(", "match_mode": "regex"}, "not a valid regular expression"),
        ({"max_results": 0}, "max_results must be between 1 and 5000"),
    ],
)
async def test_semantic_label_lookup_refuses_bad_input_before_any_request(
    kwargs: Dict[str, Any], message: str
) -> None:
    session = UnrealRuntimeSession(Settings())
    session._execute_python = AsyncMock()

    with pytest.raises(ValueError, match=message):
        await session.get_actor_by_semantic_label(**{"label": "chair", **kwargs})
    session._execute_python.assert_not_called()


# ---------------------------------------------------------------------------
# Caller data reaches the editor as data
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("spawn_actor", {"asset_path": "PAYLOAD"}),
        (
            "spawn_actor",
            {"asset_path": "/Script/Engine.PointLight", "label": "PAYLOAD"},
        ),
        ("get_actor_by_semantic_label", {"label": "PAYLOAD"}),
    ],
)
async def test_caller_strings_are_embedded_as_python_literals(
    method: str, kwargs: Dict[str, Any]
) -> None:
    payload = "'; simul_injected(); #\"\\\n"
    session = UnrealRuntimeSession(Settings())
    session._execute_python = AsyncMock(
        return_value={
            "ReturnValue": True,
            "LogOutput": [{"Type": "Info", "Output": "{}"}],
        }
    )

    await getattr(session, method)(
        **{k: payload if v == "PAYLOAD" else v for k, v in kwargs.items()}
    )

    tree = ast.parse(session._execute_python.call_args.args[0])
    assert not any(
        isinstance(node, ast.Name) and node.id == "simul_injected"
        for node in ast.walk(tree)
    )
    assert any(
        isinstance(node, ast.Constant) and node.value == payload
        for node in ast.walk(tree)
    )
