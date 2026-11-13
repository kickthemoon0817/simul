"""The Unreal scene reads at scale: listing, search, the scene graph and components.

A production-readiness audit on UE 5.7.4 found that ``query_unreal_scene_graph``
searched the Asset Registry (``/remote/search/assets``) instead of the level: it
returned no actors in levels of 138 and 5,138, ignored ``include_components``
and ``max_depth``, and always reported a depth of 1. ``list_unreal_actors`` and
``get_unreal_actor_by_semantic_label`` had no paging: past about 130 entries the
40 KB result budget cut the list while ``count`` still reported every entry
returned by the editor, and the truncation hint named Isaac parameters.

The fake editor below answers the editor scripts the way UE 5.7.4's Python API
does (verified live): actors with labels, tags, classes and attachment, a class
hierarchy that ``MathLibrary.class_is_child_of`` walks, Blueprint classes that
resolve only by path or through an actor of that class, and components with
mobility and bounds. Remote Control is served over HTTP, so an MCP client
receives exactly what an agent would.
"""

from __future__ import annotations

import io
import json
import sys
import traceback
from contextlib import asynccontextmanager, redirect_stdout
from types import SimpleNamespace
from typing import Any, AsyncIterator, Dict, List, Optional

import pytest
from aiohttp import web
from fastmcp import Client

from simul.adapters.unreal_runtime import PAGE_BUDGET_BYTES, UnrealRuntimeSession
from simul.config import Settings
from simul.mcp.result_budget import DEFAULT_RESULT_BUDGET_BYTES
from simul.mcp.schemas.unreal import (
    UnrealGetActorBySemanticLabelResponse,
    UnrealGetActorInfoResponse,
    UnrealListActorComponentsResponse,
    UnrealListActorsResponse,
    UnrealQuerySceneGraphResponse,
)
from simul.mcp.server import SimulMCPServer

PYTHON_LIBRARY = "/Script/PythonScriptPlugin.Default__PythonScriptLibrary"
WORLD = "/Game/Maps/Lab.Lab"
LEVEL = WORLD + ":PersistentLevel."


# ---------------------------------------------------------------------------
# The fake ``unreal`` module
# ---------------------------------------------------------------------------


class Class:
    """``unreal.Class``: a UClass with its parent."""

    def __init__(self, path: str, parent: Optional["Class"] = None) -> None:
        self.path, self.parent = path, parent

    def get_name(self) -> str:
        return self.path.rsplit(".", 1)[-1]

    def get_path_name(self) -> str:
        return self.path


def class_is_child_of(cls: Class, parent: Class) -> bool:
    node: Optional[Class] = cls
    while node is not None:
        if node is parent:
            return True
        node = node.parent
    return False


OBJECT = Class("/Script/CoreUObject.Object")
ACTOR = Class("/Script/Engine.Actor", OBJECT)
STATIC_MESH_ACTOR = Class("/Script/Engine.StaticMeshActor", ACTOR)
LIGHT = Class("/Script/Engine.Light", ACTOR)
POINT_LIGHT = Class("/Script/Engine.PointLight", LIGHT)
CAMERA_ACTOR = Class("/Script/Engine.CameraActor", ACTOR)
SKY_LIGHT = Class("/Script/Engine.SkyLight", ACTOR)
# A Blueprint class: Unreal's Python module has no attribute for it.
BP_LAMP = Class("/Game/Blueprints/BP_Lamp.BP_Lamp_C", POINT_LIGHT)
SCENE_COMPONENT_CLASS = Class("/Script/Engine.SceneComponent", OBJECT)
PRIMITIVE_COMPONENT_CLASS = Class("/Script/Engine.PrimitiveComponent", SCENE_COMPONENT_CLASS)
STATIC_MESH_COMPONENT_CLASS = Class("/Script/Engine.StaticMeshComponent", PRIMITIVE_COMPONENT_CLASS)
BILLBOARD_CLASS = Class("/Script/Engine.BillboardComponent", PRIMITIVE_COMPONENT_CLASS)
LIGHT_COMPONENT_CLASS = Class("/Script/Engine.PointLightComponent", SCENE_COMPONENT_CLASS)
MOVEMENT_CLASS = Class("/Script/Engine.RotatingMovementComponent", OBJECT)
CLASSES = {
    cls.path: cls
    for cls in (
        OBJECT,
        ACTOR,
        STATIC_MESH_ACTOR,
        LIGHT,
        POINT_LIGHT,
        CAMERA_ACTOR,
        BP_LAMP,
        SCENE_COMPONENT_CLASS,
        PRIMITIVE_COMPONENT_CLASS,
        STATIC_MESH_COMPONENT_CLASS,
        BILLBOARD_CLASS,
        LIGHT_COMPONENT_CLASS,
        MOVEMENT_CLASS,
    )
}


class Vector:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.x, self.y, self.z = x, y, z


class ActorComponent:
    """``unreal.ActorComponent``: no transform."""

    static_class = staticmethod(lambda: OBJECT)

    def __init__(self, owner: "Actor", name: str, cls: Class, editor_only: bool = False) -> None:
        self.owner, self.name, self.cls, self.editor_only = owner, name, cls, editor_only

    def get_name(self) -> str:
        return self.name

    def get_path_name(self) -> str:
        return self.owner.path + "." + self.name

    def get_class(self) -> Class:
        return self.cls

    def get_owner(self) -> "Actor":
        return self.owner

    def get_editor_property(self, name: str) -> Any:
        return {"is_editor_only": self.editor_only}[name]


class SceneComponent(ActorComponent):
    static_class = staticmethod(lambda: SCENE_COMPONENT_CLASS)
    mobility = SimpleNamespace(name="MOVABLE")

    def __init__(self, *args: Any, attach: Optional["SceneComponent"] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.attach = attach

    def get_attach_parent(self) -> Optional["SceneComponent"]:
        return self.attach

    def get_world_location(self) -> Vector:
        return self.owner.location

    def is_visible(self) -> bool:
        return not self.editor_only


class PrimitiveComponent(SceneComponent):
    static_class = staticmethod(lambda: PRIMITIVE_COMPONENT_CLASS)


class Actor:
    static_class = staticmethod(lambda: ACTOR)

    def __init__(self, label: str, cls: Class, tags: tuple = (), x: float = 0.0) -> None:
        self.label, self.cls, self.tags = label, cls, list(tags)
        self.path = LEVEL + label
        self.location = Vector(x, 0.0, 0.0)
        self.parent: Optional[Actor] = None
        self.attached: List[Actor] = []
        self.root_component = PrimitiveComponent(self, "Root", STATIC_MESH_COMPONENT_CLASS)
        self.components: List[ActorComponent] = [self.root_component]
        # The world the actor belongs to: the editor world once added to the level.
        self.world: Any = None

    def get_world(self) -> Any:
        return self.world

    def attach_to(self, parent: "Actor") -> "Actor":
        self.parent = parent
        parent.attached.append(self)
        self.root_component.attach = parent.root_component
        return self

    def get_actor_label(self) -> str:
        return self.label

    def get_path_name(self) -> str:
        return self.path

    def get_class(self) -> Class:
        return self.cls

    def get_actor_location(self) -> Vector:
        return self.location

    def get_actor_rotation(self) -> SimpleNamespace:
        return SimpleNamespace(pitch=0.0, yaw=90.0, roll=0.0)

    def get_actor_scale3d(self) -> Vector:
        return Vector(1.0, 1.0, 1.0)

    def get_editor_property(self, name: str) -> Any:
        return {"tags": list(self.tags), "hidden": False}[name]

    def actor_has_tag(self, tag: str) -> bool:
        return tag.lower() in (t.lower() for t in self.tags)

    def get_components_by_class(self, cls: type) -> List[ActorComponent]:
        return [c for c in self.components if isinstance(c, cls)]

    def get_attach_parent_actor(self) -> Optional["Actor"]:
        return self.parent

    def get_attached_actors(self) -> List["Actor"]:
        return list(self.attached)

    def is_editable(self) -> bool:
        return True


def native(cls: Class) -> type:
    """The Python type Unreal's module exposes for a native class."""
    return type(cls.get_name(), (Actor,), {"static_class": staticmethod(lambda: cls)})


class Editor:
    """The level and the editor subsystems the scene reads use.

    ``partition`` makes the level a World Partition map: every actor gets a
    descriptor, and the actors not loaded leave the level listing, as in a
    saved World Partition map opened in UE 5.7.4. WorldPartitionBlueprintLibrary
    answers as UE 5.7.4's does: no descriptors (None) outside such a map or
    while PIE runs, load_actors loads at once, and unload_actors releases only
    the actors load_actors loaded.
    """

    def __init__(self) -> None:
        self.actors: List[Actor] = []
        self.world = SimpleNamespace(
            get_name=lambda: "Lab",
            get_path_name=lambda: WORLD,
            get_outermost=lambda: SimpleNamespace(get_name=lambda: "/Game/Maps/Lab"),
        )
        self.pie = False
        self.descriptors: Optional[List[SimpleNamespace]] = None
        self.stored: Dict[str, Actor] = {}
        # The timeout each script was run with (None: unreal.timeout).
        self.timeouts: List[Optional[float]] = []
        self.library_loaded: set = set()

    def add(self, *actors: Actor) -> List[Actor]:
        for actor in actors:
            actor.world = self.world
        self.actors.extend(actors)
        return list(actors)

    def partition(self, loaded: int) -> None:
        """Make the level a World Partition map with only its first ``loaded`` actors loaded."""
        self.descriptors = [
            SimpleNamespace(guid=actor.label, actor_path=actor.path, is_spatially_loaded=True, location=actor.location)
            for actor in self.actors
        ]
        self.stored = {actor.label: actor for actor in self.actors}
        self.actors = self.actors[:loaded]

    def load_object(self, outer: Any, path: str) -> Any:
        return next((actor for actor in self.actors if actor.path == path), None)

    def load_class(self, outer: Any, path: str) -> Optional[Class]:
        return CLASSES.get(path)

    def world_partition(self) -> SimpleNamespace:
        editor = self

        def descriptors() -> Optional[List[SimpleNamespace]]:
            return None if editor.descriptors is None or editor.pie else list(editor.descriptors)

        def intersecting(box: SimpleNamespace) -> Optional[List[SimpleNamespace]]:
            found = descriptors()
            if found is None:
                return None
            return [
                d
                for d in found
                if all(
                    getattr(box.min, axis) <= getattr(d.location, axis) <= getattr(box.max, axis)
                    for axis in "xyz"
                )
            ]

        def load_actors(guids: List[str]) -> None:
            for guid in guids:
                actor = editor.stored[guid]
                if actor not in editor.actors:
                    editor.actors.append(actor)
                    editor.library_loaded.add(guid)

        def unload_actors(guids: List[str]) -> None:
            for guid in guids:
                if guid in editor.library_loaded:
                    editor.actors.remove(editor.stored[guid])
                    editor.library_loaded.discard(guid)

        return SimpleNamespace(
            get_actor_descs=descriptors,
            get_intersecting_actor_descs=intersecting,
            load_actors=load_actors,
            unload_actors=unload_actors,
        )

    def module(self) -> SimpleNamespace:
        actor_subsystem = SimpleNamespace(get_all_level_actors=lambda: [] if self.pie else list(self.actors))
        level_editor = SimpleNamespace(is_in_play_in_editor=lambda: self.pie)
        editor = SimpleNamespace(
            get_editor_world=lambda: None if self.pie else self.world,
            get_game_world=lambda: SimpleNamespace() if self.pie else None,
        )
        return SimpleNamespace(
            Actor=Actor,
            ActorComponent=ActorComponent,
            SceneComponent=SceneComponent,
            PrimitiveComponent=PrimitiveComponent,
            Class=Class,
            WorldSettings=type("WorldSettings", (Actor,), {}),
            StaticMeshActor=native(STATIC_MESH_ACTOR),
            Light=native(LIGHT),
            PointLight=native(POINT_LIGHT),
            CameraActor=native(CAMERA_ACTOR),
            SkyLight=native(SKY_LIGHT),
            EditorActorSubsystem="actors",
            LevelEditorSubsystem="level",
            UnrealEditorSubsystem="editor",
            LayersSubsystem="layers",
            get_editor_subsystem=lambda name: {
                "actors": actor_subsystem,
                "level": level_editor,
                "editor": editor,
                # GWorld: the editor world wherever Remote Control runs a script.
                "layers": SimpleNamespace(get_world=lambda: self.world),
            }[name],
            load_object=self.load_object,
            # actor_at finds actors among loaded objects (find_object), never loading one.
            find_object=self.load_object,
            load_class=self.load_class,
            Box=lambda min, max: SimpleNamespace(min=min, max=max),
            Vector=Vector,
            GameplayStatics=SimpleNamespace(get_all_actors_of_class=lambda world, cls: list(self.actors)),
            WorldPartitionBlueprintLibrary=self.world_partition(),
            MathLibrary=SimpleNamespace(class_is_child_of=class_is_child_of),
            SystemLibrary=SimpleNamespace(
                is_valid=lambda obj: obj is not None,
                get_component_bounds=lambda component: (
                    component.get_world_location(),
                    Vector(50.0, 50.0, 50.0),
                    86.6,
                ),
            ),
        )

    def run_python(self, code: str) -> Dict[str, Any]:
        """``ExecutePythonCommandEx`` in ExecuteFile mode."""
        output = io.StringIO()
        try:
            with redirect_stdout(output):
                exec(code, {})
        except Exception:
            return {"ReturnValue": False, "CommandResult": traceback.format_exc(), "LogOutput": []}
        return {
            "ReturnValue": True,
            "CommandResult": "None",
            "LogOutput": [{"Type": "Info", "Output": line + "\n"} for line in output.getvalue().splitlines()],
        }


@pytest.fixture
def editor(monkeypatch: pytest.MonkeyPatch) -> Editor:
    level = Editor()
    monkeypatch.setitem(sys.modules, "unreal", level.module())
    return level


@pytest.fixture
def session(editor: Editor, monkeypatch: pytest.MonkeyPatch) -> UnrealRuntimeSession:
    """A session whose Python runs in the fake editor; any other Remote Control call fails."""
    client = UnrealRuntimeSession(Settings())

    async def execute(code: str, mode: str = "ExecuteFile", timeout: Optional[float] = None) -> Dict[str, Any]:
        editor.timeouts.append(timeout)
        return editor.run_python(code)

    async def no_http(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise AssertionError(f"unexpected Remote Control request {args}")

    monkeypatch.setattr(client, "_execute_python", execute)
    monkeypatch.setattr(client, "_http_put", no_http)
    return client


@asynccontextmanager
async def remote_control(editor: Editor) -> AsyncIterator[Settings]:
    """Serve Remote Control for ``editor``: Python runs, anything else is UE 5.7's 400."""

    async def handler(request: web.Request) -> web.Response:
        body = await request.json()
        if request.path == "/remote/object/call" and body.get("objectPath") == PYTHON_LIBRARY:
            return web.json_response(editor.run_python(body["parameters"]["PythonCommand"]))
        return web.json_response({"errorMessage": "not served by this editor"}, status=400)

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    base = Settings()
    unreal = base.unreal.model_copy(
        update={"host": "127.0.0.1", "port": site._server.sockets[0].getsockname()[1], "tool_surface": "full"}
    )
    try:
        yield base.model_copy(update={"unreal": unreal})
    finally:
        await runner.cleanup()


async def call_tool(settings: Settings, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Call one tool through a real MCP client; return the JSON text an agent receives."""
    server = SimulMCPServer(settings, backends={"unreal"})
    async with Client(server.mcp) as client:
        result = await client.call_tool(name, arguments, raise_on_error=False)
    text = result.content[-1].text
    payload = json.loads(text)
    payload["_chars"] = len(text)
    return payload


def crowd(editor: Editor, count: int) -> List[Actor]:
    """``count`` crates and lamps like the audit's level: tagged, every tenth crate parenting the next."""
    actors: List[Actor] = []
    parent: Optional[Actor] = None
    for i in range(count):
        if i % 50 == 0:
            actor = Actor(f"Lamp_{i:05d}", POINT_LIGHT, ("light", f"zone_{i // 1000}"), x=i * 300.0)
        else:
            actor = Actor(f"Crate_{i:05d}", STATIC_MESH_ACTOR, ("prop", f"crate_{i % 10}"), x=i * 300.0)
            if i % 10 == 1:
                parent = actor
            elif parent is not None and i % 10 == 2:
                actor.attach_to(parent)
        actors.append(actor)
    editor.add(*actors)
    return actors


# ---------------------------------------------------------------------------
# list_unreal_actors
# ---------------------------------------------------------------------------


def test_the_page_budget_leaves_room_inside_the_result_budget() -> None:
    assert PAGE_BUDGET_BYTES + 5_000 <= DEFAULT_RESULT_BUDGET_BYTES


@pytest.mark.asyncio
async def test_a_large_level_lists_in_pages_whose_counts_are_exact(editor: Editor) -> None:
    """At 5,000 actors every page fits the result budget untrimmed, count is
    what came back, and the pages together hold every actor once."""
    actors = crowd(editor, 5000)
    seen: List[str] = []
    offset: Optional[int] = 0
    pages = 0
    async with remote_control(editor) as settings:
        while offset is not None:
            page = await call_tool(settings, "list_unreal_actors", {"max_results": 5000, "offset": offset})
            assert page["_chars"] <= DEFAULT_RESULT_BUDGET_BYTES, page["_chars"]
            assert "truncation" not in page and "oversized_bytes" not in page
            assert page["count"] == len(page["actors"]) > 50
            assert page["total"] == 5000 and page["offset"] == offset
            UnrealListActorsResponse(**{k: v for k, v in page.items() if k != "_chars"})
            seen.extend(entry["path"] for entry in page["actors"])
            if page["next_offset"] is not None:
                assert page["truncated"] is True
                assert page["next_offset"] == offset + page["count"]
                assert f"offset={page['next_offset']}" in page["hint"]
                assert "response budget" in page["hint"]
                assert "root_path" not in page["hint"] and "max_items" not in page["hint"]
            else:
                assert page["truncated"] is False and page["hint"] is None
            offset = page["next_offset"]
            pages += 1
    assert seen == [actor.path for actor in actors]
    assert pages > 1


@pytest.mark.asyncio
async def test_max_results_ends_a_page_and_says_so(session: UnrealRuntimeSession, editor: Editor) -> None:
    crowd(editor, 30)
    page = await session.list_actors(max_results=10, offset=5)
    assert [entry["name"] for entry in page["actors"]][:2] == ["Crate_00005", "Crate_00006"]
    assert (page["count"], page["total"], page["next_offset"], page["truncated"]) == (10, 30, 15, True)
    assert "reached max_results" in page["hint"] and "offset=15" in page["hint"]
    last = await session.list_actors(max_results=10, offset=25)
    assert (last["count"], last["next_offset"], last["truncated"]) == (5, None, False)
    assert "hint" not in last
    past = await session.list_actors(offset=99)
    assert (past["count"], past["total"], past["next_offset"]) == (0, 30, None)


@pytest.mark.asyncio
async def test_entries_carry_the_attachment_hierarchy(session: UnrealRuntimeSession, editor: Editor) -> None:
    table, cup, plate = editor.add(
        Actor("Table", STATIC_MESH_ACTOR), Actor("Cup", STATIC_MESH_ACTOR), Actor("Plate", STATIC_MESH_ACTOR)
    )
    cup.attach_to(table)
    plate.attach_to(table)
    listing = await session.list_actors()
    by_name = {entry["name"]: entry for entry in listing["actors"]}
    assert by_name["Table"]["parent"] is None
    assert by_name["Table"]["children"] == [cup.path, plate.path]
    assert by_name["Table"]["child_count"] == 2
    assert by_name["Cup"]["parent"] == table.path and by_name["Cup"]["child_count"] == 0
    assert "components" not in by_name["Table"]
    info = await session.get_actor_info(cup.path)
    assert (info["parent"], info["children"], info["child_count"]) == (table.path, [], 0)
    UnrealGetActorInfoResponse(success=True, **info)
    info = await session.get_actor_info(table.path)
    assert info["children"] == [cup.path, plate.path]


@pytest.mark.asyncio
async def test_a_listing_entry_names_the_first_children_and_counts_them_all(
    session: UnrealRuntimeSession, editor: Editor
) -> None:
    (rack,) = editor.add(Actor("Rack", STATIC_MESH_ACTOR))
    editor.add(*(Actor(f"Box_{i:03d}", STATIC_MESH_ACTOR).attach_to(rack) for i in range(250)))
    entry = (await session.list_actors(label_filter="rack"))["actors"][0]
    assert (len(entry["children"]), entry["child_count"]) == (20, 250)
    info = await session.get_actor_info(rack.path)
    assert (len(info["children"]), info["child_count"]) == (200, 250)


@pytest.mark.asyncio
async def test_label_filter_matches_part_of_the_label_ignoring_case(
    session: UnrealRuntimeSession, editor: Editor
) -> None:
    editor.add(Actor("Kitchen_Table", STATIC_MESH_ACTOR), Actor("Desk", STATIC_MESH_ACTOR))
    listing = await session.list_actors(label_filter="TABLE")
    assert [entry["name"] for entry in listing["actors"]] == ["Kitchen_Table"]
    assert listing["total"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("class_filter", "include_subclasses", "expected"),
    [
        ("Light", False, []),
        ("PointLight", False, ["Lamp"]),
        ("/Script/Engine.PointLight", False, ["Lamp"]),
        ("Light", True, ["Lamp", "BlueprintLamp"]),
        ("/Script/Engine.Light", True, ["Lamp", "BlueprintLamp"]),
        ("PointLight", True, ["Lamp", "BlueprintLamp"]),
        ("BP_Lamp_C", False, ["BlueprintLamp"]),
        ("BP_Lamp_C", True, ["BlueprintLamp"]),
        ("/Game/Blueprints/BP_Lamp", True, ["BlueprintLamp"]),
        ("Actor", True, ["Lamp", "BlueprintLamp", "Crate", "Camera"]),
    ],
)
async def test_class_filter_matches_by_name_or_path_and_optionally_subclasses(
    session: UnrealRuntimeSession,
    editor: Editor,
    class_filter: str,
    include_subclasses: bool,
    expected: List[str],
) -> None:
    editor.add(
        Actor("Lamp", POINT_LIGHT),
        Actor("BlueprintLamp", BP_LAMP),
        Actor("Crate", STATIC_MESH_ACTOR),
        Actor("Camera", CAMERA_ACTOR),
    )
    listing = await session.list_actors(class_filter=class_filter, include_subclasses=include_subclasses)
    assert [entry["name"] for entry in listing["actors"]] == expected
    assert listing["warnings"] == []


@pytest.mark.asyncio
async def test_a_subclass_filter_that_names_no_class_warns(session: UnrealRuntimeSession, editor: Editor) -> None:
    editor.add(Actor("Lamp", POINT_LIGHT))
    listing = await session.list_actors(class_filter="Lamp_Type", include_subclasses=True)
    assert listing["actors"] == []
    assert "No class named 'Lamp_Type'" in listing["warnings"][0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_results": 0}, "max_results must be between 1 and 5000"),
        ({"max_results": 5001}, "max_results must be between 1 and 5000"),
        ({"offset": -1}, "offset must be 0 or more"),
    ],
)
async def test_list_refuses_a_page_out_of_range(
    session: UnrealRuntimeSession, kwargs: Dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        await session.list_actors(**kwargs)


# ---------------------------------------------------------------------------
# get_unreal_actor_by_semantic_label
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_semantic_search_pages_every_match_at_scale(editor: Editor) -> None:
    actors = crowd(editor, 5000)
    props = [actor.path for actor in actors if "prop" in actor.tags]
    seen: List[str] = []
    offset: Optional[int] = 0
    async with remote_control(editor) as settings:
        while offset is not None:
            page = await call_tool(
                settings,
                "get_unreal_actor_by_semantic_label",
                {"label": "PROP", "max_results": 5000, "offset": offset},
            )
            assert page["_chars"] <= DEFAULT_RESULT_BUDGET_BYTES
            assert "truncation" not in page
            assert page["count"] == len(page["actors"])
            assert page["total_matches"] == len(props)
            UnrealGetActorBySemanticLabelResponse(**{k: v for k, v in page.items() if k != "_chars"})
            seen.extend(entry["path"] for entry in page["actors"])
            offset = page["next_offset"]
    assert seen == props


# ---------------------------------------------------------------------------
# query_unreal_scene_graph
# ---------------------------------------------------------------------------


def family(editor: Editor) -> Dict[str, Actor]:
    """Floor, a table with a cup holding a spoon, a lamp, and a camera on the lamp."""
    floor, table, cup, spoon, lamp, camera = editor.add(
        Actor("Floor", STATIC_MESH_ACTOR),
        Actor("Table", STATIC_MESH_ACTOR),
        Actor("Cup", STATIC_MESH_ACTOR),
        Actor("Spoon", STATIC_MESH_ACTOR),
        Actor("Lamp", POINT_LIGHT),
        Actor("Camera", CAMERA_ACTOR),
    )
    cup.attach_to(table)
    spoon.attach_to(cup)
    camera.attach_to(lamp)
    return {actor.label: actor for actor in (floor, table, cup, spoon, lamp, camera)}


def names(node: Dict[str, Any]) -> Any:
    """A tree as nested (name, children) pairs."""
    return [(child["name"], names(child)) for child in node["children"]]


@pytest.mark.asyncio
async def test_the_scene_graph_is_the_levels_attachment_hierarchy(editor: Editor) -> None:
    family(editor)
    async with remote_control(editor) as settings:
        graph = await call_tool(settings, "query_unreal_scene_graph", {})
    assert graph["success"] is True, graph
    assert graph["root"]["path"] == WORLD and graph["root"]["class_name"] == "World"
    assert names(graph["root"]) == [
        ("Floor", []),
        ("Table", [("Cup", [("Spoon", [])])]),
        ("Lamp", [("Camera", [])]),
    ]
    assert (graph["total_actors"], graph["total_depth"], graph["returned_actors"]) == (6, 3, 6)
    assert (graph["count"], graph["total"], graph["next_offset"], graph["truncated"]) == (3, 3, None, False)
    UnrealQuerySceneGraphResponse(**{k: v for k, v in graph.items() if k != "_chars"})


@pytest.mark.asyncio
async def test_max_depth_cuts_the_tree_and_marks_where(session: UnrealRuntimeSession, editor: Editor) -> None:
    family(editor)
    graph = await session.query_scene_graph(max_depth=2)
    table = graph["root"]["children"][1]
    cup = table["children"][0]
    assert (cup["name"], cup["children"], cup["child_count"], cup["children_truncated"]) == ("Cup", [], 1, True)
    assert "children_truncated" not in table
    assert (graph["total_depth"], graph["returned_actors"]) == (3, 5)
    flat = await session.query_scene_graph(max_depth=1)
    assert [child["children"] for child in flat["root"]["children"]] == [[], [], []]
    assert [child.get("children_truncated", False) for child in flat["root"]["children"]] == [False, True, True]


@pytest.mark.asyncio
async def test_root_path_is_that_actors_subtree(session: UnrealRuntimeSession, editor: Editor) -> None:
    actors = family(editor)
    graph = await session.query_scene_graph(root_path=actors["Table"].path)
    assert graph["root"]["path"] == actors["Table"].path
    assert graph["root"]["child_count"] == 1
    assert names(graph["root"]) == [("Cup", [("Spoon", [])])]
    assert (graph["total_actors"], graph["total_depth"]) == (2, 2)


@pytest.mark.asyncio
async def test_a_missing_root_is_an_error(session: UnrealRuntimeSession, editor: Editor) -> None:
    family(editor)
    graph = await session.query_scene_graph(root_path=LEVEL + "Nothing")
    assert graph["success"] is False and "Actor not found" in graph["error"]


@pytest.mark.asyncio
async def test_include_components_lists_each_actors_components(
    session: UnrealRuntimeSession, editor: Editor
) -> None:
    actors = family(editor)
    lamp = actors["Lamp"]
    lamp.components.append(SceneComponent(lamp, "LightComponent0", LIGHT_COMPONENT_CLASS, attach=lamp.root_component))
    lamp.components.append(ActorComponent(lamp, "Spin", MOVEMENT_CLASS))
    graph = await session.query_scene_graph(include_components=True)
    node = graph["root"]["children"][2]
    assert node["components"] == [
        {"name": "Root", "class_name": "StaticMeshComponent", "is_root": True, "attach_parent": None},
        {"name": "LightComponent0", "class_name": "PointLightComponent", "is_root": False, "attach_parent": "Root"},
        {"name": "Spin", "class_name": "RotatingMovementComponent", "is_root": False, "attach_parent": None},
    ]
    # The camera's root is attached to the lamp's: reported as the actor's parent, not here.
    assert node["children"][0]["components"][0]["attach_parent"] is None
    assert "components" not in (await session.query_scene_graph())["root"]["children"][0]


@pytest.mark.asyncio
async def test_a_class_filter_keeps_matches_and_the_path_to_them(
    session: UnrealRuntimeSession, editor: Editor
) -> None:
    family(editor)
    graph = await session.query_scene_graph(class_filter="CameraActor")
    assert names(graph["root"]) == [("Lamp", [("Camera", [])])]
    lamp = graph["root"]["children"][0]
    assert (lamp["matches_filter"], lamp["children"][0]["matches_filter"]) == (False, True)
    assert (graph["total_actors"], graph["total"]) == (2, 1)
    lights = await session.query_scene_graph(class_filter="Light", include_subclasses=True)
    assert names(lights["root"]) == [("Lamp", [])]
    assert lights["root"]["children"][0]["child_count"] == 0


@pytest.mark.asyncio
async def test_the_scene_graph_pages_the_roots_children(session: UnrealRuntimeSession, editor: Editor) -> None:
    crowd(editor, 5000)
    seen: List[str] = []
    offset: Optional[int] = 0
    while offset is not None:
        graph = await session.query_scene_graph(offset=offset, max_results=5000)
        assert len(json.dumps(graph)) <= DEFAULT_RESULT_BUDGET_BYTES
        assert graph["count"] == len(graph["root"]["children"])
        assert graph["total_actors"] == 5000 and graph["total_depth"] == 2
        seen.extend(child["path"] for child in graph["root"]["children"])
        if graph["next_offset"] is not None:
            assert f"offset={graph['next_offset']}" in graph["hint"]
        offset = graph["next_offset"]
    attached = sum(1 for actor in editor.actors if actor.parent is not None)
    assert len(seen) == len(set(seen)) == 5000 - attached


@pytest.mark.asyncio
async def test_a_subtree_too_large_for_a_page_is_drawn_shallower(
    session: UnrealRuntimeSession, editor: Editor
) -> None:
    (hub,) = editor.add(Actor("Hub", STATIC_MESH_ACTOR))
    for i in range(150):
        (spoke,) = editor.add(Actor(f"Spoke_{i:03d}", STATIC_MESH_ACTOR).attach_to(hub))
        editor.add(Actor(f"Leaf_{i:03d}", STATIC_MESH_ACTOR).attach_to(spoke))
    graph = await session.query_scene_graph(max_depth=5)
    assert len(json.dumps(graph)) <= DEFAULT_RESULT_BUDGET_BYTES
    assert graph["depth_limited"] == [{"path": hub.path, "depth": 2}]
    assert graph["truncated"] is True
    assert any("root_path" in warning for warning in graph["warnings"])
    spoke = graph["root"]["children"][0]["children"][0]
    assert spoke["children"] == [] and spoke["children_truncated"] is True
    assert graph["total_depth"] == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_depth": 0}, "max_depth must be between 1 and 64"),
        ({"max_depth": 65}, "max_depth must be between 1 and 64"),
        ({"offset": -2}, "offset must be 0 or more"),
        ({"max_results": 0}, "max_results must be between 1 and 5000"),
    ],
)
async def test_the_scene_graph_refuses_bad_paging(
    session: UnrealRuntimeSession, kwargs: Dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        await session.query_scene_graph(**kwargs)


# ---------------------------------------------------------------------------
# list_unreal_actor_components
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_components_list_with_mobility_and_bounds(editor: Editor) -> None:
    table, cup = editor.add(Actor("Table", STATIC_MESH_ACTOR, x=100.0), Actor("Cup", STATIC_MESH_ACTOR))
    cup.attach_to(table)
    table.components.append(PrimitiveComponent(table, "Sprite", BILLBOARD_CLASS, editor_only=True,
                                               attach=table.root_component))
    table.components.append(ActorComponent(table, "Spin", MOVEMENT_CLASS))
    async with remote_control(editor) as settings:
        listing = await call_tool(settings, "list_unreal_actor_components", {"actor_path": table.path})
        primitives = await call_tool(
            settings,
            "list_unreal_actor_components",
            {"actor_path": table.path, "class_filter": "PrimitiveComponent", "max_results": 1},
        )
        cups = await call_tool(settings, "list_unreal_actor_components", {"actor_path": cup.path})
    UnrealListActorComponentsResponse(**{k: v for k, v in listing.items() if k != "_chars"})
    root, sprite, spin = listing["components"]
    assert root == {
        "name": "Root",
        "path": table.path + ".Root",
        "class_name": "StaticMeshComponent",
        "is_root": True,
        "is_scene_component": True,
        "editor_only": False,
        "attach_parent": None,
        "location": [100.0, 0.0, 0.0],
        "mobility": "Movable",
        "visible": True,
        "bounds": {"origin": [100.0, 0.0, 0.0], "extent": [50.0, 50.0, 50.0], "sphere_radius": 86.6},
    }
    assert (sprite["attach_parent"], sprite["editor_only"], sprite["visible"]) == ("Root", True, False)
    assert (spin["is_scene_component"], spin["mobility"], spin["bounds"]) == (False, None, None)
    assert spin["location"] is None
    assert (listing["count"], listing["total"], listing["truncated"]) == (3, 3, False)
    assert [entry["name"] for entry in primitives["components"]] == ["Root"]
    assert (primitives["total"], primitives["next_offset"]) == (2, 1)
    # A root attached to another actor's component names it by path.
    assert cups["components"][0]["attach_parent"] == table.path + ".Root"


@pytest.mark.asyncio
async def test_components_of_a_missing_actor_is_an_error(session: UnrealRuntimeSession) -> None:
    result = await session.list_actor_components(LEVEL + "Nothing")
    assert result["success"] is False and "Actor not found" in result["error"]


# ---------------------------------------------------------------------------
# World Partition: unloaded actors are reported, and can be loaded
# ---------------------------------------------------------------------------


def world_partition_level(editor: Editor, count: int = 100, loaded: int = 2) -> List[Actor]:
    """A saved World Partition map opened normally: ``count`` tagged crates, ``loaded`` of them loaded.

    The audit's map: 100 crates 200 m apart, of which a map opened in UE 5.7.4
    loads none but its two always-loaded helper actors.
    """
    crates = [
        Actor(f"WPCrate_{i:03d}", STATIC_MESH_ACTOR, ("wp_prop",), x=(i % 10) * 20000.0) for i in range(count)
    ]
    for i, crate in enumerate(crates):
        crate.location = Vector((i % 10) * 20000.0, (i // 10) * 20000.0, 100.0)
    editor.add(*crates)
    editor.partition(loaded)
    return crates


@pytest.mark.asyncio
async def test_reads_of_a_world_partition_map_report_and_warn_of_unloaded_actors(editor: Editor) -> None:
    """A saved World Partition map with 102 descriptors read as "Total actors:
    2" and found 0 of its 100 tagged actors, saying nothing of the rest."""
    world_partition_level(editor)
    async with remote_control(editor) as settings:
        summary = await call_tool(settings, "summarize_unreal_scene", {})
        listing = await call_tool(settings, "list_unreal_actors", {})
        search = await call_tool(settings, "get_unreal_actor_by_semantic_label", {"label": "wp_prop"})
        graph = await call_tool(settings, "query_unreal_scene_graph", {})
    for reply in (summary, listing, search, graph):
        assert (reply["world_partition"], reply["unloaded_actors"]) == (True, 98), reply
    assert "98 of this map's 100 saved actors are not loaded" in summary["summary_text"]
    assert "load_unreal_world_partition_actors" in summary["summary_text"]
    for reply in (listing, search, graph):
        assert any("load_unreal_world_partition_actors" in warning for warning in reply["warnings"]), reply
    assert (listing["total"], search["total_matches"], graph["total_actors"]) == (2, 2, 2)


@pytest.mark.asyncio
async def test_a_level_that_is_not_world_partition_reports_no_unloaded_actors(
    session: UnrealRuntimeSession, editor: Editor
) -> None:
    family(editor)
    listing = await session.list_actors()
    assert (listing["world_partition"], listing["unloaded_actors"], listing["warnings"]) == (False, None, [])
    summary = await session.summarize_scene()
    assert (summary["world_partition"], summary["unloaded_actors"]) == (False, None)
    assert "World Partition" not in summary["summary_text"]


@pytest.mark.asyncio
async def test_during_pie_world_partition_is_unknown_not_reported_as_loaded(
    session: UnrealRuntimeSession, editor: Editor
) -> None:
    world_partition_level(editor)
    editor.pie = True
    listing = await session.list_actors()
    assert (listing["pie_active"], listing["world_partition"], listing["unloaded_actors"]) == (True, None, None)


@pytest.mark.asyncio
async def test_loading_a_region_loads_its_actors_and_unloading_releases_them(editor: Editor) -> None:
    crates = world_partition_level(editor)
    async with remote_control(editor) as settings:
        loaded = await call_tool(
            settings,
            "load_unreal_world_partition_actors",
            {"bounds_min": "-100,-100,-1000", "bounds_max": "30000, 30000, 1000"},
        )
        after = await call_tool(settings, "get_unreal_actor_by_semantic_label", {"label": "wp_prop"})
        released = await call_tool(
            settings,
            "load_unreal_world_partition_actors",
            {"bounds_min": "-100,-100,-1000", "bounds_max": "30000,30000,1000", "action": "unload"},
        )
    assert loaded["success"] is True, loaded
    # Crates 0, 1, 10 and 11 stand in the box; 0 and 1 were loaded already.
    assert (loaded["matched"], loaded["requested"], loaded["changed"], loaded["not_changed"]) == (4, 2, 2, 0)
    assert loaded["actors"] == [crates[10].path, crates[11].path]
    assert (loaded["loaded_actors"], loaded["unloaded_actors"]) == (4, 96)
    assert (after["total_matches"], after["unloaded_actors"]) == (4, 96)
    # Unload releases only what the tool loaded: crates 0 and 1 stay.
    assert (released["requested"], released["changed"], released["not_changed"]) == (4, 2, 2)
    assert (released["loaded_actors"], released["unloaded_actors"]) == (2, 98)


@pytest.mark.asyncio
async def test_loading_more_than_max_actors_is_refused_and_changes_nothing(
    session: UnrealRuntimeSession, editor: Editor
) -> None:
    world_partition_level(editor)
    refused = await session.load_world_partition_actors(max_actors=50)
    assert refused["success"] is False and refused["error_type"] == "ValidationError"
    assert "98 actors would load, more than max_actors (50)" in refused["error"]
    assert refused["details"] == {"code": "TooManyActors", "would_change": 98, "max_actors": 50}
    assert len(editor.actors) == 2
    # Loading takes about 2 ms an actor: the call waits the import and export timeout.
    assert editor.timeouts[-1] == float(session.settings.unreal.import_export_timeout)
    everything = await session.load_world_partition_actors(max_actors=98)
    assert (everything["changed"], everything["unloaded_actors"], len(editor.actors)) == (98, 0, 100)


@pytest.mark.asyncio
async def test_loading_needs_a_world_partition_map_outside_pie(session: UnrealRuntimeSession, editor: Editor) -> None:
    family(editor)
    not_partitioned = await session.load_world_partition_actors()
    assert (not_partitioned["success"], not_partitioned["error_type"]) == (False, "InvalidState")
    world_partition_level(editor)
    editor.pie = True
    during_pie = await session.load_world_partition_actors()
    assert (during_pie["success"], during_pie["error_type"]) == (False, "PIEActive")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"action": "pin"}, "action must be load or unload"),
        ({"max_actors": 0}, "max_actors must be between 1 and 10000"),
        ({"max_actors": 10001}, "max_actors must be between 1 and 10000"),
        ({"bounds_min": [0, 0, 0]}, "Give bounds_min and bounds_max together"),
        ({"bounds_min": [0, 0, 0], "bounds_max": [1, 1]}, "bounds_max must be three finite numbers"),
        ({"bounds_min": [5, 0, 0], "bounds_max": [1, 1, 1]}, "bounds_min must not exceed bounds_max"),
    ],
)
async def test_loading_refuses_bad_input_before_any_request(
    session: UnrealRuntimeSession, kwargs: Dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        await session.load_world_partition_actors(**kwargs)


# ---------------------------------------------------------------------------
# simul unreal list-actors / scene-graph
# ---------------------------------------------------------------------------


class _RecordingSession:
    """Stands in for the CLI's session: records the call, answers one page."""

    def __init__(self) -> None:
        self.calls: List[Any] = []

    async def list_actors(self, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(("list_actors", kwargs))
        return {
            "actors": [{"name": "Lamp", "path": LEVEL + "Lamp", "class_name": "PointLight"}],
            "count": 1,
            "total": 9,
            "next_offset": 4,
            "hint": "Call again with offset=4 for the next page.",
        }

    async def query_scene_graph(self, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(("query_scene_graph", kwargs))
        return {"root": {"name": "Lab", "children": []}, "total_actors": 0}

    async def load_world_partition_actors(self, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(("load_world_partition_actors", kwargs))
        return {"action": kwargs["action"], "changed": 4, "requested": 4, "unloaded_actors": 96,
                "actor_descriptors": 102}

    async def capture_camera(self, *args: Any) -> Dict[str, Any]:
        self.calls.append(("capture_camera", args))
        return {"path": "/shots/SimulCamera_Cam_1.png", "camera": {"horizontal_fov": 90.0},
                "intrinsics": {"fx": 640.0}}

    async def close(self) -> None:
        pass


def test_the_cli_pages_and_filters_the_listing(monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner

    from simul.cli import unreal_cli
    from simul.cli.main import app

    stub = _RecordingSession()
    monkeypatch.setattr(unreal_cli, "_session", lambda *args, **kwargs: stub)
    runner = CliRunner()
    listed = runner.invoke(
        app,
        ["unreal", "list-actors", "--class", "Light", "--subclasses", "--label", "lamp", "--offset", "3", "-n", "1"],
    )
    assert listed.exit_code == 0, listed.stdout
    assert "PointLight" in listed.stdout and "Lamp" in listed.stdout and "offset=4" in listed.stdout
    graph = runner.invoke(
        app, ["--json", "unreal", "scene-graph", "--components", "--class", "Light", "--offset", "2", "-n", "5"]
    )
    assert graph.exit_code == 0, graph.stdout
    assert stub.calls == [
        (
            "list_actors",
            {
                "class_filter": "Light",
                "tag_filter": None,
                "max_results": 1,
                "offset": 3,
                "label_filter": "lamp",
                "include_subclasses": True,
            },
        ),
        (
            "query_scene_graph",
            {
                "root_path": None,
                "max_depth": 3,
                "include_components": True,
                "class_filter": "Light",
                "include_subclasses": False,
                "offset": 2,
                "max_results": 5,
            },
        ),
    ]


def test_the_cli_loads_world_partition_actors_and_captures_cameras(monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner

    from simul.cli import unreal_cli
    from simul.cli.main import app

    stub = _RecordingSession()
    monkeypatch.setattr(unreal_cli, "_session", lambda *args, **kwargs: stub)
    runner = CliRunner()
    loaded = runner.invoke(
        app, ["--json", "unreal", "wp-load", "--min", "0,0,0", "--max", "100,100,50", "--max-actors", "9"]
    )
    assert loaded.exit_code == 0, loaded.stdout
    assert json.loads(loaded.stdout)["unloaded_actors"] == 96
    released = runner.invoke(app, ["--json", "unreal", "wp-load", "--unload"])
    assert released.exit_code == 0, released.stdout
    captured = runner.invoke(
        app, ["--json", "unreal", "capture-camera", LEVEL + "Cam", "-W", "320", "--height", "240"]
    )
    assert captured.exit_code == 0, captured.stdout
    assert json.loads(captured.stdout)["intrinsics"] == {"fx": 640.0}
    assert stub.calls == [
        (
            "load_world_partition_actors",
            {"bounds_min": [0.0, 0.0, 0.0], "bounds_max": [100.0, 100.0, 50.0], "max_actors": 9, "action": "load"},
        ),
        ("load_world_partition_actors", {"bounds_min": None, "bounds_max": None, "max_actors": 500,
                                         "action": "unload"}),
        ("capture_camera", (LEVEL + "Cam", 320, 240, "png")),
    ]
