"""set_unreal_actor_property replaces containers, reads back and stays in the editor world; the reads say more.

Two kinds of double: the editor scripts run against a small fake ``unreal``
module (types, ``find_object``, the editor world), and the session runs
against real HTTP servers answering as UE 5.7's Remote Control does.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import sys
from contextlib import asynccontextmanager, redirect_stdout
from types import SimpleNamespace
from typing import Any, AsyncIterator, Dict, List, Optional

import pytest
from aiohttp import web
from PIL import Image

from simul.adapters import unreal_runtime
from simul.adapters._unreal_errors import UnrealValidationError
from simul.adapters._unreal_python import unwrap
from simul.adapters._unreal_scripts import SCENE_SUMMARY
from simul.adapters.unreal_connection import discover_instances
from simul.adapters.unreal_runtime import THUMBNAIL_ASSET, UnrealRuntimeSession
from simul.config import Settings
from simul.mcp.schemas.unreal import UnrealHealthCheckResponse

LEVEL = "/Game/Maps/Lab.Lab:PersistentLevel."
CUBE = LEVEL + "Cube_1"


# -- a fake ``unreal`` for the editor scripts ------------------------------------


class Object:
    def __init__(self, path: str, world: Any = None) -> None:
        self.path, self.world, self.properties = path, world, {}

    def get_path_name(self) -> str:
        return self.path

    def get_class(self) -> Any:
        return SimpleNamespace(
            get_name=lambda: type(self).__name__,
            get_path_name=lambda: type(self).__name__,
        )

    def get_world(self) -> Any:
        return self.world

    def get_editor_property(self, name: str) -> Any:
        if name not in self.properties:
            raise Exception(f"Failed to find property '{name}'")
        return self.properties[name]

    def set_editor_property(self, name: str, value: Any) -> None:
        current = self.properties[name]
        if isinstance(current, Array) and not all(
            isinstance(item, (str, int, float)) for item in value
        ):
            raise TypeError("NativizeProperty: Cannot nativize")
        if isinstance(current, Array) and any(
            isinstance(item, str) and item.startswith("/") for item in value
        ):
            raise TypeError("NativizeProperty: Cannot nativize 'str' as 'Object'")
        self.properties[name] = Array(value) if isinstance(current, Array) else value


class Array(list):
    pass


class Set(set):
    pass


class Map(dict):
    pass


class Actor(Object):
    root_component = None

    def get_components_by_class(self, cls: Any) -> List[Any]:
        return [c for c in getattr(self, "components", []) if isinstance(c, cls)]


class StaticMeshActor(Actor):
    pass


class Light(Actor):
    pass


class PointLight(Light):
    pass


class SkyLight(Actor):
    pass


class CameraActor(Actor):
    pass


class CineCameraActor(CameraActor):
    pass


class LightmassImportanceVolume(Actor):
    pass


class CameraBlockingVolume(Actor):
    pass


class WorldSettings(Actor):
    pass


class ActorComponent(Object):
    def __init__(self, path: str, owner: Optional[Actor]) -> None:
        super().__init__(path)
        self.owner = owner

    def get_owner(self) -> Optional[Actor]:
        return self.owner


class MeshComponent(ActorComponent):
    pass


class StaticMeshComponent(MeshComponent):
    static_mesh: Any = None


class SkeletalMeshComponent(MeshComponent):
    pass


class BlueprintGeneratedClass:
    def __init__(self, path: str) -> None:
        self.path = path

    def get_path_name(self) -> str:
        return self.path


class Editor:
    """The editor world, its level and a transaction log."""

    def __init__(self) -> None:
        self.world = SimpleNamespace(
            get_path_name=lambda: "/Game/Maps/Lab.Lab",
            get_outermost=lambda: SimpleNamespace(get_name=lambda: "/Game/Maps/Lab"),
        )
        self.pie_world = SimpleNamespace(
            get_path_name=lambda: "/Game/Maps/UEDPIE_0_Lab.Lab"
        )
        self.pie = False
        self.objects: Dict[str, Any] = {}
        self.destroyed: set = set()
        self.transactions: List[str] = []

    def add(self, obj: Any) -> Any:
        self.objects[obj.get_path_name()] = obj
        return obj

    def module(self) -> SimpleNamespace:
        editor = self

        class ScopedEditorTransaction:
            def __init__(self, description: str) -> None:
                editor.transactions.append(description)

            def __enter__(self) -> None:
                return None

            def __exit__(self, *args: Any) -> None:
                return None

        subsystems = {
            "UnrealEditorSubsystem": SimpleNamespace(
                get_editor_world=lambda: None if editor.pie else editor.world,
                get_game_world=lambda: editor.pie_world if editor.pie else None,
            ),
            "LevelEditorSubsystem": SimpleNamespace(
                is_in_play_in_editor=lambda: editor.pie
            ),
            "LayersSubsystem": SimpleNamespace(get_world=lambda: editor.world),
            "EditorActorSubsystem": SimpleNamespace(
                get_all_level_actors=lambda: [
                    o for o in editor.objects.values() if isinstance(o, Actor)
                ]
            ),
        }
        return SimpleNamespace(
            find_object=lambda outer, path: editor.objects.get(path),
            Actor=Actor,
            ActorComponent=ActorComponent,
            MeshComponent=MeshComponent,
            StaticMeshComponent=StaticMeshComponent,
            SkeletalMeshComponent=SkeletalMeshComponent,
            StaticMeshActor=StaticMeshActor,
            Light=Light,
            SkyLight=SkyLight,
            CameraActor=CameraActor,
            WorldSettings=WorldSettings,
            BlueprintGeneratedClass=BlueprintGeneratedClass,
            Array=Array,
            Set=Set,
            Map=Map,
            ScopedEditorTransaction=ScopedEditorTransaction,
            SystemLibrary=SimpleNamespace(
                is_valid=lambda obj: obj.get_path_name() not in editor.destroyed
            ),
            get_editor_subsystem=subsystems.__getitem__,
            UnrealEditorSubsystem="UnrealEditorSubsystem",
            LevelEditorSubsystem="LevelEditorSubsystem",
            LayersSubsystem="LayersSubsystem",
            EditorActorSubsystem="EditorActorSubsystem",
        )

    def run(
        self, monkeypatch: pytest.MonkeyPatch, script: str, **args: Any
    ) -> Dict[str, Any]:
        """Run an editor script as Remote Control does; return what it printed."""
        monkeypatch.setitem(sys.modules, "unreal", self.module())
        output = io.StringIO()
        with redirect_stdout(output):
            exec(f"args = {args!r}\n" + script, {})
        return json.loads(output.getvalue())


@pytest.fixture
def editor() -> Editor:
    return Editor()


def _cube(editor: Editor) -> StaticMeshActor:
    cube = editor.add(StaticMeshActor(CUBE, editor.world))
    cube.properties.update(Tags=Array(["a", "b", "c"]), bHidden=False)
    return cube


# -- a fake Remote Control endpoint --------------------------------------------------


@asynccontextmanager
async def endpoint(handler: Any, **unreal: Any) -> AsyncIterator[UnrealRuntimeSession]:
    async def route(request: web.Request) -> web.StreamResponse:
        return await handler(request)

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", route)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    session = UnrealRuntimeSession(
        _settings(site._server.sockets[0].getsockname()[1], **unreal)
    )
    try:
        yield session
    finally:
        await session.close()
        await runner.cleanup()


def _settings(port: int, **unreal: Any) -> Settings:
    base = Settings()
    return base.model_copy(
        update={
            "unreal": base.unreal.model_copy(
                update={
                    "host": "127.0.0.1",
                    "port": port,
                    "timeout": 2.0,
                    "retry_base_delay": 0.001,
                    **unreal,
                }
            )
        }
    )


def test_a_property_name_is_required() -> None:
    with pytest.raises(UnrealValidationError):
        asyncio.run(UnrealRuntimeSession(Settings()).set_actor_property(CUBE, " ", "1"))


@pytest.mark.parametrize(
    "wanted,got,matches",
    [
        (["a", "b"], ["a", "b"], True),
        (["a"], ["a", "b"], False),
        ("Movable", "Movable", True),
        ("movable", "EComponentMobility::Movable", True),
        ("Bogus", "Movable", False),
        ("/Engine/BasicShapes/Sphere", "/Engine/BasicShapes/Sphere.Sphere", True),
        (None, "", True),
        ({"Z": 250}, {"X": 0, "Y": 0, "Z": 250}, True),
        ({"Z": 250}, {"X": 0, "Y": 0, "Z": 100}, False),
        (0.1, 0.10000000149011612, True),
        (True, True, True),
        (1, True, True),
        (2, "Movable", None),
        ({"Missing": 1}, {"X": 0}, None),
    ],
)
def test_values_are_compared_as_unreal_reads_them_back(
    wanted: Any, got: Any, matches: Optional[bool]
) -> None:
    assert unreal_runtime._json_value_matches(wanted, got) is matches


# -- describe: values, not nulls ---------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("values_status", [200, 400])
async def test_describe_fills_values_from_one_read(values_status: int) -> None:
    async def handler(request: web.Request) -> web.Response:
        if request.path == "/remote/object/describe":
            return web.json_response(
                {
                    "Class": "/Script/Engine.StaticMeshActor",
                    "Properties": [
                        {"Name": "Tags", "Type": "TArray"},
                        {"Name": "bHidden", "Type": "bool"},
                        {"Name": "OnClicked", "Type": "FActorOnClickedSignature"},
                    ],
                    "Functions": [{"Name": "K2_GetActorLocation"}],
                }
            )
        if values_status != 200:
            return web.Response(
                text='{ "errorMessage": "Unable to find the object." }',
                status=values_status,
            )
        return web.json_response({"Tags": ["robot"], "bHidden": False})

    async with endpoint(handler) as session:
        result = await session.describe_object(CUBE)

    if values_status == 200:
        assert result["properties"] == [
            {"name": "Tags", "type": "TArray", "value": ["robot"], "readable": True},
            {"name": "bHidden", "type": "bool", "value": False, "readable": True},
            {
                "name": "OnClicked",
                "type": "FActorOnClickedSignature",
                "value": None,
                "readable": False,
            },
        ]
        assert "warning" not in result
    else:
        assert [p["readable"] for p in result["properties"]] == [False, False, False]
        assert result["warning"].startswith(
            "Property values unavailable: Remote Control refused"
        )


# -- thumbnails: the size asked for, actors by their asset -------------------------


def _image(fmt: str, size: tuple = (256, 256)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, "red").save(out, format=fmt)
    return out.getvalue()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "width,height,expected",
    [
        (64, 64, (64, 64)),
        (128, 32, (32, 32)),
        (1024, 1024, (256, 256)),
        (256, 256, (256, 256)),
    ],
)
async def test_a_thumbnail_fits_the_size_asked_for_and_is_never_enlarged(
    width: int, height: int, expected: tuple
) -> None:
    stored = _image("JPEG")

    async def handler(request: web.Request) -> web.Response:
        return web.Response(body=stored, content_type="image/jpeg")

    async with endpoint(handler) as session:
        result = await session.get_actor_thumbnail(
            "/Engine/BasicShapes/Cube.Cube", width, height
        )

    with Image.open(io.BytesIO(base64.b64decode(result["image_base64"]))) as image:
        assert image.size == expected and image.format == "JPEG"
    assert (result["width"], result["height"]) == expected
    assert (result["stored_width"], result["stored_height"]) == (256, 256)
    if expected == (256, 256):
        assert base64.b64decode(result["image_base64"]) == stored


@pytest.mark.parametrize("width,height", [(0, 64), (64, -5), (2048, 64), (True, 64)])
def test_a_thumbnail_size_out_of_range_is_a_validation_error(
    width: Any, height: Any
) -> None:
    with pytest.raises(UnrealValidationError) as raised:
        asyncio.run(
            UnrealRuntimeSession(Settings()).get_actor_thumbnail(
                "/Engine/X.X", width, height
            )
        )
    assert raised.value.error_type == "ValidationError"


@pytest.mark.asyncio
async def test_a_class_icon_is_not_a_thumbnail() -> None:
    async def handler(request: web.Request) -> web.Response:
        return web.Response(
            text='<svg width="64" height="64"></svg>', content_type="image/svg+xml"
        )

    async with endpoint(handler) as session:
        result = await session.get_actor_thumbnail("/Game/NoThumbnail.NoThumbnail")

    assert (result["success"], result["error_type"]) == (False, "NotFoundError")
    assert "image/svg+xml" in result["error"]


@pytest.mark.asyncio
async def test_an_actor_is_shown_by_its_asset() -> None:
    asked: List[str] = []

    async def handler(request: web.Request) -> web.Response:
        body = await request.json()
        if body.get("functionName") == "ExecutePythonCommandEx":
            printed = {"asset": "/Engine/BasicShapes/Cube.Cube", "via": "mesh"}
            return web.json_response(
                {
                    "ReturnValue": True,
                    "LogOutput": [{"Type": "Info", "Output": json.dumps(printed)}],
                }
            )
        asked.append(body["objectPath"])
        return web.Response(body=_image("PNG", (256, 128)), content_type="image/png")

    async with endpoint(handler) as session:
        result = await session.get_actor_thumbnail(CUBE, 128, 128)

    assert asked == ["/Engine/BasicShapes/Cube.Cube"]
    assert (result["asset_path"], result["thumbnail_of"]) == (
        CUBE,
        "/Engine/BasicShapes/Cube.Cube",
    )
    assert (result["format"], result["width"], result["height"]) == ("png", 128, 64)


def test_thumbnail_asset_prefers_the_blueprint_then_a_mesh(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    blueprint = editor.add(StaticMeshActor(LEVEL + "Robot_1", editor.world))
    blueprint.get_class = lambda: BlueprintGeneratedClass("/Game/BP_Robot.BP_Robot_C")  # type: ignore[method-assign]
    cube = _cube(editor)
    mesh = StaticMeshComponent(CUBE + ".StaticMeshComponent0", cube)
    mesh.static_mesh = Object("/Engine/BasicShapes/Cube.Cube")
    cube.root_component = mesh
    lamp = editor.add(PointLight(LEVEL + "Lamp_1", editor.world))

    assert editor.run(monkeypatch, THUMBNAIL_ASSET, path=blueprint.path) == {
        "asset": "/Game/BP_Robot.BP_Robot",
        "via": "blueprint",
    }
    assert editor.run(monkeypatch, THUMBNAIL_ASSET, path=CUBE) == {
        "asset": "/Engine/BasicShapes/Cube.Cube",
        "via": "mesh",
    }
    refused = editor.run(monkeypatch, THUMBNAIL_ASSET, path=lamp.path)
    assert (
        refused["error_type"] == "NotFoundError"
        and "capture_unreal_viewport" in refused["error"]
    )


# -- summarize: lights and cameras by class -----------------------------------------


def test_lights_and_cameras_are_counted_by_class_not_by_name(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    for cls in (
        StaticMeshActor,
        PointLight,
        SkyLight,
        CineCameraActor,
        LightmassImportanceVolume,
        CameraBlockingVolume,
    ):
        editor.add(cls(LEVEL + cls.__name__, editor.world))

    scene = editor.run(monkeypatch, SCENE_SUMMARY)

    assert (
        scene["total_actors"],
        scene["static_meshes"],
        scene["lights"],
        scene["cameras"],
    ) == (6, 1, 2, 1)


# -- health keeps its warnings; busy is not down ------------------------------------


def test_health_check_warnings_reach_the_client() -> None:
    payload = {
        "success": True,
        "reachable": True,
        "connected": True,
        "engine_version": "",
        "project_name": "",
        "is_editor": True,
        "warnings": ["engine_version and project_name unavailable: gated"],
    }

    assert (
        UnrealHealthCheckResponse(**payload).model_dump()["warnings"]
        == payload["warnings"]
    )


@asynccontextmanager
async def silent_editor() -> AsyncIterator[int]:
    """A port that accepts connections and never answers, as an editor busy on its game thread does."""
    held: List[asyncio.StreamWriter] = []

    async def hold(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        held.append(writer)

    server = await asyncio.start_server(hold, "127.0.0.1", 0)
    try:
        yield server.sockets[0].getsockname()[1]
    finally:
        for writer in held:
            writer.close()
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_a_busy_editor_is_busy_not_down() -> None:
    async with silent_editor() as port:
        session = UnrealRuntimeSession(_settings(port, ping_timeout=0.3, timeout=0.3))
        try:
            ping = await session.ping()
            health = await session.health_check()
        finally:
            await session.close()

    for payload in (ping, health):
        assert (payload["reachable"], payload["busy"], payload["error_type"]) == (
            False,
            True,
            "InstanceBusy",
        )
        assert (
            "accepted the connection but Remote Control did not answer"
            in payload["error"]
        )
    assert health["connected"] is False


@pytest.mark.asyncio
async def test_a_down_editor_is_not_busy() -> None:
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    session = UnrealRuntimeSession(_settings(port, ping_timeout=0.3, max_retries=0))
    try:
        ping = await session.ping()
        health = await session.health_check()
    finally:
        await session.close()

    for payload in (ping, health):
        assert payload["reachable"] is False and "busy" not in payload
        assert payload["error"].startswith(
            f"Cannot reach Remote Control at 127.0.0.1:{port}"
        )


@pytest.mark.asyncio
async def test_a_scan_lists_a_busy_editor_with_the_reason(tmp_path: Any) -> None:
    async with silent_editor() as port:
        base = _settings(port, ping_timeout=0.3, timeout=0.3)
        settings = base.model_copy(
            update={
                "unreal": base.unreal.model_copy(
                    update={
                        "port": port + 1 if port < 65535 else port - 1,
                        "scan_port_start": port,
                        "scan_port_end": port + 1,
                        "attachment_path": str(tmp_path / "attachment.json"),
                    }
                )
            }
        )
        payload = await discover_instances(settings)

    busy = [item for item in payload["instances"] if item["port"] == port]
    assert len(busy) == 1 and busy[0]["reachable"] is False
    assert "busy" in busy[0]["error"]


# -- list_unreal_actors takes its default from unreal.max_actors --------------------


@pytest.mark.asyncio
async def test_list_actors_defaults_to_max_actors() -> None:
    scripts: List[str] = []

    async def handler(request: web.Request) -> web.Response:
        scripts.append(unwrap(await request.json())["parameters"]["PythonCommand"])
        printed = {"actors": [], "count": 0, "truncated": False, "pie_active": False}
        return web.json_response(
            {
                "ReturnValue": True,
                "LogOutput": [{"Type": "Info", "Output": json.dumps(printed)}],
            }
        )

    async with endpoint(handler, max_actors=7) as session:
        await session.list_actors()
        await session.list_actors(max_results=9)

    # The page size the paged listing script receives in its args.
    assert "'max_results': 7," in scripts[0] and "'max_results': 9," in scripts[1]
