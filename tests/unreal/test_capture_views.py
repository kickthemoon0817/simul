"""Viewport and camera captures, and viewport info, against a UE 5.7-shaped editor.

An audit on UE 5.7.4 found that a capture during a Play In Editor session
showed the level editor's camera, not the game, with nothing in the reply to
say PIE was running; that ``get_unreal_viewport_info`` always returned null
field of view, size and projection; that the 256 KB inline cap kept even a
640x360 PNG from being inlined; and that nothing captured what a camera or
sensor actor sees.

The double below answers as UE 5.7.4 did (each answer verified live):
HighResShot sent through Remote Control writes a screenshot for every level
viewport pane that draws; run in the game world it goes to the Play session's
game viewport, which during Simulate In Editor never draws; editor Python reads
the player's camera and the game viewport's size; a transient SceneCapture2D
renders what a camera sees; and ImageWriteBlueprintLibrary writes PNG or JPEG.
The editor viewport's screenshots are blue, the game view's red and the scene
captures green, so a test can tell which view an image came from.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import sys
import traceback
from contextlib import asynccontextmanager, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import pytest
from aiohttp import web
from fastmcp import Client
from PIL import Image

from simul.adapters import unreal_runtime
from simul.adapters.unreal_runtime import (
    EDITOR_VIEW_DURING_PLAY_WARNING,
    GAME_VIEW_DID_NOT_DRAW,
    GAME_VIEW_FELL_BACK_WARNING,
    GAME_VIEW_UNAVAILABLE,
    VIEWPORT_FOV_NOTE,
    UnrealRuntimeSession,
    camera_fields_of_view,
    camera_intrinsics,
)
from simul.config import Settings
from simul.mcp.schemas.unreal import (
    UnrealCaptureCameraResponse,
    UnrealCaptureViewportResponse,
    UnrealViewportInfoResponse,
)
from simul.mcp.server import SimulMCPServer

PYTHON_LIBRARY = "/Script/PythonScriptPlugin.Default__PythonScriptLibrary"
SYSTEM_LIBRARY = "/Script/Engine.Default__KismetSystemLibrary"
EDITOR_SUBSYSTEM = "/Script/UnrealEd.Default__UnrealEditorSubsystem"
WORLD = "/Game/Maps/Lab.Lab"
PIE_WORLD = "/Game/Maps/UEDPIE_0_Lab.Lab"
EDITOR_BLUE, GAME_RED, CAPTURE_GREEN = (0, 0, 255), (255, 0, 0), (0, 255, 0)
VIEWPORT_SIZE = (1305, 1265)
GAME_VIEWPORT_SIZE = (1280, 720)


# ---------------------------------------------------------------------------
# The fake ``unreal`` module
# ---------------------------------------------------------------------------


class Vector:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.x, self.y, self.z = x, y, z


class Rotator:
    def __init__(self, pitch: float = 0.0, yaw: float = 0.0, roll: float = 0.0) -> None:
        self.pitch, self.yaw, self.roll = pitch, yaw, roll


class Projection:
    PERSPECTIVE = "PERSPECTIVE"
    ORTHOGRAPHIC = "ORTHOGRAPHIC"


class Component:
    def __init__(self, owner: "Actor", location: Vector, rotation: Rotator) -> None:
        self.owner, self.location, self.rotation = owner, location, rotation
        self.post_process_settings = SimpleNamespace(name="post process")
        self.post_process_blend_weight = 1.0

    def get_world_location(self) -> Vector:
        return self.location

    def get_world_rotation(self) -> Rotator:
        return self.rotation

    def get_class(self) -> SimpleNamespace:
        return SimpleNamespace(get_name=lambda: type(self).__name__)


class CameraComponent(Component):
    def __init__(self, *args: Any, fov: float = 90.0, aspect: Optional[float] = None, ortho: bool = False) -> None:
        super().__init__(*args)
        self.field_of_view = fov
        self.aspect_ratio = aspect or 1.7777778
        self.constrain_aspect_ratio = aspect is not None
        self.projection_mode = Projection.ORTHOGRAPHIC if ortho else Projection.PERSPECTIVE
        self.ortho_width = 1536.0


class CineCameraComponent(CameraComponent):
    """A 35 mm lens on a Super 35 filmback; field_of_view reads a stale 90, as UE 5.7.4's did."""

    def __init__(self, *args: Any) -> None:
        super().__init__(*args, fov=90.0)
        self.current_focal_length = 35.0
        self.filmback = SimpleNamespace(sensor_width=23.76, sensor_height=13.365)

    def get_horizontal_field_of_view(self) -> float:
        return math.degrees(2 * math.atan(self.filmback.sensor_width / (2 * self.current_focal_length)))


class SceneCaptureComponent2D(Component):
    def __init__(self, owner: "Actor", location: Vector, rotation: Rotator, editor: "Editor") -> None:
        super().__init__(owner, location, rotation)
        self.editor = editor
        self.fov_angle = 90.0
        self.projection_type = Projection.PERSPECTIVE
        self.ortho_width = 512.0
        self.properties: Dict[str, Any] = {}
        self.hidden: List[Any] = []

    def set_editor_property(self, name: str, value: Any) -> None:
        self.properties[name] = value

    def hide_actor_components(self, actor: Any, include_from_child_actors: bool = False) -> None:
        self.hidden.append(actor)

    def capture_scene(self) -> None:
        self.properties["texture_target"].pixels = CAPTURE_GREEN
        self.editor.captures.append(self)
        self.editor.events.append("capture")


class Guid:
    def __init__(self, text: str) -> None:
        self.text = text

    def to_string(self) -> str:
        return self.text


class Actor:
    def __init__(self, path: str, label: str, world: Any = None, guid: str = "") -> None:
        self.path, self.label, self.world = path, label, world
        self.guid = guid or label.upper()
        self.components: List[Component] = []

    def get_path_name(self) -> str:
        return self.path

    def get_name(self) -> str:
        return self.path.rsplit(".", 1)[-1]

    def get_editor_property(self, name: str) -> Any:
        return {"actor_guid": Guid(self.guid)}[name]

    def get_world(self) -> Any:
        return self.world

    def get_actor_label(self) -> str:
        return self.label

    def get_class(self) -> SimpleNamespace:
        return SimpleNamespace(get_name=lambda: "Actor")

    def get_component_by_class(self, cls: type) -> Optional[Component]:
        return next((c for c in self.components if isinstance(c, cls)), None)


class SceneCapture2D(Actor):
    """The transient capture actor the tool spawns."""

    def __init__(self, editor: "Editor", world: str, location: Vector, rotation: Rotator) -> None:
        super().__init__(world + ":PersistentLevel.SceneCapture2D_99", "SceneCapture2D")
        self.editor, self.world = editor, world
        self.capture_component2d = SceneCaptureComponent2D(self, location, rotation, editor)
        self.finished = False

    @staticmethod
    def static_class() -> str:
        return "SceneCapture2DClass"

    def destroy_actor(self) -> None:
        self.editor.destroyed.append(self)
        self.editor.events.append("destroy")


class RenderTarget:
    def __init__(self, width: int, height: int) -> None:
        self.width, self.height, self.pixels = width, height, (0, 0, 0)


class Texture:
    def __init__(self, path: str) -> None:
        self.path = path


def write_png(path: str, size: Tuple[int, int], color: Tuple[int, int, int], noise: bool = False) -> None:
    if noise:
        # Incompressible pixels, so the PNG is far above the inline cap.
        image = Image.frombytes("RGB", size, os.urandom(size[0] * size[1] * 3))
    else:
        image = Image.new("RGB", size, color)
    image.save(path, format="PNG")


class Editor:
    """The editor: its screenshot directory, PIE state, viewports and actors."""

    def __init__(self, root: Path) -> None:
        self.saved = root / "Saved"
        self.shots = self.saved / "Screenshots" / "MacEditor"
        self.shots.mkdir(parents=True)
        self.pie: Optional[str] = None  # None, "play" or "simulate"
        self.panes_drawing = 1
        self.noise = False
        self.counter = 0
        self.commands: List[Tuple[str, str]] = []
        self.captures: List[SceneCaptureComponent2D] = []
        self.spawned: List[SceneCapture2D] = []
        self.destroyed: List[SceneCapture2D] = []
        # Transactions, spawns, captures, render target releases and removals, in order.
        self.events: List[str] = []
        self.released: List[RenderTarget] = []
        # A game-world HighResShot during Simulate: written late, by flush_pending().
        self.pending: Optional[Tuple[str, Tuple[int, int]]] = None
        # Called as an editor HighResShot fires, after the capture's directory listing.
        self.before_editor_shot: Optional[Any] = None
        self.camera = {"CameraLocation": {"X": -500.0, "Y": 0.0, "Z": 250.0},
                       "CameraRotation": {"Pitch": -15.0, "Yaw": 0.0, "Roll": 0.0}}
        self.actors: Dict[str, Actor] = {}
        self.editor_world = SimpleNamespace(get_path_name=lambda: WORLD,
                                            get_outermost=lambda: SimpleNamespace(get_name=lambda: "/Game/Maps/Lab"))
        self.game_world = SimpleNamespace(
            get_path_name=lambda: PIE_WORLD,
            get_outermost=lambda: SimpleNamespace(get_name=lambda: "/Game/Maps/UEDPIE_0_Lab"),
        )

    # -- actors -------------------------------------------------------------

    def add_actor(self, actor: Actor) -> Actor:
        self.actors[actor.path] = actor
        return actor

    def place_cameras(self, world: str = WORLD) -> Dict[str, Actor]:
        made = {}
        # actor_at accepts an actor of the editor world only (UE-2's live-actor check).
        owner = self.editor_world if world == WORLD else self.game_world
        for name, component in (
            ("Camera", lambda a: CameraComponent(a, Vector(-200, 0, 150), Rotator(-10, 0, 0))),
            ("Square", lambda a: CameraComponent(a, Vector(0, 0, 0), Rotator(), aspect=1.7777778)),
            ("Ortho", lambda a: CameraComponent(a, Vector(0, 0, 900), Rotator(-90, 0, 0), ortho=True)),
            ("Cine", lambda a: CineCameraComponent(a, Vector(300, -600, 150), Rotator(-10, 90, 0))),
        ):
            actor = Actor(world + ":PersistentLevel." + name, name, owner)
            actor.components.append(component(actor))
            made[name] = self.actors[actor.path] = actor
        sensor = Actor(world + ":PersistentLevel.TopCapture", "TopCapture", owner)
        sensor.components.append(SceneCaptureComponent2D(sensor, Vector(300, 0, 600), Rotator(-89, 0, 0), self))
        made["TopCapture"] = self.actors[sensor.path] = sensor
        crate = Actor(world + ":PersistentLevel.Crate", "Crate", owner)
        made["Crate"] = self.actors[crate.path] = crate
        return made

    # -- HighResShot ------------------------------------------------------------

    def next_shot(self) -> str:
        self.counter += 1
        return str(self.shots / f"HighresScreenshot{self.counter:05d}.png")

    def shot_target(self, command: str, viewport: Tuple[int, int]) -> Tuple[str, Tuple[int, int]]:
        """Where a HighResShot writes, and at what size: ``filename=`` names the file, else it is numbered."""
        words = command.split()
        size = viewport if words[1] == "1" else tuple(int(v) for v in words[1].split("x"))
        named = [word.split("=", 1)[1] for word in words[2:] if word.startswith("filename=")]
        return (str(self.shots / (named[0] + ".png")) if named else self.next_shot()), size

    def high_res_shot(self, command: str, color: Tuple[int, int, int], viewport: Tuple[int, int]) -> None:
        path, size = self.shot_target(command, viewport)
        write_png(path, size, color, noise=self.noise)

    def editor_console_command(self, command: str) -> None:
        """Remote Control's ExecuteConsoleCommand: every drawing level viewport pane takes the shot."""
        self.commands.append(("editor", command))
        if self.before_editor_shot is not None:
            self.before_editor_shot()
        for _ in range(self.panes_drawing):
            self.high_res_shot(command, EDITOR_BLUE, VIEWPORT_SIZE)

    def execute_console_command(self, world: Any, command: str) -> None:
        """SystemLibrary.execute_console_command: in the game world, to its game viewport.

        During Simulate the shot is left pending, as the level viewport showing
        the simulation wrote it late or not at all on UE 5.7.4.
        """
        self.commands.append(("game" if world is self.game_world else "other", command))
        if world is self.game_world and self.pie == "play":
            self.high_res_shot(command, GAME_RED, GAME_VIEWPORT_SIZE)
        elif world is self.game_world and self.pie == "simulate":
            self.pending = self.shot_target(command, GAME_VIEWPORT_SIZE)

    def flush_pending(self) -> Optional[str]:
        """The level viewport finally draws a pending Simulate shot."""
        if self.pending is None:
            return None
        path, size = self.pending
        self.pending = None
        write_png(path, size, EDITOR_BLUE)
        return path

    # -- the module -------------------------------------------------------------

    def module(self) -> SimpleNamespace:
        editor = self
        level_editor = SimpleNamespace(
            is_in_play_in_editor=lambda: editor.pie is not None,
            get_viewport_config_keys=lambda: [f"FourPanes2x2.Viewport 1.Viewport{i}" for i in range(4)],
            get_active_viewport_config_key=lambda: "FourPanes2x2.Viewport 1.Viewport1",
        )
        unreal_editor = SimpleNamespace(
            get_editor_world=lambda: None if editor.pie else editor.editor_world,
            get_game_world=lambda: editor.game_world if editor.pie else None,
        )

        def spawn_actor_from_class(cls: Any, location: Vector, rotation: Rotator, transient: bool = False) -> Any:
            assert cls is SceneCapture2D and transient, "a transient scene capture, nothing else"
            assert editor.pie is None, "the editor spawns nothing during PIE"
            actor = SceneCapture2D(editor, WORLD, location, rotation)
            editor.spawned.append(actor)
            editor.events.append("spawn")
            return actor

        def destroy_actor(actor: Any) -> bool:
            editor.destroyed.append(actor)
            editor.events.append("destroy")
            return True

        def begin_transaction(context: str, description: str, primary: Any) -> int:
            editor.events.append("begin")
            return 7

        def cancel_transaction(index: int) -> None:
            assert index == 7, "cancelled with the index begin_transaction returned"
            editor.events.append("cancel")

        def release_render_target2d(target: RenderTarget) -> None:
            editor.released.append(target)
            editor.events.append("release")

        def all_actors_of_class(world: Any, cls: Any) -> List[Actor]:
            return [actor for actor in editor.actors.values() if actor.get_world() is world]

        def call_method(name: str, args: tuple) -> Any:
            if name == "BeginDeferredActorSpawnFromClass":
                world, cls, transform = args
                assert world is editor.game_world and cls == "SceneCapture2DClass"
                actor = SceneCapture2D(editor, PIE_WORLD, transform.location, transform.rotation)
                editor.spawned.append(actor)
                editor.events.append("spawn")
                return actor
            if name == "FinishSpawningActor":
                args[0].finished = True
                return args[0]
            raise AssertionError(name)

        def camera_manager(world: Any, index: int) -> Any:
            if world is not editor.game_world:
                return None
            return SimpleNamespace(
                get_camera_location=lambda: Vector(0, 0, 92),
                get_camera_rotation=lambda: Rotator(0, 0, 0),
                get_fov_angle=lambda: 75.0,
            )

        def export_to_disk(texture: Any, path: str, options: Any) -> None:
            assert options.async_ is False
            if isinstance(texture, Texture):
                image = Image.open(texture.path).convert("RGB")
            else:
                image = Image.new("RGB", (texture.width, texture.height), texture.pixels)
            image.save(path, format="JPEG" if options.format == "JPG" else "PNG", quality=options.compression_quality)

        def find_object(outer: Any, path: str) -> Any:
            return editor.actors.get(path)

        def load_object(outer: Any, path: str) -> Any:
            return editor.actors.get(path) if path.startswith(WORLD + ":") else None

        subsystems = {
            "LevelEditorSubsystem": level_editor,
            "UnrealEditorSubsystem": unreal_editor,
            "EditorActorSubsystem": SimpleNamespace(
                spawn_actor_from_class=spawn_actor_from_class, destroy_actor=destroy_actor
            ),
            "LayersSubsystem": SimpleNamespace(get_world=lambda: editor.editor_world),
        }
        return SimpleNamespace(
            Actor=Actor,
            Vector=Vector,
            LinearColor=lambda *channels: channels,
            Transform=lambda location, rotation, scale: SimpleNamespace(location=location, rotation=rotation),
            CameraComponent=CameraComponent,
            CineCameraComponent=CineCameraComponent,
            SceneCaptureComponent2D=SceneCaptureComponent2D,
            SceneCapture2D=SceneCapture2D,
            CameraProjectionMode=Projection,
            SceneCaptureSource=SimpleNamespace(SCS_FINAL_COLOR_LDR="FinalColorLDR"),
            TextureRenderTargetFormat=SimpleNamespace(RTF_RGBA8="RGBA8"),
            DesiredImageFormat=SimpleNamespace(PNG="PNG", JPG="JPG"),
            ImageWriteOptions=lambda: SimpleNamespace(),
            ImageWriteBlueprintLibrary=SimpleNamespace(export_to_disk=export_to_disk),
            RenderingLibrary=SimpleNamespace(
                import_file_as_texture2d=lambda world, path: Texture(path),
                create_render_target2d=lambda world, width, height, fmt, color, mips: RenderTarget(width, height),
                release_render_target2d=release_render_target2d,
            ),
            WidgetLayoutLibrary=SimpleNamespace(
                get_viewport_size=lambda world: SimpleNamespace(x=float(GAME_VIEWPORT_SIZE[0]),
                                                                y=float(GAME_VIEWPORT_SIZE[1]))
            ),
            GameplayStatics=SimpleNamespace(
                get_player_controller=lambda world, index: object() if world is editor.game_world else None,
                get_player_camera_manager=camera_manager,
                get_all_actors_of_class=all_actors_of_class,
            ),
            SystemLibrary=SimpleNamespace(
                execute_console_command=editor.execute_console_command,
                is_valid=lambda obj: obj is not None,
                begin_transaction=begin_transaction,
                cancel_transaction=cancel_transaction,
            ),
            Paths=SimpleNamespace(
                project_saved_dir=lambda: str(editor.saved),
                screen_shot_dir=lambda: str(editor.shots) + "/",
                convert_relative_path_to_full=lambda path: path,
            ),
            EditorActorSubsystem="EditorActorSubsystem",
            LevelEditorSubsystem="LevelEditorSubsystem",
            UnrealEditorSubsystem="UnrealEditorSubsystem",
            LayersSubsystem="LayersSubsystem",
            get_editor_subsystem=subsystems.__getitem__,
            get_default_object=lambda cls: SimpleNamespace(call_method=call_method),
            find_object=find_object,
            load_object=load_object,
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
            "LogOutput": [{"Type": "Info", "Output": line} for line in output.getvalue().splitlines()],
        }


@pytest.fixture
def editor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Editor:
    level = Editor(tmp_path)
    monkeypatch.setitem(sys.modules, "unreal", level.module())
    # The poll interval and the game view's wait, shortened for the suite.
    monkeypatch.setattr(unreal_runtime, "_CAPTURE_POLL_SECONDS", 0.01)
    monkeypatch.setattr(unreal_runtime, "_GAME_VIEW_WAIT_SECONDS", 0.2)
    return level


@asynccontextmanager
async def remote_control(editor: Editor) -> AsyncIterator[Settings]:
    """Serve Remote Control for ``editor``."""

    async def handler(request: web.Request) -> web.Response:
        body = await request.json()
        target, function = body.get("objectPath"), body.get("functionName")
        parameters = body.get("parameters", {})
        if target == PYTHON_LIBRARY and function == "ExecutePythonCommandEx":
            return web.json_response(editor.run_python(parameters["PythonCommand"]))
        if target == SYSTEM_LIBRARY and function == "ExecuteConsoleCommand":
            editor.editor_console_command(parameters["Command"])
            return web.json_response({})
        if target == EDITOR_SUBSYSTEM and function == "GetLevelViewportCameraInfo":
            return web.json_response({**editor.camera, "ReturnValue": True})
        return web.json_response({"errorMessage": "not served"}, status=400)

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


async def call_tool(settings: Settings, name: str, arguments: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Any]]:
    """Call one tool through a real MCP client; return the JSON and the image blocks."""
    server = SimulMCPServer(settings, backends={"unreal"})
    async with Client(server.mcp) as client:
        result = await client.call_tool(name, arguments, raise_on_error=False)
    images = [block for block in result.content if block.type == "image"]
    return json.loads(result.content[-1].text), images


def color_of(image_block: Any) -> Tuple[int, int, int]:
    with Image.open(io.BytesIO(base64.b64decode(image_block.data))) as image:
        return image.convert("RGB").getpixel((image.width // 2, image.height // 2))


def near(color: Tuple[int, int, int], expected: Tuple[int, int, int]) -> bool:
    return all(abs(a - b) <= 8 for a, b in zip(color, expected))


def validated(model: Any, payload: Dict[str, Any]) -> None:
    model(**payload)


# ---------------------------------------------------------------------------
# capture_unreal_viewport: which view
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_outside_pie_the_editor_viewport_is_captured_and_said_so(editor: Editor) -> None:
    async with remote_control(editor) as settings:
        payload, images = await call_tool(
            settings, "capture_unreal_viewport", {"resolution_x": 64, "resolution_y": 36, "inline": True}
        )
    validated(UnrealCaptureViewportResponse, payload)
    assert (payload["view"], payload["pie_active"], payload["warnings"]) == ("editor", False, [])
    assert editor.commands == [("editor", "HighResShot 64x36")]
    assert near(color_of(images[0]), EDITOR_BLUE)
    assert (payload["inline_width"], payload["inline_height"]) == (64, 36)


@pytest.mark.asyncio
async def test_during_a_play_session_auto_captures_the_players_game_view(editor: Editor) -> None:
    """The capture used to follow the editor camera during Play, with no pie_active."""
    editor.pie = "play"
    async with remote_control(editor) as settings:
        game, game_images = await call_tool(
            settings, "capture_unreal_viewport", {"resolution_x": 64, "resolution_y": 36, "inline": True}
        )
        asked, asked_images = await call_tool(
            settings,
            "capture_unreal_viewport",
            {"resolution_x": 64, "resolution_y": 36, "inline": True, "view": "editor"},
        )
    assert (game["view"], game["pie_active"], game["warnings"]) == ("game", True, [])
    assert near(color_of(game_images[0]), GAME_RED)
    assert (asked["view"], asked["warnings"]) == ("editor", [EDITOR_VIEW_DURING_PLAY_WARNING])
    assert near(color_of(asked_images[0]), EDITOR_BLUE)
    # The game shot is named for its call; the editor shot keeps Unreal's numbered names.
    (where, command), editor_shot = editor.commands
    assert where == "game" and re.fullmatch(r"HighResShot 64x36 filename=SimulShot_[0-9a-f]{16}", command)
    assert Path(game["path"]).name == command.rsplit("=", 1)[1] + ".png"
    assert editor_shot == ("editor", "HighResShot 64x36")


@pytest.mark.asyncio
async def test_during_simulate_auto_falls_back_to_the_level_viewport_and_game_is_refused(
    editor: Editor,
) -> None:
    """Simulate In Editor has a player controller whose game viewport never draws."""
    editor.pie = "simulate"
    async with remote_control(editor) as settings:
        auto, images = await call_tool(
            settings, "capture_unreal_viewport", {"resolution_x": 64, "resolution_y": 36, "inline": True}
        )
        game, _ = await call_tool(
            settings, "capture_unreal_viewport", {"resolution_x": 64, "resolution_y": 36, "view": "game"}
        )
    assert (auto["view"], auto["pie_active"], auto["warnings"]) == ("editor", True, [GAME_VIEW_FELL_BACK_WARNING])
    assert near(color_of(images[0]), EDITOR_BLUE)
    assert (game["success"], game["error_type"], game["error"]) == (False, "InvalidState", GAME_VIEW_DID_NOT_DRAW)
    assert Path(game["details"]["pending_path"]).name.startswith("SimulShot_")


@pytest.mark.asyncio
async def test_a_shot_written_after_its_call_gave_up_is_never_taken_by_the_next(editor: Editor) -> None:
    """During Simulate a refused game shot was written 0.36 s after the reply,
    as a numbered file the next capture could take (UE 5.7.4)."""
    editor.pie = "simulate"
    async with remote_control(editor) as settings:
        refused, _ = await call_tool(
            settings, "capture_unreal_viewport", {"resolution_x": 320, "resolution_y": 180, "view": "game"}
        )
        # The late shot lands between the next capture's listing and its own shot, and so does a
        # numbered file of another size.
        late: List[str] = []
        editor.before_editor_shot = lambda: (
            late.append(editor.flush_pending() or ""),
            write_png(editor.next_shot(), (32, 32), GAME_RED),
        )
        taken, images = await call_tool(
            settings,
            "capture_unreal_viewport",
            {"resolution_x": 64, "resolution_y": 36, "inline": True, "view": "editor"},
        )
    assert late == [refused["details"]["pending_path"]] and Path(late[0]).exists()
    assert near(color_of(images[0]), EDITOR_BLUE)
    assert taken["path"] != late[0] and (taken["view"], taken["pie_active"]) == ("editor", True)
    # Neither stray counts as another pane.
    assert not any("panes drew" in warning for warning in taken["warnings"])


@pytest.mark.asyncio
async def test_the_game_view_outside_pie_is_refused_before_anything_fires(editor: Editor) -> None:
    async with remote_control(editor) as settings:
        payload, _ = await call_tool(settings, "capture_unreal_viewport", {"view": "game"})
    assert (payload["error_type"], payload["error"]) == ("InvalidState", GAME_VIEW_UNAVAILABLE)
    assert editor.commands == []


@pytest.mark.asyncio
async def test_several_drawing_panes_are_warned_of(editor: Editor) -> None:
    editor.panes_drawing = 3
    async with remote_control(editor) as settings:
        session = UnrealRuntimeSession(settings)
        try:
            result = await session.capture_viewport(64, 36)
        finally:
            await session.close()
    assert "3 level viewport panes drew this HighResShot" in result["warnings"][0]


# ---------------------------------------------------------------------------
# capture_unreal_viewport: JPEG and scaled inline copies
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_png_over_the_inline_cap_is_inlined_as_jpeg_or_as_a_scaled_copy(editor: Editor) -> None:
    """A 640x360 PNG (about 690 KB of noise here, 307 KB in the audit) could not be inlined."""
    editor.noise = True
    request = {"resolution_x": 640, "resolution_y": 360, "inline": True}
    async with remote_control(editor) as settings:
        png, png_images = await call_tool(settings, "capture_unreal_viewport", request)
        jpeg, jpeg_images = await call_tool(
            settings, "capture_unreal_viewport", {**request, "format": "jpeg", "jpeg_quality": 50}
        )
        scaled, scaled_images = await call_tool(settings, "capture_unreal_viewport",
                                                {**request, "inline_max_size": 160})
    assert png_images == [] and "pass format \"jpeg\", or pass inline_max_size" in png["inline_skipped"]
    assert png["size_bytes"] > unreal_runtime.MAX_INLINE_CAPTURE_BYTES
    assert jpeg["path"].endswith(".jpg") and jpeg["format"] == "jpeg"
    # The JPEG replaces the PNG HighResShot wrote.
    assert not Path(jpeg["path"]).with_suffix(".png").exists()
    assert jpeg_images[0].mime_type == "image/jpeg" and jpeg["size_bytes"] < png["size_bytes"]
    with Image.open(jpeg["path"]) as written:
        assert (written.format, written.size) == ("JPEG", (640, 360))
    assert (scaled["inline_width"], scaled["inline_height"], scaled["inline_skipped"]) == (160, 90, None)
    with Image.open(io.BytesIO(base64.b64decode(scaled_images[0].data))) as copy:
        assert copy.size == (160, 90)
    with Image.open(scaled["path"]) as full:
        assert full.size == (640, 360)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"view": "player"}, "view must be one of auto, editor, game"),
        ({"jpeg_quality": 0}, "jpeg_quality must be between 1 and 100"),
        ({"inline_max_size": 8}, "inline_max_size must be 0 or between 16 and 16384"),
    ],
)
async def test_capture_refuses_bad_options_before_any_request(kwargs: Dict[str, Any], message: str) -> None:
    session = UnrealRuntimeSession(Settings())
    with pytest.raises(ValueError, match=message):
        await session.capture_viewport(**kwargs)


# ---------------------------------------------------------------------------
# get_unreal_viewport_info
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_viewport_info_says_what_it_cannot_read_and_measures_the_size_on_request(editor: Editor) -> None:
    """fov, viewport_size and projection_type were always null, unexplained."""
    async with remote_control(editor) as settings:
        info, _ = await call_tool(settings, "get_unreal_viewport_info", {})
        measured, _ = await call_tool(settings, "get_unreal_viewport_info", {"measure_size": True})
    validated(UnrealViewportInfoResponse, info)
    assert (info["projection_type"], info["fov"], info["viewport_size"], info["pie_active"]) == (
        "Perspective", None, None, False
    )
    assert info["notes"][0] == VIEWPORT_FOV_NOTE
    assert any("4 level viewport panes" in note for note in info["notes"])
    assert any("measure_size" in note for note in info["notes"])
    assert info["camera_location"] == [-500.0, 0.0, 250.0]
    assert measured["viewport_size"] == list(VIEWPORT_SIZE)
    # The measuring screenshot was deleted.
    assert list(editor.shots.iterdir()) == []
    editor.panes_drawing = 3
    async with remote_control(editor) as settings:
        three, _ = await call_tool(settings, "get_unreal_viewport_info", {"measure_size": True})
    assert three["viewport_size"] == list(VIEWPORT_SIZE)
    assert any("3 viewport panes" in note for note in three["notes"])
    # Every pane's file is gone, not only the newest.
    assert list(editor.shots.iterdir()) == []


@pytest.mark.asyncio
async def test_viewport_info_during_play_reports_the_players_view(editor: Editor) -> None:
    editor.pie = "play"
    async with remote_control(editor) as settings:
        info, _ = await call_tool(settings, "get_unreal_viewport_info", {})
    assert info["pie_active"] is True
    assert info["game_view"] == {
        "camera_location": [0.0, 0.0, 92.0],
        "camera_rotation": [0.0, 0.0, 0.0],
        "fov": 75.0,
        "viewport_size": list(GAME_VIEWPORT_SIZE),
    }
    assert info["viewport_size"] == list(GAME_VIEWPORT_SIZE)


# ---------------------------------------------------------------------------
# capture_unreal_camera
# ---------------------------------------------------------------------------


def test_intrinsics_are_the_pinhole_model_of_the_image() -> None:
    k = camera_intrinsics(90.0, 640, 360, False)
    assert k == {"fx": pytest.approx(320.0), "fy": pytest.approx(320.0), "cx": 319.5, "cy": 179.5,
                 "width": 640, "height": 360}
    fields = camera_fields_of_view(90.0, 640, 360, False)
    assert fields["vertical_fov"] == pytest.approx(math.degrees(2 * math.atan(360 / 640)))
    assert camera_intrinsics(90.0, 640, 360, True) is None
    assert camera_fields_of_view(90.0, 640, 360, True) == {"horizontal_fov": None, "vertical_fov": None}


@pytest.mark.asyncio
async def test_a_camera_actor_is_captured_through_a_transient_scene_capture(editor: Editor) -> None:
    cameras = editor.place_cameras()
    async with remote_control(editor) as settings:
        payload, images = await call_tool(
            settings,
            "capture_unreal_camera",
            {"actor_path": cameras["Camera"].path, "resolution_x": 64, "resolution_y": 36, "inline": True},
        )
    validated(UnrealCaptureCameraResponse, payload)
    assert near(color_of(images[0]), CAPTURE_GREEN)
    assert (payload["world"], payload["pie_active"], payload["captured_actor"]) == (
        "editor", False, cameras["Camera"].path
    )
    assert payload["camera"]["location"] == [-200.0, 0.0, 150.0]
    assert payload["camera"]["horizontal_fov"] == 90.0 and payload["camera"]["projection"] == "perspective"
    assert payload["intrinsics"]["fx"] == pytest.approx(32.0)
    (capture,) = editor.captures
    assert capture.properties["capture_source"] == "FinalColorLDR"
    assert capture.properties["fov_angle"] == 90.0
    assert capture.properties["post_process_settings"].name == "post process"
    assert capture.hidden == [cameras["Camera"]]
    # The capture actor was transient and is gone, inside a transaction that was cancelled,
    # so Undo cannot bring it back; its render target was detached and released.
    assert editor.destroyed == editor.spawned and len(editor.spawned) == 1
    assert editor.events == ["begin", "spawn", "capture", "release", "destroy", "cancel"]
    assert capture.properties["texture_target"] is None and len(editor.released) == 1
    assert Path(payload["path"]).name.startswith("SimulCamera_Camera_")


@pytest.mark.asyncio
async def test_a_cine_camera_uses_its_lens_not_its_stale_field_of_view(editor: Editor) -> None:
    cameras = editor.place_cameras()
    async with remote_control(editor) as settings:
        payload, _ = await call_tool(
            settings, "capture_unreal_camera", {"actor_path": cameras["Cine"].path, "format": "jpeg"}
        )
    hfov = math.degrees(2 * math.atan(23.76 / 70.0))
    assert payload["camera"]["horizontal_fov"] == pytest.approx(hfov)
    assert payload["camera"]["lens"] == {"focal_length_mm": 35.0, "sensor_width_mm": 23.76,
                                         "sensor_height_mm": 13.365}
    # fx = focal length in pixels: 35 mm on a 23.76 mm sensor 1280 pixels wide.
    assert payload["intrinsics"]["fx"] == pytest.approx(35.0 * 1280 / 23.76, rel=1e-6)
    assert editor.captures[0].properties["fov_angle"] == pytest.approx(hfov)
    with Image.open(payload["path"]) as written:
        assert written.format == "JPEG"


@pytest.mark.asyncio
async def test_a_scene_capture_and_an_orthographic_camera(editor: Editor) -> None:
    cameras = editor.place_cameras()
    async with remote_control(editor) as settings:
        session = UnrealRuntimeSession(settings)
        try:
            top = await session.capture_camera(cameras["TopCapture"].path, 32, 32)
            ortho = await session.capture_camera(cameras["Ortho"].path, 32, 32)
        finally:
            await session.close()
    assert top["camera"]["component"] == "SceneCaptureComponent2D"
    assert ortho["intrinsics"] is None and ortho["camera"]["projection"] == "orthographic"
    assert ortho["camera"]["ortho_width"] == 1536.0
    assert editor.captures[1].properties["projection_type"] == Projection.ORTHOGRAPHIC


@pytest.mark.asyncio
async def test_a_capture_of_another_shape_than_the_camera_is_warned_of(editor: Editor) -> None:
    cameras = editor.place_cameras()
    async with remote_control(editor) as settings:
        payload, _ = await call_tool(
            settings,
            "capture_unreal_camera",
            {"actor_path": cameras["Square"].path, "resolution_x": 40, "resolution_y": 40},
        )
    assert "aspect ratio of 1.778" in payload["warnings"][0]
    assert payload["camera"]["vertical_fov"] == pytest.approx(90.0)


@pytest.mark.asyncio
async def test_during_pie_the_running_games_counterpart_is_captured(editor: Editor) -> None:
    editor.place_cameras()
    game_cameras = editor.place_cameras(PIE_WORLD)
    editor.pie = "play"
    async with remote_control(editor) as settings:
        payload, _ = await call_tool(
            settings, "capture_unreal_camera", {"actor_path": WORLD + ":PersistentLevel.Camera"}
        )
    assert (payload["world"], payload["pie_active"], payload["captured_actor"]) == (
        "game", True, game_cameras["Camera"].path
    )
    (spawned,) = editor.spawned
    assert spawned.world == PIE_WORLD and spawned.finished and editor.destroyed == [spawned]
    # No editor transaction in the game world; the render target is released there too.
    assert editor.events == ["spawn", "capture", "release", "destroy"]


@pytest.mark.asyncio
async def test_during_pie_a_camera_streamed_into_a_world_partition_cell_is_found(editor: Editor) -> None:
    """A spatially loaded camera lives in a cell level of its own during PIE, not
    under the game world's persistent level, and was refused as PIEActive."""
    cameras = editor.place_cameras()
    cell = editor.add_actor(
        Actor("/Memory/UEDPIE_0_Lab_9LILJQ0OB4CL3P64.Lab:PersistentLevel.Camera", "Camera", editor.game_world)
    )
    cell.components.append(CameraComponent(cell, Vector(-200, 0, 150), Rotator(-10, 0, 0)))
    editor.pie = "play"
    async with remote_control(editor) as settings:
        payload, _ = await call_tool(settings, "capture_unreal_camera", {"actor_path": cameras["Camera"].path})
    assert (payload["world"], payload["captured_actor"]) == ("game", cell.path), payload


@pytest.mark.asyncio
async def test_during_pie_the_same_name_in_two_cells_is_told_apart_by_guid(editor: Editor) -> None:
    cameras = editor.place_cameras()
    for cell, guid in (("A", "OTHER"), ("B", cameras["Camera"].guid)):
        twin = editor.add_actor(Actor(f"/Memory/UEDPIE_0_Lab_{cell}.Lab:PersistentLevel.Camera", "Camera",
                                      editor.game_world, guid=guid))
        twin.components.append(CameraComponent(twin, Vector(0, 0, 0), Rotator()))
    editor.pie = "play"
    async with remote_control(editor) as settings:
        payload, _ = await call_tool(settings, "capture_unreal_camera", {"actor_path": cameras["Camera"].path})
    assert payload["captured_actor"] == "/Memory/UEDPIE_0_Lab_B.Lab:PersistentLevel.Camera"


@pytest.mark.asyncio
async def test_during_pie_a_running_games_camera_is_captured_by_its_path(editor: Editor) -> None:
    """A camera the game spawned has no editor actor; its PIE path is taken as given."""
    editor.place_cameras()
    pawn = editor.add_actor(Actor(PIE_WORLD + ":PersistentLevel.DefaultPawn_0", "DefaultPawn", editor.game_world))
    pawn.components.append(CameraComponent(pawn, Vector(0, 0, 92), Rotator()))
    editor.pie = "play"
    async with remote_control(editor) as settings:
        payload, _ = await call_tool(settings, "capture_unreal_camera", {"actor_path": pawn.path})
        editor.pie = None
        stopped, _ = await call_tool(settings, "capture_unreal_camera", {"actor_path": pawn.path})
    assert (payload["actor_path"], payload["captured_actor"], payload["world"]) == (pawn.path, pawn.path, "game")
    # Outside PIE a game actor is no actor of the level open in the editor.
    assert stopped["error_type"] == "NotFoundError"


@pytest.mark.asyncio
async def test_camera_capture_refusals(editor: Editor) -> None:
    cameras = editor.place_cameras()
    async with remote_control(editor) as settings:
        not_camera, _ = await call_tool(settings, "capture_unreal_camera", {"actor_path": cameras["Crate"].path})
        editor.pie = "play"
        no_counterpart, _ = await call_tool(settings, "capture_unreal_camera", {"actor_path": cameras["Camera"].path})
    assert (not_camera["error_type"], editor.spawned) == ("ValidationError", [])
    assert not_camera["details"]["code"] == "NotACamera"
    assert "has no camera or scene capture component" in not_camera["error"]
    assert no_counterpart["error_type"] == "PIEActive"
