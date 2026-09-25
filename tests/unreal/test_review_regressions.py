"""Regressions for #188–196: actual HTTP bodies and executable UE scripts.

The API-shaped doubles below deliberately lack the invented actor-level
component methods and describe-response values that hid these defects.
"""

import asyncio
import base64
import io
import json
import sys
from contextlib import asynccontextmanager, redirect_stdout
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from PIL import Image

from simul_mcp.adapters import unreal_runtime
from simul_mcp.adapters.unreal_runtime import UnrealRuntimeSession
from simul_mcp.config import Settings


@asynccontextmanager
async def endpoint(handler):
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    base = Settings()
    cfg = base.unreal.model_copy(
        update={
            "host": "127.0.0.1",
            "port": site._server.sockets[0].getsockname()[1],
            "timeout": 0.03,
            "retry_base_delay": 0.001,
            "max_retries": 3,
        }
    )
    session = UnrealRuntimeSession(base.model_copy(update={"unreal": cfg}))
    try:
        yield session
    finally:
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "server_error"])
async def test_mutation_is_never_replayed_after_dispatch(failure):
    applied = []

    async def handler(request):
        applied.append(await request.json())
        if failure == "timeout":
            await asyncio.sleep(0.08)
        return web.json_response(
            {"error": "response lost after applying mutation"}, status=503
        )

    async with endpoint(handler) as session:
        with pytest.raises(Exception):
            await session.spawn_actor("/Script/Engine.StaticMeshActor")
    assert len(applied) == 1


@pytest.mark.asyncio
async def test_safe_read_still_retries():
    requests = []

    async def handler(request):
        requests.append(request.path)
        return web.json_response(
            {"ok": True}, status=503 if len(requests) == 1 else 200
        )

    async with endpoint(handler) as session:
        assert await session._http_get("/remote/info") == {"ok": True}
    assert len(requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
async def test_thumbnail_reads_binary_and_preserves_actual_format(fmt):
    buf = io.BytesIO()
    Image.new("RGB", (32, 48), "red").save(buf, format=fmt)
    image = buf.getvalue()

    async def handler(request):
        return web.Response(body=image, content_type=f"image/{fmt.lower()}")

    async with endpoint(handler) as session:
        result = await session.get_actor_thumbnail("/Engine/BasicShapes/Cube.Cube")
    assert base64.b64decode(result["image_base64"]) == image
    assert (result["width"], result["height"], result["format"]) == (
        32,
        48,
        fmt.lower(),
    )


@pytest.mark.asyncio
async def test_generic_dispatch_refused_without_any_network_request():
    base = Settings()
    session = UnrealRuntimeSession(
        base.model_copy(
            update={
                "security": base.security.model_copy(
                    update={"allow_script_execution": False}
                ),
            }
        )
    )
    session._http_put = AsyncMock(side_effect=AssertionError("must not send"))
    for result in (
        await session.call_actor_function(
            "/Script/PythonScriptPlugin.Default__PythonScriptLibrary",
            "ExecutePythonCommandEx",
            '{"PythonCommand":"print(123)"}',
        ),
        await session.batch_operations(
            [{"Url": "/remote/object/call", "Verb": "PUT", "Body": {}}]
        ),
    ):
        assert result["success"] is False
        assert result["error_type"] == "ScriptExecutionDisabled"
    session._http_put.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,value",
    [
        ("py", "print(123)"),
        ("r.ScreenPercentage", "100; py print(123)"),
        ("r.ScreenPercentage\npy", "100"),
    ],
)
async def test_render_setting_cannot_be_used_as_console_dispatch(name, value):
    session = UnrealRuntimeSession()
    session._http_put = AsyncMock()
    with pytest.raises(ValueError):
        await session.set_render_settings(name, value)
    session._http_put.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("compute_convex_hull", {}),
        ("decompose_convex_hull", {}),
        ("edit_mesh_topology", {"operation": "extrude_faces"}),
        ("subdivide_mesh", {}),
        ("simplify_mesh", {}),
        ("cut_mesh_plane", {"plane_origin": [0, 0, 0], "plane_normal": [0, 0, 1]}),
        ("validate_mesh", {}),
        ("convert_mesh_format", {"target_format": "dynamic_mesh"}),
        ("remesh_mesh", {}),
        ("compute_mesh_uv", {}),
    ],
)
async def test_generated_mesh_paths_are_data_not_python(method, kwargs):
    import ast

    session = UnrealRuntimeSession()
    session._execute_python = AsyncMock(
        return_value={
            "ReturnValue": True,
            "LogOutput": [{"Type": "Info", "Output": "{}"}],
        }
    )
    malicious_path = "'; review_injected(); #\\\n"
    await getattr(session, method)(mesh_path=malicious_path, **kwargs)
    script = session._execute_python.call_args.args[0]
    tree = ast.parse(script)
    assert not any(
        isinstance(node, ast.Name) and node.id == "review_injected"
        for node in ast.walk(tree)
    )
    assert any(
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and malicious_path in node.value
        for node in ast.walk(tree)
    )


@pytest.mark.asyncio
async def test_mesh_dimension_strings_cannot_inject_python():
    session = UnrealRuntimeSession()
    session._execute_python = AsyncMock()
    with pytest.raises(ValueError):
        await session.generate_mesh_primitive(
            "box", dimensions={"width": "1); review_injected();#"}
        )
    session._execute_python.assert_not_called()


class Vector:
    def __init__(self, x=0, y=0, z=0):
        self.x, self.y, self.z = x, y, z


class PrimitiveComponent:
    def __init__(self):
        self.mobility = SimpleNamespace(name="MOVABLE")
        self.physics = False
        self.events = []

    def get_name(self):
        return "Mesh"

    def get_class(self):
        return SimpleNamespace(get_name=lambda: "StaticMeshComponent")

    def set_simulate_physics(self, value):
        self.physics = value

    def is_simulating_physics(self):
        return self.physics

    def set_collision_profile_name(self, name):
        self.events.append(("profile", name))

    def set_collision_enabled(self, mode):
        self.events.append(("collision", mode))

    def add_impulse_at_location(self, force, location):
        self.events.append(("impulse", force.z, location.x))

    def set_mass_override_in_kg(self, bone_name, mass, override):
        self.events.append(("mass", mass))

    def set_linear_damping(self, value):
        self.events.append(("linear_damping", value))

    def set_angular_damping(self, value):
        self.events.append(("angular_damping", value))

    def set_enable_gravity(self, value):
        self.events.append(("gravity", value))


class MeshComponent(PrimitiveComponent):
    def get_num_materials(self):
        return 1

    def set_material(self, slot, material):
        self.events.append(("material", slot, material))


class Actor:
    def __init__(self, path="/Game/Test.Actor"):
        self.root_component = MeshComponent()
        self.path = path

    def get_actor_location(self):
        return Vector(100, 200, 300)

    def get_actor_rotation(self):
        return SimpleNamespace(pitch=15, yaw=30, roll=45)

    def get_actor_scale3d(self):
        return Vector(2, 3, 4)

    def get_actor_label(self):
        return "Review actor"

    def get_path_name(self):
        return self.path

    def get_class(self):
        return SimpleNamespace(
            get_name=lambda: "StaticMeshActor",
            get_path_name=lambda: "/Script/Engine.StaticMeshActor",
        )

    def get_components_by_class(self, cls):
        return [self.root_component] if isinstance(self.root_component, cls) else []

    def get_editor_property(self, name):
        return {"tags": ["review"], "hidden": True}[name]

    def actor_has_tag(self, tag):
        return tag in self.get_editor_property("tags")


@pytest.fixture
def script_session(monkeypatch, tmp_path):
    actor = Actor()
    world = SimpleNamespace(playing=False, paused=False)

    class LevelEditor:
        def is_in_play_in_editor(self):
            return world.playing

        def editor_play_simulate(self):
            world.playing = True

        def editor_request_end_play(self):
            world.playing = False

    class Material:
        pass

    material = Material()
    unreal = SimpleNamespace(
        Actor=Actor,
        ActorComponent=PrimitiveComponent,
        PrimitiveComponent=PrimitiveComponent,
        MeshComponent=MeshComponent,
        MaterialInterface=Material,
        Vector=Vector,
        EditorActorSubsystem="actors",
        LevelEditorSubsystem="level",
        UnrealEditorSubsystem="editor",
        CollisionEnabled=SimpleNamespace(QUERY_AND_PHYSICS=3, NO_COLLISION=0),
        load_object=lambda outer, path: actor if path == actor.path else None,
        load_asset=lambda path: material if path == "/Game/Material" else None,
        get_editor_subsystem=lambda name: {
            "actors": SimpleNamespace(get_all_level_actors=lambda: [actor]),
            "level": LevelEditor(),
            "editor": SimpleNamespace(
                get_game_world=lambda: world if world.playing else None
            ),
        }[name],
        GameplayStatics=SimpleNamespace(
            is_game_paused=lambda world: world.paused,
            get_time_seconds=lambda world: 12.5,
            set_game_paused=lambda world, value: setattr(world, "paused", value)
            or True,
        ),
    )
    monkeypatch.setitem(sys.modules, "unreal", unreal)
    base = Settings()
    settings = base.model_copy(
        update={
            "security": base.security.model_copy(
                update={
                    "allowed_paths": [str(tmp_path)],
                }
            )
        }
    )
    session = UnrealRuntimeSession(settings)

    async def execute(code, mode="ExecuteFile"):
        output = io.StringIO()
        try:
            with redirect_stdout(output):
                exec(code, {})
            return {
                "ReturnValue": True,
                "LogOutput": [
                    {"Type": "Info", "Output": line}
                    for line in output.getvalue().splitlines()
                ],
            }
        except Exception as exc:
            return {"ReturnValue": False, "CommandResult": str(exc)}

    monkeypatch.setattr(session, "_execute_python", execute)
    return session, actor, unreal


@pytest.mark.asyncio
async def test_actor_world_state_and_filters(script_session):
    session, actor, _ = script_session
    result = await session.get_actor_info(actor.path)
    assert result["location"] == [100, 200, 300]
    assert result["rotation"] == [15, 30, 45]
    assert result["scale"] == [2, 3, 4]
    assert result["tags"] == ["review"]
    assert result["mobility"] == "Movable"
    assert result["is_hidden"] is True
    assert result["components"][0]["is_root"] is True
    for cls in ("StaticMeshActor", "/Script/Engine.StaticMeshActor"):
        listing = await session.list_actors(
            class_filter=cls, tag_filter="review", max_results=1
        )
        assert listing["count"] == 1
        assert listing["truncated"] is False
    assert (await session.list_actors(tag_filter="missing"))["count"] == 0


@pytest.mark.asyncio
async def test_missing_actor_is_error_and_cannot_focus_origin(script_session):
    session, _, _ = script_session
    assert (await session.get_actor_info("/Missing.Actor"))["success"] is False
    with pytest.raises(RuntimeError, match="Actor not found"):
        await session._get_actor_transform("/Missing.Actor")


@pytest.mark.asyncio
async def test_component_operations_and_missing_component(script_session):
    session, actor, _ = script_session
    component = actor.root_component
    assert (await session.enable_physics(actor.path))["physics_enabled"] is True
    assert (await session.set_collision(actor.path, "BlockAll"))[
        "collision_enabled"
    ] is True
    result = await session.apply_force(
        actor.path, force_z=7, is_impulse=True, location_x=4, location_y=5, location_z=6
    )
    assert result["force_applied"] is True
    result = await session.set_physics_params(
        actor.path, mass=9, linear_damping=1, angular_damping=2, enable_gravity=False
    )
    assert result["params_set"] == 4
    assert not (await session.assign_material(actor.path, "/Game/Material")).get(
        "error"
    )
    assert ("impulse", 7, 4) in component.events
    assert ("mass", 9) in component.events
    assert ("gravity", False) in component.events
    assert any(event[0] == "material" for event in component.events)
    actor.root_component = None
    assert (await session.enable_physics(actor.path))["success"] is False


@pytest.mark.asyncio
async def test_simulation_lifecycle_and_query_errors(script_session, monkeypatch):
    session, _, unreal = script_session
    for action, state in [
        ("start", "playing"),
        ("pause", "paused"),
        ("resume", "playing"),
        ("stop", "stopped"),
    ]:
        assert (await session.control_simulation(action))["state"] == state
    assert (await session.control_simulation("pause"))["error_type"] == "InvalidState"
    assert (await session.control_simulation("step"))[
        "error_type"
    ] == "UnsupportedOperation"
    monkeypatch.setattr(unreal, "get_editor_subsystem", lambda name: None)
    result = await session.get_simulation_status()
    assert result["success"] is False
    assert "is_playing" not in result


@pytest.mark.asyncio
async def test_simulation_start_resumes_paused_session(script_session):
    session, _, _ = script_session
    session.timeout = 0
    for action, state in [
        ("start", "playing"),
        ("start", "playing"),
        ("pause", "paused"),
        ("pause", "paused"),
        ("start", "playing"),
        ("resume", "playing"),
        ("stop", "stopped"),
        ("stop", "stopped"),
    ]:
        result = await session.control_simulation(action)
        assert result.get("state") == state, result


@pytest.mark.asyncio
async def test_usd_plugin_missing_and_simready_explicitly_unsupported(
    script_session, tmp_path
):
    session, _, _ = script_session
    source, target = str(tmp_path / "in.usda"), str(tmp_path / "out.usda")
    for result in (
        await session.import_usd(source),
        await session.export_usd(["/Game/Test.Actor"], target),
    ):
        assert result["success"] is False
        assert result["error_type"] == "PluginUnavailable"
    for result in (
        await session.convert_to_simready(source, target),
        await session.validate_simready_asset(source),
    ):
        assert result["success"] is False
        assert result["error_type"] == "UnsupportedOperation"


@pytest.mark.asyncio
@pytest.mark.parametrize("fmt", ["png", "jpeg"])
async def test_capture_download_above_inline_cap_and_remote_path(
    monkeypatch, tmp_path, fmt
):
    # Real PNG above 256 KiB; the editor path does not exist on this host.
    import random

    raw = random.Random(0).randbytes(384 * 384 * 3)
    buf = io.BytesIO()
    Image.frombytes("RGB", (384, 384), raw).save(buf, format="PNG")
    blob = buf.getvalue()
    assert len(blob) > 262144
    base = Settings()
    session = UnrealRuntimeSession(
        base.model_copy(
            update={"unreal": base.unreal.model_copy(update={"host": "remote.invalid"})}
        )
    )
    session.capture_viewport = AsyncMock(
        return_value={
            "path": "/remote/Saved/Screenshots/image.png",
            "size_bytes": len(blob),
            "resolution_x": 384,
            "resolution_y": 384,
            "format": "png",
        }
    )
    offsets = []

    async def transfer(code):
        # Execute the actual bounded reader against a synthetic remote fs.
        class RemoteFile(io.BytesIO):
            def __enter__(self):
                return self

            def read(self, size=-1):
                offsets.append((self.tell(), size))
                return super().read(size)

        unreal = SimpleNamespace(
            Paths=SimpleNamespace(project_saved_dir=lambda: "/remote/Saved")
        )
        monkeypatch.setitem(sys.modules, "unreal", unreal)
        output = io.StringIO()
        with redirect_stdout(output):
            exec(code, {"open": lambda path, mode: RemoteFile(blob)})
        return json.loads(output.getvalue())

    monkeypatch.setattr(session, "_execute_json_script", transfer)
    # Shrink the chunk so this ~440 KiB capture still spans several reads.
    monkeypatch.setattr(unreal_runtime, "_CAPTURE_CHUNK_BYTES", 128 * 1024)
    target = tmp_path / f"capture.{fmt}"
    result = await session.capture_to_file(target, 384, 384, fmt)
    assert result["file_path"] == str(target.resolve())
    with Image.open(target) as saved:
        assert saved.format.lower() == fmt
        assert saved.size == (384, 384)
    assert len(offsets) > 1
    assert all(size <= 128 * 1024 for _, size in offsets)
    assert [offset for offset, _ in offsets] == list(range(0, len(blob), 128 * 1024))


@pytest.mark.asyncio
async def test_capture_failed_transfer_preserves_existing_output(monkeypatch, tmp_path):
    session = UnrealRuntimeSession()
    session.capture_viewport = AsyncMock(
        return_value={"path": "/remote/shot.png", "size_bytes": 42}
    )
    session._execute_json_script = AsyncMock(
        return_value={"data": base64.b64encode(b"short").decode()}
    )
    target = tmp_path / "keep.png"
    target.write_bytes(b"original")
    with pytest.raises(IOError, match="truncated"):
        await session.capture_to_file(target)
    assert target.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.asyncio
async def test_mesh_ops_resolve_actors_with_actor_at():
    """Mesh ops load their actors by path (ACTOR_HELPERS.actor_at) instead of
    scanning every level actor, and a missing actor is a ScriptError."""
    session = UnrealRuntimeSession()
    session._execute_python = AsyncMock(
        return_value={
            "ReturnValue": False,
            "CommandResult": "ValueError: Actor not found: /Game/M.M:PersistentLevel.Missing",
        }
    )
    result = await session.apply_mesh_boolean(
        "/Game/M.M:PersistentLevel.Body", "/Game/M.M:PersistentLevel.Missing", "union"
    )
    script = session._execute_python.call_args.args[0]
    assert "get_all_level_actors" not in script
    assert "target_actor = actor_at('/Game/M.M:PersistentLevel.Body')" in script
    assert "tool_actor = actor_at('/Game/M.M:PersistentLevel.Missing')" in script
    compile(script, "apply_mesh_boolean", "exec")
    assert result["success"] is False
    assert result["error_type"] == "ScriptError"
    assert "Actor not found" in result["error"]
