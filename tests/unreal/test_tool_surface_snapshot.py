"""The Unreal tool surface, pinned as an MCP client sees it.

``register_unreal_tools`` builds every Unreal MCP tool. This module pins what it
produces, so a change to how the tools are declared or built cannot change what
a client sees, or what reaches the editor, without a snapshot diff showing it:

* ``tools/list`` read through a real FastMCP client for the thin and the full
  surface, with script execution allowed and disallowed, in endpoint and
  attached mode: names, order, descriptions, input schemas and annotations
  (``snapshots/unreal_tool_listing.json``).
* The registration arguments a listing does not show: every tool is registered
  with ``output_schema=None`` and the server's optional task config.
* Every tool called through the client against a recording session, once with
  only its required arguments and once with every argument set: the session
  method and the arguments it binds, the response model and usage parameters
  handed to the envelope, the payload and the usage record
  (``snapshots/unreal_tool_dispatch.json``).
* The payloads and usage records of representative invalid calls (same file).

After an intended change, regenerate with
``SIMUL_UPDATE_SNAPSHOTS=1 pytest tests/unreal/test_tool_surface_snapshot.py``
and review the snapshot diff like code.
"""

from __future__ import annotations

import asyncio
import difflib
import inspect
import json
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pytest
from fastmcp import Client
from pydantic import BaseModel, ConfigDict

from simul.adapters import unreal_connection
from simul.adapters.unreal_runtime import UnrealRuntimeSession
from simul.config import Settings
from simul.mcp import backends as backends_module
from simul.mcp import server as server_module
from simul.mcp.usage_tracker import ToolUsageTracker
from simul.tool_surfaces import THIN_UNREAL_TOOLS
from tests.fakes import AvailableAdapter, FakeFastMCP

SNAPSHOTS = Path(__file__).parent / "snapshots"
LISTING = SNAPSHOTS / "unreal_tool_listing.json"
DISPATCH = SNAPSHOTS / "unreal_tool_dispatch.json"
UPDATE = os.environ.get("SIMUL_UPDATE_SNAPSHOTS") == "1"

# pydantic's error messages link to a page whose path carries its version.
PYDANTIC_URL = re.compile(r"errors\.pydantic\.dev/[0-9][^/\"\s]*/")

# (surface, script execution allowed, unreal.mode); the first is the canonical listing.
CONFIGS: Tuple[Tuple[str, bool, str], ...] = tuple(
    (surface, scripts, mode)
    for surface in ("full", "thin")
    for scripts in (True, False)
    for mode in ("endpoint", "attached")
)

ACTOR = "/Game/Maps/Test.Test:PersistentLevel.Chair_0"
PARENT = "/Game/Maps/Test.Test:PersistentLevel.Table_0"
MESH = "/Game/Meshes/SM_Chair"
OTHER_MESH = "/Game/Meshes/SM_Hole"
MATERIAL = "/Game/Materials/M_Wood"
BATCH = json.dumps(
    [{"objectPath": "/Script/Engine.Default__KismetSystemLibrary", "functionName": "GetEngineVersion"}]
)

# One value for every parameter of every tool, each away from its default where
# the tool allows it. The required-arguments call takes the required ones from here.
SAMPLES: Dict[str, Dict[str, Any]] = {
    "unreal_health_check": {},
    "ping_unreal": {},
    "list_unreal_instances": {"scan": False},
    "capture_unreal_viewport": {"resolution_x": 640, "resolution_y": 360, "format": "png", "inline": True},
    "execute_unreal_script": {"code": "import unreal\nprint('hi')", "mode": "EvaluateStatement"},
    "control_unreal_ui": {
        "agent_control": "set_property",
        "target": "Chair",
        "property_name": "rotation",
        "value": [10.0, 20.5, -30.0],
        "enabled": True,
        "agent_id": "builder",
        "position": [0.25, 0.75],
        "activity": "Placing chairs",
    },
    "get_unreal_engine_info": {},
    "get_unreal_loaded_map": {},
    "list_unreal_actors": {"class_filter": "StaticMeshActor", "tag_filter": "Prop", "max_results": 5},
    "get_unreal_actor_info": {"actor_path": ACTOR},
    "search_unreal_assets": {
        "query": "chair",
        "class_names": " StaticMesh, ,Material ",
        "package_paths": "/Game/Props,, /Game/Env ",
        "max_results": 7,
    },
    "describe_unreal_object": {"object_path": ACTOR},
    "get_unreal_actor_thumbnail": {"asset_path": MESH, "width": 64, "height": 32},
    "summarize_unreal_scene": {},
    "get_unreal_viewport_info": {},
    "set_unreal_camera_view": {
        "location_x": 100.0,
        "location_y": -200.0,
        "location_z": 300.5,
        "rotation_pitch": -15.0,
        "rotation_yaw": 90.0,
        "rotation_roll": 0.5,
    },
    "focus_unreal_on_actor": {"actor_path": ACTOR, "distance": 250.0},
    "spawn_unreal_actor": {
        "asset_path": MESH,
        "location_x": 1.5,
        "location_y": -2.0,
        "location_z": 3.0,
        "rotation_pitch": 10.0,
        "rotation_yaw": 20.0,
        "rotation_roll": 30.0,
        "label": "Chair",
    },
    "delete_unreal_actor": {"actor_path": ACTOR},
    "set_unreal_actor_transform": {
        "actor_path": ACTOR,
        "location": [1.0, 2.0, 3.0],
        "rotation": [0.0, 90.0, 0.0],
        "scale": [2.0, 2.0, 2.0],
    },
    "set_unreal_actor_property": {
        "actor_path": ACTOR,
        "property_name": "bHidden",
        "property_value": "true",
        "generate_transaction": False,
    },
    "call_unreal_actor_function": {"actor_path": ACTOR, "function_name": "SetSpeed", "parameters": '{"Speed": 2}'},
    "set_unreal_actor_parent": {"actor_path": ACTOR, "parent_path": PARENT},
    "add_unreal_component": {"actor_path": ACTOR, "component_class": "PointLightComponent", "component_name": "Lamp"},
    "set_unreal_actor_visibility": {"actor_path": ACTOR, "visible": False, "propagate": False},
    "get_unreal_material_info": {"material_path": MATERIAL},
    "set_unreal_material_params": {
        "material_path": MATERIAL,
        "scalar_params_json": '{"Roughness": 0.4}',
        "vector_params_json": '{"Tint": [1, 0.5, 0, 1]}',
        "texture_params_json": '{"Albedo": "/Game/Textures/T_Wood"}',
    },
    "create_unreal_material_instance": {
        "parent_path": MATERIAL,
        "instance_name": "MI_Chair",
        "save_path": "/Game/Materials",
    },
    "assign_unreal_material": {"actor_path": ACTOR, "material_path": MATERIAL, "slot_index": 2},
    "set_unreal_light_params": {
        "actor_path": ACTOR,
        "intensity": 5000.0,
        "color": [1.0, 0.5, 0.25],
        "temperature": 6500.0,
        "use_temperature": True,
        "attenuation_radius": 800.0,
        "cast_shadows": False,
    },
    "set_unreal_render_settings": {"setting_name": "r.ScreenPercentage", "setting_value": "50"},
    "control_unreal_simulation": {"action": "start"},
    "get_unreal_simulation_status": {},
    "enable_unreal_physics": {"actor_path": ACTOR, "enable": False, "simulate_physics": False},
    "set_unreal_collision": {"actor_path": ACTOR, "collision_preset": "OverlapAll", "collision_enabled": False},
    "apply_unreal_force": {
        "actor_path": ACTOR,
        "force_x": 1.0,
        "force_y": 2.0,
        "force_z": 3.0,
        "is_impulse": True,
        "location_x": 4.0,
        "location_y": 5.0,
        "location_z": 6.0,
    },
    "set_unreal_physics_params": {
        "actor_path": ACTOR,
        "mass": 12.5,
        "linear_damping": 0.1,
        "angular_damping": 0.2,
        "enable_gravity": False,
    },
    "import_unreal_usd": {
        "usd_path": "/tmp/in.usd",
        "destination_path": "/Game/USD",
        "import_options": {"import_lights": False},
    },
    "export_unreal_usd": {
        "actor_paths": f"{ACTOR}, {PARENT}",
        "output_path": "/tmp/out.usd",
        "export_options": {"export_sublayers": True},
    },
    "convert_to_simready": {
        "usd_path": "/tmp/in.usd",
        "output_path": "/tmp/out.usd",
        "add_physics": False,
        "add_collision": False,
        "add_semantic_labels": False,
        "target_up_axis": "Y",
        "target_units": "centimeters",
    },
    "validate_simready_asset": {"usd_path": "/tmp/in.usd", "checks": "physics, ,scale"},
    "get_unreal_interchange_info": {},
    "batch_unreal_operations": {"operations": BATCH},
    "query_unreal_scene_graph": {
        "root_path": ACTOR,
        "max_depth": 3,
        "include_components": True,
        "class_filter": "StaticMeshActor",
    },
    "analyze_unreal_scene_for_robotics": {"analysis_types": "traversability, ,graspability", "actor_filter": "Prop"},
    "generate_unreal_procedural_scene": {
        "scene_type": "warehouse",
        "parameters": '{"rows": 3}',
        "bounds_min": "-100,-100,0",
        "bounds_max": "100, 100, 50",
    },
    "get_unreal_actor_by_semantic_label": {"label": "chair", "match_mode": "contains", "max_results": 9},
    "generate_unreal_mesh_primitive": {
        "primitive_type": "box",
        "dimensions": '{"x": 100}',
        "segments": 8,
        "location": "1,2,3",
        "actor_label": "Box",
    },
    "apply_unreal_mesh_boolean": {"target_mesh_path": MESH, "tool_mesh_path": OTHER_MESH, "operation": "subtract"},
    "compute_unreal_convex_hull": {"mesh_path": MESH},
    "decompose_unreal_convex_hull": {
        "mesh_path": MESH,
        "max_hulls": 4,
        "max_vertices_per_hull": 16,
        "min_cluster_size": 64,
        "resolution": 5000,
    },
    "edit_unreal_mesh_topology": {
        "mesh_path": MESH,
        "operation": "extrude",
        "face_selection": "0,1",
        "edge_selection": "2",
        "distance": 5.0,
        "offset": 1.5,
        "scale": "1, 2,3",
        "count": 4,
    },
    "subdivide_unreal_mesh": {"mesh_path": MESH, "level": 1, "scheme": "loop"},
    "simplify_unreal_mesh": {
        "mesh_path": MESH,
        "target_triangle_count": 500,
        "target_percentage": 0.5,
        "max_error": 0.01,
    },
    "cut_unreal_mesh_plane": {
        "mesh_path": MESH,
        "plane_origin": "0, 0,10",
        "plane_normal": "0,0,1",
        "fill_holes": False,
        "keep_both_sides": True,
    },
    "validate_unreal_mesh": {"mesh_path": MESH, "checks": "manifold,, normals "},
    "convert_unreal_mesh_format": {
        "mesh_path": MESH,
        "target_format": "dynamic_mesh",
        "tessellation_options": '{"level": 2}',
    },
    "remesh_unreal_mesh": {
        "mesh_path": MESH,
        "mode": "adaptive",
        "target_edge_length": 2.5,
        "target_triangle_count": 1000,
        "smoothing_iterations": 5,
    },
    "compute_unreal_mesh_uv": {"mesh_path": MESH, "method": "planar", "uv_channel": 1, "island_padding": 4.0},
}

# Calls a client can send that the tools refuse, by their own checks or while
# preparing the session call.
ERROR_CASES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("capture_unreal_viewport", {"format": "jpg"}),
    ("execute_unreal_script", {"code": "print(1)", "mode": "Bogus"}),
    ("set_unreal_camera_view", {"location_x": 1.0, "location_y": 2.0}),
    ("set_unreal_camera_view", {"location_x": 1.0, "location_y": 2.0, "location_z": 3.0, "rotation_roll": 4.0}),
    ("control_unreal_ui", {"agent_control": "bogus"}),
    ("control_unreal_ui", {"agent_control": "set_property", "property_name": "color", "value": [1.0, 2.0, 3.0]}),
    ("set_unreal_material_params", {"material_path": MATERIAL, "scalar_params_json": "{bad"}),
    ("batch_unreal_operations", {"operations": "not json"}),
    ("generate_unreal_mesh_primitive", {"primitive_type": "box", "location": "1,x,3"}),
    ("cut_unreal_mesh_plane", {"mesh_path": MESH, "plane_origin": "0,0,0", "plane_normal": "up"}),
)


# ---------------------------------------------------------------------------
# Snapshot files
# ---------------------------------------------------------------------------


def _assert_snapshot(path: Path, document: Any) -> None:
    """Compare ``document`` with the snapshot at ``path`` byte for byte, or rewrite it."""
    text = PYDANTIC_URL.sub("errors.pydantic.dev/<version>/", json.dumps(document, indent=1) + "\n")
    if UPDATE:
        path.parent.mkdir(exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return
    expected = path.read_text(encoding="utf-8")
    if text != expected:
        diff = "".join(
            difflib.unified_diff(
                expected.splitlines(keepends=True), text.splitlines(keepends=True), str(path.name), "registered", n=2
            )
        )
        pytest.fail(
            f"{path.name} no longer matches the registered Unreal tools. If the change is intended, "
            f"regenerate with SIMUL_UPDATE_SNAPSHOTS=1 and review the diff:\n{diff[:30000]}"
        )


def _plain(value: Any) -> Any:
    """JSON form of a value, keeping a tuple distinct from a list."""
    if isinstance(value, tuple):
        return {"tuple": [_plain(item) for item in value]}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return {"repr": repr(value)}


# ---------------------------------------------------------------------------
# tools/list
# ---------------------------------------------------------------------------


def _settings(surface: str, scripts: bool, mode: str, attachment: Path, **security: Any) -> Settings:
    return Settings(
        unreal={"tool_surface": surface, "mode": mode, "attachment_path": str(attachment)},
        security={"allow_script_execution": scripts, **security},
    )


def _list_unreal_tools(instance: server_module.SimulMCPServer) -> List[Dict[str, Any]]:
    """The Unreal tools of ``tools/list`` as a client receives them, in order."""
    unreal = set(instance.tools_by_backend().get("Unreal", []))

    async def run() -> List[Any]:
        async with Client(instance.mcp) as client:
            return list(await client.list_tools())

    return [tool.model_dump(mode="json", by_alias=True) for tool in asyncio.run(run()) if tool.name in unreal]


def _config_key(surface: str, scripts: bool, mode: str) -> str:
    return f"{surface}/{'scripts' if scripts else 'no-scripts'}/{mode}"


def _listing_document(tmp_path: Path) -> Dict[str, Any]:
    """The canonical full listing, plus each configuration's names in order.

    Every configuration's entries are checked against the canonical ones byte
    for byte, so the names are all a configuration adds to the snapshot.
    """
    canonical: Dict[str, Dict[str, Any]] = {}
    tools: List[Dict[str, Any]] = []
    surfaces: Dict[str, List[str]] = {}
    for surface, scripts, mode in CONFIGS:
        settings = _settings(surface, scripts, mode, tmp_path / "attachment.json")
        listing = _list_unreal_tools(server_module.SimulMCPServer(settings, backends={"unreal"}))
        if not tools:
            tools = listing
            canonical = {entry["name"]: entry for entry in listing}
        for entry in listing:
            assert json.dumps(entry) == json.dumps(canonical[entry["name"]]), (
                f"{entry['name']} is listed differently in {_config_key(surface, scripts, mode)}"
            )
        surfaces[_config_key(surface, scripts, mode)] = [entry["name"] for entry in listing]
    return {"tools": tools, "surfaces": surfaces}


def test_tool_listing_matches_the_snapshot(tmp_path: Path) -> None:
    _assert_snapshot(LISTING, _listing_document(tmp_path))


def test_listing_snapshot_covers_the_expected_surfaces() -> None:
    """A guard on the snapshot itself: the thin set and the script gate are what the listing pins."""
    surfaces = json.loads(LISTING.read_text(encoding="utf-8"))["surfaces"]
    full = surfaces["full/scripts/endpoint"]
    assert surfaces["thin/scripts/endpoint"] == [name for name in full if name in THIN_UNREAL_TOOLS]
    assert set(surfaces["thin/scripts/endpoint"]) == set(THIN_UNREAL_TOOLS)
    gated = set(full) - set(surfaces["full/no-scripts/endpoint"])
    assert gated == {"execute_unreal_script", "call_unreal_actor_function", "batch_unreal_operations"}
    for surface in ("full", "thin"):
        for scripts in ("scripts", "no-scripts"):
            assert surfaces[f"{surface}/{scripts}/attached"] == surfaces[f"{surface}/{scripts}/endpoint"]


# ---------------------------------------------------------------------------
# Registration arguments a listing does not show
# ---------------------------------------------------------------------------


def test_every_tool_registers_without_output_schema_and_with_the_task_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = object()
    monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
    monkeypatch.setattr(server_module.SimulMCPServer, "_task_optional", lambda self: task)
    monkeypatch.setattr(backends_module, "UnrealRuntimeAdapter", AvailableAdapter)
    monkeypatch.setattr(backends_module, "is_unreal_available", lambda: True)
    settings = Settings(unreal={"tool_surface": "full"})
    instance = server_module.SimulMCPServer(settings, backends={"unreal"})

    unreal = [tool for tool in instance.mcp.tools if tool.name != "get_tool_usage_stats"]
    assert len(unreal) == len(SAMPLES)
    for tool in unreal:
        assert sorted(tool.kwargs) == ["annotations", "description", "output_schema", "task"], tool.name
        assert tool.kwargs["output_schema"] is None, tool.name
        assert tool.kwargs["task"] is task, tool.name


# ---------------------------------------------------------------------------
# Dispatch: what each call hands the envelope and the session
# ---------------------------------------------------------------------------


class _AnyPayload(BaseModel):
    """Stands in for each response model, which the envelope recording names instead."""

    model_config = ConfigDict(extra="allow")


class _RecordingSession:
    """Accepts any ``UnrealRuntimeSession`` method and records the arguments it binds."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        signature = inspect.signature(getattr(UnrealRuntimeSession, name))

        async def record(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            arguments = {key: _plain(value) for key, value in bound.arguments.items() if key != "self"}
            self.calls.append({"method": name, "arguments": arguments})
            return {"success": True, "method": name}

        return record


class _RecordingAdapter(AvailableAdapter):
    def __init__(self, session: _RecordingSession) -> None:
        super().__init__(Settings())
        self.session = session

    @contextmanager
    def create_session(self) -> Iterator[_RecordingSession]:
        yield self.session


def _result(result: Any) -> Dict[str, Any]:
    """The client's view of one call: error flag, image blocks and the JSON block."""
    texts = [block.text for block in result.content if getattr(block, "type", None) == "text"]
    images = [block.mimeType for block in result.content if getattr(block, "type", None) == "image"]
    try:
        payload: Any = json.loads(texts[-1])
    except (IndexError, ValueError):
        payload = texts
    outcome: Dict[str, Any] = {"is_error": bool(result.is_error), "payload": payload}
    if images:
        outcome["images"] = images
    return outcome


def _usage(instance: server_module.SimulMCPServer, tool: str) -> List[Dict[str, Any]]:
    """The usage records of the last call, without timing and the caller's id."""
    rows = list(reversed(instance.usage_tracker.get_recent(tool_name=tool)))
    for row in rows:
        assert row.pop("agent_id"), "every record names its agent"
        for key in ("tool", "timestamp", "duration_ms"):
            row.pop(key)
    return rows


def _dispatch_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Tuple[Any, Any, Any]:
    """A full-surface server whose Unreal session, discovery and envelope record what they receive."""
    settings = _settings("full", True, "endpoint", tmp_path / "attachment.json", rate_limiting_enabled=False)
    instance = server_module.SimulMCPServer(settings, backends={"unreal"})
    instance.usage_tracker = ToolUsageTracker(log_dir=tmp_path / "usage")
    session = _RecordingSession()
    monkeypatch.setattr(instance, "unreal_adapter", _RecordingAdapter(session))

    async def discover_instances(settings: Settings, **kwargs: Any) -> Dict[str, Any]:
        arguments = {"settings": "server" if settings is instance.settings else repr(settings), **_plain(kwargs)}
        session.calls.append({"method": "unreal_connection.discover_instances", "arguments": arguments})
        return {"success": True, "instances": []}

    monkeypatch.setattr(unreal_connection, "discover_instances", discover_instances)

    envelope: List[Dict[str, Any]] = []
    original = instance._exec_backend

    async def exec_backend(
        tool_name: str,
        adapter: Any,
        adapter_label: str,
        response_model: Any,
        call: Any,
        params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        envelope.append(
            {
                "tool": tool_name,
                "label": adapter_label,
                "response_model": response_model.__name__,
                "params": _plain(params),
            }
        )
        return await original(tool_name, adapter, adapter_label, _AnyPayload, call, params, **kwargs)

    monkeypatch.setattr(instance, "_exec_backend", exec_backend)
    return instance, session, envelope


def _run_calls(
    instance: Any, session: _RecordingSession, envelope: List[Dict[str, Any]], cases: List[Tuple[str, str, Any]]
) -> List[Dict[str, Any]]:
    """Call each ``(tool, case, arguments)`` through a real client and record what happened."""

    async def run() -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        async with Client(instance.mcp) as client:
            for tool, case, arguments in cases:
                session.calls.clear()
                envelope.clear()
                instance.usage_tracker._recent.clear()
                result = await client.call_tool(tool, arguments, raise_on_error=False)
                entries.append(
                    {
                        "tool": tool,
                        "case": case,
                        "arguments": arguments,
                        "envelope": list(envelope),
                        "session": list(session.calls),
                        "result": _result(result),
                        "usage": _usage(instance, tool),
                    }
                )
        return entries

    return asyncio.run(run())


def _dispatch_document(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Dict[str, Any]:
    instance, session, envelope = _dispatch_server(tmp_path, monkeypatch)
    schemas = {entry["name"]: entry["inputSchema"] for entry in _list_unreal_tools(instance)}
    assert list(schemas) == list(SAMPLES), "SAMPLES lists every full-surface tool, in listing order"
    cases: List[Tuple[str, str, Any]] = []
    for tool, schema in schemas.items():
        sample = SAMPLES[tool]
        assert set(sample) == set(schema.get("properties", {})), f"{tool}: SAMPLES must set every parameter"
        required = {name: sample[name] for name in schema.get("required", [])}
        cases.append((tool, "required", required))
        if sample != required:
            cases.append((tool, "every argument", sample))
    errors = [(tool, "invalid", arguments) for tool, arguments in ERROR_CASES]
    return {
        "calls": _run_calls(instance, session, envelope, cases),
        "errors": _run_calls(instance, session, envelope, errors),
    }


def test_tool_dispatch_matches_the_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _assert_snapshot(DISPATCH, _dispatch_document(tmp_path, monkeypatch))


def test_dispatch_snapshot_reaches_a_session_method_for_every_tool() -> None:
    """A guard on the snapshot itself: no call was swallowed before reaching the editor."""
    calls = json.loads(DISPATCH.read_text(encoding="utf-8"))["calls"]
    assert {entry["tool"] for entry in calls} == set(SAMPLES)
    for entry in calls:
        assert len(entry["session"]) == 1, entry["tool"]
        assert entry["result"]["payload"].get("success") is True, entry["tool"]
        assert [record["success"] for record in entry["usage"]] == [True], entry["tool"]
