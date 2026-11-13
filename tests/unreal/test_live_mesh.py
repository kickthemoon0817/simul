"""Live checks of the GeometryScript mesh tools against a disposable editor.

Set SIMUL_UNREAL_REVIEW_LIVE=1 and UNREAL__PORT for a scratch project set up
by ``simul unreal setup`` (which enables GeometryScripting). The tests spawn
DynamicMeshActors in the open level, undo with the editor's own Undo, and
destroy what they made.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List

import pytest

from simul.adapters._unreal_mesh_scripts import XATLAS_MAX_TRIANGLES
from simul.adapters.unreal_runtime import UnrealRuntimeSession
from simul.config import Settings
from simul.mcp.registration._unreal_tools import UNREAL_TOOLS

pytestmark = pytest.mark.unreal_live

MODELS = {spec.method: spec.response for spec in UNREAL_TOOLS if spec.method}


@pytest.fixture(autouse=True)
def require_disposable_editor() -> None:
    if os.environ.get("SIMUL_UNREAL_REVIEW_LIVE") != "1":
        pytest.skip("Set SIMUL_UNREAL_REVIEW_LIVE=1 only for a disposable editor")


def _session() -> UnrealRuntimeSession:
    return UnrealRuntimeSession(Settings())


# Reads what the tools reported independently of them. Everything stays inside
# the function: the editor's __main__ outlives the call.
_STATE = """
import unreal, json
def _state(path):
    actor = unreal.find_object(None, path)
    level = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors()
    if actor is None or not unreal.SystemLibrary.is_valid(actor) or actor not in level:
        return {"exists": False}
    mesh = actor.dynamic_mesh_component.get_dynamic_mesh()
    return {"exists": True, "triangles": mesh.get_triangle_count(), "mesh": mesh.get_name()}
print(json.dumps(_state(%r)))
del _state
"""

_UNDO = """
import unreal, json
unreal.SystemLibrary.execute_console_command(None, "TRANSACTION UNDO")
print(json.dumps({"undone": True}))
"""

_DESTROY = """
import unreal, json
def _destroy(paths):
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    for path in paths:
        actor = unreal.find_object(None, path)
        if actor is not None and unreal.SystemLibrary.is_valid(actor):
            actors.destroy_actor(actor)
print(json.dumps({"destroyed": True}))
_destroy(%r)
del _destroy
"""

_STATIC_CUBE = """
import unreal, json
def _spawn():
    actor = unreal.get_editor_subsystem(unreal.EditorActorSubsystem).spawn_actor_from_class(
        unreal.StaticMeshActor, unreal.Vector(0, -3000, 0))
    actor.static_mesh_component.set_static_mesh(unreal.load_asset("/Engine/BasicShapes/Cube"))
    return actor.get_path_name()
print(json.dumps({"path": _spawn()}))
del _spawn
"""


# Cuts a mesh in place the way a Blueprint or a Python GeometryScript call does,
# outside simul, and reports the mesh's and the actor's x extent.
_DIRECT_CUT = """
import unreal, json
def _cut(path):
    actor = unreal.find_object(None, path)
    mesh = actor.dynamic_mesh_component.get_dynamic_mesh()
    frame = unreal.Transform()
    frame.rotation = unreal.MathLibrary.make_rot_from_z(unreal.Vector(1, 0, 0)).quaternion()
    unreal.GeometryScript_MeshBooleans.apply_mesh_plane_cut(mesh, frame, unreal.GeometryScriptMeshPlaneCutOptions())
    origin, extent = actor.get_actor_bounds(False)
    box = unreal.GeometryScript_MeshQueries.get_mesh_bounding_box(mesh)
    return {"mesh_x_max": box.max.x, "actor_x_max": origin.x + extent.x - actor.get_actor_location().x}
print(json.dumps(_cut(%r)))
del _cut
"""


async def _state(client: UnrealRuntimeSession, path: str) -> Dict[str, Any]:
    return await client._execute_json_script(_STATE % path)


async def _generate(client: UnrealRuntimeSession, made: List[str], kind: str, **kwargs: Any) -> str:
    result = await client.generate_mesh_primitive(kind, location=[0.0, -3000.0, 0.0], **kwargs)
    assert result.get("success") is not False, result
    made.append(result["actor_path"])
    return result["actor_path"]


def _exact(method: str, payload: Dict[str, Any]) -> None:
    assert payload.get("success") is not False, (method, payload)
    model = MODELS[method]
    assert set(payload) - {"success", "error"} == set(model.model_fields) - {"success", "error"}, method
    model(**payload)


def test_every_mesh_payload_is_its_response_model() -> None:
    async def run() -> None:
        client = _session()
        made: List[str] = []
        try:
            box = await _generate(client, made, "box")
            ball = await _generate(client, made, "sphere", dimensions={"radius": 50.0})
            capsule = await _generate(client, made, "capsule", dimensions={"radius": 30.0})
            calls = [
                ("validate_mesh", {"mesh_path": box}),
                ("compute_convex_hull", {"mesh_path": ball}),
                ("simplify_mesh", {"mesh_path": ball, "target_percentage": 0.5}),
                ("remesh_mesh", {"mesh_path": box, "target_edge_length": 20.0}),
                ("compute_mesh_uv", {"mesh_path": box}),
                ("subdivide_mesh", {"mesh_path": box, "level": 1, "scheme": "uniform"}),
                ("edit_mesh_topology", {"mesh_path": box, "operation": "inset_faces", "offset": 2.0}),
                ("cut_mesh_plane", {"mesh_path": capsule, "plane_origin": [0, 0, 0], "plane_normal": [0, 0, 1]}),
                ("apply_mesh_boolean", {"target_mesh_path": box, "tool_mesh_path": ball, "operation": "union"}),
                ("convert_mesh_format", {"mesh_path": "/Engine/BasicShapes/Cube", "target_format": "dynamic_mesh"}),
            ]
            for method, kwargs in calls:
                payload = await getattr(client, method)(**kwargs)
                _exact(method, payload)
                for key in ("hull_actor_path", "result_path"):
                    if key in payload:
                        made.append(payload[key])
        finally:
            await client._execute_json_script(_DESTROY % made)
            await client.close()

    asyncio.run(run())


def test_one_undo_reverts_one_mesh_edit_and_keeps_the_actor() -> None:
    async def run() -> None:
        client = _session()
        made: List[str] = []
        try:
            ball = await _generate(client, made, "sphere", dimensions={"radius": 50.0})
            before = await _state(client, ball)
            simplified = await client.simplify_mesh(ball, target_percentage=0.25)
            assert simplified["result_triangles"] == (await _state(client, ball))["triangles"] < before["triangles"]
            failed = await client.subdivide_mesh(ball, level=2)
            assert (failed["success"], failed["error_type"]) == (False, "EngineError"), failed
            assert failed["details"]["code"] == "GEOMETRY_SCRIPT_FAILED"
            assert (await _state(client, ball))["triangles"] == simplified["result_triangles"]
            await client._execute_json_script(_UNDO)
            assert await _state(client, ball) == before
            await client._execute_json_script(_UNDO)
            assert (await _state(client, ball))["exists"] is False
        finally:
            await client._execute_json_script(_DESTROY % made)
            await client.close()

    asyncio.run(run())


def test_xatlas_is_refused_above_its_limit() -> None:
    async def run() -> None:
        client = _session()
        made: List[str] = []
        try:
            # 2 * 104 * 102 = 21,216 triangles, just above the limit.
            ball = await _generate(client, made, "sphere", segments=104)
            assert (await _state(client, ball))["triangles"] > XATLAS_MAX_TRIANGLES
            refused = await client.compute_mesh_uv(ball, method="xatlas")
            assert refused["details"]["code"] == "MESH_TOO_LARGE_FOR_XATLAS", refused
            auto = await client.compute_mesh_uv(ball)
            assert (auto["method"], auto["requested_method"]) == ("patch_builder", "auto"), auto
        finally:
            await client._execute_json_script(_DESTROY % made)
            await client.close()

    asyncio.run(run())


def test_static_mesh_inputs_are_refused_with_the_way_in() -> None:
    async def run() -> None:
        client = _session()
        cube = (await client._execute_json_script(_STATIC_CUBE))["path"]
        try:
            actor = await client.simplify_mesh(cube)
            assert (actor["error_type"], actor["details"]["code"]) == ("ValidationError", "STATIC_MESH_ACTOR")
            asset = await client.validate_mesh("/Engine/BasicShapes/Cube")
            assert (asset["error_type"], asset["details"]["code"]) == ("ValidationError", "STATIC_MESH_ASSET")
            for refusal in (actor, asset):
                assert "convert_unreal_mesh_format" in json.dumps(refusal)
        finally:
            await client._execute_json_script(_DESTROY % [cube])
            await client.close()

    asyncio.run(run())


def test_a_mesh_restored_by_undo_follows_edits_outside_simul_after_a_touch() -> None:
    """Undo left the component deaf to its restored mesh: a direct cut kept the old bounds (#337 review)."""

    async def run() -> None:
        client = _session()
        made: List[str] = []
        try:
            box = await _generate(client, made, "box")
            assert (await client.remesh_mesh(box, target_edge_length=20.0)).get("success") is not False
            await client._execute_json_script(_UNDO)
            # Any mesh tool touching the actor binds the restored mesh again.
            assert (await client.validate_mesh(box)).get("success") is not False
            cut = await client._execute_json_script(_DIRECT_CUT % box)
            assert cut["mesh_x_max"] == pytest.approx(0.0, abs=0.01)
            assert cut["actor_x_max"] == pytest.approx(0.0, abs=0.01)
        finally:
            await client._execute_json_script(_DESTROY % made)
            await client.close()

    asyncio.run(run())


def test_a_result_over_the_cap_is_refused_live() -> None:
    async def run() -> None:
        client = _session()
        made: List[str] = []
        try:
            ball = await _generate(client, made, "sphere", segments=256)
            subdivided = await client.subdivide_mesh(ball, level=3, scheme="uniform")
            assert (subdivided["error_type"], subdivided["details"]["code"]) == ("ValidationError", "RESULT_TOO_LARGE")
            assert (await _state(client, ball))["triangles"] == 2 * 256 * 254
        finally:
            await client._execute_json_script(_DESTROY % made)
            await client.close()

    asyncio.run(run())
