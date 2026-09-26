"""Execute the generated ``validate_mesh`` script against a fake ``unreal``.

UE's Python binding returns ``(return_value, *out_params)`` for UFUNCTIONs
with out-params, so ``GetNumOpenBorderLoops(TargetMesh, bool&
bAmbiguousTopologyFound)`` comes back as ``(count, ambiguous)`` on UE 5.6
and 5.7. The script used to compare that tuple with ``> 0`` and crash.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from simul_mcp.adapters.unreal_runtime import UnrealRuntimeSession

MESH_PATH = "/Game/Map.Map:PersistentLevel.DynamicMeshActor_0"


def _fake_unreal(counts: Dict[str, Any]) -> SimpleNamespace:
    mesh = object()
    actor = SimpleNamespace(
        get_path_name=lambda: MESH_PATH,
        dynamic_mesh_component=SimpleNamespace(get_dynamic_mesh=lambda: mesh),
    )
    queries = SimpleNamespace(
        **{name: (lambda m, v=value: v) for name, value in counts.items()}
    )
    return SimpleNamespace(
        GeometryScript_MeshQueries=queries,
        EditorActorSubsystem="actors",
        get_editor_subsystem=lambda _: SimpleNamespace(
            get_all_level_actors=lambda: [actor]
        ),
    )


def _session(monkeypatch: pytest.MonkeyPatch, counts: Dict[str, Any]):
    monkeypatch.setitem(sys.modules, "unreal", _fake_unreal(counts))
    session = UnrealRuntimeSession()

    async def execute(code: str, mode: str = "ExecuteFile") -> Dict[str, Any]:
        out = io.StringIO()
        try:
            with redirect_stdout(out):
                exec(code, {})
        except Exception as exc:  # surface script errors like UE does
            return {"ReturnValue": False, "CommandResult": repr(exc)}
        return {
            "ReturnValue": True,
            "LogOutput": [
                {"Type": "Info", "Output": ln} for ln in out.getvalue().splitlines()
            ],
        }

    monkeypatch.setattr(session, "_execute_python", execute)
    return session


# Shapes observed live on UE 5.7.4: only get_num_open_border_loops is a tuple.
UE57_OPEN_PLANE = {
    "get_num_triangle_i_ds": 2,
    "get_vertex_count": 4,
    "get_num_open_border_edges": 4,
    "get_num_open_border_loops": (1, False),
    "get_num_connected_components": 1,
    "get_has_triangle_normals": True,
    "get_has_triangle_id_gaps": False,
}


@pytest.mark.asyncio
async def test_validate_mesh_unwraps_out_param_tuples(monkeypatch):
    session = _session(monkeypatch, UE57_OPEN_PLANE)
    result = await session.validate_mesh(MESH_PATH)
    assert "error" not in result, result
    assert result["open_border_edges"] == 4
    assert result["open_border_loops"] == 1
    assert result["is_valid"] is False
    assert result["issues"] == [
        "Non-watertight: 4 open border edges",
        "1 open border loops",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("as_tuple", [False, True])
async def test_validate_mesh_accepts_int_or_tuple_counts(monkeypatch, as_tuple):
    counts = {
        "get_num_triangle_i_ds": 12,
        "get_vertex_count": 8,
        "get_num_open_border_edges": 0,
        "get_num_open_border_loops": 0,
        "get_num_connected_components": 1,
        "get_has_triangle_normals": True,
        "get_has_triangle_id_gaps": False,
    }
    if as_tuple:
        counts = {
            k: ((v, False) if k.startswith("get_num") or k == "get_vertex_count" else v)
            for k, v in counts.items()
        }
    session = _session(monkeypatch, counts)
    result = await session.validate_mesh(MESH_PATH)
    assert result == {
        "mesh_path": MESH_PATH,
        "is_valid": True,
        "triangle_count": 12,
        "vertex_count": 8,
        "open_border_edges": 0,
        "open_border_loops": 0,
        "connected_components": 1,
        "has_normals": True,
        "issues": [],
    }
