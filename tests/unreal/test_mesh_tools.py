"""The Unreal mesh tools, run end to end against a fake editor.

Each test runs the real editor script a mesh tool sends (``_unreal_mesh_scripts``)
in ``tests.unreal.fake_geometry``'s fake ``unreal`` module, through the real
``UnrealRuntimeSession`` method, and checks the reply against what the script
did to the fake level, and against the tool's response model.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Tuple

import pytest

from simul.adapters import _unreal_mesh_scripts as mesh_scripts
from simul.adapters.unreal_runtime import UnrealRuntimeSession
from simul.mcp.registration._unreal_tools import UNREAL_TOOLS
from tests.unreal.fake_geometry import LEVEL, DynamicMesh, FakeEditor, level_editor

BODY, BALL, BIG, CRATE, LIGHT = (LEVEL + name for name in ("Body", "Ball", "Big", "Crate", "Light"))

# Every mesh tool, its session call and the actor whose mesh it changes (None
# for the tools that create an actor or only read).
TOOLS: Dict[str, Tuple[str, Dict[str, Any], Any]] = {
    "generate_unreal_mesh_primitive": ("generate_mesh_primitive", {"primitive_type": "box"}, None),
    "apply_unreal_mesh_boolean": (
        "apply_mesh_boolean",
        {"target_mesh_path": BODY, "tool_mesh_path": BALL, "operation": "union"},
        BODY,
    ),
    "compute_unreal_convex_hull": ("compute_convex_hull", {"mesh_path": BALL}, None),
    "decompose_unreal_convex_hull": ("decompose_convex_hull", {"mesh_path": BALL, "max_hulls": 3}, None),
    "edit_unreal_mesh_topology": ("edit_mesh_topology", {"mesh_path": BODY, "operation": "extrude_faces"}, BODY),
    "subdivide_unreal_mesh": ("subdivide_mesh", {"mesh_path": BODY, "level": 1}, BODY),
    "simplify_unreal_mesh": ("simplify_mesh", {"mesh_path": BALL, "target_percentage": 0.5}, BALL),
    "cut_unreal_mesh_plane": (
        "cut_mesh_plane",
        {"mesh_path": BODY, "plane_origin": [0, 0, 0], "plane_normal": [0, 0, 1]},
        BODY,
    ),
    "validate_unreal_mesh": ("validate_mesh", {"mesh_path": BODY}, None),
    "convert_unreal_mesh_format": (
        "convert_mesh_format",
        {"mesh_path": "/Game/Meshes/SM_Crate", "target_format": "dynamic_mesh"},
        None,
    ),
    "remesh_unreal_mesh": ("remesh_mesh", {"mesh_path": BODY, "target_edge_length": 20.0}, BODY),
    "compute_unreal_mesh_uv": ("compute_mesh_uv", {"mesh_path": BALL}, BALL),
}

# The tools that change or create something, and so refuse during PIE.
MUTATING = sorted(name for name in TOOLS if name != "validate_unreal_mesh")

SPECS = {spec.name: spec for spec in UNREAL_TOOLS}


def _session(editor: FakeEditor) -> UnrealRuntimeSession:
    session = UnrealRuntimeSession()

    async def execute(code: str, mode: str = "ExecuteFile") -> Dict[str, Any]:
        return editor.run(code)

    session._execute_python = execute  # type: ignore[method-assign]
    return session


def _call(editor: FakeEditor, tool: str, **overrides: Any) -> Dict[str, Any]:
    method, kwargs, _ = TOOLS[tool]
    return asyncio.run(getattr(_session(editor), method)(**{**kwargs, **overrides}))


@pytest.fixture
def editor() -> FakeEditor:
    return level_editor()


def test_every_mesh_tool_is_covered() -> None:
    assert sorted(TOOLS) == sorted(spec.name for spec in UNREAL_TOOLS if spec.method in {
        call for call, _, _ in TOOLS.values()
    })
    assert all(SPECS[name].method == method for name, (method, _, _) in TOOLS.items())


@pytest.mark.parametrize("tool", sorted(TOOLS))
def test_payload_is_exactly_the_response_model(editor: FakeEditor, tool: str) -> None:
    """Every field a model declares is emitted, and nothing it would drop.

    The models used to require fields no script emitted, so the call failed
    with a ValidationError after the editor had changed (B4).
    """
    payload = _call(editor, tool)
    assert payload.get("success") is not False, payload
    model = SPECS[tool].response
    declared = set(model.model_fields) - {"success", "error"}
    assert set(payload) - {"success", "error"} == declared
    model(**payload)


def test_generate_spawns_in_one_transaction(editor: FakeEditor) -> None:
    payload = _call(editor, "generate_unreal_mesh_primitive", primitive_type="box",
                    dimensions={"width": 200.0, "height": 100.0, "depth": 50.0}, location=[0, 100, 0])
    actor = editor.objects[payload["actor_path"]]
    assert editor.transactions == ["simul: generate mesh box"]
    assert actor.location.y == 100
    assert actor.dynamic_mesh_component.get_dynamic_mesh().stats["volume"] == 200 * 100 * 50
    assert (payload["triangle_count"], payload["vertex_count"]) == (12, 8)


@pytest.mark.parametrize("tool", [name for name, (_, _, edited) in TOOLS.items() if edited])
def test_an_edit_replaces_the_mesh_in_one_transaction(editor: FakeEditor, tool: str) -> None:
    """The result is a new mesh swapped in, so Undo restores the old one untouched (M4)."""
    actor = editor.objects[TOOLS[tool][2]]
    before = actor.dynamic_mesh_component.get_dynamic_mesh()
    stats = dict(before.stats)
    payload = _call(editor, tool)
    after = actor.dynamic_mesh_component.get_dynamic_mesh()
    assert payload.get("success") is not False, payload
    assert after is not before and before.stats == stats
    assert len(editor.transactions) == 1 and editor.transactions[0].startswith("simul: ")
    assert actor.path in editor.modified
    assert ("set_dynamic_mesh", actor.path, after.name) in editor.events


@pytest.mark.parametrize("tool", sorted(TOOLS))
def test_every_geometry_script_call_is_checked(editor: FakeEditor, tool: str) -> None:
    """Each gets a GeometryScriptDebug, read after it returns (B11); validation only queries."""
    _call(editor, tool)
    assert editor.calls or tool == "validate_unreal_mesh", tool
    assert all(debugged for _, debugged in editor.calls), editor.calls


def test_a_geometry_script_error_fails_the_tool_and_changes_nothing(editor: FakeEditor) -> None:
    """Catmull-Clark on a mesh without quad polygroups used to report success (B11)."""
    ball = editor.objects[BALL]
    before = ball.dynamic_mesh_component.get_dynamic_mesh()
    payload = _call(editor, "subdivide_unreal_mesh", mesh_path=BALL, level=2)
    assert payload["success"] is False
    assert payload["error_type"] == "EngineError"
    assert payload["details"]["code"] == "GEOMETRY_SCRIPT_FAILED"
    assert payload["details"]["messages"] == [
        "ApplyPolygroupCatmullClarkSubD: Target Polygroup Layer does not define valid Polygons for Subdivision"
    ]
    assert "Nothing was changed" in payload["error"]
    assert ball.dynamic_mesh_component.get_dynamic_mesh() is before and before.stats["tris"] == 1920
    assert editor.transactions == []


@pytest.mark.parametrize("tool", ["compute_unreal_convex_hull", "decompose_unreal_convex_hull"])
def test_a_failed_spawning_tool_spawns_nothing(editor: FakeEditor, tool: str) -> None:
    function = {"compute_unreal_convex_hull": "compute_mesh_convex_hull",
                "decompose_unreal_convex_hull": "compute_mesh_convex_decomposition"}[tool]
    editor.fail[function] = "Hull Computation Failed"
    payload = _call(editor, tool)
    assert (payload["success"], payload["error_type"]) == (False, "EngineError")
    assert editor.spawned == 0 and editor.transactions == []


def test_warnings_are_returned(editor: FakeEditor) -> None:
    editor.warn["apply_simplify_to_triangle_count"] = "Simplifier: some edges were locked"
    payload = _call(editor, "simplify_unreal_mesh")
    assert payload["warnings"] == ["Simplifier: some edges were locked"]


@pytest.mark.parametrize("tool", sorted(TOOLS))
def test_without_geometry_scripting_every_tool_says_how_to_enable_it(editor: FakeEditor, tool: str) -> None:
    """In a project set up before simul enabled the plugin they failed with an AttributeError (M1)."""
    editor.geometry_script = False
    payload = _call(editor, tool)
    assert payload["success"] is False
    assert payload["error_type"] == "UnsupportedOperation"
    assert payload["details"]["code"] == "GEOMETRY_SCRIPTING_DISABLED"
    assert "simul unreal setup" in payload["details"]["hint"]
    assert editor.transactions == [] and editor.spawned == 0


@pytest.mark.parametrize("tool", sorted(name for name, (_, _, edited) in TOOLS.items() if edited)
                         + ["compute_unreal_convex_hull", "validate_unreal_mesh"])
def test_a_static_mesh_actor_is_refused_with_the_way_in(editor: FakeEditor, tool: str) -> None:
    """It used to fail with "AttributeError: ... dynamic_mesh_component" (M12)."""
    key = "target_mesh_path" if tool == "apply_unreal_mesh_boolean" else "mesh_path"
    payload = _call(editor, tool, **{key: CRATE})
    assert payload["success"] is False
    assert payload["error_type"] == "ValidationError"
    assert payload["details"]["code"] == "STATIC_MESH_ACTOR"
    assert payload["details"]["static_mesh"] == "/Game/Meshes/SM_Crate.SM_Crate"
    assert "convert_unreal_mesh_format" in payload["details"]["hint"]


@pytest.mark.parametrize(
    ("path", "error_type", "code"),
    [
        ("/Game/Meshes/SM_Crate", "ValidationError", "STATIC_MESH_ASSET"),
        (LIGHT, "ValidationError", "NO_DYNAMIC_MESH"),
        (LEVEL + "Missing", "NotFoundError", None),
    ],
)
def test_other_inputs_get_typed_errors(editor: FakeEditor, path: str, error_type: str, code: Any) -> None:
    """A StaticMesh asset path used to read "Actor not found" (M12)."""
    payload = _call(editor, "simplify_unreal_mesh", mesh_path=path)
    assert (payload["success"], payload["error_type"]) == (False, error_type)
    assert (payload.get("details") or {}).get("code") == code
    if error_type == "NotFoundError":
        assert payload["error"].startswith("Actor not found: " + path)


def test_an_actor_of_another_world_is_not_found(editor: FakeEditor) -> None:
    from tests.unreal.editor_world import World

    editor.objects[BODY].world = World("/Game/Maps/Other.Other")
    payload = _call(editor, "simplify_unreal_mesh", mesh_path=BODY)
    assert payload["error_type"] == "NotFoundError"


@pytest.mark.parametrize("tool", MUTATING)
def test_changes_are_refused_during_pie(editor: FakeEditor, tool: str) -> None:
    editor.pie = True
    payload = _call(editor, tool)
    assert (payload["success"], payload["error_type"]) == (False, "PIEActive")
    assert "validate_unreal_mesh keeps working" in payload["details"]["hint"]
    assert editor.transactions == [] and editor.spawned == 0 and editor.calls == []


def test_validation_reads_during_pie(editor: FakeEditor) -> None:
    editor.pie = True
    assert _call(editor, "validate_unreal_mesh")["is_valid"] is True


def test_xatlas_is_refused_above_its_limit_and_auto_picks_patch_builder(editor: FakeEditor) -> None:
    """XAtlas crashed the editor on 130k- and 318k-triangle meshes (B3)."""
    big = editor.objects[BIG]
    before = big.dynamic_mesh_component.get_dynamic_mesh()
    refused = _call(editor, "compute_unreal_mesh_uv", mesh_path=BIG, method="xatlas")
    assert (refused["success"], refused["error_type"]) == (False, "ValidationError")
    assert refused["details"] == {
        "code": "MESH_TOO_LARGE_FOR_XATLAS",
        "triangle_count": 318_400,
        "max_triangles": mesh_scripts.XATLAS_MAX_TRIANGLES,
        "suggested_method": "patch_builder",
    }
    assert editor.calls == [] and big.dynamic_mesh_component.get_dynamic_mesh() is before

    auto = _call(editor, "compute_unreal_mesh_uv", mesh_path=BIG)
    assert (auto["method"], auto["requested_method"]) == ("patch_builder", "auto")
    assert ("auto_generate_x_atlas_mesh_u_vs", True) not in editor.calls

    small = _call(editor, "compute_unreal_mesh_uv", mesh_path=BALL, method="auto_uv")
    assert (small["method"], small["requested_method"]) == ("xatlas", "auto")


def test_uv_results_are_read_from_the_mesh(editor: FakeEditor) -> None:
    payload = _call(editor, "compute_unreal_mesh_uv", mesh_path=BALL, method="patch_builder")
    assert payload["island_count"] == 11
    assert payload["coverage_ratio"] == pytest.approx(0.55)
    mesh = editor.objects[BALL].dynamic_mesh_component.get_dynamic_mesh()
    mesh.stats.update(uv_lo=(-1.0, 0.0), uv_hi=(1.0, 2.0))
    payload = _call(editor, "compute_unreal_mesh_uv", mesh_path=BALL, method="patch_builder")
    assert payload["coverage_ratio"] == pytest.approx(0.55 / 4)


def test_a_missing_uv_channel_is_refused(editor: FakeEditor) -> None:
    payload = _call(editor, "compute_unreal_mesh_uv", uv_channel=5)
    assert payload["details"] == {"code": "UV_CHANNEL_MISSING", "uv_channels": 1}
    assert editor.calls == []


def test_hull_reports_volumes(editor: FakeEditor) -> None:
    payload = _call(editor, "compute_unreal_convex_hull")
    assert payload["volume_ratio"] == pytest.approx(1.25)
    assert payload["hull_volume"] == pytest.approx(payload["source_volume"] * 1.25)
    editor.objects[BALL].dynamic_mesh_component.get_dynamic_mesh().stats["open_edges"] = 4
    open_source = _call(editor, "compute_unreal_convex_hull")
    assert open_source["source_volume"] is None and open_source["volume_ratio"] is None


def test_decomposition_reports_each_piece(editor: FakeEditor) -> None:
    payload = _call(editor, "decompose_unreal_convex_hull")
    assert payload["hull_count"] == len(payload["hulls"]) == 3
    assert [hull["index"] for hull in payload["hulls"]] == [0, 1, 2]
    assert payload["decomp_actor_path"] in editor.objects


def test_cut_reports_pieces_and_fill(editor: FakeEditor) -> None:
    filled = _call(editor, "cut_unreal_mesh_plane")
    assert filled["pieces"] == [BODY]
    assert filled["cut_faces_added"] == 6
    unfilled = _call(level_editor(), "cut_unreal_mesh_plane", fill_holes=False)
    assert unfilled["cut_faces_added"] == 0


def test_remesh_reports_the_mean_edge_length(editor: FakeEditor) -> None:
    payload = _call(editor, "remesh_unreal_mesh")
    assert payload["average_edge_length"] == pytest.approx(20.0, rel=1e-3)


def test_validation_unwraps_out_parameter_tuples(editor: FakeEditor) -> None:
    """get_num_open_border_loops returns (count, ambiguous) on UE 5.6 and 5.7."""
    mesh = editor.objects[BODY].dynamic_mesh_component.get_dynamic_mesh()
    mesh.stats.update(open_edges=4, open_loops=1)
    payload = _call(editor, "validate_unreal_mesh")
    assert payload["is_valid"] is False
    assert payload["checks"] == {"watertight": False, "compact": True, "has_normals": True}
    assert (payload["open_border_edges"], payload["open_border_loops"]) == (4, 1)
    assert payload["issues"] == ["4 open border edges in 1 open border loops"]


def test_conversion_places_a_copy_of_the_asset(editor: FakeEditor) -> None:
    payload = _call(editor, "convert_unreal_mesh_format")
    copy = editor.objects[payload["result_path"]]
    assert copy.label == "SM_Crate_Dynamic"
    assert copy.dynamic_mesh_component.get_dynamic_mesh().stats["verts"] == 24
    assert editor.transactions == ["simul: convert SM_Crate to a DynamicMeshActor"]


@pytest.mark.parametrize(
    ("overrides", "code"),
    [({"mesh_path": BODY}, "NOT_A_STATIC_MESH_ASSET"), ({"mesh_path": "/Game/Meshes/Nope"}, None)],
)
def test_conversion_needs_a_static_mesh_asset(editor: FakeEditor, overrides: Dict[str, Any], code: Any) -> None:
    payload = _call(editor, "convert_unreal_mesh_format", **overrides)
    assert payload["success"] is False
    assert (payload.get("details") or {}).get("code") == code
    assert payload["error_type"] == ("ValidationError" if code else "NotFoundError")


def test_conversion_to_a_static_mesh_is_refused_honestly(editor: FakeEditor) -> None:
    """It used to return a success with only a note."""
    payload = _call(editor, "convert_unreal_mesh_format", mesh_path=BODY, target_format="static_mesh")
    assert (payload["error_type"], payload["details"]["code"]) == ("UnsupportedOperation", "NOT_IMPLEMENTED")


@pytest.mark.parametrize(
    ("tool", "overrides", "code"),
    [
        ("generate_unreal_mesh_primitive", {"primitive_type": "pyramid"}, "UNKNOWN_PRIMITIVE"),
        ("edit_unreal_mesh_topology", {"operation": "bogus_op"}, "UNKNOWN_OPERATION"),
        ("convert_unreal_mesh_format", {"target_format": "fbx"}, "UNKNOWN_FORMAT"),
        ("compute_unreal_mesh_uv", {"method": "bogus"}, "UNKNOWN_UV_METHOD"),
    ],
)
def test_unknown_choices_are_refused_before_the_editor_is_asked(
    editor: FakeEditor, tool: str, overrides: Dict[str, Any], code: str
) -> None:
    payload = _call(editor, tool, **overrides)
    assert (payload["success"], payload["error_type"]) == (False, "ValidationError")
    assert payload["details"]["code"] == code
    assert editor.calls == [] and editor.events == []


def test_the_attachment_guard_refusal_is_typed() -> None:
    session = UnrealRuntimeSession()

    async def guard_refused(code: str, mode: str = "ExecuteFile") -> Dict[str, Any]:
        return {
            "ReturnValue": False,
            "CommandResult": (
                "Traceback (most recent call last):\n  File \"<string>\", line 52, in verify\n"
                "RuntimeError: Unreal attachment changed (map_path); run simul unreal attach again\n"
            ),
        }

    session._execute_python = guard_refused  # type: ignore[method-assign]
    payload = asyncio.run(session.simplify_mesh(BODY))
    assert payload == {
        "success": False,
        "error_type": "InvalidState",
        "error": "Unreal attachment changed (map_path); run simul unreal attach again; nothing was changed",
        "details": {
            "code": "ATTACHMENT_CHANGED",
            "hint": "The attached editor now shows another map or instance: run simul unreal attach again.",
        },
    }


def test_mesh_scripts_run_under_the_longer_timeout(editor: FakeEditor) -> None:
    session = _session(editor)
    session.timeout = 30
    seen = []
    execute = session._execute_python

    async def watch(code: str, mode: str = "ExecuteFile") -> Dict[str, Any]:
        seen.append(session.timeout)
        return await execute(code, mode)

    session._execute_python = watch  # type: ignore[method-assign]
    asyncio.run(session.validate_mesh(BODY))
    assert seen == [mesh_scripts.MESH_SCRIPT_TIMEOUT_S]
    assert session.timeout == 30


def test_scripts_bind_no_unreal_object_at_module_level(editor: FakeEditor) -> None:
    """The editor's shared __main__ keeps module-level names; an object there leaks the world."""
    namespace: Dict[str, Any] = {"__name__": "__main__"}
    script = mesh_scripts.mesh_script(mesh_scripts.VALIDATE, {"mesh": BODY})
    import io
    import sys
    from contextlib import redirect_stdout

    sys.modules["unreal"] = editor.module()  # type: ignore[assignment]
    try:
        with redirect_stdout(io.StringIO()):
            exec(script, namespace)
    finally:
        sys.modules.pop("unreal", None)
    leaked = {name for name, value in namespace.items() if isinstance(value, (DynamicMesh,))
              or type(value).__module__ == "tests.unreal.fake_geometry"}
    assert leaked == set()
