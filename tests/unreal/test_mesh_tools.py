"""The Unreal mesh tools, run end to end against a fake editor.

Each test runs the real editor script a mesh tool sends (``_unreal_mesh_scripts``)
in ``tests.unreal.fake_geometry``'s fake ``unreal`` module, through the real
``UnrealRuntimeSession`` method, and checks the reply against what the script
did to the fake level, and against the tool's response model.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import pytest

from simul.adapters import _unreal_mesh_scripts as mesh_scripts
from simul.adapters import unreal_runtime
from simul.adapters._unreal_errors import UnrealTimeoutError
from simul.adapters.unreal_runtime import UnrealRuntimeSession
from simul.mcp.registration._reg_unreal import InvalidArgument, request_model_for, session_arguments
from simul.mcp.registration._unreal_tools import UNREAL_TOOLS
from tests.unreal.fake_geometry import LEVEL, LEVEL_PACKAGE, DynamicMesh, FakeEditor, level_editor, transform

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
    "edit_unreal_mesh_topology": (
        "edit_mesh_topology",
        {"mesh_path": BODY, "operation": "extrude_faces", "face_selection": "normal:0,0,1"},
        BODY,
    ),
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

    async def execute(code: str, mode: str = "ExecuteFile", timeout: Any = None) -> Dict[str, Any]:
        session.timeouts.append(timeout)  # type: ignore[attr-defined]
        return editor.run(code)

    session.timeouts = []  # type: ignore[attr-defined]
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
    assert actor.get_actor_location().y == 100
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
    # The new mesh's physics is rebuilt once it is in, so a trace hits what the reply describes.
    swap = editor.events.index(("set_dynamic_mesh", actor.path, after.name))
    assert ("update_collision", actor.path, False) in editor.events[swap:]


def test_a_touch_after_an_undo_binds_the_component_to_its_mesh_again(editor: FakeEditor) -> None:
    """After an Undo, in-place edits outside simul left the bounds stale (#337 review).

    An Undo leaves the level dirty; the mesh the component got back is set
    again, which binds it, and the collision is rebuilt.
    """
    editor.dirty.add(LEVEL_PACKAGE)
    _call(editor, "validate_unreal_mesh")
    assert [event[0] for event in editor.events] == ["rebind", "update_collision"]
    assert editor.events[0][1] == BODY and editor.events[1] == ("update_collision", BODY, False)


def test_an_unsaved_level_is_bound_again_on_every_touch(editor: FakeEditor) -> None:
    """A never-saved level's actors are in no dirty list, though an Undo there leaves them unbound."""
    editor.package = "/Temp/__ExternalActors__/Untitled_1/6/X3/PUOGNFO1EKOSMJQ1T795XR"
    _call(editor, "validate_unreal_mesh")
    assert [event[0] for event in editor.events] == ["rebind", "update_collision"]


def test_a_read_of_a_saved_level_dirties_nothing(editor: FakeEditor) -> None:
    """Binding the mesh again renames it and dirties the level, so a clean level is left alone."""
    mesh = editor.objects[BODY].dynamic_mesh_component.get_dynamic_mesh()
    name = mesh.name
    _call(editor, "validate_unreal_mesh")
    assert editor.events == [("update_collision", BODY, False)]
    assert editor.dirty == set() and mesh.name == name


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
    editor.fail["generate_collision_from_mesh"] = "GenerateCollisionFromMesh: no shapes"
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


@pytest.mark.parametrize("tool", sorted(name for name, (_, _, edited) in TOOLS.items() if edited))
def test_a_static_mesh_actor_is_refused_with_the_way_in(editor: FakeEditor, tool: str) -> None:
    """An edit in place used to fail with "AttributeError: ... dynamic_mesh_component" (M12)."""
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


def test_an_asset_path_is_told_apart_during_pie(editor: FakeEditor) -> None:
    """EditorAssetLibrary finds nothing while PIE runs, which made an asset path "Actor not found" (#342 review).

    The asset registry still knows the asset; the editor cannot load it, which is PIEActive.
    """
    editor.pie = True
    for path in ("/Game/Meshes/SM_Crate", "/Game/Meshes/SM_Crate.SM_Crate"):
        payload = _call(editor, "validate_unreal_mesh", mesh_path=path)
        assert payload["error_type"] == "PIEActive", payload
        assert "validate_unreal_mesh keeps working" in payload["details"]["hint"]


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
    payload = _call(editor, "compute_unreal_convex_hull", mesh_path=BODY)
    assert payload["volume_ratio"] == pytest.approx(1.1)
    assert payload["hull_volume"] == pytest.approx(payload["source_volume"] * 1.1)
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
    assert filled["removed_triangle_count"] == 12
    unfilled = _call(level_editor(), "cut_unreal_mesh_plane", fill_holes=False)
    assert unfilled["cut_faces_added"] == 0


def test_remesh_reports_the_mean_edge_length(editor: FakeEditor) -> None:
    payload = _call(editor, "remesh_unreal_mesh")
    assert payload["average_edge_length"] == pytest.approx(20.0, rel=1e-3)


def test_validation_unwraps_out_parameter_tuples(editor: FakeEditor) -> None:
    """get_num_open_border_loops returns (count, ambiguous) on UE 5.6 and 5.7."""
    mesh = editor.objects[BODY].dynamic_mesh_component.get_dynamic_mesh()
    mesh.stats.update(open_edges=4, open_loops=1)
    payload = _call(editor, "validate_unreal_mesh", checks=["closed"])
    assert payload["is_valid"] is False
    assert payload["checks"] == {"closed": {"passed": False, "count": 4, "detail": "4 open border edges in 1 loops"}}
    assert payload["issues"] == ["closed: 4 open border edges in 1 loops"]


def test_conversion_places_a_copy_of_the_asset(editor: FakeEditor) -> None:
    payload = _call(editor, "convert_unreal_mesh_format")
    copy = editor.objects[payload["result_path"]]
    assert copy.label == "SM_Crate_Dynamic"
    assert copy.dynamic_mesh_component.get_dynamic_mesh().stats["verts"] == 24
    assert editor.transactions == ["simul: convert SM_Crate to a DynamicMeshActor"]


@pytest.mark.parametrize(
    ("overrides", "code"),
    [({"mesh_path": BODY}, "NOT_A_STATIC_MESH"), ({"mesh_path": "/Game/Meshes/Nope"}, None)],
)
def test_conversion_needs_a_static_mesh_asset(editor: FakeEditor, overrides: Dict[str, Any], code: Any) -> None:
    payload = _call(editor, "convert_unreal_mesh_format", **overrides)
    assert payload["success"] is False
    assert (payload.get("details") or {}).get("code") == code
    assert payload["error_type"] == ("ValidationError" if code else "NotFoundError")


@pytest.mark.parametrize(
    ("tool", "overrides", "code"),
    [
        ("generate_unreal_mesh_primitive", {"primitive_type": "pyramid"}, "UNKNOWN_PRIMITIVE"),
        ("edit_unreal_mesh_topology", {"operation": "loop_cut"}, "UNKNOWN_OPERATION"),
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

    async def guard_refused(code: str, mode: str = "ExecuteFile", timeout: Any = None) -> Dict[str, Any]:
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
    asyncio.run(session.validate_mesh(BODY))
    assert session.timeouts == [mesh_scripts.MESH_SCRIPT_TIMEOUT_S]  # type: ignore[attr-defined]
    assert session.timeout == 30


@pytest.mark.parametrize("tool", sorted(TOOLS))
def test_scripts_bind_no_unreal_object_at_module_level(editor: FakeEditor, tool: str) -> None:
    """The editor's shared __main__ keeps module-level names; an object there leaks the world."""
    payload = _call(editor, tool)
    assert payload.get("success") is not False, payload
    leaked = {name for name, value in editor.namespace.items() if isinstance(value, (DynamicMesh,))
              or type(value).__module__ == "tests.unreal.fake_geometry"}
    assert leaked == set()


# ---------------------------------------------------------------------------
# What the tools compute (PR 2: B5, B6, B13, M8-M11)
# ---------------------------------------------------------------------------


def test_box_dimensions_follow_unreal_axes(editor: FakeEditor) -> None:
    """height used to land on Y, so a 200 x 100 x 50 box came out 50 cm tall (M10)."""
    payload = _call(editor, "generate_unreal_mesh_primitive",
                    dimensions={"width": 200, "depth": 100, "height": 50})
    args = editor.arguments["append_box"]
    assert args[3:6] == (200.0, 100.0, 50.0)  # dimension_x, dimension_y, dimension_z
    assert args[1].polygroup_mode == "PER_QUAD"
    assert payload["dimensions"] == {"width": 200.0, "depth": 100.0, "height": 50.0}
    assert (payload["bounds_min"], payload["bounds_max"]) == ([-100.0, -50.0, -25.0], [100.0, 50.0, 25.0])


def test_capsule_and_torus_take_their_own_dimensions(editor: FakeEditor) -> None:
    """hemisphere_height also set the step count, and torus always raised TypeError (M10)."""
    _call(editor, "generate_unreal_mesh_primitive", primitive_type="capsule",
          dimensions={"radius": 30, "length": 400}, segments=32)
    args = editor.arguments["append_capsule"]
    assert args[3:8] == (30.0, 400.0, 8, 32, 0)  # radius, line length, hemisphere, circle, segment steps
    torus = _call(editor, "generate_unreal_mesh_primitive", primitive_type="torus",
                  dimensions={"major_radius": 60, "minor_radius": 15})
    args = editor.arguments["append_torus"]
    assert args[4:8] == (60.0, 15.0, 32, 16)
    assert torus["bounds_max"] == [75.0, 75.0, 15.0]


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"primitive_type": "capsule", "dimensions": {"hemisphere_height": 50}}, "UNKNOWN_DIMENSION"),
        ({"dimensions": {"width": 0}}, "INVALID_DIMENSION"),
        ({"dimensions": {"width": -5}}, "INVALID_DIMENSION"),
        ({"dimensions": {"width": "wide"}}, "INVALID_DIMENSION"),
        ({"primitive_type": "cone", "dimensions": {"top_radius": 60}}, "INVALID_DIMENSION"),
        ({"primitive_type": "torus", "dimensions": {"minor_radius": 80}}, "INVALID_DIMENSION"),
        ({"segments": 2}, "INVALID_SEGMENTS"),
        ({"segments": 513}, "INVALID_SEGMENTS"),
    ],
)
def test_impossible_primitives_are_refused(editor: FakeEditor, overrides: Dict[str, Any], code: str) -> None:
    """Zero and negative sizes were accepted, segments clamped silently and unbounded (M10)."""
    payload = _call(editor, "generate_unreal_mesh_primitive", **overrides)
    assert (payload["error_type"], payload["details"]["code"]) == ("ValidationError", code)
    assert editor.calls == []


def test_a_boolean_uses_both_world_transforms(editor: FakeEditor) -> None:
    """Both meshes used identity transforms, so a cutter at the box's face carved its centre (B5)."""
    editor.objects[BODY].transform = transform((0, 1000, 0), yaw=90, scale=(2, 1, 1))
    editor.objects[BALL].transform = transform((100, 1000, 0))
    payload = _call(editor, "apply_unreal_mesh_boolean", operation="difference")
    target, target_xf, tool, tool_xf, operation, options = editor.arguments["apply_mesh_boolean"]
    assert target_xf is editor.objects[BODY].transform and tool_xf is editor.objects[BALL].transform
    assert options.output_transform_space == "TARGET_TRANSFORM_SPACE"
    assert operation == "SUBTRACT"
    assert (payload["operation"], payload["requested_operation"]) == ("subtract", "difference")
    assert payload["result_volume"] < payload["previous_volume"]


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"operation": "xor"}, "UNKNOWN_OPERATION"),
        ({"tool_mesh_path": BODY}, "SAME_ACTOR"),
    ],
)
def test_a_boolean_never_falls_back_to_union(editor: FakeEditor, overrides: Dict[str, Any], code: str) -> None:
    """An unknown operation used to run a union reported as asked (B5)."""
    payload = _call(editor, "apply_unreal_mesh_boolean", **overrides)
    assert (payload["error_type"], payload["details"]["code"]) == ("ValidationError", code)
    assert ("apply_mesh_boolean", True) not in editor.calls


def test_a_hull_actor_takes_the_source_transform_and_is_collision(editor: FakeEditor) -> None:
    """It got the location only, no simplification and no collision shape (B6)."""
    editor.objects[BALL].transform = transform((0, 2000, 0), yaw=90, scale=(2, 1, 1))
    payload = _call(editor, "compute_unreal_convex_hull", max_faces=64)
    hull = editor.objects[payload["hull_actor_path"]]
    assert hull.get_actor_transform() is editor.objects[BALL].transform
    options = editor.arguments["generate_collision_from_mesh"][1]
    assert (options.method, options.max_convex_hulls_per_mesh, options.convex_hull_target_face_count) == (
        "CONVEX_HULLS", 1, 64)
    assert options.simplify_hulls is True and options.emit_transaction is False
    assert not (options.auto_detect_spheres or options.auto_detect_boxes or options.auto_detect_capsules)
    assert payload["collision"] == [{"actor_path": hull.path, "asset_path": None, "collision_shape_count": 1,
                                     "saved_file": None}]
    assert editor.transactions == ["simul: convex hull of Ball"]


def test_a_hull_can_be_the_source_collision(editor: FakeEditor) -> None:
    payload = _call(editor, "compute_unreal_convex_hull", output="collision")
    assert payload["hull_actor_path"] is None and editor.spawned == 0
    assert payload["collision"] == [{"actor_path": BALL, "asset_path": None, "collision_shape_count": 1,
                                     "saved_file": None}]
    assert BALL in editor.modified and len(editor.transactions) == 1
    both = _call(level_editor(), "compute_unreal_convex_hull", output="both")
    assert [c["actor_path"] for c in both["collision"]] == [both["hull_actor_path"], BALL]


def test_new_collision_reaches_the_physics_at_once(editor: FakeEditor) -> None:
    """After Redo a hull was in the BodySetup but traces missed it until the physics was rebuilt (#338 review)."""
    _call(editor, "compute_unreal_convex_hull", output="collision")
    collided = editor.events.index(("set_collision", BALL, 1))
    assert ("update_collision", BALL, False) in editor.events[collided:]


def test_an_exact_hull_is_not_simplified(editor: FakeEditor) -> None:
    _call(editor, "compute_unreal_convex_hull", max_faces=0)
    assert editor.arguments["generate_collision_from_mesh"][1].simplify_hulls is False


def test_decomposition_maps_its_limits_and_detects_no_shapes(editor: FakeEditor) -> None:
    """max_vertices_per_hull was passed as a face count, and spheres came back as a 100 cm cube (B6, M8)."""
    payload = _call(editor, "decompose_unreal_convex_hull", max_hulls=5, max_vertices_per_hull=16,
                    error_tolerance=2.5, min_part_thickness=0.5)
    options = editor.arguments["generate_collision_from_mesh"][1]
    assert options.max_convex_hulls_per_mesh == 5
    assert options.convex_hull_target_face_count == 2 * 16 - 4
    assert options.convex_decomposition_error_tolerance == 2.5
    assert options.convex_decomposition_min_part_thickness == 0.5
    assert not (options.auto_detect_spheres or options.auto_detect_boxes or options.auto_detect_capsules)
    assert payload["hull_count"] == 5 == len(payload["hulls"])
    assert payload["collision"][0]["collision_shape_count"] == 5


def test_an_empty_decomposition_is_an_error(editor: FakeEditor) -> None:
    editor.objects[BALL].dynamic_mesh_component.get_dynamic_mesh().stats["convex_parts"] = 0
    payload = _call(editor, "decompose_unreal_convex_hull")
    assert (payload["error_type"], payload["details"]["code"]) == ("EngineError", "EMPTY_RESULT")
    assert editor.spawned == 0 and editor.transactions == []


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"max_hulls": 0}, "INVALID_LIMIT"),
        ({"max_vertices_per_hull": 3}, "INVALID_LIMIT"),
        ({"error_tolerance": -1.0}, "INVALID_LIMIT"),
        ({"output": "asset"}, "UNKNOWN_OUTPUT"),
    ],
)
def test_decomposition_limits_are_checked(editor: FakeEditor, overrides: Dict[str, Any], code: str) -> None:
    payload = _call(editor, "decompose_unreal_convex_hull", **overrides)
    assert payload["details"]["code"] == code and editor.calls == []


def test_a_world_plane_is_cut_in_the_mesh_space(editor: FakeEditor) -> None:
    """The plane was taken in mesh-local space, so a world plane through the actor cut nothing (B13)."""
    editor.objects[BODY].transform = transform((100, 200, 300), yaw=90, scale=(2, 1, 1))
    payload = _call(editor, "cut_unreal_mesh_plane", plane_origin=[100, 210, 300], plane_normal=[0, 1, 0])
    # World +Y is the actor's +X once its 90-degree yaw is undone; X is scaled by 2.
    assert payload["plane_origin_local"] == pytest.approx([5.0, 0.0, 0.0], abs=1e-9)
    assert payload["plane_normal_local"] == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)
    local = _call(level_editor(), "cut_unreal_mesh_plane", plane_origin=[1, 2, 3], plane_normal=[0, 0, 1],
                  space="local")
    assert local["plane_origin_local"] == [1.0, 2.0, 3.0]


def test_keep_both_sides_keeps_both_halves_in_one_transaction(editor: FakeEditor) -> None:
    """keep_both_sides used to delete half the box anyway (B13)."""
    payload = _call(editor, "cut_unreal_mesh_plane", keep_both_sides=True)
    other = editor.objects[payload["pieces"][1]]
    assert payload["pieces"][0] == BODY and other.label == "Body_Cut"
    assert other.get_actor_transform() is editor.objects[BODY].transform
    assert payload["removed_triangle_count"] is None and payload["piece_triangle_counts"] == [12, 12]
    assert editor.transactions == ["simul: cut mesh"]


def test_a_plane_that_misses_the_mesh_changes_nothing(editor: FakeEditor) -> None:
    mesh = editor.objects[BODY].dynamic_mesh_component.get_dynamic_mesh()
    mesh.stats["side_tris"] = {"back": 12, "front": 0}
    payload = _call(editor, "cut_unreal_mesh_plane")
    assert payload["details"]["code"] == "PLANE_MISSES_MESH"
    assert editor.objects[BODY].dynamic_mesh_component.get_dynamic_mesh() is mesh and editor.transactions == []


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"plane_normal": [0, 0, 0]}, "INVALID_PLANE"),
        ({"space": "screen"}, "UNKNOWN_SPACE"),
    ],
)
def test_bad_planes_are_refused(editor: FakeEditor, overrides: Dict[str, Any], code: str) -> None:
    assert _call(editor, "cut_unreal_mesh_plane", **overrides)["details"]["code"] == code


def test_an_extrude_moves_the_selected_faces_along_their_normal(editor: FakeEditor) -> None:
    """Every face used to be extruded straight up, giving two shells and zero volume (M9)."""
    payload = _call(editor, "edit_unreal_mesh_topology", face_selection="normal:0,0,1,10", distance=25)
    work, options, selection = editor.arguments["apply_mesh_linear_extrude_faces"]
    assert selection.how == ("normal", (0.0, 0.0, 1.0), 10.0)
    assert (options.distance, options.direction_mode) == (25.0, "AVERAGE_FACE_NORMAL")
    assert (payload["faces_affected"], payload["edges_affected"]) == (2, None)
    assert payload["face_selection"] == "normal:0,0,1,10"


def test_an_inset_insets_each_selected_face(editor: FakeEditor) -> None:
    payload = _call(editor, "edit_unreal_mesh_topology", operation="inset_faces",
                    face_selection="box:-60,-60,40,60,60,60", offset=3)
    work, options, selection = editor.arguments["apply_mesh_inset_outset_faces"]
    assert selection.how == ("box", (-60.0, -60.0, 40.0), (60.0, 60.0, 60.0))
    assert (options.distance, options.area_mode) == (3.0, "PER_POLYGROUP")
    assert payload["faces_affected"] == 4 and payload["offset"] == 3.0


@pytest.mark.parametrize(("selection", "mode", "edges"), [
    ({"face_selection": None, "edge_selection": "sharp:30"}, "SELECTED_EDGES", 12),
    ({"face_selection": None, "edge_selection": "polygroup"}, "ALL_POLYGROUP_EDGES", None),
    ({"face_selection": "normal:0,0,1"}, "TRIANGLE_AREA", 4),
])
def test_a_bevel_bevels_what_was_selected(editor: FakeEditor, selection: Dict[str, Any], mode: str,
                                          edges: Any) -> None:
    """The bevel used an empty triangle selection and bevelled nothing (M9)."""
    payload = _call(editor, "edit_unreal_mesh_topology", operation="bevel_edges", count=2, **selection)
    work, chosen, used_mode, options = editor.arguments["apply_mesh_bevel_selection"]
    assert used_mode == mode and options.subdivisions == 2 and options.bevel_distance == 5.0
    assert payload["edges_affected"] == edges
    if "sharp" in (selection.get("edge_selection") or ""):
        assert chosen.how == ("sharp", 30.0)


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"face_selection": None}, "SELECTION_REQUIRED"),
        ({"operation": "bevel_edges", "face_selection": None}, "SELECTION_REQUIRED"),
        ({"face_selection": "top"}, "INVALID_SELECTION"),
        ({"face_selection": "normal:0,0,0"}, "INVALID_SELECTION"),
        ({"face_selection": "box:1,1,1,0,0,0"}, "INVALID_SELECTION"),
        ({"operation": "bevel_edges", "face_selection": None, "edge_selection": "round"}, "INVALID_SELECTION"),
        ({"edge_selection": "sharp"}, "SELECTION_CONFLICT"),
        ({"distance": 0.0}, "INVALID_DISTANCE"),
        (
            {"operation": "bevel_edges", "face_selection": None, "edge_selection": "polygroup", "count": 17},
            "INVALID_COUNT",
        ),
    ],
)
def test_topology_inputs_are_checked(editor: FakeEditor, overrides: Dict[str, Any], code: str) -> None:
    """loop_cut, scale_faces, selections and scale used to be accepted and ignored (M9)."""
    payload = _call(editor, "edit_unreal_mesh_topology", **overrides)
    assert (payload["error_type"], payload["details"]["code"]) == ("ValidationError", code)
    assert editor.calls == []


def test_an_empty_selection_changes_nothing(editor: FakeEditor) -> None:
    mesh = editor.objects[BODY].dynamic_mesh_component.get_dynamic_mesh()
    mesh.stats["facing"] = {(0.0, 0.0, -1.0): 0}
    payload = _call(editor, "edit_unreal_mesh_topology", face_selection="normal:0,0,-1")
    assert payload["details"]["code"] == "EMPTY_SELECTION" and editor.transactions == []


@pytest.mark.parametrize(("scheme", "function"), [
    ("catmull_clark", "apply_polygroup_catmull_clark_sub_d"),
    ("loop", "apply_triangle_loop_sub_d"),
    ("uniform", "apply_uniform_tessellation"),
])
def test_each_subdivision_scheme_runs_its_own_algorithm(editor: FakeEditor, scheme: str, function: str) -> None:
    """loop, bilinear and bogus all ran uniform tessellation (M9)."""
    payload = _call(editor, "subdivide_unreal_mesh", scheme=scheme, level=2)
    assert editor.arguments[function][1] == 2
    assert payload["scheme"] == scheme and payload["level"] == 2


def test_catmull_clark_on_a_triangle_mesh_says_to_use_loop(editor: FakeEditor) -> None:
    payload = _call(editor, "subdivide_unreal_mesh", mesh_path=BALL)
    assert payload["error_type"] == "EngineError"
    assert "scheme 'loop'" in payload["details"]["hint"]


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"level": 9}, "INVALID_LEVEL"),
        ({"level": 0}, "INVALID_LEVEL"),
        ({"scheme": "bilinear"}, "UNKNOWN_SCHEME"),
    ],
)
def test_subdivision_inputs_are_checked(editor: FakeEditor, overrides: Dict[str, Any], code: str) -> None:
    """Level 9 used to be clamped to 4 without a word (M9)."""
    assert _call(editor, "subdivide_unreal_mesh", **overrides)["details"]["code"] == code


def test_a_subdivision_too_large_is_refused_first(editor: FakeEditor) -> None:
    payload = _call(editor, "subdivide_unreal_mesh", mesh_path=BIG, level=2, scheme="loop")
    assert payload["details"]["code"] == "RESULT_TOO_LARGE"
    assert (payload["details"]["triangle_count"], payload["details"]["predicted"]) == (318_400 * 16, True)
    assert ("apply_triangle_loop_sub_d", True) not in editor.calls


def test_simplify_to_a_fraction_and_to_a_tolerance(editor: FakeEditor) -> None:
    """max_error was ignored, and target_percentage=50 was accepted and did nothing (M9)."""
    fraction = _call(editor, "simplify_unreal_mesh", target_percentage=0.25)
    assert editor.arguments["apply_simplify_to_triangle_count"][1] == 480
    assert (fraction["method"], fraction["target_triangle_count"], fraction["result_triangles"]) == (
        "triangle_count", 480, 480)
    tolerance = _call(level_editor(), "simplify_unreal_mesh", target_percentage=None, max_error=0.5)
    assert (tolerance["method"], tolerance["tolerance"]) == ("tolerance", 0.5)
    assert tolerance["target_triangle_count"] is None


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"target_percentage": 50}, "INVALID_TARGET"),
        ({"target_percentage": 0.5, "max_error": 1.0}, "CONFLICTING_TARGETS"),
        ({"target_percentage": None, "target_triangle_count": 5000}, "NOTHING_TO_SIMPLIFY"),
        ({"target_percentage": None, "max_error": -1.0}, "INVALID_TARGET"),
    ],
)
def test_simplify_inputs_are_checked(editor: FakeEditor, overrides: Dict[str, Any], code: str) -> None:
    payload = _call(editor, "simplify_unreal_mesh", **overrides)
    assert payload["details"]["code"] == code
    assert ("apply_simplify_to_triangle_count", True) not in editor.calls


def test_remesh_keeps_the_triangle_count_by_default(editor: FakeEditor) -> None:
    payload = _call(editor, "remesh_unreal_mesh", target_edge_length=None, smoothing_iterations=7)
    options, uniform = editor.arguments["apply_uniform_remesh"][1:]
    assert (uniform.target_type, uniform.target_triangle_count, options.remesh_iterations) == ("TRIANGLE_COUNT", 12, 7)
    assert (payload["target_type"], payload["target_triangle_count"], payload["passes"]) == ("triangle_count", 12, 7)


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"mode": "adaptive"}, "UNKNOWN_MODE"),
        ({"target_triangle_count": 100}, "CONFLICTING_TARGETS"),
        ({"target_edge_length": 0.001}, "RESULT_TOO_LARGE"),
        ({"smoothing_iterations": 0}, "INVALID_PASSES"),
    ],
)
def test_remesh_inputs_are_checked(editor: FakeEditor, overrides: Dict[str, Any], code: str) -> None:
    """adaptive used to be accepted and ignored (M9)."""
    assert _call(editor, "remesh_unreal_mesh", **overrides)["details"]["code"] == code


@pytest.mark.parametrize(("method", "function"), [("xatlas", "repack_mesh_u_vs"),
                                                  ("patch_builder", "auto_generate_patch_builder_mesh_u_vs")])
def test_uv_islands_are_packed_with_the_padding_asked_for(editor: FakeEditor, method: str, function: str) -> None:
    """island_padding was ignored (M9)."""
    payload = _call(editor, "compute_unreal_mesh_uv", method=method, island_padding=4, texture_resolution=1024)
    options = editor.arguments[function][2]
    packing = options if method == "xatlas" else options.packing_options
    assert packing.target_image_width == 256
    assert (payload["island_padding"], payload["packing_resolution"]) == (4.0, 256)


def test_a_clamped_packing_resolution_reports_the_padding_packed(editor: FakeEditor) -> None:
    payload = _call(editor, "compute_unreal_mesh_uv", island_padding=0.1, texture_resolution=1024)
    assert payload["packing_resolution"] == 4096 and payload["island_padding"] == 0.25


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"uv_channel": 8}, "INVALID_UV_CHANNEL"),
        ({"island_padding": 0}, "INVALID_PADDING"),
        ({"texture_resolution": 32}, "INVALID_RESOLUTION"),
    ],
)
def test_uv_inputs_are_checked(editor: FakeEditor, overrides: Dict[str, Any], code: str) -> None:
    assert _call(editor, "compute_unreal_mesh_uv", **overrides)["details"]["code"] == code


def test_validation_runs_every_check_it_names(editor: FakeEditor) -> None:
    """The description promised checks that did not exist, and checks was ignored (M11)."""
    mesh = editor.objects[BODY].dynamic_mesh_component.get_dynamic_mesh()
    mesh.stats.update(bowties=2, degenerate=3, volume=-5.0, unset_uvs=True, vertex_gaps=True)
    payload = _call(editor, "validate_unreal_mesh")
    checks = payload["checks"]
    assert list(checks) == list(mesh_scripts.VALIDATION_CHECKS)
    assert checks["closed"] == {"passed": True, "count": 0, "detail": "no open border edges"}
    assert (checks["manifold"]["passed"], checks["manifold"]["count"]) == (False, 2)
    assert (checks["degenerate_triangles"]["passed"], checks["degenerate_triangles"]["count"]) == (False, 3)
    assert checks["orientation"]["passed"] is False and "inside out" in checks["orientation"]["detail"]
    assert checks["compact"]["passed"] is False
    assert checks["uvs"]["passed"] is False and "without UVs" in checks["uvs"]["detail"]
    assert payload["is_valid"] is False and len(payload["issues"]) == 5
    # The checks ran on copies; the mesh is as it was.
    assert editor.objects[BODY].dynamic_mesh_component.get_dynamic_mesh() is mesh and mesh.stats["verts"] == 8


def test_validation_runs_only_the_checks_asked_for(editor: FakeEditor) -> None:
    payload = _call(editor, "validate_unreal_mesh", checks=["normals", "closed"])
    assert list(payload["checks"]) == ["closed", "normals"] and payload["is_valid"] is True
    assert editor.calls == []
    refused = _call(editor, "validate_unreal_mesh", checks=["self_intersection"])
    assert refused["details"]["code"] == "UNKNOWN_CHECK"


def test_an_open_mesh_has_no_orientation_to_check(editor: FakeEditor) -> None:
    editor.objects[BODY].dynamic_mesh_component.get_dynamic_mesh().stats.update(open_edges=4, open_loops=1)
    payload = _call(editor, "validate_unreal_mesh", checks=["orientation"])
    assert payload["checks"]["orientation"]["passed"] is True and payload["volume"] is None


def test_destructive_and_idempotent_hints_are_consistent() -> None:
    """Hull and decomposition disagreed; each mesh tool's hints now say what it does."""
    hints = {name: (SPECS[name].read_only, SPECS[name].destructive, SPECS[name].idempotent) for name in TOOLS}
    assert hints["validate_unreal_mesh"] == (True, False, True)
    assert hints["generate_unreal_mesh_primitive"] == (False, False, False)
    assert hints["compute_unreal_convex_hull"] == hints["decompose_unreal_convex_hull"] == (False, True, False)
    for name in ("apply_unreal_mesh_boolean", "edit_unreal_mesh_topology", "subdivide_unreal_mesh",
                 "simplify_unreal_mesh", "cut_unreal_mesh_plane", "remesh_unreal_mesh", "compute_unreal_mesh_uv"):
        assert hints[name] == (False, True, False), name


# ---------------------------------------------------------------------------
# StaticMesh interop (PR 3: B7 and the StaticMesh workflows)
# ---------------------------------------------------------------------------

CRATE_ASSET = "/Game/Meshes/SM_Crate"


@pytest.mark.parametrize(("path", "kind"), [(CRATE, "static_mesh_actor"), (CRATE_ASSET, "static_mesh_asset"),
                                            (BODY, "dynamic_mesh_actor")])
def test_validation_reads_any_mesh(editor: FakeEditor, path: str, kind: str) -> None:
    payload = _call(editor, "validate_unreal_mesh", mesh_path=path)
    vertices = 8 if kind == "dynamic_mesh_actor" else 24
    assert (payload["source_kind"], payload["is_valid"], payload["vertex_count"]) == (kind, True, vertices)


def test_a_static_mesh_actor_hull_lands_at_its_transform(editor: FakeEditor) -> None:
    payload = _call(editor, "compute_unreal_convex_hull", mesh_path=CRATE)
    hull = editor.objects[payload["hull_actor_path"]]
    assert payload["source_kind"] == "static_mesh_actor"
    assert hull.get_actor_transform() is editor.objects[CRATE].transform and hull.label == "Crate_Hull"
    assert editor.arguments["copy_mesh_from_static_mesh"][0] is editor.assets[CRATE_ASSET]


@pytest.mark.parametrize("tool", ["compute_unreal_convex_hull", "decompose_unreal_convex_hull"])
def test_collision_on_a_game_static_mesh_is_set_undoably_and_saved(editor: FakeEditor, tool: str) -> None:
    """Hulls and decompositions as a StaticMesh's simple collision, the top production need."""
    payload = _call(editor, tool, mesh_path=CRATE, output="collision", save=True)
    asset = editor.assets[CRATE_ASSET]
    assert payload["collision"] == [{"actor_path": None, "asset_path": asset.path,
                                     "collision_shape_count": asset.collision.shapes,
                                     "saved_file": "/Projects/Mesh/Content/Meshes/SM_Crate.uasset"}]
    assert ("set_static_collision", asset.path, asset.collision.shapes, True) in editor.events
    assert asset.modified == 1 and len(editor.transactions) == 1 and editor.spawned == 0
    assert editor.saved == ["/Game/Meshes/SM_Crate"]


def test_engine_content_is_never_given_collision(editor: FakeEditor) -> None:
    payload = _call(editor, "compute_unreal_convex_hull", mesh_path="/Engine/BasicShapes/Cube", output="both")
    assert (payload["error_type"], payload["details"]["code"]) == ("SandboxError", "OUTSIDE_GAME_CONTENT")
    assert editor.spawned == 0 and editor.transactions == [] and editor.saved == []


def test_a_static_mesh_asset_becomes_a_dynamic_mesh_actor_where_asked(editor: FakeEditor) -> None:
    payload = _call(editor, "convert_unreal_mesh_format", location=[10, 20, 30])
    placed = editor.objects[payload["result_path"]]
    assert (payload["source_format"], payload["placed_actor_path"]) == ("static_mesh_asset", placed.path)
    assert list(placed.get_actor_location()) == [10.0, 20.0, 30.0]
    assert placed.dynamic_mesh_component.materials == ["Material_SM_Crate"]
    assert payload["asset"] is None and payload["removed_actor_path"] is None


def test_a_static_mesh_actor_is_replaced_by_its_dynamic_copy_in_one_transaction(editor: FakeEditor) -> None:
    payload = _call(editor, "convert_unreal_mesh_format", mesh_path=CRATE, replace_actor=True)
    placed = editor.objects[payload["result_path"]]
    assert placed.get_actor_transform() is editor.objects[CRATE].transform and placed.label == "Crate"
    assert editor.objects[CRATE].valid is False and payload["removed_actor_path"] == CRATE
    assert len(editor.transactions) == 1


def test_a_dynamic_mesh_becomes_a_saved_game_static_mesh(editor: FakeEditor) -> None:
    """static_mesh used to be a no-op that returned a note (B7)."""
    editor.objects[BODY].dynamic_mesh_component.materials = ["M_Red"]
    payload = _call(editor, "convert_unreal_mesh_format", mesh_path=BODY, target_format="static_mesh")
    asset = editor.assets["/Game/Generated/SM_Body"]
    assert payload["result_path"] == asset.path == "/Game/Generated/SM_Body.SM_Body"
    assert payload["asset"] == {
        "asset_path": asset.path, "overwritten": False,
        "saved_file": "/Projects/Mesh/Content/Generated/SM_Body.uasset",
        "collision": "convex_hull", "collision_shape_count": 1, "material_slots": 1, "nanite": False,
        "triangle_count": 12, "vertex_count": 8, "fallback_triangle_count": None, "fallback_vertex_count": None,
    }
    assert asset.materials == {0: "M_Red"}
    assert editor.saved == ["/Game/Generated/SM_Body"]
    assert "empty Xform" in payload["hint"]
    assert editor.objects[BODY].valid  # left in the level without replace_actor


def test_the_component_collision_goes_with_the_mesh(editor: FakeEditor) -> None:
    _call(editor, "compute_unreal_convex_hull", mesh_path=BODY, output="collision")
    payload = _call(editor, "convert_unreal_mesh_format", mesh_path=BODY, target_format="static_mesh",
                    asset_path="/Game/Props/SM_Box", nanite=True, save=False)
    asset = editor.assets["/Game/Props/SM_Box"]
    assert payload["asset"]["collision"] == "component" and asset.collision is editor.objects[
        BODY].dynamic_mesh_component.collision
    assert payload["asset"]["nanite"] is True and asset.nanite_settings.enabled is True
    assert payload["asset"]["saved_file"] is None and editor.saved == []


def test_a_nanite_asset_reports_its_full_mesh_and_its_fallback(editor: FakeEditor) -> None:
    """The reply gave the fallback's 316 triangles of a 1,920-triangle sphere as the asset's (#342 review)."""
    payload = _call(editor, "convert_unreal_mesh_format", mesh_path=BALL, target_format="static_mesh", nanite=True)
    asset = payload["asset"]
    assert (asset["triangle_count"], asset["vertex_count"]) == (1920, 962)
    assert (asset["fallback_triangle_count"], asset["fallback_vertex_count"]) == (320, 160)


@pytest.mark.parametrize(("replace", "steps"), [(False, []), (True, ["simul: replace Body with a StaticMeshActor"])])
def test_writing_the_asset_is_no_undo_step(editor: FakeEditor, replace: bool, steps: Any) -> None:
    """Its collision was its own Undo step: one Undo dropped it and left the asset as written (#342 review).

    Unreal records no undo step for a new asset, so its collision is written the same way; with
    replace_actor, the swap of the actors is the one Undo step.
    """
    _call(editor, "convert_unreal_mesh_format", mesh_path=BODY, target_format="static_mesh", replace_actor=replace)
    assert editor.transactions == steps
    assert [e for e in editor.events if e[0] == "set_static_collision"] == [
        ("set_static_collision", "/Game/Generated/SM_Body.SM_Body", 1, False)]


def test_an_overwrite_is_no_undo_step_either(editor: FakeEditor) -> None:
    _call(editor, "convert_unreal_mesh_format", mesh_path=BODY, target_format="static_mesh",
          asset_path=CRATE_ASSET, overwrite=True)
    assert editor.arguments["copy_mesh_to_static_mesh"][2].emit_transaction is False
    assert editor.transactions == []


@pytest.mark.parametrize(
    ("label", "asset"),
    [("상자 Box", "SM_Box"), ("!!!", "SM_Mesh"), ("Crate.v2 (final)", "SM_Crate_v2_final"), ("A__b", "SM_A_b")],
)
def test_the_default_asset_name_is_ascii(editor: FakeEditor, label: str, asset: str) -> None:
    """A Korean label made /Game/Generated/SM_상자_Box, which an explicit asset_path would refuse (#342 review)."""
    editor.objects[BODY].label = label
    payload = _call(editor, "convert_unreal_mesh_format", mesh_path=BODY, target_format="static_mesh")
    assert payload["result_path"] == f"/Game/Generated/{asset}.{asset}"


def test_an_existing_static_mesh_is_kept_unless_overwrite(editor: FakeEditor) -> None:
    refused = _call(editor, "convert_unreal_mesh_format", mesh_path=BODY, target_format="static_mesh",
                    asset_path=CRATE_ASSET)
    assert refused["details"]["code"] == "ASSET_EXISTS"
    assert ("copy_mesh_to_static_mesh", True) not in editor.calls and editor.saved == []
    replaced = _call(editor, "convert_unreal_mesh_format", mesh_path=BODY, target_format="static_mesh",
                     asset_path=CRATE_ASSET, overwrite=True, collision="none")
    assert replaced["asset"]["overwritten"] is True and replaced["asset"]["collision_shape_count"] == 0
    assert editor.arguments["copy_mesh_to_static_mesh"][1] is editor.assets[CRATE_ASSET]
    assert ("create_new_static_mesh_asset_from_mesh", True) not in editor.calls


def test_replace_actor_swaps_the_dynamic_mesh_actor_for_a_static_mesh_actor(editor: FakeEditor) -> None:
    editor.objects[BODY].transform = transform((5, 6, 7), yaw=30)
    payload = _call(editor, "convert_unreal_mesh_format", mesh_path=BODY, target_format="static_mesh",
                    replace_actor=True)
    placed = editor.objects[payload["placed_actor_path"]]
    assert placed.cls_name == "StaticMeshActor" and placed.label == "Body"
    assert placed.static_mesh_component.static_mesh is editor.assets["/Game/Generated/SM_Body"]
    assert placed.get_actor_transform() is editor.objects[BODY].transform
    assert editor.objects[BODY].valid is False and payload["removed_actor_path"] == BODY
    assert payload["hint"] is None


@pytest.mark.parametrize(("overrides", "error_type", "code"), [
    ({"asset_path": "/Engine/BasicShapes/SM_Mine"}, "SandboxError", "OUTSIDE_GAME_CONTENT"),
    ({"asset_path": "/MyPlugin/Meshes/SM_Mine"}, "SandboxError", "OUTSIDE_GAME_CONTENT"),
    ({"asset_path": "/Game/../Engine/SM_Mine"}, "ValidationError", "INVALID_ASSET_PATH"),
    ({"asset_path": "/Game/My Meshes/SM_Mine"}, "ValidationError", "INVALID_ASSET_PATH"),
    ({"asset_path": "/Game/Meshes/SM_A.SM_B"}, "ValidationError", "INVALID_ASSET_PATH"),
    ({"location": [1, 2, 3]}, "ValidationError", "OPTION_NOT_APPLICABLE"),
    ({"collision": "sphere"}, "ValidationError", "UNKNOWN_COLLISION"),
])
def test_static_mesh_destinations_are_sandboxed(editor: FakeEditor, overrides: Dict[str, Any], error_type: str,
                                                code: str) -> None:
    payload = _call(editor, "convert_unreal_mesh_format", mesh_path=BODY, target_format="static_mesh", **overrides)
    assert (payload["error_type"], payload["details"]["code"]) == (error_type, code)
    assert editor.calls == [] and editor.saved == []


@pytest.mark.parametrize("overrides", [{"asset_path": "/Game/X/SM_X"}, {"overwrite": True}, {"save": False},
                                       {"nanite": True}, {"collision": "none"}])
def test_asset_options_are_refused_for_a_dynamic_mesh(editor: FakeEditor, overrides: Dict[str, Any]) -> None:
    payload = _call(editor, "convert_unreal_mesh_format", **overrides)
    assert payload["details"]["code"] == "OPTION_NOT_APPLICABLE"


def test_a_path_holding_another_asset_kind_is_refused(editor: FakeEditor) -> None:
    editor.assets["/Game/Maps/Level"] = SimpleNamespace(get_path_name=lambda: "/Game/Maps/Level.Level")
    payload = _call(editor, "convert_unreal_mesh_format", mesh_path=BODY, target_format="static_mesh",
                    asset_path="/Game/Maps/Level", overwrite=True)
    assert payload["details"]["code"] == "ASSET_PATH_TAKEN"


def test_assets_are_not_written_during_pie(editor: FakeEditor) -> None:
    editor.pie = True
    payload = _call(editor, "convert_unreal_mesh_format", mesh_path=BODY, target_format="static_mesh")
    assert payload["error_type"] == "PIEActive" and editor.saved == []


# -- the time budget ----------------------------------------------------------------


@pytest.mark.parametrize("tool", MUTATING)
def test_a_result_past_the_time_budget_is_discarded(editor: FakeEditor, tool: str, monkeypatch) -> None:
    """A remesh that outlived its HTTP call committed 3.5M triangles after the TimeoutError (#337 review)."""
    monkeypatch.setattr(unreal_runtime, "MESH_SCRIPT_TIMEOUT_S", 5)
    session = _session(editor)
    session.timeout = 1
    method, kwargs, _ = TOOLS[tool]
    payload = asyncio.run(getattr(session, method)(**kwargs))
    assert (payload["success"], payload["error_type"]) == (False, "TimeoutError")
    assert payload["details"]["code"] == "MESH_TIME_BUDGET"
    assert payload["details"]["budget_s"] == 5 - mesh_scripts.MESH_COMMIT_MARGIN_S
    assert "nothing was changed" in payload["error"]
    assert editor.transactions == [] and editor.spawned == 0
    assert not [event for event in editor.events if event[0] in ("set_dynamic_mesh", "spawn")]


def test_a_result_past_this_hosts_deadline_is_discarded(editor: FakeEditor) -> None:
    """A script that waited in the editor's queue has used up the caller's wait before it starts."""
    args = {"mesh": BODY, "target_triangle_count": 6, "target_percentage": None, "max_error": None}
    script = mesh_scripts.mesh_script(mesh_scripts.SIMPLIFY, args, budget_s=100.0, deadline=time.time() - 1)
    editor.run(script)
    assert editor.transactions == []
    assert '"error_type": "TimeoutError"' in str(editor.run(script))


def test_the_deadline_uses_this_hosts_clock_only_for_a_local_editor(editor: FakeEditor) -> None:
    for host, local in (("localhost", True), ("127.0.0.1", True), ("::1", True), ("10.0.0.5", False),
                        ("editor.lan", False)):
        assert mesh_scripts.is_loopback(host) is local, host
    session = _session(editor)
    scripts = []
    execute = session._execute_python

    async def watch(code: str, mode: str = "ExecuteFile", timeout: Any = None) -> Dict[str, Any]:
        scripts.append(code)
        return await execute(code, mode, timeout)

    session._execute_python = watch  # type: ignore[method-assign]
    session.host = "10.0.0.5"
    asyncio.run(session.validate_mesh(BODY))
    session.host = "localhost"
    asyncio.run(session.validate_mesh(BODY))
    assert "\n_DEADLINE = None\n" in scripts[0] and "\n_DEADLINE = None\n" not in scripts[1]


@pytest.mark.parametrize(("host", "says"), [("localhost", "the level stays as it was"),
                                            ("10.0.0.5", "check the level before retrying")])
def test_a_timed_out_mesh_call_says_what_becomes_of_its_result(host: str, says: str) -> None:
    session = UnrealRuntimeSession()
    session.host = host

    async def slow(code: str, mode: str = "ExecuteFile", timeout: Any = None) -> Dict[str, Any]:
        raise UnrealTimeoutError("The editor did not finish the script within 300s", "TimeoutError",
                                 details={"route": "/remote/object/call", "timeout_s": timeout})

    session._execute_python = slow  # type: ignore[method-assign]
    with pytest.raises(UnrealTimeoutError) as raised:
        asyncio.run(session.remesh_mesh(BODY, target_edge_length=5.0))
    assert says in str(raised.value) and "discards" in str(raised.value)
    assert raised.value.details == {"route": "/remote/object/call", "timeout_s": 300, "code": "MESH_TIME_BUDGET",
                                    "budget_s": 300 - mesh_scripts.MESH_COMMIT_MARGIN_S}


# -- the result cap -----------------------------------------------------------------


def _huge(editor: FakeEditor, tris: int = 2_500_000) -> str:
    editor.dynamic_actor("Huge", tris=tris, verts=tris // 2)
    return LEVEL + "Huge"


@pytest.mark.parametrize(
    ("tool", "overrides"),
    [
        ("subdivide_unreal_mesh", {"mesh_path": BIG, "level": 2, "scheme": "uniform"}),
        ("subdivide_unreal_mesh", {"mesh_path": BIG, "level": 2, "scheme": "catmull_clark"}),
        ("remesh_unreal_mesh", {"mesh_path": BALL, "target_edge_length": 0.1}),
        ("compute_unreal_mesh_uv", {"mesh_path": "huge"}),
        ("edit_unreal_mesh_topology", {"mesh_path": "huge"}),
    ],
)
def test_a_result_predicted_over_the_cap_is_refused_before_anything_runs(
    editor: FakeEditor, tool: str, overrides: Dict[str, Any]
) -> None:
    """Subdivide level 4 and a fine remesh had no bound (#337 review)."""
    if overrides.get("mesh_path") == "huge":
        overrides = {**overrides, "mesh_path": _huge(editor)}
    payload = _call(editor, tool, **overrides)
    assert (payload["success"], payload["error_type"]) == (False, "ValidationError"), payload
    assert payload["details"]["code"] == "RESULT_TOO_LARGE"
    assert payload["details"]["predicted"] is True
    assert payload["details"]["triangle_count"] > mesh_scripts.MAX_RESULT_TRIANGLES
    assert editor.calls == [] and editor.transactions == []


def test_a_result_over_the_cap_is_refused_before_it_replaces_anything(editor: FakeEditor) -> None:
    huge = _huge(editor, mesh_scripts.MAX_RESULT_TRIANGLES)
    ball = editor.objects[BALL].dynamic_mesh_component.get_dynamic_mesh()
    payload = _call(editor, "apply_unreal_mesh_boolean", target_mesh_path=BALL, tool_mesh_path=huge)
    assert (payload["error_type"], payload["details"]["code"]) == ("ValidationError", "RESULT_TOO_LARGE")
    assert payload["details"]["predicted"] is False
    assert payload["details"]["triangle_count"] == mesh_scripts.MAX_RESULT_TRIANGLES + 1920
    assert editor.transactions == [] and editor.objects[BALL].dynamic_mesh_component.get_dynamic_mesh() is ball


def test_a_subdivision_under_the_cap_runs(editor: FakeEditor) -> None:
    payload = _call(editor, "subdivide_unreal_mesh", mesh_path=BIG, level=1, scheme="uniform")
    assert payload["result_triangle_count"] == 318_400 * 4


@pytest.mark.parametrize(
    ("tool", "overrides", "code"),
    [
        ("generate_unreal_mesh_primitive", {"segments": 2}, "INVALID_SEGMENTS"),
        ("generate_unreal_mesh_primitive", {"segments": 513}, "INVALID_SEGMENTS"),
        ("remesh_unreal_mesh", {"target_edge_length": None,
                                "target_triangle_count": mesh_scripts.MAX_REMESH_TARGET_TRIANGLES + 1},
         "INVALID_TARGET"),
        ("remesh_unreal_mesh", {"target_edge_length": None, "target_triangle_count": 3}, "INVALID_TARGET"),
    ],
)
def test_inputs_that_would_exceed_the_cap_are_refused_by_the_session(
    editor: FakeEditor, tool: str, overrides: Dict[str, Any], code: str
) -> None:
    payload = _call(editor, tool, **overrides)
    assert (payload["error_type"], payload["details"]["code"]) == ("ValidationError", code)
    assert editor.calls == [] and editor.events == []


def test_the_largest_remesh_target_is_predicted_under_the_cap() -> None:
    target = mesh_scripts.MAX_REMESH_TARGET_TRIANGLES
    assert target * mesh_scripts.REMESH_OVERSHOOT <= mesh_scripts.MAX_RESULT_TRIANGLES
    assert (target + 1) * mesh_scripts.REMESH_OVERSHOOT > mesh_scripts.MAX_RESULT_TRIANGLES


def test_the_largest_primitive_is_under_the_cap(editor: FakeEditor) -> None:
    payload = _call(editor, "generate_unreal_mesh_primitive", primitive_type="sphere",
                    segments=mesh_scripts.MAX_SEGMENTS)
    assert payload["triangle_count"] == 2 * 512 * 510 < mesh_scripts.MAX_RESULT_TRIANGLES


# -- vectors -------------------------------------------------------------------------


@pytest.mark.parametrize("value", [[0, 0, 50], "0,0,50", "[0, 0, 50]", " 0, 0 , 50 ", [0.0, 0, 50.0]])
def test_a_plane_takes_a_json_array_or_text(value: Any) -> None:
    """A JSON array failed with an untyped ValueError (#337 review)."""
    spec = SPECS["cut_unreal_mesh_plane"]
    request = request_model_for(spec)(mesh_path=BODY, plane_origin=value, plane_normal=[0, 0, 1])
    arguments = session_arguments(spec, request)
    assert arguments["plane_origin"] == [0.0, 0.0, 50.0] and arguments["plane_normal"] == [0.0, 0.0, 1.0]


@pytest.mark.parametrize("value", ["up", "0,0", "[0, 0]", [1, 2, 3, 4], "1,x,3", "nan,0,0", "", []])
def test_a_plane_that_is_not_three_numbers_is_refused_by_validation(value: Any) -> None:
    """The conversion fails as an InvalidArgument, which the envelope reports as a ValidationError."""
    spec = SPECS["cut_unreal_mesh_plane"]
    request = request_model_for(spec)(mesh_path=BODY, plane_origin=value, plane_normal=[0, 0, 1])
    with pytest.raises(InvalidArgument, match="^plane_origin: expected 3 numbers") as raised:
        session_arguments(spec, request)
    assert raised.value.details["errors"][0]["field"] == "plane_origin"


def test_a_location_is_optional() -> None:
    spec = SPECS["generate_unreal_mesh_primitive"]
    model = request_model_for(spec)
    assert session_arguments(spec, model(primitive_type="box"))["location"] is None
    assert session_arguments(spec, model(primitive_type="box", location=[1, 2, 3]))["location"] == [1.0, 2.0, 3.0]
    with pytest.raises(InvalidArgument, match="^location: "):
        session_arguments(spec, model(primitive_type="box", location="1,x,3"))
