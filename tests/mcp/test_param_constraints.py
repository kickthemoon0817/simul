"""Input schemas carry the bounds and choices the tools enforce.

0 of 529 parameters had numeric bounds and 3 had enums: ranges lived in prose
("1–3840", "1 to 5000") and choice lists in sentences, and many tools quietly
clamped an out-of-range value (``max_results = max(1, min(max_results,
1000))``). These tests hold the schema and the runtime together: the clamps in
the Isaac tool source, the runtime constants, the request models' own
constraints and the documented prose all have to appear in the advertised
schema, and the MCP layer refuses a value outside it.
"""

from __future__ import annotations

import ast
import asyncio
import importlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import annotated_types
import pytest
from fastmcp import Client

from simul.config import Settings
from simul.mcp import backends as backends_module
from simul.mcp import server as server_module
from simul.mcp.registration import _constraints
from simul.mcp.registration._constraints import PARAM_CONSTRAINTS, describe_constraints
from tests.fakes import AvailableAdapter

ISAAC_TOOLS = Path(__file__).resolve().parents[2] / "src" / "simul" / "mcp" / "tools" / "isaac"


@pytest.fixture(scope="module")
def schemas() -> Dict[str, Dict[str, Any]]:
    """Every tool's input schema, full surfaces, as a real client lists it."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(backends_module, "is_headless_available", lambda: True)
        monkeypatch.setattr(backends_module, "is_blender_available", lambda: True)
        monkeypatch.setattr(backends_module, "BlenderRuntimeAdapter", AvailableAdapter)
        monkeypatch.setattr(backends_module, "UnrealRuntimeAdapter", AvailableAdapter)
        base = Settings()
        instance = server_module.SimulMCPServer(
            base.model_copy(
                update={"unreal": base.unreal.model_copy(update={"tool_surface": "full"})}
            )
        )

    async def listing() -> Dict[str, Dict[str, Any]]:
        async with Client(instance.mcp) as client:
            return {tool.name: tool.input_schema for tool in await client.list_tools()}

    return asyncio.run(listing())


def _property(schemas: Dict[str, Dict[str, Any]], tool: str, param: str) -> Dict[str, Any]:
    return schemas[tool]["properties"][param]


# ---------------------------------------------------------------------------
# The table agrees with the tools' own numbers
# ---------------------------------------------------------------------------


def test_the_table_uses_the_isaac_runtime_constants() -> None:
    from simul.mcp.tools.isaac import _shared, ui

    assert _constraints.ISAAC_MAX_CAPTURE_DIMENSION == _shared.MAX_CAPTURE_DIMENSION
    window = PARAM_CONSTRAINTS["get_isaac_ui_window"]
    assert window["max_depth"] == _constraints.Bounds(ge=1, le=ui.MAX_UI_TREE_DEPTH)
    assert window["max_widgets"] == _constraints.Bounds(ge=1, le=ui.MAX_UI_TREE_WIDGETS)
    collision = PARAM_CONSTRAINTS["create_isaac_object"]["collision"]
    assert isinstance(collision, _constraints.Choices)
    assert collision.values == _shared.COLLISION_APPROXIMATIONS


def test_the_choice_lists_are_the_tools_own() -> None:
    """The choices the schema states are the ones the tool checks against."""
    from simul.mcp.tools.isaac import _shared, joints, physics

    def values(tool: str, param: str) -> Tuple[str, ...]:
        choice = PARAM_CONSTRAINTS[tool][param]
        assert isinstance(choice, _constraints.Choices)
        return choice.values

    assert values("add_isaac_collision", "approximation") == _shared.COLLISION_APPROXIMATIONS
    assert values("set_isaac_joint_drive", "drive_axis") == joints.DRIVE_AXES
    assert values("set_isaac_joint_drive", "drive_type") == joints.DRIVE_TYPES
    assert values("set_isaac_joint_limits", "limit_axis") == joints.LIMIT_AXES
    assert values("configure_isaac_physics_scene", "solver_type") == physics.PHYSX_SOLVER_TYPES
    assert values("configure_isaac_physics_scene", "broadphase_type") == physics.PHYSX_BROADPHASE_TYPES
    pytest.importorskip("aiohttp")
    from simul.adapters import unreal_runtime

    assert set(values("execute_unreal_script", "mode")) == unreal_runtime.UNREAL_EXEC_MODES
    assert values("set_unreal_actor_visibility", "scope") == unreal_runtime.VISIBILITY_SCOPES
    assert values("capture_unreal_viewport", "format") == unreal_runtime.CAPTURE_FORMATS
    assert values("capture_unreal_camera", "format") == unreal_runtime.CAPTURE_FORMATS
    assert values("capture_unreal_viewport", "view") == unreal_runtime.CAPTURE_VIEWS
    assert values("load_unreal_world_partition_actors", "action") == unreal_runtime.WORLD_PARTITION_ACTIONS


def test_choices_never_replace_a_literal_a_parameter_declares() -> None:
    """Narrowing a Literal from the table would refuse values the tool accepts."""
    from typing import Literal, Optional

    declared = Optional[Literal["a", "b", "c"]]
    assert _constraints.apply(declared, _constraints.Choices(("c", "b", "a"))) is declared
    with pytest.raises(ValueError, match="would replace the Literal"):
        _constraints.apply(declared, _constraints.Choices(("a", "b")))
    with pytest.raises(ValueError, match="would replace the Literal"):
        _constraints.apply(Literal["a"], _constraints.Choices(("a", "z"), enforced=False))


def test_the_table_uses_the_usd_runtime_constants() -> None:
    from simul.mcp.schemas import usd as usd_schemas
    from simul.usd import values

    assert _constraints.USD_MAX_ARRAY_PREVIEW == values.MAX_ARRAY_PREVIEW == usd_schemas.MAX_ARRAY_PREVIEW
    assert _constraints.USD_MAX_SEARCH_LIMIT == usd_schemas.MAX_SEARCH_LIMIT
    assert _constraints.USD_MAX_SUBTREE_LIMIT == usd_schemas.MAX_SUBTREE_LIMIT


def test_the_table_uses_the_unreal_runtime_constants() -> None:
    pytest.importorskip("aiohttp")
    from simul.adapters import unreal_runtime

    assert _constraints.UNREAL_MIN_FOV_DEGREES == unreal_runtime.FOCUS_MIN_FOV_DEGREES
    assert _constraints.UNREAL_MAX_ASPECT_RATIO == unreal_runtime.FOCUS_MAX_ASPECT_RATIO
    assert _constraints.UNREAL_MAX_CAMERA_CAPTURE_DIMENSION == unreal_runtime.CAMERA_CAPTURE_MAX_DIMENSION
    assert _constraints.UNREAL_MAX_INLINE_SIZE == unreal_runtime.INLINE_MAX_SIZE
    assert _constraints.UNREAL_MAX_WORLD_PARTITION_ACTORS == unreal_runtime.WORLD_PARTITION_MAX_ACTORS
    assert _constraints.UNREAL_FORCE_MAX_DURATION_S == unreal_runtime.FORCE_MAX_DURATION_S
    assert _constraints.UNREAL_PHYSICS_STATE_MAX_RESULTS == unreal_runtime.PHYSICS_STATE_MAX_RESULTS
    assert _constraints.UNREAL_MAX_PAGE == unreal_runtime.SIMULATION_ACTORS_MAX_RESULTS
    assert _constraints.UNREAL_ROBOTICS_MAX_RESULTS == unreal_runtime.ROBOTICS_MAX_RESULTS
    assert _constraints.UNREAL_LIGHT_MAX_INTENSITY == unreal_runtime.LIGHT_MAX_INTENSITY
    assert _constraints.UNREAL_LIGHT_MIN_TEMPERATURE_K == unreal_runtime.LIGHT_MIN_TEMPERATURE_K
    assert _constraints.UNREAL_LIGHT_MAX_TEMPERATURE_K == unreal_runtime.LIGHT_MAX_TEMPERATURE_K


def _isaac_clamps() -> Iterator[Tuple[str, str, int, Any]]:
    """Every ``p = max(lo, min(p, hi))`` / ``p = max(lo, p)`` a decorated Isaac tool applies."""
    for path in sorted(ISAAC_TOOLS.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # A named bound (MAX_CAPTURE_DIMENSION) is read from the module itself.
        constants = vars(importlib.import_module(f"simul.mcp.tools.isaac.{path.stem}"))
        for cls in (node for node in tree.body if isinstance(node, ast.ClassDef)):
            for method in (n for n in cls.body if isinstance(n, ast.AsyncFunctionDef)):
                meta = [
                    d for d in method.decorator_list
                    if isinstance(d, ast.Call) and getattr(d.func, "id", "") == "tool_meta"
                ]
                if not meta:
                    continue
                tool = next(k.value.value for k in meta[0].keywords if k.arg == "name")  # type: ignore[attr-defined]
                params = {arg.arg for arg in method.args.args}
                for node in ast.walk(method):
                    if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
                        continue
                    target = node.targets[0]
                    if getattr(node.value.func, "id", "") != "max" or not isinstance(target, ast.Name):
                        continue
                    if target.id not in params:
                        continue
                    low = ast.literal_eval(node.value.args[0])
                    inner = node.value.args[1]
                    high = None
                    if isinstance(inner, ast.Call) and getattr(inner.func, "id", "") == "min":
                        bound = inner.args[1]
                        high = constants[bound.id] if isinstance(bound, ast.Name) else ast.literal_eval(bound)
                    yield tool, target.id, low, high


def test_every_isaac_clamp_is_in_the_schema(schemas: Dict[str, Dict[str, Any]]) -> None:
    """A value the tool would clamp is refused by the schema instead."""
    clamps = list(_isaac_clamps())
    assert len(clamps) > 25
    # The innermost clamp is the effective one where a tool applies two.
    effective: Dict[Tuple[str, str], Tuple[int, Any]] = {}
    for tool, param, low, high in clamps:
        effective[(tool, param)] = (low, high)
    mismatched = []
    for (tool, param), (low, high) in sorted(effective.items()):
        found = describe_constraints(_property(schemas, tool, param))
        if found.get("minimum") != low or found.get("maximum") != high:
            mismatched.append((tool, param, (low, high), found))
    assert mismatched == []


# ---------------------------------------------------------------------------
# Request models: what they enforce, the schema states
# ---------------------------------------------------------------------------


def test_blender_request_model_bounds_reach_the_schema(schemas: Dict[str, Dict[str, Any]]) -> None:
    from simul.mcp.registration._reg_blender import BLENDER_TOOLS

    checked = 0
    for spec in BLENDER_TOOLS:
        if spec.request is None or spec.name not in schemas:
            continue
        properties = schemas[spec.name]["properties"]
        for name, field in spec.request.model_fields.items():
            if name not in properties:
                continue
            for meta in field.metadata:
                found = describe_constraints(properties[name])
                if isinstance(meta, annotated_types.Ge):
                    assert found.get("minimum") == meta.ge, (spec.name, name)
                    checked += 1
                if isinstance(meta, annotated_types.Le):
                    assert found.get("maximum") == meta.le, (spec.name, name)
                    checked += 1
    assert checked >= 10


def test_unreal_request_model_choices_reach_the_schema(schemas: Dict[str, Dict[str, Any]]) -> None:
    control = _property(schemas, "control_unreal_ui", "agent_control")
    assert "move_cursor" in control["enum"]


# ---------------------------------------------------------------------------
# Prose: a documented range or choice list is machine-readable too
# ---------------------------------------------------------------------------

_RANGE_PROSE = re.compile(
    r"\(\s*-?\d+(?:\.\d+)?\s*[–-]\s*-?\d+(?:\.\d+)?\s*\)"  # (1–3840), (0-1)
    r"|\b\d+(?:\.\d+)?\s+to\s+\d+(?:\.\d+)?\b"  # 1 to 5000
    r"|[Cc]lamped to \[?\d+"  # Clamped to [1, 1000]
    r"|\b0-1\b"  # 0-1
)
# A choice list: "One of", two quoted names joined by "or", or three or more
# names (quoted or not, each maybe with a parenthetical) in a comma list ending
# in "or": 'a' or 'b'; "TGS" (temporal Gauss-Seidel) or "PGS"; union,
# subtract, or intersect; transX, ..., rotZ or distance.
_CHOICE_ITEM = r"[\"'`]?[\w./-]+[\"'`]?(?: \([^)]*\))?"
_CHOICE_PROSE = re.compile(
    r"\bOne of\b"
    r"|(?P<q>['\"`])[\w./-]+(?P=q)(?: \([^)]*\))? or (?P=q)[\w./-]+(?P=q)"
    rf"|\b{_CHOICE_ITEM}(?:, {_CHOICE_ITEM})+,? or [\"'`]?[\w./-]+"
)

# Ranges or choices that are documented but deliberately not enforced: list
# items (colors may be HDR, above 1; a resolution caps width and height
# differently), a comma-separated list of checks; and prose that reads like a
# choice list but is not one (file extensions, what a query matches).
_PROSE_ONLY = {
    ("create_isaac_light", "color"),
    ("create_isaac_material", "diffuse_color"),
    ("create_isaac_object", "diffuse_color"),
    ("read_isaac_aovs", "resolution"),
    ("validate_simready_asset", "checks"),
    ("export_blender_file", "file_path"),
    ("export_simready_usd", "file_path"),
    ("search_prims", "query"),
    ("set_isaac_material_property", "property_name"),
    ("set_blender_object_mode", "mode"),
    ("assign_blender_material", "object_name"),
    ("import_unreal_asset", "file_path"),
}


def _numeric(schema: Dict[str, Any]) -> bool:
    nodes: List[Dict[str, Any]] = [schema, *schema.get("anyOf", [])]
    return any(node.get("type") in ("integer", "number") for node in nodes)


def test_documented_ranges_and_choices_are_in_the_schema(schemas: Dict[str, Dict[str, Any]]) -> None:
    missing = []
    for tool, schema in sorted(schemas.items()):
        for param, prop in schema.get("properties", {}).items():
            if (tool, param) in _PROSE_ONLY:
                continue
            text = prop.get("description", "") or ""
            found = describe_constraints(prop)
            if _numeric(prop) and _RANGE_PROSE.search(text) and not (
                {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"} & set(found)
            ):
                missing.append(f"range {tool}.{param}: {text[:80]}")
            if not _numeric(prop) and _CHOICE_PROSE.search(text) and "enum" not in found:
                missing.append(f"choice {tool}.{param}: {text[:80]}")
    assert missing == []


def test_the_schemas_carry_bounds_and_choices(schemas: Dict[str, Dict[str, Any]]) -> None:
    bounded = enums = 0
    for schema in schemas.values():
        for prop in schema.get("properties", {}).values():
            found = describe_constraints(prop)
            bounded += bool({"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"} & set(found))
            enums += "enum" in found
    assert bounded >= 50 and enums >= 15


# ---------------------------------------------------------------------------
# The MCP layer refuses a value outside the schema with the shared envelope
# ---------------------------------------------------------------------------


def _isaac_server(tmp_path: Path) -> server_module.SimulMCPServer:
    instance = server_module.SimulMCPServer(
        Settings(isaac_sim={"discovery_dir": str(tmp_path / "disc")}), backends={"isaac"}
    )
    return instance


@pytest.mark.parametrize(
    ("tool", "arguments", "field"),
    [
        ("capture_isaac_viewport", {"width": 0}, "width"),
        ("capture_isaac_viewport", {"width": 5000}, "width"),
        ("list_isaac_prims", {"max_results": 1001}, "max_results"),
        ("get_isaac_ui_window", {"window_title": "Stage", "max_depth": 100}, "max_depth"),
        ("create_isaac_light", {"prim_path": "/World/L", "light_type": "SpotLight"}, "light_type"),
        ("step_isaac_simulation", {"num_steps": 0}, "num_steps"),
    ],
)
def test_out_of_range_values_are_refused_not_clamped(
    tmp_path: Path, tool: str, arguments: Dict[str, Any], field: str
) -> None:
    instance = _isaac_server(tmp_path)

    async def run() -> Any:
        async with Client(instance.mcp) as client:
            return await client.call_tool(tool, arguments, raise_on_error=False)

    result = asyncio.run(run())

    payload = json.loads(result.content[-1].text)
    assert result.is_error is True
    assert payload["error_type"] == "ValidationError"
    assert [error["field"] for error in payload["details"]["errors"]] == [field]


def test_unreal_capture_dimensions_are_a_validation_error(tmp_path: Path) -> None:
    """resolution_x=-5 used to come back as a ValueError from the editor session."""
    pytest.importorskip("aiohttp")
    instance = server_module.SimulMCPServer(
        Settings(unreal={"host": "127.0.0.1", "port": 39310, "max_retries": 0}), backends={"unreal"}
    )

    async def run() -> Any:
        async with Client(instance.mcp) as client:
            return await client.call_tool(
                "capture_unreal_viewport", {"resolution_x": -5}, raise_on_error=False
            )

    payload = json.loads(asyncio.run(run()).content[-1].text)
    assert payload["error_type"] == "ValidationError"
    assert payload["details"]["errors"][0]["field"] == "resolution_x"


@pytest.mark.parametrize(
    ("tool", "arguments", "field"),
    [
        ("capture_unreal_viewport", {"format": "gif"}, "format"),
        ("capture_unreal_viewport", {"view": "pane"}, "view"),
        ("capture_unreal_viewport", {"jpeg_quality": 0}, "jpeg_quality"),
        ("capture_unreal_viewport", {"inline_max_size": 20000}, "inline_max_size"),
        ("capture_unreal_camera", {"actor_path": "/Game/M.M:L.C", "resolution_x": 9000}, "resolution_x"),
        ("load_unreal_world_partition_actors", {"max_actors": 10001}, "max_actors"),
        ("load_unreal_world_partition_actors", {"action": "reload"}, "action"),
    ],
)
def test_unreal_capture_and_world_partition_limits_are_refused_by_the_schema(
    tool: str, arguments: Dict[str, Any], field: str
) -> None:
    """The ranges and choices these tools document are refused before the editor is reached."""
    pytest.importorskip("aiohttp")
    instance = server_module.SimulMCPServer(
        Settings(unreal={"host": "127.0.0.1", "port": 39310, "max_retries": 0, "tool_surface": "full"}),
        backends={"unreal"},
    )

    async def run() -> Any:
        async with Client(instance.mcp) as client:
            return await client.call_tool(tool, arguments, raise_on_error=False)

    payload = json.loads(asyncio.run(run()).content[-1].text)
    assert payload["error_type"] == "ValidationError"
    assert [error["field"] for error in payload["details"]["errors"]] == [field]
