"""List-shaped results are capped inside the generated script.

The MCP-side result budget trimmed payloads after Kit had serialised and
shipped every entry: 4.3 MB across the wire to keep 40 KB. Each list tool now
takes ``max_results``, applies it before ``json.dumps`` runs in Kit, and
reports ``total`` and ``truncated`` so the caller knows what was left out.
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
from pxr import Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_path))

from simul_mcp.mcp.tools.isaac_tools import IsaacTools  # noqa: E402


@pytest.fixture
def busy_stage() -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    for index in range(5):
        UsdShade.Material.Define(stage, f"/World/Looks/M{index}")
    for index in range(3):
        UsdLux.SphereLight.Define(stage, f"/World/Light{index}")
        UsdGeom.Camera.Define(stage, f"/World/Camera{index}")
    for index in range(4):
        prim = UsdGeom.Cube.Define(stage, f"/World/Body{index}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(UsdGeom.Cube.Define(stage, "/World/Wall").GetPrim())
    return stage


def _run(tools: IsaacTools, captured: List[str], run_on_stage: Any, stage: Usd.Stage, coro: Any, **run_kwargs: Any) -> Dict[str, Any]:
    asyncio.run(coro)
    return run_on_stage(captured[-1], stage, **run_kwargs)


def test_materials_cap_the_entries_and_count_the_rest(
    capturing_tools: Tuple[IsaacTools, List[str]], run_on_stage: Any, busy_stage: Usd.Stage
) -> None:
    tools, captured = capturing_tools

    capped = _run(tools, captured, run_on_stage, busy_stage, tools.list_isaac_materials(max_results=2))
    full = _run(tools, captured, run_on_stage, busy_stage, tools.list_isaac_materials())

    assert capped["count"] == 2 and len(capped["materials"]) == 2
    assert capped["total"] == 5 and capped["truncated"] is True
    assert full["count"] == 5 and full["truncated"] is False


def test_lights_cap_the_entries_and_count_the_rest(
    capturing_tools: Tuple[IsaacTools, List[str]], run_on_stage: Any, busy_stage: Usd.Stage
) -> None:
    tools, captured = capturing_tools

    capped = _run(tools, captured, run_on_stage, busy_stage, tools.list_isaac_lights(max_results=2))

    assert len(capped["lights"]) == 2
    assert capped["total"] == 3 and capped["truncated"] is True


def test_cameras_cap_the_entries_and_count_the_rest(
    capturing_tools: Tuple[IsaacTools, List[str]], run_on_stage: Any, busy_stage: Usd.Stage
) -> None:
    tools, captured = capturing_tools

    capped = _run(tools, captured, run_on_stage, busy_stage, tools.list_isaac_cameras(max_results=2))

    assert len(capped["cameras"]) == 2
    assert capped["total"] == 3 and capped["truncated"] is True


def test_physics_objects_cap_each_list_and_keep_full_counts(
    capturing_tools: Tuple[IsaacTools, List[str]], run_on_stage: Any, busy_stage: Usd.Stage
) -> None:
    tools, captured = capturing_tools

    capped = _run(tools, captured, run_on_stage, busy_stage, tools.list_isaac_physics_objects(max_results=2))

    assert capped["rigid_body_count"] == 4
    assert len(capped["rigid_bodies"]) == 2
    assert capped["collider_count"] == 1 and len(capped["colliders"]) == 1
    assert capped["truncated"] is True
    assert "[:200]" not in captured[-1], "the cap is the parameter, not a slice after the fact"


def test_selection_caps_the_entries_and_counts_the_rest(
    capturing_tools: Tuple[IsaacTools, List[str]], run_on_stage: Any, busy_stage: Usd.Stage
) -> None:
    tools, captured = capturing_tools
    selected = ["/World/Body0", "/World/Body1", "/World/Body2"]

    capped = _run(
        tools, captured, run_on_stage, busy_stage, tools.get_isaac_selection(max_results=2), selected_paths=selected
    )

    assert capped["count"] == 2 and capped["total"] == 3 and capped["truncated"] is True


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("get_isaac_graph_nodes", {"graph_path": "/World/Graph", "max_results": 7}),
        ("list_isaac_graph_node_types", {"max_results": 7}),
        ("list_aovs", {"max_results": 7}),
        ("list_render_vars", {"max_results": 7}),
    ],
)
def test_kit_only_list_tools_embed_the_cap_and_report_truncation(
    capturing_tools: Tuple[IsaacTools, List[str]], method: str, kwargs: Dict[str, Any]
) -> None:
    """These need omni.graph / replicator to run, so the script itself is checked."""
    tools, captured = capturing_tools

    asyncio.run(getattr(tools, method)(**kwargs))

    script = captured[-1]
    assert re.search(r"max_results = 7\b", script)
    assert "truncated" in script
    assert "[:max_results]" in script


@pytest.mark.parametrize(
    "method",
    [
        "list_isaac_materials",
        "list_isaac_lights",
        "list_isaac_cameras",
        "list_isaac_physics_objects",
        "get_isaac_selection",
        "get_isaac_texture_dependencies",
        "list_aovs",
        "list_render_vars",
    ],
)
def test_caps_are_clamped_before_reaching_the_script(
    capturing_tools: Tuple[IsaacTools, List[str]], method: str
) -> None:
    tools, captured = capturing_tools

    asyncio.run(getattr(tools, method)(max_results=999999))
    asyncio.run(getattr(tools, method)(max_results=0))

    assert "999999" not in captured[-2]
    assert re.search(r"max_results = 10000\b", captured[-2])
    assert re.search(r"max_results = 1\b", captured[-1])
