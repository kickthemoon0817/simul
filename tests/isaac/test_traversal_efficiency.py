"""Whole-stage traversals bound their work, not just their output.

``find_isaac_prims_in_area`` computed a world bound for every prim on the
stage and sorted the lot before slicing; ``get_isaac_texture_dependencies``
resolved every shader input with no cap; ``get_isaac_stage_info`` and
``get_isaac_runtime_info`` counted every prim on every call even though the
bridge already knew how to skip it. Each traversal now prunes, rejects early,
or runs only when asked.
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_path))
extension_root = src_path / "simul_mcp" / "bridge_ext" / "khemoo.simul.mcp"
sys.path.insert(0, str(extension_root))

from khemoo.simul.mcp.protocol import BridgeRequest  # noqa: E402
from khemoo.simul.mcp.service import BridgeCommandService  # noqa: E402
from simul_mcp.adapters.isaac_socket_client import ScriptResult  # noqa: E402
from simul_mcp.config import Settings  # noqa: E402
from simul_mcp.mcp.tools.isaac_tools import IsaacTools  # noqa: E402


def _cube(stage: Usd.Stage, path: str, position: Tuple[float, float, float]) -> UsdGeom.Cube:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.AddTranslateOp().Set(Gf.Vec3d(*position))
    cube.GetExtentAttr().Set([(-1, -1, -1), (1, 1, 1)])
    return cube


@pytest.fixture
def area_stage() -> Usd.Stage:
    """Boundables at known distances, some nested under container Xforms."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    _cube(stage, "/World/Near", (1, 0, 0))
    _cube(stage, "/World/Far", (100, 0, 0))
    UsdGeom.Xform.Define(stage, "/World/Group")
    _cube(stage, "/World/Group/Deep", (2, 0, 0))
    UsdGeom.Xform.Define(stage, "/World/Group/Deeper")
    _cube(stage, "/World/Group/Deeper/Leaf", (0, 1, 0))
    return stage


def _find(tools: IsaacTools, captured: List[str], run_on_stage: Any, stage: Usd.Stage, **kwargs: Any) -> Dict[str, Any]:
    kwargs.setdefault("center", [0.0, 0.0, 0.0])
    kwargs.setdefault("radius", 5.0)
    asyncio.run(tools.find_isaac_prims_in_area(**kwargs))
    return run_on_stage(captured[-1], stage)


# ---------------------------------------------------------------------------
# find_isaac_prims_in_area
# ---------------------------------------------------------------------------


def test_area_search_returns_boundables_sorted_by_distance(
    capturing_tools: Tuple[IsaacTools, List[str]], run_on_stage: Any, area_stage: Usd.Stage
) -> None:
    tools, captured = capturing_tools

    result = _find(tools, captured, run_on_stage, area_stage)

    assert [m["path"] for m in result["matches"]] == [
        "/World/Near",
        "/World/Group/Deeper/Leaf",
        "/World/Group/Deep",
    ]
    assert [m["distance"] for m in result["matches"]] == [1.0, 1.0, 2.0]
    assert result["truncated"] is False
    # Containers have no geometry of their own; their bound is a sub-traversal.
    assert "/World/Group" not in {m["path"] for m in result["matches"]}


def test_containers_are_candidates_only_when_prim_type_names_them(
    capturing_tools: Tuple[IsaacTools, List[str]], run_on_stage: Any, area_stage: Usd.Stage
) -> None:
    tools, captured = capturing_tools

    result = _find(tools, captured, run_on_stage, area_stage, prim_type="Xform")

    assert {m["path"] for m in result["matches"]} == {"/World/Group", "/World/Group/Deeper"}


def test_max_depth_prunes_deeper_subtrees(
    capturing_tools: Tuple[IsaacTools, List[str]], run_on_stage: Any, area_stage: Usd.Stage
) -> None:
    tools, captured = capturing_tools

    # Depth counts path components below root_path: /World is 1, /World/Near 2.
    result = _find(tools, captured, run_on_stage, area_stage, max_depth=2)

    assert [m["path"] for m in result["matches"]] == ["/World/Near"]
    assert result["max_depth"] == 2
    assert "PruneChildren" in captured[-1], "depth must bound the traversal, not just the output"


def test_search_stops_after_max_results(
    capturing_tools: Tuple[IsaacTools, List[str]], run_on_stage: Any, area_stage: Usd.Stage
) -> None:
    tools, captured = capturing_tools

    result = _find(tools, captured, run_on_stage, area_stage, max_results=2)

    assert result["count"] == 2
    assert result["truncated"] is True


def test_local_extent_rejects_prims_before_world_bounds(
    capturing_tools: Tuple[IsaacTools, List[str]],
    run_on_stage: Any,
    area_stage: Usd.Stage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prims whose extent cannot reach the search sphere never hit BBoxCache."""
    tools, captured = capturing_tools
    bounded: List[str] = []
    original = UsdGeom.BBoxCache.ComputeWorldBound

    def _counting(self: Any, prim: Any) -> Any:
        bounded.append(str(prim.GetPath()))
        return original(self, prim)

    monkeypatch.setattr(UsdGeom.BBoxCache, "ComputeWorldBound", _counting)

    result = _find(tools, captured, run_on_stage, area_stage)

    assert result["count"] == 3
    assert "/World/Far" not in bounded, "a prim 100 units away must be rejected by its extent"
    assert set(bounded) == {"/World/Near", "/World/Group/Deep", "/World/Group/Deeper/Leaf"}


def test_extent_rejection_is_conservative_under_scale(
    capturing_tools: Tuple[IsaacTools, List[str]], run_on_stage: Any
) -> None:
    """A scaled-up prim whose bound reaches the sphere must not be pre-rejected."""
    tools, captured = capturing_tools
    stage = Usd.Stage.CreateInMemory()
    big = _cube(stage, "/Big", (8, 0, 0))
    big.AddScaleOp().Set(Gf.Vec3f(10, 10, 10))

    result = _find(tools, captured, run_on_stage, stage, radius=8.5)

    assert [m["path"] for m in result["matches"]] == ["/Big"]


# ---------------------------------------------------------------------------
# get_isaac_texture_dependencies
# ---------------------------------------------------------------------------


@pytest.fixture
def textured_stage() -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    for index in range(5):
        material = UsdShade.Material.Define(stage, f"/World/Looks/M{index}")
        shader = UsdShade.Shader.Define(stage, f"/World/Looks/M{index}/Shader")
        shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(f"/tex/t{index}.png"))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
        material.CreateSurfaceOutput().ConnectToSource(
            shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        )
    return stage


def test_texture_dependencies_stop_at_max_results(
    capturing_tools: Tuple[IsaacTools, List[str]], run_on_stage: Any, textured_stage: Usd.Stage
) -> None:
    tools, captured = capturing_tools

    asyncio.run(tools.get_isaac_texture_dependencies(max_results=2))
    capped = run_on_stage(captured[-1], textured_stage)
    asyncio.run(tools.get_isaac_texture_dependencies())
    full = run_on_stage(captured[-1], textured_stage)

    assert capped["unique_textures"] == 2
    assert capped["truncated"] is True
    assert full["unique_textures"] == 5
    assert full["truncated"] is False
    assert full["textures"][0]["referenced_by"] == [{"material": "/World/Looks/M0", "input": "file"}]
    # Only asset-typed inputs are read: reading one resolves the asset.
    assert "Sdf.ValueTypeNames.Asset" in captured[-1]


# ---------------------------------------------------------------------------
# get_isaac_stage_info / get_isaac_runtime_info prim counts
# ---------------------------------------------------------------------------


def _bridged_tools(payload: Dict[str, Any]) -> IsaacTools:
    client = MagicMock()
    client.address = "127.0.0.1:8226"
    client.timeout_seconds = 30.0
    client.bridge_enabled = True
    client.fallback_to_vscode = True
    client.bridge_request = AsyncMock(return_value={"status": "ok", "payload": payload})
    client.execute = AsyncMock(return_value=ScriptResult(success=True, output="{}"))
    client.execute_vscode_only = client.execute
    return IsaacTools(client, settings=Settings())


def test_stage_info_forwards_the_prim_count_flag_to_the_bridge() -> None:
    tools = _bridged_tools({"up_axis": "Z"})

    asyncio.run(tools.get_isaac_stage_info())
    asyncio.run(tools.get_isaac_stage_info(include_prim_count=True))

    calls = [call.args for call in tools._client.bridge_request.await_args_list]
    assert calls == [
        ("get_stage_info", {"include_prim_count": False}),
        ("get_stage_info", {"include_prim_count": True}),
    ]


def test_runtime_info_forwards_the_prim_count_flag_to_the_bridge() -> None:
    tools = _bridged_tools({"app": {}})

    asyncio.run(tools.get_runtime_info())
    asyncio.run(tools.get_runtime_info(include_prim_count=True))

    calls = [call.args for call in tools._client.bridge_request.await_args_list]
    assert calls == [
        ("get_runtime_info", {"include_prim_count": False}),
        ("get_runtime_info", {"include_prim_count": True}),
    ]


def test_stage_info_script_counts_prims_only_on_request(
    capturing_tools: Tuple[IsaacTools, List[str]], run_on_stage: Any, area_stage: Usd.Stage
) -> None:
    tools, captured = capturing_tools

    asyncio.run(tools.get_isaac_stage_info())
    skipped = run_on_stage(captured[-1], area_stage)
    asyncio.run(tools.get_isaac_stage_info(include_prim_count=True))
    counted = run_on_stage(captured[-1], area_stage)

    assert skipped["total_prims"] is None
    assert skipped["root_prims"] == ["/World"]
    assert counted["total_prims"] == 7


def test_runtime_info_script_counts_lazily_and_only_on_request(
    capturing_tools: Tuple[IsaacTools, List[str]],
) -> None:
    tools, captured = capturing_tools

    asyncio.run(tools.get_runtime_info())
    off = captured[-1]
    asyncio.run(tools.get_runtime_info(include_prim_count=True))
    on = captured[-1]

    for script in (off, on):
        assert "len(list(stage.Traverse()))" not in script
        assert "sum(1 for _ in stage.Traverse())" in script
    assert "if False else None" in off
    assert "if True else None" in on


def test_bridge_runtime_info_honours_the_flag(
    area_stage: Usd.Stage, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = types.SimpleNamespace(get_stage=lambda: area_stage, get_stage_url=lambda: "anon:memory.usda")
    fake_usd = types.ModuleType("omni.usd")
    fake_usd.get_context = lambda: context  # type: ignore[attr-defined]
    fake_omni = types.ModuleType("omni")
    fake_omni.usd = fake_usd  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "omni", fake_omni)
    monkeypatch.setitem(sys.modules, "omni.usd", fake_usd)
    service = BridgeCommandService(executor=MagicMock(), allow_unsafe_execution=False)

    def _runtime(payload: Dict[str, Any]) -> Dict[str, Any]:
        request = BridgeRequest(request_id="r", action="get_runtime_info", payload=payload)
        return asyncio.run(service.dispatch(request)).payload

    assert _runtime({"include_prim_count": False})["stage"]["prim_count"] is None
    assert _runtime({"include_prim_count": True})["stage"]["prim_count"] == 7
    # Clients that predate the flag keep the count.
    assert _runtime({})["stage"]["prim_count"] == 7
