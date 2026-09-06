"""Every tool parameter an agent can see must say what it takes.

Units, axis conventions and defaults live in the tools-layer docstrings;
``with_param_descriptions`` copies them into the schema FastMCP advertises.
These tests read that schema through the real FastMCP instance and a real
MCP client, so a wrapper that forgets the decorator, or a docstring that
drops a parameter, fails here rather than in an agent's guess.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastmcp import Client

src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_path))

from simul_mcp.config import Settings  # noqa: E402
from simul_mcp.mcp import backends as backends_module  # noqa: E402
from simul_mcp.mcp import server as server_module  # noqa: E402
from simul_mcp.mcp.registration._helpers import (  # noqa: E402
    describe_params,
    resolve_deprecated_alias,
)

# Parameters whose description must name the unit or convention.
UNIT_BEARING: Dict[str, Dict[str, str]] = {
    "set_isaac_prim_transform": {
        "rotation_euler": "degrees",
        "translation": "stage units",
    },
    "create_isaac_light": {"intensity": "intensity", "angle": "degrees"},
    "raycast_isaac_scene": {"max_distance": "stage units"},
    "create_isaac_physics_scene": {
        "gravity_magnitude": "m/s^2",
        "gravity_direction": "Z-up",
    },
    "set_isaac_mass_properties": {"mass": "kg", "density": "kg/m^3"},
}


def _list_tools(settings: Settings, backends: set[str]) -> List[Any]:
    """Return the tool listing a real MCP client sees for one server."""
    instance = server_module.SimulMCPServer(settings, backends=backends)

    async def _run() -> List[Any]:
        async with Client(instance.mcp) as client:
            return await client.list_tools()

    return asyncio.run(_run())


@pytest.fixture(scope="module")
def listed_tools() -> List[Any]:
    """Every tool on every backend, as advertised over the wire.

    Blender is imported but not marked available on a machine without the
    runtime; the registration only needs the adapter class, so the availability
    probe is patched for the listing.
    """
    settings = Settings().model_copy(
        update={"unreal": Settings().unreal.model_copy(update={"tool_surface": "full"})}
    )
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(backends_module, "is_blender_available", lambda: True)
        patcher.setattr(
            backends_module.BlenderRuntimeAdapter, "is_available", lambda self: True
        )
        return _list_tools(settings, {"isaac", "usd", "unreal", "blender"})


def _property_descriptions(tool: Any) -> Dict[str, str]:
    return {
        name: str(prop.get("description", ""))
        for name, prop in tool.inputSchema.get("properties", {}).items()
    }


def test_listing_covers_every_backend(listed_tools: List[Any]) -> None:
    names = {tool.name for tool in listed_tools}
    assert {
        "list_isaac_prims",
        "load_usd_file",
        "spawn_unreal_actor",
        "get_blender_info",
    } <= names
    assert len(names) > 90


def test_every_tool_parameter_has_a_description(listed_tools: List[Any]) -> None:
    missing = {
        tool.name: [
            name
            for name, text in _property_descriptions(tool).items()
            if not text.strip()
        ]
        for tool in listed_tools
    }
    missing = {name: params for name, params in missing.items() if params}
    assert not missing, f"tool parameters without a description: {missing}"


def test_unit_bearing_parameters_name_their_units(listed_tools: List[Any]) -> None:
    by_name = {tool.name: _property_descriptions(tool) for tool in listed_tools}
    for tool_name, expectations in UNIT_BEARING.items():
        for param, needle in expectations.items():
            assert (
                needle.lower() in by_name[tool_name][param].lower()
            ), f"{tool_name}.{param} should mention {needle!r}: {by_name[tool_name][param]!r}"


def test_deprecated_aliases_point_at_the_new_name(listed_tools: List[Any]) -> None:
    by_name = {tool.name: _property_descriptions(tool) for tool in listed_tools}
    for tool_name, alias in (
        ("list_isaac_prims", "max_items"),
        ("get_isaac_subtree", "max_prims"),
        ("query_isaac_typed_prims", "max_prims"),
    ):
        assert "max_results" in by_name[tool_name], tool_name
        assert "deprecated" in by_name[tool_name][alias].lower(), tool_name
        assert "max_results" in by_name[tool_name][alias], tool_name


def test_search_query_is_required(listed_tools: List[Any]) -> None:
    search = next(tool for tool in listed_tools if tool.name == "search_isaac_prims")
    assert "query" in search.inputSchema.get("required", [])


def test_listing_tools_accept_offset(listed_tools: List[Any]) -> None:
    by_name = {tool.name: _property_descriptions(tool) for tool in listed_tools}
    for tool_name in (
        "list_isaac_prims",
        "search_isaac_prims",
        "query_isaac_typed_prims",
        "get_isaac_subtree",
        "list_isaac_extensions",
        "list_isaac_materials",
        "list_isaac_lights",
        "list_isaac_cameras",
        "list_isaac_physics_objects",
    ):
        assert "offset" in by_name[tool_name], tool_name
    assert "limit" in by_name["list_isaac_extensions"]


def test_instructions_state_the_axis_convention() -> None:
    assert "Z-up" in server_module._MCP_INSTRUCTIONS


# ---------------------------------------------------------------------------
# describe_params: the docstring parser the decorator relies on
# ---------------------------------------------------------------------------


def _documented(alpha: int, beta_gamma: str = "x", delta: float = 1.0) -> None:
    """
    Summary line.

    Longer prose that mentions Args: in passing.

    Args:
        alpha: First value in metres.
        beta_gamma (str): Second value, whose description
            continues on an indented line.
        delta: Third value.

    Returns:
        Nothing.
    """


def test_describe_params_reads_google_args_sections() -> None:
    assert describe_params(_documented) == {
        "alpha": "First value in metres.",
        "beta_gamma": "Second value, whose description continues on an indented line.",
        "delta": "Third value.",
    }


def test_describe_params_without_args_section_is_empty() -> None:
    def _bare(value: int) -> None:
        """Only a summary."""

    assert describe_params(_bare) == {}


def test_resolve_deprecated_alias_prefers_the_new_name() -> None:
    assert resolve_deprecated_alias(100, None, 100) == 100
    assert resolve_deprecated_alias(100, 25, 100) == 25
    assert resolve_deprecated_alias(40, 25, 100) == 40
