"""The script guard refuses code-running nodes and nothing else Isaac Sim ships.

The first guard matched name fragments, so "code" hit inputs:usdTimecode on
ReadPrimAttribute and nine other node types, "script" hit
omni.isaac.robot_description_editor, and "warp" refused fourteen omni.warp
nodes that run fixed kernels. Names are now split into words at separators
and camelCase boundaries. These tests run the rule over every node type and
extension in an Isaac Sim install (tests/isaac/data/shipped_node_types.json,
extracted from its .ogn files and extension directories), so a change that
admits a code node or refuses an ordinary one fails here.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Set

import pytest

from simul.mcp.tools.isaac.graph import (
    SCRIPT_CAPABLE_NODE_TYPES,
    is_script_attribute,
    is_script_capable_node_type,
    is_script_extension,
    script_guard_source,
)
from tests.isaac.test_materials_import_save_graph import _recording_tools, _settings

SHIPPED = json.loads((Path(__file__).parent / "data" / "shipped_node_types.json").read_text())
NODE_TYPES: Dict[str, List[str]] = SHIPPED["node_types"]
EXTENSIONS: List[str] = SHIPPED["extensions"]

# What runs agent-supplied Python among the shipped nodes, read from their
# .ogn definitions: these five take code (or a path to code) as an input.
CODE_NODE_TYPES: Set[str] = {
    "omni.graph.scriptnode.ScriptNode",
    "omni.warp.WarpKernel",
    "omni.replicator.core.OgnAugment",
    "omni.replicator.core.OgnAugmentCPU",
    "omni.replicator.core.OgnConditionScript",
}
CODE_INPUTS: Set[str] = {
    "script", "scriptPath", "codeStr", "codeFile", "codeProvider", "augmentationScript", "conditionScript",
}


def test_the_fixture_is_the_whole_install() -> None:
    assert len(NODE_TYPES) > 500 and len(EXTENSIONS) > 500
    assert CODE_NODE_TYPES <= set(NODE_TYPES)


def test_exactly_the_code_running_node_types_are_refused() -> None:
    refused = {name for name in NODE_TYPES if is_script_capable_node_type(name)}

    assert refused == CODE_NODE_TYPES
    assert {name.lower() for name in SCRIPT_CAPABLE_NODE_TYPES} == {name.lower() for name in CODE_NODE_TYPES}


def test_exactly_the_code_inputs_are_code_attributes() -> None:
    every_input = {name for inputs in NODE_TYPES.values() for name in inputs}

    flagged = {name for name in every_input if is_script_attribute("inputs:" + name)}

    assert flagged == CODE_INPUTS


def test_every_node_with_a_code_input_is_refused() -> None:
    """Completeness: a node that takes code is a code node, whatever it is called."""
    with_code = {name for name, inputs in NODE_TYPES.items() if any(is_script_attribute(i) for i in inputs)}

    assert with_code == CODE_NODE_TYPES
    assert all(is_script_capable_node_type(name) for name in with_code)


def test_timecode_inputs_stay_settable() -> None:
    timecode_nodes = sorted(name for name, inputs in NODE_TYPES.items() if "usdTimecode" in inputs)

    assert "omni.graph.nodes.ReadPrimAttribute" in timecode_nodes and len(timecode_nodes) >= 10
    assert not is_script_attribute("inputs:usdTimecode")
    assert not any(is_script_capable_node_type(name) for name in timecode_nodes)


def test_only_warp_kernel_is_refused_among_the_warp_nodes() -> None:
    warp_nodes = sorted(name for name in NODE_TYPES if name.startswith("omni.warp."))

    assert len(warp_nodes) >= 14
    assert [name for name in warp_nodes if is_script_capable_node_type(name)] == ["omni.warp.WarpKernel"]


def test_exactly_the_script_extensions_are_refused() -> None:
    refused = {name for name in EXTENSIONS if is_script_extension(name)}

    assert refused == {"omni.graph.scriptnode", "omni.kit.scripting", "omni.kit.window.script_editor", "omni.warp"}


@pytest.mark.parametrize(
    "extension_id",
    [
        "omni.isaac.robot_description_editor",
        "omni.isaac.ros2_bridge.robot_description",
        "isaacsim.robot_setup.xrdf_editor",
        "omni.usd.schema.omniscripting",
        "omni.warp.core",
        "omni.isaac.robot_description_editor-1.0.0",
    ],
)
def test_extensions_that_only_contain_the_letters_are_allowed(extension_id: str) -> None:
    assert is_script_extension(extension_id) is False


def test_kit_side_rule_matches_the_server_side_rule_on_every_shipped_type() -> None:
    namespace: Dict[str, Any] = {}
    exec(compile(script_guard_source(), "<script-guard>", "exec"), namespace)  # noqa: S102 - our own guard source
    kit_rule = namespace["_script_capable_type"]

    mismatches = [name for name in NODE_TYPES if kit_rule(name) != is_script_capable_node_type(name)]

    assert mismatches == []


def test_ordinary_inputs_and_extensions_reach_isaac_while_scripts_are_off() -> None:
    tools, captured = _recording_tools(_settings(scripts=False))

    asyncio.run(tools.set_isaac_graph_node_values("/World/Graph/read", {"inputs:usdTimecode": 12.0}))
    asyncio.run(tools.connect_isaac_graph_nodes(
        "/World/Graph/time.outputs:value", "/World/Graph/read.inputs:usdTimecode"
    ))
    asyncio.run(tools.create_isaac_graph_node("/World/Graph", "/World/Graph/cloth", "omni.warp.WarpClothSimulate"))
    asyncio.run(tools.enable_isaac_extension("omni.isaac.robot_description_editor"))

    assert len(captured) == 4, "each call must be sent, not refused"


def test_the_code_routes_stay_refused() -> None:
    tools, captured = _recording_tools(_settings(scripts=False))

    results = [
        asyncio.run(tools.set_isaac_graph_node_values("/World/Graph/k", {"inputs:codeStr": "import os"})),
        asyncio.run(tools.connect_isaac_graph_nodes("/World/G/c.outputs:value", "/World/G/s.inputs:script")),
        asyncio.run(tools.create_isaac_graph_node("/World/Graph", "/World/Graph/k", "omni.warp.WarpKernel")),
        asyncio.run(tools.enable_isaac_extension("omni.warp")),
    ]

    assert [result["error_type"] for result in results] == ["ScriptExecutionDisabled"] * 4
    assert captured == []
