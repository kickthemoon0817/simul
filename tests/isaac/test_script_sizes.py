"""Every Isaac tool's generated scripts stay well inside Isaac Sim 5.1's limit.

Isaac Sim 5.1's stock socket (isaacsim.code_editor.vscode) accepts at most
32 KiB of script (``MAX_VSCODE_SCRIPT_BYTES``), less the auth-token header
line when a token is configured. create_isaac_object embedded every helper on
every call and reached 31.8-32.4 KB, crossing the cap with long prim paths;
reparent_isaac_prim sat at 29 KB. Each tool is called here with typical
arguments and with worst-case ones (prim paths of ~500 characters, the
longest lists the tool accepts), and every script it sends must fit a
24 KiB budget, which leaves room for the paths and lists a caller can pass.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from simul.adapters.isaac_socket_client import MAX_VSCODE_SCRIPT_BYTES, ScriptResult
from simul.config import Settings
from simul.mcp.tools._meta import iter_tool_methods
from simul.mcp.tools.isaac_tools import PRIM_DETAIL_ASPECTS, IsaacTools

SCRIPT_BUDGET = 24 * 1024

LONG = (
    "/World/envs/env_4095/Robot_With_A_Long_Descriptive_Name/panda_link7/visuals/"
    + "x" * 400
)
TYP = "/World/Robot/link1"

TYPICAL: Dict[str, Dict[str, Any]] = {
    "get_isaac_prim_detail": {"prim_path": TYP, "aspects": ["info"]},
    "execute_isaac_script": {"code": "print('hi')"},
    "create_isaac_object": {
        "prim_path": TYP,
        "prim_type": "Cube",
        "translation": [0, 0, 1],
        "rotation_euler": [0, 0, 90],
        "scale": [1, 1, 1],
        "rigid_body": True,
        "collision": "convexHull",
        "mass": 1.0,
        "diffuse_color": [1, 0, 0],
    },
    "create_isaac_prim": {
        "prim_path": TYP,
        "prim_type": "Cube",
        "attributes": {"size": 2.0},
    },
    "delete_isaac_prim": {"prim_path": TYP},
    "set_isaac_prim_transform": {
        "prim_path": TYP,
        "translation": [1, 2, 3],
        "rotation_euler": [0, 0, 90],
    },
    "set_isaac_prim_visibility": {"prim_path": TYP, "visible": False},
    "set_isaac_prim_attribute": {
        "prim_path": TYP,
        "attribute_name": "size",
        "value": 2.0,
    },
    "duplicate_isaac_prim": {"prim_path": TYP, "new_path": TYP + "_copy"},
    "reparent_isaac_prim": {"prim_path": TYP, "new_parent_path": "/World/Other"},
    "set_isaac_camera": {"position": [1, 2, 3], "target": [0, 0, 0]},
    "capture_isaac_viewport": {"inline": True},
    "focus_isaac_viewport": {"prim_path": TYP},
    "set_isaac_physics_material": {"prim_path": TYP, "static_friction": 0.8},
    "configure_isaac_physics_scene": {
        "time_steps_per_second": 120,
        "solver_type": "TGS",
    },
    "add_isaac_rigid_body": {"prim_path": TYP},
    "add_isaac_collision": {"prim_path": TYP, "approximation": "convexHull"},
    "set_isaac_mass_properties": {"prim_path": TYP, "mass": 2.0},
    "set_isaac_joint_drive": {"joint_path": TYP + "/joint", "stiffness": 100.0},
    "set_isaac_joint_limits": {
        "joint_path": TYP + "/joint",
        "lower": -90.0,
        "upper": 90.0,
    },
    "step_isaac_simulation": {"num_steps": 10},
    "create_isaac_material": {"material_path": "/World/Looks/M"},
    "assign_isaac_material": {"prim_path": TYP, "material_path": "/World/Looks/M"},
    "set_isaac_material_property": {
        "material_path": "/World/Looks/M",
        "property_name": "diffuseColor",
        "value": [1, 0, 0],
    },
    "read_isaac_aovs": {"aov_names": ["rgb"]},
    "query_isaac_typed_prims": {"type_name": "UsdGeom.Mesh", "attributes": ["extent"]},
    "get_isaac_carb_settings": {"keys": ["/rtx/fog/enabled"]},
    "set_isaac_carb_settings": {"settings": {"/rtx/fog/enabled": True}},
    "get_isaac_graph_nodes": {"graph_path": "/World/ActionGraph"},
    "create_isaac_graph_node": {
        "graph_path": "/World/ActionGraph",
        "node_path": "/World/ActionGraph/n",
        "node_type": "omni.graph.action.OnTick",
    },
    "delete_isaac_graph_node": {
        "graph_path": "/World/ActionGraph",
        "node_path": "/World/ActionGraph/n",
    },
    "connect_isaac_graph_nodes": {
        "source_attr_path": "/World/ActionGraph/a.outputs:tick",
        "target_attr_path": "/World/ActionGraph/b.inputs:execIn",
    },
    "search_isaac_prims": {"query": "Mesh"},
    "set_isaac_graph_node_values": {
        "node_path": "/World/ActionGraph/n",
        "values": {"inputs:x": 1},
    },
    "open_isaac_stage": {"file_path": "/tmp/a.usd"},
    "import_isaac_asset": {"asset_path": "/tmp/robot.usd"},
    "add_isaac_reference": {"prim_path": TYP, "reference_path": "/tmp/robot.usd"},
    "raycast_isaac_scene": {"origin": [0, 0, 1], "direction": [0, 0, -1]},
    "find_isaac_prims_in_area": {"center": [0, 0, 0], "radius": 1.0},
    "enable_isaac_extension": {"extension_id": "omni.physx"},
    "disable_isaac_extension": {"extension_id": "omni.isaac.foo"},
    "get_isaac_ui_window": {"window_title": "Stage"},
    "set_isaac_log_level": {"level": "warn"},
    "create_isaac_light": {"prim_path": "/World/Light"},
}


def _worst(name: str, typical: Dict[str, Any]) -> Dict[str, Any]:
    args = dict(typical)
    for key, value in list(args.items()):
        if isinstance(value, str) and value.startswith("/World") and key != "file_path":
            args[key] = value.replace(TYP, LONG)
    if name == "create_isaac_object":
        args["material_path"] = LONG + "/Looks/Mat"
    if name == "create_isaac_prim":
        args["attributes"] = {"size": 2.0, "extent": [[-1, -1, -1], [1, 1, 1]]}
    if name == "query_isaac_typed_prims":
        args["attributes"] = ["attr_%02d" % i for i in range(32)]
    if name == "read_isaac_aovs":
        args["aov_names"] = ["distance_to_image_plane_%02d" % i for i in range(16)]
    if name == "get_isaac_carb_settings":
        args["keys"] = ["/rtx/some/long/setting/key/%03d" % i for i in range(200)]
    if name == "set_isaac_carb_settings":
        args["settings"] = {
            "/rtx/some/long/setting/key/%03d" % i: 1.0 for i in range(200)
        }
    if name == "set_isaac_graph_node_values":
        args["values"] = {"inputs:value_%03d" % i: [1.0, 2.0, 3.0] for i in range(100)}
    return args


def _tools(sent: List[str]) -> IsaacTools:
    def _respond(code: str, **_kwargs: Any) -> ScriptResult:
        sent.append(code)
        if (
            "dirty_layers" in code
            and "stage_url" in code
            and "save_stage_async" not in code
        ):
            payload = {
                "stage_url": "/tmp/x.usd",
                "dirty_layers": [],
                "timeline_stopped": True,
            }
            return ScriptResult(success=True, output=json.dumps(payload))
        return ScriptResult(success=True, output=json.dumps({"ok": True}))

    client = MagicMock()
    client.address = "127.0.0.1:8226"
    client.timeout_seconds = 30.0
    client.script_timeout_seconds = 29.0
    client.bridge_enabled = False
    client.fallback_to_vscode = True
    client.execute = AsyncMock(side_effect=_respond)
    client.execute_vscode_only = AsyncMock(side_effect=_respond)
    client.execute_bridge_script_only = AsyncMock(side_effect=_respond)
    client.bridge_request = AsyncMock(return_value=None)
    base = Settings()
    settings = base.model_copy(
        update={"security": base.security.model_copy(update={"sandbox_enabled": False})}
    )
    return IsaacTools(client, settings=settings)


def _calls() -> List[Any]:
    calls = []
    for method_name, _function, meta in iter_tool_methods(IsaacTools):
        typical = TYPICAL.get(meta.name, {})
        calls.append(pytest.param(method_name, typical, id=meta.name + "-typical"))
        calls.append(
            pytest.param(
                method_name, _worst(meta.name, typical), id=meta.name + "-worst"
            )
        )
    # Aspect readers take the prim path positionally (the material reader
    # names it material_path); the script fallback runs one per aspect.
    for aspect, method_name in PRIM_DETAIL_ASPECTS.items():
        calls.append(
            pytest.param(method_name, {"*": TYP}, id="aspect:" + aspect + "-typical")
        )
        calls.append(
            pytest.param(method_name, {"*": LONG}, id="aspect:" + aspect + "-worst")
        )
    return calls


def test_the_budget_leaves_headroom_under_the_51_cap() -> None:
    assert SCRIPT_BUDGET <= MAX_VSCODE_SCRIPT_BYTES - 8 * 1024


@pytest.mark.parametrize(("method_name", "kwargs"), _calls())
def test_every_script_fits_the_budget(method_name: str, kwargs: Dict[str, Any]) -> None:
    sent: List[str] = []
    method = getattr(_tools(sent), method_name)

    if "*" in kwargs:
        asyncio.run(method(kwargs["*"]))
    else:
        asyncio.run(method(**kwargs))

    sizes = [len(script.encode("utf-8")) for script in sent]
    assert all(
        size <= SCRIPT_BUDGET for size in sizes
    ), f"{method_name} sent scripts of {sizes} bytes; the budget is {SCRIPT_BUDGET}"
