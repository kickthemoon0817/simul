"""Snapshots of the Blender tools as an MCP client meets them.

Two files under ``tests/blender/snapshots`` pin the Blender tool surface, so a
change to how the tools are declared or registered cannot alter it unnoticed:

- ``blender_tools_list.json``: ``tools/list`` through a real FastMCP server and
  client (name, description, inputSchema, annotations and ``_meta``, in listing
  order) for the full and the thin surface in attached and embedded mode, and
  for the full surface with script execution switched off.
- ``blender_tool_calls.json``: what each tool does with its arguments, called
  through the same real client against ``StrictBlenderSession``, which offers
  only the ``BlenderRuntimeSession`` operations and records every call. For
  a minimal and a full set of arguments: the ``BlenderRuntimeSession`` method
  the tool calls and the arguments that method receives (bound to its
  signature with defaults applied, each value as ``repr``), the response model
  the result is checked against, and the usage record the call leaves. For
  invalid input: the error payload the caller receives. A call the strict
  session refuses (a missing operation, arguments it does not take) fails
  the suite outright rather than being snapshotted as an error payload.

Both are compared as text, byte for byte. After an intended change run the
module with ``SIMUL_UPDATE_SNAPSHOTS=1`` to rewrite them, and review the diff.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest
from fastmcp import Client

from simul.adapters import blender_runtime
from simul.adapters.blender_connection import BlenderAttachments
from simul.config import Settings
from simul.mcp import backends as backends_module
from simul.mcp import server as server_module
from simul.mcp.usage_tracker import ToolUsageTracker
from simul.tool_surfaces import THIN_BLENDER_TOOLS
from tests.fakes import (
    BlenderSessionAdapter,
    FakeFastMCP,
    SessionCall,
    StrictBlenderSession,
)

SNAPSHOTS = Path(__file__).resolve().parent / "snapshots"
LISTING_SNAPSHOT = SNAPSHOTS / "blender_tools_list.json"
CALLS_SNAPSHOT = SNAPSHOTS / "blender_tool_calls.json"

# What every recorded session call returns. ``success`` must be a bool on every
# Blender response model, so the response check always fails and names the
# model it checked against, whatever fields that model requires.
PROBE_PAYLOAD: Dict[str, Any] = {"success": "probe"}
SESSION_AGENT = "<session agent id>"
SNAPSHOT_AGENT = "snapshot-agent"
INVALID_VALUE: Dict[str, Any] = {"bogus": 1}
_PYDANTIC_URL = re.compile(r"https://errors\.pydantic\.dev/[0-9.]+/")
_MODEL_CHECK = re.compile(r"^\d+ validation errors? for (\w+)\n")

# The listing configurations: (mode, surface, script execution allowed).
SURFACES: Dict[str, Tuple[str, str, bool]] = {
    "attached/full": ("attached", "full", True),
    "attached/thin": ("attached", "thin", True),
    "embedded/full": ("embedded", "full", True),
    "embedded/thin": ("embedded", "thin", True),
    "attached/full/scripts-disabled": ("attached", "full", False),
}

# Tool name -> case -> arguments, as a client sends them. "minimal" passes the
# required parameters only; "full" passes every parameter, away from its
# default where it has one.
CALLS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "control_blender_ui": {
        "minimal": {"agent_control": "inspect"},
        "full": {
            "agent_control": "set_property",
            "target": "location",
            "area_id": "area-1",
            "position": [0.25, 0.75],
            "value": [1, 2, 3],
            "agent_id": SNAPSHOT_AGENT,
        },
    },
    "get_blender_info": {"minimal": {}},
    "attach_blender_window": {
        "minimal": {},
        "full": {"instance_id": "instance-1", "window_id": "window-1"},
    },
    "list_blender_scene_objects": {
        "minimal": {},
        "full": {"collection_name": "Props", "include_hidden": True, "max_items": 7},
    },
    "get_blender_object_info": {"minimal": {"object_name": "Cube"}},
    "get_blender_mesh_info": {"minimal": {"object_name": "Cube"}},
    "get_blender_bounding_box": {
        "minimal": {"object_name": "Cube"},
        "full": {"object_name": "Cube", "world_space": False},
    },
    "search_blender_objects": {
        "minimal": {},
        "full": {"name_pattern": "Cu*", "object_type": "MESH", "max_results": 3},
    },
    "summarize_blender_scene": {"minimal": {}},
    "get_blender_material_info": {"minimal": {"material_name": "Metal"}},
    "get_blender_distance_between": {
        "minimal": {"object_name_a": "Cube", "object_name_b": "Sphere"}
    },
    "check_blender_object_bounds": {
        "minimal": {
            "object_name": "Cube",
            "bounds_min": [-1, -2, -3],
            "bounds_max": [1, 2, 3],
        }
    },
    "capture_blender_viewport": {
        "minimal": {},
        "full": {
            "width": 640,
            "height": 480,
            "jpeg_quality": 70,
            "method": "auto",
            "use_render_fallback": True,
            "agent_id": SNAPSHOT_AGENT,
        },
    },
    "set_blender_camera_view": {
        "minimal": {"location": [1, 2, 3], "rotation_euler": [0.1, 0.2, 0.3]},
        "full": {
            "location": [1, 2, 3],
            "rotation_euler": [0.1, 0.2, 0.3],
            "camera_name": "Camera",
        },
    },
    "get_blender_camera_info": {"minimal": {}, "full": {"camera_name": "Camera"}},
    "focus_blender_on_object": {
        "minimal": {"object_name": "Cube"},
        "full": {"object_name": "Cube", "distance_factor": 3, "camera_name": "Camera"},
    },
    "get_blender_viewport_info": {"minimal": {}},
    "capture_blender_viewport_sequence": {
        "minimal": {"start_frame": 1, "end_frame": 3},
        "full": {
            "start_frame": 1,
            "end_frame": 9,
            "step": 2,
            "width": 128,
            "height": 96,
            "jpeg_quality": 60,
            "agent_id": SNAPSHOT_AGENT,
            "method": "render",
        },
    },
    "create_blender_object": {
        "minimal": {"object_type": "CUBE"},
        "full": {
            "object_type": "SPHERE",
            "name": "Ball",
            "location": [1, 2, 3],
            "rotation_euler": [0.1, 0.2, 0.3],
            "scale": [2, 2, 2],
        },
    },
    "delete_blender_object": {"minimal": {"object_name": "Cube"}},
    "set_blender_object_transform": {
        "minimal": {"object_name": "Cube"},
        "full": {
            "object_name": "Cube",
            "location": [1, 2, 3],
            "rotation_euler": [0.1, 0.2, 0.3],
            "scale": [2, 2, 2],
        },
    },
    "set_blender_object_parent": {
        "minimal": {"child_name": "Wheel", "parent_name": "Car"}
    },
    "clear_blender_object_parent": {
        "minimal": {"object_name": "Wheel"},
        "full": {"object_name": "Wheel", "keep_transform": False},
    },
    "assign_blender_material": {
        "minimal": {"object_name": "Cube"},
        "full": {
            "object_name": "Cube",
            "material_name": "Red",
            "base_color": [1, 0, 0, 1],
            "metallic": 0.5,
            "roughness": 0.25,
        },
    },
    "add_blender_modifier": {
        "minimal": {"object_name": "Cube", "modifier_type": "SUBSURF"},
        "full": {
            "object_name": "Cube",
            "modifier_type": "SUBSURF",
            "modifier_name": "Smooth",
            "params": {"levels": 2},
        },
    },
    "set_blender_light_params": {
        "minimal": {"light_name": "Light"},
        "full": {
            "light_name": "Light",
            "energy": 500,
            "color": [1, 0.5, 0.25],
            "use_shadow": False,
            "spot_size": 0.8,
            "spot_blend": 0.2,
            "shadow_soft_size": 0.1,
        },
    },
    "open_blender_file": {"minimal": {"file_path": "/tmp/scene.blend"}},
    "save_blender_file": {"minimal": {}, "full": {"file_path": "/tmp/out.blend"}},
    "import_blender_file": {
        "minimal": {"file_path": "/tmp/model.obj", "file_format": "OBJ"}
    },
    "export_blender_file": {
        "minimal": {"file_path": "/tmp/out.glb", "file_format": "GLTF"},
        "full": {
            "file_path": "/tmp/out.glb",
            "file_format": "GLTF",
            "selected_only": True,
        },
    },
    "get_blender_file_info": {"minimal": {}},
    "set_blender_frame": {"minimal": {"frame": 12}},
    "get_blender_frame": {"minimal": {}},
    "set_blender_frame_range": {"minimal": {"frame_start": 1, "frame_end": 250}},
    "play_blender_animation": {"minimal": {}, "full": {"action": "reverse"}},
    "insert_blender_keyframe": {
        "minimal": {"object_name": "Cube", "data_path": "location", "frame": 10},
        "full": {
            "object_name": "Cube",
            "data_path": "location",
            "frame": 10,
            "index": 2,
        },
    },
    "delete_blender_keyframe": {
        "minimal": {"object_name": "Cube", "data_path": "location", "frame": 10},
        "full": {
            "object_name": "Cube",
            "data_path": "location",
            "frame": 10,
            "index": 2,
        },
    },
    "get_blender_keyframes": {"minimal": {"object_name": "Cube"}},
    "setup_blender_rigid_body": {
        "minimal": {"object_name": "Cube"},
        "full": {
            "object_name": "Cube",
            "body_type": "PASSIVE",
            "mass": 2,
            "friction": 0.7,
            "restitution": 0.3,
            "collision_shape": "BOX",
            "linear_damping": 0.1,
            "angular_damping": 0.2,
        },
    },
    "add_blender_force_field": {
        "minimal": {"field_type": "WIND"},
        "full": {
            "field_type": "VORTEX",
            "strength": 3,
            "location": [1, 2, 3],
            "name": "Swirl",
        },
    },
    "get_blender_force_field_info": {"minimal": {"object_name": "Swirl"}},
    "add_blender_rigid_body_constraint": {
        "minimal": {
            "constraint_type": "HINGE",
            "object1_name": "A",
            "object2_name": "B",
        },
        "full": {
            "constraint_type": "HINGE",
            "object1_name": "A",
            "object2_name": "B",
            "location": [0, 0, 1],
            "disable_collisions": False,
        },
    },
    "get_blender_constraint_info": {"minimal": {"object_name": "Hinge"}},
    "get_blender_physics_state": {"minimal": {"object_name": "Cube"}},
    "get_blender_object_trajectory": {
        "minimal": {"object_name": "Cube", "start_frame": 1, "end_frame": 10},
        "full": {"object_name": "Cube", "start_frame": 1, "end_frame": 10, "step": 3},
    },
    "bake_blender_simulation": {"minimal": {"frame_start": 1, "frame_end": 50}},
    "free_blender_bake": {"minimal": {}},
    "execute_blender_script": {
        "minimal": {"script": "print(1)"},
        "full": {"script": "print(1)", "timeout": 5, "max_output_chars": 100},
    },
    "create_blender_mesh_from_data": {
        "minimal": {"name": "Tri", "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]]},
        "full": {
            "name": "Tri",
            "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            "edges": [[0, 1]],
            "faces": [[0, 1, 2]],
            "location": [1, 2, 3],
            "collection_name": "Props",
        },
    },
    "apply_simready_metadata": {
        "minimal": {
            "object_name": "Cube",
            "metadata": {
                "semantic": {"semantic_class": "cup"},
                "physics": {"mass_kg": 2, "collider_type": "box"},
                "material": {"substrate_type": "metal"},
            },
        }
    },
    "get_simready_metadata": {"minimal": {"object_name": "Cube"}},
    "validate_simready_compliance": {
        "minimal": {},
        "full": {
            "object_names": ["Cube"],
            "check_naming": False,
            "check_scale": False,
            "check_transforms": False,
            "check_materials": False,
            "check_hierarchy": False,
        },
    },
    "export_simready_usd": {
        "minimal": {"file_path": "/tmp/asset.usd"},
        "full": {
            "file_path": "/tmp/asset.usd",
            "object_names": ["Cube"],
            "embed_metadata": False,
            "validate_before_export": False,
        },
    },
    "setup_simready_hierarchy": {
        "minimal": {"root_name": "asset_root", "child_names": ["Cube"]},
        "full": {
            "root_name": "asset_root",
            "child_names": ["Cube"],
            "semantic": {"semantic_class": "cup", "semantic_qcode": "Q81727"},
        },
    },
}


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _settings(
    tmp_path: Path, mode: str, surface: str = "full", scripts: bool = True
) -> Settings:
    """Settings for one configuration, explicit wherever the listing depends on them."""
    return Settings(
        blender={
            "enabled": True,
            "mode": mode,
            "tool_surface": surface,
            "max_scene_objects": 200,
            "discovery_dir": str(tmp_path / "blender"),
            "attachment_path": str(tmp_path / "blender" / "attachment.json"),
        },
        security={"allow_script_execution": scripts, "rate_limiting_enabled": False},
    )


def _blender_tool_names(instance: server_module.SimulMCPServer) -> List[str]:
    return instance.tools_by_backend().get("Blender", [])


def _render(document: Any) -> str:
    """JSON text with short arrays and objects kept on one line, so diffs stay readable."""
    return _render_value(document, 0) + "\n"


def _render_value(value: Any, depth: int) -> str:
    compact = json.dumps(value, ensure_ascii=False)
    if (
        not isinstance(value, (dict, list))
        or not value
        or 2 * depth + len(compact) <= 100
    ):
        return compact
    pad = "  " * (depth + 1)
    if isinstance(value, dict):
        items = [
            f"{pad}{json.dumps(key, ensure_ascii=False)}: {_render_value(item, depth + 1)}"
            for key, item in value.items()
        ]
        return "{\n" + ",\n".join(items) + "\n" + "  " * depth + "}"
    items = [f"{pad}{_render_value(item, depth + 1)}" for item in value]
    return "[\n" + ",\n".join(items) + "\n" + "  " * depth + "]"


def _compare_or_update(path: Path, document: Any) -> None:
    """Compare ``document`` with the snapshot text, or rewrite it on request."""
    rendered = _render(document)
    if os.environ.get("SIMUL_UPDATE_SNAPSHOTS") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
        return
    expected = path.read_text(encoding="utf-8")
    if rendered != expected:
        diff = "".join(
            difflib.unified_diff(
                expected.splitlines(keepends=True),
                rendered.splitlines(keepends=True),
                fromfile=f"{path.name} (snapshot)",
                tofile=f"{path.name} (now)",
                n=2,
            )
        )
        pytest.fail(
            f"{path.name} changed; rerun with SIMUL_UPDATE_SNAPSHOTS=1 if intended.\n{diff[:20000]}"
        )


# ---------------------------------------------------------------------------
# tools/list
# ---------------------------------------------------------------------------


def _listing(settings: Settings) -> List[Dict[str, Any]]:
    """The Blender tools a real MCP client lists, in order, as they cross the wire."""
    instance = server_module.SimulMCPServer(settings, backends={"blender"})
    names = set(_blender_tool_names(instance))

    async def _run() -> List[Any]:
        async with Client(instance.mcp) as client:
            return await client.list_tools()

    return [
        tool.model_dump(mode="json", by_alias=True, exclude_none=True)
        for tool in asyncio.run(_run())
        if tool.name in names
    ]


@pytest.fixture(scope="module")
def listing_document(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Any]:
    """Every configuration's listing: the names in order, and each tool once."""
    tmp_path = tmp_path_factory.mktemp("listing")
    surfaces: Dict[str, List[str]] = {}
    tools: Dict[str, Dict[str, Any]] = {}
    with pytest.MonkeyPatch.context() as patcher:
        # Embedded mode needs bpy; the listing only needs the adapter to say so.
        patcher.setattr(backends_module, "is_blender_available", lambda: True)
        patcher.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        for key, (mode, surface, scripts) in SURFACES.items():
            listed = _listing(_settings(tmp_path, mode, surface, scripts))
            surfaces[key] = [tool["name"] for tool in listed]
            for tool in listed:
                known = tools.setdefault(tool["name"], tool)
                assert _render(known) == _render(
                    tool
                ), f"{tool['name']} differs in {key}"
    return {"surfaces": surfaces, "tools": tools}


def test_tool_listing_matches_the_snapshot(listing_document: Dict[str, Any]) -> None:
    _compare_or_update(LISTING_SNAPSHOT, listing_document)


def test_thin_surfaces_list_the_thin_tools(listing_document: Dict[str, Any]) -> None:
    for key in ("attached/thin", "embedded/thin"):
        assert set(listing_document["surfaces"][key]) == set(THIN_BLENDER_TOOLS), key


def test_disabling_scripts_removes_only_the_script_tool(
    listing_document: Dict[str, Any],
) -> None:
    surfaces = listing_document["surfaces"]
    full = [
        name for name in surfaces["attached/full"] if name != "execute_blender_script"
    ]
    assert surfaces["attached/full/scripts-disabled"] == full


def test_every_tool_registers_without_output_schema_and_with_optional_tasks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Background task support and the absent output schema are per registration."""
    task_config = object()
    monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
    monkeypatch.setattr(
        server_module.SimulMCPServer, "_task_optional", lambda self: task_config
    )
    instance = server_module.SimulMCPServer(
        _settings(tmp_path, "attached"), backends={"blender"}
    )
    names = set(_blender_tool_names(instance))
    registered = [tool for tool in instance.mcp.tools if tool.name in names]

    assert {tool.name for tool in registered} == set(CALLS)
    for tool in registered:
        assert tool.kwargs["output_schema"] is None, tool.name
        assert tool.kwargs["task"] is task_config, tool.name


# ---------------------------------------------------------------------------
# Call forwarding
# ---------------------------------------------------------------------------


def _recording_attach(calls: List[SessionCall]) -> Any:
    """``BlenderAttachments.attach`` that records its arguments instead of attaching."""

    def attach(
        self: BlenderAttachments,
        instance_id: Optional[str] = None,
        window_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        calls.append(
            (
                "BlenderAttachments.attach",
                {"instance_id": instance_id, "window_id": window_id},
            )
        )
        if instance_id is None:
            raise ValueError("Several Blender processes are running; pass instance_id")
        return dict(PROBE_PAYLOAD)

    return attach


def _snapshot_calls(
    calls: List[SessionCall], agent: Optional[str]
) -> List[Dict[str, Any]]:
    """Recorded calls with each argument as ``repr``; the session's own agent id is named, not spelled out."""
    return [
        {
            "method": method,
            "arguments": {
                name: (
                    SESSION_AGENT
                    if agent is not None and value == agent
                    else repr(value)
                )
                for name, value in arguments.items()
            },
        }
        for method, arguments in calls
    ]


def _outcome(payload: Dict[str, Any], called: bool) -> Dict[str, Any]:
    """The response, or just the model it was checked against when a call reached the session."""
    error = str(payload.get("error") or "")
    check = _MODEL_CHECK.match(error)
    if called and payload.get("error_type") == "ValidationError" and check:
        return {"checked_against": check.group(1)}
    return json.loads(
        _PYDANTIC_URL.sub("https://errors.pydantic.dev/<version>/", json.dumps(payload))
    )


def _usage_rows(instance: server_module.SimulMCPServer) -> List[Dict[str, Any]]:
    return list(reversed(instance.usage_tracker.get_recent(limit=100_000)))


def _accepts_object(prop: Dict[str, Any]) -> bool:
    """Whether a parameter's schema admits a JSON object."""
    types = [prop.get("type")] + [
        member.get("type") for member in prop.get("anyOf", [])
    ]
    return "object" in types


def _invalid_arguments(
    minimal: Dict[str, Any], schema: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """The minimal arguments with the first parameter that refuses an object given one.

    The wrapper is called directly, past FastMCP's own argument check, so the
    value reaches the tool's request validation.
    """
    properties = schema.get("properties", {})
    target = next(
        (name for name, prop in properties.items() if not _accepts_object(prop)), None
    )
    if target is None:
        return None
    arguments = {**minimal, target: INVALID_VALUE}
    if "agent_id" in properties and target != "agent_id":
        arguments["agent_id"] = SNAPSHOT_AGENT
    return arguments


def _strict_adapter(settings: Settings) -> BlenderSessionAdapter:
    """The adapter the server builds: every session is one strict, recording session."""
    return BlenderSessionAdapter(
        settings, StrictBlenderSession(default_response=PROBE_PAYLOAD)
    )


def _record_calls(tmp_path: Path, mode: str) -> Dict[str, Any]:
    """Call every tool in every case.

    Returns:
        ``outcomes`` per tool and case, ``recorded_invalid`` (usage records
        left by each invalid call) and ``rejected`` (calls the strict session
        refused).
    """
    attach_calls: List[SessionCall] = []
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(backends_module, "BlenderRuntimeAdapter", _strict_adapter)
        patcher.setattr(backends_module, "is_blender_available", lambda: True)
        patcher.setattr(BlenderAttachments, "attach", _recording_attach(attach_calls))
        patcher.setattr(BlenderAttachments, "instances", lambda self: [])
        instance = server_module.SimulMCPServer(
            _settings(tmp_path, mode), backends={"blender"}
        )
        instance.usage_tracker = ToolUsageTracker(log_dir=tmp_path / f"usage-{mode}")
        adapter = instance.blender_adapter
        assert isinstance(adapter, BlenderSessionAdapter)
        session = adapter.session
        outcomes: Dict[str, Dict[str, Any]] = {}
        recorded_invalid: Dict[str, int] = {}

        def _calls(agent: Optional[str] = None) -> List[Dict[str, Any]]:
            calls = _snapshot_calls(session.calls + attach_calls, agent)
            session.calls.clear()
            attach_calls.clear()
            return calls

        async def _run() -> None:
            async with Client(instance.mcp) as client:
                listed = {tool.name: tool for tool in await client.list_tools()}
                for name in [tool for tool in listed if tool in CALLS]:
                    tool_outcomes: Dict[str, Any] = {}
                    for case, arguments in CALLS[name].items():
                        before = len(_usage_rows(instance))
                        result = await client.call_tool(
                            name, arguments, raise_on_error=False
                        )
                        payload = json.loads(result.content[-1].text)
                        rows = _usage_rows(instance)[before:]
                        calls = _calls(rows[0]["agent_id"] if rows else None)
                        tool_outcomes[case] = {
                            "arguments": arguments,
                            "session_calls": calls,
                            "response": _outcome(payload, bool(calls)),
                            "usage": [
                                {"tool": row["tool"], "success": row["success"]}
                                for row in rows
                            ],
                        }
                    invalid = _invalid_arguments(
                        CALLS[name]["minimal"], listed[name].inputSchema
                    )
                    if invalid is not None:
                        before = len(_usage_rows(instance))
                        function = (await instance.mcp.get_tool(name)).fn
                        payload = json.loads(
                            (await function(**invalid)).content[-1].text
                        )
                        recorded_invalid[name] = len(_usage_rows(instance)) - before
                        calls = _calls()
                        tool_outcomes["invalid"] = {
                            "arguments": invalid,
                            "session_calls": calls,
                            "response": _outcome(payload, bool(calls)),
                        }
                    outcomes[name] = tool_outcomes

        asyncio.run(_run())
    return {
        "outcomes": outcomes,
        "recorded_invalid": recorded_invalid,
        "rejected": list(session.rejected),
    }


@pytest.fixture(scope="module")
def recorded(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Dict[str, Any]]:
    """The recorded calls in attached and in embedded mode."""
    return {
        mode: _record_calls(tmp_path_factory.mktemp(mode), mode)
        for mode in ("attached", "embedded")
    }


def test_tool_calls_match_the_snapshot(recorded: Dict[str, Dict[str, Any]]) -> None:
    """Attached outcomes in full; embedded mode only where it differs."""
    attached = recorded["attached"]["outcomes"]
    embedded = recorded["embedded"]["outcomes"]
    assert set(attached) == set(embedded) == set(CALLS)
    document = {
        "tools": attached,
        "embedded_differences": {
            name: embedded[name]
            for name in attached
            if _render(attached[name]) != _render(embedded[name])
        },
    }
    _compare_or_update(CALLS_SNAPSHOT, document)


def test_only_attach_blender_window_depends_on_the_mode(
    recorded: Dict[str, Dict[str, Any]],
) -> None:
    attached = recorded["attached"]["outcomes"]
    embedded = recorded["embedded"]["outcomes"]
    differing = {
        name for name in attached if _render(attached[name]) != _render(embedded[name])
    }
    assert differing == {"attach_blender_window"}


def test_invalid_input_is_recorded_only_by_the_tools_validating_inside_the_envelope(
    recorded: Dict[str, Dict[str, Any]],
) -> None:
    """Before the spec table, most tools validated outside the envelope and left no record."""
    for mode, record in recorded.items():
        rejected = {
            name
            for name, outcomes in record["outcomes"].items()
            if "invalid" in outcomes and not outcomes["invalid"]["session_calls"]
        }
        recording = {name for name in rejected if record["recorded_invalid"][name]}
        assert recording == {
            "list_blender_scene_objects",
            "get_blender_object_info",
        }, mode


def test_no_tool_calls_an_operation_the_session_lacks(
    recorded: Dict[str, Dict[str, Any]],
) -> None:
    """Every call reaches a real BlenderRuntimeSession operation with arguments it takes.

    The envelope turns the AttributeError of a missing operation into an error
    payload, which a snapshot would otherwise just record (it did, for
    check_blender_object_bounds calling check_object_bounds).
    """
    for mode, record in recorded.items():
        assert record["rejected"] == [], mode
