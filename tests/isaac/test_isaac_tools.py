"""Unit tests for IsaacTools — mocked TCP socket client."""
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_path))

from simul_mcp.adapters.isaac_socket_client import ScriptResult
from simul_mcp.mcp.tools.isaac_tools import IsaacTools


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(data: Dict[str, Any]) -> ScriptResult:
    """Create a successful ScriptResult containing JSON output."""
    return ScriptResult(success=True, output=json.dumps(data))


def _make_error_result(
    error_name: str = "RuntimeError",
    error_value: str = "Something failed",
    traceback: str = "",
) -> ScriptResult:
    """Create a failed ScriptResult."""
    return ScriptResult(
        success=False,
        output="",
        error_name=error_name,
        error_value=error_value,
        traceback=traceback,
    )


def _make_tools(
    execute_return: Optional[ScriptResult] = None,
    execute_side_effect: Optional[Exception] = None,
    bridge_response: Optional[Dict[str, Any]] = None,
    bridge_side_effect: Optional[Exception] = None,
) -> IsaacTools:
    """Build an IsaacTools instance backed by a fully-mocked client."""
    client: MagicMock = MagicMock()
    client.address = "127.0.0.1:8226"
    client.timeout_seconds = 30.0
    client.bridge_enabled = bridge_response is not None or bridge_side_effect is not None
    client.fallback_to_vscode = True
    client.execute = AsyncMock(
        return_value=execute_return,
        side_effect=execute_side_effect,
    )
    client.bridge_request = AsyncMock(
        return_value=bridge_response,
        side_effect=bridge_side_effect,
    )
    client.execute_bridge_script_only = AsyncMock(
        return_value=execute_return,
        side_effect=execute_side_effect,
    )
    client.execute_vscode_only = AsyncMock(
        return_value=execute_return,
        side_effect=execute_side_effect,
    )
    return IsaacTools(client=client, settings=MagicMock())


# ---------------------------------------------------------------------------
# Error handling — _execute_json_script paths
# ---------------------------------------------------------------------------


class TestExecuteJsonScript:
    """Verify _execute_json_script error handling for all failure modes."""

    def test_connection_refused(self) -> None:
        """ConnectionRefusedError returns an error dict."""
        tools = _make_tools(execute_side_effect=ConnectionRefusedError("refused"))
        result = asyncio.run(tools.get_isaac_stage_info())
        assert result["success"] is False
        assert result["error_type"] == "ConnectionError"

    def test_timeout(self) -> None:
        """TimeoutError returns an error dict."""
        tools = _make_tools(execute_side_effect=TimeoutError("timed out"))
        result = asyncio.run(tools.get_isaac_stage_info())
        assert result["success"] is False
        assert result["error_type"] == "TimeoutError"

    def test_generic_exception(self) -> None:
        """Unexpected exceptions are caught and reported."""
        tools = _make_tools(execute_side_effect=RuntimeError("boom"))
        result = asyncio.run(tools.get_isaac_stage_info())
        assert result["success"] is False
        assert result["error_type"] == "Exception"
        assert "boom" in result["error"]

    def test_script_failure(self) -> None:
        """Failed script execution returns error details."""
        tools = _make_tools(
            execute_return=_make_error_result(
                error_name="NameError",
                error_value="name 'x' is not defined",
                traceback="Traceback...",
            )
        )
        result = asyncio.run(tools.get_isaac_stage_info())
        assert result["success"] is False
        assert result["error_type"] == "NameError"
        assert "not defined" in result["error"]

    def test_script_failure_includes_traceback(self) -> None:
        """Traceback is included in details when present."""
        tools = _make_tools(
            execute_return=_make_error_result(traceback="line 5\nValueError")
        )
        result = asyncio.run(tools.get_isaac_stage_info())
        assert result["success"] is False
        assert result["details"]["traceback"] == "line 5\nValueError"

    def test_empty_output(self) -> None:
        """Empty script output produces an EmptyOutput error."""
        tools = _make_tools(execute_return=ScriptResult(success=True, output=""))
        result = asyncio.run(tools.get_isaac_stage_info())
        assert result["success"] is False
        assert result["error_type"] == "EmptyOutput"

    def test_whitespace_only_output(self) -> None:
        """Whitespace-only output is treated as empty."""
        r = ScriptResult(success=True, output="  \n  ")
        tools = _make_tools(execute_return=r)
        result = asyncio.run(tools.get_isaac_stage_info())
        assert result["success"] is False
        assert result["error_type"] == "EmptyOutput"

    def test_invalid_json(self) -> None:
        """Non-JSON output produces a JSONDecodeError error."""
        r = ScriptResult(success=True, output="not json at all")
        tools = _make_tools(execute_return=r)
        result = asyncio.run(tools.get_isaac_stage_info())
        assert result["success"] is False
        assert result["error_type"] == "JSONDecodeError"
        assert "raw_output" in result.get("details", {})

    def test_success_adds_flag(self) -> None:
        """Valid JSON output gets success=True injected."""
        tools = _make_tools(execute_return=_make_result({"key": "value"}))
        result = asyncio.run(tools.get_isaac_stage_info())
        assert result["success"] is True
        assert result["key"] == "value"



# ---------------------------------------------------------------------------
# Phase 1 — Scene Inspection
# ---------------------------------------------------------------------------


class TestSceneInspection:
    """Tests for the six scene inspection tools."""

    def test_get_isaac_stage_info(self) -> None:
        """Stage info returns metadata about the open stage."""
        data = {
            "stage_url": "file:///test.usd",
            "up_axis": "Y",
            "meters_per_unit": 0.01,
            "time_codes_per_second": 24.0,
            "start_time": 0.0,
            "end_time": 100.0,
            "frame_rate": 24.0,
            "total_prims": 42,
            "root_prims": ["/World"],
            "layer_count": 2,
            "default_prim": "World",
        }
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(tools.get_isaac_stage_info())
        assert result["success"] is True
        assert result["stage_url"] == "file:///test.usd"
        assert result["total_prims"] == 42
        assert result["up_axis"] == "Y"

    def test_get_isaac_stage_info_uses_typed_bridge_when_available(self) -> None:
        """Typed bridge stage info should be preferred over raw script execution."""
        tools = _make_tools(
            execute_return=_make_result({"legacy": True}),
            bridge_response={
                "status": "ok",
                "payload": {
                    "stage_url": "file:///bridge.usd",
                    "total_prims": 7,
                    "transport": "simul_bridge",
                },
            },
        )

        result = asyncio.run(tools.get_isaac_stage_info())

        assert result["success"] is True
        assert result["stage_url"] == "file:///bridge.usd"
        tools._client.bridge_request.assert_awaited_once_with("get_stage_info", {})
        tools._client.execute.assert_not_called()
        tools._client.execute_vscode_only.assert_not_called()

    def test_get_isaac_stage_info_falls_back_to_vscode_when_typed_bridge_is_unavailable(
        self,
    ) -> None:
        """Typed bridge fallbacks should use the VS Code socket, not raw bridge exec."""
        tools = _make_tools(
            execute_return=_make_result(
                {
                    "stage_url": "file:///fallback.usd",
                    "total_prims": 3,
                }
            ),
            bridge_response={
                "status": "error",
                "error": {
                    "name": "UnknownAction",
                    "message": "not implemented",
                },
            },
        )

        result = asyncio.run(tools.get_isaac_stage_info())

        assert result["success"] is True
        assert result["stage_url"] == "file:///fallback.usd"
        tools._client.bridge_request.assert_awaited_once_with("get_stage_info", {})
        tools._client.execute_vscode_only.assert_awaited_once()
        tools._client.execute_bridge_script_only.assert_not_called()

    def test_raw_script_only_tools_use_vscode_when_bridge_mode_is_enabled(self) -> None:
        """Raw-only tool paths should avoid execute_script on the typed bridge."""
        tools = _make_tools(
            execute_return=_make_result(
                {
                    "stage_url": "file:///fallback.usd",
                    "total_prims": 3,
                    "max_depth": 1,
                    "has_physics": False,
                    "has_animation": False,
                    "type_counts": {},
                    "root_prims": ["/World"],
                    "up_axis": "Z",
                    "meters_per_unit": 1.0,
                }
            ),
            bridge_response={"status": "ok", "payload": {"reachable": True}},
        )

        result = asyncio.run(tools.get_isaac_scene_summary())

        assert result["success"] is True
        # With the corrected default transport routing, raw-script tools that
        # don't explicitly set transport_mode go through execute() which
        # handles bridge-first-then-fallback internally.
        tools._client.execute.assert_awaited_once()
        tools._client.execute_bridge_script_only.assert_not_called()

    def test_list_isaac_prims_defaults(self) -> None:
        """list_isaac_prims returns prim listing with defaults."""
        data = {
            "root_path": "/",
            "type_filter": None,
            "count": 2,
            "truncated": False,
            "prims": [
                {"path": "/World", "type": "Xform", "name": "World", "active": True},
                {"path": "/World/Cube", "type": "Mesh", "name": "Cube", "active": True},
            ],
        }
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(tools.list_isaac_prims())
        assert result["success"] is True
        assert result["count"] == 2
        assert len(result["prims"]) == 2
        assert result["prims"][0]["type"] == "Xform"

    def test_list_isaac_prims_with_filter(self) -> None:
        """list_isaac_prims passes type filter."""
        data = {"root_path": "/", "type_filter": "Mesh", "count": 1, "truncated": False, "prims": []}
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(tools.list_isaac_prims(prim_type="Mesh", max_depth=3))
        assert result["success"] is True
        assert result["type_filter"] == "Mesh"

    def test_get_isaac_prim_info(self) -> None:
        """get_isaac_prim_info returns detailed prim data."""
        data = {
            "path": "/World/Cube",
            "name": "Cube",
            "type": "Mesh",
            "is_active": True,
            "is_defined": True,
            "is_instance": False,
            "children_count": 0,
            "children": [],
            "attributes": {"extent": [[-50, -50, -50], [50, 50, 50]]},
        }
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(tools.get_isaac_prim_info(prim_path="/World/Cube"))
        assert result["success"] is True
        assert result["type"] == "Mesh"
        assert result["name"] == "Cube"

    def test_get_isaac_prim_transform(self) -> None:
        """get_isaac_prim_transform returns position/rotation/scale."""
        data = {
            "prim_path": "/World/Cube",
            "space": "world",
            "translation": [1.0, 2.0, 3.0],
            "rotation_quat": [1.0, 0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
        }
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(
            tools.get_isaac_prim_transform(prim_path="/World/Cube", world_space=True)
        )
        assert result["success"] is True
        assert result["translation"] == [1.0, 2.0, 3.0]
        assert result["space"] == "world"

    def test_search_isaac_prims(self) -> None:
        """search_isaac_prims returns matching prims."""
        data = {
            "search_type": "type",
            "query": "Mesh",
            "root_path": "/",
            "count": 1,
            "truncated": False,
            "matches": [{"path": "/World/Cube", "type": "Mesh", "name": "Cube"}],
        }
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(tools.search_isaac_prims(search_type="type", query="Mesh"))
        assert result["success"] is True
        assert result["count"] == 1
        assert result["matches"][0]["path"] == "/World/Cube"

    def test_get_isaac_scene_summary(self) -> None:
        """get_isaac_scene_summary returns high-level overview."""
        data = {
            "stage_url": "file:///scene.usd",
            "total_prims": 100,
            "max_depth": 5,
            "has_physics": True,
            "has_animation": False,
            "type_counts": {"Mesh": 10, "Xform": 20},
            "root_prims": ["/World"],
            "up_axis": "Z",
            "meters_per_unit": 1.0,
        }
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(tools.get_isaac_scene_summary())
        assert result["success"] is True
        assert result["total_prims"] == 100
        assert result["has_physics"] is True
        assert result["type_counts"]["Mesh"] == 10


# ---------------------------------------------------------------------------
# Phase 2 — Viewport & Camera
# ---------------------------------------------------------------------------


class TestViewportCamera:
    """Tests for the four viewport and camera tools."""

    def test_get_isaac_camera_info(self) -> None:
        """Camera info returns lens and transform data."""
        data = {
            "camera_path": "/OmniverseKit_Persp",
            "position": [5.0, 5.0, 5.0],
            "rotation_quat": [1.0, 0.0, 0.0, 0.0],
            "focal_length": 24.0,
            "horizontal_aperture": 20.955,
            "vertical_aperture": 15.29,
            "clipping_range": [0.1, 10000.0],
            "projection": "perspective",
        }
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(tools.get_isaac_camera_info(camera_path="/OmniverseKit_Persp"))
        assert result["success"] is True
        assert result["focal_length"] == 24.0
        assert result["projection"] == "perspective"

    def test_list_isaac_cameras(self) -> None:
        """list_isaac_cameras enumerates all cameras in the scene."""
        data = {
            "count": 2,
            "active_camera": "/OmniverseKit_Persp",
            "cameras": [
                {"path": "/OmniverseKit_Persp", "name": "Persp", "focal_length": 24.0, "projection": "perspective"},
                {"path": "/World/TopCam", "name": "TopCam", "focal_length": 50.0, "projection": "perspective"},
            ],
        }
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(tools.list_isaac_cameras())
        assert result["success"] is True
        assert result["count"] == 2
        assert result["active_camera"] == "/OmniverseKit_Persp"

    def test_set_isaac_camera(self) -> None:
        """set_isaac_camera sets camera position and target."""
        data = {"camera_path": "/OmniverseKit_Persp", "position": [10.0, 10.0, 10.0], "target": [0.0, 0.0, 0.0]}
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(
            tools.set_isaac_camera(position=[10.0, 10.0, 10.0], target=[0.0, 0.0, 0.0])
        )
        assert result["success"] is True
        assert result["position"] == [10.0, 10.0, 10.0]

    def test_capture_isaac_viewport(self) -> None:
        """capture_isaac_viewport returns a base64-encoded image."""
        data = {
            "width": 1280,
            "height": 720,
            "format": "png",
            "encoding": "base64",
            "image_base64": "iVBORw0KGgoAAAANSUhEUg==",
        }
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(tools.capture_isaac_viewport(width=1280, height=720))
        assert result["success"] is True
        assert result["format"] == "png"
        assert result["encoding"] == "base64"
        assert isinstance(result["image_base64"], str)


# ---------------------------------------------------------------------------
# Phase 3 — Prim Manipulation
# ---------------------------------------------------------------------------


class TestPrimManipulation:
    """Tests for the seven prim manipulation tools."""

    def test_create_isaac_prim(self) -> None:
        """create_isaac_prim creates a new prim."""
        data = {"prim_path": "/World/NewXform", "prim_type": "Xform", "created": True}
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(tools.create_isaac_prim(prim_path="/World/NewXform", prim_type="Xform"))
        assert result["success"] is True
        assert result["created"] is True
        assert result["prim_type"] == "Xform"

    def test_create_isaac_prim_with_attributes(self) -> None:
        """create_isaac_prim passes optional attributes."""
        data = {"prim_path": "/World/Cube", "prim_type": "Cube", "created": True}
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(
            tools.create_isaac_prim(
                prim_path="/World/Cube",
                prim_type="Cube",
                attributes={"size": 100},
            )
        )
        assert result["success"] is True
        assert result["created"] is True

    def test_delete_isaac_prim(self) -> None:
        """delete_isaac_prim removes a prim."""
        data = {"prim_path": "/World/Cube", "deleted": True}
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(tools.delete_isaac_prim(prim_path="/World/Cube"))
        assert result["success"] is True
        assert result["deleted"] is True

    def test_set_isaac_prim_transform(self) -> None:
        """set_isaac_prim_transform applies translation/rotation/scale."""
        data = {
            "prim_path": "/World/Cube",
            "translation": [1.0, 2.0, 3.0],
            "rotation_euler_set": [0.0, 45.0, 0.0],
            "scale_set": [2.0, 2.0, 2.0],
        }
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(
            tools.set_isaac_prim_transform(
                prim_path="/World/Cube",
                translation=[1.0, 2.0, 3.0],
                rotation_euler=[0.0, 45.0, 0.0],
                scale=[2.0, 2.0, 2.0],
            )
        )
        assert result["success"] is True
        assert result["translation"] == [1.0, 2.0, 3.0]

    def test_set_isaac_prim_visibility(self) -> None:
        """set_isaac_prim_visibility toggles visibility."""
        data = {"prim_path": "/World/Cube", "visibility": "invisible", "effective_visibility": "invisible"}
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(tools.set_isaac_prim_visibility(prim_path="/World/Cube", visible=False))
        assert result["success"] is True
        assert result["visibility"] == "invisible"


    def test_set_isaac_prim_attribute(self) -> None:
        """set_isaac_prim_attribute sets a named attribute."""
        data = {"prim_path": "/World/Cube", "attribute": "size", "value_set": 200, "value_read": "200"}
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(
            tools.set_isaac_prim_attribute(prim_path="/World/Cube", attribute_name="size", value=200)
        )
        assert result["success"] is True
        assert result["attribute"] == "size"

    def test_duplicate_isaac_prim(self) -> None:
        """duplicate_isaac_prim copies a prim to a new path."""
        data = {"source_path": "/World/Cube", "new_path": "/World/CubeCopy", "type": "Mesh", "duplicated": True}
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(tools.duplicate_isaac_prim(prim_path="/World/Cube", new_path="/World/CubeCopy"))
        assert result["success"] is True
        assert result["duplicated"] is True
        assert result["new_path"] == "/World/CubeCopy"

    def test_reparent_isaac_prim(self) -> None:
        """reparent_isaac_prim moves a prim under a new parent."""
        data = {"old_path": "/World/Cube", "new_path": "/World/Group/Cube", "parent": "/World/Group", "reparented": True}
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(tools.reparent_isaac_prim(prim_path="/World/Cube", new_parent_path="/World/Group"))
        assert result["success"] is True
        assert result["reparented"] is True
        assert result["parent"] == "/World/Group"


# ---------------------------------------------------------------------------
# Phase 4 & 5 — Physics Inspection & Configuration
# ---------------------------------------------------------------------------


class TestPhysics:
    """Tests for physics inspection and configuration tools."""

    def test_get_isaac_physics_scene(self) -> None:
        """get_isaac_physics_scene returns physics scene config."""
        data = {
            "physics_scene_count": 1,
            "scenes": [{"path": "/PhysicsScene", "gravity_direction": [0, -1, 0], "gravity_magnitude": 9.81}],
            "has_physics": True,
        }
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(tools.get_isaac_physics_scene())
        assert result["success"] is True
        assert result["has_physics"] is True
        assert result["scenes"][0]["gravity_magnitude"] == 9.81

    def test_get_isaac_rigid_body_info(self) -> None:
        """get_isaac_rigid_body_info returns rigid body state."""
        data = {
            "prim_path": "/World/Cube",
            "rigid_body_enabled": True,
            "is_kinematic": False,
            "velocity": [0.0, 0.0, 0.0],
            "angular_velocity": [0.0, 0.0, 0.0],
            "has_mass_api": True,
            "mass": 1.0,
            "center_of_mass": [0.0, 0.0, 0.0],
            "diagonal_inertia": [1.0, 1.0, 1.0],
        }
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(tools.get_isaac_rigid_body_info(prim_path="/World/Cube"))
        assert result["success"] is True
        assert result["rigid_body_enabled"] is True
        assert result["mass"] == 1.0

    def test_list_isaac_physics_objects(self) -> None:
        """list_isaac_physics_objects enumerates all physics objects."""
        data = {
            "root_path": "/",
            "rigid_body_count": 2,
            "collider_count": 2,
            "joint_count": 0,
            "rigid_bodies": [{"path": "/World/Cube", "type": "Mesh"}],
            "colliders": [{"path": "/World/Cube", "type": "Mesh"}],
            "joints": [],
        }
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(tools.list_isaac_physics_objects(root_path="/"))
        assert result["success"] is True
        assert result["rigid_body_count"] == 2

    def test_get_isaac_collision_info(self) -> None:
        """get_isaac_collision_info returns collision details."""
        data = {"prim_path": "/World/Cube", "collision_enabled": True, "has_mesh_collision": False, "approximation": "convexHull"}
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(tools.get_isaac_collision_info(prim_path="/World/Cube"))
        assert result["success"] is True
        assert result["collision_enabled"] is True
        assert result["approximation"] == "convexHull"

    def test_get_isaac_joint_info(self) -> None:
        """get_isaac_joint_info returns joint configuration."""
        data = {
            "prim_path": "/World/RevoluteJoint",
            "joint_type": "PhysicsRevoluteJoint",
            "enabled": True,
            "body0": ["/World/Link0"],
            "body1": ["/World/Link1"],
            "exclude_from_articulation": False,
            "break_force": None,
            "break_torque": None,
            "limits": {"type": "angular", "axis": "X", "lower": -90.0, "upper": 90.0},
        }
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(tools.get_isaac_joint_info(prim_path="/World/RevoluteJoint"))
        assert result["success"] is True
        assert result["joint_type"] == "PhysicsRevoluteJoint"
        assert result["limits"]["lower"] == -90.0

    def test_get_isaac_mass_properties(self) -> None:
        """get_isaac_mass_properties returns mass/inertia data."""
        data = {
            "prim_path": "/World/Cube",
            "mass": 5.0,
            "density": 1000.0,
            "center_of_mass": [0.0, 0.0, 0.0],
            "diagonal_inertia": [1.0, 1.0, 1.0],
            "principal_axes": [1.0, 0.0, 0.0, 0.0],
        }
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(tools.get_isaac_mass_properties(prim_path="/World/Cube"))
        assert result["success"] is True
        assert result["mass"] == 5.0
        assert result["density"] == 1000.0


    def test_add_isaac_rigid_body(self) -> None:
        """add_isaac_rigid_body applies the rigid body API."""
        data = {"prim_path": "/World/Cube", "rigid_body_applied": True, "kinematic": False}
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(tools.add_isaac_rigid_body(prim_path="/World/Cube"))
        assert result["success"] is True
        assert result["rigid_body_applied"] is True
        assert result["kinematic"] is False

    def test_add_isaac_rigid_body_kinematic(self) -> None:
        """add_isaac_rigid_body with kinematic=True."""
        data = {"prim_path": "/World/Cube", "rigid_body_applied": True, "kinematic": True}
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(tools.add_isaac_rigid_body(prim_path="/World/Cube", kinematic=True))
        assert result["success"] is True
        assert result["kinematic"] is True

    def test_add_isaac_collision(self) -> None:
        """add_isaac_collision applies collision API."""
        data = {"prim_path": "/World/Cube", "collision_applied": True, "approximation": "convexHull"}
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(tools.add_isaac_collision(prim_path="/World/Cube", approximation="convexHull"))
        assert result["success"] is True
        assert result["collision_applied"] is True
        assert result["approximation"] == "convexHull"

    def test_set_isaac_mass_properties(self) -> None:
        """set_isaac_mass_properties sets mass and density."""
        data = {"prim_path": "/World/Cube", "mass": 10.0, "density": None, "center_of_mass": None}
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(tools.set_isaac_mass_properties(prim_path="/World/Cube", mass=10.0))
        assert result["success"] is True
        assert result["mass"] == 10.0

    def test_set_isaac_physics_material(self) -> None:
        """set_isaac_physics_material sets friction and restitution."""
        data = {
            "prim_path": "/World/Cube",
            "static_friction": 0.8,
            "dynamic_friction": 0.6,
            "restitution": 0.3,
            "physics_material_applied": True,
        }
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(
            tools.set_isaac_physics_material(
                prim_path="/World/Cube",
                static_friction=0.8,
                dynamic_friction=0.6,
                restitution=0.3,
            )
        )
        assert result["success"] is True
        assert result["static_friction"] == 0.8
        assert result["restitution"] == 0.3


# ---------------------------------------------------------------------------
# Phase 6 — Simulation Control
# ---------------------------------------------------------------------------


class TestSimulationControl:
    """Tests for the seven simulation control tools."""

    def test_get_isaac_simulation_state(self) -> None:
        """get_isaac_simulation_state returns current sim state."""
        data = {
            "state": "stopped",
            "current_time": 0.0,
            "time_codes_per_second": 60.0,
            "is_playing": False,
            "is_stopped": True,
        }
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(tools.get_isaac_simulation_state())
        assert result["success"] is True
        assert result["state"] == "stopped"
        assert result["is_stopped"] is True

    def test_start_isaac_simulation(self) -> None:
        """start_isaac_simulation begins playback."""
        data = {"state": "playing", "started": True}
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(tools.start_isaac_simulation())
        assert result["success"] is True
        assert result["state"] == "playing"
        assert result["started"] is True

    def test_stop_isaac_simulation(self) -> None:
        """stop_isaac_simulation halts playback."""
        data = {"state": "stopped", "stopped": True}
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(tools.stop_isaac_simulation())
        assert result["success"] is True
        assert result["state"] == "stopped"

    def test_pause_isaac_simulation(self) -> None:
        """pause_isaac_simulation pauses playback."""
        data = {"state": "paused", "paused": True}
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(tools.pause_isaac_simulation())
        assert result["success"] is True
        assert result["state"] == "paused"

    def test_step_isaac_simulation(self) -> None:
        """step_isaac_simulation advances by N steps."""
        data = {"steps": 5, "current_time": 0.0833, "state": "paused"}
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(tools.step_isaac_simulation(num_steps=5))
        assert result["success"] is True
        assert result["steps"] == 5

    def test_reset_isaac_simulation(self) -> None:
        """reset_isaac_simulation resets to initial state."""
        data = {"state": "stopped", "current_time": 0.0, "reset": True}
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(tools.reset_isaac_simulation())
        assert result["success"] is True
        assert result["reset"] is True
        assert result["current_time"] == 0.0

    def test_get_isaac_simulation_time(self) -> None:
        """get_isaac_simulation_time returns time info."""
        data = {
            "current_time": 2.5,
            "start_time": 0.0,
            "end_time": 100.0,
            "time_codes_per_second": 60.0,
            "is_playing": True,
        }
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(tools.get_isaac_simulation_time())
        assert result["success"] is True
        assert result["current_time"] == 2.5
        assert result["is_playing"] is True


# ---------------------------------------------------------------------------
# Phase 7 — Materials & Appearance
# ---------------------------------------------------------------------------


class TestMaterials:
    """Tests for the four material and appearance tools."""

    def test_get_isaac_material_info(self) -> None:
        """get_isaac_material_info returns material and shader details."""
        data = {
            "material_path": "/World/Looks/OmniPBR",
            "shader_path": "/World/Looks/OmniPBR/Shader",
            "shader_type": "OmniPBR",
            "render_context": "mdl",
            "inputs": {"diffuse_color_constant": [0.8, 0.2, 0.1]},
        }
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(tools.get_isaac_material_info(material_path="/World/Looks/OmniPBR"))
        assert result["success"] is True
        assert result["shader_type"] == "OmniPBR"
        assert result["render_context"] == "mdl"

    def test_list_isaac_materials(self) -> None:
        """list_isaac_materials enumerates all materials."""
        data = {
            "count": 2,
            "materials": [
                {"path": "/World/Looks/Mat1", "name": "Mat1", "shader_type": "OmniPBR"},
                {"path": "/World/Looks/Mat2", "name": "Mat2", "shader_type": "OmniGlass"},
            ],
        }
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(tools.list_isaac_materials())
        assert result["success"] is True
        assert result["count"] == 2
        assert result["materials"][0]["shader_type"] == "OmniPBR"

    def test_assign_isaac_material(self) -> None:
        """assign_isaac_material binds material to prim."""
        data = {"prim_path": "/World/Cube", "material_path": "/World/Looks/Mat1", "bound": True}
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(
            tools.assign_isaac_material(prim_path="/World/Cube", material_path="/World/Looks/Mat1")
        )
        assert result["success"] is True
        assert result["bound"] is True

    def test_set_isaac_material_property(self) -> None:
        """set_isaac_material_property updates a shader input."""
        data = {"material_path": "/World/Looks/Mat1", "property_name": "diffuse_color_constant", "value_set": "[0.5, 0.5, 0.5]"}
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(
            tools.set_isaac_material_property(
                material_path="/World/Looks/Mat1",
                property_name="diffuse_color_constant",
                value=[0.5, 0.5, 0.5],
            )
        )
        assert result["success"] is True
        assert result["property_name"] == "diffuse_color_constant"


# ---------------------------------------------------------------------------
# Phase 8 — Stage & Asset Operations
# ---------------------------------------------------------------------------


class TestStageAssetOps:
    """Tests for the five stage and asset operation tools."""

    def test_open_isaac_stage(self) -> None:
        """open_isaac_stage opens a USD file."""
        data = {"file_path": "/scenes/test.usd", "opened": True, "total_prims": 50}
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(tools.open_isaac_stage(file_path="/scenes/test.usd"))
        assert result["success"] is True
        assert result["opened"] is True
        assert result["total_prims"] == 50

    def test_save_isaac_stage(self) -> None:
        """save_isaac_stage saves the current stage."""
        data = {"file_path": "/scenes/test.usd", "saved": True}
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(tools.save_isaac_stage())
        assert result["success"] is True
        assert result["saved"] is True

    def test_save_isaac_stage_with_path(self) -> None:
        """save_isaac_stage with explicit file path."""
        data = {"file_path": "/scenes/output.usd", "saved": True}
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(tools.save_isaac_stage(file_path="/scenes/output.usd"))
        assert result["success"] is True
        assert result["file_path"] == "/scenes/output.usd"

    def test_new_isaac_stage(self) -> None:
        """new_isaac_stage creates a blank stage."""
        data = {"created": True, "new_stage": True}
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(tools.new_isaac_stage())
        assert result["success"] is True
        assert result["created"] is True

    def test_import_isaac_asset(self) -> None:
        """import_isaac_asset imports an external asset."""
        data = {"asset_path": "/assets/robot.usd", "target_path": "/World/Robot", "imported": True}
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(
            tools.import_isaac_asset(asset_path="/assets/robot.usd", target_path="/World/Robot")
        )
        assert result["success"] is True
        assert result["imported"] is True
        assert result["target_path"] == "/World/Robot"

    def test_add_isaac_reference(self) -> None:
        """add_isaac_reference adds a USD reference to a prim."""
        data = {"prim_path": "/World/Ref", "reference_path": "/assets/model.usd", "added": True, "total_references": 1}
        tools = _make_tools(execute_return=_make_result(data))
        result = asyncio.run(
            tools.add_isaac_reference(prim_path="/World/Ref", reference_path="/assets/model.usd")
        )
        assert result["success"] is True
        assert result["added"] is True
        assert result["total_references"] == 1
