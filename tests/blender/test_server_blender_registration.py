"""Tests for Blender tool registration inside FastMCP server."""

import asyncio
import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastmcp import Client

from simul.config import Settings
from simul.mcp import backends as backends_module
from simul.mcp import server as server_module
from tests.fakes import BlenderSessionAdapter, FakeFastMCP, StrictBlenderSession


class FakeBlenderAdapter:
    """Minimal Blender adapter stub used during registration tests."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def is_available(self) -> bool:
        """Pretend Blender runtime is available."""
        return True

    def get_capabilities(self):
        """Return predictable Blender capability list."""
        return ["blender_runtime_info"]

    @contextmanager
    def create_session(self):
        """Yield minimal session object."""
        session = SimpleNamespace(
            get_runtime_info=lambda: {
                "version": [4, 1, 0],
                "version_string": "4.1.0",
                "binary_path": "/bin/blender",
                "background": True,
                "blend_file_path": None,
            },
            list_scene_objects=lambda **kwargs: {
                "collection": kwargs.get("collection_name"),
                "include_hidden": kwargs.get("include_hidden", False),
                "max_items": kwargs.get("max_items", 200),
                "count": 0,
                "objects": [],
                "truncated": False,
            },
        )
        yield session


class TestBlenderToolRegistration:
    """Test Blender tool registration behavior in server."""

    def test_server_registers_blender_tools_when_available(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Blender tools are present in FastMCP registration set."""
        monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
        monkeypatch.setattr(server_module, "TaskConfig", None)

        monkeypatch.setattr(backends_module, "is_headless_available", lambda: False)
        monkeypatch.setattr(backends_module, "is_blender_available", lambda: True)
        monkeypatch.setattr(backends_module, "BlenderRuntimeAdapter", FakeBlenderAdapter)

        instance = server_module.SimulMCPServer(settings=Settings())
        tool_names = {tool.name for tool in instance.mcp.tools}

        assert "get_blender_info" in tool_names
        assert "list_blender_scene_objects" in tool_names
        assert "capture_viewport" not in tool_names

    def test_server_skips_blender_tools_when_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Blender tools are not registered when runtime is unavailable."""
        monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
        monkeypatch.setattr(server_module, "TaskConfig", None)

        monkeypatch.setattr(backends_module, "is_headless_available", lambda: False)
        monkeypatch.setattr(backends_module, "is_blender_available", lambda: False)

        instance = server_module.SimulMCPServer(settings=Settings())
        tool_names = {tool.name for tool in instance.mcp.tools}

        assert "get_blender_info" not in tool_names
        assert "list_blender_scene_objects" not in tool_names


_SIMULATED_BLENDER_ERROR = "Simulated runtime metadata fetch failure"


class FakeBlenderAdapterErroring:
    """Blender adapter stub whose adapter calls return failure-shape
    payloads (``{... "error": "..."}``).

    iter17 extended ``apply_success_from_error`` to all 45 sites in
    ``_reg_blender.py`` (44 historical + 1 inline already converted
    in iter16). Most current Blender adapter methods raise on failure
    rather than returning ``error``-keyed payloads — but defensive
    normalization here means any future adapter method that does
    emit ``error`` will surface as ``success: false`` automatically.
    This fake adapter exercises that defensive contract: it returns
    a payload with ``error`` set, and the wrapper must respect it
    instead of clobbering with ``success: true``.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def is_available(self) -> bool:
        return True

    def get_capabilities(self):
        return ["blender_runtime_info"]

    @contextmanager
    def create_session(self):
        session = SimpleNamespace(
            get_runtime_info=lambda: {
                "version": [0, 0, 0],
                "version_string": "",
                "binary_path": "",
                "background": False,
                "blend_file_path": None,
                "error": _SIMULATED_BLENDER_ERROR,
            },
        )
        yield session


class TestIter17WrapperSurfacesAdapterError:
    """End-to-end coverage of the iter17 invariant at the Blender tool layer.

    Mirrors the iter16 ``TestIter16WrapperSurfacesAdapterError`` test
    in ``tests/unreal/test_server_unreal_registration.py``. The 45
    Blender registration sites that previously wrote
    ``payload['success'] = True`` unconditionally have all been
    replaced with ``apply_success_from_error(payload)``. Without an
    integration test, a future revert would be caught only by grep.
    """

    def test_get_blender_info_returns_success_false_on_adapter_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
        monkeypatch.setattr(server_module, "TaskConfig", None)
        monkeypatch.setattr(backends_module, "is_headless_available", lambda: False)
        monkeypatch.setattr(backends_module, "is_blender_available", lambda: True)
        monkeypatch.setattr(
            backends_module, "BlenderRuntimeAdapter", FakeBlenderAdapterErroring
        )

        instance = server_module.SimulMCPServer(settings=Settings())
        tool = next(
            t for t in instance.mcp.tools if t.name == "get_blender_info"
        )

        result = json.loads(asyncio.run(tool.func()).content[0].text)

        # iter17 contract: success accurately reflects the presence of
        # an error in the adapter payload. iter18 added `error:
        # Optional[str]` to all Blender + SimReady response schemas
        # so the diagnostic message now survives Pydantic and reaches
        # the client.
        assert result["success"] is False, (
            f"iter17 invariant broken: registered get_blender_info tool "
            f"returned success={result.get('success')!r} when adapter "
            f"returned an error payload. Full response: {result}"
        )
        assert result.get("error") == _SIMULATED_BLENDER_ERROR, (
            f"iter18 invariant broken: error message did not survive "
            f"Pydantic. Full response: {result}"
        )


# ---------------------------------------------------------------------------
# Each tool reaches the session operation it names
# ---------------------------------------------------------------------------


def _strict_server(
    monkeypatch: pytest.MonkeyPatch, session: StrictBlenderSession, mode: str
) -> server_module.SimulMCPServer:
    """A real FastMCP server whose Blender sessions are ``session``."""
    monkeypatch.setattr(backends_module, "is_blender_available", lambda: True)
    monkeypatch.setattr(
        backends_module,
        "BlenderRuntimeAdapter",
        lambda settings: BlenderSessionAdapter(settings, session),
    )
    settings = Settings(blender={"mode": mode}, security={"rate_limiting_enabled": False})
    return server_module.SimulMCPServer(settings, backends={"blender"})


def _call_through_client(
    instance: server_module.SimulMCPServer, tool: str, arguments: dict
) -> dict:
    async def _run() -> dict:
        async with Client(instance.mcp) as client:
            result = await client.call_tool(tool, arguments)
            return json.loads(result.content[-1].text)

    return asyncio.run(_run())


@pytest.mark.parametrize("mode", ["embedded", "attached"])
def test_check_blender_object_bounds_calls_is_object_within_bounds(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """It called check_object_bounds, which BlenderRuntimeSession never had (since 70a3fbc)."""
    session = StrictBlenderSession(
        responses={
            "is_object_within_bounds": {
                "object_name": "Cube",
                "within_bounds": True,
                "object_location": [0.5, 0.5, 0.5],
            }
        }
    )
    instance = _strict_server(monkeypatch, session, mode)

    payload = _call_through_client(
        instance,
        "check_blender_object_bounds",
        {"object_name": "Cube", "bounds_min": [0, 0, 0], "bounds_max": [1, 1, 1]},
    )

    assert session.rejected == []
    assert session.calls == [
        (
            "is_object_within_bounds",
            {
                "object_name": "Cube",
                "bounds_min": [0.0, 0.0, 0.0],
                "bounds_max": [1.0, 1.0, 1.0],
            },
        )
    ]
    # The relation report is optional: an add-on that predates it still answers.
    assert payload == {
        "success": True,
        "error": None,
        "object_name": "Cube",
        "within_bounds": True,
        "object_location": [0.5, 0.5, 0.5],
        "relation": None,
        "intersects": None,
        "origin_within_bounds": None,
        "object_bbox_min": None,
        "object_bbox_max": None,
    }


@pytest.mark.parametrize("mode", ["embedded", "attached"])
def test_set_blender_light_params_forwards_every_parameter(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """It raised NameError on every call: the line defining ``col`` was dropped in 70a3fbc."""
    session = StrictBlenderSession(
        responses={
            "set_light_params": {
                "light_name": "Light",
                "light_type": "SPOT",
                "energy": 500.0,
                "color": [1.0, 0.5, 0.25],
            }
        }
    )
    instance = _strict_server(monkeypatch, session, mode)

    payload = _call_through_client(
        instance,
        "set_blender_light_params",
        {
            "light_name": "Light",
            "energy": 500,
            "color": [1, 0.5, 0.25],
            "use_shadow": False,
            "spot_size": 0.8,
            "spot_blend": 0.2,
            "shadow_soft_size": 0.1,
        },
    )

    assert session.rejected == []
    assert session.calls == [
        (
            "set_light_params",
            {
                "light_name": "Light",
                "energy": 500.0,
                "color": [1.0, 0.5, 0.25],
                "use_shadow": False,
                "spot_size": 0.8,
                "spot_blend": 0.2,
                "shadow_soft_size": 0.1,
            },
        )
    ]
    assert payload == {
        "success": True,
        "error": None,
        "light_name": "Light",
        "light_type": "SPOT",
        "energy": 500.0,
        "color": [1.0, 0.5, 0.25],
    }
    # Omitted parameters stay None, so the session leaves those properties alone.
    session.calls.clear()
    _call_through_client(instance, "set_blender_light_params", {"light_name": "Light"})
    assert session.calls == [
        (
            "set_light_params",
            {
                "light_name": "Light",
                "energy": None,
                "color": None,
                "use_shadow": None,
                "spot_size": None,
                "spot_blend": None,
                "shadow_soft_size": None,
            },
        )
    ]


# The camera tools' constraint report, as the add-on returns it (the geometry
# fields are kept to what the response schemas require).
_TRACK_TO_REPORT = {
    "placement_overridden": True,
    "override_check": "done",
    "constraints_applied": [{"name": "Track To", "type": "TRACK_TO"}],
    "warning": "Constraints on camera 'Camera' override the requested placement",
}
# For a camera Blender does not evaluate, so nothing could be checked.
_SKIPPED_REPORT = {
    "placement_overridden": None,
    "override_check": "skipped",
    "constraints_applied": [{"name": "Track To", "type": "TRACK_TO"}],
    "warning": "Blender may not evaluate camera 'Camera' for the viewport",
}
_CAMERA_REPLIES = {
    "set_blender_camera_view": (
        "set_camera_view",
        {"location": [7, -7, 5], "rotation_euler": [1.1, 0.25, 0.78]},
        {
            "camera_name": "Camera",
            "location": [7.0, -7.0, 5.0],
            "rotation_euler": [1.2, 0.0, 0.08],
        },
    ),
    "focus_blender_on_object": (
        "focus_on_object",
        {"object_name": "Cube"},
        {
            "camera_name": "Camera",
            "object_name": "Cube",
            "camera_location": [0.0, -6.5, 6.5],
            "look_at": [0.0, 0.0, 0.0],
        },
    ),
}


@pytest.mark.parametrize("tool", list(_CAMERA_REPLIES))
@pytest.mark.parametrize("addon", ["current", "predating_the_report"])
def test_camera_tools_relay_the_constraint_report_and_accept_an_older_addon(
    monkeypatch: pytest.MonkeyPatch, tool: str, addon: str
) -> None:
    """The report reaches the caller; an add-on built before it still succeeds, with nulls.

    The response schema drops undeclared fields, so the report has to be
    declared there, and optional: the add-on is installed separately and can
    be older than the server.
    """
    operation, arguments, reply = _CAMERA_REPLIES[tool]
    current = addon == "current"
    session = StrictBlenderSession(
        responses={operation: {**reply, **(_TRACK_TO_REPORT if current else {})}}
    )
    instance = _strict_server(monkeypatch, session, "attached")

    payload = _call_through_client(instance, tool, arguments)

    assert session.rejected == []
    assert payload["success"] is True
    expected = _TRACK_TO_REPORT if current else dict.fromkeys(_TRACK_TO_REPORT)
    assert {key: payload[key] for key in _TRACK_TO_REPORT} == expected


@pytest.mark.parametrize("tool", list(_CAMERA_REPLIES))
def test_camera_tools_relay_a_skipped_override_check(
    monkeypatch: pytest.MonkeyPatch, tool: str
) -> None:
    """override_check reaches the caller, and placement_overridden stays null, not false."""
    operation, arguments, reply = _CAMERA_REPLIES[tool]
    session = StrictBlenderSession(responses={operation: {**reply, **_SKIPPED_REPORT}})
    instance = _strict_server(monkeypatch, session, "attached")

    payload = _call_through_client(instance, tool, arguments)

    assert session.rejected == []
    assert payload["success"] is True
    assert {key: payload[key] for key in _SKIPPED_REPORT} == _SKIPPED_REPORT


# For a camera Blender does not evaluate, whose pose was composed from its transform.
_COMPOSED_POSE_REPORT = {
    "pose_source": "transform",
    "warning": "Blender may not evaluate camera 'Camera' for the viewport",
}


@pytest.mark.parametrize(
    "addon", ["current", "predating_the_pose_report", "predating_the_report"]
)
def test_get_blender_camera_info_relays_the_active_constraints_and_pose_source(
    monkeypatch: pytest.MonkeyPatch, addon: str
) -> None:
    """Both reports reach the caller; an add-on built before either still succeeds, with nulls.

    The response schema drops undeclared fields, so pose_source and warning
    have to be declared there, and optional: the add-on is installed
    separately and can be older than the server.
    """
    constraints = [{"name": "Track To", "type": "TRACK_TO"}]
    reply = {
        "camera_name": "Camera",
        "location": [7.0, -7.0, 5.0],
        "rotation_euler": [1.2, 0.0, 0.08],
        "lens": 50.0,
        "sensor_width": 36.0,
        "clip_start": 0.1,
        "clip_end": 100.0,
        "camera_type": "PERSP",
    }
    if addon != "predating_the_report":
        reply["constraints_applied"] = constraints
    if addon == "current":
        reply.update(_COMPOSED_POSE_REPORT)
    session = StrictBlenderSession(responses={"get_camera_info": reply})
    instance = _strict_server(monkeypatch, session, "attached")

    payload = _call_through_client(instance, "get_blender_camera_info", {})

    assert session.rejected == []
    assert payload["success"] is True
    assert payload["constraints_applied"] == (
        None if addon == "predating_the_report" else constraints
    )
    expected = (
        _COMPOSED_POSE_REPORT
        if addon == "current"
        else dict.fromkeys(_COMPOSED_POSE_REPORT)
    )
    assert {key: payload[key] for key in _COMPOSED_POSE_REPORT} == expected


class _AddonBeforeTheTransformFixes:
    """The session of an add-on built before set_object_parent's keep_transform
    and set_object_transform's rotation_quaternion: it refuses either keyword,
    as the add-on's dispatcher binds each call to its own signature."""

    def __init__(self) -> None:
        self.calls: list = []

    def set_object_parent(self, child_name: str, parent_name: str) -> dict:
        self.calls.append(("set_object_parent", child_name, parent_name))
        return {"child_name": child_name, "parent_name": parent_name}

    def set_object_transform(
        self, object_name: str, location=None, rotation_euler=None, scale=None
    ) -> dict:
        self.calls.append(("set_object_transform", object_name, rotation_euler))
        return {
            "object_name": object_name,
            "location": [0.0, 0.0, 0.0],
            "rotation_euler": rotation_euler or [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
        }


def _call(instance: server_module.SimulMCPServer, tool: str, arguments: dict) -> dict:
    async def _run() -> dict:
        async with Client(instance.mcp) as client:
            result = await client.call_tool(tool, arguments, raise_on_error=False)
            return json.loads(result.content[-1].text)

    return asyncio.run(_run())


def test_an_older_addon_still_answers_calls_that_leave_the_new_parameters_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A parameter at its default is not sent, so the add-on's own default applies.

    The add-on is installed separately and can be older than the server; sent
    always, the new keywords made it refuse every parenting and transform call.
    A call that does need the new parameter still fails, naming it.
    """
    session = _AddonBeforeTheTransformFixes()
    instance = _strict_server(monkeypatch, session, "attached")  # type: ignore[arg-type]

    parented = _call(
        instance, "set_blender_object_parent", {"child_name": "C", "parent_name": "P"}
    )
    turned = _call(
        instance,
        "set_blender_object_transform",
        {"object_name": "Cube", "rotation_euler": [0, 0, 1]},
    )
    unsupported = _call(
        instance,
        "set_blender_object_parent",
        {"child_name": "C", "parent_name": "P", "keep_transform": False},
    )

    assert parented["success"] is True
    assert parented["keep_transform"] is None and parented["world_location"] is None
    assert turned["success"] is True and turned["rotation_mode"] is None
    assert session.calls == [
        ("set_object_parent", "C", "P"),
        ("set_object_transform", "Cube", [0.0, 0.0, 1.0]),
    ]
    assert unsupported["success"] is False
    assert "keep_transform" in unsupported["error"]


class _AddonBeforeTheAnimationFixes:
    """The session of an add-on built before insert_keyframe took value,
    interpolation and keyframe_type: it refuses those keywords."""

    def __init__(self) -> None:
        self.calls: list = []

    def insert_keyframe(
        self, object_name: str, data_path: str, frame: int, index: int = -1
    ) -> dict:
        self.calls.append(("insert_keyframe", object_name, data_path, frame, index))
        return {"object_name": object_name, "data_path": data_path, "frame": frame}


def test_an_older_addon_still_keys_the_current_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sent on every call, value, interpolation and keyframe_type made an add-on
    built before them refuse every insert_blender_keyframe call."""
    session = _AddonBeforeTheAnimationFixes()
    instance = _strict_server(monkeypatch, session, "attached")  # type: ignore[arg-type]

    keyed = _call(
        instance,
        "insert_blender_keyframe",
        {"object_name": "Cube", "data_path": "location", "frame": 10, "index": 2},
    )
    unsupported = _call(
        instance,
        "insert_blender_keyframe",
        {"object_name": "Cube", "data_path": "location", "frame": 10, "value": [1, 2, 3]},
    )

    assert keyed["success"] is True
    assert keyed["value_source"] is None and keyed["keyframes"] == []
    assert session.calls == [("insert_keyframe", "Cube", "location", 10, 2)]
    assert unsupported["success"] is False
    assert "value" in unsupported["error"]


def test_frame_bounds_are_stated_and_checked_inside_the_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The schema states step >= 1 and Blender's frame limits; an argument out
    of them still gets the envelope's ValidationError, recorded and rate limited
    like any other invalid input, and never reaches Blender."""
    session = StrictBlenderSession()
    instance = _strict_server(monkeypatch, session, "attached")

    async def _schemas() -> dict:
        async with Client(instance.mcp) as client:
            return {tool.name: tool.input_schema for tool in await client.list_tools()}

    schemas = asyncio.run(_schemas())
    step = schemas["get_blender_object_trajectory"]["properties"]["step"]
    frame = schemas["set_blender_frame"]["properties"]["frame"]
    start = schemas["set_blender_frame_range"]["properties"]["frame_start"]
    assert step["minimum"] == 1
    assert (frame["minimum"], frame["maximum"]) == (-1_048_574, 1_048_574)
    assert (start["minimum"], start["maximum"]) == (0, 1_048_574)

    refused = [
        _call(
            instance,
            "get_blender_object_trajectory",
            {"object_name": "Cube", "start_frame": 1, "end_frame": 5, "step": 0},
        ),
        _call(instance, "set_blender_frame", {"frame": 10**7}),
        _call(instance, "set_blender_frame_range", {"frame_start": -1, "frame_end": 9}),
    ]

    assert [r["error_type"] for r in refused] == ["ValidationError"] * 3
    assert session.calls == []


class _AddonBeforePaging:
    """The session of an add-on built before listing pages and scene-only delete."""

    def __init__(self) -> None:
        self.calls: list = []

    def list_scene_objects(
        self, collection_name=None, include_hidden=False, max_items=200
    ) -> dict:
        self.calls.append(("list_scene_objects", max_items))
        return {
            "collection": collection_name,
            "include_hidden": include_hidden,
            "max_items": max_items,
            "count": 0,
            "objects": [],
            "truncated": False,
        }

    def search_objects(self, name_pattern=None, object_type=None, max_results=50) -> dict:
        self.calls.append(("search_objects", object_type))
        return {"pattern": name_pattern, "object_type": object_type, "count": 0, "objects": [], "truncated": False}

    def delete_object(self, object_name: str) -> dict:
        self.calls.append(("delete_object", object_name))
        return {"deleted_name": object_name}


def test_an_older_addon_still_lists_searches_and_deletes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """offset and use_global at their defaults are not sent; the reply's new fields are null."""
    session = _AddonBeforePaging()
    instance = _strict_server(monkeypatch, session, "attached")  # type: ignore[arg-type]

    listed = _call(instance, "list_blender_scene_objects", {})
    found = _call(instance, "search_blender_objects", {"object_type": "mesh"})
    deleted = _call(instance, "delete_blender_object", {"object_name": "Cube"})
    paged = _call(instance, "list_blender_scene_objects", {"offset": 200})

    assert listed["success"] is True and listed["total"] is None
    assert found["success"] is True and found["next_offset"] is None
    assert deleted["success"] is True and deleted["remaining_scenes"] is None
    # The server upper-cases the type for an add-on that compared it as given.
    assert session.calls == [
        ("list_scene_objects", 200),
        ("search_objects", "MESH"),
        ("delete_object", "Cube"),
    ]
    assert paged["success"] is False and "offset" in paged["error"]
