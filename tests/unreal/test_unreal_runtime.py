"""Tests for Unreal Engine runtime adapter functionality."""

import asyncio
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
import sys

import pytest

src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_path))

from simul_mcp.adapters import unreal_runtime  # noqa: E402
from simul_mcp.config import Settings  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeResponse:
    """Minimal aiohttp response double."""

    def __init__(self, json_data: Dict[str, Any], status: int = 200):
        self._json_data = json_data
        self.status = status

    async def json(self) -> Dict[str, Any]:
        return self._json_data

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise Exception(f"HTTP {self.status}")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class FakeClientSession:
    """Minimal aiohttp.ClientSession double."""

    def __init__(self, responses: Dict[str, FakeResponse]):
        self._responses = responses
        self.closed = False

    def get(self, path: str) -> FakeResponse:
        return self._responses.get(path, FakeResponse({}, 404))

    def put(self, path: str, json: Any = None) -> FakeResponse:
        return self._responses.get(path, FakeResponse({}, 404))

    def post(self, path: str, json: Any = None) -> FakeResponse:
        return self._responses.get(path, FakeResponse({}, 404))

    async def close(self) -> None:
        self.closed = True


REMOTE_INFO_PAYLOAD: Dict[str, Any] = {
    "EngineVersion": "5.4.0",
    "ProjectName": "TestProject",
    "LoadedMap": "/Game/Maps/TestMap",
    "IsEditor": True,
    "IsGame": False,
    "Platform": "Win64",
}


# ---------------------------------------------------------------------------
# UnrealRuntimeSession tests
# ---------------------------------------------------------------------------


class TestUnrealRuntimeSession:
    """Test cases for UnrealRuntimeSession."""

    def test_session_raises_without_aiohttp(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Session initialization fails when neither aiohttp nor embedded is available."""
        monkeypatch.setattr(unreal_runtime, "UNREAL_AVAILABLE", False)

        with pytest.raises(ImportError, match="Unreal runtime not available"):
            unreal_runtime.UnrealRuntimeSession()

    def _make_session(
        self,
        monkeypatch: pytest.MonkeyPatch,
        responses: Dict[str, FakeResponse] | None = None,
    ) -> unreal_runtime.UnrealRuntimeSession:
        """Create a session with a fake HTTP client injected."""
        monkeypatch.setattr(unreal_runtime, "UNREAL_AVAILABLE", True)
        monkeypatch.setattr(unreal_runtime, "AIOHTTP_AVAILABLE", True)

        session = unreal_runtime.UnrealRuntimeSession(settings=Settings())
        if responses is not None:
            session._session = FakeClientSession(responses)
        return session

    def test_health_check_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Health check returns connected=True with engine info."""
        session = self._make_session(monkeypatch)

        def put_fn(path: str, json: Any = None) -> FakeResponse:
            fn = (json or {}).get("functionName", "")
            if fn == "GetEngineVersion":
                return FakeResponse({"ReturnValue": "5.4.0"})
            if fn == "ExecutePythonCommandEx":
                return FakeResponse({
                    "ReturnValue": True,
                    "CommandResult": "'TestProject'",
                })
            return FakeResponse({}, 404)

        session._session = SmartFakeClientSession(
            get_responses={"/remote/info": FakeResponse(REMOTE_INFO_PAYLOAD)},
            put_fn=put_fn,
        )

        result = asyncio.run(session.health_check())

        assert result["connected"] is True
        assert result["engine_version"] == "5.4.0"
        assert result["project_name"] == "TestProject"
        assert result["is_editor"] is True

    def test_health_check_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Health check returns connected=False on HTTP error."""
        responses = {"/remote/info": FakeResponse({}, status=500)}
        session = self._make_session(monkeypatch, responses)

        result = asyncio.run(session.health_check())

        assert result["connected"] is False
        assert "error" in result

    def test_get_engine_info(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Engine info returns all expected fields."""
        session = self._make_session(monkeypatch)
        # get_engine_info calls: GetEngineVersion + 3× ExecutePythonCommandEx
        python_results = iter([
            "'TestProject'",
            "'/Game/Maps/TestMap'",
            "'Win64'",
        ])

        def put_fn(path: str, json: Any = None) -> FakeResponse:
            fn = (json or {}).get("functionName", "")
            if fn == "GetEngineVersion":
                return FakeResponse({"ReturnValue": "5.4.0"})
            if fn == "ExecutePythonCommandEx":
                return FakeResponse({
                    "ReturnValue": True,
                    "CommandResult": next(python_results),
                })
            return FakeResponse({}, 404)

        session._session = SmartFakeClientSession(put_fn=put_fn)

        result = asyncio.run(session.get_engine_info())

        assert result["engine_version"] == "5.4.0"
        assert result["project_name"] == "TestProject"
        assert result["loaded_map"] == "/Game/Maps/TestMap"
        assert result["is_editor"] is True
        assert result["is_game"] is False
        assert result["platform"] == "Win64"

    def test_get_loaded_map(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Loaded map returns the map path."""
        session = self._make_session(monkeypatch)

        def put_fn(path: str, json: Any = None) -> FakeResponse:
            fn = (json or {}).get("functionName", "")
            if fn == "ExecutePythonCommandEx":
                return FakeResponse({
                    "ReturnValue": True,
                    "CommandResult": "'/Game/Maps/TestMap'",
                })
            return FakeResponse({}, 404)

        session._session = SmartFakeClientSession(put_fn=put_fn)

        result = asyncio.run(session.get_loaded_map())

        assert result["map_path"] == "/Game/Maps/TestMap"

    def test_close_session(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Session close shuts down the underlying HTTP client."""
        fake_client = FakeClientSession({})
        session = self._make_session(monkeypatch, responses=None)
        session._session = fake_client

        asyncio.run(session.close())

        assert fake_client.closed is True
        assert session._session is None

    def test_health_check_missing_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Health check tolerates missing optional fields from the API."""
        session = self._make_session(monkeypatch)

        def put_fn(path: str, json: Any = None) -> FakeResponse:
            fn = (json or {}).get("functionName", "")
            if fn == "GetEngineVersion":
                return FakeResponse({"ReturnValue": "5.3.0"})
            if fn == "ExecutePythonCommandEx":
                return FakeResponse({
                    "ReturnValue": True,
                    "CommandResult": "''",
                })
            return FakeResponse({}, 404)

        sparse_payload: Dict[str, Any] = {"EngineVersion": "5.3.0"}
        session._session = SmartFakeClientSession(
            get_responses={"/remote/info": FakeResponse(sparse_payload)},
            put_fn=put_fn,
        )

        result = asyncio.run(session.health_check())

        assert result["connected"] is True
        assert result["engine_version"] == "5.3.0"
        assert result["project_name"] == ""
        assert result["is_editor"] is True

    def test_health_check_connected_when_metadata_calls_400(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: UE 5.3 gates Default__KismetSystemLibrary and
        Default__PythonScriptLibrary access via Remote Control with 400.
        Connectivity (`/remote/info`) still returns 200 — `connected` must
        stay True so `simul unreal setup` can declare success. Metadata
        fields go empty and warnings list the failures."""
        session = self._make_session(monkeypatch)

        def put_fn(path: str, json: Any = None) -> FakeResponse:
            # UE 5.3 default behavior: any CDO call via /remote/object/call
            # returns 400 with "cannot be accessed remotely".
            return FakeResponse(
                {"errorMessage": "Object cannot be accessed remotely"},
                status=400,
            )

        session._session = SmartFakeClientSession(
            get_responses={"/remote/info": FakeResponse(REMOTE_INFO_PAYLOAD)},
            put_fn=put_fn,
        )

        result = asyncio.run(session.health_check())

        assert result["connected"] is True
        assert result["engine_version"] == ""
        assert result["project_name"] == ""
        assert result["is_editor"] is True
        assert "warnings" in result and len(result["warnings"]) == 2
        assert any("engine_version" in w for w in result["warnings"])
        assert any("project_name" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
# Phase 1 session method tests
# ---------------------------------------------------------------------------


DESCRIBE_ACTOR_PAYLOAD: Dict[str, Any] = {
    "Name": "StaticMeshActor_0",
    "Class": "StaticMeshActor",
    "Components": [
        {"Name": "StaticMeshComponent0", "Class": "StaticMeshComponent", "IsRootComponent": True},
    ],
    "Tags": ["nav_obstacle"],
    "Mobility": "Static",
    "bHidden": False,
}

PROPERTY_LOCATION_PAYLOAD: Dict[str, Any] = {
    "RootComponent.RelativeLocation": {"X": 100.0, "Y": 200.0, "Z": 50.0},
}

PROPERTY_ROTATION_PAYLOAD: Dict[str, Any] = {
    "RootComponent.RelativeRotation": {"Pitch": 0.0, "Yaw": 45.0, "Roll": 0.0},
}

PROPERTY_SCALE_PAYLOAD: Dict[str, Any] = {
    "RootComponent.RelativeScale3D": {"X": 1.0, "Y": 1.0, "Z": 1.0},
}


class SmartFakeClientSession:
    """Fake aiohttp session that dispatches PUT by (url, body) pairs."""

    def __init__(
        self,
        get_responses: Dict[str, FakeResponse] | None = None,
        put_responses: Dict[str, FakeResponse] | None = None,
        put_fn: Any = None,
    ):
        self._get_responses = get_responses or {}
        self._put_responses = put_responses or {}
        self._put_fn = put_fn
        self.closed = False

    def get(self, path: str) -> FakeResponse:
        return self._get_responses.get(path, FakeResponse({}, 404))

    def put(self, path: str, json: Any = None) -> FakeResponse:
        if self._put_fn:
            return self._put_fn(path, json)
        return self._put_responses.get(path, FakeResponse({}, 404))

    def post(self, path: str, json: Any = None) -> FakeResponse:
        return FakeResponse({}, 404)

    async def close(self) -> None:
        self.closed = True


class TestUnrealRuntimeSessionPhase1:
    """Tests for Phase 1 session methods."""

    def _make_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> unreal_runtime.UnrealRuntimeSession:
        monkeypatch.setattr(unreal_runtime, "UNREAL_AVAILABLE", True)
        monkeypatch.setattr(unreal_runtime, "AIOHTTP_AVAILABLE", True)
        return unreal_runtime.UnrealRuntimeSession(settings=Settings())

    # -- search_assets --

    def test_search_assets_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """search_assets returns parsed asset entries."""
        session = self._make_session(monkeypatch)
        session._session = SmartFakeClientSession(put_responses={
            "/remote/search/assets": FakeResponse({
                "Assets": [
                    {"Name": "SM_Chair", "Path": "/Game/Meshes/SM_Chair", "Class": "StaticMesh", "PackagePath": "/Game/Meshes"},
                    {"Name": "M_Wood", "Path": "/Game/Materials/M_Wood", "Class": "Material", "PackagePath": "/Game/Materials"},
                ],
            }),
        })

        result = asyncio.run(session.search_assets(query="chair"))

        assert result["count"] == 2
        assert result["assets"][0]["name"] == "SM_Chair"
        assert result["assets"][1]["class_name"] == "Material"
        assert result["truncated"] is False

    def test_search_assets_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """search_assets returns empty list when no matches."""
        session = self._make_session(monkeypatch)
        session._session = SmartFakeClientSession(put_responses={
            "/remote/search/assets": FakeResponse({"Assets": []}),
        })

        result = asyncio.run(session.search_assets(query="nonexistent"))

        assert result["count"] == 0
        assert result["assets"] == []

    # -- describe_object --

    def test_describe_object_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """describe_object returns properties and functions."""
        session = self._make_session(monkeypatch)
        session._session = SmartFakeClientSession(put_responses={
            "/remote/object/describe": FakeResponse({
                "Class": "StaticMeshActor",
                "Properties": [
                    {"Name": "bCanBeDamaged", "Type": "bool", "Value": False},
                ],
                "Functions": [
                    {"Name": "GetActorLocation"},
                    {"Name": "SetActorLocation"},
                ],
            }),
        })

        result = asyncio.run(session.describe_object("/Game/Maps/Test.Test:PersistentLevel.SM_0"))

        assert result["class_name"] == "StaticMeshActor"
        assert len(result["properties"]) == 1
        assert result["properties"][0]["name"] == "bCanBeDamaged"
        assert result["functions"] == ["GetActorLocation", "SetActorLocation"]

    # -- get_actor_thumbnail --

    def test_get_actor_thumbnail_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_actor_thumbnail returns base64 image data."""
        session = self._make_session(monkeypatch)
        session._session = SmartFakeClientSession(put_responses={
            "/remote/object/thumbnail": FakeResponse({"Thumbnail": "iVBORw0KGgo="}),
        })

        result = asyncio.run(session.get_actor_thumbnail("/Game/Meshes/SM_Chair"))

        assert result["asset_path"] == "/Game/Meshes/SM_Chair"
        assert result["image_base64"] == "iVBORw0KGgo="
        assert result["width"] == 256
        assert result["height"] == 256

    # -- get_actor_info --

    def test_get_actor_info_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_actor_info returns full actor metadata with transform."""
        session = self._make_session(monkeypatch)

        def put_router(path: str, json: Any = None) -> FakeResponse:
            if path == "/remote/object/describe":
                return FakeResponse(DESCRIBE_ACTOR_PAYLOAD)
            if path == "/remote/object/property":
                prop = json.get("propertyName", "") if json else ""
                if "Location" in prop:
                    return FakeResponse(PROPERTY_LOCATION_PAYLOAD)
                if "Rotation" in prop:
                    return FakeResponse(PROPERTY_ROTATION_PAYLOAD)
                if "Scale" in prop:
                    return FakeResponse(PROPERTY_SCALE_PAYLOAD)
            return FakeResponse({}, 404)

        session._session = SmartFakeClientSession(put_fn=put_router)

        result = asyncio.run(session.get_actor_info("/Game/Maps/Test.Test:PersistentLevel.StaticMeshActor_0"))

        assert result["name"] == "StaticMeshActor_0"
        assert result["class_name"] == "StaticMeshActor"
        assert result["location"] == (100.0, 200.0, 50.0)
        assert result["rotation"] == (0.0, 45.0, 0.0)
        assert result["scale"] == (1.0, 1.0, 1.0)
        assert len(result["components"]) == 1
        assert result["components"][0]["is_root"] is True
        assert result["tags"] == ["nav_obstacle"]
        assert result["mobility"] == "Static"

    # -- list_actors --

    def test_list_actors_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """list_actors returns parsed actor entries from level."""
        session = self._make_session(monkeypatch)
        actor_paths = [
            "/Game/Maps/T.T:PersistentLevel.StaticMeshActor_0",
            "/Game/Maps/T.T:PersistentLevel.PointLight_0",
        ]

        def put_router(path: str, json: Any = None) -> FakeResponse:
            if path == "/remote/object/call":
                return FakeResponse({"ReturnValue": actor_paths})
            if path == "/remote/object/describe":
                obj_path = json.get("objectPath", "") if json else ""
                if "StaticMeshActor" in obj_path:
                    return FakeResponse(DESCRIBE_ACTOR_PAYLOAD)
                return FakeResponse({
                    "Name": "PointLight_0",
                    "Class": "PointLight",
                    "Tags": [],
                })
            if path == "/remote/object/property":
                prop = json.get("propertyName", "") if json else ""
                if "Location" in prop:
                    return FakeResponse(PROPERTY_LOCATION_PAYLOAD)
                if "Rotation" in prop:
                    return FakeResponse(PROPERTY_ROTATION_PAYLOAD)
                if "Scale" in prop:
                    return FakeResponse(PROPERTY_SCALE_PAYLOAD)
            return FakeResponse({}, 404)

        session._session = SmartFakeClientSession(put_fn=put_router)

        result = asyncio.run(session.list_actors())

        assert result["count"] == 2
        assert result["truncated"] is False
        assert result["actors"][0]["class_name"] == "StaticMeshActor"
        assert result["actors"][1]["class_name"] == "PointLight"
        assert result["actors"][0]["location"] == (100.0, 200.0, 50.0)

    def test_list_actors_with_class_filter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """list_actors filters actors by class_filter."""
        session = self._make_session(monkeypatch)
        actor_paths = [
            "/Game/Maps/T.T:PersistentLevel.StaticMeshActor_0",
            "/Game/Maps/T.T:PersistentLevel.PointLight_0",
        ]

        def put_router(path: str, json: Any = None) -> FakeResponse:
            if path == "/remote/object/call":
                return FakeResponse({"ReturnValue": actor_paths})
            if path == "/remote/object/describe":
                obj_path = json.get("objectPath", "") if json else ""
                if "StaticMeshActor" in obj_path:
                    return FakeResponse(DESCRIBE_ACTOR_PAYLOAD)
                return FakeResponse({
                    "Name": "PointLight_0",
                    "Class": "PointLight",
                    "Tags": [],
                })
            if path == "/remote/object/property":
                prop = json.get("propertyName", "") if json else ""
                if "Location" in prop:
                    return FakeResponse(PROPERTY_LOCATION_PAYLOAD)
                if "Rotation" in prop:
                    return FakeResponse(PROPERTY_ROTATION_PAYLOAD)
                if "Scale" in prop:
                    return FakeResponse(PROPERTY_SCALE_PAYLOAD)
            return FakeResponse({}, 404)

        session._session = SmartFakeClientSession(put_fn=put_router)

        result = asyncio.run(session.list_actors(class_filter="StaticMeshActor"))

        assert result["count"] == 1
        assert result["actors"][0]["class_name"] == "StaticMeshActor"
        assert result["actors"][0]["name"] == "StaticMeshActor_0"

    # -- summarize_scene --

    def test_summarize_scene_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """summarize_scene returns aggregated scene statistics."""
        session = self._make_session(monkeypatch)
        actor_paths = [
            "/Game/Maps/T.T:PersistentLevel.SM_0",
            "/Game/Maps/T.T:PersistentLevel.SM_1",
            "/Game/Maps/T.T:PersistentLevel.PointLight_0",
        ]

        def put_router(path: str, json: Any = None) -> FakeResponse:
            if path == "/remote/object/call":
                fn = (json or {}).get("functionName", "")
                if fn == "ExecutePythonCommandEx":
                    return FakeResponse({
                        "ReturnValue": True,
                        "CommandResult": "'/Game/Maps/TestMap'",
                    })
                if fn == "GetAllLevelActors":
                    return FakeResponse({"ReturnValue": actor_paths})
                return FakeResponse({})
            if path == "/remote/object/describe":
                obj_path = json.get("objectPath", "") if json else ""
                if "PointLight" in obj_path:
                    return FakeResponse({"Class": "PointLight"})
                return FakeResponse({"Class": "StaticMeshActor"})
            return FakeResponse({}, 404)

        session._session = SmartFakeClientSession(
            get_responses={"/remote/info": FakeResponse(REMOTE_INFO_PAYLOAD)},
            put_fn=put_router,
        )

        result = asyncio.run(session.summarize_scene())

        assert result["map_path"] == "/Game/Maps/TestMap"
        assert result["total_actors"] == 3
        assert result["static_meshes"] == 2
        assert result["lights"] == 1
        assert result["cameras"] == 0
        assert result["actor_class_counts"]["StaticMeshActor"] == 2
        assert result["actor_class_counts"]["PointLight"] == 1
        assert "Map: /Game/Maps/TestMap" in result["summary_text"]


class TestUnrealRuntimeSessionPhase2:
    """Tests for Phase 2 viewport session methods."""

    def _make_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> unreal_runtime.UnrealRuntimeSession:
        monkeypatch.setattr(unreal_runtime, "UNREAL_AVAILABLE", True)
        monkeypatch.setattr(unreal_runtime, "AIOHTTP_AVAILABLE", True)
        return unreal_runtime.UnrealRuntimeSession(settings=Settings())

    # -- capture_viewport --

    def test_capture_viewport_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """capture_viewport fires HighResShot via RC's ExecuteConsoleCommand
        and reads the result from a follow-up ExecutePythonCommandEx call,
        parsing the marker-prefixed base64 payload out of LogOutput."""
        session = self._make_session(monkeypatch)

        def put_fn(path: str, json: Any = None) -> FakeResponse:
            fn = (json or {}).get("functionName", "")
            if fn == "ExecuteConsoleCommand":
                # RC ack — fire-and-forget for HighResShot.
                return FakeResponse({})
            if fn == "ExecutePythonCommandEx":
                return FakeResponse({
                    "ReturnValue": True,
                    "LogOutput": [
                        # A leading info line should be ignored; the marker line wins.
                        {"Type": "Info", "Output": "LogPython: capture starting"},
                        {"Type": "Info", "Output": "@@SIMUL_SCREENSHOT@@iVBOR=="},
                    ],
                })
            return FakeResponse({}, 404)

        session._session = SmartFakeClientSession(put_fn=put_fn)

        result = asyncio.run(session.capture_viewport(
            resolution_x=1280, resolution_y=720, format="jpeg"
        ))

        assert result["image_base64"] == "iVBOR=="
        assert result["resolution_x"] == 1280
        assert result["resolution_y"] == 720
        assert result["format"] == "jpeg"

    def test_capture_viewport_no_screenshot_data(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """capture_viewport returns empty base64 when no screenshot data
        available. The adapter retries the read script with asyncio.sleep
        between attempts; we patch sleep here so the test doesn't burn
        15 s waiting for the deadline to elapse."""
        session = self._make_session(monkeypatch)

        async def _instant_sleep(_seconds: float) -> None:
            return None

        monkeypatch.setattr(unreal_runtime.asyncio, "sleep", _instant_sleep)

        # Drive the time clock forward fast so the adapter's deadline check
        # exits quickly even with sleep stubbed to 0.
        loop_time = [0.0]

        class _FakeLoop:
            def time(self) -> float:
                loop_time[0] += 1.0
                return loop_time[0]

        monkeypatch.setattr(
            unreal_runtime.asyncio, "get_event_loop", lambda: _FakeLoop()
        )

        def put_fn(path: str, json: Any = None) -> FakeResponse:
            fn = (json or {}).get("functionName", "")
            if fn == "ExecuteConsoleCommand":
                return FakeResponse({})
            if fn == "ExecutePythonCommandEx":
                # Marker present but with an empty payload: the read script
                # ran but the screenshot wasn't written. Adapter should keep
                # retrying until the deadline.
                return FakeResponse({
                    "ReturnValue": True,
                    "LogOutput": [
                        {"Type": "Info", "Output": "@@SIMUL_SCREENSHOT@@"},
                    ],
                })
            return FakeResponse({}, 404)

        session._session = SmartFakeClientSession(put_fn=put_fn)

        result = asyncio.run(session.capture_viewport())

        assert result["image_base64"] == ""
        assert result["resolution_x"] == 1920
        assert result["resolution_y"] == 1080
        assert result["format"] == "png"

    # -- get_viewport_info --

    def test_get_viewport_info_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_viewport_info returns camera location, rotation and viewport data."""
        session = self._make_session(monkeypatch)

        def put_fn(path: str, json: Any = None) -> FakeResponse:
            fn = (json or {}).get("functionName", "")
            if fn == "GetLevelViewportCameraInfo":
                return FakeResponse({
                    "CameraLocation": {"X": 100.0, "Y": 200.0, "Z": 300.0},
                    "CameraRotation": {"Pitch": -15.0, "Yaw": 45.0, "Roll": 0.0},
                })
            return FakeResponse({}, 404)

        session._session = SmartFakeClientSession(put_fn=put_fn)

        result = asyncio.run(session.get_viewport_info())

        assert result["camera_location"] == (100.0, 200.0, 300.0)
        assert result["camera_rotation"] == (-15.0, 45.0, 0.0)
        assert result["viewport_size"] == (1920, 1080)
        assert result["fov"] == 90.0
        assert result["projection_type"] == "Perspective"

    # -- set_camera_view --

    def test_set_camera_view_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """set_camera_view sends SetLevelViewportCameraInfo and returns applied state."""
        session = self._make_session(monkeypatch)
        calls_made: list = []

        def put_fn(path: str, json: Any = None) -> FakeResponse:
            fn = (json or {}).get("functionName", "")
            calls_made.append(fn)
            if fn == "SetLevelViewportCameraInfo":
                return FakeResponse({})
            return FakeResponse({}, 404)

        session._session = SmartFakeClientSession(put_fn=put_fn)

        result = asyncio.run(session.set_camera_view(
            location=(500.0, 600.0, 700.0),
            rotation=(-30.0, 90.0, 0.0),
            fov=75.0,
        ))

        assert result["location"] == (500.0, 600.0, 700.0)
        assert result["rotation"] == (-30.0, 90.0, 0.0)
        assert result["fov"] == 75.0
        assert "SetLevelViewportCameraInfo" in calls_made

    def test_set_camera_view_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """set_camera_view with no params returns default values and skips HTTP call."""
        session = self._make_session(monkeypatch)
        calls_made: list = []

        def put_fn(path: str, json: Any = None) -> FakeResponse:
            calls_made.append((json or {}).get("functionName", ""))
            return FakeResponse({})

        session._session = SmartFakeClientSession(put_fn=put_fn)

        result = asyncio.run(session.set_camera_view())

        assert result["location"] == (0.0, 0.0, 0.0)
        assert result["rotation"] == (0.0, 0.0, 0.0)
        assert result["fov"] == 90.0
        # No SetLevelViewportCameraInfo call since no params provided
        assert "SetLevelViewportCameraInfo" not in calls_made

    # -- focus_on_actor --

    def test_focus_on_actor_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """focus_on_actor selects actor, positions camera, and reads back camera."""
        session = self._make_session(monkeypatch)
        calls_made: list = []

        def put_fn(path: str, json: Any = None) -> FakeResponse:
            # _get_actor_transform reads properties via /remote/object/property
            if path == "/remote/object/property":
                prop = (json or {}).get("propertyName", "")
                if "RelativeLocation" in prop:
                    return FakeResponse({
                        "RootComponent.RelativeLocation": {
                            "X": 100.0, "Y": 200.0, "Z": 50.0,
                        },
                    })
                if "RelativeRotation" in prop:
                    return FakeResponse({
                        "RootComponent.RelativeRotation": {
                            "Pitch": 0.0, "Yaw": 45.0, "Roll": 0.0,
                        },
                    })
                if "RelativeScale3D" in prop:
                    return FakeResponse({
                        "RootComponent.RelativeScale3D": {
                            "X": 1.0, "Y": 1.0, "Z": 1.0,
                        },
                    })
                return FakeResponse({})
            # /remote/object/call dispatched by functionName
            fn = (json or {}).get("functionName", "")
            calls_made.append(fn)
            if fn == "SetActorSelectionState":
                return FakeResponse({})
            if fn == "SetLevelViewportCameraInfo":
                return FakeResponse({})
            if fn == "GetLevelViewportCameraInfo":
                return FakeResponse({
                    "CameraLocation": {"X": 150.0, "Y": 250.0, "Z": 350.0},
                    "CameraRotation": {"Pitch": -20.0, "Yaw": 60.0, "Roll": 0.0},
                })
            return FakeResponse({})

        session._session = SmartFakeClientSession(put_fn=put_fn)

        result = asyncio.run(session.focus_on_actor(
            actor_path="/Game/Maps/TestMap.TestMap:PersistentLevel.SM_Chair_1"
        ))

        assert result["actor_path"] == "/Game/Maps/TestMap.TestMap:PersistentLevel.SM_Chair_1"
        assert result["camera_location"] == (150.0, 250.0, 350.0)
        assert result["camera_rotation"] == (-20.0, 60.0, 0.0)
        assert "SetActorSelectionState" in calls_made
        assert "SetLevelViewportCameraInfo" in calls_made
        assert "GetLevelViewportCameraInfo" in calls_made


class TestUnrealRuntimeAdapter:
    """Test cases for UnrealRuntimeAdapter."""

    def test_is_available_when_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Adapter reports available when aiohttp present and config enabled."""
        monkeypatch.setattr(unreal_runtime, "UNREAL_AVAILABLE", True)
        settings = Settings(unreal={"enabled": True})
        adapter = unreal_runtime.UnrealRuntimeAdapter(settings=settings)

        assert adapter.is_available() is True

    def test_is_available_when_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Adapter reports unavailable when config disabled."""
        monkeypatch.setattr(unreal_runtime, "UNREAL_AVAILABLE", True)
        settings = Settings(unreal={"enabled": False})
        adapter = unreal_runtime.UnrealRuntimeAdapter(settings=settings)

        assert adapter.is_available() is False

    def test_is_available_no_aiohttp(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Adapter reports unavailable when aiohttp is not installed."""
        monkeypatch.setattr(unreal_runtime, "UNREAL_AVAILABLE", False)
        settings = Settings()
        adapter = unreal_runtime.UnrealRuntimeAdapter(settings=settings)

        assert adapter.is_available() is False

    def test_get_capabilities_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Capabilities include Phase 0 endpoints when available."""
        monkeypatch.setattr(unreal_runtime, "UNREAL_AVAILABLE", True)
        settings = Settings(unreal={"enabled": True})
        adapter = unreal_runtime.UnrealRuntimeAdapter(settings=settings)

        caps = adapter.get_capabilities()

        assert "unreal_health_check" in caps
        assert "get_unreal_engine_info" in caps
        assert "get_unreal_loaded_map" in caps
        assert len(caps) == 53

    def test_get_capabilities_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Capabilities list is empty when adapter not available."""
        monkeypatch.setattr(unreal_runtime, "UNREAL_AVAILABLE", False)
        adapter = unreal_runtime.UnrealRuntimeAdapter(settings=Settings())

        assert adapter.get_capabilities() == []

    def test_create_session_yields_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Adapter create_session yields an UnrealRuntimeSession instance."""
        monkeypatch.setattr(unreal_runtime, "UNREAL_AVAILABLE", True)
        monkeypatch.setattr(unreal_runtime, "AIOHTTP_AVAILABLE", True)

        adapter = unreal_runtime.UnrealRuntimeAdapter(settings=Settings())
        with adapter.create_session() as session:
            assert isinstance(session, unreal_runtime.UnrealRuntimeSession)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


class TestModuleLevelHelpers:
    """Test module-level convenience functions."""

    def test_is_unreal_available_true(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(unreal_runtime, "UNREAL_AVAILABLE", True)
        assert unreal_runtime.is_unreal_available() is True

    def test_is_unreal_available_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(unreal_runtime, "UNREAL_AVAILABLE", False)
        assert unreal_runtime.is_unreal_available() is False

    def test_create_unreal_session_returns_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(unreal_runtime, "UNREAL_AVAILABLE", True)
        monkeypatch.setattr(unreal_runtime, "AIOHTTP_AVAILABLE", True)

        session = unreal_runtime.create_unreal_session(settings=Settings())
        assert isinstance(session, unreal_runtime.UnrealRuntimeSession)



class TestUnrealRuntimeSessionPhase3:
    """Tests for Phase 3 scene manipulation session methods."""

    def _make_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> unreal_runtime.UnrealRuntimeSession:
        monkeypatch.setattr(unreal_runtime, "UNREAL_AVAILABLE", True)
        monkeypatch.setattr(unreal_runtime, "AIOHTTP_AVAILABLE", True)
        return unreal_runtime.UnrealRuntimeSession(settings=Settings())

    # -- spawn_actor --

    def test_spawn_actor_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """spawn_actor returns actor path and class."""
        session = self._make_session(monkeypatch)

        def put_fn(path: str, json: Any = None) -> FakeResponse:
            fn = (json or {}).get("functionName", "")
            if fn == "SpawnActorFromClass":
                return FakeResponse({
                    "ReturnValue": "/Game/Maps/Test.Test:PersistentLevel.Cube_0"
                })
            if path == "/remote/object/property":
                return FakeResponse({})
            return FakeResponse({}, 404)

        session._session = SmartFakeClientSession(put_fn=put_fn)

        result = asyncio.run(session.spawn_actor(
            asset_path="/Script/Engine.StaticMeshActor",
            location=(100.0, 200.0, 300.0),
            rotation=(0.0, 45.0, 0.0),
            label="TestCube",
        ))

        assert result["actor_path"] == "/Game/Maps/Test.Test:PersistentLevel.Cube_0"
        assert "actor_class" in result
        assert result["location"] == (100.0, 200.0, 300.0)

    # -- delete_actor --

    def test_delete_actor_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """delete_actor returns deleted=True."""
        session = self._make_session(monkeypatch)

        def put_fn(path: str, json: Any = None) -> FakeResponse:
            fn = (json or {}).get("functionName", "")
            if fn == "DestroyActor":
                return FakeResponse({"ReturnValue": True})
            return FakeResponse({}, 404)

        session._session = SmartFakeClientSession(put_fn=put_fn)

        result = asyncio.run(session.delete_actor(
            actor_path="/Game/Maps/Test.Test:PersistentLevel.Cube_0"
        ))

        assert result["deleted"] is True
        assert result["actor_path"] == "/Game/Maps/Test.Test:PersistentLevel.Cube_0"

    # -- set_actor_transform --

    def test_set_actor_transform_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """set_actor_transform calls K2_SetActorLocation/Rotation and SetActorScale3D."""
        session = self._make_session(monkeypatch)
        called_functions: list = []

        def put_fn(path: str, json: Any = None) -> FakeResponse:
            fn = (json or {}).get("functionName", "")
            called_functions.append(fn)
            if fn in ("K2_SetActorLocation", "K2_SetActorRotation", "SetActorScale3D"):
                return FakeResponse({"ReturnValue": True})
            return FakeResponse({}, 404)

        session._session = SmartFakeClientSession(put_fn=put_fn)

        result = asyncio.run(session.set_actor_transform(
            actor_path="/Game/Maps/Test.Test:PersistentLevel.Cube_0",
            location=(100.0, 200.0, 50.0),
            rotation=(10.0, 20.0, 30.0),
            scale=(2.0, 2.0, 2.0),
        ))

        assert result["actor_path"] == "/Game/Maps/Test.Test:PersistentLevel.Cube_0"
        assert "K2_SetActorLocation" in called_functions
        assert "K2_SetActorRotation" in called_functions
        assert "SetActorScale3D" in called_functions

    # -- set_actor_property --

    def test_set_actor_property_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """set_actor_property writes a named property."""
        session = self._make_session(monkeypatch)

        def put_fn(path: str, json: Any = None) -> FakeResponse:
            if path == "/remote/object/property":
                return FakeResponse({})
            return FakeResponse({}, 404)

        session._session = SmartFakeClientSession(put_fn=put_fn)

        result = asyncio.run(session.set_actor_property(
            actor_path="/Game/Maps/Test.Test:PersistentLevel.Cube_0",
            property_name="Mobility",
            property_value='"Movable"',
        ))

        assert result["actor_path"] == "/Game/Maps/Test.Test:PersistentLevel.Cube_0"
        assert result["property_name"] == "Mobility"

    # -- call_actor_function --

    def test_call_actor_function_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """call_actor_function invokes a UFUNCTION and returns result."""
        session = self._make_session(monkeypatch)

        def put_fn(path: str, json: Any = None) -> FakeResponse:
            if path == "/remote/object/call":
                return FakeResponse({"ReturnValue": 42})
            return FakeResponse({}, 404)

        session._session = SmartFakeClientSession(put_fn=put_fn)

        result = asyncio.run(session.call_actor_function(
            actor_path="/Game/Maps/Test.Test:PersistentLevel.Cube_0",
            function_name="GetActorBounds",
        ))

        assert result["actor_path"] == "/Game/Maps/Test.Test:PersistentLevel.Cube_0"
        assert result["function_name"] == "GetActorBounds"
        assert result["return_value"] == '{"ReturnValue": 42}'

    # -- set_actor_parent --

    def test_set_actor_parent_attach(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """set_actor_parent attaches child to parent."""
        session = self._make_session(monkeypatch)

        def put_fn(path: str, json: Any = None) -> FakeResponse:
            fn = (json or {}).get("functionName", "")
            if fn == "K2_AttachToActor":
                return FakeResponse({})
            return FakeResponse({}, 404)

        session._session = SmartFakeClientSession(put_fn=put_fn)

        result = asyncio.run(session.set_actor_parent(
            actor_path="/Game/Maps/Test.Test:PersistentLevel.Child_0",
            parent_path="/Game/Maps/Test.Test:PersistentLevel.Parent_0",
        ))

        assert result["actor_path"] == "/Game/Maps/Test.Test:PersistentLevel.Child_0"
        assert result["parent_path"] == "/Game/Maps/Test.Test:PersistentLevel.Parent_0"

    def test_set_actor_parent_detach(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """set_actor_parent detaches when parent_path is None."""
        session = self._make_session(monkeypatch)

        def put_fn(path: str, json: Any = None) -> FakeResponse:
            fn = (json or {}).get("functionName", "")
            if fn == "K2_DetachFromActor":
                return FakeResponse({})
            return FakeResponse({}, 404)

        session._session = SmartFakeClientSession(put_fn=put_fn)

        result = asyncio.run(session.set_actor_parent(
            actor_path="/Game/Maps/Test.Test:PersistentLevel.Child_0",
            parent_path=None,
        ))

        assert result["actor_path"] == "/Game/Maps/Test.Test:PersistentLevel.Child_0"
        assert result["parent_path"] is None

    # -- add_component --

    def test_add_component_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """add_component creates and returns component path."""
        session = self._make_session(monkeypatch)

        def put_fn(path: str, json: Any = None) -> FakeResponse:
            fn = (json or {}).get("functionName", "")
            if fn == "AddComponentByClass":
                return FakeResponse({
                    "ReturnValue": "/Game/Maps/Test.Test:PersistentLevel.Cube_0.PointLight_0"
                })
            return FakeResponse({}, 404)

        session._session = SmartFakeClientSession(put_fn=put_fn)

        result = asyncio.run(session.add_component(
            actor_path="/Game/Maps/Test.Test:PersistentLevel.Cube_0",
            component_class="PointLightComponent",
            component_name="PointLight_0",
        ))

        assert result["actor_path"] == "/Game/Maps/Test.Test:PersistentLevel.Cube_0"
        assert result["component_class"] == "PointLightComponent"
        assert "PointLight_0" in result["component_path"]

    # -- set_actor_visibility --

    def test_set_actor_visibility_hide(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """set_actor_visibility can hide an actor."""
        session = self._make_session(monkeypatch)

        def put_fn(path: str, json: Any = None) -> FakeResponse:
            fn = (json or {}).get("functionName", "")
            if fn == "SetActorHiddenInGame":
                return FakeResponse({})
            return FakeResponse({}, 404)

        session._session = SmartFakeClientSession(put_fn=put_fn)

        result = asyncio.run(session.set_actor_visibility(
            actor_path="/Game/Maps/Test.Test:PersistentLevel.Cube_0",
            visible=False,
            propagate=True,
        ))

        assert result["actor_path"] == "/Game/Maps/Test.Test:PersistentLevel.Cube_0"
        assert result["visible"] is False

    def test_set_actor_visibility_show(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """set_actor_visibility can show an actor."""
        session = self._make_session(monkeypatch)

        def put_fn(path: str, json: Any = None) -> FakeResponse:
            fn = (json or {}).get("functionName", "")
            if fn == "SetActorHiddenInGame":
                return FakeResponse({})
            return FakeResponse({}, 404)

        session._session = SmartFakeClientSession(put_fn=put_fn)

        result = asyncio.run(session.set_actor_visibility(
            actor_path="/Game/Maps/Test.Test:PersistentLevel.Cube_0",
            visible=True,
        ))

        assert result["visible"] is True


class TestUnrealRuntimeSessionPhase4:
    """Phase 4 — Materials, Lighting & Rendering tests."""

    def _make_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> unreal_runtime.UnrealRuntimeSession:
        monkeypatch.setattr(unreal_runtime, "UNREAL_AVAILABLE", True)
        monkeypatch.setattr(unreal_runtime, "AIOHTTP_AVAILABLE", True)
        return unreal_runtime.UnrealRuntimeSession(settings=Settings())

    # -- get_material_info --

    def test_get_material_info_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_material_info returns parameters list."""
        session = self._make_session(monkeypatch)

        def put_fn(path: str, json: Any = None) -> FakeResponse:
            if "/remote/object/describe" in path:
                return FakeResponse({
                    "Parent": "/Game/Materials/M_Base",
                    "Properties": [
                        {"Name": "Roughness", "Type": "Float", "DefaultValue": 0.5},
                        {"Name": "BaseColor", "Type": "Vector", "DefaultValue": None},
                    ],
                })
            return FakeResponse({}, 404)

        session._session = SmartFakeClientSession(put_fn=put_fn)

        result = asyncio.run(session.get_material_info(
            material_path="/Game/Materials/MI_Custom",
        ))

        assert result["material_path"] == "/Game/Materials/MI_Custom"
        assert result["parent_path"] == "/Game/Materials/M_Base"
        assert len(result["parameters"]) == 2
        assert result["parameters"][0]["name"] == "Roughness"
        assert result["parameters"][0]["param_type"] == "scalar"

    # -- set_material_params --

    def test_set_material_params_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """set_material_params sets scalar, vector, texture params."""
        session = self._make_session(monkeypatch)
        call_count = {"n": 0}

        def put_fn(path: str, json: Any = None) -> FakeResponse:
            if "/remote/object/property" in path:
                call_count["n"] += 1
                return FakeResponse({})
            return FakeResponse({}, 404)

        session._session = SmartFakeClientSession(put_fn=put_fn)

        result = asyncio.run(session.set_material_params(
            material_path="/Game/Materials/MI_Custom",
            scalar_params={"Roughness": 0.3, "Metallic": 1.0},
            vector_params={"BaseColor": [0.8, 0.2, 0.1, 1.0]},
            texture_params={"NormalMap": "/Game/Textures/T_Normal"},
        ))

        assert result["material_path"] == "/Game/Materials/MI_Custom"
        assert result["params_set"] == 4  # 2 scalar + 1 vector + 1 texture
        assert call_count["n"] == 4

    # -- create_material_instance --

    def test_create_material_instance_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """create_material_instance returns new instance path."""
        session = self._make_session(monkeypatch)

        def put_fn(path: str, json: Any = None) -> FakeResponse:
            fn = (json or {}).get("functionName", "")
            if fn == "DuplicateAsset":
                return FakeResponse({"ReturnValue": True})
            return FakeResponse({}, 404)

        session._session = SmartFakeClientSession(put_fn=put_fn)

        result = asyncio.run(session.create_material_instance(
            parent_path="/Game/Materials/M_Base",
            instance_name="MI_Whale",
        ))

        assert result["parent_path"] == "/Game/Materials/M_Base"
        assert "MI_Whale" in result["instance_path"]

    # -- assign_material --

    def test_assign_material_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """assign_material assigns material to actor."""
        session = self._make_session(monkeypatch)

        def put_fn(path: str, json: Any = None) -> FakeResponse:
            fn = (json or {}).get("functionName", "")
            if fn == "SetMaterial":
                return FakeResponse({})
            return FakeResponse({}, 404)

        session._session = SmartFakeClientSession(put_fn=put_fn)

        result = asyncio.run(session.assign_material(
            actor_path="/Game/Maps/Test.Test:PersistentLevel.Cube_0",
            material_path="/Game/Materials/MI_Whale",
            slot_index=0,
        ))

        assert result["actor_path"] == "/Game/Maps/Test.Test:PersistentLevel.Cube_0"
        assert result["material_path"] == "/Game/Materials/MI_Whale"
        assert result["slot_index"] == 0

    # -- set_light_params --

    def test_set_light_params_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """set_light_params sets multiple light properties."""
        session = self._make_session(monkeypatch)
        call_count = {"n": 0}

        def put_fn(path: str, json: Any = None) -> FakeResponse:
            if "/remote/object/property" in path:
                call_count["n"] += 1
                return FakeResponse({})
            return FakeResponse({}, 404)

        session._session = SmartFakeClientSession(put_fn=put_fn)

        result = asyncio.run(session.set_light_params(
            actor_path="/Game/Maps/Test.Test:PersistentLevel.PointLight_0",
            intensity=5000.0,
            color_r=1.0,
            color_g=0.9,
            color_b=0.8,
            cast_shadows=True,
        ))

        assert result["actor_path"].endswith("PointLight_0")
        assert result["params_set"] == 3  # Intensity + LightColor + CastShadows
        assert call_count["n"] == 3

    # -- set_render_settings --

    def test_set_render_settings_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """set_render_settings applies a console command."""
        session = self._make_session(monkeypatch)

        def put_fn(path: str, json: Any = None) -> FakeResponse:
            fn = (json or {}).get("functionName", "")
            if fn == "ExecuteConsoleCommand":
                return FakeResponse({})
            return FakeResponse({}, 404)

        session._session = SmartFakeClientSession(put_fn=put_fn)

        result = asyncio.run(session.set_render_settings(
            setting_name="r.ScreenPercentage",
            setting_value="100",
        ))

        assert result["setting_name"] == "r.ScreenPercentage"
        assert result["applied"] is True


class TestUnrealRuntimeSessionPhase5:
    """Phase 5 — Physics & Simulation Control tests."""

    def _make_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> unreal_runtime.UnrealRuntimeSession:
        monkeypatch.setattr(unreal_runtime, "UNREAL_AVAILABLE", True)
        monkeypatch.setattr(unreal_runtime, "AIOHTTP_AVAILABLE", True)
        return unreal_runtime.UnrealRuntimeSession(settings=Settings())

    # -- control_simulation --

    def test_control_simulation_start(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """control_simulation start returns playing state."""
        session = self._make_session(monkeypatch)

        def put_fn(path: str, json: Any = None) -> FakeResponse:
            fn = (json or {}).get("functionName", "")
            if fn == "EditorPlaySimulate":
                return FakeResponse({})
            return FakeResponse({}, 404)

        session._session = SmartFakeClientSession(put_fn=put_fn)

        result = asyncio.run(session.control_simulation(action="start"))

        assert result["action"] == "start"
        assert result["state"] == "playing"

    def test_control_simulation_stop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """control_simulation stop returns stopped state."""
        session = self._make_session(monkeypatch)

        def put_fn(path: str, json: Any = None) -> FakeResponse:
            fn = (json or {}).get("functionName", "")
            if fn == "EditorEndPlay":
                return FakeResponse({})
            return FakeResponse({}, 404)

        session._session = SmartFakeClientSession(put_fn=put_fn)

        result = asyncio.run(session.control_simulation(action="stop"))

        assert result["action"] == "stop"
        assert result["state"] == "stopped"

    def test_control_simulation_invalid_action(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """control_simulation raises ValueError for invalid action."""
        session = self._make_session(monkeypatch)
        session._session = SmartFakeClientSession()

        with pytest.raises(ValueError, match="Invalid PIE action"):
            asyncio.run(session.control_simulation(action="invalid"))

    # -- get_simulation_status --

    def test_get_simulation_status(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """get_simulation_status returns PIE state."""
        session = self._make_session(monkeypatch)

        def put_fn(path: str, json: Any = None) -> FakeResponse:
            fn = (json or {}).get("functionName", "")
            if fn == "IsPlayInEditorActive":
                return FakeResponse({"ReturnValue": True})
            if fn == "IsGamePaused":
                return FakeResponse({"ReturnValue": False})
            if fn == "GetGameTimeInSeconds":
                return FakeResponse({"ReturnValue": 5.25})
            return FakeResponse({}, 404)

        session._session = SmartFakeClientSession(put_fn=put_fn)

        result = asyncio.run(session.get_simulation_status())

        assert result["is_playing"] is True
        assert result["is_paused"] is False
        assert result["sim_time"] == 5.25

    # -- enable_physics --

    def test_enable_physics_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """enable_physics enables physics on actor."""
        session = self._make_session(monkeypatch)

        def put_fn(path: str, json: Any = None) -> FakeResponse:
            if "/remote/object/property" in path:
                return FakeResponse({})
            return FakeResponse({}, 404)

        session._session = SmartFakeClientSession(put_fn=put_fn)

        result = asyncio.run(session.enable_physics(
            actor_path="/Game/Maps/Test.Test:PersistentLevel.Cube_0",
            enable=True,
            simulate_physics=True,
        ))

        assert result["actor_path"].endswith("Cube_0")
        assert result["physics_enabled"] is True

    # -- set_collision --

    def test_set_collision_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """set_collision applies preset and enables collision."""
        session = self._make_session(monkeypatch)
        call_count = {"n": 0}

        def put_fn(path: str, json: Any = None) -> FakeResponse:
            call_count["n"] += 1
            return FakeResponse({})

        session._session = SmartFakeClientSession(put_fn=put_fn)

        result = asyncio.run(session.set_collision(
            actor_path="/Game/Maps/Test.Test:PersistentLevel.Cube_0",
            collision_preset="BlockAll",
            collision_enabled=True,
        ))

        assert result["collision_preset"] == "BlockAll"
        assert result["collision_enabled"] is True
        assert call_count["n"] == 2  # preset call + property set

    # -- apply_force --

    def test_apply_force_impulse(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """apply_force applies an impulse."""
        session = self._make_session(monkeypatch)

        def put_fn(path: str, json: Any = None) -> FakeResponse:
            fn = (json or {}).get("functionName", "")
            if fn == "AddImpulse":
                return FakeResponse({})
            return FakeResponse({}, 404)

        session._session = SmartFakeClientSession(put_fn=put_fn)

        result = asyncio.run(session.apply_force(
            actor_path="/Game/Maps/Test.Test:PersistentLevel.Cube_0",
            force_x=0.0,
            force_y=0.0,
            force_z=1000.0,
            is_impulse=True,
        ))

        assert result["force_applied"] is True
        assert result["force_vector"] == [0.0, 0.0, 1000.0]
        assert result["is_impulse"] is True

    # -- set_physics_params --

    def test_set_physics_params_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """set_physics_params sets mass and damping."""
        session = self._make_session(monkeypatch)
        call_count = {"n": 0}

        def put_fn(path: str, json: Any = None) -> FakeResponse:
            if "/remote/object/property" in path:
                call_count["n"] += 1
                return FakeResponse({})
            return FakeResponse({}, 404)

        session._session = SmartFakeClientSession(put_fn=put_fn)

        result = asyncio.run(session.set_physics_params(
            actor_path="/Game/Maps/Test.Test:PersistentLevel.Cube_0",
            mass=50.0,
            linear_damping=0.1,
            enable_gravity=True,
        ))

        assert result["params_set"] == 3  # mass + linear_damping + gravity
        assert call_count["n"] == 3


class TestUnrealCoordinateConversion:
    """Tests for UE5 ↔ USD coordinate conversion static methods."""

    def test_ue_to_usd_location(self) -> None:
        """cm left-hand → m right-hand: scale ÷100, negate Y."""
        x, y, z = unreal_runtime.UnrealRuntimeSession.ue_to_usd_location(
            100.0, 200.0, 300.0
        )
        assert x == pytest.approx(1.0)
        assert y == pytest.approx(-2.0)
        assert z == pytest.approx(3.0)

    def test_usd_to_ue_location(self) -> None:
        """m right-hand → cm left-hand: scale ×100, negate Y."""
        x, y, z = unreal_runtime.UnrealRuntimeSession.usd_to_ue_location(
            1.0, -2.0, 3.0
        )
        assert x == pytest.approx(100.0)
        assert y == pytest.approx(200.0)
        assert z == pytest.approx(300.0)

    def test_ue_to_usd_rotation(self) -> None:
        """Left-hand → right-hand: negate yaw."""
        p, y, r = unreal_runtime.UnrealRuntimeSession.ue_to_usd_rotation(
            10.0, 45.0, 5.0
        )
        assert p == pytest.approx(10.0)
        assert y == pytest.approx(-45.0)
        assert r == pytest.approx(5.0)

    def test_usd_to_ue_rotation(self) -> None:
        """Right-hand → left-hand: negate yaw."""
        p, y, r = unreal_runtime.UnrealRuntimeSession.usd_to_ue_rotation(
            10.0, -45.0, 5.0
        )
        assert p == pytest.approx(10.0)
        assert y == pytest.approx(45.0)
        assert r == pytest.approx(5.0)

    def test_roundtrip_location(self) -> None:
        """UE → USD → UE roundtrip preserves original values."""
        original = (150.0, -300.0, 75.0)
        usd = unreal_runtime.UnrealRuntimeSession.ue_to_usd_location(*original)
        back = unreal_runtime.UnrealRuntimeSession.usd_to_ue_location(*usd)
        assert back[0] == pytest.approx(original[0])
        assert back[1] == pytest.approx(original[1])
        assert back[2] == pytest.approx(original[2])

    def test_roundtrip_rotation(self) -> None:
        """UE → USD → UE roundtrip preserves original values."""
        original = (15.0, 90.0, -10.0)
        usd = unreal_runtime.UnrealRuntimeSession.ue_to_usd_rotation(*original)
        back = unreal_runtime.UnrealRuntimeSession.usd_to_ue_rotation(*usd)
        assert back[0] == pytest.approx(original[0])
        assert back[1] == pytest.approx(original[1])
        assert back[2] == pytest.approx(original[2])


# ---------------------------------------------------------------------------
# Phase 6 session method tests — USD / SimReady Bridge
# ---------------------------------------------------------------------------


class TestUnrealRuntimeSessionPhase6:
    """Tests for Phase 6 (USD / SimReady Bridge) session methods."""

    def _make_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> unreal_runtime.UnrealRuntimeSession:
        monkeypatch.setattr(unreal_runtime, "UNREAL_AVAILABLE", True)
        monkeypatch.setattr(unreal_runtime, "AIOHTTP_AVAILABLE", True)
        return unreal_runtime.UnrealRuntimeSession(settings=Settings())

    def test_import_usd_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """import_usd returns imported assets and actor paths."""
        session = self._make_session(monkeypatch)
        session._session = SmartFakeClientSession(put_responses={
            "/remote/object/call": FakeResponse({
                "ImportedAssets": ["/Game/Imports/Whale"],
                "ActorPaths": ["/Game/Maps/T.T:PersistentLevel.Whale_0"],
                "Warnings": [],
            }),
        })

        result = asyncio.run(session.import_usd(usd_path="/tmp/whale.usd"))

        assert result["imported_assets"] == ["/Game/Imports/Whale"]
        assert result["actor_paths"] == ["/Game/Maps/T.T:PersistentLevel.Whale_0"]
        assert result["warnings"] == []

    def test_export_usd_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """export_usd returns output path and actor count."""
        session = self._make_session(monkeypatch)
        session._session = SmartFakeClientSession(put_responses={
            "/remote/object/call": FakeResponse({
                "FileSizeBytes": 12345,
            }),
        })

        result = asyncio.run(session.export_usd(
            actor_paths=["/Game/Maps/T.T:PersistentLevel.Cube_0"],
            output_path="/tmp/export.usd",
        ))

        assert result["output_path"] == "/tmp/export.usd"
        assert result["actors_exported"] == 1
        assert result["file_size_bytes"] == 12345

    def test_convert_to_simready_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """convert_to_simready returns conversions applied."""
        session = self._make_session(monkeypatch)
        session._session = SmartFakeClientSession(put_responses={
            "/remote/object/call": FakeResponse({
                "ConversionsApplied": ["physics", "collision"],
                "Warnings": ["Scale was adjusted"],
            }),
        })

        result = asyncio.run(session.convert_to_simready(
            usd_path="/tmp/model.usd",
            output_path="/tmp/model_simready.usd",
        ))

        assert result["output_path"] == "/tmp/model_simready.usd"
        assert "physics" in result["conversions_applied"]
        assert len(result["warnings"]) == 1

    def test_validate_simready_asset_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """validate_simready_asset returns validation results."""
        session = self._make_session(monkeypatch)
        session._session = SmartFakeClientSession(put_responses={
            "/remote/object/call": FakeResponse({
                "IsValid": True,
                "CheckResults": {"physics": True, "collision": True},
                "Errors": [],
                "Suggestions": [],
            }),
        })

        result = asyncio.run(session.validate_simready_asset(usd_path="/tmp/asset.usd"))

        assert result["is_valid"] is True
        assert result["checks"]["physics"] is True
        assert result["errors"] == []

    def test_get_interchange_info_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_interchange_info returns pipeline details."""
        session = self._make_session(monkeypatch)
        session._session = SmartFakeClientSession(put_responses={
            "/remote/object/call": FakeResponse({
                "Pipelines": [{"Name": "USD"}],
                "SupportedFormats": ["usd", "usda", "usdc"],
                "Version": "1.2.0",
            }),
        })

        result = asyncio.run(session.get_interchange_info())

        assert len(result["pipelines"]) == 1
        assert "usd" in result["supported_formats"]
        assert result["interchange_version"] == "1.2.0"


# ---------------------------------------------------------------------------
# Phase 7 session method tests — Advanced Agent Tools
# ---------------------------------------------------------------------------


class TestUnrealRuntimeSessionPhase7:
    """Tests for Phase 7 (Advanced Agent Tools) session methods."""

    def _make_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> unreal_runtime.UnrealRuntimeSession:
        monkeypatch.setattr(unreal_runtime, "UNREAL_AVAILABLE", True)
        monkeypatch.setattr(unreal_runtime, "AIOHTTP_AVAILABLE", True)
        return unreal_runtime.UnrealRuntimeSession(settings=Settings())

    def test_batch_operations_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """batch_operations counts successes and failures."""
        session = self._make_session(monkeypatch)
        session._session = SmartFakeClientSession(put_responses={
            "/remote/batch": FakeResponse({
                "Responses": [
                    {"RequestId": "1", "ResponseCode": 200},
                    {"RequestId": "2", "ResponseCode": 500},
                ],
            }),
        })

        ops = [
            {"RequestId": "1", "Url": "/remote/object/call", "Verb": "PUT", "Body": {}},
            {"RequestId": "2", "Url": "/remote/object/call", "Verb": "PUT", "Body": {}},
        ]
        result = asyncio.run(session.batch_operations(operations=ops))

        assert result["total"] == 2
        assert result["succeeded"] == 1
        assert result["failed"] == 1

    def test_generate_procedural_scene_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """generate_procedural_scene returns spawned actors."""
        session = self._make_session(monkeypatch)
        session._session = SmartFakeClientSession(put_responses={
            "/remote/object/call": FakeResponse({
                "ActorsSpawned": ["/Game/Maps/T.T:PersistentLevel.Shelf_0"],
                "TotalSpawned": 1,
            }),
        })

        result = asyncio.run(session.generate_procedural_scene(scene_type="warehouse"))

        assert result["scene_type"] == "warehouse"
        assert result["total_spawned"] == 1


# ---------------------------------------------------------------------------
# Phase 8 session method tests — Geometry & Modeling
# ---------------------------------------------------------------------------


class TestUnrealRuntimeSessionPhase8:
    """Tests for Phase 8 (Geometry & Modeling) session methods."""

    def _make_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> unreal_runtime.UnrealRuntimeSession:
        monkeypatch.setattr(unreal_runtime, "UNREAL_AVAILABLE", True)
        monkeypatch.setattr(unreal_runtime, "AIOHTTP_AVAILABLE", True)
        return unreal_runtime.UnrealRuntimeSession(settings=Settings())

    @staticmethod
    def _python_response(data: dict) -> FakeResponse:
        """Wrap data dict in ExecutePythonCommandEx LogOutput envelope."""
        return FakeResponse({
            "ReturnValue": True,
            "LogOutput": [{"Type": "Info", "Output": json.dumps(data)}],
        })

    def test_generate_mesh_primitive_box(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """generate_mesh_primitive returns actor path and counts."""
        session = self._make_session(monkeypatch)
        session._session = SmartFakeClientSession(put_responses={
            "/remote/object/call": self._python_response({
                "actor_path": "/Game/Maps/T.T:PersistentLevel.DynMesh_box",
                "primitive_type": "box",
                "triangle_count": 12,
                "vertex_count": 8,
            }),
        })

        result = asyncio.run(session.generate_mesh_primitive(
            primitive_type="box",
            dimensions={"width": 100, "height": 100, "depth": 100},
        ))

        assert result["actor_path"].endswith("DynMesh_box")
        assert result["primitive_type"] == "box"
        assert result["triangle_count"] == 12
        assert result["vertex_count"] == 8

    def test_apply_mesh_boolean_subtract(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """apply_mesh_boolean returns result counts for subtract."""
        session = self._make_session(monkeypatch)
        session._session = SmartFakeClientSession(put_responses={
            "/remote/object/call": self._python_response({
                "target_mesh_path": "/Game/Maps/T.T:PersistentLevel.Body",
                "operation": "subtract",
                "result_triangle_count": 200,
                "result_vertex_count": 120,
            }),
        })

        result = asyncio.run(session.apply_mesh_boolean(
            target_mesh_path="/Game/Maps/T.T:PersistentLevel.Body",
            tool_mesh_path="/Game/Maps/T.T:PersistentLevel.Cutter",
            operation="subtract",
        ))

        assert result["operation"] == "subtract"
        assert result["result_triangle_count"] == 200

    def test_compute_convex_hull(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """compute_convex_hull returns hull info."""
        session = self._make_session(monkeypatch)
        session._session = SmartFakeClientSession(put_responses={
            "/remote/object/call": self._python_response({
                "mesh_path": "/Game/Maps/T.T:PersistentLevel.Body",
                "hull_actor_path": "/Game/Maps/T.T:PersistentLevel.Hull_0",
                "hull_vertex_count": 24,
                "hull_triangle_count": 44,
                "source_triangle_count": 100,
            }),
        })

        result = asyncio.run(session.compute_convex_hull(
            mesh_path="/Game/Maps/T.T:PersistentLevel.Body",
        ))

        assert result["hull_vertex_count"] == 24
        assert result["hull_triangle_count"] == 44

    def test_decompose_convex_hull_vhacd(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """decompose_convex_hull returns hull list with V-HACD."""
        session = self._make_session(monkeypatch)
        session._session = SmartFakeClientSession(put_responses={
            "/remote/object/call": self._python_response({
                "mesh_path": "/Game/Maps/T.T:PersistentLevel.Body",
                "hull_count": 2,
                "decomp_actor_path": "/Game/Maps/T.T:PersistentLevel.Decomp_0",
                "total_triangles": 32,
                "total_vertices": 20,
            }),
        })

        result = asyncio.run(session.decompose_convex_hull(
            mesh_path="/Game/Maps/T.T:PersistentLevel.Body",
            max_hulls=4,
        ))

        assert result["hull_count"] == 2
        assert result["total_vertices"] == 20

    def test_subdivide_mesh_catmull_clark(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """subdivide_mesh returns subdivided counts."""
        session = self._make_session(monkeypatch)
        session._session = SmartFakeClientSession(put_responses={
            "/remote/object/call": self._python_response({
                "mesh_path": "/Game/Maps/T.T:PersistentLevel.Body",
                "level": 2,
                "scheme": "catmull_clark",
                "result_triangle_count": 768,
                "result_vertex_count": 386,
                "previous_triangle_count": 192,
            }),
        })

        result = asyncio.run(session.subdivide_mesh(
            mesh_path="/Game/Maps/T.T:PersistentLevel.Body",
            level=2,
            scheme="catmull_clark",
        ))

        assert result["scheme"] == "catmull_clark"
        assert result["level"] == 2
        assert result["result_triangle_count"] == 768

    def test_simplify_mesh_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """simplify_mesh returns reduction ratio."""
        session = self._make_session(monkeypatch)
        session._session = SmartFakeClientSession(put_responses={
            "/remote/object/call": self._python_response({
                "mesh_path": "/Game/Maps/T.T:PersistentLevel.Body",
                "original_triangles": 1000,
                "result_triangles": 500,
                "result_vertex_count": 260,
                "reduction_ratio": 0.5,
            }),
        })

        result = asyncio.run(session.simplify_mesh(
            mesh_path="/Game/Maps/T.T:PersistentLevel.Body",
            target_triangle_count=500,
        ))

        assert result["original_triangles"] == 1000
        assert result["result_triangles"] == 500
        assert result["reduction_ratio"] == pytest.approx(0.5)

    def test_cut_mesh_plane_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """cut_mesh_plane returns cut result."""
        session = self._make_session(monkeypatch)
        session._session = SmartFakeClientSession(put_responses={
            "/remote/object/call": self._python_response({
                "mesh_path": "/Game/Maps/T.T:PersistentLevel.Body",
                "result_triangle_count": 200,
                "result_vertex_count": 120,
                "previous_triangle_count": 100,
            }),
        })

        result = asyncio.run(session.cut_mesh_plane(
            mesh_path="/Game/Maps/T.T:PersistentLevel.Body",
            plane_origin=[0.0, 0.0, 50.0],
            plane_normal=[0.0, 0.0, 1.0],
        ))

        assert result["result_triangle_count"] == 200
        assert result["previous_triangle_count"] == 100

    def test_edit_mesh_topology_extrude(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """edit_mesh_topology returns affected faces for extrude."""
        session = self._make_session(monkeypatch)
        session._session = SmartFakeClientSession(put_responses={
            "/remote/object/call": self._python_response({
                "mesh_path": "/Game/Maps/T.T:PersistentLevel.Body",
                "operation": "extrude_faces",
                "result_triangle_count": 48,
                "result_vertex_count": 30,
                "previous_triangle_count": 24,
            }),
        })

        result = asyncio.run(session.edit_mesh_topology(
            mesh_path="/Game/Maps/T.T:PersistentLevel.Body",
            operation="extrude_faces",
            distance=20.0,
        ))

        assert result["operation"] == "extrude_faces"
        assert result["result_triangle_count"] == 48

    def test_validate_mesh_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """validate_mesh returns per-check results."""
        session = self._make_session(monkeypatch)
        session._session = SmartFakeClientSession(put_responses={
            "/remote/object/call": self._python_response({
                "mesh_path": "/Game/Maps/T.T:PersistentLevel.Body",
                "is_valid": True,
                "triangle_count": 100,
                "vertex_count": 52,
                "open_border_edges": 0,
                "open_border_loops": 0,
                "connected_components": 1,
                "has_normals": True,
                "issues": [],
            }),
        })

        result = asyncio.run(session.validate_mesh(
            mesh_path="/Game/Maps/T.T:PersistentLevel.Body",
        ))

        assert result["is_valid"] is True
        assert result["open_border_edges"] == 0
        assert result["triangle_count"] == 100

    def test_convert_mesh_format_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """convert_mesh_format returns converted path."""
        session = self._make_session(monkeypatch)
        session._session = SmartFakeClientSession(put_responses={
            "/remote/object/call": self._python_response({
                "source_path": "/Game/Meshes/SM_Body",
                "result_path": "/Game/Meshes/DynMesh_Body",
                "target_format": "dynamic_mesh",
                "triangle_count": 200,
                "vertex_count": 120,
            }),
        })

        result = asyncio.run(session.convert_mesh_format(
            mesh_path="/Game/Meshes/SM_Body",
            target_format="dynamic_mesh",
        ))

        assert result["target_format"] == "dynamic_mesh"
        assert result["triangle_count"] == 200

    def test_remesh_mesh_uniform(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """remesh_mesh returns new triangle counts."""
        session = self._make_session(monkeypatch)
        session._session = SmartFakeClientSession(put_responses={
            "/remote/object/call": self._python_response({
                "mesh_path": "/Game/Maps/T.T:PersistentLevel.Body",
                "mode": "uniform",
                "original_triangles": 1000,
                "result_triangles": 800,
                "result_vertex_count": 420,
            }),
        })

        result = asyncio.run(session.remesh_mesh(
            mesh_path="/Game/Maps/T.T:PersistentLevel.Body",
            mode="uniform",
            target_edge_length=5.0,
        ))

        assert result["mode"] == "uniform"
        assert result["original_triangles"] == 1000
        assert result["result_triangles"] == 800

    def test_compute_mesh_uv_auto(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """compute_mesh_uv returns UV metrics."""
        session = self._make_session(monkeypatch)
        session._session = SmartFakeClientSession(put_responses={
            "/remote/object/call": self._python_response({
                "mesh_path": "/Game/Maps/T.T:PersistentLevel.Body",
                "method": "auto_uv",
                "uv_channel": 0,
                "triangle_count": 100,
                "vertex_count": 52,
            }),
        })

        result = asyncio.run(session.compute_mesh_uv(
            mesh_path="/Game/Maps/T.T:PersistentLevel.Body",
            method="auto_uv",
        ))

        assert result["method"] == "auto_uv"
        assert result["triangle_count"] == 100


# ---------------------------------------------------------------------------
# Passphrase header support — closes the iter3 deferred work where simul-mcp
# itself couldn't talk to a passphrase-enforcing UE editor.
# ---------------------------------------------------------------------------


class TestPassphraseHeader:
    """`_passphrase_to_md5` + UnrealRuntimeSession._default_headers wiring."""

    def test_md5_helper_returns_none_when_unset(self) -> None:
        assert unreal_runtime._passphrase_to_md5(None) is None

    def test_md5_helper_hashes_plaintext(self) -> None:
        # md5("password") = 5f4dcc3b5aa765d61d8327deb882cf99 (well-known).
        assert (
            unreal_runtime._passphrase_to_md5("password")
            == "5f4dcc3b5aa765d61d8327deb882cf99"
        )

    def test_md5_helper_passes_through_lowercase_hex(self) -> None:
        h = "5f4dcc3b5aa765d61d8327deb882cf99"
        assert unreal_runtime._passphrase_to_md5(h) == h

    def test_md5_helper_normalizes_uppercase_hex_to_lowercase(self) -> None:
        # Operator might paste a hash from a tool that uses uppercase hex
        # (e.g. md5sum on some platforms). UE's FMD5 output is lowercase
        # so we normalize to match.
        assert (
            unreal_runtime._passphrase_to_md5("5F4DCC3B5AA765D61D8327DEB882CF99")
            == "5f4dcc3b5aa765d61d8327deb882cf99"
        )

    def test_md5_helper_rejects_non_ascii(self) -> None:
        # UE's FMD5::HashAnsiString narrows wide chars before hashing,
        # so a non-ASCII plaintext would silently produce a different
        # hash on UE's side. Better to raise here than mismatch silently.
        with pytest.raises(UnicodeEncodeError):
            unreal_runtime._passphrase_to_md5("café")

    def _make_session_with_passphrase(
        self, monkeypatch: pytest.MonkeyPatch, passphrase: "str | None"
    ) -> Any:
        monkeypatch.setattr(unreal_runtime, "UNREAL_AVAILABLE", True)
        monkeypatch.setattr(unreal_runtime, "AIOHTTP_AVAILABLE", True)
        # UnrealConfig is a frozen pydantic BaseModel; build a fresh
        # Settings with a tweaked unreal section.
        from simul_mcp.config import UnrealConfig
        settings = Settings(unreal=UnrealConfig(passphrase=passphrase))
        return unreal_runtime.UnrealRuntimeSession(settings=settings)

    def test_default_headers_omit_passphrase_when_unconfigured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session = self._make_session_with_passphrase(monkeypatch, None)
        headers = session._default_headers()
        assert headers == {"Content-Type": "application/json"}

    def test_default_headers_include_passphrase_md5_from_plaintext(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session = self._make_session_with_passphrase(monkeypatch, "password")
        headers = session._default_headers()
        assert headers["Passphrase"] == "5f4dcc3b5aa765d61d8327deb882cf99"
        assert headers["Content-Type"] == "application/json"

    def test_default_headers_pass_through_prehashed_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Operator pre-computed the hash and stored that in env; client
        # must not double-hash.
        prehashed = "98264f6f4d06848183632e6a314112ed"  # md5("iter3-secret")
        session = self._make_session_with_passphrase(monkeypatch, prehashed)
        assert session._default_headers()["Passphrase"] == prehashed

    @pytest.mark.parametrize(
        "near_hex",
        [
            "5f4dcc3b5aa765d61d8327deb882cf9",   # 31 chars
            "5f4dcc3b5aa765d61d8327deb882cf999",  # 33 chars
            "5f4dcc3b5aa765d61d8327deb882cf9z",  # 32 chars but non-hex
        ],
    )
    def test_md5_helper_hashes_near_hex_strings_not_passthrough(
        self, near_hex: str
    ) -> None:
        """Boundary: only EXACTLY-32-char hex strings pass through. Off-by-one
        or non-hex 32-char strings get treated as plaintext and hashed.
        Pin the boundary so a future refactor can't drop the regex's hex
        constraint and silently double-hash a 32-char password."""
        result = unreal_runtime._passphrase_to_md5(near_hex)
        # If passthrough fired, result == near_hex (case-folded). If hashing
        # fired, result is the MD5 hex of near_hex.
        assert result == hashlib.md5(near_hex.encode("ascii")).hexdigest()
        assert result != near_hex.lower()

    def test_md5_helper_empty_string_hashes_to_md5_of_empty(self) -> None:
        """Documented behavior: empty string is plaintext (regex doesn't
        match), so it MD5-hashes to the well-known empty-string digest.
        A user who sets SIMUL_UNREAL__PASSPHRASE='' would inject that hash.
        Pin so the behavior is explicit, not accidental."""
        assert (
            unreal_runtime._passphrase_to_md5("")
            == "d41d8cd98f00b204e9800998ecf8427e"
        )

    def test_default_headers_reach_clientsession_via_ensure_http_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Integration guard: confirm _default_headers() is what
        _ensure_http_session passes to aiohttp.ClientSession. Catches a
        revert of the headers= kwarg to a hardcoded literal."""
        session = self._make_session_with_passphrase(monkeypatch, "password")
        captured: Dict[str, Any] = {}

        class _CaptureSession:
            closed = False

            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(
            unreal_runtime.aiohttp, "ClientSession", _CaptureSession, raising=True
        )

        asyncio.run(session._ensure_http_session())

        assert "headers" in captured
        assert captured["headers"]["Content-Type"] == "application/json"
        assert (
            captured["headers"]["Passphrase"]
            == "5f4dcc3b5aa765d61d8327deb882cf99"
        )

    def test_probe_port_omits_passphrase_header_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Backward compat: probe_port() with no passphrase_md5 must NOT
        inject the header — preserves discovery behavior for editors
        without passphrase enforcement."""
        monkeypatch.setattr(unreal_runtime, "AIOHTTP_AVAILABLE", True)
        captured: Dict[str, Any] = {}

        class _CaptureSession:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            async def __aenter__(self):
                raise RuntimeError("stop here — only the constructor matters")

            async def __aexit__(self, *args):
                pass

        monkeypatch.setattr(
            unreal_runtime.aiohttp, "ClientSession", _CaptureSession, raising=True
        )

        asyncio.run(
            unreal_runtime.UnrealRuntimeSession.probe_port(
                "127.0.0.1", 30010, timeout=1.0
            )
        )

        assert "Passphrase" not in captured["headers"]

    def test_probe_port_injects_passphrase_when_supplied(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Closes the iter4 code-review HIGH: discovery against a
        passphrase-enforcing editor must carry the header."""
        monkeypatch.setattr(unreal_runtime, "AIOHTTP_AVAILABLE", True)
        captured: Dict[str, Any] = {}
        md5_hash = "5f4dcc3b5aa765d61d8327deb882cf99"

        class _CaptureSession:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            async def __aenter__(self):
                raise RuntimeError("stop here — only the constructor matters")

            async def __aexit__(self, *args):
                pass

        monkeypatch.setattr(
            unreal_runtime.aiohttp, "ClientSession", _CaptureSession, raising=True
        )

        asyncio.run(
            unreal_runtime.UnrealRuntimeSession.probe_port(
                "127.0.0.1", 30010, timeout=1.0, passphrase_md5=md5_hash
            )
        )

        assert captured["headers"]["Passphrase"] == md5_hash