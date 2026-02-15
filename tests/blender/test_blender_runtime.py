"""Tests for Blender runtime adapter functionality."""

from types import SimpleNamespace
from typing import Any
from pathlib import Path
import sys

import pytest

src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_path))

from simul_mcp.adapters import blender_runtime  # noqa: E402


class FakeObject:
    """Simple fake Blender object for tests."""

    def __init__(self, name: str, object_type: str, visible: bool):
        self.name = name
        self.type = object_type
        self._visible = visible
        self.hide_viewport = not visible

    def visible_get(self) -> bool:
        """Mirror Blender visible_get behavior."""
        return self._visible


class TestBlenderRuntimeSession:
    """Test cases for BlenderRuntimeSession."""

    def test_session_raises_without_bpy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Session initialization fails when bpy is unavailable."""
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", False)

        with pytest.raises(ImportError, match="Blender runtime not available"):
            blender_runtime.BlenderRuntimeSession()

    def test_get_runtime_info_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Runtime info is returned from mocked bpy app/data."""
        fake_app = SimpleNamespace(
            version=(4, 1, 0),
            version_string="4.1.0",
            binary_path="/Applications/Blender.app/Contents/MacOS/Blender",
            background=False,
        )
        fake_data = SimpleNamespace(filepath="/tmp/test_scene.blend")
        fake_bpy = SimpleNamespace(app=fake_app, data=fake_data)

        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()
        runtime_info = session.get_runtime_info()

        assert runtime_info["version"] == [4, 1, 0]
        assert runtime_info["version_string"] == "4.1.0"
        assert runtime_info["binary_path"] == fake_app.binary_path
        assert runtime_info["background"] is False
        assert runtime_info["blend_file_path"] == "/tmp/test_scene.blend"

    def test_list_scene_objects_filters_hidden(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Object listing respects hidden filtering and limits."""
        visible_obj = FakeObject(name="Cube", object_type="MESH", visible=True)
        hidden_obj = FakeObject(name="Light", object_type="LIGHT", visible=False)
        fake_data = SimpleNamespace(
            filepath="",
            objects=[visible_obj, hidden_obj],
            collections={},
        )
        fake_bpy = SimpleNamespace(
            app=SimpleNamespace(
                version=(4, 1, 0),
                version_string="4.1.0",
                binary_path="/bin/blender",
                background=True,
            ),
            data=fake_data,
        )

        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.list_scene_objects(include_hidden=False, max_items=10)

        assert result["count"] == 1
        assert result["objects"][0]["name"] == "Cube"
        assert result["objects"][0]["visible"] is True

    def test_list_scene_objects_collection_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Unknown collections raise a validation error."""
        fake_data = SimpleNamespace(filepath="", objects=[], collections={})
        fake_bpy = SimpleNamespace(
            app=SimpleNamespace(
                version=(4, 1, 0),
                version_string="4.1.0",
                binary_path="/bin/blender",
                background=True,
            ),
            data=fake_data,
        )

        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()

        with pytest.raises(ValueError, match="Collection not found"):
            session.list_scene_objects(collection_name="Missing")


class TestBlenderRuntimeAdapter:
    """Test cases for BlenderRuntimeAdapter."""

    def test_is_blender_available_passthrough(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Availability helper returns module-level runtime flag."""
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", False)
        assert blender_runtime.is_blender_available() is False

        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        assert blender_runtime.is_blender_available() is True

    def test_create_session_returns_session(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Adapter create_session yields a runtime session instance."""
        fake_session: Any = object()

        class FakeSession:
            """Session replacement to isolate adapter logic."""

            def __init__(self, settings: Any = None):
                self.settings = settings

            def cleanup(self) -> None:
                return None

        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "BlenderRuntimeSession", FakeSession)

        adapter = blender_runtime.BlenderRuntimeAdapter()
        with adapter.create_session() as session:
            fake_session = session

        assert isinstance(fake_session, FakeSession)
