"""Tests for Blender tool registration inside FastMCP server."""

import asyncio
import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest


from simul_mcp.config import Settings
from simul_mcp.mcp import backends as backends_module
from simul_mcp.mcp import server as server_module
from tests.fakes import FakeFastMCP


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
