"""Tests for Blender tool registration inside FastMCP server."""

import asyncio
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_path))

from simul_mcp.config import Settings  # noqa: E402
from simul_mcp.mcp import server as server_module  # noqa: E402


class FakeFastMCP:
    """Minimal FastMCP test double for tool registration."""

    def __init__(self, name: str, version: str, **kwargs: Any):
        self.name = name
        self.version = version
        self.description = kwargs.get("description")
        self.instructions = kwargs.get("instructions")
        self.tools: list[SimpleNamespace] = []

    def tool(self, name: str, **kwargs):
        """Return decorator that records tool metadata."""

        def decorator(func):
            self.tools.append(SimpleNamespace(name=name, func=func, kwargs=kwargs))
            return func

        return decorator

    def get_tools(self):
        """Mirror FastMCP get_tools API."""
        return self.tools

    def resource(self, *args, **kwargs):
        """Stub for resource registration."""
        def decorator(func):
            return func
        return decorator

    def add_middleware(self, middleware: Any) -> None:
        """Stub for FastMCP middleware registration.

        SimulMCPServer adds a request-context middleware (PR #23) before
        any tools register. Without this stub the tests collide with the
        documented pre-existing FakeFastMCP add_middleware failure mode
        — adding the stub removes this Blender test file from that set.
        """
        if not hasattr(self, "middlewares"):
            self.middlewares = []
        self.middlewares.append(middleware)


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

        monkeypatch.setattr(server_module, "is_headless_available", lambda: False)
        monkeypatch.setattr(server_module, "is_blender_available", lambda: True)
        monkeypatch.setattr(server_module, "BlenderRuntimeAdapter", FakeBlenderAdapter)

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

        monkeypatch.setattr(server_module, "is_headless_available", lambda: False)
        monkeypatch.setattr(server_module, "is_blender_available", lambda: False)

        instance = server_module.SimulMCPServer(settings=Settings())
        tool_names = {tool.name for tool in instance.mcp.tools}

        assert "get_blender_info" not in tool_names
        assert "list_blender_scene_objects" not in tool_names


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
                "error": "Simulated runtime metadata fetch failure",
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
        monkeypatch.setattr(server_module, "is_headless_available", lambda: False)
        monkeypatch.setattr(server_module, "is_blender_available", lambda: True)
        monkeypatch.setattr(
            server_module, "BlenderRuntimeAdapter", FakeBlenderAdapterErroring
        )

        instance = server_module.SimulMCPServer(settings=Settings())
        tool = next(
            t for t in instance.mcp.tools if t.name == "get_blender_info"
        )

        result = asyncio.run(tool.func())

        # iter17 contract: success accurately reflects the presence of
        # an error in the adapter payload. Note that the Blender response
        # schemas (BlenderInfoResponse and friends) do not currently
        # declare an `error: Optional[str]` field, so Pydantic strips
        # the diagnostic message before it reaches the client. Adding
        # `error` to those schemas is iter18 follow-up territory; the
        # iter17 fix is the success-flag accuracy alone.
        assert result["success"] is False, (
            f"iter17 invariant broken: registered get_blender_info tool "
            f"returned success={result.get('success')!r} when adapter "
            f"returned an error payload. Full response: {result}"
        )
