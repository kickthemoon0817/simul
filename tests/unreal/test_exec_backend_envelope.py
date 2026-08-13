"""The shared backend envelope, which Unreal and Blender tools now delegate to.

Isaac tools have had ``_exec_isaac`` from the start, which is why each is a
single forwarding line while Unreal and Blender hand-rolled the same rate limit,
availability check, session scope, success normalisation, model validation and
error wrapping once per tool. Copying that block ~110 times is what let four
sites keep pydantic v1's ``.dict()`` long after everything else moved.

Most converted tools have no registration-level test of their own — the suite
covers the adapter methods underneath them — so these tests cover the envelope
itself, which is now the single thing every converted tool shares.
"""

from __future__ import annotations

import asyncio
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pytest
from pydantic import BaseModel, Field

src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_path))

from simul_mcp.config import Settings  # noqa: E402
from simul_mcp.mcp import server as server_module  # noqa: E402


class _Response(BaseModel):
    success: bool = Field(default=True)
    error: Optional[str] = Field(default=None)
    value: str = Field(default="")


class _Session:
    def __init__(self, payload: Dict[str, Any], boom: Optional[Exception]) -> None:
        self._payload = payload
        self._boom = boom
        self.calls = 0

    async def work(self) -> Dict[str, Any]:
        self.calls += 1
        if self._boom is not None:
            raise self._boom
        return dict(self._payload)


class _Adapter:
    def __init__(
        self,
        payload: Optional[Dict[str, Any]] = None,
        *,
        available: bool = True,
        boom: Optional[Exception] = None,
    ) -> None:
        self._available = available
        self.session = _Session(payload or {"value": "ok"}, boom)
        self.sessions_opened = 0
        self.sessions_closed = 0

    def is_available(self) -> bool:
        return self._available

    @contextmanager
    def create_session(self) -> Iterator[_Session]:
        self.sessions_opened += 1
        try:
            yield self.session
        finally:
            self.sessions_closed += 1


class FakeFastMCP:
    def __init__(self, name: str, version: str, **kwargs: Any):
        self.tools: List[Any] = []

    def tool(self, name: str, **kwargs: Any):
        def decorator(func):
            return func

        return decorator

    def resource(self, *args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def add_middleware(self, middleware: Any) -> None:
        return


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> server_module.SimulMCPServer:
    monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
    monkeypatch.setattr(server_module, "TaskConfig", None)
    monkeypatch.setattr(server_module, "is_headless_available", lambda: False)
    monkeypatch.setattr(server_module, "is_blender_available", lambda: False)
    monkeypatch.setattr(server_module, "UnrealRuntimeAdapter", None)
    return server_module.SimulMCPServer(settings=Settings())


def _run(server: server_module.SimulMCPServer, adapter: Any) -> Dict[str, Any]:
    return asyncio.run(
        server._exec_backend(
            "some_unreal_tool", adapter, "Unreal", _Response,
            lambda session: session.work(),
        )
    )


def test_payload_is_validated_and_returned(server) -> None:
    adapter = _Adapter({"value": "hello"})

    result = _run(server, adapter)

    assert result["value"] == "hello"
    assert result["success"] is True
    assert adapter.session.calls == 1


def test_session_is_closed_even_when_the_call_raises(server) -> None:
    """The session scope must survive a failure, or a backend leaks handles."""
    adapter = _Adapter(boom=RuntimeError("editor went away"))

    result = _run(server, adapter)

    assert result["error_type"] == "Exception"
    assert "editor went away" in result["error"]
    assert adapter.sessions_opened == 1
    assert adapter.sessions_closed == 1


def test_absent_backend_reports_itself_by_name(server) -> None:
    adapter = _Adapter(available=False)

    result = _run(server, adapter)

    assert result["error_type"] == "RuntimeError"
    assert "Unreal runtime not available" in result["error"]
    assert adapter.sessions_opened == 0


def test_missing_adapter_is_not_a_crash(server) -> None:
    result = _run(server, None)

    assert result["error_type"] == "RuntimeError"


def test_error_payload_flips_success(server) -> None:
    """apply_success_from_error has to keep running inside the envelope."""
    adapter = _Adapter({"error": "actor not found"})

    result = _run(server, adapter)

    assert result["success"] is False
    assert result["error"] == "actor not found"


def test_rate_limit_short_circuits_before_a_session_opens(
    server, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = _Adapter()
    monkeypatch.setattr(
        server,
        "_check_rate_limit",
        lambda name: {"success": False, "error": "rate limited", "error_type": "RateLimit"},
    )

    result = _run(server, adapter)

    assert result["error_type"] == "RateLimit"
    assert adapter.sessions_opened == 0
