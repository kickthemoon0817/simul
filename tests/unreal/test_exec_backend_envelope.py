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
import json
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

import pytest
from pydantic import BaseModel, Field


from simul_mcp.config import Settings
from simul_mcp.mcp import backends as backends_module
from simul_mcp.mcp import server as server_module
from simul_mcp.mcp.result_budget import DEFAULT_RESULT_BUDGET_BYTES
from tests.fakes import FakeFastMCP


def _payload(result: Any) -> Dict[str, Any]:
    """Read the envelope's single JSON content block."""
    assert result.structured_content is None
    assert len(result.content) == 1
    return json.loads(result.content[0].text)


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


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> server_module.SimulMCPServer:
    monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
    monkeypatch.setattr(server_module, "TaskConfig", None)
    monkeypatch.setattr(backends_module, "is_headless_available", lambda: False)
    monkeypatch.setattr(backends_module, "is_blender_available", lambda: False)
    monkeypatch.setattr(backends_module, "UnrealRuntimeAdapter", None)
    return server_module.SimulMCPServer(settings=Settings())


def _run(
    server: server_module.SimulMCPServer,
    adapter: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return _payload(
        asyncio.run(
            server._exec_backend(
                "some_unreal_tool",
                adapter,
                "Unreal",
                _Response,
                lambda session: session.work(),
                params=params,
            )
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
        lambda name, agent_id=None: {
            "success": False,
            "error": "rate limited",
            "error_type": "RateLimit",
        },
    )

    result = _run(server, adapter)

    assert result["error_type"] == "RateLimit"
    assert adapter.sessions_opened == 0


class _SyncSession:
    """Blender sessions are synchronous; the envelope must accept both."""

    def __init__(self) -> None:
        self.calls = 0

    def work(self) -> Dict[str, Any]:
        self.calls += 1
        return {"value": "sync-ok"}


def test_synchronous_session_calls_are_supported(server) -> None:
    adapter = _Adapter()
    adapter.session = _SyncSession()  # type: ignore[assignment]

    result = _payload(
        asyncio.run(
            server._exec_backend(
                "some_blender_tool",
                adapter,
                "Blender",
                _Response,
                lambda session: session.work(),
            )
        )
    )

    assert result["value"] == "sync-ok"
    assert result["success"] is True
    assert adapter.session.calls == 1


def test_tool_built_error_envelope_keeps_its_fields(server) -> None:
    """A not-found or validation envelope must not be rewritten by the schema."""
    adapter = _Adapter(
        {
            "success": False,
            "error": "Prim not found: /World/Missing",
            "error_type": "NotFoundError",
            "details": {"prim_path": "/World/Missing"},
        }
    )

    result = _run(server, adapter)

    assert result["error_type"] == "NotFoundError"
    assert result["details"] == {"prim_path": "/World/Missing"}


def test_every_call_is_recorded_with_agent_and_params(server) -> None:
    """The envelope is the audit trail for Unreal, Blender and USD tools."""
    server.usage_tracker._recent.clear()

    _run(server, _Adapter({"value": "ok"}), params={"actor": "Cube"})
    _run(server, _Adapter({"error": "actor not found"}))
    _run(server, _Adapter(boom=RuntimeError("editor went away")))

    records = server.usage_tracker.get_recent(tool_name="some_unreal_tool")
    assert [r["success"] for r in records] == [False, False, True]
    assert records[2]["params"] == {"actor": "Cube"}
    assert "actor not found" in records[1]["error"]
    assert "editor went away" in records[0]["error"]
    assert all(r["agent_id"] for r in records)


def test_rate_limited_calls_are_recorded_too(server, monkeypatch) -> None:
    server.usage_tracker._recent.clear()
    monkeypatch.setattr(
        server,
        "_check_rate_limit",
        lambda name, agent_id=None: {"success": False, "error": "x", "error_type": "RateLimitError"},
    )

    _run(server, _Adapter())

    (record,) = server.usage_tracker.get_recent(tool_name="some_unreal_tool")
    assert record["success"] is False
    assert record["error"] == "rate_limited"


class _ListResponse(BaseModel):
    success: bool = Field(default=True)
    error: Optional[str] = Field(default=None)
    items: List[str] = Field(default_factory=list)


def test_result_budget_applies_to_backend_payloads(server) -> None:
    """A large listing from Unreal or Blender is trimmed like an Isaac one."""
    adapter = _Adapter({"items": [f"/Game/Meshes/SM_Chair_{i:05d}" for i in range(20_000)]})

    result = _payload(
        asyncio.run(
            server._exec_backend(
                "list_unreal_actors", adapter, "Unreal", _ListResponse, lambda s: s.work()
            )
        )
    )

    assert result["truncated"] is True
    assert result["truncation"]["field"] == "items"
    assert len(json.dumps(result)) <= DEFAULT_RESULT_BUDGET_BYTES


@pytest.mark.parametrize(
    ("tool_name", "kwargs"),
    [
        ("get_blender_info", {}),
        ("get_blender_object_info", {"object_name": "Cube"}),
        ("search_unreal_assets", {"query": "chair"}),
    ],
)
def test_registered_tool_checks_the_rate_limit_exactly_once(
    monkeypatch: pytest.MonkeyPatch, tool_name: str, kwargs: Dict[str, Any]
) -> None:
    """A converted tool must not keep an outer check on top of the envelope's.

    Two checks per call drain two tokens from the same bucket: the effective
    rate halves, and at burst_size=1 the tool can never run at all.
    """

    class _StubAdapter:
        def __init__(self, settings: Any) -> None:
            pass

        def is_available(self) -> bool:
            return True

    monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
    monkeypatch.setattr(server_module, "TaskConfig", None)
    monkeypatch.setattr(backends_module, "is_headless_available", lambda: False)
    monkeypatch.setattr(backends_module, "is_blender_available", lambda: True)
    monkeypatch.setattr(backends_module, "BlenderRuntimeAdapter", _StubAdapter)
    monkeypatch.setattr(backends_module, "UnrealRuntimeAdapter", _StubAdapter)
    instance = server_module.SimulMCPServer(settings=Settings())

    # MCP mode registers the thin Unreal set; the converted tools live in
    # the full set, so register it the way the CLI surface does.
    from simul_mcp.mcp.registration import register_unreal_tools

    register_unreal_tools(instance, thin=False)

    checks: List[str] = []
    monkeypatch.setattr(
        instance, "_check_rate_limit", lambda name, agent_id=None: checks.append(name)
    )
    instance.blender_adapter = _Adapter(available=False)
    instance.unreal_adapter = _Adapter(available=False)

    func = instance.mcp.by_name[tool_name]
    asyncio.run(func(**kwargs))

    assert checks == [tool_name]
