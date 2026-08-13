"""Regressions for list_isaac_instances honesty and fan-out cost.

Two defects in one code path:

* An unreachable instance was still scored from session bookkeeping alone, so it
  came back ``instance_status: "free"`` / ``compatibility: "clear"`` / score 1
  next to ``reachable: false`` and ``total_discovered: 0``. The positive fields
  are what an agent acts on, so it would pick an instance it cannot talk to.
* Briefs were awaited one instance at a time, and each brief made two sequential
  bridge round trips whose first call traversed the whole stage for a prim count
  the listing only displays.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List

import pytest

src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_path))

from simul_mcp.config import Settings  # noqa: E402
from simul_mcp.mcp import server as server_module  # noqa: E402


class FakeFastMCP:
    """Minimal FastMCP double that records registered tools."""

    def __init__(self, name: str, version: str, **kwargs: Any):
        self.name = name
        self.version = version
        self.instructions = kwargs.get("instructions")
        self.tools: List[SimpleNamespace] = []

    def tool(
        self, name: str, **kwargs: Any
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.tools.append(SimpleNamespace(name=name, func=func, kwargs=kwargs))
            return func

        return decorator

    def resource(self, *args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def add_middleware(self, middleware: Any) -> None:
        return


class _FakeClient:
    """Client double exposing only what the instance listing touches."""

    def __init__(self, port: int) -> None:
        self._host = "127.0.0.1"
        self._port = port
        self._bridge_port = port + 3

    @property
    def bridge_address(self) -> str:
        return f"{self._host}:{self._bridge_port}"

    @property
    def vscode_address(self) -> str:
        return f"{self._host}:{self._port}"

    @property
    def bridge_enabled(self) -> bool:
        return True

    @property
    def fallback_to_vscode(self) -> bool:
        return True


class _FakeSessionManager:
    """Session bookkeeping with no sessions recorded — the reported case."""

    def __init__(self) -> None:
        self.status_calls = 0

    def get_instance_session(self, port: int) -> Any:
        manager = self

        class _Session:
            def get_status(self) -> Dict[str, Any]:
                manager.status_calls += 1
                return {"sessions": [], "session_count": 0, "status": "free"}

        return _Session()

    def score_compatibility(
        self, purpose: str, port: int, status: Any = None
    ) -> Dict[str, Any]:
        return {
            "compatibility": "clear",
            "score": 1.0,
            "reason": "No active sessions — instance is free",
        }


def _make_server(monkeypatch: pytest.MonkeyPatch) -> server_module.SimulMCPServer:
    monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
    monkeypatch.setattr(server_module, "TaskConfig", None)
    monkeypatch.setattr(server_module, "is_headless_available", lambda: False)
    monkeypatch.setattr(server_module, "is_blender_available", lambda: False)
    monkeypatch.setattr(server_module, "UnrealRuntimeAdapter", None)
    instance = server_module.SimulMCPServer(settings=Settings())
    instance.session_manager = _FakeSessionManager()
    return instance


def _get_tool(instance: server_module.SimulMCPServer, name: str) -> Callable[..., Any]:
    for tool in instance.mcp.tools:
        if tool.name == name:
            return tool.func
    raise AssertionError(f"Tool {name!r} not found")


def _brief(name: str, port: int, *, reachable: bool) -> Dict[str, Any]:
    return {
        "name": name,
        "host": "127.0.0.1",
        "port": port,
        "bridge_address": f"127.0.0.1:{port + 3}",
        "vscode_address": f"127.0.0.1:{port}",
        "reachable": reachable,
        "active": True,
        "stage_url": None,
        "up_axis": None,
        "prim_count": None,
        "is_playing": None,
    }


def test_unreachable_instance_is_not_reported_as_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An instance we cannot reach must never read as a usable one."""
    instance = _make_server(monkeypatch)
    instance._isaac_clients = {"default": _FakeClient(8226)}

    async def _fake_brief(name: str, client: Any) -> Dict[str, Any]:
        return _brief(name, client._port, reachable=False)

    monkeypatch.setattr(instance, "_get_instance_brief", _fake_brief)

    result = asyncio.run(_get_tool(instance, "list_isaac_instances")(scan=False))
    entry = result["instances"][0]

    assert entry["reachable"] is False
    assert entry["instance_status"] == "unreachable"
    assert entry["compatibility"] == "blocked"
    assert entry["compatibility_score"] == 0.0
    assert "reachable" in entry["compatibility_reason"].lower()
    assert result["total_discovered"] == 0


def test_reachable_instance_without_sessions_still_reads_clear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The honesty fix must not make usable free instances look blocked."""
    instance = _make_server(monkeypatch)
    instance._isaac_clients = {"default": _FakeClient(8226)}

    async def _fake_brief(name: str, client: Any) -> Dict[str, Any]:
        return _brief(name, client._port, reachable=True)

    monkeypatch.setattr(instance, "_get_instance_brief", _fake_brief)

    result = asyncio.run(_get_tool(instance, "list_isaac_instances")(scan=False))
    entry = result["instances"][0]

    assert entry["instance_status"] == "free"
    assert entry["compatibility"] == "clear"
    assert entry["compatibility_score"] == 1.0
    assert result["total_discovered"] == 1


def test_instance_briefs_are_fetched_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Listing N instances must not serialise N round trips."""
    instance = _make_server(monkeypatch)
    instance._isaac_clients = {
        "isaac-8226": _FakeClient(8226),
        "isaac-8227": _FakeClient(8227),
        "isaac-8228": _FakeClient(8228),
    }

    in_flight = 0
    peak = 0

    async def _fake_brief(name: str, client: Any) -> Dict[str, Any]:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        try:
            await asyncio.sleep(0.01)
        finally:
            in_flight -= 1
        return _brief(name, client._port, reachable=True)

    monkeypatch.setattr(instance, "_get_instance_brief", _fake_brief)

    result = asyncio.run(_get_tool(instance, "list_isaac_instances")(scan=False))

    assert len(result["instances"]) == 3
    assert peak > 1, "instance briefs were awaited one at a time"


def test_brief_does_not_pay_for_a_prim_count_traversal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The listing's prim count is incidental; it must not cost a full traverse."""
    instance = _make_server(monkeypatch)
    client = _FakeClient(8226)
    requests: List[Any] = []

    async def _bridge_request(action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        requests.append((action, payload))
        return {"status": "ok", "payload": {"stage_url": "omniverse://x.usd"}}

    client.bridge_request = _bridge_request  # type: ignore[attr-defined]

    asyncio.run(instance._get_instance_brief("default", client))

    stage_calls = [payload for action, payload in requests if action == "get_stage_info"]
    assert stage_calls, "brief never asked for stage info"
    assert stage_calls[0].get("include_prim_count") is False
