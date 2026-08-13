"""Regressions for a bridge that could hang Isaac Sim or lie about being dead.

* The bridge awaited its request handler with no timeout. A script that never
  returns therefore hung Kit permanently, with SIGKILL the only recovery, and
  the client saw nothing until its own timeout fired.
* Every bridge action, including ``ping``, went through one lock, and the
  ``step`` handler holds it across up to 1000 frame awaits — ~17 s at 60 fps and
  far longer on the heavy stages the bridge exists for. A health check issued
  during that window timed out and reported the instance *unreachable* when the
  truth was *busy*.
"""

from __future__ import annotations

import asyncio
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List

import pytest

src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_path))

from simul_mcp.config import Settings  # noqa: E402
from simul_mcp.mcp import server as server_module  # noqa: E402

extension_root = (
    Path(__file__).resolve().parents[2]
    / "src" / "simul_mcp" / "bridge_ext" / "khemoo.simul.mcp"
)
sys.path.insert(0, str(extension_root))

from khemoo.simul.mcp.lifecycle import BridgeServerLifecycle  # noqa: E402
from khemoo.simul.mcp.service import READ_ONLY_ACTIONS  # noqa: E402


# ---------------------------------------------------------------------------
# #93 — a runaway handler must not hang the simulator forever
# ---------------------------------------------------------------------------


def _request_frame(action: str, request_id: str = "req-1") -> bytes:
    body = json.dumps(
        {"request_id": request_id, "action": action, "payload": {}}
    ).encode("utf-8")
    return struct.pack(">I", len(body)) + body


async def _round_trip(lifecycle: BridgeServerLifecycle, action: str) -> Dict[str, Any]:
    assert lifecycle._server is not None
    port = next(iter(lifecycle._server.sockets)).getsockname()[1]
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(_request_frame(action))
    await writer.drain()
    header = await reader.readexactly(4)
    size = struct.unpack(">I", header)[0]
    payload = await reader.readexactly(size)
    writer.close()
    await writer.wait_closed()
    return json.loads(payload.decode("utf-8"))


def test_runaway_handler_is_cut_off_rather_than_hanging() -> None:
    """A handler that never returns must still produce a response."""

    async def _exercise() -> Dict[str, Any]:
        async def _never_returns(request: Any) -> Any:
            await asyncio.sleep(3600)

        lifecycle = BridgeServerLifecycle(
            host="127.0.0.1",
            port=0,
            request_handler=_never_returns,
            request_timeout=0.2,
        )
        await lifecycle.start()
        try:
            return await asyncio.wait_for(_round_trip(lifecycle, "ping"), timeout=5.0)
        finally:
            await lifecycle.stop()

    response = asyncio.run(_exercise())

    assert response["status"] == "error"
    assert response["error"]["name"] == "RequestTimeout"
    assert "0.2" in response["error"]["message"]


def test_a_prompt_handler_is_unaffected_by_the_timeout() -> None:
    """The guard must not truncate work that finishes in time."""

    async def _exercise() -> Dict[str, Any]:
        async def _prompt(request: Any) -> Any:
            from khemoo.simul.mcp.protocol import BridgeResponse

            return BridgeResponse.success(request.request_id, {"reachable": True})

        lifecycle = BridgeServerLifecycle(
            host="127.0.0.1", port=0, request_handler=_prompt, request_timeout=5.0
        )
        await lifecycle.start()
        try:
            return await _round_trip(lifecycle, "ping")
        finally:
            await lifecycle.stop()

    response = asyncio.run(_exercise())

    assert response["status"] == "ok"
    assert response["payload"]["reachable"] is True


# ---------------------------------------------------------------------------
# #96 — reads must not queue behind a long-running step
# ---------------------------------------------------------------------------


def test_read_only_actions_are_declared_and_exclude_mutations() -> None:
    """The bypass is only safe for actions that do not change state."""
    for action in (
        "ping",
        "capabilities",
        "get_stage_info",
        "get_simulation_state",
        "get_prim_info",
        "list_prims",
    ):
        assert action in READ_ONLY_ACTIONS, f"{action} should skip the request lock"

    for action in ("execute_script", "simulation_control", "step"):
        assert action not in READ_ONLY_ACTIONS, f"{action} must hold the request lock"


class FakeFastMCP:
    def __init__(self, name: str, version: str, **kwargs: Any):
        self.name = name
        self.version = version
        self.instructions = kwargs.get("instructions")
        self.tools: List[SimpleNamespace] = []

    def tool(self, name: str, **kwargs: Any) -> Callable[..., Any]:
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


def _make_server(monkeypatch: pytest.MonkeyPatch) -> server_module.SimulMCPServer:
    monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
    monkeypatch.setattr(server_module, "TaskConfig", None)
    monkeypatch.setattr(server_module, "is_headless_available", lambda: False)
    monkeypatch.setattr(server_module, "is_blender_available", lambda: False)
    monkeypatch.setattr(server_module, "UnrealRuntimeAdapter", None)
    return server_module.SimulMCPServer(settings=Settings())


def test_busy_instance_reports_busy_instead_of_waiting_forever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A queued call must be able to give up and say why."""
    instance = _make_server(monkeypatch)
    monkeypatch.setattr(instance, "_instance_lock_timeout", 0.1, raising=False)

    lock = instance._get_instance_lock(instance._get_effective_instance_name())

    async def _exercise() -> Dict[str, Any]:
        async def _noop() -> Dict[str, Any]:
            return {"success": True}

        async with lock:
            # Another session is mid-step and holding the instance.
            return await instance._exec_isaac("get_isaac_stage_info", _noop())

    result = asyncio.run(_exercise())

    # _exec_isaac returns a content-only ToolResult so the payload is not also
    # sent as structuredContent; the busy envelope rides in that content block.
    payload = json.loads(result.content[0].text)

    assert payload.get("error_type") == "InstanceBusy"
    assert "busy" in payload["error"].lower()
