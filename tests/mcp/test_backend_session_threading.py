"""The backend envelope offloads a session call to a thread when the adapter says so.

The decision used to be ``adapter_label == "Blender"`` plus a Blender settings
check inside the generic envelope. Adapters now declare ``session_blocks_io``;
the envelope reads only that.
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator

import pytest
from pydantic import BaseModel

from simul_mcp.adapters.blender_runtime import BlenderRuntimeAdapter
from simul_mcp.config import Settings
from simul_mcp.mcp import backends as backends_module
from simul_mcp.mcp import server as server_module
from tests.fakes import FakeFastMCP


class _Ok(BaseModel):
    success: bool = True


class _Adapter:
    def __init__(self, **attrs: Any) -> None:
        self.__dict__.update(attrs)

    def is_available(self) -> bool:
        return True

    @contextmanager
    def create_session(self) -> Iterator[object]:
        yield object()


def _make_server(monkeypatch: pytest.MonkeyPatch) -> server_module.SimulMCPServer:
    monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
    monkeypatch.setattr(server_module, "TaskConfig", None)
    monkeypatch.setattr(backends_module, "is_headless_available", lambda: False)
    monkeypatch.setattr(backends_module, "is_blender_available", lambda: False)
    monkeypatch.setattr(backends_module, "UnrealRuntimeAdapter", None)
    return server_module.SimulMCPServer(settings=Settings())


def _thread_of_call(instance: server_module.SimulMCPServer, adapter: Any, label: str) -> int:
    seen: Dict[str, int] = {}

    def call(session: Any) -> Dict[str, Any]:
        seen["thread"] = threading.get_ident()
        return {"success": True}

    async def run() -> int:
        await instance._run_backend_call(
            "probe", adapter, label, _Ok, call
        )
        return seen["thread"]

    return asyncio.run(run())


def test_adapter_declaring_blocking_io_runs_off_the_event_loop_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = _make_server(monkeypatch)
    # Deliberately not labelled "Blender": the attribute alone decides.
    thread = _thread_of_call(instance, _Adapter(session_blocks_io=True), "Other")
    assert thread != threading.get_ident()


def test_adapter_without_the_attribute_runs_inline(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = _make_server(monkeypatch)
    assert _thread_of_call(instance, _Adapter(), "Blender") == threading.get_ident()
    assert (
        _thread_of_call(instance, _Adapter(session_blocks_io=False), "Blender")
        == threading.get_ident()
    )


@pytest.mark.parametrize(("mode", "expected"), [("attached", True), ("embedded", False)])
def test_blender_adapter_declares_blocking_io_only_when_attached(mode: str, expected: bool) -> None:
    adapter = BlenderRuntimeAdapter(Settings(blender={"mode": mode}))
    assert adapter.session_blocks_io is expected
