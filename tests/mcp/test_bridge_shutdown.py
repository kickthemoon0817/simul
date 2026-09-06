"""Disabling the bridge extension must close its listener, not leak a zombie server.

Kit calls ``on_shutdown`` on the thread that drives its asyncio loop. A stop
that blocks that thread on a future scheduled onto the same loop can never
complete: the listener stays bound, the next start moves to the following
port, and clients on the configured port keep talking to a server the
extension has already forgotten.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

extension_root = (
    Path(__file__).resolve().parents[2]
    / "src" / "simul_mcp" / "bridge_ext" / "khemoo.simul.mcp"
)
sys.path.insert(0, str(extension_root))

from khemoo.simul.mcp import extension as extension_module  # noqa: E402
from khemoo.simul.mcp.extension import IsaacMCPServerExtension  # noqa: E402
from khemoo.simul.mcp.lifecycle import BridgeServerLifecycle  # noqa: E402
from khemoo.simul.mcp.protocol import BridgeRequest, BridgeResponse  # noqa: E402


class _FakeCarb:
    """Enough of ``carb`` for the extension: a settings store and a log sink."""

    def __init__(self) -> None:
        self.values: dict[str, Any] = {}
        self.warnings: list[str] = []
        self.infos: list[str] = []
        self.settings = SimpleNamespace(get_settings=lambda: self)

    def get(self, key: str) -> Any:
        return self.values.get(key)

    def set(self, key: str, value: Any) -> None:
        self.values[key] = value

    def log_info(self, message: str) -> None:
        self.infos.append(message)

    def log_warn(self, message: str) -> None:
        self.warnings.append(message)


async def _echo_handler(request: BridgeRequest) -> BridgeResponse:
    return BridgeResponse.success(request.request_id, {})


async def _port_accepts(port: int) -> bool:
    try:
        _, writer = await asyncio.open_connection("127.0.0.1", port)
    except ConnectionRefusedError:
        return False
    writer.close()
    await writer.wait_closed()
    return True


def _started_extension(
    monkeypatch: pytest.MonkeyPatch, discovery_dir: Path
) -> tuple[IsaacMCPServerExtension, _FakeCarb]:
    """Build an extension wired to a fake Kit, configured to bind an ephemeral port."""
    carb = _FakeCarb()
    monkeypatch.setattr(extension_module, "carb", carb)
    monkeypatch.setattr(extension_module, "OMNI_AVAILABLE", True)
    ext = IsaacMCPServerExtension()
    ext._port = 0
    ext._discovery_dir = str(discovery_dir)
    return ext, carb


def _discovery_files(discovery_dir: Path) -> list[Path]:
    return sorted(discovery_dir.glob("simul-mcp-*.json"))


def test_lifecycle_close_refuses_connections_before_wait_closed() -> None:
    async def scenario() -> None:
        lifecycle = BridgeServerLifecycle(host="127.0.0.1", port=0, request_handler=_echo_handler)
        await lifecycle.start()
        port = lifecycle.actual_port
        assert await _port_accepts(port)

        lifecycle.close()

        assert not await _port_accepts(port)
        await asyncio.wait_for(lifecycle.wait_closed(), timeout=1.0)
        assert lifecycle._server is None

    asyncio.run(scenario())


def test_on_shutdown_from_loop_thread_closes_port_without_stalling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def scenario() -> None:
        ext, carb = _started_extension(monkeypatch, tmp_path)
        await ext._start_bridge_async()
        assert ext._server is not None
        port = ext._server.actual_port
        assert await _port_accepts(port)
        discovery = _discovery_files(tmp_path)
        assert len(discovery) == 1
        assert json.loads(discovery[0].read_text())["port"] == port

        started = time.perf_counter()
        ext.on_shutdown()
        elapsed = time.perf_counter() - started

        assert elapsed < 1.0, f"on_shutdown blocked the loop thread for {elapsed:.2f}s"
        assert not await _port_accepts(port)
        assert _discovery_files(tmp_path) == []
        assert ext._server is None
        assert ext.get_runtime_state()["running"] is False
        # Let the scheduled drain run so the loop shuts down clean.
        await asyncio.sleep(0.05)

    asyncio.run(scenario())


def test_restart_after_shutdown_rebinds_the_same_port(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A leaked listener would push the second start onto the next port."""

    async def scenario() -> None:
        ext, carb = _started_extension(monkeypatch, tmp_path)
        await ext._start_bridge_async()
        assert ext._server is not None
        port = ext._server.actual_port

        ext.on_shutdown()
        await asyncio.sleep(0)

        ext._port = port
        await ext._start_bridge_async()
        assert ext._server is not None
        assert ext._server.actual_port == port
        assert [w for w in carb.warnings if f"configured for port {port}" in w] == []
        await ext._stop_bridge_async()

    asyncio.run(scenario())


def test_on_shutdown_from_foreign_thread_stops_through_the_loop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loop = asyncio.new_event_loop()
    runner = threading.Thread(target=loop.run_forever, daemon=True)
    runner.start()
    try:
        ext, carb = _started_extension(monkeypatch, tmp_path)
        asyncio.run_coroutine_threadsafe(ext._start_bridge_async(), loop).result(timeout=5)
        assert ext._server is not None
        port = ext._server.actual_port
        assert asyncio.run_coroutine_threadsafe(_port_accepts(port), loop).result(timeout=5)

        monkeypatch.setattr(ext, "_get_event_loop", lambda: loop)
        ext.on_shutdown()

        assert not asyncio.run_coroutine_threadsafe(_port_accepts(port), loop).result(timeout=5)
        assert _discovery_files(tmp_path) == []
        assert ext._server is None
    finally:
        loop.call_soon_threadsafe(loop.stop)
        runner.join(timeout=5)
        loop.close()


def test_start_warns_when_bound_port_differs_from_configured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def scenario() -> None:
        blocker = BridgeServerLifecycle(host="127.0.0.1", port=0, request_handler=_echo_handler)
        await blocker.start()
        ext, carb = _started_extension(monkeypatch, tmp_path)
        ext._port = blocker.actual_port
        try:
            await ext._start_bridge_async()
            assert ext._server is not None
            assert ext._server.actual_port != blocker.actual_port
            assert any(
                f"port {blocker.actual_port}" in message and str(ext._server.actual_port) in message
                for message in carb.warnings
            ), carb.warnings
        finally:
            await ext._stop_bridge_async()
            await blocker.stop()

    asyncio.run(scenario())


def test_only_the_bridge_extension_class_is_visible_to_kits_scan(monkeypatch) -> None:
    """Kit instantiates every module-level IExt subclass, so exactly one may exist.

    Kit's ``omni.ext`` is stood in for by a fake with an ``IExt`` base, and the
    extension module is re-imported against it, which is how the scan sees it
    inside the editor.
    """
    import importlib
    import inspect
    import sys
    import types

    class IExt:
        def on_shutdown(self) -> None:
            pass

    fake_omni = types.ModuleType("omni")
    fake_omni_ext = types.ModuleType("omni.ext")
    fake_omni_ext.IExt = IExt
    fake_omni.ext = fake_omni_ext
    fake_carb = types.ModuleType("carb")
    monkeypatch.setitem(sys.modules, "omni", fake_omni)
    monkeypatch.setitem(sys.modules, "omni.ext", fake_omni_ext)
    monkeypatch.setitem(sys.modules, "carb", fake_carb)
    monkeypatch.delitem(sys.modules, "khemoo.simul.mcp.extension", raising=False)
    try:
        extension_module = importlib.import_module("khemoo.simul.mcp.extension")
        exposed = [
            name
            for name, obj in vars(extension_module).items()
            if inspect.isclass(obj) and issubclass(obj, IExt)
        ]
        assert exposed == ["IsaacMCPServerExtension"]
    finally:
        sys.modules.pop("khemoo.simul.mcp.extension", None)
        importlib.import_module("khemoo.simul.mcp.extension")
