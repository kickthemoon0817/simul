"""Isaac Sim extension entrypoint for the Simul bridge transport."""

from __future__ import annotations

import asyncio
from typing import Any

from .executor import ScriptExecutor
from .lifecycle import BridgeServerLifecycle
from .protocol import BridgeRequest, BridgeResponse
from .service import BridgeCommandService
from .ui_builder import BridgeUIBuilder

try:
    import carb
    import omni.ext

    OMNI_AVAILABLE = True
    OmniExtBase: type[Any] = omni.ext.IExt
except Exception:  # pragma: no cover - exercised by import smoke tests
    carb = None
    omni = None
    OMNI_AVAILABLE = False
    OmniExtBase = object


class IsaacMCPServerExtension(OmniExtBase):
    """Typed bridge transport for Simul MCP running inside Isaac Sim."""

    def __init__(self) -> None:
        super().__init__()
        self._ext_id = ""
        self._globals: dict[str, Any] = {}
        self._server: BridgeServerLifecycle | None = None
        self._executor: ScriptExecutor | None = None
        self._service: BridgeCommandService | None = None
        self._allow_unsafe_execution = True
        self._max_request_bytes = 1024 * 1024
        self._max_response_bytes = 10 * 1024 * 1024
        self._request_lock: asyncio.Lock | None = None
        self._host = "127.0.0.1"
        self._port = 8229
        self._ui_builder: BridgeUIBuilder | None = None

    def on_startup(self, ext_id: str) -> None:
        """Start the bridge server from Kit settings."""
        if not OMNI_AVAILABLE or carb is None:
            raise RuntimeError("IsaacMCPServerExtension requires Isaac Sim runtime.")

        self._ext_id = ext_id
        self._globals = {**globals()}
        self._reload_settings_from_kit()
        self._ui_builder = BridgeUIBuilder(
            title="Simul MCP Bridge",
            get_state_fn=self.get_runtime_state,
            apply_settings_fn=self.apply_runtime_settings,
            start_fn=self.start_bridge,
            stop_fn=self.stop_bridge,
            restart_fn=self.restart_bridge,
        )
        self._ui_builder.startup()
        self.start_bridge()

    def on_shutdown(self) -> None:
        """Stop the bridge server."""
        if not OMNI_AVAILABLE:
            return
        self._sync_stop_bridge(timeout=5)
        self._server = None
        self._executor = None
        self._service = None
        self._request_lock = None
        if self._ui_builder is not None:
            self._ui_builder.shutdown()
            self._ui_builder = None

    def get_runtime_state(self) -> dict[str, Any]:
        """Return the current bridge runtime settings and status."""
        return {
            "host": self._host,
            "port": self._port,
            "address": f"{self._host}:{self._port}",
            "allow_unsafe_execution": self._allow_unsafe_execution,
            "max_request_bytes": self._max_request_bytes,
            "max_response_bytes": self._max_response_bytes,
            "running": self._server is not None,
        }

    def apply_runtime_settings(
        self,
        host: str,
        port: int,
        allow_unsafe_execution: bool,
        max_request_bytes: int,
        max_response_bytes: int,
    ) -> None:
        """Persist updated Kit settings and restart the bridge."""
        if not OMNI_AVAILABLE or carb is None:
            return
        settings = carb.settings.get_settings()
        settings.set("/exts/khemoo.simul.mcp/host", host)
        settings.set("/exts/khemoo.simul.mcp/port", int(port))
        settings.set(
            "/exts/khemoo.simul.mcp/allow_unsafe_execution",
            bool(allow_unsafe_execution),
        )
        settings.set(
            "/exts/khemoo.simul.mcp/max_request_bytes",
            int(max_request_bytes),
        )
        settings.set(
            "/exts/khemoo.simul.mcp/max_response_bytes",
            int(max_response_bytes),
        )
        self._reload_settings_from_kit()
        self.restart_bridge()

    def start_bridge(self) -> None:
        """Start the bridge server with current settings."""
        self._schedule_task(self._start_bridge_async())

    def stop_bridge(self) -> None:
        """Stop the bridge server."""
        self._schedule_task(self._stop_bridge_async())

    def restart_bridge(self) -> None:
        """Restart the bridge server."""
        self._schedule_task(self._restart_bridge_async())

    async def _handle_vscode_script(self, code: str) -> dict[str, Any]:
        """Execute raw Python code via VS Code compat protocol."""
        if self._executor is None:
            return {"status": "error", "output": "Executor not initialized."}
        if not self._allow_unsafe_execution:
            return {"status": "error", "output": "Script execution is disabled (allow_unsafe_execution=false)."}
        try:
            output, exception, trace = await self._executor.execute(code)
            if exception is not None:
                import traceback as tb_mod
                return {
                    "status": "error",
                    "output": output,
                    "ename": type(exception).__name__,
                    "evalue": str(exception),
                    "traceback": tb_mod.format_exception(type(exception), exception, exception.__traceback__),
                }
            return {"status": "ok", "output": output}
        except Exception as exc:
            import traceback
            return {
                "status": "error",
                "output": "",
                "ename": type(exc).__name__,
                "evalue": str(exc),
                "traceback": traceback.format_exception(type(exc), exc, exc.__traceback__),
            }

    async def _handle_request(self, request: BridgeRequest) -> BridgeResponse:
        """Dispatch a typed bridge request."""
        if self._service is None or self._request_lock is None:
            return BridgeResponse.failure(
                request.request_id,
                "ExtensionNotReady",
                "Bridge service is not initialized.",
            )
        # Serialize bridge requests on Kit's event loop so stage/runtime state
        # mutations do not interleave across independent client connections.
        async with self._request_lock:
            return await self._service.dispatch(request)

    @staticmethod
    def _get_event_loop() -> asyncio.AbstractEventLoop:
        """Backward-compatible event loop getter for Kit runtimes."""
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            return asyncio.get_event_loop_policy().get_event_loop()

    def _reload_settings_from_kit(self) -> None:
        """Read current bridge settings from carb.settings."""
        settings = carb.settings.get_settings()
        self._host = str(settings.get("/exts/khemoo.simul.mcp/host"))
        self._port = int(settings.get("/exts/khemoo.simul.mcp/port"))
        self._allow_unsafe_execution = bool(
            settings.get("/exts/khemoo.simul.mcp/allow_unsafe_execution")
        )
        self._max_request_bytes = int(
            settings.get("/exts/khemoo.simul.mcp/max_request_bytes")
        )
        self._max_response_bytes = int(
            settings.get("/exts/khemoo.simul.mcp/max_response_bytes")
        )
        self._max_port_retries = int(
            settings.get("/exts/khemoo.simul.mcp/max_port_retries") or 10
        )
        self._discovery_dir = str(
            settings.get("/exts/khemoo.simul.mcp/discovery_dir") or "/tmp/simul-mcp"
        )

    def _schedule_task(self, coro: asyncio.Future) -> None:
        """Schedule a bridge lifecycle task on Kit's event loop."""
        loop = self._get_event_loop()
        loop.create_task(coro)

    def _sync_stop_bridge(self, timeout: float = 5.0) -> None:
        """Stop the bridge server synchronously during shutdown."""
        if self._server is None:
            return
        future = asyncio.run_coroutine_threadsafe(
            self._stop_bridge_async(), self._get_event_loop()
        )
        try:
            future.result(timeout=timeout)
        except Exception:
            pass

    async def _start_bridge_async(self) -> None:
        """Create and start the bridge service if it is not already running."""
        if self._server is not None:
            self._refresh_ui()
            return
        self._executor = ScriptExecutor(self._globals, self._globals)
        self._service = BridgeCommandService(
            executor=self._executor,
            allow_unsafe_execution=self._allow_unsafe_execution,
        )
        self._request_lock = asyncio.Lock()
        self._server = BridgeServerLifecycle(
            host=self._host,
            port=self._port,
            request_handler=self._handle_request,
            max_request_bytes=self._max_request_bytes,
            max_response_bytes=self._max_response_bytes,
            max_port_retries=self._max_port_retries,
            vscode_handler=self._handle_vscode_script,
        )
        await self._server.start()
        # Write actual bound port back to Carb settings
        actual_port = self._server.actual_port
        if actual_port != self._port:
            carb.log_info(
                f"Configured port {self._port} was in use, bound to {actual_port} instead."
            )
            settings = carb.settings.get_settings()
            settings.set("/exts/khemoo.simul.mcp/port", actual_port)
            self._port = actual_port
        # Write discovery file
        import os
        self._server.write_discovery_file(self._discovery_dir, os.getpid())
        carb.log_info(f"Simul MCP bridge serving at {self._server.address}")
        self._refresh_ui()

    async def _stop_bridge_async(self) -> None:
        """Stop the active bridge server and clear runtime state."""
        if self._server is None:
            self._refresh_ui()
            return
        self._server.remove_discovery_file()
        await self._server.stop()
        self._server = None
        self._executor = None
        self._service = None
        self._request_lock = None
        self._refresh_ui()

    async def _restart_bridge_async(self) -> None:
        """Restart the bridge server with current settings."""
        await self._stop_bridge_async()
        await self._start_bridge_async()

    def _refresh_ui(self) -> None:
        """Refresh the UI models if the settings window is available."""
        if self._ui_builder is not None:
            self._ui_builder.refresh()
