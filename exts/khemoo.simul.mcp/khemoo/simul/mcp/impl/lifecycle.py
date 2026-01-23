"""
Lifecycle management for Isaac Sim MCP Server Extension.

This module handles the lifecycle of the MCP server within Isaac Sim,
including startup, shutdown, and runtime management.
"""

import asyncio
import threading
import time
from typing import Optional, Dict, Any, Callable
from pathlib import Path

try:
    import omni.kit.app
    import carb

    OMNI_AVAILABLE = True
except ImportError:
    OMNI_AVAILABLE = False
    omni = None
    carb = None

import sys

extension_path = Path(__file__).parent
src_path = extension_path.parents[6] / "src"
sys.path.insert(0, str(src_path))

try:
    from simul_mcp.config import Settings, get_settings
    from simul_mcp.logging import setup_logging, get_logger
    from simul_mcp.mcp.server import IsaacMCPServer
    from simul_mcp.mcp.tools.registry import get_tool_registry
    from simul_mcp.adapters import is_isaac_available

    MCP_AVAILABLE = True
except ImportError as exc:
    print(f"Warning: Could not import MCP components: {exc}")
    MCP_AVAILABLE = False


class ServerLifecycleManager:
    """Manages the lifecycle of the MCP server within Isaac Sim."""

    def __init__(self) -> None:
        """Initialize the lifecycle manager."""
        self._server: Optional[IsaacMCPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._server_loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False
        self._settings: Optional[Settings] = None
        self._logger = None

        self._status_callbacks: list[Callable[[bool], None]] = []
        self._error_callbacks: list[Callable[[str], None]] = []
        self._log_callbacks: list[Callable[[str], None]] = []

        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_running = False

        if MCP_AVAILABLE:
            self._initialize()

    def _initialize(self) -> None:
        """Initialize the lifecycle manager."""
        try:
            self._settings = get_settings()
            setup_logging(self._settings.logging)
            self._logger = get_logger(__name__)
            self._logger.info("Server lifecycle manager initialized")
        except Exception as exc:
            print(f"Error initializing lifecycle manager: {exc}")

    def add_status_callback(self, callback: Callable[[bool], None]) -> None:
        """Add a callback for server status changes."""
        self._status_callbacks.append(callback)

    def add_error_callback(self, callback: Callable[[str], None]) -> None:
        """Add a callback for server errors."""
        self._error_callbacks.append(callback)

    def add_log_callback(self, callback: Callable[[str], None]) -> None:
        """Add a callback for log messages."""
        self._log_callbacks.append(callback)

    def start_server(self, config: Dict[str, Any]) -> bool:
        """Start the MCP server."""
        if self._running:
            self._logger.warning("Server is already running")
            return False

        if not MCP_AVAILABLE:
            error_msg = "MCP components not available"
            self._notify_error(error_msg)
            return False

        try:
            if "log_level" in config:
                self._settings.logging.level = config["log_level"]
                setup_logging(self._settings.logging)

            self._server = IsaacMCPServer(self._settings)

            transport = config.get("transport", "stdio")
            self._server_thread = threading.Thread(
                target=self._run_server,
                args=(transport,),
                daemon=True,
                name="MCP-Server-Thread",
            )
            self._server_thread.start()

            self._start_monitoring()

            time.sleep(0.5)

            if self._running:
                self._logger.info(
                    "MCP server started successfully with %s transport",
                    transport,
                )
                self._notify_status(True)
                return True

            self._logger.error("MCP server failed to start")
            return False

        except Exception as exc:
            error_msg = f"Error starting server: {exc}"
            self._logger.error(error_msg)
            self._notify_error(error_msg)
            return False

    def stop_server(self) -> bool:
        """Stop the MCP server."""
        if not self._running:
            self._logger.warning("Server is not running")
            return False

        try:
            self._logger.info("Stopping MCP server...")
            self._stop_monitoring()
            self._running = False

            if self._server_loop and not self._server_loop.is_closed():
                self._server_loop.call_soon_threadsafe(self._cancel_server_tasks)

            if self._server_thread and self._server_thread.is_alive():
                self._server_thread.join(timeout=5.0)
                if self._server_thread.is_alive():
                    self._logger.warning("Server thread did not stop gracefully")

            self._server = None
            self._server_thread = None
            self._server_loop = None

            self._logger.info("MCP server stopped")
            self._notify_status(False)
            return True

        except Exception as exc:
            error_msg = f"Error stopping server: {exc}"
            self._logger.error(error_msg)
            self._notify_error(error_msg)
            return False

    def _run_server(self, transport: str) -> None:
        """Run the MCP server in a separate thread."""
        try:
            self._server_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._server_loop)
            self._running = True
            self._server_loop.run_until_complete(self._server.run(transport))
        except asyncio.CancelledError:
            self._logger.info("Server cancelled")
        except Exception as exc:
            error_msg = f"Server thread error: {exc}"
            self._logger.error(error_msg)
            self._notify_error(error_msg)
        finally:
            self._running = False
            if self._server_loop and not self._server_loop.is_closed():
                self._server_loop.close()

    def _cancel_server_tasks(self) -> None:
        """Cancel all tasks in the server event loop."""
        try:
            if self._server_loop:
                tasks = [
                    task
                    for task in asyncio.all_tasks(self._server_loop)
                    if not task.done()
                ]
                for task in tasks:
                    task.cancel()
                self._server_loop.stop()
        except Exception as exc:
            self._logger.error("Error cancelling server tasks: %s", exc)

    def _start_monitoring(self) -> None:
        """Start server monitoring."""
        if self._monitor_running:
            return

        self._monitor_running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_server,
            daemon=True,
            name="MCP-Monitor-Thread",
        )
        self._monitor_thread.start()

    def _stop_monitoring(self) -> None:
        """Stop server monitoring."""
        self._monitor_running = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)

    def _monitor_server(self) -> None:
        """Monitor server health and status."""
        while self._monitor_running:
            try:
                if self._server_thread and not self._server_thread.is_alive():
                    if self._running:
                        self._logger.error("Server thread died unexpectedly")
                        self._running = False
                        self._notify_status(False)
                        self._notify_error("Server thread died unexpectedly")
                time.sleep(1.0)
            except Exception as exc:
                self._logger.error("Monitor error: %s", exc)
                time.sleep(5.0)

    def _notify_status(self, running: bool) -> None:
        """Notify status callbacks."""
        for callback in self._status_callbacks:
            try:
                if OMNI_AVAILABLE:
                    omni.kit.app.get_app().next_update_async(lambda: callback(running))
                else:
                    callback(running)
            except Exception as exc:
                if self._logger:
                    self._logger.error("Error in status callback: %s", exc)

    def _notify_error(self, error_msg: str) -> None:
        """Notify error callbacks."""
        for callback in self._error_callbacks:
            try:
                if OMNI_AVAILABLE:
                    omni.kit.app.get_app().next_update_async(
                        lambda: callback(error_msg)
                    )
                else:
                    callback(error_msg)
            except Exception as exc:
                if self._logger:
                    self._logger.error("Error in error callback: %s", exc)

    def _notify_log(self, log_msg: str) -> None:
        """Notify log callbacks."""
        for callback in self._log_callbacks:
            try:
                if OMNI_AVAILABLE:
                    omni.kit.app.get_app().next_update_async(lambda: callback(log_msg))
                else:
                    callback(log_msg)
            except Exception as exc:
                if self._logger:
                    self._logger.error("Error in log callback: %s", exc)

    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running

    def get_server_info(self) -> Dict[str, Any]:
        """Get server information."""
        info = {
            "running": self._running,
            "isaac_available": is_isaac_available() if MCP_AVAILABLE else False,
            "mcp_available": MCP_AVAILABLE,
            "server_thread_alive": self._server_thread.is_alive()
            if self._server_thread
            else False,
            "monitor_running": self._monitor_running,
        }

        if MCP_AVAILABLE and self._server:
            try:
                registry = get_tool_registry(self._settings)
                capabilities = registry.get_capabilities()
                info.update(capabilities)
            except Exception as exc:
                if self._logger:
                    self._logger.error("Error getting server capabilities: %s", exc)

        return info

    def restart_server(self, config: Dict[str, Any]) -> bool:
        """Restart the server with new configuration."""
        self._logger.info("Restarting MCP server...")

        if self._running:
            if not self.stop_server():
                return False

        time.sleep(1.0)
        return self.start_server(config)

    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if self._running:
                self.stop_server()

            self._status_callbacks.clear()
            self._error_callbacks.clear()
            self._log_callbacks.clear()

            if self._logger:
                self._logger.info("Server lifecycle manager cleaned up")
        except Exception as exc:
            print(f"Error during cleanup: {exc}")


_lifecycle_manager: Optional[ServerLifecycleManager] = None


def get_lifecycle_manager() -> ServerLifecycleManager:
    """Get the global lifecycle manager instance."""
    global _lifecycle_manager
    if _lifecycle_manager is None:
        _lifecycle_manager = ServerLifecycleManager()
    return _lifecycle_manager
