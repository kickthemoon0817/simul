"""
Isaac Sim MCP Server Extension.

Provides a UI panel for controlling the MCP server inside Isaac Sim.
"""

from typing import Optional, Dict, Any

try:
    import omni.ext
    import omni.ui as ui
    import carb

    OMNI_AVAILABLE = True
except ImportError:
    OMNI_AVAILABLE = False
    omni = None
    ui = None
    carb = None

from .lifecycle import get_lifecycle_manager
from .ui import MCPServerPanel


class IsaacMCPServerExtension(omni.ext.IExt):
    """Isaac Sim MCP Server Extension."""

    def __init__(self) -> None:
        self._window: Optional[ui.Window] = None
        self._panel: Optional[MCPServerPanel] = None
        self._lifecycle = None

    def on_startup(self, ext_id: str) -> None:
        """Called when the extension starts up."""
        if not OMNI_AVAILABLE:
            return

        self._lifecycle = get_lifecycle_manager()
        self._create_ui()
        self._wire_callbacks()

        auto_start = carb.settings.get_settings().get(
            "/exts/khemoo.simul.mcp/server/auto_start",
            False,
        )
        if auto_start:
            self._start_server({"transport": "stdio", "log_level": "INFO"})

    def on_shutdown(self) -> None:
        """Called when the extension shuts down."""
        if not OMNI_AVAILABLE:
            return

        if self._lifecycle:
            self._lifecycle.cleanup()
            self._lifecycle = None

        if self._window:
            self._window.destroy()
            self._window = None

    def _create_ui(self) -> None:
        """Create the UI window and panel."""
        settings = carb.settings.get_settings()
        width = settings.get("/exts/khemoo.simul.mcp/ui/window_width", 400)
        height = settings.get("/exts/khemoo.simul.mcp/ui/window_height", 300)

        self._window = ui.Window(
            "Isaac Sim MCP Server",
            width=width,
            height=height,
            flags=ui.WINDOW_FLAGS_NO_COLLAPSE,
        )
        self._panel = MCPServerPanel(self._window)

    def _wire_callbacks(self) -> None:
        """Wire lifecycle callbacks into the panel."""
        if not self._panel or not self._lifecycle:
            return

        self._panel.set_start_callback(self._start_server)
        self._panel.set_stop_callback(self._stop_server)
        self._lifecycle.add_status_callback(self._handle_status_change)
        self._lifecycle.add_error_callback(self._panel.add_log_message)
        self._lifecycle.add_log_callback(self._panel.add_log_message)

        info = self._lifecycle.get_server_info()
        self._panel.update_server_status(info.get("running", False))
        self._panel.update_isaac_status(info.get("isaac_available", False))
        self._panel.update_tools_info(info)

    def _start_server(self, config: Dict[str, Any]) -> None:
        """Start the MCP server."""
        if self._lifecycle:
            self._lifecycle.start_server(config)

    def _stop_server(self) -> None:
        """Stop the MCP server."""
        if self._lifecycle:
            self._lifecycle.stop_server()

    def _handle_status_change(self, running: bool) -> None:
        """Handle lifecycle status changes."""
        if not self._panel or not self._lifecycle:
            return

        self._panel.update_server_status(running)
        info = self._lifecycle.get_server_info()
        self._panel.update_isaac_status(info.get("isaac_available", False))
        self._panel.update_tools_info(info)


def get_extension() -> IsaacMCPServerExtension:
    """Return the extension instance."""
    return IsaacMCPServerExtension()
