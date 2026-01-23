"""
UI components for Isaac Sim MCP Server Extension.

This module provides UI widgets and panels for the MCP server extension,
including server control, status monitoring, and configuration.
"""

from typing import Optional, Callable, Dict, Any

try:
    import omni.ui as ui
    import carb
    OMNI_AVAILABLE = True
except ImportError:
    OMNI_AVAILABLE = False
    ui = None
    carb = None


class ServerStatusWidget:
    """Widget for displaying server status."""

    def __init__(self, parent_frame: "ui.Frame"):
        """Initialize server status widget."""
        self.parent_frame = parent_frame
        self.status_label: Optional[ui.Label] = None
        self.isaac_status_label: Optional[ui.Label] = None
        self.tools_count_label: Optional[ui.Label] = None

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the status widget UI."""
        with self.parent_frame:
            with ui.VStack(spacing=5):
                ui.Label("Server Status", style={"font_weight": "bold"})

                with ui.HStack():
                    ui.Label("MCP Server:", width=100)
                    self.status_label = ui.Label(
                        "Stopped",
                        style={"color": 0xFF808080},
                    )

                with ui.HStack():
                    ui.Label("Isaac Sim:", width=100)
                    self.isaac_status_label = ui.Label(
                        "Checking...",
                        style={"color": 0xFFFFFF00},
                    )

                with ui.HStack():
                    ui.Label("Available Tools:", width=100)
                    self.tools_count_label = ui.Label(
                        "0",
                        style={"color": 0xFF808080},
                    )

    def update_server_status(self, running: bool) -> None:
        """Update server running status."""
        if not self.status_label:
            return

        if running:
            self.status_label.text = "Running"
            self.status_label.style = {"color": 0xFF00FF00}
        else:
            self.status_label.text = "Stopped"
            self.status_label.style = {"color": 0xFF808080}

    def update_isaac_status(self, available: bool) -> None:
        """Update Isaac Sim availability status."""
        if not self.isaac_status_label:
            return

        if available:
            self.isaac_status_label.text = "Available"
            self.isaac_status_label.style = {"color": 0xFF00FF00}
        else:
            self.isaac_status_label.text = "Not Available"
            self.isaac_status_label.style = {"color": 0xFFFF0000}

    def update_tools_count(self, count: int) -> None:
        """Update available tools count."""
        if not self.tools_count_label:
            return

        self.tools_count_label.text = str(count)
        if count > 0:
            self.tools_count_label.style = {"color": 0xFF00FF00}
        else:
            self.tools_count_label.style = {"color": 0xFFFF0000}


class ServerControlWidget:
    """Widget for server control buttons and configuration."""

    def __init__(
        self,
        parent_frame: "ui.Frame",
        start_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        stop_callback: Optional[Callable[[], None]] = None,
    ):
        """Initialize server control widget."""
        self.parent_frame = parent_frame
        self.start_callback = start_callback
        self.stop_callback = stop_callback

        self.start_button: Optional[ui.Button] = None
        self.stop_button: Optional[ui.Button] = None
        self.transport_combo: Optional[ui.ComboBox] = None
        self.log_level_combo: Optional[ui.ComboBox] = None

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the control widget UI."""
        with self.parent_frame:
            with ui.VStack(spacing=10):
                ui.Label("Server Control", style={"font_weight": "bold"})

                with ui.VStack(spacing=5):
                    with ui.HStack():
                        ui.Label("Transport:", width=80)
                        self.transport_combo = ui.ComboBox(0, "stdio", "sse")

                    with ui.HStack():
                        ui.Label("Log Level:", width=80)
                        self.log_level_combo = ui.ComboBox(
                            1,
                            "DEBUG",
                            "INFO",
                            "WARNING",
                            "ERROR",
                        )

                ui.Separator()

                with ui.HStack(spacing=10):
                    self.start_button = ui.Button(
                        "Start Server",
                        clicked_fn=self._on_start_clicked,
                        style={"background_color": 0xFF00AA00},
                    )

                    self.stop_button = ui.Button(
                        "Stop Server",
                        clicked_fn=self._on_stop_clicked,
                        enabled=False,
                        style={"background_color": 0xFFAA0000},
                    )

    def _on_start_clicked(self) -> None:
        """Handle start button click."""
        if self.start_callback:
            self.start_callback(self.get_configuration())

    def _on_stop_clicked(self) -> None:
        """Handle stop button click."""
        if self.stop_callback:
            self.stop_callback()

    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration from UI."""
        transport_options = ["stdio", "sse"]
        log_level_options = ["DEBUG", "INFO", "WARNING", "ERROR"]

        return {
            "transport": transport_options[
                self.transport_combo.model.get_item_value_model().get_value_as_int()
            ],
            "log_level": log_level_options[
                self.log_level_combo.model.get_item_value_model().get_value_as_int()
            ],
        }

    def update_button_states(self, server_running: bool) -> None:
        """Update button enabled states."""
        if self.start_button and self.stop_button:
            self.start_button.enabled = not server_running
            self.stop_button.enabled = server_running


class ToolsInfoWidget:
    """Widget for displaying available tools information."""

    def __init__(self, parent_frame: "ui.Frame"):
        """Initialize tools info widget."""
        self.parent_frame = parent_frame
        self.tools_tree: Optional[ui.TreeView] = None

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the tools info widget UI."""
        with self.parent_frame:
            with ui.VStack(spacing=5):
                ui.Label("Available Tools", style={"font_weight": "bold"})

                self.tools_tree = ui.TreeView(
                    height=150,
                    root_visible=False,
                    header_visible=True,
                )

                self.tools_tree.column_widths = [200, 100, 80]
                self.tools_tree.columns = [
                    ui.TreeView.Column("Tool Name", width=200),
                    ui.TreeView.Column("Category", width=100),
                    ui.TreeView.Column("Status", width=80),
                ]

    def update_tools(self, tools_info: Dict[str, Any]) -> None:
        """Update tools display."""
        _ = tools_info


class LogViewerWidget:
    """Widget for viewing server logs."""

    def __init__(self, parent_frame: "ui.Frame"):
        """Initialize log viewer widget."""
        self.parent_frame = parent_frame
        self.log_text: Optional[ui.StringField] = None
        self.auto_scroll = True

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the log viewer widget UI."""
        with self.parent_frame:
            with ui.VStack(spacing=5):
                with ui.HStack():
                    ui.Label("Server Logs", style={"font_weight": "bold"})
                    ui.Spacer()
                    ui.Button("Clear", clicked_fn=self._clear_logs, width=60)
                    auto_scroll_check = ui.CheckBox()
                    auto_scroll_check.model.set_value(self.auto_scroll)
                    auto_scroll_check.model.add_value_changed_fn(
                        lambda m: setattr(self, "auto_scroll", m.get_value_as_bool())
                    )
                    ui.Label("Auto-scroll")

                with ui.ScrollingFrame(height=200):
                    self.log_text = ui.StringField(
                        multiline=True,
                        readonly=True,
                        style={"font_family": "monospace", "font_size": 10},
                    )

    def _clear_logs(self) -> None:
        """Clear the log display."""
        if self.log_text:
            self.log_text.model.set_value("")

    def add_log_message(self, message: str) -> None:
        """Add a log message to the display."""
        if not self.log_text:
            return

        current_text = self.log_text.model.get_value_as_string()
        new_text = current_text + message + "\n"
        self.log_text.model.set_value(new_text)

        if self.auto_scroll:
            return


class MCPServerPanel:
    """Main panel for MCP server control."""

    def __init__(self, window: "ui.Window"):
        """Initialize MCP server panel."""
        self.window = window
        self.status_widget: Optional[ServerStatusWidget] = None
        self.control_widget: Optional[ServerControlWidget] = None
        self.tools_widget: Optional[ToolsInfoWidget] = None
        self.log_widget: Optional[LogViewerWidget] = None
        self.start_server_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self.stop_server_callback: Optional[Callable[[], None]] = None

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the main panel UI."""
        with self.window.frame:
            with ui.VStack(spacing=10):
                ui.Label(
                    "Isaac Sim MCP Server",
                    style={"font_size": 18, "font_weight": "bold"},
                )

                ui.Separator()

                with ui.CollapsableFrame("Status", collapsed=False):
                    self.status_widget = ServerStatusWidget(ui.Frame())

                with ui.CollapsableFrame("Control", collapsed=False):
                    self.control_widget = ServerControlWidget(
                        ui.Frame(),
                        start_callback=self._on_start_server,
                        stop_callback=self._on_stop_server,
                    )

                with ui.CollapsableFrame("Tools", collapsed=True):
                    self.tools_widget = ToolsInfoWidget(ui.Frame())

                with ui.CollapsableFrame("Logs", collapsed=True):
                    self.log_widget = LogViewerWidget(ui.Frame())

    def _on_start_server(self, config: Dict[str, Any]) -> None:
        """Handle start server request."""
        if self.start_server_callback:
            self.start_server_callback(config)

    def _on_stop_server(self) -> None:
        """Handle stop server request."""
        if self.stop_server_callback:
            self.stop_server_callback()

    def set_start_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Set start server callback."""
        self.start_server_callback = callback

    def set_stop_callback(self, callback: Callable[[], None]) -> None:
        """Set stop server callback."""
        self.stop_server_callback = callback

    def update_server_status(self, running: bool) -> None:
        """Update server status display."""
        if self.status_widget:
            self.status_widget.update_server_status(running)

        if self.control_widget:
            self.control_widget.update_button_states(running)

    def update_isaac_status(self, available: bool) -> None:
        """Update Isaac Sim status display."""
        if self.status_widget:
            self.status_widget.update_isaac_status(available)

    def update_tools_info(self, tools_info: Dict[str, Any]) -> None:
        """Update tools information display."""
        if self.status_widget:
            total_tools = tools_info.get("total_tools", 0)
            self.status_widget.update_tools_count(total_tools)

        if self.tools_widget:
            self.tools_widget.update_tools(tools_info)

    def add_log_message(self, message: str) -> None:
        """Add a log message to the display."""
        if self.log_widget:
            self.log_widget.add_log_message(message)
