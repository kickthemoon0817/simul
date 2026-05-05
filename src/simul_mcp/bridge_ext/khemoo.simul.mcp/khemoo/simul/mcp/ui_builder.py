"""UI controls for the Simul MCP Isaac bridge extension."""

from __future__ import annotations

from typing import Any, Callable
import weakref

try:
    import carb
    import omni.ui as ui
    from omni.kit.menu.utils import MenuItemDescription, add_menu_items, remove_menu_items

    OMNI_UI_AVAILABLE = True
except Exception:  # pragma: no cover - import smoke tests exercise this path
    carb = None
    ui = None
    MenuItemDescription = None
    add_menu_items = None
    remove_menu_items = None
    OMNI_UI_AVAILABLE = False


class BridgeUIBuilder:
    """Manage the extension window and menu actions for bridge settings."""

    def __init__(
        self,
        title: str,
        get_state_fn: Callable[[], dict[str, Any]],
        apply_settings_fn: Callable[[str, int, bool, int, int], None],
        start_fn: Callable[[], None],
        stop_fn: Callable[[], None],
        restart_fn: Callable[[], None],
    ) -> None:
        self._title = title
        self._get_state = get_state_fn
        self._apply_settings = apply_settings_fn
        self._start = start_fn
        self._stop = stop_fn
        self._restart = restart_fn
        self._menu_items: list[Any] = []
        self._window = None
        self._status_label = None
        self._host_model = ui.SimpleStringModel("") if OMNI_UI_AVAILABLE else None
        self._port_model = ui.SimpleIntModel(8229) if OMNI_UI_AVAILABLE else None
        self._unsafe_model = ui.SimpleBoolModel(True) if OMNI_UI_AVAILABLE else None
        self._max_request_model = ui.SimpleIntModel(1024 * 1024) if OMNI_UI_AVAILABLE else None
        self._max_response_model = ui.SimpleIntModel(10 * 1024 * 1024) if OMNI_UI_AVAILABLE else None

    def startup(self) -> None:
        """Create the window and menu item when Isaac UI APIs are available."""
        if not OMNI_UI_AVAILABLE or ui is None:
            return
        self._window = ui.Window(self._title, width=500, height=320, visible=False)
        self._window.set_visibility_changed_fn(self._on_window_visibility_changed)
        if MenuItemDescription is not None and add_menu_items is not None:
            proxy = weakref.proxy(self)
            self._menu_items = [
                MenuItemDescription(name=self._title, onclick_fn=lambda *_: proxy._toggle_window())
            ]
            add_menu_items(self._menu_items, "Window")

    def shutdown(self) -> None:
        """Destroy UI resources and remove the menu item."""
        if remove_menu_items is not None and self._menu_items:
            remove_menu_items(self._menu_items, "Window")
        self._menu_items = []
        if self._window is not None:
            self._window.destroy()
            self._window = None
        self._status_label = None

    def refresh(self) -> None:
        """Refresh the form from the current runtime state."""
        if not OMNI_UI_AVAILABLE or self._window is None:
            return
        self._sync_models(self._get_state())
        if self._window.visible:
            self._build_ui()

    def _build_ui(self) -> None:
        """Build the bridge management layout."""
        if self._window is None or ui is None:
            return

        state = self._get_state()
        self._sync_models(state)
        with self._window.frame:
            with ui.VStack(spacing=8, height=0):
                ui.Label("Bridge transport controls for Simul MCP.", height=0)
                ui.Label(
                    "Typed actions use the bridge. Raw scripts still use the VS Code socket.",
                    height=0,
                    word_wrap=True,
                )

                with ui.HStack(spacing=6, height=0):
                    ui.Label("Status", width=120, height=0)
                    self._status_label = ui.Label(self._format_status(state), height=0, word_wrap=True)

                with ui.HStack(spacing=6, height=0):
                    ui.Label("Host", width=120, height=0)
                    ui.StringField(model=self._host_model, height=0)

                with ui.HStack(spacing=6, height=0):
                    ui.Label("Port", width=120, height=0)
                    ui.IntField(model=self._port_model, height=0, width=140)

                with ui.HStack(spacing=6, height=0):
                    ui.Label("Allow execute_script", width=120, height=0)
                    ui.CheckBox(model=self._unsafe_model, width=24, height=0)
                    ui.Label("Unsafe raw bridge execution", height=0)

                with ui.HStack(spacing=6, height=0):
                    ui.Label("Max Request Bytes", width=120, height=0)
                    ui.IntField(model=self._max_request_model, height=0, width=180)

                with ui.HStack(spacing=6, height=0):
                    ui.Label("Max Response Bytes", width=120, height=0)
                    ui.IntField(model=self._max_response_model, height=0, width=180)

                with ui.HStack(spacing=6, height=0):
                    ui.Button("Refresh", clicked_fn=self._refresh_clicked, height=0)
                    ui.Button("Apply + Restart", clicked_fn=self._apply_clicked, height=0)
                    ui.Button("Start", clicked_fn=self._start_clicked, height=0)
                    ui.Button("Stop", clicked_fn=self._stop_clicked, height=0)
                    ui.Button("Restart", clicked_fn=self._restart_clicked, height=0)

    def _toggle_window(self) -> None:
        """Show or hide the settings window."""
        if self._window is None:
            return
        self._window.visible = not self._window.visible

    def _on_window_visibility_changed(self, visible: bool) -> None:
        """Rebuild the window when it becomes visible."""
        if visible:
            self._build_ui()

    def _refresh_clicked(self) -> None:
        """Refresh the window state from current settings."""
        self.refresh()

    def _apply_clicked(self) -> None:
        """Apply form values and restart the bridge."""
        if self._host_model is None:
            return
        self._apply_settings(
            self._host_model.get_value_as_string(),
            int(self._port_model.get_value_as_int()),
            bool(self._unsafe_model.get_value_as_bool()),
            int(self._max_request_model.get_value_as_int()),
            int(self._max_response_model.get_value_as_int()),
        )
        self.refresh()

    def _start_clicked(self) -> None:
        """Start the bridge with current settings."""
        self._start()
        self.refresh()

    def _stop_clicked(self) -> None:
        """Stop the bridge server."""
        self._stop()
        self.refresh()

    def _restart_clicked(self) -> None:
        """Restart the bridge server."""
        self._restart()
        self.refresh()

    def _sync_models(self, state: dict[str, Any]) -> None:
        """Update form models from the current bridge state."""
        if not OMNI_UI_AVAILABLE:
            return
        self._host_model.set_value(str(state.get("host", "127.0.0.1")))
        self._port_model.set_value(int(state.get("port", 8229)))
        self._unsafe_model.set_value(bool(state.get("allow_unsafe_execution", True)))
        self._max_request_model.set_value(int(state.get("max_request_bytes", 1024 * 1024)))
        self._max_response_model.set_value(int(state.get("max_response_bytes", 10 * 1024 * 1024)))
        if self._status_label is not None:
            self._status_label.text = self._format_status(state)

    def _format_status(self, state: dict[str, Any]) -> str:
        """Render the runtime bridge state as a short status string."""
        status = "running" if state.get("running") else "stopped"
        address = state.get("address", "")
        unsafe = "enabled" if state.get("allow_unsafe_execution") else "disabled"
        return f"{status} @ {address} | execute_script {unsafe}"
