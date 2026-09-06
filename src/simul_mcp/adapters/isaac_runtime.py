"""Isaac Sim backend adapter.

Isaac Sim is reached over a TCP socket rather than an in-process runtime, so
the adapter owns the socket client for the default instance and builds the
per-instance clients the server registers for multi-instance routing. Tool
calls do not open sessions on it: they go through ``IsaacTools`` and the
server's Isaac envelope, which route to the instance the MCP session picked.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, List, Optional

from ..config import Settings, get_settings
from ..logging import LoggerMixin
from .isaac_socket_client import IsaacSocketClient


class IsaacRuntimeAdapter(LoggerMixin):
    """Backend adapter over the Isaac Sim socket client."""

    name: str = "isaac"

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """
        Build the default instance's client from settings.

        Args:
            settings: Configuration settings.
        """
        self.settings: Settings = settings or get_settings()
        isaac = self.settings.isaac_sim
        self.client: IsaacSocketClient = self.build_client(
            socket_host=isaac.socket_host,
            socket_port=isaac.socket_port,
            socket_timeout=isaac.socket_timeout,
            bridge_enabled=isaac.bridge_enabled,
            bridge_host=isaac.bridge_host,
            bridge_port=isaac.bridge_port,
            bridge_timeout=isaac.bridge_timeout,
            bridge_fallback_to_vscode=isaac.bridge_fallback_to_vscode,
        )

    @contextmanager
    def create_session(self) -> Iterator[IsaacSocketClient]:
        """
        Yield the default instance's socket client.

        Yields:
            The client; it opens one connection per request, so there is
            nothing to release when the session ends.
        """
        yield self.client

    def is_available(self) -> bool:
        """
        Report whether the backend is wired up.

        Returns:
            Always True: the socket client has no import-time dependency, and
            reachability of the engine is a ping, not a capability.
        """
        return True

    def get_capabilities(self) -> List[str]:
        """
        Return the transports the default instance is configured with.

        Returns:
            ``["socket"]``, plus ``"bridge"`` when the bridge extension
            transport is enabled.
        """
        transports = ["socket"]
        if self.client.bridge_enabled:
            transports.append("bridge")
        return transports

    def close(self) -> None:
        """Nothing is held open between requests."""

    def build_client(
        self,
        *,
        socket_host: str,
        socket_port: int,
        socket_timeout: float,
        bridge_enabled: bool,
        bridge_host: Optional[str] = None,
        bridge_port: Optional[int] = None,
        bridge_timeout: Optional[float] = None,
        bridge_fallback_to_vscode: Optional[bool] = None,
        bridge_socket_path: Optional[str] = None,
        socket_protocol: Optional[str] = None,
        socket_auth_token: Optional[str] = None,
    ) -> IsaacSocketClient:
        """
        Create one bridge-aware client from per-instance values and settings defaults.

        Args:
            socket_host: Host of the stock Python socket.
            socket_port: Port of the stock Python socket.
            socket_timeout: Request timeout on the stock socket, in seconds.
            bridge_enabled: Prefer the bridge extension transport.
            bridge_host: Bridge host; the socket host when omitted.
            bridge_port: Bridge port; derived from the socket port when omitted.
            bridge_timeout: Bridge request timeout; the settings default when omitted.
            bridge_fallback_to_vscode: Fall back to the stock socket when the
                bridge is unreachable; the settings default when omitted.
            bridge_socket_path: Unix socket path advertised by a discovery file.
            socket_protocol: Stock socket flavour; the settings default when omitted.
            socket_auth_token: python_server token; the settings default when omitted.

        Returns:
            The configured client.
        """
        isaac = self.settings.isaac_sim
        return IsaacSocketClient(
            host=socket_host,
            port=socket_port,
            bridge_host=bridge_host or socket_host,
            bridge_port=(
                bridge_port if bridge_port is not None else self.bridge_port_for_socket(socket_port)
            ),
            bridge_socket_path=bridge_socket_path,
            bridge_timeout_seconds=(
                bridge_timeout if bridge_timeout is not None else isaac.bridge_timeout
            ),
            prefer_bridge=bridge_enabled,
            fallback_to_vscode=(
                bridge_fallback_to_vscode
                if bridge_fallback_to_vscode is not None
                else isaac.bridge_fallback_to_vscode
            ),
            timeout_seconds=socket_timeout,
            socket_protocol=socket_protocol or isaac.socket_protocol,
            auth_token=socket_auth_token if socket_auth_token is not None else isaac.socket_auth_token,
            bridge_failure_threshold=isaac.bridge_failure_threshold,
            bridge_cooldown_seconds=isaac.bridge_cooldown_seconds,
        )

    def bridge_port_for_socket(self, socket_port: int) -> int:
        """
        Derive an instance's bridge port from its stock socket port.

        Args:
            socket_port: The instance's stock socket port.

        Returns:
            The default bridge port shifted by the same offset, clamped to
            the unprivileged range.
        """
        isaac = self.settings.isaac_sim
        derived = isaac.bridge_port + (socket_port - isaac.socket_port)
        return max(1024, min(derived, 65535))

    def socket_port_for_bridge(self, bridge_port: int) -> int:
        """
        Derive an instance's stock socket port from its bridge port.

        Args:
            bridge_port: The instance's bridge port.

        Returns:
            The default socket port shifted by the same offset, clamped to
            the unprivileged range.
        """
        isaac = self.settings.isaac_sim
        derived = isaac.socket_port + (bridge_port - isaac.bridge_port)
        return max(1024, min(derived, 65535))
