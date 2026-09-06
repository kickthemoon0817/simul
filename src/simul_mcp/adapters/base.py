"""The interface every backend adapter presents to the MCP server.

The server only ever asks an adapter four things: whether the runtime is
wired up, a session to run one tool call in, the capability names to report,
and to release its resources at shutdown. A new backend implements this
protocol and adds one entry to ``simul_mcp.mcp.backends.BACKENDS``.
"""

from __future__ import annotations

from typing import Any, ContextManager, List, Protocol, runtime_checkable


@runtime_checkable
class BackendAdapter(Protocol):
    """Structural interface of a backend adapter.

    Attributes:
        name: The backend's registry name (``isaac``, ``usd``, ``blender``,
            ``unreal``); also the token the ROUTING instructions use.
    """

    name: str

    def is_available(self) -> bool:
        """Report whether the backend is wired up in this process.

        Returns:
            True when the runtime's dependencies import and settings enable
            it. This is not a liveness probe of the engine.
        """

    def create_session(self) -> ContextManager[Any]:
        """Open a session for one tool call.

        Returns:
            A context manager yielding the session object tools call methods
            on; closing it releases whatever the call acquired.
        """

    def get_capabilities(self) -> List[str]:
        """Return the capability names to report for the backend.

        Returns:
            Capability names; empty when the backend is unavailable.
        """

    def close(self) -> None:
        """Release every resource the adapter holds across sessions."""
