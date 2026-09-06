"""Test doubles shared across the suite.

``FakeFastMCP`` stands in for the real FastMCP instance when a test only needs
to see what the server registered. It records every tool with the keyword
arguments it was registered with, answers the handful of FastMCP calls the
server makes during construction, and mirrors FastMCP 3's ``local_provider``
component keys so the server's own tool listing works against it.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

from simul_mcp.config import Settings

ToolFunction = Callable[..., Any]


class FakeFastMCP:
    """FastMCP double that records tool registrations instead of serving them."""

    def __init__(self, name: str = "", version: str = "", **kwargs: Any) -> None:
        self.name = name
        self.version = version
        self.description: Optional[str] = kwargs.get("description")
        self.instructions: Optional[str] = kwargs.get("instructions")
        self.tools: List[SimpleNamespace] = []
        self.by_name: Dict[str, ToolFunction] = {}
        self.local_provider = SimpleNamespace(_components={})
        self.run_calls: List[Dict[str, Any]] = []

    def tool(self, name: str, **kwargs: Any) -> Callable[[ToolFunction], ToolFunction]:
        """Return a decorator that records the tool and hands the function back unchanged.

        Args:
            name: Registered tool name.
            **kwargs: Every other registration argument (description, annotations, ...).

        Returns:
            The recording decorator.
        """

        def decorator(func: ToolFunction) -> ToolFunction:
            self.tools.append(SimpleNamespace(name=name, func=func, kwargs=kwargs))
            self.by_name[name] = func
            self.local_provider._components[f"tool:{name}@"] = func
            return func

        return decorator

    def get_tools(self) -> List[SimpleNamespace]:
        """Return the recorded tools, mirroring the FastMCP method of the same name."""
        return self.tools

    async def get_tool(self, name: str) -> Optional[SimpleNamespace]:
        """Return an object carrying the tool's annotations, as the real ``get_tool`` does."""
        for tool in self.tools:
            if tool.name == name:
                return SimpleNamespace(annotations=tool.kwargs.get("annotations"))
        return None

    def resource(self, *args: Any, **kwargs: Any) -> Callable[[ToolFunction], ToolFunction]:
        """Accept a resource registration and return the function unchanged."""

        def decorator(func: ToolFunction) -> ToolFunction:
            return func

        return decorator

    def add_middleware(self, middleware: Any) -> None:
        """Accept the request-context middleware the server installs."""

    async def run_async(self, **kwargs: Any) -> None:
        """Record how the server asked to be run instead of serving."""
        self.run_calls.append(kwargs)


class AvailableAdapter:
    """Backend adapter stub that reports itself available so its tools register."""

    name: str = "stub"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.closed = False

    def is_available(self) -> bool:
        return True

    def get_capabilities(self) -> List[str]:
        return ["stub_capability"]

    def create_session(self) -> Any:
        raise NotImplementedError("registration-only stub")

    def close(self) -> None:
        self.closed = True
