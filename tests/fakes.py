"""Test doubles shared across the suite.

``FakeFastMCP`` stands in for the real FastMCP instance when a test only needs
to see what the server registered. It records every tool with the keyword
arguments it was registered with, answers the handful of FastMCP calls the
server makes during construction, and mirrors FastMCP 3's ``local_provider``
component keys so the server's own tool listing works against it.

``StrictBlenderSession`` stands in for a Blender session behind the registered
tools; ``BlenderSessionAdapter`` hands it to the server.
"""

from __future__ import annotations

import copy
import inspect
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple

from simul.config import Settings

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
        self.resources: List[str] = []
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

    def resource(
        self, uri: str, *args: Any, **kwargs: Any
    ) -> Callable[[ToolFunction], ToolFunction]:
        """Record the resource's URI; the decorator hands the function back unchanged."""

        def decorator(func: ToolFunction) -> ToolFunction:
            self.resources.append(uri)
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


SessionCall = Tuple[str, Dict[str, Any]]


class StrictBlenderSession:
    """Blender session double spec'd on ``BlenderRuntimeSession``.

    It offers exactly the public ``BlenderRuntimeSession`` operations, as the
    attached-mode ``BlenderConnection`` and the add-on do, and each call must
    bind to that operation's signature. A call to anything else raises, which
    the tool envelope turns into an error payload, and is also kept in
    ``rejected``, so a test can fail on a tool that names a missing operation
    or passes arguments the operation does not take instead of passing on the
    error payload.

    Attributes:
        calls: ``(operation, arguments)`` per accepted call, the arguments bound
            to the operation's signature with its defaults applied.
        rejected: ``(name, reason)`` per refused attribute or call.
        responses: Payload returned per operation; others get ``default_response``.
        default_response: Payload of an operation without its own response.
    """

    def __init__(
        self,
        responses: Optional[Mapping[str, Dict[str, Any]]] = None,
        default_response: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.calls: List[SessionCall] = []
        self.rejected: List[Tuple[str, str]] = []
        self.responses: Dict[str, Dict[str, Any]] = dict(responses or {})
        self.default_response: Dict[str, Any] = dict(default_response or {})

    def __getattr__(self, method: str) -> Callable[..., Dict[str, Any]]:
        from simul.adapters.blender_runtime import BlenderRuntimeSession

        if method.startswith("__"):
            raise AttributeError(method)
        operation = getattr(BlenderRuntimeSession, method, None)
        if method.startswith("_") or method == "cleanup" or not callable(operation):
            self.rejected.append(
                (method, "not a public BlenderRuntimeSession operation")
            )
            raise AttributeError(f"BlenderRuntimeSession has no operation {method!r}")

        def invoke(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            try:
                bound = inspect.signature(operation).bind(None, *args, **kwargs)
            except TypeError as exc:
                self.rejected.append((method, str(exc)))
                raise
            bound.apply_defaults()
            arguments = dict(list(bound.arguments.items())[1:])
            self.calls.append((method, arguments))
            return copy.deepcopy(self.responses.get(method, self.default_response))

        return invoke


class BlenderSessionAdapter(AvailableAdapter):
    """Blender adapter double whose every session is ``session``.

    Attached mode runs the session on a worker thread, as the real adapter
    has the server do.
    """

    name: str = "blender"

    def __init__(self, settings: Settings, session: Optional[Any] = None) -> None:
        super().__init__(settings)
        self.session = session if session is not None else StrictBlenderSession()

    @property
    def session_blocks_io(self) -> bool:
        return self.settings.blender.mode == "attached"

    @contextmanager
    def create_session(self) -> Iterator[Any]:
        yield self.session
