"""Tool metadata declared on the method that implements the tool.

An MCP tool is one implementation method plus the listing metadata a client
sees: the registered name, the description, and the safety hints. Keeping the
metadata on the method itself, as a ``@tool_meta`` decorator, means the
registration layer can derive the tool's parameters and defaults from the
method signature and its parameter descriptions from the docstring, so a tool
is written once rather than restated per registration.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, FrozenSet, Iterator, Optional, Sequence, Tuple, TypeVar

ToolMethod = TypeVar("ToolMethod", bound=Callable[..., Any])

TOOL_META_ATTRIBUTE = "__tool_meta__"


@dataclass(frozen=True)
class SandboxedPath:
    """A path parameter the MCP boundary checks against the sandbox before dispatch.

    The authoritative check lives in the tools layer; this one turns a denied
    path into an error envelope without opening a connection.
    """

    parameter: str
    write: bool = False


@dataclass(frozen=True)
class DeprecatedAlias:
    """An extra tool parameter kept for callers that still pass the old name."""

    alias: str
    target: str

    @property
    def description(self) -> str:
        """Schema description of the alias parameter."""
        return f"Deprecated alias of {self.target}; ignored when {self.target} is given."


@dataclass(frozen=True)
class ToolMeta:
    """Listing metadata and dispatch rules for one tool.

    Attributes:
        name: Registered tool name.
        description: Tool description shown in the listing.
        read_only: The tool does not modify its environment.
        destructive: The tool overwrites or deletes existing state or files, or
            runs arbitrary code.
        idempotent: Repeating the call with the same arguments has no
            additional effect.
        open_world: The tool talks to an external process or the network.
        script: The tool runs agent-authored code from its ``code`` parameter;
            it is registered only when script execution is allowed and its
            usage record carries the code's digest instead of the code.
        bypasses_instance_lock: The tool skips the per-instance lock and the
            claim check because the call it acts on is the one holding them.
        sandboxed_paths: Path parameters checked against the sandbox before
            the call is dispatched.
        deprecated_aliases: Extra parameters accepted under an old name.
        hidden_parameters: Implementation parameters the tool does not expose;
            the implementation default applies.
    """

    name: str
    description: str
    read_only: bool
    destructive: bool = False
    idempotent: bool = False
    open_world: bool = True
    script: bool = False
    bypasses_instance_lock: bool = False
    sandboxed_paths: Tuple[SandboxedPath, ...] = ()
    deprecated_aliases: Tuple[DeprecatedAlias, ...] = ()
    hidden_parameters: FrozenSet[str] = frozenset()


def tool_meta(
    *,
    name: str,
    description: str,
    read_only: bool,
    destructive: bool = False,
    idempotent: bool = False,
    open_world: bool = True,
    script: bool = False,
    bypasses_instance_lock: bool = False,
    sandboxed_paths: Sequence[SandboxedPath] = (),
    deprecated_aliases: Sequence[DeprecatedAlias] = (),
    hidden_parameters: Sequence[str] = (),
) -> Callable[[ToolMethod], ToolMethod]:
    """Attach a ``ToolMeta`` to an implementation method.

    Every parameter the metadata names is checked against the method
    signature at decoration time, so a typo fails at import rather than at
    the first call.

    Args:
        name: Registered tool name.
        description: Tool description shown in the listing.
        read_only: The tool does not modify its environment.
        destructive: The tool overwrites or deletes existing state or files.
        idempotent: Repeating the call has no additional effect.
        open_world: The tool talks to an external process or the network.
        script: The tool runs agent-authored code from its ``code`` parameter.
        bypasses_instance_lock: The tool skips the per-instance lock.
        sandboxed_paths: Path parameters checked before dispatch.
        deprecated_aliases: Extra parameters accepted under an old name.
        hidden_parameters: Implementation parameters the tool does not expose.

    Returns:
        A decorator that stores the metadata on the method and returns it
        unchanged.
    """
    meta = ToolMeta(
        name=name,
        description=description,
        read_only=read_only,
        destructive=destructive,
        idempotent=idempotent,
        open_world=open_world,
        script=script,
        bypasses_instance_lock=bypasses_instance_lock,
        sandboxed_paths=tuple(sandboxed_paths),
        deprecated_aliases=tuple(deprecated_aliases),
        hidden_parameters=frozenset(hidden_parameters),
    )

    def decorator(method: ToolMethod) -> ToolMethod:
        _check_against_signature(method, meta)
        setattr(method, TOOL_META_ATTRIBUTE, meta)
        return method

    return decorator


def get_tool_meta(method: Any) -> Optional[ToolMeta]:
    """Return the metadata declared on ``method``, or None when it is not a tool.

    Args:
        method: Any attribute of a tools class.

    Returns:
        The ``ToolMeta`` set by ``@tool_meta``, or None.
    """
    meta = getattr(method, TOOL_META_ATTRIBUTE, None)
    return meta if isinstance(meta, ToolMeta) else None


def iter_tool_methods(cls: type) -> Iterator[Tuple[str, Callable[..., Any], ToolMeta]]:
    """Yield every tool method of ``cls`` in definition order along the MRO.

    Args:
        cls: The tools class, typically assembled from per-domain mixins.

    Yields:
        ``(method_name, function, meta)`` for each decorated method, the way
        it resolves on ``cls``. Names are yielded once; a subclass override
        wins over the mixin it overrides.
    """
    seen: set[str] = set()
    for klass in cls.__mro__:
        for method_name in vars(klass):
            if method_name in seen:
                continue
            seen.add(method_name)
            meta = get_tool_meta(getattr(cls, method_name))
            if meta is not None:
                yield method_name, getattr(cls, method_name), meta


def _check_against_signature(method: Callable[..., Any], meta: ToolMeta) -> None:
    """Raise when the metadata names a parameter the method does not have.

    Args:
        method: The decorated implementation method.
        meta: The metadata about to be attached.

    Raises:
        TypeError: When a sandboxed path, alias target or hidden parameter is
            not a parameter of the method, when an alias collides with one, or
            when a script tool has no ``code`` parameter.
    """
    parameters = [name for name in inspect.signature(method).parameters if name != "self"]
    for guard in meta.sandboxed_paths:
        if guard.parameter not in parameters:
            raise TypeError(f"{meta.name}: sandboxed path {guard.parameter!r} is not a parameter")
    for alias in meta.deprecated_aliases:
        if alias.target not in parameters:
            raise TypeError(f"{meta.name}: alias target {alias.target!r} is not a parameter")
        if alias.alias in parameters:
            raise TypeError(f"{meta.name}: alias {alias.alias!r} collides with a parameter")
    for hidden in meta.hidden_parameters:
        if hidden not in parameters:
            raise TypeError(f"{meta.name}: hidden parameter {hidden!r} is not a parameter")
    if meta.script and "code" not in parameters:
        raise TypeError(f"{meta.name}: a script tool needs a 'code' parameter")
