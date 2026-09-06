"""Shared helpers for tool-registration modules.

Keep this module narrow — only utility functions used by 2+ register
modules belong here. One-off helpers stay inline at their use site.
"""

from __future__ import annotations

import inspect
import re
import typing
from typing import Annotated, Any, Callable, Optional, TypeVar

from pydantic import Field

WrappedTool = TypeVar("WrappedTool", bound=Callable[..., Any])

# Google-style docstring entry: ``name: text`` or ``name (type): text``.
_ARG_ENTRY = re.compile(
    r"^(?P<name>\*{0,2}[A-Za-z_]\w*)(?:\s*\([^)]*\))?:\s*(?P<text>.*)$"
)


def apply_success_from_error(payload: dict[str, Any]) -> dict[str, Any]:
    """Mark ``payload['success']`` from its ``error`` and ``*_error`` keys.

    Every tool envelope passes its adapter or script payload through here.
    An adapter call that returns a payload shaped like
    ``{"connected": false, "error": "..."}`` (``ping()`` on an unreachable
    port, ``health_check()`` on connection refused) must surface as
    ``success=False``, and so must a script that reports a partial failure
    under a suffixed key such as ``syntheticdata_error`` or ``timeline_error``
    while still returning the parts that worked.

    The rule:

      - If the payload already has an explicit ``success`` field (some
        adapters set it themselves — e.g. ``unreal_runtime``
        input-validation guards return ``{"success": False, "error":
        "..."}``), leave it alone. The helper does NOT resolve a
        ``success=True + error="..."`` contradiction; adapter methods are
        responsible for never emitting that shape.
      - Otherwise, set ``success`` to ``True`` iff ``error`` and every key
        ending in ``_error`` are absent or ``None``.

    Returns the same dict for chaining; mutation is in place because
    every caller already has the dict in hand.
    """
    if "success" in payload:
        return payload
    payload["success"] = not any(
        value is not None
        for key, value in payload.items()
        if key == "error" or key.endswith("_error")
    )
    return payload


def describe_params(func: Callable[..., Any]) -> dict[str, str]:
    """Parse the Google-style ``Args:`` section of ``func``'s docstring.

    Args:
        func: Any callable whose docstring documents its parameters under an
            ``Args:`` heading, one ``name: text`` entry per parameter with
            continuation lines indented deeper than the name.

    Returns:
        Parameter name to its description text, whitespace-normalised. Empty
        when the docstring has no ``Args:`` section.
    """
    docstring = inspect.getdoc(func) or ""
    descriptions: dict[str, str] = {}
    entry_indent: Optional[int] = None
    current_name: Optional[str] = None
    in_args = False
    for raw_line in docstring.splitlines():
        line = raw_line.rstrip()
        if not in_args:
            in_args = line.strip() == "Args:"
            continue
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 0:
            # The next section header, or prose at the section's own indent.
            break
        if entry_indent is None:
            entry_indent = indent
        entry = _ARG_ENTRY.match(line.strip()) if indent == entry_indent else None
        if entry is not None:
            current_name = entry.group("name").lstrip("*")
            descriptions[current_name] = entry.group("text").strip()
        elif current_name is not None and indent > entry_indent:
            descriptions[current_name] = (
                descriptions[current_name] + " " + line.strip()
            ).strip()
    return descriptions


def with_param_descriptions(
    *sources: Callable[..., Any], **overrides: str
) -> Callable[[WrappedTool], WrappedTool]:
    """Attach parameter descriptions to a tool wrapper's schema.

    FastMCP builds each tool's input schema from the wrapper's type hints, so
    ``Args:`` text that lives only in a docstring never reaches the agent.
    This decorator rewrites each documented parameter's annotation to
    ``Annotated[<hint>, Field(description=...)]`` so pydantic emits the text
    into ``inputSchema.properties[<name>].description``.

    Descriptions are looked up in this order, later entries winning:
    ``sources`` from last to first, the wrapper's own docstring, then
    ``overrides``. Parameters with no description anywhere are left as they
    are; the registration test is what enforces full coverage.

    Args:
        *sources: Callables whose ``Args:`` sections document the wrapper's
            parameters, typically the implementation method the wrapper
            delegates to.
        **overrides: Parameter name to description text, for parameters the
            wrapper adds itself (deprecated aliases, paging offsets).

    Returns:
        A decorator that mutates the wrapper's annotations in place and
        returns it unchanged otherwise.
    """

    def decorator(func: WrappedTool) -> WrappedTool:
        descriptions: dict[str, str] = {}
        for source in reversed(sources):
            descriptions.update(describe_params(source))
        descriptions.update(describe_params(func))
        descriptions.update(overrides)
        hints = typing.get_type_hints(func, include_extras=True)
        for name in inspect.signature(func).parameters:
            text = descriptions.get(name)
            if not text:
                continue
            func.__annotations__[name] = Annotated[
                hints.get(name, Any), Field(description=text)
            ]
        return func

    return decorator


def resolve_deprecated_alias(
    value: Any, alias_value: Optional[Any], default: Any
) -> Any:
    """Pick between a renamed parameter and its deprecated alias.

    The new name wins whenever the caller moved it off its default; the alias
    is honoured only when it was passed and the new name was not.

    Args:
        value: The value received under the current parameter name.
        alias_value: The value received under the deprecated name, or None.
        default: The current parameter's declared default.

    Returns:
        The value the tool should act on.
    """
    if alias_value is not None and value == default:
        return alias_value
    return value
