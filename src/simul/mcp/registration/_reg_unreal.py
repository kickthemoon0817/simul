"""
Unreal Engine tool registration for the Simul MCP server.

The tools are declared in ``_unreal_tools.UNREAL_TOOLS``; this module builds
one FastMCP tool per declaration. The wrapper advertises the declared
parameters, from which FastMCP derives the input schema, and runs every call
through the shared envelope (``SimulMCPServer._exec_backend``) with a request
model built from the declaration. The request model, the declaration's
cross-parameter ``check`` included, is validated inside the envelope, so a call
it refuses is rate limited and recorded like any other failed call. Arguments
that do not fit the input schema never reach the wrapper: FastMCP refuses them
first.
"""

from __future__ import annotations

import inspect
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Awaitable,
    Callable,
    Dict,
    Mapping,
    Optional,
    Type,
)

from fastmcp.tools import ToolResult
from pydantic import BaseModel, Field, create_model, model_validator

from ._helpers import describe_params
from ._unreal_tools import REQUIRED, ToolParam, UnrealTool, surface_tools

if TYPE_CHECKING:
    from ..server import SimulMCPServer

ToolWrapper = Callable[..., Awaitable[ToolResult]]


class InvalidToolInput(Exception):
    """A declaration's ``check`` refused the arguments.

    Deliberately not a ValueError: pydantic folds a ValueError raised in a
    validator into its own multi-line ValidationError, while any other
    exception passes through unchanged, so the envelope's ValidationError
    payload carries the check's message as written.
    """


def register_unreal_tools(server: "SimulMCPServer", thin: bool = False) -> None:
    """Register the Unreal Engine tools.

    Args:
        server: The MCP server instance.
        thin: When True, register only ``THIN_UNREAL_TOOLS`` from
              ``simul.tool_surfaces``. Selected by
              ``unreal.tool_surface`` / ``simul server --unreal-tools``;
              the full set is also reachable via ``simul unreal --help``.
    """
    # Imported here rather than at module level: the adapter needs aiohttp,
    # and this module must import even where the Unreal backend is absent.
    from ...adapters.unreal_runtime import UnrealRuntimeSession

    for spec in surface_tools(thin):
        register = server._script_tool if spec.script else server.mcp.tool
        documented = describe_params(getattr(UnrealRuntimeSession, spec.method)) if spec.method else {}
        register(
            name=spec.name,
            description=spec.description,
            annotations=server._tool_annotations(
                read_only=spec.read_only,
                idempotent=spec.idempotent,
                # Every Unreal tool talks to an editor process.
                open_world=True,
                destructive=spec.destructive,
            ),
            output_schema=None,
            task=server._task_optional(),
        )(build_tool_wrapper(server, spec, documented))


def build_tool_wrapper(
    server: "SimulMCPServer", spec: UnrealTool, documented: Mapping[str, str]
) -> ToolWrapper:
    """Build the FastMCP-facing coroutine for one declaration.

    The wrapper's signature is the declared parameters, each annotated with
    its description, so FastMCP derives the input schema from it. A call binds
    its arguments to that signature and hands them to the envelope, which
    validates them against the request model before the session opens.

    Args:
        server: The server whose envelope runs the calls; its Unreal adapter is
            read at call time, so a replaced adapter is honoured.
        spec: The declaration.
        documented: Parameter descriptions from the session method's
            docstring, for parameters without a description of their own.

    Returns:
        An async callable named after the tool.
    """
    request_model = spec.request or request_model_for(spec)
    annotations = {param.name: _described(param, documented.get(param.name)) for param in spec.params}
    signature = inspect.Signature(
        [
            inspect.Parameter(
                param.name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=param.default,
                annotation=annotations[param.name],
            )
            for param in spec.params
        ],
        return_annotation=ToolResult,
    )

    def call(session: Any, request: Any) -> Any:
        if spec.sessionless is not None:
            return spec.sessionless(server.settings, request)
        return getattr(session, str(spec.method))(**session_arguments(spec, request))

    async def tool(*args: Any, **kwargs: Any) -> ToolResult:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        return await server._exec_backend(
            spec.name,
            server.unreal_adapter,
            "Unreal",
            spec.response,
            call,
            spec.usage(arguments) if spec.usage is not None else None,
            request_model=request_model,
            request_kwargs=arguments,
            input_echo=spec.echo,
        )

    tool.__name__ = spec.name
    tool.__qualname__ = spec.name
    tool.__doc__ = spec.description
    tool.__signature__ = signature  # type: ignore[attr-defined]
    tool.__annotations__ = {**annotations, "return": ToolResult}
    return tool


def session_arguments(spec: UnrealTool, request: Any) -> Dict[str, Any]:
    """The keyword arguments ``spec.method`` receives for a validated request.

    Args:
        spec: The declaration.
        request: The validated request.

    Returns:
        What ``spec.arguments`` builds, or each parameter's value, converted,
        under its target name.
    """
    if spec.arguments is not None:
        return spec.arguments(request)
    arguments: Dict[str, Any] = {}
    for param in spec.params:
        value = getattr(request, param.name)
        arguments[param.target or param.name] = value if param.convert is None else param.convert(value)
    return arguments


def request_model_for(spec: UnrealTool) -> Type[BaseModel]:
    """Build the request model of a declaration without an explicit one.

    Args:
        spec: The declaration.

    Returns:
        A model with one field per parameter, with the declared type and
        default, enforcing ``spec.check`` after the fields validate.
    """
    fields: Dict[str, Any] = {
        param.name: (param.annotation, ... if param.default is REQUIRED else param.default)
        for param in spec.params
    }
    validators: Dict[str, Any] = {}
    if spec.check is not None:
        validators["check"] = model_validator(mode="after")(_enforce(spec.check))
    name = "".join(part.title() for part in spec.name.split("_")) + "Request"
    return create_model(name, __validators__=validators, **fields)


def _enforce(check: Callable[[Any], Optional[str]]) -> Callable[[Any], Any]:
    """Turn a check's message into the exception that fails the validation."""

    def validate(request: Any) -> Any:
        message = check(request)
        if message is not None:
            raise InvalidToolInput(message)
        return request

    return validate


def _described(param: ToolParam, documented: Optional[str]) -> Any:
    """The parameter's annotation, carrying its schema description when it has one."""
    text = param.description or documented
    if not text:
        return param.annotation
    return Annotated[param.annotation, Field(description=text)]
