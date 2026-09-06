"""
Isaac Sim tool registration for Simul MCP Server.

Every ``IsaacTools`` method decorated with ``@tool_meta`` becomes one MCP
tool: the registered name, description and safety hints come from the
decorator, the parameters and their defaults from the method signature, and
the parameter descriptions from the method docstring. ``ping_isaac`` is the
one Isaac tool without an ``IsaacTools`` method behind it, so it is written
out here.
"""

from __future__ import annotations

import hashlib
import inspect
import typing
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Dict, Optional

from fastmcp.tools.tool import ToolResult

from ..tools._meta import ToolMeta, iter_tool_methods
from ..tools.isaac_tools import IsaacTools
from ._helpers import resolve_deprecated_alias, with_param_descriptions

if TYPE_CHECKING:
    from ..server import SimulMCPServer

ToolWrapper = Callable[..., Coroutine[Any, Any, ToolResult]]


def register_isaac_tools(server: "SimulMCPServer") -> None:
    """
    Register the Isaac Sim tools.

    Args:
        server: The server whose FastMCP instance receives the tools.
    """
    _register_ping(server)
    for method_name, implementation, meta in iter_tool_methods(IsaacTools):
        register = server._script_tool if meta.script else server.mcp.tool
        register(
            name=meta.name,
            description=meta.description,
            annotations=server._tool_annotations(
                read_only=meta.read_only,
                idempotent=meta.idempotent,
                open_world=meta.open_world,
                destructive=meta.destructive,
            ),
        )(
            with_param_descriptions(
                implementation,
                **{alias.alias: alias.description for alias in meta.deprecated_aliases},
            )(_build_tool_wrapper(server, method_name, implementation, meta))
        )


def _register_ping(server: "SimulMCPServer") -> None:
    """Register ``ping_isaac``, which reads the socket client rather than IsaacTools."""

    @server.mcp.tool(
        name="ping_isaac",
        description=(
            "Pre-flight check: verify that a running Isaac Sim instance is "
            "reachable on the configured TCP socket. Call this before "
            "execute_isaac_script to confirm connectivity and get the target address."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=True
        ),
    )
    async def ping_isaac() -> ToolResult:
        """
        Ping Isaac Sim to verify connectivity.

        Returns:
            Dict with reachable status, address, and timeout. ``success``
            tracks ``reachable``: an unreachable instance is a failed ping.
        """
        rate_error = server._check_rate_limit("ping_isaac")
        if rate_error:
            return server._as_text_result(rate_error)
        client = server._get_request_isaac_client()
        reachable = await client.ping()
        payload: Dict[str, Any] = {
            "success": reachable,
            "reachable": reachable,
            "address": client.address,
            "bridge_address": client.bridge_address,
            "bridge_circuit_open": client.bridge_circuit_open,
            "vscode_address": client.vscode_address,
            "timeout_seconds": client.timeout_seconds,
        }
        if not reachable:
            payload["error"] = f"Isaac Sim is not reachable at {client.address}"
        return server._as_text_result(payload)


def _build_tool_wrapper(
    server: "SimulMCPServer",
    method_name: str,
    implementation: Callable[..., Any],
    meta: ToolMeta,
) -> ToolWrapper:
    """Build the FastMCP-facing coroutine for one ``IsaacTools`` method.

    The wrapper advertises the implementation's signature (minus ``self`` and
    the hidden parameters, plus one optional parameter per deprecated alias)
    so FastMCP derives the input schema from it, and forwards every call
    through the server's Isaac envelope.

    Args:
        server: The server whose envelope, sandbox and tools object are used.
        method_name: Attribute name of the method on ``server._isaac_tools``,
            resolved at call time so a replaced tools object is honoured.
        implementation: The decorated ``IsaacTools`` method.
        meta: The tool's metadata.

    Returns:
        An async callable named after the tool.
    """
    signature = inspect.signature(implementation)
    hints = typing.get_type_hints(implementation, include_extras=True)
    parameters = [
        parameter.replace(annotation=hints.get(name, parameter.annotation))
        for name, parameter in signature.parameters.items()
        if name != "self" and name not in meta.hidden_parameters
    ]
    defaults = {parameter.name: parameter.default for parameter in parameters}
    for alias in meta.deprecated_aliases:
        parameters.append(
            inspect.Parameter(
                alias.alias,
                inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=Optional[hints.get(alias.target, Any)],  # type: ignore[valid-type]
            )
        )

    advertised = inspect.Signature(parameters, return_annotation=ToolResult)

    async def tool(*args: Any, **kwargs: Any) -> ToolResult:
        bound = advertised.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = bound.arguments
        for alias in meta.deprecated_aliases:
            arguments[alias.target] = resolve_deprecated_alias(
                arguments[alias.target], arguments.pop(alias.alias), defaults[alias.target]
            )
        for guard in meta.sandboxed_paths:
            denial = server._sandbox_denial(arguments.get(guard.parameter), write=guard.write)
            if denial is not None:
                return server._as_text_result(denial)
        call = getattr(server._isaac_tools, method_name)(**arguments)
        if meta.bypasses_instance_lock:
            rate_error = server._check_rate_limit(meta.name)
            if rate_error is not None:
                call.close()
                return server._as_text_result(rate_error)
            return server._as_text_result(await call)
        if meta.script:
            code = str(arguments["code"])
            return await server._exec_isaac(
                meta.name,
                call,
                params={"code_bytes": len(code)},
                script_sha256=hashlib.sha256(code.encode("utf-8")).hexdigest(),
            )
        return await server._exec_isaac(meta.name, call)

    tool.__name__ = meta.name
    tool.__qualname__ = meta.name
    tool.__doc__ = implementation.__doc__
    tool.__signature__ = advertised  # type: ignore[attr-defined]
    tool.__annotations__ = {parameter.name: parameter.annotation for parameter in parameters}
    tool.__annotations__["return"] = ToolResult
    return tool
