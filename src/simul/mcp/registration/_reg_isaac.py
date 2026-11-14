"""
Isaac Sim tool registration for the Simul MCP server.

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

from fastmcp.tools import ToolResult

from ..tools._meta import ToolMeta, iter_tool_methods
from ..tools.isaac_tools import IsaacTools
from ._helpers import (
    resolve_deprecated_alias,
    utf8_byte_length,
    with_param_descriptions,
)

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
            "reachable on the configured TCP socket. Call this before other Isaac "
            "tools to confirm connectivity and get the target address. It does not "
            "wait for a call already running on the instance; busy says one is. While the "
            "instance is held after a timed-out call it answers at once, without a probe, "
            "with held_after_timeout."
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
        # A probe has to answer while another call holds the instance.
        return await server._exec_isaac("ping_isaac", _ping(server), bypass_instance_lock=True)


async def _ping(server: "SimulMCPServer") -> Dict[str, Any]:
    """Probe the instance the request targets and describe its transports.

    ``busy`` says another call holds the instance: a probe that then fails may
    only mean the engine is inside a synchronous script.

    While the instance is held after a timed-out call (by this server or
    another simul server on the machine), nothing is sent: a probe would only
    wait behind that call, and on the stock socket time out and read as
    unreachable. The answer then comes from the servers' own state:
    ``held_after_timeout`` describes the hold, ``probed`` is false, and the
    instance counts as reachable, since the hold ends as soon as the engine
    stops answering.
    """
    client = server._get_request_isaac_client()
    instance_name = server._get_effective_instance_name()
    busy = server._get_instance_lock(instance_name).locked()
    held = server.isaac_hold_report(instance_name, client)
    payload: Dict[str, Any] = {
        "address": client.address,
        "bridge_address": client.bridge_address,
        "bridge_circuit_open": client.bridge_circuit_open,
        "vscode_address": client.vscode_address,
        "timeout_seconds": client.timeout_seconds,
    }
    if held is not None:
        return {
            "success": True,
            "reachable": True,
            "busy": True,
            "probed": False,
            **payload,
            "held_after_timeout": held,
            "note": _held_note(held),
        }
    reachable = await client.ping()
    payload = {"success": reachable, "reachable": reachable, "busy": busy, **payload}
    if not reachable:
        payload["error"] = f"Isaac Sim is not reachable at {client.address}" + (
            "; another call holds the instance, and a synchronous script blocks the "
            "socket until it returns" if busy else ""
        )
    return payload


def _held_note(held: Dict[str, Any]) -> str:
    """Say why a status tool answered from the server's state during a hold."""
    holder = held.get("held_by")
    whose = (
        f"another simul server on this machine (pid {holder.get('pid')})"
        if isinstance(holder, dict)
        else "this server"
    )
    return (
        f"Isaac Sim is still running {held.get('timed_out_tool')}, whose reply timed out "
        f"{float(held.get('busy_for_seconds') or 0):.0f}s ago, and {whose} sends it nothing else "
        "until it has finished (isaac_sim.hold_after_timeout), so Isaac Sim was not asked: "
        "calls to it get InstanceBusy meanwhile. interrupt_isaac_script stops a bridge script."
    )


def held_status(server: "SimulMCPServer") -> Optional[Dict[str, Any]]:
    """A status tool's answer while the request's instance is held after a timeout, or None.

    Built from the servers' own state, without sending the engine anything:
    ``busy``, ``held_after_timeout``, the ``instance`` and the ``client``'s
    transport state (``reports_hold`` tools).
    """
    client = server._get_request_isaac_client()
    instance_name = server._get_effective_instance_name()
    held = server.isaac_hold_report(instance_name, client)
    if held is None:
        return None
    return {
        "success": True,
        "busy": True,
        "instance": instance_name,
        "held_after_timeout": held,
        "client": server._isaac_tools._client_state(),
        "note": _held_note(held),
    }


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
                # Refused before Isaac is reached, but still rate limited and
                # recorded like every other call.
                return await server._exec_payload(meta.name, denial)
        if meta.reports_hold:
            status = held_status(server)
            if status is not None:
                return await server._exec_payload(meta.name, status)
        call = getattr(server._isaac_tools, method_name)(**arguments)
        if meta.bypasses_instance_lock:
            return await server._exec_isaac(meta.name, call, bypass_instance_lock=True)
        if meta.script:
            code = str(arguments["code"])
            # surrogatepass, as utf8_byte_length counts: a lone surrogate from an
            # in-process client is hashed instead of raising before the record.
            return await server._exec_isaac(
                meta.name,
                call,
                params={"code_bytes": utf8_byte_length(code)},
                script_sha256=hashlib.sha256(code.encode("utf-8", "surrogatepass")).hexdigest(),
            )
        return await server._exec_isaac(meta.name, call)

    tool.__name__ = meta.name
    tool.__qualname__ = meta.name
    tool.__doc__ = implementation.__doc__
    tool.__signature__ = advertised  # type: ignore[attr-defined]
    tool.__annotations__ = {parameter.name: parameter.annotation for parameter in parameters}
    tool.__annotations__["return"] = ToolResult
    return tool
