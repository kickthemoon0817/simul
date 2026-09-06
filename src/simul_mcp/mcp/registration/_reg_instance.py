"""
Instance discovery, routing, and session management for Simul MCP Server.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Dict, Optional

from fastmcp.tools.tool import ToolResult

from ..schemas.common import ErrorResponse
from ._helpers import with_param_descriptions

if TYPE_CHECKING:
    from ..server import SimulMCPServer


_CLAIM_MODE_SENTENCE = {
    True: (
        "Claims are ENFORCED on this server (isaac_sim.enforce_claims=true): a live "
        "claim blocks mutating Isaac tools from every other agent with InstanceClaimed."
    ),
    False: (
        "Claims are ADVISORY on this server (isaac_sim.enforce_claims=false): a "
        "listed session tells you who is working there but does not block you."
    ),
}


def register_instance_tools(server: "SimulMCPServer") -> None:
    """Register Isaac Sim instance discovery, routing, and session tools.

    The claim tools describe the server's actual behaviour: with
    ``isaac_sim.enforce_claims`` on, a claim is a lock; off, it is a notice.
    """
    enforced: bool = server.settings.isaac_sim.enforce_claims

    @server.mcp.tool(
        name="list_isaac_instances",
        description=(
            "Discover all running Isaac Sim instances and their session status. "
            "Scans the configured port range and returns each instance's status, "
            "loaded stage URL, simulation state, and active agent sessions with "
            "their purposes. Use this to find a free or compatible instance "
            "before starting work; unreachable instances report "
            "instance_status 'unreachable'. "
            + _CLAIM_MODE_SENTENCE[enforced]
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=True
        ),
        output_schema=None,
    )
    @with_param_descriptions()
    async def list_isaac_instances(
        scan: bool = True,
        my_purpose: Optional[str] = None,
    ) -> ToolResult:
        """
        List all known and discovered Isaac Sim instances with session info.

        Args:
            scan: If True, scan the port range for new instances.
            my_purpose: Optional purpose description to score compatibility
                        with existing sessions on each instance.

        Returns:
            List of instance info dicts with session status and compatibility.
        """
        if scan:
            discovered = await server._scan_isaac_instances()
            server._isaac_clients.update(discovered)

        async def _instance_entry(name: str, client: Any) -> Dict[str, Any]:
            brief = await server._get_instance_brief(name, client)
            session_status = server.session_manager.get_instance_session(
                client._port
            ).get_status()
            brief["sessions"] = session_status["sessions"]
            brief["session_count"] = session_status["session_count"]

            if not brief.get("reachable"):
                # Session bookkeeping says nothing about whether we can talk to
                # the instance, so an empty session list must not read as
                # "available". Reporting free/clear/1.0 next to reachable=false
                # is what let an agent select an instance that was not running.
                brief["instance_status"] = "unreachable"
                brief["compatibility"] = "blocked"
                brief["compatibility_score"] = 0.0
                brief["compatibility_reason"] = (
                    "Instance is not reachable — is Isaac Sim running with the "
                    "bridge extension enabled?"
                )
                return brief

            brief["instance_status"] = session_status["status"]

            if my_purpose and session_status["sessions"]:
                compat = server.session_manager.score_compatibility(
                    my_purpose, client._port, status=session_status
                )
                brief["compatibility"] = compat["compatibility"]
                brief["compatibility_score"] = compat["score"]
                brief["compatibility_reason"] = compat["reason"]
            elif not session_status["sessions"]:
                brief["compatibility"] = "clear"
                brief["compatibility_score"] = 1.0
                brief["compatibility_reason"] = "No active sessions — instance is free"

            return brief

        instances = list(
            await asyncio.gather(
                *(
                    _instance_entry(name, client)
                    for name, client in server._isaac_clients.items()
                )
            )
        )

        reachable = [i for i in instances if i["reachable"]]
        return server._as_text_result(
            {
                "success": True,
                "instances": instances,
                "active_instance": server._get_effective_instance_name(),
                "total_discovered": len(reachable),
            }
        )

    @server.mcp.tool(
        name="set_active_isaac_instance",
        description=(
            "Switch which Isaac Sim instance all Isaac tools (names containing "
            "'isaac') target for the rest of this MCP session. "
            "Use list_isaac_instances first to see available instances and "
            "their session status. Optionally register your purpose so other "
            "agents can see what you're doing on this instance; with a purpose "
            "this is the same as claim_isaac_instance. "
            + _CLAIM_MODE_SENTENCE[enforced]
        ),
        annotations=server._tool_annotations(
            read_only=False, idempotent=True, open_world=False
        ),
        output_schema=None,
    )
    @with_param_descriptions()
    async def set_active_isaac_instance(
        instance_name: str,
        purpose: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> ToolResult:
        """
        Switch the active Isaac Sim instance and optionally register a session.

        Args:
            instance_name: Name of the instance to activate.
            purpose: Free-text description of what you're doing. Other agents
                     will see this and can decide whether to join or avoid.
            agent_id: Unique identifier for this agent session. Defaults to
                      a generated ID if not provided.

        Returns:
            Confirmation with the new active instance, session info,
            and compatibility status.
        """
        if instance_name not in server._isaac_clients:
            available = list(server._isaac_clients.keys())
            return server._as_text_result(
                ErrorResponse(
                    error=f"Unknown instance '{instance_name}'. Available: {available}",
                    error_type="NotFoundError",
                    details={"available": available},
                ).model_dump()
            )

        client = server._isaac_clients[instance_name]
        port = client._port
        _agent_id = server._resolve_agent_id(agent_id)

        compat_info: Dict[str, Any] = {}
        if purpose:
            holder = server._foreign_claim(instance_name, _agent_id)
            if holder is not None:
                return server._claimed_error(instance_name, holder)
            compat_info = server.session_manager.score_compatibility(purpose, port)

        server._set_request_active_instance(instance_name)
        reachable = await client.ping()

        session_result: Dict[str, Any] = {}
        binding_result: Dict[str, Any] = {}
        if purpose:
            inst_session = server.session_manager.get_instance_session(port)
            session_result = inst_session.register(_agent_id, purpose)
            binding = server._bind_request_session(
                instance_name=instance_name,
                port=port,
                agent_id=_agent_id,
                purpose=purpose,
            )
            binding_result = {
                "binding_id": binding.binding_id,
                "agent_id": binding.agent_id,
                "session_id": binding.session_id,
            }

        result: Dict[str, Any] = {
            "success": True,
            "active_instance": instance_name,
            "address": client.address,
            "bridge_address": client.bridge_address,
            "vscode_address": client.vscode_address,
            "reachable": reachable,
        }
        if compat_info:
            result["compatibility"] = compat_info.get("compatibility")
            result["compatibility_score"] = compat_info.get("score")
            result["compatibility_reason"] = compat_info.get("reason")
            result["existing_sessions"] = compat_info.get("sessions", [])
        if session_result:
            result["session"] = session_result
        if binding_result:
            result["binding"] = binding_result
        return server._as_text_result(result)

    @server.mcp.tool(
        name="claim_isaac_instance",
        description=(
            (
                "Claim the current active Isaac Sim instance for your purpose. "
                "Claims are ENFORCED on this server (isaac_sim.enforce_claims=true): "
                "while your claim is live, mutating Isaac tools from any other agent "
                "fail with InstanceClaimed, and this call fails the same way when "
                "another agent already holds a live claim. Read-only tools are never "
                "blocked. A claim expires after 120 s without tool activity; call "
                "release_isaac_instance when you are done."
            )
            if enforced
            else (
                "Register your purpose on the current active Isaac Sim instance. "
                "Claims are ADVISORY on this server (isaac_sim.enforce_claims=false): "
                "other agents see your purpose in list_isaac_instances and get a "
                "compatibility score when they claim, but nothing stops them from "
                "mutating the scene. Set isaac_sim.enforce_claims=true to make a "
                "claim block other agents' mutating tools. A claim expires after "
                "120 s without tool activity."
            )
        ),
        annotations=server._tool_annotations(
            read_only=False, idempotent=True, open_world=False
        ),
        output_schema=None,
    )
    @with_param_descriptions()
    async def claim_isaac_instance(
        purpose: str,
        agent_id: Optional[str] = None,
    ) -> ToolResult:
        """
        Register a purpose on the current active instance.

        Args:
            purpose: Free-text description of what you're doing.
            agent_id: Unique identifier for this agent session.

        Returns:
            Registration result with compatibility info.
        """
        instance_name = server._get_effective_instance_name()
        client = server._isaac_clients.get(instance_name)
        if not client:
            return server._as_text_result(
                ErrorResponse(
                    error="No active instance", error_type="StateError"
                ).model_dump()
            )

        port = client._port
        _agent_id = server._resolve_agent_id(agent_id)
        holder = server._foreign_claim(instance_name, _agent_id)
        if holder is not None:
            return server._claimed_error(instance_name, holder)
        inst_session = server.session_manager.get_instance_session(port)

        compat = server.session_manager.score_compatibility(purpose, port)
        reg = inst_session.register(_agent_id, purpose)
        binding = server._bind_request_session(
            instance_name=instance_name,
            port=port,
            agent_id=_agent_id,
            purpose=purpose,
        )

        return server._as_text_result(
            {
                "success": True,
                "instance": instance_name,
                "port": port,
                "session": reg,
                "binding": {
                    "binding_id": binding.binding_id,
                    "agent_id": binding.agent_id,
                    "session_id": binding.session_id,
                },
                "compatibility": compat["compatibility"],
                "compatibility_score": compat["score"],
                "compatibility_reason": compat["reason"],
                "existing_sessions": compat["sessions"],
            }
        )

    @server.mcp.tool(
        name="release_isaac_instance",
        description=(
            "Release your own claim on the active Isaac Sim instance; an "
            "agent_id that is not yours is refused. Call this when you're done "
            "so other agents can use the instance. Claims also expire after "
            "120 seconds of inactivity. "
            + (
                "Claims are ENFORCED on this server (isaac_sim.enforce_claims=true): "
                "releasing lifts the InstanceClaimed refusal for other agents."
                if enforced
                else "Claims are ADVISORY on this server (isaac_sim.enforce_claims=false): "
                "releasing only changes what other agents see."
            )
        ),
        annotations=server._tool_annotations(
            read_only=False, idempotent=True, open_world=False
        ),
        output_schema=None,
    )
    @with_param_descriptions()
    async def release_isaac_instance(
        agent_id: Optional[str] = None,
    ) -> ToolResult:
        """
        Release the current agent's session.

        Args:
            agent_id: Agent session ID. Uses current session if not specified.

        Returns:
            Release confirmation.
        """
        instance_name = server._get_effective_instance_name()
        binding = server._get_active_binding()
        _agent_id = agent_id or (binding.agent_id if binding is not None else None)
        _port = binding.port if binding is not None else None
        if not _agent_id or not _port:
            return server._as_text_result(
                {"success": False, "error": "No active session to release"}
            )
        if binding is not None and _agent_id != binding.agent_id:
            return server._as_text_result(
                ErrorResponse(
                    error=(
                        f"agent_id {_agent_id!r} is not this session's claim "
                        f"({binding.agent_id!r}); only your own claim can be released."
                    ),
                    error_type="PermissionError",
                ).model_dump()
            )

        inst_session = server.session_manager.get_instance_session(_port)
        result = inst_session.release(_agent_id)
        released = server._release_request_binding(instance_name, _agent_id)
        return server._as_text_result(
            {
                "success": True,
                **result,
                "binding_released": released.binding_id if released is not None else None,
            }
        )
