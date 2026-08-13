"""
Instance discovery, routing, and session management for Simul MCP Server.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..schemas.common import ErrorResponse

if TYPE_CHECKING:
    from ..server import SimulMCPServer


def register_instance_tools(server: "SimulMCPServer") -> None:
    """Register Isaac Sim instance discovery, routing, and session tools."""

    @server.mcp.tool(
        name="list_isaac_instances",
        description=(
            "Discover all running Isaac Sim instances and their session status. "
            "Scans the configured port range and returns each instance's status, "
            "loaded stage URL, simulation state, and active agent sessions with "
            "their purposes. Use this to find a free or compatible instance "
            "before starting work; unreachable instances report "
            "instance_status 'unreachable'."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=True
        ),
        output_schema=None,
    )
    async def list_isaac_instances(
        scan: bool = True,
        my_purpose: Optional[str] = None,
    ) -> Dict[str, Any]:
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
        return {
            "success": True,
            "instances": instances,
            "active_instance": server._get_effective_instance_name(),
            "total_discovered": len(reachable),
        }

    @server.mcp.tool(
        name="set_active_isaac_instance",
        description=(
            "Switch which Isaac Sim instance all isaac_* tools target. "
            "Use list_isaac_instances first to see available instances and "
            "their session status. Optionally register your purpose so other "
            "agents can see what you're doing on this instance."
        ),
        annotations=server._tool_annotations(
            read_only=False, idempotent=True, open_world=False
        ),
        output_schema=None,
    )
    async def set_active_isaac_instance(
        instance_name: str,
        purpose: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
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
            return ErrorResponse(
                error=f"Unknown instance '{instance_name}'. Available: {available}",
                error_type="NotFoundError",
                details={"available": available},
            ).model_dump()

        client = server._isaac_clients[instance_name]
        port = client._port

        compat_info: Dict[str, Any] = {}
        if purpose:
            compat_info = server.session_manager.score_compatibility(purpose, port)

        server._set_request_active_instance(instance_name)
        reachable = await client.ping()

        session_result: Dict[str, Any] = {}
        binding_result: Dict[str, Any] = {}
        if purpose:
            _agent_id = server._resolve_agent_id(agent_id)
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
        return result

    @server.mcp.tool(
        name="claim_isaac_instance",
        description=(
            "Register your purpose on the current active Isaac Sim instance. "
            "Other agents will see your purpose and can decide whether to join "
            "or pick a different instance. Call this at the start of your work "
            "to coordinate with other agents. The claim is advisory — it "
            "informs but does not block other agents."
        ),
        annotations=server._tool_annotations(
            read_only=False, idempotent=True, open_world=False
        ),
        output_schema=None,
    )
    async def claim_isaac_instance(
        purpose: str,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
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
            return ErrorResponse(
                error="No active instance", error_type="StateError"
            ).model_dump()

        port = client._port
        _agent_id = server._resolve_agent_id(agent_id)
        inst_session = server.session_manager.get_instance_session(port)

        compat = server.session_manager.score_compatibility(purpose, port)
        reg = inst_session.register(_agent_id, purpose)
        binding = server._bind_request_session(
            instance_name=instance_name,
            port=port,
            agent_id=_agent_id,
            purpose=purpose,
        )

        return {
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

    @server.mcp.tool(
        name="release_isaac_instance",
        description=(
            "Release your session on an Isaac Sim instance. Call this when "
            "you're done with your work so other agents know the instance "
            "is available. Sessions also auto-expire after 120 seconds of "
            "inactivity."
        ),
        annotations=server._tool_annotations(
            read_only=False, idempotent=True, open_world=False
        ),
        output_schema=None,
    )
    async def release_isaac_instance(
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
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
            return {"success": False, "error": "No active session to release"}

        inst_session = server.session_manager.get_instance_session(_port)
        result = inst_session.release(_agent_id)
        released = server._release_request_binding(instance_name, _agent_id)
        return {
            "success": True,
            **result,
            "binding_released": released.binding_id if released is not None else None,
        }
