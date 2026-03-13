"""
Instance discovery and routing tool registration for Simul MCP Server.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..schemas import *

if TYPE_CHECKING:
    from ..server import SimulMCPServer


def register_instance_tools(server: "SimulMCPServer") -> None:
    """Register Isaac Sim instance discovery and routing tools."""

    @server.mcp.tool(
        name="list_isaac_instances",
        description=(
            "Discover all running Isaac Sim instances. Scans the configured "
            "port range and returns each instance's status, loaded stage URL, "
            "prim count, and simulation state. Call this first when multiple "
            "Isaac Sim applications may be running to identify the correct target."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=True
        ),
    )
    async def list_isaac_instances(scan: bool = True) -> Dict[str, Any]:
        """
        List all known and discovered Isaac Sim instances.

        Args:
            scan: If True, scan the port range for new instances.

        Returns:
            List of instance info dicts with status and stage metadata.
        """
        if scan:
            discovered = await server._scan_isaac_instances()
            server._isaac_clients.update(discovered)

        instances = []
        for name, client in server._isaac_clients.items():
            brief = await server._get_instance_brief(name, client)
            instances.append(brief)

        reachable = [i for i in instances if i["reachable"]]
        return {
            "success": True,
            "instances": instances,
            "active_instance": server._active_instance,
            "total_discovered": len(reachable),
        }

    @server.mcp.tool(
        name="set_active_isaac_instance",
        description=(
            "Switch which Isaac Sim instance all isaac_* tools target. "
            "Use list_isaac_instances first to see available instances, "
            "then call this with the desired instance name. All subsequent "
            "isaac_* tool calls will route to the selected instance."
        ),
        annotations=server._tool_annotations(
            read_only=False, idempotent=True, open_world=False
        ),
    )
    async def set_active_isaac_instance(instance_name: str) -> Dict[str, Any]:
        """
        Switch the active Isaac Sim instance.

        Args:
            instance_name: Name of the instance to activate.

        Returns:
            Confirmation with the new active instance address.
        """
        if instance_name not in server._isaac_clients:
            available = list(server._isaac_clients.keys())
            return ErrorResponse(
                error=f"Unknown instance '{instance_name}'. Available: {available}",
                error_type="NotFoundError",
                details={"available": available},
            ).dict()

        server._switch_active_instance(instance_name)
        client = server._isaac_clients[instance_name]
        reachable = await client.ping()
        return {
            "success": True,
            "active_instance": instance_name,
            "address": client.address,
            "message": (
                f"Switched to '{instance_name}' at {client.address}"
                + (" (reachable)" if reachable else " (WARNING: not reachable)")
            ),
        }

