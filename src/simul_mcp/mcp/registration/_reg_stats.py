"""
Tool usage statistics registration for Simul MCP Server.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..server import SimulMCPServer


def register_stats_tools(server: "SimulMCPServer") -> None:
    """Register tool usage statistics tools."""

    @server.mcp.tool(
        name="get_tool_usage_stats",
        description=(
            "Get tool usage statistics: per-tool call counts, success/failure "
            "rates, average duration, and recent call log. Use tool_name to "
            "filter to a single tool. Use include_recent=true with limit to "
            "get the last N call records with parameters and timing."
        ),
        annotations=server._tool_annotations(
            read_only=True, idempotent=True, open_world=False
        ),
        output_schema=None,
    )
    async def get_tool_usage_stats(
        tool_name: Optional[str] = None,
        include_recent: bool = False,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """
        Get tool usage statistics.

        Args:
            tool_name: Filter to a specific tool.
            include_recent: Include recent call log.
            limit: Max recent records to return.

        Returns:
            Dict with stats and optional recent call log.
        """
        result = server.usage_tracker.get_stats(tool_name=tool_name)
        if include_recent:
            result["recent"] = server.usage_tracker.get_recent(
                limit=limit, tool_name=tool_name,
            )
        return result

    @server.mcp.tool(
        name="reset_tool_usage_stats",
        description="Clear all tool usage statistics and the recent call log.",
        annotations=server._tool_annotations(
            read_only=False, idempotent=True, open_world=False
        ),
        output_schema=None,
    )
    async def reset_tool_usage_stats() -> Dict[str, Any]:
        """Reset all usage tracking data."""
        server.usage_tracker.reset()
        return {"success": True, "message": "Tool usage stats cleared"}
