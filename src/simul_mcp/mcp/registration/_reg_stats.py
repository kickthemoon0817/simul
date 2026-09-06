"""
Tool usage statistics registration for Simul MCP Server.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from fastmcp.tools.tool import ToolResult

from ._helpers import with_param_descriptions

if TYPE_CHECKING:
    from ..server import SimulMCPServer


def register_stats_tools(server: "SimulMCPServer") -> None:
    """Register the read-only tool usage statistics tool.

    Clearing the log is an operator action, done from the CLI with
    ``simul-mcp stats --reset``; an agent must not be able to erase its own
    audit trail.
    """

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
    @with_param_descriptions()
    async def get_tool_usage_stats(
        tool_name: Optional[str] = None,
        include_recent: bool = False,
        limit: int = 50,
    ) -> ToolResult:
        """
        Get tool usage statistics.

        Args:
            tool_name: Filter to a specific tool.
            include_recent: Include recent call log.
            limit: Max recent records to return.

        Returns:
            Stats and optional recent call log as one JSON block.
        """
        result = server.usage_tracker.get_stats(tool_name=tool_name)
        if include_recent:
            result["recent"] = server.usage_tracker.get_recent(
                limit=limit, tool_name=tool_name,
            )
        return server._as_text_result(result)
