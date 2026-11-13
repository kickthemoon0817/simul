"""
Tool usage statistics registration for the Simul MCP server.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Dict, Literal, Optional

from fastmcp.tools import ToolResult
from pydantic import Field

from ..usage_tracker import MAX_RECENT
from ._helpers import with_param_descriptions

if TYPE_CHECKING:
    from ..server import SimulMCPServer


def register_stats_tools(server: "SimulMCPServer") -> None:
    """Register the read-only tool usage statistics tool.

    Clearing the log is an operator action, done from the CLI with
    ``simul stats --reset``; an agent must not be able to erase its own
    audit trail. Nor may it read another agent's: the records carry their
    parameters, file paths and agent labels, so the tool shows the caller its
    own session's calls, and only aggregate counts for the whole server. The
    operator reads everything with ``simul stats``.
    """

    @server.mcp.tool(
        name="get_tool_usage_stats",
        description=(
            "Get tool usage statistics for this MCP session: per-tool call counts, "
            "success/failure rates, average duration, and the recent call log. Use "
            "tool_name to filter to a single tool, include_recent=true with limit for "
            "the last N of your own call records with parameters and timing, and "
            "scope='server' for per-tool counts across every session on this server "
            "(recent records stay your own)."
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
        limit: Annotated[int, Field(ge=1, le=MAX_RECENT)] = 50,
        scope: Literal["session", "server"] = "session",
    ) -> ToolResult:
        """
        Get tool usage statistics.

        Args:
            tool_name: Filter to a specific tool.
            include_recent: Include recent call log.
            limit: Max recent records to return, 1 to 200.
            scope: 'session' (default) counts this session's calls; 'server'
                counts every session's. The recent log is always this
                session's own.

        Returns:
            Stats and optional recent call log as one JSON block.
        """

        async def compute() -> Dict[str, Any]:
            # The session label the usage records carry (see _run_envelope).
            session = server._resolve_agent_id(None)
            result = server.usage_tracker.get_stats(
                tool_name=tool_name, session=None if scope == "server" else session
            )
            result["scope"] = scope
            if include_recent:
                result["recent"] = server.usage_tracker.get_recent(
                    limit=limit, tool_name=tool_name, session=session
                )
            return result

        return await server._exec_payload("get_tool_usage_stats", compute)
