"""Drive the headless USD tools through a real MCP server over stages built in ``tmp_path``.

The read-tool tests check what an agent receives, so they go through the real
FastMCP stack (validation, envelope, result budget, JSON encoding) rather than
calling the session directly.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict

import pytest

pytest.importorskip("pxr", reason="pxr library not available")

from fastmcp import Client  # noqa: E402

from simul.config import Settings  # noqa: E402
from simul.mcp.server import SimulMCPServer  # noqa: E402


class UsdTools:
    """Call one USD-only server's tools as an MCP client would."""

    def __init__(self, server: SimulMCPServer) -> None:
        self.server = server

    def raw(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Return the CallToolResult, without raising on an MCP-level error."""

        async def _run() -> Any:
            async with Client(self.server.mcp) as client:
                return await client.call_tool(name, arguments, raise_on_error=False)

        return asyncio.run(_run())

    def text(self, name: str, arguments: Dict[str, Any]) -> str:
        """Return the tool's JSON text exactly as it crossed the wire."""
        result = self.raw(name, arguments)
        text = result.content[-1].text
        # A failed call is an MCP error and still one JSON envelope.
        assert result.is_error is (json.loads(text).get("success") is False), result.content
        return text

    def call(self, name: str, **arguments: Any) -> Dict[str, Any]:
        """Invoke a tool and decode its payload with a strict JSON parser."""
        return json.loads(self.text(name, arguments), parse_constant=_reject_constant)

    def load(self, path: Path) -> str:
        """Load ``path`` and return its stage_id."""
        payload = self.call("load_usd_file", file_path=str(path))
        assert payload.get("stage_id"), payload
        return payload["stage_id"]


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-standard JSON constant {name} in a tool result")


@pytest.fixture
def usd_tools(tmp_path: Path) -> UsdTools:
    """A USD-only server whose sandbox is ``tmp_path``, with no rate limit."""
    base = Settings()
    settings = base.model_copy(
        update={
            "security": base.security.model_copy(
                update={"allowed_paths": [str(tmp_path)], "rate_limiting_enabled": False}
            )
        }
    )
    server = SimulMCPServer(settings, backends={"usd"})
    yield UsdTools(server)
    if server.headless_adapter is not None:
        server.headless_adapter.close()
