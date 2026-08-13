"""Regression: the sandbox path policy must sit below every caller.

``_is_path_allowed`` was enforced only in the MCP registration layer, but the
CLI calls the ``IsaacTools`` methods directly — so ``simul isaac open-stage``,
``save-stage``, ``import-asset`` and ``add-reference`` read and wrote outside the
sandbox the MCP surface enforces. One control with two entry points and one
check.

These tests drive the tools layer, which is the entry point the CLI uses.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_path))

from simul_mcp.adapters.isaac_socket_client import ScriptResult
from simul_mcp.config import Settings
from simul_mcp.mcp.tools.isaac_tools import IsaacTools

# Outside every default allowed root (examples, tests/data, /tmp/simul_mcp).
OUTSIDE_SANDBOX = "/etc/shadow"
INSIDE_SANDBOX = "/tmp/simul_mcp/scene.usd"


def _tools() -> tuple[IsaacTools, MagicMock]:
    client = MagicMock()
    client.address = "127.0.0.1:8226"
    client.timeout_seconds = 30.0
    client.bridge_enabled = False
    client.fallback_to_vscode = True
    result = ScriptResult(success=True, output=json.dumps({"ok": True}))
    client.execute = AsyncMock(return_value=result)
    client.execute_vscode_only = AsyncMock(return_value=result)
    client.execute_bridge_script_only = AsyncMock(return_value=result)
    client.bridge_request = AsyncMock(return_value=None)
    return IsaacTools(client, settings=Settings()), client


def _denied(result: Dict[str, Any]) -> bool:
    return result.get("error_type") == "SandboxError"


def _assert_blocked(result: Dict[str, Any], client: MagicMock) -> None:
    assert _denied(result), f"expected a SandboxError, got {result}"
    assert client.execute.await_count == 0, "script ran despite the sandbox denial"
    assert client.execute_vscode_only.await_count == 0
    assert client.execute_bridge_script_only.await_count == 0


def test_open_stage_outside_sandbox_is_refused() -> None:
    tools, client = _tools()
    result = asyncio.run(tools.open_isaac_stage(file_path=OUTSIDE_SANDBOX))
    _assert_blocked(result, client)


def test_save_stage_outside_sandbox_is_refused() -> None:
    tools, client = _tools()
    result = asyncio.run(tools.save_isaac_stage(file_path=OUTSIDE_SANDBOX))
    _assert_blocked(result, client)


def test_import_asset_outside_sandbox_is_refused() -> None:
    tools, client = _tools()
    result = asyncio.run(tools.import_isaac_asset(asset_path=OUTSIDE_SANDBOX))
    _assert_blocked(result, client)


def test_add_reference_outside_sandbox_is_refused() -> None:
    tools, client = _tools()
    result = asyncio.run(
        tools.add_isaac_reference(
            prim_path="/World/Ref", reference_path=OUTSIDE_SANDBOX
        )
    )
    _assert_blocked(result, client)


def test_allowed_path_still_runs() -> None:
    """The check must not block work inside the sandbox."""
    tools, client = _tools()
    result = asyncio.run(tools.open_isaac_stage(file_path=INSIDE_SANDBOX))

    assert not _denied(result)
    assert client.execute.await_count == 1


def test_save_without_a_path_still_runs() -> None:
    """save-as is optional; a plain save has no path to police."""
    tools, client = _tools()
    result = asyncio.run(tools.save_isaac_stage())

    assert not _denied(result)
    assert client.execute.await_count == 1


def test_remote_urls_count_as_outside_the_sandbox() -> None:
    """Pin current behaviour: the sandbox also refuses omniverse:// URLs.

    This is what the MCP layer already did — a URL is not under an allowed root,
    so it is refused — and the CLI now matches it. Whether a Nucleus URL *should*
    be exempt from a filesystem sandbox is a policy question rather than a bug,
    so it is recorded here instead of changed silently. If the answer becomes
    "exempt remote schemes", this is the test to flip.
    """
    tools, client = _tools()
    result = asyncio.run(
        tools.open_isaac_stage(file_path="omniverse://nucleus/Projects/scene.usd")
    )
    _assert_blocked(result, client)


def test_sandbox_disabled_allows_any_path() -> None:
    """With the sandbox off the policy must get out of the way entirely."""
    settings = Settings()
    settings = settings.model_copy(
        update={"security": settings.security.model_copy(update={"sandbox_enabled": False})}
    )
    client = MagicMock()
    client.address = "127.0.0.1:8226"
    client.timeout_seconds = 30.0
    client.bridge_enabled = False
    client.fallback_to_vscode = True
    result_obj = ScriptResult(success=True, output=json.dumps({"ok": True}))
    client.execute = AsyncMock(return_value=result_obj)
    client.bridge_request = AsyncMock(return_value=None)
    tools = IsaacTools(client, settings=settings)

    result = asyncio.run(tools.open_isaac_stage(file_path=OUTSIDE_SANDBOX))

    assert not _denied(result)
    assert client.execute.await_count == 1
