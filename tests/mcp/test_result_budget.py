"""Regression: tool results must not blow past the caller's context window.

Nothing between a tool's implementation and the calling agent stood between a
large stage and the response. The only guard in the path was the transport's
10 MB cap, so a single default call could return tens of thousands of tokens and
a permitted one could exceed any context window outright.

These tests pin the budget itself, that the Isaac chokepoint applies it, and the
defaults that decide how much a caller asks for in the first place.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_path))

from simul_mcp.adapters.isaac_socket_client import ScriptResult
from simul_mcp.config import Settings
from simul_mcp.mcp import backends as backends_module
from simul_mcp.mcp import server as server_module
from simul_mcp.mcp.result_budget import (
    DEFAULT_RESULT_BUDGET_BYTES,
    HARD_RESULT_LIMIT_BYTES,
    apply_result_budget,
)
from simul_mcp.mcp.tools.isaac_tools import IsaacTools


def _prims(count: int) -> List[Dict[str, Any]]:
    return [
        {
            "path": f"/World/Environment/Building_{i}/Floor/Mesh_{i}",
            "name": f"Mesh_{i}",
            "type": "Mesh",
            "depth": 4,
            "child_count": 3,
        }
        for i in range(count)
    ]


def _encoded_size(payload: Dict[str, Any]) -> int:
    return len(json.dumps(payload, default=str))


# ---------------------------------------------------------------------------
# The hard limit: strings that cannot be trimmed are replaced, not passed through
# ---------------------------------------------------------------------------


def test_string_above_the_hard_limit_is_replaced_by_a_marker() -> None:
    """A 3 MB script output must not arrive whole in the caller's context."""
    output = "x" * 3_000_000
    payload = {"success": True, "output": output, "stdout_lines": 1}

    result = apply_result_budget(payload)

    assert result["output"] == {
        "truncated_field": "output",
        "original_bytes": 3_000_000,
        "hint": result["output"]["hint"],
    }
    assert "hint" in result["output"]
    assert result["stdout_lines"] == 1
    assert result["oversized_bytes"] > HARD_RESULT_LIMIT_BYTES
    assert _encoded_size(result) <= HARD_RESULT_LIMIT_BYTES


def test_string_between_budget_and_hard_limit_passes_through_flagged() -> None:
    """Below the hard ceiling an opaque blob is still safer whole than cut."""
    payload = {"success": True, "output": "y" * 100_000}

    result = apply_result_budget(payload)

    assert result["output"] == payload["output"]
    assert result["oversized_bytes"] == _encoded_size(payload)


def test_hard_limit_replaces_strings_largest_first_until_it_fits() -> None:
    """Only as many strings go as it takes; the rest arrive whole."""
    payload = {"a": "a" * 250_000, "b": "b" * 260_000, "c": "c" * 270_000, "small": "keep"}

    result = apply_result_budget(payload)

    assert result["c"]["truncated_field"] == "c"
    assert result["b"]["truncated_field"] == "b"
    assert result["a"] == payload["a"]
    assert result["small"] == "keep"
    assert _encoded_size(result) <= HARD_RESULT_LIMIT_BYTES


def test_hard_limit_applies_after_list_trimming_left_a_large_string() -> None:
    """Trimming the list is not enough when a string carries the bulk."""
    payload = {"prims": _prims(2000), "log": "z" * 500_000}

    result = apply_result_budget(payload)

    assert result["truncated"] is True
    assert result["log"]["original_bytes"] == 500_000
    assert _encoded_size(result) <= HARD_RESULT_LIMIT_BYTES


# ---------------------------------------------------------------------------
# The budget itself
# ---------------------------------------------------------------------------


def test_oversized_list_is_truncated_to_fit() -> None:
    payload = {"success": True, "count": 5000, "prims": _prims(5000)}
    assert _encoded_size(payload) > DEFAULT_RESULT_BUDGET_BYTES

    trimmed = apply_result_budget(payload)

    assert _encoded_size(trimmed) <= DEFAULT_RESULT_BUDGET_BYTES
    assert trimmed["truncated"] is True
    assert trimmed["truncation"]["field"] == "prims"
    assert trimmed["truncation"]["total"] == 5000
    assert trimmed["truncation"]["returned"] == len(trimmed["prims"])
    assert 0 < len(trimmed["prims"]) < 5000


def test_truncation_keeps_a_usable_prefix() -> None:
    """Truncating must return real data, not just an apology."""
    original = _prims(5000)
    trimmed = apply_result_budget({"prims": list(original)})

    kept = trimmed["prims"]
    assert kept == original[: len(kept)]
    assert trimmed["truncation"]["hint"]


def test_small_payload_is_untouched() -> None:
    payload = {"success": True, "prims": _prims(3)}
    result = apply_result_budget(dict(payload))

    assert result == payload
    assert "truncated" not in result


def test_largest_list_is_the_one_trimmed() -> None:
    """A payload with several lists must lose the one actually costing bytes."""
    payload = {
        "root_prims": ["/World", "/Environment"],
        "prims": _prims(5000),
    }
    trimmed = apply_result_budget(payload)

    assert trimmed["root_prims"] == ["/World", "/Environment"]
    assert trimmed["truncation"]["field"] == "prims"


def test_payload_without_lists_is_left_alone() -> None:
    """No list to trim means leave the structure alone rather than mangle it."""
    payload = {"success": True, "blob": "x" * (DEFAULT_RESULT_BUDGET_BYTES + 10)}
    result = apply_result_budget(dict(payload))

    assert result["blob"] == payload["blob"]


# ---------------------------------------------------------------------------
# The Isaac chokepoint applies it
# ---------------------------------------------------------------------------


class FakeFastMCP:
    def __init__(self, name: str, version: str, **kwargs: Any):
        self.name = name
        self.version = version
        self.instructions = kwargs.get("instructions")
        self.tools: List[SimpleNamespace] = []

    def tool(self, name: str, **kwargs: Any) -> Callable[..., Any]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.tools.append(SimpleNamespace(name=name, func=func, kwargs=kwargs))
            return func

        return decorator

    def resource(self, *args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def add_middleware(self, middleware: Any) -> None:
        return


def _make_server(monkeypatch: pytest.MonkeyPatch) -> server_module.SimulMCPServer:
    monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
    monkeypatch.setattr(server_module, "TaskConfig", None)
    monkeypatch.setattr(backends_module, "is_headless_available", lambda: False)
    monkeypatch.setattr(backends_module, "is_blender_available", lambda: False)
    monkeypatch.setattr(backends_module, "UnrealRuntimeAdapter", None)
    return server_module.SimulMCPServer(settings=Settings())


def test_exec_isaac_applies_the_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = _make_server(monkeypatch)

    async def _huge() -> Dict[str, Any]:
        return {"success": True, "prims": _prims(5000)}

    result = asyncio.run(instance._exec_isaac("list_isaac_prims", _huge()))

    # _exec_isaac returns a ToolResult; what crosses the wire is the JSON in its
    # single content block, so that is what the budget has to bound.
    wire_text = result.content[0].text
    payload = json.loads(wire_text)

    assert len(wire_text) <= DEFAULT_RESULT_BUDGET_BYTES
    assert payload["truncated"] is True


# ---------------------------------------------------------------------------
# Defaults: how much a caller asks for before any truncation
# ---------------------------------------------------------------------------


def _registered(instance: server_module.SimulMCPServer, name: str) -> Callable[..., Any]:
    for tool in instance.mcp.tools:
        if tool.name == name:
            return tool.func
    raise AssertionError(f"Tool {name!r} not found")


def _default_of(func: Callable[..., Any], param: str) -> Any:
    return inspect.signature(func).parameters[param].default


def test_listing_defaults_are_context_sized(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = _make_server(monkeypatch)

    assert _default_of(_registered(instance, "list_isaac_prims"), "max_results") == 100
    assert _default_of(_registered(instance, "get_isaac_subtree"), "max_results") == 150
    assert (
        _default_of(_registered(instance, "list_isaac_extensions"), "enabled_only")
        is True
    )


def _tools() -> tuple[IsaacTools, List[str]]:
    captured: List[str] = []

    def _record(code: str) -> ScriptResult:
        captured.append(code)
        return ScriptResult(success=True, output=json.dumps({"ok": True}))

    client = MagicMock()
    client.address = "127.0.0.1:8226"
    client.timeout_seconds = 30.0
    client.bridge_enabled = False
    client.fallback_to_vscode = True
    client.execute = AsyncMock(side_effect=_record)
    client.bridge_request = AsyncMock(return_value=None)
    return IsaacTools(client, settings=Settings()), captured


def test_list_prims_clamp_is_lowered() -> None:
    """The clamp is the real ceiling; the old one allowed 10000 prims."""
    tools, captured = _tools()
    asyncio.run(tools.list_isaac_prims(root_path="/World", max_results=99999))
    script = captured[0]

    assert "99999" not in script, "requested max_results reached the script unclamped"
    assert re.search(r"=\s*1000\b", script), "expected the clamped ceiling of 1000"


def test_payload_whose_bulk_is_not_a_top_level_list_is_left_alone() -> None:
    """Trimming the largest top-level list must not make things worse.

    get_isaac_prim_detail returns its bulk inside dict values with one small
    metadata list ("aspects"). Trimming that list drops the record of what was
    asked for, claims a truncation that did not happen, and leaves the payload
    over budget anyway — it even grew, because the notice costs more than the
    list did.
    """
    bulk = {f"attr_{i}": "x" * 200 for i in range(300)}
    payload = {
        "success": True,
        "prim_path": "/World/Robot",
        "aspects": ["info", "mesh", "relationships"],
        "info": dict(bulk),
        "mesh": dict(bulk),
    }
    before = _encoded_size(payload)

    result = apply_result_budget(dict(payload))

    assert result["aspects"] == ["info", "mesh", "relationships"], "metadata list was eaten"
    assert result.get("truncated") is not True, "claimed a truncation that did not happen"
    assert _encoded_size(result) <= before + 64, "result grew"


def test_oversize_payload_it_cannot_trim_says_so() -> None:
    """Silently returning an over-budget payload hides the problem."""
    bulk = {f"attr_{i}": "x" * 200 for i in range(300)}
    result = apply_result_budget({"success": True, "info": bulk})

    assert result["oversized_bytes"] > DEFAULT_RESULT_BUDGET_BYTES
