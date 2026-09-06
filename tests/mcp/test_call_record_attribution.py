"""Every Isaac call record names the agent, and script runs carry a hash.

An operator investigating a damaged scene needs to answer which agent ran
what. The agent id comes from the MCP session the request arrived on, and
``execute_isaac_script`` logs the SHA-256 of its source so identical scripts
match across sessions without storing the code.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import pytest


from simul_mcp.config import Settings
from simul_mcp.mcp import backends as backends_module
from simul_mcp.mcp import server as server_module
from simul_mcp.mcp.usage_tracker import CallRecord, ToolUsageTracker
from tests.fakes import FakeFastMCP

CODE = "print('hello')"


def _server(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> server_module.SimulMCPServer:
    monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
    monkeypatch.setattr(server_module, "TaskConfig", None)
    monkeypatch.setattr(backends_module, "is_headless_available", lambda: False)
    monkeypatch.setattr(backends_module, "is_blender_available", lambda: False)
    monkeypatch.setattr(backends_module, "UnrealRuntimeAdapter", None)
    instance = server_module.SimulMCPServer(settings=Settings())
    instance.usage_tracker = ToolUsageTracker(log_dir=tmp_path)

    async def fake_execute_script(code: str, keep_raw_output: bool = False) -> Dict[str, Any]:
        return {"success": True, "output": "ok"}

    monkeypatch.setattr(instance._isaac_tools, "execute_script", fake_execute_script)
    return instance


def _tool(instance: server_module.SimulMCPServer, name: str) -> Any:
    return next(tool.func for tool in instance.mcp.tools if tool.name == name)


def test_call_record_serialises_attribution_only_when_present() -> None:
    bare = CallRecord("t", 1.0, 2.0, True).to_dict()
    assert "agent_id" not in bare and "script_sha256" not in bare

    full = CallRecord("t", 1.0, 2.0, True, agent_id="sess-1", script_sha256="ab").to_dict()
    assert full["agent_id"] == "sess-1"
    assert full["script_sha256"] == "ab"


def test_tracker_record_accepts_attribution_and_persists_it(tmp_path: Path) -> None:
    tracker = ToolUsageTracker(log_dir=tmp_path)
    tracker.record("t", 1.0, True, agent_id="sess-1", script_sha256="ab")

    assert tracker.get_recent()[0]["agent_id"] == "sess-1"
    line = json.loads((tmp_path / "tool_usage.jsonl").read_text().strip())
    assert line["agent_id"] == "sess-1" and line["script_sha256"] == "ab"


def test_execute_isaac_script_logs_the_source_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    instance = _server(monkeypatch, tmp_path)

    asyncio.run(_tool(instance, "execute_isaac_script")(CODE))

    record = instance.usage_tracker.get_recent()[0]
    assert record["script_sha256"] == hashlib.sha256(CODE.encode("utf-8")).hexdigest()
    assert record["params"] == {"code_bytes": len(CODE)}
    assert record["agent_id"] == f"agent-{id(instance):x}"


def test_agent_id_comes_from_the_mcp_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    instance = _server(monkeypatch, tmp_path)
    monkeypatch.setattr(instance, "_get_request_session_id", lambda: "session-42")

    asyncio.run(_tool(instance, "execute_isaac_script")(CODE))

    assert instance.usage_tracker.get_recent()[0]["agent_id"] == "session-42"


def test_granular_tools_are_attributed_without_a_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    instance = _server(monkeypatch, tmp_path)

    async def fake_stage_info(include_prim_count: bool = False) -> Dict[str, Any]:
        return {"success": True}

    monkeypatch.setattr(instance._isaac_tools, "get_isaac_stage_info", fake_stage_info)
    asyncio.run(_tool(instance, "get_isaac_stage_info")())

    record = instance.usage_tracker.get_recent()[0]
    assert record["agent_id"]
    assert "script_sha256" not in record
