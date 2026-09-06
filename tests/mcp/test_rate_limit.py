"""Rate limiting is keyed per agent and per tool, with a per-agent ceiling above.

The limiter used to be one bucket per tool name shared by every agent, and only
a handful of tools consulted it: Isaac tools never did, so 90 back-to-back calls
went through in under a second, and a throttled agent could switch to
execute_isaac_script whose bucket was untouched. One looping agent starved
everyone else on the same tool, with no hint of when to retry.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest


from simul_mcp.config import Settings
from simul_mcp.mcp import backends as backends_module
from simul_mcp.mcp import server as server_module
from simul_mcp.utils import timing as timing_module
from tests.fakes import FakeFastMCP

BURST = 10
PER_TOOL_PER_MINUTE = 60


def _make_server(
    monkeypatch: pytest.MonkeyPatch, **security: Any
) -> server_module.SimulMCPServer:
    monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
    monkeypatch.setattr(server_module, "TaskConfig", None)
    monkeypatch.setattr(backends_module, "is_headless_available", lambda: False)
    monkeypatch.setattr(backends_module, "is_blender_available", lambda: False)
    monkeypatch.setattr(backends_module, "UnrealRuntimeAdapter", None)
    return server_module.SimulMCPServer(settings=Settings(security=security))


def _call_n(
    instance: server_module.SimulMCPServer, tool: str, agent: str, count: int
) -> List[Optional[Dict[str, Any]]]:
    return [instance._check_rate_limit(tool, agent) for _ in range(count)]


def _payload(result: Any) -> Dict[str, Any]:
    assert result.structured_content is None
    return json.loads(result.content[0].text)


def test_sixty_one_rapid_calls_from_one_agent_hit_the_limit_with_a_retry_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = _make_server(monkeypatch)

    outcomes = _call_n(instance, "get_isaac_stage_info", "agent-a", 61)

    assert outcomes[:BURST] == [None] * BURST, "the burst must go through untouched"
    refused = outcomes[60]
    assert refused is not None
    assert refused["success"] is False
    assert refused["error_type"] == "RateLimitError"
    assert refused["retry_after_seconds"] > 0
    assert refused["details"]["scope"] == "tool"
    assert refused["details"]["tool"] == "get_isaac_stage_info"
    assert refused["details"]["agent_id"] == "agent-a"
    assert refused["details"]["limit_per_minute"] == PER_TOOL_PER_MINUTE


def test_a_second_agent_is_unaffected_by_the_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = _make_server(monkeypatch)
    _call_n(instance, "get_isaac_stage_info", "agent-a", 61)

    assert instance._check_rate_limit("get_isaac_stage_info", "agent-b") is None
    assert _call_n(instance, "get_isaac_stage_info", "agent-b", BURST - 1) == [None] * (BURST - 1)


def test_a_refusal_spends_nothing_and_the_retry_hint_is_honest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Waiting exactly retry_after_seconds must yield the token that was promised."""
    clock = {"now": 1_000_000.0}
    monkeypatch.setattr(timing_module.time, "time", lambda: clock["now"])
    instance = _make_server(monkeypatch)

    _call_n(instance, "list_isaac_prims", "agent-a", BURST)
    refused = instance._check_rate_limit("list_isaac_prims", "agent-a")
    assert refused is not None
    for _ in range(20):
        # Hammering while refused must not push the retry further out.
        again = instance._check_rate_limit("list_isaac_prims", "agent-a")
        assert again is not None
        assert again["retry_after_seconds"] == refused["retry_after_seconds"]

    clock["now"] += refused["retry_after_seconds"]
    assert instance._check_rate_limit("list_isaac_prims", "agent-a") is None


def test_switching_tools_does_not_escape_the_per_agent_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A throttled agent that hops to a fresh tool meets the agent-wide bucket."""
    instance = _make_server(monkeypatch)
    ceiling = instance._global_rate_limit_burst
    assert ceiling >= BURST

    tool_index = 0
    granted = 0
    while granted < ceiling:
        # Stay under each tool's own burst so only the ceiling can refuse.
        for _ in range(BURST - 1):
            assert instance._check_rate_limit(f"tool_{tool_index}", "agent-a") is None
            granted += 1
            if granted == ceiling:
                break
        tool_index += 1

    refused = instance._check_rate_limit("execute_isaac_script", "agent-a")
    assert refused is not None
    assert refused["details"]["scope"] == "agent"
    assert refused["details"]["limit_per_minute"] == 600
    assert refused["retry_after_seconds"] > 0
    assert instance._check_rate_limit("execute_isaac_script", "agent-b") is None


def test_ceiling_default_does_not_throttle_one_well_behaved_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sixty calls spread over sixty tools is an ordinary session, not a runaway."""
    instance = _make_server(monkeypatch)

    outcomes = [instance._check_rate_limit(f"tool_{i}", "agent-a") for i in range(60)]

    assert outcomes == [None] * 60


def test_agent_identity_comes_from_the_mcp_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = _make_server(monkeypatch)
    monkeypatch.setattr(instance, "_get_request_session_id", lambda: "session-A")
    _call_n(instance, "get_isaac_stage_info", None, BURST)  # type: ignore[arg-type]
    refused = instance._check_rate_limit("get_isaac_stage_info")
    assert refused is not None
    assert refused["details"]["agent_id"] == "session-A"

    monkeypatch.setattr(instance, "_get_request_session_id", lambda: "session-B")
    assert instance._check_rate_limit("get_isaac_stage_info") is None


def test_exec_isaac_refuses_as_a_single_block_and_records_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = _make_server(monkeypatch)
    instance.usage_tracker._recent.clear()
    _call_n(instance, "get_isaac_stage_info", instance._resolve_agent_id(None), BURST)

    async def _never_runs() -> Dict[str, Any]:
        raise AssertionError("a refused call must not reach the backend")

    result = asyncio.run(instance._exec_isaac("get_isaac_stage_info", _never_runs()))

    payload = _payload(result)
    assert payload["error_type"] == "RateLimitError"
    assert payload["retry_after_seconds"] > 0
    (record,) = instance.usage_tracker.get_recent(tool_name="get_isaac_stage_info")
    assert record["error"] == "rate_limited"


def test_exec_backend_refuses_before_opening_a_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = _make_server(monkeypatch)
    _call_n(instance, "spawn_unreal_actor", instance._resolve_agent_id(None), BURST)
    adapter = SimpleNamespace(
        is_available=lambda: True,
        create_session=lambda: (_ for _ in ()).throw(AssertionError("session opened")),
    )

    result = asyncio.run(
        instance._exec_backend(
            "spawn_unreal_actor",
            adapter,
            "Unreal",
            server_module.ErrorResponse,
            lambda session: {},
        )
    )

    payload = _payload(result)
    assert payload["error_type"] == "RateLimitError"
    assert payload["retry_after_seconds"] > 0


def test_disabled_limiter_never_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = _make_server(monkeypatch, rate_limiting_enabled=False)

    assert _call_n(instance, "get_isaac_stage_info", "agent-a", 500) == [None] * 500
    assert instance._rate_limiters == {}


def test_global_ceiling_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    instance = _make_server(monkeypatch, global_requests_per_minute=120)

    assert instance._global_rate_limit_rate == pytest.approx(2.0)
    # Never below the per-tool burst, or a single tool's burst could never run.
    assert instance._global_rate_limit_burst == max(BURST, 120 // 6)
