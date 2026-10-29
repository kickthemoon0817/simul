"""A refused ``execute_unreal_script`` call echoes the script's size, not its source.

An invalid ``mode`` used to echo the whole script back in ``details.input``,
so a long script came straight back into the caller's context although the
usage log already keeps only its size. The echo is now the first
``SCRIPT_ECHO_CHARS`` characters with the size, the mode and whether the
script was cut. Other tools still echo their arguments as given.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional

import pytest
from fastmcp import Client

from simul.config import Settings
from simul.mcp import server as server_module
from simul.mcp.registration._unreal_tools import SCRIPT_ECHO_CHARS, _script_echo
from simul.mcp.usage_tracker import ToolUsageTracker
from tests.fakes import AvailableAdapter

LONG_SCRIPT = "import unreal\n" + "".join(f"value_{n} = {n}\n" for n in range(600))


def _server(tmp_path: Any) -> server_module.SimulMCPServer:
    settings = Settings(
        unreal={"tool_surface": "full", "attachment_path": str(tmp_path / "attachment.json")},
        security={"allow_script_execution": True, "rate_limiting_enabled": False},
    )
    instance = server_module.SimulMCPServer(settings, backends={"unreal"})
    instance.usage_tracker = ToolUsageTracker(log_dir=tmp_path / "usage")
    instance.unreal_adapter = AvailableAdapter(settings)
    return instance


def _call(instance: server_module.SimulMCPServer, tool: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    async def run() -> Any:
        async with Client(instance.mcp) as client:
            return await client.call_tool(tool, arguments, raise_on_error=False)

    result = asyncio.run(run())
    text = [block.text for block in result.content if getattr(block, "type", None) == "text"][-1]
    return {"text": text, "payload": json.loads(text)}


def test_a_long_script_with_a_bad_mode_is_echoed_by_size(tmp_path: Any) -> None:
    assert len(LONG_SCRIPT) > 5000

    outcome = _call(_server(tmp_path), "execute_unreal_script", {"code": LONG_SCRIPT, "mode": "Bogus"})

    payload = outcome["payload"]
    assert payload["success"] is False
    assert payload["error_type"] == "ValidationError"
    assert "Invalid mode 'Bogus'" in payload["error"]
    assert payload["details"] == {
        "input": {
            "code": LONG_SCRIPT[:SCRIPT_ECHO_CHARS],
            "code_bytes": len(LONG_SCRIPT),
            "mode": "Bogus",
            "code_truncated": True,
        }
    }
    # The rest of the source never reaches the caller.
    assert "value_599" not in outcome["text"]
    assert len(outcome["text"]) < 1000


def test_a_short_script_is_echoed_whole(tmp_path: Any) -> None:
    outcome = _call(_server(tmp_path), "execute_unreal_script", {"code": "print(1)", "mode": "Bogus"})

    assert outcome["payload"]["details"]["input"] == {
        "code": "print(1)",
        "code_bytes": 8,
        "mode": "Bogus",
        "code_truncated": False,
    }


def test_other_tools_still_echo_their_arguments(tmp_path: Any) -> None:
    arguments = {"resolution_x": 640, "resolution_y": 360, "format": "jpg", "inline": False}

    outcome = _call(_server(tmp_path), "capture_unreal_viewport", arguments)

    assert outcome["payload"]["error_type"] == "ValidationError"
    assert outcome["payload"]["details"] == {"input": arguments}


@pytest.mark.parametrize("size", [0, SCRIPT_ECHO_CHARS, SCRIPT_ECHO_CHARS + 1])
def test_script_echo_cuts_only_past_the_limit(size: int) -> None:
    code = "x" * size

    echo = _script_echo({"code": code, "mode": "ExecuteFile"})

    assert echo == {
        "code": code[:SCRIPT_ECHO_CHARS],
        "code_bytes": size,
        "mode": "ExecuteFile",
        "code_truncated": size > SCRIPT_ECHO_CHARS,
    }


def test_the_envelope_echoes_through_the_hook(tmp_path: Any) -> None:
    """``_exec_backend``'s ``input_echo`` replaces ``details.input``; without it the input is echoed as given."""
    from pydantic import BaseModel

    class Request(BaseModel):
        name: str
        count: int

    instance = _server(tmp_path)

    def call(session: Any, request: Any) -> Dict[str, Any]:
        raise AssertionError("an invalid request never reaches the session")

    def run(input_echo: Optional[Any]) -> Dict[str, Any]:
        result = asyncio.run(
            instance._exec_backend(
                "execute_unreal_script",
                instance.unreal_adapter,
                "Unreal",
                Request,
                call,
                request_model=Request,
                request_kwargs={"name": "n" * 500, "count": "many"},
                input_echo=input_echo,
            )
        )
        return json.loads(result.content[-1].text)

    assert run(None)["details"] == {"input": {"name": "n" * 500, "count": "many"}}
    assert run(lambda arguments: {"name_chars": len(arguments["name"])})["details"] == {
        "input": {"name_chars": 500}
    }
