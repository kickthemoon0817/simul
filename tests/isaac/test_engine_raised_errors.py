"""A failure raised inside Isaac Sim is the engine's (or the agent's script's), whatever its name.

Against a real ``IsaacSocketClient`` and a stock-socket double answering
``{"status": "error", "ename": X}``, before the fix:

* an ``OSError`` raised inside Kit (a full disk) read as ``ConnectionError``,
  "engine unreachable, retry";
* an agent script that raised ``TimeoutError`` or ``ConnectionRefusedError``
  of its own read as a transport failure;
* a ``TypeError`` from a changed Kit API read as ``ValidationError``, "fix
  the call", and a ``ModuleNotFoundError`` for a removed ``omni.isaac.*``
  module as ``NotFoundError``;
* a reply that is not JSON read as ``ValidationError`` (granular tools) or
  ``ScriptError`` (``execute_isaac_script``);
* the bridge's own refusals (``StageUnavailable``, ``UnsafeExecutionDisabled``)
  read as ``EngineError``, and a prim-detail aspect that failed inside Kit
  kept the raw class name.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Awaitable, Callable, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

from simul.adapters.isaac_socket_client import IsaacSocketClient
from simul.config import Settings
from simul.mcp.tools.isaac_tools import IsaacTools


def _against_socket(reply: bytes, call: Callable[[IsaacTools], Awaitable[Dict[str, Any]]]) -> Dict[str, Any]:
    async def scenario() -> Dict[str, Any]:
        async def answer(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            await reader.read(1 << 20)
            writer.write(reply)
            await writer.drain()
            writer.close()

        server = await asyncio.start_server(answer, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        client = IsaacSocketClient(port=port, timeout_seconds=2.0, socket_protocol="vscode")
        try:
            return await call(IsaacTools(client, settings=Settings()))
        finally:
            server.close()
            await server.wait_closed()

    return asyncio.run(scenario())


def _raised(name: str) -> bytes:
    return json.dumps(
        {"status": "error", "ename": name, "evalue": "it failed", "traceback": ["Traceback ...", f"{name}: it failed"]}
    ).encode()


async def _granular(tools: IsaacTools) -> Dict[str, Any]:
    return await tools.get_isaac_stage_info()


async def _agent_script(tools: IsaacTools) -> Dict[str, Any]:
    return await tools.execute_script("do_things()")


@pytest.mark.parametrize(
    "name",
    ["OSError", "TypeError", "ModuleNotFoundError", "TimeoutError", "ConnectionRefusedError", "JSONDecodeError"],
)
def test_an_exception_a_tool_script_raised_inside_kit_is_an_engine_error(name: str) -> None:
    result = _against_socket(_raised(name), _granular)

    assert result["success"] is False
    assert result["error_type"] == "EngineError"
    assert result["details"]["exception_type"] == name
    assert result["details"]["traceback"]


@pytest.mark.parametrize(
    "name", ["OSError", "TimeoutError", "ConnectionRefusedError", "TypeError", "ModuleNotFoundError"]
)
def test_whatever_an_agent_script_raised_is_a_script_error(name: str) -> None:
    result = _against_socket(_raised(name), _agent_script)

    assert result["success"] is False
    assert result["error_type"] == "ScriptError"
    assert result["details"]["exception_type"] == name


@pytest.mark.parametrize("call", [_granular, _agent_script], ids=["granular", "agent-script"])
def test_a_reply_that_is_not_json_is_a_protocol_error(call: Any) -> None:
    result = _against_socket(b"<html>not the Isaac socket</html>", call)

    assert result["success"] is False
    assert result["error_type"] == "BridgeProtocolError"
    assert result["details"]["exception_type"] == "JSONDecodeError"


def _bridge_client(response: Dict[str, Any]) -> MagicMock:
    client = MagicMock()
    client.address = client.bridge_address = client.bridge_endpoint = "127.0.0.1:8229"
    client.bridge_enabled = True
    client.fallback_to_vscode = False
    client.bridge_request = AsyncMock(return_value=response)
    return client


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("StageUnavailable", "InvalidState"),
        ("PrimNotFound", "NotFoundError"),
        ("InvalidAspect", "ValidationError"),
        ("RequestTimeout", "TimeoutError"),
        ("AttributeError", "EngineError"),
    ],
)
def test_the_bridge_s_refusals_keep_their_meaning(name: str, expected: str) -> None:
    client = _bridge_client({"status": "error", "error": {"name": name, "message": "refused"}})

    result = asyncio.run(IsaacTools(client, settings=Settings()).get_isaac_prim_detail("/World/X"))

    assert result["error_type"] == expected
    assert result["details"]["exception_type"] == name


def test_a_failed_prim_detail_aspect_gets_a_vocabulary_type() -> None:
    client = _bridge_client(
        {
            "status": "ok",
            "payload": {
                "prim_path": "/World/X",
                "aspects": ["info", "mesh"],
                "info": {"name": "X", "success": True},
                "mesh": {
                    "error": "'NoneType' object has no attribute 'Get'",
                    "error_type": "AttributeError",
                    "success": False,
                },
            },
        }
    )

    result = asyncio.run(
        IsaacTools(client, settings=Settings()).get_isaac_prim_detail("/World/X", ["info", "mesh"])
    )

    assert result["info"] == {"name": "X", "success": True}
    assert result["mesh"]["error_type"] == "EngineError"
    assert result["mesh"]["details"] == {"exception_type": "AttributeError"}
