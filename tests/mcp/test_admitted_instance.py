"""An Isaac call runs on the instance it was admitted to, whatever the session switches to meanwhile.

Before (round-3 audit M3-1): admission resolved the instance, took its lock
and checked its claims, but the call's client was resolved again when the
tool body ran. A call queued behind a long script on ``default`` therefore
ran on ``b`` when the session called ``set_active_isaac_instance b`` while it
waited (``r3_h5c.py``: ``/World/QueuedForDefault`` landed on ``b``'s stage),
and with ``enforce_claims`` it got past another agent's claim on ``b``
(``r3_h5b.py``), with nothing in the result saying where it ran.

Now admission binds the instance and its client to the call
(``AdmittedInstance``) and every resolution inside the call reads them.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
from fastmcp import Client

from simul.config import Settings
from simul.mcp import server as server_module
from simul.mcp.session_manager import SessionManager
from simul.mcp.usage_tracker import ToolUsageTracker
from tests.tcp_fakes import FakeTcpServer


class _Kit:
    """Kit's 5.x stock socket in miniature: one script at a time; ``sleep(N)`` holds the main thread."""

    def __init__(self) -> None:
        self.ran: List[str] = []
        self.port = 0
        self._main_thread = asyncio.Lock()
        self._server = FakeTcpServer(self._handle)

    async def start(self) -> None:
        self.port = await self._server.start()

    async def stop(self) -> None:
        await self._server.stop()

    def printed(self) -> List[str]:
        found = (re.search(r"print\('([^']*)'\)", code) for code in self.ran)
        # set_active_isaac_instance pings the instance it switches to.
        return [match.group(1) for match in found if match and match.group(1) != "pong"]

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        code = (await reader.read(65536)).decode()
        async with self._main_thread:
            pause = re.search(r"sleep\(([\d.]+)\)", code)
            if pause:
                await asyncio.sleep(float(pause.group(1)))
            self.ran.append(code)
        printed = re.search(r"print\('([^']*)'\)", code)
        reply = {"status": "ok", "output": (printed.group(1) + "\n") if printed else ""}
        try:
            writer.write(json.dumps(reply).encode())
            await writer.drain()
            writer.close()
        except (ConnectionError, OSError):
            pass


def _server(tmp_path: Path, default: _Kit, other: _Kit, *, enforce: bool) -> server_module.SimulMCPServer:
    settings = Settings(
        server={"timeout": 10},
        isaac_sim={
            "socket_port": default.port,
            "socket_timeout": 5.0,
            "socket_protocol": "vscode",
            "bridge_enabled": False,
            "enforce_claims": enforce,
            "discovery_dir": str(tmp_path / "discovery"),
            "instances": [
                {"name": "b", "port": other.port, "timeout": 5.0, "bridge_enabled": False,
                 "socket_protocol": "vscode"}
            ],
        },
        security={"rate_limiting_enabled": False},
    )
    instance = server_module.SimulMCPServer(settings, backends={"isaac"})
    instance.session_manager = SessionManager(tmp_path / "sessions")
    instance.usage_tracker = ToolUsageTracker(log_dir=tmp_path / "usage")
    return instance


def _payload(result: Any) -> Dict[str, Any]:
    return json.loads(result.content[-1].text)


async def _call(client: Client, tool: str, **arguments: Any) -> Dict[str, Any]:
    return _payload(await client.call_tool(tool, arguments, raise_on_error=False))


async def _queue_then_switch(client: Client, switch: Dict[str, Any]) -> Tuple[Dict[str, Any], ...]:
    """The auditor's sequence: a long call, a call queued behind it, then a switch to ``b``."""
    long = asyncio.ensure_future(_call(client, "execute_isaac_script", code="sleep(1.0); print('long')"))
    await asyncio.sleep(0.3)
    queued = asyncio.ensure_future(_call(client, "execute_isaac_script", code="print('queued-for-default')"))
    await asyncio.sleep(0.3)
    switched = await _call(client, "set_active_isaac_instance", **switch)
    assert not queued.done(), "the queued call was still waiting for default's lock when the switch came"
    return switched, await long, await queued


def test_a_queued_call_runs_on_the_instance_it_was_issued_for(tmp_path: Path) -> None:
    async def scenario() -> Tuple[_Kit, _Kit, Tuple[Dict[str, Any], ...], Dict[str, Any]]:
        default, other = _Kit(), _Kit()
        await default.start()
        await other.start()
        instance = _server(tmp_path, default, other, enforce=False)
        try:
            async with Client(instance.mcp, mode="legacy") as client:
                outcome = await _queue_then_switch(client, {"instance_name": "b"})
                after = await _call(client, "execute_isaac_script", code="print('after-switch')")
            return default, other, outcome, after
        finally:
            await instance.shutdown()
            await default.stop()
            await other.stop()

    default, other, (switched, long, queued), after = asyncio.run(scenario())

    assert switched["success"] is True
    assert long["success"] is True and queued["success"] is True
    assert queued["output"].strip() == "queued-for-default"
    assert default.printed() == ["long", "queued-for-default"]
    # Only calls issued after the switch reach b.
    assert other.printed() == ["after-switch"]
    assert after["success"] is True


def test_a_queued_call_does_not_get_past_the_claims_of_the_instance_switched_to(tmp_path: Path) -> None:
    """r3_h5b.py: agent X claims b; agent A's call queued on default ran on b, past X's claim."""

    async def scenario() -> Tuple[_Kit, _Kit, Tuple[Dict[str, Any], ...], Dict[str, Any], Dict[str, Any]]:
        default, other = _Kit(), _Kit()
        await default.start()
        await other.start()
        instance = _server(tmp_path, default, other, enforce=True)
        try:
            async with Client(instance.mcp, mode="legacy") as x, Client(instance.mcp, mode="legacy") as a:
                claim = await _call(
                    x, "set_active_isaac_instance", instance_name="b", purpose="X owns b", agent_id="agent-X"
                )
                outcome = await _queue_then_switch(a, {"instance_name": "b"})
                direct = await _call(a, "execute_isaac_script", code="print('a-direct-on-b')")
            return default, other, outcome, direct, claim
        finally:
            await instance.shutdown()
            await default.stop()
            await other.stop()

    default, other, (switched, long, queued), direct, claim = asyncio.run(scenario())

    assert claim["success"] is True
    assert switched["success"] is True
    assert queued["success"] is True and queued["output"].strip() == "queued-for-default"
    assert default.printed() == ["long", "queued-for-default"]
    assert other.printed() == [], "nothing of A's reached the instance X claims"
    assert direct["error_type"] == "InstanceClaimed"


def test_a_timeout_holds_the_instance_the_call_ran_on(tmp_path: Path) -> None:
    """The timeout check read the session's current instance, so the one the call ran on was not held."""

    async def scenario() -> Tuple[List[str], Dict[str, Any], Dict[str, Any]]:
        default, other = _Kit(), _Kit()
        await default.start()
        await other.start()
        instance = _server(tmp_path, default, other, enforce=False)
        instance._isaac_clients["default"] = server_module.IsaacSocketClient(
            port=default.port, timeout_seconds=1.0, socket_protocol="vscode", prefer_bridge=False
        )
        try:
            async with Client(instance.mcp, mode="legacy") as client:
                blocker = asyncio.ensure_future(
                    _call(client, "execute_isaac_script", code="sleep(0.5); print('blocker')")
                )
                await asyncio.sleep(0.2)
                slow = asyncio.ensure_future(_call(client, "execute_isaac_script", code="sleep(3.5); print('slow')"))
                await asyncio.sleep(0.2)
                await _call(client, "set_active_isaac_instance", instance_name="b")
                await blocker
                timed_out = await slow
                held = sorted(instance._isaac_holds)
                on_b = await _call(client, "execute_isaac_script", code="print('on-b')")
            return held, timed_out, on_b
        finally:
            await instance.shutdown()
            await default.stop()
            await other.stop()

    held, timed_out, on_b = asyncio.run(scenario())

    assert timed_out["error_type"] == "TimeoutError"
    assert held == ["default"]
    assert on_b["success"] is True, "b was never held: the timed-out call did not run there"


def test_the_admitted_instance_is_only_bound_inside_the_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    instance = server_module.SimulMCPServer(Settings(), backends={"isaac"})
    seen: List[Tuple[str, Any]] = []

    async def scenario() -> None:
        async def body() -> Dict[str, Any]:
            seen.append((instance._get_effective_instance_name(), instance._get_request_isaac_client()))
            return {"success": True}

        await instance._exec_isaac("isaac_probe", body())

    asyncio.run(scenario())

    assert seen == [("default", instance._isaac_clients["default"])]
    assert server_module._ADMITTED_INSTANCE.get() is None
