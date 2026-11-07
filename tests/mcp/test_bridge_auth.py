"""The bridge honours an auth token, so a protected launch has no open port left.

``simul isaac launch --generate-auth-token`` turned on Isaac Sim 6.0's
python_server ``require_auth`` for 8226, while 8229 ran the same Python for any
local process: the bridge stripped the python_server token header unread. Now
the bridge checks a token on both of its paths (framed requests and raw code)
before anything is dispatched, follows python_server's ``require_auth`` and
accepts its token, and keeps accepting everything when no token is configured
anywhere, as before.
"""

from __future__ import annotations

import asyncio
import json
import struct
from typing import Any, Dict, List, Optional

import pytest

from khemoo.simul.extension import (
    BRIDGE_AUTH_TOKEN_SETTING,
    BRIDGE_REQUIRE_AUTH_SETTING,
    PYTHON_SERVER_AUTH_TOKEN_SETTING,
    PYTHON_SERVER_REQUIRE_AUTH_SETTING,
    bridge_auth_required,
    check_bridge_auth_token,
    parse_socket_mode,
    resolve_bridge_auth_token,
)
from khemoo.simul.lifecycle import AUTH_ERROR_NAME, BridgeServerLifecycle
from khemoo.simul.protocol import BridgeRequest, BridgeResponse

from simul.adapters.isaac_socket_client import IsaacSocketClient

# ---------------------------------------------------------------------------
# Settings: when auth is required and which token is accepted
# ---------------------------------------------------------------------------


def _settings(**values: Any) -> Any:
    store: Dict[str, Any] = dict(values)
    return store.get


def test_no_auth_settings_means_no_auth_as_before() -> None:
    get = _settings()

    assert bridge_auth_required(get) is False
    assert check_bridge_auth_token(get, "") is True


def test_bridge_follows_python_server_require_auth_and_its_token() -> None:
    get = _settings(
        **{
            PYTHON_SERVER_REQUIRE_AUTH_SETTING: True,
            PYTHON_SERVER_AUTH_TOKEN_SETTING: "kit-generated",
        }
    )

    assert bridge_auth_required(get) is True
    assert check_bridge_auth_token(get, "kit-generated") is True
    assert check_bridge_auth_token(get, "") is False
    assert check_bridge_auth_token(get, "guess") is False


def test_the_bridge_own_token_wins_and_implies_auth() -> None:
    get = _settings(
        **{
            BRIDGE_AUTH_TOKEN_SETTING: "bridge-token",
            PYTHON_SERVER_AUTH_TOKEN_SETTING: "socket-token",
        }
    )

    assert bridge_auth_required(get) is True
    assert resolve_bridge_auth_token(get) == "bridge-token"
    assert check_bridge_auth_token(get, "bridge-token") is True
    assert check_bridge_auth_token(get, "socket-token") is False


def test_required_auth_without_any_token_fails_closed() -> None:
    get = _settings(**{BRIDGE_REQUIRE_AUTH_SETTING: True})

    assert check_bridge_auth_token(get, "") is False
    assert check_bridge_auth_token(get, "anything") is False


def test_an_all_digit_token_kit_parsed_as_an_int_still_matches() -> None:
    get = _settings(**{BRIDGE_AUTH_TOKEN_SETTING: 123456})

    assert check_bridge_auth_token(get, "123456") is True


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, 0o600), ("", 0o600), ("0600", 0o600), ("0666", 0o666), (666, 0o666), ("0o660", 0o660), ("rw", 0o600)],
)
def test_socket_mode_setting_reads_as_octal(value: Any, expected: int) -> None:
    assert parse_socket_mode(value) == expected


def test_the_token_never_shows_in_a_logged_request() -> None:
    request = BridgeRequest.from_json(
        json.dumps({"request_id": "r", "action": "ping", "auth_token": "s3cret"}).encode()
    )

    assert request.auth_token == "s3cret"
    assert "s3cret" not in repr(request)
    assert "auth_token" not in request.payload


# ---------------------------------------------------------------------------
# Lifecycle: both paths check the token before dispatching
# ---------------------------------------------------------------------------


class _Recorder:
    def __init__(self) -> None:
        self.requests: List[BridgeRequest] = []
        self.scripts: List[str] = []

    async def handle(self, request: BridgeRequest) -> BridgeResponse:
        self.requests.append(request)
        return BridgeResponse.success(request.request_id, {"reachable": True})

    async def run_code(self, code: str) -> Dict[str, Any]:
        self.scripts.append(code)
        return {"status": "ok", "output": "ran"}


def _lifecycle(recorder: _Recorder, token: Optional[str]) -> BridgeServerLifecycle:
    return BridgeServerLifecycle(
        host="127.0.0.1",
        port=0,
        request_handler=recorder.handle,
        vscode_handler=recorder.run_code,
        authenticate=None if token is None else (lambda presented: presented == token),
    )


async def _frame(port: int, request: Dict[str, Any]) -> Dict[str, Any]:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    body = json.dumps(request).encode()
    writer.write(struct.pack(">I", len(body)) + body)
    await writer.drain()
    size = struct.unpack(">I", await reader.readexactly(4))[0]
    reply = json.loads(await reader.readexactly(size))
    writer.close()
    return reply


async def _raw(port: int, source: str) -> Dict[str, Any]:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(source.encode())
    await writer.drain()
    writer.write_eof()
    reply = json.loads((await reader.read()).decode())
    writer.close()
    return reply


def _against(token: Optional[str], exchange: Any) -> tuple[_Recorder, Any]:
    async def scenario() -> tuple[_Recorder, Any]:
        recorder = _Recorder()
        lifecycle = _lifecycle(recorder, token)
        await lifecycle.start()
        try:
            return recorder, await exchange(lifecycle.actual_port)
        finally:
            await lifecycle.stop()

    return asyncio.run(scenario())


def test_framed_request_without_the_token_is_rejected_before_dispatch() -> None:
    recorder, reply = _against(
        "s3cret", lambda port: _frame(port, {"request_id": "1", "action": "ping", "payload": {}})
    )

    assert reply["status"] == "error"
    assert reply["error"]["name"] == AUTH_ERROR_NAME
    assert recorder.requests == []  # not even a read-only action ran


def test_framed_request_with_the_token_is_served() -> None:
    recorder, reply = _against(
        "s3cret",
        lambda port: _frame(
            port, {"request_id": "1", "action": "ping", "payload": {}, "auth_token": "s3cret"}
        ),
    )

    assert reply["status"] == "ok"
    assert len(recorder.requests) == 1


def test_raw_code_without_the_header_token_never_runs() -> None:
    recorder, reply = _against("s3cret", lambda port: _raw(port, "import os; os.remove('x')"))

    assert reply["status"] == "error"
    assert reply["ename"] == AUTH_ERROR_NAME
    assert recorder.scripts == []


def test_raw_code_with_the_python_server_header_token_runs_without_the_header() -> None:
    recorder, reply = _against(
        "s3cret", lambda port: _raw(port, "# isaacsim-python-server-token: s3cret\nprint('hi')")
    )

    assert reply["status"] == "ok"
    assert recorder.scripts == ["print('hi')"]


def test_without_an_authenticator_everything_is_served_as_before() -> None:
    recorder, reply = _against(
        None, lambda port: _frame(port, {"request_id": "1", "action": "ping", "payload": {}})
    )

    assert reply["status"] == "ok"
    assert len(recorder.requests) == 1


def test_an_authenticator_that_raises_rejects_the_request() -> None:
    async def scenario() -> tuple[_Recorder, Dict[str, Any]]:
        recorder = _Recorder()

        def _broken(_presented: str) -> bool:
            raise RuntimeError("settings unavailable")

        lifecycle = BridgeServerLifecycle(
            host="127.0.0.1", port=0, request_handler=recorder.handle, authenticate=_broken
        )
        await lifecycle.start()
        try:
            reply = await _frame(
                lifecycle.actual_port, {"request_id": "1", "action": "ping", "payload": {}}
            )
        finally:
            await lifecycle.stop()
        return recorder, reply

    recorder, reply = asyncio.run(scenario())

    assert reply["error"]["name"] == AUTH_ERROR_NAME
    assert recorder.requests == []


# ---------------------------------------------------------------------------
# End to end: the real client authenticates to a bridge that requires it
# ---------------------------------------------------------------------------


def _client_ping(token: Optional[str], client_token: Optional[str]) -> Dict[str, Any]:
    async def scenario() -> Dict[str, Any]:
        recorder = _Recorder()
        lifecycle = _lifecycle(recorder, token)
        await lifecycle.start()
        try:
            client = IsaacSocketClient(
                bridge_host="127.0.0.1",
                bridge_port=lifecycle.actual_port,
                prefer_bridge=True,
                auth_token=client_token,
                timeout_seconds=5.0,
            )
            return await client.bridge_request("ping", {})
        finally:
            await lifecycle.stop()

    return asyncio.run(scenario())


def test_client_with_the_launch_token_reaches_a_protected_bridge() -> None:
    assert _client_ping("kit-token", "kit-token")["status"] == "ok"


def test_client_without_the_token_is_refused_by_a_protected_bridge() -> None:
    reply = _client_ping("kit-token", None)

    assert reply["status"] == "error"
    assert reply["error"]["name"] == AUTH_ERROR_NAME


def test_client_token_is_harmless_against_an_open_bridge() -> None:
    assert _client_ping(None, "kit-token")["status"] == "ok"
