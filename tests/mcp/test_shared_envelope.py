"""The one envelope Isaac, USD, Blender and Unreal tools share.

``_exec_isaac`` and ``_exec_backend`` used to be two copies of rate limiting,
usage recording, exception wrapping and success normalisation, and they
diverged: Isaac named an exception by its class and dropped its details, the
backend copy reported ``Exception`` unless the exception declared a type, and
input validation ran before the backend copy, outside the rate limit and the
usage log. Both now run through ``_run_envelope``; Isaac's claim check and
instance lock plug in as its admission hook.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional

import pytest
from pydantic import BaseModel, Field

from simul.adapters.blender_connection import AttachmentStale
from simul.blender_bridge.protocol import BridgeRemoteError
from simul.config import Settings
from simul.mcp import backends as backends_module
from simul.mcp import server as server_module
from simul.mcp.usage_tracker import ToolUsageTracker
from simul.utils.errors import exception_message, exception_type_name
from simul.utils.paths import SandboxDenied
from tests.fakes import FakeFastMCP


def _server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **security: Any
) -> server_module.SimulMCPServer:
    monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
    monkeypatch.setattr(server_module, "TaskConfig", None)
    monkeypatch.setattr(backends_module, "is_headless_available", lambda: False)
    monkeypatch.setattr(backends_module, "is_blender_available", lambda: False)
    monkeypatch.setattr(backends_module, "UnrealRuntimeAdapter", None)
    instance = server_module.SimulMCPServer(settings=Settings(security=security))
    instance.usage_tracker = ToolUsageTracker(log_dir=tmp_path)
    return instance


def _payload(result: Any) -> Dict[str, Any]:
    assert result.structured_content is None
    return json.loads(result.content[-1].text)


def _records(instance: server_module.SimulMCPServer, tool: str) -> List[Dict[str, Any]]:
    rows = list(reversed(instance.usage_tracker.get_recent(tool_name=tool)))
    return [{"success": row["success"], "error": row.get("error")} for row in rows]


class _Response(BaseModel):
    success: bool = Field(default=True)
    error: Optional[str] = Field(default=None)
    value: str = Field(default="")


class _Request(BaseModel):
    name: str = Field(..., min_length=1)


class _Adapter:
    def __init__(self, session: Any) -> None:
        self.session = session
        self.sessions_opened = 0

    def is_available(self) -> bool:
        return True

    @contextmanager
    def create_session(self) -> Iterator[Any]:
        self.sessions_opened += 1
        yield self.session


def _noted(exc: Exception, *notes: str) -> Exception:
    for note in notes:
        exc.add_note(note)
    return exc


# ---------------------------------------------------------------------------
# One rule for exceptions, whichever backend raised them
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        pytest.param(
            ConnectionResetError(54, "Connection reset by peer"),
            {
                "error": "[Errno 54] Connection reset by peer",
                "error_type": "ConnectionResetError",
            },
            id="class-name",
        ),
        pytest.param(
            AttachmentStale("Blender (pid 7) is not running", {"pid": 7}),
            {
                "error": "Blender (pid 7) is not running",
                "error_type": "AttachmentStale",
                "details": {"pid": 7},
            },
            id="declared-type-and-details",
        ),
        pytest.param(
            BridgeRemoteError(
                "ValueError: Attached window is not in Object Mode", "ValueError", {}
            ),
            {
                "error": "ValueError: Attached window is not in Object Mode",
                "error_type": "ValueError",
            },
            id="class-raised-inside-blender",
        ),
        pytest.param(
            _noted(RuntimeError("poll() failed"), "Failed to add rigid body to 'Cube'"),
            {
                "error": "Failed to add rigid body to 'Cube': poll() failed",
                "error_type": "RuntimeError",
            },
            id="note-as-context",
        ),
        pytest.param(
            TimeoutError(),
            {"error": "TimeoutError", "error_type": "TimeoutError"},
            id="no-message",
        ),
    ],
)
def test_isaac_and_backend_tools_report_an_exception_identically(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    exc: Exception,
    expected: Dict[str, Any],
) -> None:
    instance = _server(monkeypatch, tmp_path)

    async def isaac_tool() -> Dict[str, Any]:
        raise exc

    def backend_call(session: Any) -> Dict[str, Any]:
        raise exc

    isaac = _payload(asyncio.run(instance._exec_isaac("isaac_tool", isaac_tool())))
    backend = _payload(
        asyncio.run(
            instance._exec_backend(
                "backend_tool", _Adapter(object()), "Unreal", _Response, backend_call
            )
        )
    )

    assert isaac == backend == {"success": False, "details": None, **expected}
    assert (
        _records(instance, "isaac_tool")
        == _records(instance, "backend_tool")
        == [{"success": False, "error": expected["error"]}]
    )


def test_a_sandbox_denial_keeps_its_envelope_on_both_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    instance = _server(monkeypatch, tmp_path)
    details = instance._path_policy.denial_details("/etc/passwd", write=False)

    async def isaac_tool() -> Dict[str, Any]:
        raise SandboxDenied("/etc/passwd", details)

    def backend_call(session: Any) -> Dict[str, Any]:
        raise SandboxDenied("/etc/passwd", details)

    isaac = _payload(asyncio.run(instance._exec_isaac("isaac_tool", isaac_tool())))
    backend = _payload(
        asyncio.run(
            instance._exec_backend(
                "backend_tool", _Adapter(object()), "USD", _Response, backend_call
            )
        )
    )

    assert isaac == backend
    assert backend["error_type"] == "SandboxError"
    assert backend["details"] == details


def test_isaac_payloads_are_normalised_like_backend_payloads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    instance = _server(monkeypatch, tmp_path)

    async def explicit_failure() -> Dict[str, Any]:
        return {"success": False, "reason": "nothing selected"}

    async def clean() -> Dict[str, Any]:
        return {"prims": []}

    failed = _payload(
        asyncio.run(instance._exec_isaac("isaac_tool", explicit_failure()))
    )
    succeeded = _payload(asyncio.run(instance._exec_isaac("isaac_tool", clean())))

    assert failed == {"success": False, "reason": "nothing selected"}
    assert succeeded == {"prims": [], "success": True}
    # An explicit success=false used to be recorded as a success when it had no error key.
    assert _records(instance, "isaac_tool") == [
        {"success": False, "error": None},
        {"success": True, "error": None},
    ]


def test_a_payload_that_is_not_an_object_is_reported_not_raised(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    instance = _server(monkeypatch, tmp_path)

    async def listing() -> Any:
        return ["/World"]

    isaac = _payload(asyncio.run(instance._exec_isaac("isaac_tool", listing())))
    backend = _payload(
        asyncio.run(
            instance._exec_backend(
                "backend_tool", _Adapter(object()), "Unreal", _Response, lambda s: None
            )
        )
    )

    assert isaac == {
        "success": False,
        "error": "isaac_tool returned list, not a JSON object",
        "error_type": "TypeError",
        "details": None,
    }
    assert backend["error"] == "backend_tool returned NoneType, not a JSON object"
    assert (
        len(_records(instance, "isaac_tool"))
        == len(_records(instance, "backend_tool"))
        == 1
    )


# ---------------------------------------------------------------------------
# Request validation inside the envelope
# ---------------------------------------------------------------------------


def test_request_model_validates_inside_the_envelope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    instance = _server(monkeypatch, tmp_path)
    received: List[Any] = []
    adapter = _Adapter(object())

    def call(session: Any, request: _Request) -> Dict[str, Any]:
        received.append((session, request))
        return {"value": request.name}

    def run(**arguments: Any) -> Dict[str, Any]:
        return _payload(
            asyncio.run(
                instance._exec_backend(
                    "named_tool",
                    adapter,
                    "Blender",
                    _Response,
                    call,
                    request_model=_Request,
                    request_kwargs=arguments,
                )
            )
        )

    invalid = run(name="")
    valid = run(name="Cube")

    assert invalid["error_type"] == "ValidationError"
    assert invalid["details"] == {"input": {"name": ""}}
    assert valid == {"success": True, "error": None, "value": "Cube"}
    assert adapter.sessions_opened == 1, "invalid input never opens a session"
    assert received == [(adapter.session, _Request(name="Cube"))]
    assert _records(instance, "named_tool") == [
        {"success": False, "error": invalid["error"]},
        {"success": True, "error": None},
    ]


def test_invalid_input_spends_a_rate_limit_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    instance = _server(monkeypatch, tmp_path, burst_size=1)

    def run(name: str) -> Dict[str, Any]:
        return _payload(
            asyncio.run(
                instance._exec_backend(
                    "named_tool",
                    _Adapter(object()),
                    "Blender",
                    _Response,
                    lambda session, request: {"value": request.name},
                    request_model=_Request,
                    request_kwargs={"name": name},
                )
            )
        )

    assert run("")["error_type"] == "ValidationError"
    assert run("Cube")["error_type"] == "RateLimitError"


def test_invalid_input_releases_what_the_caller_prepared(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A call that will not run is discarded, whichever stage stopped it."""
    instance = _server(monkeypatch, tmp_path)
    discarded: List[str] = []
    ran: List[Any] = []

    async def call(request: Any) -> Dict[str, Any]:
        ran.append(request)
        return {}

    payload = _payload(
        asyncio.run(
            instance._run_envelope(
                "named_tool",
                call,
                request_model=_Request,
                request_kwargs={"name": ""},
                discard=lambda: discarded.append("discarded"),
            )
        )
    )

    assert payload["error_type"] == "ValidationError"
    assert discarded == ["discarded"]
    assert ran == []


# ---------------------------------------------------------------------------
# Isaac's admission hook
# ---------------------------------------------------------------------------


def test_a_refused_isaac_call_closes_its_coroutine_unawaited(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    instance = _server(monkeypatch, tmp_path, burst_size=1)
    ran: List[str] = []

    async def tool(name: str) -> Dict[str, Any]:
        ran.append(name)
        return {"success": True}

    first, second = tool("first"), tool("second")
    asyncio.run(instance._exec_isaac("isaac_tool", first))
    refused = _payload(asyncio.run(instance._exec_isaac("isaac_tool", second)))

    assert refused["error_type"] == "RateLimitError"
    assert ran == ["first"]
    assert second.cr_frame is None, "the refused coroutine is closed, not left to warn"


def test_admitted_isaac_calls_hold_the_instance_lock_and_heartbeat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The lock is held for the call, and the binding heartbeat follows it, raise or not."""
    instance = _server(monkeypatch, tmp_path)
    lock = instance._get_instance_lock(instance._get_effective_instance_name())
    beats: List[str] = []
    monkeypatch.setattr(instance, "_heartbeat_active_binding", beats.append)
    held: List[bool] = []

    async def ok() -> Dict[str, Any]:
        held.append(lock.locked())
        return {"success": True}

    async def broken() -> Dict[str, Any]:
        held.append(lock.locked())
        raise RuntimeError("bridge dropped the frame")

    asyncio.run(instance._exec_isaac("isaac_ok", ok()))
    asyncio.run(instance._exec_isaac("isaac_broken", broken()))

    assert held == [True, True]
    assert not lock.locked()
    assert beats == ["isaac_ok", "isaac_broken"]


def test_lock_bypassing_isaac_tools_are_enveloped_too(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """ping_isaac and interrupt_isaac_script skip the lock, not the envelope.

    They used to check the rate limit and nothing else: no usage record, and an
    exception escaped to the MCP client unwrapped.
    """
    instance = _server(monkeypatch, tmp_path)

    async def reachable() -> bool:
        return True

    async def dropped() -> Dict[str, Any]:
        # Not one of the transport errors the tool turns into an envelope itself.
        raise RuntimeError("bridge reply carried no status")

    monkeypatch.setattr(instance.client, "ping", reachable)
    monkeypatch.setattr(instance.client, "interrupt_bridge_script", dropped)
    monkeypatch.setattr(
        type(instance.client), "bridge_enabled", property(lambda self: True)
    )

    async def scenario() -> List[Dict[str, Any]]:
        lock = instance._get_instance_lock(instance._get_effective_instance_name())
        async with lock:  # another call holds the instance
            ping = await asyncio.wait_for(instance.mcp.by_name["ping_isaac"](), 1.0)
            interrupt = await asyncio.wait_for(
                instance.mcp.by_name["interrupt_isaac_script"](), 1.0
            )
        return [_payload(ping), _payload(interrupt)]

    ping, interrupt = asyncio.run(scenario())

    assert ping["success"] is True
    assert interrupt == {
        "success": False,
        "error": "bridge reply carried no status",
        "error_type": "RuntimeError",
        "details": None,
    }
    assert _records(instance, "ping_isaac") == [{"success": True, "error": None}]
    assert _records(instance, "interrupt_isaac_script") == [
        {"success": False, "error": interrupt["error"]}
    ]


# ---------------------------------------------------------------------------
# The helpers both sides of the Blender bridge use
# ---------------------------------------------------------------------------


def test_notes_prefix_the_message_outermost_first() -> None:
    exc = _noted(RuntimeError("Error: no mesh\n"), "inner step", "outer step")

    assert exception_message(exc) == "outer step: inner step: Error: no mesh\n"


def test_an_exception_without_a_message_or_notes_is_named() -> None:
    assert exception_message(RuntimeError()) == "RuntimeError"
    assert (
        exception_message(_noted(RuntimeError(), "Failed to bake")) == "Failed to bake"
    )


def test_only_a_declared_string_type_replaces_the_class_name() -> None:
    declared = SimpleNamespace(error_type="AttachmentStale")
    bogus = SimpleNamespace(error_type=42)

    assert exception_type_name(declared) == "AttachmentStale"  # type: ignore[arg-type]
    assert exception_type_name(bogus) == "SimpleNamespace"  # type: ignore[arg-type]
    assert exception_type_name(ValueError("x")) == "ValueError"
