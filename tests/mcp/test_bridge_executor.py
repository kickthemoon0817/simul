"""Bridge executor: interrupting scripts, per-request timeouts, and coroutine driving.

The bridge runs scripts on Kit's main thread, which also owns its asyncio
loop. A synchronous script that never returns therefore blocks every other
request, and a coroutine awaited inside the handler Task can wedge on Kit
futures under Python 3.12. The executor must stop the former from another
thread and step the latter outside any Task.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

extension_root = (
    Path(__file__).resolve().parents[2]
    / "src" / "simul_mcp" / "bridge_ext" / "khemoo.simul.mcp"
)
sys.path.insert(0, str(extension_root))

from khemoo.simul.mcp.executor import ScriptExecutor, ScriptInterrupted  # noqa: E402
from khemoo.simul.mcp.protocol import BridgeRequest  # noqa: E402
from khemoo.simul.mcp.service import (  # noqa: E402
    LOCK_FREE_ACTIONS,
    READ_ONLY_ACTIONS,
    BridgeCommandService,
)


def _executor() -> ScriptExecutor:
    scope: dict[str, Any] = {}
    return ScriptExecutor(scope, scope)


def _run_in_thread(executor: ScriptExecutor, source: str, **kwargs: Any) -> tuple[threading.Thread, list[Any]]:
    """Run ``executor.execute`` on a worker thread, standing in for Kit's main thread."""
    outcome: list[Any] = []

    def _worker() -> None:
        outcome.append(asyncio.run(executor.execute(source, **kwargs)))

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return thread, outcome


def _wait_until(predicate: Any, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        time.sleep(0.005)


# ---------------------------------------------------------------------------
# Synchronous scripts: interrupt from another thread, or by timeout
# ---------------------------------------------------------------------------


def test_interrupt_stops_a_pure_python_loop_from_another_thread() -> None:
    executor = _executor()
    thread, outcome = _run_in_thread(executor, "while True:\n    pass")
    _wait_until(lambda: executor.is_running)

    assert executor.interrupt("stop requested by test") is True
    thread.join(timeout=5.0)

    assert not thread.is_alive(), "the loop was not interrupted"
    _, exception, trace = outcome[0]
    assert isinstance(exception, ScriptInterrupted)
    assert exception.reason == "stop requested by test"
    assert str(exception) == "stop requested by test"
    assert "ScriptInterrupted" in trace
    assert executor.is_running is False
    assert executor.state == {
        "busy": False,
        "busy_since": None,
        "phase": "idle",
        "interrupt_requested": False,
    }


def test_timeout_interrupts_a_runaway_sync_script() -> None:
    executor = _executor()
    started = time.monotonic()
    thread, outcome = _run_in_thread(executor, "while True:\n    pass", timeout_seconds=0.3)
    thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert time.monotonic() - started < 3.0
    _, exception, _ = outcome[0]
    assert isinstance(exception, ScriptInterrupted)
    assert str(exception) == "timed out after 0.3s"


def test_interrupt_is_a_no_op_when_idle() -> None:
    executor = _executor()
    assert executor.interrupt() is False


def test_state_reports_busy_while_a_script_runs() -> None:
    executor = _executor()
    thread, _ = _run_in_thread(executor, "while True:\n    pass")
    _wait_until(lambda: executor.is_running)

    state = executor.state
    assert state["busy"] is True
    assert state["phase"] == "sync"
    assert isinstance(state["busy_since"], float)
    assert state["busy_since"] <= time.time()

    executor.interrupt()
    thread.join(timeout=5.0)


def test_scripts_that_finish_are_unaffected() -> None:
    executor = _executor()
    output, exception, trace = asyncio.run(
        executor.execute("print('done')", timeout_seconds=5.0)
    )
    assert (output, exception, trace) == ("done\n", None, "")

    output, exception, trace = asyncio.run(executor.execute("1 / 0"))
    assert isinstance(exception, ZeroDivisionError)
    assert "ZeroDivisionError" in trace


def test_a_script_cannot_swallow_the_interrupt_with_except_exception() -> None:
    executor = _executor()
    source = "while True:\n    try:\n        pass\n    except Exception:\n        pass"
    thread, outcome = _run_in_thread(executor, source, timeout_seconds=0.2)
    thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert isinstance(outcome[0][1], ScriptInterrupted)


# ---------------------------------------------------------------------------
# Coroutine scripts: stepped by hand, outside any Task
# ---------------------------------------------------------------------------


def test_coroutine_awaiting_a_future_resolved_by_call_soon_completes() -> None:
    scope: dict[str, Any] = {}
    executor = ScriptExecutor(scope, scope)

    async def scenario() -> tuple[str, BaseException | None, str]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        scope["pending"] = future
        loop.call_soon(future.set_result, 42)
        return await executor.execute("import asyncio\nvalue = await pending\nprint(value)")

    output, exception, _ = asyncio.run(scenario())
    assert exception is None
    assert output == "42\n"


def test_coroutine_steps_run_outside_any_task() -> None:
    """The driver, not a Task, owns the coroutine, so current_task() is None inside it."""
    executor = _executor()

    async def scenario() -> tuple[str, BaseException | None, str]:
        return await executor.execute(
            "import asyncio\n"
            "await asyncio.sleep(0)\n"
            "print(asyncio.current_task() is None)"
        )

    output, exception, _ = asyncio.run(scenario())
    assert exception is None
    assert output == "True\n"


def test_coroutine_exception_is_reported_with_its_traceback() -> None:
    executor = _executor()

    async def scenario() -> tuple[str, BaseException | None, str]:
        return await executor.execute(
            "import asyncio\nawait asyncio.sleep(0)\nraise ValueError('boom')"
        )

    output, exception, trace = asyncio.run(scenario())
    assert isinstance(exception, ValueError)
    assert "boom" in trace


def test_interrupt_reaches_a_coroutine_stuck_on_a_future_that_never_resolves() -> None:
    scope: dict[str, Any] = {}
    executor = ScriptExecutor(scope, scope)

    async def scenario() -> tuple[tuple[str, BaseException | None, str], asyncio.Future, dict[str, Any]]:
        loop = asyncio.get_running_loop()
        never: asyncio.Future = loop.create_future()
        scope["never"] = never
        task = asyncio.ensure_future(
            executor.execute("print('before')\nawait never\nprint('after')")
        )
        while not executor.state["phase"] == "async":
            await asyncio.sleep(0)
        busy_state = executor.state
        assert executor.interrupt("operator asked") is True
        return await task, never, busy_state

    (output, exception, _), never, busy_state = asyncio.run(scenario())
    assert busy_state["busy"] is True
    assert isinstance(exception, ScriptInterrupted)
    assert str(exception) == "operator asked"
    assert output == "before\n"
    assert not never.done()
    assert executor.is_running is False


def test_timeout_covers_the_async_phase_too() -> None:
    scope: dict[str, Any] = {}
    executor = ScriptExecutor(scope, scope)

    async def scenario() -> tuple[str, BaseException | None, str]:
        loop = asyncio.get_running_loop()
        scope["never"] = loop.create_future()
        return await executor.execute("await never", timeout_seconds=0.2)

    started = time.monotonic()
    _, exception, _ = asyncio.run(scenario())
    assert time.monotonic() - started < 3.0
    assert isinstance(exception, ScriptInterrupted)
    assert str(exception) == "timed out after 0.2s"


def test_stale_continuation_does_not_resume_an_interrupted_coroutine() -> None:
    """A future resolved just before the interrupt must not feed the coroutine's next await."""
    scope: dict[str, Any] = {}
    executor = ScriptExecutor(scope, scope)

    async def scenario() -> tuple[str, BaseException | None, str]:
        loop = asyncio.get_running_loop()
        first: asyncio.Future = loop.create_future()
        second: asyncio.Future = loop.create_future()
        scope.update(first=first, second=second)
        source = (
            "try:\n"
            "    await first\n"
            "finally:\n"
            "    print('cleanup', await second)\n"
        )
        task = asyncio.ensure_future(executor.execute(source))
        # Wait until the coroutine is suspended on `first`, not merely scheduled.
        while executor._driver is None or executor._driver._awaiting is not first:
            await asyncio.sleep(0)
        # Resolve `first` (queues a continuation) and interrupt in the same tick.
        first.set_result("stale")
        executor.interrupt("cut")
        await asyncio.sleep(0.01)
        second.set_result("fresh")
        return await task

    output, exception, _ = asyncio.run(scenario())
    assert isinstance(exception, ScriptInterrupted)
    assert output == "cleanup fresh\n"


# ---------------------------------------------------------------------------
# Service: the interrupt action and the busy report
# ---------------------------------------------------------------------------


def test_interrupt_and_runtime_info_bypass_the_request_lock() -> None:
    assert "interrupt" in LOCK_FREE_ACTIONS
    assert "get_runtime_info" in READ_ONLY_ACTIONS
    assert "interrupt" not in READ_ONLY_ACTIONS
    assert "execute_script" not in LOCK_FREE_ACTIONS


def test_capabilities_advertise_interrupt() -> None:
    service = BridgeCommandService(_executor(), allow_unsafe_execution=True)
    assert "interrupt" in service.capabilities["actions"]


def test_interrupt_action_when_idle_reports_nothing_running() -> None:
    service = BridgeCommandService(_executor(), allow_unsafe_execution=True)
    response = asyncio.run(
        service.dispatch(BridgeRequest(request_id="r1", action="interrupt"))
    )
    assert response.status == "ok"
    assert response.payload["interrupted"] is False
    assert response.payload["was_busy"] is False
    assert response.payload["phase"] == "idle"


def test_runtime_info_reports_bridge_activity_shape() -> None:
    service = BridgeCommandService(_executor(), allow_unsafe_execution=True)
    response = asyncio.run(
        service.dispatch(BridgeRequest(request_id="r1", action="get_runtime_info"))
    )
    assert response.status == "ok"
    bridge = response.payload["bridge"]
    assert set(bridge) == {"busy", "busy_since", "busy_for_seconds", "current_action"}
    assert bridge == {
        "busy": False,
        "busy_since": None,
        "busy_for_seconds": None,
        "current_action": None,
    }


def test_runtime_info_and_interrupt_see_a_running_execute_script() -> None:
    scope: dict[str, Any] = {}
    executor = ScriptExecutor(scope, scope)
    service = BridgeCommandService(executor, allow_unsafe_execution=True)

    async def scenario() -> tuple[Any, Any, Any]:
        loop = asyncio.get_running_loop()
        scope["never"] = loop.create_future()
        running = asyncio.ensure_future(
            service.dispatch(
                BridgeRequest(
                    request_id="script",
                    action="execute_script",
                    payload={"code": "await never", "timeout": 30},
                )
            )
        )
        while executor.state["phase"] != "async":
            await asyncio.sleep(0)
        info = await service.dispatch(
            BridgeRequest(request_id="info", action="get_runtime_info")
        )
        interrupt = await service.dispatch(
            BridgeRequest(request_id="stop", action="interrupt")
        )
        return info, interrupt, await running

    info, interrupt, script = asyncio.run(scenario())
    bridge = info.payload["bridge"]
    assert bridge["busy"] is True
    assert bridge["current_action"] == "execute_script"
    assert isinstance(bridge["busy_since"], float)
    assert bridge["busy_for_seconds"] >= 0.0

    assert interrupt.payload["interrupted"] is True
    assert interrupt.payload["was_busy"] is True
    assert interrupt.payload["phase"] == "async"
    assert interrupt.payload["current_action"] == "execute_script"

    assert script.status == "error"
    assert script.error is not None
    assert script.error.name == "ScriptInterrupted"
    assert script.error.message == "interrupted by request"


def test_execute_script_timeout_payload_is_enforced() -> None:
    scope: dict[str, Any] = {}
    executor = ScriptExecutor(scope, scope)
    service = BridgeCommandService(executor, allow_unsafe_execution=True)

    async def scenario() -> Any:
        loop = asyncio.get_running_loop()
        scope["never"] = loop.create_future()
        return await service.dispatch(
            BridgeRequest(
                request_id="script",
                action="execute_script",
                payload={"code": "await never", "timeout": 0.2},
            )
        )

    response = asyncio.run(scenario())
    assert response.status == "error"
    assert response.error is not None
    assert response.error.name == "ScriptInterrupted"
    assert response.error.message == "timed out after 0.2s"
    assert service.activity["busy"] is False


@pytest.mark.parametrize("bad_timeout", ["soon", [1]])
def test_execute_script_rejects_a_non_numeric_timeout(bad_timeout: Any) -> None:
    service = BridgeCommandService(_executor(), allow_unsafe_execution=True)
    response = asyncio.run(
        service.dispatch(
            BridgeRequest(
                request_id="script",
                action="execute_script",
                payload={"code": "print(1)", "timeout": bad_timeout},
            )
        )
    )
    assert response.status == "error"
    assert response.error is not None
    assert response.error.name == "InvalidRequest"
