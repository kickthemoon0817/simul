"""Script executor for the Simul Isaac bridge.

Scripts run on Kit's main thread, which also drives its asyncio loop. That
shapes everything here:

- A synchronous script that never returns blocks the loop, so no bridge
  request can reach it. The only way to stop it is from another thread, by
  planting ``ScriptInterrupted`` in the executing thread with
  ``PyThreadState_SetAsyncExc``. That works for pure-Python loops; a
  blocking C call (``time.sleep``, a long USD operation) only sees the
  exception once it returns to Python.
- A coroutine script is stepped by hand with ``send()``/``throw()`` and
  ``loop.call_soon``, never inside an asyncio Task. On Python 3.12 awaiting
  Kit's own coroutines from within a Task can raise ``RuntimeError: Cannot
  enter into task`` and leave their futures unresolved; the stock
  ``isaacsim.code_editor.python_server`` extension drives user coroutines
  the same way for that reason. Stepping by hand also makes the coroutine
  interruptible while it waits on a future that never resolves.
"""

from __future__ import annotations

import __future__
import asyncio
import contextlib
import ctypes
import dis
import io
import threading
import time
import traceback
from typing import Any, Callable

try:
    from ast import PyCF_ALLOW_TOP_LEVEL_AWAIT
except ImportError:
    PyCF_ALLOW_TOP_LEVEL_AWAIT = 0


class ScriptInterrupted(KeyboardInterrupt):
    """Raised inside a running script when the bridge stops it.

    Derives from ``KeyboardInterrupt`` so a script's ``except Exception`` does
    not swallow it. ``reason`` is empty when the interpreter instantiated the
    class itself on delivery through ``PyThreadState_SetAsyncExc``; the
    executor fills the message in from the pending request afterwards.
    """

    def __init__(self, reason: str = "") -> None:
        super().__init__(reason)
        self.reason = reason


class _CoroutineDriver:
    """Step a coroutine to completion without wrapping it in a Task.

    Every continuation is scheduled with ``loop.call_soon`` and tagged with
    the generation it was scheduled in. ``interrupt`` bumps the generation,
    so a continuation queued before the interrupt cannot later inject a
    stale value into a coroutine that has moved on to another await.
    """

    def __init__(
        self,
        coro: Any,
        loop: asyncio.AbstractEventLoop,
        on_done: Callable[[Any, BaseException | None], None],
    ) -> None:
        self._coro = coro
        self._loop = loop
        self._on_done = on_done
        self._done = False
        self._generation = 0
        self._awaiting: asyncio.Future | None = None

    @property
    def done(self) -> bool:
        """Whether the coroutine has returned or raised."""
        return self._done

    def start(self) -> None:
        """Schedule the first step."""
        self._schedule()

    def interrupt(self, reason: str) -> None:
        """Throw ``ScriptInterrupted`` into the coroutine at its current await.

        Must run on the loop thread. A future the coroutine was waiting on is
        detached first so its later completion is ignored.
        """
        if self._done:
            return
        self._generation += 1
        if self._awaiting is not None:
            self._awaiting.remove_done_callback(self._on_future_done)
            self._awaiting = None
        self._step(None, ScriptInterrupted(reason), self._generation)

    def _schedule(self, value: Any = None, exc: BaseException | None = None) -> None:
        """Queue one step on the loop, tagged with the current generation."""
        self._loop.call_soon(self._step, value, exc, self._generation)

    def _step(self, value: Any, exc: BaseException | None, generation: int) -> None:
        """Resume the coroutine once and wire up whatever it yields."""
        if self._done or generation != self._generation:
            return
        self._awaiting = None
        try:
            if exc is not None:
                yielded = self._coro.throw(exc)
            else:
                yielded = self._coro.send(value)
        except StopIteration as stop:
            self._complete(stop.value, None)
            return
        except BaseException as error:
            self._complete(None, error)
            return
        if isinstance(yielded, asyncio.Future):
            self._awaiting = yielded
            yielded.add_done_callback(self._on_future_done)
        else:
            # A bare yield (asyncio.sleep(0) and friends): continue next tick.
            self._schedule()

    def _on_future_done(self, future: asyncio.Future) -> None:
        """Continue after the awaited future resolves, unless superseded."""
        if self._done or future is not self._awaiting:
            return
        try:
            result = future.result()
        except BaseException as error:
            self._schedule(exc=error)
        else:
            self._schedule(value=result)

    def _complete(self, result: Any, error: BaseException | None) -> None:
        """Mark the coroutine finished and hand the outcome to the owner."""
        self._done = True
        self._awaiting = None
        self._on_done(result, error)


class ScriptExecutor:
    """Execute Python code inside the Isaac Kit Python scope.

    One script runs at a time; ``state`` reports whether one is in flight and
    since when, and ``interrupt`` stops it from any thread.
    """

    _PHASE_IDLE = "idle"
    _PHASE_SYNC = "sync"
    _PHASE_ASYNC = "async"

    def __init__(self, globals_dict: dict, locals_dict: dict) -> None:
        self._globals = globals_dict
        self._locals = locals_dict
        self._compiler_flags = self._get_compiler_flags()
        self._coroutine_flag = self._get_coroutine_flag()
        self._state_lock = threading.Lock()
        self._phase = self._PHASE_IDLE
        self._started_at: float | None = None
        self._executing_thread_ident: int | None = None
        self._async_exc_sent = False
        self._interrupt_reason: str | None = None
        self._driver: _CoroutineDriver | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def is_running(self) -> bool:
        """Whether a script is currently executing."""
        with self._state_lock:
            return self._phase != self._PHASE_IDLE

    @property
    def state(self) -> dict[str, Any]:
        """Snapshot of the executor for diagnostics.

        Returns:
            ``busy`` (a script is running), ``busy_since`` (epoch seconds or
            None), ``phase`` (idle, sync or async) and ``interrupt_requested``.
        """
        with self._state_lock:
            return {
                "busy": self._phase != self._PHASE_IDLE,
                "busy_since": self._started_at,
                "phase": self._phase,
                "interrupt_requested": self._interrupt_reason is not None,
            }

    async def execute(
        self, source: str, timeout_seconds: float | None = None
    ) -> tuple[str, BaseException | None, str]:
        """Execute a statement or expression and capture stdout.

        Args:
            source: Python source to run in the bridge's globals.
            timeout_seconds: Wall-clock budget for the whole script. When it
                runs out the script is interrupted the same way ``interrupt``
                does; ``None`` or 0 means no limit.

        Returns:
            ``(stdout, exception, traceback)``; ``exception`` is a
            ``ScriptInterrupted`` when the script was stopped early.
        """
        output = io.StringIO()
        exception: BaseException | None = None
        trace = ""
        watchdog: threading.Timer | None = None
        try:
            with contextlib.redirect_stdout(output):
                self._begin(self._PHASE_SYNC)
                if timeout_seconds:
                    watchdog = threading.Timer(
                        timeout_seconds,
                        self.interrupt,
                        args=(f"timed out after {timeout_seconds:g}s",),
                    )
                    watchdog.daemon = True
                    watchdog.start()
                try:
                    code, result = self._run_sync(source)
                finally:
                    self._end_sync()
                if self._coroutine_flag != -1 and bool(code.co_flags & self._coroutine_flag):
                    reason = self._pending_interrupt_reason()
                    if reason is not None:
                        # The interrupt landed as the synchronous part was
                        # finishing; the coroutine it produced must not start.
                        result.close()
                        raise ScriptInterrupted(reason)
                    await self._drive_coroutine(result)
        except ScriptInterrupted as exc:
            exception = self._describe_interrupt(exc)
            trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        except Exception as exc:
            exception = exc
            trace = traceback.format_exc()
        finally:
            if watchdog is not None:
                watchdog.cancel()
            self._finish()
        return output.getvalue(), exception, trace

    def interrupt(self, reason: str = "interrupted by request") -> bool:
        """Stop the running script, if any. Safe to call from any thread.

        A synchronous script gets ``ScriptInterrupted`` planted in its thread
        via ``PyThreadState_SetAsyncExc``; it fires at the next bytecode
        boundary, so a loop written in Python stops at once while a blocking
        C call stops only when it returns. A coroutine script has the
        exception thrown into it at its current await, on the loop thread.

        Args:
            reason: Human-readable cause, reported back to the client.

        Returns:
            True when a script was running and the interrupt was delivered.
        """
        with self._state_lock:
            if self._phase == self._PHASE_IDLE:
                return False
            self._interrupt_reason = reason
            if self._phase == self._PHASE_SYNC:
                if not self._async_exc_sent and self._executing_thread_ident is not None:
                    self._async_exc_sent = self._raise_in_thread(
                        self._executing_thread_ident, ScriptInterrupted
                    )
                return True
            driver, loop = self._driver, self._loop
        if driver is not None and loop is not None:
            # Always via the loop, even from the loop thread: the step then
            # runs from a plain callback, outside whichever handler Task asked
            # for the interrupt, exactly like every other step.
            if threading.get_ident() == self._executing_thread_ident:
                loop.call_soon(driver.interrupt, reason)
            else:
                loop.call_soon_threadsafe(driver.interrupt, reason)
        return True

    def _run_sync(self, source: str) -> tuple[Any, Any]:
        """Compile and run ``source``, as an expression when it parses as one.

        Returns:
            The code object (its flags say whether the result is a coroutine)
            and the value ``eval`` produced.
        """
        try:
            code = compile(
                source,
                "<string>",
                "eval",
                flags=self._compiler_flags,
                dont_inherit=True,
            )
        except SyntaxError:
            code = compile(
                source,
                "<string>",
                "exec",
                flags=self._compiler_flags,
                dont_inherit=True,
            )
        return code, eval(code, self._globals, self._locals)

    async def _drive_coroutine(self, coro: Any) -> Any:
        """Run a script coroutine through the manual driver and await its outcome."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()

        def _on_done(result: Any, error: BaseException | None) -> None:
            if future.done():
                return
            if error is None:
                future.set_result(result)
            else:
                future.set_exception(error)

        driver = _CoroutineDriver(coro, loop, _on_done)
        with self._state_lock:
            self._phase = self._PHASE_ASYNC
            self._driver = driver
            self._loop = loop
        driver.start()
        return await future

    def _begin(self, phase: str) -> None:
        """Record that a script is starting on the current thread."""
        with self._state_lock:
            self._phase = phase
            self._started_at = time.time()
            self._executing_thread_ident = threading.get_ident()
            self._async_exc_sent = False
            self._interrupt_reason = None

    def _end_sync(self) -> None:
        """Leave the synchronous phase and defuse an undelivered async exception.

        An interrupt planted just as ``eval`` returned would otherwise fire
        later, inside the event loop rather than the script. Clearing it here
        under the lock closes that window; a request that did land is still
        recorded in ``_interrupt_reason`` for the coroutine check.
        """
        with self._state_lock:
            if self._async_exc_sent and self._executing_thread_ident is not None:
                self._clear_in_thread(self._executing_thread_ident)
                self._async_exc_sent = False
            self._phase = self._PHASE_IDLE

    def _pending_interrupt_reason(self) -> str | None:
        """Return the reason of an interrupt requested during the sync phase."""
        with self._state_lock:
            return self._interrupt_reason

    def _describe_interrupt(self, exc: ScriptInterrupted) -> ScriptInterrupted:
        """Attach the recorded reason to an interrupt the interpreter raised bare."""
        if exc.reason:
            return exc
        with self._state_lock:
            reason = self._interrupt_reason or "interrupted"
        described = ScriptInterrupted(reason)
        described.__traceback__ = exc.__traceback__
        return described

    def _finish(self) -> None:
        """Reset all execution state once a script has ended."""
        with self._state_lock:
            self._phase = self._PHASE_IDLE
            self._started_at = None
            self._executing_thread_ident = None
            self._async_exc_sent = False
            self._interrupt_reason = None
            self._driver = None
            self._loop = None

    @staticmethod
    def _raise_in_thread(thread_ident: int, exc_type: type[BaseException]) -> bool:
        """Plant ``exc_type`` in another thread with ``PyThreadState_SetAsyncExc``.

        Returns:
            True when exactly one thread state was modified. Never targets
            the calling thread, where the exception would fire immediately in
            the caller instead of the script.
        """
        if thread_ident == threading.get_ident():
            return False
        set_async_exc = ctypes.pythonapi.PyThreadState_SetAsyncExc
        set_async_exc.argtypes = (ctypes.c_ulong, ctypes.py_object)
        set_async_exc.restype = ctypes.c_int
        return set_async_exc(ctypes.c_ulong(thread_ident), exc_type) == 1

    @staticmethod
    def _clear_in_thread(thread_ident: int) -> None:
        """Withdraw a pending async exception; a NULL exception clears it."""
        clear_async_exc = ctypes.PyDLL(None).PyThreadState_SetAsyncExc
        clear_async_exc.argtypes = (ctypes.c_ulong, ctypes.c_void_p)
        clear_async_exc.restype = ctypes.c_int
        clear_async_exc(ctypes.c_ulong(thread_ident), None)

    @staticmethod
    def _get_compiler_flags() -> int:
        """Collect future flags supported by the current interpreter."""
        flags = 0
        for value in globals().values():
            try:
                if isinstance(value, __future__._Feature):
                    flags |= value.compiler_flag
            except BaseException:
                pass
        return flags | PyCF_ALLOW_TOP_LEVEL_AWAIT

    @staticmethod
    def _get_coroutine_flag() -> int:
        """Resolve the compiler flag used for coroutine code objects."""
        for key, value in dis.COMPILER_FLAG_NAMES.items():
            if value == "COROUTINE":
                return key
        return -1
