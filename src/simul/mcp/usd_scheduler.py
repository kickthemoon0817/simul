"""Run headless USD calls off the event loop without letting pxr calls race.

USD tools used to run their pxr work inline on the server's event loop. A
19 s ``get_mesh_info`` stalled every other request for its whole duration
(a ``validate_usd_file`` sent 0.5 s in waited 19.6 s), and a request the
client had given up on kept the loop busy anyway.

``USDCallScheduler.run`` executes each call on a worker thread of the event
loop's default executor, so the loop keeps serving other requests. pxr
allows concurrent reads but not a write concurrent with anything that
reads the same layers. Two stages can share a layer (a scene that
references the robot file another stage edits), so the scheduler takes two
kinds of lock before a call starts:

- an access lock for the whole backend: any number of reads at once, or one
  write (``create_prim``, ``update_prim_attributes``, ``delete_prim``) alone.
  Callers queue in arrival order, so a waiting write is not starved by a
  stream of reads;
- a lock per stage, held by every call bound to that stage and by the load
  that (re)registers it. Reads on one stage share per-stage state that is
  not thread-safe, such as its bounding-box cache.

A call waits at most ``server.timeout`` seconds for its locks and otherwise
fails with ``StageBusy``, so nothing queues forever behind a slow call.
Python cannot stop a thread inside pxr. A call whose request is cancelled or
times out on the client keeps running until it finishes, and it holds its
locks until then, so its stage is never touched mid-call. Later calls on
that stage wait up to ``server.timeout`` and then get ``StageBusy``. Calls on
other stages, and file-level calls such as ``validate_usd_file``, proceed
unless a write is queued.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from .schemas.common import ErrorResponse

_LOGGER = logging.getLogger(__name__)


class _AccessLock:
    """Many shared holders or one exclusive holder, granted in arrival order.

    ``release`` is synchronous so a worker thread's completion callback can
    call it from the event loop without awaiting.
    """

    def __init__(self) -> None:
        self._readers = 0
        self._writer = False
        self._waiters: Deque[Tuple[bool, "asyncio.Future[None]"]] = deque()

    def _grantable(self, exclusive: bool) -> bool:
        if exclusive:
            return not self._writer and self._readers == 0
        return not self._writer

    def _take(self, exclusive: bool) -> None:
        if exclusive:
            self._writer = True
        else:
            self._readers += 1

    def _give_back(self, exclusive: bool) -> None:
        if exclusive:
            self._writer = False
        else:
            self._readers -= 1

    def _wake(self) -> None:
        while self._waiters:
            exclusive, waiter = self._waiters[0]
            if waiter.done():
                self._waiters.popleft()
                continue
            if not self._grantable(exclusive):
                return
            self._waiters.popleft()
            self._take(exclusive)
            waiter.set_result(None)
            if exclusive:
                return

    async def acquire(self, exclusive: bool, timeout: float) -> bool:
        """Wait up to ``timeout`` seconds for the lock.

        Returns:
            True once held, False when the wait timed out.
        """
        if not self._waiters and self._grantable(exclusive):
            self._take(exclusive)
            return True
        waiter: "asyncio.Future[None]" = asyncio.get_running_loop().create_future()
        entry = (exclusive, waiter)
        self._waiters.append(entry)
        try:
            await asyncio.wait_for(asyncio.shield(waiter), timeout)
        except BaseException as exc:
            if waiter.done() and not waiter.cancelled():
                # Granted in the same instant the wait ended: hand it back.
                self._give_back(exclusive)
            else:
                waiter.cancel()
                try:
                    self._waiters.remove(entry)
                except ValueError:
                    pass
            self._wake()
            if isinstance(exc, asyncio.TimeoutError):
                return False
            raise
        return True

    def release(self, exclusive: bool) -> None:
        self._give_back(exclusive)
        self._wake()


def stage_busy(message: str, **details: Any) -> Dict[str, Any]:
    """The ``StageBusy`` error envelope."""
    return ErrorResponse(error=message, error_type="StageBusy", details=details or None).model_dump()


class USDCallScheduler:
    """Run USD calls on worker threads, reads concurrently and writes alone.

    Args:
        wait_timeout: Seconds a call may wait for its locks before it fails
            with ``StageBusy`` (``server.timeout``).
    """

    def __init__(self, wait_timeout: float) -> None:
        self._wait_timeout = float(wait_timeout)
        self._access = _AccessLock()
        # Per-stage lock and the number of calls holding or waiting for it;
        # the entry goes once that reaches zero, so ids that name no stage
        # leave nothing behind.
        self._stage_locks: Dict[str, List[Any]] = {}

    @property
    def wait_timeout(self) -> float:
        return self._wait_timeout

    def _claim_stage_lock(self, stage_key: str) -> asyncio.Lock:
        entry = self._stage_locks.get(stage_key)
        if entry is None:
            entry = self._stage_locks[stage_key] = [asyncio.Lock(), 0]
        entry[1] += 1
        return entry[0]

    def _drop_stage_lock(self, stage_key: str) -> None:
        entry = self._stage_locks[stage_key]
        entry[1] -= 1
        if entry[1] == 0:
            del self._stage_locks[stage_key]

    async def run(
        self,
        tool_name: str,
        work: Callable[[], Dict[str, Any]],
        *,
        stage_key: Optional[str] = None,
        writes: bool = False,
    ) -> Dict[str, Any]:
        """Run ``work`` on a worker thread once its locks are held.

        Args:
            tool_name: The tool, for the busy message and the log.
            work: The synchronous pxr work; returns the payload.
            stage_key: The stage the call is bound to, or None for a call
                that touches no loaded stage.
            writes: Whether the call edits a stage. A write runs alone, so it
                needs no stage lock.

        Returns:
            The payload ``work`` returned, or a ``StageBusy`` envelope when the
            locks stayed taken for ``wait_timeout`` seconds.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._wait_timeout
        if not await self._access.acquire(writes, self._wait_timeout):
            return stage_busy(
                f"{tool_name} waited {self._wait_timeout:g} s: "
                + (
                    "a write needs the USD backend alone and other calls are still running"
                    if writes
                    else "a queued edit is waiting for a running call to finish"
                )
                + ". Retry shortly.",
                waited_seconds=self._wait_timeout,
            )
        key = stage_key if stage_key is not None and not writes else None
        stage_lock = self._claim_stage_lock(key) if key is not None else None

        def release() -> None:
            if key is not None and stage_lock is not None:
                stage_lock.release()
                self._drop_stage_lock(key)
            self._access.release(writes)

        if stage_lock is not None and key is not None:
            try:
                acquired = await self._acquire_stage(stage_lock, max(deadline - loop.time(), 0.0))
            except BaseException:
                self._drop_stage_lock(key)
                self._access.release(writes)
                raise
            if not acquired:
                self._drop_stage_lock(key)
                self._access.release(writes)
                return stage_busy(
                    f"Stage {key!r} is busy with another call (a slow call, or one "
                    f"whose client gave up, is still running); {tool_name} waited "
                    f"{self._wait_timeout:g} s. Retry shortly.",
                    stage_id=key,
                    waited_seconds=self._wait_timeout,
                )

        try:
            future = loop.run_in_executor(None, work)
        except BaseException:
            release()
            raise
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            _LOGGER.warning(
                "%s was cancelled; its pxr work finishes in the background and keeps "
                "%s locked until then",
                tool_name,
                f"stage {key!r}" if key else "the USD backend",
            )
            raise
        finally:
            if future.done():
                release()
            else:
                future.add_done_callback(_release_after(release))

    @staticmethod
    async def _acquire_stage(lock: asyncio.Lock, timeout: float) -> bool:
        try:
            await asyncio.wait_for(lock.acquire(), timeout=timeout)
        except asyncio.TimeoutError:
            return False
        return True


def _release_after(release: Callable[[], None]) -> Callable[["asyncio.Future[Any]"], None]:
    """Release the locks of a cancelled call once its worker thread returns."""

    def done(future: "asyncio.Future[Any]") -> None:
        if not future.cancelled() and future.exception() is not None:
            _LOGGER.warning("A cancelled USD call failed in the background: %s", future.exception())
        release()

    return done


__all__ = ["USDCallScheduler", "stage_busy"]
