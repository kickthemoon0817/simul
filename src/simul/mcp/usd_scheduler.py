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
  write (``create_prim``, ``update_prim_attributes``, ``delete_prim``, and
  ``load_usd_file`` and ``reload_usd_stage`` when they reread files, since
  a shared layer may be reread in place) alone. Callers queue in arrival
  order, so a waiting write is not starved by a stream of reads;
- a lock per stage, held by every other call bound to that stage. Reads on
  one stage share per-stage state that is not thread-safe, such as its
  bounding-box cache;
- a lock per file, held by a call that writes that file (``save_usd_stage``
  on the stage's own file, ``export_usd_stage`` on its destination), so two
  saves of one file from different stages, or a save and an export to it,
  run one after the other and the second sees the first's write.

A call waits at most ``server.timeout`` seconds for its locks and otherwise
fails with ``StageBusy``, so nothing queues forever behind a slow call. The
one exception is housekeeping (``run_between_writes``, which unloads an
ended MCP session's stages): it runs beside reads, never beside a write,
and waits as long as it takes, since it must not race a load, reload or
edit and must not be dropped.
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

    async def acquire(self, exclusive: bool, timeout: Optional[float]) -> bool:
        """Wait up to ``timeout`` seconds for the lock; None waits as long as it takes.

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
        # Per-stage and per-file locks, each with the number of calls holding
        # or waiting for it; an entry goes once that reaches zero, so ids and
        # paths that name nothing leave nothing behind.
        self._stage_locks: Dict[str, List[Any]] = {}
        self._file_locks: Dict[str, List[Any]] = {}

    @property
    def wait_timeout(self) -> float:
        return self._wait_timeout

    @staticmethod
    def _claim(locks: Dict[str, List[Any]], key: str) -> asyncio.Lock:
        entry = locks.get(key)
        if entry is None:
            entry = locks[key] = [asyncio.Lock(), 0]
        entry[1] += 1
        return entry[0]

    @staticmethod
    def _unclaim(locks: Dict[str, List[Any]], key: str) -> None:
        entry = locks[key]
        entry[1] -= 1
        if entry[1] == 0:
            del locks[key]

    def _claim_stage_lock(self, stage_key: str) -> asyncio.Lock:
        return self._claim(self._stage_locks, stage_key)

    def _drop_stage_lock(self, stage_key: str) -> None:
        self._unclaim(self._stage_locks, stage_key)

    async def run(
        self,
        tool_name: str,
        work: Callable[[], Dict[str, Any]],
        *,
        stage_key: Optional[str] = None,
        writes: bool = False,
        file_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run ``work`` on a worker thread once its locks are held.

        Locks are taken in one order (backend, stage, file) and a call holds
        at most one of each, so two calls cannot wait on each other.

        Args:
            tool_name: The tool, for the busy message and the log.
            work: The synchronous pxr work; returns the payload.
            stage_key: The stage the call is bound to, or None for a call
                that touches no loaded stage.
            writes: Whether the call edits a stage. A write runs alone, so it
                needs no stage or file lock.
            file_key: The file the call writes (``stage_files.file_identity``), or None.
                Calls writing one file run one at a time.

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
        held: List[Tuple[Dict[str, List[Any]], str, asyncio.Lock]] = []

        def release() -> None:
            for locks, key, lock in reversed(held):
                lock.release()
                self._unclaim(locks, key)
            held.clear()
            self._access.release(writes)

        wanted = [] if writes else [
            (self._stage_locks, stage_key, "stage"),
            (self._file_locks, file_key, "file"),
        ]
        for locks, key, kind in wanted:
            if key is None:
                continue
            lock = self._claim(locks, key)
            try:
                acquired = await self._acquire_stage(lock, max(deadline - loop.time(), 0.0))
            except BaseException:
                self._unclaim(locks, key)
                release()
                raise
            if not acquired:
                self._unclaim(locks, key)
                release()
                if kind == "stage":
                    return stage_busy(
                        f"Stage {key!r} is busy with another call (a slow call, or one "
                        f"whose client gave up, is still running); {tool_name} waited "
                        f"{self._wait_timeout:g} s. Retry shortly.",
                        stage_id=key,
                        waited_seconds=self._wait_timeout,
                    )
                return stage_busy(
                    f"Another call is writing the same file; {tool_name} waited "
                    f"{self._wait_timeout:g} s. Retry shortly.",
                    file_key=key,
                    waited_seconds=self._wait_timeout,
                )
            held.append((locks, key, lock))

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
                ", ".join(
                    f"stage {key!r}" if locks is self._stage_locks else key for locks, key, _ in held
                )
                or "the USD backend",
            )
            raise
        finally:
            if future.done():
                release()
            else:
                future.add_done_callback(_release_after(release))

    async def run_between_writes(self, name: str, work: Callable[[], Any]) -> Any:
        """Run ``work`` on a worker thread beside reads but never beside a write, however long that takes.

        For housekeeping that must not race a load, reload or edit (which
        hold the backend exclusively) and must not be refused, such as
        unloading an ended session's stages. It holds the backend shared, as
        a read does: it waits for a write running or queued before it, but
        not for slow reads, so it holds up no other session's reads. It
        never gives up with ``StageBusy``.

        Args:
            name: What runs, for the log.
            work: The synchronous pxr work.

        Returns:
            What ``work`` returned.
        """
        await self._access.acquire(False, None)
        loop = asyncio.get_running_loop()

        def release() -> None:
            self._access.release(False)

        try:
            future = loop.run_in_executor(None, work)
        except BaseException:
            release()
            raise
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            _LOGGER.warning("%s was cancelled; it finishes in the background", name)
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
