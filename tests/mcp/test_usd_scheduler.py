"""The USD call scheduler keeps the event loop free and pxr calls from racing.

USD tools ran their pxr work inline on the event loop: a 19 s get_mesh_info
held up a validate_usd_file sent 0.5 s later for 19.6 s, and kept working
after the client gave up. The scheduler runs the work on worker threads;
reads run concurrently, one at a time per stage; an edit runs alone; a
call waits at most ``wait_timeout`` for its locks and then fails with
StageBusy; a cancelled call keeps its locks until its thread returns.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Callable, Dict, List, Tuple

import pytest

from simul.mcp.usd_scheduler import USDCallScheduler

Interval = Tuple[str, float, float]


class Recorder:
    """Work functions that sleep on their thread and record when they ran."""

    def __init__(self) -> None:
        self.intervals: List[Interval] = []
        self.threads: List[int] = []
        self._lock = threading.Lock()

    def work(self, label: str, seconds: float) -> Callable[[], Dict[str, Any]]:
        def run() -> Dict[str, Any]:
            start = time.monotonic()
            time.sleep(seconds)
            with self._lock:
                self.intervals.append((label, start, time.monotonic()))
                self.threads.append(threading.get_ident())
            return {"success": True, "label": label}

        return run

    def overlaps(self, a: str, b: str) -> bool:
        spans = {label: (start, end) for label, start, end in self.intervals}
        (a0, a1), (b0, b1) = spans[a], spans[b]
        return a0 < b1 and b0 < a1


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_work_runs_off_the_event_loop_thread() -> None:
    recorder = Recorder()

    async def scenario() -> None:
        await USDCallScheduler(5).run("t", recorder.work("a", 0))

    _run(scenario())
    assert recorder.threads[0] != threading.get_ident()


def test_a_slow_read_does_not_hold_up_other_stages_or_file_calls() -> None:
    """The audit's case: a fast call sent 0.5 s into a slow one used to wait it out."""
    recorder = Recorder()

    async def scenario() -> Dict[str, float]:
        scheduler = USDCallScheduler(5)
        slow = asyncio.create_task(scheduler.run("slow", recorder.work("slow", 1.0), stage_key="A"))
        await asyncio.sleep(0.1)
        started = time.monotonic()
        await scheduler.run("validate", recorder.work("validate", 0))
        file_latency = time.monotonic() - started
        started = time.monotonic()
        await scheduler.run("other", recorder.work("other", 0), stage_key="B")
        other_latency = time.monotonic() - started
        await slow
        return {"file": file_latency, "other": other_latency}

    latency = _run(scenario())
    assert latency["file"] < 0.5
    assert latency["other"] < 0.5


def test_reads_on_one_stage_run_one_at_a_time() -> None:
    recorder = Recorder()

    async def scenario() -> None:
        scheduler = USDCallScheduler(5)
        await asyncio.gather(
            scheduler.run("a", recorder.work("a", 0.3), stage_key="S"),
            scheduler.run("b", recorder.work("b", 0.3), stage_key="S"),
        )

    _run(scenario())
    assert not recorder.overlaps("a", "b")


def test_reads_on_different_stages_run_together() -> None:
    recorder = Recorder()

    async def scenario() -> None:
        scheduler = USDCallScheduler(5)
        await asyncio.gather(
            scheduler.run("a", recorder.work("a", 0.3), stage_key="S1"),
            scheduler.run("b", recorder.work("b", 0.3), stage_key="S2"),
        )

    _run(scenario())
    assert recorder.overlaps("a", "b")


def test_a_write_runs_alone() -> None:
    recorder = Recorder()

    async def scenario() -> None:
        scheduler = USDCallScheduler(5)
        read = asyncio.create_task(scheduler.run("r1", recorder.work("r1", 0.3), stage_key="S1"))
        await asyncio.sleep(0.05)
        write = asyncio.create_task(scheduler.run("w", recorder.work("w", 0.3), writes=True))
        await asyncio.sleep(0.05)
        # Queued behind the write, so a stream of reads cannot starve it.
        later = asyncio.create_task(scheduler.run("r2", recorder.work("r2", 0.1), stage_key="S2"))
        await asyncio.gather(read, write, later)

    _run(scenario())
    assert not recorder.overlaps("r1", "w")
    assert not recorder.overlaps("w", "r2")
    order = [label for label, _, _ in sorted(recorder.intervals, key=lambda item: item[1])]
    assert order == ["r1", "w", "r2"]


def test_a_call_that_cannot_get_its_locks_fails_with_stage_busy() -> None:
    recorder = Recorder()

    async def scenario() -> Dict[str, Any]:
        scheduler = USDCallScheduler(0.2)
        slow = asyncio.create_task(scheduler.run("slow", recorder.work("slow", 0.8), stage_key="S"))
        await asyncio.sleep(0.05)
        started = time.monotonic()
        busy = await scheduler.run("get_prim_info", recorder.work("never", 0), stage_key="S")
        busy["_waited"] = time.monotonic() - started
        await slow
        return busy

    busy = _run(scenario())
    assert busy["success"] is False
    assert busy["error_type"] == "StageBusy"
    assert busy["details"]["stage_id"] == "S"
    assert "get_prim_info" in busy["error"]
    assert 0.15 < busy["_waited"] < 0.6
    assert "never" not in [label for label, _, _ in recorder.intervals]


def test_a_write_behind_a_long_read_fails_with_stage_busy() -> None:
    recorder = Recorder()

    async def scenario() -> Dict[str, Any]:
        scheduler = USDCallScheduler(0.2)
        slow = asyncio.create_task(scheduler.run("slow", recorder.work("slow", 0.6), stage_key="S"))
        await asyncio.sleep(0.05)
        busy = await scheduler.run("create_prim", recorder.work("never", 0), writes=True)
        # The refused write left the queue: reads run again at once.
        started = time.monotonic()
        await scheduler.run("read", recorder.work("read", 0), stage_key="T")
        busy["_next"] = time.monotonic() - started
        await slow
        return busy

    busy = _run(scenario())
    assert busy["error_type"] == "StageBusy"
    assert busy["_next"] < 0.3


def test_a_cancelled_call_keeps_its_stage_until_its_thread_returns() -> None:
    recorder = Recorder()

    async def scenario() -> Dict[str, float]:
        scheduler = USDCallScheduler(5)
        task = asyncio.create_task(scheduler.run("slow", recorder.work("slow", 0.6), stage_key="S"))
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        started = time.monotonic()
        await scheduler.run("next", recorder.work("next", 0), stage_key="S")
        waited = time.monotonic() - started
        started = time.monotonic()
        await scheduler.run("after", recorder.work("after", 0), stage_key="S")
        return {"waited": waited, "after": time.monotonic() - started}

    timings = _run(scenario())
    # The stage stayed locked while the cancelled work ran, then was released.
    assert not recorder.overlaps("slow", "next")
    assert timings["waited"] > 0.3
    assert timings["after"] < 0.2


def test_a_call_cancelled_while_waiting_leaves_nothing_held() -> None:
    recorder = Recorder()

    async def scenario() -> float:
        scheduler = USDCallScheduler(5)
        slow = asyncio.create_task(scheduler.run("slow", recorder.work("slow", 0.3), stage_key="S"))
        await asyncio.sleep(0.05)
        waiting = asyncio.create_task(scheduler.run("w", recorder.work("w", 0), writes=True))
        await asyncio.sleep(0.05)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        await slow
        started = time.monotonic()
        await scheduler.run("again", recorder.work("again", 0), writes=True)
        return time.monotonic() - started

    assert _run(scenario()) < 0.2
    assert "w" not in [label for label, _, _ in recorder.intervals]


def test_an_exception_in_the_work_releases_the_locks() -> None:
    def boom() -> Dict[str, Any]:
        raise RuntimeError("pxr said no")

    async def scenario() -> float:
        scheduler = USDCallScheduler(0.3)
        with pytest.raises(RuntimeError):
            await scheduler.run("boom", boom, stage_key="S")
        started = time.monotonic()
        result = await scheduler.run("next", Recorder().work("next", 0), stage_key="S")
        assert result["success"] is True
        return time.monotonic() - started

    assert _run(scenario()) < 0.2


def test_stage_locks_do_not_accumulate() -> None:
    """A client can name any stage id; finished calls leave no lock behind."""
    recorder = Recorder()

    async def scenario() -> USDCallScheduler:
        scheduler = USDCallScheduler(0.2)
        await asyncio.gather(
            *(scheduler.run("q", recorder.work(f"q{i}", 0), stage_key=f"id-{i}") for i in range(50))
        )
        slow = asyncio.create_task(scheduler.run("slow", recorder.work("slow", 0.4), stage_key="S"))
        await asyncio.sleep(0.05)
        await scheduler.run("busy", recorder.work("never", 0), stage_key="S")  # times out
        await slow
        return scheduler

    scheduler = _run(scenario())
    assert scheduler._stage_locks == {}


def test_calls_writing_one_file_run_one_at_a_time() -> None:
    """Two saves of one file from different stages, or a save and an export to it, never overlap."""
    recorder = Recorder()

    async def scenario() -> None:
        scheduler = USDCallScheduler(5)
        await asyncio.gather(
            scheduler.run("save", recorder.work("save", 0.3), stage_key="S1", file_key="/f.usda"),
            scheduler.run("export", recorder.work("export", 0.3), stage_key="S2", file_key="/f.usda"),
            scheduler.run("other", recorder.work("other", 0.3), stage_key="S3", file_key="/g.usda"),
        )

    _run(scenario())
    assert not recorder.overlaps("save", "export")
    assert recorder.overlaps("save", "other")


def test_a_call_waiting_on_a_file_being_written_fails_with_stage_busy() -> None:
    recorder = Recorder()

    async def scenario() -> Tuple[Dict[str, Any], USDCallScheduler]:
        scheduler = USDCallScheduler(0.2)
        slow = asyncio.create_task(
            scheduler.run("export", recorder.work("slow", 0.6), stage_key="S1", file_key="/f.usda")
        )
        await asyncio.sleep(0.05)
        busy = await scheduler.run("save_usd_stage", recorder.work("never", 0), stage_key="S2", file_key="/f.usda")
        # The refused call released its stage at once.
        await scheduler.run("read", recorder.work("read", 0), stage_key="S2")
        await slow
        return busy, scheduler

    busy, scheduler = _run(scenario())
    assert busy["error_type"] == "StageBusy"
    assert busy["details"]["file_key"] == "/f.usda"
    assert "save_usd_stage" in busy["error"]
    assert "never" not in [label for label, _, _ in recorder.intervals]
    assert scheduler._file_locks == {} and scheduler._stage_locks == {}
