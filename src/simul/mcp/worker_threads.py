"""The calls running on worker threads, so a stopping server knows what it would wait for.

Headless USD calls (``usd_scheduler``) and attached-Blender calls
(``SimulMCPServer._run_backend_call``) run on the event loop's default
executor. A Python thread cannot be stopped, and ``asyncio.run`` waits for
every executor thread before the process can exit, so a stopping server
lasted as long as the slowest such call and swallowed any further SIGTERM
meanwhile (round-3 audit M3-3: 16 s for a ``validate_usd_file``, which may
run 300 s). ``simul server`` reads ``running`` once the server has shut down,
waits for those calls until ``server.shutdown_timeout`` has passed since the
stop began, and then exits without them.
"""

from __future__ import annotations

import asyncio
import itertools
import threading
from typing import Any, Callable, Dict, List

_lock = threading.Lock()
_running: Dict[int, str] = {}
_keys = itertools.count()


def submit(loop: asyncio.AbstractEventLoop, name: str, work: Callable[[], Any]) -> "asyncio.Future[Any]":
    """Run ``work`` on ``loop``'s default executor, listed under ``name`` until it returns.

    Args:
        loop: The running event loop.
        name: What runs (the tool's name), for the log of a stop that does
            not wait for it.
        work: The synchronous work.

    Returns:
        The executor future, as ``loop.run_in_executor`` returns it.
    """
    key = next(_keys)
    with _lock:
        _running[key] = name

    def run() -> Any:
        try:
            return work()
        finally:
            with _lock:
                _running.pop(key, None)

    try:
        return loop.run_in_executor(None, run)
    except BaseException:
        with _lock:
            _running.pop(key, None)
        raise


def running() -> List[str]:
    """The names of the calls submitted and not yet returned, oldest first."""
    with _lock:
        return [_running[key] for key in sorted(_running)]
