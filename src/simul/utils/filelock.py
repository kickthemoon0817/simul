"""A small cross-process lock on a lock file (``flock`` on POSIX, ``msvcrt`` on Windows)."""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

if sys.platform == "win32":  # pragma: no cover - exercised on Windows only
    import msvcrt

    _BUSY: tuple[type[BaseException], ...] = (OSError,)

    def _try_lock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)

    def _unlock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    _BUSY = (BlockingIOError,)

    def _try_lock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


@contextmanager
def exclusive_file_lock(lock_path: Path, timeout: float) -> Iterator[None]:
    """Hold an exclusive lock on ``lock_path`` (created 0600, never removed).

    Args:
        lock_path: The lock file.
        timeout: Seconds to wait for another process to release it.

    Raises:
        OSError: The lock file could not be opened.
        TimeoutError: Another process held the lock for ``timeout`` seconds.
    """
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        deadline = time.monotonic() + timeout
        delay = 0.0005
        while True:
            try:
                _try_lock(fd)
                break
            except _BUSY:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{lock_path} stayed locked for {timeout:g} s") from None
                time.sleep(delay)
                delay = min(delay * 2, 0.01)
        try:
            yield
        finally:
            _unlock(fd)
    finally:
        os.close(fd)
