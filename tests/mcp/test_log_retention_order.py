"""The shared log handler keeps the newest copies and goes back to its lock quickly.

Before (round-2 audit):

* M-3: rotation numbered a new copy with the lowest free suffix. Once pruning
  had deleted ``audit.jsonl.<day>``, the next copy took that name again and,
  sorting first, was the next one pruned: with ``maxBackups=3`` and 400
  records the kept copies held records 60-235 and 352-399, and the newest
  rotated ones (236-351) were deleted. A day past 30 rotations of
  ``audit.jsonl`` kept its oldest copies.
* m-8: a lock wait that timed out after 50 ms (several servers writing at
  once) sent the handler to write without the lock or rotation for 30 s, with
  a warning, however briefly the lock had been busy.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, List

import pytest

from simul import logging as simul_logging


def _handler(path: Path, **options: Any) -> simul_logging.SharedRotatingFileHandler:
    handler = simul_logging.SharedRotatingFileHandler(str(path), **options)
    handler.setFormatter(logging.Formatter("%(message)s"))
    return handler


def _emit(handler: logging.Handler, *messages: str) -> None:
    for message in messages:
        handler.emit(logging.LogRecord("x", logging.INFO, "x.py", 1, message, (), None))


def _kept(directory: Path, name: str) -> List[int]:
    seqs: List[int] = []
    for path in directory.iterdir():
        if path.name.startswith(name) and not path.name.endswith(".lock"):
            seqs += [json.loads(line)["seq"] for line in path.read_text().splitlines()]
    return sorted(seqs)


def test_retention_keeps_the_newest_copies(tmp_path: Path) -> None:
    """The auditor's rot.py: maxBytes=2000, maxBackups=3, 400 records of about 40 bytes."""
    handler = _handler(tmp_path / "audit.jsonl", maxBytes=2000, maxBackups=3)
    _emit(handler, *[json.dumps({"seq": i, "pad": "x" * 10}) for i in range(400)])

    kept = _kept(tmp_path, "audit.jsonl")
    assert kept[-1] == 399
    assert kept == list(range(kept[0], 400)), "no newer record was deleted before an older one"
    assert kept[0] > 150, kept[0]
    copies = sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("audit.jsonl."))
    assert len([name for name in copies if not name.endswith(".lock")]) == 3


def test_a_new_copy_is_numbered_after_every_copy_of_its_day(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    handler = _handler(path, maxBytes=100, maxBackups=2)
    _emit(handler, *[f"row {index:03d} " + "y" * 40 for index in range(20)])

    day = f"{date.today():%Y-%m-%d}"
    names = sorted(p.name for p in tmp_path.iterdir() if p.name.startswith(f"audit.jsonl.{day}"))
    numbers = sorted(int(name.rsplit(".", 1)[1]) for name in names if name != f"audit.jsonl.{day}")
    assert len(numbers) == 2 and numbers[1] == numbers[0] + 1
    assert numbers[0] > 2, "the pruned names were not reused"
    newest = (tmp_path / f"audit.jsonl.{day}.{numbers[1]}").read_text()
    assert "row 018" in newest or "row 017" in newest


@pytest.mark.skipif(sys.platform == "win32", reason="flock")
def test_a_busy_lock_is_tried_again_after_a_short_pause(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import fcntl

    path = tmp_path / "audit.jsonl"
    handler = _handler(path, maxBytes=1000)
    holder = os.open(str(path) + ".lock", os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(holder, fcntl.LOCK_EX)
    try:
        _emit(handler, "while busy")
    finally:
        fcntl.flock(holder, fcntl.LOCK_UN)
        os.close(holder)
    paused = handler._unlocked_until - time.monotonic()
    assert 0 < paused <= handler._CONTENTION_BACKOFF_SECONDS

    time.sleep(handler._CONTENTION_BACKOFF_SECONDS + 0.05)
    _emit(handler, "after")

    assert handler._contended == 0, "the lock was taken again"
    assert path.read_text() == "while busy\nafter\n"
    assert capsys.readouterr().err == ""


@pytest.mark.skipif(sys.platform == "win32", reason="flock")
def test_a_lock_held_through_tries_backs_off_and_warns_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import fcntl

    path = tmp_path / "simul.log"
    handler = _handler(path)
    holder = os.open(str(path) + ".lock", os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(holder, fcntl.LOCK_EX)
    pauses = []
    try:
        for index in range(8):
            handler._unlocked_until = 0.0  # the pause has passed
            started = time.monotonic()
            _emit(handler, f"line {index}")
            assert time.monotonic() - started < 0.5
            pauses.append(round(handler._unlocked_until - time.monotonic()))
    finally:
        fcntl.flock(holder, fcntl.LOCK_UN)
        os.close(holder)

    assert len(path.read_text().splitlines()) == 8
    assert pauses[:7] == [0, 1, 2, 4, 8, 16, 30] and pauses[7] == 30
    err = capsys.readouterr().err
    assert err.count("simul:") == 1 and "held its lock through 3 tries" in err
