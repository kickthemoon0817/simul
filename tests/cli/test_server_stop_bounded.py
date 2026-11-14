"""``simul server`` stops within ``server.shutdown_timeout`` with a worker-thread call running.

Before (round-3 audit M3-3): headless USD and attached-Blender calls run on
worker threads, which nothing can stop, and ``asyncio.run`` waited for them
before the process could exit. Shutdown ran at once, but the process lived on
for as long as the call did (16 s for a ``validate_usd_file`` over HTTP, 21 s
over stdio after the client closed stdin; up to
``USD__VALIDATION_TIMEOUT_SECONDS``, 300 s), and every further SIGTERM or
Ctrl-C was recorded and ignored, so Docker's grace period ended in SIGKILL.

Now the stop waits for such calls only until ``server.shutdown_timeout`` has
passed since it began, then exits 0 without them (the call is recorded with
``executed: "unknown"``), and a second SIGTERM or SIGINT ends the process at
once with ``128 + signal``.

The server under test runs with ``validate_usd_file`` slowed to 30 s; it
touches a file when the call starts.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pytest

import simul

SRC = str(Path(simul.__file__).resolve().parents[1])
REPO = Path(simul.__file__).resolve().parents[2]
SAMPLE = str(REPO / "tests" / "data" / "simple_scene.usda")

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX signals")

#: ``simul server`` with validate_usd_file sleeping SLOW_SECONDS first.
_SLOW_SERVER = """
import os, sys, time
from simul.mcp.registration import _reg_usd
_validate = _reg_usd.run_validation
def _slow(*args, **kwargs):
    open(os.environ["SLOW_STARTED"], "w").close()
    if os.environ.get("SLOW_HOLDS_GIL"):
        # After a second, one C call that keeps the interpreter lock, as pxr
        # opening a large stage does: a regular expression that backtracks
        # for ever.
        time.sleep(1.0)
        import re
        re.match(r"(a|aa)+$", "a" * 64 + "b")
    time.sleep(float(os.environ["SLOW_SECONDS"]))
    return _validate(*args, **kwargs)
_reg_usd.run_validation = _slow
from simul.cli.main import app
sys.argv[0] = "simul"
app()
"""

#: An MCP client that calls validate_usd_file over HTTP and waits.
_HTTP_CALLER = """
import asyncio, sys
from fastmcp import Client
async def main():
    async with Client(sys.argv[1], mode="legacy", timeout=120) as client:
        await client.call_tool("validate_usd_file", {"file_path": sys.argv[2]}, raise_on_error=False)
asyncio.run(main())
"""


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for(predicate, timeout: float = 60.0) -> None:  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("timed out")
        time.sleep(0.05)


def _connects(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            return True
    except OSError:
        return False


class _Server:
    """A ``simul server --backends usd`` subprocess with the slowed validation."""

    def __init__(
        self, tmp_path: Path, transport: str, shutdown_timeout: float, holds_gil: bool = False
    ) -> None:
        self.home = tmp_path
        self.started = tmp_path / "slow-started"
        self.port = _free_port()
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("SERVER__", "SECURITY__", "USD__"))
            and key not in ("SIMUL_HOME", "LOGGING__FILE_PATH", "LOGGING__AUDIT_PATH", "CONFIG_FILE")
        }
        env.update(
            HOME=str(tmp_path),
            SIMUL_HOME=str(tmp_path / "state"),
            PYTHONPATH=os.pathsep.join(filter(None, [SRC, os.environ.get("PYTHONPATH")])),
            SECURITY__ALLOWED_PATHS=json.dumps([str(REPO)]),
            SECURITY__RATE_LIMITING_ENABLED="false",
            SERVER__HOST="127.0.0.1",
            SERVER__PORT=str(self.port),
            SERVER__SHUTDOWN_TIMEOUT=str(shutdown_timeout),
            SLOW_STARTED=str(self.started),
            SLOW_SECONDS="30",
        )
        if holds_gil:
            env["SLOW_HOLDS_GIL"] = "1"
        self.env = env
        self.proc = subprocess.Popen(
            [sys.executable, "-c", _SLOW_SERVER, "server", "--transport", transport, "--backends", "usd"],
            env=env, cwd=str(tmp_path), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        self.log = tmp_path / "state" / "logs" / "simul.log"
        self._caller: Optional[subprocess.Popen] = None
        self._next_id = 1

    def start_http_call(self) -> None:
        _wait_for(lambda: _connects(self.port) or self.proc.poll() is not None)
        assert self.proc.poll() is None, self.stderr()
        self._caller = subprocess.Popen(
            [sys.executable, "-c", _HTTP_CALLER, f"http://127.0.0.1:{self.port}/mcp", SAMPLE],
            env=self.env, cwd=str(self.home), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        _wait_for(self.started.exists)

    def _send(self, message: Dict[str, object]) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(message).encode() + b"\n")
        self.proc.stdin.flush()

    def start_stdio_call(self) -> None:
        self._send({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                       "clientInfo": {"name": "stop-test", "version": "0"}},
        })
        assert self.proc.stdout is not None
        assert json.loads(self.proc.stdout.readline())["id"] == 1
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        self._send({
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "validate_usd_file", "arguments": {"file_path": SAMPLE}},
        })
        _wait_for(lambda: self.started.exists() or self.proc.poll() is not None)
        assert self.started.exists(), self.stderr()

    def close_stdin(self) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.close()

    def wait(self, timeout: float) -> Optional[int]:
        try:
            return self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def stderr(self) -> str:
        if self.proc.poll() is None or self.proc.stderr is None:
            return ""
        return self.proc.stderr.read().decode(errors="replace")

    def usage_rows(self) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for path in sorted((self.home / "state" / "logs").glob("tool_usage.jsonl*")):
            if path.suffix != ".lock":
                rows += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return rows

    def close(self) -> None:
        for proc in (self.proc, self._caller):
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait()


@pytest.fixture
def servers(tmp_path: Path):  # type: ignore[no-untyped-def]
    made: List[_Server] = []

    def make(transport: str, shutdown_timeout: float, holds_gil: bool = False) -> _Server:
        made.append(_Server(tmp_path, transport, shutdown_timeout, holds_gil))
        return made[-1]

    yield make
    for server in made:
        server.close()


def _assert_gave_up_on_the_call(server: _Server) -> None:
    text = server.log.read_text()
    assert "Simul 3D MCP Server shut down" in text
    assert "stopped waiting for validate_usd_file, still running on a worker thread" in text
    (row,) = [row for row in server.usage_rows() if row.get("tool") == "validate_usd_file"]
    assert row["cancelled"] is True and row["executed"] == "unknown"


def test_http_sigterm_exits_at_the_shutdown_timeout_with_a_usd_call_running(
    servers,  # type: ignore[no-untyped-def]
) -> None:
    server = servers("http", 1.5)
    server.start_http_call()

    sent = time.monotonic()
    server.proc.send_signal(signal.SIGTERM)
    returncode = server.wait(timeout=20)
    took = time.monotonic() - sent

    assert returncode == 0, server.stderr()
    # uvicorn's own graceful window (2 s) can outlast the 1.5 s deadline.
    assert took < 5.0, took
    _assert_gave_up_on_the_call(server)
    assert "Server stopped (SIGTERM)" in server.stderr()


@pytest.mark.parametrize("second", [signal.SIGTERM, signal.SIGINT], ids=["SIGTERM", "SIGINT"])
def test_http_a_second_signal_ends_the_stop_at_once(servers, second: int) -> None:  # type: ignore[no-untyped-def]
    server = servers("http", 30.0)
    server.start_http_call()

    server.proc.send_signal(signal.SIGTERM)
    time.sleep(0.5)
    sent = time.monotonic()
    server.proc.send_signal(second)
    returncode = server.wait(timeout=10)
    took = time.monotonic() - sent

    assert returncode == 128 + second, server.stderr()
    assert took < 2.0, took
    assert f"{signal.Signals(second).name} again while stopping" in server.log.read_text()


def test_stdio_closed_stdin_exits_at_the_shutdown_timeout_with_a_usd_call_running(
    servers,  # type: ignore[no-untyped-def]
) -> None:
    """r3_shutdown.py s10: a client closes stdin before it sends SIGTERM, as the spec has it."""
    server = servers("stdio", 1.5)
    server.start_stdio_call()

    closed = time.monotonic()
    server.close_stdin()
    returncode = server.wait(timeout=20)
    took = time.monotonic() - closed

    assert returncode == 0, server.stderr()
    assert took < 4.0, took
    _assert_gave_up_on_the_call(server)


def test_stdio_a_signal_after_stdin_closed_ends_the_wait_at_once(servers) -> None:  # type: ignore[no-untyped-def]
    server = servers("stdio", 30.0)
    server.start_stdio_call()

    server.close_stdin()
    time.sleep(1.0)  # the server has shut down and waits only for the worker thread
    assert server.proc.poll() is None
    sent = time.monotonic()
    server.proc.send_signal(signal.SIGTERM)
    returncode = server.wait(timeout=10)
    took = time.monotonic() - sent

    assert returncode == 0, server.stderr()
    assert took < 1.5, took
    assert "Simul 3D MCP Server shut down" in server.log.read_text()


def test_a_worker_thread_holding_the_interpreter_lock_cannot_keep_the_process_past_the_deadline(
    servers,  # type: ignore[no-untyped-def]
) -> None:
    """pxr inside one long call keeps the interpreter lock: nothing in Python ran past the deadline.

    The round-3 probe's 900k-prim validation overran it by 2 s that way; a
    timer that needs no lock ends the process one second after it. Here the
    worker takes the lock for good a second into the call, after the server
    has shut down and is waiting for it.
    """
    server = servers("stdio", 3.0, holds_gil=True)
    server.start_stdio_call()

    closed = time.monotonic()
    server.close_stdin()
    returncode = server.wait(timeout=20)
    took = time.monotonic() - closed

    assert returncode == 1, server.stderr()
    assert 3.0 <= took < 6.0, took
    assert "Timeout (0:00:0" in server.stderr(), "every thread's traceback is on stderr"
    text = server.log.read_text()
    assert "Shutdown: waiting up to" in text and "validate_usd_file" in text
