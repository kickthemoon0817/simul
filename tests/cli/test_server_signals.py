"""``simul server`` keeps the stop signals from its first line and gives them back.

The subprocess tests with stdin held open are in
``tests/mcp/test_state_dir_and_startup.py``; these check the parts that do
not need a process of their own.
"""

from __future__ import annotations

import asyncio
import signal
from typing import Any, List

import pytest
from typer.testing import CliRunner

from simul.cli import main as cli_main
from simul.cli.main import app
from simul.config import Settings

runner = CliRunner()


def test_a_signal_received_before_the_loop_stops_the_server_before_it_is_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: List[Any] = []

    async def _start(*args: Any, **kwargs: Any) -> None:
        started.append(args)

    monkeypatch.setattr(cli_main, "start_mcp_server", _start)
    recorder = cli_main._StopSignals()
    recorder.received.append(signal.SIGTERM)

    stopped = asyncio.run(cli_main._serve(Settings(), "stdio", {"usd"}, recorder))

    assert stopped == signal.SIGTERM
    assert started == []


def test_the_server_hands_itself_over_and_a_clean_stop_reports_no_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: List[Any] = []

    async def _start(settings: Settings, transport: str, **kwargs: Any) -> None:
        kwargs["on_server"]("the server")
        seen.append(transport)

    monkeypatch.setattr(cli_main, "start_mcp_server", _start)

    assert asyncio.run(cli_main._serve(Settings(), "stdio", {"usd"})) is None
    assert seen == ["stdio"]


def test_the_server_command_restores_the_signal_handlers(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _start(*args: Any, **kwargs: Any) -> None:
        # While the command runs, a stop signal is the command's to handle.
        for signum in (signal.SIGTERM, signal.SIGINT):
            assert signal.getsignal(signum) not in (signal.SIG_DFL, signal.default_int_handler)

    monkeypatch.setattr(cli_main, "start_mcp_server", _start)
    monkeypatch.setattr(cli_main, "_is_isaac_reachable", lambda *a, **k: False)
    before = {signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)}

    result = runner.invoke(app, ["server", "--backends", "usd"])

    assert result.exit_code == 0, result.output
    assert {signum: signal.getsignal(signum) for signum in before} == before
