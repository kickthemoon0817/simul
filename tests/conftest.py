"""Suite-wide setup: import paths, the shared FastMCP double and the Isaac platform.

The checkout's ``src`` goes first on ``sys.path`` so the tests exercise the
source they sit next to, whatever ``simul`` is installed in the
interpreter. The bridge extension root follows, for the tests that import the
Kit extension's modules directly, and the repository root last so test
modules can import ``tests.fakes``.

The server registers the Isaac Sim tools by default only on the platforms
Isaac Sim runs on (Linux, Windows). Every test sees such a platform, so the
suite behaves the same on a macOS dev machine as on Linux CI; the tests of
that gate (``tests/mcp/test_isaac_platform_gate.py``) set the platform they
need.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Iterator, Type

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
_BRIDGE_EXT = _SRC / "simul" / "bridge_ext" / "khemoo.simul"

for _path in (_REPO, _BRIDGE_EXT, _SRC):
    if str(_path) in sys.path:
        sys.path.remove(str(_path))
    sys.path.insert(0, str(_path))

from tests.fakes import FakeFastMCP  # noqa: E402


@pytest.fixture(autouse=True, scope="session")
def isaac_platform_supported() -> Iterator[None]:
    """Report an OS Isaac Sim runs on and leave ``isaac_sim.enabled`` unset.

    Patches ``is_isaac_platform_supported`` on ``simul.mcp.backends``, where
    the server reads it, like the runtime probes the tests patch there.
    Session-scoped so module-scoped fixtures that build a server see it too;
    a test's own ``monkeypatch`` still overrides it for that test.
    """
    from simul.mcp import backends

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(
            backends, "is_isaac_platform_supported", lambda platform=None: True
        )
        patcher.delenv("ISAAC_SIM__ENABLED", raising=False)
        yield


@pytest.fixture(autouse=True, scope="session")
def simul_state_outside_home(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Keep the session records and logs the suite writes out of ``~/.simul``.

    The session manager and the usage tracker default to directories under
    ``~/.simul``, and ``setup_logging`` writes the log and audit files
    ``Settings.logging`` names, so every run of the suite added test sessions
    to ``~/.simul/sessions`` and test calls to ``~/.simul/logs``. This points
    them all at one temporary directory: the two module defaults directly, the
    two logging paths through their environment overrides, which the
    subprocesses a test starts inherit too. The settings ``get_settings``
    cached while the suite was imported predate those overrides, so the cache
    is cleared. A test of the documented defaults removes the overrides first.
    """
    from simul import config
    from simul.mcp import session_manager, usage_tracker
    from simul.utils import engine_holds

    state = tmp_path_factory.mktemp("simul-state")
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(session_manager, "_DEFAULT_SESSION_DIR", state / "sessions")
        patcher.setattr(engine_holds, "_DEFAULT_HOLD_DIR", state / "holds")
        patcher.setattr(usage_tracker, "_DEFAULT_LOG_DIR", state / "logs")
        patcher.setenv("LOGGING__FILE_PATH", str(state / "logs" / "simul.log"))
        patcher.setenv("LOGGING__AUDIT_PATH", str(state / "logs" / "audit.jsonl"))
        config._get_cached_settings.cache_clear()
        yield
    config._get_cached_settings.cache_clear()


@pytest.fixture(autouse=True)
def no_hold_records_left() -> Iterator[None]:
    """Remove the engine hold records a test left in the suite's holds directory.

    A server that stops while it holds an engine leaves its record
    ``released`` for the next server to take over (``engine_holds``); a later
    test whose fake engine got the same ephemeral port would take it over and
    find its engine held.
    """
    yield
    from simul.utils import engine_holds

    try:
        entries = list(os.scandir(engine_holds._DEFAULT_HOLD_DIR))
    except OSError:
        return
    for entry in entries:
        try:
            os.unlink(entry.path)
        except OSError:
            pass


@pytest.fixture(autouse=True)
def restore_logging_state() -> Iterator[None]:
    """Put every logger back as the test found it.

    ``setup_logging`` (which ``simul server`` and other commands run) applies
    ``logging.level`` to simul's loggers and turns their propagation off, so a
    test that starts a command with ``--log-level warning`` would otherwise
    hide INFO records from every later test's ``caplog``. Levels, handlers,
    propagation and the disabled flag of the root and of every named logger
    are restored; handlers a test added are closed.
    """

    def all_loggers() -> list:
        named = [
            logger
            for logger in logging.root.manager.loggerDict.values()
            if isinstance(logger, logging.Logger)
        ]
        return [logging.getLogger(), *named]

    saved = {
        id(logger): (logger.level, list(logger.handlers), logger.propagate, logger.disabled)
        for logger in all_loggers()
    }
    yield
    for logger in all_loggers():
        state = saved.get(id(logger))
        kept = state[1] if state else []
        for handler in list(logger.handlers):
            if handler not in kept:
                logger.removeHandler(handler)
                handler.close()
        if state is None:
            logger.setLevel(logging.NOTSET)
            logger.propagate = True
            logger.disabled = False
            continue
        level, handlers, propagate, disabled = state
        for handler in handlers:
            if handler not in logger.handlers:
                logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = propagate
        logger.disabled = disabled


@pytest.fixture
def fake_fastmcp(monkeypatch: pytest.MonkeyPatch) -> Iterator[Type[FakeFastMCP]]:
    """Have the server build a recording ``FakeFastMCP`` instead of a real FastMCP.

    Background task support is switched off with it, since the double does
    not run anything.

    Yields:
        The double's class; the instance the server built is ``server.mcp``.
    """
    from simul.mcp import server as server_module

    monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
    monkeypatch.setattr(server_module, "TaskConfig", None)
    yield FakeFastMCP
