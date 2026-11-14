"""Where simul keeps its state, and how the server starts and stops.

- State lived under ``Path.home() / ".simul"`` constants that were created
  unconditionally, so with HOME unwritable the server failed to start even
  with both log paths redirected, and even ``simul tools`` created
  ``~/.simul/{logs,sessions}``. ``SIMUL_HOME`` now moves all of it, and each
  directory appears only when something is written there.
- FastMCP's banner checked PyPI on every start (+2 s offline) and cached the
  answer in the platform data directory.
- ``serverInfo.version`` came from stale dist-info (0.1.6 on an editable
  install that is 0.1.11).
- SIGTERM ended the process (-15) without ``SimulMCPServer.shutdown``.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import pytest
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

import simul
import simul.config as config_module
from simul.config import Settings
from simul.mcp import server as server_module
from simul.mcp.session_manager import SessionManager
from simul.mcp.usage_tracker import ToolUsageTracker
from simul.utils.paths import default_state_path, state_dir

SRC = str(Path(simul.__file__).resolve().parents[1])
REPO = Path(simul.__file__).resolve().parents[2]
SAMPLE = str(REPO / "tests" / "data" / "simple_scene.usda")
posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX modes and signals")


@pytest.fixture
def no_state_overrides(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Drop the suite's log-path overrides (tests/conftest.py) and any SIMUL_HOME."""
    for name in ("SIMUL_HOME", "LOGGING__FILE_PATH", "LOGGING__AUDIT_PATH", "CONFIG_FILE"):
        monkeypatch.delenv(name, raising=False)
    config_module._get_cached_settings.cache_clear()
    yield
    config_module._get_cached_settings.cache_clear()


def _tree(root: Path) -> List[str]:
    return sorted(str(path.relative_to(root)) for path in root.rglob("*")) if root.exists() else []


def _subprocess_env(home: Path, **extra: str) -> Dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("SIMUL_HOME", "LOGGING__FILE_PATH", "LOGGING__AUDIT_PATH", "CONFIG_FILE")
    }
    env.update(
        HOME=str(home),
        PYTHONPATH=os.pathsep.join(filter(None, [SRC, os.environ.get("PYTHONPATH")])),
        SECURITY__ALLOWED_PATHS=json.dumps([str(REPO)]),
    )
    env.update(extra)
    return env


# ---------------------------------------------------------------------------
# state_dir and the defaults that follow it
# ---------------------------------------------------------------------------
class TestStateDir:
    def test_the_default_is_dot_simul_in_home(self, no_state_overrides: None) -> None:
        assert state_dir() == Path.home() / ".simul"
        assert default_state_path("logs", "simul.log") == "~/.simul/logs/simul.log"

    def test_simul_home_moves_it(
        self, no_state_overrides: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("SIMUL_HOME", str(tmp_path / "state"))

        assert state_dir() == tmp_path / "state"
        assert default_state_path("logs", "simul.log") == str(tmp_path / "state" / "logs" / "simul.log")

    def test_simul_home_expands_the_home_directory(
        self, no_state_overrides: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SIMUL_HOME", "~/elsewhere")

        assert state_dir() == Path.home() / "elsewhere"

    def test_every_state_default_follows_it(
        self, no_state_overrides: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("SIMUL_HOME", str(tmp_path))

        for settings in (Settings(), Settings.from_yaml(config_module._DEFAULT_CONFIG_FILE)):
            assert settings.logging.file_path == str(tmp_path / "logs" / "simul.log")
            assert settings.logging.audit_path == str(tmp_path / "logs" / "audit.jsonl")
            assert settings.blender.discovery_dir == str(tmp_path / "blender")
            assert settings.blender.attachment_path == str(tmp_path / "blender" / "attachment.json")
            assert settings.unreal.attachment_path == str(tmp_path / "unreal" / "attachment.json")

    def test_without_it_the_defaults_read_as_before(self, no_state_overrides: None) -> None:
        settings = Settings.from_yaml(config_module._DEFAULT_CONFIG_FILE)

        assert settings.logging.file_path == "~/.simul/logs/simul.log"
        assert settings.unreal.attachment_path == "~/.simul/unreal/attachment.json"

    def test_yaml_can_name_it_as_a_placeholder(
        self, no_state_overrides: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("SIMUL_HOME", str(tmp_path))
        config = tmp_path / "config.yaml"
        config.write_text('logging:\n  file:\n    path: "${SIMUL_HOME}/custom/simul.log"\n')

        assert Settings.from_yaml(config).logging.file_path == f"{tmp_path}/custom/simul.log"

    def test_the_usage_log_and_sessions_follow_it(self, tmp_path: Path) -> None:
        """Module defaults are fixed at import, so check them in a fresh interpreter."""
        code = (
            "from simul.mcp import session_manager, usage_tracker; "
            "print(session_manager._DEFAULT_SESSION_DIR); print(usage_tracker._DEFAULT_LOG_DIR)"
        )
        env = _subprocess_env(tmp_path, SIMUL_HOME=str(tmp_path / "state"))
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)

        assert out.stdout.split() == [str(tmp_path / "state" / "sessions"), str(tmp_path / "state" / "logs")]


class TestLazyDirectories:
    def test_the_usage_tracker_creates_its_directory_on_the_first_record(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        tracker = ToolUsageTracker(log_dir=log_dir)
        tracker.reset()
        tracker.get_stats()

        assert not log_dir.exists()

        tracker.record("probe", 1.0, True)

        assert (log_dir / "tool_usage.jsonl").is_file()
        if os.name == "posix":
            assert stat.S_IMODE(log_dir.stat().st_mode) == 0o700

    def test_the_session_manager_creates_its_directory_on_the_first_write(self, tmp_path: Path) -> None:
        session_dir = tmp_path / "sessions"
        manager = SessionManager(session_dir)
        manager.get_all_sessions()
        manager.get_instance_session(8226)

        assert not session_dir.exists()

        manager.get_instance_session(8226).register("agent", "probe")

        assert (session_dir / "8226.json").is_file()

    def test_validate_settings_creates_no_log_directory(self, tmp_path: Path) -> None:
        log_path = tmp_path / "not" / "yet" / "simul.log"
        settings = Settings(logging={"file_path": str(log_path)}, security={"allowed_paths": [str(tmp_path)]})

        assert config_module.validate_settings(settings) == []
        assert not (tmp_path / "not").exists()

    @posix_only
    def test_validate_settings_reports_an_unwritable_log_directory(self, tmp_path: Path) -> None:
        locked = tmp_path / "locked"
        locked.mkdir(mode=0o500)
        try:
            settings = Settings(
                logging={"file_path": str(locked / "logs" / "simul.log")},
                security={"allowed_paths": [str(tmp_path)]},
            )
            errors = config_module.validate_settings(settings)
        finally:
            locked.chmod(0o700)

        assert any("not a writable directory" in error for error in errors)


# ---------------------------------------------------------------------------
# Real processes: nothing written where it should not be
# ---------------------------------------------------------------------------
def test_simul_tools_writes_nothing(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()

    proc = subprocess.run(
        [sys.executable, "-m", "simul.cli.main", "--json", "tools", "--backends", "usd"],
        env=_subprocess_env(home), cwd=str(home), capture_output=True, text=True, timeout=120,
    )

    assert proc.returncode == 0, proc.stderr
    assert _tree(home) == []


async def _call_tools(home: Path, cwd: Path, **env: str) -> Optional[str]:
    """Start a stdio server, call two tools, and return the serverInfo version."""
    transport = StdioTransport(
        command=sys.executable,
        args=["-m", "simul.cli.main", "server", "--backends", "usd"],
        env=_subprocess_env(home, **env),
        cwd=str(cwd),
        keep_alive=False,
    )
    async with Client(transport, timeout=60, mode="legacy") as client:
        await client.call_tool("get_tool_usage_stats", {})
        result = await client.call_tool("validate_usd_file", {"file_path": SAMPLE}, raise_on_error=False)
        assert '"SandboxError"' not in result.content[-1].text
        info = client.initialize_result
        return info.server_info.version if info is not None else None


@posix_only
def test_the_server_runs_with_home_unwritable_when_simul_home_points_elsewhere(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o555)
    state = tmp_path / "state"
    try:
        version = asyncio.run(_call_tools(home, tmp_path, SIMUL_HOME=str(state)))
    finally:
        home.chmod(0o755)

    assert _tree(home) == []
    files = _tree(state)
    assert "logs/simul.log" in files and "logs/audit.jsonl" in files and "logs/tool_usage.jsonl" in files
    # m10: the version MCP clients see is the source's, not the dist-info's.
    assert version == simul.__version__


@posix_only
def test_the_server_runs_with_home_unwritable_and_only_the_logs_redirected(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o555)
    logs = tmp_path / "logs"
    try:
        asyncio.run(
            _call_tools(
                home, tmp_path,
                LOGGING__FILE_PATH=str(logs / "simul.log"),
                LOGGING__AUDIT_PATH=str(logs / "audit.jsonl"),
            )
        )
    finally:
        home.chmod(0o755)

    assert "simul.log" in _tree(logs)


def test_no_fastmcp_banner_and_no_update_check_cache(tmp_path: Path) -> None:
    """The banner's PyPI check used to cache its answer under the platform data dir."""
    home = tmp_path / "home"
    home.mkdir()

    asyncio.run(_call_tools(home, tmp_path))

    assert not [path for path in _tree(home) if "fastmcp" in path.lower()]


def test_server_info_version_is_the_source_version(fake_fastmcp: object) -> None:
    instance = server_module.SimulMCPServer(settings=Settings(), backends={"usd"})

    assert instance.mcp.version == simul.__version__
    assert server_module._PACKAGE_VERSION == simul.__version__


# ---------------------------------------------------------------------------
# m22: SIGTERM shuts the server down
# ---------------------------------------------------------------------------
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


_INITIALIZE = json.dumps(
    {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "signal-test", "version": "0"},
        },
    }
)


def _read_line(stream, timeout: float = 60.0) -> bytes:  # type: ignore[no-untyped-def]
    import select

    ready, _, _ = select.select([stream], [], [], timeout)
    assert ready, "no answer on stdout"
    return stream.readline()


@posix_only
@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT], ids=["SIGTERM", "SIGINT"])
@pytest.mark.parametrize("transport", ["stdio", "http"])
def test_a_signal_runs_shutdown_and_exits_0_with_stdin_held_open(
    tmp_path: Path, transport: str, signum: int
) -> None:
    """Round-2 audit M-5: over stdio the server ignored SIGTERM and SIGINT while stdin stayed open.

    FastMCP reads stdin in a thread that returns only when stdin closes, and
    ``asyncio.run`` waited for it. This test never closes stdin before the
    process exits (the earlier version passed only because ``communicate()``
    closed it), and the stdio server has answered ``initialize`` first.
    """
    log = tmp_path / "simul.log"
    port = _free_port()
    env = _subprocess_env(
        tmp_path,
        SIMUL_HOME=str(tmp_path / "state"),
        LOGGING__FILE_PATH=str(log),
        SERVER__HOST="127.0.0.1",
        SERVER__PORT=str(port),
    )
    proc = subprocess.Popen(
        [sys.executable, "-m", "simul.cli.main", "server", "--transport", transport, "--backends", "usd"],
        env=env, cwd=str(tmp_path), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None and proc.stdout is not None
    try:
        started = f"with {transport} transport"
        _wait_for(lambda: log.exists() and log.read_text().count(started) >= 2 or proc.poll() is not None)
        if transport == "http":
            _wait_for(lambda: _connects(port) or proc.poll() is not None)
        else:
            proc.stdin.write(_INITIALIZE.encode() + b"\n")
            proc.stdin.flush()
            assert json.loads(_read_line(proc.stdout))["id"] == 1
        assert proc.poll() is None, proc.stderr.read().decode() if proc.stderr else ""

        sent = time.monotonic()
        proc.send_signal(signum)
        returncode = proc.wait(timeout=15)
        took = time.monotonic() - sent
        assert not proc.stdin.closed, "stdin stayed open throughout"
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
    err = proc.stderr.read().decode() if proc.stderr else ""

    assert returncode == 0, err
    assert took < 3.0, took
    name = signal.Signals(signum).name
    text = log.read_text()
    assert "Simul 3D MCP Server shut down" in text
    assert f"Stopped on {name}" in text
    assert f"Server stopped ({name})" in err
    assert "Traceback" not in err


@posix_only
def test_a_sigterm_during_startup_exits_0(tmp_path: Path) -> None:
    """A SIGTERM 0.3-0.6 s after spawn, before the loop's handlers, used to exit -15 without shutdown."""
    marker = tmp_path / "startup-reached"
    env = _subprocess_env(tmp_path, SIMUL_HOME=str(tmp_path / "state"), LOGGING__FILE_PATH=str(tmp_path / "simul.log"))
    # Hold the command at the last step before asyncio.run, so the signal
    # lands where the handlers used to be missing.
    script = (
        "import pathlib, sys, time\n"
        "import simul.cli.main as m\n"
        "real = m._keep_fastmcp_home_in_state_dir\n"
        "def slow():\n"
        f"    pathlib.Path({str(marker)!r}).write_text('x')\n"
        "    time.sleep(1.0)\n"
        "    real()\n"
        "m._keep_fastmcp_home_in_state_dir = slow\n"
        "sys.argv = ['simul', 'server', '--backends', 'usd']\n"
        "m.app()\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        env=env, cwd=str(tmp_path), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        _wait_for(lambda: marker.exists() or proc.poll() is not None)
        proc.send_signal(signal.SIGTERM)
        returncode = proc.wait(timeout=15)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
    err = proc.stderr.read().decode() if proc.stderr else ""

    assert returncode == 0, err
    assert "Server stopped (SIGTERM)" in err


def test_a_busy_http_port_exits_1_with_the_json_envelope(tmp_path: Path) -> None:
    """uvicorn exited 3 and --json printed nothing (round-2 audit m-6)."""
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen()
        port = holder.getsockname()[1]
        env = _subprocess_env(
            tmp_path,
            SIMUL_HOME=str(tmp_path / "state"),
            SERVER__HOST="127.0.0.1",
            SERVER__PORT=str(port),
        )
        proc = subprocess.run(
            [sys.executable, "-m", "simul.cli.main", "--json", "server", "--transport", "http", "--backends", "usd"],
            env=env, cwd=str(tmp_path), stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60,
        )

    assert proc.returncode == 1, proc.stderr
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["success"] is False
    assert payload["error_type"] == "AddressInUse"
    assert payload["details"] == {"host": "127.0.0.1", "port": port, "transport": "http"}
    assert f"127.0.0.1:{port}" in payload["error"]


def _connects(port: int) -> bool:
    try:
        socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
        return True
    except OSError:
        return False
