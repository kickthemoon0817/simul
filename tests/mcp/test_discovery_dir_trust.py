"""The discovery directory is only trusted when it is ours alone."""

from __future__ import annotations

import json
import logging
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, List

import pytest

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "src" / "simul_mcp" / "bridge_ext" / "khemoo.simul.mcp"))

from khemoo.simul.mcp.lifecycle import BridgeServerLifecycle  # noqa: E402
from simul_mcp.config import Settings  # noqa: E402
from simul_mcp.adapters import isaac_runtime as isaac_runtime_module  # noqa: E402
from simul_mcp.mcp import backends as backends_module  # noqa: E402
from simul_mcp.mcp import server as server_module  # noqa: E402
from simul_mcp.utils.discovery import DiscoveryDir  # noqa: E402

DEAD_PID = 2**31 - 1


_WARNING_LOGGERS = (
    "simul_mcp.utils.discovery",
    f"{server_module.SimulMCPServer.__module__}.SimulMCPServer",
)


class _Records(logging.Handler):
    """Collect records straight from the loggers under test.

    Tests elsewhere call ``setup_logging``, whose dictConfig stops propagation
    from ``simul_mcp`` and disables loggers it does not name; after that
    caplog's root handler never sees these warnings, so listen on the emitting
    loggers themselves.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.fixture
def simul_warnings() -> Any:
    handler = _Records()
    loggers = [logging.getLogger(name) for name in _WARNING_LOGGERS]
    saved = [(lg.disabled, lg.level) for lg in loggers]
    for lg in loggers:
        lg.disabled = False
        lg.setLevel(logging.NOTSET)
        lg.addHandler(handler)
    try:
        yield handler.messages
    finally:
        for lg, (disabled, level) in zip(loggers, saved):
            lg.removeHandler(handler)
            lg.disabled = disabled
            lg.setLevel(level)


class FakeFastMCP:
    def __init__(self, name: str, version: str, **kwargs: Any):
        self.tools: List[SimpleNamespace] = []

    def tool(self, name: str, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.tools.append(SimpleNamespace(name=name, func=func, kwargs=kwargs))
            return func

        return decorator

    def resource(self, *args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator

    def add_middleware(self, middleware: Any) -> None:
        return


class FakeSocketClient:
    def __init__(self, **kwargs: Any) -> None:
        self._host = kwargs["host"]
        self._port = kwargs["port"]
        self._bridge_host = kwargs["bridge_host"]
        self._bridge_port = kwargs["bridge_port"]
        self._bridge_configured = True

    async def ping(self) -> bool:
        return True


def _make_server(monkeypatch: pytest.MonkeyPatch, discovery_dir: Path) -> server_module.SimulMCPServer:
    monkeypatch.setattr(server_module, "FastMCP", FakeFastMCP)
    monkeypatch.setattr(server_module, "TaskConfig", None)
    monkeypatch.setattr(backends_module, "is_headless_available", lambda: False)
    monkeypatch.setattr(backends_module, "is_blender_available", lambda: False)
    monkeypatch.setattr(backends_module, "UnrealRuntimeAdapter", None)
    monkeypatch.setattr(isaac_runtime_module, "IsaacSocketClient", FakeSocketClient)
    monkeypatch.setattr(server_module.os, "kill", lambda pid, sig: None)
    return server_module.SimulMCPServer(
        settings=Settings(isaac_sim={"discovery_dir": str(discovery_dir)})
    )


def _write_discovery_file(discovery_dir: Path) -> None:
    (discovery_dir / "simul-mcp-4242.json").write_text(
        json.dumps({"pid": 4242, "host": "127.0.0.1", "port": 9229, "vscode_port": 9226})
    )


# ---------------------------------------------------------------------------
# DiscoveryDir.problem
# ---------------------------------------------------------------------------


def test_private_directory_has_no_problem(tmp_path: Path) -> None:
    tmp_path.chmod(0o700)
    assert DiscoveryDir(tmp_path).problem() is None


def test_missing_directory_has_no_problem(tmp_path: Path) -> None:
    assert DiscoveryDir(tmp_path / "absent").problem() is None


@pytest.mark.parametrize("mode", [0o777, 0o770, 0o702])
def test_directory_writable_by_others_is_a_problem(tmp_path: Path, mode: int) -> None:
    tmp_path.chmod(mode)
    problem = DiscoveryDir(tmp_path).problem()
    assert problem is not None
    assert "writable by other users" in problem


@pytest.mark.skipif(sys.platform == "win32", reason="ownership check is POSIX only")
def test_directory_owned_by_someone_else_is_a_problem(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tmp_path.chmod(0o700)
    real_stat = os.stat

    def foreign_stat(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
        info = real_stat(path, *args, **kwargs)
        values = list(info)
        values[stat.ST_UID] = os.getuid() + 1
        return os.stat_result(values)

    monkeypatch.setattr("simul_mcp.utils.discovery.os.stat", foreign_stat)
    problem = DiscoveryDir(tmp_path).problem()
    assert problem is not None
    assert "owned by uid" in problem


# ---------------------------------------------------------------------------
# Server side: _discover_from_files
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_discovery_refuses_world_writable_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, simul_warnings: List[str]
) -> None:
    _write_discovery_file(tmp_path)
    tmp_path.chmod(0o777)
    instance = _make_server(monkeypatch, tmp_path)
    simul_warnings.clear()

    discovered = await instance._discover_from_files()

    assert discovered == {}
    assert any("Skipping Isaac discovery files" in m and "writable by other users" in m for m in simul_warnings)
    assert (tmp_path / "simul-mcp-4242.json").exists(), "refusing must not delete the file"


@pytest.mark.asyncio
async def test_discovery_reads_private_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_discovery_file(tmp_path)
    tmp_path.chmod(0o700)
    instance = _make_server(monkeypatch, tmp_path)

    discovered = await instance._discover_from_files()

    assert "isaac-9229" in discovered


# ---------------------------------------------------------------------------
# Bridge side: write_discovery_file
# ---------------------------------------------------------------------------


def test_bridge_warns_but_still_writes_into_shared_directory(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    tmp_path.chmod(0o777)
    lifecycle = BridgeServerLifecycle(host="127.0.0.1", port=8229, request_handler=None)
    lifecycle._actual_port = 8229

    with caplog.at_level(logging.WARNING, logger="khemoo.simul.mcp.lifecycle"):
        lifecycle.write_discovery_file(str(tmp_path), pid=1, vscode_port=8226)

    assert (tmp_path / "simul-mcp-1.json").exists()
    assert any("not trustworthy" in record.getMessage() for record in caplog.records)


def test_bridge_is_quiet_on_private_directory(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    tmp_path.chmod(0o700)
    lifecycle = BridgeServerLifecycle(host="127.0.0.1", port=8229, request_handler=None)
    lifecycle._actual_port = 8229

    with caplog.at_level(logging.WARNING, logger="khemoo.simul.mcp.lifecycle"):
        lifecycle.write_discovery_file(str(tmp_path), pid=1)

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


# ---------------------------------------------------------------------------
# Auth token files
# ---------------------------------------------------------------------------


def test_write_auth_token_is_private(tmp_path: Path) -> None:
    token_file = DiscoveryDir(tmp_path / "disc").write_auth_token(os.getpid(), "s3cret")

    assert token_file.name == f"auth-token-{os.getpid()}"
    assert stat.S_IMODE(token_file.stat().st_mode) == 0o600
    assert token_file.read_text() == "s3cret\n"


def test_read_auth_token_prefers_a_live_process_and_prunes_dead_ones(tmp_path: Path) -> None:
    discovery = DiscoveryDir(tmp_path)
    dead = discovery.write_auth_token(DEAD_PID, "stale")
    os.utime(dead, (dead.stat().st_atime + 100, dead.stat().st_mtime + 100))
    live = discovery.write_auth_token(os.getpid(), "fresh")

    assert discovery.read_auth_token() == "fresh"
    assert discovery.prune_stale_auth_tokens() == 1
    assert not dead.exists()
    assert live.exists()


def test_read_auth_token_refuses_shared_directory(tmp_path: Path, simul_warnings: List[str]) -> None:
    discovery = DiscoveryDir(tmp_path)
    discovery.write_auth_token(os.getpid(), "s3cret")
    tmp_path.chmod(0o777)

    assert discovery.read_auth_token() is None
    assert any("Ignoring auth token files" in m for m in simul_warnings)


def test_token_from_launch_log_reads_the_python_server_line(tmp_path: Path) -> None:
    log = tmp_path / "launch.log"
    log.write_text(
        "[Info] [carb] Starting Kit\n"
        "[Info] [isaacsim.code_editor.python_server.extension] Python server authentication token: abc-DEF_123\n"
        "Python server authentication token: abc-DEF_123\n"
    )

    assert DiscoveryDir.token_from_launch_log(log) == "abc-DEF_123"
    assert DiscoveryDir.token_from_launch_log(tmp_path / "missing.log") is None
    (tmp_path / "quiet.log").write_text("no token yet\n")
    assert DiscoveryDir.token_from_launch_log(tmp_path / "quiet.log") is None


def test_settings_pick_up_token_file_unless_env_is_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    DiscoveryDir(tmp_path).write_auth_token(os.getpid(), "from-file")
    monkeypatch.delenv("ISAAC_SIM__SOCKET_AUTH_TOKEN", raising=False)

    picked_up = Settings(isaac_sim={"discovery_dir": str(tmp_path)})
    assert picked_up.isaac_sim.socket_auth_token == "from-file"

    monkeypatch.setenv("ISAAC_SIM__SOCKET_AUTH_TOKEN", "from-env")
    explicit = Settings(isaac_sim={"discovery_dir": str(tmp_path)})
    assert explicit.isaac_sim.socket_auth_token == "from-env"


def test_settings_stay_unauthenticated_without_token_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ISAAC_SIM__SOCKET_AUTH_TOKEN", raising=False)
    assert Settings(isaac_sim={"discovery_dir": str(tmp_path)}).isaac_sim.socket_auth_token is None
