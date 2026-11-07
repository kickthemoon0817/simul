"""The network transports refuse what a stdio server never sees.

``simul server --transport http`` used to accept any Host and Origin header,
so a web page could reach the tools (``execute_*_script`` included) through
DNS rebinding; it bound ``0.0.0.0`` without a word, and it had no way to
require a credential. These tests pin the guard, the public-bind opt-in and
the bearer token, first through ``resolve_http_security`` and then against a
real server process, the way an MCP client or a browser reaches it.
"""

from __future__ import annotations

import asyncio
import http.client
import json
import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pytest
from typer.testing import CliRunner

import simul
from simul.cli import main as cli_main
from simul.config import ServerConfig, Settings
from simul.mcp import server as server_module
from simul.mcp.http_security import (
    BEARER_CLIENT_ID,
    LOOPBACK_HOSTS,
    HttpSecurityError,
    is_loopback_host,
    read_auth_token,
    resolve_http_security,
)

SRC = str(Path(simul.__file__).resolve().parents[1])
TOKEN = "s3cret-token-for-the-tests-0123456789"
runner = CliRunner()


# ---------------------------------------------------------------------------
# resolve_http_security
# ---------------------------------------------------------------------------
class TestResolve:
    def test_the_default_is_a_loopback_bind_with_the_guard_on(self) -> None:
        security = resolve_http_security(ServerConfig())

        assert security.public is False
        assert security.host_origin_protection is True
        assert security.allowed_hosts == LOOPBACK_HOSTS
        assert security.auth_token is None
        assert security.warnings == ()
        assert security.run_kwargs() == {
            "host_origin_protection": True,
            "allowed_hosts": ["localhost", "127.0.0.1", "::1"],
            "allowed_origins": [],
        }

    @pytest.mark.parametrize("host", ["localhost", "LOCALHOST", "127.0.0.1", "127.0.0.2", "::1", "[::1]"])
    def test_loopback_hosts(self, host: str) -> None:
        assert is_loopback_host(host)

    @pytest.mark.parametrize("host", ["0.0.0.0", "::", "", "192.168.1.5", "myhost.local", "localhost.evil.example"])
    def test_hosts_that_are_not_loopback(self, host: str) -> None:
        assert not is_loopback_host(host)

    @pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.5", "myhost.local"])
    def test_a_public_bind_needs_the_opt_in(self, host: str) -> None:
        with pytest.raises(HttpSecurityError, match="--allow-public"):
            resolve_http_security(ServerConfig(host=host))

    def test_a_public_bind_without_a_token_warns(self) -> None:
        security = resolve_http_security(ServerConfig(host="0.0.0.0", allow_public=True))

        assert security.public is True
        assert len(security.warnings) == 1
        assert "no bearer token" in security.warnings[0]
        assert "execute_*_script" in security.warnings[0]
        # 0.0.0.0 is not a Host header anyone sends; FastMCP adds the address a
        # connection actually reached.
        assert security.allowed_hosts == LOOPBACK_HOSTS

    def test_a_public_bind_with_a_token_does_not_warn(self) -> None:
        security = resolve_http_security(
            ServerConfig(host="192.168.1.5", allow_public=True, auth_token=TOKEN)
        )

        assert security.warnings == ()
        assert security.allowed_hosts == (*LOOPBACK_HOSTS, "192.168.1.5")

    def test_configured_hosts_and_origins_extend_the_allowlists(self) -> None:
        security = resolve_http_security(
            ServerConfig(allowed_hosts=["simul.lan"], allowed_origins=["https://app.example"])
        )

        assert security.run_kwargs() == {
            "host_origin_protection": True,
            "allowed_hosts": ["localhost", "127.0.0.1", "::1", "simul.lan"],
            "allowed_origins": ["https://app.example"],
        }

    def test_switching_the_guard_off_is_reported(self) -> None:
        security = resolve_http_security(ServerConfig(host_origin_protection=False))

        assert security.run_kwargs() == {"host_origin_protection": False}
        assert any("DNS rebinding" in warning for warning in security.warnings)


class TestToken:
    def test_no_token_configured(self) -> None:
        assert read_auth_token(ServerConfig()) is None

    def test_an_inline_token(self) -> None:
        assert read_auth_token(ServerConfig(auth_token=f"  {TOKEN}\n")) == TOKEN

    def test_a_token_file_private_to_its_owner(self, tmp_path: Path) -> None:
        token_file = tmp_path / "token"
        token_file.write_text(f"{TOKEN}\nignored second line\n")
        token_file.chmod(0o600)

        assert read_auth_token(ServerConfig(auth_token_file=str(token_file))) == TOKEN

    @pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
    def test_a_token_file_others_can_read_is_refused(self, tmp_path: Path) -> None:
        token_file = tmp_path / "token"
        token_file.write_text(TOKEN)
        token_file.chmod(0o644)

        with pytest.raises(HttpSecurityError, match="chmod 600"):
            read_auth_token(ServerConfig(auth_token_file=str(token_file)))

    def test_a_missing_token_file_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(HttpSecurityError, match="Cannot read server.auth_token_file"):
            read_auth_token(ServerConfig(auth_token_file=str(tmp_path / "missing")))

    def test_a_short_token_is_refused(self) -> None:
        with pytest.raises(HttpSecurityError, match="at least 16"):
            read_auth_token(ServerConfig(auth_token="short"))

    def test_both_sources_at_once_is_refused(self, tmp_path: Path) -> None:
        token_file = tmp_path / "token"
        token_file.write_text(TOKEN)
        token_file.chmod(0o600)

        with pytest.raises(HttpSecurityError, match="set one"):
            read_auth_token(ServerConfig(auth_token=TOKEN, auth_token_file=str(token_file)))

    def test_the_token_never_shows_in_the_settings_repr(self) -> None:
        server = ServerConfig(auth_token=TOKEN)

        assert TOKEN not in repr(server)
        assert TOKEN not in repr(resolve_http_security(server))

    def test_the_verifier_accepts_only_the_token(self) -> None:
        verifier = resolve_http_security(ServerConfig(auth_token=TOKEN)).build_auth()
        assert verifier is not None

        accepted = asyncio.run(verifier.verify_token(TOKEN))
        assert accepted is not None and accepted.client_id == BEARER_CLIENT_ID
        assert asyncio.run(verifier.verify_token(TOKEN + "x")) is None
        assert asyncio.run(verifier.verify_token("")) is None

    def test_the_settings_read_the_token_from_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SERVER__AUTH_TOKEN", TOKEN)
        monkeypatch.setenv("SERVER__ALLOW_PUBLIC", "true")
        monkeypatch.setenv("SERVER__ALLOWED_HOSTS", '["simul.lan"]')

        server = Settings().server

        assert read_auth_token(server) == TOKEN
        assert server.allow_public is True
        assert server.allowed_hosts == ["simul.lan"]


# ---------------------------------------------------------------------------
# SimulMCPServer.run and the CLI
# ---------------------------------------------------------------------------
class TestRun:
    def test_a_public_bind_is_refused_before_serving(self, fake_fastmcp: Any) -> None:
        instance = server_module.SimulMCPServer(
            settings=Settings(server={"host": "0.0.0.0"}), backends={"usd"}
        )

        with pytest.raises(HttpSecurityError, match="--allow-public"):
            asyncio.run(instance.run("http"))
        assert instance.mcp.run_calls == []

    def test_a_token_installs_the_bearer_verifier(self, fake_fastmcp: Any) -> None:
        instance = server_module.SimulMCPServer(
            settings=Settings(server={"auth_token": TOKEN}), backends={"usd"}
        )

        asyncio.run(instance.run("sse"))

        assert instance.mcp.auth is not None
        assert asyncio.run(instance.mcp.auth.verify_token(TOKEN)) is not None
        assert instance.mcp.run_calls[0]["host_origin_protection"] is True

    def test_stdio_ignores_the_network_settings(self, fake_fastmcp: Any) -> None:
        instance = server_module.SimulMCPServer(
            settings=Settings(server={"host": "0.0.0.0"}), backends={"usd"}
        )

        asyncio.run(instance.run("stdio"))

        assert instance.mcp.run_calls == [{"transport": "stdio", "show_banner": False}]


class TestCli:
    @pytest.fixture
    def captured(self, monkeypatch: pytest.MonkeyPatch) -> List[Settings]:
        seen: List[Settings] = []

        async def _capture(settings: Settings, transport: str, **kwargs: Any) -> None:
            seen.append(settings)

        monkeypatch.setattr(cli_main, "start_mcp_server", _capture)
        return seen

    def test_a_public_bind_exits_2_without_starting(
        self, captured: List[Settings], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SERVER__HOST", "0.0.0.0")

        result = runner.invoke(cli_main.app, ["--json", "server", "--transport", "http", "--backends", "usd"])

        assert result.exit_code == 2, result.output
        envelope = json.loads(result.stdout.strip().splitlines()[-1])
        assert envelope["success"] is False
        assert envelope["error_type"] == "InsecureTransport"
        assert "--allow-public" in envelope["error"]
        assert captured == []

    def test_allow_public_starts_with_a_warning(
        self, captured: List[Settings], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SERVER__HOST", "0.0.0.0")

        result = runner.invoke(
            cli_main.app, ["server", "--transport", "http", "--backends", "usd", "--allow-public"]
        )

        assert result.exit_code == 0, result.output
        assert captured[-1].server.allow_public is True
        assert "WARNING" in result.output and "no bearer token" in result.output
        assert "Host/Origin guard: on" in result.output

    def test_stdio_does_not_check_the_bind(
        self, captured: List[Settings], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SERVER__HOST", "0.0.0.0")

        result = runner.invoke(cli_main.app, ["server", "--backends", "usd"])

        assert result.exit_code == 0, result.output
        assert len(captured) == 1


# ---------------------------------------------------------------------------
# A real server process
# ---------------------------------------------------------------------------
def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextmanager
def _server(tmp_path: Path, transport: str, **env: str) -> Iterator[int]:
    """Run ``simul server --transport <transport> --backends usd`` and yield its port."""
    port = _free_port()
    home = tmp_path / f"home-{transport}-{port}"
    home.mkdir()
    environment = {
        **os.environ,
        "HOME": str(home),
        "PYTHONPATH": os.pathsep.join(filter(None, [SRC, os.environ.get("PYTHONPATH")])),
        "SERVER__HOST": "127.0.0.1",
        "SERVER__PORT": str(port),
        "FASTMCP_SHOW_SERVER_BANNER": "false",
        **env,
    }
    log = home / "server.log"
    with log.open("w") as handle:
        proc = subprocess.Popen(
            [sys.executable, "-m", "simul.cli.main", "server", "--transport", transport, "--backends", "usd"],
            env=environment,
            cwd=str(home),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    try:
        deadline = time.monotonic() + 60
        while True:
            if proc.poll() is not None:
                pytest.fail(f"server exited {proc.returncode}: {log.read_text()[-2000:]}")
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                break
            except OSError:
                if time.monotonic() > deadline:
                    pytest.fail(f"server did not listen: {log.read_text()[-2000:]}")
                time.sleep(0.05)
        yield port
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()


def _post(
    port: int, body: Dict[str, Any], *, host: str, origin: Optional[str] = None,
    token: Optional[str] = None, session: Optional[str] = None,
) -> Tuple[int, str, Optional[str]]:
    """POST one JSON-RPC message to /mcp with exactly the given Host and Origin."""
    headers = {
        "Host": host,
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
        "MCP-Protocol-Version": "2025-11-25",
    }
    if origin:
        headers["Origin"] = origin
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if session:
        headers["Mcp-Session-Id"] = session
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        conn.request("POST", "/mcp", body=json.dumps(body), headers=headers)
        response = conn.getresponse()
        return response.status, response.read().decode("utf-8", "replace"), response.getheader("mcp-session-id")
    finally:
        conn.close()


_INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "probe", "version": "1"}},
}
_INITIALIZED = {"jsonrpc": "2.0", "method": "notifications/initialized"}
_CALL = {
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {"name": "get_tool_usage_stats", "arguments": {}},
}


def _tool_call_status(port: int, *, host: str, origin: Optional[str] = None, token: Optional[str] = None) -> List[int]:
    """Initialize, confirm and call a tool; the statuses up to the first refusal."""
    status, _, session = _post(port, _INITIALIZE, host=host, origin=origin, token=token)
    if status != 200:
        return [status]
    statuses = [status]
    for message in (_INITIALIZED, _CALL):
        status, body, _ = _post(port, message, host=host, origin=origin, token=token, session=session)
        statuses.append(status)
    assert '"result"' in body, body
    return statuses


def test_streamable_http_refuses_a_rebound_host_and_a_foreign_origin(tmp_path: Path) -> None:
    with _server(tmp_path, "http") as port:
        # DNS rebinding: the page's own host name now points at 127.0.0.1.
        assert _tool_call_status(port, host=f"evil.example:{port}", origin=f"http://evil.example:{port}") == [421]
        # A cross-origin page talking to 127.0.0.1 directly.
        assert _tool_call_status(port, host=f"127.0.0.1:{port}", origin="http://evil.example") == [403]
        # An MCP client (no Origin) and a same-origin request still get through.
        assert _tool_call_status(port, host=f"127.0.0.1:{port}") == [200, 202, 200]
        assert _tool_call_status(port, host=f"localhost:{port}", origin=f"http://localhost:{port}") == [200, 202, 200]


def _sse_status(port: int, *, host: str, origin: Optional[str] = None) -> int:
    headers = {"Host": host, "Accept": "text/event-stream"}
    if origin:
        headers["Origin"] = origin
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        conn.request("GET", "/sse", headers=headers)
        return conn.getresponse().status
    finally:
        conn.close()


def test_sse_refuses_a_rebound_host_and_a_foreign_origin(tmp_path: Path) -> None:
    with _server(tmp_path, "sse") as port:
        assert _sse_status(port, host=f"evil.example:{port}", origin=f"http://evil.example:{port}") == 421
        assert _sse_status(port, host=f"127.0.0.1:{port}", origin="http://evil.example") == 403
        assert _sse_status(port, host=f"127.0.0.1:{port}") == 200


def test_a_bearer_token_is_required_once_configured(tmp_path: Path) -> None:
    token_file = tmp_path / "token"
    token_file.write_text(TOKEN + "\n")
    token_file.chmod(0o600)

    with _server(tmp_path, "http", SERVER__AUTH_TOKEN_FILE=str(token_file)) as port:
        assert _tool_call_status(port, host=f"127.0.0.1:{port}") == [401]
        assert _tool_call_status(port, host=f"127.0.0.1:{port}", token="wrong-token-0123456789") == [401]
        assert _tool_call_status(port, host=f"127.0.0.1:{port}", token=TOKEN) == [200, 202, 200]
        # The guard still runs first: a token does not let a rebound page in.
        assert _tool_call_status(port, host=f"evil.example:{port}", token=TOKEN) == [421]
