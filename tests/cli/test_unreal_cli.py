"""CLI tests for `simul unreal setup` flag wiring and safety gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_path))

from simul_mcp.cli import unreal_cli  # noqa: E402
from simul_mcp.cli.main import app  # noqa: E402


runner = CliRunner()


# ---------------------------------------------------------------------------
# _is_loopback_bind — the predicate the safety gate trusts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "host",
    ["127.0.0.1", "127.0.0.5", "localhost", "LOCALHOST", "::1", " 127.0.0.1 ", ""],
)
def test_is_loopback_bind_recognises_loopback(host: str) -> None:
    assert unreal_cli._is_loopback_bind(host) is True


@pytest.mark.parametrize(
    "host",
    ["0.0.0.0", "::", "192.168.1.10", "10.0.0.5", "example.com"],
)
def test_is_loopback_bind_rejects_non_loopback(host: str) -> None:
    assert unreal_cli._is_loopback_bind(host) is False


# ---------------------------------------------------------------------------
# CLI safety gate — non-loopback bind without --allow-public must refuse.
# ---------------------------------------------------------------------------


def _write_uproject(tmp_path: Path) -> Path:
    p = tmp_path / "Demo.uproject"
    p.write_text(json.dumps({"FileVersion": 3, "EngineAssociation": "5.4"}), encoding="utf-8")
    return p


def test_setup_refuses_public_bind_without_allow_public(tmp_path: Path) -> None:
    """A non-loopback --bind without --allow-public must exit non-zero with
    a structured error and must not touch any project file."""
    uproject = _write_uproject(tmp_path)

    result = runner.invoke(
        app,
        [
            "--json",
            "unreal",
            "setup",
            str(uproject),
            "--bind",
            "0.0.0.0",
            "--no-launch",
            "--yes",
        ],
    )

    # emit_error exits 1 in JSON mode (existing CLI pattern); the contract
    # we care about is that the gate fires non-zero before any IO.
    assert result.exit_code != 0, result.stdout
    assert "0.0.0.0" in result.stdout
    assert "allow-public" in result.stdout
    # Most important: no config file was written — gate fires before any IO.
    assert not (tmp_path / "Config" / "DefaultRemoteControl.ini").exists()


def test_setup_public_bind_with_allow_public_proceeds(
    tmp_path: Path, monkeypatch
) -> None:
    """--bind 0.0.0.0 plus --allow-public passes the gate and writes the
    hostname into DefaultRemoteControl.ini."""
    uproject = _write_uproject(tmp_path)

    async def _fake_poll(session, timeout, interval):
        del session, timeout, interval
        return {"connected": False}

    monkeypatch.setattr(unreal_cli, "_poll_health", _fake_poll)

    result = runner.invoke(
        app,
        [
            "--json",
            "unreal",
            "setup",
            str(uproject),
            "--bind",
            "0.0.0.0",
            "--allow-public",
            "--no-launch",
            "--yes",
        ],
    )

    # exit_code == 1 is expected from the not-connected health stub; the
    # gate (which would have exited before IO) did NOT fire, proving
    # --allow-public lets the public bind through and the patch reached disk.
    assert result.exit_code == 1, result.stdout
    ini = tmp_path / "Config" / "DefaultRemoteControl.ini"
    assert ini.is_file()
    assert "RemoteControlHttpServerHostname=0.0.0.0" in ini.read_text()


def test_setup_loopback_bind_does_not_require_allow_public(
    tmp_path: Path, monkeypatch
) -> None:
    """A loopback --bind is allowed without --allow-public."""
    uproject = _write_uproject(tmp_path)

    async def _fake_poll(session, timeout, interval):
        del session, timeout, interval
        return {"connected": False}

    monkeypatch.setattr(unreal_cli, "_poll_health", _fake_poll)

    result = runner.invoke(
        app,
        [
            "--json",
            "unreal",
            "setup",
            str(uproject),
            "--bind",
            "127.0.0.1",
            "--no-launch",
            "--yes",
        ],
    )

    assert result.exit_code == 1, result.stdout
    ini = tmp_path / "Config" / "DefaultRemoteControl.ini"
    assert ini.is_file()
    assert "RemoteControlHttpServerHostname=127.0.0.1" in ini.read_text()
