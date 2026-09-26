"""CLI tests for `simul unreal setup` flag wiring and safety gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest
from typer.testing import CliRunner

from simul_mcp.cli import unreal_cli
from simul_mcp.cli.main import app

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
    # WS bind in RC ini (real URemoteControlSettings field).
    assert "RemoteControlWebsocketServerBindAddress=0.0.0.0" in ini.read_text()
    # HTTP bind in DefaultEngine.ini (per UE 5.x source — see iter6).
    engine_ini = tmp_path / "Config" / "DefaultEngine.ini"
    assert engine_ini.is_file()
    text = engine_ini.read_text()
    assert "[HTTPServer.Listeners]" in text
    assert "DefaultBindAddress=0.0.0.0" in text


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
    assert "RemoteControlWebsocketServerBindAddress=127.0.0.1" in ini.read_text()
    engine_ini = tmp_path / "Config" / "DefaultEngine.ini"
    assert engine_ini.is_file()
    assert "DefaultBindAddress=127.0.0.1" in engine_ini.read_text()


# ---------------------------------------------------------------------------
# --passphrase: setup writes the MD5 hash to ini; CLI gates loopback abuse.
# ---------------------------------------------------------------------------


def test_setup_refuses_passphrase_with_loopback_bind(tmp_path: Path) -> None:
    """--passphrase only adds value with a non-loopback bind. The default
    loopback bind already blocks remote access via the IP allowlist, so
    enabling the passphrase would only break clients (no security gain).
    The CLI refuses this combination before any IO."""
    uproject = _write_uproject(tmp_path)

    result = runner.invoke(
        app,
        [
            "--json",
            "unreal",
            "setup",
            str(uproject),
            "--passphrase",
            "secret",
            "--no-launch",
            "--yes",
        ],
    )

    assert result.exit_code != 0, result.stdout
    assert "passphrase" in result.stdout.lower()
    assert "loopback" in result.stdout.lower() or "bind" in result.stdout.lower()
    # Gate fires before IO — no ini patch.
    assert not (tmp_path / "Config" / "DefaultRemoteControl.ini").exists()


def test_setup_passphrase_with_public_bind_writes_md5_hash(
    tmp_path: Path, monkeypatch
) -> None:
    """--passphrase plus --bind 0.0.0.0 plus --allow-public writes the MD5
    hash and bEnforcePassphraseForRemoteClients=True to the ini.

    UE 5.x's FMD5::HashAnsiString hashes the ASCII bytes; lowercase hex.
    For the literal 'password', the hash is the well-known
    5f4dcc3b5aa765d61d8327deb882cf99.
    """
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
            "--passphrase",
            "password",
            "--no-launch",
            "--yes",
        ],
    )

    # exit 1 is from the not-connected health stub; the gate did NOT fire,
    # so the ini patch reached disk.
    assert result.exit_code == 1, result.stdout
    ini = tmp_path / "Config" / "DefaultRemoteControl.ini"
    assert ini.is_file()
    text = ini.read_text()
    assert "bEnforcePassphraseForRemoteClients=True" in text
    assert (
        '+Passphrases=(Identifier="simul",Passphrase='
        '"5f4dcc3b5aa765d61d8327deb882cf99")'
    ) in text
    # The JSON payload signals passphrase enablement (without leaking the
    # hash itself — it lives in the ini, not the response).
    payload = json.loads(result.stdout)
    assert payload["passphrase_enabled"] is True


def test_session_factory_default_does_not_inject_passphrase(monkeypatch) -> None:
    """Backward compat: every other `_session()` caller passes no
    passphrase. The factory's `if passphrase is not None` guard must keep
    those sessions header-free regardless of any env var leaking from the
    test runner. Locks the guard so a refactor can't silently include
    Passphrase in headers for callers that don't ask for it."""
    monkeypatch.delenv("UNREAL__PASSPHRASE", raising=False)
    session = unreal_cli._session(host="127.0.0.1", port=30010)
    assert "Passphrase" not in session._default_headers()
    assert session._passphrase_md5 is None


def test_setup_payload_includes_engine_ini_when_bind_supplied(
    tmp_path: Path, monkeypatch
) -> None:
    """Pinning regression for the iter6 verifier finding: the CLI's JSON
    payload must surface `patched.engine_ini` so callers can tell that
    the HTTP bind was actually written to Config/DefaultEngine.ini. The
    SetupResult dataclass populates it whenever --bind is supplied; the
    payload constructor in iter6 dropped it silently."""
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

    assert result.exit_code == 1, result.stdout  # not-connected stub
    payload = json.loads(result.stdout)
    assert "engine_ini" in payload["patched"]
    engine_ini = payload["patched"]["engine_ini"]
    assert engine_ini is not None
    assert engine_ini["changed"] is True
    assert "DefaultBindAddress" in engine_ini["added"]


def test_setup_payload_engine_ini_is_none_when_no_bind(
    tmp_path: Path, monkeypatch
) -> None:
    """When --bind is not supplied, no Config/DefaultEngine.ini is
    written. The payload still surfaces the `engine_ini` key so callers
    don't have to guess whether it was forgotten or intentionally skipped
    — but its value is None."""
    uproject = _write_uproject(tmp_path)

    async def _fake_poll(session, timeout, interval):
        del session, timeout, interval
        return {"connected": False}

    monkeypatch.setattr(unreal_cli, "_poll_health", _fake_poll)

    result = runner.invoke(
        app,
        ["--json", "unreal", "setup", str(uproject), "--no-launch", "--yes"],
    )

    assert result.exit_code == 1, result.stdout
    payload = json.loads(result.stdout)
    assert "engine_ini" in payload["patched"]
    assert payload["patched"]["engine_ini"] is None
    # And no DefaultEngine.ini should have been created on disk either.
    assert not (tmp_path / "Config" / "DefaultEngine.ini").exists()


def test_setup_polling_session_carries_passphrase_header(
    tmp_path: Path, monkeypatch
) -> None:
    """Pinning regression for the iter4b finding: the setup poller used to
    build its own UnrealRuntimeSession without forwarding --passphrase, so
    a passphrase-enabled editor returned 401 on /remote/info and the
    setup CLI timed out at --wait-timeout despite the editor being
    healthy. Confirm the polling session now has the configured passphrase
    so its _default_headers() includes the Passphrase header."""
    uproject = _write_uproject(tmp_path)
    captured_session: Dict[str, Any] = {}

    async def _fake_poll(session, timeout, interval):
        del timeout, interval
        captured_session["session"] = session
        return {"connected": True, "engine_version": "5.3.2", "project_name": ""}

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
            "--passphrase",
            "secret",
            "--no-launch",
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.stdout
    session = captured_session["session"]
    # The polling session must carry the MD5 of "secret" so its outgoing
    # /remote/info request authenticates against a passphrase-enabled
    # editor.
    expected = "5ebe2294ecd0e0f08eab7690d2a6ee69"  # md5("secret")
    assert session._passphrase_md5 == expected
    assert session._default_headers().get("Passphrase") == expected


def test_setup_rejects_non_ascii_passphrase_with_actionable_message(
    tmp_path: Path,
) -> None:
    """UE's FMD5::HashAnsiString narrows wide chars before hashing, so a
    non-ASCII passphrase would silently mismatch on UE's side. The CLI
    must catch this at the encode step and refuse with a user-facing
    message — not raise a raw UnicodeEncodeError stack trace."""
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
            "--allow-public",
            "--passphrase",
            "café",  # non-ASCII
            "--no-launch",
            "--yes",
        ],
    )

    assert result.exit_code != 0, result.stdout
    assert "ascii" in result.stdout.lower()
    # Should NOT be a raw exception traceback.
    assert "Traceback" not in result.stdout
    assert not (tmp_path / "Config" / "DefaultRemoteControl.ini").exists()


# ---------------------------------------------------------------------------
# `simul unreal exec` — trust UE's ReturnValue, render LogOutput as text.
# Regression for the bug surfaced during the issue #44 live test, where a
# plain `print("hello")` script returned exit 1 with the misleading message
# "No JSON output from Python execution" even though UE actually ran the
# code successfully.
# ---------------------------------------------------------------------------


def _stub_session_factory(monkeypatch, raw_result):
    """Replace unreal_cli._session with a stub returning the given raw_result.

    The stub's _execute_python is an async function so the CLI's
    asyncio.run(...) call works unchanged.
    """

    class _StubSession:
        settings = unreal_cli.get_settings()

        async def _execute_python(self, code, mode):  # noqa: D401
            del code, mode
            return raw_result

    monkeypatch.setattr(unreal_cli, "_session", lambda *a, **kw: _StubSession())


def test_exec_refuses_when_script_execution_disabled(monkeypatch):
    from unittest.mock import AsyncMock

    from simul_mcp.adapters.unreal_runtime import UnrealRuntimeSession

    base = unreal_cli.get_settings()
    settings = base.model_copy(
        update={
            "security": base.security.model_copy(
                update={"allow_script_execution": False}
            )
        }
    )
    session = UnrealRuntimeSession(settings)
    session._execute_python = AsyncMock()
    monkeypatch.setattr(unreal_cli, "_session", lambda *a, **kw: session)
    result = runner.invoke(app, ["--json", "unreal", "exec", "print('must not run')"])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["error_type"] == "ScriptExecutionDisabled"
    session._execute_python.assert_not_called()


def test_exec_plain_print_succeeds_and_renders_output(monkeypatch) -> None:
    """`exec "print('hello')"` must report success and show 'hello'.

    The script returns no JSON object — pre-fix this returned exit 1 with
    "No JSON output from Python execution" because the CLI ran the result
    through _parse_python_json (which is for internal callers).
    """
    _stub_session_factory(
        monkeypatch,
        {
            "ReturnValue": True,
            "LogOutput": [{"Type": "Info", "Output": "hello\n"}],
            "CommandResult": "",
        },
    )

    result = runner.invoke(app, ["--json", "unreal", "exec", "print('hello')"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["success"] is True
    assert payload["return_value"] is True
    assert payload["output"] == "hello\n"
    assert "error" not in payload


def test_exec_failed_python_reports_error_with_command_result(monkeypatch) -> None:
    """A Python error path: ReturnValue=False, CommandResult holds the message."""
    _stub_session_factory(
        monkeypatch,
        {
            "ReturnValue": False,
            "LogOutput": [],
            "CommandResult": "NameError: name 'undef' is not defined",
        },
    )

    result = runner.invoke(app, ["--json", "unreal", "exec", "undef"])

    assert result.exit_code == 1, result.stdout
    payload = json.loads(result.stdout)
    assert payload["success"] is False
    assert "NameError" in payload["error"]


def test_exec_json_printing_script_still_works(monkeypatch) -> None:
    """A script that prints a JSON literal must NOT be specially extracted —
    the JSON object goes into `output` as plain text, not parsed-and-replaced.
    Future-proofs against re-introducing the _parse_python_json regression.
    """
    _stub_session_factory(
        monkeypatch,
        {
            "ReturnValue": True,
            "LogOutput": [{"Type": "Info", "Output": '{"x": 1}\n'}],
            "CommandResult": "",
        },
    )

    result = runner.invoke(app, ["--json", "unreal", "exec", "print('{...}')"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["success"] is True
    assert payload["output"] == '{"x": 1}\n'


def test_exec_concatenates_multi_line_log_output(monkeypatch) -> None:
    """Multiple Info log lines join in order; Warning/Error entries route
    to their own streams instead of being silently dropped."""
    _stub_session_factory(
        monkeypatch,
        {
            "ReturnValue": True,
            "LogOutput": [
                {"Type": "Info", "Output": "line 1\n"},
                {"Type": "Warning", "Output": "deprecated path\n"},
                {"Type": "Info", "Output": "line 2\n"},
                {"Type": "Error", "Output": "non-fatal err\n"},
            ],
            "CommandResult": "",
        },
    )

    result = runner.invoke(app, ["--json", "unreal", "exec", "print('multiline')"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["output"] == "line 1\nline 2\n"
    assert payload["warnings"] == "deprecated path\n"
    assert payload["errors"] == "non-fatal err\n"


def test_exec_warnings_and_errors_surface_separately(monkeypatch) -> None:
    """unreal.log_warning and unreal.log_error must not be silently swallowed."""
    _stub_session_factory(
        monkeypatch,
        {
            "ReturnValue": True,
            "LogOutput": [
                {"Type": "Warning", "Output": "watch out\n"},
                {"Type": "Error", "Output": "but kept going\n"},
            ],
            "CommandResult": "",
        },
    )

    result = runner.invoke(
        app, ["--json", "unreal", "exec", "unreal.log_warning('w'); unreal.log_error('e')"]
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["success"] is True
    assert payload["output"] == ""
    assert payload["warnings"] == "watch out\n"
    assert payload["errors"] == "but kept going\n"


def test_exec_returnvalue_missing_treated_as_failure(monkeypatch) -> None:
    """Defensive: if UE returns a payload without ReturnValue, fail safely
    rather than silently reporting success."""
    _stub_session_factory(
        monkeypatch,
        {"LogOutput": [], "CommandResult": "weird payload"},
    )

    result = runner.invoke(app, ["--json", "unreal", "exec", "print('x')"])

    assert result.exit_code == 1, result.stdout
    payload = json.loads(result.stdout)
    assert payload["success"] is False
    assert payload["error"] == "weird payload"


def test_exec_command_result_falls_back_when_empty(monkeypatch) -> None:
    """A failure with no CommandResult must surface a default error message
    rather than an empty string the user can't act on."""
    _stub_session_factory(
        monkeypatch,
        {"ReturnValue": False, "LogOutput": [], "CommandResult": ""},
    )

    result = runner.invoke(app, ["--json", "unreal", "exec", "broken"])

    assert result.exit_code == 1, result.stdout
    payload = json.loads(result.stdout)
    assert payload["error"] == "Python execution failed"


def test_exec_raw_mode_dumps_raw_result_unchanged(monkeypatch) -> None:
    """--raw bypasses the new payload construction and dumps raw_result.
    Pin this so the raw-flag branch can't silently regress."""
    raw_result = {
        "ReturnValue": True,
        "LogOutput": [{"Type": "Info", "Output": "hi\n"}],
        "CommandResult": "",
        "extra_passthrough_field": 42,
    }
    _stub_session_factory(monkeypatch, raw_result)

    result = runner.invoke(app, ["--json", "unreal", "exec", "--raw", "print('hi')"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload == raw_result


def test_exec_human_mode_strips_single_trailing_newline(monkeypatch) -> None:
    """Non-JSON mode: a single trailing newline appended by UE is stripped
    so the output renders without a blank line. Internal newlines preserved.

    CliRunner captures sys.stdout, but Rich's Console was bound to the
    original sys.stdout at module-import time and bypasses the capture.
    Stub console.print to a recorder so we can assert against what got
    rendered.
    """
    _stub_session_factory(
        monkeypatch,
        {
            "ReturnValue": True,
            "LogOutput": [{"Type": "Info", "Output": "first\nsecond\n"}],
            "CommandResult": "",
        },
    )
    monkeypatch.setattr(unreal_cli, "is_json_mode", lambda: False)
    captured: list[str] = []
    monkeypatch.setattr(
        unreal_cli.console,
        "print",
        lambda *args, **kwargs: captured.append(str(args[0]) if args else ""),
    )

    result = runner.invoke(app, ["unreal", "exec", "print('first\\nsecond')"])

    assert result.exit_code == 0, result.output
    # Exactly one call, with the trailing newline stripped from UE's output.
    assert captured == ["first\nsecond"]


def test_exec_human_mode_renders_warnings_and_errors_with_prefixes(
    monkeypatch,
) -> None:
    """Non-JSON mode: Warning and Error log entries get visible prefixes so
    a script that calls unreal.log_warning(...) doesn't lose them."""
    _stub_session_factory(
        monkeypatch,
        {
            "ReturnValue": True,
            "LogOutput": [
                {"Type": "Info", "Output": "ok\n"},
                {"Type": "Warning", "Output": "deprecated\n"},
                {"Type": "Error", "Output": "non-fatal\n"},
            ],
            "CommandResult": "",
        },
    )
    monkeypatch.setattr(unreal_cli, "is_json_mode", lambda: False)
    captured: list[str] = []
    monkeypatch.setattr(
        unreal_cli.console,
        "print",
        lambda *args, **kwargs: captured.append(str(args[0]) if args else ""),
    )

    result = runner.invoke(app, ["unreal", "exec", "print('ok')"])

    assert result.exit_code == 0, result.output
    # Three rendered lines: output, warnings block, errors block.
    assert len(captured) == 3
    assert captured[0] == "ok"
    assert "warnings" in captured[1] and "deprecated" in captured[1]
    assert "errors" in captured[2] and "non-fatal" in captured[2]


def test_exec_human_mode_escapes_rich_markup_from_log_output(
    monkeypatch,
) -> None:
    """A LogOutput Info entry containing Rich markup (e.g. literal '[red]')
    must NOT be interpreted by Rich as styling — escape it."""
    _stub_session_factory(
        monkeypatch,
        {
            "ReturnValue": True,
            "LogOutput": [{"Type": "Info", "Output": "[red]not red[/red]\n"}],
            "CommandResult": "",
        },
    )
    monkeypatch.setattr(unreal_cli, "is_json_mode", lambda: False)
    captured: list[str] = []
    monkeypatch.setattr(
        unreal_cli.console,
        "print",
        lambda *args, **kwargs: captured.append(str(args[0]) if args else ""),
    )

    result = runner.invoke(app, ["unreal", "exec", "print('[red]not red[/red]')"])

    assert result.exit_code == 0, result.output
    # Brackets must be escaped so Rich renders them as literal text.
    assert captured[0] == r"\[red]not red\[/red]"


def test_exec_no_output_treated_as_success(monkeypatch) -> None:
    """A script with no print/log lines but ReturnValue=True is success
    with empty output (e.g. `x = 1` as a one-liner)."""
    _stub_session_factory(
        monkeypatch,
        {"ReturnValue": True, "LogOutput": [], "CommandResult": ""},
    )

    result = runner.invoke(app, ["--json", "unreal", "exec", "x = 1"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["success"] is True
    assert payload["output"] == ""


def test_exec_long_inline_script_is_not_treated_as_path(monkeypatch) -> None:
    """A script longer than NAME_MAX used to crash in Path.is_file with
    OSError (File name too long) on Python 3.11-3.13."""
    captured: Dict[str, Any] = {}

    class _StubSession:
        settings = unreal_cli.get_settings()

        async def _execute_python(self, code, mode):
            del mode
            captured["code"] = code
            return {"ReturnValue": True, "LogOutput": [], "CommandResult": ""}

    monkeypatch.setattr(unreal_cli, "_session", lambda *a, **kw: _StubSession())
    script = "x = 1  # " + "a" * 5000

    result = runner.invoke(app, ["--json", "unreal", "exec", script])

    assert result.exit_code == 0, result.stdout
    assert captured["code"] == script


def test_is_script_file(tmp_path: Path) -> None:
    script = tmp_path / "s.py"
    script.write_text("print(1)\n")
    assert unreal_cli._is_script_file(str(script)) is True
    assert unreal_cli._is_script_file(str(tmp_path / "missing.py")) is False
    assert unreal_cli._is_script_file("print('a')") is False
    assert unreal_cli._is_script_file("a" * 5000 + ".py") is False


# ---------------------------------------------------------------------------
# setup: a plain re-run clears a public bind; launch failures fail fast;
# a dead editor stops the poll; a specific --bind is where the poll goes.
# ---------------------------------------------------------------------------


async def _not_connected(session, timeout, interval, **kwargs):
    del session, timeout, interval, kwargs
    return {"connected": False}


def test_setup_rerun_without_bind_clears_public_bind(tmp_path: Path, monkeypatch) -> None:
    uproject = _write_uproject(tmp_path)
    monkeypatch.setattr(unreal_cli, "_poll_health", _not_connected)
    base = ["--json", "unreal", "setup", str(uproject), "--no-launch", "--yes"]
    runner.invoke(app, base + ["--bind", "0.0.0.0", "--allow-public"])

    result = runner.invoke(app, base)

    payload = json.loads(result.stdout)
    assert payload["bind"] is None
    assert payload["patched"]["engine_ini"]["removed"] == ["DefaultBindAddress"]
    assert "RemoteControlWebsocketServerBindAddress" in payload["patched"]["ini"]["updated"]
    assert "0.0.0.0" not in (tmp_path / "Config" / "DefaultEngine.ini").read_text()
    assert "0.0.0.0" not in (tmp_path / "Config" / "DefaultRemoteControl.ini").read_text()


def test_setup_rerun_with_allow_public_keeps_public_bind(tmp_path: Path, monkeypatch) -> None:
    uproject = _write_uproject(tmp_path)
    monkeypatch.setattr(unreal_cli, "_poll_health", _not_connected)
    base = ["--json", "unreal", "setup", str(uproject), "--no-launch", "--yes"]
    runner.invoke(app, base + ["--bind", "0.0.0.0", "--allow-public"])

    runner.invoke(app, base + ["--allow-public"])

    assert "DefaultBindAddress=0.0.0.0" in (tmp_path / "Config" / "DefaultEngine.ini").read_text()


@pytest.mark.parametrize("json_mode", [True, False])
def test_setup_launcher_not_found_fails_fast(tmp_path: Path, monkeypatch, json_mode: bool) -> None:
    """With --yes the plan panel is skipped, so the launcher error used to be
    invisible while setup polled the full --wait-timeout."""
    uproject = _write_uproject(tmp_path)

    def _no_launcher(*args, **kwargs):
        raise unreal_cli.LauncherNotFound("Could not find UnrealEditor on macOS.")

    async def _must_not_poll(*args, **kwargs):
        raise AssertionError("setup polled after a launcher failure")

    monkeypatch.setattr(unreal_cli, "resolve_launch_argv", _no_launcher)
    monkeypatch.setattr(unreal_cli, "_poll_health", _must_not_poll)
    argv = ["unreal", "setup", str(uproject), "--yes"]

    result = runner.invoke(app, (["--json"] if json_mode else []) + argv)

    assert result.exit_code != 0
    assert "Could not find UnrealEditor on macOS." in result.stdout
    assert not (tmp_path / "Config" / "DefaultRemoteControl.ini").exists()


class _FakeProc:
    pid = 4242

    def __init__(self, code):
        self._code = code

    def poll(self):
        return self._code


def test_poll_health_stops_when_editor_exits() -> None:
    import asyncio

    calls = []

    class _Session:
        async def health_check(self):
            calls.append(1)
            return {"connected": False}

    health = asyncio.run(unreal_cli._poll_health(_Session(), 60.0, 0.01, proc=_FakeProc(3)))

    assert health["editor_exited"] is True
    assert health["exit_code"] == 3
    assert "4242" in health["error"]
    assert len(calls) == 1


def test_poll_health_keeps_polling_while_editor_runs() -> None:
    import asyncio

    class _Session:
        def __init__(self):
            self.n = 0

        async def health_check(self):
            self.n += 1
            return {"connected": self.n >= 3}

    health = asyncio.run(unreal_cli._poll_health(_Session(), 60.0, 0.0, proc=_FakeProc(None)))

    assert health == {"connected": True}


def _stub_launch(monkeypatch, argv0: str, captured: Dict[str, Any]) -> None:
    monkeypatch.setattr(unreal_cli, "resolve_launch_argv", lambda *a, **k: [argv0, "x.uproject"])
    monkeypatch.setattr(unreal_cli, "launch_editor", lambda *a, **k: _FakeProc(1))

    async def _fake_poll(session, timeout, interval, **kwargs):
        del timeout, interval
        captured["host"] = session.host
        captured["proc"] = kwargs.get("proc")
        return unreal_cli._editor_exit(kwargs.get("proc")) or {"connected": False}

    monkeypatch.setattr(unreal_cli, "_poll_health", _fake_poll)


def test_setup_reports_editor_crash(tmp_path: Path, monkeypatch) -> None:
    uproject = _write_uproject(tmp_path)
    captured: Dict[str, Any] = {}
    _stub_launch(monkeypatch, "/opt/UE/UnrealEditor", captured)

    result = runner.invoke(app, ["unreal", "setup", str(uproject), "--yes"])

    assert result.exit_code == 1
    assert captured["proc"] is not None
    assert "exited with code 1" in result.stdout


def test_setup_does_not_watch_open_launcher(tmp_path: Path, monkeypatch) -> None:
    """`open -a` exits as soon as LaunchServices has the app; that exit is not a crash."""
    uproject = _write_uproject(tmp_path)
    captured: Dict[str, Any] = {}
    _stub_launch(monkeypatch, "open", captured)

    runner.invoke(app, ["--json", "unreal", "setup", str(uproject), "--yes", "--no-headless"])

    assert captured["proc"] is None


@pytest.mark.parametrize(
    "bind, expected",
    [
        (None, None),
        ("0.0.0.0", None),
        ("::", None),
        ("any", None),
        ("192.168.1.10", "192.168.1.10"),
        ("127.0.0.1", "127.0.0.1"),
        ("fe80::1", "[fe80::1]"),
    ],
)
def test_poll_host(bind, expected) -> None:
    assert unreal_cli._poll_host(bind) == expected


def test_setup_polls_specific_bind_address(tmp_path: Path, monkeypatch) -> None:
    uproject = _write_uproject(tmp_path)
    captured: Dict[str, Any] = {}

    async def _fake_poll(session, timeout, interval, **kwargs):
        del timeout, interval, kwargs
        captured["host"] = session.host
        return {"connected": False}

    monkeypatch.setattr(unreal_cli, "_poll_health", _fake_poll)
    argv = ["--json", "unreal", "setup", str(uproject), "--no-launch", "--yes"]

    runner.invoke(app, argv + ["--bind", "192.168.1.10", "--allow-public"])
    assert captured["host"] == "192.168.1.10"

    runner.invoke(app, argv + ["--bind", "0.0.0.0", "--allow-public"])
    assert captured["host"] == unreal_cli.get_settings().unreal.host
