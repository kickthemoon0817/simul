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
        async def _execute_python(self, code, mode):  # noqa: D401
            del code, mode
            return raw_result

    monkeypatch.setattr(unreal_cli, "_session", lambda *a, **kw: _StubSession())


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
