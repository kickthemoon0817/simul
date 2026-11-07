"""Tests for the shared CLI plumbing in ``simul.cli.output``."""

from __future__ import annotations

import errno
import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
from unittest.mock import AsyncMock, patch

import click
import pytest
import typer
from typer.testing import CliRunner

from simul.cli import isaac as isaac_cli
from simul.cli import output, unreal_cli
from simul.cli.main import app as main_app

runner = CliRunner()


def _invoke(body: Callable[[], None]) -> Any:
    """Run ``body`` as a one-command Typer app and return the CliRunner result."""
    app = typer.Typer()

    @app.command()
    def cmd() -> None:
        body()

    return runner.invoke(app, [])


async def _returns(payload: Any) -> Any:
    return payload


async def _raises() -> Dict[str, Any]:
    raise ConnectionRefusedError("refused")


_needs_file_modes = pytest.mark.skipif(
    sys.platform == "win32" or os.geteuid() == 0,
    reason="mode 000 binds neither root nor Windows",
)
_needs_symlinks = pytest.mark.skipif(sys.platform == "win32", reason="creating a symlink on Windows needs a privilege")


@pytest.fixture(
    params=[
        "directory",
        pytest.param("broken-symlink", marks=_needs_symlinks),
        pytest.param("symlink-loop", marks=_needs_symlinks),
        pytest.param("no-read-permission", marks=_needs_file_modes),
        pytest.param("unsearchable-parent", marks=_needs_file_modes),
    ]
)
def unreadable_py(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[tuple[Path, str]]:
    """A ``.py`` path that exists but can't be read, and the reason the OS gives."""
    if request.param == "directory":
        script = tmp_path / "job.py"
        script.mkdir()
        # Windows refuses to open a directory with EACCES rather than EISDIR.
        yield script, os.strerror(errno.EACCES if sys.platform == "win32" else errno.EISDIR)
        return
    if request.param in ("broken-symlink", "symlink-loop"):
        # The link itself is on disk; opening it follows it to nothing, or round and round.
        script = tmp_path / "job.py"
        broken = request.param == "broken-symlink"
        script.symlink_to(tmp_path / "gone.py" if broken else script)
        yield script, os.strerror(errno.ENOENT if broken else errno.ELOOP)
        return
    folder = tmp_path / "locked" if request.param == "unsearchable-parent" else tmp_path
    folder.mkdir(exist_ok=True)
    script = folder / "job.py"
    script.write_text("print(1)\n", encoding="utf-8")
    closed = folder if request.param == "unsearchable-parent" else script
    closed.chmod(0)
    try:
        yield script, os.strerror(errno.EACCES)
    finally:
        closed.chmod(0o755)


@pytest.fixture
def human_mode(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Force human mode and record what the shared console prints."""
    monkeypatch.setattr(output, "is_json_mode", lambda: False)
    printed: list[str] = []
    monkeypatch.setattr(
        output.console, "print", lambda *args, **kwargs: printed.append(str(args[0]) if args else "")
    )
    return printed


# ---------------------------------------------------------------------------
# run_or_exit
# ---------------------------------------------------------------------------
def test_run_or_exit_returns_a_successful_payload() -> None:
    seen: list[Any] = []
    result = _invoke(lambda: seen.append(output.run_or_exit(_returns({"prims": 3}))))

    assert result.exit_code == 0, result.stdout
    assert seen == [{"prims": 3, "success": True}]


def test_run_or_exit_explicit_success_false_exits_non_zero_with_the_payload() -> None:
    payload = {"success": False, "connected": False}
    result = _invoke(lambda: output.run_or_exit(_returns(payload)))

    assert result.exit_code == 1
    assert json.loads(result.stdout) == payload


def test_run_or_exit_suffixed_error_key_counts_as_failure() -> None:
    """A partial-failure payload (runtime-info's physics_error) exits non-zero and keeps its data."""
    payload = {"kit_version": "107.3", "physics_error": "no PhysicsScene"}
    result = _invoke(lambda: output.run_or_exit(_returns(payload)))

    assert result.exit_code == 1
    emitted = json.loads(result.stdout)
    assert emitted["success"] is False
    assert emitted["kit_version"] == "107.3"
    assert emitted["physics_error"] == "no PhysicsScene"


def test_run_or_exit_top_level_error_writes_the_error_envelope() -> None:
    payload = {"error": "boom", "error_type": "ScriptError", "details": {"traceback": "tb"}}
    result = _invoke(lambda: output.run_or_exit(_returns(payload)))

    assert result.exit_code == 1
    assert json.loads(result.stdout) == {
        "success": False,
        "error": "boom",
        "error_type": "ScriptError",
        "details": {"traceback": "tb"},
    }


def test_run_or_exit_error_with_success_true_still_fails() -> None:
    result = _invoke(lambda: output.run_or_exit(_returns({"success": True, "error": "contradiction"})))

    assert result.exit_code == 1
    emitted = json.loads(result.stdout)
    assert (emitted["success"], emitted["error"]) == (False, "contradiction")


def test_run_or_exit_top_level_error_keeps_the_rest_of_the_payload() -> None:
    """Data a failure carries reaches a JSON caller, as the MCP envelope passes it on."""
    payload = {
        "actor_path": "/Game/Map.Map:PersistentLevel.Actor_0",
        "location": [0.0, 0.0, 0.0],
        "success": False,
        "error": "Unreal did not apply: location",
        "error_type": "TransformNotApplied",
    }
    result = _invoke(lambda: output.run_or_exit(_returns(dict(payload))))

    assert result.exit_code == 1
    assert json.loads(result.stdout) == payload


def test_run_or_exit_top_level_error_defaults_the_error_type() -> None:
    result = _invoke(lambda: output.run_or_exit(_returns({"error": "boom", "partial": 1})))

    assert result.exit_code == 1
    assert json.loads(result.stdout) == {
        "success": False,
        "error": "boom",
        "error_type": "Error",
        "partial": 1,
    }


def test_run_or_exit_passes_non_dict_results_through() -> None:
    seen: list[Any] = []
    result = _invoke(lambda: seen.append(output.run_or_exit(_returns([1, 2]))))

    assert result.exit_code == 0
    assert seen == [[1, 2]]


def test_run_or_exit_catches_exceptions_when_asked() -> None:
    result = _invoke(lambda: output.run_or_exit(_raises(), catch_exceptions=True))

    assert result.exit_code == 1
    envelope = json.loads(result.stdout)
    assert envelope["error"] == "refused"
    assert envelope["error_type"] == "ConnectionRefusedError"


def test_run_or_exit_lets_exceptions_propagate_by_default() -> None:
    result = _invoke(lambda: output.run_or_exit(_raises()))

    assert isinstance(result.exception, ConnectionRefusedError)


def test_run_or_exit_human_mode_prints_the_error_and_traceback(human_mode: list[str]) -> None:
    payload = {"error": "boom", "error_type": "ScriptError", "details": {"traceback": "tb"}}
    result = _invoke(lambda: output.run_or_exit(_returns(payload), show_traceback=True))

    assert result.exit_code == 1
    assert human_mode[0] == "[red]ScriptError: boom[/red]"
    assert "Panel" in human_mode[1]


def test_run_or_exit_human_mode_names_the_suffixed_errors(human_mode: list[str]) -> None:
    result = _invoke(lambda: output.run_or_exit(_returns({"physics_error": "no scene"})))

    assert result.exit_code == 1
    assert human_mode == ["[red]Error: physics_error: no scene[/red]"]


# ---------------------------------------------------------------------------
# fail
# ---------------------------------------------------------------------------
def test_fail_json_mode_writes_the_envelope_and_honours_the_exit_code() -> None:
    result = _invoke(lambda: output.fail("bad arg", "InvalidArgument", {"k": 1}, exit_code=2))

    assert result.exit_code == 2
    assert json.loads(result.stdout) == {
        "success": False,
        "error": "bad arg",
        "error_type": "InvalidArgument",
        "details": {"k": 1},
    }


def test_fail_defaults_to_exit_1() -> None:
    result = _invoke(lambda: output.fail("nope"))

    assert result.exit_code == 1
    assert json.loads(result.stdout)["error_type"] == "Error"


def test_fail_human_mode_prints_escaped_message(human_mode: list[str]) -> None:
    result = _invoke(lambda: output.fail("expected one of [a, b]", "ValueError", exit_code=2))

    assert result.exit_code == 2
    assert result.stdout == ""
    assert human_mode == [r"[red]expected one of \[a, b][/red]"]


# ---------------------------------------------------------------------------
# Port and timeout options
# ---------------------------------------------------------------------------
_PORT_ERROR = "{option} takes a port number from 1 to 65535, got {value!r}"
_TIMEOUT_ERROR = "{option} takes a finite number of seconds greater than 0, got {value!r}"


def _cli_options(matches: Callable[[str], bool]) -> List[Tuple[Tuple[str, ...], str]]:
    """``(command path, long option)`` for every option of the ``simul`` CLI whose name ``matches``."""
    found: List[Tuple[Tuple[str, ...], str]] = []

    def walk(command: click.Command, path: Tuple[str, ...]) -> None:
        if isinstance(command, click.Group):
            for name, sub in command.commands.items():
                walk(sub, (*path, name))
            return
        for param in command.params:
            found.extend((path, opt) for opt in param.opts if matches(opt))

    walk(typer.main.get_command(main_app), ())
    return found


# Every port option, the shared --port and the others (--socket-port, --websocket-port, ...),
# and every timeout option (--timeout, --wait-timeout). Not --viewport.
PORT_OPTIONS = _cli_options(lambda opt: opt == "--port" or opt.endswith("-port"))
TIMEOUT_OPTIONS = _cli_options(lambda opt: opt == "--timeout" or opt.endswith("-timeout"))


def _option_id(case: Any) -> str:
    """``isaac exec --port`` for a ``(path, option)`` case, ``isaac exec print(1)`` for an argv."""
    return " ".join([*case[0], case[1]] if isinstance(case, tuple) else case)


@pytest.fixture
def no_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail a command that gets as far as building an Isaac Sim client or an Unreal session."""

    def reached(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the command reached the engine")

    monkeypatch.setattr(isaac_cli, "_tools", reached)
    monkeypatch.setattr(unreal_cli, "_session", reached)
    # Without an install root, a launch that got past its options stops at once.
    monkeypatch.delenv("ISAAC_SIM_PATH", raising=False)


def test_the_option_walk_finds_the_shared_options() -> None:
    """The cases below come from walking the CLI; it must reach both backends' shared options."""
    ports = {
        (("isaac", "exec"), "--port"),
        (("isaac", "launch"), "--socket-port"),
        (("unreal", "health"), "--port"),
        (("unreal", "instances"), "--port"),
        (("unreal", "setup"), "--websocket-port"),
    }
    timeouts = {
        (("isaac", "exec"), "--timeout"),
        (("isaac", "launch"), "--wait-timeout"),
        (("unreal", "setup"), "--wait-timeout"),
    }
    assert ports <= set(PORT_OPTIONS)
    assert timeouts <= set(TIMEOUT_OPTIONS)
    assert not [case for case in PORT_OPTIONS if case[1] == "--viewport"]


@pytest.mark.parametrize("case", PORT_OPTIONS, ids=_option_id)
def test_every_port_option_refuses_an_impossible_port_as_a_bad_invocation(
    case: Tuple[Tuple[str, ...], str], no_engine: None
) -> None:
    """Exit 2 with the envelope while the command line is parsed, for every command that takes a port.

    The option is the only argument: Click processes what was given before
    what is missing, so a command's required argument does not get in first.
    """
    path, option = case
    result = runner.invoke(main_app, ["--json", *path, option, "70000"])

    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout) == {
        "success": False,
        "error": _PORT_ERROR.format(option=option, value="70000"),
        "error_type": "InvalidArgument",
    }


@pytest.mark.parametrize("case", TIMEOUT_OPTIONS, ids=_option_id)
def test_every_timeout_option_refuses_an_impossible_timeout_as_a_bad_invocation(
    case: Tuple[Tuple[str, ...], str], no_engine: None
) -> None:
    path, option = case
    result = runner.invoke(main_app, ["--json", *path, option, "nan"])

    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout) == {
        "success": False,
        "error": _TIMEOUT_ERROR.format(option=option, value="nan"),
        "error_type": "InvalidArgument",
    }


@pytest.mark.parametrize("value", ["0", "-1", "65536", "70000", "abc", "8226.0", ""])
@pytest.mark.parametrize("argv", [["isaac", "exec", "print(1)"], ["unreal", "health"]], ids=_option_id)
def test_a_port_outside_1_to_65535_or_not_an_integer_exits_2(argv: List[str], value: str, no_engine: None) -> None:
    """A value Click cannot parse as an integer gets the same envelope, not Click's usage error on stderr."""
    result = runner.invoke(main_app, ["--json", *argv, "-p", value])

    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout) == {
        "success": False,
        "error": _PORT_ERROR.format(option="--port", value=value),
        "error_type": "InvalidArgument",
    }


@pytest.mark.parametrize("value", ["0", "-0.5", "-1", "nan", "inf", "-inf", "abc", ""])
@pytest.mark.parametrize("argv", [["isaac", "exec", "print(1)"], ["isaac", "ping"]], ids=_option_id)
def test_a_timeout_that_is_not_a_finite_number_above_0_exits_2(
    argv: List[str], value: str, no_engine: None
) -> None:
    """0, -1 and nan used to reach the socket and come back as 'timed out after nans'."""
    result = runner.invoke(main_app, ["--json", *argv, "-t", value])

    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout) == {
        "success": False,
        "error": _TIMEOUT_ERROR.format(option="--timeout", value=value),
        "error_type": "InvalidArgument",
    }


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["isaac", "exec", "print(1)", "--port", "70000"], _PORT_ERROR.format(option="--port", value="70000")),
        (["isaac", "ping", "--timeout", "0"], _TIMEOUT_ERROR.format(option="--timeout", value="0")),
        (["unreal", "health", "-p", "abc"], _PORT_ERROR.format(option="--port", value="abc")),
    ],
    ids=["isaac exec --port", "isaac ping --timeout", "unreal health -p"],
)
def test_a_bad_port_or_timeout_is_text_on_stderr_in_human_mode(
    argv: List[str], message: str, human_mode: list[str], no_engine: None
) -> None:
    result = runner.invoke(main_app, argv)

    assert result.exit_code == 2, result.output
    assert result.stdout == ""
    assert human_mode == [f"[red]{message}[/red]"]


def test_valid_ports_and_timeouts_reach_the_command_as_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Values at both ends of the range arrive as Click parsed them, and leaving the options out still means None."""
    seen: list[tuple] = []

    def tools(host: Optional[str] = None, port: Optional[int] = None, timeout: Optional[float] = None) -> Any:
        seen.append((port, timeout))
        return SimpleNamespace(_client=SimpleNamespace(ping=AsyncMock(return_value=True), address="h:1"))

    monkeypatch.setattr(isaac_cli, "_tools", tools)
    for port, timeout in (("1", "0.001"), ("8226", "30"), ("65535", "1e3")):
        result = runner.invoke(main_app, ["--json", "isaac", "ping", "--port", port, "--timeout", timeout])
        assert result.exit_code == 0, result.output
    result = runner.invoke(main_app, ["--json", "isaac", "ping"])

    assert result.exit_code == 0, result.output
    assert seen == [(1, 0.001), (8226, 30.0), (65535, 1000.0), (None, None)]


@pytest.mark.parametrize(
    ("extra", "port", "websocket_port", "wait"),
    [
        ([], 30010, None, 90.0),
        (["--port", "30011", "--websocket-port", "30021", "--wait-timeout", "5"], 30011, 30021, 5.0),
    ],
    ids=["defaults", "given"],
)
def test_setup_port_and_wait_options_keep_their_defaults_and_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra: List[str],
    port: int,
    websocket_port: Optional[int],
    wait: float,
) -> None:
    uproject = tmp_path / "Demo.uproject"
    uproject.write_text(json.dumps({"FileVersion": 3, "EngineAssociation": "5.4"}), encoding="utf-8")
    waited: list[float] = []

    async def poll(session: Any, timeout: float, interval: float, **kwargs: Any) -> Dict[str, Any]:
        waited.append(timeout)
        return {"connected": True}

    monkeypatch.setattr(unreal_cli, "_poll_health", poll)
    result = runner.invoke(main_app, ["--json", "unreal", "setup", str(uproject), "--no-launch", "--yes", *extra])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert (payload["port"], payload["websocket_port"], waited) == (port, websocket_port, [wait])
    assert f"RemoteControlHttpServerPort={port}" in (tmp_path / "Config" / "DefaultRemoteControl.ini").read_text()


def test_port_and_timeout_options_keep_their_types_in_the_commands_manifest() -> None:
    result = runner.invoke(main_app, ["commands"])

    types = {
        (entry["command"], param["name"]): param["type"]
        for entry in json.loads(result.stdout)["commands"]
        for param in entry["params"]
    }
    assert types[("isaac exec", "port")] == types[("unreal setup", "websocket_port")] == "integer"
    assert types[("isaac exec", "timeout")] == types[("unreal setup", "wait_timeout")] == "float"


# ---------------------------------------------------------------------------
# read_script_arg
# ---------------------------------------------------------------------------
def test_read_script_arg_returns_inline_code() -> None:
    assert output.read_script_arg("print('hi')") == "print('hi')"


def test_read_script_arg_reads_a_py_file(tmp_path: Path) -> None:
    script = tmp_path / "job.py"
    script.write_text("x = 1\n", encoding="utf-8")

    assert output.read_script_arg(str(script)) == "x = 1\n"


def test_read_script_arg_does_not_stat_long_inline_code() -> None:
    """A long inline script must not reach Path.is_file (ENAMETOOLONG raises OSError)."""
    code = "x = 1\n" * 2000
    assert output.read_script_arg(code) == code


def test_read_script_arg_tolerates_a_too_long_py_suffixed_string() -> None:
    code = "#" * 5000 + ".py"
    assert output.read_script_arg(code) == code


def test_read_script_arg_missing_py_file_is_treated_as_code() -> None:
    assert output.read_script_arg("does_not_exist.py") == "does_not_exist.py"


def test_read_script_arg_refuses_a_py_path_it_cannot_read(unreadable_py: tuple[Path, str]) -> None:
    """Exit 2 naming the path and the OS reason; the path must not go to the engine as code."""
    script, reason = unreadable_py

    result = _invoke(lambda: output.read_script_arg(str(script)))

    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout) == {
        "success": False,
        "error": f"Script file {script} could not be read: {reason}.",
        "error_type": "ReadError",
    }


def test_read_script_arg_reads_piped_stdin() -> None:
    seen: list[str] = []
    app = typer.Typer()

    @app.command()
    def cmd() -> None:
        seen.append(output.read_script_arg(None))

    result = runner.invoke(app, [], input="print('piped')\n")

    assert result.exit_code == 0
    assert seen == ["print('piped')\n"]


@pytest.mark.parametrize("stdin", ["terminal", "closed"])
def test_read_script_arg_without_input_is_a_bad_invocation(stdin: str) -> None:
    """No script and nothing piped exits 2, the usage-error code, whether stdin is a terminal or absent."""

    def body() -> None:
        # CliRunner swaps sys.stdin in during invoke, so make it a terminal, or take it away, here.
        if stdin == "terminal":
            no_input = patch.object(output.sys.stdin, "isatty", return_value=True)
        else:
            no_input = patch.object(output.sys, "stdin", None)
        with no_input:
            output.read_script_arg(None)

    result = _invoke(body)

    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout) == {
        "success": False,
        "error": "Provide a script string, .py file path, or pipe code via stdin.",
        "error_type": "InputError",
    }


def test_read_script_arg_rejects_a_py_file_that_is_not_utf8(tmp_path: Path) -> None:
    """Exit 2 naming the file and the byte; the offset counts the CR that universal newlines drop."""
    script = tmp_path / "job.py"
    script.write_bytes(b"x = 1\r\n\xff = 2\n")

    result = _invoke(lambda: output.read_script_arg(str(script)))

    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout) == {
        "success": False,
        "error": f"Script file {script} is not valid UTF-8: byte 0xff (invalid start byte) at offset 7.",
        "error_type": "InvalidEncoding",
    }


def test_read_script_arg_rejects_piped_stdin_that_is_not_utf8() -> None:
    """Piped bytes decode as strict UTF-8; the offset counts bytes, not characters."""
    app = typer.Typer()

    @app.command()
    def cmd() -> None:
        output.read_script_arg(None)

    result = runner.invoke(app, [], input="# 한\nx = '".encode("utf-8") + b"\xff'\n")

    assert result.exit_code == 2, result.output
    envelope = json.loads(result.stdout)
    assert envelope["error_type"] == "InvalidEncoding"
    assert envelope["error"] == "Script from stdin is not valid UTF-8: byte 0xff (invalid start byte) at offset 11."


@pytest.mark.parametrize(
    ("platform", "encoding", "code", "message"),
    [
        # POSIX hands argv to Python as bytes, decoded with the filesystem encoding and surrogateescape.
        # Under UTF-8, "# 한" and then the byte 0xff arrive with U+DCFF standing for that byte.
        (
            "linux",
            "utf-8",
            b"# \xed\x95\x9c\nx = '\xff'".decode("utf-8", "surrogateescape"),
            "Script argument is not valid UTF-8: byte 0xff at offset 11.",
        ),
        # Under ASCII the bytes of "한" can't be decoded either, though they are valid UTF-8.
        (
            "linux",
            "ascii",
            "print('한')".encode("utf-8").decode("ascii", "surrogateescape"),
            "Script argument could not be decoded in the filesystem encoding (ascii); "
            "pass the script in a .py file or on stdin, which are read as UTF-8.",
        ),
        # Windows hands argv over as UTF-16, so U+DCFF there is a lone surrogate, not a byte.
        (
            "win32",
            "utf-8",
            "# 한\nx = '\udcff'",
            "Script argument is not valid UTF-8: lone surrogate U+DCFF at offset 11.",
        ),
        # No decoding of bytes yields U+D800, so it is named as itself.
        (
            "linux",
            "utf-8",
            "# 한\nx = '\ud800'",
            "Script argument is not valid UTF-8: lone surrogate U+D800 at offset 11.",
        ),
    ],
    ids=["posix-utf8", "posix-ascii", "windows", "other-surrogate"],
)
def test_read_script_arg_rejects_an_argument_carrying_lone_surrogates(
    platform: str, encoding: str, code: str, message: str
) -> None:
    """A lone surrogate cannot go out as UTF-8 or be sized in it; the error names a byte only where argv's
    decoding makes that byte and its offset exact."""

    def body() -> None:
        with patch.object(output.sys, "platform", platform), patch.object(
            output.sys, "getfilesystemencoding", return_value=encoding
        ):
            output.read_script_arg(code)

    result = _invoke(body)

    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout) == {"success": False, "error": message, "error_type": "InvalidEncoding"}


def test_read_script_arg_reads_a_multibyte_utf8_file_as_before(tmp_path: Path) -> None:
    """The BOM stays a leading U+FEFF and universal newlines turn CRLF into LF, as they always did."""
    script = tmp_path / "job.py"
    script.write_bytes("\ufeff# 한국어 🙂\r\nprint('héllo')\r\n".encode("utf-8"))

    assert output.read_script_arg(str(script)) == "\ufeff# 한국어 🙂\nprint('héllo')\n"


def test_read_script_arg_decodes_piped_stdin_as_utf8_whatever_the_locale() -> None:
    """Valid UTF-8 on stdin comes back as sent, BOM and CRLF included, even where the text
    stdin decodes with a legacy code page (cp1252 here, as a pipe does on Windows)."""
    seen: list[str] = []
    app = typer.Typer()

    @app.command()
    def cmd() -> None:
        seen.append(output.read_script_arg(None))

    script = "\ufeff# 한국어 🙂\r\nprint('héllo')\r\n"
    result = CliRunner(charset="cp1252").invoke(app, [], input=script.encode("utf-8"))

    assert result.exit_code == 0, result.output
    assert seen == [script]


def test_read_script_arg_reads_a_text_only_stdin_as_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stdin replaced by an io.StringIO has no bytes buffer (TextIOBase allows that); its text comes back as is."""
    monkeypatch.setattr(sys, "stdin", io.StringIO("# 한\r\nprint(1)\n"))

    assert output.read_script_arg(None) == "# 한\r\nprint(1)\n"


def test_read_script_arg_checks_a_text_only_stdin_as_it_checks_an_argument(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("# 한\nx = '\udcff'"))
    monkeypatch.setattr(output, "is_json_mode", lambda: True)

    with pytest.raises(typer.Exit) as excinfo:
        output.read_script_arg(None)

    assert excinfo.value.exit_code == 2
    assert json.loads(capsys.readouterr().out) == {
        "success": False,
        "error": "Script from stdin is not valid UTF-8: byte 0xff at offset 11.",
        "error_type": "InvalidEncoding",
    }


# ---------------------------------------------------------------------------
# End to end through a backend command
# ---------------------------------------------------------------------------
class _UnrealSession:
    """Session double for ``unreal exec`` that records the scripts it is sent."""

    settings = unreal_cli.get_settings()

    def __init__(self, sent: list[str]) -> None:
        self.sent = sent

    async def _execute_python(self, code: str, mode: str) -> Dict[str, Any]:
        self.sent.append(code)
        return {"ReturnValue": True, "LogOutput": [], "CommandResult": ""}

    async def close(self) -> None:
        pass


@pytest.fixture
def sent_scripts(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stand in for Isaac Sim and Unreal behind ``exec``; collects the scripts they receive."""
    sent: list[str] = []

    async def execute_script(code: str, keep_raw_output: bool = False) -> Dict[str, Any]:
        sent.append(code)
        return {"output": ""}

    monkeypatch.setattr(isaac_cli, "_tools", lambda *args, **kwargs: SimpleNamespace(execute_script=execute_script))
    monkeypatch.setattr(unreal_cli, "_session", lambda *args, **kwargs: _UnrealSession(sent))
    return sent


@pytest.mark.parametrize("backend", ["isaac", "unreal"])
@pytest.mark.parametrize("source", ["file", "stdin", "argument"])
def test_exec_rejects_script_input_that_is_not_utf8_before_the_engine(
    backend: str, source: str, tmp_path: Path, sent_scripts: list[str]
) -> None:
    """Every command that takes a script exits 2 on bad bytes, however they arrive, and sends nothing."""
    not_utf8 = b"x = '\xff'\n"
    script = tmp_path / "job.py"
    script.write_bytes(not_utf8)
    args = {"file": [str(script)], "stdin": [], "argument": [not_utf8.decode("utf-8", "surrogateescape")]}

    result = runner.invoke(
        main_app, ["--json", backend, "exec", *args[source]], input=not_utf8 if source == "stdin" else None
    )

    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout)["error_type"] == "InvalidEncoding"
    assert sent_scripts == []


@pytest.mark.parametrize("backend", ["isaac", "unreal"])
def test_exec_refuses_a_py_path_it_cannot_read_before_the_engine(
    backend: str, unreadable_py: tuple[Path, str], sent_scripts: list[str]
) -> None:
    script, reason = unreadable_py

    result = runner.invoke(main_app, ["--json", backend, "exec", str(script)])

    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout) == {
        "success": False,
        "error": f"Script file {script} could not be read: {reason}.",
        "error_type": "ReadError",
    }
    assert sent_scripts == []


@pytest.mark.parametrize("backend", ["isaac", "unreal"])
def test_exec_without_a_script_or_piped_stdin_exits_2_before_the_engine(
    backend: str, sent_scripts: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def from_a_terminal(script: Optional[str]) -> str:
        # CliRunner swaps sys.stdin in during invoke, so make it a terminal from inside the command.
        with patch.object(output.sys.stdin, "isatty", return_value=True):
            return output.read_script_arg(script)

    monkeypatch.setattr(isaac_cli, "read_script_arg", from_a_terminal)
    monkeypatch.setattr(unreal_cli, "read_script_arg", from_a_terminal)

    result = runner.invoke(main_app, ["--json", backend, "exec"])

    assert result.exit_code == 2, result.output
    assert json.loads(result.stdout)["error_type"] == "InputError"
    assert sent_scripts == []


@pytest.mark.parametrize("backend", ["isaac", "unreal"])
def test_exec_sends_a_multibyte_script_unchanged(backend: str, sent_scripts: list[str]) -> None:
    code = "# 한국어 🙂\r\nprint('héllo')"

    result = runner.invoke(main_app, ["--json", backend, "exec", code])

    assert result.exit_code == 0, result.output
    assert sent_scripts == [code]


def test_isaac_runtime_info_partial_failure_exits_non_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    tools = SimpleNamespace(
        get_runtime_info=AsyncMock(return_value={"kit_version": "107.3", "physics_error": "no PhysicsScene"})
    )
    monkeypatch.setattr(isaac_cli, "_tools", lambda *args, **kwargs: tools)

    result = runner.invoke(main_app, ["--json", "isaac", "runtime-info"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["success"] is False
    assert payload["physics_error"] == "no PhysicsScene"


def test_isaac_runtime_info_human_mode_still_renders_working_sections(
    monkeypatch: pytest.MonkeyPatch, human_mode: list[str]
) -> None:
    """One failed section must not hide the others; the command still exits 1."""
    monkeypatch.setattr(isaac_cli, "is_json_mode", lambda: False)
    tools = SimpleNamespace(
        get_runtime_info=AsyncMock(
            return_value={
                "app": {"version": "107.3"},
                "renderer_error": "no renderer",
                "physics_error": "no PhysicsScene",
            }
        )
    )
    monkeypatch.setattr(isaac_cli, "_tools", lambda *args, **kwargs: tools)
    rendered: list[Any] = []
    monkeypatch.setattr(isaac_cli.console, "print", lambda *a, **k: rendered.append(a[0] if a else ""))

    result = _invoke(lambda: isaac_cli.runtime_info(host=None, port=None))

    assert result.exit_code == 1
    titles = [getattr(r, "title", None) for r in rendered]
    assert "App" in titles
    text = " ".join(str(r) for r in rendered)
    assert "no renderer" in text and "no PhysicsScene" in text


def test_run_or_exit_allow_partial_returns_suffixed_error_payload() -> None:
    seen: list[Any] = []

    def body() -> None:
        payload = output.run_or_exit(_returns({"a": 1, "overlay_error": "x"}), allow_partial=True)
        seen.append(payload)
        output.exit_if_failed(payload)

    result = _invoke(body)

    assert result.exit_code == 1
    assert seen == [{"a": 1, "overlay_error": "x", "success": False}]


def test_run_or_exit_allow_partial_still_fails_on_top_level_error() -> None:
    result = _invoke(lambda: output.run_or_exit(_returns({"error": "boom"}), allow_partial=True))

    assert result.exit_code == 1
    assert json.loads(result.stdout)["error"] == "boom"


def test_isaac_tools_builds_client_through_the_adapter_with_cli_overrides() -> None:
    """--host/--port/--timeout address the stock socket; the bridge stays on its configured endpoint."""
    isaac = isaac_cli.get_settings().isaac_sim
    tools = isaac_cli._tools("10.0.0.5", 9000, 12.5)
    client = tools._client

    assert client.vscode_address == "10.0.0.5:9000"
    assert client.timeout_seconds == 12.5
    if isaac.bridge_enabled:
        assert client.bridge_address == f"{isaac.bridge_host}:{isaac.bridge_port}"
