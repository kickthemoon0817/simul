"""Tests for the shared CLI plumbing in ``simul.cli.output``."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict
from unittest.mock import AsyncMock, patch

import pytest
import typer
from typer.testing import CliRunner

from simul.cli import isaac as isaac_cli
from simul.cli import output
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
    assert json.loads(result.stdout)["error"] == "contradiction"


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


def test_read_script_arg_reads_piped_stdin() -> None:
    seen: list[str] = []
    app = typer.Typer()

    @app.command()
    def cmd() -> None:
        seen.append(output.read_script_arg(None))

    result = runner.invoke(app, [], input="print('piped')\n")

    assert result.exit_code == 0
    assert seen == ["print('piped')\n"]


def test_read_script_arg_without_input_fails() -> None:
    def body() -> None:
        # CliRunner swaps sys.stdin in during invoke, so pretend it's a terminal here.
        with patch.object(output.sys.stdin, "isatty", return_value=True):
            output.read_script_arg(None)

    result = _invoke(body)

    assert result.exit_code == 1
    assert json.loads(result.stdout)["error_type"] == "InputError"


# ---------------------------------------------------------------------------
# End to end through a backend command
# ---------------------------------------------------------------------------
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
