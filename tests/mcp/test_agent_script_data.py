"""What an agent's script prints is data, on every backend (round-2 audit B-1).

``execute_isaac_script`` and ``execute_unreal_script`` ran the JSON object a
script printed through the granular tools' rule, under which any ``error`` or
non-null ``*_error`` key fails the call. So a script printing a tracking metric
(``{"position_error": 0.002}``) came back with ``isError: true``, and
``{"error": 0.0}`` had its value replaced by "The operation failed without a
message". The CLI's ``simul isaac exec`` printed ``success: false`` for it and
still exited 0 (m-7).

The rule now, for all three script tools and both ``exec`` commands: the call
fails only when the script raised or printed ``"success": false``. These tests
drive the real envelope through a real MCP client, and the CLI through Typer,
with each engine replaced by a double that answers as the engine does.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastmcp import Client
from typer.testing import CliRunner

from simul.adapters.isaac_socket_client import ScriptResult
from simul.cli import isaac as isaac_cli
from simul.cli.main import app
from simul.config import Settings
from simul.mcp import backends as backends_module
from simul.mcp import server as server_module
from simul.mcp.tools.isaac_tools import IsaacTools
from simul.mcp.usage_tracker import ToolUsageTracker
from simul.utils.script_result import apply_script_success, printed_json_object
from tests.fakes import AvailableAdapter, BlenderSessionAdapter, StrictBlenderSession

METRICS = {
    "position_error": 0.002,
    "orientation_error": 0.01,
    "error": 0.0,
    "joint": "elbow",
}
REPORTED = {"success": False, "error": "target unreachable", "position_error": 0.4}

runner = CliRunner()


def _settings() -> Settings:
    return Settings(
        security={"allow_script_execution": True, "rate_limiting_enabled": False}
    )


def _call(
    instance: server_module.SimulMCPServer, tool: str, arguments: Dict[str, Any]
) -> Tuple[bool, Dict[str, Any]]:
    """Call ``tool`` through a real MCP client; return isError and the JSON block."""

    async def run() -> Any:
        async with Client(instance.mcp) as client:
            return await client.call_tool(tool, arguments, raise_on_error=False)

    result = asyncio.run(run())
    return bool(result.is_error), json.loads(result.content[-1].text)


def _usage(instance: server_module.SimulMCPServer, tool: str) -> List[Dict[str, Any]]:
    return list(reversed(instance.usage_tracker.get_recent(tool_name=tool)))


# ---------------------------------------------------------------------------
# The rule itself
# ---------------------------------------------------------------------------


def test_only_a_printed_success_false_fails() -> None:
    assert apply_script_success(dict(METRICS)) == {**METRICS, "success": True}
    assert apply_script_success({"success": True, "error": "kept"}) == {
        "success": True,
        "error": "kept",
    }
    assert apply_script_success({"success": False}) == {
        "success": False,
        "error_type": "ScriptError",
    }
    assert (
        apply_script_success({"success": False, "error_type": "NotFoundError"})[
            "error_type"
        ]
        == "NotFoundError"
    )


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ('{"success": false}', {"success": False}),
        ('  {"a": 1}\n', {"a": 1}),
        ("hello", None),
        ("[1, 2]", None),
        ('{"a": 1}\nmore', None),
        ("", None),
        (None, None),
    ],
)
def test_printed_json_object_is_the_whole_output(output: Any, expected: Any) -> None:
    assert printed_json_object(output) == expected


# ---------------------------------------------------------------------------
# Isaac Sim over MCP
# ---------------------------------------------------------------------------


class _IsaacClient:
    """Socket client double that answers every script with ``output``."""

    address = "127.0.0.1:8226"
    bridge_enabled = False
    fallback_to_vscode = True
    timeout_seconds = 30.0

    def __init__(self, output: str) -> None:
        self.output = output

    async def execute(self, code: str) -> ScriptResult:
        return ScriptResult(success=True, output=self.output)


def _isaac_server(
    tmp_path: Path, printed: Dict[str, Any]
) -> server_module.SimulMCPServer:
    instance = server_module.SimulMCPServer(_settings(), backends={"isaac"})
    instance.usage_tracker = ToolUsageTracker(log_dir=tmp_path)
    client = _IsaacClient(json.dumps(printed))
    instance._isaac_clients = {"default": client}  # type: ignore[dict-item]
    instance.client = client  # type: ignore[assignment]
    return instance


def test_isaac_metrics_named_error_come_back_as_data(tmp_path: Path) -> None:
    instance = _isaac_server(tmp_path, METRICS)

    is_error, payload = _call(instance, "execute_isaac_script", {"code": "print(...)"})

    assert is_error is False
    assert payload == {**METRICS, "success": True}
    (record,) = _usage(instance, "execute_isaac_script")
    assert record["success"] is True
    assert record.get("error") is None


def test_isaac_script_reports_a_failure_with_success_false(tmp_path: Path) -> None:
    instance = _isaac_server(tmp_path, REPORTED)

    is_error, payload = _call(instance, "execute_isaac_script", {"code": "print(...)"})

    assert is_error is True
    assert payload["error_type"] == "ScriptError"
    assert payload["error"] == "target unreachable"
    assert payload["position_error"] == 0.4
    (record,) = _usage(instance, "execute_isaac_script")
    assert record["success"] is False


def test_isaac_success_false_without_a_message_gets_one(tmp_path: Path) -> None:
    instance = _isaac_server(tmp_path, {"success": False, "position_error": 0.4})

    is_error, payload = _call(instance, "execute_isaac_script", {"code": "print(...)"})

    assert is_error is True
    assert payload["error_type"] == "ScriptError"
    assert payload["error"] == "position_error: 0.4"


# ---------------------------------------------------------------------------
# Unreal over MCP
# ---------------------------------------------------------------------------


class _UnrealAdapter(AvailableAdapter):
    """Unreal adapter double whose session's Python call returns ``raw``."""

    name = "unreal"

    def __init__(self, settings: Settings, raw: Dict[str, Any]) -> None:
        super().__init__(settings)
        from simul.adapters.unreal_runtime import UnrealRuntimeSession

        self.session = UnrealRuntimeSession(settings)
        self.session._execute_python = AsyncMock(return_value=raw)  # type: ignore[method-assign]

    @contextmanager
    def create_session(self) -> Iterator[Any]:
        yield self.session


def _unreal_server(
    tmp_path: Path, printed: Dict[str, Any]
) -> server_module.SimulMCPServer:
    pytest.importorskip("aiohttp")
    settings = _settings()
    instance = server_module.SimulMCPServer(settings, backends={"unreal"})
    instance.usage_tracker = ToolUsageTracker(log_dir=tmp_path)
    raw = {
        "ReturnValue": True,
        "LogOutput": [{"Type": "Info", "Output": json.dumps(printed) + "\n"}],
    }
    instance.unreal_adapter = _UnrealAdapter(settings, raw)
    return instance


def test_unreal_metrics_named_error_come_back_as_data(tmp_path: Path) -> None:
    instance = _unreal_server(tmp_path, METRICS)

    is_error, payload = _call(instance, "execute_unreal_script", {"code": "print(...)"})

    assert is_error is False
    assert payload == {**METRICS, "success": True}
    (record,) = _usage(instance, "execute_unreal_script")
    assert record["success"] is True


def test_unreal_script_reports_a_failure_with_success_false(tmp_path: Path) -> None:
    instance = _unreal_server(tmp_path, REPORTED)

    is_error, payload = _call(instance, "execute_unreal_script", {"code": "print(...)"})

    assert is_error is True
    assert payload["error_type"] == "ScriptError"
    assert payload["error"] == "target unreachable"


# ---------------------------------------------------------------------------
# Blender over MCP
# ---------------------------------------------------------------------------


def _blender_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, output: str, **extra: Any
) -> Tuple[server_module.SimulMCPServer, StrictBlenderSession]:
    monkeypatch.setattr(backends_module, "is_blender_available", lambda: True)
    monkeypatch.setattr(backends_module, "BlenderRuntimeAdapter", AvailableAdapter)
    settings = _settings()
    instance = server_module.SimulMCPServer(settings, backends={"blender"})
    instance.usage_tracker = ToolUsageTracker(log_dir=tmp_path)
    session = StrictBlenderSession(
        responses={
            "execute_script": {
                "output": output,
                "output_truncated": False,
                "output_total_chars": len(output),
                "return_value": None,
                "duration_seconds": 0.01,
                **extra,
            }
        }
    )
    instance.blender_adapter = BlenderSessionAdapter(settings, session)
    return instance, session


def test_blender_printed_metrics_stay_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    instance, session = _blender_server(monkeypatch, tmp_path, json.dumps(METRICS))

    is_error, payload = _call(
        instance, "execute_blender_script", {"script": "print(...)"}
    )

    assert is_error is False
    assert payload["success"] is True
    assert json.loads(payload["output"]) == METRICS
    assert payload.get("error") is None
    assert session.rejected == []


def test_blender_script_reports_a_failure_with_success_false(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    instance, _ = _blender_server(monkeypatch, tmp_path, json.dumps(REPORTED) + "\n")

    is_error, payload = _call(
        instance, "execute_blender_script", {"script": "print(...)"}
    )

    assert is_error is True
    assert payload["error_type"] == "ScriptError"
    assert payload["error"] == "target unreachable"
    assert json.loads(payload["output"]) == REPORTED
    (record,) = _usage(instance, "execute_blender_script")
    assert record["success"] is False


@pytest.mark.parametrize(
    ("extra", "error_type"),
    [
        ({"error": "RuntimeError: boom"}, "ScriptError"),
        (
            {
                "error": "TimeoutError: script did not finish within 5s.",
                "timed_out": True,
            },
            "TimeoutError",
        ),
    ],
)
def test_blender_raised_script_is_a_script_error_as_elsewhere(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    extra: Dict[str, Any],
    error_type: str,
) -> None:
    """It used to come back as an EngineError while Isaac and Unreal said ScriptError."""
    instance, _ = _blender_server(monkeypatch, tmp_path, "", **extra)

    is_error, payload = _call(instance, "execute_blender_script", {"script": "raise"})

    assert is_error is True
    assert payload["error_type"] == error_type
    assert payload["error"] == extra["error"]


def test_blender_success_false_inside_other_output_is_only_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only a whole output that is one JSON object reports, as on Isaac."""
    instance, _ = _blender_server(
        monkeypatch, tmp_path, "step 1\n" + json.dumps(REPORTED)
    )

    is_error, payload = _call(
        instance, "execute_blender_script", {"script": "print(...)"}
    )

    assert is_error is False
    assert payload["success"] is True


# ---------------------------------------------------------------------------
# The exec commands
# ---------------------------------------------------------------------------


def _isaac_cli_tools(
    monkeypatch: pytest.MonkeyPatch, output: str, *, success: bool = True
) -> None:
    client = MagicMock()
    client.address = "127.0.0.1:8226"
    client.timeout_seconds = 30.0
    client.bridge_enabled = False
    client.fallback_to_vscode = True
    result = ScriptResult(
        success=success,
        output=output,
        error_name="" if success else "NameError",
        error_value="" if success else "name 'x' is not defined",
    )
    client.execute = AsyncMock(return_value=result)
    client.execute_vscode_only = AsyncMock(return_value=result)
    tools = IsaacTools(client, settings=_settings())
    monkeypatch.setattr(isaac_cli, "_tools", lambda *args, **kwargs: tools)


def test_isaac_exec_keeps_error_metrics_and_exits_0(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isaac_cli_tools(monkeypatch, json.dumps(METRICS))

    result = runner.invoke(app, ["--json", "isaac", "exec", "print(...)"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {**METRICS, "success": True}


def test_isaac_exec_exits_1_when_the_script_reports_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """m-7: a payload saying success: false exits 1, as docs/cli.md says."""
    _isaac_cli_tools(monkeypatch, json.dumps({"success": False, "position_error": 0.4}))

    result = runner.invoke(app, ["--json", "isaac", "exec", "print(...)"])

    assert result.exit_code == 1, result.output
    payload = json.loads(result.stdout)
    assert payload["success"] is False
    assert payload["error_type"] == "ScriptError"
    assert payload["error"] == "position_error: 0.4"
    assert payload["position_error"] == 0.4


def test_isaac_exec_exits_1_when_the_script_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isaac_cli_tools(monkeypatch, "", success=False)

    result = runner.invoke(app, ["--json", "isaac", "exec", "x"])

    assert result.exit_code == 1, result.output
    payload = json.loads(result.stdout)
    assert payload["success"] is False
    assert payload["error_type"] == "ScriptError"


def test_isaac_exec_human_mode_names_the_reported_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isaac_cli_tools(monkeypatch, json.dumps(REPORTED))

    result = runner.invoke(app, ["isaac", "exec", "print(...)"])

    assert result.exit_code == 1
    assert "target unreachable" in result.output
