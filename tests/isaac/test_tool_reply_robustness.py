"""Tool replies survive other stdout, and heavy tools get a budget that fits.

* Every transport captures the process-wide ``sys.stdout``; the stock sockets
  keep the capture on across a script's awaits. A print from anything else in
  Kit while a tool waited on frames turned a good result into
  "Failed to parse script output as JSON".
* Opening a stage, a camera's first render and a large extension enable
  outlast the default 29 s budget; the tool then reported failure for work
  Kit went on to finish.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest

from simul.adapters.isaac_socket_client import ScriptResult
from simul.config import Settings
from simul.mcp.tools.isaac._base import extract_tool_json
from simul.mcp.tools.isaac._shared import HEAVY_SCRIPT_TIMEOUT_SECONDS
from simul.mcp.tools.isaac_tools import IsaacTools


def _tools(output: str, script_timeout: Any = 29.0) -> Tuple[IsaacTools, MagicMock]:
    client = MagicMock()
    client.bridge_enabled = False
    client.fallback_to_vscode = True
    client.script_timeout_seconds = script_timeout
    client.execute = AsyncMock(return_value=ScriptResult(success=True, output=output))
    client.bridge_request = AsyncMock(return_value=None)
    return IsaacTools(client, settings=Settings()), client


def _run_json(output: str) -> Dict[str, Any]:
    tools, _ = _tools(output)
    return asyncio.run(tools._execute_json_script("print('x')"))


# ---------------------------------------------------------------------------
# Other output beside the tool's JSON
# ---------------------------------------------------------------------------


def test_lines_printed_by_other_code_before_the_result_are_set_aside() -> None:
    result = _run_json('[physx] contact report\nstep callback 3\n{"prim_path": "/World/Cube"}')

    assert result["success"] is True
    assert result["prim_path"] == "/World/Cube"
    assert result["foreign_output"] == "[physx] contact report\nstep callback 3"
    assert any("2 line(s)" in warning for warning in result["warnings"])


def test_text_glued_to_the_result_without_a_newline_is_set_aside() -> None:
    result = _run_json('progress 50%{"opened": true}')

    assert result["opened"] is True
    assert result["foreign_output"] == "progress 50%"


def test_a_result_with_its_own_warnings_list_keeps_them() -> None:
    result = _run_json('noise\n{"warnings": ["detach failed"], "aovs": {}}')

    assert result["warnings"][0] == "detach failed"
    assert len(result["warnings"]) == 2


def test_a_clean_result_is_untouched() -> None:
    result = _run_json('{"count": 3}')

    assert result == {"count": 3, "success": True}


def test_output_with_no_json_object_still_fails_to_parse() -> None:
    result = _run_json("Traceback-free garbage\nmore garbage")

    assert result["success"] is False
    assert result["error_type"] == "EngineError"
    assert result["details"]["exception_type"] == "JSONDecodeError"


def test_a_json_value_that_is_not_an_object_is_a_parse_failure_not_a_crash() -> None:
    result = _run_json("[1, 2, 3]")

    assert result["success"] is False
    assert result["error_type"] == "EngineError"
    assert result["details"]["exception_type"] == "JSONDecodeError"


def test_extraction_prefers_the_last_object_line() -> None:
    value, other = extract_tool_json('{"from": "callback"}\n{"from": "tool"}')

    assert value == {"from": "tool"}
    assert other == '{"from": "callback"}'


def test_braces_inside_strings_do_not_confuse_extraction() -> None:
    value, other = extract_tool_json('log {not json}\n{"path": "/a{b}c", "n": {"k": 1}}')

    assert value == {"path": "/a{b}c", "n": {"k": 1}}
    assert other == "log {not json}"


# ---------------------------------------------------------------------------
# Heavy tools carry a longer budget
# ---------------------------------------------------------------------------


def _timeout_of(call: Any) -> Any:
    return call.kwargs.get("timeout_seconds")


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("open_isaac_stage", {"file_path": "/tmp/scene.usd", "discard_unsaved": True}),
        ("new_isaac_stage", {"discard_unsaved": True}),
        ("read_aovs", {"aov_names": ["HdrColor"]}),
        ("enable_isaac_extension", {"extension_id": "omni.anim.people"}),
        ("capture_isaac_viewport", {}),
    ],
)
def test_heavy_tools_send_the_heavy_budget(method: str, kwargs: Dict[str, Any], tmp_path: Any) -> None:
    tools, client = _tools(json.dumps({"ok": True}))
    tools._path_policy = MagicMock()
    tools._path_policy.denial.return_value = None
    tools._path_policy.authorize.side_effect = lambda path, **_: path
    tools._path_policy.default_capture_dir.return_value = str(tmp_path)

    asyncio.run(getattr(tools, method)(**kwargs))

    assert client.execute.await_count == 1
    assert _timeout_of(client.execute.await_args) == HEAVY_SCRIPT_TIMEOUT_SECONDS


def test_ordinary_tools_keep_the_default_budget() -> None:
    tools, client = _tools(json.dumps({"count": 0, "selected_prims": []}))

    asyncio.run(tools.get_isaac_selection())

    assert _timeout_of(client.execute.await_args) is None


def test_a_longer_configured_timeout_is_never_shortened() -> None:
    tools, _ = _tools("{}", script_timeout=299.0)

    assert tools._heavy_script_timeout() == 299.0


def test_the_heavy_budget_stays_under_the_bridge_script_ceiling() -> None:
    from khemoo.simul.protocol import DEFAULT_REQUEST_TIMEOUT_SECONDS
    from khemoo.simul.service import BridgeCommandService

    service = BridgeCommandService(executor=MagicMock(), allow_unsafe_execution=True)

    assert HEAVY_SCRIPT_TIMEOUT_SECONDS < service._script_timeout_ceiling < DEFAULT_REQUEST_TIMEOUT_SECONDS


# ---------------------------------------------------------------------------
# Transport tokens stay out of tool results
# ---------------------------------------------------------------------------


def test_carb_settings_read_redacts_auth_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    import contextlib
    import io
    import sys
    import types

    captured: List[str] = []
    tools, client = _tools("{}")
    client.execute = AsyncMock(
        side_effect=lambda code, **_: captured.append(code) or ScriptResult(success=True, output="{}")
    )
    asyncio.run(
        tools.get_carb_settings(
            [
                "/exts/isaacsim.code_editor.python_server/auth_token",
                "/exts/khemoo.simul",
                "/rtx/fog/enabled",
            ]
        )
    )

    store = {
        "/exts/isaacsim.code_editor.python_server/auth_token": "kit-secret",
        "/exts/khemoo.simul": {"port": 8229, "auth_token": "bridge-secret", "require_auth": True},
        "/rtx/fog/enabled": True,
    }
    carb = types.ModuleType("carb")
    carb_settings = types.ModuleType("carb.settings")
    carb_settings.get_settings = lambda: types.SimpleNamespace(get=store.get)  # type: ignore[attr-defined]
    carb.settings = carb_settings  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "carb", carb)
    monkeypatch.setitem(sys.modules, "carb.settings", carb_settings)

    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(captured[0], {})
    settings = json.loads(stdout.getvalue())["settings"]

    assert "secret" not in json.dumps(settings)
    assert settings["/exts/isaacsim.code_editor.python_server/auth_token"] == "***"
    assert settings["/exts/khemoo.simul"] == {"port": 8229, "auth_token": "***", "require_auth": True}
    assert settings["/rtx/fog/enabled"] is True
