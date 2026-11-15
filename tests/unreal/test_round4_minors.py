"""The round-4 Unreal audit's minors (m2, m3, m4, m5, m6).

* m2: a ``simul unreal exec`` timeout promised a hold that the short-lived
  command released as it exited, so the next command queued behind the
  timed-out script. Its record is now left ``released`` as the command exits
  (#372's path for a stopping holder), and the next command takes it over
  with its own probe, within the record's expiry.
* m3: after a Play session the level viewport camera read [0,0,0], where the
  player's camera was, and the next editor-view capture rendered the origin.
  The editor camera is now saved before Play and put back on stop.
* m4: a default USD import that merges several prims into one mesh
  (``use_prim_kinds_for_collapsing``) now warns.
* m5: ``replace_existing`` left the assets of the earlier import that the new
  one no longer makes; they are now reported.
* m6: ``execute_unreal_script`` dropped what a script wrote to stderr or logged
  as a warning when it succeeded.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from simul.adapters.unreal_runtime import UnrealError, UnrealRuntimeSession, UnrealTimeoutError
from simul.config import Settings
from simul.utils import engine_holds

from tests.unreal.test_unreal_timeout_hold import _GameThreadEditor, _settings

# ---------------------------------------------------------------------------
# m2: the hold a short-lived command leaves behind
# ---------------------------------------------------------------------------


def test_an_exec_timeout_keeps_the_editor_held_for_the_next_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Three standalone sessions stand for three `simul unreal exec` runs, each its own process's keeper."""
    records = tmp_path / "holds"
    monkeypatch.setattr(engine_holds, "_DEFAULT_HOLD_DIR", records)

    async def scenario() -> Tuple[_GameThreadEditor, List[str], UnrealError, Dict[str, Any], List[str]]:
        editor = _GameThreadEditor()
        await editor.start()
        settings = _settings(editor.port)
        try:
            first = UnrealRuntimeSession(settings)
            with pytest.raises(UnrealTimeoutError):
                await first.execute_script("import time; time.sleep(1.2)")
            await first.close()  # the command exits while the editor still runs the script
            left = sorted(path.name for path in records.glob("*.json"))
            second = UnrealRuntimeSession(settings)
            with pytest.raises(UnrealError) as refused:
                await second.execute_script("marker_b = 1")
            await second.close()
            await asyncio.sleep(1.3)  # the script ends
            third = UnrealRuntimeSession(settings)
            after = await third.execute_script("after = 1")
            await third.close()
            return editor, left, refused.value, after, sorted(path.name for path in records.glob("*.json"))
        finally:
            await editor.stop()

    editor, left, refused, after, finally_left = asyncio.run(scenario())

    assert len(left) == 1, "the timed-out command left its record for the next one"
    assert refused.error_type == "InstanceBusy" and refused.details["reason"] == "finishing_timed_out_call"
    assert "nothing was sent" in str(refused)
    assert not any("marker_b" in command for command in editor.commands()), "the refused call never ran"
    assert after == {"success": True, "output": "ran"}
    assert finally_left == [], "the call that saw the editor answer removed the record"


# ---------------------------------------------------------------------------
# m6: what a script wrote to stderr or logged as a warning
# ---------------------------------------------------------------------------


def _session_answering(
    log: List[Dict[str, str]], return_value: Any = True, result: Any = None
) -> UnrealRuntimeSession:
    session = UnrealRuntimeSession(Settings())

    async def execute(code: str, mode: str = "ExecuteFile") -> Dict[str, Any]:
        return {"ReturnValue": return_value, "CommandResult": result, "LogOutput": log}

    session._execute_python = execute  # type: ignore[method-assign]
    return session


def test_a_script_s_stderr_and_warnings_come_back_beside_its_output() -> None:
    session = _session_answering([
        {"Type": "Info", "Output": "out\n"},
        {"Type": "Error", "Output": "err\n"},
        {"Type": "Warning", "Output": "careful\n"},
    ])

    result = asyncio.run(session.execute_script("import sys; print('out'); sys.stderr.write('err\\n')"))

    assert result == {"success": True, "output": "out\n", "stderr": "err\ncareful\n"}


def test_a_script_without_stderr_has_no_stderr_key() -> None:
    session = _session_answering([{"Type": "Info", "Output": "out\n"}])

    assert asyncio.run(session.execute_script("print('out')")) == {"success": True, "output": "out\n"}


def test_a_printed_json_object_comes_back_as_printed() -> None:
    """The object is the script's data (#358): nothing is added to it, its own stderr key included."""
    session = _session_answering([
        {"Type": "Info", "Output": '{"value": 1, "stderr": "mine"}'},
        {"Type": "Warning", "Output": "careful\n"},
    ])

    result = asyncio.run(session.execute_script("print('{...}')"))

    assert result == {"value": 1, "stderr": "mine", "success": True}


def test_an_evaluated_statement_s_warnings_come_back_beside_its_result() -> None:
    session = _session_answering([{"Type": "Warning", "Output": "deprecated\n"}], result="3")

    result = asyncio.run(session.execute_script("1 + 2", mode="EvaluateStatement"))

    assert result == {"success": True, "result": "3", "output": "", "stderr": "deprecated\n"}


def test_each_stderr_entry_is_a_line() -> None:
    """Review nit: log_warning('careful') then log_error('bad thing') came back as 'carefulbad thing'."""
    session = _session_answering([
        {"Type": "Warning", "Output": "careful"},
        {"Type": "Error", "Output": "bad thing"},
        {"Type": "Error", "Output": "err\n"},
    ])

    assert asyncio.run(session.execute_script("..."))["stderr"] == "careful\nbad thing\nerr\n"


def test_a_script_that_raises_keeps_its_stderr_without_the_traceback() -> None:
    session = _session_answering(
        [
            {"Type": "Info", "Output": "step 1\n"},
            {"Type": "Warning", "Output": "careful"},
            {"Type": "Error", "Output": "Traceback (most recent call last):\nValueError: boom"},
        ],
        return_value=False,
        result="Traceback (most recent call last):\nValueError: boom",
    )

    result = asyncio.run(session.execute_script("..."))

    assert (result["success"], result["error_type"]) == (False, "ScriptError")
    assert result["output"] == "step 1\n" and result["stderr"] == "careful\n"
