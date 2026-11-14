"""The round-4 Unreal audit's minors (m2, m3, m4, m5, m6).

* m2: a ``simul unreal exec`` timeout promised a hold that the short-lived
  command released as it exited, so the next command queued behind the
  timed-out script. Its record is now handed over to the editor endpoint and
  bounded by its expiry; the next call ends it once the editor answers.
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
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from simul.adapters.unreal_runtime import UnrealError, UnrealRuntimeSession, UnrealTimeoutError
from simul.config import Settings
from simul.utils import engine_holds
from simul.utils.engine_holds import SharedHolds, endpoint_key
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


def test_a_handed_over_record_outlives_its_process_until_it_expires(tmp_path: Path) -> None:
    keeper, reader = SharedHolds("unreal", tmp_path), SharedHolds("unreal", tmp_path)
    key = endpoint_key("localhost", 30200)
    since = time.time()
    record = keeper.publish(key, "simul unreal exec", since, 60.0, live=lambda: True,
                            details={"timed_out_route": "/remote/object/call", "changes_state": True})
    keeper.hand_over(record)

    held = reader.foreign(key)
    assert held is not None and held.handed_over is True
    assert held.info()["held_by"] == {"pid": os.getpid(), "exited": True}
    # Its process no longer counts: a record naming a process that is gone is still live.
    data = engine_holds.BridgeFiles.read(record)
    engine_holds.BridgeFiles.write(record, {**data, "pid": 2 ** 22 + 12345})
    assert reader.foreign(key) is not None
    # An answer to a request sent before the call timed out does not end it; one sent after does.
    assert reader.clear_handed_over(key, since - 1.0) == 0 and reader.foreign(key) is not None
    assert reader.clear_handed_over(key, since + 1.0) == 1 and reader.foreign(key) is None


def test_a_handed_over_record_past_its_expiry_is_ignored(tmp_path: Path) -> None:
    keeper, reader = SharedHolds("unreal", tmp_path), SharedHolds("unreal", tmp_path)
    key = endpoint_key("127.0.0.1", 30200)
    record = keeper.publish(key, "exec", time.time() - 10.0, 5.0, live=lambda: True)
    keeper.hand_over(record)  # already expired: withdrawn rather than handed over

    assert reader.foreign(key) is None and not list(tmp_path.glob("*.json"))


def test_a_handed_over_hold_names_the_exited_command_in_its_refusal(tmp_path: Path) -> None:
    from simul.adapters._unreal_hold import EditorHolds

    keeper = SharedHolds("unreal", tmp_path)
    record = keeper.publish(endpoint_key("127.0.0.1", 30200), "PUT /remote/object/call",
                            time.time(), 60.0, live=lambda: True)
    keeper.hand_over(record)

    refusal = EditorHolds(60.0, SharedHolds("unreal", tmp_path)).refusal("127.0.0.1:30200")

    assert refusal is not None and refusal.error_type == "InstanceBusy"
    assert "a simul command that has exited" in str(refusal)
    assert refusal.details["held_by"]["exited"] is True


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
