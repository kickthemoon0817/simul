"""Only a script's own code writes into its captured stdout.

The executor redirected the process-wide ``sys.stdout`` for the whole life of
a coroutine script, awaited frames included. Anything else that printed while
the script waited (a physics callback, an OmniGraph script node, another
extension's update) landed in the script's output, and simul parses a tool
script's output as one JSON object: the reply broke for a reason that had
nothing to do with the tool. The capture now covers the synchronous part and
each coroutine step, never the time between steps.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from khemoo.simul.executor import ScriptExecutor


def _run(source: str, scope: dict[str, Any]) -> tuple[str, Any]:
    async def scenario() -> tuple[str, Any]:
        loop = asyncio.get_running_loop()
        gate: asyncio.Future = loop.create_future()
        scope["gate"] = gate

        def _foreign_callback() -> None:
            # Runs while the script is suspended at its await, the way a Kit
            # update callback runs between the frames a script waits on.
            print("FOREIGN physics callback output")
            gate.set_result(None)

        loop.call_later(0.05, _foreign_callback)
        executor = ScriptExecutor(scope, scope)
        output, exception, _trace = await executor.execute(source)
        return output, exception

    return asyncio.run(scenario())


def test_output_printed_by_other_code_during_an_await_is_not_captured(capsys: Any) -> None:
    script = (
        "import json\n"
        "print('before')\n"
        "await gate\n"
        "print(json.dumps({'ok': True}))\n"
    )

    output, exception = _run(script, {})

    assert exception is None
    assert "FOREIGN" not in output
    assert output.splitlines() == ["before", json.dumps({"ok": True})]
    # It still went somewhere: the real stdout.
    assert "FOREIGN physics callback output" in capsys.readouterr().out


def test_a_synchronous_script_is_still_captured_whole() -> None:
    async def scenario() -> str:
        executor = ScriptExecutor({}, {})
        output, exception, _ = await executor.execute("print('a')\nprint('b')")
        assert exception is None
        return output

    assert asyncio.run(scenario()) == "a\nb\n"


def test_output_from_every_step_of_a_multi_await_script_is_kept() -> None:
    async def scenario() -> str:
        executor = ScriptExecutor({}, {})
        output, exception, _ = await executor.execute(
            "import asyncio\n"
            "for i in range(3):\n"
            "    print('step', i)\n"
            "    await asyncio.sleep(0)\n"
            "print('done')\n"
        )
        assert exception is None
        return output

    assert asyncio.run(scenario()).splitlines() == ["step 0", "step 1", "step 2", "done"]
