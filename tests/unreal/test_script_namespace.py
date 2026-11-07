"""Every Python call simul sends runs in its own namespace, which is dropped when the call ends.

The fake editor below runs commands as UE 5.7's PythonScriptPlugin does
(``FPythonScriptPlugin::ExecPythonCommandEx``): literal code is evaluated in
one console dict shared by every call (``RunString``, ``PyConsoleGlobalDict``,
whatever ``FileExecutionScope`` says), ExecuteFile / ExecuteStatement /
EvaluateStatement compile as file, single and eval input, printed lines come
back as Info log output, a raise comes back as ``ReturnValue`` False with the
formatted traceback as ``CommandResult``, and an ExecuteFile command whose
first ``.py`` is followed by whitespace or ends it is taken for the path of a
file to run (``TryExtractPathnameAndCommand``).

A name left in that dict keeps its wrapper, and so its UObject, alive: the
next map change then aborts the editor with "World Memory Leaks" when the
object is the world's package (live-verified on UE 5.7.4).
"""

from __future__ import annotations

import base64
import builtins
import gc
import io
import json
import re
import sys
import traceback
import types
import weakref
from contextlib import asynccontextmanager, redirect_stdout
from types import SimpleNamespace
from typing import Any, AsyncIterator, Dict, List

import pytest
from aiohttp import web

from simul.adapters._unreal_python import (
    PYTHON_LIBRARY,
    RUNNER_MODULE,
    console_runs_python,
    isolate,
    python_command,
    runner_command,
    unwrap,
    unwrap_command,
)
from simul.adapters.unreal_runtime import UnrealRuntimeSession
from simul.config import Settings

MODES = {"ExecuteFile": "exec", "ExecuteStatement": "single", "EvaluateStatement": "eval"}


class Probe:
    """An object a script can hold, standing in for a UObject wrapper; ``alive`` counts live ones."""

    alive: "weakref.WeakSet[Probe]" = weakref.WeakSet()

    def __init__(self) -> None:
        Probe.alive.add(self)


class builtin_function_or_method_with_closure:  # noqa: N801 - the type name of Unreal's wrapper
    """Stands in for the wrapper UE 5.7 gives a static-library function or a struct method.

    Its weak-reference slot lies one pointer past the object, so Python's
    collector crashes the editor (_PyWeakref_ClearRef) when one is cyclic
    garbage. Each attribute access makes a new one.
    """

    alive: "weakref.WeakSet[builtin_function_or_method_with_closure]" = weakref.WeakSet()

    def __init__(self) -> None:
        type(self).alive.add(self)

    def __call__(self, *args: Any) -> bool:
        return True


LibraryFunction = builtin_function_or_method_with_closure


class ConsoleEditor:
    """Remote Control's Python library over one shared console dict, as UE runs it."""

    def __init__(self) -> None:
        self.unreal = SimpleNamespace(Probe=Probe, events=[], library_function=LibraryFunction)
        self.console: Dict[str, Any] = {"__name__": "__main__", "unreal": self.unreal}
        self.commands: List[Dict[str, Any]] = []

    async def __call__(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.commands.append(body)
        if body.get("functionName") != "ExecutePythonCommandEx":
            return web.json_response({"ReturnValue": "ok"})
        parameters = body["parameters"]
        command, mode = parameters["PythonCommand"], parameters.get("ExecutionMode", "ExecuteFile")
        if mode == "ExecuteFile":
            first = command.find(".py")
            if first >= 0 and (first + 3 == len(command) or command[first + 3].isspace()):
                path = command[: first + 3]
                return web.json_response({"ReturnValue": False, "LogOutput": [],
                                          "CommandResult": f"Could not load Python file '{path}'"})
        output = io.StringIO()
        try:
            with redirect_stdout(output):
                value = eval(compile(command, "<string>", MODES[mode]), self.console)  # noqa: S307 - the editor
        except Exception:
            return web.json_response({"ReturnValue": False, "LogOutput": [], "CommandResult": traceback.format_exc()})
        log = [{"Type": "Info", "Output": line} for line in output.getvalue().splitlines()]
        return web.json_response({"ReturnValue": True, "CommandResult": repr(value), "LogOutput": log})

    def names(self) -> List[str]:
        """What earlier calls left in the shared dict."""
        return sorted(k for k in self.console if not k.startswith("__") and k != "unreal")


@pytest.fixture(autouse=True)
def isolated_process(monkeypatch: pytest.MonkeyPatch) -> Any:
    """The runner installs itself in sys.modules and the fake's statements touch builtins._."""
    monkeypatch.delitem(sys.modules, RUNNER_MODULE, raising=False)
    had, previous = hasattr(builtins, "_"), getattr(builtins, "_", None)
    yield
    sys.modules.pop(RUNNER_MODULE, None)
    sys.__dict__.pop("_simul_unreal_kept_v1", None)
    if had:
        builtins._ = previous  # type: ignore[attr-defined]
    elif hasattr(builtins, "_"):
        del builtins._  # type: ignore[attr-defined]


@asynccontextmanager
async def editor(**unreal: Any) -> AsyncIterator[tuple]:
    fake = ConsoleEditor()
    sys.modules["unreal"] = fake.unreal  # type: ignore[assignment]

    async def route(request: web.Request) -> web.StreamResponse:
        return await fake(request)

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", route)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    base = Settings()
    settings = base.model_copy(update={
        "unreal": base.unreal.model_copy(update={
            "host": "127.0.0.1", "port": site._server.sockets[0].getsockname()[1], "timeout": 5.0, **unreal,
        }),
        "security": base.security.model_copy(update={"allow_script_execution": True}),
    })
    session = UnrealRuntimeSession(settings)
    try:
        yield session, fake
    finally:
        await session.close()
        await runner.cleanup()
        sys.modules.pop("unreal", None)


PACKAGE = "import json\npkg = unreal.Probe()\nworld = pkg\nprint(json.dumps({'bound': True}))\n"


# -- nothing a call binds outlives it --------------------------------------------


@pytest.mark.asyncio
async def test_a_top_level_binding_does_not_outlive_its_call() -> None:
    async with editor() as (session, fake):
        assert await session._execute_json_script(PACKAGE) == {"bound": True}

        # On main, pkg and world stayed in the shared dict: the next map
        # change then aborted UE 5.7.4 with "World Memory Leaks".
        assert fake.names() == []
        assert len(Probe.alive) == 0


@pytest.mark.asyncio
async def test_a_call_that_raises_after_binding_leaves_nothing_either() -> None:
    script = "pkg = unreal.Probe()\ndef keep():\n    return pkg\nraise RuntimeError('after binding')\n"
    async with editor() as (session, fake):
        result = await session.execute_script(script)

        assert result["success"] is False and result["error_type"] == "ScriptError"
        assert result["error"].rstrip().endswith("RuntimeError: after binding")
        assert fake.names() == []
        # keep() and the namespace form a cycle; clearing the namespace
        # breaks it, with no gc.collect().
        assert len(Probe.alive) == 0


@pytest.mark.asyncio
async def test_each_call_starts_fresh_with_unreal_already_imported() -> None:
    async with editor() as (session, fake):
        await session.execute_script("leftover = 1")
        second = await session.execute_script("print(unreal is not None)\nprint(leftover)")

        assert second["success"] is False
        assert "NameError: name 'leftover' is not defined" in second["error"]


@pytest.mark.asyncio
async def test_a_class_the_script_defines_is_collected_without_the_library_functions_it_held() -> None:
    """On main the class stayed in the shared dict; garbage now, it would crash the collector.

    The runner keeps the wrappers the class held, in attributes, containers,
    default arguments and closures, and the class goes without them.
    """
    script = (
        "class Helpers:\n"
        "    exists = unreal.library_function()\n"
        "    table = {'checks': [unreal.library_function()]}\n"
        "    package = unreal.Probe()\n"
        "    def check(self, path, test=unreal.library_function()):\n"
        "        return test(path)\n"
        "    @staticmethod\n"
        "    def nested(*, test=unreal.library_function()):\n"
        "        return test\n"
        "kept_alone = unreal.library_function()\n"
    )
    async with editor() as (session, fake):
        gc.collect()
        before = len(LibraryFunction.alive)
        result = await session.execute_script(script)
        # The next call runs the runner's source again, and a replaced runner
        # module goes; what was kept stays kept.
        await session.execute_script("pass")
        sys.modules[RUNNER_MODULE] = types.ModuleType(RUNNER_MODULE)
        await session.execute_script("pass")
        gc.collect()

        assert result["success"] is True and fake.names() == []
        assert len(Probe.alive) == 0  # the class and what else it held are gone
        # The four the class held stay; the top-level one went with the namespace.
        assert len(LibraryFunction.alive) == before + 4


@pytest.mark.asyncio
async def test_a_cycle_the_namespace_did_not_hold_is_garbage_for_the_collector() -> None:
    """A class keeps itself alive through its own __mro__; nothing reachable holds it.

    Python's collector frees it, and UE 5.7 runs that collector before every
    garbage collection of its own (FPythonScriptPlugin::OnPreGarbageCollect),
    so the runner needs no gc.collect().
    """
    script = "class Holder:\n    package = unreal.Probe()\n"
    async with editor() as (session, fake):
        await session.execute_script(script)

        assert fake.names() == []
        gc.collect()
        assert len(Probe.alive) == 0


# -- what callers get back is unchanged ----------------------------------------


@pytest.mark.asyncio
async def test_the_json_print_protocol_and_the_three_modes_are_unchanged() -> None:
    async with editor() as (session, fake):
        printed = await session.execute_script("import json\nprint('noise')\nprint(json.dumps({'count': 4}))")
        evaluated = await session.execute_script("40 + 2", mode="EvaluateStatement")
        statement = await session.execute_script("unreal.events.append('x') or 7", mode="ExecuteStatement")
        quiet = await session.execute_script("value = 3", mode="ExecuteStatement")

    assert printed == {"count": 4}
    assert evaluated == {"success": True, "result": "42", "output": ""}
    # A statement's value is printed, as the editor's console prints it, but
    # not kept in builtins._ (sys.displayhook's cache would hold it).
    assert statement == {"success": True, "output": "7"}
    assert quiet == {"success": True, "output": ""}
    assert getattr(builtins, "_", None) != 7
    assert fake.names() == []
    assert sys.displayhook is sys.__displayhook__


@pytest.mark.asyncio
async def test_the_namespace_holds_what_a_fresh_main_module_does() -> None:
    """``__builtins__`` the module, and ``__doc__``, ``__spec__``... as ``__main__`` has them.

    UE's console dict is a copy of ``__main__``'s, made when Python starts.
    """
    script = "print(__builtins__.len([1, 2]), __name__, sorted(k for k in globals() if k.startswith('__')))"
    spec = "__spec__ is __import__('sys').modules['__main__'].__spec__"
    async with editor() as (session, fake):
        result = await session.execute_script(script)
        same = await session.execute_script(spec, mode="EvaluateStatement")

    dunders = ["__builtins__", "__doc__", "__loader__", "__name__", "__package__", "__spec__"]
    assert result == {"success": True, "output": f"2 __main__ {dunders}"}
    assert same["result"] == "True"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["ExecuteFile", "ExecuteStatement"])
async def test_a_displayhook_the_script_installs_stays(mode: str) -> None:
    async with editor() as (session, fake):
        try:
            result = await session.execute_script(
                "import sys; sys.displayhook = __import__('unreal').events.append", mode=mode
            )
            assert result["success"] is True
            assert sys.displayhook == fake.unreal.events.append
        finally:
            sys.displayhook = sys.__displayhook__


@pytest.mark.asyncio
async def test_code_that_starts_with_a_byte_order_mark_runs() -> None:
    """A file saved as "UTF-8 with BOM" starts with U+FEFF; read_script_arg keeps it."""
    async with editor() as (session, fake):
        result = await session.execute_script("\ufeffprint('after the mark')")

    assert result == {"success": True, "output": "after the mark"}


@pytest.mark.asyncio
async def test_a_traceback_shows_the_command_and_the_runner_before_the_script() -> None:
    async with editor() as (session, fake):
        result = await session.execute_script("x = 1\nraise ValueError('on line two')")

    # What Unreal prints starts at the command; the fake's own frame comes first here.
    frames = [frame for frame in re.findall(r'File "([^"]+)", line (\d+)', result["error"]) if frame[0][0] == "<"]
    assert [name for name, _ in frames] == ["<string>", "<simul runner>", "<string>"]
    assert (frames[0][1], frames[-1][1]) == ("1", "2")
    assert result["error"].rstrip().endswith("ValueError: on line two")


@pytest.mark.asyncio
async def test_a_module_subclass_in_place_of_the_runner_is_replaced() -> None:
    """A ModuleType subclass's run property would survive the per-call refresh."""

    class Hijacked(types.ModuleType):
        @property
        def run(self) -> Any:
            return lambda mode, code, guard="": exec(base64.b64decode(code).decode(), fake.console)  # noqa: S102

    async with editor() as (session, fake):
        sys.modules[RUNNER_MODULE] = Hijacked(RUNNER_MODULE)
        result = await session.execute_script("pkg = unreal.Probe()")

        assert result["success"] is True
        assert type(sys.modules[RUNNER_MODULE]) is types.ModuleType
        assert fake.names() == [] and len(Probe.alive) == 0


@pytest.mark.asyncio
async def test_simul_not_found_from_a_tool_script_is_still_not_found() -> None:
    script = (
        "class SimulNotFound(Exception):\n    pass\n"
        "raise SimulNotFound('Actor not found: /Game/Map.Map:PersistentLevel.Gone')\n"
    )
    async with editor() as (session, fake):
        result = await session._execute_json_script(script)

    assert result == {
        "success": False,
        "error_type": "NotFoundError",
        "error": "Actor not found: /Game/Map.Map:PersistentLevel.Gone",
    }


@pytest.mark.asyncio
async def test_code_that_mentions_a_py_file_runs_as_code() -> None:
    """UE takes an ExecuteFile command whose first .py is followed by whitespace for a file path."""
    async with editor() as (session, fake):
        result = await session.execute_script("print('run my.py now')")

    assert result == {"success": True, "output": "run my.py now"}
    assert all(".py" not in body["parameters"]["PythonCommand"] for body in fake.commands)


@pytest.mark.asyncio
async def test_in_attached_mode_the_check_runs_first_in_the_same_namespace(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with editor(mode="attached", attachment_path=str(tmp_path / "attachment.json")) as (session, fake):
        monkeypatch.setattr(session, "_attachment_guard", lambda: "checked = unreal.Probe()\n")
        result = await session.execute_script("print(type(checked).__name__)")

    assert result == {"success": True, "output": "Probe"}
    assert fake.names() == [] and len(Probe.alive) == 0


# -- every route that runs Python is isolated -----------------------------------


@pytest.mark.asyncio
async def test_a_remote_control_call_to_the_python_library_is_isolated() -> None:
    async with editor() as (session, fake):
        await session._call_function(
            PYTHON_LIBRARY,
            "ExecutePythonCommandEx",
            {"PythonCommand": "pkg = unreal.Probe()", "ExecutionMode": "ExecuteFile"},
        )

    assert fake.commands and fake.names() == [] and len(Probe.alive) == 0


@pytest.mark.asyncio
async def test_call_actor_function_runs_nothing_on_the_python_library() -> None:
    """It takes actors of the level open in the editor only; the library is none."""
    async with editor() as (session, fake):
        result = await session.call_actor_function(
            PYTHON_LIBRARY,
            "ExecutePythonCommandEx",
            json.dumps({"PythonCommand": "pkg = unreal.Probe()", "ExecutionMode": "ExecuteFile"}),
        )

    assert result.get("success") is False
    assert not [body for body in fake.commands if "pkg" in json.dumps(unwrap(body))]
    assert fake.names() == [] and len(Probe.alive) == 0


# -- a py console command would run in the shared dict: refused -----------------

KISMET = "/Script/Engine.Default__KismetSystemLibrary"


@pytest.mark.parametrize(
    "command",
    ["py pkg = 1", "  PY\tpkg = 1", "py", "py.x", "py_x = 1", "stat fps|py pkg = 1", "stat fps\npy pkg = 1"],
)
def test_a_py_console_command_is_found_as_unreal_parses_it(command: str) -> None:
    # FParse::Command(&Cmd, "PY"): any case, after spaces and tabs, then
    # anything but a letter or digit; each "|" part and line is a command.
    assert console_runs_python(command)


@pytest.mark.parametrize("command", ["stat fps", "HighResShot 1920x1080", "pyramid 1", "py2", "r.py 1", ""])
def test_other_console_commands_are_not_python(command: str) -> None:
    assert not console_runs_python(command)


@pytest.mark.asyncio
async def test_a_py_console_command_is_refused_before_anything_is_sent() -> None:
    async with editor() as (session, fake):
        with pytest.raises(Exception) as raised:
            await session._call_function(
                KISMET, "ExecuteConsoleCommand", {"WorldContextObject": None, "Command": "py pkg = unreal.Probe()"}
            )
        await session._call_function(KISMET, "ExecuteConsoleCommand", {"Command": "stat fps"})

    assert getattr(raised.value, "error_type", None) == "ValidationError"
    assert "execute_unreal_script" in str(raised.value)
    assert [body["parameters"]["Command"] for body in fake.commands] == ["stat fps"]


@pytest.mark.asyncio
async def test_a_py_console_command_in_a_batch_is_refused_before_anything_is_sent() -> None:
    async with editor() as (session, fake):
        result = await session.batch_operations([{
            "URL": "/remote/object/call", "Verb": "PUT",
            "Body": {"objectPath": KISMET, "functionName": "executeconsolecommand",
                     "parameters": {"command": "stat fps | py pkg = unreal.Probe()"}},
        }])

    assert result["success"] is False and result["error_type"] == "ValidationError"
    assert "py console command" in result["error"]
    assert fake.commands == []


@pytest.mark.asyncio
async def test_a_python_call_inside_a_batch_is_isolated() -> None:
    sent: List[Dict[str, Any]] = []

    async def batch(request: web.Request) -> web.Response:
        sent.extend((await request.json())["Requests"])
        return web.json_response({"Responses": [{"RequestId": 1, "ResponseCode": 200, "ResponseBody": {}}]})

    async with editor() as (session, fake):
        session._session = None  # a fresh client on the batch-only app below
        app = web.Application()
        app.router.add_put("/remote/batch", batch)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        session._base_url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
        try:
            await session.batch_operations([{
                "URL": "/remote/object/call", "Verb": "PUT",
                "Body": {"objectPath": "/Script/PythonScriptPlugin.Default__PythonScriptLibrary",
                         "functionName": "ExecutePythonCommandEx",
                         "parameters": {"PythonCommand": "pkg = 1", "ExecutionMode": "ExecuteStatement"}},
            }])
        finally:
            await session.close()
            await runner.cleanup()

    (operation,) = sent
    assert unwrap_command(operation["Body"]["parameters"]["PythonCommand"]) == ("single", "pkg = 1", "")
    assert operation["Body"]["parameters"]["ExecutionMode"] == "ExecuteFile"


@pytest.mark.asyncio
async def test_an_unknown_execution_mode_is_refused_before_anything_is_sent() -> None:
    async with editor() as (session, fake):
        with pytest.raises(Exception) as raised:
            await session._call_function(
                "/Script/PythonScriptPlugin.Default__PythonScriptLibrary",
                "ExecutePythonCommandEx",
                {"PythonCommand": "1", "ExecutionMode": "Interactive"},
            )

    assert getattr(raised.value, "error_type", None) == "ValidationError"
    assert fake.commands == []


# -- the command itself -----------------------------------------------------------


@pytest.mark.parametrize("mode", ["exec", "single", "eval"])
def test_a_runner_command_round_trips_and_binds_nothing(mode: str) -> None:
    code = "pkg.py = 'x'\n" if mode != "eval" else "1 + 1"
    command = runner_command(code, mode, "guarded = True\n" if mode == "exec" else "")

    assert unwrap_command(command) == (mode, code, "guarded = True\n" if mode == "exec" else "")
    assert ".py" not in command and "\n" not in command
    # One expression: valid as an ExecuteFile command and as an EvaluateStatement one.
    compile(command, "<string>", "eval")


def test_python_command_recognises_only_calls_that_run_source() -> None:
    library = "/Script/PythonScriptPlugin.Default__PythonScriptLibrary"
    call = {"objectPath": library, "functionName": "ExecutePythonCommandEx",
            "parameters": {"PythonCommand": "1", "ExecutionMode": "EPythonCommandExecutionMode::EvaluateStatement"}}

    assert python_command("/remote/object/call", call) == ("PythonCommand", "1", "eval")
    assert python_command("/remote/object/call", {**call, "functionName": "ExecutePythonCommand",
                                                  "parameters": {"PythonCommand": "x = 1"}}) == (
        "PythonCommand", "x = 1", "exec")
    assert python_command("/remote/object/call", {**call, "functionName": "ExecutePythonScript"}) is None
    assert python_command("/remote/object/call", {**call, "objectPath": "/Game/Map.Map:PersistentLevel.A"}) is None
    assert python_command("/remote/object/property", call) is None


def test_unwrap_restores_what_isolate_sent() -> None:
    body = {"objectPath": "/Script/PythonScriptPlugin.Default__PythonScriptLibrary",
            "functionName": "ExecutePythonCommandEx",
            "parameters": {"PythonCommand": "x", "ExecutionMode": "ExecuteStatement", "FileExecutionScope": "Public"}}

    sent = isolate(body, "check()\n")

    assert sent["parameters"]["ExecutionMode"] == "ExecuteFile"
    assert unwrap(sent) == {**body, "parameters": {**body["parameters"], "PythonCommand": "check()\nx"}}
