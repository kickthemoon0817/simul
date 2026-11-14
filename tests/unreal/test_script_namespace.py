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
from unittest.mock import AsyncMock

import pytest
from aiohttp import web

from simul.adapters._unreal_python import (
    PYTHON_LIBRARY,
    RUNNER_MODULE,
    console_runs_python,
    isolate,
    isolate_request,
    python_command,
    remote_route,
    runner_command,
    unisolated_commands,
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
    default arguments and closures, and the class goes without them. The
    collector never sweeps a kept wrapper: the dict that keeps it hangs off
    sys and survives the runner's refresh and its replacement.
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
    )
    gc.collect()
    before = len(LibraryFunction.alive)
    gc.disable()  # the class is collected below, when the test says so
    try:
        async with editor() as (session, fake):
            result = await session.execute_script(script)
            # The runner's source runs again, and a replaced runner module goes.
            sys.modules[RUNNER_MODULE] = types.ModuleType(RUNNER_MODULE)
            await session.execute_script("pass")
            gc.collect()

            assert result["success"] is True and fake.names() == []
            assert len(Probe.alive) == 0  # the class and what else it held are gone
            # Swept by the collector they would crash the editor; only the dict holds them now.
            assert len(sys._simul_unreal_kept_v1) == 4  # type: ignore[attr-defined]
            assert len(LibraryFunction.alive) == before + 4
    finally:
        gc.enable()


@pytest.mark.asyncio
async def test_a_class_bound_by_no_name_is_kept_safe_too() -> None:
    """Classes appended to a list in a loop, made with type(), or reached as an object's class.

    Only classes bound by name were walked: the others were swept with the
    wrapper they held, and the editor crashed at its next garbage collection.
    """
    script = (
        "holders = []\n"
        "for i in range(3):\n"
        "    class H:\n"
        "        exists = unreal.library_function()\n"
        "        package = unreal.Probe()\n"
        "    holders.append(H)\n"
        "made = [type('T', (), {'exists': unreal.library_function()})]\n"
        "class Node:\n"
        "    def __init__(self):\n"
        "        self.me = self\n"
        "        self.check = unreal.library_function()\n"
        "nodes = {'a': Node()}\n"
    )
    gc.collect()
    before = len(LibraryFunction.alive)
    gc.disable()
    try:
        async with editor() as (session, fake):
            result = await session.execute_script(script)
            gc.collect()

            assert result["success"] is True and fake.names() == []
            assert len(Probe.alive) == 0
            # The three loop classes', type()'s and the node's: all kept from the sweep.
            assert len(LibraryFunction.alive) == before + 5
            assert len(sys._simul_unreal_kept_v1) == 5  # type: ignore[attr-defined]
    finally:
        gc.enable()


@pytest.mark.asyncio
async def test_kept_wrappers_are_let_go_once_their_class_is_gone() -> None:
    """Each attribute access makes a new wrapper: kept forever, they grew with every call."""
    script = "".join(
        f"class C{n}:\n    a = unreal.library_function()\n    b = unreal.library_function()\n" for n in range(5)
    )
    async with editor() as (session, fake):
        sizes = []
        for _ in range(20):
            await session.execute_script(script)
            gc.collect()  # UE runs Python's collector before each of its own
            sizes.append(len(sys._simul_unreal_kept_v1))  # type: ignore[attr-defined]
        await session.execute_script("pass")
        gc.collect()

        # Each call keeps the ten its classes held and lets go of the last call's.
        assert sizes == [10] * 20
        assert len(sys._simul_unreal_kept_v1) == 0  # type: ignore[attr-defined]
        assert len(LibraryFunction.alive) == 0


@pytest.mark.asyncio
async def test_a_wrapper_still_held_elsewhere_stays_kept() -> None:
    """A class a callback keeps alive keeps its wrapper; letting go of it would be early, not unsafe."""
    script = (
        "import sys, types\n"
        "class Live:\n"
        "    exists = unreal.library_function()\n"
        "registry = types.ModuleType('r336_registry')\n"
        "registry.live = Live\n"
        "sys.modules['r336_registry'] = registry\n"
    )
    async with editor() as (session, fake):
        try:
            await session.execute_script(script)
            await session.execute_script("pass")
            gc.collect()

            assert len(sys._simul_unreal_kept_v1) == 1  # type: ignore[attr-defined]
        finally:
            sys.modules.pop("r336_registry", None)
        await session.execute_script("pass")
        gc.collect()
        await session.execute_script("pass")

        assert len(sys._simul_unreal_kept_v1) == 0  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_the_runner_is_compiled_once_per_module() -> None:
    async with editor() as (session, fake):
        await session.execute_script("pass")
        first = [v for k, v in vars(sys.modules[RUNNER_MODULE]).items() if k.startswith("_code_")]
        await session.execute_script("pass")
        second = [v for k, v in vars(sys.modules[RUNNER_MODULE]).items() if k.startswith("_code_")]

    assert len(first) == len(second) == 1 and second[0] is first[0]


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

    assert printed == {"count": 4, "success": True}
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


# -- every Python command on the wire runs in the runner, in a batch too ---------


def _python_in_commands(value: Any) -> List[str]:
    """Every PythonCommand string anywhere in a request body, at any depth, whatever calls it."""
    if isinstance(value, dict):
        found = [v for k, v in value.items() if str(k).lower() == "pythoncommand" and isinstance(v, str)]
        return found + [c for v in value.values() for c in _python_in_commands(v)]
    if isinstance(value, list):
        return [c for v in value for c in _python_in_commands(v)]
    return []


@asynccontextmanager
async def recording_batch(session: UnrealRuntimeSession) -> AsyncIterator[List[Dict[str, Any]]]:
    """Point ``session`` at a Remote Control that records every body and answers each batch part with 200."""
    sent: List[Dict[str, Any]] = []

    async def answer(request: web.Request) -> web.Response:
        body = await request.json()
        sent.append({"path": request.path, "body": body})
        parts = body.get("Requests", []) if request.path == "/remote/batch" else []
        return web.json_response({"Responses": [
            {"RequestId": part["RequestId"], "ResponseCode": 200,
             "ResponseBody": {"ReturnValue": True, "LogOutput": [{"Type": "Info", "Output": "{}"}]}}
            for part in parts
        ], "ReturnValue": True, "LogOutput": []})

    session._session = None
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", answer)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    session._base_url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    try:
        yield sent
    finally:
        await session.close()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_set_actor_property_sends_no_python_past_the_runner() -> None:
    """Round 2 B1, checked on the wire: the property write's batch embedded its value in raw ExecuteFile
    commands, and _http_request isolated only /remote/object/call."""
    async with editor() as (session, fake):
        async with recording_batch(session) as sent:
            session._execute_json_script = AsyncMock(  # type: ignore[method-assign]
                return_value={"target_class": "StaticMeshActor", "container": "array", "replaceable": True})
            await session.set_actor_property("/Game/M.M:PersistentLevel.A", "Tags", '["see setup.py first"]')

    assert [request for request in sent if request["path"] == "/remote/batch"], sent
    commands = [command for request in sent for command in _python_in_commands(request["body"])]
    # Each is a runner command, whose source travels base64-encoded: the
    # value's "setup.py first" never reaches Unreal's ExecuteFile parsing.
    assert commands and all(unwrap_command(command) is not None for command in commands)
    assert not any(".py" in command for command in commands)
    assert any("see setup.py first" in unwrap_command(command)[1] for command in commands)  # type: ignore[index]


@pytest.mark.asyncio
async def test_any_batch_the_session_sends_has_its_python_isolated() -> None:
    call = {"objectPath": PYTHON_LIBRARY, "functionName": "ExecutePythonCommandEx",
            "parameters": {"PythonCommand": "print('see my.py first')", "ExecutionMode": "ExecuteFile"}}
    async with editor() as (session, fake):
        async with recording_batch(session) as sent:
            await session._http_put("/remote/batch", {"Requests": [
                {"RequestId": 1, "URL": "/remote/object/call?x=1", "Verb": "PUT", "Body": call},
                {"RequestId": 2, "URL": "/REMOTE/OBJECT/CALL/", "Verb": "PUT", "Body": call},
                {"RequestId": 3, "URL": "/remote/object/property", "Verb": "PUT", "Body": {"objectPath": "a"}},
            ]})

    (request,) = sent
    parts = request["body"]["Requests"]
    for part in parts[:2]:
        assert unwrap_command(part["Body"]["parameters"]["PythonCommand"]) == (
            "exec", "print('see my.py first')", "")
    assert parts[2]["Body"] == {"objectPath": "a"}


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/remote/object/call", "/remote/batch"])
async def test_a_python_command_to_anything_but_the_library_is_never_sent(path: str) -> None:
    """A PythonCommand python_command does not recognise would go out as it is: refused instead."""
    call = {"objectPath": "/Script/PythonScriptPlugin.PythonScriptLibrary", "functionName": "ExecutePythonCommandEx",
            "parameters": {"PythonCommand": "pkg = unreal.Probe()"}}
    body = call if path == "/remote/object/call" else {
        "Requests": [{"RequestId": 1, "URL": "/remote/object/call", "Verb": "PUT", "Body": call}]}
    async with editor() as (session, fake):
        async with recording_batch(session) as sent:
            with pytest.raises(Exception) as raised:
                await session._http_put(path, body)

    assert getattr(raised.value, "error_type", None) == "ValidationError"
    assert raised.value.details["code"] == "python_outside_runner"  # type: ignore[attr-defined]
    assert sent == []


@pytest.mark.asyncio
async def test_batch_operations_refuses_a_python_command_to_anything_but_the_library() -> None:
    async with editor() as (session, fake):
        result = await session.batch_operations([{
            "URL": "/remote/object/call", "Verb": "PUT",
            "Body": {"objectPath": "/Game/M.M:PersistentLevel.A", "functionName": "ExecutePythonCommand",
                     "parameters": {"PythonCommand": "pkg = 1"}},
        }])

    assert (result["success"], result["error_type"]) == (False, "ValidationError")
    assert "operation 0 passes a PythonCommand" in result["error"]
    assert fake.commands == []


@pytest.mark.asyncio
@pytest.mark.parametrize("batched", [False, True])
async def test_a_python_command_to_a_preset_function_is_never_sent(batched: bool) -> None:
    """Review of #355: a preset can expose ExecutePythonCommandEx; its route names no object to recognise."""
    route = "/remote/preset/Tools/function/RunPython"
    call = {"Parameters": {"PythonCommand": "pkg = unreal.Probe()"}, "GenerateTransaction": False}
    path, body = (route, call) if not batched else (
        "/remote/batch", {"Requests": [{"RequestId": 1, "URL": route + "?x=1", "Verb": "PUT", "Body": call}]})
    async with editor() as (session, fake):
        async with recording_batch(session) as sent:
            with pytest.raises(Exception) as raised:
                await session._http_put(path, body)
            # A preset function that passes no PythonCommand goes out as it is.
            await session._http_put(route, {"Parameters": {"Speed": 2}})

    assert raised.value.details["code"] == "python_outside_runner"  # type: ignore[attr-defined]
    assert sent == [{"path": route, "body": {"Parameters": {"Speed": 2}}}]
    assert unisolated_commands("/remote/preset/Tools/property/Speed", {"Parameters": {"PythonCommand": "x"}}) == []


def test_isolate_request_leaves_other_requests_alone_and_refuses_what_python_command_refuses() -> None:
    body = {"objectPath": "/Game/M.M:PersistentLevel.A", "propertyName": "Tags"}
    assert isolate_request("/remote/object/property", body) is body
    assert isolate_request("/remote/batch", {"Requests": "not a list"}) == {"Requests": "not a list"}
    console = {"objectPath": KISMET, "functionName": "ExecuteConsoleCommand",
               "parameters": {"Command": "py pkg = 1"}}
    with pytest.raises(ValueError, match="operation 1: A py console command"):
        isolate_request("/remote/batch", {"Requests": [
            {"URL": "/remote/object/property", "Body": body},
            {"URL": "/remote/object/call", "Body": console},
        ]})
    assert unisolated_commands("/remote/object/property", {"parameters": {"PythonCommand": "x"}}) == []


# -- Unreal's log is written to disk once per call, not once per line ------------


class UnrealLogger(io.BufferedIOBase):
    """unreal_core._Logger: each write is one log entry; flush writes the whole log to disk (GLog->Flush)."""

    def __init__(self, kind: str, entries: List[Any], flushes: List[str]) -> None:
        self.log_func = lambda text: entries.append((kind, text))
        self.flush_func = lambda: flushes.append(kind)

    def writable(self) -> bool:
        return True

    def write(self, b: Any) -> None:  # type: ignore[override]
        self.log_func(bytes(b).decode("utf-8"))

    def flush(self) -> None:
        self.flush_func()


@pytest.fixture
def unreal_streams(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Installs sys.stdout and sys.stderr as unreal_core makes them; returns (entries, flushes, stdout, stderr).

    Called from the test body: pytest's own capture swaps sys.stdout back in
    after fixtures are set up.
    """

    def install() -> Any:
        entries: List[Any] = []
        flushes: List[str] = []
        stdout = io.TextIOWrapper(UnrealLogger("Info", entries, flushes), encoding="utf-8", line_buffering=True)
        stderr = io.TextIOWrapper(UnrealLogger("Error", entries, flushes), encoding="utf-8", line_buffering=True)
        monkeypatch.setattr(sys, "stdout", stdout)
        monkeypatch.setattr(sys, "stderr", stderr)
        monkeypatch.setitem(sys.modules, "unreal", SimpleNamespace(log=lambda text: entries.append(("Info", text))))
        return entries, flushes, stdout, stderr

    return install


SCRIPT = ("import sys\nfor i in range(3):\n    print(i)\nunreal.log('direct')\nsys.stderr.write('e\\n')\n"
          "print({'a': 1}, end='')\nprint()\nsys.stdout.write('tail')\n")


def test_the_runner_makes_the_same_log_entries_and_writes_the_log_once(unreal_streams: Any) -> None:
    entries, flushes, stdout, stderr = unreal_streams()
    exec(compile(SCRIPT, "<string>", "exec"), {"unreal": sys.modules["unreal"]})  # noqa: S102 - as Unreal ran it
    stdout.flush()
    plain, plain_flushes = list(entries), len(flushes)
    entries.clear()
    flushes.clear()

    exec(runner_command(SCRIPT, "exec"), {})  # noqa: S102 - the command the editor runs

    # The same entries in the same order, the unfinished last line included;
    # Unreal flushed its log to disk after every line, the runner once.
    assert entries == plain
    assert plain_flushes >= 5 and len(flushes) == 1  # either stream's flush writes the whole log
    assert sys.stdout is stdout and sys.stderr is stderr


def test_a_raise_still_logs_what_was_printed_and_restores_the_streams(unreal_streams: Any) -> None:
    entries, flushes, stdout, stderr = unreal_streams()

    with pytest.raises(RuntimeError, match="boom"):
        exec(runner_command("print('before')\nraise RuntimeError('boom')\n", "exec"), {})  # noqa: S102

    assert entries == [("Info", "before\n")] and flushes == ["Info"]
    assert sys.stdout is stdout and sys.stderr is stderr


def test_a_call_that_prints_nothing_does_not_write_the_log(unreal_streams: Any) -> None:
    entries, flushes, stdout, stderr = unreal_streams()

    assert eval(runner_command("1 + 1", "eval"), {}) == 2  # noqa: S307 - the command the editor runs

    assert entries == [] and flushes == []


def test_a_stream_the_script_installs_stays_and_others_are_left_alone(unreal_streams: Any) -> None:
    entries, flushes, stdout, stderr = unreal_streams()
    exec(runner_command("import io, sys\nsys.stdout = io.StringIO()\n", "exec"), {})  # noqa: S102

    assert isinstance(sys.stdout, io.StringIO) and sys.stderr is stderr
    mine = sys.stdout
    exec(runner_command("print('kept')\n", "exec"), {})  # noqa: S102

    assert mine.getvalue() == "kept\n" and entries == []


DETACH = "import io, sys\nsys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')\nprint('mine')\n"


def test_a_script_that_rewraps_stdout_succeeds_and_keeps_its_output_in_its_call(unreal_streams: Any) -> None:
    """Review of #355: after sys.stdout.detach() the swapped stream had no buffer, and the restore raised
    AttributeError: a call that ran reported a ScriptError, its namespace was not cleared and its output
    came out in the next call."""
    entries, flushes, stdout, stderr = unreal_streams()
    namespace: Dict[str, Any] = {}

    exec(runner_command(DETACH, "exec"), namespace)  # noqa: S102 - must not raise

    assert entries == [("Info", "mine\n")] and len(flushes) == 1
    # The stream the script installed stays (as the stream a script installs always does).
    assert sys.stdout is not stdout and sys.stderr is stderr
    entries.clear()
    flushes.clear()

    exec(runner_command("import sys\nprint('next')\nsys.stdout.write('tail')\n", "exec"), {})  # noqa: S102

    # The next call's output is its own, and Unreal's log is written once for it.
    assert entries == [("Info", "next\n"), ("Info", "tail")] and len(flushes) == 1


def test_the_namespace_is_cleared_whatever_the_script_did_to_the_streams(unreal_streams: Any) -> None:
    unreal_streams()
    sys.modules["unreal"].Probe = Probe  # type: ignore[attr-defined]

    exec(runner_command(DETACH + "sys.stderr = None\npkg = unreal.Probe()\n", "exec"), {})  # noqa: S102

    assert len(Probe.alive) == 0 and sys.stderr is None


# -- a path below a route is that route ------------------------------------------


@pytest.mark.parametrize(
    ("url", "route"),
    [
        ("/remote/object/call", "/remote/object/call"),
        ("/Remote/Object/Call/x?y=1", "/remote/object/call"),
        ("/remote/object/call/x/y/", "/remote/object/call"),
        ("/remote/batch/x", "/remote/batch"),
        ("/remote/preset/Tools/function/RunPython/x", "/remote/preset/tools/function/runpython"),
        ("/remote/object/callx", "/remote/object/callx"),
        ("/remote/object/property/x", "/remote/object/property/x"),
    ],
)
def test_remote_route_folds_a_path_below_a_route_into_it(url: str, route: str) -> None:
    """Re-review of #355: Unreal hands /remote/object/call/x to the object call handler and /remote/batch/x
    to the batch handler, which never read the rest of the path; simul compared routes exactly."""
    assert remote_route(url) == route


def _library_call(code: str) -> Dict[str, Any]:
    return {"objectPath": PYTHON_LIBRARY, "functionName": "ExecutePythonCommandEx",
            "parameters": {"PythonCommand": code, "ExecutionMode": "ExecuteFile"}}


@pytest.mark.asyncio
async def test_python_sent_to_a_path_below_the_object_call_route_runs_in_the_runner() -> None:
    async with editor() as (session, fake):
        async with recording_batch(session) as sent:
            result = await session.batch_operations([
                {"URL": "/remote/object/call/x", "Verb": "PUT", "Body": _library_call("print('see setup.py first')")},
            ])

    assert result.get("success", True) is True, result
    (request,) = sent
    (part,) = request["body"]["Requests"]
    assert unwrap_command(part["Body"]["parameters"]["PythonCommand"]) == ("exec", "print('see setup.py first')", "")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (
            {"URL": "/remote/object/call/x", "Verb": "PUT",
             "Body": {"objectPath": KISMET, "functionName": "ExecuteConsoleCommand",
                      "parameters": {"WorldContextObject": "", "Command": "py pkg = 1"}}},
            "py console command",
        ),
        (
            {"URL": "/remote/batch/x", "Verb": "PUT",
             "Body": {"Requests": [{"RequestId": 7, "URL": "/remote/object/call", "Verb": "PUT",
                                    "Body": _library_call("pkg = 1")}]}},
            "a batch cannot contain another",
        ),
        (
            {"URL": "/remote/preset/Tools/function/RunPython/x", "Verb": "PUT",
             "Body": {"Parameters": {"PythonCommand": "pkg = 1"}}},
            "passes a PythonCommand",
        ),
    ],
)
async def test_a_path_below_a_route_is_refused_as_the_route_is(operation: Dict[str, Any], message: str) -> None:
    async with editor() as (session, fake):
        async with recording_batch(session) as sent:
            result = await session.batch_operations([operation])

    assert (result["success"], result["error_type"]) == (False, "ValidationError")
    assert message in result["error"] and sent == []


@pytest.mark.asyncio
async def test_raw_python_in_a_batch_nested_below_the_batch_route_is_never_sent() -> None:
    nested = {"Requests": [{"RequestId": 1, "URL": "/remote/batch/x", "Verb": "PUT", "Body": {"Requests": [
        {"RequestId": 7, "URL": "/remote/object/call/y", "Verb": "PUT", "Body": _library_call("pkg = 1")}]}}]}
    async with editor() as (session, fake):
        async with recording_batch(session) as sent:
            with pytest.raises(Exception) as raised:
                await session._http_put("/remote/batch", nested)

    assert raised.value.details["code"] == "python_outside_runner"  # type: ignore[attr-defined]
    assert sent == []
