"""Every Python call simul sends to Unreal runs in a namespace of its own.

UE 5.7 runs literal Python from ``PythonScriptLibrary.ExecutePythonCommandEx``
in one dict shared by every call and by the editor's Python console
(``FPythonScriptPlugin::RunString`` evaluates in ``PyConsoleGlobalDict``;
``FileExecutionScope`` applies only to a ``.py`` file, never to literal code).
So a name a script binds at its top level outlives the call, and a wrapper it
holds keeps its UObject reachable for Unreal's garbage collector
(``FPyReferenceCollector``). When the editor tears down a map it purges the
wrappers of the world and the objects inside it
(``OnPrepareToCleanseEditorObject``, ``ForEachObjectWithOuter``) but not
those of the world's package or of objects outside it that reference it, such
as an ``AssetExportTask``. One kept ``pkg = world.get_outermost()`` makes the
next map load or ``new_blank_map`` abort the editor with "World Memory Leaks:
Old Package ... not cleaned up by GC", losing unsaved work.

Each call therefore goes through ``RUNNER_SOURCE``'s ``run``, installed as a
module in ``sys.modules`` and refreshed on every call. ``run`` compiles the
code into a fresh dict holding what a fresh ``__main__`` holds
(``__builtins__`` the module, ``__name__``, ``__doc__``, ``__spec__``...) and
the ``unreal`` module, and clears that dict when the call ends, returned or
raised. The command
itself is one expression that binds nothing in the shared dict. Clearing
breaks the usual dict-to-function cycles at once; any cycle left (a class
holding an object, say) is unreachable and is collected by Python's
collector, which the plugin runs before every Unreal garbage collection
(``FPythonScriptPlugin::OnPreGarbageCollect``), so no ``gc.collect()`` is
needed here. In statement mode ``sys.displayhook`` would keep the last
expression's value in ``builtins._``, so ``run`` swaps in one that only
prints it.

That collection is also where UE 5.7 crashes on a library-function wrapper
(``builtin_function_or_method_with_closure``, whose weak-reference slot lies
one pointer past the object) that is part of the garbage. A class is always
in a cycle, and a class the script defined used to stay in the shared dict,
so one holding such a function was never collected. ``run`` therefore keeps
the wrappers its namespace reaches when the call ends: through the classes
there, bound by name or not (in a list, as an object's class), their
attributes, containers, methods' default arguments and closures. The classes
are collected without them, and the next call drops each wrapper nothing
else holds any more, which frees it by reference count, outside the collector.

A ``py`` console command (``KismetSystemLibrary.ExecuteConsoleCommand``)
runs its code in the shared dict, past the runner, so ``python_command``
refuses one.

The source travels base64-encoded: UE treats an ``ExecuteFile`` command whose
first ``.py`` is followed by whitespace as the path of a file to run, and the
encoding (and the runner, which never spells it) keeps that from ever
matching.
"""

from __future__ import annotations

import base64
import hashlib
import re
from typing import Any, Dict, Optional, Tuple

#: The module ``run`` lives in, in the editor's ``sys.modules``.
RUNNER_MODULE = "_simul_unreal_runner_v1"

#: Defines ``run(mode, code, guard)``: ``mode`` is the ``compile`` mode
#: (exec, single or eval), ``code`` and ``guard`` base64 UTF-8 source. The
#: guard (the attachment check, or "") runs first, in the same namespace.
#: Returns the expression's value in eval mode, else None. Its exec and eval
#: run exactly the source the request carried, which Unreal would otherwise
#: run itself: this is the execution path, not a new one, and the same
#: script-execution policy gates what reaches it.
RUNNER_SOURCE = '''
import base64, builtins, itertools, sys, types

def _source(encoded):
    # utf-8-sig: a file saved as "UTF-8 with BOM" starts with U+FEFF, which
    # compile() refuses in a str; Unreal, compiling bytes, skipped it.
    return base64.b64decode(encoded).decode("utf-8-sig")

def _display(value):
    if value is not None:
        print(repr(value))

def _namespace():
    # What a fresh __main__ holds, as the console dict Unreal copied from it.
    main = vars(sys.modules["__main__"]) if "__main__" in sys.modules else {}
    namespace = {key: main.get(key) for key in ("__doc__", "__package__", "__loader__", "__spec__")}
    namespace.update(__name__="__main__", __builtins__=builtins, unreal=sys.modules["unreal"])
    return namespace

# The library-function wrappers the script's classes held, kept so that
# Python's collector never sweeps one. The dict hangs off sys, in no cycle:
# this source runs again on every call, and this module, whose functions make
# its dict a cycle, may be replaced; dropped, the dict would free the
# wrappers by reference count, which is safe.
_kept = sys.__dict__.setdefault("_simul_unreal_kept_v1", {})

def _release_kept():
    # A wrapper only the dict holds (getrefcount counts the dict and its own
    # argument) belonged to a class the collector has freed: dropping it now
    # frees it by reference count. Each attribute access makes a new wrapper,
    # so without this the dict grew with every call that defined a class.
    for key in list(_kept):
        if sys.getrefcount(_kept[key]) == 2:
            del _kept[key]

def _unsafe(value):
    kind = type(value)
    return kind.__name__ == "builtin_function_or_method_with_closure" or (
        0 < kind.__weakrefoffset__ >= kind.__basicsize__ and not kind.__itemsize__)

def _keep_unsafe(namespace, budget=10000):
    # From everything the call left, so a class bound by no name (appended to
    # a list in a loop, made with type(), an object's class) counts too.
    pending = [value for key, value in namespace.items() if key not in ("__builtins__", "unreal")]
    seen = set()
    while pending and len(seen) < budget:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if _unsafe(value):
            _kept[id(value)] = value
        elif isinstance(value, type):
            if value.__module__ == "__main__":
                pending.extend(vars(value).values())
        elif isinstance(value, (staticmethod, classmethod)):
            pending.append(value.__func__)
        elif isinstance(value, property):
            pending.extend((value.fget, value.fset, value.fdel))
        elif isinstance(value, types.FunctionType):
            pending.extend(value.__defaults__ or ())
            pending.extend((value.__kwdefaults__ or {}).values())
            for cell in value.__closure__ or ():
                try:
                    pending.append(cell.cell_contents)
                except ValueError:
                    pass
        elif isinstance(value, (list, tuple, set, frozenset)):
            pending.extend(itertools.islice(value, budget))
        elif isinstance(value, dict):
            pending.extend(itertools.islice(value.values(), budget))
        elif type(value).__module__ == "__main__" and isinstance(getattr(value, "__dict__", None), dict):
            pending.extend(itertools.islice(value.__dict__.values(), budget))

def run(mode, code, guard=""):
    _release_kept()
    namespace = _namespace()
    try:
        if guard:
            exec(compile(_source(guard), "<string>", "exec"), namespace)
        if mode == "eval":
            return eval(compile(_source(code), "<string>", "eval"), namespace)
        displayhook = sys.displayhook
        if mode == "single":
            sys.displayhook = _display
        try:
            exec(compile(_source(code), "<string>", mode), namespace)
        finally:
            # A hook the script installed itself stays.
            if sys.displayhook is _display:
                sys.displayhook = displayhook
    finally:
        try:
            _keep_unsafe(namespace)
        except Exception:
            pass  # a guard against an engine crash, never a reason to fail the call
        namespace.clear()
'''

#: Where the runner module keeps RUNNER_SOURCE compiled; the digest makes a
#: new source compile anew.
_RUNNER_CODE_KEY = "_code_" + hashlib.sha256(RUNNER_SOURCE.encode("utf-8")).hexdigest()[:16]

#: The Python library Remote Control calls run Python through.
PYTHON_LIBRARY = "/Script/PythonScriptPlugin.Default__PythonScriptLibrary"

# The PythonScriptLibrary functions that run Python source given as
# PythonCommand. ExecutePythonCommand has no ExecutionMode and runs a file's
# worth of code. ExecutePythonScript reads its outputs from the globals the
# script leaves, so it is left as it is.
_COMMAND_FUNCTIONS = ("executepythoncommandex", "executepythoncommand")

#: ``ExecutionMode`` to ``compile`` mode, and back.
_MODES = {"executefile": "exec", "executestatement": "single", "evaluatestatement": "eval"}
_EXECUTION_MODES = {"exec": "ExecuteFile", "single": "ExecuteStatement", "eval": "EvaluateStatement"}


def _key(mapping: Dict[str, Any], name: str) -> Optional[str]:
    """The key of ``mapping`` that is ``name`` ignoring case, as Remote Control matches names."""
    return next((key for key in mapping if str(key).lower() == name.lower()), None)


# A console command Unreal hands to Python: FParse::Command(&Cmd, "PY")
# matches "py" ignoring case after spaces and tabs, followed by anything but
# a letter or digit ("py.x" runs ".x"). Unreal runs each line of a command,
# and each part between "|", as a command of its own (FParse::Line); the
# split here also cuts inside quotes, so it refuses more, never less.
_PY_CONSOLE = re.compile(r"[ \t]*py(?![^\W_])", re.IGNORECASE)
_CONSOLE_PARTS = re.compile(r"[\r\n|]")


def console_runs_python(command: str) -> bool:
    """Whether a console command, or one part of it, is a ``py`` command."""
    return any(_PY_CONSOLE.match(part) for part in _CONSOLE_PARTS.split(command))


def _refuse_python_console(body: Dict[str, Any], function_key: str) -> None:
    """Refuse an ``ExecuteConsoleCommand`` call whose command runs Python.

    ``py <code>`` runs the code in the editor's shared console dict
    (``FPythonScriptPlugin::Exec_Runtime``), past the runner: a name it binds
    there, such as a world's package, aborts the editor at the next map change.
    """
    if str(body[function_key]).lower() != "executeconsolecommand":
        return
    parameters = body.get(_key(body, "parameters") or "parameters")
    command_key = _key(parameters, "Command") if isinstance(parameters, dict) else None
    if command_key is not None and console_runs_python(str(parameters[command_key])):
        raise ValueError(
            "A py console command runs its Python in the editor's shared console namespace, where what "
            "it binds outlives the call and can abort the editor at the next map change; run the code "
            "with execute_unreal_script, or ExecutePythonCommandEx, instead"
        )


def python_command(path: str, body: Optional[Dict[str, Any]]) -> Optional[Tuple[str, str, str]]:
    """What a request runs, when it runs Python source in the editor.

    Args:
        path: The request's route.
        body: The request's body.

    Returns:
        The ``PythonCommand`` parameter's key, the source and the ``compile``
        mode, for a ``/remote/object/call`` of ``ExecutePythonCommandEx`` or
        ``ExecutePythonCommand`` on the Python library; otherwise None.

    Raises:
        ValueError: The ``ExecutionMode`` is not one Unreal has, or the call
            is an ``ExecuteConsoleCommand`` of a ``py`` command, which no
            runner can isolate.
    """
    if path.rstrip("/").lower() != "/remote/object/call" or not isinstance(body, dict):
        return None
    object_key, function_key = _key(body, "objectPath"), _key(body, "functionName")
    if object_key is None or function_key is None:
        return None
    _refuse_python_console(body, function_key)
    if str(body[object_key]).lower() != PYTHON_LIBRARY.lower():
        return None
    function = str(body[function_key]).lower()
    if function not in _COMMAND_FUNCTIONS:
        return None
    parameters = body.get(_key(body, "parameters") or "parameters")
    if not isinstance(parameters, dict):
        return None
    command_key = _key(parameters, "PythonCommand")
    if command_key is None or not isinstance(parameters[command_key], str):
        return None
    mode = "exec"
    mode_key = _key(parameters, "ExecutionMode")
    if function == "executepythoncommandex" and mode_key is not None:
        # Remote Control takes the enum's name, with or without its type.
        name = str(parameters[mode_key]).rsplit("::", 1)[-1].lower()
        if name not in _MODES:
            raise ValueError(
                f"ExecutionMode must be ExecuteFile, ExecuteStatement or EvaluateStatement, "
                f"got {parameters[mode_key]!r}"
            )
        mode = _MODES[name]
    return command_key, parameters[command_key], mode


def _encode(source: str) -> str:
    return base64.b64encode(source.encode("utf-8")).decode("ascii")


def runner_command(code: str, mode: str, guard: str = "") -> str:
    """The ``PythonCommand`` that runs ``code`` in a namespace of its own.

    One expression: as an ``ExecuteFile`` command its value is discarded, as
    an ``EvaluateStatement`` command it is the value of ``code``.

    Args:
        code: Python source.
        mode: ``exec``, ``single`` or ``eval``.
        guard: Source to run first in the same namespace, or "".
    """
    # The module, replaced unless it is a plain module: a subclass could
    # carry a ``run`` property that the refresh below would not override.
    name = repr(RUNNER_MODULE)
    install = (
        f"(lambda s, t: s.modules[{name}] if type(s.modules.get({name})) is t.ModuleType "
        f"else s.modules.__setitem__({name}, t.ModuleType({name})) or s.modules[{name}])"
        "(__import__('sys'), __import__('types'))"
    )
    # A traceback shows two frames before the script's own: this command's,
    # always File "<string>", line 1, and the runner's, "<simul runner>".
    # The script's frames follow as "<string>" with its own line numbers.
    # Compiled once per module and source: compiling the runner on every
    # call cost about a millisecond.
    code_key = repr(_RUNNER_CODE_KEY)
    compiled = (
        f"(m.__dict__.get({code_key}) or m.__dict__.setdefault({code_key}, "
        f"compile({RUNNER_SOURCE!r}, '<simul runner>', 'exec')))"
    )
    return (
        f"(lambda m: (exec({compiled}, m.__dict__), m)[1])({install})"
        f".run({mode!r}, {_encode(code)!r}, {_encode(guard) if guard else ''!r})"
    )


# The ``.run(...)`` call that ends a runner_command.
_RUN_CALL = re.compile(
    r"\.run\('(?P<mode>exec|single|eval)', '(?P<code>[A-Za-z0-9+/=]*)', '(?P<guard>[A-Za-z0-9+/=]*)'\)$"
)


def unwrap_command(command: str) -> Optional[Tuple[str, str, str]]:
    """The mode, source and guard a ``runner_command`` carries, or None for any other command.

    What the editor's runner executes, for logs, tests and fake editors.
    """
    match = _RUN_CALL.search(command)
    if match is None or not command.startswith("(lambda m: (exec("):
        return None
    return (
        match["mode"],
        base64.b64decode(match["code"]).decode("utf-8"),
        base64.b64decode(match["guard"]).decode("utf-8"),
    )


def unwrap(body: Dict[str, Any]) -> Dict[str, Any]:
    """``body`` with an isolated ``PythonCommand`` turned back into the source it runs.

    The guard, when there is one, is put back in front of the code, as the
    runner runs it; the ``ExecutionMode`` becomes the original one. Any other
    body comes back as it is.
    """
    parameters_key = _key(body, "parameters") if isinstance(body, dict) else None
    parameters = body.get(parameters_key) if parameters_key is not None else None
    if parameters_key is None or not isinstance(parameters, dict):
        return body
    command_key = _key(parameters, "PythonCommand")
    unwrapped = unwrap_command(str(parameters[command_key])) if command_key is not None else None
    if command_key is None or unwrapped is None:
        return body
    mode, code, guard = unwrapped
    parameters = {**parameters, command_key: guard + code}
    mode_key = _key(parameters, "ExecutionMode")
    if mode_key is not None:
        parameters[mode_key] = _EXECUTION_MODES[mode]
    return {**body, parameters_key: parameters}


def isolate(body: Dict[str, Any], guard: str = "") -> Dict[str, Any]:
    """A Python-running request rewritten to run in a namespace of its own.

    Args:
        body: A body ``python_command`` recognises.
        guard: Source to run before the code, in the same namespace.

    Returns:
        A copy of ``body`` whose ``PythonCommand`` is ``runner_command``'s,
        and whose ``ExecutionMode`` is ExecuteFile, or EvaluateStatement for
        an expression (statement mode runs inside the runner).
    """
    found = python_command("/remote/object/call", body)
    if found is None:
        raise ValueError("Not a call that runs Python source")
    command_key, code, mode = found
    parameters_key = _key(body, "parameters") or "parameters"
    parameters = dict(body[parameters_key])
    parameters[command_key] = runner_command(code, mode, guard)
    mode_key = _key(parameters, "ExecutionMode")
    if mode_key is not None or mode == "eval":
        parameters[mode_key or "ExecutionMode"] = "EvaluateStatement" if mode == "eval" else "ExecuteFile"
    return {**body, parameters_key: parameters}
