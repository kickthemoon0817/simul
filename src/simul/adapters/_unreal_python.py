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
code into a fresh dict holding only ``__name__`` and the ``unreal`` module,
and clears that dict when the call ends, returned or raised. The command
itself is one expression that binds nothing in the shared dict. Clearing
breaks the usual dict-to-function cycles at once; any cycle left (a class
holding an object, say) is unreachable and is collected by Python's
collector, which the plugin runs before every Unreal garbage collection
(``FPythonScriptPlugin::OnPreGarbageCollect``), so no ``gc.collect()`` is
needed here. In statement mode ``sys.displayhook`` would keep the last
expression's value in ``builtins._``, so ``run`` swaps in one that only
prints it.

The source travels base64-encoded: UE treats an ``ExecuteFile`` command whose
first ``.py`` is followed by whitespace as the path of a file to run, and the
encoding (and the runner, which never spells it) keeps that from ever
matching.
"""

from __future__ import annotations

import base64
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
import base64, sys

def _source(encoded):
    return base64.b64decode(encoded).decode("utf-8")

def _display(value):
    if value is not None:
        print(repr(value))

def run(mode, code, guard=""):
    namespace = {"__name__": "__main__", "unreal": sys.modules["unreal"]}
    displayhook = sys.displayhook
    try:
        if guard:
            exec(compile(_source(guard), "<string>", "exec"), namespace)
        if mode == "eval":
            return eval(compile(_source(code), "<string>", "eval"), namespace)
        if mode == "single":
            sys.displayhook = _display
        exec(compile(_source(code), "<string>", mode), namespace)
    finally:
        sys.displayhook = displayhook
        namespace.clear()
'''

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
        ValueError: The ``ExecutionMode`` is not one Unreal has.
    """
    if path.rstrip("/").lower() != "/remote/object/call" or not isinstance(body, dict):
        return None
    object_key, function_key = _key(body, "objectPath"), _key(body, "functionName")
    if object_key is None or function_key is None:
        return None
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
    install = (
        f"__import__('sys').modules.setdefault({RUNNER_MODULE!r}, "
        f"__import__('types').ModuleType({RUNNER_MODULE!r}))"
    )
    # The runner's own frame shows as "<simul runner>" in a traceback, so
    # the caller's frames, "<string>", stand apart from it.
    return (
        f"(lambda m: (exec(compile({RUNNER_SOURCE!r}, '<simul runner>', 'exec'), m.__dict__), m)[1])({install})"
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
