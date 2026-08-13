"""Regression: every generated Isaac script must be valid Python at default args.

``IsaacTools`` builds Python-in-Python scripts from f-string templates and embeds
call parameters into them. A parameter embedded with ``json.dumps`` renders
``None`` as the bare name ``null`` — valid JSON, but not Python. The script then
dies inside Isaac Sim with ``NameError: name 'null' is not defined``, and it does
so on the *default* argument path, which is how these tools are normally called.

The check is deliberately static: render each script, confirm it parses, and
reject JSON literals that leaked into the Python source as free variables. A
static check sees embeds sitting on branches an ``exec`` would never reach, and
needs no ``omni``/``pxr`` stubs to do it.

Scope: script *generation* only. Behaviour inside Kit is not covered here.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest

src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_path))

from simul_mcp.adapters.isaac_socket_client import ScriptResult
from simul_mcp.mcp.tools.isaac_tools import IsaacTools

# JSON spellings of Python's None/True/False. Bare in Python source they are
# undefined names, so any load of one is an embed that used json.dumps where it
# needed a Python repr.
JSON_LITERALS = {"null", "true", "false"}

# Guards against the capture silently covering nothing: if a refactor changes
# how scripts are dispatched, this floor fails loudly rather than passing an
# empty sweep. Well below the ~80 methods that generate scripts today.
_MIN_EXPECTED_SCRIPTS = 60


def _synthesize(param: inspect.Parameter) -> Any:
    """Build a plausible value for a required parameter from its annotation.

    Only required parameters are filled. Optional ones are deliberately left at
    their defaults — the default path is exactly where the embed bug lives.
    """
    annotation = str(param.annotation)
    name = param.name

    if "List[float]" in annotation:
        return [1.0, 2.0, 3.0]
    if "List[str]" in annotation:
        return ["/World/Alpha"]
    if "Dict" in annotation:
        return {"someKey": 1}
    if annotation == "<class 'bool'>":
        return False
    if annotation == "<class 'int'>":
        return 1
    if annotation == "<class 'float'>":
        return 1.0
    if "Any" in annotation:
        return 1
    if name.endswith("path") or name.endswith("paths"):
        return "/World/Alpha"
    return "alpha"


def _capture_generated_scripts() -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """Call every public IsaacTools method and collect the scripts it emits.

    Returns (scripts_by_method, errors_by_method).
    """
    scripts: Dict[str, List[str]] = {}
    errors: Dict[str, str] = {}
    current: List[str] = []

    def _record(code: str) -> ScriptResult:
        current.append(code)
        return ScriptResult(success=True, output=json.dumps({"ok": True}))

    client = MagicMock()
    client.address = "127.0.0.1:8226"
    client.timeout_seconds = 30.0
    # Bridge off, so every tool falls through to the raw-script path and the
    # generated source is observable on client.execute.
    client.bridge_enabled = False
    client.fallback_to_vscode = True
    client.execute = AsyncMock(side_effect=_record)
    client.execute_bridge_script_only = AsyncMock(side_effect=_record)
    client.execute_vscode_only = AsyncMock(side_effect=_record)
    client.bridge_request = AsyncMock(return_value=None)

    tools = IsaacTools(client)

    for name, fn in inspect.getmembers(type(tools), predicate=inspect.isfunction):
        if name.startswith("_"):
            continue
        signature = inspect.signature(fn)
        kwargs = {
            p.name: _synthesize(p)
            for p in signature.parameters.values()
            if p.name != "self" and p.default is inspect.Parameter.empty
        }

        current.clear()
        try:
            asyncio.run(getattr(tools, name)(**kwargs))
        except Exception as exc:  # noqa: BLE001 — reported, not swallowed
            errors[name] = f"{type(exc).__name__}: {exc}"
        if current:
            scripts[name] = list(current)

    return scripts, errors


@pytest.fixture(scope="module")
def generated() -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    return _capture_generated_scripts()


def _leaked_json_literals(script: str) -> List[str]:
    """Return JSON literals loaded as free variables in ``script``."""
    tree = ast.parse(script)
    assigned = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    return sorted(
        {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in JSON_LITERALS
            and node.id not in assigned
        }
    )


def test_capture_covers_the_tool_surface(generated) -> None:
    """The sweep must actually render scripts, or the checks below prove nothing."""
    scripts, errors = generated
    assert not errors, f"methods raised while generating a script: {errors}"
    assert len(scripts) >= _MIN_EXPECTED_SCRIPTS, (
        f"only captured {len(scripts)} generated scripts, expected at least "
        f"{_MIN_EXPECTED_SCRIPTS} — has script dispatch changed?"
    )


def test_generated_scripts_parse(generated) -> None:
    """Every generated script must be syntactically valid Python."""
    scripts, _ = generated
    broken = {}
    for method, rendered in scripts.items():
        for script in rendered:
            try:
                ast.parse(script)
            except SyntaxError as exc:
                broken[method] = f"{exc.msg} (line {exc.lineno})"
    assert not broken, f"generated scripts failed to parse: {broken}"


def test_generated_scripts_embed_no_json_literals(generated) -> None:
    """No script may reference bare ``null``/``true``/``false``.

    These are what ``json.dumps`` produces for ``None``/``True``/``False``; in
    Python source they are undefined names and raise ``NameError`` at runtime
    inside Isaac Sim.
    """
    scripts, _ = generated
    offenders: Dict[str, List[str]] = {}
    for method, rendered in scripts.items():
        for script in rendered:
            leaked = _leaked_json_literals(script)
            if leaked:
                offenders[method] = leaked
    assert not offenders, (
        "generated scripts reference JSON literals that are undefined names in "
        f"Python: {offenders}"
    )
