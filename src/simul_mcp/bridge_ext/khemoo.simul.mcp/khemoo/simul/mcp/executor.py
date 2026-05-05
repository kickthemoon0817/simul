"""Script executor for the Simul Isaac bridge."""

from __future__ import annotations

import __future__
import contextlib
import dis
import io
import traceback

try:
    from ast import PyCF_ALLOW_TOP_LEVEL_AWAIT
except ImportError:
    PyCF_ALLOW_TOP_LEVEL_AWAIT = 0


class ScriptExecutor:
    """Execute Python code inside the Isaac Kit Python scope."""

    def __init__(self, globals_dict: dict, locals_dict: dict) -> None:
        self._globals = globals_dict
        self._locals = locals_dict
        self._compiler_flags = self._get_compiler_flags()
        self._coroutine_flag = self._get_coroutine_flag()

    async def execute(self, source: str) -> tuple[str, Exception | None, str]:
        """Execute a statement or expression and capture stdout."""
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                do_exec_step = True
                try:
                    code = compile(
                        source,
                        "<string>",
                        "eval",
                        flags=self._compiler_flags,
                        dont_inherit=True,
                    )
                except SyntaxError:
                    pass
                else:
                    result = eval(code, self._globals, self._locals)
                    do_exec_step = False
                if do_exec_step:
                    code = compile(
                        source,
                        "<string>",
                        "exec",
                        flags=self._compiler_flags,
                        dont_inherit=True,
                    )
                    result = eval(code, self._globals, self._locals)
                if self._coroutine_flag != -1 and bool(code.co_flags & self._coroutine_flag):
                    result = await result
        except Exception as exc:
            return output.getvalue(), exc, traceback.format_exc()
        return output.getvalue(), None, ""

    @staticmethod
    def _get_compiler_flags() -> int:
        """Collect future flags supported by the current interpreter."""
        flags = 0
        for value in globals().values():
            try:
                if isinstance(value, __future__._Feature):
                    flags |= value.compiler_flag
            except BaseException:
                pass
        return flags | PyCF_ALLOW_TOP_LEVEL_AWAIT

    @staticmethod
    def _get_coroutine_flag() -> int:
        """Resolve the compiler flag used for coroutine code objects."""
        for key, value in dis.COMPILER_FLAG_NAMES.items():
            if value == "COROUTINE":
                return key
        return -1
