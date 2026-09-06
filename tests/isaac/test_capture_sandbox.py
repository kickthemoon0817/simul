"""Regression: a viewport capture stays inside the sandbox and leaves the viewport as it found it.

``capture_isaac_viewport`` wrote to ``tempfile.gettempdir()`` — outside every
allowed root — deleted other ``simul_capture_*.png`` files it found there, and
set the live viewport resolution without ever restoring it, all while annotated
read-only. The generated script now targets a policy-checked capture directory,
reclaims old captures in that directory only, and restores the resolution in a
``finally``.

The script is Kit-side Python, so it is executed here against stubbed ``omni``
modules: the filesystem effects and the resolution round-trip are what matter,
and both are observable without Kit.
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock


from simul_mcp.adapters.isaac_socket_client import ScriptResult
from simul_mcp.config import Settings
from simul_mcp.mcp.tools.isaac_tools import MAX_RETAINED_CAPTURES, IsaacTools


def _settings(sandbox_root: Path, capture_dir: Optional[str] = None) -> Settings:
    settings = Settings()
    security = settings.security.model_copy(update={"allowed_paths": [str(sandbox_root)]})
    viewport = settings.viewport.model_copy(update={"capture_dir": capture_dir})
    return settings.model_copy(update={"security": security, "viewport": viewport})


def _tools(settings: Settings) -> Tuple[IsaacTools, List[str]]:
    captured: List[str] = []

    def _record(code: str) -> ScriptResult:
        captured.append(code)
        return ScriptResult(success=True, output=json.dumps({"path": "unused"}))

    client = MagicMock()
    client.address = "127.0.0.1:8226"
    client.timeout_seconds = 30.0
    client.bridge_enabled = False
    client.fallback_to_vscode = True
    client.execute = AsyncMock(side_effect=_record)
    client.bridge_request = AsyncMock(return_value=None)
    return IsaacTools(client, settings=settings), captured


class _FakeViewport:
    """Just enough of ``ViewportAPI`` for the capture script: a resolution and a capture hook."""

    def __init__(self) -> None:
        self.resolution: Tuple[int, int] = (1920, 1080)
        self.resolution_history: List[Tuple[int, int]] = []
        self.fail_capture: bool = False

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "resolution":
            self.__dict__.setdefault("resolution_history", []).append(tuple(value))
        object.__setattr__(self, name, value)

    def schedule_capture(self, capture: Any) -> None:
        if self.fail_capture:
            raise RuntimeError("renderer unavailable")
        Path(capture.path).write_bytes(b"\x89PNG fake")


class _FakeFileCapture:
    def __init__(self, path: str) -> None:
        self.path = path


def _run_capture_script(script: str, viewport: _FakeViewport) -> Dict[str, Any]:
    """Execute the Kit-side script in-process with stubbed ``omni`` modules."""
    app = types.SimpleNamespace(next_update_async=AsyncMock(return_value=None))
    modules = {
        "omni": types.ModuleType("omni"),
        "omni.kit": types.ModuleType("omni.kit"),
        "omni.kit.app": types.ModuleType("omni.kit.app"),
        "omni.kit.viewport": types.ModuleType("omni.kit.viewport"),
        "omni.kit.viewport.utility": types.ModuleType("omni.kit.viewport.utility"),
        "omni.kit.widget": types.ModuleType("omni.kit.widget"),
        "omni.kit.widget.viewport": types.ModuleType("omni.kit.widget.viewport"),
        "omni.kit.widget.viewport.capture": types.ModuleType("omni.kit.widget.viewport.capture"),
    }
    modules["omni.kit.app"].get_app = lambda: app  # type: ignore[attr-defined]
    modules["omni.kit.viewport.utility"].get_active_viewport = lambda: viewport  # type: ignore[attr-defined]
    modules["omni.kit.widget.viewport.capture"].FileCapture = _FakeFileCapture  # type: ignore[attr-defined]
    modules["omni"].kit = modules["omni.kit"]  # type: ignore[attr-defined]
    modules["omni.kit"].app = modules["omni.kit.app"]  # type: ignore[attr-defined]

    printed: List[str] = []
    namespace: Dict[str, Any] = {"print": lambda *parts: printed.append(" ".join(str(p) for p in parts))}
    code = compile(script, "<capture>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)

    saved = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        # The script under test is the code being verified; eval is how a
        # top-level-await module is run in-process.
        result = eval(code, namespace)  # noqa: S307
        if asyncio.iscoroutine(result):
            asyncio.run(result)
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    assert len(printed) == 1, printed
    return json.loads(printed[0])


def _seed_stale_captures(directory: Path, count: int) -> List[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    files: List[Path] = []
    base = time.time() - 10_000
    for index in range(count):
        path = directory / f"simul_capture_{index:012x}.png"
        path.write_bytes(b"old")
        os.utime(path, (base + index, base + index))
        files.append(path)
    return files


def test_default_capture_dir_is_under_an_allowed_root(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    tools, captured = _tools(_settings(sandbox))

    asyncio.run(tools.capture_isaac_viewport())

    script = captured[0]
    assert repr(str(sandbox.resolve() / "captures")) in script
    assert "tempfile" not in script


def test_configured_capture_dir_is_used_when_inside_the_sandbox(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    configured = sandbox / "shots"
    tools, captured = _tools(_settings(sandbox, capture_dir=str(configured)))

    asyncio.run(tools.capture_isaac_viewport())

    assert repr(str(configured.resolve())) in captured[0]


def test_configured_capture_dir_outside_the_sandbox_is_refused(tmp_path: Path) -> None:
    """A misconfigured capture dir must fail loudly, naming the setting and the roots."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    outside = tmp_path / "elsewhere"
    tools, captured = _tools(_settings(sandbox, capture_dir=str(outside)))

    result = asyncio.run(tools.capture_isaac_viewport())

    assert captured == [], "script ran despite the capture dir being outside the sandbox"
    assert result["error_type"] == "SandboxError"
    assert result["details"]["setting"] == "viewport.capture_dir"
    assert result["details"]["access"] == "write"
    assert str(sandbox.resolve()) in result["details"]["allowed_roots"]


def test_cleanup_only_touches_the_capture_dir(tmp_path: Path) -> None:
    """Stale captures elsewhere — even matching the name pattern — are not ours to delete."""
    sandbox = tmp_path / "sandbox"
    capture_dir = sandbox / "captures"
    sibling = sandbox / "other"
    parent_files = _seed_stale_captures(sandbox, 3)
    sibling_files = _seed_stale_captures(sibling, 3)
    stale = _seed_stale_captures(capture_dir, MAX_RETAINED_CAPTURES + 5)
    tools, captured = _tools(_settings(sandbox))

    asyncio.run(tools.capture_isaac_viewport())
    payload = _run_capture_script(captured[0], _FakeViewport())

    assert Path(payload["path"]).parent == capture_dir
    assert Path(payload["path"]).exists()
    remaining = sorted(p for p in capture_dir.iterdir() if p.name.startswith("simul_capture_"))
    assert len(remaining) == MAX_RETAINED_CAPTURES
    # The oldest ones went; the newest survivors plus the fresh capture stay.
    for path in stale[: len(stale) - (MAX_RETAINED_CAPTURES - 1)]:
        assert not path.exists()
    for path in parent_files + sibling_files:
        assert path.exists(), f"cleanup reached outside the capture dir: {path}"


def test_viewport_resolution_is_restored_after_capture(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    tools, captured = _tools(_settings(sandbox))
    viewport = _FakeViewport()

    asyncio.run(tools.capture_isaac_viewport(width=640, height=360))
    payload = _run_capture_script(captured[0], viewport)

    assert payload["width"] == 640 and payload["height"] == 360
    assert viewport.resolution_history[-2:] == [(640, 360), (1920, 1080)]
    assert viewport.resolution == (1920, 1080)


def test_viewport_resolution_is_restored_when_the_capture_fails(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    tools, captured = _tools(_settings(sandbox))
    viewport = _FakeViewport()
    viewport.fail_capture = True

    asyncio.run(tools.capture_isaac_viewport(width=640, height=360))
    payload = _run_capture_script(captured[0], viewport)

    assert "error" in payload
    assert viewport.resolution == (1920, 1080)


def test_capture_creates_the_capture_dir(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    tools, captured = _tools(_settings(sandbox))

    asyncio.run(tools.capture_isaac_viewport())
    payload = _run_capture_script(captured[0], _FakeViewport())

    assert (sandbox / "captures").is_dir()
    assert payload["size_bytes"] > 0
