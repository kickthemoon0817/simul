"""Fixtures for exercising generated Isaac scripts without Isaac Sim.

The scripts ``IsaacTools`` generates only need ``omni.usd`` for the stage and
``pxr`` for everything else. With ``omni.usd`` stubbed to hand back an
in-memory ``Usd.Stage`` the scripts run under plain ``exec`` and their
behaviour on real USD data becomes observable.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import types
from typing import Any, Callable, Dict, List, Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest

from simul_mcp.adapters.isaac_socket_client import ScriptResult
from simul_mcp.config import Settings
from simul_mcp.mcp.tools.isaac_tools import IsaacTools

RunOnStage = Callable[[str, Any], Dict[str, Any]]


@pytest.fixture
def capturing_tools() -> Tuple[IsaacTools, List[str]]:
    """An IsaacTools whose client records every script instead of sending it."""
    captured: List[str] = []

    def _record(code: str) -> ScriptResult:
        captured.append(code)
        return ScriptResult(success=True, output=json.dumps({"ok": True}))

    client = MagicMock()
    client.address = "127.0.0.1:8226"
    client.timeout_seconds = 30.0
    client.bridge_enabled = False
    client.fallback_to_vscode = True
    client.execute = AsyncMock(side_effect=_record)
    client.execute_vscode_only = AsyncMock(side_effect=_record)
    client.execute_bridge_script_only = AsyncMock(side_effect=_record)
    client.bridge_request = AsyncMock(return_value=None)
    return IsaacTools(client, settings=Settings()), captured


@pytest.fixture
def run_on_stage(monkeypatch: pytest.MonkeyPatch) -> RunOnStage:
    """Run a generated script against a ``Usd.Stage`` and parse what it prints."""

    def _run(script: str, stage: Any, selected_paths: List[str] | None = None) -> Dict[str, Any]:
        selection = types.SimpleNamespace(
            get_selected_prim_paths=lambda: list(selected_paths or [])
        )
        context = types.SimpleNamespace(
            get_stage=lambda: stage,
            get_stage_url=lambda: "anon:memory.usda",
            get_selection=lambda: selection,
        )
        fake_usd = types.ModuleType("omni.usd")
        fake_usd.get_context = lambda: context  # type: ignore[attr-defined]
        fake_omni = types.ModuleType("omni")
        fake_omni.usd = fake_usd  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "omni", fake_omni)
        monkeypatch.setitem(sys.modules, "omni.usd", fake_usd)

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exec(compile(script, "<isaac-script>", "exec"), {"__name__": "__isaac_script__"})
        return json.loads(stdout.getvalue())

    return _run
