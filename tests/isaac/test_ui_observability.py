"""GUI/state observability tools: script generation checks.

Scope mirrors ``test_generated_scripts.py``: render each script with the
client mocked out, confirm it is valid Python, and check the parts that vary
with input — parameter embedding and the section markers the consolidated
snapshot promises. Behaviour inside Kit is not covered here.
"""

from __future__ import annotations

import ast
import asyncio
import json
import sys
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest

src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_path))

from simul_mcp.adapters.isaac_socket_client import ScriptResult
from simul_mcp.mcp.tools.isaac_tools import IsaacTools


@pytest.fixture()
def captured() -> List[str]:
    return []


@pytest.fixture()
def tools(captured: List[str]) -> IsaacTools:
    def _record(code: str) -> ScriptResult:
        captured.append(code)
        return ScriptResult(success=True, output=json.dumps({"ok": True}))

    client = MagicMock()
    client.address = "127.0.0.1:8226"
    client.timeout_seconds = 30.0
    client.bridge_enabled = False
    client.fallback_to_vscode = True
    client.execute = AsyncMock(side_effect=_record)
    client.execute_bridge_script_only = AsyncMock(side_effect=_record)
    client.execute_vscode_only = AsyncMock(side_effect=_record)
    client.bridge_request = AsyncMock(return_value=None)
    return IsaacTools(client)


class TestUiStateScript:
    def test_parses_and_covers_promised_sections(
        self, tools: IsaacTools, captured: List[str]
    ) -> None:
        """The snapshot script is valid Python and probes every section."""
        asyncio.run(tools.get_isaac_ui_state())

        assert len(captured) == 1
        script = captured[0]
        ast.parse(script)
        for marker in (
            "Workspace.get_windows",
            "get_active_viewport",
            "get_selected_prim_paths",
            "get_timeline_interface",
            "get_stage_url",
            "/app/window/enabled",
        ):
            assert marker in script

    def test_headless_degradation_is_per_section(
        self, tools: IsaacTools, captured: List[str]
    ) -> None:
        """omni.ui failure must be caught inside the script, not raised."""
        asyncio.run(tools.get_isaac_ui_state())

        script = captured[0]
        # The ui section reports unavailability instead of dying.
        assert '"available": False' in script


class TestUiWindowScript:
    def test_defaults_parse(self, tools: IsaacTools, captured: List[str]) -> None:
        asyncio.run(tools.get_isaac_ui_window(window_title="Stage"))

        assert len(captured) == 1
        script = captured[0]
        ast.parse(script)
        assert "_TARGET = 'Stage'" in script
        assert "_MAX_DEPTH = 4" in script
        assert "_MAX_WIDGETS = 400" in script

    def test_caps_embed_as_python_ints(
        self, tools: IsaacTools, captured: List[str]
    ) -> None:
        asyncio.run(
            tools.get_isaac_ui_window(window_title="Stage", max_depth=2, max_widgets=10)
        )

        script = captured[0]
        ast.parse(script)
        assert "_MAX_DEPTH = 2" in script
        assert "_MAX_WIDGETS = 10" in script

    def test_hostile_title_embeds_safely(
        self, tools: IsaacTools, captured: List[str]
    ) -> None:
        """Quotes and backslashes in a title must not break the script."""
        title = 'Sneaky "Window"\\'
        asyncio.run(tools.get_isaac_ui_window(window_title=title))

        script = captured[0]
        tree = ast.parse(script)
        # The embedded value round-trips to the exact original title.
        first = tree.body[0]
        assert isinstance(first, ast.Assign)
        assert first.targets[0].id == "_TARGET"
        assert ast.literal_eval(first.value) == title
