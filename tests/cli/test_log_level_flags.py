"""Regression: --log-level and --verbose must apply, not crash.

The server command assigned ``settings.logging.level`` directly, but
``LoggingConfig`` is deliberately declared ``frozen=True``, so pydantic v2
raises ``ValidationError: Instance is frozen`` on assignment. Both flags
therefore crashed the command they were meant to configure.

The fix rebuilds the section rather than unfreezing it, so the immutability that
review asked for stays in place.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest
from typer.testing import CliRunner

src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_path))

from simul_mcp.cli import main as cli_main  # noqa: E402
from simul_mcp.cli.main import app  # noqa: E402
from simul_mcp.config import Settings  # noqa: E402

runner = CliRunner()


@pytest.fixture
def captured_settings(monkeypatch: pytest.MonkeyPatch) -> List[Settings]:
    """Stop before the server actually starts, keeping the settings it built."""
    seen: List[Settings] = []

    async def _capture(settings: Settings, transport: str, **kwargs: Any) -> None:
        seen.append(settings)

    monkeypatch.setattr(cli_main, "start_mcp_server", _capture)
    monkeypatch.setattr(cli_main, "_is_isaac_reachable", lambda *a, **k: False)
    return seen


def test_log_level_flag_is_applied(captured_settings: List[Settings]) -> None:
    result = runner.invoke(app, ["server", "--log-level", "debug"])

    assert result.exit_code == 0, result.output
    assert captured_settings, "server never received settings"
    assert captured_settings[-1].logging.level == "DEBUG"


def test_verbose_flag_is_applied(captured_settings: List[Settings]) -> None:
    result = runner.invoke(app, ["server", "--verbose"])

    assert result.exit_code == 0, result.output
    assert captured_settings, "server never received settings"
    assert captured_settings[-1].logging.level == "DEBUG"


def test_logging_section_stays_immutable() -> None:
    """The fix must not work by unfreezing what review deliberately froze."""
    settings = Settings()

    with pytest.raises(Exception):
        settings.logging.level = "DEBUG"


def test_unreal_tools_flag_overrides_tool_surface(
    captured_settings: List[Settings],
) -> None:
    result = runner.invoke(app, ["server", "--unreal-tools", "FULL"])

    assert result.exit_code == 0, result.output
    assert captured_settings[-1].unreal.tool_surface == "full"


def test_unreal_tools_flag_rejects_unknown_value(
    captured_settings: List[Settings],
) -> None:
    result = runner.invoke(app, ["server", "--unreal-tools", "medium"])

    assert result.exit_code == 1
    assert "Unknown --unreal-tools value" in result.output
    assert not captured_settings, "server started despite an invalid flag"


def test_unreal_tools_default_keeps_thin_surface(
    captured_settings: List[Settings],
) -> None:
    result = runner.invoke(app, ["server"])

    assert result.exit_code == 0, result.output
    assert captured_settings[-1].unreal.tool_surface == "thin"
