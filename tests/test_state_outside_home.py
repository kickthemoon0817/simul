"""The suite writes no session records or logs under the real ``~/.simul``."""

from __future__ import annotations

from pathlib import Path

from simul.config import Settings
from simul.mcp import session_manager, usage_tracker


def test_the_suite_writes_simul_state_outside_the_home_directory() -> None:
    """Every run of the suite used to add test sessions to ~/.simul/sessions
    and test tool calls to ~/.simul/logs (tests/conftest.py now redirects them)."""
    home_state = Path.home() / ".simul"
    settings = Settings()

    assert not session_manager._DEFAULT_SESSION_DIR.is_relative_to(home_state)
    assert not usage_tracker._DEFAULT_LOG_DIR.is_relative_to(home_state)
    assert not Path(settings.logging.file_path).expanduser().is_relative_to(home_state)
    assert not Path(settings.logging.audit_path).expanduser().is_relative_to(home_state)
