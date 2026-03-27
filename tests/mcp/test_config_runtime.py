"""Regression tests for config, logging, and pxr-optional imports."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from typer.testing import CliRunner

src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_path))

from simul_mcp.cli.main import app
from simul_mcp.config import Settings, get_settings, load_settings
from simul_mcp.logging import setup_logging


def test_usd_package_imports_without_pxr() -> None:
    """The USD package must remain importable even when pxr is absent."""
    import simul_mcp.usd as usd_module

    assert hasattr(usd_module, "extract_mesh_data")
    assert hasattr(usd_module, "BBoxCache")


def test_load_settings_supports_nested_repo_config_without_isaac_env(
    monkeypatch,
) -> None:
    """The checked-in Isaac YAML should load without ISAAC_SIM_PATH set."""
    monkeypatch.delenv("ISAAC_SIM_PATH", raising=False)

    settings = load_settings("config/isaac/default.yaml")

    assert settings.isaac_sim.path is None
    assert settings.isaac_sim.headless is False
    assert settings.logging.file_enabled is True
    assert settings.logging.file_path == "logs/simul_mcp.log"
    assert settings.usd.cache_enabled is True
    assert any(path.endswith("/examples") for path in settings.security.allowed_paths)


def test_get_settings_tracks_config_file_changes(tmp_path, monkeypatch) -> None:
    """Changing CONFIG_FILE should return a different cached Settings object."""
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("server:\n  name: First\n", encoding="utf-8")
    second.write_text("server:\n  name: Second\n", encoding="utf-8")

    monkeypatch.setenv("CONFIG_FILE", str(first))
    settings_a = get_settings()
    monkeypatch.setenv("CONFIG_FILE", str(second))
    settings_b = get_settings()

    assert settings_a.server.name == "First"
    assert settings_b.server.name == "Second"


def test_validate_config_reports_validate_settings_failures(
    tmp_path, monkeypatch
) -> None:
    """The CLI must fail when validate_settings finds repo-specific errors."""
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        "security:\n  allowed_paths:\n    - definitely_missing_path_12345\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("CONFIG_FILE", raising=False)
    runner = CliRunner()
    result = runner.invoke(app, ["--json", "validate-config", str(config_path)])

    payload = json.loads(result.stdout)
    assert result.exit_code == 1
    assert payload["valid"] is False
    assert payload["errors"] == ["Allowed path does not exist: definitely_missing_path_12345"]


def test_setup_logging_fallback_honors_disabled_handlers(tmp_path) -> None:
    """Fallback logging must not create files when file logging is disabled."""
    log_path = tmp_path / "logs" / "simul.log"
    settings = Settings(
        logging={
            "file_enabled": False,
            "console_enabled": False,
            "file_path": str(log_path),
        }
    )

    setup_logging(settings=settings, config_file=tmp_path / "missing-logging.yaml")

    assert not log_path.exists()
