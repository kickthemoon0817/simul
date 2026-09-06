"""Regression tests for config, logging, and pxr-dependent imports."""

from __future__ import annotations

import json
import logging
import sys
import importlib

import pytest
from typer.testing import CliRunner


from simul_mcp.cli.main import app
from simul_mcp.config import Settings, get_settings, load_settings
from simul_mcp.logging import setup_logging
from simul_mcp.resources import resource_filesystem_path


def test_usd_package_import_requires_pxr() -> None:
    """The USD package should reflect whether pxr is installed."""
    sys.modules.pop("simul_mcp.usd", None)
    try:
        import pxr  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="pxr"):
            importlib.import_module("simul_mcp.usd")
    else:
        usd_module = importlib.import_module("simul_mcp.usd")
        assert hasattr(usd_module, "extract_mesh_data")
        assert hasattr(usd_module, "BBoxCache")


def test_load_settings_supports_nested_repo_config_without_isaac_env(
    monkeypatch,
) -> None:
    """The packaged default YAML should load without ISAAC_SIM_PATH set."""
    monkeypatch.delenv("ISAAC_SIM_PATH", raising=False)

    settings = load_settings(resource_filesystem_path("config", "default.yaml"))

    assert settings.isaac_sim.path is None
    assert settings.isaac_sim.headless is False
    assert settings.logging.file_enabled is True
    assert settings.logging.file_path == "~/.simul/logs/simul_mcp.log"
    assert settings.logging.file_backup_count == 5
    assert settings.logging.console_colored is True
    assert settings.usd.cache_enabled is True
    assert settings.isaac_sim.bridge_enabled is True
    assert settings.isaac_sim.bridge_port == 8229
    assert settings.isaac_sim.bridge_fallback_to_vscode is True
    assert settings.usd.max_concurrent_operations == 10
    assert settings.viewport.fov == 45.0
    assert settings.viewport.max_bounces == 4
    assert settings.security.rate_limiting_enabled is True
    assert settings.security.requests_per_minute == 60
    assert settings.security.global_requests_per_minute == 600
    assert "examples" in settings.security.allowed_paths


def test_load_settings_normalizes_nested_instance_bridge_config(tmp_path) -> None:
    """Per-instance nested bridge settings should flatten into the Settings model."""
    config_path = tmp_path / "instances.yaml"
    config_path.write_text(
        "\n".join(
            [
                "isaac_sim:",
                "  instances:",
                "    - name: verifier",
                "      host: 127.0.0.1",
                "      port: 8326",
                "      timeout: 12.5",
                "      bridge:",
                "        enabled: true",
                "        host: 127.0.0.1",
                "        port: 8329",
                "        timeout: 8.0",
                "        fallback_to_vscode: false",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings(config_path)

    assert len(settings.isaac_sim.instances) == 1
    instance = settings.isaac_sim.instances[0]
    assert instance.name == "verifier"
    assert instance.port == 8326
    assert instance.timeout == 12.5
    assert instance.bridge_enabled is True
    assert instance.bridge_host == "127.0.0.1"
    assert instance.bridge_port == 8329
    assert instance.bridge_timeout == 8.0
    assert instance.bridge_fallback_to_vscode is False


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

    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level
    original_propagate = root_logger.propagate
    try:
        setup_logging(settings=settings, config_file=tmp_path / "missing-logging.yaml")
        assert not log_path.exists()
    finally:
        for handler in list(root_logger.handlers):
            handler.close()
        root_logger.handlers = original_handlers
        root_logger.setLevel(original_level)
        root_logger.propagate = original_propagate
