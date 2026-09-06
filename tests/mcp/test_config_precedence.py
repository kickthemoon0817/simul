"""Settings precedence, the shipped default YAML, and the documented env vars.

The YAML file is a settings source ranked below the environment, so every
``SECTION__KEY`` variable documented in ``.env.example`` must take effect
against the ``default.yaml`` that ships inside the package, and the YAML must
still fill in values when no variable is set.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterator

import pytest
import yaml
from pydantic import BaseModel
from typer.testing import CliRunner

import simul_mcp.config as config_module
from simul_mcp.cli.main import app
from simul_mcp.config import IsaacSimConfig, LoggingConfig, Settings
from simul_mcp.resources import resource_filesystem_path

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_YAML = resource_filesystem_path("config", "default.yaml")
_ENV_EXAMPLE = _REPO / ".env.example"

# Documented SECTION__KEY overrides that the shipped YAML also defines, each with
# a value that differs from what the YAML says.
_DOCUMENTED_OVERRIDES: tuple[tuple[str, str, tuple[str, str], object], ...] = (
    ("LOGGING__LEVEL", "DEBUG", ("logging", "level"), "DEBUG"),
    ("SERVER__PORT", "9911", ("server", "port"), 9911),
    ("ISAAC_SIM__BRIDGE_ENABLED", "false", ("isaac_sim", "bridge_enabled"), False),
    ("ISAAC_SIM__SOCKET_PROTOCOL", "vscode", ("isaac_sim", "socket_protocol"), "vscode"),
    ("SECURITY__SANDBOX_ENABLED", "false", ("security", "sandbox_enabled"), False),
    ("SECURITY__ALLOW_SCRIPT_EXECUTION", "false", ("security", "allow_script_execution"), False),
    ("USD__CACHE_ENABLED", "false", ("usd", "cache_enabled"), False),
)


@pytest.fixture
def clean_settings_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip every documented override and the config-file pointer from the environment."""
    for key, _value, _field, _expected in _DOCUMENTED_OVERRIDES:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("CONFIG_FILE", raising=False)
    monkeypatch.delenv("ISAAC_SIM_PATH", raising=False)
    monkeypatch.delenv("ISAAC_SIM__PATH", raising=False)
    config_module._get_cached_settings.cache_clear()
    yield
    config_module._get_cached_settings.cache_clear()


def _leaf(settings: Settings, field_path: tuple[str, str]) -> object:
    section, leaf = field_path
    return getattr(getattr(settings, section), leaf)


def test_shipped_default_yaml_defines_every_documented_override_key() -> None:
    """The precedence test is only meaningful if the YAML actually sets the keys."""
    payload = config_module._normalise_settings_payload(yaml.safe_load(_DEFAULT_YAML.read_text()))
    for _key, _value, (section, leaf), _expected in _DOCUMENTED_OVERRIDES:
        assert leaf in payload[section], f"{section}.{leaf} is not set by {_DEFAULT_YAML}"


@pytest.mark.usefixtures("clean_settings_env")
@pytest.mark.parametrize(("env_name", "env_value", "field_path", "expected"), _DOCUMENTED_OVERRIDES)
def test_env_var_overrides_key_present_in_shipped_yaml(
    monkeypatch: pytest.MonkeyPatch,
    env_name: str,
    env_value: str,
    field_path: tuple[str, str],
    expected: object,
) -> None:
    monkeypatch.setenv(env_name, env_value)

    settings = Settings.from_yaml(_DEFAULT_YAML)

    assert _leaf(settings, field_path) == expected


@pytest.mark.usefixtures("clean_settings_env")
def test_get_settings_applies_env_over_shipped_yaml_and_keeps_other_yaml_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ISAAC_SIM__SOCKET_PROTOCOL", "python_server")

    settings = config_module.get_settings()

    assert settings.isaac_sim.socket_protocol == "python_server"
    # Sibling keys of the overridden leaf still come from the YAML.
    assert settings.isaac_sim.bridge_port == 8229
    assert settings.server.cors_origins == [
        "http://localhost:8765",
        "https://localhost:8765",
        "http://localhost:8226",
        "http://localhost:8229",
    ]


@pytest.mark.usefixtures("clean_settings_env")
def test_yaml_values_apply_when_no_env_var_is_set() -> None:
    settings = Settings.from_yaml(_DEFAULT_YAML)

    assert settings.isaac_sim.socket_protocol == "auto"
    assert settings.logging.level == "INFO"
    assert settings.logging.components == {
        "usd": "INFO",
        "mesh": "INFO",
        "viewport": "INFO",
        "server": "INFO",
        "isaac": "INFO",
    }
    assert settings.server.cors_origins[-1] == "http://localhost:8229"
    assert settings.security.allowed_paths == ["examples", "tests/data", "/tmp/simul_mcp"]
    assert settings.logging.audit_path == "~/.simul/logs/audit.jsonl"


@pytest.mark.usefixtures("clean_settings_env")
def test_bare_settings_ignore_the_yaml_file() -> None:
    """``Settings()`` stays defaults plus environment; only ``from_yaml`` adds the file."""
    settings = Settings()

    assert settings.server.cors_origins == ["http://localhost:*", "https://localhost:*"]
    assert settings.logging.components == {}


@pytest.mark.usefixtures("clean_settings_env")
def test_init_kwargs_still_beat_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SERVER__PORT", "9911")

    settings = Settings(server={"port": 9000})

    assert settings.server.port == 9000


@pytest.mark.usefixtures("clean_settings_env")
def test_observability_logging_fields_are_reachable_from_yaml(tmp_path: Path) -> None:
    config_file = tmp_path / "logging.yaml"
    config_file.write_text(
        "logging:\n"
        "  file:\n"
        "    per_instance: true\n"
        "    retention_days: 3\n"
        "  audit:\n"
        "    enabled: false\n"
        "    path: /var/log/simul/audit.jsonl\n"
        "  structured:\n"
        "    enabled: true\n",
        encoding="utf-8",
    )

    settings = Settings.from_yaml(config_file)

    assert settings.logging.per_instance is True
    assert settings.logging.retention_days == 3
    assert settings.logging.audit_enabled is False
    assert settings.logging.audit_path == "/var/log/simul/audit.jsonl"
    assert settings.logging.structured_enabled is True


def test_normaliser_covers_every_logging_field() -> None:
    payload = config_module._normalise_settings_payload(
        {
            "logging": {
                "level": "DEBUG",
                "format": "json",
                "file_enabled": False,
                "file_path": "x.log",
                "file_max_size": "1MB",
                "file_backup_count": 1,
                "console_enabled": False,
                "console_colored": False,
                "components": {"usd": "DEBUG"},
                "per_instance": True,
                "retention_days": 2,
                "audit_enabled": False,
                "audit_path": "a.jsonl",
                "structured_enabled": True,
            }
        }
    )
    assert set(payload["logging"]) == set(LoggingConfig.model_fields)


def test_shipped_yaml_has_no_sections_the_settings_model_ignores() -> None:
    raw = yaml.safe_load(_DEFAULT_YAML.read_text(encoding="utf-8"))

    assert set(raw) <= set(Settings.model_fields), sorted(set(raw) - set(Settings.model_fields))
    assert "extensions" not in raw["isaac_sim"]
    assert "simulation" not in raw["isaac_sim"]
    assert "rotation" not in raw["logging"]["file"]


def _env_example_keys() -> list[str]:
    """Every ``KEY=`` assignment in .env.example, including commented-out examples."""
    pattern = re.compile(r"^#?\s*([A-Z][A-Z0-9_]*)=")
    keys: list[str] = []
    for line in _ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            keys.append(match.group(1))
    return keys


def test_env_example_lists_documented_variables() -> None:
    keys = _env_example_keys()
    assert len(keys) > 50
    assert "CONFIG_FILE" in keys


@pytest.mark.parametrize("key", [k for k in _env_example_keys() if k != "CONFIG_FILE"])
def test_env_example_variable_names_a_real_settings_field(key: str) -> None:
    """``CONFIG_FILE`` is read by the loader, not the model; everything else is ``SECTION__FIELD``."""
    section, _, field = key.lower().partition("__")

    assert section in Settings.model_fields, f"{key}: no Settings section {section!r}"
    section_model = Settings.model_fields[section].annotation
    assert isinstance(section_model, type) and issubclass(section_model, BaseModel)
    assert field in section_model.model_fields, f"{key}: {section}.{field} is not a field"


@pytest.mark.usefixtures("clean_settings_env")
def test_stale_isaac_sim_path_is_a_warning_not_a_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("ISAAC_SIM_PATH", "/nonexistent/isaac-sim")
    # setup_logging in other tests turns propagation off for simul_mcp loggers,
    # so capture at the emitting logger rather than at the root.
    config_logger = logging.getLogger("simul_mcp.config")
    config_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger="simul_mcp.config"):
            settings = Settings()
    finally:
        config_logger.removeHandler(caplog.handler)

    assert settings.isaac_sim.path == "/nonexistent/isaac-sim"
    assert settings.isaac_sim.path_error == "Isaac Sim path does not exist: /nonexistent/isaac-sim"
    assert any("Isaac Sim path does not exist" in record.getMessage() for record in caplog.records)


def test_isaac_path_without_python_launcher_is_reported(tmp_path: Path) -> None:
    config = IsaacSimConfig(path=str(tmp_path))

    assert config.path_error == f"Isaac Sim python executable not found in: {tmp_path}"

    (tmp_path / "python.sh").write_text("#!/bin/sh\n")
    assert IsaacSimConfig(path=str(tmp_path)).path_error is None
    assert IsaacSimConfig(path=None).path_error is None


@pytest.mark.usefixtures("clean_settings_env")
def test_info_command_succeeds_with_stale_isaac_sim_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ISAAC_SIM_PATH", "/nonexistent/isaac-sim")

    result = CliRunner().invoke(app, ["--json", "info"])

    assert result.exit_code == 0, result.output
    assert "/nonexistent/isaac-sim" in result.output
    assert "Isaac Sim path does not exist" in result.output
