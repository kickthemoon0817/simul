"""A ``.env`` in the working directory no longer feeds the settings.

MCP clients start ``simul server`` in the user's project, so a cloned
repository's ``.env`` with ``SECURITY__ALLOWED_PATHS=["/"]`` widened the file
sandbox to the whole disk (``validate_usd_file("/etc/hosts")`` succeeded), and
the same file could re-enable script execution or move the HTTP bind. Only the
file ``SIMUL_ENV_FILE`` names is read now; a working-directory ``.env`` that
sets simul variables is reported and ignored.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import pytest

import simul.config as config_module
from simul.config import ConfigError, Settings, get_settings
from simul.utils.paths import PathPolicy

_HOSTILE = 'SECURITY__ALLOWED_PATHS=["/"]\nSECURITY__ALLOW_SCRIPT_EXECUTION=true\nSERVER__HOST=0.0.0.0\n'


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A working directory holding a hostile ``.env``, with no simul variables exported."""
    for name in ("SECURITY__ALLOWED_PATHS", "SECURITY__ALLOW_SCRIPT_EXECUTION", "SERVER__HOST", "SIMUL_ENV_FILE"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / ".env").write_text(_HOSTILE)
    monkeypatch.chdir(tmp_path)
    config_module._IGNORED_DOTENV_WARNED.clear()
    config_module._get_cached_settings.cache_clear()
    yield tmp_path
    config_module._get_cached_settings.cache_clear()


@pytest.fixture
def config_log(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Capture ``simul.config`` at the emitting logger.

    ``setup_logging`` in other tests turns propagation off for simul loggers,
    so records may never reach the root handler caplog installs.
    """
    logger = logging.getLogger("simul.config")
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger="simul.config"):
            yield caplog
    finally:
        logger.removeHandler(caplog.handler)


def test_a_working_directory_env_does_not_widen_the_sandbox(project: Path) -> None:
    settings = Settings()

    assert settings.security.allowed_paths == ["examples", "tests/data", "/tmp/simul-work"]
    assert settings.server.host == "localhost"
    assert not PathPolicy.from_settings(settings).is_allowed("/etc/hosts")


def test_get_settings_ignores_it_too(project: Path) -> None:
    assert get_settings().security.allowed_paths == ["examples", "tests/data", "/tmp/simul-work"]


def test_the_ignored_file_is_reported_once_with_the_keys_it_sets(
    project: Path, config_log: pytest.LogCaptureFixture
) -> None:
    Settings()
    Settings()

    # By identity: with propagation on, caplog's handler sees each record twice.
    records = {id(record): record for record in config_log.records if "Ignoring" in record.getMessage()}
    warnings = [record.getMessage() for record in records.values()]
    assert len(warnings) == 1
    assert "SECURITY__ALLOWED_PATHS" in warnings[0]
    assert "SERVER__HOST" in warnings[0]
    assert "SIMUL_ENV_FILE" in warnings[0]


def test_a_project_env_without_simul_variables_is_not_reported(
    project: Path, config_log: pytest.LogCaptureFixture
) -> None:
    (project / ".env").write_text("DATABASE_URL=postgres://x\nDEBUG=1\n")

    Settings()

    assert not [record for record in config_log.records if "Ignoring" in record.getMessage()]


def test_simul_env_file_opts_a_file_in(project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chosen = tmp_path / "simul.env"
    chosen.write_text("UNREAL__PORT=30111\nSECURITY__ALLOW_SCRIPT_EXECUTION=false\n")
    monkeypatch.setenv("SIMUL_ENV_FILE", str(chosen))

    settings = Settings()

    assert settings.unreal.port == 30111
    assert settings.security.allow_script_execution is False
    # Only that file: the working directory's .env still contributes nothing.
    assert settings.security.allowed_paths == ["examples", "tests/data", "/tmp/simul-work"]


def test_the_process_environment_still_beats_the_chosen_file(
    project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chosen = tmp_path / "simul.env"
    chosen.write_text("UNREAL__PORT=30111\n")
    monkeypatch.setenv("SIMUL_ENV_FILE", str(chosen))
    monkeypatch.setenv("UNREAL__PORT", "30222")

    assert Settings().unreal.port == 30222


def test_simul_env_file_naming_nothing_is_an_error(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIMUL_ENV_FILE", str(project / "missing.env"))

    with pytest.raises(ConfigError, match="SIMUL_ENV_FILE names no file"):
        Settings()
