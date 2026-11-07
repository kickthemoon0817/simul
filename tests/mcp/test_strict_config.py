"""Configuration that cannot be used as written stops simul instead of being skipped.

Before, a YAML file with one mis-indented line contributed nothing and
``validate-config`` still said ``valid: true``, so ``allow_script_execution:
false`` ran with scripts on; ``CONFIG_FILE`` naming no file meant defaults; a
typo'd key was dropped; ``SERVER=prod`` in the environment stopped every
command with a traceback; one invalid value (``UNREAL__PORT=70000``) did the
same, even under ``--json``; ``ISAAC_SIM__BRIDGE__PORT`` was silently ignored;
and invalid CLI flags exited 1 with ``ValueError`` (or, for ``--log-level``,
were accepted).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List

import click
import pytest
import yaml
from typer.testing import CliRunner

import simul
import simul.config as config_module
from simul.cli import main as cli_main
from simul.cli.main import app
from simul.config import (
    ConfigError,
    Settings,
    find_unknown_env_names,
    find_unknown_yaml_keys,
    get_settings,
    inspect_config_file,
    load_settings,
)

runner = CliRunner()
SRC = str(Path(simul.__file__).resolve().parents[1])

BAD_YAML = (
    "server:\n"
    "  port: 40000\n"
    "security:\n"
    "  allow_script_execution: false\n"
    "   rate_limiting:   # bad indent\n"
    "    requests_per_minute: 5\n"
)
GOOD_YAML = "security:\n  allow_script_execution: false\n  rate_limiting:\n    requests_per_minute: 5\n"


@pytest.fixture(autouse=True)
def fresh_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """No config file pointer, no cached settings, and warnings reported afresh."""
    monkeypatch.delenv("CONFIG_FILE", raising=False)
    config_module._get_cached_settings.cache_clear()
    config_module._UNKNOWN_WARNED.clear()
    yield
    config_module._get_cached_settings.cache_clear()


@pytest.fixture
def config_log(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Capture ``simul.config`` at the emitting logger (other tests turn propagation off)."""
    logger = logging.getLogger("simul.config")
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger="simul.config"):
            yield caplog
    finally:
        logger.removeHandler(caplog.handler)


def _write(tmp_path: Path, text: str, name: str = "config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _envelope(output: str) -> Dict[str, Any]:
    return json.loads(output.strip().splitlines()[-1])


# ---------------------------------------------------------------------------
# B1: a config file that cannot be used fails the load
# ---------------------------------------------------------------------------
class TestBrokenYaml:
    def test_a_syntax_error_fails_the_load(self, tmp_path: Path) -> None:
        path = _write(tmp_path, BAD_YAML)

        with pytest.raises(ConfigError, match="not valid YAML") as raised:
            load_settings(path)

        assert "line 5" in raised.value.message

    def test_get_settings_fails_too(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CONFIG_FILE", str(_write(tmp_path, BAD_YAML)))

        with pytest.raises(ConfigError, match="not valid YAML"):
            get_settings()

    @pytest.mark.parametrize(
        ("text", "match"),
        [
            ("- server\n- port\n", "must hold a mapping of sections, not a list"),
            ("security: strict\n", "section 'security' must be a mapping"),
            ("isaac_sim:\n  bridge: 8229\n", "isaac_sim.bridge must be a mapping"),
            ("isaac_sim:\n  instances: {name: a}\n", "isaac_sim.instances must be a list"),
        ],
    )
    def test_a_file_of_the_wrong_shape_fails_the_load(self, tmp_path: Path, text: str, match: str) -> None:
        with pytest.raises(ConfigError, match=match):
            load_settings(_write(tmp_path, text))

    def test_an_empty_file_is_just_defaults(self, tmp_path: Path) -> None:
        assert load_settings(_write(tmp_path, "")).server.port == 8765

    def test_tools_refuses_the_file_instead_of_registering_script_tools(self, tmp_path: Path) -> None:
        path = _write(tmp_path, BAD_YAML)

        result = runner.invoke(app, ["--json", "tools", "--config", str(path), "--backends", "unreal"])

        assert result.exit_code == 2, result.output
        envelope = _envelope(result.stdout)
        assert envelope["error_type"] == "ConfigError"
        assert "not valid YAML" in envelope["error"]
        assert "execute_unreal_script" not in result.stdout

    def test_validate_config_reports_it_invalid(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["--json", "validate-config", str(_write(tmp_path, BAD_YAML))])

        payload = _envelope(result.stdout)
        assert result.exit_code == 1
        assert payload["valid"] is False and payload["success"] is False
        assert any("not valid YAML" in error for error in payload["errors"])

    def test_the_good_file_still_validates(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["--json", "validate-config", str(_write(tmp_path, GOOD_YAML))])

        payload = _envelope(result.stdout)
        assert result.exit_code == 0, result.output
        assert payload["valid"] is True
        assert payload["allow_script_execution"] is False


class TestMissingFile:
    def test_config_file_naming_nothing_is_an_error(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "missing.yaml"))

        with pytest.raises(ConfigError, match="CONFIG_FILE names no file"):
            get_settings()

    def test_every_command_exits_2_on_it(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "missing.yaml"))

        result = runner.invoke(app, ["--json", "version"])

        assert result.exit_code == 2
        assert _envelope(result.stdout)["error_type"] == "ConfigError"

    @pytest.mark.parametrize("command", [["server"], ["info"], ["tools"], ["validate-config"]])
    def test_an_explicit_missing_config_exits_2(self, tmp_path: Path, command: List[str]) -> None:
        missing = str(tmp_path / "missing.yaml")
        argv = [*command, missing] if command == ["validate-config"] else [*command, "--config", missing]

        result = runner.invoke(app, ["--json", *argv])

        assert result.exit_code == 2, result.output
        assert "not found" in result.output


class TestUnknownKeys:
    def test_an_unknown_security_key_fails_the_load_with_a_suggestion(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "security:\n  allow_scripts_execution: false\n")

        with pytest.raises(ConfigError, match="did you mean security.allow_script_execution") as raised:
            load_settings(path)

        assert raised.value.errors[0]["setting"] == "security.allow_scripts_execution"
        assert raised.value.errors[0]["suggestion"] == "security.allow_script_execution"

    def test_a_misspelt_security_section_fails_too(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="did you mean security"):
            load_settings(_write(tmp_path, "securty:\n  allow_script_execution: false\n"))

    def test_an_unknown_key_elsewhere_warns_and_loads(
        self, tmp_path: Path, config_log: pytest.LogCaptureFixture
    ) -> None:
        settings = load_settings(_write(tmp_path, "unreal:\n  portt: 30111\n  port: 30112\n"))

        assert settings.unreal.port == 30112
        messages = [record.getMessage() for record in config_log.records]
        assert any("unknown setting unreal.portt (did you mean unreal.port?)" in message for message in messages)

    def test_validate_config_makes_every_unknown_key_an_error(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "unreal:\n  portt: 30111\nisaac_sim:\n  bridge:\n    prot: 1\n")

        result = runner.invoke(app, ["--json", "validate-config", str(path)])

        payload = _envelope(result.stdout)
        assert result.exit_code == 1
        assert any("unreal.portt (did you mean unreal.port?)" in error for error in payload["errors"])
        assert any(
            "isaac_sim.bridge.prot (did you mean isaac_sim.bridge.port?)" in error for error in payload["errors"]
        )

    def test_retired_keys_load_silently_and_validate_as_warnings(
        self, tmp_path: Path, config_log: pytest.LogCaptureFixture
    ) -> None:
        path = _write(
            tmp_path,
            "server:\n  cors_origins: ['http://x']\n  max_connections: 10\n"
            "logging:\n  file:\n    rotation: size\n    backup_count: 5\n"
            "mesh:\n  decimation: {enabled: true}\n",
        )

        load_settings(path)
        assert not [record for record in config_log.records if "Ignoring" in record.getMessage()]

        payload = _envelope(runner.invoke(app, ["--json", "validate-config", str(path)]).stdout)
        assert payload["valid"] is True
        assert any("server.cors_origins is a retired setting" in warning for warning in payload["warnings"])
        assert any("mesh is a retired setting" in warning for warning in payload["warnings"])

    def test_the_shipped_default_yaml_has_no_unknown_keys(self) -> None:
        raw = yaml.safe_load(config_module._DEFAULT_CONFIG_FILE.read_text(encoding="utf-8"))

        assert find_unknown_yaml_keys(raw, include_retired=True) == []

    def test_unknown_keys_inside_isaac_instances(self) -> None:
        raw = {"isaac_sim": {"instances": [{"name": "a", "prot": 1, "bridge": {"hots": "x"}}]}}

        names = [item.name for item in find_unknown_yaml_keys(raw)]

        assert names == ["isaac_sim.instances[0].prot", "isaac_sim.instances[0].bridge.hots"]


class TestYamlLayout:
    def test_every_nested_group_key_names_a_real_field(self) -> None:
        for section, groups in config_module._YAML_GROUPS.items():
            fields = config_module._section_fields(section)
            for group, mapping in groups.items():
                for key, target in mapping.items():
                    assert target in fields, f"{section}.{group}.{key} -> {target}"

    def test_every_nested_group_key_reaches_its_field(self) -> None:
        """Feed each nested key a distinct value and find it on the field it maps to."""
        samples = {
            int: 4321,
            float: 12.5,
            bool: False,
            str: "/tmp/sentinel",
        }
        raw: Dict[str, Any] = {}
        expected: Dict[str, Any] = {}
        for section, groups in config_module._YAML_GROUPS.items():
            model = Settings.model_fields[section].annotation
            for group, mapping in groups.items():
                for key, target in mapping.items():
                    annotation = model.model_fields[target].annotation
                    value = ["/tmp/sentinel"] if "List" in str(annotation) or "list" in str(annotation) else None
                    for kind, sample in samples.items():
                        if value is None and (annotation is kind or str(kind.__name__) in str(annotation)):
                            value = sample
                    assert value is not None, (section, target, annotation)
                    raw.setdefault(section, {}).setdefault(group, {})[key] = value
                    expected[f"{section}.{target}"] = value

        payload = config_module._normalise_settings_payload(raw)

        for dotted, value in expected.items():
            section, target = dotted.split(".")
            assert payload[section][target] == value, dotted


# ---------------------------------------------------------------------------
# M4: bare section-name variables no longer stop simul
# ---------------------------------------------------------------------------
class TestBareSectionNames:
    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("SERVER", "prod"),
            ("LOGGING", "1"),
            ("USD", "1"),
            ("SECURITY", "strict"),
            ("VIEWPORT", "main"),
            ("UNREAL", "yes"),
            ("isaac_sim", "x"),
            ("SERVER", "[1, 2]"),
        ],
    )
    def test_a_non_object_value_is_ignored(self, monkeypatch: pytest.MonkeyPatch, name: str, value: str) -> None:
        monkeypatch.setenv(name, value)

        settings = Settings()

        assert settings.server.port == 8765
        assert settings.security.allow_script_execution is True

    def test_a_json_object_still_applies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SECURITY", '{"allow_script_execution": false}')

        assert Settings().security.allow_script_execution is False

    def test_commands_run_with_it_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SERVER", "prod")
        monkeypatch.setattr(cli_main, "_is_isaac_reachable", lambda *a, **k: False)

        result = runner.invoke(app, ["--json", "version"])

        assert result.exit_code == 0, result.output

    def test_the_env_file_gets_the_same_treatment(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        env_file = _write(tmp_path, "SERVER=prod\nSERVER__PORT=9100\n", name="simul.env")
        monkeypatch.setenv("SIMUL_ENV_FILE", str(env_file))

        assert Settings().server.port == 9100


# ---------------------------------------------------------------------------
# m8: one invalid value is a clean ConfigError naming the variable
# ---------------------------------------------------------------------------
class TestInvalidValues:
    def test_the_error_names_the_variable_and_the_reason(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("UNREAL__PORT", "70000")

        with pytest.raises(ConfigError) as raised:
            get_settings()

        entry = raised.value.errors[0]
        assert entry["setting"] == "unreal.port"
        assert entry["source"] == "UNREAL__PORT"
        assert entry["value"] == "70000"
        assert "less than or equal to 65535" in entry["reason"]

    @pytest.mark.parametrize(
        "argv",
        [["isaac", "ping"], ["logs", "paths"], ["version"], ["usd", "validate", "tests/data/simple_scene.usda"]],
    )
    def test_every_command_reports_it_in_the_envelope_with_exit_2(
        self, monkeypatch: pytest.MonkeyPatch, argv: List[str]
    ) -> None:
        monkeypatch.setenv("UNREAL__PORT", "70000")
        repo = Path(__file__).resolve().parents[2]
        argv = [str(repo / part) if part.startswith("tests/") else part for part in argv]

        result = runner.invoke(app, ["--json", *argv])

        assert result.exit_code == 2, result.output
        envelope = _envelope(result.stdout)
        assert envelope["error_type"] == "ConfigError"
        assert envelope["details"]["errors"][0]["source"] == "UNREAL__PORT"
        assert "Traceback" not in result.output

    def test_human_mode_prints_the_message_not_a_traceback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from simul.cli import output

        monkeypatch.setenv("UNREAL__PORT", "70000")
        monkeypatch.setattr(output, "is_json_mode", lambda: False)

        result = runner.invoke(app, ["logs", "paths"])

        assert result.exit_code == 2
        assert "unreal.port" in result.output and "UNREAL__PORT" in result.output
        assert "Traceback" not in result.output
        assert result.exception is None or isinstance(result.exception, SystemExit)

    def test_a_list_variable_that_is_not_json_is_named(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SECURITY__ALLOWED_PATHS", "/tmp/a,/tmp/b")

        with pytest.raises(ConfigError, match="SECURITY__ALLOWED_PATHS") as raised:
            get_settings()

        assert raised.value.errors[0]["setting"] == "security.allowed_paths"

    def test_a_bad_value_in_the_file_names_the_file(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "security:\n  allow_script_execution: nope\n")

        with pytest.raises(ConfigError) as raised:
            load_settings(path)

        assert raised.value.errors[0]["source"] == str(path)

    def test_secret_values_are_masked(self) -> None:
        assert config_module._shown_value(("server", "auth_token"), "hunter2-hunter2-hunter2") == "***"
        assert config_module._shown_value(("unreal", "port"), "70000") == "70000"

    def test_the_stdio_server_exits_2_and_keeps_stdout_clean(self, tmp_path: Path) -> None:
        """stdout is the MCP channel: the error goes to stderr."""
        env = {
            **os.environ,
            "HOME": str(tmp_path),
            "PYTHONPATH": os.pathsep.join(filter(None, [SRC, os.environ.get("PYTHONPATH")])),
            "UNREAL__PORT": "70000",
        }

        proc = subprocess.run(
            [sys.executable, "-m", "simul.cli.main", "server", "--backends", "usd"],
            env=env, cwd=str(tmp_path), capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL,
        )

        assert proc.returncode == 2, proc.stderr
        assert proc.stdout == ""
        assert "ConfigError" in proc.stderr and "UNREAL__PORT" in proc.stderr
        assert "Traceback" not in proc.stderr


# ---------------------------------------------------------------------------
# m21: variables that look like settings but name none are reported
# ---------------------------------------------------------------------------
class TestUnknownVariables:
    def test_the_yaml_nesting_spelt_as_a_variable_warns_with_the_flat_name(
        self, monkeypatch: pytest.MonkeyPatch, config_log: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("ISAAC_SIM__BRIDGE__PORT", "9999")

        settings = Settings()

        assert settings.isaac_sim.bridge_port == 8229
        assert any(
            "ISAAC_SIM__BRIDGE__PORT (did you mean ISAAC_SIM__BRIDGE_PORT?)" in record.getMessage()
            for record in config_log.records
        )

    def test_an_unknown_security_variable_fails_the_load(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SECURITY__RATE_LIMITING__REQUESTS_PER_MINUTE", "5")

        with pytest.raises(ConfigError, match="did you mean SECURITY__REQUESTS_PER_MINUTE"):
            Settings()

    def test_names_that_are_fine(self) -> None:
        names = [
            "SERVER__PORT",
            "server__port",
            "LOGGING__COMPONENTS__USD",
            "SERVER__CORS_ORIGINS",  # retired, still loads
            "ISAAC_SIM_PATH",
            "FOO__BAR",
            "PATH",
            "SIMUL_ENV_FILE",
        ]

        assert find_unknown_env_names(names) == []

    def test_suggestions(self) -> None:
        found = {item.name: item.suggestion for item in find_unknown_env_names(
            ["UNREAL__PORTT", "LOGGING__FILE__PATH", "USD__CACHE__ENABLED", "VIEWPORT__NOPE_AT_ALL_XYZ"]
        )}

        assert found == {
            "UNREAL__PORTT": "UNREAL__PORT",
            "LOGGING__FILE__PATH": "LOGGING__FILE_PATH",
            "USD__CACHE__ENABLED": "USD__CACHE_ENABLED",
            "VIEWPORT__NOPE_AT_ALL_XYZ": None,
        }

    def test_validate_config_lists_them_as_warnings(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("UNREAL__PORTT", "1")

        report = inspect_config_file(_write(tmp_path, GOOD_YAML))

        assert report.valid
        assert any("UNREAL__PORTT (did you mean UNREAL__PORT?)" in warning for warning in report.warnings)


# ---------------------------------------------------------------------------
# m9: CLI flags are validated and exit 2 with InvalidArgument
# ---------------------------------------------------------------------------
class TestFlags:
    @pytest.fixture
    def started(self, monkeypatch: pytest.MonkeyPatch) -> List[Settings]:
        seen: List[Settings] = []

        async def _capture(settings: Settings, transport: str, **kwargs: Any) -> None:
            seen.append(settings)

        monkeypatch.setattr(cli_main, "start_mcp_server", _capture)
        monkeypatch.setattr(cli_main, "_is_isaac_reachable", lambda *a, **k: False)
        return seen

    @pytest.mark.parametrize(
        "argv",
        [
            ["server", "--transport", "grpc"],
            ["server", "--log-level", "verbose-garbage"],
            ["server", "--backends", "houdini"],
            ["tools", "--unreal-tools", "fat"],
            ["info", "--blender-mode", "guess"],
        ],
    )
    def test_a_bad_flag_exits_2_with_invalid_argument(self, started: List[Settings], argv: List[str]) -> None:
        result = runner.invoke(app, ["--json", *argv])

        assert result.exit_code == 2, result.output
        assert _envelope(result.stdout)["error_type"] == "InvalidArgument"
        assert started == []

    def test_log_level_is_case_insensitive(self, started: List[Settings]) -> None:
        result = runner.invoke(app, ["server", "--backends", "usd", "--log-level", "warning"])

        assert result.exit_code == 0, result.output
        assert started[-1].logging.level == "WARNING"

    def test_a_section_change_is_validated(self) -> None:
        with pytest.raises(click.exceptions.Exit) as raised:
            cli_main._with_section(Settings(), "logging", flag="--log-level", level="verbose-garbage")

        assert raised.value.exit_code == 2
