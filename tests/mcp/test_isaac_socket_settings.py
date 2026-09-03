"""Settings for the stock Isaac Sim Python socket: protocol flavour and auth token."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_path))

from simul_mcp.config import IsaacSimConfig, Settings  # noqa: E402


def test_defaults_probe_and_send_no_token() -> None:
    config = IsaacSimConfig()
    assert config.socket_protocol == "auto"
    assert config.socket_auth_token is None


def test_env_overrides_protocol_and_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ISAAC_SIM__SOCKET_PROTOCOL", "python_server")
    monkeypatch.setenv("ISAAC_SIM__SOCKET_AUTH_TOKEN", "abc123")
    settings = Settings()
    assert settings.isaac_sim.socket_protocol == "python_server"
    assert settings.isaac_sim.socket_auth_token == "abc123"


def test_unknown_protocol_is_rejected() -> None:
    with pytest.raises(ValueError):
        IsaacSimConfig(socket_protocol="telnet")  # type: ignore[arg-type]


def test_socket_protocol_env_override_survives_the_shipped_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The documented escape hatch must actually work against config/isaac/default.yaml.

    pydantic-settings lets an explicitly passed value beat the environment, and
    the loader passes every key the YAML defines. A key added to the shipped
    config therefore silently disables its own env var.
    """
    import simul_mcp.config as config_module

    config_path = (
        Path(__file__).resolve().parents[2] / "config" / "isaac" / "default.yaml"
    )
    monkeypatch.setenv("CONFIG_FILE", str(config_path))
    monkeypatch.setenv("ISAAC_SIM__SOCKET_PROTOCOL", "vscode")
    config_module._get_cached_settings.cache_clear()
    try:
        assert config_module.get_settings().isaac_sim.socket_protocol == "vscode"
    finally:
        config_module._get_cached_settings.cache_clear()
