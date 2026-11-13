"""
Configuration management for Simul.

This module provides Pydantic-based configuration management with support for
environment variables, YAML files, and validation.
"""

import difflib
import json
import logging
import os
import re
from contextvars import ContextVar
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from functools import lru_cache
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Mapping, Optional, Set, Tuple, Union, cast, get_origin

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    SettingsError,
    YamlConfigSettingsSource,
)

from .resources import find_checkout_root, resource_filesystem_path
from .tool_surfaces import THIN_BLENDER_TOOLS, THIN_UNREAL_TOOLS, describe_tools
from .utils.paths import STATE_DIR_VARIABLE, default_state_path, state_dir

_CHECKOUT_ROOT: Optional[Path] = find_checkout_root()
_DEFAULT_CONFIG_FILE: Path = resource_filesystem_path("config", "default.yaml")
_ENV_PLACEHOLDER_PATTERN = re.compile(r"\$\{([^}]+)\}")
_LOGGER = logging.getLogger(__name__)

# Names the one .env file the settings read. Unset, no .env is read at all.
ENV_FILE_VARIABLE = "SIMUL_ENV_FILE"

# Working-directory .env files already warned about, by (path, mtime).
_IGNORED_DOTENV_WARNED: Set[Tuple[str, float]] = set()


class ConfigError(Exception):
    """The configuration cannot be used as given.

    Raised for a config file that was named but is missing, is not YAML or
    is not a mapping; for a value that fails validation (from the
    environment, the env file or the YAML); and for an unknown key under
    ``security``, where a typo would silently leave a protection off. The CLI
    reports it as ``error_type: ConfigError`` with exit code 2.

    Deliberately not a ``ValueError``: command handlers that catch
    ``ValueError`` for their own argument problems must let it through to the
    one place that reports it.

    Attributes:
        message: One line naming every problem.
        errors: One entry per problem, with ``reason`` and, when known,
            ``setting`` (``unreal.port``), ``source`` (``UNREAL__PORT``, a
            file) and ``value``.
    """

    def __init__(self, message: str, errors: Optional[List[Dict[str, Any]]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.errors: List[Dict[str, Any]] = errors if errors is not None else [{"reason": message}]


def explicit_env_file() -> Optional[Path]:
    """Return the ``.env`` file ``$SIMUL_ENV_FILE`` names, or None when it is unset.

    Returns:
        The absolute path of the file.

    Raises:
        ConfigError: When the variable names a file that does not exist.
    """
    raw = os.environ.get(ENV_FILE_VARIABLE, "").strip()
    if not raw:
        return None
    path = Path(os.path.expandvars(raw)).expanduser().absolute()
    if not path.is_file():
        raise ConfigError(f"{ENV_FILE_VARIABLE} names no file: {path}")
    return path


def _warn_about_ignored_dotenv() -> None:
    """Warn once when the working directory's ``.env`` sets simul variables.

    Earlier releases read that file. It is ignored now, so a user who kept
    settings there learns why they stopped applying; a project ``.env`` that
    sets nothing simul reads stays silent.
    """
    candidate = Path.cwd() / ".env"
    try:
        key = (str(candidate), candidate.stat().st_mtime)
    except OSError:
        return
    if key in _IGNORED_DOTENV_WARNED:
        return
    _IGNORED_DOTENV_WARNED.add(key)
    try:
        from dotenv import dotenv_values

        names = [str(name) for name in dotenv_values(candidate)]
    except Exception:  # noqa: BLE001 - an unparsable project file is not ours to report
        return
    prefixes = tuple(f"{section.upper()}__" for section in Settings.model_fields)
    ours = sorted(name for name in names if name.upper().startswith(prefixes) or name.upper() == "CONFIG_FILE")
    if ours:
        _LOGGER.warning(
            "Ignoring %s, which sets %s: simul does not read a .env from the working "
            "directory. Export the variables, or set %s=%s to load that file.",
            candidate,
            ", ".join(ours),
            ENV_FILE_VARIABLE,
            candidate,
        )


def _check_scan_port_range(start: int, end: int) -> None:
    """Raise ``ValueError`` when a discovery scan range would be empty."""
    if start >= end:
        raise ValueError(
            f"scan_port_start ({start}) must be less than scan_port_end ({end})"
        )


class ServerConfig(BaseModel):
    """MCP Server configuration.

    ``host`` and ``port`` only matter for ``--transport http`` and ``sse``; the
    remaining fields harden those network transports (see
    ``simul.mcp.http_security``). Over stdio the client spawns the server and
    none of them apply.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(
        default="simul",
        description="Server display name for MCP clients",
    )
    host: str = Field(default="localhost", description="Server host")
    port: int = Field(default=8765, description="Server port", ge=1024, le=65535)
    timeout: int = Field(default=30, description="Connection timeout in seconds", ge=1)
    allow_public: bool = Field(
        default=False,
        description=(
            "Let --transport http/sse bind a host that is not loopback (0.0.0.0, a LAN "
            "address, a host name). Off, the server refuses to start on such a host. "
            "Env: SERVER__ALLOW_PUBLIC; CLI: simul server --allow-public."
        ),
    )
    host_origin_protection: bool = Field(
        default=True,
        description=(
            "Refuse HTTP/SSE requests whose Host header is not a loopback name, the "
            "bound host or one of allowed_hosts (421), and browser requests whose "
            "Origin is neither same-origin, loopback nor one of allowed_origins (403). "
            "This is the DNS-rebinding defense; switching it off lets any web page the "
            "user opens reach the tools. Env: SERVER__HOST_ORIGIN_PROTECTION."
        ),
    )
    allowed_hosts: List[str] = Field(
        default_factory=list,
        description=(
            "Extra Host header names to accept (port ignored, fnmatch wildcards), for "
            "example the DNS name clients use to reach a public bind. Loopback names "
            "and the bound address are always accepted. Env: SERVER__ALLOWED_HOSTS."
        ),
    )
    allowed_origins: List[str] = Field(
        default_factory=list,
        description=(
            "Extra browser origins to accept, as scheme://host[:port] (fnmatch "
            "wildcards). Env: SERVER__ALLOWED_ORIGINS."
        ),
    )
    auth_token: Optional[SecretStr] = Field(
        default=None,
        description=(
            "Bearer token every HTTP/SSE request must carry as 'Authorization: Bearer "
            "<token>' (at least 16 characters). Prefer auth_token_file. "
            "Env: SERVER__AUTH_TOKEN."
        ),
    )
    auth_token_file: Optional[str] = Field(
        default=None,
        description=(
            "File holding the bearer token on its first line; it must be readable by "
            "its owner only (chmod 600). Env: SERVER__AUTH_TOKEN_FILE."
        ),
    )


class IsaacInstanceConfig(BaseModel):
    """Configuration for a single Isaac Sim instance."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Human-readable instance identifier")
    host: str = Field(default="127.0.0.1", description="TCP socket host")
    port: int = Field(default=8226, description="TCP socket port", ge=1024, le=65535)
    timeout: float = Field(default=30.0, description="Socket timeout in seconds", gt=0.0)
    bridge_enabled: bool = Field(
        default=True,
        description="Prefer the custom bridge extension for this instance",
    )
    bridge_host: Optional[str] = Field(
        default=None,
        description="Override host for the custom bridge extension",
    )
    bridge_port: Optional[int] = Field(
        default=None,
        description="Override port for the custom bridge extension",
        ge=1024,
        le=65535,
    )
    bridge_timeout: Optional[float] = Field(
        default=None,
        description="Override timeout for bridge transport operations",
        gt=0.0,
    )
    bridge_fallback_to_vscode: bool = Field(
        default=True,
        description="Allow fallback to the VS Code socket for this instance",
    )
    socket_protocol: Optional[Literal["auto", "python_server", "vscode"]] = Field(
        default=None,
        description="Override the stock Python socket flavour for this instance",
    )
    socket_auth_token: Optional[str] = Field(
        default=None,
        description="Override the python_server auth token for this instance",
    )


class IsaacSimConfig(BaseModel):
    """Isaac Sim configuration."""

    model_config = ConfigDict(frozen=True)

    enabled: Optional[bool] = Field(
        default=None,
        description=(
            "Register the Isaac Sim MCP tools. Unset (or 'auto') registers them on "
            "Linux and Windows, the platforms Isaac Sim runs on, and skips them "
            "elsewhere (macOS) unless `--backends` names isaac; true registers them "
            "on any OS; false never registers them. Env: ISAAC_SIM__ENABLED."
        ),
    )
    path: Optional[str] = Field(
        # Read from the environment so the field is the one source of truth.
        # It existed and was never populated, while callers reached past it to
        # os.environ directly.
        default_factory=lambda: os.environ.get("ISAAC_SIM_PATH") or None,
        description="Path to Isaac Sim installation (defaults to $ISAAC_SIM_PATH)",
    )

    # TCP socket connection to the stock Isaac Sim Python socket server:
    # isaacsim.code_editor.python_server on 6.0+, isaacsim.code_editor.vscode on 5.x
    socket_host: str = Field(
        default="127.0.0.1",
        description="Host for the Isaac Sim Python socket server",
    )
    socket_port: int = Field(
        default=8226,
        description="Port for the Isaac Sim Python socket server",
        ge=1024,
        le=65535,
    )
    socket_timeout: float = Field(
        default=30.0,
        description="Timeout in seconds for Isaac Sim TCP socket operations",
        gt=0.0,
    )
    socket_protocol: Literal["auto", "python_server", "vscode"] = Field(
        default="auto",
        description=(
            "Stock socket flavour: 'python_server' (Isaac Sim 6.0+, executes after "
            "TCP half-close), 'vscode' (Isaac Sim 5.x), or 'auto' to probe once"
        ),
    )
    socket_auth_token: Optional[str] = Field(
        default=None,
        description=(
            "Auth token for an Isaac Sim 6.0+ python_server started with "
            "require_auth=true; sent as the '# isaacsim-python-server-token:' header. "
            "Precedence: an explicit value (ISAAC_SIM__SOCKET_AUTH_TOKEN, the SIMUL_ENV_FILE file or YAML) "
            "wins; when unset, the token file `simul isaac launch "
            "--generate-auth-token` wrote to <discovery_dir>/auth-token-<pid> for the "
            "newest still-running editor is used"
        ),
    )
    bridge_enabled: bool = Field(
        default=True,
        description="Prefer the custom Isaac Sim bridge extension when available",
    )
    bridge_host: str = Field(
        default="127.0.0.1",
        description="Host for the custom Isaac Sim bridge extension",
    )
    bridge_port: int = Field(
        default=8229,
        description="Port for the custom Isaac Sim bridge extension",
        ge=1024,
        le=65535,
    )
    bridge_timeout: float = Field(
        default=30.0,
        description="Timeout in seconds for bridge transport operations",
        gt=0.0,
    )
    bridge_fallback_to_vscode: bool = Field(
        default=True,
        description="Fallback to the VS Code extension transport when the bridge is unavailable",
    )
    bridge_failure_threshold: int = Field(
        default=3,
        description=(
            "Consecutive bridge transport failures after which the client stops "
            "dialling the bridge and goes straight to the stock socket"
        ),
        ge=1,
    )
    bridge_cooldown_seconds: float = Field(
        default=30.0,
        description=(
            "How long an opened bridge circuit skips the bridge before probing "
            "it again; a successful probe closes the circuit"
        ),
        ge=0.0,
    )

    # Multi-instance support
    instances: List[IsaacInstanceConfig] = Field(
        default_factory=list,
        description="Additional named Isaac Sim instances beyond the default",
    )
    scan_port_start: int = Field(
        default=8226,
        description="Start port for auto-discovery scan",
        ge=1024,
        le=65535,
    )
    scan_port_end: int = Field(
        default=8236,
        description="End port (exclusive) for auto-discovery scan",
        ge=1024,
        le=65535,
    )
    discovery_dir: str = Field(
        default="/tmp/simul",
        description="Directory where bridge extensions write port discovery files",
    )
    enforce_claims: bool = Field(
        default=False,
        description=(
            "Turn claim_isaac_instance into a lock: while another agent holds a live "
            "claim on an instance, mutating Isaac tools from anyone else are refused "
            "with InstanceClaimed. Off means claims are advisory"
        ),
    )

    @field_validator("enabled", mode="before")
    @classmethod
    def _auto_means_unset(cls, value: Any) -> Any:
        """Read ``auto`` or an empty value as unset, the platform default."""
        if isinstance(value, str) and value.strip().lower() in ("", "auto"):
            return None
        return value

    @model_validator(mode="after")
    def _pick_up_generated_auth_token(self) -> "IsaacSimConfig":
        """Fall back to the token file ``simul isaac launch`` left in the discovery dir.

        Only when nothing explicit was configured: the environment, the env file and
        YAML all rank above the file. The file is read only from a discovery
        directory that is owned by the current user and writable by nobody else.
        """
        if self.socket_auth_token is not None:
            return self
        # Imported here: simul.utils pulls in simul.logging, which imports this module.
        from .utils.discovery import DiscoveryDir

        token = DiscoveryDir(self.discovery_dir).read_auth_token()
        if token is None:
            return self
        return self.model_copy(update={"socket_auth_token": token})

    @property
    def path_error(self) -> Optional[str]:
        """Explain why ``path`` is not a usable Isaac Sim install root.

        A stale ``ISAAC_SIM_PATH`` must not stop the other backends from
        loading, so the problem is reported here instead of rejecting the
        model. Callers that actually need the install (``simul isaac
        launch``, ``install-bridge``) surface it when they resolve the root.

        Returns:
            A human-readable problem description, or ``None`` when the path is
            unset or names a directory holding ``python.sh``/``python.bat``.
        """
        if self.path is None:
            return None
        root = Path(self.path).expanduser()
        if not root.exists():
            return f"Isaac Sim path does not exist: {self.path}"
        if not (root / "python.sh").exists() and not (root / "python.bat").exists():
            return f"Isaac Sim python executable not found in: {self.path}"
        return None

    @model_validator(mode="after")
    def _validate_port_range(self) -> "IsaacSimConfig":
        """Ensure scan_port_start < scan_port_end to avoid empty scan ranges."""
        _check_scan_port_range(self.scan_port_start, self.scan_port_end)
        return self


class BlenderConfig(BaseModel):
    """Blender runtime configuration."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(default=True, description="Enable Blender runtime tools")
    mode: Literal["embedded", "attached"] = Field(
        default="embedded", description="Use local bpy or an explicitly attached Blender window"
    )
    discovery_dir: str = Field(
        default_factory=lambda: default_state_path("blender"),
        description="Local Blender bridge discovery directory (default $SIMUL_HOME/blender, else ~/.simul/blender)",
    )
    attachment_path: str = Field(
        default_factory=lambda: default_state_path("blender", "attachment.json"),
        description="Target written by simul blender attach (default under the state directory)",
    )
    connection_timeout: float = Field(default=30.0, gt=0, le=3600)
    binary_path: Optional[str] = Field(
        default=None,
        description="Path to Blender binary when explicitly configured",
    )
    default_collection: Optional[str] = Field(
        default=None,
        description="Default Blender collection filter for scene listing",
    )
    max_scene_objects: int = Field(
        default=200,
        description="Default maximum scene objects returned from Blender tools",
        ge=1,
        le=5000,
    )
    tool_surface: Literal["thin", "full"] = Field(
        default="full",
        description=(
            f"Which Blender tools the MCP server registers. 'full' (default) exposes "
            f"every Blender and SimReady tool; 'thin' exposes only "
            f"{describe_tools(THIN_BLENDER_TOOLS)}. Env: BLENDER__TOOL_SURFACE."
        ),
    )

    @field_validator("binary_path")
    @classmethod
    def validate_binary_path(cls, v: Optional[str]) -> Optional[str]:
        """Validate Blender binary path when provided."""
        if v is not None:
            path = Path(v)
            if not path.exists():
                raise ValueError(f"Blender binary path does not exist: {v}")
            if not path.is_file():
                raise ValueError(f"Blender binary path is not a file: {v}")
        return v


class UnrealConfig(BaseModel):
    """Unreal Engine Remote Control API configuration."""

    model_config = ConfigDict(frozen=True)

    mode: Literal["endpoint", "attached"] = Field(
        "endpoint", description="Use configured host/port or the explicitly attached editor"
    )
    attachment_path: str = Field(
        default_factory=lambda: default_state_path("unreal", "attachment.json"),
        description="Saved editor/map/viewport selection (default under the state directory)",
    )

    enabled: bool = Field(default=True, description="Enable Unreal Engine runtime tools")
    host: str = Field(default="localhost", description="Remote Control API host")
    port: int = Field(
        default=30010,
        description="Remote Control API port",
        ge=1024,
        le=65535,
    )
    timeout: int = Field(
        default=30,
        description="HTTP request timeout in seconds",
        ge=1,
    )
    import_export_timeout: int = Field(
        default=300,
        description=(
            "Timeout in seconds for the asset import and export tools (import_unreal_usd, "
            "export_unreal_usd), which can run for minutes: the first USD export of an editor "
            "session loads the USD Python modules, 25-40 s on UE 5.7. Never shorter than timeout. "
            "Env: UNREAL__IMPORT_EXPORT_TIMEOUT."
        ),
        ge=1,
    )
    embedded_mode: bool = Field(
        default=False,
        description="Running inside UE5 Python interpreter",
    )
    max_actors: int = Field(
        default=200,
        description=(
            "Default max_results of list_unreal_actors and simul unreal list-actors. "
            "Env: UNREAL__MAX_ACTORS."
        ),
        ge=1,
        le=5000,
    )
    tool_surface: Literal["thin", "full"] = Field(
        default="thin",
        description=(
            f"Which Unreal tools the MCP server registers. 'thin' exposes only "
            f"{describe_tools(THIN_UNREAL_TOOLS)}; 'full' exposes every granular "
            f"Unreal tool. Env: UNREAL__TOOL_SURFACE."
        ),
    )

    # Retry / resilience
    max_retries: int = Field(
        default=3,
        description="Maximum HTTP request retries on transient failures",
        ge=0,
        le=10,
    )
    retry_base_delay: float = Field(
        default=0.5,
        description="Base delay in seconds for exponential backoff between retries",
        gt=0.0,
        le=10.0,
    )
    ping_timeout: float = Field(
        default=3.0,
        description="Short timeout in seconds used for ping / discovery probes",
        gt=0.0,
        le=30.0,
    )

    # Auth (matches simul unreal setup --passphrase). When set, the
    # Remote Control client sends a `Passphrase: <md5>` HTTP header on every
    # request. Accepts either the raw plaintext (we MD5-hash it on send) or
    # a pre-computed 32-character hex digest (we pass it through). Required
    # whenever the editor was started with --passphrase enabled —
    # without it the editor returns 401 "Given Passphrase is not correct!"
    # Read from env var UNREAL__PASSPHRASE (or the file SIMUL_ENV_FILE names).
    passphrase: Optional[str] = Field(
        default=None,
        description=(
            "Plaintext passphrase or pre-hashed MD5 hex for UE Remote "
            "Control auth. None = client sends no Passphrase header. "
            "Env: UNREAL__PASSPHRASE."
        ),
    )

    # Multi-instance discovery
    scan_port_start: int = Field(
        default=30010,
        description="Start port for auto-discovery scan of Unreal instances",
        ge=1024,
        le=65535,
    )
    scan_port_end: int = Field(
        default=30020,
        description="End port (exclusive) for auto-discovery scan",
        ge=1024,
        le=65535,
    )

    @model_validator(mode="after")
    def _validate_port_range(self) -> "UnrealConfig":
        """Ensure scan_port_start < scan_port_end to avoid empty scan ranges."""
        _check_scan_port_range(self.scan_port_start, self.scan_port_end)
        return self


class USDConfig(BaseModel):
    """USD configuration."""

    model_config = ConfigDict(frozen=True)

    cache_enabled: bool = Field(default=True, description="Enable USD caching")
    allowed_extensions: List[str] = Field(
        default_factory=lambda: [".usd", ".usda", ".usdc", ".usdz"],
        description="Allowed USD file extensions",
    )
    max_file_size_mb: int = Field(
        default=500, description="Maximum USD file size in MB", ge=1
    )
    validation_timeout_seconds: float = Field(
        default=300.0,
        description=(
            "Longest validate_usd_file may take when it checks a file in a separate process "
            "(because a loaded stage holds an older copy of a file it uses)"
        ),
        gt=0,
    )


class ViewportConfig(BaseModel):
    """Viewport configuration."""

    model_config = ConfigDict(frozen=True)

    max_size: int = Field(default=2048, description="Maximum viewport size", ge=640)
    capture_dir: Optional[str] = Field(
        default=None,
        description=(
            "Directory viewport captures are written to. Must sit inside the sandbox "
            "when it is enabled; None picks <writable allowed root>/captures"
        ),
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""

    model_config = ConfigDict(frozen=True)

    level: str = Field(default="INFO", description="Log level")
    format: str = Field(default="detailed", description="Log format")
    file_enabled: bool = Field(default=True, description="Enable file logging")
    file_path: str = Field(
        default_factory=lambda: default_state_path("logs", "simul.log"),
        description="Log file path (default $SIMUL_HOME/logs/simul.log, else ~/.simul/logs/simul.log)",
    )
    console_enabled: bool = Field(default=True, description="Enable console logging")
    console_colored: bool = Field(
        default=True, description="Enable colored console output"
    )
    components: Dict[str, str] = Field(
        default_factory=dict,
        description="Component-specific logging levels",
    )

    # --- structured / observability extensions -----------------------------
    per_instance: bool = Field(
        default=False,
        description=(
            "Suffix log filenames with the process PID so concurrent server "
            "instances do not interleave into the same file."
        ),
    )
    retention_days: int = Field(
        default=14,
        ge=1,
        description=(
            "Number of daily log rotations to keep "
            "(applies to TimedRotatingFileHandler.backupCount)."
        ),
    )
    audit_enabled: bool = Field(
        default=True,
        description="Write a per-tool-call audit JSONL file alongside main logs.",
    )
    audit_path: str = Field(
        default_factory=lambda: default_state_path("logs", "audit.jsonl"),
        description="Destination of the per-tool-call audit JSONL stream.",
    )
    structured_enabled: bool = Field(
        default=False,
        description=(
            "Also write the structured JSON file (simul.json) for every "
            "log record. Off by default because the audit JSONL covers most "
            "tooling needs and double-formatting on the hot path is wasteful."
        ),
    )

    @field_validator("level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()


class SecurityConfig(BaseModel):
    """Security configuration."""

    model_config = ConfigDict(frozen=True)

    sandbox_enabled: bool = Field(default=True, description="Enable sandbox mode")
    allowed_paths: List[str] = Field(
        default_factory=lambda: ["examples", "tests/data", "/tmp/simul-work"],
        description="Allowed file paths",
    )
    allowed_url_schemes: List[str] = Field(
        default_factory=lambda: ["omniverse"],
        description="URL schemes that may be read from (opened, imported, referenced)",
    )
    allowed_write_url_schemes: List[str] = Field(
        default_factory=list,
        description="URL schemes that may be written to; empty keeps writes local-only",
    )
    relative_path_base: Optional[str] = Field(
        default=None,
        description=(
            "Directory a relative path passed to a tool resolves against; unset uses "
            "the directory the server was started in"
        ),
    )
    allow_script_execution: bool = Field(
        default=True,
        description=(
            "Register the arbitrary-code tools (execute_isaac_script, "
            "execute_unreal_script, execute_blender_script) and Unreal's generic "
            "function/batch dispatch. Fixed granular tools stay available when this is off."
        ),
    )
    rate_limiting_enabled: bool = Field(
        default=True, description="Enable rate limiting"
    )
    requests_per_minute: int = Field(
        default=60, description="Requests per minute limit", ge=1
    )
    burst_size: int = Field(
        default=10, description="Burst size for rate limiting", ge=1
    )
    global_requests_per_minute: int = Field(
        default=600,
        description="Requests per minute one agent may make across all tools",
        ge=1,
    )


class NormalisedYamlSettingsSource(YamlConfigSettingsSource):
    """YAML settings source that flattens the repo's nested layout first.

    The shipped ``default.yaml`` groups keys the way operators read them
    (``isaac_sim.bridge.port``) while the Settings model is flat within each
    section (``isaac_sim.bridge_port``). ``_normalise_settings_payload``
    bridges the two before pydantic-settings merges this source below the
    environment.
    """

    def _read_file(self, file_path: Union[Path, Traversable]) -> Dict[str, Any]:
        """Load one YAML file and return it in the Settings model shape.

        A file that cannot be used fails the whole load: before, a syntax
        error made the file contribute nothing, so a config that said
        ``allow_script_execution: false`` ran with scripts on and only a
        stderr line to show for it. Unknown keys are reported through
        ``_report_unknown_settings``.

        Args:
            file_path: YAML file to read, a filesystem path or (from
                pydantic-settings 2.15) a packaged resource; pydantic-settings
                only calls this for files that exist.

        Returns:
            The normalised payload; ``{}`` for an empty file.

        Raises:
            ConfigError: When the file cannot be read, is not YAML, does not
                hold a mapping of sections, or names an unknown security key.
        """
        raw = read_config_yaml(file_path, encoding=self.yaml_file_encoding or "utf-8")
        _report_unknown_settings(find_unknown_yaml_keys(raw), source=str(file_path))
        try:
            return _normalise_settings_payload(raw)
        except ValueError as exc:
            raise ConfigError(f"Config file {file_path}: {exc}", [{"source": str(file_path), "reason": str(exc)}])


def read_config_yaml(file_path: Union[Path, Traversable], encoding: str = "utf-8") -> Dict[str, Any]:
    """Parse a config file into its raw mapping of sections.

    Args:
        file_path: The YAML file.
        encoding: Its text encoding.

    Returns:
        The top-level mapping; ``{}`` for an empty file.

    Raises:
        ConfigError: When the file cannot be read, is not YAML, or does not
            hold a mapping.
    """
    try:
        with file_path.open(encoding=encoding) as handle:
            raw = yaml.safe_load(handle)
    except OSError as exc:
        reason = f"cannot read it: {exc.strerror or exc}"
        raise ConfigError(f"Config file {file_path}: {reason}", [{"source": str(file_path), "reason": reason}])
    except yaml.YAMLError as exc:
        reason = f"not valid YAML: {' '.join(str(exc).split())}"
        raise ConfigError(f"Config file {file_path} is {reason}", [{"source": str(file_path), "reason": reason}])
    except UnicodeDecodeError as exc:
        reason = f"not {encoding} text: {exc}"
        raise ConfigError(f"Config file {file_path} is {reason}", [{"source": str(file_path), "reason": reason}])
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        reason = f"must hold a mapping of sections, not a {type(raw).__name__}"
        raise ConfigError(f"Config file {file_path} {reason}", [{"source": str(file_path), "reason": reason}])
    return raw


class _SectionEnvMixin:
    """Skip a bare section-name variable that is not a JSON object.

    With no prefix and case-insensitive names, ``SERVER=prod`` or
    ``LOGGING=1`` (ordinary variable names in many shells, CI systems and
    ``.env`` files) were read as the whole ``server`` / ``logging`` section,
    failed to parse as JSON, and stopped every simul command. Only
    ``SECTION__FIELD`` names are documented; a bare section name still
    applies when its value is a JSON object, as it did, and is ignored
    otherwise.
    """

    def get_field_value(self, field: Any, field_name: str) -> Tuple[Any, str, bool]:
        value, key, is_complex = super().get_field_value(field, field_name)  # type: ignore[misc]
        if isinstance(value, str) and field_name in _section_names() and not _is_json_object(value):
            _LOGGER.debug(
                "Ignoring %s=%r: not a JSON object, and simul settings are named SECTION__FIELD",
                key.upper(),
                value,
            )
            return None, key, is_complex
        return value, key, is_complex


class _SimulEnvSettingsSource(_SectionEnvMixin, EnvSettingsSource):
    """The process environment, with bare section names handled as ``_SectionEnvMixin`` says."""


class _SimulDotEnvSettingsSource(_SectionEnvMixin, DotEnvSettingsSource):
    """The ``SIMUL_ENV_FILE`` file, with bare section names handled as ``_SectionEnvMixin`` says."""


def _is_json_object(value: str) -> bool:
    try:
        return isinstance(json.loads(value), dict)
    except ValueError:
        return False


class Settings(BaseSettings):
    """Main settings class that combines all configuration sections.

    Value precedence per leaf, highest first: constructor arguments, process
    environment, the ``.env`` file named by ``$SIMUL_ENV_FILE`` (none by
    default), the YAML file named by ``model_config["yaml_file"]`` (set through
    :meth:`from_yaml`), model defaults.

    A ``.env`` in the working directory is never read: MCP clients start the
    server in the user's project, so a cloned repository's ``.env`` could
    otherwise widen the sandbox or re-enable script execution.
    """

    # Configuration sections
    server: ServerConfig = Field(default_factory=ServerConfig)
    isaac_sim: IsaacSimConfig = Field(default_factory=IsaacSimConfig)
    blender: BlenderConfig = Field(default_factory=BlenderConfig)
    unreal: UnrealConfig = Field(default_factory=UnrealConfig)
    usd: USDConfig = Field(default_factory=USDConfig)
    viewport: ViewportConfig = Field(default_factory=ViewportConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    model_config = SettingsConfigDict(
        # No implicit .env: see the class docstring and SIMUL_ENV_FILE.
        env_file=None,
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        """Rank the YAML file below the environment and the explicit ``.env``.

        Args:
            settings_cls: The concrete settings class being built.
            init_settings: Constructor-argument source.
            env_settings: Process-environment source.
            dotenv_settings: ``.env`` file source; replaced by one reading
                ``$SIMUL_ENV_FILE`` when that is set.
            file_secret_settings: Secrets-directory source.

        Returns:
            The sources in priority order, highest first.

        Raises:
            ConfigError: When ``SIMUL_ENV_FILE`` names a file that does not
                exist, or a variable names an unknown security setting.
        """
        env_file = explicit_env_file()
        env_names: List[Tuple[str, str]] = [(name, "the environment") for name in os.environ]
        if env_file is not None:
            dotenv_settings = _SimulDotEnvSettingsSource(settings_cls, env_file=env_file)
            env_names += [(name, str(env_file)) for name in _dotenv_names(env_file)]
        else:
            _warn_about_ignored_dotenv()
        for source in dict.fromkeys(label for _name, label in env_names):
            _report_unknown_settings(
                find_unknown_env_names(name for name, label in env_names if label == source), source=source
            )
        return (
            init_settings,
            _SimulEnvSettingsSource(settings_cls),
            dotenv_settings,
            NormalisedYamlSettingsSource(settings_cls),
            file_secret_settings,
        )

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "Settings":
        """Build settings backed by a YAML file that the environment still overrides.

        Args:
            config_path: YAML file in the repo's nested layout. A missing file
                contributes nothing; callers decide whether that is an error.

        Returns:
            Settings merged from defaults, the YAML file, ``.env`` and the
            environment, in increasing precedence.
        """
        yaml_backed_config = SettingsConfigDict(**cls.model_config)
        yaml_backed_config["yaml_file"] = str(config_path)
        yaml_backed_cls = cast(
            type[Settings],
            type(f"{cls.__name__}FromYaml", (cls,), {"model_config": yaml_backed_config}),
        )
        return yaml_backed_cls()

    @model_validator(mode="after")
    def _warn_about_unusable_isaac_path(self) -> "Settings":
        """Log a stale Isaac Sim install path instead of failing every backend."""
        if self.isaac_sim.path_error is not None:
            _LOGGER.warning(
                "%s. Isaac Sim tools and `simul isaac launch` report this when they "
                "need the install; other backends are unaffected.",
                self.isaac_sim.path_error,
            )
        return self


def _compact_dict(values: Dict[str, Any]) -> Dict[str, Any]:
    """Drop None values so Settings can keep model defaults."""
    return {key: value for key, value in values.items() if value is not None}


def _coalesce(*values: Any) -> Any:
    """Return the first non-None value."""
    for value in values:
        if value is not None:
            return value
    return None


def _resolve_config_file(config_file: Union[str, Path]) -> Path:
    """Resolve config paths relative to cwd first, then the source checkout if any."""
    candidate = Path(os.path.expandvars(str(config_file))).expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.exists() or _CHECKOUT_ROOT is None:
        return candidate.resolve()
    return (_CHECKOUT_ROOT / candidate).resolve()


def _expand_string(value: str, env: Dict[str, str]) -> str:
    """Expand ${VAR} placeholders using environment values."""

    def _replace(match: re.Match[str]) -> str:
        var_name = match.group(1)
        return env.get(var_name, match.group(0))

    return _ENV_PLACEHOLDER_PATTERN.sub(_replace, value)


def _expand_placeholders(value: Any, env: Dict[str, str]) -> Any:
    """Recursively expand ${VAR} placeholders in nested YAML data."""
    if isinstance(value, dict):
        return {key: _expand_placeholders(item, env) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_placeholders(item, env) for item in value]
    if isinstance(value, str):
        return _expand_string(value, env)
    return value


def _normalise_optional_path(value: Any) -> Optional[str]:
    """Treat unresolved placeholders in optional path fields as unset."""
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or _ENV_PLACEHOLDER_PATTERN.search(stripped):
            return None
        return stripped
    return str(value)


# Nested YAML groups, by section: group -> {key inside the group: Settings field}.
# ``isaac_sim.bridge.port`` in YAML is the ``isaac_sim.bridge_port`` field; the
# flat field name is accepted too and wins when both are given. The normaliser
# and the unknown-key check both read this table, so they cannot drift apart.
_YAML_GROUPS: Dict[str, Dict[str, Dict[str, str]]] = {
    "isaac_sim": {
        "bridge": {
            "enabled": "bridge_enabled",
            "host": "bridge_host",
            "port": "bridge_port",
            "timeout": "bridge_timeout",
            "fallback_to_vscode": "bridge_fallback_to_vscode",
            "failure_threshold": "bridge_failure_threshold",
            "cooldown_seconds": "bridge_cooldown_seconds",
        },
    },
    "usd": {
        "cache": {"enabled": "cache_enabled"},
        "files": {"allowed_extensions": "allowed_extensions", "max_file_size_mb": "max_file_size_mb"},
        "validation": {"timeout_seconds": "validation_timeout_seconds"},
    },
    "viewport": {"capture": {"max_size": "max_size", "directory": "capture_dir"}},
    "logging": {
        "file": {
            "enabled": "file_enabled",
            "path": "file_path",
            "per_instance": "per_instance",
            "retention_days": "retention_days",
        },
        "console": {"enabled": "console_enabled", "colored": "console_colored"},
        "audit": {"enabled": "audit_enabled", "path": "audit_path"},
        "structured": {"enabled": "structured_enabled"},
    },
    "security": {
        "sandbox": {
            "enabled": "sandbox_enabled",
            "allowed_paths": "allowed_paths",
            "allowed_url_schemes": "allowed_url_schemes",
            "allowed_write_url_schemes": "allowed_write_url_schemes",
            "relative_path_base": "relative_path_base",
        },
        "rate_limiting": {
            "enabled": "rate_limiting_enabled",
            "requests_per_minute": "requests_per_minute",
            "burst_size": "burst_size",
            "global_requests_per_minute": "global_requests_per_minute",
        },
    },
}

# The ``bridge`` group of one ``isaac_sim.instances`` entry.
_INSTANCE_BRIDGE_GROUP: Dict[str, str] = {
    "enabled": "bridge_enabled",
    "host": "bridge_host",
    "port": "bridge_port",
    "timeout": "bridge_timeout",
    "fallback_to_vscode": "bridge_fallback_to_vscode",
}

# Fields whose value maps free-form names (logger names) rather than settings.
_FREEFORM_FIELDS: Set[Tuple[str, str]] = {("logging", "components")}

# YAML keys (and whole subtrees) earlier releases shipped and nothing reads any
# more. Files copied from an old default.yaml still carry them, so they load
# without a runtime warning; ``simul validate-config`` lists them for cleanup.
_RETIRED_YAML_KEYS: Set[str] = {
    "mesh",
    "performance",
    "features",
    "development",
    "integrations",
    "server.cors_origins",
    "server.enable_cors",
    "server.max_connections",
    "isaac_sim.headless",
    "isaac_sim.width",
    "isaac_sim.height",
    "isaac_sim.kit",
    "isaac_sim.extensions",
    "isaac_sim.simulation",
    "usd.stage",
    "usd.performance",
    "usd.cache.size",
    "usd.cache.stage_cache_limit",
    "viewport.format",
    "viewport.quality",
    "viewport.fov",
    "viewport.camera",
    "viewport.rendering",
    "viewport.capture.width",
    "viewport.capture.height",
    "viewport.capture.format",
    "viewport.capture.quality",
    "logging.file_max_size",
    "logging.file_backup_count",
    "logging.file.max_size",
    "logging.file.backup_count",
    "logging.file.rotation",
}

# The same retired settings as environment variables (see .env.example history).
_RETIRED_ENV_NAMES: Set[str] = {
    "SERVER__CORS_ORIGINS",
    "SERVER__ENABLE_CORS",
    "SERVER__MAX_CONNECTIONS",
    "ISAAC_SIM__HEADLESS",
    "ISAAC_SIM__WIDTH",
    "ISAAC_SIM__HEIGHT",
    "VIEWPORT__FORMAT",
    "VIEWPORT__QUALITY",
    "VIEWPORT__FOV",
    "LOGGING__FILE_MAX_SIZE",
    "LOGGING__FILE_BACKUP_COUNT",
}


def _section_names() -> Tuple[str, ...]:
    """The Settings sections, in model order."""
    return tuple(Settings.model_fields)


def _section_fields(section: str) -> Tuple[str, ...]:
    """The field names of one Settings section."""
    model = Settings.model_fields[section].annotation
    return tuple(cast(type[BaseModel], model).model_fields)


def _closest(name: str, candidates: Iterable[str]) -> Optional[str]:
    """The candidate ``name`` most likely misspells, or None when none is close."""
    matches = difflib.get_close_matches(name.lower(), list(candidates), n=1, cutoff=0.6)
    return matches[0] if matches else None


def _is_retired(path: str) -> bool:
    """Whether ``path`` (``viewport.capture.format``) sits in a retired subtree."""
    parts = path.split(".")
    return any(".".join(parts[:end]) in _RETIRED_YAML_KEYS for end in range(1, len(parts) + 1))


@dataclass(frozen=True)
class UnknownSetting:
    """A config key or environment variable that names no setting.

    Attributes:
        name: The key as written (``security.allow_scripts_execution``,
            ``ISAAC_SIM__BRIDGE__PORT``).
        suggestion: The setting it most likely meant, spelled the same way,
            or None.
        section: The section it belongs to or, for a misspelt section, the
            one it most likely meant; None when there is no guess.
        retired: True for a setting earlier releases had and nothing reads now.
    """

    name: str
    suggestion: Optional[str] = None
    section: Optional[str] = None
    retired: bool = False

    @property
    def security(self) -> bool:
        """Whether a protection could be silently off because of it."""
        return self.section == "security" and not self.retired

    def describe(self) -> str:
        """One phrase for a log line or a validate-config error."""
        if self.retired:
            return f"{self.name} is a retired setting that nothing reads; remove it"
        hint = f" (did you mean {self.suggestion}?)" if self.suggestion else ""
        return f"unknown setting {self.name}{hint}"

    def as_error(self, source: str) -> Dict[str, Any]:
        """The ``ConfigError.errors`` entry for this key."""
        entry: Dict[str, Any] = {"setting": self.name, "source": source, "reason": self.describe()}
        if self.suggestion:
            entry["suggestion"] = self.suggestion
        return entry


def find_unknown_yaml_keys(raw: Mapping[str, Any], *, include_retired: bool = False) -> List[UnknownSetting]:
    """List the keys of a raw config mapping that no setting reads.

    Args:
        raw: The parsed YAML, before normalisation.
        include_retired: Also list retired keys (flagged ``retired``).

    Returns:
        One entry per unknown key, in file order.
    """
    unknown: List[UnknownSetting] = []
    sections = _section_names()

    def flag(path: str, suggestion: Optional[str], section: Optional[str]) -> None:
        if _is_retired(path):
            if include_retired:
                unknown.append(UnknownSetting(path, None, section, retired=True))
            return
        unknown.append(UnknownSetting(path, suggestion, section))

    for key, value in raw.items():
        section = str(key)
        if section not in sections:
            match = _closest(section, sections)
            flag(section, match, match)
            continue
        if not isinstance(value, dict):
            continue  # the normaliser reports a section that is not a mapping
        fields = _section_fields(section)
        groups = _YAML_GROUPS.get(section, {})
        for sub_key, sub_value in value.items():
            name = str(sub_key)
            path = f"{section}.{name}"
            if name in groups:
                if isinstance(sub_value, dict):
                    for leaf in sub_value:
                        if str(leaf) not in groups[name]:
                            match = _closest(str(leaf), groups[name])
                            flag(f"{path}.{leaf}", f"{path}.{match}" if match else None, section)
                continue
            if name in fields:
                if (section, name) == ("isaac_sim", "instances") and isinstance(sub_value, list):
                    unknown += _unknown_instance_keys(sub_value)
                continue
            match = _closest(name, [*fields, *groups])
            flag(path, f"{section}.{match}" if match else None, section)
    return unknown


def _unknown_instance_keys(instances: List[Any]) -> List[UnknownSetting]:
    """The unknown keys of each ``isaac_sim.instances`` entry."""
    fields = tuple(IsaacInstanceConfig.model_fields)
    unknown: List[UnknownSetting] = []
    for index, item in enumerate(instances):
        if not isinstance(item, dict):
            continue  # the normaliser reports it
        prefix = f"isaac_sim.instances[{index}]"
        for key, value in item.items():
            name = str(key)
            if name == "bridge" and isinstance(value, dict):
                for leaf in value:
                    if str(leaf) not in _INSTANCE_BRIDGE_GROUP:
                        match = _closest(str(leaf), _INSTANCE_BRIDGE_GROUP)
                        suggestion = f"{prefix}.bridge.{match}" if match else None
                        unknown.append(UnknownSetting(f"{prefix}.bridge.{leaf}", suggestion, "isaac_sim"))
            elif name not in fields and name != "bridge":
                match = _closest(name, [*fields, "bridge"])
                unknown.append(UnknownSetting(f"{prefix}.{name}", f"{prefix}.{match}" if match else None, "isaac_sim"))
    return unknown


def find_unknown_env_names(names: Iterable[str], *, include_retired: bool = False) -> List[UnknownSetting]:
    """List the ``SECTION__...`` variable names that no setting reads.

    Only names that start with a section and ``__`` are judged, so unrelated
    variables never are. ``ISAAC_SIM__BRIDGE__PORT`` (the YAML nesting spelt as
    a variable) is unknown and suggests ``ISAAC_SIM__BRIDGE_PORT``.

    Args:
        names: Environment variable names.
        include_retired: Also list retired variables (flagged ``retired``).

    Returns:
        One entry per unknown name.
    """
    unknown: List[UnknownSetting] = []
    sections = _section_names()
    for name in names:
        upper = name.upper()
        head, delimiter, rest = upper.partition("__")
        section = head.lower()
        if not delimiter or not rest or section not in sections:
            continue
        fields = _section_fields(section)
        field = rest.lower()
        first = field.split("__", 1)[0]
        if field in fields or ((section, first) in _FREEFORM_FIELDS and "__" in field):
            continue
        if upper in _RETIRED_ENV_NAMES:
            if include_retired:
                unknown.append(UnknownSetting(name, None, section, retired=True))
            continue
        unknown.append(UnknownSetting(name, _env_suggestion(section, field, fields), section))
    return unknown


def _env_suggestion(section: str, field: str, fields: Tuple[str, ...]) -> Optional[str]:
    """The variable an unknown ``SECTION__<field>`` most likely meant."""
    parts = field.split("__")
    groups = _YAML_GROUPS.get(section, {})
    target: Optional[str]
    if len(parts) == 2 and parts[1] in groups.get(parts[0], {}):
        target = groups[parts[0]][parts[1]]
    elif "_".join(parts) in fields:
        target = "_".join(parts)
    else:
        target = _closest(field, fields)
    return f"{section.upper()}__{target.upper()}" if target else None


# Unknown settings already logged, by (source, name), so each is reported once.
_UNKNOWN_WARNED: Set[Tuple[str, str]] = set()

# Off while ``inspect_config_file`` loads a file whose unknown keys it reports itself.
_REPORT_UNKNOWN: ContextVar[bool] = ContextVar("simul_report_unknown_settings", default=True)


def _report_unknown_settings(unknown: List[UnknownSetting], *, source: str) -> None:
    """Refuse unknown security settings and warn once about every other one.

    Unknown keys outside ``security`` only warn: retired keys must keep
    loading, and a config written for a newer simul should still start an
    older one. Under ``security`` (or a key that most likely meant it) a typo
    such as ``allow_scripts_execution: false`` would silently leave the
    protection it names off, so the load stops instead.

    Args:
        unknown: What ``find_unknown_yaml_keys`` / ``find_unknown_env_names`` found.
        source: Where the keys came from, for the message.

    Raises:
        ConfigError: When any of them belongs to the security section.
    """
    if not unknown or not _REPORT_UNKNOWN.get():
        return
    fatal = [item for item in unknown if item.security]
    if fatal:
        described = "; ".join(item.describe() for item in fatal)
        raise ConfigError(
            f"{source}: {described}. Unknown security settings are refused so that a "
            "typo cannot leave a protection off.",
            [item.as_error(source) for item in fatal],
        )
    for item in unknown:
        if (source, item.name) in _UNKNOWN_WARNED:
            continue
        _UNKNOWN_WARNED.add((source, item.name))
        _LOGGER.warning("Ignoring %s from %s", item.describe(), source)


def _dotenv_names(env_file: Path) -> List[str]:
    """The variable names an env file assigns."""
    from dotenv import dotenv_values

    return [str(name) for name in dotenv_values(env_file)]


def _normalise_isaac_instances(value: Any) -> Optional[List[Dict[str, Any]]]:
    """Flatten optional per-instance bridge settings from nested YAML layout."""
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("isaac_sim.instances must be a list of mappings")

    instances: List[Dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"isaac_sim.instances[{index}] must be a mapping, not a {type(item).__name__}")
        bridge = item.get("bridge") or {}
        if not isinstance(bridge, dict):
            raise ValueError(f"isaac_sim.instances[{index}].bridge must be a mapping")
        entry = {field: item.get(field) for field in IsaacInstanceConfig.model_fields}
        for key, field in _INSTANCE_BRIDGE_GROUP.items():
            entry[field] = _coalesce(entry.get(field), bridge.get(key))
        instances.append(_compact_dict(entry))
    return instances


def _normalise_section(section: str, data: Any) -> Dict[str, Any]:
    """Map one YAML section onto its model's fields, flat keys before nested ones."""
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"section {section!r} must be a mapping, not a {type(data).__name__}")
    groups = _YAML_GROUPS.get(section, {})
    group_data: Dict[str, Dict[str, Any]] = {}
    for group in groups:
        value = data.get(group)
        if value is not None and not isinstance(value, dict):
            raise ValueError(f"{section}.{group} must be a mapping, not a {type(value).__name__}")
        group_data[group] = value or {}
    values: Dict[str, Any] = {}
    for field in _section_fields(section):
        nested = [
            group_data[group].get(key)
            for group, mapping in groups.items()
            for key, target in mapping.items()
            if target == field
        ]
        values[field] = _coalesce(data.get(field), *nested)
    return _compact_dict(values)


def _normalise_settings_payload(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten the repo's nested YAML layout into the Settings model shape.

    Raises:
        ValueError: When a section, a group or ``isaac_sim.instances`` has the
            wrong shape.
    """
    env = dict(os.environ)
    if _CHECKOUT_ROOT is not None:
        env.setdefault("PROJECT_ROOT", str(_CHECKOUT_ROOT))
    # ``${SIMUL_HOME}`` in a YAML value is the state directory, set or not.
    env.setdefault(STATE_DIR_VARIABLE, str(state_dir()))
    raw = _expand_placeholders(config_data, env)

    payload: Dict[str, Any] = {}
    for section in _section_names():
        values = _normalise_section(section, raw.get(section))
        if section == "isaac_sim":
            values["path"] = _normalise_optional_path(values.get("path"))
            values["instances"] = _normalise_isaac_instances(values.get("instances"))
        elif section == "server":
            values["auth_token_file"] = _normalise_optional_path(values.get("auth_token_file"))
        payload[section] = _compact_dict(values)
    return payload


_SECRET_FIELDS = ("token", "passphrase", "password", "secret")


def _env_lookup(name: str) -> Optional[str]:
    """The process environment's value for ``name``, matched case-insensitively."""
    if name in os.environ:
        return os.environ[name]
    lowered = name.lower()
    for key, value in os.environ.items():
        if key.lower() == lowered:
            return value
    return None


def _setting_source(loc: Tuple[Any, ...], yaml_file: Optional[str]) -> Optional[str]:
    """Best guess at where the value of setting ``loc`` came from."""
    if not loc:
        return None
    section = str(loc[0])
    if len(loc) >= 2:
        variable = f"{section}__{loc[1]}".upper()
        if _env_lookup(variable) is not None:
            return variable
        env_file = os.environ.get(ENV_FILE_VARIABLE, "").strip()
        if env_file:
            try:
                names = {name.upper() for name in _dotenv_names(Path(env_file).expanduser())}
            except Exception:  # noqa: BLE001 - only a hint for the message
                names = set()
            if variable in names:
                return f"{variable} in {env_file}"
    if _env_lookup(section.upper()) is not None:
        return section.upper()
    return yaml_file


def _shown_value(loc: Tuple[Any, ...], value: Any) -> Any:
    """The rejected value as an error may show it: secrets masked, long values cut."""
    if any(marker in str(part).lower() for part in loc for marker in _SECRET_FIELDS):
        return "***"
    if isinstance(value, (str, int, float, bool)) or value is None:
        text = value if not isinstance(value, str) or len(value) <= 200 else value[:200] + "..."
        return text
    rendered = repr(value)
    return rendered if len(rendered) <= 200 else rendered[:200] + "..."


def config_error_from(exc: Exception, yaml_file: Optional[Union[str, Path]] = None) -> ConfigError:
    """Turn a settings-loading failure into a ``ConfigError`` naming each bad setting.

    Args:
        exc: A pydantic ``ValidationError``, a pydantic-settings
            ``SettingsError`` (a ``SECTION__FIELD`` value that is not valid
            JSON for a list or mapping field), or a ``ConfigError``.
        yaml_file: The config file in use, named when a bad value is not
            from the environment.

    Returns:
        The error; a ``ConfigError`` comes back unchanged.
    """
    if isinstance(exc, ConfigError):
        return exc
    file_label = str(yaml_file) if yaml_file is not None else None
    if isinstance(exc, ValidationError):
        errors: List[Dict[str, Any]] = []
        for problem in exc.errors():
            loc = tuple(problem.get("loc", ()))
            setting = ".".join(str(part) for part in loc)
            entry: Dict[str, Any] = {"setting": setting, "reason": problem.get("msg", "invalid")}
            source = _setting_source(loc, file_label)
            if source:
                entry["source"] = source
            if "input" in problem:
                entry["value"] = _shown_value(loc, problem["input"])
            errors.append(entry)
        described = "; ".join(
            f"{entry['setting'] or 'settings'}"
            + (f" = {entry['value']!r}" if "value" in entry else "")
            + (f" (from {entry['source']})" if "source" in entry else "")
            + f": {entry['reason']}"
            for entry in errors
        )
        return ConfigError(f"Invalid setting: {described}", errors)
    cause = exc.__cause__ or exc.__context__
    reason = f"{exc}: {cause}" if cause is not None else str(exc)
    match = re.search(r'field "([^"]+)"', str(exc))
    section = match.group(1) if match else None
    culprits = _unparsable_env_vars(section) if section else []
    if culprits:
        errors = [
            {
                "setting": f"{section}.{name.split('__', 1)[1].lower()}",
                "source": name,
                "value": _shown_value((section, name), value),
                "reason": (
                    "not valid JSON; list and mapping settings take JSON in the environment, "
                    f"for example {name.upper()}='[\"a\", \"b\"]'"
                ),
            }
            for name, value in culprits
        ]
        described = "; ".join(f"{entry['source']} = {entry['value']!r}: {entry['reason']}" for entry in errors)
        return ConfigError(f"Invalid setting: {described}", errors)
    entry: Dict[str, Any] = {"reason": reason}
    if section:
        entry["setting"] = section
    return ConfigError(f"Invalid setting: {reason}", [entry])


def _unparsable_env_vars(section: str) -> List[Tuple[str, str]]:
    """The ``SECTION__FIELD`` variables of list or mapping fields whose value is not JSON."""
    if section not in _section_names():
        return []
    model = cast(type[BaseModel], Settings.model_fields[section].annotation)
    culprits: List[Tuple[str, str]] = []
    for name, value in os.environ.items():
        head, _, rest = name.partition("__")
        info = model.model_fields.get(rest.lower()) if head.lower() == section else None
        if info is None or get_origin(info.annotation) not in (list, dict):
            continue
        try:
            json.loads(value)
        except ValueError:
            culprits.append((name, value))
    return culprits


def _build(factory: Callable[[], Settings], yaml_file: Optional[Union[str, Path]]) -> Settings:
    """Run a Settings constructor, reporting every failure as a ``ConfigError``."""
    try:
        return factory()
    except (ValidationError, SettingsError) as exc:
        raise config_error_from(exc, yaml_file) from exc


@lru_cache(maxsize=1)
def _get_cached_settings(cache_key: Tuple[str, Optional[float]]) -> Settings:
    """Load settings with a cache key that changes with path or mtime."""
    config_path, _mtime = cache_key
    return _build(lambda: Settings.from_yaml(config_path), config_path)


def get_settings() -> Settings:
    """Get settings keyed by the resolved config path and file mtime.

    The file is ``$CONFIG_FILE`` when set, otherwise the ``default.yaml``
    shipped inside the package.

    Raises:
        ConfigError: When ``$CONFIG_FILE`` names no file, the file or the
            environment holds an invalid value, or anything else stops the
            settings from loading (see ``ConfigError``).
    """
    named = os.getenv("CONFIG_FILE")
    resolved = _resolve_config_file(named or _DEFAULT_CONFIG_FILE)
    try:
        mtime: Optional[float] = resolved.stat().st_mtime
    except OSError:
        if named:
            raise ConfigError(
                f"CONFIG_FILE names no file: {resolved}",
                [{"setting": "CONFIG_FILE", "source": "CONFIG_FILE", "value": named, "reason": "no such file"}],
            ) from None
        mtime = None
        _LOGGER.warning("Packaged config file %s is missing; using defaults and the environment", resolved)
    return _get_cached_settings((str(resolved), mtime))


def load_config_from_file(config_path: Union[str, Path]) -> Settings:
    """Load configuration from a specific YAML file.

    Raises:
        ConfigError: When the file does not exist or anything stops it loading.
    """
    resolved = _resolve_config_file(config_path)
    if not resolved.is_file():
        raise ConfigError(
            f"Configuration file not found: {resolved}",
            [{"source": str(resolved), "reason": "no such file"}],
        )
    return _build(lambda: Settings.from_yaml(resolved), resolved)


def load_settings(config_path: Union[str, Path]) -> Settings:
    """Backward-compatible wrapper for config loading."""
    return load_config_from_file(config_path)


@dataclass
class ConfigReport:
    """What ``simul validate-config`` found in one config file.

    Attributes:
        path: The resolved file.
        settings: The settings the file loads to, or None when it does not load.
        errors: Problems that make the file invalid.
        warnings: Problems worth fixing that do not.
    """

    path: Path
    settings: Optional[Settings] = None
    errors: List[str] = dataclass_field(default_factory=list)
    warnings: List[str] = dataclass_field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors


def inspect_config_file(config_path: Union[str, Path]) -> ConfigReport:
    """Check a config file strictly, the way ``simul validate-config`` reports it.

    Stricter than loading it: every unknown key is an error (loading only
    refuses unknown ``security`` keys and warns about the rest), with the
    setting it most likely meant; retired keys are warnings. Unknown
    ``SECTION__...`` environment variables are warnings, since they belong to
    the environment rather than the file, except under ``security``.

    Args:
        config_path: The file to check.

    Returns:
        The report; ``report.valid`` is False when anything is an error.
    """
    resolved = _resolve_config_file(config_path)
    report = ConfigReport(path=resolved)
    if not resolved.is_file():
        report.errors.append(f"Configuration file not found: {resolved}")
        return report
    try:
        raw = read_config_yaml(resolved)
    except ConfigError as exc:
        report.errors.append(exc.message)
        return report
    for item in find_unknown_yaml_keys(raw, include_retired=True):
        (report.warnings if item.retired else report.errors).append(f"{resolved}: {item.describe()}")
    for item in find_unknown_env_names(os.environ, include_retired=True):
        target = report.errors if item.security else report.warnings
        target.append(f"environment: {item.describe()}")
    token = _REPORT_UNKNOWN.set(False)
    try:
        report.settings = _build(lambda: Settings.from_yaml(resolved), resolved)
    except ConfigError as exc:
        report.errors += [_describe_error(entry) for entry in exc.errors]
    finally:
        _REPORT_UNKNOWN.reset(token)
    if report.settings is not None:
        report.errors += validate_settings(report.settings)
    return report


def _describe_error(entry: Dict[str, Any]) -> str:
    """One line for a ``ConfigError.errors`` entry."""
    parts = [entry.get("setting") or "settings"]
    if "value" in entry:
        parts.append(f"= {entry['value']!r}")
    if "source" in entry:
        parts.append(f"(from {entry['source']})")
    return " ".join(parts) + f": {entry.get('reason', 'invalid')}"


def validate_settings(settings: Settings) -> List[str]:
    """Validate settings and return list of validation errors."""
    errors: List[str] = []
    relative_root = _CHECKOUT_ROOT or Path.cwd()

    # Validate the log file directory without creating it: validating a
    # config is not a reason to write anything.
    if settings.logging.file_enabled:
        log_path = Path(settings.logging.file_path).expanduser()
        if not log_path.is_absolute():
            log_path = relative_root / log_path
        log_dir = log_path.parent
        existing = log_dir
        while not existing.exists() and existing.parent != existing:
            existing = existing.parent
        if not existing.is_dir() or not os.access(existing, os.W_OK | os.X_OK):
            errors.append(f"Cannot write the log directory {log_dir}: {existing} is not a writable directory")

    # Validate allowed paths exist. Relative entries only name something inside a
    # source checkout; the sandbox policy drops them everywhere else.
    for path_str in settings.security.allowed_paths:
        path = Path(path_str)
        if not path.is_absolute():
            if _CHECKOUT_ROOT is None:
                continue
            path = _CHECKOUT_ROOT / path
        if not path.exists() and not path_str.startswith("/tmp"):
            errors.append(f"Allowed path does not exist: {path_str}")

    return errors
