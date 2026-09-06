"""
Configuration management for Simul MCP Server.

This module provides Pydantic-based configuration management with support for
environment variables, YAML files, and validation.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast
from functools import lru_cache

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from .resources import find_checkout_root, resource_filesystem_path


_CHECKOUT_ROOT: Optional[Path] = find_checkout_root()
_DEFAULT_CONFIG_FILE: Path = resource_filesystem_path("config", "default.yaml")
_ENV_PLACEHOLDER_PATTERN = re.compile(r"\$\{([^}]+)\}")
_LOGGER = logging.getLogger(__name__)


class ServerConfig(BaseModel):
    """MCP Server configuration."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(
        default="Simul – 3D Simulation & DCC Tools",
        description="Server display name for MCP clients",
    )
    host: str = Field(default="localhost", description="Server host")
    port: int = Field(default=8765, description="Server port", ge=1024, le=65535)
    max_connections: int = Field(
        default=10, description="Maximum concurrent connections", ge=1
    )
    timeout: int = Field(default=30, description="Connection timeout in seconds", ge=1)
    enable_cors: bool = Field(default=True, description="Enable CORS")
    cors_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost:*", "https://localhost:*"]
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

    path: Optional[str] = Field(
        # Read from the environment so the field is the one source of truth.
        # It existed and was never populated, while callers reached past it to
        # os.environ directly.
        default_factory=lambda: os.environ.get("ISAAC_SIM_PATH") or None,
        description="Path to Isaac Sim installation (defaults to $ISAAC_SIM_PATH)",
    )
    headless: bool = Field(default=False, description="Run in headless mode")
    enable_livestream: bool = Field(default=False, description="Enable livestream")
    livestream_port: int = Field(
        default=8211, description="Livestream port", ge=1024, le=65535
    )
    enable_webrtc: bool = Field(default=False, description="Enable WebRTC")
    width: int = Field(default=1920, description="Viewport width", ge=640)
    height: int = Field(default=1080, description="Viewport height", ge=480)

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
            "Precedence: an explicit value (ISAAC_SIM__SOCKET_AUTH_TOKEN, .env or YAML) "
            "wins; when unset, the token file `simul-mcp isaac launch "
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
        default="/tmp/simul-mcp",
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

    @model_validator(mode="after")
    def _pick_up_generated_auth_token(self) -> "IsaacSimConfig":
        """Fall back to the token file ``simul-mcp isaac launch`` left in the discovery dir.

        Only when nothing explicit was configured: the environment, ``.env`` and
        YAML all rank above the file. The file is read only from a discovery
        directory that is owned by the current user and writable by nobody else.
        """
        if self.socket_auth_token is not None:
            return self
        # Imported here: simul_mcp.utils pulls in simul_mcp.logging, which imports this module.
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
        model. Callers that actually need the install (``simul-mcp isaac
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
        if self.scan_port_start >= self.scan_port_end:
            raise ValueError(
                f"scan_port_start ({self.scan_port_start}) must be less than "
                f"scan_port_end ({self.scan_port_end})"
            )
        return self


class BlenderConfig(BaseModel):
    """Blender runtime configuration."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(default=True, description="Enable Blender runtime tools")
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
    embedded_mode: bool = Field(
        default=False,
        description="Running inside UE5 Python interpreter",
    )
    max_actors: int = Field(
        default=200,
        description="Default maximum actors returned from Unreal tools",
        ge=1,
        le=5000,
    )
    tool_surface: Literal["thin", "full"] = Field(
        default="thin",
        description=(
            "Which Unreal tools the MCP server registers. 'thin' exposes only "
            "unreal_health_check, ping_unreal, list_unreal_instances, "
            "capture_unreal_viewport and execute_unreal_script; 'full' exposes "
            "every granular Unreal tool. Env: UNREAL__TOOL_SURFACE."
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

    # Auth (matches simul-mcp unreal setup --passphrase). When set, the
    # Remote Control client sends a `Passphrase: <md5>` HTTP header on every
    # request. Accepts either the raw plaintext (we MD5-hash it on send) or
    # a pre-computed 32-character hex digest (we pass it through). Required
    # whenever the editor was started with --passphrase enabled —
    # without it the editor returns 401 "Given Passphrase is not correct!"
    # Read from env var UNREAL__PASSPHRASE (or .env file).
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
        if self.scan_port_start >= self.scan_port_end:
            raise ValueError(
                f"scan_port_start ({self.scan_port_start}) must be less than "
                f"scan_port_end ({self.scan_port_end})"
            )
        return self


class USDConfig(BaseModel):
    """USD configuration."""

    model_config = ConfigDict(frozen=True)

    cache_enabled: bool = Field(default=True, description="Enable USD caching")
    cache_size: int = Field(default=1000, description="USD cache size", ge=1)
    stage_cache_limit: int = Field(default=10, description="Stage cache limit", ge=1)
    load_rules: str = Field(default="LoadAll", description="USD load rules")
    population_mask: str = Field(default="", description="USD population mask")
    interpolation_type: str = Field(
        default="Linear", description="USD interpolation type"
    )
    enable_instancing: bool = Field(default=True, description="Enable USD instancing")
    enable_multithreading: bool = Field(
        default=True, description="Enable USD multithreading"
    )
    max_concurrent_operations: int = Field(
        default=10, description="Max concurrent USD operations", ge=1
    )
    operation_timeout: int = Field(
        default=30, description="USD operation timeout", ge=1
    )
    allowed_extensions: List[str] = Field(
        default_factory=lambda: [".usd", ".usda", ".usdc", ".usdz"],
        description="Allowed USD file extensions",
    )
    max_file_size_mb: int = Field(
        default=500, description="Maximum USD file size in MB", ge=1
    )
class ViewportConfig(BaseModel):
    """Viewport configuration."""

    model_config = ConfigDict(frozen=True)

    default_width: int = Field(
        default=1920, description="Default viewport width", ge=640
    )
    default_height: int = Field(
        default=1080, description="Default viewport height", ge=480
    )
    max_size: int = Field(default=2048, description="Maximum viewport size", ge=640)
    format: str = Field(default="png", description="Image format")
    quality: int = Field(default=95, description="Image quality", ge=1, le=100)
    samples_per_pixel: int = Field(default=1, description="Samples per pixel", ge=1)
    max_bounces: int = Field(default=4, description="Maximum ray bounces", ge=1)
    enable_denoising: bool = Field(default=True, description="Enable denoising")
    fov: float = Field(default=45.0, description="Field of view", ge=1.0, le=179.0)
    near_plane: float = Field(default=0.1, description="Near clipping plane", gt=0.0)
    far_plane: float = Field(default=1000.0, description="Far clipping plane", gt=0.0)
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
    file_path: str = Field(default="~/.simul/logs/simul_mcp.log", description="Log file path")
    file_max_size: str = Field(default="10MB", description="Maximum log file size")
    file_backup_count: int = Field(default=5, description="Log file backup count", ge=1)
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
        default="~/.simul/logs/audit.jsonl",
        description="Destination of the per-tool-call audit JSONL stream.",
    )
    structured_enabled: bool = Field(
        default=False,
        description=(
            "Also write the structured JSON file (simul_mcp.json) for every "
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
        default_factory=lambda: ["examples", "tests/data", "/tmp/simul_mcp"],
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
    rate_limiting_enabled: bool = Field(
        default=True, description="Enable rate limiting"
    )
    requests_per_minute: int = Field(
        default=60, description="Requests per minute limit", ge=1
    )
    burst_size: int = Field(
        default=10, description="Burst size for rate limiting", ge=1
    )


class NormalisedYamlSettingsSource(YamlConfigSettingsSource):
    """YAML settings source that flattens the repo's nested layout first.

    The shipped ``default.yaml`` groups keys the way operators read them
    (``isaac_sim.bridge.port``) while the Settings model is flat within each
    section (``isaac_sim.bridge_port``). ``_normalise_settings_payload``
    bridges the two before pydantic-settings merges this source below the
    environment.
    """

    def _read_file(self, file_path: Path) -> Dict[str, Any]:
        """Load one YAML file and return it in the Settings model shape.

        Args:
            file_path: YAML file to read; pydantic-settings only calls this for
                files that exist.

        Returns:
            The normalised payload, or ``{}`` when the file is unreadable or
            does not hold a mapping. Unreadable files are logged as warnings.
        """
        try:
            with file_path.open(encoding=self.yaml_file_encoding or "utf-8") as handle:
                raw = yaml.safe_load(handle)
        except (OSError, yaml.YAMLError) as exc:
            _LOGGER.warning("Failed to load config file %s: %s", file_path, exc)
            return {}
        if not isinstance(raw, dict):
            return {}
        return _normalise_settings_payload(raw)


class Settings(BaseSettings):
    """Main settings class that combines all configuration sections.

    Value precedence per leaf, highest first: constructor arguments, process
    environment, ``.env`` file, the YAML file named by ``model_config["yaml_file"]``
    (set through :meth:`from_yaml`), model defaults.
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
        env_file=".env",
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
        """Rank the YAML file below the environment and ``.env``.

        Args:
            settings_cls: The concrete settings class being built.
            init_settings: Constructor-argument source.
            env_settings: Process-environment source.
            dotenv_settings: ``.env`` file source.
            file_secret_settings: Secrets-directory source.

        Returns:
            The sources in priority order, highest first.
        """
        return (
            init_settings,
            env_settings,
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
                "%s. Isaac Sim tools and `simul-mcp isaac launch` report this when they "
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


def _normalise_isaac_instances(value: Any) -> Optional[List[Dict[str, Any]]]:
    """Flatten optional per-instance bridge settings from nested YAML layout."""
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("instances must be a list of dicts")

    instances: List[Dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"Each instance must be a dict, got {type(item).__name__}")


        bridge = item.get("bridge") or {}
        instances.append(
            _compact_dict(
                {
                    "name": item.get("name"),
                    "host": item.get("host"),
                    "port": item.get("port"),
                    "timeout": item.get("timeout"),
                    "bridge_enabled": _coalesce(
                        item.get("bridge_enabled"),
                        bridge.get("enabled"),
                    ),
                    "bridge_host": _coalesce(
                        item.get("bridge_host"),
                        bridge.get("host"),
                    ),
                    "bridge_port": _coalesce(
                        item.get("bridge_port"),
                        bridge.get("port"),
                    ),
                    "bridge_timeout": _coalesce(
                        item.get("bridge_timeout"),
                        bridge.get("timeout"),
                    ),
                    "bridge_fallback_to_vscode": _coalesce(
                        item.get("bridge_fallback_to_vscode"),
                        bridge.get("fallback_to_vscode"),
                    ),
                    "socket_protocol": item.get("socket_protocol"),
                    "socket_auth_token": item.get("socket_auth_token"),
                }
            )
        )
    return instances


def _normalise_settings_payload(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten the repo's nested YAML layout into the Settings model shape."""
    env = dict(os.environ)
    if _CHECKOUT_ROOT is not None:
        env.setdefault("PROJECT_ROOT", str(_CHECKOUT_ROOT))
    raw = _expand_placeholders(config_data, env)

    server = raw.get("server") or {}
    isaac = raw.get("isaac_sim") or {}
    isaac_kit = isaac.get("kit") or {}
    isaac_bridge = isaac.get("bridge") or {}
    isaac_resolution = isaac_kit.get("resolution") or {}
    usd = raw.get("usd") or {}
    usd_stage = usd.get("stage") or {}
    usd_cache = usd.get("cache") or {}
    usd_files = usd.get("files") or {}
    usd_performance = usd.get("performance") or {}
    viewport = raw.get("viewport") or {}
    viewport_capture = viewport.get("capture") or {}
    viewport_rendering = viewport.get("rendering") or {}
    viewport_camera = viewport.get("camera") or {}
    logging_cfg = raw.get("logging") or {}
    logging_file = logging_cfg.get("file") or {}
    logging_console = logging_cfg.get("console") or {}
    logging_audit = logging_cfg.get("audit") or {}
    logging_structured = logging_cfg.get("structured") or {}
    security = raw.get("security") or {}
    security_sandbox = security.get("sandbox") or {}
    security_rate = security.get("rate_limiting") or {}

    return _compact_dict(
        {
            "server": _compact_dict(
                {
                    "name": server.get("name"),
                    "host": server.get("host"),
                    "port": server.get("port"),
                    "max_connections": server.get("max_connections"),
                    "timeout": server.get("timeout"),
                    "enable_cors": server.get("enable_cors"),
                    "cors_origins": server.get("cors_origins"),
                }
            ),
            "isaac_sim": _compact_dict(
                {
                    "path": _normalise_optional_path(isaac.get("path")),
                    "headless": _coalesce(isaac.get("headless"), isaac_kit.get("headless")),
                    "enable_livestream": _coalesce(
                        isaac.get("enable_livestream"),
                        isaac_kit.get("enable_livestream"),
                    ),
                    "livestream_port": _coalesce(
                        isaac.get("livestream_port"),
                        isaac_kit.get("livestream_port"),
                    ),
                    "enable_webrtc": _coalesce(
                        isaac.get("enable_webrtc"),
                        isaac_kit.get("enable_webrtc"),
                    ),
                    "width": _coalesce(isaac.get("width"), isaac_resolution.get("width")),
                    "height": _coalesce(isaac.get("height"), isaac_resolution.get("height")),
                    "socket_host": isaac.get("socket_host"),
                    "socket_port": isaac.get("socket_port"),
                    "socket_timeout": isaac.get("socket_timeout"),
                    "socket_protocol": isaac.get("socket_protocol"),
                    "socket_auth_token": isaac.get("socket_auth_token"),
                    "bridge_enabled": _coalesce(
                        isaac.get("bridge_enabled"), isaac_bridge.get("enabled")
                    ),
                    "bridge_host": _coalesce(
                        isaac.get("bridge_host"), isaac_bridge.get("host")
                    ),
                    "bridge_port": _coalesce(
                        isaac.get("bridge_port"), isaac_bridge.get("port")
                    ),
                    "bridge_timeout": _coalesce(
                        isaac.get("bridge_timeout"), isaac_bridge.get("timeout")
                    ),
                    "bridge_fallback_to_vscode": _coalesce(
                        isaac.get("bridge_fallback_to_vscode"),
                        isaac_bridge.get("fallback_to_vscode"),
                    ),
                    "instances": _normalise_isaac_instances(isaac.get("instances")),
                    "scan_port_start": isaac.get("scan_port_start"),
                    "scan_port_end": isaac.get("scan_port_end"),
                    "discovery_dir": isaac.get("discovery_dir"),
                    "enforce_claims": isaac.get("enforce_claims"),
                }
            ),
            "blender": raw.get("blender"),
            "unreal": raw.get("unreal"),
            "usd": _compact_dict(
                {
                    "cache_enabled": _coalesce(
                        usd.get("cache_enabled"), usd_cache.get("enabled")
                    ),
                    "cache_size": _coalesce(usd.get("cache_size"), usd_cache.get("size")),
                    "stage_cache_limit": _coalesce(
                        usd.get("stage_cache_limit"), usd_cache.get("stage_cache_limit")
                    ),
                    "load_rules": _coalesce(usd.get("load_rules"), usd_stage.get("load_rules")),
                    "population_mask": _coalesce(
                        usd.get("population_mask"), usd_stage.get("population_mask")
                    ),
                    "interpolation_type": _coalesce(
                        usd.get("interpolation_type"), usd_stage.get("interpolation_type")
                    ),
                    "enable_instancing": _coalesce(
                        usd.get("enable_instancing"), usd_stage.get("enable_instancing")
                    ),
                    "enable_multithreading": _coalesce(
                        usd.get("enable_multithreading"),
                        usd_performance.get("enable_multithreading"),
                    ),
                    "max_concurrent_operations": _coalesce(
                        usd.get("max_concurrent_operations"),
                        usd_performance.get("max_concurrent_operations"),
                    ),
                    "operation_timeout": _coalesce(
                        usd.get("operation_timeout"), usd_performance.get("operation_timeout")
                    ),
                    "allowed_extensions": _coalesce(
                        usd.get("allowed_extensions"), usd_files.get("allowed_extensions")
                    ),
                    "max_file_size_mb": _coalesce(
                        usd.get("max_file_size_mb"), usd_files.get("max_file_size_mb")
                    ),
                }
            ),
            "viewport": _compact_dict(
                {
                    "default_width": _coalesce(
                        viewport.get("default_width"), viewport_capture.get("width")
                    ),
                    "default_height": _coalesce(
                        viewport.get("default_height"), viewport_capture.get("height")
                    ),
                    "max_size": _coalesce(viewport.get("max_size"), viewport_capture.get("max_size")),
                    "format": _coalesce(viewport.get("format"), viewport_capture.get("format")),
                    "quality": _coalesce(viewport.get("quality"), viewport_capture.get("quality")),
                    "capture_dir": _coalesce(
                        viewport.get("capture_dir"), viewport_capture.get("directory")
                    ),
                    "samples_per_pixel": _coalesce(
                        viewport.get("samples_per_pixel"),
                        viewport_rendering.get("samples_per_pixel"),
                    ),
                    "max_bounces": _coalesce(
                        viewport.get("max_bounces"), viewport_rendering.get("max_bounces")
                    ),
                    "enable_denoising": _coalesce(
                        viewport.get("enable_denoising"),
                        viewport_rendering.get("enable_denoising"),
                    ),
                    "fov": _coalesce(viewport.get("fov"), viewport_camera.get("fov")),
                    "near_plane": _coalesce(
                        viewport.get("near_plane"), viewport_camera.get("near_plane")
                    ),
                    "far_plane": _coalesce(
                        viewport.get("far_plane"), viewport_camera.get("far_plane")
                    ),
                }
            ),
            "logging": _compact_dict(
                {
                    "level": logging_cfg.get("level"),
                    "format": logging_cfg.get("format"),
                    "file_enabled": _coalesce(
                        logging_cfg.get("file_enabled"), logging_file.get("enabled")
                    ),
                    "file_path": _coalesce(logging_cfg.get("file_path"), logging_file.get("path")),
                    "file_max_size": _coalesce(
                        logging_cfg.get("file_max_size"), logging_file.get("max_size")
                    ),
                    "file_backup_count": _coalesce(
                        logging_cfg.get("file_backup_count"),
                        logging_file.get("backup_count"),
                    ),
                    "console_enabled": _coalesce(
                        logging_cfg.get("console_enabled"), logging_console.get("enabled")
                    ),
                    "console_colored": _coalesce(
                        logging_cfg.get("console_colored"), logging_console.get("colored")
                    ),
                    "components": logging_cfg.get("components"),
                    "per_instance": _coalesce(
                        logging_cfg.get("per_instance"), logging_file.get("per_instance")
                    ),
                    "retention_days": _coalesce(
                        logging_cfg.get("retention_days"), logging_file.get("retention_days")
                    ),
                    "audit_enabled": _coalesce(
                        logging_cfg.get("audit_enabled"), logging_audit.get("enabled")
                    ),
                    "audit_path": _coalesce(logging_cfg.get("audit_path"), logging_audit.get("path")),
                    "structured_enabled": _coalesce(
                        logging_cfg.get("structured_enabled"), logging_structured.get("enabled")
                    ),
                }
            ),
            "security": _compact_dict(
                {
                    "sandbox_enabled": _coalesce(
                        security.get("sandbox_enabled"), security_sandbox.get("enabled")
                    ),
                    "allowed_paths": _coalesce(
                        security.get("allowed_paths"), security_sandbox.get("allowed_paths")
                    ),
                    "allowed_url_schemes": _coalesce(
                        security.get("allowed_url_schemes"),
                        security_sandbox.get("allowed_url_schemes"),
                    ),
                    "allowed_write_url_schemes": _coalesce(
                        security.get("allowed_write_url_schemes"),
                        security_sandbox.get("allowed_write_url_schemes"),
                    ),
                    "rate_limiting_enabled": _coalesce(
                        security.get("rate_limiting_enabled"),
                        security_rate.get("enabled"),
                    ),
                    "requests_per_minute": _coalesce(
                        security.get("requests_per_minute"),
                        security_rate.get("requests_per_minute"),
                    ),
                    "burst_size": _coalesce(
                        security.get("burst_size"), security_rate.get("burst_size")
                    ),
                }
            ),
        }
    )


@lru_cache(maxsize=1)
def _get_cached_settings(cache_key: Tuple[str, Optional[float]]) -> Settings:
    """Load settings with a cache key that changes with path or mtime."""
    config_path, _mtime = cache_key
    return Settings.from_yaml(config_path)


def get_settings() -> Settings:
    """Get settings keyed by the resolved config path and file mtime.

    The file is ``$CONFIG_FILE`` when set, otherwise the ``default.yaml``
    shipped inside the package.
    """
    config_file: Union[str, Path] = os.getenv("CONFIG_FILE") or _DEFAULT_CONFIG_FILE
    resolved = _resolve_config_file(config_file)
    try:
        mtime: Optional[float] = resolved.stat().st_mtime
    except OSError:
        mtime = None
        _LOGGER.warning("Config file %s not found; using defaults and the environment", resolved)
    return _get_cached_settings((str(resolved), mtime))


def load_config_from_file(config_path: Union[str, Path]) -> Settings:
    """Load configuration from a specific YAML file."""
    resolved = _resolve_config_file(config_path)

    if not resolved.exists():
        raise FileNotFoundError(f"Configuration file not found: {resolved}")

    return Settings.from_yaml(resolved)


def load_settings(config_path: Union[str, Path]) -> Settings:
    """Backward-compatible wrapper for config loading."""
    return load_config_from_file(config_path)


def validate_settings(settings: Settings) -> List[str]:
    """Validate settings and return list of validation errors."""
    errors: List[str] = []
    relative_root = _CHECKOUT_ROOT or Path.cwd()

    # Validate log file directory
    if settings.logging.file_enabled:
        log_path = Path(settings.logging.file_path).expanduser()
        if not log_path.is_absolute():
            log_path = relative_root / log_path
        log_dir = log_path.parent
        if not log_dir.exists():
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create log directory {log_dir}: {e}")

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
