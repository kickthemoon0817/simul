"""
Configuration management for Simul MCP Server.

This module provides Pydantic-based configuration management with support for
environment variables, YAML files, and validation.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from functools import lru_cache

import yaml
from pydantic import BaseSettings, Field, validator
from pydantic.env_settings import SettingsSourceCallable


class ServerConfig(BaseSettings):
    """MCP Server configuration."""

    name: str = Field(default="SimulMCP", description="Server name")
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


class IsaacSimConfig(BaseSettings):
    """Isaac Sim configuration."""

    path: Optional[str] = Field(
        default=None, description="Path to Isaac Sim installation"
    )
    headless: bool = Field(default=False, description="Run in headless mode")
    enable_livestream: bool = Field(default=False, description="Enable livestream")
    livestream_port: int = Field(
        default=8211, description="Livestream port", ge=1024, le=65535
    )
    enable_webrtc: bool = Field(default=False, description="Enable WebRTC")
    width: int = Field(default=1920, description="Viewport width", ge=640)
    height: int = Field(default=1080, description="Viewport height", ge=480)

    @validator("path")
    def validate_isaac_path(cls, v):
        """Validate Isaac Sim path if provided."""
        if v is not None:
            path = Path(v)
            if not path.exists():
                raise ValueError(f"Isaac Sim path does not exist: {v}")
            if not (path / "python.sh").exists() and not (path / "python.bat").exists():
                raise ValueError(f"Isaac Sim python executable not found in: {v}")
        return v


class USDConfig(BaseSettings):
    """USD configuration."""

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


class MeshConfig(BaseSettings):
    """Mesh processing configuration."""

    decimation_enabled: bool = Field(default=True, description="Enable mesh decimation")
    max_faces: int = Field(
        default=50000, description="Maximum faces for decimation", ge=1
    )
    preserve_boundaries: bool = Field(
        default=True, description="Preserve mesh boundaries"
    )
    preserve_topology: bool = Field(default=False, description="Preserve mesh topology")
    compute_normals: bool = Field(default=True, description="Compute mesh normals")
    compute_tangents: bool = Field(default=False, description="Compute mesh tangents")
    validate_topology: bool = Field(default=True, description="Validate mesh topology")
    include_materials: bool = Field(
        default=True, description="Include materials in export"
    )
    include_textures: bool = Field(
        default=True, description="Include textures in export"
    )
    texture_resolution: int = Field(
        default=1024, description="Texture resolution", ge=64
    )


class ViewportConfig(BaseSettings):
    """Viewport configuration."""

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


class LoggingConfig(BaseSettings):
    """Logging configuration."""

    level: str = Field(default="INFO", description="Log level")
    format: str = Field(default="detailed", description="Log format")
    file_enabled: bool = Field(default=True, description="Enable file logging")
    file_path: str = Field(default="logs/simul_mcp.log", description="Log file path")
    file_max_size: str = Field(default="10MB", description="Maximum log file size")
    file_backup_count: int = Field(default=5, description="Log file backup count", ge=1)
    console_enabled: bool = Field(default=True, description="Enable console logging")
    console_colored: bool = Field(
        default=True, description="Enable colored console output"
    )

    @validator("level")
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()


class SecurityConfig(BaseSettings):
    """Security configuration."""

    sandbox_enabled: bool = Field(default=True, description="Enable sandbox mode")
    allowed_paths: List[str] = Field(
        default_factory=lambda: ["examples", "tests/data", "/tmp/simul_mcp"],
        description="Allowed file paths",
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


class PerformanceConfig(BaseSettings):
    """Performance configuration."""

    memory_limit_gb: int = Field(default=8, description="Memory limit in GB", ge=1)
    gc_threshold: int = Field(
        default=1000, description="Garbage collection threshold", ge=1
    )
    enable_memory_profiling: bool = Field(
        default=False, description="Enable memory profiling"
    )
    max_workers: int = Field(default=4, description="Maximum worker threads", ge=1)
    enable_async_operations: bool = Field(
        default=True, description="Enable async operations"
    )
    enable_result_caching: bool = Field(
        default=True, description="Enable result caching"
    )
    cache_ttl_seconds: int = Field(
        default=300, description="Cache TTL in seconds", ge=1
    )
    max_cache_entries: int = Field(
        default=1000, description="Maximum cache entries", ge=1
    )


class FeatureFlags(BaseSettings):
    """Feature flags configuration."""

    enable_mesh_analysis: bool = Field(default=True, description="Enable mesh analysis")
    enable_material_extraction: bool = Field(
        default=True, description="Enable material extraction"
    )
    enable_animation_support: bool = Field(
        default=False, description="Enable animation support"
    )
    enable_physics_queries: bool = Field(
        default=True, description="Enable physics queries"
    )
    enable_lighting_analysis: bool = Field(
        default=True, description="Enable lighting analysis"
    )
    enable_scene_graph_export: bool = Field(
        default=True, description="Enable scene graph export"
    )
    enable_batch_operations: bool = Field(
        default=True, description="Enable batch operations"
    )
    enable_streaming: bool = Field(default=False, description="Enable streaming")


class DevelopmentConfig(BaseSettings):
    """Development configuration."""

    debug_mode: bool = Field(default=False, description="Enable debug mode")
    profiling_enabled: bool = Field(default=False, description="Enable profiling")
    verbose_usd_logging: bool = Field(
        default=False, description="Enable verbose USD logging"
    )
    enable_hot_reload: bool = Field(default=False, description="Enable hot reload")
    enable_mock_isaac: bool = Field(default=False, description="Enable mock Isaac Sim")
    use_test_data: bool = Field(default=False, description="Use test data")
    skip_gpu_operations: bool = Field(default=False, description="Skip GPU operations")


class Settings(BaseSettings):
    """Main settings class that combines all configuration sections."""

    # Configuration sections
    server: ServerConfig = Field(default_factory=ServerConfig)
    isaac_sim: IsaacSimConfig = Field(default_factory=IsaacSimConfig)
    usd: USDConfig = Field(default_factory=USDConfig)
    mesh: MeshConfig = Field(default_factory=MeshConfig)
    viewport: ViewportConfig = Field(default_factory=ViewportConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    features: FeatureFlags = Field(default_factory=FeatureFlags)
    development: DevelopmentConfig = Field(default_factory=DevelopmentConfig)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"
        case_sensitive = False

    @classmethod
    def customise_sources(
        cls,
        init_settings: SettingsSourceCallable,
        env_settings: SettingsSourceCallable,
        file_secret_settings: SettingsSourceCallable,
    ) -> tuple[SettingsSourceCallable, ...]:
        """Customize settings sources to include YAML file loading."""
        return (
            init_settings,
            yaml_config_settings_source,
            env_settings,
            file_secret_settings,
        )


def yaml_config_settings_source(settings: BaseSettings) -> Dict[str, Any]:
    """Load settings from YAML configuration file."""
    config_file = os.getenv("CONFIG_FILE", "config/default.yaml")

    if not os.path.exists(config_file):
        return {}

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        return config_data or {}
    except Exception as e:
        # Log error but don't fail - fall back to other sources
        print(f"Warning: Failed to load config file {config_file}: {e}")
        return {}


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def load_config_from_file(config_path: Union[str, Path]) -> Settings:
    """Load configuration from a specific YAML file."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    original_config = os.getenv("CONFIG_FILE")
    os.environ["CONFIG_FILE"] = str(config_path)

    try:
        get_settings.cache_clear()
        settings = Settings()
        return settings
    finally:
        if original_config is not None:
            os.environ["CONFIG_FILE"] = original_config
        else:
            os.environ.pop("CONFIG_FILE", None)
        get_settings.cache_clear()


def load_settings(config_path: Union[str, Path]) -> Settings:
    """Backward-compatible wrapper for config loading."""
    return load_config_from_file(config_path)


def validate_settings(settings: Settings) -> List[str]:
    """Validate settings and return list of validation errors."""
    errors = []

    # Validate Isaac Sim path if provided
    if settings.isaac_sim.path:
        isaac_path = Path(settings.isaac_sim.path)
        if not isaac_path.exists():
            errors.append(f"Isaac Sim path does not exist: {settings.isaac_sim.path}")
        else:
            python_exe = isaac_path / ("python.bat" if os.name == "nt" else "python.sh")
            if not python_exe.exists():
                errors.append(f"Isaac Sim python executable not found: {python_exe}")

    # Validate log file directory
    log_path = Path(settings.logging.file_path)
    log_dir = log_path.parent
    if not log_dir.exists():
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create log directory {log_dir}: {e}")

    # Validate allowed paths exist
    for path_str in settings.security.allowed_paths:
        path = Path(path_str)
        if not path.exists() and not path_str.startswith("/tmp"):
            errors.append(f"Allowed path does not exist: {path_str}")

    return errors
