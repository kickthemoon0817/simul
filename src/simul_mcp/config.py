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
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerConfig(BaseModel):
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


class IsaacSimConfig(BaseModel):
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

    @field_validator("path")
    @classmethod
    def validate_isaac_path(cls, v: Optional[str]) -> Optional[str]:
        """Validate Isaac Sim path if provided."""
        if v is not None:
            path = Path(v)
            if not path.exists():
                raise ValueError(f"Isaac Sim path does not exist: {v}")
            if not (path / "python.sh").exists() and not (path / "python.bat").exists():
                raise ValueError(f"Isaac Sim python executable not found in: {v}")
        return v


class BlenderConfig(BaseModel):
    """Blender runtime configuration."""

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


class USDConfig(BaseModel):
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


class MeshConfig(BaseModel):
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


class ViewportConfig(BaseModel):
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


class LoggingConfig(BaseModel):
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


class PerformanceConfig(BaseModel):
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


class FeatureFlags(BaseModel):
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


class DevelopmentConfig(BaseModel):
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
    blender: BlenderConfig = Field(default_factory=BlenderConfig)
    unreal: UnrealConfig = Field(default_factory=UnrealConfig)
    usd: USDConfig = Field(default_factory=USDConfig)
    mesh: MeshConfig = Field(default_factory=MeshConfig)
    viewport: ViewportConfig = Field(default_factory=ViewportConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    features: FeatureFlags = Field(default_factory=FeatureFlags)
    development: DevelopmentConfig = Field(default_factory=DevelopmentConfig)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )


def _load_yaml_settings(config_file: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML settings from file path and return dict payload."""
    config_path = Path(config_file)
    if not config_path.exists():
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        if isinstance(config_data, dict):
            return config_data
        return {}
    except Exception as e:
        print(f"Warning: Failed to load config file {config_path}: {e}")
        return {}


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    config_file = os.getenv("CONFIG_FILE", "config/default.yaml")
    config_data = _load_yaml_settings(config_file)
    return Settings(**config_data)


def load_config_from_file(config_path: Union[str, Path]) -> Settings:
    """Load configuration from a specific YAML file."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config_data = _load_yaml_settings(config_path)
    return Settings(**config_data)


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
