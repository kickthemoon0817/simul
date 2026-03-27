"""
Logging configuration and utilities for Isaac Sim MCP Server.

This module provides centralized logging configuration using Python's dictConfig
with support for multiple handlers, formatters, and log levels.
"""

import importlib.util
import logging
import logging.config
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union
import yaml

from .config import Settings, get_settings


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_LOGGING_CONFIG = "config/isaac/logging.yaml"


def setup_logging(
    settings: Optional[Settings] = None,
    config_file: Optional[Union[str, Path]] = None,
    log_level: Optional[str] = None,
    profile: Optional[str] = None,
) -> None:
    """
    Setup logging configuration.

    Args:
        settings: Settings instance. If None, uses get_settings()
        config_file: Path to logging config file. If None, uses config/logging.yaml
        log_level: Override log level
        profile: Logging profile to use (development, production, testing, json_logging)
    """
    if settings is None:
        settings = get_settings()

    # Determine config file path
    if config_file is None:
        config_file = _resolve_logging_config_path(_DEFAULT_LOGGING_CONFIG)
    else:
        config_file = _resolve_logging_config_path(config_file)

    # Load logging configuration
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            _apply_settings_to_dictconfig(config, settings)

            # Apply profile if specified
            if profile and "profiles" in config and profile in config["profiles"]:
                profile_config = config["profiles"][profile]
                # Merge profile configuration
                if "loggers" in profile_config:
                    config.setdefault("loggers", {}).update(profile_config["loggers"])
                if "handlers" in profile_config:
                    config.setdefault("handlers", {}).update(profile_config["handlers"])

            # Override log level if specified
            if log_level:
                # Update all loggers with the new level
                for logger_config in config.get("loggers", {}).values():
                    logger_config["level"] = log_level.upper()
                # Update root logger
                if "root" in config:
                    config["root"]["level"] = log_level.upper()

            # Ensure log directories exist
            _ensure_log_directories(config)

            # Apply the configuration
            logging.config.dictConfig(config)

        except Exception as e:
            # Fallback to basic configuration
            print(f"Warning: Failed to load logging config from {config_file}: {e}")
            _setup_fallback_logging(settings, log_level)
    else:
        # Use fallback configuration
        _setup_fallback_logging(settings, log_level)

    logger = logging.getLogger("simul_mcp.logging")
    logger.info("Logging system initialized")
    if profile:
        logger.info(f"Using logging profile: {profile}")


def _ensure_log_directories(config: Dict[str, Any]) -> None:
    """Ensure all log directories exist."""
    handlers = config.get("handlers", {})

    for handler_config in handlers.values():
        if "filename" in handler_config:
            log_file = Path(handler_config["filename"])
            log_dir = log_file.parent
            if not log_dir.exists():
                try:
                    log_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    print(f"Warning: Could not create log directory {log_dir}: {e}")


def _resolve_logging_config_path(config_file: Union[str, Path]) -> Path:
    """Resolve logging config relative to cwd first, then the repo root."""
    candidate = Path(config_file).expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    return (_PROJECT_ROOT / candidate).resolve()


def _remove_handlers(config: Dict[str, Any], disabled: set[str]) -> None:
    """Strip disabled handlers from root/logger handler lists."""
    handlers = config.get("handlers", {})
    for handler_name in disabled:
        handlers.pop(handler_name, None)

    root = config.get("root")
    if isinstance(root, dict):
        root["handlers"] = [
            name for name in root.get("handlers", []) if name not in disabled
        ]

    for logger_config in config.get("loggers", {}).values():
        if isinstance(logger_config, dict) and "handlers" in logger_config:
            logger_config["handlers"] = [
                name
                for name in logger_config.get("handlers", [])
                if name not in disabled
            ]


def _colorlog_available() -> bool:
    """Return True when colorlog is importable."""
    return importlib.util.find_spec("colorlog") is not None


def _configure_file_handler_paths(
    handlers: Dict[str, Any], base_path: str
) -> None:
    """Update file handler paths to match Settings.file_path."""
    target = Path(base_path)
    suffix = target.suffix or ".log"
    stem = target.stem if target.suffix else target.name
    parent = target.parent

    if "file" in handlers:
        handlers["file"]["filename"] = str(target)
    if "file_json" in handlers:
        handlers["file_json"]["filename"] = str(parent / f"{stem}.json")
    if "error_file" in handlers:
        handlers["error_file"]["filename"] = str(parent / f"{stem}_errors{suffix}")
    if "debug_file" in handlers:
        handlers["debug_file"]["filename"] = str(parent / f"{stem}_debug{suffix}")


def _apply_settings_to_dictconfig(
    config: Dict[str, Any], settings: Settings
) -> None:
    """Apply Settings-level toggles to the YAML dictConfig payload."""
    handlers = config.setdefault("handlers", {})
    disabled_handlers: set[str] = set()

    if not settings.logging.console_enabled:
        disabled_handlers.update({"console", "console_simple"})
    if not settings.logging.file_enabled:
        disabled_handlers.update({"file", "file_json", "error_file", "debug_file"})
    else:
        _configure_file_handler_paths(handlers, settings.logging.file_path)

    if disabled_handlers:
        _remove_handlers(config, disabled_handlers)

    if "colored" in config.get("formatters", {}) and (
        not settings.logging.console_colored or not _colorlog_available()
    ):
        config["formatters"].pop("colored", None)
        for handler_name in ("console", "console_simple"):
            handler = handlers.get(handler_name)
            if handler and handler.get("formatter") == "colored":
                handler["formatter"] = "simple"

    component_levels = settings.logging.components or {}
    for logger_name, level in component_levels.items():
        config.setdefault("loggers", {}).setdefault(logger_name, {})["level"] = level


def _setup_fallback_logging(
    settings: Settings, log_level: Optional[str] = None
) -> None:
    """Setup fallback logging configuration when YAML config fails."""
    level = log_level or settings.logging.level

    handlers: list[logging.Handler] = []
    if settings.logging.console_enabled:
        handlers.append(logging.StreamHandler(sys.stderr))

    if settings.logging.file_enabled:
        log_path = Path(settings.logging.file_path)
        log_dir = log_path.parent
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(settings.logging.file_path, encoding="utf-8"))

    if not handlers:
        handlers.append(logging.NullHandler())

    # Basic logging configuration
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )

    for logger_name, logger_level in (settings.logging.components or {}).items():
        logging.getLogger(logger_name).setLevel(logger_level)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name, typically __name__ of the calling module

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def configure_isaac_logging(verbose: bool = False) -> None:
    """
    Configure logging for Isaac Sim components.

    Args:
        verbose: Enable verbose logging for Isaac Sim components
    """
    isaac_loggers = [
        "omni",
        "carb",
        "pxr",
        "omniverse",
        "isaac",
    ]

    level = logging.DEBUG if verbose else logging.WARNING

    for logger_name in isaac_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)


def set_log_level(logger_name: str, level: Union[str, int]) -> None:
    """
    Set log level for a specific logger.

    Args:
        logger_name: Name of the logger
        level: Log level (string or logging constant)
    """
    logger = logging.getLogger(logger_name)

    if isinstance(level, str):
        level = getattr(logging, level.upper())

    logger.setLevel(level)


def enable_debug_logging() -> None:
    """Enable debug logging for all simul_mcp loggers."""
    worv_loggers = [
        "simul_mcp",
        "simul_mcp.server",
        "simul_mcp.usd",
        "simul_mcp.mesh",
        "simul_mcp.adapters",
        "simul_mcp.mcp",
        "simul_mcp.cli",
    ]

    for logger_name in worv_loggers:
        set_log_level(logger_name, "DEBUG")


def disable_external_logging() -> None:
    """Disable or reduce logging from external libraries."""
    external_loggers = [
        "urllib3",
        "asyncio",
        "websockets",
        "aiohttp",
    ]

    for logger_name in external_loggers:
        set_log_level(logger_name, "WARNING")


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return get_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")


class ContextLogger:
    """Context manager for temporary log level changes."""

    def __init__(self, logger_name: str, level: Union[str, int]):
        self.logger_name = logger_name
        self.new_level = (
            level if isinstance(level, int) else getattr(logging, level.upper())
        )
        self.original_level = None

    def __enter__(self):
        logger = logging.getLogger(self.logger_name)
        self.original_level = logger.level
        logger.setLevel(self.new_level)
        return logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_level is not None:
            logger = logging.getLogger(self.logger_name)
            logger.setLevel(self.original_level)


def log_function_call(logger: logging.Logger, level: int = logging.DEBUG):
    """Decorator to log function calls."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.log(
                level, f"Calling {func.__name__} with args={args}, kwargs={kwargs}"
            )
            try:
                result = func(*args, **kwargs)
                logger.log(level, f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} failed with error: {e}")
                raise

        return wrapper

    return decorator


def log_performance(logger: logging.Logger, level: int = logging.INFO):
    """Decorator to log function performance."""
    import time

    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.log(level, f"{func.__name__} completed in {duration:.3f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"{func.__name__} failed after {duration:.3f}s with error: {e}"
                )
                raise

        return wrapper

    return decorator


# Convenience functions for common logging patterns
def log_isaac_startup(logger: logging.Logger) -> None:
    """Log Isaac Sim startup information."""
    logger.info("Starting Isaac Sim MCP Server")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")

    # Log Isaac Sim path if available
    isaac_path = os.getenv("ISAAC_SIM_PATH")
    if isaac_path:
        logger.info(f"Isaac Sim path: {isaac_path}")
    else:
        logger.warning("ISAAC_SIM_PATH not set")


def log_usd_operation(logger: logging.Logger, operation: str, file_path: str) -> None:
    """Log USD operation."""
    logger.info(f"USD {operation}: {file_path}")


def log_mesh_operation(
    logger: logging.Logger, operation: str, mesh_info: Dict[str, Any]
) -> None:
    """Log mesh operation with details."""
    logger.info(f"Mesh {operation}: {mesh_info}")


def log_viewport_capture(
    logger: logging.Logger, width: int, height: int, format: str
) -> None:
    """Log viewport capture operation."""
    logger.info(f"Viewport capture: {width}x{height} {format}")


# Module-level logger for this module
_logger = get_logger(__name__)
