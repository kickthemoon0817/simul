"""
Logging configuration and utilities for Isaac Sim MCP Server.

This module provides centralized logging configuration using Python's dictConfig
with support for multiple handlers, formatters, and log levels.
"""

import logging
import logging.config
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union
import yaml

from .config import Settings, get_settings


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
        config_file = Path("config/logging.yaml")
    else:
        config_file = Path(config_file)

    # Load logging configuration
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

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


def _setup_fallback_logging(
    settings: Settings, log_level: Optional[str] = None
) -> None:
    """Setup fallback logging configuration when YAML config fails."""
    level = log_level or settings.logging.level

    # Create logs directory
    log_path = Path(settings.logging.file_path)
    log_dir = log_path.parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # Basic logging configuration
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(settings.logging.file_path, encoding="utf-8"),
        ],
    )


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
