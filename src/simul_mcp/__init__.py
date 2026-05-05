"""
Simul MCP Server

A Model Context Protocol (MCP) server for simulation and DCC tools that provides
USD scene understanding, mesh operations, and runtime integration capabilities.
"""

__version__ = "0.0.38"
__author__ = "khemoo"
__email__ = ""

# Package metadata
__title__ = "simul-mcp"
__description__ = "MCP server for simulation and DCC tools with USD scene understanding"
__url__ = "https://github.com/khemoo/simul-mcp"
__license__ = "MIT"

# Version info tuple
VERSION = tuple(map(int, __version__.split(".")))

# Export main components
from .config import Settings, get_settings
from .logging import setup_logging, get_logger

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "VERSION",
    "Settings",
    "get_settings",
    "setup_logging",
    "get_logger",
]
