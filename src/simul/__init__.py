"""
Simul -- an agent toolkit for 3D simulation.

Connects AI agents to Isaac Sim, Unreal Engine 5, Blender and OpenUSD through
an MCP server (``simul.mcp``), an engine setup CLI (``simul.cli``), in-editor
bridges (``simul.bridge_ext``, ``simul.blender_bridge``) and a headless USD
library (``simul.usd``). Distributed as ``simul-toolkit``.
"""

__version__ = "0.1.3"
__author__ = "khemoo"
__email__ = ""

# Package metadata
__title__ = "simul-toolkit"
__description__ = "Agent toolkit for 3D simulation: MCP server, engine setup CLI and in-editor bridges for Isaac Sim, Unreal Engine 5, Blender and OpenUSD"
__url__ = "https://github.com/kickthemoon0817/simul"
__license__ = "Apache-2.0"

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
