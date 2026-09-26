"""The thin MCP tool surfaces, defined once.

``unreal.tool_surface`` and ``blender.tool_surface`` pick between a backend's
full tool set and a short list of essentials that keeps an agent's context
small. The short lists live here and nowhere else: the registration modules
gate on them, and the config field descriptions and CLI help text are built
from them, so the documented list cannot drift from the registered one.

This module has no dependencies so the CLI can import it without paying for
the server.
"""

from __future__ import annotations

from typing import Tuple

#: Values accepted by ``*.tool_surface`` and the ``--*-tools`` CLI flags.
TOOL_SURFACES: Tuple[str, ...] = ("thin", "full")

#: Unreal tools registered when ``unreal.tool_surface`` is ``thin`` (the default).
THIN_UNREAL_TOOLS: Tuple[str, ...] = (
    "unreal_health_check",
    "ping_unreal",
    "list_unreal_instances",
    "control_unreal_ui",
    "capture_unreal_viewport",
    "execute_unreal_script",
)

#: Blender tools registered when ``blender.tool_surface`` is ``thin`` (opt-in;
#: the default is ``full``). Status, a scene overview and object listing to
#: orient, script execution for everything else, a viewport capture to check
#: the result, UI control for attached windows, and opening/saving the file.
THIN_BLENDER_TOOLS: Tuple[str, ...] = (
    "get_blender_info",
    "summarize_blender_scene",
    "list_blender_scene_objects",
    "execute_blender_script",
    "capture_blender_viewport",
    "control_blender_ui",
    "open_blender_file",
    "save_blender_file",
    "attach_blender_window",
)


def describe_tools(names: Tuple[str, ...]) -> str:
    """Join tool names for prose: ``a, b and c``.

    Args:
        names: Tool names in the order to state them.

    Returns:
        The names separated by commas, the last joined with ``and``.
    """
    if len(names) <= 1:
        return "".join(names)
    return f"{', '.join(names[:-1])} and {names[-1]}"
