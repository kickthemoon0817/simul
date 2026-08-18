"""Shared constants and helpers for the Isaac tool mixins."""

"""
Isaac Sim MCP tools for Simul MCP Server.

This module provides granular Isaac Sim tools over the repo-owned typed bridge
when available, falling back to raw script execution only when required by the
current coverage. For legacy compatibility, raw script execution can still use
the stock VS Code socket when bridge usage is disabled.
"""

import json
import textwrap
from typing import Annotated, Any, Callable, Dict, List, Optional

from pydantic import BeforeValidator

from ....adapters import IsaacSocketClient, ScriptResult
from ....config import Settings, get_settings
from ....logging import LoggerMixin, get_logger
from ....utils.paths import PathPolicy
from ...schemas.common import ErrorResponse

logger = get_logger(__name__)


def _coerce_str_to_float_list(value: Any) -> Any:
    """Pydantic ``BeforeValidator``: accept ``'[1, 2, 3]'`` as a list of floats.

    Some MCP clients serialise list arguments as JSON-encoded strings. Pydantic
    2.12 dropped the implicit ``str`` -> ``list`` coercion that earlier versions
    accepted, so we normalise here before validation runs.
    """
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return value
        if isinstance(parsed, list):
            return parsed
    return value


# List of floats that tolerates JSON-string inputs from forgiving MCP clients.
FloatList = Annotated[List[float], BeforeValidator(_coerce_str_to_float_list)]

# Aspect name -> the method that reads it. Fifteen tools took exactly one
# prim_path and differed only in which of these they returned, so they are one
# tool with a selector rather than fifteen the caller has to tell apart.
PRIM_DETAIL_ASPECTS = {
    "info": "get_isaac_prim_info",
    "transform": "get_isaac_prim_transform",
    "ancestors": "get_isaac_prim_ancestors",
    "relationships": "get_isaac_prim_relationships",
    "variants": "get_isaac_prim_variants",
    "bounding_box": "get_isaac_bounding_box",
    "mesh": "get_isaac_mesh_info",
    "light": "get_isaac_light_info",
    "material": "get_isaac_material_info",
    "rigid_body": "get_isaac_rigid_body_info",
    "collision": "get_isaac_collision_info",
    "joint": "get_isaac_joint_info",
    "mass": "get_isaac_mass_properties",
    "animation": "get_isaac_animation_info",
}

# get_isaac_texture_dependencies takes a *root* path and scans for materials
# beneath it, so as a per-prim "aspect" it answers a different question than
# its name promises: on the standard layout (materials under /World/Looks,
# mesh at /World/Cube) asking a mesh for its textures returns nothing. Left
# out until it can take a prim and resolve that prim's bound material.

# Largest raw script accepted for execution.
MAX_SCRIPT_BYTES = 100_000

# 4K on the long edge. The old ceiling of 7680 allowed a 59-megapixel capture —
# tens of megabytes pushed through several buffers before anything checked
# whether the result could be delivered.
MAX_CAPTURE_DIMENSION = 3840

# Largest capture returned as inline base64. Encoding grows a file by ~33%, and
# ~256 KB of PNG is already a large tool result; past this the caller gets the
# path instead, which is the shape that keeps working.
MAX_INLINE_CAPTURE_BYTES = 262_144

# Captures are kept rather than deleted, since the caller is handed a path — so
# something has to reclaim them. Keep the most recent N and drop the rest: an
# A/B pair needs two, a comparison sweep a few more, and nobody wants the whole
# session's frames sitting in the temp dir.
MAX_RETAINED_CAPTURES = 20

# How much of a Kit log to read when returning its tail. Kit logs reach several
# gigabytes and the read happens on Kit's main thread, so the window bounds the
# freeze; 2 MB still holds far more lines than the 500-entry maximum.
LOG_SCAN_WINDOW_BYTES = 2 * 1024 * 1024

# Array attributes big enough that pulling their value is the cost being
# avoided. Gating on these names rather than on isArray matters: xformOpOrder
# is a one-element token[] telling a caller which xform ops exist, and
# primvars:displayColor is a one-element color3f[] carrying the object colour.
# Both are arrays, neither is bulk, and skipping them loses information the
# caller needs while saving nothing. Anything not listed here takes the normal
# path, which already collapses arrays over 16 elements to a count.
BULK_GEOMETRY_ATTRIBUTES = frozenset(
    {
        "points",
        "normals",
        "velocities",
        "accelerations",
        "faceVertexIndices",
        "faceVertexCounts",
        "holeIndices",
        "cornerIndices",
        "cornerSharpnesses",
        "creaseIndices",
        "creaseLengths",
        "creaseSharpnesses",
        "curveVertexCounts",
        "widths",
        "primvars:st",
        "primvars:normals",
        # PointInstancer
        "positions",
        "orientations",
        "scales",
        "protoIndices",
        "invisibleIds",
        "ids",
    }
)



def _pyval(value: Any) -> str:
    """Render a call parameter as a Python literal for embedding in a script.

    The generated scripts are Python source, so parameters must be embedded as
    Python literals. ``json.dumps`` spells ``None``/``True``/``False`` as
    ``null``/``true``/``false``, which are undefined names once they land in the
    script and raise ``NameError`` inside Isaac Sim. ``repr`` keeps them Python.

    Use this for every embedded parameter, including required ones — a single
    idiom is what keeps the ``null`` class of bug from coming back.
    """
    return repr(value)


