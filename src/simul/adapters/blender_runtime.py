"""
Blender runtime adapter for Simul.

This module provides an adapter for Blender runtime operations through the
optional `bpy` Python module.
"""

from __future__ import annotations

import base64
import difflib
import io
import json
import logging
import math
import numbers
import os
import re
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    overload,
)

# The Blender add-on bundles these modules next to this one; simready_usd
# imports pxr only when it is called.
from ..utils.regex_guard import exponential_regex_construct
from .simready_usd import author_simready_layer, usd_identifier

try:
    import bpy

    BLENDER_AVAILABLE = True
except ImportError:
    bpy = None
    BLENDER_AVAILABLE = False

if TYPE_CHECKING:
    from ..config import Settings
    from ..utils.paths import PathPolicy

logger = logging.getLogger(__name__)

_CONTEXT_MEMBER_ERROR = "AttributeError: 'Context' object has no attribute"
CONTEXT_OVERRIDE_HINT = (
    "Hint: bpy.context lost its window/view-layer members, usually because the "
    "script loaded a file (read_homefile/open_mainfile), which replaces the window "
    "the call was bound to. After the load, wrap context-dependent operators such "
    "as exporters and render in an override built from the new window: "
    "win = bpy.context.window_manager.windows[0]; "
    "area = next(a for a in win.screen.areas if a.type == 'VIEW_3D'); "
    "region = next(r for r in area.regions if r.type == 'WINDOW'); "
    "with bpy.context.temp_override(window=win, screen=win.screen, scene=win.scene, "
    "view_layer=win.view_layer, area=area, region=region): bpy.ops.export_scene.gltf(...)"
)

# execute_script returns at most this much printed output unless the caller
# asks for more, up to the maximum (issue #231). BlenderExecuteScriptRequest
# repeats both numbers: this module ships inside the Blender add-on and cannot
# import the MCP schemas. The default keeps a typical printed result inside the
# server's 40 KB result budget.
DEFAULT_SCRIPT_OUTPUT_CHARS = 32_000
MAX_SCRIPT_OUTPUT_CHARS = 100_000

# The server measures a result as JSON text with non-ASCII escaped, and above
# its 400 KB hard limit it replaces the whole field instead of passing this
# prefix through. A character costs 1 to 12 bytes there: printable ASCII at most
# 2 (an escaped quote or newline), control characters and other non-ASCII text 6,
# an emoji 12. So the prefix is also held under this many JSON bytes, which
# printable ASCII never reaches (2 * MAX_SCRIPT_OUTPUT_CHARS + 2 quotes), and
# which leaves room for return_value and the rest of the reply. The MCP schemas
# repeat it as SCRIPT_OUTPUT_JSON_BYTES_MAX.
MAX_SCRIPT_OUTPUT_JSON_BYTES = 300_000

# capture_viewport methods (issue #228). "auto" means "viewport" in attached
# mode and "render" in embedded mode, which has no editor window.
CAPTURE_METHODS = ("auto", "viewport", "render")

# Engines the draw manager draws itself, so an offscreen redraw of a 3D View
# shows them. An external engine (Cycles, third-party) draws its Rendered view
# only into the on-screen editor; offscreen, only the overlays would appear.
_DRAW_MANAGER_ENGINES = frozenset(
    {"BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "BLENDER_WORKBENCH"}
)

# Object types without renderable bounds, left out when framing a temporary camera.
_UNFRAMED_TYPES = frozenset({"CAMERA", "LIGHT", "LIGHT_PROBE", "SPEAKER", "EMPTY"})

# A temporary framing camera looks at the objects from this side, the 3/4 view
# of Blender's default camera.
_FRAMING_DIRECTION = (0.68, -0.64, 0.46)

# focus_on_object places the camera on this side of the object: in front of and
# above it, along the -Y +Z diagonal (Blender's front view side).
_FOCUS_DIRECTION = (0.0, -math.sqrt(0.5), math.sqrt(0.5))

# A framed bounding sphere fills about 1 / _FRAMING_MARGIN of the frame's
# narrower side.
_FRAMING_MARGIN = 1.05

# Size framed for an object whose bounding box is a single point: an empty, a
# light, a camera or a mesh without vertices. An empty's default display size.
_POINT_RADIUS = 1.0

# Blender's matrices, mathutils and object transforms are single precision:
# rounding to float32 changes a number by up to half of FLT_EPSILON of its size.
_FLOAT32_EPSILON = 2.0**-23

# A camera counts as evaluated where it was placed when its location is within
# _PLACEMENT_LOCATION_PRECISION times the farthest distance from the origin
# involved (_placement_extent), and its view and up axes within the angles of
# _placement_turn. Rounding grows with those distances, so the location
# tolerance does too, with no fixed floor: on Blender 5.0.1,
# cameras on random rigs 0.1 mm to 100 km out were evaluated up to 2.5
# epsilons of the extent from their placement (3 cm with a 146 km extent,
# 32 nm with 0.17 m), and the former 0.1 mm floor hid a 50 micrometre Copy
# Location in a millimetre scene. The rotation Blender stores does not grow
# with the scene: those rigs turned cameras by up to 2e-5 rad, and by more
# near gimbal lock, where Blender's Euler decomposition loses precision; the
# check reads that from the stored transform instead (_stores_placement).
# _PLACEMENT_ANGLE_TOLERANCE, 1e-4 rad, is about a quarter of a pixel across a
# 1920-pixel frame at 50 mm. A constraint that aims the camera turns it by more,
# as it takes the camera's direction from two locations that rounding moves.
_PLACEMENT_LOCATION_PRECISION = 8 * _FLOAT32_EPSILON
_PLACEMENT_ANGLE_TOLERANCE = 1e-4

# Constraints that turn their owner toward a target, along the direction from
# the owner's location to the target's.
_AIM_CONSTRAINT_TYPES = frozenset(
    {"DAMPED_TRACK", "LOCKED_TRACK", "STRETCH_TO", "TRACK_TO"}
)

# How far single precision can turn an axis of an object's world matrix: its
# rotation angles round by up to half an epsilon of their size each (pi / 2
# epsilons for a half turn), and its entries by about an epsilon more. On
# Blender 5.0.1 a Track To's Target Z behaved as if turned by up to 1 epsilon.
_TARGET_AXIS_ROUNDING = 3 * _FLOAT32_EPSILON

# Constraints that only turn or scale their owner: Blender leaves its location
# as it was, so their targets do not enter it.
_IN_PLACE_CONSTRAINT_TYPES = _AIM_CONSTRAINT_TYPES | {"COPY_ROTATION", "COPY_SCALE"}

# The object types with a modifier stack (BKE_object_supports_modifiers,
# Blender 5.0.1).
_MODIFIER_OWNER_TYPES = frozenset(
    {
        "CURVE",
        "CURVES",
        "FONT",
        "GREASEPENCIL",
        "LATTICE",
        "MESH",
        "POINTCLOUD",
        "SURFACE",
        "VOLUME",
    }
)

# The settings of each modifier type that name an object whose world matrix
# the modifier reads whenever one is set: Blender 5.0.1 relates the modifier to
# that object's transform (its update_depsgraph). _modifier_transform_sources
# adds the settings that some types read only under a condition.
_MODIFIER_TRANSFORM_SETTINGS: Dict[str, Tuple[str, ...]] = {
    "ARMATURE": ("object",),
    "ARRAY": ("offset_object",),
    "CAST": ("object",),
    "CURVE": ("object",),
    "GREASE_PENCIL_ARMATURE": ("object",),
    "GREASE_PENCIL_ARRAY": ("offset_object",),
    "GREASE_PENCIL_BUILD": ("object",),
    "GREASE_PENCIL_HOOK": ("object",),
    "GREASE_PENCIL_LATTICE": ("object",),
    "GREASE_PENCIL_MIRROR": ("object",),
    "GREASE_PENCIL_OUTLINE": ("object",),
    "GREASE_PENCIL_SHRINKWRAP": ("target", "auxiliary_target"),
    "GREASE_PENCIL_TINT": ("object",),
    "GREASE_PENCIL_VERTEX_WEIGHT_PROXIMITY": ("object",),
    "HOOK": ("object",),
    "LATTICE": ("object",),
    "LINEART": ("light_contour_object",),
    "MASK": ("armature",),
    "MESH_DEFORM": ("object",),
    "MESH_TO_VOLUME": ("object",),
    "MIRROR": ("mirror_object",),
    "NORMAL_EDIT": ("target",),
    "PARTICLE_INSTANCE": ("object",),
    "SCREW": ("object",),
    "SHRINKWRAP": ("target", "auxiliary_target"),
    "SIMPLE_DEFORM": ("origin",),
    "UV_WARP": ("object_from", "object_to"),
    "VERTEX_WEIGHT_PROXIMITY": ("target",),
    "VOLUME_TO_MESH": ("object",),
    "WAVE": ("start_position_object",),
}

# Modifier types that map a texture through an object's space, as (texture,
# mapping, object) settings: they read the object's transform only with a
# texture and the OBJECT mapping.
_MODIFIER_TEXTURE_SETTINGS: Dict[str, Tuple[str, str, str]] = {
    "DISPLACE": ("texture", "texture_coords", "texture_coords_object"),
    "VERTEX_WEIGHT_EDIT": ("mask_texture", "mask_tex_mapping", "mask_tex_map_object"),
    "VERTEX_WEIGHT_MIX": ("mask_texture", "mask_tex_mapping", "mask_tex_map_object"),
    "VERTEX_WEIGHT_PROXIMITY": (
        "mask_texture",
        "mask_tex_mapping",
        "mask_tex_map_object",
    ),
    "WARP": ("texture", "texture_coords", "texture_coords_object"),
    "WAVE": ("texture", "texture_coords", "texture_coords_object"),
}

# export_file: the extensions each format writes. Blender's glTF exporter
# replaces any other extension with .glb or .gltf, so a path ending in .gltf
# exported as GLB overwrote a .glb beside it; its USD exporter refuses other
# extensions, and the others write whatever path they get, a .blend included.
EXPORT_EXTENSIONS: Dict[str, Tuple[str, ...]] = {
    "OBJ": (".obj",),
    "FBX": (".fbx",),
    "GLTF": (".glb", ".gltf"),
    "USD": (".usd", ".usda", ".usdc", ".usdz"),
    "STL": (".stl",),
    "PLY": (".ply",),
}

# The glTF layouts export_file writes: one binary .glb, a .gltf with its .bin
# and textures beside it, or a single .gltf with everything inline. Blender
# offers the last only while its glTF add-on's preference allows it.
GLTF_FORMATS = ("GLB", "GLTF_SEPARATE", "GLTF_EMBEDDED")
GltfFormat = Literal["GLB", "GLTF_SEPARATE", "GLTF_EMBEDDED"]
ExportAxis = Literal["X", "Y", "Z", "-X", "-Y", "-Z"]
UsdUpAxis = Literal["Y", "Z"]

# The operator property each export_file option sets, per format, in the order
# tried: the first one this Blender's exporter defines is used. Candidates cover
# Blender 4.2 through 5.0.1 (checked against each version's operators) and the
# Python exporters of earlier versions. A format without an entry has no such
# option: asking for it there is an error rather than a silent no-op.
_EXPORT_SELECTION_PROPERTIES: Dict[str, Tuple[str, ...]] = {
    "OBJ": ("export_selected_objects", "use_selection"),
    "FBX": ("use_selection",),
    "GLTF": ("use_selection", "export_selected"),
    "USD": ("selected_objects_only",),
    "STL": ("export_selected_objects", "use_selection"),
    "PLY": ("export_selected_objects", "use_selection"),
}
_EXPORT_MODIFIER_PROPERTIES: Dict[str, Tuple[str, ...]] = {
    "OBJ": ("apply_modifiers", "use_mesh_modifiers"),
    "FBX": ("use_mesh_modifiers",),
    "GLTF": ("export_apply",),
    "STL": ("apply_modifiers", "use_mesh_modifiers"),
    "PLY": ("apply_modifiers", "use_mesh_modifiers"),
}
_EXPORT_ANIMATION_PROPERTIES: Dict[str, Tuple[str, ...]] = {
    "FBX": ("bake_anim",),
    "GLTF": ("export_animations",),
    "USD": ("export_animation",),
}
_EXPORT_AXIS_PROPERTIES: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {
    "FBX": (("axis_forward",), ("axis_up",)),
    "OBJ": (("forward_axis", "axis_forward"), ("up_axis", "axis_up")),
    "STL": (("forward_axis", "axis_forward"), ("up_axis", "axis_up")),
    "PLY": (("forward_axis", "axis_forward"), ("up_axis", "axis_up")),
}
# Import scale: Blender's glTF importer has no such option, and on 5.0.1 its
# USD importer dropped its scale for a USD file Blender had written, whose
# generated root prim it does not import. Those two are scaled afterwards.
_IMPORT_SCALE_PROPERTIES: Dict[str, Tuple[str, ...]] = {
    "OBJ": ("global_scale",),
    "FBX": ("global_scale",),
    "STL": ("global_scale",),
    "PLY": ("global_scale",),
}

# USD's named scene units (convert_scene_units, Blender 4.4+), by metersPerUnit.
_USD_SCENE_UNITS = {
    1.0: "METERS",
    1000.0: "KILOMETERS",
    0.01: "CENTIMETERS",
    0.001: "MILLIMETERS",
    0.0254: "INCHES",
    0.3048: "FEET",
    0.9144: "YARDS",
}

# Subfolders of the export folder searched for files an export wrote, besides
# any it created: Blender's USD exporter writes its textures into "textures".
_EXPORT_TEXTURE_DIRS = ("textures",)

# At most this many folder entries are compared to find the files an export
# wrote, so that exporting into a huge folder stays quick.
_EXPORT_SCAN_LIMIT = 20_000

# Output-format settings a capture changes and restores, in restore order:
# media_type first, so that the saved file_format is valid again.
_IMAGE_FORMAT_FIELDS = (
    "media_type",
    "file_format",
    "color_mode",
    "color_depth",
    "quality",
)

# Settings of a modifier that add_modifier takes from its own arguments rather
# than from params: the name comes from modifier_name, the type from
# modifier_type, and rna_type is Blender's description of the struct.
_RESERVED_MODIFIER_SETTINGS = frozenset({"name", "type", "rna_type"})

# Where a datablock named for a pointer setting is looked up: the bpy.data
# collection of each ID type such a setting can point to, by its RNA struct
# identifier. A subtype (GeometryNodeTree, say) is found through its base.
_ID_COLLECTIONS: Dict[str, str] = {
    "Action": "actions",
    "CacheFile": "cache_files",
    "Camera": "cameras",
    "Collection": "collections",
    "Curve": "curves",
    "Image": "images",
    "Light": "lights",
    "Material": "materials",
    "Mesh": "meshes",
    "NodeTree": "node_groups",
    "Object": "objects",
    "Texture": "textures",
    "VectorFont": "fonts",
}

# Modifier types whose viewport and render results can differ through a
# second level setting: (viewport setting, render setting).
_RENDER_LEVEL_SETTINGS: Dict[str, Tuple[str, str]] = {
    "MULTIRES": ("levels", "render_levels"),
    "SUBSURF": ("levels", "render_levels"),
}


class UnsavedChangesError(RuntimeError):
    """Opening a file would discard the open file's unsaved changes.

    Its class name is the ``error_type`` clients see, and ``details`` name the
    open file (None when it was never saved).
    """

    def __init__(self, open_file: Optional[str], requested: str) -> None:
        where = f"'{open_file}'" if open_file else "(never saved)"
        save = (
            "save_blender_file"
            if open_file
            else "save_blender_file with a file_path, since it was never saved,"
        )
        super().__init__(
            f"The open file {where} has unsaved changes, which opening '{requested}' "
            f"would discard. Save them first with {save} and open the file again, or "
            "call open_blender_file with discard_unsaved=true to discard them."
        )
        self.details = {"blend_file_path": open_file, "is_dirty": True}


def add_context_hint(error: Optional[str]) -> Optional[str]:
    """Append the context-override recipe to a missing ``bpy.context`` member error.

    Idempotent, so both the add-on and the attached client may apply it.
    """
    if not error or _CONTEXT_MEMBER_ERROR not in error or CONTEXT_OVERRIDE_HINT in error:
        return error
    return f"{error}. {CONTEXT_OVERRIDE_HINT}"


def _script_error(exc: BaseException) -> Optional[str]:
    """Describe what an ``execute_script`` script raised, for its ``error`` field.

    Args:
        exc: Anything the script raised, ``SystemExit`` and
            ``KeyboardInterrupt`` included.

    Returns:
        ``<class>: <message>``, with the context-override recipe for a missing
        ``bpy.context`` member and what happened for an exit or interrupt.
    """
    if isinstance(exc, (SystemExit, KeyboardInterrupt)):
        name = type(exc).__name__
        error = f"{name}: {exc}" if str(exc) else name
        if isinstance(exc, SystemExit):
            return (
                f"{error}. The script called sys.exit() or exit(), which ends only the "
                "script here: the lines after it did not run, and Blender keeps running."
            )
        return f"{error}. The script was interrupted; the lines after it did not run."
    return add_context_hint(f"{type(exc).__name__}: {exc}")


def script_output_prefix(output: str, max_chars: int) -> str:
    """Return the part of a script's printed output that execute_script sends back.

    That is its first ``max_chars`` characters, cut further when their JSON
    encoding exceeds MAX_SCRIPT_OUTPUT_JSON_BYTES, so that the server passes
    the prefix through instead of replacing it whole.

    Args:
        output: The script's full printed output.
        max_chars: The caller's character cap.

    Returns:
        The longest such prefix.
    """
    prefix = output[:max_chars]
    if len(json.dumps(prefix)) <= MAX_SCRIPT_OUTPUT_JSON_BYTES:
        return prefix
    # json.dumps escapes each character on its own, so the encoded length
    # grows with the prefix and a bisection finds the longest one that fits.
    low, high = 0, len(prefix)
    while low < high:
        middle = (low + high + 1) // 2
        if len(json.dumps(prefix[:middle])) <= MAX_SCRIPT_OUTPUT_JSON_BYTES:
            low = middle
        else:
            high = middle - 1
    return prefix[:low]


def _bpy_op(description: str, operator: Callable[..., Any], /, **properties: Any) -> Any:
    """Run one ``bpy.ops`` operator, noting ``description`` on its failure.

    The exception propagates unchanged, so its class reaches the error
    envelope's ``error_type``: bpy raises RuntimeError for a failed poll or an
    error the operator reports, and TypeError for a bad property value. The
    note names what bpy's message leaves out, such as the object or the
    resolved file path; error envelopes render it as a ``description: message``
    prefix (``simul.utils.errors.exception_message``).

    Args:
        description: What failed, e.g. ``Failed to add rigid body to 'Cube'``.
        operator: The ``bpy.ops`` operator to call.
        **properties: The operator's properties.

    Returns:
        The operator's result.
    """
    try:
        return operator(**properties)
    except Exception as exc:
        exc.add_note(description)
        raise


def _mathutils() -> Any:
    """Return Blender's math module; it is importable wherever bpy is.

    Imported when used, as bpy-only code is, rather than with this module.
    """
    import mathutils  # type: ignore[import-not-found]

    return mathutils


# The order of the Euler angles that set_object_transform and control_ui's
# set_property take, and that get_object_info and set_object_transform report
# as rotation_euler, whatever the object's rotation_mode; the camera tools'
# world-space rotation_euler is XYZ as well.
TRANSFORM_EULER_ORDER = "XYZ"


def _finite(values: Sequence[float], count: int, name: str) -> Tuple[float, ...]:
    """Return ``count`` finite floats, or raise ValueError naming ``name``."""
    numbers = tuple(float(v) for v in values)
    if len(numbers) != count or not all(math.isfinite(v) for v in numbers):
        raise ValueError(f"{name} must be {count} finite numbers")
    return numbers


def local_rotation_quaternion(obj: Any) -> Any:
    """
    Return the rotation of an object's own transform as a unit quaternion.

    Blender keeps one rotation per object, in the property its
    ``rotation_mode`` names: ``rotation_euler`` (its angles applied in the
    mode's order), ``rotation_quaternion`` or ``rotation_axis_angle``. The
    other two keep whatever they held when the mode changed, so on a glTF
    import (QUATERNION) ``rotation_euler`` stays [0, 0, 0] however the object
    is turned. Blender normalizes the quaternion, and takes an axis-angle
    with a zero axis for no rotation, when it builds the matrix.

    Args:
        obj: The Blender object.

    Returns:
        A ``mathutils.Quaternion`` (w, x, y, z), before parenting, deltas and
        constraints.
    """
    mathutils = _mathutils()
    mode = obj.rotation_mode
    if mode == "QUATERNION":
        quaternion = mathutils.Quaternion(tuple(obj.rotation_quaternion))
        if quaternion.magnitude == 0.0:
            return mathutils.Quaternion()
        return quaternion.normalized()
    if mode == "AXIS_ANGLE":
        angle, *axis = tuple(obj.rotation_axis_angle)
        direction = mathutils.Vector(axis)
        if direction.length == 0.0:
            return mathutils.Quaternion()
        return mathutils.Quaternion(direction.normalized(), angle)
    return mathutils.Euler(tuple(obj.rotation_euler), mode).to_quaternion()


def local_rotation_euler(obj: Any) -> List[float]:
    """
    Return the rotation of an object's own transform as XYZ Euler angles in radians.

    An XYZ object's own angles come back as Blender stores them; any other
    rotation mode is converted.

    Args:
        obj: The Blender object.

    Returns:
        [x, y, z] in TRANSFORM_EULER_ORDER.
    """
    if obj.rotation_mode == TRANSFORM_EULER_ORDER:
        return [float(a) for a in obj.rotation_euler]
    return [
        float(a) for a in local_rotation_quaternion(obj).to_euler(TRANSFORM_EULER_ORDER)
    ]


def stored_rotation(obj: Any) -> List[float]:
    """
    Return the rotation as Blender stores it for the object's rotation_mode.

    Args:
        obj: The Blender object.

    Returns:
        Euler angles [x, y, z] (applied in the mode's order), a quaternion
        [w, x, y, z] for QUATERNION, or [angle, x, y, z] for AXIS_ANGLE.
    """
    mode = obj.rotation_mode
    if mode == "QUATERNION":
        return [float(v) for v in obj.rotation_quaternion]
    if mode == "AXIS_ANGLE":
        return [float(v) for v in obj.rotation_axis_angle]
    return [float(v) for v in obj.rotation_euler]


def write_rotation(
    obj: Any,
    rotation_euler: Optional[Sequence[float]] = None,
    rotation_quaternion: Optional[Sequence[float]] = None,
) -> None:
    """
    Give an object's own transform a rotation, whatever its rotation_mode.

    Blender reads only the property of the object's rotation mode, so a
    write to ``rotation_euler`` changes nothing on a QUATERNION or
    AXIS_ANGLE object, and another Euler order reads the angles in its own
    order. The rotation is converted to the mode's property instead, and the
    mode is kept. A converted Euler rotation takes the angles nearest the
    object's current ones, and a quaternion converted from Euler angles the
    sign of its current one, so a small turn stays a small change of the
    stored values (no flip by a whole turn or to the opposite quaternion).

    Args:
        obj: The Blender object.
        rotation_euler: XYZ Euler angles [x, y, z] in radians
            (TRANSFORM_EULER_ORDER).
        rotation_quaternion: A quaternion [w, x, y, z]; it is normalized.

    Raises:
        ValueError: Neither or both are given, a value is not finite, or the
            quaternion has zero length.
    """
    if (rotation_euler is None) == (rotation_quaternion is None):
        raise ValueError("Give either rotation_euler or rotation_quaternion")
    mathutils = _mathutils()
    mode = obj.rotation_mode
    if rotation_euler is not None:
        angles = _finite(rotation_euler, 3, "rotation_euler")
        if mode == TRANSFORM_EULER_ORDER:
            obj.rotation_euler = angles
            return
        quaternion = mathutils.Euler(angles, TRANSFORM_EULER_ORDER).to_quaternion()
        if mode == "QUATERNION":
            quaternion.make_compatible(local_rotation_quaternion(obj))
    else:
        values = _finite(rotation_quaternion or (), 4, "rotation_quaternion")
        quaternion = mathutils.Quaternion(values)
        if quaternion.magnitude == 0.0:
            raise ValueError("rotation_quaternion must not have zero length")
        quaternion = quaternion.normalized()
    if mode == "QUATERNION":
        obj.rotation_quaternion = quaternion
    elif mode == "AXIS_ANGLE":
        axis, angle = quaternion.to_axis_angle()
        obj.rotation_axis_angle = (float(angle), *(float(v) for v in axis))
    else:
        obj.rotation_euler = quaternion.to_euler(mode, obj.rotation_euler)


def _nearest_rotation(linear: List[List[float]]) -> Optional[List[List[float]]]:
    """
    Return the rotation nearest a 3x3 matrix: the orthogonal factor of its polar decomposition.

    A turned child of a parent with non-uniform scale has a sheared world
    matrix, which no rotation and scale rebuild. ``Matrix.decompose`` and
    ``to_euler`` then read the rotation from some of its entries, and the two
    disagree: on Blender 5.0.1 a glTF link under a parent scaled 0.5 on Z
    came out 0.19 and 0.15 rad from the rotation of its axes. Newton's
    iteration U = (U + U^-T) / 2 converges to the orthogonal factor, which for
    such a matrix is the parent's rotation times the child's. A mirroring
    matrix gives an orthogonal factor that mirrors too; it is negated into a
    rotation, the scale negative on every axis, as Blender decomposes it.

    Args:
        linear: The matrix's rows.

    Returns:
        The rotation's rows, or None for a singular matrix.
    """
    current = [[float(v) for v in row] for row in linear]
    for _ in range(64):
        a = current
        # A^-T is the cofactor matrix divided by the determinant.
        cofactors = [
            [
                a[(i + 1) % 3][(j + 1) % 3] * a[(i + 2) % 3][(j + 2) % 3]
                - a[(i + 1) % 3][(j + 2) % 3] * a[(i + 2) % 3][(j + 1) % 3]
                for j in range(3)
            ]
            for i in range(3)
        ]
        determinant = sum(a[0][j] * cofactors[0][j] for j in range(3))
        if determinant == 0.0 or not math.isfinite(determinant):
            return None
        following = [
            [(a[i][j] + cofactors[i][j] / determinant) / 2 for j in range(3)]
            for i in range(3)
        ]
        change = max(abs(following[i][j] - a[i][j]) for i in range(3) for j in range(3))
        current = following
        if change < 1e-12:
            break
    mirrored = (
        current[0][0] * (current[1][1] * current[2][2] - current[1][2] * current[2][1])
        - current[0][1] * (current[1][0] * current[2][2] - current[1][2] * current[2][0])
        + current[0][2] * (current[1][0] * current[2][1] - current[1][1] * current[2][0])
    ) < 0
    return [[-v for v in row] for row in current] if mirrored else current


def world_transform(obj: Any) -> Dict[str, Any]:
    """
    Report an object's world matrix, and that matrix decomposed.

    The caller evaluates transforms first (_evaluate_transforms): matrix_world
    lags a transform written since the last depsgraph evaluation in
    background Blender and embedded mode. The location and scale are
    ``Matrix.decompose``'s, which gives a mirrored matrix a scale negative on
    every axis. The rotation is the one nearest the matrix (_nearest_rotation),
    which also holds for the sheared matrix of a turned child under a parent
    with non-uniform scale; ``matrix_world`` is exact.

    Args:
        obj: The Blender object.

    Returns:
        world_location, world_rotation_euler (XYZ), world_rotation_quaternion
        (w, x, y, z), world_scale and matrix_world (four rows of four).
    """
    matrix = obj.matrix_world
    location, rotation, scale = matrix.decompose()
    rows = [[float(v) for v in row] for row in matrix]
    nearest = _nearest_rotation([row[:3] for row in rows[:3]])
    if nearest is not None:
        rotation = _mathutils().Matrix(nearest).to_quaternion()
    return {
        "world_location": [float(v) for v in location],
        "world_rotation_euler": [
            float(a) for a in rotation.to_euler(TRANSFORM_EULER_ORDER)
        ],
        "world_rotation_quaternion": [float(v) for v in rotation],
        "world_scale": [float(v) for v in scale],
        "matrix_world": rows,
    }


# Blender 4.4 introduced slotted actions: one action can animate several IDs,
# each through its own slot, and keeps each slot's F-curves in a channelbag
# inside the action's layers and strips. Blender 5.0 removed Action.fcurves,
# the only way to read an action's F-curves before 4.4.
SLOTTED_ACTIONS_VERSION = (4, 4, 0)


def action_fcurves(
    anim_data: Any, blender_version: Tuple[int, int, int]
) -> Optional[List[Any]]:
    """Return the F-curves through which an ID's action animates that ID.

    On Blender 4.4 and later these are the F-curves of the action slot the ID
    is assigned. An action shared with other IDs keeps theirs under other
    slots, and the older ``Action.fcurves`` (removed in 5.0) shows only the
    first slot's. Before 4.4 they are all of the action's F-curves.

    Args:
        anim_data: The ID's ``animation_data``; None when it has none.
        blender_version: The running Blender version.

    Returns:
        The F-curves, empty when the action has none for the ID, or None when
        no action animates the ID: it has no action or, on 4.4 and later, no
        slot of it.
    """
    action = None if anim_data is None else anim_data.action
    if action is None:
        return None
    if blender_version < SLOTTED_ACTIONS_VERSION:
        return list(action.fcurves)
    slot = anim_data.action_slot
    if slot is None:
        # Not only a shortcut: Blender 4.4.0's bpy_extras helper passes a
        # missing slot on to ActionKeyframeStrip.channelbag, which dereferences
        # it. 4.4.1 and later return None.
        return None
    from bpy_extras import anim_utils  # type: ignore[import-not-found]

    channelbag = anim_utils.action_get_channelbag_for_slot(action, slot)
    return [] if channelbag is None else list(channelbag.fcurves)


def _channel_name(data_path: str, index: int) -> str:
    """Name a keyframed property, or one element of it, as ``location[2]``."""
    return data_path if index < 0 else f"{data_path}[{index}]"


def _did_you_mean(name: str, choices: Sequence[str]) -> str:
    """``; did you mean 'a' or 'b'?`` for the choices closest to a mistyped name."""
    close = difflib.get_close_matches(name, list(choices), n=3, cutoff=0.6)
    if not close:
        return ""
    return "; did you mean " + " or ".join(f"'{choice}'" for choice in close) + "?"


def _enum_item(value: Any, items: Sequence[str], what: str) -> str:
    """The enum item identifier a value names, matched case-insensitively.

    An empty ``items`` means the enum lists its items only at run time, so the
    value passes through for Blender to check.
    """
    if not isinstance(value, str):
        raise ValueError(f"{what} takes one of {', '.join(items)}, got {value!r}")
    if not items or value in items:
        return value
    folded = [item for item in items if item.lower() == value.lower()]
    if len(folded) == 1:
        return folded[0]
    raise ValueError(
        f"{what} must be one of {', '.join(items)}, got '{value}'{_did_you_mean(value.upper(), items)}"
    )


def _check_rna_number(prop: Any, value: Any, what: str) -> None:
    """Refuse a value outside an INT or FLOAT property's hard limits.

    Blender clamps such a value without an error, so a DECIMATE ratio of 7
    would silently become 1.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{what} takes a number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{what} must be finite, got {value!r}")
    if prop.type == "INT" and not float(value).is_integer():
        raise ValueError(f"{what} takes a whole number, got {value!r}")
    if value < prop.hard_min or value > prop.hard_max:
        raise ValueError(
            f"{what} must lie between {prop.hard_min:g} and {prop.hard_max:g}, got {value!r}"
        )


def _finite_triple(value: Any, what: str) -> Tuple[float, float, float]:
    """Three finite coordinates as floats, or a ValueError naming ``what``.

    ``from_pydata`` flattens its vertex list before storing it, so a vertex
    of two or four numbers shifted every later vertex instead of failing.
    """
    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__"):
        raise ValueError(f"{what} must be [x, y, z], got {value!r}")
    if len(value) != 3:
        raise ValueError(f"{what} must have exactly 3 coordinates [x, y, z], got {len(value)}")
    triple = []
    for component in value:
        if isinstance(component, bool) or not isinstance(component, numbers.Real):
            raise ValueError(f"{what} coordinates must be numbers, got {component!r}")
        if not math.isfinite(component):
            raise ValueError(f"{what} coordinates must be finite, got {component!r}")
        triple.append(float(component))
    return (triple[0], triple[1], triple[2])


def _mesh_index(value: Any, what: str, vertex_count: int) -> int:
    """A vertex index of an edge or face, checked to be an integer within the vertex list."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{what} index {value!r} is not an integer")
    index = int(value)
    if index < 0 or index >= vertex_count:
        raise ValueError(f"{what} index {index} out of range [0, {vertex_count})")
    return index


@dataclass
class BlenderObjectEntry:
    """Serializable Blender object summary."""

    name: str
    object_type: str
    collection: Optional[str]
    visible: bool


class BlenderRuntimeSession:
    """
    Blender runtime session for scene inspection operations.

    This class provides read-focused operations that are safe for environments
    where Blender is running as a Python module.
    """

    logger = logger

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        path_policy: Optional[PathPolicy] = None,
        scene_scope: bool = False,
    ):
        """
        Initialize Blender runtime session.

        Args:
            settings: Configuration settings.
            path_policy: Injected policy for a standalone Blender add-on.
            scene_scope: Restrict object lookup to the selected window's scene.
        """
        if not BLENDER_AVAILABLE:
            raise ImportError(
                "Blender runtime not available. Please run in an environment "
                "where the bpy module is installed."
            )

        if settings is None and path_policy is None:
            from ..config import get_settings

            settings = get_settings()
        self.settings = settings
        self._scene_scope = scene_scope

        blender_module: Any = bpy
        version_raw = blender_module.app.version
        self._blender_version: Tuple[int, int, int] = (
            int(version_raw[0]),
            int(version_raw[1]),
            int(version_raw[2]),
        )
        self._is_blender_4_plus: bool = self._blender_version >= (4, 0, 0)
        self._is_blender_5_plus: bool = self._blender_version >= (5, 0, 0)

        self.logger.info(
            "Blender runtime session initialized (v%s)", self._blender_version
        )
        if path_policy is None:
            from ..utils.paths import PathPolicy

            assert self.settings is not None
            path_policy = PathPolicy.from_settings(self.settings)
        self._path_policy = path_policy

    @overload
    def _deny_outside_sandbox(self, path: str, *, write: bool = False) -> str: ...

    @overload
    def _deny_outside_sandbox(self, path: None, *, write: bool = False) -> None: ...

    def _deny_outside_sandbox(
        self, path: Optional[str], *, write: bool = False
    ) -> Optional[str]:
        """Refuse a filesystem path the sandbox policy does not allow.

        Enforced here, in the session layer, so every caller — MCP
        registration, a future CLI, tests — sits above the check. Raising
        keeps the void and dict-returning methods on one idiom; the shared
        envelope turns it into the standard error payload.

        Returns the policy-resolved path, which the caller must use for
        every later filesystem and ``bpy`` operation: the policy resolves
        ``~``/``$VAR``/relative prefixes before the containment test, so the
        raw string can name a different file than the one that was checked.
        With the sandbox disabled the raw path passes through untouched.

        Args:
            path: Filesystem path supplied by the caller, or None.
            write: Whether the caller writes to the location.

        Returns:
            The path to use for the operation, or None when none was given.

        Raises:
            SandboxDenied: If the path is outside the sandbox.
        """
        if path is None:
            return None
        return self._path_policy.authorize(path, write=write)

    @property
    def blender_version(self) -> Tuple[int, int, int]:
        """
        Blender version as a three-element tuple.

        Returns:
            Tuple of (major, minor, patch).
        """
        return self._blender_version

    @property
    def is_blender_4_plus(self) -> bool:
        """
        Whether the runtime is Blender 4.0 or newer.

        Returns:
            True when Blender version >= 4.0.0.
        """
        return self._is_blender_4_plus

    @property
    def is_blender_5_plus(self) -> bool:
        """
        Whether the runtime is Blender 5.0 or newer.

        Returns:
            True when Blender version >= 5.0.0.
        """
        return self._is_blender_5_plus

    def control_ui(
        self,
        agent_control: str,
        target: Optional[str] = None,
        area_id: Optional[str] = None,
        position: Optional[List[float]] = None,
        value: Optional[List[float]] = None,
        agent_id: str = "agent",
    ) -> Dict[str, Any]:
        """Perform a named UI action only in an explicitly attached GUI window."""
        if not self._scene_scope:
            raise RuntimeError("Agent control requires --blender-mode attached and simul blender attach")
        from ..blender_bridge.agent_control import control_ui

        return control_ui(agent_control, target, area_id, position, value, agent_id)

    def get_runtime_info(self) -> Dict[str, Any]:
        """
        Get Blender runtime information.

        Returns:
            Dictionary containing Blender runtime metadata.
        """
        if bpy is None:
            raise RuntimeError("bpy module is unavailable during runtime info query")

        blender_module: Any = bpy
        blender_app = blender_module.app

        return {
            "version": list(self._blender_version),
            "version_string": blender_app.version_string,
            "binary_path": blender_app.binary_path,
            "background": bool(blender_app.background),
            "blend_file_path": blender_module.data.filepath or None,
        }

    def list_scene_objects(
        self,
        collection_name: Optional[str] = None,
        include_hidden: bool = False,
        max_items: int = 200,
    ) -> Dict[str, Any]:
        """
        List scene objects from the active Blender data context.

        Args:
            collection_name: Optional collection name to filter objects.
            include_hidden: Include hidden objects when True.
            max_items: Maximum number of objects returned.

        Returns:
            Dictionary with object summaries and truncation metadata.
        """
        if max_items < 1:
            raise ValueError("max_items must be greater than zero")

        source_objects = self._resolve_object_source(collection_name)
        object_entries: List[BlenderObjectEntry] = []

        for scene_object in source_objects:
            object_visible = self._is_object_visible(scene_object)
            if not include_hidden and not object_visible:
                continue

            entry = BlenderObjectEntry(
                name=scene_object.name,
                object_type=scene_object.type,
                collection=collection_name,
                visible=object_visible,
            )
            object_entries.append(entry)

            if len(object_entries) >= max_items:
                break

        serialized_objects = [
            {
                "name": entry.name,
                "object_type": entry.object_type,
                "collection": entry.collection,
                "visible": entry.visible,
            }
            for entry in object_entries
        ]

        return {
            "collection": collection_name,
            "include_hidden": include_hidden,
            "max_items": max_items,
            "count": len(serialized_objects),
            "objects": serialized_objects,
            "truncated": len(serialized_objects) >= max_items,
        }

    def get_object_info(self, object_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a single Blender object.

        ``location``, ``rotation_euler`` and ``scale`` are the object's own
        transform, as Blender's Location, Rotation and Scale fields show it:
        relative to its parent, before constraints. The ``world_*`` values
        and ``matrix_world`` give where it is in the scene.

        Args:
            object_name: Name of the target object.

        Returns:
            Dictionary with local and world transforms, rotation mode, parent,
            modifiers, constraints, material slots, and visibility.
        """
        obj = self._get_object_or_raise(object_name)
        # matrix_world lags a transform written by an earlier call until the
        # depsgraph is evaluated (background Blender / embedded mode).
        self._evaluate_transforms()

        location = [float(v) for v in obj.location]
        scale = [float(v) for v in obj.scale]

        parent_name: Optional[str] = obj.parent.name if obj.parent else None
        children_names = [child.name for child in getattr(obj, "children", [])]

        modifiers = [
            {"name": mod.name, "modifier_type": mod.type}
            for mod in getattr(obj, "modifiers", [])
        ]
        constraints = [
            {"name": con.name, "constraint_type": con.type}
            for con in getattr(obj, "constraints", [])
        ]
        material_slots = [
            {
                "slot_index": idx,
                "material_name": slot.material.name if slot.material else None,
            }
            for idx, slot in enumerate(getattr(obj, "material_slots", []))
        ]

        return {
            "name": obj.name,
            "object_type": obj.type,
            "location": location,
            "rotation_euler": local_rotation_euler(obj),
            "scale": scale,
            "rotation_mode": obj.rotation_mode,
            "rotation_quaternion": [float(v) for v in local_rotation_quaternion(obj)],
            "rotation_in_mode": stored_rotation(obj),
            **world_transform(obj),
            "parent_name": parent_name,
            "children_names": children_names,
            "modifiers": modifiers,
            "constraints": constraints,
            "material_slots": material_slots,
            "visible": self._is_object_visible(obj),
        }

    def get_mesh_info(self, object_name: str) -> Dict[str, Any]:
        """
        Get a mesh object's geometry counts, before and after its modifiers.

        The base counts are the mesh data as stored; the evaluated counts are
        what Blender's depsgraph makes of it with the modifiers enabled in the
        viewport, which is the geometry an export writes. A mesh in Edit Mode
        is written back from the edit mesh first, so neither count is stale.

        Args:
            object_name: Name of the mesh object.

        Returns:
            Dictionary with vertex/edge/face/triangle counts of the base and
            of the evaluated mesh, the modifiers evaluated, UV layer names and
            warnings.
        """
        obj = self._get_object_or_raise(object_name)
        if obj.type != "MESH":
            raise ValueError(f"Object '{object_name}' is not a MESH (type={obj.type})")

        edit_mode = getattr(obj, "mode", "OBJECT") == "EDIT"
        if edit_mode:
            # Edit Mode keeps its changes in an edit mesh that the mesh data
            # sees only once they are written back.
            obj.update_from_editmode()
        mesh_data: Any = obj.data
        base = self._mesh_counts(mesh_data)
        evaluated = self._evaluated_mesh_counts(obj)
        modifiers = list(getattr(obj, "modifiers", []))
        warnings: List[str] = []
        if evaluated is None:
            warnings.append(
                f"Blender does not evaluate '{obj.name}' (it is not in the view "
                "layer), so the counts after its modifiers are unknown"
            )
        render_differs = [mod.name for mod in modifiers if self._render_differs(mod)]
        if render_differs:
            warnings.append(
                "The evaluated counts are the viewport's, and these modifiers are set "
                f"differently for render, which USD export evaluates: {', '.join(render_differs)}"
            )

        return {
            "object_name": object_name,
            **base,
            **{
                f"evaluated_{key}": (None if evaluated is None else evaluated[key])
                for key in base
            },
            "evaluated_modifiers": [mod.name for mod in modifiers if mod.show_viewport],
            "edit_mode": edit_mode,
            "uv_layer_names": [layer.name for layer in mesh_data.uv_layers],
            "has_shape_keys": mesh_data.shape_keys is not None,
            "warnings": warnings,
        }

    @staticmethod
    def _mesh_counts(mesh: Any) -> Dict[str, int]:
        """Vertex, edge, face and triangle counts of one mesh datablock.

        A face of n corners triangulates into n - 2 triangles, so the triangle
        count is the corner (loop) count less two per face.
        """
        faces = len(mesh.polygons)
        return {
            "vertex_count": len(mesh.vertices),
            "edge_count": len(mesh.edges),
            "face_count": faces,
            "triangle_count": len(mesh.loops) - 2 * faces,
        }

    def _evaluated_mesh_counts(self, obj: Any) -> Optional[Dict[str, int]]:
        """Counts of the mesh Blender evaluates for ``obj``, its modifiers applied.

        Returns:
            The counts, or None when the depsgraph holds no evaluated copy of
            the object (it is not in the view layer).
        """
        blender_module: Any = bpy
        depsgraph = blender_module.context.evaluated_depsgraph_get()
        evaluated = obj.evaluated_get(depsgraph)
        if not getattr(evaluated, "is_evaluated", False):
            return None
        mesh = evaluated.to_mesh()
        try:
            return self._mesh_counts(mesh)
        finally:
            evaluated.to_mesh_clear()

    @staticmethod
    def _render_differs(modifier: Any) -> bool:
        """Whether a modifier gives render a different result than the viewport."""
        if bool(modifier.show_render) != bool(modifier.show_viewport):
            return True
        if not modifier.show_render:
            return False
        levels = _RENDER_LEVEL_SETTINGS.get(modifier.type)
        return levels is not None and getattr(modifier, levels[0], None) != getattr(
            modifier, levels[1], None
        )

    def get_bounding_box(
        self, object_name: str, world_space: bool = True
    ) -> Dict[str, Any]:
        """
        Get the eight bounding-box corners of an object.

        Args:
            object_name: Name of the target object.
            world_space: Return corners in world space when True.

        Returns:
            Dictionary with corners, axis-aligned min/max, and space flag.
        """
        obj = self._get_object_or_raise(object_name)

        if world_space:
            # matrix_world lags a transform written by an earlier call until
            # the depsgraph is evaluated (background Blender / embedded mode).
            self._evaluate_transforms()
            corners = self._world_bound_box(obj)
        else:
            corners = [list(co) for co in obj.bound_box]

        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        zs = [c[2] for c in corners]

        return {
            "object_name": object_name,
            "corners": corners,
            "bbox_min": [min(xs), min(ys), min(zs)],
            "bbox_max": [max(xs), max(ys), max(zs)],
            "world_space": world_space,
        }

    def search_objects(
        self,
        name_pattern: Optional[str] = None,
        object_type: Optional[str] = None,
        max_results: int = 50,
    ) -> Dict[str, Any]:
        """
        Search for objects matching optional name pattern and type filter.

        Args:
            name_pattern: Python regular expression searched for in each object
                name; a repeated group that holds a quantifier or alternatives,
                as in (\\w+)+, is refused because it can run for hours.
            object_type: Blender object type filter (MESH, LIGHT, etc.).
            max_results: Maximum number of results.

        Returns:
            Dictionary with matched objects and truncation flag.

        Raises:
            ValueError: name_pattern is not a valid regular expression, or
                repeats a group that holds a quantifier or alternatives (see
                ``simul.utils.regex_guard``).
        """
        compiled_pattern = None
        if name_pattern:
            try:
                compiled_pattern = re.compile(name_pattern)
            except re.error as exc:
                raise ValueError(
                    f"name_pattern is not a valid regular expression: {exc}"
                ) from exc
            # The search runs on Blender's main thread, where nothing can stop it.
            construct = exponential_regex_construct(name_pattern)
            if construct is not None:
                raise ValueError(
                    f"name_pattern has {construct}: a failing search for it can "
                    "freeze Blender for hours, since every way of splitting an "
                    "object name between the repetitions is tried. Repeat a plain "
                    "character class instead (\\w+ rather than (\\w+)+)"
                )
        all_objects = self._resolve_object_source(None)
        matches: List[Dict[str, Any]] = []

        for obj in all_objects:
            if compiled_pattern and not compiled_pattern.search(obj.name):
                continue
            if object_type and obj.type != object_type:
                continue

            matches.append(
                {
                    "name": obj.name,
                    "object_type": obj.type,
                    "collection": None,
                    "visible": self._is_object_visible(obj),
                }
            )
            if len(matches) >= max_results:
                break

        return {
            "pattern": name_pattern,
            "object_type": object_type,
            "count": len(matches),
            "objects": matches,
            "truncated": len(matches) >= max_results,
        }

    def summarize_scene(self) -> Dict[str, Any]:
        """
        Produce a high-level scene summary grouped by object type.

        Returns:
            Dictionary with total counts, type breakdown, collections,
            active camera, and frame range.
        """
        blender_module: Any = bpy
        all_objects = self._resolve_object_source(None)
        type_counts: Dict[str, int] = {}

        for obj in all_objects:
            type_counts[obj.type] = type_counts.get(obj.type, 0) + 1

        scene = blender_module.context.scene
        collection_names = list(blender_module.data.collections.keys())
        active_camera: Optional[str] = scene.camera.name if scene.camera else None

        return {
            "total_objects": len(all_objects),
            "type_counts": type_counts,
            "collection_names": collection_names,
            "active_camera": active_camera,
            "frame_current": scene.frame_current,
            "frame_start": scene.frame_start,
            "frame_end": scene.frame_end,
        }

    def get_material_info(self, material_name: str) -> Dict[str, Any]:
        """
        Get material information with bounded node tree traversal.

        Traverses at most ``_MAX_MATERIAL_NODE_DEPTH`` levels from the
        output node to keep context small.

        Args:
            material_name: Name of the Blender material.

        Returns:
            Dictionary with node summary and Principled BSDF parameters.
        """
        blender_module: Any = bpy
        mat = blender_module.data.materials.get(material_name)
        if mat is None:
            raise ValueError(f"Material not found: {material_name}")

        use_nodes = bool(getattr(mat, "use_nodes", False))
        nodes: List[Dict[str, str]] = []
        principled_params: Optional[Dict[str, Any]] = None

        if use_nodes and mat.node_tree:
            for node in mat.node_tree.nodes:
                nodes.append(
                    {
                        "name": node.name,
                        "node_type": node.bl_idname,
                        "label": getattr(node, "label", ""),
                    }
                )
                if node.bl_idname == "ShaderNodeBsdfPrincipled":
                    principled_params = self._extract_principled_params(node)

        return {
            "material_name": material_name,
            "use_nodes": use_nodes,
            "nodes": nodes,
            "principled_params": principled_params,
        }

    def get_distance_between(
        self, object_name_a: str, object_name_b: str
    ) -> Dict[str, Any]:
        """
        Compute the Euclidean distance between two objects.

        Args:
            object_name_a: First object name.
            object_name_b: Second object name.

        Returns:
            Dictionary with distance and both world locations.
        """
        obj_a = self._get_object_or_raise(object_name_a)
        obj_b = self._get_object_or_raise(object_name_b)
        # matrix_world lags a transform written by an earlier call until the
        # depsgraph is evaluated (background Blender / embedded mode).
        self._evaluate_transforms()

        loc_a = [float(v) for v in obj_a.matrix_world.translation]
        loc_b = [float(v) for v in obj_b.matrix_world.translation]
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(loc_a, loc_b)))

        return {
            "object_name_a": object_name_a,
            "object_name_b": object_name_b,
            "distance": distance,
            "location_a": loc_a,
            "location_b": loc_b,
        }

    def is_object_within_bounds(
        self,
        object_name: str,
        bounds_min: List[float],
        bounds_max: List[float],
    ) -> Dict[str, Any]:
        """
        Check whether an object's world location falls within axis-aligned bounds.

        Args:
            object_name: Object name to check.
            bounds_min: Minimum corner [x, y, z].
            bounds_max: Maximum corner [x, y, z].

        Returns:
            Dictionary with within_bounds flag and object location.
        """
        obj = self._get_object_or_raise(object_name)
        self._evaluate_transforms()
        loc = [float(v) for v in obj.matrix_world.translation]

        within = all(bounds_min[i] <= loc[i] <= bounds_max[i] for i in range(3))

        return {
            "object_name": object_name,
            "within_bounds": within,
            "object_location": loc,
        }

    # -- Phase 2: Visual observation methods --------------------------------

    def capture_viewport(
        self,
        width: int = 512,
        height: int = 512,
        jpeg_quality: int = 85,
        use_render_fallback: bool = False,
        agent_id: str = "agent",
        method: str = "auto",
    ) -> Dict[str, Any]:
        """
        Capture the 3D View or a camera render as a base64-encoded JPEG image.

        ``viewport`` redraws the 3D View offscreen with its own view and
        projection matrices and shading, so the image is what the user sees
        without UI or popups, and needs no camera. A Rendered view of an
        external engine such as Cycles exists only on screen, so it is read
        from the screen instead (``viewport_screen``). ``render`` runs
        ``bpy.ops.render.render``; a scene without a camera gets a temporary
        one that is removed afterwards. Neither path falls back to the other.

        Args:
            width: Maximum image width in pixels. A viewport capture, or a render
                through a temporary camera aligned to the 3D View, keeps the view's
                aspect ratio inside width x height; other renders are exactly width x height.
            height: Maximum image height in pixels; see width.
            jpeg_quality: JPEG compression quality (1-100).
            use_render_fallback: Deprecated; same as method="render".
            agent_id: Visible capture feedback label in attached mode; ignored in embedded mode.
            method: "viewport": the 3D View as the user sees it (view angle, shading,
                overlays, Edit Mode, unsaved changes); attached mode only.
                "render": a render through the scene camera, or through a temporary
                camera when the scene has none (aligned to the 3D View when attached,
                framing the visible objects otherwise), removed afterwards.
                "auto" (default): "viewport" when attached, "render" in embedded mode.

        Returns:
            Dictionary with image_base64, width, height, engine, format and
            capture_method: viewport, viewport_screen, render,
            render_temp_camera_view or render_temp_camera_framing.
        """
        method = self._resolve_capture_method(method, use_render_fallback)
        blender_module: Any = bpy
        engine = blender_module.context.scene.render.engine
        if method == "viewport":
            return self._capture_view3d(width, height, jpeg_quality, engine)
        return self._capture_render(width, height, jpeg_quality, engine)

    def set_camera_view(
        self,
        location: List[float],
        rotation_euler: List[float],
        camera_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Place a camera at a world-space location and rotation.

        Both are world space whatever the camera's parent, and the rotation is
        an XYZ Euler whatever the camera's rotation mode: Blender stores it
        relative to the parent, in the camera's own mode (another Euler order,
        QUATERNION or AXIS_ANGLE). The camera's scale is kept, a mirror included.

        The placement goes into the camera's transform. Blender evaluates the
        camera's constraints (Track To, Copy Location, Child Of, ...) and
        drivers on top of it, so the camera can end up elsewhere; the reply
        reports where Blender evaluates it and says when that differs. The
        constraints are left as they are.

        Args:
            location: World-space camera location [x, y, z].
            rotation_euler: World-space camera rotation [rx, ry, rz] in radians, as an XYZ Euler.
            camera_name: Target camera name. Uses active camera when None.

        Returns:
            Dictionary with camera_name, the world-space location and
            rotation_euler where Blender evaluates the camera (the placement
            when the check is skipped), and placement_overridden,
            override_check, constraints_applied and warning (_placement_check).
        """
        cam_obj = self._resolve_camera(camera_name)
        self._evaluate_transforms()
        disablers = self._viewport_disablers(cam_obj)
        before = self._evaluated_pose(cam_obj) if disablers else None
        placement = self._place_in_world(cam_obj, location, rotation_euler)
        stored = self._matrix_without_constraints(cam_obj)
        self._evaluate_transforms()
        report = self._placement_check(cam_obj, placement, stored, disablers, before)
        if report["override_check"] == "done":
            world_location, world_rotation = self._world_placement(
                cam_obj, rotation_euler
            )
        else:
            # Blender did not evaluate the camera after the placement, so
            # matrix_world holds no pose it renders.
            world_location, world_rotation = list(location), list(rotation_euler)
        return {
            "camera_name": cam_obj.name,
            "location": world_location,
            "rotation_euler": world_rotation,
            **report,
        }

    def get_camera_info(self, camera_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a camera's world-space placement and lens properties.

        The pose is where Blender evaluates the camera, its constraints
        included. Blender's viewport skips a camera disabled in viewports
        (_viewport_disablers) unless something it evaluates depends on the
        camera, and the matrix_world of a camera it skips keeps the pose of
        an earlier evaluation: on Blender 5.0.1 a camera moved after it was
        disabled was reported 11 m from where its transform and a render
        through it put it. Unless a child, a constraint or a modifier of an
        object Blender evaluates needs the camera (_evaluated_dependent), its
        pose is composed from its own transform instead
        (_matrix_without_constraints), and the warning says what that leaves
        out (_transform_pose_warning).

        Args:
            camera_name: Target camera name. Uses active camera when None.

        Returns:
            Dictionary with the world-space location and rotation_euler (XYZ
            Euler radians, as set_camera_view takes them); pose_source,
            "evaluated" for the matrix_world Blender evaluated (constraints,
            drivers and keys included) or "transform" for the pose composed
            from the camera's transform; warning, which explains a pose that
            may not be where Blender evaluates the camera and is None
            otherwise; constraints_applied, the constraints that act on it;
            lens, sensor, clip distances and camera type.
        """
        cam_obj = self._resolve_camera(camera_name)
        cam_data = cam_obj.data
        self._evaluate_transforms()
        constraints = self._active_constraints(cam_obj)
        pose_source, warning, composed = "evaluated", None, None
        disablers = self._viewport_disablers(cam_obj)
        if disablers:
            # One scan of the scene serves the camera and its parent.
            needs = self._scene_dependents()
            if self._evaluated_dependent(cam_obj, needs) is None:
                parent = cam_obj.parent
                # Before composing: a parent found needed is evaluated again
                # (_evaluated_dependent), and the pose is composed through it.
                stale_parent = (
                    parent is not None
                    and bool(self._viewport_disablers(parent))
                    and self._evaluated_dependent(parent, needs) is None
                )
                composed = self._matrix_without_constraints(cam_obj)
                if composed is not None:
                    pose_source = "transform"
                warning = self._transform_pose_warning(
                    cam_obj, disablers, constraints, composed is not None, stale_parent
                )
        # Read only now: a camera found needed has just been evaluated again.
        matrix = cam_obj.matrix_world if composed is None else composed
        location, rotation = self._world_placement(cam_obj, matrix=matrix)
        return {
            "camera_name": cam_obj.name,
            "location": location,
            "rotation_euler": rotation,
            "pose_source": pose_source,
            "warning": warning,
            "constraints_applied": constraints,
            "lens": cam_data.lens,
            "sensor_width": cam_data.sensor_width,
            "clip_start": cam_data.clip_start,
            "clip_end": cam_data.clip_end,
            "camera_type": cam_data.type,
        }

    def focus_on_object(
        self,
        object_name: str,
        distance_factor: float = 1.0,
        camera_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Frame an object with a camera.

        The camera moves in front of and above the center of the object's
        world-space bounding box (along the -Y +Z diagonal, Blender's front
        view side) and turns level toward it. It stands where the box's
        bounding sphere fits its frame: its field of view through the scene's
        render aspect ratio, sensor fit and lens shift. An orthographic camera
        gets the ortho_scale that fits the sphere instead. A clip range that
        does not span the sphere is widened, so the object is not clipped
        away. The transform is world space whatever the camera's parent and
        rotation mode.

        As with set_camera_view, the camera's constraints and drivers apply on
        top of this placement and are left as they are; the reply reports
        where Blender evaluates the camera and says when that is not the
        framing placement.

        Args:
            object_name: Object to focus on.
            distance_factor: Multiple of the framing distance (of ortho_scale for an
                orthographic camera): 1.0 fits the object's bounding sphere in the
                frame with 5% to spare, 2.0 shows the object at half that size, and
                below 1.0 crops it.
            camera_name: Target camera name. Uses active camera when None.

        Returns:
            Dictionary with camera_name, object_name, camera_location and
            camera_rotation_euler (world space, where Blender evaluates the
            camera; the framing placement when the check is skipped), look_at
            (the bounding-box center the placement aims at), distance (from
            the placement to look_at), clip_start, clip_end, clip_adjusted
            (whether the clip range was widened), ortho_scale (None unless the
            camera is orthographic), and placement_overridden, override_check,
            constraints_applied and warning (_placement_check).

        Raises:
            ValueError: When the camera's lens shift moves its frame off its
                view axis, so no placement centers the object in it.
        """
        obj = self._get_object_or_raise(object_name)
        cam_obj = self._resolve_camera(camera_name)
        blender_module: Any = bpy
        render = blender_module.context.scene.render
        data = cam_obj.data

        self._evaluate_transforms()
        disablers = self._viewport_disablers(cam_obj)
        before = self._evaluated_pose(cam_obj) if disablers else None
        center, radius = self._bounding_sphere(self._world_bound_box(obj))
        if radius <= 0.0:
            radius = _POINT_RADIUS
        half = min(
            self._frame_half_extents(
                data,
                render.resolution_x * render.pixel_aspect_x,
                render.resolution_y * render.pixel_aspect_y,
            )
        )
        if half <= 0.0:
            raise ValueError(
                f"Camera {cam_obj.name!r} has a lens shift (shift_x {data.shift_x}, "
                f"shift_y {data.shift_y}) that moves its frame off its view axis; "
                "reduce the shift to focus it on an object"
            )
        ortho_scale: Optional[float] = None
        if data.type == "ORTHO":
            # The frame's size is ortho_scale; the distance only has to keep
            # the camera in front of the sphere.
            ortho_scale = radius * _FRAMING_MARGIN * distance_factor / half
            data.ortho_scale = ortho_scale
            distance = 2.0 * radius
        else:
            distance = self._fitting_distance(radius, half) * distance_factor
        location = [c + v * distance for c, v in zip(center, _FOCUS_DIRECTION)]
        rotation = self._look_at_rotation(location, center)
        placement = self._place_in_world(cam_obj, location, rotation)
        stored = self._matrix_without_constraints(cam_obj)
        clip_adjusted = self._clip_around(data, distance, radius)
        self._evaluate_transforms()
        report = self._placement_check(cam_obj, placement, stored, disablers, before)
        if report["override_check"] == "done":
            camera_location, camera_rotation = self._world_placement(cam_obj, rotation)
        else:
            # Blender did not evaluate the camera after the placement, so
            # matrix_world holds no pose it renders.
            camera_location, camera_rotation = list(location), list(rotation)

        return {
            "camera_name": cam_obj.name,
            "object_name": object_name,
            "camera_location": camera_location,
            "camera_rotation_euler": camera_rotation,
            "look_at": center,
            "distance": distance,
            "clip_start": data.clip_start,
            "clip_end": data.clip_end,
            "clip_adjusted": clip_adjusted,
            "ortho_scale": ortho_scale,
            **report,
        }

    def get_viewport_info(self) -> Dict[str, Any]:
        """
        Get active viewport / render settings summary.

        Returns:
            Dictionary with render engine, resolution, film, active camera.
        """
        blender_module: Any = bpy
        scene = blender_module.context.scene
        render = scene.render
        active_cam = scene.camera
        return {
            "render_engine": render.engine,
            "resolution_x": render.resolution_x,
            "resolution_y": render.resolution_y,
            "resolution_percentage": render.resolution_percentage,
            "film_transparent": render.film_transparent,
            "active_camera": active_cam.name if active_cam else None,
        }

    def capture_viewport_sequence(
        self,
        start_frame: int,
        end_frame: int,
        step: int = 1,
        width: int = 512,
        height: int = 512,
        jpeg_quality: int = 85,
        agent_id: str = "agent",
        method: str = "auto",
    ) -> Dict[str, Any]:
        """
        Capture viewport at multiple frames as base64-encoded JPEGs.

        Args:
            start_frame: First frame to capture.
            end_frame: Last frame to capture (inclusive).
            step: Frame step between captures.
            width: Maximum image width, as for capture_blender_viewport.
            height: Maximum image height, as for capture_blender_viewport.
            jpeg_quality: JPEG compression quality.
            agent_id: Visible capture feedback label in attached mode; ignored in embedded mode.
            method: "viewport", "render" or "auto" (default), as for capture_blender_viewport.

        Returns:
            Dictionary with frames list, frame_count, capture_method.
        """
        method = self._resolve_capture_method(method)
        blender_module: Any = bpy
        scene = blender_module.context.scene
        original_frame = scene.frame_current

        frames: List[Dict[str, Any]] = []
        capture_method = method

        try:
            for frame_num in range(start_frame, end_frame + 1, step):
                scene.frame_set(frame_num)
                result = self.capture_viewport(
                    width=width,
                    height=height,
                    jpeg_quality=jpeg_quality,
                    method=method,
                )
                frames.append(
                    {
                        "frame": frame_num,
                        "image_base64": result["image_base64"],
                    }
                )
                capture_method = result.get("capture_method", capture_method)
        finally:
            scene.frame_set(original_frame)

        return {
            "frames": frames,
            "frame_count": len(frames),
            "capture_method": capture_method,
        }

    # -- Phase 3: Scene manipulation methods ---------------------------------

    _PRIMITIVE_OPS: Dict[str, str] = {
        "CUBE": "mesh.primitive_cube_add",
        "SPHERE": "mesh.primitive_uv_sphere_add",
        "CYLINDER": "mesh.primitive_cylinder_add",
        "CONE": "mesh.primitive_cone_add",
        "PLANE": "mesh.primitive_plane_add",
        "TORUS": "mesh.primitive_torus_add",
        "POINT_LIGHT": "object.light_add",
        "SUN_LIGHT": "object.light_add",
        "SPOT_LIGHT": "object.light_add",
        "AREA_LIGHT": "object.light_add",
        "CAMERA": "object.camera_add",
        "EMPTY": "object.empty_add",
    }

    _LIGHT_TYPE_MAP: Dict[str, str] = {
        "POINT_LIGHT": "POINT",
        "SUN_LIGHT": "SUN",
        "SPOT_LIGHT": "SPOT",
        "AREA_LIGHT": "AREA",
    }

    def create_object(
        self,
        object_type: str,
        name: Optional[str] = None,
        location: Optional[List[float]] = None,
        rotation_euler: Optional[List[float]] = None,
        scale: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new object in the Blender scene.

        Supports mesh primitives, lights, cameras, and empties.

        Args:
            object_type: Type key (CUBE, SPHERE, POINT_LIGHT, CAMERA, etc.).
            name: Optional object name.
            location: Initial location [x, y, z].
            rotation_euler: Initial rotation [rx, ry, rz] radians.
            scale: Initial scale [sx, sy, sz].

        Returns:
            Dictionary with name, object_type, location.
        """
        blender_module: Any = bpy
        op_path = self._PRIMITIVE_OPS.get(object_type.upper())
        if op_path is None:
            raise ValueError(
                f"Unsupported object type: {object_type}. "
                f"Valid types: {', '.join(self._PRIMITIVE_OPS.keys())}"
            )

        loc = tuple(location or [0.0, 0.0, 0.0])
        rot = tuple(rotation_euler or [0.0, 0.0, 0.0])

        parts = op_path.split(".")
        op_module = getattr(blender_module.ops, parts[0])
        op_func = getattr(op_module, parts[1])

        light_type = self._LIGHT_TYPE_MAP.get(object_type.upper())
        if light_type:
            op_func(type=light_type, location=loc, rotation=rot)
        else:
            op_func(location=loc, rotation=rot)

        created = blender_module.context.active_object
        if name:
            created.name = name
        if scale:
            created.scale = tuple(scale)

        return {
            "name": created.name,
            "object_type": object_type.upper(),
            "location": list(created.location),
        }

    def delete_object(self, object_name: str) -> Dict[str, Any]:
        """
        Delete an object from the Blender scene by name.

        Args:
            object_name: Name of the object to delete.

        Returns:
            Dictionary with deleted_name.
        """
        blender_module: Any = bpy
        obj = self._get_object_or_raise(object_name)
        blender_module.data.objects.remove(obj, do_unlink=True)
        return {"deleted_name": object_name}

    def set_object_transform(
        self,
        object_name: str,
        location: Optional[List[float]] = None,
        rotation_euler: Optional[List[float]] = None,
        scale: Optional[List[float]] = None,
        rotation_quaternion: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Set an object's own location, rotation, and/or scale.

        These are the values of Blender's Location, Rotation and Scale fields:
        relative to the object's parent, before its constraints. The rotation
        is written in the object's rotation_mode, which is kept: a glTF import
        stores quaternions, and Blender ignores rotation_euler on a QUATERNION
        or AXIS_ANGLE object. Every value is checked before any is written.

        Args:
            object_name: Object to transform.
            location: New location [x, y, z] relative to the parent, or None to keep current.
            rotation_euler: New rotation as XYZ Euler angles [rx, ry, rz] in radians,
                converted to the object's rotation_mode; None to keep current.
            scale: New scale [sx, sy, sz] or None to keep current.
            rotation_quaternion: New rotation as a quaternion [w, x, y, z], normalized
                and converted to the object's rotation_mode; instead of rotation_euler.

        Returns:
            Dictionary with the object's own transform after the change, its
            rotation mode, the rotation as a quaternion, and its world location.

        Raises:
            ValueError: A value is not finite, the quaternion has zero length,
                or both rotation_euler and rotation_quaternion are given.
        """
        obj = self._get_object_or_raise(object_name)
        if rotation_euler is not None and rotation_quaternion is not None:
            raise ValueError("Give either rotation_euler or rotation_quaternion, not both")
        new_location = None if location is None else _finite(location, 3, "location")
        new_scale = None if scale is None else _finite(scale, 3, "scale")
        if rotation_euler is not None:
            _finite(rotation_euler, 3, "rotation_euler")
        if rotation_quaternion is not None:
            quaternion = _finite(rotation_quaternion, 4, "rotation_quaternion")
            if not any(quaternion):
                raise ValueError("rotation_quaternion must not have zero length")
        if new_location is not None:
            obj.location = new_location
        if rotation_euler is not None or rotation_quaternion is not None:
            write_rotation(obj, rotation_euler, rotation_quaternion)
        if new_scale is not None:
            obj.scale = new_scale
        self._evaluate_transforms()

        return {
            "object_name": object_name,
            "location": [float(v) for v in obj.location],
            "rotation_euler": local_rotation_euler(obj),
            "scale": [float(v) for v in obj.scale],
            "rotation_mode": obj.rotation_mode,
            "rotation_quaternion": [float(v) for v in local_rotation_quaternion(obj)],
            "world_location": [float(v) for v in obj.matrix_world.translation],
        }

    def set_object_parent(
        self, child_name: str, parent_name: str, keep_transform: bool = True
    ) -> Dict[str, Any]:
        """
        Parent one object to another.

        With keep_transform, as Blender's Object > Parent > Object (Keep
        Transform), the child stays where it is: its parent inverse matrix
        cancels the parent's current world transform. A child without a
        previous parent keeps its Location, Rotation and Scale values too; one
        moved from another parent gets the values of its world transform
        before constraints, which still apply on top. Without keep_transform
        the parent inverse is cleared: the child's own values become relative
        to the parent, and it moves with the parent's transform.

        Args:
            child_name: Object to parent.
            parent_name: Object to be the parent.
            keep_transform: Keep the child's world transform (default); false clears
                the parent inverse, so the child's own values become relative to the
                parent and it moves.

        Returns:
            Dictionary with child_name, parent_name, keep_transform, the
            previous parent and the child's world location afterwards.

        Raises:
            ValueError: The parent is the child itself or one of its
                descendants, which would make a parent loop (Blender refuses
                such a parent without an error).
        """
        child = self._get_object_or_raise(child_name)
        parent = self._get_object_or_raise(parent_name)
        if parent == child:
            raise ValueError(f"Object '{child_name}' cannot be its own parent")
        ancestor, seen = parent.parent, [parent]
        while ancestor is not None and ancestor not in seen:
            if ancestor == child:
                raise ValueError(
                    f"Cannot parent '{child_name}' to '{parent_name}': '{parent_name}' is a "
                    f"descendant of '{child_name}', so the parenting would make a loop. "
                    f"Clear the parent of '{parent_name}' (or of the object between them) first"
                )
            seen.append(ancestor)
            ancestor = ancestor.parent
        # The world matrices below must hold transforms written by earlier calls.
        self._evaluate_transforms()
        previous = child.parent
        previous_name = previous.name if previous is not None else None
        world = (
            self._matrix_before_constraints(child).copy()
            if keep_transform
            else None
        )
        child.parent = parent
        child.parent_type = "OBJECT"
        if world is None:
            child.matrix_parent_inverse = _mathutils().Matrix.Identity(4)
        else:
            child.matrix_parent_inverse = parent.matrix_world.inverted_safe()
            # Without a previous parent the child's own values already give the
            # world matrix; rewriting them could only round them. A curve
            # parent's matrix can include its path, which only Blender computes
            # when matrix_world is assigned.
            if previous is not None or parent.type == "CURVE":
                child.matrix_world = world
        self._evaluate_transforms()
        return {
            "child_name": child_name,
            "parent_name": parent_name,
            "keep_transform": keep_transform,
            "previous_parent": previous_name,
            "world_location": [float(v) for v in child.matrix_world.translation],
        }

    def clear_object_parent(
        self, object_name: str, keep_transform: bool = True
    ) -> Dict[str, Any]:
        """
        Unparent an object.

        Args:
            object_name: Object to unparent.
            keep_transform: Preserve world transform after unparenting.

        Returns:
            Dictionary with object_name and previous_parent.
        """
        obj = self._get_object_or_raise(object_name)
        prev_parent = obj.parent.name if obj.parent else None
        if keep_transform and obj.parent:
            # matrix_world lags a transform written by an earlier call until
            # the depsgraph is evaluated (background Blender / embedded mode).
            self._evaluate_transforms()
            # Keep rotation and scale as well as translation: the full world
            # matrix, not just its translation, becomes the local transform.
            # Its constraints stay out of it, as they apply on top again.
            world_matrix = self._matrix_before_constraints(obj).copy()
            obj.parent = None
            obj.matrix_world = world_matrix
        else:
            obj.parent = None
        return {"object_name": object_name, "previous_parent": prev_parent}

    def assign_material(
        self,
        object_name: str,
        material_name: Optional[str] = None,
        base_color: Optional[List[float]] = None,
        metallic: float = 0.0,
        roughness: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Create a Principled BSDF material and assign it to an object.

        Args:
            object_name: Object to assign material to.
            material_name: Material name (auto-generated if omitted).
            base_color: Base color RGBA [r, g, b, a].
            metallic: Metallic value 0-1.
            roughness: Roughness value 0-1.

        Returns:
            Dictionary with object_name and material_name.
        """
        blender_module: Any = bpy
        obj = self._get_object_or_raise(object_name)
        if obj.data is None:
            raise ValueError(f"Object '{object_name}' has no data to receive materials")

        mat_name = material_name or f"Material_{object_name}"
        mat = blender_module.data.materials.new(name=mat_name)
        mat.use_nodes = True
        tree = mat.node_tree
        bsdf = None
        for node in tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                bsdf = node
                break

        if bsdf is not None:
            color = base_color or [0.8, 0.8, 0.8, 1.0]
            bsdf.inputs["Base Color"].default_value = tuple(color)
            bsdf.inputs["Metallic"].default_value = metallic
            bsdf.inputs["Roughness"].default_value = roughness

        if hasattr(obj.data, "materials"):
            obj.data.materials.append(mat)
        else:
            raise ValueError(f"Object '{object_name}' does not support materials")

        return {
            "object_name": object_name,
            "material_name": mat.name,
        }

    def add_modifier(
        self,
        object_name: str,
        modifier_type: str,
        modifier_name: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Add a modifier to an object.

        Every key of ``params`` must be a writable setting of the modifier
        type, checked against Blender's RNA: an unknown key is refused with
        the closest setting names, an enum value must be one of its items, a
        number must lie within the setting's limits, and an object, collection
        or other datablock setting takes the datablock's name. When a setting
        fails, the modifier is removed again, so a refused call leaves the
        object as it was. Decimate is refused on a mesh with invalid geometry
        (faces repeating a vertex, duplicate faces or edges), which can make
        Blender's collapse loop forever; clean_mesh repairs such a mesh.

        Args:
            object_name: Object to add modifier to.
            modifier_type: Modifier type (SUBSURF, MIRROR, ARRAY, etc.).
            modifier_name: Modifier display name.
            params: Modifier settings by RNA name, e.g. {"ratio": 0.5} for
                DECIMATE or {"object": "Cutter"} for BOOLEAN.

        Returns:
            Dictionary with object_name, modifier_name, modifier_type and the
            settings as Blender stored them.
        """
        blender_module: Any = bpy
        obj = self._get_object_or_raise(object_name)
        if obj.type not in _MODIFIER_OWNER_TYPES:
            raise ValueError(
                f"Object '{object_name}' is a {obj.type} object, which has no modifier "
                f"stack; modifiers go on {', '.join(sorted(_MODIFIER_OWNER_TYPES))} objects"
            )
        mod_type = modifier_type.upper()
        known_types = self._enum_items(
            getattr(getattr(blender_module, "types", None), "Modifier", None), "type"
        )
        if known_types and mod_type not in known_types:
            raise ValueError(
                f"Unknown modifier type '{modifier_type}'{_did_you_mean(mod_type, known_types)}"
            )
        if mod_type == "DECIMATE" and obj.type == "MESH":
            self._refuse_invalid_geometry(obj, "Decimate")

        mod = obj.modifiers.new(name=modifier_name or modifier_type.title(), type=mod_type)
        if mod is None:
            raise ValueError(
                f"Blender cannot add a {mod_type} modifier to {obj.type} object '{object_name}'"
            )
        try:
            applied = self._set_rna_settings(mod, params or {}, f"{mod_type} modifier")
        except Exception:
            obj.modifiers.remove(mod)
            raise

        return {
            "object_name": object_name,
            "modifier_name": mod.name,
            "modifier_type": mod.type,
            "params": applied,
        }

    def apply_modifier(self, object_name: str, modifier_name: str) -> Dict[str, Any]:
        """
        Apply one modifier to a mesh object's data, baking its result.

        Blender's own Apply: the modifier's result replaces the mesh data and
        the modifier leaves the stack. A modifier below others is applied to
        the base mesh as if the ones above it were not there, and the reply
        warns about it. Mesh data that other objects share is first copied for
        this object, so their geometry does not change. A modifier disabled in
        the viewport is refused, since Blender would drop it unapplied, and
        Decimate is refused on a mesh with invalid geometry, as by
        add_modifier. The object must be in Object Mode.

        Args:
            object_name: Mesh object holding the modifier.
            modifier_name: Name of the modifier to apply.

        Returns:
            Dictionary with the applied modifier, the mesh counts afterwards,
            the remaining modifiers, made_single_user and warnings.
        """
        blender_module: Any = bpy
        obj = self._get_object_or_raise(object_name)
        if obj.type != "MESH":
            raise ValueError(
                f"Object '{object_name}' is a {obj.type} object; only mesh modifiers can be applied"
            )
        self._require_object_mode(obj, "apply a modifier")
        mod = self._modifier_or_raise(obj, modifier_name)
        mod_type = mod.type
        if not mod.show_viewport:
            # Blender 5.0.1 answers FINISHED and drops such a modifier from
            # the stack without applying it.
            raise ValueError(
                f"Modifier '{modifier_name}' of '{object_name}' is disabled in the viewport, "
                "so Blender would remove it without applying it; enable it (show_viewport) "
                "first, or use remove_blender_modifier"
            )
        if mod_type == "DECIMATE":
            self._refuse_invalid_geometry(obj, "Decimate")
        warnings: List[str] = []
        names = [other.name for other in obj.modifiers]
        above = names[: names.index(mod.name)]
        if above:
            warnings.append(
                f"'{modifier_name}' was not first in the stack: Blender applied it to the "
                f"base mesh without the modifiers above it ({', '.join(above)})"
            )
        shared = obj.data.users > 1
        with blender_module.context.temp_override(object=obj, active_object=obj):
            result = _bpy_op(
                f"Failed to apply modifier '{modifier_name}' to '{object_name}'",
                blender_module.ops.object.modifier_apply,
                modifier=mod.name,
                single_user=shared,
            )
        if "FINISHED" not in result:
            raise RuntimeError(
                f"Blender did not apply modifier '{modifier_name}' to '{object_name}' "
                f"({', '.join(sorted(result))})"
            )
        return {
            "object_name": object_name,
            "modifier_name": modifier_name,
            "modifier_type": mod_type,
            **self._mesh_counts(obj.data),
            "remaining_modifiers": [other.name for other in obj.modifiers],
            "made_single_user": shared,
            "warnings": warnings,
        }

    def remove_modifier(self, object_name: str, modifier_name: str) -> Dict[str, Any]:
        """
        Remove one modifier from an object's stack without applying it.

        Args:
            object_name: Object holding the modifier.
            modifier_name: Name of the modifier to remove.

        Returns:
            Dictionary with the removed modifier and the remaining modifiers.
        """
        obj = self._get_object_or_raise(object_name)
        mod = self._modifier_or_raise(obj, modifier_name)
        mod_type = mod.type
        obj.modifiers.remove(mod)
        return {
            "object_name": object_name,
            "modifier_name": modifier_name,
            "modifier_type": mod_type,
            "remaining_modifiers": [other.name for other in obj.modifiers],
        }

    @staticmethod
    def _modifier_or_raise(obj: Any, modifier_name: str) -> Any:
        """The object's modifier of that name, or a ValueError listing its modifiers."""
        modifiers = getattr(obj, "modifiers", None)
        mod = None if modifiers is None else modifiers.get(modifier_name)
        if mod is None:
            names = [other.name for other in modifiers or []]
            raise ValueError(
                f"Object '{obj.name}' has no modifier '{modifier_name}'; "
                + (f"its modifiers are: {', '.join(names)}" if names else "it has none")
            )
        return mod

    @staticmethod
    def _require_object_mode(obj: Any, action: str) -> None:
        """Refuse to change an object's mesh data while it is edited in another mode."""
        mode = getattr(obj, "mode", "OBJECT")
        if mode != "OBJECT":
            raise ValueError(
                f"Object '{obj.name}' is in {mode.replace('_', ' ').title()} Mode; "
                f"switch it to Object Mode to {action}"
            )

    @staticmethod
    def _refuse_invalid_geometry(obj: Any, purpose: str) -> None:
        """Refuse a mesh that Blender's validation would change.

        Faces that repeat a vertex, duplicate faces and duplicate edges are
        invalid topology that Blender's tools assume away: on Blender 5.0.1,
        Decimate's collapse looped forever on such a mesh, pinning the main
        thread. The check validates a throwaway copy, as Blender's C validation
        is both exact and fast (0.17 s for a million faces); leaving the mesh
        itself unchanged keeps the repair an explicit clean_mesh call.

        Args:
            obj: The mesh object.
            purpose: What the mesh is checked for, named in the error.
        """
        blender_module: Any = bpy
        if getattr(obj, "mode", "OBJECT") == "EDIT":
            obj.update_from_editmode()
        probe = obj.data.copy()
        try:
            invalid = bool(probe.validate(verbose=False, clean_customdata=False))
        finally:
            blender_module.data.meshes.remove(probe)
        if invalid:
            raise ValueError(
                f"Mesh of '{obj.name}' has invalid geometry (faces that repeat a vertex, "
                f"duplicate faces or duplicate edges), on which {purpose} can freeze Blender; "
                "run clean_blender_mesh on it first"
            )

    def _set_rna_settings(
        self, struct: Any, settings: Dict[str, Any], what: str
    ) -> Dict[str, Any]:
        """Set settings on a Blender struct after checking each against its RNA.

        Args:
            struct: The struct, such as a modifier.
            settings: Values by RNA property identifier.
            what: The struct as an error names it, e.g. ``DECIMATE modifier``.

        Returns:
            Each setting as Blender stored it, in JSON-friendly form.

        Raises:
            ValueError: For an unknown, reserved or read-only setting, a value
                outside its limits or items, or a datablock that is not found
                or that Blender refuses for the setting.
        """
        properties = struct.bl_rna.properties
        writable = sorted(
            prop.identifier
            for prop in properties
            if not prop.is_readonly
            and prop.identifier not in _RESERVED_MODIFIER_SETTINGS
        )
        applied: Dict[str, Any] = {}
        for key, value in settings.items():
            if key in ("name", "type"):
                raise ValueError(
                    f"Set the {what}'s {key} through modifier_{key}, not params"
                )
            prop = None if key in _RESERVED_MODIFIER_SETTINGS else properties.get(key)
            if prop is None:
                raise ValueError(
                    f"{what} has no setting '{key}'{_did_you_mean(key, writable)}; "
                    f"its settings are: {', '.join(writable)}"
                )
            if prop.is_readonly:
                raise ValueError(f"{what} setting '{key}' is read-only")
            converted = self._rna_value(prop, value, f"{what} setting '{key}'")
            setattr(struct, key, converted)
            stored = getattr(struct, key)
            if prop.type == "POINTER":
                if stored != converted:
                    # Blender drops a datablock its poll function refuses,
                    # such as a camera for a Boolean's object, without an error.
                    raise ValueError(
                        f"Blender refused '{value}' for {what} setting '{key}': it does not "
                        f"take that {prop.fixed_type.identifier}"
                    )
                applied[key] = None if stored is None else stored.name
            else:
                applied[key] = self._rna_echo(stored)
        return applied

    def _rna_value(self, prop: Any, value: Any, what: str) -> Any:
        """The value to assign to an RNA property, checked against its definition.

        Enum items match case-insensitively and an enum flag takes a list;
        a datablock is named; a number must lie within the hard limits that
        Blender would otherwise clamp it to silently.
        """
        kind = prop.type
        if kind == "ENUM":
            items = [item.identifier for item in prop.enum_items]
            if prop.is_enum_flag:
                values = [value] if isinstance(value, str) else value
                if not isinstance(values, (list, tuple, set, frozenset)):
                    raise ValueError(f"{what} takes a list of {', '.join(items)}")
                return {_enum_item(item, items, what) for item in values}
            return _enum_item(value, items, what)
        if kind == "POINTER":
            return self._named_datablock(prop, value, what)
        if kind in ("INT", "FLOAT"):
            length = getattr(prop, "array_length", 0)
            if not length:
                _check_rna_number(prop, value, what)
                return int(value) if kind == "INT" else value
            if not isinstance(value, (list, tuple)) or len(value) != length:
                raise ValueError(f"{what} takes {length} numbers")
            for item in value:
                _check_rna_number(prop, item, what)
            return [int(item) for item in value] if kind == "INT" else list(value)
        return value

    def _named_datablock(self, prop: Any, value: Any, what: str) -> Any:
        """Resolve the name given for a pointer setting to its datablock."""
        if value is None:
            return None
        type_name = prop.fixed_type.identifier
        if not isinstance(value, str):
            raise ValueError(
                f"{what} takes the name of a {type_name}, got {type(value).__name__}"
            )
        rna = prop.fixed_type
        while rna is not None and rna.identifier not in _ID_COLLECTIONS:
            rna = getattr(rna, "base", None)
        if rna is None:
            raise ValueError(f"{what} points to a {type_name}, which cannot be set by name")
        if rna.identifier == "Object":
            # Objects resolve like every other object argument: within the
            # attached scene in attached mode.
            try:
                return self._get_object_or_raise(value)
            except ValueError:
                raise ValueError(f"{what}: object '{value}' not found") from None
        blender_module: Any = bpy
        datablock = getattr(blender_module.data, _ID_COLLECTIONS[rna.identifier]).get(value)
        if datablock is None:
            raise ValueError(f"{what}: {type_name} '{value}' not found")
        return datablock

    @staticmethod
    def _rna_echo(value: Any) -> Any:
        """A stored RNA value in JSON-friendly form: datablocks by name, sets sorted."""
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, (set, frozenset)):
            return sorted(value)
        if hasattr(value, "bl_rna") and hasattr(value, "name"):
            return value.name
        try:
            return [BlenderRuntimeSession._rna_echo(item) for item in value]
        except TypeError:
            return str(value)

    @staticmethod
    def _enum_items(struct_type: Any, identifier: str) -> List[str]:
        """The item identifiers of an enum property of a Blender type, or [] when unknown."""
        try:
            prop = struct_type.bl_rna.properties[identifier]
        except (AttributeError, KeyError, TypeError):
            return []
        return [item.identifier for item in prop.enum_items]

    def set_light_params(
        self,
        light_name: str,
        energy: Optional[float] = None,
        color: Optional[List[float]] = None,
        use_shadow: Optional[bool] = None,
        spot_size: Optional[float] = None,
        spot_blend: Optional[float] = None,
        shadow_soft_size: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Set properties on a light object.

        Args:
            light_name: Light object name.
            energy: Light energy/power.
            color: Light color RGB [r, g, b].
            use_shadow: Enable shadow casting.
            spot_size: Spot cone angle in radians (SPOT only).
            spot_blend: Spot edge blend factor (SPOT only).
            shadow_soft_size: Shadow softness / radius.

        Returns:
            Dictionary with light_name, light_type, energy, color.
        """
        obj = self._get_object_or_raise(light_name)
        if obj.type != "LIGHT":
            raise ValueError(f"Object '{light_name}' is type '{obj.type}', not LIGHT")
        light = obj.data
        if energy is not None:
            light.energy = energy
        if color is not None:
            light.color = tuple(color[:3])
        if use_shadow is not None:
            light.use_shadow = use_shadow
        if spot_size is not None and light.type == "SPOT":
            light.spot_size = spot_size
        if spot_blend is not None and light.type == "SPOT":
            light.spot_blend = spot_blend
        if shadow_soft_size is not None:
            light.shadow_soft_size = shadow_soft_size

        return {
            "light_name": light_name,
            "light_type": light.type,
            "energy": light.energy,
            "color": list(light.color),
        }

    # -- End Phase 3 methods -------------------------------------------------

    # -- Phase 4: File I/O methods -------------------------------------------

    # Version-aware import operator map.
    # Keys: uppercase format name. Values: "module.operator" for bpy.ops.
    _IMPORT_OPS_LEGACY: Dict[str, str] = {
        "OBJ": "import_scene.obj",
        "FBX": "import_scene.fbx",
        "GLTF": "import_scene.gltf",
        "USD": "wm.usd_import",
        "STL": "import_mesh.stl",
        "PLY": "import_mesh.ply",
    }
    _IMPORT_OPS_V4: Dict[str, str] = {
        "OBJ": "wm.obj_import",
        "FBX": "import_scene.fbx",
        "GLTF": "import_scene.gltf",
        "USD": "wm.usd_import",
        "STL": "wm.stl_import",
        "PLY": "wm.ply_import",
    }

    _EXPORT_OPS_LEGACY: Dict[str, str] = {
        "OBJ": "export_scene.obj",
        "FBX": "export_scene.fbx",
        "GLTF": "export_scene.gltf",
        "USD": "wm.usd_export",
        "STL": "export_mesh.stl",
        "PLY": "export_mesh.ply",
    }
    _EXPORT_OPS_V4: Dict[str, str] = {
        "OBJ": "wm.obj_export",
        "FBX": "export_scene.fbx",
        "GLTF": "export_scene.gltf",
        "USD": "wm.usd_export",
        "STL": "wm.stl_export",
        "PLY": "wm.ply_export",
    }

    _SUPPORTED_FORMATS: Tuple[str, ...] = ("OBJ", "FBX", "GLTF", "USD", "STL", "PLY")

    def open_blend_file(
        self, file_path: str, discard_unsaved: bool = False
    ) -> Dict[str, Any]:
        """
        Open a .blend file, replacing the current scene.

        Refuses while the open file has unsaved changes, which opening would
        discard, unless ``discard_unsaved`` is true. Only Blender with a window
        tracks them: headless Blender (embedded mode) reports a session that
        never loaded a file as modified and later edits as unmodified, so it
        opens without the check.

        Args:
            file_path: Absolute path to the .blend file.
            discard_unsaved: Open even though the current file has unsaved changes,
                which are then lost. To keep them, save first with save_blender_file.

        Returns:
            Dict with file_path and object_count.

        Raises:
            PermissionError: If the path is outside the sandbox policy.
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not a .blend file.
            UnsavedChangesError: If the open file has unsaved changes and
                ``discard_unsaved`` is false.
        """
        file_path = self._deny_outside_sandbox(file_path)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        if not file_path.lower().endswith(".blend"):
            raise ValueError(f"Not a .blend file: {file_path}")

        blender_module: Any = bpy
        if (
            not discard_unsaved
            and not getattr(blender_module.app, "background", False)
            and blender_module.data.is_dirty
        ):
            raise UnsavedChangesError(blender_module.data.filepath or None, file_path)
        _bpy_op(
            f"Failed to open .blend file '{file_path}'",
            blender_module.ops.wm.open_mainfile,
            filepath=file_path,
        )

        obj_count = len(blender_module.data.objects)
        return {"file_path": file_path, "object_count": obj_count}

    def save_blend_file(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Save the current scene to a .blend file.

        Saving under another path makes that file the open document, so only a
        .blend path is accepted, as open_blend_file accepts only .blend files.

        Args:
            file_path: Path of the .blend file to save to. None saves in place.

        Returns:
            Dict with file_path.

        Raises:
            PermissionError: If the path is outside the sandbox policy.
            ValueError: If file_path does not end in .blend, or no path is
                given and the file was never saved.
        """
        file_path = self._deny_outside_sandbox(file_path, write=True)
        blender_module: Any = bpy

        if file_path is not None:
            if not file_path.lower().endswith(".blend"):
                raise ValueError(
                    f"Not a .blend file path: {file_path}. save_blend_file writes the "
                    "open document as a .blend file; use export_file for other formats"
                )
            _bpy_op(
                f"Failed to save .blend file to '{file_path}'",
                blender_module.ops.wm.save_as_mainfile,
                filepath=file_path,
            )
            return {"file_path": file_path}

        current = blender_module.data.filepath
        if not current:
            raise ValueError(
                "No file path provided and the file has never been saved. "
                "Pass a file_path argument."
            )
        self._deny_outside_sandbox(current, write=True)
        blender_module.ops.wm.save_mainfile()
        return {"file_path": current}

    def import_file(
        self, file_path: str, file_format: str, scale: float = 1.0
    ) -> Dict[str, Any]:
        """
        Import a file into the scene. Version-aware operator dispatch.

        Args:
            file_path: Absolute path to the file.
            file_format: One of OBJ, FBX, GLTF, USD, STL, PLY.
            scale: Uniform scale applied to the imported objects about the world
                origin (1.0 keeps them as the file has them). OBJ, FBX, STL and
                PLY pass it to Blender's importer as global_scale. For glTF and
                USD the imported root objects' location and scale are multiplied
                by it: Blender's glTF importer has no such option, and Blender
                5.0.1's USD importer ignored its own for a USD file Blender
                wrote. Keys or a Transform
                Cache constraint on such a root's transform take over again at
                the next frame change, and warning names those roots.

        Returns:
            Dict with file_path, file_format, imported_objects, scale and
            warning (None unless something needs attention).

        Raises:
            PermissionError: If the path is outside the sandbox policy.
            FileNotFoundError: If the file does not exist.
            ValueError: If the format is unsupported or scale is not positive.
        """
        file_path = self._deny_outside_sandbox(file_path)
        fmt = self._file_format(file_format)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        if not (math.isfinite(scale) and scale > 0.0):
            raise ValueError(f"scale must be a positive number, not {scale!r}")

        blender_module: Any = bpy
        existing_names = set(blender_module.data.objects.keys())

        op_map = (
            self._IMPORT_OPS_V4 if self._is_blender_4_plus else self._IMPORT_OPS_LEGACY
        )
        op_func = self._operator(op_map[fmt])
        kwargs: Dict[str, Any] = {"filepath": file_path}
        scale_roots = 1.0
        if scale != 1.0:
            prop = self._operator_property(op_func, _IMPORT_SCALE_PROPERTIES.get(fmt, ()))
            if prop is None:
                scale_roots = scale
            else:
                kwargs[prop] = scale
        _bpy_op(f"Failed to import {fmt} file '{file_path}'", op_func, **kwargs)

        new_names = [
            n for n in blender_module.data.objects.keys() if n not in existing_names
        ]
        warning = None
        if scale_roots != 1.0:
            warning = self._scale_imported_roots(new_names, scale_roots)
        return {
            "file_path": file_path,
            "file_format": fmt,
            "imported_objects": new_names,
            "scale": scale,
            "warning": warning,
        }

    def export_file(
        self,
        file_path: str,
        file_format: str,
        selected_only: bool = False,
        object_names: Optional[List[str]] = None,
        collection_name: Optional[str] = None,
        apply_modifiers: bool = True,
        export_animation: bool = True,
        frame_start: Optional[int] = None,
        frame_end: Optional[int] = None,
        gltf_format: Optional[GltfFormat] = None,
        axis_forward: Optional[ExportAxis] = None,
        axis_up: Optional[ExportAxis] = None,
        global_scale: Optional[float] = None,
        usd_up_axis: Optional[UsdUpAxis] = None,
        meters_per_unit: Optional[float] = None,
        root_prim_path: Optional[str] = None,
        overwrite_companion_files: bool = False,
    ) -> Dict[str, Any]:
        """
        Export scene objects to a file. Version-aware operator dispatch.

        The file is written at file_path exactly: its extension must be one the
        format writes, and for GLTF it picks the layout unless gltf_format
        does. A missing folder is created. Edit, Pose and paint modes are left
        for the export, so unsaved mesh edits are included, and entered again
        afterwards. An export that would replace an existing file other than
        file_path (a glTF's .bin, an OBJ's .mtl) that the file at file_path
        does not refer to is refused before anything is written, unless
        overwrite_companion_files. Texture files that GLTF_SEPARATE writes
        beside the .gltf replace files of the same name.

        Args:
            file_path: Path to export to, ending in .obj, .fbx, .glb or .gltf,
                .usd/.usda/.usdc/.usdz, .stl or .ply for the format.
            file_format: One of OBJ, FBX, GLTF, USD, STL, PLY.
            selected_only: Export only the selected objects; an error when none
                is selected.
            object_names: Export exactly these objects (children are not added).
                They are selected for the export and the previous selection is
                restored afterwards. Not with selected_only or collection_name.
            collection_name: Export the objects of this collection and its child
                collections that are in the view layer, the same way.
            apply_modifiers: Export the meshes with their modifiers applied
                (default, for every format). USD always applies them, except
                that Blender writes a last Subdivision Surface as a USD
                subdivision scheme, so false is an error for USD.
            export_animation: Include animation (default): glTF actions, FBX
                baked animation, USD time samples over the scene's frame range.
                Blender's USD exporter leaves animation out unless asked; this
                default includes it. OBJ, STL and PLY have no animation.
            frame_start: First frame of the exported animation (glTF, FBX, USD),
                with frame_end; the scene's frame range is restored afterwards.
                Without them FBX and USD use the scene's frame range and glTF
                each action's whole range.
            frame_end: Last frame of the exported animation, with frame_start.
            gltf_format: GLB (one binary .glb), GLTF_SEPARATE (.gltf plus .bin
                and textures beside it) or GLTF_EMBEDDED (one .gltf with its data
                inline). Default: GLB for .glb and GLTF_SEPARATE for .gltf; it
                must agree with the extension.
            axis_forward: FBX, OBJ, STL and PLY: the axis written as forward, one
                of X, Y, Z, -X, -Y, -Z, together with axis_up on another axis.
                Default: the exporter's (FBX and OBJ -Z forward, Y up; STL and
                PLY Y forward, Z up).
            axis_up: The axis written as up, together with axis_forward.
            global_scale: FBX, OBJ, STL and PLY: scale factor applied to the
                exported data (default 1.0).
            usd_up_axis: USD: the stage's up axis, Y (Blender rotates the scene
                from Z-up to Y-up) or Z (default, Blender's own).
            meters_per_unit: USD: the stage's metersPerUnit (default 1.0); Blender
                scales the root so objects keep their size. Blender 4.4 and
                later.
            root_prim_path: USD: path of the prim everything is exported under
                (default /root); an empty string exports without one.
            overwrite_companion_files: Let the export replace an existing .bin
                (GLTF_SEPARATE) or .mtl (OBJ) beside file_path that the file at
                file_path does not refer to. file_path itself is always replaced.

        Returns:
            Dict with file_path (the file written), file_format, gltf_format
            (GLTF only), files_written (every file the export created or
            changed in file_path's folder and in its new or textures
            subfolders, file_path first), object_count (the objects chosen for
            export; None for the whole scene), exporter_options (the operator
            properties set besides filepath) and warning.

        Raises:
            PermissionError: If the path is outside the sandbox policy.
            ValueError: If the format, extension or an option is invalid, or
                no object is chosen.
            FileExistsError: If a companion file would be replaced.
        """
        file_path = self._deny_outside_sandbox(file_path, write=True)
        fmt = self._file_format(file_format)
        target = self._export_target(file_path)
        variant = self._gltf_variant(target, fmt, gltf_format)
        if sum((bool(selected_only), object_names is not None, collection_name is not None)) > 1:
            raise ValueError(
                "Choose the objects to export with one of selected_only, object_names "
                "or collection_name"
            )
        if (frame_start is None) != (frame_end is None):
            raise ValueError("Pass frame_start and frame_end together")
        if frame_start is not None and frame_end is not None and frame_start > frame_end:
            raise ValueError(
                f"frame_start ({frame_start}) is after frame_end ({frame_end})"
            )

        blender_module: Any = bpy
        view_layer = blender_module.context.view_layer
        to_select: Optional[List[Any]] = None
        object_count: Optional[int] = None
        if selected_only:
            chosen = [o for o in view_layer.objects if o.select_get()]
            if not chosen:
                raise ValueError(
                    f"selected_only is set but no object is selected in view layer "
                    f"{view_layer.name!r}; select objects, or pass object_names"
                )
            object_count = len(chosen)
        elif object_names is not None:
            to_select = self._named_export_objects(object_names)
            object_count = len(to_select)
        elif collection_name is not None:
            to_select = self._collection_export_objects(collection_name)
            object_count = len(to_select)

        op_func = self._operator(self._export_op_path(fmt))
        kwargs = self._export_options(
            op_func,
            fmt,
            target,
            variant,
            subset=selected_only or to_select is not None,
            apply_modifiers=apply_modifiers,
            export_animation=export_animation,
            frame_range=frame_start is not None,
            axes=(axis_forward, axis_up),
            global_scale=global_scale,
            usd_up_axis=usd_up_axis,
            meters_per_unit=meters_per_unit,
            root_prim_path=root_prim_path,
        )

        main = Path(target)
        if not overwrite_companion_files:
            foreign = [
                str(companion)
                for companion in self._companion_files(main, fmt, variant)
                if companion.exists() and not self._refers_to(main, companion.name)
            ]
            if foreign:
                raise FileExistsError(
                    f"Exporting {main.name} would replace {', '.join(foreign)}, which "
                    f"{main.name} does not refer to; export under another name, or pass "
                    "overwrite_companion_files=true to replace it"
                )

        created = self._make_folders(main.parent)
        before = self._folder_state(main.parent)
        frames = (
            None
            if frame_start is None or frame_end is None
            else (frame_start, frame_end)
        )
        try:
            warning = self._run_export(
                f"Failed to export {fmt} file '{target}'",
                op_func,
                kwargs,
                to_select=to_select,
                frame_range=frames,
                embedded_gltf=variant == "GLTF_EMBEDDED",
            )
        except Exception:
            self._remove_empty_folders(created)
            raise
        written = self._written_files(main.parent, before, target)
        if target not in written:
            self._remove_empty_folders(created)
            raise RuntimeError(
                f"Blender's {fmt} exporter reported success but wrote no {main.name}: the "
                "exported objects hold nothing this format can store (OBJ, STL and PLY "
                "need geometry)"
            )
        return {
            "file_path": target,
            "file_format": fmt,
            "gltf_format": variant,
            "files_written": written,
            "object_count": object_count,
            "exporter_options": {k: v for k, v in kwargs.items() if k != "filepath"},
            "warning": warning,
        }

    def _file_format(self, file_format: str) -> str:
        """Return the upper-case format name, or refuse an unsupported one."""
        fmt = file_format.upper()
        if fmt not in self._SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {file_format}. "
                f"Valid: {', '.join(self._SUPPORTED_FORMATS)}"
            )
        return fmt

    @staticmethod
    def _operator(op_path: str) -> Any:
        """Return the ``bpy.ops`` operator named ``module.name``."""
        blender_module: Any = bpy
        module, name = op_path.split(".")
        return getattr(getattr(blender_module.ops, module), name)

    def _export_op_path(self, fmt: str) -> str:
        """Name the version-aware export operator of a format."""
        op_map = (
            self._EXPORT_OPS_V4 if self._is_blender_4_plus else self._EXPORT_OPS_LEGACY
        )
        return op_map[fmt]

    @staticmethod
    def _operator_properties(op_func: Any) -> Dict[str, Any]:
        """Return an operator's properties by identifier."""
        properties = op_func.get_rna_type().properties
        return {name: properties[name] for name in properties.keys()}

    @classmethod
    def _operator_property(
        cls, op_func: Any, candidates: Sequence[str]
    ) -> Optional[str]:
        """Return the first of ``candidates`` that the operator defines, or None."""
        defined = op_func.get_rna_type().properties.keys()
        return next((name for name in candidates if name in defined), None)

    @staticmethod
    def _export_target(file_path: str) -> str:
        """Return the absolute path an export writes, resolving Blender's ``//`` prefix.

        With the sandbox enabled the path is already absolute; with it
        disabled, ``//`` is relative to the open .blend file, as bpy reads it.
        """
        if file_path.startswith("//"):
            blender_module: Any = bpy
            file_path = blender_module.path.abspath(file_path)
        return os.path.abspath(file_path)

    @staticmethod
    def _gltf_variant(
        target: str, fmt: str, gltf_format: Optional[str]
    ) -> Optional[str]:
        """Check the file extension against the format; return the glTF layout.

        Raises:
            ValueError: When the extension is not one the format writes, or
                gltf_format names a layout with another extension.
        """
        name = os.path.basename(target)
        extension = os.path.splitext(name)[1].lower()
        allowed = EXPORT_EXTENSIONS[fmt]
        if extension not in allowed:
            raise ValueError(
                f"{fmt} export writes a file ending in {' or '.join(allowed)}; "
                f"file_path ends in {name!r}"
            )
        if fmt != "GLTF":
            if gltf_format is not None:
                raise ValueError("gltf_format applies to GLTF export only")
            return None
        if gltf_format is None:
            return "GLB" if extension == ".glb" else "GLTF_SEPARATE"
        if gltf_format not in GLTF_FORMATS:
            raise ValueError(
                f"gltf_format must be one of {', '.join(GLTF_FORMATS)}, not {gltf_format!r}"
            )
        expected = ".glb" if gltf_format == "GLB" else ".gltf"
        if extension != expected:
            # Blender's glTF exporter would swap the extension and write another file.
            raise ValueError(
                f"gltf_format {gltf_format} writes a {expected} file; file_path ends in "
                f"{extension}. Use a {expected} path, or the gltf_format of a {extension} file"
            )
        return gltf_format

    def _named_export_objects(self, object_names: List[str]) -> List[Any]:
        """Resolve object_names to objects, in order, without repeats."""
        if not object_names:
            raise ValueError("object_names is empty; name at least one object")
        objects: List[Any] = []
        for name in dict.fromkeys(object_names):
            objects.append(self._get_object_or_raise(name))
        return objects

    @staticmethod
    def _collection_export_objects(collection_name: str) -> List[Any]:
        """Return the objects of a collection and its children in the view layer."""
        blender_module: Any = bpy
        collection = blender_module.data.collections.get(collection_name)
        if collection is None:
            raise ValueError(f"Collection not found: {collection_name}")
        view_layer = blender_module.context.view_layer
        objects = [o for o in collection.all_objects if o.name in view_layer.objects]
        if not objects:
            raise ValueError(
                f"Collection {collection_name!r} holds no object in view layer "
                f"{view_layer.name!r} (it is empty, excluded or in another scene)"
            )
        return objects

    def _export_options(
        self,
        op_func: Any,
        fmt: str,
        target: str,
        variant: Optional[str],
        *,
        subset: bool,
        apply_modifiers: bool,
        export_animation: bool,
        frame_range: bool,
        axes: Tuple[Optional[str], Optional[str]],
        global_scale: Optional[float],
        usd_up_axis: Optional[str],
        meters_per_unit: Optional[float],
        root_prim_path: Optional[str],
    ) -> Dict[str, Any]:
        """Translate export_file's options into the exporter's operator properties.

        An option this format has no equivalent for, or this Blender's exporter
        lacks, is an error, so that nothing asked for is silently left out.
        Only apply_modifiers and export_animation, which default to on, are
        left to the exporter where it defines neither.

        Returns:
            The operator's keyword arguments, filepath included.
        """
        properties = self._operator_properties(op_func)
        version = ".".join(str(part) for part in self._blender_version)

        def require(option: str, candidates: Sequence[str]) -> str:
            name = next((c for c in candidates if c in properties), None)
            if name is None:
                raise ValueError(
                    f"{option} is not supported by Blender {version}'s {fmt} exporter"
                )
            return name

        def only_for(option: str, value: Any, formats: Sequence[str]) -> None:
            if value is not None and fmt not in formats:
                raise ValueError(f"{option} applies to {', '.join(formats)} export only")

        kwargs: Dict[str, Any] = {"filepath": target}
        if subset:
            kwargs[require("selected_only", _EXPORT_SELECTION_PROPERTIES[fmt])] = True
        if variant is not None:
            kwargs[require("gltf_format", ("export_format",))] = variant

        modifiers = self._operator_property(op_func, _EXPORT_MODIFIER_PROPERTIES.get(fmt, ()))
        if modifiers is not None:
            kwargs[modifiers] = apply_modifiers
        elif fmt == "USD" and not apply_modifiers:
            raise ValueError(
                "USD export always applies modifiers: Blender's USD exporter has no "
                "option to leave them out"
            )

        animation = self._operator_property(
            op_func, _EXPORT_ANIMATION_PROPERTIES.get(fmt, ())
        )
        if animation is not None:
            kwargs[animation] = export_animation
        if frame_range:
            if fmt not in _EXPORT_ANIMATION_PROPERTIES:
                raise ValueError("frame_start and frame_end apply to GLTF, FBX and USD export only")
            if not export_animation:
                raise ValueError("frame_start and frame_end need export_animation")
            if fmt == "GLTF":
                kwargs[require("frame_start", ("export_frame_range",))] = True

        forward, up = axes
        if forward is not None or up is not None:
            if fmt not in _EXPORT_AXIS_PROPERTIES:
                raise ValueError(
                    "axis_forward and axis_up apply to FBX, OBJ, STL and PLY export only"
                )
            if forward is None or up is None:
                raise ValueError("Pass axis_forward and axis_up together")
            if forward[-1] == up[-1]:
                raise ValueError(
                    f"axis_forward ({forward}) and axis_up ({up}) must be different axes"
                )
            forward_candidates, up_candidates = _EXPORT_AXIS_PROPERTIES[fmt]
            for value, candidates, option in (
                (forward, forward_candidates, "axis_forward"),
                (up, up_candidates, "axis_up"),
            ):
                name = require(option, candidates)
                kwargs[name] = self._axis_item(properties[name], value)
        if global_scale is not None:
            only_for("global_scale", fmt, ("FBX", "OBJ", "STL", "PLY"))
            if not (math.isfinite(global_scale) and global_scale > 0.0):
                raise ValueError(f"global_scale must be a positive number, not {global_scale!r}")
            kwargs[require("global_scale", ("global_scale",))] = global_scale

        only_for("usd_up_axis", usd_up_axis, ("USD",))
        only_for("meters_per_unit", meters_per_unit, ("USD",))
        only_for("root_prim_path", root_prim_path, ("USD",))
        if usd_up_axis == "Y":
            kwargs[require("usd_up_axis", ("convert_orientation",))] = True
            kwargs[require("usd_up_axis", ("export_global_up_selection",))] = "Y"
            kwargs[require("usd_up_axis", ("export_global_forward_selection",))] = "NEGATIVE_Z"
        elif usd_up_axis == "Z" and "convert_orientation" in properties:
            kwargs["convert_orientation"] = False
        elif usd_up_axis not in (None, "Z"):
            raise ValueError(f"usd_up_axis must be Y or Z, not {usd_up_axis!r}")
        if meters_per_unit is not None:
            if not 0.0001 <= meters_per_unit <= 1000.0:
                raise ValueError(
                    "meters_per_unit must be between 0.0001 and 1000 (Blender's limits), "
                    f"not {meters_per_unit!r}"
                )
            if "convert_scene_units" in properties:
                unit = next(
                    (
                        name
                        for value, name in _USD_SCENE_UNITS.items()
                        if math.isclose(value, meters_per_unit, rel_tol=1e-9)
                    ),
                    "CUSTOM",
                )
                kwargs["convert_scene_units"] = unit
                if unit == "CUSTOM":
                    kwargs[require("meters_per_unit", ("meters_per_unit",))] = meters_per_unit
            else:
                # Before Blender 4.4 the exporter has neither option.
                kwargs[require("meters_per_unit", ("meters_per_unit",))] = meters_per_unit
        if root_prim_path is not None:
            if root_prim_path and not re.fullmatch(
                r"(/[A-Za-z_][A-Za-z0-9_]*)+", root_prim_path
            ):
                raise ValueError(
                    f"root_prim_path must be an absolute USD prim path such as /World, or "
                    f"empty, not {root_prim_path!r}"
                )
            kwargs[require("root_prim_path", ("root_prim_path",))] = root_prim_path
        return kwargs

    @staticmethod
    def _axis_item(prop: Any, axis: str) -> str:
        """Spell an axis as the exporter's enum does: -Z, or NEGATIVE_Z in the C++ exporters."""
        items = {item.identifier for item in prop.enum_items}
        if axis in items:
            return axis
        spelled = f"NEGATIVE_{axis[1:]}" if axis.startswith("-") else axis
        if spelled not in items:
            raise ValueError(f"The exporter offers no axis {axis!r}")
        return spelled

    @staticmethod
    def _companion_files(main: Path, fmt: str, variant: Optional[str]) -> List[Path]:
        """Name the files an export writes beside file_path under fixed names."""
        if fmt == "OBJ":
            return [main.with_suffix(".mtl")]
        if variant == "GLTF_SEPARATE":
            return [main.with_suffix(".bin")]
        return []

    @staticmethod
    def _refers_to(main: Path, companion: str) -> bool:
        """Report whether the existing file at ``main`` refers to a companion file.

        Such a companion belongs to the file being replaced: an OBJ's mtllib
        line, a .gltf's buffer URI.
        """
        if not main.is_file():
            return False
        try:
            if main.suffix.lower() == ".gltf":
                with main.open("r", encoding="utf-8") as handle:
                    document = json.load(handle)
                from urllib.parse import unquote

                return any(
                    unquote(str(buffer.get("uri", ""))) == companion
                    for buffer in document.get("buffers", [])
                    if isinstance(buffer, dict)
                )
            with main.open("r", encoding="utf-8", errors="replace") as handle:
                for number, line in enumerate(handle):
                    if number > 10_000:
                        break
                    if line.startswith("mtllib") and companion in line[6:].strip():
                        return True
        except (OSError, ValueError, AttributeError):
            return False
        return False

    @staticmethod
    def _make_folders(folder: Path) -> List[Path]:
        """Create a missing export folder; return the folders created, deepest first."""
        created: List[Path] = []
        probe = folder
        while not probe.exists() and probe.parent != probe:
            created.append(probe)
            probe = probe.parent
        folder.mkdir(parents=True, exist_ok=True)
        return created

    @staticmethod
    def _remove_empty_folders(folders: List[Path]) -> None:
        """Remove folders created for a failed export, deepest first, if still empty."""
        for folder in folders:
            try:
                folder.rmdir()
            except OSError:
                return

    @staticmethod
    def _folder_state(
        folder: Path, subfolders: Sequence[str] = _EXPORT_TEXTURE_DIRS
    ) -> Dict[str, Tuple[int, int]]:
        """Record each file's modification time and size in a folder.

        Covers the files in the folder and in the named subfolders, at most
        _EXPORT_SCAN_LIMIT entries; subfolders are recorded under their path
        plus a separator.
        """
        state: Dict[str, Tuple[int, int]] = {}
        budget = [_EXPORT_SCAN_LIMIT]

        def scan(directory: Path, depth: int, names: Optional[Sequence[str]]) -> None:
            try:
                entries = list(os.scandir(directory))
            except OSError:
                return
            for entry in entries:
                if budget[0] <= 0:
                    return
                budget[0] -= 1
                try:
                    if entry.is_file(follow_symlinks=False):
                        stat = entry.stat(follow_symlinks=False)
                        state[entry.path] = (stat.st_mtime_ns, stat.st_size)
                    elif entry.is_dir(follow_symlinks=False):
                        state[entry.path + os.sep] = (0, -1)
                        if depth > 0 and (names is None or entry.name in names):
                            scan(Path(entry.path), depth - 1, None)
                except OSError:
                    continue

        scan(folder, 1, subfolders)
        return state

    @classmethod
    def _written_files(
        cls, folder: Path, before: Dict[str, Tuple[int, int]], target: str
    ) -> List[str]:
        """List the files an export created or changed, its main file first."""
        new_folders = [
            Path(path).name
            for path in cls._folder_state(folder)
            if path.endswith(os.sep) and path not in before
        ]
        after = cls._folder_state(folder, (*_EXPORT_TEXTURE_DIRS, *new_folders))
        written = sorted(
            path
            for path, state in after.items()
            if not path.endswith(os.sep) and before.get(path) != state
        )
        if target in written:
            written.remove(target)
            written.insert(0, target)
        return written

    def _run_export(
        self,
        description: str,
        op_func: Any,
        kwargs: Dict[str, Any],
        *,
        to_select: Optional[List[Any]] = None,
        frame_range: Optional[Tuple[int, int]] = None,
        embedded_gltf: bool = False,
    ) -> Optional[str]:
        """Run an exporter in Object Mode with the chosen objects selected.

        Args:
            description: Context for an exporter failure.
            op_func: The export operator.
            kwargs: Its properties.
            to_select: Objects to select for the export instead of the current
                selection, which is restored afterwards.
            frame_range: Scene frame range for the export, restored afterwards.
            embedded_gltf: Allow the glTF add-on's GLTF_EMBEDDED layout for the export.

        Returns:
            A warning when the previous mode could not be entered again, else None.
        """
        blender_module: Any = bpy
        scene = blender_module.context.scene
        warnings: List[str] = []
        with self._object_mode(warnings):
            with self._export_selection(to_select):
                with self._scene_frame_range(scene, frame_range):
                    with self._gltf_embedded_allowed(embedded_gltf):
                        _bpy_op(description, op_func, **kwargs)
        return warnings[0] if warnings else None

    @staticmethod
    @contextmanager
    def _object_mode(warnings: List[str]) -> Iterator[None]:
        """Leave Edit, Pose or a paint mode for the block, then enter it again.

        Leaving Edit Mode writes the edited mesh back, so the export includes
        unsaved edits; Blender's glTF, OBJ and USD exporters switched the
        window to Object Mode themselves and stayed there.

        Args:
            warnings: Receives a note when the mode cannot be entered again.
        """
        blender_module: Any = bpy
        view_layer = blender_module.context.view_layer
        active = view_layer.objects.active
        mode = getattr(active, "mode", "OBJECT") if active is not None else "OBJECT"
        if mode == "OBJECT":
            yield
            return
        _bpy_op(
            f"Failed to leave {mode} mode for the export",
            blender_module.ops.object.mode_set,
            mode="OBJECT",
        )
        try:
            yield
        finally:
            try:
                view_layer.objects.active = active
                blender_module.ops.object.mode_set(mode=mode)
            except Exception as exc:  # The export itself is complete.
                logger.exception("Could not enter %s mode again after the export", mode)
                warnings.append(
                    f"The export left {mode} mode and could not enter it again ({exc}); "
                    "the window is in Object Mode"
                )

    @staticmethod
    @contextmanager
    def _export_selection(objects: Optional[List[Any]]) -> Iterator[None]:
        """Select exactly ``objects`` for the block, then restore the selection.

        Raises:
            ValueError: When an object cannot be selected: hidden, disabled in
                the viewport or outside the view layer, so the exporters would
                leave it out.
        """
        if objects is None:
            yield
            return
        blender_module: Any = bpy
        view_layer = blender_module.context.view_layer
        previous = [o for o in view_layer.objects if o.select_get()]
        active = view_layer.objects.active
        try:
            for obj in previous:
                obj.select_set(False)
            for obj in objects:
                try:
                    obj.select_set(True)
                except RuntimeError as exc:
                    raise ValueError(
                        f"Object {obj.name!r} is not in view layer {view_layer.name!r}, "
                        "so it cannot be exported"
                    ) from exc
                if not obj.select_get():
                    raise ValueError(
                        f"Object {obj.name!r} is hidden or disabled in the viewport, so the "
                        "exporter would leave it out; unhide it or leave it out of the list"
                    )
            yield
        finally:
            for obj in objects:
                try:
                    obj.select_set(False)
                except RuntimeError:
                    pass
            for obj in previous:
                try:
                    obj.select_set(True)
                except RuntimeError:
                    pass
            view_layer.objects.active = active

    @classmethod
    @contextmanager
    def _scene_frame_range(
        cls, scene: Any, frame_range: Optional[Tuple[int, int]]
    ) -> Iterator[None]:
        """Give the scene a frame range for the block, then restore it.

        The preview range is switched off meanwhile: the glTF exporter limits
        animation to it instead when it is on.
        """
        if frame_range is None:
            yield
            return
        saved = (scene.frame_start, scene.frame_end)
        preview = scene.use_preview_range
        cls._set_frame_range(scene, *frame_range)
        scene.use_preview_range = False
        try:
            yield
        finally:
            cls._set_frame_range(scene, *saved)
            scene.use_preview_range = preview

    @staticmethod
    def _set_frame_range(scene: Any, start: int, end: int) -> None:
        """Set a scene's frame range; Blender clamps each end to the other, so order matters."""
        if start > scene.frame_end:
            scene.frame_end = end
            scene.frame_start = start
        else:
            scene.frame_start = start
            scene.frame_end = end

    @staticmethod
    @contextmanager
    def _gltf_embedded_allowed(allow: bool) -> Iterator[None]:
        """Let the glTF exporter write GLTF_EMBEDDED for the block.

        Blender's glTF add-on offers that layout only while its preference
        allow_embedded_format is on; it is switched back off afterwards.
        """
        blender_module: Any = bpy
        addon = blender_module.context.preferences.addons.get("io_scene_gltf2") if allow else None
        preferences = getattr(addon, "preferences", None)
        if preferences is None or getattr(preferences, "allow_embedded_format", True):
            yield
            return
        preferences.allow_embedded_format = True
        try:
            yield
        finally:
            preferences.allow_embedded_format = False

    def _scale_imported_roots(self, names: List[str], factor: float) -> Optional[str]:
        """Scale imported objects about the world origin through their root objects.

        Returns:
            A warning naming roots whose transform is keyed, else None.
        """
        blender_module: Any = bpy
        imported = [blender_module.data.objects[name] for name in names]
        members = {obj.name for obj in imported}
        animated: List[str] = []
        for obj in imported:
            if obj.parent is not None and obj.parent.name in members:
                continue
            obj.location = [c * factor for c in obj.location]
            obj.scale = [c * factor for c in obj.scale]
            anim_data = getattr(obj, "animation_data", None)
            keyed = anim_data is not None and anim_data.action is not None
            # The USD importer animates transforms through this constraint.
            cached = any(c.type == "TRANSFORM_CACHE" for c in getattr(obj, "constraints", ()))
            if keyed or cached:
                animated.append(obj.name)
        if not animated:
            return None
        return (
            f"Scaled the imported roots by {factor}, but the transform of "
            f"{', '.join(repr(n) for n in animated[:5])} is animated: its keys or "
            "Transform Cache constraint take over again at the next frame change"
        )

    def get_file_info(self) -> Dict[str, Any]:
        """
        Return information about the current .blend file.

        Returns:
            Dict with file_path, is_saved, is_dirty, object_count, scene_name.
        """
        blender_module: Any = bpy
        filepath = blender_module.data.filepath or ""
        return {
            "file_path": filepath,
            "is_saved": bool(filepath),
            "is_dirty": blender_module.data.is_dirty,
            "object_count": len(blender_module.data.objects),
            "scene_name": blender_module.context.scene.name,
        }

    # -- End Phase 4 methods -------------------------------------------------

    # -- Phase 5: Animation & Timeline methods --------------------------------

    def set_frame(self, frame: int) -> Dict[str, Any]:
        """
        Set the current animation frame.

        Args:
            frame: Frame number to set.

        Returns:
            Dict with success status and frame number.
        """
        blender_module: Any = bpy
        blender_module.context.scene.frame_set(frame)
        return {"frame": blender_module.context.scene.frame_current}

    def get_frame(self) -> Dict[str, Any]:
        """
        Get the current frame number and animation range.

        Returns:
            Dict with current_frame, frame_start, frame_end, fps.
        """
        blender_module: Any = bpy
        scene = blender_module.context.scene
        return {
            "current_frame": scene.frame_current,
            "frame_start": scene.frame_start,
            "frame_end": scene.frame_end,
            "fps": float(scene.render.fps),
        }

    def set_frame_range(self, frame_start: int, frame_end: int) -> Dict[str, Any]:
        """
        Set the animation start and end frames.

        An unbaked rigid-body cache is fitted to the new range, since Blender
        simulates nothing past the cache's end frame; a baked cache keeps its
        range and the warning says what part of the scene range it leaves out.

        Args:
            frame_start: Start frame of the animation range.
            frame_end: End frame of the animation range.

        Returns:
            Dict with the frame range that was set, rigid_body_cache and warning.
        """
        if frame_start >= frame_end:
            raise ValueError(
                f"frame_start ({frame_start}) must be less than "
                f"frame_end ({frame_end})"
            )
        blender_module: Any = bpy
        scene = blender_module.context.scene
        scene.frame_start = frame_start
        scene.frame_end = frame_end
        # An unbaked rigid-body cache follows the scene range (bodies froze at
        # its old end frame otherwise); a baked one is reported.
        rigid_body_cache, warning = self._sync_rigid_body_cache(scene)
        return {
            "frame_start": scene.frame_start,
            "frame_end": scene.frame_end,
            "rigid_body_cache": rigid_body_cache,
            "warning": warning,
        }

    def play_animation(self, action: str = "play") -> Dict[str, Any]:
        """
        Control animation playback.

        Args:
            action: One of 'play', 'stop', or 'reverse'.

        Returns:
            Dict with action performed and playback state.
        """
        valid_actions = ("play", "stop", "reverse")
        if action not in valid_actions:
            raise ValueError(
                f"Invalid action '{action}'. Must be one of {valid_actions}"
            )
        blender_module: Any = bpy
        screen = blender_module.context.screen

        if action == "stop":
            if screen.is_animation_playing:
                blender_module.ops.screen.animation_cancel(restore_frame=False)
            return {
                "action": "stop",
                "is_playing": False,
            }

        if action == "reverse":
            blender_module.ops.screen.animation_play(reverse=True)
        else:
            blender_module.ops.screen.animation_play()

        return {
            "action": action,
            "is_playing": screen.is_animation_playing,
        }

    def insert_keyframe(
        self,
        object_name: str,
        data_path: str,
        frame: int,
        index: int = -1,
    ) -> Dict[str, Any]:
        """
        Insert a keyframe on an object property at a given frame.

        Args:
            object_name: Name of the object.
            data_path: Property data path (e.g. 'location', 'rotation_euler').
            frame: Frame number for the keyframe.
            index: Array index (-1 for all channels).

        Returns:
            Dict confirming the keyframe insertion.

        Raises:
            RuntimeError: When Blender inserted no keyframe: on an F-curve that
                is locked, sampled or overridden by an F-modifier, or one its
                NLA setup cannot remap a value to.
        """
        obj = self._get_object_or_raise(object_name)
        # Blender reports refused keys by returning False, not by raising. On
        # Blender 5.0.1 it refused a locked F-curve (or one in a locked group),
        # a sampled one and one under an Envelope modifier, and, with "Due to
        # the NLA stack setup", a key on an action at zero influence over an NLA
        # strip or blended by Multiply over a strip that holds 0. With index -1
        # it keys the channels it can and returns True.
        if not obj.keyframe_insert(data_path, index=index, frame=frame):
            raise RuntimeError(
                f"Blender inserted no keyframe on '{object_name}' "
                f"{_channel_name(data_path, index)} at frame {frame}; it refuses "
                "an F-curve that is locked (or in a locked group), sampled or "
                "overridden by an F-modifier, and a key its NLA setup cannot remap "
                "onto the active action, such as an action at zero influence or "
                "one blended over NLA strips in a way that cannot be inverted"
            )
        return {
            "object_name": object_name,
            "data_path": data_path,
            "frame": frame,
        }

    def delete_keyframe(
        self,
        object_name: str,
        data_path: str,
        frame: int,
        index: int = -1,
    ) -> Dict[str, Any]:
        """
        Delete a keyframe from an object property at a given frame.

        Args:
            object_name: Name of the object.
            data_path: Property data path (e.g. 'location', 'rotation_euler').
            frame: Frame number of the keyframe to delete.
            index: Array index (-1 for all channels).

        Returns:
            Dict confirming the keyframe deletion.

        Raises:
            ValueError: When no action animates the object, or no keyframe of
                the property lies on that frame (or its F-curve is locked).
        """
        obj = self._get_object_or_raise(object_name)
        # The same test as get_keyframes' has_animation.
        if action_fcurves(obj.animation_data, self._blender_version) is None:
            raise ValueError(
                f"Object '{object_name}' is not animated, so it has no keyframe "
                "to delete"
            )
        # Blender reports a frame without a deletable key by returning False.
        if not obj.keyframe_delete(data_path, index=index, frame=frame):
            raise ValueError(
                f"Object '{object_name}' has no keyframe on "
                f"{_channel_name(data_path, index)} at frame {frame} to delete, "
                "or its F-curve is locked; get_blender_keyframes lists the keyed "
                "channels"
            )
        return {
            "object_name": object_name,
            "data_path": data_path,
            "frame": frame,
        }

    def get_keyframes(self, object_name: str) -> Dict[str, Any]:
        """
        Get a summary of keyframe data for an object.

        Returns summary data (channel data paths, keyframe counts, frame
        ranges) rather than full curve data to keep responses compact.
        On Blender 4.4 and later an action can animate several objects, and
        only the channels of this object's action slot are listed
        (``action_fcurves``).

        Args:
            object_name: Name of the object.

        Returns:
            Dict with object_name, has_animation (whether an action animates
            the object) and channels list.
        """
        obj = self._get_object_or_raise(object_name)
        fcurves = action_fcurves(obj.animation_data, self._blender_version)
        channels: List[Dict[str, Any]] = []
        for fcurve in fcurves or []:
            kps = fcurve.keyframe_points
            if len(kps) == 0:
                continue
            frames = [int(kp.co[0]) for kp in kps]
            channels.append(
                {
                    "data_path": fcurve.data_path,
                    "array_index": fcurve.array_index,
                    "keyframe_count": len(kps),
                    "frame_range": [min(frames), max(frames)],
                }
            )

        return {
            "object_name": object_name,
            "has_animation": fcurves is not None,
            "channels": channels,
        }

    # -- End Phase 5 methods -------------------------------------------------

    # -- Phase 6: Physics & Simulation methods -------------------------------

    def setup_rigid_body(
        self,
        object_name: str,
        body_type: str = "ACTIVE",
        mass: float = 1.0,
        friction: float = 0.5,
        restitution: float = 0.0,
        collision_shape: str = "CONVEX_HULL",
        linear_damping: float = 0.04,
        angular_damping: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Set up rigid body physics on a Blender object.

        Args:
            object_name: Name of the target object.
            body_type: ACTIVE (dynamic) or PASSIVE (static).
            mass: Mass in kilograms.
            friction: Surface friction coefficient.
            restitution: Bounciness (0-1).
            collision_shape: Collision shape type.
            linear_damping: Linear damping factor.
            angular_damping: Angular damping factor.

        Returns:
            Dict with object_name, body_type, mass, collision_shape.
        """
        blender_module: Any = bpy
        obj = self._get_object_or_raise(object_name)

        scene = blender_module.context.scene
        self._ensure_rigid_body_world(scene)

        blender_module.context.view_layer.objects.active = obj
        obj.select_set(True)
        _bpy_op(
            f"Failed to add rigid body to '{object_name}'",
            blender_module.ops.rigidbody.object_add,
            type=body_type.upper(),
        )

        rb = obj.rigid_body
        rb.mass = mass
        rb.friction = friction
        rb.restitution = restitution
        rb.collision_shape = collision_shape.upper()
        rb.linear_damping = linear_damping
        rb.angular_damping = angular_damping

        return {
            "object_name": object_name,
            "body_type": rb.type,
            "mass": rb.mass,
            "collision_shape": rb.collision_shape,
        }

    def add_force_field(
        self,
        field_type: str,
        strength: float = 1.0,
        location: Optional[List[float]] = None,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Add a force field to the scene.

        Args:
            field_type: Force field type (FORCE, WIND, VORTEX, etc.).
            strength: Field strength.
            location: World-space XYZ location; the world origin when omitted.
            name: Optional name for the field object.

        Returns:
            Dict with name, field_type, strength, location.
        """
        blender_module: Any = bpy
        loc = (0.0, 0.0, 0.0) if location is None else tuple(location)

        blender_module.ops.object.effector_add(type=field_type.upper(), location=loc)
        field_obj = blender_module.context.active_object
        if name:
            field_obj.name = name
        field_obj.field.strength = strength

        return {
            "name": field_obj.name,
            "field_type": field_obj.field.type,
            "strength": field_obj.field.strength,
            "location": list(field_obj.location),
        }

    def get_force_field_info(self, object_name: str) -> Dict[str, Any]:
        """
        Get force field properties from a field object.

        Args:
            object_name: Name of the force field object.

        Returns:
            Dict with object_name, field_type, strength, shape, flow, location.
        """
        obj = self._get_object_or_raise(object_name)
        field = getattr(obj, "field", None)
        if field is None:
            raise ValueError(f"Object has no force field: {object_name}")

        return {
            "object_name": object_name,
            "field_type": field.type,
            "strength": float(field.strength),
            "shape": field.shape,
            "flow": float(field.flow),
            "location": self._world_location(obj),
        }

    def add_rigid_body_constraint(
        self,
        constraint_type: str,
        object1_name: str,
        object2_name: str,
        location: Optional[List[float]] = None,
        disable_collisions: bool = True,
    ) -> Dict[str, Any]:
        """
        Add a rigid body constraint between two objects.

        Args:
            constraint_type: Constraint type (FIXED, POINT, HINGE, etc.).
            object1_name: First constrained object.
            object2_name: Second constrained object.
            location: Optional world-space location for the constraint empty.
            disable_collisions: Disable collisions between constrained objects.

        Returns:
            Dict with constraint_name, constraint_type, object1_name,
            object2_name.
        """
        blender_module: Any = bpy
        obj1 = self._get_object_or_raise(object1_name)
        obj2 = self._get_object_or_raise(object2_name)

        scene = blender_module.context.scene
        self._ensure_rigid_body_world(scene)

        loc = tuple(location or [0.0, 0.0, 0.0])
        blender_module.ops.object.empty_add(location=loc)
        empty = blender_module.context.active_object
        empty.name = f"RBC_{object1_name}_{object2_name}"

        empty.select_set(True)
        blender_module.context.view_layer.objects.active = empty
        blender_module.ops.rigidbody.constraint_add(type=constraint_type.upper())

        rbc = empty.rigid_body_constraint
        rbc.object1 = obj1
        rbc.object2 = obj2
        rbc.disable_collisions = disable_collisions

        return {
            "constraint_name": empty.name,
            "constraint_type": rbc.type,
            "object1_name": object1_name,
            "object2_name": object2_name,
        }

    def get_constraint_info(self, object_name: str) -> Dict[str, Any]:
        """
        Get rigid body constraint details from a constraint empty.

        Args:
            object_name: Name of the constraint empty object.

        Returns:
            Dict with object_name, constraint_type, object1_name,
            object2_name, enabled, disable_collisions.
        """
        obj = self._get_object_or_raise(object_name)
        rbc = getattr(obj, "rigid_body_constraint", None)
        if rbc is None:
            raise ValueError(f"Object has no rigid body constraint: {object_name}")

        return {
            "object_name": object_name,
            "constraint_type": rbc.type,
            "object1_name": rbc.object1.name if rbc.object1 else None,
            "object2_name": rbc.object2.name if rbc.object2 else None,
            "enabled": rbc.enabled,
            "disable_collisions": rbc.disable_collisions,
        }

    @staticmethod
    def _evaluated_world_matrix(obj: Any) -> Any:
        """World matrix after animation, constraints, parents and rigid bodies.

        ``obj.location`` is the authored local value: it misses parent space
        and never moves under a rigid-body simulation, which writes only the
        evaluated matrix.
        """
        blender_module: Any = bpy
        depsgraph = blender_module.context.evaluated_depsgraph_get()
        return obj.evaluated_get(depsgraph).matrix_world

    @classmethod
    def _world_location(cls, obj: Any) -> List[float]:
        return [float(v) for v in cls._evaluated_world_matrix(obj).translation]

    @classmethod
    def _world_pose(cls, obj: Any) -> Tuple[List[float], List[float]]:
        """World location and XYZ-order (or the object's own Euler order) rotation."""
        matrix = cls._evaluated_world_matrix(obj)
        order = obj.rotation_mode if len(obj.rotation_mode) == 3 else "XYZ"
        return (
            [float(v) for v in matrix.translation],
            [float(v) for v in matrix.to_euler(order)],
        )

    def get_physics_state(
        self, object_name: str, step_simulation: bool = True
    ) -> Dict[str, Any]:
        """
        Get an object's world pose and rigid-body state at the current frame.

        An unbaked rigid-body simulation advances one frame at a time from its
        cache start, so after a jump to a frame Blender has not simulated, a
        body still sits where the simulation last left it (its starting pose
        on a fresh cache). With step_simulation the scene steps from the cache
        start to the current frame first, as playback would; frames Blender
        has cached are read, not simulated again. pose_source and
        simulated_for_frame say whether the pose is the simulation's result
        for this frame.

        Args:
            object_name: Name of the object.
            step_simulation: Step an unbaked rigid-body simulation from its
                cache start to the current frame before reading the pose, at
                most frames x bodies = 400 000; false reads the pose as it is.

        Returns:
            Dict with object_name, frame, location, rotation_euler, is_active,
            has_rigid_body, mass, collision_shape, pose_source,
            simulated_for_frame, stepped_frames, world_enabled,
            rigid_body_cache and warning.
        """
        blender_module: Any = bpy
        obj = self._get_object_or_raise(object_name)
        scene = blender_module.context.scene
        rb = getattr(obj, "rigid_body", None)
        world = getattr(scene, "rigidbody_world", None)
        frame = int(scene.frame_current)
        source, simulated, warning, stepped = self._simulated_pose(
            scene, world, obj, rb, frame, step_simulation
        )

        location, rotation = self._world_pose(obj)
        return {
            "object_name": object_name,
            "frame": frame,
            "location": location,
            "rotation_euler": rotation,
            "is_active": rb is not None and rb.type == "ACTIVE",
            "has_rigid_body": rb is not None,
            "mass": float(rb.mass) if rb is not None else None,
            "collision_shape": rb.collision_shape if rb is not None else None,
            "pose_source": source,
            "simulated_for_frame": simulated,
            "stepped_frames": stepped,
            "world_enabled": bool(world.enabled) if world is not None else None,
            "rigid_body_cache": (
                self._cache_state(world.point_cache, scene)
                if world is not None
                else None
            ),
            "warning": warning,
        }

    def _simulated_pose(
        self,
        scene: Any,
        world: Any,
        obj: Any,
        rb: Any,
        frame: int,
        step_simulation: bool,
    ) -> Tuple[str, Optional[bool], Optional[str], int]:
        """Where the object's pose at ``frame`` comes from, stepping the simulation when asked.

        Returns:
            ``(pose_source, simulated_for_frame, warning, stepped_frames)``.
        """
        if rb is None:
            return "evaluated", None, None, 0
        if world is None or not self._in_rigid_body_world(world, obj):
            return (
                "evaluated",
                None,
                (
                    f"'{obj.name}' has rigid-body settings but is not in the scene's "
                    "rigid body world, so nothing simulates it"
                ),
                0,
            )
        if rb.type != "ACTIVE" or rb.kinematic or not rb.enabled:
            # Passive, animated (kinematic) and disabled bodies follow their own
            # transform and animation; the simulation only collides with them.
            return "evaluated", None, None, 0
        cache = world.point_cache
        start, end = int(cache.frame_start), int(cache.frame_end)
        if not world.enabled:
            return (
                "world_disabled",
                False,
                (
                    "The rigid body world is disabled, so the body is at its own "
                    "transform, not simulated; set_blender_rigid_body_world "
                    "enabled=true turns the simulation on"
                ),
                0,
            )
        if frame < start:
            return (
                "before_cache",
                False,
                (
                    f"Frame {frame} is before the rigid-body cache start {start}: "
                    "the body is at its starting pose until the simulation begins"
                ),
                0,
            )
        stepped = 0
        if not cache.is_baked and frame > start and step_simulation:
            stepped = self._step_rigid_bodies(scene, world, frame)
        if frame > end:
            return (
                "past_cache_end",
                False,
                (
                    f"Frame {frame} is past the rigid-body cache end {end}: Blender "
                    f"simulates nothing later and holds the bodies at their frame-{end} "
                    "pose. Extend the cache with set_blender_rigid_body_world "
                    "cache_frame_end (after free_blender_bake when it is baked), or "
                    "with set_blender_frame_range while it is unbaked"
                ),
                stepped,
            )
        if cache.is_baked:
            return "baked", True, None, 0
        if frame == start or stepped:
            return "simulation", True, None, stepped
        if not step_simulation:
            reason = "step_simulation is false"
        else:
            reason = (
                f"stepping {frame - start + 1} frames of "
                f"{self._rigid_body_count(world)} bodies exceeds "
                f"{self._STEP_WORK_BUDGET} body-frames"
            )
        return (
            "unknown",
            None,
            (
                f"The rigid-body cache is not baked and the simulation was not "
                f"stepped to frame {frame} ({reason}): the pose is the simulation's "
                "only if Blender already simulated this frame, and otherwise where "
                "the body was last simulated. Bake with bake_blender_simulation, or "
                "sample with get_blender_object_trajectory"
            ),
            0,
        )

    def get_object_trajectory(
        self,
        object_name: str,
        start_frame: int,
        end_frame: int,
        step: int = 1,
    ) -> Dict[str, Any]:
        """
        Record object world position and rotation across a frame range.

        Samples the evaluated world matrix, so parenting, constraints and
        rigid-body simulation are reflected (``obj.location`` is none of these).

        Args:
            object_name: Name of the object to track.
            start_frame: First frame of trajectory.
            end_frame: Last frame of trajectory (inclusive).
            step: Frame step size.

        Returns:
            Dict with object_name, point_count, points list.
        """
        blender_module: Any = bpy
        obj = self._get_object_or_raise(object_name)
        scene = blender_module.context.scene
        original_frame: int = scene.frame_current
        fps: float = float(scene.render.fps)

        points: List[Dict[str, Any]] = []
        prev_location: Optional[List[float]] = None
        sampled = set(range(start_frame, end_frame + 1, max(step, 1)))
        # An unbaked rigid-body world only advances one frame at a time from
        # its cache start; jumping straight to a sample frame leaves bodies
        # at rest. Walk every frame from there and sample the requested ones.
        walk: range = range(start_frame, end_frame + 1, max(step, 1))
        rigidbody_world = getattr(scene, "rigidbody_world", None)
        if (
            sampled
            and rigidbody_world is not None
            and getattr(rigidbody_world, "enabled", True)
            and not rigidbody_world.point_cache.is_baked
        ):
            cache_start = int(rigidbody_world.point_cache.frame_start)
            walk = range(min(start_frame, cache_start), end_frame + 1)

        try:
            for frame in walk:
                scene.frame_set(frame)
                if frame not in sampled:
                    continue
                loc, rot = self._world_pose(obj)
                time_sec = frame / fps

                velocity: Optional[List[float]] = None
                if prev_location is not None and step > 0:
                    dt = step / fps
                    if dt > 0:
                        velocity = [(loc[i] - prev_location[i]) / dt for i in range(3)]

                points.append(
                    {
                        "frame": frame,
                        "time": time_sec,
                        "location": loc,
                        "rotation_euler": rot,
                        "velocity": velocity,
                    }
                )
                prev_location = loc
        finally:
            scene.frame_set(original_frame)

        return {
            "object_name": object_name,
            "point_count": len(points),
            "points": points,
        }

    def bake_simulation(
        self,
        frame_start: int,
        frame_end: int,
        scope: str = "rigid_body",
    ) -> Dict[str, Any]:
        """
        Bake the physics simulation over a frame range, replacing any earlier bake.

        Blender skips a cache that is already baked, so a second bake used to
        change only the reported range. The earlier bake is freed first and the
        simulation restarts from frame_start with the bodies' current
        transforms. The scene returns to its current frame afterwards.

        Args:
            frame_start: First frame to bake.
            frame_end: Last frame to bake; after frame_start.
            scope: rigid_body bakes the rigid-body world's cache only (stepping
                each frame, then marking the cache baked), leaving cloth and
                other caches as they are; all frees and bakes every point cache
                in the scene, as Blender's Bake All Dynamics does: rigid bodies,
                cloth, soft bodies and dynamic paint over frame_start-frame_end,
                particle systems over the range their emission settings give.

        Returns:
            Dict with scope, frame_start, frame_end and frame_count of the
            rigid-body cache (null without one), rigid_body_cache, caches (each
            cache's owner, kind, range and bake state) and warnings.
        """
        blender_module: Any = bpy
        scope = self._bake_scope(scope)
        self._check_cache_range(int(frame_start), int(frame_end))
        scene = blender_module.context.scene
        warnings: List[str] = []
        original_frame = int(scene.frame_current)
        if scope == "rigid_body":
            world = self._simulating_world(scene, "bake")
            try:
                self._bake_rigid_body_cache(scene, world, frame_start, frame_end)
            finally:
                scene.frame_set(original_frame)
        else:
            caches = self._point_caches(scene)
            if not caches:
                raise ValueError(
                    "Nothing to bake: the scene has no rigid body world, cloth, soft "
                    "body, dynamic paint or particle cache"
                )
            try:
                warnings.extend(
                    self._bake_all_caches(scene, caches, frame_start, frame_end)
                )
            finally:
                scene.frame_set(original_frame)
        return self._bake_report(scene, scope, warnings)

    def free_bake(self, scope: str = "rigid_body") -> Dict[str, Any]:
        """
        Free baked physics simulation data, over the same scope bake_simulation takes.

        A freed cache gets the scene's frame range back (particle systems keep
        the range Blender derives for them), so the simulation is not cut off
        at the last bake's end frame; rigid-body frames cached from the same
        start frame stay valid. Freeing an unbaked cache changes nothing else,
        except that an unbaked rigid-body cache is fitted to the scene range.

        Args:
            scope: rigid_body frees the rigid-body world's cache only; all frees
                every point cache in the scene (rigid bodies, cloth, soft bodies,
                dynamic paint, particles).

        Returns:
            Dict with scope, freed (the caches that were baked), caches,
            rigid_body_cache and warnings.
        """
        blender_module: Any = bpy
        scope = self._bake_scope(scope)
        scene = blender_module.context.scene
        world: Any = getattr(scene, "rigidbody_world", None)
        caches = self._point_caches(scene)
        if scope == "rigid_body":
            if world is None:
                raise ValueError(
                    "No rigid body world to free; use scope 'all' for cloth, soft-body, "
                    "particle or dynamic paint caches"
                )
            caches = caches[:1]
        baked = [(entry, cache) for entry, cache in caches if cache.is_baked]
        freed = [self._cache_entry(entry, cache) for entry, cache in baked]
        if scope == "rigid_body":
            self._free_cache(scene, world.point_cache)
        elif caches:
            with blender_module.context.temp_override(scene=scene):
                blender_module.ops.ptcache.free_bake_all()
            # A bake over part of the scene range would otherwise cut the
            # simulation off at its end frame; particle ranges are derived.
            for entry, cache in baked:
                if entry["kind"] not in ("RIGID_BODY", "PARTICLES"):
                    cache.frame_start = int(scene.frame_start)
                    cache.frame_end = max(int(scene.frame_end), int(scene.frame_start) + 1)
        state, warning = self._sync_rigid_body_cache(scene)
        return {
            "scope": scope,
            "freed": freed,
            "caches": [self._cache_entry(entry, cache) for entry, cache in caches],
            "rigid_body_cache": state,
            "warnings": [warning] if warning else [],
        }

    def _bake_rigid_body_cache(
        self, scene: Any, world: Any, frame_start: int, frame_end: int
    ) -> None:
        """Bake the rigid-body cache alone.

        ``ptcache.bake`` given the rigid-body cache bakes every cache in the
        scene on Blender 5.0.1 (a cloth came out baked over its own 250 frames)
        while ``ptcache.free_bake`` frees only that one. Stepping each frame
        fills the rigid-body cache as playback does, and ``bake_from_cache``
        marks only that cache baked. Other simulations step along, as during
        playback, without being baked.
        """
        blender_module: Any = bpy
        cache = world.point_cache
        if cache.is_baked:
            self._free_cache(scene, cache)
        cache.frame_start = frame_start
        cache.frame_end = frame_end
        self._invalidate_rigid_body_cache(world)
        for frame in range(frame_start, frame_end + 1):
            scene.frame_set(frame)
        with blender_module.context.temp_override(scene=scene, point_cache=cache):
            blender_module.ops.ptcache.bake_from_cache()
        if not cache.is_baked:
            raise RuntimeError(
                "Blender did not mark the rigid-body cache baked after stepping "
                f"frames {frame_start}-{frame_end}"
            )

    def _bake_all_caches(
        self,
        scene: Any,
        caches: List[Tuple[Dict[str, Any], Any]],
        frame_start: int,
        frame_end: int,
    ) -> List[str]:
        """Free and bake every point cache, as Blender's Bake All Dynamics.

        Bake All skips caches that are already baked, so all are freed first.
        Particle systems keep the range Blender derives from their emission
        settings and the scene range. A disabled rigid body world stores no
        simulation, so its cache is left unbaked.

        Returns:
            Warnings for the reply.
        """
        blender_module: Any = bpy
        warnings: List[str] = []
        world = getattr(scene, "rigidbody_world", None)
        with blender_module.context.temp_override(scene=scene):
            blender_module.ops.ptcache.free_bake_all()
        for entry, cache in caches:
            if entry["kind"] != "PARTICLES":
                cache.frame_start = frame_start
                cache.frame_end = frame_end
        if world is not None and world.enabled:
            self._invalidate_rigid_body_cache(world)
        scene.frame_set(frame_start)
        with blender_module.context.temp_override(scene=scene):
            blender_module.ops.ptcache.bake_all(bake=True)
        if world is not None and not world.enabled:
            self._free_cache(scene, world.point_cache)
            warnings.append(
                "The rigid body world is disabled, so its cache was left unbaked; "
                "enable it with set_blender_rigid_body_world enabled=true and bake again"
            )
        return warnings

    def _bake_report(self, scene: Any, scope: str, warnings: List[str]) -> Dict[str, Any]:
        """The reply of bake_simulation: the range the caches actually hold."""
        world = getattr(scene, "rigidbody_world", None)
        state = (
            self._cache_state(world.point_cache, scene) if world is not None else None
        )
        if state is not None and not state["covers_scene_range"]:
            warnings.append(
                f"The rigid-body bake covers frames {state['frame_start']}-"
                f"{state['frame_end']} of the scene range {scene.frame_start}-"
                f"{scene.frame_end}: bodies hold their pose outside it"
            )
        if scope == "all":
            caches = self._point_caches(scene)
        else:
            caches = self._point_caches(scene)[:1]
        return {
            "scope": scope,
            "frame_start": state["frame_start"] if state else None,
            "frame_end": state["frame_end"] if state else None,
            "frame_count": (
                state["frame_end"] - state["frame_start"] + 1 if state else None
            ),
            "rigid_body_cache": state,
            "caches": [self._cache_entry(entry, cache) for entry, cache in caches],
            "warnings": warnings,
        }

    # -- Rigid-body world and point caches -----------------------------------

    #: What bake_simulation and free_bake act on: the rigid-body world's cache
    #: only, or every point cache in the scene, as Blender's Bake All Dynamics.
    _BAKE_SCOPES = ("rigid_body", "all")

    #: Modifier types whose ``point_cache`` Blender's Bake All Dynamics bakes.
    _CACHED_MODIFIER_TYPES = frozenset({"CLOTH", "SOFT_BODY"})

    #: The frame range a point cache accepts (PointCache RNA, Blender 4.2-5.0).
    _CACHE_FRAME_MIN = -1048574
    _CACHE_FRAME_MAX = 1048574

    #: Most work get_physics_state steps an unbaked simulation through, in
    #: body-frames (frames x rigid bodies): on Blender 5.0.1 a frame cost about
    #: 0.3 ms plus 20-30 microseconds per body in contact with a ground, so this
    #: stays near 10 s, a third of the attached-mode timeout.
    _STEP_WORK_BUDGET = 400_000

    @staticmethod
    def _in_rigid_body_world(world: Any, obj: Any) -> bool:
        """Whether the world's collection, which is what Blender simulates, holds ``obj``."""
        collection = world.collection
        return collection is not None and any(
            member == obj for member in collection.all_objects
        )

    @staticmethod
    def _rigid_body_count(world: Any) -> int:
        collection = world.collection
        return len(collection.all_objects) if collection is not None else 0

    def _step_rigid_bodies(self, scene: Any, world: Any, frame: int) -> int:
        """Step an unbaked rigid-body simulation from its cache start up to ``frame``.

        Blender advances an unbaked simulation only from one frame to the next,
        so this visits every frame from the cache start, as playback would.
        Frames already cached are read, not simulated again. Past the cache
        end nothing more is simulated; the walk stops there and ends on
        ``frame``. Nothing is stepped when the walk would exceed
        ``_STEP_WORK_BUDGET``.

        Returns:
            The number of frames visited, 0 when the walk was over budget.
        """
        cache = world.point_cache
        start = int(cache.frame_start)
        last = min(frame, int(cache.frame_end))
        frames = last - start + 1
        if frames * max(self._rigid_body_count(world), 1) > self._STEP_WORK_BUDGET:
            return 0
        for step in range(start, last + 1):
            scene.frame_set(step)
        if last != frame:
            scene.frame_set(frame)
        return frames

    @staticmethod
    def _cache_state(cache: Any, scene: Any) -> Dict[str, Any]:
        """The range and bake state of a rigid-body point cache."""
        start, end = int(cache.frame_start), int(cache.frame_end)
        return {
            "frame_start": start,
            "frame_end": end,
            "is_baked": bool(cache.is_baked),
            "is_outdated": bool(cache.is_outdated),
            "info": str(cache.info),
            "covers_scene_range": start <= int(scene.frame_start)
            and end >= int(scene.frame_end),
        }

    @staticmethod
    def _invalidate_rigid_body_cache(world: Any) -> None:
        """Mark an unbaked rigid-body cache outdated, so the simulation restarts at its start.

        Writing a world setting runs Blender's rigid-body reset even when the
        value stays the same, and Python has no other handle on it. Blender then
        drops the cached frames and rebuilds the physics world from the bodies'
        own transforms when the scene next steps from the cache start to the
        frame after. Without it the frames of an earlier simulation stay
        readable: on Blender 5.0.1, after a freed bake of frames 1-200, a bake
        of 10-50 read the old frames simulated from frame 1.
        """
        world.time_scale = world.time_scale

    def _set_cache_range(self, world: Any, frame_start: int, frame_end: int) -> None:
        """Set the rigid-body cache range; a new start frame invalidates the cache.

        Frames cached from the same start stay valid when only the end moves.
        """
        cache = world.point_cache
        moved_start = int(cache.frame_start) != frame_start
        cache.frame_start = frame_start
        cache.frame_end = frame_end
        if moved_start:
            self._invalidate_rigid_body_cache(world)

    def _sync_rigid_body_cache(
        self, scene: Any
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Fit an unbaked rigid-body cache to the scene's frame range.

        Blender simulates rigid bodies only inside the cache's range and holds
        them at its last frame after it. ``rigidbody.world_add`` makes the range
        1-250 whatever the scene's, and changing the scene range leaves it
        alone, so bodies froze at frame 250 of a 1000-frame scene. Call this
        wherever the scene range changes or the cache is created or freed
        (set_frame_range, free_bake, world creation). A baked cache keeps its
        range, since a new range would not change the bake; the warning says
        what it leaves out.

        Args:
            scene: The scene whose rigid-body world to sync.

        Returns:
            The cache state (None without a rigid body world) and a warning,
            or None.
        """
        world = getattr(scene, "rigidbody_world", None)
        if world is None:
            return None, None
        cache = world.point_cache
        start, end = int(scene.frame_start), int(scene.frame_end)
        warning = None
        if not cache.is_baked:
            self._set_cache_range(world, start, max(end, start + 1))
        elif int(cache.frame_start) > start or int(cache.frame_end) < end:
            warning = (
                f"The rigid-body cache is baked for frames {cache.frame_start}-"
                f"{cache.frame_end}, which leaves out part of the scene range "
                f"{start}-{end}: bodies hold their pose outside it. Bake again over "
                "the new range with bake_blender_simulation"
            )
        return self._cache_state(cache, scene), warning

    def _ensure_rigid_body_world(self, scene: Any) -> Tuple[Any, bool]:
        """Return the scene's rigid body world, adding one fitted to the scene range.

        Returns:
            The world and whether it was added.
        """
        world = scene.rigidbody_world
        if world is not None:
            return world, False
        blender_module: Any = bpy
        _bpy_op(
            "Failed to add a rigid body world",
            blender_module.ops.rigidbody.world_add,
        )
        world = scene.rigidbody_world
        self._sync_rigid_body_cache(scene)
        return world, True

    def _point_caches(self, scene: Any) -> List[Tuple[Dict[str, Any], Any]]:
        """Every point cache Blender's Bake All Dynamics covers, with a description.

        The rigid-body world's, then per object those of cloth and soft-body
        modifiers, dynamic paint canvas surfaces and particle systems. Fluid
        domains bake through their own cache, not a point cache.

        Returns:
            ``(entry, cache)`` pairs; ``entry`` names the owner and kind.
        """
        found: List[Tuple[Dict[str, Any], Any]] = []
        world = getattr(scene, "rigidbody_world", None)
        if world is not None:
            found.append(
                (
                    {"owner": scene.name, "kind": "RIGID_BODY", "name": "RigidBodyWorld"},
                    world.point_cache,
                )
            )
        for obj in scene.objects:
            for modifier in getattr(obj, "modifiers", ()):
                if modifier.type in self._CACHED_MODIFIER_TYPES:
                    found.append(
                        (
                            {"owner": obj.name, "kind": modifier.type, "name": modifier.name},
                            modifier.point_cache,
                        )
                    )
                elif modifier.type == "DYNAMIC_PAINT":
                    canvas = getattr(modifier, "canvas_settings", None)
                    for surface in getattr(canvas, "canvas_surfaces", ()) or ():
                        found.append(
                            (
                                {"owner": obj.name, "kind": "DYNAMIC_PAINT", "name": surface.name},
                                surface.point_cache,
                            )
                        )
            for system in getattr(obj, "particle_systems", ()):
                found.append(
                    (
                        {"owner": obj.name, "kind": "PARTICLES", "name": system.name},
                        system.point_cache,
                    )
                )
        return found

    @staticmethod
    def _cache_entry(entry: Dict[str, Any], cache: Any) -> Dict[str, Any]:
        return {
            **entry,
            "frame_start": int(cache.frame_start),
            "frame_end": int(cache.frame_end),
            "is_baked": bool(cache.is_baked),
        }

    @staticmethod
    def _free_cache(scene: Any, cache: Any) -> None:
        """Free one point cache's bake (a no-op on an unbaked cache)."""
        blender_module: Any = bpy
        # Blender 4.0 removed the positional context-override dict.
        with blender_module.context.temp_override(scene=scene, point_cache=cache):
            blender_module.ops.ptcache.free_bake()

    @classmethod
    def _bake_scope(cls, scope: str) -> str:
        if scope not in cls._BAKE_SCOPES:
            raise ValueError(
                f"Invalid scope '{scope}'. Must be one of {cls._BAKE_SCOPES}"
            )
        return scope

    def _simulating_world(self, scene: Any, action: str) -> Any:
        """The scene's rigid body world, refusing ``action`` without one or while it is disabled."""
        world = getattr(scene, "rigidbody_world", None)
        if world is None:
            raise ValueError(
                f"No rigid body world to {action}: setup_blender_rigid_body adds one "
                "with its first body, set_blender_rigid_body_world adds an empty one. "
                "Use scope 'all' for cloth, soft-body, particle or dynamic paint caches"
            )
        if not world.enabled:
            raise ValueError(
                f"The rigid body world is disabled, so a {action} would store no "
                "simulation; enable it with set_blender_rigid_body_world enabled=true"
            )
        return world

    def get_rigid_body_world(self) -> Dict[str, Any]:
        """
        Get the scene's rigid body world: its settings, bodies and cache.

        Gravity is the scene's, which cloth, soft bodies and particles share.
        The cache's range bounds what Blender simulates: past its end, bodies
        hold their pose. ``warnings`` names what makes the simulation differ
        from what playback suggests (a disabled world, a cache that leaves out
        part of the scene range, a baked cache that settings no longer change).

        Returns:
            Dict with exists, enabled, gravity, use_gravity, substeps_per_frame,
            solver_iterations, time_scale, use_split_impulse, collection,
            body_count, constraint_count, rigid_body_cache, scene_frame_start,
            scene_frame_end and warnings.
        """
        blender_module: Any = bpy
        return self._world_report(blender_module.context.scene)

    def set_rigid_body_world(
        self,
        enabled: Optional[bool] = None,
        gravity: Optional[List[float]] = None,
        use_gravity: Optional[bool] = None,
        substeps_per_frame: Optional[int] = None,
        solver_iterations: Optional[int] = None,
        time_scale: Optional[float] = None,
        use_split_impulse: Optional[bool] = None,
        cache_frame_start: Optional[int] = None,
        cache_frame_end: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Change the scene's rigid body world, adding one when there is none.

        Only the given settings change. A new world's cache covers the scene's
        frame range. An unbaked cache restarts from its start frame after a
        change; a baked one keeps playing its bake, so changed settings apply
        only after free_blender_bake and a new bake, and its range cannot
        change until it is freed.

        Args:
            enabled: Run the simulation; a disabled world leaves bodies at their
                own transforms.
            gravity: Scene gravity [x, y, z] in m/s^2, shared with cloth, soft
                bodies and particles; Blender's default is [0, 0, -9.81].
            use_gravity: Apply the scene gravity.
            substeps_per_frame: Simulation steps per frame, 1-32767.
            solver_iterations: Constraint solver iterations per step, 1-1000.
            time_scale: Simulation speed relative to the scene's frame rate, 0-100.
            use_split_impulse: Reduce extra velocity from penetration recovery.
            cache_frame_start: First simulated frame.
            cache_frame_end: Last simulated frame; after cache_frame_start.

        Returns:
            The world as get_rigid_body_world reports it, with created.
        """
        blender_module: Any = bpy
        scene = blender_module.context.scene
        world = getattr(scene, "rigidbody_world", None)
        range_requested = cache_frame_start is not None or cache_frame_end is not None
        if world is None:
            current = (int(scene.frame_start), max(int(scene.frame_end), int(scene.frame_start) + 1))
            baked = False
        else:
            current = (int(world.point_cache.frame_start), int(world.point_cache.frame_end))
            baked = bool(world.point_cache.is_baked)
        new_range = (
            current[0] if cache_frame_start is None else int(cache_frame_start),
            current[1] if cache_frame_end is None else int(cache_frame_end),
        )
        if range_requested:
            self._check_cache_range(*new_range, label="cache_frame")
            if baked and new_range != current:
                raise ValueError(
                    f"The rigid-body cache is baked for frames {current[0]}-{current[1]}; "
                    "free it with free_blender_bake before changing its range"
                )
        if gravity is not None:
            gravity = [float(v) for v in gravity]
            if len(gravity) != 3 or not all(math.isfinite(v) for v in gravity):
                raise ValueError("gravity must be three finite numbers")

        world, created = self._ensure_rigid_body_world(scene)
        try:
            settings = {
                "enabled": enabled,
                "substeps_per_frame": substeps_per_frame,
                "solver_iterations": solver_iterations,
                "time_scale": time_scale,
                "use_split_impulse": use_split_impulse,
            }
            changed = False
            for name, value in settings.items():
                if value is not None:
                    setattr(world, name, value)
                    changed = True
            if gravity is not None:
                scene.gravity = gravity
                changed = True
            if use_gravity is not None:
                scene.use_gravity = use_gravity
                changed = True
            if range_requested and not baked:
                self._set_cache_range(world, *new_range)
            if changed and not world.point_cache.is_baked:
                self._invalidate_rigid_body_cache(world)
        except Exception:
            if created:
                blender_module.ops.rigidbody.world_remove()
            raise
        report = self._world_report(scene)
        report["created"] = created
        if changed and world.point_cache.is_baked:
            report["warnings"].append(
                "The rigid-body cache is baked, so playback still reads the bake: the "
                "changed settings apply after free_blender_bake and a new bake"
            )
        return report

    def _check_cache_range(self, frame_start: int, frame_end: int, label: str = "frame") -> None:
        """Refuse a range a point cache cannot hold, or one that is empty or inverted."""
        if frame_start >= frame_end:
            raise ValueError(
                f"{label}_start ({frame_start}) must be less than {label}_end ({frame_end})"
            )
        if frame_start < self._CACHE_FRAME_MIN or frame_end > self._CACHE_FRAME_MAX:
            raise ValueError(
                f"A point cache covers frames {self._CACHE_FRAME_MIN} to "
                f"{self._CACHE_FRAME_MAX}; got {frame_start}-{frame_end}"
            )

    def _world_report(self, scene: Any) -> Dict[str, Any]:
        world = getattr(scene, "rigidbody_world", None)
        report: Dict[str, Any] = {
            "exists": world is not None,
            "gravity": [float(v) for v in scene.gravity],
            "use_gravity": bool(scene.use_gravity),
            "scene_frame_start": int(scene.frame_start),
            "scene_frame_end": int(scene.frame_end),
            "warnings": [],
        }
        if world is None:
            report["warnings"].append(
                "The scene has no rigid body world: setup_blender_rigid_body adds one "
                "with its first body, set_blender_rigid_body_world adds an empty one"
            )
            return report
        cache = world.point_cache
        state = self._cache_state(cache, scene)
        constraints = world.constraints
        report.update(
            enabled=bool(world.enabled),
            substeps_per_frame=int(world.substeps_per_frame),
            solver_iterations=int(world.solver_iterations),
            time_scale=float(world.time_scale),
            use_split_impulse=bool(world.use_split_impulse),
            collection=world.collection.name if world.collection is not None else None,
            body_count=self._rigid_body_count(world),
            constraint_count=(
                len(constraints.all_objects) if constraints is not None else 0
            ),
            rigid_body_cache=state,
        )
        if not world.enabled:
            report["warnings"].append(
                "The rigid body world is disabled: nothing simulates and bakes are refused"
            )
        if not state["covers_scene_range"]:
            report["warnings"].append(
                f"The rigid-body cache covers frames {state['frame_start']}-"
                f"{state['frame_end']}, not the whole scene range "
                f"{report['scene_frame_start']}-{report['scene_frame_end']}: bodies hold "
                "their pose outside it"
            )
        return report

    # -- End Phase 6 methods -------------------------------------------------

    # -- Scripting & mesh-from-data methods ----------------------------------

    def execute_script(
        self,
        script: str,
        timeout: Optional[float] = None,
        max_output_chars: int = DEFAULT_SCRIPT_OUTPUT_CHARS,
    ) -> Dict[str, Any]:
        """
        Execute arbitrary Python code with access to bpy.

        The script runs via ``exec()`` in a namespace that already contains
        ``bpy`` and the standard builtins.  If the script assigns a value to
        ``__result__``, its ``repr()`` is returned in *return_value*.

        Captured ``print()`` output is returned in *output*, cut to its first
        ``max_output_chars`` characters, and further when those encode to
        more than MAX_SCRIPT_OUTPUT_JSON_BYTES of JSON (non-ASCII or control
        characters). A cut is never silent: ``output_truncated`` is true and
        ``output_total_chars`` gives the full length, so a truncated JSON dump
        is not mistaken for a script bug.

        Whatever the script raises is reported in *error*, with the output
        printed before it; that includes ``SystemExit`` from ``sys.exit()``,
        which ends only the script, never Blender.

        With a ``timeout`` the script runs on a worker thread and this call
        returns a structured error once the budget is spent. Python offers
        no safe way to kill that thread, so the script keeps running in the
        background until it finishes on its own; it just no longer blocks
        the MCP server. Its output is discarded. Without a timeout the
        script runs inline.

        Args:
            script: Python source code to execute.
            timeout: Maximum wall-clock seconds to wait for the script;
                ``None`` waits for it to finish.
            max_output_chars: Most characters of printed output to return, 1 to 100000
                (default 32000). Longer output is cut to its first max_output_chars
                characters with output_truncated=true and output_total_chars set.
                Output that JSON escapes heavily (non-ASCII text, control characters)
                is cut earlier, at 300000 bytes of JSON, so the prefix always arrives.
                Write larger results to a file and return its path instead.

        Returns:
            Dict with output, output_truncated, output_total_chars,
            return_value, duration_seconds, and optionally error. A timeout
            adds ``timed_out: True``.

        Raises:
            ValueError: If max_output_chars is not an integer from 1 to
                MAX_SCRIPT_OUTPUT_CHARS.
        """
        import contextlib
        import threading
        import time

        if (
            isinstance(max_output_chars, bool)
            or not isinstance(max_output_chars, int)
            or not 1 <= max_output_chars <= MAX_SCRIPT_OUTPUT_CHARS
        ):
            raise ValueError(
                f"max_output_chars must be an integer from 1 to {MAX_SCRIPT_OUTPUT_CHARS}"
            )

        blender_module: Any = bpy

        stdout_capture = io.StringIO()
        namespace: Dict[str, Any] = {
            "bpy": blender_module,
            "__builtins__": __builtins__,
            "__result__": None,
        }

        def _captured_output() -> Dict[str, Any]:
            raw_output = stdout_capture.getvalue()
            output = script_output_prefix(raw_output, max_output_chars)
            return {
                "output": output or None,
                "output_truncated": len(output) < len(raw_output),
                "output_total_chars": len(raw_output),
            }

        def _run() -> Dict[str, Any]:
            start = time.monotonic()
            try:
                with contextlib.redirect_stdout(stdout_capture):
                    exec(script, namespace)  # noqa: S102
            except BaseException as exc:
                # SystemExit and KeyboardInterrupt are the script's errors too:
                # raised past here, sys.exit() would end Blender (or the MCP
                # server in embedded mode), not the script.
                return {
                    **_captured_output(),
                    "return_value": None,
                    "duration_seconds": round(time.monotonic() - start, 4),
                    "error": _script_error(exc),
                }
            elapsed = time.monotonic() - start
            return_value: Optional[str] = None
            if namespace.get("__result__") is not None:
                return_value = repr(namespace["__result__"])
            return {
                **_captured_output(),
                "return_value": return_value,
                "duration_seconds": round(elapsed, 4),
            }

        if timeout is None:
            return _run()

        outcome: List[Dict[str, Any]] = []
        worker = threading.Thread(
            target=lambda: outcome.append(_run()),
            name="blender-execute-script",
            daemon=True,
        )
        worker.start()
        worker.join(timeout)
        if outcome:
            return outcome[0]
        self.logger.warning(
            "Blender script exceeded its %.1fs timeout; the worker thread keeps "
            "running until the script returns on its own.",
            timeout,
        )
        return {
            "output": None,
            "output_truncated": False,
            "output_total_chars": 0,
            "return_value": None,
            "duration_seconds": round(float(timeout), 4),
            "timed_out": True,
            "error": (
                f"TimeoutError: script did not finish within {timeout:g}s. "
                "It is still running on a background thread and cannot be "
                "stopped; its output is discarded."
            ),
        }

    def create_mesh_from_data(
        self,
        name: str,
        vertices: List[List[float]],
        edges: Optional[List[List[int]]] = None,
        faces: Optional[List[List[int]]] = None,
        location: Optional[List[float]] = None,
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a mesh object from raw vertex, edge, and face data.

        Every argument is checked before anything is created: each vertex is
        exactly three finite numbers, each index lies within the vertex list,
        no face repeats a vertex or the vertex set of an earlier face, and no
        edge joins a vertex to itself or repeats another edge. Such geometry
        is invalid in Blender and froze its Decimate. The collection must be
        in the current scene. The mesh is then built with
        ``mesh.from_pydata()`` and checked with ``mesh.validate()``; when any
        step fails, the mesh and object made so far are removed again.

        Args:
            name: Desired object name.
            vertices: Vertex positions, each exactly [x, y, z] finite numbers.
            edges: Loose edges as [v0, v1] pairs of two different vertices;
                face edges are made from the faces.
            faces: Faces as lists of three or more vertex indices, each vertex
                once and each vertex set once.
            location: Optional world-space origin [x, y, z].
            collection_name: Target collection in the current scene (the
                scene collection if None).

        Returns:
            Dict with object_name, mesh_name, vertex_count, edge_count,
            face_count and warnings.

        Raises:
            ValueError: If the geometry, location or collection is invalid.
        """
        blender_module: Any = bpy
        coordinates = [
            _finite_triple(vertex, f"Vertex {index}")
            for index, vertex in enumerate(vertices)
        ]
        if not coordinates:
            raise ValueError("vertices list must not be empty")
        edge_list = self._mesh_edges(edges or [], len(coordinates))
        face_list = self._mesh_faces(faces or [], len(coordinates))
        origin = None if location is None else _finite_triple(location, "location")
        collection = self._link_collection(collection_name)
        warnings = self._collection_warnings(collection)

        mesh: Any = None
        obj: Any = None
        try:
            mesh = blender_module.data.meshes.new(name)
            mesh.from_pydata(coordinates, edge_list, face_list)
            if mesh.validate(verbose=False):
                warnings.append(
                    "Blender's mesh validation changed the geometry; the counts are "
                    "what it kept"
                )
            mesh.update()
            obj = blender_module.data.objects.new(name, mesh)
            if origin is not None:
                obj.location = origin
            collection.objects.link(obj)
        except Exception:
            # Leave no orphan behind: an unlinked object or mesh is still
            # written by whole-scene exports, which then fail on it.
            if obj is not None:
                blender_module.data.objects.remove(obj, do_unlink=True)
            if mesh is not None:
                blender_module.data.meshes.remove(mesh)
            raise

        if obj.name != name:
            warnings.append(
                f"The name '{name}' was taken or too long; Blender named the object '{obj.name}'"
            )
        return {
            "object_name": obj.name,
            "mesh_name": mesh.name,
            "vertex_count": len(mesh.vertices),
            "edge_count": len(mesh.edges),
            "face_count": len(mesh.polygons),
            "warnings": warnings,
        }

    @staticmethod
    def _mesh_edges(edges: Sequence[Sequence[int]], vertex_count: int) -> List[Tuple[int, int]]:
        """Checked edge index pairs: in range, two distinct vertices, no edge twice."""
        checked: List[Tuple[int, int]] = []
        seen: Dict[Tuple[int, int], int] = {}
        for ei, edge in enumerate(edges):
            if len(edge) != 2:
                raise ValueError(f"Edge {ei} must have exactly 2 indices, got {len(edge)}")
            first, second = (_mesh_index(idx, f"Edge {ei}", vertex_count) for idx in edge)
            if first == second:
                raise ValueError(f"Edge {ei} joins vertex {first} to itself")
            key = (min(first, second), max(first, second))
            if key in seen:
                raise ValueError(f"Edge {ei} repeats edge {seen[key]}")
            seen[key] = ei
            checked.append((first, second))
        return checked

    @staticmethod
    def _mesh_faces(faces: Sequence[Sequence[int]], vertex_count: int) -> List[Tuple[int, ...]]:
        """Checked faces: three or more indices in range, each vertex once, no face twice.

        Two faces over the same vertex set count as duplicates whatever their
        order: they overlap entirely.
        """
        checked: List[Tuple[int, ...]] = []
        seen: Dict[Tuple[int, ...], int] = {}
        for fi, face in enumerate(faces):
            if len(face) < 3:
                raise ValueError(f"Face {fi} must have >= 3 indices, got {len(face)}")
            indices = tuple(_mesh_index(idx, f"Face {fi}", vertex_count) for idx in face)
            if len(set(indices)) != len(indices):
                repeated = next(idx for idx in indices if indices.count(idx) > 1)
                raise ValueError(
                    f"Face {fi} repeats vertex {repeated}; a face uses each vertex once"
                )
            key = tuple(sorted(indices))
            if key in seen:
                raise ValueError(
                    f"Face {fi} uses the same vertices as face {seen[key]} (a duplicate face)"
                )
            seen[key] = fi
            checked.append(indices)
        return checked

    def _link_collection(self, collection_name: Optional[str]) -> Any:
        """The collection a new object goes into: one that belongs to the current scene.

        An object linked only to a collection outside the scene is not part
        of it: attached-mode lookups do not find it, and whole-scene exports
        fail on it.
        """
        blender_module: Any = bpy
        scene = blender_module.context.scene
        if collection_name is None:
            return scene.collection
        collection = blender_module.data.collections.get(collection_name)
        if collection is None:
            raise ValueError(f"Collection '{collection_name}' not found")
        if collection not in scene.collection.children_recursive:
            raise ValueError(
                f"Collection '{collection_name}' is not in scene '{scene.name}', so the "
                "object would not be part of the scene; link the collection to the "
                "scene, or omit collection_name"
            )
        if getattr(collection, "library", None) is not None:
            raise ValueError(
                f"Collection '{collection_name}' is linked from a library and cannot take new objects"
            )
        return collection

    @staticmethod
    def _collection_warnings(collection: Any) -> List[str]:
        """A warning when the view layer excludes the collection an object goes into."""
        blender_module: Any = bpy
        view_layer: Any = getattr(blender_module.context, "view_layer", None)
        pending = [getattr(view_layer, "layer_collection", None)]
        while pending:
            layer = pending.pop()
            if layer is None:
                continue
            if layer.collection == collection:
                if layer.exclude:
                    return [
                        f"Collection '{collection.name}' is excluded from view layer "
                        f"'{view_layer.name}': Blender neither evaluates nor exports the object there"
                    ]
                return []
            pending.extend(layer.children)
        return []

    def clean_mesh(
        self,
        object_name: str,
        merge_distance: Optional[float] = None,
        dissolve_degenerate_distance: Optional[float] = None,
        recalculate_normals: bool = False,
    ) -> Dict[str, Any]:
        """
        Repair a mesh object's data: validate it, then optionally merge, dissolve and recalculate.

        ``mesh.validate()`` always runs first and removes invalid topology:
        faces that repeat a vertex, duplicate faces and duplicate edges, which
        can make Decimate loop forever. The optional steps then run in order
        on the valid mesh: merge vertices closer than ``merge_distance``
        (Merge by Distance), dissolve edges shorter and faces smaller than
        ``dissolve_degenerate_distance`` (Degenerate Dissolve), and point the
        face normals consistently outward. Mesh data shared by several objects
        changes for all of them. The object must be in Object Mode.

        Args:
            object_name: Mesh object whose data to clean.
            merge_distance: Merge vertices within this distance; None skips it.
            dissolve_degenerate_distance: Dissolve edges and faces smaller than
                this; None skips it.
            recalculate_normals: Make face normals point consistently outside.

        Returns:
            Dictionary with the counts before and after, whether validation
            changed anything, the merged vertex count, the operations run, the
            mesh's user count and warnings.
        """
        obj = self._get_object_or_raise(object_name)
        if obj.type != "MESH":
            raise ValueError(f"Object '{object_name}' is not a MESH (type={obj.type})")
        self._require_object_mode(obj, "clean its mesh")
        distances: Dict[str, Any] = {
            "merge_distance": merge_distance,
            "dissolve_degenerate_distance": dissolve_degenerate_distance,
        }
        for label, distance in distances.items():
            if distance is not None and (
                isinstance(distance, bool)
                or not isinstance(distance, numbers.Real)
                or not math.isfinite(distance)
                or distance <= 0
            ):
                raise ValueError(f"{label} must be a finite distance above 0, got {distance!r}")

        mesh = obj.data
        before = self._mesh_counts(mesh)
        validation_fixed = bool(mesh.validate(verbose=False))
        operations = ["validate"]
        vertices_merged = 0
        if merge_distance or dissolve_degenerate_distance or recalculate_normals:
            import bmesh  # type: ignore[import-not-found]

            bm = bmesh.new()
            try:
                bm.from_mesh(mesh)
                if merge_distance:
                    count = len(bm.verts)
                    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=merge_distance)
                    vertices_merged = count - len(bm.verts)
                    operations.append("merge_by_distance")
                if dissolve_degenerate_distance:
                    bmesh.ops.dissolve_degenerate(
                        bm, dist=dissolve_degenerate_distance, edges=list(bm.edges)
                    )
                    operations.append("dissolve_degenerate")
                if recalculate_normals:
                    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
                    operations.append("recalculate_normals")
                bm.to_mesh(mesh)
            finally:
                bm.free()
            mesh.update()

        warnings: List[str] = []
        if mesh.users > 1:
            warnings.append(
                f"Mesh '{mesh.name}' is shared by {mesh.users} users; all of them changed"
            )
        return {
            "object_name": object_name,
            "mesh_name": mesh.name,
            "before": before,
            "after": self._mesh_counts(mesh),
            "validation_fixed": validation_fixed,
            "vertices_merged": vertices_merged,
            "operations": operations,
            "mesh_users": int(mesh.users),
            "warnings": warnings,
        }

    # -- End scripting & mesh-from-data methods ------------------------------

    # -- SimReady Asset Format methods -----------------------------------------

    # Custom property prefix for all SimReady metadata stored on Blender objects
    _SIMREADY_PREFIX: str = "simready_"

    # Valid SimReady naming pattern: lowercase letters, digits, underscores only
    _SIMREADY_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

    def apply_simready_metadata(
        self,
        object_name: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Apply SimReady metadata as custom properties on a Blender object.

        Properties are stored with a ``simready_`` prefix so they survive
        USD export as custom attributes.

        Args:
            object_name: Target Blender object.
            metadata: Dict with optional keys ``semantic``, ``physics``,
                ``material``, each containing relevant sub-fields.

        Returns:
            Dict with object_name and list of applied_properties.
        """
        obj = self._get_object_or_raise(object_name)
        applied: List[str] = []

        semantic = metadata.get("semantic")
        if semantic:
            for key in ("semantic_class", "semantic_hierarchy", "semantic_qcode"):
                val = semantic.get(key)
                if val is not None:
                    prop_key = f"{self._SIMREADY_PREFIX}{key}"
                    obj[prop_key] = val
                    applied.append(prop_key)
            extra = semantic.get("additional_labels")
            if extra and isinstance(extra, dict):
                for label_key, label_val in extra.items():
                    prop_key = f"{self._SIMREADY_PREFIX}label_{label_key}"
                    obj[prop_key] = label_val
                    applied.append(prop_key)

        physics = metadata.get("physics")
        if physics:
            for key in (
                "mass_kg",
                "collider_type",
                "static_friction",
                "dynamic_friction",
                "restitution",
                "density",
            ):
                val = physics.get(key)
                if val is not None:
                    prop_key = f"{self._SIMREADY_PREFIX}{key}"
                    obj[prop_key] = val
                    applied.append(prop_key)
            if physics.get("is_rigid_body"):
                prop_key = f"{self._SIMREADY_PREFIX}is_rigid_body"
                obj[prop_key] = 1
                applied.append(prop_key)

        mat = metadata.get("material")
        if mat:
            for key in (
                "substrate_type",
                "material_naming",
                "shader_type",
                "texel_density",
            ):
                val = mat.get(key)
                if val is not None:
                    prop_key = f"{self._SIMREADY_PREFIX}{key}"
                    obj[prop_key] = val
                    applied.append(prop_key)

        return {"object_name": object_name, "applied_properties": applied}

    def get_simready_metadata(
        self,
        object_name: str,
    ) -> Dict[str, Any]:
        """
        Read SimReady metadata stored as custom properties on a Blender object.

        Args:
            object_name: Blender object to inspect.

        Returns:
            Dict with object_name, metadata (or None), and has_simready_data.
        """
        obj = self._get_object_or_raise(object_name)
        stored = self._read_simready_properties(obj)
        semantic = stored["semantic"]
        physics = stored["physics"]
        material = stored["material"]

        has_data = bool(semantic or physics or material)
        result_metadata: Optional[Dict[str, Any]] = None
        if has_data:
            result_metadata = {}
            if semantic:
                result_metadata["semantic"] = semantic
            if physics:
                if "is_rigid_body" in physics:
                    physics["is_rigid_body"] = bool(physics["is_rigid_body"])
                result_metadata["physics"] = physics
            if material:
                result_metadata["material"] = material

        return {
            "object_name": object_name,
            "metadata": result_metadata,
            "has_simready_data": has_data,
        }

    _SIMREADY_SEMANTIC_KEYS = ("semantic_class", "semantic_hierarchy", "semantic_qcode")
    _SIMREADY_PHYSICS_KEYS = (
        "mass_kg",
        "collider_type",
        "static_friction",
        "dynamic_friction",
        "restitution",
        "density",
        "is_rigid_body",
    )
    _SIMREADY_MATERIAL_KEYS = (
        "substrate_type",
        "material_naming",
        "shader_type",
        "texel_density",
    )

    def _read_simready_properties(self, obj: Any) -> Dict[str, Dict[str, Any]]:
        """The object's ``simready_*`` custom properties as stored, grouped like the metadata.

        Additional labels (``simready_label_<key>``) are gathered under
        ``semantic["additional_labels"]``; unknown ``simready_`` keys are left out.
        """
        semantic: Dict[str, Any] = {}
        physics: Dict[str, Any] = {}
        material: Dict[str, Any] = {}
        additional_labels: Dict[str, str] = {}
        prefix = self._SIMREADY_PREFIX
        for key in obj.keys():
            if not key.startswith(prefix):
                continue
            stripped = key[len(prefix) :]
            val = obj[key]
            # Convert IDPropertyArray / other exotic types to Python
            if hasattr(val, "to_list"):
                val = val.to_list()
            elif hasattr(val, "to_dict"):
                val = val.to_dict()

            if stripped.startswith("label_"):
                additional_labels[stripped[6:]] = str(val)
            elif stripped in self._SIMREADY_SEMANTIC_KEYS:
                semantic[stripped] = val
            elif stripped in self._SIMREADY_PHYSICS_KEYS:
                physics[stripped] = val
            elif stripped in self._SIMREADY_MATERIAL_KEYS:
                material[stripped] = val
        if additional_labels:
            semantic["additional_labels"] = additional_labels
        return {"semantic": semantic, "physics": physics, "material": material}

    def validate_simready_compliance(
        self,
        object_names: Optional[List[str]] = None,
        check_naming: bool = True,
        check_scale: bool = True,
        check_transforms: bool = True,
        check_materials: bool = True,
        check_hierarchy: bool = True,
    ) -> Dict[str, Any]:
        """
        Validate objects against NVIDIA SimReady conventions.

        Checks naming (lowercase_underscore), scale (meter-range dimensions),
        clean transforms (no stacked rotation/scale), material segmentation,
        and hierarchy structure.

        Args:
            object_names: Objects to check.  ``None`` checks all mesh objects.
            check_naming: Validate naming convention.
            check_scale: Validate real-world meter scale.
            check_transforms: Validate clean transforms.
            check_materials: Validate material segmentation.
            check_hierarchy: Validate hierarchy structure.

        Returns:
            Dict with compliant flag, object_count, issue_count, issues list.
        """
        issues: List[Dict[str, Any]] = []
        unit_scale = self._scene_unit_scale()

        if object_names is None:
            targets = [o for o in self._resolve_object_source(None) if o.type == "MESH"]
        else:
            targets = [self._get_object_or_raise(n) for n in object_names]

        for obj in targets:
            name: str = obj.name

            if check_naming and not self._SIMREADY_NAME_RE.match(name):
                issues.append(
                    {
                        "object_name": name,
                        "check": "naming",
                        "severity": "error",
                        "message": (
                            f"Name '{name}' violates SimReady convention "
                            "(lowercase letters, digits, underscores only)"
                        ),
                        "suggestion": (
                            "Rename to: "
                            + re.sub(r"[^a-z0-9_]", "_", name.lower()).strip("_")
                        ),
                    }
                )

            if check_scale:
                issues.extend(self._simready_size_issues(obj, unit_scale))

            if check_transforms:
                rot = tuple(obj.rotation_euler)
                scl = tuple(obj.scale)
                has_rotation = any(abs(r) > 1e-6 for r in rot)
                has_non_unit_scale = any(abs(s - 1.0) > 1e-6 for s in scl)
                if has_rotation:
                    issues.append(
                        {
                            "object_name": name,
                            "check": "transforms",
                            "severity": "warning",
                            "message": (
                                f"Non-zero rotation {rot} — SimReady requires "
                                "applied (zero) transforms"
                            ),
                            "suggestion": "Apply rotation: Ctrl+A → Rotation",
                        }
                    )
                if has_non_unit_scale:
                    issues.append(
                        {
                            "object_name": name,
                            "check": "transforms",
                            "severity": "warning",
                            "message": (
                                f"Non-unit scale {scl} — SimReady requires "
                                "applied (1,1,1) scale"
                            ),
                            "suggestion": "Apply scale: Ctrl+A → Scale",
                        }
                    )

            if check_materials and hasattr(getattr(obj, "data", None), "materials"):
                # Cameras, lights and empties carry no material slots.
                mat_count = len(obj.data.materials)
                if mat_count == 0:
                    issues.append(
                        {
                            "object_name": name,
                            "check": "materials",
                            "severity": "error",
                            "message": "No material assigned",
                            "suggestion": (
                                "Assign at least one material per mesh prim"
                            ),
                        }
                    )

            if check_hierarchy and obj.parent is None:
                has_children = len(obj.children) > 0
                if not has_children and obj.type == "MESH":
                    issues.append(
                        {
                            "object_name": name,
                            "check": "hierarchy",
                            "severity": "warning",
                            "message": (
                                "Mesh has no parent empty — SimReady assets "
                                "should be under a root XForm"
                            ),
                            "suggestion": (
                                "Use setup_simready_hierarchy to create a root"
                            ),
                        }
                    )

        error_count = sum(1 for i in issues if i["severity"] == "error")
        return {
            "compliant": error_count == 0,
            "object_count": len(targets),
            "issue_count": len(issues),
            "issues": issues,
            "meters_per_unit": unit_scale,
        }

    # A part's largest real-world dimension (Blender units times the scene's
    # Unit Scale) outside these bounds means the scene's units are not the ones
    # it was modelled in: a 0.5 m part modelled in millimetres in a metre scene
    # measures 500 m. Above the warning size an asset is plausible (a vehicle,
    # a rack) but worth a second look.
    _SIMREADY_MAX_SIZE_M = 100.0
    _SIMREADY_WARN_SIZE_M = 20.0
    _SIMREADY_MIN_SIZE_M = 0.001

    def _scene_unit_scale(self) -> float:
        """Meters per Blender unit: the scene's Unit Scale, 1.0 when unset or invalid."""
        blender_module: Any = bpy
        scene = getattr(blender_module.context, "scene", None)
        settings = getattr(scene, "unit_settings", None)
        try:
            scale = float(getattr(settings, "scale_length", 1.0))
        except (TypeError, ValueError):
            return 1.0
        if not math.isfinite(scale) or scale <= 0:
            return 1.0
        # Blender stores the scale in single precision (0.01 reads 0.00999999977).
        return float(f"{scale:.7g}")

    @staticmethod
    def _world_dimensions(obj: Any) -> List[float]:
        """The object's world-space bounding-box size in Blender units.

        Read from the bounding box corners and the world matrix, so a parent's
        scale counts; ``dimensions`` (object scale only) when either is missing.
        """
        corners = getattr(obj, "bound_box", None)
        matrix = getattr(obj, "matrix_world", None)
        if corners is not None and matrix is not None:
            try:
                rows = [[float(matrix[i][j]) for j in range(4)] for i in range(3)]
                points = [
                    [row[0] * c[0] + row[1] * c[1] + row[2] * c[2] + row[3] for row in rows]
                    for c in (tuple(corner) for corner in corners)
                ]
                if points:
                    return [
                        max(p[axis] for p in points) - min(p[axis] for p in points)
                        for axis in range(3)
                    ]
            except (TypeError, ValueError, IndexError):
                pass
        return [abs(float(d)) for d in getattr(obj, "dimensions", ())]

    def _simready_size_issues(self, obj: Any, unit_scale: float) -> List[Dict[str, Any]]:
        """A scale issue when a mesh's real-world size is implausible in the declared units."""
        if getattr(obj, "type", "MESH") != "MESH":
            return []
        dimensions = self._world_dimensions(obj)
        largest_units = max(dimensions) if dimensions else 0.0
        largest = largest_units * unit_scale
        if largest <= 0:
            return []
        declared = (
            f"{largest:.4g} m ({largest_units:.4g} Blender units at a Unit Scale of "
            f"{unit_scale:g} m)"
        )
        if largest > self._SIMREADY_MAX_SIZE_M:
            severity = "error"
            message = (
                f"Largest dimension {declared} is implausibly large for a SimReady "
                f"asset (over {self._SIMREADY_MAX_SIZE_M:g} m)"
            )
        elif largest < self._SIMREADY_MIN_SIZE_M:
            severity = "error"
            message = (
                f"Largest dimension {declared} is implausibly small for a SimReady "
                f"asset (under {self._SIMREADY_MIN_SIZE_M * 1000:g} mm)"
            )
        elif largest > self._SIMREADY_WARN_SIZE_M:
            severity = "warning"
            message = f"Largest dimension {declared} is large for a SimReady asset"
        else:
            return []
        return [
            {
                "object_name": obj.name,
                "check": "scale",
                "severity": severity,
                "message": message,
                "suggestion": (
                    "If it was modelled in other units, set Scene > Units > Unit Scale "
                    "to the modelling unit (0.001 for millimetres, 0.01 for centimetres) "
                    "or scale the mesh to its real size; pass check_scale=false for an "
                    "asset this size"
                ),
            }
        ]

    _SIMREADY_USD_EXTENSIONS = EXPORT_EXTENSIONS["USD"]
    # Object types left out unless include_lights_and_cameras is true.
    _SIMREADY_SCENE_TYPES = frozenset({"LIGHT", "CAMERA"})
    # Longest lists an export result carries; the counts give the totals.
    _SIMREADY_RESULT_ITEMS = 50

    def export_simready_usd(
        self,
        file_path: str,
        object_names: Optional[List[str]] = None,
        embed_metadata: bool = True,
        validate_before_export: bool = True,
        include_children: bool = True,
        include_lights_and_cameras: bool = False,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Export a SimReady asset as USD that a simulator reads as physics and semantics.

        Blender's USD exporter writes the geometry and render materials; the
        export then authors, from each object's simready_ metadata, UsdPhysics
        RigidBodyAPI, MassAPI (mass, and density in stage units), CollisionAPI
        with a MeshCollisionAPI approximation on the meshes, a physics material
        bound with the physics purpose, and UsdSemantics LabelsAPI labels (Isaac
        Sim's SemanticsAPI convention where Blender's USD has no UsdSemantics).
        The asset root is the default prim and holds the materials (Looks,
        PhysicsMaterials); metersPerUnit is the scene's Unit Scale. The file is
        written only when everything succeeded. As export_file does, the export
        applies modifiers, leaves Edit, Pose and paint modes for the export (so
        unsaved mesh edits are included) and enters them again, and restores the
        selection; a SimReady asset is static, so animation is left out.

        Args:
            file_path: Output .usd, .usda, .usdc or .usdz path.
            object_names: Objects to export, each with its descendants unless
                include_children is false; listed objects are exported even when
                hidden. Null exports the scene's objects that are visible in
                renders. An empty list is refused.
            embed_metadata: Also keep the simready_ custom properties (and
                Blender's names) as userProperties attributes, which Blender's
                USD import reads back as custom properties.
            validate_before_export: Validate the meshes first.
            include_children: Export the descendants of the listed objects.
            include_lights_and_cameras: Also export lights, cameras and the
                World's dome light, which are left out of an asset by default.
            force: Export although validation found errors.

        Returns:
            Dict with file_path (written exactly there; a missing folder is
            created), files_written, object_count (objects written), prims (object
            name and USD path), default_prim, meters_per_unit, up_axis, physics
            and semantics summaries, skipped_objects, validation_passed, issues
            and warnings. When validation found errors and force is false, an
            error result with the issues instead, and no file is written.

        Raises:
            PermissionError: If the path is outside the sandbox policy.
            ValueError: For an unsupported extension, an empty object_names, an
                unknown object, or a listed object that cannot be exported.
        """
        # Written at file_path exactly, as export_file does.
        file_path = self._export_target(self._deny_outside_sandbox(file_path, write=True))
        extension = os.path.splitext(file_path)[1].lower()
        if extension not in self._SIMREADY_USD_EXTENSIONS:
            raise ValueError(
                f"Unsupported SimReady export extension '{extension or file_path}'; "
                f"use one of {', '.join(self._SIMREADY_USD_EXTENSIONS)}"
            )
        if object_names is not None and not object_names:
            raise ValueError(
                "object_names is empty; list the objects to export, or pass null "
                "to export the whole scene"
            )
        targets, listed, skipped = self._simready_export_targets(
            object_names, include_children, include_lights_and_cameras
        )
        if not targets:
            raise ValueError(
                "Nothing to export: "
                + "; ".join(f"{s['object_name']}: {s['reason']}" for s in skipped)
            )

        issues: Optional[List[Dict[str, Any]]] = None
        validation_passed: Optional[bool] = None
        if validate_before_export:
            result = self.validate_simready_compliance(
                object_names=[o.name for o in targets if o.type == "MESH"]
            )
            issues = list(result.get("issues") or [])
            validation_passed = bool(result.get("compliant", True))
            if not validation_passed and not force:
                return self._simready_validation_refusal(issues)

        plan = {
            "objects": [
                {
                    "name": obj.name,
                    "parent": obj.parent.name if obj.parent is not None else None,
                    **self._read_simready_properties(obj),
                }
                for obj in targets
            ],
            "asset_name": usd_identifier(os.path.splitext(os.path.basename(file_path))[0]),
            "meters_per_unit": self._scene_unit_scale(),
            "embed_metadata": embed_metadata,
        }
        main_file = Path(file_path)
        created = self._make_folders(main_file.parent)
        before = self._folder_state(main_file.parent)
        workdir = tempfile.mkdtemp(prefix="simul-simready-")
        raw_path = self._simready_raw_path(file_path, workdir)
        export_warning: Optional[str] = None
        try:
            op_func = self._operator(self._export_op_path("USD"))
            # export_file's option translation: the selection keyword this
            # Blender's exporter defines, modifiers applied, Z up. A SimReady
            # asset is static, so animation is left out.
            kwargs = self._export_options(
                op_func,
                "USD",
                raw_path,
                None,
                subset=True,
                apply_modifiers=True,
                export_animation=False,
                frame_range=False,
                axes=(None, None),
                global_scale=None,
                usd_up_axis="Z",
                meters_per_unit=None,
                root_prim_path=None,
            )
            defined = op_func.get_rna_type().properties.keys()
            optional = {
                # The layer is read back through Blender's object names.
                "export_custom_properties": True,
                "author_blender_name": True,
                "root_prim_path": "/root",
                "export_lights": include_lights_and_cameras,
                "export_cameras": include_lights_and_cameras,
                "convert_world_material": include_lights_and_cameras,
                # Blender units go out unscaled; metersPerUnit declares them.
                "convert_scene_units": "METERS",
            }
            kwargs.update({k: v for k, v in optional.items() if k in defined})
            with self._simready_shown(targets, listed):
                # Object Mode, then exactly the targets selected; both restored.
                export_warning = self._run_export(
                    f"Failed to export SimReady USD file '{file_path}'",
                    op_func,
                    kwargs,
                    to_select=targets,
                )
            layer, report = author_simready_layer(raw_path, plan)
            missing = report["missing_objects"]
            listed_missing = sorted(name for name in missing if name in listed)
            if listed_missing:
                raise RuntimeError(
                    "Blender's USD exporter did not write "
                    + ", ".join(listed_missing)
                    + "; nothing was written to "
                    + file_path
                )
            skipped.extend(
                {"object_name": name, "reason": "not written by Blender's USD exporter"}
                for name in missing
            )
            self._simready_write(layer, file_path, workdir)
        except Exception:
            self._remove_empty_folders(created)
            raise
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
            if os.path.exists(raw_path) and not raw_path.startswith(workdir):
                os.remove(raw_path)

        prims = sorted(report["prims"].items(), key=lambda item: item[1])
        limit = self._SIMREADY_RESULT_ITEMS
        warnings = report["warnings"] + ([export_warning] if export_warning else [])
        return {
            "file_path": file_path,
            "files_written": self._written_files(main_file.parent, before, file_path),
            "object_count": len(prims),
            "default_prim": report["default_prim"],
            "prims": [{"object_name": n, "prim_path": p} for n, p in prims[:limit]],
            "meters_per_unit": report["meters_per_unit"],
            "up_axis": report["up_axis"],
            "kilograms_per_unit": report["kilograms_per_unit"],
            "physics": {
                "rigid_body_count": len(report["rigid_bodies"]),
                "collider_count": len(report["colliders"]),
                "mass_count": len(report["mass_prims"]),
                "physics_material_count": len(report["physics_materials"]),
                "rigid_bodies": report["rigid_bodies"][:limit],
                "colliders": report["colliders"][:limit],
                "physics_materials": report["physics_materials"][:limit],
            },
            "semantics": {
                "labels_schema": report["semantics_schema"],
                "labelled_prim_count": len(report["labelled_prims"]),
                "labelled_prims": report["labelled_prims"][:limit],
            },
            "skipped_objects": skipped[:limit],
            "validation_passed": validation_passed,
            "forced": bool(force and validation_passed is False),
            "issue_count": len(issues) if issues is not None else 0,
            "issues": issues[:limit] if issues is not None else None,
            "warnings": warnings[:limit],
        }

    def _simready_validation_refusal(self, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """The error result of an export that validation stopped."""
        errors = [i for i in issues if i.get("severity") == "error"]
        checks = sorted({str(i.get("check")) for i in errors})
        first = errors[0] if errors else {}
        return {
            "success": False,
            "error_type": "SimReadyValidationFailed",
            "error": (
                f"Not exported: validation found {len(errors)} error(s) "
                f"({', '.join(checks)}), first {first.get('object_name')}: "
                f"{first.get('message')}. Fix them, or pass force=true to export anyway"
            ),
            "details": {
                "file_written": False,
                "issue_count": len(issues),
                "issues": issues[: self._SIMREADY_RESULT_ITEMS],
            },
        }

    @staticmethod
    def _descendants(obj: Any) -> List[Any]:
        """Every object parented below ``obj``, depth first."""
        recursive = getattr(obj, "children_recursive", None)
        if recursive is not None:
            return list(recursive)
        found: List[Any] = []
        for child in getattr(obj, "children", ()) or ():
            found.append(child)
            found.extend(BlenderRuntimeSession._descendants(child))
        return found

    @staticmethod
    def _render_hidden(obj: Any) -> bool:
        """Hidden in renders by the object itself or one of its collections."""
        if getattr(obj, "hide_render", False):
            return True
        return any(
            getattr(c, "hide_render", False) for c in getattr(obj, "users_collection", ())
        )

    def _simready_export_targets(
        self,
        object_names: Optional[List[str]],
        include_children: bool,
        include_lights_and_cameras: bool,
    ) -> Tuple[List[Any], set, List[Dict[str, str]]]:
        """The objects to export, the listed ones' names, and the left-out ones with why."""
        blender_module: Any = bpy
        layer_objects = blender_module.context.view_layer.objects
        listed: List[Any] = []
        if object_names is None:
            candidates = list(blender_module.context.scene.objects)
        else:
            listed = [self._get_object_or_raise(name) for name in object_names]
            outside = sorted(o.name for o in listed if layer_objects.get(o.name) is None)
            if outside:
                raise ValueError(
                    "Cannot export "
                    + ", ".join(outside)
                    + ": not in the active view layer (another scene, or an excluded "
                    "collection)"
                )
            candidates = list(listed)
            if include_children:
                for obj in listed:
                    candidates.extend(self._descendants(obj))
        listed_names = {o.name for o in listed}
        targets: List[Any] = []
        skipped: List[Dict[str, str]] = []
        seen: set = set()
        for obj in candidates:
            if obj.name in seen:
                continue
            seen.add(obj.name)
            reason = None
            if obj.type in self._SIMREADY_SCENE_TYPES and not include_lights_and_cameras:
                reason = "lights and cameras are left out; pass include_lights_and_cameras"
            elif obj.name not in listed_names:
                if layer_objects.get(obj.name) is None:
                    reason = "not in the active view layer"
                elif self._render_hidden(obj):
                    reason = "hidden in renders; list it in object_names to export it"
            if reason is None:
                targets.append(obj)
            else:
                skipped.append({"object_name": obj.name, "reason": reason})
        return targets, listed_names, skipped

    @staticmethod
    def _simready_raw_path(file_path: str, workdir: str) -> str:
        """Where Blender's exporter writes before the SimReady layer is authored.

        Next to the destination, so texture paths stay relative to it; a .usdz
        is packaged from a .usdc in ``workdir`` instead, as Blender does.
        """
        directory, base = os.path.split(file_path)
        stem, extension = os.path.splitext(base)
        if extension.lower() == ".usdz":
            return os.path.join(workdir, f"blender_{usd_identifier(stem)}.usdc")
        return os.path.join(directory, f".{stem}.simul-{uuid.uuid4().hex[:8]}{extension}")

    @staticmethod
    def _simready_write(layer: Any, file_path: str, workdir: str) -> None:
        """Write the authored layer to ``file_path``; a .usdz is packaged with its textures."""
        if not file_path.lower().endswith(".usdz"):
            if not layer.Export(file_path):
                raise RuntimeError(f"Could not write {file_path}")
            return
        from pxr import Sdf, UsdUtils

        stem = usd_identifier(os.path.splitext(os.path.basename(file_path))[0])
        root_layer = os.path.join(workdir, f"{stem}.usdc")
        if not layer.Export(root_layer):
            raise RuntimeError(f"Could not write {root_layer}")
        directory = os.path.dirname(file_path) or "."
        package = os.path.join(directory, f".{stem}.simul-{uuid.uuid4().hex[:8]}.usdz")
        try:
            if not UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(root_layer), package):
                raise RuntimeError(f"Could not package {file_path}")
            os.replace(package, file_path)
        finally:
            if os.path.exists(package):
                os.remove(package)

    @staticmethod
    @contextmanager
    def _simready_shown(targets: List[Any], listed: set) -> Iterator[None]:
        """Show ``targets`` to Blender's exporter for the block, then hide them again.

        An object hidden in the viewport cannot be selected, so _run_export
        could not export it; a listed object is also shown in renders, which
        the exporter otherwise skips. _run_export leaves Edit Mode for the
        export and selects exactly the targets, and restores both.
        """
        shown: List[Tuple[Any, Tuple[bool, bool, bool, bool]]] = []
        try:
            for obj in targets:
                shown.append(
                    (obj, (obj.hide_get(), obj.hide_viewport, obj.hide_render, obj.hide_select))
                )
                obj.hide_viewport = False
                obj.hide_select = False
                obj.hide_set(False)
                if obj.name in listed:
                    obj.hide_render = False
            yield
        finally:
            for obj, (hidden, viewport, render, unselectable) in reversed(shown):
                obj.hide_viewport = viewport
                obj.hide_render = render
                obj.hide_select = unselectable
                obj.hide_set(hidden)

    def setup_simready_hierarchy(
        self,
        root_name: str,
        child_names: List[str],
        semantic: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a SimReady-compliant hierarchy with a root empty (XForm)
        and parent the given children under it.

        The root empty is placed at the origin with identity transforms,
        matching SimReady modeling best-practices.  Optionally applies
        semantic labels to the root.

        Args:
            root_name: Name for the root empty object.
            child_names: Existing objects to parent under the root.
            semantic: Optional semantic labels dict to apply to the root.

        Returns:
            Dict with root_name, children, hierarchy_path.
        """
        blender_module: Any = bpy

        # Validate root name follows SimReady conventions
        if not self._SIMREADY_NAME_RE.match(root_name):
            raise ValueError(
                f"Root name '{root_name}' violates SimReady naming "
                "(lowercase letters, digits, underscores only)"
            )

        # Create root empty at origin
        empty = blender_module.data.objects.new(root_name, None)
        blender_module.context.scene.collection.objects.link(empty)
        empty.empty_display_type = "PLAIN_AXES"
        empty.location = (0.0, 0.0, 0.0)
        empty.rotation_euler = (0.0, 0.0, 0.0)
        empty.scale = (1.0, 1.0, 1.0)

        # Parent children
        parented: List[str] = []
        for child_name in child_names:
            child = self._get_object_or_raise(child_name)
            child.parent = empty
            parented.append(child.name)

        # Apply semantic labels if provided
        if semantic:
            self.apply_simready_metadata(
                empty.name,
                {"semantic": semantic},
            )

        hierarchy_parts = [f"/{empty.name}"]
        for c in parented:
            hierarchy_parts.append(f"/{empty.name}/{c}")
        hierarchy_path = hierarchy_parts[0]

        return {
            "root_name": empty.name,
            "children": parented,
            "hierarchy_path": hierarchy_path,
        }

    # -- End SimReady methods --------------------------------------------------

    def cleanup(self) -> None:
        """Clean up resources for the Blender session."""
        self.logger.debug("Blender runtime session cleaned up")

    def _get_object_or_raise(self, object_name: str) -> Any:
        """
        Look up a Blender object by name or raise ValueError.

        Args:
            object_name: Object name to look up.

        Returns:
            The Blender object reference.
        """
        blender_module: Any = bpy
        objects = (
            blender_module.context.scene.objects
            if self._scene_scope
            else blender_module.data.objects
        )
        obj = objects.get(object_name)
        if obj is None:
            raise ValueError(f"Object not found: {object_name}")
        return obj

    def _extract_principled_params(self, node: Any) -> Dict[str, Any]:
        """
        Extract common Principled BSDF input values.

        Version-aware: Blender 4.0+ renamed several socket labels.

        Args:
            node: ShaderNodeBsdfPrincipled node.

        Returns:
            Dictionary of parameter name → value.
        """
        params: Dict[str, Any] = {}
        # Socket names that are stable across versions
        target_sockets = (
            "Base Color",
            "Metallic",
            "Roughness",
            "IOR",
            "Alpha",
            "Normal",
            "Emission Color" if self._is_blender_4_plus else "Emission",
            "Emission Strength",
        )
        for sock_name in target_sockets:
            sock = node.inputs.get(sock_name)
            if sock is None:
                continue
            value = sock.default_value
            # Convert Vector/Color to list for serialisation
            if hasattr(value, "__iter__"):
                params[sock_name] = list(value)
            else:
                params[sock_name] = value
        return params

    @staticmethod
    def _is_object_visible(scene_object: Any) -> bool:
        """Determine object visibility in a version-compatible way."""
        if hasattr(scene_object, "visible_get"):
            try:
                return bool(scene_object.visible_get())
            except Exception as exc:
                logger.debug(
                    "visible_get() failed, using hide_viewport fallback: %s", exc
                )
        return not bool(getattr(scene_object, "hide_viewport", False))

    def _resolve_object_source(self, collection_name: Optional[str]) -> Any:
        """
        Resolve object iterable from collection or global object list.

        Args:
            collection_name: Optional collection to source objects from.

        Returns:
            Iterable of Blender objects.
        """
        if bpy is None:
            raise RuntimeError("bpy module is unavailable during object listing")

        blender_module: Any = bpy

        if not collection_name:
            return (
                blender_module.context.scene.objects
                if self._scene_scope
                else blender_module.data.objects
            )

        collection = blender_module.data.collections.get(collection_name)
        if collection is None:
            raise ValueError(f"Collection not found: {collection_name}")
        if self._scene_scope:
            return [
                obj
                for obj in collection.objects
                if obj.name in blender_module.context.scene.objects
            ]
        return collection.objects

    def _resolve_camera(self, camera_name: Optional[str] = None) -> Any:
        """
        Resolve a camera object by name or return the active scene camera.

        Args:
            camera_name: Explicit camera name, or None for active camera.

        Returns:
            Camera object reference.
        """
        blender_module: Any = bpy
        if camera_name:
            objects = (
                blender_module.context.scene.objects
                if self._scene_scope
                else blender_module.data.objects
            )
            cam = objects.get(camera_name)
            if cam is None or cam.type != "CAMERA":
                raise ValueError(f"Camera not found: {camera_name}")
            return cam

        scene_cam = blender_module.context.scene.camera
        if scene_cam is None:
            raise ValueError("No active camera in scene")
        return scene_cam

    def _resolve_capture_method(
        self, method: str, use_render_fallback: bool = False
    ) -> str:
        """
        Turn a requested capture method into ``viewport`` or ``render``.

        Args:
            method: One of CAPTURE_METHODS.
            use_render_fallback: The deprecated spelling of ``method="render"``.

        Returns:
            ``viewport`` or ``render``.

        Raises:
            ValueError: For an unknown method, or use_render_fallback together
                with method="viewport".
        """
        if method not in CAPTURE_METHODS:
            raise ValueError(
                f"method must be one of {', '.join(CAPTURE_METHODS)}, not {method!r}"
            )
        if use_render_fallback:
            if method == "viewport":
                raise ValueError(
                    "use_render_fallback=True asks for a render; it contradicts method='viewport'"
                )
            return "render"
        if method == "auto":
            return "viewport" if self._scene_scope else "render"
        return method

    @staticmethod
    def _fit_inside(
        source_width: int, source_height: int, width: int, height: int
    ) -> Tuple[int, int]:
        """
        Scale a source size to fit inside width x height, keeping its aspect ratio.

        Args:
            source_width: Width of the view being captured.
            source_height: Height of the view being captured.
            width: Largest output width.
            height: Largest output height.

        Returns:
            The output (width, height), each at least one pixel.
        """
        scale = min(width / source_width, height / source_height)
        return max(1, round(source_width * scale)), max(1, round(source_height * scale))

    @staticmethod
    def _view3d_target() -> Tuple[Any, Any, Any]:
        """
        Find the 3D View to capture: the context's, else the window's largest.

        The attached bridge puts the window's largest 3D View in the context.
        Embedded bpy runs in background mode: the startup file's windows and
        areas exist there, but nothing was ever drawn in them.

        Returns:
            The (area, region, space) of the 3D View.

        Raises:
            RuntimeError: When no 3D View with a drawable region exists.
        """
        blender_module: Any = bpy
        context = blender_module.context
        window = getattr(context, "window", None)
        area = None
        if window is not None and not getattr(blender_module.app, "background", False):
            area = getattr(context, "area", None)
            if area is None or area.type != "VIEW_3D":
                views = [a for a in window.screen.areas if a.type == "VIEW_3D"]
                area = max(views, key=lambda a: a.width * a.height, default=None)
        if area is None:
            raise RuntimeError(
                "No 3D View to capture: method='viewport' needs an attached Blender window "
                "that shows a 3D View (embedded bpy has no editor window). Open a 3D View, "
                "or pass method='render' to render through a camera."
            )
        space = area.spaces.active
        regions = [r for r in area.regions if r.type == "WINDOW"]
        # Quad view has four WINDOW regions; region_3d is the one not locked to an axis.
        region = next((r for r in regions if r.data == space.region_3d), None) or max(
            regions, key=lambda r: r.width * r.height, default=None
        )
        if (
            region is None
            or region.data is None
            or region.width < 2
            or region.height < 2
        ):
            raise RuntimeError("The 3D View has no drawable region to capture")
        return area, region, space

    @staticmethod
    def _drawn_by_external_engine(space: Any, engine: str) -> bool:
        """
        Report whether the 3D View's image comes from an external render engine.

        Such an engine (Cycles, third-party) draws only into the on-screen
        editor; an offscreen redraw would show the overlays without the scene.

        Args:
            space: The SpaceView3D being captured.
            engine: The scene's render engine identifier.

        Returns:
            True when the view must be read from the screen.
        """
        shading = space.shading.type
        if engine in _DRAW_MANAGER_ENGINES or shading in ("WIREFRAME", "SOLID"):
            return False
        if shading == "RENDERED":
            return True
        # Material Preview draws with EEVEE unless the engine opts out of it.
        blender_module: Any = bpy
        engine_type = next(
            (
                cls
                for cls in blender_module.types.RenderEngine.__subclasses__()
                if getattr(cls, "bl_idname", None) == engine
            ),
            None,
        )
        return engine_type is not None and not getattr(
            engine_type, "bl_use_eevee_viewport", False
        )

    @contextmanager
    def _capture_workdir(self) -> Iterator[Path]:
        """
        Yield a scratch directory for capture files inside the sandbox.

        Blender writes images only to files, so the encoded JPEG passes
        through here; the directory is removed afterwards.

        Yields:
            The temporary directory.

        Raises:
            PermissionError: When the sandbox allows no writable capture directory.
        """
        capture_dir = self._path_policy.default_capture_dir()
        if capture_dir is None:
            raise PermissionError(
                "No writable capture directory is allowed by the sandbox"
            )
        capture_dir = self._deny_outside_sandbox(capture_dir, write=True)
        Path(capture_dir).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="blender-", dir=capture_dir
        ) as temporary:
            yield Path(temporary)

    @staticmethod
    @contextmanager
    def _image_format(settings: Any, file_format: str, **values: Any) -> Iterator[None]:
        """
        Set the scene's output image format for one capture, then restore it.

        Blender 5.0 offers only video formats while ``media_type`` is VIDEO,
        so an image format needs ``media_type`` IMAGE first; older versions
        have no ``media_type``. Choosing JPEG also downgrades the color mode
        and depth (RGBA to RGB, 16 to 8 bit), so those are restored as well.

        Args:
            settings: The scene's ImageFormatSettings.
            file_format: Image format to use, such as JPEG or PNG.
            **values: Further ImageFormatSettings attributes to set.

        Yields:
            Nothing; the format applies inside the block.
        """
        names = [
            name
            for name in dict.fromkeys((*_IMAGE_FORMAT_FIELDS, *values))
            if hasattr(settings, name)
        ]
        saved = {name: getattr(settings, name) for name in names}
        try:
            if "media_type" in saved:
                settings.media_type = "IMAGE"
            settings.file_format = file_format
            for name, value in values.items():
                setattr(settings, name, value)
            yield
        finally:
            for name in names:
                try:
                    setattr(settings, name, saved[name])
                except (AttributeError, TypeError, ValueError):
                    logger.exception("Could not restore render image setting %s", name)

    @staticmethod
    def _new_image(width: int, height: int, pixels: Any) -> Any:
        """
        Create a temporary byte image from RGBA float pixels.

        Args:
            width: Image width.
            height: Image height.
            pixels: Flat RGBA values in [0, 1], bottom row first.

        Returns:
            The new image; the caller removes it.
        """
        blender_module: Any = bpy
        image = blender_module.data.images.new(
            "simul_capture", width, height, alpha=False
        )
        try:
            image.pixels.foreach_set(pixels)
        except Exception:
            blender_module.data.images.remove(image)
            raise
        return image

    @staticmethod
    def _save_jpeg(image: Any, jpeg_quality: int, workdir: Path) -> str:
        """
        Encode an image as JPEG with Blender's writer; Pillow is not in Blender.

        Args:
            image: The image to encode.
            jpeg_quality: JPEG compression quality.
            workdir: Scratch directory for the file Blender writes.

        Returns:
            The base64-encoded JPEG.
        """
        output = workdir / "capture.jpg"
        image.file_format = "JPEG"
        image.save(filepath=str(output), quality=jpeg_quality)
        return base64.b64encode(output.read_bytes()).decode("ascii")

    def _capture_view3d(
        self,
        width: int,
        height: int,
        jpeg_quality: int,
        engine: str,
    ) -> Dict[str, Any]:
        """
        Capture the 3D View as the user sees it; see capture_viewport.

        Args:
            width: Largest image width.
            height: Largest image height.
            jpeg_quality: JPEG compression quality.
            engine: Render engine identifier.

        Returns:
            Dictionary with image_base64, dimensions, engine, capture_method.
        """
        blender_module: Any = bpy
        context = blender_module.context
        area, region, space = self._view3d_target()
        out_width, out_height = self._fit_inside(
            region.width, region.height, width, height
        )
        with self._capture_workdir() as workdir:
            if self._drawn_by_external_engine(space, engine):
                image = self._read_view3d_screen(context.window, area, region, workdir)
                capture_method = "viewport_screen"
            else:
                image = self._draw_view3d_offscreen(
                    context.scene,
                    context.view_layer,
                    space,
                    region,
                    out_width,
                    out_height,
                )
                capture_method = "viewport"
            try:
                if tuple(image.size) != (out_width, out_height):
                    image.scale(out_width, out_height)
                b64 = self._save_jpeg(image, jpeg_quality, workdir)
            finally:
                blender_module.data.images.remove(image)
        return {
            "image_base64": b64,
            "width": out_width,
            "height": out_height,
            "engine": engine,
            "capture_method": capture_method,
            "format": "jpeg",
        }

    def _draw_view3d_offscreen(
        self,
        scene: Any,
        view_layer: Any,
        space: Any,
        region: Any,
        width: int,
        height: int,
    ) -> Any:
        """
        Redraw the 3D View offscreen with its own matrices, shading and overlays.

        Python draw handlers do not run, so agent annotations stay out.

        Args:
            scene: The window's scene.
            view_layer: The window's view layer.
            space: The SpaceView3D to draw.
            region: Its WINDOW region.
            width: Image width; the region's aspect ratio.
            height: Image height.

        Returns:
            A temporary image with the pixels; the caller removes it.
        """
        import gpu  # type: ignore[import-not-found]

        view = region.data
        # Blender refreshes view matrices on redraw; a view changed since the
        # last one, or a window that is not redrawing, would be stale.
        view.update()
        offscreen = gpu.types.GPUOffScreen(width, height)
        try:
            offscreen.draw_view3d(
                scene,
                view_layer,
                space,
                region,
                view.view_matrix,
                view.window_matrix,
                do_color_management=True,
            )
            with offscreen.bind():
                pixels = gpu.state.active_framebuffer_get().read_color(
                    0, 0, width, height, 4, 0, "FLOAT"
                )
        finally:
            offscreen.free()
        pixels.dimensions = width * height * 4
        return self._new_image(width, height, pixels)

    def _read_view3d_screen(
        self, window: Any, area: Any, region: Any, workdir: Path
    ) -> Any:
        """
        Read the 3D View's on-screen pixels, for views only an external engine draws.

        The screenshot covers the whole editor, so overlapping UI (header,
        toolbar, open popups) is included; it is cropped to the main region.

        ``screenshot_area`` redraws the regions tagged for redraw and reads the
        frame it drew. With nothing tagged it returns an older frame from the
        window's swap chain instead (one to four draws old on Blender 5.0.1),
        which can show agent annotations drawn before the bridge hid them for
        this capture, or an outdated view. Tagging the region right before the
        screenshot makes it draw the region now, without those annotations.
        Writing the scene's image settings, as ``_image_format`` does here,
        happens to cause that redraw as well; the tag does not rely on it.

        Args:
            window: The window showing the 3D View.
            area: The 3D View's area.
            region: Its WINDOW region.
            workdir: Scratch directory for the screenshot file.

        Returns:
            A temporary region-sized image; the caller scales and removes it.

        Raises:
            RuntimeError: When Blender cannot take the screenshot.
        """
        import numpy as np

        blender_module: Any = bpy
        shot = workdir / "view3d.png"
        settings = blender_module.context.scene.render.image_settings
        with self._image_format(settings, "PNG", color_mode="RGB", color_depth="8"):
            with blender_module.context.temp_override(
                window=window, area=area, region=region
            ):
                # Nothing may draw between the tag and the screenshot: a redraw
                # in between (bpy.ops.wm.redraw_timer, say) uses up the tag, and
                # the screenshot reads a stale frame again.
                region.tag_redraw()
                status = blender_module.ops.screen.screenshot_area(filepath=str(shot))
        if "FINISHED" not in status or not shot.is_file():
            raise RuntimeError("Blender could not read the 3D View from the screen")
        source = blender_module.data.images.load(str(shot), check_existing=False)
        try:
            shot_width, shot_height = source.size
            pixels = np.empty(shot_width * shot_height * 4, dtype=np.float32)
            source.pixels.foreach_get(pixels)
        finally:
            blender_module.data.images.remove(source)
        scale_x, scale_y = shot_width / area.width, shot_height / area.height
        left = round((region.x - area.x) * scale_x)
        bottom = round((region.y - area.y) * scale_y)
        crop = pixels.reshape(shot_height, shot_width, 4)[
            bottom : bottom + round(region.height * scale_y),
            left : left + round(region.width * scale_x),
        ]
        crop_height, crop_width = crop.shape[:2]
        if crop_width < 1 or crop_height < 1:
            raise RuntimeError("The 3D View lies outside Blender's screenshot")
        return self._new_image(
            crop_width, crop_height, np.ascontiguousarray(crop).ravel()
        )

    def _capture_render(
        self,
        width: int,
        height: int,
        jpeg_quality: int,
        engine: str,
    ) -> Dict[str, Any]:
        """
        Capture via bpy.ops.render.render (all engines), through a camera.

        Temporarily adjusts render settings and lets Blender encode the JPEG.
        Render Result.pixels can be empty even after a successful render. Using
        save_render also avoids requiring Pillow inside the Blender add-on.
        A scene without a camera gets a temporary one for this render only.

        Args:
            width: Image width.
            height: Image height.
            jpeg_quality: JPEG compression quality.
            engine: Render engine identifier.

        Returns:
            Dictionary with image_base64, dimensions, engine, capture_method.
        """
        blender_module: Any = bpy
        scene = blender_module.context.scene
        render = scene.render
        capture_method = "render"
        with self._capture_workdir() as workdir:
            saved = (
                render.resolution_x,
                render.resolution_y,
                render.resolution_percentage,
            )
            camera = None
            try:
                if scene.camera is None:
                    camera, capture_method, (width, height) = (
                        self._add_temporary_camera(scene, width, height)
                    )
                render.resolution_x = width
                render.resolution_y = height
                render.resolution_percentage = 100
                with self._image_format(
                    render.image_settings, "JPEG", quality=jpeg_quality
                ):
                    blender_module.ops.render.render(write_still=False)
                    result_image = blender_module.data.images.get("Render Result")
                    if result_image is None:
                        raise RuntimeError("Render Result image not available")
                    output = workdir / "capture.jpg"
                    result_image.save_render(filepath=str(output), scene=scene)
                    b64 = base64.b64encode(output.read_bytes()).decode("ascii")
            finally:
                (
                    render.resolution_x,
                    render.resolution_y,
                    render.resolution_percentage,
                ) = saved
                if camera is not None:
                    self._remove_temporary_camera(scene, camera)
        return {
            "image_base64": b64,
            "width": width,
            "height": height,
            "engine": engine,
            "capture_method": capture_method,
            "format": "jpeg",
        }

    def _add_temporary_camera(
        self, scene: Any, width: int, height: int
    ) -> Tuple[Any, str, Tuple[int, int]]:
        """
        Add and activate a camera for one render of a scene that has none.

        Aligned to the 3D View when there is one, keeping its aspect ratio;
        otherwise framing the visible objects at width x height.

        Args:
            scene: The scene to render.
            width: Requested image width.
            height: Requested image height.

        Returns:
            The camera object, the capture_method naming its placement, and
            the render size.
        """
        blender_module: Any = bpy
        data = blender_module.data.cameras.new("simul_temp_camera")
        camera = blender_module.data.objects.new("simul_temp_camera", data)
        try:
            scene.collection.objects.link(camera)
            try:
                _area, region, space = self._view3d_target()
            except RuntimeError:
                region = None
            if region is not None:
                width, height = self._fit_inside(
                    region.width, region.height, width, height
                )
                self._align_camera_to_view(camera, space, region.data, width, height)
                capture_method = "render_temp_camera_view"
            else:
                self._frame_camera_on_objects(camera, scene, width, height)
                capture_method = "render_temp_camera_framing"
            scene.camera = camera
        except Exception:
            self._remove_temporary_camera(scene, camera)
            raise
        return camera, capture_method, (width, height)

    @staticmethod
    def _remove_temporary_camera(scene: Any, camera: Any) -> None:
        """
        Delete a temporary camera and its data, leaving the scene without a camera again.

        Args:
            scene: The scene it was added to.
            camera: The temporary camera object.
        """
        blender_module: Any = bpy
        if scene.camera == camera:
            scene.camera = None
        data = camera.data
        blender_module.data.objects.remove(camera, do_unlink=True)
        blender_module.data.cameras.remove(data)

    @staticmethod
    def _align_camera_to_view(
        camera: Any, space: Any, view: Any, width: int, height: int
    ) -> None:
        """
        Give a camera the 3D View's position, orientation and field of view.

        Args:
            camera: The camera object.
            space: The SpaceView3D, for its clip range.
            view: The region's RegionView3D.
            width: Render width; the view's aspect ratio.
            height: Render height.
        """
        view.update()
        data = camera.data
        data.sensor_fit = "AUTO"
        projection = view.window_matrix
        # AUTO fit spreads the camera's field over the longer side.
        scale = projection[0][0] if width >= height else projection[1][1]
        view_to_world = view.view_matrix.inverted()
        camera.matrix_world = view_to_world
        if view.is_perspective:
            data.type = "PERSP"
            # From the projection, not space.lens: the 3D View shows the field
            # of a camera with half its focal length.
            data.angle = 2.0 * math.atan(1.0 / scale)
            data.clip_start = space.clip_start
            data.clip_end = space.clip_end
            return
        data.type = "ORTHO"
        data.ortho_scale = 2.0 / scale
        # An orthographic 3D View also shows what lies behind its eye point,
        # half its clip range deep; back the camera off by that much.
        depth = space.clip_end
        backward = view_to_world.col[2]
        camera.location = tuple(
            origin + axis * depth / 2
            for origin, axis in zip(view_to_world.translation, backward)
        )
        data.clip_start = 0.001
        data.clip_end = depth

    def _frame_camera_on_objects(
        self, camera: Any, scene: Any, width: int, height: int
    ) -> None:
        """
        Point a camera at the visible objects from Blender's default camera side.

        Args:
            camera: The camera object.
            scene: The scene whose visible, renderable objects are framed.
            width: Render width.
            height: Render height.
        """
        self._evaluate_transforms()
        corners: List[List[float]] = []
        for obj in scene.objects:
            if (
                obj.type in _UNFRAMED_TYPES
                or getattr(obj, "hide_render", False)
                or not self._is_object_visible(obj)
            ):
                continue
            corners.extend(self._world_bound_box(obj))
        if corners:
            center, radius = self._bounding_sphere(corners)
            radius = max(radius, 0.1)
        else:
            center, radius = [0.0, 0.0, 0.0], 1.0
        data = camera.data
        data.sensor_fit = "AUTO"
        # The frame's narrower side decides how far back the camera must stand
        # to see the whole sphere.
        distance = self._fitting_distance(
            radius, min(self._frame_half_extents(data, width, height))
        )
        norm = math.sqrt(sum(v * v for v in _FRAMING_DIRECTION))
        location = [c + v / norm * distance for c, v in zip(center, _FRAMING_DIRECTION)]
        # A new camera has no parent and an XYZ Euler rotation, so these are
        # its world transform.
        camera.location = tuple(location)
        camera.rotation_euler = self._look_at_rotation(location, center)
        data.clip_start = max(0.001, (distance - radius) / 2)
        data.clip_end = distance + 2 * radius

    @staticmethod
    def _bounding_sphere(points: List[List[float]]) -> Tuple[List[float], float]:
        """
        Return a sphere around points: centered in their axis-aligned box.

        Args:
            points: World-space points, such as bounding-box corners.

        Returns:
            The center, and the radius reaching the farthest point.
        """
        low = [min(p[i] for p in points) for i in range(3)]
        high = [max(p[i] for p in points) for i in range(3)]
        center = [(lo + hi) / 2 for lo, hi in zip(low, high)]
        return center, max(math.dist(center, p) for p in points)

    @staticmethod
    def _frame_half_extents(
        data: Any, aspect_x: float, aspect_y: float
    ) -> Tuple[float, float]:
        """
        Return how far a camera's frame reaches from its view axis, across and up.

        Per unit of depth for a perspective camera (the tangents of half the
        field of view), per unit of ortho_scale for an orthographic one; a
        panoramic camera counts as a perspective one with its lens.
        Blender fits the sensor to one side of the frame, the field of view
        ``data.angle`` spans that side: the width for sensor_fit HORIZONTAL,
        the height for VERTICAL, the longer side for AUTO. The other side
        follows the aspect ratio. A lens shift moves the frame off the axis by
        a fraction of the fitted side; the nearer edge counts.

        Args:
            data: The camera's data (bpy.types.Camera).
            aspect_x: Frame width: resolution times pixel aspect.
            aspect_y: Frame height.

        Returns:
            The (horizontal, vertical) reach; not positive when the shift moves
            that side of the frame past the axis.
        """
        fit = data.sensor_fit
        if fit == "AUTO":
            fit = "HORIZONTAL" if aspect_x >= aspect_y else "VERTICAL"
        fitted = 0.5 if data.type == "ORTHO" else math.tan(data.angle / 2)
        if fit == "HORIZONTAL":
            across, up = fitted, fitted * aspect_y / aspect_x
        else:
            across, up = fitted * aspect_x / aspect_y, fitted
        return (
            across - 2 * fitted * abs(data.shift_x),
            up - 2 * fitted * abs(data.shift_y),
        )

    @staticmethod
    def _fitting_distance(radius: float, half_tangent: float) -> float:
        """
        Return how far a perspective camera stands to fit a sphere in its frame.

        Args:
            radius: The sphere's radius.
            half_tangent: The frame's reach from the view axis per unit of
                depth, on its narrower side.

        Returns:
            The distance from the camera to the sphere's center, with the
            framing margin.
        """
        return radius * _FRAMING_MARGIN / math.sin(math.atan(half_tangent))

    @staticmethod
    def _clip_around(data: Any, distance: float, radius: float) -> bool:
        """
        Widen a camera's clip range to span a sphere ``distance`` ahead of it.

        Args:
            data: The camera's data.
            distance: Depth of the sphere's center along the view axis.
            radius: The sphere's radius.

        Returns:
            Whether the clip range changed.
        """
        near, far = distance - radius, distance + radius
        changed = False
        if far > data.clip_end:
            data.clip_end = far + radius
            changed = True
        # A camera inside the sphere sees only part of it, whatever its clip_start.
        if 0.0 < near < data.clip_start:
            data.clip_start = near / 2
            changed = True
        return changed

    @staticmethod
    def _evaluate_transforms() -> None:
        """
        Bring every object's matrix_world up to date with its transform.

        Blender recomputes matrix_world when it evaluates the depsgraph:
        between events in a GUI, but only on demand in background Blender and
        in the bpy module (embedded mode). There a location, scale or parent
        written by an earlier call stays out of matrix_world until then. On
        Blender 5.0.1 in the background, get_camera_info reported a camera
        where it stood before set_object_transform moved it, focus_on_object
        put back the scale just written, and set_camera_view missed by as much
        as the camera's rig had just moved.
        """
        blender_module: Any = bpy
        blender_module.context.evaluated_depsgraph_get()

    @classmethod
    def _place_in_world(
        cls, obj: Any, location: List[float], rotation_euler: Sequence[float]
    ) -> Any:
        """
        Give an object a world-space location and XYZ Euler rotation, keeping its scale.

        Assigning ``matrix_world`` makes Blender solve the local transform
        through the parent and its inverse matrix, and store the rotation in
        the object's rotation_mode. Writing ``location`` and ``rotation_euler``
        instead places a parented object relative to its parent; a QUATERNION
        or AXIS_ANGLE object ignores rotation_euler, and another Euler order
        reads the angles in its own order. The object's constraints are not
        solved for: Blender applies them on top when it next evaluates the
        object.

        A mirrored object (an odd number of negative scales, its own or its
        parents') stays mirrored, on its X axis: a camera then still looks
        down its -Z with +Y up as asked, and its image is flipped left to
        right. ``Matrix.to_scale()`` of a mirrored matrix is negative on every
        axis, which would turn the camera to look backward.

        Args:
            obj: The object to place.
            location: World-space location [x, y, z].
            rotation_euler: World-space rotation [rx, ry, rz] in radians, XYZ order.

        Returns:
            The world matrix written, the placement before constraints.
        """
        mathutils = _mathutils()
        current = cls._matrix_before_constraints(obj)
        scale = [abs(s) for s in current.to_scale()]
        if current.is_negative:
            scale[0] = -scale[0]
        placement = mathutils.Matrix.LocRotScale(
            mathutils.Vector(location),
            mathutils.Euler(rotation_euler, "XYZ"),
            mathutils.Vector(scale),
        )
        obj.matrix_world = placement
        return placement

    @classmethod
    def _matrix_before_constraints(cls, obj: Any) -> Any:
        """
        Return the world matrix an object's parent and own transform give it.

        Blender evaluates an object's parent matrix times its
        matrix_parent_inverse and matrix_basis (location, rotation and scale,
        deltas included), then applies its constraints; matrix_world is the
        result. A placement that kept the scale of that result would write a
        constraint's scaling into the object's own scale, where the
        constraint scales it again: on Blender 5.0.1 every set_camera_view
        doubled the scale of a camera whose Child Of target had been scaled
        by 2 after the constraint was set up.

        Args:
            obj: The object, its transforms evaluated.

        Returns:
            The matrix; matrix_world when no constraint acts on the object, or
            when only Blender computes its parent matrix
            (_matrix_without_constraints).
        """
        if not cls._active_constraints(obj):
            return obj.matrix_world
        own = cls._matrix_without_constraints(obj)
        return obj.matrix_world if own is None else own

    @staticmethod
    def _matrix_without_constraints(obj: Any) -> Any:
        """
        Compose the world matrix an object has before its constraints apply.

        That is its parent's matrix times its matrix_parent_inverse and
        matrix_basis (location, rotation and scale, deltas included), or
        matrix_basis alone without a parent. An object's matrix_basis reads
        its transform whenever it is used, as matrices from bpy properties
        do, so the matrix returned is a copy: on Blender 5.0.1 one read
        before an evaluation held a driver's value after it.

        The parent's matrix is its matrix_world for parent type OBJECT, and
        for ARMATURE and LATTICE too: Blender 5.0.1 (BKE_object_get_parent_matrix,
        PARSKEL) takes the parent's world matrix for them, and their deform
        moves only the child's geometry, so a posed bone leaves the object
        where it is. On Blender 5.0.1 a camera without constraints, parented
        so to a posed armature or to a lattice, both moved, was composed
        exactly where a render put it.

        Args:
            obj: The object, its transforms evaluated.

        Returns:
            The matrix, or None when the object has a curve parent (it can
            follow the curve's path) or parent type BONE, VERTEX or VERTEX_3,
            whose parent matrix only Blender computes.
        """
        parent = obj.parent
        if parent is None:
            return obj.matrix_basis.copy()
        if obj.parent_type in ("ARMATURE", "LATTICE") or (
            obj.parent_type == "OBJECT" and parent.type != "CURVE"
        ):
            return parent.matrix_world @ obj.matrix_parent_inverse @ obj.matrix_basis
        return None

    @classmethod
    def _active_constraints(cls, obj: Any) -> List[Dict[str, str]]:
        """
        Return the constraints Blender applies when it evaluates an object.

        A muted constraint, one without influence, one Blender reports
        invalid (a Track To without a target, for instance) and one whose
        target was deleted since (_lacks_target) change nothing.

        Args:
            obj: The object.

        Returns:
            Each constraint's name and type (TRACK_TO, CHILD_OF, ...), in
            stack order.
        """
        return [
            {"name": constraint.name, "type": constraint.type}
            for constraint in obj.constraints
            if cls._applies(constraint)
        ]

    @classmethod
    def _applies(cls, constraint: Any) -> bool:
        """
        Whether Blender applies a constraint when it evaluates its owner.

        Args:
            constraint: The constraint.

        Returns:
            False for a muted constraint, one without influence, one Blender
            reports invalid and one that lacks its target (_lacks_target).
        """
        return (
            not constraint.mute
            and constraint.influence > 0.0
            and constraint.is_valid
            and not cls._lacks_target(constraint)
        )

    @staticmethod
    def _lacks_target(constraint: Any) -> bool:
        """
        Whether a constraint Blender reports valid lacks the target it acts with.

        Blender checks a constraint's targets when the constraint is edited,
        not when a target goes away. On Blender 5.0.1, deleting a target
        object leaves ``target`` None, and deleting the bone an Armature
        constraint's target names leaves that ``subtarget`` naming nothing;
        either way ``is_valid`` stays True until the next edit, while the
        constraint does nothing. A Pivot without a target pivots about the
        object itself, and an Action that uses eval_time needs none. An
        Armature constraint skips a target without weight. A target that no
        scene links still acts, and so does another constraint whose bone or
        vertex group (``subtarget``) is gone: Blender reads the target object
        instead.

        Args:
            constraint: The constraint.

        Returns:
            True when the constraint has no target object or, for an Armature
            constraint, a target with weight has no object or bone.
        """
        if constraint.type == "ARMATURE":
            return any(
                entry.weight > 0.0
                and (
                    entry.target is None
                    or entry.subtarget not in entry.target.data.bones
                )
                for entry in constraint.targets
            )
        if constraint.type == "PIVOT" or (
            constraint.type == "ACTION" and constraint.use_eval_time
        ):
            return False
        return hasattr(constraint, "target") and constraint.target is None

    def _placement_report(
        self, camera: Any, placement: Any, stored: Any
    ) -> Dict[str, Any]:
        """
        Say whether Blender evaluates a placed camera where it was placed.

        Call once the depsgraph has been evaluated after the placement, so
        ``matrix_world`` holds the evaluated matrix. Constraints apply on top
        of the transform a placement writes (a Track To turns the camera to
        its target, a Copy Location moves it), drivers on the transform
        overwrite it, and under a parent with non-uniform scale no transform
        carries every rotation: in each case the camera ends up away from the
        placement while its transform holds it. Blender also stores the
        placement in single precision, which near gimbal lock, or for an
        axis-angle near zero, turns the camera off it (_stores_placement).
        So the camera counts as held when
        Blender evaluates it at the placement, or at the transform it stored
        for the placement when nothing changed that since and its constraints
        leave it as it is.

        Args:
            camera: The placed camera object.
            placement: The world matrix written (_place_in_world).
            stored: The camera's own world matrix read right after the
                placement (_matrix_without_constraints), None when only
                Blender computes its parent matrix.

        Returns:
            Dictionary with placement_overridden (whether the evaluated
            location, view or up direction, or mirroring differs from the
            placement and from the transform stored for it),
            constraints_applied (_active_constraints) and warning, which
            explains an override (_override_warning) and is None otherwise.
        """
        evaluated = camera.matrix_world
        extent = self._placement_extent(camera, placement, evaluated)
        turn = self._placement_turn(camera, placement)
        own = self._matrix_without_constraints(camera)
        if own is None or stored is None:
            moved = held = False
        else:
            moved = not self._stores_placement(camera, placement, stored, own, extent)
            held = self._same_view(own, evaluated, extent, turn)
        overridden = not (
            self._same_view(placement, evaluated, extent, turn) or (held and not moved)
        )
        constraints = self._active_constraints(camera)
        warning = (
            self._override_warning(camera, constraints, moved, held)
            if overridden
            else None
        )
        return {
            "placement_overridden": overridden,
            "constraints_applied": constraints,
            "warning": warning,
        }

    @classmethod
    def _placement_extent(cls, camera: Any, placement: Any, evaluated: Any) -> float:
        """
        Return the farthest distance from the origin that placing a camera computes with.

        Blender's matrices and an object's location are single precision, so
        the evaluated location is only as exact as the longest translation in
        the arithmetic: the placement and the evaluated location, and between
        them the parent's matrix, its product with matrix_parent_inverse, the
        camera's own location and the targets of the constraints Blender
        applies. Those can be kilometres long while both ends lie near the
        origin: a camera parented to a rig 27 km out, the rig then moved back,
        keeps a 27 km own location, which single precision rounds by up to a
        millimetre (by 2 cm 100 km out on Blender 5.0.1). A Child Of
        multiplies by its target's matrix, which rounds as much when the
        target stands kilometres out, even when the target has not moved
        since the constraint's inverse was set and the camera stays in place.
        So does an Armature constraint with its armatures' matrices, and a
        constraint whose target names a bone or vertex group (subtarget) with
        that bone's or group's matrix, which can stand far from the target's
        origin: on Blender 5.0.1, without them, 37 of 40 cameras held by an
        Armature constraint on a resting armature 300 m out, and 31 of 40
        held by a satisfied Child Of on a vertex group 300 m from its mesh's
        origin, were reported overridden. A constraint that only turns or
        scales the camera (_IN_PLACE_CONSTRAINT_TYPES) leaves its location as
        it was, so its target is left out: a Copy Scale whose target stands
        50 km out does not widen the tolerance.

        Args:
            camera: The placed camera.
            placement: The world matrix written (_place_in_world).
            evaluated: The camera's evaluated world matrix.

        Returns:
            The distance, in world units.
        """
        lengths = [
            math.hypot(*placement.translation),
            math.hypot(*evaluated.translation),
        ]
        # How long one unit of the camera's own location is in the world.
        reach = 1.0
        parent = camera.parent
        if parent is not None:
            to_world = parent.matrix_world @ camera.matrix_parent_inverse
            lengths.append(math.hypot(*parent.matrix_world.translation))
            lengths.append(math.hypot(*to_world.translation))
            reach = max(abs(s) for s in to_world.to_scale())
        lengths.append(reach * math.hypot(*camera.location))
        for constraint in camera.constraints:
            if constraint.type in _IN_PLACE_CONSTRAINT_TYPES or not cls._applies(
                constraint
            ):
                continue
            for target, subtarget in cls._constraint_targets(constraint):
                lengths.append(math.hypot(*target.matrix_world.translation))
                if subtarget:
                    # Blender computes with the posed bone, or the vertex
                    # group's vertices, which the target's bounding box holds.
                    lengths.extend(
                        math.hypot(*corner) for corner in cls._world_bound_box(target)
                    )
        return max(lengths)

    @staticmethod
    def _constraint_targets(constraint: Any) -> List[Tuple[Any, str]]:
        """
        Return the objects a constraint computes with, and the bone or vertex group each names.

        An Armature constraint lists its bones in ``targets`` and skips one
        without weight; the other constraints with a target have one
        ``target`` and, most of them, a ``subtarget``. Limit Location and
        the like have none.

        Args:
            constraint: The constraint.

        Returns:
            (target object, subtarget) pairs, the subtarget "" when none is
            named; a target that is None (a Pivot or an Action that uses
            eval_time acts without one) is left out.
        """
        if constraint.type == "ARMATURE":
            pairs = [
                (entry.target, entry.subtarget)
                for entry in constraint.targets
                if entry.weight > 0.0
            ]
        else:
            pairs = [
                (
                    getattr(constraint, "target", None),
                    getattr(constraint, "subtarget", ""),
                )
            ]
        return [
            (target, subtarget) for target, subtarget in pairs if target is not None
        ]

    @classmethod
    def _placement_turn(cls, camera: Any, placement: Any) -> Tuple[float, float]:
        """
        Return how far single precision can turn a placed camera's up and view axes.

        The rounding of the rotation a camera stores does not grow with the
        camera's distance from the origin (_PLACEMENT_ANGLE_TOLERANCE), but a
        constraint that aims it (Track To, Damped Track, Locked Track, Stretch
        To) takes its direction from the camera's location to the target's,
        both single precision. That aim misses the one the placement asked
        for by what moved the camera off the placement across the aim (its
        own rounding, millimetres where its own location is kilometres long)
        and by an epsilon of each coordinate of the two locations (rounding
        the requested location, and Blender's subtraction): the aim turns by
        that over the lever, the distance between them. Only the aim's own
        two locations count: the target of another constraint, however far
        out, does not widen it. A Track To also rolls the camera, taking the
        roll from the aim (_track_to_roll). On Blender 5.0.1 a Track To aiming
        a camera where it was placed turned it by up to 0.13 rad, for 242 of
        1,200 cameras focused on the object it tracks 1 m to 100 km out. A
        bone or vertex group that a target names (subtarget) is left out, as
        only Blender computes the point it aims at.

        An aim with no direction adds nothing: the camera standing exactly at
        its target, Blender's Track To turns it to look straight down, +Y up,
        while Damped Track and Locked Track leave it as it is, and either
        pose does not follow the placement, so a placement off it is an
        override.

        Args:
            camera: The placed camera, its transforms evaluated.
            placement: The world matrix written (_place_in_world).

        Returns:
            The angles, in radians, that the up and view axes may turn by:
            _PLACEMENT_ANGLE_TOLERANCE plus the largest roll and turn of the
            aim constraints Blender applies.
        """
        location = list(camera.matrix_world.translation)
        moved = [c - p for c, p in zip(location, placement.translation)]
        up = view = 0.0
        aims = (
            constraint
            for constraint in camera.constraints
            if constraint.type in _AIM_CONSTRAINT_TYPES and cls._applies(constraint)
        )
        for constraint in aims:
            for target, subtarget in cls._constraint_targets(constraint):
                if subtarget:
                    continue
                world = target.matrix_world
                aimed = list(world.translation)
                aim = [t - c for t, c in zip(aimed, location)]
                lever = math.hypot(*aim)
                if lever == 0.0:
                    continue
                rounding = [
                    _FLOAT32_EPSILON * (abs(c) + abs(t))
                    for c, t in zip(location, aimed)
                ]
                along = sum(m * a for m, a in zip(moved, aim)) / lever**2
                aside = math.hypot(*(m - along * a for m, a in zip(moved, aim)))
                turn = (aside + math.hypot(*rounding)) / lever + _FLOAT32_EPSILON
                roll = turn
                if constraint.type == "TRACK_TO":
                    if constraint.use_target_z:
                        z = [world[row][2] for row in range(3)]
                        tilt = _TARGET_AXIS_ROUNDING
                    else:
                        z, tilt = [0.0, 0.0, 1.0], 0.0
                    roll += cls._track_to_roll(aim, moved, rounding, z, tilt)
                up = max(up, roll)
                view = max(view, turn)
        return _PLACEMENT_ANGLE_TOLERANCE + up, _PLACEMENT_ANGLE_TOLERANCE + view

    @staticmethod
    def _track_to_roll(
        aim: List[float],
        moved: List[float],
        rounding: List[float],
        z: List[float],
        tilt: float,
    ) -> float:
        """
        Return how far single precision can roll a camera that a Track To aims.

        The Track To keeps the camera's up axis as near world Z (the target's
        Z axis with Target Z) as it can, so it takes the roll from the
        direction of the aim's part across that Z. The camera's move off the
        placement and the rounding of the two locations turn that direction
        by their error across the part over the part's length, and once their
        error along it can reach its length, the part can point any way: the
        roll is lost. Blender 5.0.1's Track To computes the roll itself to
        within an epsilon however close the aim comes to that Z (0.71 epsilon
        at worst over 2,334 exact aims 1e-9 to 0.1 rad from it), while its up
        axis leans out of square with its view by up to the smaller of that
        angle and epsilon over it (3.5e-4 rad at worst). So a target 1e-5 m
        beside a camera 10 m straight above it, both near the origin, fixes
        the roll, while a millimetre beside a camera 10 km out, where single
        precision moves both locations by more, does not. The target's Z axis
        is single precision too: turned by ``tilt``, it moves the aim's part
        by that times the aim's height along it, so 1e-4 rad from it the roll
        misses by about tilt over that angle (on Blender 5.0.1, 4 of 1,200
        cameras placed within 1e-4 rad of the target's Z axis were rolled by
        0.7 to 1 epsilon over its tangent).

        Aimed exactly along that Z, the Track To has no direction to take the
        roll from and gives the camera a fixed one: on Blender 5.0.1 it rolled
        every camera straight above or below its target to world +Y up,
        whatever roll the camera was placed with. That roll does not follow
        the placement, so it adds nothing here, also where rounding lined the
        aim up: a camera within single precision of straight above its target
        (a millimetre 10 km out) is rolled by up to half a turn.

        Args:
            aim: From the camera's location to the target's.
            moved: From the placement's location to the camera's.
            rounding: How far rounding can move the aim along each axis.
            z: The up direction the Track To keeps the camera's up axis near.
            tilt: How far rounding can turn that direction, in radians.

        Returns:
            The roll in radians, beyond the turn of the aim: math.pi when
            rounding loses the roll's direction.
        """
        length = math.hypot(*z)
        if not length:
            return 0.0
        z = [value / length for value in z]
        height = sum(a * u for a, u in zip(aim, z))
        part = [a - height * u for a, u in zip(aim, z)]
        across = math.hypot(*part)
        if not across:
            return 0.0
        heading = [p / across for p in part]
        # Across both the part and that Z: the way a roll moves the aim's part.
        swing = [
            heading[1] * z[2] - heading[2] * z[1],
            heading[2] * z[0] - heading[0] * z[2],
            heading[0] * z[1] - heading[1] * z[0],
        ]

        def error(direction: List[float]) -> float:
            """How far the move and rounding can shift the part along ``direction``."""
            shift = abs(sum(m * d for m, d in zip(moved, direction)))
            spread = sum(r * abs(d) for r, d in zip(rounding, direction))
            return shift + spread + tilt * abs(height)

        slip = error(heading)
        if slip >= across:
            return math.pi
        sine = across / math.hypot(*aim)
        lean = min(sine, _FLOAT32_EPSILON / sine)
        return error(swing) / (across - slip) + _FLOAT32_EPSILON + lean

    @staticmethod
    def _override_warning(
        camera: Any,
        constraints: List[Dict[str, str]],
        moved: bool,
        held: bool,
    ) -> str:
        """
        Explain why Blender evaluates a placed camera away from the placement.

        Constraints act on the camera's own transform (_matrix_without_constraints),
        which misses the placement itself when a driver overwrote it or a
        parent with non-uniform scale shears it. Comparing that transform with
        the placement and with the evaluated matrix tells the causes apart, so
        a constraint that leaves the pose alone (a Limit Distance it meets) is
        not blamed for what a driver did. When only Blender knows the parent
        matrix (a bone, vertices, a curve), the active constraints are named.

        Args:
            camera: The placed camera, its transforms evaluated.
            constraints: The camera's active constraints (_active_constraints).
            moved: Whether its own transform misses the placement as Blender
                stores it (_stores_placement); False when only Blender knows
                that transform.
            held: Whether its constraints leave its own transform as it is.

        Returns:
            The warning.
        """
        transform = (
            "a driver on its transform, or a parent with non-uniform scale that "
            "cannot carry this rotation, changes it"
        )
        if not constraints:
            return (
                f"Blender evaluates camera {camera.name!r} away from the requested "
                f"placement although no constraint acts on it: {transform}. The "
                "reported pose is where Blender evaluates the camera."
            )
        names = ", ".join(f"'{c['name']}' ({c['type']})" for c in constraints)
        if moved and held:
            return (
                f"Blender evaluates camera {camera.name!r} away from the requested "
                f"placement before its constraints apply, and they ({names}) leave "
                f"that pose as it is: {transform}. The reported pose is where Blender "
                "evaluates the camera."
            )
        if moved:
            cause = f"before they apply, {transform} as well, and"
            remedy = "and remove that driver or parent scale, "
        else:
            cause = "its transform holds the placement and"
            remedy = ""
        return (
            f"Constraints on camera {camera.name!r} override the requested placement: "
            f"{names}. The reported pose is where Blender evaluates the camera; {cause} "
            "the constraints were left unchanged. To place it exactly, mute them (the "
            "eye icon in Object Constraint Properties, or constraint.mute = True) or "
            f"set their influence to 0, {remedy}then place the camera again."
        )

    @classmethod
    def _stores_placement(
        cls, camera: Any, placement: Any, stored: Any, own: Any, extent: float
    ) -> bool:
        """
        Whether a placed camera's own transform is the placement as Blender stores it.

        Blender solves the placement through the camera's parent and stores it
        in single precision, the rotation in the camera's rotation mode, by a
        conversion that is ill-conditioned for an Euler order near gimbal
        lock and for an axis-angle near zero. On Blender 5.0.1, with nothing
        acting on them, that put 141 of 1,200 cameras placed 4e-5 to 1.1e-3
        rad from gimbal lock up to 4.7e-3 rad off the placement, and 160 of
        1,200 axis-angle cameras turned by 1e-4 to 1.7e-3 rad up to 5.6e-4
        rad off (the cosine of half the angle rounds to 1, and the rotation
        is lost). Read right after the placement, before any evaluation, the
        stored transform is what that conversion gave, so its rotation is
        held to the placement's only under a parent that can shear it
        (_shears), where Blender's storage can miss by any angle. Its location
        and mirroring keep the fixed tolerances. Evaluation recomputes the same
        matrix from the same stored values, so a change beyond rounding there
        is a driver on the transform or a parent that moved.

        Args:
            camera: The placed camera, its transforms evaluated.
            placement: The world matrix written (_place_in_world).
            stored: The camera's own world matrix read right after the
                placement (_matrix_without_constraints).
            own: That matrix after the evaluation.
            extent: The farthest distance involved (_placement_extent).

        Returns:
            True when the stored transform holds the placement as far as its
            storage allows, and the evaluation left it as stored.
        """
        storage = (
            (_PLACEMENT_ANGLE_TOLERANCE, _PLACEMENT_ANGLE_TOLERANCE)
            if cls._shears(camera)
            else (math.inf, math.inf)
        )
        rounding = (_PLACEMENT_LOCATION_PRECISION, _PLACEMENT_LOCATION_PRECISION)
        stores = cls._same_view(placement, stored, extent, storage)
        return stores and cls._same_view(stored, own, extent, rounding)

    @staticmethod
    def _shears(camera: Any) -> bool:
        """
        Whether a camera's parent can shear its placement: scale it unevenly or skew it.

        A camera's own transform is a location, a rotation and a scale along
        its own axes, so its parent (matrix_world times the camera's
        matrix_parent_inverse) carries every placement only as a rotation and
        a uniform scale. Under any other parent matrix the placement can be
        sheared, and Blender's storage then misses it by however much the
        parent stretches the shear: on Blender 5.0.1, under parents scaled 22
        to 1,000 times as much along one axis, cameras turned 1.2e-7 to
        8.5e-7 rad off the parent's axes were evaluated up to 0.53 rad off
        their placement, so such a placement keeps the fixed tolerances.
        Rounding leaves a uniform parent uniform to a few epsilons.

        Args:
            camera: The placed camera; its parent, if any, gives it that
                parent's matrix_world (_matrix_without_constraints).

        Returns:
            True when the parent matrix's axes differ in length, or lean off
            right angles, by more than _PLACEMENT_LOCATION_PRECISION.
        """
        parent = camera.parent
        if parent is None:
            return False
        to_parent = parent.matrix_world @ camera.matrix_parent_inverse
        axes = [[to_parent[row][col] for row in range(3)] for col in range(3)]
        lengths = [math.hypot(*axis) for axis in axes]
        if not all(lengths):
            return True
        uneven = (max(lengths) - min(lengths)) / max(lengths)
        skew: float = max(
            abs(sum(a * b for a, b in zip(axes[i], axes[j])))
            / (lengths[i] * lengths[j])
            for i, j in ((0, 1), (0, 2), (1, 2))
        )
        return max(uneven, skew) > _PLACEMENT_LOCATION_PRECISION

    @staticmethod
    def _same_view(
        first: Any,
        second: Any,
        extent: float,
        turn: Tuple[float, float] = (
            _PLACEMENT_ANGLE_TOLERANCE,
            _PLACEMENT_ANGLE_TOLERANCE,
        ),
    ) -> bool:
        """
        Whether two world matrices put a camera in one place, looking one way.

        Compares the locations, the directions of the local Y (up) and Z
        (opposite the view) axes, which the X-axis mirror of _place_in_world
        leaves alone, and whether the matrices mirror. The scale is left out:
        it does not change what a camera sees.

        Args:
            first: A world matrix.
            second: The other world matrix.
            extent: The farthest distance from the origin involved
                (_placement_extent), which single precision blurs.
            turn: The angles the up and view axes may differ by
                (_placement_turn; math.inf leaves an axis out);
                _PLACEMENT_ANGLE_TOLERANCE by default, for matrices no
                constraint comes between.

        Returns:
            True when they agree within the placement tolerances.
        """
        if bool(first.is_negative) != bool(second.is_negative):
            return False
        tolerance = _PLACEMENT_LOCATION_PRECISION * extent
        if math.dist(first.translation, second.translation) > tolerance:
            return False
        for axis, angle in zip((1, 2), turn):
            a = [first[row][axis] for row in range(3)]
            b = [second[row][axis] for row in range(3)]
            cross = (
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            )
            dot = sum(x * y for x, y in zip(a, b))
            if math.atan2(math.hypot(*cross), dot) > angle:
                return False
        return True

    @staticmethod
    def _world_placement(
        obj: Any, near: Optional[Sequence[float]] = None, matrix: Any = None
    ) -> Tuple[List[float], List[float]]:
        """
        Return an object's world-space location and rotation as an XYZ Euler.

        Several Euler triples give the same rotation; the one nearest ``near``
        (the object's own rotation_euler by default) is returned, so a
        rotation set as given reads back as given. A mirrored object counts
        as mirrored on its X axis, as _place_in_world places it.

        Args:
            obj: The object.
            near: Euler angles the result stays close to.
            matrix: The world matrix to read; the object's matrix_world when None.

        Returns:
            The location [x, y, z] and rotation [rx, ry, rz] in radians.
        """
        mathutils = _mathutils()
        if matrix is None:
            matrix = obj.matrix_world
        if matrix.is_negative:
            # Not a rotation: its Euler angles would describe no orientation.
            matrix = matrix @ mathutils.Matrix.Scale(-1.0, 4, (1.0, 0.0, 0.0))
        compatible = mathutils.Euler(
            obj.rotation_euler if near is None else near, "XYZ"
        )
        return list(matrix.translation), list(matrix.to_euler("XYZ", compatible))

    def _placement_check(
        self,
        camera: Any,
        placement: Any,
        stored: Any,
        disablers: List[str],
        before: Optional[List[List[float]]],
    ) -> Dict[str, Any]:
        """
        Report whether Blender keeps a placed camera at the placement, when it can tell.

        The check compares the placement with the matrix_world Blender
        evaluates for the camera (_placement_report). Blender's viewport
        evaluates a camera disabled in viewports (_viewport_disablers) only
        when something visible depends on it: in full when that reads its
        transform (a child, a constraint or a driver variable that reads it),
        in part when only a property, and in part too while its visibility is
        animated. A camera evaluated in part, or not at all, keeps the
        matrix_world the placement wrote, or one from an earlier evaluation,
        whatever its constraints, drivers and parent do in a render. Neither
        depsgraph.ids nor is_evaluated tells it from one evaluated in full: on
        Blender 5.0.1 depsgraph.ids, which lists only directly visible IDs,
        left out a disabled camera a visible object's constraint targets, and
        a camera evaluated in part reported is_evaluated True. So a disabled
        camera is checked only when the depsgraph holds an evaluated copy of
        it whose pose changed with the placement; the check is skipped
        otherwise, rather than compared with a stale matrix. The scene camera
        and a marker's camera count as disabled only while their
        hide_viewport is animated.

        Args:
            camera: The placed camera, the depsgraph evaluated since.
            placement: The world matrix written (_place_in_world).
            stored: The camera's own world matrix read right after the
                placement (_placement_report).
            disablers: The settings that disable the camera (_viewport_disablers).
            before: Its evaluated pose before the placement (_evaluated_pose),
                read when ``disablers`` is not empty.

        Returns:
            The _placement_report with override_check "done". Otherwise
            override_check "skipped", placement_overridden None,
            constraints_applied (the constraints that apply when Blender
            evaluates it) and a warning that names the settings that disable
            it, and for the scene camera or a marker's camera its animated
            hide_viewport.
        """
        after = self._evaluated_pose(camera) if disablers else None
        if not disablers or (after is not None and after != before):
            return {
                **self._placement_report(camera, placement, stored),
                "override_check": "done",
            }
        if after is None:
            unchecked = "Blender did not evaluate it after the placement"
        else:
            unchecked = (
                "Its evaluated pose did not change with the placement: Blender "
                "evaluates such a camera only in part when nothing visible reads its "
                "transform (a driver reads one of its properties, or its visibility is "
                "animated), and a placement that leaves the pose as it was (where it "
                "already is, or under constraints that set its whole transform) "
                "changes nothing either"
            )
        constraints = self._active_constraints(camera)
        if constraints:
            names = ", ".join(f"'{c['name']}' ({c['type']})" for c in constraints)
            effect = (
                f"Its constraints ({names}) apply whenever Blender evaluates it, as a "
                "render through it does, and can move it off the placement; to keep it "
                "there, mute them or set their influence to 0."
            )
        else:
            effect = (
                "No constraint acts on it, but a driver on its transform or a parent "
                "with non-uniform scale would still apply whenever Blender evaluates it."
            )
        parent = (
            " Its transform was solved through the last matrix Blender evaluated for "
            "its parent, so it lands elsewhere if that parent has moved since."
            if camera.parent is not None
            else ""
        )
        role = self._camera_role(camera)
        if role is None:
            cause = (
                "as it is neither the scene camera nor a timeline marker's camera and "
                f"{' and '.join(disablers)}"
            )
            # As the scene camera too, a camera whose hide_viewport is animated
            # is evaluated only while it is enabled in viewports
            # (_viewport_disablers).
            remedy = "make it the scene camera" + (
                " and remove the animation of its hide_viewport"
                if self._animates_visibility(camera)
                else ""
            )
        else:
            cause = (
                f"although it is {role}, as {' and '.join(disablers)}: Blender "
                "evaluates a camera whose hide_viewport is animated only while it is "
                "enabled in viewports"
            )
            remedy = "remove the animation of its hide_viewport"
        return {
            "placement_overridden": None,
            "override_check": "skipped",
            "constraints_applied": constraints,
            "warning": (
                f"Blender may not evaluate camera {camera.name!r} for the viewport, "
                f"{cause}. {unchecked}, so the placement was not checked. The reported "
                f"pose is the requested placement, unchecked.{parent} {effect} To check "
                f"it, change those settings or {remedy}, then place it again."
            ),
        }

    @staticmethod
    def _evaluated_pose(obj: Any) -> Optional[List[List[float]]]:
        """
        Return the world matrix of the object's copy that the viewport depsgraph evaluated.

        Args:
            obj: The object, the depsgraph evaluated since its last change.

        Returns:
            The matrix as rows of floats, a copy: Blender's matrix_world
            follows later evaluations. None when the depsgraph holds no
            evaluated copy of the object: evaluated_get then gives back the
            original, whose is_evaluated is False.
        """
        blender_module: Any = bpy
        depsgraph = blender_module.context.evaluated_depsgraph_get()
        evaluated = obj.evaluated_get(depsgraph)
        if not evaluated.is_evaluated:
            return None
        matrix = evaluated.matrix_world
        return [[float(matrix[row][column]) for column in range(4)] for row in range(4)]

    def _viewport_disablers(self, obj: Any) -> List[str]:
        """
        Return the settings that keep Blender from evaluating an object for the viewport.

        evaluated_depsgraph_get() evaluates the scene camera, every camera a
        timeline marker binds (on Blender 5.0.1 even one disabled in viewports
        or in an excluded collection, whatever the current frame), and the view
        layer's objects that are enabled in viewports: not disabled in
        viewports themselves (hide_viewport, the monitor icon), and in a
        collection that its own checkbox does not exclude from the view layer
        (excluding a collection unchecks its children too, but a child checked
        again counts) and that is not disabled in viewports, nor are its
        parent collections. Hiding in the view layer (the eye icon) leaves an
        object evaluated. Blender evaluates another object only when something
        visible depends on it, and sometimes only in part (_placement_check).

        The scene camera and a marker's camera lose that exception while their
        hide_viewport is animated (_animates_visibility): Blender then builds
        them through their base in the view layer, which each collection not
        excluded from it gives them, and evaluates them only while that base
        is enabled in viewports. On Blender 5.0.1 such a camera, while its
        hide_viewport was on or the collections giving it a base were disabled
        in viewports, kept the matrix_world of its last evaluation after a
        placement, while a render through it applied its constraints at the
        placement; one without a base, in no collection of the view layer or
        only in excluded ones, was evaluated.

        Args:
            obj: The object.

        Returns:
            One description per setting that disables it; empty for an object
            enabled in viewports, and for the scene camera and a camera a
            timeline marker binds unless their hide_viewport is animated.
        """
        role = self._camera_role(obj)
        if role is not None and not self._animates_visibility(obj):
            return []
        blender_module: Any = bpy
        view_layer = blender_module.context.view_layer
        owners = list(obj.users_collection)
        root = view_layer.layer_collection
        # The scene's own collection can be neither excluded nor disabled.
        enabled = has_base = root.collection in owners
        blocked: List[str] = []
        # Depth first, in outliner order: (layer collection, disabled ancestor).
        pending: List[Tuple[Any, Optional[str]]] = [
            (child, None) for child in reversed(list(root.children))
        ]
        while pending:
            layer, disabled_by = pending.pop()
            collection = layer.collection
            if disabled_by is None and collection.hide_viewport:
                disabled_by = collection.name
            if collection in owners:
                has_base = has_base or not layer.exclude
                if not layer.exclude and disabled_by is None:
                    enabled = True
                else:
                    reason = (
                        f"its collection {collection.name!r} is excluded from view "
                        f"layer {view_layer.name!r}"
                        if layer.exclude
                        else f"collection {disabled_by!r} is disabled in viewports"
                    )
                    if reason not in blocked:
                        blocked.append(reason)
            pending.extend(
                (child, disabled_by) for child in reversed(list(layer.children))
            )
        if role is not None:
            if not has_base or (enabled and not obj.hide_viewport):
                return []
            animated = "its hide_viewport is animated" + (
                " and on at this frame" if obj.hide_viewport else ""
            )
            return [animated] + ([] if enabled else blocked)
        reasons = (
            ["it is disabled in viewports (hide_viewport)"] if obj.hide_viewport else []
        )
        if not enabled:
            reasons.extend(
                blocked or [f"it is in no collection of view layer {view_layer.name!r}"]
            )
        return reasons

    def _scene_dependents(self) -> Tuple[Dict[Any, List[Any]], Dict[Any, List[Any]]]:
        """
        Map each object to the objects of the scene that compute with its world matrix.

        A child needs its parent's world matrix and the owner of a constraint
        Blender applies its targets', whenever Blender evaluates the child's or
        the owner's transform. A child of a bone (parent type BONE) needs too
        the targets of the constraints of that bone and of every bone above
        it, whether its armature is enabled or not: Blender 5.0.1 relates the
        child to its bone (build_object_parent) and each bone to its parent
        bone (build_rig). An armature needs the targets of its bones'
        constraints for its pose, and an object those its modifiers read
        (_modifier_transform_sources) for its geometry, which Blender
        evaluates with the whole object. An IK or a Spline IK constraint
        counts only where Blender relates its targets (_relates_targets).
        get_camera_info scans the scene once for a camera and its parent: on
        Blender 5.0.1 a scan of 20 000 objects, 2 000 of them meshes with two
        modifiers, took 16 ms. Only an armature has a pose, and only the types
        of _MODIFIER_OWNER_TYPES a modifier stack; reading every object's took
        6 ms more. Each bone's chain is read once, however many children its
        bones have: 5 000 empties parented to the bones of a rig of 1 000
        constrained bones added 5 ms. A child of a bone is listed under each
        target of its chain, though: with a distinct target on each bone,
        5 000 empties on the bones of ten chains of 100 added 16 ms, and on
        the last bone of one chain of 1 000, 180 ms.

        Returns:
            (dependents, readers): for each object, the children, children of
            bones and constraint owners that need it; and the armatures and
            modifier owners that need it.
        """
        blender_module: Any = bpy
        dependents: Dict[Any, List[Any]] = {}
        readers: Dict[Any, List[Any]] = {}
        # The constraint targets of a pose bone and the bones above it, by
        # (armature, bone name).
        chains: Dict[Tuple[Any, str], List[Any]] = {}

        def targets(constraints: Any, bone: Any = None) -> List[Any]:
            return [
                target
                for constraint in constraints
                if self._applies(constraint) and self._relates_targets(constraint, bone)
                for target, _ in self._constraint_targets(constraint)
            ]

        def bone_targets(armature: Any, name: str) -> List[Any]:
            if (armature, name) in chains:
                return chains[(armature, name)]
            pose = armature.pose if armature.type == "ARMATURE" else None
            bone = None if pose is None else pose.bones.get(name)
            below: List[Any] = []
            while bone is not None and (armature, bone.name) not in chains:
                below.append(bone)
                bone = bone.parent
            found = [] if bone is None else chains[(armature, bone.name)]
            for chained in reversed(below):
                # Each target once, however many bones of the chain it has.
                found = list(
                    dict.fromkeys(targets(chained.constraints, chained) + found)
                )
                chains[(armature, chained.name)] = found
            # Also for a bone the pose lacks, which needs nothing.
            chains[(armature, name)] = found
            return found

        for other in blender_module.context.scene.objects:
            parent = other.parent
            sources = [] if parent is None else [parent]
            if parent is not None and other.parent_type == "BONE":
                sources.extend(bone_targets(parent, other.parent_bone))
            for source in sources + targets(other.constraints):
                dependents.setdefault(source, []).append(other)
            kind = other.type
            if kind == "ARMATURE":
                pose = other.pose
                bones = [] if pose is None else pose.bones
                read = [s for bone in bones for s in targets(bone.constraints, bone)]
            elif kind in _MODIFIER_OWNER_TYPES:
                read = [
                    source
                    for modifier in other.modifiers
                    for source in self._modifier_transform_sources(modifier)
                ]
            else:
                continue
            for source in read:
                readers.setdefault(source, []).append(other)
        return dependents, readers

    @staticmethod
    def _relates_targets(constraint: Any, bone: Any) -> bool:
        """
        Whether Blender relates a constraint it applies to the transform of its targets.

        Blender 5.0.1 relates the targets of an IK or a Spline IK constraint
        to the IK solver of its armature's pose (build_ik_pose,
        build_splineik_pose), not to the constraint stack, and holds them in
        the depsgraph either way. So it relates them to nothing on an object,
        which has no pose, nor for an IK whose use_tail is off on a bone
        without a parent: that IK has no chain, so no solver. On Blender 5.0.1
        a disabled camera kept a matrix_world 11 m from a render when a
        visible empty's IK targeted it, or its Spline IK a disabled curve
        parented to the camera, and when an IK without a chain targeted it
        from a bone of a visible armature, or of a disabled one with a visible
        empty parented to that bone or to the bone below it. With use_tail
        on, on a bone with a parent, or for a bone's Spline IK, the camera was
        evaluated in full.

        Args:
            constraint: The constraint.
            bone: The pose bone it is on, or None for an object's constraint.

        Returns:
            False for those IK and Spline IK constraints, True otherwise.
        """
        kind = constraint.type
        if kind not in ("IK", "SPLINE_IK"):
            return True
        if bone is None:
            return False
        return kind == "SPLINE_IK" or constraint.use_tail or bone.parent is not None

    @classmethod
    def _modifier_transform_sources(cls, modifier: Any) -> List[Any]:
        """
        Return the objects whose world matrix a modifier reads.

        Blender 5.0.1 relates a modifier to the transform of each object it
        reads (its update_depsgraph) and evaluates that transform whenever it
        evaluates the modifier's object, unless the modifier is disabled in
        viewports. Most types read the object a setting names whenever one is
        set (_MODIFIER_TRANSFORM_SETTINGS), an Array its offset object even
        with use_object_offset off. A texture mapped through an object's space
        needs a texture and the OBJECT mapping (_MODIFIER_TEXTURE_SETTINGS),
        a Volume Displace the OBJECT mapping, a Warp both its from and to
        objects and a Line Art use_custom_camera for its camera, and a UV
        Project the projectors it counts. On Blender 5.0.1 each of these, on a
        visible object, got a disabled camera it read, or a disabled child of
        the camera, evaluated in full; none did short of its condition or
        disabled in viewports. A particle system's modifier reads what its
        settings name (_particle_transform_sources), enabled in viewports or
        not: Blender relates the particle system itself.

        A Boolean's OBJECT operand and a Data Transfer's use_object_transform
        are left out: Blender rebuilds no relations when operand_type or
        use_object_transform changes, and keeps evaluating as before until
        something else rebuilds them. On Blender 5.0.1 a disabled camera
        whose disabled child such a modifier named, switched to the OBJECT
        operand or to use_object_transform after the camera moved, kept a
        matrix_world 11 m from a render, while Blender held the camera through
        the modifier's object (_evaluated_dependent cannot tell). A Geometry
        Nodes modifier's object inputs are left out too: Blender relates an
        input that a script sets (modifier[key] = object) only at its next
        relations update, as such a write tags nothing. On Blender 5.0.1 a
        disabled camera so named after it moved, while a muted Hook of another
        mesh held it in the depsgraph, kept a matrix_world 11 m from a render.

        Args:
            modifier: The modifier.

        Returns:
            The objects; empty for a modifier disabled in viewports, other
            than a particle system's.
        """
        kind = modifier.type
        if kind == "PARTICLE_SYSTEM":
            return cls._particle_transform_sources(modifier.particle_system.settings)
        if not modifier.show_viewport:
            return []
        sources = [
            getattr(modifier, setting)
            for setting in _MODIFIER_TRANSFORM_SETTINGS.get(kind, ())
        ]
        if kind in _MODIFIER_TEXTURE_SETTINGS:
            texture, mapping, setting = _MODIFIER_TEXTURE_SETTINGS[kind]
            if (
                getattr(modifier, texture) is not None
                and getattr(modifier, mapping) == "OBJECT"
            ):
                sources.append(getattr(modifier, setting))
        if kind == "LINEART" and modifier.use_custom_camera:
            sources.append(modifier.source_camera)
        elif kind == "VOLUME_DISPLACE" and modifier.texture_map_mode == "OBJECT":
            sources.append(modifier.texture_map_object)
        elif kind == "WARP":
            if modifier.object_from is not None and modifier.object_to is not None:
                sources.extend((modifier.object_from, modifier.object_to))
        elif kind == "UV_PROJECT":
            sources.extend(projector.object for projector in modifier.projectors)
        return [source for source in sources if source is not None]

    @staticmethod
    def _particle_transform_sources(settings: Any) -> List[Any]:
        """
        Return the objects whose world matrix a particle system reads through its settings.

        Blender 5.0.1 relates a particle system to the transform of each Avoid
        and Follow Leader boid rule's object, in every boid state and whatever
        the physics type while the settings keep boid data
        (build_particle_systems), and the settings to an object that a texture
        slot with a texture maps through, in OBJECT coordinates
        (build_particle_settings). It builds neither object for them, so it
        relates one only while the depsgraph holds it for something else
        (_enabled_dependent). On Blender 5.0.1 each of these, on a visible
        mesh, got a disabled camera that a muted Hook of another mesh held
        evaluated in full, its modifier enabled in viewports or not; without
        the Hook, or for a Goal rule's object or a slot without a texture or
        in other coordinates, the camera kept a matrix_world 11 m from a
        render.

        The instance object and collection are left out: Blender relates
        them only while render_type shows them, and rebuilds no relations
        when render_type changes. On Blender 5.0.1 a disabled camera named
        as a visible mesh's instance object while render_type was HALO, which
        a script switched to OBJECT after an evaluation, kept a matrix_world
        11 m from a render while a muted Hook of another mesh held it. So are
        keyed targets, which Blender builds with their particle system's
        object, without a relation to their transform: on Blender 5.0.1 a
        disabled camera such a target named kept a stale matrix_world when an
        object reading that mesh came before it in the view layer.

        Args:
            settings: The particle system's settings.

        Returns:
            The objects.
        """
        sources: List[Any] = []
        boids = settings.boids
        if boids is not None:
            sources.extend(
                rule.object
                for state in boids.states
                for rule in state.rules
                if rule.type in ("AVOID", "FOLLOW_LEADER")
            )
        sources.extend(
            slot.object
            for slot in settings.texture_slots
            if slot is not None
            and slot.texture is not None
            and slot.texture_coords == "OBJECT"
        )
        return [source for source in sources if source is not None]

    def _evaluated_dependent(
        self, obj: Any, needs: Tuple[Dict[Any, List[Any]], Dict[Any, List[Any]]]
    ) -> Optional[Any]:
        """
        Return an object Blender evaluates that computes with this one's world matrix.

        Blender evaluates an object disabled in viewports (_viewport_disablers)
        in full when an object it evaluates needs its world matrix
        (_scene_dependents): a child of it, or the owner of a constraint
        Blender applies that targets it. That dependent can be disabled too
        and evaluated through a dependent of its own. An armature's pose needs
        the targets of its bones' constraints, and an object's geometry what
        its modifiers read, which count only while that armature or object is
        enabled: whether Blender evaluates a disabled one in full can depend
        on the order of the view layer's objects alone. On Blender 5.0.1 a
        visible child of a disabled armature got the target of a bone's Copy
        Location evaluated when the child came first in the view layer, and
        left it stale when an object reading the child came first, which
        built the child, and the armature with it, as not visible; a disabled
        mesh's Hook alike. A visible Copy Location of the armature itself left
        it stale. A child of the constrained bone, or of a bone below it,
        needs the target through relations instead, as a child needs its
        parent (_scene_dependents), whatever the armature: Blender 5.0.1
        evaluated the target for such a child in all 24 orders of it, an
        object reading it, the armature and the camera, and for a child of
        the armature itself, of a bone above the constrained one or of another
        bone only in the 12 where the child came before its reader. What a
        particle system's settings name counts only while the depsgraph holds
        it (_particle_transform_sources): on Blender 5.0.1 a visible mesh's
        Avoid boid rule naming a disabled child of a held camera left the
        camera stale, and got it evaluated once a muted Hook held the child
        too. Blender 5.0.1 kept the matrix_world a render gave a disabled
        camera with a child, with a grandchild through a disabled child,
        targeted by another object's Copy Location or by one on a bone of an
        enabled armature, or read by a visible mesh's Hook, its Track To
        included; one whose only child was disabled kept a stale one. Blender
        evaluated it in full for a muted constraint and for a driver that read
        its transform too, which this leaves out: nothing needs the target of a
        muted constraint, and a driver that read only a custom property of the
        camera got it evaluated in part, its matrix_world stale, so drivers
        are not counted.

        Blender's viewport depsgraph can lag behind those settings. An object
        it does not hold at all is evaluated by nothing, whatever names it: on
        Blender 5.0.1 a camera that only a reference not yet related named (a
        Geometry Nodes input a script had set) was not in the depsgraph. An
        object Blender comes to need without a relations update (a modifier's
        show_viewport switched on, or keyed; the keyed hide_viewport of the
        object that reads it) is evaluated again only once it is tagged
        itself: on Blender 5.0.1 such a camera kept its stale matrix_world
        through later evaluations, and was evaluated where a render put it
        once tagged. So an object found needed is tagged and the depsgraph
        evaluated again before this returns, which evaluates what computes
        with it again too. In a GUI session that evaluation also runs the
        depsgraph_update handlers, as any other evaluation does.

        Args:
            obj: The object.
            needs: The scene's dependents and readers (_scene_dependents).

        Returns:
            The first dependent or reader found that is enabled in viewports,
            or None, also when the depsgraph does not hold the object.
        """
        blender_module: Any = bpy
        depsgraph = blender_module.context.evaluated_depsgraph_get()
        if not obj.evaluated_get(depsgraph).is_evaluated:
            return None
        found = self._enabled_dependent(obj, needs, depsgraph)
        if found is not None:
            obj.update_tag(refresh={"OBJECT"})
            blender_module.context.evaluated_depsgraph_get()
        return found

    def _enabled_dependent(
        self,
        obj: Any,
        needs: Tuple[Dict[Any, List[Any]], Dict[Any, List[Any]]],
        depsgraph: Any,
    ) -> Optional[Any]:
        """
        Search the scene's dependents for one that Blender evaluates (_evaluated_dependent).

        Args:
            obj: The object.
            needs: The scene's dependents and readers (_scene_dependents).
            depsgraph: The evaluated viewport depsgraph.

        Returns:
            The first dependent or reader found that is enabled in viewports,
            or None.
        """
        dependents, readers = needs
        enabled: Dict[Any, bool] = {}

        def is_enabled(candidate: Any) -> bool:
            # The depsgraph holds every enabled object. Telling one it does not
            # hold costs nothing, while _viewport_disablers reads
            # users_collection and walks the view layer's collections: once
            # per object of a disabled set of 10 000, which the walk visited
            # through their parent, it took 3.4 s on Blender 5.0.1.
            if candidate not in enabled:
                enabled[candidate] = bool(
                    candidate.evaluated_get(depsgraph).is_evaluated
                ) and not self._viewport_disablers(candidate)
            return enabled[candidate]

        pending, seen = [obj], {obj}
        while pending:
            source = pending.pop()
            # Blender builds what a constraint or a modifier names, but not
            # what a particle system's settings name: it relates that only to
            # an object it holds for something else.
            if source.evaluated_get(depsgraph).is_evaluated:
                for reader in readers.get(source, []):
                    if is_enabled(reader):
                        return reader
            for dependent in dependents.get(source, []):
                if dependent in seen:
                    continue
                if is_enabled(dependent):
                    return dependent
                seen.add(dependent)
                pending.append(dependent)
        return None

    def _transform_pose_warning(
        self,
        camera: Any,
        disablers: List[str],
        constraints: List[Dict[str, str]],
        composed: bool,
        stale_parent: bool,
    ) -> str:
        """
        Explain the pose get_camera_info reports for a camera Blender may not evaluate.

        Args:
            camera: The camera, disabled in viewports and needed by nothing
                Blender evaluates (_evaluated_dependent).
            disablers: The settings that disable it (_viewport_disablers).
            constraints: Its active constraints (_active_constraints).
            composed: Whether the pose was composed from its transform
                (_matrix_without_constraints); False when only Blender
                computes its parent matrix, and the pose is its matrix_world.
            stale_parent: Whether its parent is disabled in viewports and
                needed by nothing Blender evaluates (_evaluated_dependent).

        Returns:
            The warning: why Blender may not evaluate the camera and what the
            pose then misses. For a composed pose, that is its constraints
            (named), drivers and keys, and, when its parent may not be
            evaluated either, the parent's moves since its last evaluation.
            For the scene camera or a marker's camera, it names the role and
            its animated hide_viewport (_viewport_disablers).
        """
        role = self._camera_role(camera)
        if role is None:
            cause = (
                "as it is neither the scene camera nor a timeline marker's camera, "
                f"{' and '.join(disablers)}"
            )
            remedy = "make it the scene camera"
        else:
            cause = (
                f"although it is {role}, as {' and '.join(disablers)} (Blender "
                "evaluates a camera whose hide_viewport is animated only while it is "
                "enabled in viewports)"
            )
            remedy = "remove the animation of its hide_viewport"
        unevaluated = (
            f"Blender may not evaluate camera {camera.name!r} for the viewport, "
            f"{cause}, and nothing Blender evaluates depends on it through a child, "
            "a constraint or a modifier, so its matrix_world may hold the pose of an "
            "earlier evaluation."
        )
        remedy = f"Change those settings or {remedy} to have Blender evaluate it."
        parent = camera.parent
        if not composed:
            kind = (
                f"parent type {camera.parent_type}"
                if camera.parent_type != "OBJECT"
                else "a curve"
            )
            return (
                f"{unevaluated} Only Blender computes the matrix its parent "
                f"{parent.name!r} gives it ({kind}), so the reported pose is that "
                "matrix_world, which misses any change to the camera or its parent "
                f"since. {remedy}"
            )
        if constraints:
            names = ", ".join(f"'{c['name']}' ({c['type']})" for c in constraints)
            left_out = (
                f"It leaves out its constraints ({names}) and any drivers and keys "
                "on its transform, which apply whenever Blender evaluates it, as a "
                "render through it does."
            )
        else:
            left_out = (
                "No constraint acts on it, but drivers and keys on its transform "
                "would apply whenever Blender evaluates it, as a render through it "
                "does."
            )
        parent_clause = (
            f" Its parent {parent.name!r} may not be evaluated either, so the "
            "pose was composed through the last matrix Blender evaluated for "
            "that parent: it is elsewhere if the parent has moved since."
            if stale_parent
            else ""
        )
        through = ", through its parent" if parent is not None else ""
        return (
            f"{unevaluated} The reported pose is the one its own transform gives "
            f"(its location, rotation and scale{through}). {left_out}{parent_clause} "
            f"{remedy}"
        )

    @staticmethod
    def _camera_role(obj: Any) -> Optional[str]:
        """
        Name the role for which Blender's viewport evaluates a camera whatever its visibility.

        Args:
            obj: The object.

        Returns:
            "the scene camera", "a timeline marker's camera" when a marker
            binds it, or None.
        """
        blender_module: Any = bpy
        scene = blender_module.context.scene
        if obj == scene.camera:
            return "the scene camera"
        if any(marker.camera == obj for marker in scene.timeline_markers):
            return "a timeline marker's camera"
        return None

    def _animates_visibility(self, obj: Any) -> bool:
        """
        Whether Blender's depsgraph takes an object's hide_viewport for animated.

        Blender 5.0.1 (BKE_fcurves_id_cb) reads the F-curves of the action
        slot the object is assigned, of the action and slot that tweaking an
        NLA strip stashes away (_tweak_storage_fcurves), its drivers and the
        action slots of the strips, meta strips' included, in its NLA tracks
        that are enabled: not muted, and solo while a track is solo. A muted
        or empty F-curve counts, as do a muted driver, a muted strip and a
        strip while NLA evaluation is off.

        Args:
            obj: The object.

        Returns:
            True when one of those F-curves animates hide_viewport.
        """
        anim_data = obj.animation_data
        if anim_data is None:
            return False
        fcurves = list(action_fcurves(anim_data, self._blender_version) or [])
        fcurves.extend(self._tweak_storage_fcurves(anim_data))
        fcurves.extend(anim_data.drivers)
        tracks = list(anim_data.nla_tracks)
        solo = any(track.is_solo for track in tracks)
        strips = [
            strip
            for track in tracks
            if (track.is_solo if solo else not track.mute)
            for strip in track.strips
        ]
        while strips:
            strip = strips.pop()
            strips.extend(strip.strips)
            # A strip holds an action and its slot as animation data does.
            fcurves.extend(action_fcurves(strip, self._blender_version) or [])
        return any(fcurve.data_path == "hide_viewport" for fcurve in fcurves)

    def _tweak_storage_fcurves(self, anim_data: Any) -> List[Any]:
        """
        Return the F-curves through which the action that NLA tweak mode stashes away animates its ID.

        While a strip is tweaked, the strip's action and slot are the assigned
        ones, and the ID's own action waits in action_tweak_storage, its slot's
        handle in action_slot_handle_tweak_storage (Blender 4.5 and later).
        Blender 4.4 does not expose that handle, so each slot of the stashed
        action counts there. On Blender 5.0.1 a scene camera whose stashed
        action keyed its hide_viewport kept the matrix_world of its last
        evaluation while its collection was disabled in viewports.

        Args:
            anim_data: The ID's animation_data.

        Returns:
            The F-curves; empty outside tweak mode, or when no action was
            assigned on entering it.
        """
        action = anim_data.action_tweak_storage
        if action is None:
            return []
        if self._blender_version < SLOTTED_ACTIONS_VERSION:
            return list(action.fcurves)
        from bpy_extras import anim_utils

        handle = getattr(anim_data, "action_slot_handle_tweak_storage", None)
        fcurves: List[Any] = []
        for slot in action.slots:
            if handle is not None and slot.handle != handle:
                continue
            channelbag = anim_utils.action_get_channelbag_for_slot(action, slot)
            if channelbag is not None:
                fcurves.extend(channelbag.fcurves)
        return fcurves

    @staticmethod
    def _world_bound_box(obj: Any) -> List[List[float]]:
        """
        Return an object's eight bounding-box corners in world space.

        ``bound_box`` items are ``bpy_prop_array``s, which ``mathutils.Matrix``
        refuses to multiply (``TypeError: Matrix multiplication: not supported
        between 'Matrix' and 'bpy_prop_array' types`` on Blender 5.0.1), so
        each corner becomes a ``Vector`` first.

        Args:
            obj: The Blender object.

        Returns:
            The corners as [x, y, z] lists, in ``bound_box`` order.
        """
        vector = _mathutils().Vector
        matrix = obj.matrix_world
        return [list(matrix @ vector(corner)) for corner in obj.bound_box]

    @staticmethod
    def _look_at_rotation(
        location: List[float], target: List[float]
    ) -> Tuple[float, float, float]:
        """
        Return the XYZ Euler rotation that turns a camera at ``location`` to ``target``.

        A camera looks down its local -Z. Euler XYZ (x, 0, z) turns that into
        (-sin z sin x, cos z sin x, -cos x); solving for the direction to the
        target keeps the camera level, with no roll.

        Args:
            location: Camera position [x, y, z].
            target: Point to look at [x, y, z].

        Returns:
            The (x, y, z) rotation in radians.
        """
        dx, dy, dz = (t - loc for t, loc in zip(target, location))
        return (
            math.pi / 2 + math.atan2(dz, math.hypot(dx, dy)),
            0.0,
            math.atan2(-dx, dy),
        )


class BlenderRuntimeAdapter:
    """Adapter for Blender runtime operations."""

    name: str = "blender"
    logger = logger

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize Blender runtime adapter.

        Args:
            settings: Configuration settings.
        """
        from ..config import get_settings

        self.settings = settings or get_settings()
        self.logger.info("Blender runtime adapter initialized")

    @contextmanager
    def create_session(self) -> Any:
        """
        Create a Blender runtime session context manager.

        Yields:
            BlenderRuntimeSession instance.
        """
        if self.settings.blender.mode == "attached":
            from .blender_connection import BlenderConnection

            yield BlenderConnection(self.settings)
            return
        session = BlenderRuntimeSession(self.settings)
        try:
            yield session
        finally:
            session.cleanup()

    @property
    def session_blocks_io(self) -> bool:
        """Report whether session calls block on I/O rather than touching ``bpy``.

        Attached sessions only wait on the socket to the Blender editor, so the
        server runs them on a worker thread; embedded sessions call ``bpy``,
        which must stay on the calling thread.

        Returns:
            True in attached mode.
        """
        return self.settings.blender.mode == "attached"

    def is_available(self) -> bool:
        """
        Check whether Blender runtime is available.

        Returns:
            True when configured for attachment or the local bpy module is available.
        """
        return self.settings.blender.enabled and (
            self.settings.blender.mode == "attached" or BLENDER_AVAILABLE
        )

    def close(self) -> None:
        """Nothing is held between sessions; each one cleans up after itself."""

    def get_capabilities(self) -> List[str]:
        """
        Get list of Blender runtime capabilities.

        Returns:
            Capability list for Blender runtime adapter.
        """
        if not self.is_available():
            return []

        return [
            "blender_runtime_info",
            "blender_scene_listing",
            "blender_object_info",
            "blender_mesh_info",
            "blender_bounding_box",
            "blender_search_objects",
            "blender_scene_summary",
            "blender_material_info",
            "blender_distance_between",
            "blender_bounds_check",
            "blender_capture_viewport",
            "blender_set_camera_view",
            "blender_camera_info",
            "blender_focus_on_object",
            "blender_viewport_info",
            "blender_capture_viewport_sequence",
            "blender_create_object",
            "blender_delete_object",
            "blender_set_transform",
            "blender_set_parent",
            "blender_clear_parent",
            "blender_assign_material",
            "blender_add_modifier",
            "blender_apply_modifier",
            "blender_remove_modifier",
            "blender_set_light_params",
            "blender_open_file",
            "blender_save_file",
            "blender_import_file",
            "blender_export_file",
            "blender_file_info",
            "blender_set_frame",
            "blender_get_frame",
            "blender_set_frame_range",
            "blender_play_animation",
            "blender_insert_keyframe",
            "blender_delete_keyframe",
            "blender_get_keyframes",
            "blender_setup_rigid_body",
            "blender_add_force_field",
            "blender_force_field_info",
            "blender_add_constraint",
            "blender_constraint_info",
            "blender_physics_state",
            "blender_object_trajectory",
            "blender_bake_simulation",
            "blender_free_bake",
            "execute_blender_script",
            "create_blender_mesh_from_data",
            "blender_clean_mesh",
            "apply_simready_metadata",
            "get_simready_metadata",
            "validate_simready_compliance",
            "export_simready_usd",
            "setup_simready_hierarchy",
        ]


def create_blender_session(
    settings: Optional[Settings] = None,
) -> BlenderRuntimeSession:
    """
    Create a Blender runtime session.

    Args:
        settings: Configuration settings.

    Returns:
        BlenderRuntimeSession instance.
    """
    return BlenderRuntimeSession(settings)


def is_blender_available() -> bool:
    """Check whether Blender runtime is available."""
    return BLENDER_AVAILABLE
