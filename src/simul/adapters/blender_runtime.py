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
import secrets
import shutil
import tempfile
import time
import uuid
from array import array
from contextlib import contextmanager
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    overload,
)

# The Blender add-on bundles these modules next to this one; simready_usd
# imports pxr only when it is called.
from ..utils.asset_references import (
    UncheckableReferences,
    blender_fbx_reads as _fbx_reads,
    blender_gltf_reads as _gltf_reads,
    blender_obj_reads as _obj_reads,
    outside_references,
)
from ..utils.asset_references import resolve_ncase as _resolve_ncase  # noqa: F401 - re-exported for tests
from ..utils.regex_guard import exponential_regex_construct
from .simready_usd import COLLIDER_TYPES as SIMREADY_COLLIDER_TYPES
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

# A page of list_scene_objects or search_objects stops before its entries' JSON
# passes this many bytes. The server cuts any list that takes a reply past its
# 40 KB result budget (simul.mcp.result_budget) and keeps the count it was
# given: at 6,006 objects, a page of 5000 came back as 429 with count 5000. The
# rest of the reply takes well under a kilobyte. The MCP schemas repeat it as
# LISTING_JSON_BYTES_MAX; this module ships inside the Blender add-on and
# cannot import them.
LISTING_JSON_BYTES_MAX = 30_000

# capture_viewport methods (issue #228). "auto" means "viewport" in attached
# mode and "render" in embedded mode, which has no editor window.
CAPTURE_METHODS = ("auto", "viewport", "render")

# capture_viewport_sequence writes its frames to files (B1-05): five 512 px
# frames inline exceeded the server's 40 KB result budget and came back as one,
# and 1500 frames ran 7 s on Blender's main thread before failing to encode.
# One call captures at most MAX_SEQUENCE_FRAMES frames (DEFAULT_SEQUENCE_FRAMES
# unless asked), so that its list of paths stays well inside the budget, and
# starts no frame after SEQUENCE_TIME_BUDGET_SECONDS, well inside the default
# 30 s client timeout; the reply names the frame to continue from.
# BlenderCaptureSequenceRequest repeats the frame numbers: this module ships
# inside the Blender add-on and cannot import the MCP schemas.
DEFAULT_SEQUENCE_FRAMES = 50
MAX_SEQUENCE_FRAMES = 100
SEQUENCE_TIME_BUDGET_SECONDS = 10.0

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

# The Principled BSDF input each assign_material texture drives.
_TEXTURE_SOCKETS: Dict[str, str] = {
    "base_color": "Base Color",
    "metallic": "Metallic",
    "roughness": "Roughness",
    "normal": "Normal",
}

# The settings each light type has besides energy, color and use_shadow, as
# set_light_params takes them. A sun's softness is its angle and an area
# light's its size; shadow_soft_size is the radius of point and spot lights.
_LIGHT_TYPE_SETTINGS: Dict[str, Tuple[str, ...]] = {
    "POINT": ("shadow_soft_size",),
    "SPOT": ("shadow_soft_size", "spot_size", "spot_blend"),
    "SUN": ("angle",),
    "AREA": ("shape", "size", "size_y"),
}

# Limits of the light settings: Blender 5.0.1's hard limits, which it clamps
# to silently, except that energy may not be negative here (Blender allows
# it). None leaves that side open. The angles are 1 degree and pi as Blender
# stores them, in single precision, so a value read back passes again; the
# MCP schemas repeat them as LIGHT_ANGLE_MIN and LIGHT_ANGLE_MAX.
_LIGHT_ANGLE_MIN = 0.01745329238474369
_LIGHT_ANGLE_MAX = 3.1415927410125732
_LIGHT_LIMITS: Dict[str, Tuple[float, Optional[float]]] = {
    "energy": (0.0, None),
    "shadow_soft_size": (0.0, None),
    "spot_size": (_LIGHT_ANGLE_MIN, _LIGHT_ANGLE_MAX),
    "spot_blend": (0.0, 1.0),
    "angle": (0.0, _LIGHT_ANGLE_MAX),
    "size": (0.0, None),
    "size_y": (0.0, None),
}

_AREA_LIGHT_SHAPES = ("SQUARE", "RECTANGLE", "DISK", "ELLIPSE")

# uv_unwrap_mesh methods and the uv operator each runs.
_UV_UNWRAP_METHODS: Dict[str, str] = {
    "SMART_PROJECT": "smart_project",
    "CUBE_PROJECT": "cube_project",
}

# Modifier types whose viewport and render results can differ through a
# second level setting: (viewport setting, render setting).
_RENDER_LEVEL_SETTINGS: Dict[str, Tuple[str, str]] = {
    "MULTIRES": ("levels", "render_levels"),
    "SUBSURF": ("levels", "render_levels"),
}


class ValidationError(ValueError):
    """Arguments a Blender tool refuses, with a stable reason in ``details.code``.

    The class name is the envelope's ``error_type`` in both modes: the server
    names an exception by its class, and so does the add-on's bridge. It
    subclasses ValueError, which these refusals raised before they had a type.
    """

    def __init__(self, message: str, code: str, **details: Any) -> None:
        super().__init__(message)
        self.details: Dict[str, Any] = {"code": code, **details}


class NotFoundError(ValueError):
    """Something an argument names does not exist; ``details.code`` names what."""

    def __init__(self, message: str, code: str, **details: Any) -> None:
        super().__init__(message)
        self.details: Dict[str, Any] = {"code": code, **details}


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


# 4x4 matrices as row lists of floats. The SimReady tools compose and invert
# world matrices themselves: Blender stores no fresh matrix_world for an object
# its depsgraph leaves out, and mathutils is not importable outside Blender.
_Matrix4 = List[List[float]]
_IDENTITY4: _Matrix4 = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]


def _m4(matrix: Any) -> _Matrix4:
    """A mathutils Matrix, a stand-in or nested sequences as four rows of four floats."""
    return [[float(matrix[i][j]) for j in range(4)] for i in range(4)]


def _m4_mul(a: _Matrix4, b: _Matrix4) -> _Matrix4:
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def _m4_inverse(m: _Matrix4) -> Optional[_Matrix4]:
    """The inverse by Gauss-Jordan elimination, or None for a matrix with no inverse."""
    rows = [list(row) + [1.0 if i == j else 0.0 for j in range(4)] for i, row in enumerate(m)]
    for col in range(4):
        pivot = max(range(col, 4), key=lambda r: abs(rows[r][col]))
        if abs(rows[pivot][col]) < 1e-12:
            return None
        rows[col], rows[pivot] = rows[pivot], rows[col]
        lead = rows[col][col]
        rows[col] = [v / lead for v in rows[col]]
        for r in range(4):
            factor = rows[r][col]
            if r != col and factor:
                rows[r] = [v - factor * w for v, w in zip(rows[r], rows[col])]
    return [row[4:] for row in rows]


def _m4_shape(m: _Matrix4) -> Tuple[List[float], List[float], float, bool]:
    """Translation, axis lengths, shear and mirroring of a transform matrix.

    Shear is the largest cosine between two of its axes: 0 for a matrix that
    location, rotation and scale can express, as Blender's loc/rot/scale and
    USD's translate/rotate/scale ops can.
    """
    axes = [[m[r][c] for r in range(3)] for c in range(3)]
    lengths = [math.sqrt(sum(v * v for v in axis)) for axis in axes]
    units = [[v / n for v in axis] if n > 0 else axis for axis, n in zip(axes, lengths)]
    shear = max(
        abs(sum(a * b for a, b in zip(units[i], units[j])))
        for i, j in ((0, 1), (0, 2), (1, 2))
    )
    a, b, c = axes
    det = (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - b[0] * (a[1] * c[2] - a[2] * c[1])
        + c[0] * (a[1] * b[2] - a[2] * b[1])
    )
    return [m[0][3], m[1][3], m[2][3]], lengths, shear, det < 0


def _m4_is_identity(m: _Matrix4, tolerance: float = 1e-5) -> bool:
    return all(abs(m[i][j] - _IDENTITY4[i][j]) <= tolerance for i in range(4) for j in range(4))


class _WorldResolver:
    """Where Blender places objects, as its USD exporter writes them.

    An object the depsgraph evaluates has a fresh ``matrix_world``. One it
    leaves out (in an excluded collection, a collection or object disabled in
    viewports) keeps the matrix of its last evaluation, which is stale once its
    parent moved; its world is composed from its parent as Blender composes it:
    parent world @ matrix_parent_inverse @ matrix_basis. Without a depsgraph
    (a stand-in bpy) every ``matrix_world`` is taken as fresh.
    """

    def __init__(self, depsgraph: Any) -> None:
        self._depsgraph = depsgraph
        self._worlds: Dict[str, _Matrix4] = {}
        self._evaluated: Dict[str, bool] = {}

    def _copy(self, obj: Any) -> Any:
        if self._depsgraph is None or not hasattr(obj, "evaluated_get"):
            return obj
        return obj.evaluated_get(self._depsgraph)

    def evaluated(self, obj: Any) -> bool:
        """Whether Blender evaluates the object, so that its matrix_world is current."""
        if obj.name not in self._evaluated:
            if self._depsgraph is None or not hasattr(obj, "evaluated_get"):
                fresh = True
            else:
                # The original comes back for an object the depsgraph left out.
                fresh = bool(getattr(self._copy(obj), "is_evaluated", False))
            self._evaluated[obj.name] = fresh
        return self._evaluated[obj.name]

    def world(self, obj: Any) -> _Matrix4:
        """The object's world matrix: evaluated, or composed from its parent."""
        if obj.name in self._worlds:
            return self._worlds[obj.name]
        if self.evaluated(obj):
            world = _m4(self._copy(obj).matrix_world)
        else:
            local = _m4_mul(_m4(obj.matrix_parent_inverse), _m4(obj.matrix_basis))
            parent = obj.parent
            if parent is None:
                world = local
            elif getattr(obj, "parent_type", "OBJECT") == "OBJECT":
                world = _m4_mul(self.world(parent), local)
            else:
                # A bone or vertex parent of an object Blender does not evaluate:
                # its last stored matrix is all there is.
                world = _m4(obj.matrix_world)
        self._worlds[obj.name] = world
        return world


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


def _box_relation(
    corners: List[List[float]],
    rows: List[List[float]],
    low: Sequence[float],
    high: Sequence[float],
) -> str:
    """
    Say where an object's world-space bounding box lies relative to an axis-aligned box.

    The object's box is the parallelepiped its eight world-space corners span,
    its edges along the columns of its world matrix. Two convex boxes are
    apart exactly when some axis separates their projections, and the axes to
    try are each box's face normals and the cross products of their edges
    (the separating axis theorem). Comparisons allow for single-precision
    rounding of the corners, so touching counts as intersecting and a box
    that fills the bounds as inside.

    Args:
        corners: The eight world-space corners, in any order.
        rows: The first three rows of the object's world matrix.
        low: The bounds' minimum corner.
        high: The bounds' maximum corner.

    Returns:
        "inside", "intersecting" or "outside".
    """
    extent = max(
        [1.0] + [abs(v) for corner in corners for v in corner] + [abs(v) for v in (*low, *high)]
    )
    slack = 4 * _FLOAT32_EPSILON * extent
    if all(
        low[i] - slack <= corner[i] <= high[i] + slack for corner in corners for i in range(3)
    ):
        return "inside"

    def cross(a: Sequence[float], b: Sequence[float]) -> List[float]:
        return [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]

    world_axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    edges = [[rows[0][i], rows[1][i], rows[2][i]] for i in range(3)]
    axes = (
        world_axes
        + [cross(edges[1], edges[2]), cross(edges[2], edges[0]), cross(edges[0], edges[1])]
        + [cross(w, e) for w in world_axes for e in edges]
    )
    middle = [(a + b) / 2 for a, b in zip(low, high)]
    half = [(b - a) / 2 for a, b in zip(low, high)]
    for axis in axes:
        length = math.sqrt(sum(v * v for v in axis))
        if length <= 1e-12:
            continue
        unit = [v / length for v in axis]
        projected = [sum(u * c for u, c in zip(unit, corner)) for corner in corners]
        centre = sum(u * m for u, m in zip(unit, middle))
        reach = sum(abs(u) * h for u, h in zip(unit, half))
        if max(projected) < centre - reach - slack or min(projected) > centre + reach + slack:
            return "outside"
    return "intersecting"


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


# -- Files an import reads besides the one named (B1-20) ---------------------
#
# The readers for glTF, OBJ and FBX live in utils/asset_references.py, shared
# with the Unreal backend; the USD reader below follows a stage's layers with
# the pxr Blender bundles.

# USD layers an import composes, and so reads, besides textures.
_USD_LAYER_EXTENSIONS = (".usd", ".usda", ".usdc", ".usdz")


def _usd_reads(path: Path, allowed: Callable[[str], bool]) -> List[List[str]]:
    """List the layers and assets a USD file composes, following its layers.

    Sublayers, references, payloads and asset-valued attributes (textures)
    of each layer, resolved against that layer, through pxr (Blender bundles
    it). Layers outside the sandbox are not opened: they are refused anyway.
    """
    try:
        from pxr import Ar, Sdf, UsdUtils
    except ImportError:
        return _usda_text_reads(path)

    reads: List[List[str]] = []
    pending: List[str] = [str(path)]
    seen: set = set()
    while pending:
        identifier = pending.pop()
        if identifier in seen or len(seen) >= 1000:
            continue
        seen.add(identifier)
        try:
            layer = Sdf.Layer.FindOrOpen(identifier)
        except Exception:
            # The importer cannot open it either, so it reads nothing through it.
            layer = None
        if layer is None:
            continue
        anchor = os.path.dirname(Ar.SplitPackageRelativePathOuter(identifier)[0])
        try:
            sublayers, references, payloads = UsdUtils.ExtractExternalReferences(identifier)
        except Exception as exc:
            raise UncheckableReferences(f"pxr could not list the references of {identifier}: {exc}") from exc
        for asset in (*sublayers, *references, *payloads):
            if not asset:
                continue
            resolved = layer.ComputeAbsolutePath(asset)
            outer, inner = Ar.SplitPackageRelativePathOuter(resolved)
            if not os.path.isabs(outer):
                outer = os.path.join(anchor, outer)
            outer = os.path.normpath(outer)
            if "<UDIM>" in outer:
                import glob

                tiles = glob.glob(outer.replace("<UDIM>", "[0-9][0-9][0-9][0-9]"))
                reads.extend([tile] for tile in tiles)
                continue
            # A path such as "textures/a.png" that does not exist beside the
            # layer is looked up in the resolver's search paths
            # (PXR_AR_DEFAULT_SEARCH_PATH), as the import's stage does.
            found = _usd_search_path_hit(Ar, asset, identifier, outer)
            reads.append([outer] if found is None else [outer, found])
            target = found if found is not None and not os.path.exists(outer) else outer
            if not (target.lower().endswith(_USD_LAYER_EXTENSIONS) and allowed(target)):
                continue
            if inner and target == outer:
                pending.append(resolved)
            elif os.path.isfile(target):
                pending.append(target)
    return reads


def _usd_search_path_hit(ar: Any, asset: str, layer_path: str, anchored: str) -> Optional[str]:
    """Where the USD resolver finds ``asset`` for a layer, when that is not beside it.

    Returns:
        The file the resolver resolves the asset to, outside its package if
        it is in one, or None when that is the anchored path or nothing.
    """
    if os.path.exists(anchored):
        return None
    try:
        resolver = ar.GetResolver()
        identifier = resolver.CreateIdentifier(asset, ar.ResolvedPath(layer_path))
        hit = str(resolver.Resolve(identifier))
    except Exception:
        return None
    if not hit:
        return None
    outer = os.path.normpath(ar.SplitPackageRelativePathOuter(hit)[0])
    return None if outer == anchored else outer


def _usda_text_reads(path: Path) -> List[List[str]]:
    """Without pxr: the asset paths of a text .usda that names no other layer.

    Raises:
        UncheckableReferences: For a binary layer, or a text layer naming
            other layers, which this cannot follow.
    """
    with path.open("rb") as handle:
        head = handle.read(8)
    if not head.startswith(b"#usda"):
        raise UncheckableReferences(f"{path.name} is a binary USD file and pxr is not available")
    text = path.read_text(encoding="utf-8", errors="replace")
    reads = []
    for match in re.finditer(r"@@@(.*?)@@@|@([^@\n]+)@", text):
        asset = match.group(1) or match.group(2) or ""
        local = asset.split("[", 1)[0]
        if local.lower().endswith(_USD_LAYER_EXTENSIONS):
            raise UncheckableReferences(
                f"{path.name} composes {asset} and pxr is not available to follow it"
            )
        reads.append([os.path.normpath(os.path.join(str(path.parent), local))])
    return reads


# Importer options that search folders for a file a reference names, set off
# while the sandbox is on: the FBX importer's use_image_search walks the
# texture's folder and the FBX's, subfolders included, for any file of the
# same name (bpy_extras.image_utils.load_image), which found and loaded an
# image outside the sandbox. No other importer searches.
_IMPORT_SEARCH_OPTIONS: Dict[str, Tuple[str, ...]] = {"FBX": ("use_image_search",)}

_IMPORT_READS: Dict[str, Callable[[Path, Callable[[str], bool]], List[List[str]]]] = {
    "GLTF": lambda path, _allowed: _gltf_reads(path),
    "OBJ": lambda path, _allowed: _obj_reads(path),
    "FBX": _fbx_reads,
    "USD": _usd_reads,
}


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


def _finite_number(value: Any, what: str) -> float:
    """A finite real number as a float (not a bool), or a ValidationError naming ``what``."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValidationError(f"{what} must be a number, got {value!r}", "invalid_value", field=what)
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(f"{what} must be finite, got {value!r}", "invalid_value", field=what)
    return number


def _unit_value(value: Any, what: str) -> float:
    """A finite number in [0, 1], or a ValidationError naming ``what``."""
    number = _finite_number(value, what)
    if not 0.0 <= number <= 1.0:
        raise ValidationError(
            f"{what} must lie between 0 and 1, got {value!r}", "out_of_range", field=what
        )
    return number


def _unit_color(value: Any, what: str) -> Tuple[float, float, float, Optional[float]]:
    """An RGB or RGBA colour in [0, 1] as (r, g, b, alpha), alpha None for RGB.

    Blender's Base Color socket takes four numbers and refused three with
    "sequences of dimension 0 should contain 4 items, not 3".
    """
    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__") or len(value) not in (3, 4):
        raise ValidationError(
            f"{what} must be [r, g, b] or [r, g, b, a], got {value!r}", "invalid_value", field=what
        )
    channels = [_unit_value(channel, what) for channel in value]
    return (channels[0], channels[1], channels[2], channels[3] if len(channels) == 4 else None)


def _real_path(path: str) -> str:
    """A file path in one spelling: absolute, symlinks resolved, case-folded where the OS folds."""
    return os.path.normcase(os.path.realpath(path))


def _image_file(image: Any) -> Optional[str]:
    """The file an image datablock reads, as ``_real_path`` spells it; None when it reads none."""
    path = getattr(image, "filepath", "") or ""
    if not path:
        return None
    resolve = getattr(getattr(bpy, "path", None), "abspath", None)
    if resolve is not None:
        # Blender-relative paths (//textures/a.png) are relative to the .blend
        # file, or to the library the image is linked from.
        path = resolve(path, library=getattr(image, "library", None))
    return _real_path(path)


def _name_list(names: Sequence[str], limit: int = 5) -> str:
    """``'a', 'b' and 3 more`` for the first ``limit`` names."""
    shown = ", ".join(f"'{name}'" for name in names[:limit])
    rest = len(names) - limit
    return shown + (f" and {rest} more" if rest > 0 else "")


def _live_links(socket: Any) -> List[Any]:
    """The links into a socket that Blender evaluates: neither muted nor invalid."""
    return [
        link
        for link in socket.links
        if not getattr(link, "is_muted", False) and getattr(link, "is_valid", True)
    ]


def _link_source(link: Any) -> Tuple[Any, Any]:
    """The node and output socket a link really comes from, through any Reroute nodes.

    Returns:
        That node and socket, or the last Reroute's when one has nothing
        plugged in.
    """
    source, output = link.from_node, link.from_socket
    for _hop in range(1000):
        if source.type != "REROUTE":
            break
        feeding = [live for socket in source.inputs for live in _live_links(socket)]
        if not feeding:
            break
        source, output = feeding[0].from_node, feeding[0].from_socket
    return source, output


def _mesh_index(value: Any, what: str, vertex_count: int) -> int:
    """A vertex index of an edge or face, checked to be an integer within the vertex list."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{what} index {value!r} is not an integer")
    index = int(value)
    if index < 0 or index >= vertex_count:
        raise ValueError(f"{what} index {index} out of range [0, {vertex_count})")
    return index


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
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        List scene objects from the active Blender data context, a page at a time.

        A page holds at most ``max_items`` objects, and fewer when their JSON
        would pass LISTING_JSON_BYTES_MAX, so that it fits the server's result
        budget instead of being cut below the count it reports; ``next_offset``
        continues the listing.

        Args:
            collection_name: Optional collection name to filter objects.
            include_hidden: Include hidden objects when True.
            max_items: Maximum number of objects returned.
            offset: How many matching objects to skip, as next_offset gives it.

        Returns:
            The page (objects, count), how many objects match (total), the
            offset of the next page (next_offset, None after the last one) and
            whether more follow (truncated).
        """
        if max_items < 1:
            raise ValueError("max_items must be greater than zero")
        if offset < 0:
            raise ValueError("offset must not be negative")

        source_objects = self._resolve_object_source(collection_name)
        matching = []
        for scene_object in source_objects:
            visible = self._is_object_visible(scene_object)
            if include_hidden or visible:
                matching.append((scene_object, visible))
        page = self._page(
            matching[offset : offset + max_items], offset, len(matching), collection_name
        )

        return {
            "collection": collection_name,
            "include_hidden": include_hidden,
            "max_items": max_items,
            **page,
        }

    def _page(
        self,
        candidates: List[Tuple[Any, bool]],
        offset: int,
        total: int,
        collection_name: Optional[str],
    ) -> Dict[str, Any]:
        """
        Describe one page of matching objects.

        Args:
            candidates: The matching objects from ``offset`` on, as many as
                the page may hold, each with whether it is visible.
            offset: How many matching objects precede them.
            total: How many objects match.
            collection_name: The collection the listing filters by, which
                each entry names; otherwise each names the object's own.

        Returns:
            offset, count, total, next_offset, objects and truncated.
        """
        scene_collections = self._scene_collection_pointers()
        objects: List[Dict[str, Any]] = []
        size = 2
        for scene_object, visible in candidates:
            entry = {
                "name": scene_object.name,
                "object_type": scene_object.type,
                "collection": collection_name
                or self._collection_of(scene_object, scene_collections),
                "visible": visible,
            }
            # The server measures its budget as json.dumps does: non-ASCII
            # escaped, ", " between list items.
            entry_size = len(json.dumps(entry)) + 2
            if objects and size + entry_size > LISTING_JSON_BYTES_MAX:
                break
            objects.append(entry)
            size += entry_size
        following = offset + len(objects)
        more = following < total
        return {
            "offset": offset,
            "count": len(objects),
            "total": total,
            "next_offset": following if more else None,
            "objects": objects,
            "truncated": more,
        }

    @staticmethod
    def _scene_collection_pointers() -> set:
        """The pointers of the current scene's collections, its master collection included."""
        blender_module: Any = bpy
        root = blender_module.context.scene.collection
        return {root.as_pointer()} | {
            child.as_pointer() for child in root.children_recursive
        }

    @staticmethod
    def _collection_of(obj: Any, scene_collections: set) -> Optional[str]:
        """
        Name a collection that holds an object, one of the current scene's when it can.

        An object can be in several collections and scenes. The scene's own
        master collection is named "Scene Collection".

        Args:
            obj: The Blender object.
            scene_collections: _scene_collection_pointers().

        Returns:
            The first of its collections in the current scene, else its first
            collection anywhere (embedded mode lists every object in the
            file), or None for an object in none.
        """
        holders = list(obj.users_collection)
        in_scene = [c for c in holders if c.as_pointer() in scene_collections]
        chosen = (in_scene or holders or [None])[0]
        return None if chosen is None else chosen.name

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
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        Search for objects matching optional name pattern and type filter, a page at a time.

        A page is cut short to fit the server's result budget as
        list_scene_objects' is; ``next_offset`` continues the search.

        Args:
            name_pattern: Python regular expression searched for in each object
                name; a repeated group that holds a quantifier or alternatives,
                as in (\\w+)+, is refused because it can run for hours.
            object_type: Blender object type filter (MESH, LIGHT, etc.), in any case.
            max_results: Maximum number of results.
            offset: How many matching objects to skip, as next_offset gives it.

        Returns:
            The page (objects, count), how many objects match (total), the
            offset of the next page (next_offset, None after the last one) and
            whether more follow (truncated).

        Raises:
            ValueError: name_pattern is not a valid regular expression, or
                repeats a group that holds a quantifier or alternatives (see
                ``simul.utils.regex_guard``); object_type is not one of the
                running Blender's object types.
        """
        if max_results < 1:
            raise ValueError("max_results must be greater than zero")
        if offset < 0:
            raise ValueError("offset must not be negative")
        if object_type is not None:
            # Blender's type identifiers are upper case: "mesh" matched nothing.
            object_type = object_type.strip().upper()
            known = self._object_type_names()
            if object_type not in known:
                raise ValueError(
                    f"Unknown object_type {object_type!r}; Blender's object types: "
                    + ", ".join(known)
                )
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
        matching = [
            obj
            for obj in all_objects
            if (compiled_pattern is None or compiled_pattern.search(obj.name))
            and (object_type is None or obj.type == object_type)
        ]
        # Visibility is read only for the objects the page can describe.
        page = self._page(
            [
                (obj, self._is_object_visible(obj))
                for obj in matching[offset : offset + max_results]
            ],
            offset,
            len(matching),
            None,
        )

        return {"pattern": name_pattern, "object_type": object_type, **page}

    @staticmethod
    def _object_type_names() -> List[str]:
        """The running Blender's object type identifiers (MESH, LIGHT, ...)."""
        blender_module: Any = bpy
        return [
            item.identifier
            for item in blender_module.types.Object.bl_rna.properties["type"].enum_items
        ]

    def summarize_scene(self) -> Dict[str, Any]:
        """
        Produce a high-level scene summary grouped by object type.

        Returns:
            Dictionary with total counts, type breakdown, the scene's top-level
            collections, active camera, and frame range.
        """
        blender_module: Any = bpy
        all_objects = self._resolve_object_source(None)
        type_counts: Dict[str, int] = {}

        for obj in all_objects:
            type_counts[obj.type] = type_counts.get(obj.type, 0) + 1

        scene = blender_module.context.scene
        # The collections directly under the scene's own; bpy.data.collections
        # also holds nested ones, other scenes' and orphans.
        collection_names = [child.name for child in scene.collection.children]
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

    def get_material_info(self, material_name: str, max_nodes: int = 50) -> Dict[str, Any]:
        """
        Get a material's shading: its Principled BSDF values and what drives them.

        ``principled_params`` holds the values of the Principled BSDF that
        feeds the active Material Output; an input driven by a link has None
        there, as its own value is unused, and ``linked_inputs`` names the
        node, socket and, for an image texture (also behind a Normal Map
        node), the image's file, colour space and whether it is packed.
        ``nodes`` lists at most ``max_nodes`` nodes, those feeding the output
        first; ``node_count`` counts them all.

        Args:
            material_name: Name of the Blender material.
            max_nodes: Most nodes to list.

        Returns:
            Dictionary with the node list and count, Principled BSDF values,
            linked inputs, the render method and the user count.
        """
        if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or max_nodes < 1:
            raise ValidationError(
                f"max_nodes must be a whole number of at least 1, got {max_nodes!r}",
                "out_of_range",
                field="max_nodes",
            )
        blender_module: Any = bpy
        mat = blender_module.data.materials.get(material_name)
        if mat is None:
            raise NotFoundError(f"Material not found: {material_name}", "material_not_found")

        tree = self._material_tree(mat)
        listed: List[Any] = []
        principled = None
        if tree is not None:
            listed = self._nodes_feeding_output(tree, max_nodes)
            principled = self._principled_node(mat)

        return {
            "material_name": material_name,
            "use_nodes": tree is not None,
            "nodes": [
                {"name": node.name, "node_type": node.bl_idname, "label": getattr(node, "label", "")}
                for node in listed
            ],
            "node_count": 0 if tree is None else len(tree.nodes),
            "principled_params": (
                None if principled is None else self._extract_principled_params(principled)
            ),
            "linked_inputs": {} if principled is None else self._linked_inputs(principled),
            "surface_render_method": self._render_method(mat),
            "users": int(getattr(mat, "users", 0)),
        }

    def _nodes_feeding_output(self, tree: Any, limit: int) -> List[Any]:
        """Up to ``limit`` nodes: those the active output reads, nearest first, then the rest."""
        output = self._material_output(tree)
        ordered: List[Any] = []
        # Node names are unique within a tree; bpy may wrap one node in a new
        # Python object at each access, so identity cannot tell them apart.
        seen: set = set()
        pending = [] if output is None else [output]
        while pending and len(ordered) < limit:
            node = pending.pop(0)
            if node.name in seen:
                continue
            seen.add(node.name)
            ordered.append(node)
            pending.extend(link.from_node for socket in node.inputs for link in socket.links)
        for node in tree.nodes:
            if len(ordered) >= limit:
                break
            if node.name not in seen:
                seen.add(node.name)
                ordered.append(node)
        return ordered

    @staticmethod
    def _linked_inputs(bsdf: Any) -> Dict[str, Dict[str, Any]]:
        """What drives each linked input of a Principled BSDF, images included.

        A muted link drives nothing, so its input is not listed; a source
        behind Reroute nodes, and an image behind a Normal Map node, are
        followed to the node that provides them.
        """
        blender_module: Any = bpy
        linked: Dict[str, Dict[str, Any]] = {}
        for socket in bsdf.inputs:
            live = _live_links(socket)
            if not live:
                continue
            source, output = _link_source(live[0])
            entry: Dict[str, Any] = {
                "from_node": source.name,
                "from_node_type": source.bl_idname,
                "from_socket": output.name,
            }
            texture = source
            if source.type == "NORMAL_MAP":
                color = source.inputs.get("Color")
                feeding = [] if color is None else _live_links(color)
                texture = _link_source(feeding[0])[0] if feeding else None
            image = (
                getattr(texture, "image", None)
                if texture is not None and texture.type == "TEX_IMAGE"
                else None
            )
            if image is not None:
                path = getattr(image, "filepath", "") or ""
                resolve = getattr(getattr(blender_module, "path", None), "abspath", None)
                entry.update(
                    image_name=image.name,
                    image_path=(resolve(path) if resolve and path else path) or None,
                    image_colorspace=image.colorspace_settings.name,
                    image_packed=getattr(image, "packed_file", None) is not None,
                )
            linked[socket.name] = entry
        return linked

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
        Check where an object's world-space bounding box lies relative to axis-aligned bounds.

        The object's bounding box (``bound_box``, as get_bounding_box reports
        it) is turned, scaled and moved with the object, and tested as that
        box, not as the axis-aligned box around it: ``inside`` when all of it
        is within the bounds, ``intersecting`` when part of it is, and
        ``outside`` when none of it is. Its origin alone said a 10 m plane fit
        a 2 m box. A box touching the bounds intersects them; single-precision
        rounding of the corners is allowed for.

        Args:
            object_name: Object name to check.
            bounds_min: Minimum corner [x, y, z].
            bounds_max: Maximum corner [x, y, z], at least bounds_min on every axis.

        Returns:
            within_bounds (the whole box inside), relation, intersects,
            origin_within_bounds, the object's origin (object_location) and its
            world-space axis-aligned extent (object_bbox_min, object_bbox_max).

        Raises:
            ValueError: A bound is not finite, or bounds_min exceeds bounds_max.
        """
        low = _finite(bounds_min, 3, "bounds_min")
        high = _finite(bounds_max, 3, "bounds_max")
        inverted = [axis for axis, (a, b) in zip("xyz", zip(low, high)) if a > b]
        if inverted:
            raise ValueError(
                f"bounds_min exceeds bounds_max on {', '.join(inverted)}: "
                f"{list(low)} > {list(high)}"
            )
        obj = self._get_object_or_raise(object_name)
        self._evaluate_transforms()
        loc = [float(v) for v in obj.matrix_world.translation]
        corners = self._world_bound_box(obj)
        relation = _box_relation(
            corners, [list(row) for row in obj.matrix_world][:3], low, high
        )

        return {
            "object_name": object_name,
            "within_bounds": relation == "inside",
            "relation": relation,
            "intersects": relation != "outside",
            "origin_within_bounds": all(
                low[i] <= loc[i] <= high[i] for i in range(3)
            ),
            "object_location": loc,
            "object_bbox_min": [min(c[i] for c in corners) for i in range(3)],
            "object_bbox_max": [max(c[i] for c in corners) for i in range(3)],
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
        camera_name: Optional[str] = None,
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
            camera_name: Capture through this camera instead of the scene camera or
                the 3D View's angle; the scene camera and the view are left as they
                were. A render renders through it; a viewport capture draws the 3D
                View's shading from it (capture_method viewport_camera), keeping the
                render's aspect ratio inside width x height.

        Returns:
            Dictionary with image_base64, width, height, engine, format,
            camera_name (the camera captured through; None for the 3D View's
            angle or a temporary camera) and capture_method: viewport,
            viewport_camera, viewport_screen, render, render_temp_camera_view
            or render_temp_camera_framing.
        """
        method = self._resolve_capture_method(method, use_render_fallback)
        camera = None if camera_name is None else self._resolve_camera(camera_name)
        blender_module: Any = bpy
        engine = blender_module.context.scene.render.engine
        if method == "viewport":
            result = self._capture_view3d(width, height, jpeg_quality, engine, camera)
        else:
            result = self._capture_render(width, height, jpeg_quality, engine, camera)
        return result

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
        Get the render settings and, in attached mode, the 3D View's view state.

        Returns:
            Dictionary with render engine, resolution, film and active camera,
            plus view3d: the largest 3D View's view_perspective (PERSP, ORTHO
            or CAMERA), is_perspective, view_matrix (world to view, rows),
            eye_location (where the view looks from), view_location (the
            point it orbits), view_rotation (quaternion w, x, y, z),
            view_distance, lens, clip_start, clip_end, shading_type,
            view_camera (the camera a CAMERA view looks through) and its
            region size. view3d is None without an editor window (embedded
            bpy, blender -b) or a 3D View, and view3d_unavailable_reason says
            which.
        """
        blender_module: Any = bpy
        scene = blender_module.context.scene
        render = scene.render
        active_cam = scene.camera
        view3d, reason = self._view3d_state()
        return {
            "render_engine": render.engine,
            "resolution_x": render.resolution_x,
            "resolution_y": render.resolution_y,
            "resolution_percentage": render.resolution_percentage,
            "film_transparent": render.film_transparent,
            "active_camera": active_cam.name if active_cam else None,
            "view3d": view3d,
            "view3d_unavailable_reason": reason,
        }

    def _view3d_state(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Describe the 3D View the captures use, or say why there is none."""
        blender_module: Any = bpy
        try:
            _area, region, space = self._view3d_target()
        except RuntimeError:
            if getattr(blender_module.app, "background", False) or getattr(
                blender_module.context, "window", None
            ) is None:
                return None, (
                    "Blender runs without an editor window (embedded bpy or blender -b), "
                    "so there is no 3D View; the render settings still apply"
                )
            return None, "The attached window shows no 3D View with a drawable region"
        view = region.data
        view.update()
        perspective = view.view_perspective
        camera = None
        if perspective == "CAMERA":
            camera = space.camera if space.use_local_camera else blender_module.context.scene.camera
        return {
            "view_perspective": perspective,
            "is_perspective": bool(view.is_perspective),
            "view_matrix": [list(row) for row in view.view_matrix],
            "eye_location": list(view.view_matrix.inverted().translation),
            "view_location": list(view.view_location),
            "view_rotation": list(view.view_rotation),
            "view_distance": view.view_distance,
            "lens": space.lens,
            "clip_start": space.clip_start,
            "clip_end": space.clip_end,
            "shading_type": space.shading.type,
            "view_camera": camera.name if camera is not None else None,
            "region_width": region.width,
            "region_height": region.height,
        }, None

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
        camera_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        max_frames: int = DEFAULT_SEQUENCE_FRAMES,
    ) -> Dict[str, Any]:
        """
        Capture a frame range as JPEG files and return their paths.

        Each frame is captured like capture_viewport and written to output_dir
        as frame_<number>.jpg, so the reply carries paths and sizes rather than
        images: inline images of more than a few frames exceeded the server's
        result budget. Blender's main thread is busy while frames are captured,
        so one call captures at most max_frames frames and stops starting new
        ones after SEQUENCE_TIME_BUDGET_SECONDS; next_start_frame then says
        where to continue. The current frame is restored afterwards.

        Args:
            start_frame: First frame to capture.
            end_frame: Last frame to capture (inclusive); not before start_frame.
            step: Frame step between captures.
            width: Maximum image width, as for capture_blender_viewport.
            height: Maximum image height, as for capture_blender_viewport.
            jpeg_quality: JPEG compression quality.
            agent_id: Visible capture feedback label in attached mode; ignored in embedded mode.
            method: "viewport", "render" or "auto" (default), as for capture_blender_viewport.
            camera_name: Capture through this camera, as for capture_blender_viewport.
            output_dir: Folder for the frame files, inside the sandbox; created when
                missing, and existing frame files of the same numbers are replaced.
                Default: a new sequence-<time>-<id> folder in the capture directory.
            max_frames: Most frames captured by this call, 1 to 100.

        Returns:
            Dictionary with frames ([{frame, file_path}]), frame_count (captured
            now), requested_frame_count (in the range), complete, next_start_frame
            (the first frame left, None when complete), stopped_by (None,
            "max_frames" or "time_budget"), output_dir, width, height, format,
            camera_name and capture_method.

        Raises:
            ValueError: When end_frame is before start_frame, step is below 1,
                or max_frames is out of range.
        """
        if end_frame < start_frame:
            raise ValueError(
                f"end_frame ({end_frame}) is before start_frame ({start_frame})"
            )
        if step < 1:
            raise ValueError(f"step must be at least 1, not {step}")
        if not 1 <= max_frames <= MAX_SEQUENCE_FRAMES:
            raise ValueError(
                f"max_frames must be between 1 and {MAX_SEQUENCE_FRAMES}, not {max_frames}"
            )
        method = self._resolve_capture_method(method)
        if camera_name is not None:
            self._resolve_camera(camera_name)
        folder = self._sequence_folder(output_dir)
        blender_module: Any = bpy
        scene = blender_module.context.scene
        # frame_set(7) alone would turn frame 7.25 into 7.0.
        original_frame = scene.frame_current
        original_subframe = float(getattr(scene, "frame_subframe", 0.0))
        requested = range(start_frame, end_frame + 1, step)

        frames: List[Dict[str, Any]] = []
        capture_method = method
        size: Tuple[Optional[int], Optional[int]] = (None, None)
        stopped_by: Optional[str] = None
        started = time.monotonic()
        try:
            for frame_num in requested:
                if len(frames) >= max_frames:
                    stopped_by = "max_frames"
                    break
                if frames and time.monotonic() - started > SEQUENCE_TIME_BUDGET_SECONDS:
                    stopped_by = "time_budget"
                    break
                scene.frame_set(frame_num)
                result = self.capture_viewport(
                    width=width,
                    height=height,
                    jpeg_quality=jpeg_quality,
                    method=method,
                    camera_name=camera_name,
                )
                path = folder / f"frame_{frame_num:06d}.jpg"
                path.write_bytes(base64.b64decode(result["image_base64"]))
                frames.append({"frame": frame_num, "file_path": str(path)})
                capture_method = result.get("capture_method", capture_method)
                size = (result.get("width"), result.get("height"))
        finally:
            scene.frame_set(original_frame, subframe=original_subframe)

        next_start = requested[len(frames)] if len(frames) < len(requested) else None
        return {
            "frames": frames,
            "frame_count": len(frames),
            "requested_frame_count": len(requested),
            "complete": next_start is None,
            "next_start_frame": next_start,
            "stopped_by": stopped_by,
            "output_dir": str(folder),
            "width": size[0],
            "height": size[1],
            "format": "jpeg",
            "camera_name": camera_name,
            "capture_method": capture_method,
        }

    def _sequence_folder(self, output_dir: Optional[str]) -> Path:
        """Create the folder a sequence's frames go to, inside the sandbox.

        Raises:
            PermissionError: When the sandbox refuses output_dir, or allows no
                writable capture directory for the default one.
        """
        if output_dir is not None:
            folder = Path(self._export_target(self._deny_outside_sandbox(output_dir, write=True)))
        else:
            capture_dir = self._path_policy.default_capture_dir()
            if capture_dir is None:
                raise PermissionError(
                    "No writable capture directory is allowed by the sandbox"
                )
            base = Path(self._deny_outside_sandbox(capture_dir, write=True))
            stamp = time.strftime("%Y%m%d-%H%M%S")
            folder = base / f"sequence-{stamp}-{secrets.token_hex(3)}"
        folder.mkdir(parents=True, exist_ok=True)
        return folder

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

        Supports mesh primitives, lights, cameras, and empties. The object goes
        in the active collection, as Blender's Add menu puts it, unless that
        collection, or one holding it, is excluded from the view layer, hidden
        in it or disabled in viewports: the object would be invisible there
        (and in Blender 5.0.1 the 3D View then left context.active_object
        unset), so it goes in the scene's own collection, and the reply's
        warning says so. The object is told apart from the others by what the
        add operator created, never by the active object, and removed again
        when naming or scaling it fails. It never enters Edit Mode, whatever
        the "Enter Edit Mode" preference says, and outside Object Mode the
        call is refused before anything is added.

        Args:
            object_type: Type key (CUBE, SPHERE, POINT_LIGHT, CAMERA, etc.).
            name: Optional object name.
            location: Initial location [x, y, z].
            rotation_euler: Initial rotation [rx, ry, rz] radians.
            scale: Initial scale [sx, sy, sz].

        Returns:
            Dictionary with name, object_type, location, the collection it
            went in, the mode afterwards and a warning when it went elsewhere.
        """
        blender_module: Any = bpy
        kind = object_type.upper()
        op_path = self._PRIMITIVE_OPS.get(kind)
        if op_path is None:
            raise ValueError(
                f"Unsupported object type: {object_type}. "
                f"Valid types: {', '.join(self._PRIMITIVE_OPS.keys())}"
            )

        loc = _finite(location or [0.0, 0.0, 0.0], 3, "location")
        rot = _finite(rotation_euler or [0.0, 0.0, 0.0], 3, "rotation_euler")
        size = None if scale is None else _finite(scale, 3, "scale")

        parts = op_path.split(".")
        op_module = getattr(blender_module.ops, parts[0])
        op_func = getattr(op_module, parts[1])
        properties: Dict[str, Any] = {"location": loc, "rotation": rot}
        light_type = self._LIGHT_TYPE_MAP.get(kind)
        if light_type:
            properties["type"] = light_type
        if "enter_editmode" in op_func.get_rna_type().properties.keys():
            # The add operators read the preference when this is unset.
            properties["enter_editmode"] = False

        view_layer = blender_module.context.view_layer
        mode_before = blender_module.context.mode
        if mode_before != "OBJECT":
            # In Edit Mode an add operator adds to the edited mesh instead of
            # creating an object. Attached mode's bridge refuses this up front;
            # embedded mode has no such guard.
            raise ValueError(
                f"Blender is in {mode_before} mode; set_blender_object_mode switches back to "
                "Object Mode, where objects can be created"
            )
        active_collection = view_layer.active_layer_collection
        target, warning = self._creation_collection(view_layer)
        existing = {obj.as_pointer() for obj in blender_module.data.objects}

        def created_objects() -> List[Any]:
            return [
                obj
                for obj in blender_module.data.objects
                if obj.as_pointer() not in existing
            ]

        # bpy gives a new wrapper on every access; == compares what it wraps.
        switch = target != active_collection
        try:
            if switch:
                view_layer.active_layer_collection = target
            try:
                _bpy_op(f"Failed to add a {kind}", op_func, **properties)
            finally:
                if switch:
                    view_layer.active_layer_collection = active_collection
            created = created_objects()
            if len(created) != 1:
                raise RuntimeError(
                    f"Adding a {kind} created {len(created)} objects, not one"
                )
            obj = created[0]
            if blender_module.context.mode != "OBJECT":
                with blender_module.context.temp_override(active_object=obj, object=obj):
                    _bpy_op(
                        f"Failed to return '{obj.name}' to Object Mode",
                        blender_module.ops.object.mode_set,
                        mode="OBJECT",
                    )
            if name:
                obj.name = name
            if size is not None:
                obj.scale = size
        except Exception:
            # Never leave an unnamed, unscaled object behind a failed call.
            for stray in created_objects():
                self._remove_object_and_data(stray)
            raise

        return {
            "name": obj.name,
            "object_type": kind,
            "location": [float(v) for v in obj.location],
            "collection": target.collection.name,
            "mode": blender_module.context.mode,
            "warning": warning,
        }

    @staticmethod
    def _creation_collection(view_layer: Any) -> Tuple[Any, Optional[str]]:
        """
        Choose the layer collection a new object goes in.

        Args:
            view_layer: The view layer the object is added to.

        Returns:
            The active layer collection and None, or the view layer's root
            (the scene's own collection) and the reason, when the active one
            or one holding it is excluded, hidden or disabled in viewports.
        """
        root = view_layer.layer_collection
        active = view_layer.active_layer_collection

        def path_to(node: Any) -> Optional[List[Any]]:
            if node == active:
                return [node]
            for child in node.children:
                found = path_to(child)
                if found is not None:
                    return [node, *found]
            return None

        for layer in (path_to(root) or [active])[1:]:
            reason = (
                "is excluded from the view layer"
                if layer.exclude
                else "is hidden in the view layer"
                if layer.hide_viewport
                else "is disabled in viewports"
                if layer.collection.hide_viewport
                else None
            )
            if reason is not None:
                return root, (
                    f"The active collection '{active.collection.name}' is not visible: "
                    f"collection '{layer.collection.name}' {reason}, so the object was "
                    f"created in the scene collection '{root.collection.name}' instead"
                )
        return active, None

    @staticmethod
    def _remove_object_and_data(obj: Any) -> None:
        """Remove an object, and its data (mesh, light, camera) once nothing else uses it."""
        blender_module: Any = bpy
        data = obj.data
        blender_module.data.objects.remove(obj, do_unlink=True)
        if data is None or data.users:
            return
        for store in ("meshes", "lights", "cameras"):
            collection = getattr(blender_module.data, store)
            if collection.get(data.name) == data:
                collection.remove(data)
                return

    def delete_object(self, object_name: str, use_global: bool = False) -> Dict[str, Any]:
        """
        Delete an object from the current scene, as Blender's Delete does.

        This runs Blender's own Delete operator on the object alone, the
        user's selection untouched. As in Blender, the object leaves the
        current scene's collections and stays in the file while another scene
        still links it; with use_global it is deleted from every scene (Delete
        Globally). The operator also resets the rigid-body cache of a deleted
        rigid body: removing the object directly left that cache in place, so
        another body evaluated after a frame jump took the deleted one's cached
        positions.

        Args:
            object_name: Name of the object to delete.
            use_global: Delete it from every scene and the file, not only from
                the current scene.

        Returns:
            Dictionary with deleted_name, use_global, whether the object left
            the file (removed_from_file) and the scenes that still link it.

        Raises:
            ValueError: Without use_global, the object is not in the current
                scene (embedded mode looks objects up in the whole file).
            RuntimeError: Blender did not delete it, as for linked library data.
        """
        blender_module: Any = bpy
        obj = self._get_object_or_raise(object_name)
        scene = blender_module.context.scene
        if not use_global and scene.objects.get(object_name) is None:
            raise ValueError(
                f"Object '{object_name}' is not in scene '{scene.name}'; pass "
                "use_global=true to delete it from every scene"
            )
        operator = blender_module.ops.object.delete
        properties: Dict[str, Any] = {"use_global": use_global}
        if "confirm" in operator.get_rna_type().properties.keys():
            properties["confirm"] = False
        with blender_module.context.temp_override(
            selected_objects=[obj],
            selected_editable_objects=[obj],
            active_object=obj,
            object=obj,
        ):
            _bpy_op(f"Failed to delete '{object_name}'", operator, **properties)
        if scene.objects.get(object_name) is not None:
            raise RuntimeError(
                f"Blender did not delete '{object_name}' (linked library data cannot be deleted)"
            )
        return {
            "deleted_name": object_name,
            "use_global": use_global,
            "removed_from_file": blender_module.data.objects.get(object_name) is None,
            "remaining_scenes": [
                other.name
                for other in blender_module.data.scenes
                if other.objects.get(object_name) is not None
            ],
        }

    def set_object_mode(
        self, mode: str = "OBJECT", object_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Switch an object into a mode, as the 3D View's mode menu does.

        The way back to Object Mode for a window left in Edit or Pose Mode,
        where object creation and the other mode-sensitive tools refuse to run.
        With object_name, that object becomes the active one first.

        Args:
            mode: OBJECT (default), EDIT, POSE, SCULPT, VERTEX_PAINT,
                WEIGHT_PAINT, TEXTURE_PAINT or another mode Blender's
                object.mode_set takes, in any case.
            object_name: The object to switch; the active object by default.

        Returns:
            object_name, mode (the object's afterwards), previous_mode,
            context_mode (Blender's, such as EDIT_MESH) and changed.

        Raises:
            ValueError: The mode is not one Blender takes, or there is no
                object to switch to a mode other than OBJECT.
        """
        blender_module: Any = bpy
        operator = blender_module.ops.object.mode_set
        wanted = mode.strip().upper()
        modes = [
            item.identifier
            for item in operator.get_rna_type().properties["mode"].enum_items
        ]
        if wanted not in modes:
            raise ValueError(f"Unknown mode {mode!r}; Blender's modes: {', '.join(modes)}")
        view_layer = blender_module.context.view_layer
        obj = (
            view_layer.objects.active
            if object_name is None
            else self._get_object_or_raise(object_name)
        )
        if obj is None:
            if wanted == "OBJECT":
                return {
                    "object_name": None,
                    "mode": "OBJECT",
                    "previous_mode": "OBJECT",
                    "context_mode": blender_module.context.mode,
                    "changed": False,
                }
            raise ValueError(f"No active object to switch to {wanted} mode; pass object_name")
        previous = obj.mode
        if previous != wanted:
            was_active = view_layer.objects.active
            if was_active != obj:
                view_layer.objects.active = obj
            try:
                with blender_module.context.temp_override(active_object=obj, object=obj):
                    _bpy_op(
                        f"Failed to switch '{obj.name}' to {wanted} mode",
                        operator,
                        mode=wanted,
                    )
            except Exception:
                # A refused switch leaves the user's active object as it was.
                view_layer.objects.active = was_active
                raise
        return {
            "object_name": obj.name,
            "mode": obj.mode,
            "previous_mode": previous,
            "context_mode": blender_module.context.mode,
            "changed": obj.mode != previous,
        }

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
        metallic: Optional[float] = None,
        roughness: Optional[float] = None,
        slot_index: Optional[int] = None,
        face_indices: Optional[List[int]] = None,
        link: str = "DATA",
        single_user: bool = False,
        reuse_existing: bool = True,
        base_color_texture: Optional[str] = None,
        metallic_texture: Optional[str] = None,
        roughness_texture: Optional[str] = None,
        normal_texture: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Assign a Principled BSDF material to an object, reusing it when it exists.

        The material named ``material_name`` is reused when it exists (so a
        repeated call changes nothing more), and created otherwise; pass
        ``reuse_existing=False`` to always create a new one. Shading arguments
        left out keep the material's own values (Blender's defaults for a new
        one), so an existing material can be assigned as it is.

        By default the material replaces the object's: it goes into slot 0
        and every face uses slot 0. ``slot_index`` puts it into that slot
        instead (the slot count appends one) and leaves the faces alone;
        ``face_indices`` assigns it to those faces only, through its existing
        slot or a new one. ``link="OBJECT"`` puts it in this object's slot
        instead of the mesh's, but faces and the slot count belong to the
        mesh: with link OBJECT, changing either on mesh data that other
        objects share is refused unless ``single_user`` gives this object its
        own copy of the mesh first.

        An image already loaded in the file keeps its colour space: an input
        that reads it differently gets its own image datablock. Everything is
        checked before the scene changes, and a failure later removes what
        the call created.

        Args:
            object_name: Object to assign the material to (mesh, curve, text,
                or another type with material slots).
            material_name: Material name; Material_<object> when omitted.
            base_color: RGB or RGBA in [0, 1]; the alpha becomes the
                Principled Alpha, and below 1 the material renders blended.
            metallic: Metallic value 0-1.
            roughness: Roughness value 0-1.
            slot_index: Slot to put the material in; the slot count adds one.
            face_indices: Mesh faces to assign the material to.
            link: DATA (the mesh's slot, shared by linked duplicates) or
                OBJECT (this object's slot).
            single_user: Give this object its own copy of shared data first,
                so the change leaves the objects sharing it alone.
            reuse_existing: Reuse a material of that name when one exists.
            base_color_texture: Image file driving Base Color (in the colour
                space Blender loads it in: sRGB, or linear for float images).
            metallic_texture: Image file driving Metallic (Non-Color).
            roughness_texture: Image file driving Roughness (Non-Color).
            normal_texture: Tangent-space normal map image (Non-Color),
                through a Normal Map node.

        Returns:
            Dictionary with the material, whether it was created, the slot,
            the link, the faces assigned, the object's slots, the Principled
            values, its render method, the textures and warnings.
        """
        blender_module: Any = bpy
        obj = self._get_object_or_raise(object_name)
        data = obj.data
        if data is None or not hasattr(data, "materials"):
            raise ValidationError(
                f"Object '{object_name}' is a {obj.type} object, which takes no materials",
                "no_material_slots",
                object_type=obj.type,
            )
        if getattr(obj, "library", None) is not None:
            raise ValidationError(
                f"Object '{object_name}' is linked from library '{obj.library.name}' and "
                "cannot be changed; make it local first",
                "linked_from_library",
            )
        link = link.upper() if isinstance(link, str) else link
        if link not in ("DATA", "OBJECT"):
            raise ValidationError(
                f"link must be DATA or OBJECT, got {link!r}", "invalid_value", field="link"
            )
        color = None if base_color is None else _unit_color(base_color, "base_color")
        for label, value in (("metallic", metallic), ("roughness", roughness)):
            if value is not None:
                _unit_value(value, label)
        slot_count = len(obj.material_slots)
        requested_slot: Any = slot_index
        if requested_slot is not None and (
            isinstance(requested_slot, bool)
            or not isinstance(requested_slot, numbers.Integral)
            or not 0 <= int(requested_slot) <= slot_count
        ):
            raise ValidationError(
                f"slot_index must lie between 0 and {slot_count} (the slot count, which "
                f"adds a slot), got {slot_index!r}",
                "out_of_range",
                field="slot_index",
            )
        faces = None
        if face_indices is not None:
            if obj.type != "MESH":
                raise ValidationError(
                    f"face_indices applies to mesh objects; '{object_name}' is a {obj.type} object",
                    "invalid_value",
                    field="face_indices",
                )
            faces = self._checked_face_indices(data, face_indices)
        writes_faces = obj.type == "MESH" and (faces is not None or slot_index is None)
        if writes_faces:
            self._require_object_mode(obj, "assign materials to its faces")
        textures = self._texture_files(
            {
                "base_color": base_color_texture,
                "metallic": metallic_texture,
                "roughness": roughness_texture,
                "normal": normal_texture,
            }
        )
        name = material_name or f"Material_{object_name}"
        material = blender_module.data.materials.get(name) if reuse_existing else None
        shading = bool(
            color is not None or metallic is not None or roughness is not None or textures
        )
        if material is not None and shading:
            if getattr(material, "library", None) is not None:
                raise ValidationError(
                    f"Material '{name}' is linked from a library and cannot be changed; assign "
                    "it without shading arguments, or pass reuse_existing=false for a local one",
                    "linked_from_library",
                )
            if self._principled_node(material) is None:
                raise ValidationError(
                    f"Material '{name}' has no Principled BSDF to take base_color, metallic, "
                    "roughness or textures; assign it without them, or pass "
                    "reuse_existing=false for a new material",
                    "no_principled_bsdf",
                )

        # Where the material goes and what that does to the data, worked out
        # before anything changes.
        if faces is not None and slot_index is None:
            index = next(
                (i for i, slot in enumerate(obj.material_slots) if slot.material == material),
                slot_count,
            )
        else:
            index = 0 if slot_index is None else int(slot_index)
        appends = index == slot_count
        previous_faces: Optional[List[int]] = None
        new_faces: Optional[List[int]] = None
        if writes_faces:
            previous_faces = self._face_materials(data)
            new_faces = list(previous_faces)
            for face in range(len(new_faces)) if faces is None else faces:
                new_faces[face] = index
        face_change = new_faces is not None and new_faces != previous_faces
        changes_data = link == "DATA" or appends or face_change
        sharers = self._other_users(obj)
        linked_data = getattr(data, "library", None) is not None
        # single_user also makes library data local, as Blender's Make Single User does.
        copy_data = bool(single_user and (sharers or linked_data))
        kind = obj.type.title() if obj.type != "MESH" else "Mesh"
        if sharers and link == "OBJECT" and (appends or face_change) and not copy_data:
            what = "assigning faces" if face_change else "adding a slot"
            raise ValidationError(
                f"{kind} data '{data.name}' is shared with {_name_list(sharers)}; with "
                f"link='OBJECT' the slot's material is this object's, but {what} changes the "
                f"shared data for all of them. Pass single_user=true to give '{object_name}' "
                "its own copy first, or slot_index to an existing slot to change only the "
                "material",
                "shared_data",
                shared_with=sharers,
            )
        if linked_data and changes_data and not copy_data:
            raise ValidationError(
                f"{kind} data '{data.name}' is linked from a library and cannot be changed; "
                "pass single_user=true for a local copy, or link='OBJECT' with slot_index "
                "to an existing slot",
                "linked_from_library",
            )
        if copy_data:
            # Blender 5.0.1 ignores obj.data = copy, without an error, while
            # the object is in Edit Mode.
            self._require_object_mode(obj, "give it its own copy of its data")
        warnings = self._material_warnings(obj, link, textures, sharers, copy_data)
        replaced = None if appends else obj.material_slots[index].material

        created = material is None
        new_images: List[Any] = []
        appended = False
        made_copy: Any = None
        # Set once the copy really is the object's data.
        original_data: Any = None
        previous_slot: Optional[Tuple[int, str, Any]] = None
        faces_written = False
        try:
            images = self._load_texture_images(textures, new_images)
            if copy_data:
                made_copy = data.copy()
                obj.data = made_copy
                if obj.data != made_copy:
                    raise RuntimeError(
                        f"Blender kept '{data.name}' on '{object_name}' instead of its copy; "
                        "nothing was changed"
                    )
                original_data, data = data, made_copy
            if material is None:
                material = blender_module.data.materials.new(name=name)
                if material.node_tree is None:
                    # Before Blender 5.0 a new material has no node tree until
                    # use_nodes is set; 5.0 always has one and deprecates the flag.
                    material.use_nodes = True
            if shading:
                self._shade_principled(material, color, metallic, roughness, images)
            if appends:
                data.materials.append(material if link == "DATA" else None)
                appended = True
            slot = obj.material_slots[index]
            previous_slot = (index, slot.link, slot.material)
            slot.link = link
            slot.material = material
            if face_change and new_faces is not None:
                faces_written = True
                self._set_face_materials(data, new_faces)
        except Exception:
            # Each step is undone on its own, so one that fails does not keep
            # the others from running.
            def undo(step: Callable[[], Any]) -> None:
                try:
                    step()
                except Exception:
                    self.logger.exception("Could not undo part of a failed material assignment")

            def restore_faces() -> None:
                if faces_written and previous_faces is not None and original_data is None:
                    self._set_face_materials(data, previous_faces)

            def restore_data() -> None:
                # Only the copy this call made is removed, its faces and slots
                # with it, and only once no object uses it: removing data an
                # object still uses deletes that object.
                if made_copy is None:
                    return
                if original_data is not None and obj.data == made_copy:
                    obj.data = original_data
                if obj.data == made_copy:
                    self.logger.error(
                        "Left the copy '%s' on '%s': Blender did not take the original back",
                        made_copy.name,
                        object_name,
                    )
                    return
                self._data_collection(obj).remove(made_copy)

            def restore_slot() -> None:
                # The slot's link and an OBJECT-linked material are the
                # object's, so they are restored also after a copy is dropped.
                if appended:
                    if original_data is None:
                        data.materials.pop(index=-1)
                elif previous_slot is not None:
                    restored = obj.material_slots[previous_slot[0]]
                    restored.link = previous_slot[1]
                    restored.material = previous_slot[2]

            def remove_material() -> None:
                if created and material is not None:
                    blender_module.data.materials.remove(material)

            def remove_images() -> None:
                for image in new_images:
                    blender_module.data.images.remove(image)

            for step in (restore_faces, restore_data, restore_slot, remove_material, remove_images):
                undo(step)
            raise

        if original_data is not None:
            library = getattr(original_data, "library", None)
            if library is not None:
                # A local copy of library data keeps its name.
                kept = f"; {_name_list(sharers)} keep the linked one" if sharers else ""
                warnings.append(
                    f"'{object_name}' now has a local copy of {kind.lower()} data "
                    f"'{original_data.name}' from library '{library.name}'{kept}"
                )
            else:
                kept = f", which {_name_list(sharers)} keep" if sharers else ""
                warnings.append(
                    f"'{object_name}' now has its own copy '{data.name}' of {kind.lower()} data "
                    f"'{original_data.name}'{kept}"
                )
        if not created and shading and material.users > 1:
            warnings.append(
                f"Material '{material.name}' has {material.users} users; the changed shading "
                "shows wherever it is used"
            )
        if replaced is not None and replaced != material and not replaced.users:
            warnings.append(
                f"Material '{replaced.name}' has no users now; Blender discards it when the "
                "file is saved and reopened (set its use_fake_user to keep it)"
            )
        bsdf = self._principled_node(material)
        return {
            "object_name": object_name,
            "material_name": material.name,
            "created": created,
            "slot_index": index,
            "link": link,
            "faces_assigned": (
                None if new_faces is None else len(new_faces) if faces is None else len(faces)
            ),
            "material_slots": [
                slot.material.name if slot.material else None for slot in obj.material_slots
            ],
            "principled_params": None if bsdf is None else self._extract_principled_params(bsdf),
            "surface_render_method": self._render_method(material),
            "textures": textures,
            "warnings": warnings,
        }

    @staticmethod
    def _other_users(obj: Any) -> List[str]:
        """The other objects whose data is ``obj``'s, by name.

        Counted from the objects rather than ``data.users``, which a fake
        user or another datablock's reference raises too.
        """
        blender_module: Any = bpy
        data = obj.data
        return sorted(
            other.name
            for other in blender_module.data.objects
            if other.name != obj.name and getattr(other, "data", None) == data
        )

    @staticmethod
    def _data_collection(obj: Any) -> Any:
        """The bpy.data collection that holds an object's data."""
        blender_module: Any = bpy
        name = {
            "MESH": "meshes",
            "CURVE": "curves",
            "SURFACE": "curves",
            "FONT": "curves",
            "META": "metaballs",
            "CURVES": "hair_curves",
            "POINTCLOUD": "pointclouds",
            "VOLUME": "volumes",
            "GREASEPENCIL": "grease_pencils",
        }.get(obj.type, "meshes")
        return getattr(blender_module.data, name)

    @staticmethod
    def _checked_face_indices(mesh: Any, face_indices: Sequence[Any]) -> List[int]:
        """Face indices in range, each once, at least one, in the order given."""
        if not face_indices:
            raise ValidationError(
                "face_indices must name at least one face", "invalid_value", field="face_indices"
            )
        count = len(mesh.polygons)
        checked: List[int] = []
        seen: set = set()
        for value in face_indices:
            if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                raise ValidationError(
                    f"face_indices index {value!r} is not an integer",
                    "invalid_value",
                    field="face_indices",
                )
            index = int(value)
            if index < 0 or index >= count:
                raise ValidationError(
                    f"face_indices index {index} out of range [0, {count})",
                    "out_of_range",
                    field="face_indices",
                )
            if index in seen:
                raise ValidationError(
                    f"face_indices lists face {index} twice", "invalid_value", field="face_indices"
                )
            seen.add(index)
            checked.append(index)
        return checked

    def _texture_files(self, textures: Dict[str, Optional[str]]) -> Dict[str, str]:
        """The texture paths given, each allowed by the sandbox and an existing file."""
        files: Dict[str, str] = {}
        for key, path in textures.items():
            if path is None:
                continue
            resolved = self._deny_outside_sandbox(path)
            if not os.path.isfile(resolved):
                raise NotFoundError(
                    f"{key}_texture '{path}' is not a file",
                    "texture_not_found",
                    field=f"{key}_texture",
                    path=path,
                )
            files[key] = resolved
        return files

    @staticmethod
    def _load_texture_images(textures: Dict[str, str], new_images: List[Any]) -> Dict[str, Any]:
        """Find or load each texture image, in the colour space its socket reads it in.

        A data map (metallic, roughness, normal) is read Non-Color; a base
        colour reads the colour space Blender loads the file in: sRGB for an
        8-bit image, linear for a float one such as an EXR. An image of the same
        file already read that way is reused, so a repeated call adds no
        datablock. An image already in the file is never changed, since other
        materials may read it: when every one of the file reads the other way,
        this input gets a datablock of its own. Images the call loads are added
        to ``new_images``, for the caller to remove on failure.
        bpy.data.images.load accepts a file that is no image and only fails to
        read it later, so an image that reads as 0 x 0 pixels is refused.
        """
        blender_module: Any = bpy
        images: Dict[str, Any] = {}
        for key, path in textures.items():
            data_map = key != "base_color"
            same_file = [
                image
                for image in blender_module.data.images
                if _image_file(image) == _real_path(path)
            ]
            image = next(
                (
                    candidate
                    for candidate in same_file
                    if bool(candidate.colorspace_settings.is_data) == data_map
                ),
                None,
            )
            fresh = image is None
            if image is None:
                image = blender_module.data.images.load(path, check_existing=False)
                new_images.append(image)
            if not image.size[0]:
                raise ValidationError(
                    f"{key}_texture '{path}' is not an image Blender can read",
                    "not_an_image",
                    field=f"{key}_texture",
                    path=path,
                )
            if fresh and data_map:
                image.colorspace_settings.name = "Non-Color"
            images[key] = image
        return images

    def _material_warnings(
        self,
        obj: Any,
        link: str,
        textures: Dict[str, str],
        sharers: List[str],
        copy_data: bool,
    ) -> List[str]:
        """Warnings about where an assignment shows: shared data, missing UVs."""
        warnings: List[str] = []
        data = obj.data
        if link == "DATA" and sharers and not copy_data:
            kind = "Mesh" if obj.type == "MESH" else obj.type.title()
            warnings.append(
                f"{kind} data '{data.name}' is shared with {_name_list(sharers)}; its slots "
                "and face assignments change on all of them (single_user=true changes only "
                "this object)"
            )
        if textures and obj.type == "MESH" and not len(data.uv_layers):
            warnings.append(
                f"Mesh '{data.name}' has no UV map, which image textures need; run "
                "uv_unwrap_blender_mesh on the object"
            )
        return warnings

    def _shade_principled(
        self,
        material: Any,
        color: Optional[Tuple[float, float, float, Optional[float]]],
        metallic: Optional[float],
        roughness: Optional[float],
        images: Dict[str, Any],
    ) -> None:
        """Set the given values and textures on the material's Principled BSDF."""
        bsdf = self._principled_node(material)
        if color is not None:
            red, green, blue, alpha = color
            bsdf.inputs["Base Color"].default_value = (red, green, blue, 1.0)
            material.diffuse_color = (red, green, blue, 1.0 if alpha is None else alpha)
            if alpha is not None:
                bsdf.inputs["Alpha"].default_value = alpha
                self._set_render_method(material, transparent=alpha < 1.0)
        if metallic is not None:
            bsdf.inputs["Metallic"].default_value = metallic
        if roughness is not None:
            bsdf.inputs["Roughness"].default_value = roughness
        for row, (key, image) in enumerate(images.items()):
            self._link_texture(material.node_tree, bsdf, key, image, row)

    @staticmethod
    def _set_render_method(material: Any, transparent: bool) -> None:
        """Blend the material when it is transparent, as alpha alone does not.

        Blender 4.2+ (EEVEE Next) sets this as surface_render_method; earlier
        versions as blend_method.
        """
        if hasattr(material, "surface_render_method"):
            material.surface_render_method = "BLENDED" if transparent else "DITHERED"
        else:
            material.blend_method = "BLEND" if transparent else "OPAQUE"

    @staticmethod
    def _render_method(material: Any) -> Optional[str]:
        """The material's render method (surface_render_method, else blend_method)."""
        for name in ("surface_render_method", "blend_method"):
            value = getattr(material, name, None)
            if value is not None:
                return str(value)
        return None

    def _link_texture(self, tree: Any, bsdf: Any, key: str, image: Any, row: int) -> None:
        """Drive one Principled input with an image texture, replacing what drove it."""
        socket = bsdf.inputs[_TEXTURE_SOCKETS[key]]
        for old in list(socket.links):
            source = old.from_node
            tree.links.remove(old)
            self._drop_unused_texture_node(tree, source)
        x, y = bsdf.location[0], bsdf.location[1]
        texture = tree.nodes.new("ShaderNodeTexImage")
        texture.image = image
        texture.label = f"{_TEXTURE_SOCKETS[key]} Texture"
        texture.location = (x - 700, y - 300 * row)
        if key == "normal":
            normal_map = tree.nodes.new("ShaderNodeNormalMap")
            normal_map.location = (x - 250, y - 300 * row)
            tree.links.new(texture.outputs["Color"], normal_map.inputs["Color"])
            tree.links.new(normal_map.outputs["Normal"], socket)
        else:
            tree.links.new(texture.outputs["Color"], socket)

    def _drop_unused_texture_node(self, tree: Any, node: Any) -> None:
        """Remove an Image Texture, Normal Map or Reroute node that no longer feeds anything.

        The nodes upstream of it go too when nothing else uses them, so a
        replaced texture behind Reroutes leaves no stray nodes.
        """
        if node.type not in ("TEX_IMAGE", "NORMAL_MAP", "REROUTE") or any(
            output.is_linked for output in node.outputs
        ):
            return
        upstream = [link.from_node for socket in node.inputs for link in socket.links]
        tree.nodes.remove(node)
        for source in upstream:
            self._drop_unused_texture_node(tree, source)

    @staticmethod
    def _material_output(tree: Any) -> Any:
        """The node tree's active Material Output, else its first one, else None."""
        outputs = [node for node in tree.nodes if node.type == "OUTPUT_MATERIAL"]
        return next((node for node in outputs if node.is_active_output), None) or next(
            iter(outputs), None
        )

    def _material_tree(self, material: Any) -> Any:
        """The material's node tree when it renders with nodes, else None.

        Blender 5.0 materials always use nodes and deprecate Material.use_nodes
        (removed in 6.0), so it is read only before 5.0.
        """
        tree = getattr(material, "node_tree", None)
        if tree is None:
            return None
        if self._is_blender_5_plus:
            return tree
        return tree if material.use_nodes else None

    def _principled_node(self, material: Any) -> Any:
        """The Principled BSDF feeding the active output's Surface, else the first one."""
        tree = self._material_tree(material)
        if tree is None:
            return None
        output = self._material_output(tree)
        surface = None if output is None else output.inputs.get("Surface")
        for link in [] if surface is None else _live_links(surface):
            source = _link_source(link)[0]
            if source.type == "BSDF_PRINCIPLED":
                return source
        return next((node for node in tree.nodes if node.type == "BSDF_PRINCIPLED"), None)

    @staticmethod
    def _face_materials(mesh: Any) -> List[int]:
        """Every face's material index, in face order."""
        values = [0] * len(mesh.polygons)
        mesh.polygons.foreach_get("material_index", values)
        return values

    @staticmethod
    def _set_face_materials(mesh: Any, values: List[int]) -> None:
        """Write every face's material index."""
        mesh.polygons.foreach_set("material_index", values)
        mesh.update()

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
        """Refuse to change an object's data while it, or its data, is edited in another mode.

        Edit Mode keeps its own copy of the data and writes it back on leaving,
        over what was changed meanwhile, also when the data is edited through
        another object that shares it.
        """
        mode = getattr(obj, "mode", "OBJECT")
        if mode != "OBJECT":
            raise ValidationError(
                f"Object '{obj.name}' is in {mode.replace('_', ' ').title()} Mode; "
                f"switch it to Object Mode to {action}",
                "wrong_mode",
                mode=mode,
            )
        data: Any = getattr(obj, "data", None)
        if getattr(data, "is_editmode", False):
            raise ValidationError(
                f"The data '{data.name}' of '{obj.name}' is in Edit Mode through another "
                f"object that shares it; switch that object to Object Mode to {action}",
                "wrong_mode",
                mode="EDIT",
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
                raise NotFoundError(
                    f"{what}: object '{value}' not found", "object_not_found", object_name=value
                ) from None
        blender_module: Any = bpy
        datablock = getattr(blender_module.data, _ID_COLLECTIONS[rna.identifier]).get(value)
        if datablock is None:
            raise NotFoundError(
                f"{what}: {type_name} '{value}' not found", "datablock_not_found", name=value
            )
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
        light_type: Optional[str] = None,
        angle: Optional[float] = None,
        size: Optional[float] = None,
        size_y: Optional[float] = None,
        shape: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Set properties on a light object, checked for its type and range.

        Each setting must belong to the light's type (after ``light_type``,
        when given): shadow_soft_size is the radius of POINT and SPOT lights,
        spot_size and spot_blend are SPOT's, angle is a SUN's softness, and
        shape, size and size_y are an AREA light's (size_y for RECTANGLE and
        ELLIPSE). A value outside its range is refused rather than clamped,
        and energy may not be negative. Nothing changes unless every value is
        valid. The reply reports every setting of the light's type as stored.

        Args:
            light_name: Light object name.
            energy: Light power in watts (irradiance in W/m² for SUN), >= 0.
            color: Light color RGB [r, g, b], each >= 0.
            use_shadow: Enable shadow casting.
            spot_size: Spot cone angle in radians, 1 degree to pi (SPOT).
            spot_blend: Spot edge softness 0-1 (SPOT).
            shadow_soft_size: Light radius for soft shadows, >= 0 (POINT, SPOT).
            light_type: Change the light to POINT, SUN, SPOT or AREA first.
            angle: Sun's angular diameter in radians, 0 to pi (SUN).
            size: Area light size, >= 0 (AREA; its width for RECTANGLE and ELLIPSE).
            size_y: Area light height, >= 0 (AREA with shape RECTANGLE or ELLIPSE).
            shape: Area light shape: SQUARE, RECTANGLE, DISK or ELLIPSE (AREA).

        Returns:
            Dictionary with light_name, light_type, energy, color, use_shadow,
            the settings of its type and warnings.
        """
        obj = self._get_object_or_raise(light_name)
        if obj.type != "LIGHT":
            raise ValidationError(
                f"Object '{light_name}' is type '{obj.type}', not LIGHT",
                "not_a_light",
                object_type=obj.type,
            )
        light = obj.data
        target = light.type if light_type is None else str(light_type).upper()
        if target not in _LIGHT_TYPE_SETTINGS:
            raise ValidationError(
                f"light_type must be one of {', '.join(_LIGHT_TYPE_SETTINGS)}, got {light_type!r}",
                "invalid_value",
                field="light_type",
            )
        typed: Dict[str, Any] = {
            name: value
            for name, value in (
                ("shadow_soft_size", shadow_soft_size),
                ("spot_size", spot_size),
                ("spot_blend", spot_blend),
                ("angle", angle),
                ("shape", shape),
                ("size", size),
                ("size_y", size_y),
            )
            if value is not None
        }
        for name in typed:
            if name not in _LIGHT_TYPE_SETTINGS[target]:
                raise ValidationError(
                    self._light_setting_error(name, light_name, target),
                    "setting_not_for_light_type",
                    field=name,
                    light_type=target,
                )
        numbers_given: Dict[str, Any] = {
            name: value for name, value in typed.items() if name != "shape"
        }
        if energy is not None:
            numbers_given["energy"] = energy
        for name, value in numbers_given.items():
            low, high = _LIGHT_LIMITS[name]
            number = _finite_number(value, name)
            if number < low or (high is not None and number > high):
                bound = f"at least {low:g}" if high is None else f"between {low:g} and {high:g}"
                raise ValidationError(f"{name} must be {bound}, got {value!r}", "out_of_range", field=name)
        rgb: Any = color
        if rgb is not None:
            if isinstance(rgb, (str, bytes)) or not hasattr(rgb, "__len__") or len(rgb) != 3:
                raise ValidationError(f"color must be [r, g, b], got {rgb!r}", "invalid_value", field="color")
            if any(_finite_number(channel, "color") < 0 for channel in rgb):
                raise ValidationError(
                    f"color channels must be at least 0, got {rgb!r}", "out_of_range", field="color"
                )
        shadows: Any = use_shadow
        if shadows is not None and not isinstance(shadows, bool):
            raise ValidationError(
                f"use_shadow must be true or false, got {shadows!r}", "invalid_value", field="use_shadow"
            )
        if shape is not None:
            shape = str(shape).upper()
            if shape not in _AREA_LIGHT_SHAPES:
                raise ValidationError(
                    f"shape must be one of {', '.join(_AREA_LIGHT_SHAPES)}, got {shape!r}",
                    "invalid_value",
                    field="shape",
                )
        if size_y is not None:
            effective = shape or (light.shape if light.type == "AREA" else "SQUARE")
            if effective not in ("RECTANGLE", "ELLIPSE"):
                raise ValidationError(
                    f"size_y applies to RECTANGLE and ELLIPSE area lights, and '{light_name}' "
                    f"would be {effective}; pass shape as well",
                    "setting_not_for_light_type",
                    field="size_y",
                    shape=effective,
                )

        if light.type != target:
            light.type = target
            # The light datablock's Python struct keeps its old type's
            # settings until it is fetched again.
            light = obj.data
        if energy is not None:
            light.energy = energy
        if color is not None:
            light.color = tuple(color)
        if use_shadow is not None:
            light.use_shadow = use_shadow
        if shape is not None:
            light.shape = shape
        for name, value in typed.items():
            if name != "shape":
                setattr(light, name, value)

        result: Dict[str, Any] = {
            "light_name": light_name,
            "light_type": light.type,
            "energy": light.energy,
            "color": list(light.color),
            "use_shadow": bool(light.use_shadow),
        }
        for name in _LIGHT_TYPE_SETTINGS[light.type]:
            result[name] = getattr(light, name)
        users = int(getattr(light, "users", 1))
        result["warnings"] = (
            [f"Light data '{light.name}' is shared by {users} objects; all of them changed"]
            if users > 1
            else []
        )
        return result

    @staticmethod
    def _light_setting_error(name: str, light_name: str, light_type: str) -> str:
        """Why a setting does not apply to a light type, and what does instead."""
        owners = [kind for kind, settings in _LIGHT_TYPE_SETTINGS.items() if name in settings]
        message = (
            f"{name} applies to {' and '.join(owners)} lights, and '{light_name}' is a "
            f"{light_type} light"
        )
        if name == "shadow_soft_size" and light_type == "SUN":
            return f"{message}; a sun's softness is its angle"
        if name == "shadow_soft_size" and light_type == "AREA":
            return f"{message}; an area light's softness is its size"
        return f"{message}; pass light_type to change its type"

    def uv_unwrap_mesh(
        self,
        object_name: str,
        method: str = "SMART_PROJECT",
        uv_map_name: Optional[str] = None,
        angle_limit: float = math.radians(66.0),
        island_margin: float = 0.0,
        cube_size: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Generate UV coordinates for every face of a mesh object, hidden ones included.

        SMART_PROJECT splits the mesh into islands where faces meet at more
        than ``angle_limit`` and packs them (Blender's Smart UV Project);
        CUBE_PROJECT projects each face onto the cube side it faces, at
        ``cube_size``. The result goes into ``uv_map_name`` (created when
        missing), else the active UV map, else a new "UVMap". The active UV
        map stays what it was (a mesh's first UV map becomes active and the
        render UV map, as Blender makes it); the reply names the active and
        render UV maps, the one image textures read. The operators need Edit
        Mode, so the object briefly becomes the active, only selected object
        and its hidden faces are shown; its mesh selection and hidden faces,
        the object selection and the active object are restored afterwards.
        The object, and Blender, must be in Object Mode.

        Args:
            object_name: Mesh object to unwrap.
            method: SMART_PROJECT or CUBE_PROJECT.
            uv_map_name: UV map to write; the active one (or a new UVMap) if omitted.
            angle_limit: SMART_PROJECT: island angle in radians, above 0 to pi/2.
            island_margin: SMART_PROJECT: space between islands, 0-1.
            cube_size: CUBE_PROJECT: size of the projection cube, above 0.

        Returns:
            Dictionary with the UV map written, whether it was created, the
            method, the mesh's UV map names, the active and render UV maps, the
            face count and warnings.
        """
        blender_module: Any = bpy
        obj = self._get_object_or_raise(object_name)
        if obj.type != "MESH":
            raise ValidationError(
                f"Object '{object_name}' is not a MESH (type={obj.type})",
                "not_a_mesh",
                object_type=obj.type,
            )
        self._require_object_mode(obj, "unwrap it")
        context_mode = getattr(blender_module.context, "mode", "OBJECT")
        if context_mode != "OBJECT":
            raise ValidationError(
                f"Blender is in {context_mode.replace('_', ' ').title()} mode; switch to Object Mode "
                "to unwrap a mesh",
                "wrong_mode",
                mode=context_mode,
            )
        operator = _UV_UNWRAP_METHODS.get(str(method).upper())
        if operator is None:
            raise ValidationError(
                f"method must be one of {', '.join(_UV_UNWRAP_METHODS)}, got {method!r}",
                "invalid_value",
                field="method",
            )
        limits: List[Tuple[str, Any, float, Optional[float], bool]] = [
            ("angle_limit", angle_limit, 0.0, math.pi / 2, True),
            ("island_margin", island_margin, 0.0, 1.0, False),
            ("cube_size", cube_size, 0.0, None, True),
        ]
        for label, value, low, high, open_low in limits:
            number = _finite_number(value, label)
            if (number <= low if open_low else number < low) or (high is not None and number > high):
                raise ValidationError(
                    f"{label} must be {'above' if open_low else 'at least'} {low:g}"
                    + (f" and at most {high:g}" if high is not None else "")
                    + f", got {value!r}",
                    "out_of_range",
                    field=label,
                )
        mesh = obj.data
        for owner, what in ((obj, "Object"), (mesh, "Mesh")):
            library = getattr(owner, "library", None)
            if library is not None:
                raise ValidationError(
                    f"{what} '{owner.name}' is linked from library '{library.name}' and cannot "
                    "be changed; make it local first",
                    "linked_from_library",
                )
        if not obj.visible_get():
            raise ValidationError(
                f"Object '{object_name}' is hidden in the view layer and cannot enter Edit Mode; "
                "unhide it first",
                "hidden_object",
            )

        layer = mesh.uv_layers.get(uv_map_name) if uv_map_name else mesh.uv_layers.active
        previous_uv = mesh.uv_layers.active
        previous_uv_name = None if previous_uv is None else previous_uv.name
        created = layer is None
        if created:
            layer = mesh.uv_layers.new(name=uv_map_name or "UVMap")
            if layer is None:
                raise ValidationError(
                    f"Mesh '{mesh.name}' has the most UV maps Blender allows", "too_many_uv_maps"
                )
        mesh.uv_layers.active = layer
        # Leaving Edit Mode rebuilds the mesh's layers, so a UV layer struct
        # fetched before it points at freed memory afterwards (on Blender 5.0.1
        # its name read back as ""): layers are looked up again by name.
        uv_name = layer.name

        view_layer = blender_module.context.view_layer
        previous_active = view_layer.objects.active
        previous_selected = [other for other in view_layer.objects if other.select_get()]
        domains = ("vertices", "edges", "polygons")
        flags = {
            (domain, flag): self._element_flags(getattr(mesh, domain), flag)
            for domain in domains
            for flag in ("select", "hide")
        }
        try:
            for other in previous_selected:
                other.select_set(False)
            obj.select_set(True)
            view_layer.objects.active = obj
            _bpy_op(
                f"Failed to enter Edit Mode on '{object_name}'",
                blender_module.ops.object.mode_set,
                mode="EDIT",
            )
            try:
                # The operators skip hidden faces: show them for the unwrap.
                blender_module.ops.mesh.reveal(select=True)
                blender_module.ops.mesh.select_all(action="SELECT")
                options = (
                    {"angle_limit": angle_limit, "island_margin": island_margin}
                    if operator == "smart_project"
                    else {"cube_size": cube_size}
                )
                _bpy_op(
                    f"Failed to unwrap '{object_name}'",
                    getattr(blender_module.ops.uv, operator),
                    **options,
                )
            finally:
                blender_module.ops.object.mode_set(mode="OBJECT")
        except Exception:
            stale = mesh.uv_layers.get(uv_name)
            if created and stale is not None:
                mesh.uv_layers.remove(stale)
            raise
        finally:
            for (domain, flag), values in flags.items():
                getattr(mesh, domain).foreach_set(flag, values)
            restored_uv = None if previous_uv_name is None else mesh.uv_layers.get(previous_uv_name)
            if restored_uv is not None:
                mesh.uv_layers.active = restored_uv
            mesh.update()
            obj.select_set(False)
            for other in previous_selected:
                other.select_set(True)
            view_layer.objects.active = previous_active

        warnings: List[str] = []
        sharers = self._other_users(obj)
        if sharers:
            warnings.append(
                f"Mesh '{mesh.name}' is shared with {_name_list(sharers)}; their UVs changed too"
            )
        active = mesh.uv_layers.active
        render = next((uv for uv in mesh.uv_layers if getattr(uv, "active_render", False)), None)
        if render is not None and render.name != uv_name:
            warnings.append(
                f"Image textures read the render UV map '{render.name}', not '{uv_name}'"
            )
        return {
            "object_name": object_name,
            "uv_map": uv_name,
            "created_uv_map": created,
            "method": str(method).upper(),
            "uv_layer_names": [uv.name for uv in mesh.uv_layers],
            "active_uv_map": None if active is None else active.name,
            "render_uv_map": None if render is None else render.name,
            "face_count": len(mesh.polygons),
            "warnings": warnings,
        }

    @staticmethod
    def _element_flags(elements: Any, flag: str) -> List[bool]:
        """One flag (select or hide) of every vertex, edge or face, in order."""
        values = [False] * len(elements)
        elements.foreach_get(flag, values)
        return values

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

        Files the import would read besides file_path (textures, buffers, MTL
        libraries, USD layers) must be inside the sandbox too
        (_refuse_reads_outside_sandbox). While the sandbox is on, the FBX
        importer's recursive image search (use_image_search) is off: a texture
        must be at the path the FBX stores, or beside the FBX.

        Raises:
            PermissionError: If the path, or a file the import would read, is
                outside the sandbox policy.
            FileNotFoundError: If the file does not exist.
            ValueError: If the format is unsupported or scale is not positive.
        """
        file_path = self._deny_outside_sandbox(file_path)
        fmt = self._file_format(file_format)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        if not (math.isfinite(scale) and scale > 0.0):
            raise ValueError(f"scale must be a positive number, not {scale!r}")
        self._refuse_reads_outside_sandbox(file_path, fmt)

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
        if self._path_policy.enabled:
            # Search options that find files the reference check cannot vouch for.
            for option in _IMPORT_SEARCH_OPTIONS.get(fmt, ()):
                if self._operator_property(op_func, (option,)):
                    kwargs[option] = False
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
            raise NotFoundError(
                f"Collection not found: {collection_name}",
                "collection_not_found",
                collection_name=collection_name,
            )
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
        # Right after bpy.data.objects.remove, Blender 5.0.1 yields None for the
        # removed object here until the view layer syncs again.
        previous = [o for o in view_layer.objects if o is not None and o.select_get()]
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

    def _refuse_reads_outside_sandbox(self, file_path: str, fmt: str) -> None:
        """Refuse an import that would read files outside the sandbox (B1-20).

        The files a glTF's buffers and images, an OBJ's MTL libraries and their
        textures, an FBX's textures, and a USD file's layers and assets name
        are found where Blender's importer looks for them; the first of its
        candidates that exists must be allowed. A reference to a file that does
        not exist reads nothing and passes. STL and PLY name no other files.

        Raises:
            SandboxDenied: When a referenced file outside the sandbox exists,
                or when the references cannot be listed (a binary USD file
                without pxr, an FBX without Blender's FBX add-on).
        """
        reader = _IMPORT_READS.get(fmt)
        if reader is None or not self._path_policy.enabled:
            return
        from ..utils.paths import SandboxDenied

        name = os.path.basename(file_path)
        try:
            reads = reader(Path(file_path), self._path_policy.is_allowed)
        except Exception as exc:
            if fmt != "USD" and not isinstance(exc, UncheckableReferences):
                # Blender's importer reads the same file and says why it cannot.
                logger.warning("Could not list the files %s refers to: %s", name, exc)
                return
            details = self._path_policy.denial_details(file_path)
            details.update(
                referenced_by=file_path,
                hint=(
                    f"Could not check the files {name} refers to against the sandbox "
                    f"({exc}), so it is not imported"
                ),
            )
            raise SandboxDenied(file_path, details) from exc
        # Each refused file is reported as it is, without ../ steps or symlinks.
        outside = outside_references(reads, self._path_policy)
        if not outside:
            return
        details = self._path_policy.denial_details(outside[0])
        details.update(
            referenced_by=file_path,
            references_outside=outside[:20],
            hint=(
                f"{name} refers to files outside the sandbox, which the import would read: "
                f"{', '.join(outside[:5])}. Copy them under an allowed root and point "
                f"{name} at the copies, or widen security.allowed_paths"
            ),
        )
        raise SandboxDenied(outside[0], details)

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
            Dict with file_path, is_saved, is_dirty, object_count (objects in
            the current scene), file_object_count (objects in the file: every
            scene's and those no scene links) and scene_name.
        """
        blender_module: Any = bpy
        filepath = blender_module.data.filepath or ""
        scene = blender_module.context.scene
        return {
            "file_path": filepath,
            "is_saved": bool(filepath),
            "is_dirty": blender_module.data.is_dirty,
            "object_count": len(scene.objects),
            "file_object_count": len(blender_module.data.objects),
            "scene_name": scene.name,
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
        kinematic: Optional[bool] = None,
        collision_margin: Optional[float] = None,
        collision_collections: Optional[List[int]] = None,
        use_deactivation: Optional[bool] = None,
        start_deactivated: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Set up rigid body physics on a mesh object, or replace its settings.

        Every argument is checked before anything changes, and a failure
        leaves the scene as it was: no new world, rigid body or selection
        change. The settings before kinematic are written on every call;
        kinematic, collision_margin, collision_collections, use_deactivation
        and start_deactivated change only when given (a new body starts with
        Blender's defaults).

        Args:
            object_name: Name of the target mesh object.
            body_type: ACTIVE (simulated) or PASSIVE (static or animated collider).
            mass: Mass in kilograms, at least 0.001.
            friction: Surface friction coefficient, 0 or more.
            restitution: Bounciness from 0 (none) to 1 (fully elastic).
            collision_shape: BOX, SPHERE, CAPSULE, CYLINDER, CONE, CONVEX_HULL,
                MESH or COMPOUND (a compound of the children's shapes).
            linear_damping: Fraction of linear velocity lost per second, 0-1.
            angular_damping: Fraction of angular velocity lost per second, 0-1.
            kinematic: Animated: the body follows its own animation and pushes
                simulated bodies instead of being simulated.
            collision_margin: Collision margin in meters, 0-1; omitted keeps
                Blender's automatic margin for a new body.
            collision_collections: Collision collections (0-19) the body is
                in; bodies collide only when they share one. A new body is in 0.
            use_deactivation: Let the body go to sleep when it nearly stops.
            start_deactivated: Start asleep until something hits it
                (needs use_deactivation).

        Returns:
            Dict with object_name, the settings as Blender stored them,
            world_created and replaced (whether the object already had a
            rigid body).
        """
        blender_module: Any = bpy
        obj = self._get_object_or_raise(object_name)
        body_type = self._choice("body_type", body_type, self._BODY_TYPES)
        collision_shape = self._choice(
            "collision_shape", collision_shape, self._COLLISION_SHAPES
        )
        self._check_range("mass", mass, 0.001)
        self._check_range("friction", friction, 0.0)
        self._check_range("restitution", restitution, 0.0, 1.0)
        self._check_range("linear_damping", linear_damping, 0.0, 1.0)
        self._check_range("angular_damping", angular_damping, 0.0, 1.0)
        if collision_margin is not None:
            self._check_range("collision_margin", collision_margin, 0.0, 1.0)
        if collision_collections is not None:
            collision_collections = sorted(set(int(i) for i in collision_collections))
            if any(not 0 <= i < 20 for i in collision_collections):
                raise ValueError("collision_collections are numbered 0 to 19")
        if obj.type != "MESH":
            raise ValueError(
                f"Rigid bodies need a mesh object, and '{object_name}' is a "
                f"{obj.type} object; convert it to a mesh first"
            )

        scene = blender_module.context.scene
        previous = self._rigid_body_snapshot(getattr(obj, "rigid_body", None))
        _, world_created = self._ensure_rigid_body_world(scene)
        try:
            # Under an override the operator acts on this object alone and the
            # user's selection and active object stay as they were.
            with blender_module.context.temp_override(
                object=obj,
                active_object=obj,
                selected_objects=[obj],
                selected_editable_objects=[obj],
            ):
                _bpy_op(
                    f"Failed to add rigid body to '{object_name}'",
                    blender_module.ops.rigidbody.object_add,
                    type=body_type,
                )
            rb = obj.rigid_body
            rb.type = body_type
            rb.mass = mass
            rb.friction = friction
            rb.restitution = restitution
            rb.collision_shape = collision_shape
            rb.linear_damping = linear_damping
            rb.angular_damping = angular_damping
            if kinematic is not None:
                rb.kinematic = kinematic
            if collision_margin is not None:
                rb.use_margin = True
                rb.collision_margin = collision_margin
            if collision_collections is not None:
                rb.collision_collections = [i in collision_collections for i in range(20)]
            if use_deactivation is not None:
                rb.use_deactivation = use_deactivation
            if start_deactivated is not None:
                rb.use_start_deactivated = start_deactivated
        except Exception:
            self._restore_rigid_body(scene, obj, previous, world_created)
            raise

        return {
            "object_name": object_name,
            **self._rigid_body_report(obj.rigid_body),
            "world_created": world_created,
            "replaced": previous is not None,
        }

    def add_force_field(
        self,
        field_type: str,
        strength: float = 1.0,
        location: Optional[List[float]] = None,
        name: Optional[str] = None,
        direction: Optional[List[float]] = None,
        shape: Optional[str] = None,
        flow: Optional[float] = None,
        falloff_type: Optional[str] = None,
        falloff_power: Optional[float] = None,
        distance_max: Optional[float] = None,
        noise: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Add a force field to the scene, oriented in world space.

        The field's local +Z (the direction WIND blows, the axis VORTEX turns
        about, a PLANE shape's normal) points along direction, world +Z when
        omitted, whatever the editor's Align preference. Settings left out
        keep Blender's defaults.

        Args:
            field_type: FORCE, WIND, VORTEX, MAGNET, HARMONIC, CHARGE, LENNARDJ,
                TEXTURE, GUIDE, BOID, TURBULENCE, DRAG or FLUID.
            strength: Field strength.
            location: World-space XYZ location; the world origin when omitted.
            name: Optional name for the field object.
            direction: World-space direction of the field's local +Z axis.
            shape: POINT, LINE, PLANE, SURFACE or POINTS.
            flow: Air flow: how strongly the field also drags toward its speed.
            falloff_type: SPHERE, TUBE or CONE.
            falloff_power: How fast the field weakens with distance, 0-10.
            distance_max: Distance past which the field has no effect; 0 or
                omitted for no limit.
            noise: Random noise in the field, 0-10.

        Returns:
            Dict with name, field_type, strength, location, rotation_euler,
            direction, shape, flow, falloff_type, falloff_power, distance_max
            and noise.
        """
        blender_module: Any = bpy
        field_type = self._field_type(field_type)
        self._check_range("strength", strength)
        loc = self._finite_vector("location", location or (0.0, 0.0, 0.0))
        rotation = self._axis_rotation(direction, "Z", "direction")
        if shape is not None:
            shape = self._choice("shape", shape, self._FIELD_SHAPES)
        if falloff_type is not None:
            falloff_type = self._choice("falloff_type", falloff_type, self._FALLOFF_TYPES)
        if flow is not None:
            self._check_range("flow", flow)
        if falloff_power is not None:
            self._check_range("falloff_power", falloff_power, 0.0, 10.0)
        if distance_max is not None:
            self._check_range("distance_max", distance_max, 0.0)
        if noise is not None:
            self._check_range("noise", noise, 0.0, 10.0)

        # align='WORLD' and an explicit rotation: without them the operator
        # follows the user's "Align: View" preference and turns the field to
        # the 3D View. The new object is found by what the operator added:
        # context.active_object is None in the GUI when the active collection
        # is disabled in viewports, though the field was created.
        view_layer = blender_module.context.view_layer
        active_collection = view_layer.active_layer_collection
        target, warning = self._physics_collection(view_layer, "force field")
        existing = {obj.as_pointer() for obj in blender_module.data.objects}
        # bpy gives a new wrapper on every access; == compares what it wraps.
        switch = target != active_collection
        try:
            if switch:
                view_layer.active_layer_collection = target
            blender_module.ops.object.effector_add(
                type=field_type, align="WORLD", location=loc, rotation=rotation
            )
        finally:
            if switch:
                view_layer.active_layer_collection = active_collection
        created = [
            obj
            for obj in blender_module.data.objects
            if obj.as_pointer() not in existing
        ]
        if len(created) != 1:
            for obj in created:
                blender_module.data.objects.remove(obj, do_unlink=True)
            raise RuntimeError(
                f"Adding a {field_type} force field created {len(created)} objects, not one"
            )
        field_obj = created[0]
        try:
            if name:
                field_obj.name = name
            field = field_obj.field
            field.strength = strength
            if shape is not None:
                field.shape = shape
            if flow is not None:
                field.flow = flow
            if falloff_type is not None:
                field.falloff_type = falloff_type
            if falloff_power is not None:
                field.falloff_power = falloff_power
            if distance_max is not None:
                field.use_max_distance = distance_max > 0
                field.distance_max = distance_max
            if noise is not None:
                field.noise = noise
        except Exception:
            blender_module.data.objects.remove(field_obj, do_unlink=True)
            raise

        return {
            "name": field_obj.name,
            **self._field_report(field_obj),
            "warning": warning,
        }

    def get_force_field_info(self, object_name: str) -> Dict[str, Any]:
        """
        Get force field properties from a field object.

        An object whose field type is NONE has no force field and is refused:
        Blender keeps collision settings in the same structure, so a
        collision-only object used to be reported as a field of type NONE.

        Args:
            object_name: Name of the force field object.

        Returns:
            Dict with object_name, field_type, strength, shape, flow, location,
            rotation_euler, direction, falloff_type, falloff_power,
            distance_max and noise.
        """
        obj = self._get_object_or_raise(object_name)
        field = getattr(obj, "field", None)
        if field is None or field.type == "NONE":
            raise ValueError(f"Object has no force field: {object_name}")
        return {"object_name": object_name, **self._field_report(obj)}

    def add_rigid_body_constraint(
        self,
        constraint_type: str,
        object1_name: str,
        object2_name: str,
        location: Optional[List[float]] = None,
        disable_collisions: bool = True,
        axis: Optional[List[float]] = None,
        limits: Optional[Dict[str, List[float]]] = None,
        springs: Optional[Dict[str, Dict[str, float]]] = None,
        motor_angular_velocity: Optional[float] = None,
        motor_angular_max_impulse: Optional[float] = None,
        motor_linear_velocity: Optional[float] = None,
        motor_linear_max_impulse: Optional[float] = None,
        breaking_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Add a rigid body constraint between two distinct rigid bodies.

        The constraint is an empty at the pivot, whose local axes are the
        constraint's: HINGE turns about local Z, SLIDER and PISTON move along
        local X (PISTON also turns about it), MOTOR drives along and about
        local X, and GENERIC limits apply per local axis. Everything is checked
        before anything changes; a failure leaves no empty behind and the
        selection as it was.

        Args:
            constraint_type: FIXED, POINT, HINGE, SLIDER, PISTON, GENERIC,
                GENERIC_SPRING or MOTOR.
            object1_name: First constrained rigid body.
            object2_name: Second constrained rigid body.
            location: World-space pivot; midway between the two objects'
                origins when omitted.
            disable_collisions: Disable collisions between the two objects.
            axis: World-space direction of the constraint's axis: the hinge
                axis (local Z) for HINGE, the slide or motor axis (local X) for
                SLIDER, PISTON and MOTOR, local Z for the other types. Omitted,
                the empty is unrotated: a HINGE turns about world Z (a door,
                not a pendulum) and a SLIDER moves along world X.
            limits: Limits by local axis, as [lower, upper] (meters for lin_*,
                radians for ang_*, within +-2 pi): ang_z for HINGE, lin_x for
                SLIDER, lin_x and ang_x for PISTON, any of lin_x, lin_y, lin_z,
                ang_x, ang_y, ang_z for GENERIC and GENERIC_SPRING.
            springs: GENERIC_SPRING only: springs by local axis (x, y, z,
                ang_x, ang_y, ang_z), each {"stiffness": ..., "damping": ...}.
            motor_angular_velocity: MOTOR only: target angular velocity about
                local X, rad/s. Blender drives object1 relative to object2, so
                with object1 the anchor a positive velocity turns object2 the
                negative way about the axis (-1.91 rad in 23 frames at +2 rad/s
                on Blender 5.0.1); swap the objects or the sign to reverse it.
            motor_angular_max_impulse: MOTOR only: most impulse the angular
                motor applies per step (Blender's default 1).
            motor_linear_velocity: MOTOR only: target velocity along local X,
                m/s, with the same convention: with object1 the anchor, object2
                moves against the sign (-0.91 m in 23 frames at +1 m/s).
            motor_linear_max_impulse: MOTOR only: most impulse the linear motor
                applies per step (Blender's default 1).
            breaking_threshold: Impulse that breaks the constraint; omitted, it
                never breaks.

        Returns:
            Dict with constraint_name, constraint_type, object1_name,
            object2_name and the constraint as get_constraint_info reports it
            (pivot, axis, local_axes, rotation_euler, limits, springs, motor,
            breaking_threshold), plus warnings.
        """
        blender_module: Any = bpy
        constraint_type = self._choice(
            "constraint_type", constraint_type, self._CONSTRAINT_TYPES
        )
        obj1 = self._get_object_or_raise(object1_name)
        obj2 = self._get_object_or_raise(object2_name)
        if obj1 == obj2:
            raise ValueError(
                f"A rigid body constraint joins two different objects; got "
                f"'{object1_name}' twice"
            )
        scene = blender_module.context.scene
        world = getattr(scene, "rigidbody_world", None)
        for obj in (obj1, obj2):
            if (
                getattr(obj, "rigid_body", None) is None
                or world is None
                or not self._in_rigid_body_world(world, obj)
            ):
                raise ValueError(
                    f"'{obj.name}' is not a rigid body; set it up with "
                    "setup_blender_rigid_body first (a PASSIVE body for an anchor)"
                )
        settings = self._constraint_settings(
            constraint_type,
            limits,
            springs,
            {
                "motor_ang_target_velocity": motor_angular_velocity,
                "motor_ang_max_impulse": motor_angular_max_impulse,
                "motor_lin_target_velocity": motor_linear_velocity,
                "motor_lin_max_impulse": motor_linear_max_impulse,
            },
            breaking_threshold,
        )
        if location is None:
            first, second = self._world_location(obj1), self._world_location(obj2)
            pivot = tuple((a + b) / 2.0 for a, b in zip(first, second))
        else:
            pivot = self._finite_vector("location", location)
        primary = self._CONSTRAINT_AXES.get(constraint_type, "Z")
        rotation = self._axis_rotation(axis, primary, "axis")

        # A new empty, not empty_add: that operator follows the user's
        # "Align: View" preference and selects the empty.
        view_layer = blender_module.context.view_layer
        layer, collection_warning = self._physics_collection(
            view_layer, "constraint empty"
        )
        collection = layer.collection
        empty = blender_module.data.objects.new(
            f"RBC_{object1_name}_{object2_name}", None
        )
        empty.location = pivot
        empty.rotation_mode = "XYZ"
        empty.rotation_euler = rotation
        previous_active = view_layer.objects.active
        try:
            # Inside the try: a failed link (a linked collection) must not leave
            # the empty behind with no users.
            collection.objects.link(empty)
            # constraint_add acts on the view layer's active object, whatever
            # the context override says.
            view_layer.objects.active = empty
            _bpy_op(
                f"Failed to add a {constraint_type} constraint",
                blender_module.ops.rigidbody.constraint_add,
                type=constraint_type,
            )
            rbc = empty.rigid_body_constraint
            if rbc is None:
                raise RuntimeError(
                    f"Blender added no rigid body constraint to '{empty.name}'"
                )
            rbc.object1 = obj1
            rbc.object2 = obj2
            rbc.disable_collisions = disable_collisions
            for setting, value in settings.items():
                setattr(rbc, setting, value)
        except Exception:
            if getattr(empty, "rigid_body_constraint", None) is not None:
                view_layer.objects.active = empty
                blender_module.ops.rigidbody.constraint_remove()
            blender_module.data.objects.remove(empty, do_unlink=True)
            raise
        finally:
            view_layer.objects.active = previous_active

        warnings: List[str] = [collection_warning] if collection_warning else []
        if constraint_type == "MOTOR" and (
            motor_angular_velocity is None and motor_linear_velocity is None
        ):
            warnings.append(
                "A MOTOR constraint drives nothing until motor_angular_velocity or "
                "motor_linear_velocity is set"
            )
        return {
            "constraint_name": empty.name,
            **self._constraint_report(empty),
            "warnings": warnings,
        }

    def get_constraint_info(self, object_name: str) -> Dict[str, Any]:
        """
        Get rigid body constraint details from a constraint empty.

        Args:
            object_name: Name of the constraint empty object.

        Returns:
            Dict with object_name, constraint_type, object1_name, object2_name,
            enabled, disable_collisions, pivot, axis, local_axes,
            rotation_euler, limits, springs, motor and breaking_threshold.
        """
        obj = self._get_object_or_raise(object_name)
        rbc = getattr(obj, "rigid_body_constraint", None)
        if rbc is None:
            raise ValueError(f"Object has no rigid body constraint: {object_name}")
        return {"object_name": object_name, **self._constraint_report(obj)}

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
        allow_long_bake: bool = False,
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
            allow_long_bake: Bake even above 20 000 frames or 400 000
                body-frames (frames x bodies), which can outlast the client's
                timeout while Blender keeps baking.

        Returns:
            Dict with scope, frame_start, frame_end and frame_count of the
            rigid-body cache (null without one), rigid_body_cache, caches (each
            cache's owner, kind, range and bake state) and warnings.
        """
        blender_module: Any = bpy
        scope = self._bake_scope(scope)
        self._check_cache_range(int(frame_start), int(frame_end))
        scene = blender_module.context.scene
        if not allow_long_bake:
            self._check_bake_work(scene, scope, frame_end - frame_start + 1)
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

    #: Most a bake does without allow_long_bake: frames, and frames x bodies.
    #: On Blender 5.0.1 a frame took about 0.3 ms plus 30-40 microseconds per
    #: body in contact: 1000 bodies x 1000 frames took 29.4 s, against the
    #: attached-mode client's 30 s timeout, and 401 cubes x 900 frames 14.8 s
    #: headless (1000 frames 8.7 s in the GUI). The budget takes up to about
    #: 16 s, inside that timeout.
    _BAKE_FRAME_BUDGET = 20_000
    _BAKE_WORK_BUDGET = 400_000

    def _check_bake_work(self, scene: Any, scope: str, frames: int) -> None:
        """Refuse a bake likely to outlast the client's timeout on Blender's main thread."""
        world = getattr(scene, "rigidbody_world", None)
        bodies = self._rigid_body_count(world) if world is not None else 0
        if scope == "all":
            bodies += sum(
                1 for entry, _ in self._point_caches(scene) if entry["kind"] != "RIGID_BODY"
            )
        work = frames * max(bodies, 1)
        if frames <= self._BAKE_FRAME_BUDGET and work <= self._BAKE_WORK_BUDGET:
            return
        raise ValueError(
            f"Baking {frames} frames of {bodies} simulated objects is {work} body-frames, "
            f"over this tool's limit of {self._BAKE_FRAME_BUDGET} frames and "
            f"{self._BAKE_WORK_BUDGET} body-frames (up to about 16 s on Blender 5.0.1). A bake "
            "runs on Blender's main thread and cannot be stopped once it starts: past "
            "the client timeout (BLENDER__CONNECTION_TIMEOUT, 30 s by default) the call "
            "reports a timeout while Blender keeps baking, and the next calls wait "
            "behind it. Bake a shorter range, or raise the timeout and pass "
            "allow_long_bake=true"
        )

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
    #: body-frames (frames x rigid bodies), as _BAKE_WORK_BUDGET: up to 16 s of
    #: simulation on Blender 5.0.1 when no frame is cached yet, inside the
    #: attached-mode client's 30 s timeout. Cached frames are only read.
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

    # -- Rigid-body, force-field and constraint setup helpers ----------------

    #: Blender 4.2-5.0's RigidBodyObject.type and collision_shape items.
    _BODY_TYPES = ("ACTIVE", "PASSIVE")
    _COLLISION_SHAPES = (
        "BOX",
        "SPHERE",
        "CAPSULE",
        "CYLINDER",
        "CONE",
        "CONVEX_HULL",
        "MESH",
        "COMPOUND",
    )

    #: The types object.effector_add takes (Blender 4.2-5.0).
    _FIELD_TYPES = (
        "FORCE",
        "WIND",
        "VORTEX",
        "MAGNET",
        "HARMONIC",
        "CHARGE",
        "LENNARDJ",
        "TEXTURE",
        "GUIDE",
        "BOID",
        "TURBULENCE",
        "DRAG",
        "FLUID",
    )
    #: Other spellings of an effector_add type: the one simul used to document,
    #: and the one FieldSettings.type reports for a FLUID field.
    _FIELD_TYPE_ALIASES = {"MAGNETIC": "MAGNET", "FLUID_FLOW": "FLUID"}
    _FIELD_SHAPES = ("POINT", "LINE", "PLANE", "SURFACE", "POINTS")
    _FALLOFF_TYPES = ("SPHERE", "TUBE", "CONE")

    #: RigidBodyConstraint.type items.
    _CONSTRAINT_TYPES = (
        "FIXED",
        "POINT",
        "HINGE",
        "SLIDER",
        "PISTON",
        "GENERIC",
        "GENERIC_SPRING",
        "MOTOR",
    )
    #: The constraint empty's local axis each type works along; Z otherwise.
    #: Verified on Blender 5.0.1: a HINGE turns about local Z, a SLIDER moves
    #: along local X, a MOTOR's angular motor turns about local X.
    _CONSTRAINT_AXES = {"HINGE": "Z", "SLIDER": "X", "PISTON": "X", "MOTOR": "X"}
    _LIMIT_AXES = ("lin_x", "lin_y", "lin_z", "ang_x", "ang_y", "ang_z")
    #: The limits each constraint type applies (Blender's constraint panel).
    _CONSTRAINT_LIMITS: Dict[str, Tuple[str, ...]] = {
        "HINGE": ("ang_z",),
        "SLIDER": ("lin_x",),
        "PISTON": ("lin_x", "ang_x"),
        "GENERIC": _LIMIT_AXES,
        "GENERIC_SPRING": _LIMIT_AXES,
    }
    _SPRING_AXES = ("x", "y", "z", "ang_x", "ang_y", "ang_z")
    #: Angular limits Blender accepts, in radians.
    _ANGLE_LIMIT = 2 * math.pi

    @staticmethod
    def _choice(label: str, value: Any, choices: Sequence[str]) -> str:
        """``value`` upper-cased when it is one of ``choices``; ValueError otherwise."""
        text = str(value).upper()
        if text not in choices:
            raise ValueError(
                f"Invalid {label} '{value}'. Must be one of: {', '.join(choices)}"
            )
        return text

    @classmethod
    def _field_type(cls, value: Any) -> str:
        text = str(value).upper()
        return cls._choice(
            "field_type", cls._FIELD_TYPE_ALIASES.get(text, text), cls._FIELD_TYPES
        )

    @staticmethod
    def _check_range(
        label: str,
        value: Any,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
    ) -> float:
        """``value`` as a finite float within ``minimum``-``maximum``; ValueError otherwise.

        Blender's RNA stores some of these settings unclamped: on 5.0.1 a mass
        of -5 was kept, and the body fell through the ground.
        """
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{label} must be a finite number, got {value}")
        if minimum is not None and number < minimum:
            raise ValueError(f"{label} must be at least {minimum}, got {value}")
        if maximum is not None and number > maximum:
            raise ValueError(f"{label} must be at most {maximum}, got {value}")
        return number

    @staticmethod
    def _finite_vector(label: str, values: Any) -> Tuple[float, float, float]:
        vector = tuple(float(v) for v in values)
        if len(vector) != 3 or not all(math.isfinite(v) for v in vector):
            raise ValueError(f"{label} must be three finite numbers")
        return vector[0], vector[1], vector[2]

    @classmethod
    def _axis_rotation(
        cls, axis: Optional[Sequence[float]], primary: str, label: str
    ) -> Tuple[float, float, float]:
        """The XYZ Euler rotation that turns local ``primary`` (X or Z) onto world ``axis``.

        The other local axes stay as close to the world's as they can: the
        local X of a Z-primary frame lies along world X (world Y when the axis
        is near X), and the local Z of an X-primary frame along world Z (world
        X when the axis is near Z). No axis means no rotation.
        """
        if axis is None:
            return 0.0, 0.0, 0.0
        vector = cls._finite_vector(label, axis)
        length = math.sqrt(sum(v * v for v in vector))
        if length < 1e-9:
            raise ValueError(f"{label} must not be the zero vector")
        unit = tuple(v / length for v in vector)

        def orthogonal(reference: Tuple[float, ...], to: Tuple[float, ...]) -> Tuple[float, ...]:
            dot = sum(r * t for r, t in zip(reference, to))
            rest = tuple(r - dot * t for r, t in zip(reference, to))
            size = math.sqrt(sum(v * v for v in rest))
            return tuple(v / size for v in rest)

        def cross(a: Tuple[float, ...], b: Tuple[float, ...]) -> Tuple[float, ...]:
            return (
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            )

        if primary == "Z":
            z = unit
            x = orthogonal((1.0, 0.0, 0.0) if abs(z[0]) < 0.9 else (0.0, 1.0, 0.0), z)
        else:
            x = unit
            z = orthogonal((0.0, 0.0, 1.0) if abs(x[2]) < 0.9 else (1.0, 0.0, 0.0), x)
        y = cross(z, x)
        # Blender's XYZ Euler is R = Rz Ry Rx; the matrix's columns are x, y, z.
        sin_y = -x[2]
        if abs(sin_y) < 1.0 - 1e-9:
            return (
                math.atan2(y[2], z[2]),
                math.asin(sin_y),
                math.atan2(x[1], x[0]),
            )
        if sin_y > 0:
            return math.atan2(y[0], z[0]), math.pi / 2, 0.0
        return math.atan2(-y[0], -z[0]), -math.pi / 2, 0.0

    @staticmethod
    def _matrix_axis(matrix: Any, column: int) -> List[float]:
        """A world matrix's local axis ``column`` (0 X, 1 Y, 2 Z) as a unit vector."""
        values = [float(matrix[row][column]) for row in range(3)]
        size = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / size for v in values]

    def _physics_collection(self, view_layer: Any, what: str) -> Tuple[Any, Optional[str]]:
        """The layer collection a new force field or constraint empty goes in.

        Where create_object puts objects (_creation_collection): the active
        collection, or the scene's own when the active one, or one holding it,
        is excluded, hidden or disabled in viewports. Also the scene's own when
        the active one is linked from a library, which Blender refuses to link
        into. Either simulates: the rigid body world reads its constraints
        collection.

        Returns:
            The layer collection, and a warning naming why it is not the
            active one, or None.
        """
        layer, warning = self._creation_collection(view_layer)
        collection = layer.collection
        if getattr(collection, "library", None) is not None or not getattr(
            collection, "is_editable", True
        ):
            root = view_layer.layer_collection
            return root, (
                f"The active collection '{collection.name}' is linked from a library, so "
                f"the {what} was created in the scene collection '{root.collection.name}' "
                "instead"
            )
        if warning is not None:
            warning = warning.replace("so the object was", f"so the {what} was")
        return layer, warning

    @staticmethod
    def _rigid_body_snapshot(rb: Any) -> Optional[Dict[str, Any]]:
        """The settings setup_rigid_body writes, to restore them after a failure."""
        if rb is None:
            return None
        names = (
            "type",
            "mass",
            "friction",
            "restitution",
            "collision_shape",
            "linear_damping",
            "angular_damping",
            "kinematic",
            "use_margin",
            "collision_margin",
            "use_deactivation",
            "use_start_deactivated",
        )
        snapshot = {name: getattr(rb, name) for name in names if hasattr(rb, name)}
        if hasattr(rb, "collision_collections"):
            snapshot["collision_collections"] = list(rb.collision_collections)
        return snapshot

    def _restore_rigid_body(
        self,
        scene: Any,
        obj: Any,
        previous: Optional[Dict[str, Any]],
        world_created: bool,
    ) -> None:
        """Undo a failed setup_rigid_body: its rigid body or settings, and its world.

        A failure here is logged, not raised, so the setup's own error reaches
        the caller.
        """
        blender_module: Any = bpy
        try:
            rb = getattr(obj, "rigid_body", None)
            if previous is None and rb is not None:
                with blender_module.context.temp_override(
                    object=obj,
                    active_object=obj,
                    selected_objects=[obj],
                    selected_editable_objects=[obj],
                ):
                    blender_module.ops.rigidbody.object_remove()
            elif previous is not None and rb is not None:
                for name, value in previous.items():
                    setattr(rb, name, value)
            if world_created and scene.rigidbody_world is not None:
                with blender_module.context.temp_override(scene=scene):
                    blender_module.ops.rigidbody.world_remove()
        except Exception:
            self.logger.exception(
                "Could not undo the failed rigid body setup of '%s'", obj.name
            )

    @staticmethod
    def _rigid_body_report(rb: Any) -> Dict[str, Any]:
        collections = getattr(rb, "collision_collections", None)
        return {
            "body_type": rb.type,
            "mass": float(rb.mass),
            "friction": float(rb.friction),
            "restitution": float(rb.restitution),
            "collision_shape": rb.collision_shape,
            "linear_damping": float(rb.linear_damping),
            "angular_damping": float(rb.angular_damping),
            "kinematic": bool(rb.kinematic),
            "collision_margin": (
                float(rb.collision_margin) if rb.use_margin else None
            ),
            "collision_collections": (
                [i for i, member in enumerate(collections) if member]
                if collections is not None
                else None
            ),
            "use_deactivation": bool(rb.use_deactivation),
            "start_deactivated": bool(rb.use_start_deactivated),
        }

    def _field_report(self, obj: Any) -> Dict[str, Any]:
        """A force field's settings and world pose; direction is its local +Z."""
        matrix = self._evaluated_world_matrix(obj)
        field = obj.field
        return {
            "field_type": field.type,
            "strength": float(field.strength),
            "location": [float(v) for v in matrix.translation],
            "rotation_euler": [float(v) for v in matrix.to_euler("XYZ")],
            "direction": self._matrix_axis(matrix, 2),
            "shape": field.shape,
            "flow": float(field.flow),
            "falloff_type": field.falloff_type,
            "falloff_power": float(field.falloff_power),
            "distance_max": (
                float(field.distance_max) if field.use_max_distance else None
            ),
            "noise": float(field.noise),
        }

    def _constraint_settings(
        self,
        constraint_type: str,
        limits: Optional[Dict[str, List[float]]],
        springs: Optional[Dict[str, Dict[str, float]]],
        motor: Dict[str, Optional[float]],
        breaking_threshold: Optional[float],
    ) -> Dict[str, Any]:
        """The RigidBodyConstraint settings to write, after checking each against the type."""
        settings: Dict[str, Any] = {}
        allowed = self._CONSTRAINT_LIMITS.get(constraint_type, ())
        for key, bounds in (limits or {}).items():
            if key not in allowed:
                raise ValueError(
                    f"A {constraint_type} constraint has no {key} limit; it limits "
                    f"{', '.join(allowed) or 'nothing'}"
                )
            values = [self._check_range(f"limits.{key}", v) for v in bounds]
            if len(values) != 2 or values[0] > values[1]:
                raise ValueError(f"limits.{key} must be [lower, upper] with lower <= upper")
            if key.startswith("ang_") and max(abs(v) for v in values) > self._ANGLE_LIMIT:
                raise ValueError(f"limits.{key} must be within +-2 pi radians")
            settings[f"use_limit_{key}"] = True
            settings[f"limit_{key}_lower"], settings[f"limit_{key}_upper"] = values
        if springs:
            if constraint_type != "GENERIC_SPRING":
                raise ValueError("springs apply to GENERIC_SPRING constraints only")
            for key, spring in springs.items():
                if key not in self._SPRING_AXES:
                    raise ValueError(
                        f"Invalid spring axis '{key}'. Must be one of: "
                        f"{', '.join(self._SPRING_AXES)}"
                    )
                settings[f"use_spring_{key}"] = True
                settings[f"spring_stiffness_{key}"] = self._check_range(
                    f"springs.{key}.stiffness", spring.get("stiffness", 10.0), 0.0
                )
                settings[f"spring_damping_{key}"] = self._check_range(
                    f"springs.{key}.damping", spring.get("damping", 0.5), 0.0
                )
        given = {name: value for name, value in motor.items() if value is not None}
        if given and constraint_type != "MOTOR":
            raise ValueError("motor settings apply to MOTOR constraints only")
        for name, value in given.items():
            minimum = 0.0 if name.endswith("max_impulse") else None
            settings[name] = self._check_range(name, value, minimum)
        if "motor_ang_target_velocity" in given:
            settings["use_motor_ang"] = True
        if "motor_lin_target_velocity" in given:
            settings["use_motor_lin"] = True
        if breaking_threshold is not None:
            settings["use_breaking"] = True
            settings["breaking_threshold"] = self._check_range(
                "breaking_threshold", breaking_threshold, 0.0
            )
        return settings

    def _constraint_report(self, empty: Any) -> Dict[str, Any]:
        """A constraint's settings and world pose, with its local axes in world space."""
        rbc = empty.rigid_body_constraint
        matrix = self._evaluated_world_matrix(empty)
        axes = {name: self._matrix_axis(matrix, column) for column, name in enumerate("xyz")}
        kind = rbc.type
        limits = {
            key: [
                float(getattr(rbc, f"limit_{key}_lower")),
                float(getattr(rbc, f"limit_{key}_upper")),
            ]
            for key in self._CONSTRAINT_LIMITS.get(kind, ())
            if getattr(rbc, f"use_limit_{key}", False)
        }
        springs = (
            {
                key: {
                    "stiffness": float(getattr(rbc, f"spring_stiffness_{key}")),
                    "damping": float(getattr(rbc, f"spring_damping_{key}")),
                }
                for key in self._SPRING_AXES
                if getattr(rbc, f"use_spring_{key}", False)
            }
            if kind == "GENERIC_SPRING"
            else {}
        )
        motor = None
        if kind == "MOTOR":
            motor = {
                "angular_velocity": (
                    float(rbc.motor_ang_target_velocity) if rbc.use_motor_ang else None
                ),
                "angular_max_impulse": float(rbc.motor_ang_max_impulse),
                "linear_velocity": (
                    float(rbc.motor_lin_target_velocity) if rbc.use_motor_lin else None
                ),
                "linear_max_impulse": float(rbc.motor_lin_max_impulse),
            }
        return {
            "constraint_type": kind,
            "object1_name": rbc.object1.name if rbc.object1 else None,
            "object2_name": rbc.object2.name if rbc.object2 else None,
            "enabled": bool(rbc.enabled),
            "disable_collisions": bool(rbc.disable_collisions),
            "pivot": [float(v) for v in matrix.translation],
            "axis": axes[self._CONSTRAINT_AXES.get(kind, "Z").lower()],
            "local_axes": axes,
            "rotation_euler": [float(v) for v in matrix.to_euler("XYZ")],
            "limits": limits,
            "springs": springs,
            "motor": motor,
            "breaking_threshold": (
                float(rbc.breaking_threshold) if rbc.use_breaking else None
            ),
        }

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
            raise NotFoundError(
                f"Collection '{collection_name}' not found",
                "collection_not_found",
                collection_name=collection_name,
            )
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
    # Additional semantic labels are stored as simready_label_<key>.
    _SIMREADY_LABEL_PREFIX: str = "simready_label_"

    # A name that is a USD identifier becomes the prim name as it is; SimReady
    # accepts camelCase as well as lowercase_underscore. Any other name is
    # rewritten by the exporter, which validation reports as a warning.
    _SIMREADY_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    _SIMREADY_QCODE_RE = re.compile(r"^Q[1-9][0-9]*$")
    # A label key names a USD semantics taxonomy, so it must be an identifier.
    _SIMREADY_LABEL_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    # The taxonomies the semantic fields are written under, by field; an
    # additional label may not reuse them, nor qcode, the Q-code taxonomy an
    # earlier export wrote.
    _SIMREADY_RESERVED_LABELS = {
        "class": "semantic_class",
        "hierarchy": "semantic_hierarchy",
        "wikidata_qcode": "semantic_qcode",
        "qcode": "semantic_qcode",
    }

    # The metadata fields by group, with the rule a value must meet. The MCP
    # request models (mcp/schemas/simready.py) state the same rules for the
    # tool's input schema; this module ships inside the Blender add-on and
    # cannot import them, so tests/blender/test_simready_metadata.py keeps the
    # two in step.
    _SIMREADY_FIELDS: Dict[str, Dict[str, Tuple[Any, ...]]] = {
        "semantic": {
            "semantic_class": ("text",),
            "semantic_hierarchy": ("text",),
            "semantic_qcode": ("qcode",),
        },
        "physics": {
            "mass_kg": ("number", 0.0, None),
            "collider_type": ("token", SIMREADY_COLLIDER_TYPES),
            "static_friction": ("number", 0.0, None),
            "dynamic_friction": ("number", 0.0, None),
            "restitution": ("number", 0.0, 1.0),
            "density": ("number", 0.0, None),
            "is_rigid_body": ("flag",),
        },
        "material": {
            "substrate_type": ("text",),
            "material_naming": ("text",),
            "shader_type": ("text",),
            "texel_density": ("positive",),
        },
    }
    _SIMREADY_CLEAR_ALL = "all"
    _SIMREADY_LABELS = "additional_labels"

    @classmethod
    def _simready_field_names(cls) -> List[str]:
        return [name for fields in cls._SIMREADY_FIELDS.values() for name in fields]

    @staticmethod
    def _unknown_name(kind: str, name: Any, choices: Sequence[str]) -> str:
        """``unknown <kind> 'x' (did you mean 'y'?); valid: ...``."""
        close = difflib.get_close_matches(str(name), list(choices), n=1, cutoff=0.5)
        hint = f" (did you mean '{close[0]}'?)" if close else ""
        return f"unknown {kind} '{name}'{hint}; valid: {', '.join(choices)}"

    @classmethod
    def _simready_value_problem(cls, rule: Tuple[Any, ...], value: Any) -> Optional[str]:
        """Why ``value`` breaks ``rule``, or None when it meets it."""
        kind = rule[0]
        if kind == "flag":
            if isinstance(value, bool) or (isinstance(value, int) and value in (0, 1)):
                return None
            return "must be true or false"
        if kind in ("number", "positive"):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                return "must be a number"
            if kind == "positive":
                return None if value > 0 else "must be greater than 0"
            low, high = rule[1], rule[2]
            if value < low:
                return f"must be at least {low:g}"
            if high is not None and value > high:
                return f"must be at most {high:g}"
            return None
        if not isinstance(value, str) or not value.strip():
            return "must be a non-empty string"
        if kind == "token" and value not in rule[1]:
            return f"must be one of {', '.join(rule[1])}"
        if kind == "qcode" and not cls._SIMREADY_QCODE_RE.match(value):
            return "must be a Wikidata Q-code such as Q1420"
        return None

    def _simready_writes(self, metadata: Any) -> List[Tuple[str, Any]]:
        """The custom properties ``metadata`` sets, or ValueError naming every problem."""
        if not isinstance(metadata, Mapping):
            raise ValidationError(
                "metadata must be an object with semantic, physics or material",
                "InvalidMetadata",
            )
        problems: List[str] = []
        writes: List[Tuple[str, Any]] = []
        for group, fields in metadata.items():
            spec = self._SIMREADY_FIELDS.get(group)
            if spec is None:
                problems.append(self._unknown_name("group", group, list(self._SIMREADY_FIELDS)))
                continue
            if fields is None:
                continue
            if not isinstance(fields, Mapping):
                problems.append(f"{group} must be an object")
                continue
            choices = list(spec) + ([self._SIMREADY_LABELS] if group == "semantic" else [])
            for name, value in fields.items():
                if group == "semantic" and name == self._SIMREADY_LABELS:
                    if value is None:
                        continue
                    if not isinstance(value, Mapping):
                        problems.append("semantic.additional_labels must be an object")
                        continue
                    for key, label in value.items():
                        if not self._SIMREADY_LABEL_KEY_RE.match(str(key)):
                            problems.append(
                                f"label key '{key}' must be a USD identifier "
                                "(letters, digits, underscores, not starting with a digit)"
                            )
                        elif key in self._SIMREADY_RESERVED_LABELS:
                            # A label named after a field's taxonomy sets that field,
                            # so it cannot overwrite the field's label in the USD.
                            target = self._SIMREADY_RESERVED_LABELS[key]
                            given = fields.get(target)
                            problem = self._simready_value_problem(spec[target], label)
                            if given is not None and given != label:
                                problems.append(
                                    f"label '{key}'={label!r} sets semantic.{target}, which is "
                                    f"also given as {given!r}; give one of them"
                                )
                            elif problem is not None:
                                problems.append(f"label '{key}'={label!r} {problem}")
                            elif given is None:
                                writes.append((f"{self._SIMREADY_PREFIX}{target}", label))
                        elif not isinstance(label, str) or not label.strip():
                            problems.append(f"label '{key}' must be a non-empty string")
                        else:
                            writes.append((f"{self._SIMREADY_LABEL_PREFIX}{key}", label))
                    continue
                rule = spec.get(name)
                if rule is None:
                    problems.append(self._unknown_name(f"{group} field", name, choices))
                    continue
                if value is None:
                    continue  # null keeps the stored value; clear removes it
                problem = self._simready_value_problem(rule, value)
                if problem is not None:
                    problems.append(f"{group}.{name}={value!r} {problem}")
                    continue
                if rule[0] == "flag":
                    value = 1 if value else 0
                writes.append((f"{self._SIMREADY_PREFIX}{name}", value))
        if problems:
            raise ValidationError(
                "Invalid SimReady metadata: " + "; ".join(problems),
                "InvalidMetadata",
                problems=problems,
            )
        return writes

    def _simready_clear_keys(self, obj: Any, clear: Sequence[str]) -> List[str]:
        """The stored custom properties ``clear`` names, or ValueError for an unknown name."""
        prefix = self._SIMREADY_PREFIX
        stored = [key for key in obj.keys() if key.startswith(prefix)]
        choices = (
            self._simready_field_names()
            + list(self._SIMREADY_FIELDS)
            + [self._SIMREADY_LABELS, self._SIMREADY_CLEAR_ALL]
        )
        keys: List[str] = []
        problems: List[str] = []
        for name in clear:
            if name == self._SIMREADY_CLEAR_ALL:
                keys.extend(stored)
            elif name == self._SIMREADY_LABELS:
                keys.extend(k for k in stored if k.startswith(self._SIMREADY_LABEL_PREFIX))
            elif name in self._SIMREADY_FIELDS:
                keys.extend(f"{prefix}{field}" for field in self._SIMREADY_FIELDS[name])
                if name == "semantic":
                    keys.extend(k for k in stored if k.startswith(self._SIMREADY_LABEL_PREFIX))
            elif name in self._simready_field_names():
                keys.append(f"{prefix}{name}")
            else:
                problems.append(self._unknown_name("clear name", name, choices))
        if problems:
            raise ValidationError(
                "Invalid clear: " + "; ".join(problems), "InvalidClear", problems=problems
            )
        return list(dict.fromkeys(keys))

    def _simready_editable(self, objects: Sequence[Any], tool: str) -> None:
        """Refuse outside Object Mode and for linked data, before anything changes.

        In Edit Mode Blender records the bridge's undo step on the mesh undo
        stack, which does not hold object changes: the first Ctrl+Z did
        nothing and the second also left Edit Mode. A linked object's parent
        and custom properties are reset when the file is reopened.
        """
        blender_module: Any = bpy
        mode = getattr(blender_module.context, "mode", "OBJECT")
        if mode != "OBJECT":
            raise ValidationError(
                f"{tool} needs Object Mode, as Edit Mode's undo cannot step back over it; "
                f"the window is in {mode}",
                "ObjectModeRequired",
                mode=mode,
            )
        linked = sorted(
            obj.name
            for obj in objects
            if getattr(obj, "library", None) is not None
            or getattr(obj, "is_editable", True) is False
        )
        if linked:
            raise ValidationError(
                f"{', '.join(linked)} is linked from a library: changes to it are lost when "
                "the file is reopened; make it local (Object > Relations > Make Local) first",
                "LinkedData",
                objects=linked,
            )

    def apply_simready_metadata(
        self,
        object_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        clear: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Store SimReady metadata on a Blender object, for export_simready_usd.

        Each value is a ``simready_<field>`` custom property (an additional
        label ``simready_label_<key>``); export_simready_usd writes them as
        UsdPhysics and UsdSemantics schemas. A field left out, or null, keeps
        its stored value; ``clear`` removes stored values before the new ones
        are written. Unknown groups and fields and invalid values are refused,
        with the valid names, before anything changes.

        Args:
            object_name: Target Blender object.
            metadata: The ``semantic``, ``physics`` and ``material`` groups to store.
            clear: Stored values to remove: field names such as mass_kg, a group
                (semantic, physics, material), additional_labels, or all.

        Returns:
            Dict with object_name, applied_properties and cleared_properties.

        Raises:
            NotFoundError: For an unknown object (``details.code`` object_not_found).
            ValidationError: For an unknown group, field or clear name, an
                invalid value, an additional label named after a semantic
                field's taxonomy (class, hierarchy, wikidata_qcode, qcode) that
                disagrees with that field (otherwise it sets the field), a call
                that neither stores nor clears anything, Edit Mode, or a linked
                object; ``details.code`` names which.
        """
        obj = self._get_object_or_raise(object_name)
        self._simready_editable([obj], "apply_simready_metadata")
        writes = self._simready_writes(metadata or {})
        removals = self._simready_clear_keys(obj, clear or [])
        if not writes and not clear:
            raise ValidationError(
                "Nothing to apply: pass metadata values, or clear", "NothingToApply"
            )
        present = set(obj.keys())
        cleared = [key for key in removals if key in present]
        for key in cleared:
            del obj[key]
        for key, value in writes:
            obj[key] = value
        return {
            "object_name": object_name,
            "applied_properties": [key for key, _ in writes],
            "cleared_properties": cleared,
        }

    @staticmethod
    def _plain_value(value: Any) -> Any:
        """A stored custom property as JSON: IDProperty arrays and groups as lists and dicts."""
        if hasattr(value, "to_list"):
            return value.to_list()
        if hasattr(value, "to_dict"):
            return value.to_dict()
        if value is None or isinstance(value, (str, int, float, bool, list, dict)):
            return value
        return str(value)

    def _read_simready_properties(self, obj: Any) -> Dict[str, Dict[str, Any]]:
        """The object's ``simready_*`` custom properties as stored, grouped like the metadata.

        Additional labels (``simready_label_<key>``) are gathered under
        ``semantic["additional_labels"]``; other ``simready_`` keys that are
        no metadata field go under ``unknown``, by property name.
        """
        groups: Dict[str, Dict[str, Any]] = {name: {} for name in self._SIMREADY_FIELDS}
        unknown: Dict[str, Any] = {}
        labels: Dict[str, Any] = {}
        owner = {
            field: group for group, fields in self._SIMREADY_FIELDS.items() for field in fields
        }
        for key in obj.keys():
            if not key.startswith(self._SIMREADY_PREFIX):
                continue
            value = self._plain_value(obj[key])
            if key.startswith(self._SIMREADY_LABEL_PREFIX):
                labels[key[len(self._SIMREADY_LABEL_PREFIX) :]] = value
                continue
            field = key[len(self._SIMREADY_PREFIX) :]
            if field in owner:
                groups[owner[field]][field] = value
            else:
                unknown[key] = value
        if labels:
            groups["semantic"][self._SIMREADY_LABELS] = labels
        groups["unknown"] = unknown
        return groups

    def _simready_checked(
        self, stored: Mapping[str, Mapping[str, Any]]
    ) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
        """Split stored metadata into the values that meet the rules and the ones that do not."""
        valid: Dict[str, Dict[str, Any]] = {}
        invalid: List[Dict[str, Any]] = []
        for group, spec in self._SIMREADY_FIELDS.items():
            for name, value in (stored.get(group) or {}).items():
                if group == "semantic" and name == self._SIMREADY_LABELS:
                    good: Dict[str, str] = {}
                    for key, label in value.items():
                        if not self._SIMREADY_LABEL_KEY_RE.match(str(key)):
                            problem = "label key must be a USD identifier"
                        elif key in self._SIMREADY_RESERVED_LABELS:
                            problem = (
                                "label key is the taxonomy "
                                f"{self._SIMREADY_RESERVED_LABELS[key]} is written under"
                            )
                        elif not isinstance(label, str) or not label.strip():
                            problem = "must be a non-empty string"
                        else:
                            good[key] = label
                            continue
                        invalid.append(
                            {
                                "property": f"{self._SIMREADY_LABEL_PREFIX}{key}",
                                "value": label,
                                "problem": problem,
                            }
                        )
                    if good:
                        valid.setdefault(group, {})[name] = good
                    continue
                broken = self._simready_value_problem(spec[name], value)
                if broken is None:
                    if spec[name][0] == "flag":
                        value = bool(value)
                    valid.setdefault(group, {})[name] = value
                else:
                    invalid.append(
                        {"property": f"{self._SIMREADY_PREFIX}{name}", "value": value, "problem": broken}
                    )
        for key, value in (stored.get("unknown") or {}).items():
            invalid.append(
                {
                    "property": key,
                    "value": value,
                    "problem": self._unknown_name(
                        "SimReady field",
                        key[len(self._SIMREADY_PREFIX) :],
                        self._simready_field_names(),
                    ),
                }
            )
        return valid, invalid

    def get_simready_metadata(
        self,
        object_name: str,
    ) -> Dict[str, Any]:
        """
        Read the SimReady metadata stored on a Blender object.

        Stored values that break the metadata rules (set by a script, an
        import or an older Simul) are not returned as metadata; they are
        listed in ``invalid_values`` with the problem, so one bad value does
        not hide the rest.

        Args:
            object_name: Blender object to inspect.

        Returns:
            Dict with object_name, metadata (null when nothing valid is
            stored), has_simready_data and invalid_values.

        Raises:
            NotFoundError: For an unknown object (``details.code`` object_not_found).
        """
        obj = self._get_object_or_raise(object_name)
        stored = self._read_simready_properties(obj)
        valid, invalid = self._simready_checked(stored)
        return {
            "object_name": object_name,
            "metadata": valid or None,
            "has_simready_data": bool(valid or invalid),
            "invalid_values": invalid,
        }

    # Checks validate_simready_compliance runs, by its check_<name> parameter.
    _SIMREADY_CHECKS = (
        "naming",
        "scale",
        "transforms",
        "materials",
        "hierarchy",
        "metadata",
        "physics",
        "geometry",
    )
    _SIMREADY_SEVERITIES = ("all", "error", "warning")
    # validate_simready_compliance returns at most this many issues per page,
    # and keeps a page under this many bytes of JSON, inside the server's
    # result budget.
    _SIMREADY_PAGE_MAX = 500
    _SIMREADY_PAGE_BYTES = 30_000
    # A face this small (in square Blender units) has collapsed to a line or point.
    _SIMREADY_DEGENERATE_AREA = 1e-12
    # Axes whose cosine exceeds this are skewed: no location, rotation and scale
    # express the transform.
    _SIMREADY_SHEAR = 1e-5
    _SIMREADY_TRIANGLE_MESH = frozenset({"none", "mesh"})
    #: Rigid body constraint type -> the UsdPhysics joint the export writes for it.
    #: MOTOR has none: Blender's motor is a free joint whose drive is capped by an
    #: impulse per simulation step, which no UsdPhysics drive expresses.
    _SIMREADY_JOINT_TYPES = {
        "FIXED": "fixed",
        "POINT": "spherical",
        "HINGE": "revolute",
        "SLIDER": "prismatic",
        "PISTON": "d6",
        "GENERIC": "d6",
        "GENERIC_SPRING": "d6",
    }
    #: A Blender limit axis -> the UsdPhysics D6 degree of freedom (LimitAPI instance).
    _SIMREADY_D6_AXES = {
        "lin_x": "transX",
        "lin_y": "transY",
        "lin_z": "transZ",
        "ang_x": "rotX",
        "ang_y": "rotY",
        "ang_z": "rotZ",
    }
    #: The degrees of freedom a PISTON frees: it slides along and turns about local X.
    _SIMREADY_PISTON_AXES = frozenset({"lin_x", "ang_x"})
    #: The limits Blender measures as object1 relative to object2, the reverse of
    #: UsdPhysics (body1 relative to body0, body0 being object1), so their range is
    #: negated. Measured on Blender 5.0.1, gravity against a one-sided limit: a
    #: HINGE's ang_z and a GENERIC's ang_x/y/z; linear limits and a PISTON's
    #: ang_x read object2 relative to object1.
    _SIMREADY_REVERSED_LIMITS = frozenset({("HINGE", "ang_z")}) | frozenset(
        (kind, axis)
        for kind in ("GENERIC", "GENERIC_SPRING")
        for axis in ("ang_x", "ang_y", "ang_z")
    )
    #: A D6 degree of freedom with its low limit above its high one is locked.
    _SIMREADY_LOCKED = [1.0, -1.0]

    def validate_simready_compliance(
        self,
        object_names: Optional[List[str]] = None,
        check_naming: bool = True,
        check_scale: bool = True,
        check_transforms: bool = True,
        check_materials: bool = True,
        check_hierarchy: bool = True,
        check_metadata: bool = True,
        check_physics: bool = True,
        check_geometry: bool = True,
        include_children: bool = True,
        severity: str = "all",
        offset: int = 0,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """
        Validate objects against SimReady conventions; any error makes them non-compliant.

        It checks what export_simready_usd would export: the listed objects
        with their descendants, or the scene's objects that are visible in
        renders, without lights and cameras. The rules follow the SimReady
        Robotics-Prop profile and Isaac Sim's asset validation (the codes in
        brackets):

        - errors: the asset root (the prim that becomes the default prim) not
          at the origin, unrotated and unscaled [VG.025]; a rigid body whose
          world transform is skewed [RB.009]; no material; no semantic label
          on a mesh or an exported object above it [SL.001]; a stored value
          that breaks the metadata rules; a rigid body without mass or density
          [RB.007], without a collider, or nested in another rigid body with
          no joint the export writes between them [RB.012]; a rigid body
          constraint between exported objects the export cannot write as a
          joint (MOTOR, or both objects parts of one rigid body); degenerate
          faces [VG.014];
        - warnings: names that are not USD identifiers (camelCase is fine),
          mesh data names likewise; sizes over 100 m or under 1 mm in the
          scene's units and a Unit Scale other than 1 [UN.007]; scaled or
          mirrored parts, and a part skewed relative to its parent; a
          material slot faces use left empty; no collider on a mesh; a
          triangle-mesh collider on a dynamic body; no rigid body at all
          [RB.001]; a constraint to an object the export leaves out or between
          two static objects (not exported), and what a joint cannot carry
          (springs, a breaking threshold, several rotation axes limited to
          ranges); no UV map; n-gons.

        Transforms are read from the world matrices, as the exporter writes
        them (also for objects Blender does not evaluate, such as ones in an
        excluded collection), and meshes in Edit Mode are read with their edits.

        Args:
            object_names: Objects to check, each with its descendants unless
                include_children is false. Null checks the scene's objects that
                are visible in renders. An empty list is refused.
            check_naming: Validate object, material and mesh-data names.
            check_scale: Validate real-world size in the scene's units.
            check_transforms: Validate the asset root's and the parts' transforms.
            check_materials: Validate material slots.
            check_hierarchy: Validate hierarchy structure.
            check_metadata: Validate semantic labels, colliders and stored values.
            check_physics: Validate rigid bodies (mass, colliders, nesting) and
                the joints their constraints become.
            check_geometry: Validate UVs, n-gons and degenerate faces.
            include_children: Also check the descendants of the listed objects.
            severity: Return all issues, or only error or warning issues.
            offset: Index of the first issue returned.
            limit: Most issues returned (1 to 500); next_offset gives the
                next page.

        Returns:
            Dict with compliant (no errors at all), object_count, issue_count,
            error_count, warning_count, issues (the page), offset,
            next_offset (null on the last page), skipped_objects (left out as
            the export leaves them out) and meters_per_unit.

        Raises:
            NotFoundError: For an unknown object (``details.code`` object_not_found).
            ValidationError: For an empty object list, a listed object outside
                the view layer, or a bad severity, offset or limit.
        """
        if object_names is not None and not object_names:
            raise ValidationError(
                "object_names is empty; list the objects to check, or pass null for the scene",
                "EmptyObjectList",
            )
        if severity not in self._SIMREADY_SEVERITIES:
            raise ValidationError(
                f"severity must be one of {', '.join(self._SIMREADY_SEVERITIES)}",
                "InvalidPaging",
            )
        if offset < 0 or not 1 <= limit <= self._SIMREADY_PAGE_MAX:
            raise ValidationError(
                f"offset must be >= 0 and limit 1 to {self._SIMREADY_PAGE_MAX}", "InvalidPaging"
            )
        targets, _, skipped = self._simready_export_targets(
            object_names, include_children, include_lights_and_cameras=False
        )
        checks = {
            "naming": check_naming,
            "scale": check_scale,
            "transforms": check_transforms,
            "materials": check_materials,
            "hierarchy": check_hierarchy,
            "metadata": check_metadata,
            "physics": check_physics,
            "geometry": check_geometry,
        }
        unit_scale = self._scene_unit_scale()
        issues = self._simready_issues(targets, checks, unit_scale)
        errors = sum(1 for issue in issues if issue["severity"] == "error")
        selected = [i for i in issues if severity == "all" or i["severity"] == severity]
        page: List[Dict[str, Any]] = []
        size = 0
        for issue in selected[offset : offset + limit]:
            size += len(json.dumps(issue)) + 2
            if page and size > self._SIMREADY_PAGE_BYTES:
                break
            page.append(issue)
        end = offset + len(page)
        return {
            "compliant": errors == 0,
            "object_count": len(targets),
            "issue_count": len(issues),
            "error_count": errors,
            "warning_count": len(issues) - errors,
            "issues": page,
            "offset": offset,
            "next_offset": end if end < len(selected) else None,
            "skipped_objects": skipped[: self._SIMREADY_RESULT_ITEMS],
            "meters_per_unit": unit_scale,
        }

    def _simready_world_resolver(self) -> "_WorldResolver":
        """World matrices as the exporter writes them, for the objects of one call."""
        blender_module: Any = bpy
        graph_get = getattr(blender_module.context, "evaluated_depsgraph_get", None)
        return _WorldResolver(graph_get() if graph_get is not None else None)

    @staticmethod
    def _simready_flush_edit_mode(targets: Sequence[Any]) -> None:
        """Write Edit Mode's mesh edits back to the mesh, so the checks see them."""
        for obj in targets:
            if getattr(obj, "mode", "OBJECT") == "EDIT" and hasattr(obj, "update_from_editmode"):
                obj.update_from_editmode()

    def _simready_default_prim(self, targets: Sequence[Any]) -> Optional[Any]:
        """The object that becomes the default prim, or None when a new root wraps several.

        Blender writes the unexported ancestors of exported objects too, so it
        is the top-most ancestor when every target has the same one.
        """
        tops: Dict[str, Any] = {}
        for obj in targets:
            top = obj
            for ancestor in self._ancestors(obj):
                top = ancestor
            tops.setdefault(top.name, top)
        return next(iter(tops.values())) if len(tops) == 1 else None

    def _simready_issues(
        self, targets: Sequence[Any], checks: Mapping[str, bool], unit_scale: float
    ) -> List[Dict[str, Any]]:
        """Every issue of ``targets``, in object order, then the asset-wide ones."""
        self._simready_flush_edit_mode(targets)
        issues: List[Dict[str, Any]] = []
        exported = {obj.name: obj for obj in targets}
        values = {
            obj.name: self._simready_checked(self._read_simready_properties(obj))
            for obj in targets
        }
        worlds = self._simready_world_resolver()
        root = self._simready_default_prim(targets)
        named: set = set()

        def add(obj: Any, check: str, severity: str, message: str, suggestion: Optional[str]) -> None:
            issues.append(
                {
                    "object_name": getattr(obj, "name", str(obj)),
                    "check": check,
                    "severity": severity,
                    "message": message,
                    "suggestion": suggestion,
                }
            )

        def chain(obj: Any) -> Iterator[Any]:
            """The object and its exported ancestors: what its metadata inherits from."""
            yield obj
            for ancestor in self._ancestors(obj):
                if ancestor.name in exported:
                    yield ancestor

        def field(obj: Any, group: str, name: str) -> Any:
            return values[obj.name][0].get(group, {}).get(name)

        def nearest(obj: Any, group: str, name: str) -> Optional[Any]:
            return next((o for o in chain(obj) if field(o, group, name) is not None), None)

        def rigid(obj: Any) -> bool:
            return field(obj, "physics", "is_rigid_body") is True

        def body_of(obj: Any) -> Optional[Any]:
            return next((o for o in chain(obj) if rigid(o)), None)

        # The rigid bodies that some mesh of theirs collides for.
        collided = {
            body.name
            for body in (
                body_of(o)
                for o in targets
                if o.type == "MESH" and nearest(o, "physics", "collider_type") is not None
            )
            if body is not None
        }
        joints, joint_issues = (
            self._simready_joint_plan(targets, worlds) if checks["physics"] else ([], [])
        )
        jointed = {frozenset((j["body0"], j["body1"])) for j in joints}
        if checks["transforms"] and root is not None:
            self._simready_root_transform_issues(root, worlds, add)
        for obj in targets:
            is_mesh = obj.type == "MESH"
            if checks["naming"]:
                self._simready_naming_issues(obj, add, named)
            if checks["scale"]:
                issues.extend(self._simready_size_issues(obj, unit_scale, worlds.world(obj)))
            if checks["transforms"] and obj is not root:
                self._simready_part_transform_issues(obj, rigid(obj), worlds, add)
            stats = self._simready_mesh_stats(obj) if is_mesh else None
            if checks["materials"] and is_mesh:
                self._simready_material_issues(obj, stats, add)
            if checks["hierarchy"]:
                if is_mesh and obj.parent is None and not obj.children:
                    add(obj, "hierarchy", "warning",
                        "Mesh has no parent empty — SimReady assets should be under a root XForm",
                        "Use setup_simready_hierarchy to create a root")
                elif root is not None and obj.name == root.name and obj.type != "EMPTY":
                    add(obj, "hierarchy", "warning",
                        f"The asset root is a {obj.type.lower()}, not an empty (Xform)",
                        "Use setup_simready_hierarchy to put it under a root empty")
            if checks["metadata"]:
                for entry in values[obj.name][1]:
                    add(obj, "metadata", "error",
                        f"Stored {entry['property']}={entry['value']!r} {entry['problem']}",
                        "Set a valid value with apply_simready_metadata, or clear it")
                if is_mesh and not any(self._simready_has_label(values[o.name][0]) for o in chain(obj)):
                    add(obj, "metadata", "error",
                        "No semantic label on the mesh or an exported object above it",
                        "Set semantic.semantic_class (or another label) on the asset root or the "
                        "mesh with apply_simready_metadata")
                if is_mesh and nearest(obj, "physics", "collider_type") is None:
                    add(obj, "metadata", "warning",
                        "No collider_type on the mesh or an exported object above it; it will "
                        "not collide in a simulator",
                        "Set physics.collider_type (e.g. convexHull) with apply_simready_metadata")
            if checks["physics"]:
                self._simready_body_issues(
                    obj, add, field, nearest, rigid, body_of, chain, collided, jointed
                )
            if checks["geometry"] and is_mesh:
                self._simready_geometry_issues(obj, stats, add)
        for name, severity, message, suggestion in joint_issues:
            add(name, "physics", severity, message, suggestion)
        if checks["physics"] and targets and not any(rigid(o) for o in targets):
            add(root or targets[0], "physics", "warning",
                "No rigid body: the asset is static in a simulator",
                "Set physics.is_rigid_body on the asset root if it should move")
        if checks["scale"] and targets and abs(unit_scale - 1.0) > 1e-9:
            add(root or targets[0], "scale", "warning",
                f"The scene's Unit Scale is {unit_scale:g}, so the asset's metersPerUnit is "
                f"{unit_scale:g}; SimReady assets use metres (1.0)",
                "Scale the asset to metres and set Scene > Units > Unit Scale to 1, or keep it: "
                "Isaac Sim converts the units when it references the asset")
        return issues

    @staticmethod
    def _simready_has_label(valid: Mapping[str, Mapping[str, Any]]) -> bool:
        """Whether the object carries a semantic label of any taxonomy."""
        semantic = valid.get("semantic") or {}
        return any(semantic.get(key) for key in semantic)

    def _simready_body_issues(
        self,
        obj: Any,
        add: Callable[..., None],
        field: Callable[[Any, str, str], Any],
        nearest: Callable[[Any, str, str], Optional[Any]],
        rigid: Callable[[Any], bool],
        body_of: Callable[[Any], Optional[Any]],
        chain: Callable[[Any], Iterator[Any]],
        collided: set,
        jointed: set,
    ) -> None:
        """A rigid body's mass, colliders and nesting; a triangle-mesh collider on a body.

        ``jointed`` holds the body pairs (frozensets of names) an exported joint
        joins: a rigid body nested in another is valid with one [RB.012].
        """
        if obj.type == "MESH":
            source = nearest(obj, "physics", "collider_type")
            body = body_of(obj)
            if (
                source is not None
                and body is not None
                and field(source, "physics", "collider_type") in self._SIMREADY_TRIANGLE_MESH
            ):
                add(obj, "physics", "warning",
                    f"A triangle-mesh collider on the dynamic body {body.name}; PhysX does not "
                    "simulate it",
                    "Use collider_type convexHull or convexDecomposition")
        if not rigid(obj):
            return
        if field(obj, "physics", "mass_kg") is None and field(obj, "physics", "density") is None:
            add(obj, "physics", "error",
                "Rigid body without mass_kg or density",
                "Set physics.mass_kg (or density) with apply_simready_metadata")
        parent = next((a for a in chain(obj) if a is not obj and rigid(a)), None)
        if parent is not None and frozenset((obj.name, parent.name)) not in jointed:
            add(obj, "physics", "error",
                f"Rigid body nested in the rigid body {parent.name} with no joint between them "
                "that the export writes, so a simulator cannot tell how they move together",
                f"Join {obj.name} to {parent.name} with add_blender_rigid_body_constraint (the "
                "export writes it as a UsdPhysics joint), or keep one rigid body")
        if obj.name not in collided:
            add(obj, "physics", "error",
                "Rigid body without a collider: no mesh of it has a collider_type",
                "Set physics.collider_type on the body or its meshes")

    @staticmethod
    def _simready_constraint_empty(obj: Any) -> bool:
        """An empty that only carries a rigid body constraint: a joint, not a part."""
        return (
            getattr(obj, "type", None) == "EMPTY"
            and getattr(obj, "rigid_body_constraint", None) is not None
            and not getattr(obj, "children", None)
        )

    def _simready_joint_plan(
        self, targets: Sequence[Any], worlds: "_WorldResolver"
    ) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str, str, str]]]:
        """The UsdPhysics joints for the rigid body constraints between exported objects.

        A constraint joins the rigid bodies of its two objects: each object's
        own, else the nearest exported rigid body above it, else the object
        itself as a static body. Its frame is the constraint object's world
        matrix without scale, as Blender's rigid body world reads it; body0 is
        object1 and body1 object2.

        Returns:
            The joints export_simready_usd writes (simready_usd's plan entries),
            and issues ``(constraint name, severity, message, suggestion)`` for
            the constraints it cannot write, or writes without some setting.
        """
        blender_module: Any = bpy
        exported = {obj.name: obj for obj in targets}
        rigid_names = {
            name
            for name, obj in exported.items()
            if self._simready_checked(self._read_simready_properties(obj))[0]
            .get("physics", {})
            .get("is_rigid_body")
            is True
        }

        def body_of(obj: Any) -> Optional[Any]:
            if obj.name in rigid_names:
                return obj
            return next((a for a in self._ancestors(obj) if a.name in rigid_names), None)

        joints: List[Dict[str, Any]] = []
        issues: List[Tuple[str, str, str, str]] = []
        carriers = sorted(
            (
                o
                for o in blender_module.context.scene.objects
                if o is not None and getattr(o, "rigid_body_constraint", None) is not None
            ),
            key=lambda o: o.name,
        )
        for carrier in carriers:
            rbc = carrier.rigid_body_constraint
            first, second = rbc.object1, rbc.object2
            if first is None or second is None or first.name == second.name:
                continue  # Blender simulates nothing for it either
            kind, name = str(rbc.type), carrier.name
            inside = [o for o in (first, second) if o.name in exported]
            if not inside:
                continue
            if len(inside) == 1:
                outside = second if inside[0] is first else first
                issues.append((
                    name, "warning",
                    f"The {kind} constraint {name} joins {inside[0].name} to {outside.name}, "
                    "which the export leaves out, so the joint is not exported",
                    "Export both objects, or remove the constraint",
                ))
                continue
            joint_type = self._SIMREADY_JOINT_TYPES.get(kind)
            if joint_type is None:
                issues.append((
                    name, "error",
                    f"The {kind} constraint {name} between {first.name} and {second.name} has "
                    "no UsdPhysics joint: Blender's motor is a free joint whose drive is capped "
                    "by an impulse per simulation step, which no UsdPhysics drive expresses, so "
                    "the export would drop it",
                    "Join them with a HINGE or SLIDER constraint and set up its drive in the "
                    "simulator, or remove the constraint",
                ))
                continue
            body0, body1 = body_of(first), body_of(second)
            if body0 is None and body1 is None:
                issues.append((
                    name, "warning",
                    f"The {kind} constraint {name} joins {first.name} and {second.name}, neither "
                    "of which is a rigid body or part of one, so it does nothing in a simulator "
                    "and is not exported",
                    "Set physics.is_rigid_body on the part that moves",
                ))
                continue
            if body0 is not None and body1 is not None and body0.name == body1.name:
                issues.append((
                    name, "error",
                    f"The {kind} constraint {name} joins {first.name} and {second.name}, which "
                    f"are parts of the same rigid body {body0.name}; a joint needs two bodies, "
                    "so the export would drop it",
                    f"Make {first.name} and {second.name} rigid bodies of their own",
                ))
                continue
            body0, body1 = body0 or first, body1 or second
            joints.append({
                "name": name,
                "constraint_type": kind,
                "type": joint_type,
                "body0": body0.name,
                "body1": body1.name,
                "frame": worlds.world(carrier),
                "body0_world": worlds.world(body0),
                "body1_world": worlds.world(body1),
                "enabled": bool(getattr(rbc, "enabled", True)),
                "collision_enabled": not bool(getattr(rbc, "disable_collisions", True)),
                **self._simready_joint_limits(rbc, kind, joint_type),
            })
            issues.extend(
                (name, "warning", message, suggestion)
                for message, suggestion in self._simready_joint_losses(rbc, kind, name)
            )
        return joints, issues

    def _simready_joint_limits(self, rbc: Any, kind: str, joint_type: str) -> Dict[str, Any]:
        """A joint's axis and limits as UsdPhysics reads them: degrees, body1 relative to body0."""

        def bounds(axis: str) -> Optional[List[float]]:
            if not getattr(rbc, f"use_limit_{axis}", False):
                return None  # Blender leaves an unlimited axis free
            lower = float(getattr(rbc, f"limit_{axis}_lower"))
            upper = float(getattr(rbc, f"limit_{axis}_upper"))
            if axis.startswith("ang_"):
                lower, upper = math.degrees(lower), math.degrees(upper)
            if (kind, axis) in self._SIMREADY_REVERSED_LIMITS:
                lower, upper = -upper, -lower
            return [lower + 0.0, upper + 0.0]

        if joint_type == "revolute":
            return {"axis": "Z", "limit": bounds("ang_z")}
        if joint_type == "prismatic":
            return {"axis": "X", "limit": bounds("lin_x")}
        if joint_type != "d6":
            return {}
        limits: Dict[str, List[float]] = {}
        for axis, dof in self._SIMREADY_D6_AXES.items():
            if kind == "PISTON" and axis not in self._SIMREADY_PISTON_AXES:
                limits[dof] = list(self._SIMREADY_LOCKED)
                continue
            value = bounds(axis)
            if value is None:
                continue
            if abs(value[0]) < 1e-9 and abs(value[1]) < 1e-9:
                value = list(self._SIMREADY_LOCKED)
            limits[dof] = value
        return {"limits": limits}

    def _simready_joint_losses(self, rbc: Any, kind: str, name: str) -> List[Tuple[str, str]]:
        """What of a constraint its UsdPhysics joint leaves out: (message, suggestion) pairs."""
        losses: List[Tuple[str, str]] = []
        if kind == "GENERIC_SPRING" and any(
            getattr(rbc, f"use_spring_{axis}", False) for axis in self._SPRING_AXES
        ):
            losses.append((
                f"The springs of the GENERIC_SPRING constraint {name} are not exported: "
                "Blender's spring damping has no exact UsdPhysics drive equivalent, so the "
                "joint is exported with its limits only",
                "Add drives with stiffness and damping to the joint in the simulator",
            ))
        if getattr(rbc, "use_breaking", False):
            losses.append((
                f"The breaking threshold of the {kind} constraint {name} is not exported: "
                "Blender's threshold is an impulse per simulation step, which UsdPhysics' "
                "break force and torque do not express, so the joint does not break",
                "Set physics:breakForce and physics:breakTorque on the joint in the simulator",
            ))
        if kind in ("GENERIC", "GENERIC_SPRING"):
            ranged = [
                axis
                for axis in ("ang_x", "ang_y", "ang_z")
                if getattr(rbc, f"use_limit_{axis}", False)
                and abs(
                    float(getattr(rbc, f"limit_{axis}_upper"))
                    - float(getattr(rbc, f"limit_{axis}_lower"))
                )
                > 1e-9
            ]
            if len(ranged) > 1:
                losses.append((
                    f"The {kind} constraint {name} limits {', '.join(ranged)} to ranges: "
                    "Blender reads them as Euler angles, PhysX as twist and swing, so away "
                    "from the rest pose the limits differ",
                    "Limit one rotation axis per joint, or check the limits in the simulator",
                ))
        return losses

    def _simready_root_transform_issues(
        self, root: Any, worlds: "_WorldResolver", add: Callable[..., None]
    ) -> None:
        """The default prim's transform must be the identity [VG.025]."""
        world = worlds.world(root)
        if _m4_is_identity(world):
            return
        translation, lengths, shear, mirrored = _m4_shape(world)
        rotated = any(
            abs(world[r][c] / (lengths[c] or 1.0) - _IDENTITY4[r][c]) > 1e-5
            for r in range(3)
            for c in range(3)
        )
        what = []
        if any(abs(v) > 1e-6 for v in translation):
            what.append(f"at {tuple(round(v, 6) for v in translation)}")
        if rotated:
            what.append("rotated")
        if any(abs(n - 1.0) > 1e-6 for n in lengths) or mirrored or shear > self._SIMREADY_SHEAR:
            what.append(f"scaled {tuple(round(n, 6) for n in lengths)}")
        add(root, "transforms", "error",
            f"The asset root {root.name} becomes the default prim but is "
            f"{', '.join(what) or 'transformed'}; SimReady requires an identity root transform",
            "Clear the root's transform (Alt+G, Alt+R, Alt+S) and apply it to its children, "
            "or put the asset under setup_simready_hierarchy's root")

    def _simready_part_transform_issues(
        self, obj: Any, is_rigid: bool, worlds: "_WorldResolver", add: Callable[..., None]
    ) -> None:
        """A part's transform: a rigid body's in the world [RB.009], any part's relative to its parent."""
        world = worlds.world(obj)
        fix = (
            "Apply the parent's non-uniform scale (Ctrl+A > Scale on the parent) before "
            "rotating its children"
        )
        world_shear = _m4_shape(world)[2]
        if is_rigid and world_shear > self._SIMREADY_SHEAR:
            # A simulator moves the body rigidly: a skew its parents' scale and its
            # rotation build up is not a shape it can keep.
            add(obj, "transforms", "error",
                f"The rigid body's world transform is skewed (two of its axes meet "
                f"{math.degrees(math.acos(min(1.0, world_shear))):.1f} degrees apart, not 90); "
                "a rigid body's transform must not skew", fix)
        if obj.parent is None:
            local = world
        else:
            inverse = _m4_inverse(worlds.world(obj.parent))
            if inverse is None:
                add(obj, "transforms", "warning",
                    f"Its parent {obj.parent.name} has zero scale, so its transform cannot be "
                    "written", "Give the parent a scale")
                return
            local = _m4_mul(inverse, world)
        _, lengths, shear, mirrored = _m4_shape(local)
        if shear > self._SIMREADY_SHEAR and not (is_rigid and world_shear > self._SIMREADY_SHEAR):
            add(obj, "transforms", "warning",
                "The transform relative to its parent is skewed (its axes are not "
                "perpendicular)", fix)
        if any(abs(n - 1.0) > 1e-6 for n in lengths) or mirrored:
            add(obj, "transforms", "warning",
                f"The object is scaled {tuple(round(n, 6) for n in lengths)}"
                + (" and mirrored" if mirrored else ""),
                "Apply scale: Ctrl+A > Scale (and clear delta transforms)")

    def _simready_naming_issues(self, obj: Any, add: Callable[..., None], named: set) -> None:
        """Names that are not USD identifiers, which the exporter rewrites (warnings)."""

        def suggestion(name: str) -> str:
            return "Rename to: " + usd_identifier(name.strip(), "part")

        name = obj.name
        if not self._SIMREADY_NAME_RE.match(name):
            add(obj, "naming", "warning",
                f"Name '{name}' is not a USD identifier; the export writes the prim as "
                f"'{usd_identifier(name)}'", suggestion(name))
        sibling = (getattr(obj.parent, "name", None), usd_identifier(name))
        if sibling in named:
            add(obj, "naming", "warning",
                f"Another object under the same parent also becomes the prim name "
                f"'{sibling[1]}'; the export renames one of them", suggestion(name))
        named.add(sibling)
        if obj.type != "MESH":
            return
        for material in self._material_slots(obj):
            material_name = getattr(material, "name", material)
            if material is None or ("material", material_name) in named:
                continue
            named.add(("material", material_name))
            if not self._SIMREADY_NAME_RE.match(str(material_name)):
                add(obj, "naming", "warning",
                    f"Material name '{material_name}' is not a USD identifier; the export "
                    f"writes the material as '{usd_identifier(str(material_name))}'",
                    suggestion(str(material_name)))
        data_name = getattr(getattr(obj, "data", None), "name", None)
        if data_name and ("mesh", data_name) not in named:
            named.add(("mesh", data_name))
            if not self._SIMREADY_NAME_RE.match(data_name):
                add(obj, "naming", "warning",
                    f"Mesh data name '{data_name}' is not a USD identifier; it becomes the "
                    f"mesh prim's name as '{usd_identifier(data_name)}'", suggestion(data_name))

    @staticmethod
    def _material_slots(obj: Any) -> List[Any]:
        """The material in each of the object's slots, None for an empty slot."""
        slots = getattr(obj, "material_slots", None)
        if slots is not None:
            return [getattr(slot, "material", None) for slot in slots]
        data = getattr(obj, "data", None)
        return list(getattr(data, "materials", None) or [])

    def _simready_material_issues(
        self, obj: Any, stats: Optional[Dict[str, Any]], add: Callable[..., None]
    ) -> None:
        """No material at all is an error; an empty slot is a warning when faces use it."""
        slots = self._material_slots(obj)
        if not slots:
            add(obj, "materials", "error", "No material assigned",
                "Assign at least one material per mesh prim")
            return
        used = set((stats or {}).get("material_indices") or ())
        empty = [str(i) for i, material in enumerate(slots) if material is None and i in used]
        if empty:
            add(obj, "materials", "warning",
                f"Faces use the empty material slot(s) {', '.join(empty)}",
                "Assign a material to the slot, or move its faces to another slot")

    def _simready_mesh_stats(self, obj: Any) -> Optional[Dict[str, Any]]:
        """Face counts, corners, areas, material indices and UV maps of the mesh the export writes."""
        mesh, release = self._simready_export_mesh(obj)
        try:
            polygons = getattr(mesh, "polygons", None)
            if polygons is None:
                return None
            count = len(polygons)
            if hasattr(polygons, "foreach_get"):
                corners = array("i", [0]) * count
                areas = array("f", [0.0]) * count
                indices = array("i", [0]) * count
                polygons.foreach_get("loop_total", corners)
                polygons.foreach_get("area", areas)
                polygons.foreach_get("material_index", indices)
            else:
                corners = array("i", [len(p.vertices) for p in polygons])
                areas = array("f", [float(p.area) for p in polygons])
                indices = array("i", [int(getattr(p, "material_index", 0)) for p in polygons])
            return {
                "count": count,
                "corners": corners,
                "areas": areas,
                "material_indices": set(indices),
                "uv_maps": len(getattr(mesh, "uv_layers", ())),
            }
        finally:
            release()

    def _simready_geometry_issues(
        self, obj: Any, stats: Optional[Dict[str, Any]], add: Callable[..., None]
    ) -> None:
        """UV maps (warning), n-gons (warning) and degenerate faces (error) [VG.014]."""
        if not stats:
            return
        count = stats["count"]
        ngons = sum(1 for c in stats["corners"] if c > 4)
        degenerate = sum(1 for a in stats["areas"] if a <= self._SIMREADY_DEGENERATE_AREA)
        if count and not stats["uv_maps"]:
            add(obj, "geometry", "warning", "Mesh has no UV map; textures cannot be mapped",
                "Unwrap the mesh (U > Smart UV Project)")
        if ngons:
            add(obj, "geometry", "warning",
                f"{ngons} of {count} faces are n-gons (more than 4 corners)",
                "Triangulate or quad-fill them (Ctrl+T, or a Triangulate modifier)")
        if degenerate:
            add(obj, "geometry", "error",
                f"{degenerate} of {count} faces are degenerate (zero area)",
                "Remove them: Mesh > Clean Up > Degenerate Dissolve, or Merge by Distance")

    def _simready_export_mesh(self, obj: Any) -> Tuple[Any, Callable[[], None]]:
        """The mesh the USD export writes (with modifiers evaluated), and how to release it."""
        blender_module: Any = bpy
        modifiers = getattr(obj, "modifiers", None)
        graph_get = getattr(blender_module.context, "evaluated_depsgraph_get", None)
        if modifiers and len(modifiers) and graph_get is not None:
            evaluated = obj.evaluated_get(graph_get())
            mesh = evaluated.to_mesh()
            return mesh, evaluated.to_mesh_clear
        return getattr(obj, "data", None), lambda: None

    # A part's largest real-world dimension (Blender units times the scene's
    # Unit Scale) outside these bounds suggests the scene's units are not the
    # ones it was modelled in: a 0.5 m part modelled in millimetres in a metre
    # scene measures 500 m. Large assets exist (a warehouse), so it is a warning.
    _SIMREADY_MAX_SIZE_M = 100.0
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
    def _world_dimensions(obj: Any, world: Optional[_Matrix4] = None) -> List[float]:
        """The object's world-space bounding-box size in Blender units.

        Read from the bounding box corners and the world matrix (``world``, or
        the object's matrix_world), so a parent's scale counts; ``dimensions``
        (object scale only) when either is missing.
        """
        corners = getattr(obj, "bound_box", None)
        matrix = world if world is not None else getattr(obj, "matrix_world", None)
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

    def _simready_size_issues(
        self, obj: Any, unit_scale: float, world: Optional[_Matrix4] = None
    ) -> List[Dict[str, Any]]:
        """A warning when a mesh's real-world size is implausible in the declared units."""
        if getattr(obj, "type", "MESH") != "MESH":
            return []
        dimensions = self._world_dimensions(obj, world)
        largest_units = max(dimensions) if dimensions else 0.0
        largest = largest_units * unit_scale
        if largest <= 0:
            return []
        declared = (
            f"{largest:.4g} m ({largest_units:.4g} Blender units at a Unit Scale of "
            f"{unit_scale:g} m)"
        )
        if largest > self._SIMREADY_MAX_SIZE_M:
            message = (
                f"Largest dimension {declared} is large for a SimReady asset "
                f"(over {self._SIMREADY_MAX_SIZE_M:g} m); verify the units"
            )
        elif largest < self._SIMREADY_MIN_SIZE_M:
            message = (
                f"Largest dimension {declared} is small for a SimReady asset "
                f"(under {self._SIMREADY_MIN_SIZE_M * 1000:g} mm); verify the units"
            )
        else:
            return []
        return [
            {
                "object_name": obj.name,
                "check": "scale",
                "severity": "warning",
                "message": message,
                "suggestion": (
                    "If it was modelled in other units, set Scene > Units > Unit Scale "
                    "to the modelling unit (0.001 for millimetres, 0.01 for centimetres) "
                    "or scale the mesh to its real size"
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
        skip_checks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Export a SimReady asset as USD that a simulator reads as physics and semantics.

        Blender's USD exporter writes the geometry and render materials; the
        export then authors, from each object's simready_ metadata, UsdPhysics
        RigidBodyAPI, MassAPI (mass, and density in stage units), CollisionAPI
        with a MeshCollisionAPI approximation on the meshes, a physics material
        bound with the physics purpose, and UsdSemantics LabelsAPI labels (Isaac
        Sim's SemanticsAPI convention where Blender's USD has no UsdSemantics).
        Each rigid body constraint between two exported objects becomes a
        UsdPhysics joint under <asset root>/Joints: FIXED a FixedJoint, POINT a
        SphericalJoint, HINGE a RevoluteJoint (about the constraint's local Z),
        SLIDER a PrismaticJoint (along local X), PISTON, GENERIC and
        GENERIC_SPRING a D6 joint with a LimitAPI per locked or limited axis;
        body0 is object1's rigid body, body1 object2's, the limits are in
        degrees and read as UsdPhysics reads them (body1 relative to body0),
        and the constraint empty is not written as a prim. A MOTOR has no
        UsdPhysics joint; springs and breaking thresholds are not exported
        (each is reported). The asset root is the default prim and holds the
        materials (Looks, PhysicsMaterials) and the joints; metersPerUnit is
        the scene's Unit Scale. The file is written only when everything
        succeeded. As export_file does, the export
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
            validate_before_export: First run every validate_simready_compliance
                check on the exported objects, the asset root included, after
                writing Edit Mode's edits back to the mesh.
            include_children: Export the descendants of the listed objects.
            include_lights_and_cameras: Also export lights, cameras and the
                World's dome light, which are left out of an asset by default.
            force: Export although validation found errors.
            skip_checks: Validation checks to leave out (naming, scale,
                transforms, materials, hierarchy, metadata, physics, geometry),
                so that one check that does not apply does not need force,
                which also lets every other error through.

        Returns:
            Dict with file_path (written exactly there; a missing folder is
            created), files_written, object_count (objects written), prims (object
            name and USD path), default_prim, meters_per_unit, up_axis, physics
            (with joint_count and joints) and semantics summaries,
            skipped_objects, validation_passed, issues and warnings (what the
            joints leave out among them, validated or not). When validation
            found errors and force is false, an error result with the issues
            instead, and no file is written.

        Raises:
            PermissionError: If the path is outside the sandbox policy.
            NotFoundError: For an unknown object (``details.code`` object_not_found).
            ValidationError: For an unsupported extension (UnsupportedExtension),
                an empty object_names (EmptyObjectList), an unknown check
                (UnknownCheck), a listed object that cannot be exported
                (NotInViewLayer) or nothing to export (NothingToExport).
        """
        # Written at file_path exactly, as export_file does.
        file_path = self._export_target(self._deny_outside_sandbox(file_path, write=True))
        extension = os.path.splitext(file_path)[1].lower()
        if extension not in self._SIMREADY_USD_EXTENSIONS:
            raise ValidationError(
                f"Unsupported SimReady export extension '{extension or file_path}'; "
                f"use one of {', '.join(self._SIMREADY_USD_EXTENSIONS)}",
                "UnsupportedExtension",
            )
        if object_names is not None and not object_names:
            raise ValidationError(
                "object_names is empty; list the objects to export, or pass null "
                "to export the whole scene",
                "EmptyObjectList",
            )
        unknown_checks = sorted(set(skip_checks or ()) - set(self._SIMREADY_CHECKS))
        if unknown_checks:
            raise ValidationError(
                self._unknown_name("check", unknown_checks[0], self._SIMREADY_CHECKS),
                "UnknownCheck",
            )
        targets, listed, skipped = self._simready_export_targets(
            object_names, include_children, include_lights_and_cameras
        )
        if not targets:
            raise ValidationError(
                "Nothing to export: "
                + "; ".join(f"{s['object_name']}: {s['reason']}" for s in skipped),
                "NothingToExport",
            )

        issues: Optional[List[Dict[str, Any]]] = None
        validation_passed: Optional[bool] = None
        if validate_before_export:
            # Every exported object is checked, the asset root included.
            issues = self._simready_issues(
                targets,
                {check: check not in (skip_checks or ()) for check in self._SIMREADY_CHECKS},
                self._scene_unit_scale(),
            )
            validation_passed = not any(i["severity"] == "error" for i in issues)
            if not validation_passed and not force:
                return self._simready_validation_refusal(issues)

        objects = []
        stored_problems: List[str] = []
        for obj in targets:
            valid, invalid = self._simready_checked(self._read_simready_properties(obj))
            objects.append(
                {
                    "name": obj.name,
                    "parent": obj.parent.name if obj.parent is not None else None,
                    "semantic": valid.get("semantic", {}),
                    "physics": valid.get("physics", {}),
                }
            )
            stored_problems.extend(
                f"{obj.name}: stored {i['property']}={i['value']!r} {i['problem']}; not exported"
                for i in invalid
            )
        joints, joint_issues = self._simready_joint_plan(targets, self._simready_world_resolver())
        plan = {
            "objects": objects,
            "joints": joints,
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
        # What the joints left out holds with or without validation.
        warnings = (
            [message for _, _, message, _ in joint_issues]
            + report["warnings"]
            + ([export_warning] if export_warning else [])
        )
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
                "joint_count": len(report["joints"]),
                "rigid_bodies": report["rigid_bodies"][:limit],
                "colliders": report["colliders"][:limit],
                "physics_materials": report["physics_materials"][:limit],
                "joints": report["joints"][:limit],
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
            "warnings": (stored_problems + warnings)[:limit],
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
            candidates = [o for o in blender_module.context.scene.objects if o is not None]
        else:
            listed = self._simready_objects(object_names)
            outside = sorted(o.name for o in listed if layer_objects.get(o.name) is None)
            if outside:
                raise ValidationError(
                    "Cannot export "
                    + ", ".join(outside)
                    + ": not in the active view layer (another scene, or an excluded "
                    "collection)",
                    "NotInViewLayer",
                    objects=outside,
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
            elif self._simready_constraint_empty(obj):
                reason = (
                    "a rigid body constraint: exported as a UsdPhysics joint between its two "
                    "objects, not as a prim"
                )
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
        reuse_root: bool = False,
    ) -> Dict[str, Any]:
        """
        Put objects under a SimReady root empty (an Xform in the USD), keeping where they are.

        Each listed object without a listed ancestor is parented to the root;
        a listed object under another listed object keeps its parent, so an
        existing nesting stays. The object's matrix before constraints is
        kept: from its former parent's world matrix (composed from the parent
        chain for an object Blender does not evaluate, such as one in an
        excluded collection). The transform moves into the object's own
        location, rotation and scale, or into its parent inverse matrix when
        its former parent's non-uniform scale skews it, which those cannot
        express, or when object constraints act on it: then its location,
        rotation and scale stay as they are, and so does the space a
        constraint works in, local or parent space included, so the
        constraints give the same pose. A bone or vertex parent becomes an
        object parent. Everything is checked before anything changes, and a
        failure part-way restores the parents and removes a root this call
        created.

        Args:
            root_name: Name for the root empty: a USD identifier (letters,
                digits and underscores, not starting with a digit).
            child_names: Existing objects to put under the root.
            semantic: Semantic labels to store on the root.
            reuse_root: Put the objects under an existing empty named root_name;
                without it, an existing object of that name is refused.

        Returns:
            Dict with root_name, root_created, children, reparented, kept_parent
            (each with its current parent, which may be an unlisted object
            between it and its listed ancestor), hierarchy_path (the root's
            prim path in export_simready_usd's USD), prim_paths of the children
            and warnings.

        Raises:
            NotFoundError: For unknown objects (``details.code`` object_not_found,
                ``details.objects`` the missing names). Nothing is changed.
            ValidationError: Nothing is changed. ``details.code`` names the
                reason: InvalidRootName, NoChildren, RootIsChild, RootExists,
                RootNotEmpty, RootNotInScene, RootCycle, RootNotInvertible,
                UnsupportedParent, ObjectModeRequired, LinkedData, or
                InvalidMetadata for the semantic labels.
        """
        blender_module: Any = bpy
        if not self._SIMREADY_NAME_RE.match(root_name):
            raise ValidationError(
                f"Root name '{root_name}' is not a USD identifier (letters, digits and "
                "underscores, not starting with a digit)",
                "InvalidRootName",
            )
        if not child_names:
            raise ValidationError(
                "child_names is empty; list the objects to put under the root", "NoChildren"
            )
        children = self._simready_objects(list(dict.fromkeys(child_names)))
        if any(child.name == root_name for child in children):
            raise ValidationError(f"'{root_name}' cannot be put under itself", "RootIsChild")
        root = blender_module.data.objects.get(root_name)
        if root is not None:
            if not reuse_root:
                raise ValidationError(
                    f"An object named '{root_name}' already exists; pass reuse_root=true to "
                    "put the objects under it, or choose another root_name. Nothing was changed",
                    "RootExists",
                )
            if root.type != "EMPTY":
                raise ValidationError(
                    f"'{root_name}' is a {str(root.type).lower()}, not an empty; a SimReady "
                    "root must be an empty (an Xform in the USD)",
                    "RootNotEmpty",
                )
            if blender_module.context.scene.objects.get(root_name) is None:
                raise ValidationError(
                    f"'{root_name}' is not in the current scene", "RootNotInScene"
                )
            above = {a.name for a in self._ancestors(root)}
            cycle = sorted(child.name for child in children if child.name in above)
            if cycle:
                raise ValidationError(
                    f"{', '.join(cycle)} is above '{root_name}'; it cannot also go under it",
                    "RootCycle",
                    objects=cycle,
                )
        self._simready_editable(
            children + ([root] if root is not None else []), "setup_simready_hierarchy"
        )
        semantic_writes = self._simready_writes({"semantic": semantic}) if semantic else []
        listed = {child.name for child in children}
        top = [c for c in children if not any(a.name in listed for a in self._ancestors(c))]
        kept = [c for c in children if c not in top]

        update = getattr(getattr(blender_module.context, "view_layer", None), "update", None)
        if update is not None:
            update()  # current world matrices to keep
        worlds = self._simready_world_resolver()
        root_world = worlds.world(root) if root is not None else [list(r) for r in _IDENTITY4]
        root_inverse = _m4_inverse(root_world)
        if root_inverse is None:
            raise ValidationError(
                f"The root '{root_name}' has zero scale, so nothing can keep its place under it",
                "RootNotInvertible",
            )
        plans = [
            (child, self._simready_parent_space(child, worlds))
            for child in top
            if root is None or child.parent is not root
        ]
        before = {child.name: worlds.world(child) for child, _ in plans}

        created = root is None
        if created:
            root = blender_module.data.objects.new(root_name, None)
            if root.name != root_name:
                blender_module.data.objects.remove(root)
                raise ValidationError(
                    f"Blender named the root '{root.name}'; nothing was changed", "RootExists"
                )
            blender_module.context.scene.collection.objects.link(root)
            root.empty_display_type = "PLAIN_AXES"
            root.location = (0.0, 0.0, 0.0)
            root.rotation_euler = (0.0, 0.0, 0.0)
            root.scale = (1.0, 1.0, 1.0)
        undo: List[Tuple[Any, Dict[str, Any]]] = []
        reparented: List[str] = []
        warnings: List[str] = []
        try:
            for child, space in plans:
                undo.append((child, self._simready_parenting(child)))
                child.parent = root
                if getattr(child, "parent_type", "OBJECT") != "OBJECT":
                    child.parent_type = "OBJECT"
                    child.parent_bone = ""
                basis = _m4_mul(space, _m4(child.matrix_basis))
                skewed = _m4_shape(basis)[2] > self._SIMREADY_SHEAR
                if not skewed and not self._simready_constrained(child):
                    # root @ inverse(root) @ basis: the transform is the object's own.
                    child.matrix_parent_inverse = self._matrix(root_inverse)
                    child.matrix_basis = self._matrix(basis)
                else:
                    # The former parent space goes into the parent inverse and the
                    # basis stays, so parent world @ parent inverse is what it was:
                    # a constraint sees the same local and parent space as before.
                    child.matrix_parent_inverse = self._matrix(_m4_mul(root_inverse, space))
                    if skewed:
                        warnings.append(
                            f"{child.name}: its former parent's non-uniform scale skews it, "
                            "which location, rotation and scale cannot express, so the skew is "
                            "kept in its parent inverse matrix; validation reports the skew"
                        )
                reparented.append(child.name)
            for key, value in semantic_writes:
                root[key] = value
        except Exception:
            for child, state in reversed(undo):
                # Best effort: one object that refuses must not keep the rest moved.
                for attribute, value in state.items():
                    try:
                        setattr(child, attribute, value)
                    except Exception:  # noqa: BLE001
                        logger.exception("Could not restore %s.%s", child.name, attribute)
            if created:
                blender_module.data.objects.remove(root)
            raise
        if update is not None:
            update()
        after = self._simready_world_resolver()
        for child, _ in plans:
            if after.evaluated(child):
                moved = max(
                    abs(a - b)
                    for row_a, row_b in zip(before[child.name], after.world(child))
                    for a, b in zip(row_a, row_b)
                )
                if moved > 1e-4:
                    warnings.append(
                        f"{child.name}: its evaluated transform changed by up to {moved:.4g} "
                        "after re-parenting (a constraint or driver that reads its parent); "
                        "check its pose"
                    )

        root_path = f"/{usd_identifier(root.name)}"
        paths: Dict[str, str] = {}

        def prim_path(obj: Any) -> str:
            names = [usd_identifier(obj.name)]
            for ancestor in self._ancestors(obj):
                if ancestor is root:
                    break
                names.append(usd_identifier(ancestor.name))
            return root_path + "/" + "/".join(reversed(names))

        for child in children:
            path = prim_path(child)
            clash = next((n for n, p in paths.items() if p == path), None)
            if clash is not None:
                warnings.append(
                    f"{clash} and {child.name} both become the prim {path}; the export renames "
                    "one of them (it reports the paths it wrote)"
                )
            paths[child.name] = path
        return {
            "root_name": root.name,
            "root_created": created,
            "children": [child.name for child in children],
            "reparented": reparented,
            "kept_parent": [
                {"object_name": child.name, "parent": child.parent.name} for child in kept
            ],
            "hierarchy_path": root_path,
            "prim_paths": paths,
            "warnings": warnings,
        }

    def _simready_parent_space(self, child: Any, worlds: "_WorldResolver") -> _Matrix4:
        """The matrix the child's basis is multiplied by: its world before constraints.

        For an object parent that is the parent's world matrix times the
        parent inverse matrix, whether or not Blender evaluates the child. A
        bone or vertex parent's matrix is read from the child's evaluated
        world matrix, which needs an evaluated child without constraints.

        Raises:
            ValidationError: For a bone or vertex parent that cannot be read.
        """
        parent = child.parent
        if parent is None:
            return [list(row) for row in _IDENTITY4]
        if getattr(child, "parent_type", "OBJECT") == "OBJECT":
            return _m4_mul(worlds.world(parent), _m4(child.matrix_parent_inverse))
        basis_inverse = _m4_inverse(_m4(child.matrix_basis))
        if (
            self._simready_constrained(child)
            or not worlds.evaluated(child)
            or basis_inverse is None
        ):
            raise ValidationError(
                f"{child.name} has a {str(child.parent_type).lower()} parent whose matrix cannot "
                "be read (it has constraints, is not evaluated, or has zero scale); parent it to "
                "an object first",
                "UnsupportedParent",
                objects=[child.name],
            )
        return _m4_mul(worlds.world(child), basis_inverse)

    @staticmethod
    def _simready_constrained(obj: Any) -> bool:
        """Whether an object constraint acts on the object (not muted, with influence)."""
        return any(
            not getattr(c, "mute", False) and float(getattr(c, "influence", 1.0)) > 0.0
            for c in getattr(obj, "constraints", ()) or ()
        )

    @staticmethod
    def _simready_parenting(obj: Any) -> Dict[str, Any]:
        """What re-parenting changes on an object, for the rollback."""
        state = {
            "parent": obj.parent,
            "matrix_parent_inverse": obj.matrix_parent_inverse.copy(),
            "matrix_basis": obj.matrix_basis.copy(),
        }
        if hasattr(obj, "parent_type"):
            state = {
                "parent": obj.parent,
                "parent_type": obj.parent_type,
                "parent_bone": getattr(obj, "parent_bone", ""),
                "matrix_parent_inverse": state["matrix_parent_inverse"],
                "matrix_basis": state["matrix_basis"],
            }
        return state

    @staticmethod
    def _matrix(rows: _Matrix4) -> Any:
        """A mathutils Matrix of ``rows`` for a matrix property (the rows outside Blender)."""
        try:
            return _mathutils().Matrix(rows)
        except ImportError:
            return rows

    @staticmethod
    def _ancestors(obj: Any) -> Iterator[Any]:
        """The object's parent, its parent's parent, and so on."""
        seen: set = set()
        current = getattr(obj, "parent", None)
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            yield current
            current = getattr(current, "parent", None)

    # -- End SimReady methods --------------------------------------------------

    def cleanup(self) -> None:
        """Clean up resources for the Blender session."""
        self.logger.debug("Blender runtime session cleaned up")

    def _simready_objects(self, object_names: Sequence[str]) -> List[Any]:
        """The named objects in order, or one NotFoundError naming every missing one.

        Its ``details`` are _get_object_or_raise's (code object_not_found and
        the first missing object_name) plus ``objects``, all the missing names.
        """
        found: List[Any] = []
        missing: List[str] = []
        for name in object_names:
            try:
                found.append(self._get_object_or_raise(name))
            except NotFoundError:
                missing.append(name)
        if missing:
            raise NotFoundError(
                f"Object{'s' if len(missing) > 1 else ''} not found: {', '.join(missing)}; "
                "nothing was changed",
                "object_not_found",
                object_name=missing[0],
                objects=missing,
            )
        return found

    def _get_object_or_raise(self, object_name: str) -> Any:
        """
        Look up a Blender object by name or raise NotFoundError (a ValueError).

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
            raise NotFoundError(
                f"Object not found: {object_name}", "object_not_found", object_name=object_name
            )
        return obj

    def _extract_principled_params(self, node: Any) -> Dict[str, Any]:
        """
        Extract common Principled BSDF input values.

        Version-aware: Blender 4.0+ renamed several socket labels. An input a
        link drives reports None: its own value is unused, and reporting it
        made a textured Base Color read as the default grey.

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
            if getattr(sock, "is_linked", False) and _live_links(sock):
                # A muted link drives nothing: the socket's own value applies.
                params[sock_name] = None
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
            raise NotFoundError(
                f"Collection not found: {collection_name}",
                "collection_not_found",
                collection_name=collection_name,
            )
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
                raise NotFoundError(
                    f"Camera not found: {camera_name}", "camera_not_found", camera_name=camera_name
                )
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
        camera: Any = None,
    ) -> Dict[str, Any]:
        """
        Capture the 3D View as the user sees it, or through a camera; see capture_viewport.

        Args:
            width: Largest image width.
            height: Largest image height.
            jpeg_quality: JPEG compression quality.
            engine: Render engine identifier.
            camera: Camera to draw the 3D View's shading from instead of its
                own angle, or None.

        Returns:
            Dictionary with image_base64, dimensions, engine, capture_method.

        Raises:
            RuntimeError: When there is no 3D View, or a camera is asked for
                while an external engine draws the view, which only the
                screen shows.
        """
        blender_module: Any = bpy
        context = blender_module.context
        area, region, space = self._view3d_target()
        matrices = None
        if camera is not None:
            if self._drawn_by_external_engine(space, engine):
                raise RuntimeError(
                    f"The 3D View shows {engine}'s Rendered shading, which Blender draws "
                    f"only on screen, so it cannot be drawn through camera {camera.name!r}; "
                    "pass method='render' to render through it"
                )
            render = context.scene.render
            out_width, out_height = self._fit_inside(
                render.resolution_x * render.pixel_aspect_x,
                render.resolution_y * render.pixel_aspect_y,
                width,
                height,
            )
            matrices = self._camera_matrices(camera, out_width, out_height)
        else:
            out_width, out_height = self._fit_inside(
                region.width, region.height, width, height
            )
        with self._capture_workdir() as workdir:
            if camera is None and self._drawn_by_external_engine(space, engine):
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
                    matrices,
                )
                capture_method = "viewport" if camera is None else "viewport_camera"
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
            "camera_name": None if camera is None else camera.name,
        }

    @staticmethod
    def _camera_matrices(camera: Any, width: int, height: int) -> Tuple[Any, Any]:
        """Return a camera's view and projection matrices for a width x height image.

        The pose is the evaluated one, constraints included, as a render uses.
        """
        blender_module: Any = bpy
        depsgraph = blender_module.context.evaluated_depsgraph_get()
        evaluated = camera.evaluated_get(depsgraph)
        view_matrix = evaluated.matrix_world.inverted()
        projection = evaluated.calc_matrix_camera(depsgraph, x=width, y=height)
        return view_matrix, projection

    def _draw_view3d_offscreen(
        self,
        scene: Any,
        view_layer: Any,
        space: Any,
        region: Any,
        width: int,
        height: int,
        matrices: Optional[Tuple[Any, Any]] = None,
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
            matrices: View and projection matrices to draw with instead of the
                region's, such as a camera's.

        Returns:
            A temporary image with the pixels; the caller removes it.
        """
        import gpu  # type: ignore[import-not-found]

        if matrices is None:
            view = region.data
            # Blender refreshes view matrices on redraw; a view changed since the
            # last one, or a window that is not redrawing, would be stale.
            view.update()
            matrices = (view.view_matrix, view.window_matrix)
        offscreen = gpu.types.GPUOffScreen(width, height)
        try:
            offscreen.draw_view3d(
                scene,
                view_layer,
                space,
                region,
                matrices[0],
                matrices[1],
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
        chosen: Any = None,
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
            chosen: Camera to render through instead of the scene camera, which
                is restored afterwards; None renders through the scene camera.

        Returns:
            Dictionary with image_base64, dimensions, engine, capture_method
            and camera_name.
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
            scene_camera = scene.camera
            camera = None
            try:
                if chosen is not None:
                    scene.camera = chosen
                elif scene.camera is None:
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
                if chosen is not None:
                    scene.camera = scene_camera
        used = chosen if chosen is not None else scene_camera
        return {
            "image_base64": b64,
            "width": width,
            "height": height,
            "engine": engine,
            "capture_method": capture_method,
            "format": "jpeg",
            "camera_name": None if camera is not None or used is None else used.name,
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
            "blender_uv_unwrap",
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
