"""
Blender runtime adapter for Simul.

This module provides an adapter for Blender runtime operations through the
optional `bpy` Python module.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import os
import re
import tempfile
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
    Optional,
    Sequence,
    Tuple,
    overload,
)

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

# Output-format settings a capture changes and restores, in restore order:
# media_type first, so that the saved file_format is valid again.
_IMAGE_FORMAT_FIELDS = (
    "media_type",
    "file_format",
    "color_mode",
    "color_depth",
    "quality",
)


def add_context_hint(error: Optional[str]) -> Optional[str]:
    """Append the context-override recipe to a missing ``bpy.context`` member error.

    Idempotent, so both the add-on and the attached client may apply it.
    """
    if not error or _CONTEXT_MEMBER_ERROR not in error or CONTEXT_OVERRIDE_HINT in error:
        return error
    return f"{error}. {CONTEXT_OVERRIDE_HINT}"


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

        Args:
            object_name: Name of the target object.

        Returns:
            Dictionary with transforms, parent, modifiers, constraints,
            material slots, and visibility.
        """
        obj = self._get_object_or_raise(object_name)

        location = list(obj.location)
        rotation = list(obj.rotation_euler)
        scale = list(obj.scale)

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
            "rotation_euler": rotation,
            "scale": scale,
            "parent_name": parent_name,
            "children_names": children_names,
            "modifiers": modifiers,
            "constraints": constraints,
            "material_slots": material_slots,
            "visible": self._is_object_visible(obj),
        }

    def get_mesh_info(self, object_name: str) -> Dict[str, Any]:
        """
        Get counts-only mesh geometry information (O(1) attribute access).

        Args:
            object_name: Name of the mesh object.

        Returns:
            Dictionary with vertex/edge/face counts and UV layer names.
        """
        obj = self._get_object_or_raise(object_name)
        if obj.type != "MESH":
            raise ValueError(f"Object '{object_name}' is not a MESH (type={obj.type})")

        mesh_data: Any = obj.data
        uv_layer_names = [layer.name for layer in mesh_data.uv_layers]
        has_shape_keys = mesh_data.shape_keys is not None

        return {
            "object_name": object_name,
            "vertex_count": len(mesh_data.vertices),
            "edge_count": len(mesh_data.edges),
            "face_count": len(mesh_data.polygons),
            "uv_layer_names": uv_layer_names,
            "has_shape_keys": has_shape_keys,
        }

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
            name_pattern: Regex pattern for object name matching.
            object_type: Blender object type filter (MESH, LIGHT, etc.).
            max_results: Maximum number of results.

        Returns:
            Dictionary with matched objects and truncation flag.
        """
        all_objects = self._resolve_object_source(None)

        compiled_pattern = re.compile(name_pattern) if name_pattern else None
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

        loc_a = list(obj_a.matrix_world.translation)
        loc_b = list(obj_b.matrix_world.translation)
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
        loc = list(obj.matrix_world.translation)

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
        QUATERNION or AXIS_ANGLE). The camera's scale is kept.

        Args:
            location: World-space camera location [x, y, z].
            rotation_euler: World-space camera rotation [rx, ry, rz] in radians, as an XYZ Euler.
            camera_name: Target camera name. Uses active camera when None.

        Returns:
            Dictionary with camera_name and the camera's world-space location
            and rotation_euler.
        """
        cam_obj = self._resolve_camera(camera_name)
        self._place_in_world(cam_obj, location, rotation_euler)
        world_location, world_rotation = self._world_placement(cam_obj, rotation_euler)
        return {
            "camera_name": cam_obj.name,
            "location": world_location,
            "rotation_euler": world_rotation,
        }

    def get_camera_info(self, camera_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a camera's world-space placement and lens properties.

        Args:
            camera_name: Target camera name. Uses active camera when None.

        Returns:
            Dictionary with the world-space location and rotation_euler (XYZ
            Euler radians, as set_camera_view takes them), lens, sensor, clip
            distances and camera type.
        """
        cam_obj = self._resolve_camera(camera_name)
        cam_data = cam_obj.data
        location, rotation = self._world_placement(cam_obj)
        return {
            "camera_name": cam_obj.name,
            "location": location,
            "rotation_euler": rotation,
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

        Args:
            object_name: Object to focus on.
            distance_factor: Multiple of the framing distance (of ortho_scale for an
                orthographic camera): 1.0 fits the object's bounding sphere in the
                frame with 5% to spare, 2.0 shows the object at half that size, and
                below 1.0 crops it.
            camera_name: Target camera name. Uses active camera when None.

        Returns:
            Dictionary with camera_name, object_name, camera_location (world
            space), look_at, distance (camera to look_at), clip_start,
            clip_end, clip_adjusted (whether the clip range was widened) and
            ortho_scale (None unless the camera is orthographic).

        Raises:
            ValueError: When the camera's lens shift moves its frame off its
                view axis, so no placement centers the object in it.
        """
        obj = self._get_object_or_raise(object_name)
        cam_obj = self._resolve_camera(camera_name)
        blender_module: Any = bpy
        render = blender_module.context.scene.render
        data = cam_obj.data

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
        self._place_in_world(
            cam_obj, location, self._look_at_rotation(location, center)
        )
        clip_adjusted = self._clip_around(data, distance, radius)

        return {
            "camera_name": cam_obj.name,
            "object_name": object_name,
            "camera_location": list(cam_obj.matrix_world.translation),
            "look_at": center,
            "distance": distance,
            "clip_start": data.clip_start,
            "clip_end": data.clip_end,
            "clip_adjusted": clip_adjusted,
            "ortho_scale": ortho_scale,
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
    ) -> Dict[str, Any]:
        """
        Set an object's location, rotation, and/or scale.

        Args:
            object_name: Object to transform.
            location: New location [x, y, z] or None to keep current.
            rotation_euler: New rotation [rx, ry, rz] or None to keep current.
            scale: New scale [sx, sy, sz] or None to keep current.

        Returns:
            Dictionary with final transform values.
        """
        obj = self._get_object_or_raise(object_name)
        if location is not None:
            obj.location = tuple(location)
        if rotation_euler is not None:
            obj.rotation_euler = tuple(rotation_euler)
        if scale is not None:
            obj.scale = tuple(scale)

        return {
            "object_name": object_name,
            "location": list(obj.location),
            "rotation_euler": list(obj.rotation_euler),
            "scale": list(obj.scale),
        }

    def set_object_parent(self, child_name: str, parent_name: str) -> Dict[str, Any]:
        """
        Parent one object to another.

        Args:
            child_name: Object to parent.
            parent_name: Object to be the parent.

        Returns:
            Dictionary with child_name and parent_name.
        """
        child = self._get_object_or_raise(child_name)
        parent = self._get_object_or_raise(parent_name)
        child.parent = parent
        return {"child_name": child_name, "parent_name": parent_name}

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
            # Keep rotation and scale as well as translation: the full world
            # matrix, not just its translation, becomes the local transform.
            world_matrix = obj.matrix_world.copy()
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

        Args:
            object_name: Object to add modifier to.
            modifier_type: Modifier type (SUBSURF, MIRROR, ARRAY, etc.).
            modifier_name: Modifier display name.
            params: Modifier-specific parameters to set.

        Returns:
            Dictionary with object_name, modifier_name, modifier_type.
        """
        obj = self._get_object_or_raise(object_name)
        mod_name = modifier_name or modifier_type.title()
        mod = obj.modifiers.new(name=mod_name, type=modifier_type.upper())

        if params:
            for key, value in params.items():
                if hasattr(mod, key):
                    setattr(mod, key, value)
                else:
                    self.logger.warning(
                        "Modifier %s has no attribute '%s'", mod.type, key
                    )

        return {
            "object_name": object_name,
            "modifier_name": mod.name,
            "modifier_type": mod.type,
        }

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

    # Selection keyword varies by format and version.
    _SELECTION_KW_LEGACY: Dict[str, str] = {
        "OBJ": "use_selection",
        "FBX": "use_selection",
        "GLTF": "export_selected",
        "USD": "selected_objects_only",
        "STL": "use_selection",
        "PLY": "use_selection",
    }
    _SELECTION_KW_V4: Dict[str, str] = {
        "OBJ": "export_selected_objects",
        "FBX": "use_selection",
        "GLTF": "export_selected",
        "USD": "selected_objects_only",
        "STL": "export_selected_objects",
        "PLY": "export_selected_objects",
    }

    _SUPPORTED_FORMATS: Tuple[str, ...] = ("OBJ", "FBX", "GLTF", "USD", "STL", "PLY")

    def open_blend_file(self, file_path: str) -> Dict[str, Any]:
        """
        Open a .blend file, replacing the current scene.

        Args:
            file_path: Absolute path to the .blend file.

        Returns:
            Dict with file_path and object_count.

        Raises:
            PermissionError: If the path is outside the sandbox policy.
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not a .blend file.
        """
        file_path = self._deny_outside_sandbox(file_path)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        if not file_path.lower().endswith(".blend"):
            raise ValueError(f"Not a .blend file: {file_path}")

        blender_module: Any = bpy
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

        Args:
            file_path: Path to save to. None saves in place.

        Returns:
            Dict with file_path.

        Raises:
            PermissionError: If the path is outside the sandbox policy.
            ValueError: If no path given and file was never saved.
        """
        file_path = self._deny_outside_sandbox(file_path, write=True)
        blender_module: Any = bpy

        if file_path is not None:
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

    def import_file(self, file_path: str, file_format: str) -> Dict[str, Any]:
        """
        Import a file into the scene. Version-aware operator dispatch.

        Args:
            file_path: Absolute path to the file.
            file_format: One of OBJ, FBX, GLTF, USD, STL, PLY.

        Returns:
            Dict with file_path, file_format, and imported_objects list.

        Raises:
            PermissionError: If the path is outside the sandbox policy.
            FileNotFoundError: If the file does not exist.
            ValueError: If the format is unsupported.
        """
        file_path = self._deny_outside_sandbox(file_path)
        fmt = file_format.upper()
        if fmt not in self._SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {file_format}. "
                f"Valid: {', '.join(self._SUPPORTED_FORMATS)}"
            )
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        blender_module: Any = bpy
        existing_names = set(blender_module.data.objects.keys())

        op_map = (
            self._IMPORT_OPS_V4 if self._is_blender_4_plus else self._IMPORT_OPS_LEGACY
        )
        op_path = op_map[fmt]
        parts = op_path.split(".")
        op_func = getattr(getattr(blender_module.ops, parts[0]), parts[1])
        op_func(filepath=file_path)

        new_names = [
            n for n in blender_module.data.objects.keys() if n not in existing_names
        ]
        return {
            "file_path": file_path,
            "file_format": fmt,
            "imported_objects": new_names,
        }

    def export_file(
        self,
        file_path: str,
        file_format: str,
        selected_only: bool = False,
    ) -> Dict[str, Any]:
        """
        Export scene objects to a file. Version-aware operator dispatch.

        Args:
            file_path: Absolute path to export to.
            file_format: One of OBJ, FBX, GLTF, USD, STL, PLY.
            selected_only: If True, export only selected objects.

        Returns:
            Dict with file_path and file_format.

        Raises:
            PermissionError: If the path is outside the sandbox policy.
            ValueError: If the format is unsupported.
        """
        file_path = self._deny_outside_sandbox(file_path, write=True)
        return self._run_export(file_path, file_format, selected_only)

    def _run_export(
        self,
        file_path: str,
        file_format: str,
        selected_only: bool = False,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Dispatch the version-aware export operator.

        ``options`` are extra operator properties, passed only when this
        Blender's operator defines them.
        """
        fmt = file_format.upper()
        if fmt not in self._SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {file_format}. "
                f"Valid: {', '.join(self._SUPPORTED_FORMATS)}"
            )

        blender_module: Any = bpy
        op_map = (
            self._EXPORT_OPS_V4 if self._is_blender_4_plus else self._EXPORT_OPS_LEGACY
        )
        op_path = op_map[fmt]
        parts = op_path.split(".")
        op_func = getattr(getattr(blender_module.ops, parts[0]), parts[1])

        kwargs: Dict[str, Any] = {"filepath": file_path}
        if selected_only:
            sel_kw_map = (
                self._SELECTION_KW_V4
                if self._is_blender_4_plus
                else self._SELECTION_KW_LEGACY
            )
            kwargs[sel_kw_map[fmt]] = True
        if options:
            supported = op_func.get_rna_type().properties.keys()
            kwargs.update({k: v for k, v in options.items() if k in supported})

        op_func(**kwargs)
        return {"file_path": file_path, "file_format": fmt}

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

        Args:
            frame_start: Start frame of the animation range.
            frame_end: End frame of the animation range.

        Returns:
            Dict with the frame range that was set.
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
        return {
            "frame_start": scene.frame_start,
            "frame_end": scene.frame_end,
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
            RuntimeError: When Blender inserted no keyframe, e.g. into a
                locked F-curve.
        """
        obj = self._get_object_or_raise(object_name)
        # Blender reports refused keys by returning False, not by raising.
        if not obj.keyframe_insert(data_path, index=index, frame=frame):
            raise RuntimeError(
                f"Blender inserted no keyframe on '{object_name}' "
                f"{_channel_name(data_path, index)} at frame {frame}; its "
                "F-curve may be locked or sampled"
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
        if scene.rigidbody_world is None:
            blender_module.ops.rigidbody.world_add()

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
        if scene.rigidbody_world is None:
            blender_module.ops.rigidbody.world_add()

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

    def get_physics_state(self, object_name: str) -> Dict[str, Any]:
        """
        Get current physics state of an object.

        Args:
            object_name: Name of the object.

        Returns:
            Dict with object_name, location, rotation_euler, is_active,
            has_rigid_body, mass, collision_shape.
        """
        obj = self._get_object_or_raise(object_name)
        rb = getattr(obj, "rigid_body", None)
        has_rb = rb is not None

        location, rotation = self._world_pose(obj)
        return {
            "object_name": object_name,
            "location": location,
            "rotation_euler": rotation,
            "is_active": rb is not None and rb.type == "ACTIVE",
            "has_rigid_body": has_rb,
            "mass": float(rb.mass) if rb is not None else None,
            "collision_shape": rb.collision_shape if rb is not None else None,
        }

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
    ) -> Dict[str, Any]:
        """
        Bake all physics simulations in the scene.

        Args:
            frame_start: First frame to bake.
            frame_end: Last frame to bake.

        Returns:
            Dict with frame_start, frame_end.
        """
        blender_module: Any = bpy
        scene = blender_module.context.scene

        if scene.rigidbody_world is None:
            raise ValueError("No rigid body world to bake")

        pc = scene.rigidbody_world.point_cache
        pc.frame_start = frame_start
        pc.frame_end = frame_end

        # Blender 4.0 removed the positional context-override dict.
        with blender_module.context.temp_override(scene=scene, point_cache=pc):
            blender_module.ops.ptcache.bake(bake=True)

        return {
            "frame_start": frame_start,
            "frame_end": frame_end,
        }

    def free_bake(self) -> Dict[str, Any]:
        """
        Free all baked physics simulation data.

        Returns:
            Empty dict (success indicated by caller).
        """
        blender_module: Any = bpy
        scene = blender_module.context.scene

        if scene.rigidbody_world is None:
            raise ValueError("No rigid body world to free")

        pc = scene.rigidbody_world.point_cache
        with blender_module.context.temp_override(scene=scene, point_cache=pc):
            blender_module.ops.ptcache.free_bake()

        return {}

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
            except Exception as exc:
                return {
                    **_captured_output(),
                    "return_value": None,
                    "duration_seconds": round(time.monotonic() - start, 4),
                    "error": add_context_hint(f"{type(exc).__name__}: {exc}"),
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

        Uses ``bpy.data.meshes.new()`` + ``mesh.from_pydata()`` which is the
        standard Blender Python API for procedural mesh creation.

        Args:
            name: Desired object name.
            vertices: List of [x, y, z] vertex positions.
            edges: List of [v0, v1] edge index pairs.
            faces: List of vertex-index lists per face.
            location: Optional world-space origin [x, y, z].
            collection_name: Target collection (scene collection if None).

        Returns:
            Dict with object_name, mesh_name, vertex_count,
            edge_count, face_count.

        Raises:
            ValueError: If vertices list is empty or indices are invalid.
        """
        if not vertices:
            raise ValueError("vertices list must not be empty")

        blender_module: Any = bpy
        safe_edges: List[List[int]] = edges or []
        safe_faces: List[List[int]] = faces or []

        # Validate index bounds
        num_verts = len(vertices)
        for ei, edge in enumerate(safe_edges):
            if len(edge) != 2:
                raise ValueError(
                    f"Edge {ei} must have exactly 2 indices, got {len(edge)}"
                )
            for idx in edge:
                if idx < 0 or idx >= num_verts:
                    raise ValueError(
                        f"Edge {ei} index {idx} out of range " f"[0, {num_verts})"
                    )
        for fi, face in enumerate(safe_faces):
            if len(face) < 3:
                raise ValueError(f"Face {fi} must have >= 3 indices, got {len(face)}")
            for idx in face:
                if idx < 0 or idx >= num_verts:
                    raise ValueError(
                        f"Face {fi} index {idx} out of range " f"[0, {num_verts})"
                    )

        # Create mesh data-block and populate
        mesh = blender_module.data.meshes.new(name)
        vert_tuples = [tuple(v) for v in vertices]
        edge_tuples = [tuple(e) for e in safe_edges]
        face_tuples = [tuple(f) for f in safe_faces]
        mesh.from_pydata(vert_tuples, edge_tuples, face_tuples)
        mesh.update()

        # Create object and link to scene
        obj = blender_module.data.objects.new(name, mesh)
        if location is not None:
            obj.location = tuple(location)

        if collection_name is not None:
            col = blender_module.data.collections.get(collection_name)
            if col is None:
                raise ValueError(f"Collection '{collection_name}' not found")
            col.objects.link(obj)
        else:
            blender_module.context.scene.collection.objects.link(obj)

        return {
            "object_name": obj.name,
            "mesh_name": mesh.name,
            "vertex_count": len(mesh.vertices),
            "edge_count": len(mesh.edges),
            "face_count": len(mesh.polygons),
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

        semantic: Dict[str, Any] = {}
        physics: Dict[str, Any] = {}
        material: Dict[str, Any] = {}
        prefix = self._SIMREADY_PREFIX
        additional_labels: Dict[str, str] = {}

        for key in obj.keys():
            if not key.startswith(prefix):
                continue
            stripped = key[len(prefix) :]
            val = obj[key]
            # Convert IDPropertyArray / other exotic types to Python
            if hasattr(val, "to_list"):
                val = val.to_list()

            if stripped.startswith("label_"):
                additional_labels[stripped[6:]] = str(val)
            elif stripped in ("semantic_class", "semantic_hierarchy", "semantic_qcode"):
                semantic[stripped] = val
            elif stripped in (
                "mass_kg",
                "collider_type",
                "static_friction",
                "dynamic_friction",
                "restitution",
                "density",
                "is_rigid_body",
            ):
                physics[stripped] = val
            elif stripped in (
                "substrate_type",
                "material_naming",
                "shader_type",
                "texel_density",
            ):
                material[stripped] = val

        if additional_labels:
            semantic["additional_labels"] = additional_labels

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

            if check_scale and hasattr(obj, "dimensions"):
                dims = list(obj.dimensions)
                max_dim = max(abs(d) for d in dims) if dims else 0.0
                if max_dim > 1000.0:
                    issues.append(
                        {
                            "object_name": name,
                            "check": "scale",
                            "severity": "warning",
                            "message": (
                                f"Largest dimension {max_dim:.2f}m exceeds "
                                "1000m — verify scene is in meters"
                            ),
                            "suggestion": "Ensure scene unit scale is 1.0 (meters)",
                        }
                    )
                if max_dim < 1e-4 and max_dim > 0:
                    issues.append(
                        {
                            "object_name": name,
                            "check": "scale",
                            "severity": "warning",
                            "message": (
                                f"Largest dimension {max_dim:.6f}m is very small "
                                "— may not be in meter scale"
                            ),
                            "suggestion": "Verify units; SimReady uses meters",
                        }
                    )

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
        }

    def export_simready_usd(
        self,
        file_path: str,
        object_names: Optional[List[str]] = None,
        embed_metadata: bool = True,
        validate_before_export: bool = True,
    ) -> Dict[str, Any]:
        """
        Export a SimReady-compliant USD file.

        Validates objects before export (optionally), selects the requested
        objects, and delegates to the version-aware USD exporter.  SimReady
        custom properties on the Blender objects are carried into the USD
        as custom attributes automatically by Blender's USD exporter.

        Args:
            file_path: Output .usd / .usda / .usdc path.
            object_names: Objects to export (None = all).
            embed_metadata: Include simready_ custom properties.
            validate_before_export: Run validation first.

        Returns:
            Dict with file_path, object_count, validation_passed, issues.

        Raises:
            PermissionError: If the path is outside the sandbox policy.
        """
        file_path = self._deny_outside_sandbox(file_path, write=True)
        issues: Optional[List[Dict[str, Any]]] = None
        validation_passed = True

        if object_names:
            targets = [self._get_object_or_raise(n) for n in object_names]
        else:
            targets = list(self._resolve_object_source(None))

        if validate_before_export:
            # SimReady rules apply to mesh prims; cameras and lights would
            # only produce spurious "No material" errors.
            names = [o.name for o in targets if o.type == "MESH"]
            result = self.validate_simready_compliance(object_names=names)
            issues = result.get("issues")
            validation_passed = result.get("compliant", True)

        # Select only the requested objects
        for obj in self._resolve_object_source(None):
            obj.select_set(False)
        for obj in targets:
            obj.select_set(True)

        self._run_export(
            file_path,
            "USD",
            selected_only=bool(object_names),
            options={"export_custom_properties": embed_metadata},
        )

        return {
            "file_path": file_path,
            "object_count": len(targets),
            "validation_passed": validation_passed,
            "issues": issues,
        }

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
    def _place_in_world(
        obj: Any, location: List[float], rotation_euler: Sequence[float]
    ) -> None:
        """
        Give an object a world-space location and XYZ Euler rotation, keeping its scale.

        Assigning ``matrix_world`` makes Blender solve the local transform
        through the parent and its inverse matrix, and store the rotation in
        the object's rotation_mode. Writing ``location`` and ``rotation_euler``
        instead places a parented object relative to its parent; a QUATERNION
        or AXIS_ANGLE object ignores rotation_euler, and another Euler order
        reads the angles in its own order.

        Args:
            obj: The object to place.
            location: World-space location [x, y, z].
            rotation_euler: World-space rotation [rx, ry, rz] in radians, XYZ order.
        """
        mathutils = _mathutils()
        obj.matrix_world = mathutils.Matrix.LocRotScale(
            mathutils.Vector(location),
            mathutils.Euler(rotation_euler, "XYZ"),
            obj.matrix_world.to_scale(),
        )

    @staticmethod
    def _world_placement(
        obj: Any, near: Optional[Sequence[float]] = None
    ) -> Tuple[List[float], List[float]]:
        """
        Return an object's world-space location and rotation as an XYZ Euler.

        Several Euler triples give the same rotation; the one nearest ``near``
        (the object's own rotation_euler by default) is returned, so a
        rotation set as given reads back as given.

        Args:
            obj: The object.
            near: Euler angles the result stays close to.

        Returns:
            The location [x, y, z] and rotation [rx, ry, rz] in radians.
        """
        matrix = obj.matrix_world
        compatible = _mathutils().Euler(
            obj.rotation_euler if near is None else near, "XYZ"
        )
        return list(matrix.translation), list(matrix.to_euler("XYZ", compatible))

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
