"""
Blender runtime adapter for Simul MCP Server.

This module provides an adapter for Blender runtime operations through the
optional `bpy` Python module.
"""

import base64
import io
import math
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import bpy

    BLENDER_AVAILABLE = True
except ImportError:
    bpy = None
    BLENDER_AVAILABLE = False

from ..config import Settings, get_settings
from ..logging import LoggerMixin, get_logger
from ..utils.paths import PathPolicy

logger = get_logger(__name__)


@dataclass
class BlenderObjectEntry:
    """Serializable Blender object summary."""

    name: str
    object_type: str
    collection: Optional[str]
    visible: bool


class BlenderRuntimeSession(LoggerMixin):
    """
    Blender runtime session for scene inspection operations.

    This class provides read-focused operations that are safe for environments
    where Blender is running as a Python module.
    """

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize Blender runtime session.

        Args:
            settings: Configuration settings.
        """
        if not BLENDER_AVAILABLE:
            raise ImportError(
                "Blender runtime not available. Please run in an environment "
                "where the bpy module is installed."
            )

        self.settings = settings or get_settings()

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
        self._path_policy = PathPolicy.from_settings(self.settings)

    def _deny_outside_sandbox(self, path: Optional[str]) -> Optional[str]:
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
        """
        if path is None:
            return None
        if not self._path_policy.enabled:
            return path
        if not self._path_policy.is_allowed(path):
            raise PermissionError(f"File path is not allowed by sandbox policy: {path}")
        return str(self._path_policy.resolve(path))

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
            matrix = obj.matrix_world
            corners = [list(matrix @ co) for co in obj.bound_box]
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
        blender_module: Any = bpy
        all_objects = blender_module.data.objects

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
        all_objects = blender_module.data.objects
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
    ) -> Dict[str, Any]:
        """
        Capture the viewport as a base64-encoded JPEG image.

        Fast path uses gpu.types.GPUOffScreen when available (EEVEE only).
        Fallback renders via bpy.ops.render.render (supports Cycles).

        Args:
            width: Output image width in pixels.
            height: Output image height in pixels.
            jpeg_quality: JPEG compression quality (1-100).
            use_render_fallback: Force the render fallback path.

        Returns:
            Dictionary with image_base64, dimensions, engine, capture_method.
        """
        blender_module: Any = bpy
        scene = blender_module.context.scene
        engine = scene.render.engine

        if not use_render_fallback:
            try:
                return self._capture_gpu_offscreen(width, height, jpeg_quality, engine)
            except Exception as exc:
                logger.debug(
                    "GPU offscreen capture failed, falling back to render: %s", exc
                )

        return self._capture_render_fallback(width, height, jpeg_quality, engine)

    def set_camera_view(
        self,
        location: List[float],
        rotation_euler: List[float],
        camera_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Set the active camera's location and rotation.

        Args:
            location: Camera location [x, y, z].
            rotation_euler: Camera rotation [rx, ry, rz] in radians.
            camera_name: Target camera name. Uses active camera when None.

        Returns:
            Dictionary with camera_name, location, rotation_euler.
        """
        cam_obj = self._resolve_camera(camera_name)
        cam_obj.location = tuple(location)
        cam_obj.rotation_euler = tuple(rotation_euler)
        return {
            "camera_name": cam_obj.name,
            "location": list(cam_obj.location),
            "rotation_euler": list(cam_obj.rotation_euler),
        }

    def get_camera_info(self, camera_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get active camera properties.

        Args:
            camera_name: Target camera name. Uses active camera when None.

        Returns:
            Dictionary with lens, sensor, clip distances, camera type.
        """
        cam_obj = self._resolve_camera(camera_name)
        cam_data = cam_obj.data
        return {
            "camera_name": cam_obj.name,
            "location": list(cam_obj.location),
            "rotation_euler": list(cam_obj.rotation_euler),
            "lens": cam_data.lens,
            "sensor_width": cam_data.sensor_width,
            "clip_start": cam_data.clip_start,
            "clip_end": cam_data.clip_end,
            "camera_type": cam_data.type,
        }

    def focus_on_object(
        self,
        object_name: str,
        distance_factor: float = 2.0,
        camera_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Position the camera to look at a specific object.

        Camera is placed along the +Z +Y diagonal from the object center
        at a distance proportional to the bounding box diagonal.

        Args:
            object_name: Object to focus on.
            distance_factor: Distance multiplier from bbox diagonal.
            camera_name: Target camera name. Uses active camera when None.

        Returns:
            Dictionary with camera_name, object_name, camera_location, look_at.
        """
        obj = self._get_object_or_raise(object_name)
        cam_obj = self._resolve_camera(camera_name)

        bbox_corners = [list(obj.matrix_world @ corner) for corner in obj.bound_box]
        center = [sum(c[i] for c in bbox_corners) / len(bbox_corners) for i in range(3)]
        diag = math.sqrt(
            sum(
                (max(c[i] for c in bbox_corners) - min(c[i] for c in bbox_corners)) ** 2
                for i in range(3)
            )
        )
        offset_dist = max(diag * distance_factor, 1.0)
        direction = [0.0, -0.7071, 0.7071]
        cam_loc = [center[i] + direction[i] * offset_dist for i in range(3)]

        cam_obj.location = tuple(cam_loc)

        dx = center[0] - cam_loc[0]
        dy = center[1] - cam_loc[1]
        dz = center[2] - cam_loc[2]
        dist_xy = math.sqrt(dx * dx + dy * dy)
        pitch = math.atan2(dz, dist_xy)
        yaw = math.atan2(dx, -dy)
        cam_obj.rotation_euler = (math.pi / 2 - pitch, 0.0, yaw)

        return {
            "camera_name": cam_obj.name,
            "object_name": object_name,
            "camera_location": list(cam_obj.location),
            "look_at": center,
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
    ) -> Dict[str, Any]:
        """
        Capture viewport at multiple frames as base64-encoded JPEGs.

        Args:
            start_frame: First frame to capture.
            end_frame: Last frame to capture (inclusive).
            step: Frame step between captures.
            width: Output image width.
            height: Output image height.
            jpeg_quality: JPEG compression quality.

        Returns:
            Dictionary with frames list, frame_count, capture_method.
        """
        blender_module: Any = bpy
        scene = blender_module.context.scene
        original_frame = scene.frame_current

        frames: List[Dict[str, Any]] = []
        capture_method = "render_fallback"

        try:
            for frame_num in range(start_frame, end_frame + 1, step):
                scene.frame_set(frame_num)
                result = self.capture_viewport(
                    width=width,
                    height=height,
                    jpeg_quality=jpeg_quality,
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
            world_loc = list(obj.matrix_world.translation)
            obj.parent = None
            obj.location = tuple(world_loc)
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
        try:
            blender_module.ops.wm.open_mainfile(filepath=file_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to open .blend file '{file_path}': {exc}"
            ) from exc

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
        file_path = self._deny_outside_sandbox(file_path)
        blender_module: Any = bpy

        if file_path is not None:
            try:
                blender_module.ops.wm.save_as_mainfile(filepath=file_path)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to save .blend file to '{file_path}': {exc}"
                ) from exc
            return {"file_path": file_path}

        current = blender_module.data.filepath
        if not current:
            raise ValueError(
                "No file path provided and the file has never been saved. "
                "Pass a file_path argument."
            )
        try:
            blender_module.ops.wm.save_mainfile()
        except Exception as exc:
            raise RuntimeError(f"Failed to save .blend file: {exc}") from exc
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
        file_path = self._deny_outside_sandbox(file_path)
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
                try:
                    blender_module.ops.screen.animation_cancel(restore_frame=False)
                except Exception as exc:
                    raise RuntimeError(f"Failed to cancel animation: {exc}") from exc
            return {
                "action": "stop",
                "is_playing": False,
            }

        try:
            if action == "reverse":
                blender_module.ops.screen.animation_play(reverse=True)
            else:
                blender_module.ops.screen.animation_play()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to play animation (action={action}): {exc}"
            ) from exc

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
        """
        obj = self._get_object_or_raise(object_name)
        obj.keyframe_insert(data_path=data_path, frame=frame, index=index)
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
        """
        obj = self._get_object_or_raise(object_name)
        obj.keyframe_delete(data_path=data_path, frame=frame, index=index)
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
        Version-aware: uses fcurves on Blender 3.6-4.x and channelbag
        on Blender 5.0+.

        Args:
            object_name: Name of the object.

        Returns:
            Dict with object_name, has_animation, and channels list.
        """
        obj = self._get_object_or_raise(object_name)
        anim_data = obj.animation_data
        if anim_data is None or anim_data.action is None:
            return {
                "object_name": object_name,
                "has_animation": False,
                "channels": [],
            }

        action = anim_data.action
        channels: List[Dict[str, Any]] = []

        if self._is_blender_5_plus and hasattr(action, "channelbags"):
            # Blender 5.0+: channelbag API
            for channelbag in action.channelbags:
                for fcurve in channelbag.fcurves:
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
        else:
            # Blender 3.6 - 4.x: legacy fcurves API
            for fcurve in action.fcurves:
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
            "has_animation": True,
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
            try:
                blender_module.ops.rigidbody.world_add()
            except Exception as exc:
                raise RuntimeError(f"Failed to create rigid body world: {exc}") from exc

        blender_module.context.view_layer.objects.active = obj
        obj.select_set(True)
        try:
            blender_module.ops.rigidbody.object_add(type=body_type.upper())
        except Exception as exc:
            raise RuntimeError(
                f"Failed to add rigid body to '{object_name}': {exc}"
            ) from exc

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
            location: World-space XYZ location.
            name: Optional name for the field object.

        Returns:
            Dict with name, field_type, strength, location.
        """
        blender_module: Any = bpy
        loc = tuple(location or [0.0, 0.0, 0.0])

        try:
            blender_module.ops.object.effector_add(
                type=field_type.upper(), location=loc
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to add force field '{field_type}': {exc}"
            ) from exc
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
            "location": [float(v) for v in obj.location],
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
            try:
                blender_module.ops.rigidbody.world_add()
            except Exception as exc:
                raise RuntimeError(f"Failed to create rigid body world: {exc}") from exc

        loc = tuple(location or [0.0, 0.0, 0.0])
        try:
            blender_module.ops.object.empty_add(location=loc)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to add constraint empty at {loc}: {exc}"
            ) from exc
        empty = blender_module.context.active_object
        empty.name = f"RBC_{object1_name}_{object2_name}"

        empty.select_set(True)
        blender_module.context.view_layer.objects.active = empty
        try:
            blender_module.ops.rigidbody.constraint_add(type=constraint_type.upper())
        except Exception as exc:
            raise RuntimeError(
                f"Failed to add rigid body constraint '{constraint_type}': {exc}"
            ) from exc

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

        return {
            "object_name": object_name,
            "location": [float(v) for v in obj.location],
            "rotation_euler": [float(v) for v in obj.rotation_euler],
            "is_active": has_rb and rb.type == "ACTIVE",
            "has_rigid_body": has_rb,
            "mass": float(rb.mass) if has_rb else None,
            "collision_shape": rb.collision_shape if has_rb else None,
        }

    def get_object_trajectory(
        self,
        object_name: str,
        start_frame: int,
        end_frame: int,
        step: int = 1,
    ) -> Dict[str, Any]:
        """
        Record object position and rotation across a frame range.

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

        try:
            for frame in range(start_frame, end_frame + 1, max(step, 1)):
                scene.frame_set(frame)
                loc = [float(v) for v in obj.location]
                rot = [float(v) for v in obj.rotation_euler]
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

        override = {"scene": scene, "point_cache": pc}
        try:
            blender_module.ops.ptcache.bake(override, bake=True)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to bake simulation (frames {frame_start}-{frame_end}): {exc}"
            ) from exc

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
        override = {"scene": scene, "point_cache": pc}
        try:
            blender_module.ops.ptcache.free_bake(override)
        except Exception as exc:
            raise RuntimeError(f"Failed to free baked simulation data: {exc}") from exc

        return {}

    # -- End Phase 6 methods -------------------------------------------------

    # -- Scripting & mesh-from-data methods ----------------------------------

    def execute_script(
        self,
        script: str,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute arbitrary Python code with access to bpy.

        The script runs via ``exec()`` in a namespace that already contains
        ``bpy`` and the standard builtins.  If the script assigns a value to
        ``__result__``, its ``repr()`` is returned in *return_value*.

        Captured ``print()`` output is returned in *output* (capped at 4 KiB
        to keep response sizes agent-friendly).

        Args:
            script: Python source code to execute.
            timeout: Maximum wall-clock seconds (unused in synchronous
                Blender; reserved for future async support).

        Returns:
            Dict with output, return_value, duration_seconds, and
            optionally error.
        """
        import contextlib
        import time

        blender_module: Any = bpy

        stdout_capture = io.StringIO()
        namespace: Dict[str, Any] = {
            "bpy": blender_module,
            "__builtins__": __builtins__,
            "__result__": None,
        }

        start = time.monotonic()
        try:
            with contextlib.redirect_stdout(stdout_capture):
                exec(script, namespace)  # noqa: S102
            elapsed = time.monotonic() - start

            raw_output = stdout_capture.getvalue()
            max_output_bytes = 4096
            output = (
                raw_output[:max_output_bytes]
                if len(raw_output) > max_output_bytes
                else raw_output
            ) or None

            return_value: Optional[str] = None
            if namespace.get("__result__") is not None:
                return_value = repr(namespace["__result__"])

            return {
                "output": output,
                "return_value": return_value,
                "duration_seconds": round(elapsed, 4),
            }
        except Exception as exc:
            elapsed = time.monotonic() - start
            return {
                "output": stdout_capture.getvalue() or None,
                "return_value": None,
                "duration_seconds": round(elapsed, 4),
                "error": f"{type(exc).__name__}: {exc}",
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
        blender_module: Any = bpy
        issues: List[Dict[str, Any]] = []

        if object_names is None:
            targets = [o for o in blender_module.data.objects if o.type == "MESH"]
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

            if check_materials and hasattr(obj, "data") and obj.data:
                mat_count = len(getattr(obj.data, "materials", []))
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
        file_path = self._deny_outside_sandbox(file_path)
        blender_module: Any = bpy
        issues: Optional[List[Dict[str, Any]]] = None
        validation_passed = True

        if object_names:
            targets = [self._get_object_or_raise(n) for n in object_names]
        else:
            targets = list(blender_module.data.objects)

        if validate_before_export:
            names = [o.name for o in targets]
            result = self.validate_simready_compliance(object_names=names)
            issues = result.get("issues")
            validation_passed = result.get("compliant", True)

        # Select only the requested objects
        for obj in blender_module.data.objects:
            obj.select_set(False)
        for obj in targets:
            obj.select_set(True)

        self.export_file(
            file_path=file_path,
            file_format="USD",
            selected_only=bool(object_names),
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
        obj = blender_module.data.objects.get(object_name)
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

    @staticmethod
    def _resolve_object_source(collection_name: Optional[str]) -> Any:
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
            return blender_module.data.objects

        collection = blender_module.data.collections.get(collection_name)
        if collection is None:
            raise ValueError(f"Collection not found: {collection_name}")
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
            cam = blender_module.data.objects.get(camera_name)
            if cam is None or cam.type != "CAMERA":
                raise ValueError(f"Camera not found: {camera_name}")
            return cam

        scene_cam = blender_module.context.scene.camera
        if scene_cam is None:
            raise ValueError("No active camera in scene")
        return scene_cam

    def _capture_gpu_offscreen(
        self,
        width: int,
        height: int,
        jpeg_quality: int,
        engine: str,
    ) -> Dict[str, Any]:
        """
        Capture viewport via gpu.types.GPUOffScreen (fast path, EEVEE only).

        Args:
            width: Image width.
            height: Image height.
            jpeg_quality: JPEG compression quality.
            engine: Render engine identifier.

        Returns:
            Dictionary with image_base64, dimensions, engine, capture_method.
        """
        import gpu

        blender_module: Any = bpy
        scene = blender_module.context.scene
        cam_obj = scene.camera
        if cam_obj is None:
            raise ValueError("No active camera for GPU capture")

        offscreen = gpu.types.GPUOffScreen(width, height)
        try:
            view_matrix = cam_obj.matrix_world.inverted()
            projection_matrix = cam_obj.calc_matrix_camera(
                blender_module.context.evaluated_depsgraph_get(),
                x=width,
                y=height,
            )
            offscreen.draw_view3d(
                scene,
                blender_module.context.view_layer,
                blender_module.context.space_data,
                blender_module.context.region,
                view_matrix,
                projection_matrix,
                do_color_management=True,
            )
            buffer = offscreen.read_color(0, 0, width, height, 4, 0, "UBYTE")
        finally:
            offscreen.free()

        import numpy as np

        pixels = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 4)
        pixels = np.flipud(pixels)[:, :, :3]

        from PIL import Image as PILImage

        img = PILImage.fromarray(pixels, "RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        return {
            "image_base64": b64,
            "width": width,
            "height": height,
            "engine": engine,
            "capture_method": "gpu_offscreen",
        }

    def _capture_render_fallback(
        self,
        width: int,
        height: int,
        jpeg_quality: int,
        engine: str,
    ) -> Dict[str, Any]:
        """
        Capture viewport via bpy.ops.render.render (slow path, all engines).

        Temporarily adjusts render resolution, renders to an in-memory image,
        converts to JPEG base64, then restores original settings.

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

        orig_x = render.resolution_x
        orig_y = render.resolution_y
        orig_pct = render.resolution_percentage
        orig_fmt = render.image_settings.file_format

        try:
            render.resolution_x = width
            render.resolution_y = height
            render.resolution_percentage = 100
            render.image_settings.file_format = "JPEG"
            render.image_settings.quality = jpeg_quality

            blender_module.ops.render.render(write_still=False)

            result_image = blender_module.data.images.get("Render Result")
            if result_image is None:
                raise RuntimeError("Render Result image not available")

            pixels = list(result_image.pixels)
            import numpy as np

            arr = np.array(pixels, dtype=np.float32).reshape(height, width, 4)
            arr = np.flipud(arr)
            rgb = (arr[:, :, :3] * 255).clip(0, 255).astype(np.uint8)

            from PIL import Image as PILImage

            img = PILImage.fromarray(rgb, "RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=jpeg_quality)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")

            return {
                "image_base64": b64,
                "width": width,
                "height": height,
                "engine": engine,
                "capture_method": "render_fallback",
            }
        finally:
            render.resolution_x = orig_x
            render.resolution_y = orig_y
            render.resolution_percentage = orig_pct
            render.image_settings.file_format = orig_fmt


class BlenderRuntimeAdapter(LoggerMixin):
    """Adapter for Blender runtime operations."""

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize Blender runtime adapter.

        Args:
            settings: Configuration settings.
        """
        self.settings = settings or get_settings()
        self.logger.info("Blender runtime adapter initialized")

    @contextmanager
    def create_session(self) -> Any:
        """
        Create a Blender runtime session context manager.

        Yields:
            BlenderRuntimeSession instance.
        """
        session = BlenderRuntimeSession(self.settings)
        try:
            yield session
        finally:
            session.cleanup()

    def is_available(self) -> bool:
        """
        Check whether Blender runtime is available.

        Returns:
            True when bpy module is available.
        """
        return BLENDER_AVAILABLE and self.settings.blender.enabled

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
