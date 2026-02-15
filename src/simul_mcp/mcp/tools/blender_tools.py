"""
Blender-specific MCP tools for Simul MCP Server.

This module provides read-focused Blender runtime tools backed by the
optional bpy module.
"""

from typing import Any, Dict, List, Optional

from ...adapters import BlenderRuntimeAdapter, is_blender_available
from ...config import Settings, get_settings
from ...logging import LoggerMixin, get_logger
from ..schemas import (
    BlenderAddConstraintResponse,
    BlenderAddForceFieldResponse,
    BlenderAddModifierResponse,
    BlenderAssignMaterialResponse,
    BlenderBakeSimulationResponse,
    BlenderBoundingBoxResponse,
    BlenderBoundsCheckResponse,
    BlenderCameraInfoResponse,
    BlenderCaptureSequenceResponse,
    BlenderCaptureViewportResponse,
    BlenderClearParentResponse,
    BlenderCreateMeshFromDataResponse,
    BlenderCreateObjectResponse,
    BlenderDeleteKeyframeResponse,
    BlenderDeleteObjectResponse,
    BlenderDistanceResponse,
    BlenderExecuteScriptResponse,
    BlenderExportFileResponse,
    BlenderFileInfoResponse,
    BlenderFocusOnObjectResponse,
    BlenderFreeBakeResponse,
    BlenderGetConstraintInfoResponse,
    BlenderGetForceFieldInfoResponse,
    BlenderGetFrameResponse,
    BlenderGetKeyframesResponse,
    BlenderGetPhysicsStateResponse,
    BlenderGetTrajectoryResponse,
    BlenderImportFileResponse,
    BlenderInfoResponse,
    BlenderInsertKeyframeResponse,
    BlenderMaterialInfoResponse,
    BlenderMeshInfoResponse,
    BlenderObjectInfoResponse,
    BlenderOpenFileResponse,
    BlenderPlayAnimationResponse,
    BlenderSaveFileResponse,
    BlenderSceneObjectsRequest,
    BlenderSceneObjectsResponse,
    BlenderSceneSummaryResponse,
    BlenderSearchObjectsResponse,
    BlenderSetCameraViewResponse,
    BlenderSetFrameRangeResponse,
    BlenderSetFrameResponse,
    BlenderSetLightParamsResponse,
    BlenderSetParentResponse,
    BlenderSetTransformResponse,
    BlenderSetupRigidBodyResponse,
    BlenderViewportInfoResponse,
    ErrorResponse,
)

logger = get_logger(__name__)


class BlenderTools(LoggerMixin):
    """Tools for Blender runtime operations."""

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize Blender tools.

        Args:
            settings: Configuration settings.
        """
        self.settings = settings or get_settings()
        self.blender_adapter = (
            BlenderRuntimeAdapter(self.settings) if is_blender_available() else None
        )

    async def get_blender_info(self) -> Dict[str, Any]:
        """
        Get Blender runtime information.

        Returns:
            Runtime information dictionary or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                runtime_info = session.get_runtime_info()
                runtime_info["success"] = True
                return BlenderInfoResponse(**runtime_info).dict()

        except Exception as e:
            self.logger.error(f"Error getting Blender runtime info: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def list_blender_scene_objects(
        self,
        collection_name: Optional[str] = None,
        include_hidden: bool = False,
        max_items: int = 200,
    ) -> Dict[str, Any]:
        """
        List objects from the active Blender scene.

        Args:
            collection_name: Optional collection name filter.
            include_hidden: Include hidden objects when true.
            max_items: Maximum number of objects to return.

        Returns:
            Scene object listing dictionary or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            request = BlenderSceneObjectsRequest(
                collection_name=collection_name,
                include_hidden=include_hidden,
                max_items=max_items or self.settings.blender.max_scene_objects,
            )

            with self.blender_adapter.create_session() as session:
                objects_payload = session.list_scene_objects(
                    collection_name=request.collection_name,
                    include_hidden=request.include_hidden,
                    max_items=request.max_items,
                )
                objects_payload["success"] = True
                return BlenderSceneObjectsResponse(**objects_payload).dict()

        except Exception as e:
            self.logger.error(f"Error listing Blender scene objects: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def get_blender_object_info(self, object_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a single Blender object.

        Args:
            object_name: Name of the target object.

        Returns:
            Object info dictionary or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                payload = session.get_object_info(object_name)
                payload["success"] = True
                return BlenderObjectInfoResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error getting object info: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def get_blender_mesh_info(self, object_name: str) -> Dict[str, Any]:
        """
        Get counts-only mesh geometry information.

        Args:
            object_name: Name of the mesh object.

        Returns:
            Mesh info dictionary or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                payload = session.get_mesh_info(object_name)
                payload["success"] = True
                return BlenderMeshInfoResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error getting mesh info: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def get_blender_bounding_box(
        self, object_name: str, world_space: bool = True
    ) -> Dict[str, Any]:
        """
        Get eight bounding-box corners of a Blender object.

        Args:
            object_name: Name of the target object.
            world_space: Return corners in world space when True.

        Returns:
            Bounding box dictionary or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                payload = session.get_bounding_box(object_name, world_space)
                payload["success"] = True
                return BlenderBoundingBoxResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error getting bounding box: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def search_blender_objects(
        self,
        name_pattern: Optional[str] = None,
        object_type: Optional[str] = None,
        max_results: int = 50,
    ) -> Dict[str, Any]:
        """
        Search for Blender objects by name pattern and/or type.

        Args:
            name_pattern: Regex pattern for object name matching.
            object_type: Filter by object type (MESH, LIGHT, etc.).
            max_results: Maximum number of results.

        Returns:
            Search results dictionary or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                payload = session.search_objects(
                    name_pattern=name_pattern,
                    object_type=object_type,
                    max_results=max_results,
                )
                payload["success"] = True
                return BlenderSearchObjectsResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error searching objects: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def summarize_blender_scene(self) -> Dict[str, Any]:
        """
        Get a high-level summary of the current Blender scene.

        Returns:
            Scene summary dictionary or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                payload = session.summarize_scene()
                payload["success"] = True
                return BlenderSceneSummaryResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error summarizing scene: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def get_blender_material_info(self, material_name: str) -> Dict[str, Any]:
        """
        Get material information with bounded node tree traversal.

        Args:
            material_name: Name of the Blender material.

        Returns:
            Material info dictionary or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                payload = session.get_material_info(material_name)
                payload["success"] = True
                return BlenderMaterialInfoResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error getting material info: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def get_blender_distance_between(
        self, object_name_a: str, object_name_b: str
    ) -> Dict[str, Any]:
        """
        Compute the Euclidean distance between two Blender objects.

        Args:
            object_name_a: First object name.
            object_name_b: Second object name.

        Returns:
            Distance measurement dictionary or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                payload = session.get_distance_between(object_name_a, object_name_b)
                payload["success"] = True
                return BlenderDistanceResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error computing distance: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def check_blender_object_bounds(
        self,
        object_name: str,
        bounds_min: List[float],
        bounds_max: List[float],
    ) -> Dict[str, Any]:
        """
        Check whether a Blender object is within spatial bounds.

        Args:
            object_name: Object name to check.
            bounds_min: Minimum bounds [x, y, z].
            bounds_max: Maximum bounds [x, y, z].

        Returns:
            Bounds check dictionary or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                payload = session.is_object_within_bounds(
                    object_name, bounds_min, bounds_max
                )
                payload["success"] = True
                return BlenderBoundsCheckResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error checking object bounds: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    # -- Phase 2: Visual observation tools ----------------------------------

    async def capture_blender_viewport(
        self,
        width: int = 512,
        height: int = 512,
        jpeg_quality: int = 85,
        use_render_fallback: bool = False,
    ) -> Dict[str, Any]:
        """
        Capture the Blender viewport as a base64-encoded JPEG.

        Args:
            width: Output image width.
            height: Output image height.
            jpeg_quality: JPEG compression quality (1-100).
            use_render_fallback: Force bpy.ops.render path.

        Returns:
            Viewport capture dictionary or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                payload = session.capture_viewport(
                    width=width,
                    height=height,
                    jpeg_quality=jpeg_quality,
                    use_render_fallback=use_render_fallback,
                )
                payload["success"] = True
                return BlenderCaptureViewportResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error capturing viewport: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def set_blender_camera_view(
        self,
        location: List[float],
        rotation_euler: List[float],
        camera_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Set the active Blender camera's location and rotation.

        Args:
            location: Camera location [x, y, z].
            rotation_euler: Camera rotation [rx, ry, rz] radians.
            camera_name: Target camera name. Uses active camera when None.

        Returns:
            Camera view dictionary or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                payload = session.set_camera_view(
                    location=location,
                    rotation_euler=rotation_euler,
                    camera_name=camera_name,
                )
                payload["success"] = True
                return BlenderSetCameraViewResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error setting camera view: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def get_blender_camera_info(
        self, camera_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get active Blender camera properties.

        Args:
            camera_name: Target camera name. Uses active camera when None.

        Returns:
            Camera info dictionary or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                payload = session.get_camera_info(camera_name=camera_name)
                payload["success"] = True
                return BlenderCameraInfoResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error getting camera info: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def focus_blender_on_object(
        self,
        object_name: str,
        distance_factor: float = 2.0,
        camera_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Focus the Blender camera on a specific object.

        Args:
            object_name: Object to focus on.
            distance_factor: Distance multiplier from bbox diagonal.
            camera_name: Target camera name. Uses active camera when None.

        Returns:
            Focus result dictionary or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                payload = session.focus_on_object(
                    object_name=object_name,
                    distance_factor=distance_factor,
                    camera_name=camera_name,
                )
                payload["success"] = True
                return BlenderFocusOnObjectResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error focusing on object: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def get_blender_viewport_info(self) -> Dict[str, Any]:
        """
        Get active Blender viewport / render settings summary.

        Returns:
            Viewport info dictionary or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                payload = session.get_viewport_info()
                payload["success"] = True
                return BlenderViewportInfoResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error getting viewport info: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def capture_blender_viewport_sequence(
        self,
        start_frame: int,
        end_frame: int,
        step: int = 1,
        width: int = 512,
        height: int = 512,
        jpeg_quality: int = 85,
    ) -> Dict[str, Any]:
        """
        Capture Blender viewport at multiple frames.

        Args:
            start_frame: First frame to capture.
            end_frame: Last frame to capture.
            step: Frame step between captures.
            width: Output image width.
            height: Output image height.
            jpeg_quality: JPEG compression quality.

        Returns:
            Sequence capture dictionary or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                payload = session.capture_viewport_sequence(
                    start_frame=start_frame,
                    end_frame=end_frame,
                    step=step,
                    width=width,
                    height=height,
                    jpeg_quality=jpeg_quality,
                )
                payload["success"] = True
                return BlenderCaptureSequenceResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error capturing viewport sequence: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    # -- Phase 3: Scene manipulation tools -----------------------------------

    async def create_blender_object(
        self,
        object_type: str,
        name: Optional[str] = None,
        location: Optional[List[float]] = None,
        rotation_euler: Optional[List[float]] = None,
        scale: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new object in the Blender scene.

        Args:
            object_type: Object type (CUBE, SPHERE, POINT_LIGHT, CAMERA, etc.).
            name: Optional object name.
            location: Initial location [x, y, z].
            rotation_euler: Initial rotation [rx, ry, rz] radians.
            scale: Initial scale [sx, sy, sz].

        Returns:
            Creation result dictionary or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                payload = session.create_object(
                    object_type=object_type,
                    name=name,
                    location=location,
                    rotation_euler=rotation_euler,
                    scale=scale,
                )
                return BlenderCreateObjectResponse(success=True, **payload).dict()

        except Exception as e:
            self.logger.error(f"Error creating object: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def delete_blender_object(
        self,
        object_name: str,
    ) -> Dict[str, Any]:
        """
        Delete an object from the Blender scene.

        Args:
            object_name: Name of object to delete.

        Returns:
            Deletion result dictionary or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                payload = session.delete_object(object_name=object_name)
                return BlenderDeleteObjectResponse(success=True, **payload).dict()

        except Exception as e:
            self.logger.error(f"Error deleting object: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def set_blender_object_transform(
        self,
        object_name: str,
        location: Optional[List[float]] = None,
        rotation_euler: Optional[List[float]] = None,
        scale: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Set an object's transform (location, rotation, scale).

        Args:
            object_name: Object to transform.
            location: New location [x, y, z].
            rotation_euler: New rotation [rx, ry, rz] radians.
            scale: New scale [sx, sy, sz].

        Returns:
            Transform result dictionary or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                payload = session.set_object_transform(
                    object_name=object_name,
                    location=location,
                    rotation_euler=rotation_euler,
                    scale=scale,
                )
                return BlenderSetTransformResponse(success=True, **payload).dict()

        except Exception as e:
            self.logger.error(f"Error setting transform: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def set_blender_object_parent(
        self,
        child_name: str,
        parent_name: str,
    ) -> Dict[str, Any]:
        """
        Parent one object to another.

        Args:
            child_name: Object to parent.
            parent_name: Object to be the parent.

        Returns:
            Parenting result dictionary or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                payload = session.set_object_parent(
                    child_name=child_name,
                    parent_name=parent_name,
                )
                return BlenderSetParentResponse(success=True, **payload).dict()

        except Exception as e:
            self.logger.error(f"Error setting parent: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def clear_blender_object_parent(
        self,
        object_name: str,
        keep_transform: bool = True,
    ) -> Dict[str, Any]:
        """
        Unparent an object.

        Args:
            object_name: Object to unparent.
            keep_transform: Preserve world transform after unparenting.

        Returns:
            Unparenting result dictionary or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                payload = session.clear_object_parent(
                    object_name=object_name,
                    keep_transform=keep_transform,
                )
                return BlenderClearParentResponse(success=True, **payload).dict()

        except Exception as e:
            self.logger.error(f"Error clearing parent: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def assign_blender_material(
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
            metallic: Metallic value (0-1).
            roughness: Roughness value (0-1).

        Returns:
            Material assignment result dictionary or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                payload = session.assign_material(
                    object_name=object_name,
                    material_name=material_name,
                    base_color=base_color,
                    metallic=metallic,
                    roughness=roughness,
                )
                return BlenderAssignMaterialResponse(success=True, **payload).dict()

        except Exception as e:
            self.logger.error(f"Error assigning material: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def add_blender_modifier(
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
            params: Modifier-specific parameters.

        Returns:
            Modifier result dictionary or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                payload = session.add_modifier(
                    object_name=object_name,
                    modifier_type=modifier_type,
                    modifier_name=modifier_name,
                    params=params or {},
                )
                return BlenderAddModifierResponse(success=True, **payload).dict()

        except Exception as e:
            self.logger.error(f"Error adding modifier: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def set_blender_light_params(
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
        Set properties on a Blender light object.

        Args:
            light_name: Light object name.
            energy: Light energy/power.
            color: Light color RGB [r, g, b].
            use_shadow: Enable shadow casting.
            spot_size: Spot cone angle in radians (SPOT only).
            spot_blend: Spot edge blend factor (SPOT only).
            shadow_soft_size: Shadow softness / radius.

        Returns:
            Light params result dictionary or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                payload = session.set_light_params(
                    light_name=light_name,
                    energy=energy,
                    color=color,
                    use_shadow=use_shadow,
                    spot_size=spot_size,
                    spot_blend=spot_blend,
                    shadow_soft_size=shadow_soft_size,
                )
                return BlenderSetLightParamsResponse(success=True, **payload).dict()

        except Exception as e:
            self.logger.error(f"Error setting light params: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    # -- Phase 4: File I/O tools ---------------------------------------------

    async def open_blender_file(self, file_path: str) -> Dict[str, Any]:
        """
        Open a .blend file, replacing the current scene.

        Args:
            file_path: Absolute path to the .blend file.

        Returns:
            Open result dictionary or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                payload = session.open_blend_file(file_path=file_path)
                return BlenderOpenFileResponse(success=True, **payload).dict()

        except Exception as e:
            self.logger.error(f"Error opening blend file: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def save_blender_file(
        self, file_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Save the current scene to a .blend file.

        Args:
            file_path: Path to save to. None saves in place.

        Returns:
            Save result dictionary or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                payload = session.save_blend_file(file_path=file_path)
                return BlenderSaveFileResponse(success=True, **payload).dict()

        except Exception as e:
            self.logger.error(f"Error saving blend file: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def import_blender_file(
        self, file_path: str, file_format: str
    ) -> Dict[str, Any]:
        """
        Import a file into the Blender scene.

        Args:
            file_path: Absolute path to the file to import.
            file_format: File format (OBJ, FBX, GLTF, USD, STL, PLY).

        Returns:
            Import result dictionary or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                payload = session.import_file(
                    file_path=file_path, file_format=file_format
                )
                return BlenderImportFileResponse(success=True, **payload).dict()

        except Exception as e:
            self.logger.error(f"Error importing file: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def export_blender_file(
        self,
        file_path: str,
        file_format: str,
        selected_only: bool = False,
    ) -> Dict[str, Any]:
        """
        Export scene objects to a file.

        Args:
            file_path: Absolute path to export to.
            file_format: File format (OBJ, FBX, GLTF, USD, STL, PLY).
            selected_only: If True, export only selected objects.

        Returns:
            Export result dictionary or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                payload = session.export_file(
                    file_path=file_path,
                    file_format=file_format,
                    selected_only=selected_only,
                )
                return BlenderExportFileResponse(success=True, **payload).dict()

        except Exception as e:
            self.logger.error(f"Error exporting file: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def get_blender_file_info(self) -> Dict[str, Any]:
        """
        Get information about the current .blend file.

        Returns:
            File info dictionary or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                payload = session.get_file_info()
                return BlenderFileInfoResponse(success=True, **payload).dict()

        except Exception as e:
            self.logger.error(f"Error getting file info: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    # -- Phase 5: Animation & Timeline tools ---------------------------------

    async def set_blender_frame(self, frame: int) -> Dict[str, Any]:
        """
        Set the current animation frame in Blender.

        Args:
            frame: Frame number to set.

        Returns:
            Frame confirmation or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                payload = session.set_frame(frame)
                return BlenderSetFrameResponse(success=True, **payload).dict()

        except Exception as e:
            self.logger.error(f"Error setting frame: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def get_blender_frame(self) -> Dict[str, Any]:
        """
        Get the current frame and animation range.

        Returns:
            Frame info dictionary or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                payload = session.get_frame()
                return BlenderGetFrameResponse(success=True, **payload).dict()

        except Exception as e:
            self.logger.error(f"Error getting frame: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def set_blender_frame_range(
        self, frame_start: int, frame_end: int
    ) -> Dict[str, Any]:
        """
        Set the animation frame range.

        Args:
            frame_start: Start frame of the animation range.
            frame_end: End frame of the animation range.

        Returns:
            Frame range confirmation or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                payload = session.set_frame_range(frame_start, frame_end)
                return BlenderSetFrameRangeResponse(success=True, **payload).dict()

        except Exception as e:
            self.logger.error(f"Error setting frame range: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def play_blender_animation(self, action: str = "play") -> Dict[str, Any]:
        """
        Control animation playback (play, stop, reverse).

        Args:
            action: Playback action — 'play', 'stop', or 'reverse'.

        Returns:
            Playback state or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                payload = session.play_animation(action)
                return BlenderPlayAnimationResponse(success=True, **payload).dict()

        except Exception as e:
            self.logger.error(f"Error controlling animation: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def insert_blender_keyframe(
        self,
        object_name: str,
        data_path: str,
        frame: int,
        index: int = -1,
    ) -> Dict[str, Any]:
        """
        Insert a keyframe on an object property.

        Args:
            object_name: Name of the Blender object.
            data_path: Property data path (e.g. 'location').
            frame: Frame number for the keyframe.
            index: Array index (-1 for all channels).

        Returns:
            Keyframe confirmation or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                payload = session.insert_keyframe(object_name, data_path, frame, index)
                return BlenderInsertKeyframeResponse(success=True, **payload).dict()

        except Exception as e:
            self.logger.error(f"Error inserting keyframe: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def delete_blender_keyframe(
        self,
        object_name: str,
        data_path: str,
        frame: int,
        index: int = -1,
    ) -> Dict[str, Any]:
        """
        Delete a keyframe from an object property.

        Args:
            object_name: Name of the Blender object.
            data_path: Property data path (e.g. 'location').
            frame: Frame number of the keyframe to delete.
            index: Array index (-1 for all channels).

        Returns:
            Deletion confirmation or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                payload = session.delete_keyframe(object_name, data_path, frame, index)
                return BlenderDeleteKeyframeResponse(success=True, **payload).dict()

        except Exception as e:
            self.logger.error(f"Error deleting keyframe: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def get_blender_keyframes(self, object_name: str) -> Dict[str, Any]:
        """
        Get keyframe summary for an object.

        Args:
            object_name: Name of the Blender object.

        Returns:
            Keyframe summary or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                payload = session.get_keyframes(object_name)
                return BlenderGetKeyframesResponse(success=True, **payload).dict()

        except Exception as e:
            self.logger.error(f"Error getting keyframes: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    # -- Phase 6: Physics & Simulation tool wrappers -------------------------

    async def setup_blender_rigid_body(
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
            body_type: ACTIVE or PASSIVE.
            mass: Mass in kilograms.
            friction: Surface friction coefficient.
            restitution: Bounciness (0-1).
            collision_shape: Collision shape type.
            linear_damping: Linear damping factor.
            angular_damping: Angular damping factor.

        Returns:
            Rigid body setup result or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                payload = session.setup_rigid_body(
                    object_name,
                    body_type,
                    mass,
                    friction,
                    restitution,
                    collision_shape,
                    linear_damping,
                    angular_damping,
                )
                return BlenderSetupRigidBodyResponse(success=True, **payload).dict()

        except Exception as e:
            self.logger.error(f"Error setting up rigid body: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def add_blender_force_field(
        self,
        field_type: str,
        strength: float = 1.0,
        location: Optional[List[float]] = None,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Add a force field to the Blender scene.

        Args:
            field_type: Force field type (FORCE, WIND, etc.).
            strength: Field strength.
            location: World-space XYZ location.
            name: Optional name for the field.

        Returns:
            Force field creation result or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                payload = session.add_force_field(
                    field_type,
                    strength,
                    location,
                    name,
                )
                return BlenderAddForceFieldResponse(success=True, **payload).dict()

        except Exception as e:
            self.logger.error(f"Error adding force field: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def get_blender_force_field_info(self, object_name: str) -> Dict[str, Any]:
        """
        Get force field information from a Blender object.

        Args:
            object_name: Name of the force field object.

        Returns:
            Force field info or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                payload = session.get_force_field_info(object_name)
                return BlenderGetForceFieldInfoResponse(success=True, **payload).dict()

        except Exception as e:
            self.logger.error(f"Error getting force field info: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def add_blender_rigid_body_constraint(
        self,
        constraint_type: str,
        object1_name: str,
        object2_name: str,
        location: Optional[List[float]] = None,
        disable_collisions: bool = True,
    ) -> Dict[str, Any]:
        """
        Add a rigid body constraint between two Blender objects.

        Args:
            constraint_type: Constraint type (FIXED, POINT, HINGE, etc.).
            object1_name: First constrained object.
            object2_name: Second constrained object.
            location: Optional world-space location.
            disable_collisions: Disable collisions between objects.

        Returns:
            Constraint creation result or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                payload = session.add_rigid_body_constraint(
                    constraint_type,
                    object1_name,
                    object2_name,
                    location,
                    disable_collisions,
                )
                return BlenderAddConstraintResponse(success=True, **payload).dict()

        except Exception as e:
            self.logger.error(f"Error adding rigid body constraint: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def get_blender_constraint_info(self, object_name: str) -> Dict[str, Any]:
        """
        Get rigid body constraint details from a Blender object.

        Args:
            object_name: Name of the constraint empty object.

        Returns:
            Constraint info or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                payload = session.get_constraint_info(object_name)
                return BlenderGetConstraintInfoResponse(success=True, **payload).dict()

        except Exception as e:
            self.logger.error(f"Error getting constraint info: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def get_blender_physics_state(self, object_name: str) -> Dict[str, Any]:
        """
        Get physics state of a Blender object.

        Args:
            object_name: Name of the object.

        Returns:
            Physics state or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                payload = session.get_physics_state(object_name)
                return BlenderGetPhysicsStateResponse(success=True, **payload).dict()

        except Exception as e:
            self.logger.error(f"Error getting physics state: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def get_blender_object_trajectory(
        self,
        object_name: str,
        start_frame: int,
        end_frame: int,
        step: int = 1,
    ) -> Dict[str, Any]:
        """
        Record object trajectory across a frame range.

        Args:
            object_name: Name of the object to track.
            start_frame: First frame of trajectory.
            end_frame: Last frame of trajectory.
            step: Frame step size.

        Returns:
            Trajectory data or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                payload = session.get_object_trajectory(
                    object_name,
                    start_frame,
                    end_frame,
                    step,
                )
                return BlenderGetTrajectoryResponse(success=True, **payload).dict()

        except Exception as e:
            self.logger.error(f"Error getting object trajectory: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def bake_blender_simulation(
        self,
        frame_start: int,
        frame_end: int,
    ) -> Dict[str, Any]:
        """
        Bake physics simulation in the Blender scene.

        Args:
            frame_start: First frame to bake.
            frame_end: Last frame to bake.

        Returns:
            Bake result or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                payload = session.bake_simulation(frame_start, frame_end)
                return BlenderBakeSimulationResponse(success=True, **payload).dict()

        except Exception as e:
            self.logger.error(f"Error baking simulation: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def free_blender_bake(self) -> Dict[str, Any]:
        """
        Free all baked physics simulation data.

        Returns:
            Free bake result or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                session.free_bake()
                return BlenderFreeBakeResponse(success=True).dict()

        except Exception as e:
            self.logger.error(f"Error freeing bake: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    # -- Scripting & mesh-from-data tool wrappers ----------------------------

    async def execute_blender_script(
        self,
        script: str,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute arbitrary Python code inside Blender with bpy access.

        Args:
            script: Python source code to execute.
            timeout: Maximum execution seconds (reserved for future use).

        Returns:
            Execution result dictionary or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                payload = session.execute_script(script, timeout)
                has_error = payload.get("error") is not None
                payload["success"] = not has_error
                return BlenderExecuteScriptResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error executing Blender script: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def create_blender_mesh_from_data(
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

        Args:
            name: Name for the new mesh object.
            vertices: List of [x, y, z] vertex positions.
            edges: List of [v0, v1] edge pairs.
            faces: List of vertex-index lists per face.
            location: World-space location [x, y, z].
            collection_name: Target collection name.

        Returns:
            Mesh creation result dictionary or error response.
        """
        try:
            if not self.blender_adapter or not self.blender_adapter.is_available():
                return ErrorResponse(
                    error="Blender runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.blender_adapter.create_session() as session:
                payload = session.create_mesh_from_data(
                    name=name,
                    vertices=vertices,
                    edges=edges or [],
                    faces=faces or [],
                    location=location,
                    collection_name=collection_name,
                )
                return BlenderCreateMeshFromDataResponse(
                    success=True, **payload
                ).dict()

        except Exception as e:
            self.logger.error(f"Error creating mesh from data: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()
