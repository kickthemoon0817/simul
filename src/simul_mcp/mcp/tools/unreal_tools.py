"""
Unreal Engine MCP tools for Simul MCP Server.

This module provides read-focused Unreal Engine tools backed by the
Remote Control API via aiohttp.
"""

from typing import Any, Dict, List, Optional

from ...adapters import UnrealRuntimeAdapter, is_unreal_available
from ...config import Settings, get_settings
from ...logging import LoggerMixin, get_logger
from ..schemas import (
    ErrorResponse,
    UnrealAddComponentResponse,
    UnrealApplyForceResponse,
    UnrealAssignMaterialResponse,
    UnrealCallActorFunctionResponse,
    UnrealCaptureViewportResponse,
    UnrealControlSimulationResponse,
    UnrealCreateMaterialInstanceResponse,
    UnrealDeleteActorResponse,
    UnrealDescribeObjectResponse,
    UnrealEnablePhysicsResponse,
    UnrealFocusActorResponse,
    UnrealGetActorInfoResponse,
    UnrealGetMaterialInfoResponse,
    UnrealGetSimulationStatusResponse,
    UnrealGetThumbnailResponse,
    UnrealListActorsResponse,
    UnrealSceneSummaryResponse,
    UnrealSearchAssetsResponse,
    UnrealSetActorParentResponse,
    UnrealSetActorPropertyResponse,
    UnrealSetActorTransformResponse,
    UnrealSetActorVisibilityResponse,
    UnrealSetCameraViewResponse,
    UnrealSetCollisionResponse,
    UnrealSetLightParamsResponse,
    UnrealSetMaterialParamsResponse,
    UnrealSetPhysicsParamsResponse,
    UnrealSetRenderSettingsResponse,
    UnrealSpawnActorResponse,
    UnrealViewportInfoResponse,
    # Phase 6: USD/SimReady
    UnrealImportUsdResponse,
    UnrealExportUsdResponse,
    UnrealConvertToSimreadyResponse,
    UnrealValidateSimreadyResponse,
    UnrealGetInterchangeInfoResponse,
    # Phase 7: Agent Tools
    UnrealBatchOperationsResponse,
    UnrealQuerySceneGraphResponse,
    UnrealAnalyzeSceneForRoboticsResponse,
    UnrealGenerateProceduralSceneResponse,
    UnrealGetActorBySemanticLabelResponse,
    # Phase 8: Geometry & Modeling
    UnrealGenerateMeshPrimitiveResponse,
    UnrealApplyMeshBooleanResponse,
    UnrealComputeConvexHullResponse,
    UnrealDecomposeConvexHullResponse,
    UnrealEditMeshTopologyResponse,
    UnrealSubdivideMeshResponse,
    UnrealSimplifyMeshResponse,
    UnrealCutMeshPlaneResponse,
    UnrealValidateMeshResponse,
    UnrealConvertMeshFormatResponse,
    UnrealRemeshMeshResponse,
    UnrealComputeMeshUvResponse,
)

logger = get_logger(__name__)


class UnrealTools(LoggerMixin):
    """Tools for Unreal Engine runtime operations via Remote Control API."""

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize Unreal tools.

        Args:
            settings: Configuration settings.
        """
        self.settings = settings or get_settings()
        self.unreal_adapter = (
            UnrealRuntimeAdapter(self.settings) if is_unreal_available() else None
        )

    async def list_unreal_actors(
        self,
        class_filter: Optional[str] = None,
        tag_filter: Optional[str] = None,
        max_results: int = 200,
    ) -> Dict[str, Any]:
        """
        List actors in the current Unreal level.

        Args:
            class_filter: Optional UClass name filter.
            tag_filter: Optional tag filter.
            max_results: Maximum number of actors to return.

        Returns:
            Actor listing dictionary or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.list_actors(
                    class_filter=class_filter,
                    tag_filter=tag_filter,
                    max_results=max_results,
                )
                payload["success"] = True
                return UnrealListActorsResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error listing Unreal actors: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def get_unreal_actor_info(
        self, actor_path: str
    ) -> Dict[str, Any]:
        """
        Get detailed information about a specific actor.

        Args:
            actor_path: Full object path of the actor.

        Returns:
            Actor info dictionary or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.get_actor_info(actor_path)
                payload["success"] = True
                return UnrealGetActorInfoResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error getting Unreal actor info: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def search_unreal_assets(
        self,
        query: str = "",
        class_names: Optional[List[str]] = None,
        package_paths: Optional[List[str]] = None,
        max_results: int = 100,
    ) -> Dict[str, Any]:
        """
        Search the Unreal Asset Registry.

        Args:
            query: Search query string.
            class_names: Filter by UClass names.
            package_paths: Package path prefixes to search within.
            max_results: Maximum number of results.

        Returns:
            Asset search results or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.search_assets(
                    query=query,
                    class_names=class_names,
                    package_paths=package_paths,
                    max_results=max_results,
                )
                payload["success"] = True
                return UnrealSearchAssetsResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error searching Unreal assets: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def describe_unreal_object(
        self, object_path: str
    ) -> Dict[str, Any]:
        """
        Get full property/function metadata for a UObject.

        Args:
            object_path: Full object path.

        Returns:
            Object description or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.describe_object(object_path)
                payload["success"] = True
                return UnrealDescribeObjectResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error describing Unreal object: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def get_unreal_actor_thumbnail(
        self,
        asset_path: str,
        width: int = 256,
        height: int = 256,
    ) -> Dict[str, Any]:
        """
        Get a thumbnail image for an asset.

        Args:
            asset_path: Full asset path.
            width: Thumbnail width in pixels.
            height: Thumbnail height in pixels.

        Returns:
            Thumbnail data or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.get_actor_thumbnail(
                    asset_path=asset_path, width=width, height=height
                )
                payload["success"] = True
                return UnrealGetThumbnailResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error getting Unreal thumbnail: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def summarize_unreal_scene(self) -> Dict[str, Any]:
        """
        Generate an LLM-friendly scene digest.

        Returns:
            Scene summary or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.summarize_scene()
                payload["success"] = True
                return UnrealSceneSummaryResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error summarizing Unreal scene: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    # ------------------------------------------------------------------
    # Phase 2 — Viewport & Visual Observation
    # ------------------------------------------------------------------

    async def capture_unreal_viewport(
        self,
        resolution_x: int = 1920,
        resolution_y: int = 1080,
        format: str = "png",
    ) -> Dict[str, Any]:
        """
        Capture a viewport screenshot.

        Args:
            resolution_x: Capture width in pixels.
            resolution_y: Capture height in pixels.
            format: Image format (png or jpeg).

        Returns:
            Capture result or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.capture_viewport(
                    resolution_x=resolution_x,
                    resolution_y=resolution_y,
                    format=format,
                )
                payload["success"] = True
                return UnrealCaptureViewportResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error capturing Unreal viewport: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def get_unreal_viewport_info(self) -> Dict[str, Any]:
        """
        Get active viewport camera and render information.

        Returns:
            Viewport info or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.get_viewport_info()
                payload["success"] = True
                return UnrealViewportInfoResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error getting Unreal viewport info: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def set_unreal_camera_view(
        self,
        location_x: float = 0.0,
        location_y: float = 0.0,
        location_z: float = 0.0,
        rotation_pitch: float = 0.0,
        rotation_yaw: float = 0.0,
        rotation_roll: float = 0.0,
        fov: float = 90.0,
    ) -> Dict[str, Any]:
        """
        Set the editor viewport camera.

        Args:
            location_x: Camera X position in cm.
            location_y: Camera Y position in cm.
            location_z: Camera Z position in cm.
            rotation_pitch: Camera pitch in degrees.
            rotation_yaw: Camera yaw in degrees.
            rotation_roll: Camera roll in degrees.
            fov: Field of view in degrees.

        Returns:
            Applied camera state or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.set_camera_view(
                    location=(location_x, location_y, location_z),
                    rotation=(rotation_pitch, rotation_yaw, rotation_roll),
                    fov=fov,
                )
                payload["success"] = True
                return UnrealSetCameraViewResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error setting Unreal camera view: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def focus_unreal_on_actor(
        self,
        actor_path: str,
        distance: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Focus the viewport camera on a specific actor.

        Args:
            actor_path: Full actor path to focus on.
            distance: Camera distance from actor (0 = auto-fit).

        Returns:
            Focus result or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.focus_on_actor(
                    actor_path=actor_path, distance=distance
                )
                payload["success"] = True
                return UnrealFocusActorResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error focusing on Unreal actor: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    # ------------------------------------------------------------------
    # Phase 3 — Scene Manipulation
    # ------------------------------------------------------------------

    async def spawn_unreal_actor(
        self,
        asset_path: str,
        location_x: float = 0.0,
        location_y: float = 0.0,
        location_z: float = 0.0,
        rotation_pitch: float = 0.0,
        rotation_yaw: float = 0.0,
        rotation_roll: float = 0.0,
        label: str = "",
    ) -> Dict[str, Any]:
        """
        Spawn an actor from a class or asset path.

        Args:
            asset_path: Asset or class path to spawn from.
            location_x: Spawn X position in cm.
            location_y: Spawn Y position in cm.
            location_z: Spawn Z position in cm.
            rotation_pitch: Spawn pitch in degrees.
            rotation_yaw: Spawn yaw in degrees.
            rotation_roll: Spawn roll in degrees.
            label: Optional actor label.

        Returns:
            Spawn result or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.spawn_actor(
                    asset_path=asset_path,
                    location=(location_x, location_y, location_z),
                    rotation=(rotation_pitch, rotation_yaw, rotation_roll),
                    label=label or None,
                )
                payload["success"] = True
                return UnrealSpawnActorResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error spawning Unreal actor: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def delete_unreal_actor(self, actor_path: str) -> Dict[str, Any]:
        """
        Delete an actor from the level.

        Args:
            actor_path: Full actor path to delete.

        Returns:
            Deletion result or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.delete_actor(actor_path=actor_path)
                payload["success"] = True
                return UnrealDeleteActorResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error deleting Unreal actor: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def set_unreal_actor_transform(
        self,
        actor_path: str,
        location_x: float = 0.0,
        location_y: float = 0.0,
        location_z: float = 0.0,
        rotation_pitch: float = 0.0,
        rotation_yaw: float = 0.0,
        rotation_roll: float = 0.0,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        scale_z: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Set an actor's transform.

        Args:
            actor_path: Full actor path.
            location_x: X position in cm.
            location_y: Y position in cm.
            location_z: Z position in cm.
            rotation_pitch: Pitch in degrees.
            rotation_yaw: Yaw in degrees.
            rotation_roll: Roll in degrees.
            scale_x: X scale.
            scale_y: Y scale.
            scale_z: Z scale.

        Returns:
            Transform result or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.set_actor_transform(
                    actor_path=actor_path,
                    location=(location_x, location_y, location_z),
                    rotation=(rotation_pitch, rotation_yaw, rotation_roll),
                    scale=(scale_x, scale_y, scale_z),
                )
                payload["success"] = True
                return UnrealSetActorTransformResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error setting Unreal actor transform: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def set_unreal_actor_property(
        self,
        actor_path: str,
        property_name: str,
        property_value: str,
        generate_transaction: bool = True,
    ) -> Dict[str, Any]:
        """
        Set a property on an actor.

        Args:
            actor_path: Full actor path.
            property_name: Property name.
            property_value: Value as JSON string.
            generate_transaction: Generate undo transaction.

        Returns:
            Property set result or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.set_actor_property(
                    actor_path=actor_path,
                    property_name=property_name,
                    property_value=property_value,
                    generate_transaction=generate_transaction,
                )
                payload["success"] = True
                return UnrealSetActorPropertyResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error setting Unreal actor property: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def call_unreal_actor_function(
        self,
        actor_path: str,
        function_name: str,
        parameters: str = "",
    ) -> Dict[str, Any]:
        """
        Call a BlueprintCallable function on an actor.

        Args:
            actor_path: Full actor path.
            function_name: Function name to call.
            parameters: Parameters as JSON string.

        Returns:
            Function call result or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.call_actor_function(
                    actor_path=actor_path,
                    function_name=function_name,
                    parameters=parameters or None,
                )
                payload["success"] = True
                return UnrealCallActorFunctionResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error calling Unreal actor function: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def set_unreal_actor_parent(
        self,
        actor_path: str,
        parent_path: str = "",
    ) -> Dict[str, Any]:
        """
        Attach an actor to a parent or detach it.

        Args:
            actor_path: Child actor path.
            parent_path: Parent actor path (empty to detach).

        Returns:
            Parent set result or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.set_actor_parent(
                    actor_path=actor_path,
                    parent_path=parent_path or None,
                )
                payload["success"] = True
                return UnrealSetActorParentResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error setting Unreal actor parent: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def add_unreal_component(
        self,
        actor_path: str,
        component_class: str,
        component_name: str = "",
    ) -> Dict[str, Any]:
        """
        Add a component to an actor.

        Args:
            actor_path: Full actor path.
            component_class: Component class name.
            component_name: Optional name for the component.

        Returns:
            Component add result or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.add_component(
                    actor_path=actor_path,
                    component_class=component_class,
                    component_name=component_name or None,
                )
                payload["success"] = True
                return UnrealAddComponentResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error adding Unreal component: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def set_unreal_actor_visibility(
        self,
        actor_path: str,
        visible: bool = True,
        propagate: bool = True,
    ) -> Dict[str, Any]:
        """
        Set actor visibility.

        Args:
            actor_path: Full actor path.
            visible: Whether visible.
            propagate: Propagate to children.

        Returns:
            Visibility result or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.set_actor_visibility(
                    actor_path=actor_path,
                    visible=visible,
                    propagate=propagate,
                )
                payload["success"] = True
                return UnrealSetActorVisibilityResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error setting Unreal actor visibility: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    # ------------------------------------------------------------------
    # Phase 4 — Materials, Lighting & Rendering
    # ------------------------------------------------------------------

    async def get_unreal_material_info(
        self,
        material_path: str,
    ) -> Dict[str, Any]:
        """
        Get material instance parameters and metadata.

        Args:
            material_path: Material or MIC asset path.

        Returns:
            Material info or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.get_material_info(
                    material_path=material_path,
                )
                payload["success"] = True
                return UnrealGetMaterialInfoResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error getting Unreal material info: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def set_unreal_material_params(
        self,
        material_path: str,
        scalar_params_json: str = "",
        vector_params_json: str = "",
        texture_params_json: str = "",
    ) -> Dict[str, Any]:
        """
        Set parameters on a Material Instance Constant.

        Args:
            material_path: Material Instance path.
            scalar_params_json: JSON string of scalar params.
            vector_params_json: JSON string of vector params.
            texture_params_json: JSON string of texture params.

        Returns:
            Params set result or error response.
        """
        import json as json_lib

        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            scalar_params = (
                json_lib.loads(scalar_params_json) if scalar_params_json else None
            )
            vector_params = (
                json_lib.loads(vector_params_json) if vector_params_json else None
            )
            texture_params = (
                json_lib.loads(texture_params_json) if texture_params_json else None
            )

            with self.unreal_adapter.create_session() as session:
                payload = await session.set_material_params(
                    material_path=material_path,
                    scalar_params=scalar_params,
                    vector_params=vector_params,
                    texture_params=texture_params,
                )
                payload["success"] = True
                return UnrealSetMaterialParamsResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error setting Unreal material params: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def create_unreal_material_instance(
        self,
        parent_path: str,
        instance_name: str,
        save_path: str = "",
    ) -> Dict[str, Any]:
        """
        Create a Material Instance Constant from a parent material.

        Args:
            parent_path: Parent material asset path.
            instance_name: New instance name.
            save_path: Content-relative save path.

        Returns:
            Created MIC info or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.create_material_instance(
                    parent_path=parent_path,
                    instance_name=instance_name,
                    save_path=save_path,
                )
                payload["success"] = True
                return UnrealCreateMaterialInstanceResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error creating Unreal material instance: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def assign_unreal_material(
        self,
        actor_path: str,
        material_path: str,
        slot_index: int = 0,
    ) -> Dict[str, Any]:
        """
        Assign a material to a mesh component.

        Args:
            actor_path: Target actor path.
            material_path: Material asset path.
            slot_index: Material slot index.

        Returns:
            Assignment result or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.assign_material(
                    actor_path=actor_path,
                    material_path=material_path,
                    slot_index=slot_index,
                )
                payload["success"] = True
                return UnrealAssignMaterialResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error assigning Unreal material: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def set_unreal_light_params(
        self,
        actor_path: str,
        intensity: Optional[float] = None,
        color_r: Optional[float] = None,
        color_g: Optional[float] = None,
        color_b: Optional[float] = None,
        temperature: Optional[float] = None,
        use_temperature: Optional[bool] = None,
        attenuation_radius: Optional[float] = None,
        cast_shadows: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Set light component parameters.

        Args:
            actor_path: Light actor path.
            intensity: Light intensity.
            color_r: Red (0-1).
            color_g: Green (0-1).
            color_b: Blue (0-1).
            temperature: Color temperature in Kelvin.
            use_temperature: Use temperature mode.
            attenuation_radius: Attenuation radius (cm).
            cast_shadows: Enable shadows.

        Returns:
            Light params result or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.set_light_params(
                    actor_path=actor_path,
                    intensity=intensity,
                    color_r=color_r,
                    color_g=color_g,
                    color_b=color_b,
                    temperature=temperature,
                    use_temperature=use_temperature,
                    attenuation_radius=attenuation_radius,
                    cast_shadows=cast_shadows,
                )
                payload["success"] = True
                return UnrealSetLightParamsResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error setting Unreal light params: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def set_unreal_render_settings(
        self,
        setting_name: str,
        setting_value: str,
    ) -> Dict[str, Any]:
        """
        Set rendering or post-process settings.

        Args:
            setting_name: Console variable or render setting name.
            setting_value: Value as string.

        Returns:
            Render settings result or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.set_render_settings(
                    setting_name=setting_name,
                    setting_value=setting_value,
                )
                payload["success"] = True
                return UnrealSetRenderSettingsResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error setting Unreal render settings: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    # ------------------------------------------------------------------
    # Phase 5: Physics & Simulation Control
    # ------------------------------------------------------------------

    async def control_unreal_simulation(
        self,
        action: str,
    ) -> Dict[str, Any]:
        """
        Control Play-In-Editor (PIE) session.

        Args:
            action: One of start, stop, pause, resume, step.

        Returns:
            Simulation control result or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.control_simulation(action=action)
                payload["success"] = True
                return UnrealControlSimulationResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error controlling Unreal simulation: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def get_unreal_simulation_status(self) -> Dict[str, Any]:
        """
        Get current PIE simulation status.

        Returns:
            Simulation status or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.get_simulation_status()
                payload["success"] = True
                return UnrealGetSimulationStatusResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error getting Unreal simulation status: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def enable_unreal_physics(
        self,
        actor_path: str,
        enable: bool = True,
        simulate_physics: bool = True,
    ) -> Dict[str, Any]:
        """
        Enable or disable physics simulation on an actor.

        Args:
            actor_path: Full actor object path.
            enable: Enable or disable physics.
            simulate_physics: Whether the body actively simulates.

        Returns:
            Physics enable result or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.enable_physics(
                    actor_path=actor_path,
                    enable=enable,
                    simulate_physics=simulate_physics,
                )
                payload["success"] = True
                return UnrealEnablePhysicsResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error enabling Unreal physics: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def set_unreal_collision(
        self,
        actor_path: str,
        collision_preset: str = "",
        collision_enabled: bool = True,
    ) -> Dict[str, Any]:
        """
        Set collision configuration on an actor.

        Args:
            actor_path: Full actor object path.
            collision_preset: Named collision preset.
            collision_enabled: Whether collision is enabled.

        Returns:
            Collision config result or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.set_collision(
                    actor_path=actor_path,
                    collision_preset=collision_preset,
                    collision_enabled=collision_enabled,
                )
                payload["success"] = True
                return UnrealSetCollisionResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error setting Unreal collision: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def apply_unreal_force(
        self,
        actor_path: str,
        force_x: float = 0.0,
        force_y: float = 0.0,
        force_z: float = 0.0,
        is_impulse: bool = False,
        location_x: Optional[float] = None,
        location_y: Optional[float] = None,
        location_z: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Apply force or impulse to an actor's physics body.

        Args:
            actor_path: Full actor object path.
            force_x: Force X component.
            force_y: Force Y component.
            force_z: Force Z component.
            is_impulse: True for impulse, False for continuous force.
            location_x: Application point X (None = center of mass).
            location_y: Application point Y.
            location_z: Application point Z.

        Returns:
            Force application result or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.apply_force(
                    actor_path=actor_path,
                    force_x=force_x,
                    force_y=force_y,
                    force_z=force_z,
                    is_impulse=is_impulse,
                    location_x=location_x,
                    location_y=location_y,
                    location_z=location_z,
                )
                payload["success"] = True
                return UnrealApplyForceResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error applying Unreal force: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def set_unreal_physics_params(
        self,
        actor_path: str,
        mass: Optional[float] = None,
        linear_damping: Optional[float] = None,
        angular_damping: Optional[float] = None,
        enable_gravity: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Set physics body parameters on an actor.

        Args:
            actor_path: Full actor object path.
            mass: Mass in kg.
            linear_damping: Linear damping coefficient.
            angular_damping: Angular damping coefficient.
            enable_gravity: Whether gravity affects this body.

        Returns:
            Physics params result or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.set_physics_params(
                    actor_path=actor_path,
                    mass=mass,
                    linear_damping=linear_damping,
                    angular_damping=angular_damping,
                    enable_gravity=enable_gravity,
                )
                payload["success"] = True
                return UnrealSetPhysicsParamsResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error setting Unreal physics params: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    # ------------------------------------------------------------------
    # Phase 6: USD / SimReady Bridge
    # ------------------------------------------------------------------

    async def import_unreal_usd(
        self,
        usd_path: str,
        target_path: Optional[str] = None,
        import_animations: bool = True,
        import_materials: bool = True,
        scale_factor: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Import a USD file into Unreal via Interchange Framework.

        Args:
            usd_path: Path to the .usd/.usda/.usdc file.
            target_path: Target content path in Unreal.
            import_animations: Import animation data.
            import_materials: Import material assignments.
            scale_factor: Scale factor for import.

        Returns:
            Import result or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.import_usd(
                    usd_path=usd_path,
                    target_path=target_path,
                    import_animations=import_animations,
                    import_materials=import_materials,
                    scale_factor=scale_factor,
                )
                payload["success"] = True
                return UnrealImportUsdResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error importing USD to Unreal: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def export_unreal_usd(
        self,
        actor_paths: List[str],
        output_path: str,
        export_materials: bool = True,
        export_animations: bool = True,
        convert_to_meters: bool = True,
    ) -> Dict[str, Any]:
        """
        Export Unreal actors to USD via Interchange Framework.

        Args:
            actor_paths: Actor paths to export.
            output_path: Output file path.
            export_materials: Export material assignments.
            export_animations: Export animation data.
            convert_to_meters: Convert cm to meters.

        Returns:
            Export result or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.export_usd(
                    actor_paths=actor_paths,
                    output_path=output_path,
                    export_materials=export_materials,
                    export_animations=export_animations,
                    convert_to_meters=convert_to_meters,
                )
                payload["success"] = True
                return UnrealExportUsdResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error exporting Unreal USD: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def convert_to_simready(
        self,
        actor_paths: List[str],
        output_directory: str,
        add_physics: bool = True,
        add_collision: bool = True,
        semantic_labels: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Convert Unreal actors to NVIDIA SimReady asset format.

        Args:
            actor_paths: Actor paths to convert.
            output_directory: Output directory for SimReady assets.
            add_physics: Add physics schemas.
            add_collision: Generate collision geometry.
            semantic_labels: Semantic label mapping.

        Returns:
            Conversion result or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.convert_to_simready(
                    actor_paths=actor_paths,
                    output_directory=output_directory,
                    add_physics=add_physics,
                    add_collision=add_collision,
                    semantic_labels=semantic_labels,
                )
                payload["success"] = True
                return UnrealConvertToSimreadyResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error converting to SimReady: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def validate_simready_asset(
        self,
        asset_path: str,
        checks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Validate an asset against SimReady requirements.

        Args:
            asset_path: Path to asset to validate.
            checks: Specific checks to run.

        Returns:
            Validation result or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.validate_simready_asset(
                    asset_path=asset_path,
                    checks=checks,
                )
                payload["success"] = True
                return UnrealValidateSimreadyResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error validating SimReady asset: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def get_unreal_interchange_info(self) -> Dict[str, Any]:
        """
        Query available Interchange pipelines and supported formats.

        Returns:
            Interchange info or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.get_interchange_info()
                payload["success"] = True
                return UnrealGetInterchangeInfoResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error getting interchange info: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    # ------------------------------------------------------------------
    # Phase 7: Advanced Agent Tools
    # ------------------------------------------------------------------

    async def batch_unreal_operations(
        self,
        operations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Execute multiple Remote Control operations in one HTTP call.

        Args:
            operations: List of operation dicts with RequestId, Url, Verb, Body.

        Returns:
            Batch results or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.batch_operations(
                    operations=operations,
                )
                payload["success"] = True
                return UnrealBatchOperationsResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error in batch Unreal operations: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def query_unreal_scene_graph(
        self,
        root_path: Optional[str] = None,
        max_depth: int = 10,
        include_components: bool = False,
        class_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Query the Unreal scene graph hierarchy.

        Args:
            root_path: Root actor path (None for level root).
            max_depth: Maximum traversal depth.
            include_components: Include component sub-trees.
            class_filter: Filter actors by UClass name.

        Returns:
            Scene graph tree or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.query_scene_graph(
                    root_path=root_path,
                    max_depth=max_depth,
                    include_components=include_components,
                    class_filter=class_filter,
                )
                payload["success"] = True
                return UnrealQuerySceneGraphResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error querying Unreal scene graph: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def analyze_unreal_scene_for_robotics(
        self,
        analysis_types: Optional[List[str]] = None,
        actor_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze the Unreal scene for robotics use-cases.

        Args:
            analysis_types: Analyses to run.
            actor_filter: Filter to specific actor subtree.

        Returns:
            Analysis result or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.analyze_scene_for_robotics(
                    analysis_types=analysis_types,
                    actor_filter=actor_filter,
                )
                payload["success"] = True
                return UnrealAnalyzeSceneForRoboticsResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error analyzing scene for robotics: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def generate_unreal_procedural_scene(
        self,
        scene_type: str,
        parameters: Optional[Dict[str, Any]] = None,
        bounds_min: Optional[List[float]] = None,
        bounds_max: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a procedural scene via PCG or scripted spawning.

        Args:
            scene_type: Scene type (warehouse, outdoor, room, corridor).
            parameters: Generation params (size, density, seed, etc.).
            bounds_min: Min bounds [x, y, z] in cm.
            bounds_max: Max bounds [x, y, z] in cm.

        Returns:
            Generation result or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.generate_procedural_scene(
                    scene_type=scene_type,
                    parameters=parameters,
                    bounds_min=bounds_min,
                    bounds_max=bounds_max,
                )
                payload["success"] = True
                return UnrealGenerateProceduralSceneResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error generating procedural scene: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def get_unreal_actor_by_semantic_label(
        self,
        label: str,
        match_mode: str = "exact",
        max_results: int = 100,
    ) -> Dict[str, Any]:
        """
        Find actors by semantic tag or label.

        Args:
            label: Semantic label to search for.
            match_mode: Match mode (exact, contains, regex).
            max_results: Maximum results.

        Returns:
            Matching actors or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.get_actor_by_semantic_label(
                    label=label,
                    match_mode=match_mode,
                    max_results=max_results,
                )
                payload["success"] = True
                return UnrealGetActorBySemanticLabelResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error finding actors by semantic label: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    # ------------------------------------------------------------------
    # Phase 8: Geometry & Modeling
    # ------------------------------------------------------------------

    async def generate_unreal_mesh_primitive(
        self,
        primitive_type: str,
        dimensions: Optional[Dict[str, float]] = None,
        segments: int = 32,
        location: Optional[List[float]] = None,
        actor_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a parametric mesh primitive via GeometryScript.

        Args:
            primitive_type: box, sphere, cylinder, cone, torus, capsule.
            dimensions: Type-specific dimensions.
            segments: Tessellation segments.
            location: Spawn location [x, y, z] in cm.
            actor_label: Optional label for the actor.

        Returns:
            Mesh creation result or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.generate_mesh_primitive(
                    primitive_type=primitive_type,
                    dimensions=dimensions,
                    segments=segments,
                    location=location,
                    actor_label=actor_label,
                )
                payload["success"] = True
                return UnrealGenerateMeshPrimitiveResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error generating mesh primitive: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def apply_unreal_mesh_boolean(
        self,
        target_mesh_path: str,
        tool_mesh_path: str,
        operation: str,
    ) -> Dict[str, Any]:
        """
        Apply boolean operation between two meshes.

        Args:
            target_mesh_path: Target mesh actor (modified in-place).
            tool_mesh_path: Tool mesh actor (operand).
            operation: union, subtract, or intersect.

        Returns:
            Boolean result or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.apply_mesh_boolean(
                    target_mesh_path=target_mesh_path,
                    tool_mesh_path=tool_mesh_path,
                    operation=operation,
                )
                payload["success"] = True
                return UnrealApplyMeshBooleanResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error applying mesh boolean: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def compute_unreal_convex_hull(
        self,
        mesh_path: str,
    ) -> Dict[str, Any]:
        """
        Compute convex hull envelope of a mesh.

        Args:
            mesh_path: Source mesh actor path.

        Returns:
            Hull result or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.compute_convex_hull(
                    mesh_path=mesh_path,
                )
                payload["success"] = True
                return UnrealComputeConvexHullResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error computing convex hull: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def decompose_unreal_convex_hull(
        self,
        mesh_path: str,
        max_hulls: int = 16,
        max_vertices_per_hull: int = 32,
        min_cluster_size: int = 256,
        resolution: int = 100000,
    ) -> Dict[str, Any]:
        """
        V-HACD convex decomposition for collision geometry.

        Args:
            mesh_path: Source mesh actor path.
            max_hulls: Maximum convex pieces.
            max_vertices_per_hull: Max vertices per hull.
            min_cluster_size: Minimum cluster size.
            resolution: Voxelization resolution.

        Returns:
            Decomposition result or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.decompose_convex_hull(
                    mesh_path=mesh_path,
                    max_hulls=max_hulls,
                    max_vertices_per_hull=max_vertices_per_hull,
                    min_cluster_size=min_cluster_size,
                    resolution=resolution,
                )
                payload["success"] = True
                return UnrealDecomposeConvexHullResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error decomposing convex hull: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def edit_unreal_mesh_topology(
        self,
        mesh_path: str,
        operation: str,
        face_selection: Optional[str] = None,
        edge_selection: Optional[str] = None,
        distance: Optional[float] = None,
        offset: Optional[float] = None,
        scale: Optional[List[float]] = None,
        count: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Edit mesh topology (extrude, bevel, inset, loop cut, scale_faces).

        Args:
            mesh_path: Mesh actor path.
            operation: extrude_faces, bevel_edges, inset_faces, loop_cut, scale_faces.
            face_selection: Face selection filter.
            edge_selection: Edge selection filter.
            distance: Extrude distance.
            offset: Bevel/inset offset.
            scale: Scale factors [x, y, z].
            count: Loop cut count.

        Returns:
            Topology edit result or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.edit_mesh_topology(
                    mesh_path=mesh_path,
                    operation=operation,
                    face_selection=face_selection,
                    edge_selection=edge_selection,
                    distance=distance,
                    offset=offset,
                    scale=scale,
                    count=count,
                )
                payload["success"] = True
                return UnrealEditMeshTopologyResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error editing mesh topology: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def subdivide_unreal_mesh(
        self,
        mesh_path: str,
        level: int = 2,
        scheme: str = "catmull_clark",
    ) -> Dict[str, Any]:
        """
        Catmull-Clark / Loop / bilinear subdivision.

        Args:
            mesh_path: Mesh actor path.
            level: Subdivision level (1-4).
            scheme: catmull_clark, loop, or bilinear.

        Returns:
            Subdivision result or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.subdivide_mesh(
                    mesh_path=mesh_path,
                    level=level,
                    scheme=scheme,
                )
                payload["success"] = True
                return UnrealSubdivideMeshResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error subdividing mesh: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def simplify_unreal_mesh(
        self,
        mesh_path: str,
        target_triangle_count: Optional[int] = None,
        target_percentage: Optional[float] = None,
        max_error: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Simplify/decimate a mesh.

        Args:
            mesh_path: Mesh actor path.
            target_triangle_count: Target triangle count.
            target_percentage: Target percentage (0.0-1.0).
            max_error: Max geometric error tolerance.

        Returns:
            Simplification result or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.simplify_mesh(
                    mesh_path=mesh_path,
                    target_triangle_count=target_triangle_count,
                    target_percentage=target_percentage,
                    max_error=max_error,
                )
                payload["success"] = True
                return UnrealSimplifyMeshResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error simplifying mesh: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def cut_unreal_mesh_plane(
        self,
        mesh_path: str,
        plane_origin: List[float],
        plane_normal: List[float],
        fill_holes: bool = True,
        keep_both_sides: bool = False,
    ) -> Dict[str, Any]:
        """
        Cut/slice a mesh along an arbitrary plane.

        Args:
            mesh_path: Mesh actor path.
            plane_origin: Plane origin [x, y, z] in cm.
            plane_normal: Plane normal [x, y, z].
            fill_holes: Fill cut holes with faces.
            keep_both_sides: Keep both halves.

        Returns:
            Cut result or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.cut_mesh_plane(
                    mesh_path=mesh_path,
                    plane_origin=plane_origin,
                    plane_normal=plane_normal,
                    fill_holes=fill_holes,
                    keep_both_sides=keep_both_sides,
                )
                payload["success"] = True
                return UnrealCutMeshPlaneResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error cutting mesh with plane: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def validate_unreal_mesh(
        self,
        mesh_path: str,
        checks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Validate mesh geometry (watertight, manifold, normals, etc.).

        Args:
            mesh_path: Mesh actor path.
            checks: Checks to run.

        Returns:
            Validation result or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.validate_mesh(
                    mesh_path=mesh_path,
                    checks=checks,
                )
                payload["success"] = True
                return UnrealValidateMeshResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error validating mesh: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def convert_unreal_mesh_format(
        self,
        mesh_path: str,
        target_format: str,
        tessellation_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Convert between StaticMesh, DynamicMesh, or CAD tessellation.

        Args:
            mesh_path: Source mesh actor/asset path.
            target_format: static_mesh, dynamic_mesh, or cad_tessellation.
            tessellation_options: CAD tessellation params.

        Returns:
            Conversion result or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.convert_mesh_format(
                    mesh_path=mesh_path,
                    target_format=target_format,
                    tessellation_options=tessellation_options,
                )
                payload["success"] = True
                return UnrealConvertMeshFormatResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error converting mesh format: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def remesh_unreal_mesh(
        self,
        mesh_path: str,
        mode: str = "uniform",
        target_edge_length: Optional[float] = None,
        target_triangle_count: Optional[int] = None,
        smoothing_iterations: int = 3,
    ) -> Dict[str, Any]:
        """
        Remesh for clean topology (uniform or adaptive).

        Args:
            mesh_path: Mesh actor path.
            mode: uniform or adaptive.
            target_edge_length: Target edge length in cm (uniform).
            target_triangle_count: Target triangle count (adaptive).
            smoothing_iterations: Smoothing passes.

        Returns:
            Remesh result or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.remesh_mesh(
                    mesh_path=mesh_path,
                    mode=mode,
                    target_edge_length=target_edge_length,
                    target_triangle_count=target_triangle_count,
                    smoothing_iterations=smoothing_iterations,
                )
                payload["success"] = True
                return UnrealRemeshMeshResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error remeshing mesh: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()

    async def compute_unreal_mesh_uv(
        self,
        mesh_path: str,
        method: str = "auto_uv",
        uv_channel: int = 0,
        island_padding: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Generate UV coordinates for a mesh.

        Args:
            mesh_path: Mesh actor path.
            method: auto_uv, box, planar, cylindrical, atlas_pack.
            uv_channel: UV channel index.
            island_padding: Padding between UV islands.

        Returns:
            UV generation result or error response.
        """
        try:
            if not self.unreal_adapter or not self.unreal_adapter.is_available():
                return ErrorResponse(
                    error="Unreal runtime not available",
                    error_type="RuntimeError",
                ).dict()

            with self.unreal_adapter.create_session() as session:
                payload = await session.compute_mesh_uv(
                    mesh_path=mesh_path,
                    method=method,
                    uv_channel=uv_channel,
                    island_padding=island_padding,
                )
                payload["success"] = True
                return UnrealComputeMeshUvResponse(**payload).dict()

        except Exception as e:
            self.logger.error(f"Error computing mesh UVs: {e}")
            return ErrorResponse(error=str(e), error_type="Exception").dict()