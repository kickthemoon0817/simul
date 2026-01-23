"""
Isaac Sim specific MCP tools for Isaac Sim MCP Server.

This module provides MCP tools for Isaac Sim specific functionality including
viewport capture, simulation control, and camera operations.
"""

from typing import Dict, List, Optional, Any, Tuple
import base64

from ...logging import get_logger, LoggerMixin
from ...config import Settings, get_settings
from ...adapters import IsaacRuntimeAdapter, is_isaac_available
from ..schemas import *

logger = get_logger(__name__)


class ViewportTools(LoggerMixin):
    """Tools for viewport operations."""
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize viewport tools."""
        self.settings = settings or get_settings()
        self.isaac_adapter = IsaacRuntimeAdapter(self.settings) if is_isaac_available() else None
    
    async def capture_viewport(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        format: str = "png",
        save_to_file: bool = False,
        file_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Capture the Isaac Sim viewport.
        
        Args:
            width: Image width
            height: Image height
            format: Image format (png, jpg, exr)
            save_to_file: Save image to file
            file_path: File path for saved image
            
        Returns:
            Viewport capture response or error
        """
        try:
            if not self.isaac_adapter or not self.isaac_adapter.is_available():
                return ErrorResponse(
                    error="Isaac Sim runtime not available",
                    error_type="RuntimeError"
                ).dict()
            
            # Validate format
            valid_formats = ['png', 'jpg', 'jpeg', 'exr']
            if format.lower() not in valid_formats:
                return ErrorResponse(
                    error=f"Invalid format: {format}. Must be one of {valid_formats}",
                    error_type="ValidationError"
                ).dict()
            
            # Validate dimensions
            max_size = self.settings.viewport.max_size
            if width and width > max_size:
                return ErrorResponse(
                    error=f"Width {width} exceeds maximum {max_size}",
                    error_type="ValidationError"
                ).dict()
            
            if height and height > max_size:
                return ErrorResponse(
                    error=f"Height {height} exceeds maximum {max_size}",
                    error_type="ValidationError"
                ).dict()
            
            with self.isaac_adapter.create_session() as session:
                capture = session.capture_viewport(
                    width=width,
                    height=height,
                    format=format,
                    save_to_file=save_to_file,
                    file_path=file_path
                )
                
                if capture:
                    response_data = ViewportCaptureResponse(
                        success=True,
                        width=capture.width,
                        height=capture.height,
                        format=capture.format,
                        file_path=capture.file_path
                    ).dict()
                    
                    # Add base64 encoded image data if not saving to file
                    if not save_to_file and capture.image_data:
                        response_data['image_data_base64'] = base64.b64encode(capture.image_data).decode('utf-8')
                    
                    return response_data
                else:
                    return ErrorResponse(
                        error="Failed to capture viewport",
                        error_type="CaptureError"
                    ).dict()
            
        except Exception as e:
            self.logger.error(f"Error capturing viewport: {e}")
            return ErrorResponse(
                error=str(e),
                error_type="Exception",
                details={
                    "width": width,
                    "height": height,
                    "format": format,
                    "save_to_file": save_to_file,
                    "file_path": file_path
                }
            ).dict()
    
    async def get_viewport_info(self) -> Dict[str, Any]:
        """
        Get information about the current viewport.
        
        Returns:
            Viewport information or error
        """
        try:
            if not self.isaac_adapter or not self.isaac_adapter.is_available():
                return ErrorResponse(
                    error="Isaac Sim runtime not available",
                    error_type="RuntimeError"
                ).dict()
            
            with self.isaac_adapter.create_session() as session:
                info = session.get_viewport_info()
                info["success"] = True
                return info
            
        except Exception as e:
            self.logger.error(f"Error getting viewport info: {e}")
            return ErrorResponse(
                error=str(e),
                error_type="Exception"
            ).dict()


class SimulationTools(LoggerMixin):
    """Tools for simulation control."""
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize simulation tools."""
        self.settings = settings or get_settings()
        self.isaac_adapter = IsaacRuntimeAdapter(self.settings) if is_isaac_available() else None
    
    async def control_simulation(self, action: str, steps: int = 1) -> Dict[str, Any]:
        """
        Control Isaac Sim simulation.
        
        Args:
            action: Action (play, pause, stop, reset, step)
            steps: Number of steps (for step action)
            
        Returns:
            Success status or error
        """
        try:
            if not self.isaac_adapter or not self.isaac_adapter.is_available():
                return ErrorResponse(
                    error="Isaac Sim runtime not available",
                    error_type="RuntimeError"
                ).dict()
            
            # Validate action
            valid_actions = ['play', 'pause', 'stop', 'reset', 'step']
            if action not in valid_actions:
                return ErrorResponse(
                    error=f"Invalid action: {action}. Must be one of {valid_actions}",
                    error_type="ValidationError"
                ).dict()
            
            # Validate steps
            if action == 'step' and steps < 1:
                return ErrorResponse(
                    error=f"Steps must be positive for step action: {steps}",
                    error_type="ValidationError"
                ).dict()
            
            with self.isaac_adapter.create_session() as session:
                # Initialize world if needed
                if not session.get_world():
                    world_initialized = session.initialize_world()
                    if not world_initialized:
                        return ErrorResponse(
                            error="Failed to initialize Isaac Sim World",
                            error_type="InitializationError"
                        ).dict()
                
                success = False
                if action == "play":
                    success = session.play_simulation()
                elif action == "pause":
                    success = session.pause_simulation()
                elif action == "stop":
                    success = session.stop_simulation()
                elif action == "reset":
                    success = session.reset_simulation()
                elif action == "step":
                    success = session.step_simulation(steps)
                
                return {
                    "success": success,
                    "action": action,
                    "steps": steps if action == "step" else None,
                    "message": f"Simulation {action} {'successful' if success else 'failed'}"
                }
            
        except Exception as e:
            self.logger.error(f"Error controlling simulation: {e}")
            return ErrorResponse(
                error=str(e),
                error_type="Exception",
                details={"action": action, "steps": steps}
            ).dict()
    
    async def get_simulation_status(self) -> Dict[str, Any]:
        """
        Get current simulation status.
        
        Returns:
            Simulation status or error
        """
        try:
            if not self.isaac_adapter or not self.isaac_adapter.is_available():
                return ErrorResponse(
                    error="Isaac Sim runtime not available",
                    error_type="RuntimeError"
                ).dict()
            
            with self.isaac_adapter.create_session() as session:
                status = session.get_simulation_status()
                status["success"] = True
                return status
            
        except Exception as e:
            self.logger.error(f"Error getting simulation status: {e}")
            return ErrorResponse(
                error=str(e),
                error_type="Exception"
            ).dict()


class RigidBodyTools(LoggerMixin):
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.isaac_adapter = IsaacRuntimeAdapter(self.settings) if is_isaac_available() else None

    async def enable_rigid_body(self, prim_path: str, mass: Optional[float] = None) -> Dict[str, Any]:
        try:
            if not self.isaac_adapter or not self.isaac_adapter.is_available():
                return ErrorResponse(
                    error="Isaac Sim runtime not available",
                    error_type="RuntimeError"
                ).dict()

            with self.isaac_adapter.create_session() as session:
                success = session.enable_rigid_body(prim_path, mass)
                return {
                    "success": success,
                    "prim_path": prim_path,
                    "message": f"Rigid body {'enabled' if success else 'not enabled'} for {prim_path}",
                }

        except Exception as e:
            self.logger.error(f"Error enabling rigid body: {e}")
            return ErrorResponse(
                error=str(e),
                error_type="Exception",
                details={"prim_path": prim_path, "mass": mass}
            ).dict()

    async def set_rigid_body_velocity(
        self,
        prim_path: str,
        linear_velocity: Optional[List[float]] = None,
        angular_velocity: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        try:
            if not self.isaac_adapter or not self.isaac_adapter.is_available():
                return ErrorResponse(
                    error="Isaac Sim runtime not available",
                    error_type="RuntimeError"
                ).dict()

            with self.isaac_adapter.create_session() as session:
                success = session.set_rigid_body_velocity(prim_path, linear_velocity, angular_velocity)
                return {
                    "success": success,
                    "prim_path": prim_path,
                    "message": f"Rigid body velocity {'updated' if success else 'not updated'} for {prim_path}",
                }

        except Exception as e:
            self.logger.error(f"Error setting rigid body velocity: {e}")
            return ErrorResponse(
                error=str(e),
                error_type="Exception",
                details={
                    "prim_path": prim_path,
                    "linear_velocity": linear_velocity,
                    "angular_velocity": angular_velocity,
                }
            ).dict()

    async def get_rigid_body_state(self, prim_path: str) -> Dict[str, Any]:
        try:
            if not self.isaac_adapter or not self.isaac_adapter.is_available():
                return ErrorResponse(
                    error="Isaac Sim runtime not available",
                    error_type="RuntimeError"
                ).dict()

            with self.isaac_adapter.create_session() as session:
                state = session.get_rigid_body_state(prim_path)
                if not state:
                    return ErrorResponse(
                        error=f"Rigid body not found: {prim_path}",
                        error_type="NotFoundError"
                    ).dict()

                return {
                    "success": True,
                    **state,
                }

        except Exception as e:
            self.logger.error(f"Error getting rigid body state: {e}")
            return ErrorResponse(
                error=str(e),
                error_type="Exception",
                details={"prim_path": prim_path}
            ).dict()


class CameraTools(LoggerMixin):
    """Tools for camera control."""
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize camera tools."""
        self.settings = settings or get_settings()
        self.isaac_adapter = IsaacRuntimeAdapter(self.settings) if is_isaac_available() else None
    
    async def set_camera_view(
        self,
        eye: List[float],
        target: List[float],
        up: List[float] = [0, 1, 0]
    ) -> Dict[str, Any]:
        """
        Set camera view in the viewport.
        
        Args:
            eye: Camera position [x, y, z]
            target: Camera target [x, y, z]
            up: Up vector [x, y, z]
            
        Returns:
            Success status or error
        """
        try:
            if not self.isaac_adapter or not self.isaac_adapter.is_available():
                return ErrorResponse(
                    error="Isaac Sim runtime not available",
                    error_type="RuntimeError"
                ).dict()
            
            # Validate input vectors
            if len(eye) != 3:
                return ErrorResponse(
                    error=f"Eye position must have 3 components: {eye}",
                    error_type="ValidationError"
                ).dict()
            
            if len(target) != 3:
                return ErrorResponse(
                    error=f"Target position must have 3 components: {target}",
                    error_type="ValidationError"
                ).dict()
            
            if len(up) != 3:
                return ErrorResponse(
                    error=f"Up vector must have 3 components: {up}",
                    error_type="ValidationError"
                ).dict()
            
            with self.isaac_adapter.create_session() as session:
                eye_tuple = (eye[0], eye[1], eye[2])
                target_tuple = (target[0], target[1], target[2])
                up_tuple = (up[0], up[1], up[2])

                success = session.set_camera_view(
                    eye=eye_tuple,
                    target=target_tuple,
                    up=up_tuple
                )
                
                return {
                    "success": success,
                    "eye": eye,
                    "target": target,
                    "up": up,
                    "message": f"Camera view {'set successfully' if success else 'failed to set'}"
                }
            
        except Exception as e:
            self.logger.error(f"Error setting camera view: {e}")
            return ErrorResponse(
                error=str(e),
                error_type="Exception",
                details={"eye": eye, "target": target, "up": up}
            ).dict()
    
    async def get_camera_info(self) -> Dict[str, Any]:
        """
        Get information about the current camera.
        
        Returns:
            Camera information or error
        """
        try:
            if not self.isaac_adapter or not self.isaac_adapter.is_available():
                return ErrorResponse(
                    error="Isaac Sim runtime not available",
                    error_type="RuntimeError"
                ).dict()
            
            with self.isaac_adapter.create_session() as session:
                info = session.get_camera_info()
                info["success"] = True
                info["can_control"] = info.get("camera_available", False)
                return info
            
        except Exception as e:
            self.logger.error(f"Error getting camera info: {e}")
            return ErrorResponse(
                error=str(e),
                error_type="Exception"
            ).dict()
    
    async def focus_on_prim(self, stage_id: str, prim_path: str) -> Dict[str, Any]:
        """
        Focus camera on a specific prim.
        
        Args:
            stage_id: Stage identifier
            prim_path: Path to the prim to focus on
            
        Returns:
            Success status or error
        """
        try:
            if not self.isaac_adapter or not self.isaac_adapter.is_available():
                return ErrorResponse(
                    error="Isaac Sim runtime not available",
                    error_type="RuntimeError"
                ).dict()
            
            with self.isaac_adapter.create_session() as session:
                # Get prim bounding box to determine focus point
                bbox_dict = session.get_prim_bbox(stage_id, prim_path, world_space=True)
                if not bbox_dict:
                    return ErrorResponse(
                        error=f"Could not get bounding box for prim: {prim_path}",
                        error_type="ComputationError"
                    ).dict()
                
                # Calculate camera position based on bounding box
                min_point = bbox_dict['min']
                max_point = bbox_dict['max']
                
                # Center of the bounding box
                center = [
                    (min_point[0] + max_point[0]) / 2,
                    (min_point[1] + max_point[1]) / 2,
                    (min_point[2] + max_point[2]) / 2
                ]
                
                # Calculate size to determine camera distance
                size = [
                    max_point[0] - min_point[0],
                    max_point[1] - min_point[1],
                    max_point[2] - min_point[2]
                ]
                max_size = max(size)
                
                # Position camera at a distance based on object size
                distance = max_size * 2.0  # Adjust multiplier as needed
                eye = [center[0] + distance, center[1] + distance, center[2] + distance]
                
                eye_tuple = (eye[0], eye[1], eye[2])
                center_tuple = (center[0], center[1], center[2])

                success = session.set_camera_view(
                    eye=eye_tuple,
                    target=center_tuple,
                    up=(0, 1, 0)
                )
                
                return {
                    "success": success,
                    "stage_id": stage_id,
                    "prim_path": prim_path,
                    "focus_point": center,
                    "camera_position": eye,
                    "message": f"Camera {'focused on' if success else 'failed to focus on'} {prim_path}"
                }
            
        except Exception as e:
            self.logger.error(f"Error focusing on prim {stage_id}:{prim_path}: {e}")
            return ErrorResponse(
                error=str(e),
                error_type="Exception",
                details={"stage_id": stage_id, "prim_path": prim_path}
            ).dict()
