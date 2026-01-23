"""
Isaac Sim runtime adapter for Isaac Sim MCP Server.

This module provides an adapter for Isaac Sim runtime operations including
viewport capture, simulation control, and omni/carb API integration.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
import numpy as np

# Try to import Isaac Sim / Omniverse modules
try:
    import omni
    import carb
    from omni.isaac.core import World
    from omni.isaac.core.utils.stage import get_current_stage
    from omni.isaac.core.utils.viewports import set_camera_view
    import omni.kit.viewport.utility as viewport_utils
    ISAAC_AVAILABLE = True
except ImportError:
    ISAAC_AVAILABLE = False
    # Mock classes for development
    omni = None
    carb = None
    World = None

# Always available - pxr should be available in Isaac Sim
try:
    from pxr import Usd, UsdGeom, UsdPhysics
    PXR_AVAILABLE = True
except ImportError:
    PXR_AVAILABLE = False
    Usd = None
    UsdGeom = None
    UsdPhysics = None

from ..logging import get_logger, LoggerMixin
from ..utils.timing import Timer, monitor_performance
from ..config import Settings, get_settings
from ..usd import USDReader, BBoxCache, SceneSummarizer
from .headless_usd import HeadlessUSDSession

logger = get_logger(__name__)


@dataclass
class ViewportCapture:
    """Viewport capture result."""
    image_data: bytes
    width: int
    height: int
    format: str
    file_path: Optional[str] = None


class IsaacRuntimeSession(LoggerMixin):
    """
    Isaac Sim runtime session for simulation and viewport operations.
    
    Provides Isaac Sim-specific functionality including viewport capture,
    simulation control, and omni/carb API access.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize Isaac Sim runtime session.
        
        Args:
            settings: Configuration settings
        """
        if not ISAAC_AVAILABLE:
            raise ImportError("Isaac Sim runtime not available. Please run within Isaac Sim environment.")
        
        self.settings = settings or get_settings()
        
        # Initialize Isaac Sim components
        self._world: Optional[World] = None
        self._stage: Optional[Usd.Stage] = None
        self._bbox_cache: Optional[BBoxCache] = None
        
        # Fallback to headless operations for USD analysis
        self._headless_session = HeadlessUSDSession(settings)
        
        self.logger.info("Isaac Sim runtime session initialized")
    
    @monitor_performance("isaac_session.initialize_world")
    def initialize_world(self, physics_dt: Optional[float] = None, rendering_dt: Optional[float] = None) -> bool:
        """
        Initialize Isaac Sim World.
        
        Args:
            physics_dt: Physics timestep
            rendering_dt: Rendering timestep
            
        Returns:
            True if successful
        """
        try:
            if self._world is not None:
                self.logger.warning("World already initialized")
                return True
            
            # Use settings or defaults
            physics_dt = physics_dt or 1.0/60.0  # 60 FPS
            rendering_dt = rendering_dt or 1.0/60.0  # 60 FPS
            
            self._world = World(
                physics_dt=physics_dt,
                rendering_dt=rendering_dt,
                stage_units_in_meters=self.settings.isaac_sim.meters_per_unit if hasattr(self.settings.isaac_sim, 'meters_per_unit') else 1.0
            )
            
            # Get the current stage
            self._stage = get_current_stage()
            if self._stage:
                self._bbox_cache = BBoxCache(self._stage)
            
            self.logger.info("Isaac Sim World initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing Isaac Sim World: {e}")
            return False
    
    def get_world(self) -> Optional[World]:
        """Get the Isaac Sim World instance."""
        return self._world
    
    def get_stage(self) -> Optional[Usd.Stage]:
        """Get the current USD stage."""
        if self._stage is None:
            try:
                self._stage = get_current_stage()
                if self._stage and self._bbox_cache is None:
                    self._bbox_cache = BBoxCache(self._stage)
            except Exception as e:
                self.logger.debug(f"Could not get current stage: {e}")
        return self._stage
    
    @monitor_performance("isaac_session.capture_viewport")
    def capture_viewport(
        self, 
        width: Optional[int] = None,
        height: Optional[int] = None,
        format: str = "png",
        save_to_file: bool = False,
        file_path: Optional[str] = None
    ) -> Optional[ViewportCapture]:
        """
        Capture the current viewport.
        
        Args:
            width: Image width (None for current viewport size)
            height: Image height (None for current viewport size)
            format: Image format (png, jpg, exr)
            save_to_file: Save image to file
            file_path: File path (auto-generated if None)
            
        Returns:
            ViewportCapture object or None if failed
        """
        try:
            # Get viewport
            viewport_api = omni.kit.viewport.utility.get_active_viewport()
            if not viewport_api:
                self.logger.error("No active viewport found")
                return None
            
            # Get current viewport size if not specified
            if width is None or height is None:
                viewport_size = viewport_api.get_texture_resolution()
                width = width or viewport_size[0]
                height = height or viewport_size[1]
            
            # Ensure size limits
            max_size = self.settings.viewport.max_size
            if width > max_size or height > max_size:
                # Scale down maintaining aspect ratio
                aspect_ratio = width / height
                if width > height:
                    width = max_size
                    height = int(max_size / aspect_ratio)
                else:
                    height = max_size
                    width = int(max_size * aspect_ratio)
            
            # Capture viewport
            image_data = viewport_api.get_viewport_texture(width, height)
            
            if image_data is None:
                self.logger.error("Failed to capture viewport texture")
                return None
            
            # Convert to bytes based on format
            if format.lower() == "png":
                # Convert numpy array to PNG bytes
                from PIL import Image
                if isinstance(image_data, np.ndarray):
                    # Ensure correct format (RGB/RGBA)
                    if image_data.shape[2] == 4:  # RGBA
                        image = Image.fromarray(image_data, 'RGBA')
                    else:  # RGB
                        image = Image.fromarray(image_data, 'RGB')
                    
                    # Save to bytes
                    import io
                    byte_buffer = io.BytesIO()
                    image.save(byte_buffer, format='PNG')
                    image_bytes = byte_buffer.getvalue()
                else:
                    image_bytes = image_data
            else:
                # For other formats, assume data is already in correct format
                image_bytes = image_data if isinstance(image_data, bytes) else image_data.tobytes()
            
            # Save to file if requested
            saved_file_path = None
            if save_to_file:
                if file_path is None:
                    # Generate temporary file path
                    temp_dir = tempfile.gettempdir()
                    file_path = os.path.join(temp_dir, f"isaac_viewport_{os.getpid()}.{format}")
                
                try:
                    with open(file_path, 'wb') as f:
                        f.write(image_bytes)
                    saved_file_path = file_path
                    self.logger.debug(f"Saved viewport capture to: {file_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to save viewport capture: {e}")
            
            capture = ViewportCapture(
                image_data=image_bytes,
                width=width,
                height=height,
                format=format,
                file_path=saved_file_path
            )
            
            self.logger.info(f"Captured viewport: {width}x{height} {format}")
            return capture
            
        except Exception as e:
            self.logger.error(f"Error capturing viewport: {e}")
            return None
    
    def set_camera_view(
        self, 
        eye: Tuple[float, float, float],
        target: Tuple[float, float, float],
        up: Tuple[float, float, float] = (0, 1, 0)
    ) -> bool:
        """
        Set camera view in the viewport.
        
        Args:
            eye: Camera position
            target: Camera target
            up: Up vector
            
        Returns:
            True if successful
        """
        try:
            set_camera_view(eye=eye, target=target, up=up)
            self.logger.debug(f"Set camera view: eye={eye}, target={target}")
            return True
        except Exception as e:
            self.logger.error(f"Error setting camera view: {e}")
            return False
    
    def play_simulation(self) -> bool:
        """Start simulation playback."""
        try:
            if self._world:
                self._world.play()
                self.logger.info("Started simulation playback")
                return True
            else:
                self.logger.error("World not initialized")
                return False
        except Exception as e:
            self.logger.error(f"Error starting simulation: {e}")
            return False
    
    def pause_simulation(self) -> bool:
        """Pause simulation playback."""
        try:
            if self._world:
                self._world.pause()
                self.logger.info("Paused simulation playback")
                return True
            else:
                self.logger.error("World not initialized")
                return False
        except Exception as e:
            self.logger.error(f"Error pausing simulation: {e}")
            return False
    
    def stop_simulation(self) -> bool:
        """Stop simulation playback."""
        try:
            if self._world:
                self._world.stop()
                self.logger.info("Stopped simulation playback")
                return True
            else:
                self.logger.error("World not initialized")
                return False
        except Exception as e:
            self.logger.error(f"Error stopping simulation: {e}")
            return False
    
    def reset_simulation(self) -> bool:
        """Reset simulation to initial state."""
        try:
            if self._world:
                self._world.reset()
                self.logger.info("Reset simulation")
                return True
            else:
                self.logger.error("World not initialized")
                return False
        except Exception as e:
            self.logger.error(f"Error resetting simulation: {e}")
            return False
    
    def step_simulation(self, steps: int = 1) -> bool:
        """
        Step simulation forward.
        
        Args:
            steps: Number of steps to advance
            
        Returns:
            True if successful
        """
        try:
            if self._world:
                for _ in range(steps):
                    self._world.step(render=True)
                self.logger.debug(f"Stepped simulation {steps} steps")
                return True
            else:
                self.logger.error("World not initialized")
                return False
        except Exception as e:
            self.logger.error(f"Error stepping simulation: {e}")
            return False

    def get_simulation_status(self) -> Dict[str, Any]:
        """Get current simulation status."""
        status = {
            "world_initialized": self._world is not None,
            "is_playing": False,
            "current_time": 0.0,
            "physics_dt": None,
            "rendering_dt": None,
        }

        try:
            if self._world:
                if hasattr(self._world, "is_playing"):
                    status["is_playing"] = self._world.is_playing()
                if hasattr(self._world, "get_physics_dt"):
                    status["physics_dt"] = self._world.get_physics_dt()
                if hasattr(self._world, "get_rendering_dt"):
                    status["rendering_dt"] = self._world.get_rendering_dt()

            if omni and hasattr(omni, "timeline"):
                timeline = omni.timeline.get_timeline_interface()
                status["is_playing"] = timeline.is_playing()
                status["current_time"] = timeline.get_current_time()
        except Exception as e:
            self.logger.debug(f"Unable to fetch simulation status: {e}")

        return status

    def enable_rigid_body(self, prim_path: str, mass: Optional[float] = None) -> bool:
        """Enable rigid body physics on a prim."""
        if not UsdPhysics:
            self.logger.error("USD physics schema not available")
            return False

        prim = self._get_prim(prim_path)
        if not prim:
            self.logger.error(f"Prim not found: {prim_path}")
            return False

        try:
            UsdPhysics.RigidBodyAPI.Apply(prim)
            if mass is not None:
                mass_api = UsdPhysics.MassAPI.Apply(prim)
                mass_api.GetMassAttr().Set(mass)
            self.logger.debug(f"Enabled rigid body on {prim_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to enable rigid body on {prim_path}: {e}")
            return False

    def set_rigid_body_velocity(
        self,
        prim_path: str,
        linear_velocity: Optional[List[float]] = None,
        angular_velocity: Optional[List[float]] = None,
    ) -> bool:
        """Set linear/angular velocity on a rigid body prim."""
        rigid_api = self._get_rigid_body_api(prim_path)
        if not rigid_api:
            self.logger.error(f"Rigid body API not found on {prim_path}")
            return False

        try:
            if linear_velocity is not None:
                rigid_api.GetVelocityAttr().Set(linear_velocity)
            if angular_velocity is not None:
                rigid_api.GetAngularVelocityAttr().Set(angular_velocity)
            self.logger.debug(f"Set rigid body velocity on {prim_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to set velocity on {prim_path}: {e}")
            return False

    def get_rigid_body_state(self, prim_path: str) -> Optional[Dict[str, Any]]:
        """Get rigid body state from a prim."""
        rigid_api = self._get_rigid_body_api(prim_path)
        if not rigid_api:
            return None

        mass = None
        mass_api = UsdPhysics.MassAPI.Get(rigid_api.GetPrim())
        if mass_api:
            mass = mass_api.GetMassAttr().Get()

        linear_velocity = None
        angular_velocity = None
        try:
            linear_velocity = rigid_api.GetVelocityAttr().Get()
            angular_velocity = rigid_api.GetAngularVelocityAttr().Get()
        except Exception:
            pass

        enabled_attr = rigid_api.GetRigidBodyEnabledAttr()
        enabled = bool(enabled_attr.Get()) if enabled_attr else True

        return {
            "prim_path": prim_path,
            "enabled": enabled,
            "mass": mass,
            "linear_velocity": linear_velocity,
            "angular_velocity": angular_velocity,
        }

    def get_viewport_info(self) -> Dict[str, Any]:
        """Get current viewport information."""
        info = {
            "viewport_available": False,
            "width": None,
            "height": None,
            "camera_path": None,
            "supported_formats": ["png", "jpg", "jpeg", "exr"],
            "max_capture_size": self.settings.viewport.max_size,
        }

        try:
            viewport_api = omni.kit.viewport.utility.get_active_viewport()
            if not viewport_api:
                return info

            info["viewport_available"] = True
            width, height = viewport_api.get_texture_resolution()
            info["width"] = int(width)
            info["height"] = int(height)
            if hasattr(viewport_api, "get_active_camera"):
                info["camera_path"] = viewport_api.get_active_camera()
        except Exception as e:
            self.logger.debug(f"Unable to fetch viewport info: {e}")

        return info

    def get_camera_info(self) -> Dict[str, Any]:
        """Get active camera information."""
        info = {
            "camera_available": False,
            "camera_path": None,
            "focal_length": None,
            "horizontal_aperture": None,
            "vertical_aperture": None,
        }

        try:
            viewport_api = omni.kit.viewport.utility.get_active_viewport()
            if viewport_api and hasattr(viewport_api, "get_active_camera"):
                camera_path = viewport_api.get_active_camera()
                info["camera_path"] = camera_path

            stage = self.get_stage()
            if stage and info["camera_path"]:
                prim = stage.GetPrimAtPath(info["camera_path"])
                if prim and prim.IsValid() and prim.IsA(UsdGeom.Camera):
                    camera = UsdGeom.Camera(prim)
                    info["camera_available"] = True
                    info["focal_length"] = camera.GetFocalLengthAttr().Get()
                    info["horizontal_aperture"] = camera.GetHorizontalApertureAttr().Get()
                    info["vertical_aperture"] = camera.GetVerticalApertureAttr().Get()
        except Exception as e:
            self.logger.debug(f"Unable to fetch camera info: {e}")

        return info

    # Delegate USD operations to headless session
    def load_stage(self, file_path: Union[str, Path]) -> Optional[str]:
        """Load USD stage (delegates to headless session)."""
        return self._headless_session.load_stage(file_path)
    
    def get_stage_info(self, stage_id: str):
        """Get stage info (delegates to headless session)."""
        return self._headless_session.get_stage_info(stage_id)
    
    def summarize_stage(self, stage_id: str, include_meshes: bool = True):
        """Summarize stage (delegates to headless session)."""
        return self._headless_session.summarize_stage(stage_id, include_meshes)

    def get_prim_info(self, stage_id: str, prim_path: str):
        """Get prim info (delegates to headless session)."""
        return self._headless_session.get_prim_info(stage_id, prim_path)

    def find_prims_by_type(self, stage_id: str, prim_type: str) -> List[str]:
        """Find prims by type (delegates to headless session)."""
        return self._headless_session.find_prims_by_type(stage_id, prim_type)

    def find_prims_by_name(self, stage_id: str, name_pattern: str, exact_match: bool = False) -> List[str]:
        """Find prims by name (delegates to headless session)."""
        return self._headless_session.find_prims_by_name(stage_id, name_pattern, exact_match)

    def get_prim_bbox(self, stage_id: str, prim_path: str, world_space: bool = True):
        """Get prim bounding box (delegates to headless session)."""
        return self._headless_session.get_prim_bbox(stage_id, prim_path, world_space)

    def get_stage_bbox(self, stage_id: str):
        """Get stage bounding box (delegates to headless session)."""
        return self._headless_session.get_stage_bbox(stage_id)

    def get_mesh_info(self, stage_id: str, prim_path: str) -> Optional[Dict[str, Any]]:
        """Get mesh info (delegates to headless session)."""
        return self._headless_session.get_mesh_info(stage_id, prim_path)

    def get_prim_transform(self, stage_id: str, prim_path: str) -> Optional[Dict[str, List[float]]]:
        """Get prim transform (delegates to headless session)."""
        return self._headless_session.get_prim_transform(stage_id, prim_path)

    def get_children_type_counts(self, stage_id: str, prim_path: str) -> Dict[str, int]:
        """Get child prim type counts (delegates to headless session)."""
        return self._headless_session.get_children_type_counts(stage_id, prim_path)

    def get_material_bindings(self, stage_id: str, prim_path: str) -> List[str]:
        """Get bound material paths (delegates to headless session)."""
        return self._headless_session.get_material_bindings(stage_id, prim_path)

    def create_prim(
        self,
        stage_id: str,
        prim_path: str,
        prim_type: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> bool:
        return self._headless_session.create_prim(stage_id, prim_path, prim_type, attributes)

    def update_prim_attributes(
        self,
        stage_id: str,
        prim_path: str,
        attributes: Dict[str, Any],
    ) -> bool:
        return self._headless_session.update_prim_attributes(stage_id, prim_path, attributes)

    def delete_prim(self, stage_id: str, prim_path: str) -> bool:
        return self._headless_session.delete_prim(stage_id, prim_path)

    def _get_prim(self, prim_path: str) -> Optional[Usd.Prim]:
        """Fetch a prim from the current stage."""
        stage = self.get_stage()
        if not stage:
            return None
        prim = stage.GetPrimAtPath(prim_path)
        if prim and prim.IsValid():
            return prim
        return None

    def _get_rigid_body_api(self, prim_path: str) -> Optional[UsdPhysics.RigidBodyAPI]:
        """Fetch the rigid body API for a prim if applied."""
        if not UsdPhysics:
            return None
        prim = self._get_prim(prim_path)
        if not prim:
            return None
        rigid_api = UsdPhysics.RigidBodyAPI.Get(prim)
        return rigid_api if rigid_api else None

    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if self._world:
                self._world.clear()
                self._world = None
            
            self._stage = None
            self._bbox_cache = None
            
            if self._headless_session:
                self._headless_session.cleanup()
            
            self.logger.info("Isaac Sim runtime session cleaned up")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


class IsaacRuntimeAdapter(LoggerMixin):
    """
    Adapter for Isaac Sim runtime operations.
    
    Provides high-level interface for Isaac Sim-specific functionality.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize Isaac Sim runtime adapter.
        
        Args:
            settings: Configuration settings
        """
        self.settings = settings or get_settings()
        self.logger.info("Isaac Sim runtime adapter initialized")
    
    @contextmanager
    def create_session(self):
        """
        Create an Isaac Sim runtime session context manager.
        
        Yields:
            IsaacRuntimeSession instance
        """
        session = IsaacRuntimeSession(self.settings)
        try:
            yield session
        finally:
            session.cleanup()
    
    def is_available(self) -> bool:
        """
        Check if Isaac Sim runtime is available.
        
        Returns:
            True if Isaac Sim modules are available
        """
        return ISAAC_AVAILABLE
    
    def get_capabilities(self) -> List[str]:
        """
        Get list of supported capabilities.
        
        Returns:
            List of capability names
        """
        capabilities = []
        
        if ISAAC_AVAILABLE:
            capabilities.extend([
                "viewport_capture",
                "simulation_control",
                "camera_control",
                "world_management",
                "physics_simulation",
                "rendering_control"
            ])
        
        # Always include USD capabilities if pxr is available
        if PXR_AVAILABLE:
            capabilities.extend([
                "load_usd_files",
                "analyze_scene_structure",
                "extract_mesh_data",
                "compute_bounding_boxes",
                "generate_scene_summaries"
            ])
        
        return capabilities


# Convenience functions
def create_isaac_session(settings: Optional[Settings] = None) -> IsaacRuntimeSession:
    """
    Create an Isaac Sim runtime session.
    
    Args:
        settings: Configuration settings
        
    Returns:
        IsaacRuntimeSession instance
    """
    return IsaacRuntimeSession(settings)


def is_isaac_available() -> bool:
    """Check if Isaac Sim runtime is available."""
    return ISAAC_AVAILABLE
