"""
USD-related MCP tools for Isaac Sim MCP Server.

This module provides MCP tools for USD file operations, scene analysis,
mesh operations, and bounding box computations.
"""

from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from ...logging import get_logger, LoggerMixin
from ...config import Settings, get_settings
from ...adapters import HeadlessUSDAdapter, is_headless_available
from ..schemas import *

logger = get_logger(__name__)


class USDFileTools(LoggerMixin):
    """Tools for USD file operations."""
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize USD file tools."""
        self.settings = settings or get_settings()
        self.headless_adapter = HeadlessUSDAdapter(self.settings) if is_headless_available() else None

    def get_adapter(self):
        """Get the best available adapter."""
        return self.headless_adapter
    
    async def load_usd_file(self, file_path: str) -> Dict[str, Any]:
        """
        Load a USD file and return stage information.
        
        Args:
            file_path: Path to USD file
            
        Returns:
            Stage information or error response
        """
        try:
            # Validate file path
            path = Path(file_path)
            if not path.exists():
                return ErrorResponse(
                    error=f"File not found: {file_path}",
                    error_type="FileNotFoundError"
                ).dict()
            
            if not path.is_file():
                return ErrorResponse(
                    error=f"Path is not a file: {file_path}",
                    error_type="InvalidPathError"
                ).dict()
            
            # Check file extension
            valid_extensions = ['.usd', '.usda', '.usdc', '.usdz']
            if path.suffix.lower() not in valid_extensions:
                return ErrorResponse(
                    error=f"Invalid USD file extension: {path.suffix}",
                    error_type="InvalidFileTypeError"
                ).dict()
            
            adapter = self.get_adapter()
            if not adapter:
                return ErrorResponse(
                    error="No USD adapter available",
                    error_type="AdapterError"
                ).dict()
            
            with adapter.create_session() as session:
                stage_id = session.load_stage(str(path.resolve()))
                if stage_id:
                    stage_info = session.get_stage_info(stage_id)
                    if stage_info:
                        return StageInfo(
                            stage_id=stage_id,
                            file_path=str(path.resolve()),
                            up_axis=stage_info.up_axis,
                            meters_per_unit=stage_info.meters_per_unit,
                            time_codes_per_second=stage_info.time_codes_per_second,
                            start_time=stage_info.start_time_code,
                            end_time=stage_info.end_time_code,
                            frame_rate=stage_info.frame_rate,
                            total_prims=len(stage_info.all_prims),
                            root_prims=stage_info.root_prims,
                            has_animation=stage_info.start_time_code != stage_info.end_time_code,
                            layer_count=len(stage_info.layers),
                            default_prim=stage_info.default_prim
                        ).dict()
            
            return ErrorResponse(
                error=f"Failed to load USD file: {file_path}",
                error_type="LoadError"
            ).dict()
            
        except Exception as e:
            self.logger.error(f"Error loading USD file {file_path}: {e}")
            return ErrorResponse(
                error=str(e),
                error_type="Exception",
                details={"file_path": file_path}
            ).dict()
    
    async def validate_usd_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate a USD file without loading it.
        
        Args:
            file_path: Path to USD file
            
        Returns:
            Validation result
        """
        try:
            path = Path(file_path)
            
            # Basic file checks
            file_exists = path.exists()
            is_file = path.is_file() if file_exists else False
            file_size = path.stat().st_size if is_file else 0
            
            # Extension check
            valid_extensions = ['.usd', '.usda', '.usdc', '.usdz']
            valid_extension = path.suffix.lower() in valid_extensions
            
            # Size check
            max_size_mb = self.settings.usd.max_file_size_mb
            size_ok = file_size <= (max_size_mb * 1024 * 1024)
            
            return USDFileInfo(
                file_path=str(path.resolve()),
                file_size=file_size,
                format=path.suffix.lower().lstrip('.'),
                is_valid=file_exists and is_file and valid_extension and size_ok,
                can_read=file_exists and is_file and valid_extension
            ).dict()
            
        except Exception as e:
            self.logger.error(f"Error validating USD file {file_path}: {e}")
            return ErrorResponse(
                error=str(e),
                error_type="Exception",
                details={"file_path": file_path}
            ).dict()


class USDSceneTools(LoggerMixin):
    """Tools for USD scene operations."""
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize USD scene tools."""
        self.settings = settings or get_settings()
        self.headless_adapter = HeadlessUSDAdapter(self.settings) if is_headless_available() else None

    def get_adapter(self):
        """Get the best available adapter."""
        return self.headless_adapter
    
    async def get_prim_info(self, stage_id: str, prim_path: str) -> Dict[str, Any]:
        """
        Get information about a USD prim.
        
        Args:
            stage_id: Stage identifier
            prim_path: Path to the prim
            
        Returns:
            Prim information or error
        """
        try:
            adapter = self.get_adapter()
            if not adapter:
                return ErrorResponse(
                    error="No USD adapter available",
                    error_type="AdapterError"
                ).dict()
            
            with adapter.create_session() as session:
                prim_info = session.get_prim_info(stage_id, prim_path)
                if prim_info:
                    # Get bounding box if available
                    bbox = None
                    if hasattr(session, 'get_prim_bbox'):
                        bbox_dict = session.get_prim_bbox(stage_id, prim_path)
                        if bbox_dict:
                            bbox = BoundingBox(**bbox_dict)
                    
                    transform = None
                    if hasattr(session, "get_prim_transform"):
                        transform_dict = session.get_prim_transform(stage_id, prim_path)
                        if transform_dict:
                            transform = Transform(**transform_dict)

                    children_types = {}
                    if hasattr(session, "get_children_type_counts"):
                        children_types = session.get_children_type_counts(stage_id, prim_path)

                    material_bindings = []
                    if hasattr(session, "get_material_bindings"):
                        material_bindings = session.get_material_bindings(stage_id, prim_path)

                    return PrimInfo(
                        path=prim_info.path,
                        name=prim_info.name,
                        type=prim_info.type_name,
                        is_active=prim_info.is_active,
                        is_loaded=prim_info.is_loaded,
                        is_defined=prim_info.is_defined,
                        is_instance=prim_info.is_instance,
                        purpose=prim_info.purpose,
                        visibility=prim_info.visibility,
                        kind=prim_info.kind,
                        bbox=bbox,
                        transform=transform,
                        children_count=len(prim_info.children),
                        children_types=children_types,
                        material_bindings=material_bindings,
                        attributes=prim_info.attributes,
                        metadata=prim_info.metadata
                    ).dict()
            
            return ErrorResponse(
                error=f"Prim not found: {prim_path}",
                error_type="NotFoundError",
                details={"stage_id": stage_id, "prim_path": prim_path}
            ).dict()
            
        except Exception as e:
            self.logger.error(f"Error getting prim info {stage_id}:{prim_path}: {e}")
            return ErrorResponse(
                error=str(e),
                error_type="Exception",
                details={"stage_id": stage_id, "prim_path": prim_path}
            ).dict()

    async def create_prim(
        self,
        stage_id: str,
        prim_path: str,
        prim_type: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            adapter = self.get_adapter()
            if not adapter:
                return ErrorResponse(
                    error="No USD adapter available",
                    error_type="AdapterError"
                ).dict()

            with adapter.create_session() as session:
                success = session.create_prim(stage_id, prim_path, prim_type, attributes or {})
                if success:
                    return PrimActionResponse(
                        success=True,
                        stage_id=stage_id,
                        prim_path=prim_path,
                        message=f"Created prim {prim_path}",
                    ).dict()

            return ErrorResponse(
                error=f"Failed to create prim: {prim_path}",
                error_type="CreateError",
                details={"stage_id": stage_id, "prim_type": prim_type}
            ).dict()

        except Exception as e:
            self.logger.error(f"Error creating prim {stage_id}:{prim_path}: {e}")
            return ErrorResponse(
                error=str(e),
                error_type="Exception",
                details={"stage_id": stage_id, "prim_path": prim_path}
            ).dict()

    async def update_prim_attributes(
        self,
        stage_id: str,
        prim_path: str,
        attributes: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            adapter = self.get_adapter()
            if not adapter:
                return ErrorResponse(
                    error="No USD adapter available",
                    error_type="AdapterError"
                ).dict()

            with adapter.create_session() as session:
                success = session.update_prim_attributes(stage_id, prim_path, attributes)
                if success:
                    return PrimActionResponse(
                        success=True,
                        stage_id=stage_id,
                        prim_path=prim_path,
                        message=f"Updated prim {prim_path}",
                    ).dict()

            return ErrorResponse(
                error=f"Failed to update prim: {prim_path}",
                error_type="UpdateError",
                details={"stage_id": stage_id}
            ).dict()

        except Exception as e:
            self.logger.error(f"Error updating prim {stage_id}:{prim_path}: {e}")
            return ErrorResponse(
                error=str(e),
                error_type="Exception",
                details={"stage_id": stage_id, "prim_path": prim_path}
            ).dict()

    async def delete_prim(self, stage_id: str, prim_path: str) -> Dict[str, Any]:
        try:
            adapter = self.get_adapter()
            if not adapter:
                return ErrorResponse(
                    error="No USD adapter available",
                    error_type="AdapterError"
                ).dict()

            with adapter.create_session() as session:
                success = session.delete_prim(stage_id, prim_path)
                if success:
                    return PrimActionResponse(
                        success=True,
                        stage_id=stage_id,
                        prim_path=prim_path,
                        message=f"Deleted prim {prim_path}",
                    ).dict()

            return ErrorResponse(
                error=f"Failed to delete prim: {prim_path}",
                error_type="DeleteError",
                details={"stage_id": stage_id}
            ).dict()

        except Exception as e:
            self.logger.error(f"Error deleting prim {stage_id}:{prim_path}: {e}")
            return ErrorResponse(
                error=str(e),
                error_type="Exception",
                details={"stage_id": stage_id, "prim_path": prim_path}
            ).dict()
    
    async def search_prims(self, stage_id: str, search_type: str, query: str, exact_match: bool = False) -> Dict[str, Any]:
        """
        Search for prims in a USD stage.
        
        Args:
            stage_id: Stage identifier
            search_type: Search type (by_type, by_name)
            query: Search query
            exact_match: Use exact matching for name search
            
        Returns:
            Search results or error
        """
        try:
            # Validate search type
            valid_search_types = ['by_type', 'by_name']
            if search_type not in valid_search_types:
                return ErrorResponse(
                    error=f"Invalid search type: {search_type}. Must be one of {valid_search_types}",
                    error_type="ValidationError"
                ).dict()
            
            adapter = self.get_adapter()
            if not adapter:
                return ErrorResponse(
                    error="No USD adapter available",
                    error_type="AdapterError"
                ).dict()
            
            with adapter.create_session() as session:
                results: List[str] = []
                if search_type == "by_type":
                    results = session.find_prims_by_type(stage_id, query)
                elif search_type == "by_name":
                    results = session.find_prims_by_name(stage_id, query, exact_match)

                return PrimSearchResponse(
                    success=True,
                    stage_id=stage_id,
                    search_type=search_type,
                    query=query,
                    results=results,
                    count=len(results)
                ).dict()
            
        except Exception as e:
            self.logger.error(f"Error searching prims {stage_id}: {e}")
            return ErrorResponse(
                error=str(e),
                error_type="Exception",
                details={"stage_id": stage_id, "search_type": search_type, "query": query}
            ).dict()
    
    async def summarize_scene(self, stage_id: str, include_meshes: bool = True, format: str = "json") -> Dict[str, Any]:
        """
        Generate a summary of a USD scene.
        
        Args:
            stage_id: Stage identifier
            include_meshes: Include detailed mesh information
            format: Output format (json, text)
            
        Returns:
            Scene summary or error
        """
        try:
            # Validate format
            valid_formats = ['json', 'text']
            if format not in valid_formats:
                return ErrorResponse(
                    error=f"Invalid format: {format}. Must be one of {valid_formats}",
                    error_type="ValidationError"
                ).dict()
            
            adapter = self.get_adapter()
            if not adapter:
                return ErrorResponse(
                    error="No USD adapter available",
                    error_type="AdapterError"
                ).dict()
            
            with adapter.create_session() as session:
                summary = session.summarize_stage(stage_id, include_meshes)
                if summary:
                    # Convert summary to dict
                    summary_dict = {
                        'file_path': summary.file_path,
                        'stage_info': summary.stage_info,
                        'total_prims': summary.total_prims,
                        'prim_type_counts': summary.prim_type_counts,
                        'scene_bbox': summary.scene_bbox,
                        'scene_center': summary.scene_center,
                        'scene_size': summary.scene_size,
                        'mesh_statistics': summary.mesh_statistics,
                        'hierarchy_depth': summary.hierarchy_depth,
                        'animation_info': summary.animation_info
                    }
                    
                    # Generate digest if text format requested
                    digest = None
                    if format == "text":
                        from ...usd.summarize import generate_scene_digest
                        digest = generate_scene_digest(summary)
                    
                    return SceneSummaryResponse(
                        success=True,
                        stage_id=stage_id,
                        summary=summary_dict,
                        digest=digest
                    ).dict()
                else:
                    return ErrorResponse(
                        error="Could not generate scene summary",
                        error_type="ComputationError",
                        details={"stage_id": stage_id}
                    ).dict()
            
        except Exception as e:
            self.logger.error(f"Error summarizing scene {stage_id}: {e}")
            return ErrorResponse(
                error=str(e),
                error_type="Exception",
                details={"stage_id": stage_id, "include_meshes": include_meshes, "format": format}
            ).dict()


class USDMeshTools(LoggerMixin):
    """Tools for USD mesh operations."""
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize USD mesh tools."""
        self.settings = settings or get_settings()
        self.headless_adapter = HeadlessUSDAdapter(self.settings) if is_headless_available() else None

    def get_adapter(self):
        """Get the best available adapter."""
        return self.headless_adapter
    
    async def get_mesh_info(self, stage_id: str, prim_path: str) -> Dict[str, Any]:
        """
        Get mesh information for a mesh prim.
        
        Args:
            stage_id: Stage identifier
            prim_path: Path to mesh prim
            
        Returns:
            Mesh information or error
        """
        try:
            adapter = self.get_adapter()
            if not adapter:
                return ErrorResponse(
                    error="No USD adapter available",
                    error_type="AdapterError"
                ).dict()
            
            with adapter.create_session() as session:
                mesh_info_dict = session.get_mesh_info(stage_id, prim_path)
                if mesh_info_dict:
                    # Convert bbox to BoundingBox object
                    bbox_data = mesh_info_dict.get('bbox')
                    if bbox_data and isinstance(bbox_data, tuple) and len(bbox_data) == 2:
                        min_point, max_point = bbox_data
                        bbox = BoundingBox(min=min_point, max=max_point)
                    else:
                        bbox = BoundingBox(min=[0, 0, 0], max=[0, 0, 0])
                    
                    return MeshInfo(
                        prim_path=mesh_info_dict['prim_path'],
                        vertex_count=mesh_info_dict['vertex_count'],
                        face_count=mesh_info_dict['face_count'],
                        point_count=mesh_info_dict.get('point_count', 0),
                        has_normals=mesh_info_dict['has_normals'],
                        has_uvs=mesh_info_dict['has_uvs'],
                        has_colors=mesh_info_dict['has_colors'],
                        has_tangents=mesh_info_dict.get('has_tangents', False),
                        subdivision_scheme=mesh_info_dict.get('subdivision_scheme', 'none'),
                        topology_valid=mesh_info_dict['topology_valid'],
                        is_closed=mesh_info_dict['is_closed'],
                        surface_area=mesh_info_dict['surface_area'],
                        volume=mesh_info_dict['volume'],
                        bbox=bbox,
                        materials=mesh_info_dict.get('materials', []),
                        subsets=mesh_info_dict.get('subsets', [])
                    ).dict()
                else:
                    return ErrorResponse(
                        error=f"Could not get mesh info for prim: {prim_path}",
                        error_type="ComputationError",
                        details={"stage_id": stage_id, "prim_path": prim_path}
                    ).dict()
            
        except Exception as e:
            self.logger.error(f"Error getting mesh info {stage_id}:{prim_path}: {e}")
            return ErrorResponse(
                error=str(e),
                error_type="Exception",
                details={"stage_id": stage_id, "prim_path": prim_path}
            ).dict()


class USDBBoxTools(LoggerMixin):
    """Tools for USD bounding box operations."""
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize USD bounding box tools."""
        self.settings = settings or get_settings()
        self.headless_adapter = HeadlessUSDAdapter(self.settings) if is_headless_available() else None

    def get_adapter(self):
        """Get the best available adapter."""
        return self.headless_adapter
    
    async def get_bounding_box(self, stage_id: str, prim_path: Optional[str] = None, world_space: bool = True) -> Dict[str, Any]:
        """
        Get bounding box for a prim or entire stage.
        
        Args:
            stage_id: Stage identifier
            prim_path: Prim path (None for stage bbox)
            world_space: Compute in world space
            
        Returns:
            Bounding box information or error
        """
        try:
            adapter = self.get_adapter()
            if not adapter:
                return ErrorResponse(
                    error="No USD adapter available",
                    error_type="AdapterError"
                ).dict()
            
            with adapter.create_session() as session:
                if prim_path:
                    bbox_dict = session.get_prim_bbox(stage_id, prim_path, world_space)
                else:
                    bbox_dict = session.get_stage_bbox(stage_id)
                
                if bbox_dict:
                    bbox = BoundingBox(**bbox_dict)
                    return BBoxResponse(
                        success=True,
                        stage_id=stage_id,
                        prim_path=prim_path,
                        bbox=bbox,
                        world_space=world_space
                    ).dict()
                else:
                    return ErrorResponse(
                        error="Could not compute bounding box",
                        error_type="ComputationError",
                        details={"stage_id": stage_id, "prim_path": prim_path, "world_space": world_space}
                    ).dict()
            
        except Exception as e:
            self.logger.error(f"Error computing bounding box {stage_id}:{prim_path}: {e}")
            return ErrorResponse(
                error=str(e),
                error_type="Exception",
                details={"stage_id": stage_id, "prim_path": prim_path, "world_space": world_space}
            ).dict()
