"""
Mesh operations for USD data in Isaac Sim MCP Server.

This module provides mesh extraction, analysis, decimation, and export
functionality for USD mesh prims using the pxr library.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from ._pxr import Usd, UsdGeom

from ..logging import get_logger, LoggerMixin
from ..utils.timing import monitor_performance
from ..utils.math import Vector3, BBox, bbox_from_points

logger = get_logger(__name__)


@dataclass
class MeshInfo:
    """Information about a USD mesh."""
    prim_path: str
    vertex_count: int
    face_count: int
    point_count: int
    has_normals: bool
    has_uvs: bool
    has_colors: bool
    has_tangents: bool
    subdivision_scheme: str
    face_varying_linear_interpolation: str
    interpolate_boundary: str
    topology_valid: bool
    is_closed: bool
    surface_area: float
    volume: float
    bbox: BBox
    materials: List[str] = field(default_factory=list)
    subsets: List[Dict[str, Any]] = field(default_factory=list)


class MeshOperations(LoggerMixin):
    """
    USD mesh operations and analysis.
    
    Provides functionality to extract, analyze, and manipulate mesh data
    from USD mesh prims.
    """
    
    def __init__(self):
        """Initialize mesh operations."""
        self.logger.info("Mesh operations initialized")

    @staticmethod
    def _default_time_code() -> Any:
        """Return the USD default time code."""
        return Usd.TimeCode.Default()

    @staticmethod
    def _path_to_str(path_value: Any) -> str:
        """Normalise real USD paths and mocked path objects to plain strings."""
        return getattr(path_value, "pathString", str(path_value))
    
    @monitor_performance("mesh_ops.extract_mesh_data")
    def extract_mesh_data(
        self, mesh_prim: Usd.Prim, time_code: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Extract comprehensive mesh data from a USD mesh prim.
        
        Args:
            mesh_prim: USD mesh prim
            time_code: Time code for animated data
            
        Returns:
            Dictionary containing mesh data
        """
        if not mesh_prim.IsA(UsdGeom.Mesh):
            raise ValueError(f"Prim {mesh_prim.GetPath()} is not a mesh")
        
        mesh = UsdGeom.Mesh(mesh_prim)
        time_code = self._default_time_code() if time_code is None else time_code
        
        try:
            # Get basic mesh data
            points = mesh.GetPointsAttr().Get(time_code)
            face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get(time_code)
            face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get(time_code)
            
            # Convert to numpy arrays for easier manipulation
            points_array = np.array(points) if points else np.array([])
            face_counts_array = np.array(face_vertex_counts) if face_vertex_counts else np.array([])
            face_indices_array = np.array(face_vertex_indices) if face_vertex_indices else np.array([])
            
            # Get normals
            normals = None
            normals_attr = mesh.GetNormalsAttr()
            if normals_attr and normals_attr.Get(time_code):
                normals = np.array(normals_attr.Get(time_code))
            
            # Get UVs (texture coordinates)
            uvs = None
            primvars = UsdGeom.PrimvarsAPI(mesh.GetPrim()).GetPrimvars()
            for primvar in primvars:
                if primvar.GetPrimvarName() in ['st', 'uv', 'texCoord']:
                    uv_data = primvar.Get(time_code)
                    if uv_data:
                        uvs = np.array(uv_data)
                        break
            
            # Get vertex colors
            colors = None
            for primvar in primvars:
                if primvar.GetPrimvarName() in ['displayColor', 'color']:
                    color_data = primvar.Get(time_code)
                    if color_data:
                        colors = np.array(color_data)
                        break
            
            # Get subdivision scheme
            subdivision_scheme = mesh.GetSubdivisionSchemeAttr().Get()
            if not subdivision_scheme:
                subdivision_scheme = "none"
            
            # Get interpolation settings
            face_varying_linear_interpolation = mesh.GetFaceVaryingLinearInterpolationAttr().Get()
            interpolate_boundary = mesh.GetInterpolateBoundaryAttr().Get()
            
            mesh_data = {
                'prim_path': self._path_to_str(mesh_prim.GetPath()),
                'points': points_array.tolist() if points_array.size > 0 else [],
                'face_vertex_counts': face_counts_array.tolist() if face_counts_array.size > 0 else [],
                'face_vertex_indices': face_indices_array.tolist() if face_indices_array.size > 0 else [],
                'normals': normals.tolist() if normals is not None else None,
                'uvs': uvs.tolist() if uvs is not None else None,
                'colors': colors.tolist() if colors is not None else None,
                'subdivision_scheme': subdivision_scheme,
                'face_varying_linear_interpolation': face_varying_linear_interpolation,
                'interpolate_boundary': interpolate_boundary,
                'vertex_count': len(points_array) if points_array.size > 0 else 0,
                'face_count': len(face_counts_array) if face_counts_array.size > 0 else 0,
            }
            
            self.logger.debug(f"Extracted mesh data for {mesh_prim.GetPath()}: "
                            f"{mesh_data['vertex_count']} vertices, {mesh_data['face_count']} faces")
            
            return mesh_data
            
        except Exception as e:
            self.logger.error(f"Error extracting mesh data from {mesh_prim.GetPath()}: {e}")
            raise
    
    @monitor_performance("mesh_ops.get_mesh_statistics")
    def get_mesh_statistics(
        self, mesh_prim: Usd.Prim, time_code: Optional[Any] = None
    ) -> MeshInfo:
        """
        Get comprehensive statistics about a mesh.
        
        Args:
            mesh_prim: USD mesh prim
            time_code: Time code for animated data
            
        Returns:
            MeshInfo with detailed statistics
        """
        if not mesh_prim.IsA(UsdGeom.Mesh):
            raise ValueError(f"Prim {mesh_prim.GetPath()} is not a mesh")
        
        mesh = UsdGeom.Mesh(mesh_prim)
        time_code = self._default_time_code() if time_code is None else time_code
        
        try:
            # Extract basic mesh data
            mesh_data = self.extract_mesh_data(mesh_prim, time_code)
            
            points = np.array(mesh_data['points'])
            face_counts = np.array(mesh_data['face_vertex_counts'])
            face_indices = np.array(mesh_data['face_vertex_indices'])
            
            # Calculate statistics
            vertex_count = len(points)
            face_count = len(face_counts)
            point_count = len(face_indices)
            
            has_normals = mesh_data['normals'] is not None
            has_uvs = mesh_data['uvs'] is not None
            has_colors = mesh_data['colors'] is not None
            has_tangents = self._has_tangents(mesh)
            
            # Validate topology
            topology_valid = self._validate_mesh_topology(face_counts, face_indices, vertex_count)
            
            # Check if mesh is closed (watertight)
            is_closed = self._is_mesh_closed(points, face_counts, face_indices)
            
            # Calculate surface area and volume
            surface_area = self._calculate_surface_area(points, face_counts, face_indices)
            volume = self._calculate_volume(points, face_counts, face_indices) if is_closed else 0.0
            
            # Calculate bounding box
            bbox = bbox_from_points(points.tolist()) if len(points) > 0 else (([0, 0, 0], [0, 0, 0]))
            
            # Get materials
            materials = self._get_mesh_materials(mesh_prim)
            
            # Get geometry subsets
            subsets = self._get_geometry_subsets(mesh_prim)
            
            mesh_info = MeshInfo(
                prim_path=self._path_to_str(mesh_prim.GetPath()),
                vertex_count=vertex_count,
                face_count=face_count,
                point_count=point_count,
                has_normals=has_normals,
                has_uvs=has_uvs,
                has_colors=has_colors,
                has_tangents=has_tangents,
                subdivision_scheme=mesh_data['subdivision_scheme'],
                face_varying_linear_interpolation=mesh_data['face_varying_linear_interpolation'],
                interpolate_boundary=mesh_data['interpolate_boundary'],
                topology_valid=topology_valid,
                is_closed=is_closed,
                surface_area=surface_area,
                volume=volume,
                bbox=bbox,
                materials=materials,
                subsets=subsets
            )
            
            self.logger.debug(f"Generated mesh statistics for {mesh_prim.GetPath()}")
            return mesh_info
            
        except Exception as e:
            self.logger.error(f"Error generating mesh statistics for {mesh_prim.GetPath()}: {e}")
            raise
    
    def _validate_mesh_topology(self, face_counts: np.ndarray, face_indices: np.ndarray, vertex_count: int) -> bool:
        """Validate mesh topology."""
        try:
            # Check if face indices are within valid range
            if len(face_indices) > 0:
                if np.min(face_indices) < 0 or np.max(face_indices) >= vertex_count:
                    return False
            
            # Check if face counts sum matches face indices length
            if len(face_counts) > 0 and np.sum(face_counts) != len(face_indices):
                return False
            
            # Check for degenerate faces (faces with less than 3 vertices)
            if len(face_counts) > 0 and np.min(face_counts) < 3:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _is_mesh_closed(self, points: np.ndarray, face_counts: np.ndarray, face_indices: np.ndarray) -> bool:
        """Check if mesh is closed (watertight)."""
        # This is a simplified check - a full implementation would need edge analysis
        try:
            if len(points) == 0 or len(face_counts) == 0:
                return False
            
            # For now, just check if we have a reasonable number of faces relative to vertices
            # A proper implementation would check for boundary edges
            return len(face_counts) >= len(points) // 2
            
        except Exception:
            return False
    
    def _calculate_surface_area(self, points: np.ndarray, face_counts: np.ndarray, face_indices: np.ndarray) -> float:
        """Calculate mesh surface area."""
        try:
            if len(points) == 0 or len(face_counts) == 0:
                return 0.0
            
            total_area = 0.0
            index_offset = 0
            
            for face_count in face_counts:
                if face_count >= 3:
                    # Get face vertices
                    face_verts = []
                    for i in range(face_count):
                        vert_idx = face_indices[index_offset + i]
                        face_verts.append(points[vert_idx])
                    
                    # Calculate area for triangulated face
                    if face_count == 3:
                        # Triangle area
                        v0, v1, v2 = face_verts[0], face_verts[1], face_verts[2]
                        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
                        total_area += area
                    else:
                        # Triangulate polygon and sum areas
                        for i in range(1, face_count - 1):
                            v0, v1, v2 = face_verts[0], face_verts[i], face_verts[i + 1]
                            area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
                            total_area += area
                
                index_offset += face_count
            
            return float(total_area)
            
        except Exception:
            return 0.0
    
    def _calculate_volume(self, points: np.ndarray, face_counts: np.ndarray, face_indices: np.ndarray) -> float:
        """Calculate mesh volume (for closed meshes)."""
        try:
            if len(points) == 0 or len(face_counts) == 0:
                return 0.0
            
            # Use divergence theorem to calculate volume
            total_volume = 0.0
            index_offset = 0
            
            for face_count in face_counts:
                if face_count >= 3:
                    # Get face vertices
                    face_verts = []
                    for i in range(face_count):
                        vert_idx = face_indices[index_offset + i]
                        face_verts.append(points[vert_idx])
                    
                    # Triangulate and calculate volume contribution
                    for i in range(1, face_count - 1):
                        v0, v1, v2 = face_verts[0], face_verts[i], face_verts[i + 1]
                        # Volume of tetrahedron formed by origin and triangle
                        volume = np.dot(v0, np.cross(v1, v2)) / 6.0
                        total_volume += volume
                
                index_offset += face_count
            
            return abs(float(total_volume))
            
        except Exception:
            return 0.0
    
    def _has_tangents(self, mesh: UsdGeom.Mesh) -> bool:
        """Check if a mesh has tangent data."""
        try:
            tangent_names = {"tangent", "tangents", "tangentVectors"}
            for primvar in UsdGeom.PrimvarsAPI(mesh.GetPrim()).GetPrimvars():
                if primvar.GetPrimvarName() in tangent_names:
                    return True
            return False
        except Exception:
            return False

    def _get_mesh_materials(self, mesh_prim: Usd.Prim) -> List[str]:
        """Get materials bound to a mesh."""
        materials = []

        try:
            # Check for material bindings
            if hasattr(UsdGeom, 'MaterialBindingAPI'):
                binding_api = UsdGeom.MaterialBindingAPI(mesh_prim)
                if binding_api:
                    material_rel = binding_api.GetDirectBindingRel()
                    if material_rel:
                        targets = material_rel.GetTargets()
                        materials = [str(target) for target in targets]
        except Exception as e:
            self.logger.debug(f"Could not get materials for {mesh_prim.GetPath()}: {e}")
        
        return materials
    
    def _get_geometry_subsets(self, mesh_prim: Usd.Prim) -> List[Dict[str, Any]]:
        """Get geometry subsets for the mesh."""
        subsets = []
        try:
            # Look for GeomSubset children
            for child in mesh_prim.GetChildren():
                if child.IsA(UsdGeom.Subset):
                    subset = UsdGeom.Subset(child)
                    subset_info = {
                        'name': child.GetName(),
                        'path': self._path_to_str(child.GetPath()),
                        'element_type': subset.GetElementTypeAttr().Get(),
                        'family_name': subset.GetFamilyNameAttr().Get(),
                        'indices': list(subset.GetIndicesAttr().Get() or [])
                    }
                    subsets.append(subset_info)
        except Exception as e:
            self.logger.debug(f"Could not get geometry subsets for {mesh_prim.GetPath()}: {e}")
        
        return subsets


# Convenience functions
def extract_mesh_data(
    mesh_prim: Usd.Prim, time_code: Optional[Any] = None
) -> Dict[str, Any]:
    """Extract mesh data from a USD mesh prim."""
    ops = MeshOperations()
    return ops.extract_mesh_data(mesh_prim, time_code)


def get_mesh_statistics(
    mesh_prim: Usd.Prim, time_code: Optional[Any] = None
) -> MeshInfo:
    """Get mesh statistics."""
    ops = MeshOperations()
    return ops.get_mesh_statistics(mesh_prim, time_code)


def validate_mesh_topology(mesh_prim: Usd.Prim) -> bool:
    """Validate mesh topology."""
    ops = MeshOperations()
    mesh_data = ops.extract_mesh_data(mesh_prim)
    face_counts = np.array(mesh_data['face_vertex_counts'])
    face_indices = np.array(mesh_data['face_vertex_indices'])
    vertex_count = mesh_data['vertex_count']
    return ops._validate_mesh_topology(face_counts, face_indices, vertex_count)


def compute_mesh_normals(mesh_prim: Usd.Prim) -> bool:
    """Compute mesh normals and store on the prim."""
    try:
        mesh = UsdGeom.Mesh(mesh_prim)
        points = mesh.GetPointsAttr().Get()
        face_counts = mesh.GetFaceVertexCountsAttr().Get()
        face_indices = mesh.GetFaceVertexIndicesAttr().Get()

        if not points or not face_counts or not face_indices:
            logger.error(f"Mesh data missing for {mesh_prim.GetPath()}")
            return False

        points_array = np.array([[p[0], p[1], p[2]] for p in points], dtype=float)
        normals = np.zeros_like(points_array)

        index_offset = 0
        for count in face_counts:
            if count < 3:
                index_offset += count
                continue
            indices = face_indices[index_offset:index_offset + count]
            v0 = points_array[indices[0]]
            v1 = points_array[indices[1]]
            v2 = points_array[indices[2]]
            face_normal = np.cross(v1 - v0, v2 - v0)
            length = np.linalg.norm(face_normal)
            if length > 0:
                face_normal = face_normal / length
                for idx in indices:
                    normals[idx] += face_normal
            index_offset += count

        for i in range(len(normals)):
            length = np.linalg.norm(normals[i])
            if length > 0:
                normals[i] = normals[i] / length

        mesh.GetNormalsAttr().Set([tuple(n) for n in normals.tolist()])
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        logger.info(f"Computed normals for {mesh_prim.GetPath()}")
        return True
    except Exception as e:
        logger.error(f"Failed to compute normals for {mesh_prim.GetPath()}: {e}")
        return False


def decimate_mesh(mesh_prim: Usd.Prim, target_faces: int) -> bool:
    """Naively decimate a mesh by truncating faces."""
    try:
        mesh = UsdGeom.Mesh(mesh_prim)
        face_counts = list(mesh.GetFaceVertexCountsAttr().Get())
        face_indices = list(mesh.GetFaceVertexIndicesAttr().Get())
        total_faces = len(face_counts)

        if target_faces <= 0:
            logger.error(f"Target face count must be positive: {target_faces}")
            return False

        if target_faces >= total_faces:
            logger.info(f"Mesh {mesh_prim.GetPath()} already <= target faces")
            return True

        trimmed_counts = face_counts[:target_faces]
        trimmed_indices = face_indices[:sum(trimmed_counts)]

        mesh.GetFaceVertexCountsAttr().Set(trimmed_counts)
        mesh.GetFaceVertexIndicesAttr().Set(trimmed_indices)
        logger.info(f"Decimated mesh {mesh_prim.GetPath()} to {target_faces} faces")
        return True
    except Exception as e:
        logger.error(f"Failed to decimate mesh {mesh_prim.GetPath()}: {e}")
        return False


def export_mesh_data(mesh_prim: Usd.Prim, output_path: str, format: str = "obj") -> bool:
    """Export mesh data to an OBJ file."""
    if format.lower() != "obj":
        logger.error(f"Unsupported export format: {format}")
        return False

    try:
        mesh = UsdGeom.Mesh(mesh_prim)
        points = mesh.GetPointsAttr().Get()
        face_counts = mesh.GetFaceVertexCountsAttr().Get()
        face_indices = mesh.GetFaceVertexIndicesAttr().Get()

        if not points or not face_counts or not face_indices:
            logger.error(f"Mesh data missing for {mesh_prim.GetPath()}")
            return False

        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with output_path_obj.open("w", encoding="utf-8") as handle:
            for point in points:
                handle.write(f"v {point[0]} {point[1]} {point[2]}\n")

            index_offset = 0
            for count in face_counts:
                indices = face_indices[index_offset:index_offset + count]
                if indices:
                    face = " ".join(str(idx + 1) for idx in indices)
                    handle.write(f"f {face}\n")
                index_offset += count

        logger.info(f"Exported mesh {mesh_prim.GetPath()} to {output_path_obj}")
        return True
    except Exception as e:
        logger.error(f"Failed to export mesh {mesh_prim.GetPath()}: {e}")
        return False
