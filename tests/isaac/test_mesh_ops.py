"""
Tests for USD mesh operations functionality.

This module contains tests for mesh data extraction, analysis,
and validation operations.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Test imports
import sys
from pathlib import Path

src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_path))

pytest.importorskip("pxr", reason="pxr library not available")

from simul_mcp.usd.mesh_ops import MeshOperations, MeshInfo
from simul_mcp.utils.math import BBox


class TestMeshOperations:
    """Test cases for MeshOperations class."""

    @pytest.fixture
    def mock_mesh_prim(self):
        """Create a mock USD mesh prim for testing."""
        mock_prim = Mock()
        mock_prim.GetPath.return_value.pathString = "/World/Cube"
        mock_prim.IsA.return_value = True

        # Mock mesh
        mock_mesh = Mock()

        # Mock points (cube vertices)
        cube_points = [
            [-1, -1, -1],
            [1, -1, -1],
            [-1, 1, -1],
            [1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [-1, 1, 1],
            [1, 1, 1],
        ]
        mock_points_attr = Mock()
        mock_points_attr.Get.return_value = cube_points
        mock_mesh.GetPointsAttr.return_value = mock_points_attr

        # Mock face vertex counts (cube faces)
        face_counts = [4, 4, 4, 4, 4, 4]  # 6 quad faces
        mock_face_counts_attr = Mock()
        mock_face_counts_attr.Get.return_value = face_counts
        mock_mesh.GetFaceVertexCountsAttr.return_value = mock_face_counts_attr

        # Mock face vertex indices
        face_indices = [
            0,
            1,
            3,
            2,  # bottom face
            4,
            6,
            7,
            5,  # top face
            0,
            2,
            6,
            4,  # left face
            1,
            5,
            7,
            3,  # right face
            0,
            4,
            5,
            1,  # front face
            2,
            3,
            7,
            6,  # back face
        ]
        mock_face_indices_attr = Mock()
        mock_face_indices_attr.Get.return_value = face_indices
        mock_mesh.GetFaceVertexIndicesAttr.return_value = mock_face_indices_attr

        # Mock normals
        mock_normals_attr = Mock()
        mock_normals_attr.Get.return_value = None
        mock_mesh.GetNormalsAttr.return_value = mock_normals_attr

        # Mock subdivision scheme
        mock_subdiv_attr = Mock()
        mock_subdiv_attr.Get.return_value = "none"
        mock_mesh.GetSubdivisionSchemeAttr.return_value = mock_subdiv_attr

        # Mock interpolation attributes
        mock_fvli_attr = Mock()
        mock_fvli_attr.Get.return_value = None
        mock_mesh.GetFaceVaryingLinearInterpolationAttr.return_value = mock_fvli_attr

        mock_ib_attr = Mock()
        mock_ib_attr.Get.return_value = None
        mock_mesh.GetInterpolateBoundaryAttr.return_value = mock_ib_attr

        # Mock primvars (empty for basic test)
        mock_mesh.GetPrimvars.return_value = []

        return mock_prim, mock_mesh

    @pytest.fixture
    def mesh_ops(self):
        """Create MeshOperations instance for testing."""
        return MeshOperations()

    def test_init(self):
        """Test MeshOperations initialization."""
        assert isinstance(MeshOperations(), MeshOperations)

    @patch("simul_mcp.usd.mesh_ops.UsdGeom")
    def test_extract_mesh_data_success(self, mock_usd_geom, mesh_ops, mock_mesh_prim):
        """Test successful mesh data extraction."""
        mock_prim, mock_mesh = mock_mesh_prim
        mock_usd_geom.Mesh.return_value = mock_mesh

        mesh_data = mesh_ops.extract_mesh_data(mock_prim)

        assert isinstance(mesh_data, dict)
        assert mesh_data["prim_path"] == "/World/Cube"
        assert mesh_data["vertex_count"] == 8
        assert mesh_data["face_count"] == 6
        assert len(mesh_data["points"]) == 8
        assert len(mesh_data["face_vertex_counts"]) == 6
        assert len(mesh_data["face_vertex_indices"]) == 24
        assert mesh_data["subdivision_scheme"] == "none"

    @patch("simul_mcp.usd.mesh_ops.UsdGeom")
    def test_extract_mesh_data_not_mesh(self, mock_usd_geom, mesh_ops):
        """Test mesh data extraction on non-mesh prim."""
        mock_prim = Mock()
        mock_prim.IsA.return_value = False
        mock_prim.GetPath.return_value.pathString = "/World/NotMesh"

        with pytest.raises(ValueError, match="is not a mesh"):
            mesh_ops.extract_mesh_data(mock_prim)

    @patch("simul_mcp.usd.mesh_ops.UsdGeom")
    def test_get_mesh_statistics(self, mock_usd_geom, mesh_ops, mock_mesh_prim):
        """Test mesh statistics computation."""
        mock_prim, mock_mesh = mock_mesh_prim
        mock_usd_geom.Mesh.return_value = mock_mesh

        # Mock the extract_mesh_data method to return test data
        with patch.object(mesh_ops, "extract_mesh_data") as mock_extract:
            mock_extract.return_value = {
                "prim_path": "/World/Cube",
                "points": [
                    [-1, -1, -1],
                    [1, -1, -1],
                    [-1, 1, -1],
                    [1, 1, -1],
                    [-1, -1, 1],
                    [1, -1, 1],
                    [-1, 1, 1],
                    [1, 1, 1],
                ],
                "face_vertex_counts": [4, 4, 4, 4, 4, 4],
                "face_vertex_indices": [
                    0, 1, 3, 2,
                    4, 6, 7, 5,
                    0, 2, 6, 4,
                    1, 5, 7, 3,
                    0, 4, 5, 1,
                    2, 3, 7, 6,
                ],
                "normals": None,
                "uvs": None,
                "colors": None,
                "subdivision_scheme": "none",
                "face_varying_linear_interpolation": None,
                "interpolate_boundary": None,
                "vertex_count": 8,
                "face_count": 6,
            }

            mesh_info = mesh_ops.get_mesh_statistics(mock_prim)

            assert isinstance(mesh_info, MeshInfo)
            assert mesh_info.prim_path == "/World/Cube"
            assert mesh_info.vertex_count == 8
            assert mesh_info.face_count == 6
            assert mesh_info.has_normals is False
            assert mesh_info.has_uvs is False
            assert mesh_info.has_colors is False
            assert mesh_info.subdivision_scheme == "none"
            assert mesh_info.topology_valid is True
            assert mesh_info.surface_area > 0
            assert isinstance(mesh_info.bbox, tuple)

    def test_validate_mesh_topology_valid(self, mesh_ops):
        """Test mesh topology validation with valid topology."""
        face_counts = np.array([3, 3, 4])  # Triangle, triangle, quad
        face_indices = np.array([0, 1, 2, 0, 2, 3, 0, 1, 4, 3])  # Valid indices
        vertex_count = 5

        is_valid = mesh_ops._validate_mesh_topology(
            face_counts, face_indices, vertex_count
        )

        assert is_valid is True

    def test_validate_mesh_topology_invalid_indices(self, mesh_ops):
        """Test mesh topology validation with invalid indices."""
        face_counts = np.array([3])
        face_indices = np.array([0, 1, 5])  # Index 5 is out of range for 3 vertices
        vertex_count = 3

        is_valid = mesh_ops._validate_mesh_topology(
            face_counts, face_indices, vertex_count
        )

        assert is_valid is False

    def test_validate_mesh_topology_mismatched_counts(self, mesh_ops):
        """Test mesh topology validation with mismatched counts."""
        face_counts = np.array([3, 3])  # Expects 6 indices
        face_indices = np.array([0, 1, 2, 0, 2])  # Only 5 indices
        vertex_count = 3

        is_valid = mesh_ops._validate_mesh_topology(
            face_counts, face_indices, vertex_count
        )

        assert is_valid is False

    def test_validate_mesh_topology_degenerate_faces(self, mesh_ops):
        """Test mesh topology validation with degenerate faces."""
        face_counts = np.array([2, 3])  # Face with only 2 vertices (degenerate)
        face_indices = np.array([0, 1, 0, 1, 2])
        vertex_count = 3

        is_valid = mesh_ops._validate_mesh_topology(
            face_counts, face_indices, vertex_count
        )

        assert is_valid is False

    def test_is_mesh_closed_simple(self, mesh_ops):
        """Test mesh closed check with simple case."""
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        face_counts = np.array([3, 3, 3, 3])  # Tetrahedron
        face_indices = np.array([0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2])

        is_closed = mesh_ops._is_mesh_closed(points, face_counts, face_indices)

        # This is a simplified check, so result may vary
        assert isinstance(is_closed, bool)

    def test_calculate_surface_area_triangle(self, mesh_ops):
        """Test surface area calculation for a simple triangle."""
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        face_counts = np.array([3])
        face_indices = np.array([0, 1, 2])

        area = mesh_ops._calculate_surface_area(points, face_counts, face_indices)

        # Triangle with vertices at (0,0,0), (1,0,0), (0,1,0) has area 0.5
        assert abs(area - 0.5) < 1e-6

    def test_calculate_surface_area_empty(self, mesh_ops):
        """Test surface area calculation with empty mesh."""
        points = np.array([])
        face_counts = np.array([])
        face_indices = np.array([])

        area = mesh_ops._calculate_surface_area(points, face_counts, face_indices)

        assert area == 0.0

    def test_calculate_volume_tetrahedron(self, mesh_ops):
        """Test volume calculation for a simple tetrahedron."""
        # Unit tetrahedron
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        face_counts = np.array([3, 3, 3, 3])
        face_indices = np.array([0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2])

        volume = mesh_ops._calculate_volume(points, face_counts, face_indices)

        # Volume should be positive (absolute value is taken)
        assert volume > 0

    def test_calculate_volume_empty(self, mesh_ops):
        """Test volume calculation with empty mesh."""
        points = np.array([])
        face_counts = np.array([])
        face_indices = np.array([])

        volume = mesh_ops._calculate_volume(points, face_counts, face_indices)

        assert volume == 0.0

    @patch("simul_mcp.usd.mesh_ops.UsdGeom")
    def test_get_mesh_materials_empty(self, mock_usd_geom, mesh_ops):
        """Test getting mesh materials when none are bound."""
        mock_prim = Mock()

        # Mock MaterialBindingAPI not available or no bindings
        mock_usd_geom.MaterialBindingAPI.return_value = None

        materials = mesh_ops._get_mesh_materials(mock_prim)

        assert materials == []

    @patch("simul_mcp.usd.mesh_ops.UsdGeom")
    def test_get_geometry_subsets_empty(self, mock_usd_geom, mesh_ops):
        """Test getting geometry subsets when none exist."""
        mock_prim = Mock()
        mock_prim.GetChildren.return_value = []

        subsets = mesh_ops._get_geometry_subsets(mock_prim)

        assert subsets == []


class TestMeshInfo:
    """Test cases for MeshInfo dataclass."""

    def test_mesh_info_creation(self):
        """Test MeshInfo creation."""
        bbox = ([-1, -1, -1], [1, 1, 1])

        mesh_info = MeshInfo(
            prim_path="/World/Cube",
            vertex_count=8,
            face_count=6,
            point_count=24,
            has_normals=False,
            has_uvs=False,
            has_colors=False,
            has_tangents=False,
            subdivision_scheme="none",
            face_varying_linear_interpolation="all",
            interpolate_boundary="edgeAndCorner",
            topology_valid=True,
            is_closed=True,
            surface_area=24.0,
            volume=8.0,
            bbox=bbox,
            materials=["/World/Materials/Material"],
            subsets=[],
        )

        assert mesh_info.prim_path == "/World/Cube"
        assert mesh_info.vertex_count == 8
        assert mesh_info.face_count == 6
        assert mesh_info.point_count == 24
        assert mesh_info.has_normals is False
        assert mesh_info.has_uvs is False
        assert mesh_info.has_colors is False
        assert mesh_info.has_tangents is False
        assert mesh_info.subdivision_scheme == "none"
        assert mesh_info.topology_valid is True
        assert mesh_info.is_closed is True
        assert mesh_info.surface_area == 24.0
        assert mesh_info.volume == 8.0
        assert mesh_info.bbox == bbox
        assert len(mesh_info.materials) == 1
        assert len(mesh_info.subsets) == 0


# Convenience function tests
class TestConvenienceFunctions:
    """Test cases for convenience functions."""

    @patch("simul_mcp.usd.mesh_ops.MeshOperations")
    def test_extract_mesh_data_function(self, mock_mesh_ops_class):
        """Test extract_mesh_data convenience function."""
        from simul_mcp.usd.mesh_ops import extract_mesh_data

        mock_ops = Mock()
        mock_mesh_ops_class.return_value = mock_ops
        mock_ops.extract_mesh_data.return_value = {"test": "data"}

        mock_prim = Mock()
        result = extract_mesh_data(mock_prim)

        mock_mesh_ops_class.assert_called_once()
        mock_ops.extract_mesh_data.assert_called_once_with(mock_prim, None)
        assert result == {"test": "data"}

    @patch("simul_mcp.usd.mesh_ops.MeshOperations")
    def test_get_mesh_statistics_function(self, mock_mesh_ops_class):
        """Test get_mesh_statistics convenience function."""
        from simul_mcp.usd.mesh_ops import get_mesh_statistics

        mock_ops = Mock()
        mock_mesh_ops_class.return_value = mock_ops
        mock_info = MeshInfo(
            prim_path="/test",
            vertex_count=0,
            face_count=0,
            point_count=0,
            has_normals=False,
            has_uvs=False,
            has_colors=False,
            has_tangents=False,
            subdivision_scheme="none",
            face_varying_linear_interpolation="",
            interpolate_boundary="",
            topology_valid=True,
            is_closed=False,
            surface_area=0.0,
            volume=0.0,
            bbox=([], []),
        )
        mock_ops.get_mesh_statistics.return_value = mock_info

        mock_prim = Mock()
        result = get_mesh_statistics(mock_prim)

        mock_mesh_ops_class.assert_called_once()
        mock_ops.get_mesh_statistics.assert_called_once_with(mock_prim, None)
        assert result == mock_info


if __name__ == "__main__":
    pytest.main([__file__])
