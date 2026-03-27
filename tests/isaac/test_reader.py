"""
Tests for USD reader functionality.

This module contains tests for the USD reader components including
stage loading, prim traversal, and information extraction.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Test imports
import sys

src_path = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(src_path))

from simul_mcp.usd.reader import (
    PXR_AVAILABLE as READER_PXR_AVAILABLE,
    PrimType,
    USDLayerInfo,
    USDPrimInfo,
    USDReader,
    USDStageInfo,
)


@pytest.mark.skipif(not READER_PXR_AVAILABLE, reason="pxr library not available")
class TestUSDReader:
    """Test cases for USDReader class."""

    @pytest.fixture
    def mock_usd_stage(self):
        """Create a mock USD stage for testing."""
        mock_stage = Mock()
        mock_stage.GetRootLayer.return_value.identifier = "/test/path.usd"
        mock_stage.GetUpAxis.return_value = "Y"
        mock_stage.GetMetersPerUnit.return_value = 1.0
        mock_stage.GetTimeCodesPerSecond.return_value = 24.0
        mock_stage.GetStartTimeCode.return_value = 1.0
        mock_stage.GetEndTimeCode.return_value = 100.0
        mock_stage.GetFramesPerSecond.return_value = 24.0
        mock_stage.GetDefaultPrim.return_value = None

        # Mock layer stack
        mock_layer = Mock()
        mock_layer.identifier = "/test/path.usd"
        mock_stage.GetLayerStack.return_value = [mock_layer]

        return mock_stage

    @pytest.fixture
    def mock_usd_prim(self):
        """Create a mock USD prim for testing."""
        mock_prim = Mock()
        mock_prim.GetPath.return_value.pathString = "/World/Cube"
        mock_prim.GetName.return_value = "Cube"
        mock_prim.GetTypeName.return_value = "Mesh"
        mock_prim.IsActive.return_value = True
        mock_prim.IsLoaded.return_value = True
        mock_prim.IsDefined.return_value = True
        mock_prim.IsInstance.return_value = False
        mock_prim.GetChildren.return_value = []
        mock_prim.GetAttributes.return_value = []
        mock_prim.GetMetadata.return_value = {}

        return mock_prim

    @pytest.fixture
    def usd_reader(self):
        """Create USDReader instance for testing."""
        return USDReader(enable_caching=False)

    def test_init_without_pxr(self):
        """Test USDReader initialization without pxr library."""
        with patch("simul_mcp.usd.reader.PXR_AVAILABLE", False):
            with pytest.raises(ImportError, match="pxr library not available"):
                USDReader()

    @patch("simul_mcp.usd.reader.PXR_AVAILABLE", True)
    @patch("simul_mcp.usd.reader.Usd")
    def test_open_stage_success(self, mock_usd, usd_reader, mock_usd_stage):
        """Test successful stage opening."""
        mock_usd.Stage.Open.return_value = mock_usd_stage

        stage = usd_reader.open_stage("/test/path.usd")

        assert stage == mock_usd_stage
        mock_usd.Stage.Open.assert_called_once_with("/test/path.usd")

    @patch("simul_mcp.usd.reader.PXR_AVAILABLE", True)
    @patch("simul_mcp.usd.reader.Usd")
    def test_open_stage_failure(self, mock_usd, usd_reader):
        """Test stage opening failure."""
        mock_usd.Stage.Open.return_value = None

        stage = usd_reader.open_stage("/nonexistent/path.usd")

        assert stage is None

    @patch("simul_mcp.usd.reader.PXR_AVAILABLE", True)
    def test_get_stage_info(self, usd_reader, mock_usd_stage):
        """Test stage information extraction."""
        # Mock prim traversal
        mock_prim1 = Mock()
        mock_prim1.GetPath.return_value.pathString = "/World"
        mock_prim2 = Mock()
        mock_prim2.GetPath.return_value.pathString = "/World/Cube"

        mock_usd_stage.Traverse.return_value = [mock_prim1, mock_prim2]
        mock_usd_stage.GetPseudoRoot.return_value.GetChildren.return_value = [
            mock_prim1
        ]

        stage_info = usd_reader.get_stage_info(mock_usd_stage)

        assert isinstance(stage_info, USDStageInfo)
        assert stage_info.up_axis == "Y"
        assert stage_info.meters_per_unit == 1.0
        assert stage_info.time_codes_per_second == 24.0
        assert stage_info.start_time_code == 1.0
        assert stage_info.end_time_code == 100.0
        assert stage_info.frame_rate == 24.0
        assert len(stage_info.all_prims) == 2
        assert len(stage_info.root_prims) == 1
        assert len(stage_info.layers) == 1

    @patch("simul_mcp.usd.reader.PXR_AVAILABLE", True)
    def test_get_prim_info(self, usd_reader, mock_usd_prim):
        """Test prim information extraction."""
        prim_info = usd_reader.get_prim_info(mock_usd_prim)

        assert isinstance(prim_info, USDPrimInfo)
        assert prim_info.path == "/World/Cube"
        assert prim_info.name == "Cube"
        assert prim_info.type_name == "Mesh"
        assert prim_info.is_active is True
        assert prim_info.is_loaded is True
        assert prim_info.is_defined is True
        assert prim_info.is_instance is False
        assert len(prim_info.children) == 0

    @patch("simul_mcp.usd.reader.PXR_AVAILABLE", True)
    def test_find_prims_by_type(self, usd_reader, mock_usd_stage):
        """Test finding prims by type."""
        # Mock prims
        mesh_prim = Mock()
        mesh_prim.GetTypeName.return_value = "Mesh"

        xform_prim = Mock()
        xform_prim.GetTypeName.return_value = "Xform"

        mock_usd_stage.Traverse.return_value = [mesh_prim, xform_prim]

        mesh_prims = usd_reader.find_prims_by_type(mock_usd_stage, "Mesh")

        assert len(mesh_prims) == 1
        assert mesh_prims[0] == mesh_prim

    @patch("simul_mcp.usd.reader.PXR_AVAILABLE", True)
    def test_find_prims_by_name_exact(self, usd_reader, mock_usd_stage):
        """Test finding prims by exact name match."""
        # Mock prims
        cube_prim = Mock()
        cube_prim.GetName.return_value = "Cube"

        sphere_prim = Mock()
        sphere_prim.GetName.return_value = "Sphere"

        mock_usd_stage.Traverse.return_value = [cube_prim, sphere_prim]

        cube_prims = usd_reader.find_prims_by_name(
            mock_usd_stage, "Cube", exact_match=True
        )

        assert len(cube_prims) == 1
        assert cube_prims[0] == cube_prim

    @patch("simul_mcp.usd.reader.PXR_AVAILABLE", True)
    def test_find_prims_by_name_pattern(self, usd_reader, mock_usd_stage):
        """Test finding prims by name pattern."""
        # Mock prims
        cube1_prim = Mock()
        cube1_prim.GetName.return_value = "Cube_001"

        cube2_prim = Mock()
        cube2_prim.GetName.return_value = "Cube_002"

        sphere_prim = Mock()
        sphere_prim.GetName.return_value = "Sphere"

        mock_usd_stage.Traverse.return_value = [cube1_prim, cube2_prim, sphere_prim]

        cube_prims = usd_reader.find_prims_by_name(
            mock_usd_stage, "Cube", exact_match=False
        )

        assert len(cube_prims) == 2
        assert cube1_prim in cube_prims
        assert cube2_prim in cube_prims

    def test_classify_prim_type(self, usd_reader):
        """Test prim type classification."""
        assert usd_reader.classify_prim_type("Mesh") == PrimType.MESH
        assert usd_reader.classify_prim_type("Xform") == PrimType.XFORM
        assert usd_reader.classify_prim_type("Sphere") == PrimType.SPHERE
        assert usd_reader.classify_prim_type("Cube") == PrimType.CUBE
        assert usd_reader.classify_prim_type("Camera") == PrimType.CAMERA
        assert usd_reader.classify_prim_type("Light") == PrimType.LIGHT
        assert usd_reader.classify_prim_type("Material") == PrimType.MATERIAL
        assert usd_reader.classify_prim_type("UnknownType") == PrimType.UNKNOWN

    def test_caching_enabled(self):
        """Test reader with caching enabled."""
        reader = USDReader(enable_caching=True)
        assert reader.cache_enabled is True
        assert reader._stage_cache is not None
        assert reader._prim_cache is not None

    def test_caching_disabled(self):
        """Test reader with caching disabled."""
        reader = USDReader(enable_caching=False)
        assert reader.cache_enabled is False
        assert reader._stage_cache is None
        assert reader._prim_cache is None

    def test_clear_cache(self, usd_reader):
        """Test cache clearing."""
        # Enable caching for this test
        usd_reader.cache_enabled = True
        usd_reader._stage_cache = {"test": "value"}
        usd_reader._prim_cache = {"test": "value"}

        usd_reader.clear_cache()

        assert len(usd_reader._stage_cache) == 0
        assert len(usd_reader._prim_cache) == 0


class TestUSDStageInfo:
    """Test cases for USDStageInfo dataclass."""

    def test_stage_info_creation(self):
        """Test USDStageInfo creation."""
        layer = USDLayerInfo(
            identifier="/test/path.usd",
            display_name="path.usd",
            real_path="/test/path.usd",
            file_format="usd",
            version="1.0",
            comment="",
            documentation="",
            is_anonymous=False,
            is_dirty=False,
            permission="edit",
        )
        stage_info = USDStageInfo(
            root_layer_path="/test/path.usd",
            session_layer_path=None,
            layers=[layer],
            default_prim_path="/World",
            up_axis="Y",
            meters_per_unit=1.0,
            time_codes_per_second=24.0,
            start_time_code=1.0,
            end_time_code=100.0,
            frame_rate=24.0,
            prim_count=2,
            root_prims=["/World"],
            stage_variables={},
        )

        assert stage_info.up_axis == "Y"
        assert stage_info.meters_per_unit == 1.0
        assert stage_info.time_codes_per_second == 24.0
        assert stage_info.start_time_code == 1.0
        assert stage_info.end_time_code == 100.0
        assert stage_info.frame_rate == 24.0
        assert stage_info.prim_count == 2
        assert len(stage_info.root_prims) == 1
        assert len(stage_info.layers) == 1
        assert stage_info.default_prim_path == "/World"


class TestUSDPrimInfo:
    """Test cases for USDPrimInfo dataclass."""

    def test_prim_info_creation(self):
        """Test USDPrimInfo creation."""
        prim_info = USDPrimInfo(
            path="/World/Cube",
            name="Cube",
            type_name="Mesh",
            prim_type=PrimType.MESH,
            is_active=True,
            is_loaded=True,
            is_defined=True,
            is_abstract=False,
            is_instance=False,
            is_prototype=False,
            has_authored_references=False,
            has_authored_payloads=False,
            has_authored_inherits=False,
            has_authored_specializes=False,
            purpose="default",
            visibility="inherited",
            kind="component",
            children=["/World/Cube/Material"],
            attributes={"points": "array"},
            metadata={"comment": "Test cube"},
            relationships={},
            parent="/World",
        )

        assert prim_info.path == "/World/Cube"
        assert prim_info.name == "Cube"
        assert prim_info.type_name == "Mesh"
        assert prim_info.prim_type == PrimType.MESH
        assert prim_info.is_active is True
        assert prim_info.is_loaded is True
        assert prim_info.is_defined is True
        assert prim_info.is_instance is False
        assert prim_info.purpose == "default"
        assert prim_info.visibility == "inherited"
        assert prim_info.kind == "component"
        assert len(prim_info.children) == 1
        assert "points" in prim_info.attributes
        assert "comment" in prim_info.metadata


# Integration tests (require actual USD files)
@pytest.mark.skipif(not READER_PXR_AVAILABLE, reason="pxr library not available")
class TestUSDReaderIntegration:
    """Integration tests for USDReader with real USD files."""

    @pytest.fixture
    def temp_usd_file(self):
        """Create a temporary USD file for testing."""
        # This would create a simple USD file for testing
        # For now, skip if pxr is not available
        pytest.skip("Integration tests require pxr library and test USD files")

    def test_load_real_usd_file(self, temp_usd_file):
        """Test loading a real USD file."""
        reader = USDReader()
        stage = reader.open_stage(temp_usd_file)

        assert stage is not None

        stage_info = reader.get_stage_info(stage)
        assert isinstance(stage_info, USDStageInfo)
        assert len(stage_info.all_prims) > 0


if __name__ == "__main__":
    pytest.main([__file__])
