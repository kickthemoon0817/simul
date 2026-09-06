"""Tests for Blender runtime adapter functionality."""

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest


from simul_mcp.adapters import blender_runtime
from simul_mcp.config import SecurityConfig, Settings


class FakeVector:
    """Minimal 3-element vector that supports iteration and indexing."""

    def __init__(self, values: tuple) -> None:
        self._values = tuple(float(v) for v in values)

    def __iter__(self):  # type: ignore[override]
        return iter(self._values)

    def __getitem__(self, idx: int) -> float:
        return self._values[idx]

    def __len__(self) -> int:
        return len(self._values)


class FakeMatrix:
    """Minimal 4×4 matrix mock with translation and matmul support."""

    def __init__(self, translation: tuple = (0.0, 0.0, 0.0)) -> None:
        self.translation = FakeVector(translation)

    def __matmul__(self, other: Any) -> list:
        """Identity-like transform: offset each corner by translation."""
        t = self.translation
        return [other[0] + t[0], other[1] + t[1], other[2] + t[2]]


class FakeObject:
    """Simple fake Blender object for tests."""

    def __init__(
        self,
        name: str,
        object_type: str,
        visible: bool,
        location: tuple = (0.0, 0.0, 0.0),
        rotation_euler: tuple = (0.0, 0.0, 0.0),
        scale: tuple = (1.0, 1.0, 1.0),
        parent: Optional[Any] = None,
        children: Optional[list] = None,
        modifiers: Optional[list] = None,
        constraints: Optional[list] = None,
        material_slots: Optional[list] = None,
        matrix_world: Optional[FakeMatrix] = None,
        bound_box: Optional[list] = None,
        data: Optional[Any] = None,
    ) -> None:
        self.name = name
        self.type = object_type
        self._visible = visible
        self.hide_viewport = not visible
        self.location = FakeVector(location)
        self.rotation_euler = FakeVector(rotation_euler)
        self.scale = FakeVector(scale)
        self.parent = parent
        self.children = children or []
        self.modifiers = modifiers or []
        self.constraints = constraints or []
        self.material_slots = material_slots or []
        self.matrix_world = matrix_world or FakeMatrix(location)
        self.bound_box = bound_box or []
        self.data = data

    def visible_get(self) -> bool:
        """Mirror Blender visible_get behavior."""
        return self._visible


class TestBlenderRuntimeSession:
    """Test cases for BlenderRuntimeSession."""

    def test_session_raises_without_bpy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Session initialization fails when bpy is unavailable."""
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", False)

        with pytest.raises(ImportError, match="Blender runtime not available"):
            blender_runtime.BlenderRuntimeSession()

    def test_get_runtime_info_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Runtime info is returned from mocked bpy app/data."""
        fake_app = SimpleNamespace(
            version=(4, 1, 0),
            version_string="4.1.0",
            binary_path="/Applications/Blender.app/Contents/MacOS/Blender",
            background=False,
        )
        fake_data = SimpleNamespace(filepath="/tmp/test_scene.blend")
        fake_bpy = SimpleNamespace(app=fake_app, data=fake_data)

        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()
        runtime_info = session.get_runtime_info()

        assert runtime_info["version"] == [4, 1, 0]
        assert runtime_info["version_string"] == "4.1.0"
        assert runtime_info["binary_path"] == fake_app.binary_path
        assert runtime_info["background"] is False
        assert runtime_info["blend_file_path"] == "/tmp/test_scene.blend"

    def test_list_scene_objects_filters_hidden(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Object listing respects hidden filtering and limits."""
        visible_obj = FakeObject(name="Cube", object_type="MESH", visible=True)
        hidden_obj = FakeObject(name="Light", object_type="LIGHT", visible=False)
        fake_data = SimpleNamespace(
            filepath="",
            objects=[visible_obj, hidden_obj],
            collections={},
        )
        fake_bpy = SimpleNamespace(
            app=SimpleNamespace(
                version=(4, 1, 0),
                version_string="4.1.0",
                binary_path="/bin/blender",
                background=True,
            ),
            data=fake_data,
        )

        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.list_scene_objects(include_hidden=False, max_items=10)

        assert result["count"] == 1
        assert result["objects"][0]["name"] == "Cube"
        assert result["objects"][0]["visible"] is True

    def test_list_scene_objects_collection_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Unknown collections raise a validation error."""
        fake_data = SimpleNamespace(filepath="", objects=[], collections={})
        fake_bpy = SimpleNamespace(
            app=SimpleNamespace(
                version=(4, 1, 0),
                version_string="4.1.0",
                binary_path="/bin/blender",
                background=True,
            ),
            data=fake_data,
        )

        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()

        with pytest.raises(ValueError, match="Collection not found"):
            session.list_scene_objects(collection_name="Missing")


class TestBlenderRuntimeAdapter:
    """Test cases for BlenderRuntimeAdapter."""

    def test_is_blender_available_passthrough(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Availability helper returns module-level runtime flag."""
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", False)
        assert blender_runtime.is_blender_available() is False

        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        assert blender_runtime.is_blender_available() is True

    def test_create_session_returns_session(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Adapter create_session yields a runtime session instance."""
        fake_session: Any = object()

        class FakeSession:
            """Session replacement to isolate adapter logic."""

            def __init__(self, settings: Any = None):
                self.settings = settings

            def cleanup(self) -> None:
                return None

        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "BlenderRuntimeSession", FakeSession)

        adapter = blender_runtime.BlenderRuntimeAdapter()
        with adapter.create_session() as session:
            fake_session = session

        assert isinstance(fake_session, FakeSession)


class TestBlenderVersionDetection:
    """Test cases for Blender version detection flags."""

    @staticmethod
    def _make_fake_bpy(
        version: tuple,
    ) -> SimpleNamespace:
        """Build a minimal bpy mock with the given version tuple."""
        return SimpleNamespace(
            app=SimpleNamespace(
                version=version,
                version_string=".".join(str(v) for v in version),
                binary_path="/bin/blender",
                background=True,
            ),
            data=SimpleNamespace(filepath=""),
        )

    def test_version_36_flags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Blender 3.6.x sets both 4+ and 5+ flags to False."""
        fake_bpy = self._make_fake_bpy((3, 6, 5))
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()
        assert session.blender_version == (3, 6, 5)
        assert session.is_blender_4_plus is False
        assert session.is_blender_5_plus is False

    def test_version_40_flags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Blender 4.0.0 sets 4+ flag True and 5+ flag False."""
        fake_bpy = self._make_fake_bpy((4, 0, 0))
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()
        assert session.blender_version == (4, 0, 0)
        assert session.is_blender_4_plus is True
        assert session.is_blender_5_plus is False

    def test_version_50_flags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Blender 5.0.0 sets both flags True."""
        fake_bpy = self._make_fake_bpy((5, 0, 0))
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()
        assert session.blender_version == (5, 0, 0)
        assert session.is_blender_4_plus is True
        assert session.is_blender_5_plus is True

    def test_version_stored_as_int_tuple(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Version is stored as exact 3-element int tuple."""
        fake_bpy = self._make_fake_bpy((4, 2, 1))
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()
        version = session.blender_version
        assert isinstance(version, tuple)
        assert len(version) == 3
        assert all(isinstance(v, int) for v in version)
        assert version == (4, 2, 1)

    def test_runtime_info_uses_stored_version(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """get_runtime_info returns the version stored at init time."""
        fake_bpy = self._make_fake_bpy((3, 6, 0))
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()
        info = session.get_runtime_info()
        assert info["version"] == [3, 6, 0]


class TestObservationTools:
    """Test cases for Phase 1 observation adapter methods."""

    @staticmethod
    def _make_fake_bpy_with_objects(
        objects: list,
        version: tuple = (4, 1, 0),
        collections: Optional[Dict[str, Any]] = None,
        materials: Optional[Dict[str, Any]] = None,
        scene_camera: Optional[Any] = None,
        frame_current: int = 1,
        frame_start: int = 1,
        frame_end: int = 250,
    ) -> SimpleNamespace:
        """Build a bpy mock with a populated data.objects store."""
        obj_store = {obj.name: obj for obj in objects}

        class ObjectStore:
            """Fake bpy.data.objects with list and dict behaviour."""

            def get(self, name: str) -> Optional[Any]:
                return obj_store.get(name)

            def __iter__(self):  # type: ignore[override]
                return iter(objects)

            def __len__(self) -> int:
                return len(objects)

        mat_store = materials or {}

        class MaterialStore:
            """Fake bpy.data.materials with get()."""

            def get(self, name: str) -> Optional[Any]:
                return mat_store.get(name)

        col_names = list((collections or {}).keys())

        class CollectionStore:
            """Fake bpy.data.collections with keys()."""

            def keys(self) -> list:
                return col_names

        scene = SimpleNamespace(
            camera=scene_camera,
            frame_current=frame_current,
            frame_start=frame_start,
            frame_end=frame_end,
        )

        return SimpleNamespace(
            app=SimpleNamespace(
                version=version,
                version_string=".".join(str(v) for v in version),
                binary_path="/bin/blender",
                background=True,
            ),
            data=SimpleNamespace(
                filepath="",
                objects=ObjectStore(),
                materials=MaterialStore(),
                collections=CollectionStore(),
            ),
            context=SimpleNamespace(scene=scene),
        )

    # -- get_object_info --------------------------------------------------

    def test_get_object_info_basic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Full object info is returned with transforms and metadata."""
        parent_obj = FakeObject(name="Parent", object_type="EMPTY", visible=True)
        child_obj = FakeObject(name="Child", object_type="MESH", visible=True)
        mod = SimpleNamespace(name="Subdiv", type="SUBSURF")
        con = SimpleNamespace(name="TrackTo", type="TRACK_TO")
        mat = SimpleNamespace(material=SimpleNamespace(name="Material.001"))
        obj = FakeObject(
            name="Cube",
            object_type="MESH",
            visible=True,
            location=(1.0, 2.0, 3.0),
            rotation_euler=(0.1, 0.2, 0.3),
            scale=(2.0, 2.0, 2.0),
            parent=parent_obj,
            children=[child_obj],
            modifiers=[mod],
            constraints=[con],
            material_slots=[mat],
        )
        fake_bpy = self._make_fake_bpy_with_objects([obj, parent_obj, child_obj])
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.get_object_info("Cube")

        assert result["name"] == "Cube"
        assert result["object_type"] == "MESH"
        assert list(result["location"]) == [1.0, 2.0, 3.0]
        assert result["parent_name"] == "Parent"
        assert result["children_names"] == ["Child"]
        assert len(result["modifiers"]) == 1
        assert result["modifiers"][0]["name"] == "Subdiv"
        assert len(result["constraints"]) == 1
        assert result["constraints"][0]["constraint_type"] == "TRACK_TO"
        assert len(result["material_slots"]) == 1
        assert result["material_slots"][0]["material_name"] == "Material.001"
        assert result["visible"] is True

    def test_get_object_info_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Requesting a non-existent object raises ValueError."""
        fake_bpy = self._make_fake_bpy_with_objects([])
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()
        with pytest.raises(ValueError, match="Object not found"):
            session.get_object_info("Missing")

    # -- get_mesh_info ----------------------------------------------------

    def test_get_mesh_info_basic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Counts-only mesh info returned for a MESH object."""
        mesh_data = SimpleNamespace(
            vertices=[None] * 8,
            edges=[None] * 12,
            polygons=[None] * 6,
            uv_layers=[SimpleNamespace(name="UVMap")],
            shape_keys=None,
        )
        obj = FakeObject(name="Cube", object_type="MESH", visible=True, data=mesh_data)
        fake_bpy = self._make_fake_bpy_with_objects([obj])
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.get_mesh_info("Cube")

        assert result["vertex_count"] == 8
        assert result["edge_count"] == 12
        assert result["face_count"] == 6
        assert result["uv_layer_names"] == ["UVMap"]
        assert result["has_shape_keys"] is False

    def test_get_mesh_info_rejects_non_mesh(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-MESH type raises ValueError."""
        obj = FakeObject(name="Light", object_type="LIGHT", visible=True)
        fake_bpy = self._make_fake_bpy_with_objects([obj])
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()
        with pytest.raises(ValueError, match="not a MESH"):
            session.get_mesh_info("Light")

    # -- get_bounding_box -------------------------------------------------

    def test_get_bounding_box_world_space(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Bounding box corners are transformed by matrix_world."""
        # Unit cube local corners at (0,0,0)-(1,1,1)
        local_corners = [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ]
        matrix = FakeMatrix(translation=(10.0, 20.0, 30.0))
        obj = FakeObject(
            name="Box",
            object_type="MESH",
            visible=True,
            matrix_world=matrix,
            bound_box=local_corners,
        )
        fake_bpy = self._make_fake_bpy_with_objects([obj])
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.get_bounding_box("Box", world_space=True)

        assert result["world_space"] is True
        assert len(result["corners"]) == 8
        assert result["bbox_min"] == [10.0, 20.0, 30.0]
        assert result["bbox_max"] == [11.0, 21.0, 31.0]

    def test_get_bounding_box_local_space(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Local-space bounding box skips matrix_world transform."""
        local_corners = [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ]
        obj = FakeObject(
            name="Sphere",
            object_type="MESH",
            visible=True,
            bound_box=local_corners,
        )
        fake_bpy = self._make_fake_bpy_with_objects([obj])
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.get_bounding_box("Sphere", world_space=False)

        assert result["world_space"] is False
        assert result["bbox_min"] == [-1, -1, -1]
        assert result["bbox_max"] == [1, 1, 1]

    # -- search_objects ---------------------------------------------------

    def test_search_objects_by_pattern(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regex pattern filters objects by name."""
        objs = [
            FakeObject(name="Cube", object_type="MESH", visible=True),
            FakeObject(name="Cube.001", object_type="MESH", visible=True),
            FakeObject(name="Light", object_type="LIGHT", visible=True),
        ]
        fake_bpy = self._make_fake_bpy_with_objects(objs)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.search_objects(name_pattern="^Cube")

        assert result["count"] == 2
        names = [o["name"] for o in result["objects"]]
        assert "Cube" in names
        assert "Cube.001" in names
        assert "Light" not in names

    def test_search_objects_by_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Type filter restricts results to matching object type."""
        objs = [
            FakeObject(name="Cube", object_type="MESH", visible=True),
            FakeObject(name="Light", object_type="LIGHT", visible=True),
            FakeObject(name="Camera", object_type="CAMERA", visible=True),
        ]
        fake_bpy = self._make_fake_bpy_with_objects(objs)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.search_objects(object_type="LIGHT")

        assert result["count"] == 1
        assert result["objects"][0]["name"] == "Light"

    def test_search_objects_max_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """max_results caps the returned list and sets truncated flag."""
        objs = [
            FakeObject(name=f"Obj{i}", object_type="MESH", visible=True)
            for i in range(10)
        ]
        fake_bpy = self._make_fake_bpy_with_objects(objs)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.search_objects(max_results=3)

        assert result["count"] == 3
        assert result["truncated"] is True

    # -- summarize_scene --------------------------------------------------

    def test_summarize_scene(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Scene summary includes type counts, collections, camera, frames."""
        cam_obj = FakeObject(name="Camera", object_type="CAMERA", visible=True)
        objs = [
            FakeObject(name="Cube", object_type="MESH", visible=True),
            FakeObject(name="Light", object_type="LIGHT", visible=True),
            cam_obj,
        ]
        fake_bpy = self._make_fake_bpy_with_objects(
            objs,
            collections={"Collection": None, "Props": None},
            scene_camera=cam_obj,
            frame_current=24,
            frame_start=1,
            frame_end=120,
        )
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.summarize_scene()

        assert result["total_objects"] == 3
        assert result["type_counts"]["MESH"] == 1
        assert result["type_counts"]["LIGHT"] == 1
        assert result["type_counts"]["CAMERA"] == 1
        assert set(result["collection_names"]) == {"Collection", "Props"}
        assert result["active_camera"] == "Camera"
        assert result["frame_current"] == 24
        assert result["frame_start"] == 1
        assert result["frame_end"] == 120

    # -- get_material_info ------------------------------------------------

    def test_get_material_info_with_principled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Material info extracts Principled BSDF parameters."""
        base_color_sock = SimpleNamespace(default_value=[0.8, 0.2, 0.1, 1.0])
        metallic_sock = SimpleNamespace(default_value=0.0)
        roughness_sock = SimpleNamespace(default_value=0.5)

        class FakeInputs:
            """Fake node inputs with get()."""

            _sockets = {
                "Base Color": base_color_sock,
                "Metallic": metallic_sock,
                "Roughness": roughness_sock,
            }

            def get(self, name: str) -> Optional[Any]:
                return self._sockets.get(name)

        principled_node = SimpleNamespace(
            name="Principled BSDF",
            bl_idname="ShaderNodeBsdfPrincipled",
            label="",
            inputs=FakeInputs(),
        )
        output_node = SimpleNamespace(
            name="Material Output",
            bl_idname="ShaderNodeOutputMaterial",
            label="",
        )
        node_tree = SimpleNamespace(nodes=[output_node, principled_node])
        material = SimpleNamespace(
            name="MetalMat",
            use_nodes=True,
            node_tree=node_tree,
        )

        fake_bpy = self._make_fake_bpy_with_objects(
            [], materials={"MetalMat": material}
        )
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.get_material_info("MetalMat")

        assert result["material_name"] == "MetalMat"
        assert result["use_nodes"] is True
        assert len(result["nodes"]) == 2
        assert result["principled_params"] is not None
        assert result["principled_params"]["Base Color"] == [0.8, 0.2, 0.1, 1.0]
        assert result["principled_params"]["Metallic"] == 0.0
        assert result["principled_params"]["Roughness"] == 0.5

    def test_get_material_info_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-existent material raises ValueError."""
        fake_bpy = self._make_fake_bpy_with_objects([])
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()
        with pytest.raises(ValueError, match="Material not found"):
            session.get_material_info("Ghost")

    def test_get_material_info_emission_socket_v4(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Blender 4.0+ uses 'Emission Color' socket name."""
        emission_sock = SimpleNamespace(default_value=[1.0, 0.0, 0.0, 1.0])

        class FakeInputs:
            _sockets = {"Emission Color": emission_sock}

            def get(self, name: str) -> Optional[Any]:
                return self._sockets.get(name)

        principled_node = SimpleNamespace(
            name="Principled BSDF",
            bl_idname="ShaderNodeBsdfPrincipled",
            label="",
            inputs=FakeInputs(),
        )
        node_tree = SimpleNamespace(nodes=[principled_node])
        material = SimpleNamespace(name="Glow", use_nodes=True, node_tree=node_tree)
        fake_bpy = self._make_fake_bpy_with_objects(
            [], version=(4, 2, 0), materials={"Glow": material}
        )
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.get_material_info("Glow")
        assert result["principled_params"]["Emission Color"] == [1.0, 0.0, 0.0, 1.0]

    # -- get_distance_between ---------------------------------------------

    def test_get_distance_between(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Euclidean distance computed correctly between two objects."""
        obj_a = FakeObject(
            name="A",
            object_type="MESH",
            visible=True,
            matrix_world=FakeMatrix(translation=(0.0, 0.0, 0.0)),
        )
        obj_b = FakeObject(
            name="B",
            object_type="MESH",
            visible=True,
            matrix_world=FakeMatrix(translation=(3.0, 4.0, 0.0)),
        )
        fake_bpy = self._make_fake_bpy_with_objects([obj_a, obj_b])
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.get_distance_between("A", "B")

        assert result["distance"] == pytest.approx(5.0)
        assert result["location_a"] == [0.0, 0.0, 0.0]
        assert result["location_b"] == [3.0, 4.0, 0.0]

    # -- is_object_within_bounds ------------------------------------------

    def test_is_object_within_bounds_inside(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Object inside bounds returns within_bounds=True."""
        obj = FakeObject(
            name="Ball",
            object_type="MESH",
            visible=True,
            matrix_world=FakeMatrix(translation=(5.0, 5.0, 5.0)),
        )
        fake_bpy = self._make_fake_bpy_with_objects([obj])
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.is_object_within_bounds(
            "Ball", [0.0, 0.0, 0.0], [10.0, 10.0, 10.0]
        )
        assert result["within_bounds"] is True
        assert result["object_location"] == [5.0, 5.0, 5.0]

    def test_is_object_within_bounds_outside(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Object outside bounds returns within_bounds=False."""
        obj = FakeObject(
            name="Far",
            object_type="MESH",
            visible=True,
            matrix_world=FakeMatrix(translation=(100.0, 0.0, 0.0)),
        )
        fake_bpy = self._make_fake_bpy_with_objects([obj])
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.is_object_within_bounds(
            "Far", [0.0, 0.0, 0.0], [10.0, 10.0, 10.0]
        )
        assert result["within_bounds"] is False


class TestVisualObservationTools:
    """Test cases for Phase 2 visual observation adapter methods."""

    @staticmethod
    def _make_camera_object(
        name: str = "Camera",
        location: tuple = (0.0, 0.0, 5.0),
        rotation_euler: tuple = (0.0, 0.0, 0.0),
        lens: float = 50.0,
        sensor_width: float = 36.0,
        clip_start: float = 0.1,
        clip_end: float = 1000.0,
        camera_type: str = "PERSP",
    ) -> SimpleNamespace:
        """
        Build a fake Camera object with data sub-object.

        Returns:
            SimpleNamespace camera object with .data for camera properties.
        """
        cam_data = SimpleNamespace(
            lens=lens,
            sensor_width=sensor_width,
            clip_start=clip_start,
            clip_end=clip_end,
            type=camera_type,
        )
        cam_obj = SimpleNamespace(
            name=name,
            type="CAMERA",
            location=list(location),
            rotation_euler=list(rotation_euler),
            data=cam_data,
        )
        return cam_obj

    @staticmethod
    def _make_fake_bpy_with_camera(
        objects: Optional[list] = None,
        camera_name: str = "Camera",
        version: tuple = (4, 1, 0),
        render_engine: str = "BLENDER_EEVEE_NEXT",
        resolution_x: int = 1920,
        resolution_y: int = 1080,
        resolution_percentage: int = 100,
        film_transparent: bool = False,
        frame_current: int = 1,
        frame_start: int = 1,
        frame_end: int = 250,
    ) -> SimpleNamespace:
        """
        Build a bpy mock with camera and render settings.

        Returns:
            SimpleNamespace mimicking bpy module.
        """
        all_objects = list(objects or [])
        cam_obj = None
        for obj in all_objects:
            if (
                getattr(obj, "name", None) == camera_name
                and getattr(obj, "type", None) == "CAMERA"
            ):
                cam_obj = obj
                break

        obj_store = {obj.name: obj for obj in all_objects}

        class ObjectStore:
            def get(self, name: str) -> Optional[Any]:
                return obj_store.get(name)

            def __iter__(self):  # type: ignore[override]
                return iter(all_objects)

            def __len__(self) -> int:
                return len(all_objects)

        image_settings = SimpleNamespace(
            file_format="PNG",
            quality=90,
        )
        render = SimpleNamespace(
            engine=render_engine,
            resolution_x=resolution_x,
            resolution_y=resolution_y,
            resolution_percentage=resolution_percentage,
            film_transparent=film_transparent,
            image_settings=image_settings,
        )
        scene = SimpleNamespace(
            camera=cam_obj,
            render=render,
            frame_current=frame_current,
            frame_start=frame_start,
            frame_end=frame_end,
        )
        scene.frame_set = lambda f: setattr(scene, "frame_current", f)

        class MaterialStore:
            def get(self, name: str) -> Optional[Any]:
                return None

        class CollectionStore:
            def keys(self) -> list:
                return []

        class ImageStore:
            def get(self, name: str) -> Optional[Any]:
                return None

        return SimpleNamespace(
            app=SimpleNamespace(
                version=version,
                version_string=".".join(str(v) for v in version),
                binary_path="/bin/blender",
                background=True,
            ),
            data=SimpleNamespace(
                filepath="",
                objects=ObjectStore(),
                materials=MaterialStore(),
                collections=CollectionStore(),
                images=ImageStore(),
            ),
            context=SimpleNamespace(scene=scene),
        )

    # -- set_camera_view ---------------------------------------------------

    def test_set_camera_view(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """set_camera_view sets location and rotation on the active camera."""
        cam = self._make_camera_object()
        fake_bpy = self._make_fake_bpy_with_camera(objects=[cam])
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.set_camera_view(
            location=[1.0, 2.0, 3.0],
            rotation_euler=[0.1, 0.2, 0.3],
        )

        assert result["camera_name"] == "Camera"
        assert result["location"] == [1.0, 2.0, 3.0]
        assert result["rotation_euler"] == [0.1, 0.2, 0.3]

    def test_set_camera_view_by_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """set_camera_view targets a named camera."""
        cam1 = self._make_camera_object(name="Main")
        cam2 = self._make_camera_object(name="Side", location=(10.0, 0.0, 0.0))
        fake_bpy = self._make_fake_bpy_with_camera(
            objects=[cam1, cam2], camera_name="Main"
        )
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.set_camera_view(
            location=[5.0, 5.0, 5.0],
            rotation_euler=[0.0, 0.0, 0.0],
            camera_name="Side",
        )
        assert result["camera_name"] == "Side"
        assert result["location"] == [5.0, 5.0, 5.0]

    def test_set_camera_view_no_camera_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """set_camera_view raises when no active camera exists."""
        fake_bpy = self._make_fake_bpy_with_camera(objects=[])
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()
        with pytest.raises(ValueError, match="No active camera"):
            session.set_camera_view([0, 0, 0], [0, 0, 0])

    # -- get_camera_info ---------------------------------------------------

    def test_get_camera_info(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_camera_info returns lens, sensor, clip, type."""
        cam = self._make_camera_object(
            lens=35.0, sensor_width=32.0, clip_start=0.01, clip_end=500.0
        )
        fake_bpy = self._make_fake_bpy_with_camera(objects=[cam])
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.get_camera_info()

        assert result["camera_name"] == "Camera"
        assert result["lens"] == 35.0
        assert result["sensor_width"] == 32.0
        assert result["clip_start"] == 0.01
        assert result["clip_end"] == 500.0
        assert result["camera_type"] == "PERSP"

    def test_get_camera_info_named(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_camera_info on a named camera."""
        cam = self._make_camera_object(name="Ortho", camera_type="ORTHO", lens=24.0)
        fake_bpy = self._make_fake_bpy_with_camera(objects=[cam], camera_name="Ortho")
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.get_camera_info(camera_name="Ortho")

        assert result["camera_type"] == "ORTHO"
        assert result["lens"] == 24.0

    # -- get_viewport_info -------------------------------------------------

    def test_get_viewport_info(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_viewport_info returns render engine and resolution."""
        cam = self._make_camera_object()
        fake_bpy = self._make_fake_bpy_with_camera(
            objects=[cam],
            render_engine="CYCLES",
            resolution_x=1280,
            resolution_y=720,
            resolution_percentage=50,
            film_transparent=True,
        )
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.get_viewport_info()

        assert result["render_engine"] == "CYCLES"
        assert result["resolution_x"] == 1280
        assert result["resolution_y"] == 720
        assert result["resolution_percentage"] == 50
        assert result["film_transparent"] is True
        assert result["active_camera"] == "Camera"

    def test_get_viewport_info_no_camera(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_viewport_info returns None for active_camera when missing."""
        fake_bpy = self._make_fake_bpy_with_camera(objects=[])
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.get_viewport_info()

        assert result["active_camera"] is None

    # -- focus_on_object ---------------------------------------------------

    def test_focus_on_object(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """focus_on_object positions camera based on object bounding box."""
        # Unit cube at origin: 8 corners from (-1,-1,-1) to (1,1,1)
        corners = [
            (-1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0),
            (1.0, 1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, 1.0),
            (-1.0, 1.0, 1.0),
        ]
        mesh_obj = FakeObject(
            name="Cube",
            object_type="MESH",
            visible=True,
            bound_box=corners,
            matrix_world=FakeMatrix(translation=(0.0, 0.0, 0.0)),
        )
        cam = self._make_camera_object()
        fake_bpy = self._make_fake_bpy_with_camera(objects=[cam, mesh_obj])
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.focus_on_object("Cube")

        assert result["camera_name"] == "Camera"
        assert result["object_name"] == "Cube"
        # Center of unit cube at origin should be [0, 0, 0]
        assert result["look_at"] == [0.0, 0.0, 0.0]
        # Camera location should be offset from center
        cam_loc = result["camera_location"]
        assert len(cam_loc) == 3
        # Camera should be placed along +Z +Y diagonal, i.e. negative Y and positive Z
        assert cam_loc[1] < 0.0  # -0.7071 direction
        assert cam_loc[2] > 0.0  # +0.7071 direction

    def test_focus_on_object_custom_distance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """focus_on_object with larger distance_factor pushes camera further."""
        corners = [
            (-1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0),
            (1.0, 1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, 1.0),
            (-1.0, 1.0, 1.0),
        ]
        mesh_obj = FakeObject(
            name="Box",
            object_type="MESH",
            visible=True,
            bound_box=corners,
            matrix_world=FakeMatrix(translation=(0.0, 0.0, 0.0)),
        )
        cam = self._make_camera_object()
        fake_bpy = self._make_fake_bpy_with_camera(objects=[cam, mesh_obj])
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()
        result_near = session.focus_on_object("Box", distance_factor=1.0)
        result_far = session.focus_on_object("Box", distance_factor=5.0)

        near_dist = math.sqrt(sum(c**2 for c in result_near["camera_location"]))
        far_dist = math.sqrt(sum(c**2 for c in result_far["camera_location"]))
        assert far_dist > near_dist

    def test_focus_on_object_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """focus_on_object raises ValueError for missing object."""
        cam = self._make_camera_object()
        fake_bpy = self._make_fake_bpy_with_camera(objects=[cam])
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()
        with pytest.raises(ValueError, match="Object not found"):
            session.focus_on_object("NonExistent")

    # -- capture_viewport --------------------------------------------------

    def test_capture_viewport_gpu_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """capture_viewport uses GPU offscreen fast path when available."""
        cam = self._make_camera_object()
        fake_bpy = self._make_fake_bpy_with_camera(objects=[cam])
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()

        gpu_result = {
            "image_base64": "AAAA",
            "width": 512,
            "height": 512,
            "engine": "BLENDER_EEVEE_NEXT",
            "capture_method": "gpu_offscreen",
        }
        monkeypatch.setattr(
            session, "_capture_gpu_offscreen", lambda w, h, q, e: gpu_result
        )

        result = session.capture_viewport(width=512, height=512, jpeg_quality=85)
        assert result["capture_method"] == "gpu_offscreen"
        assert result["image_base64"] == "AAAA"

    def test_capture_viewport_fallback_on_gpu_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """capture_viewport falls back to render when GPU path raises."""
        cam = self._make_camera_object()
        fake_bpy = self._make_fake_bpy_with_camera(objects=[cam])
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()

        def _gpu_fail(w: int, h: int, q: int, e: str) -> Dict[str, Any]:
            raise RuntimeError("No GPU context")

        fallback_result = {
            "image_base64": "BBBB",
            "width": 512,
            "height": 512,
            "engine": "BLENDER_EEVEE_NEXT",
            "capture_method": "render_fallback",
        }
        monkeypatch.setattr(session, "_capture_gpu_offscreen", _gpu_fail)
        monkeypatch.setattr(
            session, "_capture_render_fallback", lambda w, h, q, e: fallback_result
        )

        result = session.capture_viewport(width=512, height=512, jpeg_quality=85)
        assert result["capture_method"] == "render_fallback"
        assert result["image_base64"] == "BBBB"

    def test_capture_viewport_force_render_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """capture_viewport with use_render_fallback=True bypasses GPU."""
        cam = self._make_camera_object()
        fake_bpy = self._make_fake_bpy_with_camera(objects=[cam])
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()

        gpu_called = False

        def _gpu_spy(w: int, h: int, q: int, e: str) -> Dict[str, Any]:
            nonlocal gpu_called
            gpu_called = True
            return {"image_base64": "GPU", "capture_method": "gpu_offscreen"}

        fallback_result = {
            "image_base64": "RENDER",
            "width": 256,
            "height": 256,
            "engine": "CYCLES",
            "capture_method": "render_fallback",
        }
        monkeypatch.setattr(session, "_capture_gpu_offscreen", _gpu_spy)
        monkeypatch.setattr(
            session, "_capture_render_fallback", lambda w, h, q, e: fallback_result
        )

        result = session.capture_viewport(use_render_fallback=True)
        assert gpu_called is False
        assert result["capture_method"] == "render_fallback"
        assert result["image_base64"] == "RENDER"

    # -- capture_viewport_sequence -----------------------------------------

    def test_capture_viewport_sequence_basic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """capture_viewport_sequence captures frames and restores frame."""
        cam = self._make_camera_object()
        fake_bpy = self._make_fake_bpy_with_camera(objects=[cam], frame_current=10)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()

        call_count = 0

        def _mock_capture(**kwargs: Any) -> Dict[str, Any]:
            nonlocal call_count
            call_count += 1
            return {
                "image_base64": f"FRAME_{call_count}",
                "capture_method": "gpu_offscreen",
            }

        monkeypatch.setattr(session, "capture_viewport", _mock_capture)

        result = session.capture_viewport_sequence(start_frame=1, end_frame=3, step=1)

        assert result["frame_count"] == 3
        assert len(result["frames"]) == 3
        assert result["frames"][0]["frame"] == 1
        assert result["frames"][1]["frame"] == 2
        assert result["frames"][2]["frame"] == 3
        assert result["frames"][0]["image_base64"] == "FRAME_1"
        assert result["capture_method"] == "gpu_offscreen"
        # Original frame restored
        scene = fake_bpy.context.scene
        assert scene.frame_current == 10

    def test_capture_viewport_sequence_step(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """capture_viewport_sequence respects step parameter."""
        cam = self._make_camera_object()
        fake_bpy = self._make_fake_bpy_with_camera(objects=[cam])
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()

        monkeypatch.setattr(
            session,
            "capture_viewport",
            lambda **kw: {"image_base64": "X", "capture_method": "render_fallback"},
        )

        result = session.capture_viewport_sequence(start_frame=1, end_frame=10, step=3)

        captured_frames = [f["frame"] for f in result["frames"]]
        assert captured_frames == [1, 4, 7, 10]
        assert result["frame_count"] == 4


# ---------------------------------------------------------------------------
# Phase 3: Scene manipulation tests
# ---------------------------------------------------------------------------


class FakeLight:
    """Minimal fake Blender light data-block for tests."""

    def __init__(
        self,
        light_type: str = "POINT",
        energy: float = 10.0,
        color: tuple = (1.0, 1.0, 1.0),
        use_shadow: bool = True,
        spot_size: float = 0.785,
        spot_blend: float = 0.15,
        shadow_soft_size: float = 0.25,
    ) -> None:
        self.type = light_type
        self.energy = energy
        self.color = list(color)
        self.use_shadow = use_shadow
        self.spot_size = spot_size
        self.spot_blend = spot_blend
        self.shadow_soft_size = shadow_soft_size


class FakeMaterialSlotList:
    """Tracks materials appended to an object."""

    def __init__(self) -> None:
        self._items: List[Any] = []

    def append(self, mat: Any) -> None:
        self._items.append(mat)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):  # type: ignore[override]
        return iter(self._items)


class FakeBSDFNode:
    """Minimal Principled BSDF node mock."""

    def __init__(self) -> None:
        self.type = "BSDF_PRINCIPLED"
        self.inputs: Dict[str, SimpleNamespace] = {
            "Base Color": SimpleNamespace(default_value=(0.8, 0.8, 0.8, 1.0)),
            "Metallic": SimpleNamespace(default_value=0.0),
            "Roughness": SimpleNamespace(default_value=0.5),
        }


class FakeNodeTree:
    """Minimal node tree with a BSDF node."""

    def __init__(self) -> None:
        self.nodes = [FakeBSDFNode()]


class FakeMaterial:
    """Minimal Blender material mock."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.use_nodes = False
        self.node_tree = FakeNodeTree()


class FakeModifier:
    """Minimal Blender modifier mock."""

    def __init__(self, name: str, mod_type: str) -> None:
        self.name = name
        self.type = mod_type
        self.levels = 1
        self.render_levels = 2


class TestSceneManipulationTools:
    """Tests for Phase 3 scene manipulation adapter methods."""

    @staticmethod
    def _make_fake_bpy_with_manipulation(
        objects: Optional[Dict[str, Any]] = None,
        active_object: Optional[Any] = None,
    ) -> SimpleNamespace:
        """
        Build a fake bpy module for scene manipulation tests.

        Args:
            objects: Dict of name -> FakeObject.
            active_object: Object that bpy.context.active_object returns.

        Returns:
            SimpleNamespace mimicking bpy.
        """
        objs = objects or {}

        # bpy.data.objects
        data_objects = SimpleNamespace(
            get=lambda name: objs.get(name),
            remove=lambda obj, do_unlink=False: objs.pop(obj.name, None),
        )

        # Track materials created
        created_materials: List[Any] = []

        def _new_material(name: str) -> FakeMaterial:
            mat = FakeMaterial(name)
            created_materials.append(mat)
            return mat

        data_materials = SimpleNamespace(new=_new_material)

        data_ns = SimpleNamespace(
            objects=data_objects,
            materials=data_materials,
        )

        # bpy.ops — record calls and set active_object
        ops_calls: List[Dict[str, Any]] = []
        context_holder = SimpleNamespace(active_object=active_object)

        def _make_op(op_name: str):
            def op_fn(**kwargs: Any) -> None:
                ops_calls.append({"op": op_name, **kwargs})

            return op_fn

        ops_mesh = SimpleNamespace(
            primitive_cube_add=_make_op("mesh.primitive_cube_add"),
            primitive_uv_sphere_add=_make_op("mesh.primitive_uv_sphere_add"),
            primitive_cylinder_add=_make_op("mesh.primitive_cylinder_add"),
            primitive_cone_add=_make_op("mesh.primitive_cone_add"),
            primitive_plane_add=_make_op("mesh.primitive_plane_add"),
            primitive_torus_add=_make_op("mesh.primitive_torus_add"),
        )
        ops_object = SimpleNamespace(
            light_add=_make_op("object.light_add"),
            camera_add=_make_op("object.camera_add"),
            empty_add=_make_op("object.empty_add"),
        )

        ops_ns = SimpleNamespace(mesh=ops_mesh, object=ops_object)

        fake_bpy = SimpleNamespace(
            app=SimpleNamespace(version=(4, 0, 0)),
            data=data_ns,
            context=context_holder,
            ops=ops_ns,
        )

        # Expose internals for assertions
        fake_bpy._ops_calls = ops_calls  # type: ignore[attr-defined]
        fake_bpy._created_materials = created_materials  # type: ignore[attr-defined]

        return fake_bpy

    # -- create_object -------------------------------------------------------

    def test_create_object_cube(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Create a mesh cube via the create_object adapter."""
        created_obj = FakeObject("Cube", "MESH", True, location=(1.0, 2.0, 3.0))
        fake_bpy = self._make_fake_bpy_with_manipulation(
            active_object=created_obj,
        )
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.create_object(object_type="CUBE", location=[1.0, 2.0, 3.0])

        assert result["name"] == "Cube"
        assert result["object_type"] == "CUBE"
        assert fake_bpy._ops_calls[0]["op"] == "mesh.primitive_cube_add"

    def test_create_object_point_light(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Create a point light passes the type kwarg."""
        created_obj = FakeObject("Point", "LIGHT", True)
        fake_bpy = self._make_fake_bpy_with_manipulation(
            active_object=created_obj,
        )
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.create_object(object_type="POINT_LIGHT")

        assert result["object_type"] == "POINT_LIGHT"
        call = fake_bpy._ops_calls[0]
        assert call["op"] == "object.light_add"
        assert call["type"] == "POINT"

    def test_create_object_invalid_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unsupported type raises ValueError."""
        fake_bpy = self._make_fake_bpy_with_manipulation()
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        with pytest.raises(ValueError, match="Unsupported object type"):
            session.create_object(object_type="INVALID_TYPE")

    def test_create_object_with_name_and_scale(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Custom name and scale are applied after creation."""
        created_obj = FakeObject("Cube", "MESH", True)
        fake_bpy = self._make_fake_bpy_with_manipulation(
            active_object=created_obj,
        )
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.create_object(
            object_type="CUBE", name="MyCube", scale=[2.0, 2.0, 2.0]
        )

        assert created_obj.name == "MyCube"
        assert result["name"] == "MyCube"

    # -- delete_object -------------------------------------------------------

    def test_delete_object(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Delete removes the object from data.objects."""
        obj = FakeObject("ToDelete", "MESH", True)
        objs = {"ToDelete": obj}
        fake_bpy = self._make_fake_bpy_with_manipulation(objects=objs)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.delete_object("ToDelete")

        assert result["deleted_name"] == "ToDelete"
        assert "ToDelete" not in objs

    def test_delete_object_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Delete on a missing object raises ValueError."""
        fake_bpy = self._make_fake_bpy_with_manipulation(objects={})
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        with pytest.raises(ValueError, match="Object not found"):
            session.delete_object("Ghost")

    # -- set_object_transform ------------------------------------------------

    def test_set_transform_location_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Setting only location leaves rotation and scale untouched."""
        obj = FakeObject("Cube", "MESH", True, scale=(2.0, 2.0, 2.0))
        fake_bpy = self._make_fake_bpy_with_manipulation(objects={"Cube": obj})
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.set_object_transform("Cube", location=[5.0, 6.0, 7.0])

        assert result["location"] == [5.0, 6.0, 7.0]
        assert result["scale"] == [2.0, 2.0, 2.0]

    def test_set_transform_full(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Setting all three transform components."""
        obj = FakeObject("Cube", "MESH", True)
        fake_bpy = self._make_fake_bpy_with_manipulation(objects={"Cube": obj})
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.set_object_transform(
            "Cube",
            location=[1.0, 2.0, 3.0],
            rotation_euler=[0.1, 0.2, 0.3],
            scale=[3.0, 3.0, 3.0],
        )

        assert result["location"] == [1.0, 2.0, 3.0]
        assert result["rotation_euler"] == [0.1, 0.2, 0.3]
        assert result["scale"] == [3.0, 3.0, 3.0]

    # -- set_object_parent / clear_object_parent ------------------------------

    def test_set_parent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Parent child to parent."""
        child = FakeObject("Child", "MESH", True)
        parent = FakeObject("Parent", "MESH", True)
        objs = {"Child": child, "Parent": parent}
        fake_bpy = self._make_fake_bpy_with_manipulation(objects=objs)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.set_object_parent("Child", "Parent")

        assert result["child_name"] == "Child"
        assert result["parent_name"] == "Parent"
        assert child.parent is parent

    def test_clear_parent_keep_transform(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unparent with keep_transform preserves world location."""
        parent = FakeObject("Parent", "MESH", True, location=(10.0, 0.0, 0.0))
        child = FakeObject(
            "Child",
            "MESH",
            True,
            location=(5.0, 0.0, 0.0),
            parent=parent,
        )
        # matrix_world translation simulates the world-space location
        child.matrix_world = FakeMatrix((15.0, 0.0, 0.0))
        objs = {"Child": child, "Parent": parent}
        fake_bpy = self._make_fake_bpy_with_manipulation(objects=objs)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.clear_object_parent("Child", keep_transform=True)

        assert result["previous_parent"] == "Parent"
        assert child.parent is None
        # location should be set to world translation
        assert list(child.location) == [15.0, 0.0, 0.0]

    def test_clear_parent_no_keep(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unparent without keeping transform just sets parent to None."""
        parent = FakeObject("Parent", "MESH", True)
        child = FakeObject("Child", "MESH", True, parent=parent)
        objs = {"Child": child, "Parent": parent}
        fake_bpy = self._make_fake_bpy_with_manipulation(objects=objs)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.clear_object_parent("Child", keep_transform=False)

        assert result["previous_parent"] == "Parent"
        assert child.parent is None

    # -- assign_material -----------------------------------------------------

    def test_assign_material(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Assign a material with custom BSDF params."""
        mat_slots = FakeMaterialSlotList()
        mesh_data = SimpleNamespace(materials=mat_slots)
        obj = FakeObject("Cube", "MESH", True, data=mesh_data)
        objs = {"Cube": obj}
        fake_bpy = self._make_fake_bpy_with_manipulation(objects=objs)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.assign_material(
            object_name="Cube",
            material_name="RedMetal",
            base_color=[1.0, 0.0, 0.0, 1.0],
            metallic=1.0,
            roughness=0.1,
        )

        assert result["object_name"] == "Cube"
        assert result["material_name"] == "RedMetal"
        assert len(mat_slots) == 1
        # Verify BSDF node inputs were set
        created_mat = fake_bpy._created_materials[0]
        bsdf = created_mat.node_tree.nodes[0]
        assert bsdf.inputs["Metallic"].default_value == 1.0
        assert bsdf.inputs["Roughness"].default_value == 0.1

    def test_assign_material_no_data_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Assigning material to an empty (no data) raises ValueError."""
        obj = FakeObject("Empty", "EMPTY", True, data=None)
        objs = {"Empty": obj}
        fake_bpy = self._make_fake_bpy_with_manipulation(objects=objs)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        with pytest.raises(ValueError, match="has no data"):
            session.assign_material(object_name="Empty")

    # -- add_modifier --------------------------------------------------------

    def test_add_modifier(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Add a SUBSURF modifier with custom params."""
        added_mods: List[Any] = []

        class FakeModifiers:
            def new(self, name: str, type: str) -> FakeModifier:
                mod = FakeModifier(name, type)
                added_mods.append(mod)
                return mod

        obj = FakeObject("Cube", "MESH", True)
        obj.modifiers = FakeModifiers()
        objs = {"Cube": obj}
        fake_bpy = self._make_fake_bpy_with_manipulation(objects=objs)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.add_modifier(
            object_name="Cube",
            modifier_type="SUBSURF",
            modifier_name="MySubsurf",
            params={"levels": 3, "render_levels": 4},
        )

        assert result["modifier_name"] == "MySubsurf"
        assert result["modifier_type"] == "SUBSURF"
        mod = added_mods[0]
        assert mod.levels == 3
        assert mod.render_levels == 4

    # -- set_light_params ----------------------------------------------------

    def test_set_light_params_point(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Set energy and color on a point light."""
        light_data = FakeLight(light_type="POINT", energy=10.0)
        obj = FakeObject("PointLight", "LIGHT", True, data=light_data)
        objs = {"PointLight": obj}
        fake_bpy = self._make_fake_bpy_with_manipulation(objects=objs)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.set_light_params(
            light_name="PointLight",
            energy=500.0,
            color=[1.0, 0.5, 0.0],
            use_shadow=False,
        )

        assert result["light_type"] == "POINT"
        assert result["energy"] == 500.0
        assert light_data.energy == 500.0
        assert list(light_data.color) == [1.0, 0.5, 0.0]
        assert light_data.use_shadow is False

    def test_set_light_params_spot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Set spot-specific params on a spot light."""
        light_data = FakeLight(light_type="SPOT")
        obj = FakeObject("SpotLight", "LIGHT", True, data=light_data)
        objs = {"SpotLight": obj}
        fake_bpy = self._make_fake_bpy_with_manipulation(objects=objs)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.set_light_params(
            light_name="SpotLight",
            spot_size=1.2,
            spot_blend=0.5,
        )

        assert result["light_type"] == "SPOT"
        assert light_data.spot_size == 1.2
        assert light_data.spot_blend == 0.5

    def test_set_light_params_not_a_light(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Setting light params on a non-light object raises ValueError."""
        obj = FakeObject("Cube", "MESH", True)
        objs = {"Cube": obj}
        fake_bpy = self._make_fake_bpy_with_manipulation(objects=objs)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        with pytest.raises(ValueError, match="not LIGHT"):
            session.set_light_params(light_name="Cube", energy=100.0)


class TestFileIOTools:
    """Tests for Phase 4 file I/O adapter methods."""

    @staticmethod
    def _make_fake_bpy_with_file_io(
        objects: Optional[Dict[str, Any]] = None,
        filepath: str = "",
        is_dirty: bool = False,
        scene_name: str = "Scene",
        version: tuple = (4, 0, 0),
    ) -> SimpleNamespace:
        """
        Build a fake bpy module for file I/O tests.

        Args:
            objects: Dict of name -> object. Defaults to empty dict.
            filepath: Current blend file path.
            is_dirty: Whether there are unsaved changes.
            scene_name: Active scene name.
            version: Blender version tuple.

        Returns:
            A SimpleNamespace mimicking bpy for file I/O tests.
        """
        if objects is None:
            objects = {}

        ops_calls: List[Dict[str, Any]] = []

        class FakeWM:
            @staticmethod
            def open_mainfile(filepath: str = "") -> None:
                ops_calls.append({"op": "wm.open_mainfile", "filepath": filepath})

            @staticmethod
            def save_mainfile() -> None:
                ops_calls.append({"op": "wm.save_mainfile"})

            @staticmethod
            def save_as_mainfile(filepath: str = "") -> None:
                ops_calls.append({"op": "wm.save_as_mainfile", "filepath": filepath})

            @staticmethod
            def obj_import(**kwargs: Any) -> None:
                ops_calls.append({"op": "wm.obj_import", **kwargs})

            @staticmethod
            def obj_export(**kwargs: Any) -> None:
                ops_calls.append({"op": "wm.obj_export", **kwargs})

            @staticmethod
            def usd_import(**kwargs: Any) -> None:
                ops_calls.append({"op": "wm.usd_import", **kwargs})

            @staticmethod
            def usd_export(**kwargs: Any) -> None:
                ops_calls.append({"op": "wm.usd_export", **kwargs})

            @staticmethod
            def stl_import(**kwargs: Any) -> None:
                ops_calls.append({"op": "wm.stl_import", **kwargs})

            @staticmethod
            def stl_export(**kwargs: Any) -> None:
                ops_calls.append({"op": "wm.stl_export", **kwargs})

            @staticmethod
            def ply_import(**kwargs: Any) -> None:
                ops_calls.append({"op": "wm.ply_import", **kwargs})

            @staticmethod
            def ply_export(**kwargs: Any) -> None:
                ops_calls.append({"op": "wm.ply_export", **kwargs})

        class FakeImportScene:
            @staticmethod
            def obj(**kwargs: Any) -> None:
                ops_calls.append({"op": "import_scene.obj", **kwargs})

            @staticmethod
            def fbx(**kwargs: Any) -> None:
                ops_calls.append({"op": "import_scene.fbx", **kwargs})

            @staticmethod
            def gltf(**kwargs: Any) -> None:
                ops_calls.append({"op": "import_scene.gltf", **kwargs})

        class FakeExportScene:
            @staticmethod
            def obj(**kwargs: Any) -> None:
                ops_calls.append({"op": "export_scene.obj", **kwargs})

            @staticmethod
            def fbx(**kwargs: Any) -> None:
                ops_calls.append({"op": "export_scene.fbx", **kwargs})

            @staticmethod
            def gltf(**kwargs: Any) -> None:
                ops_calls.append({"op": "export_scene.gltf", **kwargs})

        class FakeImportMesh:
            @staticmethod
            def stl(**kwargs: Any) -> None:
                ops_calls.append({"op": "import_mesh.stl", **kwargs})

            @staticmethod
            def ply(**kwargs: Any) -> None:
                ops_calls.append({"op": "import_mesh.ply", **kwargs})

        class FakeExportMesh:
            @staticmethod
            def stl(**kwargs: Any) -> None:
                ops_calls.append({"op": "export_mesh.stl", **kwargs})

            @staticmethod
            def ply(**kwargs: Any) -> None:
                ops_calls.append({"op": "export_mesh.ply", **kwargs})

        fake_ops = SimpleNamespace(
            wm=FakeWM,
            import_scene=FakeImportScene,
            export_scene=FakeExportScene,
            import_mesh=FakeImportMesh,
            export_mesh=FakeExportMesh,
        )

        fake_scene = SimpleNamespace(name=scene_name)
        fake_context = SimpleNamespace(scene=fake_scene)
        fake_data = SimpleNamespace(
            objects=objects,
            filepath=filepath,
            is_dirty=is_dirty,
        )
        fake_app = SimpleNamespace(version=version)

        fake_bpy = SimpleNamespace(
            app=fake_app,
            ops=fake_ops,
            data=fake_data,
            context=fake_context,
        )
        fake_bpy._ops_calls = ops_calls  # type: ignore[attr-defined]
        return fake_bpy

    @staticmethod
    def _session_allowing(tmp_path: Path) -> "blender_runtime.BlenderRuntimeSession":
        """Build a session whose sandbox policy also admits the pytest tmp dir."""
        security = SecurityConfig(
            allowed_paths=[*SecurityConfig().allowed_paths, str(tmp_path)]
        )
        return blender_runtime.BlenderRuntimeSession(
            settings=Settings(security=security)
        )

    # -- open_blend_file -----------------------------------------------------

    def test_open_blend_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Open a .blend file sets filepath and returns object count."""
        blend_file = tmp_path / "test.blend"
        blend_file.write_bytes(b"fake")  # create the file on disk

        objs = {"Cube": FakeObject("Cube", "MESH", True)}
        fake_bpy = self._make_fake_bpy_with_file_io(objects=objs)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = self._session_allowing(tmp_path)
        result = session.open_blend_file(str(blend_file))

        assert result["file_path"] == str(blend_file)
        assert result["object_count"] == 1
        assert fake_bpy._ops_calls[0]["op"] == "wm.open_mainfile"

    def test_open_blend_file_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Opening a non-existent file raises FileNotFoundError."""
        fake_bpy = self._make_fake_bpy_with_file_io()
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        with pytest.raises(FileNotFoundError, match="File not found"):
            session.open_blend_file("/tmp/simul_mcp/nonexistent/path.blend")

    def test_open_blend_file_wrong_ext(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Opening a non-.blend file raises ValueError."""
        txt_file = tmp_path / "not_blend.txt"
        txt_file.write_text("hello")

        fake_bpy = self._make_fake_bpy_with_file_io()
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = self._session_allowing(tmp_path)
        with pytest.raises(ValueError, match="Not a .blend file"):
            session.open_blend_file(str(txt_file))

    # -- save_blend_file -----------------------------------------------------

    def test_save_blend_file_as(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Save-as with explicit path calls save_as_mainfile."""
        fake_bpy = self._make_fake_bpy_with_file_io()
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.save_blend_file(file_path="/tmp/simul_mcp/out.blend")

        assert result["file_path"] == "/tmp/simul_mcp/out.blend"
        assert fake_bpy._ops_calls[0]["op"] == "wm.save_as_mainfile"

    def test_save_blend_file_in_place(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Save without path calls save_mainfile when filepath is set."""
        fake_bpy = self._make_fake_bpy_with_file_io(filepath="/existing.blend")
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.save_blend_file()

        assert result["file_path"] == "/existing.blend"
        assert fake_bpy._ops_calls[0]["op"] == "wm.save_mainfile"

    def test_save_blend_file_no_path_unsaved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Save without path on unsaved file raises ValueError."""
        fake_bpy = self._make_fake_bpy_with_file_io(filepath="")
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        with pytest.raises(ValueError, match="No file path provided"):
            session.save_blend_file()

    # -- get_file_info -------------------------------------------------------

    def test_get_file_info(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """File info returns current state."""
        objs = {"Cube": FakeObject("Cube", "MESH", True)}
        fake_bpy = self._make_fake_bpy_with_file_io(
            objects=objs,
            filepath="/my/scene.blend",
            is_dirty=True,
            scene_name="MyScene",
        )
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.get_file_info()

        assert result["file_path"] == "/my/scene.blend"
        assert result["is_saved"] is True
        assert result["is_dirty"] is True
        assert result["object_count"] == 1
        assert result["scene_name"] == "MyScene"

    def test_get_file_info_unsaved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """File info for an unsaved file shows is_saved=False."""
        fake_bpy = self._make_fake_bpy_with_file_io(filepath="")
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.get_file_info()

        assert result["is_saved"] is False
        assert result["file_path"] == ""

    # -- import_file ---------------------------------------------------------

    def test_import_obj_v4(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """OBJ import on Blender 4.x uses wm.obj_import."""
        obj_file = tmp_path / "model.obj"
        obj_file.write_text("# OBJ")

        objs: Dict[str, Any] = {}
        fake_bpy = self._make_fake_bpy_with_file_io(objects=objs, version=(4, 2, 0))

        def fake_obj_import(**kwargs: Any) -> None:
            fake_bpy._ops_calls.append({"op": "wm.obj_import", **kwargs})
            objs["ImportedMesh"] = FakeObject("ImportedMesh", "MESH", True)

        fake_bpy.ops.wm.obj_import = fake_obj_import
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = self._session_allowing(tmp_path)
        result = session.import_file(str(obj_file), "OBJ")

        assert result["file_format"] == "OBJ"
        assert "ImportedMesh" in result["imported_objects"]
        assert fake_bpy._ops_calls[0]["op"] == "wm.obj_import"

    def test_import_obj_v36(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """OBJ import on Blender 3.6 uses import_scene.obj."""
        obj_file = tmp_path / "model.obj"
        obj_file.write_text("# OBJ")

        objs: Dict[str, Any] = {}
        fake_bpy = self._make_fake_bpy_with_file_io(objects=objs, version=(3, 6, 0))

        def fake_legacy_import(**kwargs: Any) -> None:
            fake_bpy._ops_calls.append({"op": "import_scene.obj", **kwargs})
            objs["LegacyMesh"] = FakeObject("LegacyMesh", "MESH", True)

        fake_bpy.ops.import_scene.obj = fake_legacy_import
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = self._session_allowing(tmp_path)
        result = session.import_file(str(obj_file), "OBJ")

        assert result["file_format"] == "OBJ"
        assert "LegacyMesh" in result["imported_objects"]
        assert fake_bpy._ops_calls[0]["op"] == "import_scene.obj"

    def test_import_fbx(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """FBX import uses import_scene.fbx (same for v3.6 and v4)."""
        fbx_file = tmp_path / "model.fbx"
        fbx_file.write_bytes(b"\x00")

        objs: Dict[str, Any] = {}
        fake_bpy = self._make_fake_bpy_with_file_io(objects=objs, version=(4, 0, 0))

        def fake_fbx_import(**kwargs: Any) -> None:
            fake_bpy._ops_calls.append({"op": "import_scene.fbx", **kwargs})
            objs["FBXObj"] = FakeObject("FBXObj", "MESH", True)

        fake_bpy.ops.import_scene.fbx = fake_fbx_import
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = self._session_allowing(tmp_path)
        result = session.import_file(str(fbx_file), "FBX")

        assert result["file_format"] == "FBX"
        assert "FBXObj" in result["imported_objects"]

    def test_import_unsupported_format(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Importing an unsupported format raises ValueError."""
        fake_bpy = self._make_fake_bpy_with_file_io()
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        with pytest.raises(ValueError, match="Unsupported format"):
            session.import_file("/tmp/simul_mcp/file.abc", "ABC")

    def test_import_file_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Importing a nonexistent file raises FileNotFoundError."""
        fake_bpy = self._make_fake_bpy_with_file_io()
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        with pytest.raises(FileNotFoundError, match="File not found"):
            session.import_file("/tmp/simul_mcp/nonexistent/model.obj", "OBJ")

    # -- export_file ---------------------------------------------------------

    def test_export_obj_v4(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """OBJ export on Blender 4.x uses wm.obj_export."""
        fake_bpy = self._make_fake_bpy_with_file_io(version=(4, 2, 0))
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.export_file("/tmp/simul_mcp/out.obj", "OBJ")

        assert result["file_format"] == "OBJ"
        assert result["file_path"] == "/tmp/simul_mcp/out.obj"
        assert fake_bpy._ops_calls[0]["op"] == "wm.obj_export"

    def test_export_obj_v36(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """OBJ export on Blender 3.6 uses export_scene.obj."""
        fake_bpy = self._make_fake_bpy_with_file_io(version=(3, 6, 0))
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.export_file("/tmp/simul_mcp/out.obj", "OBJ")

        assert result["file_format"] == "OBJ"
        assert fake_bpy._ops_calls[0]["op"] == "export_scene.obj"

    def test_export_selected_only_v4(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Export with selected_only on v4 passes export_selected_objects."""
        fake_bpy = self._make_fake_bpy_with_file_io(version=(4, 0, 0))
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        session.export_file("/tmp/simul_mcp/out.obj", "OBJ", selected_only=True)

        call = fake_bpy._ops_calls[0]
        assert call["export_selected_objects"] is True

    def test_export_unsupported_format(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Exporting an unsupported format raises ValueError."""
        fake_bpy = self._make_fake_bpy_with_file_io()
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        with pytest.raises(ValueError, match="Unsupported format"):
            session.export_file("/tmp/simul_mcp/out.abc", "ABC")


class TestAnimationTools:
    """Tests for Phase 5 animation & timeline adapter methods."""

    @staticmethod
    def _make_fake_bpy_with_animation(
        version: tuple = (4, 0, 0),
        current_frame: int = 1,
        frame_start: int = 1,
        frame_end: int = 250,
        fps: float = 24.0,
        is_playing: bool = False,
        objects: Optional[Dict[str, Any]] = None,
    ) -> SimpleNamespace:
        """
        Build a fake bpy module for animation tests.

        Args:
            version: Blender version tuple.
            current_frame: Current scene frame.
            frame_start: Animation start frame.
            frame_end: Animation end frame.
            fps: Frames per second.
            is_playing: Whether animation is currently playing.
            objects: Dict of name -> object. Defaults to empty dict.

        Returns:
            A SimpleNamespace mimicking bpy for animation tests.
        """
        if objects is None:
            objects = {}

        ops_calls: List[Dict[str, Any]] = []

        fake_render = SimpleNamespace(fps=fps)
        fake_scene = SimpleNamespace(
            frame_current=current_frame,
            frame_start=frame_start,
            frame_end=frame_end,
            render=fake_render,
        )

        def frame_set(frame: int) -> None:
            fake_scene.frame_current = frame

        fake_scene.frame_set = frame_set

        fake_screen = SimpleNamespace(is_animation_playing=is_playing)

        class FakeScreenOps:
            @staticmethod
            def animation_play(**kwargs: Any) -> None:
                fake_screen.is_animation_playing = True
                ops_calls.append({"op": "screen.animation_play", **kwargs})

            @staticmethod
            def animation_cancel(**kwargs: Any) -> None:
                fake_screen.is_animation_playing = False
                ops_calls.append({"op": "screen.animation_cancel", **kwargs})

        fake_context = SimpleNamespace(scene=fake_scene, screen=fake_screen)
        fake_data = SimpleNamespace(objects=objects)
        fake_app = SimpleNamespace(version=version)
        fake_ops = SimpleNamespace(screen=FakeScreenOps)

        fake_bpy = SimpleNamespace(
            app=fake_app,
            context=fake_context,
            data=fake_data,
            ops=fake_ops,
        )
        fake_bpy._ops_calls = ops_calls  # type: ignore[attr-defined]
        return fake_bpy

    # -- set_frame -----------------------------------------------------------

    def test_set_frame(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Setting a frame updates the scene frame_current."""
        fake_bpy = self._make_fake_bpy_with_animation(current_frame=1)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.set_frame(42)

        assert result["frame"] == 42
        assert fake_bpy.context.scene.frame_current == 42

    # -- get_frame -----------------------------------------------------------

    def test_get_frame(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Get frame returns current frame and range info."""
        fake_bpy = self._make_fake_bpy_with_animation(
            current_frame=10, frame_start=1, frame_end=100, fps=30.0
        )
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.get_frame()

        assert result["current_frame"] == 10
        assert result["frame_start"] == 1
        assert result["frame_end"] == 100
        assert result["fps"] == 30.0

    # -- set_frame_range -----------------------------------------------------

    def test_set_frame_range(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Setting frame range updates scene frame_start and frame_end."""
        fake_bpy = self._make_fake_bpy_with_animation()
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.set_frame_range(10, 200)

        assert result["frame_start"] == 10
        assert result["frame_end"] == 200

    def test_set_frame_range_invalid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Setting frame range with start >= end raises ValueError."""
        fake_bpy = self._make_fake_bpy_with_animation()
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        with pytest.raises(ValueError, match="must be less than"):
            session.set_frame_range(100, 50)

    # -- play_animation ------------------------------------------------------

    def test_play_animation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Play action triggers animation_play operator."""
        fake_bpy = self._make_fake_bpy_with_animation(is_playing=False)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.play_animation("play")

        assert result["action"] == "play"
        assert result["is_playing"] is True
        assert fake_bpy._ops_calls[0]["op"] == "screen.animation_play"

    def test_stop_animation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stop action cancels animation when playing."""
        fake_bpy = self._make_fake_bpy_with_animation(is_playing=True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.play_animation("stop")

        assert result["action"] == "stop"
        assert result["is_playing"] is False
        assert fake_bpy._ops_calls[0]["op"] == "screen.animation_cancel"

    def test_reverse_animation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reverse action triggers animation_play with reverse=True."""
        fake_bpy = self._make_fake_bpy_with_animation(is_playing=False)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.play_animation("reverse")

        assert result["action"] == "reverse"
        call = fake_bpy._ops_calls[0]
        assert call["op"] == "screen.animation_play"
        assert call["reverse"] is True

    def test_play_animation_invalid_action(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Invalid action raises ValueError."""
        fake_bpy = self._make_fake_bpy_with_animation()
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        with pytest.raises(ValueError, match="Invalid action"):
            session.play_animation("fast_forward")

    # -- insert_keyframe -----------------------------------------------------

    def test_insert_keyframe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Insert keyframe calls keyframe_insert on the object."""
        kf_calls: List[Dict[str, Any]] = []
        obj = FakeObject("Cube", "MESH", True)
        # type: ignore[attr-defined]
        obj.keyframe_insert = lambda **kw: kf_calls.append(kw)

        objs = {"Cube": obj}
        fake_bpy = self._make_fake_bpy_with_animation(objects=objs)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.insert_keyframe("Cube", "location", 10, index=0)

        assert result["object_name"] == "Cube"
        assert result["data_path"] == "location"
        assert result["frame"] == 10
        assert kf_calls[0]["data_path"] == "location"
        assert kf_calls[0]["frame"] == 10
        assert kf_calls[0]["index"] == 0

    # -- delete_keyframe -----------------------------------------------------

    def test_delete_keyframe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Delete keyframe calls keyframe_delete on the object."""
        kf_calls: List[Dict[str, Any]] = []
        obj = FakeObject("Cube", "MESH", True)
        # type: ignore[attr-defined]
        obj.keyframe_delete = lambda **kw: kf_calls.append(kw)

        objs = {"Cube": obj}
        fake_bpy = self._make_fake_bpy_with_animation(objects=objs)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.delete_keyframe("Cube", "location", 10, index=0)

        assert result["object_name"] == "Cube"
        assert result["frame"] == 10
        assert kf_calls[0]["data_path"] == "location"
        assert kf_calls[0]["frame"] == 10

    # -- get_keyframes -------------------------------------------------------

    def test_get_keyframes_no_animation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Object without animation data returns has_animation=False."""
        obj = FakeObject("Cube", "MESH", True)
        obj.animation_data = None  # type: ignore[attr-defined]

        objs = {"Cube": obj}
        fake_bpy = self._make_fake_bpy_with_animation(objects=objs)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.get_keyframes("Cube")

        assert result["has_animation"] is False
        assert result["channels"] == []

    def test_get_keyframes_legacy_fcurves(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Get keyframes using legacy fcurves (Blender 3.6-4.x)."""
        kp1 = SimpleNamespace(co=(1.0, 0.0))
        kp2 = SimpleNamespace(co=(10.0, 1.0))
        kp3 = SimpleNamespace(co=(20.0, 2.0))

        fcurve_x = SimpleNamespace(
            data_path="location",
            array_index=0,
            keyframe_points=[kp1, kp2],
        )
        fcurve_y = SimpleNamespace(
            data_path="location",
            array_index=1,
            keyframe_points=[kp3],
        )

        fake_action = SimpleNamespace(fcurves=[fcurve_x, fcurve_y])
        fake_anim_data = SimpleNamespace(action=fake_action)

        obj = FakeObject("Cube", "MESH", True)
        obj.animation_data = fake_anim_data  # type: ignore[attr-defined]

        objs = {"Cube": obj}
        fake_bpy = self._make_fake_bpy_with_animation(objects=objs, version=(4, 0, 0))
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.get_keyframes("Cube")

        assert result["has_animation"] is True
        assert len(result["channels"]) == 2

        ch0 = result["channels"][0]
        assert ch0["data_path"] == "location"
        assert ch0["array_index"] == 0
        assert ch0["keyframe_count"] == 2
        assert ch0["frame_range"] == [1, 10]

        ch1 = result["channels"][1]
        assert ch1["array_index"] == 1
        assert ch1["keyframe_count"] == 1
        assert ch1["frame_range"] == [20, 20]

    def test_get_keyframes_v5_channelbag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Get keyframes using channelbag API (Blender 5.0+)."""
        kp1 = SimpleNamespace(co=(5.0, 0.0))
        kp2 = SimpleNamespace(co=(15.0, 1.0))

        fcurve = SimpleNamespace(
            data_path="rotation_euler",
            array_index=2,
            keyframe_points=[kp1, kp2],
        )
        channelbag = SimpleNamespace(fcurves=[fcurve])
        fake_action = SimpleNamespace(
            channelbags=[channelbag],
            fcurves=[],  # legacy fallback should not be used
        )
        fake_anim_data = SimpleNamespace(action=fake_action)

        obj = FakeObject("Cube", "MESH", True)
        obj.animation_data = fake_anim_data  # type: ignore[attr-defined]

        objs = {"Cube": obj}
        fake_bpy = self._make_fake_bpy_with_animation(objects=objs, version=(5, 0, 0))
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.get_keyframes("Cube")

        assert result["has_animation"] is True
        assert len(result["channels"]) == 1

        ch = result["channels"][0]
        assert ch["data_path"] == "rotation_euler"
        assert ch["array_index"] == 2
        assert ch["keyframe_count"] == 2
        assert ch["frame_range"] == [5, 15]

    def test_get_keyframes_object_not_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Get keyframes for nonexistent object raises ValueError."""
        fake_bpy = self._make_fake_bpy_with_animation()
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        with pytest.raises(ValueError, match="Object not found"):
            session.get_keyframes("Nonexistent")

    def test_stop_animation_when_not_playing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Stop when already stopped does not call animation_cancel."""
        fake_bpy = self._make_fake_bpy_with_animation(is_playing=False)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.play_animation("stop")

        assert result["action"] == "stop"
        assert result["is_playing"] is False
        assert len(fake_bpy._ops_calls) == 0  # No cancel called


class TestExecuteScript:
    """Tests for execute_script adapter method."""

    @staticmethod
    def _make_fake_bpy_for_script() -> SimpleNamespace:
        """Build a fake bpy module for script execution tests."""
        return SimpleNamespace(
            app=SimpleNamespace(version=(4, 0, 0)),
            data=SimpleNamespace(
                objects=SimpleNamespace(get=lambda name: None),
                filepath="/tmp/simul_mcp/test.blend",
            ),
            context=SimpleNamespace(
                scene=SimpleNamespace(name="Scene"),
            ),
        )

    def test_execute_simple_script(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Execute a script that prints output."""
        fake_bpy = self._make_fake_bpy_for_script()
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.execute_script("print('hello world')")

        assert result["output"] == "hello world\n"
        assert result["return_value"] is None
        assert result["duration_seconds"] >= 0.0
        assert "error" not in result

    def test_execute_script_with_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Script assigns __result__ and it is returned."""
        fake_bpy = self._make_fake_bpy_for_script()
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.execute_script("__result__ = 42")

        assert result["return_value"] == "42"
        assert "error" not in result

    def test_execute_script_has_bpy_access(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Script can access bpy module."""
        fake_bpy = self._make_fake_bpy_for_script()
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.execute_script("__result__ = bpy.context.scene.name")

        assert result["return_value"] == "'Scene'"
        assert "error" not in result

    def test_execute_script_syntax_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Script with syntax error returns error in payload."""
        fake_bpy = self._make_fake_bpy_for_script()
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.execute_script("def bad(")

        assert result["error"] is not None
        assert "SyntaxError" in result["error"]
        assert result["duration_seconds"] >= 0.0

    def test_execute_script_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Script with runtime error returns error in payload."""
        fake_bpy = self._make_fake_bpy_for_script()
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.execute_script("x = 1 / 0")

        assert result["error"] is not None
        assert "ZeroDivisionError" in result["error"]

    def test_execute_script_timeout_returns_instead_of_blocking(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A script that never returns must not hang the MCP server."""
        import threading
        import time

        fake_bpy = self._make_fake_bpy_for_script()
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        release = threading.Event()
        fake_bpy.release = release

        session = blender_runtime.BlenderRuntimeSession()
        started = time.monotonic()
        result = session.execute_script("bpy.release.wait()\nprint('late')", timeout=0.2)
        elapsed = time.monotonic() - started
        release.set()

        assert elapsed < 2.0
        assert result["timed_out"] is True
        assert result["error"] is not None
        assert result["error"].startswith("TimeoutError")
        assert "still running" in result["error"]
        assert result["output"] is None
        assert result["return_value"] is None
        assert result["duration_seconds"] == 0.2

    def test_execute_script_with_timeout_returns_normally_when_fast(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_bpy = self._make_fake_bpy_for_script()
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.execute_script("__result__ = 7\nprint('quick')", timeout=5.0)

        assert result["output"] == "quick\n"
        assert result["return_value"] == "7"
        assert "error" not in result
        assert "timed_out" not in result

    def test_execute_script_with_timeout_reports_script_errors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_bpy = self._make_fake_bpy_for_script()
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.execute_script("x = 1 / 0", timeout=5.0)

        assert "ZeroDivisionError" in result["error"]
        assert "timed_out" not in result

    def test_execute_script_output_capped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Script output is capped at 4096 chars."""
        fake_bpy = self._make_fake_bpy_for_script()
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        # Print >4096 chars
        result = session.execute_script("print('A' * 10000)")

        assert result["output"] is not None
        assert len(result["output"]) == 4096


class TestCreateMeshFromData:
    """Tests for create_mesh_from_data adapter method."""

    @staticmethod
    def _make_fake_bpy_for_mesh_data(
        collections: Optional[Dict[str, Any]] = None,
    ) -> SimpleNamespace:
        """
        Build a fake bpy module for mesh-from-data tests.

        Args:
            collections: Dict of collection name -> SimpleNamespace.

        Returns:
            SimpleNamespace mimicking bpy.
        """
        created_meshes: List[Any] = []
        created_objects: List[Any] = []
        linked_objects: List[Any] = []

        class FakeMeshData:
            """Mock mesh data-block with from_pydata support."""

            def __init__(self, name: str) -> None:
                self.name = name
                self.vertices: list = []
                self.edges: list = []
                self.polygons: list = []

            def from_pydata(
                self,
                verts: list,
                edges: list,
                faces: list,
            ) -> None:
                self.vertices = verts
                self.edges = edges
                self.polygons = faces

            def update(self) -> None:
                pass

        def _new_mesh(name: str) -> FakeMeshData:
            mesh = FakeMeshData(name)
            created_meshes.append(mesh)
            return mesh

        class FakeObjData:
            """Mock object returned by bpy.data.objects.new()."""

            def __init__(self, name: str, data: Any) -> None:
                self.name = name
                self.data = data
                self.location = (0.0, 0.0, 0.0)

        def _new_object(name: str, data: Any) -> FakeObjData:
            obj = FakeObjData(name, data)
            created_objects.append(obj)
            return obj

        scene_col = SimpleNamespace(
            objects=SimpleNamespace(
                link=lambda obj: linked_objects.append(obj),
            ),
        )

        cols = collections or {}

        fake_bpy = SimpleNamespace(
            app=SimpleNamespace(version=(4, 0, 0)),
            data=SimpleNamespace(
                meshes=SimpleNamespace(new=_new_mesh),
                objects=SimpleNamespace(
                    new=_new_object,
                    get=lambda name: None,
                ),
                collections=SimpleNamespace(
                    get=lambda name: cols.get(name),
                ),
            ),
            context=SimpleNamespace(
                scene=SimpleNamespace(
                    name="Scene",
                    collection=scene_col,
                ),
            ),
        )

        fake_bpy._created_meshes = created_meshes  # type: ignore[attr-defined]
        fake_bpy._created_objects = created_objects  # type: ignore[attr-defined]
        fake_bpy._linked_objects = linked_objects  # type: ignore[attr-defined]

        return fake_bpy

    # -- Basic creation ------------------------------------------------------

    def test_create_mesh_triangle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Create a basic triangle from 3 vertices and 1 face."""
        fake_bpy = self._make_fake_bpy_for_mesh_data()
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.create_mesh_from_data(
            name="Triangle",
            vertices=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]],
            faces=[[0, 1, 2]],
        )

        assert result["object_name"] == "Triangle"
        assert result["mesh_name"] == "Triangle"
        assert result["vertex_count"] == 3
        assert result["face_count"] == 1
        assert len(fake_bpy._linked_objects) == 1

    def test_create_mesh_with_location(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Object location is set when provided."""
        fake_bpy = self._make_fake_bpy_for_mesh_data()
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        session.create_mesh_from_data(
            name="Placed",
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            faces=[[0, 1, 2]],
            location=[5.0, 6.0, 7.0],
        )

        obj = fake_bpy._created_objects[0]
        assert obj.location == (5.0, 6.0, 7.0)

    def test_create_mesh_edges_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Create a wireframe mesh with edges but no faces."""
        fake_bpy = self._make_fake_bpy_for_mesh_data()
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.create_mesh_from_data(
            name="Wire",
            vertices=[[0, 0, 0], [1, 0, 0], [1, 1, 0]],
            edges=[[0, 1], [1, 2]],
        )

        assert result["edge_count"] == 2
        assert result["face_count"] == 0

    # -- Validation ----------------------------------------------------------

    def test_create_mesh_empty_vertices_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty vertices list raises ValueError."""
        fake_bpy = self._make_fake_bpy_for_mesh_data()
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        with pytest.raises(ValueError, match="vertices list must not be empty"):
            session.create_mesh_from_data(name="Bad", vertices=[])

    def test_create_mesh_invalid_edge_index_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Out-of-range edge index raises ValueError."""
        fake_bpy = self._make_fake_bpy_for_mesh_data()
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        with pytest.raises(ValueError, match="Edge 0 index 5 out of range"):
            session.create_mesh_from_data(
                name="Bad",
                vertices=[[0, 0, 0], [1, 0, 0]],
                edges=[[0, 5]],
            )

    def test_create_mesh_invalid_face_index_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Out-of-range face index raises ValueError."""
        fake_bpy = self._make_fake_bpy_for_mesh_data()
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        with pytest.raises(ValueError, match="Face 0 index 10 out of range"):
            session.create_mesh_from_data(
                name="Bad",
                vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                faces=[[0, 1, 10]],
            )

    def test_create_mesh_face_too_few_indices_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Face with fewer than 3 indices raises ValueError."""
        fake_bpy = self._make_fake_bpy_for_mesh_data()
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        with pytest.raises(ValueError, match="Face 0 must have >= 3 indices"):
            session.create_mesh_from_data(
                name="Bad",
                vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                faces=[[0, 1]],
            )

    # -- Collection linking --------------------------------------------------

    def test_create_mesh_to_collection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mesh is linked to named collection when specified."""
        col_linked: List[Any] = []
        fake_col = SimpleNamespace(
            objects=SimpleNamespace(
                link=lambda obj: col_linked.append(obj),
            ),
        )
        fake_bpy = self._make_fake_bpy_for_mesh_data(
            collections={"MyCol": fake_col},
        )
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        session.create_mesh_from_data(
            name="InCol",
            vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            faces=[[0, 1, 2]],
            collection_name="MyCol",
        )

        assert len(col_linked) == 1
        # Scene collection should NOT have it
        assert len(fake_bpy._linked_objects) == 0

    def test_create_mesh_collection_not_found_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing collection raises ValueError."""
        fake_bpy = self._make_fake_bpy_for_mesh_data()
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        with pytest.raises(ValueError, match="Collection 'NoSuch' not found"):
            session.create_mesh_from_data(
                name="Bad",
                vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                faces=[[0, 1, 2]],
                collection_name="NoSuch",
            )


class FakeObjectWithProps:
    """FakeObject with dict-like custom property support for SimReady tests."""

    def __init__(
        self,
        name: str,
        object_type: str = "MESH",
        location: tuple = (0.0, 0.0, 0.0),
        rotation_euler: tuple = (0.0, 0.0, 0.0),
        scale: tuple = (1.0, 1.0, 1.0),
        dimensions: tuple = (1.0, 1.0, 1.0),
        parent: Optional[Any] = None,
        children: Optional[list] = None,
        data: Optional[Any] = None,
    ) -> None:
        self.name = name
        self.type = object_type
        self.location = FakeVector(location)
        self.rotation_euler = FakeVector(rotation_euler)
        self.scale = FakeVector(scale)
        self.dimensions = FakeVector(dimensions)
        self.parent = parent
        self.children = children or []
        self.data = data
        self._custom_props: Dict[str, Any] = {}
        self.hide_viewport = False
        self.empty_display_type = ""

    def __setitem__(self, key: str, value: Any) -> None:
        self._custom_props[key] = value

    def __getitem__(self, key: str) -> Any:
        return self._custom_props[key]

    def keys(self) -> list:
        return list(self._custom_props.keys())

    def select_set(self, val: bool) -> None:
        self._selected = val

    def visible_get(self) -> bool:
        return not self.hide_viewport


class TestSimReadyMethods:
    """Tests for SimReady Asset Format adapter methods."""

    @staticmethod
    def _make_fake_bpy(
        objects: Optional[List[Any]] = None,
    ) -> SimpleNamespace:
        """Create a fake bpy module with the given objects."""
        objs = objects or []

        class ObjCollection:
            def __init__(self, items: list) -> None:
                self._items = items
                self._dict = {o.name: o for o in items}

            def __iter__(self):  # type: ignore[override]
                return iter(self._items)

            def __len__(self) -> int:
                return len(self._items)

            def get(self, name: str) -> Optional[Any]:
                return self._dict.get(name)

            def new(self, name: str, data: Any) -> Any:
                obj = FakeObjectWithProps(
                    name=name,
                    object_type="EMPTY" if data is None else "MESH",
                )
                obj.data = data
                self._items.append(obj)
                self._dict[name] = obj
                return obj

            def link(self, obj: Any) -> None:
                if obj.name not in self._dict:
                    self._items.append(obj)
                    self._dict[obj.name] = obj

        data_objects = ObjCollection(list(objs))
        scene_col = SimpleNamespace(objects=data_objects)
        scene = SimpleNamespace(collection=scene_col)
        return SimpleNamespace(
            app=SimpleNamespace(
                version=(5, 0, 1),
                version_string="5.0.1",
                binary_path="/bin/blender",
                background=True,
            ),
            data=SimpleNamespace(
                filepath="",
                objects=data_objects,
            ),
            context=SimpleNamespace(scene=scene),
            ops=SimpleNamespace(
                wm=SimpleNamespace(
                    usd_export=lambda **kw: None,
                ),
            ),
        )

    def test_apply_simready_metadata_semantic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Apply semantic metadata and verify custom properties."""
        obj = FakeObjectWithProps(name="fire_truck")
        fake_bpy = self._make_fake_bpy([obj])
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.apply_simready_metadata(
            object_name="fire_truck",
            metadata={
                "semantic": {
                    "semantic_class": "truck",
                    "semantic_hierarchy": "machine/vehicle/truck",
                    "semantic_qcode": "Q43193",
                },
            },
        )

        assert result["object_name"] == "fire_truck"
        assert "simready_semantic_class" in result["applied_properties"]
        assert obj["simready_semantic_class"] == "truck"
        assert obj["simready_semantic_hierarchy"] == "machine/vehicle/truck"
        assert obj["simready_semantic_qcode"] == "Q43193"

    def test_apply_simready_metadata_physics(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Apply physics metadata and verify custom properties."""
        obj = FakeObjectWithProps(name="box_crate")
        fake_bpy = self._make_fake_bpy([obj])
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.apply_simready_metadata(
            object_name="box_crate",
            metadata={
                "physics": {
                    "mass_kg": 12.5,
                    "collider_type": "convexHull",
                    "is_rigid_body": True,
                },
            },
        )

        assert "simready_mass_kg" in result["applied_properties"]
        assert "simready_collider_type" in result["applied_properties"]
        assert "simready_is_rigid_body" in result["applied_properties"]
        assert obj["simready_mass_kg"] == 12.5
        assert obj["simready_is_rigid_body"] == 1

    def test_get_simready_metadata_roundtrip(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Apply then read back metadata for roundtrip verification."""
        obj = FakeObjectWithProps(name="coffee_cup")
        fake_bpy = self._make_fake_bpy([obj])
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        session.apply_simready_metadata(
            object_name="coffee_cup",
            metadata={
                "semantic": {
                    "semantic_class": "cup",
                    "semantic_qcode": "Q81727",
                },
                "physics": {
                    "mass_kg": 0.35,
                },
                "material": {
                    "substrate_type": "ceramic",
                    "shader_type": "OmniPBR",
                },
            },
        )

        result = session.get_simready_metadata(object_name="coffee_cup")
        assert result["has_simready_data"] is True
        meta = result["metadata"]
        assert meta["semantic"]["semantic_class"] == "cup"
        assert meta["physics"]["mass_kg"] == 0.35
        assert meta["material"]["substrate_type"] == "ceramic"

    def test_get_simready_metadata_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Object with no simready_ properties returns has_simready_data=False."""
        obj = FakeObjectWithProps(name="plain_cube")
        fake_bpy = self._make_fake_bpy([obj])
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.get_simready_metadata(object_name="plain_cube")
        assert result["has_simready_data"] is False
        assert result["metadata"] is None

    def test_validate_naming_issue(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Uppercase names should produce a naming error."""
        obj = FakeObjectWithProps(
            name="MyBadName",
            data=SimpleNamespace(materials=["mat1"]),
        )
        obj.parent = FakeObjectWithProps(name="root", object_type="EMPTY")
        fake_bpy = self._make_fake_bpy([obj])
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.validate_simready_compliance(
            object_names=["MyBadName"],
            check_scale=False,
            check_transforms=False,
            check_materials=False,
            check_hierarchy=False,
        )

        assert result["compliant"] is False
        assert result["issue_count"] >= 1
        naming_issues = [i for i in result["issues"] if i["check"] == "naming"]
        assert len(naming_issues) == 1
        assert "MyBadName" in naming_issues[0]["message"]

    def test_validate_compliant_object(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A well-named, correctly-scaled object with material passes."""
        parent = FakeObjectWithProps(name="root", object_type="EMPTY")
        obj = FakeObjectWithProps(
            name="coffee_cup",
            dimensions=(0.08, 0.08, 0.12),
            data=SimpleNamespace(materials=["ceramic_white"]),
            parent=parent,
        )
        fake_bpy = self._make_fake_bpy([obj])
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.validate_simready_compliance(
            object_names=["coffee_cup"],
        )

        assert result["compliant"] is True
        error_issues = [i for i in result["issues"] if i["severity"] == "error"]
        assert len(error_issues) == 0

    def test_validate_no_material_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Object with no material assigned produces material error."""
        parent = FakeObjectWithProps(name="root", object_type="EMPTY")
        obj = FakeObjectWithProps(
            name="bare_cube",
            data=SimpleNamespace(materials=[]),
            parent=parent,
        )
        fake_bpy = self._make_fake_bpy([obj])
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.validate_simready_compliance(
            object_names=["bare_cube"],
            check_naming=False,
            check_scale=False,
            check_transforms=False,
            check_hierarchy=False,
        )

        assert result["compliant"] is False
        mat_issues = [i for i in result["issues"] if i["check"] == "materials"]
        assert len(mat_issues) == 1

    def test_setup_simready_hierarchy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Hierarchy setup creates root empty and parents children."""
        child = FakeObjectWithProps(name="body_mesh")
        fake_bpy = self._make_fake_bpy([child])
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.setup_simready_hierarchy(
            root_name="fire_truck",
            child_names=["body_mesh"],
            semantic={
                "semantic_class": "truck",
                "semantic_qcode": "Q43193",
            },
        )

        assert result["root_name"] == "fire_truck"
        assert "body_mesh" in result["children"]
        assert result["hierarchy_path"] == "/fire_truck"
        # Child should be parented
        assert child.parent is not None
        assert child.parent.name == "fire_truck"

    def test_setup_simready_hierarchy_bad_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Hierarchy setup rejects non-SimReady root names."""
        fake_bpy = self._make_fake_bpy([])
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        with pytest.raises(ValueError, match="violates SimReady naming"):
            session.setup_simready_hierarchy(
                root_name="BadName",
                child_names=[],
            )
