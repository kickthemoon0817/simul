"""Tests for Blender runtime adapter functionality."""

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest


from simul.adapters import blender_runtime
from simul.config import SecurityConfig, Settings
from tests.fakes import (
    FakeCollection,
    FakeConstraint,
    FakeEuler,
    FakeLayerCollection,
    FakeMatrix,
    FakePropArray,
    FakeTransformObject,
    FakeVector,
    FakeViewLayer,
    euler_matrix,
)


def _box(low: float = -1.0, high: float = 1.0) -> List[FakePropArray]:
    """Return a cube's eight bound_box corners, as Blender's bpy_prop_arrays."""
    return [
        FakePropArray((x, y, z))
        for x in (low, high)
        for y in (low, high)
        for z in (low, high)
    ]


def _rotate_xyz(rotation_euler: Any, vector: Sequence[float]) -> List[float]:
    """Turn a vector by an XYZ Euler rotation: about X, then Y, then Z."""
    rx, ry, rz = rotation_euler
    x, y, z = vector
    y, z = y * math.cos(rx) - z * math.sin(rx), y * math.sin(rx) + z * math.cos(rx)
    x, z = x * math.cos(ry) + z * math.sin(ry), -x * math.sin(ry) + z * math.cos(ry)
    x, y = x * math.cos(rz) - y * math.sin(rz), x * math.sin(rz) + y * math.cos(rz)
    return [x, y, z]


def _camera_forward(rotation_euler: Any) -> List[float]:
    """Return where a camera with this XYZ Euler rotation looks: its local -Z."""
    return _rotate_xyz(rotation_euler, (0.0, 0.0, -1.0))


class FakeCameraData:
    """Camera data whose field of view follows its lens and sensor, as in Blender.

    ``angle`` spans the side of the frame the sensor is fitted to: it comes
    from sensor_height for sensor_fit VERTICAL, from sensor_width otherwise.
    """

    def __init__(
        self,
        lens: float = 50.0,
        sensor_width: float = 36.0,
        sensor_height: float = 24.0,
        sensor_fit: str = "AUTO",
        camera_type: str = "PERSP",
        clip_start: float = 0.1,
        clip_end: float = 1000.0,
        shift_x: float = 0.0,
        shift_y: float = 0.0,
        ortho_scale: float = 6.0,
    ) -> None:
        self.lens = lens
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height
        self.sensor_fit = sensor_fit
        self.type = camera_type
        self.clip_start = clip_start
        self.clip_end = clip_end
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.ortho_scale = ortho_scale

    @property
    def angle(self) -> float:
        sensor = (
            self.sensor_height if self.sensor_fit == "VERTICAL" else self.sensor_width
        )
        return 2 * math.atan(sensor / 2 / self.lens)


def _rig(stored: bool = False) -> FakeTransformObject:
    """An empty moved, turned and scaled, as a camera rig is."""
    return FakeTransformObject(
        "Rig",
        location=(5.0, -3.0, 2.0),
        rotation_euler=(0.3, -0.2, math.radians(90)),
        scale=(2.0, 2.0, 2.0),
        stored=stored,
    )


def _move_rig(rig: FakeTransformObject) -> None:
    """Move a rig after a camera was parented to it, so the camera's local
    transform no longer equals its world transform."""
    rig.location = FakeVector((8.0, 1.0, -1.0))
    rig.rotation_euler = FakeEuler((0.3, -0.2, math.radians(135)))


# How a camera is attached and stores its rotation: (parented, rotation_mode).
# Writing location and rotation_euler directly misplaces every camera here
# but the unparented XYZ one; set_camera_view's rotation also turns a ZXY
# camera wrongly, while focus_on_object's level (x, 0, 0) rotation does not.
CAMERA_SETUPS = {
    "unparented": (False, "XYZ"),
    "parented": (True, "XYZ"),
    "quaternion": (False, "QUATERNION"),
    "axis_angle": (False, "AXIS_ANGLE"),
    "zxy_euler": (False, "ZXY"),
    "parented_quaternion": (True, "QUATERNION"),
}


def _track_to(
    target: Sequence[float], name: str = "Track To", **state: Any
) -> FakeConstraint:
    """A Track To as Blender sets one up on a camera: its -Z toward ``target``, +Y up."""

    def solve(matrix: np.ndarray) -> np.ndarray:
        forward = np.asarray(target, dtype=float) - matrix[:3, 3]
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        solved = matrix.copy()
        solved[:3, :3] = np.column_stack(
            [right, np.cross(right, forward), -forward]
        ) * np.linalg.norm(matrix[:3, :3], axis=0)
        return solved

    return FakeConstraint(name, "TRACK_TO", solve, **state)


def _copy_location(
    target: Sequence[float], name: str = "Copy Location", **state: Any
) -> FakeConstraint:
    """A Copy Location: the object stands where ``target`` does, turned as before."""

    def solve(matrix: np.ndarray) -> np.ndarray:
        solved = matrix.copy()
        solved[:3, 3] = target
        return solved

    return FakeConstraint(name, "COPY_LOCATION", solve, **state)


def _child_of(
    target_change: np.ndarray, name: str = "Child Of", **state: Any
) -> FakeConstraint:
    """A Child Of whose target moved by ``target_change`` after its inverse was set."""
    return FakeConstraint(
        name, "CHILD_OF", lambda matrix: target_change @ matrix, **state
    )


def _satisfied(name: str = "Limit Distance", **state: Any) -> FakeConstraint:
    """An active constraint that leaves this pose alone, such as a Limit Distance it meets."""
    return FakeConstraint(name, "LIMIT_DISTANCE", lambda matrix: matrix, **state)


def _evaluated_rotation(obj: Any) -> np.ndarray:
    """An object's world rotation, its scale divided out."""
    world = np.asarray(obj.matrix_world.rows)[:3, :3]
    return world / np.linalg.norm(world, axis=0)


class _DrivenCamera(FakeTransformObject):
    """A camera with a driver on location[0] (stored, as in background Blender).

    Each evaluation writes the driver's value into the camera's location
    before composing matrix_world, as Blender's active depsgraph writes a
    driver's result back to the original object.
    """

    def __init__(self, driven_x: float, **kwargs: Any) -> None:
        self._driven_x = driven_x
        super().__init__("Camera", "CAMERA", FakeCameraData(), stored=True, **kwargs)

    def evaluate(self) -> None:
        self.location = FakeVector((self._driven_x, *list(self.location)[1:]))
        super().evaluate()


class _SinglePrecisionCamera(FakeTransformObject):
    """A camera whose location is single precision, as Blender's Object.location is."""

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "location":
            value = FakeVector(np.asarray(list(value), dtype=np.float32).tolist())
        super().__setattr__(name, value)


class _PartlyEvaluatedCamera(FakeTransformObject):
    """A camera disabled in viewports whose custom property a visible object's driver reads.

    Blender evaluates that camera only in part: on Blender 5.0.1 each
    evaluation gave matrix_world back the matrix the camera had before it was
    disabled, whatever its transform held since.
    """

    def __init__(self, *constraints: FakeConstraint) -> None:
        self._before: Optional[FakeMatrix] = None
        super().__init__(
            "Side",
            "CAMERA",
            FakeCameraData(),
            location=(1.0, 2.0, 3.0),
            stored=True,
            constraints=constraints,
        )
        self._before = self.matrix_world
        self.hide_viewport = True

    def evaluate(self) -> None:
        if self._before is None:
            super().evaluate()
        else:
            self._stored = FakeMatrix(self._before.rows)


def _world_axes(obj: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """An object's local X, Y and -Z axes in world space, unit length."""
    rotation = np.asarray(obj.matrix_world.rows)[:3, :3]
    rotation = rotation / np.linalg.norm(rotation, axis=0)
    return rotation[:, 0], rotation[:, 1], -rotation[:, 2]


def _in_frame(
    camera: Any,
    point: Sequence[float],
    half: Tuple[float, float],
    offset: Tuple[float, float] = (0.0, 0.0),
    ortho_scale: Optional[float] = None,
) -> Tuple[float, float, float]:
    """Where a world point appears to a camera, as world_to_camera_view reports it.

    (x, y) run from 0 to 1 across the frame, z is the depth along the view
    axis. ``half`` is the frame's half width and height at unit depth (per
    unit of ortho_scale for an orthographic camera) and ``offset`` its center
    off the view axis; each test states them for its lens and aspect.
    """
    matrix = np.asarray(camera.matrix_world.rows)
    matrix[:3, :3] = matrix[:3, :3] / np.linalg.norm(matrix[:3, :3], axis=0)
    x, y, z, _w = np.linalg.inv(matrix) @ np.array([*point, 1.0])
    depth = -z
    scale = ortho_scale if ortho_scale is not None else depth
    return (
        0.5 + (x / scale - offset[0]) / (2 * half[0]),
        0.5 + (y / scale - offset[1]) / (2 * half[1]),
        depth,
    )


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
        self.matrix_world = matrix_world or FakeMatrix.Translation(location)
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
            # Real bpy always has it; world-space reads evaluate transforms first.
            context=SimpleNamespace(scene=scene, evaluated_depsgraph_get=lambda: "depsgraph"),
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
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        """Bounding box corners are transformed by matrix_world.

        The corners are bpy_prop_arrays, which a mathutils Matrix does not
        multiply (TypeError on Blender 5.0.1); each becomes a Vector first.
        """
        matrix = FakeMatrix.Translation((10.0, 20.0, 30.0))
        obj = FakeObject(
            name="Box",
            object_type="MESH",
            visible=True,
            matrix_world=matrix,
            bound_box=_box(0.0, 1.0),
        )
        fake_bpy = self._make_fake_bpy_with_objects([obj])
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.get_bounding_box("Box", world_space=True)

        assert result["world_space"] is True
        assert len(result["corners"]) == 8
        assert all(type(corner) is list for corner in result["corners"])
        assert result["bbox_min"] == [10.0, 20.0, 30.0]
        assert result["bbox_max"] == [11.0, 21.0, 31.0]

    def test_get_bounding_box_world_space_applies_rotation_and_scale(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        """The whole world matrix applies, not just its translation."""
        # A quarter turn about Z, X scaled by 2, then moved to (5, 0, 1).
        matrix = FakeMatrix(
            [(0, -1, 0, 5.0), (2, 0, 0, 0.0), (0, 0, 1, 1.0), (0, 0, 0, 1)]
        )
        obj = FakeObject(
            name="Turned",
            object_type="MESH",
            visible=True,
            matrix_world=matrix,
            bound_box=_box(-1.0, 1.0),
        )
        fake_bpy = self._make_fake_bpy_with_objects([obj])
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.get_bounding_box("Turned")

        assert result["corners"][0] == [6.0, -2.0, 0.0]  # local (-1, -1, -1)
        assert result["bbox_min"] == [4.0, -2.0, 0.0]
        assert result["bbox_max"] == [6.0, 2.0, 2.0]

    def test_get_bounding_box_local_space(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Local-space bounding box skips matrix_world transform."""
        obj = FakeObject(
            name="Sphere",
            object_type="MESH",
            visible=True,
            bound_box=_box(-1.0, 1.0),
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

    @pytest.mark.parametrize(
        ("pattern", "refusal"),
        [
            (r"^(\w+)+_x$", "a quantifier inside a repeated group"),
            (r"(\w+_)*light", "a quantifier inside a repeated group"),
            (r"(a|ab)*c", "alternatives inside a repeated group"),
            ("Cube(", "not a valid regular expression"),
        ],
    )
    def test_search_objects_refuses_a_pattern_before_searching_any_name(
        self, monkeypatch: pytest.MonkeyPatch, pattern: str, refusal: str
    ) -> None:
        """The search runs on Blender's main thread, where nothing interrupts it.

        A failing search for a repeated group that can match the same text in
        several ways tries every split of the name: ^(\\w+)+_x$ took hours on
        a 40-character object name and froze the editor, and
        search_blender_objects stays registered with scripting disabled. An
        invalid pattern raised re.error, reported as error_type "error".
        """
        searched: List[str] = []

        class Recorded(FakeObject):
            @property
            def name(self) -> str:
                searched.append(self._name)
                return self._name

            @name.setter
            def name(self, value: str) -> None:
                self._name = value

        # Short enough that a search without the guard fails this test, not hangs it.
        objects = [Recorded(name="a" * 12, object_type="MESH", visible=True)]
        fake_bpy = self._make_fake_bpy_with_objects(objects)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        searched.clear()  # the fake's own name index read them

        session = blender_runtime.BlenderRuntimeSession()
        with pytest.raises(ValueError, match=refusal):
            session.search_objects(name_pattern=pattern)
        assert searched == []

    @pytest.mark.parametrize(
        ("pattern", "names"),
        [
            (r"^Cube\.\d+$", ["Cube.001"]),
            (r"^(Cube|Light)$", ["Cube", "Light"]),
            (r"(\.\d+)?$", ["Cube", "Cube.001", "Light"]),
            (r".*ub.*", ["Cube", "Cube.001"]),
        ],
    )
    def test_search_objects_still_takes_patterns_that_cannot_split_a_name(
        self, monkeypatch: pytest.MonkeyPatch, pattern: str, names: List[str]
    ) -> None:
        objects = [
            FakeObject(name="Cube", object_type="MESH", visible=True),
            FakeObject(name="Cube.001", object_type="MESH", visible=True),
            FakeObject(name="Light", object_type="LIGHT", visible=True),
        ]
        fake_bpy = self._make_fake_bpy_with_objects(objects)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.search_objects(name_pattern=pattern)

        assert [o["name"] for o in result["objects"]] == names

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
            matrix_world=FakeMatrix.Translation((0.0, 0.0, 0.0)),
        )
        obj_b = FakeObject(
            name="B",
            object_type="MESH",
            visible=True,
            matrix_world=FakeMatrix.Translation((3.0, 4.0, 0.0)),
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
            matrix_world=FakeMatrix.Translation((5.0, 5.0, 5.0)),
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
            matrix_world=FakeMatrix.Translation((100.0, 0.0, 0.0)),
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
        setup: str = "unparented",
        scale: tuple = (1.0, 1.0, 1.0),
        stored: bool = False,
        **data: Any,
    ) -> FakeTransformObject:
        """
        Build a fake camera object whose world transform behaves as in Blender.

        Args:
            setup: A CAMERA_SETUPS key: parented to a rig that moved after the
                parenting, and the rotation mode.
            stored: Keep matrix_world (the camera's and its rig's) until the
                depsgraph is evaluated, as background Blender and the bpy
                module do; built evaluated.
            **data: Further FakeCameraData settings (sensor_fit, shift_x, ...).

        Returns:
            The camera object, with its FakeCameraData as .data.
        """
        parented, rotation_mode = CAMERA_SETUPS[setup]
        rig = _rig(stored) if parented else None
        cam_obj = FakeTransformObject(
            name,
            "CAMERA",
            FakeCameraData(
                lens=lens,
                sensor_width=sensor_width,
                clip_start=clip_start,
                clip_end=clip_end,
                camera_type=camera_type,
                **data,
            ),
            location=location,
            rotation_euler=rotation_euler,
            scale=scale,
            rotation_mode=rotation_mode,
            parent=rig,
            stored=stored,
        )
        if rig is not None:
            _move_rig(rig)
        cam_obj.evaluate()
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
        view_layer: Optional[FakeViewLayer] = None,
        left_out: Sequence[Any] = (),
        marker_cameras: Sequence[Any] = (),
    ) -> SimpleNamespace:
        """
        Build a bpy mock with camera and render settings.

        Args:
            view_layer: The context's view layer. By default one whose scene
                collection holds every object not put in a collection.
            left_out: Objects that evaluated_depsgraph_get() does not
                evaluate, as Blender leaves out a camera disabled in viewports.
            marker_cameras: The cameras the scene's timeline markers bind,
                one marker each.

        Returns:
            SimpleNamespace mimicking bpy module.
        """
        all_objects = list(objects or [])
        if view_layer is None:
            scene_collection = FakeCollection("Scene Collection")
            view_layer = FakeViewLayer(FakeLayerCollection(scene_collection))
            for obj in all_objects:
                if not getattr(obj, "users_collection", None):
                    obj.users_collection = [scene_collection]
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
            pixel_aspect_x=1.0,
            pixel_aspect_y=1.0,
            film_transparent=film_transparent,
            image_settings=image_settings,
        )
        scene = SimpleNamespace(
            camera=cam_obj,
            render=render,
            frame_current=frame_current,
            frame_start=frame_start,
            frame_end=frame_end,
            timeline_markers=[
                SimpleNamespace(name=f"F_{frame}", frame=frame, camera=camera)
                for frame, camera in enumerate(marker_cameras, start=frame_current)
            ],
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

        def evaluated_depsgraph_get() -> str:
            """Recompute every stored matrix_world, as Blender's evaluation does."""
            for obj in all_objects:
                if isinstance(obj, FakeTransformObject) and not any(
                    obj is out for out in left_out
                ):
                    obj.evaluate()
            return "depsgraph"

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
            context=SimpleNamespace(
                scene=scene,
                view_layer=view_layer,
                evaluated_depsgraph_get=evaluated_depsgraph_get,
            ),
        )

    # -- set_camera_view ---------------------------------------------------

    def test_set_camera_view(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        """set_camera_view places the active camera and reports where it is."""
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
        assert result["location"] == pytest.approx([1.0, 2.0, 3.0])
        assert result["rotation_euler"] == pytest.approx([0.1, 0.2, 0.3])
        assert list(cam.location) == pytest.approx([1.0, 2.0, 3.0])
        assert list(cam.rotation_euler) == pytest.approx([0.1, 0.2, 0.3])

    def test_set_camera_view_by_name(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
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
        assert result["location"] == pytest.approx([5.0, 5.0, 5.0])
        assert list(cam2.location) == pytest.approx([5.0, 5.0, 5.0])
        assert list(cam1.location) == [0.0, 0.0, 5.0]

    @pytest.mark.parametrize("setup", list(CAMERA_SETUPS))
    def test_set_camera_view_places_the_camera_in_world_space(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any, setup: str
    ) -> None:
        """The camera ends up at the world location and XYZ Euler rotation given.

        set_camera_view wrote location and rotation_euler, which placed a
        parented camera relative to its rig, left a QUATERNION or AXIS_ANGLE
        camera (glTF imports store quaternions) unturned, and turned a ZXY
        camera by the angles taken in its own order. The camera's scale stays.
        """
        cam = self._make_camera_object(setup=setup, scale=(1.5, 1.5, 1.5))
        world_scale = list(cam.matrix_world.to_scale())
        fake_bpy = self._make_fake_bpy_with_camera(objects=[cam])
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        location, rotation = [7.0, -7.0, 5.0], [1.1, 0.25, 0.78]

        session = blender_runtime.BlenderRuntimeSession()
        result = session.set_camera_view(location=location, rotation_euler=rotation)

        world = np.asarray(cam.matrix_world.rows)
        scale = np.linalg.norm(world[:3, :3], axis=0)
        assert world[:3, 3] == pytest.approx(location)
        assert world[:3, :3] / scale == pytest.approx(euler_matrix(rotation))
        assert scale == pytest.approx(world_scale)
        assert result["location"] == pytest.approx(location)
        assert result["rotation_euler"] == pytest.approx(rotation)

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

    def test_get_camera_info(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        """get_camera_info returns the placement, lens, sensor, clip and type."""
        cam = self._make_camera_object(
            location=(1.0, 2.0, 3.0),
            rotation_euler=(0.1, 0.2, 4.0),
            lens=35.0,
            sensor_width=32.0,
            clip_start=0.01,
            clip_end=500.0,
        )
        fake_bpy = self._make_fake_bpy_with_camera(objects=[cam])
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.get_camera_info()

        assert result["camera_name"] == "Camera"
        assert result["location"] == pytest.approx([1.0, 2.0, 3.0])
        # The camera's own angles, not an equivalent a whole turn away (4 - 2 pi).
        assert result["rotation_euler"] == pytest.approx([0.1, 0.2, 4.0])
        assert result["lens"] == 35.0
        assert result["sensor_width"] == 32.0
        assert result["clip_start"] == 0.01
        assert result["clip_end"] == 500.0
        assert result["camera_type"] == "PERSP"

    @pytest.mark.parametrize("setup", ["parented", "quaternion", "parented_quaternion"])
    def test_get_camera_info_reports_the_world_placement(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any, setup: str
    ) -> None:
        """location and rotation_euler say where the camera is, as set_camera_view takes them.

        They were the local values: relative to the parent, and for a
        quaternion camera a rotation_euler Blender does not use at all.
        """
        cam = self._make_camera_object(
            setup=setup, location=(1.0, 2.0, 3.0), rotation_euler=(0.4, 0.5, 0.6)
        )
        cam.rotation_quaternion = FakeVector((0.8, 0.2, 0.1, 0.55))
        fake_bpy = self._make_fake_bpy_with_camera(objects=[cam])
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.get_camera_info()

        world = np.asarray(cam.matrix_world.rows)
        rotation = world[:3, :3] / np.linalg.norm(world[:3, :3], axis=0)
        assert result["location"] == pytest.approx(list(world[:3, 3]))
        assert euler_matrix(result["rotation_euler"]) == pytest.approx(rotation)
        # Neither is the local value the camera stores.
        assert result["rotation_euler"] != pytest.approx([0.4, 0.5, 0.6])
        parented, _mode = CAMERA_SETUPS[setup]
        assert (result["location"] != pytest.approx([1.0, 2.0, 3.0])) is parented

    def test_get_camera_info_named(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
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

    @staticmethod
    def _cube(
        name: str = "Cube",
        offset: Sequence[float] = (0.0, 0.0, 0.0),
        half: float = 1.0,
        object_type: str = "MESH",
    ) -> FakeObject:
        """A cube ``2 * half`` wide at ``offset``; ``half=0`` gives a point's bound_box."""
        return FakeObject(
            name=name,
            object_type=object_type,
            visible=True,
            bound_box=_box(-half, half),
            matrix_world=FakeMatrix.Translation(offset),
        )

    def _focus(
        self,
        monkeypatch: pytest.MonkeyPatch,
        target: Any,
        camera: Any,
        resolution: Tuple[int, int] = (1920, 1080),
        **arguments: Any,
    ) -> Dict[str, Any]:
        """Call focus_on_object on ``target`` with ``camera`` as the scene camera."""
        fake_bpy = self._make_fake_bpy_with_camera(
            objects=[camera, target],
            camera_name=camera.name,
            resolution_x=resolution[0],
            resolution_y=resolution[1],
        )
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        session = blender_runtime.BlenderRuntimeSession()
        return session.focus_on_object(target.name, **arguments)

    @pytest.mark.parametrize(
        ("resolution", "data", "half", "offset"),
        [
            # A 50 mm lens on a 36 x 24 mm sensor; AUTO fits the 36 mm to the
            # longer side, so a 16:9 frame reaches 18/50 across and 9/16 of
            # that up: the frame's height decides the distance.
            ((1920, 1080), {}, (18 / 50, 18 / 50 * 9 / 16), (0.0, 0.0)),
            ((1080, 1920), {}, (18 / 50 * 9 / 16, 18 / 50), (0.0, 0.0)),
            # VERTICAL fits the 24 mm sensor height to the frame's height.
            (
                (1920, 1080),
                {"sensor_fit": "VERTICAL"},
                (12 / 50 * 16 / 9, 12 / 50),
                (0.0, 0.0),
            ),
            # HORIZONTAL fits the 36 mm to the width even in a portrait frame.
            (
                (1080, 1920),
                {"sensor_fit": "HORIZONTAL"},
                (18 / 50, 18 / 50 * 16 / 9),
                (0.0, 0.0),
            ),
            # A lens shift of 0.1 moves the frame up by a tenth of its width,
            # leaving less of it below the axis.
            (
                (1920, 1080),
                {"shift_y": 0.1},
                (18 / 50, 18 / 50 * 9 / 16),
                (0.0, 0.1 * 2 * 18 / 50),
            ),
        ],
        ids=["landscape", "portrait", "vertical_fit", "horizontal_fit", "shift"],
    )
    def test_focus_on_object_fits_the_object_in_the_frame(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mathutils: Any,
        resolution: Tuple[int, int],
        data: Dict[str, Any],
        half: Tuple[float, float],
        offset: Tuple[float, float],
    ) -> None:
        """The camera stands where the bounding sphere fits the frame's narrower reach.

        It stood twice the box's diagonal away whatever its lens and aspect:
        the corners of the default cube fell just outside a 16:9 frame.
        """
        center = (3.0, -2.0, 1.0)
        cube = self._cube(offset=center)
        cam = self._make_camera_object(**data)

        result = self._focus(monkeypatch, cube, cam, resolution)

        reach = min(half[0] - abs(offset[0]), half[1] - abs(offset[1]))
        expected = math.sqrt(3) * 1.05 / math.sin(math.atan(reach))
        assert result["distance"] == pytest.approx(expected)
        assert result["look_at"] == pytest.approx(list(center))
        for corner in _box():
            x, y, depth = _in_frame(
                cam, [c + v for c, v in zip(center, corner)], half, offset
            )
            assert 0.0 < x < 1.0 and 0.0 < y < 1.0, (x, y)
            assert cam.data.clip_start < depth < cam.data.clip_end

    @pytest.mark.parametrize("offset", [(0.0, 0.0, 0.0), (-40.0, 25.0, -7.5)])
    @pytest.mark.parametrize("setup", list(CAMERA_SETUPS))
    def test_focus_on_object_aims_the_camera_whatever_its_parent_and_rotation_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mathutils: Any,
        setup: str,
        offset: tuple,
    ) -> None:
        """In world space the camera stands in front of and above the object and looks at it, level.

        focus_on_object always looks along the same diagonal, so the cases
        differ in the local transform that gets there: through a rig that
        moved after the camera was parented, or stored as a quaternion or
        axis-angle. Writing location and rotation_euler, as it did, misplaced
        those cameras.
        """
        cam = self._make_camera_object(setup=setup, scale=(1.5, 1.5, 1.5))
        world_scale = list(cam.matrix_world.to_scale())

        result = self._focus(monkeypatch, self._cube(offset=offset), cam)

        right, up, forward = _world_axes(cam)
        location = np.asarray(cam.matrix_world.rows)[:3, 3]
        diagonal = np.array([0.0, -math.sqrt(0.5), math.sqrt(0.5)])
        assert location == pytest.approx(np.add(offset, diagonal * result["distance"]))
        assert result["camera_location"] == pytest.approx(list(location))
        assert forward == pytest.approx(-diagonal)
        assert right[2] == pytest.approx(0.0, abs=1e-9)
        assert up[2] > 0.0
        assert list(cam.matrix_world.to_scale()) == pytest.approx(world_scale)

    def test_focus_on_object_distance_factor_scales_the_framing_distance(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        """2.5 stands 2.5 times as far back on the same line as 1.0, the default."""
        cam = self._make_camera_object()
        default = self._focus(monkeypatch, self._cube(), cam)
        farther = self._focus(monkeypatch, self._cube(), cam, distance_factor=2.5)

        assert farther["distance"] == pytest.approx(2.5 * default["distance"])
        assert farther["camera_location"] == pytest.approx(
            [2.5 * v for v in default["camera_location"]]
        )

    @pytest.mark.parametrize(
        ("half", "start_moved", "end_moved"),
        [
            # A 60 m cube stands about 275 m away, past the 100 m clip_end.
            (30.0, False, True),
            # A 1 cm cube stands about 5 cm away, nearer than the 10 cm clip_start.
            (0.005, True, False),
            # The default cube inside the default clip range changes nothing.
            (1.0, False, False),
        ],
        ids=["far", "near", "inside"],
    )
    def test_focus_on_object_widens_a_clip_range_that_cuts_the_object_away(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mathutils: Any,
        half: float,
        start_moved: bool,
        end_moved: bool,
    ) -> None:
        """Every corner lies inside the clip range afterwards, and the reply says so.

        The 60 m cube used to land 165 to 250 m away, beyond clip_end, so the
        camera saw nothing while the tool reported success.
        """
        cam = self._make_camera_object(clip_start=0.1, clip_end=100.0)

        result = self._focus(monkeypatch, self._cube(half=half), cam)

        depths = [
            _in_frame(cam, [half * v for v in corner], (0.36, 0.2025))[2]
            for corner in _box()
        ]
        assert cam.data.clip_start < min(depths) <= max(depths) < cam.data.clip_end
        assert (result["clip_start"], result["clip_end"]) == (
            cam.data.clip_start,
            cam.data.clip_end,
        )
        assert (cam.data.clip_start != 0.1) is start_moved
        assert (cam.data.clip_end != 100.0) is end_moved
        assert result["clip_adjusted"] is (start_moved or end_moved)

    def test_focus_on_object_fits_an_orthographic_camera_by_its_scale(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        """An orthographic frame is ortho_scale wide whatever the distance, so that is what changes."""
        cam = self._make_camera_object(camera_type="ORTHO")

        result = self._focus(monkeypatch, self._cube(), cam)

        # AUTO spreads ortho_scale over the 16:9 frame's width.
        half = (0.5, 0.5 * 9 / 16)
        expected = math.sqrt(3) * 1.05 / half[1]
        assert result["ortho_scale"] == pytest.approx(expected)
        assert cam.data.ortho_scale == pytest.approx(expected)
        assert result["distance"] == pytest.approx(2 * math.sqrt(3))
        for corner in _box():
            x, y, depth = _in_frame(cam, corner, half, ortho_scale=expected)
            assert 0.0 < x < 1.0 and 0.0 < y < 1.0, (x, y)
            assert cam.data.clip_start < depth < cam.data.clip_end

    def test_focus_on_object_frames_a_point_as_a_unit_sphere(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        """An empty's bounding box is one point; it gets the size of an empty's display."""
        empty = self._cube(
            "Target", offset=(2.0, 3.0, 4.0), half=0.0, object_type="EMPTY"
        )
        cam = self._make_camera_object()

        result = self._focus(monkeypatch, empty, cam)

        assert result["look_at"] == pytest.approx([2.0, 3.0, 4.0])
        assert result["distance"] == pytest.approx(
            1.05 / math.sin(math.atan(18 / 50 * 9 / 16))
        )

    def test_focus_on_object_refuses_a_shift_that_moves_the_frame_off_the_axis(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        """With shift_y 0.5 the 16:9 frame lies wholly above the axis: no distance centers the object."""
        cam = self._make_camera_object(shift_y=0.5)

        with pytest.raises(ValueError, match="lens shift"):
            self._focus(monkeypatch, self._cube(), cam)
        assert list(cam.location) == [0.0, 0.0, 5.0]

    # Background Blender and the bpy module (embedded mode) recompute an
    # object's matrix_world only when the depsgraph is evaluated, so what an
    # earlier call wrote stays out of it until then. Read as it was, Blender
    # 5.0.1 gave the three results these tests rule out.

    def test_get_camera_info_reports_a_move_written_since_the_last_evaluation(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        """It reported the camera where it stood before set_object_transform moved it."""
        cam = self._make_camera_object(stored=True)
        fake_bpy = self._make_fake_bpy_with_camera(objects=[cam])
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        session = blender_runtime.BlenderRuntimeSession()
        session.set_object_transform("Camera", location=[1.0, 2.0, 3.0])

        result = session.get_camera_info()

        assert result["location"] == pytest.approx([1.0, 2.0, 3.0])

    def test_focus_on_object_keeps_a_scale_written_since_the_last_evaluation(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        """It put back the scale the camera had before set_object_transform scaled it."""
        cam = self._make_camera_object(stored=True)
        cam.scale = FakeVector((2.0, 2.0, 2.0))

        self._focus(monkeypatch, self._cube(), cam)

        assert list(cam.scale) == pytest.approx([2.0, 2.0, 2.0])

    def test_get_bounding_box_reports_a_move_written_since_the_last_evaluation(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        """World-space corners came from the matrix_world before the move."""
        cube = FakeTransformObject("Cube", object_type="MESH", stored=True)
        cube.bound_box = _box(-1.0, 1.0)
        fake_bpy = self._make_fake_bpy_with_camera(objects=[cube])
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        session = blender_runtime.BlenderRuntimeSession()
        session.set_object_transform("Cube", location=[10.0, 0.0, 0.0])

        result = session.get_bounding_box("Cube", world_space=True)

        assert result["bbox_min"] == pytest.approx([9.0, -1.0, -1.0])
        assert result["bbox_max"] == pytest.approx([11.0, 1.0, 1.0])

    def test_set_camera_view_places_a_camera_whose_rig_moved_since_the_last_evaluation(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        """It solved the camera's local transform through the rig's old matrix,
        so the camera landed off by the rig's move, 10 m, and the reply said
        it was where asked."""
        cam = self._make_camera_object(setup="parented", stored=True)
        cam.parent.location = FakeVector((18.0, 1.0, -1.0))
        fake_bpy = self._make_fake_bpy_with_camera(objects=[cam])
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        location, rotation = [7.0, -7.0, 5.0], [1.1, 0.25, 0.78]

        session = blender_runtime.BlenderRuntimeSession()
        session.set_camera_view(location=location, rotation_euler=rotation)
        fake_bpy.context.evaluated_depsgraph_get()

        world = np.asarray(cam.matrix_world.rows)
        assert world[:3, 3] == pytest.approx(location)
        rotation_part = world[:3, :3] / np.linalg.norm(world[:3, :3], axis=0)
        assert rotation_part == pytest.approx(euler_matrix(rotation))

    @pytest.mark.parametrize(
        "mirror",
        [(-1.5, 1.5, 1.5), (1.5, -1.5, 1.5), (1.5, 1.5, -1.5)],
        ids=["x", "y", "z"],
    )
    @pytest.mark.parametrize("setup", ["unparented", "parented_quaternion"])
    def test_set_camera_view_turns_a_mirrored_camera_as_asked(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mathutils: Any,
        mirror: Tuple[float, float, float],
        setup: str,
    ) -> None:
        """A negative scale flips the camera's image; the camera still looks as asked.

        Matrix.to_scale() of a mirrored matrix is negative on every axis, and
        placing the camera with it turned it to look straight back: on Blender
        5.0.1 its view direction was the opposite of the one asked, and the
        reply gave (1.1, 3.39, 0.78) for (1.1, 0.25, 0.78). The mirror stays,
        on the camera's X axis, and so does the size of its scale.
        """
        cam = self._make_camera_object(setup=setup, scale=mirror)
        world_scale = np.linalg.norm(np.asarray(cam.matrix_world.rows)[:3, :3], axis=0)
        fake_bpy = self._make_fake_bpy_with_camera(objects=[cam])
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        location, rotation = [7.0, -7.0, 5.0], [1.1, 0.25, 0.78]

        session = blender_runtime.BlenderRuntimeSession()
        result = session.set_camera_view(location=location, rotation_euler=rotation)

        world = np.asarray(cam.matrix_world.rows)
        axes = world[:3, :3] / world_scale
        asked = euler_matrix(rotation)
        assert world[:3, 3] == pytest.approx(location)
        # Looking down the asked -Z with the asked +Y up, flipped left to right.
        assert axes @ [0.0, 0.0, -1.0] == pytest.approx(asked @ [0.0, 0.0, -1.0])
        assert axes @ [0.0, 1.0, 0.0] == pytest.approx(asked @ [0.0, 1.0, 0.0])
        assert axes @ [1.0, 0.0, 0.0] == pytest.approx(-(asked @ [1.0, 0.0, 0.0]))
        assert np.linalg.norm(world[:3, :3], axis=0) == pytest.approx(world_scale)
        assert result["location"] == pytest.approx(location)
        assert result["rotation_euler"] == pytest.approx(rotation)
        # Read back later the angles may differ by whole turns, not the rotation.
        info = session.get_camera_info()["rotation_euler"]
        assert euler_matrix(info) == pytest.approx(asked)

    def test_focus_on_object_keeps_a_mirrored_camera_facing_the_object(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        """A camera scaled -1 on X was turned away: the object stood behind it."""
        center = (3.0, -2.0, 1.0)
        cam = self._make_camera_object(scale=(-1.0, 1.0, 1.0))

        self._focus(monkeypatch, self._cube(offset=center), cam)

        for corner in _box():
            x, y, depth = _in_frame(
                cam,
                [c + v for c, v in zip(center, corner)],
                (18 / 50, 18 / 50 * 9 / 16),
            )
            assert 0.0 < x < 1.0 and 0.0 < y < 1.0, (x, y)
            assert cam.data.clip_start < depth < cam.data.clip_end

    # -- constraints on the camera -----------------------------------------
    # Blender applies a camera's constraints on top of the transform a
    # placement writes when it next evaluates the depsgraph. These cameras
    # keep matrix_world until then (stored=True), as in Blender, so a
    # constraint acts only on evaluation.

    def _session_on(
        self, monkeypatch: pytest.MonkeyPatch, *objects: Any
    ) -> blender_runtime.BlenderRuntimeSession:
        fake_bpy = self._make_fake_bpy_with_camera(objects=list(objects))
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        return blender_runtime.BlenderRuntimeSession()

    def _constrained_camera(
        self, *constraints: FakeConstraint, setup: str = "unparented", **camera: Any
    ) -> FakeTransformObject:
        cam = self._make_camera_object(setup=setup, stored=True, **camera)
        cam.constraints.extend(constraints)
        cam.evaluate()
        return cam

    @pytest.mark.parametrize(
        ("constraint", "moved", "turned"),
        [
            (_track_to((5.0, 5.0, 0.0)), False, True),
            (_copy_location((5.0, 5.0, 0.0)), True, False),
        ],
        ids=["track_to", "copy_location"],
    )
    def test_set_camera_view_reports_constraints_that_override_the_placement(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mathutils: Any,
        constraint: FakeConstraint,
        moved: bool,
        turned: bool,
    ) -> None:
        """A Track To turned the camera to its target, a Copy Location moved it,
        and the reply gave the requested pose as the camera's.

        On Blender 5.0.1 a camera tracking another object was reported at
        rotation (1.1, 0.25, 0.78) and evaluated at (1.205, 0, 0.077).
        """
        cam = self._constrained_camera(constraint)
        session = self._session_on(monkeypatch, cam)
        location, rotation = [7.0, -7.0, 5.0], [1.1, 0.25, 0.78]

        result = session.set_camera_view(location=location, rotation_euler=rotation)

        # Blender evaluates the depsgraph before it draws or renders the camera.
        cam.evaluate()
        world = np.asarray(cam.matrix_world.rows)
        assert result["location"] == pytest.approx(list(world[:3, 3]))
        assert euler_matrix(result["rotation_euler"]) == pytest.approx(
            _evaluated_rotation(cam)
        )
        assert (not np.allclose(result["location"], location)) is moved
        turned_by = euler_matrix(result["rotation_euler"])
        assert (not np.allclose(turned_by, euler_matrix(rotation))) is turned
        assert result["placement_overridden"] is True
        assert result["constraints_applied"] == [
            {"name": constraint.name, "type": constraint.type}
        ]
        assert f"'{constraint.name}' ({constraint.type})" in result["warning"]
        assert "mute" in result["warning"] and "influence to 0" in result["warning"]
        # The camera's own transform holds the placement; the constraint is untouched.
        basis = np.asarray(cam.matrix_basis.rows)
        assert basis[:3, 3] == pytest.approx(location)
        assert basis[:3, :3] == pytest.approx(euler_matrix(rotation))
        assert (constraint.mute, constraint.influence) == (False, 1.0)

    @pytest.mark.parametrize(
        ("constraint", "applied"),
        [
            (_track_to((5.0, 5.0, 0.0), mute=True), []),
            (_track_to((5.0, 5.0, 0.0), influence=0.0), []),
            # A Track To without a target: Blender marks it invalid and skips it.
            (_track_to((5.0, 5.0, 0.0), is_valid=False), []),
            (_satisfied(), [{"name": "Limit Distance", "type": "LIMIT_DISTANCE"}]),
        ],
        ids=["muted", "no_influence", "invalid", "satisfied"],
    )
    def test_set_camera_view_reports_a_placement_no_constraint_changes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mathutils: Any,
        constraint: FakeConstraint,
        applied: List[Dict[str, str]],
    ) -> None:
        """Constraints that Blender skips, or that leave the pose alone, override nothing."""
        cam = self._constrained_camera(constraint)
        session = self._session_on(monkeypatch, cam)
        location, rotation = [7.0, -7.0, 5.0], [1.1, 0.25, 0.78]

        result = session.set_camera_view(location=location, rotation_euler=rotation)

        assert result["location"] == pytest.approx(location)
        assert result["rotation_euler"] == pytest.approx(rotation)
        assert result["placement_overridden"] is False
        assert result["constraints_applied"] == applied
        assert result["warning"] is None

    def test_set_camera_view_reports_an_unconstrained_placement_held(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        cam = self._make_camera_object(setup="parented_quaternion", stored=True)
        session = self._session_on(monkeypatch, cam)

        result = session.set_camera_view([7.0, -7.0, 5.0], [1.1, 0.25, 0.78])

        assert result["placement_overridden"] is False
        assert result["constraints_applied"] == []
        assert result["warning"] is None

    @pytest.mark.parametrize(
        ("setup", "scale"),
        [
            ("unparented", (1.0, 1.0, 1.0)),
            ("parented", (1.0, 1.0, 1.0)),
            # A scale of the camera's own stays as it was, not reset to 1.
            ("unparented", (0.5, 0.5, 0.5)),
        ],
        ids=["unparented", "parented", "own_scale"],
    )
    def test_set_camera_view_keeps_the_scale_of_a_camera_its_child_of_scales(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mathutils: Any,
        setup: str,
        scale: Tuple[float, float, float],
    ) -> None:
        """Each call doubled the scale of a camera whose Child Of target had
        been scaled by 2 after the constraint was set up.

        The placement kept the evaluated scale, the constraint's included, in
        the camera's own scale, where the constraint applied it again: 2, 4
        and 8 on Blender 5.0.1, the camera evaluated at (14, -14, 10) each
        time and the reply at (7, -7, 5). A camera of scale 0.5 kept 0.5 on
        Blender 5.0.1.
        """
        cam = self._constrained_camera(
            _child_of(np.diag([2.0, 2.0, 2.0, 1.0])), setup=setup, scale=scale
        )
        own_scale = list(cam.scale)
        session = self._session_on(monkeypatch, cam)

        for _ in range(3):
            result = session.set_camera_view([7.0, -7.0, 5.0], [1.1, 0.25, 0.78])
            assert list(cam.scale) == pytest.approx(own_scale)

        cam.evaluate()
        assert list(cam.matrix_world.translation) == pytest.approx([14.0, -14.0, 10.0])
        assert result["location"] == pytest.approx([14.0, -14.0, 10.0])
        assert result["placement_overridden"] is True
        assert result["constraints_applied"] == [
            {"name": "Child Of", "type": "CHILD_OF"}
        ]

    @pytest.mark.parametrize(
        ("parent_type", "parent_object"),
        [("OBJECT", "CURVE"), ("BONE", "ARMATURE"), ("VERTEX", "MESH")],
        ids=["curve_path", "bone", "vertex"],
    )
    def test_matrix_before_constraints_falls_back_where_only_blender_knows_the_parent(
        self, mathutils: Any, parent_type: str, parent_object: str
    ) -> None:
        """A curve's path, a bone or vertices place the child in ways only
        Blender computes; the evaluated matrix is the best estimate left."""
        parent = FakeTransformObject("Parent", object_type=parent_object)
        cam = FakeTransformObject("Camera", "CAMERA", parent=parent)
        cam.parent_type = parent_type
        cam.constraints.append(_child_of(np.diag([2.0, 2.0, 2.0, 1.0])))

        own = blender_runtime.BlenderRuntimeSession._matrix_before_constraints(cam)

        assert own.rows == cam.matrix_world.rows

    def test_set_camera_view_reports_a_rotation_a_non_uniform_parent_cannot_carry(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        """Under a parent scaled (1, 3, 1) no local transform gives the asked rotation.

        Blender 5.0.1 turned such a camera to look along (-0.41, 0.88, -0.25)
        for (-0.71, 0.56, -0.44), no constraint involved.
        """
        rig = FakeTransformObject("Rig", scale=(1.0, 3.0, 1.0), stored=True)
        cam = FakeTransformObject(
            "Camera", "CAMERA", FakeCameraData(), parent=rig, stored=True
        )
        # Parented without keeping the transform: the rig's scale reaches the camera.
        cam.matrix_parent_inverse = FakeMatrix.Identity(4)
        cam.evaluate()
        session = self._session_on(monkeypatch, cam)

        result = session.set_camera_view([7.0, -7.0, 5.0], [1.1, 0.25, 0.78])

        assert result["location"] == pytest.approx([7.0, -7.0, 5.0])
        assert result["placement_overridden"] is True
        assert result["constraints_applied"] == []
        assert "no constraint acts on it" in result["warning"]
        assert "non-uniform scale" in result["warning"]

    @staticmethod
    def _sheared_camera() -> FakeTransformObject:
        """A camera under a rig scaled (1, 3, 1), its scale reaching the camera."""
        rig = FakeTransformObject("Rig", scale=(1.0, 3.0, 1.0), stored=True)
        cam = FakeTransformObject(
            "Camera", "CAMERA", FakeCameraData(), parent=rig, stored=True
        )
        cam.matrix_parent_inverse = FakeMatrix.Identity(4)
        return cam

    @pytest.mark.parametrize("cause", ["driver", "sheared_parent"])
    def test_set_camera_view_blames_no_constraint_that_leaves_the_pose_alone(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any, cause: str
    ) -> None:
        """A driver, or a parent scaled unevenly, moved the camera; its Limit
        Distance, met, changed nothing.

        The warning named the Limit Distance as the override and said to mute
        it, which leaves the camera where it was. On Blender 5.0.1 a camera
        with a driver of 3 on location.x was evaluated at x = 3, its Limit
        Distance (1000 m) met, and the warning named only the Limit Distance.
        """
        cam = _DrivenCamera(3.0) if cause == "driver" else self._sheared_camera()
        cam.constraints.append(_satisfied())
        cam.evaluate()
        session = self._session_on(monkeypatch, cam)

        result = session.set_camera_view([7.0, -7.0, 5.0], [1.1, 0.25, 0.78])

        if cause == "driver":
            assert result["location"] == pytest.approx([3.0, -7.0, 5.0])
        assert result["placement_overridden"] is True
        assert result["constraints_applied"] == [
            {"name": "Limit Distance", "type": "LIMIT_DISTANCE"}
        ]
        warning = result["warning"]
        assert "before its constraints apply" in warning
        assert "a driver on its transform" in warning and "non-uniform scale" in warning
        assert "mute" not in warning and "override the requested placement" not in warning

    def test_set_camera_view_names_constraints_and_a_driver_that_both_move_the_camera(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        """Muting the Copy Location alone would leave the camera where its driver puts it."""
        cam = _DrivenCamera(3.0)
        cam.constraints.append(_copy_location((5.0, 5.0, 0.0)))
        cam.evaluate()
        session = self._session_on(monkeypatch, cam)

        result = session.set_camera_view([7.0, -7.0, 5.0], [1.1, 0.25, 0.78])

        assert result["location"] == pytest.approx([5.0, 5.0, 0.0])
        assert result["placement_overridden"] is True
        warning = result["warning"]
        assert "override the requested placement: 'Copy Location' (COPY_LOCATION)" in warning
        assert "mute" in warning and "influence to 0" in warning
        assert "a driver on its transform" in warning and "as well" in warning
        assert "its transform holds the placement" not in warning

    def test_set_camera_view_tolerates_single_precision_through_a_far_parent_inverse(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        """A camera parented to a rig 27 km out, and the rig then moved back.

        The camera's own location is 27 km long, and single precision rounds
        it by up to a millimetre, so Blender evaluates the camera that far
        from the placement although nothing overrides it. The tolerance
        followed the placed, evaluated and parent locations only, all near the
        origin, and reported an override with a warning blaming a driver or
        the parent's scale: on Blender 5.0.1, for 273 of 300 such placements,
        which missed by up to 2 cm with the rig 100 km out.
        """
        rig = FakeTransformObject("Rig", location=(27000.0, 0.0, 0.0), stored=True)
        cam = _SinglePrecisionCamera(
            "Camera", "CAMERA", FakeCameraData(), parent=rig, stored=True
        )
        rig.location = FakeVector((0.0, 0.0, 0.0))
        cam.evaluate()
        session = self._session_on(monkeypatch, cam)
        location = [7.3, -7.1, 5.2]

        result = session.set_camera_view(location, [1.1, 0.25, 0.78])

        assert list(cam.location)[0] == pytest.approx(27007.3, abs=1e-3)
        # Single precision put the camera a little off the placement...
        assert math.dist(result["location"], location) > 1e-4
        # ...which is no override.
        assert result["placement_overridden"] is False
        assert result["warning"] is None

    def test_focus_on_object_reports_a_track_to_that_turns_the_camera_away(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        """The camera tracked another object, so the focused cube left the frame
        while the reply described the framing as done.

        On Blender 5.0.1 the cube's center appeared at (0.015, -0.543) in the
        camera's view, below the frame, and the tracked object at its center.
        """
        decoy = (6.0, 6.0, 0.0)
        cam = self._constrained_camera(_track_to(decoy))

        result = self._focus(monkeypatch, self._cube(), cam)

        # Blender evaluates the depsgraph before it draws or renders the camera.
        cam.evaluate()
        half = (18 / 50, 18 / 50 * 9 / 16)
        assert _in_frame(cam, decoy, half)[:2] == pytest.approx((0.5, 0.5))
        corners = [_in_frame(cam, corner, half) for corner in _box()]
        assert not all(0.0 < x < 1.0 and 0.0 < y < 1.0 for x, y, _ in corners)
        assert result["look_at"] == pytest.approx([0.0, 0.0, 0.0])
        world = np.asarray(cam.matrix_world.rows)
        assert result["camera_location"] == pytest.approx(list(world[:3, 3]))
        assert euler_matrix(result["camera_rotation_euler"]) == pytest.approx(
            _evaluated_rotation(cam)
        )
        assert result["placement_overridden"] is True
        assert result["constraints_applied"] == [
            {"name": "Track To", "type": "TRACK_TO"}
        ]
        assert "'Track To' (TRACK_TO)" in result["warning"]

    def test_focus_on_object_reports_the_framing_held_without_constraints(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        cam = self._make_camera_object(stored=True)

        result = self._focus(monkeypatch, self._cube(), cam)

        diagonal = np.array([0.0, -math.sqrt(0.5), math.sqrt(0.5)])
        assert result["camera_location"] == pytest.approx(
            list(diagonal * result["distance"])
        )
        rotation = blender_runtime.BlenderRuntimeSession._look_at_rotation(
            result["camera_location"], [0.0, 0.0, 0.0]
        )
        assert result["camera_rotation_euler"] == pytest.approx(list(rotation))
        assert result["placement_overridden"] is False
        assert result["constraints_applied"] == []
        assert result["warning"] is None

    def test_get_camera_info_lists_the_constraints_that_act_on_the_camera(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        """Only the constraints Blender applies, and the pose they give the camera."""
        target = (5.0, 5.0, 0.0)
        cam = self._constrained_camera(
            _copy_location((1.0, 1.0, 1.0), mute=True),
            _track_to(target),
            _satisfied(influence=0.0),
            _copy_location((2.0, 2.0, 2.0), name="Broken", is_valid=False),
        )
        session = self._session_on(monkeypatch, cam)

        result = session.get_camera_info()

        assert result["constraints_applied"] == [
            {"name": "Track To", "type": "TRACK_TO"}
        ]
        assert result["location"] == pytest.approx([0.0, 0.0, 5.0])
        assert euler_matrix(result["rotation_euler"]) == pytest.approx(
            _evaluated_rotation(cam)
        )
        to_target = np.subtract(target, result["location"])
        forward = euler_matrix(result["rotation_euler"]) @ [0.0, 0.0, -1.0]
        assert forward == pytest.approx(to_target / np.linalg.norm(to_target))

    # -- cameras Blender does not evaluate for the viewport -----------------
    # evaluated_depsgraph_get() leaves out a camera disabled in viewports that
    # is not the scene camera: on Blender 5.0.1 its constraints never ran
    # there, while a render through it applied them. ``left_out`` has the fake
    # depsgraph skip such a camera as Blender does.

    @staticmethod
    def _side_camera(
        how: str, *constraints: FakeConstraint
    ) -> Tuple[FakeTransformObject, FakeViewLayer]:
        """
        A camera "Side" in collection Cameras, itself in collection Rig.

        Args:
            how: "enabled"; disabled in viewports by "object" (its own
                hide_viewport), "collection" (Cameras disabled in viewports),
                "parent_collection" (Rig disabled), "excluded" (Cameras
                excluded from the view layer) or "collection_linked_twice"
                (Cameras disabled, and linked into Rig and Other); or enabled
                although "excluded_parent" (Rig excluded, Cameras included
                again) or "second_collection" (Cameras excluded, Side also in
                Other); "nowhere" (in no collection of the view layer).
            constraints: The camera's constraints.

        Returns:
            The camera, evaluated, and the view layer.
        """
        side = FakeTransformObject(
            "Side",
            "CAMERA",
            FakeCameraData(),
            location=(0.0, 0.0, 5.0),
            stored=True,
            constraints=constraints,
        )
        rig, cameras, other = (
            FakeCollection("Rig"),
            FakeCollection("Cameras"),
            FakeCollection("Other"),
        )
        cameras_layer = FakeLayerCollection(cameras)
        rig_layer = FakeLayerCollection(rig, [cameras_layer])
        other_layer = FakeLayerCollection(other)
        view_layer = FakeViewLayer(
            FakeLayerCollection(
                FakeCollection("Scene Collection"), [rig_layer, other_layer]
            )
        )
        if how != "nowhere":
            cameras.link(side)
        if how == "object":
            side.hide_viewport = True
        elif how == "collection":
            cameras.hide_viewport = True
        elif how == "parent_collection":
            rig.hide_viewport = True
        elif how == "excluded":
            cameras_layer.exclude = True
        elif how == "collection_linked_twice":
            cameras.hide_viewport = True
            other_layer.children.append(FakeLayerCollection(cameras))
        elif how == "excluded_parent":
            rig_layer.exclude = True
        elif how == "second_collection":
            cameras_layer.exclude = True
            other.link(side)
        side.evaluate()
        return side, view_layer

    def _session_in(
        self,
        monkeypatch: pytest.MonkeyPatch,
        view_layer: FakeViewLayer,
        *objects: Any,
        left_out: Sequence[Any] = (),
        marker_cameras: Sequence[Any] = (),
    ) -> blender_runtime.BlenderRuntimeSession:
        """A session whose scene camera is "Camera", in ``view_layer``."""
        fake_bpy = self._make_fake_bpy_with_camera(
            objects=[self._make_camera_object(), *objects],
            view_layer=view_layer,
            left_out=left_out,
            marker_cameras=marker_cameras,
        )
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        return blender_runtime.BlenderRuntimeSession()

    @pytest.mark.parametrize(
        ("how", "reason"),
        [
            ("object", "it is disabled in viewports (hide_viewport)"),
            ("collection", "collection 'Cameras' is disabled in viewports"),
            ("parent_collection", "collection 'Rig' is disabled in viewports"),
            (
                "excluded",
                "its collection 'Cameras' is excluded from view layer 'ViewLayer'",
            ),
        ],
    )
    def test_set_camera_view_skips_the_check_for_a_camera_blender_does_not_evaluate(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any, how: str, reason: str
    ) -> None:
        """The Track To of a camera disabled in viewports never ran, and the
        reply said placement_overridden false; a render through the camera
        turned it to the Track To's target.

        On Blender 5.0.1 a render put such a camera at rotation
        (1.3657, 0, 0.7854) for a placement at (1.1, 0.25, 0.78), reported held.
        """
        side, view_layer = self._side_camera(how, _track_to((5.0, 5.0, 0.0)))
        session = self._session_in(monkeypatch, view_layer, side, left_out=[side])
        location, rotation = [7.0, -7.0, 5.0], [1.1, 0.25, 0.78]

        result = session.set_camera_view(location, rotation, camera_name="Side")

        assert result["placement_overridden"] is None
        assert result["override_check"] == "skipped"
        assert result["constraints_applied"] == [
            {"name": "Track To", "type": "TRACK_TO"}
        ]
        warning = result["warning"]
        clause = (
            "neither the scene camera nor a timeline marker's camera, and "
            f"{reason}, so the placement was not checked"
        )
        assert clause in warning
        assert "'Track To' (TRACK_TO)" in warning and "influence to 0" in warning
        # The reply gives the placement, which the camera's transform holds...
        assert result["location"] == pytest.approx(location)
        assert result["rotation_euler"] == pytest.approx(rotation)
        basis = np.asarray(side.matrix_basis.rows)
        assert basis[:3, 3] == pytest.approx(location)
        assert basis[:3, :3] == pytest.approx(euler_matrix(rotation))
        # ...which a render, evaluating the camera, does not keep.
        side.evaluate()
        assert not np.allclose(_evaluated_rotation(side), euler_matrix(rotation))

    def test_set_camera_view_skips_the_check_for_an_unevaluated_camera_without_constraints(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        """Nothing may move it, but a driver or a parent's scale would go unseen too."""
        side, view_layer = self._side_camera("object")
        session = self._session_in(monkeypatch, view_layer, side, left_out=[side])

        result = session.set_camera_view(
            [7.0, -7.0, 5.0], [1.1, 0.25, 0.78], camera_name="Side"
        )

        assert result["placement_overridden"] is None
        assert result["override_check"] == "skipped"
        assert result["constraints_applied"] == []
        assert "No constraint acts on it" in result["warning"]

    def test_set_camera_view_reports_the_placement_for_a_camera_evaluated_in_part(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        """Blender gave the camera's matrix_world back from before it was
        disabled, and the reply gave that pose, with placement_overridden true
        and a warning blaming its Track To.

        On Blender 5.0.1 the reply put the camera at (1, 2, 3), where it stood
        before it was disabled, for a placement at (7, -7, 5).
        """
        side = _PartlyEvaluatedCamera(_track_to((5.0, 5.0, 0.0)))
        root = FakeCollection("Scene Collection")
        root.link(side)
        session = self._session_in(
            monkeypatch, FakeViewLayer(FakeLayerCollection(root)), side
        )
        location, rotation = [7.0, -7.0, 5.0], [1.1, 0.25, 0.78]

        result = session.set_camera_view(location, rotation, camera_name="Side")

        assert list(side.matrix_world.translation) == pytest.approx([1.0, 2.0, 3.0])
        assert result["location"] == pytest.approx(location)
        assert result["rotation_euler"] == pytest.approx(rotation)
        assert result["placement_overridden"] is None
        assert result["override_check"] == "skipped"

    def test_focus_on_object_skips_the_check_for_a_camera_blender_does_not_evaluate(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        """The reply gives the framing placement, not the matrix Blender kept."""
        side = _PartlyEvaluatedCamera(_track_to((6.0, 6.0, 0.0)))
        root = FakeCollection("Scene Collection")
        root.link(side)
        session = self._session_in(
            monkeypatch, FakeViewLayer(FakeLayerCollection(root)), side, self._cube()
        )

        result = session.focus_on_object("Cube", camera_name="Side")

        diagonal = np.array([0.0, -math.sqrt(0.5), math.sqrt(0.5)])
        assert result["camera_location"] == pytest.approx(
            list(diagonal * result["distance"])
        )
        rotation = blender_runtime.BlenderRuntimeSession._look_at_rotation(
            result["camera_location"], [0.0, 0.0, 0.0]
        )
        assert result["camera_rotation_euler"] == pytest.approx(list(rotation))
        assert result["placement_overridden"] is None
        assert result["override_check"] == "skipped"
        assert result["constraints_applied"] == [
            {"name": "Track To", "type": "TRACK_TO"}
        ]
        assert "it is disabled in viewports (hide_viewport)" in result["warning"]

    @pytest.mark.parametrize("how", ["enabled", "excluded_parent", "second_collection"])
    def test_set_camera_view_checks_a_camera_a_collection_enables(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any, how: str
    ) -> None:
        """Excluding a collection leaves its child collections, and the
        camera's other collections, in the view layer: Blender 5.0.1
        evaluated such a camera."""
        side, view_layer = self._side_camera(how, _track_to((5.0, 5.0, 0.0)))
        session = self._session_in(monkeypatch, view_layer, side)

        result = session.set_camera_view(
            [7.0, -7.0, 5.0], [1.1, 0.25, 0.78], camera_name="Side"
        )

        assert result["override_check"] == "done"
        assert result["placement_overridden"] is True
        assert "override the requested placement" in result["warning"]

    @pytest.mark.parametrize("how", ["object", "excluded"])
    def test_set_camera_view_checks_a_camera_a_timeline_marker_binds(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any, how: str
    ) -> None:
        """Blender evaluates every camera a timeline marker binds, whatever its
        visibility and the current frame.

        On Blender 5.0.1 such a camera, disabled in viewports or in an excluded
        collection while another camera was the scene camera, was turned by its
        Track To in the viewport depsgraph, and the reply reported the override;
        the check must not be skipped for it.
        """
        side, view_layer = self._side_camera(how, _track_to((5.0, 5.0, 0.0)))
        session = self._session_in(monkeypatch, view_layer, side, marker_cameras=[side])

        result = session.set_camera_view(
            [7.0, -7.0, 5.0], [1.1, 0.25, 0.78], camera_name="Side"
        )

        assert result["override_check"] == "done"
        assert result["placement_overridden"] is True
        assert "override the requested placement" in result["warning"]
        assert blender_runtime.BlenderRuntimeSession._viewport_disablers(side) == []

    def test_set_camera_view_checks_the_scene_camera_disabled_in_viewports(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        """Blender evaluates the scene camera whatever its visibility (5.0.1)."""
        cam = self._constrained_camera(_track_to((5.0, 5.0, 0.0)))
        cam.hide_viewport = True
        session = self._session_on(monkeypatch, cam)

        result = session.set_camera_view([7.0, -7.0, 5.0], [1.1, 0.25, 0.78])

        assert result["override_check"] == "done"
        assert result["placement_overridden"] is True

    @pytest.mark.parametrize(
        ("how", "reasons"),
        [
            ("enabled", []),
            ("excluded_parent", []),
            ("second_collection", []),
            ("object", ["it is disabled in viewports (hide_viewport)"]),
            ("parent_collection", ["collection 'Rig' is disabled in viewports"]),
            # One reason per setting, though Cameras is in the view layer twice.
            (
                "collection_linked_twice",
                ["collection 'Cameras' is disabled in viewports"],
            ),
            ("nowhere", ["it is in no collection of view layer 'ViewLayer'"]),
        ],
    )
    def test_viewport_disablers_name_each_setting_that_disables_the_camera(
        self, monkeypatch: pytest.MonkeyPatch, how: str, reasons: List[str]
    ) -> None:
        side, view_layer = self._side_camera(how)
        self._session_in(monkeypatch, view_layer, side)

        disablers = blender_runtime.BlenderRuntimeSession._viewport_disablers(side)

        assert disablers == reasons

    def test_viewport_disablers_ignore_the_scene_collection_hide_viewport(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Blender ignores the flag on the scene's own collection (5.0.1)."""
        root = FakeCollection("Scene Collection", hide_viewport=True)
        side, _ = self._side_camera("nowhere")
        root.link(side)
        self._session_in(monkeypatch, FakeViewLayer(FakeLayerCollection(root)), side)

        assert blender_runtime.BlenderRuntimeSession._viewport_disablers(side) == []

    @pytest.mark.parametrize(
        ("location", "rotation", "scale", "extent", "same"),
        [
            ((1.0, 2.0, 3.0), (0.4, 0.5, 0.6), (1.0, 1.0, 1.0), 4.0, True),
            # Scale does not change what a camera sees.
            ((1.0, 2.0, 3.0), (0.4, 0.5, 0.6), (2.0, 2.0, 2.0), 4.0, True),
            # Single precision blurs 50 micrometres, not a millimetre...
            ((1.00005, 2.0, 3.0), (0.4, 0.5, 0.6), (1.0, 1.0, 1.0), 4.0, True),
            ((1.001, 2.0, 3.0), (0.4, 0.5, 0.6), (1.0, 1.0, 1.0), 4.0, False),
            # ...except 10 km out, where a millimetre is within its reach.
            ((1.001, 2.0, 3.0), (0.4, 0.5, 0.6), (1.0, 1.0, 1.0), 1.0e4, True),
            # A turn about the camera's X axis turns both compared axes by as much.
            ((1.0, 2.0, 3.0), (0.40001, 0.5, 0.6), (1.0, 1.0, 1.0), 4.0, True),
            ((1.0, 2.0, 3.0), (0.401, 0.5, 0.6), (1.0, 1.0, 1.0), 4.0, False),
            # A mirror flips the image even where the axes compared agree.
            ((1.0, 2.0, 3.0), (0.4, 0.5, 0.6), (-1.0, 1.0, 1.0), 4.0, False),
        ],
        ids=[
            "same",
            "scaled",
            "within_precision",
            "moved_1mm",
            "moved_1mm_far_out",
            "turned_within_precision",
            "turned_1mrad",
            "mirrored",
        ],
    )
    def test_same_view_tolerates_single_precision_only(
        self,
        location: Tuple[float, float, float],
        rotation: Tuple[float, float, float],
        scale: Tuple[float, float, float],
        extent: float,
        same: bool,
    ) -> None:
        placed = FakeMatrix.LocRotScale(
            FakeVector((1.0, 2.0, 3.0)),
            FakeEuler((0.4, 0.5, 0.6)),
            FakeVector((1.0, 1.0, 1.0)),
        )
        other = FakeMatrix.LocRotScale(
            FakeVector(location), FakeEuler(rotation), FakeVector(scale)
        )

        assert (
            blender_runtime.BlenderRuntimeSession._same_view(placed, other, extent)
            is same
        )

    def test_focus_on_object_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """focus_on_object raises ValueError for missing object."""
        cam = self._make_camera_object()
        fake_bpy = self._make_fake_bpy_with_camera(objects=[cam])
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)

        session = blender_runtime.BlenderRuntimeSession()
        with pytest.raises(ValueError, match="Object not found"):
            session.focus_on_object("NonExistent")

    @pytest.mark.parametrize(
        "direction",
        [
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (1.0, 2.0, -3.0),
            (-2.0, -1.0, 0.5),
            (0.3, -0.4, 5.0),
            (-0.2, 0.1, -8.0),
            (0.0, 0.0, -1.0),
            (0.0, 0.0, 1.0),
        ],
    )
    def test_look_at_rotation_turns_a_level_camera_to_the_target(
        self, direction: Tuple[float, float, float]
    ) -> None:
        """Each direction gets its own rotation: the camera's -Z runs to the target,
        its X axis stays horizontal (no roll) and its Y axis does not point down."""
        location = [1.0, -2.0, 0.5]
        target = [p + d for p, d in zip(location, direction)]

        rotation = blender_runtime.BlenderRuntimeSession._look_at_rotation(
            location, target
        )

        length = math.sqrt(sum(d * d for d in direction))
        assert _camera_forward(rotation) == pytest.approx(
            [d / length for d in direction]
        )
        assert _rotate_xyz(rotation, (1.0, 0.0, 0.0))[2] == pytest.approx(
            0.0, abs=1e-12
        )
        assert _rotate_xyz(rotation, (0.0, 1.0, 0.0))[2] >= -1e-12

    # capture_viewport's method selection, viewport and render paths are
    # covered against a fuller bpy stand-in in test_viewport_capture.py.

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
        """Unparent with keep_transform re-applies the full world matrix.

        Copying only the translation dropped the parent's rotation and scale;
        the whole matrix must be written back once the parent is gone.
        """
        world_matrix = SimpleNamespace(label="rotated+scaled world")
        assigned: List[Any] = []

        class RecordingChild(FakeObject):
            def __setattr__(self, name: str, value: Any) -> None:
                if name == "matrix_world" and "parent" in self.__dict__:
                    assigned.append((value, self.parent))
                super().__setattr__(name, value)

        parent = FakeObject("Parent", "MESH", True, location=(10.0, 0.0, 0.0))
        child = RecordingChild("Child", "MESH", True, location=(5.0, 0.0, 0.0), parent=parent)
        child.__dict__["matrix_world"] = SimpleNamespace(copy=lambda: world_matrix)
        assigned.clear()  # ignore the constructor's own assignment
        objs = {"Child": child, "Parent": parent}
        fake_bpy = self._make_fake_bpy_with_manipulation(objects=objs)
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.clear_object_parent("Child", keep_transform=True)

        assert result["previous_parent"] == "Parent"
        assert child.parent is None
        # The copied world matrix is written after the parent is cleared.
        assert assigned == [(world_matrix, None)]

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
            session.open_blend_file("/tmp/simul-work/nonexistent/path.blend")

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
        result = session.save_blend_file(file_path="/tmp/simul-work/out.blend")

        assert result["file_path"] == str(Path("/tmp/simul-work/out.blend").resolve())
        assert fake_bpy._ops_calls[0]["op"] == "wm.save_as_mainfile"

    def test_save_blend_file_in_place(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Save without path calls save_mainfile when filepath is set."""
        fake_bpy = self._make_fake_bpy_with_file_io(filepath="/tmp/simul-work/existing.blend")
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.save_blend_file()

        assert result["file_path"] == "/tmp/simul-work/existing.blend"
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
            session.import_file("/tmp/simul-work/file.abc", "ABC")

    def test_import_file_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Importing a nonexistent file raises FileNotFoundError."""
        fake_bpy = self._make_fake_bpy_with_file_io()
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        with pytest.raises(FileNotFoundError, match="File not found"):
            session.import_file("/tmp/simul-work/nonexistent/model.obj", "OBJ")

    # -- export_file ---------------------------------------------------------

    def test_export_obj_v4(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """OBJ export on Blender 4.x uses wm.obj_export."""
        fake_bpy = self._make_fake_bpy_with_file_io(version=(4, 2, 0))
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.export_file("/tmp/simul-work/out.obj", "OBJ")

        assert result["file_format"] == "OBJ"
        assert result["file_path"] == str(Path("/tmp/simul-work/out.obj").resolve())
        assert fake_bpy._ops_calls[0]["op"] == "wm.obj_export"

    def test_export_obj_v36(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """OBJ export on Blender 3.6 uses export_scene.obj."""
        fake_bpy = self._make_fake_bpy_with_file_io(version=(3, 6, 0))
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.export_file("/tmp/simul-work/out.obj", "OBJ")

        assert result["file_format"] == "OBJ"
        assert fake_bpy._ops_calls[0]["op"] == "export_scene.obj"

    def test_export_selected_only_v4(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Export with selected_only on v4 passes export_selected_objects."""
        fake_bpy = self._make_fake_bpy_with_file_io(version=(4, 0, 0))
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        session.export_file("/tmp/simul-work/out.obj", "OBJ", selected_only=True)

        call = fake_bpy._ops_calls[0]
        assert call["export_selected_objects"] is True

    def test_export_unsupported_format(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Exporting an unsupported format raises ValueError."""
        fake_bpy = self._make_fake_bpy_with_file_io()
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        with pytest.raises(ValueError, match="Unsupported format"):
            session.export_file("/tmp/simul-work/out.abc", "ABC")


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

    # insert/delete/get_keyframes: tests/blender/test_keyframes.py covers them
    # against the action API of each Blender version.

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
                filepath="/tmp/simul-work/test.blend",
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
        assert "temp_override" not in result["error"]

    def test_execute_script_context_member_error_gets_override_recipe(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Issue #215: a missing bpy.context member names the override that fixes it."""
        fake_bpy = self._make_fake_bpy_for_script()
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.execute_script(
            "class Context:\n    pass\nContext().active_object"
        )

        assert result["error"].startswith(
            "AttributeError: 'Context' object has no attribute 'active_object'"
        )
        assert "view_layer=win.view_layer" in result["error"]
        assert blender_runtime.add_context_hint(result["error"]) == result["error"]

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
        # Discarded, not cut: there is no truncation to report.
        assert result["output_truncated"] is False
        assert result["output_total_chars"] == 0
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

    def test_execute_script_returns_6000_chars_whole(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Issue #231: 6000 printed chars used to come back cut at 4096, silently."""
        fake_bpy = self._make_fake_bpy_for_script()
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.execute_script("print('x' * 6000)")

        assert result["output"] == "x" * 6000 + "\n"
        assert result["output_truncated"] is False
        assert result["output_total_chars"] == 6001

    def test_execute_script_reports_a_cut_at_the_default_cap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_bpy = self._make_fake_bpy_for_script()
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.execute_script("print('A' * 50000)")

        cap = blender_runtime.DEFAULT_SCRIPT_OUTPUT_CHARS
        assert cap == 32_000
        assert result["output"] == "A" * cap
        assert result["output_truncated"] is True
        assert result["output_total_chars"] == 50001
        assert "error" not in result

    def test_execute_script_cap_is_per_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_bpy = self._make_fake_bpy_for_script()
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        small = session.execute_script("print('{\"k\": 1}' * 10)", max_output_chars=25)
        large = session.execute_script(
            "print('B' * 90000)", max_output_chars=blender_runtime.MAX_SCRIPT_OUTPUT_CHARS
        )

        assert small["output"] == ('{"k": 1}' * 10)[:25]
        assert small["output_truncated"] is True and small["output_total_chars"] == 81
        assert large["output"] == "B" * 90000 + "\n"
        assert large["output_truncated"] is False

    def test_execute_script_caps_output_before_an_exception(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_bpy = self._make_fake_bpy_for_script()
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        result = session.execute_script("print('C' * 500)\nraise ValueError('late')", max_output_chars=100)

        assert result["error"] == "ValueError: late"
        assert result["output"] == "C" * 100
        assert result["output_truncated"] is True and result["output_total_chars"] == 501

    @pytest.mark.parametrize(
        "text",
        ["가", "é", "\x00", "\U0001F600"],
        ids=["hangul", "latin1", "control", "emoji"],
    )
    def test_execute_script_holds_escape_heavy_output_under_the_json_cap(
        self, monkeypatch: pytest.MonkeyPatch, text: str
    ) -> None:
        """The server counts escaped JSON bytes (6 or 12 here) and replaces a field over 400 KB.

        100000 such characters would be replaced whole, so the prefix stops at
        the JSON byte cap instead, and the cut is reported like any other.
        """
        fake_bpy = self._make_fake_bpy_for_script()
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        cap = blender_runtime.MAX_SCRIPT_OUTPUT_CHARS
        result = session.execute_script(f"print({text!r} * {cap})", max_output_chars=cap)

        output = result["output"]
        limit = blender_runtime.MAX_SCRIPT_OUTPUT_JSON_BYTES
        assert output == text * len(output) and 0 < len(output) < cap
        # The longest prefix that fits: one more character would not.
        assert len(json.dumps(output)) <= limit < len(json.dumps(output + text))
        assert result["output_truncated"] is True
        assert result["output_total_chars"] == cap + 1

    def test_execute_script_json_cap_never_cuts_printable_ascii(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Escaped quotes cost 2 bytes each; even 100000 of them stay under the JSON cap."""
        fake_bpy = self._make_fake_bpy_for_script()
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        cap = blender_runtime.MAX_SCRIPT_OUTPUT_CHARS
        result = session.execute_script(f"print('\"' * {cap})", max_output_chars=cap)

        assert result["output"] == '"' * cap
        assert result["output_truncated"] is True and result["output_total_chars"] == cap + 1

    def test_script_output_prefix_is_the_longest_that_fits(self) -> None:
        limit = blender_runtime.MAX_SCRIPT_OUTPUT_JSON_BYTES
        # 6-byte, 1-byte and 12-byte characters: 299972 JSON bytes, then emoji.
        head = "가" * 49_990 + "a" * 30
        assert len(json.dumps(head)) == 299_972
        output = head + "\U0001F600" * 5

        prefix = blender_runtime.script_output_prefix(output, 100_000)

        assert prefix == head + "\U0001F600" * 2
        assert len(json.dumps(prefix)) <= limit < len(json.dumps(prefix + "\U0001F600"))
        assert blender_runtime.script_output_prefix(output, 10) == "가" * 10

    @pytest.mark.parametrize("bad", [0, -1, 100_001, True, 12.5, "32000"])
    def test_execute_script_rejects_an_out_of_range_cap(
        self, monkeypatch: pytest.MonkeyPatch, bad: Any
    ) -> None:
        """The add-on validates too: it may serve a client that skipped the schema."""
        fake_bpy = self._make_fake_bpy_for_script()
        monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)

        session = blender_runtime.BlenderRuntimeSession()
        with pytest.raises(ValueError, match="max_output_chars"):
            session.execute_script("print('never runs')", max_output_chars=bad)


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
