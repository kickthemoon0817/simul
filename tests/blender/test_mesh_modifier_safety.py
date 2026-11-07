"""Mesh creation, modifier and mesh-info safety of the Blender runtime.

Each test pins a defect of the production-readiness audit on Blender 5.0.1:
vertices of other than three numbers scrambled the mesh (B3), degenerate faces
then Decimate froze Blender (B5), failed creations left orphans that broke
exports (M12), a collection of another scene hid the new object (M13),
modifier settings were ignored or left a half-made modifier behind (M5, M6),
and mesh info reported the base mesh instead of what is exported (M14).
"""

from __future__ import annotations

import math
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List

import pytest
from pydantic import ValidationError

from simul.adapters import blender_runtime
from simul.mcp.schemas.blender import (
    BlenderCleanMeshRequest,
    BlenderCreateMeshFromDataRequest,
)
from tests.blender.mesh_fakes import (
    CUBE_FACES,
    CUBE_VERTICES,
    FakeBpy,
    FakeMesh,
)

TRIANGLE = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
QUAD = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]


@pytest.fixture
def fake_bpy(monkeypatch: pytest.MonkeyPatch) -> FakeBpy:
    fake = FakeBpy()
    monkeypatch.setattr(blender_runtime, "bpy", fake)
    monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
    return fake


@pytest.fixture
def session(fake_bpy: FakeBpy) -> blender_runtime.BlenderRuntimeSession:
    return blender_runtime.BlenderRuntimeSession()


def _nothing_created(fake: FakeBpy) -> None:
    assert list(fake.data.objects) == []
    assert list(fake.data.meshes) == []


# ---------------------------------------------------------------------------
# create_mesh_from_data: vertex shape (B3)
# ---------------------------------------------------------------------------


class TestMeshVertices:
    @pytest.mark.parametrize(
        "vertices, message",
        [
            ([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]], "Vertex 0 must have exactly 3 coordinates"),
            ([[0, 0], [1, 0], [0, 1]], "Vertex 0 must have exactly 3 coordinates"),
            ([[0, 0, 0], [1, 0], [0, 1, 0, 5]], "Vertex 1 must have exactly 3 coordinates"),
            ([[0, 0, 0], [1, 0, 0], [0, 1, math.nan]], "Vertex 2 coordinates must be finite"),
            ([[math.inf, 0, 0], [1, 0, 0], [0, 1, 0]], "Vertex 0 coordinates must be finite"),
            ([[0, 0, 0], [-math.inf, 0, 0], [0, 1, 0]], "Vertex 1 coordinates must be finite"),
            ([[True, 0, 0], [1, 0, 0], [0, 1, 0]], "Vertex 0 coordinates must be numbers"),
            ([[0, 0, 0], "abc", [0, 1, 0]], "Vertex 1 must be"),
        ],
    )
    def test_a_vertex_of_other_than_three_finite_numbers_is_refused_before_creation(
        self,
        fake_bpy: FakeBpy,
        session: blender_runtime.BlenderRuntimeSession,
        vertices: List[Any],
        message: str,
    ) -> None:
        """from_pydata flattened [[0,0,0,0],...] into (0,0,0),(0,1,0),(0,0,0) and succeeded."""
        with pytest.raises(ValueError, match=message):
            session.create_mesh_from_data("v", vertices, faces=[[0, 1, 2]])
        _nothing_created(fake_bpy)

    def test_from_pydata_receives_float_triples(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession
    ) -> None:
        session.create_mesh_from_data("tri", TRIANGLE, faces=[[0, 1, 2]])

        vertices, edges, faces = fake_bpy.data.meshes[0].pydata
        assert vertices == [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        assert all(isinstance(c, float) for vertex in vertices for c in vertex)
        assert (edges, faces) == ([], [(0, 1, 2)])

    @pytest.mark.parametrize("location", [[1, 2], [1, 2, 3, 4], [1, 2, math.nan]])
    def test_location_must_be_three_finite_numbers(
        self,
        fake_bpy: FakeBpy,
        session: blender_runtime.BlenderRuntimeSession,
        location: List[float],
    ) -> None:
        with pytest.raises(ValueError, match="location"):
            session.create_mesh_from_data("tri", TRIANGLE, faces=[[0, 1, 2]], location=location)
        _nothing_created(fake_bpy)

    def test_the_request_model_refuses_the_same_vertices(self) -> None:
        good = {"name": "m", "vertices": TRIANGLE, "faces": [[0, 1, 2]]}
        BlenderCreateMeshFromDataRequest.model_validate(good)
        for vertices in (
            [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]],
            [[0, 0], [1, 0], [0, 1]],
            [[0, 0, 0], [1, 0, 0], [0, 1, math.nan]],
            [[0, 0, 0], [1, 0, 0], [0, 1, math.inf]],
        ):
            with pytest.raises(ValidationError):
                BlenderCreateMeshFromDataRequest.model_validate({**good, "vertices": vertices})
        for bad in (
            {"faces": [[0, 1]]},
            {"faces": [[0, 1, -1]]},
            {"faces": [[True, 1, 2]]},
            {"edges": [[0, 1, 2]]},
            {"edges": [[0]]},
            {"location": [0, 0, math.nan]},
        ):
            with pytest.raises(ValidationError):
                BlenderCreateMeshFromDataRequest.model_validate({**good, **bad})


# ---------------------------------------------------------------------------
# create_mesh_from_data: degenerate topology (B5)
# ---------------------------------------------------------------------------


class TestMeshTopology:
    @pytest.mark.parametrize(
        "faces, message",
        [
            ([[0, 1, 2, 1]], "Face 0 repeats vertex 1"),
            ([[0, 1, 2], [2, 3, 3]], "Face 1 repeats vertex 3"),
            ([[0, 1, 2], [0, 1, 2]], "Face 1 uses the same vertices as face 0"),
            ([[0, 1, 2], [2, 1, 0]], "Face 1 uses the same vertices as face 0"),
            ([[0, 1, 2, 3], [1, 2, 3, 0]], "Face 1 uses the same vertices as face 0"),
            ([[0, 1, 2, 3], [0, 2, 1, 3]], "Face 1 uses the same vertices as face 0"),
            ([[0, 1, 2], [0, 1, 7]], r"Face 1 index 7 out of range \[0, 4\)"),
            ([[0, 1, 2.0]], "Face 0 index 2.0 is not an integer"),
            ([[0, 1, True]], "Face 0 index True is not an integer"),
        ],
    )
    def test_invalid_faces_are_refused_before_creation(
        self,
        fake_bpy: FakeBpy,
        session: blender_runtime.BlenderRuntimeSession,
        faces: List[List[Any]],
        message: str,
    ) -> None:
        """The audit's freeze: these faces, then Decimate, pinned Blender's main thread."""
        with pytest.raises(ValueError, match=message):
            session.create_mesh_from_data("bad_mesh", QUAD, faces=faces)
        _nothing_created(fake_bpy)

    @pytest.mark.parametrize(
        "edges, message",
        [
            ([[1, 1]], "Edge 0 joins vertex 1 to itself"),
            ([[0, 1], [1, 0]], "Edge 1 repeats edge 0"),
            ([[0, 1], [2, 3], [0, 1]], "Edge 2 repeats edge 0"),
            ([[0, 9]], r"Edge 0 index 9 out of range \[0, 4\)"),
            ([[0, 1, 2]], "Edge 0 must have exactly 2 indices"),
        ],
    )
    def test_invalid_edges_are_refused_before_creation(
        self,
        fake_bpy: FakeBpy,
        session: blender_runtime.BlenderRuntimeSession,
        edges: List[List[int]],
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            session.create_mesh_from_data("wire", QUAD, edges=edges)
        _nothing_created(fake_bpy)

    def test_the_created_mesh_is_validated(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession
    ) -> None:
        result = session.create_mesh_from_data("box", CUBE_VERTICES, faces=CUBE_FACES)

        mesh = fake_bpy.data.meshes.get("box")
        assert mesh.validate_calls == [{"verbose": False, "clean_customdata": True}]
        assert result == {
            "object_name": "box",
            "mesh_name": "box",
            "vertex_count": 8,
            "edge_count": 12,
            "face_count": 6,
            "warnings": [],
        }

    def test_a_validation_change_is_reported(
        self,
        fake_bpy: FakeBpy,
        session: blender_runtime.BlenderRuntimeSession,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(FakeMesh, "validate", lambda self, verbose=False, clean_customdata=True: True)

        result = session.create_mesh_from_data("tri", TRIANGLE, faces=[[0, 1, 2]])

        assert result["warnings"] == [
            "Blender's mesh validation changed the geometry; the counts are what it kept"
        ]


# ---------------------------------------------------------------------------
# create_mesh_from_data: collections and orphans (M12, M13)
# ---------------------------------------------------------------------------


class TestMeshCollection:
    def test_a_missing_collection_leaves_no_orphan_and_the_retry_keeps_its_name(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession
    ) -> None:
        """The retry used to become probe.001 next to an orphan probe."""
        with pytest.raises(ValueError, match="Collection 'nope' not found"):
            session.create_mesh_from_data("probe", TRIANGLE, faces=[[0, 1, 2]], collection_name="nope")
        _nothing_created(fake_bpy)

        result = session.create_mesh_from_data("probe", TRIANGLE, faces=[[0, 1, 2]])
        assert result["object_name"] == "probe"
        assert fake_bpy.context.scene.collection.objects == [fake_bpy.data.objects.get("probe")]

    def test_a_collection_of_another_scene_is_refused(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession
    ) -> None:
        """It used to succeed, after which get_blender_object_info said Object not found."""
        fake_bpy.add_collection("elsewhere", in_scene=False)

        with pytest.raises(ValueError, match="Collection 'elsewhere' is not in scene 'Scene'"):
            session.create_mesh_from_data("probe", TRIANGLE, faces=[[0, 1, 2]], collection_name="elsewhere")
        _nothing_created(fake_bpy)

    def test_a_nested_collection_of_the_scene_takes_the_object(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession
    ) -> None:
        parent = fake_bpy.add_collection("props")
        child = fake_bpy.add_collection("bolts", in_scene=False)
        parent.children.append(child)

        session.create_mesh_from_data("bolt", TRIANGLE, faces=[[0, 1, 2]], collection_name="bolts")

        assert [obj.name for obj in child.objects] == ["bolt"]
        assert fake_bpy.context.scene.collection.objects == []

    def test_a_library_collection_is_refused(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession
    ) -> None:
        fake_bpy.add_collection("linked", library=SimpleNamespace(name="lib.blend"))

        with pytest.raises(ValueError, match="linked from a library"):
            session.create_mesh_from_data("probe", TRIANGLE, faces=[[0, 1, 2]], collection_name="linked")
        _nothing_created(fake_bpy)

    def test_a_failure_after_creation_removes_the_mesh_and_object(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession
    ) -> None:
        fake_bpy.add_collection("locked", fail_link="Collection 'locked' is not editable")

        with pytest.raises(RuntimeError, match="not editable"):
            session.create_mesh_from_data("probe", TRIANGLE, faces=[[0, 1, 2]], collection_name="locked")

        _nothing_created(fake_bpy)
        assert [obj.name for obj in fake_bpy.data.objects.removed] == ["probe"]
        assert [mesh.name for mesh in fake_bpy.data.meshes.removed] == ["probe"]

    def test_an_excluded_collection_is_reported(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession
    ) -> None:
        collection = fake_bpy.add_collection("hidden")
        fake_bpy.context.view_layer.layer_collection.find(collection).exclude = True

        result = session.create_mesh_from_data("probe", TRIANGLE, faces=[[0, 1, 2]], collection_name="hidden")

        assert result["warnings"] == [
            "Collection 'hidden' is excluded from view layer 'ViewLayer': Blender neither "
            "evaluates nor exports the object there"
        ]

    def test_a_taken_name_is_reported(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession
    ) -> None:
        session.create_mesh_from_data("box", TRIANGLE, faces=[[0, 1, 2]])

        result = session.create_mesh_from_data("box", TRIANGLE, faces=[[0, 1, 2]])

        assert result["object_name"] == "box.001"
        assert result["warnings"] == [
            "The name 'box' was taken or too long; Blender named the object 'box.001'"
        ]


# ---------------------------------------------------------------------------
# add_modifier (M5, M6, B5)
# ---------------------------------------------------------------------------


class TestAddModifier:
    @pytest.fixture
    def sphere(self, fake_bpy: FakeBpy) -> Any:
        return fake_bpy.add_mesh_object("sphere", CUBE_VERTICES, CUBE_FACES)

    def test_an_unknown_setting_is_refused_with_suggestions_and_leaves_no_modifier(
        self, session: blender_runtime.BlenderRuntimeSession, sphere: Any
    ) -> None:
        """{"ratioo": 0.1} used to succeed with the ratio left at 1.0."""
        with pytest.raises(ValueError, match="DECIMATE modifier has no setting 'ratioo'; did you mean 'ratio'"):
            session.add_modifier("sphere", "DECIMATE", params={"ratioo": 0.1})
        assert list(sphere.modifiers) == []
        assert [mod.type for mod in sphere.modifiers.removed] == ["DECIMATE"]

    def test_a_bad_enum_value_is_refused_and_the_modifier_removed(
        self, session: blender_runtime.BlenderRuntimeSession, sphere: Any
    ) -> None:
        """TRIANGULATE quad_method NOPE errored but left the modifier, triangulating the export."""
        with pytest.raises(ValueError, match="quad_method' must be one of BEAUTY, FIXED"):
            session.add_modifier("sphere", "TRIANGULATE", params={"quad_method": "NOPE"})
        assert list(sphere.modifiers) == []

    def test_a_failure_on_a_later_setting_removes_the_modifier(
        self, session: blender_runtime.BlenderRuntimeSession, sphere: Any
    ) -> None:
        with pytest.raises(ValueError, match="ratio' must lie between 0 and 1, got 7"):
            session.add_modifier("sphere", "DECIMATE", params={"iterations": 2, "ratio": 7.0})
        assert list(sphere.modifiers) == []

    @pytest.mark.parametrize(
        "params, message",
        [
            ({"ratio": "half"}, "ratio' takes a number, got 'half'"),
            ({"ratio": math.nan}, "ratio' must be finite"),
            ({"iterations": 2.5}, "iterations' takes a whole number"),
            ({"iterations": True}, "iterations' takes a number"),
            ({"face_count": 3}, "face_count' is read-only"),
            ({"name": "renamed"}, "through modifier_name, not params"),
            ({"type": "WELD"}, "through modifier_type, not params"),
            ({"rna_type": None}, "has no setting 'rna_type'"),
            ({"delimit": 3}, "delimit' takes a list of NORMAL"),
            ({"delimit": ["NORMAL", "BOGUS"]}, "delimit' must be one of"),
        ],
    )
    def test_refused_settings(
        self,
        session: blender_runtime.BlenderRuntimeSession,
        sphere: Any,
        params: Dict[str, Any],
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            session.add_modifier("sphere", "DECIMATE", params=params)
        assert list(sphere.modifiers) == []

    def test_settings_are_stored_and_echoed(
        self, session: blender_runtime.BlenderRuntimeSession, sphere: Any
    ) -> None:
        result = session.add_modifier(
            "sphere",
            "decimate",
            modifier_name="dec",
            params={"ratio": 0.25, "decimate_type": "dissolve", "delimit": ["uv", "SEAM"], "iterations": 3.0},
        )

        assert result == {
            "object_name": "sphere",
            "modifier_name": "dec",
            "modifier_type": "DECIMATE",
            "params": {"ratio": 0.25, "decimate_type": "DISSOLVE", "delimit": ["SEAM", "UV"], "iterations": 3},
        }
        modifier = sphere.modifiers.get("dec")
        assert (modifier.ratio, modifier.decimate_type, modifier.delimit) == (0.25, "DISSOLVE", {"SEAM", "UV"})
        assert isinstance(modifier.iterations, int)

    def test_a_vector_setting_takes_its_length(
        self, session: blender_runtime.BlenderRuntimeSession, sphere: Any
    ) -> None:
        result = session.add_modifier("sphere", "ARRAY", params={"relative_offset_displace": [0, 2, 0]})
        assert result["params"] == {"relative_offset_displace": [0, 2, 0]}

        with pytest.raises(ValueError, match="relative_offset_displace' takes 3 numbers"):
            session.add_modifier("sphere", "ARRAY", params={"relative_offset_displace": [0, 2]})
        assert [mod.type for mod in sphere.modifiers] == ["ARRAY"]

    def test_a_boolean_object_is_set_by_name(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, sphere: Any
    ) -> None:
        """{"object": "cutter"} used to fail with 'expected a Object type, not str'."""
        cutter = fake_bpy.add_mesh_object("cutter", CUBE_VERTICES, CUBE_FACES)

        result = session.add_modifier("sphere", "BOOLEAN", params={"object": "cutter", "operation": "UNION"})

        assert result["params"] == {"object": "cutter", "operation": "UNION"}
        assert sphere.modifiers.get("Boolean").object is cutter

    @pytest.mark.parametrize(
        "target, message",
        [
            ("missing", "object 'missing' not found"),
            ("cam", "Blender refused 'cam' for BOOLEAN modifier setting 'object'"),
            (3, "takes the name of a Object, got int"),
        ],
    )
    def test_a_boolean_object_blender_cannot_take_is_refused(
        self,
        fake_bpy: FakeBpy,
        session: blender_runtime.BlenderRuntimeSession,
        sphere: Any,
        target: Any,
        message: str,
    ) -> None:
        fake_bpy.add_object("cam", "CAMERA")

        with pytest.raises(ValueError, match=message):
            session.add_modifier("sphere", "BOOLEAN", params={"object": target})
        assert list(sphere.modifiers) == []

    def test_the_object_itself_is_refused_by_blender_and_the_modifier_removed(
        self, session: blender_runtime.BlenderRuntimeSession, sphere: Any
    ) -> None:
        with pytest.raises(TypeError, match="does not support assignment to itself"):
            session.add_modifier("sphere", "BOOLEAN", params={"object": "sphere"})
        assert list(sphere.modifiers) == []

    def test_a_collection_setting_is_set_by_name(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, sphere: Any
    ) -> None:
        cutters = fake_bpy.add_collection("cutters")

        result = session.add_modifier(
            "sphere", "BOOLEAN", params={"operand_type": "COLLECTION", "collection": "cutters"}
        )

        assert result["params"]["collection"] == "cutters"
        assert sphere.modifiers.get("Boolean").collection is cutters
        with pytest.raises(ValueError, match="Collection 'nope' not found"):
            session.add_modifier("sphere", "BOOLEAN", params={"collection": "nope"})

    @pytest.mark.parametrize("object_type", ["EMPTY", "LIGHT", "CAMERA"])
    def test_an_object_without_a_modifier_stack_is_refused(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, object_type: str
    ) -> None:
        """It used to fail with 'NoneType' object has no attribute 'name'."""
        fake_bpy.add_object("thing", object_type)

        with pytest.raises(ValueError, match=f"Object 'thing' is a {object_type} object, which has no modifier stack"):
            session.add_modifier("thing", "DECIMATE")

    def test_a_type_blender_does_not_add_to_the_object_is_refused(
        self, session: blender_runtime.BlenderRuntimeSession, sphere: Any
    ) -> None:
        with pytest.raises(ValueError, match="Blender cannot add a GREASE_PENCIL_TINT modifier to MESH object"):
            session.add_modifier("sphere", "GREASE_PENCIL_TINT")

    def test_an_unknown_type_is_refused_with_suggestions(
        self, session: blender_runtime.BlenderRuntimeSession, sphere: Any
    ) -> None:
        with pytest.raises(ValueError, match="Unknown modifier type 'DECIMTE'; did you mean 'DECIMATE'"):
            session.add_modifier("sphere", "DECIMTE")

    def test_decimate_on_invalid_geometry_is_refused(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession
    ) -> None:
        """B5: faces repeating a vertex plus duplicates froze BM_mesh_decimate_collapse."""
        fake_bpy.add_mesh_object("bad_mesh", QUAD, [[0, 1, 2, 1], [0, 1, 2], [0, 1, 2], [2, 3, 3]],
                                 invalid_faces=[0, 2, 3])

        with pytest.raises(ValueError, match="invalid geometry .* run clean_blender_mesh on it first"):
            session.add_modifier("bad_mesh", "DECIMATE", params={"ratio": 0.5})

        obj = fake_bpy.data.objects.get("bad_mesh")
        assert list(obj.modifiers) == []
        # The check validated a throwaway copy, removed again; the mesh is unchanged.
        assert [mesh.name for mesh in fake_bpy.data.meshes] == ["bad_mesh"]
        assert len(obj.data.polygons) == 4
        probe = fake_bpy.data.meshes.removed[0]
        assert probe.validate_calls == [{"verbose": False, "clean_customdata": False}]

    def test_decimate_on_valid_geometry_is_added(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, sphere: Any
    ) -> None:
        session.add_modifier("sphere", "DECIMATE", params={"ratio": 0.5})

        assert [mod.ratio for mod in sphere.modifiers] == [0.5]
        assert [mesh.name for mesh in fake_bpy.data.meshes] == ["sphere"]
        assert len(fake_bpy.data.meshes.removed) == 1


# ---------------------------------------------------------------------------
# get_mesh_info (M14)
# ---------------------------------------------------------------------------


class TestMeshInfo:
    def test_base_and_evaluated_counts(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession
    ) -> None:
        """After Bevel/Decimate/Triangulate/Weld it reported 6 faces while the USD had 94 triangles."""
        obj = fake_bpy.add_mesh_object("palm", CUBE_VERTICES, CUBE_FACES)
        session.add_modifier("palm", "TRIANGULATE")
        session.add_modifier("palm", "WELD", params={"show_viewport": False})
        triangles = [[0, 1, 2]] * 94
        obj.evaluated_mesh = FakeMesh(fake_bpy.data.meshes, "evaluated", [[0, 0, 0]] * 60, triangles)

        result = session.get_mesh_info("palm")

        assert result == {
            "object_name": "palm",
            "vertex_count": 8,
            "edge_count": 12,
            "face_count": 6,
            "triangle_count": 12,
            "evaluated_vertex_count": 60,
            "evaluated_edge_count": 3,
            "evaluated_face_count": 94,
            "evaluated_triangle_count": 94,
            "evaluated_modifiers": ["Triangulate"],
            "edit_mode": False,
            "uv_layer_names": [],
            "has_shape_keys": False,
            "warnings": ["The evaluated counts are the viewport's, and these modifiers are set differently "
                         "for render, which USD export evaluates: Weld"],
        }
        assert obj.cleared_evaluated == 1

    def test_edit_mode_edits_are_written_back_before_counting(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession
    ) -> None:
        """The counts were stale while the mesh was in Edit Mode."""
        obj = fake_bpy.add_mesh_object("box", CUBE_VERTICES, CUBE_FACES)
        obj.mode = "EDIT"
        obj.edit_mesh = (CUBE_VERTICES, [], CUBE_FACES[:4])

        result = session.get_mesh_info("box")

        assert obj.editmode_updates == 1
        assert (result["edit_mode"], result["face_count"], result["evaluated_face_count"]) == (True, 4, 4)

    def test_an_object_blender_does_not_evaluate_reports_no_evaluated_counts(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession
    ) -> None:
        obj = fake_bpy.add_mesh_object("hidden", CUBE_VERTICES, CUBE_FACES)
        obj.evaluated_mesh = None

        result = session.get_mesh_info("hidden")

        assert result["evaluated_face_count"] is None
        assert result["evaluated_triangle_count"] is None
        assert result["warnings"] == [
            "Blender does not evaluate 'hidden' (it is not in the view layer), so the counts "
            "after its modifiers are unknown"
        ]

    def test_render_levels_that_differ_are_reported(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession
    ) -> None:
        fake_bpy.add_mesh_object("smooth", CUBE_VERTICES, CUBE_FACES)
        session.add_modifier("smooth", "SUBSURF", params={"levels": 1, "render_levels": 2})
        session.add_modifier("smooth", "SUBSURF", modifier_name="same", params={"levels": 2, "render_levels": 2})

        warnings = session.get_mesh_info("smooth")["warnings"]

        assert warnings == ["The evaluated counts are the viewport's, and these modifiers are set differently "
                            "for render, which USD export evaluates: Subsurf"]


# ---------------------------------------------------------------------------
# apply_modifier / remove_modifier
# ---------------------------------------------------------------------------


class TestApplyAndRemoveModifier:
    @pytest.fixture
    def box(self, fake_bpy: FakeBpy) -> Any:
        return fake_bpy.add_mesh_object("box", CUBE_VERTICES, CUBE_FACES)

    def test_apply_bakes_the_modifier_into_the_mesh(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, box: Any
    ) -> None:
        session.add_modifier("box", "DECIMATE", params={"ratio": 0.5})

        result = session.apply_modifier("box", "Decimate")

        assert fake_bpy.applied == [{"object": "box", "modifier": "Decimate", "single_user": False}]
        assert fake_bpy.context.object is None  # the override ended
        assert result == {
            "object_name": "box",
            "modifier_name": "Decimate",
            "modifier_type": "DECIMATE",
            "vertex_count": 8,
            "edge_count": 10,
            "face_count": 3,
            "triangle_count": 6,
            "remaining_modifiers": [],
            "made_single_user": False,
            "warnings": [],
        }

    def test_apply_below_other_modifiers_warns(
        self, session: blender_runtime.BlenderRuntimeSession, box: Any
    ) -> None:
        session.add_modifier("box", "TRIANGULATE")
        session.add_modifier("box", "DECIMATE", params={"ratio": 0.5})

        result = session.apply_modifier("box", "Decimate")

        assert result["remaining_modifiers"] == ["Triangulate"]
        assert result["warnings"] == [
            "'Decimate' was not first in the stack: Blender applied it to the base mesh "
            "without the modifiers above it (Triangulate)"
        ]

    def test_apply_on_shared_mesh_data_copies_it_first(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, box: Any
    ) -> None:
        """Blender refuses 'Modifiers cannot be applied to multi-user data' otherwise."""
        twin = fake_bpy.data.objects.new("twin", box.data)
        session.add_modifier("box", "DECIMATE", params={"ratio": 0.5})

        result = session.apply_modifier("box", "Decimate")

        assert result["made_single_user"] is True
        assert fake_bpy.applied[0]["single_user"] is True
        assert box.data is not twin.data
        assert (len(box.data.polygons), len(twin.data.polygons)) == (3, 6)

    def test_an_apply_blender_cancels_is_an_error(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, box: Any
    ) -> None:
        session.add_modifier("box", "WELD")
        fake_bpy.apply_result = {"CANCELLED"}

        with pytest.raises(RuntimeError, match="Blender did not apply modifier 'Weld' to 'box' \\(CANCELLED\\)"):
            session.apply_modifier("box", "Weld")

    def test_apply_refuses_a_modifier_disabled_in_the_viewport(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, box: Any
    ) -> None:
        """Blender 5.0.1 answered FINISHED and dropped such a modifier unapplied."""
        session.add_modifier("box", "WELD", params={"show_viewport": False})

        with pytest.raises(ValueError, match="disabled in the viewport, so Blender would remove it"):
            session.apply_modifier("box", "Weld")
        assert fake_bpy.applied == []
        assert [mod.name for mod in box.modifiers] == ["Weld"]

    def test_apply_refuses_edit_mode(self, session: blender_runtime.BlenderRuntimeSession, box: Any) -> None:
        session.add_modifier("box", "WELD")
        box.mode = "EDIT"

        with pytest.raises(ValueError, match="Object 'box' is in Edit Mode; switch it to Object Mode"):
            session.apply_modifier("box", "Weld")

    def test_apply_names_the_modifiers_when_one_is_missing(
        self, session: blender_runtime.BlenderRuntimeSession, box: Any
    ) -> None:
        with pytest.raises(ValueError, match="Object 'box' has no modifier 'Weld'; it has none"):
            session.apply_modifier("box", "Weld")
        session.add_modifier("box", "TRIANGULATE")
        with pytest.raises(ValueError, match="has no modifier 'Weld'; its modifiers are: Triangulate"):
            session.apply_modifier("box", "Weld")

    def test_apply_decimate_on_invalid_geometry_is_refused(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, box: Any
    ) -> None:
        session.add_modifier("box", "DECIMATE", params={"ratio": 0.5})
        box.data.invalid_faces = [1]

        with pytest.raises(ValueError, match="invalid geometry"):
            session.apply_modifier("box", "Decimate")
        assert fake_bpy.applied == []

    def test_apply_on_a_curve_is_refused(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession
    ) -> None:
        fake_bpy.add_object("path", "CURVE")

        with pytest.raises(ValueError, match="only mesh modifiers can be applied"):
            session.apply_modifier("path", "Array")

    def test_remove_drops_the_modifier_unapplied(
        self, session: blender_runtime.BlenderRuntimeSession, box: Any
    ) -> None:
        session.add_modifier("box", "TRIANGULATE")
        session.add_modifier("box", "DECIMATE")

        result = session.remove_modifier("box", "Triangulate")

        assert result == {
            "object_name": "box",
            "modifier_name": "Triangulate",
            "modifier_type": "TRIANGULATE",
            "remaining_modifiers": ["Decimate"],
        }
        assert len(box.data.polygons) == 6


# ---------------------------------------------------------------------------
# clean_mesh
# ---------------------------------------------------------------------------


class _FakeBMesh:
    def __init__(self, log: List[Any], merge: int) -> None:
        self.log = log
        self.merge = merge
        self.verts: List[Any] = []
        self.edges: List[Any] = []
        self.faces: List[Any] = []

    def from_mesh(self, mesh: Any) -> None:
        self.log.append("from_mesh")
        self.verts = list(mesh.vertices)
        self.edges = list(mesh.edges)
        self.faces = [p.vertices for p in mesh.polygons]

    def to_mesh(self, mesh: Any) -> None:
        self.log.append("to_mesh")
        mesh._set_geometry(self.verts, [], [f for f in self.faces if max(f) < len(self.verts)])

    def free(self) -> None:
        self.log.append("free")


@pytest.fixture
def bmesh_log(monkeypatch: pytest.MonkeyPatch) -> List[Any]:
    """A ``bmesh`` stand-in recording the operators run on it; remove_doubles merges 2 vertices."""
    log: List[Any] = []
    module = ModuleType("bmesh")

    def remove_doubles(bm: _FakeBMesh, verts: List[Any], dist: float) -> None:
        log.append(("remove_doubles", len(verts), dist))
        bm.verts = bm.verts[: len(bm.verts) - bm.merge]

    module.new = lambda: _FakeBMesh(log, merge=2)  # type: ignore[attr-defined]
    module.ops = SimpleNamespace(  # type: ignore[attr-defined]
        remove_doubles=remove_doubles,
        dissolve_degenerate=lambda bm, dist, edges: log.append(("dissolve_degenerate", len(edges), dist)),
        recalc_face_normals=lambda bm, faces: log.append(("recalc_face_normals", len(faces))),
    )
    monkeypatch.setitem(sys.modules, "bmesh", module)
    return log


class TestCleanMesh:
    def test_validation_removes_invalid_topology(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, bmesh_log: List[Any]
    ) -> None:
        fake_bpy.add_mesh_object("bad_mesh", QUAD, [[0, 1, 2, 1], [0, 1, 2], [0, 1, 2], [2, 3, 3]],
                                 invalid_faces=[0, 2, 3])

        result = session.clean_mesh("bad_mesh")

        assert result == {
            "object_name": "bad_mesh",
            "mesh_name": "bad_mesh",
            "before": {"vertex_count": 4, "edge_count": 5, "face_count": 4, "triangle_count": 5},
            "after": {"vertex_count": 4, "edge_count": 3, "face_count": 1, "triangle_count": 1},
            "validation_fixed": True,
            "vertices_merged": 0,
            "operations": ["validate"],
            "mesh_users": 1,
            "warnings": [],
        }
        assert bmesh_log == []  # nothing optional asked, so no BMesh round trip

    def test_the_optional_steps_run_in_order_on_the_validated_mesh(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, bmesh_log: List[Any]
    ) -> None:
        fake_bpy.add_mesh_object("box", CUBE_VERTICES, CUBE_FACES)

        result = session.clean_mesh(
            "box", merge_distance=0.001, dissolve_degenerate_distance=0.0001, recalculate_normals=True
        )

        assert bmesh_log == [
            "from_mesh",
            ("remove_doubles", 8, 0.001),
            ("dissolve_degenerate", 12, 0.0001),
            ("recalc_face_normals", 6),
            "to_mesh",
            "free",
        ]
        assert result["operations"] == ["validate", "merge_by_distance", "dissolve_degenerate", "recalculate_normals"]
        assert result["vertices_merged"] == 2
        assert result["after"]["vertex_count"] == 6

    @pytest.mark.parametrize("field", ["merge_distance", "dissolve_degenerate_distance"])
    @pytest.mark.parametrize("value", [0, -1.0, math.nan, math.inf, True])
    def test_distances_must_be_finite_and_positive(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, field: str, value: Any
    ) -> None:
        obj = fake_bpy.add_mesh_object("box", CUBE_VERTICES, CUBE_FACES)

        with pytest.raises(ValueError, match=f"{field} must be a finite distance above 0"):
            session.clean_mesh("box", **{field: value})
        assert obj.data.validate_calls == []
        if value is not True:  # the request model reads true as 1.0, like any JSON number field
            with pytest.raises(ValidationError):
                BlenderCleanMeshRequest.model_validate({"object_name": "box", field: value})

    def test_edit_mode_is_refused(self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession) -> None:
        fake_bpy.add_mesh_object("box", CUBE_VERTICES, CUBE_FACES).mode = "EDIT"

        with pytest.raises(ValueError, match="is in Edit Mode; switch it to Object Mode to clean its mesh"):
            session.clean_mesh("box")

    def test_shared_mesh_data_is_reported(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession
    ) -> None:
        box = fake_bpy.add_mesh_object("box", CUBE_VERTICES, CUBE_FACES)
        fake_bpy.data.objects.new("twin", box.data)

        result = session.clean_mesh("box")

        assert result["mesh_users"] == 2
        assert result["warnings"] == ["Mesh 'box' is shared by 2 users; all of them changed"]
