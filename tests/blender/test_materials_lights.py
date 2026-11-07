"""Material, texture, light and UV tools of the Blender runtime.

Each test pins a defect or gap of the production-readiness audit on Blender
5.0.1: a second material never reached the faces (B4), the same name made
``.001`` copies (M9), a refused call left orphan materials (M10), alpha was
ignored, textured materials were misreported and the node list was unbounded
(M15), light settings of another type, out of range or negative were accepted
silently, and there was no way to texture or unwrap a mesh.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

import pytest
from pydantic import ValidationError

from simul.adapters import blender_runtime
from simul.mcp.schemas import blender as blender_schemas
from simul.mcp.schemas.blender import (
    BlenderAssignMaterialRequest,
    BlenderSetLightParamsRequest,
    BlenderUVUnwrapRequest,
)
from simul.utils.paths import PathPolicy
from tests.blender.mesh_fakes import (
    CUBE_FACES,
    CUBE_VERTICES,
    FakeBpy,
    FakeLightData,
    FakeNodeTree,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 16
_NO_SANDBOX = PathPolicy(enabled=False, allowed_paths=[], project_root=Path.cwd())


@pytest.fixture
def fake_bpy(monkeypatch: pytest.MonkeyPatch) -> FakeBpy:
    fake = FakeBpy()
    monkeypatch.setattr(blender_runtime, "bpy", fake)
    monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
    return fake


@pytest.fixture
def session(fake_bpy: FakeBpy, tmp_path: Path) -> blender_runtime.BlenderRuntimeSession:
    policy = PathPolicy(enabled=True, allowed_paths=[str(tmp_path)], project_root=tmp_path)
    return blender_runtime.BlenderRuntimeSession(path_policy=policy)


@pytest.fixture
def crate(fake_bpy: FakeBpy) -> Any:
    obj = fake_bpy.add_mesh_object("crate", CUBE_VERTICES, CUBE_FACES)
    obj.data.uv_layers.new("UVMap")
    return obj


def _faces(obj: Any) -> List[int]:
    return [polygon.material_index for polygon in obj.data.polygons]


def _slots(obj: Any) -> List[Any]:
    return [slot.material.name if slot.material else None for slot in obj.material_slots]


def _image(tmp_path: Path, name: str, content: bytes = PNG) -> str:
    path = tmp_path / name
    path.write_bytes(content)
    return str(path)


# ---------------------------------------------------------------------------
# assign_material: slots and faces (B4)
# ---------------------------------------------------------------------------


class TestAssignMaterialSlots:
    def test_a_second_material_replaces_the_first(
        self, session: blender_runtime.BlenderRuntimeSession, crate: Any
    ) -> None:
        """B4: steel used to land in a second slot that no face used; the USD bound oak."""
        session.assign_material("crate", "opaque_wood_oak", base_color=[0.5, 0.3, 0.1, 1.0])

        result = session.assign_material("crate", "opaque_metal_steel_plate", metallic=1.0)

        assert _slots(crate) == ["opaque_metal_steel_plate"]
        assert _faces(crate) == [0] * 6
        assert (result["slot_index"], result["faces_assigned"], result["created"]) == (0, 6, True)
        assert result["material_slots"] == ["opaque_metal_steel_plate"]

    def test_the_default_moves_every_face_of_a_multi_material_object_to_slot_0(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, crate: Any
    ) -> None:
        red, blue = fake_bpy.data.materials.new("red"), fake_bpy.data.materials.new("blue")
        crate.data.materials.extend([red, blue])
        for polygon in crate.data.polygons[:2]:
            polygon.material_index = 1

        session.assign_material("crate", "green")

        assert _slots(crate) == ["green", "blue"]
        assert _faces(crate) == [0] * 6

    def test_slot_index_sets_one_slot_and_leaves_the_faces(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, crate: Any
    ) -> None:
        crate.data.materials.extend([fake_bpy.data.materials.new("red"), fake_bpy.data.materials.new("blue")])
        crate.data.polygons[0].material_index = 1

        replaced = session.assign_material("crate", "green", slot_index=1)
        appended = session.assign_material("crate", "white", slot_index=2)

        assert _slots(crate) == ["red", "green", "white"]
        assert _faces(crate) == [1, 0, 0, 0, 0, 0]
        assert (replaced["slot_index"], replaced["faces_assigned"]) == (1, None)
        assert appended["slot_index"] == 2

    @pytest.mark.parametrize("slot_index", [-1, 3, True, 1.5])
    def test_a_slot_index_outside_the_slots_is_refused(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, crate: Any, slot_index: Any
    ) -> None:
        crate.data.materials.extend([fake_bpy.data.materials.new("a"), fake_bpy.data.materials.new("b")])

        with pytest.raises(ValueError, match="slot_index must lie between 0 and 2"):
            session.assign_material("crate", "green", slot_index=slot_index)
        assert fake_bpy.data.materials.get("green") is None

    def test_face_indices_assign_those_faces_through_one_slot(
        self, session: blender_runtime.BlenderRuntimeSession, crate: Any
    ) -> None:
        """Per-face assignment was a missing capability."""
        session.assign_material("crate", "wood")

        first = session.assign_material("crate", "steel", face_indices=[0, 1])
        second = session.assign_material("crate", "steel", face_indices=[5])

        assert _slots(crate) == ["wood", "steel"]
        assert _faces(crate) == [1, 1, 0, 0, 0, 1]
        assert (first["slot_index"], first["faces_assigned"]) == (1, 2)
        assert (second["slot_index"], second["created"]) == (1, False)

    @pytest.mark.parametrize(
        "faces, message",
        [
            ([6], r"face_indices index 6 out of range \[0, 6\)"),
            ([0, 0], "face_indices lists face 0 twice"),
            ([], "face_indices must name at least one face"),
            ([True], "face_indices index True is not an integer"),
        ],
    )
    def test_bad_face_indices_are_refused(
        self,
        fake_bpy: FakeBpy,
        session: blender_runtime.BlenderRuntimeSession,
        crate: Any,
        faces: List[Any],
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            session.assign_material("crate", "steel", face_indices=faces)
        assert list(fake_bpy.data.materials) == []
        assert _slots(crate) == []

    def test_face_indices_need_a_mesh(self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession) -> None:
        fake_bpy.add_object("path", "CURVE", SimpleNamespace(name="path", materials=[], users=1))

        with pytest.raises(ValueError, match="face_indices applies to mesh objects"):
            session.assign_material("path", "steel", face_indices=[0])

    def test_writing_faces_in_edit_mode_is_refused_but_a_slot_is_not(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, crate: Any
    ) -> None:
        """Edit Mode would overwrite the face indices when it is left."""
        crate.mode = "EDIT"

        with pytest.raises(ValueError, match="is in Edit Mode; switch it to Object Mode"):
            session.assign_material("crate", "steel")
        assert list(fake_bpy.data.materials) == []
        result = session.assign_material("crate", "steel", slot_index=0)
        assert (result["faces_assigned"], _slots(crate)) == (None, ["steel"])

    def test_object_link_keeps_the_material_off_linked_duplicates(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, crate: Any
    ) -> None:
        """The material went on the mesh data, so every linked duplicate changed."""
        twin = fake_bpy.data.objects.new("twin", crate.data)
        session.assign_material("crate", "wood")

        result = session.assign_material("twin", "steel", link="OBJECT")

        assert (_slots(crate), _slots(twin)) == (["wood"], ["steel"])
        assert result["link"] == "OBJECT"
        assert result["warnings"] == []
        shared = session.assign_material("crate", "oak")
        assert shared["warnings"] == [
            "Mesh data 'crate' is shared by 2 users; a DATA slot changes on all of them "
            "(link='OBJECT' changes only this object)"
        ]

    def test_link_must_be_data_or_object(self, session: blender_runtime.BlenderRuntimeSession, crate: Any) -> None:
        with pytest.raises(ValueError, match="link must be DATA or OBJECT"):
            session.assign_material("crate", "steel", link="SCENE")


# ---------------------------------------------------------------------------
# assign_material: reuse and rollback (M9, M10)
# ---------------------------------------------------------------------------


class TestAssignMaterialReuse:
    def test_the_same_name_reuses_the_material(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, crate: Any
    ) -> None:
        """M9: the second call made opaque_rubber_black.001, and the USD held two materials."""
        first = session.assign_material("crate", "opaque_rubber_black", base_color=[0.02, 0.02, 0.02, 1.0])
        second = session.assign_material("crate", "opaque_rubber_black", base_color=[0.02, 0.02, 0.02, 1.0])

        assert fake_bpy.data.materials.keys() == ["opaque_rubber_black"]
        assert (first["created"], second["created"]) == (True, False)
        assert _slots(crate) == ["opaque_rubber_black"]

    def test_an_existing_material_is_assigned_as_it_is(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, crate: Any
    ) -> None:
        """Assigning an existing material was the top missing capability."""
        imported = fake_bpy.data.materials.new("imported")
        imported.principled.inputs["Base Color"].default_value = (1.0, 0.0, 0.0, 1.0)
        imported.principled.inputs["Roughness"].default_value = 0.9

        result = session.assign_material("crate", "imported")

        assert imported.principled.inputs["Base Color"].default_value == (1.0, 0.0, 0.0, 1.0)
        assert imported.principled.inputs["Roughness"].default_value == 0.9
        assert result["created"] is False
        assert result["principled_params"]["Roughness"] == 0.9

    def test_reusing_with_new_shading_updates_it_and_warns_when_shared(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, crate: Any
    ) -> None:
        other = fake_bpy.add_mesh_object("other", CUBE_VERTICES, CUBE_FACES)
        session.assign_material("other", "paint", base_color=[1, 0, 0])

        result = session.assign_material("crate", "paint", roughness=0.2)

        paint = fake_bpy.data.materials.get("paint")
        assert paint.principled.inputs["Roughness"].default_value == 0.2
        assert paint.principled.inputs["Base Color"].default_value == (1.0, 0.0, 0.0, 1.0)
        assert result["warnings"] == [
            "Material 'paint' has 2 users; the changed shading shows wherever it is used"
        ]
        assert _slots(other) == ["paint"]

    def test_reuse_existing_false_creates_another(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, crate: Any
    ) -> None:
        session.assign_material("crate", "paint")

        result = session.assign_material("crate", "paint", reuse_existing=False)

        assert (result["material_name"], result["created"]) == ("paint.001", True)
        assert fake_bpy.data.materials.keys() == ["paint", "paint.001"]

    def test_shading_a_material_without_a_principled_bsdf_is_refused(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, crate: Any
    ) -> None:
        glow = fake_bpy.data.materials.new("glow")
        glow.node_tree = FakeNodeTree()
        emission = glow.node_tree.nodes.new("ShaderNodeEmission")
        output = glow.node_tree.nodes.new("ShaderNodeOutputMaterial")
        glow.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])

        with pytest.raises(ValueError, match="Material 'glow' has no Principled BSDF"):
            session.assign_material("crate", "glow", base_color=[1, 1, 1])
        assert _slots(crate) == []
        assert session.assign_material("crate", "glow")["principled_params"] is None

    def test_shading_a_library_material_is_refused(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, crate: Any
    ) -> None:
        fake_bpy.data.materials.new("linked").library = SimpleNamespace(name="lib.blend")

        with pytest.raises(ValueError, match="linked from a library"):
            session.assign_material("crate", "linked", metallic=1.0)
        assert session.assign_material("crate", "linked")["created"] is False

    @pytest.mark.parametrize("object_type", ["LIGHT", "CAMERA", "EMPTY"])
    def test_an_object_without_material_slots_is_refused_before_anything_is_made(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, object_type: str
    ) -> None:
        """M10: targeting a light left a lightmat material behind."""
        fake_bpy.add_object("thing", object_type, None if object_type == "EMPTY" else SimpleNamespace(name="d"))

        with pytest.raises(ValueError, match=f"is a {object_type} object, which takes no materials"):
            session.assign_material("thing", "lightmat")
        assert list(fake_bpy.data.materials) == []

    def test_a_failure_after_the_material_was_made_removes_it_again(
        self,
        fake_bpy: FakeBpy,
        session: blender_runtime.BlenderRuntimeSession,
        crate: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session.assign_material("crate", "wood")
        calls: List[Any] = []
        real = blender_runtime.BlenderRuntimeSession._set_face_materials

        def flaky(mesh: Any, values: List[int]) -> None:
            calls.append(list(values))
            if len(calls) == 1:
                raise RuntimeError("face write failed")
            real(mesh, values)

        monkeypatch.setattr(blender_runtime.BlenderRuntimeSession, "_set_face_materials", staticmethod(flaky))

        with pytest.raises(RuntimeError, match="face write failed"):
            session.assign_material("crate", "steel", face_indices=[0])

        assert fake_bpy.data.materials.keys() == ["wood"]
        assert [material.name for material in fake_bpy.data.materials.removed] == ["steel"]
        assert _slots(crate) == ["wood"]
        assert _faces(crate) == [0] * 6


# ---------------------------------------------------------------------------
# assign_material: colour, alpha and the deprecated use_nodes (M10, minor)
# ---------------------------------------------------------------------------


class TestAssignMaterialShading:
    def test_rgb_is_accepted(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, crate: Any
    ) -> None:
        """M10: RGB failed with 'sequences of dimension 0 should contain 4 items, not 3' and left rgbmat."""
        result = session.assign_material("crate", "rgbmat", base_color=[1.0, 0.0, 0.0])

        bsdf = fake_bpy.data.materials.get("rgbmat").principled
        assert bsdf.inputs["Base Color"].default_value == (1.0, 0.0, 0.0, 1.0)
        assert bsdf.inputs["Alpha"].default_value == 1.0
        assert result["principled_params"]["Base Color"] == [1.0, 0.0, 0.0, 1.0]

    def test_alpha_sets_the_principled_alpha_and_blended_rendering(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, crate: Any
    ) -> None:
        """The alpha channel was ignored, so USD opacity stayed 1.0."""
        result = session.assign_material("crate", "glass_like", base_color=[1, 0, 0, 0.3])

        material = fake_bpy.data.materials.get("glass_like")
        assert material.principled.inputs["Alpha"].default_value == 0.3
        assert material.principled.inputs["Base Color"].default_value == (1.0, 0.0, 0.0, 1.0)
        assert material.diffuse_color == (1.0, 0.0, 0.0, 0.3)
        assert result["surface_render_method"] == "BLENDED"

        opaque = session.assign_material("crate", "glass_like", base_color=[1, 0, 0, 1])
        assert opaque["surface_render_method"] == "DITHERED"

    def test_before_blender_4_2_alpha_sets_the_blend_method(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        fake = FakeBpy(version=(4, 1, 0))
        monkeypatch.setattr(blender_runtime, "bpy", fake)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        session = blender_runtime.BlenderRuntimeSession(path_policy=_NO_SANDBOX)
        fake.add_mesh_object("crate", CUBE_VERTICES, CUBE_FACES)
        material = fake.data.materials.new("old")
        del material.surface_render_method
        material.node_tree = FakeNodeTree.default()
        material._use_nodes = True

        result = session.assign_material("crate", "old", base_color=[1, 1, 1, 0.5])

        assert material.blend_method == "BLEND"
        assert result["surface_render_method"] == "BLEND"

    @pytest.mark.parametrize(
        "base_color, message",
        [
            ([1, 0], r"base_color must be \[r, g, b\] or \[r, g, b, a\]"),
            ([1, 0, 0, 1, 1], r"base_color must be \[r, g, b\] or \[r, g, b, a\]"),
            ([2, 0, 0], "base_color must lie between 0 and 1"),
            ([-0.1, 0, 0], "base_color must lie between 0 and 1"),
            ([math.nan, 0, 0], "base_color must be finite"),
            (["red", 0, 0], "base_color must be a number"),
        ],
    )
    def test_a_bad_color_is_refused_before_anything_is_made(
        self,
        fake_bpy: FakeBpy,
        session: blender_runtime.BlenderRuntimeSession,
        crate: Any,
        base_color: List[Any],
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            session.assign_material("crate", "rgbmat", base_color=base_color)
        assert list(fake_bpy.data.materials) == []
        assert _slots(crate) == []

    @pytest.mark.parametrize("field", ["metallic", "roughness"])
    def test_factors_are_checked(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, crate: Any, field: str
    ) -> None:
        with pytest.raises(ValueError, match=f"{field} must lie between 0 and 1"):
            session.assign_material("crate", "m", **{field: 1.5})
        assert list(fake_bpy.data.materials) == []

    def test_blender_5_never_touches_the_deprecated_use_nodes(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, crate: Any
    ) -> None:
        """Material.use_nodes warns on Blender 5.0 that it goes away in 6.0."""
        session.assign_material("crate", "m", base_color=[1, 1, 1])
        session.get_material_info("m")

        assert fake_bpy.data.materials.get("m").use_nodes_access == 0

    def test_before_blender_5_a_new_material_gets_its_node_tree(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeBpy(version=(4, 2, 0))
        monkeypatch.setattr(blender_runtime, "bpy", fake)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        session = blender_runtime.BlenderRuntimeSession(path_policy=_NO_SANDBOX)
        fake.add_mesh_object("crate", CUBE_VERTICES, CUBE_FACES)

        session.assign_material("crate", "m", metallic=1.0)

        material = fake.data.materials.get("m")
        assert material.node_tree is not None
        assert material.principled.inputs["Metallic"].default_value == 1.0

    def test_the_request_model_accepts_rgb_and_rgba_only(self) -> None:
        for good in ([1, 0, 0], [1, 0, 0, 0.5]):
            assert BlenderAssignMaterialRequest(object_name="c", base_color=good).base_color == good
        for bad in ([1, 0], [1, 0, 0, 1, 1], [2, 0, 0], [-1, 0, 0]):
            with pytest.raises(ValidationError):
                BlenderAssignMaterialRequest(object_name="c", base_color=bad)
        assert BlenderAssignMaterialRequest(object_name="c").base_color is None
        with pytest.raises(ValidationError):
            BlenderAssignMaterialRequest(object_name="c", link="SCENE")
        with pytest.raises(ValidationError):
            BlenderAssignMaterialRequest(object_name="c", face_indices=[])


# ---------------------------------------------------------------------------
# assign_material: textures
# ---------------------------------------------------------------------------


class TestAssignMaterialTextures:
    def test_textures_drive_their_inputs_with_the_right_colour_space(
        self,
        fake_bpy: FakeBpy,
        session: blender_runtime.BlenderRuntimeSession,
        crate: Any,
        tmp_path: Path,
    ) -> None:
        paths = {name: _image(tmp_path, f"{name}.png") for name in ("albedo", "metal", "rough", "normal")}

        result = session.assign_material(
            "crate",
            "textured",
            base_color_texture=paths["albedo"],
            metallic_texture=paths["metal"],
            roughness_texture=paths["rough"],
            normal_texture=paths["normal"],
        )

        bsdf = fake_bpy.data.materials.get("textured").principled
        source = {
            name: bsdf.inputs[name].links[0].from_node
            for name in ("Base Color", "Metallic", "Roughness", "Normal")
        }
        assert source["Base Color"].image.filepath == paths["albedo"]
        assert source["Base Color"].image.colorspace_settings.name == "sRGB"
        assert source["Roughness"].image.colorspace_settings.name == "Non-Color"
        assert source["Metallic"].image.colorspace_settings.name == "Non-Color"
        assert source["Normal"].type == "NORMAL_MAP"
        normal_image = source["Normal"].inputs["Color"].links[0].from_node.image
        assert (normal_image.filepath, normal_image.colorspace_settings.name) == (paths["normal"], "Non-Color")
        assert result["textures"] == {
            "base_color": paths["albedo"],
            "metallic": paths["metal"],
            "roughness": paths["rough"],
            "normal": paths["normal"],
        }
        assert result["principled_params"]["Base Color"] is None
        assert result["warnings"] == []

    def test_a_new_texture_replaces_the_old_one_and_its_nodes(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, crate: Any, tmp_path: Path
    ) -> None:
        session.assign_material("crate", "m", base_color_texture=_image(tmp_path, "a.png"),
                                normal_texture=_image(tmp_path, "n1.png"))
        session.assign_material("crate", "m", base_color_texture=_image(tmp_path, "b.png"),
                                normal_texture=_image(tmp_path, "n2.png"))

        tree = fake_bpy.data.materials.get("m").node_tree
        textures = [node for node in tree.nodes if node.type == "TEX_IMAGE"]
        assert sorted(node.image.name for node in textures) == ["b.png", "n2.png"]
        assert len([node for node in tree.nodes if node.type == "NORMAL_MAP"]) == 1

    def test_a_texture_outside_the_sandbox_is_refused(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, crate: Any, tmp_path: Path
    ) -> None:
        outside = tmp_path.parent / "outside_texture.png"
        outside.write_bytes(PNG)

        with pytest.raises(PermissionError):
            session.assign_material("crate", "m", base_color_texture=str(outside))
        assert list(fake_bpy.data.materials) == []

    def test_a_missing_texture_file_is_refused(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, crate: Any, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="roughness_texture '.*nope.png' is not a file"):
            session.assign_material("crate", "m", roughness_texture=str(tmp_path / "nope.png"))
        assert list(fake_bpy.data.materials) == []

    def test_a_file_that_is_no_image_is_refused_and_unloaded(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, crate: Any, tmp_path: Path
    ) -> None:
        """bpy.data.images.load accepts such a file; it then reads as 0 x 0 pixels."""
        good = _image(tmp_path, "good.png")
        bad = _image(tmp_path, "notes.png", b"not an image")

        with pytest.raises(ValueError, match="normal_texture '.*notes.png' is not an image Blender can read"):
            session.assign_material("crate", "m", base_color_texture=good, normal_texture=bad)

        assert list(fake_bpy.data.materials) == []
        assert list(fake_bpy.data.images) == []
        assert sorted(image.name for image in fake_bpy.data.images.removed) == ["good.png", "notes.png"]

    def test_a_mesh_without_uvs_is_warned_about(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, tmp_path: Path
    ) -> None:
        fake_bpy.add_mesh_object("plate", CUBE_VERTICES, CUBE_FACES)

        result = session.assign_material("plate", "m", base_color_texture=_image(tmp_path, "a.png"))

        assert result["warnings"] == [
            "Mesh 'plate' has no UV map, which image textures need; run uv_unwrap_blender_mesh on the object"
        ]


# ---------------------------------------------------------------------------
# get_material_info (M15)
# ---------------------------------------------------------------------------


class TestMaterialInfo:
    def test_a_textured_input_reports_its_source_and_image(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, crate: Any, tmp_path: Path
    ) -> None:
        """It used to report the unused default [0.8, 0.8, 0.8, 1] and no link."""
        albedo, normal = _image(tmp_path, "albedo.png"), _image(tmp_path, "normal.png")
        session.assign_material("crate", "opaque_textured", base_color_texture=albedo, normal_texture=normal,
                                roughness=0.4)

        info = session.get_material_info("opaque_textured")

        assert info["principled_params"]["Base Color"] is None
        assert info["principled_params"]["Normal"] is None
        assert info["principled_params"]["Roughness"] == 0.4
        assert info["linked_inputs"] == {
            "Base Color": {
                "from_node": "Image Texture",
                "from_node_type": "ShaderNodeTexImage",
                "from_socket": "Color",
                "image_name": "albedo.png",
                "image_path": albedo,
                "image_colorspace": "sRGB",
                "image_packed": False,
            },
            "Normal": {
                "from_node": "Normal Map",
                "from_node_type": "ShaderNodeNormalMap",
                "from_socket": "Normal",
                "image_name": "normal.png",
                "image_path": normal,
                "image_colorspace": "Non-Color",
                "image_packed": False,
            },
        }
        assert info["surface_render_method"] == "DITHERED"
        assert info["users"] == 1

    def test_the_node_list_is_bounded_and_starts_at_the_output(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession
    ) -> None:
        """The 'bounded traversal' returned all 302 nodes."""
        material = fake_bpy.data.materials.new("bignodes")
        for _ in range(300):
            material.node_tree.nodes.new("ShaderNodeMath")

        info = session.get_material_info("bignodes")
        few = session.get_material_info("bignodes", max_nodes=2)

        assert (len(info["nodes"]), info["node_count"]) == (50, 302)
        assert [node["name"] for node in few["nodes"]] == ["Material Output", "Principled BSDF"]

    def test_a_node_feeding_two_inputs_is_listed_once(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, crate: Any, tmp_path: Path
    ) -> None:
        session.assign_material("crate", "shared", base_color_texture=_image(tmp_path, "a.png"))
        material = fake_bpy.data.materials.get("shared")
        texture = material.principled.inputs["Base Color"].links[0].from_node
        material.node_tree.links.new(texture.outputs["Color"], material.principled.inputs["Roughness"])

        info = session.get_material_info("shared")

        names = [node["name"] for node in info["nodes"]]
        assert names == ["Material Output", "Principled BSDF", "Image Texture"]
        assert info["node_count"] == 3

    @pytest.mark.parametrize("max_nodes", [0, -1, True, 2.5])
    def test_max_nodes_must_be_positive(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, max_nodes: Any
    ) -> None:
        fake_bpy.data.materials.new("m")
        with pytest.raises(ValueError, match="max_nodes must be a whole number of at least 1"):
            session.get_material_info("m", max_nodes=max_nodes)


# ---------------------------------------------------------------------------
# set_light_params
# ---------------------------------------------------------------------------


class TestLightParams:
    def test_spot_settings_on_a_point_light_are_refused(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession
    ) -> None:
        """They used to be ignored with success."""
        lamp = fake_bpy.add_light("pt", "POINT")

        with pytest.raises(ValueError, match="spot_size applies to SPOT lights, and 'pt' is a POINT light"):
            session.set_light_params("pt", energy=500.0, spot_size=0.5)
        assert lamp.store.values["energy"] == 10.0

    @pytest.mark.parametrize(
        "light_type, hint",
        [("SUN", "a sun's softness is its angle"), ("AREA", "an area light's softness is its size")],
    )
    def test_shadow_soft_size_names_the_setting_that_applies(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, light_type: str, hint: str
    ) -> None:
        fake_bpy.add_light("lamp", light_type)

        with pytest.raises(ValueError, match=hint):
            session.set_light_params("lamp", shadow_soft_size=0.5)

    @pytest.mark.parametrize(
        "settings, message",
        [
            ({"spot_size": 10.0}, "spot_size must be between 0.0174533 and 3.14159"),
            ({"spot_blend": 5.0}, "spot_blend must be between 0 and 1"),
            ({"energy": -100.0}, "energy must be at least 0"),
            ({"shadow_soft_size": -1.0}, "shadow_soft_size must be at least 0"),
            ({"energy": math.inf}, "energy must be finite"),
            ({"color": [2.0, -1.0, 0.5]}, "color channels must be at least 0"),
            ({"color": [1, 1]}, r"color must be \[r, g, b\]"),
            ({"use_shadow": 1}, "use_shadow must be true or false"),
        ],
    )
    def test_values_outside_their_range_are_refused_and_nothing_changes(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, settings: Any, message: str
    ) -> None:
        """They used to be clamped silently (spot_size 10 became pi) and negative energy was accepted."""
        lamp = fake_bpy.add_light("spot", "SPOT")
        before = dict(lamp.store.values)

        with pytest.raises(ValueError, match=message):
            session.set_light_params("spot", energy=50.0, **settings) if "energy" not in settings else \
                session.set_light_params("spot", **settings)
        assert lamp.store.values == before

    def test_the_reply_echoes_every_setting_of_the_type(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession
    ) -> None:
        fake_bpy.add_light("spot", "SPOT")

        result = session.set_light_params("spot", energy=800.0, color=[1, 0.9, 0.8], spot_size=1.2, spot_blend=0.3)

        assert result == {
            "light_name": "spot",
            "light_type": "SPOT",
            "energy": 800.0,
            "color": [1, 0.9, 0.8],
            "use_shadow": True,
            "shadow_soft_size": 0.25,
            "spot_size": 1.2,
            "spot_blend": 0.3,
            "warnings": [],
        }

    def test_light_type_changes_first_and_its_settings_then_apply(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession
    ) -> None:
        """Light type, area size and sun angle were missing capabilities."""
        lamp = fake_bpy.add_light("lamp", "POINT")

        area = session.set_light_params("lamp", light_type="area", shape="RECTANGLE", size=2.0, size_y=0.5)
        sun = session.set_light_params("lamp", light_type="SUN", angle=0.1, energy=3.0)

        assert (area["light_type"], area["shape"], area["size"], area["size_y"]) == ("AREA", "RECTANGLE", 2.0, 0.5)
        assert "shadow_soft_size" not in area
        assert (sun["light_type"], sun["angle"], sun["energy"]) == ("SUN", 0.1, 3.0)
        assert lamp.store.values["type"] == "SUN"

    def test_size_y_needs_a_rectangle_or_ellipse(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession
    ) -> None:
        fake_bpy.add_light("area", "AREA")

        with pytest.raises(ValueError, match="size_y applies to RECTANGLE and ELLIPSE area lights"):
            session.set_light_params("area", size_y=1.0)
        assert session.set_light_params("area", shape="ELLIPSE", size_y=1.0)["size_y"] == 1.0

    def test_unknown_type_and_shape_are_refused(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession
    ) -> None:
        fake_bpy.add_light("lamp", "AREA")

        with pytest.raises(ValueError, match="light_type must be one of POINT, SPOT, SUN, AREA"):
            session.set_light_params("lamp", light_type="LASER")
        with pytest.raises(ValueError, match="shape must be one of SQUARE, RECTANGLE, DISK, ELLIPSE"):
            session.set_light_params("lamp", shape="STAR")

    def test_a_shared_light_is_warned_about(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession
    ) -> None:
        first = fake_bpy.add_light("a", "POINT")
        fake_bpy.add_light("b", "POINT", store=first.store)

        result = session.set_light_params("a", energy=5.0)

        assert result["warnings"] == ["Light data 'a' is shared by 2 objects; all of them changed"]

    def test_the_request_model_refuses_the_same_values(self) -> None:
        for bad in (
            {"energy": -1},
            {"spot_size": 10},
            {"spot_blend": 1.5},
            {"shadow_soft_size": -1},
            {"color": [-1, 0, 0]},
            {"light_type": "LASER"},
            {"shape": "STAR"},
            {"angle": 4},
        ):
            with pytest.raises(ValidationError):
                BlenderSetLightParamsRequest(light_name="l", **bad)
        stored_pi = 3.1415927410125732
        assert BlenderSetLightParamsRequest(light_name="l", spot_size=stored_pi).spot_size == stored_pi

    def test_the_schema_limits_mirror_the_runtime(self) -> None:
        assert blender_schemas.LIGHT_ANGLE_MIN == blender_runtime._LIGHT_LIMITS["spot_size"][0]
        assert blender_schemas.LIGHT_ANGLE_MAX == blender_runtime._LIGHT_LIMITS["spot_size"][1]
        assert blender_schemas.LIGHT_ANGLE_MAX == blender_runtime._LIGHT_LIMITS["angle"][1]


# ---------------------------------------------------------------------------
# uv_unwrap_mesh
# ---------------------------------------------------------------------------


class TestUVUnwrap:
    @pytest.fixture
    def plate(self, fake_bpy: FakeBpy) -> Any:
        """A mesh without UVs, while another object is active and selected and two faces are selected."""
        other = fake_bpy.add_mesh_object("other", CUBE_VERTICES, CUBE_FACES)
        other.select_set(True)
        fake_bpy.context.view_layer.objects.active = other
        plate = fake_bpy.add_mesh_object("plate", CUBE_VERTICES, CUBE_FACES)
        plate.data.polygons[0].select = plate.data.polygons[3].select = True
        return plate

    def test_smart_project_writes_a_new_uv_map_and_restores_the_selection(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, plate: Any
    ) -> None:
        result = session.uv_unwrap_mesh("plate", island_margin=0.02)

        assert fake_bpy.edit_ops == [
            ("mode_set", {"mode": "EDIT", "object": "plate"}),
            ("select_all", {"action": "SELECT"}),
            ("smart_project", {"angle_limit": math.radians(66.0), "island_margin": 0.02}),
            ("mode_set", {"mode": "OBJECT", "object": "plate"}),
        ]
        assert plate.data.uv_layers.active.written_by[0] == "smart_project"
        assert result == {
            "object_name": "plate",
            "uv_map": "UVMap",
            "created_uv_map": True,
            "method": "SMART_PROJECT",
            "uv_layer_names": ["UVMap"],
            "face_count": 6,
            "warnings": [],
        }
        assert [p.select for p in plate.data.polygons] == [True, False, False, True, False, False]
        other = fake_bpy.data.objects.get("other")
        assert fake_bpy.context.view_layer.objects.active is other
        assert (other.select_get(), plate.select_get(), plate.mode) == (True, False, "OBJECT")

    def test_cube_project_into_a_named_uv_map(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, plate: Any
    ) -> None:
        plate.data.uv_layers.new("UVMap")

        result = session.uv_unwrap_mesh("plate", method="cube_project", uv_map_name="Lightmap", cube_size=2.0)

        assert ("cube_project", {"cube_size": 2.0}) in fake_bpy.edit_ops
        assert (result["uv_map"], result["created_uv_map"], result["uv_layer_names"]) == (
            "Lightmap", True, ["UVMap", "Lightmap"]
        )
        assert plate.data.uv_layers.active.name == "Lightmap"

    def test_the_active_uv_map_is_rewritten_by_default(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, plate: Any
    ) -> None:
        plate.data.uv_layers.new("Existing")

        result = session.uv_unwrap_mesh("plate")

        assert (result["uv_map"], result["created_uv_map"]) == ("Existing", False)

    def test_a_failed_unwrap_removes_the_new_map_and_restores_everything(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, plate: Any
    ) -> None:
        fake_bpy.unwrap_error = "Error: no faces"

        with pytest.raises(RuntimeError, match="no faces"):
            session.uv_unwrap_mesh("plate")

        assert list(plate.data.uv_layers) == []
        assert plate.mode == "OBJECT"
        assert fake_bpy.context.view_layer.objects.active is fake_bpy.data.objects.get("other")
        assert [p.select for p in plate.data.polygons] == [True, False, False, True, False, False]

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"method": "LIGHTMAP"}, "method must be one of SMART_PROJECT, CUBE_PROJECT"),
            ({"angle_limit": 0}, "angle_limit must be above 0 and at most 1.5708"),
            ({"angle_limit": 2.0}, "angle_limit must be above 0 and at most 1.5708"),
            ({"island_margin": -0.1}, "island_margin must be at least 0 and at most 1"),
            ({"cube_size": 0}, "cube_size must be above 0"),
            ({"cube_size": math.nan}, "cube_size must be finite"),
        ],
    )
    def test_bad_options_are_refused(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, plate: Any, kwargs: Any, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            session.uv_unwrap_mesh("plate", **kwargs)
        assert fake_bpy.edit_ops == []

    def test_edit_mode_hidden_objects_and_non_meshes_are_refused(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, plate: Any
    ) -> None:
        plate.mode = "EDIT"
        with pytest.raises(ValueError, match="is in Edit Mode"):
            session.uv_unwrap_mesh("plate")
        plate.mode = "OBJECT"
        fake_bpy.context.mode = "EDIT_MESH"
        with pytest.raises(ValueError, match="Blender is in Edit Mesh mode"):
            session.uv_unwrap_mesh("plate")
        fake_bpy.context.mode = "OBJECT"
        plate.hidden = True
        with pytest.raises(ValueError, match="is hidden in the view layer"):
            session.uv_unwrap_mesh("plate")
        fake_bpy.add_light("lamp", "POINT")
        with pytest.raises(ValueError, match="is not a MESH"):
            session.uv_unwrap_mesh("lamp")
        assert fake_bpy.edit_ops == []

    def test_the_request_model(self) -> None:
        assert BlenderUVUnwrapRequest(object_name="p").method == "SMART_PROJECT"
        for bad in ({"method": "LIGHTMAP"}, {"angle_limit": 0}, {"island_margin": 2}, {"cube_size": 0},
                    {"uv_map_name": ""}):
            with pytest.raises(ValidationError):
                BlenderUVUnwrapRequest(object_name="p", **bad)


def test_light_data_fake_retypes_only_when_fetched_again() -> None:
    """The fake mirrors Blender: a struct fetched before a type change keeps the old type's settings."""
    store = FakeLightData("l", "POINT")
    view = SimpleNamespace(data=None)
    from tests.blender.mesh_fakes import FakeLightView

    view = FakeLightView(store)
    view.type = "SPOT"
    with pytest.raises(AttributeError):
        view.spot_size = 1.0
    assert FakeLightView(store).spot_size == 0.785398
