"""Shared data, shared images and typed refusals of the Blender material and UV tools.

Each test pins a finding of the #328 review on Blender 5.0.1: ``link="OBJECT"``
rewrote the faces of linked duplicates, an image already in the file had its
colour space rewritten (and a refused call left it so), a float base colour
was forced to sRGB, the face check was quadratic, Edit Mode through a duplicate
and library data were not caught, the unwrap skipped hidden faces, and a muted
link or a Reroute misled the material report.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

import pytest
from fastmcp import Client

from simul.adapters import blender_runtime
from simul.adapters.blender_runtime import NotFoundError, ValidationError
from simul.config import Settings
from simul.mcp import backends as backends_module
from simul.mcp import server as server_module
from simul.utils.paths import PathPolicy
from tests.blender.mesh_fakes import CUBE_FACES, CUBE_VERTICES, FakeBpy, FakeMesh
from tests.fakes import BlenderSessionAdapter, StrictBlenderSession

PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 16


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


def _faces(obj: Any) -> List[int]:
    return [polygon.material_index for polygon in obj.data.polygons]


def _slots(obj: Any) -> List[Any]:
    return [slot.material.name if slot.material else None for slot in obj.material_slots]


def _image(tmp_path: Path, name: str, content: bytes = PNG) -> str:
    path = tmp_path / name
    path.write_bytes(content)
    return str(path)


# ---------------------------------------------------------------------------
# link="OBJECT" on shared mesh data
# ---------------------------------------------------------------------------


class TestSharedMeshData:
    @pytest.fixture
    def twin(self, fake_bpy: FakeBpy) -> Any:
        """Twin and TwinDup share one cube whose faces alternate between slots t0 and t1."""
        twin = fake_bpy.add_mesh_object("Twin", CUBE_VERTICES, CUBE_FACES)
        fake_bpy.data.objects.new("TwinDup", twin.data)
        twin.data.materials.extend([fake_bpy.data.materials.new("t0"), fake_bpy.data.materials.new("t1")])
        for polygon in twin.data.polygons:
            polygon.material_index = polygon.index % 2
        return twin

    def test_an_object_link_that_would_rewrite_shared_faces_is_refused(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, twin: Any
    ) -> None:
        """TwinDup's faces used to become [0, 0, 0, 0, 0, 0], with no warning."""
        dup = fake_bpy.data.objects.get("TwinDup")

        with pytest.raises(ValidationError, match="shared with 'TwinDup'.*single_user=true") as refused:
            session.assign_material("Twin", "only_twin", link="OBJECT")

        assert refused.value.details == {"code": "shared_data", "shared_with": ["TwinDup"]}
        assert _faces(dup) == [0, 1, 0, 1, 0, 1]
        assert (_slots(twin), _slots(dup)) == (["t0", "t1"], ["t0", "t1"])
        assert fake_bpy.data.materials.get("only_twin") is None

    def test_an_object_link_that_would_add_a_slot_to_shared_data_is_refused(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, twin: Any
    ) -> None:
        """With face_indices the duplicate used to get an empty slot its faces 0 and 1 used."""
        dup = fake_bpy.data.objects.get("TwinDup")

        with pytest.raises(ValidationError, match="assigning faces changes the shared data"):
            session.assign_material("Twin", "only_twin", link="OBJECT", face_indices=[0, 1])

        assert (len(dup.material_slots), _faces(dup)) == (2, [0, 1, 0, 1, 0, 1])

    def test_single_user_gives_the_object_its_own_mesh_first(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, twin: Any
    ) -> None:
        dup = fake_bpy.data.objects.get("TwinDup")
        shared = twin.data

        result = session.assign_material("Twin", "only_twin", link="OBJECT", single_user=True)

        assert twin.data is not shared and dup.data is shared
        assert (_faces(twin), _faces(dup)) == ([0] * 6, [0, 1, 0, 1, 0, 1])
        assert (_slots(twin), _slots(dup)) == (["only_twin", "t1"], ["t0", "t1"])
        assert result["warnings"] == [
            f"'Twin' now has its own copy '{twin.data.name}' of mesh data 'Twin', which 'TwinDup' keep"
        ]

    def test_a_failure_after_the_copy_puts_the_shared_mesh_back(
        self,
        fake_bpy: FakeBpy,
        session: blender_runtime.BlenderRuntimeSession,
        twin: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        shared = twin.data

        def fail(mesh: Any, values: List[int]) -> None:
            raise RuntimeError("face write failed")

        monkeypatch.setattr(blender_runtime.BlenderRuntimeSession, "_set_face_materials", staticmethod(fail))

        with pytest.raises(RuntimeError, match="face write failed"):
            session.assign_material("Twin", "only_twin", link="OBJECT", single_user=True)

        assert twin.data is shared
        assert [mesh.name for mesh in fake_bpy.data.meshes] == ["Twin"]
        assert fake_bpy.data.materials.get("only_twin") is None
        assert _slots(twin) == ["t0", "t1"]

    def test_an_object_link_into_an_existing_slot_changes_only_this_object(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, twin: Any
    ) -> None:
        dup = fake_bpy.data.objects.get("TwinDup")

        result = session.assign_material("Twin", "only_twin", link="OBJECT", slot_index=1)

        assert (_slots(twin), _slots(dup)) == (["t0", "only_twin"], ["t0", "t1"])
        assert _faces(dup) == [0, 1, 0, 1, 0, 1]
        assert result["warnings"] == []

    def test_a_data_link_names_the_objects_it_changes(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, twin: Any
    ) -> None:
        result = session.assign_material("Twin", "for_both")

        assert _faces(fake_bpy.data.objects.get("TwinDup")) == [0] * 6
        assert result["warnings"][0] == (
            "Mesh data 'Twin' is shared with 'TwinDup'; its slots and face assignments change on "
            "all of them (single_user=true changes only this object)"
        )

    def test_edit_mode_through_a_duplicate_is_refused(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, twin: Any
    ) -> None:
        """The face write was lost when TwinDup held the mesh in Edit Mode."""
        twin.data.is_editmode = True

        with pytest.raises(ValidationError, match="in Edit Mode through another object") as refused:
            session.assign_material("Twin", "only_twin", single_user=True)
        assert refused.value.details["code"] == "wrong_mode"
        with pytest.raises(ValidationError, match="through another object"):
            session.uv_unwrap_mesh("Twin")
        with pytest.raises(ValidationError, match="through another object"):
            session.clean_mesh("Twin")
        assert fake_bpy.data.materials.get("only_twin") is None


class TestLibraryData:
    def test_a_linked_object_is_refused_up_front(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession
    ) -> None:
        """It failed with a raw AttributeError ... read-only and left an empty slot."""
        crate = fake_bpy.add_mesh_object("crate", CUBE_VERTICES, CUBE_FACES)
        crate.library = SimpleNamespace(name="props.blend")

        with pytest.raises(ValidationError, match="linked from library 'props.blend'") as refused:
            session.assign_material("crate", "steel")

        assert refused.value.details["code"] == "linked_from_library"
        assert (_slots(crate), list(fake_bpy.data.materials)) == ([], [])

    def test_linked_mesh_data_needs_a_local_copy(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession
    ) -> None:
        crate = fake_bpy.add_mesh_object("crate", CUBE_VERTICES, CUBE_FACES)
        linked = crate.data
        linked.library = SimpleNamespace(name="props.blend")

        with pytest.raises(ValidationError, match="linked from a library"):
            session.assign_material("crate", "steel")
        with pytest.raises(ValidationError, match="linked from library"):
            session.uv_unwrap_mesh("crate")

        session.assign_material("crate", "steel", single_user=True)
        assert crate.data is not linked and _slots(crate) == ["steel"]


# ---------------------------------------------------------------------------
# Images already in the file
# ---------------------------------------------------------------------------


class TestTextureColourSpaces:
    @pytest.fixture
    def crate(self, fake_bpy: FakeBpy) -> Any:
        obj = fake_bpy.add_mesh_object("crate", CUBE_VERTICES, CUBE_FACES)
        obj.data.uv_layers.new("UVMap")
        return obj

    def test_an_image_another_material_reads_keeps_its_colour_space(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, crate: Any, tmp_path: Path
    ) -> None:
        """Using the other material's sRGB albedo as roughness made it Non-Color there too."""
        albedo = _image(tmp_path, "albedo.png")
        other = fake_bpy.add_mesh_object("other", CUBE_VERTICES, CUBE_FACES)
        other.data.uv_layers.new("UVMap")
        session.assign_material("other", "painted", base_color_texture=albedo)
        shared = fake_bpy.data.images.get("albedo.png")

        session.assign_material("crate", "rough", roughness_texture=albedo)

        assert shared.colorspace_settings.name == "sRGB"
        roughness = fake_bpy.data.materials.get("rough").principled.inputs["Roughness"].links[0].from_node
        assert roughness.image is not shared
        assert (roughness.image.name, roughness.image.colorspace_settings.name) == ("albedo.png.001", "Non-Color")

    def test_a_refused_call_leaves_an_image_in_the_file_as_it_was(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, crate: Any, tmp_path: Path
    ) -> None:
        """metal.png, loaded Non-Color, stayed sRGB after a refused call used it as a base colour."""
        metal = _image(tmp_path, "metal.png")
        loaded = fake_bpy.data.images.load(metal)
        loaded.colorspace_settings.name = "Non-Color"
        bogus = _image(tmp_path, "notes.png", b"not an image")

        with pytest.raises(ValidationError, match="not an image Blender can read"):
            session.assign_material("crate", "m", base_color_texture=metal, normal_texture=bogus)

        assert loaded.colorspace_settings.name == "Non-Color"
        assert list(fake_bpy.data.images) == [loaded]
        assert list(fake_bpy.data.materials) == []

    def test_one_file_read_as_colour_and_as_data_gets_two_datablocks(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, crate: Any, tmp_path: Path
    ) -> None:
        """Base colour and roughness from one file left both Non-Color."""
        path = _image(tmp_path, "packed.png")

        session.assign_material("crate", "m", base_color_texture=path, roughness_texture=path)

        bsdf = fake_bpy.data.materials.get("m").principled
        color = bsdf.inputs["Base Color"].links[0].from_node.image
        rough = bsdf.inputs["Roughness"].links[0].from_node.image
        assert color is not rough
        assert (color.colorspace_settings.name, rough.colorspace_settings.name) == ("sRGB", "Non-Color")

    def test_a_float_base_colour_stays_linear(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, crate: Any, tmp_path: Path
    ) -> None:
        """An EXR base colour was forced to sRGB: 0.5 read back as 0.214 and the USD said sRGB."""
        path = _image(tmp_path, "albedo.exr")

        result = session.assign_material("crate", "m", base_color_texture=path)

        image = fake_bpy.data.materials.get("m").principled.inputs["Base Color"].links[0].from_node.image
        assert image.colorspace_settings.name == "Linear Rec.709"
        info = session.get_material_info("m")
        assert info["linked_inputs"]["Base Color"]["image_colorspace"] == "Linear Rec.709"
        assert result["textures"] == {"base_color": path}

    def test_a_missing_texture_file_is_a_typed_not_found(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, crate: Any, tmp_path: Path
    ) -> None:
        missing = str(tmp_path / "gone.png")

        with pytest.raises(NotFoundError) as refused:
            session.assign_material("crate", "m", metallic_texture=missing)

        assert refused.value.details == {
            "code": "texture_not_found",
            "field": "metallic_texture",
            "path": missing,
        }
        assert type(refused.value).__name__ == "NotFoundError"


# ---------------------------------------------------------------------------
# Material report: muted links and reroutes
# ---------------------------------------------------------------------------


class TestMaterialReportLinks:
    @pytest.fixture
    def textured(self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, tmp_path: Path) -> Any:
        fake_bpy.add_mesh_object("crate", CUBE_VERTICES, CUBE_FACES).data.uv_layers.new("UVMap")
        session.assign_material("crate", "m", base_color=[0.2, 0.4, 0.6], base_color_texture=_image(tmp_path, "a.png"))
        return fake_bpy.data.materials.get("m")

    def test_a_muted_link_drives_nothing(
        self, session: blender_runtime.BlenderRuntimeSession, textured: Any
    ) -> None:
        """A muted link was reported as driving the input."""
        textured.principled.inputs["Base Color"].links[0].is_muted = True

        info = session.get_material_info("m")

        assert info["principled_params"]["Base Color"] == [0.2, 0.4, 0.6, 1.0]
        assert "Base Color" not in info["linked_inputs"]

    def test_an_image_behind_a_reroute_is_found(
        self, session: blender_runtime.BlenderRuntimeSession, textured: Any
    ) -> None:
        """An image behind a Reroute node was not found."""
        tree = textured.node_tree
        link = textured.principled.inputs["Base Color"].links[0]
        texture = link.from_node
        reroute = tree.nodes.new("NodeReroute")
        tree.links.new(texture.outputs["Color"], reroute.inputs["Input"])
        tree.links.new(reroute.outputs["Output"], textured.principled.inputs["Base Color"])

        entry = session.get_material_info("m")["linked_inputs"]["Base Color"]

        assert (entry["from_node"], entry["from_socket"], entry["image_name"]) == ("Image Texture", "Color", "a.png")


# ---------------------------------------------------------------------------
# Smaller findings
# ---------------------------------------------------------------------------


class TestSmallerFindings:
    def test_the_face_index_check_is_linear(self) -> None:
        """202,500 indices took 95 s on Blender's main thread with the list lookup."""
        mesh = SimpleNamespace(polygons=range(300_000))
        started = time.perf_counter()

        checked = blender_runtime.BlenderRuntimeSession._checked_face_indices(mesh, list(range(300_000)))

        assert len(checked) == 300_000
        assert time.perf_counter() - started < 5.0

    def test_a_replaced_material_left_without_users_is_reported(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession
    ) -> None:
        fake_bpy.add_mesh_object("crate", CUBE_VERTICES, CUBE_FACES)
        session.assign_material("crate", "oak")

        result = session.assign_material("crate", "steel")

        assert result["warnings"] == [
            "Material 'oak' has no users now; Blender discards it when the file is saved and "
            "reopened (set its use_fake_user to keep it)"
        ]

    def test_hidden_faces_are_unwrapped_and_stay_hidden(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession
    ) -> None:
        """select_all skipped faces hidden in Edit Mode, yet the reply said face_count 6."""
        plate = fake_bpy.add_mesh_object("plate", CUBE_VERTICES, CUBE_FACES)
        for polygon in plate.data.polygons[:2]:
            polygon.hide = True
        plate.data.polygons[4].select = True

        result = session.uv_unwrap_mesh("plate")

        assert plate.data.uv_layers.get("UVMap").faces_unwrapped == 6
        assert [p.hide for p in plate.data.polygons] == [True, True, False, False, False, False]
        assert [p.select for p in plate.data.polygons] == [False, False, False, False, True, False]
        assert result["face_count"] == 6

    @pytest.mark.parametrize(
        "kwargs, code",
        [
            ({"base_color": [2, 0, 0]}, "out_of_range"),
            ({"base_color": [1, 0]}, "invalid_value"),
            ({"slot_index": 5}, "out_of_range"),
            ({"face_indices": [0, 0]}, "invalid_value"),
            ({"link": "SCENE"}, "invalid_value"),
        ],
    )
    def test_refusals_carry_a_code(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, kwargs: Any, code: str
    ) -> None:
        fake_bpy.add_mesh_object("crate", CUBE_VERTICES, CUBE_FACES)

        with pytest.raises(ValidationError) as refused:
            session.assign_material("crate", "m", **kwargs)
        assert refused.value.details["code"] == code

    def test_light_and_material_refusals_carry_a_code(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession
    ) -> None:
        fake_bpy.add_light("lamp", "POINT")
        with pytest.raises(ValidationError) as refused:
            session.set_light_params("lamp", spot_size=0.5)
        assert refused.value.details == {
            "code": "setting_not_for_light_type",
            "field": "spot_size",
            "light_type": "POINT",
        }
        with pytest.raises(NotFoundError) as missing:
            session.get_material_info("ghost")
        assert missing.value.details == {"code": "material_not_found"}


def test_true_is_not_a_face_index_through_the_mcp_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """face_indices [true] was read as face 1."""
    session = StrictBlenderSession()
    monkeypatch.setattr(backends_module, "is_blender_available", lambda: True)
    monkeypatch.setattr(
        backends_module, "BlenderRuntimeAdapter", lambda settings: BlenderSessionAdapter(settings, session)
    )
    instance = server_module.SimulMCPServer(
        Settings(blender={"mode": "embedded"}, security={"rate_limiting_enabled": False}),
        backends={"blender"},
    )

    async def call(arguments: dict) -> Any:
        async with Client(instance.mcp) as client:
            return await client.call_tool("assign_blender_material", arguments, raise_on_error=False)

    for arguments in (
        {"object_name": "Cube", "face_indices": [True]},
        {"object_name": "Cube", "slot_index": True},
    ):
        result = asyncio.run(call(arguments))
        text = result.content[-1].text
        assert result.is_error or json.loads(text).get("success") is False, text
    assert session.calls == []


def test_the_fake_mesh_tracks_edit_mode_and_libraries() -> None:
    """The fields the refusals read exist on the fake, as on bpy.types.Mesh."""
    registry = SimpleNamespace(bpy=SimpleNamespace(data=SimpleNamespace(objects=[])))
    mesh = FakeMesh(registry, "m")  # type: ignore[arg-type]
    assert (mesh.is_editmode, mesh.library) == (False, None)


# ---------------------------------------------------------------------------
# Re-review: single_user in Edit Mode, repeated textures, reroutes
# ---------------------------------------------------------------------------


class TestSingleUserSafety:
    @pytest.fixture
    def za(self, fake_bpy: FakeBpy) -> Any:
        """ZA and ZB share ZAMesh, which has two slots."""
        za = fake_bpy.add_mesh_object("ZA", CUBE_VERTICES, CUBE_FACES)
        za.data.name = "ZAMesh"
        fake_bpy.data.objects.new("ZB", za.data)
        za.data.materials.extend([fake_bpy.data.materials.new("a"), fake_bpy.data.materials.new("b")])
        return za

    @pytest.mark.parametrize("arguments", [{"slot_index": 2}, {"slot_index": 0}])
    def test_single_user_in_edit_mode_is_refused_and_removes_nothing(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, za: Any, arguments: Any
    ) -> None:
        """Blender ignores obj.data = copy in Edit Mode; the rollback then removed ZAMesh, ZA and ZB."""
        za.mode = "EDIT"
        shared = za.data

        with pytest.raises(ValidationError, match="to give it its own copy of its data") as refused:
            session.assign_material("ZA", "za_own", single_user=True, **arguments)

        assert refused.value.details["code"] == "wrong_mode"
        assert za.data is shared and fake_bpy.data.objects.get("ZB").data is shared
        assert [obj.name for obj in fake_bpy.data.objects] == ["ZA", "ZB"]
        assert [mesh.name for mesh in fake_bpy.data.meshes] == ["ZAMesh"]
        assert fake_bpy.data.meshes.removed == []
        assert _slots(za) == ["a", "b"]

    def test_a_copy_blender_does_not_take_is_dropped_and_the_original_kept(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, za: Any
    ) -> None:
        """Only the copy this call made is removed, never the data the object still uses."""
        shared = za.data
        za.data_locked = True

        with pytest.raises(RuntimeError, match="Blender kept 'ZAMesh' on 'ZA' instead of its copy"):
            session.assign_material("ZA", "za_own", slot_index=0, single_user=True)

        assert za.data is shared and fake_bpy.data.objects.get("ZB").data is shared
        assert [mesh.name for mesh in fake_bpy.data.meshes] == ["ZAMesh"]
        assert [mesh.name for mesh in fake_bpy.data.meshes.removed] == ["ZAMesh.001"]
        assert _slots(za) == ["a", "b"]
        assert fake_bpy.data.materials.get("za_own") is None

    def test_a_local_copy_of_library_data_is_worded_as_one(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession
    ) -> None:
        """It read "own copy 'LibMesh' of mesh data 'LibMesh'"."""
        obj = fake_bpy.add_mesh_object("lib_user", CUBE_VERTICES, CUBE_FACES)
        obj.data.name = "LibMesh"
        obj.data.library = SimpleNamespace(name="props.blend")

        result = session.assign_material("lib_user", "steel", single_user=True)

        assert result["warnings"] == [
            "'lib_user' now has a local copy of mesh data 'LibMesh' from library 'props.blend'"
        ]


class TestRepeatedTextures:
    @pytest.fixture
    def crate(self, fake_bpy: FakeBpy) -> Any:
        obj = fake_bpy.add_mesh_object("AgentBox", CUBE_VERTICES, CUBE_FACES)
        obj.data.uv_layers.new("UVMap")
        return obj

    def test_a_repeated_call_reuses_the_image_it_loaded(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, crate: Any, tmp_path: Path
    ) -> None:
        """Four calls left albedo.png plus .001 to .004, the older ones with no users."""
        albedo = _image(tmp_path, "albedo.png")
        other = fake_bpy.add_mesh_object("painted", CUBE_VERTICES, CUBE_FACES)
        other.data.uv_layers.new("UVMap")
        session.assign_material("painted", "painted_mat", base_color_texture=albedo)

        for _ in range(4):
            session.assign_material("AgentBox", "agent_rough", roughness_texture=albedo)

        assert sorted(image.name for image in fake_bpy.data.images) == ["albedo.png", "albedo.png.001"]

    def test_one_file_as_colour_and_data_stays_at_two_datablocks(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, crate: Any, tmp_path: Path
    ) -> None:
        path = _image(tmp_path, "packed.png")

        for _ in range(3):
            session.assign_material("AgentBox", "m", base_color_texture=path, roughness_texture=path)

        assert len(fake_bpy.data.images) == 2

    def test_the_same_file_by_another_spelling_is_reused(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, crate: Any, tmp_path: Path
    ) -> None:
        path = _image(tmp_path, "albedo.png")
        session.assign_material("AgentBox", "m", base_color_texture=path)

        session.assign_material("AgentBox", "m", base_color_texture=str(tmp_path / "." / "albedo.png"))

        assert len(fake_bpy.data.images) == 1

    def test_a_texture_replaced_behind_a_reroute_leaves_no_stray_nodes(
        self, fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession, crate: Any, tmp_path: Path
    ) -> None:
        """The old Image Texture and Reroute nodes stayed in the tree."""
        session.assign_material("AgentBox", "m", base_color_texture=_image(tmp_path, "a.png"))
        material = fake_bpy.data.materials.get("m")
        tree = material.node_tree
        texture = material.principled.inputs["Base Color"].links[0].from_node
        reroute = tree.nodes.new("NodeReroute")
        tree.links.new(texture.outputs["Color"], reroute.inputs["Input"])
        tree.links.new(reroute.outputs["Output"], material.principled.inputs["Base Color"])

        session.assign_material("AgentBox", "m", base_color_texture=_image(tmp_path, "b.png"))

        kinds = sorted(node.type for node in tree.nodes)
        assert kinds == ["BSDF_PRINCIPLED", "OUTPUT_MATERIAL", "TEX_IMAGE"]
        assert material.principled.inputs["Base Color"].links[0].from_node.image.name == "b.png"


def test_a_missing_object_is_a_typed_not_found(
    fake_bpy: FakeBpy, session: blender_runtime.BlenderRuntimeSession
) -> None:
    with pytest.raises(NotFoundError, match="Object not found: Ghost") as missing:
        session.assign_material("Ghost", "m")
    assert missing.value.details == {"code": "object_not_found", "object_name": "Ghost"}
