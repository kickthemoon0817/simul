"""The files an Interchange import reads besides the model, as Unreal 5.7 looks for them.

A review of ``import_unreal_asset`` found the sandbox checked only the file
handed to the tool: a model inside the allowed folders imported a texture, an
MTL library or a geometry buffer from outside them, through a relative or an
absolute path (the reviewer's fixtures: OBJ, glTF and FBX routes, each below).
``simul.utils.asset_references`` lists those reads where Interchange's
translators look; these tests pin down each route.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import pytest

from simul.config import Settings
from simul.utils.asset_references import (
    UncheckableReferences,
    fbx_file_names,
    fbx_relative_file_names,
    interchange_outside_references,
    udim_tiles,
)
from simul.utils.paths import PathPolicy


@pytest.fixture
def layout(tmp_path: Path) -> Tuple[Path, Path, PathPolicy]:
    """``sbx`` (the sandbox) and ``outside`` beside it, holding the secret image."""
    sbx, outside = tmp_path / "sbx", tmp_path / "outside"
    sbx.mkdir()
    outside.mkdir()
    (outside / "secret.png").write_bytes(b"\x89PNG secret")
    base = Settings()
    settings = base.model_copy(update={
        "security": base.security.model_copy(update={"allowed_paths": [str(sbx)]}),
    })
    return sbx, outside, PathPolicy.from_settings(settings)


def _real(path: Path) -> str:
    return os.path.realpath(path)


def _obj(folder: Path, name: str, mtllib: str, texture: str = "", mtl_at: Any = None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    model = folder / f"{name}.obj"
    model.write_text(f"mtllib {mtllib}\no {name}\nv 0 0 0\nv 1 0 0\nv 0 1 0\nusemtl Mat\nf 1 2 3\n")
    if texture:
        mtl = Path(mtl_at) if mtl_at is not None else folder / mtllib
        mtl.write_text(f"newmtl Mat\nKd 1 1 1\nmap_Kd {texture}\n")
    return model


def _gltf(folder: Path, name: str, buffer_uri: str, image_uri: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    if not buffer_uri.startswith(("data:", "..", "/")):
        (folder / buffer_uri).write_bytes(b"\0" * 12)
    document = {
        "asset": {"version": "2.0"}, "images": [{"uri": image_uri}],
        "buffers": [{"uri": buffer_uri, "byteLength": 12}],
    }
    model = folder / f"{name}.gltf"
    model.write_text(json.dumps(document))
    return model


# ---------------------------------------------------------------------------
# OBJ and MTL
# ---------------------------------------------------------------------------


def test_an_obj_texture_outside_by_a_relative_path_is_refused(layout: Any) -> None:
    """Fixture obj_rel: the MTL inside names ../../outside/secret.png."""
    sbx, outside, policy = layout
    model = _obj(sbx / "obj_rel", "relcube", "relcube.mtl", "../../outside/secret.png")

    refused, inside = interchange_outside_references(model, policy)

    assert refused == [_real(outside / "secret.png")]
    assert inside == [_real(sbx / "obj_rel" / "relcube.mtl")]


def test_an_obj_texture_outside_by_an_absolute_path_is_refused(layout: Any) -> None:
    """Fixture obj_abs."""
    sbx, outside, policy = layout
    model = _obj(sbx / "obj_abs", "abscube", "abscube.mtl", str(outside / "secret.png"))

    refused, _ = interchange_outside_references(model, policy)

    assert refused == [_real(outside / "secret.png")]


def test_an_mtl_library_outside_is_refused_and_its_textures_join_the_obj_folder(layout: Any) -> None:
    """Fixture obj_mtl: the MTL is outside; Interchange joins its texture to the OBJ's folder."""
    sbx, outside, policy = layout
    model = _obj(sbx / "obj_mtl", "mtlcube", "../../outside/evil.mtl", "secret.png", outside / "evil.mtl")

    refused, _ = interchange_outside_references(model, policy)

    # secret.png is looked for in sbx/obj_mtl, where there is none: only the library is read outside.
    assert refused == [_real(outside / "evil.mtl")]


def test_an_obj_whose_files_are_all_inside_passes_and_lists_them(layout: Any) -> None:
    """Fixture obj_ok, the control."""
    sbx, _, policy = layout
    model = _obj(sbx / "obj_ok", "okcube", "okcube.mtl", "inside.png")
    (sbx / "obj_ok" / "inside.png").write_bytes(b"\x89PNG")

    refused, inside = interchange_outside_references(model, policy)

    assert refused == []
    assert inside == [_real(sbx / "obj_ok" / "okcube.mtl"), _real(sbx / "obj_ok" / "inside.png")]


@pytest.mark.parametrize(
    "line",
    [
        "map_Kd -s 1 1 1 -bm 0.5 ../../outside/secret.png",
        "MAP_KD ../../outside/secret.png",
        "map_Kd\t..\\..\\outside\\secret.png",
        "bump -bm 2 ../../outside/secret.png",
        "map_Pr ../../outside/secret.png  # roughness",
        "norm ../../outside/secret.png",
    ],
)
def test_every_way_an_mtl_line_names_an_outside_texture_is_refused(layout: Any, line: str) -> None:
    """Options, case, tabs, backslashes and comments, as Interchange's OBJ translator reads them."""
    sbx, outside, policy = layout
    model = _obj(sbx / "obj", "cube", "cube.mtl")
    (sbx / "obj" / "cube.mtl").write_text(f"newmtl Mat\r\n{line}\r\n")

    refused, _ = interchange_outside_references(model, policy)

    assert refused == [_real(outside / "secret.png")]


def test_a_library_that_is_not_an_mtl_file_is_not_read(layout: Any) -> None:
    """Interchange reads a library only when its extension is .mtl."""
    sbx, outside, policy = layout
    (outside / "notes.txt").write_text("map_Kd secret.png\n")
    model = _obj(sbx / "obj", "cube", "../../outside/notes.txt")

    assert interchange_outside_references(model, policy) == ([], [])


# ---------------------------------------------------------------------------
# glTF and GLB
# ---------------------------------------------------------------------------


def test_a_gltf_image_outside_by_a_relative_uri_is_refused(layout: Any) -> None:
    """Fixture gltf_rel."""
    sbx, outside, policy = layout
    model = _gltf(sbx / "gltf_rel", "relball", "relball.bin", "../../outside/secret.png")

    refused, inside = interchange_outside_references(model, policy)

    assert refused == [_real(outside / "secret.png")]
    assert inside == [_real(sbx / "gltf_rel" / "relball.bin")]


def test_a_gltf_image_outside_by_a_file_url_is_refused(layout: Any) -> None:
    """Fixture gltf_abs: file:///…/outside/secret.png."""
    sbx, outside, policy = layout
    model = _gltf(sbx / "gltf_abs", "absball", "absball.bin", "file://" + str(outside / "secret.png"))

    refused, _ = interchange_outside_references(model, policy)

    assert refused == [_real(outside / "secret.png")]


def test_a_gltf_buffer_outside_is_refused(layout: Any) -> None:
    """Fixture gltf_bin: the geometry itself comes from outside."""
    sbx, outside, policy = layout
    (outside / "geom.bin").write_bytes(b"\0" * 12)
    model = _gltf(sbx / "gltf_bin", "binball", "../../outside/geom.bin", "inside.png")

    refused, _ = interchange_outside_references(model, policy)

    assert refused == [_real(outside / "geom.bin")]


def test_a_url_encoded_uri_is_decoded_as_interchange_decodes_it(layout: Any) -> None:
    sbx, outside, policy = layout
    model = _gltf(sbx / "gltf", "ball", "ball.bin", "..%2F..%2Foutside%2Fsecret.png")

    refused, _ = interchange_outside_references(model, policy)

    assert refused == [_real(outside / "secret.png")]


def test_a_glb_names_its_files_in_its_json_chunk(layout: Any) -> None:
    sbx, outside, policy = layout
    chunk = json.dumps({"asset": {"version": "2.0"}, "images": [{"uri": "../outside/secret.png"}]}).encode()
    chunk += b" " * (-len(chunk) % 4)
    body = struct.pack("<II", len(chunk), 0x4E4F534A) + chunk
    model = sbx / "ball.glb"
    model.write_bytes(struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body)

    refused, _ = interchange_outside_references(model, policy)

    assert refused == [_real(outside / "secret.png")]


def test_a_gltf_with_data_uris_and_inside_files_passes(layout: Any) -> None:
    """Fixture gltf_ok, the control."""
    sbx, _, policy = layout
    model = _gltf(sbx / "gltf_ok", "okball", "data:application/octet-stream;base64,AAAA", "inside.png")
    (sbx / "gltf_ok" / "inside.png").write_bytes(b"\x89PNG")

    assert interchange_outside_references(model, policy) == ([], [_real(sbx / "gltf_ok" / "inside.png")])


# ---------------------------------------------------------------------------
# FBX, binary and ASCII
# ---------------------------------------------------------------------------

Node = Tuple[str, Sequence[str], Sequence[Any]]


def _fbx_node(node: Node, offset: int, wide: bool) -> bytes:
    name, properties, children = node
    head = struct.Struct("<QQQB" if wide else "<IIIB")
    values = b"".join(b"S" + struct.pack("<I", len(value.encode())) + value.encode() for value in properties)
    position = offset + head.size + len(name) + len(values)
    nested = b""
    for child in children:
        encoded = _fbx_node(child, position + len(nested), wide)
        nested += encoded
    if children:
        nested += b"\0" * head.size
    end = offset + head.size + len(name) + len(values) + len(nested)
    return head.pack(end, len(properties), len(values), len(name)) + name.encode() + values + nested


def _fbx_binary(path: Path, nodes: List[Node], version: int = 7400) -> Path:
    wide = version >= 7500
    data = b"Kaydara FBX Binary  \x00\x1a\x00" + struct.pack("<I", version)
    for node in nodes:
        data += _fbx_node(node, len(data), wide)
    data += b"\0" * (25 if wide else 13)
    path.write_bytes(data)
    return path


def _from_cwd(cwd: Path) -> Any:
    """Where the editor opens a relative name: its working directory, here ``cwd``."""
    return lambda name: [str(cwd / name)]


def _texture(file_name: str, relative: str) -> Node:
    return ("Objects", [], [("Texture", ["Texture::Wood"], [
        ("FileName", [file_name], []), ("RelativeFilename", [relative], []),
    ])])


@pytest.mark.parametrize("version", [7400, 7500])
def test_a_binary_fbx_texture_outside_by_its_absolute_file_name_is_refused(layout: Any, version: int) -> None:
    """Fixture fbx/crate.fbx: Interchange's FBX SDK parser opens FileName as given first."""
    sbx, outside, policy = layout
    model = _fbx_binary(sbx / "crate.fbx", [_texture(str(outside / "secret.png"), "tex/wood.png")], version)

    refused, _ = interchange_outside_references(model, policy, _from_cwd(sbx / "cwd"))

    assert refused == [_real(outside / "secret.png")]


def test_a_binary_fbx_texture_outside_by_its_relative_name_is_refused(layout: Any) -> None:
    """RelativeFilename is joined to the FBX's folder, with backslashes made slashes."""
    sbx, outside, policy = layout
    model = _fbx_binary(sbx / "crate.fbx", [_texture("C:/gone/wood.png", "..\\outside\\secret.png")])

    refused, _ = interchange_outside_references(model, policy, _from_cwd(sbx / "cwd"))

    assert refused == [_real(outside / "secret.png")]


def test_a_video_path_in_a_property_list_is_read_too(layout: Any) -> None:
    sbx, outside, policy = layout
    record: Node = ("P", ["Path", "KString", "XRefUrl", "", str(outside / "secret.png")], [])
    model = _fbx_binary(sbx / "crate.fbx", [("Objects", [], [("Video", ["Video::V"], [
        ("Properties70", [], [record]),
    ])])])

    assert fbx_file_names(model) == [str(outside / "secret.png")]
    assert interchange_outside_references(model, policy)[0] == [_real(outside / "secret.png")]


def test_an_ascii_fbx_names_its_textures_the_same_way(layout: Any) -> None:
    sbx, outside, policy = layout
    model = sbx / "crate.fbx"
    model.write_text(
        "; FBX 7.4.0 project file\nObjects:  {\n  Texture: 1, \"Texture::Wood\", \"\" {\n"
        f"    FileName: \"{outside / 'secret.png'}\"\n    RelativeFilename: \"wood.png\"\n  }}\n}}\n"
    )

    refused, _ = interchange_outside_references(model, policy, _from_cwd(sbx / "cwd"))

    assert refused == [_real(outside / "secret.png")]


def test_a_relative_file_name_is_checked_where_the_editor_opens_it(layout: Any) -> None:
    """The FBX SDK opens a relative FileName from the editor's working directory."""
    sbx, outside, policy = layout
    model = _fbx_binary(sbx / "crate.fbx", [_texture("secret.png", "secret.png")])

    assert fbx_relative_file_names(model) == ["secret.png"]
    with pytest.raises(UncheckableReferences, match="by a relative path"):
        interchange_outside_references(model, policy)
    refused, _ = interchange_outside_references(model, policy, _from_cwd(outside))
    assert refused == [_real(outside / "secret.png")]


def test_a_binary_fbx_that_cannot_be_parsed_cannot_be_vouched_for(layout: Any) -> None:
    sbx, outside, policy = layout
    model = _fbx_binary(sbx / "crate.fbx", [_texture(str(outside / "secret.png"), "x.png")])
    model.write_bytes(model.read_bytes()[:40])

    with pytest.raises(UncheckableReferences, match="could not read the textures crate.fbx names"):
        interchange_outside_references(model, policy)


# ---------------------------------------------------------------------------
# UDIM tiles, and `..` judged as Unreal collapses it
# ---------------------------------------------------------------------------


def test_a_udim_tile_that_links_outside_is_refused(layout: Any) -> None:
    """Interchange imports wood.1002.png with wood.1001.png; the tile may be a symlink out."""
    sbx, outside, policy = layout
    folder = sbx / "obj"
    model = _obj(folder, "cube", "cube.mtl", "wood.1001.png")
    (folder / "wood.1001.png").write_bytes(b"\x89PNG")
    os.symlink(outside / "secret.png", folder / "wood.1002.png")

    assert udim_tiles(str(folder / "wood.1001.png")) == [str(folder / "wood.1002.png")]
    refused, _ = interchange_outside_references(model, policy)
    assert refused == [_real(outside / "secret.png")]


def test_dot_dot_is_judged_as_unreal_collapses_it_before_any_symlink(layout: Any, tmp_path: Path) -> None:
    """sbx/link points deep elsewhere; the OS follows it, Unreal's paths collapse link/.. away."""
    sbx, outside, policy = layout
    deep = tmp_path / "elsewhere" / "deep"
    deep.mkdir(parents=True)
    folder = sbx / "obj"
    folder.mkdir()
    os.symlink(deep, folder / "link")
    # The OS reads elsewhere/outside/secret.png, which does not exist; Unreal reads sbx/../outside/secret.png.
    model = _obj(folder, "cube", "cube.mtl", "link/../../../outside/secret.png")

    refused, _ = interchange_outside_references(model, policy)

    assert refused == [_real(outside / "secret.png")]


def test_a_reference_to_a_file_that_does_not_exist_reads_nothing(layout: Any) -> None:
    sbx, _, policy = layout
    model = _obj(sbx / "obj", "cube", "cube.mtl", "../../outside/missing.png")

    refused, _ = interchange_outside_references(model, policy)

    assert refused == []
