"""The files a 3D model names besides itself, and where each importer looks for them.

A sandbox that checks only the file a tool is handed lets that file pull in
others: a glTF inside the allowed folders whose image or buffer URI points
outside them, an OBJ whose MTL library or texture does, an FBX whose texture
does. The importer reads them, and the tool hands their content back, as a
texture asset, a mesh built from outside bytes or a name in an error.

For one reference a file makes, a reader lists the paths an importer tries,
in the order it tries them; the first that exists is what it reads.
``outside_references`` returns those reads that a ``PathPolicy`` refuses. Two
importers are modelled, because they look in different places:

- Blender's (``blender_*``, used by ``blender_runtime.import_file``).
- Unreal 5.7's Interchange (``interchange_*``, used by
  ``unreal_runtime.import_asset``), read from the engine's sources: OBJ
  textures join the OBJ's folder even when the MTL lives elsewhere and name
  the last token of the line, glTF URIs are URL-decoded and joined to the
  file's folder, an FBX texture is tried as its stored FileName (a relative
  one from the editor's base directory), then RelativeFilename and FileName
  joined to the FBX's folder, and a texture named like a UDIM tile pulls in
  its sibling tiles.

USD composition is held to the sandbox by ``simul.usd.sandbox``, which walks
layers with pxr; Blender's USD reader lives in ``blender_runtime``.

The module imports neither bpy nor unreal: the MCP server uses it, and the
Blender add-on bundles it next to ``blender_runtime``.
"""

from __future__ import annotations

import json
import mmap
import os
import re
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .paths import PathPolicy


class UncheckableReferences(Exception):
    """The files an import would read cannot be listed, so the sandbox cannot vouch for them."""


# -- Blender's importers (blender_runtime.import_file, B1-20) -----------------
#
# The sandbox checked only the path handed to import_file, so a glTF inside the
# allowed folders whose image URI pointed outside them imported that image, and
# a later GLB export embedded it. Each blender_* function lists, for one
# reference the file makes, the paths Blender's importer tries in the order it
# tries them; the first that exists is what it reads.

# MTL texture options and how many arguments each takes at most (the OBJ
# importer's parse_texture_option); numeric ones stop at the first non-number.
MTL_OPTION_ARGUMENTS = {
    "-o": 3,
    "-s": 3,
    "-t": 3,
    "-mm": 2,
    "-bm": 1,
    "-boost": 1,
    "-texres": 1,
    "-blendu": 1,
    "-blendv": 1,
    "-clamp": 1,
    "-cc": 1,
    "-imfchan": 1,
    "-type": 1,
}
MTL_NUMERIC_OPTIONS = frozenset({"-o", "-s", "-t", "-mm", "-bm", "-boost", "-texres"})
MTL_MAP_KEYWORDS = frozenset({"bump", "refl", "norm", "disp", "decal"})


def unquote_path(text: str) -> str:
    """Strip whitespace and the surrounding double quotes the OBJ importer strips."""
    text = text.strip()
    if len(text) > 2 and text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    return text


def blender_joined(directory: str, path: str) -> str:
    """Join as Blender's BLI_path_join does: a later absolute part stays inside."""
    return os.path.normpath(os.path.join(directory, path.lstrip("/\\")))


def gltf_document(path: Path) -> Dict[str, Any]:
    """Read the JSON of a .gltf, or the JSON chunk of a .glb."""
    with path.open("rb") as handle:
        header = handle.read(20)
        if header[:4] == b"glTF":
            length, kind = struct.unpack_from("<II", header, 12)
            if kind != 0x4E4F534A:  # "JSON"
                raise ValueError(f"{path.name} has no JSON chunk")
            return dict(json.loads(handle.read(length)))
        return dict(json.loads(header + handle.read()))


def blender_gltf_reads(path: Path) -> List[List[str]]:
    """List the buffer and image files a glTF or GLB names, as its importer opens them."""
    from urllib.parse import unquote

    document = gltf_document(path)
    base = str(path.parent)
    reads = []
    for key in ("buffers", "images"):
        for entry in document.get(key) or []:
            uri = entry.get("uri") if isinstance(entry, dict) else None
            if not isinstance(uri, str) or not uri or uri.startswith("data:"):
                continue
            # io_scene_gltf2: join(dirname(gltf), normpath(unquote(uri))).
            local = os.path.normpath(unquote(uri.replace("\\", "/")))
            reads.append([os.path.join(base, local)])
    return reads


def mtl_texture_path(rest: str) -> str:
    """Return the image path of an MTL map line, after its options."""
    text = rest.strip()
    while text.startswith("-"):
        option, _, text = text.partition(" ")
        text = text.lstrip()
        for _ in range(MTL_OPTION_ARGUMENTS.get(option, 1)):
            argument, _, remainder = text.partition(" ")
            if not argument:
                break
            if option in MTL_NUMERIC_OPTIONS:
                try:
                    float(argument)
                except ValueError:
                    break
            text = remainder.lstrip()
    return unquote_path(text)


def blender_obj_reads(path: Path) -> List[List[str]]:
    """List the MTL libraries an OBJ names and the textures they name.

    The OBJ importer joins each mtllib to the OBJ's folder, and tries a
    texture joined to the MTL's folder, as given (absolute, or relative to the
    working directory), with underscores turned to spaces, and its base name
    in the MTL's folder.
    """
    reads: List[List[str]] = []
    folder = str(path.parent)
    libraries: List[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("mtllib"):
                name = unquote_path(line[6:])
                if name:
                    libraries.append(blender_joined(folder, name))
    for library in dict.fromkeys(libraries):
        reads.append([library])
        if not os.path.isfile(library):
            continue
        library_folder = os.path.dirname(library)
        with open(library, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                keyword, _, rest = line.strip().partition(" ")
                keyword = keyword.lower()
                if not (keyword.startswith("map_") or keyword in MTL_MAP_KEYWORDS):
                    continue
                image = mtl_texture_path(rest)
                if image:
                    reads.append(
                        [
                            blender_joined(library_folder, image),
                            os.path.abspath(image),
                            os.path.abspath(image.replace("_", " ")),
                            os.path.join(library_folder, os.path.basename(image)),
                        ]
                    )
    return reads


def resolve_ncase(path: str) -> str:
    """Find a path's real spelling on a case-sensitive file system.

    Mirrors ``bpy.path.resolve_ncase`` (Blender 4.2 through 5.0.1), which
    ``bpy_extras.image_utils.load_image`` tries after each path: a component
    that does not exist is matched case-insensitively among its folder's
    entries. Returns the path unchanged when nothing matches.
    """

    def found(candidate: str) -> Tuple[str, bool]:
        if not candidate or os.path.exists(candidate):
            return candidate, True
        filename, folder = os.path.basename(candidate), os.path.dirname(candidate)
        suffix = ""
        if not filename:
            if len(folder) < len(candidate):
                suffix = candidate[len(folder) :]
            filename, folder = os.path.basename(folder), os.path.dirname(folder)
        if not os.path.exists(folder):
            if folder == candidate:
                return candidate, False
            folder, ok = found(folder)
            if not ok:
                return candidate, False
        if not os.path.isdir(folder):
            return candidate, False
        try:
            entries = os.listdir(folder)
        except OSError:
            return candidate, False
        lower = filename.lower()
        match = next((entry for entry in entries if entry.lower() == lower), None)
        if match is None:
            return candidate, False
        return os.path.join(folder, match) + suffix, True

    return found(path)[0]


def fbx_element_name(element: Any) -> str:
    """An FBX object's name: the part of its name property before the class."""
    props = getattr(element, "props", None) or []
    if len(props) < 2 or not isinstance(props[-2], bytes):
        return ""
    return props[-2].split(b"\x00\x01")[0].decode("utf-8", "replace")


def blender_fbx_reads(path: Path, allowed: Callable[[str], bool]) -> List[List[str]]:
    """List the texture files an FBX names, as Blender's FBX importer looks for them.

    blen_read_texture_image takes a texture's RelativeFilename joined to the
    FBX's folder (stripped of a leading separator), else its FileName or
    Filename, else the texture's name. bpy_extras.image_utils.load_image then
    tries that path, the path joined to the folder and its base name there,
    each also with its case resolved (resolve_ncase). Its recursive search of
    folders, the importer's use_image_search, is switched off for an import
    the sandbox checks (import_file), so it reads nothing found that way.
    Blender's own FBX parser reads the file; a file it cannot parse fails the
    import itself.
    """
    try:
        from io_scene_fbx import parse_fbx  # type: ignore[import-not-found]
    except ImportError as exc:
        raise UncheckableReferences("Blender's FBX add-on is not available") from exc

    try:
        root, _version = parse_fbx.parse(str(path))
    except Exception:
        return []
    base = str(path.parent)
    reads: List[List[str]] = []
    for section in root.elems:
        if section.id != b"Objects":
            continue
        for element in section.elems:
            if element.id not in (b"Texture", b"Video"):
                continue
            names: Dict[bytes, str] = {}
            for prop in element.elems:
                if prop.id in (b"RelativeFilename", b"FileName", b"Filename") and prop.props:
                    value = prop.props[0]
                    if isinstance(value, bytes):
                        value = value.decode("utf-8", "replace")
                    names.setdefault(prop.id, str(value))
            relative = names.get(b"RelativeFilename")
            if relative:
                # str.lstrip(None) strips whitespace, as the importer's
                # lstrip(os.path.altsep) does where altsep is None.
                image = os.path.join(base, relative.lstrip(os.path.sep).lstrip(os.path.altsep))
            else:
                image = names.get(b"FileName") or names.get(b"Filename") or ""
            if not image:
                image = fbx_element_name(element)
            if not image:
                continue
            # bpy.path.native_pathsep, as load_image applies it.
            image = image.replace("\\", "/") if os.sep == "/" else image.replace("/", "\\")
            short = image[2:] if image[:2] == "//" else image
            candidates = [image]
            for variant in (image, os.path.join(base, image), os.path.join(base, os.path.basename(short))):
                candidates.extend((variant, resolve_ncase(variant)))
            reads.append(list(dict.fromkeys(candidates)))
    return reads


# -- Judging the reads ---------------------------------------------------------


def _found(candidates: Sequence[str], lexical: bool) -> List[str]:
    """The file an importer reads for one reference: the first candidate that exists.

    With ``lexical``, the candidates are also tried with ``..`` collapsed
    before any symlink is followed, as Unreal's path functions collapse them
    (``FPaths::CollapseRelativeDirectories``); the operating system follows a
    symlink first. Each reading's file is returned, so both are judged.
    """
    forms = [list(candidates)]
    if lexical:
        forms.append([os.path.normpath(candidate) for candidate in candidates])
    hits: List[str] = []
    for form in forms:
        hit = next((candidate for candidate in form if candidate and os.path.exists(candidate)), None)
        if hit is not None and hit not in hits:
            hits.append(hit)
    return hits


def outside_references(
    reads: Iterable[Sequence[str]], policy: "PathPolicy", *, lexical: bool = False
) -> List[str]:
    """The files the reads would load that ``policy`` refuses.

    A reference to a file that does not exist reads nothing and passes.

    Args:
        reads: For each reference, the paths the importer tries in order.
        policy: The sandbox the imported file itself passed.
        lexical: Also judge each candidate with ``..`` collapsed (Unreal).

    Returns:
        Each refused file as it is, without ``..`` steps or symlinks, once.
    """
    outside: List[str] = []
    for candidates in reads:
        for found in _found(candidates, lexical):
            if not policy.is_allowed(found):
                real = str(policy.resolve(found))
                if real not in outside:
                    outside.append(real)
    return outside


# -- Unreal 5.7 Interchange (unreal_runtime.import_asset) ----------------------

#: The model formats Interchange imports for ``import_asset``, by extension.
INTERCHANGE_FORMATS: Dict[str, str] = {".fbx": "FBX", ".gltf": "GLTF", ".glb": "GLTF", ".obj": "OBJ"}

# MTL keywords whose argument is a texture (InterchangeOBJTranslator's
# MtlKeywordMap). Keywords match ignoring case, as an FString-keyed TMap does.
# Any other map_ keyword, and Blender's extra ones, are held to the sandbox too.
_INTERCHANGE_MTL_TEXTURES = frozenset(
    {"map_kd", "map_ka", "map_ks", "map_ke", "map_d", "map_tr", "map_ns", "map_bump", "bump",
     "map_pr", "map_pm", "norm", "refl", "disp", "decal"}
)

# Image files Interchange's texture translators read, for the UDIM rule.
_IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".tga", ".exr", ".tif", ".tiff", ".bmp", ".psd", ".hdr", ".dds", ".tx",
     ".uej", ".ies", ".pcx"}
)

# UTextureImportSettings::UdimRegexPattern's default, "(.+?)[._](\d{4})$".
_UDIM_NAME = re.compile(r"^(.+?)[._](\d{4})$")


def _ue_join(directory: str, relative: str) -> str:
    """``Directory / Relative`` as Unreal's FString::PathAppend joins, with backslashes made slashes.

    A leading slash does not make ``relative`` absolute: ``"/a" / "/b"`` is
    ``/a/b``. Unreal's file layer turns backslashes into slashes
    (``FPaths::NormalizeFilename``) before the operating system sees a path.
    """
    relative = relative.replace("\\", "/")
    if not relative:
        return directory
    if directory.endswith("/") or relative.startswith("/"):
        return directory + relative
    return directory + "/" + relative


def _absolute_as_given(name: str) -> Optional[str]:
    """``name`` as an absolute path, when it is one: read too, in case an importer reads it as given."""
    name = name.replace("\\", "/")
    if name.lower().startswith("file://"):
        from urllib.parse import unquote, urlparse

        name = unquote(urlparse(name).path)
    return name if os.path.isabs(name) else None


def interchange_gltf_reads(path: Path) -> List[List[str]]:
    """The buffer and image files a glTF or GLB names, as Interchange's glTF reader opens them.

    GLTF::FFileReader URL-decodes each URI that is not a ``data:`` URI and
    joins it to the file's folder (``Path / UrlDecode(URI)``). Both decodings
    of ``+`` are tried, and a URI that is an absolute path or a ``file://``
    URL is also tried as given.
    """
    from urllib.parse import unquote, unquote_plus

    try:
        document = gltf_document(path)
    except (OSError, ValueError, struct.error):
        # Interchange's reader cannot parse it either, and reads nothing through it.
        return []
    base = str(path.parent)
    reads: List[List[str]] = []
    for key in ("buffers", "images"):
        for entry in document.get(key) or []:
            uri = entry.get("uri") if isinstance(entry, dict) else None
            if not isinstance(uri, str) or not uri or uri.startswith("data:"):
                continue
            for decoded in dict.fromkeys((unquote(uri), unquote_plus(uri))):
                reads.append([_ue_join(base, decoded)])
                given = _absolute_as_given(decoded)
                if given is not None:
                    reads.append([given])
    return reads


def _ue_tokens(line: str) -> Tuple[str, str]:
    """A line's keyword and the rest after it, as the OBJ translator's GetToken splits them.

    Tokens end at a space or a tab; a token that starts with ``#`` is a
    comment that ends the line.
    """
    rest = line.lstrip(" \t")
    if not rest or rest.startswith("#"):
        return "", ""
    end = len(rest)
    for index, char in enumerate(rest):
        if char in " \t":
            end = index
            break
    return rest[:end], rest[end + 1:].lstrip(" \t")


def _mtl_texture_names(rest: str) -> List[str]:
    """Every token of an MTL map line, each a name Interchange might open.

    InterchangeOBJTranslator takes the line's last token as the file, except
    that ``-bm`` swallows the token after it; every token is checked, so a
    difference in how an option is parsed cannot let a file through.
    """
    names: List[str] = []
    tokens = rest
    while True:
        token, tokens = _ue_tokens(tokens)
        if not token:
            return names
        names.append(token)


def _text_lines(path: str) -> List[str]:
    """A text file's lines, split on ``\\r\\n``, ``\\r`` or ``\\n`` as Unreal's line visitor splits them."""
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as handle:
        return re.split(r"\r\n|\r|\n", handle.read())


def interchange_obj_reads(path: Path) -> List[List[str]]:
    """The MTL libraries an OBJ names and the textures they name, as Interchange's OBJ translator opens them.

    ``mtllib`` takes the rest of its line and is joined to the OBJ's folder; a
    library whose extension is not ``.mtl`` is not read. A texture is the
    last token of its map line, joined to the OBJ's folder too, wherever the
    MTL is; every token of the line is checked. A name that is an absolute
    path is also tried as given.
    """
    folder = str(path.parent)
    libraries: List[str] = []
    for line in _text_lines(str(path)):
        keyword, rest = _ue_tokens(line)
        if keyword.lower() != "mtllib" or not rest:
            continue
        for name in dict.fromkeys((rest, rest.rstrip(" \t"))):
            if os.path.splitext(name)[1].lower() != ".mtl":
                continue
            libraries.append(_ue_join(folder, name))
            given = _absolute_as_given(name)
            if given is not None:
                libraries.append(given)
    reads: List[List[str]] = []
    for library in dict.fromkeys(libraries):
        reads.append([library])
        if not os.path.isfile(library):
            continue
        for line in _text_lines(library):
            keyword, rest = _ue_tokens(line)
            keyword = keyword.lower()
            if not (keyword in _INTERCHANGE_MTL_TEXTURES or keyword.startswith("map_")):
                continue
            for name in _mtl_texture_names(rest):
                reads.append([_ue_join(folder, name)])
                given = _absolute_as_given(name)
                if given is not None:
                    reads.append([given])
    return reads


# -- FBX, without Blender's parser ---------------------------------------------

_FBX_BINARY_MAGIC = b"Kaydara FBX Binary  \x00"
# The nodes naming a texture's or video's file, and the property records
# (FBX 7 Properties70 P, FBX 6 Properties60 Property) whose "Path" names one.
_FBX_FILE_NODES = frozenset({b"RelativeFilename", b"FileName", b"Filename"})
_FBX_PROPERTY_LISTS = frozenset({b"Properties70", b"Properties60"})
_FBX_PROPERTY_RECORDS = frozenset({b"P", b"Property"})
_FBX_MAX_NODES = 5_000_000
_FBX_SCALARS = {ord("Y"): 2, ord("C"): 1, ord("I"): 4, ord("F"): 4, ord("D"): 8, ord("L"): 8}
_FBX_ARRAYS = frozenset(ord(code) for code in "fdlib")


def _fbx_properties(buffer: Any, start: int, end: int, count: int) -> List[Any]:
    """The properties of one binary FBX node; strings as str, other values as None."""
    values: List[Any] = []
    position = start
    for _ in range(count):
        if position >= end:
            raise ValueError("an FBX property runs past its node")
        code = buffer[position]
        position += 1
        if code in (ord("S"), ord("R")):
            (length,) = struct.unpack_from("<I", buffer, position)
            position += 4
            raw = bytes(buffer[position:position + length])
            position += length
            values.append(raw.decode("utf-8", "replace") if code == ord("S") else None)
        elif code in _FBX_SCALARS:
            position += _FBX_SCALARS[code]
            values.append(None)
        elif code in _FBX_ARRAYS:
            _length, _encoding, compressed = struct.unpack_from("<III", buffer, position)
            position += 12 + compressed
            values.append(None)
        else:
            raise ValueError(f"unknown FBX property type {chr(code)!r}")
    if position > end:
        raise ValueError("an FBX property runs past its node")
    return values


def _fbx_binary_files(buffer: Any) -> List[str]:
    """The file names a binary FBX stores, walking every node."""
    if len(buffer) < 27:
        raise ValueError("the FBX header is truncated")
    (version,) = struct.unpack_from("<I", buffer, 23)
    header = struct.Struct("<QQQB" if version >= 7500 else "<IIIB")
    found: List[str] = []
    stack: List[Tuple[int, int, bytes]] = [(27, len(buffer), b"")]
    nodes = 0
    while stack:
        position, end, parent = stack.pop()
        closed = False
        while position + header.size <= end:
            end_offset, count, length, name_length = header.unpack_from(buffer, position)
            if end_offset == 0:
                closed = True  # the null record that closes a node list
                break
            nodes += 1
            if nodes > _FBX_MAX_NODES:
                raise ValueError("the FBX file has too many nodes to check")
            name_start = position + header.size
            properties_start = name_start + name_length
            children_start = properties_start + length
            if not (position < properties_start <= children_start <= end_offset <= end):
                raise ValueError("an FBX node's offsets are inconsistent")
            name = bytes(buffer[name_start:properties_start])
            if name in _FBX_FILE_NODES or (name in _FBX_PROPERTY_RECORDS and parent in _FBX_PROPERTY_LISTS):
                values = _fbx_properties(buffer, properties_start, children_start, count)
                strings = [value for value in values if isinstance(value, str)]
                if name in _FBX_FILE_NODES and strings:
                    found.append(strings[0])
                elif strings and strings[0] == "Path" and len(strings) > 1:
                    found.append(strings[-1])
            if children_start < end_offset:
                stack.append((children_start, end_offset, name))
            position = end_offset
        if not closed:
            raise ValueError("an FBX node list ends without its null record")
    return found


#: Where the editor opens a relative FBX texture name as given: one path, a
#: list of the places it might, or None when that is not known.
RelativeResolver = Callable[[str], Union[None, str, Sequence[str]]]

_FBX_ASCII_FILE = re.compile(r'\b(?:RelativeFilename|FileName|Filename)\s*:\s*"([^"]*)"')
_FBX_ASCII_PATH = re.compile(
    r'\b(?:P|Property)\s*:\s*"Path"\s*,\s*"[^"]*"\s*,\s*"[^"]*"\s*,\s*(?:"[^"]*"\s*,\s*)?"([^"]*)"'
)


def fbx_file_names(path: Path) -> List[str]:
    """Every texture or video file name an FBX stores: FileName, Filename, RelativeFilename and Path.

    Binary FBX of any version is walked node by node, skipping (never
    inflating) compressed arrays; ASCII FBX is searched for the same fields.
    The names are not grouped by texture: each is checked on its own, which
    refuses whatever a grouped check would.

    Raises:
        UncheckableReferences: For a binary FBX this cannot parse, which
            Unreal's FBX SDK might still read.
    """
    with open(path, "rb") as handle:
        head = handle.read(len(_FBX_BINARY_MAGIC))
        if head != _FBX_BINARY_MAGIC:
            handle.seek(0)
            text = handle.read().decode("utf-8", "replace").replace("&quot;", '"')
            names = _FBX_ASCII_FILE.findall(text) + _FBX_ASCII_PATH.findall(text)
            return list(dict.fromkeys(names))
        try:
            with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as buffer:
                return list(dict.fromkeys(_fbx_binary_files(buffer)))
        except (ValueError, struct.error, OverflowError) as exc:
            raise UncheckableReferences(f"could not read the textures {path.name} names ({exc})") from exc


def fbx_relative_file_names(path: Path) -> List[str]:
    """The FileName or Path values an FBX stores that are not absolute paths.

    Unreal tries a texture's FileName as given, and the editor resolves a
    relative one from its base directory; ``interchange_fbx_reads`` needs
    those resolved by the editor.
    """
    return [name for name in fbx_file_names(path) if name and _absolute_as_given(name) is None]


def interchange_fbx_reads(path: Path, resolve_relative: RelativeResolver = lambda name: None) -> List[List[str]]:
    """The texture files an FBX names, as Interchange's FBX parsers look for them.

    FFbxMaterial (the FBX SDK parser, Unreal 5.7's default) and the ufbx
    parser try a texture's FileName as given, its RelativeFilename joined to
    the FBX's folder, and its FileName joined to that folder; neither
    searches further. Each name stored is checked both ways, as given and
    joined.

    Args:
        path: The FBX file.
        resolve_relative: Where the editor opens a relative name as given
            (its working directory, or the base directory
            ``unreal.Paths.convert_relative_path_to_full`` uses): one path or
            several, each checked; None when that is not known.

    Raises:
        UncheckableReferences: The file cannot be parsed, or a relative name
            cannot be resolved.
    """
    base = str(path.parent)
    reads: List[List[str]] = []
    for name in fbx_file_names(path):
        if not name:
            continue
        absolute = _absolute_as_given(name)
        given: Union[None, str, Sequence[str]] = absolute if absolute is not None else resolve_relative(name)
        if not given:
            raise UncheckableReferences(
                f"{path.name} names the texture {name!r} by a relative path, which the editor "
                "opens from its own working directory"
            )
        for opened in [given] if isinstance(given, str) else given:
            reads.append([opened, _ue_join(base, name)])
    return reads


def udim_tiles(path: str) -> List[str]:
    """The other UDIM tiles Interchange imports with a texture named like one (``wood.1001.png``).

    Unreal's texture pipeline (bImportUDIMs, on by default) gathers every
    file in the texture's folder with the same prefix, a four-digit tile and
    the same extension.
    """
    folder, filename = os.path.split(path)
    stem, extension = os.path.splitext(filename)
    match = _UDIM_NAME.match(stem)
    if match is None or extension.lower() not in _IMAGE_EXTENSIONS:
        return []
    tile = re.compile(re.escape(match.group(1)) + r"[._]\d{4}" + re.escape(extension) + "$", re.IGNORECASE)
    try:
        entries = os.listdir(folder or ".")
    except OSError:
        return []
    return [os.path.join(folder, entry) for entry in sorted(entries) if entry != filename and tile.match(entry)]


def interchange_reads(path: Path, resolve_relative: RelativeResolver = lambda name: None) -> List[List[str]]:
    """The files Interchange reads importing ``path`` (FBX, glTF, GLB or OBJ), besides ``path``."""
    kind = INTERCHANGE_FORMATS.get(path.suffix.lower())
    if kind == "GLTF":
        return interchange_gltf_reads(path)
    if kind == "OBJ":
        return interchange_obj_reads(path)
    if kind == "FBX":
        return interchange_fbx_reads(path, resolve_relative)
    return []


def interchange_outside_references(
    path: Path,
    policy: "PathPolicy",
    resolve_relative: RelativeResolver = lambda name: None,
) -> Tuple[List[str], List[str]]:
    """The files an Interchange import of ``path`` would read outside the sandbox, and those inside.

    Every referenced file is judged as Unreal opens it and with ``..``
    collapsed (``outside_references`` with ``lexical``), and the UDIM tiles a
    texture inside the sandbox pulls in are judged too (a tile may be a
    symlink out of it).

    Returns:
        The refused files, resolved; and the allowed files the import reads
        (for telling the assets it made from them apart).

    Raises:
        UncheckableReferences: The references cannot be listed.
    """
    reads = interchange_reads(path, resolve_relative)
    outside = outside_references(reads, policy, lexical=True)
    inside: List[str] = []
    tiles: List[List[str]] = []
    for candidates in reads:
        for found in _found(candidates, True):
            if policy.is_allowed(found):
                inside.append(str(policy.resolve(found)))
                tiles.extend([tile] for tile in udim_tiles(found))
    for refused in outside_references(tiles, policy):
        if refused not in outside:
            outside.append(refused)
    for candidates in tiles:
        for found in _found(candidates, False):
            if policy.is_allowed(found):
                inside.append(str(policy.resolve(found)))
    return outside, list(dict.fromkeys(inside))


__all__ = [
    "INTERCHANGE_FORMATS",
    "RelativeResolver",
    "UncheckableReferences",
    "blender_fbx_reads",
    "blender_gltf_reads",
    "blender_joined",
    "blender_obj_reads",
    "fbx_file_names",
    "fbx_relative_file_names",
    "gltf_document",
    "interchange_fbx_reads",
    "interchange_gltf_reads",
    "interchange_obj_reads",
    "interchange_outside_references",
    "interchange_reads",
    "mtl_texture_path",
    "outside_references",
    "resolve_ncase",
    "udim_tiles",
    "unquote_path",
]
