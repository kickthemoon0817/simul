"""A ``bpy`` stand-in for the mesh, modifier, material and light tools.

``FakeBpy`` holds datablock registries (``bpy.data.objects``, ``meshes``,
``collections``, ...) that name new datablocks as Blender does (``Cube``,
``Cube.001``) and record removals, so a test can check that a refused call
leaves nothing behind. Modifiers carry an RNA description
(``bl_rna.properties``) with the identifiers, types, limits and enum items of
Blender 5.0.1, and assigning to one behaves as Blender does where the tools
depend on it: an enum takes only its items, an enum flag only a set, a pointer
drops a datablock its poll refuses without an error, and a number outside the
hard limits is clamped.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

FLT_MAX = 3.4028234663852886e38
INT_MAX = 2147483647


class FakeRNAProp:
    """One ``bpy.types.Property``: identifier, type, writability, limits and items."""

    def __init__(
        self,
        identifier: str,
        kind: str,
        *,
        readonly: bool = False,
        items: Sequence[str] = (),
        flag: bool = False,
        fixed_type: Optional[str] = None,
        hard: Tuple[float, float] = (-FLT_MAX, FLT_MAX),
        array_length: int = 0,
        poll: Optional[Callable[[Any], bool]] = None,
        default: Any = None,
    ) -> None:
        self.identifier = identifier
        self.type = kind
        self.is_readonly = readonly
        self.enum_items = [SimpleNamespace(identifier=item) for item in items]
        self.is_enum_flag = flag
        self.fixed_type = (
            None if fixed_type is None else SimpleNamespace(identifier=fixed_type, base=None)
        )
        self.hard_min, self.hard_max = hard
        self.array_length = array_length
        self.poll = poll
        self.default = default


class FakeRNAProperties(list):
    """``bl_rna.properties``: the properties in order, and ``get`` by identifier."""

    def get(self, identifier: str, default: Any = None) -> Any:
        return next((prop for prop in self if prop.identifier == identifier), default)

    def __getitem__(self, key: Any) -> Any:  # type: ignore[override]
        if isinstance(key, str):
            prop = self.get(key)
            if prop is None:
                raise KeyError(key)
            return prop
        return super().__getitem__(key)


class FakeRNAStruct:
    """A struct whose writable settings are checked as ``bpy_struct.__setattr__`` checks them."""

    _rna: FakeRNAProperties

    def __init__(self, rna: FakeRNAProperties, **values: Any) -> None:
        object.__setattr__(self, "_rna", rna)
        object.__setattr__(self, "bl_rna", SimpleNamespace(properties=rna))
        for prop in rna:
            object.__setattr__(self, prop.identifier, prop.default)
        for key, value in values.items():
            object.__setattr__(self, key, value)

    def __setattr__(self, key: str, value: Any) -> None:
        prop = self._rna.get(key)
        if prop is None:
            object.__setattr__(self, key, value)
            return
        name = f"{type(self).__name__}.{key}"
        if prop.is_readonly:
            raise AttributeError(f"bpy_struct: attribute \"{key}\" from \"{name}\" is read-only")
        if prop.type == "ENUM":
            items = [item.identifier for item in prop.enum_items]
            if prop.is_enum_flag:
                if not isinstance(value, set):
                    raise TypeError(f"bpy_struct: item.attr = val:, {name} expected a set, not a list")
                if not value <= set(items):
                    raise TypeError(f"bpy_struct: item.attr = val: enum {value} not found in {items}")
            elif value not in items:
                raise TypeError(f"bpy_struct: item.attr = val: enum \"{value}\" not found in {items}")
        elif prop.type == "POINTER":
            if value is self.__dict__.get("_owner") and value is not None:
                raise TypeError(
                    f"bpy_struct: item.attr = val: {name} ID type does not support assignment to itself"
                )
            if value is not None and isinstance(value, str):
                raise TypeError(
                    f"bpy_struct: item.attr = val: {name} expected a {prop.fixed_type.identifier} type, not str"
                )
            if value is not None and prop.poll is not None and not prop.poll(value):
                # Blender leaves a pointer its poll refuses unchanged, silently.
                return
        elif prop.type in ("INT", "FLOAT"):
            values = list(value) if prop.array_length else [value]
            if prop.type == "INT" and not all(isinstance(item, int) for item in values):
                raise TypeError(f"bpy_struct: item.attr = val: {name} expected an int type, not float")
            clamped = [min(max(item, prop.hard_min), prop.hard_max) for item in values]
            value = clamped if prop.array_length else clamped[0]
        object.__setattr__(self, key, value)


def _common_modifier_rna() -> List[FakeRNAProp]:
    return [
        FakeRNAProp("rna_type", "POINTER", readonly=True, fixed_type="Struct"),
        FakeRNAProp("name", "STRING"),
        FakeRNAProp("type", "ENUM", readonly=True, items=MODIFIER_TYPES),
        FakeRNAProp("show_viewport", "BOOLEAN", default=True),
        FakeRNAProp("show_render", "BOOLEAN", default=True),
        FakeRNAProp("show_in_editmode", "BOOLEAN", default=False),
    ]


def _is_mesh(obj: Any) -> bool:
    return getattr(obj, "type", None) == "MESH"


# The settings of the modifier types the tests use, as Blender 5.0.1 defines them.
MODIFIER_SETTINGS: Dict[str, List[FakeRNAProp]] = {
    "DECIMATE": [
        FakeRNAProp("decimate_type", "ENUM", items=("COLLAPSE", "UNSUBDIV", "DISSOLVE"), default="COLLAPSE"),
        FakeRNAProp("ratio", "FLOAT", hard=(0.0, 1.0), default=1.0),
        FakeRNAProp("iterations", "INT", hard=(0, 32767), default=0),
        FakeRNAProp("angle_limit", "FLOAT", hard=(0.0, 3.1415927410125732), default=0.0872665),
        FakeRNAProp(
            "delimit", "ENUM", items=("NORMAL", "MATERIAL", "SEAM", "SHARP", "UV"), flag=True, default={"NORMAL"}
        ),
        FakeRNAProp("use_collapse_triangulate", "BOOLEAN", default=False),
        FakeRNAProp("face_count", "INT", readonly=True, hard=(-INT_MAX - 1, INT_MAX), default=0),
    ],
    "BOOLEAN": [
        FakeRNAProp("object", "POINTER", fixed_type="Object", poll=_is_mesh),
        FakeRNAProp("collection", "POINTER", fixed_type="Collection"),
        FakeRNAProp("operation", "ENUM", items=("INTERSECT", "UNION", "DIFFERENCE"), default="DIFFERENCE"),
        FakeRNAProp("operand_type", "ENUM", items=("OBJECT", "COLLECTION"), default="OBJECT"),
        FakeRNAProp("solver", "ENUM", items=("FLOAT", "EXACT", "MANIFOLD"), default="EXACT"),
    ],
    "TRIANGULATE": [
        FakeRNAProp(
            "quad_method",
            "ENUM",
            items=("BEAUTY", "FIXED", "FIXED_ALTERNATE", "SHORTEST_DIAGONAL", "LONGEST_DIAGONAL"),
            default="SHORTEST_DIAGONAL",
        ),
        FakeRNAProp("ngon_method", "ENUM", items=("BEAUTY", "CLIP"), default="BEAUTY"),
        FakeRNAProp("min_vertices", "INT", hard=(4, INT_MAX), default=4),
    ],
    "WELD": [
        FakeRNAProp("mode", "ENUM", items=("ALL", "CONNECTED"), default="ALL"),
        FakeRNAProp("merge_threshold", "FLOAT", hard=(0.0, FLT_MAX), default=0.001),
    ],
    "SUBSURF": [
        FakeRNAProp("levels", "INT", hard=(0, 11), default=1),
        FakeRNAProp("render_levels", "INT", hard=(0, 11), default=2),
    ],
    "ARRAY": [
        FakeRNAProp("count", "INT", hard=(1, 1000), default=2),
        FakeRNAProp("relative_offset_displace", "FLOAT", hard=(-FLT_MAX, FLT_MAX), array_length=3,
                    default=[1.0, 0.0, 0.0]),
        FakeRNAProp("offset_object", "POINTER", fixed_type="Object"),
    ],
}

MODIFIER_TYPES: Tuple[str, ...] = (
    "ARRAY", "BEVEL", "BOOLEAN", "DECIMATE", "MIRROR", "SOLIDIFY", "SUBSURF", "TRIANGULATE", "WELD",
    "GREASE_PENCIL_TINT",
)

# Modifier types a mesh cannot take: modifiers.new returns None, as Blender does.
_NOT_FOR_MESHES = frozenset({"GREASE_PENCIL_TINT"})


class FakeModifier(FakeRNAStruct):
    """A modifier of one type with that type's RNA settings."""

    def __init__(self, owner: Any, name: str, modifier_type: str) -> None:
        rna = FakeRNAProperties(_common_modifier_rna() + MODIFIER_SETTINGS.get(modifier_type, []))
        super().__init__(rna)
        object.__setattr__(self, "_owner", owner)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "type", modifier_type)


def unique_name(name: str, taken: Sequence[str]) -> str:
    """The name Blender gives a new datablock or modifier: ``name``, else ``name.001``, ..."""
    if name not in taken:
        return name
    number = 1
    while f"{name}.{number:03d}" in taken:
        number += 1
    return f"{name}.{number:03d}"


class FakeModifierStack(list):
    """``Object.modifiers``: new, get by name, remove, in stack order."""

    def __init__(self, owner: Any) -> None:
        super().__init__()
        self.owner = owner
        self.removed: List[Any] = []

    def new(self, name: str, type: str) -> Optional[FakeModifier]:  # noqa: A002 - bpy's keyword
        if type not in MODIFIER_TYPES:
            raise TypeError(f'ObjectModifiers.new(): error with argument 2, "type" - enum "{type}" not found')
        if self.owner.type in ("EMPTY", "LIGHT", "CAMERA") or (
            self.owner.type == "MESH" and type in _NOT_FOR_MESHES
        ):
            return None
        modifier = FakeModifier(self.owner, unique_name(name, [m.name for m in self]), type)
        self.append(modifier)
        return modifier

    def get(self, name: str, default: Any = None) -> Any:
        return next((modifier for modifier in self if modifier.name == name), default)

    def remove(self, modifier: Any) -> None:  # type: ignore[override]
        super().remove(modifier)
        self.removed.append(modifier)


class FakePolygon(SimpleNamespace):
    """One face: its vertex indices, material index and select flag."""


class FakeVertex(tuple):
    """One vertex: its position, plus the select and hide flags Blender stores per vertex."""

    select = False
    hide = False


class FakeEdge(tuple):
    """One edge: its two vertex indices, plus its select and hide flags."""

    select = False
    hide = False


class FakeElements(list):
    """``Mesh.vertices`` / ``edges`` / ``polygons``: foreach_get and foreach_set by attribute."""

    def foreach_get(self, attribute: str, values: List[Any]) -> None:
        for index, element in enumerate(self):
            values[index] = getattr(element, attribute)

    def foreach_set(self, attribute: str, values: Sequence[Any]) -> None:
        assert len(values) == len(self), (attribute, len(values), len(self))
        for element, value in zip(self, values):
            setattr(element, attribute, value)


class FakeUVLayers(list):
    """``Mesh.uv_layers``: new, get, remove and the active layer (the first one made by default)."""

    def __init__(self) -> None:
        super().__init__()
        self.active: Any = None
        self.limit = 8

    def new(self, name: str = "UVMap") -> Any:
        if len(self) >= self.limit:
            return None
        layer = SimpleNamespace(
            name=unique_name(name, [uv.name for uv in self]), written_by=None, active_render=not self
        )
        self.append(layer)
        if self.active is None:
            self.active = layer
        return layer

    def get(self, name: str, default: Any = None) -> Any:
        return next((uv for uv in self if uv.name == name), default)

    def remove(self, layer: Any) -> None:  # type: ignore[override]
        super().remove(layer)
        if self.active is layer:
            self.active = self[0] if self else None


class FakeMesh:
    """``bpy.types.Mesh``: counts, validation, copies and users.

    ``invalid_faces`` lists face indices that ``validate()`` removes (faces
    that repeat a vertex or duplicate another); a copy carries them, so a
    validation probe on the copy sees what the original holds.
    """

    def __init__(
        self,
        registry: "FakeIDCollection",
        name: str,
        vertices: Sequence[Sequence[float]] = (),
        faces: Sequence[Sequence[int]] = (),
        edges: Optional[Sequence[Sequence[int]]] = None,
        invalid_faces: Sequence[int] = (),
    ) -> None:
        self._registry = registry
        self.name = name
        self.validate_calls: List[Dict[str, Any]] = []
        self.pydata: Optional[Tuple[Any, Any, Any]] = None
        self.uv_layers = FakeUVLayers()
        self.shape_keys = None
        self.extra_users = 0
        self.materials = FakeMaterialSlots()
        # Edit Mode through any object that uses the mesh; a library it is linked from.
        self.is_editmode = False
        self.library = None
        self._set_geometry(vertices, edges, faces)
        self.invalid_faces = list(invalid_faces)

    def _set_geometry(self, vertices: Any, edges: Any, faces: Any) -> None:
        self.vertices = FakeElements(FakeVertex(vertex) for vertex in vertices)
        face_edges = {tuple(sorted((f[i], f[(i + 1) % len(f)]))) for f in faces for i in range(len(f))}
        loose = {tuple(sorted(edge)) for edge in (edges or [])}
        self.edges = FakeElements(
            FakeEdge(edge) for edge in (sorted(face_edges | loose) if edges is not None or faces else [])
        )
        self.polygons = FakeElements(
            FakePolygon(vertices=tuple(face), material_index=0, index=i, select=False, hide=False)
            for i, face in enumerate(faces)
        )
        self.loops = [corner for face in faces for corner in face]

    @property
    def users(self) -> int:
        return sum(1 for obj in self._registry.bpy.data.objects if obj.data is self) + self.extra_users

    def from_pydata(self, vertices: Any, edges: Any, faces: Any, shade_flat: bool = True) -> None:
        self.pydata = (vertices, edges, faces)
        self._set_geometry(vertices, edges, faces)

    def update(self, **_unused: Any) -> None:
        return None

    def validate(self, verbose: bool = False, clean_customdata: bool = True) -> bool:
        self.validate_calls.append({"verbose": verbose, "clean_customdata": clean_customdata})
        if not self.invalid_faces:
            return False
        kept = [p.vertices for p in self.polygons if p.index not in self.invalid_faces]
        self._set_geometry(self.vertices, [], kept)
        self.invalid_faces = []
        return True

    def copy(self) -> "FakeMesh":
        """A local copy, as ``ID.copy()`` makes: geometry, face materials, slots and UV maps."""
        clone = self._registry.new(self.name)
        clone._set_geometry(self.vertices, [], [p.vertices for p in self.polygons])
        for source, copied in zip(self.polygons, clone.polygons):
            copied.material_index = source.material_index
        clone.materials.extend(self.materials)
        for layer in self.uv_layers:
            clone.uv_layers.new(layer.name)
        clone.invalid_faces = list(self.invalid_faces)
        return clone


class FakeMaterialSlots(list):
    """``Mesh.materials``: the mesh's material slots (None for an empty slot)."""

    def append(self, material: Any) -> None:  # type: ignore[override]
        super().append(material)

    def pop(self, index: int = -1) -> Any:  # type: ignore[override]
        return super().pop(index)


class FakeMaterialSlot:
    """``MaterialSlot``: its link (DATA or OBJECT) and the material that link selects."""

    def __init__(self, owner: "FakeMeshObject", index: int) -> None:
        self._owner = owner
        self._index = index

    @property
    def link(self) -> str:
        return self._owner.slot_links.get(self._index, "DATA")

    @link.setter
    def link(self, value: str) -> None:
        self._owner.slot_links[self._index] = value

    @property
    def material(self) -> Any:
        if self.link == "OBJECT":
            return self._owner.object_materials.get(self._index)
        return self._owner.data.materials[self._index]

    @material.setter
    def material(self, value: Any) -> None:
        if self.link == "OBJECT":
            self._owner.object_materials[self._index] = value
        else:
            self._owner.data.materials[self._index] = value


# -- Shader nodes ---------------------------------------------------------------

# bl_idname -> (type, default name, inputs as (name, default value), outputs).
NODE_SPECS: Dict[str, Tuple[str, str, List[Tuple[str, Any]], List[str]]] = {
    "ShaderNodeBsdfPrincipled": (
        "BSDF_PRINCIPLED",
        "Principled BSDF",
        [
            ("Base Color", (0.8, 0.8, 0.8, 1.0)),
            ("Metallic", 0.0),
            ("Roughness", 0.5),
            ("IOR", 1.5),
            ("Alpha", 1.0),
            ("Normal", (0.0, 0.0, 0.0)),
            ("Emission Color", (1.0, 1.0, 1.0, 1.0)),
            ("Emission Strength", 0.0),
        ],
        ["BSDF"],
    ),
    "ShaderNodeOutputMaterial": (
        "OUTPUT_MATERIAL",
        "Material Output",
        [("Surface", None), ("Volume", None), ("Displacement", (0.0, 0.0, 0.0))],
        [],
    ),
    "ShaderNodeTexImage": ("TEX_IMAGE", "Image Texture", [("Vector", (0.0, 0.0, 0.0))], ["Color", "Alpha"]),
    "ShaderNodeNormalMap": ("NORMAL_MAP", "Normal Map", [("Strength", 1.0), ("Color", (0.5, 0.5, 1.0, 1.0))],
                            ["Normal"]),
    "ShaderNodeMath": ("MATH", "Math", [("Value", 0.5), ("Value_001", 0.5)], ["Value"]),
    "NodeReroute": ("REROUTE", "Reroute", [("Input", None)], ["Output"]),
    "ShaderNodeEmission": ("EMISSION", "Emission", [("Color", (1.0, 1.0, 1.0, 1.0)), ("Strength", 1.0)],
                           ["Emission"]),
}


class FakeSocket:
    """``NodeSocket``: name, default value, and the links into or out of it."""

    def __init__(self, node: "FakeNode", name: str, default_value: Any = None) -> None:
        self.node = node
        self.name = name
        self.identifier = name
        self.default_value = default_value
        self.links: List["FakeLink"] = []

    @property
    def is_linked(self) -> bool:
        return bool(self.links)


class FakeSockets(list):
    """``Node.inputs`` / ``outputs``: by index or name, and ``get``."""

    def get(self, name: str, default: Any = None) -> Any:
        return next((socket for socket in self if socket.name == name), default)

    def __getitem__(self, key: Any) -> Any:  # type: ignore[override]
        if isinstance(key, str):
            socket = self.get(key)
            if socket is None:
                raise KeyError(key)
            return socket
        return super().__getitem__(key)


class FakeLink:
    """``NodeLink`` from an output socket to an input socket."""

    def __init__(self, from_socket: FakeSocket, to_socket: FakeSocket) -> None:
        self.from_socket = from_socket
        self.to_socket = to_socket
        self.from_node = from_socket.node
        self.to_node = to_socket.node
        self.is_muted = False
        self.is_valid = True


class FakeNode:
    """``ShaderNode`` of one ``NODE_SPECS`` type."""

    def __init__(self, bl_idname: str, name: str) -> None:
        kind, _default_name, inputs, outputs = NODE_SPECS[bl_idname]
        self.bl_idname = bl_idname
        self.type = kind
        self.name = name
        self.label = ""
        self.location: Tuple[float, float] = (0.0, 0.0)
        self.image: Any = None
        self.is_active_output = kind == "OUTPUT_MATERIAL"
        self.inputs = FakeSockets(FakeSocket(self, n, v) for n, v in inputs)
        self.outputs = FakeSockets(FakeSocket(self, n) for n in outputs)


class FakeNodeLinks(list):
    """``NodeTree.links``: new replaces a link already into that input, as Blender does."""

    def new(self, from_socket: FakeSocket, to_socket: FakeSocket) -> FakeLink:
        for old in list(to_socket.links):
            self.remove(old)
        link = FakeLink(from_socket, to_socket)
        self.append(link)
        from_socket.links.append(link)
        to_socket.links.append(link)
        return link

    def remove(self, link: FakeLink) -> None:  # type: ignore[override]
        super().remove(link)
        link.from_socket.links.remove(link)
        link.to_socket.links.remove(link)


class FakeNodes(list):
    """``NodeTree.nodes``: new by bl_idname (Blender naming), get, remove (with its links)."""

    def __init__(self, tree: "FakeNodeTree") -> None:
        super().__init__()
        self.tree = tree
        self.removed: List[FakeNode] = []

    def new(self, bl_idname: str) -> FakeNode:
        node = FakeNode(bl_idname, unique_name(NODE_SPECS[bl_idname][1], [n.name for n in self]))
        self.append(node)
        return node

    def get(self, name: str, default: Any = None) -> Any:
        return next((node for node in self if node.name == name), default)

    def remove(self, node: FakeNode) -> None:  # type: ignore[override]
        for socket in [*node.inputs, *node.outputs]:
            for link in list(socket.links):
                self.tree.links.remove(link)
        super().remove(node)
        self.removed.append(node)


class FakeNodeTree:
    """``ShaderNodeTree``; ``default()`` is a new material's: Principled BSDF into Material Output."""

    def __init__(self) -> None:
        self.links = FakeNodeLinks()
        self.nodes = FakeNodes(self)

    @classmethod
    def default(cls) -> "FakeNodeTree":
        tree = cls()
        bsdf = tree.nodes.new("ShaderNodeBsdfPrincipled")
        output = tree.nodes.new("ShaderNodeOutputMaterial")
        tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
        return tree


class FakeMaterial:
    """``bpy.types.Material``.

    On Blender 5.0 a new material has its node tree, and reading or writing
    ``use_nodes`` is deprecated (counted in ``use_nodes_access``). Before 5.0
    the tree appears only once ``use_nodes`` is set.
    """

    def __init__(self, registry: "FakeIDCollection", name: str) -> None:
        self._registry = registry
        self.name = name
        self.library = None
        self.diffuse_color = (0.8, 0.8, 0.8, 1.0)
        self.surface_render_method = "DITHERED"
        self.blend_method = "OPAQUE"
        modern = registry.bpy.app.version >= (5, 0, 0)
        self.node_tree: Optional[FakeNodeTree] = FakeNodeTree.default() if modern else None
        self._use_nodes = modern
        self.use_nodes_access = 0

    @property
    def use_nodes(self) -> bool:
        self.use_nodes_access += 1
        return self._use_nodes

    @use_nodes.setter
    def use_nodes(self, value: bool) -> None:
        self.use_nodes_access += 1
        self._use_nodes = value
        if value and self.node_tree is None:
            self.node_tree = FakeNodeTree.default()

    @property
    def principled(self) -> FakeNode:
        assert self.node_tree is not None
        return next(node for node in self.node_tree.nodes if node.type == "BSDF_PRINCIPLED")

    @property
    def users(self) -> int:
        """Mesh slots and OBJECT-linked object slots holding the material."""
        bpy = self._registry.bpy
        data_users = sum(list(mesh.materials).count(self) for mesh in bpy.data.meshes)
        object_users = sum(
            1 for obj in bpy.data.objects for material in getattr(obj, "object_materials", {}).values()
            if material is self
        )
        return data_users + object_users


class FakeImage:
    """``bpy.types.Image`` loaded from a file: 4 x 4 pixels, or 0 x 0 when the file is no image."""

    def __init__(self, registry: "FakeIDCollection", name: str, filepath: str = "", readable: bool = True) -> None:
        self.name = name
        self.filepath = filepath
        # Blender reads an 8-bit file as sRGB and a float one (EXR) as linear.
        self.is_float = filepath.lower().endswith(".exr")
        self.colorspace_settings = FakeColorspace("Linear Rec.709" if self.is_float else "sRGB")
        self.packed_file = None
        self.size = (4, 4) if readable else (0, 0)


class FakeColorspace:
    """``ColorManagedInputColorspaceSettings``: the name, and is_data for Non-Color."""

    def __init__(self, name: str) -> None:
        self.name = name

    @property
    def is_data(self) -> bool:
        return self.name == "Non-Color"


# -- Lights ---------------------------------------------------------------------

LIGHT_TYPE_ATTRIBUTES = {
    "POINT": ("shadow_soft_size",),
    "SPOT": ("shadow_soft_size", "spot_size", "spot_blend"),
    "SUN": ("angle",),
    "AREA": ("shape", "size", "size_y"),
}
_LIGHT_COMMON = ("name", "type", "energy", "color", "use_shadow", "users")
_LIGHT_CLAMPS = {
    "spot_size": (0.01745329238474369, 3.1415927410125732),
    "spot_blend": (0.0, 1.0),
    "angle": (0.0, 3.1415927410125732),
    "shadow_soft_size": (0.0, FLT_MAX),
    "size": (0.0, FLT_MAX),
    "size_y": (0.0, FLT_MAX),
}


class FakeLightData:
    """A light datablock's values, whatever its type; ``FakeLightView`` is its typed struct."""

    def __init__(self, name: str, light_type: str) -> None:
        self.values: Dict[str, Any] = {
            "name": name,
            "type": light_type,
            "energy": 10.0,
            "color": [1.0, 1.0, 1.0],
            "use_shadow": True,
            "shadow_soft_size": 0.25,
            "spot_size": 0.785398,
            "spot_blend": 0.15,
            "angle": 0.00918043,
            "shape": "SQUARE",
            "size": 0.25,
            "size_y": 0.25,
            "users": 1,
        }


class FakeLightView:
    """``PointLight`` / ``SpotLight`` / ``SunLight`` / ``AreaLight``.

    Like Blender's Python struct it keeps the type it was fetched as: after
    ``type`` changes, only a struct fetched again (``obj.data``) has the new
    type's settings. Numbers are clamped to Blender's hard limits.
    """

    def __init__(self, store: FakeLightData) -> None:
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_kind", store.values["type"])

    def _allowed(self, name: str) -> bool:
        return name in _LIGHT_COMMON or name in LIGHT_TYPE_ATTRIBUTES[self._kind]

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or not self._allowed(name):
            raise AttributeError(f"'{self._kind.title()}Light' object has no attribute '{name}'")
        return self._store.values[name]

    def __setattr__(self, name: str, value: Any) -> None:
        if not self._allowed(name):
            raise AttributeError(f"'{self._kind.title()}Light' object has no attribute '{name}'")
        if name in _LIGHT_CLAMPS:
            low, high = _LIGHT_CLAMPS[name]
            value = min(max(value, low), high)
        if name == "color":
            value = [min(max(channel, 0.0), FLT_MAX) for channel in value]
        self._store.values[name] = value


class FakeLightObject:
    """A light object: ``data`` returns a freshly typed struct each time, as ``obj.data`` does."""

    def __init__(self, name: str, store: FakeLightData) -> None:
        self.name = name
        self.type = "LIGHT"
        self.store = store
        self.modifiers = FakeModifierStack(self)

    @property
    def data(self) -> FakeLightView:
        return FakeLightView(self.store)


class FakeMeshObject:
    """``bpy.types.Object`` for the mesh tools: type, data, mode, modifiers and its evaluated copy.

    ``evaluated_mesh`` is what the depsgraph makes of the object; None means
    the depsgraph holds no evaluated copy (the object is not in the view layer).
    """

    def __init__(self, name: str, object_type: str = "MESH", data: Any = None, mode: str = "OBJECT") -> None:
        self.name = name
        self.type = object_type
        self.data = data
        self.mode = mode
        self.location: Tuple[float, ...] = (0.0, 0.0, 0.0)
        self.modifiers = FakeModifierStack(self)
        self.evaluated_mesh: Any = data
        self.edit_mesh: Any = None
        self.editmode_updates = 0
        self.cleared_evaluated = 0
        self.hidden = False
        self.library = None
        self._selected = False
        # Per slot: its link, and the material an OBJECT-linked slot holds.
        self.slot_links: Dict[int, str] = {}
        self.object_materials: Dict[int, Any] = {}

    @property
    def material_slots(self) -> List["FakeMaterialSlot"]:
        """One slot per entry of the data's material list, as Blender derives them."""
        materials = getattr(self.data, "materials", None)
        return [] if materials is None else [FakeMaterialSlot(self, i) for i in range(len(materials))]

    def select_get(self) -> bool:
        return self._selected

    def select_set(self, state: bool) -> None:
        self._selected = bool(state)

    def visible_get(self) -> bool:
        return not self.hidden

    def evaluated_get(self, depsgraph: Any) -> Any:
        owner = self
        evaluated = self.evaluated_mesh

        class Evaluated:
            is_evaluated = evaluated is not None

            @staticmethod
            def to_mesh() -> Any:
                return evaluated if evaluated is not None else FakeMesh(owner.data._registry, "empty")

            @staticmethod
            def to_mesh_clear() -> None:
                owner.cleared_evaluated += 1

        return Evaluated()

    def update_from_editmode(self) -> bool:
        """Write the edit mesh (``edit_mesh``: vertices, edges, faces) back to the mesh data."""
        self.editmode_updates += 1
        if self.edit_mesh is not None:
            self.data._set_geometry(*self.edit_mesh)
        return True


class FakeIDCollection(list):
    """A ``bpy.data`` collection: new (Blender naming), get, remove, and the removed datablocks."""

    def __init__(self, bpy: "FakeBpy", factory: Callable[..., Any]) -> None:
        super().__init__()
        self.bpy = bpy
        self.factory = factory
        self.removed: List[Any] = []

    def new(self, name: str, *args: Any, **kwargs: Any) -> Any:
        datablock = self.factory(self, unique_name(name, [item.name for item in self]), *args, **kwargs)
        self.append(datablock)
        return datablock

    def get(self, name: str, default: Any = None) -> Any:
        return next((item for item in self if item.name == name), default)

    def keys(self) -> List[str]:
        return [item.name for item in self]

    def remove(self, datablock: Any, do_unlink: bool = True) -> None:  # type: ignore[override]
        super().remove(datablock)
        self.removed.append(datablock)
        if hasattr(datablock, "users_collection"):
            for collection in datablock.users_collection:
                collection.objects.unlink(datablock)


class FakeImages(FakeIDCollection):
    """``bpy.data.images`` with ``load``: like Blender, a file that is no image loads but reads as 0 x 0."""

    def load(self, filepath: str, check_existing: bool = False) -> FakeImage:
        import os

        if check_existing:
            known = next((image for image in self if image.filepath == filepath), None)
            if known is not None:
                return known
        if not os.path.exists(filepath):
            raise RuntimeError(f"Error: Cannot read '{filepath}': No such file or directory\n")
        with open(filepath, "rb") as handle:
            readable = handle.read(4) == b"\x89PNG"
        image = FakeImage(self, unique_name(os.path.basename(filepath), self.keys()), filepath, readable)
        self.append(image)
        return image


class FakeCollectionObjects(list):
    """``Collection.objects``: link and unlink."""

    def __init__(self, collection: "FakeCollectionData", fail: Optional[str] = None) -> None:
        super().__init__()
        self.collection = collection
        self.fail = fail

    def link(self, obj: Any) -> None:
        if self.fail:
            raise RuntimeError(self.fail)
        self.append(obj)
        obj.users_collection = [*getattr(obj, "users_collection", []), self.collection]

    def unlink(self, obj: Any) -> None:
        if obj in self:
            self.remove(obj)


class FakeCollectionData:
    """``bpy.types.Collection``: its objects and child collections."""

    def __init__(self, name: str, *, library: Any = None, fail_link: Optional[str] = None) -> None:
        self.name = name
        self.library = library
        self.children: List["FakeCollectionData"] = []
        self.objects = FakeCollectionObjects(self, fail_link)

    @property
    def children_recursive(self) -> List["FakeCollectionData"]:
        found: List["FakeCollectionData"] = []
        for child in self.children:
            found.append(child)
            found.extend(child.children_recursive)
        return found


class FakeLayerCollection:
    """``LayerCollection``: a collection's exclusion in the view layer, and its children."""

    def __init__(self, collection: FakeCollectionData, exclude: bool = False) -> None:
        self.collection = collection
        self.exclude = exclude
        self.children = [FakeLayerCollection(child) for child in collection.children]

    def find(self, collection: Any) -> Optional["FakeLayerCollection"]:
        if self.collection is collection:
            return self
        for child in self.children:
            found = child.find(collection)
            if found is not None:
                return found
        return None


class FakeBpy:
    """The parts of ``bpy`` the mesh, modifier, material and light tools use.

    Attributes:
        data: ``objects``, ``meshes``, ``collections``, ``materials``,
            ``images`` and ``lights`` registries.
        context: ``scene`` (with its scene collection), ``view_layer``,
            ``object`` (set inside ``temp_override``) and the depsgraph.
        ops: ``object.modifier_apply``, recording its calls in ``applied``.
    """

    def __init__(self, version: Tuple[int, int, int] = (5, 0, 1)) -> None:
        self.app = SimpleNamespace(version=version, version_string=".".join(map(str, version)))
        self.data = SimpleNamespace()
        self.data.objects = FakeIDCollection(self, lambda registry, name, data=None: FakeMeshObject(
            name, "MESH" if isinstance(data, FakeMesh) else "EMPTY", data))
        self.data.meshes = FakeIDCollection(self, FakeMesh)
        self.data.collections = FakeIDCollection(self, lambda registry, name: FakeCollectionData(name))
        self.data.materials = FakeIDCollection(self, FakeMaterial)
        self.data.images = FakeImages(self, FakeImage)
        self.data.lights = FakeIDCollection(self, lambda registry, name: SimpleNamespace(name=name))
        scene_collection = FakeCollectionData("Scene Collection")
        self.context = SimpleNamespace(
            scene=SimpleNamespace(name="Scene", collection=scene_collection, objects=self.data.objects),
            view_layer=SimpleNamespace(
                name="ViewLayer",
                layer_collection=FakeLayerCollection(scene_collection),
                objects=FakeViewLayerObjects(self.data.objects),
            ),
            mode="OBJECT",
            object=None,
            active_object=None,
            evaluated_depsgraph_get=lambda: "depsgraph",
            temp_override=self._temp_override,
        )
        self.path = SimpleNamespace(abspath=lambda path: path)
        self.applied: List[Dict[str, Any]] = []
        self.apply_result = {"FINISHED"}
        # Edit Mode operator calls, in order: (operator, keyword arguments).
        self.edit_ops: List[Tuple[str, Dict[str, Any]]] = []
        self.unwrap_error: Optional[str] = None
        self.ops = SimpleNamespace(
            object=SimpleNamespace(modifier_apply=self._modifier_apply, mode_set=self._mode_set),
            mesh=SimpleNamespace(select_all=self._select_all, reveal=self._reveal),
            uv=SimpleNamespace(smart_project=self._unwrap("smart_project"), cube_project=self._unwrap("cube_project")),
        )
        self.types = SimpleNamespace(
            Modifier=SimpleNamespace(
                bl_rna=SimpleNamespace(
                    properties=FakeRNAProperties([FakeRNAProp("type", "ENUM", items=MODIFIER_TYPES)])
                )
            )
        )

    @contextmanager
    def _temp_override(self, **members: Any) -> Iterator[None]:
        saved = {key: getattr(self.context, key, None) for key in members}
        for key, value in members.items():
            setattr(self.context, key, value)
        try:
            yield
        finally:
            for key, value in saved.items():
                setattr(self.context, key, value)

    def _modifier_apply(self, modifier: str, single_user: bool = False, **_unused: Any) -> set:
        obj = self.context.object
        self.applied.append({"object": getattr(obj, "name", None), "modifier": modifier, "single_user": single_user})
        if obj is None or self.apply_result != {"FINISHED"}:
            return set(self.apply_result) if obj is not None else {"CANCELLED"}
        if obj.data.users > 1:
            if not single_user:
                raise RuntimeError("Error: Modifiers cannot be applied to multi-user data\n")
            clone = obj.data.copy()
            obj.data = clone
        target = obj.modifiers.get(modifier)
        if target.type == "DECIMATE":
            faces = [p.vertices for p in obj.data.polygons]
            obj.data._set_geometry(obj.data.vertices, [], faces[: max(1, int(len(faces) * target.ratio))])
        obj.modifiers.remove(target)
        return {"FINISHED"}

    def _mode_set(self, mode: str) -> set:
        """``object.mode_set``: the view layer's active object enters or leaves the mode."""
        active = self.context.view_layer.objects.active
        self.edit_ops.append(("mode_set", {"mode": mode, "object": getattr(active, "name", None)}))
        if active is None:
            raise RuntimeError("Error: Operator bpy.ops.object.mode_set.poll() failed, context is incorrect")
        if active.mode == "EDIT" and mode == "OBJECT":
            # Leaving Edit Mode rebuilds the mesh's layers: a UV layer struct
            # fetched before reads as freed memory (its name "" on Blender 5.0.1).
            layers = active.data.uv_layers
            rebuilt = [SimpleNamespace(**vars(old)) for old in layers]
            active_index = next((i for i, old in enumerate(layers) if old is layers.active), None)
            for old in layers:
                old.name = ""
            layers[:] = rebuilt
            layers.active = None if active_index is None else rebuilt[active_index]
        active.mode = mode
        self.context.mode = "EDIT_MESH" if mode == "EDIT" else "OBJECT"
        return {"FINISHED"}

    def _select_all(self, action: str) -> set:
        active = self.context.view_layer.objects.active
        self.edit_ops.append(("select_all", {"action": action}))
        for domain in ("vertices", "edges", "polygons"):
            for element in getattr(active.data, domain):
                # Like Blender, selecting all leaves hidden elements alone.
                if not element.hide:
                    element.select = action == "SELECT"
        return {"FINISHED"}

    def _reveal(self, select: bool = True) -> set:
        """``mesh.reveal``: show hidden elements, selecting them when asked."""
        active = self.context.view_layer.objects.active
        self.edit_ops.append(("reveal", {"select": select}))
        for domain in ("vertices", "edges", "polygons"):
            for element in getattr(active.data, domain):
                if element.hide:
                    element.hide = False
                    element.select = select
        return {"FINISHED"}

    def _unwrap(self, operator: str) -> Callable[..., set]:
        def run(**options: Any) -> set:
            active = self.context.view_layer.objects.active
            self.edit_ops.append((operator, dict(options)))
            if active.mode != "EDIT":
                raise RuntimeError(f"Error: Operator bpy.ops.uv.{operator}.poll() failed, context is incorrect")
            if self.unwrap_error:
                raise RuntimeError(self.unwrap_error)
            active.data.uv_layers.active.written_by = (operator, dict(options))
            # The operators unwrap the selected faces that are not hidden.
            active.data.uv_layers.active.faces_unwrapped = sum(
                1 for face in active.data.polygons if face.select and not face.hide
            )
            return {"FINISHED"}

        return run

    # -- builders -------------------------------------------------------------

    def add_collection(self, name: str, *, in_scene: bool = True, **kwargs: Any) -> FakeCollectionData:
        """A collection in bpy.data, linked under the scene collection unless ``in_scene`` is False."""
        collection = FakeCollectionData(name, **kwargs)
        self.data.collections.append(collection)
        if in_scene:
            self.context.scene.collection.children.append(collection)
            self.context.view_layer.layer_collection = FakeLayerCollection(self.context.scene.collection)
        return collection

    def add_mesh_object(
        self,
        name: str,
        vertices: Sequence[Sequence[float]] = (),
        faces: Sequence[Sequence[int]] = (),
        **mesh: Any,
    ) -> FakeMeshObject:
        """A mesh object linked to the scene collection."""
        data = self.data.meshes.new(name, vertices, faces, **mesh)
        obj = self.data.objects.new(name, data)
        self.context.scene.collection.objects.link(obj)
        return obj

    def add_object(self, name: str, object_type: str, data: Any = None) -> FakeMeshObject:
        """A non-mesh object (EMPTY, LIGHT, CAMERA, ...) linked to the scene collection."""
        obj = FakeMeshObject(name, object_type, data)
        self.data.objects.append(obj)
        self.context.scene.collection.objects.link(obj)
        return obj

    def add_light(self, name: str, light_type: str, store: Optional[FakeLightData] = None) -> FakeLightObject:
        """A light object of that type (sharing ``store`` when given) linked to the scene collection."""
        if store is None:
            store = FakeLightData(name, light_type)
        else:
            store.values["users"] += 1
        obj = FakeLightObject(name, store)
        self.data.objects.append(obj)
        self.context.scene.collection.objects.link(obj)
        return obj


class FakeViewLayerObjects:
    """``ViewLayer.objects``: the objects in the view layer and its active object."""

    def __init__(self, objects: FakeIDCollection) -> None:
        self._objects = objects
        self.active: Any = None

    def __iter__(self) -> Iterator[Any]:
        return iter(self._objects)


# A unit cube: 8 vertices, 6 quads.
CUBE_VERTICES = [
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
]
CUBE_FACES = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
