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
    """One face: its vertex indices and material index."""


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
        self.uv_layers: List[Any] = []
        self.shape_keys = None
        self.extra_users = 0
        self.materials = FakeMaterialSlots()
        self._set_geometry(vertices, edges, faces)
        self.invalid_faces = list(invalid_faces)

    def _set_geometry(self, vertices: Any, edges: Any, faces: Any) -> None:
        self.vertices = [tuple(vertex) for vertex in vertices]
        face_edges = {tuple(sorted((f[i], f[(i + 1) % len(f)]))) for f in faces for i in range(len(f))}
        loose = {tuple(sorted(edge)) for edge in (edges or [])}
        self.edges = sorted(face_edges | loose) if edges is not None or faces else []
        self.polygons = [FakePolygon(vertices=tuple(face), material_index=0, index=i) for i, face in enumerate(faces)]
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
        clone = self._registry.new(self.name)
        clone._set_geometry(self.vertices, [], [p.vertices for p in self.polygons])
        clone.invalid_faces = list(self.invalid_faces)
        return clone


class FakeMaterialSlots(list):
    """``Mesh.materials``: the mesh's material slots (None for an empty slot)."""

    def append(self, material: Any) -> None:  # type: ignore[override]
        super().append(material)

    def pop(self, index: int = -1) -> Any:  # type: ignore[override]
        return super().pop(index)


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
        self.material_slots: List[Any] = []

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
        self.data.materials = FakeIDCollection(self, lambda registry, name: SimpleNamespace(name=name))
        self.data.images = FakeIDCollection(self, lambda registry, name: SimpleNamespace(name=name))
        self.data.lights = FakeIDCollection(self, lambda registry, name: SimpleNamespace(name=name))
        scene_collection = FakeCollectionData("Scene Collection")
        self.context = SimpleNamespace(
            scene=SimpleNamespace(name="Scene", collection=scene_collection, objects=self.data.objects),
            view_layer=SimpleNamespace(name="ViewLayer", layer_collection=FakeLayerCollection(scene_collection)),
            object=None,
            active_object=None,
            evaluated_depsgraph_get=lambda: "depsgraph",
            temp_override=self._temp_override,
        )
        self.applied: List[Dict[str, Any]] = []
        self.apply_result = {"FINISHED"}
        self.ops = SimpleNamespace(object=SimpleNamespace(modifier_apply=self._modifier_apply))
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


# A unit cube: 8 vertices, 6 quads.
CUBE_VERTICES = [
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
]
CUBE_FACES = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
