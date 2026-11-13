"""A fake ``unreal`` module with enough GeometryScript to run the mesh tool scripts.

The mesh tools (``simul.adapters._unreal_mesh_scripts``) run fixed Python in
the editor. These tests run the same scripts against this fake, which keeps a
small level (DynamicMeshActors, a StaticMeshActor, a StaticMesh asset), fakes
the GeometryScript libraries over meshes described by their counts and bounds,
and records what a script did: every GeometryScript call with its arguments
(and whether it got a ``GeometryScriptDebug``), every transaction, every
spawn. The numbers follow the rules below rather than real geometry, which is
enough to check what a script reports against what it did.

Shapes the fake reproduces from UE 5.7.4:

- UFUNCTIONs with out-parameters return ``(value, *outs)`` tuples
  (``get_num_open_border_loops``, ``get_num_uv_islands``, ...).
- GeometryScript reports a failure in the ``debug`` object it was given and
  returns normally (``fail`` names the functions that fail).
- ``set_dynamic_mesh`` gives the new mesh to the component; the old one is
  left as it was. Setting the mesh the component already holds renames it
  and dirties the level (``Rename`` without ``REN_DoNotDirty``), the way a
  component is bound to its mesh again after an Undo.
- A transaction dirties the level; ``update_collision`` dirties nothing.
- ``EditorAssetLibrary`` finds and loads nothing while PIE runs; the asset
  registry still answers.
- ``unreal.Rotator`` takes (roll, pitch, yaw) positionally; the fake takes
  keywords only.
"""

from __future__ import annotations

import copy
import io
import math
import sys
from contextlib import redirect_stdout
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

from tests.unreal.editor_world import World, is_valid

LEVEL = "/Game/Maps/Mesh.Mesh:PersistentLevel."

#: The package of the fake level.
LEVEL_PACKAGE = "/Game/Maps/Mesh"


class Vector:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.x, self.y, self.z = float(x), float(y), float(z)

    def __iter__(self):
        return iter((self.x, self.y, self.z))

    def __repr__(self) -> str:
        return f"Vector({self.x}, {self.y}, {self.z})"


class Vector2D:
    def __init__(self, x: float = 0.0, y: float = 0.0) -> None:
        self.x, self.y = float(x), float(y)


class Rotator:
    """Yaw-only rotations are enough for the tests."""

    def __init__(self, *, pitch: float = 0.0, yaw: float = 0.0, roll: float = 0.0) -> None:
        self.pitch, self.yaw, self.roll = pitch, yaw, roll

    def quaternion(self) -> "Quat":
        return Quat(self)


class Quat:
    def __init__(self, rotator: Optional[Rotator] = None) -> None:
        self._rotator = rotator or Rotator()

    def rotator(self) -> Rotator:
        return self._rotator

    def rotate_vector(self, v: Vector) -> Vector:
        a = math.radians(self._rotator.yaw)
        return Vector(v.x * math.cos(a) - v.y * math.sin(a), v.x * math.sin(a) + v.y * math.cos(a), v.z)

    def unrotate_vector(self, v: Vector) -> Vector:
        a = -math.radians(self._rotator.yaw)
        return Vector(v.x * math.cos(a) - v.y * math.sin(a), v.x * math.sin(a) + v.y * math.cos(a), v.z)


class Transform:
    def __init__(self) -> None:
        self.translation = Vector()
        self.rotation = Quat()
        self.scale3d = Vector(1, 1, 1)

    def inverse_transform_location(self, v: Vector) -> Vector:
        moved = self.rotation.unrotate_vector(Vector(v.x - self.translation.x, v.y - self.translation.y,
                                                     v.z - self.translation.z))
        return Vector(moved.x / self.scale3d.x, moved.y / self.scale3d.y, moved.z / self.scale3d.z)


def transform(location: Tuple[float, float, float] = (0, 0, 0), yaw: float = 0.0,
              scale: Tuple[float, float, float] = (1, 1, 1)) -> Transform:
    xf = Transform()
    xf.translation = Vector(*location)
    xf.rotation = Quat(Rotator(yaw=yaw))
    xf.scale3d = Vector(*scale)
    return xf


class Box:
    def __init__(self, lo: Tuple[float, ...] = (0, 0, 0), hi: Tuple[float, ...] = (0, 0, 0)) -> None:
        self.min, self.max = Vector(*lo), Vector(*hi)
        self.is_valid = False


class Box2D:
    def __init__(self, lo: Tuple[float, float], hi: Tuple[float, float], valid: bool = True) -> None:
        self.min, self.max = Vector2D(*lo), Vector2D(*hi)
        self.is_valid = valid


class Options:
    """Every GeometryScript options struct: attributes the script sets, over their defaults."""

    defaults: Dict[str, Any] = {}

    def __init__(self, **values: Any) -> None:
        self.__dict__.update(copy.deepcopy(type(self).defaults))
        self.__dict__.update(values)


def options_struct(name: str, **defaults: Any) -> type:
    return type(name, (Options,), {"defaults": defaults})


#: The GeometryScript option structs the scripts build, without their
#: ``GeometryScript`` prefix, and the defaults the scripts read.
OPTION_STRUCTS: Dict[str, Dict[str, Any]] = {
    "PrimitiveOptions": {},
    "MeshBooleanOptions": {},
    "MeshSelection": {},
    "MeshLinearExtrudeOptions": {},
    "MeshBevelSelectionOptions": {},
    "MeshInsetOutsetFacesOptions": {},
    "GroupLayer": {},
    "SimplifyMeshOptions": {},
    "MeshPlaneCutOptions": {},
    "RemeshOptions": {},
    "UniformRemeshOptions": {},
    "XAtlasOptions": {},
    "PatchBuilderOptions": {"auto_pack": True, "packing_options": None},
    "RepackUVsOptions": {"target_image_width": 512, "optimize_island_rotation": True},
    "CopyMeshFromAssetOptions": {},
    "MeshReadLOD": {},
    "CollisionFromMeshOptions": {"auto_detect_spheres": True, "auto_detect_boxes": True,
                                 "auto_detect_capsules": True, "max_convex_hulls_per_mesh": 1,
                                 "convex_hull_target_face_count": 25, "simplify_hulls": True},
    "SetSimpleCollisionOptions": {"emit_transaction": True},
    "SimpleCollisionTriangulationOptions": {},
    "RevolveOptions": {},
    "DegenerateTriangleOptions": {"mode": "REPAIR_OR_DELETE", "min_triangle_area": 0.001,
                                  "min_edge_length": 0.0001},
    "SetStaticMeshCollisionOptions": {"mark_as_customized": True},
    "CreateNewStaticMeshAssetOptions": {"enable_collision": True, "enable_nanite": False,
                                        "nanite_settings": SimpleNamespace(enabled=False)},
    "CopyMeshToAssetOptions": {"apply_nanite_settings": False, "new_nanite_settings": None},
    "MeshWriteLOD": {},
}


class Enum:
    """A GeometryScript enum: members are their names."""

    def __init__(self, *names: str) -> None:
        for name in names:
            setattr(self, name, name)


def mesh_stats(**overrides: Any) -> Dict[str, Any]:
    """The description of a fake mesh: a 100 cm closed box unless overridden."""
    stats = {
        "tris": 12,
        "verts": 8,
        "lo": (-50.0, -50.0, -50.0),
        "hi": (50.0, 50.0, 50.0),
        "area": 60000.0,
        "volume": 1000000.0,
        "open_edges": 0,
        "open_loops": 0,
        "components": 1,
        "normals": True,
        "gaps": False,
        "vertex_gaps": False,
        "bowties": 0,
        "degenerate": 0,
        "uv_sets": 1,
        "uv_islands": 6,
        "uv_area": 0.6,
        "uv_lo": (0.0, 0.0),
        "uv_hi": (1.0, 1.0),
        "unset_uvs": False,
        "quad_groups": True,
        # A sample of the mesh's geometry for the edge-length average:
        # (positions, triangles).
        "sample": ([(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 10.0, 0.0)], [(0, 1, 2)]),
    }
    stats.update(overrides)
    return stats


class DynamicMesh:
    """``unreal.DynamicMesh``: its stats are what the queries report."""

    count = 0

    def __init__(self, **stats: Any) -> None:
        DynamicMesh.count += 1
        self.name = f"DynamicMesh_{DynamicMesh.count}"
        self.outer = "/Engine/Transient"
        self.stats = mesh_stats(tris=0, verts=0, area=0.0, volume=0.0, uv_islands=0, uv_area=0.0) if not stats \
            else mesh_stats(**stats)

    def get_triangle_count(self) -> int:
        return self.stats["tris"]

    def get_name(self) -> str:
        return self.name

    def get_path_name(self) -> str:
        return self.outer + "." + self.name

    def modify(self) -> bool:
        return True


class SimpleCollision:
    def __init__(self, shapes: int = 0) -> None:
        self.shapes = shapes


class Selection:
    """A mesh selection: what selected it and how many elements."""

    def __init__(self, kind: str, count: int, how: Any = None) -> None:
        self.kind, self.count, self.how = kind, count, how


class GeometryScriptDebug:
    def __init__(self) -> None:
        self.messages: List[SimpleNamespace] = []


ERROR, WARNING = "ERROR_MESSAGE", "WARNING_MESSAGE"


class Component:
    def __init__(self, owner: "Actor") -> None:
        self.owner = owner

    def get_path_name(self) -> str:
        return self.owner.path + "." + type(self).__name__

    def modify(self) -> bool:
        self.owner.editor.modified.append(self.get_path_name())
        return True


class DynamicMeshComponent(Component):
    def __init__(self, owner: "Actor", mesh: DynamicMesh) -> None:
        super().__init__(owner)
        self.mesh = mesh
        self.collision = SimpleCollision()
        self.materials: List[Any] = [None]

    def get_materials(self) -> List[Any]:
        return list(self.materials)

    def get_owner(self) -> "Actor":
        return self.owner

    def get_dynamic_mesh(self) -> DynamicMesh:
        return self.mesh

    def set_dynamic_mesh(self, mesh: DynamicMesh) -> None:
        if mesh is self.mesh:
            DynamicMesh.count += 1
            mesh.name = f"DynamicMesh_{DynamicMesh.count}"
            self.owner.editor.dirty.add(LEVEL_PACKAGE)
            self.owner.editor.record("rebind", self.owner.path, mesh.name)
            return
        mesh.outer = self.get_path_name()
        self.mesh = mesh
        self.owner.editor.record("set_dynamic_mesh", self.owner.path, mesh.name)

    def update_collision(self, only_if_pending: bool = True) -> None:
        self.owner.editor.record("update_collision", self.owner.path, only_if_pending)


class StaticMeshComponent(Component):
    def __init__(self, owner: "Actor", static_mesh: Any) -> None:
        super().__init__(owner)
        self.static_mesh = static_mesh

    def get_editor_property(self, name: str) -> Any:
        return getattr(self, name)

    def set_static_mesh(self, mesh: Any) -> bool:
        self.static_mesh = mesh
        return True


class Actor:
    cls_name = "Actor"

    def __init__(self, editor: "FakeEditor", name: str, world: World) -> None:
        self.editor = editor
        self.name = name
        self.label = name
        self.path = LEVEL + name
        self.world = world
        self.valid = True
        self.root_component: Any = None
        self.components: List[Any] = []
        self.transform = Transform()

    def get_actor_label(self) -> str:
        return self.label

    def set_actor_label(self, label: str) -> None:
        self.label = label

    def get_path_name(self) -> str:
        return self.path

    def get_class(self) -> SimpleNamespace:
        return SimpleNamespace(get_name=lambda: self.cls_name)

    def get_world(self) -> World:
        return self.world

    def get_outermost(self) -> SimpleNamespace:
        return SimpleNamespace(get_name=lambda: LEVEL_PACKAGE)

    def get_components_by_class(self, cls: type) -> List[Any]:
        return [c for c in self.components if isinstance(c, cls)]

    def get_actor_location(self) -> Vector:
        return self.transform.translation

    def get_actor_transform(self) -> Transform:
        return self.transform

    def set_actor_transform(self, xf: Transform, sweep: bool, teleport: bool) -> None:
        self.transform = xf

    def modify(self) -> bool:
        self.editor.modified.append(self.path)
        return True


class DynamicMeshActor(Actor):
    cls_name = "DynamicMeshActor"

    def __init__(self, editor: "FakeEditor", name: str, world: World, mesh: DynamicMesh) -> None:
        super().__init__(editor, name, world)
        self.dynamic_mesh_component = DynamicMeshComponent(self, mesh)
        mesh.outer = self.dynamic_mesh_component.get_path_name()
        self.root_component = self.dynamic_mesh_component
        self.components = [self.dynamic_mesh_component]


class StaticMeshActor(Actor):
    cls_name = "StaticMeshActor"

    def __init__(self, editor: "FakeEditor", name: str, world: World, static_mesh: Any = None) -> None:
        super().__init__(editor, name, world)
        if isinstance(static_mesh, DynamicMesh):
            static_mesh = None  # what the fake spawn passes every class
        self.static_mesh_component = StaticMeshComponent(self, static_mesh)
        self.root_component = self.static_mesh_component
        self.components = [self.static_mesh_component]


class StaticMesh:
    def __init__(self, path: str, **stats: Any) -> None:
        self.path = path if "." in path else path + "." + path.rpartition("/")[2]
        self.stats = mesh_stats(**stats)
        self.static_materials: List[Any] = [SimpleNamespace(material_interface=None)]
        self.materials: Dict[int, Any] = {}
        self.collision = SimpleCollision()
        self.nanite_settings = SimpleNamespace(enabled=False)
        self.modified = 0

    def get_path_name(self) -> str:
        return self.path

    def get_name(self) -> str:
        return self.path.rpartition("/")[2].partition(".")[0]

    # With Nanite, LOD 0 is the fallback mesh: 316 of 1,920 triangles for a sphere (UE 5.7.4).
    def get_num_triangles(self, lod: int) -> int:
        return self.stats["tris"] // 6 if self.nanite_settings.enabled else self.stats["tris"]

    def get_num_vertices(self, lod: int) -> int:
        return self.stats["verts"] // 6 if self.nanite_settings.enabled else self.stats["verts"]

    def set_material(self, index: int, material: Any) -> None:
        self.materials[index] = material

    def get_editor_property(self, name: str) -> Any:
        return getattr(self, name)

    def modify(self) -> bool:
        self.modified += 1
        return True


class FakeEditor:
    """One editor: its level, its assets, and what the scripts did to them.

    Attributes:
        pie: Play In Editor runs.
        geometry_script: The GeometryScripting plugin is loaded.
        fail: GeometryScript function name -> the error message it reports.
        warn: GeometryScript function name -> the warning it reports.
        calls: (function, got a debug object) per GeometryScript call.
        arguments: function -> the positional arguments of its last call.
        transactions: Descriptions of the transactions that ended (not cancelled).
        events: What the scripts did to the level, in order.
        dirty: The packages with unsaved changes.
        namespace: The module namespace the last script ran in.
    """

    def __init__(self) -> None:
        self.world = World("/Game/Maps/Mesh.Mesh")
        self.pie = False
        self.geometry_script = True
        self.fail: Dict[str, str] = {}
        self.warn: Dict[str, str] = {}
        self.calls: List[Tuple[str, bool]] = []
        self.arguments: Dict[str, Tuple[Any, ...]] = {}
        self.transactions: List[str] = []
        self.events: List[Tuple[Any, ...]] = []
        self.modified: List[str] = []
        self.objects: Dict[str, Any] = {}
        self.assets: Dict[str, Any] = {}
        self.saved: List[str] = []
        self.spawned = 0
        self.dirty: set = set()
        self.namespace: Dict[str, Any] = {}

    def add_asset(self, asset: "StaticMesh") -> "StaticMesh":
        package = asset.path.partition(".")[0]
        self.assets[package] = asset
        self.assets[asset.path] = asset
        return asset

    # -- level ---------------------------------------------------------------

    def add(self, actor: Actor) -> Actor:
        self.objects[actor.path] = actor
        return actor

    def dynamic_actor(self, name: str, **stats: Any) -> DynamicMeshActor:
        """A DynamicMeshActor whose mesh is ``mesh_stats(**stats)``: a 100 cm box by default."""
        mesh = DynamicMesh()
        mesh.stats = mesh_stats(**stats)
        return self.add(DynamicMeshActor(self, name, self.world, mesh))  # type: ignore[return-value]

    def record(self, *event: Any) -> None:
        self.events.append(event)

    def level_actors(self) -> List[Actor]:
        return [a for a in self.objects.values() if isinstance(a, Actor) and a.valid and a.world is self.world]

    # -- GeometryScript ------------------------------------------------------

    def gs(
        self,
        name: str,
        implementation: Callable[..., Any],
        refuse: Optional[Callable[..., Optional[str]]] = None,
    ) -> Callable[..., Any]:
        """A GeometryScript function.

        It reports ``fail[name]``, or what ``refuse`` says about the
        arguments, as an error in its debug and returns its first argument
        unchanged, as GeometryScript does; otherwise it runs
        ``implementation``.
        """

        def function(*args: Any, debug: Optional[GeometryScriptDebug] = None, **kwargs: Any) -> Any:
            self.calls.append((name, debug is not None))
            self.arguments[name] = args
            if name in self.warn and debug is not None:
                debug.messages.append(SimpleNamespace(message=self.warn[name], message_type=WARNING))
            error = self.fail.get(name) or (refuse(*args, **kwargs) if refuse is not None else None)
            if error:
                if debug is not None:
                    debug.messages.append(SimpleNamespace(message=error, message_type=ERROR))
                return args[0] if args else None
            return implementation(*args, **kwargs)

        return function

    def module(self) -> SimpleNamespace:  # noqa: C901 - one fake module, one place
        """The fake ``unreal`` module."""
        editor = self
        gs = self.gs

        def query(field: str, tuple_out: bool = False) -> Callable[..., Any]:
            def read(mesh: DynamicMesh, *rest: Any) -> Any:
                value = mesh.stats[field]
                return (value, False) if tuple_out else value

            return read

        def copy_to(source: DynamicMesh, target: DynamicMesh) -> Tuple[DynamicMesh, DynamicMesh, DynamicMesh]:
            target.stats = copy.deepcopy(source.stats)
            return source, target, target

        def append(stats: Dict[str, Any]) -> Callable[[DynamicMesh], DynamicMesh]:
            def apply(mesh: DynamicMesh) -> DynamicMesh:
                mesh.stats = mesh_stats(**stats)
                return mesh

            return apply

        def append_box(mesh, options, xf, dx=100.0, dy=100.0, dz=100.0, *rest):
            return append(dict(lo=(-dx / 2, -dy / 2, -dz / 2), hi=(dx / 2, dy / 2, dz / 2), volume=dx * dy * dz,
                               area=2 * (dx * dy + dy * dz + dx * dz),
                               quad_groups=options.polygroup_mode == "PER_QUAD"))(mesh)

        def append_sphere(mesh, options, xf, radius=50.0, phi=10, theta=16, *rest):
            r = radius
            return append(dict(tris=2 * theta * (phi - 2), verts=theta * (phi - 2) + 2, lo=(-r, -r, -r), hi=(r, r, r),
                               volume=4.18879 * r ** 3, area=12.566 * r * r, uv_islands=1,
                               quad_groups=options.polygroup_mode == "PER_QUAD"))(mesh)

        def append_round(mesh, options, xf, radius=50.0, height=100.0, steps=12, *rest):
            return append(dict(tris=4 * steps, verts=2 * steps + 2, lo=(-radius, -radius, -height / 2),
                               hi=(radius, radius, height / 2), volume=3.14159 * radius * radius * height,
                               uv_islands=3))(mesh)

        def append_cone(mesh, options, xf, base=50.0, top=0.0, height=100.0, steps=12, *rest):
            return append_round(mesh, options, xf, base, height, steps)

        def append_capsule(mesh, options, xf, radius=30.0, length=75.0, hemi=5, circle=8, *rest):
            half = length / 2 + radius
            return append(dict(tris=circle * hemi * 4, verts=circle * hemi * 2 + 2, uv_islands=1,
                               lo=(-radius, -radius, -half), hi=(radius, radius, half)))(mesh)

        def append_torus(mesh, options, xf, revolve, major=50.0, minor=25.0, steps=16, minor_steps=8, *rest):
            outer = major + minor
            return append(dict(tris=2 * steps * minor_steps, verts=steps * minor_steps, uv_islands=1,
                               lo=(-outer, -outer, -minor), hi=(outer, outer, minor)))(mesh)

        def boolean(target, target_xf, tool, tool_xf, operation, options):
            target.stats["tris"] += tool.stats["tris"] if operation != "INTERSECTION" else 0
            target.stats["verts"] += tool.stats["verts"] // 2
            if operation == "SUBTRACT":
                target.stats["volume"] -= tool.stats["volume"] / 2
            return target

        def collision_from_mesh(mesh, options):
            hulls = options.max_convex_hulls_per_mesh
            return SimpleCollision(shapes=min(hulls, mesh.stats.get("convex_parts", hulls)))

        def collision_shapes(mesh, options, xf, collision, triangulation):
            n = collision.shapes
            faces = 12 if n == 1 else 12 * n
            mesh.stats = mesh_stats(tris=faces, verts=faces // 2 + 2, components=n, volume=1100000.0)
            return mesh

        def set_collision(collision, component, options):
            component.collision = collision
            editor.record("set_collision", component.owner.path, collision.shapes)

        def split(mesh, pool):
            n = mesh.stats["components"]
            parts = [DynamicMesh(tris=mesh.stats["tris"] // n, verts=mesh.stats["verts"] // n,
                                 volume=mesh.stats["volume"] / n) for _ in range(n)]
            return mesh, parts

        def scale_tris(factor: Callable[..., int]) -> Callable[..., DynamicMesh]:
            def apply(mesh, *args):
                mesh.stats["tris"] = factor(mesh.stats["tris"], *args)
                mesh.stats["verts"] = mesh.stats["tris"] // 2 + 2
                return mesh

            return apply

        def needs_quads(mesh, *rest):
            if mesh.stats["quad_groups"]:
                return None
            return ("ApplyPolygroupCatmullClarkSubD: Target Polygroup Layer does not define valid Polygons "
                    "for Subdivision")

        def needs_uv_set(label: str) -> Callable[..., Optional[str]]:
            def check(mesh, channel, options):
                return None if channel < mesh.stats["uv_sets"] else label + ": UVSetIndex does not exist on TargetMesh"

            return check

        def plane_cut(mesh, frame, options):
            side = mesh.stats.get("side_tris", {})
            kept = side.get("front" if options.flip_cut_side else "back", mesh.stats["tris"] // 2)
            mesh.stats["tris"] = kept + (6 if options.fill_holes and kept else 0)
            mesh.stats["verts"] = mesh.stats["tris"] // 2 + 2
            return mesh

        def remesh(mesh, options, uniform):
            mesh.stats["tris"] = uniform.target_triangle_count if uniform.target_type == "TRIANGLE_COUNT" else 478
            mesh.stats["verts"] = mesh.stats["tris"] // 2 + 2
            mesh.stats["sample"] = ([(0.0, 0.0, 0.0), (20.0, 0.0, 0.0), (10.0, 17.32, 0.0)], [(0, 1, 2)])
            return mesh

        def unwrap(islands: int) -> Callable[..., DynamicMesh]:
            def apply(mesh, channel, options):
                mesh.stats["uv_islands"] = islands
                mesh.stats["uv_area"] = 0.55
                return mesh

            return apply

        def repack(mesh, channel, options):
            mesh.stats["uv_area"] = 0.5
            return mesh

        def static_copy(asset, mesh, options, lod):
            mesh.stats = copy.deepcopy(asset.stats)
            return mesh, "SUCCESS"

        def copy_to_static(mesh, asset, options, lod, use_section_materials):
            asset.stats = copy.deepcopy(mesh.stats)
            if getattr(options, "apply_nanite_settings", False):
                asset.nanite_settings = options.new_nanite_settings
            return mesh, "SUCCESS"

        def new_static_mesh(mesh, path, options):
            asset = editor.add_asset(StaticMesh(path, **copy.deepcopy(mesh.stats)))
            if options.enable_nanite:
                asset.nanite_settings = options.nanite_settings
            editor.record("create_asset", asset.path)
            return asset, "SUCCESS"

        def set_static_collision(collision, asset, options, static_options):
            asset.collision = collision
            editor.record("set_static_collision", asset.path, collision.shapes, options.emit_transaction)

        def positions(mesh, skip_gaps):
            return mesh, [Vector(*p) for p in mesh.stats["sample"][0]], False

        def triangles(mesh, skip_gaps):
            return mesh, [SimpleNamespace(x=a, y=b, z=c) for a, b, c in mesh.stats["sample"][1]], False

        def uv_box(mesh, channel):
            return Box2D(mesh.stats["uv_lo"], mesh.stats["uv_hi"]), True, False

        def uv_size_info(mesh, channel, selection):
            return (mesh, mesh.stats["area"], mesh.stats["uv_area"], None, None, channel < mesh.stats["uv_sets"],
                    mesh.stats["unset_uvs"])

        def split_bowties(mesh, mesh_bowties, attribute_bowties):
            mesh.stats["verts"] += mesh.stats["bowties"]
            return mesh

        def repair_degenerates(mesh, options):
            mesh.stats["tris"] -= mesh.stats["degenerate"]
            return mesh

        def select_all(mesh, kind):
            return mesh, Selection(kind, mesh.stats["tris"], "all")

        def select_normal(mesh, normal, angle, kind, invert, points):
            facing = mesh.stats.get("facing", {}).get(tuple(normal), 2)
            return mesh, Selection(kind, facing, ("normal", tuple(normal), angle))

        def select_box(mesh, box, kind, invert, points):
            assert box.is_valid
            return mesh, Selection(kind, mesh.stats.get("in_box", 4), ("box", tuple(box.min), tuple(box.max)))

        def select_sharp(mesh, angle):
            return mesh, Selection("EDGES", mesh.stats.get("sharp_edges", 12), ("sharp", angle))

        def boundary(mesh, selection, exclude):
            return mesh, Selection("EDGES", 4, ("boundary", selection.how))

        class EditorActorSubsystem:
            def spawn_actor_from_class(self, cls, location, rotation=None):
                if editor.pie:
                    return None
                editor.spawned += 1
                actor = cls(editor, f"DynamicMeshActor_{editor.spawned}", editor.world, DynamicMesh())
                actor.transform = transform(tuple(location))
                editor.add(actor)
                editor.record("spawn", actor.path)
                return actor

            def destroy_actor(self, actor):
                actor.valid = False
                editor.record("destroy", actor.path)
                return True

            def get_all_level_actors(self):
                return editor.level_actors()

        class ScopedEditorTransaction:
            def __init__(self, description: str) -> None:
                self.description = description
                self.cancelled = False

            def __enter__(self):
                return self

            def __exit__(self, *exc: Any) -> None:
                if not self.cancelled:
                    editor.transactions.append(self.description)
                    editor.dirty.add(LEVEL_PACKAGE)

            def cancel(self) -> None:
                self.cancelled = True

        def assets_by_package_name(package: str) -> List[SimpleNamespace]:
            asset = editor.assets.get(package)
            if asset is None:
                return []
            return [SimpleNamespace(asset_name=package.rpartition("/")[2],
                                    asset_class_path=SimpleNamespace(asset_name=type(asset).__name__))]

        def load_asset(path: str) -> Any:
            if editor.pie:
                return None
            return editor.assets.get(path) or editor.assets.get(path.partition(".")[0])

        def save_asset(path: str, only_if_dirty: bool) -> bool:
            editor.saved.append(path)
            return True

        def dirty_packages() -> List[SimpleNamespace]:
            return [SimpleNamespace(get_name=lambda name=name: name) for name in sorted(editor.dirty)]

        subsystems = {
            "EditorActorSubsystem": EditorActorSubsystem(),
            "LevelEditorSubsystem": SimpleNamespace(is_in_play_in_editor=lambda: editor.pie),
            "UnrealEditorSubsystem": SimpleNamespace(
                get_editor_world=lambda: None if editor.pie else editor.world,
                get_game_world=lambda: World("/Game/Maps/UEDPIE_0_Mesh.Mesh") if editor.pie else None,
            ),
            "LayersSubsystem": SimpleNamespace(get_world=lambda: editor.world),
        }

        libraries: Dict[str, Any] = {
            "GeometryScript_MeshQueries": SimpleNamespace(
                get_vertex_count=query("verts"),
                get_num_open_border_edges=query("open_edges"),
                get_num_open_border_loops=query("open_loops", tuple_out=True),
                get_num_connected_components=query("components"),
                get_has_triangle_normals=query("normals"),
                get_has_triangle_id_gaps=query("gaps"),
                get_has_vertex_id_gaps=query("vertex_gaps"),
                get_is_closed_mesh=lambda mesh: mesh.stats["open_edges"] == 0,
                get_num_uv_sets=query("uv_sets"),
                get_num_uv_islands=lambda mesh, channel: (mesh.stats["uv_islands"], True),
                get_mesh_uv_area=lambda mesh, channel: (mesh.stats["uv_area"], True),
                get_uv_set_bounding_box=uv_box,
                get_mesh_volume_area=lambda mesh: (mesh.stats["area"], mesh.stats["volume"]),
                get_mesh_bounding_box=lambda mesh: Box(mesh.stats["lo"], mesh.stats["hi"]),
                get_all_vertex_positions=positions,
                get_all_triangle_indices=triangles,
            ),
            "GeometryScript_List": SimpleNamespace(
                convert_vector_list_to_array=lambda items: items,
                convert_triangle_list_to_array=lambda items: items,
            ),
            "GeometryScript_Primitives": SimpleNamespace(
                append_box=gs("append_box", append_box),
                append_sphere_lat_long=gs("append_sphere_lat_long", append_sphere),
                append_cylinder=gs("append_cylinder", append_round),
                append_cone=gs("append_cone", append_cone),
                append_capsule=gs("append_capsule", append_capsule),
                append_torus=gs("append_torus", append_torus),
                append_simple_collision_shapes=gs("append_simple_collision_shapes", collision_shapes),
            ),
            "GeometryScript_MeshDecomposition": SimpleNamespace(
                copy_mesh_to_mesh=gs("copy_mesh_to_mesh", copy_to),
                split_mesh_by_components=gs("split_mesh_by_components", split),
            ),
            "GeometryScript_MeshBooleans": SimpleNamespace(
                apply_mesh_boolean=gs("apply_mesh_boolean", boolean),
                apply_mesh_plane_cut=gs("apply_mesh_plane_cut", plane_cut),
            ),
            "GeometryScript_Collision": SimpleNamespace(
                generate_collision_from_mesh=gs("generate_collision_from_mesh", collision_from_mesh),
                set_simple_collision_of_dynamic_mesh_component=gs("set_simple_collision_of_dynamic_mesh_component",
                                                                  set_collision),
                get_simple_collision_from_component=lambda component: component.collision,
                get_simple_collision_from_static_mesh=lambda asset: asset.collision,
                set_simple_collision_of_static_mesh=gs("set_simple_collision_of_static_mesh", set_static_collision),
                get_simple_collision_shape_count=lambda collision: collision.shapes,
            ),
            "GeometryScript_MeshSelection": SimpleNamespace(
                create_select_all_mesh_selection=select_all,
                select_mesh_elements_by_normal_angle=select_normal,
                select_mesh_elements_in_box=select_box,
                select_mesh_sharp_edges=select_sharp,
                select_selection_boundary_edges=boundary,
                get_mesh_unique_selection_info=lambda mesh, selection: (selection.kind, selection.count),
            ),
            "GeometryScript_MeshModeling": SimpleNamespace(
                apply_mesh_linear_extrude_faces=gs("apply_mesh_linear_extrude_faces",
                                                   scale_tris(lambda tris, *a: tris + 8)),
                apply_mesh_bevel_selection=gs("apply_mesh_bevel_selection", scale_tris(lambda tris, *a: tris * 3)),
                apply_mesh_inset_outset_faces=gs("apply_mesh_inset_outset_faces",
                                                 scale_tris(lambda tris, *a: tris * 5)),
            ),
            "GeometryScript_OpenSubdiv": SimpleNamespace(
                apply_polygroup_catmull_clark_sub_d=gs("apply_polygroup_catmull_clark_sub_d",
                                                       scale_tris(lambda tris, level, layer: tris * 4 ** level),
                                                       needs_quads),
                apply_triangle_loop_sub_d=gs("apply_triangle_loop_sub_d",
                                             scale_tris(lambda tris, level: tris * 4 ** level)),
            ),
            "GeometryScript_MeshSubdivide": SimpleNamespace(
                apply_uniform_tessellation=gs("apply_uniform_tessellation",
                                              scale_tris(lambda tris, level: tris * (level + 1) ** 2)),
            ),
            "GeometryScript_MeshSimplification": SimpleNamespace(
                apply_simplify_to_triangle_count=gs("apply_simplify_to_triangle_count",
                                                    scale_tris(lambda tris, target, options: min(tris, target))),
                apply_simplify_to_tolerance=gs("apply_simplify_to_tolerance",
                                               scale_tris(lambda tris, tolerance, options: max(4, tris // 4))),
            ),
            "GeometryScript_MeshRepair": SimpleNamespace(
                split_mesh_bowties=gs("split_mesh_bowties", split_bowties),
                repair_mesh_degenerate_geometry=gs("repair_mesh_degenerate_geometry", repair_degenerates),
            ),
            "GeometryScript_Remeshing": SimpleNamespace(apply_uniform_remesh=gs("apply_uniform_remesh", remesh)),
            "GeometryScript_UVs": SimpleNamespace(
                auto_generate_x_atlas_mesh_u_vs=gs("auto_generate_x_atlas_mesh_u_vs", unwrap(7),
                                                   needs_uv_set("AutoGenerateXAtlasMeshUVs")),
                auto_generate_patch_builder_mesh_u_vs=gs("auto_generate_patch_builder_mesh_u_vs", unwrap(11),
                                                         needs_uv_set("AutoGeneratePatchBuilderMeshUVs")),
                repack_mesh_u_vs=gs("repack_mesh_u_vs", repack),
                get_mesh_uv_size_info=uv_size_info,
            ),
            "GeometryScript_AssetUtils": SimpleNamespace(
                copy_mesh_from_static_mesh=gs("copy_mesh_from_static_mesh", static_copy),
                copy_mesh_to_static_mesh=gs("copy_mesh_to_static_mesh", copy_to_static),
                get_section_material_list_from_static_mesh=lambda asset, lod: (
                    ["Material_" + asset.get_name()], [0], ["Slot"], "SUCCESS"),
            ),
            "GeometryScript_NewAssetUtils": SimpleNamespace(
                create_new_static_mesh_asset_from_mesh=gs("create_new_static_mesh_asset_from_mesh", new_static_mesh),
            ),
            "GeometryScript_SceneUtils": SimpleNamespace(
                set_component_material_list=lambda component, materials: setattr(component, "materials", materials),
            ),
        }
        if not self.geometry_script:
            libraries = {}

        module = SimpleNamespace(
            Actor=Actor,
            DynamicMeshActor=DynamicMeshActor,
            DynamicMeshComponent=DynamicMeshComponent,
            StaticMeshComponent=StaticMeshComponent,
            StaticMeshActor=StaticMeshActor,
            DynamicMesh=DynamicMesh,
            Vector=Vector,
            Transform=Transform,
            Box=Box,
            ScopedEditorTransaction=ScopedEditorTransaction,
            GeometryScriptDebug=GeometryScriptDebug,
            GeometryScriptDebugMessageType=SimpleNamespace(ERROR_MESSAGE=ERROR, WARNING_MESSAGE=WARNING),
            GeometryScriptBooleanOperation=Enum("UNION", "SUBTRACT", "INTERSECTION"),
            GeometryScriptBooleanOutputSpace=Enum("TARGET_TRANSFORM_SPACE", "TOOL_TRANSFORM_SPACE"),
            GeometryScriptPrimitivePolygroupMode=Enum("PER_FACE", "PER_QUAD", "SINGLE_GROUP"),
            GeometryScriptPrimitiveOriginMode=Enum("CENTER", "BASE"),
            GeometryScriptUniformRemeshTargetType=Enum("TARGET_EDGE_LENGTH", "TRIANGLE_COUNT"),
            GeometryScriptMeshBevelSelectionMode=Enum("TRIANGLE_AREA", "ALL_POLYGROUP_EDGES", "SELECTED_EDGES"),
            GeometryScriptMeshSelectionType=Enum("TRIANGLES", "EDGES"),
            GeometryScriptLinearExtrudeDirection=Enum("FIXED_DIRECTION", "AVERAGE_FACE_NORMAL"),
            GeometryScriptPolyOperationArea=Enum("ENTIRE_SELECTION", "PER_POLYGROUP"),
            GeometryScriptCollisionGenerationMethod=Enum("CONVEX_HULLS", "MIN_VOLUME_SHAPES"),
            GeometryScriptRepairMeshMode=Enum("DELETE_ONLY", "REPAIR_OR_DELETE"),
            MathLibrary=SimpleNamespace(make_rot_from_z=lambda vector: Rotator()),
            SystemLibrary=SimpleNamespace(is_valid=is_valid),
            EditorAssetLibrary=SimpleNamespace(
                load_asset=load_asset, save_asset=save_asset, does_asset_exist=lambda path: path in editor.assets,
            ),
            AssetRegistryHelpers=SimpleNamespace(get_asset_registry=lambda: SimpleNamespace(
                get_assets_by_package_name=assets_by_package_name)),
            EditorLoadingAndSavingUtils=SimpleNamespace(get_dirty_map_packages=dirty_packages,
                                                        get_dirty_content_packages=lambda: []),
            Paths=SimpleNamespace(project_content_dir=lambda: "../../Project/Content/",
                                  convert_relative_path_to_full=lambda path: "/Projects/Mesh/Content/"),
            StaticMesh=StaticMesh,
            GeometryScriptSimpleCollision=SimpleCollision,
            GeometryScriptOutcomePins=Enum("SUCCESS", "FAILURE"),
            find_object=lambda outer, path: editor.objects.get(path),
            get_editor_subsystem=lambda name: subsystems[name],
            **{name: name for name in subsystems},
            **libraries,
        )
        for name, defaults in OPTION_STRUCTS.items():
            setattr(module, "GeometryScript" + name, options_struct("GeometryScript" + name, **defaults))
        return module

    # -- running scripts -----------------------------------------------------

    def run(self, code: str) -> Dict[str, Any]:
        """Run ``code`` as Remote Control's ExecutePythonCommandEx would."""
        previous = sys.modules.get("unreal")
        sys.modules["unreal"] = self.module()  # type: ignore[assignment]
        out = io.StringIO()
        self.namespace = {"__name__": "__main__"}
        try:
            with redirect_stdout(out):
                exec(code, self.namespace)
        except Exception as exc:  # surface script errors like UE does
            return {"ReturnValue": False, "CommandResult": f"Traceback ...\n{type(exc).__name__}: {exc}"}
        finally:
            if previous is None:
                sys.modules.pop("unreal", None)
            else:
                sys.modules["unreal"] = previous
        return {"ReturnValue": True,
                "LogOutput": [{"Type": "Info", "Output": line} for line in out.getvalue().splitlines()]}


def level_editor() -> FakeEditor:
    """An editor with the level the mesh tests use.

    ``Body`` is a 100 cm box with quad polygroups, ``Ball`` a lat-long sphere
    of 1,920 triangles (no quad polygroups), ``Big`` a 318,400-triangle mesh,
    ``Crate`` a StaticMeshActor rendering the ``/Game/Meshes/SM_Crate`` asset
    and ``Light`` an actor with neither mesh component.
    """
    editor = FakeEditor()
    editor.dynamic_actor("Body")
    editor.dynamic_actor("Ball", tris=1920, verts=962, quad_groups=False, uv_islands=1)
    editor.dynamic_actor("Big", tris=318_400, verts=159_202, quad_groups=False)
    asset = editor.add_asset(StaticMesh("/Game/Meshes/SM_Crate.SM_Crate", tris=12, verts=24))
    editor.add_asset(StaticMesh("/Engine/BasicShapes/Cube.Cube", tris=12, verts=24))
    crate = editor.add(StaticMeshActor(editor, "Crate", editor.world, asset))
    crate.transform = transform((0, 500, 0), yaw=45, scale=(2, 1, 1))
    editor.add(Actor(editor, "Light", editor.world))
    return editor
