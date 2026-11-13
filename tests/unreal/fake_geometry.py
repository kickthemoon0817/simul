"""A fake ``unreal`` module with enough GeometryScript to run the mesh tool scripts.

The mesh tools (``simul.adapters._unreal_mesh_scripts``) run fixed Python in
the editor. These tests run the same scripts against this fake, which keeps a
small level (DynamicMeshActors, a StaticMeshActor, a StaticMesh asset), fakes
the GeometryScript libraries over meshes described by their counts and bounds,
and records what a script did: every GeometryScript call (with whether it got
a ``GeometryScriptDebug``), every transaction, every spawn. The numbers follow
the rules below rather than real geometry, which is enough to check what a
script reports against what it did.

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
"""

from __future__ import annotations

import copy
import io
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


class Vector2D:
    def __init__(self, x: float = 0.0, y: float = 0.0) -> None:
        self.x, self.y = float(x), float(y)


class Rotator:
    def __init__(self, pitch: float = 0.0, yaw: float = 0.0, roll: float = 0.0) -> None:
        self.pitch, self.yaw, self.roll = pitch, yaw, roll

    def quaternion(self) -> "Quat":
        return Quat(self)


class Quat:
    def __init__(self, rotator: Optional[Rotator] = None) -> None:
        self._rotator = rotator or Rotator()

    def rotator(self) -> Rotator:
        return self._rotator


class Transform:
    def __init__(self) -> None:
        self.translation = Vector()
        self.rotation = Quat()
        self.scale3d = Vector(1, 1, 1)


class Box:
    def __init__(self, lo: Tuple[float, ...], hi: Tuple[float, ...]) -> None:
        self.min, self.max = Vector(*lo), Vector(*hi)
        self.is_valid = True


class Box2D:
    def __init__(self, lo: Tuple[float, float], hi: Tuple[float, float], valid: bool = True) -> None:
        self.min, self.max = Vector2D(*lo), Vector2D(*hi)
        self.is_valid = valid


class Options:
    """Every GeometryScript options struct: attributes the script sets."""

    def __init__(self, **values: Any) -> None:
        self.__dict__.update(values)


#: The GeometryScript option structs the scripts build, without their
#: ``GeometryScript`` prefix.
OPTION_STRUCTS = (
    "PrimitiveOptions", "MeshBooleanOptions", "ConvexHullOptions", "ConvexDecompositionOptions",
    "MeshSelection", "MeshLinearExtrudeOptions", "MeshBevelSelectionOptions", "MeshInsetOutsetFacesOptions",
    "GroupLayer", "SimplifyMeshOptions", "MeshPlaneCutOptions", "RemeshOptions", "UniformRemeshOptions",
    "XAtlasOptions", "PatchBuilderOptions", "CopyMeshFromAssetOptions", "MeshReadLOD",
)


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
        "uv_sets": 1,
        "uv_islands": 6,
        "uv_area": 0.6,
        "uv_lo": (0.0, 0.0),
        "uv_hi": (1.0, 1.0),
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
        self.location = Vector()
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
        return SimpleNamespace(get_name=lambda: self.editor.package)

    def get_components_by_class(self, cls: type) -> List[Any]:
        return [c for c in self.components if isinstance(c, cls)]

    def get_actor_location(self) -> Vector:
        return self.location

    def set_actor_transform(self, transform: Transform, sweep: bool, teleport: bool) -> None:
        self.transform = transform
        self.location = transform.translation

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

    def __init__(self, editor: "FakeEditor", name: str, world: World, static_mesh: Any) -> None:
        super().__init__(editor, name, world)
        self.static_mesh_component = StaticMeshComponent(self, static_mesh)
        self.root_component = self.static_mesh_component
        self.components = [self.static_mesh_component]


class StaticMesh:
    def __init__(self, path: str, **stats: Any) -> None:
        self.path = path
        self.stats = mesh_stats(**stats)

    def get_path_name(self) -> str:
        return self.path

    def get_name(self) -> str:
        return self.path.rpartition("/")[2].partition(".")[0]


class FakeEditor:
    """One editor: its level, its assets, and what the scripts did to them.

    Attributes:
        pie: Play In Editor runs.
        geometry_script: The GeometryScripting plugin is loaded.
        fail: GeometryScript function name -> the error message it reports.
        warn: GeometryScript function name -> the warning it reports.
        calls: (function, got a debug object) per GeometryScript call.
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
        self.transactions: List[str] = []
        self.events: List[Tuple[Any, ...]] = []
        self.modified: List[str] = []
        self.objects: Dict[str, Any] = {}
        self.assets: Dict[str, Any] = {}
        self.spawned = 0
        self.dirty: set = set()
        self.package = LEVEL_PACKAGE
        self.namespace: Dict[str, Any] = {}

    # -- level ---------------------------------------------------------------

    def add(self, actor: Actor) -> Actor:
        self.objects[actor.path] = actor
        return actor

    def dynamic_actor(self, name: str, **stats: Any) -> DynamicMeshActor:
        return self.add(DynamicMeshActor(self, name, self.world, DynamicMesh(**stats)))  # type: ignore[return-value]

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
            if name in self.warn and debug is not None:
                debug.messages.append(SimpleNamespace(message=self.warn[name], message_type=WARNING))
            error = self.fail.get(name) or (refuse(*args, **kwargs) if refuse is not None else None)
            if error:
                if debug is not None:
                    debug.messages.append(SimpleNamespace(message=error, message_type=ERROR))
                return args[0] if args else None
            return implementation(*args, **kwargs)

        return function

    def module(self) -> SimpleNamespace:
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
                               area=2 * (dx * dy + dy * dz + dx * dz)))(mesh)

        def append_sphere(mesh, options, xf, radius=50.0, phi=10, theta=16, *rest):
            r = radius
            return append(dict(tris=2 * theta * (phi - 2), verts=theta * (phi - 2) + 2, lo=(-r, -r, -r), hi=(r, r, r),
                               volume=4.18879 * r ** 3, area=12.566 * r * r, uv_islands=1,
                               quad_groups=False))(mesh)

        def append_round(mesh, options, xf, radius=50.0, height=100.0, steps=12, *rest):
            return append(dict(tris=4 * steps, verts=2 * steps + 2, lo=(-radius, -radius, -height / 2),
                               hi=(radius, radius, height / 2), volume=3.14159 * radius * radius * height,
                               uv_islands=3))(mesh)

        def append_cone(mesh, options, xf, base=50.0, top=0.0, height=100.0, steps=12, *rest):
            return append_round(mesh, options, xf, base, height, steps)

        def append_capsule(mesh, options, xf, radius=30.0, length=75.0, hemi=5, circle=8, *rest):
            return append(dict(tris=circle * hemi * 4, verts=circle * hemi * 2 + 2, uv_islands=1))(mesh)

        def boolean(target, target_xf, tool, tool_xf, operation, options):
            target.stats["tris"] += tool.stats["tris"] if operation != "INTERSECTION" else 0
            target.stats["verts"] += tool.stats["verts"] // 2
            return target

        def hull(mesh, out, selection, options):
            out.stats = mesh_stats(tris=min(mesh.stats["tris"], 40), verts=min(mesh.stats["verts"], 22),
                                   lo=mesh.stats["lo"], hi=mesh.stats["hi"], volume=mesh.stats["volume"] * 1.25)
            return mesh, out, out

        def decompose(mesh, out, options):
            pieces = max(1, min(options.num_hulls, 3))
            out.stats = mesh_stats(tris=12 * pieces, verts=8 * pieces, components=pieces,
                                   volume=mesh.stats["volume"] * 1.1)
            return mesh, out, out

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

        def catmull_clark(mesh, level, layer):
            return scale_tris(lambda tris, lvl, _layer: tris * 4 ** lvl)(mesh, level, layer)

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
            kept = mesh.stats["tris"] // 2
            mesh.stats["tris"] = kept + (6 if options.fill_holes else 0)
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

        def static_copy(asset, mesh, options, lod):
            mesh.stats = copy.deepcopy(asset.stats)
            return mesh, "SUCCESS"

        def positions(mesh, skip_gaps):
            return mesh, [Vector(*p) for p in mesh.stats["sample"][0]], False

        def triangles(mesh, skip_gaps):
            return mesh, [SimpleNamespace(x=a, y=b, z=c) for a, b, c in mesh.stats["sample"][1]], False

        def uv_box(mesh, channel):
            return Box2D(mesh.stats["uv_lo"], mesh.stats["uv_hi"]), True, False

        class EditorActorSubsystem:
            def spawn_actor_from_class(self, cls, location, rotation=None):
                if editor.pie:
                    return None
                editor.spawned += 1
                actor = cls(editor, f"DynamicMeshActor_{editor.spawned}", editor.world, DynamicMesh())
                actor.location = location
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
            ),
            "GeometryScript_MeshDecomposition": SimpleNamespace(
                copy_mesh_to_mesh=gs("copy_mesh_to_mesh", copy_to),
                split_mesh_by_components=gs("split_mesh_by_components", split),
            ),
            "GeometryScript_MeshBooleans": SimpleNamespace(
                apply_mesh_boolean=gs("apply_mesh_boolean", boolean),
                apply_mesh_plane_cut=gs("apply_mesh_plane_cut", plane_cut),
            ),
            "GeometryScript_Containment": SimpleNamespace(
                compute_mesh_convex_hull=gs("compute_mesh_convex_hull", hull),
                compute_mesh_convex_decomposition=gs("compute_mesh_convex_decomposition", decompose),
            ),
            "GeometryScript_MeshModeling": SimpleNamespace(
                apply_mesh_linear_extrude_faces=gs("apply_mesh_linear_extrude_faces",
                                                   scale_tris(lambda tris, *a: tris * 2)),
                apply_mesh_bevel_selection=gs("apply_mesh_bevel_selection", scale_tris(lambda tris, *a: tris * 3)),
                apply_mesh_inset_outset_faces=gs("apply_mesh_inset_outset_faces",
                                                 scale_tris(lambda tris, *a: tris * 5)),
            ),
            "GeometryScript_OpenSubdiv": SimpleNamespace(
                apply_polygroup_catmull_clark_sub_d=gs("apply_polygroup_catmull_clark_sub_d", catmull_clark,
                                                       needs_quads),
            ),
            "GeometryScript_MeshSubdivide": SimpleNamespace(
                apply_uniform_tessellation=gs("apply_uniform_tessellation",
                                              scale_tris(lambda tris, level: tris * (level + 1) ** 2)),
            ),
            "GeometryScript_MeshSimplification": SimpleNamespace(
                apply_simplify_to_triangle_count=gs("apply_simplify_to_triangle_count",
                                                    scale_tris(lambda tris, target, options: min(tris, target))),
            ),
            "GeometryScript_Remeshing": SimpleNamespace(apply_uniform_remesh=gs("apply_uniform_remesh", remesh)),
            "GeometryScript_UVs": SimpleNamespace(
                auto_generate_x_atlas_mesh_u_vs=gs("auto_generate_x_atlas_mesh_u_vs", unwrap(7),
                                                   needs_uv_set("AutoGenerateXAtlasMeshUVs")),
                auto_generate_patch_builder_mesh_u_vs=gs("auto_generate_patch_builder_mesh_u_vs", unwrap(11),
                                                         needs_uv_set("AutoGeneratePatchBuilderMeshUVs")),
            ),
            "GeometryScript_AssetUtils": SimpleNamespace(
                copy_mesh_from_static_mesh=gs("copy_mesh_from_static_mesh", static_copy),
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
            ScopedEditorTransaction=ScopedEditorTransaction,
            GeometryScriptDebug=GeometryScriptDebug,
            GeometryScriptDebugMessageType=SimpleNamespace(ERROR_MESSAGE=ERROR, WARNING_MESSAGE=WARNING),
            GeometryScriptBooleanOperation=Enum("UNION", "SUBTRACT", "INTERSECTION"),
            GeometryScriptPrimitivePolygroupMode=Enum("PER_FACE", "PER_QUAD", "SINGLE_GROUP"),
            GeometryScriptPrimitiveOriginMode=Enum("CENTER", "BASE"),
            GeometryScriptUniformRemeshTargetType=Enum("TARGET_EDGE_LENGTH", "TRIANGLE_COUNT"),
            GeometryScriptMeshBevelSelectionMode=Enum("TRIANGLE_AREA", "ALL_POLYGROUP_EDGES"),
            MathLibrary=SimpleNamespace(make_rot_from_z=lambda vector: Rotator()),
            SystemLibrary=SimpleNamespace(is_valid=is_valid),
            EditorAssetLibrary=SimpleNamespace(load_asset=load_asset),
            AssetRegistryHelpers=SimpleNamespace(get_asset_registry=lambda: SimpleNamespace(
                get_assets_by_package_name=assets_by_package_name)),
            EditorLoadingAndSavingUtils=SimpleNamespace(get_dirty_map_packages=dirty_packages,
                                                        get_dirty_content_packages=lambda: []),
            find_object=lambda outer, path: editor.objects.get(path),
            get_editor_subsystem=lambda name: subsystems[name],
            **{name: name for name in subsystems},
            **libraries,
        )
        for name in OPTION_STRUCTS:
            setattr(module, "GeometryScript" + name, Options)
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
    asset = StaticMesh("/Game/Meshes/SM_Crate.SM_Crate", tris=12, verts=24)
    editor.assets["/Game/Meshes/SM_Crate"] = asset
    editor.assets["/Game/Meshes/SM_Crate.SM_Crate"] = asset
    editor.add(StaticMeshActor(editor, "Crate", editor.world, asset))
    editor.add(Actor(editor, "Light", editor.world))
    return editor
