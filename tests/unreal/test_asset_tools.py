"""The Unreal asset, material and USD tools neither crash the editor nor lose work.

An audit of these tools on UE 5.7.4 found, and this module pins down:

- ``search_unreal_assets`` with ``max_results=-1`` killed the editor ("Trying
  to resize TArray to an invalid size of 4294967295"): nothing bounded it.
- ``export_unreal_usd`` left its AssetExportTask, whose Object is the editor
  world, in the console namespace Remote Control runs scripts in; the next
  level load failed the editor's world leak check and took it down. The
  import's factory holds the scene actor it spawned the same way.
- ``import_unreal_usd`` ignored ``replace_existing=False``: Unreal's USD
  importer replaced an earlier import's assets (a customized mesh lost its
  collision and lightmap resolution) and deleted its actors.
- One import with ``{"import_actors": false}`` saved bImportActors=False to
  the project's EditorPerProjectUserSettings.ini (UsdStageImportOptions saves
  its config on every notified change), so every later import spawned no
  actors.
- ``assign_unreal_material`` opened no transaction and marked nothing dirty:
  a save kept nothing, a reload reverted it and Undo did not undo it.
- An actor path in an unopened map made the tools load that map
  (``unreal.load_object``): assign edited its actor, export wrote an empty
  stage reporting the actor exported.
- ``create_unreal_material_instance`` saved an instance of an /Engine parent
  into the engine install, and ``set_unreal_material_params`` edited the
  engine's own instances.
- The first USD export of a session ran 26-40 s against the 30 s timeout and
  came back as a bare TimeoutError while its files were still written.

The fake editor below answers as UE 5.7.4 did. Scripts run in one persistent
namespace, as Remote Control's ``FileExecutionScope: Public`` runs them, so a
reference a script leaves behind stays visible to the next one.
"""

from __future__ import annotations

import ast
import asyncio
import io
import json
import os
import sys
from contextlib import contextmanager, redirect_stdout
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Tuple
from unittest.mock import AsyncMock

import pytest
from fastmcp import Client

import simul.usd.sandbox  # noqa: F401 - imported with the real pxr, before the fake editor's pxr
from simul.adapters import unreal_runtime
from simul.adapters.unreal_runtime import MAX_ASSET_SEARCH_RESULTS, UnrealRuntimeSession
from simul.config import Settings
from simul.utils.paths import PathPolicy
from simul.mcp import server as server_module
from simul.mcp.registration import _unreal_tools
from simul.mcp.schemas.unreal import (
    UnrealAssignMaterialResponse,
    UnrealCreateMaterialInstanceResponse,
    UnrealExportUsdResponse,
    UnrealImportUsdResponse,
)
from tests.fakes import AvailableAdapter

MAIN = "/Game/Maps/Main.Main"
OTHER = "/Game/Maps/Other.Other"
LEVEL = MAIN + ":PersistentLevel."
CUBE = LEVEL + "Cube"
SPHERE = LEVEL + "Sphere"
SETTINGS_ACTOR = LEVEL + "WorldSettings"
OFF_MAP = OTHER + ":PersistentLevel.OtherCube"
RED = "/Game/Materials/MI_Red"
BASIC = "/Engine/BasicShapes/BasicShapeMaterial"
ASSETS_PIPELINE = "/Interchange/Pipelines/DefaultAssetsPipeline.DefaultAssetsPipeline"


class Enum:
    """An Unreal enum value: compared and named by its member name."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Enum) and other.name == self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return "<" + self.name + ">"


COLLISION = SimpleNamespace(**{name: Enum(name) for name in ("NONE", "CONVEX_HULL", "CUBE")})

#: UsdStageImportOptions' saved settings as UE 5.7 ships them (its constructor defaults).
UE_DEFAULTS: Dict[str, Any] = {
    "import_actors": True,
    "import_geometry": True,
    "import_materials": True,
    "import_skeletal_animations": True,
    "import_level_sequences": True,
    "import_groom_assets": True,
    "import_sparse_volume_textures": True,
    "import_sounds": True,
    "import_only_used_materials": False,
    "use_prim_kinds_for_collapsing": True,
    "kinds_to_collapse": 18,
    "merge_identical_material_slots": True,
    "share_assets_for_identical_prims": True,
    "prim_path_folder_structure": False,
    "interpret_lods": True,
    "nanite_triangle_threshold": 2147483647,
    "fallback_collision_type": COLLISION.NONE,
    "override_stage_options": False,
    "existing_actor_policy": "REPLACE",
    "existing_asset_policy": "REPLACE",
}
#: UsdStageImportOptions properties that are not saved settings.
NOT_SAVED = {"prims_to_import": ["/"], "stage_options": None}
LOG_TIME = "[2026.11.07-18.46.14:701][753]"
#: The editor's base directory, which FPaths::ConvertRelativePathToFull resolves against.
ENGINE_BINARIES = "/Users/Shared/Epic Games/UE_5.7/Engine/Binaries/Mac"


# ---------------------------------------------------------------------------
# The fake editor
# ---------------------------------------------------------------------------


class UObject:
    def __init__(self, path: str) -> None:
        self.path = path

    def get_path_name(self) -> str:
        return self.path

    def get_name(self) -> str:
        return self.path.replace(":", ".").rsplit(".", 1)[-1]

    def get_class(self) -> SimpleNamespace:
        return SimpleNamespace(get_name=lambda: type(self).__name__)


def _static_class(path: str) -> Any:
    return classmethod(lambda cls: SimpleNamespace(get_path_name=lambda: path))


class World(UObject):
    def get_package(self) -> UObject:
        return UObject(self.path.split(".", 1)[0])


class MaterialInterface(UObject):
    static_class = _static_class("/Script/Engine.MaterialInterface")


class Material(MaterialInterface):
    static_class = _static_class("/Script/Engine.Material")


class MaterialInstanceConstant(MaterialInterface):
    static_class = _static_class("/Script/Engine.MaterialInstanceConstant")


class StaticMesh(UObject):
    static_class = _static_class("/Script/Engine.StaticMesh")


class SceneComponent(UObject):
    pass


class MeshComponent(SceneComponent):
    def __init__(self, path: str, editor: "Editor", slots: Tuple[str, ...] = ("Default",)) -> None:
        super().__init__(path)
        self.editor = editor
        self.slot_names = list(slots)
        self.materials: List[Optional[MaterialInterface]] = [editor.basic] * len(slots)

    def get_num_materials(self) -> int:
        return len(self.materials)

    def get_material_slot_names(self) -> List[str]:
        return list(self.slot_names)

    def get_material_index(self, name: str) -> int:
        return self.slot_names.index(name) if name in self.slot_names else -1

    def get_material(self, index: int) -> Optional[MaterialInterface]:
        return self.materials[index]

    def modify(self, always_mark_dirty: bool = True) -> bool:
        self.editor.events.append(("modify", self.path, self.editor.open_transactions))
        self.editor.dirty.add(self.path.split(".", 1)[0])
        return True

    def set_material(self, index: int, material: MaterialInterface) -> None:
        self.editor.events.append(("set_material", index, material.path, self.editor.open_transactions))
        self.materials[index] = material


class DynamicMeshComponent(MeshComponent):
    pass


class Level(UObject):
    def __init__(self, path: str, outer: World) -> None:
        super().__init__(path)
        self.outer = outer

    def get_outer(self) -> World:
        return self.outer


class Actor(UObject):
    def __init__(self, path: str, world: World, editor: "Editor", exported: bool = True,
                 meshes: Optional[Dict[str, Tuple[str, ...]]] = None, mesh_root: bool = True,
                 dynamic: bool = False, level_world: Optional[World] = None) -> None:
        super().__init__(path)
        self.world = world
        # The level holding the actor: the persistent level's outer is the world, a sublevel's its own.
        self.level = Level(path.split(":", 1)[0] + ":PersistentLevel", level_world or world)
        meshes = meshes or {"StaticMeshComponent0": ("Default",)}
        kind = DynamicMeshComponent if dynamic else MeshComponent
        self.components = [kind(path + "." + name, editor, slots) for name, slots in meshes.items()]
        for component in self.components:
            component.owner = self  # type: ignore[attr-defined]
        self.root_component: Any = self.components[0] if mesh_root else SceneComponent(path + ".Root")
        self.root_component.owner = self
        self.exported = exported

    def get_world(self) -> World:
        return self.world

    def get_level(self) -> Any:
        return self.level

    def get_actor_label(self) -> str:
        return self.get_name()

    def get_components_by_class(self, cls: type) -> List[Any]:
        return [component for component in self.components if isinstance(component, cls)]


class AssetExportTask:
    def __init__(self) -> None:
        self.object: Any = None
        self.exporter: Any = None
        self.options: Any = None
        self.errors: List[str] = []


class AssetImportTask:
    def __init__(self) -> None:
        self.factory: Any = None
        self.options: Any = None
        self.imported_object_paths: List[str] = []
        self.destination_name = ""


class UsdStageImportFactory:
    def __init__(self) -> None:
        # FUsdStageImportContext.SceneActor: the actor the import spawned.
        self.scene_actor: Any = None


class LevelExporterUSD:
    pass


class Inner:
    def __init__(self) -> None:
        self.selection_only = False
        self.export_actor_folders = False
        self.export_sublayers = False
        self.root_prim_name = "Root"
        self.asset_folder: Any = None

    def copy(self) -> "Inner":
        copy = Inner()
        copy.__dict__.update(self.__dict__)
        return copy

    def set_editor_property(self, name: str, value: Any) -> None:
        setattr(self, name, value)


class Stage:
    def __init__(self, prims: List[str]) -> None:
        self.prims = prims

    def GetPrimAtPath(self, path: str) -> Any:  # noqa: N802 - pxr's name
        return SimpleNamespace() if path in self.prims else None


class AssetData:
    """FAssetData: what the registry knows of an asset, loading nothing."""

    def __init__(self, package: str, kind: str, tags: Optional[Dict[str, str]] = None) -> None:
        self.package_name = package
        self.package_path, _, self.asset_name = package.rpartition("/")
        self.asset_class_path = SimpleNamespace(package_name="/Script/Engine", asset_name=kind)
        self.tags = tags or {}

    def is_redirector(self) -> bool:
        return self.asset_class_path.asset_name == "ObjectRedirector"

    def get_tag_value(self, tag: str) -> Optional[str]:
        return self.tags.get(tag)


#: A sliver of the class tree the fake registry filters with.
SUBCLASSES = {"/Script/Engine.MaterialInterface": {"Material", "MaterialInstanceConstant"}}

NOTIFY = SimpleNamespace(DEFAULT="DEFAULT", NEVER="NEVER", ALWAYS="ALWAYS")


class Editor:
    """One editor: a level with a cube and a sphere, materials, a content folder and a config file."""

    def __init__(self, tmp_path: Any) -> None:
        self.tmp_path = tmp_path
        self.pie = False
        self.events: List[Any] = []
        self.open_transactions = 0
        self.transactions: List[str] = []
        self.dirty: set = set()
        self.unsaveable: set = set()
        self.saved_packages: List[str] = []
        self.save_ok = True  # what EditorAssetLibrary.save_loaded_asset answers
        self.maps_loaded: List[str] = []
        self.console_commands: List[str] = []
        self.basic = Material(BASIC + ".BasicShapeMaterial")
        self.red = MaterialInterface(RED + ".MI_Red")
        self.world = World(MAIN)
        self.other_world = World(OTHER)
        self.actors = {
            path: Actor(path, self.world, self)
            for path in (CUBE, SPHERE)
        }
        self.actors[SETTINGS_ACTOR] = Actor(SETTINGS_ACTOR, self.world, self, exported=False)
        self.off_map = {OFF_MAP: Actor(OFF_MAP, self.other_world, self)}
        self.selection: List[Actor] = []
        # Content: package path -> class name, and assets the importer wrote.
        self.content: Dict[str, str] = {
            "/Game/Maps/Main": "World", "/Game/Maps/Other": "World",
            "/Engine/BasicShapes/BasicShapeMaterial": "Material", "/Game/Materials/MI_Red": "MaterialInstanceConstant",
            "/Engine/BasicShapes/Cube": "StaticMesh", "/Engine/BasicShapes/Sphere": "StaticMesh",
            "/Engine/BasicShapes/Cube_Redirect": "ObjectRedirector", "/Game/Props/SM_Cube": "StaticMesh",
            "/Game/Props/Big/SM_CubeLarge": "StaticMesh", "/Game/Textures/T_Cube": "Texture2D",
        }
        self.tags: Dict[str, Dict[str, str]] = {
            "/Engine/BasicShapes/Cube": {"Triangles": "48", "LODs": "1", "NaniteEnabled": "False",
                                         "CollisionPrims": "1", "StaticMaterials": "WorldGridMaterial"},
        }
        self.replaced_assets: List[str] = []
        # The project's saved settings (EditorPerProjectUserSettings.ini) and each write to them.
        self.saved_config = dict(UE_DEFAULTS)
        self.config_writes: List[Dict[str, Any]] = []
        self.stages: Dict[str, List[str]] = {}
        self.export_tasks: List[AssetExportTask] = []
        self.import_tasks: List[AssetImportTask] = []
        self.import_options: List[Any] = []  # what each import ran with; the script clears task.options
        self.import_log: List[str] = []  # lines the next import logs
        self.duplicate_new_actors = False  # UE 5.7 listed an imported actor twice
        self.export_cdo = SimpleNamespace(inner=Inner())
        self.log_path = tmp_path / "Fake.log"
        self.log_path.write_text("")
        # The files each imported asset's AssetImportData names (extract_filenames).
        self.import_sources: Dict[str, List[str]] = {}
        # Packages outside /Game the next import leaves dirty, as Interchange did its shared material.
        self.import_dirties: List[str] = []
        # The console namespace Remote Control runs every script in.
        self.console: Dict[str, Any] = {}
        self.scripts: List[str] = []

    # -- what the editor does -------------------------------------------------

    def log(self, category: str, verbosity: str, message: str) -> None:
        with open(self.log_path, "a") as handle:
            handle.write(f"{LOG_TIME}{category}: {verbosity}: {message}\n")

    def log_python(self, message: str) -> None:
        """unreal.log: a LogPython line, at Display verbosity."""
        with open(self.log_path, "a") as handle:
            handle.write(f"{LOG_TIME}LogPython: {message}\n")

    def level_actors(self) -> List[Actor]:
        actors = list(self.actors.values())
        if self.duplicate_new_actors:
            actors += [actor for actor in actors if getattr(actor, "imported", False)]
        return actors

    def find_object(self, outer: Any, path: str) -> Any:
        return self.actors.get(path)

    def load_object(self, outer: Any, path: str) -> Any:
        # Loading an actor of a map that is not open loads that map.
        if path in self.off_map:
            self.maps_loaded.append(path.split(":", 1)[0])
            return self.off_map[path]
        return self.actors.get(path)

    def load_asset(self, path: str) -> Any:
        package = path.split(".", 1)[0]
        kind = self.content.get(package)
        if kind == "World":
            self.maps_loaded.append(path)
            return World(path)
        if package in self.import_sources:
            sources = list(self.import_sources[package])
            imported = UObject(package + "." + package.rsplit("/", 1)[1])
            data = SimpleNamespace(extract_filenames=lambda: sources)
            imported.get_editor_property = lambda name: data  # type: ignore[attr-defined]
            return imported
        if kind == "StaticMesh":
            return StaticMesh(package + "." + package.rsplit("/", 1)[1])
        return {BASIC: self.basic, RED: self.red}.get(package)

    def run_export(self, task: AssetExportTask) -> bool:
        """LevelExporterUSD: the selected actors of the task's world that it exports, plus a mesh file each."""
        self.export_tasks.append(task)
        exported = [actor for actor in self.selection if actor.get_world() is task.object and actor.exported]
        self.stages[task.filename] = ["/Root"] + ["/Root/" + actor.get_actor_label() for actor in exported]
        with open(task.filename, "w") as handle:
            handle.write("#usda 1.0\n")
        folder = task.options.inner.asset_folder.path
        for actor in exported:
            with open(folder + "/" + actor.get_actor_label() + "_Mesh.usda", "w") as handle:
                handle.write("#usda 1.0\n")
        if task.options.inner.export_sublayers:
            beside = os.path.dirname(task.filename)
            names = {"MaterialOverrides"} | {actor.level.get_outer().get_name() for actor in exported
                                             if actor.level.get_outer() is not task.object}
            for name in names:
                with open(beside + "/" + name + ".usda", "w") as handle:
                    handle.write("#usda 1.0\n")
        return True

    def run_import(self, tasks: List[AssetImportTask]) -> None:
        self.dirty.update(self.import_dirties)
        for task in tasks:
            self.import_tasks.append(task)
            for line in self.import_log:
                self.log("LogUsd", "Warning", line)
            if task.factory is None:
                self.run_interchange(task)
            else:
                self.run_usd(task)

    def run_usd(self, task: AssetImportTask) -> None:
        """The USD stage importer, with its existing-asset and existing-actor policies."""
        options = task.options
        self.import_options.append(options)
        folder = task.destination_path + "/" + task.destination_name
        names = ["SM_Root"] if options.use_prim_kinds_for_collapsing else ["SM_Cube", "SM_Sphere"]
        task.imported_object_paths = []
        for name in names:
            mesh = folder + "/StaticMeshes/" + name
            if mesh in self.content:
                if options.existing_asset_policy == "REPLACE":
                    self.replaced_assets.append(mesh)
                elif options.existing_asset_policy == "APPEND":
                    mesh += "_1"
            self.content[mesh] = "StaticMesh"
            task.imported_object_paths.append(mesh + "." + mesh.rsplit("/", 1)[1])
        if options.import_actors:
            label = task.destination_name
            existing = [path for path, actor in self.actors.items() if actor.get_actor_label() == label]
            if existing and options.existing_actor_policy == "REPLACE":
                for path in existing:
                    del self.actors[path]
            scene = Actor(LEVEL + label + "_" + str(len(self.import_tasks)), self.world, self)
            scene.get_actor_label = lambda label=label: label  # type: ignore[method-assign]
            scene.imported = True  # type: ignore[attr-defined]
            self.actors[scene.path] = scene
            task.factory.scene_actor = scene

    def run_interchange(self, task: AssetImportTask) -> None:
        """Interchange: a mesh named after the file, its material and a texture.

        An existing asset is kept, with a log line, when replace_existing is
        false. When it is true, an existing mesh named after the file turns
        the import into a reimport of that mesh alone (UE 5.7's
        convert-to-reimport), which keeps the earlier material and texture
        without a word.
        """
        stem = task.filename.rsplit("/", 1)[1].rsplit(".", 1)[0]
        if stem == "garbage":
            self.log("LogInterchangeEngine", "Error", "Cannot open FBX file '" + task.filename + "'.")
            task.imported_object_paths = []
            return
        mesh = task.destination_path + "/" + stem
        material = task.destination_path + "/M_" + stem
        texture = task.destination_path + "/T_" + stem
        reimport = task.replace_existing and mesh in self.content
        task.imported_object_paths = []
        made = [(material, "MaterialInstanceConstant", task.filename), (mesh, "StaticMesh", task.filename)]
        if os.path.exists(self.texture_source(task.filename)):
            made.insert(0, (texture, "Texture2D", self.texture_source(task.filename)))
        for path, kind, source in made:
            if reimport and path != mesh and path in self.content:
                continue
            if path in self.content and not task.replace_existing:
                self.log("LogInterchangeEngine", "Warning", "The option bReplaceExisting is false, we are not "
                         f"overriding the asset named '{kind} {path}.{path.rsplit('/', 1)[1]}'.")
                continue
            if path in self.content:
                self.replaced_assets.append(path)
            self.content[path] = kind
            self.import_sources[path] = [source]
            # UAssetImportData's registry tag, which lists the source files relative to the package.
            self.tags[path] = {"AssetImportData": json.dumps([{"RelativeFilename": "../../" + source}])}
            task.imported_object_paths.append(path + "." + path.rsplit("/", 1)[1])

    @staticmethod
    def texture_source(model: str) -> str:
        """The texture Interchange reads for a model, when it is there: the image named after it."""
        return model.rsplit(".", 1)[0] + ".png"

    def create_asset(self, name: str, folder: str, cls: type, factory: Any) -> Any:
        package = folder + "/" + name
        self.content[package] = cls.__name__
        return cls(package + "." + name)

    def registry_assets(self) -> List[AssetData]:
        return [AssetData(package, kind, self.tags.get(package)) for package, kind in self.content.items()]

    def registry_filter(self, ar_filter: Any) -> List[AssetData]:
        found = []
        for asset in self.registry_assets():
            if ar_filter.class_paths:
                kind = asset.asset_class_path.asset_name
                if not any(path.package_name + "." + path.asset_name == "/Script/Engine." + kind
                           or (ar_filter.recursive_classes
                               and kind in SUBCLASSES.get(path.package_name + "." + path.asset_name, ()))
                           for path in ar_filter.class_paths):
                    continue
            if ar_filter.package_paths:
                if not any(asset.package_path == folder
                           or (ar_filter.recursive_paths and asset.package_path.startswith(folder + "/"))
                           for folder in ar_filter.package_paths):
                    continue
            found.append(asset)
        return found

    def dirty_packages(self, maps: bool) -> List[UObject]:
        return [UObject(name) for name in sorted(self.dirty) if (self.content.get(name) == "World") == maps]

    def save_packages(self, packages: List[UObject], only_dirty: bool) -> bool:
        for package in packages:
            name = package.get_name()
            if name not in self.unsaveable:
                self.saved_packages.append(name)
                self.dirty.discard(name)
        return not any(package.get_name() in self.unsaveable for package in packages)

    # -- the unreal module -----------------------------------------------------

    def module(self) -> SimpleNamespace:
        editor = self

        @contextmanager
        def scoped_transaction(description: str) -> Iterator[None]:
            editor.transactions.append(description)
            editor.open_transactions += 1
            try:
                yield
            finally:
                editor.open_transactions -= 1

        class UsdStageImportOptions:
            """A fresh object starts from the saved settings; a notified change saves them all."""

            def __init__(self) -> None:
                self.__dict__.update(editor.saved_config)
                self.__dict__.update(NOT_SAVED)

            def set_editor_property(self, name: str, value: Any, notify_mode: str = NOTIFY.DEFAULT) -> None:
                if name not in UE_DEFAULTS and name not in NOT_SAVED:
                    raise Exception(
                        f"UsdStageImportOptions: Failed to find property '{name}' for attribute '{name}' "
                        "on 'UsdStageImportOptions'"
                    )
                setattr(self, name, value)
                if notify_mode != NOTIFY.NEVER:
                    # PostEditChangeProperty -> SaveConfig: every config property, to the ini and the CDO.
                    editor.saved_config = {key: getattr(self, key) for key in UE_DEFAULTS}
                    editor.config_writes.append(dict(editor.saved_config))

            def get_editor_property(self, name: str) -> Any:
                return getattr(self, name)

        class UsdStageOptions:
            def set_editor_property(self, name: str, value: Any) -> None:
                setattr(self, name, value)

        class SavedImportOptions:
            def get_editor_property(self, name: str) -> Any:
                return editor.saved_config[name]

        class LevelExporterUSDOptions:
            pass

        class InterchangeProjectSettings:
            pass

        def stack(*pipelines: str) -> SimpleNamespace:
            soft = [SimpleNamespace(export_text=lambda path=path: path) for path in pipelines]
            return SimpleNamespace(get_editor_property=lambda name: soft)

        interchange_settings = SimpleNamespace(get_editor_property=lambda name: {
            "content_import_settings": SimpleNamespace(get_editor_property=lambda key: {
                "pipeline_stacks": {"Assets": stack(ASSETS_PIPELINE)},
                "default_pipeline_stack": "Assets"}[key]),
            "scene_import_settings": SimpleNamespace(get_editor_property=lambda key: {
                "pipeline_stacks": {"Scene": stack("/Interchange/Pipelines/DefaultSceneAssetsPipeline.A",
                                                   "/Interchange/Pipelines/DefaultSceneLevelPipeline.B")},
                "default_pipeline_stack": "Scene"}[key]),
        }[name])

        def get_default_object(cls: type) -> Any:
            if cls is UsdStageImportOptions:
                return SavedImportOptions()
            if cls is LevelExporterUSDOptions:
                return editor.export_cdo
            if cls is InterchangeProjectSettings:
                return interchange_settings
            raise AssertionError(cls)

        subsystems = {
            "UnrealEditorSubsystem": SimpleNamespace(
                get_editor_world=lambda: None if editor.pie else editor.world,
                get_game_world=lambda: World("/Game/Maps/UEDPIE_0_Main.Main") if editor.pie else None,
            ),
            "LayersSubsystem": SimpleNamespace(get_world=lambda: editor.world),
            "LevelEditorSubsystem": SimpleNamespace(is_in_play_in_editor=lambda: editor.pie),
            "EditorActorSubsystem": SimpleNamespace(
                get_all_level_actors=editor.level_actors,
                get_selected_level_actors=lambda: list(editor.selection),
                set_selected_level_actors=lambda actors: setattr(editor, "selection", list(actors)),
            ),
        }

        def asset_registry() -> SimpleNamespace:
            def by_path(folder: str, recursive: bool = False) -> List[AssetData]:
                return [asset for asset in editor.registry_assets() if asset.package_name.startswith(folder + "/")]

            def by_package(package: str) -> List[AssetData]:
                return [asset for asset in editor.registry_assets() if asset.package_name == package]

            return SimpleNamespace(get_assets_by_path=by_path, get_assets_by_package_name=by_package,
                                   get_all_assets=editor.registry_assets, get_assets=editor.registry_filter)

        formats = {"ASSETS": ["fbx;Filmbox", "obj;OBJ File Format"], "SCENES": ["fbx;Filmbox"], "ACTORS": []}
        names = {name: type(name, (), {}) for name in subsystems}
        return SimpleNamespace(
            Actor=Actor,
            MeshComponent=MeshComponent,
            DynamicMeshComponent=DynamicMeshComponent,
            MaterialInterface=MaterialInterface,
            Material=Material,
            StaticMesh=StaticMesh,
            AssetExportTask=AssetExportTask,
            AssetImportTask=AssetImportTask,
            LevelExporterUSD=LevelExporterUSD,
            LevelExporterUSDOptions=LevelExporterUSDOptions,
            UsdStageImportFactory=UsdStageImportFactory,
            UsdStageImportOptions=UsdStageImportOptions,
            UsdStageOptions=UsdStageOptions,
            UsdCollisionType=COLLISION,
            UsdUpAxis=SimpleNamespace(Y_AXIS="Y_AXIS", Z_AXIS="Z_AXIS"),
            PropertyAccessChangeNotifyMode=NOTIFY,
            ReplaceActorPolicy=SimpleNamespace(APPEND="APPEND", REPLACE="REPLACE", IGNORE="IGNORE"),
            ReplaceAssetPolicy=SimpleNamespace(APPEND="APPEND", REPLACE="REPLACE", IGNORE="IGNORE"),
            DirectoryPath=lambda path: SimpleNamespace(path=path),
            Exporter=SimpleNamespace(run_asset_export_task=editor.run_export),
            AssetToolsHelpers=SimpleNamespace(
                get_asset_tools=lambda: SimpleNamespace(
                    import_asset_tasks=editor.run_import, create_asset=editor.create_asset
                )
            ),
            EditorAssetLibrary=SimpleNamespace(
                load_asset=editor.load_asset,
                does_asset_exist=lambda path: path.split(".", 1)[0] in editor.content,
                save_loaded_asset=lambda asset: editor.save_ok,
            ),
            MaterialEditingLibrary=SimpleNamespace(
                set_material_instance_parent=lambda instance, parent: setattr(instance, "parent", parent)
            ),
            MaterialInstanceConstant=MaterialInstanceConstant,
            MaterialInstanceConstantFactoryNew=SimpleNamespace,
            AssetRegistryHelpers=SimpleNamespace(get_asset_registry=asset_registry),
            ARFilter=lambda **fields: SimpleNamespace(**{
                "class_paths": [], "package_paths": [], "recursive_paths": False, "recursive_classes": False, **fields,
            }),
            TopLevelAssetPath=lambda package, asset: SimpleNamespace(package_name=package, asset_name=asset),
            UsdConversionLibrary=SimpleNamespace(
                get_prim_path_for_object=lambda component, parent, folders, root: (
                    "/" + root + "/" + component.owner.get_actor_label()
                )
            ),
            EditorLoadingAndSavingUtils=SimpleNamespace(
                get_dirty_map_packages=lambda: editor.dirty_packages(True),
                get_dirty_content_packages=lambda: editor.dirty_packages(False),
                save_packages=editor.save_packages,
            ),
            InterchangeManager=SimpleNamespace(get_interchange_manager_scripted=lambda: SimpleNamespace(
                get_supported_formats=lambda kind: formats[kind])),
            InterchangeTranslatorType=SimpleNamespace(ASSETS="ASSETS", SCENES="SCENES", ACTORS="ACTORS"),
            InterchangeProjectSettings=InterchangeProjectSettings,
            Paths=SimpleNamespace(project_log_dir=lambda: str(editor.tmp_path) + "/",
                                  convert_relative_path_to_full=lambda path: (
                                      path if os.path.isabs(path) else os.path.join(ENGINE_BINARIES, path))),
            log=editor.log_python,
            SystemLibrary=SimpleNamespace(
                is_valid=lambda obj: obj is not None,
                get_game_name=lambda: "Fake",
                get_engine_version=lambda: "5.7.4-fake",
                execute_console_command=lambda world, command: editor.console_commands.append(command),
            ),
            ScopedEditorTransaction=scoped_transaction,
            get_default_object=get_default_object,
            get_editor_subsystem=lambda cls: subsystems[cls.__name__],
            find_object=editor.find_object,
            load_object=editor.load_object,
            load_asset=editor.load_asset,
            **names,
        )

    def pxr(self) -> SimpleNamespace:
        editor = self
        return SimpleNamespace(
            Usd=SimpleNamespace(Stage=SimpleNamespace(
                LoadNone="LoadNone", Open=lambda path, load=None: Stage(editor.stages.get(path, [])),
            )),
            Tf=SimpleNamespace(MakeValidIdentifier=lambda name: "".join(
                char if char.isalnum() or char == "_" else "_" for char in name)),
        )

    def leaked(self) -> List[Any]:
        """Objects referencing the world, or a scene actor, left in the console namespace."""
        return [
            value for value in self.console.values()
            if isinstance(value, (AssetExportTask, AssetImportTask, UsdStageImportFactory))
        ]


@pytest.fixture
def editor(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> SimpleNamespace:
    """A session whose scripts run in-process against the fake editor, in one console namespace."""
    fake = Editor(tmp_path)
    monkeypatch.setitem(sys.modules, "unreal", fake.module())
    monkeypatch.setitem(sys.modules, "pxr", fake.pxr())
    (tmp_path / "scene.usda").write_text("#usda 1.0\n")
    base = Settings()
    settings = base.model_copy(update={
        "security": base.security.model_copy(update={"allowed_paths": [str(tmp_path)]}),
    })
    session = UnrealRuntimeSession(settings)
    timeouts: List[Optional[float]] = []

    async def execute(code: str, mode: str = "ExecuteFile", timeout: Optional[float] = None) -> Dict[str, Any]:
        fake.scripts.append(code)
        timeouts.append(timeout)
        out = io.StringIO()
        try:
            with redirect_stdout(out):
                exec(code, fake.console)
        except Exception as exc:  # UE reports a raised script this way
            return {"ReturnValue": False, "CommandResult": f"{type(exc).__name__}: {exc}"}
        return {
            "ReturnValue": True,
            "LogOutput": [{"Type": "Info", "Output": line} for line in out.getvalue().splitlines()],
        }

    monkeypatch.setattr(session, "_execute_python", execute)
    return SimpleNamespace(session=session, fake=fake, timeouts=timeouts, tmp_path=tmp_path)


def _run(coro: Any) -> Dict[str, Any]:
    return asyncio.run(coro)


def _script_args(script: str) -> Dict[str, Any]:
    """The ``args = {...}`` a tool script starts with."""
    first = script.split("\n", 1)[0]
    assert first.startswith("args = "), first
    return ast.literal_eval(first[len("args = "):])


# ---------------------------------------------------------------------------
# search_unreal_assets: the result count is bounded
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("max_results", [-1, 0, MAX_ASSET_SEARCH_RESULTS + 1, True, 2.5])
def test_an_out_of_range_search_limit_is_refused_before_any_request(max_results: Any) -> None:
    """Remote Control sized its result array to Limit, and -1 killed the editor."""
    session = UnrealRuntimeSession(Settings())
    session._http_put = AsyncMock()  # type: ignore[method-assign]

    result = _run(session.search_assets(query="Cube", max_results=max_results))

    assert result["success"] is False
    assert result["error_type"] == "ValidationError"
    assert f"from 1 to {MAX_ASSET_SEARCH_RESULTS}" in result["error"]
    session._http_put.assert_not_called()


def test_the_search_limit_bound_is_in_the_tool_schema() -> None:
    assert _unreal_tools.MAX_ASSET_SEARCH_RESULTS == MAX_ASSET_SEARCH_RESULTS
    spec = {tool.name: tool for tool in _unreal_tools.UNREAL_TOOLS}["search_unreal_assets"]
    model = _unreal_tools_request_model(spec)
    schema = model.model_json_schema()["properties"]["max_results"]
    assert (schema["minimum"], schema["maximum"]) == (1, MAX_ASSET_SEARCH_RESULTS)


def _unreal_tools_request_model(spec: Any) -> Any:
    from simul.mcp.registration._reg_unreal import request_model_for

    return request_model_for(spec)


class _EditorAdapter(AvailableAdapter):
    def __init__(self, session: Any) -> None:
        super().__init__(Settings())
        self.session = session

    @contextmanager
    def create_session(self) -> Iterator[Any]:
        yield self.session


def _call_tool(session: Any, monkeypatch: pytest.MonkeyPatch, tool: str, arguments: Dict[str, Any]) -> Any:
    """What an agent receives: the call through the real MCP server and its response model."""
    base = Settings()
    settings = base.model_copy(update={"unreal": base.unreal.model_copy(update={"tool_surface": "full"})})
    instance = server_module.SimulMCPServer(settings, backends={"unreal"})
    monkeypatch.setattr(instance, "unreal_adapter", _EditorAdapter(session))

    async def run() -> Any:
        async with Client(instance.mcp) as client:
            return await client.call_tool(tool, arguments, raise_on_error=False)

    return asyncio.run(run())


def test_an_mcp_call_with_a_negative_limit_never_reaches_the_editor(monkeypatch: pytest.MonkeyPatch) -> None:
    """FastMCP refuses the argument against the schema; the session is never called."""
    session = SimpleNamespace(search_assets=AsyncMock())

    result = _call_tool(session, monkeypatch, "search_unreal_assets", {"query": "Cube", "max_results": -1})

    assert result.is_error
    session.search_assets.assert_not_called()


# ---------------------------------------------------------------------------
# export_unreal_usd: nothing keeps the world alive; only actors in this level
# ---------------------------------------------------------------------------


def test_an_export_leaves_no_task_referencing_the_world_in_the_console(editor: SimpleNamespace) -> None:
    """The crash: Old World not cleaned up by GC, held by AssetExportTask::Object."""
    target = str(editor.tmp_path / "out" / "cube.usda")

    result = _run(editor.session.export_usd([CUBE], target))

    assert result["actors_exported"] == 1, result
    assert editor.fake.leaked() == []
    (task,) = editor.fake.export_tasks
    assert (task.object, task.exporter, task.options) == (None, None, None)


def test_an_export_restores_the_selection_and_the_exporter_options(editor: SimpleNamespace) -> None:
    editor.fake.selection = [editor.fake.actors[SPHERE]]
    before = dict(editor.fake.export_cdo.inner.__dict__)

    _run(editor.session.export_usd([CUBE], str(editor.tmp_path / "cube.usda"), {"export_sublayers": True}))

    assert editor.fake.selection == [editor.fake.actors[SPHERE]]
    assert editor.fake.export_cdo.inner.__dict__ == before


def test_an_export_reports_the_prims_the_stage_has_for_each_actor(editor: SimpleNamespace) -> None:
    """Unreal's exporter drops editor helpers silently; actors_exported counts what was written."""
    result = _run(editor.session.export_usd([CUBE, SETTINGS_ACTOR], str(editor.tmp_path / "two.usda")))

    assert result["actors_exported"] == 1
    assert result["exported_prims"] == [{"actor_path": CUBE, "prim_path": "/Root/Cube"}]
    assert result["skipped_actors"] == [SETTINGS_ACTOR]
    assert UnrealExportUsdResponse(**result).model_dump(exclude={"success", "error"}) == {
        key: result[key] for key in UnrealExportUsdResponse.model_fields if key not in ("success", "error")
    }


def test_an_export_that_writes_no_actor_fails(editor: SimpleNamespace) -> None:
    result = _run(editor.session.export_usd([SETTINGS_ACTOR], str(editor.tmp_path / "none.usda")))

    assert result["success"] is False
    assert result["error_type"] == "EngineError"
    assert result["actors_exported"] == 0
    assert SETTINGS_ACTOR in result["error"]


def test_an_actor_of_an_unopened_map_is_refused_without_loading_the_map(editor: SimpleNamespace) -> None:
    """It used to load L_Other, export nothing and report actors_exported 1."""
    target = editor.tmp_path / "other.usda"

    result = _run(editor.session.export_usd([OFF_MAP], str(target)))

    assert result["success"] is False
    assert f"Actor not found: {OFF_MAP}" in result["error"]
    assert editor.fake.maps_loaded == []
    assert not target.exists()
    assert editor.fake.export_tasks == []


# ---------------------------------------------------------------------------
# import_unreal_usd: nothing replaced unless asked, nothing saved to config
# ---------------------------------------------------------------------------


def _import(editor: SimpleNamespace, **kwargs: Any) -> Dict[str, Any]:
    kwargs.setdefault("usd_path", str(editor.tmp_path / "scene.usda"))
    kwargs.setdefault("destination_path", "/Game/Imports")
    return _run(editor.session.import_usd(**kwargs))


def test_a_usd_file_referencing_a_layer_outside_the_sandbox_is_refused(
    editor: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The review's fixture: references = @../../outside/ref.usda@ imported the outside layer's prims."""
    sbx, outside = _confine(editor)
    (outside / "ref.usda").write_text('#usda 1.0\ndef Xform "Secret" {}\n')
    (sbx / "usd").mkdir()
    stage = sbx / "usd" / "refin.usda"
    stage.write_text('#usda 1.0\ndef Xform "Root" (\n    references = @../../outside/ref.usda@\n) {}\n')

    payload = json.loads(_call_tool(editor.session, monkeypatch, "import_unreal_usd", {
        "usd_path": str(stage), "destination_path": "/Game/Imports",
    }).content[-1].text)

    assert payload["error_type"] == "SandboxError", payload
    (entry,) = payload["details"]["outside_sandbox"]
    assert (entry["kind"], entry["asset_path"]) == ("reference", "../../outside/ref.usda")
    assert payload["details"]["references_outside"] == [os.path.realpath(outside / "ref.usda")]
    assert payload["details"]["referenced_by"] == str(stage)
    assert editor.fake.scripts == [] and editor.fake.import_tasks == []


def test_a_usd_file_whose_texture_is_outside_is_refused(editor: SimpleNamespace) -> None:
    sbx, outside = _confine(editor)
    stage = sbx / "tex.usda"
    stage.write_text(
        '#usda 1.0\ndef Shader "Tex" {\n    asset inputs:file = @../outside/secret.png@\n}\n'
    )

    with pytest.raises(unreal_runtime.SandboxDenied) as denied:
        _import(editor, usd_path=str(stage))

    assert denied.value.details["references_outside"] == [os.path.realpath(outside / "secret.png")]
    assert editor.fake.scripts == []


def test_a_usd_file_drawing_only_on_the_sandbox_imports(editor: SimpleNamespace) -> None:
    sbx, _ = _confine(editor)
    (sbx / "part.usda").write_text('#usda 1.0\ndef Xform "Part" {}\n')
    stage = sbx / "scene.usda"
    stage.write_text('#usda 1.0\n(\n    subLayers = [@./part.usda@]\n)\n')

    result = _import(editor, usd_path=str(stage))

    assert result["imported_assets"], result


UNREAL_EXPORT = (
    '#usda 1.0\ndef Xform "Root" {{\n    def Material "M" {{\n'
    '        uniform asset info:unreal:sourceAsset = @/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial@\n'
    "{extra}    }}\n}}\n"
)


def test_unreal_content_paths_in_its_own_exports_are_not_files_outside_the_sandbox(editor: SimpleNamespace) -> None:
    """Unreal's exporter names each mesh's source material by content path; the round trip was refused."""
    sbx, _ = _confine(editor)
    stage = sbx / "export.usda"
    stage.write_text(UNREAL_EXPORT.format(extra=""))

    result = _import(editor, usd_path=str(stage))

    assert result["imported_assets"], result


@pytest.mark.parametrize(
    "authored",
    [
        "asset inputs:file = @/Engine/x.usda@",  # a file name under a content root's name
        "asset inputs:file = @/Engine/usda.usda@",  # Name.Name, but the suffix is a file extension
        "asset inputs:file = @/Engine/BasicShapes/Cube.Sphere@",  # not a package and its own asset
        "asset inputs:file = @/MyPlugin/Materials/M_Wood.M_Wood@",  # only /Engine, /Game and /Script
        "asset inputs:file = @../outside/BasicShapeMaterial.BasicShapeMaterial@",  # relative
        "asset inputs:file = @../../Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial@",  # relative
    ],
)
def test_only_unreal_object_paths_pass_as_content(editor: SimpleNamespace, authored: str) -> None:
    """The exemption is narrow: anything else naming a place outside the sandbox is refused."""
    sbx, _ = _confine(editor)
    stage = sbx / "export.usda"
    stage.write_text(UNREAL_EXPORT.format(extra="        " + authored + "\n"))

    with pytest.raises(unreal_runtime.SandboxDenied) as denied:
        _import(editor, usd_path=str(stage))

    assert [entry["asset_path"] for entry in denied.value.details["outside_sandbox"]] == [
        authored.split("@")[1]
    ]
    assert editor.fake.scripts == []


def test_a_layer_named_like_an_unreal_object_path_is_still_refused(editor: SimpleNamespace) -> None:
    """A reference is a file read whatever its shape."""
    sbx, _ = _confine(editor)
    stage = sbx / "export.usda"
    stage.write_text('#usda 1.0\ndef "Root" (\n    references = @/Engine/Ref.Ref@\n) {}\n')

    with pytest.raises(unreal_runtime.SandboxDenied) as denied:
        _import(editor, usd_path=str(stage))

    assert [entry["kind"] for entry in denied.value.details["outside_sandbox"]] == ["reference"]


def test_a_file_outside_beside_unreal_content_paths_is_still_refused(editor: SimpleNamespace) -> None:
    sbx, outside = _confine(editor)
    stage = sbx / "export.usda"
    stage.write_text(UNREAL_EXPORT.format(extra="        asset inputs:file = @../outside/secret.png@\n"))

    with pytest.raises(unreal_runtime.SandboxDenied) as denied:
        _import(editor, usd_path=str(stage))

    assert [entry["asset_path"] for entry in denied.value.details["outside_sandbox"]] == ["../outside/secret.png"]
    assert denied.value.details["outside_count"] == 1
    assert denied.value.error_message.endswith(
        "draws on 1 file(s) outside the sandbox: " + os.path.realpath(outside / "secret.png")
    )
    assert editor.fake.scripts == []


def test_a_usd_file_pxr_cannot_open_is_not_vouched_for(editor: SimpleNamespace) -> None:
    sbx, _ = _confine(editor)
    (sbx / "broken.usda").write_text("#usda 1.0\ndef Xform \"Root\" (\n")

    with pytest.raises(unreal_runtime.SandboxDenied, match="could not be checked against the sandbox"):
        _import(editor, usd_path=str(sbx / "broken.usda"))

    assert editor.fake.scripts == []


def test_the_usd_import_reports_engine_packages_it_dirtied(editor: SimpleNamespace) -> None:
    editor.fake.import_dirties = ["/InterchangeAssets/Materials/PBRSurfaceMaterial_MR"]

    result = _import(editor)

    assert result["dirty_engine_packages"] == ["/InterchangeAssets/Materials/PBRSurfaceMaterial_MR"]
    assert "Save All" in result["warnings"][-1]


def test_an_import_saves_none_of_its_options_to_the_project_config(editor: SimpleNamespace) -> None:
    """set_editor_property notified UsdStageImportOptions, which saved bImportActors=False."""
    result = _import(editor, import_options={"import_actors": False})

    assert result["actor_paths"] == [], result
    assert editor.fake.config_writes == []
    assert editor.fake.saved_config == UE_DEFAULTS
    later = _import(editor, destination_path="/Game/Later")
    assert len(later["actor_paths"]) == 1  # a later default import still spawns actors


def test_the_documented_options_are_set_on_every_import_whatever_the_saved_settings(
    editor: SimpleNamespace,
) -> None:
    """A fresh options object starts from the saved settings, which an older simul could have changed."""
    editor.fake.saved_config["import_actors"] = False

    result = _import(editor)

    (options,) = editor.fake.import_options
    assert (options.import_actors, options.import_geometry, options.import_materials) == (True, True, True)
    assert len(result["actor_paths"]) == 1
    assert editor.fake.saved_config["import_actors"] is False  # reported, left alone
    assert any("bImportActors=False" in warning and "import_actors=false" in warning
               for warning in result["warnings"]), result["warnings"]


def test_an_import_into_a_folder_that_holds_assets_is_refused_without_replace(
    editor: SimpleNamespace,
) -> None:
    """replace_existing=False was ignored and the importer replaced the earlier import."""
    first = _import(editor)
    actors_before = dict(editor.fake.actors)

    again = _import(editor)

    assert again["success"] is False
    assert again["error_type"] == "InvalidState"
    assert again["details"]["existing_assets"] == ["/Game/Imports/scene/StaticMeshes/SM_Root"]
    assert "replace_existing=true" in again["details"]["hint"]
    assert len(editor.fake.import_tasks) == 1
    assert editor.fake.replaced_assets == []
    assert editor.fake.actors == actors_before
    assert first["actor_paths"][0] in editor.fake.actors


def test_an_import_without_replace_keeps_the_actors_an_earlier_import_spawned(
    editor: SimpleNamespace,
) -> None:
    first = _import(editor)

    second = _import(editor, destination_path="/Game/Elsewhere")

    (first_actor,) = first["actor_paths"]
    assert first_actor in editor.fake.actors
    assert second["removed_actor_paths"] == []
    options = editor.fake.import_options[-1]
    assert (options.existing_actor_policy, options.existing_asset_policy) == ("APPEND", "IGNORE")


def test_replace_existing_replaces_the_assets_and_the_earlier_actors_and_says_so(
    editor: SimpleNamespace,
) -> None:
    first = _import(editor)

    replaced = _import(editor, replace_existing=True)

    assert replaced.get("success", True) is not False, replaced
    assert editor.fake.replaced_assets == ["/Game/Imports/scene/StaticMeshes/SM_Root"]
    assert replaced["removed_actor_paths"] == first["actor_paths"]
    options = editor.fake.import_options[-1]
    assert (options.existing_actor_policy, options.existing_asset_policy) == ("REPLACE", "REPLACE")
    payload = UnrealImportUsdResponse(**replaced).model_dump()
    assert payload["removed_actor_paths"] == first["actor_paths"]


def test_an_import_leaves_no_factory_holding_its_scene_actor_in_the_console(
    editor: SimpleNamespace,
) -> None:
    _import(editor)

    assert editor.fake.leaked() == []
    (task,) = editor.fake.import_tasks
    assert (task.factory, task.options) == (None, None)


def test_the_import_folder_is_the_file_name_as_unreal_names_assets(editor: SimpleNamespace) -> None:
    usd = editor.tmp_path / "my robot.v2.usda"
    usd.write_text("#usda 1.0\n")

    _import(editor, usd_path=str(usd))

    assert _script_args(editor.fake.scripts[-1])["name"] == "my_robot_v2"
    assert editor.fake.import_tasks[-1].destination_name == "my_robot_v2"


def test_import_usd_is_annotated_destructive() -> None:
    spec = {tool.name: tool for tool in _unreal_tools.UNREAL_TOOLS}["import_unreal_usd"]
    assert spec.destructive is True
    assert [param.name for param in spec.params][-1] == "replace_existing"


# ---------------------------------------------------------------------------
# Timeouts: the import and export tools wait longer and say the work may finish
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["import", "export"])
def test_import_and_export_wait_the_import_export_timeout(editor: SimpleNamespace, method: str) -> None:
    if method == "import":
        _import(editor)
    else:
        _run(editor.session.export_usd([CUBE], str(editor.tmp_path / "cube.usda")))

    assert editor.timeouts == [float(Settings().unreal.import_export_timeout)]


def test_the_import_export_timeout_is_never_shorter_than_the_request_timeout() -> None:
    session = UnrealRuntimeSession(Settings(unreal={"timeout": 900, "import_export_timeout": 60}))
    assert session._import_export_timeout() == 900.0


@pytest.mark.parametrize("method", ["import", "export"])
def test_a_timed_out_import_or_export_says_the_work_may_still_complete(
    editor: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    async def slow(code: str, mode: str = "ExecuteFile", timeout: Optional[float] = None) -> Dict[str, Any]:
        raise asyncio.TimeoutError()

    monkeypatch.setattr(editor.session, "_execute_python", slow)
    target = str(editor.tmp_path / "late.usda")
    if method == "import":
        result = _import(editor)
    else:
        result = _run(editor.session.export_usd([CUBE], target))

    assert result["success"] is False
    assert result["error_type"] == "TimeoutError"
    assert "may still complete" in result["error"]
    assert "a retry runs it again" in result["error"]
    assert result["details"]["timeout_seconds"] == 300.0
    assert "UNREAL__IMPORT_EXPORT_TIMEOUT" in result["details"]["hint"]


# ---------------------------------------------------------------------------
# assign_unreal_material: undoable, dirties the level, only this level's actors
# ---------------------------------------------------------------------------


def test_assigning_a_material_records_the_component_in_an_undoable_transaction(
    editor: SimpleNamespace,
) -> None:
    """set_material alone recorded and dirtied nothing: no save, no undo."""
    result = _run(editor.session.assign_material(CUBE, RED))

    component = editor.fake.actors[CUBE].root_component.path
    assert editor.fake.transactions == ["simul: assign material"]
    assert editor.fake.events == [
        ("modify", component, 1),
        ("set_material", 0, RED + ".MI_Red", 1),
    ]
    assert "/Game/Maps/Main" in editor.fake.dirty
    assert result == {
        "actor_path": CUBE,
        "component_path": component,
        "component_name": "StaticMeshComponent0",
        "material_path": RED + ".MI_Red",
        "slot_index": 0,
        "slot_name": "Default",
        "previous_material_path": BASIC + ".BasicShapeMaterial",
    }
    assert UnrealAssignMaterialResponse(**result).model_dump(exclude={"success", "error"}) == result


def test_assigning_to_an_actor_of_an_unopened_map_loads_nothing(editor: SimpleNamespace) -> None:
    """It used to load L_Other and edit its actor, attached mode included."""
    result = _run(editor.session.assign_material(OFF_MAP, RED))

    assert result["success"] is False
    assert f"Actor not found: {OFF_MAP}" in result["error"]
    assert editor.fake.maps_loaded == []
    assert editor.fake.events == []


@pytest.mark.parametrize(
    ("material", "error_type", "error"),
    [
        ("/Game/Maps/Other", "InvalidAssetType", "/Game/Maps/Other is a level, not a material"),
        ("/Engine/BasicShapes/Cube", "InvalidAssetType", "/Engine/BasicShapes/Cube is a StaticMesh, not a material"),
        ("/Game/Nope/M_Nope", "NotFoundError", "No material at '/Game/Nope/M_Nope'"),
    ],
)
def test_a_material_path_that_is_no_material_is_typed_as_in_create_instance(
    editor: SimpleNamespace, material: str, error_type: str, error: str
) -> None:
    """A level given as the material was NotFoundError here and InvalidAssetType in create-instance."""
    result = _run(editor.session.assign_material(CUBE, material))

    assert result["error_type"] == error_type
    assert error in result["error"]
    assert editor.fake.maps_loaded == []  # a level is never loaded as an asset
    assert editor.fake.events == []


def test_an_out_of_range_slot_is_a_validation_error_that_changes_nothing(editor: SimpleNamespace) -> None:
    result = _run(editor.session.assign_material(CUBE, RED, slot_index=3))

    assert result["error_type"] == "ValidationError"
    assert "has 1 material slot(s)" in result["error"]
    assert editor.fake.events == []


# ---------------------------------------------------------------------------
# create_unreal_material_instance: written under /Game unless allowed
# ---------------------------------------------------------------------------


def _created_folder(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> Any:
    session = UnrealRuntimeSession(Settings())
    scripts: List[str] = []

    async def execute(code: str, mode: str = "ExecuteFile") -> Dict[str, Any]:
        scripts.append(code)
        return {"ReturnValue": True, "LogOutput": [{"Type": "Info", "Output": json.dumps({
            "instance_path": "/Game/Materials/MI_X.MI_X", "parent_path": BASIC + ".BasicShapeMaterial",
            "class_name": "MaterialInstanceConstant",
        })}]}

    monkeypatch.setattr(session, "_execute_python", execute)
    result = _run(session.create_material_instance(**kwargs))
    return result, (_script_args(scripts[0])["folder"] if scripts else None)


def test_an_instance_of_an_engine_parent_goes_to_game_materials(monkeypatch: pytest.MonkeyPatch) -> None:
    """It went next to the parent, into the engine install's Engine/Content/BasicShapes."""
    result, folder = _created_folder(monkeypatch, parent_path=BASIC, instance_name="MI_X")

    assert folder == "/Game/Materials"
    assert UnrealCreateMaterialInstanceResponse(**result).class_name == "MaterialInstanceConstant"


def test_an_instance_of_a_project_parent_goes_next_to_it(monkeypatch: pytest.MonkeyPatch) -> None:
    _, folder = _created_folder(monkeypatch, parent_path="/Game/Props/M_Wood", instance_name="MI_Wood")

    assert folder == "/Game/Props"


@pytest.mark.parametrize("save_path", ["/Engine/BasicShapes", "/MyPlugin/Materials", "/Game2/Materials"])
def test_a_folder_outside_game_is_refused_unless_allowed(
    monkeypatch: pytest.MonkeyPatch, save_path: str
) -> None:
    refused, folder = _created_folder(monkeypatch, parent_path=BASIC, instance_name="MI_X", save_path=save_path)

    assert refused["error_type"] == "SandboxError"
    assert refused["details"]["allowed_root"] == "/Game"
    assert folder is None  # nothing sent to the editor

    allowed, folder = _created_folder(
        monkeypatch, parent_path=BASIC, instance_name="MI_X", save_path=save_path, allow_engine_content=True
    )
    assert folder == save_path
    assert allowed["instance_path"]


@pytest.mark.parametrize(
    ("save_path", "instance_name"),
    [("/Game/../Engine", "MI_X"), ("/Game//Materials", "MI_X"), ("Game/Materials", "MI_X"),
     ("/Game/Mat:erials", "MI_X"), ("/Game/Materials", "MI X"), ("/Game/Materials", "")],
)
def test_a_folder_or_name_unreal_would_rewrite_is_a_validation_error(
    monkeypatch: pytest.MonkeyPatch, save_path: str, instance_name: str
) -> None:
    """/Game/../Engine was sanitized into a folder named __ under /Game."""
    result, folder = _created_folder(
        monkeypatch, parent_path=BASIC, instance_name=instance_name, save_path=save_path
    )

    assert result["error_type"] == "ValidationError"
    assert folder is None


def test_the_material_tools_that_write_document_allow_engine_content() -> None:
    specs = {tool.name: tool for tool in _unreal_tools.UNREAL_TOOLS}
    for name in ("create_unreal_material_instance", "set_unreal_material_params"):
        param = {p.name: p for p in specs[name].params}["allow_engine_content"]
        assert param.default is False
        assert "/Engine" in (param.description or "")


def test_unreal_runtime_exports_the_search_bound() -> None:
    assert unreal_runtime.MAX_ASSET_SEARCH_RESULTS == 1000


# ---------------------------------------------------------------------------
# MCP: the response models carry every field the scripts report
# ---------------------------------------------------------------------------


def test_the_export_payload_reaches_the_agent_whole(editor: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _call_tool(editor.session, monkeypatch, "export_unreal_usd", {
        "actor_paths": f"{CUBE},{SETTINGS_ACTOR}", "output_path": str(editor.tmp_path / "two.usda"),
    })

    payload = json.loads(result.content[-1].text)
    assert payload["exported_prims"] == [{"actor_path": CUBE, "prim_path": "/Root/Cube"}]
    assert payload["skipped_actors"] == [SETTINGS_ACTOR]
    assert payload["actors_exported"] == 1


def test_the_import_payload_reaches_the_agent_whole(editor: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    arguments = {"usd_path": str(editor.tmp_path / "scene.usda"), "destination_path": "/Game/Imports"}
    first = json.loads(_call_tool(editor.session, monkeypatch, "import_unreal_usd", arguments).content[-1].text)
    refused = json.loads(_call_tool(editor.session, monkeypatch, "import_unreal_usd", arguments).content[-1].text)
    replaced = json.loads(_call_tool(
        editor.session, monkeypatch, "import_unreal_usd", dict(arguments, replace_existing=True)
    ).content[-1].text)

    assert first["success"] is True and len(first["actor_paths"]) == 1
    assert refused["error_type"] == "InvalidState"
    assert replaced["removed_actor_paths"] == first["actor_paths"]
    assert replaced["warnings"] == []


def test_the_assign_payload_reaches_the_agent_whole(editor: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _call_tool(editor.session, monkeypatch, "assign_unreal_material", {
        "actor_path": CUBE, "material_path": RED,
    })

    payload = json.loads(result.content[-1].text)
    assert payload["component_path"] == CUBE + ".StaticMeshComponent0"
    assert payload["previous_material_path"] == BASIC + ".BasicShapeMaterial"


# ---------------------------------------------------------------------------
# search_unreal_assets: the asset registry, every content root, paged
# ---------------------------------------------------------------------------


def _search(editor: SimpleNamespace, **kwargs: Any) -> Dict[str, Any]:
    # Remote Control's search, which the tool used before, answers nothing here.
    editor.session._http_put = AsyncMock(return_value={"Assets": []})
    return _run(editor.session.search_assets(**kwargs))


def _paths(result: Dict[str, Any]) -> List[str]:
    return [asset["path"] for asset in result["assets"]]


def test_a_short_class_name_is_resolved_to_its_class_path(editor: SimpleNamespace) -> None:
    """StaticMesh matched nothing (and fired an engine ensure); only /Script/Engine.StaticMesh worked."""
    result = _search(editor, class_names=["StaticMesh"])

    assert result["class_paths"] == ["/Script/Engine.StaticMesh"]
    assert _paths(result) == [
        "/Engine/BasicShapes/Cube.Cube",
        "/Engine/BasicShapes/Sphere.Sphere",
        "/Game/Props/Big/SM_CubeLarge.SM_CubeLarge",
        "/Game/Props/SM_Cube.SM_Cube",
    ]


def test_a_bare_query_searches_every_content_root_and_skips_redirectors(editor: SimpleNamespace) -> None:
    """A bare query searched only /Game: "Cube" found nothing in /Engine."""
    result = _search(editor, query="cube")

    assert _paths(result) == [
        "/Engine/BasicShapes/Cube.Cube",
        "/Game/Props/Big/SM_CubeLarge.SM_CubeLarge",
        "/Game/Props/SM_Cube.SM_Cube",
        "/Game/Textures/T_Cube.T_Cube",
    ]


def test_package_paths_include_their_subfolders_unless_told_otherwise(editor: SimpleNamespace) -> None:
    recursive = _search(editor, package_paths=["/Game/Props/"])
    flat = _search(editor, package_paths=["/Game/Props"], recursive_paths=False)

    assert _paths(recursive) == ["/Game/Props/Big/SM_CubeLarge.SM_CubeLarge", "/Game/Props/SM_Cube.SM_Cube"]
    assert _paths(flat) == ["/Game/Props/SM_Cube.SM_Cube"]


def test_results_page_by_offset_and_say_what_follows(editor: SimpleNamespace) -> None:
    """truncated was always false at the limit, so an agent never knew more matched."""
    first = _search(editor, class_names=["StaticMesh"], max_results=3)
    last = _search(editor, class_names=["StaticMesh"], max_results=3, offset=first["next_offset"])

    assert (first["count"], first["total"], first["truncated"], first["next_offset"]) == (3, 4, True, 3)
    assert (last["count"], last["total"], last["truncated"], last["next_offset"]) == (1, 4, False, None)
    assert _paths(first) + _paths(last) == _paths(_search(editor, class_names=["StaticMesh"]))


def test_each_asset_carries_its_package_path_and_registry_metadata(editor: SimpleNamespace) -> None:
    """package_path was always empty and the triangle, LOD, Nanite and collision tags were dropped."""
    (cube,) = _search(editor, query="Cube", package_paths=["/Engine"])["assets"]

    assert cube == {
        "name": "Cube",
        "path": "/Engine/BasicShapes/Cube.Cube",
        "class_name": "StaticMesh",
        "class_path": "/Script/Engine.StaticMesh",
        "package_name": "/Engine/BasicShapes/Cube",
        "package_path": "/Engine/BasicShapes",
        "metadata": {"Triangles": "48", "LODs": "1", "NaniteEnabled": "False", "CollisionPrims": "1"},
    }


def test_subclasses_are_included_unless_told_otherwise(editor: SimpleNamespace) -> None:
    with_subclasses = _search(editor, class_names=["MaterialInterface"])
    exact = _search(editor, class_names=["MaterialInterface"], include_subclasses=False)

    assert _paths(with_subclasses) == [
        "/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial", "/Game/Materials/MI_Red.MI_Red",
    ]
    assert _paths(exact) == []


def test_an_unknown_class_is_a_validation_error(editor: SimpleNamespace) -> None:
    result = _search(editor, class_names=["StaticMesh", "NotAClass"])

    assert result["error_type"] == "ValidationError"
    assert "Unknown asset class: NotAClass" in result["error"]


@pytest.mark.parametrize("kwargs", [{"package_paths": ["/Game/../Engine"]}, {"offset": -1}, {"offset": True}])
def test_a_malformed_folder_or_offset_is_refused_before_any_request(
    editor: SimpleNamespace, kwargs: Dict[str, Any]
) -> None:
    result = _search(editor, **kwargs)

    assert result["error_type"] == "ValidationError"
    assert editor.fake.scripts == []


def test_the_search_payload_reaches_the_agent_whole(editor: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _call_tool(editor.session, monkeypatch, "search_unreal_assets", {
        "query": "Cube", "package_paths": "/Engine", "max_results": 1,
    })

    payload = json.loads(result.content[-1].text)
    assert payload["assets"][0]["metadata"]["Triangles"] == "48"
    assert payload["assets"][0]["package_path"] == "/Engine/BasicShapes"
    assert (payload["total"], payload["next_offset"], payload["class_paths"]) == (1, None, [])


# ---------------------------------------------------------------------------
# import_unreal_usd: Unreal 5.7's options, set on every import
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"import_lights": True}, "import_lights is not a USD import option in Unreal 5.7"),
        ({"import_cameras": False}, "lights and cameras are imported as actors, with import_actors"),
        ({"collapse": True}, "Unknown USD import option 'collapse'"),
        ({"kinds_to_collapse": ["widget"]}, "kinds_to_collapse must be a list of USD kinds"),
        ({"fallback_collision_type": "hull"}, "fallback_collision_type must be one of"),
        ({"meters_per_unit": 1.0}, "give both"),
        ({"meters_per_unit": 0, "up_axis": "y"}, "meters_per_unit must be a number above 0"),
        ({"meters_per_unit": 1, "up_axis": "x"}, "up_axis must be 'y' or 'z'"),
        ({"prims_to_import": ["Root"]}, "prims_to_import must be a non-empty list of prim paths"),
        ({"import_actors": "yes"}, "import_actors must be true or false"),
    ],
)
def test_an_option_unreal_does_not_take_is_refused_before_anything_runs(
    editor: SimpleNamespace, options: Dict[str, Any], message: str
) -> None:
    """import_lights and import_cameras failed in the editor with "Failed to find property"."""
    result = _import(editor, import_options=options)

    assert result["error_type"] == "ValidationError"
    assert message in result["error"]
    assert editor.fake.scripts == []


def test_the_collapse_collision_and_stage_options_reach_the_importer(editor: SimpleNamespace) -> None:
    """Three exported actors came back as one merged SM_Root, and nothing exposed collapsing."""
    result = _import(editor, import_options={
        "use_prim_kinds_for_collapsing": False,
        "kinds_to_collapse": ["component", "assembly"],
        "fallback_collision_type": "convex_hull",
        "nanite_triangle_threshold": 1000,
        "prims_to_import": ["/Root/Cube"],
        "meters_per_unit": 1.0,
        "up_axis": "y",
    })

    (options,) = editor.fake.import_options
    assert options.use_prim_kinds_for_collapsing is False
    assert options.kinds_to_collapse == 2 + 8
    assert options.fallback_collision_type == COLLISION.CONVEX_HULL
    assert options.nanite_triangle_threshold == 1000
    assert options.prims_to_import == ["/Root/Cube"]
    assert options.override_stage_options is True
    assert (options.stage_options.meters_per_unit, options.stage_options.up_axis) == (1.0, "Y_AXIS")
    assert len(result["imported_assets"]) == 2  # one mesh per prim
    assert editor.fake.config_writes == []


def test_options_left_out_take_unreals_defaults_not_the_saved_settings(editor: SimpleNamespace) -> None:
    editor.fake.saved_config.update(kinds_to_collapse=2, fallback_collision_type=COLLISION.CUBE)

    result = _import(editor)

    (options,) = editor.fake.import_options
    assert (options.kinds_to_collapse, options.fallback_collision_type) == (18, COLLISION.NONE)
    assert options.override_stage_options is False
    assert any('fallback_collision_type="cube"' in warning and "kinds_to_collapse=2" in warning
               for warning in result["warnings"]), result["warnings"]


def test_the_level_actors_an_import_adds_are_listed_once(editor: SimpleNamespace) -> None:
    """UE 5.7 listed an imported actor twice, and actor_paths grew a duplicate per import."""
    editor.fake.duplicate_new_actors = True
    first = _import(editor)

    replaced = _import(editor, replace_existing=True)
    second = _import(editor, destination_path="/Game/Second")

    assert len(replaced["actor_paths"]) == 1
    assert replaced["removed_actor_paths"] == first["actor_paths"]  # each replaced actor once
    assert len(second["actor_paths"]) == 1


def test_the_warnings_are_what_the_editor_logged_during_the_import(editor: SimpleNamespace) -> None:
    """warnings was always []."""
    editor.fake.import_log = ["Prim '/Root/Lamp' has no material binding"]

    result = _import(editor)

    assert result["warnings"] == ["LogUsd: Warning: Prim '/Root/Lamp' has no material binding"]
    assert "FLUSHLOG" in editor.fake.console_commands


@pytest.mark.parametrize(
    ("destination", "error_type"),
    [("/Engine/BasicShapes/X", "SandboxError"), ("/Game/../Engine", "ValidationError"), ("Game", "ValidationError")],
)
def test_an_import_destination_outside_game_is_refused(
    editor: SimpleNamespace, destination: str, error_type: str
) -> None:
    result = _import(editor, destination_path=destination)

    assert result["error_type"] == error_type
    assert editor.fake.scripts == []


def test_game_itself_is_a_destination(editor: SimpleNamespace) -> None:
    """The bare /Game destination was refused."""
    result = _import(editor, destination_path="/Game")

    assert result["imported_assets"] == ["/Game/scene/StaticMeshes/SM_Root.SM_Root"]


# ---------------------------------------------------------------------------
# export_unreal_usd: no silent overwrite, every file reported
# ---------------------------------------------------------------------------


def test_an_existing_output_is_kept_unless_overwrite(editor: SimpleNamespace) -> None:
    """An export overwrote an existing file without being asked."""
    target = editor.tmp_path / "out" / "cube.usda"
    target.parent.mkdir()
    target.write_text("#usda 1.0\n# the user's file\n")

    refused = _run(editor.session.export_usd([CUBE], str(target)))

    assert refused["error_type"] == "InvalidState"
    assert "overwrite=true" in refused["details"]["hint"]
    assert target.read_text().endswith("# the user's file\n")
    assert editor.fake.export_tasks == []

    replaced = _run(editor.session.export_usd([CUBE], str(target), overwrite=True))

    assert replaced["actors_exported"] == 1
    assert replaced["overwritten_files"] == [str(target)]


def test_the_mesh_files_written_next_to_the_output_are_reported(editor: SimpleNamespace) -> None:
    """The exporter's per-mesh files were written and never mentioned."""
    folder = editor.tmp_path / "out"
    result = _run(editor.session.export_usd([CUBE, SPHERE], str(folder / "pair.usda")))

    assert result["written_files"] == [
        str(folder / "pair.usda"), str(folder / "pair_assets" / "Cube_Mesh.usda"),
        str(folder / "pair_assets" / "Sphere_Mesh.usda"),
    ]
    assert result["overwritten_files"] == []
    assert result["warnings"] == []


def test_an_export_beside_another_keeps_its_meshes_apart(editor: SimpleNamespace) -> None:
    """Two exports into one folder wrote their mesh files over each other's."""
    folder = editor.tmp_path / "out"
    _run(editor.session.export_usd([CUBE], str(folder / "a.usda")))

    second = _run(editor.session.export_usd([CUBE], str(folder / "b.usda")))

    assert second["written_files"] == [str(folder / "b.usda"), str(folder / "b_assets" / "Cube_Mesh.usda")]
    assert (folder / "a_assets" / "Cube_Mesh.usda").exists()


def test_a_mesh_file_left_in_the_asset_folder_is_kept_unless_overwrite(editor: SimpleNamespace) -> None:
    """The overwrite gate covered the output alone; the mesh files beside it were replaced."""
    folder = editor.tmp_path / "out"
    (folder / "cube_assets").mkdir(parents=True)
    mesh = folder / "cube_assets" / "Cube_Mesh.usda"
    mesh.write_text("#usda 1.0\n# the user's mesh\n")

    refused = _run(editor.session.export_usd([CUBE], str(folder / "cube.usda")))

    assert refused["error_type"] == "InvalidState"
    assert refused["details"]["existing_files"] == [str(mesh)]
    assert editor.fake.export_tasks == []
    assert not (folder / "cube.usda").exists()
    replaced = _run(editor.session.export_usd([CUBE], str(folder / "cube.usda"), overwrite=True))
    assert replaced["overwritten_files"] == [str(mesh)]


def test_the_sublevel_layers_and_material_overrides_are_gated_too(editor: SimpleNamespace) -> None:
    """export_sublayers writes a layer per sublevel, and MaterialOverrides, beside the output."""
    sub_world = World("/Game/Maps/Sub.Sub")
    editor.fake.actors[LEVEL + "Lamp"] = Actor(LEVEL + "Lamp", editor.fake.world, editor.fake, level_world=sub_world)
    folder = editor.tmp_path / "out"
    folder.mkdir()
    (folder / "Sub.usda").write_text("#usda 1.0\n# the user's layer\n")

    refused = _run(editor.session.export_usd(
        [CUBE, LEVEL + "Lamp"], str(folder / "scene.usda"), export_options={"export_sublayers": True}
    ))

    assert refused["error_type"] == "InvalidState"
    assert refused["details"]["existing_files"] == [str(folder / "Sub.usda")]
    assert editor.fake.export_tasks == []
    (folder / "Sub.usda").unlink()
    done = _run(editor.session.export_usd(
        [CUBE, LEVEL + "Lamp"], str(folder / "scene.usda"), export_options={"export_sublayers": True}
    ))
    assert {str(folder / "MaterialOverrides.usda"), str(folder / "Sub.usda")} <= set(done["written_files"])
    again = _run(editor.session.export_usd(
        [CUBE], str(folder / "other.usda"), export_options={"export_sublayers": True}
    ))
    assert again["details"]["existing_files"] == [str(folder / "MaterialOverrides.usda")]


def test_a_dynamic_mesh_actor_exported_without_its_mesh_is_a_warning(editor: SimpleNamespace) -> None:
    """Unreal's USD exporter writes a DynamicMeshActor as an empty Xform, and said nothing."""
    blob = LEVEL + "Blob"
    editor.fake.actors[blob] = Actor(blob, editor.fake.world, editor.fake, dynamic=True)

    result = _run(editor.session.export_usd([CUBE, blob], str(editor.tmp_path / "out" / "blob.usda")))

    assert result["warnings"] == [
        "Unreal's USD exporter writes DynamicMeshActors as empty Xforms, without their mesh: " + blob
        + ". convert_unreal_mesh_format with target_format static_mesh and replace_actor true makes them "
        "StaticMeshActors, which export with their geometry."
    ]


# ---------------------------------------------------------------------------
# assign_unreal_material: any mesh component, a slot by name
# ---------------------------------------------------------------------------


MULTI = LEVEL + "Crate"


def _multi_mesh_actor(editor: SimpleNamespace) -> Actor:
    actor = Actor(MULTI, editor.fake.world, editor.fake,
                  meshes={"Body": ("Wood", "Metal"), "Lid": ("Wood",)}, mesh_root=False)
    editor.fake.actors[MULTI] = actor
    return actor


def test_an_actor_with_several_mesh_components_names_them(editor: SimpleNamespace) -> None:
    """It failed with a traceback: "exactly one MeshComponent"."""
    _multi_mesh_actor(editor)

    result = _run(editor.session.assign_material(MULTI, RED))

    assert result["error_type"] == "ValidationError"
    assert result["error"] == (
        "The actor has 2 mesh components and none is its root; pass component_name, one of: "
        "Body (MeshComponent), Lid (MeshComponent)"
    )
    assert editor.fake.events == []


def test_a_component_and_a_slot_are_picked_by_name(editor: SimpleNamespace) -> None:
    actor = _multi_mesh_actor(editor)

    result = _run(editor.session.assign_material(MULTI, RED, slot_name="Metal", component_name="Body"))

    body = actor.components[0]
    assert body.materials == [editor.fake.basic, editor.fake.red]
    assert (result["component_name"], result["slot_index"], result["slot_name"]) == ("Body", 1, "Metal")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"component_name": "Wheel"}, "The actor has no mesh component named 'Wheel'; its mesh components: "),
        ({"component_name": "Body", "slot_name": "Glass"}, "Body has no material slot named 'Glass'; its slots: "
                                                           "Wood, Metal"),
        ({"component_name": "Lid", "slot_index": 1},
         "slot_index 1 is out of range: Lid has 1 material slot(s) (Wood)"),
    ],
)
def test_an_unknown_component_or_slot_names_what_the_actor_has(
    editor: SimpleNamespace, kwargs: Dict[str, Any], message: str
) -> None:
    _multi_mesh_actor(editor)

    result = _run(editor.session.assign_material(MULTI, RED, **kwargs))

    assert result["error_type"] == "ValidationError"
    assert message in result["error"]
    assert editor.fake.events == []


# ---------------------------------------------------------------------------
# create_unreal_material_instance: typed errors, the save checked
# ---------------------------------------------------------------------------


def _create(editor: SimpleNamespace, parent: str, name: str = "MI_New") -> Dict[str, Any]:
    return _run(editor.session.create_material_instance(parent, name, "/Game/Materials"))


@pytest.mark.parametrize(
    ("parent", "error_type", "error"),
    [
        ("/Engine/BasicShapes/Cube", "InvalidAssetType", "/Engine/BasicShapes/Cube is a StaticMesh, not a material"),
        ("/Game/Maps/Other", "InvalidAssetType", "/Game/Maps/Other is a level, not a material"),
        ("/Game/Nope/M_Nope", "NotFoundError", "No parent material at '/Game/Nope/M_Nope'"),
    ],
)
def test_a_parent_that_is_no_material_is_a_typed_error(
    editor: SimpleNamespace, parent: str, error_type: str, error: str
) -> None:
    """A StaticMesh parent was reported as "Parent material not found", untyped."""
    result = _create(editor, parent)

    assert result["error_type"] == error_type
    assert error in result["error"]
    assert editor.fake.maps_loaded == []  # a level is never loaded as an asset
    assert "/Game/Materials/MI_New" not in editor.fake.content


def test_an_existing_instance_name_is_an_invalid_state(editor: SimpleNamespace) -> None:
    result = _create(editor, BASIC, "MI_Red")

    assert result["error_type"] == "InvalidState"
    assert "Asset already exists: /Game/Materials/MI_Red" in result["error"]


def test_an_instance_unreal_does_not_save_is_reported(editor: SimpleNamespace) -> None:
    """save_loaded_asset's result was ignored."""
    editor.fake.save_ok = False

    result = _create(editor, BASIC)

    assert result["error_type"] == "EngineError"
    assert result["instance_path"] == "/Game/Materials/MI_New.MI_New"
    assert "did not save it" in result["error"]


# ---------------------------------------------------------------------------
# get_unreal_interchange_info: pipelines and every kind of format
# ---------------------------------------------------------------------------


def test_interchange_info_lists_the_pipeline_stacks_and_every_kind_of_format(editor: SimpleNamespace) -> None:
    """pipelines was always [] and only the SCENES formats were listed."""
    result = _run(editor.session.get_interchange_info())

    assert result["pipelines"] == [
        {"name": "Assets", "kind": "content",
         "pipelines": ["/Interchange/Pipelines/DefaultAssetsPipeline.DefaultAssetsPipeline"]},
        {"name": "Scene", "kind": "scene",
         "pipelines": ["/Interchange/Pipelines/DefaultSceneAssetsPipeline.A",
                       "/Interchange/Pipelines/DefaultSceneLevelPipeline.B"]},
    ]
    assert result["supported_formats_by_kind"] == {
        "assets": ["fbx;Filmbox", "obj;OBJ File Format"], "scenes": ["fbx;Filmbox"], "actors": [],
    }
    assert result["supported_formats"] == ["fbx;Filmbox", "obj;OBJ File Format"]
    assert result["default_pipeline_stacks"] == {"content": "Assets", "scene": "Scene"}
    assert result["pipeline_enumeration_available"] is True


# ---------------------------------------------------------------------------
# import_unreal_asset: FBX, glTF and OBJ through Interchange
# ---------------------------------------------------------------------------


def _asset_import(editor: SimpleNamespace, name: str = "crate.fbx", **kwargs: Any) -> Dict[str, Any]:
    path = editor.tmp_path / name
    if not path.exists():
        path.write_text("data")
    kwargs.setdefault("destination_path", "/Game/Props")
    return _run(editor.session.import_asset(str(path), **kwargs))


def _textured_obj(editor: SimpleNamespace) -> str:
    """crate.obj, whose MTL library names crate.png beside it."""
    (editor.tmp_path / "crate.obj").write_text("mtllib crate.mtl\nv 0 0 0\n")
    (editor.tmp_path / "crate.mtl").write_text("newmtl Wood\nmap_Kd crate.png\n")
    (editor.tmp_path / "crate.png").write_bytes(b"\x89PNG")
    return "crate.obj"


def test_an_fbx_file_imports_its_assets_and_leaves_nothing_in_the_console(editor: SimpleNamespace) -> None:
    result = _asset_import(editor)

    assert result["imported_assets"] == ["/Game/Props/M_crate.M_crate", "/Game/Props/crate.crate"]
    assert (result["replaced_assets"], result["kept_assets"]) == ([], [])
    assert result["dirty_engine_packages"] == []
    (task,) = editor.fake.import_tasks
    assert (task.automated, task.save, task.replace_existing) == (True, True, False)
    assert _script_args(editor.fake.scripts[-1])["sources"] == [str(editor.tmp_path / "crate.fbx")]
    assert editor.fake.leaked() == []


def test_the_files_a_model_reads_are_passed_as_its_sources(editor: SimpleNamespace) -> None:
    """The texture an OBJ's MTL names is how its texture asset is told apart from others."""
    result = _asset_import(editor, _textured_obj(editor))

    assert result["imported_assets"] == [
        "/Game/Props/T_crate.T_crate", "/Game/Props/M_crate.M_crate", "/Game/Props/crate.crate",
    ]
    real = os.path.realpath(editor.tmp_path)
    assert _script_args(editor.fake.scripts[-1])["sources"] == [
        str(editor.tmp_path / "crate.obj"), real + "/crate.mtl", real + "/crate.png",
    ]


def test_a_file_whose_assets_all_exist_imports_nothing_without_replace(editor: SimpleNamespace) -> None:
    """The texture and the material are listed with the mesh: existing_assets used to omit them."""
    _asset_import(editor, _textured_obj(editor))

    again = _asset_import(editor, "crate.obj")

    assert again["error_type"] == "InvalidState"
    assert again["details"]["existing_assets"] == [
        "/Game/Props/M_crate.M_crate", "/Game/Props/T_crate.T_crate", "/Game/Props/crate.crate",
    ]
    assert editor.fake.replaced_assets == []


def test_existing_assets_are_kept_and_reported_beside_the_new_ones(editor: SimpleNamespace) -> None:
    editor.fake.content["/Game/Props/M_crate"] = "MaterialInstanceConstant"

    result = _asset_import(editor)

    assert result["imported_assets"] == ["/Game/Props/crate.crate"]
    assert result["kept_assets"] == ["/Game/Props/M_crate.M_crate"]


def test_replace_existing_reports_the_assets_unreals_reimport_kept(editor: SimpleNamespace) -> None:
    """replace_existing replaced the mesh alone and kept the texture and material without a word."""
    _asset_import(editor, _textured_obj(editor))

    result = _asset_import(editor, "crate.obj", replace_existing=True)

    assert result["imported_assets"] == ["/Game/Props/crate.crate"]
    assert result["replaced_assets"] == ["/Game/Props/crate.crate"]
    assert result["kept_assets"] == ["/Game/Props/M_crate.M_crate", "/Game/Props/T_crate.T_crate"]
    (warning,) = result["warnings"]
    assert warning.startswith("replace_existing reimported /Game/Props/crate.crate, but Unreal's Interchange "
                              "reimport keeps the other assets an earlier import of this file made, as they were: "
                              "/Game/Props/M_crate.M_crate, /Game/Props/T_crate.T_crate.")
    assert editor.fake.replaced_assets == ["/Game/Props/crate"]


def test_an_asset_another_file_imported_is_not_reported_as_kept(editor: SimpleNamespace) -> None:
    _asset_import(editor, "barrel.fbx")
    _asset_import(editor)

    result = _asset_import(editor, replace_existing=True)

    assert result["kept_assets"] == ["/Game/Props/M_crate.M_crate"]


def test_engine_packages_an_import_dirties_are_reported(editor: SimpleNamespace) -> None:
    """Interchange set bUsedWithNanite on its shared plugin material and left it dirty, unreported."""
    shared = "/InterchangeAssets/Materials/PBRSurfaceMaterial_MR"
    editor.fake.import_dirties = [shared, "/Game/Props/Scratch"]

    result = _asset_import(editor)

    assert result["dirty_engine_packages"] == [shared]
    assert result["warnings"][-1].startswith(
        f"The editor holds 1 package(s) outside /Game modified in memory: {shared}. They belong to the engine "
        "or a plugin"
    )
    assert shared not in editor.fake.saved_packages


def test_an_engine_package_dirtied_after_an_earlier_import_returned_is_reported_by_the_next(
    editor: SimpleNamespace,
) -> None:
    """Unreal set the flag frames after the import whose mesh needed it: the later call names it."""
    shared = "/InterchangeAssets/Materials/PBRSurfaceMaterial_MR"
    first = _asset_import(editor)
    editor.fake.dirty.add(shared)  # a later editor frame

    second = _asset_import(editor, "barrel.fbx")

    assert first["dirty_engine_packages"] == []
    assert second["dirty_engine_packages"] == [shared]


def test_the_warnings_come_from_this_editors_log_not_a_sibling_editors(editor: SimpleNamespace) -> None:
    """A second editor of the project writes <Project>_2.log; <Project>.log was the other editor's."""
    other = editor.tmp_path / "Fake_2.log"
    other.write_text("")
    editor.fake.log_path, sibling = other, editor.fake.log_path
    sibling.write_text(f"{LOG_TIME}LogInterchangeEngine: Error: the other editor's import failed\n")
    editor.fake.content["/Game/Props/M_crate"] = "MaterialInstanceConstant"

    result = _asset_import(editor)

    assert "the other editor's import failed" not in " ".join(result["warnings"])
    assert any("we are not overriding the asset named" in line for line in result["warnings"])


def _confine(editor: SimpleNamespace) -> Tuple[Any, Any]:
    """Allow only ``sbx`` under the test's folder; ``outside`` beside it holds a secret image."""
    sbx, outside = editor.tmp_path / "sbx", editor.tmp_path / "outside"
    sbx.mkdir()
    outside.mkdir()
    (outside / "secret.png").write_bytes(b"\x89PNG secret")
    base = Settings()
    settings = base.model_copy(update={
        "security": base.security.model_copy(update={"allowed_paths": [str(sbx)]}),
    })
    editor.session._path_policy = PathPolicy.from_settings(settings)
    return sbx, outside


def _refused_before_the_editor(editor: SimpleNamespace, payload: Dict[str, Any], *outside: Any) -> None:
    assert payload["error_type"] == "SandboxError", payload
    assert payload["details"]["references_outside"] == [os.path.realpath(path) for path in outside]
    assert editor.fake.import_tasks == []


@pytest.mark.parametrize(
    ("name", "write"),
    [
        ("relcube.obj", lambda sbx, out: (
            (sbx / "relcube.obj").write_text("mtllib relcube.mtl\nv 0 0 0\n"),
            (sbx / "relcube.mtl").write_text("newmtl M\nmap_Kd ../outside/secret.png\n"))),
        ("mtlcube.obj", lambda sbx, out: (
            (sbx / "mtlcube.obj").write_text("mtllib ../outside/evil.mtl\nv 0 0 0\n"),
            (out / "evil.mtl").write_text("newmtl M\nmap_Kd secret.png\n"))),
        ("relball.gltf", lambda sbx, out: (sbx / "relball.gltf").write_text(json.dumps(
            {"asset": {"version": "2.0"}, "images": [{"uri": "../outside/secret.png"}]}))),
    ],
)
def test_a_model_naming_a_file_outside_the_sandbox_is_refused_before_the_editor_reads_it(
    editor: SimpleNamespace, name: str, write: Any
) -> None:
    """The review's fixtures: a texture or an MTL library outside the sandbox was imported."""
    sbx, outside = _confine(editor)
    write(sbx, outside)
    culprit = outside / ("evil.mtl" if name == "mtlcube.obj" else "secret.png")

    with pytest.raises(unreal_runtime.SandboxDenied) as denied:
        _run(editor.session.import_asset(str(sbx / name), "/Game/Props"))

    assert denied.value.details["referenced_by"] == str(sbx / name)
    _refused_before_the_editor(editor, {"error_type": "SandboxError", **vars(denied.value)}, culprit)
    assert editor.fake.scripts == []


def test_the_refusal_reaches_the_agent_as_a_sandbox_error(
    editor: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    sbx, outside = _confine(editor)
    document = {"asset": {"version": "2.0"}, "images": [{"uri": "../outside/secret.png"}]}
    (sbx / "ball.gltf").write_text(json.dumps(document))

    payload = json.loads(_call_tool(editor.session, monkeypatch, "import_unreal_asset", {
        "file_path": str(sbx / "ball.gltf"), "destination_path": "/Game/Props",
    }).content[-1].text)

    _refused_before_the_editor(editor, payload, outside / "secret.png")
    assert payload["details"]["referenced_by"] == str(sbx / "ball.gltf")
    assert payload["error"].startswith("ball.gltf refers to 1 file(s) outside the sandbox, so it was not imported")


def test_an_fbx_relative_texture_name_is_resolved_where_the_editor_opens_it(
    editor: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The FBX SDK opens a relative FileName from the editor's working directory."""
    sbx, outside = _confine(editor)
    model = sbx / "crate.fbx"
    model.write_text('; FBX 7.4.0 project file\nTexture: 1, "Texture::W", "" {\n  FileName: "secret.png"\n}\n')
    monkeypatch.chdir(outside)  # the fake editor runs in this process: its working directory is outside

    with pytest.raises(unreal_runtime.SandboxDenied) as denied:
        _run(editor.session.import_asset(str(model), "/Game/Props"))

    assert denied.value.details["references_outside"] == [os.path.realpath(outside / "secret.png")]
    (script,) = editor.fake.scripts
    assert _script_args(script) == {"names": ["secret.png"]}
    assert editor.fake.import_tasks == []


def test_a_binary_fbx_that_cannot_be_parsed_is_refused(editor: SimpleNamespace) -> None:
    sbx, _ = _confine(editor)
    (sbx / "crate.fbx").write_bytes(b"Kaydara FBX Binary  \x00\x1a\x00" + b"\xff" * 9)

    with pytest.raises(unreal_runtime.SandboxDenied, match="could not be checked against the sandbox"):
        _run(editor.session.import_asset(str(sbx / "crate.fbx"), "/Game/Props"))

    assert editor.fake.scripts == []


def test_without_the_sandbox_a_model_imports_whatever_it_names(editor: SimpleNamespace) -> None:
    base = Settings()
    settings = base.model_copy(update={"security": base.security.model_copy(update={"sandbox_enabled": False})})
    editor.session._path_policy = PathPolicy.from_settings(settings)
    (editor.tmp_path / "crate.obj").write_text("mtllib ../elsewhere.mtl\n")

    result = _asset_import(editor, "crate.obj")

    assert result["imported_assets"] == ["/Game/Props/M_crate.M_crate", "/Game/Props/crate.crate"]


def test_a_file_unreal_cannot_read_fails_with_the_logged_reason(editor: SimpleNamespace) -> None:
    result = _asset_import(editor, "garbage.fbx")

    assert result["error_type"] == "EngineError"
    assert "Cannot open FBX file" in result["error"]


@pytest.mark.parametrize(
    ("name", "destination", "error_type"),
    [("scene.usda", "/Game/Props", "ValidationError"), ("crate.fbx", "/Engine/Props", "SandboxError")],
)
def test_a_usd_file_or_a_folder_outside_game_is_refused(
    editor: SimpleNamespace, name: str, destination: str, error_type: str
) -> None:
    result = _asset_import(editor, name, destination_path=destination)

    assert result["error_type"] == error_type
    assert editor.fake.scripts == []


def test_an_asset_import_is_refused_during_pie(editor: SimpleNamespace) -> None:
    editor.fake.pie = True

    result = _asset_import(editor)

    assert result["error_type"] == "PIEActive"
    assert editor.fake.import_tasks == []


# ---------------------------------------------------------------------------
# save_unreal_assets: only modified /Game packages, never during PIE
# ---------------------------------------------------------------------------


EXTERNAL = "/Game/__ExternalActors__/Maps/Main/A/B/0123"


def _save(editor: SimpleNamespace, **kwargs: Any) -> Dict[str, Any]:
    return _run(editor.session.save_assets(**kwargs))


def test_save_level_saves_the_level_and_its_actor_packages_only(editor: SimpleNamespace) -> None:
    editor.fake.dirty.update({"/Game/Maps/Main", EXTERNAL, "/Game/Materials/MI_Red"})

    result = _save(editor, save_level=True)

    assert result["saved"] == ["/Game/Maps/Main", EXTERNAL]
    assert editor.fake.dirty == {"/Game/Materials/MI_Red"}
    assert result["level_package"] == "/Game/Maps/Main"


def test_all_dirty_skips_engine_and_never_saved_packages(editor: SimpleNamespace) -> None:
    editor.fake.dirty.update({"/Game/Materials/MI_Red", "/Engine/BasicShapes/BasicShapeMaterial", "/Temp/Untitled_1"})

    result = _save(editor, all_dirty=True)

    assert result["saved"] == ["/Game/Materials/MI_Red"]
    assert [entry["package"] for entry in result["skipped"]] == [
        "/Engine/BasicShapes/BasicShapeMaterial", "/Temp/Untitled_1",
    ]
    assert "allow_engine_content=true" in result["skipped"][0]["reason"]
    assert "/Engine/BasicShapes/BasicShapeMaterial" in editor.fake.dirty


def test_a_dry_run_lists_what_would_be_saved_and_saves_nothing(editor: SimpleNamespace) -> None:
    editor.fake.dirty.add("/Game/Materials/MI_Red")

    result = _save(editor, asset_paths=["/Game/Materials/MI_Red.MI_Red", "/Game/Props/SM_Cube"], dry_run=True)

    assert result["would_save"] == ["/Game/Materials/MI_Red"]
    assert result["skipped"] == [{"package": "/Game/Props/SM_Cube", "reason": "not modified, or not loaded"}]
    assert editor.fake.saved_packages == []


def test_a_package_unreal_does_not_save_fails_the_call(editor: SimpleNamespace) -> None:
    editor.fake.dirty.add("/Game/Materials/MI_Red")
    editor.fake.unsaveable.add("/Game/Materials/MI_Red")

    result = _save(editor, all_dirty=True)

    assert result["error_type"] == "EngineError"
    assert result["failed"] == ["/Game/Materials/MI_Red"]


def test_a_level_never_saved_cannot_be_saved(editor: SimpleNamespace) -> None:
    editor.fake.world.path = "/Temp/Untitled_1.Untitled_1"

    result = _save(editor, save_level=True)

    assert result["error_type"] == "InvalidState"
    assert "has never been saved" in result["error"]


@pytest.mark.parametrize(
    ("kwargs", "error_type"),
    [({}, "ValidationError"), ({"asset_paths": ["/Engine/BasicShapes/Cube"]}, "SandboxError"),
     ({"asset_paths": ["Cube"]}, "ValidationError")],
)
def test_a_save_asking_nothing_or_engine_content_is_refused_before_any_request(
    editor: SimpleNamespace, kwargs: Dict[str, Any], error_type: str
) -> None:
    result = _save(editor, **kwargs)

    assert result["error_type"] == error_type
    assert editor.fake.scripts == []


def test_saving_is_refused_during_pie(editor: SimpleNamespace) -> None:
    editor.fake.dirty.add("/Game/Materials/MI_Red")
    editor.fake.pie = True

    result = _save(editor, all_dirty=True)

    assert result["error_type"] == "PIEActive"
    assert editor.fake.saved_packages == []


def test_the_new_tools_are_declared_with_their_hints() -> None:
    specs = {tool.name: tool for tool in _unreal_tools.UNREAL_TOOLS}
    assert (specs["import_unreal_asset"].read_only, specs["import_unreal_asset"].destructive) == (False, True)
    assert (specs["save_unreal_assets"].read_only, specs["save_unreal_assets"].destructive) == (False, True)
    assert [p.name for p in specs["save_unreal_assets"].params] == [
        "asset_paths", "save_level", "all_dirty", "allow_engine_content", "dry_run",
    ]


def test_the_save_and_import_payloads_reach_the_agent_whole(
    editor: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    editor.fake.dirty.add("/Game/Materials/MI_Red")
    (editor.tmp_path / "crate.fbx").write_text("data")

    saved = json.loads(_call_tool(editor.session, monkeypatch, "save_unreal_assets", {
        "all_dirty": True, "dry_run": True,
    }).content[-1].text)
    imported = json.loads(_call_tool(editor.session, monkeypatch, "import_unreal_asset", {
        "file_path": str(editor.tmp_path / "crate.fbx"), "destination_path": "/Game/Props",
    }).content[-1].text)

    assert saved["would_save"] == ["/Game/Materials/MI_Red"] and saved["dry_run"] is True
    assert imported["imported_assets"] == ["/Game/Props/M_crate.M_crate", "/Game/Props/crate.crate"]


# ---------------------------------------------------------------------------
# Attached mode: a changed attachment is an InvalidState, not a traceback
# ---------------------------------------------------------------------------


GUARD_TRACEBACK = (
    "Traceback (most recent call last):\n  File \"<string>\", line 40, in <module>\n"
    "RuntimeError: Unreal attachment changed (map_path); run simul unreal attach again\n"
)


def _attached_session(monkeypatch: pytest.MonkeyPatch) -> UnrealRuntimeSession:
    session = UnrealRuntimeSession(Settings(unreal={"mode": "attached"}))

    async def execute(code: str, mode: str = "ExecuteFile", timeout: Optional[float] = None) -> Dict[str, Any]:
        return {"ReturnValue": False, "CommandResult": GUARD_TRACEBACK}

    monkeypatch.setattr(session, "_execute_python", execute)
    return session


def test_a_changed_attachment_is_an_invalid_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard's RuntimeError came back as the tool's ScriptError traceback."""
    session = _attached_session(monkeypatch)

    for result in (
        _run(session.get_interchange_info()),
        _run(session.execute_script("print(1)")),
    ):
        assert result == {
            "success": False,
            "error_type": "InvalidState",
            "error": "Unreal attachment changed (map_path); run simul unreal attach again",
            "details": {
                "reason": "attachment_changed",
                "hint": "Nothing was run. Run simul unreal attach again (or activate the attached viewport), "
                        "then retry.",
            },
        }


def test_the_same_text_outside_attached_mode_stays_a_script_error(monkeypatch: pytest.MonkeyPatch) -> None:
    session = UnrealRuntimeSession(Settings())

    async def execute(code: str, mode: str = "ExecuteFile", timeout: Optional[float] = None) -> Dict[str, Any]:
        return {"ReturnValue": False, "CommandResult": GUARD_TRACEBACK}

    monkeypatch.setattr(session, "_execute_python", execute)

    assert _run(session.get_interchange_info())["error_type"] == "ScriptError"


def test_a_refused_remote_control_preflight_is_named_invalid_state() -> None:
    from simul.adapters.unreal_runtime import UnrealAttachmentRefused, _attachment_refusal
    from simul.utils.errors import exception_type_name

    refusal = _attachment_refusal(GUARD_TRACEBACK)
    assert refusal is not None
    exc = UnrealAttachmentRefused(refusal)
    assert exception_type_name(exc) == "InvalidState"
    assert exc.details["reason"] == "attachment_changed"
