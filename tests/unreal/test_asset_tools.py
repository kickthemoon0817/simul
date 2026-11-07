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
import sys
from contextlib import contextmanager, redirect_stdout
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional
from unittest.mock import AsyncMock

import pytest
from fastmcp import Client

from simul.adapters import unreal_runtime
from simul.adapters.unreal_runtime import MAX_ASSET_SEARCH_RESULTS, UnrealRuntimeSession
from simul.config import Settings
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

#: UsdStageImportOptions' saved settings as UE 5.7 ships them (its constructor defaults).
UE_DEFAULTS = {
    "import_actors": True,
    "import_geometry": True,
    "import_materials": True,
    "existing_actor_policy": "REPLACE",
    "existing_asset_policy": "REPLACE",
}


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


class World(UObject):
    pass


class MaterialInterface(UObject):
    pass


class MeshComponent(UObject):
    def __init__(self, path: str, editor: "Editor", slots: int = 1) -> None:
        super().__init__(path)
        self.editor = editor
        self.materials: List[Optional[MaterialInterface]] = [editor.basic] * slots

    def get_num_materials(self) -> int:
        return len(self.materials)

    def get_material(self, index: int) -> Optional[MaterialInterface]:
        return self.materials[index]

    def modify(self, always_mark_dirty: bool = True) -> bool:
        self.editor.events.append(("modify", self.path, self.editor.open_transactions))
        self.editor.dirty.add(self.path.split(".", 1)[0])
        return True

    def set_material(self, index: int, material: MaterialInterface) -> None:
        self.editor.events.append(("set_material", index, material.path, self.editor.open_transactions))
        self.materials[index] = material


class Actor(UObject):
    def __init__(self, path: str, world: World, editor: "Editor", exported: bool = True) -> None:
        super().__init__(path)
        self.world = world
        self.root_component = MeshComponent(path + ".StaticMeshComponent0", editor)
        self.root_component.owner = self  # type: ignore[attr-defined]
        self.exported = exported

    def get_world(self) -> World:
        return self.world

    def get_actor_label(self) -> str:
        return self.get_name()

    def get_components_by_class(self, cls: type) -> List[Any]:
        return [self.root_component] if isinstance(self.root_component, cls) else []


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
        self.maps_loaded: List[str] = []
        self.basic = MaterialInterface(BASIC + ".BasicShapeMaterial")
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
        self.content: Dict[str, str] = {"/Game/Maps/Main": "World", "/Game/Maps/Other": "World"}
        self.replaced_assets: List[str] = []
        # The project's saved settings (EditorPerProjectUserSettings.ini) and each write to them.
        self.saved_config = dict(UE_DEFAULTS)
        self.config_writes: List[Dict[str, Any]] = []
        self.stages: Dict[str, List[str]] = {}
        self.export_tasks: List[AssetExportTask] = []
        self.import_tasks: List[AssetImportTask] = []
        self.import_options: List[Any] = []  # what each import ran with; the script clears task.options
        self.export_cdo = SimpleNamespace(inner=Inner())
        # The console namespace Remote Control runs every script in.
        self.console: Dict[str, Any] = {}
        self.scripts: List[str] = []

    # -- what the editor does -------------------------------------------------

    def level_actors(self) -> List[Actor]:
        return list(self.actors.values())

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
        if self.content.get(package) == "World":
            self.maps_loaded.append(path)
            return World(path)
        return {BASIC: self.basic, RED: self.red}.get(package)

    def run_export(self, task: AssetExportTask) -> bool:
        """LevelExporterUSD: the selected actors of the task's world that it exports."""
        self.export_tasks.append(task)
        prims = ["/Root"] + [
            "/Root/" + actor.get_actor_label()
            for actor in self.selection
            if actor.get_world() is task.object and actor.exported
        ]
        self.stages[task.filename] = prims
        with open(task.filename, "w") as handle:
            handle.write("#usda 1.0\n")
        return True

    def run_import(self, tasks: List[AssetImportTask]) -> None:
        """The USD stage importer, with its existing-asset and existing-actor policies."""
        for task in tasks:
            self.import_tasks.append(task)
            options = task.options
            self.import_options.append(options)
            folder = task.destination_path + "/" + task.destination_name
            mesh = folder + "/StaticMeshes/SM_Root"
            if mesh in self.content:
                if options.existing_asset_policy == "REPLACE":
                    self.replaced_assets.append(mesh)
                elif options.existing_asset_policy == "APPEND":
                    mesh += "_1"
            self.content[mesh] = "StaticMesh"
            task.imported_object_paths = [mesh + "." + mesh.rsplit("/", 1)[1]]
            if options.import_actors:
                label = task.destination_name
                existing = [path for path, actor in self.actors.items() if actor.get_actor_label() == label]
                if existing and options.existing_actor_policy == "REPLACE":
                    for path in existing:
                        del self.actors[path]
                scene = Actor(LEVEL + label + "_" + str(len(self.import_tasks)), self.world, self)
                scene.get_actor_label = lambda label=label: label  # type: ignore[method-assign]
                self.actors[scene.path] = scene
                task.factory.scene_actor = scene

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

            def set_editor_property(self, name: str, value: Any, notify_mode: str = NOTIFY.DEFAULT) -> None:
                if name not in UE_DEFAULTS:
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

        class SavedImportOptions:
            def get_editor_property(self, name: str) -> Any:
                return editor.saved_config[name]

        class LevelExporterUSDOptions:
            pass

        def get_default_object(cls: type) -> Any:
            if cls is UsdStageImportOptions:
                return SavedImportOptions()
            if cls is LevelExporterUSDOptions:
                return editor.export_cdo
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
            def by_path(folder: str, recursive: bool = False) -> List[SimpleNamespace]:
                return [SimpleNamespace(package_name=package) for package in editor.content
                        if package.startswith(folder + "/")]

            def by_package(package: str) -> List[SimpleNamespace]:
                kind = editor.content.get(package)
                if kind is None:
                    return []
                return [SimpleNamespace(asset_class_path=SimpleNamespace(asset_name=kind))]

            return SimpleNamespace(get_assets_by_path=by_path, get_assets_by_package_name=by_package)

        names = {name: type(name, (), {}) for name in subsystems}
        return SimpleNamespace(
            Actor=Actor,
            MeshComponent=MeshComponent,
            MaterialInterface=MaterialInterface,
            AssetExportTask=AssetExportTask,
            AssetImportTask=AssetImportTask,
            LevelExporterUSD=LevelExporterUSD,
            LevelExporterUSDOptions=LevelExporterUSDOptions,
            UsdStageImportFactory=UsdStageImportFactory,
            UsdStageImportOptions=UsdStageImportOptions,
            PropertyAccessChangeNotifyMode=NOTIFY,
            ReplaceActorPolicy=SimpleNamespace(APPEND="APPEND", REPLACE="REPLACE", IGNORE="IGNORE"),
            ReplaceAssetPolicy=SimpleNamespace(APPEND="APPEND", REPLACE="REPLACE", IGNORE="IGNORE"),
            DirectoryPath=lambda path: SimpleNamespace(path=path),
            Exporter=SimpleNamespace(run_asset_export_task=editor.run_export),
            AssetToolsHelpers=SimpleNamespace(
                get_asset_tools=lambda: SimpleNamespace(import_asset_tasks=editor.run_import)
            ),
            AssetRegistryHelpers=SimpleNamespace(get_asset_registry=asset_registry),
            UsdConversionLibrary=SimpleNamespace(
                get_prim_path_for_object=lambda component, parent, folders, root: (
                    "/" + root + "/" + component.owner.get_actor_label()
                )
            ),
            SystemLibrary=SimpleNamespace(is_valid=lambda obj: obj is not None),
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
        return SimpleNamespace(Usd=SimpleNamespace(Stage=SimpleNamespace(
            LoadNone="LoadNone", Open=lambda path, load=None: Stage(editor.stages.get(path, [])),
        )))

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
        "material_path": RED + ".MI_Red",
        "slot_index": 0,
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


def test_a_map_path_given_as_the_material_is_not_loaded(editor: SimpleNamespace) -> None:
    result = _run(editor.session.assign_material(CUBE, "/Game/Maps/Other"))

    assert result["error_type"] == "NotFoundError"
    assert editor.fake.maps_loaded == []
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
