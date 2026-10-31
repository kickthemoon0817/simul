"""Unreal tools while Play In Editor (PIE) runs.

Observed on UE 5.7.4 while ``control_unreal_simulation`` had PIE running:
EditorActorSubsystem spawns nothing (``spawn_actor_from_class`` and
``spawn_actor_from_object`` return None), destroys nothing (``DestroyActor``
returns False), lists no level actors and selects none, and
``UnrealEditorSubsystem.get_editor_world()`` returns None. So
``spawn_unreal_actor`` told the agent to pass a different class or asset,
``delete_unreal_actor`` reported ``deleted: false`` with no reason, the level
reads returned 0 actors, ``get_unreal_loaded_map`` raised on None, and in
attached mode every call (``control_unreal_simulation`` stop included) failed
with "No editor world is available". ``LayersSubsystem.get_world()`` still
returns the editor world then: it returns GWorld, which Unreal keeps on the
editor world wherever Remote Control runs a script, also after the game opens
another level and the PIE world's package (``UEDPIE_<n>_<map>``) no longer
names the editor's map. ``focus_unreal_on_actor`` reported success as if it
had selected the actor: ``set_actor_selection_state`` is not refused, but what
it selects during PIE reads back as nothing and is cleared when PIE stops,
while the viewport camera does move.

Now an editor-world change Unreal refuses comes back as a ``PIEActive`` error
whose hint names the tool that stops PIE, the reads find the editor world and
report ``pie_active``, the attachment check finds it too, and focus moves the
camera and says the actor is not selected. The fake editor below answers the
editor scripts the way UE 5.7.4 did, in and out of PIE.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import math
import sys
from contextlib import contextmanager, redirect_stdout
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional

import pytest
from fastmcp import Client

from simul.adapters import _unreal_attach_scripts as attach_scripts
from simul.adapters import unreal_runtime
from simul.adapters._unreal_scripts import PIE_ACTIVE, PIE_STOP_HINT, pie_refusal
from simul.config import Settings
from simul.mcp import server as server_module
from simul.mcp.usage_tracker import ToolUsageTracker
from simul.utils.private_files import BridgeFiles
from tests.fakes import AvailableAdapter

LEVEL = "/Game/Maps/Lab.Lab:PersistentLevel."
VIEWPORT = "FourPanes2x2.Viewport 1.Viewport1"
ATTACHMENT_MODULE = "_simul_unreal_attachment_v1"


# ---------------------------------------------------------------------------
# A fake editor with UE 5.7.4's answers in and out of PIE
# ---------------------------------------------------------------------------


class _Class:
    def __init__(self, name: str) -> None:
        self.name = name

    def get_name(self) -> str:
        return self.name

    def get_path_name(self) -> str:
        return "/Script/Engine." + self.name


class _Vector:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.x, self.y, self.z = x, y, z


class Actor:
    """The fields ``actor_info`` and the listings read."""

    def __init__(
        self,
        name: str,
        class_name: str = "StaticMeshActor",
        tags: tuple = (),
        editable: bool = True,
    ) -> None:
        self.name, self.cls, self.tags, self.editable = (
            name,
            _Class(class_name),
            list(tags),
            editable,
        )
        self.root_component = SimpleNamespace(mobility=SimpleNamespace(name="STATIC"))
        self.world: Any = None
        # False for an actor Unreal will not select: hidden in the editor or in
        # a hidden or locked level, a child actor, an actor inside a level
        # instance (unreal_runtime.FOCUS_NOT_SELECTED_WARNING lists them).
        self.selectable = True

    def get_actor_label(self) -> str:
        return self.name

    def get_path_name(self) -> str:
        return LEVEL + self.name

    def get_class(self) -> _Class:
        return self.cls

    def get_actor_location(self) -> _Vector:
        return _Vector(1.0, 2.0, 3.0)

    def get_actor_rotation(self) -> SimpleNamespace:
        return SimpleNamespace(pitch=0.0, yaw=0.0, roll=0.0)

    def get_actor_scale3d(self) -> _Vector:
        return _Vector(1.0, 1.0, 1.0)

    def get_actor_bounds(
        self, only_colliding_components: bool, include_from_child_actors: bool = False
    ) -> tuple:
        """A 1 m box standing on the actor's location, in and out of PIE."""
        return _Vector(1.0, 2.0, 53.0), _Vector(50.0, 50.0, 50.0)

    def get_editor_property(self, name: str) -> Any:
        return {"tags": list(self.tags), "hidden": False}[name]

    def get_components_by_class(self, cls: Any) -> List[Any]:
        return []

    def get_all_child_actors(self, include_descendants: bool = True) -> List[Any]:
        return []

    def actor_has_tag(self, tag: str) -> bool:
        return tag in self.tags

    def is_editable(self) -> bool:
        return self.editable

    def get_world(self) -> Any:
        return self.world


class WorldSettings(Actor):
    def __init__(self) -> None:
        super().__init__("WorldSettings", "WorldSettings")


class World:
    def __init__(self, package: str, name: str) -> None:
        self.package, self.name = package, name

    def get_outermost(self) -> SimpleNamespace:
        return SimpleNamespace(get_name=lambda: self.package)

    def get_name(self) -> str:
        return self.name

    def get_path_name(self) -> str:
        return self.package + "." + self.name


class Editor:
    """The level and the editor subsystems, with ``pie`` switching their answers."""

    def __init__(self) -> None:
        self.pie = False
        self.editor_world = World("/Game/Maps/Lab", "Lab")
        self.pie_world = World("/Game/Maps/UEDPIE_0_Lab", "Lab")
        # GWorld, which LayersSubsystem.get_world() returns: the editor world
        # wherever Remote Control runs a script, in and out of PIE.
        self.gworld: Optional[World] = self.editor_world
        self.level = [
            Actor("Floor", tags=("floor",)),
            Actor("Crate", tags=("prop", "crate")),
            Actor("Lamp", "PointLight"),
        ]
        # Actors of the editor world Unreal's level listing leaves out.
        self.unlisted = [
            WorldSettings(),
            Actor("DefaultPhysicsVolume_0", "DefaultPhysicsVolume", editable=False),
        ]
        for actor in self.level + self.unlisted:
            actor.world = self.editor_world
        self.spawn_calls: List[str] = []
        self.transactions: List[str] = []
        self.calls: List[str] = []
        # The editor's selection, and the labels set_actor_selection_state got.
        self.selection: List[Actor] = []
        self.selection_calls: List[str] = []
        # The level viewport camera, as Remote Control reads and writes it.
        self.camera: Dict[str, Any] = {
            "CameraLocation": {"X": 0.0, "Y": 0.0, "Z": 1000.0},
            "CameraRotation": {"Pitch": 0.0, "Yaw": 0.0, "Roll": 0.0},
        }

    # EditorActorSubsystem: refuses everything during PIE.
    def get_all_level_actors(self) -> List[Actor]:
        return [] if self.pie else list(self.level)

    def spawn_actor_from_class(
        self, cls: Any, location: Any, rotation: Any = None
    ) -> Optional[Actor]:
        self.spawn_calls.append(getattr(cls, "name", str(cls)))
        if self.pie:
            return None
        actor = Actor(f"Spawned_{len(self.spawn_calls)}")
        self.level.append(actor)
        return actor

    spawn_actor_from_object = spawn_actor_from_class

    def destroy_actor(self, actor: Actor) -> bool:
        return False if self.pie else True

    def get_selected_level_actors(self) -> List[Actor]:
        return [] if self.pie else list(self.selection)

    def set_actor_selection_state(self, actor: Actor, should_be_selected: bool) -> None:
        """Not refused during PIE, but what it selects then reads back as
        nothing above, and Unreal clears it when PIE stops."""
        self.selection_calls.append(actor.get_actor_label())
        if not actor.selectable:
            return
        if should_be_selected and actor not in self.selection:
            self.selection.append(actor)
        elif not should_be_selected and actor in self.selection:
            self.selection.remove(actor)

    # UnrealEditorSubsystem's viewport camera over Remote Control: answered
    # in and out of PIE.
    def call_function(self, function_name: str, parameters: Any) -> Dict[str, Any]:
        if function_name == "SetLevelViewportCameraInfo":
            self.camera = dict(parameters)
            return {}
        if function_name == "GetLevelViewportCameraInfo":
            return {**self.camera, "ReturnValue": True}
        raise AssertionError("Unexpected Remote Control call: " + function_name)

    @contextmanager
    def transaction(self, description: str) -> Iterator[SimpleNamespace]:
        self.transactions.append(description)
        yield SimpleNamespace(cancel=lambda: None)

    def module(self) -> SimpleNamespace:
        editor = self

        def end_play() -> None:
            editor.pie = False

        level_editor = SimpleNamespace(
            is_in_play_in_editor=lambda: editor.pie,
            editor_request_end_play=end_play,
            on_map_changed=SimpleNamespace(add_callable=lambda callback: None),
            get_viewport_config_keys=lambda: [VIEWPORT],
            get_active_viewport_config_key=lambda: VIEWPORT,
            editor_get_game_view=lambda viewport: False,
            get_pilot_level_actor=lambda viewport: None,
        )
        unreal_editor = SimpleNamespace(
            get_editor_world=lambda: None if editor.pie else editor.editor_world,
            get_game_world=lambda: editor.pie_world if editor.pie else None,
        )
        subsystems = {
            "EditorActorSubsystem": editor,
            "LevelEditorSubsystem": level_editor,
            "UnrealEditorSubsystem": unreal_editor,
            "LayersSubsystem": SimpleNamespace(get_world=lambda: editor.gworld),
            "StaticMeshEditorSubsystem": SimpleNamespace(),
        }

        def get_all_actors_of_class(world: Any, cls: Any) -> List[Actor]:
            editor.calls.append("GameplayStatics.get_all_actors_of_class")
            assert world is editor.editor_world, "the editor world, not the PIE world"
            return list(editor.level) + list(editor.unlisted)

        def load_object(outer: Any, path: str) -> Any:
            editor.calls.append("load_object " + path)
            if path == "/Engine/BasicShapes/Cube.Cube":
                return SimpleNamespace(get_class=lambda: _Class("StaticMesh"))
            return next((a for a in editor.level if a.get_path_name() == path), None)

        return SimpleNamespace(
            Actor=Actor,
            ActorComponent=object,
            PrimitiveComponent=object,
            DrawFrustumComponent=type("DrawFrustumComponent", (), {}),
            WorldSettings=WorldSettings,
            Class=_Class,
            Vector=_Vector,
            Rotator=lambda **kwargs: SimpleNamespace(**kwargs),
            StaticMeshActor=_Class("StaticMeshActor"),
            StaticMesh=type("StaticMesh", (), {}),
            EditorActorSubsystem="EditorActorSubsystem",
            LevelEditorSubsystem="LevelEditorSubsystem",
            UnrealEditorSubsystem="UnrealEditorSubsystem",
            LayersSubsystem="LayersSubsystem",
            StaticMeshEditorSubsystem="StaticMeshEditorSubsystem",
            get_editor_subsystem=subsystems.__getitem__,
            GameplayStatics=SimpleNamespace(
                get_all_actors_of_class=get_all_actors_of_class,
                get_platform_name=lambda: "Mac",
                is_game_paused=lambda world: False,
                get_time_seconds=lambda world: 12.5,
            ),
            SystemLibrary=SimpleNamespace(
                get_engine_version=lambda: "5.7.4", get_game_name=lambda: "Lab"
            ),
            Paths=SimpleNamespace(
                project_dir=lambda: "../../Lab/",
                convert_relative_path_to_full=lambda path: "/Projects/Lab/",
                get_project_file_path=lambda: "/Projects/Lab/Lab.uproject",
            ),
            EditorLoadingAndSavingUtils=SimpleNamespace(
                get_dirty_map_packages=lambda: []
            ),
            ScopedEditorTransaction=editor.transaction,
            MathLibrary=SimpleNamespace(class_is_child_of=lambda cls, parent: True),
            load_object=load_object,
            load_asset=lambda path: None,
            UsdStageImportFactory=object,
            LevelExporterUSD=object,
        )

    def run(self, code: str) -> Dict[str, Any]:
        """``ExecutePythonCommandEx`` in ExecuteFile mode."""
        out = io.StringIO()
        try:
            with redirect_stdout(out):
                exec(code, {})
        except Exception as exc:
            return {"ReturnValue": False, "CommandResult": repr(exc)}
        return {
            "ReturnValue": True,
            "LogOutput": [
                {"Type": "Info", "Output": line} for line in out.getvalue().splitlines()
            ],
        }


@pytest.fixture
def editor(monkeypatch: pytest.MonkeyPatch) -> Editor:
    level = Editor()
    monkeypatch.setitem(sys.modules, "unreal", level.module())
    # The attachment scripts install a module in sys.modules; start without
    # one and remove the one a test installs.
    monkeypatch.setitem(sys.modules, ATTACHMENT_MODULE, None)
    monkeypatch.delitem(sys.modules, ATTACHMENT_MODULE)
    return level


def _session(
    editor: Editor, monkeypatch: pytest.MonkeyPatch, settings: Optional[Settings] = None
) -> Any:
    session = unreal_runtime.UnrealRuntimeSession(settings or Settings())
    session.scripts = []  # type: ignore[attr-defined]
    session.calls = []  # type: ignore[attr-defined]

    async def execute(code: str, mode: str = "ExecuteFile") -> Dict[str, Any]:
        session.scripts.append(code)  # type: ignore[attr-defined]
        return editor.run(code)

    async def call_function(
        object_path: str, function_name: str, parameters: Any = None
    ) -> Dict[str, Any]:
        session.calls.append(function_name)  # type: ignore[attr-defined]
        return editor.call_function(function_name, parameters)

    monkeypatch.setattr(session, "_execute_python", execute)
    monkeypatch.setattr(session, "_call_function", call_function)
    return session


def _errors(caplog: pytest.LogCaptureFixture) -> List[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.ERROR
    ]


# ---------------------------------------------------------------------------
# Editor-world changes Unreal refuses during PIE
# ---------------------------------------------------------------------------


def test_the_refusal_names_the_tool_that_stops_pie() -> None:
    refusal = pie_refusal("spawn actors")

    assert refusal == {
        "success": False,
        "error_type": PIE_ACTIVE,
        "error": "Play In Editor is running, and Unreal's editor cannot spawn actors until it stops",
        "details": {"hint": PIE_STOP_HINT},
    }
    assert "control_unreal_simulation" in PIE_STOP_HINT and "'stop'" in PIE_STOP_HINT
    assert "simul unreal sim stop" in PIE_STOP_HINT


def test_spawn_during_pie_is_refused_before_the_path_is_resolved(
    editor: Editor, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """It used to be a SpawnError telling the agent to pass another class or asset."""
    editor.pie = True
    caplog.set_level(logging.INFO, logger="simul")

    result = asyncio.run(
        _session(editor, monkeypatch).spawn_actor("/Engine/BasicShapes/Cube.Cube")
    )

    assert result == pie_refusal("spawn actors")
    assert editor.spawn_calls == [] and not any(
        call.startswith("load_object") for call in editor.calls
    )
    # A refusal the caller can act on, not a server fault.
    assert _errors(caplog) == []
    assert any(
        "Unreal refused the request (PIEActive)" in r.getMessage()
        for r in caplog.records
    )


def test_spawn_outside_pie_still_spawns(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = asyncio.run(
        _session(editor, monkeypatch).spawn_actor("/Engine/BasicShapes/Cube.Cube")
    )

    assert result["actor_path"] == LEVEL + "Spawned_1"
    assert len(editor.spawn_calls) == 1 and editor.transactions == [
        "simul: spawn actor"
    ]


def test_a_procedural_scene_during_pie_is_refused_before_its_transaction(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    editor.pie = True

    result = asyncio.run(
        _session(editor, monkeypatch).generate_procedural_scene("room")
    )

    assert result == pie_refusal("spawn actors")
    assert editor.transactions == [] and editor.spawn_calls == []


def test_delete_during_pie_is_refused_instead_of_reporting_deleted_false(
    editor: Editor, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    editor.pie = True
    session = _session(editor, monkeypatch)
    calls: List[str] = []

    async def call_function(
        object_path: str, function_name: str, parameters: Any = None
    ) -> Dict[str, Any]:
        calls.append(function_name)
        return {"ReturnValue": False}  # DestroyActor refuses during PIE

    monkeypatch.setattr(session, "_call_function", call_function)
    caplog.set_level(logging.INFO, logger="simul")

    result = asyncio.run(session.delete_actor(LEVEL + "Crate"))

    assert calls == ["DestroyActor"]
    assert result == pie_refusal("delete actors")
    assert _errors(caplog) == []


@pytest.mark.parametrize("reply", [{"ReturnValue": False}, {"ReturnValue": True}])
def test_delete_outside_pie_reports_what_unreal_did(
    editor: Editor, monkeypatch: pytest.MonkeyPatch, reply: Dict[str, Any]
) -> None:
    session = _session(editor, monkeypatch)

    async def call_function(
        object_path: str, function_name: str, parameters: Any = None
    ) -> Dict[str, Any]:
        return reply

    monkeypatch.setattr(session, "_call_function", call_function)

    result = asyncio.run(session.delete_actor(LEVEL + "Crate"))

    assert result == {"actor_path": LEVEL + "Crate", "deleted": reply["ReturnValue"]}
    # The PIE state is read only to explain a delete Unreal refused.
    assert len(session.scripts) == (0 if reply["ReturnValue"] else 1)


def test_delete_reports_deleted_false_when_the_pie_state_cannot_be_read(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _session(editor, monkeypatch)

    async def call_function(
        object_path: str, function_name: str, parameters: Any = None
    ) -> Dict[str, Any]:
        return {"ReturnValue": False}

    async def unreachable(code: str, mode: str = "ExecuteFile") -> Dict[str, Any]:
        raise ConnectionError("editor went away")

    monkeypatch.setattr(session, "_call_function", call_function)
    monkeypatch.setattr(session, "_execute_python", unreachable)

    assert asyncio.run(session.delete_actor(LEVEL + "Crate")) == {
        "actor_path": LEVEL + "Crate",
        "deleted": False,
    }


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("generate_mesh_primitive", {"primitive_type": "box"}),
        ("compute_convex_hull", {"mesh_path": LEVEL + "Crate"}),
        ("decompose_convex_hull", {"mesh_path": LEVEL + "Crate"}),
        (
            "convert_mesh_format",
            {"mesh_path": "/Game/Meshes/SM_Crate", "target_format": "dynamic_mesh"},
        ),
    ],
)
def test_mesh_tools_that_spawn_actors_are_refused_during_pie(
    editor: Editor, monkeypatch: pytest.MonkeyPatch, method: str, kwargs: Dict[str, Any]
) -> None:
    """Each spawns its result with EditorActorSubsystem, which returns None during PIE."""
    session = _session(editor, monkeypatch)

    editor.pie = True
    refused = asyncio.run(getattr(session, method)(**kwargs))
    editor.pie = False
    ran = asyncio.run(getattr(session, method)(**kwargs))

    assert refused["success"] is False and refused["error_type"] == PIE_ACTIVE
    assert refused["error"] == pie_refusal("spawn actors")["error"]
    # Outside PIE the body runs: the fake editor has no GeometryScript, so it raises.
    assert "GeometryScript" in ran["error"] or "EditorAssetLibrary" in ran["error"]


@pytest.mark.parametrize(
    ("method", "kwargs", "action"),
    [
        ("import_usd", {"usd_path": "examples/whale.usd"}, "import USD into the level"),
        (
            "export_usd",
            {
                "actor_paths": [LEVEL + "Crate"],
                "output_path": "/tmp/simul-work/pie.usda",
            },
            "export actors to USD",
        ),
    ],
)
def test_usd_import_and_export_are_refused_during_pie(
    editor: Editor,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    kwargs: Dict[str, Any],
    action: str,
) -> None:
    """Import finds its actors in the level listing and export selects them and
    exports the editor world, none of which Unreal allows during PIE."""
    editor.pie = True

    result = asyncio.run(getattr(_session(editor, monkeypatch), method)(**kwargs))

    assert result["error_type"] == PIE_ACTIVE
    assert result["error"] == pie_refusal(action)["error"]


# ---------------------------------------------------------------------------
# Material instances: an asset library Unreal withholds during PIE
# ---------------------------------------------------------------------------


class MaterialInterface:
    def __init__(self, path: str) -> None:
        self.path = path
        self.parent: Optional[MaterialInterface] = None

    def get_path_name(self) -> str:
        return self.path

    def get_class(self) -> _Class:
        return _Class(type(self).__name__)


class MaterialInstanceConstant(MaterialInterface):
    pass


class AssetLibrary:
    """EditorAssetLibrary, AssetTools and MaterialEditingLibrary over a few assets.

    Observed on UE 5.7.4 while PIE ran: EditorAssetLibrary loads, finds and
    saves nothing (load_asset returns None, does_asset_exist and
    save_loaded_asset False, for assets that exist; the log says "The Editor is
    currently in a play mode"), while AssetTools.create_asset still creates.
    """

    def __init__(self, editor: Editor) -> None:
        self.editor = editor
        self.assets: Dict[str, MaterialInterface] = {
            "/Game/Materials/M_Wood": MaterialInterface("/Game/Materials/M_Wood.M_Wood")
        }
        self.calls: List[str] = []
        self.saved: List[str] = []

    def load_asset(self, path: str) -> Optional[MaterialInterface]:
        self.calls.append("load_asset")
        return None if self.editor.pie else self.assets.get(path.split(".")[0])

    def does_asset_exist(self, path: str) -> bool:
        self.calls.append("does_asset_exist")
        return not self.editor.pie and path.split(".")[0] in self.assets

    def save_loaded_asset(self, asset: MaterialInterface) -> bool:
        self.calls.append("save_loaded_asset")
        if self.editor.pie:
            return False
        self.saved.append(asset.get_path_name())
        return True

    def create_asset(
        self, name: str, folder: str, cls: Any, factory: Any
    ) -> MaterialInterface:
        self.calls.append("create_asset")
        asset = cls(f"{folder}/{name}.{name}")
        self.assets[f"{folder}/{name}"] = asset
        return asset

    def set_material_instance_parent(
        self, instance: MaterialInterface, parent: MaterialInterface
    ) -> None:
        self.calls.append("set_material_instance_parent")
        instance.parent = parent

    def module_attributes(self) -> Dict[str, Any]:
        return {
            "EditorAssetLibrary": SimpleNamespace(
                load_asset=self.load_asset,
                does_asset_exist=self.does_asset_exist,
                save_loaded_asset=self.save_loaded_asset,
            ),
            "AssetToolsHelpers": SimpleNamespace(
                get_asset_tools=lambda: SimpleNamespace(create_asset=self.create_asset)
            ),
            "MaterialEditingLibrary": SimpleNamespace(
                set_material_instance_parent=self.set_material_instance_parent
            ),
            "MaterialInterface": MaterialInterface,
            "MaterialInstanceConstant": MaterialInstanceConstant,
            "MaterialInstanceConstantFactoryNew": SimpleNamespace,
        }


@pytest.fixture
def asset_library(editor: Editor, monkeypatch: pytest.MonkeyPatch) -> AssetLibrary:
    library = AssetLibrary(editor)
    for name, value in library.module_attributes().items():
        monkeypatch.setattr(sys.modules["unreal"], name, value, raising=False)
    return library


def test_a_material_instance_during_pie_is_refused_before_any_asset_call(
    editor: Editor,
    asset_library: AssetLibrary,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """It used to be a ScriptError saying the parent material, which exists,
    was not found: EditorAssetLibrary.load_asset returns None during PIE."""
    editor.pie = True
    caplog.set_level(logging.INFO, logger="simul")

    result = asyncio.run(
        _session(editor, monkeypatch).create_material_instance(
            "/Game/Materials/M_Wood", "MI_Chair"
        )
    )

    assert result == pie_refusal("create material instances")
    # Refused up front: nothing is created, although AssetTools still could.
    assert asset_library.calls == []
    assert "/Game/Materials/MI_Chair" not in asset_library.assets
    assert _errors(caplog) == []
    assert any(
        "Unreal refused the request (PIEActive)" in r.getMessage()
        for r in caplog.records
    )


def test_a_material_instance_outside_pie_is_created_parented_and_saved(
    editor: Editor, asset_library: AssetLibrary, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = asyncio.run(
        _session(editor, monkeypatch).create_material_instance(
            "/Game/Materials/M_Wood", "MI_Chair"
        )
    )

    assert result == {
        "instance_path": "/Game/Materials/MI_Chair.MI_Chair",
        "parent_path": "/Game/Materials/M_Wood.M_Wood",
        "class_name": "MaterialInstanceConstant",
    }
    instance = asset_library.assets["/Game/Materials/MI_Chair"]
    assert instance.parent is asset_library.assets["/Game/Materials/M_Wood"]
    assert asset_library.saved == ["/Game/Materials/MI_Chair.MI_Chair"]


@pytest.mark.parametrize(
    ("parent_path", "instance_name", "error"),
    [
        (
            "/Game/Materials/M_Missing",
            "MI_Chair",
            "Parent material not found: /Game/Materials/M_Missing",
        ),
        (
            "/Game/Materials/M_Wood",
            "M_Wood",
            "Asset already exists: /Game/Materials/M_Wood",
        ),
    ],
)
def test_a_material_instance_outside_pie_reports_a_missing_parent_or_a_taken_name(
    editor: Editor,
    asset_library: AssetLibrary,
    monkeypatch: pytest.MonkeyPatch,
    parent_path: str,
    instance_name: str,
    error: str,
) -> None:
    result = asyncio.run(
        _session(editor, monkeypatch).create_material_instance(
            parent_path, instance_name
        )
    )

    assert result["success"] is False and result["error_type"] == "ScriptError"
    assert result["error"] == error
    assert "create_asset" not in asset_library.calls


# ---------------------------------------------------------------------------
# Reads during PIE: the editor world, flagged
# ---------------------------------------------------------------------------


def test_list_actors_during_pie_reads_the_editor_world(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It used to return 0 actors without saying why."""
    session = _session(editor, monkeypatch)

    editor.pie = True
    during = asyncio.run(session.list_actors())
    editor.pie = False
    outside = asyncio.run(session.list_actors())

    assert during["pie_active"] is True and outside["pie_active"] is False
    # The same actors either way: the WorldSettings and the non-editable
    # helper Unreal's listing leaves out stay out.
    assert [a["path"] for a in during["actors"]] == [
        a["path"] for a in outside["actors"]
    ]
    assert [a["name"] for a in during["actors"]] == ["Floor", "Crate", "Lamp"]
    assert editor.calls.count("GameplayStatics.get_all_actors_of_class") == 1


def test_list_actors_filters_apply_during_pie(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    editor.pie = True
    session = _session(editor, monkeypatch)

    tagged = asyncio.run(session.list_actors(tag_filter="prop"))
    lights = asyncio.run(session.list_actors(class_filter="PointLight"))

    assert [a["name"] for a in tagged["actors"]] == ["Crate"]
    assert [a["name"] for a in lights["actors"]] == ["Lamp"]


def test_semantic_label_lookup_during_pie_finds_editor_actors(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    editor.pie = True

    result = asyncio.run(
        _session(editor, monkeypatch).get_actor_by_semantic_label("crate")
    )

    assert [a["path"] for a in result["actors"]] == [LEVEL + "Crate"]
    assert result["total_matches"] == 1 and result["pie_active"] is True


def test_scene_summary_during_pie_names_the_editor_map(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    editor.pie = True

    result = asyncio.run(_session(editor, monkeypatch).summarize_scene())

    assert result["map_path"] == "/Game/Maps/Lab.Lab"
    assert result["total_actors"] == 3
    assert result["pie_active"] is True
    assert "Play In Editor is running" in result["summary_text"]


def test_robotics_analysis_during_pie_reads_the_editor_world(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    editor.pie = True

    result = asyncio.run(
        _session(editor, monkeypatch).analyze_scene_for_robotics(["graspability"])
    )

    assert result["total_actors_analyzed"] == 3
    assert result["pie_active"] is True


def test_loaded_map_engine_info_and_actor_info_during_pie(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """get_unreal_loaded_map used to raise on the None editor world, and
    get_unreal_engine_info reported an empty loaded_map."""
    editor.pie = True
    session = _session(editor, monkeypatch)

    loaded = asyncio.run(session.get_loaded_map())
    info = asyncio.run(session.get_engine_info())
    actor = asyncio.run(session.get_actor_info(LEVEL + "Crate"))

    assert loaded == {"map_path": "/Game/Maps/Lab.Lab", "pie_active": True}
    assert info["loaded_map"] == "/Game/Maps/Lab.Lab" and info["pie_active"] is True
    assert actor["name"] == "Crate" and actor["pie_active"] is True


def test_reads_outside_pie_report_pie_active_false(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _session(editor, monkeypatch)

    assert asyncio.run(session.get_loaded_map()) == {
        "map_path": "/Game/Maps/Lab.Lab",
        "pie_active": False,
    }
    assert asyncio.run(session.get_engine_info())["pie_active"] is False
    assert asyncio.run(session.summarize_scene())["pie_active"] is False
    assert "GameplayStatics.get_all_actors_of_class" not in editor.calls


# ---------------------------------------------------------------------------
# Focus during PIE: the camera moves, the selection does not
# ---------------------------------------------------------------------------

# Where focus_on_actor puts the camera for an actor whose bounds are a 1 m box
# centered at (1, 2, 53): far enough behind the center that the box's bounding
# sphere fills the height of UE's default view (90 degrees across, 16:9) with
# 10% to spare, half that distance above it, pitched down at it.
FOCUS_DISTANCE = (
    1.1 * 50.0 * math.sqrt(3.0) / math.sin(math.atan(9.0 / 16.0)) / math.hypot(1.0, 0.5)
)
FOCUS_CAMERA = (
    pytest.approx((1.0 - FOCUS_DISTANCE, 2.0, 53.0 + FOCUS_DISTANCE / 2)),
    pytest.approx((-math.degrees(math.atan2(0.5, 1.0)), 0.0, 0.0)),
)


def test_focus_during_pie_moves_the_camera_and_says_the_actor_is_not_selected(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It reported success as if the actor were selected. On UE 5.7.4 a
    selection made during PIE reads back as none and is cleared when PIE
    stops, so focus selects nothing then; the camera does move."""
    editor.pie = True
    session = _session(editor, monkeypatch)

    result = asyncio.run(session.focus_on_actor(LEVEL + "Crate"))

    assert result == {
        "actor_path": LEVEL + "Crate",
        "camera_location": FOCUS_CAMERA[0],
        "camera_rotation": FOCUS_CAMERA[1],
        "distance": pytest.approx(FOCUS_DISTANCE),
        "selected": False,
        "pie_active": True,
        "warning": unreal_runtime.FOCUS_PIE_WARNING,
    }
    assert editor.selection_calls == []
    assert session.calls == ["SetLevelViewportCameraInfo", "GetLevelViewportCameraInfo"]
    assert (
        "control_unreal_simulation" in result["warning"]
        and "'stop'" in result["warning"]
    )


def test_focus_outside_pie_selects_the_actor_and_reads_the_selection_back(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _session(editor, monkeypatch)

    result = asyncio.run(session.focus_on_actor(LEVEL + "Crate"))

    assert result == {
        "actor_path": LEVEL + "Crate",
        "camera_location": FOCUS_CAMERA[0],
        "camera_rotation": FOCUS_CAMERA[1],
        "distance": pytest.approx(FOCUS_DISTANCE),
        "selected": True,
        "pie_active": False,
    }
    assert [actor.name for actor in editor.selection] == ["Crate"]
    # One script selects, reads the selection back and reads the location.
    assert len(session.scripts) == 1


def test_focus_reports_an_actor_unreal_does_not_select(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Outside PIE too, Unreal skips an actor hidden in the editor without a
    word (UE 5.7.4); that was reported as success as well."""
    editor.level[1].selectable = False
    session = _session(editor, monkeypatch)

    result = asyncio.run(session.focus_on_actor(LEVEL + "Crate"))

    assert editor.selection_calls == ["Crate"] and editor.selection == []
    assert (result["selected"], result["pie_active"]) == (False, False)
    assert result["warning"] == unreal_runtime.FOCUS_NOT_SELECTED_WARNING
    # Unreal also skips child actors and actors inside a level instance
    # (selected only through their parent), so the warning cannot blame
    # visibility alone.
    assert "child actor" in result["warning"] and "level instance" in result["warning"]
    assert result["camera_location"] == FOCUS_CAMERA[0]


def test_focus_on_a_missing_actor_fails_before_the_camera_moves(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _session(editor, monkeypatch)

    result = asyncio.run(session.focus_on_actor(LEVEL + "Missing"))

    assert result["success"] is False and result["error_type"] == "ScriptError"
    assert "Actor not found: " + LEVEL + "Missing" in result["error"]
    assert session.calls == [] and editor.selection_calls == []


# ---------------------------------------------------------------------------
# Attached mode during PIE
# ---------------------------------------------------------------------------


def _attached_run(editor: Editor, code: str) -> Dict[str, Any]:
    result = editor.run(code)
    assert result["ReturnValue"], result.get("CommandResult")
    return json.loads(result["LogOutput"][-1]["Output"])


def test_the_attachment_identity_and_check_work_during_pie(editor: Editor) -> None:
    """Every attached call verifies the attachment first, and the check read
    get_editor_world(): during PIE each call, stop included, failed with "No
    editor world is available"."""
    before = _attached_run(editor, attach_scripts.EDITOR_IDENTITY)
    editor.pie = True
    during = _attached_run(editor, attach_scripts.EDITOR_IDENTITY)
    target = {
        key: before[key]
        for key in ("instance_id", "document_id", "project_path", "map_path")
    }
    verified = _attached_run(
        editor,
        attach_scripts.EDITOR_STATE
        + f"_simul_editor.verify({dict(target, viewport=VIEWPORT)!r})\nprint(json.dumps({{'verified': True}}))\n",
    )

    assert during == before
    assert during["map_path"] == "/Game/Maps/Lab.Lab"
    assert verified == {"verified": True}


def test_an_editor_with_an_older_attachment_module_gets_the_current_identity(
    editor: Editor,
) -> None:
    """An editor keeps the module the first simul to attach installed; its
    identity() and verify() are replaced on every call, its ids are kept."""
    old = SimpleNamespace(
        instance_id="old-instance",
        document_id="old-document",
        identity=lambda: (_ for _ in ()).throw(
            RuntimeError("No editor world is available")
        ),
    )
    module = type(sys)(ATTACHMENT_MODULE)
    module.__dict__.update(vars(old))
    sys.modules[ATTACHMENT_MODULE] = module
    editor.pie = True

    identity = _attached_run(editor, attach_scripts.EDITOR_IDENTITY)

    assert (identity["instance_id"], identity["document_id"]) == (
        "old-instance",
        "old-document",
    )
    assert identity["map_path"] == "/Game/Maps/Lab.Lab"


@pytest.mark.parametrize(
    "action", ["select_actor", "set_property", "set_game_view", "clear_selection"]
)
def test_named_controls_are_refused_during_pie_but_inspect_works(
    editor: Editor, action: str
) -> None:
    """They raised a ValueError, which arrived as a traceback; now the PIEActive envelope."""
    args = {
        "action": action,
        "target": None,
        "property_name": None,
        "value": None,
        "enabled": None,
        "viewport": VIEWPORT,
        "actions": [],
        "agent_id": "agent",
        "position": None,
        "activity": None,
    }
    script = (
        attach_scripts.EDITOR_STATE + f"args = {args!r}\n" + attach_scripts.CONTROL_UI
    )
    editor.pie = True

    refused = _attached_run(editor, script)
    inspected = _attached_run(
        editor, script.replace(f"'action': '{action}'", "'action': 'inspect'")
    )

    assert refused == pie_refusal("change editor controls")
    assert (
        inspected["is_playing"] is True
        and inspected["map_path"] == "/Game/Maps/Lab.Lab"
    )


# ---------------------------------------------------------------------------
# PIE on the editor's level, PIE after the game opens another level
# ---------------------------------------------------------------------------


def _play(editor: Editor, game_map: str) -> None:
    """Run PIE with the game on ``game_map``; "Other" is a level the game opened."""
    editor.pie = True
    editor.pie_world = World(f"/Game/Maps/UEDPIE_0_{game_map}", game_map)


class _RemoteControl:
    """``PUT /remote/object/call`` answered by running its Python in the fake editor."""

    def __init__(self, editor: Editor) -> None:
        self.editor = editor

    def put(self, path: str, json: Dict[str, Any]) -> "_Reply":
        assert path == "/remote/object/call"
        return _Reply(self.editor.run(json["parameters"]["PythonCommand"]))


class _Reply:
    status = 200

    def __init__(self, body: Dict[str, Any]) -> None:
        self.body = body

    async def __aenter__(self) -> "_Reply":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    async def json(self) -> Dict[str, Any]:
        return self.body


def _attached_session(
    editor: Editor, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> Any:
    """An attached-mode session, attached to ``editor`` as it is now; the
    attachment check guards each of its calls, as over Remote Control."""
    identity = _attached_run(editor, attach_scripts.EDITOR_IDENTITY)
    target = {
        key: identity[key]
        for key in ("instance_id", "document_id", "project_path", "map_path")
    }
    path = tmp_path / "attachment.json"
    BridgeFiles.write(
        path,
        {
            "version": 1,
            "host": "localhost",
            "port": 30010,
            "target": dict(target, viewport=VIEWPORT),
        },
    )
    session = unreal_runtime.UnrealRuntimeSession(
        Settings(unreal={"mode": "attached", "attachment_path": str(path)})
    )
    remote_control = _RemoteControl(editor)

    async def http_session(timeout_override: Optional[float] = None) -> Any:
        return remote_control

    monkeypatch.setattr(session, "_ensure_http_session", http_session)
    return session


@pytest.mark.parametrize("game_map", ["Lab", "Other"])
def test_reads_during_pie_find_the_editor_world_whichever_level_the_game_is_on(
    editor: Editor, monkeypatch: pytest.MonkeyPatch, game_map: str
) -> None:
    """After the game opened another level, the PIE world's package no longer
    named the editor's map: the reads returned 0 actors with pie_active true and
    get_unreal_loaded_map raised."""
    _play(editor, game_map)
    session = _session(editor, monkeypatch)

    listing = asyncio.run(session.list_actors())
    loaded = asyncio.run(session.get_loaded_map())
    info = asyncio.run(session.get_engine_info())
    summary = asyncio.run(session.summarize_scene())

    assert [a["name"] for a in listing["actors"]] == ["Floor", "Crate", "Lamp"]
    assert listing["pie_active"] is True
    assert loaded == {"map_path": "/Game/Maps/Lab.Lab", "pie_active": True}
    assert info["loaded_map"] == "/Game/Maps/Lab.Lab"
    assert summary["map_path"] == "/Game/Maps/Lab.Lab"
    assert summary["total_actors"] == 3


@pytest.mark.parametrize("game_map", ["Lab", "Other"])
def test_attached_calls_and_the_pie_stop_work_whichever_level_the_game_is_on(
    editor: Editor, monkeypatch: pytest.MonkeyPatch, tmp_path: Any, game_map: str
) -> None:
    """Each attached call checks the attachment against the editor world first;
    with the game on another level the check raised "No editor world is
    available", so every call was refused, the PIE stop included."""
    session = _attached_session(editor, monkeypatch, tmp_path)
    _play(editor, game_map)

    async def calls() -> List[Dict[str, Any]]:
        return [
            await session.attachment_status(),
            await session.get_loaded_map(),
            await session.control_simulation("stop"),
        ]

    status, loaded, stopped = asyncio.run(calls())

    assert status["map_path"] == "/Game/Maps/Lab.Lab"
    assert loaded == {"map_path": "/Game/Maps/Lab.Lab", "pie_active": True}
    assert stopped == {"action": "stop", "state": "stopped"}
    assert editor.pie is False


@pytest.mark.parametrize("gworld", ["none", "the PIE world"])
def test_without_an_editor_world_nothing_reads_the_pie_world_and_attached_calls_are_refused(
    editor: Editor, monkeypatch: pytest.MonkeyPatch, tmp_path: Any, gworld: str
) -> None:
    """GWorld holding no world, or a PIE world (as while Unreal ticks the game),
    leaves no editor world to read: the reads find none instead of the PIE world,
    and the attachment check refuses every call."""
    attached = _attached_session(editor, monkeypatch, tmp_path)
    _play(editor, "Other")
    editor.gworld = editor.pie_world if gworld == "the PIE world" else None
    session = _session(editor, monkeypatch)

    listing = asyncio.run(session.list_actors())
    loaded = asyncio.run(session.get_loaded_map())
    info = asyncio.run(session.get_engine_info())
    stop = asyncio.run(attached.control_simulation("stop"))

    assert listing["actors"] == [] and listing["pie_active"] is True
    assert "GameplayStatics.get_all_actors_of_class" not in editor.calls
    assert loaded["error_type"] == "ScriptError"
    assert "No editor world is loaded" in loaded["error"]
    assert info["loaded_map"] == "" and info["pie_active"] is True
    assert stop["success"] is False
    assert "No editor world is available" in stop["error"]
    assert editor.pie is True


# ---------------------------------------------------------------------------
# Through the MCP envelope
# ---------------------------------------------------------------------------


class _Adapter(AvailableAdapter):
    def __init__(self, session: Any) -> None:
        super().__init__(session.settings)
        self.session = session

    @contextmanager
    def create_session(self) -> Iterator[Any]:
        yield self.session


def _call(
    editor: Editor,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    tool: str,
    arguments: Dict[str, Any],
) -> Any:
    settings = Settings(
        unreal={
            "tool_surface": "full",
            "attachment_path": str(tmp_path / "attachment.json"),
        },
        security={"rate_limiting_enabled": False},
    )
    instance = server_module.SimulMCPServer(settings, backends={"unreal"})
    instance.usage_tracker = ToolUsageTracker(log_dir=tmp_path / "usage")
    instance.unreal_adapter = _Adapter(_session(editor, monkeypatch, settings))

    async def run() -> Any:
        async with Client(instance.mcp) as client:
            return await client.call_tool(tool, arguments, raise_on_error=False)

    result = asyncio.run(run())
    return json.loads(
        [
            block.text
            for block in result.content
            if getattr(block, "type", None) == "text"
        ][-1]
    )


def test_an_agent_sees_the_refusal_and_the_hint(
    editor: Editor,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    editor.pie = True
    caplog.set_level(logging.INFO, logger="simul")

    payload = _call(
        editor,
        monkeypatch,
        tmp_path,
        "spawn_unreal_actor",
        {"asset_path": "/Script/Engine.PointLight"},
    )

    assert payload == pie_refusal("spawn actors")
    assert _errors(caplog) == []


def test_an_agent_creating_a_material_instance_during_pie_sees_the_refusal(
    editor: Editor,
    asset_library: AssetLibrary,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    editor.pie = True

    payload = _call(
        editor,
        monkeypatch,
        tmp_path,
        "create_unreal_material_instance",
        {"parent_path": "/Game/Materials/M_Wood", "instance_name": "MI_Chair"},
    )

    assert payload == pie_refusal("create material instances")
    assert asset_library.calls == []


def test_an_agent_sees_pie_active_on_the_reads(
    editor: Editor, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    editor.pie = True

    listing = _call(editor, monkeypatch, tmp_path, "list_unreal_actors", {})
    found = _call(
        editor,
        monkeypatch,
        tmp_path,
        "get_unreal_actor_by_semantic_label",
        {"label": "floor"},
    )
    loaded = _call(editor, monkeypatch, tmp_path, "get_unreal_loaded_map", {})

    assert (
        listing["success"] is True
        and listing["pie_active"] is True
        and listing["count"] == 3
    )
    assert found["pie_active"] is True and [a["name"] for a in found["actors"]] == [
        "Floor"
    ]
    assert loaded == {
        "success": True,
        "error": None,
        "map_path": "/Game/Maps/Lab.Lab",
        "pie_active": True,
    }


def test_an_agent_sees_whether_focus_selected_the_actor(
    editor: Editor, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """The response schema keeps selected, pie_active and the warning."""
    editor.pie = True
    during = _call(
        editor,
        monkeypatch,
        tmp_path,
        "focus_unreal_on_actor",
        {"actor_path": LEVEL + "Crate"},
    )
    editor.pie = False
    outside = _call(
        editor,
        monkeypatch,
        tmp_path,
        "focus_unreal_on_actor",
        {"actor_path": LEVEL + "Crate"},
    )

    assert during["success"] is True and during["camera_location"] == FOCUS_CAMERA[0]
    assert (during["selected"], during["pie_active"], during["warning"]) == (
        False,
        True,
        unreal_runtime.FOCUS_PIE_WARNING,
    )
    assert (outside["selected"], outside["pie_active"], outside["warning"]) == (
        True,
        False,
        None,
    )
