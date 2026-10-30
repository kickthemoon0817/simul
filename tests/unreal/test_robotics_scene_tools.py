"""``analyze_unreal_scene_for_robotics`` and ``generate_unreal_procedural_scene``.

Both used to call ``/Script/UnrealEd.Default__EditorLevelLibrary``, which UE
5.7 does not have (``HTTP 400: Object ... does not exist``), and the
generator called a ``GenerateProceduralScene`` function no Unreal version
has. They now run editor Python like the rest of the adapter:

* the generator plans the layout here (``_unreal_procedural.plan_scene``) and
  one editor script (``PLACE_BASIC_SHAPES``) spawns the pieces in one
  transaction, removing what it spawned when a piece fails;
* the analysis is one editor script (``ROBOTICS_SCENE_ANALYSIS``) over the
  level's actors and the collision of their static meshes.

The scripts run here against an API-shaped fake ``unreal`` module with the
calls and return shapes observed on UE 5.7.4 (``get_actor_bounds`` returns
``(origin, extent)``, enums carry ``.name``, ``StaticMeshEditorSubsystem``
counts a cylinder's collision as one convex hull and no simple shape).
"""

from __future__ import annotations

import ast
import asyncio
import io
import json
import math
import sys
from contextlib import contextmanager, redirect_stdout
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pytest
from fastmcp import Client

from simul.adapters import _unreal_procedural as procedural
from simul.adapters import unreal_runtime
from simul.adapters._unreal_procedural import CUBE, CYLINDER, SPHERE, plan_scene
from simul.config import Settings
from simul.mcp import server as server_module
from simul.mcp.usage_tracker import ToolUsageTracker
from tests.fakes import AvailableAdapter

PYTHON_LIBRARY = "/Script/PythonScriptPlugin.Default__PythonScriptLibrary"
LEVEL = "/Game/Maps/Test.Test:PersistentLevel."

# ---------------------------------------------------------------------------
# A fake editor: actors, components and meshes with UE 5.7's Python shapes
# ---------------------------------------------------------------------------


def _vector(x: float, y: float, z: float) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=z)


class _Named:
    """An enum value or a class: Unreal's Python exposes ``.name`` / ``get_name()``."""

    def __init__(self, name: str) -> None:
        self.name = name

    def get_name(self) -> str:
        return self.name


class _Mesh:
    def __init__(self, path: str) -> None:
        self.path = path

    def get_path_name(self) -> str:
        return self.path


class PrimitiveComponent:
    def __init__(
        self,
        class_name: str = "PrimitiveComponent",
        collision: str = "QUERY_AND_PHYSICS",
        profile: str = "BlockAll",
        mesh: Optional[_Mesh] = None,
        mass: float = 0.0,
        simulate: bool = False,
    ) -> None:
        self.class_name = class_name
        self.collision = collision
        self.profile = profile
        self.mesh = mesh
        self.mass = mass
        self.simulate = simulate

    def get_collision_enabled(self) -> _Named:
        return _Named(self.collision)

    def get_collision_profile_name(self) -> str:
        return self.profile

    def get_class(self) -> _Named:
        return _Named(self.class_name)

    def get_mass(self) -> float:
        return self.mass

    def get_editor_property(self, name: str) -> Any:
        if name == "static_mesh":
            return self.mesh
        if name == "body_instance":
            return SimpleNamespace(get_editor_property={"simulate_physics": self.simulate}.__getitem__)
        raise AttributeError(name)


class StaticMeshComponent(PrimitiveComponent):
    def __init__(self, mesh: _Mesh, **kwargs: Any) -> None:
        super().__init__("StaticMeshComponent", mesh=mesh, **kwargs)


class FakeActor:
    def __init__(
        self,
        label: str,
        components: List[PrimitiveComponent],
        size: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        mobility: Optional[str] = "STATIC",
        tilt_degrees: float = 0.0,
        class_name: str = "StaticMeshActor",
    ) -> None:
        self.label = label
        self.components = components
        self.size = size
        self.center = center
        self.root_component = SimpleNamespace(mobility=_Named(mobility)) if mobility else None
        self.up_z = math.cos(math.radians(tilt_degrees))
        self.class_name = class_name

    def get_actor_label(self) -> str:
        return self.label

    def get_path_name(self) -> str:
        return LEVEL + self.label

    def get_class(self) -> _Named:
        return _Named(self.class_name)

    def get_components_by_class(self, cls: type) -> List[PrimitiveComponent]:
        return [component for component in self.components if isinstance(component, cls)]

    def get_actor_bounds(self, only_colliding: bool) -> Tuple[SimpleNamespace, SimpleNamespace]:
        assert only_colliding is True, "the analysis sizes actors by their colliding components"
        return _vector(*self.center), _vector(*(side / 2 for side in self.size))

    def get_actor_up_vector(self) -> SimpleNamespace:
        return _vector(0.0, math.sqrt(1 - self.up_z**2), self.up_z)

    def is_editable(self) -> bool:
        return True


# Mesh collision as StaticMeshEditorSubsystem reports it: (simple, convex, complexity).
MESHES: Dict[str, Tuple[int, int, str]] = {
    "/Engine/BasicShapes/Cube.Cube": (1, 0, "CTF_USE_DEFAULT"),
    "/Engine/BasicShapes/Cylinder.Cylinder": (0, 1, "CTF_USE_DEFAULT"),
    "/Game/Meshes/SM_Statue.SM_Statue": (0, 0, "CTF_USE_DEFAULT"),
    "/Game/Meshes/SM_Cliff.SM_Cliff": (0, 0, "CTF_USE_COMPLEX_AS_SIMPLE"),
}
CUBE_MESH = _Mesh("/Engine/BasicShapes/Cube.Cube")


def _scene() -> List[FakeActor]:
    """A level with one actor for each way the analysis can classify one."""
    return [
        FakeActor("Floor", [StaticMeshComponent(CUBE_MESH)], (1000, 1000, 10), (500, 500, 5)),
        FakeActor("Ramp", [StaticMeshComponent(CUBE_MESH)], (400, 200, 20), (0, 0, 50), tilt_degrees=30),
        FakeActor("Wall", [StaticMeshComponent(CUBE_MESH)], (1000, 20, 300), (500, 0, 150)),
        FakeActor(
            "Box",
            [StaticMeshComponent(CUBE_MESH, mass=12.5, simulate=True)],
            (40, 30, 20),
            (100, 100, 20),
            mobility="MOVABLE",
        ),
        FakeActor("Crate", [StaticMeshComponent(CUBE_MESH, mass=300.0)], (100, 100, 100), mobility="MOVABLE"),
        FakeActor("PinnedBox", [StaticMeshComponent(CUBE_MESH)], (40, 40, 40)),
        FakeActor(
            "Sensor",
            [StaticMeshComponent(CUBE_MESH, collision="QUERY_ONLY", profile="OverlapAll")],
            (20, 20, 20),
            mobility="MOVABLE",
        ),
        FakeActor(
            "Statue",
            [StaticMeshComponent(_Mesh("/Game/Meshes/SM_Statue.SM_Statue"))],
            (80, 80, 200),
        ),
        FakeActor(
            "Cliff",
            [StaticMeshComponent(_Mesh("/Game/Meshes/SM_Cliff.SM_Cliff"))],
            (3000, 2000, 1500),
        ),
        FakeActor(
            "LandscapeStreamingProxy_0",
            [
                PrimitiveComponent("LandscapeComponent", collision="NO_COLLISION", profile="NoCollision"),
                PrimitiveComponent("LandscapeHeightfieldCollisionComponent"),
            ],
            (25200, 25200, 200),
            (0, 0, 0),
            class_name="LandscapeStreamingProxy",
        ),
        FakeActor(
            "PointLight",
            [PrimitiveComponent("BillboardComponent", collision="NO_COLLISION", profile="NoCollision")],
            mobility="MOVABLE",
            class_name="PointLight",
        ),
        FakeActor("WorldSettings", [], mobility=None, class_name="WorldSettings"),
    ]


class FakeEditor:
    """The ``unreal`` module the scripts see, recording what they do to the level."""

    def __init__(self, actors: List[FakeActor], missing_meshes: Tuple[str, ...] = ()) -> None:
        self.actors = actors
        self.missing_meshes = missing_meshes
        self.spawned: List[Dict[str, Any]] = []
        self.destroyed: List[str] = []
        self.transactions: List[Dict[str, Any]] = []
        self.spawn_returns_none_after: Optional[int] = None
        # Spawn indices whose destroy_actor raises, or returns False.
        self.destroy_raises: Tuple[int, ...] = ()
        self.destroy_refuses: Tuple[int, ...] = ()
        # Spawn index whose set_folder_path raises.
        self.folder_raises: Optional[int] = None
        # LevelEditorSubsystem.is_in_play_in_editor; tests/unreal/test_pie.py covers PIE.
        self.pie = False

    def module(self) -> SimpleNamespace:
        editor = self

        class StaticMesh(_Mesh):
            pass

        class Transaction:
            def __init__(self, description: str) -> None:
                self.record = {"description": description, "cancelled": False, "open": False}
                editor.transactions.append(self.record)

            def __enter__(self) -> "Transaction":
                self.record["open"] = True
                return self

            def __exit__(self, *exc_info: Any) -> None:
                self.record["open"] = False

            def cancel(self) -> None:
                assert self.record["open"], "cancel inside the transaction"
                self.record["cancelled"] = True

        class SpawnedActor:
            def __init__(self, index: int, location: SimpleNamespace) -> None:
                self.index = index
                self.record: Dict[str, Any] = {
                    "path": f"{LEVEL}StaticMeshActor_{index}",
                    "location": [location.x, location.y, location.z],
                }
                record = self.record

                def set_mobility(mobility: str) -> None:
                    record["mobility"] = mobility

                def set_static_mesh(mesh: _Mesh) -> None:
                    record["mesh"] = mesh.get_path_name()

                self.static_mesh_component = SimpleNamespace(
                    set_mobility=set_mobility, set_static_mesh=set_static_mesh
                )

            def get_path_name(self) -> str:
                return self.record["path"]

            def set_actor_scale3d(self, scale: SimpleNamespace) -> None:
                self.record["scale"] = [scale.x, scale.y, scale.z]

            def set_actor_label(self, label: str) -> None:
                self.record["label"] = label

            def set_folder_path(self, folder: str) -> None:
                if self.index == editor.folder_raises:
                    raise RuntimeError("folder refused")
                self.record["folder"] = folder

            def set_editor_property(self, name: str, value: Any) -> None:
                self.record[name] = value

        def spawn_actor_from_class(cls: Any, location: SimpleNamespace, rotation: Any) -> Any:
            assert cls == "StaticMeshActor"
            assert rotation == (0.0, 0.0, 0.0)
            if editor.spawn_returns_none_after is not None and len(editor.spawned) >= editor.spawn_returns_none_after:
                return None
            actor = SpawnedActor(len(editor.spawned), location)
            editor.spawned.append(actor.record)
            return actor

        def destroy_actor(actor: Any) -> bool:
            if actor.index in editor.destroy_raises:
                raise RuntimeError("destroy refused")
            if actor.index in editor.destroy_refuses:
                return False
            editor.destroyed.append(actor.get_path_name())
            return True

        def load_asset(path: str) -> Optional[_Mesh]:
            return None if path in editor.missing_meshes else StaticMesh(path)

        def collision(index: int) -> Any:
            # During PIE, StaticMeshEditorSubsystem refuses: -1 and the default complexity.
            refused = (-1, -1, "CTF_USE_DEFAULT")[index]
            return lambda mesh: refused if editor.pie else MESHES[mesh.get_path_name()][index]

        def world(package: str) -> SimpleNamespace:
            return SimpleNamespace(
                get_outermost=lambda: SimpleNamespace(get_name=lambda: package),
                get_name=lambda: "Test",
                get_path_name=lambda: package + ".Test",
            )

        editor_world, pie_world = world("/Game/Maps/Test"), world("/Game/Maps/UEDPIE_0_Test")

        def all_actors_of_class(in_world: Any, cls: Any) -> List[FakeActor]:
            assert in_world is editor_world, "the editor world, not the PIE world"
            return list(editor.actors)

        subsystems = {
            # During PIE, EditorActorSubsystem lists nothing and the editor world is None.
            "EditorActorSubsystem": SimpleNamespace(
                get_all_level_actors=lambda: [] if editor.pie else list(editor.actors),
                spawn_actor_from_class=spawn_actor_from_class,
                destroy_actor=destroy_actor,
            ),
            "LevelEditorSubsystem": SimpleNamespace(is_in_play_in_editor=lambda: editor.pie),
            "UnrealEditorSubsystem": SimpleNamespace(
                get_editor_world=lambda: None if editor.pie else editor_world,
                get_game_world=lambda: pie_world if editor.pie else None,
            ),
            "StaticMeshEditorSubsystem": SimpleNamespace(
                get_simple_collision_count=collision(0),
                get_convex_collision_count=collision(1),
                get_collision_complexity=lambda mesh: _Named(collision(2)(mesh)),
            ),
        }
        return SimpleNamespace(
            EditorActorSubsystem="EditorActorSubsystem",
            LevelEditorSubsystem="LevelEditorSubsystem",
            UnrealEditorSubsystem="UnrealEditorSubsystem",
            StaticMeshEditorSubsystem="StaticMeshEditorSubsystem",
            GameplayStatics=SimpleNamespace(get_all_actors_of_class=all_actors_of_class),
            Actor=FakeActor,
            WorldSettings=type("WorldSettings", (FakeActor,), {}),
            find_object=lambda outer, path: editor_world if path == "/Game/Maps/Test.Test" else None,
            get_editor_subsystem=subsystems.__getitem__,
            PrimitiveComponent=PrimitiveComponent,
            StaticMeshComponent=StaticMeshComponent,
            StaticMesh=StaticMesh,
            StaticMeshActor="StaticMeshActor",
            ScopedEditorTransaction=Transaction,
            ComponentMobility=SimpleNamespace(MOVABLE="MOVABLE", STATIC="STATIC"),
            Vector=_vector,
            Rotator=lambda pitch, yaw, roll: (pitch, yaw, roll),
            Name=str,
            load_asset=load_asset,
        )


def _session(monkeypatch: pytest.MonkeyPatch, editor: FakeEditor) -> unreal_runtime.UnrealRuntimeSession:
    """A session whose editor Python runs here against ``editor``, as Remote Control would run it."""
    monkeypatch.setitem(sys.modules, "unreal", editor.module())
    session = unreal_runtime.UnrealRuntimeSession(Settings())
    session.scripts = []  # type: ignore[attr-defined]

    async def execute(code: str, mode: str = "ExecuteFile") -> Dict[str, Any]:
        session.scripts.append(code)  # type: ignore[attr-defined]
        out = io.StringIO()
        try:
            with redirect_stdout(out):
                exec(code, {})
        except Exception as exc:  # UE reports a raising script as ReturnValue False
            return {"ReturnValue": False, "CommandResult": repr(exc)}
        lines = out.getvalue().splitlines()
        return {"ReturnValue": True, "LogOutput": [{"Type": "Info", "Output": line} for line in lines]}

    monkeypatch.setattr(session, "_execute_python", execute)
    return session


def _script_args(script: str) -> Dict[str, Any]:
    """The ``args`` literal the adapter put in front of an editor script."""
    first = script.split("\n", 1)[0]
    assert first.startswith("args = ")
    return ast.literal_eval(first[len("args = "):])


# ---------------------------------------------------------------------------
# analyze_scene_for_robotics
# ---------------------------------------------------------------------------


def _analyze(monkeypatch: pytest.MonkeyPatch, editor: FakeEditor, **kwargs: Any) -> Dict[str, Any]:
    return asyncio.run(_session(monkeypatch, editor).analyze_scene_for_robotics(**kwargs))


def test_analysis_classifies_surfaces_graspable_objects_and_collision(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _analyze(monkeypatch, FakeEditor(_scene()))

    assert result["analysis_types"] == ["traversability", "graspability", "collision_complexity"]
    assert result["criteria"] == unreal_runtime.ROBOTICS_CRITERIA
    assert result["total_actors_analyzed"] == 12
    assert result["traversable_surfaces"] == [
        {
            "name": "Floor",
            "path": LEVEL + "Floor",
            "class_name": "StaticMeshActor",
            "size": [1000, 1000, 10],
            "top_z": 10,
            "area_m2": 100.0,
            "tilt_degrees": 0.0,
        },
        {
            "name": "LandscapeStreamingProxy_0",
            "path": LEVEL + "LandscapeStreamingProxy_0",
            "class_name": "LandscapeStreamingProxy",
            "size": [25200, 25200, 200],
            "top_z": 100,
            "area_m2": 63504.0,
            "tilt_degrees": 0.0,
        },
    ]
    assert result["graspable_objects"] == [
        {
            "name": "Box",
            "path": LEVEL + "Box",
            "class_name": "StaticMeshActor",
            "size": [40, 30, 20],
            "simulate_physics": True,
            "mass_kg": 12.5,
        }
    ]
    assert result["collision_summary"] == {
        "total_actors": 12,
        "with_collision": 10,
        "physics_collision": 9,
        "static_mesh_components": 9,
        "simple_shapes": 7,
        "convex_hulls": 0,
        "complex_collision": 1,
        "missing_simple_collision": 1,
        "heightfields": 1,
        "by_profile": {"BlockAll": 9, "OverlapAll": 1},
        "missing_simple_collision_actors": [LEVEL + "Statue"],
    }


def test_analysis_runs_only_the_requested_analyses(monkeypatch: pytest.MonkeyPatch) -> None:
    editor = FakeEditor(_scene())
    session = _session(monkeypatch, editor)

    result = asyncio.run(session.analyze_scene_for_robotics([" Graspability ", ""]))

    assert result["analysis_types"] == ["graspability"]
    assert [entry["name"] for entry in result["graspable_objects"]] == ["Box"]
    assert result["traversable_surfaces"] == []
    assert result["collision_summary"] == {}
    assert _script_args(session.scripts[0])["analyses"] == ["graspability"]  # type: ignore[attr-defined]


def test_actor_filter_matches_label_or_path_ignoring_case(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _analyze(monkeypatch, FakeEditor(_scene()), actor_filter="BOX")

    assert result["total_actors_analyzed"] == 2  # Box and PinnedBox
    assert [entry["name"] for entry in result["graspable_objects"]] == ["Box"]
    assert result["collision_summary"]["with_collision"] == 2

    by_path = _analyze(monkeypatch, FakeEditor(_scene()), actor_filter="persistentlevel.floor")
    assert [entry["name"] for entry in by_path["traversable_surfaces"]] == ["Floor"]


def test_collision_summary_lists_a_bounded_number_of_actors(monkeypatch: pytest.MonkeyPatch) -> None:
    statue = _Mesh("/Game/Meshes/SM_Statue.SM_Statue")
    actors = [FakeActor(f"Statue{n}", [StaticMeshComponent(statue)], (80, 80, 200)) for n in range(4)]
    monkeypatch.setattr(unreal_runtime, "ROBOTICS_LISTED_LIMIT", 2)

    summary = _analyze(monkeypatch, FakeEditor(actors))["collision_summary"]

    assert summary["missing_simple_collision"] == 4
    assert summary["missing_simple_collision_actors"] == [LEVEL + "Statue0", LEVEL + "Statue1"]


def test_mesh_collision_counts_are_null_while_pie_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """While PIE runs, the editor world is read directly, and StaticMeshEditorSubsystem
    answers -1 and the default complexity for every mesh (UE 5.7.4): those used to
    be summed as counts once the level could be read."""
    outside = _analyze(monkeypatch, FakeEditor(_scene()))
    editor = FakeEditor(_scene())
    editor.pie = True

    during = _analyze(monkeypatch, editor)

    assert (outside["pie_active"], during["pie_active"]) == (False, True)
    summary = during["collision_summary"]
    mesh_counts = ("simple_shapes", "convex_hulls", "complex_collision", "missing_simple_collision")
    assert [summary[key] for key in mesh_counts] == [None] * 4
    # The list those answers decide is unknown too, not empty: outside PIE the
    # same scene lists an actor missing simple collision.
    assert outside["collision_summary"]["missing_simple_collision_actors"] == [LEVEL + "Statue"]
    assert summary["missing_simple_collision_actors"] is None
    # Everything that does not come from the mesh asset is the same as outside PIE.
    for key in ("total_actors", "with_collision", "physics_collision", "static_mesh_components", "by_profile"):
        assert summary[key] == outside["collision_summary"][key], key
    assert during["traversable_surfaces"] == outside["traversable_surfaces"]
    assert during["graspable_objects"] == outside["graspable_objects"]


def test_an_unknown_analysis_type_is_refused_before_the_editor_is_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _session(monkeypatch, FakeEditor(_scene()))

    with pytest.raises(ValueError, match="analysis_types: unknown 'collision'; use traversability, graspability"):
        asyncio.run(session.analyze_scene_for_robotics(["traversability", "collision"]))

    assert session.scripts == []  # type: ignore[attr-defined]


def test_a_failed_analysis_script_is_a_script_error(monkeypatch: pytest.MonkeyPatch) -> None:
    editor = FakeEditor([FakeActor("Broken", [StaticMeshComponent(_Mesh("/Game/Unknown.Unknown"))], (1, 1, 1))])

    result = _analyze(monkeypatch, editor)

    assert result["success"] is False
    assert result["error_type"] == "ScriptError"
    assert "/Game/Unknown.Unknown" in result["error"]


# ---------------------------------------------------------------------------
# generate_procedural_scene: the editor script
# ---------------------------------------------------------------------------


#: The generation id the tests pin; production draws a fresh one per call.
GENERATION = "0badc0de"


def _generate(monkeypatch: pytest.MonkeyPatch, editor: FakeEditor, *args: Any, **kwargs: Any) -> Any:
    monkeypatch.setattr(unreal_runtime, "new_generation", lambda: GENERATION)
    session = _session(monkeypatch, editor)
    return session, asyncio.run(session.generate_procedural_scene(*args, **kwargs))


def test_generation_spawns_the_planned_pieces_in_one_transaction(monkeypatch: pytest.MonkeyPatch) -> None:
    editor = FakeEditor([])
    plan = plan_scene("room", {"seed": 7}, [0, 0, 0], [800, 600, 300])

    session, result = _generate(monkeypatch, editor, "room", {"seed": 7}, [0, 0, 0], [800, 600, 300])

    assert len(session.scripts) == 1
    assert result == {
        "actors_spawned": [record["path"] for record in editor.spawned],
        "total_spawned": len(plan.pieces),
        "scene_type": "room",
        "seed": 7,
        "tag": f"simul_room_7_{GENERATION}",
        "folder": f"Simul/Room_7_{GENERATION}",
    }
    assert editor.transactions == [{"description": "simul: generate room scene", "cancelled": False, "open": False}]
    assert editor.destroyed == []
    for piece, record in zip(plan.pieces, editor.spawned):
        spec = piece.spawn_args()
        assert record == {
            "path": record["path"],
            "location": spec["location"],
            "mesh": spec["mesh"],
            "mobility": "MOVABLE" if piece.movable else "STATIC",
            "scale": spec["scale"],
            "label": piece.label,
            "folder": f"Simul/Room_7_{GENERATION}",
            "tags": ["simul_procedural", f"simul_room_7_{GENERATION}"],
        }


@pytest.mark.parametrize("failure", ["missing mesh", "no actor"])
def test_a_failed_spawn_removes_what_was_spawned(monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    editor = FakeEditor([])
    if failure == "missing mesh":
        editor.missing_meshes = (CYLINDER,)  # the first trunk, after the ground
    else:
        editor.spawn_returns_none_after = 3

    _, result = _generate(monkeypatch, editor, "outdoor")

    assert result["success"] is False
    assert result["error_type"] == "SpawnError"
    assert result["error"].startswith("Spawned nothing: ")
    assert "folder" not in result["error"]
    assert editor.destroyed == [record["path"] for record in editor.spawned]
    assert editor.spawned, "the failure came after some pieces were spawned"
    assert editor.transactions[0]["cancelled"] is True
    # Unreal makes an outliner folder the first time an actor is filed in it
    # and keeps it, empty, after the actors go (observed on UE 5.7.4); actors
    # are filed only once every piece has spawned, so this failure made none.
    assert not any("folder" in record for record in editor.spawned)


def test_rollback_destroys_every_actor_and_cancels_when_a_destroy_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raising destroy_actor used to skip the rest of the rollback and the cancel."""
    editor = FakeEditor([])
    editor.spawn_returns_none_after = 4
    editor.destroy_raises = (1,)
    editor.destroy_refuses = (2,)

    _, result = _generate(monkeypatch, editor, "outdoor")

    paths = [record["path"] for record in editor.spawned]
    assert len(paths) == 4
    assert editor.destroyed == [paths[0], paths[3]], "the actors after the failing destroys were still removed"
    assert editor.transactions[0]["cancelled"] is True
    assert result["success"] is False
    assert result["error_type"] == "SpawnError"
    assert result["error"] == (
        "Unreal spawned no actor for Outdoor_Crown_1; rollback could not remove 2 of the 4 actors spawned: "
        f"{paths[1]}, {paths[2]}"
    )
    assert result["details"] == {"actors_left": [paths[1], paths[2]]}


def test_a_failure_while_filing_the_actors_is_rolled_back_and_names_the_folder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    editor = FakeEditor([])
    editor.folder_raises = 2

    _, result = _generate(monkeypatch, editor, "room", {"seed": 7, "props": 0}, [0, 0, 0], [800, 600, 300])

    assert editor.destroyed == [record["path"] for record in editor.spawned]
    assert editor.transactions[0]["cancelled"] is True
    assert result["error_type"] == "SpawnError"
    assert result["error"] == (
        f"Spawned nothing: folder refused; the outliner folder Simul/Room_7_{GENERATION} may remain empty, "
        "as Unreal's Python API cannot delete outliner folders"
    )


def test_generating_the_same_scene_twice_gives_each_call_its_own_tag_and_folder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both calls used to share simul_room_9 and Simul/Room_9, so a tag lookup mixed them."""
    editor = FakeEditor([])
    session = _session(monkeypatch, editor)

    first = asyncio.run(session.generate_procedural_scene("room", {"seed": 9, "props": 1}))
    second = asyncio.run(session.generate_procedural_scene("room", {"seed": 9, "props": 1}))

    assert first["tag"] != second["tag"] and first["folder"] != second["folder"]
    for result in (first, second):
        assert result["tag"].startswith("simul_room_9_")
        assert result["folder"].startswith("Simul/Room_9_")
        assert result["tag"].rsplit("_", 1)[1] == result["folder"].rsplit("_", 1)[1]
    spawned = editor.spawned
    half = len(spawned) // 2
    assert {record["tags"][1] for record in spawned[:half]} == {first["tag"]}
    assert {record["tags"][1] for record in spawned[half:]} == {second["tag"]}
    # The layout is the plan's: the same arguments place the same pieces.
    assert [record["location"] for record in spawned[:half]] == [record["location"] for record in spawned[half:]]


def test_the_editor_reporting_a_different_count_is_a_script_error(monkeypatch: pytest.MonkeyPatch) -> None:
    session = unreal_runtime.UnrealRuntimeSession(Settings())

    async def execute(code: str, mode: str = "ExecuteFile") -> Dict[str, Any]:
        return {"ReturnValue": True, "LogOutput": [{"Type": "Info", "Output": json.dumps({"actors_spawned": ["a"]})}]}

    monkeypatch.setattr(session, "_execute_python", execute)

    result = asyncio.run(session.generate_procedural_scene("corridor"))

    assert result["success"] is False
    assert result["error_type"] == "ScriptError"
    assert "6 planned actors" in result["error"]


def test_an_invalid_scene_is_refused_before_the_editor_is_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _session(monkeypatch, FakeEditor([]))

    with pytest.raises(ValueError, match="scene_type must be one of room, corridor, warehouse, outdoor"):
        asyncio.run(session.generate_procedural_scene("garage"))
    with pytest.raises(ValueError, match="room does not take 'density'"):
        asyncio.run(session.generate_procedural_scene("room", {"density": 2}))

    assert session.scripts == []  # type: ignore[attr-defined]


def test_the_script_receives_the_plan_as_a_literal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Labels and paths reach the editor as Python literals, never as source."""
    session, _ = _generate(monkeypatch, FakeEditor([]), "warehouse", {"rows": 1, "props": 2, "seed": 3})

    args = _script_args(session.scripts[0])
    plan = plan_scene("warehouse", {"rows": 1, "props": 2, "seed": 3})
    assert args == {
        "pieces": [piece.spawn_args() for piece in plan.pieces],
        "folder": f"Simul/Warehouse_3_{GENERATION}",
        "tags": ["simul_procedural", f"simul_warehouse_3_{GENERATION}"],
        "transaction": "simul: generate warehouse scene",
    }


# ---------------------------------------------------------------------------
# The layout planner
# ---------------------------------------------------------------------------


def _bounds_of(piece: procedural.Piece) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    low = tuple(c - s / 2 for c, s in zip(piece.center, piece.size))
    high = tuple(c + s / 2 for c, s in zip(piece.center, piece.size))
    return low, high  # type: ignore[return-value]


def _gap(a: procedural.Piece, b: procedural.Piece) -> float:
    """Distance in plan between two pieces' footprints; negative when they overlap."""
    (ax0, ay0, _), (ax1, ay1, _) = _bounds_of(a)
    (bx0, by0, _), (bx1, by1, _) = _bounds_of(b)
    return max(bx0 - ax1, ax0 - bx1, by0 - ay1, ay0 - by1)


def _kinds(plan: procedural.ScenePlan) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for piece in plan.pieces:
        kind = piece.label.split("_")[1]
        counts[kind] = counts.get(kind, 0) + 1
    return counts


@pytest.mark.parametrize(
    "scene_type, kinds",
    [
        # 960 x 960 cm inside the walls: 92 m^2 at one prop per 16 m^2.
        ("room", {"Floor": 1, "Wall": 5, "Box": 5}),
        # 10 m of corridor at one prop per 3 m.
        ("corridor", {"Floor": 1, "Wall": 2, "Box": 3}),
        # Two 460 x 100 cm shelves leave 83 m^2 of free floor.
        ("warehouse", {"Floor": 1, "Wall": 4, "Shelf": 2, "Box": 5}),
        # 100 m^2 of ground: one tree per 20 m^2, one rock per 10 m^2.
        ("outdoor", {"Ground": 1, "Trunk": 5, "Crown": 5, "Rock": 10}),
    ],
)
def test_default_scenes_fit_their_bounds(scene_type: str, kinds: Dict[str, int]) -> None:
    lo, hi = (-500.0, 200.0, 50.0), (500.0, 1200.0, 550.0)

    plan = plan_scene(scene_type, {}, lo, hi)

    assert plan.scene_type == scene_type and plan.seed == 42
    assert _kinds(plan) == kinds
    for piece in plan.pieces:
        low, high = _bounds_of(piece)
        assert all(value >= bound - 1e-6 for value, bound in zip(low, lo)), piece
        assert all(value <= bound + 1e-6 for value, bound in zip(high, hi)), piece
        assert piece.movable is (piece.label.split("_")[1] == "Box")
        assert piece.mesh in (CUBE, CYLINDER, SPHERE)
    floor = plan.pieces[0]
    assert _bounds_of(floor) == ((lo[0], lo[1], lo[2]), (hi[0], hi[1], lo[2] + procedural.FLOOR_THICKNESS))
    ground = lo[2] + procedural.FLOOR_THICKNESS
    placed = [piece for piece in plan.pieces if piece.label.split("_")[1] in ("Box", "Shelf", "Crown", "Rock")]
    for piece in placed:
        if piece.label.split("_")[1] != "Crown":
            assert _bounds_of(piece)[0][2] == pytest.approx(ground), piece
    for index, piece in enumerate(placed):
        for other in placed[index + 1:] + [p for p in plan.pieces if "_Wall_" in p.label]:
            assert _gap(piece, other) >= procedural.CLEARANCE - 1e-6, (piece.label, other.label)


def test_the_same_arguments_plan_the_same_scene() -> None:
    first = plan_scene("warehouse", {"seed": 11}, [0, 0, 0], [2000, 1500, 600])

    assert plan_scene("warehouse", {"seed": 11}, [0, 0, 0], [2000, 1500, 600]) == first
    assert plan_scene("Warehouse ", {"seed": 12}, [0, 0, 0], [2000, 1500, 600]).pieces != first.pieces


@pytest.mark.parametrize("seed", [-1, -9, procedural.MAX_SEED + 1, 10**40])
def test_a_seed_outside_0_to_max_is_refused(seed: int) -> None:
    """random.Random seeds with the absolute value, so -9 used to plan seed 9's
    layout under the name simul_room_-9."""
    with pytest.raises(ValueError, match=rf"seed must be an integer from 0 to 4294967295, got {seed}$"):
        plan_scene("room", {"seed": seed})


def test_every_seed_from_0_to_max_plans_its_own_layout() -> None:
    plans = [plan_scene("room", {"seed": seed, "props": 3}) for seed in (0, 1, 9, procedural.MAX_SEED)]

    assert [plan.seed for plan in plans] == [0, 1, 9, procedural.MAX_SEED]
    assert len({plan.pieces for plan in plans}) == len(plans)


def test_each_generation_names_its_own_tag_and_folder() -> None:
    plan = plan_scene("outdoor", {"seed": 5})
    first, second = procedural.new_generation(), procedural.new_generation()

    assert first != second
    for generation in (first, second):
        assert len(generation) == 8 and int(generation, 16) >= 0 and generation == generation.lower()
    assert plan.tag(first) == f"simul_outdoor_5_{first}"
    assert plan.folder(first) == f"Simul/Outdoor_5_{first}"
    assert plan.tag(first) != plan.tag(second) and plan.folder(first) != plan.folder(second)


def test_spawn_args_scale_the_basic_shapes() -> None:
    floor = plan_scene("room", {"props": 0}, [0, 0, 0], [1000, 400, 300]).pieces[0]

    assert floor.spawn_args() == {
        "label": "Room_Floor_0",
        "mesh": CUBE,
        "location": [500.0, 200.0, 5.0],
        "scale": [10.0, 4.0, 0.1],
        "movable": False,
    }


def test_a_room_doorway_is_centred_and_kept_clear() -> None:
    plan = plan_scene("room", {"door_width": 120, "props": 30}, [0, 0, 0], [800, 800, 300])
    south = [_bounds_of(p) for p in plan.pieces if "_Wall_" in p.label and _bounds_of(p)[0][1] == 0]

    assert [(low[0], high[0]) for low, high in south] == [(0, 340), (460, 800)]
    for box in (p for p in plan.pieces if "_Box_" in p.label):
        (x0, y0, _), (x1, _, _) = _bounds_of(box)
        assert x1 <= 340 or x0 >= 460 or y0 >= 20 + procedural.DOORWAY_DEPTH, box.label

    closed = plan_scene("room", {"door_width": 0, "props": 0}, [0, 0, 0], [800, 800, 300])
    assert _kinds(closed) == {"Floor": 1, "Wall": 4}


@pytest.mark.parametrize("bounds_max", [[3000, 600, 300], [600, 3000, 300]])
def test_corridor_props_leave_a_passage(bounds_max: List[float]) -> None:
    along = 0 if bounds_max[0] > bounds_max[1] else 1
    across = 1 - along
    plan = plan_scene("corridor", {"props": 12, "seed": 5}, [0, 0, 0], bounds_max)
    walls = [_bounds_of(p) for p in plan.pieces if "_Wall_" in p.label]
    inner = (walls[0][1][across], walls[1][0][across])
    boxes = [_bounds_of(p) for p in plan.pieces if "_Box_" in p.label]

    assert len(boxes) == 12
    clearance = procedural.CLEARANCE
    for low, high in boxes:
        beside_first = low[across] == pytest.approx(inner[0] + clearance)
        assert beside_first or high[across] == pytest.approx(inner[1] - clearance)
        assert (inner[1] - inner[0]) - clearance - (high[across] - low[across]) >= procedural.PASSAGE - 1e-6
    spans = sorted((low[along], high[along]) for low, high in boxes)
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert start - end >= procedural.CLEARANCE - 1e-6


def test_warehouse_shelves_run_along_the_longer_side_with_aisles() -> None:
    plan = plan_scene("warehouse", {"aisle_width": 200, "shelf_depth": 80, "props": 0}, [0, 0, 0], [1200, 2400, 600])
    shelves = [_bounds_of(p) for p in plan.pieces if "_Shelf_" in p.label]

    # 1160 cm across inside the walls: (1160 - 200) // (80 + 200) rows.
    assert len(shelves) == 3
    for low, high in shelves:
        assert high[1] - low[1] > high[0] - low[0], "along y, the longer side"
        assert (low[1], high[1]) == (20 + 200, 2380 - 200)
        assert high[2] - low[2] == 300
    xs = sorted((low[0], high[0]) for low, high in shelves)
    assert xs[0][0] - 20 >= 200 and 1180 - xs[-1][1] >= 200
    assert all(b[0] - a[1] == pytest.approx(200) for a, b in zip(xs, xs[1:]))


def test_trees_stand_on_the_ground_under_their_crowns() -> None:
    plan = plan_scene("outdoor", {"trees": 3, "rocks": 0}, [0, 0, 0], [2000, 2000, 400])
    trunks = [p for p in plan.pieces if "_Trunk_" in p.label]
    crowns = [p for p in plan.pieces if "_Crown_" in p.label]

    for trunk, crown in zip(trunks, crowns):
        assert trunk.center[:2] == crown.center[:2]
        assert _bounds_of(trunk)[0][2] == pytest.approx(procedural.FLOOR_THICKNESS)
        assert _bounds_of(trunk)[1][2] > _bounds_of(crown)[0][2], "the trunk reaches into the crown"
        assert _bounds_of(crown)[1][2] <= 400 + 1e-6


@pytest.mark.parametrize(
    "scene_type, parameters, bounds, message",
    [
        ("room", {"seed": True}, None, "seed must be an integer"),
        ("room", {"seed": 1.5}, None, "seed must be an integer"),
        ("room", {"props": -1}, None, "props must be an integer from 0 to 200"),
        ("room", {"props": 201}, None, "props must be an integer from 0 to 200"),
        ("room", {"wall_thickness": 0}, None, "wall_thickness must be a number of centimetres above 0"),
        ("room", {"wall_thickness": float("nan")}, None, "wall_thickness must be a number"),
        ("room", {"wall_thickness": 480}, None, "leaves less than 50 cm inside the walls"),
        ("room", {"wall_height": 600}, None, "wall_height 600 cm exceeds the 490 cm"),
        ("room", {"door_width": 961}, None, "door_width 961 cm does not fit in the 960 cm"),
        ("room", {"door_width": 100}, ([0, 0, 0], [120, 400, 300]), "door_width 100 cm does not fit in the 80 cm"),
        ("room", {"door_width": "wide"}, None, "door_width must be a number"),
        ("room", {"props": 200}, ([0, 0, 0], [300, 300, 300]), "props: only"),
        (
            "room",
            {"props": 1},
            ([0, 0, 0], [3000, 100, 300]),
            "props: the room is 60 cm across inside its walls; props need 70 cm",
        ),
        (
            "warehouse",
            {"rows": 0, "props": 1},
            ([0, 0, 0], [3000, 100, 300]),
            "props: the warehouse is 60 cm across inside its walls; props need 70 cm",
        ),
        ("corridor", {"props": 1}, ([0, 0, 0], [2000, 140, 300]), "100 cm wide inside its walls; props need 110"),
        ("warehouse", {"rows": 3}, None, "rows: 3 rows do not fit; 2 rows"),
        ("warehouse", {}, ([0, 0, 0], [400, 400, 300]), "pass rows 0"),
        ("warehouse", {"rows": 1, "shelf_depth": 10}, ([0, 0, 0], [560, 560, 300]), "Shelves need 600 cm along"),
        ("warehouse", {"aisle_width": 80, "props": 1}, None, "80 cm aisles leave no room for props"),
        ("warehouse", {"shelf_height": 900}, None, "shelf_height 900 cm exceeds"),
        ("outdoor", {"trees": 200, "rocks": 200}, None, "would have 601 actors; at most 500"),
        ("outdoor", {"trees": 1}, ([0, 0, 0], [1000, 1000, 90]), "trees: need 100 cm"),
        ("outdoor", {"rows": 1}, None, "outdoor does not take 'rows'; its parameters are seed, trees, rocks"),
        ("room", None, ([0, 0], [1, 1, 1]), "bounds_min must be three finite numbers"),
        ("room", None, ([0, 0, 0], [0, 1000, 500]), "bounds_max must exceed bounds_min by at least 100 cm"),
        ("room", None, ([0, 0, 0], [1000, 1000, 50]), "and 60 cm in z"),
    ],
)
def test_invalid_layouts_are_refused(
    scene_type: str, parameters: Any, bounds: Any, message: str
) -> None:
    lo, hi = bounds if bounds else (None, None)

    with pytest.raises(ValueError, match=message.replace("(", r"\(")):
        plan_scene(scene_type, parameters, lo, hi)


def test_parameters_must_be_an_object() -> None:
    with pytest.raises(ValueError, match="parameters must be a JSON object, got list"):
        plan_scene("room", [1, 2])  # type: ignore[arg-type]


def test_rows_zero_builds_an_empty_hall() -> None:
    plan = plan_scene("warehouse", {"rows": 0, "props": 0}, [0, 0, 0], [400, 400, 300])

    assert _kinds(plan) == {"Floor": 1, "Wall": 4}


def test_narrow_warehouse_aisles_get_smaller_props_or_none() -> None:
    narrow = plan_scene("warehouse", {"aisle_width": 100, "shelf_depth": 60}, [0, 0, 0], [1500, 1500, 400])
    boxes = [p for p in narrow.pieces if "_Box_" in p.label]

    assert boxes and all(max(p.size[:2]) <= 100 - 3 * procedural.CLEARANCE for p in boxes)
    assert "Box" not in _kinds(plan_scene("warehouse", {"aisle_width": 80}))


@pytest.mark.parametrize("bounds_max", [[300, 300, 300], [5000, 5000, 1000], [12000, 8000, 800]])
@pytest.mark.parametrize("scene_type", ["room", "corridor", "outdoor"])
def test_default_counts_fit_small_and_large_bounds(scene_type: str, bounds_max: List[float]) -> None:
    for seed in range(20):
        plan = plan_scene(scene_type, {"seed": seed}, [0, 0, 0], bounds_max)
        assert len(plan.pieces) <= procedural.MAX_PIECES


@pytest.mark.parametrize(
    "scene_type, parameters",
    [("room", {}), ("corridor", {}), ("warehouse", {"rows": 0}), ("outdoor", {})],
)
@pytest.mark.parametrize(
    "bounds_max", [[120, 400, 300], [3000, 110, 300], [110, 3000, 300], [3000, 100, 500], [100, 3000, 500]]
)
def test_default_scenes_fit_narrow_bounds(
    scene_type: str, parameters: Dict[str, Any], bounds_max: List[float]
) -> None:
    """Defaults used to fail here: a 100 cm doorway in a narrower room, props or crowns too wide for the floor."""
    for seed in range(10):
        plan = plan_scene(scene_type, {**parameters, "seed": seed}, [0, 0, 0], bounds_max)
        for piece in plan.pieces:
            low, high = _bounds_of(piece)
            assert min(low) >= -1e-6, piece
            assert all(value <= bound + 1e-6 for value, bound in zip(high, bounds_max)), piece


def test_a_narrow_room_narrows_its_doorway_and_its_props() -> None:
    closet = plan_scene("room", {"props": 0}, [0, 0, 0], [120, 400, 300])
    south = [_bounds_of(p) for p in closet.pieces if "_Wall_" in p.label and _bounds_of(p)[0][1] == 0]
    # The default doorway takes the 80 cm between the side walls.
    assert [(low[0], high[0]) for low, high in south] == [(0, 20), (100, 120)]

    hall = plan_scene("room", {}, [0, 0, 0], [3000, 110, 300])
    boxes = [p for p in hall.pieces if "_Box_" in p.label]
    # 70 cm inside the walls, less 20 cm of clearance on each side.
    assert boxes and all(max(p.size[:2]) <= 30 + 1e-6 for p in boxes)
    assert "Box" not in _kinds(plan_scene("room", {}, [0, 0, 0], [3000, 100, 300]))


def test_trees_shrink_to_fit_a_narrow_strip() -> None:
    plan = plan_scene("outdoor", {"trees": 5, "rocks": 0}, [0, 0, 0], [4000, 100, 500])
    crowns = [p for p in plan.pieces if "_Crown_" in p.label]

    assert len(crowns) == 5
    assert all(max(p.size[:2]) <= 100 + 1e-6 for p in crowns)


# ---------------------------------------------------------------------------
# Through the MCP envelope: the response models keep what the tools return
# ---------------------------------------------------------------------------


class _ScriptAdapter(AvailableAdapter):
    """Hands the server a session whose editor Python runs against the fake editor."""

    def __init__(self, session: unreal_runtime.UnrealRuntimeSession) -> None:
        super().__init__(session.settings)
        self.session = session

    @contextmanager
    def create_session(self) -> Iterator[unreal_runtime.UnrealRuntimeSession]:
        yield self.session


def _call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, editor: FakeEditor, tool: str, arguments: Dict[str, Any]
) -> Dict[str, Any]:
    settings = Settings(
        unreal={"tool_surface": "full", "attachment_path": str(tmp_path / "attachment.json")},
        security={"rate_limiting_enabled": False},
    )
    instance = server_module.SimulMCPServer(settings, backends={"unreal"})
    instance.usage_tracker = ToolUsageTracker(log_dir=tmp_path / "usage")
    instance.unreal_adapter = _ScriptAdapter(_session(monkeypatch, editor))

    async def run() -> Any:
        async with Client(instance.mcp) as client:
            return await client.call_tool(tool, arguments, raise_on_error=False)

    result = asyncio.run(run())
    return json.loads([block.text for block in result.content if getattr(block, "type", None) == "text"][-1])


def test_the_analysis_tool_returns_criteria_and_analyses(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    payload = _call(
        monkeypatch,
        tmp_path,
        FakeEditor(_scene()),
        "analyze_unreal_scene_for_robotics",
        {"analysis_types": "traversability, graspability", "actor_filter": ""},
    )

    assert payload["success"] is True
    assert payload["analysis_types"] == ["traversability", "graspability"]
    assert payload["criteria"] == unreal_runtime.ROBOTICS_CRITERIA
    assert [entry["name"] for entry in payload["traversable_surfaces"]] == ["Floor", "LandscapeStreamingProxy_0"]
    assert payload["collision_summary"] == {}


def test_the_generator_tool_returns_its_tag_and_folder(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    editor = FakeEditor([])
    monkeypatch.setattr(unreal_runtime, "new_generation", lambda: GENERATION)

    payload = _call(
        monkeypatch,
        tmp_path,
        editor,
        "generate_unreal_procedural_scene",
        {"scene_type": "corridor", "parameters": '{"seed": 9, "props": 1}', "bounds_max": "2000, 400, 300"},
    )

    assert payload == {
        "success": True,
        "error": None,
        "actors_spawned": [record["path"] for record in editor.spawned],
        "total_spawned": 4,
        "scene_type": "corridor",
        "seed": 9,
        "tag": f"simul_corridor_9_{GENERATION}",
        "folder": f"Simul/Corridor_9_{GENERATION}",
    }


@pytest.mark.parametrize(
    "arguments, error_type, message",
    [
        ({"scene_type": "garage"}, "ValueError", "scene_type must be one of"),
        ({"scene_type": "room", "parameters": "{bad"}, "JSONDecodeError", "Expecting property name"),
        ({"scene_type": "room", "bounds_min": "0,0"}, "ValueError", "bounds_min must be three finite numbers"),
        ({"scene_type": "room", "bounds_min": "0,0,nan"}, "ValueError", "bounds_min must be three finite numbers"),
    ],
)
def test_the_generator_tool_reports_invalid_input(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, arguments: Dict[str, Any], error_type: str, message: str
) -> None:
    editor = FakeEditor([])

    payload = _call(monkeypatch, tmp_path, editor, "generate_unreal_procedural_scene", arguments)

    assert payload["success"] is False
    assert payload["error_type"] == error_type
    assert message in payload["error"]
    assert editor.spawned == []


def test_the_tools_only_ask_the_python_library(monkeypatch: pytest.MonkeyPatch) -> None:
    """No request goes to an object UE 5.7 does not have, such as EditorLevelLibrary."""
    sent: List[Dict[str, Any]] = []
    session = unreal_runtime.UnrealRuntimeSession(Settings())

    async def http_put(path: str, body: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        sent.append({"path": path, **body})
        return {"ReturnValue": True, "LogOutput": [{"Type": "Info", "Output": json.dumps(reply)}]}

    monkeypatch.setattr(session, "_http_put", http_put)
    reply: Dict[str, Any] = {"traversable_surfaces": [], "graspable_objects": [], "collision_summary": {},
                             "total_actors_analyzed": 0}
    asyncio.run(session.analyze_scene_for_robotics())
    reply = {"actors_spawned": [f"a{n}" for n in range(6)]}
    asyncio.run(session.generate_procedural_scene("corridor"))

    assert [(entry["path"], entry["objectPath"], entry["functionName"]) for entry in sent] == [
        ("/remote/object/call", PYTHON_LIBRARY, "ExecutePythonCommandEx")
    ] * 2
