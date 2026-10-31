"""focus_unreal_on_actor with distance 0 fits the actor in the view.

Distance 0 was documented as auto-fit, but the camera always stood a fixed
500 cm behind the actor's location (UE 5.7.4): a 20 m cube put the camera
inside the cube, and a 10 cm one was a speck. Now distance 0 reads the actor's
bounds and backs the camera off until their bounding sphere fills the view's
height with a margin, looking at their center. Editor Python cannot read the
level viewport's field of view, so UE's default is assumed: 90 degrees across
on a 16:9 viewport. An actor without bounds (zero extent) keeps the 500 cm and
the reply says so; an explicit distance places the camera as before.

The adapter tests run the real editor script against a stand-in for the
``unreal`` module and the camera calls against a stand-in level viewport. The
fit is checked geometrically: the actor's bounding box is projected through the
camera read back.
"""

from __future__ import annotations

import asyncio
import io
import json
import math
import sys
from contextlib import contextmanager, redirect_stdout
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Sequence, Tuple

import pytest
from fastmcp import Client

from simul.adapters import unreal_runtime
from simul.config import Settings
from simul.mcp import server as server_module
from simul.mcp.usage_tracker import ToolUsageTracker
from tests.fakes import AvailableAdapter

LEVEL = "/Game/Maps/Lab.Lab:PersistentLevel."

# UE's default level viewport, which the fit assumes: 90 degrees across on a
# 16:9 viewport, so the view reaches tan(45 degrees) across and 9/16 of that up
# per unit of depth. Unreal clips anything nearer than 10 cm.
HALF_WIDTH = math.tan(math.radians(45.0))
HALF_HEIGHT = HALF_WIDTH * 9.0 / 16.0
NEAR_CLIP = 10.0


# ---------------------------------------------------------------------------
# A stand-in for the parts of ``unreal`` the focus script uses
# ---------------------------------------------------------------------------


class _Vector:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.x, self.y, self.z = x, y, z


class PrimitiveComponent:
    """A primitive component and its world bounds (``Bounds``) as a center and a half-size."""

    def __init__(self, center: Sequence[float], extent: Sequence[float]) -> None:
        self.center, self.extent = tuple(center), tuple(extent)


class DrawFrustumComponent(PrimitiveComponent):
    """The frustum the editor draws for a camera: a cube reaching its far end (1000 cm by
    default) on every side of the camera (UDrawFrustumComponent::CalcBounds, UE 5.7)."""

    def __init__(self, camera: Sequence[float], far: float = 1000.0) -> None:
        super().__init__(camera, (far, far, far))


def get_component_bounds(
    component: PrimitiveComponent,
) -> Tuple[_Vector, _Vector, float]:
    """SystemLibrary.get_component_bounds: (origin, box_extent, sphere_radius)."""
    return (
        _Vector(*component.center),
        _Vector(*component.extent),
        math.hypot(*component.extent),
    )


def _union(
    components: Sequence[PrimitiveComponent],
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """The center and half-size of the box around the components' bounds."""
    low = [min(c.center[i] - c.extent[i] for c in components) for i in range(3)]
    high = [max(c.center[i] + c.extent[i] for c in components) for i in range(3)]
    return (
        tuple((lo + hi) / 2 for lo, hi in zip(low, high)),
        tuple((hi - lo) / 2 for lo, hi in zip(low, high)),
    )


class Actor:
    """An actor with a location and the bounds ``get_actor_bounds`` reports.

    ``components`` are its primitive components and ``children`` the actors its
    child actor components spawned; ``get_actor_bounds`` reports ``origin`` and
    ``extent`` whatever they are, as given.
    """

    def __init__(
        self,
        name: str,
        location: Sequence[float],
        origin: Sequence[float],
        extent: Sequence[float],
        components: Sequence[PrimitiveComponent] = (),
        children: Sequence["Actor"] = (),
    ) -> None:
        self.name = name
        self.location = tuple(location)
        self.origin, self.extent = tuple(origin), tuple(extent)
        self.components, self.children = list(components), list(children)
        self.bounds_calls: List[Tuple[bool, bool]] = []

    def get_path_name(self) -> str:
        return LEVEL + self.name

    def get_actor_location(self) -> _Vector:
        return _Vector(*self.location)

    def get_actor_rotation(self) -> SimpleNamespace:
        return SimpleNamespace(pitch=0.0, yaw=0.0, roll=0.0)

    def get_actor_scale3d(self) -> _Vector:
        return _Vector(1.0, 1.0, 1.0)

    def get_actor_bounds(
        self, only_colliding_components: bool, include_from_child_actors: bool = False
    ) -> Tuple[_Vector, _Vector]:
        """AActor::GetActorBounds: the box around the primitive components."""
        self.bounds_calls.append((only_colliding_components, include_from_child_actors))
        return _Vector(*self.origin), _Vector(*self.extent)

    def get_components_by_class(self, component_class: type) -> List[Any]:
        return [c for c in self.components if isinstance(c, component_class)]

    def get_all_child_actors(self, include_descendants: bool = True) -> List["Actor"]:
        """The actors spawned by its child actor components, children of children included."""
        found: List[Actor] = []
        for child in self.children:
            found.append(child)
            if include_descendants:
                found.extend(child.get_all_child_actors())
        return found


class Editor:
    """The level, the editor selection and the level viewport camera."""

    def __init__(self) -> None:
        self.pie = False
        self.actors: Dict[str, Actor] = {}
        self.selection: List[Actor] = []
        self.camera: Dict[str, Any] = {
            "CameraLocation": {"X": 0.0, "Y": 0.0, "Z": 1000.0},
            "CameraRotation": {"Pitch": 0.0, "Yaw": 0.0, "Roll": 0.0},
        }
        self.camera_writes: List[Dict[str, Any]] = []

    def add(self, actor: Actor) -> str:
        self.actors[actor.get_path_name()] = actor
        return actor.get_path_name()

    # EditorActorSubsystem
    def set_actor_selection_state(self, actor: Actor, should_be_selected: bool) -> None:
        if should_be_selected and actor not in self.selection:
            self.selection.append(actor)

    def get_selected_level_actors(self) -> List[Actor]:
        return [] if self.pie else list(self.selection)

    def module(self) -> SimpleNamespace:
        editor = self
        subsystems = {
            "EditorActorSubsystem": editor,
            "LevelEditorSubsystem": SimpleNamespace(
                is_in_play_in_editor=lambda: editor.pie
            ),
        }
        return SimpleNamespace(
            Actor=Actor,
            PrimitiveComponent=PrimitiveComponent,
            DrawFrustumComponent=DrawFrustumComponent,
            SystemLibrary=SimpleNamespace(get_component_bounds=get_component_bounds),
            EditorActorSubsystem="EditorActorSubsystem",
            LevelEditorSubsystem="LevelEditorSubsystem",
            get_editor_subsystem=subsystems.__getitem__,
            load_object=lambda outer, path: editor.actors.get(path),
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

    # UnrealEditorSubsystem's level viewport camera over Remote Control.
    def call_function(self, function_name: str, parameters: Any) -> Dict[str, Any]:
        if function_name == "SetLevelViewportCameraInfo":
            self.camera_writes.append(parameters)
            self.camera = dict(parameters)
            return {}
        if function_name == "GetLevelViewportCameraInfo":
            return {**self.camera, "ReturnValue": True}
        raise AssertionError("Unexpected Remote Control call: " + function_name)


@pytest.fixture
def editor(monkeypatch: pytest.MonkeyPatch) -> Editor:
    level = Editor()
    monkeypatch.setitem(sys.modules, "unreal", level.module())
    return level


def _session(
    editor: Editor, monkeypatch: pytest.MonkeyPatch, settings: Any = None
) -> Any:
    session = unreal_runtime.UnrealRuntimeSession(settings or Settings())

    async def execute(code: str, mode: str = "ExecuteFile") -> Dict[str, Any]:
        return editor.run(code)

    async def call_function(
        object_path: str, function_name: str, parameters: Any = None
    ) -> Dict[str, Any]:
        return editor.call_function(function_name, parameters)

    monkeypatch.setattr(session, "_execute_python", execute)
    monkeypatch.setattr(session, "_call_function", call_function)
    return session


def _focus(
    editor: Editor, monkeypatch: pytest.MonkeyPatch, path: str, **kwargs: Any
) -> Dict[str, Any]:
    return asyncio.run(_session(editor, monkeypatch).focus_on_actor(path, **kwargs))


# ---------------------------------------------------------------------------
# What the camera read back sees
# ---------------------------------------------------------------------------


def _in_view(
    camera: Sequence[float], rotation: Sequence[float], point: Sequence[float]
) -> Tuple[float, float, float]:
    """Where ``point`` is in the view of a camera at ``camera`` with (pitch, yaw, roll) ``rotation``.

    Returns the depth along the view axis, and how far across and up the
    point is, as fractions of the view's half-width and half-height at that
    depth: the point is in the view when both are within [-1, 1].
    """
    pitch, yaw = math.radians(rotation[0]), math.radians(rotation[1])
    assert rotation[2] == 0.0, "no roll"
    forward = (
        math.cos(pitch) * math.cos(yaw),
        math.cos(pitch) * math.sin(yaw),
        math.sin(pitch),
    )
    right = (-math.sin(yaw), math.cos(yaw), 0.0)
    up = (
        -math.sin(pitch) * math.cos(yaw),
        -math.sin(pitch) * math.sin(yaw),
        math.cos(pitch),
    )
    offset = [p - c for p, c in zip(point, camera)]

    def along(axis: Sequence[float]) -> float:
        return sum(o * a for o, a in zip(offset, axis))

    depth = along(forward)
    return depth, along(right) / depth / HALF_WIDTH, along(up) / depth / HALF_HEIGHT


def _box_in_view(
    result: Dict[str, Any], origin: Sequence[float], extent: Sequence[float]
) -> Tuple[float, float]:
    """The nearest depth of the box's corners, and the largest share of the view they reach.

    Fails when a corner is behind the camera, where it has no place in the view.
    """
    reach, nearest = 0.0, math.inf
    for signs in [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]:
        corner = [o + s * e for o, s, e in zip(origin, signs, extent)]
        depth, across, up = _in_view(
            result["camera_location"], result["camera_rotation"], corner
        )
        assert depth > 0.0, f"corner {corner} is behind the camera"
        nearest, reach = min(nearest, depth), max(reach, abs(across), abs(up))
    return nearest, reach


def _placement(
    target: Sequence[float], distance: float
) -> Tuple[Tuple[float, ...], Tuple[float, float, float]]:
    """Where focus_on_actor puts the camera to look at ``target``: ``distance``
    behind it toward -X, half that above it, pitched down at it."""
    return (
        (target[0] - distance, target[1], target[2] + distance / 2),
        (-math.degrees(math.atan2(distance / 2, distance)), 0.0, 0.0),
    )


# ---------------------------------------------------------------------------
# Distance 0 fits the actor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "location, origin, extent",
    [
        # A 20 m cube standing on its location: 500 cm put the camera inside it.
        ((0.0, 0.0, 0.0), (0.0, 0.0, 1000.0), (1000.0, 1000.0, 1000.0)),
        # A 30 cm box: 500 cm made it a speck.
        ((300.0, -200.0, 0.0), (300.0, -200.0, 15.0), (15.0, 15.0, 15.0)),
        # A 40 m bridge across the view: 500 cm cut off both ends.
        ((50.0, 60.0, 70.0), (50.0, 60.0, 120.0), (100.0, 2000.0, 50.0)),
    ],
    ids=["20 m cube", "30 cm box", "40 m bridge"],
)
def test_distance_0_fits_the_actor_in_the_view(
    editor: Editor,
    monkeypatch: pytest.MonkeyPatch,
    location: Tuple[float, float, float],
    origin: Tuple[float, float, float],
    extent: Tuple[float, float, float],
) -> None:
    actor = Actor("Crate", location, origin, extent)
    path = editor.add(actor)

    result = _focus(editor, monkeypatch, path)

    nearest, reach = _box_in_view(result, origin, extent)
    # The whole box is in the view, beyond the near clip plane, and fills a
    # good part of it (less for a long box, whose bounding sphere is fitted).
    assert nearest > NEAR_CLIP
    assert 0.4 < reach <= 1.0
    # The camera looks at the center of the bounds, not at the location.
    _, across, up = _in_view(
        result["camera_location"], result["camera_rotation"], origin
    )
    assert (across, up) == pytest.approx((0.0, 0.0), abs=1e-9)
    # The bounds of every primitive component, child actors' included.
    assert actor.bounds_calls == [(False, True)]
    # The reply reports the distance the camera stands at.
    camera, rotation = _placement(origin, result["distance"])
    assert result["camera_location"] == pytest.approx(camera)
    assert result["camera_rotation"] == pytest.approx(rotation)
    assert "fit_warning" not in result
    assert (result["selected"], result["pie_active"]) == (True, False)


def test_an_actor_smaller_than_the_editor_focus_minimum_stays_clear_of_the_near_clip_plane(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Like the editor's own Focus, a bounding sphere under 10 cm in radius is framed
    as a 10 cm one, so the camera does not move so close that Unreal clips the actor."""
    results = []
    for name, size in (("Pebble", 2.0), ("Grain", 0.2)):
        extent = (size / 2,) * 3
        path = editor.add(Actor(name, (0.0, 0.0, 0.0), (0.0, 0.0, size / 2), extent))
        result = _focus(editor, monkeypatch, path)
        nearest, reach = _box_in_view(result, (0.0, 0.0, size / 2), extent)
        assert nearest > NEAR_CLIP and reach <= 1.0
        results.append(result)

    assert results[0]["distance"] == results[1]["distance"]


def test_distance_0_frames_a_camera_by_its_mesh_not_by_its_frustum(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In the editor every camera (and 2D scene capture) owns a DrawFrustumComponent whose
    bounds reach 10 m on every side of it, and get_actor_bounds includes them: the camera
    stood 35 m away and its mesh covered 2% of the view. The editor's own Focus leaves the
    frustum out, and so does the fit: the camera is framed by its mesh."""
    camera = (400.0, -300.0, 200.0)
    mesh = PrimitiveComponent(camera, (40.0, 15.0, 25.0))
    frustum = DrawFrustumComponent(camera)
    actor = Actor(
        "CameraActor_0", camera, *_union([mesh, frustum]), components=[mesh, frustum]
    )
    path = editor.add(actor)

    result = _focus(editor, monkeypatch, path)

    nearest, reach = _box_in_view(result, mesh.center, mesh.extent)
    assert nearest > NEAR_CLIP and 0.4 < reach <= 1.0
    assert result["distance"] == pytest.approx(
        unreal_runtime._focus_fit_distance(mesh.extent)
    )
    _, across, up = _in_view(
        result["camera_location"], result["camera_rotation"], camera
    )
    assert (across, up) == pytest.approx((0.0, 0.0), abs=1e-9)
    assert "fit_warning" not in result


def test_the_frustum_of_a_child_actor_camera_is_left_out_too(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rig whose child actor component spawned a camera: the box spans the rig's mesh
    and the camera's mesh, not the camera's frustum."""
    body = PrimitiveComponent((0.0, 0.0, 50.0), (50.0, 50.0, 50.0))
    lens = PrimitiveComponent((0.0, 0.0, 300.0), (20.0, 10.0, 15.0))
    frustum = DrawFrustumComponent((0.0, 0.0, 300.0))
    camera = Actor(
        "Rig_Camera",
        (0.0, 0.0, 300.0),
        *_union([lens, frustum]),
        components=[lens, frustum],
    )
    rig = Actor(
        "Rig",
        (0.0, 0.0, 0.0),
        *_union([body, lens, frustum]),
        components=[body],
        children=[camera],
    )
    path = editor.add(rig)

    result = _focus(editor, monkeypatch, path)

    origin, extent = _union([body, lens])
    nearest, reach = _box_in_view(result, origin, extent)
    assert nearest > NEAR_CLIP and 0.4 < reach <= 1.0
    assert result["distance"] == pytest.approx(
        unreal_runtime._focus_fit_distance(extent)
    )


def test_without_a_frustum_the_bounds_are_those_unreal_reports(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """get_actor_bounds stays the source of the box whenever no frustum is left out: it is
    virtual, and a level instance adds its level's content to its own components', so
    the components alone would frame less than Unreal reports."""
    icon = PrimitiveComponent((0.0, 0.0, 0.0), (64.0, 64.0, 64.0))
    content = ((500.0, 0.0, 200.0), (800.0, 600.0, 200.0))
    instance = Actor("LevelInstance_0", (0.0, 0.0, 0.0), *content, components=[icon])
    frustum_only = Actor(
        "Frustum",
        (0.0, 5000.0, 0.0),
        (0.0, 5000.0, 0.0),
        (1000.0,) * 3,
        components=[DrawFrustumComponent((0.0, 5000.0, 0.0))],
    )
    for actor, (origin, extent) in (
        (instance, content),
        # An actor with nothing but a frustum keeps it, as the editor's own Focus does.
        (frustum_only, ((0.0, 5000.0, 0.0), (1000.0,) * 3)),
    ):
        result = _focus(editor, monkeypatch, editor.add(actor))

        assert actor.bounds_calls == [(False, True)]
        assert result["distance"] == pytest.approx(
            unreal_runtime._focus_fit_distance(extent)
        )
        assert result["camera_location"] == pytest.approx(
            _placement(origin, result["distance"])[0]
        )


def test_distance_0_on_an_actor_without_bounds_keeps_500_cm_and_says_so(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bounds without a size: a StaticMeshActor without a mesh, or an actor without a
    primitive component, whose bounds Unreal centers on the world origin. The camera
    looks at the actor's location instead."""
    path = editor.add(
        Actor("NoMesh", (300.0, -200.0, 50.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    )

    result = _focus(editor, monkeypatch, path)

    assert (result["camera_location"], result["camera_rotation"]) == _placement(
        (300.0, -200.0, 50.0), 500.0
    )
    assert result["distance"] == 500.0
    assert result["fit_warning"] == unreal_runtime.FOCUS_NO_BOUNDS_WARNING
    assert "no bounds" in result["fit_warning"] and "500 cm" in result["fit_warning"]
    assert result["selected"] is True and "warning" not in result


def test_an_explicit_distance_places_the_camera_as_before(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A positive distance still looks at the actor's location, whatever its bounds."""
    path = editor.add(
        Actor("Lamp", (100.0, 200.0, 0.0), (100.0, 200.0, 150.0), (50.0, 50.0, 150.0))
    )

    result = _focus(editor, monkeypatch, path, distance=250.0)

    assert editor.camera_writes == [
        {
            "CameraLocation": {"X": -150.0, "Y": 200.0, "Z": 125.0},
            "CameraRotation": {
                "Pitch": -math.degrees(math.atan2(125.0, 250.0)),
                "Yaw": 0.0,
                "Roll": 0.0,
            },
        }
    ]
    assert result == {
        "actor_path": path,
        "camera_location": (-150.0, 200.0, 125.0),
        "camera_rotation": (-math.degrees(math.atan2(125.0, 250.0)), 0.0, 0.0),
        "distance": 250.0,
        "selected": True,
        "pie_active": False,
    }


def test_distance_0_fits_the_actor_during_pie_too(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """During PIE the actor is not selected, but the camera still frames it."""
    editor.pie = True
    origin, extent = (0.0, 0.0, 1000.0), (1000.0, 1000.0, 1000.0)
    path = editor.add(Actor("Tower", (0.0, 0.0, 0.0), origin, extent))

    result = _focus(editor, monkeypatch, path)

    nearest, reach = _box_in_view(result, origin, extent)
    assert nearest > NEAR_CLIP and 0.4 < reach <= 1.0
    assert (result["selected"], result["pie_active"]) == (False, True)
    assert result["warning"] == unreal_runtime.FOCUS_PIE_WARNING


# ---------------------------------------------------------------------------
# Through the MCP tool
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
            return await client.call_tool(
                "focus_unreal_on_actor", arguments, raise_on_error=False
            )

    result = asyncio.run(run())
    return json.loads(
        [
            block.text
            for block in result.content
            if getattr(block, "type", None) == "text"
        ][-1]
    )


def test_an_agent_sees_the_distance_used_and_why_it_was_not_fitted(
    editor: Editor, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """The response schema keeps distance and fit_warning."""
    fitted = editor.add(
        Actor("Crate", (0.0, 0.0, 0.0), (0.0, 0.0, 50.0), (50.0, 50.0, 50.0))
    )
    empty = editor.add(
        Actor("NoMesh", (10.0, 20.0, 30.0), (10.0, 20.0, 30.0), (0.0, 0.0, 0.0))
    )

    fit = _call(editor, monkeypatch, tmp_path, {"actor_path": fitted})
    fallback = _call(editor, monkeypatch, tmp_path, {"actor_path": empty})

    assert fit["success"] is True and fit["fit_warning"] is None
    assert fit["camera_location"] == pytest.approx(
        list(_placement((0.0, 0.0, 50.0), fit["distance"])[0])
    )
    assert (fallback["success"], fallback["distance"]) == (True, 500.0)
    assert fallback["fit_warning"] == unreal_runtime.FOCUS_NO_BOUNDS_WARNING
