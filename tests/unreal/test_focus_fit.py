"""focus_unreal_on_actor with distance 0 fits the actor in the view.

Distance 0 was documented as auto-fit, but the camera always stood a fixed
500 cm behind the actor's location (UE 5.7.4): a 20 m cube put the camera
inside the cube, and a 10 cm one was a speck. Now distance 0 reads the actor's
bounds and backs the camera off until their bounding sphere fills the narrower
side of the view with a margin, looking at their center. Editor Python cannot
read the level viewport's field of view, so the caller can pass the view's
field of view across and its aspect ratio; the defaults are UE's default, 90
degrees across, on the 16:9 of the default capture. An actor without bounds
(zero extent) keeps the 500 cm and the reply says so; an explicit distance
places the camera as before.

The bounds leave out the editor-only components the editor's own Focus leaves
out, whose bounds can reach far beyond the actor: the frustum the editor draws
for a camera, a CineCamera's debug focus plane, an icon not attached to the
actor. They also leave out a shape whose collision is turned off, such as a
reflection capture's influence sphere, which collides with nothing. Any other
component that is not editor-only counts, a trigger's shape included, and an
actor that reports bounds other than the box around its components keeps them.

The adapter tests run the real editor script against a stand-in for the
``unreal`` module and the camera calls against a stand-in level viewport. The
fit is checked geometrically: the actor's bounding box is projected through the
camera read back.
"""

from __future__ import annotations

import asyncio
import enum
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
from simul.mcp.registration._unreal_tools import UNREAL_TOOLS
from simul.mcp.usage_tracker import ToolUsageTracker
from tests.fakes import AvailableAdapter
from tests.unreal.editor_world import (
    SUBSYSTEM_NAMES,
    World,
    editor_subsystems,
    is_valid,
)

LEVEL = "/Game/Maps/Lab.Lab:PersistentLevel."
WORLD = World("/Game/Maps/Lab.Lab")

# The view the fit assumes by default: UE's default level viewport field of
# view, 90 degrees across, on the 16:9 of the default capture. A view keeps its
# field of view across its width, so it reaches tan(fov / 2) across and 1 /
# aspect of that up per unit of depth. Unreal clips anything nearer than 10 cm.
FOV = 90.0
ASPECT = 16.0 / 9.0
NEAR_CLIP = 10.0


# ---------------------------------------------------------------------------
# A stand-in for the parts of ``unreal`` the focus script uses
# ---------------------------------------------------------------------------


class _Vector:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.x, self.y, self.z = x, y, z


class SceneComponent:
    """A scene component and the component it is attached to (None: attached to nothing)."""

    def __init__(self, parent: Any = None) -> None:
        self.parent: Any = parent

    def get_attach_parent(self) -> Any:
        return self.parent


class CollisionEnabled(enum.Enum):
    """``unreal.CollisionEnabled``."""

    NO_COLLISION = 0
    QUERY_ONLY = 1
    PHYSICS_ONLY = 2
    QUERY_AND_PHYSICS = 3


class PrimitiveComponent(SceneComponent):
    """A primitive component and its world bounds (``Bounds``) as a center and a half-size.

    ``editor_only`` is ``bIsEditorOnly``, ``ignore_focus`` what
    ``GetIgnoreBoundsForEditorFocus`` returns and ``collision`` what
    ``GetCollisionEnabled`` returns. An ``attached`` component is attached under the
    root of the actor it is given to; any other one is attached to nothing.
    """

    def __init__(
        self,
        center: Sequence[float],
        extent: Sequence[float],
        editor_only: bool = False,
        ignore_focus: bool = False,
        attached: bool = True,
        collision: CollisionEnabled = CollisionEnabled.QUERY_AND_PHYSICS,
    ) -> None:
        super().__init__()
        self.center, self.extent = tuple(center), tuple(extent)
        self.editor_only, self.ignore_focus = editor_only, ignore_focus
        self.attached, self.collision = attached, collision

    def get_editor_property(self, name: str) -> Any:
        return {"is_editor_only": self.editor_only}[name]

    def get_ignore_bounds_for_editor_focus(self) -> bool:
        return self.ignore_focus

    def get_collision_enabled(self) -> CollisionEnabled:
        return self.collision


class ShapeComponent(PrimitiveComponent):
    """A box, sphere or capsule component: the editor's own Focus ignores every shape
    (UShapeComponent::GetIgnoreBoundsForEditorFocus, UE 5.7). Unreal makes a trigger's
    shape and a reflection capture's influence shapes as plain components, not
    editor-only ones; the trigger's collides (the Trigger profile, query only), the
    influence shapes collide with nothing (UE 5.7.4)."""

    def __init__(
        self,
        center: Sequence[float],
        extent: Sequence[float],
        collision: CollisionEnabled,
        attached: bool = True,
    ) -> None:
        super().__init__(
            center, extent, ignore_focus=True, attached=attached, collision=collision
        )


class DrawFrustumComponent(PrimitiveComponent):
    """The frustum the editor draws for a camera: a cube reaching its far end (1000 cm by
    default) on every side of the camera (UDrawFrustumComponent::CalcBounds, UE 5.7). It is
    editor-only, and the editor's own Focus ignores its bounds."""

    def __init__(self, camera: Sequence[float], far: float = 1000.0) -> None:
        super().__init__(camera, (far, far, far), editor_only=True, ignore_focus=True)


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


def _reported(
    components: Sequence[PrimitiveComponent],
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """The same box as GetActorBounds reports it: FBox::GetCenterAndExtents puts the center
    at Min + 0.5 * (Max - Min), which can differ from ``_union``'s in the last bit."""
    low = [min(c.center[i] - c.extent[i] for c in components) for i in range(3)]
    high = [max(c.center[i] + c.extent[i] for c in components) for i in range(3)]
    half = [0.5 * (hi - lo) for lo, hi in zip(low, high)]
    return tuple(lo + h for lo, h in zip(low, half)), tuple(half)


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
        self.root_component = SceneComponent()
        for component in self.components:
            if component.attached:
                component.parent = self.root_component
        self.bounds_calls: List[Tuple[bool, bool]] = []

    def get_world(self) -> World:
        return WORLD

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
            **editor_subsystems(WORLD, pie=lambda: editor.pie),
        }
        return SimpleNamespace(
            Actor=Actor,
            PrimitiveComponent=PrimitiveComponent,
            ShapeComponent=ShapeComponent,
            CollisionEnabled=CollisionEnabled,
            DrawFrustumComponent=DrawFrustumComponent,
            SystemLibrary=SimpleNamespace(get_component_bounds=get_component_bounds, is_valid=is_valid),
            EditorActorSubsystem="EditorActorSubsystem",
            LevelEditorSubsystem="LevelEditorSubsystem",
            **SUBSYSTEM_NAMES,
            get_editor_subsystem=subsystems.__getitem__,
            find_object=lambda outer, path: editor.actors.get(path),
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
    camera: Sequence[float],
    rotation: Sequence[float],
    point: Sequence[float],
    fov: float = FOV,
    aspect: float = ASPECT,
) -> Tuple[float, float, float]:
    """Where ``point`` is in the view of a camera at ``camera`` with (pitch, yaw, roll) ``rotation``.

    The view is ``fov`` degrees across and ``aspect`` wide per unit of height.
    Returns the depth along the view axis, and how far across and up the
    point is, as fractions of the view's half-width and half-height at that
    depth: the point is in the view when both are within [-1, 1].
    """
    half_width = math.tan(math.radians(fov) / 2)
    half_height = half_width / aspect
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
    return depth, along(right) / depth / half_width, along(up) / depth / half_height


def _box_in_view(
    result: Dict[str, Any],
    origin: Sequence[float],
    extent: Sequence[float],
    fov: float = FOV,
    aspect: float = ASPECT,
) -> Tuple[float, float]:
    """The nearest depth of the box's corners, and the largest share of the view they reach.

    Fails when a corner is behind the camera, where it has no place in the view.
    """
    reach, nearest = 0.0, math.inf
    for signs in [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]:
        corner = [o + s * e for o, s, e in zip(origin, signs, extent)]
        depth, across, up = _in_view(
            result["camera_location"], result["camera_rotation"], corner, fov, aspect
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


def test_distance_0_frames_a_cine_camera_by_its_mesh_not_by_its_debug_focus_plane(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CineCamera's "Draw Debug Focus Plane" adds an editor-only plane, 100 m across at
    the focus distance (1 km by default), and get_actor_bounds includes it: the camera
    stood 1 km away and the CineCamera was out of sight (UE 5.7.4), also once the plane
    was turned off, which only hides it. The editor's own Focus ignores the plane's
    bounds, and so does the fit: the CineCamera is framed by its mesh, which is
    editor-only as well but counts, as in the editor.

    As in the editor, the mesh, the frustum and the plane hang from the
    CineCameraComponent, which is attached to the actor's root, and get_actor_bounds
    centers its box as Unreal does, here a last bit off the center of the box around
    the components. The mesh still counts as attached under the root, and the box
    Unreal reports still counts as the box around the components."""
    camera = (250.6, 10000.0, 200.0)
    mesh = PrimitiveComponent(
        (204.6, 10000.0, 202.0), (47.4, 21.6, 27.7), editor_only=True
    )
    frustum = DrawFrustumComponent(camera)
    plane = PrimitiveComponent(
        (100250.6, 10000.0, 200.0),
        (0.0, 5000.0, 5000.0),
        editor_only=True,
        ignore_focus=True,
    )
    components = [mesh, frustum, plane]
    assert _reported(components) != _union(components)
    actor = Actor(
        "CineCameraActor_0", camera, *_reported(components), components=components
    )
    cine_camera_component = SceneComponent(parent=actor.root_component)
    for component in components:
        component.parent = cine_camera_component
    path = editor.add(actor)

    result = _focus(editor, monkeypatch, path)

    nearest, reach = _box_in_view(result, mesh.center, mesh.extent)
    assert nearest > NEAR_CLIP and 0.4 < reach <= 1.0
    assert result["distance"] == pytest.approx(
        unreal_runtime._focus_fit_distance(mesh.extent)
    )
    _, across, up = _in_view(
        result["camera_location"], result["camera_rotation"], mesh.center
    )
    assert (across, up) == pytest.approx((0.0, 0.0), abs=1e-9)
    assert "fit_warning" not in result


def test_an_editor_only_component_not_attached_to_the_actor_is_left_out(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A SkyLight keeps a spare editor icon attached to nothing, so it sits at the world
    origin, and get_actor_bounds includes it: a SkyLight 800 m from the origin was framed
    from 805 m away (UE 5.7.4). The editor's own Focus visits only the components
    attached under the actor's root, and the fit leaves out an editor-only one that is
    not: the SkyLight is framed by its own icon."""
    light = (0.0, 80000.0, 200.0)
    icon = PrimitiveComponent(light, (256.0,) * 3, editor_only=True)
    spare = PrimitiveComponent(
        (0.0, 0.0, 0.0), (1.0,) * 3, editor_only=True, attached=False
    )
    actor = Actor("SkyLight_0", light, *_union([spare, icon]), components=[spare, icon])

    result = _focus(editor, monkeypatch, editor.add(actor))

    assert result["distance"] == pytest.approx(
        unreal_runtime._focus_fit_distance(icon.extent)
    )
    assert result["camera_location"] == pytest.approx(
        _placement(light, result["distance"])[0]
    )


def test_a_component_that_is_not_editor_only_counts_even_where_the_editor_focus_skips_it(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The editor's own Focus ignores collision shapes too, and frames a trigger box by
    its icon. The shape is the trigger's volume, which collides, not a drawing, so it
    counts, like any component that is not editor-only, attached to the actor or not."""
    volume = ShapeComponent(
        (0.0, 0.0, 200.0), (500.0, 500.0, 200.0), CollisionEnabled.QUERY_ONLY
    )
    icon = PrimitiveComponent((0.0, 0.0, 200.0), (128.0,) * 3, editor_only=True)
    loose = PrimitiveComponent((2000.0, 0.0, 200.0), (10.0,) * 3, attached=False)
    components = [volume, icon, loose]
    path = editor.add(
        Actor(
            "TriggerBox_0",
            (0.0, 0.0, 200.0),
            *_union(components),
            components=components,
        )
    )

    result = _focus(editor, monkeypatch, path)

    origin, extent = _union(components)
    assert result["distance"] == pytest.approx(
        unreal_runtime._focus_fit_distance(extent)
    )
    assert result["camera_location"] == pytest.approx(
        _placement(origin, result["distance"])[0]
    )


@pytest.mark.parametrize(
    "influence",
    [
        # A SphereReflectionCapture: its influence sphere, 30 m in radius by default,
        # and the small sphere of its capture radius.
        [((3000.0,) * 3), ((32.0,) * 3)],
        # A BoxReflectionCapture: its influence box and its capture box.
        [(1000.0, 1000.0, 400.0), (900.0, 900.0, 300.0)],
    ],
    ids=["sphere reflection capture", "box reflection capture"],
)
def test_distance_0_frames_a_reflection_capture_by_its_icon_not_by_its_influence_shapes(
    editor: Editor, monkeypatch: pytest.MonkeyPatch, influence: List[Tuple[float, ...]]
) -> None:
    """Unreal draws a reflection capture's influence with shapes it makes as plain
    components, not editor-only ones, so they counted: a SphereReflectionCapture was
    framed by its 30 m influence sphere from 104 m away, its icon a speck (UE 5.7.4).
    The shapes collide with nothing, which tells them from a trigger's, and the
    editor's own Focus ignores every shape: the capture is framed by its icons, as a
    light is."""
    capture = (0.0, 50000.0, 200.0)
    # Its icons collide with nothing either, but they are not shapes.
    sprite, offset = (
        PrimitiveComponent(
            capture, extent, editor_only=True, collision=CollisionEnabled.NO_COLLISION
        )
        for extent in ((128.0,) * 3, (51.2,) * 3)
    )
    shapes = [
        ShapeComponent(capture, extent, CollisionEnabled.NO_COLLISION)
        for extent in influence
    ]
    components = [sprite, offset, *shapes]
    path = editor.add(
        Actor(
            "ReflectionCapture_0", capture, *_union(components), components=components
        )
    )

    result = _focus(editor, monkeypatch, path)

    assert result["distance"] == pytest.approx(
        unreal_runtime._focus_fit_distance(sprite.extent)
    )
    assert result["camera_location"] == pytest.approx(
        _placement(capture, result["distance"])[0]
    )
    nearest, reach = _box_in_view(result, capture, sprite.extent)
    assert nearest > NEAR_CLIP and 0.4 < reach <= 1.0
    assert "fit_warning" not in result


def test_with_nothing_left_out_the_bounds_are_those_unreal_reports(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """get_actor_bounds stays the source of the box whenever nothing is left out: it is
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
    marker = ((0.0, 9000.0, 100.0), (400.0, 300.0, 100.0))
    marker_only = Actor(
        "Marker",
        marker[0],
        *marker,
        components=[ShapeComponent(*marker, CollisionEnabled.NO_COLLISION)],
    )
    for actor, (origin, extent) in (
        (instance, content),
        # An actor with nothing but a frustum, or nothing but a shape that collides
        # with nothing, keeps it, as the editor's own Focus does.
        (frustum_only, ((0.0, 5000.0, 0.0), (1000.0,) * 3)),
        (marker_only, marker),
    ):
        result = _focus(editor, monkeypatch, editor.add(actor))

        assert actor.bounds_calls == [(False, True)]
        assert result["distance"] == pytest.approx(
            unreal_runtime._focus_fit_distance(extent)
        )
        assert result["camera_location"] == pytest.approx(
            _placement(origin, result["distance"])[0]
        )


def test_an_actor_reporting_bounds_of_its_own_keeps_them(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A NavLinkProxy reports the box around its links, not around its components, and
    its editor-only drawing of the links is one whose bounds the editor's own Focus
    ignores. Leaving that out would frame a 12 m link by the proxy's icon: components
    are left out only while get_actor_bounds is the box around all of them."""
    links = ((0.0, 0.0, 100.0), (0.0, 600.0, 10.0))
    drawing = PrimitiveComponent(
        (0.0, 0.0, 100.0), (0.0, 600.0, 0.0), editor_only=True, ignore_focus=True
    )
    icon = PrimitiveComponent((0.0, 0.0, 100.0), (128.0,) * 3, editor_only=True)
    proxy = Actor(
        "NavLinkProxy_0", (0.0, 0.0, 100.0), *links, components=[drawing, icon]
    )

    result = _focus(editor, monkeypatch, editor.add(proxy))

    assert result["distance"] == pytest.approx(
        unreal_runtime._focus_fit_distance(links[1])
    )
    assert result["camera_location"] == pytest.approx(
        _placement(links[0], result["distance"])[0]
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
# The view the actor is fitted in
# ---------------------------------------------------------------------------


def test_the_default_view_is_the_default_field_of_view_on_the_default_capture() -> None:
    """Without a view, the fit is unchanged: 90 degrees across on 16:9, which put a
    20 m cube 3475.9 cm away and a 10 cm one 20.07 cm away (UE 5.7.4). The tool
    advertises the session's defaults."""
    extent = (1000.0, 1000.0, 1000.0)
    assert unreal_runtime._focus_fit_distance(extent) == pytest.approx(
        3475.93, abs=0.01
    )
    assert unreal_runtime._focus_fit_distance((5.0, 5.0, 5.0)) == pytest.approx(
        20.07, abs=0.01
    )
    assert unreal_runtime._focus_fit_distance(
        extent
    ) == unreal_runtime._focus_fit_distance(extent, FOV, ASPECT)
    spec = next(s for s in UNREAL_TOOLS if s.name == "focus_unreal_on_actor")
    defaults = {param.name: param.default for param in spec.params}
    assert (defaults["fov"], defaults["aspect_ratio"]) == (
        unreal_runtime.FOCUS_FOV_DEGREES,
        unreal_runtime.FOCUS_ASPECT_RATIO,
    )


@pytest.mark.parametrize(
    "fov, aspect, cropped_by_default",
    [
        (50.0, ASPECT, True),
        (90.0, 3.0, True),
        (30.0, 1.0, True),
        # Taller than wide: the width is the narrower side, and the default fit,
        # which stood back for the height of a 16:9 view, left more room than needed.
        (90.0, 9.0 / 16.0, False),
    ],
    ids=["narrower field of view", "3:1", "30 degrees square", "9:16"],
)
def test_distance_0_fits_the_actor_in_the_view_it_is_given(
    editor: Editor,
    monkeypatch: pytest.MonkeyPatch,
    fov: float,
    aspect: float,
    cropped_by_default: bool,
) -> None:
    """Editor Python cannot read the level viewport's field of view, so the fit
    assumed UE's default, 90 degrees across, on 16:9: a narrower field of view or a
    view wider than about 2:1 cropped the actor. Given the view, the fit backs off
    until the actor fits it, with the same margin."""
    origin, extent = (0.0, 0.0, 1000.0), (1000.0, 1000.0, 1000.0)
    path = editor.add(Actor("Tower", (0.0, 0.0, 0.0), origin, extent))

    default = _focus(editor, monkeypatch, path)
    result = _focus(editor, monkeypatch, path, fov=fov, aspect_ratio=aspect)

    assert (_box_in_view(default, origin, extent, fov, aspect)[1] > 1.0) is (
        cropped_by_default
    )
    nearest, reach = _box_in_view(result, origin, extent, fov, aspect)
    assert nearest > NEAR_CLIP and 0.4 < reach <= 1.0
    assert result["distance"] == pytest.approx(
        unreal_runtime._focus_fit_distance(extent, fov, aspect)
    )
    camera, rotation = _placement(origin, result["distance"])
    assert result["camera_location"] == pytest.approx(camera)
    assert result["camera_rotation"] == pytest.approx(rotation)
    assert "fit_warning" not in result


@pytest.mark.parametrize(
    "size, fov, aspect",
    [
        (10.0, 90.0, 1.0),
        (10.0, 90.0, 9.0 / 16.0),
        (10.0, 120.0, ASPECT),
        (2.0, 170.0, ASPECT),
        (30.0, 170.0, ASPECT),
    ],
    ids=[
        "10 cm cube, square",
        "10 cm cube, 9:16",
        "10 cm cube, 120 degrees",
        "2 cm pebble, 170 degrees",
        "30 cm box, 170 degrees",
    ],
)
def test_a_small_actor_in_a_wide_or_square_view_stays_clear_of_the_near_clip_plane(
    editor: Editor,
    monkeypatch: pytest.MonkeyPatch,
    size: float,
    fov: float,
    aspect: float,
) -> None:
    """In the default view the 10 cm minimum radius keeps the camera far enough back,
    but a view as tall as it is wide, or a wider field of view, fitted a small actor
    from so close that the near clip plane cut it: a 10 cm cube's nearest corner was
    8.8 cm from the camera in a square view. The camera stays back until the bounding
    sphere, with the fit's margin, is beyond the plane."""
    origin, extent = (0.0, 0.0, size / 2), (size / 2,) * 3
    path = editor.add(Actor("Pebble", (0.0, 0.0, 0.0), origin, extent))

    result = _focus(editor, monkeypatch, path, fov=fov, aspect_ratio=aspect)

    nearest, reach = _box_in_view(result, origin, extent, fov, aspect)
    assert nearest > NEAR_CLIP and reach <= 1.0
    radius = max(math.hypot(*extent), unreal_runtime.FOCUS_MIN_RADIUS)
    assert result["distance"] * math.hypot(1.0, 0.5) == pytest.approx(
        radius * unreal_runtime.FOCUS_FIT_MARGIN + NEAR_CLIP
    )
    camera, rotation = _placement(origin, result["distance"])
    assert result["camera_location"] == pytest.approx(camera)
    assert result["camera_rotation"] == pytest.approx(rotation)


def test_the_view_leaves_an_explicit_distance_alone(
    editor: Editor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fov and aspect_ratio shape the auto-fit only: a positive distance still looks
    at the actor's location from that distance."""
    path = editor.add(
        Actor("Lamp", (100.0, 200.0, 0.0), (100.0, 200.0, 150.0), (50.0, 50.0, 150.0))
    )

    result = _focus(
        editor, monkeypatch, path, distance=250.0, fov=30.0, aspect_ratio=1.0
    )

    assert result["camera_location"] == (-150.0, 200.0, 125.0)
    assert result["distance"] == 250.0


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


def test_an_agent_can_give_the_view_to_fit(
    editor: Editor, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """fov and aspect_ratio reach the fit through the tool."""
    extent = (50.0, 50.0, 50.0)
    path = editor.add(Actor("Crate", (0.0, 0.0, 0.0), (0.0, 0.0, 50.0), extent))

    fit = _call(
        editor,
        monkeypatch,
        tmp_path,
        {"actor_path": path, "fov": 40.0, "aspect_ratio": 2.4},
    )

    assert fit["success"] is True
    assert fit["distance"] == pytest.approx(
        unreal_runtime._focus_fit_distance(extent, 40.0, 2.4)
    )
    nearest, reach = _box_in_view(fit, (0.0, 0.0, 50.0), extent, 40.0, 2.4)
    assert nearest > NEAR_CLIP and 0.4 < reach <= 1.0


@pytest.mark.parametrize(
    "view, named",
    [
        ({"fov": 0.0}, "fov"),
        ({"fov": 180.0}, "fov"),
        ({"fov": -45.0}, "fov"),
        # Narrower than Unreal renders: 5e-324 divided by zero in the fit, after
        # the actor was selected.
        ({"fov": 0.0005}, "fov"),
        ({"fov": 5e-324}, "fov"),
        ({"aspect_ratio": 0.0}, "aspect_ratio"),
        ({"aspect_ratio": -1.5}, "aspect_ratio"),
        # Wider or narrower than a capture can be: 1e308 put the camera at infinity.
        ({"aspect_ratio": 20000.0}, "aspect_ratio"),
        ({"aspect_ratio": 1.0 / 20000.0}, "aspect_ratio"),
        ({"aspect_ratio": 1e308}, "aspect_ratio"),
    ],
)
def test_a_view_the_fit_cannot_use_is_refused_before_anything_moves(
    editor: Editor,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    view: Dict[str, float],
    named: str,
) -> None:
    """A field of view narrower than Unreal renders (0.001 degrees) or not under 180,
    or an aspect ratio no capture has (outside 1/16384 to 16384), has no fit: the tool
    refuses it as invalid input, and the session raises, before the actor is selected
    or the camera moves."""
    path = editor.add(
        Actor("Crate", (0.0, 0.0, 0.0), (0.0, 0.0, 50.0), (50.0, 50.0, 50.0))
    )

    refused = _call(editor, monkeypatch, tmp_path, {"actor_path": path, **view})

    assert (refused["success"], refused["error_type"]) == (False, "ValidationError")
    assert named in refused["error"]
    with pytest.raises(ValueError, match=named):
        asyncio.run(_session(editor, monkeypatch).focus_on_actor(path, **view))
    assert editor.selection == [] and editor.camera_writes == []


@pytest.mark.parametrize(
    "fov, aspect",
    [
        (unreal_runtime.FOCUS_MIN_FOV_DEGREES, unreal_runtime.FOCUS_MAX_ASPECT_RATIO),
        (179.999, 1.0 / unreal_runtime.FOCUS_MAX_ASPECT_RATIO),
    ],
    ids=["narrowest view", "widest view"],
)
def test_the_views_at_the_edges_of_the_range_still_fit_at_a_finite_distance(
    fov: float, aspect: float
) -> None:
    """The views the tool takes all have a fit, even for bounds larger than any world."""
    assert unreal_runtime.focus_view_error(fov, aspect) is None
    for extent in ((0.0, 0.0, 0.0), (1e12, 1e12, 1e12)):
        assert math.isfinite(unreal_runtime._focus_fit_distance(extent, fov, aspect))
