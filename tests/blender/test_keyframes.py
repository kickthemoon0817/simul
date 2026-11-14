"""Keyframe tools against the action API of each Blender version the add-on supports.

``get_blender_keyframes`` read ``Action.fcurves``, which Blender 5.0 removed:
since 4.4 an action keeps the F-curves of each ID it animates in a channelbag
per action slot, inside its layers and strips (slotted actions). On 5.0 the
tool failed with ``'Action' object has no attribute 'fcurves'`` right after
``insert_blender_keyframe`` succeeded, and on 4.4 and 4.5 the deprecated
``Action.fcurves`` holds only the action's first slot, whichever object asks.

The doubles mirror ``bpy.types`` of Blender 4.2/4.3, 4.4/4.5 and 5.0 with
``__slots__``, so reading an attribute a version does not have raises the
AttributeError Blender raises, and ``bpy_extras.anim_utils`` offers exactly the
functions each version ships.
"""

from __future__ import annotations

import json
import math
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest

from simul.adapters import blender_runtime
from simul.utils.paths import PathPolicy

Version = Tuple[int, int, int]

BLENDER_4_2 = (4, 2, 0)
BLENDER_4_4 = (4, 4, 0)  # The least forgiving 4.4/4.5 release, see ANIM_UTILS.
BLENDER_5_0 = (5, 0, 1)

every_version = pytest.mark.parametrize(
    "version", [BLENDER_4_2, BLENDER_4_4, BLENDER_5_0], ids=["4.2", "4.4", "5.0"]
)
slotted_versions = pytest.mark.parametrize(
    "version", [BLENDER_4_4, BLENDER_5_0], ids=["4.4", "5.0"]
)


# -- bpy.types doubles -------------------------------------------------------


class Keyframe:
    """``bpy.types.Keyframe``: its (frame, value) point, interpolation, key type and handles.

    A new key interpolates as Blender's default preference does, Bezier, with
    automatic clamped handles a frame to either side.
    """

    __slots__ = (
        "co",
        "interpolation",
        "type",
        "easing",
        "handle_left",
        "handle_right",
        "handle_left_type",
        "handle_right_type",
    )

    def __init__(
        self, frame: float, value: float = 0.0, key_type: str = "KEYFRAME"
    ) -> None:
        self.co = [frame, value]
        self.interpolation = "BEZIER"
        self.type = key_type
        self.easing = "AUTO"
        self.handle_left = [frame - 1.0, value]
        self.handle_right = [frame + 1.0, value]
        self.handle_left_type = "AUTO_CLAMPED"
        self.handle_right_type = "AUTO_CLAMPED"


class KeyframePoints(list):
    """``FCurveKeyframePoints``: the keys in order, and ``foreach_get`` of their points."""

    def foreach_get(self, attribute: str, buffer: List[float]) -> None:
        assert attribute == "co" and len(buffer) == 2 * len(self)
        buffer[:] = [v for key in self for v in key.co]


class FCurve:
    """``bpy.types.FCurve``: what it animates, its keys, whether Blender may key it, and ``evaluate``.

    ``evaluate`` holds a Constant key's value to the next key and goes in a
    straight line from any other, standing in for Blender's curves.
    """

    __slots__ = (
        "data_path",
        "array_index",
        "_keyframe_points",
        "lock",
        "group",
        "mute",
        "sampled_points",
        "modifiers",
    )

    def __init__(self, data_path: str, array_index: int, *frames: float) -> None:
        self.data_path = data_path
        self.array_index = array_index
        self.keyframe_points = [Keyframe(frame) for frame in frames]
        self.lock = False
        # An ActionGroup stand-in with ``lock`` and ``mute``, or None.
        self.group: Any = None
        self.mute = False
        self.sampled_points: List[Any] = []
        # FModifier stand-ins: type, mute, is_valid and, for a generator, use_additive.
        self.modifiers: List[Any] = []

    @property
    def keyframe_points(self) -> KeyframePoints:
        return self._keyframe_points

    @keyframe_points.setter
    def keyframe_points(self, keys: List[Keyframe]) -> None:
        self._keyframe_points = KeyframePoints(keys)

    def evaluate(self, frame: float) -> float:
        keys = sorted(self.keyframe_points, key=lambda k: k.co[0])
        if frame <= keys[0].co[0]:
            return float(keys[0].co[1])
        for left, right in zip(keys, keys[1:]):
            if frame < right.co[0]:
                if left.interpolation == "CONSTANT":
                    return float(left.co[1])
                share = (frame - left.co[0]) / (right.co[0] - left.co[0])
                return float(left.co[1] + share * (right.co[1] - left.co[1]))
        return float(keys[-1].co[1])


def curve(data_path: str, array_index: int, *points: Tuple[float, float]) -> FCurve:
    """An F-curve keyed at (frame, value) ``points``."""
    fcurve = FCurve(data_path, array_index)
    fcurve.keyframe_points = [Keyframe(frame, value) for frame, value in points]
    return fcurve


class ActionSlot:
    """``bpy.types.ActionSlot`` (4.4+)."""

    __slots__ = ("identifier",)

    def __init__(self, identifier: str) -> None:
        self.identifier = identifier


class ActionChannelbag:
    """``bpy.types.ActionChannelbag`` (4.4+): the F-curves of one slot."""

    __slots__ = ("slot", "fcurves")

    def __init__(self, slot: ActionSlot, fcurves: List[FCurve]) -> None:
        self.slot = slot
        self.fcurves = fcurves


class ActionKeyframeStrip:
    """``bpy.types.ActionKeyframeStrip`` (4.4+)."""

    __slots__ = ("channelbags",)

    def __init__(self, channelbags: List[ActionChannelbag]) -> None:
        self.channelbags = channelbags

    def channelbag(
        self, slot: Optional[ActionSlot], ensure: bool = False
    ) -> Optional[ActionChannelbag]:
        if slot is None:
            # Blender 4.4.1 and later refuse a missing slot so; 4.4.0
            # dereferences it and crashes.
            raise RuntimeError("Error: Cannot return channelbag when slot is None")
        return next((bag for bag in self.channelbags if bag.slot is slot), None)


class ActionLayer:
    """``bpy.types.ActionLayer`` (4.4+)."""

    __slots__ = ("strips",)

    def __init__(self, strips: List[ActionKeyframeStrip]) -> None:
        self.strips = strips


class Blender42:
    """``bpy.types`` of Blender 4.2 and 4.3: an action's F-curves are all its ID's."""

    class Action:
        __slots__ = ("fcurves", "groups")

        def __init__(self, fcurves: List[FCurve]) -> None:
            self.fcurves = fcurves
            self.groups: List[Any] = []

    class AnimData:
        __slots__ = ("action", "nla_tracks", "use_nla", "drivers")

        def __init__(self, action: Any) -> None:
            self.action = action
            self.nla_tracks: List[Any] = []
            self.use_nla = True
            # Driver F-curves: data_path and array_index.
            self.drivers: List[Any] = []


class Blender50:
    """``bpy.types`` of Blender 5.0: slotted actions only, no ``Action.fcurves``."""

    class Action:
        __slots__ = ("layers", "slots")

        def __init__(self, layers: List[ActionLayer], slots: List[ActionSlot]) -> None:
            self.layers = layers
            self.slots = slots

    class AnimData:
        __slots__ = ("action", "action_slot", "nla_tracks", "use_nla", "drivers")

        def __init__(self, action: Any, action_slot: Optional[ActionSlot]) -> None:
            self.action = action
            self.action_slot = action_slot
            self.nla_tracks: List[Any] = []
            self.use_nla = True
            # Driver F-curves: data_path and array_index.
            self.drivers: List[Any] = []


class Blender44:
    """``bpy.types`` of Blender 4.4 and 4.5: slotted actions.

    ``Action.fcurves`` is still there, deprecated, and holds the F-curves of the
    action's first slot only.
    """

    class Action(Blender50.Action):
        __slots__ = ()

        @property
        def fcurves(self) -> List[FCurve]:
            bag = self.layers[0].strips[0].channelbag(self.slots[0])
            return [] if bag is None else bag.fcurves

    AnimData = Blender50.AnimData


def channelbag_for_slot_4_4_0(action: Any, slot: Any) -> Optional[ActionChannelbag]:
    """``bpy_extras.anim_utils.action_get_channelbag_for_slot`` of Blender 4.4.0,
    which passes a missing slot on to ``channelbag``."""
    for layer in action.layers:
        for strip in layer.strips:
            channelbag = strip.channelbag(slot)
            if channelbag:
                return channelbag
    return None


def channelbag_for_slot(action: Any, slot: Any) -> Optional[ActionChannelbag]:
    """The same function from Blender 4.4.1 on, which returns None for it."""
    if not action or not slot:
        return None
    return channelbag_for_slot_4_4_0(action, slot)


# What each version's bpy_extras.anim_utils has besides its bake functions.
ANIM_UTILS: Dict[Version, Dict[str, Callable[..., Any]]] = {
    BLENDER_4_2: {},
    BLENDER_4_4: {"action_get_channelbag_for_slot": channelbag_for_slot_4_4_0},
    BLENDER_5_0: {"action_get_channelbag_for_slot": channelbag_for_slot},
}


def bpy_types(version: Version) -> Any:
    """The doubles of ``version``'s action types."""
    if version < BLENDER_4_4:
        return Blender42
    return Blender44 if version < (5, 0, 0) else Blender50


def slotted_action(version: Version, slots: Dict[ActionSlot, List[FCurve]]) -> Any:
    """An action as Blender keys one: a layer, a strip, a channelbag per slot."""
    strip = ActionKeyframeStrip(
        [ActionChannelbag(slot, fcurves) for slot, fcurves in slots.items()]
    )
    return bpy_types(version).Action([ActionLayer([strip])], list(slots))


class Object:
    """``bpy.types.Object``: its animation data, keyable properties and keyframe functions.

    ``keyframe_insert`` and ``keyframe_delete`` take their arguments as Blender
    5.0 documents them and, like Blender, report a refused or missing key by
    returning False. ``keyframe_insert`` keys the property's current value
    into ``fcurves``, the curves Blender keys this object into, replacing the
    value of a key within 0.01 frames, which keeps its frame, as on Blender
    5.0.1; ``keyframe_delete`` removes the keys on the frame from them.
    ``location`` stands for an array property, ``props`` for custom
    properties (``obj["name"]``) and ``paths`` for the structs ``path_resolve``
    finds, such as a pose bone.
    """

    __slots__ = (
        "name",
        "animation_data",
        "fcurves",
        "locked",
        "calls",
        "location",
        "props",
        "paths",
        "pose",
        "tags",
    )

    def __init__(
        self,
        name: str,
        animation_data: Any = None,
        fcurves: Optional[List[FCurve]] = None,
        locked: bool = False,
    ) -> None:
        self.name = name
        self.animation_data = animation_data
        self.fcurves = fcurves if fcurves is not None else []
        self.locked = locked
        self.calls: List[Tuple[Any, ...]] = []
        self.location: List[float] = [0.0, 0.0, 0.0]
        self.props: Dict[str, Any] = {}
        self.paths: Dict[str, Any] = {}
        # An armature's pose: ``bones.get(name)`` finds a pose bone.
        self.pose: Any = None
        # The refresh sets of update_tag calls the next evaluation applies.
        self.tags: List[set] = []

    def update_tag(self, refresh: Any = ()) -> None:
        self.tags.append(set(refresh))

    def evaluate_animation(self, frame: float) -> None:
        """Write every channel's value at ``frame`` to its property, as Blender's
        animation evaluation does for a tagged ID."""
        for fcurve in self.fcurves:
            if fcurve.keyframe_points:
                self._write(fcurve.data_path, fcurve.array_index, fcurve.evaluate(frame))

    def _write(self, data_path: str, index: int, value: float) -> None:
        if data_path.startswith('["'):
            key = data_path[2:-2]
            current = self.props[key]
            self.props[key] = int(value) if isinstance(current, int) else value
            return
        owner_path, _, name = data_path.rpartition(".")
        owner = self.path_resolve(owner_path) if owner_path else self
        current = getattr(owner, name)
        if isinstance(current, list):
            current[index] = value
        else:
            setattr(owner, name, value)

    def keys(self) -> List[str]:
        return list(self.props)

    def __getitem__(self, key: str) -> Any:
        return self.props[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.props[key] = value

    def path_resolve(self, path: str) -> Any:
        if path not in self.paths:
            raise ValueError(f'Object.path_resolve("{path}") could not be resolved')
        return self.paths[path]

    def _current(self, data_path: str) -> Any:
        if data_path.startswith('["'):
            return self.props[data_path[2:-2]]
        owner_path, _, name = data_path.rpartition(".")
        owner = self.path_resolve(owner_path) if owner_path else self
        if not hasattr(owner, name):
            raise TypeError(
                f'bpy_struct.keyframe_insert() property "{data_path}" not found'
            )
        return getattr(owner, name)

    def keyframe_insert(
        self,
        data_path: str,
        /,
        *,
        index: int = -1,
        frame: float = 1.0,
        group: str = "",
        options: Any = frozenset(),
        keytype: str = "KEYFRAME",
    ) -> bool:
        self.calls.append(("keyframe_insert", data_path, index, frame, keytype))
        if self.locked:
            return False
        current = self._current(data_path)
        values = list(current) if isinstance(current, list) else [current]
        for element in range(len(values)) if index < 0 else [index]:
            fcurve = next(
                (
                    c
                    for c in self.fcurves
                    if c.data_path == data_path and c.array_index == element
                ),
                None,
            )
            if fcurve is None:
                fcurve = FCurve(data_path, element)
                self.fcurves.append(fcurve)
            near = [k for k in fcurve.keyframe_points if abs(k.co[0] - frame) <= 0.01]
            if near:
                near[0].co[1] = float(values[element])
                near[0].type = keytype
                continue
            key = Keyframe(frame, float(values[element]), keytype)
            fcurve.keyframe_points = sorted(
                list(fcurve.keyframe_points) + [key], key=lambda k: k.co[0]
            )
        return True

    def keyframe_delete(
        self,
        data_path: str,
        /,
        *,
        index: int = -1,
        frame: float = 1.0,
        group: str = "",
    ) -> bool:
        self.calls.append(("keyframe_delete", data_path, index, frame))
        if self.animation_data is None:
            raise RuntimeError(
                "Error: No ID block and/or AnimData to delete keyframe from"
            )
        if self.locked:
            return False
        deleted = False
        for fcurve in self.fcurves:
            if fcurve.data_path != data_path or index not in (-1, fcurve.array_index):
                continue
            kept = [k for k in fcurve.keyframe_points if abs(k.co[0] - frame) >= 0.01]
            deleted = deleted or len(kept) < len(fcurve.keyframe_points)
            fcurve.keyframe_points = kept
        return deleted


def keyed_ball(version: Version, *fcurves: FCurve, locked: bool = False) -> Object:
    """An object 'Ball' whose own action holds ``fcurves`` as ``version`` keeps them."""
    curves = list(fcurves)
    if version < BLENDER_4_4:
        animation = Blender42.AnimData(Blender42.Action(curves))
    else:
        own = ActionSlot("OBBall")
        action = slotted_action(version, {own: curves})
        animation = bpy_types(version).AnimData(action, own)
    return Object("Ball", animation, curves, locked=locked)


def session_on(
    monkeypatch: pytest.MonkeyPatch, version: Version, *objects: Object
) -> blender_runtime.BlenderRuntimeSession:
    """A session on a stand-in ``bpy`` and ``bpy_extras`` of ``version``."""
    anim_utils = ModuleType("bpy_extras.anim_utils")
    for name, function in ANIM_UTILS[version].items():
        setattr(anim_utils, name, function)
    bpy_extras = ModuleType("bpy_extras")
    setattr(bpy_extras, "anim_utils", anim_utils)
    monkeypatch.setitem(sys.modules, "bpy_extras", bpy_extras)
    monkeypatch.setitem(sys.modules, "bpy_extras.anim_utils", anim_utils)
    scene = SimpleNamespace(
        frame_current=1,
        frame_subframe=0.0,
        render=SimpleNamespace(frame_map_old=100, frame_map_new=100),
    )

    def evaluated_depsgraph_get() -> None:
        """Evaluate what update_tag tagged, as bpy's call does on the active depsgraph."""
        for obj in objects:
            if any("TIME" in refresh for refresh in obj.tags):
                obj.evaluate_animation(scene.frame_current)
            obj.tags.clear()

    fake_bpy = SimpleNamespace(
        app=SimpleNamespace(version=version),
        context=SimpleNamespace(
            scene=scene, evaluated_depsgraph_get=evaluated_depsgraph_get
        ),
        data=SimpleNamespace(objects={obj.name: obj for obj in objects}),
    )
    monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
    monkeypatch.setattr(blender_runtime, "bpy", fake_bpy)
    return blender_runtime.BlenderRuntimeSession(
        path_policy=PathPolicy(enabled=True, allowed_paths=[])
    )


def channel_summary(result: Dict[str, Any]) -> List[Tuple[str, int, int, List[int]]]:
    """get_keyframes' channels as (data_path, index, count, frame_range) tuples."""
    return [
        (c["data_path"], c["array_index"], c["keyframe_count"], c["frame_range"])
        for c in result["channels"]
    ]


def empty_page(result: Dict[str, Any]) -> Tuple[str, bool]:
    """The object and has_animation of a reply that lists no channel, checked to be the last page."""
    assert result["channels"] == [] and result["total_channels"] == 0
    assert result["next_offset"] is None and result["truncated"] is False
    return result["object_name"], result["has_animation"]


# -- the doubles fail where Blender fails --------------------------------------


class TestDoubles:
    """Without this, a test could pass on an attribute Blender no longer has."""

    def test_blender_5_action_has_no_legacy_fcurves(self) -> None:
        action = Blender50.Action([], [])
        # What get_blender_keyframes raised on Blender 5.0.1.
        with pytest.raises(
            AttributeError, match="'Action' object has no attribute 'fcurves'"
        ):
            action.fcurves
        with pytest.raises(AttributeError):
            action.groups
        assert not hasattr(action, "channelbags")

    def test_blender_4_2_has_no_slots(self) -> None:
        with pytest.raises(
            AttributeError, match="'AnimData' object has no attribute 'action_slot'"
        ):
            Blender42.AnimData(None).action_slot
        with pytest.raises(AttributeError):
            Blender42.Action([]).layers
        assert "action_get_channelbag_for_slot" not in ANIM_UTILS[BLENDER_4_2]

    def test_a_missing_slot_reaches_the_strip_only_on_4_4_0(self) -> None:
        with pytest.raises(RuntimeError, match="slot is None"):
            ActionKeyframeStrip([]).channelbag(None)
        action = slotted_action(BLENDER_4_4, {})
        with pytest.raises(RuntimeError, match="slot is None"):
            channelbag_for_slot_4_4_0(action, None)
        assert channelbag_for_slot(action, None) is None

    def test_keyframe_functions_take_the_documented_arguments(self) -> None:
        """Blender 5.0 documents ``keyframe_insert(data_path, /, *, index, frame, ...)``.

        5.0.1 still accepts ``data_path=``; the double holds callers to the
        documented form.
        """
        ball = keyed_ball(BLENDER_5_0, FCurve("location", 0, 1))
        with pytest.raises(TypeError):
            ball.keyframe_insert(data_path="location", frame=1)
        with pytest.raises(TypeError):
            ball.keyframe_delete("location", 0, 1)


# -- action_fcurves ------------------------------------------------------------


class TestActionFcurves:
    """The one place that knows where each Blender version keeps F-curves."""

    @every_version
    def test_nothing_animates_an_id_without_an_action(
        self, monkeypatch: pytest.MonkeyPatch, version: Version
    ) -> None:
        session_on(monkeypatch, version)
        no_action = (
            Blender42.AnimData(None)
            if version < BLENDER_4_4
            else Blender50.AnimData(None, None)
        )
        assert blender_runtime.action_fcurves(None, version) is None
        assert blender_runtime.action_fcurves(no_action, version) is None

    def test_before_4_4_every_fcurve_of_the_action(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session_on(monkeypatch, BLENDER_4_2)
        curves = [FCurve("location", 0, 1), FCurve("scale", 2, 5)]
        anim_data = Blender42.AnimData(Blender42.Action(curves))
        assert blender_runtime.action_fcurves(anim_data, BLENDER_4_2) == curves

    @slotted_versions
    def test_the_fcurves_of_the_ids_own_slot(
        self, monkeypatch: pytest.MonkeyPatch, version: Version
    ) -> None:
        session_on(monkeypatch, version)
        ball, other = ActionSlot("OBBall"), ActionSlot("OBOther")
        ball_curves = [FCurve("location", 2, 1, 10)]
        other_curves = [FCurve("scale", 0, 5)]
        shared = slotted_action(version, {ball: ball_curves, other: other_curves})
        for slot, curves in ((ball, ball_curves), (other, other_curves)):
            anim_data = bpy_types(version).AnimData(shared, slot)
            assert blender_runtime.action_fcurves(anim_data, version) == curves

    def test_4_4_does_not_read_the_first_slot_for_every_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """4.4 and 4.5 still have Action.fcurves, but it is another ID's curves."""
        session_on(monkeypatch, BLENDER_4_4)
        ball, other = ActionSlot("OBBall"), ActionSlot("OBOther")
        ball_curves = [FCurve("location", 0, 1)]
        other_curves = [FCurve("scale", 0, 5)]
        shared = slotted_action(BLENDER_4_4, {ball: ball_curves, other: other_curves})
        other_fcurves = blender_runtime.action_fcurves(
            Blender44.AnimData(shared, other), BLENDER_4_4
        )
        assert shared.fcurves == ball_curves
        assert other_fcurves == other_curves

    @slotted_versions
    def test_an_action_without_the_ids_slot_animates_nothing(
        self, monkeypatch: pytest.MonkeyPatch, version: Version
    ) -> None:
        """Assigning an action assigns no slot, which 4.4.0 must never look up."""
        session_on(monkeypatch, version)
        shared = slotted_action(
            version, {ActionSlot("OBOther"): [FCurve("location", 0, 1)]}
        )
        anim_data = bpy_types(version).AnimData(shared, None)
        assert blender_runtime.action_fcurves(anim_data, version) is None

    @slotted_versions
    def test_a_slot_without_a_channelbag_has_no_fcurves(
        self, monkeypatch: pytest.MonkeyPatch, version: Version
    ) -> None:
        session_on(monkeypatch, version)
        slot = ActionSlot("OBBall")
        empty = bpy_types(version).Action([], [slot])
        anim_data = bpy_types(version).AnimData(empty, slot)
        assert blender_runtime.action_fcurves(anim_data, version) == []


# -- the keyframe tools ----------------------------------------------------------


class TestGetKeyframes:
    @every_version
    def test_summarizes_the_keyed_channels(
        self, monkeypatch: pytest.MonkeyPatch, version: Version
    ) -> None:
        ball = keyed_ball(
            version,
            FCurve("location", 0, 1),
            FCurve("location", 2, 1, 10),
            FCurve("rotation_euler", 1),
        )
        session = session_on(monkeypatch, version, ball)

        result = session.get_keyframes("Ball")

        assert result["object_name"] == "Ball"
        assert result["has_animation"] is True
        # A curve without keys is left out.
        assert channel_summary(result) == [
            ("location", 0, 1, [1, 1]),
            ("location", 2, 2, [1, 10]),
        ]

    def test_keys_between_frames_keep_their_frame(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Keys at -0.5 and 120.9 were reported as frame_range [0, 120]."""
        ball = keyed_ball(BLENDER_5_0, FCurve("location", 0, 1, -0.5, 120.9))
        session = session_on(monkeypatch, BLENDER_5_0, ball)

        (channel,) = session.get_keyframes("Ball")["channels"]

        assert channel["frame_range"] == [-0.5, 120.9]
        assert all(isinstance(frame, float) for frame in channel["frame_range"])

    @slotted_versions
    def test_lists_only_the_objects_slot_of_a_shared_action(
        self, monkeypatch: pytest.MonkeyPatch, version: Version
    ) -> None:
        ball_slot, other_slot = ActionSlot("OBBall"), ActionSlot("OBOther")
        shared = slotted_action(
            version,
            {
                ball_slot: [FCurve("location", 0, 1)],
                other_slot: [FCurve("scale", 1, 5, 6)],
            },
        )
        animation = bpy_types(version).AnimData
        ball = Object("Ball", animation(shared, ball_slot))
        other = Object("Other", animation(shared, other_slot))
        session = session_on(monkeypatch, version, ball, other)

        assert channel_summary(session.get_keyframes("Other")) == [
            ("scale", 1, 2, [5, 6])
        ]
        assert channel_summary(session.get_keyframes("Ball")) == [
            ("location", 0, 1, [1, 1])
        ]

    @every_version
    def test_object_without_animation(
        self, monkeypatch: pytest.MonkeyPatch, version: Version
    ) -> None:
        session = session_on(monkeypatch, version, Object("Ball"))
        assert empty_page(session.get_keyframes("Ball")) == ("Ball", False)

    @slotted_versions
    def test_action_without_the_objects_slot(
        self, monkeypatch: pytest.MonkeyPatch, version: Version
    ) -> None:
        shared = slotted_action(
            version, {ActionSlot("OBOther"): [FCurve("location", 0, 1)]}
        )
        ball = Object("Ball", bpy_types(version).AnimData(shared, None))
        session = session_on(monkeypatch, version, ball)
        assert empty_page(session.get_keyframes("Ball")) == ("Ball", False)

    @every_version
    def test_an_action_whose_keys_were_all_deleted(
        self, monkeypatch: pytest.MonkeyPatch, version: Version
    ) -> None:
        session = session_on(monkeypatch, version, keyed_ball(version))
        assert empty_page(session.get_keyframes("Ball")) == ("Ball", True)

    def test_object_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        session = session_on(monkeypatch, BLENDER_5_0)
        with pytest.raises(ValueError, match="Object not found"):
            session.get_keyframes("Nonexistent")


class TestInsertKeyframe:
    @every_version
    def test_keys_the_current_value_and_reads_the_key_back(
        self, monkeypatch: pytest.MonkeyPatch, version: Version
    ) -> None:
        ball = keyed_ball(version)
        ball.location = [1.0, 2.0, 3.0]
        session = session_on(monkeypatch, version, ball)

        result = session.insert_keyframe("Ball", "location", 10, index=2)

        assert result == {
            "object_name": "Ball",
            "data_path": "location",
            "frame": 10,
            "value_source": "current",
            "keyframes": [
                {
                    "array_index": 2,
                    "value": 3.0,
                    "interpolation": "BEZIER",
                    "keyframe_type": "KEYFRAME",
                }
            ],
        }
        assert ball.calls == [("keyframe_insert", "location", 2, 10, "KEYFRAME")]

    @every_version
    def test_keys_a_given_value_and_shows_what_the_animation_gives(
        self, monkeypatch: pytest.MonkeyPatch, version: Version
    ) -> None:
        """Keying a value at another frame keyed the current one: there was no value.

        A GUI Blender 5.0.1 showed a first key at once and a later one only
        after a frame change, background Blender neither; the call writes what
        the keyed channels' F-curves give either way.
        """
        ball = keyed_ball(version)
        ball.location = [1.0, 2.0, 3.0]
        session = session_on(monkeypatch, version, ball)

        result = session.insert_keyframe("Ball", "location", 10, value=[4.0, 5.0, 6.0])

        assert result["value_source"] == "given"
        assert [(k["array_index"], k["value"]) for k in result["keyframes"]] == [
            (0, 4.0),
            (1, 5.0),
            (2, 6.0),
        ]
        assert channel_summary(session.get_keyframes("Ball")) == [
            ("location", 0, 1, [10, 10]),
            ("location", 1, 1, [10, 10]),
            ("location", 2, 1, [10, 10]),
        ]
        # Each channel has the one key, so its animation holds that value on frame 1.
        assert ball.location == [4.0, 5.0, 6.0]
        assert ball.tags == []

    def test_a_key_between_keys_leaves_the_animations_value_on_the_current_frame(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Blender showed the restored value until the next frame change, not its animation."""
        curve = FCurve("location", 0, 1, 30)
        ball = keyed_ball(BLENDER_5_0, curve)
        ball.location = [7.0, 0.0, 0.0]
        session = session_on(monkeypatch, BLENDER_5_0, ball)

        session.insert_keyframe("Ball", "location", 20, index=0, value=10.0)

        assert [k.co for k in curve.keyframe_points] == [
            [1, 0.0],
            [20, 10.0],
            [30, 0.0],
        ]
        # Frame 1 is on the first key, 0.
        assert ball.location == [0.0, 0.0, 0.0]

    def test_keying_the_current_value_leaves_evaluation_to_blender(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without value it is Blender's I key: nothing more is evaluated."""
        ball = keyed_ball(BLENDER_5_0, FCurve("location", 0, 1, 30))
        ball.location = [7.0, 0.0, 0.0]
        session = session_on(monkeypatch, BLENDER_5_0, ball)

        session.insert_keyframe("Ball", "location", 20, index=0)

        assert ball.location == [7.0, 0.0, 0.0]
        assert ball.tags == []

    def test_a_given_value_on_the_current_frame_is_what_the_property_shows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ball = keyed_ball(BLENDER_5_0)
        session = session_on(monkeypatch, BLENDER_5_0, ball)
        blender_runtime.bpy.context.scene.frame_current = 10

        session.insert_keyframe("Ball", "location", 10, value=[4.0, 5.0, 6.0])

        assert ball.location == [4.0, 5.0, 6.0]

    @every_version
    def test_a_given_value_leaves_unkeyed_edits_on_other_channels(
        self, monkeypatch: pytest.MonkeyPatch, version: Version
    ) -> None:
        """Blender's own keyframe_insert kept an unkeyed edit on another animated
        channel; evaluating the object's animation put every channel back."""
        curves = [curve("location", i, (1, 0.0), (10, 10.0)) for i in range(3)]
        gear = curve('["gear"]', 0, (1, 1.0), (10, 5.0))
        ball = keyed_ball(version, *curves, gear)
        # Unkeyed edits at frame 1, where every curve gives its first key.
        ball.location = [5.0, 7.0, 2.0]
        ball.props["gear"] = 3
        session = session_on(monkeypatch, version, ball)

        result = session.insert_keyframe("Ball", "location", 5, index=2, value=1.0)

        assert "warning" not in result
        assert [k.co for k in curves[2].keyframe_points] == [
            [1, 0.0],
            [5, 1.0],
            [10, 10.0],
        ]
        # z shows its curve at frame 1; x, y and gear keep their unkeyed edits.
        assert ball.location == [5.0, 7.0, 0.0]
        assert ball.props["gear"] == 3

    def test_keying_one_bone_leaves_another_bones_unkeyed_pose(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        b0 = SimpleNamespace(rotation_euler=[0.5, 0.0, 0.0])
        b1 = SimpleNamespace(rotation_euler=[0.4, 0.25, 0.0])
        curves = [
            curve(f'pose.bones["{bone}"].rotation_euler', i, (1, 0.0), (10, 1.0))
            for bone in ("b0", "b1")
            for i in range(3)
        ]
        rig = keyed_ball(BLENDER_5_0, *curves)
        rig.paths['pose.bones["b0"]'] = b0
        rig.paths['pose.bones["b1"]'] = b1
        session = session_on(monkeypatch, BLENDER_5_0, rig)

        session.insert_keyframe(
            "Ball", 'pose.bones["b1"].rotation_euler', 5, index=0, value=0.75
        )

        assert b0.rotation_euler == [0.5, 0.0, 0.0]
        # b1's x shows its curve at frame 1; its y keeps the unkeyed edit.
        assert b1.rotation_euler == [0.0, 0.25, 0.0]

    def test_shows_the_curve_at_the_scenes_time(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The frame, its subframe and the render's time remapping, as Blender
        evaluates animation: frame 5.5 under old 100 and new 200 is time 2.75."""
        x = curve("location", 0, (1, 0.0), (11, 10.0))
        ball = keyed_ball(BLENDER_5_0, x)
        session = session_on(monkeypatch, BLENDER_5_0, ball)
        scene = blender_runtime.bpy.context.scene
        scene.frame_current, scene.frame_subframe = 5, 0.5
        scene.render.frame_map_new = 200

        session.insert_keyframe("Ball", "location", 20, index=0, value=3.0)

        assert ball.location[0] == pytest.approx(1.75)

    def test_shows_the_curve_with_the_interpolation_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The interpolation shapes the curve after the new key: frame 25 holds
        the Constant key at 20, where a straight line to frame 30 gave 5."""
        x = curve("location", 0, (1, 0.0), (30, 0.0))
        ball = keyed_ball(BLENDER_5_0, x)
        session = session_on(monkeypatch, BLENDER_5_0, ball)
        blender_runtime.bpy.context.scene.frame_current = 25

        session.insert_keyframe(
            "Ball", "location", 20, index=0, value=10.0, interpolation="CONSTANT"
        )

        assert ball.location[0] == 10.0

    def test_an_integer_property_shows_a_whole_number(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ball = keyed_ball(BLENDER_5_0, curve('["gear"]', 0, (1, 1.0), (10, 8.0)))
        ball.props["gear"] = 1
        session = session_on(monkeypatch, BLENDER_5_0, ball)
        blender_runtime.bpy.context.scene.frame_current = 4

        session.insert_keyframe("Ball", '["gear"]', 20, value=2.0)

        # The double's straight line gives 3.33 at frame 4; Blender rounds an
        # integer property's curve.
        assert ball.props["gear"] == 3 and isinstance(ball.props["gear"], int)

    def test_nla_tracks_leave_the_keyed_value_until_a_frame_change(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Blender blends NLA tracks with the action on a frame change; the
        F-curve alone is not what the channel would show."""
        x = curve("location", 0, (1, 0.0), (10, 10.0))
        ball = keyed_ball(BLENDER_5_0, x)
        ball.location = [6.0, 0.0, 0.0]
        ball.animation_data.nla_tracks = [SimpleNamespace(name="NlaTrack")]
        session = session_on(monkeypatch, BLENDER_5_0, ball)

        result = session.insert_keyframe("Ball", "location", 5, index=0, value=1.0)

        assert ball.location == [6.0, 0.0, 0.0]
        assert "'Ball' has NLA tracks" in result["warning"]

        ball.animation_data.use_nla = False
        result = session.insert_keyframe("Ball", "location", 6, index=0, value=2.0)

        # With the NLA off Blender evaluates the action alone.
        assert ball.location == [0.0, 0.0, 0.0]
        assert "warning" not in result

    def test_a_muted_or_driven_channel_keeps_its_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Blender's animation writes neither: a driver sets the one, nothing the other."""
        curves = [curve("location", i, (1, 0.0), (10, 10.0)) for i in range(3)]
        curves[0].mute = True
        curves[1].group = SimpleNamespace(mute=True)
        ball = keyed_ball(BLENDER_5_0, *curves)
        ball.animation_data.drivers = [
            SimpleNamespace(data_path="location", array_index=2)
        ]
        ball.location = [4.0, 5.0, 6.0]
        session = session_on(monkeypatch, BLENDER_5_0, ball)

        result = session.insert_keyframe("Ball", "location", 5, value=[1.0, 1.0, 1.0])

        assert ball.location == [4.0, 5.0, 6.0]
        assert "warning" not in result

    def test_finds_a_key_blender_merged_into_a_nearby_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Blender 5.0.1 replaced a key 0.005 frames away and kept its frame."""
        x = curve("location", 0, (1, 0.0), (4.995, 0.0))
        ball = keyed_ball(BLENDER_5_0, x)
        session = session_on(monkeypatch, BLENDER_5_0, ball)

        result = session.insert_keyframe(
            "Ball", "location", 5, index=0, value=2.0, interpolation="LINEAR"
        )

        assert [k.co for k in x.keyframe_points] == [[1, 0.0], [4.995, 2.0]]
        assert result["keyframes"] == [
            {
                "array_index": 0,
                "value": 2.0,
                "interpolation": "LINEAR",
                "keyframe_type": "KEYFRAME",
            }
        ]
        assert "warning" not in result

    def test_one_channel_takes_one_number(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ball = keyed_ball(BLENDER_5_0)
        ball.location = [1.0, 2.0, 3.0]
        session = session_on(monkeypatch, BLENDER_5_0, ball)

        result = session.insert_keyframe("Ball", "location", 7, index=1, value=-2.5)

        assert [(k["array_index"], k["value"]) for k in result["keyframes"]] == [
            (1, -2.5)
        ]
        assert ball.location == [1.0, -2.5, 3.0]

    def test_sets_interpolation_and_key_type(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every key came out Bezier, KEYFRAME, whatever the caller needed."""
        ball = keyed_ball(BLENDER_5_0)
        session = session_on(monkeypatch, BLENDER_5_0, ball)

        result = session.insert_keyframe(
            "Ball",
            "location",
            3,
            index=0,
            interpolation="LINEAR",
            keyframe_type="BREAKDOWN",
        )

        assert result["keyframes"] == [
            {
                "array_index": 0,
                "value": 0.0,
                "interpolation": "LINEAR",
                "keyframe_type": "BREAKDOWN",
            }
        ]
        assert ball.calls == [("keyframe_insert", "location", 0, 3, "BREAKDOWN")]
        assert ball.fcurves[0].keyframe_points[0].interpolation == "LINEAR"

    @pytest.mark.parametrize(
        ("index", "value", "message"),
        [
            (-1, [1.0, 2.0], "has 3 elements: pass 3 numbers"),
            (-1, 1.0, "has 3 elements: pass 3 numbers"),
            (1, [1.0], r"location\[1\] is one channel"),
            (5, 1.0, "index 5 is out of range"),
        ],
        ids=["short-list", "number-for-array", "list-for-channel", "bad-index"],
    )
    def test_refuses_a_value_that_does_not_fit(
        self,
        monkeypatch: pytest.MonkeyPatch,
        index: int,
        value: Any,
        message: str,
    ) -> None:
        ball = keyed_ball(BLENDER_5_0)
        ball.location = [1.0, 2.0, 3.0]
        session = session_on(monkeypatch, BLENDER_5_0, ball)

        with pytest.raises(ValueError, match=message):
            session.insert_keyframe("Ball", "location", 4, index=index, value=value)

        assert ball.calls == []
        assert ball.location == [1.0, 2.0, 3.0]

    def test_custom_and_pose_bone_properties(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ball = keyed_ball(BLENDER_5_0)
        ball.props["gear"] = 2
        bone = SimpleNamespace(rotation_euler=[0.0, 0.0, 0.0])
        ball.paths['pose.bones["J1"]'] = bone
        session = session_on(monkeypatch, BLENDER_5_0, ball)

        custom = session.insert_keyframe("Ball", '["gear"]', 5, value=3.0)
        posed = session.insert_keyframe(
            "Ball", 'pose.bones["J1"].rotation_euler', 5, value=[0.1, 0.2, 0.3]
        )

        assert [k["value"] for k in custom["keyframes"]] == [3.0]
        assert [k["value"] for k in posed["keyframes"]] == [0.1, 0.2, 0.3]
        # Their only keys: the animation holds those values on the current frame.
        assert ball.props["gear"] == 3
        assert bone.rotation_euler == [0.1, 0.2, 0.3]
        with pytest.raises(ValueError, match="not a whole number"):
            session.insert_keyframe("Ball", '["gear"]', 6, value=2.5)
        with pytest.raises(ValueError, match="could not be resolved"):
            session.insert_keyframe(
                "Ball", 'pose.bones["J9"].rotation_euler', 6, value=[0, 0, 0]
            )
        with pytest.raises(ValueError, match="no custom property 'mass'"):
            session.insert_keyframe("Ball", '["mass"]', 6, value=1.0)

    def test_a_refused_key_restores_the_given_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ball = keyed_ball(BLENDER_5_0, locked=True)
        ball.location = [1.0, 2.0, 3.0]
        session = session_on(monkeypatch, BLENDER_5_0, ball)
        blender_runtime.bpy.context.scene.frame_current = 4

        with pytest.raises(RuntimeError, match="inserted no keyframe"):
            session.insert_keyframe("Ball", "location", 4, value=[7.0, 8.0, 9.0])

        assert ball.location == [1.0, 2.0, 3.0]

    def test_refuses_an_unknown_interpolation_or_key_type(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ball = keyed_ball(BLENDER_5_0)
        session = session_on(monkeypatch, BLENDER_5_0, ball)
        with pytest.raises(ValueError, match="Invalid interpolation 'SMOOTH'"):
            session.insert_keyframe("Ball", "location", 1, interpolation="SMOOTH")
        with pytest.raises(ValueError, match="Invalid keyframe_type 'KEY'"):
            session.insert_keyframe("Ball", "location", 1, keyframe_type="KEY")
        assert ball.calls == []

    def test_says_so_when_the_key_is_not_on_the_frame(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """In NLA tweak mode the key lands at the strip's own frame."""
        ball = Object("Ball")
        session = session_on(monkeypatch, BLENDER_5_0, ball)

        result = session.insert_keyframe(
            "Ball", "location", 10, interpolation="CONSTANT"
        )

        assert result["keyframes"] == []
        assert "interpolation was left as Blender set it" in result["warning"]

    def test_reports_a_key_blender_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """keyframe_insert returns False instead of raising, for a locked F-curve
        or an NLA setup that cannot take the key; the error names both causes.

        It named locked or sampled F-curves only. Blender 5.0.1 also refused a
        key "due to the NLA stack setup": on an action at zero influence over
        an NLA strip, or blended by Multiply over a strip that holds 0.
        """
        ball = keyed_ball(BLENDER_5_0, FCurve("location", 0, 1), locked=True)
        session = session_on(monkeypatch, BLENDER_5_0, ball)
        with pytest.raises(
            RuntimeError,
            match=r"inserted no keyframe on 'Ball' location at frame 5; .* locked",
        ) as refused:
            session.insert_keyframe("Ball", "location", 5)
        message = str(refused.value)
        assert "sampled" in message and "F-modifier" in message
        assert "NLA setup" in message and "zero influence" in message

    def test_object_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        session = session_on(monkeypatch, BLENDER_5_0)
        with pytest.raises(ValueError, match="Object not found"):
            session.insert_keyframe("Nonexistent", "location", 1)


class TestDeleteKeyframe:
    @every_version
    def test_round_trip_with_get_keyframes(
        self, monkeypatch: pytest.MonkeyPatch, version: Version
    ) -> None:
        ball = keyed_ball(
            version,
            FCurve("location", 0, 1),
            FCurve("location", 1, 1),
            FCurve("location", 2, 1, 10),
        )
        session = session_on(monkeypatch, version, ball)

        deleted = session.delete_keyframe("Ball", "location", 10, index=2)

        assert deleted == {"object_name": "Ball", "data_path": "location", "frame": 10}
        assert ball.calls == [("keyframe_delete", "location", 2, 10)]
        assert channel_summary(session.get_keyframes("Ball")) == [
            ("location", 0, 1, [1, 1]),
            ("location", 1, 1, [1, 1]),
            ("location", 2, 1, [1, 1]),
        ]
        # Nothing is left at frame 10: the tool says so instead of succeeding.
        with pytest.raises(
            ValueError, match=r"'Ball' has no keyframe on location at frame 10"
        ):
            session.delete_keyframe("Ball", "location", 10)
        session.delete_keyframe("Ball", "location", 1)
        assert session.get_keyframes("Ball")["channels"] == []

    @every_version
    def test_refuses_an_object_nothing_animates(
        self, monkeypatch: pytest.MonkeyPatch, version: Version
    ) -> None:
        ball = Object("Ball")
        session = session_on(monkeypatch, version, ball)
        with pytest.raises(ValueError, match="'Ball' is not animated"):
            session.delete_keyframe("Ball", "location", 1)
        assert ball.calls == []

    @slotted_versions
    def test_refuses_an_action_without_the_objects_slot(
        self, monkeypatch: pytest.MonkeyPatch, version: Version
    ) -> None:
        shared = slotted_action(
            version, {ActionSlot("OBOther"): [FCurve("location", 0, 1)]}
        )
        ball = Object("Ball", bpy_types(version).AnimData(shared, None))
        session = session_on(monkeypatch, version, ball)
        with pytest.raises(ValueError, match="'Ball' is not animated"):
            session.delete_keyframe("Ball", "location", 1)
        assert ball.calls == []

    def test_reports_a_locked_fcurve(self, monkeypatch: pytest.MonkeyPatch) -> None:
        curve = FCurve("location", 2, 1)
        curve.lock = True
        ball = keyed_ball(BLENDER_5_0, curve, locked=True)
        session = session_on(monkeypatch, BLENDER_5_0, ball)
        with pytest.raises(
            blender_runtime.ValidationError,
            match=r"location\[2\] at frame 1 is on a locked F-curve",
        ) as refused:
            session.delete_keyframe("Ball", "location", 1, index=2)
        assert refused.value.details["code"] == "KeyframeLocked"
        assert channel_summary(session.get_keyframes("Ball")) == [
            ("location", 2, 1, [1, 1])
        ]


# -- paging, filters and readback (B3-13) ------------------------------------------


def rig(version: Version, bones: int, keys: Tuple[float, ...] = (1, 50, 100)) -> Object:
    """An armature 'Rig' keyed on every bone's location, quaternion and scale: 10 channels a bone."""
    curves = [
        FCurve(f'pose.bones["B{bone:03d}"].{path}', index, *keys)
        for bone in range(bones)
        for path, size in (("location", 3), ("rotation_quaternion", 4), ("scale", 3))
        for index in range(size)
    ]
    armature = keyed_ball(version, *curves)
    armature.name = "Rig"
    armature.pose = SimpleNamespace(
        bones={
            f"B{bone:03d}": SimpleNamespace(name=f"B{bone:03d}")
            for bone in range(bones)
        }
    )
    return armature


def every_page(session: Any, name: str, **filters: Any) -> List[Dict[str, Any]]:
    """Every page of get_keyframes, following next_offset and next_key_offset."""
    pages, offset, key_offset = [], 0, 0
    while True:
        page = session.get_keyframes(
            name, offset=offset, key_offset=key_offset, **filters
        )
        pages.append(page)
        if page["next_offset"] is None:
            return pages
        offset, key_offset = page["next_offset"], page["next_key_offset"]


class TestKeyframePages:
    def test_pages_join_into_every_channel_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An 80-bone rig returned 301 of 800 channels, with a hint naming root_path and max_prims."""
        armature = rig(BLENDER_5_0, 80)
        session = session_on(monkeypatch, BLENDER_5_0, armature)

        pages = every_page(session, "Rig")

        listed = [
            (c["data_path"], c["array_index"]) for p in pages for c in p["channels"]
        ]
        assert listed == [(c.data_path, c.array_index) for c in armature.fcurves]
        assert len(pages) > 1
        assert all(p["total_channels"] == 800 for p in pages)
        assert all(
            len(json.dumps(p)) <= blender_runtime.RESULT_PAGE_BYTES for p in pages
        )
        first = pages[0]
        assert first["truncated"] is True
        assert first["truncation"]["returned"] == len(first["channels"])
        assert first["truncation"]["total"] == 800
        hint = first["truncation"]["hint"]
        assert f"offset={first['next_offset']}" in hint
        assert "root_path" not in hint and "max_prims" not in hint
        last = pages[-1]
        assert (last["truncated"], last["truncation"], last["next_key_offset"]) == (
            False,
            None,
            None,
        )

    def test_filters_by_bone_and_data_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session = session_on(monkeypatch, BLENDER_5_0, rig(BLENDER_5_0, 3))

        bone = session.get_keyframes("Rig", bone_name="B001")
        quaternion = session.get_keyframes(
            "Rig", bone_name="B001", data_path="rotation_quaternion"
        )
        exact = session.get_keyframes("Rig", data_path='pose.bones["B002"].scale')

        assert bone["total_channels"] == 10
        assert {c["bone_name"] for c in bone["channels"]} == {"B001"}
        assert [c["array_index"] for c in quaternion["channels"]] == [0, 1, 2, 3]
        assert {c["data_path"] for c in quaternion["channels"]} == {
            'pose.bones["B001"].rotation_quaternion'
        }
        assert [c["data_path"] for c in exact["channels"]] == [
            'pose.bones["B002"].scale'
        ] * 3
        with pytest.raises(ValueError, match="Armature 'Rig' has no bone 'B999'"):
            session.get_keyframes("Rig", bone_name="B999")

    def test_a_bone_filter_needs_an_armature(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session = session_on(monkeypatch, BLENDER_5_0, keyed_ball(BLENDER_5_0))
        with pytest.raises(ValueError, match="'Ball' is not an armature"):
            session.get_keyframes("Ball", bone_name="Arm")

    def test_limit_and_offset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        session = session_on(monkeypatch, BLENDER_5_0, rig(BLENDER_5_0, 2))

        page = session.get_keyframes("Rig", offset=4, limit=3)

        assert [c["data_path"] for c in page["channels"]] == [
            'pose.bones["B000"].rotation_quaternion'
        ] * 3
        assert (page["offset"], page["next_offset"], page["next_key_offset"]) == (
            4,
            7,
            0,
        )
        for bad in ({"offset": -1}, {"limit": 0}, {"key_offset": 2}):
            with pytest.raises(ValueError):
                session.get_keyframes("Rig", **bad)

    def test_reads_every_key_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Keys had no value, interpolation or handles to read back."""
        curve = FCurve("location", 0, 1, 10, 20)
        bezier, linear, eased = curve.keyframe_points
        bezier.co, bezier.handle_left, bezier.handle_right = (
            [1, 0.1],
            [0, 0.1],
            [4, 0.5],
        )
        linear.interpolation, linear.co = "LINEAR", [10, 2.5]
        eased.interpolation, eased.easing, eased.type = "SINE", "EASE_IN", "BREAKDOWN"
        session = session_on(monkeypatch, BLENDER_5_0, keyed_ball(BLENDER_5_0, curve))

        (channel,) = session.get_keyframes("Ball", include_keys=True)["channels"]

        assert channel["key_offset"] == 0
        assert channel["keys"] == [
            {
                "frame": 1.0,
                "value": 0.1,
                "interpolation": "BEZIER",
                "keyframe_type": "KEYFRAME",
                "handle_left": [0.0, 0.1],
                "handle_right": [4.0, 0.5],
                "handle_left_type": "AUTO_CLAMPED",
                "handle_right_type": "AUTO_CLAMPED",
            },
            {
                "frame": 10.0,
                "value": 2.5,
                "interpolation": "LINEAR",
                "keyframe_type": "KEYFRAME",
            },
            {
                "frame": 20.0,
                "value": 0.0,
                "interpolation": "SINE",
                "keyframe_type": "BREAKDOWN",
                "easing": "EASE_IN",
            },
        ]

    def test_key_offset_pages_through_one_long_channel(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        frames = tuple(float(f) for f in range(3000))
        session = session_on(
            monkeypatch,
            BLENDER_5_0,
            keyed_ball(
                BLENDER_5_0, FCurve("location", 0, *frames), FCurve("scale", 1, 5)
            ),
        )

        pages = every_page(session, "Ball", include_keys=True)

        location = [
            (c["key_offset"], [k["frame"] for k in c["keys"]])
            for p in pages
            for c in p["channels"]
            if c["data_path"] == "location"
        ]
        assert [f for _, keys in location for f in keys] == list(frames)
        assert [offset for offset, _ in location][0] == 0
        assert all(
            later == earlier + len(keys)
            for (earlier, keys), (later, _) in zip(location, location[1:])
        )
        assert pages[-1]["channels"][-1]["data_path"] == "scale"
        assert all(
            len(json.dumps(p)) <= blender_runtime.RESULT_PAGE_BYTES for p in pages
        )

    def test_float32_values_are_reported_in_their_shortest_form(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """0.1 as Blender stores it is 0.10000000149011612 in Python."""
        curve = FCurve("location", 0, 1)
        curve.keyframe_points[0].co = [120.9000015258789, 0.10000000149011612]
        session = session_on(monkeypatch, BLENDER_5_0, keyed_ball(BLENDER_5_0, curve))

        (channel,) = session.get_keyframes("Ball", include_keys=True)["channels"]

        assert channel["frame_range"] == [120.9, 120.9]
        assert (channel["keys"][0]["frame"], channel["keys"][0]["value"]) == (
            120.9,
            0.1,
        )


class TestInsertKeyframes:
    def test_keys_every_frame_of_every_channel(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 6-DOF arm over 1 000 frames took some 13 000 calls of one key each."""
        ball = keyed_ball(BLENDER_5_0)
        ball.location = [9.0, 9.0, 9.0]
        ball.props["gear"] = 1
        session = session_on(monkeypatch, BLENDER_5_0, ball)

        result = session.insert_keyframes(
            "Ball",
            [
                {
                    "data_path": "location",
                    "frames": [1, 2, 3],
                    "values": [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
                },
                {"data_path": '["gear"]', "frames": [5], "values": [3]},
            ],
            interpolation="LINEAR",
            keyframe_type="BREAKDOWN",
        )

        assert result["key_count"] == 10
        assert result["frame_range"] == [1.0, 5.0]
        assert [
            (c["data_path"], c["array_index"], c["inserted"])
            for c in result["channels"]
        ] == [
            ("location", 0, 3),
            ("location", 1, 3),
            ("location", 2, 3),
            ('["gear"]', 0, 1),
        ]
        keys = {(c.data_path, c.array_index): c.keyframe_points for c in ball.fcurves}
        assert [k.co for k in keys[("location", 1)]] == [[1, 0.0], [2, 1.0], [3, 2.0]]
        assert {k.interpolation for points in keys.values() for k in points} == {
            "LINEAR"
        }
        assert {k.type for points in keys.values() for k in points} == {"BREAKDOWN"}
        # Each keyed channel shows its curve: frame 1's keys, and gear's only key.
        assert ball.location == [0.0, 0.0, 0.0]
        assert ball.props["gear"] == 3
        assert "warning" not in result

    def test_keeps_unkeyed_edits_on_the_channels_it_does_not_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Re-evaluating the object's animation put every animated channel back."""
        curves = [curve("location", i, (1, 0.0), (10, 10.0)) for i in range(3)]
        bones = [
            curve(f'pose.bones["{bone}"].rotation_euler', i, (1, 0.0), (10, 1.0))
            for bone in ("b0", "b1")
            for i in range(3)
        ]
        ball = keyed_ball(BLENDER_5_0, *curves, *bones, curve('["gear"]', 0, (1, 1.0)))
        b0 = SimpleNamespace(rotation_euler=[0.5, 0.25, 0.0])
        b1 = SimpleNamespace(rotation_euler=[0.4, 0.125, 0.0])
        ball.paths['pose.bones["b0"]'] = b0
        ball.paths['pose.bones["b1"]'] = b1
        ball.location = [5.0, 7.0, 2.0]
        ball.props["gear"] = 3
        session = session_on(monkeypatch, BLENDER_5_0, ball)

        result = session.insert_keyframes(
            "Ball",
            [
                {"data_path": "location", "index": 2, "frames": [5], "values": [1.0]},
                {
                    "data_path": 'pose.bones["b1"].rotation_euler',
                    "index": 0,
                    "frames": [5, 7],
                    "values": [0.75, 0.5],
                },
            ],
        )

        assert result["key_count"] == 3 and "warning" not in result
        # The keyed channels show their curves at frame 1; the rest keep their edits.
        assert ball.location == [5.0, 7.0, 0.0]
        assert b0.rotation_euler == [0.5, 0.25, 0.0]
        assert b1.rotation_euler == [0.0, 0.125, 0.0]
        assert ball.props["gear"] == 3

    def test_nla_tracks_leave_the_keyed_values_until_a_frame_change(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ball = keyed_ball(BLENDER_5_0, curve("location", 0, (1, 0.0), (10, 10.0)))
        ball.animation_data.nla_tracks = [SimpleNamespace(name="NlaTrack")]
        ball.location = [6.0, 0.0, 0.0]
        session = session_on(monkeypatch, BLENDER_5_0, ball)

        result = session.insert_keyframes(
            "Ball", [{"data_path": "location", "index": 0, "frames": [5], "values": [1]}]
        )

        assert ball.location == [6.0, 0.0, 0.0]
        assert "'Ball' has NLA tracks" in result["warning"]

    def test_interpolation_reaches_a_key_merged_into_a_nearby_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Blender 5.0.1 replaced a key 0.005 frames away and kept its frame."""
        x = curve("location", 0, (1, 0.0), (4.995, 0.0))
        ball = keyed_ball(BLENDER_5_0, x)
        session = session_on(monkeypatch, BLENDER_5_0, ball)

        session.insert_keyframes(
            "Ball",
            [{"data_path": "location", "index": 0, "frames": [5], "values": [2.0]}],
            interpolation="LINEAR",
        )

        assert [(k.co, k.interpolation) for k in x.keyframe_points] == [
            ([1, 0.0], "BEZIER"),
            ([4.995, 2.0], "LINEAR"),
        ]

    def test_checks_every_entry_before_keying_any(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ball = keyed_ball(BLENDER_5_0)
        session = session_on(monkeypatch, BLENDER_5_0, ball)
        good = {"data_path": "location", "frames": [1], "values": [[0, 0, 0]]}
        bad_entries = [
            (
                {"data_path": "location", "frames": [1, 2], "values": [[0, 0, 0]]},
                "2 frames but 1 values",
            ),
            (
                {"data_path": "location", "frames": [1], "values": [[0, 0]]},
                "has 3 elements",
            ),
            (
                {
                    "data_path": "location",
                    "frames": [float("inf")],
                    "values": [[0, 0, 0]],
                },
                "finite",
            ),
            (
                {"data_path": "locaton", "frames": [1], "values": [1.0]},
                "no property 'locaton'",
            ),
            ({"frames": [1], "values": [1.0]}, "data_path is required"),
        ]
        for entry, message in bad_entries:
            with pytest.raises(ValueError, match=rf"channels\[1\]: .*{message}"):
                session.insert_keyframes("Ball", [good, entry])
        assert ball.calls == []

    def test_refuses_a_locked_or_sampled_curve_before_keying(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        locked, sampled = FCurve("location", 2, 1), FCurve('["gear"]', 0, 1)
        locked.lock = True
        sampled.sampled_points = [object()]
        ball = keyed_ball(BLENDER_5_0, locked, sampled)
        ball.props["gear"] = 1
        session = session_on(monkeypatch, BLENDER_5_0, ball)

        with pytest.raises(ValueError, match=r"location\[2\] is locked"):
            session.insert_keyframes(
                "Ball",
                [{"data_path": "location", "frames": [4], "values": [[0, 0, 0]]}],
            )
        with pytest.raises(ValueError, match=r'\["gear"\] is sampled'):
            session.insert_keyframes(
                "Ball", [{"data_path": '["gear"]', "frames": [4], "values": [2]}]
            )
        assert ball.calls == []

    def test_refuses_more_keys_than_one_call_takes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ball = keyed_ball(BLENDER_5_0)
        session = session_on(monkeypatch, BLENDER_5_0, ball)
        frames = list(range(blender_runtime.MAX_KEYFRAMES_PER_CALL // 3 + 1))

        with pytest.raises(ValueError, match="more than the 10000 one call inserts"):
            session.insert_keyframes(
                "Ball",
                [
                    {
                        "data_path": "location",
                        "frames": frames,
                        "values": [[0, 0, 0]] * len(frames),
                    }
                ],
            )
        assert ball.calls == []

    def test_a_refusal_part_way_says_what_went_in(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class RefusesTheThird(Object):
            __slots__ = ()

            def keyframe_insert(self, data_path: str, /, **kwargs: Any) -> bool:
                if len(self.calls) == 2:
                    self.calls.append(("refused", data_path))
                    return False
                return super().keyframe_insert(data_path, **kwargs)

        keyed = keyed_ball(BLENDER_5_0)
        ball = RefusesTheThird("Ball", keyed.animation_data, keyed.fcurves)
        keyed.fcurves.append(curve('["gear"]', 0, (1, 1.0), (10, 5.0)))
        ball.props["gear"] = 4
        ball.location = [7.0, 7.0, 7.0]
        session = session_on(monkeypatch, BLENDER_5_0, ball)

        result = session.insert_keyframes(
            "Ball",
            [
                {
                    "data_path": "location",
                    "frames": [3, 4, 5],
                    "values": [[1, 1, 1]] * 3,
                },
                {"data_path": '["gear"]', "frames": [6], "values": [2]},
            ],
        )

        # Returned, not raised: the keys stay, and an attached add-on records the
        # completed call as an undo step.
        assert result["success"] is False
        assert result["error_type"] == "KeyframesPartlyInserted"
        assert "after this call inserted 6 keys, which stay" in result["error"]
        assert result["details"] == {
            "inserted": 6,
            "refused": {"data_path": "location", "index": -1, "frame": 5.0},
        }
        # The channels that got keys show them: frame 1 holds the first key, 1.
        # gear, never reached, keeps its unkeyed value.
        assert ball.location == [1.0, 1.0, 1.0]
        assert ball.props["gear"] == 4
        assert "warning" not in result

    def test_refuses_a_channel_keyed_twice_on_one_frame(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The later key replaced the earlier, and key_count counted both."""
        ball = keyed_ball(BLENDER_5_0)
        session = session_on(monkeypatch, BLENDER_5_0, ball)
        twice = {"data_path": "location", "frames": [2, 2], "values": [[0, 0, 0]] * 2}
        whole = {"data_path": "location", "frames": [5], "values": [[1, 2, 3]]}
        element = {"data_path": "location", "index": 1, "frames": [5], "values": [9]}

        with pytest.raises(
            ValueError, match=r"channels\[0\] keys location\[0\] at frame 2 twice"
        ):
            session.insert_keyframes("Ball", [twice])
        with pytest.raises(
            ValueError,
            match=r"channels\[1\] and channels\[0\] both key location\[1\] at frame 5",
        ):
            session.insert_keyframes("Ball", [whole, element])
        assert ball.calls == []
        other = {"data_path": "location", "index": 1, "frames": [6], "values": [9]}
        assert session.insert_keyframes("Ball", [whole, other])["key_count"] == 4

    def test_refuses_frames_blender_would_merge_into_one_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Blender keeps one key for keys within 0.01 frames: 2 and 2.005 made one."""
        ball = keyed_ball(BLENDER_5_0)
        session = session_on(monkeypatch, BLENDER_5_0, ball)
        near = {"data_path": "location", "index": 0, "frames": [2.0, 2.005], "values": [1, 2]}
        first = {"data_path": "location", "index": 0, "frames": [2.0], "values": [1]}
        second = {"data_path": "location", "frames": [2.004], "values": [[2, 2, 2]]}

        with pytest.raises(
            ValueError,
            match=r"channels\[0\] keys location\[0\] at frames 2 and 2\.005, which "
            r"Blender keeps as one key",
        ):
            session.insert_keyframes("Ball", [near])
        with pytest.raises(
            ValueError,
            match=r"channels\[1\] and channels\[0\] both key location\[0\] at frames "
            r"2 and 2\.004",
        ):
            session.insert_keyframes("Ball", [first, second])
        assert ball.calls == []
        # 0.0101 apart, as 32-bit frames, is two keys in Blender.
        apart = {"data_path": "location", "index": 0, "frames": [2.0, 2.0101], "values": [1, 2]}
        assert session.insert_keyframes("Ball", [apart])["key_count"] == 2

    @pytest.mark.parametrize(
        ("modifier", "keyed"),
        [
            (SimpleNamespace(type="LIMITS", mute=False, is_valid=True), False),
            (SimpleNamespace(type="ENVELOPE", mute=False, is_valid=True), False),
            (SimpleNamespace(type="LIMITS", mute=True, is_valid=True), True),
            (SimpleNamespace(type="LIMITS", mute=False, is_valid=False), True),
            (SimpleNamespace(type="CYCLES", mute=False, is_valid=True), True),
            (
                SimpleNamespace(
                    type="GENERATOR", mute=False, is_valid=True, use_additive=False
                ),
                False,
            ),
            (
                SimpleNamespace(
                    type="GENERATOR", mute=False, is_valid=True, use_additive=True
                ),
                True,
            ),
        ],
        ids=[
            "limits",
            "envelope",
            "muted-limits",
            "invalid-limits",
            "cycles",
            "generator",
            "additive-generator",
        ],
    )
    def test_refuses_an_fcurve_an_fmodifier_overrides_before_keying(
        self, monkeypatch: pytest.MonkeyPatch, modifier: Any, keyed: bool
    ) -> None:
        """A LIMITS modifier on location[2] let index 0 be keyed first; those keys stayed."""
        blocked = FCurve("location", 2, 1)
        blocked.modifiers = [modifier]
        ball = keyed_ball(BLENDER_5_0, blocked)
        session = session_on(monkeypatch, BLENDER_5_0, ball)
        channels = [
            {
                "data_path": "location",
                "index": 0,
                "frames": [1, 2, 3],
                "values": [1, 2, 3],
            },
            {"data_path": "location", "index": 2, "frames": [4], "values": [5]},
        ]

        if keyed:
            assert session.insert_keyframes("Ball", channels)["key_count"] == 4
            return
        with pytest.raises(
            ValueError,
            match=rf"channels\[1\]: the F-curve of location\[2\] has a {modifier.type} "
            "F-modifier that overrides its keys",
        ):
            session.insert_keyframes("Ball", channels)
        assert ball.calls == []

    def test_refuses_an_empty_call_and_unknown_choices(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session = session_on(monkeypatch, BLENDER_5_0, keyed_ball(BLENDER_5_0))
        with pytest.raises(ValueError, match="channels is empty"):
            session.insert_keyframes("Ball", [])
        with pytest.raises(ValueError, match="Invalid interpolation"):
            session.insert_keyframes("Ball", [{}], interpolation="SMOOTH")
        with pytest.raises(ValueError, match="Invalid keyframe_type"):
            session.insert_keyframes("Ball", [{}], keyframe_type="KEY")


# -- round 2 of the audit: key values, sub-frame deletion, missing things ------------


class TestKeyValuesFitAnFCurve:
    """m-1: NaN was keyed as NaN and inf as 3.4e38; a key value must be finite and fit float32."""

    @pytest.mark.parametrize(
        "value",
        [[math.nan, 0.0, 0.0], [math.inf, 0.0, 0.0], [1e39, 0.0, 0.0], [0.0, -3.5e38, 0.0]],
        ids=["nan", "inf", "1e39", "-3.5e38"],
    )
    def test_insert_keyframe_refuses_the_value_before_keying(
        self, monkeypatch: pytest.MonkeyPatch, value: List[float]
    ) -> None:
        ball = keyed_ball(BLENDER_5_0)
        ball.location = [1.0, 2.0, 3.0]
        session = session_on(monkeypatch, BLENDER_5_0, ball)

        with pytest.raises(ValueError, match="is not a finite number of at most 3.402823e"):
            session.insert_keyframe("Ball", "location", 10, value=value)

        assert ball.calls == []
        assert ball.location == [1.0, 2.0, 3.0]

    def test_insert_keyframes_refuses_the_value_before_keying(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ball = keyed_ball(BLENDER_5_0)
        session = session_on(monkeypatch, BLENDER_5_0, ball)
        channel = {"data_path": "location", "index": 0, "frames": [1, 2], "values": [math.nan, 1]}

        with pytest.raises(ValueError, match=r"channels\[0\]: .*nan is not a finite number"):
            session.insert_keyframes("Ball", [channel])

        assert ball.calls == []

    def test_the_largest_float32_is_keyed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ball = keyed_ball(BLENDER_5_0)
        session = session_on(monkeypatch, BLENDER_5_0, ball)

        session.insert_keyframe("Ball", "location", 10, index=0, value=3.4028234663852886e38)

        assert channel_summary(session.get_keyframes("Ball")) == [("location", 0, 1, [10, 10])]

    @pytest.mark.parametrize("value", [math.nan, math.inf, [0.0, math.nan, 0.0]])
    def test_the_request_models_refuse_non_finite_values(self, value: Any) -> None:
        from pydantic import ValidationError as PydanticValidationError

        from simul.mcp.schemas.blender import (
            BlenderInsertKeyframeRequest,
            BlenderKeyframeChannelData,
        )

        with pytest.raises(PydanticValidationError):
            BlenderInsertKeyframeRequest(
                object_name="Ball", data_path="location", frame=1, value=value
            )
        with pytest.raises(PydanticValidationError):
            BlenderKeyframeChannelData(data_path="location", frames=[1], values=[value])
        with pytest.raises(PydanticValidationError):
            BlenderKeyframeChannelData(data_path="location", frames=[math.nan], values=[1.0])


class TestDeleteKeyframeRound2:
    def test_deletes_a_sub_frame_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """m-2: frame was an integer, so a key insert_keyframes made at 10.25 could not go."""
        from simul.mcp.schemas.blender import (
            BlenderDeleteKeyframeRequest,
            BlenderDeleteKeyframeResponse,
        )

        ball = keyed_ball(BLENDER_5_0, FCurve("location", 0, 1, 10.25))
        session = session_on(monkeypatch, BLENDER_5_0, ball)
        request = BlenderDeleteKeyframeRequest(
            object_name="Ball", data_path="location", frame=10.25, index=0
        )

        deleted = session.delete_keyframe("Ball", "location", request.frame, index=0)

        assert deleted == {"object_name": "Ball", "data_path": "location", "frame": 10.25}
        assert BlenderDeleteKeyframeResponse(success=True, **deleted).frame == 10.25
        assert channel_summary(session.get_keyframes("Ball")) == [("location", 0, 1, [1, 1])]
        # A whole frame is still reported as one.
        whole = session.delete_keyframe("Ball", "location", 1.0, index=0)
        assert whole["frame"] == 1 and isinstance(whole["frame"], int)

    def test_a_missing_key_or_animation_is_not_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """m-6: the error vocabulary's NotFoundError, with details.code."""
        ball = keyed_ball(BLENDER_5_0, FCurve("location", 0, 1))
        session = session_on(monkeypatch, BLENDER_5_0, ball)

        with pytest.raises(blender_runtime.NotFoundError) as missing:
            session.delete_keyframe("Ball", "location", 7, index=0)
        assert missing.value.details == {
            "code": "keyframe_not_found",
            "object_name": "Ball",
            "data_path": "location",
            "frame": 7,
        }
        still = keyed_ball(BLENDER_5_0)
        still.animation_data = None
        session = session_on(monkeypatch, BLENDER_5_0, still)
        with pytest.raises(blender_runtime.NotFoundError) as unanimated:
            session.delete_keyframe("Ball", "location", 1)
        assert unanimated.value.details["code"] == "not_animated"


def test_a_missing_bone_is_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """m-6: get_keyframes' bone filter names a bone the armature lacks."""
    session = session_on(monkeypatch, BLENDER_5_0, rig(BLENDER_5_0, 2))

    with pytest.raises(blender_runtime.NotFoundError, match="has no bone 'B999'") as missing:
        session.get_keyframes("Rig", bone_name="B999")

    assert missing.value.details == {
        "code": "bone_not_found",
        "object_name": "Rig",
        "bone_name": "B999",
    }
