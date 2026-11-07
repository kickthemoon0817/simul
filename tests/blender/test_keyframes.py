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
    """``bpy.types.Keyframe``: its (frame, value) point, interpolation and key type.

    A new key interpolates as Blender's default preference does, Bezier.
    """

    __slots__ = ("co", "interpolation", "type")

    def __init__(
        self, frame: float, value: float = 0.0, key_type: str = "KEYFRAME"
    ) -> None:
        self.co = [frame, value]
        self.interpolation = "BEZIER"
        self.type = key_type


class FCurve:
    """``bpy.types.FCurve``."""

    __slots__ = ("data_path", "array_index", "keyframe_points")

    def __init__(self, data_path: str, array_index: int, *frames: float) -> None:
        self.data_path = data_path
        self.array_index = array_index
        self.keyframe_points = [Keyframe(frame) for frame in frames]


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
        __slots__ = ("action",)

        def __init__(self, action: Any) -> None:
            self.action = action


class Blender50:
    """``bpy.types`` of Blender 5.0: slotted actions only, no ``Action.fcurves``."""

    class Action:
        __slots__ = ("layers", "slots")

        def __init__(self, layers: List[ActionLayer], slots: List[ActionSlot]) -> None:
            self.layers = layers
            self.slots = slots

    class AnimData:
        __slots__ = ("action", "action_slot")

        def __init__(self, action: Any, action_slot: Optional[ActionSlot]) -> None:
            self.action = action
            self.action_slot = action_slot


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
    into ``fcurves``, the curves Blender keys this object into, replacing a key
    on the same frame; ``keyframe_delete`` removes the keys on the frame from
    them. ``location`` stands for an array property, ``props`` for custom
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
            kept = [k for k in fcurve.keyframe_points if abs(k.co[0] - frame) >= 0.01]
            key = Keyframe(frame, float(values[element]), keytype)
            fcurve.keyframe_points = sorted(kept + [key], key=lambda k: k.co[0])
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
    fake_bpy = SimpleNamespace(
        app=SimpleNamespace(version=version),
        context=SimpleNamespace(scene=SimpleNamespace(frame_current=1)),
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
        assert session.get_keyframes("Ball") == {
            "object_name": "Ball",
            "has_animation": False,
            "channels": [],
        }

    @slotted_versions
    def test_action_without_the_objects_slot(
        self, monkeypatch: pytest.MonkeyPatch, version: Version
    ) -> None:
        shared = slotted_action(
            version, {ActionSlot("OBOther"): [FCurve("location", 0, 1)]}
        )
        ball = Object("Ball", bpy_types(version).AnimData(shared, None))
        session = session_on(monkeypatch, version, ball)
        assert session.get_keyframes("Ball") == {
            "object_name": "Ball",
            "has_animation": False,
            "channels": [],
        }

    @every_version
    def test_an_action_whose_keys_were_all_deleted(
        self, monkeypatch: pytest.MonkeyPatch, version: Version
    ) -> None:
        session = session_on(monkeypatch, version, keyed_ball(version))
        assert session.get_keyframes("Ball") == {
            "object_name": "Ball",
            "has_animation": True,
            "channels": [],
        }

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
    def test_keys_a_given_value_and_leaves_the_property_as_it_was(
        self, monkeypatch: pytest.MonkeyPatch, version: Version
    ) -> None:
        """Keying a value at another frame keyed the current one: there was no value."""
        ball = keyed_ball(version)
        ball.location = [1.0, 2.0, 3.0]
        session = session_on(monkeypatch, version, ball)

        result = session.insert_keyframe(
            "Ball", "location", 10, value=[4.0, 5.0, 6.0]
        )

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
        # The frame shown is 1: the property still shows its value there.
        assert ball.location == [1.0, 2.0, 3.0]

    def test_a_given_value_on_the_current_frame_is_what_the_property_shows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ball = keyed_ball(BLENDER_5_0)
        session = session_on(monkeypatch, BLENDER_5_0, ball)
        blender_runtime.bpy.context.scene.frame_current = 10

        session.insert_keyframe("Ball", "location", 10, value=[4.0, 5.0, 6.0])

        assert ball.location == [4.0, 5.0, 6.0]

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
        assert ball.location == [1.0, 2.0, 3.0]

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
        assert ball.props["gear"] == 2
        assert bone.rotation_euler == [0.0, 0.0, 0.0]
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
        ball = keyed_ball(BLENDER_5_0, FCurve("location", 2, 1), locked=True)
        session = session_on(monkeypatch, BLENDER_5_0, ball)
        with pytest.raises(
            ValueError, match=r"no keyframe on location\[2\] at frame 1 .* locked"
        ):
            session.delete_keyframe("Ball", "location", 1, index=2)
        assert channel_summary(session.get_keyframes("Ball")) == [
            ("location", 2, 1, [1, 1])
        ]
