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
    """``bpy.types.Keyframe``: its (frame, value) point."""

    __slots__ = ("co",)

    def __init__(self, frame: float) -> None:
        self.co = (frame, 0.0)


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
    """``bpy.types.Object``: its animation data and keyframe functions.

    ``keyframe_insert`` and ``keyframe_delete`` take their arguments as Blender
    5.0 documents them and, like Blender, report a refused or missing key by
    returning False. ``keyframe_delete`` removes the keys on the frame from
    ``fcurves``, the curves Blender keys this object into.
    """

    __slots__ = ("name", "animation_data", "fcurves", "locked", "calls")

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
        self.calls: List[Tuple[str, str, int, float]] = []

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
        self.calls.append(("keyframe_insert", data_path, index, frame))
        return not self.locked

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
    def test_inserts_through_blender(
        self, monkeypatch: pytest.MonkeyPatch, version: Version
    ) -> None:
        ball = Object("Ball")
        session = session_on(monkeypatch, version, ball)

        result = session.insert_keyframe("Ball", "location", 10, index=2)

        assert result == {"object_name": "Ball", "data_path": "location", "frame": 10}
        assert ball.calls == [("keyframe_insert", "location", 2, 10)]

    def test_reports_a_key_blender_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A locked F-curve: keyframe_insert returns False instead of raising."""
        ball = keyed_ball(BLENDER_5_0, FCurve("location", 0, 1), locked=True)
        session = session_on(monkeypatch, BLENDER_5_0, ball)
        with pytest.raises(
            RuntimeError,
            match=r"inserted no keyframe on 'Ball' location at frame 5; .* locked",
        ):
            session.insert_keyframe("Ball", "location", 5)

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
