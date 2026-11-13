"""Playback, frame range and trajectory sampling against stand-ins that behave as Blender does.

Each case pins a defect Blender 5.0.1 showed through the animation tools:

- ``screen.animation_play`` toggles playback, so a second ``play`` (or
  ``reverse``) stopped it; in background mode playback reported
  ``is_playing: true`` on a frame no event loop ever advances.
- ``render.fps_base`` was ignored (29.97 fps read as 30) and a trajectory's
  time was ``frame / fps``, so frame 1 was 0.0417 s.
- A one-frame range was refused, and frames Blender clamps were accepted.
- A trajectory's world Euler angles wrapped (a joint keyed 0 to 6 pi read 0 at
  the end), without naming their order; ``step`` 0 or -1 and a reversed range
  were accepted; the subframe was not restored.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple, get_args

import pytest
from pydantic import ValidationError

from simul.adapters import blender_runtime
from simul.mcp.registration._reg_blender import BLENDER_TOOLS
from simul.mcp.schemas.blender import (
    BlenderGetTrajectoryRequest,
    BlenderKeyframeInterpolation,
    BlenderKeyframeType,
    BlenderPlayAnimationRequest,
    BlenderPlaybackAction,
    BlenderSetFrameRangeRequest,
    BlenderSetFrameRequest,
)
from simul.utils.paths import PathPolicy
from tests.fakes import (
    FakeDepsgraph,
    FakeEuler,
    FakeMatrix,
    FakePlayback,
    FakeTransformObject,
    FakeVector,
    euler_matrix,
    fake_app_handlers,
    matrix_quaternion,
)

Animate = Callable[[int], None]


def _blender(
    monkeypatch: pytest.MonkeyPatch,
    *,
    background: bool = False,
    frame: int = 1,
    fps: int = 24,
    fps_base: float = 1.0,
    frame_start: int = 1,
    objects: Tuple[Any, ...] = (),
    animate: Optional[Animate] = None,
) -> Tuple[blender_runtime.BlenderRuntimeSession, SimpleNamespace, FakePlayback]:
    """A session on a stand-in bpy: one scene, its playback and its objects.

    ``animate(frame)`` runs on every frame change, as Blender evaluates the
    animation of the frame it sets.
    """
    scene = SimpleNamespace(
        frame_current=frame,
        frame_subframe=0.0,
        frame_start=frame_start,
        frame_end=250,
        render=SimpleNamespace(fps=fps, fps_base=fps_base),
        rigidbody_world=None,
        objects={obj.name: obj for obj in objects},
    )

    def frame_set(new_frame: int, subframe: float = 0.0) -> None:
        scene.frame_current = new_frame
        scene.frame_subframe = subframe
        if animate is not None:
            animate(new_frame)

    scene.frame_set = frame_set
    handlers = fake_app_handlers()
    playback = FakePlayback(scene, handlers)
    fake = SimpleNamespace(
        app=SimpleNamespace(
            version=(5, 0, 1), background=background, handlers=handlers
        ),
        context=SimpleNamespace(
            scene=scene,
            screen=playback,
            evaluated_depsgraph_get=lambda: FakeDepsgraph(),
        ),
        data=SimpleNamespace(objects=scene.objects),
        ops=SimpleNamespace(screen=playback),
    )
    monkeypatch.setattr(blender_runtime, "bpy", fake)
    monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
    session = blender_runtime.BlenderRuntimeSession(
        path_policy=PathPolicy(enabled=True, allowed_paths=[])
    )
    return session, scene, playback


def _state(result: Dict[str, Any]) -> Tuple[bool, Optional[str], bool, int]:
    return (
        result["is_playing"],
        result["direction"],
        result["changed"],
        result["frame"],
    )


# -- playback ------------------------------------------------------------------


class TestPlayback:
    def test_play_again_keeps_playing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Play, a second later play again: the second toggled playback off at frame 26."""
        session, _, playback = _blender(monkeypatch)

        assert _state(session.play_animation("play")) == (True, "forward", True, 1)
        playback.advance(24)
        assert _state(session.play_animation("play")) == (True, "forward", False, 25)
        playback.advance(1)

        assert playback.is_animation_playing
        assert playback.scene.frame_current == 26
        assert playback.calls == [("animation_play", {"reverse": False})]

    def test_reverse_again_keeps_reversing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session, _, playback = _blender(monkeypatch, frame=50)

        session.play_animation("reverse")
        playback.advance(5)
        result = session.play_animation("reverse")

        assert _state(result) == (True, "reverse", False, 45)
        assert playback.is_animation_playing and playback.reverse
        assert playback.calls == [("animation_play", {"reverse": True})]

    def test_turns_playback_around_where_it_is(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session, _, playback = _blender(monkeypatch)
        session.play_animation("play")
        playback.advance(10)

        reversed_ = session.play_animation("reverse")
        playback.advance(3)
        forward = session.play_animation("play")

        assert _state(reversed_) == (True, "reverse", True, 11)
        assert _state(forward) == (True, "forward", True, 8)
        assert playback.calls == [
            ("animation_play", {"reverse": False}),
            ("animation_cancel", {"restore_frame": False}),
            ("animation_play", {"reverse": True}),
            ("animation_cancel", {"restore_frame": False}),
            ("animation_play", {"reverse": False}),
        ]

    def test_pause_keeps_the_frame_and_stop_returns_to_the_start(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session, scene, playback = _blender(monkeypatch, frame=5)
        session.play_animation("play")
        playback.advance(10)

        assert _state(session.play_animation("pause")) == (False, None, True, 15)

        session.play_animation("play")
        playback.advance(5)
        # Turning around restarts Blender's playback, but stop still returns to
        # where this playback began, not to the turn.
        session.play_animation("reverse")
        playback.advance(2)
        assert _state(session.play_animation("stop")) == (False, None, True, 15)
        assert scene.frame_current == 15
        assert not playback.is_animation_playing

    @pytest.mark.parametrize("action", ["pause", "stop"])
    def test_pause_or_stop_without_playback_changes_nothing(
        self, monkeypatch: pytest.MonkeyPatch, action: str
    ) -> None:
        session, _, playback = _blender(monkeypatch, frame=9)

        assert _state(session.play_animation(action)) == (False, None, False, 9)
        assert _state(session.play_animation(action)) == (False, None, False, 9)
        assert playback.calls == []

    def test_stops_playback_started_elsewhere_where_blender_started_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session, scene, playback = _blender(monkeypatch, frame=3)
        playback.animation_play()  # the user, in the timeline
        playback.advance(7)

        assert _state(session.play_animation("stop")) == (False, None, True, 3)
        assert playback.calls[-1] == ("animation_cancel", {"restore_frame": True})

    def test_restarts_playback_whose_direction_nothing_reports(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The user restarted playback in reverse: the recorded direction is void."""
        session, _, playback = _blender(monkeypatch)
        session.play_animation("play")
        playback.animation_cancel(restore_frame=False)  # the user, Esc ...
        playback.animation_play(reverse=True)  # ... then Shift+Ctrl+Space
        playback.advance(4)

        result = session.play_animation("play")
        playback.advance(2)

        assert _state(result) == (True, "forward", True, -3)
        assert not playback.reverse
        assert playback.scene.frame_current == -1
        assert session.play_animation("play")["changed"] is False

    @pytest.mark.parametrize("action", ["play", "reverse"])
    def test_background_mode_refuses_to_play(
        self, monkeypatch: pytest.MonkeyPatch, action: str
    ) -> None:
        """blender -b reported is_playing: true while the frame never moved."""
        session, _, playback = _blender(monkeypatch, background=True)

        with pytest.raises(RuntimeError, match="background mode .* no event loop"):
            session.play_animation(action)

        assert playback.calls == []
        assert _state(session.play_animation("stop")) == (False, None, False, 1)

    def test_without_a_screen_nothing_plays(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session, _, playback = _blender(monkeypatch)
        blender_runtime.bpy.context.screen = None

        assert _state(session.play_animation("stop")) == (False, None, False, 1)
        with pytest.raises(RuntimeError, match="No Blender screen"):
            session.play_animation("play")
        assert playback.calls == []

    def test_watches_playback_through_one_handler_each(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session, _, playback = _blender(monkeypatch)
        handlers = playback.handlers

        def _forget_playback(*_args: Any) -> None:
            """What a reloaded add-on's module left behind."""

        _forget_playback.__module__ = blender_runtime.__name__
        handlers.animation_playback_post.append(_forget_playback)

        session.play_animation("play")
        session.play_animation("pause")
        session.play_animation("play")

        expected = [blender_runtime._forget_playback]
        assert handlers.animation_playback_pre == expected
        assert handlers.animation_playback_post == expected
        assert getattr(blender_runtime._forget_playback, "_bpy_persistent", False)

    def test_the_handlers_go_when_playback_is_no_longer_watched(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The handlers are persistent; the add-on's unregister removes them."""
        session, _, playback = _blender(monkeypatch)
        session.play_animation("play")

        blender_runtime.stop_watching_playback()

        assert playback.handlers.animation_playback_pre == []
        assert playback.handlers.animation_playback_post == []
        assert blender_runtime._playback == {}

    def test_refuses_an_unknown_action(self, monkeypatch: pytest.MonkeyPatch) -> None:
        session, _, playback = _blender(monkeypatch)
        with pytest.raises(ValueError, match="Invalid action 'toggle'"):
            session.play_animation("toggle")
        assert playback.calls == []


# -- frame and range -------------------------------------------------------------


class TestFrames:
    def test_fps_divides_by_fps_base(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """NTSC 29.97 fps (30 / 1.001) was reported as 30.0."""
        session, _, _ = _blender(monkeypatch, fps=30, fps_base=1.001)

        result = session.get_frame()

        assert result["fps"] == pytest.approx(30 / 1.001)
        assert result["fps_base"] == 1.001

    def test_a_one_frame_range(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Blender allows frame_start == frame_end; the tool refused it."""
        session, scene, _ = _blender(monkeypatch)

        result = session.set_frame_range(10, 10)

        assert (result["frame_start"], result["frame_end"]) == (10, 10)
        assert (scene.frame_start, scene.frame_end) == (10, 10)

    @pytest.mark.parametrize(
        ("start", "end", "message"),
        [
            (11, 10, r"frame_start \(11\) must not be after frame_end \(10\)"),
            (-5, 10, r"frame_start \(-5\) is outside Blender's range 0\.\.1048574"),
            (1, 2_000_000, r"frame_end \(2000000\) is outside"),
        ],
    )
    def test_refuses_a_range_blender_would_change(
        self, monkeypatch: pytest.MonkeyPatch, start: int, end: int, message: str
    ) -> None:
        """Blender 5.0.1 clamped frame_start -5 to 0 and frame_end 2e6 to 1048574."""
        session, scene, _ = _blender(monkeypatch)

        with pytest.raises(ValueError, match=message):
            session.set_frame_range(start, end)

        assert (scene.frame_start, scene.frame_end) == (1, 250)

    def test_refuses_a_frame_blender_would_clamp(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        session, scene, _ = _blender(monkeypatch)

        with pytest.raises(ValueError, match=r"frame \(10000000\) is outside"):
            session.set_frame(10**7)
        assert session.set_frame(-5) == {"frame": -5}
        assert session.set_frame(-1_048_574) == {"frame": -1_048_574}
        assert scene.frame_current == -1_048_574


# -- trajectory --------------------------------------------------------------------


def _trajectory(
    monkeypatch: pytest.MonkeyPatch, obj: Any, animate: Animate, **scene: Any
) -> Tuple[blender_runtime.BlenderRuntimeSession, SimpleNamespace]:
    session, fake_scene, _ = _blender(
        monkeypatch, objects=(obj,), animate=animate, **scene
    )
    animate(fake_scene.frame_current)
    return session, fake_scene


class TestTrajectory:
    def test_time_starts_at_frame_start_at_the_rate_blender_plays(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        """Frame 1 was 1/30 s and velocity used 30 fps for 29.97."""
        cart = FakeTransformObject("Cart")

        def animate(frame: int) -> None:
            cart.location = FakeVector((float(frame), 0.0, 0.0))

        session, _ = _trajectory(monkeypatch, cart, animate, fps=30, fps_base=1.001)

        result = session.get_object_trajectory("Cart", 1, 31, 30)

        assert result["fps"] == pytest.approx(30 / 1.001)
        assert result["time_origin_frame"] == 1
        times = [p["time"] for p in result["points"]]
        assert times == pytest.approx([0.0, 30 * 1.001 / 30])
        velocity = result["points"][1]["velocity"]
        assert velocity == pytest.approx([30 / 1.001, 0.0, 0.0])

    def test_a_multi_turn_joint_is_reported_unwrapped(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        """A joint keyed from 0 to 6 pi over frames 1-97 read 0.0 at the end."""
        spin = FakeTransformObject("Spin")

        def angle(frame: int) -> float:
            return (frame - 1) / 96 * 6 * math.pi

        def animate(frame: int) -> None:
            spin.rotation_euler = FakeEuler((0.0, 0.0, angle(frame)))

        session, _ = _trajectory(monkeypatch, spin, animate)

        result = session.get_object_trajectory("Spin", 1, 97, 8)

        assert (result["rotation_mode"], result["euler_order"]) == ("XYZ", "XYZ")
        authored = [angle(p["frame"]) for p in result["points"]]
        assert [p["local_rotation"][2] for p in result["points"]] == pytest.approx(
            authored
        )
        assert [p["rotation_euler"][2] for p in result["points"]] == pytest.approx(
            authored, abs=1e-9
        )
        assert result["points"][-1]["rotation_euler"][2] == pytest.approx(6 * math.pi)
        for point in result["points"]:
            expected = matrix_quaternion(euler_matrix(point["local_rotation"]))
            assert abs(
                sum(a * b for a, b in zip(point["rotation_quaternion"], expected))
            ) == pytest.approx(1.0)
        quaternions = [p["rotation_quaternion"] for p in result["points"]]
        assert all(
            sum(a * b for a, b in zip(q, r)) > 0
            for q, r in zip(quaternions, quaternions[1:])
        )

    def test_names_the_euler_order_of_the_object(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        arm = FakeTransformObject(
            "Arm", rotation_mode="ZYX", rotation_euler=(0.3, 0.2, 0.1)
        )
        session, _ = _trajectory(monkeypatch, arm, lambda frame: None)

        (point,) = session.get_object_trajectory("Arm", 1, 1)["points"]

        assert point["rotation_euler"] == pytest.approx([0.3, 0.2, 0.1])
        assert point["local_rotation"] == pytest.approx([0.3, 0.2, 0.1])

    def test_a_quaternion_object_reports_its_own_channels(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        wheel = FakeTransformObject("Wheel", rotation_mode="QUATERNION")

        def animate(frame: int) -> None:
            half = frame * 0.3 / 2
            wheel.rotation_quaternion = FakeVector(
                (math.cos(half), 0.0, 0.0, math.sin(half))
            )

        session, _ = _trajectory(monkeypatch, wheel, animate)

        result = session.get_object_trajectory("Wheel", 1, 20, 1)

        assert (result["rotation_mode"], result["euler_order"]) == ("QUATERNION", "XYZ")
        assert result["points"][-1]["local_rotation"] == pytest.approx(
            [math.cos(3.0), 0.0, 0.0, math.sin(3.0)]
        )
        # Past half a turn the quaternion flips sign where it is read from the
        # matrix; the reported one stays continuous.
        quaternions = [p["rotation_quaternion"] for p in result["points"]]
        assert all(
            sum(a * b for a, b in zip(q, r)) > 0
            for q, r in zip(quaternions, quaternions[1:])
        )
        assert quaternions[-1] == pytest.approx(
            [math.cos(3.0), 0.0, 0.0, math.sin(3.0)]
        )

    def test_world_pose_follows_the_parent(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        base = FakeTransformObject("Base", location=(0.0, 0.0, 1.0))
        link = FakeTransformObject("Link", parent=base)
        link.matrix_parent_inverse = type(base.matrix_world).Identity(4)
        tip = FakeTransformObject("Tip", parent=link)
        tip.matrix_parent_inverse = type(base.matrix_world).Identity(4)
        tip.location = FakeVector((1.0, 0.0, 0.0))

        def animate(frame: int) -> None:
            link.rotation_euler = FakeEuler((0.0, frame * 0.1, 0.0))

        session, _ = _trajectory(monkeypatch, tip, animate)

        result = session.get_object_trajectory("Tip", 1, 10, 3)

        for point in result["points"]:
            q = point["frame"] * 0.1
            # Base at z=1, the link turning about Y, the tip 1 m along the link's X.
            assert point["location"] == pytest.approx(
                [math.cos(q), 0.0, 1.0 - math.sin(q)]
            )
            assert point["rotation_euler"] == pytest.approx([0.0, q, 0.0])
            assert point["local_rotation"] == pytest.approx([0.0, 0.0, 0.0])

    @pytest.mark.parametrize(
        ("start", "end", "step", "message"),
        [
            (1, 5, 0, "step must be at least 1, got 0"),
            (1, 5, -1, "step must be at least 1, got -1"),
            (10, 5, 1, r"start_frame \(10\) must not be after end_frame \(5\)"),
        ],
    )
    def test_refuses_a_step_below_one_and_a_reversed_range(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mathutils: Any,
        start: int,
        end: int,
        step: int,
        message: str,
    ) -> None:
        """step 0 or -1 returned points without any velocity; 10..5 an empty success."""
        frames: List[int] = []
        cart = FakeTransformObject("Cart")
        session, _ = _trajectory(monkeypatch, cart, frames.append)
        frames.clear()

        with pytest.raises(ValueError, match=message):
            session.get_object_trajectory("Cart", start, end, step)

        assert frames == []

    def test_a_sheared_child_reports_the_rotation_get_object_info_reports(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        """Under a parent scaled (3, 1, 1), a child turned 30 degrees about Z read 0.190 rad."""
        parent = FakeTransformObject("Parent", scale=(3.0, 1.0, 1.0))
        child = FakeTransformObject(
            "Link", parent=parent, rotation_euler=(0.0, 0.0, math.radians(30))
        )
        child.matrix_parent_inverse = FakeMatrix.Identity(4)
        session, _ = _trajectory(monkeypatch, child, lambda frame: None)

        (point,) = session.get_object_trajectory("Link", 1, 1)["points"]

        reference = blender_runtime.world_transform(child)
        assert point["rotation_euler"] == pytest.approx(
            reference["world_rotation_euler"], abs=1e-6
        )
        assert point["rotation_euler"][2] == pytest.approx(math.radians(30), abs=1e-6)
        assert point["rotation_quaternion"] == pytest.approx(
            reference["world_rotation_quaternion"], abs=1e-6
        )

    def test_restores_the_frame_and_subframe(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        cart = FakeTransformObject("Cart")
        session, scene = _trajectory(monkeypatch, cart, lambda frame: None, frame=7)
        scene.frame_subframe = 0.25

        session.get_object_trajectory("Cart", 1, 20, 5)

        assert (scene.frame_current, scene.frame_subframe) == (7, 0.25)


# -- request models and annotations ----------------------------------------------


class TestDeclarations:
    def test_trajectory_request_refuses_what_the_operation_refuses(self) -> None:
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            BlenderGetTrajectoryRequest(
                object_name="Cube", start_frame=1, end_frame=5, step=0
            )
        with pytest.raises(ValidationError, match="must not be after end_frame"):
            BlenderGetTrajectoryRequest(object_name="Cube", start_frame=9, end_frame=5)
        assert BlenderGetTrajectoryRequest(
            object_name="Cube", start_frame=5, end_frame=5
        )

    def test_frame_requests_refuse_what_blender_would_clamp(self) -> None:
        assert BlenderSetFrameRequest(frame=-1_048_574).frame == -1_048_574
        for frame in (-1_048_575, 1_048_575):
            with pytest.raises(ValidationError):
                BlenderSetFrameRequest(frame=frame)
        with pytest.raises(ValidationError):
            BlenderSetFrameRangeRequest(frame_start=-1, frame_end=10)
        with pytest.raises(ValidationError):
            BlenderSetFrameRangeRequest(frame_start=1, frame_end=1_048_575)
        assert (blender_runtime.FRAME_LIMIT, blender_runtime.SCENE_FRAME_MIN) == (
            1_048_574,
            0,
        )

    def test_range_request_allows_one_frame(self) -> None:
        assert BlenderSetFrameRangeRequest(frame_start=10, frame_end=10)
        with pytest.raises(ValidationError, match="must not be after"):
            BlenderSetFrameRangeRequest(frame_start=11, frame_end=10)

    def test_playback_request_takes_the_four_actions(self) -> None:
        for action in blender_runtime.PLAYBACK_ACTIONS:
            assert BlenderPlayAnimationRequest(action=action).action == action
        with pytest.raises(ValidationError):
            BlenderPlayAnimationRequest(action="toggle")

    def test_schema_choices_mirror_the_add_on_constants(self) -> None:
        """The runtime ships in the add-on, which cannot import the schemas."""
        assert get_args(BlenderPlaybackAction) == blender_runtime.PLAYBACK_ACTIONS
        assert (
            get_args(BlenderKeyframeInterpolation)
            == blender_runtime.KEYFRAME_INTERPOLATIONS
        )
        assert get_args(BlenderKeyframeType) == blender_runtime.KEYFRAME_TYPES

    def test_sampling_and_playback_are_not_read_only(self) -> None:
        """The trajectory moves the frame and fills caches; playback is now idempotent."""
        hints = {spec.name: spec.hints for spec in BLENDER_TOOLS}
        trajectory = hints["get_blender_object_trajectory"]
        playback = hints["play_blender_animation"]
        assert (trajectory.read_only, trajectory.idempotent) == (False, True)
        assert (playback.read_only, playback.idempotent) == (False, True)
