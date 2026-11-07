"""Paging, bones, several targets and the rigid-body walk of the trajectory tools, and pose bones.

Each case pins what Blender 5.0.1 showed through the animation tools:

- A 1 000-frame trajectory came back as frames 1-199, cut by the server's
  result budget with a hint naming parameters these tools do not have; nothing
  reached the rest.
- A bone's trajectory failed with "Object not found: J2"; each object needed
  its own call, and so its own walk of the frames.
- A 2-point request on an unbaked 1 000-body scene took 29.3 s: it walked the
  simulation from the cache start again, through frames the cache held.
- No tool set a pose bone's transform.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple, get_args

import numpy as np
import pytest
from pydantic import ValidationError

from simul.adapters import blender_runtime
from simul.mcp.result_budget import DEFAULT_RESULT_BUDGET_BYTES, encode_result_budget
from simul.mcp.schemas.blender import (
    BlenderGetKeyframesRequest,
    BlenderGetTrajectoriesRequest,
    BlenderGetTrajectoryRequest,
    BlenderGetTrajectoryResponse,
    BlenderInsertKeyframesRequest,
    BlenderSetPoseBoneTransformRequest,
    BlenderTrajectoryField,
)
from simul.utils.paths import PathPolicy
from tests.fakes import (
    FakeDepsgraph,
    FakeEuler,
    FakeMatrix,
    FakePoseBones,
    FakeTransformObject,
    FakeVector,
    euler_matrix,
    fake_app_handlers,
    matrix_quaternion,
)

Animate = Callable[[int], None]


class _Objects(dict):
    """``bpy.data.objects``: found by name, iterated as objects."""

    def __iter__(self) -> Any:  # type: ignore[override]
        return iter(self.values())


def _session(
    monkeypatch: pytest.MonkeyPatch,
    *objects: Any,
    world: Any = None,
    frame: int = 1,
    animate: Optional[Animate] = None,
) -> Tuple[blender_runtime.BlenderRuntimeSession, SimpleNamespace, List[int]]:
    """A session on a stand-in bpy; the list records every frame the scene is set to."""
    frames: List[int] = []
    scene = SimpleNamespace(
        frame_current=frame,
        frame_subframe=0.0,
        frame_start=1,
        frame_end=250,
        render=SimpleNamespace(fps=24, fps_base=1.0),
        rigidbody_world=world,
        objects=_Objects({obj.name: obj for obj in objects}),
    )

    def frame_set(new_frame: int, subframe: float = 0.0) -> None:
        frames.append(new_frame)
        scene.frame_current = new_frame
        scene.frame_subframe = subframe
        if animate is not None:
            animate(new_frame)

    scene.frame_set = frame_set
    fake = SimpleNamespace(
        app=SimpleNamespace(
            version=(5, 0, 1), background=True, handlers=fake_app_handlers()
        ),
        context=SimpleNamespace(
            scene=scene, evaluated_depsgraph_get=lambda: FakeDepsgraph()
        ),
        data=SimpleNamespace(objects=scene.objects),
    )
    monkeypatch.setattr(blender_runtime, "bpy", fake)
    monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
    session = blender_runtime.BlenderRuntimeSession(
        path_policy=PathPolicy(enabled=True, allowed_paths=[])
    )
    return session, scene, frames


def _cart(speed: float = 1.0) -> Tuple[FakeTransformObject, Animate]:
    """An object moving ``speed`` along X a frame, and what moves it."""
    cart = FakeTransformObject("Cart")

    def animate(frame: int) -> None:
        cart.location = FakeVector((speed * frame, 0.0, 0.0))

    return cart, animate


def _every_page(call: Callable[..., Dict[str, Any]]) -> List[Dict[str, Any]]:
    pages, offset = [], 0
    while offset is not None:
        page = call(offset=offset)
        pages.append(page)
        offset = page["next_offset"]
    return pages


# -- paging ---------------------------------------------------------------------------


class TestTrajectoryPages:
    def test_pages_join_into_what_one_reply_would_hold(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        """A 1 000-frame trajectory came back as frames 1-199, with no way to the rest."""
        cart, animate = _cart()
        session, scene, _ = _session(monkeypatch, cart, frame=7, animate=animate)

        pages = _every_page(
            lambda offset: session.get_object_trajectory("Cart", 1, 400, offset=offset)
        )

        points = [p for page in pages for p in page["points"]]
        assert [p["frame"] for p in points] == list(range(1, 401))
        assert len(pages) > 1
        assert all(page["total_points"] == 400 for page in pages)
        assert all(
            len(json.dumps(page)) <= blender_runtime.RESULT_PAGE_BYTES for page in pages
        )
        # Velocity across a page boundary uses the sample before it, as one reply would.
        assert points[0]["velocity"] is None
        assert all(p["velocity"] == [24.0, 0.0, 0.0] for p in points[1:])
        first = pages[0]
        assert first["truncation"]["hint"].count(f"offset={first['next_offset']}") == 1
        assert first["truncation"]["total"] == 400
        assert pages[-1]["truncated"] is False and pages[-1]["truncation"] is None
        assert scene.frame_current == 7

    def test_fields_choose_what_a_sample_reports(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        cart, animate = _cart()
        session, _, _ = _session(monkeypatch, cart, animate=animate)

        every = session.get_object_trajectory("Cart", 1, 1000)
        located = session.get_object_trajectory(
            "Cart", 1, 1000, fields=["velocity", "location"]
        )

        assert set(located["points"][1]) == {"frame", "location", "velocity"}
        assert located["point_count"] > 2 * every["point_count"]
        with pytest.raises(ValueError, match=r"Unknown trajectory fields \['speed'\]"):
            session.get_object_trajectory("Cart", 1, 10, fields=["speed"])

    def test_a_page_holds_as_many_samples_whatever_their_values(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        """The page size is known before the walk, so the walk is known before it starts."""
        counts = []
        for speed in (1e-7, 123456.789):
            cart, animate = _cart(speed)
            session, _, _ = _session(monkeypatch, cart, animate=animate)
            counts.append(session.get_object_trajectory("Cart", 1, 1000)["point_count"])
        assert counts[0] == counts[1]

    def test_limit_offset_and_step(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        cart, animate = _cart()
        session, _, _ = _session(monkeypatch, cart, animate=animate)

        page = session.get_object_trajectory("Cart", 1, 1000, 10, offset=3, limit=4)

        assert [p["frame"] for p in page["points"]] == [31, 41, 51, 61]
        assert (page["total_points"], page["next_offset"]) == (100, 7)
        assert page["points"][0]["velocity"] == [24.0, 0.0, 0.0]
        with pytest.raises(
            ValueError, match="offset 101 is past the last of the 100 samples"
        ):
            session.get_object_trajectory("Cart", 1, 1000, 10, offset=101)
        with pytest.raises(ValueError, match="limit must be at least 1"):
            session.get_object_trajectory("Cart", 1, 10, limit=0)

    def test_values_are_the_float32_blender_holds_in_their_shortest_form(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        cart = FakeTransformObject(
            "Cart", location=(np.float32(0.1), 0.0, np.float32(-2.7))
        )
        session, _, _ = _session(monkeypatch, cart)

        (point,) = session.get_object_trajectory("Cart", 1, 1, fields=["location"])[
            "points"
        ]

        assert point["location"] == [0.1, 0.0, -2.7]


# -- bones and several targets ---------------------------------------------------------


def _rig() -> Tuple[FakeTransformObject, SimpleNamespace, Animate]:
    """An armature at (1, 2, 3) with bone J1, its pose turning about X by 0.1 rad a frame."""
    rig = FakeTransformObject("Rig", "ARMATURE", location=(1.0, 2.0, 3.0))
    bone = SimpleNamespace(
        name="J1",
        rotation_mode="QUATERNION",
        rotation_quaternion=FakeVector((1.0, 0.0, 0.0, 0.0)),
        rotation_euler=FakeEuler(),
        rotation_axis_angle=FakeVector((0.0, 0.0, 1.0, 0.0)),
        location=FakeVector(),
        scale=FakeVector((1.0, 1.0, 1.0)),
        head=FakeVector((0.0, 0.0, 1.0)),
        tail=FakeVector((0.0, 0.0, 2.0)),
        matrix=FakeMatrix.Identity(4),
    )
    rig.pose = SimpleNamespace(bones=FakePoseBones([bone]))

    def animate(frame: int) -> None:
        angle = 0.1 * frame
        pose = np.eye(4)
        pose[:3, :3] = euler_matrix((angle, 0.0, 0.0))
        pose[:3, 3] = (0.0, 0.0, 1.0)
        bone.matrix = FakeMatrix(pose)
        bone.rotation_quaternion = FakeVector(
            (math.cos(angle / 2), math.sin(angle / 2), 0.0, 0.0)
        )

    return rig, bone, animate


class TestBones:
    def test_samples_a_bone_in_world_space(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        """A bone's name as the object gave "Object not found: J2"; no bone could be sampled."""
        rig, _, animate = _rig()
        session, _, _ = _session(monkeypatch, rig, animate=animate)

        result = session.get_object_trajectory("Rig", 1, 5, bone_name="J1")

        assert (
            result["bone_name"],
            result["rotation_mode"],
            result["euler_order"],
        ) == (
            "J1",
            "QUATERNION",
            "XYZ",
        )
        for point in result["points"]:
            angle = 0.1 * point["frame"]
            assert point["location"] == pytest.approx([1.0, 2.0, 4.0])
            assert point["rotation_euler"] == pytest.approx([angle, 0.0, 0.0], abs=1e-6)
            expected = matrix_quaternion(euler_matrix((angle, 0.0, 0.0)))
            assert point["rotation_quaternion"] == pytest.approx(expected, abs=1e-6)
            assert point["local_rotation"] == pytest.approx(
                [math.cos(angle / 2), math.sin(angle / 2), 0.0, 0.0], abs=1e-6
            )

    def test_a_bone_named_as_the_object_names_its_armature(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        rig, _, _ = _rig()
        session, _, _ = _session(monkeypatch, rig)

        with pytest.raises(
            ValueError,
            match=r"Object not found: J1; it is a bone of armature 'Rig': pass "
            r"object_name='Rig' and bone_name='J1'",
        ):
            session.get_object_trajectory("J1", 1, 5)
        with pytest.raises(ValueError, match="Object not found: Nobody$"):
            session.get_object_trajectory("Nobody", 1, 5)
        with pytest.raises(ValueError, match="Armature 'Rig' has no bone 'J9'"):
            session.get_object_trajectory("Rig", 1, 5, bone_name="J9")

    def test_samples_every_target_on_one_walk(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        rig, _, move_rig = _rig()
        cart, move_cart = _cart()

        def animate(frame: int) -> None:
            move_rig(frame)
            move_cart(frame)

        session, _, frames = _session(monkeypatch, rig, cart, animate=animate)

        result = session.get_trajectories(
            [{"object_name": "Rig", "bone_name": "J1"}, {"object_name": "Cart"}],
            1,
            30,
            2,
        )

        assert frames[:-1] == list(range(1, 31, 2))
        bone, moved = result["trajectories"]
        assert (bone["bone_name"], moved["object_name"]) == ("J1", "Cart")
        alone = session.get_object_trajectory("Cart", 1, 30, 2)
        assert moved["points"] == alone["points"]
        assert result["point_count"] == len(moved["points"]) == 15
        with pytest.raises(ValueError, match="targets is empty"):
            session.get_trajectories([], 1, 2)
        with pytest.raises(ValueError, match="33 targets is more than the 32"):
            session.get_trajectories([{"object_name": "Cart"}] * 33, 1, 2)


# -- the rigid-body walk (B3-12) -----------------------------------------------------


def _world(
    bodies: int = 5, info: str = "", outdated: bool = False, baked: bool = False
) -> SimpleNamespace:
    """An unbaked rigid-body world of ``bodies`` bodies, its cache 1-250 holding what ``info`` says."""
    return SimpleNamespace(
        enabled=True,
        collection=SimpleNamespace(all_objects=[object()] * bodies),
        point_cache=SimpleNamespace(
            frame_start=1,
            frame_end=250,
            is_baked=baked,
            is_outdated=outdated,
            info=info,
        ),
    )


class TestRigidBodyWalk:
    def test_resumes_where_the_cache_ends(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        """It walked from the cache start again, through 40 frames the cache held."""
        world = _world(info="40 frames in memory (12 KiB).")
        session, _, frames = _session(
            monkeypatch, FakeTransformObject("Body"), world=world
        )

        result = session.get_object_trajectory("Body", 30, 60, 10)

        # Frame 30 is read from the cache, 40 is its last, and 41-60 are simulated.
        assert frames == [30, *range(40, 61), 1]
        assert result["simulated_frames"] == 20
        assert result["warning"] is None

    @pytest.mark.parametrize(
        ("info", "outdated"),
        [
            ("", False),
            ("40 frames in memory (12 KiB), cache is outdated!", True),
            ("40 frames in memory (12 KiB), not exact since frame 12", False),
            ("메모리에 40 프레임 (12 KiB).", False),
        ],
        ids=["empty", "outdated", "frames-skipped", "translated"],
    )
    def test_walks_from_the_cache_start_when_what_it_holds_is_unknown(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any, info: str, outdated: bool
    ) -> None:
        world = _world(info=info, outdated=outdated)
        session, _, frames = _session(
            monkeypatch, FakeTransformObject("Body"), world=world
        )

        result = session.get_object_trajectory("Body", 30, 60, 10)

        assert frames == [*range(1, 61), 1]
        assert result["simulated_frames"] == 59

    def test_refuses_a_long_walk_before_setting_any_frame(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        """2 samples of 1 000 bodies over 1 000 frames took 29.3 s without asking."""
        world = _world(bodies=1000)
        world.point_cache.frame_end = 1000
        session, _, frames = _session(
            monkeypatch, FakeTransformObject("Body"), world=world
        )

        with pytest.raises(ValueError) as refused:
            session.get_object_trajectory("Body", 990, 1000, 10)

        message = str(refused.value)
        assert (
            "through 999 frames (frames 2-1000) of 1000 bodies, 999000 bodies x frames"
            in message
        )
        assert (
            "bake_blender_simulation" in message
            and "max_simulated_frames=999" in message
        )
        assert frames == []

    def test_the_bound_keeps_an_unasked_walk_inside_the_call_timeout(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        """A GUI Blender 5.0.1 took 24 s for 400 000 bodies x frames; a call waits 30 s."""
        world = _world(bodies=1000)
        session, _, frames = _session(monkeypatch, FakeTransformObject("Body"), world=world)

        session.get_object_trajectory("Body", 201, 201)
        assert frames[:-1] == list(range(1, 202))
        with pytest.raises(ValueError, match="BLENDER__CONNECTION_TIMEOUT") as refused:
            session.get_object_trajectory("Body", 202, 202)
        assert "through 201 frames (frames 2-202) of 1000 bodies" in str(refused.value)

    def test_max_simulated_frames_confirms_a_walk_or_bounds_it(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        world = _world(bodies=1000)
        world.point_cache.frame_end = 1000
        session, _, frames = _session(
            monkeypatch, FakeTransformObject("Body"), world=world
        )

        result = session.get_object_trajectory(
            "Body", 990, 1000, 10, max_simulated_frames=999
        )

        assert result["simulated_frames"] == 999
        assert frames == [*range(1, 1001), 1]
        frames.clear()
        with pytest.raises(ValueError, match="max_simulated_frames is 10"):
            session.get_object_trajectory("Body", 1, 30, max_simulated_frames=10)
        assert frames == []

    @pytest.mark.parametrize("state", ["baked", "disabled"])
    def test_a_baked_or_disabled_world_is_read_not_walked(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any, state: str
    ) -> None:
        world = _world(bodies=1000, baked=state == "baked")
        world.enabled = state != "disabled"
        session, _, frames = _session(
            monkeypatch, FakeTransformObject("Body"), world=world
        )

        result = session.get_object_trajectory("Body", 100, 200, 50)

        assert frames == [100, 150, 200, 1]
        assert result["simulated_frames"] == 0

    def test_says_when_samples_are_past_the_cache_end(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        world = _world(info="245 frames in memory (60 KiB).")
        session, _, frames = _session(
            monkeypatch, FakeTransformObject("Body"), world=world
        )

        result = session.get_object_trajectory("Body", 240, 260, 10)

        assert frames == [240, *range(245, 251), 260, 1]
        assert result["simulated_frames"] == 5
        assert "after frame 250 are past the rigid-body cache end" in result["warning"]
        assert "frame-250 pose" in result["warning"]


# -- the pose-bone tool ------------------------------------------------------------------


class TestPoseBoneTransform:
    def test_writes_xyz_euler_angles_into_a_quaternion_bone(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        """A bone's rotation_euler is ignored while its mode is QUATERNION, a new bone's default."""
        rig, bone, _ = _rig()
        session, _, _ = _session(monkeypatch, rig)

        result = session.set_pose_bone_transform(
            "Rig",
            "J1",
            location=[0, 0.5, 0],
            rotation_euler=[0.4, 0.0, 0.2],
            scale=[1, 1, 2],
        )

        expected = matrix_quaternion(euler_matrix((0.4, 0.0, 0.2)))
        assert list(bone.rotation_quaternion) == pytest.approx(expected)
        assert list(bone.rotation_euler) == [0.0, 0.0, 0.0]
        assert (list(bone.location), list(bone.scale)) == (
            [0.0, 0.5, 0.0],
            [1.0, 1.0, 2.0],
        )
        assert result["rotation_mode"] == "QUATERNION"
        assert result["rotation_euler"] == pytest.approx([0.4, 0.0, 0.2], abs=1e-6)
        assert result["rotation_in_mode"] == pytest.approx(expected, abs=1e-6)

    def test_keeps_the_quaternion_sign_it_had(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        rig, bone, _ = _rig()
        bone.rotation_quaternion = FakeVector((-1.0, 0.0, 0.0, 0.0))
        session, _, _ = _session(monkeypatch, rig)

        session.set_pose_bone_transform("Rig", "J1", rotation_euler=[0.1, 0.0, 0.0])

        assert bone.rotation_quaternion[0] < 0

    def test_writes_a_quaternion_into_an_euler_bone_of_another_order(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        rig, bone, _ = _rig()
        bone.rotation_mode = "ZYX"
        session, _, _ = _session(monkeypatch, rig)
        quaternion = matrix_quaternion(euler_matrix((0.3, -0.2, 0.5), "ZYX"))

        result = session.set_pose_bone_transform(
            "Rig", "J1", rotation_quaternion=list(quaternion)
        )

        assert list(bone.rotation_euler) == pytest.approx([0.3, -0.2, 0.5])
        assert result["rotation_in_mode"] == pytest.approx([0.3, -0.2, 0.5], abs=1e-6)

    def test_writes_an_axis_angle_bone(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        rig, bone, _ = _rig()
        bone.rotation_mode = "AXIS_ANGLE"
        session, _, _ = _session(monkeypatch, rig)

        session.set_pose_bone_transform("Rig", "J1", rotation_euler=[0.0, 0.5, 0.0])

        assert list(bone.rotation_axis_angle) == pytest.approx([0.5, 0.0, 1.0, 0.0])

    @pytest.mark.parametrize(
        ("arguments", "message"),
        [
            ({"location": [math.nan, 0, 0]}, "location must be 3 finite numbers"),
            ({"rotation_quaternion": [0, 0, 0, 0]}, "zero length"),
            (
                {"rotation_euler": [0, 0, 0], "rotation_quaternion": [1, 0, 0, 0]},
                "not both",
            ),
            ({"scale": [1, 1]}, "scale must be 3 finite numbers"),
        ],
    )
    def test_checks_every_value_before_writing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mathutils: Any,
        arguments: Dict[str, Any],
        message: str,
    ) -> None:
        rig, bone, _ = _rig()
        session, _, _ = _session(monkeypatch, rig)

        # A valid location goes with each bad value: it must not be written either.
        with pytest.raises(ValueError, match=message):
            session.set_pose_bone_transform(
                "Rig", "J1", **{"location": [1, 1, 1], **arguments}
            )

        assert list(bone.location) == [0.0, 0.0, 0.0]

    def test_refuses_what_is_not_a_pose_bone(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        rig, _, _ = _rig()
        session, _, _ = _session(monkeypatch, rig, FakeTransformObject("Cart"))
        with pytest.raises(ValueError, match="Armature 'Rig' has no bone 'J9'"):
            session.set_pose_bone_transform("Rig", "J9", location=[0, 0, 0])
        with pytest.raises(ValueError, match="'Cart' is not an armature"):
            session.set_pose_bone_transform("Cart", "J1", location=[0, 0, 0])

    def test_reports_where_blender_evaluates_the_bone(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        rig, _, _ = _rig()
        session, _, _ = _session(monkeypatch, rig)

        result = session.set_pose_bone_transform("Rig", "J1", location=[0, 0, 0])

        assert result["world_head"] == pytest.approx([1.0, 2.0, 4.0])
        assert result["world_tail"] == pytest.approx([1.0, 2.0, 5.0])


# -- declarations ---------------------------------------------------------------------


class TestDeclarations:
    def test_a_page_fits_the_server_budget_untouched(
        self, monkeypatch: pytest.MonkeyPatch, mathutils: Any
    ) -> None:
        """The server cuts a reply over its budget; a page must never need that."""
        assert blender_runtime.RESULT_PAGE_BYTES + 4000 <= DEFAULT_RESULT_BUDGET_BYTES
        cart, animate = _cart(123456.789)
        session, _, _ = _session(monkeypatch, cart, animate=animate)
        page = session.get_object_trajectory("Cart", 1, 5000)
        dumped = BlenderGetTrajectoryResponse(success=True, **page).model_dump()

        assert json.loads(encode_result_budget(dumped)) == dumped
        assert "rotation_euler" in dumped["points"][0]
        located = session.get_object_trajectory("Cart", 1, 5, fields=["location"])
        trimmed = BlenderGetTrajectoryResponse(success=True, **located).model_dump()
        # A field nobody asked for is left out, not reported as null.
        assert set(trimmed["points"][0]) == {"frame", "location"}

    def test_request_models_refuse_what_the_operations_refuse(self) -> None:
        with pytest.raises(ValidationError):
            BlenderGetKeyframesRequest(object_name="Rig", offset=-1)
        with pytest.raises(ValidationError, match="2 frames but 1 values"):
            BlenderInsertKeyframesRequest(
                object_name="Cube",
                channels=[
                    {"data_path": "location", "frames": [1, 2], "values": [[0, 0, 0]]}
                ],
            )
        with pytest.raises(ValidationError):
            BlenderGetTrajectoriesRequest(
                targets=[{"object_name": "Cube"}] * 33, start_frame=1, end_frame=2
            )
        with pytest.raises(ValidationError, match="must not be after"):
            BlenderGetTrajectoriesRequest(
                targets=[{"object_name": "Cube"}], start_frame=3, end_frame=2
            )
        with pytest.raises(ValidationError):
            BlenderGetTrajectoryRequest(
                object_name="Cube", start_frame=1, end_frame=2, fields=["speed"]
            )
        with pytest.raises(ValidationError, match="not both"):
            BlenderSetPoseBoneTransformRequest(
                object_name="Rig",
                bone_name="J1",
                rotation_euler=[0, 0, 0],
                rotation_quaternion=[1, 0, 0, 0],
            )

    def test_trajectory_fields_mirror_the_add_on_constant(self) -> None:
        assert get_args(BlenderTrajectoryField) == blender_runtime.TRAJECTORY_FIELDS
