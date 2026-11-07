"""Rigid-body cache, bake scope and world settings against a stand-in Blender.

The stand-in behaves as Blender 5.0.1 did in the audit that found these
defects:

- an unbaked simulation advances only from one frame to the next, from its
  cache start, so a frame jump leaves bodies where they were last simulated;
- past the cache end Blender simulates nothing and holds the bodies at the
  end frame;
- ``ptcache.bake`` does nothing to a cache that is already baked, and given
  the rigid-body cache it bakes every cache in the scene;
- ``rigidbody.world_add`` makes the cache 1-250 whatever the scene range;
- writing a world setting marks the cache outdated, which drops the cached
  frames when the scene next steps from the cache start.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional

import pytest
from pydantic import ValidationError

from simul.adapters import blender_runtime
from simul.mcp.schemas.blender import (
    BlenderBakeSimulationRequest,
    BlenderBakeSimulationResponse,
    BlenderFreeBakeResponse,
    BlenderGetPhysicsStateResponse,
    BlenderRigidBodyWorldResponse,
    BlenderSetFrameRangeResponse,
    BlenderSetRigidBodyWorldRequest,
)

GRAVITY = 9.81
FPS = 24


class _Vector(list):
    @property
    def translation(self) -> "_Vector":
        return self


class _Matrix:
    def __init__(self, location: List[float]) -> None:
        self.translation = location

    def to_euler(self, order: str) -> List[float]:
        return [0.0, 0.0, 0.0]


class FakeCache:
    """A point cache: range, bake flag and the frames simulated into it."""

    def __init__(self, frame_start: int = 1, frame_end: int = 250) -> None:
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.is_baked = False
        self.is_outdated = True
        self.frames: set = set()

    @property
    def info(self) -> str:
        return f"{len(self.frames)} frames in memory"


class FakeWorld:
    """Blender's rigid body world, stepping one frame at a time like Bullet's."""

    def __init__(self, scene: "FakeScene") -> None:
        self._scene = scene
        self.enabled = True
        self.point_cache = FakeCache()
        self.collection: Optional[SimpleNamespace] = None
        self.constraints = None
        self.substeps_per_frame = 10
        self.solver_iterations = 10
        self.use_split_impulse = False
        self._time_scale = 1.0
        self.resets = 0
        self.last: Optional[int] = None
        # The frame whose simulated pose the bodies show; None is the rest pose.
        self.shown: Optional[int] = None

    @property
    def time_scale(self) -> float:
        return self._time_scale

    @time_scale.setter
    def time_scale(self, value: float) -> None:
        # rna_RigidBodyWorld_reset: any write marks the cache outdated.
        self._time_scale = value
        self.point_cache.is_outdated = True
        self.resets += 1

    def add(self, obj: Any) -> None:
        if self.collection is None:
            self.collection = SimpleNamespace(name="RigidBodyWorld", all_objects=[])
        self.collection.all_objects.append(obj)

    def advance(self, frame: int) -> None:
        cache = self.point_cache
        start, end = cache.frame_start, cache.frame_end
        if not self.enabled:
            self.shown = None
            return
        if cache.is_baked:
            self.shown = min(max(frame, start), end) if cache.frames else None
            return
        if frame <= start:
            self.last = start
            self.shown = None
            return
        target = min(frame, end)
        if target in cache.frames and not cache.is_outdated:
            self.last = target
            self.shown = target
            return
        if self.last is not None and target == self.last + 1:
            if cache.is_outdated and target == start + 1:
                cache.frames = {start}
                cache.is_outdated = False
            if not cache.is_outdated:
                cache.frames.add(target)
                self.last = target
                self.shown = target
        # Otherwise Blender cannot simulate this frame: the bodies stay put.


class FakeScene:
    def __init__(self, frame_start: int = 1, frame_end: int = 250) -> None:
        self.name = "Scene"
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.frame_current = frame_start
        self.render = SimpleNamespace(fps=FPS)
        self.gravity = (0.0, 0.0, -GRAVITY)
        self.use_gravity = True
        self.rigidbody_world: Optional[FakeWorld] = None
        self.objects: List[Any] = []
        self.frames: List[int] = []

    def frame_set(self, frame: int) -> None:
        self.frames.append(frame)
        self.frame_current = frame
        if self.rigidbody_world is not None:
            self.rigidbody_world.advance(frame)


def _body(scene: FakeScene, name: str, z0: float = 100.0, **rigid_body: Any) -> Any:
    settings = {
        "type": "ACTIVE",
        "kinematic": False,
        "enabled": True,
        "mass": 1.0,
        "collision_shape": "BOX",
    }
    settings.update(rigid_body)
    obj = SimpleNamespace(
        name=name,
        type="MESH",
        rotation_mode="XYZ",
        modifiers=[],
        particle_systems=[],
        rigid_body=SimpleNamespace(**settings),
    )

    def evaluated_get(depsgraph: Any) -> Any:
        world = scene.rigidbody_world
        z = z0
        if world is not None and world.shown is not None:
            t = (world.shown - world.point_cache.frame_start) / FPS
            z = z0 - 0.5 * GRAVITY * t * t
        return SimpleNamespace(matrix_world=_Matrix(_Vector([0.0, 0.0, z])))

    obj.evaluated_get = evaluated_get
    scene.objects.append(obj)
    if scene.rigidbody_world is not None:
        scene.rigidbody_world.add(obj)
    return obj


def _cloth(scene: FakeScene, name: str = "Cloth") -> FakeCache:
    cache = FakeCache()
    cache.is_outdated = False
    obj = SimpleNamespace(
        name=name,
        type="MESH",
        modifiers=[SimpleNamespace(type="CLOTH", name="Cloth", point_cache=cache)],
        particle_systems=[],
        rigid_body=None,
    )
    scene.objects.append(obj)
    return cache


def _z(frame: int, start: int = 1, z0: float = 100.0) -> float:
    t = (frame - start) / FPS
    return z0 - 0.5 * GRAVITY * t * t


class FakeBlender(SimpleNamespace):
    """bpy with the rigid-body and point-cache operators the session calls."""


def _fake_bpy(scene: FakeScene) -> FakeBlender:
    calls: List[Any] = []
    overrides: List[Dict[str, Any]] = []

    def all_caches() -> List[FakeCache]:
        caches = []
        if scene.rigidbody_world is not None:
            caches.append(scene.rigidbody_world.point_cache)
        for obj in scene.objects:
            caches.extend(m.point_cache for m in obj.modifiers if m.type == "CLOTH")
        return caches

    @contextmanager
    def temp_override(**kwargs: Any) -> Iterator[None]:
        overrides.append(kwargs)
        try:
            yield
        finally:
            overrides.pop()

    def current_cache() -> FakeCache:
        return overrides[-1]["point_cache"]

    def world_add() -> set:
        calls.append("rigidbody.world_add")
        scene.rigidbody_world = FakeWorld(scene)
        return {"FINISHED"}

    def world_remove() -> set:
        calls.append("rigidbody.world_remove")
        scene.rigidbody_world = None
        return {"FINISHED"}

    def bake(bake: bool = False) -> set:
        # Blender 5.0.1: skips a baked cache, and bakes every cache in the scene.
        calls.append("ptcache.bake")
        for cache in all_caches():
            if cache.is_baked:
                continue
            for frame in range(cache.frame_start, cache.frame_end + 1):
                scene.frame_set(frame)
            cache.frames = set(range(cache.frame_start, cache.frame_end + 1))
            cache.is_baked = True
        return {"FINISHED"}

    def bake_from_cache() -> set:
        calls.append("ptcache.bake_from_cache")
        current_cache().is_baked = True
        return {"FINISHED"}

    def free_bake() -> set:
        calls.append("ptcache.free_bake")
        current_cache().is_baked = False
        return {"FINISHED"}

    def bake_all(bake: bool = False) -> set:
        calls.append("ptcache.bake_all")
        for cache in all_caches():
            if not cache.is_baked:
                cache.frames = set(range(cache.frame_start, cache.frame_end + 1))
                cache.is_outdated = False
                cache.is_baked = True
        return {"FINISHED"}

    def free_bake_all() -> set:
        calls.append("ptcache.free_bake_all")
        for cache in all_caches():
            cache.is_baked = False
        return {"FINISHED"}

    return FakeBlender(
        calls=calls,
        app=SimpleNamespace(version=(5, 0, 1), version_string="5.0.1", background=True),
        data=SimpleNamespace(objects={}),
        context=SimpleNamespace(
            scene=scene,
            temp_override=temp_override,
            evaluated_depsgraph_get=lambda: "depsgraph",
        ),
        ops=SimpleNamespace(
            rigidbody=SimpleNamespace(world_add=world_add, world_remove=world_remove),
            ptcache=SimpleNamespace(
                bake=bake,
                bake_from_cache=bake_from_cache,
                free_bake=free_bake,
                bake_all=bake_all,
                free_bake_all=free_bake_all,
            ),
        ),
    )


@pytest.fixture
def blender(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Install a stand-in bpy over ``scene`` and return a scene-scoped session."""

    def install(scene: FakeScene) -> Any:
        fake = _fake_bpy(scene)
        monkeypatch.setattr(blender_runtime, "bpy", fake)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        session = blender_runtime.BlenderRuntimeSession(
            scene_scope=True, path_policy=object()
        )
        session._get_object_or_raise = lambda name: next(  # type: ignore[method-assign]
            o for o in scene.objects if o.name == name
        )
        return fake, session

    return install


def _world_scene(frame_end: int = 250) -> FakeScene:
    scene = FakeScene(1, frame_end)
    world = FakeWorld(scene)
    world.point_cache.frame_end = min(frame_end, 250)
    scene.rigidbody_world = world
    return scene


def _show_frame(session: Any, scene: FakeScene, frame: int) -> float:
    scene.frame_set(frame)
    return session.get_physics_state("Drop", step_simulation=False)["location"][2]


# -- B3-01: a re-bake bakes the new range ------------------------------------


def test_rebake_frees_the_old_bake_and_simulates_the_new_range(blender: Any) -> None:
    scene = _world_scene()
    _body(scene, "Drop")
    fake, session = blender(scene)

    first = session.bake_simulation(1, 100)
    second = session.bake_simulation(1, 200)

    cache = scene.rigidbody_world.point_cache
    assert (first["frame_end"], second["frame_end"]) == (100, 200)
    assert second["frame_count"] == 200
    assert (cache.frame_start, cache.frame_end, cache.is_baked) == (1, 200, True)
    assert "ptcache.free_bake" in fake.calls
    # The body falls past frame 100 instead of hanging where the first bake ended.
    assert _show_frame(session, scene, 150) == pytest.approx(_z(150))
    assert _show_frame(session, scene, 200) == pytest.approx(_z(200))


def test_shorter_rebake_leaves_no_frames_of_the_longer_one(blender: Any) -> None:
    scene = _world_scene()
    _body(scene, "Drop")
    _, session = blender(scene)

    session.bake_simulation(1, 100)
    result = session.bake_simulation(1, 60)

    assert (result["frame_start"], result["frame_end"]) == (1, 60)
    # Past the new end the body holds the frame-60 pose, as Blender does.
    assert _show_frame(session, scene, 80) == pytest.approx(_z(60))


def test_rebake_from_a_later_start_restarts_the_simulation(blender: Any) -> None:
    """Blender 5.0.1 read the old frames simulated from frame 1 without the invalidation."""
    scene = _world_scene()
    _body(scene, "Drop")
    _, session = blender(scene)

    session.bake_simulation(1, 200)
    session.free_bake()
    session.bake_simulation(10, 50)

    assert scene.rigidbody_world.resets >= 2
    assert _show_frame(session, scene, 30) == pytest.approx(_z(30, start=10))


def test_bake_returns_to_the_frame_it_started_on(blender: Any) -> None:
    scene = _world_scene()
    _body(scene, "Drop")
    _, session = blender(scene)
    scene.frame_set(37)

    session.bake_simulation(1, 100)

    assert scene.frame_current == 37
    assert scene.frames[-1] == 37


# -- B3-02: an inverted range is refused -------------------------------------


@pytest.mark.parametrize(("start", "end"), [(90, 10), (10, 10)])
def test_bake_refuses_an_empty_or_inverted_range(
    blender: Any, start: int, end: int
) -> None:
    scene = _world_scene()
    _body(scene, "Drop")
    fake, session = blender(scene)

    with pytest.raises(ValueError, match="must be less than"):
        session.bake_simulation(start, end)
    with pytest.raises(ValidationError, match="must be less than"):
        BlenderBakeSimulationRequest(frame_start=start, frame_end=end)

    assert (
        scene.rigidbody_world.point_cache.frame_start,
        scene.rigidbody_world.point_cache.frame_end,
    ) == (1, 250)
    assert fake.calls == []


def test_free_restores_a_cache_range_that_cannot_simulate(blender: Any) -> None:
    """A range inverted by a script (or an old add-on) is fitted to the scene again."""
    scene = _world_scene()
    _body(scene, "Drop")
    _, session = blender(scene)
    cache = scene.rigidbody_world.point_cache
    cache.frame_start, cache.frame_end = 90, 10

    result = session.free_bake()

    assert (cache.frame_start, cache.frame_end) == (1, 250)
    assert result["rigid_body_cache"]["frame_end"] == 250
    scene.frame_set(1)
    scene.frame_set(48)
    state = session.get_physics_state("Drop")
    assert state["location"][2] == pytest.approx(_z(48))


# -- B3-07: the cache follows the scene range --------------------------------


def test_free_gives_the_cache_the_scene_range_back(blender: Any) -> None:
    scene = _world_scene()
    _body(scene, "Drop")
    _, session = blender(scene)

    session.bake_simulation(1, 100)
    result = session.free_bake()

    cache = scene.rigidbody_world.point_cache
    assert (cache.frame_start, cache.frame_end, cache.is_baked) == (1, 250, False)
    assert [entry["kind"] for entry in result["freed"]] == ["RIGID_BODY"]
    assert result["freed"][0]["frame_end"] == 100
    scene.frame_set(200)
    state = session.get_physics_state("Drop")
    assert state["location"][2] == pytest.approx(_z(200))
    assert state["pose_source"] == "simulation"


def test_set_frame_range_extends_an_unbaked_cache(blender: Any) -> None:
    scene = _world_scene()
    _body(scene, "Drop")
    _, session = blender(scene)

    result = session.set_frame_range(1, 1000)

    assert (
        scene.rigidbody_world.point_cache.frame_start,
        scene.rigidbody_world.point_cache.frame_end,
    ) == (1, 1000)
    assert result["rigid_body_cache"]["frame_end"] == 1000
    assert result["warning"] is None
    BlenderSetFrameRangeResponse(success=True, **result)
    scene.frame_set(500)
    assert session.get_physics_state("Drop")["location"][2] == pytest.approx(_z(500))


def test_set_frame_range_warns_about_a_bake_it_cannot_extend(blender: Any) -> None:
    scene = _world_scene()
    _body(scene, "Drop")
    _, session = blender(scene)
    session.bake_simulation(1, 250)

    result = session.set_frame_range(1, 1000)

    assert (
        scene.rigidbody_world.point_cache.frame_end,
        scene.rigidbody_world.point_cache.is_baked,
    ) == (250, True)
    assert "baked for frames 1-250" in result["warning"]


def test_set_frame_range_without_a_world_reports_no_cache(blender: Any) -> None:
    scene = FakeScene()
    _, session = blender(scene)

    result = session.set_frame_range(1, 500)

    assert result["rigid_body_cache"] is None
    assert result["warning"] is None


def test_a_new_world_covers_the_scene_range(blender: Any) -> None:
    """world_add makes the cache 1-250; a 400-frame scene froze its bodies at 250."""
    scene = FakeScene(5, 400)
    _, session = blender(scene)

    result = session.set_rigid_body_world()

    cache = scene.rigidbody_world.point_cache
    assert (cache.frame_start, cache.frame_end) == (5, 400)
    assert result["created"] is True
    assert result["rigid_body_cache"]["covers_scene_range"] is True


# -- B3-06: physics state after a frame jump ---------------------------------


def test_physics_state_steps_to_a_jumped_frame(blender: Any) -> None:
    scene = _world_scene()
    _body(scene, "Drop", z0=1000.0)
    _, session = blender(scene)
    scene.frame_set(1)
    scene.frame_set(48)

    state = session.get_physics_state("Drop")

    assert state["location"][2] == pytest.approx(_z(48, z0=1000.0))
    assert state["pose_source"] == "simulation"
    assert state["simulated_for_frame"] is True
    assert state["stepped_frames"] == 48
    assert state["frame"] == 48
    assert scene.frame_current == 48
    BlenderGetPhysicsStateResponse(success=True, **state)


def test_physics_state_without_stepping_says_the_pose_is_unknown(blender: Any) -> None:
    scene = _world_scene()
    _body(scene, "Drop", z0=1000.0)
    _, session = blender(scene)
    scene.frame_set(1)
    scene.frame_set(48)

    state = session.get_physics_state("Drop", step_simulation=False)

    assert state["location"][2] == pytest.approx(1000.0)  # the rest pose
    assert state["pose_source"] == "unknown"
    assert state["simulated_for_frame"] is None
    assert "step_simulation is false" in state["warning"]


def test_physics_state_flags_frames_past_the_cache_end(blender: Any) -> None:
    scene = _world_scene(frame_end=250)
    _body(scene, "Drop")
    _, session = blender(scene)
    scene.frame_set(300)

    state = session.get_physics_state("Drop")

    assert state["pose_source"] == "past_cache_end"
    assert state["simulated_for_frame"] is False
    assert "cache end 250" in state["warning"]
    assert state["location"][2] == pytest.approx(_z(250))
    assert scene.frame_current == 300


def test_physics_state_reads_a_bake_without_stepping(blender: Any) -> None:
    scene = _world_scene()
    _body(scene, "Drop")
    _, session = blender(scene)
    session.bake_simulation(1, 100)
    scene.frames.clear()
    scene.frame_set(60)

    state = session.get_physics_state("Drop")

    assert state["pose_source"] == "baked"
    assert state["simulated_for_frame"] is True
    assert scene.frames == [60]


def test_physics_state_does_not_step_past_the_work_budget(
    blender: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = _world_scene()
    for index in range(10):
        _body(scene, "Drop" if index == 0 else f"B{index}")
    _, session = blender(scene)
    monkeypatch.setattr(blender_runtime.BlenderRuntimeSession, "_STEP_WORK_BUDGET", 400)
    scene.frame_set(100)
    scene.frames.clear()

    state = session.get_physics_state("Drop")

    assert scene.frames == []
    assert state["pose_source"] == "unknown"
    assert "exceeds 400 body-frames" in state["warning"]


@pytest.mark.parametrize(
    ("settings", "source"),
    [({"type": "PASSIVE"}, "evaluated"), ({"kinematic": True}, "evaluated")],
)
def test_physics_state_of_a_body_the_simulation_does_not_move(
    blender: Any, settings: Dict[str, Any], source: str
) -> None:
    scene = _world_scene()
    _body(scene, "Drop", **settings)
    _, session = blender(scene)
    scene.frame_set(40)
    scene.frames.clear()

    state = session.get_physics_state("Drop")

    assert state["pose_source"] == source
    assert state["simulated_for_frame"] is None
    assert scene.frames == []


def test_physics_state_in_a_disabled_world(blender: Any) -> None:
    scene = _world_scene()
    _body(scene, "Drop")
    scene.rigidbody_world.enabled = False
    _, session = blender(scene)
    scene.frame_set(40)

    state = session.get_physics_state("Drop")

    assert state["pose_source"] == "world_disabled"
    assert state["simulated_for_frame"] is False
    assert state["world_enabled"] is False


# -- B3-08: bake and free cover the same caches ------------------------------


def test_rigid_body_scope_bakes_and_frees_only_the_rigid_body_cache(
    blender: Any,
) -> None:
    scene = _world_scene()
    _body(scene, "Drop")
    cloth = _cloth(scene)
    fake, session = blender(scene)

    baked = session.bake_simulation(1, 20)
    assert (cloth.is_baked, scene.rigidbody_world.point_cache.is_baked) == (False, True)
    assert "ptcache.bake" not in fake.calls
    assert [entry["kind"] for entry in baked["caches"]] == ["RIGID_BODY"]

    freed = session.free_bake()
    assert [entry["kind"] for entry in freed["freed"]] == ["RIGID_BODY"]
    assert cloth.is_baked is False
    BlenderBakeSimulationResponse(success=True, **baked)
    BlenderFreeBakeResponse(success=True, **freed)


def test_all_scope_bakes_and_frees_every_cache(blender: Any) -> None:
    scene = _world_scene()
    _body(scene, "Drop")
    cloth = _cloth(scene)
    fake, session = blender(scene)

    baked = session.bake_simulation(1, 20, scope="all")

    assert (cloth.is_baked, scene.rigidbody_world.point_cache.is_baked) == (True, True)
    assert (cloth.frame_start, cloth.frame_end) == (1, 20)
    assert fake.calls.index("ptcache.free_bake_all") < fake.calls.index(
        "ptcache.bake_all"
    )
    assert {entry["kind"]: entry["frame_end"] for entry in baked["caches"]} == {
        "RIGID_BODY": 20,
        "CLOTH": 20,
    }

    freed = session.free_bake(scope="all")

    assert {entry["kind"] for entry in freed["freed"]} == {"RIGID_BODY", "CLOTH"}
    assert (cloth.is_baked, scene.rigidbody_world.point_cache.is_baked) == (
        False,
        False,
    )
    assert (cloth.frame_end, scene.rigidbody_world.point_cache.frame_end) == (250, 250)


def test_a_cloth_only_scene_bakes_with_scope_all(blender: Any) -> None:
    scene = FakeScene()
    cloth = _cloth(scene)
    _, session = blender(scene)

    with pytest.raises(ValueError, match="scope 'all'"):
        session.bake_simulation(1, 20)
    result = session.bake_simulation(1, 20, scope="all")

    assert cloth.is_baked is True
    assert result["frame_start"] is None and result["rigid_body_cache"] is None
    with pytest.raises(ValueError, match="scope 'all'"):
        session.free_bake()
    assert session.free_bake(scope="all")["freed"][0]["kind"] == "CLOTH"


def test_all_scope_with_nothing_to_bake_is_refused(blender: Any) -> None:
    _, session = blender(FakeScene())

    with pytest.raises(ValueError, match="Nothing to bake"):
        session.bake_simulation(1, 20, scope="all")


def test_an_unknown_scope_is_refused(blender: Any) -> None:
    scene = _world_scene()
    _, session = blender(scene)

    with pytest.raises(ValueError, match="Invalid scope"):
        session.bake_simulation(1, 20, scope="everything")
    with pytest.raises(ValueError, match="Invalid scope"):
        session.free_bake(scope="everything")


# -- B3-21: a disabled world is not baked ------------------------------------


def test_bake_refuses_a_disabled_world(blender: Any) -> None:
    scene = _world_scene()
    _body(scene, "Drop")
    scene.rigidbody_world.enabled = False
    fake, session = blender(scene)

    with pytest.raises(ValueError, match="disabled"):
        session.bake_simulation(1, 20)

    assert scene.rigidbody_world.point_cache.is_baked is False
    assert fake.calls == []


def test_all_scope_leaves_a_disabled_world_unbaked(blender: Any) -> None:
    scene = _world_scene()
    _body(scene, "Drop")
    _cloth(scene)
    scene.rigidbody_world.enabled = False
    _, session = blender(scene)

    result = session.bake_simulation(1, 20, scope="all")

    assert scene.rigidbody_world.point_cache.is_baked is False
    assert any("disabled" in warning for warning in result["warnings"])


# -- The rigid body world tool -----------------------------------------------


def test_world_report_without_a_world(blender: Any) -> None:
    _, session = blender(FakeScene())

    report = session.get_rigid_body_world()

    assert report["exists"] is False
    assert report["gravity"] == [0.0, 0.0, -GRAVITY]
    assert "no rigid body world" in report["warnings"][0]
    BlenderRigidBodyWorldResponse(success=True, **report)


def test_world_report_flags_a_cache_short_of_the_scene_range(blender: Any) -> None:
    scene = _world_scene(frame_end=1000)
    _body(scene, "Drop")
    _, session = blender(scene)

    report = session.get_rigid_body_world()

    assert report["body_count"] == 1
    assert report["rigid_body_cache"]["frame_end"] == 250
    assert report["rigid_body_cache"]["covers_scene_range"] is False
    assert any("not the whole scene range 1-1000" in w for w in report["warnings"])
    BlenderRigidBodyWorldResponse(success=True, **report)


def test_set_world_changes_only_the_given_settings(blender: Any) -> None:
    scene = _world_scene()
    world = scene.rigidbody_world
    _, session = blender(scene)

    report = session.set_rigid_body_world(
        gravity=[0.0, 0.0, -1.62],
        substeps_per_frame=20,
        cache_frame_end=400,
    )

    assert scene.gravity == [0.0, 0.0, -1.62]
    assert (world.substeps_per_frame, world.solver_iterations) == (20, 10)
    assert world.point_cache.frame_end == 400
    assert world.point_cache.is_outdated is True
    assert report["created"] is False
    assert report["substeps_per_frame"] == 20


def test_set_world_refuses_a_new_range_for_a_baked_cache(blender: Any) -> None:
    scene = _world_scene()
    _body(scene, "Drop")
    _, session = blender(scene)
    session.bake_simulation(1, 100)

    with pytest.raises(ValueError, match="free it with free_blender_bake"):
        session.set_rigid_body_world(cache_frame_end=200)
    report = session.set_rigid_body_world(time_scale=0.5)

    assert any("playback still reads the bake" in w for w in report["warnings"])


@pytest.mark.parametrize(
    "arguments",
    [
        {"cache_frame_start": 50, "cache_frame_end": 50},
        {"cache_frame_start": 300},
        {"gravity": [0.0, float("nan"), 0.0]},
    ],
)
def test_set_world_refuses_invalid_settings_before_changing_anything(
    blender: Any, arguments: Dict[str, Any]
) -> None:
    scene = FakeScene()
    _, session = blender(scene)

    with pytest.raises(ValueError):
        session.set_rigid_body_world(**arguments)

    assert scene.rigidbody_world is None


@pytest.mark.parametrize(
    "arguments",
    [
        {"cache_frame_start": 50, "cache_frame_end": 50},
        {"gravity": [0.0, float("inf"), 0.0]},
        {"substeps_per_frame": 0},
        {"solver_iterations": 1001},
        {"time_scale": -1.0},
        {"cache_frame_end": 0},
    ],
)
def test_world_request_bounds(arguments: Dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        BlenderSetRigidBodyWorldRequest(**arguments)
