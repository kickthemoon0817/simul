"""The work bound of bake_simulation, get_physics_state and the trajectory walk.

Round 2 of the Blender audit (M-1): the bound counted frames x bodies only,
so 300 bodies x 1000 frames at 100 substeps passed it and took 62.7 s headless
(an attached call timed out at 30 s while Blender kept baking for two more
minutes), and an 80x80 cloth over 1000 frames counted as one body and took
55.6 s. The estimate now prices every simulation a frame steps, at rates
measured on Blender 5.0.1 (the module's ``_*_SECONDS``).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List, Optional

import pytest

from simul.adapters import blender_runtime
from simul.adapters.blender_runtime import (
    SIMULATION_SECONDS_BUDGET,
    BlenderRuntimeSession,
    ValidationError,
)

Session = BlenderRuntimeSession

#: Seconds a frame of one body at Blender's default 10 substeps and 10 solver iterations.
BODY_FRAME = 10 * (
    blender_runtime._RIGID_BODY_SUBSTEP_SECONDS + 10 * blender_runtime._RIGID_BODY_ITERATION_SECONDS
)


@pytest.fixture
def session(monkeypatch: pytest.MonkeyPatch) -> Session:
    monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
    monkeypatch.setattr(
        blender_runtime,
        "bpy",
        SimpleNamespace(app=SimpleNamespace(version=(5, 0, 1))),
    )
    return Session(path_policy=object())


def _cache(baked: bool = False) -> SimpleNamespace:
    return SimpleNamespace(frame_start=1, frame_end=250, is_baked=baked, is_outdated=False, info="")


def _world(
    bodies: int, substeps: int = 10, iterations: int = 10, enabled: bool = True
) -> SimpleNamespace:
    return SimpleNamespace(
        enabled=enabled,
        substeps_per_frame=substeps,
        solver_iterations=iterations,
        collection=SimpleNamespace(all_objects=[object()] * bodies),
        point_cache=_cache(),
    )


def _mesh(name: str, vertices: int, *modifiers: Any, particle_systems: Any = ()) -> Any:
    return SimpleNamespace(
        name=name,
        type="MESH",
        data=SimpleNamespace(vertices=[None] * vertices),
        modifiers=list(modifiers),
        particle_systems=list(particle_systems),
    )


def _cloth(quality: int = 5, self_collision: bool = False, baked: bool = False) -> Any:
    return SimpleNamespace(
        type="CLOTH",
        name="Cloth",
        settings=SimpleNamespace(quality=quality),
        collision_settings=SimpleNamespace(use_self_collision=self_collision),
        point_cache=_cache(baked),
    )


def _soft_body(step_min: int = 10, self_collision: bool = False) -> Any:
    return SimpleNamespace(
        type="SOFT_BODY",
        name="Softbody",
        settings=SimpleNamespace(step_min=step_min, use_self_collision=self_collision),
        point_cache=_cache(),
    )


def _subsurf(levels: int) -> Any:
    return SimpleNamespace(type="SUBSURF", name="Subdivision", levels=levels, show_viewport=True)


def _scene(world: Optional[Any] = None, objects: Optional[List[Any]] = None) -> Any:
    return SimpleNamespace(rigidbody_world=world, objects=list(objects or []))


# -- the estimate ----------------------------------------------------------------


def test_rigid_bodies_cost_bodies_times_substeps_times_iterations(session: Session) -> None:
    default, parts = session._simulation_cost(_scene(_world(300)), "rigid_body")
    substeps, _ = session._simulation_cost(_scene(_world(300, substeps=100)), "rigid_body")
    iterations, _ = session._simulation_cost(_scene(_world(300, iterations=100)), "rigid_body")

    frame = blender_runtime._FRAME_SECONDS
    assert default == pytest.approx(frame + 300 * BODY_FRAME)
    assert substeps - frame == pytest.approx(10 * (default - frame))
    assert iterations - frame == pytest.approx(
        300 * 10 * (4.2e-6 + 100 * 0.11e-6)
    )
    assert parts == ["300 rigid bodies at 10 substeps and 10 solver iterations: 15.9 ms a frame"]


def test_a_disabled_world_costs_nothing(session: Session) -> None:
    seconds, parts = session._simulation_cost(
        _scene(_world(1000, enabled=False)), "rigid_body"
    )

    assert seconds == blender_runtime._FRAME_SECONDS
    assert parts == []


def test_cloth_costs_its_vertices_times_its_quality(session: Session) -> None:
    """An 80x80 grid (6 561 vertices) took 9.2 us a vertex a frame at quality 5."""
    q5, parts = session._simulation_cost(_scene(None, [_mesh("Sheet", 6561, _cloth())]), "all")
    q10, _ = session._simulation_cost(_scene(None, [_mesh("Sheet", 6561, _cloth(10))]), "all")
    selfish, _ = session._simulation_cost(
        _scene(None, [_mesh("Sheet", 6561, _cloth(self_collision=True))]), "all"
    )

    frame = blender_runtime._FRAME_SECONDS
    assert (q5 - frame) / 6561 == pytest.approx(1.2e-6 + 5 * 1.6e-6)
    assert (q10 - frame) / 6561 == pytest.approx(1.2e-6 + 10 * 1.6e-6)
    assert selfish - frame == pytest.approx(1.3 * (q5 - frame))
    assert parts == ["cloth 'Cloth' on 'Sheet' with 6561 vertices at quality 5: 60.4 ms a frame"]


def test_a_subdivision_before_a_cloth_multiplies_its_vertices(session: Session) -> None:
    before, parts = session._simulation_cost(
        _scene(None, [_mesh("Sheet", 100, _subsurf(2), _cloth())]), "all"
    )
    after, _ = session._simulation_cost(
        _scene(None, [_mesh("Sheet", 100, _cloth(), _subsurf(2))]), "all"
    )

    assert "with 1600 vertices" in parts[0]
    assert before - blender_runtime._FRAME_SECONDS == pytest.approx(
        16 * (after - blender_runtime._FRAME_SECONDS)
    )


def test_soft_bodies_particles_and_hair_dynamics_count(session: Session) -> None:
    hair = SimpleNamespace(
        name="Fur",
        settings=SimpleNamespace(type="HAIR", count=100, hair_step=5),
        use_hair_dynamics=True,
        cloth=SimpleNamespace(settings=SimpleNamespace(quality=5)),
        point_cache=_cache(),
    )
    static_hair = SimpleNamespace(
        name="Grass",
        settings=SimpleNamespace(type="HAIR", count=10_000, hair_step=5),
        use_hair_dynamics=False,
        point_cache=_cache(),
    )
    emitter = SimpleNamespace(
        name="Sparks", settings=SimpleNamespace(type="EMITTER", count=10_000), point_cache=_cache()
    )
    objects = [
        _mesh("Jelly", 1000, _soft_body(step_min=20, self_collision=True)),
        _mesh("Emitter", 4, particle_systems=[emitter, hair, static_hair]),
    ]

    seconds, parts = session._simulation_cost(_scene(None, objects), "all")

    assert [part.split(":")[0] for part in parts] == [
        "soft body 'Softbody' on 'Jelly' with 1000 vertices, at least 20 steps a frame with "
        "self-collision",
        "particle system 'Sparks' with 10000 particles",
        "hair dynamics 'Fur' with 600 points at quality 5",
    ]
    assert seconds == pytest.approx(
        blender_runtime._FRAME_SECONDS
        + 1000 * 5e-6 * 2 * 2.5
        + 10_000 * 0.1e-6
        + 600 * (1.2e-6 + 5 * 1.6e-6)
    )


def test_an_image_dynamic_paint_surface_has_no_point_cache_to_cost(session: Session) -> None:
    def canvas(surface_format: str) -> Any:
        return SimpleNamespace(
            type="DYNAMIC_PAINT",
            name="DynamicPaint",
            canvas_settings=SimpleNamespace(
                canvas_surfaces=[
                    SimpleNamespace(
                        name="Surface", surface_format=surface_format, point_cache=_cache()
                    )
                ]
            ),
        )

    vertex, parts = session._simulation_cost(
        _scene(None, [_mesh("Canvas", 10_000, canvas("VERTEX"))]), "all"
    )
    image, none = session._simulation_cost(
        _scene(None, [_mesh("Canvas", 10_000, canvas("IMAGE"))]), "all"
    )

    assert vertex - blender_runtime._FRAME_SECONDS == pytest.approx(10_000 * 0.5e-6)
    assert parts[0].startswith("dynamic paint surface 'Surface' on 'Canvas' with 10000 vertices")
    assert (image, none) == (blender_runtime._FRAME_SECONDS, [])


def test_a_rigid_body_bake_counts_the_unbaked_simulations_that_step_along(
    session: Session,
) -> None:
    """scope rigid_body steps the frames: an unbaked cloth simulates along, a baked one is read."""
    unbaked = _scene(_world(10), [_mesh("Sheet", 6561, _cloth())])
    baked = _scene(_world(10), [_mesh("Sheet", 6561, _cloth(baked=True))])

    stepping, parts = session._simulation_cost(unbaked, "rigid_body")
    reading, _ = session._simulation_cost(baked, "rigid_body")
    rebaking, _ = session._simulation_cost(baked, "all")

    assert len(parts) == 2
    assert reading == pytest.approx(blender_runtime._FRAME_SECONDS + 10 * BODY_FRAME)
    assert stepping == pytest.approx(rebaking)
    assert stepping > reading


# -- bake_simulation's refusal ----------------------------------------------------


def test_the_audit_bake_at_100_substeps_is_refused(session: Session) -> None:
    """300 cubes over a passive ground at 100 substeps took 62.7 s headless and passed."""
    scene = _scene(_world(301, substeps=100))

    with pytest.raises(ValidationError, match="allow_long_bake=true") as refused:
        session._check_bake_work(scene, "rigid_body", 1, 1000)

    message = str(refused.value)
    assert "Baking frames 1-1000 (1000 frames) is estimated at 159.6 s" in message
    assert "301 rigid bodies at 100 substeps and 10 solver iterations" in message
    assert "over this tool's limit of 16 s and 20000 frames" in message
    assert "BLENDER__CONNECTION_TIMEOUT" in message
    assert refused.value.details == {
        "code": "BakeOverBudget",
        "estimated_seconds": 159.6,
        "limit_seconds": SIMULATION_SECONDS_BUDGET,
        "frame_limit": 20_000,
        "frames_that_fit": 100,
    }
    assert "(about 100 frames fit)" in message


def test_the_round_one_bake_at_default_settings_still_runs(session: Session) -> None:
    """300 bodies x 1000 frames at 10 substeps took 7.6 s scattered; piled up 15 s."""
    session._check_bake_work(_scene(_world(301)), "rigid_body", 1, 1000)


def test_solver_iterations_alone_can_put_a_bake_over_the_limit(session: Session) -> None:
    session._check_bake_work(_scene(_world(200)), "rigid_body", 1, 1000)

    with pytest.raises(ValidationError, match="100 solver iterations"):
        session._check_bake_work(_scene(_world(200, iterations=100)), "rigid_body", 1, 1000)


def test_the_audit_cloth_is_refused_and_a_shorter_range_fits(session: Session) -> None:
    """An 80x80 cloth over 1000 frames counted as one body and took 55.6 s."""
    scene = _scene(None, [_mesh("Cloth", 6561, _cloth())])

    with pytest.raises(ValidationError) as refused:
        session._check_bake_work(scene, "all", 1, 1000)

    assert "cloth 'Cloth' on 'Cloth' with 6561 vertices at quality 5" in str(refused.value)
    fitting = refused.value.details["frames_that_fit"]
    assert fitting == 264
    session._check_bake_work(scene, "all", 1, fitting)


def test_a_cheap_bake_is_still_held_to_the_frame_limit(session: Session) -> None:
    with pytest.raises(ValidationError, match="20000 frames"):
        session._check_bake_work(_scene(_world(1)), "rigid_body", 1, 20_001)


def test_allow_long_bake_skips_the_estimate(session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    checked: List[Any] = []
    monkeypatch.setattr(Session, "_check_bake_work", lambda *args: checked.append(args))
    monkeypatch.setattr(Session, "_simulating_world", lambda self, scene, what: scene.rigidbody_world)
    monkeypatch.setattr(Session, "_bake_rigid_body_cache", lambda *args: None)
    monkeypatch.setattr(Session, "_bake_report", lambda self, scene, scope, warnings: {})
    scene = _scene(_world(301, substeps=100))
    scene.frame_current = 1
    scene.frame_set = lambda frame: None
    blender_runtime.bpy.context = SimpleNamespace(scene=scene)

    session.bake_simulation(1, 1000, allow_long_bake=True)
    session.bake_simulation(1, 1000)

    assert [args[1:] for args in checked] == [(scene, "rigid_body", 1, 1000)]


# -- stepping an unbaked simulation for get_physics_state -------------------------


def test_stepping_counts_only_the_frames_the_cache_does_not_hold(session: Session) -> None:
    world = _world(1000)
    scene = _scene(world)
    per_frame = blender_runtime._FRAME_SECONDS + 1000 * BODY_FRAME

    assert session._step_seconds(scene, world, 101) == pytest.approx(100 * per_frame)
    world.point_cache.info = "60 frames in memory (1.2 MiB)."
    assert session._step_seconds(scene, world, 101) == pytest.approx(41 * per_frame)
    world.point_cache.info = "200 frames in memory (4 MiB)."
    assert session._step_seconds(scene, world, 101) == 0.0
    # Past the cache end nothing more is simulated.
    world.point_cache.info = ""
    assert session._step_seconds(scene, world, 10_000) == pytest.approx(249 * per_frame)
