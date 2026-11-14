"""The work bound of bake_simulation, get_physics_state and the trajectory walk.

Round 2 of the Blender audit (M-1): the bound counted frames x bodies only,
so 300 bodies x 1000 frames at 100 substeps passed it and took 62.7 s headless
(an attached call timed out at 30 s while Blender kept baking for two more
minutes), and an 80x80 cloth over 1000 frames counted as one body and took
55.6 s. The estimate now prices every simulation a frame steps, at rates
measured on Blender 5.0.1 (the module's ``_*_SECONDS``).

Round 3 (R3-BL-02): it charged every body the piled rate and every mesh
vertex of a convex hull, and costed a passive MESH by its triangles, so a
6-link hinge arm that bakes in 0.04 s was refused at 38 s. Bodies now cost by
what their bounding boxes say they touch and by their hull's own vertices, a
passive MESH by the triangles under each body, and a call refuses only work
estimated at 3 times its budget, since a 22 s deadline stops the rest.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any, List, Optional, Sequence, Tuple

import pytest

from simul.adapters import blender_runtime
from simul.adapters.blender_runtime import (
    SIMULATION_SECONDS_BUDGET,
    BlenderRuntimeSession,
    ValidationError,
)

Session = BlenderRuntimeSession

#: Seconds a frame of one piled body at Blender's default 10 substeps and 10 solver iterations.
BODY_FRAME = 10 * (
    blender_runtime._RIGID_BODY_SUBSTEP_SECONDS + 10 * blender_runtime._RIGID_BODY_ITERATION_SECONDS
)
#: The same for a lone body (resting on, or falling onto, a static one).
LONE_FRAME = 10 * (
    blender_runtime._LONE_BODY_SUBSTEP_SECONDS + 10 * blender_runtime._LONE_BODY_ITERATION_SECONDS
)
FRAME = blender_runtime._FRAME_SECONDS
#: A cloth's contact when it collides with nothing, against the rates' own light contact.
ALONE = 1 / blender_runtime._CLOTH_RATES_CONTACT
IDENTITY = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


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
    """A world of ``bodies`` bodies the estimate knows nothing of: piled cubes."""
    return SimpleNamespace(
        enabled=enabled,
        substeps_per_frame=substeps,
        solver_iterations=iterations,
        collection=SimpleNamespace(all_objects=[object()] * bodies),
        point_cache=_cache(),
    )


class _Polygons(list):
    """A mesh's polygons with ``foreach_get("area")``, as bpy_prop_collection has."""

    def __init__(self, count: int, area: float) -> None:
        super().__init__([None] * count)
        self.area = area

    def foreach_get(self, name: str, out: Any) -> None:
        assert name == "area"
        for index in range(len(self)):
            out[index] = self.area / len(self)


def _corners(low: Sequence[float], high: Sequence[float]) -> List[Tuple[float, float, float]]:
    return [(x, y, z) for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])]


def _mesh(
    name: str,
    vertices: int,
    *modifiers: Any,
    particle_systems: Any = (),
    box: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    faces: int = 0,
    area: float = 0.0,
) -> Any:
    """An object with a mesh; ``box`` gives it a bounding box (world space, unrotated)."""
    obj = SimpleNamespace(
        name=name,
        type="MESH",
        data=SimpleNamespace(
            vertices=[None] * vertices,
            polygons=_Polygons(faces, area) if area else [None] * faces,
        ),
        modifiers=list(modifiers),
        particle_systems=list(particle_systems),
    )
    if box is not None:
        obj.bound_box = _corners(*box)
        obj.matrix_world = IDENTITY
    return obj


def _cloth(
    quality: int = 5, self_collision: bool = False, baked: bool = False, use_collision: bool = True
) -> Any:
    return SimpleNamespace(
        type="CLOTH",
        name="Cloth",
        settings=SimpleNamespace(quality=quality),
        collision_settings=SimpleNamespace(
            use_self_collision=self_collision, use_collision=use_collision, distance_min=0.015
        ),
        point_cache=_cache(baked),
    )


def _soft_body(step_min: int = 10, self_collision: bool = False) -> Any:
    return SimpleNamespace(
        type="SOFT_BODY",
        name="Softbody",
        settings=SimpleNamespace(step_min=step_min, use_self_collision=self_collision),
        point_cache=_cache(),
    )


def _collision() -> Any:
    return SimpleNamespace(type="COLLISION", name="Collision")


def _subsurf(levels: int) -> Any:
    return SimpleNamespace(type="SUBSURF", name="Subdivision", levels=levels, show_viewport=True)


def _scene(world: Optional[Any] = None, objects: Optional[List[Any]] = None) -> Any:
    return SimpleNamespace(rigidbody_world=world, objects=list(objects or []))


# -- the estimate ----------------------------------------------------------------


def test_rigid_bodies_cost_bodies_times_substeps_times_iterations(session: Session) -> None:
    default, parts = session._simulation_cost(_scene(_world(300)), "rigid_body")
    substeps, _ = session._simulation_cost(_scene(_world(300, substeps=100)), "rigid_body")
    iterations, _ = session._simulation_cost(_scene(_world(300, iterations=100)), "rigid_body")

    assert default == pytest.approx(FRAME + 300 * BODY_FRAME)
    assert substeps - FRAME == pytest.approx(10 * (default - FRAME))
    assert iterations - FRAME == pytest.approx(300 * 10 * (4.2e-6 + 100 * 0.11e-6))
    assert parts == [
        "300 rigid bodies at 10 substeps and 10 solver iterations (300 moving, 300 of them "
        "touching another moving body): 15.9 ms a frame"
    ]


def test_a_disabled_world_costs_nothing(session: Session) -> None:
    seconds, parts = session._simulation_cost(
        _scene(_world(1000, enabled=False)), "rigid_body"
    )

    assert seconds == FRAME
    assert parts == []


def test_cloth_costs_its_vertices_times_its_quality(session: Session) -> None:
    """An 80x80 grid (6 561 vertices) took 9.2 us a vertex a frame at quality 5 with a quarter
    of it on a sphere, 0.44 of that without a collider."""
    q5, parts = session._simulation_cost(_scene(None, [_mesh("Sheet", 6561, _cloth())]), "all")
    q10, _ = session._simulation_cost(_scene(None, [_mesh("Sheet", 6561, _cloth(10))]), "all")
    selfish, _ = session._simulation_cost(
        _scene(None, [_mesh("Sheet", 6561, _cloth(self_collision=True))]), "all"
    )

    assert (q5 - FRAME) / 6561 == pytest.approx((1.2e-6 + 5 * 1.6e-6) * ALONE)
    assert (q10 - FRAME) / 6561 == pytest.approx((1.2e-6 + 10 * 1.6e-6) * ALONE)
    assert selfish - FRAME == pytest.approx(1.3 * (q5 - FRAME))
    assert parts == ["cloth 'Cloth' on 'Sheet' with 6561 vertices at quality 5: 26.3 ms a frame"]


def test_a_subdivision_before_a_cloth_multiplies_its_vertices(session: Session) -> None:
    before, parts = session._simulation_cost(
        _scene(None, [_mesh("Sheet", 100, _subsurf(2), _cloth())]), "all"
    )
    after, _ = session._simulation_cost(
        _scene(None, [_mesh("Sheet", 100, _cloth(), _subsurf(2))]), "all"
    )

    assert "with 1600 vertices" in parts[0]
    assert before - FRAME == pytest.approx(16 * (after - FRAME))


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
        FRAME + 1000 * 5e-6 * 2 * 2.5 + 10_000 * 0.1e-6 + 600 * (1.2e-6 + 5 * 1.6e-6)
    )


def test_a_fluid_domains_own_particle_systems_cost_nothing(session: Session) -> None:
    """Round-3 audit (m-2): a liquid domain's FLIP system was listed as a 0-cost line."""
    flip = SimpleNamespace(
        name="Liquid", settings=SimpleNamespace(type="FLIP", count=0), point_cache=_cache()
    )
    empty = SimpleNamespace(
        name="Nothing", settings=SimpleNamespace(type="EMITTER", count=0), point_cache=_cache()
    )

    seconds, parts = session._simulation_cost(
        _scene(None, [_mesh("Domain", 8, particle_systems=[flip, empty])]), "all"
    )

    assert (seconds, parts) == (FRAME, [])


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

    assert vertex - FRAME == pytest.approx(10_000 * 0.5e-6)
    assert parts[0].startswith("dynamic paint surface 'Surface' on 'Canvas' with 10000 vertices")
    assert (image, none) == (FRAME, [])


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
    assert reading == pytest.approx(FRAME + 10 * BODY_FRAME)
    assert stepping == pytest.approx(rebaking)
    assert stepping > reading


# -- bake_simulation's refusal ----------------------------------------------------


def test_the_audit_bake_at_100_substeps_is_refused(session: Session) -> None:
    """300 cubes over a passive ground at 100 substeps took 62.7 s headless and passed."""
    scene = _scene(_world(301, substeps=100))

    with pytest.raises(ValidationError, match="allow_long_bake=true") as refused:
        session._check_bake_work(scene, "rigid_body", 1, 1000)

    message = str(refused.value)
    estimated = round(1000 * session._simulation_cost(scene, "rigid_body")[0], 1)
    assert estimated == pytest.approx(159.6, abs=0.1)
    assert f"Baking frames 1-1000 (1000 frames) is estimated at {estimated} s" in message
    assert "301 rigid bodies at 100 substeps and 10 solver iterations" in message
    assert "more than 3 times the 12 s a call is meant to take" in message
    assert "stops after 22 s with the frames simulated so far" in message
    # The client timeout is the server's setting: an agent cannot raise it.
    assert "BLENDER__CONNECTION_TIMEOUT, a server setting" in message
    assert "raise" not in message
    assert refused.value.details == {
        "code": "BakeOverBudget",
        "estimated_seconds": estimated,
        "limit_seconds": 3 * SIMULATION_SECONDS_BUDGET,
        "budget_seconds": SIMULATION_SECONDS_BUDGET,
        "frame_limit": 20_000,
        "frames_that_fit": 75,
    }
    assert "(about 75 frames fit)" in message


def test_a_bake_estimated_up_to_three_times_the_budget_goes_ahead(session: Session) -> None:
    """300 piled bodies x 1000 frames at 10 substeps took 16.2 s headless: over the 12 s a call
    is meant to take, but the 22 s deadline stops a bake the estimate underrates, so only an
    estimate over 36 s (2 253 frames here) is refused, with the 751 frames that take 12 s."""
    session._check_bake_work(_scene(_world(301)), "rigid_body", 1, 2253)
    with pytest.raises(ValidationError, match="about 751 frames fit"):
        session._check_bake_work(_scene(_world(301)), "rigid_body", 1, 2254)


def test_solver_iterations_alone_can_put_a_bake_over_the_limit(session: Session) -> None:
    session._check_bake_work(_scene(_world(200)), "rigid_body", 1, 2000)

    with pytest.raises(ValidationError, match="100 solver iterations"):
        session._check_bake_work(_scene(_world(200, iterations=100)), "rigid_body", 1, 2000)


def _sheet(subdivisions: int, size: float = 4.0, z: float = 2.0, **cloth: Any) -> Any:
    """A square cloth grid of ``subdivisions`` cuts and ``size`` metres, at height ``z``."""
    half = size / 2
    side = subdivisions + 1
    return _mesh(
        "Sheet",
        side * side,
        _cloth(**cloth),
        box=((-half, -half, z), (half, half, z)),
        faces=subdivisions * subdivisions,
        area=size * size,
    )


def _ball(faces: int, radius: float = 0.8, area: float = 8.04) -> Any:
    return _mesh(
        "Ball",
        0,
        _collision(),
        box=((-radius,) * 3, (radius,) * 3),
        faces=faces,
        area=area,
    )


def test_the_audit_cloth_is_refused_and_a_shorter_range_fits(session: Session) -> None:
    """An 80x80 cloth over 1000 frames counted as one body and took 55.6 s; it falls on a
    512-face sphere, the contact the cloth rates were measured with."""
    scene = _scene(None, [_sheet(80), _ball(512)])

    with pytest.raises(ValidationError) as refused:
        session._check_bake_work(scene, "all", 1, 1000)

    assert "cloth 'Cloth' on 'Sheet' with 6561 vertices at quality 5:" in str(refused.value)
    fitting = refused.value.details["frames_that_fit"]
    assert fitting == 198
    session._check_bake_work(scene, "all", 1, fitting)


def test_a_cheap_bake_is_still_held_to_the_frame_limit(session: Session) -> None:
    with pytest.raises(ValidationError, match="20000 frames") as refused:
        session._check_bake_work(_scene(_world(1)), "rigid_body", 1, 20_001)
    assert "cut short" not in str(refused.value)


def test_allow_long_bake_skips_the_estimate(session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    checked: List[Any] = []
    monkeypatch.setattr(Session, "_check_bake_work", lambda *args: checked.append(args))
    monkeypatch.setattr(Session, "_simulating_world", lambda self, scene, what: scene.rigidbody_world)
    monkeypatch.setattr(Session, "_bake_rigid_body_cache", lambda *args: args[4])
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
    per_frame = FRAME + 1000 * BODY_FRAME

    assert session._step_seconds(scene, world, 101) == pytest.approx(100 * per_frame)
    world.point_cache.info = "60 frames in memory (1.2 MiB)."
    assert session._step_seconds(scene, world, 101) == pytest.approx(41 * per_frame)
    world.point_cache.info = "200 frames in memory (4 MiB)."
    assert session._step_seconds(scene, world, 101) == 0.0
    # Past the cache end nothing more is simulated.
    world.point_cache.info = ""
    assert session._step_seconds(scene, world, 10_000) == pytest.approx(249 * per_frame)


# -- collision shapes -------------------------------------------------------------


def _body(
    shape: str,
    vertices: int = 8,
    loops: int = 0,
    faces: int = 0,
    box: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    kind: str = "ACTIVE",
    **extra: Any,
) -> Any:
    data = SimpleNamespace(
        vertices=[None] * vertices,
        loops=[None] * loops,
        polygons=[None] * faces,
    )
    data.as_pointer = lambda: id(data)
    obj = SimpleNamespace(
        name=extra.get("name", "Body"),
        rigid_body=SimpleNamespace(
            type=kind,
            enabled=True,
            kinematic=False,
            collision_shape=shape,
            mesh_source=extra.get("source", "BASE"),
        ),
        data=data,
        modifiers=extra.get("modifiers", []),
    )
    if box is not None:
        obj.bound_box = _corners(*box)
        obj.matrix_world = IDENTITY
    return obj


@pytest.mark.parametrize(
    ("body", "piled", "lone"),
    [
        (_body("CONVEX_HULL", 8), 1.0, 1.0),
        (_body("CONVEX_HULL", 482), 482 / 14, 1 + 474 / 48),
        (_body("CONVEX_HULL", 162), 162 / 14, 1 + 154 / 48),
        (_body("SPHERE", 482), 1.0, 1.0),
        (_body("BOX", 100_000), 0.5, 1.0),
        # A 32x16 UV sphere: 448 quads and 64 triangles, 1 984 loops, so 960 triangles.
        (_body("MESH", loops=1984, faces=512), (1984 - 1024) / 48, (1984 - 1024) / 48),
        (_body("MESH", loops=24, faces=6), 1.0, 1.0),
        # mesh_source FINAL counts the modifiers; BASE does not.
        (
            _body("CONVEX_HULL", 8, source="FINAL", modifiers=[_subsurf(3)]),
            8 * 64 / 14,
            1 + (8 * 64 - 8) / 48,
        ),
        (_body("CONVEX_HULL", 8, modifiers=[_subsurf(3)]), 1.0, 1.0),
    ],
    ids=["cube", "uv-sphere", "ico3", "sphere-shape", "box-shape", "mesh-sphere", "mesh-cube",
         "final-subsurf", "base-subsurf"],
)
def test_a_collision_shape_costs_by_its_hull_or_triangles(
    body: Any, piled: float, lone: float
) -> None:
    """Piled 482-vertex hulls ran 25 times a piled cube, scattered ones 11.6 times a scattered
    one; 960-triangle MESH shapes 54 us piled and 131 us scattered (GImpact either way)."""
    assert Session._body_shape_units(body, {}, piled=True) == pytest.approx(piled)
    assert Session._body_shape_units(body, {}, piled=False) == pytest.approx(lone)


def _fake_bmesh(monkeypatch: pytest.MonkeyPatch, made: List[Any]) -> None:
    """A ``bmesh`` whose hull keeps the vertices flagged 1 or 2 and dissolves all but the 2s.

    A vertex is ``(x, y, z, kind)``: 0 inside, 1 on a face or an edge of the hull, 2 a corner.
    """

    class BMVert:
        def __init__(self, co: Any) -> None:
            self.co = co

    class Verts(list):
        def new(self, co: Any) -> BMVert:
            vertex = BMVert(co)
            self.append(vertex)
            return vertex

    class BMesh:
        def __init__(self) -> None:
            self.verts: Verts = Verts()
            self.edges: List[Any] = []
            made.append(self)

        def from_mesh(self, data: Any) -> None:
            for co in data.vertices:
                self.verts.new(co)

        def free(self) -> None:
            pass

    def convex_hull(mesh: Any, input: Any) -> Any:
        return {"geom": [vertex for vertex in input if vertex.co[3]]}

    def dissolve_limit(mesh: Any, angle_limit: float, verts: Any, edges: Any) -> None:
        mesh.verts[:] = [vertex for vertex in mesh.verts if vertex.co[3] == 2]

    module = ModuleType("bmesh")
    module.new = BMesh  # type: ignore[attr-defined]
    module.types = SimpleNamespace(BMVert=BMVert)  # type: ignore[attr-defined]
    module.ops = SimpleNamespace(  # type: ignore[attr-defined]
        convex_hull=convex_hull, dissolve_limit=dissolve_limit
    )
    monkeypatch.setitem(sys.modules, "bmesh", module)


def test_a_hull_counts_the_hull_s_corners_once_per_mesh(monkeypatch: pytest.MonkeyPatch) -> None:
    """Round-3 audit (R3-BL-02): Blender builds a hull shape from the hull's own points, which
    a 64-segment cylinder link of 5 760 vertices has 128 of; all of them were counted."""
    made: List[Any] = []
    _fake_bmesh(monkeypatch, made)
    corners = [(0.0, 0.0, 0.0, 2)] * 128
    on_faces = [(0.0, 0.0, 0.0, 1)] * 208
    inside = [(0.0, 0.0, 0.0, 0)] * (5760 - 128 - 208)
    data = SimpleNamespace(vertices=corners + on_faces + inside, polygons=[None] * 5000)
    data.as_pointer = lambda: 7

    assert Session._hull_vertices(data) == 128
    assert Session._hull_vertices(data) == 128
    assert len(made) == 2  # one mesh and one hull, the first time only
    # A mesh without bmesh, too large, or a cube counts its vertices.
    monkeypatch.setitem(sys.modules, "bmesh", None)
    data.as_pointer = lambda: 8
    assert Session._hull_vertices(data) == 5760


def test_bodies_sharing_a_mesh_are_costed_once(session: Session) -> None:
    sphere = _body("CONVEX_HULL", 482)
    world = _world(0)
    world.collection.all_objects = [sphere] * 300
    seconds, parts = session._simulation_cost(_scene(world), "rigid_body")

    assert seconds - FRAME == pytest.approx(300 * (482 / 14) * BODY_FRAME)
    assert parts[0].startswith(
        "300 rigid bodies at 10 substeps and 10 solver iterations (300 moving, 300 of them "
        "touching another moving body, their collision shapes costing as 10329 lone cubes)"
    )


# -- what the bodies touch ----------------------------------------------------------


def _cube_at(x: float, z: float = 0.25, name: str = "Cube", kind: str = "ACTIVE") -> Any:
    return _body(
        "CONVEX_HULL", 8, box=((x - 0.25, -0.25, z - 0.25), (x + 0.25, 0.25, z + 0.25)),
        kind=kind, name=name,
    )


def _rigid_world(bodies: List[Any], constraints: Sequence[Any] = ()) -> Any:
    world = _world(0)
    world.collection.all_objects = list(bodies)
    world.constraints = SimpleNamespace(all_objects=list(constraints))
    return world


def test_a_body_touching_no_other_moving_body_costs_the_lone_rate(session: Session) -> None:
    """300 scattered cubes over 1000 frames took 5.4 s, estimated at 16 s as piles."""
    pile = [_cube_at(0.0, name="A"), _cube_at(0.52, name="B")]
    alone = _cube_at(5.0, name="Alone")
    ground = _body("BOX", box=((-50, -50, -0.1), (50, 50, 0.0)), kind="PASSIVE", name="Ground")

    seconds, parts = session._simulation_cost(_scene(_rigid_world([*pile, alone, ground])), "rigid_body")

    assert seconds - FRAME == pytest.approx(2 * BODY_FRAME + LONE_FRAME)
    assert "(3 moving, 2 of them touching another moving body)" in parts[0]


def test_bodies_a_constraint_keeps_apart_do_not_pile(session: Session) -> None:
    """A hinge arm's links nearly touch, but their joint disables their collisions: the
    round-3 audit's 6-link arm baked 250 frames in 0.04 s and was refused at 38 s."""
    links = [_cube_at(0.52 * i, name=f"Link{i}") for i in range(7)]
    links[0].rigid_body.type = "PASSIVE"
    joints = [
        SimpleNamespace(
            name=f"Joint{i}",
            rigid_body_constraint=SimpleNamespace(
                object1=links[i - 1], object2=links[i], disable_collisions=True, enabled=True
            ),
        )
        for i in range(1, 7)
    ]
    scene = _scene(_rigid_world(links, joints))

    seconds, parts = session._simulation_cost(scene, "rigid_body")

    joint_frame = 10 * blender_runtime._CONSTRAINT_ITERATION_SECONDS * 10
    assert seconds - FRAME == pytest.approx(6 * LONE_FRAME + 6 * joint_frame)
    assert "(6 moving, none of them touching another moving body, 6 constraints)" in parts[0]
    session._check_bake_work(scene, "rigid_body", 1, 250)
    for joint in joints:
        joint.rigid_body_constraint.disable_collisions = False
    piled, _ = session._simulation_cost(scene, "rigid_body")
    assert piled - FRAME == pytest.approx(6 * BODY_FRAME + 6 * joint_frame)


def test_a_passive_mesh_costs_the_triangles_under_each_body(session: Session) -> None:
    """20 cubes on an 80 000-triangle terrain took 0.65 s over 250 frames, estimated at 19 s
    by the terrain's triangles; each cube touches its share of them (5.6 here, at most 64)."""
    terrain = _body(
        "MESH", loops=160_000, faces=40_000, box=((-30, -30, -0.1), (30, 30, 0.0)),
        kind="PASSIVE", name="Terrain",
    )
    cubes = [_cube_at(2.0 * i - 19, name=f"C{i}") for i in range(20)]

    seconds, parts = session._simulation_cost(_scene(_rigid_world([terrain, *cubes])), "rigid_body")

    under = 80_000 * 0.25 / 3600
    assert seconds - FRAME == pytest.approx(20 * under * LONE_FRAME)
    assert "20 over a passive MESH shape" in parts[0]
    fine = _body(
        "MESH", loops=16_000_000, faces=4_000_000, box=((-30, -30, -0.1), (30, 30, 0.0)),
        kind="PASSIVE", name="Fine",
    )
    capped, _ = session._simulation_cost(_scene(_rigid_world([fine, cubes[0]])), "rigid_body")
    assert capped - FRAME == pytest.approx(64 * LONE_FRAME)


# -- cloth contact ------------------------------------------------------------------


def test_array_and_mirror_before_a_cloth_multiply_its_vertices(session: Session) -> None:
    """An Array of 16 before a cloth ran 9 times the estimate."""
    array = SimpleNamespace(type="ARRAY", count=16, show_viewport=True)
    mirror = SimpleNamespace(type="MIRROR", use_axis=(True, True, False), show_viewport=True)

    _, parts = session._simulation_cost(
        _scene(None, [_mesh("Sheet", 441, array, mirror, _cloth())]), "all"
    )

    assert "with 28224 vertices" in parts[0]


def test_a_dense_collider_under_a_cloth_raises_its_cost(session: Session) -> None:
    """A 40x40 cloth over a 32 768-face sphere took 4.1 times the cloth rates."""
    light, _ = session._simulation_cost(_scene(None, [_sheet(40), _ball(512)]), "all")
    dense, parts = session._simulation_cost(_scene(None, [_sheet(40), _ball(32_768)]), "all")

    rates = 1681 * (1.2e-6 + 5 * 1.6e-6)
    reach = (16 / 1681) ** 0.5 + 0.03

    def contact(faces: int) -> float:
        reached = max(1.0, reach / (8.04 / faces) ** 0.5)
        return (1 + 0.24 * (1.4 + 4.0 * reached)) / blender_runtime._CLOTH_RATES_CONTACT

    # The rates' own collider: about one face a vertex, a quarter of the cloth on it.
    assert light - FRAME == pytest.approx(rates * contact(512))
    assert contact(512) == pytest.approx(1.0, abs=0.01)
    assert dense - FRAME == pytest.approx(rates * contact(32_768))
    assert contact(32_768) == pytest.approx(3.99, abs=0.01)
    assert "draping 'Ball' (4.0 times the cost of light contact)" in parts[0]


def test_a_cloth_draping_a_body_costs_as_the_audit_measured(session: Session) -> None:
    """Round-3 audit (R3-BL-01): a 2 500-vertex cloth draping a 21 634-face body over 164
    frames took 31.3 s where the estimate said 12 s; it now says 26.4 s, and refuses 250
    frames (51 s) up front."""
    cloth = _sheet(49, size=1.2, z=0.9)
    body = _mesh(
        "Body", 0, _collision(), box=((-0.3, -0.3, -0.8), (0.3, 0.3, 0.8)), faces=21_634,
        area=3.24,
    )
    scene = _scene(None, [cloth, body])

    per_frame, parts = session._simulation_cost(scene, "all")

    assert 164 * per_frame == pytest.approx(26.4, abs=0.1)
    assert "draping 'Body' (7.0 times the cost of light contact)" in parts[0]
    with pytest.raises(ValidationError) as refused:
        session._check_bake_work(scene, "all", 1, 250)
    assert refused.value.details["frames_that_fit"] == 74


def test_a_collider_away_from_the_cloth_or_no_collision_is_no_contact(session: Session) -> None:
    far = _ball(32_768)
    far.bound_box = _corners((20, 20, -1), (22, 22, 1))
    away, _ = session._simulation_cost(_scene(None, [_sheet(40), far]), "all")
    off, _ = session._simulation_cost(
        _scene(None, [_sheet(40, use_collision=False), _ball(32_768)]), "all"
    )

    alone = 1681 * (1.2e-6 + 5 * 1.6e-6) * ALONE
    assert away - FRAME == pytest.approx(alone)
    assert off - FRAME == pytest.approx(alone)


# -- fluids --------------------------------------------------------------------------


def test_sph_particles_and_a_replaying_fluid_domain_count(session: Session) -> None:
    """SPH particles ran 6.4 times the estimate; a smoke domain in Replay mode, not counted
    at all, 28 times at resolution 32; resolution 64 took 3.3 times resolution 32's time
    for 8 times the cells."""
    sph = SimpleNamespace(
        name="Water",
        settings=SimpleNamespace(type="EMITTER", count=5000, physics_type="FLUID"),
        point_cache=_cache(),
    )

    def domain(cache_type: str, resolution: int = 32, domain_type: str = "GAS") -> Any:
        fluid = SimpleNamespace(
            type="FLUID",
            name="Fluid",
            fluid_type="DOMAIN",
            domain_settings=SimpleNamespace(
                cache_type=cache_type, resolution_max=resolution, domain_type=domain_type
            ),
        )
        obj = _mesh("Domain", 8, fluid)
        obj.dimensions = (4.0, 4.0, 2.0)
        return obj

    _, parts = session._simulation_cost(
        _scene(None, [_mesh("Emitter", 4, particle_systems=[sph]), domain("REPLAY")]), "all"
    )
    modular, none = session._simulation_cost(_scene(None, [domain("MODULAR")]), "all")
    fine, fine_parts = session._simulation_cost(_scene(None, [domain("REPLAY", 64)]), "all")
    liquid, _ = session._simulation_cost(
        _scene(None, [domain("REPLAY", domain_type="LIQUID")]), "all"
    )

    assert parts == [
        "SPH fluid particle system 'Water' with 5000 particles: 5 ms a frame",
        "smoke domain 'Domain' replaying at resolution 32 (16384 cells): 9.83 ms a frame",
    ]
    assert (modular, none) == (FRAME, [])
    assert fine - FRAME == pytest.approx(0.6e-6 * 32_768 * 4**0.6)
    assert "(131072 cells)" in fine_parts[0]
    assert liquid - FRAME == pytest.approx(16_384 * 1.5e-6)


def test_the_estimate_reads_the_bodies_where_the_depsgraph_puts_them(
    session: Session,
) -> None:
    """Review of #371: a headless Blender evaluates only when asked, so right after a script
    moved 20 cubes 2 m apart their matrix_world still had them piled; the view layer is
    updated first."""
    updates: List[str] = []
    blender_runtime.bpy.context = SimpleNamespace(
        view_layer=SimpleNamespace(update=lambda: updates.append("update"))
    )

    session._simulation_cost(_scene(_world(3)), "rigid_body")

    assert updates == ["update"]
