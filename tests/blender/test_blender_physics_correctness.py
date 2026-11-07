"""Physics sampling, bake dispatch and SimReady validation against a stand-in bpy.

Each case pins a behaviour that real Blender 5.0 exposed: rigid bodies only
move in the evaluated world matrix, the positional context-override dict is
gone since 4.0, and cameras/lights carry no material slots.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import pytest

from simul.adapters import blender_runtime
from tests.fakes import FakeMatrix, euler_matrix


def _Matrix(translation: List[float], euler: List[float]) -> FakeMatrix:
    """A world matrix at ``translation``, turned by the XYZ Euler angles ``euler``."""
    rows = np.eye(4)
    rows[:3, :3] = euler_matrix(euler)
    rows[:3, 3] = translation
    return FakeMatrix(rows)


class _Objects(dict):
    def __iter__(self) -> Iterator[Any]:  # type: ignore[override]
        return iter(self.values())


def _object(name: str, kind: str = "MESH", **extra: Any) -> SimpleNamespace:
    obj = SimpleNamespace(
        name=name,
        type=kind,
        location=(0.0, 0.0, 10.0),  # authored value, never moved by physics
        rotation_euler=(0.0, 0.0, 0.0),
        rotation_mode="XYZ",
        scale=(1.0, 1.0, 1.0),
        parent=None,
        children=[],
        dimensions=(1.0, 1.0, 1.0),
        selected=False,
        **extra,
    )
    obj.select_set = lambda value: setattr(obj, "selected", value)
    return obj


def _fake_bpy(
    objects: List[Any],
    rigidbody_world: Optional[Any] = None,
    evaluated: Optional[Dict[int, FakeMatrix]] = None,
) -> SimpleNamespace:
    calls: Dict[str, Any] = {"frames": [], "overrides": [], "ops": []}
    scene = SimpleNamespace(
        name="Scene",
        frame_current=1,
        frame_subframe=0.0,
        frame_start=1,
        frame_end=250,
        render=SimpleNamespace(fps=10, fps_base=1.0),
        rigidbody_world=rigidbody_world,
        objects=_Objects({o.name: o for o in objects}),
    )

    def frame_set(frame: int, subframe: float = 0.0) -> None:
        calls["frames"].append(frame)
        scene.frame_current = frame
        scene.frame_subframe = subframe

    scene.frame_set = frame_set

    def evaluated_get(obj: Any, depsgraph: Any) -> Any:
        return SimpleNamespace(
            matrix_world=evaluated[scene.frame_current],
            rotation_euler=obj.rotation_euler,
        )

    for obj in objects:
        obj.evaluated_get = lambda depsgraph, _obj=obj: evaluated_get(_obj, depsgraph)

    @contextmanager
    def temp_override(**kwargs: Any) -> Iterator[None]:
        calls["overrides"].append(kwargs)
        yield

    def op(name: str, then: Any = None) -> Any:
        def run(*args: Any, **kwargs: Any) -> set:
            if args:
                raise TypeError("1-2 args execution context is supported")
            calls["ops"].append((name, kwargs))
            if then is not None:
                then(calls["overrides"][-1])
            return {"FINISHED"}

        return run

    def mark_baked(baked: bool) -> Any:
        return lambda override: setattr(override["point_cache"], "is_baked", baked)

    usd_export = op("wm.usd_export")
    usd_export.get_rna_type = lambda: SimpleNamespace(
        properties={"filepath": 0, "selected_objects_only": 0, "export_custom_properties": 0}
    )
    return SimpleNamespace(
        calls=calls,
        app=SimpleNamespace(version=(5, 0, 1), version_string="5.0.1", background=True),
        data=SimpleNamespace(objects=scene.objects),
        context=SimpleNamespace(
            scene=scene,
            # Object Mode: no active object, so the export changes no mode.
            view_layer=SimpleNamespace(name="ViewLayer", objects=SimpleNamespace(active=None)),
            temp_override=temp_override,
            evaluated_depsgraph_get=lambda: "depsgraph",
        ),
        ops=SimpleNamespace(
            ptcache=SimpleNamespace(
                bake=op("ptcache.bake"),
                bake_from_cache=op("ptcache.bake_from_cache", mark_baked(True)),
                free_bake=op("ptcache.free_bake", mark_baked(False)),
            ),
            wm=SimpleNamespace(usd_export=usd_export),
        ),
    )


@pytest.fixture
def install(monkeypatch: pytest.MonkeyPatch, mathutils: Any) -> Any:
    def _install(fake: SimpleNamespace) -> Any:
        monkeypatch.setattr(blender_runtime, "bpy", fake)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        return blender_runtime.BlenderRuntimeSession()

    return _install


def _rigidbody_world(baked: bool = False, *bodies: Any) -> SimpleNamespace:
    return SimpleNamespace(
        enabled=True,
        time_scale=1.0,
        collection=SimpleNamespace(name="RigidBodyWorld", all_objects=list(bodies)),
        point_cache=SimpleNamespace(
            frame_start=1, frame_end=250, is_baked=baked, is_outdated=False, info=""
        ),
    )


def _falling(frames: range) -> Dict[int, FakeMatrix]:
    return {f: _Matrix([0.0, 0.0, 10.0 - 0.5 * (f - 1)], [0.1 * f, 0.0, 0.0]) for f in frames}


def test_trajectory_samples_the_evaluated_world_pose_of_a_rigid_body(install: Any) -> None:
    body = _object("faller")
    fake = _fake_bpy([body], _rigidbody_world(), _falling(range(0, 40)))
    session = install(fake)

    traj = session.get_object_trajectory("faller", 1, 21, 10)

    assert [p["frame"] for p in traj["points"]] == [1, 11, 21]
    assert [p["location"][2] for p in traj["points"]] == [10.0, 5.0, 0.0]
    assert traj["points"][1]["rotation_euler"] == pytest.approx([1.1, 0.0, 0.0])
    assert traj["points"][2]["velocity"] == [0.0, 0.0, -5.0]
    # An unbaked simulation is stepped through every frame, not jumped.
    assert fake.calls["frames"][:-1] == list(range(1, 22))
    assert fake.calls["frames"][-1] == 1  # original frame restored


def test_trajectory_jumps_straight_to_samples_without_a_live_simulation(install: Any) -> None:
    body = _object("faller")
    fake = _fake_bpy([body], _rigidbody_world(baked=True), _falling(range(0, 40)))
    session = install(fake)

    session.get_object_trajectory("faller", 1, 21, 10)

    assert fake.calls["frames"] == [1, 11, 21, 1]


def test_physics_state_reports_world_pose_not_authored_location(install: Any) -> None:
    body = _object(
        "faller",
        rigid_body=SimpleNamespace(
            type="ACTIVE", mass=2.0, collision_shape="BOX", kinematic=False, enabled=True
        ),
    )
    fake = _fake_bpy([body], _rigidbody_world(False, body), _falling(range(0, 40)))
    fake.context.scene.frame_current = 31
    session = install(fake)

    state = session.get_physics_state("faller")

    assert state["location"] == [0.0, 0.0, -5.0]
    assert state["rotation_euler"] == pytest.approx([3.1, 0.0, 0.0])
    # An unbaked simulation is stepped from its cache start to the frame first.
    assert fake.calls["frames"] == list(range(1, 32))


def test_bake_and_free_bake_use_temp_override(install: Any) -> None:
    world = _rigidbody_world()
    fake = _fake_bpy([], world)
    session = install(fake)

    session.bake_simulation(1, 20)
    session.free_bake()

    assert [name for name, _ in fake.calls["ops"]] == [
        "ptcache.bake_from_cache",
        "ptcache.free_bake",
    ]
    assert [o["point_cache"] for o in fake.calls["overrides"]] == [world.point_cache] * 2
    # The bake stepped every frame of its range, then returned to frame 1.
    assert fake.calls["frames"] == list(range(1, 21)) + [1]
    # Freed, the cache covers the scene range again.
    assert (world.point_cache.frame_start, world.point_cache.frame_end) == (1, 250)


def _default_scene() -> List[Any]:
    return [
        _object("box", data=SimpleNamespace(materials=["mat"])),
        _object("Camera", "CAMERA", data=SimpleNamespace(lens=50.0)),
        _object("Light", "LIGHT", data=SimpleNamespace(energy=1000.0)),
    ]


def test_validation_skips_material_check_for_objects_without_slots(install: Any) -> None:
    fake = _fake_bpy(_default_scene())
    session = install(fake)

    result = session.validate_simready_compliance(
        object_names=["Camera", "Light"], check_naming=False, check_hierarchy=False,
        check_transforms=False,
    )

    assert result["issues"] == []


# export_simready_usd is covered against a Blender-like exporter, with the
# written file checked through pxr, in test_simready_export.py.
