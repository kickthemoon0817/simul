"""Physics sampling, bake dispatch and SimReady export against a stand-in bpy.

Each case pins a behaviour that real Blender 5.0 exposed: rigid bodies only
move in the evaluated world matrix, the positional context-override dict is
gone since 4.0, and cameras/lights carry no material slots.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional

import pytest

from simul_mcp.adapters import blender_runtime


class _Matrix:
    def __init__(self, translation: List[float], euler: List[float]) -> None:
        self.translation = translation
        self._euler = euler
        self.orders: List[str] = []

    def to_euler(self, order: str) -> List[float]:
        self.orders.append(order)
        return self._euler


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
    evaluated: Optional[Dict[int, _Matrix]] = None,
) -> SimpleNamespace:
    calls: Dict[str, Any] = {"frames": [], "overrides": [], "ops": []}
    scene = SimpleNamespace(
        frame_current=1,
        render=SimpleNamespace(fps=10),
        rigidbody_world=rigidbody_world,
        objects=_Objects({o.name: o for o in objects}),
    )

    def frame_set(frame: int) -> None:
        calls["frames"].append(frame)
        scene.frame_current = frame

    scene.frame_set = frame_set

    def evaluated_get(obj: Any, depsgraph: Any) -> Any:
        return SimpleNamespace(matrix_world=evaluated[scene.frame_current])

    for obj in objects:
        obj.evaluated_get = lambda depsgraph, _obj=obj: evaluated_get(_obj, depsgraph)

    @contextmanager
    def temp_override(**kwargs: Any) -> Iterator[None]:
        calls["overrides"].append(kwargs)
        yield

    def op(name: str) -> Any:
        def run(*args: Any, **kwargs: Any) -> set:
            if args:
                raise TypeError("1-2 args execution context is supported")
            calls["ops"].append((name, kwargs))
            return {"FINISHED"}

        return run

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
            temp_override=temp_override,
            evaluated_depsgraph_get=lambda: "depsgraph",
        ),
        ops=SimpleNamespace(
            ptcache=SimpleNamespace(bake=op("ptcache.bake"), free_bake=op("ptcache.free_bake")),
            wm=SimpleNamespace(usd_export=usd_export),
        ),
    )


@pytest.fixture
def install(monkeypatch: pytest.MonkeyPatch) -> Any:
    def _install(fake: SimpleNamespace) -> Any:
        monkeypatch.setattr(blender_runtime, "bpy", fake)
        monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
        return blender_runtime.BlenderRuntimeSession()

    return _install


def _rigidbody_world(baked: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        enabled=True,
        point_cache=SimpleNamespace(frame_start=1, frame_end=250, is_baked=baked),
    )


def _falling(frames: range) -> Dict[int, _Matrix]:
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
    body = _object("faller", rigid_body=SimpleNamespace(type="ACTIVE", mass=2.0, collision_shape="BOX"))
    fake = _fake_bpy([body], _rigidbody_world(), _falling(range(0, 40)))
    fake.context.scene.frame_current = 31
    session = install(fake)

    state = session.get_physics_state("faller")

    assert state["location"] == [0.0, 0.0, -5.0]
    assert state["rotation_euler"] == pytest.approx([3.1, 0.0, 0.0])


def test_bake_and_free_bake_use_temp_override(install: Any) -> None:
    world = _rigidbody_world()
    fake = _fake_bpy([], world)
    session = install(fake)

    session.bake_simulation(1, 20)
    session.free_bake()

    assert [name for name, _ in fake.calls["ops"]] == ["ptcache.bake", "ptcache.free_bake"]
    assert fake.calls["ops"][0][1] == {"bake": True}
    assert [o["point_cache"] for o in fake.calls["overrides"]] == [world.point_cache] * 2
    assert world.point_cache.frame_end == 20


def _default_scene() -> List[Any]:
    return [
        _object("box", data=SimpleNamespace(materials=["mat"])),
        _object("Camera", "CAMERA", data=SimpleNamespace(lens=50.0)),
        _object("Light", "LIGHT", data=SimpleNamespace(energy=1000.0)),
    ]


@pytest.mark.parametrize("embed", [True, False])
def test_simready_export_validates_meshes_and_forwards_embed_metadata(
    install: Any, embed: bool
) -> None:
    fake = _fake_bpy(_default_scene())
    session = install(fake)

    result = session.export_simready_usd("/tmp/simul_mcp/asset.usda", embed_metadata=embed)

    assert {i["object_name"] for i in result["issues"]} <= {"box"}
    assert not [i for i in result["issues"] if i["message"] == "No material assigned"]
    name, kwargs = fake.calls["ops"][-1]
    assert name == "wm.usd_export"
    assert kwargs["export_custom_properties"] is embed


def test_validation_skips_material_check_for_objects_without_slots(install: Any) -> None:
    fake = _fake_bpy(_default_scene())
    session = install(fake)

    result = session.validate_simready_compliance(
        object_names=["Camera", "Light"], check_naming=False, check_hierarchy=False,
        check_transforms=False,
    )

    assert result["issues"] == []


def test_export_drops_options_the_operator_does_not_define(install: Any) -> None:
    fake = _fake_bpy(_default_scene())
    fake.ops.wm.usd_export.get_rna_type = lambda: SimpleNamespace(
        properties={"filepath": 0, "selected_objects_only": 0}
    )
    session = install(fake)

    session.export_simready_usd("/tmp/simul_mcp/asset.usda", validate_before_export=False)

    assert "export_custom_properties" not in fake.calls["ops"][-1][1]
