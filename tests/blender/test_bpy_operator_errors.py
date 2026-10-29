"""bpy operator failures keep their own exception class.

The session used to wrap each operator in ``try/except Exception`` and
re-raise a RuntimeError with a "Failed to ..." prefix, because the backend
envelope reported every exception as ``Exception`` anyway. A TypeError for a
bad enum value became a RuntimeError on the way. Now the operator's exception
propagates unchanged; where the old prefix named something bpy's message
leaves out (the object, the resolved file path), ``_bpy_op`` keeps it as an
exception note, which error envelopes render as the same prefix.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from simul.adapters import blender_runtime
from simul.config import Settings
from simul.utils.errors import exception_message

POLL_FAILED = (
    "Operator bpy.ops.rigidbody.object_add.poll() failed, context is incorrect"
)
REPORTED = "Error: Cannot open file /x.blend@ for writing: Permission denied\n"


def _raising(exc: BaseException) -> Any:
    def operator(*args: Any, **kwargs: Any) -> Any:
        raise exc

    return operator


def _session(
    monkeypatch: pytest.MonkeyPatch, ops: Dict[str, Any], tmp_path: Path
) -> Any:
    cube = SimpleNamespace(name="Cube", type="MESH", select_set=lambda value: None)
    world = SimpleNamespace(point_cache=SimpleNamespace(frame_start=1, frame_end=250))
    scene = SimpleNamespace(rigidbody_world=world, objects={"Cube": cube})
    fake = SimpleNamespace(
        app=SimpleNamespace(version=(5, 0, 1), version_string="5.0.1", background=True),
        data=SimpleNamespace(
            objects={"Cube": cube}, filepath=str(tmp_path / "scene.blend")
        ),
        context=SimpleNamespace(
            scene=scene,
            screen=SimpleNamespace(is_animation_playing=True),
            view_layer=SimpleNamespace(objects=SimpleNamespace(active=None)),
            temp_override=lambda **kwargs: _NullContext(),
        ),
        ops=SimpleNamespace(
            **{group: SimpleNamespace(**members) for group, members in ops.items()}
        ),
    )
    monkeypatch.setattr(blender_runtime, "bpy", fake)
    monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
    settings = Settings(security={"allowed_paths": [str(tmp_path)]})
    return blender_runtime.BlenderRuntimeSession(settings)


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> None:
        return None


def test_bpy_op_adds_a_note_and_reraises_the_same_exception() -> None:
    original = TypeError('enum "BOGUS" not found')

    with pytest.raises(TypeError) as raised:
        blender_runtime._bpy_op(
            "Failed to add rigid body to 'Cube'", _raising(original), type="BOGUS"
        )

    assert raised.value is original
    assert raised.value.__notes__ == ["Failed to add rigid body to 'Cube'"]


def test_bpy_op_passes_properties_and_returns_the_result() -> None:
    calls: List[Dict[str, Any]] = []

    def operator(**properties: Any) -> set:
        calls.append(properties)
        return {"FINISHED"}

    assert blender_runtime._bpy_op("unused", operator, type="ACTIVE") == {"FINISHED"}
    assert calls == [{"type": "ACTIVE"}]


def test_rigid_body_failure_names_the_object(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = _session(
        monkeypatch,
        {"rigidbody": {"object_add": _raising(RuntimeError(POLL_FAILED))}},
        tmp_path,
    )

    with pytest.raises(RuntimeError) as raised:
        session.setup_rigid_body("Cube")

    assert str(raised.value) == POLL_FAILED
    assert (
        exception_message(raised.value)
        == f"Failed to add rigid body to 'Cube': {POLL_FAILED}"
    )


def test_file_failures_name_the_resolved_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = _session(
        monkeypatch,
        {
            "wm": {
                "save_as_mainfile": _raising(RuntimeError(REPORTED)),
                "open_mainfile": _raising(RuntimeError(REPORTED)),
            }
        },
        tmp_path,
    )
    target = tmp_path / "out.blend"
    existing = tmp_path / "in.blend"
    existing.write_bytes(b"BLENDER")

    with pytest.raises(RuntimeError) as saved:
        session.save_blend_file(str(target))
    with pytest.raises(RuntimeError) as opened:
        session.open_blend_file(str(existing))

    assert (
        exception_message(saved.value)
        == f"Failed to save .blend file to '{target}': {REPORTED}"
    )
    assert (
        exception_message(opened.value)
        == f"Failed to open .blend file '{existing}': {REPORTED}"
    )


@pytest.mark.parametrize(
    ("group", "operator", "operation"),
    [
        ("wm", "save_mainfile", lambda s: s.save_blend_file()),
        ("screen", "animation_cancel", lambda s: s.play_animation("stop")),
        ("screen", "animation_play", lambda s: s.play_animation("play")),
        ("object", "effector_add", lambda s: s.add_force_field("BOGUS")),
        ("ptcache", "bake", lambda s: s.bake_simulation(1, 10)),
        ("ptcache", "free_bake", lambda s: s.free_bake()),
    ],
    ids=["save-in-place", "cancel", "play", "force-field", "bake", "free-bake"],
)
@pytest.mark.parametrize(
    "make_error",
    [lambda: RuntimeError(REPORTED), lambda: TypeError('enum "BOGUS" not found')],
    ids=["runtime-error", "type-error"],
)
def test_operators_without_added_context_raise_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    group: str,
    operator: str,
    operation: Any,
    make_error: Any,
) -> None:
    raised = make_error()
    session = _session(monkeypatch, {group: {operator: _raising(raised)}}, tmp_path)

    with pytest.raises(type(raised)) as caught:
        operation(session)

    assert caught.value is raised
    assert exception_message(caught.value) == str(raised)
