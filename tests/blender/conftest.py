"""Fixtures shared by the Blender tests."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Iterator, List

import pytest

from simul.adapters import blender_runtime
from tests.fakes import fake_anim_utils, fake_mathutils


@pytest.fixture(autouse=True)
def _no_recorded_playback() -> Iterator[None]:
    """Start each test without a playback that an earlier one's stand-in recorded.

    play_animation keeps what it started in a module-level record, as the one
    Blender process the add-on lives in would; each test is a new process.
    """
    blender_runtime._playback.clear()
    yield
    blender_runtime._playback.clear()


@pytest.fixture(autouse=True)
def _no_recorded_constraints() -> Iterator[None]:
    """Start each test with no constraint recorded as simulated, and no hull counted.

    The crash guard's record of the constraints the simulation stepped and the
    hull-vertex cache are module-level, as in the one Blender process the
    add-on lives in.
    """
    for record in (
        blender_runtime._built_constraints,
        blender_runtime._constraint_watch,
        blender_runtime._HULL_VERTICES,
    ):
        record.clear()
    yield
    for record in (
        blender_runtime._built_constraints,
        blender_runtime._constraint_watch,
        blender_runtime._HULL_VERTICES,
    ):
        record.clear()


@pytest.fixture
def anim_utils(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Install the ``bpy_extras.anim_utils`` stand-in that reads a ``FakeAction``.

    The Blender runtime reads the F-curves of an action slot through it on
    Blender 4.4 and later.

    Returns:
        The installed module.
    """
    module = fake_anim_utils()
    package = ModuleType("bpy_extras")
    setattr(package, "anim_utils", module)
    monkeypatch.setitem(sys.modules, "bpy_extras", package)
    monkeypatch.setitem(sys.modules, "bpy_extras.anim_utils", module)
    return module


@pytest.fixture
def mathutils(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Install the strict ``mathutils`` stand-in that the Blender runtime imports.

    Its Matrix multiplies only its Vector, as real mathutils does, so code that
    multiplies a ``bound_box`` corner directly fails here as it does in Blender.

    Returns:
        The installed module.
    """
    module = fake_mathutils()
    monkeypatch.setitem(sys.modules, "mathutils", module)
    return module


@pytest.fixture
def fbx_parser(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Install a stand-in for Blender's ``io_scene_fbx.parse_fbx``.

    import_file reads an FBX's texture paths through it before importing.
    Append ``{b"FileName": ..., b"RelativeFilename": ...}`` dicts to the
    returned module's ``textures`` for the Texture objects the file holds; a
    ``b"name"`` entry is the object's name, which the importer falls back to.

    Returns:
        The installed ``parse_fbx`` module.
    """
    from types import SimpleNamespace

    module = ModuleType("io_scene_fbx.parse_fbx")
    textures: list = []

    def parse(path: str) -> tuple:
        def element(id_: bytes, props: list, elems: list) -> SimpleNamespace:
            return SimpleNamespace(id=id_, props=props, elems=elems)

        objects = [
            element(
                b"Texture",
                [0, texture.get(b"name", "").encode() + b"\x00\x01Texture", b""],
                [
                    element(key, [value.encode()], [])
                    for key, value in texture.items()
                    if key != b"name"
                ],
            )
            for texture in textures
        ]
        return element(b"", [], [element(b"Objects", [], objects)]), 7400

    setattr(module, "parse", parse)
    setattr(module, "textures", textures)
    package = ModuleType("io_scene_fbx")
    setattr(package, "parse_fbx", module)
    monkeypatch.setitem(sys.modules, "io_scene_fbx", package)
    monkeypatch.setitem(sys.modules, "io_scene_fbx.parse_fbx", module)
    return module


@pytest.fixture
def frozen_clock(monkeypatch: pytest.MonkeyPatch) -> List[float]:
    """``blender_runtime``'s ``time.monotonic``, which a test moves by hand: ``clock[0] += 1``."""
    import time as real_time
    from types import SimpleNamespace

    clock = [0.0]
    monkeypatch.setattr(
        blender_runtime,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0], strftime=real_time.strftime),
    )
    return clock
