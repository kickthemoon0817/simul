"""Fixtures shared by the Blender tests."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from tests.fakes import fake_anim_utils, fake_mathutils


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
