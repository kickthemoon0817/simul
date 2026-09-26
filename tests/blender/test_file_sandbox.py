"""Regression: Blender file tools must honour the sandbox path policy.

Their Isaac counterparts gained enforcement in the tools layer via the shared
PathPolicy; the Blender family never did, so open/save/import/export accepted
any filesystem path from every caller. The check lives in the session layer —
below the MCP registration and any future CLI — mirroring the layering that
review chose for Isaac after finding registration-level checks bypassable.
"""

from __future__ import annotations

from unittest.mock import MagicMock
from pathlib import Path

import pytest


from simul.adapters import blender_runtime
from simul.config import Settings

OUTSIDE = "/etc/shadow"


@pytest.fixture
def session(monkeypatch: pytest.MonkeyPatch) -> blender_runtime.BlenderRuntimeSession:
    # The denial must fire before any Blender API is touched, so a bare bpy
    # stand-in is all these tests need.
    monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
    monkeypatch.setattr(blender_runtime, "bpy", MagicMock(), raising=False)
    return blender_runtime.BlenderRuntimeSession(settings=Settings())


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("open_blend_file", {"file_path": OUTSIDE}),
        ("save_blend_file", {"file_path": OUTSIDE}),
        ("import_file", {"file_path": OUTSIDE, "file_format": "OBJ"}),
        ("export_file", {"file_path": OUTSIDE, "file_format": "OBJ"}),
        ("export_simready_usd", {"file_path": OUTSIDE}),
    ],
)
def test_out_of_sandbox_paths_are_refused(
    session: blender_runtime.BlenderRuntimeSession, method: str, kwargs: dict
) -> None:
    """The denial must fire before any Blender API is touched, so these tests
    need no bpy at all."""
    with pytest.raises(PermissionError, match="sandbox"):
        getattr(session, method)(**kwargs)


def test_save_without_a_path_checks_the_open_file(
    session: blender_runtime.BlenderRuntimeSession,
) -> None:
    """Attaching to an outside file must not grant permission to overwrite it."""
    blender_runtime.bpy.data.filepath = "/outside/project.blend"
    with pytest.raises(PermissionError, match="sandbox"):
        session.save_blend_file()
    blender_runtime.bpy.ops.wm.save_mainfile.assert_not_called()


def test_in_sandbox_path_passes_the_policy(
    session: blender_runtime.BlenderRuntimeSession,
) -> None:
    """A path under an allowed root reaches the runtime layer; with the
    MagicMock bpy the call completes, proving the policy stepped aside."""
    session.save_blend_file("/tmp/simul-work/out.blend")


def test_checked_path_is_the_used_path(
    session: blender_runtime.BlenderRuntimeSession,
) -> None:
    """A divergent-prefix path must reach Blender in its resolved form.

    ``~/../../tmp/simul-work/...`` resolves inside the sandbox, so the policy
    admits it — but to bpy the ``~`` is a literal path component anchored at
    the server cwd. Whatever the policy checked is what must be used."""
    result = session.save_blend_file("~/../../tmp/simul-work/out.blend")
    assert result["file_path"] == str(Path("/tmp/simul-work/out.blend").resolve())


def test_disabled_sandbox_passes_raw_paths_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the sandbox off, paths must not be rewritten — Blender-relative
    forms like ``//textures/x.png`` are meaningful to bpy as given."""
    monkeypatch.setattr(blender_runtime, "BLENDER_AVAILABLE", True)
    monkeypatch.setattr(blender_runtime, "bpy", MagicMock(), raising=False)
    session = blender_runtime.BlenderRuntimeSession(
        settings=Settings(security={"sandbox_enabled": False})
    )
    result = session.save_blend_file("/anywhere/out.blend")
    assert result["file_path"] == "/anywhere/out.blend"
