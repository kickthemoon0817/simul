"""Regression: Blender file tools must honour the sandbox path policy.

Their Isaac counterparts gained enforcement in the tools layer via the shared
PathPolicy; the Blender family never did, so open/save/import/export accepted
any filesystem path from every caller. The check lives in the session layer —
below the MCP registration and any future CLI — mirroring the layering that
review chose for Isaac after finding registration-level checks bypassable.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from simul_mcp.adapters import blender_runtime  # noqa: E402
from simul_mcp.config import Settings  # noqa: E402

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


def test_save_without_a_path_is_not_policed(
    session: blender_runtime.BlenderRuntimeSession,
) -> None:
    """An in-place save names no path; there is nothing to police. With the
    MagicMock bpy it completes — a PermissionError here would mean the policy
    fired on nothing."""
    session.save_blend_file()


def test_in_sandbox_path_passes_the_policy(
    session: blender_runtime.BlenderRuntimeSession,
) -> None:
    """A path under an allowed root reaches the runtime layer; with the
    MagicMock bpy the call completes, proving the policy stepped aside."""
    session.save_blend_file("/tmp/simul_mcp/out.blend")
