"""A sandbox root inside a world-writable directory must belong to this user.

The default writable root is ``/tmp/simul-work``. Any local user can create
that directory before simul does, owned by them and open to everyone, and then
read what agents write there or plant files for them to load. The policy now
creates a missing root with mode 0700 before writing, and refuses a root in a
world-writable directory that someone else owns or can write to.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from simul.utils import paths as paths_module
from simul.utils.paths import PathPolicy, SandboxDenied, shared_directory_problem

pytestmark = pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX ownership")


@pytest.fixture
def shared(tmp_path: Path) -> Path:
    """A world-writable sticky directory, the shape of /tmp."""
    directory = tmp_path / "shared"
    directory.mkdir()
    directory.chmod(0o1777)
    return directory


def _policy(*roots: Path) -> PathPolicy:
    return PathPolicy(enabled=True, allowed_paths=[str(root) for root in roots])


def test_a_missing_root_is_created_private_before_the_first_write(shared: Path) -> None:
    root = shared / "work"
    policy = _policy(root)

    assert policy.authorize(str(root / "scene.usda"), write=True) == str((root / "scene.usda").resolve())
    assert stat.S_IMODE(root.stat().st_mode) == 0o700


def test_a_read_does_not_create_the_root(shared: Path) -> None:
    root = shared / "work"

    _policy(root).authorize(str(root / "scene.usda"))

    assert not root.exists()


def test_a_root_others_can_write_is_refused(shared: Path) -> None:
    root = shared / "work"
    root.mkdir()
    root.chmod(0o777)

    with pytest.raises(SandboxDenied) as denied:
        _policy(root).authorize(str(root / "scene.usda"), write=True)

    details = denied.value.details
    assert details["untrusted_root"] == str(root.resolve())
    assert "writable by group or others" in details["reason"]
    assert "chmod 700" in details["hint"]


def test_a_root_someone_else_owns_is_refused(shared: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = shared / "work"
    root.mkdir(mode=0o700)
    me = os.getuid()
    monkeypatch.setattr(paths_module.os, "getuid", lambda: me + 1)

    with pytest.raises(SandboxDenied) as denied:
        _policy(root).authorize(str(root / "scene.usda"))

    assert "owned by uid" in denied.value.details["reason"]


def test_a_symlink_someone_else_planted_under_the_root_name_is_refused(
    shared: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "private"
    target.mkdir(mode=0o700)
    root = shared / "work"
    root.symlink_to(target)
    me = os.getuid()
    monkeypatch.setattr(paths_module.os, "getuid", lambda: me + 1)

    assert shared_directory_problem(root) is not None
    with pytest.raises(SandboxDenied):
        _policy(root).authorize(str(root / "scene.usda"), write=True)


def test_our_own_private_root_is_accepted(shared: Path) -> None:
    root = shared / "work"
    root.mkdir(mode=0o700)

    assert _policy(root).is_allowed(str(root / "scene.usda"), write=True)


def test_a_world_writable_directory_without_the_sticky_bit_is_refused(tmp_path: Path) -> None:
    open_dir = tmp_path / "open"
    open_dir.mkdir()
    open_dir.chmod(0o777)
    root = open_dir / "work"
    root.mkdir(mode=0o700)

    with pytest.raises(SandboxDenied) as denied:
        _policy(root).authorize(str(root / "scene.usda"))

    assert "sticky bit" in denied.value.details["reason"]


def test_roots_outside_world_writable_directories_are_not_judged(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    root.chmod(0o777)  # the user's own business outside a shared directory

    assert _policy(root).is_allowed(str(root / "scene.usda"), write=True)


def test_the_capture_root_skips_an_untrusted_root(shared: Path, tmp_path: Path) -> None:
    hijacked = shared / "work"
    hijacked.mkdir()
    hijacked.chmod(0o777)
    fallback = tmp_path / "fallback"
    fallback.mkdir()

    policy = _policy(hijacked, fallback)

    assert policy.writable_root() == fallback.resolve()


def test_the_capture_root_creates_a_missing_shared_root_privately(shared: Path) -> None:
    root = shared / "work"

    assert _policy(root).default_capture_dir() == str(root.resolve() / "captures")
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
