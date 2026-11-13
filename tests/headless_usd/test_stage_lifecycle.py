"""Loaded stages are isolated, notice changes on disk, and can be saved, exported, reloaded and unloaded.

These run the headless session against real ``pxr`` stages on disk: a robot
file, a scene that sublayers one file and references the robot, and the
operations an agent needs to keep or drop its edits. The session's owner
resolver stands in for the MCP session id, so the per-session isolation the
server relies on is tested without a server.
"""

from __future__ import annotations

import os
import secrets
import stat
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Tuple

import pytest

pytest.importorskip("pxr", reason="pxr library not available")

from pxr import Sdf, Usd, UsdGeom, UsdUtils  # noqa: E402

from simul.adapters import headless_usd  # noqa: E402
from simul.adapters.headless_usd import HeadlessUSDSession  # noqa: E402
from simul.config import Settings  # noqa: E402
from simul.usd import stage_files, validation  # noqa: E402
from simul.usd.sandbox import CompositionSandbox  # noqa: E402
from simul.usd.editing import USDEditError  # noqa: E402
from simul.utils.paths import PathPolicy, SandboxDenied  # noqa: E402


def _write_robot(path: Path) -> None:
    stage = Usd.Stage.CreateNew(str(path))
    robot = UsdGeom.Xform.Define(stage, "/Robot")
    stage.SetDefaultPrim(robot.GetPrim())
    UsdGeom.Xform.Define(stage, "/Robot/link1")
    UsdGeom.Cube.Define(stage, "/Robot/link2").GetSizeAttr().Set(1.0)
    stage.GetRootLayer().Save()


def _build(directory: Path) -> Dict[str, Path]:
    """Write robot.usda, sub.usda, a texture and scene.usda (sublayer + reference + asset)."""
    _write_robot(directory / "robot.usda")
    sub = Usd.Stage.CreateNew(str(directory / "sub.usda"))
    UsdGeom.Xform.Define(sub, "/World")
    UsdGeom.Xform.Define(sub, "/World/FromSub")
    sub.GetRootLayer().Save()
    (directory / "tex").mkdir()
    (directory / "tex" / "wood.png").write_bytes(b"\x89PNG\r\n\x1a\nnot-really")
    stage = Usd.Stage.CreateNew(str(directory / "scene.usda"))
    stage.GetRootLayer().subLayerPaths.append("./sub.usda")
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    ref = UsdGeom.Xform.Define(stage, "/World/Robot_ref")
    ref.GetPrim().GetReferences().AddReference("./robot.usda")
    ref.GetPrim().CreateAttribute("custom:texture", Sdf.ValueTypeNames.Asset).Set("./tex/wood.png")
    stage.GetRootLayer().Save()
    return {
        "robot": directory / "robot.usda",
        "sub": directory / "sub.usda",
        "scene": directory / "scene.usda",
    }


def _touch_later(path: Path) -> None:
    """Make an external edit visible to mtime checks even on a coarse clock."""
    stat = os.stat(path)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))


def _edit_externally(path: Path, prim_path: str) -> None:
    """Add a prim to ``path`` the way another program would: a separate layer, saved."""
    layer = Sdf.Layer.OpenAsAnonymous(str(path))
    Sdf.CreatePrimInLayer(layer, prim_path).specifier = Sdf.SpecifierDef
    layer.Export(str(path))
    _touch_later(path)


def _on_disk(path: Path) -> Sdf.Layer:
    """Read ``path`` fresh, bypassing every layer the session holds."""
    return Sdf.Layer.OpenAsAnonymous(str(path))


@pytest.fixture
def owner() -> Dict[str, Any]:
    return {"id": "session-a"}


@pytest.fixture
def session(owner: Dict[str, Any]) -> Iterator[HeadlessUSDSession]:
    usd_session = HeadlessUSDSession()
    usd_session.set_owner_resolver(lambda: owner["id"])
    yield usd_session
    usd_session.cleanup()


@pytest.fixture
def files(tmp_path: Path) -> Dict[str, Path]:
    return _build(tmp_path)


def _refusal(call, *args: Any, **kwargs: Any) -> USDEditError:
    with pytest.raises(USDEditError) as caught:
        call(*args, **kwargs)
    return caught.value


class TestIsolation:
    def test_unsaved_edits_do_not_leak_into_a_referencing_stage(self, session, files) -> None:
        robot_id = session.load_stage(files["robot"])
        session.delete_prim(robot_id, "/Robot/link2")

        scene = session.get_stage(session.load_stage(files["scene"]))

        assert scene.GetPrimAtPath("/World/Robot_ref/link2")
        assert not session.get_stage(robot_id).GetPrimAtPath("/Robot/link2")

    def test_a_stage_with_edits_is_not_handed_out_again(self, session, files) -> None:
        first = session.load_stage(files["robot"])
        session.create_prim(first, "/Robot/Extra", "Xform")

        second = session.load_stage(files["robot"])

        assert second != first
        assert not session.get_stage(second).GetPrimAtPath("/Robot/Extra")
        assert session.get_stage(first).GetPrimAtPath("/Robot/Extra")

    def test_a_clean_current_stage_is_reused(self, session, files) -> None:
        assert session.load_stage(files["robot"]) == session.load_stage(files["robot"])

    def test_stages_belong_to_the_session_that_loaded_them(self, session, files, owner) -> None:
        mine = session.load_stage(files["robot"])
        owner["id"] = "session-b"

        assert session.get_stage(mine) is None
        assert session.list_stages() == []
        error = _refusal(session.update_prim_attributes, mine, "/Robot", {"custom:a": 1})
        assert error.error_type == "NotFoundError"
        assert error.message.startswith(f"Stage not loaded: {mine!r}")
        theirs = session.load_stage(files["robot"])
        assert theirs != mine

        owner["id"] = "session-a"
        assert session.list_stages() == [mine]

    def test_stage_ids_are_not_derived_from_the_path(self, session, files, owner) -> None:
        first = session.load_stage(files["robot"])
        owner["id"] = "session-b"
        second = session.load_stage(files["robot"])

        assert first.startswith("robot_") and second.startswith("robot_")
        assert first != second

    def test_an_ended_session_releases_only_its_stages(self, session, files, owner, caplog) -> None:
        mine = session.load_stage(files["robot"])
        session.create_prim(mine, "/Robot/Unsaved", "Xform")
        clean = session.load_stage(files["scene"])
        owner["id"] = "session-b"
        theirs = session.load_stage(files["robot"])

        with caplog.at_level("WARNING"):
            released = session.release_owner("session-a")

        assert sorted(released) == sorted([mine, clean])
        assert session.list_stages() == [theirs]
        owner["id"] = "session-a"
        assert session.list_stages() == []
        dropped = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        assert len(dropped) == 1 and mine in dropped[0] and clean not in dropped[0]

    def test_a_session_cannot_hold_unlimited_stages(self, session, files, monkeypatch) -> None:
        monkeypatch.setattr(headless_usd, "MAX_STAGES_PER_OWNER", 2)
        robot = session.load_stage(files["robot"])
        session.load_stage(files["scene"])

        error = _refusal(session.load_stage, files["sub"])

        assert error.error_type == "RefusedOperation"
        assert session.load_stage(files["robot"]) == robot


class TestExternalChanges:
    def test_a_changed_root_file_is_read_again(self, session, files) -> None:
        old = session.load_stage(files["robot"])
        _edit_externally(files["robot"], "/Robot/AddedOutside")

        assert session.stage_status(old)["changed_on_disk"] == [str(files["robot"].resolve())]
        new = session.load_stage(files["robot"])

        assert new != old
        assert session.get_stage(new).GetPrimAtPath("/Robot/AddedOutside")

    def test_a_changed_sublayer_is_read_again(self, session, files) -> None:
        session.load_stage(files["scene"])
        _edit_externally(files["sub"], "/World/FromSubToo")

        stage = session.get_stage(session.load_stage(files["scene"]))

        assert stage.GetPrimAtPath("/World/FromSubToo")

    def test_reload_rereads_the_files(self, session, files) -> None:
        stage_id = session.load_stage(files["scene"])
        _edit_externally(files["scene"], "/World/AddedOutside")

        result = session.reload_stage(stage_id)

        assert session.get_stage(stage_id).GetPrimAtPath("/World/AddedOutside")
        assert str(files["scene"].resolve()) in result["reloaded_files"]
        assert session.stage_status(stage_id)["changed_on_disk"] == []


class TestSave:
    def test_save_writes_the_root_layer_only(self, session, files) -> None:
        stage_id = session.load_stage(files["scene"])
        session.create_prim(stage_id, "/World/Saved", "Cube", {"size": 3})
        sub_mtime = os.stat(files["sub"]).st_mtime_ns

        result = session.save_stage(stage_id)

        assert result["saved"] is True
        assert result["files_written"] == [
            {"path": str(files["scene"].resolve()), "bytes": os.path.getsize(files["scene"])}
        ]
        assert _on_disk(files["scene"]).GetPrimAtPath("/World/Saved")
        assert os.stat(files["sub"]).st_mtime_ns == sub_mtime
        assert session.stage_status(stage_id)["has_unsaved_changes"] is False

    def test_nothing_to_save(self, session, files) -> None:
        result = session.save_stage(session.load_stage(files["robot"]))

        assert result["saved"] is False and result["files_written"] == []

    def test_save_refuses_to_clobber_an_external_change(self, session, files) -> None:
        stage_id = session.load_stage(files["robot"])
        session.create_prim(stage_id, "/Robot/Mine", "Xform")
        _edit_externally(files["robot"], "/Robot/Theirs")

        error = _refusal(session.save_stage, stage_id)

        assert error.error_type == "RefusedOperation"
        assert error.details["override"] == "overwrite_external_changes"
        assert _on_disk(files["robot"]).GetPrimAtPath("/Robot/Theirs")

        session.save_stage(stage_id, overwrite_external_changes=True)

        assert _on_disk(files["robot"]).GetPrimAtPath("/Robot/Mine")

    def test_usdz_is_not_saved_in_place(self, session, files, tmp_path) -> None:
        package = tmp_path / "robot.usdz"
        session.export_stage(session.load_stage(files["robot"]), str(package))
        stage_id = session.load_stage(package)
        session.create_prim(stage_id, "/Robot/New", "Xform")

        error = _refusal(session.save_stage, stage_id)

        assert error.error_type == "UnsupportedOperation"
        assert "export_usd_stage" in error.message


class TestExport:
    def test_export_elsewhere_reanchors_relative_asset_paths(self, session, files, tmp_path) -> None:
        stage_id = session.load_stage(files["scene"])
        session.create_prim(stage_id, "/World/Edited", "Xform")
        (tmp_path / "out").mkdir()
        destination = tmp_path / "out" / "copy.usda"

        result = session.export_stage(stage_id, str(destination))

        assert result["rewritten_asset_paths"] == 3
        assert result["files_written"][0]["path"] == str(destination.resolve())
        layer = _on_disk(destination)
        assert list(layer.subLayerPaths) == ["../sub.usda"]
        exported = Usd.Stage.Open(str(destination))
        assert exported.GetPrimAtPath("/World/Edited")
        assert exported.GetPrimAtPath("/World/FromSub")
        assert exported.GetPrimAtPath("/World/Robot_ref/link2")
        texture = exported.GetPrimAtPath("/World/Robot_ref").GetAttribute("custom:texture").Get()
        assert Path(texture.resolvedPath) == (files["scene"].parent / "tex" / "wood.png").resolve()
        assert session.stage_status(stage_id)["has_unsaved_changes"] is True

    def test_flatten(self, session, files, tmp_path) -> None:
        stage_id = session.load_stage(files["scene"])
        destination = tmp_path / "flat.usdc"

        result = session.export_stage(stage_id, str(destination), flatten=True)

        assert result["format"] == "usdc" and result["flatten"] is True
        layer = _on_disk(destination)
        assert list(layer.subLayerPaths) == []
        assert layer.GetPrimAtPath("/World/Robot_ref/link2") is not None
        assert layer.GetPrimAtPath("/World/FromSub") is not None

    def test_usdz_intermediate_stays_beside_the_package(self, session, files, tmp_path, monkeypatch) -> None:
        """The sandbox authorized the package's directory, not the system temp dir."""
        real = stage_files.tempfile.TemporaryDirectory
        used = []

        def spy(*args: Any, **kwargs: Any) -> Any:
            used.append(kwargs.get("dir"))
            return real(*args, **kwargs)

        monkeypatch.setattr(stage_files.tempfile, "TemporaryDirectory", spy)
        (tmp_path / "out").mkdir()
        destination = tmp_path / "out" / "pkg.usdz"

        session.export_stage(session.load_stage(files["robot"]), str(destination))

        assert used == [str(destination.parent.resolve())]
        assert sorted(p.name for p in destination.parent.iterdir()) == ["pkg.usdz"]

    @pytest.mark.parametrize("flatten", [False, True])
    def test_usdz_package(self, session, files, tmp_path, flatten) -> None:
        stage_id = session.load_stage(files["scene"])
        destination = tmp_path / f"pkg_{flatten}.usdz"

        session.export_stage(stage_id, str(destination), flatten=flatten)

        package = Usd.Stage.Open(str(destination))
        assert package.GetPrimAtPath("/World/Robot_ref/link2")
        assert package.GetPrimAtPath("/World/FromSub")

    def test_export_refusals(self, session, files, tmp_path) -> None:
        stage_id = session.load_stage(files["robot"])
        existing = tmp_path / "taken.usda"
        existing.write_text("#usda 1.0\n")

        refused = _refusal(session.export_stage, stage_id, str(existing))
        assert refused.error_type == "RefusedOperation" and refused.details["override"] == "overwrite"
        assert existing.read_text() == "#usda 1.0\n"
        session.export_stage(stage_id, str(existing), overwrite=True)
        assert _on_disk(existing).GetPrimAtPath("/Robot/link1")

        for destination, code in [
            (tmp_path / "x.obj", "InvalidFilePath"),
            (tmp_path / "missing" / "x.usda", "InvalidFilePath"),
        ]:
            error = _refusal(session.export_stage, stage_id, str(destination))
            assert (error.error_type, error.details["code"]) == ("ValidationError", code)
        same = _refusal(session.export_stage, stage_id, str(files["robot"]))
        assert same.error_type == "RefusedOperation" and "save_usd_stage" in same.message


class TestSaveTracksOnlyWhatItWrote:
    """A save writes the root file, so only the root file's change tracking moves."""

    def test_a_changed_sublayer_stays_changed_after_the_stage_is_saved(self, session, files) -> None:
        scene = session.load_stage(files["scene"])
        sub = session.load_stage(files["sub"])
        session.create_prim(sub, "/World/FromSub/Added", "Cube", {"size": 42})
        session.save_stage(sub)
        sub_path = str(files["sub"].resolve())
        assert session.stage_status(scene)["changed_on_disk"] == [sub_path]

        session.create_prim(scene, "/World/Mine", "Xform")
        session.save_stage(scene)

        assert session.stage_status(scene)["changed_on_disk"] == [sub_path]
        again = session.load_stage(files["scene"])
        assert again != scene
        assert session.get_stage(again).GetPrimAtPath("/World/FromSub/Added").GetAttribute("size").Get() == 42
        assert session.stage_status(again)["changed_on_disk"] == []

    def test_a_saved_stage_is_current_for_its_own_file(self, session, files) -> None:
        stage_id = session.load_stage(files["robot"])
        session.create_prim(stage_id, "/Robot/Mine", "Xform")

        session.save_stage(stage_id)

        assert session.stage_status(stage_id)["changed_on_disk"] == []
        assert session.load_stage(files["robot"]) == stage_id


def _race(calls: List[Tuple[str, Callable[[], Any]]]) -> Dict[str, Any]:
    """Run each call on its own thread, released together; return each one's result or USDEditError."""
    barrier = threading.Barrier(len(calls))
    outcomes: Dict[str, Any] = {}

    def run(name: str, call: Callable[[], Any]) -> None:
        barrier.wait()
        try:
            outcomes[name] = call()
        except USDEditError as exc:
            outcomes[name] = exc

    threads = [threading.Thread(target=run, args=item) for item in calls]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    return outcomes


class TestConcurrentWrites:
    """Two writers of one file: the check and the write happen as one step per file."""

    def test_two_stages_saving_one_file_at_once_lose_no_write(self, session, files, monkeypatch) -> None:
        with session.acting_for("session-a"):
            first = session.load_stage(files["robot"])
            session.create_prim(first, "/Robot/FromA", "Xform")
        with session.acting_for("session-b"):
            second = session.load_stage(files["robot"])
            session.create_prim(second, "/Robot/FromB", "Xform")
        real = stage_files.changed_files

        def slow_check(stage: Any, recorded: Any) -> List[str]:
            changed = real(stage, recorded)
            time.sleep(0.3)  # without a lock, both saves pass their check before either writes
            return changed

        monkeypatch.setattr(stage_files, "changed_files", slow_check)

        def save(owner: str, stage_id: str) -> Callable[[], Any]:
            def call() -> Any:
                with session.acting_for(owner):
                    return session.save_stage(stage_id)

            return call

        outcomes = _race([("a", save("session-a", first)), ("b", save("session-b", second))])

        saved = [name for name, outcome in outcomes.items() if isinstance(outcome, dict)]
        refused = [outcome for outcome in outcomes.values() if isinstance(outcome, USDEditError)]
        assert len(saved) == 1 and len(refused) == 1, outcomes
        assert refused[0].details["override"] == "overwrite_external_changes"
        on_disk = _on_disk(files["robot"])
        assert on_disk.GetPrimAtPath("/Robot/FromA" if saved == ["a"] else "/Robot/FromB")
        loser = ("session-b", second) if saved == ["a"] else ("session-a", first)
        with session.acting_for(loser[0]):
            assert session.stage_status(loser[1])["has_unsaved_changes"] is True

    def test_two_exports_to_one_new_path_create_it_once(self, session, files, tmp_path, monkeypatch) -> None:
        robot = session.load_stage(files["robot"])
        scene = session.load_stage(files["scene"])
        destination = tmp_path / "out.usda"
        real = stage_files.export_stage

        def slow_export(*args: Any, **kwargs: Any) -> Any:
            time.sleep(0.3)  # without a lock, both exports pass the existence check first
            return real(*args, **kwargs)

        monkeypatch.setattr(stage_files, "export_stage", slow_export)

        outcomes = _race(
            [
                ("robot", lambda: session.export_stage(robot, str(destination))),
                ("scene", lambda: session.export_stage(scene, str(destination))),
            ]
        )

        written = [name for name, outcome in outcomes.items() if isinstance(outcome, dict)]
        refused = [outcome for outcome in outcomes.values() if isinstance(outcome, USDEditError)]
        assert len(written) == 1 and len(refused) == 1, outcomes
        assert refused[0].details["override"] == "overwrite"
        root_prim = "/Robot" if written == ["robot"] else "/World"
        assert _on_disk(destination).GetPrimAtPath(root_prim)
        assert [p.name for p in tmp_path.iterdir() if ".simul-" in p.name] == []

    def test_a_destination_another_program_creates_meanwhile_is_kept(
        self, session, files, tmp_path, monkeypatch
    ) -> None:
        destination = tmp_path / "out.usda"
        real = stage_files._commit

        def created_meanwhile(written: str, target: str, *, overwrite: bool) -> None:
            Path(target).write_text("#usda 1.0\n")
            real(written, target, overwrite=overwrite)

        monkeypatch.setattr(stage_files, "_commit", created_meanwhile)

        error = _refusal(session.export_stage, session.load_stage(files["robot"]), str(destination))

        assert error.error_type == "RefusedOperation" and error.details["override"] == "overwrite"
        assert destination.read_text() == "#usda 1.0\n"
        assert [p.name for p in tmp_path.iterdir() if ".simul-" in p.name] == []

    def test_exclusive_create_without_hard_links(self, session, files, tmp_path, monkeypatch) -> None:
        def no_links(*args: Any) -> None:
            raise PermissionError("hard links not supported")

        monkeypatch.setattr(stage_files.os, "link", no_links)
        stage = session.get_stage(session.load_stage(files["robot"]))
        destination = str(tmp_path / "out.usdc")

        stage_files.export_stage(stage, destination, flatten=False)

        assert _on_disk(Path(destination)).GetPrimAtPath("/Robot/link1")
        with pytest.raises(stage_files.DestinationExists):
            stage_files.export_stage(stage, destination, flatten=True)
        assert [p.name for p in tmp_path.iterdir() if ".simul-" in p.name] == []

    def test_the_no_hard_link_fallback_removes_its_placeholder_when_the_move_fails(
        self, tmp_path, monkeypatch
    ) -> None:
        written = tmp_path / "written.usda"
        written.write_text("#usda 1.0\n")
        destination = tmp_path / "out.usda"

        def no_links(*args: Any) -> None:
            raise PermissionError("hard links not supported")

        def failing_replace(*args: Any) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(stage_files.os, "link", no_links)
        monkeypatch.setattr(stage_files.os, "replace", failing_replace)

        with pytest.raises(OSError, match="disk full"):
            stage_files._commit(str(written), str(destination), overwrite=False)

        assert not destination.exists()

    def test_overwrite_keeps_the_replaced_files_permissions(self, session, files, tmp_path) -> None:
        destination = tmp_path / "out.usda"
        destination.write_text("#usda 1.0\n")
        destination.chmod(0o640)

        session.export_stage(session.load_stage(files["robot"]), str(destination), overwrite=True)

        assert stat.S_IMODE(destination.stat().st_mode) == 0o640
        assert _on_disk(destination).GetPrimAtPath("/Robot")

    def test_a_save_in_the_same_clock_tick_still_counts_as_a_change(self, session, files) -> None:
        """The stamp carries size and inode as well as the modification time."""
        stage_id = session.load_stage(files["robot"])
        session.create_prim(stage_id, "/Robot/Mine", "Xform")
        before = os.stat(files["robot"])
        _edit_externally(files["robot"], "/Robot/Theirs")
        os.utime(files["robot"], ns=(before.st_atime_ns, before.st_mtime_ns))

        assert session.stage_status(stage_id)["changed_on_disk"] == [str(files["robot"].resolve())]
        assert _refusal(session.save_stage, stage_id).details["override"] == "overwrite_external_changes"


def _other_spelling(path: Path, spell: Callable[[str], str]) -> Path:
    """``path`` with its name spelled by ``spell``; skips unless the filesystem finds the same file."""
    other = path.with_name(spell(path.name))
    if other.name == path.name or not other.exists() or not os.path.samefile(other, path):
        pytest.skip("this filesystem tells the two spellings apart")
    return other


_SPELLINGS = {
    "case": lambda name: name.upper(),
    "unicode": lambda name: unicodedata.normalize("NFD", name),
}


class TestOneFileManySpellings:
    """Two names for one file (case on APFS/NTFS, NFC and NFD) are one file to the write locks."""

    def test_the_identity_of_a_file_ignores_spelling_and_survives_a_save(self, session, tmp_path) -> None:
        path = tmp_path / "장면.usda"
        path.write_text('#usda 1.0\ndef Xform "W"\n{\n}\n')
        key = stage_files.file_identity(str(path))
        stage_id = session.load_stage(path)
        session.create_prim(stage_id, "/W/New", "Xform")
        session.save_stage(stage_id)  # a .usda save replaces the file: a new inode

        assert stage_files.file_identity(str(path)) == key
        assert stage_files.file_identity(unicodedata.normalize("NFD", str(path))) == key
        assert stage_files.file_identity(str(path.with_name(path.name.upper()))) == key
        assert stage_files.file_identity(str(tmp_path / "new.usda")) != key

    @pytest.mark.parametrize("spelling", sorted(_SPELLINGS))
    def test_two_spellings_saving_at_once_lose_no_write(self, session, tmp_path, monkeypatch, spelling) -> None:
        path = tmp_path / "Scène.usda"
        path.write_text('#usda 1.0\ndef Xform "W"\n{\n}\n')
        other = _other_spelling(path, _SPELLINGS[spelling])
        with session.acting_for("session-a"):
            first = session.load_stage(path)
            session.create_prim(first, "/W/FromA", "Xform")
        with session.acting_for("session-b"):
            second = session.load_stage(other)
            session.create_prim(second, "/W/FromB", "Xform")
        real = stage_files.changed_files

        def slow_check(stage: Any, recorded: Any) -> List[str]:
            changed = real(stage, recorded)
            time.sleep(0.3)
            return changed

        monkeypatch.setattr(stage_files, "changed_files", slow_check)

        def save(owner: str, stage_id: str) -> Callable[[], Any]:
            def call() -> Any:
                with session.acting_for(owner):
                    return session.save_stage(stage_id)

            return call

        outcomes = _race([("a", save("session-a", first)), ("b", save("session-b", second))])

        saved = [name for name, outcome in outcomes.items() if isinstance(outcome, dict)]
        assert len(saved) == 1, outcomes
        assert _on_disk(path).GetPrimAtPath("/W/FromA" if saved == ["a"] else "/W/FromB")

    def test_exporting_onto_the_stages_own_file_in_another_case_is_refused(self, session, files) -> None:
        other = _other_spelling(files["robot"], _SPELLINGS["case"])

        error = _refusal(session.export_stage, session.load_stage(files["robot"]), str(other), overwrite=True)

        assert error.error_type == "RefusedOperation" and "save_usd_stage" in error.message


_BROKEN = '#usda 1.0\ndef Xform "World"\n{\n    this is not usda\n'


class TestUnparseableRereads:
    """A changed file pxr cannot parse now keeps its content and stays changed on disk."""

    def test_reload_reports_a_sublayer_that_no_longer_parses(self, session, files) -> None:
        stage_id = session.load_stage(files["scene"])
        sub = str(files["sub"].resolve())
        good = files["sub"].read_text()
        files["sub"].write_text(_BROKEN)
        _touch_later(files["sub"])

        result = session.reload_stage(stage_id)

        assert sub not in result["reloaded_files"]
        assert [entry["path"] for entry in result["failed_files"]] == [sub]
        assert result["failed_files"][0]["reason"]
        assert session.get_stage(stage_id).GetPrimAtPath("/World/FromSub")
        assert session.stage_status(stage_id)["changed_on_disk"] == [sub]

        files["sub"].write_text(good.replace('def Xform "FromSub"', 'def Xform "Fixed"'))
        _touch_later(files["sub"])
        result = session.reload_stage(stage_id)

        assert sub in result["reloaded_files"] and result["failed_files"] == []
        assert session.get_stage(stage_id).GetPrimAtPath("/World/Fixed")
        assert session.stage_status(stage_id)["changed_on_disk"] == []

    def test_a_load_keeps_an_unparseable_shared_file_changed_on_disk(self, session, files) -> None:
        (files["scene"].parent / "scene2.usda").write_text(files["scene"].read_text())
        session.load_stage(files["scene"])
        files["sub"].write_text(_BROKEN)
        _touch_later(files["sub"])

        other = session.load_stage(files["scene"].parent / "scene2.usda")

        assert session.get_stage(other).GetPrimAtPath("/World/FromSub")
        assert session.stage_status(other)["changed_on_disk"] == [str(files["sub"].resolve())]

    def test_repeated_loads_reuse_a_stage_while_a_sublayer_stays_unparseable(self, session, files) -> None:
        held = session.load_stage(files["scene"])
        sub = str(files["sub"].resolve())
        files["sub"].write_text(_BROKEN)
        _touch_later(files["sub"])

        loaded = [session.load_stage(files["scene"]) for _ in range(4)]

        assert loaded[0] != held and set(loaded[1:]) <= {held, loaded[0]}
        assert len(session.list_stages()) == 2
        for stage_id in (held, loaded[-1]):
            assert [path for path, _ in session.unread_files(stage_id)] == [sub]
            assert session.stage_status(stage_id)["changed_on_disk"] == [sub]

        files["sub"].write_text('#usda 1.0\ndef Xform "World"\n{\n    def Xform "Fixed"\n    {\n    }\n}\n')
        _touch_later(files["sub"])
        fixed = session.load_stage(files["scene"])

        assert fixed not in loaded and session.get_stage(fixed).GetPrimAtPath("/World/Fixed")
        assert session.unread_files(fixed) == []

    def test_a_root_that_no_longer_parses_keeps_the_unsaved_edits(self, session, files) -> None:
        stage_id = session.load_stage(files["robot"])
        session.create_prim(stage_id, "/Robot/Unsaved", "Xform")
        files["robot"].write_text(_BROKEN)
        _touch_later(files["robot"])

        error = _refusal(session.reload_stage, stage_id, discard_unsaved=True)

        assert "Reloading" in error.message
        assert session.get_stage(stage_id).GetPrimAtPath("/Robot/Unsaved")
        assert session.stage_status(stage_id)["has_unsaved_changes"] is True


class TestSelfReference:
    def test_a_file_that_references_itself_is_recognised(self, session, tmp_path, files) -> None:
        path = tmp_path / "self.usda"
        path.write_text(
            '#usda 1.0\ndef Xform "W"\n{\n    def Cube "Proto"\n    {\n        double size = 1\n    }\n'
            '    def "Inst" (\n        references = @./self.usda@</W/Proto>\n    )\n    {\n    }\n}\n'
        )

        stage = session.get_stage(session.load_stage(path))

        assert stage_files.composes_own_file(stage)
        assert not stage_files.composes_own_file(session.get_stage(session.load_stage(files["scene"])))


class TestLoadOwnership:
    def test_a_load_for_a_session_that_ended_meanwhile_is_not_kept(self, session, files) -> None:
        with session.acting_for("gone", alive=lambda: False):
            error = _refusal(session.load_stage_or_raise, files["robot"])

        assert error.error_type == "RefusedOperation" and "ended" in error.message
        assert session._stage_records == {} and session._active_stages == {}
        with session.acting_for("open", alive=lambda: True):
            assert session.load_stage_or_raise(files["robot"])

    def test_reuse_lookup_reads_the_registry_and_checks_on_demand(self, session, files) -> None:
        assert session.latest_stage_for(files["robot"]) is None
        first = session.load_stage(files["robot"])
        assert session.latest_stage_for(files["robot"]) == first
        assert session.reuse_stage(first, files["robot"]) is True

        session.create_prim(first, "/Robot/Extra", "Xform")
        assert session.reuse_stage(first, files["robot"]) is False
        second = session.load_stage(files["robot"])
        assert session.latest_stage_for(files["robot"]) == second

        _edit_externally(files["robot"], "/Robot/Outside")
        assert session.reuse_stage(second, files["robot"]) is False


def _package(directory: Path) -> Path:
    """A .usdz whose root layer sublayers a file inside it and names a texture inside it."""
    source = directory / "pkg_src"
    (source / "tex").mkdir(parents=True)
    (source / "tex" / "wood.png").write_bytes(b"\x89PNG\r\n\x1a\nnot-really")
    (source / "inner.usda").write_text(
        '#usda 1.0\ndef Xform "Pkg"\n{\n    def Xform "FromSublayer"\n    {\n    }\n}\n'
    )
    (source / "root.usda").write_text(
        '#usda 1.0\n(\n    defaultPrim = "Pkg"\n    subLayers = [@./inner.usda@]\n)\n'
        'over "Pkg"\n{\n    asset custom:texture = @./tex/wood.png@\n}\n'
    )
    package = directory / "pkg" / "pkg.usdz"
    package.parent.mkdir()
    assert UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(source / "root.usda")), str(package))
    return package


def _replace_package(package: Path, inner: str) -> None:
    """Build a new ``package`` whose inner.usda is ``inner``, from a fresh source tree, and swap it in."""
    source = package.parent.parent / f"pkg_src_{secrets.token_hex(4)}"
    source.mkdir()
    (source / "inner.usda").write_text(inner)
    (source / "root.usda").write_text(
        '#usda 1.0\n(\n    defaultPrim = "Pkg"\n    subLayers = [@./inner.usda@]\n)\n'
    )
    staged = package.with_name("staged.usdz")
    assert UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(source / "root.usda")), str(staged))
    os.replace(staged, package)
    _touch_later(package)


class TestChangedPackage:
    """A layer inside a .usdz changes with its package file."""

    def test_reload_rereads_the_layers_of_a_replaced_package(self, session, tmp_path) -> None:
        package = _package(tmp_path)
        stage_id = session.load_stage(package)
        inner = f"{package.resolve()}[inner.usda]"
        _replace_package(package, '#usda 1.0\ndef Xform "Pkg"\n{\n    def Xform "Replaced"\n    {\n    }\n}\n')

        assert inner in session.stage_status(stage_id)["changed_on_disk"]
        result = session.reload_stage(stage_id)

        stage = session.get_stage(stage_id)
        assert stage.GetPrimAtPath("/Pkg/Replaced") and not stage.GetPrimAtPath("/Pkg/FromSublayer")
        assert inner in result["reloaded_files"]
        assert session.stage_status(stage_id)["changed_on_disk"] == []

    def test_validation_reads_a_replaced_package_from_disk(self, session, tmp_path) -> None:
        package = _package(tmp_path)
        session.load_stage(package)  # holds the package's inner layer
        _replace_package(package, "#usda 1.0\n(\n    subLayers = [@./missing.usda@]\n)\n")

        report = validation.validate_usd_file(
            str(package.resolve()), Settings().usd, layer_is_current=session.layer_is_current
        )

        assert "MissingSublayer" in [issue.code for issue in report.issues]
        assert not report.is_valid


class TestUsdzExport:
    """A stage loaded from a .usdz exports without flatten, and the result composes like the package."""

    @pytest.mark.parametrize("suffix", [".usda", ".usdc", ".usdz"])
    def test_a_usdz_stage_exports_with_its_package_content(self, session, tmp_path, suffix) -> None:
        stage_id = session.load_stage(_package(tmp_path))
        session.create_prim(stage_id, "/Pkg/Edited", "Xform")
        (tmp_path / "out").mkdir()
        destination = tmp_path / "out" / f"edited{suffix}"

        result = session.export_stage(stage_id, str(destination))

        assert result["rewritten_asset_paths"] == 2
        exported = Usd.Stage.Open(str(destination))
        assert exported.GetPrimAtPath("/Pkg/FromSublayer")
        assert exported.GetPrimAtPath("/Pkg/Edited")
        texture = exported.GetPrimAtPath("/Pkg").GetAttribute("custom:texture").Get()
        assert texture.resolvedPath
        if suffix != ".usdz":
            assert list(exported.GetRootLayer().subLayerPaths) == ["../pkg/pkg.usdz[inner.usda]"]
        assert validation.validate_usd_file(str(destination), Settings().usd).is_valid

    def test_a_reference_into_another_package_is_reanchored(self, session, files, tmp_path) -> None:
        package = _package(tmp_path)
        scene = tmp_path / "uses_pkg.usda"
        scene.write_text(
            '#usda 1.0\ndef Xform "World"\n{\n    def "Inner" (\n'
            "        references = @./pkg/pkg.usdz[inner.usda]@</Pkg>\n    )\n    {\n    }\n}\n"
        )
        assert package.exists()
        (tmp_path / "out").mkdir()
        destination = tmp_path / "out" / "copy.usda"

        session.export_stage(session.load_stage(scene), str(destination))

        assert Usd.Stage.Open(str(destination)).GetPrimAtPath("/World/Inner/FromSublayer")


class TestValidationReadsTheDisk:
    """pxr's in-memory copy of a file a loaded stage holds is not what validation checks."""

    _DANGLING_ASSET = '#usda 1.0\n(\n    defaultPrim = "Gone"\n)\ndef Xform "A"\n{\n}\n'
    _USER = (
        '#usda 1.0\ndef Xform "W"\n{\n    def "R" (\n        references = @./asset.usda@</A>\n'
        "    )\n    {\n    }\n}\n"
    )

    @staticmethod
    def _codes(report: validation.ValidationReport) -> List[str]:
        return [issue.code for issue in report.issues]

    def test_a_file_fixed_and_saved_validates_as_fixed(self, session, tmp_path) -> None:
        (tmp_path / "asset.usda").write_text(self._DANGLING_ASSET)
        (tmp_path / "user.usda").write_text(self._USER)
        session.load_stage(tmp_path / "user.usda")  # holds asset.usda in memory
        stale = validation.validate_usd_file(str(tmp_path / "asset.usda"), Settings().usd)
        assert "DanglingDefaultPrim" in self._codes(stale)
        fix = session.load_stage(tmp_path / "asset.usda")
        session.create_prim(fix, "/Gone", "Xform")
        session.save_stage(fix)

        report = validation.validate_usd_file(str(tmp_path / "asset.usda"), Settings().usd)

        assert report.is_valid, self._codes(report)

    def test_a_sublayer_broken_on_disk_fails_validation(self, session, files) -> None:
        session.load_stage(files["scene"])  # holds sub.usda in memory
        files["sub"].write_text(_BROKEN)
        _touch_later(files["sub"])

        report = validation.validate_usd_file(str(files["scene"]), Settings().usd)

        assert not report.is_valid
        assert "MissingSublayer" in self._codes(report)
        assert report.files[0] == str(files["scene"].resolve())

    def test_a_file_no_stage_holds_is_validated_in_process(self, files, monkeypatch) -> None:
        def no_child(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("validation started a process although nothing was in memory")

        monkeypatch.setattr(validation.subprocess, "run", no_child)

        report = validation.validate_usd_file(str(files["scene"]), Settings().usd)

        assert report.can_read and report.files[0] == str(files["scene"].resolve())

    def test_a_file_that_now_reaches_outside_is_refused(self, session, tmp_path) -> None:
        inside, outside = tmp_path / "inside", tmp_path / "outside"
        inside.mkdir()
        outside.mkdir()
        (outside / "secret.usda").write_text('#usda 1.0\ndef "S"\n{\n}\n')
        (inside / "asset.usda").write_text(_ASSET)
        (inside / "scene.usda").write_text(_SCENE)
        policy = PathPolicy(enabled=True, allowed_paths=[str(inside)])
        session.load_stage_or_raise(inside / "scene.usda", sandbox=CompositionSandbox(policy))
        (inside / "asset.usda").write_text(_ASSET_REACHING_OUT)
        _touch_later(inside / "asset.usda")

        with pytest.raises(SandboxDenied) as caught:
            validation.validate_usd_file(
                str(inside / "scene.usda"), Settings().usd, sandbox=CompositionSandbox(policy)
            )

        assert caught.value.details["outside_sandbox"][0]["location"] == str((outside / "secret.usda").resolve())

    def test_a_child_that_dies_is_reported_as_an_error(self, session, files, monkeypatch) -> None:
        import subprocess

        session.load_stage(files["scene"])
        monkeypatch.setattr(
            validation.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(a, -11, b"", b"Segmentation fault: 11\n"),
        )

        report = validation.validate_usd_file(str(files["scene"]), Settings().usd)

        assert report.can_read is False
        assert [(issue.severity, issue.code) for issue in report.issues] == [("error", "OpenFailed")]
        assert "Segmentation fault" in report.issues[0].message

    def test_a_child_that_times_out_raises_with_the_setting(self, session, files, monkeypatch) -> None:
        import subprocess

        from simul.config import USDConfig

        session.load_stage(files["scene"])
        seen: Dict[str, Any] = {}

        def too_slow(*args: Any, **kwargs: Any) -> Any:
            seen["timeout"] = kwargs["timeout"]
            raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

        monkeypatch.setattr(validation.subprocess, "run", too_slow)

        with pytest.raises(TimeoutError, match="usd.validation_timeout_seconds"):
            validation.validate_usd_file(str(files["scene"]), USDConfig(validation_timeout_seconds=7))

        assert seen["timeout"] == 7

    @pytest.mark.parametrize("failure", ["exit", "start"])
    def test_a_child_that_cannot_run_falls_back_with_a_warning(self, session, files, monkeypatch, failure) -> None:
        """No pxr in the child, or no child at all, says nothing about the file."""
        import subprocess

        session.load_stage(files["scene"])

        def broken(*args: Any, **kwargs: Any) -> Any:
            if failure == "start":
                raise FileNotFoundError("no python there")
            return subprocess.CompletedProcess(args, 1, b"", b"ModuleNotFoundError: No module named 'pxr'\n")

        monkeypatch.setattr(validation.subprocess, "run", broken)

        report = validation.validate_usd_file(str(files["scene"]), Settings().usd)

        assert report.can_read is True and report.is_valid is True
        warning = [issue for issue in report.issues if issue.code == "ValidatedInMemoryCopy"]
        assert len(warning) == 1 and warning[0].severity == "warning"
        assert ("ModuleNotFoundError" if failure == "exit" else "could not start") in warning[0].message
        assert str(files["sub"].resolve()) in warning[0].message

    def test_held_layers_unchanged_since_read_are_validated_in_process(self, session, files, monkeypatch) -> None:
        import subprocess

        session.load_stage(files["scene"])  # holds sub.usda and robot.usda, unchanged
        calls: List[Any] = []

        def child(*args: Any, **kwargs: Any) -> Any:
            calls.append(args)
            return subprocess.CompletedProcess(args, -11, b"", b"")

        monkeypatch.setattr(validation.subprocess, "run", child)

        report = validation.validate_usd_file(
            str(files["scene"]), Settings().usd, layer_is_current=session.layer_is_current
        )

        assert calls == [] and report.is_valid is True

        _edit_externally(files["sub"], "/World/AddedOutside")
        stale = validation.validate_usd_file(
            str(files["scene"]), Settings().usd, layer_is_current=session.layer_is_current
        )

        assert len(calls) == 1 and [issue.code for issue in stale.issues] == ["OpenFailed"]

    def test_the_fresh_process_lets_unreal_content_paths_through(self, session, tmp_path, monkeypatch) -> None:
        """An Unreal export's /Engine/... material paths pass the sandbox in the child as in the server."""
        (tmp_path / "Cube.usda").write_text(
            '#usda 1.0\n(\n    defaultPrim = "Cube"\n)\ndef Xform "Cube"\n{\n    def Shader "UnrealShader"\n    {\n'
            '        uniform token info:implementationSource = "sourceAsset"\n'
            "        uniform asset info:unreal:sourceAsset = "
            "@/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial@\n    }\n}\n"
        )
        (tmp_path / "level.usda").write_text(
            '#usda 1.0\n(\n    defaultPrim = "Root"\n    metersPerUnit = 0.01\n    upAxis = "Z"\n)\n'
            'def Xform "Root"\n{\n    def "Cube" (\n        prepend references = @./Cube.usda@\n    )\n'
            "    {\n    }\n}\n"
        )
        policy = PathPolicy(enabled=True, allowed_paths=[str(tmp_path)])
        session.load_stage_or_raise(tmp_path / "level.usda", sandbox=CompositionSandbox(policy))
        _touch_later(tmp_path / "Cube.usda")  # the held copy is now older than the file
        real_run, runs = validation.subprocess.run, []

        def spy(*args: Any, **kwargs: Any) -> Any:
            runs.append(args)
            return real_run(*args, **kwargs)

        monkeypatch.setattr(validation.subprocess, "run", spy)

        report = validation.validate_usd_file(
            str(tmp_path / "level.usda"),
            Settings().usd,
            sandbox=CompositionSandbox(policy),
            layer_is_current=session.layer_is_current,
        )

        assert len(runs) == 1
        assert report.is_valid, [(issue.code, issue.message) for issue in report.issues if issue.severity == "error"]
        assert report.files and report.files[0] == str((tmp_path / "level.usda").resolve())

    def test_the_child_gets_the_sandbox_options(self, session, tmp_path, monkeypatch) -> None:
        import pickle
        import subprocess

        (tmp_path / "sub.usda").write_text('#usda 1.0\ndef Xform "World"\n{\n}\n')
        (tmp_path / "root.usda").write_text("#usda 1.0\n(\n    subLayers = [@./sub.usda@]\n)\n")
        session.load_stage(tmp_path / "root.usda")
        _touch_later(tmp_path / "sub.usda")
        policy = PathPolicy(enabled=True, allowed_paths=[str(tmp_path)])
        sent: Dict[str, Any] = {}

        def capture(*args: Any, **kwargs: Any) -> Any:
            sent.update(pickle.loads(kwargs["input"]))
            return subprocess.CompletedProcess(args, -11, b"", b"")

        monkeypatch.setattr(validation.subprocess, "run", capture)

        report = validation.validate_usd_file(
            str(tmp_path / "root.usda"), Settings().usd, sandbox=CompositionSandbox(policy)
        )

        assert set(sent["sandbox"]) == set(CompositionSandbox(policy).options())
        assert sent["sandbox"]["policy"].allowed_roots == policy.allowed_roots
        assert [(issue.code, "signal 11" in issue.message) for issue in report.issues] == [("OpenFailed", True)]


class TestDiscard:
    def test_reload_refuses_unsaved_edits_unless_discarded(self, session, files) -> None:
        stage_id = session.load_stage(files["robot"])
        session.delete_prim(stage_id, "/Robot/link2")

        error = _refusal(session.reload_stage, stage_id)
        assert error.details["override"] == "discard_unsaved"
        assert not session.get_stage(stage_id).GetPrimAtPath("/Robot/link2")

        result = session.reload_stage(stage_id, discard_unsaved=True)

        assert result["discarded_unsaved"] is True
        assert session.get_stage(stage_id).GetPrimAtPath("/Robot/link2")

    def test_unload_refuses_unsaved_edits_unless_discarded(self, session, files) -> None:
        stage_id = session.load_stage(files["robot"])
        session.create_prim(stage_id, "/Robot/Extra", "Xform")

        error = _refusal(session.unload_stage, stage_id)
        assert error.details["override"] == "discard_unsaved"
        assert session.get_stage(stage_id) is not None

        assert session.unload_stage(stage_id, discard_unsaved=True) is True
        assert session.get_stage(stage_id) is None
        assert not session.get_stage(session.load_stage(files["robot"])).GetPrimAtPath("/Robot/Extra")


_ASSET = '#usda 1.0\n(\n    defaultPrim = "A"\n)\ndef Xform "A"\n{\n}\n'
_ASSET_REACHING_OUT = (
    '#usda 1.0\n(\n    defaultPrim = "A"\n)\ndef Xform "A"\n{\n    def "Extra" (\n'
    "        references = @../outside/secret.usda@\n    )\n    {\n    }\n}\n"
)
_SCENE = (
    '#usda 1.0\ndef Xform "World"\n{\n    def "Asset" (\n'
    "        references = @./asset.usda@\n    )\n    {\n    }\n}\n"
)
_ASSET_WITH_A_CHILD = _ASSET.replace('def Xform "A"\n{\n}', 'def Xform "A"\n{\n    def "Added"\n    {\n    }\n}')


class TestRereadsStayInTheSandbox:
    """A changed shared file is held to the composition sandbox before it is reread in place.

    A reload recomposes every stage using the layer, other sessions' too, so
    without the check a file rewritten to reference ``../outside`` would put
    outside content into a stage that passed the sandbox when it loaded.
    """

    @pytest.fixture
    def tree(self, tmp_path: Path) -> Dict[str, Path]:
        inside, outside = tmp_path / "inside", tmp_path / "outside"
        inside.mkdir()
        outside.mkdir()
        (outside / "secret.usda").write_text(
            '#usda 1.0\n(\n    defaultPrim = "S"\n)\ndef "S"\n{\n    custom string marker = "OUTSIDE"\n}\n'
        )
        (inside / "asset.usda").write_text(_ASSET)
        for name in ("scene", "other"):
            (inside / f"{name}.usda").write_text(_SCENE)
        return {"inside": inside, "outside": outside}

    @pytest.fixture
    def guarded(self, session: HeadlessUSDSession, tree: Dict[str, Path]) -> HeadlessUSDSession:
        policy = PathPolicy(enabled=True, allowed_paths=[str(tree["inside"])])
        session.set_sandbox_factory(lambda: CompositionSandbox(policy))
        return session

    @staticmethod
    def _rewrite(path: Path, text: str) -> None:
        path.write_text(text)
        _touch_later(path)

    @staticmethod
    def _load(session: HeadlessUSDSession, path: Path) -> str:
        """Load as the server does: the load's own sandbox from the same policy."""
        return session.load_stage_or_raise(path, sandbox=session._sandbox_factory())

    def test_a_load_does_not_reread_a_shared_file_that_now_reaches_outside(self, guarded, tree) -> None:
        scene = self._load(guarded, tree["inside"] / "scene.usda")
        self._rewrite(tree["inside"] / "asset.usda", _ASSET_REACHING_OUT)

        with pytest.raises(SandboxDenied) as caught:
            self._load(guarded, tree["inside"] / "other.usda")

        assert caught.value.details["outside_sandbox"][0]["layer"] == str((tree["inside"] / "asset.usda").resolve())
        stage = guarded.get_stage(scene)
        assert not stage.GetPrimAtPath("/World/Asset/Extra")
        assert guarded.list_stages() == [scene]

    def test_reload_refuses_a_changed_file_that_reaches_outside(self, guarded, tree) -> None:
        scene = self._load(guarded, tree["inside"] / "scene.usda")
        self._rewrite(tree["inside"] / "asset.usda", _ASSET_REACHING_OUT)

        with pytest.raises(SandboxDenied):
            guarded.reload_stage(scene)
        assert not guarded.get_stage(scene).GetPrimAtPath("/World/Asset/Extra")

        self._rewrite(tree["inside"] / "asset.usda", _ASSET)
        self._rewrite(
            tree["inside"] / "scene.usda", _SCENE.replace("./asset.usda", "../outside/secret.usda")
        )

        with pytest.raises(SandboxDenied):
            guarded.reload_stage(scene)
        stage = guarded.get_stage(scene)
        assert stage is not None and stage.GetPrimAtPath("/World/Asset")
        assert not stage.GetPrimAtPath("/World/Asset").GetAttribute("marker")

    def test_a_refused_reload_keeps_the_unsaved_edits(self, guarded, tree) -> None:
        """Every file is checked before anything is reread, the root included."""
        scene = self._load(guarded, tree["inside"] / "scene.usda")
        guarded.create_prim(scene, "/World/Unsaved", "Xform")
        self._rewrite(tree["inside"] / "asset.usda", _ASSET_REACHING_OUT)

        with pytest.raises(SandboxDenied):
            guarded.reload_stage(scene, discard_unsaved=True)

        stage = guarded.get_stage(scene)
        assert stage.GetPrimAtPath("/World/Unsaved")
        assert guarded.stage_status(scene)["has_unsaved_changes"] is True
        assert not stage.GetPrimAtPath("/World/Asset/Extra")

    def test_a_refused_reload_keeps_the_edits_when_the_new_root_names_a_held_layer(self, guarded, tree) -> None:
        """The new root content sublayers a file another stage holds, whose file now reaches outside."""
        inside = tree["inside"]
        (inside / "shared.usda").write_text('#usda 1.0\ndef Xform "World"\n{\n}\n')
        (inside / "holder.usda").write_text("#usda 1.0\n(\n    subLayers = [@./shared.usda@]\n)\n")
        (inside / "p.usda").write_text('#usda 1.0\ndef Xform "P"\n{\n}\n')
        self._load(guarded, inside / "holder.usda")
        p = self._load(guarded, inside / "p.usda")
        guarded.create_prim(p, "/P/Unsaved", "Xform")
        self._rewrite(inside / "shared.usda", "#usda 1.0\n(\n    subLayers = [@../outside/secret.usda@]\n)\n")
        self._rewrite(inside / "p.usda", '#usda 1.0\n(\n    subLayers = [@./shared.usda@]\n)\ndef Xform "P"\n{\n}\n')

        with pytest.raises(SandboxDenied):
            guarded.reload_stage(p, discard_unsaved=True)

        assert guarded.get_stage(p).GetPrimAtPath("/P/Unsaved")
        assert guarded.stage_status(p)["has_unsaved_changes"] is True

    def test_an_allowed_change_is_reread_by_every_stage_using_it(self, guarded, tree) -> None:
        scene = self._load(guarded, tree["inside"] / "scene.usda")
        self._rewrite(tree["inside"] / "asset.usda", _ASSET_WITH_A_CHILD)

        other = self._load(guarded, tree["inside"] / "other.usda")

        assert guarded.get_stage(other).GetPrimAtPath("/World/Asset/Added")
        assert guarded.get_stage(scene).GetPrimAtPath("/World/Asset/Added")

    def test_a_stage_that_reaches_outside_after_a_reread_is_dropped(self, guarded, tree, monkeypatch) -> None:
        """The backstop: a file swapped between the check and the reload."""
        scene = self._load(guarded, tree["inside"] / "scene.usda")
        self._rewrite(tree["inside"] / "asset.usda", _ASSET_REACHING_OUT)
        real = CompositionSandbox.check_layers
        swapped = {"done": False}

        def check_old_content_once(sandbox: CompositionSandbox, layer: Any) -> Any:
            if not swapped["done"]:
                swapped["done"] = True
                return []  # as if the file still had its old content when it was checked
            return real(sandbox, layer)

        monkeypatch.setattr(CompositionSandbox, "check_layers", check_old_content_once)

        with pytest.raises(SandboxDenied):
            guarded.reload_stage(scene)

        assert guarded.get_stage(scene) is None
