"""Loaded stages are isolated, notice changes on disk, and can be saved, exported, reloaded and unloaded.

These run the headless session against real ``pxr`` stages on disk: a robot
file, a scene that sublayers one file and references the robot, and the
operations an agent needs to keep or drop its edits. The session's owner
resolver stands in for the MCP session id, so the per-session isolation the
server relies on is tested without a server.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterator

import pytest

pytest.importorskip("pxr", reason="pxr library not available")

from pxr import Sdf, Usd, UsdGeom  # noqa: E402

from simul.adapters import headless_usd  # noqa: E402
from simul.adapters.headless_usd import HeadlessUSDSession  # noqa: E402
from simul.usd.editing import USDEditError  # noqa: E402


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
