"""CLI tests for `simul usd info`, `validate` and `summary` against real generated files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from simul.cli import usd_cli
from simul.cli.main import app

pxr = pytest.importorskip("pxr")

from tests import usd_fixtures  # noqa: E402

runner = CliRunner()


@pytest.fixture
def usda_file(tmp_path: Path) -> Path:
    from pxr import Usd, UsdGeom

    path = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Cube.Define(stage, "/World/Cube")
    mesh = UsdGeom.Mesh.Define(stage, "/World/Tri")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    stage.GetRootLayer().Save()
    return path


def test_usd_info_json(usda_file: Path) -> None:
    result = runner.invoke(app, ["--json", "usd", "info", str(usda_file)])
    assert result.exit_code == 0, result.output
    line = next(ln for ln in result.stdout.splitlines() if ln.startswith("{"))
    data = json.loads(line)
    assert data["success"] is True
    assert data["total_prims"] == 3
    assert data["root_prims"] == 1
    assert data["default_prim"] == "/World"
    assert data["up_axis"] == "Z"
    assert data["prim_type_counts"]["Mesh"] == 1


def test_usd_info_human(usda_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(usd_cli, "is_json_mode", lambda: False)
    result = runner.invoke(app, ["usd", "info", str(usda_file)])
    assert result.exit_code == 0, result.output
    assert "Total Prims" in result.output
    assert "/World" in result.output
    assert "Done" in result.output


def _json_line(output: str) -> dict:
    return json.loads(next(ln for ln in output.splitlines() if ln.startswith("{")))


@pytest.mark.parametrize(
    ("builder", "name", "code"),
    [
        ("write_corrupt_usda", "corrupt.usda", "ParseError"),
        ("write_empty", "empty.usda", "EmptyFile"),
        ("write_garbage", "garbage.usdc", "CorruptFile"),
        ("write_text_named_usdc", "text.usdc", "CorruptFile"),
        ("write_dangling_default_prim", "dangling.usda", "DanglingDefaultPrim"),
        ("write_bad_physics", "bad_physics.usda", "NestedRigidBody"),
    ],
)
def test_usd_validate_exits_1_for_invalid_files(
    tmp_path: Path, builder: str, name: str, code: str
) -> None:
    """``simul usd validate`` used to exit 0 "valid" for a file with a syntax error."""
    path = getattr(usd_fixtures, builder)(tmp_path / name)

    result = runner.invoke(app, ["--json", "usd", "validate", str(path)])

    assert result.exit_code == 1, result.output
    data = _json_line(result.stdout)
    assert data["success"] is False
    assert data["valid"] is False
    assert data["is_valid"] is False
    assert data["error_type"] == "ValidationFailed"
    assert code in {issue["code"] for issue in data["issues"]}


def test_usd_validate_broken_references_exit_1(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["--json", "usd", "validate", str(usd_fixtures.write_broken_refs(tmp_path))]
    )

    assert result.exit_code == 1, result.output
    data = _json_line(result.stdout)
    assert data["can_read"] is True
    assert "MissingReference" in {issue["code"] for issue in data["issues"]}


def test_usd_validate_valid_file_exits_0(tmp_path: Path) -> None:
    path = usd_fixtures.write_robot(tmp_path / "robot.usda")

    result = runner.invoke(app, ["--json", "usd", "validate", str(path)])

    assert result.exit_code == 0, result.output
    data = _json_line(result.stdout)
    assert (data["success"], data["valid"], data["issues"]) == (True, True, [])


def test_usd_validate_human_lists_issues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(usd_cli, "is_json_mode", lambda: False)
    path = usd_fixtures.write_corrupt_usda(tmp_path / "corrupt.usda")

    result = runner.invoke(app, ["usd", "validate", str(path)], terminal_width=200)

    assert result.exit_code == 1
    assert "ParseError" in result.output
    assert "line 9, column 1" in result.output
    assert "Validation failed" in result.output


@pytest.mark.parametrize("command", ["info", "summary"])
def test_usd_load_failure_names_pxr_reason(tmp_path: Path, command: str) -> None:
    path = usd_fixtures.write_corrupt_usda(tmp_path / "corrupt.usda")

    result = runner.invoke(app, ["--json", "usd", command, str(path)])

    assert result.exit_code == 1
    data = _json_line(result.stdout)
    assert data["error_type"] == "LoadError"
    assert data["details"]["reason"] == "ParseError"
    assert "line 9, column 1" in data["error"]


def test_usd_info_lists_load_warnings(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["--json", "usd", "info", str(usd_fixtures.write_broken_refs(tmp_path))]
    )

    assert result.exit_code == 0, result.output
    data = _json_line(result.stdout)
    assert "MissingReference" in {warning["code"] for warning in data["warnings"]}
