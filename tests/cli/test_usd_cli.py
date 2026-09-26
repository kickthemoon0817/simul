"""CLI tests for `simul usd info` against a real generated .usda."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from simul_mcp.cli import usd_cli
from simul_mcp.cli.main import app

pxr = pytest.importorskip("pxr")

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
