"""Materials, import, save, graph and extension tools do what they report.

Each defect pinned here reported success for work that did not happen, or
did the wrong work:

- ``set_isaac_material_property`` created missing inputs with the wrong type
  (bool -> float, texture path -> string) and created dead inputs for names
  the shader never reads, the tool's own example among them;
- ``import_isaac_asset`` reported ``imported: true`` for an asset without a
  defaultPrim (nothing composed), merged a second import into the first, and
  ignored a metersPerUnit / upAxis mismatch;
- ``save_isaac_stage`` saved in place without the sandbox seeing the path,
  trusted ``os.path.exists`` for ``omniverse://`` targets, and saved
  simulated poses mid-simulation;
- ``set_isaac_graph_node_values`` returned success when every attribute failed;
- with script execution disabled, a ScriptNode still ran agent Python;
- ``disable_isaac_extension`` reported success for an extension that stayed on.

USD-level scripts run against real pxr through ``conftest.run_on_stage``; the
Kit-only parts (timeline, OmniGraph, the extension manager, the save calls)
run against small fakes that record what the script asked for.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import io
import json
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from simul.adapters.isaac_socket_client import ScriptResult
from simul.config import Settings
from simul.mcp.registration._helpers import apply_success_from_error
from simul.mcp.tools.isaac.graph import (
    extension_bare_name,
    is_script_attribute,
    is_script_capable_node_type,
    script_guard_source,
)
from simul.mcp.tools.isaac_tools import IsaacTools

Captured = Tuple[IsaacTools, List[str]]


def _settings(*, sandbox: bool = True, scripts: bool = True, write_schemes: Tuple[str, ...] = ()) -> Settings:
    base = Settings()
    return base.model_copy(
        update={
            "security": base.security.model_copy(
                update={
                    "sandbox_enabled": sandbox,
                    "allow_script_execution": scripts,
                    "allowed_write_url_schemes": list(write_schemes),
                }
            )
        }
    )


def _recording_tools(settings: Settings, outputs: Optional[List[Dict[str, Any]]] = None) -> Captured:
    """IsaacTools whose client records scripts and answers with ``outputs`` in turn."""
    captured: List[str] = []
    replies = list(outputs or [])

    def _record(code: str, **_kwargs: Any) -> ScriptResult:
        captured.append(code)
        reply = replies.pop(0) if replies else {"ok": True}
        return ScriptResult(success=True, output=json.dumps(reply))

    client = MagicMock()
    client.address = "127.0.0.1:8226"
    client.timeout_seconds = 30.0
    client.bridge_enabled = False
    client.fallback_to_vscode = True
    client.execute = AsyncMock(side_effect=_record)
    client.execute_vscode_only = AsyncMock(side_effect=_record)
    client.execute_bridge_script_only = AsyncMock(side_effect=_record)
    return IsaacTools(client, settings=settings), captured


def _call(tools: Captured, run_on_stage: Any, stage: Usd.Stage, method: str, **kwargs: Any) -> Dict[str, Any]:
    """Run one tool against ``stage``; a call refused before sending returns as is."""
    isaac, captured = tools
    sent = len(captured)
    payload = asyncio.run(getattr(isaac, method)(**kwargs))
    if len(captured) == sent:
        return payload
    return apply_success_from_error(run_on_stage(captured[-1], stage))


async def _run_kit_script_async(
    script: str, monkeypatch: pytest.MonkeyPatch, modules: Dict[str, types.ModuleType]
) -> Dict[str, Any]:
    """Run a generated script the way Kit does (top-level await) against fake modules."""
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    code = compile(script, "<isaac-script>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        # The script is one IsaacTools just generated, nothing external; eval
        # is how a top-level-await module runs in-process.
        result = eval(code, {"__name__": "__isaac_script__"})  # noqa: S307
        if asyncio.iscoroutine(result):
            await result
    return apply_success_from_error(json.loads(stdout.getvalue()))


def _run_kit_script(
    script: str, monkeypatch: pytest.MonkeyPatch, modules: Dict[str, types.ModuleType]
) -> Dict[str, Any]:
    """``_run_kit_script_async`` outside an event loop."""
    return asyncio.run(_run_kit_script_async(script, monkeypatch, modules))


# ---------------------------------------------------------------------------
# F11: set_isaac_material_property types inputs and refuses dead names
# ---------------------------------------------------------------------------


@pytest.fixture
def omnipbr_stage(capturing_tools: Captured, run_on_stage: Any) -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    _call(
        capturing_tools, run_on_stage, stage, "create_isaac_material",
        material_path="/World/Looks/M", shader_type="OmniPBR",
    )
    return stage


def _shader(stage: Usd.Stage) -> UsdShade.Shader:
    return UsdShade.Shader(stage.GetPrimAtPath("/World/Looks/M/Shader"))


@pytest.mark.parametrize(
    ("name", "value", "type_name", "read_back"),
    [
        ("project_uvw", True, "bool", True),
        ("diffuse_texture", "/tmp/simul-work/textures/wood.png", "asset", "/tmp/simul-work/textures/wood.png"),
        ("uv_space_index", 1, "int", 1),
        ("texture_scale", [2, 2], "float2", [2.0, 2.0]),
        ("emissive_color", [1, 0.5, 0], "color3f", [1.0, 0.5, 0.0]),
        ("inputs:metallic_texture", "/tmp/simul-work/metal.png", "asset", "/tmp/simul-work/metal.png"),
    ],
)
def test_new_omnipbr_inputs_take_the_shader_type(
    capturing_tools: Captured, run_on_stage: Any, omnipbr_stage: Usd.Stage,
    name: str, value: Any, type_name: str, read_back: Any,
) -> None:
    result = _call(
        capturing_tools, run_on_stage, omnipbr_stage, "set_isaac_material_property",
        material_path="/World/Looks/M", property_name=name, value=value,
    )

    bare = name.split(":", 1)[-1]
    assert result["success"] is True, result
    assert result["created"] is True
    assert result["input_type"] == type_name
    if type_name == "asset":
        # A path is written as the sandbox resolved it (/tmp is /private/tmp on macOS).
        read_back = capturing_tools[0]._path_policy.authorize(read_back)
    assert result["value_set"] == read_back
    shader_input = _shader(omnipbr_stage).GetInput(bare)
    assert str(shader_input.GetTypeName()) == type_name


def test_the_old_example_name_is_refused_with_the_real_one_suggested(
    capturing_tools: Captured, run_on_stage: Any, omnipbr_stage: Usd.Stage
) -> None:
    result = _call(
        capturing_tools, run_on_stage, omnipbr_stage, "set_isaac_material_property",
        material_path="/World/Looks/M", property_name="roughness_constant", value=0.9,
    )

    assert result["success"] is False
    assert "reflection_roughness_constant" in result["error"]
    assert result["error_type"] == "ValidationError" and result["details"]["code"] == "UNKNOWN_INPUT"
    missing = _call(
        capturing_tools, run_on_stage, omnipbr_stage, "set_isaac_material_property",
        material_path="/World/Looks/Nope", property_name="metallic_constant", value=1.0,
    )
    assert missing["error_type"] == "NotFoundError"
    assert not _shader(omnipbr_stage).GetInput("roughness_constant")
    real = _call(
        capturing_tools, run_on_stage, omnipbr_stage, "set_isaac_material_property",
        material_path="/World/Looks/M", property_name="reflection_roughness_constant", value=0.9,
    )
    assert real["created"] is False and real["value_set"] == pytest.approx(0.9)


def test_a_value_that_does_not_fit_the_input_is_refused_and_creates_nothing(
    capturing_tools: Captured, run_on_stage: Any, omnipbr_stage: Usd.Stage
) -> None:
    flag = _call(
        capturing_tools, run_on_stage, omnipbr_stage, "set_isaac_material_property",
        material_path="/World/Looks/M", property_name="enable_emission", value="yes",
    )
    color = _call(
        capturing_tools, run_on_stage, omnipbr_stage, "set_isaac_material_property",
        material_path="/World/Looks/M", property_name="diffuse_color_constant", value=[1, 0],
    )

    assert flag["success"] is False and "true or false" in flag["error"]
    assert not _shader(omnipbr_stage).GetInput("enable_emission")
    assert color["success"] is False and "3 numbers" in color["error"]
    assert list(_shader(omnipbr_stage).GetInput("diffuse_color_constant").Get()) == pytest.approx([0.8] * 3)


def test_allow_unknown_input_creates_an_inferred_input_with_a_warning(
    capturing_tools: Captured, run_on_stage: Any, omnipbr_stage: Usd.Stage
) -> None:
    result = _call(
        capturing_tools, run_on_stage, omnipbr_stage, "set_isaac_material_property",
        material_path="/World/Looks/M", property_name="custom_flag", value=True,
        allow_unknown_input=True,
    )

    assert result["success"] is True and result["input_type"] == "bool"
    assert "not a known OmniPBR input" in result["warning"]


def test_usd_preview_surface_inputs_are_checked_against_its_own_list(
    capturing_tools: Captured, run_on_stage: Any
) -> None:
    stage = Usd.Stage.CreateInMemory()
    _call(capturing_tools, run_on_stage, stage, "create_isaac_material", material_path="/Looks/P")

    ior = _call(
        capturing_tools, run_on_stage, stage, "set_isaac_material_property",
        material_path="/Looks/P", property_name="ior", value=1.45,
    )
    omnipbr_name = _call(
        capturing_tools, run_on_stage, stage, "set_isaac_material_property",
        material_path="/Looks/P", property_name="reflection_roughness_constant", value=0.2,
    )

    assert ior["success"] is True and ior["input_type"] == "float" and ior["created"] is True
    assert omnipbr_name["success"] is False and "UsdPreviewSurface" in omnipbr_name["error"]


def test_an_empty_property_name_is_refused_before_sending(capturing_tools: Captured) -> None:
    tools, captured = capturing_tools
    result = asyncio.run(tools.set_isaac_material_property("/World/Looks/M", "  ", 1.0))
    assert result["success"] is False and captured == []


# ---------------------------------------------------------------------------
# F14: import_isaac_asset / add_isaac_reference compose what they report
# ---------------------------------------------------------------------------


def _asset(
    path: Path, *, default_prim: bool = True, up_axis: Optional[str] = "Z",
    meters_per_unit: Optional[float] = 1.0, prims: bool = True,
    root_ops: Optional[str] = None, tip_at: Tuple[float, float, float] = (0, 100, 0),
) -> str:
    layer_stage = Usd.Stage.CreateNew(str(path))
    if up_axis is not None:
        UsdGeom.SetStageUpAxis(layer_stage, up_axis)
    if meters_per_unit is not None:
        UsdGeom.SetStageMetersPerUnit(layer_stage, meters_per_unit)
    if prims:
        root = UsdGeom.Xform.Define(layer_stage, "/Robot")
        if root_ops == "translate":
            root.AddTranslateOp().Set(Gf.Vec3d(0, 100, 0))
        elif root_ops == "matrix":
            root.AddTransformOp().Set(Gf.Matrix4d(1).SetTranslate(Gf.Vec3d(0, 100, 0)))
        tip = UsdGeom.Xform.Define(layer_stage, "/Robot/tip")
        tip.AddTranslateOp().Set(Gf.Vec3d(*tip_at))
        if default_prim:
            layer_stage.SetDefaultPrim(layer_stage.GetPrimAtPath("/Robot"))
    layer_stage.Save()
    return str(path)


def _world_position(stage: Usd.Stage, path: str) -> List[float]:
    xformable = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    return list(xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation())


@pytest.fixture
def open_tools() -> Captured:
    """Tools with the sandbox off, so assets under tmp_path can be referenced."""
    return _recording_tools(_settings(sandbox=False))


def _world_stage() -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    return stage


def test_an_asset_without_default_prim_references_its_first_root_prim(
    open_tools: Captured, run_on_stage: Any, tmp_path: Path
) -> None:
    asset = _asset(tmp_path / "nodefault.usda", default_prim=False)
    stage = _world_stage()

    result = _call(open_tools, run_on_stage, stage, "import_isaac_asset", asset_path=asset)

    assert result["success"] is True, result
    assert result["referenced_prim"] == "/Robot" and result["used_default_prim"] is False
    assert "no defaultPrim" in result["warning"]
    prim = stage.GetPrimAtPath("/World/ImportedAsset")
    assert prim.GetTypeName() == "Xform"
    assert [child.GetName() for child in prim.GetChildren()] == ["tip"]


def test_an_asset_with_nothing_to_reference_fails_and_leaves_nothing(
    open_tools: Captured, run_on_stage: Any, tmp_path: Path
) -> None:
    empty = _asset(tmp_path / "empty.usda", prims=False)
    stage = _world_stage()

    result = _call(open_tools, run_on_stage, stage, "import_isaac_asset", asset_path=empty)

    assert result["success"] is False and "no root prim" in result["error"]
    assert not stage.GetPrimAtPath("/World/ImportedAsset").IsValid()


def test_a_reference_that_composes_nothing_is_undone(
    open_tools: Captured, run_on_stage: Any, tmp_path: Path
) -> None:
    asset = _asset(tmp_path / "robot.usda")
    other = _asset(tmp_path / "other.usda")
    stage = _world_stage()
    holder = stage.DefinePrim("/World/Holder")
    holder.GetReferences().AddReference(other)

    imported = _call(
        open_tools, run_on_stage, stage, "import_isaac_asset", asset_path=asset, asset_prim_path="/Nope"
    )
    referenced = _call(
        open_tools, run_on_stage, stage, "add_isaac_reference",
        prim_path="/World/Holder", reference_path=asset, asset_prim_path="/Nope",
    )

    assert imported["success"] is False and "No prim /Nope" in imported["error"]
    assert imported["error_type"] == referenced["error_type"] == "NotFoundError"
    assert referenced["success"] is False
    assert list(holder.GetMetadata("references").GetAppliedItems()) == [Sdf.Reference(other)]


def test_importing_twice_never_merges_into_the_first_import(
    open_tools: Captured, run_on_stage: Any, tmp_path: Path
) -> None:
    first = _asset(tmp_path / "a.usda")
    second = _asset(tmp_path / "b.usda")
    stage = _world_stage()

    one = _call(open_tools, run_on_stage, stage, "import_isaac_asset", asset_path=first)
    two = _call(open_tools, run_on_stage, stage, "import_isaac_asset", asset_path=second)

    assert one["target_path"] == "/World/ImportedAsset" and one["renamed"] is False
    assert two["target_path"] == "/World/ImportedAsset_1" and two["renamed"] is True
    references = stage.GetPrimAtPath("/World/ImportedAsset").GetMetadata("references")
    assert list(references.GetAppliedItems()) == [Sdf.Reference(first)]


@pytest.mark.parametrize(
    ("root_ops", "tip_at"),
    [(None, (0, 100, 0)), ("translate", (0, 0, 0)), ("matrix", (0, 0, 0))],
)
def test_a_y_up_centimetre_asset_is_converted_to_the_z_up_metre_stage(
    open_tools: Captured, run_on_stage: Any, tmp_path: Path, root_ops: Optional[str], tip_at: Tuple[float, ...]
) -> None:
    """The asset root's own transform is in asset units too, so it must be converted with the rest."""
    asset = _asset(tmp_path / "cm.usda", up_axis="Y", meters_per_unit=0.01, root_ops=root_ops, tip_at=tip_at)
    stage = _world_stage()

    result = _call(open_tools, run_on_stage, stage, "import_isaac_asset", asset_path=asset)

    units = result["units"]
    assert units["applied"] is True
    assert units["scale_needed"] == pytest.approx(0.01) and units["rotate_x_needed"] == 90.0
    assert units["correction_prim"] == "/World/ImportedAsset"
    assert units["xform_op_order"] == ["xformOp:rotateX:unitsResolve", "xformOp:scale:unitsResolve"]
    assert result["asset_root_path"] == "/World/ImportedAsset/Robot"
    # 100 cm up the asset's Y axis is 1 m up the stage's Z axis.
    assert _world_position(stage, "/World/ImportedAsset/Robot/tip") == pytest.approx([0.0, 0.0, 1.0], abs=1e-6)


def test_a_transform_set_after_import_keeps_the_units_correction(
    open_tools: Captured, run_on_stage: Any, tmp_path: Path
) -> None:
    asset = _asset(tmp_path / "cm.usda", up_axis="Y", meters_per_unit=0.01, root_ops="matrix", tip_at=(0, 0, 0))
    stage = _world_stage()
    _call(open_tools, run_on_stage, stage, "import_isaac_asset", asset_path=asset)

    moved = _call(
        open_tools, run_on_stage, stage, "set_isaac_prim_transform",
        prim_path="/World/ImportedAsset", translation=[5, 0, 0], rotation_euler=[0, 0, 90],
    )

    assert moved["success"] is True, moved
    wrapper = UsdGeom.Xformable(stage.GetPrimAtPath("/World/ImportedAsset"))
    order = [str(name) for name in wrapper.GetXformOpOrderAttr().Get()]
    assert order[-2:] == ["xformOp:rotateX:unitsResolve", "xformOp:scale:unitsResolve"]
    # Up 1 m (the root's own translate, converted), then rotated about Z and moved 5 m along X.
    assert _world_position(stage, "/World/ImportedAsset/Robot/tip") == pytest.approx([5.0, 0.0, 1.0], abs=1e-6)


@pytest.mark.parametrize(("root_ops", "applied"), [(None, True), ("translate", False)])
def test_add_reference_corrects_units_in_place_only_without_root_transform_ops(
    open_tools: Captured, run_on_stage: Any, tmp_path: Path, root_ops: Optional[str], applied: bool
) -> None:
    asset = _asset(tmp_path / "cm.usda", up_axis="Y", meters_per_unit=0.01, root_ops=root_ops, tip_at=(0, 0, 0))
    stage = _world_stage()

    result = _call(
        open_tools, run_on_stage, stage, "add_isaac_reference",
        prim_path="/World/Ref", reference_path=asset, correct_units=True,
    )

    assert result["success"] is True and result["units"]["applied"] is applied
    if applied:
        assert "warning" not in result
    else:
        # Asked for and declined: said at the top level, not only in the notes.
        assert "correct_units was not applied" in result["warning"]
        assert "import_isaac_asset" in result["warning"]
        assert any("import_isaac_asset" in note for note in result["units"]["notes"])
        assert not stage.GetPrimAtPath("/World/Ref").HasAttribute("xformOp:scale:unitsResolve")


def test_a_failed_import_removes_the_ancestors_it_created(
    open_tools: Captured, run_on_stage: Any, tmp_path: Path
) -> None:
    dangling = tmp_path / "dangling.usda"
    layer_stage = Usd.Stage.CreateNew(str(dangling))
    UsdGeom.Xform.Define(layer_stage, "/Robot")
    layer_stage.GetRootLayer().defaultPrim = "Missing"
    layer_stage.Save()
    stage = _world_stage()

    imported = _call(
        open_tools, run_on_stage, stage, "import_isaac_asset",
        asset_path=str(dangling), target_path="/World/New/Deep/Asset",
    )
    referenced = _call(
        open_tools, run_on_stage, stage, "add_isaac_reference",
        prim_path="/World/Other/Deep/Ref", reference_path=str(dangling),
    )

    for result in (imported, referenced):
        assert result["success"] is False and result["error_type"] == "ValidationError"
        assert result["details"]["code"] == "REFERENCE_COMPOSED_NOTHING"
    assert [child.GetName() for child in stage.GetPrimAtPath("/World").GetChildren()] == []


def test_units_are_only_reported_when_correction_is_off_or_metrics_are_missing(
    open_tools: Captured, run_on_stage: Any, tmp_path: Path
) -> None:
    mismatched = _asset(tmp_path / "cm.usda", up_axis="Y", meters_per_unit=0.01)
    bare = _asset(tmp_path / "bare.usda", up_axis=None, meters_per_unit=None)
    stage = _world_stage()

    reported = _call(
        open_tools, run_on_stage, stage, "import_isaac_asset", asset_path=mismatched, correct_units=False
    )
    unknown = _call(open_tools, run_on_stage, stage, "import_isaac_asset", asset_path=bare)

    assert reported["units"]["applied"] is False and reported["units"]["scale_needed"] == pytest.approx(0.01)
    assert not stage.GetPrimAtPath("/World/ImportedAsset").HasAttribute("xformOp:scale:unitsResolve")
    assert unknown["units"]["asset_meters_per_unit"] is None
    assert unknown["units"]["applied"] is False
    assert any("centimetres" in note for note in unknown["units"]["notes"])


def test_add_reference_counts_references_and_does_not_duplicate_one(
    open_tools: Captured, run_on_stage: Any, tmp_path: Path
) -> None:
    asset = _asset(tmp_path / "robot.usda")
    stage = _world_stage()

    first = _call(
        open_tools, run_on_stage, stage, "add_isaac_reference", prim_path="/World/Ref", reference_path=asset
    )
    again = _call(
        open_tools, run_on_stage, stage, "add_isaac_reference", prim_path="/World/Ref", reference_path=asset
    )

    assert first["added"] is True and first["total_references"] == 1
    assert again["success"] is True and again["added"] is False and again["total_references"] == 1


def test_a_non_usd_file_names_the_conversion_step(open_tools: Captured, run_on_stage: Any, tmp_path: Path) -> None:
    fbx = tmp_path / "chair.fbx"
    fbx.write_text("not usd")
    stage = _world_stage()

    result = _call(open_tools, run_on_stage, stage, "import_isaac_asset", asset_path=str(fbx))

    assert result["success"] is False and "asset_converter" in result["error"]
    assert not stage.GetPrimAtPath("/World/ImportedAsset").IsValid()


def test_a_relative_asset_prim_path_is_refused_before_sending(open_tools: Captured) -> None:
    tools, captured = open_tools
    result = asyncio.run(tools.import_isaac_asset("/tmp/simul-work/a.usd", asset_prim_path="Robot"))
    assert result["success"] is False and captured == []


# ---------------------------------------------------------------------------
# F16: save_isaac_stage is sandboxed, guards remote targets and the timeline
# ---------------------------------------------------------------------------


class _Timeline:
    def __init__(self, stopped: bool) -> None:
        self.stopped = stopped

    def is_stopped(self) -> bool:
        return self.stopped


class _SaveContext:
    """The omni.usd context calls the save scripts make, recorded."""

    def __init__(self, stage: Any, url: str) -> None:
        self.stage = stage
        self.url = url
        self.saved: List[str] = []

    def get_stage(self) -> Any:
        return self.stage

    def get_stage_url(self) -> str:
        return self.url

    async def save_stage_async(self) -> Tuple[bool, str, List[str]]:
        self.saved.append(self.url)
        return True, "", [self.url]

    async def save_as_stage_async(self, target: str) -> Tuple[bool, str, List[str]]:
        self.saved.append(target)
        return True, "", [target]


def _kit_modules(context: _SaveContext, timeline: _Timeline) -> Dict[str, types.ModuleType]:
    omni = types.ModuleType("omni")
    usd = types.ModuleType("omni.usd")
    usd.get_context = lambda: context  # type: ignore[attr-defined]
    timeline_module = types.ModuleType("omni.timeline")
    timeline_module.get_timeline_interface = lambda: timeline  # type: ignore[attr-defined]
    omni.usd = usd  # type: ignore[attr-defined]
    omni.timeline = timeline_module  # type: ignore[attr-defined]
    return {"omni": omni, "omni.usd": usd, "omni.timeline": timeline_module}


def _saving_tools(**settings: Any) -> Tuple[IsaacTools, List[str], List[Any]]:
    """Tools whose client runs each script against the fake Kit in ``kit``."""
    captured: List[str] = []
    kit: List[Any] = []  # [monkeypatch, modules]

    async def _record(code: str, **_kwargs: Any) -> ScriptResult:
        captured.append(code)
        payload = await _run_kit_script_async(code, kit[0], kit[1])
        return ScriptResult(success=True, output=json.dumps(payload))

    client = MagicMock()
    client.address = "127.0.0.1:8226"
    client.timeout_seconds = 30.0
    client.bridge_enabled = False
    client.fallback_to_vscode = True
    client.execute = AsyncMock(side_effect=_record)
    return IsaacTools(client, settings=_settings(**settings)), captured, kit


def test_probe_lists_the_stage_file_and_every_dirty_layer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sub = Sdf.Layer.CreateNew(str(tmp_path / "sub.usda"))
    root = Sdf.Layer.CreateNew(str(tmp_path / "scene.usda"))
    root.subLayerPaths.append(sub.identifier)
    stage = Usd.Stage.Open(root)
    root.Save()
    stage.SetEditTarget(sub)
    stage.DefinePrim("/World/Edited")
    tools, captured, kit = _saving_tools(sandbox=True)
    context = _SaveContext(stage, root.identifier)
    kit[:] = [monkeypatch, _kit_modules(context, _Timeline(stopped=True))]

    result = asyncio.run(tools.save_isaac_stage())

    # Both layers sit outside the default sandbox: nothing is written.
    assert result["success"] is False and result["error_type"] == "SandboxError"
    assert result["details"]["layer"] == root.identifier
    assert len(captured) == 1 and context.saved == []
    probe = _run_kit_script(captured[0], monkeypatch, kit[1])
    assert probe["dirty_layers"] == [sub.identifier]
    assert probe["timeline_stopped"] is True


def test_in_place_save_runs_inside_the_sandbox_and_saves_once(monkeypatch: pytest.MonkeyPatch) -> None:
    tools, captured, kit = _saving_tools(sandbox=False)
    context = _SaveContext(Usd.Stage.CreateInMemory(), "/tmp/simul-work/scene.usd")
    kit[:] = [monkeypatch, _kit_modules(context, _Timeline(stopped=True))]

    result = asyncio.run(tools.save_isaac_stage())

    assert result["success"] is True and result["saved"] is True
    assert context.saved == ["/tmp/simul-work/scene.usd"]
    assert len(captured) == 2


@pytest.mark.parametrize("file_path", [None, "/tmp/simul-work/out.usd"])
def test_saving_mid_simulation_is_refused_unless_allowed(
    monkeypatch: pytest.MonkeyPatch, file_path: Optional[str]
) -> None:
    tools, _captured, kit = _saving_tools(sandbox=False)
    context = _SaveContext(Usd.Stage.CreateInMemory(), "/tmp/simul-work/scene.usd")
    kit[:] = [monkeypatch, _kit_modules(context, _Timeline(stopped=False))]

    refused = asyncio.run(tools.save_isaac_stage(file_path=file_path))
    allowed = asyncio.run(tools.save_isaac_stage(file_path=file_path, allow_while_playing=True))

    assert refused["success"] is False and refused["error_type"] == "RefusedOperation"
    assert "allow_while_playing" in refused["error"]
    assert allowed["success"] is True
    assert context.saved == [file_path or "/tmp/simul-work/scene.usd"]


def test_a_url_target_needs_overwrite_because_its_existence_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tools, captured, kit = _saving_tools(sandbox=True, write_schemes=("omniverse",))
    context = _SaveContext(Usd.Stage.CreateInMemory(), "/tmp/simul-work/scene.usd")
    kit[:] = [monkeypatch, _kit_modules(context, _Timeline(stopped=True))]
    url = "omniverse://nucleus/Projects/scene.usd"

    refused = asyncio.run(tools.save_isaac_stage(file_path=url))
    allowed = asyncio.run(tools.save_isaac_stage(file_path=url, overwrite=True))

    assert refused["success"] is False and refused["details"]["override"] == "overwrite"
    assert allowed["success"] is True and context.saved == [url]
    assert len(captured) == 1


def test_a_stage_swapped_between_probe_and_save_is_not_saved(monkeypatch: pytest.MonkeyPatch) -> None:
    tools, captured, kit = _saving_tools(sandbox=False)
    context = _SaveContext(Usd.Stage.CreateInMemory(), "/tmp/simul-work/scene.usd")
    kit[:] = [monkeypatch, _kit_modules(context, _Timeline(stopped=True))]

    original = tools._execute_json_script

    async def _swap_after_probe(script: str, transport_mode: str = "default", **kwargs: Any) -> Dict[str, Any]:
        result = await original(script, transport_mode, **kwargs)
        context.url = "/tmp/simul-work/other.usd"
        return result

    monkeypatch.setattr(tools, "_execute_json_script", _swap_after_probe)
    result = asyncio.run(tools.save_isaac_stage())

    assert result["success"] is False and "changed" in result["error"]
    assert context.saved == []
    assert len(captured) == 2


def test_a_never_saved_stage_asks_for_file_path(monkeypatch: pytest.MonkeyPatch) -> None:
    tools, captured, kit = _saving_tools(sandbox=True)
    context = _SaveContext(Usd.Stage.CreateInMemory(), "anon:0x1234:World0.usd")
    kit[:] = [monkeypatch, _kit_modules(context, _Timeline(stopped=True))]

    result = asyncio.run(tools.save_isaac_stage())

    assert result["success"] is False and "file_path" in result["error"]
    assert len(captured) == 1 and context.saved == []


# ---------------------------------------------------------------------------
# F12 and the script-node guard: graph tools
# ---------------------------------------------------------------------------


class _Node:
    def __init__(self, type_name: str) -> None:
        self.type_name = type_name
        self.values: Dict[str, Any] = {}

    def is_valid(self) -> bool:
        return True

    def get_type_name(self) -> str:
        return self.type_name


class _Controller:
    """``og.Controller``: knows a few nodes; ``inputs:bad`` refuses any value."""

    nodes: Dict[str, _Node] = {}
    connections: List[Tuple[str, str]] = []

    @classmethod
    def connect(cls, source: str, target: str) -> None:
        cls.connections.append((source, target))

    @classmethod
    def node(cls, path: str) -> _Node:
        if path not in cls.nodes:
            raise RuntimeError("no node " + path)
        return cls.nodes[path]

    @staticmethod
    def attribute(name: str, node_path: str) -> Tuple[str, str]:
        return node_path, name

    @classmethod
    def set(cls, attribute: Tuple[str, str], value: Any) -> None:
        node_path, name = attribute
        if name == "inputs:bad":
            raise ValueError("type mismatch for inputs:bad")
        cls.nodes[node_path].values[name] = value


def _og_modules() -> Dict[str, types.ModuleType]:
    _Controller.nodes = {
        "/World/Graph/Move": _Node("omni.graph.nodes.Add"),
        "/World/Graph/Script": _Node("omni.graph.scriptnode.ScriptNode"),
        "/World/Graph/Kernel": _Node("omni.warp.WarpKernel"),
        "/World/Graph/Augment": _Node("omni.replicator.core.OgnAugment"),
        "/World/Graph/Const": _Node("omni.graph.nodes.ConstantString"),
    }
    _Controller.connections = []
    # The nodes' prims, as Kit keeps them; Const has an asset-typed input.
    stage = Usd.Stage.CreateInMemory()
    for path in _Controller.nodes:
        stage.DefinePrim(path)
    stage.GetPrimAtPath("/World/Graph/Const").CreateAttribute("inputs:texture", Sdf.ValueTypeNames.Asset)
    omni = types.ModuleType("omni")
    graph = types.ModuleType("omni.graph")
    core = types.ModuleType("omni.graph.core")
    core.Controller = _Controller  # type: ignore[attr-defined]
    usd = types.ModuleType("omni.usd")
    usd.get_context = lambda: types.SimpleNamespace(get_stage=lambda: stage)  # type: ignore[attr-defined]
    omni.graph = graph  # type: ignore[attr-defined]
    omni.usd = usd  # type: ignore[attr-defined]
    graph.core = core  # type: ignore[attr-defined]
    return {"omni": omni, "omni.graph": graph, "omni.graph.core": core, "omni.usd": usd}


def _graph_call(monkeypatch: pytest.MonkeyPatch, scripts: bool, **kwargs: Any) -> Tuple[Dict[str, Any], List[str]]:
    tools, captured = _recording_tools(_settings(scripts=scripts))
    payload = asyncio.run(tools.set_isaac_graph_node_values(**kwargs))
    if not captured:
        return payload, captured
    return _run_kit_script(captured[-1], monkeypatch, _og_modules()), captured


def test_any_failed_attribute_fails_the_call_and_reports_the_rest(monkeypatch: pytest.MonkeyPatch) -> None:
    result, _ = _graph_call(
        monkeypatch, True, node_path="/World/Graph/Move", values={"inputs:a": 1.5, "inputs:bad": "x"}
    )

    assert result["success"] is False
    assert "1 of 2" in result["error"] and "inputs:bad" in result["error"]
    assert result["applied"] == {"inputs:a": 1.5}
    assert "type mismatch" in result["errors"]["inputs:bad"]


def test_all_attributes_set_is_a_success(monkeypatch: pytest.MonkeyPatch) -> None:
    result, _ = _graph_call(monkeypatch, True, node_path="/World/Graph/Move", values={"inputs:a": 2})
    missing, _ = _graph_call(monkeypatch, True, node_path="/World/Graph/Nope", values={"inputs:a": 2})

    assert result["success"] is True and result["count"] == 1
    assert missing["success"] is False and "Node not found" in missing["error"]
    assert missing["error_type"] == "NotFoundError"


def test_script_writes_are_refused_while_script_execution_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    by_name, sent = _graph_call(
        monkeypatch, False, node_path="/World/Graph/Script", values={"inputs:script": "import os"}
    )
    by_node, _ = _graph_call(
        monkeypatch, False, node_path="/World/Graph/Script", values={"inputs:usePath": True}
    )
    allowed, _ = _graph_call(
        monkeypatch, True, node_path="/World/Graph/Script", values={"inputs:script": "print(1)"}
    )

    assert by_name["error_type"] == "ScriptExecutionDisabled" and sent == []
    assert by_node["success"] is False and by_node["error_type"] == "ScriptExecutionDisabled"
    assert _Controller.nodes["/World/Graph/Script"].values == {"inputs:script": "print(1)"}
    assert allowed["success"] is True


@pytest.mark.parametrize(
    ("node_path", "values"),
    [
        ("/World/Graph/Kernel", {"inputs:codeStr": "import os"}),
        ("/World/Graph/Kernel", {"inputs:CodeFile": "/tmp/k.py"}),
        ("/World/Graph/Augment", {"inputs:augmentationScript": "x"}),
        ("/World/Graph/Const", {"inputs:conditionScript": "x"}),
    ],
)
def test_code_attributes_are_refused_before_sending(
    monkeypatch: pytest.MonkeyPatch, node_path: str, values: Dict[str, Any]
) -> None:
    result, sent = _graph_call(monkeypatch, False, node_path=node_path, values=values)

    assert result["error_type"] == "ScriptExecutionDisabled" and sent == []


@pytest.mark.parametrize("node_path", ["/World/Graph/Kernel", "/World/Graph/Augment"])
def test_any_write_to_a_code_node_without_script_in_its_name_is_refused(
    monkeypatch: pytest.MonkeyPatch, node_path: str
) -> None:
    result, sent = _graph_call(monkeypatch, False, node_path=node_path, values={"inputs:dim": 4})

    assert len(sent) == 1
    assert result["success"] is False and result["error_type"] == "ScriptExecutionDisabled"
    assert _Controller.nodes[node_path].values == {}


def _connect(
    monkeypatch: pytest.MonkeyPatch, scripts: bool, source: str, target: str
) -> Tuple[Dict[str, Any], List[str]]:
    tools, captured = _recording_tools(_settings(scripts=scripts))
    payload = asyncio.run(tools.connect_isaac_graph_nodes(source, target))
    if not captured:
        return payload, captured
    return _run_kit_script(captured[-1], monkeypatch, _og_modules()), captured


@pytest.mark.parametrize(
    "target",
    ["/World/Graph/Script.inputs:script", "/World/Graph/Kernel.inputs:codeStr", "/World/Graph/Kernel.inputs:dim"],
)
def test_connections_into_code_are_refused_while_script_execution_is_off(
    monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    _og_modules()  # resets the recorded connections
    refused, _ = _connect(monkeypatch, False, "/World/Graph/Const.inputs:value", target)
    assert refused["success"] is False and refused["error_type"] == "ScriptExecutionDisabled"
    assert _Controller.connections == []

    allowed, _ = _connect(monkeypatch, True, "/World/Graph/Const.inputs:value", target)
    assert allowed["success"] is True and _Controller.connections == [("/World/Graph/Const.inputs:value", target)]


def test_an_ordinary_connection_still_goes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    result, _ = _connect(monkeypatch, False, "/World/Graph/Const.inputs:value", "/World/Graph/Move.inputs:a")

    assert result["success"] is True and result["connected"] is True


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("inputs:script", True), ("inputs:scriptPath", True), ("inputs:codeStr", True),
        ("inputs:CODEFILE", True), ("inputs:augmentationScript", True),
        ("inputs:sourceCode", True), ("inputs:conditionScript", True), ("inputs:usdTimecode", False),
        ("/World/G/script.inputs:value", False), ("inputs:value", False), ("outputs:sum", False),
    ],
)
def test_code_attribute_predicate(name: str, expected: bool) -> None:
    assert is_script_attribute(name) is expected


@pytest.mark.parametrize(
    ("node_type", "expected"),
    [
        ("omni.graph.scriptnode.ScriptNode", True), ("omni.warp.WarpKernel", True),
        ("OMNI.WARP.WARPKERNEL", True), ("omni.replicator.core.OgnAugmentCPU", True),
        ("omni.replicator.core.OgnConditionScript", True), ("some.ext.WarpKernel", True),
        ("my.ext.WarpSolver", False), ("omni.warp.WarpClothSimulate", False),
        ("omni.graph.nodes.ReadPrimAttribute", False),
        ("omni.graph.nodes.Add", False), ("omni.graph.action.OnPlaybackTick", False),
    ],
)
def test_script_capable_node_type_predicate(node_type: str, expected: bool) -> None:
    assert is_script_capable_node_type(node_type) is expected


def test_the_shared_predicates_keep_the_signatures_other_tools_call() -> None:
    """The prim tools import these by name and keyword; the names are the contract."""
    assert is_script_capable_node_type(type_name="omni.warp.WarpKernel") is True
    assert is_script_attribute(name="inputs:codeStr") is True


def test_the_kit_side_guard_finds_code_nodes_in_a_subtree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World/Graph", "OmniGraph")
    for name, node_type in (("add", "omni.graph.nodes.Add"), ("kernel", "omni.warp.WarpKernel")):
        node = stage.DefinePrim("/World/Graph/" + name, "OmniGraphNode")
        node.CreateAttribute("node:type", Sdf.ValueTypeNames.Token).Set(node_type)
    hidden = stage.DefinePrim("/World/Graph/off", "OmniGraphNode")
    hidden.CreateAttribute("node:type", Sdf.ValueTypeNames.Token).Set("omni.graph.scriptnode.ScriptNode")
    hidden.SetActive(False)
    namespace: Dict[str, Any] = {}
    exec(compile(script_guard_source(), "<guard>", "exec"), namespace)  # noqa: S102 - generated guard source

    found = namespace["_script_capable_prims_under"](stage.GetPrimAtPath("/World"))

    assert found == ["/World/Graph/kernel", "/World/Graph/off"]
    assert namespace["_script_capable_prim"](stage.GetPrimAtPath("/World/Graph/add")) is False


def _code_node_asset(path: Path, node_type: str) -> str:
    layer_stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.Xform.Define(layer_stage, "/Asset")
    layer_stage.DefinePrim("/Asset/Graph", "OmniGraph")
    node = layer_stage.DefinePrim("/Asset/Graph/node", "OmniGraphNode")
    node.CreateAttribute("node:type", Sdf.ValueTypeNames.Token).Set(node_type)
    node.CreateAttribute("inputs:script", Sdf.ValueTypeNames.String).Set("import os")
    layer_stage.SetDefaultPrim(layer_stage.GetPrimAtPath("/Asset"))
    layer_stage.Save()
    return str(path)


@pytest.mark.parametrize("node_type", ["omni.graph.scriptnode.ScriptNode", "omni.warp.WarpKernel"])
def test_assets_carrying_code_nodes_are_not_imported_while_script_execution_is_off(
    run_on_stage: Any, tmp_path: Path, node_type: str
) -> None:
    asset = _code_node_asset(tmp_path / "graph.usda", node_type)
    clean = _asset(tmp_path / "clean.usda")
    stage = _world_stage()
    holder = stage.DefinePrim("/World/Holder")
    holder.GetReferences().AddReference(clean)
    guarded = _recording_tools(_settings(sandbox=False, scripts=False))

    imported = _call(guarded, run_on_stage, stage, "import_isaac_asset", asset_path=asset, target_path="/World/A/B")
    referenced = _call(guarded, run_on_stage, stage, "add_isaac_reference", prim_path="/World/Holder",
                       reference_path=asset)

    for result in (imported, referenced):
        assert result["success"] is False and result["error_type"] == "ScriptExecutionDisabled"
        assert result["script_capable_prims"]
    assert not stage.GetPrimAtPath("/World/A").IsValid()
    assert list(holder.GetMetadata("references").GetAppliedItems()) == [Sdf.Reference(clean)]

    allowed = _call(
        _recording_tools(_settings(sandbox=False, scripts=True)), run_on_stage, stage, "import_isaac_asset",
        asset_path=asset,
    )
    assert allowed["success"] is True


@pytest.mark.parametrize("node_type", ["omni.graph.scriptnode.ScriptNode", "my.ext.PythonScript",
                                       "omni.warp.WarpKernel", "omni.replicator.core.OgnAugment"])
def test_script_nodes_are_not_created_while_script_execution_is_off(node_type: str) -> None:
    off, off_sent = _recording_tools(_settings(scripts=False))
    on, on_sent = _recording_tools(_settings(scripts=True))

    refused = asyncio.run(off.create_isaac_graph_node("/World/Graph", "/World/Graph/S", node_type))
    asyncio.run(on.create_isaac_graph_node("/World/Graph", "/World/Graph/S", node_type))

    assert refused["success"] is False and refused["error_type"] == "ScriptExecutionDisabled"
    assert off_sent == [] and len(on_sent) == 1
    assert is_script_capable_node_type(node_type)
    assert not is_script_capable_node_type("omni.graph.nodes.OnPlaybackTick")


# ---------------------------------------------------------------------------
# Extensions: script extensions refused, disable read back (F34)
# ---------------------------------------------------------------------------


class _ExtensionManager:
    def __init__(self, sticky: bool) -> None:
        self.sticky = sticky
        self.enabled = True

    def set_extension_enabled_immediate(self, ext_id: str, enabled: bool) -> None:
        if not self.sticky:
            self.enabled = enabled

    def get_extensions(self) -> List[Dict[str, Any]]:
        return [{"id": "omni.physx.foo-1.0.0", "name": "omni.physx.foo", "enabled": self.enabled, "version": "1.0.0"}]


def _kit_app(manager: _ExtensionManager) -> Dict[str, types.ModuleType]:
    omni = types.ModuleType("omni")
    kit = types.ModuleType("omni.kit")
    app = types.ModuleType("omni.kit.app")
    app.get_app = lambda: types.SimpleNamespace(get_extension_manager=lambda: manager)  # type: ignore[attr-defined]
    omni.kit = kit  # type: ignore[attr-defined]
    kit.app = app  # type: ignore[attr-defined]
    return {"omni": omni, "omni.kit": kit, "omni.kit.app": app}


@pytest.mark.parametrize("sticky", [True, False])
def test_disable_reports_an_extension_that_stayed_enabled(monkeypatch: pytest.MonkeyPatch, sticky: bool) -> None:
    tools, captured = _recording_tools(_settings())
    asyncio.run(tools.disable_isaac_extension("omni.physx.foo"))

    result = _run_kit_script(captured[-1], monkeypatch, _kit_app(_ExtensionManager(sticky)))

    assert result["success"] is (not sticky)
    assert result["enabled"] is sticky
    if sticky:
        assert "still enabled" in result["error"]


@pytest.mark.parametrize(
    "extension_id",
    [
        "omni.graph.scriptnode", "omni.kit.scripting-1.2.0", "omni.warp", "omni.warp-1.7.1-rc1",
        "omni.graph.scriptnode-1.50.0+107.3.0", "omni.behavior.scripting.bundle",
    ],
)
def test_script_extensions_are_not_enabled_while_script_execution_is_off(extension_id: str) -> None:
    off, off_sent = _recording_tools(_settings(scripts=False))
    on, on_sent = _recording_tools(_settings(scripts=True))

    refused = asyncio.run(off.enable_isaac_extension(extension_id))
    asyncio.run(on.enable_isaac_extension(extension_id))

    assert refused["error_type"] == "ScriptExecutionDisabled" and off_sent == []
    assert len(on_sent) == 1


@pytest.mark.parametrize(
    ("extension_id", "bare"),
    [("omni.warp-1.7.1-rc1", "omni.warp"), ("khemoo.simul-0.1.0-rc1", "khemoo.simul"), ("omni.physx", "omni.physx")],
)
def test_extension_ids_end_their_name_at_the_first_hyphen(extension_id: str, bare: str) -> None:
    assert extension_bare_name(extension_id) == bare


def test_a_prerelease_transport_extension_id_is_still_protected() -> None:
    tools, captured = _recording_tools(_settings())
    result = asyncio.run(tools.disable_isaac_extension("khemoo.simul-0.1.0-rc1"))
    assert result["error_type"] == "RefusedOperation" and captured == []
