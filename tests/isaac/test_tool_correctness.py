"""Correctness of generated Isaac tool scripts, run against real ``pxr``.

Each test here pins a defect where a tool reported success for work it did not
do, or did the wrong work: transforms composed in the wrong op order, a
fallback serializer that crashed on asset paths, listings that treated a
missing root as an empty one, and an extension toggle that reported a no-op as
success. The scripts run under ``exec`` with ``omni.usd`` stubbed to an
in-memory stage (see ``conftest.run_on_stage``), so USD-level behaviour is
observed directly.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import io
import json
import sys
import types
from typing import Any, Dict, List, Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from simul_mcp.mcp.tools.isaac_tools import IsaacTools


def _stage() -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    return stage


def _world(stage: Usd.Stage, path: str) -> Gf.Matrix4d:
    return UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )


def _run_transform(
    capturing_tools: Tuple[IsaacTools, List[str]],
    run_on_stage: Any,
    stage: Usd.Stage,
    **kwargs: Any,
) -> Dict[str, Any]:
    tools, captured = capturing_tools
    asyncio.run(tools.set_isaac_prim_transform(**kwargs))
    return run_on_stage(captured[-1], stage)


def _expected(translation: List[float], euler: List[float], scale: List[float]) -> Gf.Matrix4d:
    reference = _stage()
    xf = UsdGeom.Xformable(UsdGeom.Xform.Define(reference, "/World/Ref").GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*translation))
    xf.AddRotateXYZOp().Set(Gf.Vec3f(*euler))
    xf.AddScaleOp().Set(Gf.Vec3f(*scale))
    return _world(reference, "/World/Ref")


# ---------------------------------------------------------------------------
# set_isaac_prim_transform: canonical op order, existing orient reused
# ---------------------------------------------------------------------------


def test_translate_after_scale_is_not_scaled(
    capturing_tools: Tuple[IsaacTools, List[str]], run_on_stage: Any
) -> None:
    """Scale first, translate second: the translation must stay in parent units."""
    stage = _stage()
    UsdGeom.Xform.Define(stage, "/World/A")

    _run_transform(capturing_tools, run_on_stage, stage, prim_path="/World/A", scale=[2, 2, 2])
    result = _run_transform(
        capturing_tools, run_on_stage, stage, prim_path="/World/A", translation=[1, 0, 0]
    )

    assert result["translation"] == [1.0, 0.0, 0.0]
    assert result["xform_op_order"] == ["xformOp:translate", "xformOp:scale"]
    assert _world(stage, "/World/A").ExtractTranslation() == Gf.Vec3d(1, 0, 0)


def test_rotation_added_between_translate_and_scale(
    capturing_tools: Tuple[IsaacTools, List[str]], run_on_stage: Any
) -> None:
    stage = _stage()
    UsdGeom.Xform.Define(stage, "/World/A")

    _run_transform(capturing_tools, run_on_stage, stage, prim_path="/World/A", scale=[1, 2, 3])
    _run_transform(
        capturing_tools, run_on_stage, stage, prim_path="/World/A", translation=[4, 5, 6]
    )
    result = _run_transform(
        capturing_tools, run_on_stage, stage, prim_path="/World/A", rotation_euler=[10, 20, 30]
    )

    assert result["xform_op_order"] == [
        "xformOp:translate",
        "xformOp:rotateXYZ",
        "xformOp:scale",
    ]
    assert Gf.IsClose(
        _world(stage, "/World/A"), _expected([4, 5, 6], [10, 20, 30], [1, 2, 3]), 1e-5
    )


def test_existing_orient_is_written_not_doubled(
    capturing_tools: Tuple[IsaacTools, List[str]], run_on_stage: Any
) -> None:
    """Isaac assets carry translate/orient/scale; a rotation must land in orient."""
    stage = _stage()
    xf = UsdGeom.Xformable(UsdGeom.Xform.Define(stage, "/World/Robot").GetPrim())
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1, 2, 3))
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))
    xf.AddScaleOp().Set(Gf.Vec3f(2, 2, 2))

    result = _run_transform(
        capturing_tools,
        run_on_stage,
        stage,
        prim_path="/World/Robot",
        rotation_euler=[30, 45, 60],
    )

    assert result["xform_op_order"] == [
        "xformOp:translate",
        "xformOp:orient",
        "xformOp:scale",
    ]
    assert not stage.GetPrimAtPath("/World/Robot").HasAttribute("xformOp:rotateXYZ")
    assert Gf.IsClose(
        _world(stage, "/World/Robot"), _expected([1, 2, 3], [30, 45, 60], [2, 2, 2]), 1e-5
    )


def test_float_orient_precision_is_respected(
    capturing_tools: Tuple[IsaacTools, List[str]], run_on_stage: Any
) -> None:
    stage = _stage()
    xf = UsdGeom.Xformable(UsdGeom.Xform.Define(stage, "/World/B").GetPrim())
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(1, 0, 0, 0))

    _run_transform(
        capturing_tools, run_on_stage, stage, prim_path="/World/B", rotation_euler=[0, 0, 90]
    )

    value = stage.GetPrimAtPath("/World/B").GetAttribute("xformOp:orient").Get()
    assert isinstance(value, Gf.Quatf)
    assert Gf.IsClose(_world(stage, "/World/B"), _expected([0, 0, 0], [0, 0, 90], [1, 1, 1]), 1e-5)


def test_other_rotation_order_is_replaced_in_place(
    capturing_tools: Tuple[IsaacTools, List[str]], run_on_stage: Any
) -> None:
    stage = _stage()
    xf = UsdGeom.Xformable(UsdGeom.Xform.Define(stage, "/World/C").GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, 1))
    xf.AddRotateZYXOp().Set(Gf.Vec3f(5, 5, 5))
    xf.AddScaleOp().Set(Gf.Vec3f(3, 3, 3))

    result = _run_transform(
        capturing_tools, run_on_stage, stage, prim_path="/World/C", rotation_euler=[0, 90, 0]
    )

    assert result["xform_op_order"] == [
        "xformOp:translate",
        "xformOp:rotateXYZ",
        "xformOp:scale",
    ]
    assert Gf.IsClose(_world(stage, "/World/C"), _expected([0, 0, 1], [0, 90, 0], [3, 3, 3]), 1e-5)


def test_pivot_ops_are_left_alone(
    capturing_tools: Tuple[IsaacTools, List[str]], run_on_stage: Any
) -> None:
    stage = _stage()
    xf = UsdGeom.Xformable(UsdGeom.Xform.Define(stage, "/World/P").GetPrim())
    pivot = xf.AddTranslateOp(opSuffix="pivot")
    pivot.Set(Gf.Vec3d(1, 1, 1))
    xf.AddScaleOp().Set(Gf.Vec3f(2, 2, 2))
    xf.AddTranslateOp(opSuffix="pivot", isInverseOp=True)

    result = _run_transform(
        capturing_tools, run_on_stage, stage, prim_path="/World/P", translation=[5, 0, 0]
    )

    assert result["xform_op_order"] == [
        "xformOp:translate",
        "xformOp:translate:pivot",
        "xformOp:scale",
        "!invert!xformOp:translate:pivot",
    ]
    assert pivot.Get() == Gf.Vec3d(1, 1, 1)


def test_all_three_on_a_bare_prim(
    capturing_tools: Tuple[IsaacTools, List[str]], run_on_stage: Any
) -> None:
    stage = _stage()
    UsdGeom.Cube.Define(stage, "/World/Box")

    result = _run_transform(
        capturing_tools,
        run_on_stage,
        stage,
        prim_path="/World/Box",
        translation=[1, 2, 3],
        rotation_euler=[0, 0, 90],
        scale=[2, 2, 2],
    )

    assert result["xform_op_order"] == [
        "xformOp:translate",
        "xformOp:rotateXYZ",
        "xformOp:scale",
    ]
    assert Gf.IsClose(_world(stage, "/World/Box"), _expected([1, 2, 3], [0, 0, 90], [2, 2, 2]), 1e-5)


# ---------------------------------------------------------------------------
# step_isaac_simulation: exactly N frames, timeline left paused
# ---------------------------------------------------------------------------


class FakeTimeline:
    """A timeline that moves only while playing, after a warm-up from stop.

    Mirrors the Kit behaviour the step tool must cope with: app updates do not
    advance a paused timeline, and the first updates after ``play`` from a
    stopped timeline initialise physics without moving time.
    """

    DT = 1.0 / 60.0
    WARMUP_UPDATES = 3

    def __init__(self, state: str, time: float = 0.0, end_time: float = 1e9) -> None:
        self.state = state
        self.time = time
        self.end_time = end_time
        self.warmup = 0

    def get_current_time(self) -> float:
        return self.time

    def is_playing(self) -> bool:
        return self.state == "playing"

    def is_stopped(self) -> bool:
        return self.state == "stopped"

    def play(self) -> None:
        if self.state == "stopped":
            self.warmup = self.WARMUP_UPDATES
        self.state = "playing"

    def pause(self) -> None:
        self.state = "paused"

    def commit(self) -> None:
        pass

    def update(self) -> None:
        if self.state != "playing":
            return
        if self.warmup:
            self.warmup -= 1
            return
        self.time = min(self.end_time, self.time + self.DT)


class FakeApp:
    def __init__(self, timeline: FakeTimeline) -> None:
        self.timeline = timeline

    async def next_update_async(self) -> None:
        self.timeline.update()


def _install_kit(monkeypatch: pytest.MonkeyPatch, timeline: FakeTimeline) -> None:
    app = FakeApp(timeline)
    omni = types.ModuleType("omni")
    omni_timeline = types.ModuleType("omni.timeline")
    omni_timeline.get_timeline_interface = lambda: timeline  # type: ignore[attr-defined]
    omni_kit = types.ModuleType("omni.kit")
    omni_kit_app = types.ModuleType("omni.kit.app")
    omni_kit_app.get_app = lambda: app  # type: ignore[attr-defined]
    omni.timeline = omni_timeline  # type: ignore[attr-defined]
    omni.kit = omni_kit  # type: ignore[attr-defined]
    omni_kit.app = omni_kit_app  # type: ignore[attr-defined]
    for name, module in (
        ("omni", omni),
        ("omni.timeline", omni_timeline),
        ("omni.kit", omni_kit),
        ("omni.kit.app", omni_kit_app),
    ):
        monkeypatch.setitem(sys.modules, name, module)


def _run_async_script(script: str) -> Dict[str, Any]:
    # The script is one IsaacTools just generated, run the way Kit runs it
    # (top-level await allowed); nothing external reaches this eval.
    code = compile(script, "<isaac-script>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        result = eval(code, {"__name__": "__isaac_script__"})
        if result is not None:
            asyncio.run(result)
    return json.loads(stdout.getvalue())


@pytest.mark.parametrize("initial", ["stopped", "paused", "playing"])
def test_step_script_advances_exactly_n_and_pauses(
    capturing_tools: Tuple[IsaacTools, List[str]],
    monkeypatch: pytest.MonkeyPatch,
    initial: str,
) -> None:
    tools, captured = capturing_tools
    asyncio.run(tools.step_isaac_simulation(num_steps=5))
    timeline = FakeTimeline(initial, time=1.0)
    _install_kit(monkeypatch, timeline)

    result = _run_async_script(captured[-1])

    assert result["steps"] == 5
    assert result["steps_requested"] == 5
    assert result["state"] == "paused"
    assert "error" not in result
    assert timeline.state == "paused"
    assert result["start_time"] == 1.0
    assert result["time_delta"] == pytest.approx(5 * FakeTimeline.DT)
    assert timeline.time == pytest.approx(1.0 + 5 * FakeTimeline.DT)


def test_step_script_reports_a_timeline_that_cannot_advance(
    capturing_tools: Tuple[IsaacTools, List[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tools, captured = capturing_tools
    asyncio.run(tools.step_isaac_simulation(num_steps=5))
    timeline = FakeTimeline("paused", time=2.0, end_time=2.0 + 2 * FakeTimeline.DT)
    _install_kit(monkeypatch, timeline)

    result = _run_async_script(captured[-1])

    assert result["steps"] == 2
    assert "advanced 2 of 5" in result["error"]
    assert timeline.state == "paused"


@pytest.mark.parametrize("initial", ["stopped", "paused", "playing"])
def test_bridge_step_advances_exactly_n_and_pauses(
    monkeypatch: pytest.MonkeyPatch, initial: str
) -> None:
    from khemoo.simul.mcp.protocol import BridgeRequest
    from khemoo.simul.mcp.service import BridgeCommandService

    timeline = FakeTimeline(initial)
    _install_kit(monkeypatch, timeline)
    service = BridgeCommandService(executor=MagicMock(), allow_unsafe_execution=False)

    response = asyncio.run(
        service.dispatch(
            BridgeRequest(
                request_id="s",
                action="simulation_control",
                payload={"command": "step", "num_steps": 4},
            )
        )
    )

    assert response.status == "ok"
    assert response.payload["steps"] == 4
    assert response.payload["state"] == "paused"
    assert response.payload["time_delta"] == pytest.approx(4 * FakeTimeline.DT)
    assert "error" not in response.payload
    assert timeline.state == "paused"


# ---------------------------------------------------------------------------
# get_isaac_prim_info: asset-valued attributes serialize
# ---------------------------------------------------------------------------


def test_prim_info_serializes_asset_paths(
    capturing_tools: Tuple[IsaacTools, List[str]], run_on_stage: Any
) -> None:
    tools, captured = capturing_tools
    stage = _stage()
    prim = stage.DefinePrim("/World/Dome", "DomeLight")
    prim.CreateAttribute("inputs:texture:file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("/maps/sky.hdr")
    )
    prim.CreateAttribute("rel_target", Sdf.ValueTypeNames.String).Set("plain")
    prim.CreateAttribute("frames", Sdf.ValueTypeNames.AssetArray).Set(
        Sdf.AssetPathArray([Sdf.AssetPath("a.png"), Sdf.AssetPath("b.png")])
    )

    asyncio.run(tools.get_isaac_prim_info("/World/Dome"))
    result = run_on_stage(captured[-1], stage)

    assert "error" not in result
    assert result["attributes"]["inputs:texture:file"] == "/maps/sky.hdr"
    assert result["attributes"]["frames"] == ["a.png", "b.png"]
    assert result["attributes"]["rel_target"] == "plain"


# ---------------------------------------------------------------------------
# Listings under a root: a missing root is an error, not an empty result
# ---------------------------------------------------------------------------


def test_query_typed_prims_rejects_a_missing_root(
    capturing_tools: Tuple[IsaacTools, List[str]], run_on_stage: Any
) -> None:
    tools, captured = capturing_tools
    stage = _stage()
    UsdGeom.Cube.Define(stage, "/World/Box")

    asyncio.run(tools.query_usd_typed_prims("UsdGeom.Cube", root_path="/Wrold"))
    missing = run_on_stage(captured[-1], stage)
    asyncio.run(tools.query_usd_typed_prims("UsdGeom.Cube", root_path="/World"))
    found = run_on_stage(captured[-1], stage)

    assert missing == {"error": "Root path not found: /Wrold"}
    assert [p["path"] for p in found["prims"]] == ["/World/Box"]


def test_list_physics_objects_rejects_a_missing_root(
    capturing_tools: Tuple[IsaacTools, List[str]], run_on_stage: Any
) -> None:
    tools, captured = capturing_tools
    stage = _stage()
    box = UsdGeom.Cube.Define(stage, "/World/Box").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(box)

    asyncio.run(tools.list_isaac_physics_objects(root_path="/Wrold"))
    missing = run_on_stage(captured[-1], stage)
    asyncio.run(tools.list_isaac_physics_objects(root_path="/"))
    found = run_on_stage(captured[-1], stage)

    assert missing == {"error": "Root path not found: /Wrold"}
    assert found["rigid_bodies"] == [{"path": "/World/Box", "type": "Cube"}]


# ---------------------------------------------------------------------------
# import_isaac_asset / add_isaac_reference: a missing asset fails
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["import_isaac_asset", "add_isaac_reference"])
def test_referencing_a_missing_asset_fails(run_on_stage: Any, tmp_path: Any, method: str) -> None:
    tools, captured = _sandboxed_tools(tmp_path)
    stage = _stage()
    missing = str(tmp_path / "nope.usd")
    kwargs = (
        {"asset_path": missing, "target_path": "/World/Asset"}
        if method == "import_isaac_asset"
        else {"prim_path": "/World/Asset", "reference_path": missing}
    )

    asyncio.run(getattr(tools, method)(**kwargs))
    result = run_on_stage(captured[-1], stage)

    assert result["error"].startswith("Asset not found: ")
    assert not stage.GetPrimAtPath("/World/Asset").IsValid(), "a prim was created for nothing"


@pytest.mark.parametrize("method", ["import_isaac_asset", "add_isaac_reference"])
def test_referencing_an_existing_asset_composes(run_on_stage: Any, tmp_path: Any, method: str) -> None:
    tools, captured = _sandboxed_tools(tmp_path)
    asset = Usd.Stage.CreateNew(str(tmp_path / "chair.usda"))
    UsdGeom.Cube.Define(asset, "/Chair")
    asset.SetDefaultPrim(asset.GetPrimAtPath("/Chair"))
    asset.GetRootLayer().Save()
    stage = _stage()
    path = str(tmp_path / "chair.usda")
    kwargs = (
        {"asset_path": path, "target_path": "/World/Asset"}
        if method == "import_isaac_asset"
        else {"prim_path": "/World/Asset", "reference_path": path}
    )

    asyncio.run(getattr(tools, method)(**kwargs))
    result = run_on_stage(captured[-1], stage)

    assert "error" not in result
    assert stage.GetPrimAtPath("/World/Asset").GetTypeName() == "Cube"


# ---------------------------------------------------------------------------
# create_isaac_light: texture_file is sandboxed like every other file param
# ---------------------------------------------------------------------------


def _sandboxed_tools(sandbox: Any) -> Tuple[IsaacTools, List[str]]:
    from simul_mcp.adapters.isaac_socket_client import ScriptResult
    from simul_mcp.config import Settings

    captured: List[str] = []

    def _record(code: str) -> ScriptResult:
        captured.append(code)
        return ScriptResult(success=True, output=json.dumps({"ok": True}))

    settings = Settings()
    security = settings.security.model_copy(update={"allowed_paths": [str(sandbox)]})
    client = MagicMock()
    client.bridge_enabled = False
    client.fallback_to_vscode = True
    client.execute = AsyncMock(side_effect=_record)
    client.bridge_request = AsyncMock(return_value=None)
    tools = IsaacTools(client, settings=settings.model_copy(update={"security": security}))
    return tools, captured


def test_light_texture_outside_the_sandbox_is_refused(tmp_path: Any) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    tools, captured = _sandboxed_tools(sandbox)

    result = asyncio.run(
        tools.create_isaac_light("/World/Dome", texture_file="/etc/shadow")
    )

    assert result["error_type"] == "SandboxError"
    assert captured == [], "the script ran with a path outside the sandbox"


def test_light_texture_inside_the_sandbox_is_resolved(tmp_path: Any) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    tools, captured = _sandboxed_tools(sandbox)

    asyncio.run(
        tools.create_isaac_light("/World/Dome", texture_file=str(sandbox / "sky.hdr"))
    )

    assert repr(str((sandbox / "sky.hdr").resolve())) in captured[-1]


def test_light_without_texture_is_unaffected(tmp_path: Any) -> None:
    tools, captured = _sandboxed_tools(tmp_path)

    asyncio.run(tools.create_isaac_light("/World/Sun", light_type="DistantLight"))

    assert len(captured) == 1


# ---------------------------------------------------------------------------
# read_isaac_aovs: nothing read is a failure
# ---------------------------------------------------------------------------


class _FakeAnnotator:
    def __init__(self, data: Any) -> None:
        self.data = data

    def attach(self, _rps: Any) -> None:
        pass

    def detach(self, _rps: Any) -> None:
        pass

    def get_data(self) -> Any:
        return self.data


def _install_replicator(monkeypatch: pytest.MonkeyPatch, annotators: Dict[str, Any]) -> None:
    def get_annotator(name: str) -> _FakeAnnotator:
        if name not in annotators:
            raise KeyError(f"Annotator {name} not registered")
        return _FakeAnnotator(annotators[name])

    async def step_async(**_kwargs: Any) -> None:
        return None

    rp = types.SimpleNamespace(path="/Render/RP", destroy=lambda: None)
    rep = types.ModuleType("omni.replicator.core")
    rep.create = types.SimpleNamespace(render_product=lambda *_a: rp)  # type: ignore[attr-defined]
    rep.AnnotatorRegistry = types.SimpleNamespace(get_annotator=get_annotator)  # type: ignore[attr-defined]
    rep.orchestrator = types.SimpleNamespace(step_async=step_async)  # type: ignore[attr-defined]
    replicator = types.ModuleType("omni.replicator")
    replicator.core = rep  # type: ignore[attr-defined]
    kit_app = types.ModuleType("omni.kit.app")
    kit_app.get_app = lambda: types.SimpleNamespace(update=lambda: None)  # type: ignore[attr-defined]
    kit = types.ModuleType("omni.kit")
    kit.app = kit_app  # type: ignore[attr-defined]
    omni = types.ModuleType("omni")
    omni.replicator = replicator  # type: ignore[attr-defined]
    omni.kit = kit  # type: ignore[attr-defined]
    for name, module in (
        ("omni", omni),
        ("omni.replicator", replicator),
        ("omni.replicator.core", rep),
        ("omni.kit", kit),
        ("omni.kit.app", kit_app),
    ):
        monkeypatch.setitem(sys.modules, name, module)


def _read_aovs(
    capturing_tools: Tuple[IsaacTools, List[str]],
    monkeypatch: pytest.MonkeyPatch,
    names: List[str],
    annotators: Dict[str, Any],
) -> Dict[str, Any]:
    from simul_mcp.mcp.registration._helpers import apply_success_from_error

    tools, captured = capturing_tools
    asyncio.run(tools.read_aovs(aov_names=names, num_frames=1))
    _install_replicator(monkeypatch, annotators)
    return apply_success_from_error(_run_async_script(captured[-1]))


def test_read_aovs_with_no_data_is_a_failure(
    capturing_tools: Tuple[IsaacTools, List[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    np = pytest.importorskip("numpy")

    result = _read_aovs(
        capturing_tools, monkeypatch, ["Bogus", "HdrColor"], {"HdrColor": np.zeros(0)}
    )

    assert result["success"] is False
    assert "Bogus" in result["error"] and "HdrColor" in result["error"]
    assert result["failed_aovs"] == ["Bogus", "HdrColor"]


def test_read_aovs_partial_data_succeeds_and_names_failures(
    capturing_tools: Tuple[IsaacTools, List[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    np = pytest.importorskip("numpy")

    result = _read_aovs(
        capturing_tools,
        monkeypatch,
        ["Bogus", "HdrColor"],
        {"HdrColor": np.ones((2, 2, 4), dtype=np.float32)},
    )

    assert result["success"] is True
    assert result["failed_aovs"] == ["Bogus"]
    assert result["aovs"]["HdrColor"]["nonzero_pixels"] == 4


# ---------------------------------------------------------------------------
# enable_isaac_extension: a refused enable is a failure
# ---------------------------------------------------------------------------


def _run_enable(
    capturing_tools: Tuple[IsaacTools, List[str]],
    monkeypatch: pytest.MonkeyPatch,
    enabled_after: bool,
) -> Dict[str, Any]:
    from simul_mcp.mcp.registration._helpers import apply_success_from_error

    tools, captured = capturing_tools
    asyncio.run(tools.enable_isaac_extension("worv.env.sun"))
    manager = types.SimpleNamespace(
        set_extension_enabled_immediate=lambda *_a: None,
        get_extensions=lambda: [
            {"id": "worv.env.sun-0.3.0", "name": "worv.env.sun", "enabled": enabled_after}
        ],
    )
    kit_app = types.ModuleType("omni.kit.app")
    kit_app.get_app = lambda: types.SimpleNamespace(  # type: ignore[attr-defined]
        get_extension_manager=lambda: manager
    )
    kit = types.ModuleType("omni.kit")
    kit.app = kit_app  # type: ignore[attr-defined]
    omni = types.ModuleType("omni")
    omni.kit = kit  # type: ignore[attr-defined]
    for name, module in (("omni", omni), ("omni.kit", kit), ("omni.kit.app", kit_app)):
        monkeypatch.setitem(sys.modules, name, module)
    return apply_success_from_error(_run_async_script(captured[-1]))


def test_enable_extension_that_stays_disabled_fails(
    capturing_tools: Tuple[IsaacTools, List[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _run_enable(capturing_tools, monkeypatch, enabled_after=False)

    assert result["success"] is False
    assert result["enabled"] is False
    assert "worv.env.sun-0.3.0" in result["error"]


def test_enable_extension_that_enables_succeeds(
    capturing_tools: Tuple[IsaacTools, List[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _run_enable(capturing_tools, monkeypatch, enabled_after=True)

    assert result["success"] is True
    assert result["enabled"] is True
