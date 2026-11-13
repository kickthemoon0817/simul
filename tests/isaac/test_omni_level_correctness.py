"""Omni-level tools report what Isaac Sim actually does, checked against fakes.

No Isaac Sim runs on the hosts this suite targets, so each test runs the
script a tool generates, in process, against fake ``omni`` / ``carb`` modules
that behave the way Isaac Sim's own source shows:

* GPU physics (SimulationManager.enable_fabric) sets /physics/updateToUsd and
  /physics/updateVelocitiesToUsd to false, so USD poses go stale (F08).
* semantic_segmentation, the bounding boxes and pointcloud return
  {"data": ..., "info": ...} (F10).
* Edit > Focus Prim calls omni.kit.viewport.utility.frame_viewport_selection()
  (F24).
* PhysX scene queries want a unit direction and see colliders only once the
  simulation has started; results carry "collision" and "rigidBody" (F19).
* A frame runs frame_dt / physics_dt physics steps, and the app is
  rate-limited (F22); resetOnStop reverts edits made while paused (F39).
* The runtime-info keys, render-var listing, camera resolution, log path,
  carb-settings guards and the 6.0 API note (F25, F26, F27, F31, F32, F35, F37).
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import io
import sys
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from simul.adapters.isaac_socket_client import ScriptResult
from simul.config import Settings
from simul.mcp.tools.isaac._base import merge_annotations, split_annotations
from simul.mcp.tools.isaac._shared import (
    ANNOTATION_MARKER,
    PAUSED_EDIT_WARNING,
    PLAYING_EDIT_WARNING,
    PROTECTED_CARB_SETTING_PREFIXES,
    USD_READBACK_STALE_WARNING,
    USD_VELOCITY_STALE_WARNING,
)
from simul.mcp.tools.isaac_tools import IsaacTools

# ---------------------------------------------------------------------------
# Harness: a client that runs every script against fake Kit modules
# ---------------------------------------------------------------------------


class _Timeline:
    def __init__(
        self, state: str = "stopped", time: float = 0.0, dt: float = 1.0 / 60.0
    ) -> None:
        self.state = state
        self.time = time
        self.dt = dt

    def is_playing(self) -> bool:
        return self.state == "playing"

    def is_stopped(self) -> bool:
        return self.state == "stopped"

    def play(self) -> None:
        self.state = "playing"

    def pause(self) -> None:
        self.state = "paused"

    def get_current_time(self) -> float:
        return self.time

    def get_time_codes_per_seconds(self) -> float:
        return 60.0

    def update(self) -> None:
        if self.state == "playing":
            self.time += self.dt


class _Selection:
    def __init__(self, paths: Iterable[str] = ()) -> None:
        self.paths = list(paths)
        self.history: List[List[str]] = []

    def get_selected_prim_paths(self) -> List[str]:
        return list(self.paths)

    def set_selected_prim_paths(self, paths: List[str], _expand: bool) -> None:
        self.paths = list(paths)
        self.history.append(list(paths))


class _Kit:
    """The fake modules one script sees."""

    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, stage: Optional[Usd.Stage] = None
    ) -> None:
        self.monkeypatch = monkeypatch
        self.stage = stage
        self.settings: Dict[str, Any] = {}
        self.timeline = _Timeline()
        self.enabled_extensions: set[str] = set()
        self.selection = _Selection()
        self.extra: Dict[str, types.ModuleType] = {}

    def module(self, name: str, **attrs: Any) -> types.ModuleType:
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        self.extra[name] = mod
        return mod

    def install(self) -> None:
        kit = self

        async def next_update_async() -> None:
            kit.timeline.update()
            await asyncio.sleep(0)

        manager = types.SimpleNamespace(
            is_extension_enabled=lambda name: name in kit.enabled_extensions
        )
        app = types.SimpleNamespace(
            get_extension_manager=lambda: manager,
            next_update_async=next_update_async,
        )
        context = types.SimpleNamespace(
            get_stage=lambda: kit.stage,
            get_stage_url=lambda: "anon:memory.usda",
            get_selection=lambda: kit.selection,
        )
        modules: Dict[str, types.ModuleType] = {
            "omni": types.ModuleType("omni"),
            "omni.usd": types.ModuleType("omni.usd"),
            "omni.timeline": types.ModuleType("omni.timeline"),
            "omni.kit": types.ModuleType("omni.kit"),
            "omni.kit.app": types.ModuleType("omni.kit.app"),
            "carb": types.ModuleType("carb"),
            "carb.settings": types.ModuleType("carb.settings"),
        }
        modules["omni.usd"].get_context = lambda: context  # type: ignore[attr-defined]
        modules["omni.timeline"].get_timeline_interface = lambda: kit.timeline  # type: ignore[attr-defined]
        modules["omni.kit.app"].get_app = lambda: app  # type: ignore[attr-defined]
        modules["carb.settings"].get_settings = lambda: types.SimpleNamespace(  # type: ignore[attr-defined]
            get=kit.settings.get, set=getattr(kit.settings, "__setitem__", None)
        )
        modules.update(self.extra)
        for name, module in modules.items():
            self.monkeypatch.setitem(sys.modules, name, module)
        # Wire each module onto its parent package, as an import would.
        for name, module in sorted(modules.items()):
            parent, _, child = name.rpartition(".")
            if parent in modules:
                setattr(modules[parent], child, module)


class _KitClient:
    """Client double that runs each script in process, the way Kit runs it."""

    bridge_enabled = False
    fallback_to_vscode = True
    script_timeout_seconds = 29.0

    def __init__(self, kit: _Kit) -> None:
        self.kit = kit
        self.calls: List[Tuple[str, Any]] = []

    async def execute(self, code: str, timeout_seconds: Any = None) -> ScriptResult:
        self.calls.append((code, timeout_seconds))
        self.kit.install()
        stdout = io.StringIO()
        try:
            # The code is a script IsaacTools just generated, run the way Kit
            # runs it (top-level await allowed); nothing external reaches it.
            compiled = compile(
                code, "<isaac-script>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT
            )
            with contextlib.redirect_stdout(stdout):
                result = eval(compiled, {"__name__": "__isaac_script__"})
                if result is not None:
                    await result
        except Exception as exc:  # noqa: BLE001 - reported like the transports do
            return ScriptResult(
                success=False,
                output=stdout.getvalue(),
                error_name=type(exc).__name__,
                error_value=str(exc),
            )
        return ScriptResult(success=True, output=stdout.getvalue())

    execute_vscode_only = execute
    execute_bridge_script_only = execute

    async def bridge_request(self, *_args: Any, **_kwargs: Any) -> Dict[str, Any]:
        raise AssertionError("bridge is disabled in these tests")


def _tools(kit: _Kit) -> Tuple[IsaacTools, _KitClient]:
    client = _KitClient(kit)
    return IsaacTools(client, settings=Settings()), client  # type: ignore[arg-type]


def _stage_with_cube() -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    cube = UsdGeom.Cube.Define(stage, "/World/Cube")
    cube.AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 3.0))
    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    return stage


# ---------------------------------------------------------------------------
# F08: USD poses and velocities lag the simulation under GPU physics
# ---------------------------------------------------------------------------


def _fabric_kit(monkeypatch: pytest.MonkeyPatch, state: str, **settings: Any) -> _Kit:
    kit = _Kit(monkeypatch, _stage_with_cube())
    kit.timeline.state = state
    kit.settings.update(settings)
    return kit


def test_pose_read_under_fabric_is_flagged_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kit = _fabric_kit(
        monkeypatch,
        "playing",
        **{"/physics/updateToUsd": False, "/physics/updateVelocitiesToUsd": False},
    )
    kit.enabled_extensions.add("omni.physx.fabric")
    tools, _ = _tools(kit)

    result = asyncio.run(tools.get_isaac_prim_transform("/World/Cube"))

    assert result["success"] is True
    assert result["translation"] == [
        1.0,
        2.0,
        3.0,
    ]  # the pre-play value, still returned
    assert result["stale_under_fabric"] is True
    assert result["usd_readback"]["physx_fabric_enabled"] is True
    assert USD_READBACK_STALE_WARNING in result["warnings"]


def test_velocity_read_under_fabric_is_flagged_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kit = _fabric_kit(
        monkeypatch, "paused", **{"/physics/updateVelocitiesToUsd": False}
    )
    tools, _ = _tools(kit)

    result = asyncio.run(tools.get_isaac_rigid_body_info("/World/Cube"))

    assert result["success"] is True
    assert result["stale_under_fabric"] is True
    assert result["usd_readback"]["update_velocities_to_usd"] is False
    assert result["usd_readback"]["poses_stale"] is False
    assert USD_VELOCITY_STALE_WARNING in result["warnings"]


def test_velocity_write_back_alone_does_not_flag_a_pose_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Poses were flagged stale when only /physics/updateVelocitiesToUsd was off."""
    kit = _fabric_kit(
        monkeypatch, "playing", **{"/physics/updateVelocitiesToUsd": False}
    )
    tools, _ = _tools(kit)

    result = asyncio.run(tools.get_isaac_prim_transform("/World/Cube"))

    assert result["success"] is True
    assert "stale_under_fabric" not in result
    assert "warnings" not in result


def test_a_write_back_setting_of_zero_counts_as_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The check compared with ``is False``, so an int 0 read as on."""
    kit = _fabric_kit(monkeypatch, "playing", **{"/physics/updateToUsd": 0})
    tools, _ = _tools(kit)

    result = asyncio.run(tools.get_isaac_prim_transform("/World/Cube"))

    assert result["stale_under_fabric"] is True
    assert USD_READBACK_STALE_WARNING in result["warnings"]


@pytest.mark.parametrize(
    ("state", "settings"),
    [
        ("stopped", {"/physics/updateToUsd": False}),  # USD holds the authored state
        (
            "playing",
            {"/physics/updateToUsd": True, "/physics/updateVelocitiesToUsd": True},
        ),
        ("playing", {}),
    ],
)
def test_pose_read_is_not_flagged_when_usd_is_current(
    monkeypatch: pytest.MonkeyPatch, state: str, settings: Dict[str, Any]
) -> None:
    tools, _ = _tools(_fabric_kit(monkeypatch, state, **settings))

    result = asyncio.run(tools.get_isaac_prim_transform("/World/Cube"))

    assert result["success"] is True
    assert "stale_under_fabric" not in result
    assert "usd_readback" not in result
    assert "warnings" not in result


def test_every_pose_reader_carries_the_check(monkeypatch: pytest.MonkeyPatch) -> None:
    kit = _fabric_kit(monkeypatch, "playing", **{"/physics/updateToUsd": False})
    camera = UsdGeom.Camera.Define(kit.stage, "/World/Cam")
    camera.AddTranslateOp().Set(Gf.Vec3d(0, 0, 5))
    tools, _ = _tools(kit)

    readers = {
        "prim_info": tools.get_isaac_prim_info("/World/Cube"),
        "transform": tools.get_isaac_prim_transform("/World/Cube"),
        "rigid_body": tools.get_isaac_rigid_body_info("/World/Cube"),
        "bounding_box": tools.get_isaac_bounding_box("/World/Cube"),
        "area": tools.find_isaac_prims_in_area([1.0, 2.0, 3.0], 10.0),
        "camera": tools.get_isaac_camera_info("/World/Cam"),
    }
    for name, call in readers.items():
        result = asyncio.run(call)
        assert result.get("stale_under_fabric") is True, (name, result)


def test_annotation_failures_never_fail_the_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kit = _fabric_kit(monkeypatch, "playing")

    # carb.settings.get raising stands in for any failure inside the check.
    def _broken_get(_key: str) -> Any:
        raise RuntimeError("boom")

    kit.settings = types.SimpleNamespace(get=_broken_get)  # type: ignore[assignment]
    tools, _ = _tools(kit)

    result = asyncio.run(tools.get_isaac_prim_transform("/World/Cube"))

    assert result["success"] is True
    assert "stale_under_fabric" not in result


def test_annotations_are_stripped_before_the_result_is_parsed() -> None:
    output = (
        '{"translation": [1, 2, 3]}\n'
        + ANNOTATION_MARKER
        + '{"fields": {"stale_under_fabric": true}, "warnings": ["w"]}'
    )

    remaining, notes = split_annotations(output)
    data: Dict[str, Any] = {"translation": [1, 2, 3], "stale_under_fabric": "kept"}
    merge_annotations(data, notes)

    assert remaining == '{"translation": [1, 2, 3]}'
    assert data["stale_under_fabric"] == "kept"  # never overrides the tool's own key
    assert data["warnings"] == ["w"]


def test_bridge_pose_actions_carry_the_same_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from khemoo.simul.protocol import BridgeRequest
    from khemoo.simul.service import BridgeCommandService, usd_readback_note

    kit = _fabric_kit(monkeypatch, "playing", **{"/physics/updateToUsd": False})
    kit.install()
    service = BridgeCommandService(
        executor=types.SimpleNamespace(), allow_unsafe_execution=False  # type: ignore[arg-type]
    )

    transform = asyncio.run(
        service.dispatch(
            BridgeRequest("1", "get_prim_transform", {"prim_path": "/World/Cube"})
        )
    )
    detail = asyncio.run(
        service.dispatch(
            BridgeRequest(
                "2",
                "get_prim_detail",
                {"prim_path": "/World/Cube", "aspects": ["mesh", "rigid_body"]},
            )
        )
    )
    mesh_only = asyncio.run(
        service.dispatch(
            BridgeRequest(
                "3",
                "get_prim_detail",
                {"prim_path": "/World/Cube", "aspects": ["mesh"]},
            )
        )
    )

    assert usd_readback_note()["stale_under_fabric"] is True
    assert transform.payload["stale_under_fabric"] is True
    assert detail.payload["stale_under_fabric"] is True
    assert "stale_under_fabric" not in mesh_only.payload  # nothing pose-shaped was read
    kit.timeline.state = "stopped"
    assert usd_readback_note() is None


def test_bridge_flags_velocity_write_back_only_where_velocities_are_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from khemoo.simul.protocol import BridgeRequest
    from khemoo.simul.service import BridgeCommandService

    kit = _fabric_kit(monkeypatch, "playing", **{"/physics/updateVelocitiesToUsd": 0})
    kit.install()
    service = BridgeCommandService(
        executor=types.SimpleNamespace(), allow_unsafe_execution=False  # type: ignore[arg-type]
    )

    transform = asyncio.run(
        service.dispatch(
            BridgeRequest("1", "get_prim_transform", {"prim_path": "/World/Cube"})
        )
    )
    rigid_body = asyncio.run(
        service.dispatch(
            BridgeRequest(
                "2",
                "get_prim_detail",
                {"prim_path": "/World/Cube", "aspects": ["rigid_body"]},
            )
        )
    )

    assert "stale_under_fabric" not in transform.payload
    assert rigid_body.payload["stale_under_fabric"] is True
    assert USD_VELOCITY_STALE_WARNING in rigid_body.payload["warnings"]


# ---------------------------------------------------------------------------
# F39: edits made while paused after a step are reverted by the next stop
# ---------------------------------------------------------------------------


def test_transform_edit_while_paused_warns_about_reset_on_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kit = _Kit(monkeypatch, _stage_with_cube())
    kit.timeline.state = "paused"
    kit.settings["/physics/resetOnStop"] = True
    tools, _ = _tools(kit)

    result = asyncio.run(
        tools.set_isaac_prim_transform("/World/Cube", translation=[0.0, 0.0, 9.0])
    )

    assert result["success"] is True
    assert result["timeline_state"] == "paused"
    assert PAUSED_EDIT_WARNING in result["warnings"]


def test_transform_edit_while_playing_warns_about_reset_on_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reset-on-stop also reverts an edit made during play; that one was silent."""
    kit = _Kit(monkeypatch, _stage_with_cube())
    kit.timeline.state = "playing"
    kit.settings["/physics/resetOnStop"] = True
    tools, _ = _tools(kit)

    result = asyncio.run(
        tools.set_isaac_prim_transform("/World/Cube", translation=[0.0, 0.0, 9.0])
    )

    assert result["success"] is True
    assert result["timeline_state"] == "playing"
    assert PLAYING_EDIT_WARNING in result["warnings"]


@pytest.mark.parametrize(
    ("state", "reset_on_stop"), [("stopped", True), ("paused", False), ("playing", 0)]
)
def test_no_paused_edit_warning_otherwise(
    monkeypatch: pytest.MonkeyPatch, state: str, reset_on_stop: bool
) -> None:
    kit = _Kit(monkeypatch, _stage_with_cube())
    kit.timeline.state = state
    kit.settings["/physics/resetOnStop"] = reset_on_stop
    tools, _ = _tools(kit)
    # reset_on_stop=0 is off too: the check tests truthiness, not ``is False``.

    result = asyncio.run(
        tools.set_isaac_prim_transform("/World/Cube", translation=[0.0, 0.0, 9.0])
    )

    assert result["success"] is True
    assert "warnings" not in result


def test_step_description_documents_the_paused_timeline_and_reset_on_stop() -> None:
    from simul.mcp.tools.isaac.simulation import SimulationMixin

    description = SimulationMixin.step_isaac_simulation.__tool_meta__.description  # type: ignore[attr-defined]

    assert "paused" in description and "resetOnStop" in description
    assert "physics_dt" in description


# ---------------------------------------------------------------------------
# F22: frames, physics steps and a budget that scales with N
# ---------------------------------------------------------------------------


def test_step_reports_frame_dt_and_physics_rates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdPhysics.Scene.Define(stage, "/physicsScene")
    kit = _Kit(monkeypatch, stage)
    kit.timeline = _Timeline("stopped", dt=1.0 / 60.0)
    kit.settings.update(
        {
            "/app/runLoops/main/rateLimitEnabled": True,
            "/app/runLoops/main/rateLimitFrequency": 60,
        }
    )
    _install_fake_physx_schema(monkeypatch, rate=120.0)
    tools, client = _tools(kit)

    result = asyncio.run(tools.step_isaac_simulation(num_steps=6))

    assert result["success"] is True
    assert result["steps"] == 6
    assert result["frame_dt"] == pytest.approx(1.0 / 60.0)
    physics = result["physics"]
    assert physics["physics_scene"] == "/physicsScene"
    assert physics["physics_steps_per_second"] == 120.0
    assert physics["physics_dt"] == pytest.approx(1.0 / 120.0)
    assert physics["physics_steps_per_frame"] == pytest.approx(2.0)
    assert physics["physics_steps_estimated"] == 12
    assert physics["rate_limit_hz"] == 60
    assert physics["timeline_time_codes_per_second"] == 60.0


def test_step_budget_scales_with_frames_and_is_capped() -> None:
    tools, _ = _tools(_Kit(pytest.MonkeyPatch()))

    assert tools._step_budget(1) == 29.0  # never below the default script budget
    assert tools._step_budget(500) == pytest.approx(60.0)
    assert tools._step_budget(1000) == 110.0


def test_step_sends_the_budget_to_the_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kit = _Kit(monkeypatch, Usd.Stage.CreateInMemory())
    tools, client = _tools(kit)

    asyncio.run(tools.step_isaac_simulation(num_steps=1000))

    _, timeout = client.calls[-1]
    assert timeout == 115.0
    assert "budget = 110.0" in client.calls[-1][0]


def test_step_that_runs_out_of_budget_stops_paused_and_says_how_far(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kit = _Kit(monkeypatch, Usd.Stage.CreateInMemory())
    tools, _ = _tools(kit)
    monkeypatch.setattr(tools, "_step_budget", lambda _n: 0.0)

    result = asyncio.run(tools.step_isaac_simulation(num_steps=10))

    assert result["success"] is False
    assert "budget ran out" in result["error"]
    assert "Step fewer frames" in result["error"]
    assert kit.timeline.state == "paused"


def test_bridge_step_reports_rates_and_honours_the_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from khemoo.simul import service as service_module
    from khemoo.simul.service import BridgeCommandService

    timeline = _Timeline("paused", dt=1.0 / 60.0)

    async def next_update_async() -> None:
        timeline.update()

    app = types.SimpleNamespace(next_update_async=next_update_async)
    monkeypatch.setattr(
        service_module,
        "_physics_scene_rate",
        lambda: ("/physicsScene", 240.0, "physxScene:timeStepsPerSecond"),
    )

    full = asyncio.run(BridgeCommandService._step_timeline(timeline, app, 3))
    cut = asyncio.run(
        BridgeCommandService._step_timeline(timeline, app, 3, budget_seconds=0.0)
    )

    assert full["steps"] == 3
    assert full["physics"]["physics_steps_per_frame"] == pytest.approx(4.0)
    assert full["physics"]["physics_steps_estimated"] == 12
    assert cut["steps"] == 0
    assert "budget ran out" in cut["error"]


def test_bridge_step_payload_carries_the_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import AsyncMock, MagicMock

    client = MagicMock()
    client.bridge_enabled = True
    client.fallback_to_vscode = True
    client.script_timeout_seconds = 29.0
    client.bridge_request = AsyncMock(
        return_value={"status": "ok", "payload": {"steps": 1000}}
    )
    tools = IsaacTools(client, settings=Settings())

    asyncio.run(tools.step_isaac_simulation(num_steps=1000))

    args, kwargs = client.bridge_request.await_args
    assert args[1] == {"command": "step", "num_steps": 1000, "timeout": 110.0}
    assert kwargs == {"read_timeout": 115.0}


def _install_fake_physx_schema(monkeypatch: pytest.MonkeyPatch, rate: float) -> None:
    """PhysxSchema ships with Isaac Sim, not usd-core; fake just what is read."""

    class _Attr:
        def __init__(self, value: Any) -> None:
            self.value = value

        def Get(self) -> Any:
            return self.value

    class PhysxSceneAPI:
        def __init__(self, prim: Any) -> None:
            self.prim = prim

        def GetTimeStepsPerSecondAttr(self) -> _Attr:
            return _Attr(rate)

        def GetEnableGPUDynamicsAttr(self) -> _Attr:
            return _Attr(True)

        def GetSolverTypeAttr(self) -> _Attr:
            return _Attr("TGS")

        def GetBroadphaseTypeAttr(self) -> _Attr:
            return _Attr("GPU")

    module = types.ModuleType("pxr.PhysxSchema")
    module.PhysxSceneAPI = PhysxSceneAPI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pxr.PhysxSchema", module)
    import pxr

    monkeypatch.setattr(pxr, "PhysxSchema", module, raising=False)
    # Usd.Prim.HasAPI needs a registered schema; report it present for scenes.
    original = Usd.Prim.HasAPI

    def _has_api(prim: Usd.Prim, api: Any, *args: Any) -> bool:
        if api is PhysxSceneAPI:
            return prim.IsA(UsdPhysics.Scene)
        return original(prim, api, *args)

    monkeypatch.setattr(Usd.Prim, "HasAPI", _has_api)


# ---------------------------------------------------------------------------
# F10: annotators that return {"data": ..., "info": ...}
# ---------------------------------------------------------------------------


class _Annotator:
    def __init__(self, data: Any) -> None:
        self.data = data

    def attach(self, _products: Any) -> None:
        pass

    def detach(self, _products: Any) -> None:
        pass

    def get_data(self) -> Any:
        return self.data


def _read_aovs_with(
    monkeypatch: pytest.MonkeyPatch, annotators: Dict[str, Any]
) -> Dict[str, Any]:
    kit = _Kit(monkeypatch)

    async def step_async(**_kwargs: Any) -> None:
        return None

    rp = types.SimpleNamespace(path="/Render/RP", destroy=lambda: None)
    kit.module(
        "omni.replicator.core",
        create=types.SimpleNamespace(render_product=lambda *_a: rp),
        AnnotatorRegistry=types.SimpleNamespace(
            get_annotator=lambda name: _Annotator(annotators[name])
        ),
        orchestrator=types.SimpleNamespace(step_async=step_async),
    )
    kit.module("omni.replicator")
    tools, _ = _tools(kit)
    return asyncio.run(tools.read_aovs(aov_names=list(annotators), num_frames=1))


def test_dict_shaped_annotators_report_data_and_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np = pytest.importorskip("numpy")
    boxes = np.zeros(
        2,
        dtype=[
            ("semanticId", "<u4"),
            ("x_min", "<i4"),
            ("y_min", "<i4"),
            ("x_max", "<i4"),
            ("y_max", "<i4"),
        ],
    )
    boxes["x_max"] = [10, 30]
    semantic = np.zeros((4, 4, 1), dtype=np.uint32)
    semantic[1:3, 1:3] = 1

    result = _read_aovs_with(
        monkeypatch,
        {
            "semantic_segmentation": {
                "data": semantic,
                "info": {
                    "idToLabels": {"0": {"class": "BACKGROUND"}, "1": {"class": "cube"}}
                },
            },
            "bounding_box_2d_tight": {
                "data": boxes,
                "info": {"primPaths": ["/World/A", "/World/B"]},
            },
            "pointcloud": {
                "data": np.ones((5, 3), dtype=np.float32),
                "info": {"pointRgb": np.zeros((5, 4))},
            },
            "HdrColor": np.ones((2, 2, 4), dtype=np.float32),
        },
    )

    assert result["success"] is True
    assert "failed_aovs" not in result
    aovs = result["aovs"]
    seg = aovs["semantic_segmentation"]
    assert seg["output"] == "dict" and seg["max"] == 1.0
    assert seg["info"]["idToLabels"]["sample"]["1"] == {"class": "cube"}
    bbox = aovs["bounding_box_2d_tight"]
    assert bbox["count"] == 2 and "x_max" in bbox["fields"]
    assert bbox["field_ranges"]["x_max"] == {"min": 10.0, "max": 30.0}
    assert bbox["info"]["primPaths"]["count"] == 2
    assert aovs["pointcloud"]["shape"] == [5, 3]
    assert aovs["pointcloud"]["info"]["pointRgb"] == {
        "shape": [5, 4],
        "dtype": "float64",
    }
    assert aovs["HdrColor"]["output"] == "array"


def test_no_boxes_in_view_is_data_not_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np = pytest.importorskip("numpy")
    empty = np.zeros(0, dtype=[("semanticId", "<u4"), ("x_min", "<i4")])

    result = _read_aovs_with(
        monkeypatch,
        {"bounding_box_2d_loose": {"data": empty, "info": {"primPaths": []}}},
    )

    assert result["success"] is True
    assert result["aovs"]["bounding_box_2d_loose"]["count"] == 0


def test_an_empty_dict_output_is_still_no_data(monkeypatch: pytest.MonkeyPatch) -> None:
    np = pytest.importorskip("numpy")

    result = _read_aovs_with(
        monkeypatch, {"semantic_segmentation": {"data": np.zeros(0), "info": {}}}
    )

    assert result["success"] is False
    assert result["failed_aovs"] == ["semantic_segmentation"]


# ---------------------------------------------------------------------------
# F24: focus frames the selection the way Isaac Sim's Edit menu does
# ---------------------------------------------------------------------------


def _focus_kit(monkeypatch: pytest.MonkeyPatch, framing: Optional[Any]) -> _Kit:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Cube.Define(stage, "/World/Cube")
    camera = UsdGeom.Camera.Define(stage, "/OmniverseKit_Persp")
    translate = camera.AddTranslateOp()
    translate.Set(Gf.Vec3d(0, 0, 50))
    kit = _Kit(monkeypatch, stage)
    kit.selection = _Selection(["/World/Other"])
    viewport = types.SimpleNamespace(
        camera_path=Sdf.Path("/OmniverseKit_Persp"), resolution=(1280, 720)
    )
    attrs: Dict[str, Any] = {"get_active_viewport": lambda: viewport}
    if framing is not None:
        attrs["frame_viewport_selection"] = lambda: framing(kit, translate)
    kit.module("omni.kit.viewport")
    kit.module("omni.kit.viewport.utility", **attrs)
    return kit


def test_focus_frames_through_the_module_function_and_restores_the_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: List[List[str]] = []

    def framing(kit: _Kit, translate: Any) -> bool:
        seen.append(kit.selection.get_selected_prim_paths())
        translate.Set(Gf.Vec3d(0, 0, 3))
        return True

    kit = _focus_kit(monkeypatch, framing)
    tools, _ = _tools(kit)

    result = asyncio.run(tools.focus_isaac_viewport("/World/Cube"))

    assert result["success"] is True
    assert seen == [["/World/Cube"]]  # framed with the prim selected
    assert result["focused"] is True and result["camera_moved"] is True
    assert result["selection_restored"] is True
    assert kit.selection.get_selected_prim_paths() == ["/World/Other"]


def test_focus_without_evidence_is_not_reported_as_focused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kit = _focus_kit(monkeypatch, lambda _kit, _translate: None)
    tools, _ = _tools(kit)

    result = asyncio.run(tools.focus_isaac_viewport("/World/Cube"))

    assert result["focused"] is False
    assert result["camera_moved"] is False
    assert "warning" in result


def test_focus_without_the_framing_function_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kit = _focus_kit(monkeypatch, None)
    tools, _ = _tools(kit)

    result = asyncio.run(tools.focus_isaac_viewport("/World/Cube"))

    assert result["success"] is False
    assert "frame_viewport_selection" in result["error"]
    assert kit.selection.get_selected_prim_paths() == ["/World/Other"]  # untouched


# ---------------------------------------------------------------------------
# F19: raycast direction, stopped timeline, shape and body
# ---------------------------------------------------------------------------


def _raycast(
    monkeypatch: pytest.MonkeyPatch,
    state: str,
    hit: Optional[Dict[str, Any]],
    direction: List[float],
) -> Tuple[Dict[str, Any], List[Any]]:
    kit = _Kit(monkeypatch)
    kit.timeline.state = state
    calls: List[Any] = []

    def raycast_closest(origin: Any, unit: Any, distance: float) -> Dict[str, Any]:
        calls.append((origin, unit, distance))
        return hit or {"hit": False}

    query = types.SimpleNamespace(raycast_closest=raycast_closest)
    kit.module("omni.physx", get_physx_scene_query_interface=lambda: query)
    tools, _ = _tools(kit)
    return (
        asyncio.run(
            tools.raycast_isaac_scene([0.0, 0.0, 10.0], direction, max_distance=100.0)
        ),
        calls,
    )


def test_raycast_normalizes_the_direction_and_returns_shape_and_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hit = {
        "hit": True,
        "collision": "/World/Robot/base/collisions",
        "rigidBody": "/World/Robot/base",
        "position": (0.0, 0.0, 1.0),
        "normal": (0.0, 0.0, 1.0),
        "distance": 9.0,
    }

    result, calls = _raycast(monkeypatch, "playing", hit, [0.0, 0.0, -5.0])

    assert calls[0][1] == (0.0, 0.0, -1.0)
    assert result["collision_path"] == "/World/Robot/base/collisions"
    assert result["rigid_body_path"] == "/World/Robot/base"
    assert result["prim_path"] == "/World/Robot/base/collisions"


def test_a_miss_on_a_stopped_timeline_is_not_reported_as_no_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _ = _raycast(monkeypatch, "stopped", None, [0.0, 0.0, -1.0])

    assert result["success"] is False
    assert result["error_type"] == "SimulationNotRunning"
    assert result["status"] == "simulation_not_running"


def test_a_miss_while_simulating_is_a_plain_no_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _ = _raycast(monkeypatch, "playing", None, [0.0, 0.0, -1.0])

    assert result["success"] is True
    assert result["hit"] is False and result["status"] == "no_hit"


def test_a_zero_direction_is_refused_before_anything_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tools, client = _tools(_Kit(monkeypatch))

    result = asyncio.run(tools.raycast_isaac_scene([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]))

    assert result["success"] is False
    assert client.calls == []


# ---------------------------------------------------------------------------
# F25 / F26 / F27: runtime info, render vars, camera resolution
# ---------------------------------------------------------------------------


def test_runtime_info_reports_keys_that_exist(monkeypatch: pytest.MonkeyPatch) -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    kit = _Kit(monkeypatch, stage)
    kit.settings.update(
        {
            "/physics/cudaDevice": 0,
            "/physics/updateToUsd": False,
            "/physics/resetOnStop": True,
        }
    )
    kit.enabled_extensions.add("omni.physx.fabric")
    _install_fake_physx_schema(monkeypatch, rate=60.0)
    tools, _ = _tools(kit)

    result = asyncio.run(tools.get_runtime_info())

    physics = result["physics"]
    assert physics["cuda_device"] == 0
    assert physics["update_to_usd"] is False
    assert physics["physx_fabric_enabled"] is True
    assert physics["reset_on_stop"] is True
    assert "stats_unavailable" not in physics
    config = result["physics_config"]
    assert config["scene_path"] == "/World/physicsScene"
    assert config["time_steps_per_second"] == 60.0
    assert config["physics_dt"] == pytest.approx(1.0 / 60.0)
    assert config["gpu_dynamics_enabled"] is True
    assert config["solver_type"] == "TGS"


def test_runtime_info_no_longer_reads_keys_isaac_sim_does_not_have() -> None:
    import inspect

    from simul.mcp.tools.isaac import diagnostics
    from khemoo.simul import service

    for source in (inspect.getsource(diagnostics), inspect.getsource(service)):
        for missing in (
            "/physics/gpuDynamicsEnabled",
            "/persistent/simulation/defaultPhysicsDt",
            "/persistent/physics/solverType",
        ):
            assert f'settings.get("{missing}")' not in source


def test_runtime_info_description_drops_the_physics_stats_promise() -> None:
    from simul.mcp.tools.isaac.diagnostics import DiagnosticsMixin

    description = DiagnosticsMixin.get_runtime_info.__tool_meta__.description  # type: ignore[attr-defined]

    assert "articulation/shape counts" not in description
    assert "CUDA availability" not in description


def test_render_vars_list_only_what_syntheticdata_has(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SyntheticData:
        @staticmethod
        def get_registered_visualization_template_names() -> List[str]:
            return ["SemanticSegmentationDisplay", "LdrColorDisplay"]

    kit = _Kit(monkeypatch)
    kit.module("omni.syntheticdata", SyntheticData=SyntheticData)
    tools, _ = _tools(kit)

    result = asyncio.run(tools.list_render_vars())

    assert result["success"] is True
    assert result["render_var_templates"] == [
        "LdrColorDisplay",
        "SemanticSegmentationDisplay",
    ]
    assert "sensor_types" not in result


def test_camera_info_reports_the_active_viewport_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Camera.Define(stage, "/OmniverseKit_Persp")
    UsdGeom.Camera.Define(stage, "/World/Other")
    kit = _Kit(monkeypatch, stage)
    viewport = types.SimpleNamespace(
        camera_path=Sdf.Path("/OmniverseKit_Persp"), resolution=(1920, 1080)
    )
    kit.module("omni.kit.viewport")
    kit.module("omni.kit.viewport.utility", get_active_viewport=lambda: viewport)
    tools, _ = _tools(kit)

    active = asyncio.run(tools.get_isaac_camera_info())
    other = asyncio.run(tools.get_isaac_camera_info("/World/Other"))

    assert active["resolution"] == [1920, 1080]
    assert active["resolution_source"] == "active_viewport"
    assert other["resolution"] is None


# ---------------------------------------------------------------------------
# F31 / F32 / F35 / F37
# ---------------------------------------------------------------------------


def test_execute_script_description_says_the_namespace_is_shared() -> None:
    from simul.mcp.tools.isaac.core import CoreToolsMixin

    description = CoreToolsMixin.execute_script.__tool_meta__.description  # type: ignore[attr-defined]

    assert "independent" not in description
    assert "shared namespace" in description


def test_logs_read_this_instance_log_named_by_kit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    log = tmp_path / "kit_this_instance.log"
    log.write_text("2026-01-01T00:00:00Z [Error] [omni.physx] boom\n")
    kit = _Kit(monkeypatch)
    kit.settings["/log/file"] = str(log)
    tools, _ = _tools(kit)

    result = asyncio.run(tools.get_isaac_logs(level="error"))

    assert result["log_file"] == str(log)
    assert result["log_file_source"] == "carb:/log/file"
    assert "warning" not in result
    assert result["entries"][0]["message"] == "boom"


def test_logs_fallback_says_it_may_be_another_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    logs = tmp_path / ".nvidia-omniverse" / "logs" / "Kit" / "app"
    logs.mkdir(parents=True)
    (logs / "kit_1.log").write_text("[Warning] [x] hi\n")
    monkeypatch.setenv("HOME", str(tmp_path))
    tools, _ = _tools(_Kit(monkeypatch))

    result = asyncio.run(tools.get_isaac_logs())

    assert result["log_file_source"] == "newest_kit_log"
    assert "another Kit process" in result["warning"]


def test_the_5x_transport_keys_are_protected() -> None:
    assert "/exts/isaacsim.code_editor.vscode/" in PROTECTED_CARB_SETTING_PREFIXES


def test_persistent_writes_need_explicit_consent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kit = _Kit(monkeypatch)
    tools, client = _tools(kit)

    refused = asyncio.run(
        tools.set_carb_settings({"/persistent/app/viewport/grid/enabled": False})
    )
    assert refused["error_type"] == "RefusedOperation"
    assert client.calls == []

    applied = asyncio.run(
        tools.set_carb_settings(
            {"/persistent/app/viewport/grid/enabled": False}, allow_persistent=True
        )
    )
    assert applied["success"] is True
    assert applied["persistent_keys"] == ["/persistent/app/viewport/grid/enabled"]
    assert "later sessions" in applied["warning"]


def test_core_api_resource_has_the_6_0_note() -> None:
    from simul.resources import __file__ as resources_init

    core = (Path(resources_init).parent / "docs" / "api" / "core.md").read_text()

    assert "Version note" in core
    assert "isaacsim.core.experimental.prims" in core
    assert "XformPrim" in core
