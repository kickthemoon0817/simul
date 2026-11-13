"""Round-2 M-1: every Isaac value that becomes a file location goes through the sandbox.

``create_isaac_light``'s ``texture_file`` and ``set_isaac_prim_attribute``
refused an outside path, but ``create_isaac_prim``'s ``attributes`` wrote one
into ``inputs:texture:file`` unchecked, and ``set_isaac_graph_node_values``
and ``set_isaac_carb_settings`` passed file locations (an asset-typed input,
``/log/file``) straight to Kit. Kit then reads a local image a capture shows
the agent, fetches an ``http`` URL, or on Windows leaks NTLM credentials to a
UNC host.

The tools that take free-form values are found from their signatures, so a
new one fails ``test_every_free_form_tool_has_a_case`` until it is covered.
Each is run on real ``pxr`` (and a stand-in OmniGraph) with every outside
form, and must answer ``SandboxError`` and leave the stage as it was.
"""

from __future__ import annotations

import asyncio
import inspect
import types
import typing
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import pytest
from pxr import Sdf, Usd, UsdGeom, UsdShade

from simul.config import Settings
from simul.mcp.tools._meta import TOOL_META_ATTRIBUTE
from simul.mcp.tools.isaac._shared import names_file_location
from simul.mcp.tools.isaac_tools import IsaacTools
from tests.isaac.conftest import StageTools

# Every outside form the audit tried: absolute, "..", "~", file://, http://, UNC.
OUTSIDE = [
    "/etc/hosts.png",
    "{sandbox}/../../etc/x.png",
    "~/x.png",
    "file:///etc/x.png",
    "http://evil.example/x.png",
    "\\\\attacker.example\\share\\t.png",
]


def _settings(sandbox: Path) -> Settings:
    settings = Settings()
    security = settings.security.model_copy(
        update={"allowed_paths": [str(sandbox)], "relative_path_base": str(sandbox.parent)}
    )
    return settings.model_copy(update={"security": security})


def _run(coro: Any) -> Dict[str, Any]:
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# A stage with one of everything the free-form tools write to
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
    """``og.Controller`` over one node, /World/Graph/Reader."""

    nodes: Dict[str, _Node] = {}

    @classmethod
    def node(cls, path: str) -> _Node:
        return cls.nodes[path]

    @staticmethod
    def attribute(name: str, node_path: str) -> Tuple[str, str]:
        return node_path, name

    @classmethod
    def set(cls, attribute: Tuple[str, str], value: Any) -> None:
        node_path, name = attribute
        cls.nodes[node_path].values[name] = value


def _og_modules() -> Dict[str, types.ModuleType]:
    _Controller.nodes = {"/World/Graph/Reader": _Node("isaacsim.core.nodes.IsaacReadFilePath")}
    omni_graph = types.ModuleType("omni.graph")
    core = types.ModuleType("omni.graph.core")
    core.Controller = _Controller  # type: ignore[attr-defined]
    omni_graph.core = core  # type: ignore[attr-defined]
    return {"omni.graph": omni_graph, "omni.graph.core": core}


def _stage() -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    material = UsdShade.Material.Define(stage, "/World/Looks/M")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/M/Shader")
    shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    material.CreateSurfaceOutput("mdl").ConnectToSource(
        shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    )
    prim = stage.DefinePrim("/World/Dome", "DomeLight")
    prim.CreateAttribute("textures", Sdf.ValueTypeNames.AssetArray)
    node = stage.DefinePrim("/World/Graph/Reader")
    node.CreateAttribute("inputs:texture", Sdf.ValueTypeNames.Asset)
    return stage


def _tools(stage_tools: StageTools, stage: Usd.Stage, sandbox: Path) -> Tuple[IsaacTools, List[str]]:
    return stage_tools(stage, settings=_settings(sandbox), modules=_og_modules())


def _texture(stage: Usd.Stage, path: str, name: str) -> Any:
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return None
    attr = prim.GetAttribute(name)
    return attr.Get() if attr.IsValid() else None


# Each free-form tool: (call with an outside value, what the stage holds after).
Case = Tuple[Callable[[IsaacTools, Any], Any], Callable[[Usd.Stage], Any]]
CASES: Dict[Tuple[str, str], Case] = {
    ("create_isaac_prim", "attributes"): (
        lambda tools, bad: tools.create_isaac_prim(
            "/World/Sky", "DomeLight", {"inputs:intensity": 500.0, "inputs:texture:file": bad}
        ),
        lambda stage: stage.GetPrimAtPath("/World/Sky").IsValid(),
    ),
    ("set_isaac_prim_attribute", "value"): (
        lambda tools, bad: tools.set_isaac_prim_attribute("/World/Dome", "textures", ["", bad]),
        lambda stage: _texture(stage, "/World/Dome", "textures"),
    ),
    ("set_isaac_material_property", "value"): (
        lambda tools, bad: tools.set_isaac_material_property("/World/Looks/M", "diffuse_texture", bad),
        lambda stage: _texture(stage, "/World/Looks/M/Shader", "inputs:diffuse_texture"),
    ),
    ("set_isaac_graph_node_values", "values"): (
        lambda tools, bad: tools.set_isaac_graph_node_values(
            "/World/Graph/Reader", {"inputs:texture": bad}
        ),
        lambda stage: _Controller.nodes["/World/Graph/Reader"].values or None,
    ),
    ("set_isaac_carb_settings", "settings"): (
        lambda tools, bad: tools.set_carb_settings({"/rtx/fog/enabled": True, "/log/file": bad}),
        lambda stage: None,
    ),
}


def _free_form_parameters() -> List[Tuple[str, str]]:
    """(tool, parameter) for every Isaac tool parameter typed ``Any`` or a dict."""
    found = []
    for _, method in inspect.getmembers(IsaacTools, inspect.isfunction):
        meta = getattr(method, TOOL_META_ATTRIBUTE, None)
        if meta is None or meta.script:
            continue
        hints = typing.get_type_hints(method)
        for name in inspect.signature(method).parameters:
            hint = hints.get(name)
            if hint is Any or "Dict" in str(hint) or "dict" in str(hint):
                found.append((meta.name, name))
    return sorted(found)


def test_every_free_form_tool_has_a_case() -> None:
    """A new tool taking a free-form value must be added to CASES (and sandboxed)."""
    assert _free_form_parameters() == sorted(CASES)


@pytest.mark.parametrize("bad", OUTSIDE)
@pytest.mark.parametrize("case", sorted(CASES), ids=lambda case: case[0])
def test_an_outside_path_is_refused_and_nothing_is_written(
    stage_tools: StageTools, tmp_path: Path, case: Tuple[str, str], bad: str
) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    stage = _stage()
    tools, sent = _tools(stage_tools, stage, sandbox)
    call, written = CASES[case]

    result = _run(call(tools, bad.format(sandbox=sandbox)))

    assert result["success"] is False
    assert result["error_type"] == "SandboxError", result
    assert not written(stage)
    if case[0] == "set_isaac_carb_settings":
        assert sent == [], "a refused settings write sends nothing"


# ---------------------------------------------------------------------------
# Inside the sandbox: written as the path the check resolved
# ---------------------------------------------------------------------------


def test_create_prim_writes_an_inside_texture_resolved(stage_tools: StageTools, tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    stage = _stage()
    tools, _ = _tools(stage_tools, stage, sandbox)

    result = _run(tools.create_isaac_prim(
        "/World/Sky", "DomeLight", {"inputs:texture:file": str(sandbox / "sub" / ".." / "sky.hdr")}
    ))

    assert result["success"] is True, result
    written = _texture(stage, "/World/Sky", "inputs:texture:file")
    assert written.path == str((sandbox / "sky.hdr").resolve())


def test_material_texture_inside_is_resolved_and_a_number_is_a_type_error(
    stage_tools: StageTools, tmp_path: Path
) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    stage = _stage()
    tools, _ = _tools(stage_tools, stage, sandbox)

    inside = _run(tools.set_isaac_material_property(
        "/World/Looks/M", "diffuse_texture", str(sandbox / "albedo.png")
    ))
    number = _run(tools.set_isaac_material_property("/World/Looks/M", "diffuse_texture", 5))

    assert inside["success"] is True and inside["value_set"] == str((sandbox / "albedo.png").resolve())
    assert number["error_type"] == "ValidationError"


def test_graph_file_inputs_go_through_the_sandbox(stage_tools: StageTools, tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    stage = _stage()
    tools, sent = _tools(stage_tools, stage, sandbox)

    by_name = _run(tools.set_isaac_graph_node_values("/World/Graph/Reader", {"inputs:path": "/etc/passwd"}))
    assert by_name["error_type"] == "SandboxError" and sent == []
    assert by_name["details"]["attribute"] == "inputs:path"

    inside = _run(tools.set_isaac_graph_node_values(
        "/World/Graph/Reader",
        {"inputs:path": str(sandbox / "a.txt"), "inputs:texture": str(sandbox / "t.png"), "inputs:primPath": "/World"},
    ))
    assert inside["success"] is True, inside
    assert _Controller.nodes["/World/Graph/Reader"].values == {
        "inputs:path": str((sandbox / "a.txt").resolve()),
        "inputs:texture": str((sandbox / "t.png").resolve()),
        "inputs:primPath": "/World",
    }


def test_carb_file_keys_are_checked_as_writes_and_resolved(stage_tools: StageTools, tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    tools, sent = _tools(stage_tools, _stage(), sandbox)

    nested = _run(tools.set_carb_settings({"/log": {"level": "info", "file": "/etc/x.log"}}))
    listed = _run(tools.set_carb_settings({"/app/exts/folders": [str(sandbox), "/opt/exts"]}))
    omniverse = _run(tools.set_carb_settings({"/app/captureDir": "omniverse://host/out"}))
    assert [r["error_type"] for r in (nested, listed, omniverse)] == ["SandboxError"] * 3
    assert nested["details"]["setting"] == "/log/file"
    assert sent == []

    tools._client.execute.reset_mock()
    _run(tools.set_carb_settings({
        "/log/file": str(sandbox / "kit.log"),
        "/rtx/rendermode": "PathTracing",
        "/app/viewport/defaultCameraPath": "/OmniverseKit_Persp",
    }))
    script = sent[-1]
    assert repr(str((sandbox / "kit.log").resolve())) in script
    assert "'PathTracing'" in script and "'/OmniverseKit_Persp'" in script


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("/log/file", True),
        ("/crashreporter/dumpDir", True),
        ("/app/exts/folders", True),
        ("inputs:path", True),
        ("inputs:outputDir", True),
        ("inputs:usdPath", True),
        ("/telemetry/endpointURL", True),
        ("inputs:primPath", False),
        ("/app/viewport/defaultCameraPath", False),
        ("inputs:targetPaths", False),
        ("/rtx/pathtracing/spp", False),
        ("/rtx/rendermode", False),
        ("inputs:velocity", False),
    ],
)
def test_names_file_location(name: str, expected: bool) -> None:
    assert names_file_location(name) is expected


def test_the_sandbox_off_passes_values_through(stage_tools: StageTools, tmp_path: Path) -> None:
    settings = Settings()
    settings = settings.model_copy(
        update={"security": settings.security.model_copy(update={"sandbox_enabled": False})}
    )
    stage = _stage()
    tools, _ = stage_tools(stage, settings=settings, modules=_og_modules())

    result = _run(tools.create_isaac_prim("/World/Sky", "DomeLight", {"inputs:texture:file": "/etc/sky.hdr"}))

    assert result["success"] is True
    assert _texture(stage, "/World/Sky", "inputs:texture:file").path == "/etc/sky.hdr"
