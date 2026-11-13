"""Unreal's own USD exports load and validate inside the sandbox.

Unreal's exporter writes each mesh's source material as an asset-valued
attribute holding a content path,
``info:unreal:sourceAsset = @/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial@``.
That names an object in the editor's content, not a file, but the composition
sandbox held it to the allowed roots like a texture, so ``load_usd_file`` and
``validate_usd_file`` refused every Unreal export with ``SandboxError``, and with
the sandbox off validation failed it on pxr's ``UnresolvableDependency`` and
``MissingSourceTypeInRegistry``. The exemption lives in ``CompositionSandbox``
now, so the headless tools and Unreal's import check (``_check_usd_composition``)
apply one rule: only an asset value (never a sublayer, reference, payload or
clip) shaped like ``/Engine|Game|Script/.../Name.Name`` that names no file.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

pytest.importorskip("pxr", reason="pxr library not available")

from fastmcp import Client  # noqa: E402

from simul.adapters.headless_usd import HeadlessUSDSession  # noqa: E402
from simul.config import Settings  # noqa: E402
from simul.mcp.server import SimulMCPServer  # noqa: E402
from simul.usd import sandbox as usd_sandbox  # noqa: E402
from simul.usd.sandbox import CompositionSandbox, is_unreal_content_path  # noqa: E402
from simul.usd.validation import validate_usd_file  # noqa: E402
from simul.utils.paths import PathPolicy, SandboxDenied  # noqa: E402

BASIC = "/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial"
GRID = "/Engine/EngineMaterials/WorldGridMaterial.WorldGridMaterial"

# The shape of Unreal 5.7's export of a level holding one engine cube: a mesh
# layer naming its material by content path, and a level layer referencing
# it and overriding that material. ``{identifier}`` is where the export was
# written; ``assetInfo`` records it.
UE_CUBE = """#usda 1.0
(
    defaultPrim = "Cube"
    metersPerUnit = 0.01
    upAxis = "Z"
)

def Mesh "Cube" (
    prepend apiSchemas = ["MaterialBindingAPI"]
    assetInfo = {{
        asset identifier = @{identifier}@
        string name = "Cube"
    }}
)
{{
    float3[] extent = [(-50, -50, -50), (50, 50, 50)]
    int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
    int[] faceVertexIndices = [0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7]
    point3f[] points = [(-50, -50, -50), (50, -50, -50), (50, 50, -50), (-50, 50, -50),
        (-50, -50, 50), (50, -50, 50), (50, 50, 50), (-50, 50, 50)]
    rel material:binding = </Cube/UnrealMaterial>

    def Material "UnrealMaterial"
    {{
        token outputs:unreal:surface.connect = </Cube/UnrealMaterial/UnrealShader.outputs:out>

        def Shader "UnrealShader"
        {{
            uniform token info:implementationSource = "sourceAsset"
            uniform asset info:unreal:sourceAsset = @{grid}@
            token outputs:out
        }}
    }}
}}
"""

UE_LEVEL = """#usda 1.0
(
    defaultPrim = "Root"
    metersPerUnit = 0.01
    upAxis = "Z"
)

def Xform "Root" (
    assetInfo = {{
        asset identifier = @{identifier}@
        string name = "L_Main"
    }}
)
{{
    def "UE6Cube" (
        prepend apiSchemas = ["MaterialBindingAPI"]
        prepend references = @Cube.usda@
    )
    {{
        over "UnrealMaterial"
        {{
            over "UnrealShader"
            {{
                uniform asset info:unreal:sourceAsset = @{basic}@
            }}
        }}
    }}
}}
"""


def _export(directory: Path, identifier_dir: Path) -> Path:
    """Write an Unreal-style export into ``directory``; ``assetInfo`` names ``identifier_dir``."""
    (directory / "Cube.usda").write_text(UE_CUBE.format(identifier=identifier_dir / "Cube.usda", grid=GRID))
    level = directory / "three.usda"
    level.write_text(UE_LEVEL.format(identifier=identifier_dir / "three.usda", basic=BASIC))
    return level


def _layer(directory: Path, body: str, name: str = "scene.usda") -> Path:
    path = directory / name
    path.write_text("#usda 1.0\n" + body)
    return path


@pytest.fixture
def dirs(tmp_path: Path) -> Tuple[Path, Path]:
    inside = (tmp_path / "inside").resolve()
    outside = (tmp_path / "outside").resolve()
    inside.mkdir()
    outside.mkdir()
    return inside, outside


@pytest.fixture
def policy(dirs: Tuple[Path, Path]) -> PathPolicy:
    return PathPolicy(enabled=True, allowed_paths=[str(dirs[0])])


@pytest.fixture
def session() -> HeadlessUSDSession:
    usd_session = HeadlessUSDSession()
    yield usd_session
    usd_session.cleanup()


def _denied(session: HeadlessUSDSession, path: Path, policy: PathPolicy) -> List[Dict[str, str]]:
    with pytest.raises(SandboxDenied) as caught:
        session.load_stage_or_raise(path, sandbox=CompositionSandbox(policy))
    return caught.value.details["outside_sandbox"]


@pytest.mark.parametrize(
    "path",
    [BASIC, GRID, "/Game/Props/SM_Chair.SM_Chair", "/Script/Engine.Engine", "/Game/My Folder/M_A-1.M_A-1"],
)
def test_unreal_object_paths_are_content(path: str) -> None:
    assert is_unreal_content_path(path)


@pytest.mark.parametrize(
    "path",
    [
        "/Engine/x.usda",  # a file name under a content root's name
        "/Engine/usda.usda",  # Name.Name, but the suffix is a file extension
        "/Engine/BasicShapes/Cube.Sphere",  # not a package and its own asset
        "/MyPlugin/Materials/M_Wood.M_Wood",  # only /Engine, /Game and /Script
        "Engine/BasicShapes/Cube.Cube",  # relative
        "../../Engine/BasicShapes/Cube.Cube",
    ],
)
def test_other_shapes_are_not_content(path: str) -> None:
    assert not is_unreal_content_path(path)


def test_a_content_path_that_names_a_file_is_a_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(usd_sandbox.os.path, "lexists", lambda path: path == BASIC)

    assert not is_unreal_content_path(BASIC)
    assert is_unreal_content_path(GRID)


def test_an_unreal_export_inside_the_sandbox_loads_and_lists_its_content_paths(
    session: HeadlessUSDSession, dirs: Tuple[Path, Path], policy: PathPolicy
) -> None:
    inside, _ = dirs
    level = _export(inside, inside)

    stage_id = session.load_stage_or_raise(level, sandbox=CompositionSandbox(policy))

    stage = session.get_stage(stage_id)
    assert stage.GetPrimAtPath("/Root/UE6Cube/UnrealMaterial/UnrealShader")
    listed = {(entry.kind, entry.asset_path, Path(entry.layer.split(":SDF_FORMAT_ARGS:")[0]).name)
              for entry in session.unreal_content_paths(stage_id)}
    assert listed == {("asset", BASIC, "three.usda"), ("asset", GRID, "Cube.usda")}


def test_a_content_path_is_listed_again_after_a_reload(
    session: HeadlessUSDSession, dirs: Tuple[Path, Path], policy: PathPolicy
) -> None:
    inside, _ = dirs
    level = _export(inside, inside)
    session.set_sandbox_factory(lambda: CompositionSandbox(policy))
    stage_id = session.load_stage_or_raise(level, sandbox=CompositionSandbox(policy))

    session.reload_stage(stage_id)

    assert {entry.asset_path for entry in session.unreal_content_paths(stage_id)} == {BASIC, GRID}


def test_a_copied_export_is_still_refused_for_its_asset_info_identifier(
    session: HeadlessUSDSession, dirs: Tuple[Path, Path], policy: PathPolicy
) -> None:
    """assetInfo.identifier stays held to the sandbox, now labelled ``asset`` rather than ``clip``."""
    inside, outside = dirs
    level = _export(inside, outside)

    listed = _denied(session, level, policy)

    assert {(entry["kind"], entry["asset_path"]) for entry in listed} == {
        ("asset", str(outside / "three.usda")),
        ("asset", str(outside / "Cube.usda")),
    }


@pytest.mark.parametrize(
    "body, kind",
    [
        (f'(\n    subLayers = [@{BASIC}@]\n)\n', "sublayer"),
        (f'def "R" (\n    references = @{BASIC}@\n) {{}}\n', "reference"),
        (f'def "R" (\n    payload = @{BASIC}@\n) {{}}\n', "payload"),
        (
            'def "R" (\n    clips = {\n        dictionary c = {\n'
            f'            asset[] assetPaths = [@{BASIC}@]\n            string primPath = "/M"\n'
            "        }\n    }\n) {}\n",
            "clip",
        ),
        (
            'def "R" (\n    variants = { string v = "a" }\n    prepend variantSets = "v"\n) {\n'
            '    variantSet "v" = {\n        "a" (\n            clips = {\n                dictionary c = {\n'
            f"                    asset manifestAssetPath = @{BASIC}@\n"
            '                    string primPath = "/M"\n                }\n            }\n'
            "        ) {}\n    }\n}\n",
            "clip",
        ),
    ],
)
def test_a_content_path_as_a_composition_arc_or_clip_is_still_refused(
    session: HeadlessUSDSession, dirs: Tuple[Path, Path], policy: PathPolicy, body: str, kind: str
) -> None:
    """Only asset values pass: a sublayer, reference, payload or clip is a file read whatever its shape."""
    inside, _ = dirs
    path = _layer(inside, body)

    listed = _denied(session, path, policy)

    assert [(entry["kind"], entry["asset_path"]) for entry in listed] == [(kind, BASIC)]


def test_a_template_clip_set_keeps_its_expanded_paths_clips(
    session: HeadlessUSDSession, dirs: Tuple[Path, Path], policy: PathPolicy
) -> None:
    inside, outside = dirs
    for frame in (1, 2):
        _layer(outside, 'def "M" {\n    double x.timeSamples = {1: 1}\n}\n', f"clip.00{frame}.usda")
    path = _layer(
        inside,
        'def "R" (\n    clips = {\n        dictionary c = {\n'
        f'            string templateAssetPath = "{outside}/clip.###.usda"\n'
        "            double templateStartTime = 1\n            double templateEndTime = 2\n"
        '            double templateStride = 1\n            string primPath = "/M"\n'
        "        }\n    }\n) {}\n",
    )

    listed = _denied(session, path, policy)

    assert {entry["kind"] for entry in listed} == {"clip"}
    assert {entry["asset_path"] for entry in listed} >= {f"{outside}/clip.001.usda", f"{outside}/clip.002.usda"}


def test_a_file_outside_beside_content_paths_is_still_refused(
    session: HeadlessUSDSession, dirs: Tuple[Path, Path], policy: PathPolicy
) -> None:
    inside, outside = dirs
    level = _export(inside, inside)
    texture = outside / "secret.png"
    texture.write_bytes(b"png")
    cube = inside / "Cube.usda"
    cube.write_text(
        cube.read_text().replace(
            "            token outputs:out\n",
            f"            token outputs:out\n            asset inputs:file = @{texture}@\n",
        )
    )

    listed = _denied(session, level, policy)

    assert [(entry["kind"], entry["asset_path"]) for entry in listed] == [("asset", str(texture))]


def _issues(report: Dict[str, Any], code: str) -> List[Dict[str, Any]]:
    return [issue for issue in report["issues"] if issue["code"] == code]


@pytest.mark.parametrize("sandboxed", [True, False])
def test_validation_downgrades_pxr_findings_about_content_paths_only(
    dirs: Tuple[Path, Path], policy: PathPolicy, sandboxed: bool
) -> None:
    inside, _ = dirs
    level = _export(inside, inside)
    cube = inside / "Cube.usda"
    cube.write_text(
        cube.read_text().replace(
            "            token outputs:out\n",
            "            token outputs:out\n            asset inputs:file = @./missing.png@\n",
        )
    )
    sandbox = CompositionSandbox(policy) if sandboxed else None  # None: the CLI, no sandbox

    report = validate_usd_file(str(level), Settings().usd, sandbox=sandbox).to_dict()

    unresolved = {issue["location"]["asset_path"]: issue for issue in _issues(report, "UnresolvableDependency")}
    for content in (BASIC, GRID):
        assert unresolved[content]["severity"] == "warning"
        assert "Unreal content path" in unresolved[content]["message"]
    missing_texture = [issue for path, issue in unresolved.items() if path.endswith("missing.png")]
    assert [issue["severity"] for issue in missing_texture] == ["error"]
    source_type = _issues(report, "MissingSourceTypeInRegistry")
    assert source_type and {issue["severity"] for issue in source_type} == {"warning"}
    assert report["is_valid"] is False  # the missing texture alone


def test_an_unreal_export_validates_clean_of_errors(dirs: Tuple[Path, Path], policy: PathPolicy) -> None:
    inside, _ = dirs
    level = _export(inside, inside)

    report = validate_usd_file(str(level), Settings().usd, sandbox=CompositionSandbox(policy)).to_dict()

    assert report["is_valid"] is True, [i for i in report["issues"] if i["severity"] == "error"]


def test_a_source_type_error_on_another_asset_stays_an_error(dirs: Tuple[Path, Path], policy: PathPolicy) -> None:
    """Only the shader whose sourceAsset is a content path is downgraded."""
    inside, _ = dirs
    path = _layer(
        inside,
        'def Shader "S"\n{\n    uniform token info:implementationSource = "sourceAsset"\n'
        "    uniform asset info:foo:sourceAsset = @./shader.foo@\n}\n",
    )
    (inside / "shader.foo").write_text("x")

    report = validate_usd_file(str(path), Settings().usd, sandbox=CompositionSandbox(policy)).to_dict()

    assert {issue["severity"] for issue in _issues(report, "MissingSourceTypeInRegistry")} == {"error"}


def _server(root: Path) -> SimulMCPServer:
    settings = Settings()
    security = settings.security.model_copy(update={"allowed_paths": [str(root)], "rate_limiting_enabled": False})
    return SimulMCPServer(settings.model_copy(update={"security": security}), backends={"usd"})


def _call(server: SimulMCPServer, name: str, **arguments: Any) -> Dict[str, Any]:
    async def run() -> Dict[str, Any]:
        async with Client(server.mcp) as client:
            result = await client.call_tool(name, arguments, raise_on_error=False)
            return json.loads(result.content[0].text)

    return asyncio.run(run())


def test_load_usd_file_reports_the_content_paths_as_warnings(dirs: Tuple[Path, Path]) -> None:
    inside, _ = dirs
    level = _export(inside, inside)

    payload = _call(_server(inside), "load_usd_file", file_path=str(level))

    assert payload["success"] is True, payload
    content = [issue for issue in payload["warnings"] if issue["code"] == "UnrealContentPath"]
    assert {issue["location"]["asset_path"]: Path(issue["location"]["layer"]).name for issue in content} == {
        BASIC: "three.usda",
        GRID: "Cube.usda",
    }
    assert all(issue["location"]["layer"].startswith(str(inside)) for issue in content)
    assert all("SDF_FORMAT_ARGS" not in json.dumps(issue) for issue in content)


def test_validate_usd_file_passes_an_unreal_export(dirs: Tuple[Path, Path]) -> None:
    inside, _ = dirs
    level = _export(inside, inside)

    payload = _call(_server(inside), "validate_usd_file", file_path=str(level))

    assert payload["success"] is True, payload
    assert payload["is_valid"] is True, payload
