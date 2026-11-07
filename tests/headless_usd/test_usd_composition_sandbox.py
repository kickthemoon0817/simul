"""Every layer and asset a USD stage draws on is held to the file sandbox.

Only the root file used to be checked, so a layer inside the sandbox that
referenced ``../outside/secret.usda`` or sublayered an absolute path loaded the
outside file, and ``get_prim_info`` returned its attributes. These build real
pxr layers in an ``inside`` directory (the sandbox) and an ``outside`` one.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

pytest.importorskip("pxr", reason="pxr library not available")

from pxr import Sdf, Usd, UsdGeom, UsdShade  # noqa: E402

from simul.adapters.headless_usd import HeadlessUSDSession  # noqa: E402
from simul.config import Settings  # noqa: E402
from simul.usd.sandbox import CompositionSandbox  # noqa: E402
from simul.usd.validation import validate_usd_file  # noqa: E402
from simul.utils.paths import PathPolicy, SandboxDenied  # noqa: E402

SECRET = "sk-OUTSIDE-SANDBOX-SECRET"


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


def _secret(directory: Path, name: str = "secret.usda") -> Path:
    path = directory / name
    stage = Usd.Stage.CreateNew(str(path))
    prim = stage.DefinePrim("/Secret", "Xform")
    stage.SetDefaultPrim(prim)
    prim.CreateAttribute("custom:api_key", Sdf.ValueTypeNames.String).Set(SECRET)
    stage.GetRootLayer().Save()
    return path


def _stage(path: Path) -> Usd.Stage:
    stage = Usd.Stage.CreateNew(str(path))
    world = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(world)
    return stage


def _load(session: HeadlessUSDSession, path: Path, policy: PathPolicy) -> str:
    return session.load_stage_or_raise(path, sandbox=CompositionSandbox(policy))


def _denied(session: HeadlessUSDSession, path: Path, policy: PathPolicy) -> Dict:
    with pytest.raises(SandboxDenied) as caught:
        _load(session, path, policy)
    details = caught.value.details
    assert caught.value.error_message.startswith(f"USD file {path} draws on")
    assert details["file_path"] == str(path)
    assert details["outside_count"] == len(details["outside_sandbox"])
    assert "security.allowed_paths" in details["hint"]
    # A refused stage is neither registered nor left in the reader's cache.
    assert session.list_stages() == []
    assert str(path) not in session.usd_reader._stage_cache
    return details


def _kinds(details: Dict) -> List[Tuple[str, str]]:
    return [(entry["kind"], entry["location"]) for entry in details["outside_sandbox"]]


class TestArcsLeavingTheSandbox:
    def test_relative_reference_to_outside(self, dirs, policy, session) -> None:
        inside, outside = dirs
        secret = _secret(outside)
        path = inside / "ref_outside.usda"
        stage = _stage(path)
        stage.DefinePrim("/World/Leak").GetReferences().AddReference("../outside/secret.usda")
        stage.GetRootLayer().Save()

        details = _denied(session, path, policy)

        assert _kinds(details) == [("reference", str(secret))]
        assert details["outside_sandbox"][0]["asset_path"] == "../outside/secret.usda"
        assert details["outside_sandbox"][0]["layer"] == str(path)

    def test_absolute_sublayer(self, dirs, policy, session) -> None:
        inside, outside = dirs
        secret = _secret(outside)
        path = inside / "sublayer_outside.usda"
        layer = Sdf.Layer.CreateNew(str(path))
        layer.subLayerPaths.append(str(secret))
        layer.Save()

        assert _kinds(_denied(session, path, policy)) == [("sublayer", str(secret))]

    def test_payload(self, dirs, policy, session) -> None:
        inside, outside = dirs
        secret = _secret(outside)
        path = inside / "payload.usda"
        stage = _stage(path)
        stage.DefinePrim("/World/P").GetPayloads().AddPayload(str(secret))
        stage.GetRootLayer().Save()

        assert _kinds(_denied(session, path, policy)) == [("payload", str(secret))]

    def test_reference_in_an_unselected_variant(self, dirs, policy, session) -> None:
        """A variant switch must not be able to compose an outside file later."""
        inside, outside = dirs
        secret = _secret(outside)
        path = inside / "variant.usda"
        stage = _stage(path)
        world = stage.GetPrimAtPath("/World")
        variants = world.GetVariantSets().AddVariantSet("v")
        variants.AddVariant("safe")
        variants.AddVariant("leak")
        variants.SetVariantSelection("leak")
        with variants.GetVariantEditContext():
            stage.DefinePrim("/World/Leak").GetReferences().AddReference(str(secret))
        variants.SetVariantSelection("safe")
        stage.GetRootLayer().Save()

        assert _kinds(_denied(session, path, policy)) == [("reference", str(secret))]

    def test_value_clip(self, dirs, policy, session) -> None:
        inside, outside = dirs
        clip = _secret(outside, "clip.usda")
        path = inside / "clips.usda"
        stage = _stage(path)
        clips = Usd.ClipsAPI(stage.DefinePrim("/World/Clipped", "Xform"))
        clips.SetClipAssetPaths([Sdf.AssetPath(str(clip))])
        clips.SetClipPrimPath("/Secret")
        clips.SetClipActive([(0, 0)])
        clips.SetClipTimes([(0, 0)])
        stage.GetRootLayer().Save()

        assert str(clip) in [location for _, location in _kinds(_denied(session, path, policy))]

    def test_texture_asset(self, dirs, policy, session) -> None:
        inside, outside = dirs
        texture = outside / "wood.png"
        texture.write_bytes(b"\x89PNG")
        path = inside / "material.usda"
        stage = _stage(path)
        shader = UsdShade.Shader.Define(stage, "/World/Tex")
        shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(texture)))
        stage.GetRootLayer().Save()

        assert _kinds(_denied(session, path, policy)) == [("asset", str(texture))]

    def test_chain_reports_the_layer_that_leaves(self, dirs, policy, session) -> None:
        inside, outside = dirs
        secret = _secret(outside)
        middle = inside / "middle.usda"
        mid = _stage(middle)
        mid.GetPrimAtPath("/World").GetReferences().AddReference(str(secret))
        mid.GetRootLayer().Save()
        path = inside / "top.usda"
        top = _stage(path)
        top.DefinePrim("/World/Mid").GetReferences().AddReference("./middle.usda")
        top.GetRootLayer().Save()

        details = _denied(session, path, policy)

        assert _kinds(details) == [("reference", str(secret))]
        assert details["outside_sandbox"][0]["layer"] == str(middle)

    def test_symlink_inside_pointing_outside(self, dirs, policy, session) -> None:
        inside, outside = dirs
        secret = _secret(outside)
        link = inside / "link.usda"
        try:
            os.symlink(secret, link)
        except OSError:
            pytest.skip("symlinks unavailable")
        path = inside / "via_link.usda"
        stage = _stage(path)
        stage.DefinePrim("/World/L").GetReferences().AddReference("./link.usda")
        stage.GetRootLayer().Save()

        assert _kinds(_denied(session, path, policy)) == [("reference", str(link))]

    def test_search_path_found_in_the_working_directory(
        self, dirs, policy, session, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """pxr tries the working directory for a search path; that location is checked."""
        inside, outside = dirs
        secret = _secret(outside, "cwd_asset.usda")
        monkeypatch.chdir(outside)
        path = inside / "search.usda"
        stage = _stage(path)
        stage.DefinePrim("/World/S").GetReferences().AddReference("cwd_asset.usda")
        stage.GetRootLayer().Save()

        assert _kinds(_denied(session, path, policy)) == [("reference", str(secret))]

    def test_missing_outside_reference_is_refused_too(self, dirs, policy, session) -> None:
        """Refused whether or not it exists, so a refusal reveals nothing about outside files."""
        inside, outside = dirs
        path = inside / "missing_outside.usda"
        stage = _stage(path)
        stage.DefinePrim("/World/M").GetReferences().AddReference(str(outside / "nope.usda"))
        stage.GetRootLayer().Save()

        assert _kinds(_denied(session, path, policy)) == [("reference", str(outside / "nope.usda"))]

    def test_url_with_a_scheme_the_sandbox_refuses(self, dirs, policy, session) -> None:
        path = dirs[0] / "http.usda"
        stage = _stage(path)
        stage.DefinePrim("/World/H").GetReferences().AddReference("http://example.com/a.usda")
        stage.GetRootLayer().Save()

        assert _kinds(_denied(session, path, policy)) == [
            ("reference", "http://example.com/a.usda")
        ]

    def test_absolute_reference_inside_a_usdz(self, dirs, policy, session) -> None:
        inside, outside = dirs
        secret = _secret(outside)
        layer_path = inside / "pkg_root.usda"
        stage = _stage(layer_path)
        stage.DefinePrim("/World/L").GetReferences().AddReference(str(secret))
        stage.GetRootLayer().Save()
        package = inside / "pkg.usdz"
        with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.write(layer_path, arcname="pkg_root.usda")
        layer_path.unlink()

        assert _kinds(_denied(session, package, policy)) == [("reference", str(secret))]


class TestWhatStillLoads:
    def test_references_inside_the_sandbox(self, dirs, policy, session) -> None:
        inside, _ = dirs
        _secret(inside, "asset.usda")
        path = inside / "ok.usda"
        stage = _stage(path)
        stage.DefinePrim("/World/A").GetReferences().AddReference("./asset.usda")
        stage.GetRootLayer().Save()

        stage_id = _load(session, path, policy)

        assert session.get_prim_info(stage_id, "/World/A").attributes["custom:api_key"] == SECRET

    def test_missing_inside_reference_loads(self, dirs, policy, session) -> None:
        path = dirs[0] / "missing_inside.usda"
        stage = _stage(path)
        stage.DefinePrim("/World/M").GetReferences().AddReference("./nope.usda")
        stage.GetRootLayer().Save()

        assert _load(session, path, policy)

    def test_unanchored_search_path_names_no_location(self, dirs, policy, session) -> None:
        path = dirs[0] / "mdl.usda"
        stage = _stage(path)
        shader = UsdShade.Shader.Define(stage, "/World/Mdl")
        shader.CreateInput("mdl", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("OmniPBR.mdl"))
        stage.GetRootLayer().Save()

        assert _load(session, path, policy)

    def test_allowed_url_scheme(self, dirs, policy, session) -> None:
        path = dirs[0] / "nucleus.usda"
        stage = _stage(path)
        stage.DefinePrim("/World/N").GetReferences().AddReference(
            "omniverse://localhost/Projects/robot.usd"
        )
        stage.GetRootLayer().Save()

        assert _load(session, path, policy)

    def test_disabled_sandbox_checks_nothing(self, dirs, session) -> None:
        inside, outside = dirs
        _secret(outside)
        path = inside / "ref_outside.usda"
        stage = _stage(path)
        stage.DefinePrim("/World/Leak").GetReferences().AddReference("../outside/secret.usda")
        stage.GetRootLayer().Save()

        stage_id = _load(session, path, PathPolicy(enabled=False, allowed_paths=[]))
        assert session.get_prim_info(stage_id, "/World/Leak").attributes["custom:api_key"] == SECRET

    def test_no_sandbox_argument_keeps_the_cli_unrestricted(self, dirs, session) -> None:
        inside, outside = dirs
        _secret(outside)
        path = inside / "ref_outside.usda"
        stage = _stage(path)
        stage.DefinePrim("/World/Leak").GetReferences().AddReference("../outside/secret.usda")
        stage.GetRootLayer().Save()

        assert session.load_stage_or_raise(path)


def test_post_composition_check_catches_what_the_walk_did_not_see(dirs, policy) -> None:
    """``check_stage`` is the backstop: it inspects the layers the stage really used."""
    inside, outside = dirs
    secret = _secret(outside)
    path = inside / "ref_outside.usda"
    stage = _stage(path)
    stage.DefinePrim("/World/Leak").GetReferences().AddReference(str(secret))
    stage.GetRootLayer().Save()
    composed = Usd.Stage.Open(str(path))  # composed without the first pass

    with pytest.raises(SandboxDenied) as caught:
        CompositionSandbox(policy).check_stage(composed)

    assert str(secret) in [entry["location"] for entry in caught.value.details["outside_sandbox"]]


def test_validation_applies_the_same_sandbox(dirs, policy) -> None:
    inside, outside = dirs
    _secret(outside)
    path = inside / "ref_outside.usda"
    stage = _stage(path)
    stage.DefinePrim("/World/Leak").GetReferences().AddReference("../outside/secret.usda")
    stage.GetRootLayer().Save()

    with pytest.raises(SandboxDenied):
        validate_usd_file(str(path), Settings().usd, sandbox=CompositionSandbox(policy))
    # Without a sandbox (the CLI) the file validates; the reference resolves.
    assert validate_usd_file(str(path), Settings().usd).can_read is True


def test_denial_lists_at_most_25_paths(dirs, policy, session) -> None:
    inside, outside = dirs
    path = inside / "many.usda"
    stage = _stage(path)
    for index in range(30):
        prim = stage.DefinePrim(f"/World/R{index}")
        prim.GetReferences().AddReference(str(outside / f"missing_{index}.usda"))
    UsdGeom.SetStageUpAxis(stage, "Z")
    stage.GetRootLayer().Save()

    details = _denied_many(session, path, policy)

    assert details["outside_count"] == 30
    assert len(details["outside_sandbox"]) == 25


def _denied_many(session: HeadlessUSDSession, path: Path, policy: PathPolicy) -> Dict:
    with pytest.raises(SandboxDenied) as caught:
        _load(session, path, policy)
    assert "and 27 more" in caught.value.error_message
    return caught.value.details
