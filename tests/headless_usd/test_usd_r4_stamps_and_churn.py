"""USD round 4: a file written while pxr reads it is never vouched for, and session churn does not fail validation.

- B1: each file is stamped before pxr reads it (the root before it is
  opened, each layer as the walk opens it, each reread and reload before it
  runs). A write that lands while a load reads a file used to be stamped as
  already read, so the stage kept the old content with nothing listed in
  ``changed_on_disk``, and a later ``save_usd_stage`` overwrote the other
  program's write. Each test makes the write land at one exact point of the
  load by wrapping the step pxr reads in.
- M1: ``Sdf.Layer.GetLoadedLayers`` raises while another thread releases a
  layer; the listing is asked for again, and validation falls back to the
  fresh process.
- M2: an instanceable asset animated by value clips reports its samples.
- m1 to m3: ``ValidatorFailed`` reasons are capped, a layer path with spaces
  is located, and a reread in place clears ``changed_on_disk`` on every stage
  showing it.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import pytest

pytest.importorskip("pxr", reason="pxr library not available")

from pxr import Sdf, Tf, Usd, UsdGeom  # noqa: E402

from simul.adapters import headless_usd  # noqa: E402
from simul.adapters.headless_usd import HeadlessUSDSession  # noqa: E402
from simul.config import Settings  # noqa: E402
from simul.usd import pxr_errors, sandbox as sandbox_module, stage_files, validation  # noqa: E402
from simul.usd.editing import USDEditError  # noqa: E402
from simul.usd.reader import animation_range  # noqa: E402
from simul.usd.sandbox import CompositionSandbox  # noqa: E402
from simul.utils.paths import PathPolicy  # noqa: E402

USD = Settings().usd

_ROOT_OLD = '#usda 1.0\ndef Xform "World"\n{\n    def Cube "Marker"\n    {\n        double size = 1\n    }\n}\n'
_ROOT_NEW = (
    '#usda 1.0\ndef Xform "World"\n{\n    custom string note = "written by another program"\n'
    '    def Cube "Marker"\n    {\n        double size = 2\n    }\n}\n'
)
_SUB = '#usda 1.0\ndef Xform "World"\n{\n    def Cube "FromSub"\n    {\n        double size = %d\n    }\n}\n'
# Size 2, and a negative mass that validation must report.
_SUB_BAD = (
    '#usda 1.0\ndef Xform "World"\n{\n    def Cube "FromSub" (\n'
    '        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]\n    )\n'
    "    {\n        double size = 2\n        float physics:mass = -1\n    }\n}\n"
)


def _external_write(path: Path, text: str) -> None:
    """Replace ``path`` as another program saves it: a new file moved into place, with a later mtime."""
    staged = path.with_name(path.name + ".partial")
    staged.write_text(text)
    os.replace(staged, path)
    info = os.stat(path)
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 2_000_000_000))


def _age(path: Path, seconds: float = 3600.0) -> None:
    """Set ``path``'s modification time well before now."""
    then = time.time() - seconds
    os.utime(path, (then, then))


def _shared_sublayer(directory: Path) -> Dict[str, Path]:
    """a.usda and b.usda, each sublayering sub.usda (a Cube of size 1)."""
    (directory / "sub.usda").write_text(_SUB % 1)
    for name in ("a", "b"):
        (directory / f"{name}.usda").write_text("#usda 1.0\n(\n    subLayers = [@./sub.usda@]\n)\n")
    return {name: directory / f"{name}.usda" for name in ("a", "b", "sub")}


def _size(session: HeadlessUSDSession, stage_id: str, prim: str = "/World/FromSub") -> float:
    return session.get_stage(stage_id).GetPrimAtPath(prim).GetAttribute("size").Get()


def _changed(session: HeadlessUSDSession, stage_id: str) -> List[str]:
    return session.stage_status(stage_id)["changed_on_disk"]


def _policy(directory: Path) -> PathPolicy:
    return PathPolicy(enabled=True, allowed_paths=[str(directory)])


class _CheckHook(CompositionSandbox):
    """A sandbox that runs ``hook`` when it checks a composed stage, once.

    ``check_stage`` runs after pxr read the files and before the session
    records their stamps, so the hook is a write landing in that window.
    """

    def __init__(self, policy: PathPolicy, hook: Callable[[], None], fired: List[bool]) -> None:
        super().__init__(policy)
        self._hook = hook
        self._fired = fired

    def check_stage(self, stage: Any) -> None:
        if not self._fired:
            self._fired.append(True)
            self._hook()
        super().check_stage(stage)


@pytest.fixture
def session() -> Iterator[HeadlessUSDSession]:
    usd_session = HeadlessUSDSession()
    usd_session.set_owner_resolver(lambda: "session-a")
    yield usd_session
    usd_session.cleanup()


def _once(write: Callable[[], None]) -> Callable[[], None]:
    done: List[bool] = []

    def run() -> None:
        if not done:
            done.append(True)
            write()

    return run


class TestRootWrittenDuringTheLoad:
    """B1: the root file is stamped before ``open_private_root`` reads it."""

    @pytest.fixture
    def root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        path = tmp_path / "root.usda"
        path.write_text(_ROOT_OLD)
        opened = stage_files.open_private_root
        write = _once(lambda: _external_write(path, _ROOT_NEW))

        def open_then_written(identifier: str) -> Any:
            layer = opened(identifier)  # pxr has read the file
            write()
            return layer

        monkeypatch.setattr(stage_files, "open_private_root", open_then_written)
        return path

    def test_the_stage_lists_its_root_as_changed(self, session, root) -> None:
        stage_id = session.load_stage_or_raise(root)

        assert _size(session, stage_id, "/World/Marker") == 1.0
        assert _changed(session, stage_id) == [str(root.resolve())]
        again = session.load_stage_or_raise(root)
        assert again != stage_id
        assert _size(session, again, "/World/Marker") == 2.0
        assert _changed(session, again) == []

    def test_a_save_does_not_overwrite_the_other_programs_write(self, session, root) -> None:
        stage_id = session.load_stage_or_raise(root)
        session.update_prim_attributes(stage_id, "/World/Marker", {"size": 5.0})

        with pytest.raises(USDEditError) as caught:
            session.save_stage(stage_id)

        assert caught.value.error_type == "RefusedOperation"
        assert caught.value.details["override"] == "overwrite_external_changes"
        assert "written by another program" in root.read_text()
        assert session.save_stage(stage_id, overwrite_external_changes=True)["saved"] is True
        assert "written by another program" not in root.read_text()


class TestSharedLayerWrittenDuringTheLoad:
    """B1: a sublayer is stamped as the walk opens it, before pxr reads it."""

    @pytest.mark.parametrize("sandboxed", [False, True], ids=["no-sandbox", "sandbox"])
    def test_a_write_after_the_read_is_reread_before_the_load_returns(
        self, session, tmp_path, monkeypatch, sandboxed
    ) -> None:
        paths = _shared_sublayer(tmp_path)
        compose = stage_files.compose_stage
        write = _once(lambda: _external_write(paths["sub"], _SUB_BAD))

        def compose_then_written(root: Any) -> Any:
            stage = compose(root)  # the walk read sub.usda before this
            write()
            return stage

        monkeypatch.setattr(stage_files, "compose_stage", compose_then_written)
        sandbox = CompositionSandbox(_policy(tmp_path)) if sandboxed else None
        stage_id = session.load_stage_or_raise(paths["a"], sandbox=sandbox)

        assert _size(session, stage_id) == 2.0
        assert _changed(session, stage_id) == []
        assert session.layer_is_current(str(paths["sub"].resolve())) is True

    def test_a_write_after_the_check_stays_changed_and_fails_validation(self, session, tmp_path) -> None:
        paths = _shared_sublayer(tmp_path)
        sub = str(paths["sub"].resolve())
        hooked = _CheckHook(_policy(tmp_path), lambda: _external_write(paths["sub"], _SUB_BAD), [])

        stage_id = session.load_stage_or_raise(paths["a"], sandbox=hooked)

        assert _size(session, stage_id) == 1.0
        assert _changed(session, stage_id) == [sub]
        assert session.layer_is_current(sub) is False
        report = validation.validate_usd_file(
            str(paths["a"].resolve()), USD, layer_is_current=session.layer_is_current
        )
        assert report.is_valid is False
        assert "MassInvalidValues" in [issue.code for issue in report.issues]
        again = session.load_stage_or_raise(paths["b"])
        assert _size(session, again) == 2.0
        assert _size(session, stage_id) == 2.0  # reread in place for every stage
        assert _changed(session, stage_id) == []

    def test_a_layer_composition_read_unrecorded_is_reread_when_recent(
        self, session, tmp_path, monkeypatch
    ) -> None:
        paths = _shared_sublayer(tmp_path)
        # No walk opens anything: composition reads sub.usda itself.
        monkeypatch.setattr(headless_usd, "open_arc_layers", lambda root, reads: [])
        compose = stage_files.compose_stage
        write = _once(lambda: _external_write(paths["sub"], _SUB_BAD))

        def compose_then_written(root: Any) -> Any:
            stage = compose(root)
            write()
            return stage

        monkeypatch.setattr(stage_files, "compose_stage", compose_then_written)
        stage_id = session.load_stage_or_raise(paths["a"])

        assert _size(session, stage_id) == 2.0
        assert _changed(session, stage_id) == []

    def test_an_old_file_composition_read_unrecorded_is_not_reread(self, session, tmp_path, monkeypatch) -> None:
        paths = _shared_sublayer(tmp_path)
        _age(paths["sub"])
        monkeypatch.setattr(headless_usd, "open_arc_layers", lambda root, reads: [])
        reread: List[Any] = []
        original = session._reread
        monkeypatch.setattr(session, "_reread", lambda layers, stage: reread.extend(layers) or original(layers, stage))

        stage_id = session.load_stage_or_raise(paths["a"])

        assert reread == []
        assert _changed(session, stage_id) == []


class TestRereadsAndReloads:
    """B1: a reread or a reload stamps the file before pxr reads it again."""

    def test_a_root_written_while_reload_reads_it_stays_changed(self, session, tmp_path) -> None:
        root = tmp_path / "root.usda"
        root.write_text(_ROOT_OLD)
        stage_id = session.load_stage_or_raise(root)
        fired: List[bool] = []
        policy = _policy(tmp_path)
        # The reload's check of the composed stage runs after root.Reload read the file.
        session.set_sandbox_factory(lambda: _CheckHook(policy, lambda: _external_write(root, _ROOT_NEW), fired))

        session.reload_stage(stage_id)

        assert fired == [True]
        assert _size(session, stage_id, "/World/Marker") == 1.0
        assert _changed(session, stage_id) == [str(root.resolve())]
        session.update_prim_attributes(stage_id, "/World/Marker", {"size": 5.0})
        with pytest.raises(USDEditError):
            session.save_stage(stage_id)
        assert "written by another program" in root.read_text()

    def test_a_file_written_while_it_is_reread_stays_changed(self, session, tmp_path) -> None:
        paths = _shared_sublayer(tmp_path)
        sub = str(paths["sub"].resolve())
        first = session.load_stage_or_raise(paths["a"])
        _external_write(paths["sub"], _SUB % 2)
        fired: List[bool] = []
        policy = _policy(tmp_path)
        # After a reread in place, every other stage using the layer is checked again.
        session.set_sandbox_factory(lambda: _CheckHook(policy, lambda: _external_write(paths["sub"], _SUB % 3), fired))

        second = session.load_stage_or_raise(paths["b"])

        assert fired == [True]
        assert (_size(session, first), _size(session, second)) == (2.0, 2.0)
        assert _changed(session, first) == [sub]
        assert _changed(session, second) == [sub]
        assert session.layer_is_current(sub) is False
        assert _size(session, session.load_stage_or_raise(paths["a"])) == 3.0

    def test_a_reread_that_fails_records_the_stamp_from_before_it(self, session, tmp_path, monkeypatch) -> None:
        paths = _shared_sublayer(tmp_path)
        sub = str(paths["sub"].resolve())
        session.load_stage_or_raise(paths["a"])
        _external_write(paths["sub"], "#usda 1.0\ndef Xform \"World\" {\n")  # pxr cannot parse it
        candidate = stage_files.open_candidate
        fixed = _once(lambda: _external_write(paths["sub"], _SUB % 4))

        def fails_then_fixed(layer: Any) -> Any:
            try:
                return candidate(layer)
            finally:
                fixed()  # a fix saved while pxr was failing to parse the file

        monkeypatch.setattr(stage_files, "open_candidate", fails_then_fixed)
        session.load_stage_or_raise(paths["b"])
        monkeypatch.setattr(stage_files, "open_candidate", candidate)

        assert session._shared_unread[sub][0] != stage_files.stamp(sub)
        assert _size(session, session.load_stage_or_raise(paths["a"])) == 4.0


class TestArcLayersWalk:
    """Without the sandbox's walk, a load still opens what the arcs name before composing."""

    def test_records_a_stamp_for_each_layer_it_reads(self, tmp_path) -> None:
        (tmp_path / "sub.usda").write_text(_SUB % 1)
        (tmp_path / "ref.usda").write_text('#usda 1.0\ndef Xform "R"\n{\n}\n')
        (tmp_path / "var.usda").write_text('#usda 1.0\ndef Xform "V"\n{\n}\n')
        (tmp_path / "root.usda").write_text(
            '#usda 1.0\n(\n    subLayers = [@./sub.usda@]\n)\n'
            'def Xform "W" (\n    references = @./ref.usda@</R>\n'
            '    variants = { string v = "a" }\n    prepend variantSets = "v"\n)\n{\n'
            '    variantSet "v" = {\n        "b" (references = @./var.usda@</V>) {\n        }\n    }\n}\n'
        )
        held = Sdf.Layer.FindOrOpen(str(tmp_path / "ref.usda"))  # pxr holds it: no read, no stamp
        root = Sdf.Layer.FindOrOpen(str(tmp_path / "root.usda"))
        reads: Dict[str, Optional[Any]] = {}

        opened = sandbox_module.open_arc_layers(root, reads)

        assert sorted(Path(layer.realPath).name for layer in opened) == ["ref.usda", "sub.usda", "var.usda"]
        assert sorted(Path(path).name for path in reads) == ["sub.usda", "var.usda"]
        assert reads[str(tmp_path / "sub.usda")] == stage_files.stamp(str(tmp_path / "sub.usda"))
        assert held

    def test_unrecorded_read_stamp(self) -> None:
        now = time.time_ns()
        old = (now - 10_000_000_000, 5, 1)
        recent = (now - 1_000_000_000, 5, 1)

        assert headless_usd._unrecorded_read_stamp(old, now) == old
        assert headless_usd._unrecorded_read_stamp(recent, now) is None
        assert headless_usd._unrecorded_read_stamp(None, now) is None
        assert headless_usd._unrecorded_read_stamp(recent, None) == recent


_RACE = "Multiple Python objects for C++ object 0x76ffa0790: (Existing python object id 0x0 with type expired)"


class TestLayerListingRace:
    """M1: pxr's layer listing raises while another thread releases a layer."""

    def test_the_listing_is_asked_again(self, monkeypatch) -> None:
        real = Sdf.Layer.GetLoadedLayers
        layer = Sdf.Layer.CreateNew(str(Path(os.environ.get("TMPDIR", "/tmp")) / f"r4-{os.getpid()}.usda"))
        calls: List[int] = []

        def flaky() -> Any:
            calls.append(1)
            if len(calls) < 3:
                raise Tf.ErrorException(_RACE)
            return real()

        monkeypatch.setattr(Sdf.Layer, "GetLoadedLayers", flaky)

        held = stage_files.layers_in_memory()

        assert len(calls) == 3
        assert layer.identifier in held
        os.unlink(layer.realPath)

    def test_a_listing_that_keeps_failing_raises_its_own_error(self, monkeypatch) -> None:
        def broken() -> Any:
            raise Tf.ErrorException(_RACE)

        monkeypatch.setattr(Sdf.Layer, "GetLoadedLayers", broken)

        with pytest.raises(stage_files.LoadedLayersUnavailable):
            stage_files.layers_in_memory()

    @pytest.fixture
    def unlisted(self, monkeypatch) -> None:
        def unavailable() -> Any:
            raise stage_files.LoadedLayersUnavailable(_RACE)

        monkeypatch.setattr(validation, "layers_in_memory", unavailable)
        monkeypatch.setattr(stage_files, "layers_in_memory", unavailable)

    def test_validation_checks_the_disk_in_the_fresh_process(self, tmp_path, monkeypatch, unlisted) -> None:
        paths = _shared_sublayer(tmp_path)
        _external_write(paths["sub"], _SUB_BAD)
        fresh = validation._validate_in_fresh_process
        ran: List[str] = []
        monkeypatch.setattr(
            validation, "_validate_in_fresh_process", lambda *args: ran.append(args[0]) or fresh(*args)
        )

        report = validation.validate_usd_file(str(paths["a"].resolve()), USD, layer_is_current=lambda path: True)

        assert ran == [str(paths["a"].resolve())]
        assert report.is_valid is False
        assert "MassInvalidValues" in [issue.code for issue in report.issues]

    def test_without_a_fresh_process_it_says_what_it_checked(self, tmp_path, monkeypatch, unlisted) -> None:
        paths = _shared_sublayer(tmp_path)

        def cannot_start(file_path: str, usd: Any, sandbox: Any) -> Any:
            report = validation.ValidationReport(file_path=file_path, file_size=0, format="usda")
            report.problem = "it could not start: test"
            return report

        monkeypatch.setattr(validation, "_validate_in_fresh_process", cannot_start)

        report = validation.validate_usd_file(str(paths["a"].resolve()), USD)

        warning = next(issue for issue in report.issues if issue.code == "ValidatedInMemoryCopy")
        assert "whatever layers pxr holds" in warning.message

    def test_a_load_rereads_the_layers_it_cannot_vouch_for(self, session, tmp_path, unlisted) -> None:
        paths = _shared_sublayer(tmp_path)
        held = Sdf.Layer.FindOrOpen(str(paths["sub"]))  # in memory, stamped by nobody
        _external_write(paths["sub"], _SUB % 2)
        _age(paths["sub"])  # old enough that a fresh read would be taken for current

        stage_id = session.load_stage_or_raise(paths["a"])

        assert _size(session, stage_id) == 2.0
        assert held.GetPrimAtPath("/World/FromSub").attributes["size"].default == 2.0


def _instanced_clip_scene(directory: Path) -> Path:
    """The auditor's caseA: an asset whose /Arm/link is animated by a clip set with a manifest beside it,
    referenced three times as instanceable with layer offsets 0, 100 and 1000 (scale 2)."""
    anim = directory / "assets" / "anim"
    anim.mkdir(parents=True)
    clip = Sdf.Layer.CreateNew(str(anim / "arm_0.usda"))
    spec = Sdf.CreatePrimInLayer(clip, "/Src/joint")
    spec.specifier = Sdf.SpecifierOver
    angle = Sdf.AttributeSpec(spec, "custom:angle", Sdf.ValueTypeNames.Float)
    for frame in range(21):
        clip.SetTimeSample(angle.path, frame, float(frame))
    clip.Save()
    manifest = Sdf.Layer.CreateNew(str(anim / "arm_manifest.usda"))
    Sdf.AttributeSpec(Sdf.CreatePrimInLayer(manifest, "/Src/joint"), "custom:angle", Sdf.ValueTypeNames.Float)
    manifest.Save()
    asset = Usd.Stage.CreateNew(str(directory / "assets" / "arm.usda"))
    asset.SetDefaultPrim(UsdGeom.Xform.Define(asset, "/Arm").GetPrim())
    asset.DefinePrim("/Arm/link", "Xform")
    asset.DefinePrim("/Arm/link/joint", "Xform").CreateAttribute("custom:angle", Sdf.ValueTypeNames.Float)
    clips = Usd.ClipsAPI(asset.GetPrimAtPath("/Arm/link"))
    clips.SetClipAssetPaths([Sdf.AssetPath("./anim/arm_0.usda")])
    clips.SetClipPrimPath("/Src")
    clips.SetClipManifestAssetPath(Sdf.AssetPath("./anim/arm_manifest.usda"))
    clips.SetClipActive([(0.0, 0)])
    clips.SetClipTimes([(0.0, 0.0), (20.0, 20.0)])
    asset.GetRootLayer().Save()
    scene = Usd.Stage.CreateNew(str(directory / "scene.usda"))
    scene.SetDefaultPrim(UsdGeom.Xform.Define(scene, "/World").GetPrim())
    for index, (offset, scale) in enumerate([(0.0, 1.0), (100.0, 1.0), (1000.0, 2.0)]):
        prim = scene.DefinePrim(f"/World/arm_{index}", "Xform")
        prim.GetReferences().AddReference(
            Sdf.Reference("./assets/arm.usda", layerOffset=Sdf.LayerOffset(offset, scale))
        )
        prim.SetInstanceable(True)
    scene.GetRootLayer().Save()
    return directory / "scene.usda"


def _pxr_sample_range(stage: Usd.Stage) -> Optional[List[float]]:
    """pxr's own answer: GetTimeSamples over every attribute, instance proxies included."""
    first = last = None
    predicate = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), predicate):
        for attribute in prim.GetAttributes():
            times = attribute.GetTimeSamples()
            if times:
                first = times[0] if first is None else min(first, times[0])
                last = times[-1] if last is None else max(last, times[-1])
    return None if first is None else [float(first), float(last)]


class TestInstancedClips:
    """M2: pxr's ValueMightBeTimeVarying says False for a clip-valued attribute of an instance proxy."""

    def test_the_clip_samples_of_every_instance_are_reported(self, tmp_path) -> None:
        stage = Usd.Stage.Open(str(_instanced_clip_scene(tmp_path)))

        result = animation_range(stage)

        assert result["time_sample_range"] == [0.0, 1040.0] == _pxr_sample_range(stage)
        assert result["has_animation"] is True

    def test_the_session_reports_it_on_load_and_in_the_summary(self, session, tmp_path) -> None:
        stage_id = session.load_stage_or_raise(_instanced_clip_scene(tmp_path))

        assert session.get_stage_info(stage_id).has_animation is True
        assert session.summarize_stage(stage_id, include_meshes=False).animation_info["time_sample_range"] == [
            0.0,
            1040.0,
        ]


_ROBOT = """#usda 1.0
(
    defaultPrim = "W"
    upAxis = "Z"
    metersPerUnit = 1
)

def Xform "W"
{
    def Cube "Body" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {
    }
%s}
"""
_UNREADABLE_JOINT = """
    def PhysicsRevoluteJoint "Bad%d"
    {
        custom string physics:body0 = "/W/Body"
    }
"""


class TestValidatorFailureReport:
    """m1 and m2: a ValidatorFailed issue stays small and names the layer, whatever its path."""

    def test_reasons_are_capped_with_a_count(self, tmp_path) -> None:
        path = tmp_path / "robot.usda"
        path.write_text(_ROBOT % "".join(_UNREADABLE_JOINT % index for index in range(10)))

        report = validation.validate_usd_file(str(path), USD)

        failure = next(issue for issue in report.issues if issue.code == "ValidatorFailed")
        assert failure.message.count("is not a relationship") == validation.MAX_VALIDATOR_FAILURE_REASONS
        assert "; and 7 more like these." in failure.message
        assert failure.layer == str(path.resolve())
        assert failure.property_path is not None and failure.property_path.startswith("/W/Bad")

    def test_a_layer_path_with_spaces_is_located(self, tmp_path) -> None:
        directory = tmp_path / "dir with space"
        directory.mkdir()
        path = directory / "robot file.usda"
        path.write_text(_ROBOT % (_UNREADABLE_JOINT % 0))

        report = validation.validate_usd_file(str(path), USD)

        failure = next(issue for issue in report.issues if issue.code == "ValidatorFailed")
        assert failure.location() == {
            "layer": str(path.resolve()),
            "prim_path": "/W/Bad0",
            "property_path": "/W/Bad0.physics:body0",
        }
        assert "@" not in failure.message and "dir with space" not in failure.message

    def test_an_unreadable_property_in_a_spaced_path_is_located(self, tmp_path) -> None:
        directory = tmp_path / "dir with space"
        directory.mkdir()
        path = directory / "robot file.usda"
        path.write_text(_ROBOT % (_UNREADABLE_JOINT % 0))
        stage = Usd.Stage.Open(str(path))
        problems: List[Dict[str, Any]] = []
        relationship = stage.GetPrimAtPath("/W/Bad0").GetRelationship("physics:body0")

        targets = pxr_errors.relationship_targets(relationship, problems)

        assert targets is None
        (problem,) = problems
        assert problem["location"]["layer"] == str(path.resolve())
        assert "@" not in problem["message"]

    def test_spec_reference_takes_no_text_between_unrelated_at_signs(self) -> None:
        found = validation.SPEC_REFERENCE.search("@a.usda@ and @/b c/d.usda@</P.x> and '@/e\"f.usda@</Q>'")

        assert (found.group("layer"), found.group("path")) == ("/b c/d.usda", "/P.x")
        assert [match.group("layer") for match in validation.SPEC_REFERENCE.finditer("x @/e\"f.usda@</Q>")] == [
            "/e\"f.usda"
        ]


class TestRereadClearsEveryStage:
    """m3: a reread in place shows the file's new content in every stage composing it, so none lists it."""

    def test_a_stage_another_load_brought_up_to_date_is_current_and_reused(self, session, tmp_path) -> None:
        paths = _shared_sublayer(tmp_path)
        sub = str(paths["sub"].resolve())
        first = session.load_stage_or_raise(paths["a"])
        _external_write(paths["sub"], _SUB % 2)
        assert _changed(session, first) == [sub]

        session.load_stage_or_raise(paths["b"])  # rereads sub.usda in place

        assert _size(session, first) == 2.0
        assert _changed(session, first) == []
        assert session.load_stage_or_raise(paths["a"]) == first

    def test_a_stage_of_that_file_itself_keeps_its_root_changed(self, session, tmp_path) -> None:
        paths = _shared_sublayer(tmp_path)
        sub = str(paths["sub"].resolve())
        own = session.load_stage_or_raise(paths["sub"])  # its root is a private copy of sub.usda
        session.load_stage_or_raise(paths["a"])
        _external_write(paths["sub"], _SUB % 2)

        session.load_stage_or_raise(paths["b"])

        assert _size(session, own) == 1.0
        assert _changed(session, own) == [sub]
