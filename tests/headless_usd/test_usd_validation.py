"""USD validation opens the file with pxr and reports what is really wrong with it.

Validation used to check only existence, extension and size, so a syntax
error, an empty file, random bytes or text named ``.usdc`` read as valid and
readable while the loader refused them, and missing references, a dangling
defaultPrim and physics mistakes passed unremarked. The loader's failures all
read "Failed to load USD file" whatever pxr had said. These run against real
pxr files.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

pytest.importorskip("pxr", reason="pxr library not available")

from simul.adapters.headless_usd import HeadlessUSDSession  # noqa: E402
from simul.config import Settings, USDConfig  # noqa: E402
from simul.usd import validation  # noqa: E402
from simul.usd.validation import (  # noqa: E402
    USDLoadError,
    ValidationReport,
    composition_issues,
    validate_usd_file,
)
from tests import usd_fixtures as fx  # noqa: E402

USD = Settings().usd


def _validate(path: Path, usd: USDConfig = USD) -> ValidationReport:
    return validate_usd_file(str(path), usd)


def _codes(report: ValidationReport, severity: str = "error") -> List[str]:
    return [issue.code for issue in report.issues if issue.severity == severity]


def _issue(report: ValidationReport, code: str) -> Dict:
    return next(issue.to_dict() for issue in report.issues if issue.code == code)


@pytest.fixture
def session() -> HeadlessUSDSession:
    usd_session = HeadlessUSDSession()
    yield usd_session
    usd_session.cleanup()


def _unreadable(path: Path) -> Path:
    if sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0):
        pytest.skip("mode 000 does not stop this user from reading")
    path.chmod(0)
    return path


class TestFilesPxrCannotOpen:
    """Each is unreadable, and both validate and load say why in pxr's terms."""

    def test_syntax_error_reports_line_and_column(self, tmp_path: Path) -> None:
        report = _validate(fx.write_corrupt_usda(tmp_path / "corrupt.usda"))

        assert report.is_valid is False
        assert report.can_read is False
        issue = _issue(report, "ParseError")
        assert issue["location"]["line"] == 9
        assert issue["location"]["column"] == 1
        assert "syntax error at line 9" in issue["message"]
        assert "pxrInternal" not in issue["message"]

    def test_empty_file(self, tmp_path: Path) -> None:
        report = _validate(fx.write_empty(tmp_path / "empty.usda"))
        assert (report.is_valid, report.can_read, _codes(report)) == (False, False, ["EmptyFile"])

    def test_random_bytes_as_usdc(self, tmp_path: Path) -> None:
        report = _validate(fx.write_garbage(tmp_path / "garbage.usdc"))
        assert (report.is_valid, report.can_read, _codes(report)) == (False, False, ["CorruptFile"])
        assert "crate" in _issue(report, "CorruptFile")["message"]

    def test_text_in_a_usdc_names_the_mismatch(self, tmp_path: Path) -> None:
        report = _validate(fx.write_text_named_usdc(tmp_path / "text.usdc"))
        assert (report.is_valid, report.can_read, _codes(report)) == (False, False, ["CorruptFile"])
        assert "rename it to .usda" in _issue(report, "CorruptFile")["message"]

    def test_mode_000_file_is_not_readable(self, tmp_path: Path) -> None:
        path = _unreadable(fx.write_robot(tmp_path / "noperm.usda"))
        try:
            report = _validate(path)
        finally:
            path.chmod(0o644)
        assert (report.is_valid, report.can_read, _codes(report)) == (
            False,
            False,
            ["PermissionDenied"],
        )

    def test_missing_file_and_directory(self, tmp_path: Path) -> None:
        assert _codes(_validate(tmp_path / "nope.usda")) == ["FileNotFound"]
        directory = tmp_path / "dir.usda"
        directory.mkdir()
        report = _validate(directory)
        assert _codes(report) == ["NotAFile"]
        assert report.file_size == 0

    def test_size_limit_applies_to_validation(self, tmp_path: Path) -> None:
        path = tmp_path / "big.usda"
        path.write_text("#usda 1.0\n" + "#" * (1024 * 1024 + 10) + "\n")
        report = _validate(path, USDConfig(max_file_size_mb=1))
        assert (report.can_read, _codes(report)) == (False, ["FileTooLarge"])


class TestExtensionsComeFromSettings:
    """Validate and load share ``usd.allowed_extensions`` (they used to differ)."""

    def test_validate_refuses_an_extension_load_refuses(
        self, tmp_path: Path
    ) -> None:
        usd = USDConfig(allowed_extensions=[".usda"])
        path = fx.write_robot(tmp_path / "robot.usd")

        assert _codes(_validate(path, usd)) == ["UnsupportedExtension"]
        load_session = HeadlessUSDSession(Settings(usd=usd))
        with pytest.raises(USDLoadError) as caught:
            load_session.load_stage_or_raise(path)
        assert caught.value.reason == "UnsupportedExtension"

    def test_validate_accepts_an_extension_the_settings_add(self, tmp_path: Path) -> None:
        usd = USDConfig(allowed_extensions=[".usda", ".layer"])
        path = tmp_path / "robot.layer"
        fx.write_robot(tmp_path / "robot.usda").rename(path)
        # The access check passes; pxr then has no file format for .layer.
        assert _codes(_validate(path, usd)) == ["UnknownFormat"]


class TestContentProblems:
    """Files pxr opens but that are broken or will misbehave."""

    def test_clean_robot_is_valid(self, tmp_path: Path) -> None:
        report = _validate(fx.write_robot(tmp_path / "robot.usda"))

        assert report.is_valid is True
        assert report.can_read is True
        assert report.issues == []
        assert (report.default_prim, report.up_axis, report.meters_per_unit) == ("/Robot", "Z", 1.0)
        assert report.validators, "pxr's UsdValidation validators should run"

    def test_missing_sublayer_reference_and_payload(self, tmp_path: Path) -> None:
        report = _validate(fx.write_broken_refs(tmp_path))

        assert report.can_read is True
        assert report.is_valid is False
        assert sorted(_codes(report)) == [
            "MissingPayload",
            "MissingReference",
            "MissingSublayer",
            "UnresolvedPrimPath",
        ]
        reference = _issue(report, "MissingReference")["location"]
        assert reference["prim_path"] == "/World/MissingRef"
        assert reference["asset_path"].endswith("does_not_exist.usda")
        # pxr's dependency validator reports the same assets; they are not repeated.
        assert "UnresolvableDependency" not in _codes(report)

    def test_dangling_default_prim_is_an_error(self, tmp_path: Path) -> None:
        report = _validate(fx.write_dangling_default_prim(tmp_path / "dangling.usda"))
        assert report.is_valid is False
        assert _codes(report) == ["DanglingDefaultPrim"]
        assert _issue(report, "DanglingDefaultPrim")["location"]["prim_path"] == "/Nope"

    def test_unset_metadata_warns_without_failing(self, tmp_path: Path) -> None:
        report = _validate(fx.write_no_metadata(tmp_path / "nometa.usda"))
        assert report.is_valid is True
        assert _codes(report, "warning") == [
            "MissingDefaultPrim",
            "MissingUpAxis",
            "MissingMetersPerUnit",
        ]

    def test_physics_errors(self, tmp_path: Path) -> None:
        report = _validate(fx.write_bad_physics(tmp_path / "bad_physics.usda"))

        assert report.is_valid is False
        assert set(_codes(report)) == {
            "NestedRigidBody",
            "InvertedJointLimits",
            "MassInvalidValues",
            "NestedArticulation",
            "JointInvalidPrimRel",
        }
        assert _issue(report, "JointInvalidPrimRel")["location"]["prim_path"] == "/World/BadJoint"
        assert _issue(report, "NestedRigidBody")["source"] == "simul"
        assert _issue(report, "NestedArticulation")["source"].startswith("usdPhysicsValidators")

    def test_reset_xform_stack_allows_a_nested_body(self, tmp_path: Path) -> None:
        from pxr import Usd, UsdGeom, UsdPhysics

        path = fx.write_robot(tmp_path / "robot.usda")
        stage = Usd.Stage.Open(str(path))
        child = UsdGeom.Xform.Define(stage, "/Robot/base/tool")
        UsdPhysics.RigidBodyAPI.Apply(child.GetPrim())
        child.SetResetXformStack(True)
        stage.GetRootLayer().Save()

        assert "NestedRigidBody" not in _codes(_validate(path))


_BROKEN_ROBOT = """#usda 1.0
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
        float physics:mass = -5
        float3 physics:diagonalInertia = (-1, 0, 0)
    }

    def Shader "Tex"
    {
        uniform token info:id = "UsdUVTexture"
        asset inputs:file = @./missing_texture.png@
    }
%s}
"""
# physics:body0 authored as an attribute where USD expects a relationship.
_UNREADABLE_JOINT = """
    def PhysicsRevoluteJoint "Bad"
    {
        custom string physics:body0 = "/W/Body"
    }
"""


def _validator(name: str) -> Any:
    return SimpleNamespace(GetMetadata=lambda: SimpleNamespace(name=name))


class _FakeContext:
    """A ValidationContext that, like pxr's, raises for all when one validator raises."""

    sizes: List[int] = []

    def __init__(self, validators: List[Any]) -> None:
        self.names = [validator.GetMetadata().name for validator in validators]
        _FakeContext.sizes.append(len(self.names))

    def Validate(self, stage: Any) -> List[str]:
        if "bad" in self.names:
            raise RuntimeError("@/x/robot.usda@</W/Bad.physics:body0> is not a relationship")
        return [f"finding of {name}" for name in self.names]


class TestValidatorFailures:
    """USD round-3 B1: one validator pxr cannot finish used to drop every finding, and is_valid came back true."""

    def test_the_other_validators_findings_survive(self, tmp_path: Path) -> None:
        clean = tmp_path / "base.usda"
        clean.write_text(_BROKEN_ROBOT % "")
        broken = tmp_path / "malformed.usda"
        broken.write_text(_BROKEN_ROBOT % _UNREADABLE_JOINT)

        before = _validate(clean)
        report = _validate(broken)

        assert report.is_valid is False
        assert set(_codes(before)) == {"MassInvalidValues", "InertiaInvalidValues", "UnresolvableDependency"}
        assert set(_codes(report)) == set(_codes(before)) | {"ValidatorFailed"}
        failure = _issue(report, "ValidatorFailed")
        assert failure["source"] == "usdPhysicsValidators:PhysicsJointChecker"
        assert "/W/Bad.physics:body0 is not a relationship" in failure["message"]
        assert "pxrInternal" not in failure["message"] and "@" not in failure["message"]
        assert failure["location"]["prim_path"] == "/W/Bad"
        assert failure["location"]["property_path"] == "/W/Bad.physics:body0"
        assert failure["location"]["layer"] == str(broken.resolve())

    def test_the_fresh_process_reports_the_same(self, tmp_path: Path) -> None:
        broken = tmp_path / "malformed.usda"
        broken.write_text(_BROKEN_ROBOT % _UNREADABLE_JOINT)

        here = _validate(broken)
        fresh = validation._validate_in_fresh_process(str(broken.resolve()), USD, None)

        assert fresh.problem is None
        assert fresh.is_valid is False
        assert sorted(issue.to_dict()["code"] for issue in fresh.issues) == sorted(
            issue.code for issue in here.issues
        )

    def test_one_pass_unless_a_validator_raises(self) -> None:
        fake = SimpleNamespace(ValidationContext=_FakeContext)
        _FakeContext.sizes = []

        clean_errors, clean_failures = validation._run_validators(fake, [_validator("a"), _validator("b")], None)
        clean_sizes = list(_FakeContext.sizes)
        _FakeContext.sizes = []
        errors, failures = validation._run_validators(
            fake, [_validator("a"), _validator("bad"), _validator("b")], None
        )

        assert (clean_errors, clean_failures, clean_sizes) == (["finding of a", "finding of b"], [], [2])
        assert errors == ["finding of a", "finding of b"]
        assert _FakeContext.sizes == [3, 1, 1, 1]
        (failure,) = failures
        assert (failure.severity, failure.code, failure.source) == ("error", "ValidatorFailed", "bad")
        assert failure.to_dict()["location"]["property_path"] == "/W/Bad.physics:body0"


class TestUsdzRules:
    def test_valid_package_runs_the_usdz_validators(self, tmp_path: Path) -> None:
        robot = fx.write_robot(tmp_path / "robot.usda")
        package = _validate(fx.write_usdz(tmp_path / "robot.usdz", robot))
        plain = _validate(robot)

        assert package.is_valid is True
        assert "usdUtilsValidators:RootPackageValidator" in package.validators
        assert "usdUtilsValidators:RootPackageValidator" not in plain.validators

    def test_package_rules_report_misaligned_and_foreign_files(self, tmp_path: Path) -> None:
        robot = fx.write_robot(tmp_path / "robot.usda")
        report = _validate(fx.write_unaligned_usdz(tmp_path / "loose.usdz", robot))

        assert report.can_read is True
        assert report.is_valid is False
        codes = _codes(report)
        assert "ByteMisalignment" in codes
        assert "UnsupportedFileExtensionInPackage" in codes
        # Two usdz validators report each misaligned file; it is listed once.
        assert codes.count("ByteMisalignment") == 2

    def test_compressed_package_cannot_be_read(self, tmp_path: Path) -> None:
        robot = fx.write_robot(tmp_path / "robot.usda")
        report = _validate(fx.write_compressed_usdz(tmp_path / "zipped.usdz", robot))
        assert (report.can_read, _codes(report)) == (False, ["CorruptFile"])


class TestLoadReportsTheReason:
    """``load_stage_or_raise`` carries pxr's reason; ``load_stage`` still returns None."""

    @pytest.mark.parametrize(
        ("builder", "name", "reason"),
        [
            (fx.write_corrupt_usda, "corrupt.usda", "ParseError"),
            (fx.write_empty, "empty.usda", "EmptyFile"),
            (fx.write_garbage, "garbage.usdc", "CorruptFile"),
            (fx.write_text_named_usdc, "text.usdc", "CorruptFile"),
        ],
    )
    def test_unopenable_file(
        self, tmp_path: Path, session: HeadlessUSDSession, builder, name: str, reason: str
    ) -> None:
        path = builder(tmp_path / name)

        with pytest.raises(USDLoadError) as caught:
            session.load_stage_or_raise(path)

        assert caught.value.reason == reason
        assert caught.value.error_type == "LoadError"
        assert caught.value.details["file_path"] == str(path.resolve())
        assert str(path.resolve()) in str(caught.value)
        assert session.load_stage(path) is None

    def test_parse_error_details_carry_the_position(
        self, tmp_path: Path, session: HeadlessUSDSession
    ) -> None:
        with pytest.raises(USDLoadError) as caught:
            session.load_stage_or_raise(fx.write_corrupt_usda(tmp_path / "corrupt.usda"))
        assert caught.value.details["line"] == 9
        assert "line 9, column 1" in str(caught.value)

    def test_unreadable_file(self, tmp_path: Path, session: HeadlessUSDSession) -> None:
        path = _unreadable(fx.write_robot(tmp_path / "noperm.usda"))
        try:
            with pytest.raises(USDLoadError) as caught:
                session.load_stage_or_raise(path)
        finally:
            path.chmod(0o644)
        assert caught.value.reason == "PermissionDenied"

    def test_size_limit(self, tmp_path: Path) -> None:
        path = tmp_path / "big.usda"
        path.write_text("#usda 1.0\n" + "#" * (1024 * 1024 + 10) + "\n")
        limited = HeadlessUSDSession(Settings(usd=USDConfig(max_file_size_mb=1)))
        with pytest.raises(USDLoadError) as caught:
            limited.load_stage_or_raise(path)
        assert caught.value.reason == "FileTooLarge"
        assert "usd.max_file_size_mb (1 MB)" in str(caught.value)

    def test_broken_references_load_with_warnings(
        self, tmp_path: Path, session: HeadlessUSDSession
    ) -> None:
        stage_id = session.load_stage_or_raise(fx.write_broken_refs(tmp_path))

        warnings = composition_issues(session.get_stage(stage_id), "warning")
        assert sorted(issue.code for issue in warnings) == [
            "MissingPayload",
            "MissingReference",
            "MissingSublayer",
            "UnresolvedPrimPath",
        ]
        assert {issue.severity for issue in warnings} == {"warning"}
