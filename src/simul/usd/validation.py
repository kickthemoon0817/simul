"""Validate USD files with pxr: open them, compose them, check what composed.

``validate_usd_file`` and ``simul usd validate`` used to check only that a path
existed, had a USD extension and fit the size limit, so a syntax error, an
empty file or random bytes named ``.usdc`` read as valid while
``load_usd_file`` refused them. Everything here opens the file the way the
loader does, so the two cannot disagree:

- ``check_usd_file``: the access checks load and validate share. The file
  must exist, be a regular file, carry an extension from
  ``usd.allowed_extensions``, be readable by this process, be non-empty and
  fit ``usd.max_file_size_mb``.
- ``open_root_layer``: ``Sdf.Layer.FindOrOpen``, with pxr's own diagnostic
  (a parse error with line and column, crate corruption, an unknown format)
  raised as a ``USDLoadError`` instead of printed to stderr.
- ``composition_issues``: the stage's composition errors, such as missing
  sublayers, references and payloads, or unresolved prim paths.
- ``validate_usd_file``: all of the above plus stage metadata (defaultPrim,
  upAxis, metersPerUnit), physics sanity checks and pxr's UsdValidation
  validators (the usdz package rules only for a package), as a
  ``ValidationReport`` whose issues carry a severity and a location.
"""

from __future__ import annotations

import math
import re
import stat
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set, Tuple

from pxr import Ar

from ..utils.paths import PathPolicy
from ._pxr import Sdf, Tf, Usd, UsdGeom

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import USDConfig
    from .sandbox import CompositionSandbox

ERROR = "error"
WARNING = "warning"
INFO = "info"
_SEVERITY_RANK = {ERROR: 0, WARNING: 1, INFO: 2}

# pxr mangles its namespace into every function name it reports.
_PXR_NAMESPACE = re.compile(r"pxrInternal_v\d+(?:_\d+)*__pxrReserved__::")
# The text parser prefixes its message with ``<file>:<line>:<column>:``.
_TEXT_POSITION = re.compile(r"^(?P<file>.+?):(?P<line>\d+):(?P<column>\d+):\s*(?P<message>.*)$", re.S)
_ASSET_IN_MESSAGE = re.compile(r"@([^@]+)@")
_QUOTED = re.compile(r"'([^']+)'")


@dataclass(frozen=True)
class ValidationIssue:
    """One problem found in a USD file.

    Attributes:
        severity: ``error`` (the file is broken or will misbehave),
            ``warning`` (it works, but probably not as intended) or ``info``.
        code: Stable CamelCase identifier, e.g. ``ParseError`` or
            ``MissingReference``; pxr validator findings keep pxr's name.
        message: What is wrong, in pxr's words where pxr reported it.
        source: ``simul`` for the checks in this module, otherwise the pxr
            validator that reported it.
        layer: Layer identifier the problem sits in, when known.
        prim_path: Prim the problem is reported on, when known.
        property_path: Property the problem is reported on, when known.
        asset_path: The asset path that failed to resolve, when there is one.
        line: Line in a text layer, for a parse error.
        column: Column in a text layer, for a parse error.
    """

    severity: str
    code: str
    message: str
    source: str = "simul"
    layer: Optional[str] = None
    prim_path: Optional[str] = None
    property_path: Optional[str] = None
    asset_path: Optional[str] = None
    line: Optional[int] = None
    column: Optional[int] = None

    def location(self) -> Dict[str, Any]:
        """Return the set location fields, keyed as the MCP payload reports them."""
        fields = (
            ("layer", self.layer),
            ("prim_path", self.prim_path),
            ("property_path", self.property_path),
            ("asset_path", self.asset_path),
            ("line", self.line),
            ("column", self.column),
        )
        return {key: value for key, value in fields if value is not None}

    def to_dict(self) -> Dict[str, Any]:
        """Return the JSON shape the MCP tools and the CLI report."""
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "source": self.source,
            "location": self.location(),
        }


class USDLoadError(Exception):
    """A USD file could not be opened; ``reason`` says why.

    The tool envelope reports it as ``error_type`` ``LoadError`` with
    ``details.reason`` set to one of ``FileNotFound``, ``NotAFile``,
    ``UnsupportedExtension``, ``PermissionDenied``, ``Unreadable``,
    ``EmptyFile``, ``FileTooLarge``, ``ParseError``, ``CorruptFile``,
    ``UnknownFormat``, ``UnsupportedURL`` or ``OpenFailed``.
    """

    error_type = "LoadError"

    def __init__(
        self,
        file_path: str,
        reason: str,
        detail: str,
        *,
        hint: Optional[str] = None,
        line: Optional[int] = None,
        column: Optional[int] = None,
    ) -> None:
        message = f"Cannot load USD file {file_path}: {detail}"
        super().__init__(f"{message}. {hint}" if hint else message)
        self.file_path = file_path
        self.reason = reason
        self.detail = detail
        self.hint = hint
        self.line = line
        self.column = column

    @property
    def details(self) -> Dict[str, Any]:
        """The ``details`` mapping of the error envelope."""
        details: Dict[str, Any] = {"reason": self.reason, "file_path": self.file_path}
        if self.line is not None:
            details["line"] = self.line
        if self.column is not None:
            details["column"] = self.column
        if self.hint:
            details["hint"] = self.hint
        return details

    def to_issue(self) -> ValidationIssue:
        """Report the failure as a validation issue."""
        message = f"{self.detail}. {self.hint}" if self.hint else self.detail
        return ValidationIssue(
            ERROR,
            self.reason,
            message,
            layer=self.file_path,
            line=self.line,
            column=self.column,
        )


@dataclass(frozen=True)
class FileFacts:
    """What ``check_usd_file`` learned about a file that passed it."""

    size: int
    head: bytes


def _size_mb(size: int) -> str:
    return f"{size / (1024 * 1024):.1f} MB"


def check_usd_file(file_path: str, usd_config: "USDConfig") -> FileFacts:
    """Check what both load and validate require before pxr sees the file.

    Args:
        file_path: Absolute path of the file.
        usd_config: The ``usd`` settings; ``allowed_extensions`` and
            ``max_file_size_mb`` apply.

    Returns:
        The file's size and first bytes.

    Raises:
        USDLoadError: The file is missing, not a regular file, has an
            extension outside ``usd.allowed_extensions``, cannot be read, is
            empty or exceeds ``usd.max_file_size_mb``.
    """
    try:
        info = Path(file_path).stat()
    except FileNotFoundError:
        raise USDLoadError(file_path, "FileNotFound", "no such file") from None
    except PermissionError:
        raise USDLoadError(
            file_path,
            "PermissionDenied",
            "permission denied while looking the file up (a parent directory is not searchable)",
        ) from None
    except OSError as exc:
        raise USDLoadError(file_path, "Unreadable", exc.strerror or str(exc)) from None
    if not stat.S_ISREG(info.st_mode):
        kind = "a directory" if stat.S_ISDIR(info.st_mode) else "not a regular file"
        raise USDLoadError(file_path, "NotAFile", f"the path is {kind}")

    allowed = [extension.lower() for extension in usd_config.allowed_extensions]
    if not any(file_path.lower().endswith(extension) for extension in allowed):
        suffix = Path(file_path).suffix or "(none)"
        raise USDLoadError(
            file_path,
            "UnsupportedExtension",
            f"extension {suffix} is not in usd.allowed_extensions {allowed}",
        )

    try:
        with open(file_path, "rb") as handle:
            head = handle.read(16)
    except PermissionError:
        raise USDLoadError(
            file_path,
            "PermissionDenied",
            "permission denied: the server process cannot read the file",
        ) from None
    except OSError as exc:
        raise USDLoadError(file_path, "Unreadable", exc.strerror or str(exc)) from None

    if info.st_size == 0:
        raise USDLoadError(file_path, "EmptyFile", "the file is empty (0 bytes)")
    limit_mb = usd_config.max_file_size_mb
    if info.st_size > limit_mb * 1024 * 1024:
        raise USDLoadError(
            file_path,
            "FileTooLarge",
            f"the file is {_size_mb(info.st_size)}, over usd.max_file_size_mb ({limit_mb} MB)",
        )
    return FileFacts(size=info.st_size, head=head)


def url_scheme(path: str) -> Optional[str]:
    """The lower-cased scheme of a URL other than ``file://``, or None for a local path."""
    scheme = PathPolicy.url_scheme(path)
    return None if scheme in (None, "file") else scheme


def resolver_uri_schemes() -> Set[str]:
    """URI schemes an installed asset resolver handles; usd-core registers none."""
    try:
        return {scheme.lower() for scheme in Ar.GetRegisteredURISchemes()}
    except Exception:  # pragma: no cover - an Ar without URI resolvers
        return set()


def check_usd_url(url: str) -> None:
    """Refuse a URL no installed asset resolver can open.

    pxr opens a URL only through an asset resolver registered for its scheme
    (Omniverse's for ``omniverse://``). usd-core ships the filesystem resolver
    alone, so such a URL passed the sandbox's scheme allowlist and then failed
    with a generic error.

    Raises:
        USDLoadError: ``UnsupportedURL`` naming the scheme and the remedy.
    """
    scheme = url_scheme(url) or ""
    registered = resolver_uri_schemes()
    if scheme in registered:
        return
    raise USDLoadError(
        url,
        "UnsupportedURL",
        f"no asset resolver for {scheme}:// URLs is installed in this server "
        f"(registered URI schemes: {', '.join(sorted(registered)) or 'none'}); the headless "
        "USD tools open local files",
        hint="Copy the asset to a directory under security.allowed_paths and pass its local path.",
    )


def content_hint(file_path: str, head: bytes) -> Optional[str]:
    """Name a mismatch between a file's extension and the bytes it holds.

    Args:
        file_path: Path of the file, for its extension.
        head: The file's first bytes.

    Returns:
        A sentence saying what the file really holds, or None (also when no
        bytes were read, as for a URL).
    """
    if not head:
        return None
    suffix = Path(file_path).suffix.lower()
    if head.startswith(b"#usda"):
        if suffix in (".usdc", ".usdz"):
            return f"The file holds text USD (#usda), not what {suffix} promises; rename it to .usda."
        return None
    if head.startswith(b"PXR-USDC"):
        if suffix in (".usda", ".usdz"):
            return f"The file holds binary crate USD (PXR-USDC), not what {suffix} promises; rename it to .usdc."
        return None
    if head.startswith(b"PK\x03\x04"):
        if suffix != ".usdz":
            return "The file is a zip archive; a USDZ package must be named .usdz."
        return None
    if suffix in (".usd", ".usda", ".usdc", ".usdz"):
        return "The file starts with none of the USD signatures (#usda, PXR-USDC, a zip header)."
    return None


def tf_error_messages(exc: BaseException) -> List[str]:
    """Return the messages a ``Tf.ErrorException`` carries, without pxr's internals.

    Args:
        exc: The exception pxr raised.

    Returns:
        Each ``Tf.Error``'s commentary, with pxr's mangled namespace removed.
    """
    messages: List[str] = []
    for error in getattr(exc, "args", ()):
        text = getattr(error, "commentary", None)
        cleaned = _PXR_NAMESPACE.sub("", str(text if text is not None else error)).strip()
        if cleaned:
            messages.append(cleaned)
    if not messages:
        cleaned = _PXR_NAMESPACE.sub("", str(exc)).strip()
        if cleaned:
            messages.append(cleaned)
    return messages


def _load_error_from_tf(file_path: str, exc: BaseException, head: bytes) -> USDLoadError:
    """Classify the error pxr raised while opening a root layer."""
    messages = tf_error_messages(exc) or ["pxr gave no reason"]
    # UsdStage::Open adds a generic "Failed to open layer" after the cause.
    specific = [m for m in messages if not m.startswith("Failed to open layer")]
    primary = (specific or messages)[0]
    hint = content_hint(file_path, head)

    position = _TEXT_POSITION.match(primary)
    if position is not None:
        line, column = int(position.group("line")), int(position.group("column"))
        return USDLoadError(
            file_path,
            "ParseError",
            f"syntax error at line {line}, column {column}: {position.group('message').strip()}",
            hint=hint,
            line=line,
            column=column,
        )
    lowered = primary.lower()
    if "not a valid usda layer" in lowered or "syntax error" in lowered or "parse error" in lowered:
        return USDLoadError(file_path, "ParseError", primary, hint=hint)
    if "crate" in lowered or "bootstrap" in lowered:
        return USDLoadError(file_path, "CorruptFile", f"corrupt binary crate data: {primary}", hint=hint)
    if any(word in lowered for word in ("zip", "package", "usdz", "compressed")):
        return USDLoadError(file_path, "CorruptFile", f"invalid usdz package: {primary}", hint=hint)
    if "cannot determine file format" in lowered:
        return USDLoadError(file_path, "UnknownFormat", primary, hint=hint)
    return USDLoadError(file_path, "OpenFailed", "; ".join(specific or messages), hint=hint)


def open_root_layer(file_path: str, head: bytes = b"") -> "Sdf.Layer":
    """Open a file's root layer, raising pxr's reason when it cannot.

    The caller keeps the returned layer alive while it opens the stage, so
    ``Usd.Stage.Open`` on the same path finds it in the layer registry instead
    of parsing the file again.

    Args:
        file_path: Absolute path of a file that passed ``check_usd_file``.
        head: The file's first bytes, for a hint about a mismatched extension.

    Returns:
        The opened layer.

    Raises:
        USDLoadError: pxr could not open or parse the layer.
    """
    try:
        layer = Sdf.Layer.FindOrOpen(file_path)
    except Tf.ErrorException as exc:
        raise _load_error_from_tf(file_path, exc, head) from None
    if layer is None:
        raise USDLoadError(
            file_path,
            "OpenFailed",
            "pxr could not open the layer and reported no reason",
            hint=content_hint(file_path, head),
        )
    return layer


def _pcp_code(error: Any, message: str) -> str:
    """Name a Pcp composition error for the issue's ``code``."""
    name = str(getattr(error, "errorType", "")).rsplit("ErrorType_", 1)[-1] or type(error).__name__
    if name == "InvalidSublayerPath":
        return "MissingSublayer"
    if name == "InvalidAssetPath":
        if "for payload" in message:
            return "MissingPayload"
        if "for reference" in message:
            return "MissingReference"
        return "MissingAsset"
    return name


def composition_issues(stage: "Usd.Stage", severity: str = ERROR) -> List[ValidationIssue]:
    """Report the stage's composition errors.

    A missing sublayer, reference or payload, or a reference to a prim the
    target layer lacks, leaves the stage open with that part of the scene
    missing; pxr prints each as a warning on stderr and composes on.

    Args:
        stage: The composed stage.
        severity: The severity to report them at. Validation reports them as
            errors; a successful load lists them as warnings.

    Returns:
        One issue per composition error.
    """
    get_errors = getattr(stage, "GetCompositionErrors", None)
    if get_errors is None:
        return []
    issues: List[ValidationIssue] = []
    for error in get_errors():
        message = _PXR_NAMESPACE.sub("", str(error)).strip()
        asset = _ASSET_IN_MESSAGE.search(message)
        site = getattr(error, "rootSite", None)
        prim_path = str(site.path) if site is not None and site.path else None
        issues.append(
            ValidationIssue(
                severity,
                _pcp_code(error, message),
                message,
                layer=stage.GetRootLayer().identifier,
                prim_path=prim_path,
                asset_path=asset.group(1) if asset else None,
            )
        )
    return issues


def _default_prim_path(token: str) -> str:
    return token if token.startswith("/") else f"/{token}"


def metadata_issues(stage: "Usd.Stage", root_layer: "Sdf.Layer") -> List[ValidationIssue]:
    """Check the stage's defaultPrim, upAxis and metersPerUnit.

    A dangling defaultPrim is an error: referencing the file without a prim
    path fails. An unset one is a warning, since a top-level scene needs none.
    Unset upAxis and metersPerUnit are warnings because consumers fall back to
    defaults (Y up, centimeters) that robotics content rarely means; an
    authored value pxr cannot use is an error.

    Args:
        stage: The composed stage.
        root_layer: Its root layer.

    Returns:
        The metadata issues found.
    """
    layer_id = root_layer.identifier
    issues: List[ValidationIssue] = []

    authored_default = root_layer.defaultPrim
    if not authored_default:
        issues.append(
            ValidationIssue(
                WARNING,
                "MissingDefaultPrim",
                "The root layer sets no defaultPrim; a reference to this file that names "
                "no prim path will fail.",
                layer=layer_id,
            )
        )
    elif not stage.GetDefaultPrim():
        target = _default_prim_path(str(authored_default))
        issues.append(
            ValidationIssue(
                ERROR,
                "DanglingDefaultPrim",
                f"defaultPrim is {str(authored_default)!r}, but the stage has no prim at {target}; "
                "a reference to this file that names no prim path will fail.",
                layer=layer_id,
                prim_path=target,
            )
        )

    if not stage.HasAuthoredMetadata(UsdGeom.Tokens.upAxis):
        issues.append(
            ValidationIssue(
                WARNING,
                "MissingUpAxis",
                f"The stage sets no upAxis; consumers fall back to {UsdGeom.GetFallbackUpAxis()}.",
                layer=layer_id,
            )
        )
    else:
        up_axis = str(stage.GetMetadata(UsdGeom.Tokens.upAxis))
        if up_axis not in ("Y", "Z"):
            issues.append(
                ValidationIssue(
                    ERROR,
                    "InvalidUpAxis",
                    f"upAxis is {up_axis!r}; USD supports only 'Y' and 'Z'.",
                    layer=layer_id,
                )
            )

    if not stage.HasAuthoredMetadata(UsdGeom.Tokens.metersPerUnit):
        issues.append(
            ValidationIssue(
                WARNING,
                "MissingMetersPerUnit",
                "The stage sets no metersPerUnit; consumers fall back to 0.01 (centimeters).",
                layer=layer_id,
            )
        )
    else:
        meters = stage.GetMetadata(UsdGeom.Tokens.metersPerUnit)
        if not isinstance(meters, (int, float)) or not math.isfinite(meters) or meters <= 0:
            issues.append(
                ValidationIssue(
                    ERROR,
                    "InvalidMetersPerUnit",
                    f"metersPerUnit is {meters!r}; it must be a positive number.",
                    layer=layer_id,
                )
            )
    return issues


def _authored_float(attribute: Any) -> Optional[float]:
    if not attribute or not attribute.HasAuthoredValue():
        return None
    value = attribute.Get()
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def physics_issues(stage: "Usd.Stage") -> List[ValidationIssue]:
    """Physics sanity checks pxr's UsdPhysics validators do not make.

    pxr's validators already report dangling joint body targets, nested
    articulation roots, negative mass and inertia mistakes. This adds a rigid
    body nested under another rigid body without ``!resetXformStack!`` (physics
    engines simulate such a hierarchy unpredictably) and a revolute or
    prismatic joint whose lowerLimit exceeds its upperLimit.

    Args:
        stage: The composed stage.

    Returns:
        The physics issues found.
    """
    from pxr import UsdPhysics

    issues: List[ValidationIssue] = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            reset = prim.IsA(UsdGeom.Xformable) and UsdGeom.Xformable(prim).GetResetXformStack()
            ancestor = prim.GetParent()
            while ancestor and not ancestor.IsPseudoRoot():
                if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                    break
                ancestor = ancestor.GetParent()
            if ancestor and not ancestor.IsPseudoRoot() and not reset:
                issues.append(
                    ValidationIssue(
                        ERROR,
                        "NestedRigidBody",
                        f"Rigid body {prim.GetPath()} is nested under rigid body "
                        f"{ancestor.GetPath()} without resetting its xform stack; physics "
                        "engines simulate such a hierarchy unpredictably.",
                        prim_path=str(prim.GetPath()),
                    )
                )
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
            joint = (
                UsdPhysics.RevoluteJoint(prim)
                if prim.IsA(UsdPhysics.RevoluteJoint)
                else UsdPhysics.PrismaticJoint(prim)
            )
            lower = _authored_float(joint.GetLowerLimitAttr())
            upper = _authored_float(joint.GetUpperLimitAttr())
            if lower is not None and upper is not None and lower > upper:
                issues.append(
                    ValidationIssue(
                        ERROR,
                        "InvertedJointLimits",
                        f"Joint {prim.GetPath()} has lowerLimit {lower:g} greater than "
                        f"upperLimit {upper:g}.",
                        prim_path=str(prim.GetPath()),
                        property_path=str(prim.GetPath().AppendProperty("physics:lowerLimit")),
                    )
                )
    return issues


# Replaced by metadata_issues (which splits "unset" from "wrong") and by
# composition_issues (which reports the asset path that failed).
_REPLACED_VALIDATORS = frozenset(
    {
        "usdValidation:StageMetadataChecker",
        "usdGeomValidators:StageMetadataChecker",
        "usdValidation:CompositionErrorTest",
    }
)
_USDZ_KEYWORD = "UsdzValidators"
_VALIDATOR_LOCK = threading.Lock()
_VALIDATOR_CACHE: Dict[bool, Tuple[Tuple[str, ...], Any]] = {}


def _usd_validation() -> Any:
    try:
        from pxr import UsdValidation
    except ImportError:
        return None
    return UsdValidation


def _validators(package: bool) -> Tuple[Tuple[str, ...], Any]:
    """Return the names and loaded pxr validators to run, cached per package flag."""
    UsdValidation = _usd_validation()
    with _VALIDATOR_LOCK:
        cached = _VALIDATOR_CACHE.get(package)
        if cached is None:
            registry = UsdValidation.ValidationRegistry()
            names = tuple(
                sorted(
                    metadata.name
                    for metadata in registry.GetAllValidatorMetadata()
                    if not metadata.isSuite
                    and metadata.name not in _REPLACED_VALIDATORS
                    and (package or _USDZ_KEYWORD not in list(metadata.GetKeywords()))
                )
            )
            cached = (names, registry.GetOrLoadValidatorsByName(list(names)))
            _VALIDATOR_CACHE[package] = cached
    return cached


def _site_location(site: Any) -> Dict[str, Optional[str]]:
    location: Dict[str, Optional[str]] = {}
    try:
        if site.IsPrim():
            location["prim_path"] = str(site.GetPrim().GetPath())
        elif site.IsProperty():
            prop = site.GetProperty()
            location["prim_path"] = str(prop.GetPrimPath())
            location["property_path"] = str(prop.GetPath())
        layer = site.GetLayer()
        if layer:
            location["layer"] = layer.identifier
    except Exception:  # pragma: no cover - a malformed site keeps the message
        pass
    return location


def compliance_issues(
    stage: "Usd.Stage", *, package: bool, known_assets: Iterable[str] = ()
) -> Tuple[List[ValidationIssue], List[str]]:
    """Run pxr's UsdValidation validators over the stage.

    Args:
        stage: The composed stage.
        package: Whether the root layer is a usdz package; only then do the
            usdz package rules (``UsdzValidators``) run.
        known_assets: Asset paths composition_issues already reported, so the
            unresolved-dependency validator does not repeat them.

    Returns:
        The issues, and the names of the validators that ran.
    """
    UsdValidation = _usd_validation()
    if UsdValidation is None:
        return (
            [
                ValidationIssue(
                    INFO,
                    "ComplianceCheckerUnavailable",
                    "This usd-core has no UsdValidation module; pxr's compliance checks were skipped.",
                )
            ],
            [],
        )
    names, validators = _validators(package)
    severities = {
        UsdValidation.ValidationErrorType.Error: ERROR,
        UsdValidation.ValidationErrorType.Warn: WARNING,
        UsdValidation.ValidationErrorType.Info: INFO,
    }
    reported: Set[str] = set(known_assets)
    # Overlapping validators (the root-package and every-package usdz rules)
    # report the same finding; keep one.
    seen: Set[Tuple[str, str, Optional[str]]] = set()
    issues: List[ValidationIssue] = []
    try:
        errors = UsdValidation.ValidationContext(validators).Validate(stage)
    except Exception as exc:
        message = "; ".join(tf_error_messages(exc)) or str(exc)
        return (
            [ValidationIssue(WARNING, "ComplianceCheckFailed", f"pxr's validators raised: {message}")],
            list(names),
        )
    for error in errors:
        if error.HasNoError():
            continue
        identifier = error.GetIdentifier() or ""
        code = error.GetName() or identifier.rsplit(".", 1)[-1] or "ValidationError"
        source = identifier.rsplit(".", 1)[0] if "." in identifier else identifier or "pxr"
        message = _PXR_NAMESPACE.sub("", error.GetMessage() or "").strip()
        asset_path: Optional[str] = None
        if code == "UnresolvableDependency":
            quoted = _QUOTED.search(message)
            asset_path = quoted.group(1) if quoted else None
            if asset_path is not None and asset_path in reported:
                continue
        sites = list(error.GetSites())
        location = _site_location(sites[0]) if sites else {}
        key = (code, message, location.get("prim_path"))
        if key in seen:
            continue
        seen.add(key)
        issues.append(
            ValidationIssue(
                severities.get(error.GetType(), WARNING),
                code,
                message,
                source=source,
                layer=location.get("layer"),
                prim_path=location.get("prim_path"),
                property_path=location.get("property_path"),
                asset_path=asset_path,
            )
        )
    return issues, list(names)


def unsaved_edit_issues(stage: "Usd.Stage") -> List[ValidationIssue]:
    """Say when validation read a layer that a loaded stage has edited in memory.

    pxr shares one in-memory copy of a layer between every stage that opens
    it, so a file a loaded stage has edited validates as edited, not as it is
    on disk.
    """
    return [
        ValidationIssue(
            INFO,
            "UnsavedEdits",
            f"Validated the in-memory copy of {layer.identifier}, which a loaded stage has "
            "edited; the file on disk may differ.",
            layer=layer.identifier,
        )
        for layer in stage.GetUsedLayers()
        if not layer.anonymous and layer.dirty
    ]


@dataclass
class ValidationReport:
    """Everything validation found out about one file."""

    file_path: str
    file_size: int
    format: str
    can_read: bool = False
    layer_format: Optional[str] = None
    default_prim: Optional[str] = None
    up_axis: Optional[str] = None
    meters_per_unit: Optional[float] = None
    issues: List[ValidationIssue] = field(default_factory=list)
    validators: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True when pxr opened the file and no check reported an error."""
        return self.can_read and not any(issue.severity == ERROR for issue in self.issues)

    def count(self, severity: str) -> int:
        """Return how many issues carry ``severity``."""
        return sum(1 for issue in self.issues if issue.severity == severity)

    def sorted_issues(self) -> List[ValidationIssue]:
        """Errors first, then warnings, then info; discovery order within each."""
        return sorted(self.issues, key=lambda issue: _SEVERITY_RANK.get(issue.severity, 3))

    def to_dict(self) -> Dict[str, Any]:
        """Return the JSON shape ``validate_usd_file`` and ``simul usd validate`` report."""
        return {
            "file_path": self.file_path,
            "file_size": self.file_size,
            "format": self.format,
            "is_valid": self.is_valid,
            "can_read": self.can_read,
            "layer_format": self.layer_format,
            "default_prim": self.default_prim,
            "up_axis": self.up_axis,
            "meters_per_unit": self.meters_per_unit,
            "error_count": self.count(ERROR),
            "warning_count": self.count(WARNING),
            "info_count": self.count(INFO),
            "issues": [issue.to_dict() for issue in self.sorted_issues()],
            "validators": list(self.validators),
        }


def _file_size(file_path: str) -> int:
    """Size of a regular file, 0 for anything else or a path that cannot be stat'ed."""
    try:
        info = Path(file_path).stat()
    except OSError:
        return 0
    return info.st_size if stat.S_ISREG(info.st_mode) else 0


def validate_usd_file(
    file_path: str, usd_config: "USDConfig", sandbox: Optional["CompositionSandbox"] = None
) -> ValidationReport:
    """Open a USD file the way the loader does and report every problem found.

    Args:
        file_path: Absolute path of the file. The MCP tool passes the path the
            sandbox resolved; the CLI passes the resolved argument.
        usd_config: The ``usd`` settings shared with the loader.
        sandbox: Holds every layer and asset the stage draws on to the file
            sandbox, as the loader does; None (the CLI) checks nothing beyond
            the file.

    Returns:
        The report. ``can_read`` says whether pxr opened the file (and so
        whether ``load_usd_file`` can); ``is_valid`` additionally needs no
        error-severity issue.

    Raises:
        SandboxDenied: The file's composition reaches outside the sandbox.
    """
    remote = url_scheme(file_path) is not None
    report = ValidationReport(
        file_path=file_path,
        file_size=0 if remote else _file_size(file_path),
        format=Path(file_path).suffix.lower().lstrip("."),
    )
    try:
        if remote:
            check_usd_url(file_path)
            root_layer = open_root_layer(file_path)
        else:
            facts = check_usd_file(file_path, usd_config)
            root_layer = open_root_layer(file_path, facts.head)
    except USDLoadError as exc:
        report.issues.append(exc.to_issue())
        return report

    file_format = root_layer.GetFileFormat()
    report.layer_format = file_format.formatId
    walked = sandbox.check_layers(root_layer) if sandbox is not None else []
    try:
        stage = Usd.Stage.Open(root_layer, Usd.Stage.LoadAll)
    except Tf.ErrorException as exc:
        report.issues.append(
            ValidationIssue(
                ERROR,
                "StageOpenFailed",
                "; ".join(tf_error_messages(exc)) or "pxr could not compose the stage",
                layer=file_path,
            )
        )
        return report
    if stage is None:
        report.issues.append(
            ValidationIssue(ERROR, "StageOpenFailed", "pxr could not compose the stage", layer=file_path)
        )
        return report

    if sandbox is not None:
        sandbox.check_stage(stage)
    del walked
    report.can_read = True
    default_prim = stage.GetDefaultPrim()
    report.default_prim = str(default_prim.GetPath()) if default_prim else None
    report.up_axis = str(UsdGeom.GetStageUpAxis(stage))
    report.meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)

    composition = composition_issues(stage, ERROR)
    report.issues.extend(composition)
    report.issues.extend(metadata_issues(stage, root_layer))
    report.issues.extend(physics_issues(stage))
    compliance, names = compliance_issues(
        stage,
        package=bool(file_format.IsPackage()),
        known_assets=[issue.asset_path for issue in composition if issue.asset_path],
    )
    report.issues.extend(compliance)
    report.validators = names
    report.issues.extend(unsaved_edit_issues(stage))
    return report


__all__ = [
    "ERROR",
    "WARNING",
    "INFO",
    "FileFacts",
    "USDLoadError",
    "ValidationIssue",
    "ValidationReport",
    "check_usd_file",
    "check_usd_url",
    "composition_issues",
    "compliance_issues",
    "content_hint",
    "metadata_issues",
    "open_root_layer",
    "physics_issues",
    "resolver_uri_schemes",
    "tf_error_messages",
    "unsaved_edit_issues",
    "url_scheme",
    "validate_usd_file",
]
