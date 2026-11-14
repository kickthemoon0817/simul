"""All-or-nothing prim edits for the headless USD tools.

Every edit runs in two phases. The plan phase checks the whole request
against the composed stage (path syntax, prim existence, the schema registry,
each attribute's type, its allowed tokens, prim metadata) and writes nothing;
it collects every problem it finds before reporting. The write phase applies
the plan under an :class:`EditJournal`, which snapshots exactly the specs the
edit can touch in the edit target layer and restores them if any write fails.
A call that reports failure has therefore changed nothing.

Failures are :class:`USDEditError` instances: a stable ``error_type``, a
message naming the prim and the attribute at fault with the actual reason,
and structured ``details``. pxr's own exception text (C++ function names,
source paths, and sometimes the wrong cause) never reaches the caller.
"""

from __future__ import annotations

import difflib
import uuid
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

from pxr import Pcp

from . import coerce, lookup, schema_properties
from ._pxr import Kind, Sdf, Tf, Usd, UsdGeom
from .pxr_errors import tf_error_reason
from .timecode import resolve_time_code
from .validation import ValidationIssue

# Error types, from the vocabulary every backend's envelope uses.
VALIDATION = "ValidationError"
NOT_FOUND = "NotFoundError"
ENGINE = "EngineError"
REFUSED = "RefusedOperation"
UNSUPPORTED = "UnsupportedOperation"

# Codes for the check that failed, reported in ``details.code``. Each maps to
# the error type above that the envelope reports.
INVALID_PRIM_PATH = "InvalidPrimPath"
UNKNOWN_PRIM_TYPE = "UnknownPrimType"
INVALID_ATTRIBUTE = "InvalidAttribute"
EDIT_FAILED = "EditFailed"
INVALID_TIME_CODE = "InvalidTimeCode"
# A relationship the prim does not have, or has authored as an attribute.
RELATIONSHIP_NOT_EDITABLE = "RelationshipNotEditable"
# Stage lifecycle (save, export, reload) codes, used by the headless session.
INVALID_FILE_PATH = "InvalidFilePath"
SAVE_FAILED = "SaveFailed"
RELOAD_FAILED = "ReloadFailed"
CODE_ERROR_TYPES: Dict[str, str] = {
    INVALID_PRIM_PATH: VALIDATION,
    UNKNOWN_PRIM_TYPE: VALIDATION,
    INVALID_ATTRIBUTE: VALIDATION,
    INVALID_TIME_CODE: VALIDATION,
    INVALID_FILE_PATH: VALIDATION,
    EDIT_FAILED: ENGINE,
    SAVE_FAILED: ENGINE,
    RELOAD_FAILED: ENGINE,
    RELATIONSHIP_NOT_EDITABLE: UNSUPPORTED,
}
# Warning code for an attribute written for a schema the prim does not have.
UNAPPLIED_SCHEMA_ATTRIBUTE = "UnappliedSchemaAttribute"

# Prim metadata the tools set through the proper USD API rather than as an
# attribute of the same name.
SETTABLE_METADATA: Tuple[str, ...] = ("active", "instanceable", "kind")

# Composition and namespace fields a prim spec carries besides the metadata
# Sdf lists for it; none of them is an attribute.
_EXTRA_PRIM_FIELDS = frozenset(
    {
        "specifier",
        "references",
        "inheritPaths",
        "specializes",
        "variantSelection",
        "variantSetNames",
        "primOrder",
        "propertyOrder",
    }
)

_ROOT = Sdf.Path.absoluteRootPath
_XFORMABLE_TYPE = Tf.Type.FindByName("UsdGeomXformable")
# Shading nodes: their inputs:/outputs: are typed by the node (Sdr), not by a light API.
_SHADING_NODE_TYPES = tuple(Tf.Type.FindByName(name) for name in ("UsdShadeShader", "UsdShadeNodeGraph"))
_NODE_NAMESPACES = ("inputs:", "outputs:")


class USDEditError(Exception):
    """A refused or failed edit, reported as one structured error envelope.

    Created with either an error type (``NotFoundError``, ``RefusedOperation``,
    ``UnsupportedOperation``) or a code from ``CODE_ERROR_TYPES``; a code is
    reported as its error type (``ValidationError``, ``EngineError``) with the
    code in ``details.code``, so every USD tool reports the same mistake under
    the same type while the code still says which check failed.

    Attributes:
        error_type: The envelope's ``error_type``.
        code: The specific check, or None for a plain error type.
        message: What went wrong, naming the prim, attribute and reason.
        details: Structured context (``code``, paths, the rejected
            attributes, the flag that overrides a refusal).
    """

    def __init__(self, kind: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.error_type = CODE_ERROR_TYPES.get(kind, kind)
        self.code: Optional[str] = kind if kind in CODE_ERROR_TYPES else None
        self.message = message
        self.details: Dict[str, Any] = {"code": self.code} if self.code else {}
        self.details.update({key: value for key, value in details.items() if value is not None})

    def to_payload(self) -> Dict[str, Any]:
        """Return the error envelope fields for this error."""
        return {
            "success": False,
            "error": self.message,
            "error_type": self.error_type,
            "details": self.details or None,
        }


def describe_value(value: Any) -> str:
    """Describe a JSON value's shape for an error message."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, (int, float)):
        return "a number"
    if isinstance(value, str):
        return "a string"
    if isinstance(value, Mapping):
        return "a JSON object"
    if isinstance(value, (list, tuple)):
        if not value:
            return "an empty list"
        count = len(value)
        if all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
            return f"a list of {count} numbers"
        if all(isinstance(item, str) for item in value):
            return f"a list of {count} strings"
        if all(isinstance(item, (list, tuple)) for item in value):
            return f"a list of {count} lists"
        return f"a mixed list of {count} items"
    return type(value).__name__


# ---------------------------------------------------------------------------
# Paths, prims and types
# ---------------------------------------------------------------------------


def _path_string_problem(text: str) -> Optional[str]:
    """Return why ``text`` is not a valid SdfPath string, or None when it is."""
    result = Sdf.Path.IsValidPathString(text)
    if result is True:
        return None
    if isinstance(result, tuple):
        return str(result[1]) if len(result) > 1 and result[1] else "ill-formed path"
    return None if result else "ill-formed path"


def _path_hint(raw: Any) -> Tuple[str, Dict[str, Any]]:
    """Say what is wrong with a prim path ``lookup.parse_prim_path`` refused, and how to fix it.

    Returns:
        A sentence for the message (empty when there is nothing to add) and
        extra ``details`` (``suggestion``, ``element``).
    """
    if not isinstance(raw, str) or not raw.strip():
        return "Pass an absolute prim path such as /World/Box.", {}
    if raw != raw.strip():
        return "It has leading or trailing whitespace.", {"suggestion": raw.strip()}
    if not raw.startswith("/"):
        suggestion = "/" + raw
        return f"It is relative; did you mean {suggestion!r}?", {"suggestion": suggestion}
    if raw != "/" and raw.endswith("/"):
        suggestion = raw.rstrip("/") or "/"
        return f"It ends with '/'; did you mean {suggestion!r}?", {"suggestion": suggestion}
    if _path_string_problem(raw) is not None:
        element = next(
            (
                name
                for name in raw[1:].split("/")
                if _path_string_problem("/" + name.split(".", 1)[0]) is not None
            ),
            None,
        )
        if element is None:
            return "", {}
        return (
            f"The prim name {element!r} is not an identifier (letters, digits and underscores; "
            "no spaces or punctuation; not starting with a digit).",
            {"element": element},
        )
    path = Sdf.Path(raw)
    if path.IsPropertyPath():
        prim_part = str(path.GetPrimPath())
        return (
            f"It names the property {path.name!r} of {prim_part}; pass {prim_part} and put the "
            "property in attributes.",
            {"suggestion": prim_part},
        )
    return "", {}


def parse_prim_path(raw: Any) -> Sdf.Path:
    """Parse an absolute prim path, as the read tools do, explaining what is wrong with a bad one.

    The check itself is ``lookup.parse_prim_path``, so a path the read tools
    refuse is refused here with the same wording; the message adds which
    element is at fault and the likely fix. The pseudo-root ``/`` parses;
    callers that need a real prim refuse it. Variant selection paths, which
    the lookup accepts, are refused for edits.

    Args:
        raw: The path the caller supplied.

    Returns:
        The parsed path.

    Raises:
        USDEditError: ``ValidationError`` with ``details.code``
            ``InvalidPrimPath``.
    """
    try:
        path = lookup.parse_prim_path(raw if isinstance(raw, str) else "")
    except lookup.USDPathError as exc:
        hint, extra = _path_hint(raw)
        message = f"{exc}." + (f" {hint}" if hint else "")
        raise USDEditError(INVALID_PRIM_PATH, message, prim_path=raw, **extra) from None
    if path.ContainsPrimVariantSelection():
        raise USDEditError(
            INVALID_PRIM_PATH,
            f"Invalid prim path {raw!r}: it contains a variant selection, and editing inside a "
            "variant is not supported.",
            prim_path=raw,
        )
    return path


def _child_names(prim: Usd.Prim, limit: int = 12) -> List[str]:
    names = [child.GetName() for child in prim.GetChildren()]
    return names[:limit]


def missing_prim_error(stage: Usd.Stage, path: Sdf.Path) -> USDEditError:
    """Build the NotFoundError for ``path``, naming the nearest prim that does exist.

    The message starts as the read tools' does (``Prim not found: <path>``).
    """
    ancestor = path.GetParentPath()
    while ancestor != _ROOT and not stage.GetPrimAtPath(ancestor):
        ancestor = ancestor.GetParentPath()
    anchor = stage.GetPrimAtPath(ancestor) if ancestor != _ROOT else stage.GetPseudoRoot()
    if ancestor != _ROOT and anchor and not anchor.IsActive():
        hint = (
            f" Its ancestor {ancestor} is inactive, and the children of an inactive prim are "
            "not composed; activate it first with update_prim_attributes "
            '{"active": true}.'
        )
    else:
        children = _child_names(anchor) if anchor else []
        listing = f" (children: {', '.join(children)})" if children else ""
        hint = f" The nearest existing prim is {ancestor}{listing}."
    return USDEditError(
        NOT_FOUND,
        f"Prim not found: {path}.{hint}",
        prim_path=str(path),
        nearest_existing_prim=str(ancestor),
    )


def require_prim(stage: Usd.Stage, path: Sdf.Path) -> Usd.Prim:
    """Return the prim at ``path`` (through ``lookup.lookup_prim``) or raise NotFoundError."""
    try:
        return lookup.lookup_prim(stage, str(path))
    except lookup.USDLookupError:
        raise missing_prim_error(stage, path) from None


def _instance_of(prim: Usd.Prim) -> Optional[Usd.Prim]:
    """Return the instanceable prim whose prototype supplies instance proxy ``prim``."""
    current = prim
    while current and current.IsInstanceProxy():
        current = current.GetParent()
    return current if current and current.IsInstance() else None


def refuse_instance_proxy(prim: Usd.Prim, action: str) -> None:
    """Raise UnsupportedOperation when ``prim`` is an instance proxy.

    Args:
        prim: The prim the edit targets.
        action: What the edit does, for the message ("delete", "edit").
    """
    if not prim.IsInstanceProxy():
        return
    instance = _instance_of(prim)
    owner = str(instance.GetPath()) if instance else "an instanceable ancestor"
    raise USDEditError(
        UNSUPPORTED,
        f"Cannot {action} {prim.GetPath()}: it is an instance proxy, read from the shared "
        f"prototype of instanceable prim {owner}, and USD does not allow authoring it. "
        f"Make {owner} editable first with update_prim_attributes "
        '{"instanceable": false}.',
        prim_path=str(prim.GetPath()),
        instance=owner,
    )


@lru_cache(maxsize=1)
def concrete_prim_types() -> Tuple[str, ...]:
    """Every concrete (instantiable) prim type name the schema registry knows."""
    typed = Tf.Type.FindByName("UsdTyped")
    names = {Usd.SchemaRegistry.GetConcreteSchemaTypeName(t) for t in typed.GetAllDerivedTypes()}
    names.discard("")
    return tuple(sorted(names))


def check_prim_type(type_name: str) -> Usd.PrimDefinition:
    """Return the prim definition of a registered concrete prim type.

    Args:
        type_name: The requested ``prim_type``.

    Returns:
        Its prim definition.

    Raises:
        USDEditError: ``ValidationError`` (code ``UnknownPrimType``) for an
            unknown, abstract or API schema type, with close matches when
            there are any.
    """
    definition = Usd.SchemaRegistry().FindConcretePrimDefinition(type_name)
    if definition is not None:
        return definition
    tf_type = Usd.SchemaRegistry.GetTypeFromName(type_name)
    known = concrete_prim_types()
    suggestions = [name for name in known if name.lower() == type_name.lower()]
    suggestions += [
        name
        for name in difflib.get_close_matches(type_name, known, n=5, cutoff=0.6)
        if name not in suggestions
    ]
    if not tf_type.isUnknown and Usd.SchemaRegistry.IsAppliedAPISchema(tf_type):
        reason = f"{type_name!r} is an API schema, which is applied to a prim, not a prim type"
    elif not tf_type.isUnknown:
        reason = f"{type_name!r} is an abstract schema type, which no prim can be defined as"
    else:
        reason = f"{type_name!r} is not a prim type registered with USD"
    hint = (
        f" Did you mean {', '.join(suggestions)}?"
        if suggestions
        else " Use a concrete type such as Xform, Scope, Mesh, Cube or Sphere."
    )
    message = f"Cannot create a prim of type {type_name!r}: {reason}.{hint}"
    # The refusal is about one argument, so it names it, as a schema
    # refusal does (round-3 audit m3-3: details.errors[0].field was null).
    error: Dict[str, Any] = {
        "field": "prim_type",
        "message": message,
        "type": "value_error",
        "input": type_name,
    }
    if suggestions:
        error["suggestion"] = suggestions[0]
    raise USDEditError(
        UNKNOWN_PRIM_TYPE,
        message,
        prim_type=type_name,
        suggestions=suggestions or None,
        errors=[error],
    )


@lru_cache(maxsize=1)
def _prim_metadata_keys() -> FrozenSet[str]:
    """Prim metadata and composition fields, none of which is an attribute."""
    layer = Sdf.Layer.CreateAnonymous("simul-metadata-keys")
    spec = Sdf.CreatePrimInLayer(layer, "/Probe")
    return frozenset(spec.GetMetaDataInfoKeys()) | _EXTRA_PRIM_FIELDS


def _layer_label(layer: Sdf.Layer) -> str:
    return str(layer.GetDisplayName() or layer.identifier)


def _edit_layer(stage: Usd.Stage) -> Sdf.Layer:
    return stage.GetEditTarget().GetLayer()


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


@dataclass
class PlannedWrite:
    """One validated change, ready to apply.

    Attributes:
        name: Attribute, relationship or metadata key.
        kind: ``attribute``, ``relationship`` or ``metadata``.
        value: The value converted for pxr (targets for a relationship).
        type_name: The attribute's value type; None for the other kinds.
        create: The attribute does not exist yet and is created by the write.
        time: Time code of the sample to write; None writes the default.
        xform_op: ``(op type name, precision, suffix)`` when the write
            creates a transform op.
        append_to_op_order: Add the op to the prim's ``xformOpOrder`` so it
            takes effect.
        custom: A created attribute is custom; False for one a schema
            declares, authored as the schema would author it.
        variability: Variability of a created attribute.
        warning: A ``USDIssue`` the edit reports with its result.
    """

    name: str
    kind: str
    value: Any
    type_name: Optional[Sdf.ValueTypeName] = None
    create: bool = False
    time: Optional[float] = None
    xform_op: Optional[Tuple[str, Any, str]] = None
    append_to_op_order: bool = False
    custom: bool = True
    variability: Any = Sdf.VariabilityVarying
    warning: Optional[Dict[str, Any]] = None

    @property
    def label(self) -> str:
        return f"{self.name} (metadata)" if self.kind == "metadata" else self.name


class _Problem(Exception):
    """One rejected entry, collected by the planner, with the code of a refusal that has one."""

    def __init__(self, reason: str, code: Optional[str] = None) -> None:
        super().__init__(reason)
        self.code = code


def _plan_metadata(name: str, value: Any) -> PlannedWrite:
    if name in ("active", "instanceable"):
        if not isinstance(value, bool):
            raise _Problem(f"prim metadata that takes true or false, not {describe_value(value)}")
        return PlannedWrite(name, "metadata", value)
    # kind
    if not isinstance(value, str) or not Kind.Registry.HasKind(value):
        kinds = ", ".join(str(kind) for kind in Kind.Registry.GetAllKinds())
        shown = repr(value) if isinstance(value, str) else describe_value(value)
        raise _Problem(f"prim metadata that takes a registered kind ({kinds}), not {shown}")
    return PlannedWrite(name, "metadata", value)


def _plan_relationship(name: str, value: Any) -> PlannedWrite:
    raw_targets = [value] if isinstance(value, str) else value
    if not isinstance(raw_targets, (list, tuple)) or not all(isinstance(t, str) for t in raw_targets):
        raise _Problem(
            "a relationship, which takes a target path or a list of target paths, "
            f"not {describe_value(value)}"
        )
    targets: List[Sdf.Path] = []
    for target in raw_targets:
        if not target.startswith("/") or _path_string_problem(target) is not None:
            raise _Problem(f"a relationship, and target {target!r} is not an absolute USD path")
        targets.append(Sdf.Path(target))
    return PlannedWrite(name, "relationship", targets)


def _time_samples_note(attribute: Usd.Attribute) -> Optional[str]:
    """Describe an attribute's animation, or None when it has a default value only."""
    samples = attribute.GetTimeSamples()
    clips = _clip_owner(attribute)
    if samples:
        span = f"{samples[0]:g}" if len(samples) == 1 else f"{samples[0]:g} to {samples[-1]:g}"
        source = f" from value clips on {clips}" if clips else ""
        return f"{len(samples)} time sample(s) at {span}{source}"
    if clips or attribute.ValueMightBeTimeVarying():
        return f"values from value clips on {clips}" if clips else "values from clips"
    return None


def _clip_owner(attribute: Usd.Attribute, time: Optional[Usd.TimeCode] = None) -> Optional[str]:
    """Return the prim whose value clips supply the attribute's value at ``time``, or None.

    The attribute must resolve from clips at that time (with no time, its
    strongest source of animated values); the prim is the attribute's own or
    the nearest ancestor with clip sets authored.
    """
    info = attribute.GetResolveInfo() if time is None else attribute.GetResolveInfo(time)
    if info.GetSource() != Usd.ResolveInfoSourceValueClips:
        return None
    prim = attribute.GetPrim()
    while prim and not prim.IsPseudoRoot():
        if Usd.ClipsAPI(prim).GetClips():
            return str(prim.GetPath())
        prim = prim.GetParent()
    return str(attribute.GetPrim().GetPath())


def _time_sample_layer(attribute: Usd.Attribute) -> Optional[Sdf.Layer]:
    """Return the layer whose time samples the attribute resolves, or None without samples.

    Value resolution takes every sample from the strongest spec that has
    any, so that layer owns the animation.
    """
    for spec in attribute.GetPropertyStack(Usd.TimeCode.Default()):
        if spec.layer.GetNumTimeSamplesForPath(spec.path):
            return spec.layer
    return None


class _Planner:
    """Resolves each requested entry to a typed write, or to the reason it is refused."""

    def __init__(
        self,
        prim: Optional[Usd.Prim],
        definition: Optional[Usd.PrimDefinition],
        attributes: Mapping[str, Any],
        attribute_types: Mapping[str, Any],
        time_code: Optional[float],
        xformable: bool,
        edit_layer: Optional[Sdf.Layer] = None,
        prim_path: Optional[Sdf.Path] = None,
        shading_node: bool = False,
    ) -> None:
        self.prim = prim
        self.definition = definition
        self.shading_node = shading_node
        self.prim_path = prim_path if prim_path is not None else (prim.GetPath() if prim is not None else _ROOT)
        self.attributes = attributes
        self.time_code = time_code
        self.xformable = xformable
        self.edit_layer = edit_layer
        self.explicit_op_order = "xformOpOrder" in attributes
        self.op_order: List[str] = []
        if prim is not None and xformable:
            self.op_order = [str(entry) for entry in UsdGeom.Xformable(prim).GetXformOpOrderAttr().Get() or []]
        self.problems: List[Dict[str, str]] = []
        self.requested: Dict[str, Sdf.ValueTypeName] = {}
        for name, text in attribute_types.items():
            if name not in attributes:
                self.reject(name, "attribute_types names an attribute that is not in attributes")
                continue
            try:
                self.requested[name] = coerce.parse_type_name(text)
            except coerce.ValueProblem as problem:
                self.reject(name, str(problem))

    def reject(self, name: Any, reason: str, code: Optional[str] = None) -> None:
        problem = {"attribute": str(name), "reason": reason}
        if code is not None:
            problem["code"] = code
        self.problems.append(problem)

    def plan(self) -> List[PlannedWrite]:
        properties: List[PlannedWrite] = []
        metadata: List[PlannedWrite] = []
        for name, value in self.attributes.items():
            if any(problem["attribute"] == str(name) for problem in self.problems):
                continue
            try:
                write = self._plan_one(name, value)
            except (_Problem, coerce.ValueProblem) as problem:
                self.reject(name, str(problem), getattr(problem, "code", None))
                continue
            (metadata if write.kind == "metadata" else properties).append(write)
        for write in properties:
            if write.name == "xformOpOrder":
                try:
                    self._check_op_order(write.value)
                except _Problem as problem:
                    self.reject(write.name, str(problem))
        return properties + metadata

    def _plan_one(self, name: Any, value: Any) -> PlannedWrite:
        if not isinstance(name, str) or not name:
            raise _Problem("attribute names must be non-empty strings")
        if name in SETTABLE_METADATA:
            if self.time_code is not None:
                raise _Problem(
                    "prim metadata, which has no time samples; set it in a call without time_code"
                )
            if name in self.requested:
                raise _Problem("attribute_types does not apply to prim metadata")
            return _plan_metadata(name, value)
        if not Sdf.Path.IsValidNamespacedIdentifier(name):
            raise _Problem("not a valid attribute name (use identifiers joined by ':', such as custom:mass)")
        prim, definition = self.prim, self.definition
        attribute = prim.GetAttribute(name) if prim is not None else None
        relationship = prim.GetRelationship(name) if prim is not None else None
        if relationship or (not attribute and definition is not None and definition.GetRelationshipDefinition(name)):
            if self.time_code is not None:
                raise _Problem(
                    "a relationship, which has no time samples; set it in a call without time_code"
                )
            if name in self.requested:
                raise _Problem("a relationship, which attribute_types does not apply to")
            layer = _attribute_spec_layer(relationship) if relationship else None
            if layer is not None:
                # The prim's own schema makes it a relationship, and pxr refuses the attribute spec.
                owner = _declaring_schema(prim, name) if prim is not None else None
                raise _Problem(_relationship_refusal(name, None, layer, owner), RELATIONSHIP_NOT_EDITABLE)
            return _plan_relationship(name, value)
        if self.shading_node and name.startswith(_NODE_NAMESPACES):
            declared = None  # a shader or node graph parameter, which no light schema types
        else:
            declared = schema_properties.lookup(name)
        if declared is not None and declared.relationship:
            if not attribute:
                raise _Problem(_relationship_refusal(name, declared, None), RELATIONSHIP_NOT_EDITABLE)
            if schema_properties.is_material_binding_name(name):
                # Binding resolution reads it on every prim, and pxr refuses the attribute.
                layer = _attribute_spec_layer(attribute)
                raise _Problem(_relationship_refusal(name, declared, layer), RELATIONSHIP_NOT_EDITABLE)
            # Otherwise only a schema this prim lacks makes it a relationship; pxr reads it as the
            # attribute it is, so it stays editable.
        requested = self.requested.get(name)
        if attribute:
            return self._plan_existing(name, value, attribute, requested, declared)
        attr_def = definition.GetAttributeDefinition(name) if definition is not None else None
        if attr_def:
            type_name = attr_def.GetTypeName()
            self._check_requested(name, type_name, requested)
            return self._plan_value(
                name, value, type_name, attr_def.GetMetadata("allowedTokens"), attr_def.GetVariability()
            )
        if name in _prim_metadata_keys():
            raise _Problem(
                "prim metadata, not an attribute; these tools set only the "
                f"{', '.join(SETTABLE_METADATA)} metadata"
            )
        op = coerce.parse_xform_op_name(name)
        if op is not None:
            return self._plan_new_op(name, value, op, requested)
        if declared is not None:
            return self._plan_schema_attribute(name, value, declared, requested)
        type_name = requested or coerce.infer_type(value)
        if type_name is None:
            if value is None or isinstance(value, Mapping):
                coerce.coerce(value, Sdf.ValueTypeNames.String)  # raises with the reason for these
            raise _Problem(
                f"cannot infer a USD type for a new attribute from {describe_value(value)}; name one "
                f'with attribute_types, for example {{"{name}": "float3[]"}}'
            )
        write = self._plan_value(name, value, type_name, None, Sdf.VariabilityVarying)
        write.create = True
        return write

    @staticmethod
    def _check_requested(
        name: str, type_name: Sdf.ValueTypeName, requested: Optional[Sdf.ValueTypeName]
    ) -> None:
        if requested is not None and requested != type_name:
            raise _Problem(f"a {type_name} attribute; attribute_types cannot change it to {requested}")

    def _plan_existing(
        self,
        name: str,
        value: Any,
        attribute: Usd.Attribute,
        requested: Optional[Sdf.ValueTypeName],
        declared: Optional[schema_properties.SchemaProperty] = None,
    ) -> PlannedWrite:
        type_name = attribute.GetTypeName()
        self._check_requested(name, type_name, requested)
        if self.time_code is None:
            animation = _time_samples_note(attribute)
            if animation is not None:
                raise _Problem(
                    f"animated ({animation}), so a default value would be hidden at every "
                    "frame; pass time_code to write the sample at one time"
                )
        else:
            clips = _clip_owner(attribute, resolve_time_code(self.time_code))
            if clips is not None:
                raise _Problem(
                    f"its value at time {self.time_code:g} comes from value clips (authored on {clips}); "
                    "a time sample authored in a layer is stronger than every clip, so one sample written "
                    "here would replace the whole clip animation"
                )
            source = _time_sample_layer(attribute)
            if source is not None and self.edit_layer is not None and source != self.edit_layer:
                raise _Problem(
                    f"its time samples come from {_layer_label(source)}, a weaker layer than "
                    f"{_layer_label(self.edit_layer)}; USD takes all of an attribute's samples from "
                    "the strongest layer that has any, so one sample written here would replace "
                    "that whole animation"
                )
        write = self._plan_value(
            name, value, type_name, self._allowed_tokens(name, attribute, declared), attribute.GetVariability()
        )
        try:
            op = coerce.parse_xform_op_name(name) if self.xformable else None
        except coerce.ValueProblem:
            op = None  # an existing attribute in the xformOp namespace that is not an op
        if op is not None:
            try:
                coerce.xform_op_precision(op[0], type_name)
            except coerce.ValueProblem as problem:
                raise _Problem(
                    f"exists but cannot drive the transform: {problem}; write a new op "
                    f"name such as xformOp:{op[0]}:offset instead"
                ) from None
            if not self.explicit_op_order and name not in self.op_order:
                write.append_to_op_order = True
        return write

    def _allowed_tokens(
        self, name: str, attribute: Usd.Attribute, declared: Optional[schema_properties.SchemaProperty]
    ) -> Optional[Sequence[str]]:
        """The tokens a write to an existing attribute may set; None or empty when any value goes.

        The attribute's own ``allowedTokens``, which the prim's schemas
        provide. An attribute of a schema the prim lacks (written earlier,
        with an ``UnappliedSchemaAttribute`` warning) carries none, so the
        declaring schema's list applies, as it did when the attribute was
        created: the value must still be one the schema accepts once it is
        applied.
        """
        tokens = attribute.GetMetadata("allowedTokens")
        if tokens or declared is None or declared.relationship:
            return tokens
        if self.definition is not None and self.definition.GetAttributeDefinition(name):
            return tokens  # the prim's own schema declares it, without a list
        type_name = attribute.GetTypeName()
        if type_name not in declared.type_names:
            return tokens
        return declared.declaration_for(type_name).allowed_tokens

    def _plan_schema_attribute(
        self,
        name: str,
        value: Any,
        declared: schema_properties.SchemaProperty,
        requested: Optional[Sdf.ValueTypeName],
    ) -> PlannedWrite:
        """Plan a new attribute that a schema the prim lacks declares: the schema's type, plus a warning."""
        types = declared.type_names
        listed = " or ".join(str(type_name) for type_name in types)
        if requested is not None:
            if requested not in types:
                raise _Problem(
                    f"declared as {listed} by {declared.declared_by()}; attribute_types cannot make it {requested}"
                )
            type_name = requested
        elif len(types) == 1:
            type_name = types[0]
        else:
            raise _Problem(
                f"declared as {listed} by different schemas ({declared.declared_by(limit=6)}); name the one "
                f'you mean with attribute_types, for example {{"{name}": "{types[0]}"}}'
            )
        declaration = declared.declaration_for(type_name)
        write = self._plan_value(name, value, type_name, declaration.allowed_tokens, declaration.variability)
        write.create = True
        write.custom = False
        write.variability = declaration.variability
        write.warning = self._schema_warning(name, schema_properties.restricted_to(declared, type_name), type_name)
        return write

    def _schema_warning(
        self, name: str, declared: schema_properties.SchemaProperty, type_name: Sdf.ValueTypeName
    ) -> Optional[Dict[str, Any]]:
        """The warning for an attribute written for schemas the prim lacks; ``declared`` holds those of its type.

        None for a primvar a prim type declares: primvars are inherited, so a
        descendant of that type reads it.
        """
        apis = declared.api_schemas
        if apis:
            needed = apis[0] if len(apis) == 1 else f"one of {', '.join(apis)}"
            message = (
                f"{name} is an attribute of {needed}, which {self.prim_path} does not have applied. It was "
                f"written as {type_name}, the schema's type, but USD and simulators ignore it until "
                f"{needed} is applied to the prim; simul has no tool that applies API schemas yet."
            )
        elif name.startswith("primvars:"):
            return None
        else:
            message = (
                f"{name} is an attribute of the prim type {declared.declared_by()}, not of {self.prim_path}'s "
                f"type. It was written as {type_name}, the schema's type, but nothing reads it on this prim."
            )
        return ValidationIssue(
            "warning",
            UNAPPLIED_SCHEMA_ATTRIBUTE,
            message,
            prim_path=str(self.prim_path),
            property_path=str(self.prim_path.AppendProperty(name)),
        ).to_dict()

    def _plan_new_op(
        self,
        name: str,
        value: Any,
        op: Tuple[str, str],
        requested: Optional[Sdf.ValueTypeName],
    ) -> PlannedWrite:
        op_name, suffix = op
        if not self.xformable:
            raise _Problem("a transform op, but this prim's type is not xformable")
        precision = coerce.xform_op_precision(op_name, requested)
        type_name = coerce.xform_op_value_type(op_name, precision)
        write = self._plan_value(name, value, type_name, None, Sdf.VariabilityVarying)
        write.create = True
        write.xform_op = (op_name, precision, suffix)
        # An order that already lists the op (its attribute was missing) needs no entry.
        write.append_to_op_order = not self.explicit_op_order and name not in self.op_order
        return write

    def _plan_value(
        self,
        name: str,
        value: Any,
        type_name: Sdf.ValueTypeName,
        allowed_tokens: Optional[Sequence[str]],
        variability: Any,
    ) -> PlannedWrite:
        converted = coerce.coerce(value, type_name)
        tokens = [str(token) for token in allowed_tokens or ()]
        if tokens and type_name.scalarType == Sdf.ValueTypeNames.Token:
            items = list(converted) if type_name.isArray else [converted]
            bad = [str(item) for item in items if str(item) not in tokens]
            if bad:
                raise _Problem(f"{bad[0]!r} is not an allowed token; allowed: {', '.join(tokens)}")
        if self.time_code is not None and variability == Sdf.VariabilityUniform:
            raise _Problem("uniform, so it cannot have time samples; set it in a call without time_code")
        return PlannedWrite(name, "attribute", converted, type_name=type_name, time=self.time_code)

    def _check_op_order(self, order: Sequence[Any]) -> None:
        if not self.xformable:
            raise _Problem("xformOpOrder applies to xformable prims only")
        for index, raw in enumerate(order):
            entry = str(raw)
            if entry == coerce.RESET_XFORM_STACK:
                if index:
                    raise _Problem(f"{coerce.RESET_XFORM_STACK} may only come first")
                continue
            op_name = entry[len(coerce.INVERT_PREFIX):] if entry.startswith(coerce.INVERT_PREFIX) else entry
            try:
                if coerce.parse_xform_op_name(op_name) is None:
                    raise coerce.ValueProblem(f"{entry!r} is not a transform op name")
            except coerce.ValueProblem as problem:
                raise _Problem(f"xformOpOrder entry {index}: {problem}") from None
            exists = self.prim is not None and bool(self.prim.GetAttribute(op_name))
            if not exists and op_name not in self.attributes:
                raise _Problem(
                    f"xformOpOrder lists {entry!r}, but the prim has no {op_name} attribute and this call "
                    "does not create one"
                )


def _declaring_schema(prim: Usd.Prim, name: str) -> Optional[str]:
    """Name the schema of ``prim`` that declares property ``name``: its type or an applied API."""
    registry = Usd.SchemaRegistry()
    type_name = prim.GetTypeName()
    definition = registry.FindConcretePrimDefinition(type_name) if type_name else None
    if definition is not None and definition.GetPropertyDefinition(name):
        return f"this prim's type, {type_name}"
    for applied in prim.GetAppliedSchemas():
        api, instance = Usd.SchemaRegistry.GetTypeNameAndInstance(applied)
        api_definition = registry.FindAppliedAPIPrimDefinition(api)
        if api_definition is None:
            continue
        names = api_definition.GetPropertyNames()
        if instance:
            names = [Usd.SchemaRegistry.MakeMultipleApplyNameInstance(template, instance) for template in names]
        if name in names:
            return f"its applied {applied}"
    return None


def _attribute_spec_layer(prop: Usd.Property) -> Optional[str]:
    """The layer holding the strongest attribute spec for ``prop``, or None when no layer has one."""
    for spec in prop.GetPropertyStack():
        if isinstance(spec, Sdf.AttributeSpec):
            return _layer_label(spec.layer)
    return None


def _relationship_refusal(
    name: str,
    declared: Optional[schema_properties.SchemaProperty],
    authored_in: Optional[str],
    owner: Optional[str] = None,
) -> str:
    """Why a relationship name is refused: these tools only retarget relationships a prim already has.

    Args:
        name: The property name.
        declared: The registry's declarations; None when the prim's own
            schema declares the relationship.
        authored_in: The layer that authors it as an attribute, when one
            does.
        owner: Names the prim's own schema when ``declared`` is None.
    """
    if schema_properties.is_material_binding_name(name):
        where = "UsdShade's material:binding namespace (MaterialBindingAPI)"
    elif declared is not None and declared.declared_by():
        where = declared.declared_by()
    else:
        where = owner or "this prim's schema"
    if authored_in is not None:
        return (
            f"a relationship in USD (declared by {where}) that {authored_in} authors as an attribute on this "
            f"prim, which pxr cannot read; relationships are not editable here, so correct it in {authored_in}"
        )
    return (
        f"a relationship in USD (declared by {where}), not an attribute, and relationships are not editable "
        "here: only one the prim already has can be retargeted. Writing it as an attribute would leave a "
        "property pxr cannot read"
    )


def plan_writes(
    prim: Optional[Usd.Prim],
    definition: Optional[Usd.PrimDefinition],
    attributes: Optional[Mapping[str, Any]],
    *,
    prim_path: Sdf.Path,
    attribute_types: Optional[Mapping[str, Any]] = None,
    time_code: Optional[float] = None,
    xformable: bool = False,
    edit_layer: Optional[Sdf.Layer] = None,
    shading_node: bool = False,
) -> List[PlannedWrite]:
    """Validate and type every requested change without writing anything.

    Each name resolves, in order, to settable prim metadata (``active``,
    ``instanceable``, ``kind``); a property the prim already has (authored
    or defined by its schemas); a property the type being created defines;
    a new transform op (``xformOp:<type>[:<suffix>]``), typed by the op and
    added to ``xformOpOrder``; an attribute another schema declares (an API
    schema the prim does not have, a multiple-apply instance such as
    ``drive:angular:physics:stiffness``, or for a namespaced name another
    prim type; see :func:`schema_properties.lookup`), typed by that schema
    and reported with an ``UnappliedSchemaAttribute`` warning; or a new
    custom attribute typed by ``attribute_types`` or inferred from the value.
    Other prim metadata keys are refused, since writing them as attributes
    would silently do nothing. So is a new name a schema declares a
    relationship (``material:binding``, ``physics:body0``,
    ``skel:skeleton``) unless the prim already has that relationship, and a
    relationship pxr cannot read because a layer authors it as an attribute
    (one the prim's own schema declares, or a ``material:binding``).
    Values are converted to the resolved type (see :mod:`simul.usd.coerce`).

    Args:
        prim: The existing prim, or None when it is about to be created.
        definition: The prim definition of the type being created, if any.
        attributes: Name to JSON value.
        prim_path: The prim's path, for messages.
        attribute_types: Name to USD value type name, for new attributes
            (and the precision of new transform ops).
        time_code: Write attribute values as samples at this time; None
            writes defaults, and refuses attributes that are animated.
        xformable: The prim is (or will be) a UsdGeomXformable.
        edit_layer: The layer the writes go to; a time sample is refused
            for an attribute whose animation a weaker layer or value clips
            hold.
        shading_node: The prim is (or will be) a Shader or NodeGraph
            (Material): its ``inputs:`` and ``outputs:`` are parameters
            the node types, so no schema lookup types them.

    Returns:
        The planned writes: properties in request order, then metadata.

    Raises:
        USDEditError: ``ValidationError`` with code ``InvalidAttribute``
            listing every rejected entry in ``details.problems``, or with
            code ``InvalidTimeCode``; ``UnsupportedOperation`` with code
            ``RelationshipNotEditable`` when every rejected entry is a
            relationship the tools cannot author.
    """
    if time_code is not None:
        try:
            if not coerce.is_finite_time(time_code):
                raise ValueError(f"time_code must be a finite number, got {time_code!r}")
            resolve_time_code(time_code)  # the read tools' rule, so both refuse the same values
        except (TypeError, ValueError) as exc:
            raise USDEditError(
                INVALID_TIME_CODE, f"{exc}.", prim_path=str(prim_path), time_code=repr(time_code)
            ) from None
    attributes = attributes or {}
    planner = _Planner(
        prim, definition, attributes, attribute_types or {}, time_code, xformable, edit_layer, prim_path, shading_node
    )
    writes = planner.plan()
    if planner.problems:
        problems = planner.problems
        count = len(attributes)
        listing = "; ".join(f"{p['attribute']!r}: {p['reason']}" for p in problems)
        relationships_only = all(p.get("code") == RELATIONSHIP_NOT_EDITABLE for p in problems)
        raise USDEditError(
            RELATIONSHIP_NOT_EDITABLE if relationships_only else INVALID_ATTRIBUTE,
            f"Rejected {len(problems)} of {count} attribute(s) on {prim_path}; nothing was changed. {listing}.",
            prim_path=str(prim_path),
            problems=problems,
        )
    return writes


# ---------------------------------------------------------------------------
# Writing and rollback
# ---------------------------------------------------------------------------


class _WriteFailure(Exception):
    """A write pxr rejected despite the plan; triggers the rollback."""

    def __init__(self, name: Optional[str], reason: str) -> None:
        super().__init__(reason)
        self.name = name
        self.reason = reason


def _write_attribute(prim: Usd.Prim, write: PlannedWrite) -> None:
    time = resolve_time_code(write.time)
    if write.xform_op is not None:
        op_name, precision, suffix = write.xform_op
        if write.append_to_op_order:
            op = UsdGeom.Xformable(prim).AddXformOp(coerce.xform_op_type(op_name), precision, suffix)
            attr = op.GetAttr() if op else None
        else:
            attr = prim.CreateAttribute(write.name, write.type_name, False)
    else:
        attr = prim.GetAttribute(write.name)
        if not attr:
            attr = prim.CreateAttribute(write.name, write.type_name, write.custom, write.variability)
    if not attr or not attr.Set(write.value, time):
        raise _WriteFailure(write.name, "pxr rejected the value")
    if write.append_to_op_order and write.xform_op is None:
        order_attr = UsdGeom.Xformable(prim).GetXformOpOrderAttr()
        order = [str(entry) for entry in order_attr.Get() or []]
        if not order_attr.Set(order + [write.name]):
            raise _WriteFailure("xformOpOrder", f"pxr rejected adding {write.name}")


def _apply_write(prim: Usd.Prim, write: PlannedWrite) -> None:
    """Apply one planned write, raising _WriteFailure when pxr rejects it."""
    try:
        if write.kind == "attribute":
            _write_attribute(prim, write)
        elif write.kind == "relationship":
            rel = prim.GetRelationship(write.name) or prim.CreateRelationship(write.name, True)
            if not rel or not rel.SetTargets(write.value):
                raise _WriteFailure(write.name, "pxr rejected the targets")
        elif write.name == "active":
            if not prim.SetActive(write.value):
                raise _WriteFailure(write.name, "pxr rejected the active metadata")
        elif write.name == "instanceable":
            if not prim.SetInstanceable(write.value):
                raise _WriteFailure(write.name, "pxr rejected the instanceable metadata")
        elif write.name == "kind":
            if not Usd.ModelAPI(prim).SetKind(write.value):
                raise _WriteFailure(write.name, "pxr rejected the kind metadata")
    except _WriteFailure:
        raise
    except Exception as exc:  # pxr raises Tf.ErrorException; anything else is a failure too
        raise _WriteFailure(write.name, tf_error_reason(exc)) from exc


def _apply_writes(prim: Usd.Prim, writes: Sequence[PlannedWrite]) -> None:
    for write in writes:
        _apply_write(prim, write)


def _summary(writes: Sequence[PlannedWrite]) -> Dict[str, Any]:
    """The ``changed`` names, ``created_attributes`` types and ``warnings`` an edit reports."""
    changed = [write.label for write in writes]
    if any(write.append_to_op_order for write in writes) and "xformOpOrder" not in changed:
        changed.append("xformOpOrder")
    created = {write.name: str(write.type_name) for write in writes if write.create}
    warnings = [write.warning for write in writes if write.warning is not None]
    return {"changed": changed, "created_attributes": created, "warnings": warnings}


def _owner_spec(layer: Sdf.Layer, path: Sdf.Path) -> Sdf.PrimSpec:
    parent = path.GetParentPath()
    return layer.pseudoRoot if parent == _ROOT else layer.GetPrimAtPath(parent)


class EditJournal:
    """Snapshot of the specs one prim edit can touch in one layer, restorable on failure.

    An edit of the prim at ``path`` can author specs for ``path`` itself, its
    properties and metadata, and for ancestors that had no spec in the layer
    (or, for ``DefinePrim``, that were not defined). The journal records, for
    each ancestor, whether it had a spec and its specifier and type name; for
    the prim, its metadata fields and a copy of each property spec, or with
    ``subtree`` (deletes) a copy of the whole subtree and its index among its
    siblings. The snapshot is proportional to what the edit can change, not
    to the layer.
    """

    def __init__(self, layer: Sdf.Layer, path: Sdf.Path, *, subtree: bool = False) -> None:
        self._layer = layer
        self._path = path
        self._subtree = subtree
        self._chain: List[Tuple[Sdf.Path, Optional[Tuple[Any, str]]]] = []
        for prefix in path.GetPrefixes():
            spec = layer.GetPrimAtPath(prefix)
            self._chain.append((prefix, None if spec is None else (spec.specifier, spec.typeName)))
        self._info: Dict[str, Any] = {}
        self._properties: List[str] = []
        self._sibling_index: Optional[int] = None
        self._backup: Optional[Sdf.Layer] = None
        spec = layer.GetPrimAtPath(path)
        if spec is None:
            return
        backup = Sdf.Layer.CreateAnonymous("simul-edit-journal")
        if path.GetParentPath() != _ROOT:
            Sdf.CreatePrimInLayer(backup, path.GetParentPath())
        if subtree:
            Sdf.CopySpec(layer, path, backup, path)
            self._sibling_index = list(_owner_spec(layer, path).nameChildren.keys()).index(path.name)
        else:
            Sdf.CreatePrimInLayer(backup, path)
            self._info = {key: spec.GetInfo(key) for key in spec.ListInfoKeys()}
            self._properties = list(spec.properties.keys())
            for name in self._properties:
                prop_path = path.AppendProperty(name)
                Sdf.CopySpec(layer, prop_path, backup, prop_path)
        self._backup = backup

    def restore(self) -> None:
        """Put every recorded spec back as it was when the journal was made."""
        layer = self._layer
        with Sdf.ChangeBlock():
            for prefix, head in self._chain:
                spec = layer.GetPrimAtPath(prefix)
                if head is None:
                    # This prefix was created by the edit: drop it and all below.
                    if spec is not None:
                        del _owner_spec(layer, prefix).nameChildren[prefix.name]
                    return
                if spec is None:
                    continue
                specifier, type_name = head
                if spec.specifier != specifier:
                    spec.specifier = specifier
                if spec.typeName != type_name:
                    spec.typeName = type_name
            if self._subtree:
                self._restore_subtree()
            else:
                self._restore_prim()

    def _restore_prim(self) -> None:
        layer, path, backup = self._layer, self._path, self._backup
        spec = layer.GetPrimAtPath(path)
        if spec is None or backup is None:
            return
        for name in list(spec.properties.keys()):
            if name not in self._properties:
                spec.RemoveProperty(spec.properties[name])
        for name in self._properties:
            prop_path = path.AppendProperty(name)
            Sdf.CopySpec(backup, prop_path, layer, prop_path)
        for key in list(spec.ListInfoKeys()):
            if key not in self._info:
                spec.ClearInfo(key)
        for key, value in self._info.items():
            if spec.GetInfo(key) != value:
                spec.SetInfo(key, value)

    def _restore_subtree(self) -> None:
        layer, path, backup = self._layer, self._path, self._backup
        if backup is None:
            return
        if layer.GetPrimAtPath(path) is not None:
            del _owner_spec(layer, path).nameChildren[path.name]
        Sdf.CopySpec(backup, path, layer, path)
        owner = _owner_spec(layer, path)
        if self._sibling_index is None or list(owner.nameChildren.keys()).index(path.name) == self._sibling_index:
            return
        # CopySpec appends; move the prim back to its old index through a
        # temporary holder (a move keeps the spec's data as it is).
        holder_name = f"simul_restore_{uuid.uuid4().hex}"
        holder = Sdf.PrimSpec(layer, holder_name, Sdf.SpecifierOver)
        holder.nameChildren.append(layer.GetPrimAtPath(path))
        owner.nameChildren.insert(self._sibling_index, layer.GetPrimAtPath(holder.path.AppendChild(path.name)))
        del layer.rootPrims[holder_name]


def _rolled_back(action: str, path: Sdf.Path, failure: _WriteFailure) -> USDEditError:
    where = f" while writing {failure.name!r}" if failure.name else ""
    return USDEditError(
        EDIT_FAILED,
        f"{action} {path} failed{where}: {failure.reason}. The edit was rolled back; nothing was changed.",
        prim_path=str(path),
        attribute=failure.name,
        reason=failure.reason,
        rolled_back=True,
    )


# ---------------------------------------------------------------------------
# Edits
# ---------------------------------------------------------------------------


def _is_shading_node(tf_type: Tf.Type) -> bool:
    """Whether a prim type is a Shader or a NodeGraph (Material derives from NodeGraph)."""
    return any(tf_type.IsA(node_type) for node_type in _SHADING_NODE_TYPES)


def _check_creatable_under(stage: Usd.Stage, path: Sdf.Path) -> None:
    """Refuse a new prim beneath an instance or an inactive prim."""
    ancestor = path.GetParentPath()
    while ancestor != _ROOT and not stage.GetPrimAtPath(ancestor):
        ancestor = ancestor.GetParentPath()
    if ancestor == _ROOT:
        return
    prim = stage.GetPrimAtPath(ancestor)
    if prim.IsInstance() or prim.IsInstanceProxy():
        instance = prim if prim.IsInstance() else _instance_of(prim)
        owner = str(instance.GetPath()) if instance else str(ancestor)
        raise USDEditError(
            UNSUPPORTED,
            f"Cannot create {path}: {owner} is instanceable, so the prims beneath it come from a "
            "shared prototype and cannot be authored. Make it editable first with "
            f'update_prim_attributes {{"instanceable": false}} on {owner}.',
            prim_path=str(path),
            instance=owner,
        )
    if not prim.IsActive():
        raise USDEditError(
            UNSUPPORTED,
            f"Cannot create {path}: its ancestor {ancestor} is inactive, and the children of an "
            'inactive prim are not composed. Activate it first with update_prim_attributes {"active": true}.',
            prim_path=str(path),
            inactive_ancestor=str(ancestor),
        )


def create_prim(
    stage: Usd.Stage,
    prim_path: Any,
    prim_type: str,
    attributes: Optional[Mapping[str, Any]] = None,
    *,
    redefine_existing: bool = False,
    attribute_types: Optional[Mapping[str, Any]] = None,
    time_code: Optional[float] = None,
) -> Dict[str, Any]:
    """Define a prim of a registered concrete type, with its initial attributes.

    Args:
        stage: The stage to edit (its edit target receives the specs).
        prim_path: Absolute path of the new prim; missing ancestors are
            defined as typeless prims.
        prim_type: A concrete prim type the schema registry knows.
        attributes: Initial attribute values, relationship targets and the
            ``active`` / ``instanceable`` / ``kind`` metadata.
        redefine_existing: Allow ``prim_path`` to name an existing prim,
            whose type is changed to ``prim_type``.
        attribute_types: USD value type per new attribute name.
        time_code: Write the attribute values as samples at this time.

    Returns:
        ``prim_path``, ``prim_type``, ``redefined``, the ``changed`` names,
        ``created_attributes`` (name to type) and ``time_code``.

    Raises:
        USDEditError: For any refusal or failure; nothing was changed.
    """
    path = parse_prim_path(prim_path)
    if path == _ROOT:
        raise USDEditError(
            INVALID_PRIM_PATH,
            "Cannot create '/': it is the stage pseudo-root. Name the new prim, for example /World.",
            prim_path=str(prim_path),
        )
    definition = check_prim_type(prim_type)
    existing = stage.GetPrimAtPath(path)
    if existing and existing.IsValid():
        refuse_instance_proxy(existing, "redefine")
        if not redefine_existing:
            current = existing.GetTypeName() or "typeless"
            raise USDEditError(
                REFUSED,
                f"A prim already exists at {path} (type {current}). create_prim does not "
                f"replace it; pass redefine_existing=true to change its type to {prim_type} "
                "(it keeps its children, properties and composition arcs), or use "
                "update_prim_attributes to change its attributes.",
                prim_path=str(path),
                existing_type=current,
                requested_type=prim_type,
                override="redefine_existing",
            )
    else:
        existing = None
        _check_creatable_under(stage, path)
    writes = plan_writes(
        existing,
        definition,
        attributes,
        prim_path=path,
        attribute_types=attribute_types,
        time_code=time_code,
        xformable=Usd.SchemaRegistry.GetTypeFromName(prim_type).IsA(_XFORMABLE_TYPE),
        edit_layer=_edit_layer(stage),
        shading_node=_is_shading_node(Usd.SchemaRegistry.GetTypeFromName(prim_type)),
    )
    journal = EditJournal(_edit_layer(stage), path)
    try:
        try:
            prim = stage.DefinePrim(path, prim_type)
        except Exception as exc:
            raise _WriteFailure(None, tf_error_reason(exc)) from exc
        if not prim or prim.GetTypeName() != prim_type:
            raise _WriteFailure(None, f"the prim could not be defined as {prim_type}")
        _apply_writes(prim, writes)
    except _WriteFailure as failure:
        journal.restore()
        raise _rolled_back("Creating", path, failure) from failure
    return {
        "prim_path": str(path),
        "prim_type": prim_type,
        "redefined": existing is not None,
        "time_code": time_code,
        **_summary(writes),
    }


def update_prim(
    stage: Usd.Stage,
    prim_path: Any,
    attributes: Optional[Mapping[str, Any]],
    *,
    attribute_types: Optional[Mapping[str, Any]] = None,
    time_code: Optional[float] = None,
) -> Dict[str, Any]:
    """Set attributes, relationship targets and metadata on an existing prim, all or nothing.

    Args:
        stage: The stage to edit.
        prim_path: Absolute path of an existing prim.
        attributes: Name to JSON value; see :func:`plan_writes`.
        attribute_types: USD value type per new attribute name.
        time_code: Write the attribute values as samples at this time;
            without it, animated attributes are refused.

    Returns:
        ``prim_path``, the ``changed`` names, ``created_attributes`` and
        ``time_code``.

    Raises:
        USDEditError: For any refusal or failure; nothing was changed.
    """
    path = parse_prim_path(prim_path)
    if path == _ROOT:
        raise USDEditError(
            INVALID_PRIM_PATH,
            "'/' is the stage pseudo-root and holds no attributes; pass a prim path such as /World.",
            prim_path=str(prim_path),
        )
    prim = require_prim(stage, path)
    refuse_instance_proxy(prim, "edit")
    writes = plan_writes(
        prim,
        prim.GetPrimDefinition(),
        attributes,
        prim_path=path,
        attribute_types=attribute_types,
        time_code=time_code,
        xformable=prim.IsA(UsdGeom.Xformable),
        edit_layer=_edit_layer(stage),
        shading_node=_is_shading_node(Usd.SchemaRegistry.GetTypeFromName(prim.GetTypeName())),
    )
    if writes:
        journal = EditJournal(_edit_layer(stage), path)
        try:
            _apply_writes(prim, writes)
        except _WriteFailure as failure:
            journal.restore()
            raise _rolled_back("Updating", path, failure) from failure
    return {"prim_path": str(path), "time_code": time_code, **_summary(writes)}


def surviving_opinions(stage: Usd.Stage, prim: Usd.Prim, layer: Sdf.Layer) -> List[Dict[str, str]]:
    """List the opinions that would keep ``prim`` alive after ``layer``'s specs for it are removed.

    Removing the prim's subtree from ``layer`` also removes every
    composition arc authored there on the prim, and everything those arcs
    bring in. What survives: specs in the other layers of the root layer
    stack (sublayers, the session layer), arcs authored on an ancestor (the
    prim is part of a referenced or payloaded asset), and arcs on the prim
    authored in another layer.

    Args:
        stage: The prim's stage.
        prim: The prim to delete.
        layer: The layer the delete edits.

    Returns:
        One ``{"layer", "via"}`` entry per surviving source; empty when the
        delete removes the prim.
    """
    root_stack = set(stage.GetLayerStack(True))
    session = stage.GetSessionLayer()
    survivors: List[Dict[str, str]] = []
    for spec in prim.GetPrimStack():
        if spec.layer in root_stack and spec.layer != layer:
            via = "the session layer" if spec.layer == session else "a sublayer"
            survivors.append({"layer": _layer_label(spec.layer), "via": via})
    for arc in Usd.PrimCompositionQuery(prim).GetCompositionArcs():
        if arc.GetArcType() == Pcp.ArcTypeRoot or not arc.HasSpecs():
            continue
        arc_name = str(arc.GetArcType()).rsplit("ArcType", 1)[-1].lower()
        target = arc.GetTargetNode().layerStack.identifier.rootLayer
        if arc.IsAncestral():
            survivors.append(
                {"layer": _layer_label(target), "via": f"the {arc_name} on {arc.GetIntroducingPrimPath()}"}
            )
        elif arc.IsIntroducedInRootLayerStack() and arc.GetIntroducingLayer() != layer:
            survivors.append(
                {
                    "layer": _layer_label(target),
                    "via": f"a {arc_name} authored in {_layer_label(arc.GetIntroducingLayer())}",
                }
            )
    unique: List[Dict[str, str]] = []
    for entry in survivors:
        if entry not in unique:
            unique.append(entry)
    return unique


def _still_composed(stage: Usd.Stage, path: Sdf.Path) -> bool:
    """Whether the stage still composes a prim at ``path`` (the delete post-check)."""
    return bool(stage.GetPrimAtPath(path))


def delete_prim(stage: Usd.Stage, prim_path: Any, *, allow_root_delete: bool = False) -> Dict[str, Any]:
    """Remove a prim and its subtree from the edit target, only when that removes it from the stage.

    Args:
        stage: The stage to edit.
        prim_path: Absolute path of an existing prim.
        allow_root_delete: Permit deleting the stage's default prim (or an
            ancestor of it).

    Returns:
        ``prim_path``, the deleted ``prim_type`` and ``removed_prims`` (the
        prim plus its descendants).

    Raises:
        USDEditError: For any refusal or failure; nothing was changed.
    """
    path = parse_prim_path(prim_path)
    if path == _ROOT:
        raise USDEditError(
            REFUSED,
            "Refusing to delete the pseudo-root '/': it is the stage itself, not a prim. Delete a "
            "child prim instead, or unload the stage.",
            prim_path=str(prim_path),
        )
    prim = require_prim(stage, path)
    refuse_instance_proxy(prim, "delete")
    default_prim = stage.GetDefaultPrim()
    if default_prim and default_prim.GetPath().HasPrefix(path) and not allow_root_delete:
        what = "the stage's default prim" if default_prim.GetPath() == path else (
            f"an ancestor of the stage's default prim {default_prim.GetPath()}"
        )
        raise USDEditError(
            REFUSED,
            f"Refusing to delete {path}: it is {what}, which holds the scene. Pass "
            "allow_root_delete=true to delete it anyway.",
            prim_path=str(path),
            default_prim=str(default_prim.GetPath()),
            override="allow_root_delete",
        )
    layer = _edit_layer(stage)
    survivors = surviving_opinions(stage, prim, layer)
    if survivors:
        sources = "; ".join(f"{entry['layer']} (via {entry['via']})" for entry in survivors)
        raise USDEditError(
            UNSUPPORTED,
            f"Cannot delete {path} from {_layer_label(layer)}: it also has opinions from {sources}, "
            f"which removing {_layer_label(layer)}'s specs cannot remove (it would only drop the local "
            "overrides). Deactivate it instead with update_prim_attributes "
            '{"active": false}.',
            prim_path=str(path),
            edit_layer=_layer_label(layer),
            surviving_opinions=survivors,
        )
    prim_type = prim.GetTypeName()
    removed = sum(1 for _ in Usd.PrimRange(prim))
    edit = Sdf.BatchNamespaceEdit()
    edit.Add(Sdf.NamespaceEdit.Remove(path))
    # CanApply answers True, or (False, [NamespaceEditDetail, ...]).
    verdict = layer.CanApply(edit)
    if isinstance(verdict, tuple):
        allowed, edit_details = bool(verdict[0]), list(verdict[1]) if len(verdict) > 1 else []
    else:
        allowed, edit_details = bool(verdict), []
    if not allowed:
        reason = "; ".join(str(getattr(detail, "reason", detail)) for detail in edit_details) or "refused by Sdf"
        raise USDEditError(
            EDIT_FAILED,
            f"Cannot delete {path}: {reason}. Nothing was changed.",
            prim_path=str(path),
            reason=reason,
        )
    journal = EditJournal(layer, path, subtree=True)
    try:
        try:
            applied = layer.Apply(edit)
        except Exception as exc:
            raise _WriteFailure(None, tf_error_reason(exc)) from exc
        if not applied:
            raise _WriteFailure(None, "Sdf could not apply the removal")
        if _still_composed(stage, path):
            raise _WriteFailure(None, "the prim still has opinions after its specs were removed")
    except _WriteFailure as failure:
        journal.restore()
        raise _rolled_back("Deleting", path, failure) from failure
    return {"prim_path": str(path), "prim_type": prim_type, "removed_prims": removed}


__all__ = [
    "CODE_ERROR_TYPES",
    "EDIT_FAILED",
    "ENGINE",
    "VALIDATION",
    "INVALID_ATTRIBUTE",
    "INVALID_PRIM_PATH",
    "INVALID_TIME_CODE",
    "NOT_FOUND",
    "REFUSED",
    "RELATIONSHIP_NOT_EDITABLE",
    "UNAPPLIED_SCHEMA_ATTRIBUTE",
    "SETTABLE_METADATA",
    "UNKNOWN_PRIM_TYPE",
    "UNSUPPORTED",
    "EditJournal",
    "PlannedWrite",
    "USDEditError",
    "check_prim_type",
    "concrete_prim_types",
    "create_prim",
    "delete_prim",
    "describe_value",
    "missing_prim_error",
    "parse_prim_path",
    "plan_writes",
    "refuse_instance_proxy",
    "require_prim",
    "surviving_opinions",
    "tf_error_reason",
    "update_prim",
]
