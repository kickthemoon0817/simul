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
import re
import uuid
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

from pxr import Pcp

from . import lookup
from ._pxr import Kind, Sdf, Tf, Usd

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
CODE_ERROR_TYPES: Dict[str, str] = {
    INVALID_PRIM_PATH: VALIDATION,
    UNKNOWN_PRIM_TYPE: VALIDATION,
    INVALID_ATTRIBUTE: VALIDATION,
    EDIT_FAILED: ENGINE,
}

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

_TF_ERROR = re.compile(
    r"Error in '[^']*' at line \d+ in file .*? : '(?P<message>.*?)'(?=\s*(?:Error in|$))",
    re.S,
)

_ROOT = Sdf.Path.absoluteRootPath


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


def tf_error_reason(exc: BaseException) -> str:
    """Return the human part of a pxr error, without C++ function and file names.

    Args:
        exc: Usually a ``Tf.ErrorException``; any exception is accepted.

    Returns:
        The messages pxr attached, joined with ``; ``, or the exception text.
    """
    text = str(exc)
    messages = [match.group("message").strip() for match in _TF_ERROR.finditer(text)]
    if messages:
        return "; ".join(messages)
    return text.strip() or type(exc).__name__


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
    raise USDEditError(
        UNKNOWN_PRIM_TYPE,
        f"Cannot create a prim of type {type_name!r}: {reason}.{hint}",
        prim_type=type_name,
        suggestions=suggestions or None,
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
        create: The attribute does not exist yet and is created as custom.
    """

    name: str
    kind: str
    value: Any
    type_name: Optional[Sdf.ValueTypeName] = None
    create: bool = False

    @property
    def label(self) -> str:
        return f"{self.name} (metadata)" if self.kind == "metadata" else self.name


class _Problem(Exception):
    """One rejected entry, collected by the planner."""


def _infer_value_type(value: Any) -> Optional[Sdf.ValueTypeName]:
    """Pick a value type for a new attribute from its JSON value."""
    if isinstance(value, bool):
        return Sdf.ValueTypeNames.Bool
    if isinstance(value, int):
        return Sdf.ValueTypeNames.Int
    if isinstance(value, float):
        return Sdf.ValueTypeNames.Float
    if isinstance(value, str):
        return Sdf.ValueTypeNames.String
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        if all(isinstance(v, bool) for v in value):
            return Sdf.ValueTypeNames.BoolArray
        if all(isinstance(v, int) and not isinstance(v, bool) for v in value):
            return Sdf.ValueTypeNames.IntArray
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
            return {
                2: Sdf.ValueTypeNames.Float2,
                3: Sdf.ValueTypeNames.Float3,
                4: Sdf.ValueTypeNames.Float4,
            }.get(len(value), Sdf.ValueTypeNames.FloatArray)
        if all(isinstance(v, str) for v in value):
            return Sdf.ValueTypeNames.StringArray
    return None


def _to_pxr(value: Any, type_name: Sdf.ValueTypeName) -> Any:
    """Shape a JSON value the way pxr's Python conversion expects for ``type_name``.

    Fixed-size vector types take a tuple; arrays take a list whose vector
    elements are tuples.
    """
    if type_name.isArray:
        if isinstance(value, (list, tuple)):
            return [tuple(item) if isinstance(item, list) else item for item in value]
        return value
    if isinstance(value, list):
        return tuple(value)
    return value


class _ConversionCheck:
    """Dry-run ``Set`` calls on a scratch in-memory stage."""

    def __init__(self) -> None:
        self._prim: Optional[Usd.Prim] = None
        self._stage: Optional[Usd.Stage] = None
        self._count = 0

    def accepts(self, type_name: Sdf.ValueTypeName, value: Any) -> bool:
        if self._prim is None:
            self._stage = Usd.Stage.CreateInMemory("simul-conversion-check.usda")
            self._prim = self._stage.DefinePrim("/Check")
        self._count += 1
        attr = self._prim.CreateAttribute(f"v{self._count}", type_name)
        try:
            return bool(attr.Set(value))
        except Tf.ErrorException:
            return False


def _plan_metadata(name: str, value: Any) -> PlannedWrite:
    if name in ("active", "instanceable"):
        if not isinstance(value, bool):
            raise _Problem(f"{name} is prim metadata and takes true or false, not {describe_value(value)}")
        return PlannedWrite(name, "metadata", value)
    # kind
    if not isinstance(value, str) or not Kind.Registry.HasKind(value):
        kinds = ", ".join(str(kind) for kind in Kind.Registry.GetAllKinds())
        shown = repr(value) if isinstance(value, str) else describe_value(value)
        raise _Problem(f"kind is prim metadata and takes a registered kind ({kinds}), not {shown}")
    return PlannedWrite(name, "metadata", value)


def _plan_relationship(name: str, value: Any) -> PlannedWrite:
    raw_targets = [value] if isinstance(value, str) else value
    if not isinstance(raw_targets, (list, tuple)) or not all(isinstance(t, str) for t in raw_targets):
        raise _Problem(
            f"{name} is a relationship; it takes a target path or a list of target paths, "
            f"not {describe_value(value)}"
        )
    targets: List[Sdf.Path] = []
    for target in raw_targets:
        if not target.startswith("/") or _path_string_problem(target) is not None:
            raise _Problem(f"{name} is a relationship; target {target!r} is not an absolute USD path")
        targets.append(Sdf.Path(target))
    return PlannedWrite(name, "relationship", targets)


def _reject_non_value(value: Any) -> None:
    """Refuse the JSON values no attribute type can hold."""
    if value is None:
        raise _Problem("null is not a value (clearing an authored value is not supported)")
    if isinstance(value, Mapping):
        raise _Problem("a JSON object is not an attribute value")


def _plan_attribute(
    name: str,
    value: Any,
    type_name: Sdf.ValueTypeName,
    allowed_tokens: Optional[Sequence[str]],
    create: bool,
    check: _ConversionCheck,
) -> PlannedWrite:
    _reject_non_value(value)
    converted = _to_pxr(value, type_name)
    if not check.accepts(type_name, converted):
        raise _Problem(f"{describe_value(value)} cannot be stored in a {type_name} attribute")
    tokens = [str(token) for token in allowed_tokens or ()]
    if tokens and type_name.scalarType == Sdf.ValueTypeNames.Token:
        values = converted if type_name.isArray else [converted]
        bad = [str(item) for item in values if str(item) not in tokens]
        if bad:
            raise _Problem(f"{bad[0]!r} is not an allowed token; allowed: {', '.join(tokens)}")
    return PlannedWrite(name, "attribute", converted, type_name=type_name, create=create)


def plan_writes(
    prim: Optional[Usd.Prim],
    definition: Optional[Usd.PrimDefinition],
    attributes: Optional[Mapping[str, Any]],
    *,
    prim_path: Sdf.Path,
) -> List[PlannedWrite]:
    """Validate every requested change without writing anything.

    Each name resolves, in order, to settable prim metadata (``active``,
    ``instanceable``, ``kind``); a property the prim already has (authored
    or defined by its schemas); a property the target type's schema defines
    (when creating); or a new custom attribute whose type is inferred from
    the value. Other prim metadata keys are refused, since writing them as
    attributes would silently do nothing.

    Args:
        prim: The existing prim, or None when it is about to be created.
        definition: The prim definition of the type being created, if any.
        attributes: Name to JSON value.
        prim_path: The prim's path, for messages.

    Returns:
        The planned writes: properties in request order, then metadata.

    Raises:
        USDEditError: ``ValidationError`` (code ``InvalidAttribute``) listing
            every rejected entry in ``details.problems``.
    """
    properties: List[PlannedWrite] = []
    metadata: List[PlannedWrite] = []
    problems: List[Dict[str, str]] = []
    check = _ConversionCheck()
    for name, value in (attributes or {}).items():
        try:
            if not isinstance(name, str) or not name:
                raise _Problem("attribute names must be non-empty strings")
            if name in SETTABLE_METADATA:
                metadata.append(_plan_metadata(name, value))
                continue
            if not Sdf.Path.IsValidNamespacedIdentifier(name):
                raise _Problem(
                    "not a valid attribute name (use identifiers joined by ':', such as custom:mass)"
                )
            relationship = prim.GetRelationship(name) if prim is not None else None
            attribute = prim.GetAttribute(name) if prim is not None else None
            if relationship:
                properties.append(_plan_relationship(name, value))
                continue
            if attribute:
                properties.append(
                    _plan_attribute(
                        name,
                        value,
                        attribute.GetTypeName(),
                        attribute.GetMetadata("allowedTokens"),
                        False,
                        check,
                    )
                )
                continue
            if definition is not None:
                if definition.GetRelationshipDefinition(name):
                    properties.append(_plan_relationship(name, value))
                    continue
                attr_def = definition.GetAttributeDefinition(name)
                if attr_def:
                    properties.append(
                        _plan_attribute(
                            name,
                            value,
                            attr_def.GetTypeName(),
                            attr_def.GetMetadata("allowedTokens"),
                            False,
                            check,
                        )
                    )
                    continue
            if name in _prim_metadata_keys():
                raise _Problem(
                    f"{name} is prim metadata, not an attribute; these tools set only the "
                    f"{', '.join(SETTABLE_METADATA)} metadata"
                )
            _reject_non_value(value)
            type_name = _infer_value_type(value)
            if type_name is None:
                raise _Problem(f"cannot infer a USD type for a new attribute from {describe_value(value)}")
            properties.append(_plan_attribute(name, value, type_name, None, True, check))
        except _Problem as problem:
            problems.append({"attribute": str(name), "reason": str(problem)})
    if problems:
        count = len(attributes or {})
        listing = "; ".join(f"{p['attribute']!r}: {p['reason']}" for p in problems)
        raise USDEditError(
            INVALID_ATTRIBUTE,
            f"Rejected {len(problems)} of {count} attribute(s) on {prim_path}; nothing was changed. {listing}.",
            prim_path=str(prim_path),
            problems=problems,
        )
    return properties + metadata


# ---------------------------------------------------------------------------
# Writing and rollback
# ---------------------------------------------------------------------------


class _WriteFailure(Exception):
    """A write pxr rejected despite the plan; triggers the rollback."""

    def __init__(self, name: Optional[str], reason: str) -> None:
        super().__init__(reason)
        self.name = name
        self.reason = reason


def _apply_write(prim: Usd.Prim, write: PlannedWrite) -> None:
    """Apply one planned write, raising _WriteFailure when pxr rejects it."""
    try:
        if write.kind == "attribute":
            attr = prim.GetAttribute(write.name)
            if not attr:
                attr = prim.CreateAttribute(write.name, write.type_name, True)
            if not attr or not attr.Set(write.value):
                raise _WriteFailure(write.name, "pxr rejected the value")
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

    Returns:
        ``prim_path``, ``prim_type``, ``redefined`` and the ``changed`` names.

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
    writes = plan_writes(existing, definition, attributes, prim_path=path)
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
        "changed": [write.label for write in writes],
    }


def update_prim(
    stage: Usd.Stage, prim_path: Any, attributes: Optional[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Set attributes, relationship targets and metadata on an existing prim, all or nothing.

    Args:
        stage: The stage to edit.
        prim_path: Absolute path of an existing prim.
        attributes: Name to JSON value; see :func:`plan_writes`.

    Returns:
        ``prim_path`` and the ``changed`` names.

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
    writes = plan_writes(prim, prim.GetPrimDefinition(), attributes, prim_path=path)
    if not writes:
        return {"prim_path": str(path), "changed": []}
    journal = EditJournal(_edit_layer(stage), path)
    try:
        _apply_writes(prim, writes)
    except _WriteFailure as failure:
        journal.restore()
        raise _rolled_back("Updating", path, failure) from failure
    return {"prim_path": str(path), "changed": [write.label for write in writes]}


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
    "NOT_FOUND",
    "REFUSED",
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
