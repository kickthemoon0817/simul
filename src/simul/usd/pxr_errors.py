"""pxr failures as the headless USD tools report them.

pxr raises ``Tf.ErrorException`` with text such as ``Error in
'pxrInternal_v0_26_11__pxrReserved__::PcpComposeTargetPaths' at line 625 in
file /Users/runner/... : '<the reason>'``. :func:`tf_error_reason` keeps only
the reason, and :class:`PxrError` carries it out of a tool call, so no envelope
shows pxr's C++ function names or build paths.

A file can hold a property pxr cannot read, such as ``material:binding``
authored as an attribute where USD expects a relationship. The read helpers
here turn such a failure into a warning about that one property, so the rest
of the prim is still reported.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ._pxr import Sdf, Tf, Usd, UsdShade
from .validation import ValidationIssue

_TF_ERROR = re.compile(
    r"Error in '[^']*' at line \d+ in file .*? : '(?P<message>.*?)'(?=\s*(?:Error in|$))",
    re.S,
)

# A spec pxr names in a message, ``@<layer identifier>@</Prim.property>``.
_SPEC_REFERENCE = re.compile(r"@(?P<layer>[^@\s]*)@<(?P<path>[^>]*)>")

# Warning code for a property a read could not resolve.
UNREADABLE_PROPERTY = "UnreadableProperty"


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


class PxrError(Exception):
    """A pxr error that escaped a USD tool, reported as ``EngineError`` with pxr's reason only.

    Attributes:
        error_type: ``EngineError``, the envelope type.
        reason: pxr's message, without its C++ function and file names.
        details: ``exception_type`` keeps the name of pxr's exception.
    """

    error_type = "EngineError"

    def __init__(self, tool_name: str, exc: BaseException) -> None:
        self.reason = tf_error_reason(exc)
        super().__init__(f"{tool_name} failed in pxr: {self.reason}")
        self.details: Dict[str, Any] = {"exception_type": type(exc).__name__}


def is_pxr_error(exc: BaseException) -> bool:
    """Whether ``exc`` is the exception pxr raises for its own errors."""
    return isinstance(exc, Tf.ErrorException)


def _read_failure(exc: BaseException) -> Tuple[str, Optional[str]]:
    """pxr's reason for a failed read, and the file of the first spec it names.

    pxr names a spec as ``@<layer identifier>@</Prim.property>``; the reason
    keeps the path (a path in that layer, which a reference may map to a
    different path on the stage) and the layer is returned as its file path,
    without the format arguments the session adds to a root layer's
    identifier.
    """
    reason = tf_error_reason(exc)
    match = _SPEC_REFERENCE.search(reason)
    layer = Sdf.Layer.SplitIdentifier(match.group("layer"))[0] if match else None
    return _SPEC_REFERENCE.sub(lambda found: found.group("path"), reason), layer or None


def unreadable_issue(
    prim: Usd.Prim, message: str, property_name: Optional[str] = None, layer: Optional[str] = None
) -> Dict[str, Any]:
    """A warning (``USDIssue`` fields) about a property of ``prim`` a read could not resolve.

    Args:
        prim: The prim the read was for.
        message: What could not be read and why.
        property_name: The property, when the failure is one property of
            ``prim``.
        layer: The file holding the spec pxr refused.
    """
    property_path = prim.GetPath().AppendProperty(property_name).pathString if property_name else None
    return ValidationIssue(
        "warning",
        UNREADABLE_PROPERTY,
        message,
        layer=layer,
        prim_path=prim.GetPath().pathString,
        property_path=property_path,
    ).to_dict()


def relationship_targets(rel: Usd.Relationship, problems: List[Dict[str, Any]]) -> Optional[List[str]]:
    """Return a relationship's target paths, or None (with a warning) when pxr cannot compose them.

    pxr refuses a relationship whose name some layer authors as an attribute
    (``... is not a relationship``).
    """
    try:
        return [target.pathString for target in rel.GetTargets()]
    except Tf.ErrorException as exc:
        reason, layer = _read_failure(exc)
        problems.append(
            unreadable_issue(
                rel.GetPrim(),
                f"Cannot read relationship {rel.GetName()}: {reason}. A layer authors it as an attribute, "
                "which USD cannot read as a relationship; it is left out of relationships.",
                rel.GetName(),
                layer,
            )
        )
        return None


def _direct_binding(prim: Usd.Prim, purposes: Sequence[str]) -> Optional[str]:
    """The material ``prim`` itself binds for the first of ``purposes`` it binds, or None.

    A binding pxr cannot read on this prim counts as none.
    """
    api = UsdShade.MaterialBindingAPI(prim)
    for purpose in purposes:
        try:
            material = api.GetDirectBinding(purpose).GetMaterial()
        except Tf.ErrorException:
            continue
        if material:
            return material.GetPath().pathString
    return None


def _nearest_readable_binding(prim: Usd.Prim, purpose: str) -> Optional[str]:
    """The nearest direct binding on ``prim`` or an ancestor that pxr can read.

    The fallback when ``ComputeBoundMaterial`` fails on a malformed binding
    somewhere up the chain: a prim's own valid binding still counts. A
    purpose-specific binding is preferred to the all-purpose one at each
    level, as USD resolves them; binding strength and collection bindings are
    not considered.
    """
    purposes = [purpose, UsdShade.Tokens.allPurpose] if purpose else [UsdShade.Tokens.allPurpose]
    current = prim
    while current and not current.IsPseudoRoot():
        material = _direct_binding(current, purposes)
        if material:
            return material
        current = current.GetParent()
    return None


def bound_material_path(
    prim: Usd.Prim, purpose: str = "", problems: Optional[List[Dict[str, Any]]] = None
) -> Optional[str]:
    """Path of the material ``UsdShade.MaterialBindingAPI`` resolves for ``prim``, or None.

    Resolution reads the ``material:binding`` relationships of the prim and
    its ancestors. When one of them is authored as an attribute pxr raises;
    the binding is then the nearest direct binding that pxr can read (the
    prim's own, or an ancestor's), with a warning in ``problems`` naming the
    property pxr refused.

    Args:
        prim: The prim.
        purpose: Binding purpose (``physics``, ``preview``, ``full``); empty
            for the all-purpose binding.
        problems: Collects the warning when pxr cannot resolve the binding.
    """
    api = UsdShade.MaterialBindingAPI(prim)
    try:
        material, _relationship = api.ComputeBoundMaterial(purpose) if purpose else api.ComputeBoundMaterial()
    except Tf.ErrorException as exc:
        fallback = _nearest_readable_binding(prim, purpose)
        if problems is not None:
            which = f"{purpose} material binding" if purpose else "material binding"
            reason, layer = _read_failure(exc)
            outcome = (
                f"the nearest binding pxr can read, {fallback}, is reported"
                if fallback
                else "the binding is reported as none"
            )
            problems.append(
                unreadable_issue(
                    prim,
                    f"Cannot resolve the {which} of {prim.GetPath()}: {reason}. A material:binding "
                    f"property on it or an ancestor is authored as an attribute, which USD cannot read as a "
                    f"relationship; {outcome}.",
                    layer=layer,
                )
            )
        return fallback
    return material.GetPath().pathString if material else None


__all__ = [
    "UNREADABLE_PROPERTY",
    "PxrError",
    "bound_material_path",
    "is_pxr_error",
    "relationship_targets",
    "tf_error_reason",
    "unreadable_issue",
]
