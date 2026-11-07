"""Turn USD values into plain JSON.

Tool responses are JSON text, and the envelope used to encode whatever pxr
handed back with ``json.dumps(..., default=str)``. That made a vector or a
quaternion the string ``"(0, 0, 0.2)"`` while transforms and bounds were lists,
and wrote a float such as ``physics:breakForce`` (infinite by default) as the
bare token ``Infinity``, which is not JSON at all.

The rules here:

- numbers stay numbers; a non-finite float becomes the string ``"Infinity"``,
  ``"-Infinity"`` or ``"NaN"`` (strings, so the document stays valid JSON and a
  reader can still tell an unlimited force from a missing one);
- ``Gf`` vectors are lists, quaternions are ``[w, x, y, z]`` lists (the order
  ``transform.rotation`` uses), matrices are lists of rows;
- asset paths are their authored path string, ``Sdf.Path`` values their path
  string, time codes their number;
- an array is described by its length and its leading elements, never
  expanded whole, so a million-point mesh costs no more than a cube.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from ._pxr import Sdf, Usd

# Cap on how many leading elements of an array a caller may ask for. Bulk
# geometry belongs in get_mesh_info's statistics, not in a prim listing.
MAX_ARRAY_PREVIEW = 10_000

DEFAULT_ARRAY_PREVIEW = 16


def json_float(value: float) -> Any:
    """Return ``value``, or its string spelling when it is not finite."""
    if math.isfinite(value):
        return value
    if math.isnan(value):
        return "NaN"
    return "Infinity" if value > 0 else "-Infinity"


def _module(value: Any) -> str:
    return type(value).__module__ or ""


def _is_vt_array(value: Any) -> bool:
    return _module(value).startswith("pxr.Vt") and type(value).__name__.endswith("Array")


def to_json(value: Any) -> Any:
    """Convert one USD value (scalar, ``Gf`` type, dictionary or array) to JSON.

    Arrays are converted whole here; use ``describe_array`` to bound them.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return json_float(value)
    if isinstance(value, Sdf.AssetPath):
        return value.path
    if isinstance(value, Sdf.Path):
        return value.pathString
    if isinstance(value, Sdf.TimeCode):
        return json_float(float(value.GetValue()))
    if isinstance(value, Usd.TimeCode):
        return None if value.IsDefault() else json_float(float(value.GetValue()))
    if isinstance(value, dict):
        return {str(key): to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)) or _is_vt_array(value):
        return [to_json(item) for item in value]
    if _module(value).startswith("pxr.Gf"):
        return _gf_to_json(value)
    if hasattr(value, "GetAddedOrExplicitItems"):
        # Sdf list ops (apiSchemas, references, ...): the items they add.
        return [to_json(item) for item in value.GetAddedOrExplicitItems()]
    if hasattr(value, "item") and hasattr(value, "dtype"):
        # numpy scalars
        return to_json(value.item())
    return str(value)


def _gf_to_json(value: Any) -> Any:
    name = type(value).__name__
    if name.startswith("Quat"):
        imaginary = value.GetImaginary()
        return [json_float(float(value.GetReal()))] + [json_float(float(c)) for c in imaginary]
    if name.startswith("Matrix"):
        return [[json_float(float(c)) for c in row] for row in value]
    if name.startswith("Vec"):
        return [c if isinstance(c, int) else json_float(float(c)) for c in value]
    if name.startswith("Range"):
        return {"min": to_json(value.GetMin()), "max": to_json(value.GetMax())}
    return str(value)


def describe_array(value: Any, preview: int = DEFAULT_ARRAY_PREVIEW) -> Dict[str, Any]:
    """Describe an array by its length and its first ``preview`` elements.

    Args:
        value: A ``Vt`` array (or any sequence).
        preview: How many leading elements to include, 0 to
            ``MAX_ARRAY_PREVIEW``.

    Returns:
        ``length`` and ``value`` (the leading elements), plus
        ``truncated: True`` when the array is longer than ``preview``.
    """
    preview = max(0, min(int(preview), MAX_ARRAY_PREVIEW))
    length = len(value)
    # Index rather than slice: an empty slice of a Vt array is None, not [].
    head = [to_json(value[index]) for index in range(min(length, preview))]
    described: Dict[str, Any] = {"length": length, "value": head}
    if length > preview:
        described["truncated"] = True
    return described


def describe_attribute(
    attr: Usd.Attribute,
    time_code: Optional[Usd.TimeCode] = None,
    array_preview: int = DEFAULT_ARRAY_PREVIEW,
) -> Dict[str, Any]:
    """Describe one attribute: its type, JSON value and whether it is authored.

    Args:
        attr: The attribute.
        time_code: Time to read the value at; None for the default time.
        array_preview: Leading elements of an array value to include.

    Returns:
        ``type`` (the Sdf value type, e.g. ``point3f[]``), ``value`` and
        ``authored`` (False when the value is the schema fallback). An array
        adds ``length`` (and ``truncated`` when cut), an animated attribute
        ``time_samples`` (their count), a connected one ``connections``.
    """
    time = Usd.TimeCode.Default() if time_code is None else time_code
    type_name = attr.GetTypeName()
    raw = attr.Get(time)
    record: Dict[str, Any] = {"type": str(type_name)}
    if raw is not None and type_name.isArray:
        record.update(describe_array(raw, array_preview))
    else:
        record["value"] = to_json(raw)
    record["authored"] = bool(attr.HasAuthoredValue())
    samples = attr.GetNumTimeSamples()
    if samples:
        record["time_samples"] = samples
    if attr.HasAuthoredConnections():
        record["connections"] = [path.pathString for path in attr.GetConnections()]
    return record


__all__ = [
    "DEFAULT_ARRAY_PREVIEW",
    "MAX_ARRAY_PREVIEW",
    "describe_array",
    "describe_attribute",
    "json_float",
    "to_json",
]
