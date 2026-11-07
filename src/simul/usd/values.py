"""JSON to USD value conversion for the headless edit tools.

A JSON value carries no USD type: ``[1, 2, 3]`` may be a ``double3``
translate, a ``float3`` position, an ``int[]`` or the first row of a matrix.
The edit tools therefore convert a value against a known target type (the
existing attribute's, the schema's, the transform op's, or one the caller
names) rather than guessing, and only fall back to :func:`infer_type` for a
brand-new custom attribute.

Conventions:

- numbers widen to the target's numeric type (``45`` into a ``float``); an
  integer type refuses a fraction and checks its range;
- vectors are lists of their dimension (``[x, y, z]``);
- quaternions are ``[w, x, y, z]``, real part first, as ``Gf.Quat*`` stores
  them and as the tools report rotations;
- matrices are a list of rows (``[[...], ...]``) or the flat row-major list;
- assets are path strings, tokens and strings are strings;
- arrays are lists of their element's form.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

from ._pxr import Sdf, UsdGeom

_INT_RANGES: Dict[str, Tuple[int, int]] = {
    "unsigned char": (0, 255),
    "int": (-(2**31), 2**31 - 1),
    "unsigned int": (0, 2**32 - 1),
    "long long": (-(2**63), 2**63 - 1),
    "unsigned long long": (0, 2**64 - 1),
}
_FLOAT_TYPES = frozenset({"pxr_half::half", "float", "double"})


class ValueProblem(ValueError):
    """A JSON value that does not fit the target USD type; the message says why."""


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _shape(value: Any) -> str:
    """Short description of a JSON value, for messages."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "a boolean"
    if _is_number(value):
        return f"the number {value!r}"
    if isinstance(value, str):
        return f"the string {value!r}" if len(value) <= 40 else "a string"
    if isinstance(value, dict):
        return "a JSON object"
    if isinstance(value, (list, tuple)):
        if not value:
            return "an empty list"
        if all(_is_number(item) for item in value):
            return f"a list of {len(value)} numbers"
        if all(isinstance(item, (list, tuple)) for item in value):
            return f"a list of {len(value)} lists"
        return f"a list of {len(value)} items"
    return type(value).__name__


def parse_type_name(text: Any) -> Sdf.ValueTypeName:
    """Resolve a USD value type name such as ``float3``, ``quatf`` or ``point3f[]``.

    Raises:
        ValueProblem: The name is not a registered Sdf value type.
    """
    if isinstance(text, str):
        type_name = Sdf.ValueTypeNames.Find(text.strip())
        if type_name:
            return type_name
    raise ValueProblem(
        f"{text!r} is not a USD value type; use a name such as float, double3, quatf, "
        "matrix4d, token, asset or float3[]"
    )


def _number(value: Any, cpp_type: str, type_name: Any) -> Any:
    if not _is_number(value):
        raise ValueProblem(f"{type_name} takes a number, not {_shape(value)}")
    if cpp_type in _INT_RANGES:
        if isinstance(value, float):
            if not value.is_integer():
                raise ValueProblem(f"{type_name} takes an integer, not {value!r}")
            value = int(value)
        low, high = _INT_RANGES[cpp_type]
        if not low <= value <= high:
            raise ValueProblem(f"{value} is out of range for {type_name} ({low} to {high})")
        return int(value)
    return float(value)


def _numbers(value: Any, count: int, what: str, integers: bool = False) -> List[Any]:
    if not isinstance(value, (list, tuple)) or len(value) != count or not all(_is_number(v) for v in value):
        raise ValueProblem(f"{what}, not {_shape(value)}")
    if integers:
        if not all(float(v).is_integer() for v in value):
            raise ValueProblem(f"{what} (integers), not {_shape(value)} with fractions")
        return [int(v) for v in value]
    return [float(v) for v in value]


def _scalar_converter(scalar: Sdf.ValueTypeName) -> Optional[Callable[[Any], Any]]:
    """Return a converter from one JSON element to ``scalar``'s pxr value, or None."""
    cpp_type = scalar.type.typeName
    python_class = scalar.type.pythonClass
    if cpp_type == "bool":

        def to_bool(value: Any) -> bool:
            if not isinstance(value, bool):
                raise ValueProblem(f"{scalar} takes true or false, not {_shape(value)}")
            return value

        return to_bool
    if cpp_type in _INT_RANGES or cpp_type in _FLOAT_TYPES:
        return lambda value: _number(value, cpp_type, scalar)
    if cpp_type in ("string", "TfToken"):

        def to_text(value: Any) -> str:
            if not isinstance(value, str):
                raise ValueProblem(f"{scalar} takes a string, not {_shape(value)}")
            return value

        return to_text
    if cpp_type == "SdfAssetPath":

        def to_asset(value: Any) -> Sdf.AssetPath:
            if not isinstance(value, str):
                raise ValueProblem(f"asset takes a path string, not {_shape(value)}")
            return Sdf.AssetPath(value)

        return to_asset
    if cpp_type in ("GfTimeCode", "SdfTimeCode"):
        default_type = type(scalar.defaultValue)
        return lambda value: default_type(_number(value, "double", scalar))
    if cpp_type.startswith("GfVec") and python_class is not None:
        dimension = int(python_class.dimension)
        integers = cpp_type.endswith("i")
        what = f"{scalar} takes {dimension} {'integers' if integers else 'numbers'} [{', '.join('xyzw'[:dimension])}]"
        return lambda value: python_class(*_numbers(value, dimension, what, integers))
    if cpp_type.startswith("GfQuat") and python_class is not None:
        what = f"{scalar} takes a quaternion as 4 numbers [w, x, y, z], real part first"
        return lambda value: python_class(*_numbers(value, 4, what))
    if cpp_type.startswith("GfMatrix") and python_class is not None:
        rows, columns = python_class.dimension

        def to_matrix(value: Any) -> Any:
            what = f"{scalar} takes {rows} rows of {columns} numbers, or {rows * columns} numbers row by row"
            if isinstance(value, (list, tuple)) and len(value) == rows and all(
                isinstance(row, (list, tuple)) for row in value
            ):
                flat = [item for row in value for item in _numbers(row, columns, what)]
            else:
                flat = _numbers(value, rows * columns, what)
            return python_class(*flat)

        return to_matrix
    return None


def coerce(value: Any, type_name: Sdf.ValueTypeName) -> Any:
    """Convert a JSON value into the pxr value an attribute of ``type_name`` stores.

    Args:
        value: The decoded JSON value.
        type_name: The attribute's value type.

    Returns:
        A value ``UsdAttribute.Set`` accepts for ``type_name``.

    Raises:
        ValueProblem: The value does not fit; the message names the expected
            form and what was given.
    """
    if value is None:
        raise ValueProblem("null is not a value (clearing an authored value is not supported)")
    if isinstance(value, dict):
        raise ValueProblem("a JSON object is not an attribute value")
    scalar = type_name.scalarType
    convert = _scalar_converter(scalar)
    if convert is None:
        raise ValueProblem(f"setting {type_name} attributes is not supported")
    if not type_name.isArray:
        if isinstance(value, (list, tuple)) and scalar.type.pythonClass is None:
            raise ValueProblem(f"{type_name} takes a single value, not {_shape(value)}")
        return convert(value)
    if not isinstance(value, (list, tuple)):
        raise ValueProblem(f"{type_name} takes a list, not {_shape(value)}")
    elements = []
    for index, item in enumerate(value):
        try:
            elements.append(convert(item))
        except ValueProblem as problem:
            raise ValueProblem(f"element {index}: {problem}") from None
    array_class = type_name.type.pythonClass
    return array_class(elements) if array_class is not None else elements


def infer_type(value: Any) -> Optional[Sdf.ValueTypeName]:
    """Pick a value type for a new custom attribute from its JSON value.

    Booleans, integers, numbers and strings map to ``bool``, ``int``,
    ``float`` and ``string``; a list of 2 to 4 numbers to ``float2`` ..
    ``float4``; any other list of numbers to ``float[]`` (``int[]`` when all
    are integers); a list of equal-length 2 to 4 number lists to
    ``float2[]`` .. ``float4[]``; a list of strings to ``string[]``.

    Returns:
        The type, or None when nothing fits (null, objects, an empty or
        mixed list); the caller then asks for an explicit type.
    """
    names = Sdf.ValueTypeNames
    if isinstance(value, bool):
        return names.Bool
    if isinstance(value, int):
        return names.Int
    if isinstance(value, float):
        return names.Float
    if isinstance(value, str):
        return names.String
    if not isinstance(value, (list, tuple)) or not value:
        return None
    if all(isinstance(v, bool) for v in value):
        return names.BoolArray
    if all(_is_number(v) for v in value):
        if all(isinstance(v, int) for v in value):
            return names.IntArray
        return {2: names.Float2, 3: names.Float3, 4: names.Float4}.get(len(value), names.FloatArray)
    if all(isinstance(v, str) for v in value):
        return names.StringArray
    if all(isinstance(v, (list, tuple)) for v in value):
        lengths = {len(v) for v in value}
        if len(lengths) == 1 and all(_is_number(x) for v in value for x in v):
            return {2: names.Float2Array, 3: names.Float3Array, 4: names.Float4Array}.get(lengths.pop())
    return None


# ---------------------------------------------------------------------------
# Transform ops
# ---------------------------------------------------------------------------

_DOUBLE, _FLOAT, _HALF = (
    UsdGeom.XformOp.PrecisionDouble,
    UsdGeom.XformOp.PrecisionFloat,
    UsdGeom.XformOp.PrecisionHalf,
)
_T = Sdf.ValueTypeNames
_VEC3_TYPES = {_DOUBLE: _T.Double3, _FLOAT: _T.Float3, _HALF: _T.Half3}
_SCALAR_TYPES = {_DOUBLE: _T.Double, _FLOAT: _T.Float, _HALF: _T.Half}
_QUAT_TYPES = {_DOUBLE: _T.Quatd, _FLOAT: _T.Quatf, _HALF: _T.Quath}
_MATRIX_TYPES = {_DOUBLE: _T.Matrix4d}

# Op type name -> (UsdGeom.XformOp type, value type per precision). This is
# UsdGeomXformOp::GetValueTypeName, which pxr does not expose to Python.
_OPS: Dict[str, Tuple[Any, Dict[Any, Sdf.ValueTypeName]]] = {
    "translate": (UsdGeom.XformOp.TypeTranslate, _VEC3_TYPES),
    "scale": (UsdGeom.XformOp.TypeScale, _VEC3_TYPES),
    "rotateX": (UsdGeom.XformOp.TypeRotateX, _SCALAR_TYPES),
    "rotateY": (UsdGeom.XformOp.TypeRotateY, _SCALAR_TYPES),
    "rotateZ": (UsdGeom.XformOp.TypeRotateZ, _SCALAR_TYPES),
    "rotateXYZ": (UsdGeom.XformOp.TypeRotateXYZ, _VEC3_TYPES),
    "rotateXZY": (UsdGeom.XformOp.TypeRotateXZY, _VEC3_TYPES),
    "rotateYXZ": (UsdGeom.XformOp.TypeRotateYXZ, _VEC3_TYPES),
    "rotateYZX": (UsdGeom.XformOp.TypeRotateYZX, _VEC3_TYPES),
    "rotateZXY": (UsdGeom.XformOp.TypeRotateZXY, _VEC3_TYPES),
    "rotateZYX": (UsdGeom.XformOp.TypeRotateZYX, _VEC3_TYPES),
    "orient": (UsdGeom.XformOp.TypeOrient, _QUAT_TYPES),
    "transform": (UsdGeom.XformOp.TypeTransform, _MATRIX_TYPES),
}
# Single-axis translate and scale ops exist in newer USD releases only.
for _axis in "XYZ":
    for _base in ("translate", "scale"):
        _enum = getattr(UsdGeom.XformOp, f"Type{_base.capitalize()}{_axis}", None)
        if _enum is not None:
            _OPS[f"{_base}{_axis}"] = (_enum, _SCALAR_TYPES)

# USD's own defaults for new ops (the UsdGeomXformable Add*Op helpers).
_DEFAULT_PRECISION: Dict[str, Any] = {"translate": _DOUBLE, "transform": _DOUBLE}
RESET_XFORM_STACK = "!resetXformStack!"
INVERT_PREFIX = "!invert!"


def parse_xform_op_name(name: str) -> Optional[Tuple[str, str]]:
    """Split ``xformOp:<opType>[:<suffix>]`` into ``(opType, suffix)``.

    Returns:
        None when ``name`` is not in the ``xformOp`` namespace.

    Raises:
        ValueProblem: The name is in the namespace but its op type is unknown.
    """
    parts = name.split(":")
    if parts[0] != "xformOp":
        return None
    if len(parts) < 2 or parts[1] not in _OPS:
        raise ValueProblem(
            f"{name!r} is not a transform op name; ops are xformOp:<type>[:<suffix>] with type "
            f"one of {', '.join(_OPS)}"
        )
    return parts[1], ":".join(parts[2:])


def xform_op_type(op_name: str) -> Any:
    """Return the ``UsdGeom.XformOp`` type enum for an op type name such as ``translate``."""
    return _OPS[op_name][0]


def xform_op_precision(op_name: str, requested: Optional[Sdf.ValueTypeName]) -> Any:
    """Pick the precision of an op: the one whose value type is ``requested``, or USD's default.

    Raises:
        ValueProblem: ``requested`` is not a value type this op can have.
    """
    types = _OPS[op_name][1]
    if requested is None:
        default = _DEFAULT_PRECISION.get(op_name, _FLOAT)
        return default if default in types else next(iter(types))
    for precision, type_name in types.items():
        if type_name == requested:
            return precision
    allowed = ", ".join(str(type_name) for type_name in types.values())
    raise ValueProblem(f"a {op_name} op holds {allowed}, not {requested}")


def xform_op_value_type(op_name: str, precision: Any) -> Sdf.ValueTypeName:
    """The attribute type of an op of ``op_name`` at ``precision``."""
    return _OPS[op_name][1][precision]


def is_finite_time(value: Any) -> bool:
    """Whether ``value`` can be a USD time code (a finite number)."""
    return _is_number(value) and math.isfinite(float(value))


__all__ = [
    "INVERT_PREFIX",
    "RESET_XFORM_STACK",
    "ValueProblem",
    "coerce",
    "infer_type",
    "is_finite_time",
    "parse_type_name",
    "parse_xform_op_name",
    "xform_op_precision",
    "xform_op_type",
    "xform_op_value_type",
]
