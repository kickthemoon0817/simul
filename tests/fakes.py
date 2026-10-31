"""Test doubles shared across the suite.

``FakeFastMCP`` stands in for the real FastMCP instance when a test only needs
to see what the server registered. It records every tool with the keyword
arguments it was registered with, answers the handful of FastMCP calls the
server makes during construction, and mirrors FastMCP's ``local_provider``
component keys so the server's own tool listing works against it.

``StrictBlenderSession`` stands in for a Blender session behind the registered
tools; ``BlenderSessionAdapter`` hands it to the server.
``FakeVector``, ``FakeMatrix``, ``FakeEuler`` and ``FakePropArray`` stand in
for Blender's ``mathutils`` types and ``bpy_prop_array``; ``fake_mathutils``
packages the first three as the module the Blender runtime imports.
``FakeTransformObject`` is an object whose ``matrix_world`` follows its parent,
rotation mode and ``FakeConstraint`` stack as a Blender object's does.
"""

from __future__ import annotations

import copy
import inspect
import math
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from simul.config import Settings

ToolFunction = Callable[..., Any]


class FakeFastMCP:
    """FastMCP double that records tool registrations instead of serving them."""

    def __init__(self, name: str = "", version: str = "", **kwargs: Any) -> None:
        self.name = name
        self.version = version
        self.description: Optional[str] = kwargs.get("description")
        self.instructions: Optional[str] = kwargs.get("instructions")
        self.tools: List[SimpleNamespace] = []
        self.by_name: Dict[str, ToolFunction] = {}
        self.resources: List[str] = []
        self.extensions: List[Any] = []
        self.local_provider = SimpleNamespace(_components={})
        self.run_calls: List[Dict[str, Any]] = []

    def tool(self, name: str, **kwargs: Any) -> Callable[[ToolFunction], ToolFunction]:
        """Return a decorator that records the tool and hands the function back unchanged.

        Args:
            name: Registered tool name.
            **kwargs: Every other registration argument (description, annotations, ...).

        Returns:
            The recording decorator.
        """

        def decorator(func: ToolFunction) -> ToolFunction:
            self.tools.append(SimpleNamespace(name=name, func=func, kwargs=kwargs))
            self.by_name[name] = func
            self.local_provider._components[f"tool:{name}@"] = func
            return func

        return decorator

    def get_tools(self) -> List[SimpleNamespace]:
        """Return the recorded tools, mirroring the FastMCP method of the same name."""
        return self.tools

    async def get_tool(self, name: str) -> Optional[SimpleNamespace]:
        """Return an object carrying the tool's annotations, as the real ``get_tool`` does."""
        for tool in self.tools:
            if tool.name == name:
                return SimpleNamespace(annotations=tool.kwargs.get("annotations"))
        return None

    def resource(
        self, uri: str, *args: Any, **kwargs: Any
    ) -> Callable[[ToolFunction], ToolFunction]:
        """Record the resource's URI; the decorator hands the function back unchanged."""

        def decorator(func: ToolFunction) -> ToolFunction:
            self.resources.append(uri)
            return func

        return decorator

    def add_middleware(self, middleware: Any) -> None:
        """Accept the request-context middleware the server installs."""

    def add_extension(self, extension: Any) -> None:
        """Record a server extension (the background tasks one) the server registers."""
        self.extensions.append(extension)

    async def run_async(self, **kwargs: Any) -> None:
        """Record how the server asked to be run instead of serving."""
        self.run_calls.append(kwargs)


class AvailableAdapter:
    """Backend adapter stub that reports itself available so its tools register."""

    name: str = "stub"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.closed = False

    def is_available(self) -> bool:
        return True

    def get_capabilities(self) -> List[str]:
        return ["stub_capability"]

    def create_session(self) -> Any:
        raise NotImplementedError("registration-only stub")

    def close(self) -> None:
        self.closed = True


SessionCall = Tuple[str, Dict[str, Any]]


class StrictBlenderSession:
    """Blender session double spec'd on ``BlenderRuntimeSession``.

    It offers exactly the public ``BlenderRuntimeSession`` operations, as the
    attached-mode ``BlenderConnection`` and the add-on do, and each call must
    bind to that operation's signature. A call to anything else raises, which
    the tool envelope turns into an error payload, and is also kept in
    ``rejected``, so a test can fail on a tool that names a missing operation
    or passes arguments the operation does not take instead of passing on the
    error payload.

    Attributes:
        calls: ``(operation, arguments)`` per accepted call, the arguments bound
            to the operation's signature with its defaults applied.
        rejected: ``(name, reason)`` per refused attribute or call.
        responses: Payload returned per operation; others get ``default_response``.
        default_response: Payload of an operation without its own response.
    """

    def __init__(
        self,
        responses: Optional[Mapping[str, Dict[str, Any]]] = None,
        default_response: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.calls: List[SessionCall] = []
        self.rejected: List[Tuple[str, str]] = []
        self.responses: Dict[str, Dict[str, Any]] = dict(responses or {})
        self.default_response: Dict[str, Any] = dict(default_response or {})

    def __getattr__(self, method: str) -> Callable[..., Dict[str, Any]]:
        from simul.adapters.blender_runtime import BlenderRuntimeSession

        if method.startswith("__"):
            raise AttributeError(method)
        operation = getattr(BlenderRuntimeSession, method, None)
        if method.startswith("_") or method == "cleanup" or not callable(operation):
            self.rejected.append(
                (method, "not a public BlenderRuntimeSession operation")
            )
            raise AttributeError(f"BlenderRuntimeSession has no operation {method!r}")

        def invoke(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            try:
                bound = inspect.signature(operation).bind(None, *args, **kwargs)
            except TypeError as exc:
                self.rejected.append((method, str(exc)))
                raise
            bound.apply_defaults()
            arguments = dict(list(bound.arguments.items())[1:])
            self.calls.append((method, arguments))
            return copy.deepcopy(self.responses.get(method, self.default_response))

        return invoke


class BlenderSessionAdapter(AvailableAdapter):
    """Blender adapter double whose every session is ``session``.

    Attached mode runs the session on a worker thread, as the real adapter
    has the server do.
    """

    name: str = "blender"

    def __init__(self, settings: Settings, session: Optional[Any] = None) -> None:
        super().__init__(settings)
        self.session = session if session is not None else StrictBlenderSession()

    @property
    def session_blocks_io(self) -> bool:
        return self.settings.blender.mode == "attached"

    @contextmanager
    def create_session(self) -> Iterator[Any]:
        yield self.session


class FakeVector:
    """``mathutils.Vector`` stand-in: a float sequence a ``FakeMatrix`` multiplies."""

    def __init__(self, values: Iterable[float] = (0.0, 0.0, 0.0)) -> None:
        self._values = tuple(float(v) for v in values)

    def __iter__(self) -> Iterator[float]:
        return iter(self._values)

    def __getitem__(self, index: int) -> float:
        return self._values[index]

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return f"Vector({self._values!r})"


class FakePropArray(tuple):
    """``bpy_prop_array`` stand-in, such as an ``Object.bound_box`` corner: not a Vector."""


def _axis_rotation(axis: int, angle: float) -> np.ndarray:
    """Right-handed rotation by ``angle`` radians about axis 0 (X), 1 (Y) or 2 (Z)."""
    cos, sin = math.cos(angle), math.sin(angle)
    first, second = (axis + 1) % 3, (axis + 2) % 3
    matrix = np.eye(3)
    matrix[first, first] = matrix[second, second] = cos
    matrix[first, second], matrix[second, first] = -sin, sin
    return matrix


def euler_matrix(angles: Sequence[float], order: str = "XYZ") -> np.ndarray:
    """Rotation matrix of Euler angles (x, y, z) as Blender composes them.

    The first axis of ``order`` turns first: XYZ is Rz @ Ry @ Rx, ZXY is
    Ry @ Rx @ Rz. The angles are always given as (x, y, z).
    """
    matrix = np.eye(3)
    for letter in order:
        axis = "XYZ".index(letter)
        matrix = _axis_rotation(axis, angles[axis]) @ matrix
    return matrix


def matrix_euler(matrix: np.ndarray, order: str = "XYZ") -> Tuple[float, float, float]:
    """Euler angles (x, y, z) in ``order`` of a rotation matrix; inverts ``euler_matrix``."""
    i, j, k = ("XYZ".index(letter) for letter in order)
    # XYZ, YZX and ZXY are even permutations; the other three flip the signs.
    sign = 1.0 if (j - i) % 3 == 1 else -1.0
    angles = [0.0, 0.0, 0.0]
    angles[j] = math.asin(max(-1.0, min(1.0, -sign * matrix[k, i])))
    angles[i] = math.atan2(sign * matrix[k, j], matrix[k, k])
    angles[k] = math.atan2(sign * matrix[j, i], matrix[i, i])
    return angles[0], angles[1], angles[2]


def quaternion_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """Rotation matrix of a quaternion (w, x, y, z), as ``rotation_quaternion`` stores it."""
    w, x, y, z = np.asarray(quaternion, dtype=float) / np.linalg.norm(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_quaternion(matrix: np.ndarray) -> Tuple[float, float, float, float]:
    """The quaternion (w, x, y, z) of a rotation matrix; inverts ``quaternion_matrix``."""
    m = matrix
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 2 * math.sqrt(trace + 1)
        return (
            s / 4,
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
        )
    if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2 * math.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2])
        return (
            (m[2, 1] - m[1, 2]) / s,
            s / 4,
            (m[0, 1] + m[1, 0]) / s,
            (m[0, 2] + m[2, 0]) / s,
        )
    if m[1, 1] > m[2, 2]:
        s = 2 * math.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2])
        return (
            (m[0, 2] - m[2, 0]) / s,
            (m[0, 1] + m[1, 0]) / s,
            s / 4,
            (m[1, 2] + m[2, 1]) / s,
        )
    s = 2 * math.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1])
    return (
        (m[1, 0] - m[0, 1]) / s,
        (m[0, 2] + m[2, 0]) / s,
        (m[1, 2] + m[2, 1]) / s,
        s / 4,
    )


def axis_angle_matrix(axis_angle: Sequence[float]) -> np.ndarray:
    """Rotation matrix of (angle, x, y, z), as ``rotation_axis_angle`` stores it."""
    angle, *axis = axis_angle
    direction = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    return quaternion_matrix((math.cos(angle / 2), *(direction * math.sin(angle / 2))))


def matrix_axis_angle(matrix: np.ndarray) -> Tuple[float, float, float, float]:
    """The (angle, x, y, z) of a rotation matrix; inverts ``axis_angle_matrix``."""
    w, x, y, z = matrix_quaternion(matrix)
    angle = 2 * math.acos(max(-1.0, min(1.0, w)))
    sin = math.sin(angle / 2)
    if abs(sin) < 1e-12:
        return 0.0, 0.0, 1.0, 0.0
    return angle, x / sin, y / sin, z / sin


class FakeEuler:
    """``mathutils.Euler`` stand-in: angles (x, y, z) in radians and their order."""

    def __init__(
        self, angles: Iterable[float] = (0.0, 0.0, 0.0), order: str = "XYZ"
    ) -> None:
        self._angles = tuple(float(a) for a in angles)
        self.order = order

    def __iter__(self) -> Iterator[float]:
        return iter(self._angles)

    def __getitem__(self, index: int) -> float:
        return self._angles[index]

    def __len__(self) -> int:
        return len(self._angles)

    def __repr__(self) -> str:
        return f"Euler({self._angles!r}, {self.order!r})"

    def to_matrix(self) -> "FakeMatrix":
        return FakeMatrix(euler_matrix(self._angles, self.order))


class FakeMatrix:
    """``mathutils.Matrix`` stand-in for a 3x3 or 4x4 transform.

    Like mathutils, ``@`` multiplies only a Vector or a Matrix. Blender 5.0.1
    raises ``TypeError: Matrix multiplication: not supported between 'Matrix'
    and 'bpy_prop_array' types`` for a bare ``bound_box`` corner; a stand-in
    that multiplied any sequence hid exactly that bug. ``to_euler`` likewise
    stays near an Euler only, as mathutils does.
    """

    def __init__(self, rows: Any) -> None:
        self.rows = [tuple(float(v) for v in row) for row in rows]

    @classmethod
    def Translation(cls, offset: Sequence[float]) -> "FakeMatrix":  # noqa: N802
        """Return a translation matrix, as ``mathutils.Matrix.Translation`` does."""
        x, y, z = offset
        return cls([(1, 0, 0, x), (0, 1, 0, y), (0, 0, 1, z), (0, 0, 0, 1)])

    @classmethod
    def Identity(cls, size: int) -> "FakeMatrix":  # noqa: N802
        return cls(np.eye(size))

    @classmethod
    def Scale(  # noqa: N802
        cls, factor: float, size: int, axis: Sequence[float]
    ) -> "FakeMatrix":
        """Scale by ``factor`` along ``axis`` (3x3 or 4x4), as ``mathutils.Matrix.Scale`` does."""
        direction = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
        matrix = np.eye(size)
        matrix[:3, :3] += (factor - 1.0) * np.outer(direction, direction)
        return cls(matrix)

    @classmethod
    def LocRotScale(  # noqa: N802
        cls, location: Any, rotation: Any, scale: Any
    ) -> "FakeMatrix":
        """Translation @ rotation @ scale, as ``mathutils.Matrix.LocRotScale`` builds it."""
        matrix = np.eye(4)
        if rotation is not None:
            if isinstance(rotation, FakeEuler):
                rotation = rotation.to_matrix()
            matrix[:3, :3] = np.asarray(rotation.rows)[:3, :3]
        if scale is not None:
            matrix[:3, :3] = matrix[:3, :3] * np.asarray(list(scale))
        if location is not None:
            matrix[:3, 3] = list(location)
        return cls(matrix)

    @property
    def translation(self) -> FakeVector:
        return FakeVector(row[3] for row in self.rows[:3])

    def __iter__(self) -> Iterator[Tuple[float, ...]]:
        return iter(self.rows)

    def __getitem__(self, index: int) -> Tuple[float, ...]:
        return self.rows[index]

    def __len__(self) -> int:
        return len(self.rows)

    def inverted(self) -> "FakeMatrix":
        return FakeMatrix(np.linalg.inv(np.asarray(self.rows)))

    def copy(self) -> "FakeMatrix":
        """A matrix of these values that no later change to its source reaches."""
        return FakeMatrix(self.rows)

    @property
    def is_negative(self) -> bool:
        """Whether the matrix mirrors (a negative determinant), as mathutils reports it."""
        return bool(np.linalg.det(np.asarray(self.rows)[:3, :3]) < 0)

    def to_scale(self) -> FakeVector:
        """The axes' lengths, as mathutils returns them: negative on every axis
        for a mirroring matrix (Blender 5.0.1 gives (-1.5, -1.5, -1.5) for a
        scale of (-1.5, 1.5, 1.5))."""
        lengths = np.linalg.norm(np.asarray(self.rows)[:3, :3], axis=0)
        return FakeVector(-lengths if self.is_negative else lengths)

    def to_euler(self, order: str = "XYZ", compatible: Any = None) -> FakeEuler:
        if compatible is not None and not isinstance(compatible, FakeEuler):
            raise TypeError(
                f"to_euler() argument 2 must be Euler, not {type(compatible).__name__}"
            )
        rotation = np.asarray(self.rows)[:3, :3]
        angles: Sequence[float] = matrix_euler(
            rotation / np.linalg.norm(rotation, axis=0), order
        )
        if compatible is not None:
            # The equivalent angle nearest the compatible one: whole turns apart.
            angles = [
                a + 2 * math.pi * round((c - a) / (2 * math.pi))
                for a, c in zip(angles, compatible)
            ]
        return FakeEuler(angles, order)

    def __matmul__(self, other: Any) -> Any:
        if isinstance(other, FakeMatrix):
            return FakeMatrix(np.asarray(self.rows) @ np.asarray(other.rows))
        if not isinstance(other, FakeVector):
            raise TypeError(
                "Matrix multiplication: not supported between 'Matrix' and "
                f"'{type(other).__name__}' types"
            )
        # A 3D vector is a point: mathutils extends it with w = 1.
        point = (*other, 1.0) if len(other) == 3 else tuple(other)
        product = [sum(a * b for a, b in zip(row, point)) for row in self.rows]
        return FakeVector(product[: len(other)])


# The constraint types with a target object (``target``) on Blender 5.0.1. An
# Armature constraint has a list of targets (``targets``) instead.
TARGET_CONSTRAINT_TYPES = frozenset(
    {
        "ACTION",
        "CHILD_OF",
        "CLAMP_TO",
        "COPY_LOCATION",
        "COPY_ROTATION",
        "COPY_SCALE",
        "COPY_TRANSFORMS",
        "DAMPED_TRACK",
        "FLOOR",
        "FOLLOW_PATH",
        "GEOMETRY_ATTRIBUTE",
        "IK",
        "LIMIT_DISTANCE",
        "LOCKED_TRACK",
        "PIVOT",
        "SHRINKWRAP",
        "SPLINE_IK",
        "STRETCH_TO",
        "TRACK_TO",
        "TRANSFORM",
    }
)


class FakeConstraint:
    """``bpy.types.Constraint`` stand-in: what Blender reports of it, and its effect.

    ``solve`` takes the world matrix that parenting and the object's own
    transform give (a 4x4 array) and returns the one the constraint makes of
    it, as Blender's constraint stack does. Blender skips a muted constraint,
    one without influence and an invalid one (``is_valid`` False, e.g. a Track
    To without a target); so does ``apply``. Blender blends a partial influence;
    that is not modeled, and ``apply`` refuses it rather than guess.

    Only a type in ``TARGET_CONSTRAINT_TYPES`` has ``target`` (and may have
    ``subtarget``), only an Armature has ``targets`` (each with ``target``,
    ``subtarget`` and ``weight``), only an Action ``use_eval_time`` and only a
    Track To ``use_target_z`` (False, Blender's default, unless given), as on
    a real constraint, so code reading them from another type fails here as
    it does in Blender. Deleting a target object leaves ``target`` None and
    ``is_valid`` True, since Blender 5.0.1 checks the targets only when the
    constraint is edited, and Blender's evaluation then does nothing; nor
    does ``apply``. Likewise for an Armature constraint once a target with
    weight lacks its object or bone. A Pivot without a target still pivots
    about its owner, and an Action that uses eval_time needs no target.
    """

    def __init__(
        self,
        name: str,
        constraint_type: str,
        solve: Callable[[np.ndarray], np.ndarray],
        mute: bool = False,
        influence: float = 1.0,
        is_valid: bool = True,
        **settings: Any,
    ) -> None:
        required = set()
        optional = set()
        defaults: Dict[str, Any] = {}
        if constraint_type in TARGET_CONSTRAINT_TYPES:
            required.add("target")
            optional.add("subtarget")
        if constraint_type == "ARMATURE":
            required.add("targets")
        if constraint_type == "ACTION":
            required.add("use_eval_time")
        if constraint_type == "TRACK_TO":
            # Target Z: the target's Z axis, not world Z, is the up direction.
            defaults["use_target_z"] = False
        optional |= set(defaults)
        if not required <= set(settings) <= required | optional:
            raise TypeError(
                f"A {constraint_type} constraint takes {sorted(required)} "
                f"(optionally {sorted(optional)}), not {sorted(settings)}"
            )
        self.name = name
        self.type = constraint_type
        self.mute = mute
        self.influence = influence
        self.is_valid = is_valid
        self._solve = solve
        for setting, value in {**defaults, **settings}.items():
            setattr(self, setting, value)

    def _lacks_target(self) -> bool:
        """Whether Blender's evaluation finds none of the target it acts with."""
        if self.type == "ARMATURE":
            return any(
                entry.weight > 0.0
                and (
                    entry.target is None
                    or entry.subtarget not in entry.target.data.bones
                )
                for entry in self.targets
            )
        if self.type == "PIVOT" or (self.type == "ACTION" and self.use_eval_time):
            return False
        return self.type in TARGET_CONSTRAINT_TYPES and self.target is None

    def apply(self, matrix: np.ndarray) -> np.ndarray:
        """Return ``matrix`` with this constraint applied, or unchanged when Blender skips it."""
        if (
            self.mute
            or self.influence <= 0.0
            or not self.is_valid
            or self._lacks_target()
        ):
            return matrix
        if self.influence < 1.0:
            raise NotImplementedError("a partial constraint influence is not modeled")
        return self._solve(matrix)


class FakeLiveMatrix(FakeMatrix):
    """A ``FakeMatrix`` that reads its values from its source whenever it is used.

    A matrix that bpy returns for an object's property, such as
    ``matrix_basis``, stays bound to that property: on Blender 5.0.1 one read
    before a depsgraph evaluation held what a driver wrote during it. Its
    ``copy()``, like any product with it, keeps the values of the moment.
    """

    def __init__(self, source: Callable[[], Any]) -> None:
        self._source = source

    @property  # type: ignore[override]
    def rows(self) -> List[Tuple[float, ...]]:
        return [tuple(float(v) for v in row) for row in self._source()]


class FakeTransformObject:
    """A Blender object whose ``matrix_world`` follows its parent, rotation mode and constraints.

    Reading ``matrix_world`` composes it as Blender evaluates the object: the
    parent's world matrix, ``matrix_parent_inverse``, then the local
    location, rotation and scale (``matrix_basis``), the rotation read from
    ``rotation_euler`` in the order ``rotation_mode`` names, from
    ``rotation_quaternion`` or from ``rotation_axis_angle``, and last the
    ``constraints`` in stack order. Assigning it does what bpy's setter does:
    it solves the local transform through the parent, not the constraints,
    and stores the rotation in the property of the rotation mode only, a
    mirroring matrix as a rotation and a scale negative on every axis. So a
    write to ``location`` or ``rotation_euler`` lands where it does in
    Blender: relative to the parent, and ignored by a QUATERNION or
    AXIS_ANGLE object.

    With ``stored=True`` the object keeps ``matrix_world`` as Blender keeps
    it for an original object: ``evaluate()`` (a depsgraph evaluation)
    recomputes it, the parent's first, and assigning it stores the value
    given. Writing location, rotation or scale, or moving the parent, leaves
    it stale until then, as in background Blender and the bpy module, which
    evaluate only on demand; so does assigning it to a constrained object,
    whose constraints act only on evaluation. By default it follows every
    write at once.
    """

    def __init__(
        self,
        name: str,
        object_type: str = "EMPTY",
        data: Any = None,
        location: Sequence[float] = (0.0, 0.0, 0.0),
        rotation_euler: Sequence[float] = (0.0, 0.0, 0.0),
        scale: Sequence[float] = (1.0, 1.0, 1.0),
        rotation_mode: str = "XYZ",
        parent: Optional["FakeTransformObject"] = None,
        stored: bool = False,
        constraints: Sequence[FakeConstraint] = (),
    ) -> None:
        self.name = name
        self.type = object_type
        self.data = data
        # Blender gives an object without geometry (an empty, a camera) eight
        # bound_box corners at its origin; a test sets an armature's or mesh's.
        self.bound_box: List[FakePropArray] = [FakePropArray((0.0, 0.0, 0.0))] * 8
        self.location: Any = FakeVector(location)
        self.rotation_mode = rotation_mode
        self.rotation_euler: Any = FakeEuler(rotation_euler)
        self.rotation_quaternion: Any = FakeVector((1.0, 0.0, 0.0, 0.0))
        self.rotation_axis_angle: Any = FakeVector((0.0, 0.0, 1.0, 0.0))
        self.scale: Any = FakeVector(scale)
        self.constraints: List[FakeConstraint] = list(constraints)
        # Disable in Viewports (the monitor icon), and the collections that hold it.
        self.hide_viewport = False
        self.users_collection: List["FakeCollection"] = []
        self.parent = parent
        self.parent_type = "OBJECT"
        # What Blender's Parent (Keep Transform) sets.
        self.matrix_parent_inverse = (
            parent.matrix_world.inverted()
            if parent is not None
            else FakeMatrix.Identity(4)
        )
        self._stores_matrix = stored
        self._stored: Optional[FakeMatrix] = None
        self.evaluate()

    def evaluate(self) -> None:
        """Recompute a stored matrix_world, the parent's first, as a depsgraph evaluation does."""
        if self.parent is not None:
            self.parent.evaluate()
        if self._stores_matrix:
            self._stored = self._composed()

    def _rotation(self) -> np.ndarray:
        if self.rotation_mode == "QUATERNION":
            return quaternion_matrix(list(self.rotation_quaternion))
        if self.rotation_mode == "AXIS_ANGLE":
            return axis_angle_matrix(list(self.rotation_axis_angle))
        return euler_matrix(list(self.rotation_euler), self.rotation_mode)

    def _parent_matrix(self) -> np.ndarray:
        if self.parent is None:
            return np.eye(4)
        return np.asarray(self.parent.matrix_world.rows) @ np.asarray(
            self.matrix_parent_inverse.rows
        )

    def _basis(self) -> np.ndarray:
        basis = np.eye(4)
        basis[:3, :3] = self._rotation() * np.asarray(list(self.scale))
        basis[:3, 3] = list(self.location)
        return basis

    @property
    def matrix_basis(self) -> FakeMatrix:
        """The local location, rotation and scale, before parenting and constraints.

        Bound to the object, as bpy's is: it reads the transform when used.
        """
        return FakeLiveMatrix(self._basis)

    def _composed(self) -> FakeMatrix:
        matrix = self._parent_matrix() @ self._basis()
        for constraint in self.constraints:
            matrix = constraint.apply(matrix)
        return FakeMatrix(matrix)

    @property
    def matrix_world(self) -> FakeMatrix:
        if self._stored is not None:
            return FakeMatrix(self._stored.rows)
        return self._composed()

    def evaluated_get(self, depsgraph: "FakeDepsgraph") -> Any:
        """``Object.evaluated_get``: this object's evaluated copy, or the original when left out."""
        return depsgraph.evaluated(self)

    @matrix_world.setter
    def matrix_world(self, value: FakeMatrix) -> None:
        local = np.linalg.inv(self._parent_matrix()) @ np.asarray(value.rows)
        scale = np.linalg.norm(local[:3, :3], axis=0)
        rotation = local[:3, :3] / scale
        if np.linalg.det(rotation) < 0:
            # mat4_to_loc_rot_size: a mirror becomes a rotation and a scale
            # negative on every axis.
            rotation, scale = -rotation, -scale
        if self._stores_matrix:
            self._stored = FakeMatrix(value.rows)
        self.location = FakeVector(local[:3, 3])
        self.scale = FakeVector(scale)
        self._store_rotation(rotation)

    def _store_rotation(self, rotation: np.ndarray) -> None:
        """Store a rotation matrix in the property of the rotation mode, exactly."""
        if self.rotation_mode == "QUATERNION":
            self.rotation_quaternion = FakeVector(matrix_quaternion(rotation))
        elif self.rotation_mode == "AXIS_ANGLE":
            self.rotation_axis_angle = FakeVector(matrix_axis_angle(rotation))
        else:
            self.rotation_euler = FakeEuler(
                matrix_euler(rotation, self.rotation_mode), self.rotation_mode
            )


def fake_mathutils() -> ModuleType:
    """Return a ``mathutils`` module made of the strict stand-ins above."""
    module = ModuleType("mathutils")
    module.Vector = FakeVector  # type: ignore[attr-defined]
    module.Matrix = FakeMatrix  # type: ignore[attr-defined]
    module.Euler = FakeEuler  # type: ignore[attr-defined]
    return module


class FakeCollection:
    """``bpy.types.Collection`` stand-in: its name and Disable in Viewports (hide_viewport)."""

    def __init__(self, name: str, hide_viewport: bool = False) -> None:
        self.name = name
        self.hide_viewport = hide_viewport

    def link(self, obj: Any) -> None:
        """Put an object in this collection, as ``collection.objects.link`` does."""
        obj.users_collection.append(self)


class FakeLayerCollection:
    """``bpy.types.LayerCollection`` stand-in: a collection's place in a view layer.

    ``exclude`` is its checkbox in the outliner; ``children`` are the layer
    collections of its child collections. The eye icon (hide_viewport here)
    is left out: hiding never keeps Blender from evaluating an object.
    """

    def __init__(
        self,
        collection: FakeCollection,
        children: Sequence["FakeLayerCollection"] = (),
        exclude: bool = False,
    ) -> None:
        self.collection = collection
        self.children = list(children)
        self.exclude = exclude


class FakeViewLayer:
    """``bpy.types.ViewLayer`` stand-in: its name and the scene collection's layer collection."""

    def __init__(
        self, layer_collection: FakeLayerCollection, name: str = "ViewLayer"
    ) -> None:
        self.name = name
        self.layer_collection = layer_collection


class FakeDepsgraph:
    """``bpy.types.Depsgraph`` stand-in for ``Object.evaluated_get``.

    Blender gives back an object's evaluated copy (``is_evaluated`` True) when
    the depsgraph holds the object, and the original itself (``is_evaluated``
    False) when it left the object out, as the viewport depsgraph leaves out a
    camera disabled in viewports that nothing visible depends on. The copy's
    ``matrix_world`` is the object's: what its last evaluation stored.
    """

    def __init__(self, left_out: Sequence[Any] = ()) -> None:
        self._left_out = list(left_out)

    def evaluated(self, obj: Any) -> SimpleNamespace:
        """The copy of ``obj`` this depsgraph evaluated, or ``obj`` as an original."""
        held = not any(obj is out for out in self._left_out)
        return SimpleNamespace(
            original=obj, is_evaluated=held, matrix_world=obj.matrix_world
        )
