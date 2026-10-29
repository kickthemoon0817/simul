"""Test doubles shared across the suite.

``FakeFastMCP`` stands in for the real FastMCP instance when a test only needs
to see what the server registered. It records every tool with the keyword
arguments it was registered with, answers the handful of FastMCP calls the
server makes during construction, and mirrors FastMCP 3's ``local_provider``
component keys so the server's own tool listing works against it.

``FakeVector``, ``FakeMatrix`` and ``FakePropArray`` stand in for Blender's
``mathutils`` types and ``bpy_prop_array``; ``fake_mathutils`` packages the
first two as the module the Blender runtime imports.
"""

from __future__ import annotations

from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence

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


class FakeMatrix:
    """``mathutils.Matrix`` stand-in for a 4x4 affine transform.

    Like mathutils, ``@`` multiplies only a Vector. Blender 5.0.1 raises
    ``TypeError: Matrix multiplication: not supported between 'Matrix' and
    'bpy_prop_array' types`` for a bare ``bound_box`` corner; a stand-in that
    multiplied any sequence hid exactly that bug.
    """

    def __init__(self, rows: Sequence[Sequence[float]]) -> None:
        self.rows = [tuple(float(v) for v in row) for row in rows]

    @classmethod
    def Translation(cls, offset: Sequence[float]) -> "FakeMatrix":  # noqa: N802
        """Return a translation matrix, as ``mathutils.Matrix.Translation`` does."""
        x, y, z = offset
        return cls([(1, 0, 0, x), (0, 1, 0, y), (0, 0, 1, z), (0, 0, 0, 1)])

    @property
    def translation(self) -> FakeVector:
        return FakeVector(row[3] for row in self.rows[:3])

    def __matmul__(self, other: Any) -> FakeVector:
        if not isinstance(other, FakeVector):
            raise TypeError(
                "Matrix multiplication: not supported between 'Matrix' and "
                f"'{type(other).__name__}' types"
            )
        # A 3D vector is a point: mathutils extends it with w = 1.
        point = (*other, 1.0) if len(other) == 3 else tuple(other)
        product = [sum(a * b for a, b in zip(row, point)) for row in self.rows]
        return FakeVector(product[: len(other)])


def fake_mathutils() -> ModuleType:
    """Return a ``mathutils`` module made of the strict stand-ins above."""
    module = ModuleType("mathutils")
    module.Vector = FakeVector  # type: ignore[attr-defined]
    module.Matrix = FakeMatrix  # type: ignore[attr-defined]
    return module
