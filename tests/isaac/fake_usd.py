"""An in-memory stand-in for the ``omni.usd`` / ``pxr`` surface the generated
listing scripts touch, so their paging and counting can run outside Kit.

The fakes model only what the scripts call: a prim tree with pre-order
traversal and ``PruneChildren``, type names, applied APIs and schema kinds,
and the handful of attribute getters the listings read.
"""

from __future__ import annotations

import asyncio
import io
import json
import sys
import types
from contextlib import redirect_stdout
from typing import Any, Dict, Iterable, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest

from simul.adapters.isaac_socket_client import ScriptResult
from simul.config import Settings
from simul.mcp.tools.isaac_tools import IsaacTools


class FakeAttribute:
    """A USD attribute holding one value."""

    def __init__(self, value: Any) -> None:
        self.value = value

    def Get(self) -> Any:
        return self.value

    def Set(self, value: Any) -> None:
        self.value = value


class FakePrim:
    """One node of the fake stage."""

    def __init__(
        self,
        path: str,
        type_name: str = "Xform",
        *,
        apis: Iterable[Any] = (),
        kinds: Iterable[Any] = (),
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.path = path
        self.type_name = type_name
        self.apis = tuple(apis)
        self.kinds = tuple(kinds)
        self.attributes = {
            name: FakeAttribute(value) for name, value in (attributes or {}).items()
        }
        self.children: List[FakePrim] = []

    def IsValid(self) -> bool:
        return True

    def IsPseudoRoot(self) -> bool:
        return self.path == "/"

    def IsActive(self) -> bool:
        return True

    def IsInstanceProxy(self) -> bool:
        return False

    def GetFilteredChildren(self, predicate: Any) -> List["FakePrim"]:
        return list(self.children)

    def GetPath(self) -> str:
        return self.path

    def GetName(self) -> str:
        return self.path.rsplit("/", 1)[-1]

    def GetTypeName(self) -> str:
        return self.type_name

    def GetChildren(self) -> List["FakePrim"]:
        return list(self.children)

    def HasAPI(self, api: Any) -> bool:
        return api in self.apis

    def IsA(self, kind: Any) -> bool:
        return kind in self.kinds

    def GetAttribute(self, name: str) -> Optional[FakeAttribute]:
        return self.attributes.get(name)


class InvalidPrim:
    """What ``GetPrimAtPath`` returns for an unknown path."""

    def IsValid(self) -> bool:
        return False


class FakePredicate:
    """A ``Usd`` prim flag or predicate; every fake prim satisfies all of them."""

    def __and__(self, other: Any) -> "FakePredicate":
        return self

    def __invert__(self) -> "FakePredicate":
        return self


PREDICATE = FakePredicate()


class FakePrimRange:
    """Pre-order traversal with ``PruneChildren`` like ``Usd.PrimRange``."""

    def __init__(self, root: FakePrim, predicate: Any = None) -> None:
        self._stack: List[FakePrim] = [root]
        self._current: Optional[FakePrim] = None
        self._pruned = False

    def __iter__(self) -> "FakePrimRange":
        return self

    def __next__(self) -> FakePrim:
        if self._current is not None and not self._pruned:
            self._stack.extend(reversed(self._current.children))
        self._pruned = False
        if not self._stack:
            raise StopIteration
        self._current = self._stack.pop()
        return self._current

    def PruneChildren(self) -> None:
        self._pruned = True


class FakeStage:
    """A stage built from ``(path, type_name)`` pairs plus per-prim extras."""

    def __init__(self, prims: Iterable[FakePrim]) -> None:
        self.by_path: Dict[str, FakePrim] = {"/": FakePrim("/", "")}
        for prim in prims:
            self.by_path[prim.path] = prim
        for path, prim in self.by_path.items():
            if path == "/":
                continue
            parent_path = path.rsplit("/", 1)[0] or "/"
            self.by_path[parent_path].children.append(prim)

    def GetPrimAtPath(self, path: str) -> Any:
        return self.by_path.get(path, InvalidPrim())

    def GetPseudoRoot(self) -> FakePrim:
        return self.by_path["/"]

    def Traverse(self, predicate: Any = None) -> Iterable[FakePrim]:
        for prim in FakePrimRange(self.by_path["/"]):
            if not prim.IsPseudoRoot():
                yield prim

    def GetEndTimeCode(self) -> float:
        return 0.0

    def GetStartTimeCode(self) -> float:
        return 0.0

    def DefinePrim(self, path: str, type_name: str) -> FakePrim:
        prim = FakePrim(path, type_name)
        self.by_path[path] = prim
        return prim


class _NullSchema:
    """A schema wrapper whose every attribute getter holds no value."""

    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: FakeAttribute(None)


class Sentinel:
    """A schema class or API marker prims can be tagged with.

    Calling it, as a script does with ``UsdGeom.Mesh(prim)``, yields a wrapper
    whose attributes are all unset.
    """

    # A sentinel doubles as its own ``Tf.Type``: the registry fakes below hand
    # it back, and ``FakePrim.IsA`` / ``HasAPI`` compare against it.
    isUnknown = False

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name

    def __call__(self, prim: Any) -> _NullSchema:
        return _NullSchema()

    @property
    def pythonClass(self) -> "Sentinel":
        return self

    @property
    def typeName(self) -> str:
        return self.name.replace(".", "")


def _module(name: str, **attrs: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


# Schema classes and API markers are module-level so a prim tagged before the
# modules are built matches what the script imports.
MESH_KIND = Sentinel("UsdGeom.Mesh")
XFORM_KIND = Sentinel("UsdGeom.Xform")
LIGHT_API = Sentinel("UsdLux.LightAPI")
RIGID_BODY_API = Sentinel("UsdPhysics.RigidBodyAPI")
COLLISION_API = Sentinel("UsdPhysics.CollisionAPI")
JOINT_KIND = Sentinel("UsdPhysics.Joint")

TYPED_KINDS = (MESH_KIND, XFORM_KIND, JOINT_KIND)
API_KINDS = (LIGHT_API, RIGID_BODY_API, COLLISION_API)
# Registered schema type names, as Usd.SchemaRegistry knows them.
SCHEMA_NAMES = {
    "Mesh": MESH_KIND,
    "Xform": XFORM_KIND,
    "PhysicsJoint": JOINT_KIND,
    "LightAPI": LIGHT_API,
    "PhysicsRigidBodyAPI": RIGID_BODY_API,
    "PhysicsCollisionAPI": COLLISION_API,
}


class _UnknownType:
    """``Tf.Type.Unknown``."""

    isUnknown = True
    pythonClass = None
    typeName = ""


UNKNOWN_TYPE = _UnknownType()

SCHEMA_REGISTRY = types.SimpleNamespace(
    GetTypeFromSchemaTypeName=lambda name: SCHEMA_NAMES.get(name, UNKNOWN_TYPE),
    IsTyped=lambda tf_type: tf_type in TYPED_KINDS,
    IsAppliedAPISchema=lambda tf_type: tf_type in API_KINDS,
)

TF = types.SimpleNamespace(
    Type=types.SimpleNamespace(
        Find=lambda cls: cls if isinstance(cls, Sentinel) else UNKNOWN_TYPE,
        FindByName=lambda name: UNKNOWN_TYPE,
        Unknown=UNKNOWN_TYPE,
    )
)


class Camera:
    """``UsdGeom.Camera``: both the kind prims are tagged with and the wrapper."""

    def __init__(self, prim: FakePrim) -> None:
        self.prim = prim

    def GetFocalLengthAttr(self) -> FakeAttribute:
        return self.prim.attributes.get("focalLength", FakeAttribute(24.0))

    def GetProjectionAttr(self) -> FakeAttribute:
        return FakeAttribute("perspective")


class Material:
    """``UsdShade.Material`` with no surface shader bound."""

    def __init__(self, prim: FakePrim) -> None:
        self.prim = prim

    def ComputeSurfaceSource(self, context: str = "") -> Tuple[Any, ...]:
        return (None,)


class Scene:
    """``UsdPhysics.Scene`` storing gravity on the prim's attributes."""

    def __init__(self, prim: FakePrim) -> None:
        self.prim = prim
        self.prim.attributes.setdefault(
            "gravityDirection", FakeAttribute((0.0, -1.0, 0.0))
        )
        self.prim.attributes.setdefault("gravityMagnitude", FakeAttribute(1.62))

    def CreateGravityDirectionAttr(self, value: Any = None) -> FakeAttribute:
        attribute = self.prim.attributes["gravityDirection"]
        if value is not None:
            attribute.Set(value)
        return attribute

    def CreateGravityMagnitudeAttr(self, value: Any = None) -> FakeAttribute:
        attribute = self.prim.attributes["gravityMagnitude"]
        if value is not None:
            attribute.Set(value)
        return attribute

    def GetGravityDirectionAttr(self) -> FakeAttribute:
        return self.prim.attributes["gravityDirection"]

    def GetGravityMagnitudeAttr(self) -> FakeAttribute:
        return self.prim.attributes["gravityMagnitude"]


def usd_modules(stage: FakeStage) -> Dict[str, types.ModuleType]:
    """Build the ``sys.modules`` entries a listing script imports.

    Args:
        stage: The fake stage ``omni.usd.get_context().get_stage()`` returns.

    Returns:
        Module name to module object, ready for ``sys.modules``.
    """
    omni = _module("omni")
    omni_usd = _module(
        "omni.usd",
        get_context=lambda: types.SimpleNamespace(
            get_stage=lambda: stage, get_stage_url=lambda: "memory://stage"
        ),
    )
    omni.usd = omni_usd  # type: ignore[attr-defined]
    submodules = {
        "Usd": types.SimpleNamespace(
            PrimRange=FakePrimRange,
            TimeCode=types.SimpleNamespace(Default=lambda: 0.0),
            SchemaRegistry=SCHEMA_REGISTRY,
            PrimIsActive=PREDICATE,
            PrimIsDefined=PREDICATE,
            PrimIsLoaded=PREDICATE,
            PrimIsAbstract=PREDICATE,
            PrimDefaultPredicate=PREDICATE,
            PrimAllPrimsPredicate=PREDICATE,
            TraverseInstanceProxies=lambda predicate=PREDICATE: predicate,
        ),
        "UsdGeom": types.SimpleNamespace(
            Camera=Camera,
            Mesh=MESH_KIND,
            Xform=XFORM_KIND,
            GetStageUpAxis=lambda _stage: "Z",
            GetStageMetersPerUnit=lambda _stage: 1.0,
        ),
        "UsdShade": types.SimpleNamespace(Material=Material),
        "UsdLux": types.SimpleNamespace(LightAPI=LIGHT_API),
        "UsdPhysics": types.SimpleNamespace(
            RigidBodyAPI=RIGID_BODY_API,
            CollisionAPI=COLLISION_API,
            Joint=JOINT_KIND,
            Scene=Scene,
        ),
        "Gf": types.SimpleNamespace(Vec3f=lambda *xyz: tuple(float(v) for v in xyz)),
        "Sdf": types.SimpleNamespace(AssetPath=type("AssetPath", (), {})),
        "Tf": TF,
    }
    pxr = _module("pxr", **submodules)
    modules: Dict[str, Any] = {"omni": omni, "omni.usd": omni_usd, "pxr": pxr}
    # importlib.import_module("pxr.UsdPhysics") reads sys.modules first.
    modules.update({f"pxr.{name}": module for name, module in submodules.items()})
    return modules


def capture_script(method: str, **kwargs: Any) -> str:
    """Return the script ``IsaacTools.<method>`` sends for ``kwargs``.

    Args:
        method: IsaacTools method name.
        **kwargs: Call arguments.

    Returns:
        The generated Python source.
    """
    captured: List[str] = []

    def _record(code: str) -> ScriptResult:
        captured.append(code)
        return ScriptResult(success=True, output=json.dumps({"ok": True}))

    client = MagicMock()
    client.address = "127.0.0.1:8226"
    client.timeout_seconds = 30.0
    client.bridge_enabled = False
    client.fallback_to_vscode = True
    client.execute = AsyncMock(side_effect=_record)
    client.bridge_request = AsyncMock(return_value=None)

    asyncio.run(getattr(IsaacTools(client, settings=Settings()), method)(**kwargs))
    assert captured, f"{method} generated no script"
    return captured[0]


def run_script(script: str, modules: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a generated script against fake modules and parse its output.

    Args:
        script: Generated Python source.
        modules: ``sys.modules`` entries to install while it runs.

    Returns:
        The single JSON object the script printed.
    """
    stdout = io.StringIO()
    namespace: Dict[str, Any] = {"__name__": "__main__"}
    with pytest.MonkeyPatch.context() as patcher:
        for name, module in modules.items():
            patcher.setitem(sys.modules, name, module)
        with redirect_stdout(stdout):
            exec(compile(script, "<generated>", "exec"), namespace)
    printed = stdout.getvalue().strip().splitlines()
    assert len(printed) == 1, printed
    return json.loads(printed[0])
