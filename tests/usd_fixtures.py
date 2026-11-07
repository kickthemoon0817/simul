"""Real USD files for the headless USD tests: valid ones and every way one breaks.

Each builder writes with ``pxr`` (or raw bytes, for files pxr would refuse to
write) under the directory it is given and returns the path, so a test can
point the session, the MCP tools or the CLI at the same file. The broken
files mirror the production-readiness audit's fixtures: a syntax error, an
empty file, random bytes and text named ``.usdc``, missing sublayers,
references and payloads, a dangling defaultPrim and physics mistakes.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils

CORRUPT_USDA = (
    '#usda 1.0\n(\n    defaultPrim = "World"\n)\n\n'
    'def Xform "World"\n{\n    double3 xformOp:translate = (0, 0\n'
)


def _stage_with_metadata(path: Path, *, up_axis: str = "Z", meters: float = 1.0) -> Usd.Stage:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, up_axis)
    UsdGeom.SetStageMetersPerUnit(stage, meters)
    return stage


def write_robot(path: Path) -> Path:
    """A clean articulated robot: Z up, meters, defaultPrim, two linked bodies."""
    stage = _stage_with_metadata(path)
    robot = UsdGeom.Xform.Define(stage, "/Robot")
    stage.SetDefaultPrim(robot.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(robot.GetPrim())
    bodies = []
    for name, z in (("base", 0.0), ("arm", 0.5)):
        link = UsdGeom.Xform.Define(stage, f"/Robot/{name}")
        link.AddTranslateOp().Set(Gf.Vec3d(0, 0, z))
        UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
        UsdPhysics.MassAPI.Apply(link.GetPrim()).CreateMassAttr(1.0)
        cube = UsdGeom.Cube.Define(stage, f"/Robot/{name}/collision")
        cube.CreateSizeAttr(0.2)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        bodies.append(link.GetPath())
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/Robot/shoulder")
    joint.CreateBody0Rel().SetTargets([bodies[0]])
    joint.CreateBody1Rel().SetTargets([bodies[1]])
    joint.CreateAxisAttr(UsdPhysics.Tokens.z)
    joint.CreateLowerLimitAttr(-90.0)
    joint.CreateUpperLimitAttr(90.0)
    stage.GetRootLayer().Save()
    return path


def write_usdz(path: Path, source: Path) -> Path:
    """Package ``source`` as a usdz pxr considers well formed."""
    assert UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(str(source)), str(path))
    return path


def write_compressed_usdz(path: Path, source: Path) -> Path:
    """A usdz whose root layer is deflate-compressed, which the usdz rules forbid."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(source, arcname=source.name)
    return path


def write_unaligned_usdz(path: Path, source: Path) -> Path:
    """A stored (uncompressed) usdz pxr opens, but with unaligned files and a .txt inside."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(source, arcname=source.name)
        archive.writestr("notes.txt", "not an asset")
    return path


def write_corrupt_usda(path: Path) -> Path:
    """A text layer cut off inside a tuple: a syntax error on line 9."""
    path.write_text(CORRUPT_USDA)
    return path


def write_empty(path: Path) -> Path:
    path.write_bytes(b"")
    return path


def write_garbage(path: Path) -> Path:
    """Bytes that are no USD format at all."""
    path.write_bytes(bytes(range(256)) * 16)
    return path


def write_text_named_usdc(path: Path) -> Path:
    """Valid text USD under a binary crate extension."""
    path.write_text('#usda 1.0\ndef Xform "World" {}\n')
    return path


def write_broken_refs(directory: Path) -> Path:
    """A layer whose sublayer, reference and payload are missing, plus a bad prim path."""
    target = write_robot(directory / "robot.usda")
    path = directory / "broken_ref.usda"
    stage = _stage_with_metadata(path)
    stage.GetRootLayer().subLayerPaths.append("./missing_sublayer.usda")
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Xform.Define(stage, "/World/MissingRef").GetPrim().GetReferences().AddReference(
        "./does_not_exist.usda"
    )
    UsdGeom.Xform.Define(stage, "/World/MissingPayload").GetPrim().GetPayloads().AddPayload(
        "./missing_payload.usd"
    )
    UsdGeom.Xform.Define(stage, "/World/BadRefPrim").GetPrim().GetReferences().AddReference(
        f"./{target.name}", "/NoSuchPrim"
    )
    stage.GetRootLayer().Save()
    return path


def write_dangling_default_prim(path: Path) -> Path:
    stage = _stage_with_metadata(path)
    UsdGeom.Xform.Define(stage, "/A")
    stage.GetRootLayer().defaultPrim = "Nope"
    stage.GetRootLayer().Save()
    return path


def write_no_metadata(path: Path) -> Path:
    """Two root prims and no defaultPrim, upAxis or metersPerUnit."""
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.Xform.Define(stage, "/A")
    UsdGeom.Xform.Define(stage, "/B")
    stage.GetRootLayer().Save()
    return path


def write_bad_physics(path: Path) -> Path:
    """Nested rigid bodies and articulation roots, mass -3, a dangling body0, inverted limits."""
    stage = _stage_with_metadata(path)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    outer = UsdGeom.Xform.Define(stage, "/World/Outer")
    UsdPhysics.RigidBodyAPI.Apply(outer.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(outer.GetPrim())
    inner = UsdGeom.Xform.Define(stage, "/World/Outer/Inner")
    UsdPhysics.RigidBodyAPI.Apply(inner.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(inner.GetPrim())
    UsdPhysics.MassAPI.Apply(inner.GetPrim()).CreateMassAttr(-3.0)
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/BadJoint")
    joint.CreateBody0Rel().SetTargets([Sdf.Path("/World/DoesNotExist")])
    joint.CreateBody1Rel().SetTargets([inner.GetPath()])
    joint.CreateLowerLimitAttr(90.0)
    joint.CreateUpperLimitAttr(-90.0)
    stage.GetRootLayer().Save()
    return path
