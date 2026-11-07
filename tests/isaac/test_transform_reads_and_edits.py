"""Transform reads and edits, typed queries and attribute writes on real ``pxr``.

Each test reproduces a defect the tools shipped with, on a usd-core stage
that stands in for Isaac Sim (``conftest.run_on_stage`` runs the generated
script; the bridge's ``PrimDetailReader`` is called directly):

- rotation read with ``ExtractRotation()`` from a matrix that still held
  scale (90 degrees under a parent scaled by 2 read as 60), and scale read
  from the first scale op (a matrix op scaled by 3 read as 1);
- ``set_isaac_prim_transform`` stacking a translate on an
  ``xformOp:transform`` matrix, writing the default under time samples while
  reporting the request, and labelling a world translation as the result of
  a local edit;
- ``set_isaac_prim_attribute`` unable to write a quaternion or a matrix;
- an unset centre of mass emitted as ``-Infinity``;
- ``query_isaac_typed_prims`` crashing on Gf values, returning arrays
  unbounded, and matching nothing for ``UsdPhysics.*`` names.
"""

from __future__ import annotations

import asyncio
import json
import math
from typing import Any, Dict, List, Tuple

import pytest
from khemoo.simul.prim_detail import PrimDetailReader
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from simul.mcp.tools.isaac_tools import IsaacTools

Tools = Tuple[IsaacTools, List[str]]

HALF = math.sqrt(0.5)


def _stage() -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    return stage


def _call(
    tools: Tools,
    run_on_stage: Any,
    stage: Usd.Stage,
    method: str,
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    instance, captured = tools
    asyncio.run(getattr(instance, method)(*args, **kwargs))
    return run_on_stage(captured[-1], stage)


def _strict(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Round-trip through JSON that rejects NaN and Infinity, as MCP clients do."""
    return json.loads(json.dumps(payload, allow_nan=False))


def _transform(
    source: str,
    tools: Tools,
    run_on_stage: Any,
    stage: Usd.Stage,
    path: str,
    world: bool = True,
) -> Dict[str, Any]:
    if source == "bridge":
        return _strict(
            PrimDetailReader(stage).transform(stage.GetPrimAtPath(path), world)
        )
    return _strict(
        _call(
            tools,
            run_on_stage,
            stage,
            "get_isaac_prim_transform",
            path,
            world_space=world,
        )
    )


def _close(actual: List[float], expected: List[float], tol: float = 1e-6) -> bool:
    return len(actual) == len(expected) and all(
        abs(a - b) <= tol for a, b in zip(actual, expected)
    )


def _same_rotation(quat: List[float], expected: List[float]) -> bool:
    """Quaternions q and -q are the same rotation."""
    return _close(quat, expected) or _close([-c for c in quat], expected)


def _world(stage: Usd.Stage, path: str) -> Gf.Matrix4d:
    return UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )


SOURCES = pytest.mark.parametrize("source", ["script", "bridge"])


# ---------------------------------------------------------------------------
# Reads: rotation and scale come from the composed matrix, both spaces
# ---------------------------------------------------------------------------


@SOURCES
def test_rotation_under_a_scaled_parent_reads_true(
    source: str, capturing_tools: Tools, run_on_stage: Any
) -> None:
    """90 degrees about Z under a parent scaled by 2 used to read back as 60."""
    stage = _stage()
    UsdGeom.Xformable(stage.GetPrimAtPath("/World")).AddScaleOp().Set(Gf.Vec3f(2, 2, 2))
    UsdGeom.Xform.Define(stage, "/World/Link").AddRotateZOp().Set(90.0)

    world = _transform(source, capturing_tools, run_on_stage, stage, "/World/Link")

    assert world["space"] == "world"
    assert _same_rotation(world["rotation_quat"], [HALF, 0, 0, HALF])
    assert _close(world["rotation_euler_xyz"], [0, 0, 90])
    assert world["scale"] == [2.0, 2.0, 2.0], "world scale includes the parent"
    assert world["local"]["scale"] == [1.0, 1.0, 1.0]
    assert world["world"] == {
        k: world[k]
        for k in ("translation", "rotation_quat", "rotation_euler_xyz", "scale")
    }
    assert "sheared" not in world


@SOURCES
def test_rotation_under_a_non_uniform_parent_is_flagged_sheared(
    source: str, capturing_tools: Tools, run_on_stage: Any
) -> None:
    """30 degrees about X under a (1, 1, 3) parent used to read back as 0."""
    stage = _stage()
    UsdGeom.Xformable(stage.GetPrimAtPath("/World")).AddScaleOp().Set(Gf.Vec3f(1, 1, 3))
    UsdGeom.Xform.Define(stage, "/World/Link").AddRotateXOp().Set(30.0)

    world = _transform(source, capturing_tools, run_on_stage, stage, "/World/Link")

    assert _close(world["rotation_euler_xyz"], [30, 0, 0])
    assert world["sheared"] is True
    assert "sheared" in world["note"]
    assert (
        "sheared" not in world["local"]
    ), "the prim's own transform is a plain rotation"
    assert _close(world["local"]["rotation_euler_xyz"], [30, 0, 0])


@SOURCES
def test_matrix_op_scale_is_read_from_the_matrix(
    source: str, capturing_tools: Tools, run_on_stage: Any
) -> None:
    """A prim whose only op is a matrix scaled by 3 used to report [1, 1, 1]."""
    stage = _stage()
    UsdGeom.Xform.Define(stage, "/World/M").AddTransformOp().Set(
        Gf.Matrix4d().SetScale(Gf.Vec3d(3, 3, 3))
    )

    local = _transform(
        source, capturing_tools, run_on_stage, stage, "/World/M", world=False
    )

    assert local["space"] == "local"
    assert local["scale"] == [3.0, 3.0, 3.0]
    assert local["world"]["scale"] == [3.0, 3.0, 3.0]
    assert local["xform_op_order"] == ["xformOp:transform"]


@SOURCES
def test_info_aspect_reports_the_same_pose(
    source: str, capturing_tools: Tools, run_on_stage: Any
) -> None:
    stage = _stage()
    UsdGeom.Xformable(stage.GetPrimAtPath("/World")).AddScaleOp().Set(Gf.Vec3f(2, 2, 2))
    UsdGeom.Cube.Define(stage, "/World/Box").AddRotateZOp().Set(90.0)

    if source == "bridge":
        info = _strict(PrimDetailReader(stage).info(stage.GetPrimAtPath("/World/Box")))
    else:
        info = _strict(
            _call(
                capturing_tools,
                run_on_stage,
                stage,
                "get_isaac_prim_info",
                "/World/Box",
            )
        )

    assert info["transform"]["space"] == "world"
    assert _same_rotation(info["transform"]["rotation_quat"], [HALF, 0, 0, HALF])
    assert info["transform"]["scale"] == [2.0, 2.0, 2.0]


def test_camera_orientation_under_a_scaled_parent(
    capturing_tools: Tools, run_on_stage: Any
) -> None:
    stage = _stage()
    UsdGeom.Xformable(stage.GetPrimAtPath("/World")).AddScaleOp().Set(Gf.Vec3f(2, 2, 2))
    UsdGeom.Camera.Define(stage, "/World/Cam").AddRotateZOp().Set(90.0)

    info = _call(
        capturing_tools, run_on_stage, stage, "get_isaac_camera_info", "/World/Cam"
    )

    assert _same_rotation(info["rotation_quat"], [HALF, 0, 0, HALF])


def test_euler_read_back_matches_what_set_takes(
    capturing_tools: Tools, run_on_stage: Any
) -> None:
    """rotation_euler_xyz is in the order set_isaac_prim_transform writes."""
    stage = _stage()
    UsdGeom.Xform.Define(stage, "/World/A")

    result = _call(
        capturing_tools,
        run_on_stage,
        stage,
        "set_isaac_prim_transform",
        "/World/A",
        rotation_euler=[30, 45, 60],
    )

    assert _close(result["rotation_euler_xyz"], [30, 45, 60], 1e-4)


# ---------------------------------------------------------------------------
# set_isaac_prim_transform: matrix ops, time samples, labelled spaces
# ---------------------------------------------------------------------------


def test_translation_on_a_matrix_op_replaces_rather_than_stacks(
    capturing_tools: Tools, run_on_stage: Any
) -> None:
    """Asking for [5, 0, 0] on a matrix translated to (1, 2, 3) put the prim at (6, 2, 3)."""
    stage = _stage()
    matrix = (
        Gf.Matrix4d().SetScale(Gf.Vec3d(2, 2, 2))
        * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), 90))
        * Gf.Matrix4d().SetTranslate(Gf.Vec3d(1, 2, 3))
    )
    UsdGeom.Xform.Define(stage, "/World/Asset").AddTransformOp().Set(matrix)

    result = _call(
        capturing_tools,
        run_on_stage,
        stage,
        "set_isaac_prim_transform",
        "/World/Asset",
        translation=[5, 0, 0],
    )

    assert result["converted_matrix_op"] is True
    assert result["xform_op_order"] == [
        "xformOp:translate",
        "xformOp:orient",
        "xformOp:scale",
    ]
    assert list(_world(stage, "/World/Asset").ExtractTranslation()) == [5.0, 0.0, 0.0]
    assert result["translation"] == [5.0, 0.0, 0.0]
    # The matrix's rotation and scale survive the conversion.
    assert _same_rotation(result["rotation_quat"], [HALF, 0, 0, HALF])
    assert result["scale"] == [2.0, 2.0, 2.0]


def test_sheared_matrix_op_is_refused_unchanged(
    capturing_tools: Tools, run_on_stage: Any
) -> None:
    stage = _stage()
    sheared = Gf.Matrix4d(1, 0, 0, 0, 0.5, 1, 0, 0, 0, 0, 1, 0, 1, 2, 3, 1)
    UsdGeom.Xform.Define(stage, "/World/Skew").AddTransformOp().Set(sheared)

    result = _call(
        capturing_tools,
        run_on_stage,
        stage,
        "set_isaac_prim_transform",
        "/World/Skew",
        translation=[5, 0, 0],
    )

    assert "shear" in result["error"]
    prim = stage.GetPrimAtPath("/World/Skew")
    assert list(UsdGeom.Xformable(prim).GetXformOpOrderAttr().Get()) == [
        "xformOp:transform"
    ]
    assert not prim.HasAttribute("xformOp:translate")


def test_time_sampled_op_is_refused_without_a_time_code(
    capturing_tools: Tools, run_on_stage: Any
) -> None:
    """Only the default used to be written under the samples, yet the reply echoed the request."""
    stage = _stage()
    op = UsdGeom.Xform.Define(stage, "/World/Anim").AddTranslateOp()
    op.Set(Gf.Vec3d(0, 0, 0), 1.0)
    op.Set(Gf.Vec3d(0, 0, 1), 10.0)

    refused = _call(
        capturing_tools,
        run_on_stage,
        stage,
        "set_isaac_prim_transform",
        "/World/Anim",
        translation=[7, 7, 7],
    )

    assert "time_code" in refused["error"]
    assert refused["time_sampled_ops"] == ["xformOp:translate"]
    assert op.Get(Usd.TimeCode.Default()) is None, "nothing was written"

    written = _call(
        capturing_tools,
        run_on_stage,
        stage,
        "set_isaac_prim_transform",
        "/World/Anim",
        translation=[7, 7, 7],
        time_code=10,
    )

    assert written["time_code"] == 10.0
    assert written["translation"] == [7.0, 7.0, 7.0]
    assert op.Get(10.0) == Gf.Vec3d(7, 7, 7)
    assert op.Get(1.0) == Gf.Vec3d(0, 0, 0), "other samples are left alone"


def test_animated_rotate_op_is_not_silently_replaced(
    capturing_tools: Tools, run_on_stage: Any
) -> None:
    """Swapping an animated rotateZYX for a rotateXYZ would drop its animation."""
    stage = _stage()
    op = UsdGeom.Xform.Define(stage, "/World/Spin").AddRotateZYXOp()
    op.Set(Gf.Vec3f(0, 0, 0), 1.0)
    op.Set(Gf.Vec3f(0, 0, 90), 10.0)

    result = _call(
        capturing_tools,
        run_on_stage,
        stage,
        "set_isaac_prim_transform",
        "/World/Spin",
        rotation_euler=[0, 0, 45],
        time_code=5,
    )

    assert "rotateZYX" in result["error"]
    order = (
        UsdGeom.Xformable(stage.GetPrimAtPath("/World/Spin"))
        .GetXformOpOrderAttr()
        .Get()
    )
    assert list(order) == ["xformOp:rotateZYX"]


def test_reply_labels_local_input_and_world_result(
    capturing_tools: Tools, run_on_stage: Any
) -> None:
    """The reply used to call the world translation 'translation' after a local edit."""
    stage = _stage()
    UsdGeom.Xformable(stage.GetPrimAtPath("/World")).AddTranslateOp().Set(
        Gf.Vec3d(10, 0, 0)
    )
    UsdGeom.Xform.Define(stage, "/World/Child")

    result = _call(
        capturing_tools,
        run_on_stage,
        stage,
        "set_isaac_prim_transform",
        "/World/Child",
        translation=[1, 0, 0],
    )

    assert result["space"] == "local"
    assert result["translation"] == [1.0, 0.0, 0.0]
    assert result["translation_set"] == [1.0, 0.0, 0.0]
    assert result["world"]["translation"] == [11.0, 0.0, 0.0]


def test_create_object_reports_the_transform_step_in_both_spaces(
    capturing_tools: Tools, run_on_stage: Any
) -> None:
    stage = _stage()
    UsdGeom.Xformable(stage.GetPrimAtPath("/World")).AddTranslateOp().Set(
        Gf.Vec3d(0, 0, 1)
    )

    result = _call(
        capturing_tools,
        run_on_stage,
        stage,
        "create_isaac_object",
        "/World/Box",
        translation=[1, 0, 0],
    )

    step = result["steps"]["transform"]
    assert step["space"] == "local"
    assert step["translation"] == [1.0, 0.0, 0.0]
    assert step["world"]["translation"] == [1.0, 0.0, 1.0]


def test_short_vector_is_reported_not_raised(
    capturing_tools: Tools, run_on_stage: Any
) -> None:
    stage = _stage()
    UsdGeom.Xform.Define(stage, "/World/A")

    result = _call(
        capturing_tools,
        run_on_stage,
        stage,
        "set_isaac_prim_transform",
        "/World/A",
        translation=[1, 2],
    )

    assert "3 components" in result["error"]


# ---------------------------------------------------------------------------
# set_isaac_prim_attribute: quaternions, matrices, arrays
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "precision", [UsdGeom.XformOp.PrecisionFloat, UsdGeom.XformOp.PrecisionDouble]
)
def test_orient_takes_w_x_y_z(
    precision: Any, capturing_tools: Tools, run_on_stage: Any
) -> None:
    """xformOp:orient, Isaac's default rotation op, could not be set at all."""
    stage = _stage()
    UsdGeom.Xform.Define(stage, "/World/X").AddOrientOp(precision).Set(
        Gf.Quatf(1, 0, 0, 0)
        if precision == UsdGeom.XformOp.PrecisionFloat
        else Gf.Quatd(1, 0, 0, 0)
    )

    result = _call(
        capturing_tools,
        run_on_stage,
        stage,
        "set_isaac_prim_attribute",
        "/World/X",
        "xformOp:orient",
        [HALF, 0, 0, HALF],
    )

    assert "error" not in result
    assert _close(result["value_read"], [HALF, 0, 0, HALF], 1e-6)
    pose = _strict(PrimDetailReader(stage).transform(stage.GetPrimAtPath("/World/X")))
    assert _close(pose["rotation_euler_xyz"], [0, 0, 90], 1e-4)


def test_joint_local_rotation_takes_a_quaternion(
    capturing_tools: Tools, run_on_stage: Any
) -> None:
    stage = _stage()
    UsdPhysics.RevoluteJoint.Define(stage, "/World/Joint").CreateLocalRot0Attr(
        Gf.Quatf(1, 0, 0, 0)
    )

    result = _call(
        capturing_tools,
        run_on_stage,
        stage,
        "set_isaac_prim_attribute",
        "/World/Joint",
        "physics:localRot0",
        [0, 1, 0, 0],
    )

    assert result["type"] == "quatf"
    assert UsdPhysics.Joint(
        stage.GetPrimAtPath("/World/Joint")
    ).GetLocalRot0Attr().Get() == Gf.Quatf(0, 1, 0, 0)


@pytest.mark.parametrize(
    "value",
    [
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [4, 5, 6, 1]],
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 4, 5, 6, 1],
    ],
    ids=["rows", "flat"],
)
def test_matrix_takes_rows_or_a_flat_list(
    value: Any, capturing_tools: Tools, run_on_stage: Any
) -> None:
    stage = _stage()
    UsdGeom.Xform.Define(stage, "/World/M").AddTransformOp().Set(Gf.Matrix4d(1))

    result = _call(
        capturing_tools,
        run_on_stage,
        stage,
        "set_isaac_prim_attribute",
        "/World/M",
        "xformOp:transform",
        value,
    )

    assert result["value_read"][3] == [4.0, 5.0, 6.0, 1.0]
    assert list(_world(stage, "/World/M").ExtractTranslation()) == [4.0, 5.0, 6.0]


def test_quaternion_array_takes_a_list_of_quaternions(
    capturing_tools: Tools, run_on_stage: Any
) -> None:
    stage = _stage()
    prim = stage.DefinePrim("/World/Instancer", "PointInstancer")
    UsdGeom.PointInstancer(prim).CreateOrientationsAttr([Gf.Quath(1, 0, 0, 0)])

    result = _call(
        capturing_tools,
        run_on_stage,
        stage,
        "set_isaac_prim_attribute",
        "/World/Instancer",
        "orientations",
        [[1, 0, 0, 0], [0, 0, 0, 1]],
    )

    assert "error" not in result
    assert list(UsdGeom.PointInstancer(prim).GetOrientationsAttr().Get()) == [
        Gf.Quath(1, 0, 0, 0),
        Gf.Quath(0, 0, 0, 1),
    ]


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("xformOp:orient", [1, 0, 0], "[w, x, y, z]"),
        ("xformOp:translate", [1, 2], "3 components"),
    ],
)
def test_wrong_shape_names_the_expected_one(
    attribute: str, value: Any, message: str, capturing_tools: Tools, run_on_stage: Any
) -> None:
    stage = _stage()
    xform = UsdGeom.Xform.Define(stage, "/World/X")
    xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
    xform.AddOrientOp().Set(Gf.Quatf(1, 0, 0, 0))

    result = _call(
        capturing_tools,
        run_on_stage,
        stage,
        "set_isaac_prim_attribute",
        "/World/X",
        attribute,
        value,
    )

    assert message in result["error"]


def test_attribute_under_time_samples_warns(
    capturing_tools: Tools, run_on_stage: Any
) -> None:
    stage = _stage()
    size = UsdGeom.Cube.Define(stage, "/World/Cube").GetSizeAttr()
    size.Set(1.0, 1.0)

    result = _call(
        capturing_tools,
        run_on_stage,
        stage,
        "set_isaac_prim_attribute",
        "/World/Cube",
        "size",
        3.0,
    )

    assert result["value_read"] == 3.0
    assert "time samples" in result["warning"]


# ---------------------------------------------------------------------------
# Unset centre of mass is JSON, not -Infinity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("aspect", ["mass", "rigid_body"])
def test_unset_center_of_mass_is_null_with_a_note(
    aspect: str, capturing_tools: Tools, run_on_stage: Any
) -> None:
    stage = _stage()
    UsdGeom.Cube.Define(stage, "/World/Box")
    _call(capturing_tools, run_on_stage, stage, "add_isaac_rigid_body", "/World/Box")
    _call(
        capturing_tools,
        run_on_stage,
        stage,
        "set_isaac_mass_properties",
        "/World/Box",
        mass=2.0,
    )
    method = {
        "mass": "get_isaac_mass_properties",
        "rigid_body": "get_isaac_rigid_body_info",
    }[aspect]

    from_script = _strict(
        _call(capturing_tools, run_on_stage, stage, method, "/World/Box")
    )
    from_bridge = _strict(
        PrimDetailReader(stage).read(stage.GetPrimAtPath("/World/Box"), aspect)
    )

    assert from_script == from_bridge
    assert from_script["center_of_mass"] is None
    assert "Not authored" in from_script["center_of_mass_note"]
    assert from_script["mass"] == 2.0


def test_authored_center_of_mass_is_reported_without_a_note(
    capturing_tools: Tools, run_on_stage: Any
) -> None:
    stage = _stage()
    UsdGeom.Cube.Define(stage, "/World/Box")
    _call(
        capturing_tools,
        run_on_stage,
        stage,
        "set_isaac_mass_properties",
        "/World/Box",
        mass=2.0,
        center_of_mass=[0, 0, 1],
    )

    result = _strict(PrimDetailReader(stage).mass(stage.GetPrimAtPath("/World/Box")))

    assert result["center_of_mass"] == [0.0, 0.0, 1.0]
    assert result["center_of_mass_note"] is None


# ---------------------------------------------------------------------------
# query_isaac_typed_prims: Gf values, bounded arrays, schema resolution
# ---------------------------------------------------------------------------


def _query_stage() -> Usd.Stage:
    stage = _stage()
    mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
    mesh.CreatePointsAttr([(float(i), 0.0, 0.0) for i in range(100)])
    mesh.CreateExtentAttr([(0, 0, 0), (99, 0, 0)])
    mesh.AddOrientOp().Set(Gf.Quatf(HALF, 0, 0, HALF))
    UsdPhysics.RigidBodyAPI.Apply(mesh.GetPrim())
    revolute = UsdPhysics.RevoluteJoint.Define(stage, "/World/Hinge")
    revolute.CreateAxisAttr("Z")
    UsdPhysics.PrismaticJoint.Define(stage, "/World/Slider")
    return stage


def test_gf_values_serialize_and_arrays_are_bounded(
    capturing_tools: Tools, run_on_stage: Any
) -> None:
    """extent, points and xformOp:orient failed with 'Vec3f is not JSON serializable'."""
    stage = _query_stage()

    result = _call(
        capturing_tools,
        run_on_stage,
        stage,
        "query_usd_typed_prims",
        "UsdGeom.Mesh",
        ["extent", "points", "xformOp:orient"],
    )

    attributes = result["prims"][0]["attributes"]
    assert attributes["extent"] == [[0.0, 0.0, 0.0], [99.0, 0.0, 0.0]]
    assert attributes["points"] == {
        "type": "point3f[]",
        "length": 100,
        "elements_omitted": True,
    }
    assert _close(attributes["xformOp:orient"], [HALF, 0, 0, HALF])

    full = _call(
        capturing_tools,
        run_on_stage,
        stage,
        "query_usd_typed_prims",
        "UsdGeom.Mesh",
        ["points"],
        max_array_elements=1000,
    )
    assert len(full["prims"][0]["attributes"]["points"]) == 100


@pytest.mark.parametrize(
    ("type_name", "paths"),
    [
        ("UsdPhysics.RevoluteJoint", ["/World/Hinge"]),
        ("UsdPhysics.Joint", ["/World/Hinge", "/World/Slider"]),
        ("PhysicsRevoluteJoint", ["/World/Hinge"]),
        ("UsdPhysics.RigidBodyAPI", ["/World/Mesh"]),
        ("Mesh", ["/World/Mesh"]),
        ("UsdGeom.Xformable", ["/World", "/World/Mesh"]),
    ],
)
def test_schema_names_resolve(
    type_name: str, paths: List[str], capturing_tools: Tools, run_on_stage: Any
) -> None:
    """UsdPhysics.* names used to match nothing and report count 0, success."""
    stage = _query_stage()

    result = _call(
        capturing_tools, run_on_stage, stage, "query_usd_typed_prims", type_name
    )

    assert [prim["path"] for prim in result["prims"]] == paths


@pytest.mark.parametrize(
    "type_name",
    ["UsdPhysics.Nope", "NoSuchModule.Thing", "cube", "UsdGeom.PrimvarsAPI"],
)
def test_unknown_schema_names_are_refused(
    type_name: str, capturing_tools: Tools, run_on_stage: Any
) -> None:
    stage = _query_stage()

    result = _call(
        capturing_tools, run_on_stage, stage, "query_usd_typed_prims", type_name
    )

    assert "error" in result
    assert "prims" not in result


def test_unregistered_type_name_still_matches_prims_that_carry_it(
    capturing_tools: Tools, run_on_stage: Any
) -> None:
    """A type from a plugin this process lacks is still findable by its exact name."""
    stage = _stage()
    stage.DefinePrim("/World/Custom", "SomePluginType")

    result = _call(
        capturing_tools, run_on_stage, stage, "query_usd_typed_prims", "SomePluginType"
    )

    assert result["schema"]["kind"] == "type_name"
    assert [prim["path"] for prim in result["prims"]] == ["/World/Custom"]


def test_infinite_values_come_back_as_strict_json(
    capturing_tools: Tools, run_on_stage: Any
) -> None:
    """An open joint limit falls back to +/-inf."""
    stage = _query_stage()

    result = _strict(
        _call(
            capturing_tools,
            run_on_stage,
            stage,
            "query_usd_typed_prims",
            "UsdPhysics.RevoluteJoint",
            ["lowerLimit", "breakForce", "axis"],
        )
    )

    attributes = result["prims"][0]["attributes"]
    assert attributes == {"lowerLimit": None, "breakForce": None, "axis": "Z"}
    assert result["non_finite_values"] == 2


def test_asset_path_values_serialize(capturing_tools: Tools, run_on_stage: Any) -> None:
    stage = _stage()
    prim = UsdGeom.Xform.Define(stage, "/World/Tex").GetPrim()
    prim.CreateAttribute("texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("/textures/wood.png")
    )

    result = _call(
        capturing_tools,
        run_on_stage,
        stage,
        "query_usd_typed_prims",
        "UsdGeom.Xform",
        ["texture"],
    )

    assert result["prims"][1]["attributes"]["texture"] == "/textures/wood.png"
