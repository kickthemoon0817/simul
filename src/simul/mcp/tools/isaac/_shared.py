"""Shared constants and helpers for the Isaac tool mixins."""

import json
import textwrap
from typing import Annotated, Any, List

from pydantic import BeforeValidator

from ....logging import get_logger

logger = get_logger(__name__)

# Extensions the MCP server speaks through. Disabling one saws off the
# transport the agent is using and leaves the instance unreachable.
PROTECTED_EXTENSIONS: frozenset[str] = frozenset(
    {
        "khemoo.simul",
        "isaacsim.code_editor.python_server",
        "isaacsim.code_editor.vscode",
    }
)

# Carb settings namespaces that configure those transports: the bridge, 6.0's
# python_server and 5.x's VS Code socket (host, port, auth on 6.0).
PROTECTED_CARB_SETTING_PREFIXES: tuple[str, ...] = (
    "/exts/khemoo.simul/",
    "/exts/isaacsim.code_editor.python_server/",
    "/exts/isaacsim.code_editor.vscode/",
)

# Carb settings under this prefix are saved to the user's Kit configuration
# (Isaac Sim's .kit files set settings.persistent = true) and apply to every
# later session, not just this one.
PERSISTENT_CARB_SETTING_PREFIX = "/persistent/"

# Prim paths whose removal empties the scene rather than editing it.
STAGE_ROOT_PATH = "/"
DEFAULT_WORLD_PATH = "/World"


def _coerce_str_to_float_list(value: Any) -> Any:
    """Pydantic ``BeforeValidator``: accept ``'[1, 2, 3]'`` as a list of floats.

    Some MCP clients serialise list arguments as JSON-encoded strings, and
    Pydantic (>= 2.12) doesn't coerce ``str`` to ``list``, so normalise here.
    """
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return value
        if isinstance(parsed, list):
            return parsed
    return value


# List of floats that tolerates JSON-string inputs from forgiving MCP clients.
FloatList = Annotated[List[float], BeforeValidator(_coerce_str_to_float_list)]

# Aspect name -> the method that reads it. Fifteen tools took exactly one
# prim_path and differed only in which of these they returned, so they are one
# tool with a selector rather than fifteen the caller has to tell apart.
PRIM_DETAIL_ASPECTS = {
    "info": "get_isaac_prim_info",
    "transform": "get_isaac_prim_transform",
    "ancestors": "get_isaac_prim_ancestors",
    "relationships": "get_isaac_prim_relationships",
    "variants": "get_isaac_prim_variants",
    "bounding_box": "get_isaac_bounding_box",
    "mesh": "get_isaac_mesh_info",
    "light": "get_isaac_light_info",
    "material": "get_isaac_material_info",
    "rigid_body": "get_isaac_rigid_body_info",
    "collision": "get_isaac_collision_info",
    "joint": "get_isaac_joint_info",
    "mass": "get_isaac_mass_properties",
    "animation": "get_isaac_animation_info",
}

# get_isaac_texture_dependencies takes a *root* path and scans for materials
# beneath it, so as a per-prim "aspect" it answers a different question than
# its name promises: on the standard layout (materials under /World/Looks,
# mesh at /World/Cube) asking a mesh for its textures returns nothing. Left
# out until it can take a prim and resolve that prim's bound material.

# Largest raw script accepted for execution, in UTF-8 bytes, the unit the usage
# log's code_bytes reports a script's size in.
MAX_SCRIPT_BYTES = 100_000

# Script budget, in seconds, for operations that routinely outlast the default
# (the socket timeout less one second, 29 s): opening or creating a stage, a
# camera's first render while shaders compile, enabling a large extension. Past
# the budget the bridge interrupts the script while Kit finishes the work, so
# the tool reported failure for work that then completed. Kept under the
# bridge's per-script ceiling (its 120 s request timeout less 5 s) so the
# bridge still interrupts and reports instead of abandoning the request.
HEAVY_SCRIPT_TIMEOUT_SECONDS = 110.0

# step_isaac_simulation's wall-clock budget: the warm-up a stopped timeline
# spends after play() plus a floor of 10 frames per second (Isaac Sim's full
# app is rate-limited to 60 Hz, and heavy scenes run slower), never below the
# default script budget and never above the heavy one. Stepping stops and
# reports how far it got when the budget runs out.
STEP_BUDGET_BASE_SECONDS = 10.0
STEP_BUDGET_SECONDS_PER_FRAME = 0.1

# Script core shared by the step tool: the rates that decide how many physics
# steps one frame runs. PhysX consumes the timeline's time in steps of the
# physics scene's dt (Isaac Sim's SimulationContext derives its substeps as
# rendering_dt / physics_dt), so a frame is frame_dt * steps_per_second of
# them. Mirrors _physics_rates in the bridge's service.py.
PHYSICS_RATES_CORE = """\
    def _physics_rates(timeline, frame_dt, time_delta):
        rates = {
            "physics_scene": None,
            "physics_steps_per_second": None,
            "physics_dt": None,
            "rate_source": None,
            "timeline_time_codes_per_second": None,
            "rate_limit_hz": None,
            "physics_steps_per_frame": None,
            "physics_steps_estimated": None,
        }
        try:
            rates["timeline_time_codes_per_second"] = timeline.get_time_codes_per_seconds()
        except Exception:
            pass
        try:
            import carb.settings
            settings = carb.settings.get_settings()
            if settings.get("/app/runLoops/main/rateLimitEnabled"):
                rates["rate_limit_hz"] = settings.get("/app/runLoops/main/rateLimitFrequency")
        except Exception:
            pass
        rate = None
        try:
            import omni.usd
            from pxr import PhysxSchema, Usd, UsdPhysics
            stage = omni.usd.get_context().get_stage()
            prims = iter(Usd.PrimRange(stage.GetPseudoRoot())) if stage is not None else iter(())
            for prim in prims:
                # Only the top two levels: /physicsScene, /World/physicsScene.
                if prim.GetPath().pathElementCount >= 2:
                    prims.PruneChildren()
                if not prim.IsA(UsdPhysics.Scene):
                    continue
                rates["physics_scene"] = str(prim.GetPath())
                if prim.HasAPI(PhysxSchema.PhysxSceneAPI):
                    rate = PhysxSchema.PhysxSceneAPI(prim).GetTimeStepsPerSecondAttr().Get()
                    rates["rate_source"] = "physxScene:timeStepsPerSecond"
                else:
                    rate = 60.0
                    rates["rate_source"] = "PhysxSceneAPI fallback"
                break
        except Exception:
            pass
        if rate:
            rate = float(rate)
            rates["physics_steps_per_second"] = rate
            rates["physics_dt"] = 1.0 / rate
            if frame_dt:
                rates["physics_steps_per_frame"] = round(frame_dt * rate, 3)
            rates["physics_steps_estimated"] = int(round(time_delta * rate))
        return rates
"""

# 4K on the long edge, so a capture can't grow to tens of megabytes before
# anything checks whether the result can be delivered.
MAX_CAPTURE_DIMENSION = 3840

# Largest capture returned as inline base64. Encoding grows a file by ~33%, and
# ~256 KB of PNG is already a large tool result; past this the caller gets the
# path instead, which is the shape that keeps working.
MAX_INLINE_CAPTURE_BYTES = 262_144

# Captures are kept rather than deleted, since the caller is handed a path — so
# something has to reclaim them. Keep the most recent N and drop the rest: an
# A/B pair needs two, a comparison sweep a few more, and nobody wants the whole
# session's frames sitting in the temp dir.
MAX_RETAINED_CAPTURES = 20

# How much of a Kit log to read when returning its tail. Kit logs reach several
# gigabytes and the read happens on Kit's main thread, so the window bounds the
# freeze; 2 MB still holds far more lines than the 500-entry maximum.
LOG_SCAN_WINDOW_BYTES = 2 * 1024 * 1024

# Array attributes big enough that pulling their value is the cost being
# avoided. Gating on these names rather than on isArray matters: xformOpOrder
# is a one-element token[] telling a caller which xform ops exist, and
# primvars:displayColor is a one-element color3f[] carrying the object colour.
# Both are arrays, neither is bulk, and skipping them loses information the
# caller needs while saving nothing. Anything not listed here takes the normal
# path, which already collapses arrays over 16 elements to a count.
BULK_GEOMETRY_ATTRIBUTES = frozenset(
    {
        "points",
        "normals",
        "velocities",
        "accelerations",
        "faceVertexIndices",
        "faceVertexCounts",
        "holeIndices",
        "cornerIndices",
        "cornerSharpnesses",
        "creaseIndices",
        "creaseLengths",
        "creaseSharpnesses",
        "curveVertexCounts",
        "widths",
        "primvars:st",
        "primvars:normals",
        # PointInstancer
        "positions",
        "orientations",
        "scales",
        "protoIndices",
        "invisibleIds",
        "ids",
    }
)


def _pyval(value: Any) -> str:
    """Render a call parameter as a Python literal for embedding in a script.

    The generated scripts are Python source, so parameters must be embedded as
    Python literals. ``json.dumps`` spells ``None``/``True``/``False`` as
    ``null``/``true``/``false``, which are undefined names once they land in the
    script and raise ``NameError`` inside Isaac Sim. ``repr`` keeps them Python.

    Use this for every embedded parameter, including required ones — a single
    idiom is what keeps the ``null`` class of bug from coming back.
    """
    return repr(value)


def _compose_script(*fragments: str) -> str:
    """Join dedented source fragments into one module-level script.

    The script cores below are plain Python source. Embedding one inside an
    f-string template would indent only its first line, so each fragment is
    dedented on its own and the pieces are concatenated at column zero.

    Args:
        *fragments: Source fragments, each indented however its literal is.

    Returns:
        The concatenated script.
    """
    return "\n".join(textwrap.dedent(fragment) for fragment in fragments)


# What the mass and rigid_body aspects say beside a null centre of mass.
# MassAPI's fallback is (-inf, -inf, -inf), the normal state after
# set_isaac_mass_properties(mass=...); json.dumps spells it -Infinity, which
# is not JSON. The bridge's PrimDetailReader carries the same text.
UNSET_CENTER_OF_MASS_NOTE = (
    "Not authored: the schema fallback (-inf) means PhysX derives the centre of mass "
    "from the collision shapes."
)

# Script cores shared by the single-step tools and create_isaac_object. Each
# defines one module-level function that takes the open stage plus plain
# parameters and returns the dict the tool prints; a dict carrying "error"
# reports failure without raising. They are Python *source*, not callables:
# the tools embed them in the generated script so the logic exists once and
# runs inside Kit, whichever tool invoked it.

# Transform readback shared by every tool that reports a prim's pose. The
# bridge extension's PrimDetailReader carries the same logic (it cannot import
# this package inside Kit); test_prim_detail_bridge pins the two together.
# Needs ``Gf`` and ``Usd`` imported by the script.
XFORM_READ_CORE = """\
    def _xf_round(values):
        # Nine decimals drop float noise (6e-17 for zero, 2.0000000000000004)
        # without losing anything a pose needs; + 0.0 turns -0.0 into 0.0.
        return [round(float(value), 9) + 0.0 for value in values]

    def _decompose_matrix(matrix):
        # Matrix4d.ExtractRotation() orthonormalises a matrix that still
        # carries scale, so a rotation under a scaled parent reads wrong (90
        # degrees about Z under a parent scaled by 2 comes back as 60). The
        # first scale op is not the scale either: it misses parents and
        # matrix ops. Gf.Transform factors the scale out first.
        xform = Gf.Transform(matrix)
        rotation = xform.GetRotation()
        quat = rotation.GetQuat()
        # rotateXYZ composes X, then Y, then Z; Decompose takes the axes in
        # the reverse of that order, so read Z, Y, X and flip.
        z_y_x = rotation.Decompose(Gf.Vec3d(0, 0, 1), Gf.Vec3d(0, 1, 0), Gf.Vec3d(1, 0, 0))
        result = {
            "translation": _xf_round(matrix.ExtractTranslation()),
            "rotation_quat": _xf_round([quat.GetReal()] + list(quat.GetImaginary())),
            "rotation_euler_xyz": _xf_round(reversed(z_y_x)),
            "scale": _xf_round(xform.GetScale()),
        }
        # A rotated prim under a non-uniformly scaled parent has axes that
        # are no longer perpendicular: no rotation and per-axis scale
        # reproduce it, so say so rather than present the fit as exact.
        rows = [Gf.Vec3d(matrix[i][0], matrix[i][1], matrix[i][2]) for i in range(3)]
        for a, b in ((0, 1), (0, 2), (1, 2)):
            la = rows[a].GetLength()
            lb = rows[b].GetLength()
            if la > 1e-12 and lb > 1e-12 and abs(Gf.Dot(rows[a], rows[b])) > 1e-6 * la * lb:
                result["sheared"] = True
                break
        return result

    def _transform_report(xformable, world_space, time_code):
        local = _decompose_matrix(xformable.GetLocalTransformation(time_code))
        world = _decompose_matrix(xformable.ComputeLocalToWorldTransform(time_code))
        ops = xformable.GetOrderedXformOps()
        report = {"space": "world" if world_space else "local"}
        report.update(world if world_space else local)
        report["local"] = local
        report["world"] = world
        report["xform_op_order"] = [op.GetOpName() for op in ops]
        animated = [op.GetOpName() for op in ops if op.GetNumTimeSamples() > 0]
        if animated:
            report["time_sampled_ops"] = animated
        if xformable.GetResetXformStack():
            report["resets_xform_stack"] = True
        if local.get("sheared") or world.get("sheared"):
            report["note"] = (
                "The transform is sheared (a rotated prim under a non-uniform scale): "
                "rotation is the closest pure rotation and scale is measured along "
                "the stretch axes, not the prim's own."
            )
        return report
"""

# USD value -> JSON. json.dumps cannot take Gf vectors, quaternions, matrices
# or Vt arrays, and spells an infinite float (the -inf fallback of an unset
# centre of mass, a joint's open limit) as -Infinity, which is not JSON.
# Non-finite floats become null and are counted in _JSON_NON_FINITE so the
# caller can say so. Quaternions read [w, x, y, z], real part first, the
# order set_isaac_prim_attribute accepts. Needs ``Gf`` and ``Sdf``.
JSON_VALUE_CORE = """\
    import math as _jv_math

    _JSON_NON_FINITE = [0]

    def _json_value(value):
        if value is None or isinstance(value, (bool, int, str)):
            return value
        if isinstance(value, float):
            if _jv_math.isfinite(value):
                return value
            _JSON_NON_FINITE[0] += 1
            return None
        if isinstance(value, Sdf.AssetPath):
            return value.path
        if isinstance(value, (Gf.Quatf, Gf.Quatd, Gf.Quath)):
            return [_json_value(float(value.GetReal()))] + [_json_value(float(c)) for c in value.GetImaginary()]
        dimension = getattr(value, "dimension", None)
        if isinstance(dimension, tuple):
            return [[_json_value(float(value[i][j])) for j in range(dimension[1])] for i in range(dimension[0])]
        if hasattr(value, "__len__") and hasattr(value, "__getitem__"):
            try:
                return [_json_value(item) for item in value]
            except Exception:
                pass
        try:
            return _json_value(float(value))
        except Exception:
            return str(value)
"""

DEFINE_PRIM_CORE = """\
    def _define_prim(stage, prim_path, prim_type, attributes):
        existing = stage.GetPrimAtPath(prim_path)
        if existing.IsValid():
            return {"error": "Prim already exists: " + prim_path}
        prim = stage.DefinePrim(prim_path, prim_type)
        if not prim.IsValid():
            return {"error": "Failed to create prim: " + prim_path}
        for name, value in attributes.items():
            attr = prim.GetAttribute(name)
            if attr.IsValid():
                attr.Set(value)
        return {
            "prim_path": str(prim.GetPath()),
            "prim_type": prim.GetTypeName(),
            "created": True,
        }
"""

SET_PRIM_TRANSFORM_CORE = """\
    def _set_prim_transform(stage, prim_path, translation, rotation_euler, scale, time_code=None):
        # Input is the prim's *local* transform; the reply reads back both the
        # local and the world pose and labels which is which. Needs
        # XFORM_READ_CORE in the same script.
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return {"error": "Prim not found: " + prim_path}
        if not prim.IsA(UsdGeom.Xformable):
            return {"error": "Prim is not Xformable: " + prim_path}
        for label, value in (("translation", translation), ("rotation_euler", rotation_euler), ("scale", scale)):
            if value is not None and len(value) != 3:
                return {"error": label + " must have 3 components [x, y, z]; got " + str(len(value))}
        xformable = UsdGeom.Xformable(prim)
        Op = UsdGeom.XformOp
        default_time = Usd.TimeCode.Default()
        when = default_time if time_code is None else Usd.TimeCode(float(time_code))
        rotate_types = (
            Op.TypeRotateXYZ, Op.TypeRotateXZY, Op.TypeRotateYXZ, Op.TypeRotateYZX,
            Op.TypeRotateZXY, Op.TypeRotateZYX, Op.TypeRotateX, Op.TypeRotateY,
            Op.TypeRotateZ, Op.TypeOrient,
        )
        ops = list(xformable.GetOrderedXformOps())
        extra = {}

        def _animated(op):
            return op.GetAttr().GetNumTimeSamples() > 0

        # The prim's own translate / rotate / scale are the unsuffixed,
        # non-inverse ops; pivots and other suffixed ops are left untouched.
        def _is_primary(op, types):
            return (
                not op.IsInverseOp()
                and op.GetOpType() in types
                and len(op.GetOpName().split(":")) == 2
            )

        def _primary(types):
            for index, op in enumerate(ops):
                if _is_primary(op, types):
                    return index
            return None

        def _set_vec(op, value, at):
            precision = op.GetPrecision()
            if precision == Op.PrecisionDouble:
                op.Set(Gf.Vec3d(*value), at)
            elif precision == Op.PrecisionHalf:
                op.Set(Gf.Vec3h(*value), at)
            else:
                op.Set(Gf.Vec3f(*value), at)

        def _set_quat(op, quat, at):
            precision = op.GetPrecision()
            if precision == Op.PrecisionDouble:
                op.Set(Gf.Quatd(quat), at)
            elif precision == Op.PrecisionHalf:
                op.Set(Gf.Quath(quat), at)
            else:
                op.Set(Gf.Quatf(quat), at)

        def _euler_quat(euler):
            # rotateXYZ semantics: rotate about X first, then Y, then Z.
            return (
                Gf.Rotation(Gf.Vec3d(1, 0, 0), float(euler[0]))
                * Gf.Rotation(Gf.Vec3d(0, 1, 0), float(euler[1]))
                * Gf.Rotation(Gf.Vec3d(0, 0, 1), float(euler[2]))
            ).GetQuat()

        def _op_for(attr_name, add_op, precision):
            # An attribute left over from an earlier op stack is reused;
            # adding the op again would fail on the existing attribute.
            attr = prim.GetAttribute(attr_name)
            return UsdGeom.XformOp(attr) if attr.IsValid() else add_op(precision)

        def _hidden_error(names):
            return {
                "error": (
                    ", ".join(names) + " on " + prim_path + " is animated (time-sampled): a value "
                    "written at the default time is hidden by the samples. Pass time_code to "
                    "write a sample at that time."
                ),
                "time_sampled_ops": names,
            }

        # A matrix op holds the whole transform; a translate stacked on it
        # composes with the matrix's own translation instead of replacing it.
        # Swap it for the equivalent translate / orient / scale stack first.
        if any(op.GetOpType() == Op.TypeTransform for op in ops):
            leftovers = [prim.GetAttribute(name) for name in ("xformOp:translate", "xformOp:orient", "xformOp:scale")]
            animated = [op.GetOpName() for op in ops if _animated(op)] + [
                attr.GetName() for attr in leftovers if attr.IsValid() and attr.GetNumTimeSamples() > 0
            ]
            if animated:
                return {
                    "error": (
                        prim_path + " has a matrix op (xformOp:transform) in an animated op stack ("
                        + ", ".join(animated) + "); converting it to translate/orient/scale would "
                        "drop the animation. Set xformOp:transform per time code with "
                        "set_isaac_prim_attribute instead."
                    ),
                    "time_sampled_ops": animated,
                }
            local = xformable.GetLocalTransformation(default_time)
            parts = Gf.Transform(local)
            rebuilt = Gf.Transform()
            rebuilt.SetScale(parts.GetScale())
            rebuilt.SetRotation(parts.GetRotation())
            rebuilt.SetTranslation(parts.GetTranslation())
            if not Gf.IsClose(rebuilt.GetMatrix(), local, 1e-6):
                return {
                    "error": (
                        "The matrix op on " + prim_path + " carries shear, which translate/orient/"
                        "scale cannot express, so it was left unchanged. Set xformOp:transform "
                        "with set_isaac_prim_attribute instead."
                    )
                }
            t_op = _op_for("xformOp:translate", xformable.AddTranslateOp, Op.PrecisionDouble)
            r_op = _op_for("xformOp:orient", xformable.AddOrientOp, Op.PrecisionDouble)
            s_op = _op_for("xformOp:scale", xformable.AddScaleOp, Op.PrecisionDouble)
            _set_vec(t_op, parts.GetTranslation(), default_time)
            _set_quat(r_op, parts.GetRotation().GetQuat(), default_time)
            _set_vec(s_op, parts.GetScale(), default_time)
            ops = [t_op, r_op, s_op]
            extra["converted_matrix_op"] = True
            extra["matrix_op_note"] = (
                "The matrix op was replaced by the equivalent translate/orient/scale stack so its "
                "components could be set; xformOp:transform is no longer in xformOpOrder."
            )

        t_index = _primary((Op.TypeTranslate,))
        r_index = _primary(rotate_types)
        s_index = _primary((Op.TypeScale,))

        # Refuse before anything is written: a default value under time
        # samples changes nothing a viewer sees, and replacing an animated
        # rotate op would throw its animation away.
        if time_code is None:
            hidden = []
            if translation is not None and t_index is not None and _animated(ops[t_index]):
                hidden.append(ops[t_index].GetOpName())
            if (
                rotation_euler is not None
                and r_index is not None
                and ops[r_index].GetOpType() in (Op.TypeOrient, Op.TypeRotateXYZ)
                and _animated(ops[r_index])
            ):
                hidden.append(ops[r_index].GetOpName())
            if scale is not None and s_index is not None and _animated(ops[s_index]):
                hidden.append(ops[s_index].GetOpName())
            if hidden:
                return _hidden_error(hidden)
        if rotation_euler is not None:
            keeps = r_index is not None and ops[r_index].GetOpType() in (Op.TypeOrient, Op.TypeRotateXYZ)
            dropped = [
                op.GetOpName() for index, op in enumerate(ops)
                if _is_primary(op, rotate_types) and not (keeps and index == r_index) and _animated(op)
            ]
            if dropped:
                return {
                    "error": (
                        "rotation_euler would replace the animated rotate op(s) " + ", ".join(dropped)
                        + " on " + prim_path + " with an XYZ rotation and drop their animation; "
                        "set those ops with set_isaac_prim_attribute instead."
                    ),
                    "time_sampled_ops": dropped,
                }

        # A missing op is inserted at its canonical slot (translate, then
        # rotate, then scale) rather than appended: an op appended after an
        # existing scale would, for example, scale the translation.
        if translation is not None:
            if t_index is None:
                ops.insert(0, _op_for("xformOp:translate", xformable.AddTranslateOp, Op.PrecisionDouble))
                t_index = 0
                r_index = None if r_index is None else r_index + 1
                s_index = None if s_index is None else s_index + 1
            _set_vec(ops[t_index], translation, when)

        if rotation_euler is not None:
            rotate_op = None if r_index is None else ops[r_index]
            if rotate_op is not None and rotate_op.GetOpType() == Op.TypeOrient:
                _set_quat(rotate_op, _euler_quat(rotation_euler), when)
            elif rotate_op is not None and rotate_op.GetOpType() == Op.TypeRotateXYZ:
                _set_vec(rotate_op, rotation_euler, when)
            else:
                # No rotation yet, or one in another axis order: the value is
                # XYZ Euler, so a rotateXYZ op takes that slot. An op of
                # another order leaves the stack; its attribute stays.
                new_op = _op_for("xformOp:rotateXYZ", xformable.AddRotateXYZOp, Op.PrecisionFloat)
                _set_vec(new_op, rotation_euler, when)
                if r_index is not None:
                    ops[r_index] = new_op
                else:
                    if s_index is not None:
                        r_index = s_index
                        s_index += 1
                    else:
                        r_index = 0 if t_index is None else t_index + 1
                    ops.insert(r_index, new_op)
            # The value is the prim's whole rotation. Any other unsuffixed
            # rotate op (a split rotateX / rotateY / rotateZ stack) would
            # compose on top of it, so those leave the stack too.
            kept = ops[r_index]
            ops = [
                op for index, op in enumerate(ops)
                if index == r_index or not _is_primary(op, rotate_types)
            ]
            r_index = next(index for index, op in enumerate(ops) if op is kept)
            t_index = _primary((Op.TypeTranslate,))
            s_index = _primary((Op.TypeScale,))

        if scale is not None:
            if s_index is None:
                anchor = r_index if r_index is not None else t_index
                s_index = 0 if anchor is None else anchor + 1
                ops.insert(s_index, _op_for("xformOp:scale", xformable.AddScaleOp, Op.PrecisionFloat))
            _set_vec(ops[s_index], scale, when)

        xformable.SetXformOpOrder(ops, xformable.GetResetXformStack())
        report = {"prim_path": prim_path}
        report.update(_transform_report(xformable, False, when))
        report["time_code"] = None if time_code is None else float(time_code)
        report["translation_set"] = translation
        report["rotation_euler_set"] = rotation_euler
        report["scale_set"] = scale
        report.update(extra)
        return report
"""

OPEN_ASSET_CORE = """\
    def _open_asset(path):
        # AddReference accepts a missing or unreadable file silently and
        # composes nothing; opening the layer first makes that a failure.
        try:
            layer = Sdf.Layer.FindOrOpen(path)
        except Exception as exc:
            return "Asset is not a readable USD layer: " + path + " (" + str(exc).strip() + ")"
        if layer is None:
            return "Asset not found: " + path
        return None
"""

APPLY_RIGID_BODY_CORE = """\
    def _apply_rigid_body(stage, prim_path, kinematic):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return {"error": "Prim not found: " + prim_path}
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return {"error": "RigidBodyAPI already applied: " + prim_path}
        UsdPhysics.RigidBodyAPI.Apply(prim)
        if kinematic:
            UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Set(True)
        return {
            "prim_path": prim_path,
            "rigid_body_applied": True,
            "kinematic": kinematic,
        }
"""

APPLY_COLLISION_CORE = """\
    def _apply_collision(stage, prim_path, approximation):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return {"error": "Prim not found: " + prim_path}
        UsdPhysics.CollisionAPI.Apply(prim)
        if approximation != "none":
            UsdPhysics.MeshCollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Set(approximation)
        return {
            "prim_path": prim_path,
            "collision_applied": True,
            "approximation": approximation,
        }
"""

SET_MASS_PROPERTIES_CORE = """\
    def _set_mass_properties(stage, prim_path, mass, density, center_of_mass):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return {"error": "Prim not found: " + prim_path}
        if not prim.HasAPI(UsdPhysics.MassAPI):
            UsdPhysics.MassAPI.Apply(prim)
        mass_api = UsdPhysics.MassAPI(prim)
        if mass is not None:
            mass_api.GetMassAttr().Set(mass)
        if density is not None:
            mass_api.GetDensityAttr().Set(density)
        if center_of_mass is not None:
            mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(*center_of_mass))
        # The schema fallback is (-inf, -inf, -inf), meaning "derive from the
        # shape"; json.dumps would spell that as -Infinity, which is not JSON.
        com = mass_api.GetCenterOfMassAttr().Get()
        authored = com is not None and all(abs(c) != float("inf") for c in com)
        return {
            "prim_path": prim_path,
            "mass": mass_api.GetMassAttr().Get(),
            "density": mass_api.GetDensityAttr().Get(),
            "center_of_mass": list(com) if authored else None,
        }
"""

DEFINE_MATERIAL_CORE = """\
    def _define_material(stage, material_path, shader_type, diffuse_color, roughness, metallic, opacity):
        existing = stage.GetPrimAtPath(material_path)
        if existing.IsValid():
            return {"error": "Prim already exists at " + material_path}
        mat = UsdShade.Material(stage.DefinePrim(material_path, "Material"))
        shader_path = material_path + "/Shader"
        shader = UsdShade.Shader(stage.DefinePrim(shader_path, "Shader"))
        color = Gf.Vec3f(*diffuse_color)
        if shader_type == "OmniPBR":
            shader.CreateIdAttr("OmniPBR")
            shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
            shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
            shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
            shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(color)
            shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(roughness)
            shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(metallic)
            shader.CreateInput("enable_opacity", Sdf.ValueTypeNames.Bool).Set(opacity < 1.0)
            if opacity < 1.0:
                shader.CreateInput("opacity_constant", Sdf.ValueTypeNames.Float).Set(opacity)
            out = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
            mat.CreateSurfaceOutput("mdl").ConnectToSource(out)
        else:
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
            shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
            out = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
            mat.CreateSurfaceOutput().ConnectToSource(out)
        return {
            "material_path": material_path,
            "shader_path": shader_path,
            "shader_type": shader_type,
            "diffuse_color": list(diffuse_color),
            "roughness": roughness,
            "metallic": metallic,
            "opacity": opacity,
            "created": True,
        }
"""

BIND_MATERIAL_CORE = """\
    def _bind_material(stage, prim_path, material_path):
        prim = stage.GetPrimAtPath(prim_path)
        mat_prim = stage.GetPrimAtPath(material_path)
        if not prim.IsValid():
            return {"error": "Prim not found: " + prim_path}
        if not mat_prim.IsValid() or not mat_prim.IsA(UsdShade.Material):
            return {"error": "Material not found: " + material_path}
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material(mat_prim))
        return {
            "prim_path": prim_path,
            "material_path": material_path,
            "bound": True,
        }
"""

# Collision approximations UsdPhysics.MeshCollisionAPI accepts, plus "none"
# for a plain CollisionAPI without a mesh approximation.
COLLISION_APPROXIMATIONS: tuple[str, ...] = (
    "none",
    "convexHull",
    "convexDecomposition",
    "meshSimplification",
    "boundingSphere",
    "boundingCube",
)

# Default cap on list-shaped tool results, applied inside the generated script
# so the trim happens before Kit serialises the payload.
DEFAULT_MAX_RESULTS = 200

# Annotations: checks appended to a tool script, after the line printing the
# tool's own JSON, that report a condition about the result rather than the
# result itself. Each prints one line starting with ANNOTATION_MARKER followed
# by {"fields": {...}, "warnings": [...]}; the tool layer strips those lines
# before parsing the result and merges them into it. Each is a function that
# is called and deleted, so the shared script namespace keeps no names, and
# swallows every exception: an annotation must never fail a tool that worked.
ANNOTATION_MARKER = "[simul-annotation] "

# Isaac Sim's SimulationManager switches PhysX to Fabric for GPU physics:
# enable_fabric() enables omni.physx.fabric and sets /physics/updateToUsd and
# /physics/updateVelocitiesToUsd to false (v5.1.0 simulation_manager.py
# set_physics_sim_device / enable_fabric; v6.0.1 set_device). From then on the
# simulation no longer writes poses and velocities back to USD, so every USD
# read below returns the values from before play while the robot moves.
USD_READBACK_STALE_WARNING = (
    "PhysX is writing simulation results to Fabric, not USD "
    "(/physics/updateToUsd or /physics/updateVelocitiesToUsd is false, as Isaac "
    "Sim's SimulationManager sets for GPU physics), so the poses and velocities "
    "read from USD here are the values from before play, not the simulated state. "
    "Read the simulated state through a physics view (omni.physics.tensors, or "
    "isaacsim.core.experimental.prims on 6.0) with execute_isaac_script."
)

USD_READBACK_ANNOTATION = f'''\
def _simul_annotate_usd_readback():
    try:
        import json
        import carb.settings
        import omni.timeline
        if omni.timeline.get_timeline_interface().is_stopped():
            return
        settings = carb.settings.get_settings()
        update_to_usd = settings.get("/physics/updateToUsd")
        update_velocities_to_usd = settings.get("/physics/updateVelocitiesToUsd")
        fabric_enabled = None
        try:
            import omni.kit.app
            manager = omni.kit.app.get_app().get_extension_manager()
            fabric_enabled = bool(manager.is_extension_enabled("omni.physx.fabric"))
        except Exception:
            pass
        stale = update_to_usd is False or update_velocities_to_usd is False
        if not stale and not fabric_enabled:
            return
        fields = {{
            "usd_readback": {{
                "update_to_usd": update_to_usd,
                "update_velocities_to_usd": update_velocities_to_usd,
                "physx_fabric_enabled": fabric_enabled,
                "stale_under_fabric": stale,
            }},
        }}
        warnings = []
        if stale:
            fields["stale_under_fabric"] = True
            warnings.append({USD_READBACK_STALE_WARNING!r})
        print({ANNOTATION_MARKER!r} + json.dumps({{"fields": fields, "warnings": warnings}}))
    except Exception:
        pass


_simul_annotate_usd_readback()
del _simul_annotate_usd_readback
'''

# Isaac Sim's apps set physics.resetOnStop = true (base and full .kit files of
# 5.1.0 and 6.0.1): stopping or resetting restores what the simulation moved to
# its pre-play state. step_isaac_simulation leaves the timeline paused, so an
# edit made then to a simulated prim is reverted by the next stop or reset.
PAUSED_EDIT_WARNING = (
    "The timeline is paused (step_isaac_simulation leaves it paused) and "
    "/physics/resetOnStop is on: stopping or resetting the simulation restores the "
    "prims it simulates to their pre-play state, which also reverts edits made now "
    "to those prims. Stop the simulation before editing to keep the edit."
)

PAUSED_EDIT_ANNOTATION = f'''\
def _simul_annotate_paused_edit():
    try:
        import json
        import carb.settings
        import omni.timeline
        timeline = omni.timeline.get_timeline_interface()
        if timeline.is_playing() or timeline.is_stopped():
            return
        if carb.settings.get_settings().get("/physics/resetOnStop") is False:
            return
        note = {{"fields": {{"timeline_state": "paused"}}, "warnings": [{PAUSED_EDIT_WARNING!r}]}}
        print({ANNOTATION_MARKER!r} + json.dumps(note))
    except Exception:
        pass


_simul_annotate_paused_edit()
del _simul_annotate_paused_edit
'''

SCRIPT_ANNOTATIONS: dict[str, str] = {
    "usd_readback": USD_READBACK_ANNOTATION,
    "paused_edit": PAUSED_EDIT_ANNOTATION,
}
