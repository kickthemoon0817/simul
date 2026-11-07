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


# Script cores shared by the single-step tools and create_isaac_object. Each
# defines one module-level function that takes the open stage plus plain
# parameters and returns the dict the tool prints; a dict carrying "error"
# reports failure without raising. They are Python *source*, not callables:
# the tools embed them in the generated script so the logic exists once and
# runs inside Kit, whichever tool invoked it.

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
    def _set_prim_transform(stage, prim_path, translation, rotation_euler, scale):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return {"error": "Prim not found: " + prim_path}
        if not prim.IsA(UsdGeom.Xformable):
            return {"error": "Prim is not Xformable: " + prim_path}
        xformable = UsdGeom.Xformable(prim)
        Op = UsdGeom.XformOp
        rotate_types = (
            Op.TypeRotateXYZ, Op.TypeRotateXZY, Op.TypeRotateYXZ, Op.TypeRotateYZX,
            Op.TypeRotateZXY, Op.TypeRotateZYX, Op.TypeRotateX, Op.TypeRotateY,
            Op.TypeRotateZ, Op.TypeOrient,
        )
        ops = list(xformable.GetOrderedXformOps())

        # The prim's own translate / rotate / scale are the unsuffixed,
        # non-inverse ops; pivots and other suffixed ops are left untouched.
        def _primary(types):
            for index, op in enumerate(ops):
                if op.IsInverseOp() or op.GetOpType() not in types:
                    continue
                if len(op.GetOpName().split(":")) == 2:
                    return index
            return None

        def _set_vec(op, value):
            precision = op.GetPrecision()
            if precision == Op.PrecisionDouble:
                op.Set(Gf.Vec3d(*value))
            elif precision == Op.PrecisionHalf:
                op.Set(Gf.Vec3h(*value))
            else:
                op.Set(Gf.Vec3f(*value))

        def _set_orient(op, euler):
            # rotateXYZ semantics: rotate about X first, then Y, then Z.
            quat = (
                Gf.Rotation(Gf.Vec3d(1, 0, 0), float(euler[0]))
                * Gf.Rotation(Gf.Vec3d(0, 1, 0), float(euler[1]))
                * Gf.Rotation(Gf.Vec3d(0, 0, 1), float(euler[2]))
            ).GetQuat()
            precision = op.GetPrecision()
            if precision == Op.PrecisionDouble:
                op.Set(Gf.Quatd(quat))
            elif precision == Op.PrecisionHalf:
                op.Set(Gf.Quath(quat))
            else:
                op.Set(Gf.Quatf(quat))

        def _op_for(attr_name, add_op, precision):
            # An attribute left over from an earlier op stack is reused;
            # adding the op again would fail on the existing attribute.
            attr = prim.GetAttribute(attr_name)
            return UsdGeom.XformOp(attr) if attr.IsValid() else add_op(precision)

        t_index = _primary((Op.TypeTranslate,))
        r_index = _primary(rotate_types)
        s_index = _primary((Op.TypeScale,))

        # A missing op is inserted at its canonical slot (translate, then
        # rotate, then scale) rather than appended: an op appended after an
        # existing scale would, for example, scale the translation.
        if translation is not None:
            if t_index is None:
                ops.insert(0, _op_for("xformOp:translate", xformable.AddTranslateOp, Op.PrecisionDouble))
                t_index = 0
                r_index = None if r_index is None else r_index + 1
                s_index = None if s_index is None else s_index + 1
            _set_vec(ops[t_index], translation)

        if rotation_euler is not None:
            rotate_op = None if r_index is None else ops[r_index]
            if rotate_op is not None and rotate_op.GetOpType() == Op.TypeOrient:
                _set_orient(rotate_op, rotation_euler)
            elif rotate_op is not None and rotate_op.GetOpType() == Op.TypeRotateXYZ:
                _set_vec(rotate_op, rotation_euler)
            else:
                # No rotation yet, or one in another axis order: the value is
                # XYZ Euler, so a rotateXYZ op takes that slot. An op of
                # another order leaves the stack; its attribute stays.
                new_op = _op_for("xformOp:rotateXYZ", xformable.AddRotateXYZOp, Op.PrecisionFloat)
                _set_vec(new_op, rotation_euler)
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
                if index == r_index
                or op.IsInverseOp()
                or op.GetOpType() not in rotate_types
                or len(op.GetOpName().split(":")) != 2
            ]
            r_index = next(index for index, op in enumerate(ops) if op is kept)
            t_index = _primary((Op.TypeTranslate,))
            s_index = _primary((Op.TypeScale,))

        if scale is not None:
            if s_index is None:
                anchor = r_index if r_index is not None else t_index
                s_index = 0 if anchor is None else anchor + 1
                ops.insert(s_index, _op_for("xformOp:scale", xformable.AddScaleOp, Op.PrecisionFloat))
            _set_vec(ops[s_index], scale)

        xformable.SetXformOpOrder(ops, xformable.GetResetXformStack())
        world = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        return {
            "prim_path": prim_path,
            "translation": list(world.ExtractTranslation()),
            "rotation_euler_set": rotation_euler,
            "scale_set": scale,
            "xform_op_order": [op.GetOpName() for op in ops],
        }
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
