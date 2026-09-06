"""Shared constants and helpers for the Isaac tool mixins."""

"""
Isaac Sim MCP tools for Simul MCP Server.

This module provides granular Isaac Sim tools over the repo-owned typed bridge
when available, falling back to raw script execution only when required by the
current coverage. For legacy compatibility, raw script execution can still use
the stock VS Code socket when bridge usage is disabled.
"""

import json
import textwrap
from typing import Annotated, Any, Callable, Dict, List, Optional

from pydantic import BeforeValidator

from ....adapters import IsaacSocketClient, ScriptResult
from ....config import Settings, get_settings
from ....logging import LoggerMixin, get_logger
from ....utils.paths import PathPolicy
from ...schemas.common import ErrorResponse

logger = get_logger(__name__)

# Extensions the MCP server speaks through. Disabling one saws off the
# transport the agent is using and leaves the instance unreachable.
PROTECTED_EXTENSIONS: frozenset[str] = frozenset(
    {
        "khemoo.simul.mcp",
        "isaacsim.code_editor.python_server",
        "isaacsim.code_editor.vscode",
    }
)

# Carb settings namespaces that configure those transports.
PROTECTED_CARB_SETTING_PREFIXES: tuple[str, ...] = (
    "/exts/khemoo.simul.mcp/",
    "/exts/isaacsim.code_editor.python_server/",
)

# Prim paths whose removal empties the scene rather than editing it.
STAGE_ROOT_PATH = "/"
DEFAULT_WORLD_PATH = "/World"


def _coerce_str_to_float_list(value: Any) -> Any:
    """Pydantic ``BeforeValidator``: accept ``'[1, 2, 3]'`` as a list of floats.

    Some MCP clients serialise list arguments as JSON-encoded strings. Pydantic
    2.12 dropped the implicit ``str`` -> ``list`` coercion that earlier versions
    accepted, so we normalise here before validation runs.
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

# Largest raw script accepted for execution.
MAX_SCRIPT_BYTES = 100_000

# 4K on the long edge. The old ceiling of 7680 allowed a 59-megapixel capture —
# tens of megabytes pushed through several buffers before anything checked
# whether the result could be delivered.
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
        requested = (
            (translation, UsdGeom.XformOp.TypeTranslate, xformable.AddTranslateOp, Gf.Vec3d),
            (rotation_euler, UsdGeom.XformOp.TypeRotateXYZ, xformable.AddRotateXYZOp, Gf.Vec3f),
            (scale, UsdGeom.XformOp.TypeScale, xformable.AddScaleOp, Gf.Vec3f),
        )
        for value, op_type, add_op, vector in requested:
            if value is None:
                continue
            existing = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == op_type]
            if existing:
                existing[0].Set(vector(*value))
            else:
                add_op().Set(vector(*value))
        world = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        return {
            "prim_path": prim_path,
            "translation": list(world.ExtractTranslation()),
            "rotation_euler_set": rotation_euler,
            "scale_set": scale,
        }
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

