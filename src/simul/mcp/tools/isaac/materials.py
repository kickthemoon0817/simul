"""Materials & Appearance tools for Isaac Sim."""

import json
import textwrap
from typing import Any, Dict, Optional

from ...schemas.common import ErrorResponse
from ._shared import (
    ASSET_SANDBOX_CORE,
    BIND_MATERIAL_CORE,
    DEFINE_MATERIAL_CORE,
    TRAVERSAL_PREDICATE_CORE,
    FloatList,
    _compose_script,
    _listing_predicate,
    _pyval,
)
from .._meta import tool_meta

# The inputs of the two surface shaders simul creates, with the Sdf value type
# each is declared with: OmniPBR.mdl's parameters (MDL enums are int inputs in
# USD) and the UsdPreviewSurface specification. An input that does not exist
# on the shader yet is created with this type; a name missing from the table
# is a typo far more often than a real input, so it is refused.
OMNIPBR_INPUTS: Dict[str, str] = {
    "diffuse_color_constant": "Color3f",
    "diffuse_texture": "Asset",
    "albedo_desaturation": "Float",
    "albedo_add": "Float",
    "albedo_brightness": "Float",
    "diffuse_tint": "Color3f",
    "reflection_roughness_constant": "Float",
    "reflection_roughness_texture_influence": "Float",
    "reflectionroughness_texture": "Asset",
    "metallic_constant": "Float",
    "metallic_texture_influence": "Float",
    "metallic_texture": "Asset",
    "specular_level": "Float",
    "enable_ORM_texture": "Bool",
    "ORM_texture": "Asset",
    "ao_to_diffuse": "Float",
    "ao_texture": "Asset",
    "enable_emission": "Bool",
    "emissive_color": "Color3f",
    "emissive_color_texture": "Asset",
    "emissive_mask_texture": "Asset",
    "emissive_intensity": "Float",
    "enable_opacity": "Bool",
    "enable_opacity_texture": "Bool",
    "opacity_constant": "Float",
    "opacity_texture": "Asset",
    "opacity_mode": "Int",
    "opacity_threshold": "Float",
    "bump_factor": "Float",
    "normalmap_texture": "Asset",
    "detail_bump_factor": "Float",
    "detail_normalmap_texture": "Asset",
    "flip_tangent_u": "Bool",
    "flip_tangent_v": "Bool",
    "project_uvw": "Bool",
    "world_or_object": "Bool",
    "uv_space_index": "Int",
    "texture_translate": "Float2",
    "texture_rotate": "Float",
    "texture_scale": "Float2",
    "detail_texture_translate": "Float2",
    "detail_texture_rotate": "Float",
    "detail_texture_scale": "Float2",
}

# The shader types create_isaac_material builds, keyed by lower-case name.
MATERIAL_SHADER_TYPES: Dict[str, str] = {
    "usdpreviewsurface": "UsdPreviewSurface",
    "omnipbr": "OmniPBR",
}

USD_PREVIEW_SURFACE_INPUTS: Dict[str, str] = {
    "diffuseColor": "Color3f",
    "emissiveColor": "Color3f",
    "useSpecularWorkflow": "Int",
    "specularColor": "Color3f",
    "metallic": "Float",
    "roughness": "Float",
    "clearcoat": "Float",
    "clearcoatRoughness": "Float",
    "opacity": "Float",
    "opacityMode": "Token",
    "opacityThreshold": "Float",
    "ior": "Float",
    "normal": "Normal3f",
    "displacement": "Float",
    "occlusion": "Float",
}

# Sets one surface-shader input with the value converted to the input's type.
# Defines _set_material_input(stage, material_path, name, value, allow_unknown,
# asset_checks). Needs ASSET_SANDBOX_CORE.
SET_MATERIAL_INPUT_CORE = """\
    import difflib

    KNOWN_INPUTS = {"OmniPBR": OMNIPBR_INPUTS, "UsdPreviewSurface": USD_PREVIEW_SURFACE_INPUTS}
    IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".exr", ".hdr", ".tga", ".tif", ".tiff", ".bmp", ".dds", ".ktx")
    VECTOR_TYPES = {
        "float2": (Gf.Vec2f, 2), "double2": (Gf.Vec2d, 2), "texcoord2f": (Gf.Vec2f, 2),
        "float3": (Gf.Vec3f, 3), "color3f": (Gf.Vec3f, 3), "normal3f": (Gf.Vec3f, 3),
        "vector3f": (Gf.Vec3f, 3), "point3f": (Gf.Vec3f, 3),
        "double3": (Gf.Vec3d, 3), "color3d": (Gf.Vec3d, 3),
        "float4": (Gf.Vec4f, 4), "color4f": (Gf.Vec4f, 4), "double4": (Gf.Vec4d, 4),
    }

    def _shader_kind(shader):
        sub_id = shader.GetPrim().GetAttribute("info:mdl:sourceAsset:subIdentifier")
        if sub_id and sub_id.Get():
            return str(sub_id.Get())
        shader_id = shader.GetIdAttr().Get()
        return str(shader_id) if shader_id else None

    def _inferred_type(name, value):
        # Only for an input no table knows: bool before int, since a bool is
        # an int in Python, and a path-looking string is an asset.
        if isinstance(value, bool):
            return "Bool"
        if isinstance(value, int):
            return "Int"
        if isinstance(value, float):
            return "Float"
        if isinstance(value, str):
            lowered = name.lower()
            if lowered.endswith("texture") or lowered.endswith("_file") or value.lower().endswith(IMAGE_SUFFIXES):
                return "Asset"
            return "String"
        if isinstance(value, list) and len(value) in (2, 3, 4):
            return {2: "Float2", 3: "Color3f", 4: "Float4"}[len(value)]
        return None

    def _coerce(value, type_name):
        kind = type_name.lower()
        number = isinstance(value, (int, float)) and not isinstance(value, bool)
        if kind == "bool":
            if isinstance(value, bool):
                return value
            if number and value in (0, 1):
                return bool(value)
            raise ValueError("expects true or false")
        if kind in ("int", "uint", "uchar", "int64", "uint64"):
            if number and float(value).is_integer():
                return int(value)
            raise ValueError("expects an integer")
        if kind in ("float", "double", "half"):
            if number:
                return float(value)
            raise ValueError("expects a number")
        if kind == "asset":
            if isinstance(value, str):
                return Sdf.AssetPath(value)
            raise ValueError("expects a file path or URL")
        if kind in ("string", "token"):
            if isinstance(value, str):
                return value
            raise ValueError("expects a string")
        if kind in VECTOR_TYPES:
            vector, size = VECTOR_TYPES[kind]
            if (
                isinstance(value, list) and len(value) == size
                and all(isinstance(c, (int, float)) and not isinstance(c, bool) for c in value)
            ):
                return vector(*[float(c) for c in value])
            raise ValueError("expects a list of " + str(size) + " numbers")
        raise ValueError("has a type this tool does not set; use set_isaac_prim_attribute")

    def _jsonable(value):
        if isinstance(value, Sdf.AssetPath):
            return value.path
        if hasattr(value, "__len__") and not isinstance(value, str):
            return [float(c) for c in value]
        return value

    def _set_material_input(stage, material_path, name, value, allow_unknown, asset_checks=None):
        prim = stage.GetPrimAtPath(material_path)
        if not prim.IsValid() or not prim.IsA(UsdShade.Material):
            return {"error": "Material not found: " + material_path, "error_type": "NotFoundError"}
        material = UsdShade.Material(prim)
        source = material.ComputeSurfaceSource()
        shader = source[0] if source else None
        if not shader:
            source = material.ComputeSurfaceSource("mdl")
            shader = source[0] if source else None
        if not shader:
            return {
                "error": "No surface shader found on material " + material_path,
                "error_type": "NotFoundError",
            }
        if name.startswith("inputs:"):
            name = name[len("inputs:"):]
        kind = _shader_kind(shader)
        known = KNOWN_INPUTS.get(kind)
        result = {
            "material_path": material_path,
            "shader_path": str(shader.GetPath()),
            "shader_type": kind,
            "property_name": name,
        }
        shader_input = shader.GetInput(name)
        created = False
        if shader_input and shader_input.GetAttr().IsValid():
            type_name = str(shader_input.GetTypeName())
        else:
            if known is not None and name in known:
                type_name = known[name]
            elif known is not None and not allow_unknown:
                close = difflib.get_close_matches(name, list(known), n=3, cutoff=0.5)
                message = (
                    name + " is not an input of " + kind + "; setting it would create an input "
                    "the shader never reads"
                )
                if close:
                    message += ". Did you mean: " + ", ".join(close) + "?"
                message += " Pass allow_unknown_input=true to create it anyway."
                result["error"] = message
                result["error_type"] = "ValidationError"
                result["details"] = {"code": "UNKNOWN_INPUT", "suggestions": close}
                result["known_inputs"] = sorted(known)
                return result
            else:
                type_name = _inferred_type(name, value)
                if type_name is None:
                    result["error"] = "Cannot infer an input type for value " + json.dumps(value)
                    result["error_type"] = "ValidationError"
                    result["details"] = {"code": "UNINFERABLE_INPUT_TYPE"}
                    return result
                result["warning"] = (
                    "Created input " + name + " as " + type_name + ", inferred from the value: "
                    + ("it is not a known " + kind + " input" if known is not None
                       else "no input list is known for shader type " + str(kind))
                )
            shader_input = shader.CreateInput(name, getattr(Sdf.ValueTypeNames, type_name))
            created = True
            type_name = str(shader_input.GetTypeName())
        if isinstance(value, str):
            # A file the renderer will read is checked against the read
            # sandbox, as create_isaac_light checks its texture_file; the
            # path written is the one the check resolved. Other values get
            # _coerce's type error.
            value, asset_denial = _sandboxed_asset(shader_input.GetTypeName(), value, asset_checks)
            if asset_denial is not None:
                if created:
                    shader.GetPrim().RemoveProperty(shader_input.GetAttr().GetName())
                denied = dict(asset_denial)
                denied.update({"material_path": material_path, "property_name": name})
                return denied
        try:
            coerced = _coerce(value, type_name)
        except ValueError as exc:
            if created:
                shader.GetPrim().RemoveProperty(shader_input.GetAttr().GetName())
            result["error"] = "Input " + name + " (" + type_name + ") " + str(exc) + "; got " + json.dumps(value)
            result["error_type"] = "ValidationError"
            result["details"] = {"code": "VALUE_TYPE_MISMATCH", "input_type": type_name}
            return result
        if not shader_input.Set(coerced):
            if created:
                shader.GetPrim().RemoveProperty(shader_input.GetAttr().GetName())
            result["error"] = "USD refused the value for input " + name + " (" + type_name + ")"
            return result
        result["input_type"] = type_name
        result["created"] = created
        result["value_set"] = _jsonable(shader_input.Get())
        return result
"""


class MaterialsMixin:
    # ------------------------------------------------------------------
    # Phase 7: Materials & Appearance
    # ------------------------------------------------------------------

    async def get_isaac_material_info(
        self, material_path: str
    ) -> Dict[str, Any]:
        """
        Get information about a USD material.

        Args:
            material_path: USD path of the Material prim.

        Returns:
            Dict with material shader, inputs, and bound prims.
        """
        _mat_path = _pyval(material_path)
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import UsdShade

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                prim = stage.GetPrimAtPath({_mat_path})
                if not prim.IsValid():
                    print(json.dumps({{"error": "Prim not found: " + {_mat_path}}}))
                elif not prim.IsA(UsdShade.Material):
                    print(json.dumps({{"error": "Prim is not a Material: " + {_mat_path}}}))
                else:
                    mat = UsdShade.Material(prim)
                    # Try default render context, then MDL
                    shader_result = mat.ComputeSurfaceSource()
                    shader_obj = shader_result[0] if shader_result else None
                    render_context = ""
                    if not shader_obj:
                        shader_result = mat.ComputeSurfaceSource("mdl")
                        shader_obj = shader_result[0] if shader_result else None
                        render_context = "mdl"
                    shader_path = None
                    shader_type = None
                    inputs = {{}}
                    if shader_obj:
                        shader_path = str(shader_obj.GetPath())
                        sp = shader_obj.GetPrim()
                        sub_id = sp.GetAttribute("info:mdl:sourceAsset:subIdentifier")
                        if sub_id and sub_id.Get():
                            shader_type = str(sub_id.Get())
                        else:
                            sid = shader_obj.GetIdAttr().Get()
                            shader_type = str(sid) if sid else None
                        for inp in shader_obj.GetInputs():
                            val = inp.Get()
                            name = inp.GetBaseName()
                            if val is not None:
                                try:
                                    if hasattr(val, '__len__') and not isinstance(val, str):
                                        inputs[name] = [float(x) for x in val]
                                    else:
                                        inputs[name] = float(val) if isinstance(val, (int, float)) else str(val)
                                except Exception:
                                    inputs[name] = str(val)
                    print(json.dumps({{
                        "material_path": {_mat_path},
                        "shader_path": shader_path,
                        "shader_type": shader_type,
                        "render_context": render_context,
                        "inputs": inputs,
                    }}))
        """)
        return await self._execute_json_script(script)

    @tool_meta(
        name="list_isaac_materials",
        description=(
            "List all material prims in the current stage. Looks inside instanceable "
            "prims (materials there carry instance_proxy=true) unless "
            "include_instance_proxies=false; include_inactive=true also lists deactivated "
            "materials."
        ),
        read_only=True,
        idempotent=True,
    )
    async def list_isaac_materials(
        self,
        max_results: int = 200,
        offset: int = 0,
        include_instance_proxies: bool = True,
        include_inactive: bool = False,
    ) -> Dict[str, Any]:
        """
        List all materials in the current stage.

        Args:
            max_results: Maximum number of materials to return per page.
                Clamped to [1, 1000]; the effective cap is reported as
                applied_limit.
            offset: Number of materials to skip before the page starts; pass
                the previous page's next_offset to continue.
            include_instance_proxies: Look inside instanceable prims, as the
                other listings do; materials there carry
                ``instance_proxy: true``.
            include_inactive: Also list deactivated materials
                (``active: false``).

        Returns:
            Dict with a page of material paths and basic info plus the total
            count.
        """
        max_results = max(1, min(max_results, 1000))
        offset = max(0, offset)
        _predicate = _listing_predicate(include_instance_proxies, include_inactive)
        script = _compose_script(
            """\
            import json
            import omni.usd
            from pxr import Usd, UsdShade
            """,
            TRAVERSAL_PREDICATE_CORE,
            f"""\
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                offset = {offset}
                limit = {max_results}
                materials = []
                total = 0
                for p in Usd.PrimRange(stage.GetPseudoRoot(), {_predicate}):
                    if not p.IsA(UsdShade.Material):
                        continue
                    index = total
                    total += 1
                    # Resolving the surface shader is the expensive part of
                    # each entry, so it only runs for entries on the page.
                    if index < offset or len(materials) >= limit:
                        continue
                    mat = UsdShade.Material(p)
                    shader_result = mat.ComputeSurfaceSource()
                    shader_obj = shader_result[0] if shader_result else None
                    if not shader_obj:
                        shader_result = mat.ComputeSurfaceSource("mdl")
                        shader_obj = shader_result[0] if shader_result else None
                    shader_type = None
                    if shader_obj:
                        sp = shader_obj.GetPrim()
                        sub_id = sp.GetAttribute("info:mdl:sourceAsset:subIdentifier")
                        if sub_id and sub_id.Get():
                            shader_type = str(sub_id.Get())
                        else:
                            sid = shader_obj.GetIdAttr().Get()
                            shader_type = str(sid) if sid else None
                    entry = {{
                        "path": str(p.GetPath()),
                        "name": p.GetName(),
                        "shader_type": shader_type,
                    }}
                    if p.IsInstanceProxy():
                        entry["instance_proxy"] = True
                    if not p.IsActive():
                        entry["active"] = False
                    materials.append(entry)
                page = materials
                truncated = offset + len(page) < total
                print(json.dumps({{
                    "count": len(page),
                    "total": total,
                    "offset": offset,
                    "applied_limit": limit,
                    "truncated": truncated,
                    "next_offset": offset + len(page) if truncated else None,
                    "include_instance_proxies": {_pyval(bool(include_instance_proxies))},
                    "include_inactive": {_pyval(bool(include_inactive))},
                    "materials": page,
                }}))
            """,
        )
        return await self._execute_json_script(script)

    @tool_meta(
        name="assign_isaac_material",
        description=(
            "Bind a material to a prim so the prim renders with that material's appearance."
        ),
        read_only=False,
        destructive=True,
        idempotent=True,
    )
    async def assign_isaac_material(
        self, prim_path: str, material_path: str
    ) -> Dict[str, Any]:
        """
        Assign a material to a prim.

        Args:
            prim_path: USD path of the target prim.
            material_path: USD path of the Material to assign.

        Returns:
            Dict confirming the material binding.
        """
        _prim_path = _pyval(prim_path)
        _mat_path = _pyval(material_path)
        script = _compose_script(
            """\
            import json
            import omni.usd
            from pxr import UsdShade
            """,
            BIND_MATERIAL_CORE,
            f"""\
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                print(json.dumps(_bind_material(stage, {_prim_path}, {_mat_path})))
            """,
        )
        return await self._execute_json_script(script)

    @tool_meta(
        name="set_isaac_material_property",
        description=(
            "Set an input on a material's surface shader, typed the way the shader declares "
            "it: an existing input keeps its type; a new one takes the OmniPBR or "
            "UsdPreviewSurface type (texture paths become assets, flags bools, enums ints). "
            "A name that is not an input of a known shader is refused with suggestions "
            "(OmniPBR roughness is reflection_roughness_constant, color "
            "diffuse_color_constant, texture diffuse_texture). A file path for an asset input "
            "must be inside the read sandbox (security.allowed_paths). Returns the input's type and "
            "the value read back."
        ),
        read_only=False,
        destructive=True,
        idempotent=True,
    )
    async def set_isaac_material_property(
        self,
        material_path: str,
        property_name: str,
        value: Any,
        allow_unknown_input: bool = False,
    ) -> Dict[str, Any]:
        """
        Set an input on a material's surface shader.

        Args:
            material_path: USD path of the Material prim.
            property_name: Shader input name, with or without the "inputs:"
                prefix: "reflection_roughness_constant" or "diffuse_texture"
                on OmniPBR, "roughness" or "diffuseColor" on
                UsdPreviewSurface.
            value: The value: a number, a bool, a string (a file path for a
                texture input), or a list of 2-4 numbers for a vector or
                color. It must fit the input's type.
            allow_unknown_input: Create an input that is not one of the known
                OmniPBR / UsdPreviewSurface inputs, with a type inferred from
                the value. Off by default, because a misspelt name creates an
                input the shader never reads.

        Returns:
            Dict with the shader type, ``input_type``, ``created`` (whether
            the input was new) and ``value_set`` as read back, or an error
            naming the expected type or the closest known input names.
        """
        if not property_name or not property_name.strip():
            return ErrorResponse(error="property_name is empty", error_type="ValidationError").model_dump()
        # Whether a string is a file path depends on the input's type, which
        # only Isaac knows, so the sandbox's verdict goes along and the script
        # applies it to an asset input.
        asset_checks = self._asset_value_checks([value])
        script = _compose_script(
            """\
            import json
            import omni.usd
            from pxr import Gf, Sdf, UsdShade
            """,
            f"OMNIPBR_INPUTS = {_pyval(OMNIPBR_INPUTS)}\n"
            f"USD_PREVIEW_SURFACE_INPUTS = {_pyval(USD_PREVIEW_SURFACE_INPUTS)}\n",
            ASSET_SANDBOX_CORE,
            SET_MATERIAL_INPUT_CORE,
            f"""\
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                value = json.loads({_pyval(json.dumps(value))})
                print(json.dumps(_set_material_input(
                    stage, {_pyval(material_path)}, {_pyval(property_name.strip())}, value,
                    {_pyval(bool(allow_unknown_input))}, {_pyval(asset_checks)}
                )))
            """,
        )
        return await self._execute_json_script(script)

    @tool_meta(
        name="create_isaac_material",
        description=(
            "Create a new material with a surface shader: shader_type UsdPreviewSurface "
            "(portable) or OmniPBR (NVIDIA MDL), in any case; any other type is refused. Set "
            "diffuse color, roughness, metallic, and opacity. After creation, use "
            "assign_isaac_material to bind it to prims."
        ),
        read_only=False,
    )
    async def create_isaac_material(
        self,
        material_path: str,
        shader_type: str = "UsdPreviewSurface",
        diffuse_color: Optional[FloatList] = None,
        roughness: float = 0.5,
        metallic: float = 0.0,
        opacity: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Create a new material with a surface shader and common properties.

        Args:
            material_path: USD path for the new Material prim (e.g. "/World/Looks/Red").
            shader_type: Shader type — "UsdPreviewSurface" or "OmniPBR".
            diffuse_color: RGB color as [r, g, b] floats 0-1. Defaults to [0.8, 0.8, 0.8].
            roughness: Surface roughness 0-1.
            metallic: Metallic factor 0-1.
            opacity: Opacity 0-1.

        Returns:
            Dict confirming the material and shader were created, or a
            ValidationError for a shader type other than the two supported.
        """
        canonical = MATERIAL_SHADER_TYPES.get((shader_type or "").strip().lower())
        if canonical is None:
            return ErrorResponse(
                error=(
                    f"Unsupported shader_type {shader_type!r}; use one of "
                    + ", ".join(sorted(set(MATERIAL_SHADER_TYPES.values())))
                    + " (any case). Other MDL shaders need execute_isaac_script."
                ),
                error_type="ValidationError",
                details={"code": "UNSUPPORTED_SHADER_TYPE"},
            ).model_dump()
        _mat_path = _pyval(material_path)
        _shader_type = _pyval(canonical)
        _color = _pyval([float(c) for c in (diffuse_color or [0.8, 0.8, 0.8])])
        script = _compose_script(
            """\
            import json
            import omni.usd
            from pxr import UsdShade, Sdf, Gf
            """,
            DEFINE_MATERIAL_CORE,
            f"""\
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                print(json.dumps(_define_material(
                    stage, {_mat_path}, {_shader_type}, {_color},
                    {float(roughness)}, {float(metallic)}, {float(opacity)}
                )))
            """,
        )
        return await self._execute_json_script(script)
