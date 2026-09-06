"""Materials & Appearance tools for Isaac Sim."""

import json
import textwrap
from typing import Any, Callable, Dict, List, Optional

from ....adapters import IsaacSocketClient, ScriptResult
from ...schemas.common import ErrorResponse
from ._shared import (
    BIND_MATERIAL_CORE,
    BULK_GEOMETRY_ATTRIBUTES,
    DEFAULT_MAX_RESULTS,
    DEFINE_MATERIAL_CORE,
    LOG_SCAN_WINDOW_BYTES,
    MAX_CAPTURE_DIMENSION,
    MAX_INLINE_CAPTURE_BYTES,
    MAX_RETAINED_CAPTURES,
    MAX_SCRIPT_BYTES,
    PRIM_DETAIL_ASPECTS,
    FloatList,
    _compose_script,
    _pyval,
    logger,
)


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

    async def list_isaac_materials(
        self, max_results: int = 200, offset: int = 0
    ) -> Dict[str, Any]:
        """
        List all materials in the current stage.

        Args:
            max_results: Maximum number of materials to return per page.
                Clamped to [1, 1000]; the effective cap is reported as
                applied_limit.
            offset: Number of materials to skip before the page starts; pass
                the previous page's next_offset to continue.

        Returns:
            Dict with a page of material paths and basic info plus the total
            count.
        """
        max_results = max(1, min(max_results, 1000))
        offset = max(0, offset)
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import UsdShade

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                offset = {offset}
                limit = {max_results}
                materials = []
                total = 0
                for p in stage.Traverse():
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
                    materials.append({{
                        "path": str(p.GetPath()),
                        "name": p.GetName(),
                        "shader_type": shader_type,
                    }})
                page = materials
                truncated = offset + len(page) < total
                print(json.dumps({{
                    "count": len(page),
                    "total": total,
                    "offset": offset,
                    "applied_limit": limit,
                    "truncated": truncated,
                    "next_offset": offset + len(page) if truncated else None,
                    "materials": page,
                }}))
        """)
        return await self._execute_json_script(script)

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

    async def set_isaac_material_property(
        self,
        material_path: str,
        property_name: str,
        value: Any,
    ) -> Dict[str, Any]:
        """
        Set an input property on a material's surface shader.

        Args:
            material_path: USD path of the Material prim.
            property_name: Name of the shader input (e.g. "diffuseColor").
            value: Value to set. Lists become Gf.Vec3f for 3-element arrays.

        Returns:
            Dict confirming the property was set.
        """
        _mat_path = _pyval(material_path)
        _prop_name = _pyval(property_name)
        _val_str = _pyval(json.dumps(value))
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import UsdShade, Sdf, Gf

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                prim = stage.GetPrimAtPath({_mat_path})
                if not prim.IsValid() or not prim.IsA(UsdShade.Material):
                    print(json.dumps({{"error": "Material not found: " + {_mat_path}}}))
                else:
                    mat = UsdShade.Material(prim)
                    # Try default render context, then MDL
                    shader_result = mat.ComputeSurfaceSource()
                    shader_obj = shader_result[0] if shader_result else None
                    if not shader_obj:
                        shader_result = mat.ComputeSurfaceSource("mdl")
                        shader_obj = shader_result[0] if shader_result else None
                    if not shader_obj:
                        print(json.dumps({{"error": "No surface shader found on material"}}))
                    else:
                        val = json.loads({_val_str})
                        prop_name = {_prop_name}
                        inp = shader_obj.GetInput(prop_name)
                        if not inp or not inp.GetAttr().IsValid():
                            # Create the input for MDL shaders
                            if isinstance(val, list) and len(val) == 3:
                                inp = shader_obj.CreateInput(prop_name, Sdf.ValueTypeNames.Color3f)
                            elif isinstance(val, list) and len(val) == 4:
                                inp = shader_obj.CreateInput(prop_name, Sdf.ValueTypeNames.Float4)
                            elif isinstance(val, (int, float)):
                                inp = shader_obj.CreateInput(prop_name, Sdf.ValueTypeNames.Float)
                            elif isinstance(val, bool):
                                inp = shader_obj.CreateInput(prop_name, Sdf.ValueTypeNames.Bool)
                            elif isinstance(val, str):
                                inp = shader_obj.CreateInput(prop_name, Sdf.ValueTypeNames.String)
                            else:
                                inp = shader_obj.CreateInput(prop_name, Sdf.ValueTypeNames.Float)
                        if isinstance(val, list) and len(val) == 3:
                            val = Gf.Vec3f(*val)
                        elif isinstance(val, list) and len(val) == 4:
                            val = Gf.Vec4f(*val)
                        inp.Set(val)
                        print(json.dumps({{
                            "material_path": {_mat_path},
                            "property_name": prop_name,
                            "value_set": str(val),
                        }}))
        """)
        return await self._execute_json_script(script)

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
            Dict confirming the material and shader were created.
        """
        _mat_path = _pyval(material_path)
        _shader_type = _pyval(shader_type)
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

