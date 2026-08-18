"""Materials & Appearance tools for Isaac Sim."""

import json
import textwrap
from typing import Any, Callable, Dict, List, Optional

from ....adapters import IsaacSocketClient, ScriptResult
from ...schemas.common import ErrorResponse
from ._shared import (
    BULK_GEOMETRY_ATTRIBUTES,
    LOG_SCAN_WINDOW_BYTES,
    MAX_CAPTURE_DIMENSION,
    MAX_INLINE_CAPTURE_BYTES,
    MAX_RETAINED_CAPTURES,
    MAX_SCRIPT_BYTES,
    PRIM_DETAIL_ASPECTS,
    FloatList,
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

    async def list_isaac_materials(self) -> Dict[str, Any]:
        """
        List all materials in the current stage.

        Returns:
            Dict with list of material paths and basic info.
        """
        script = textwrap.dedent("""\
            import json
            import omni.usd
            from pxr import UsdShade

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({"error": "No stage is currently open"}))
            else:
                materials = []
                for p in stage.Traverse():
                    if p.IsA(UsdShade.Material):
                        mat = UsdShade.Material(p)
                        shader_result = mat.ComputeSurfaceSource()
                        shader_obj = shader_result[0] if shader_result else None
                        render_ctx = ""
                        if not shader_obj:
                            shader_result = mat.ComputeSurfaceSource("mdl")
                            shader_obj = shader_result[0] if shader_result else None
                            render_ctx = "mdl"
                        shader_type = None
                        if shader_obj:
                            sp = shader_obj.GetPrim()
                            sub_id = sp.GetAttribute("info:mdl:sourceAsset:subIdentifier")
                            if sub_id and sub_id.Get():
                                shader_type = str(sub_id.Get())
                            else:
                                sid = shader_obj.GetIdAttr().Get()
                                shader_type = str(sid) if sid else None
                        materials.append({
                            "path": str(p.GetPath()),
                            "name": p.GetName(),
                            "shader_type": shader_type,
                        })
                print(json.dumps({
                    "count": len(materials),
                    "materials": materials,
                }))
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
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import UsdShade

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                prim = stage.GetPrimAtPath({_prim_path})
                mat_prim = stage.GetPrimAtPath({_mat_path})
                if not prim.IsValid():
                    print(json.dumps({{"error": "Prim not found: " + {_prim_path}}}))
                elif not mat_prim.IsValid() or not mat_prim.IsA(UsdShade.Material):
                    print(json.dumps({{"error": "Material not found: " + {_mat_path}}}))
                else:
                    mat = UsdShade.Material(mat_prim)
                    UsdShade.MaterialBindingAPI(prim).Bind(mat)
                    print(json.dumps({{
                        "prim_path": {_prim_path},
                        "material_path": {_mat_path},
                        "bound": True,
                    }}))
        """)
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
        color = diffuse_color or [0.8, 0.8, 0.8]
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import UsdShade, Sdf, Gf

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                existing = stage.GetPrimAtPath({_mat_path})
                if existing.IsValid():
                    print(json.dumps({{"error": "Prim already exists at " + {_mat_path}}}))
                else:
                    mat_prim = stage.DefinePrim({_mat_path}, "Material")
                    mat = UsdShade.Material(mat_prim)
                    shader_type = {_shader_type}
                    shader_path = {_mat_path} + "/Shader"

                    if shader_type == "OmniPBR":
                        shader_prim = stage.DefinePrim(shader_path, "Shader")
                        shader = UsdShade.Shader(shader_prim)
                        shader.CreateIdAttr("OmniPBR")
                        shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
                        shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
                        shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
                        shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(
                            Gf.Vec3f({color[0]}, {color[1]}, {color[2]})
                        )
                        shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set({roughness})
                        shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set({metallic})
                        shader.CreateInput("enable_opacity", Sdf.ValueTypeNames.Bool).Set({opacity} < 1.0)
                        if {opacity} < 1.0:
                            shader.CreateInput("opacity_constant", Sdf.ValueTypeNames.Float).Set({opacity})
                        out = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
                        mat.CreateSurfaceOutput("mdl").ConnectToSource(out)
                    else:
                        shader_prim = stage.DefinePrim(shader_path, "Shader")
                        shader = UsdShade.Shader(shader_prim)
                        shader.CreateIdAttr("UsdPreviewSurface")
                        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                            Gf.Vec3f({color[0]}, {color[1]}, {color[2]})
                        )
                        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set({roughness})
                        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set({metallic})
                        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set({opacity})
                        out = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
                        mat.CreateSurfaceOutput().ConnectToSource(out)

                    print(json.dumps({{
                        "material_path": {_mat_path},
                        "shader_path": shader_path,
                        "shader_type": shader_type,
                        "diffuse_color": [{color[0]}, {color[1]}, {color[2]}],
                        "roughness": {roughness},
                        "metallic": {metallic},
                        "opacity": {opacity},
                        "created": True,
                    }}))
        """)
        return await self._execute_json_script(script)

