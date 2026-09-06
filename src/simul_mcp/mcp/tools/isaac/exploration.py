"""Scene Inspection & Exploration tools for Isaac Sim."""

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
from .scene import COUNTED_PRIMS_HELPER


class ExplorationMixin:
    # ------------------------------------------------------------------
    # Phase 9: Scene Inspection & Exploration
    # ------------------------------------------------------------------

    async def get_isaac_bounding_box(self, prim_path: str) -> Dict[str, Any]:
        """
        Get the world-space axis-aligned bounding box of a prim.

        Args:
            prim_path: USD path of the prim.

        Returns:
            Dict with min, max, size, and center of the bounding box.
        """
        _prim_path = _pyval(prim_path)
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import Usd, UsdGeom, Gf

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                prim = stage.GetPrimAtPath({_prim_path})
                if not prim.IsValid():
                    print(json.dumps({{"error": "Prim not found: " + {_prim_path}}}))
                else:
                    bbox_cache = UsdGeom.BBoxCache(
                        Usd.TimeCode.Default(), ["default", "render"]
                    )
                    bbox = bbox_cache.ComputeWorldBound(prim)
                    rng = bbox.ComputeAlignedRange()
                    if rng.IsEmpty():
                        print(json.dumps({{
                            "prim_path": {_prim_path},
                            "empty": True,
                            "error": "Bounding box is empty (prim may have no geometry)",
                        }}))
                    else:
                        mn = rng.GetMin()
                        mx = rng.GetMax()
                        sz = mx - mn
                        ct = (mn + mx) * 0.5
                        print(json.dumps({{
                            "prim_path": {_prim_path},
                            "min": [mn[0], mn[1], mn[2]],
                            "max": [mx[0], mx[1], mx[2]],
                            "size": [sz[0], sz[1], sz[2]],
                            "center": [ct[0], ct[1], ct[2]],
                        }}))
        """)
        return await self._execute_json_script(script)

    async def get_isaac_mesh_info(self, prim_path: str) -> Dict[str, Any]:
        """
        Get mesh geometry details: vertex count, face count, normals, UVs.

        Args:
            prim_path: USD path of a Mesh prim.

        Returns:
            Dict with vertex_count, face_count, has_normals, has_uvs, etc.
        """
        _prim_path = _pyval(prim_path)
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import UsdGeom

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                prim = stage.GetPrimAtPath({_prim_path})
                if not prim.IsValid():
                    print(json.dumps({{"error": "Prim not found: " + {_prim_path}}}))
                elif not prim.IsA(UsdGeom.Mesh):
                    print(json.dumps({{"error": "Prim is not a Mesh: " + {_prim_path}}}))
                else:
                    mesh = UsdGeom.Mesh(prim)
                    points = mesh.GetPointsAttr().Get()
                    face_counts = mesh.GetFaceVertexCountsAttr().Get()
                    face_indices = mesh.GetFaceVertexIndicesAttr().Get()
                    normals = mesh.GetNormalsAttr().Get()
                    subdiv = mesh.GetSubdivisionSchemeAttr().Get()
                    has_uvs = False
                    pv_api = UsdGeom.PrimvarsAPI(prim)
                    for pv in pv_api.GetPrimvars():
                        name = pv.GetPrimvarName()
                        if name in ("st", "UVMap", "st0", "st1"):
                            has_uvs = True
                            break
                    print(json.dumps({{
                        "prim_path": {_prim_path},
                        "vertex_count": len(points) if points else 0,
                        "face_count": len(face_counts) if face_counts else 0,
                        "face_vertex_count": len(face_indices) if face_indices else 0,
                        "has_normals": normals is not None and len(normals) > 0,
                        "has_uvs": has_uvs,
                        "subdivision_scheme": str(subdiv) if subdiv else "none",
                    }}))
        """)
        return await self._execute_json_script(script)

    async def list_isaac_lights(
        self, root_path: str = "/", max_results: int = 200, offset: int = 0
    ) -> Dict[str, Any]:
        """
        List all light prims in the scene under a root path.

        Args:
            root_path: USD prim path to search under.
            max_results: Maximum number of lights to return per page. Clamped
                to [1, 1000]; the effective cap is reported as applied_limit.
            offset: Number of lights to skip before the page starts; pass the
                previous page's next_offset to continue.

        Returns:
            Dict with a page of lights (type, intensity, color) and the total
            count.
        """
        _root_path = _pyval(root_path)
        max_results = max(1, min(max_results, 1000))
        offset = max(0, offset)
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import UsdLux, Usd

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                root = stage.GetPrimAtPath({_root_path})
                if not root.IsValid():
                    print(json.dumps({{"error": "Root path not found: " + {_root_path}}}))
                else:
                    lights = []
                    for p in Usd.PrimRange(root):
                        if not p.HasAPI(UsdLux.LightAPI):
                            continue
                        info = {{
                            "path": str(p.GetPath()),
                            "name": p.GetName(),
                            "type": p.GetTypeName(),
                        }}
                        intensity = p.GetAttribute("inputs:intensity")
                        if intensity and intensity.Get() is not None:
                            info["intensity"] = float(intensity.Get())
                        color = p.GetAttribute("inputs:color")
                        if color and color.Get() is not None:
                            c = color.Get()
                            info["color"] = [float(c[0]), float(c[1]), float(c[2])]
                        enabled = p.GetAttribute("inputs:enableColorTemperature")
                        if enabled and enabled.Get():
                            temp = p.GetAttribute("inputs:colorTemperature")
                            if temp and temp.Get() is not None:
                                info["color_temperature"] = float(temp.Get())
                        lights.append(info)
                    offset = {offset}
                    limit = {max_results}
                    page = lights[offset:offset + limit]
                    truncated = offset + len(page) < len(lights)
                    print(json.dumps({{
                        "root_path": {_root_path},
                        "count": len(page),
                        "total": len(lights),
                        "offset": offset,
                        "applied_limit": limit,
                        "truncated": truncated,
                        "next_offset": offset + len(page) if truncated else None,
                        "lights": page,
                    }}))
        """)
        return await self._execute_json_script(script)

    async def get_isaac_light_info(self, prim_path: str) -> Dict[str, Any]:
        """
        Get detailed properties of a light prim.

        Args:
            prim_path: USD path of the light prim.

        Returns:
            Dict with light type, intensity, color, shadow settings, etc.
        """
        _prim_path = _pyval(prim_path)
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import UsdLux

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                prim = stage.GetPrimAtPath({_prim_path})
                if not prim.IsValid():
                    print(json.dumps({{"error": "Prim not found: " + {_prim_path}}}))
                elif not prim.HasAPI(UsdLux.LightAPI):
                    print(json.dumps({{"error": "Prim is not a light: " + {_prim_path}}}))
                else:
                    info = {{
                        "prim_path": {_prim_path},
                        "type": prim.GetTypeName(),
                    }}
                    attr_names = [
                        "inputs:intensity", "inputs:exposure", "inputs:color",
                        "inputs:enableColorTemperature", "inputs:colorTemperature",
                        "inputs:diffuse", "inputs:specular",
                        "inputs:radius", "inputs:width", "inputs:height",
                        "inputs:length", "inputs:angle", "inputs:softness",
                        "inputs:shaping:cone:angle", "inputs:shaping:cone:softness",
                        "inputs:shaping:focus",
                    ]
                    for attr_name in attr_names:
                        attr = prim.GetAttribute(attr_name)
                        if attr and attr.Get() is not None:
                            val = attr.Get()
                            key = attr_name.replace("inputs:", "")
                            try:
                                if hasattr(val, '__len__') and not isinstance(val, str):
                                    info[key] = [float(x) for x in val]
                                else:
                                    info[key] = float(val) if isinstance(val, (int, float, bool)) else str(val)
                            except (TypeError, ValueError):
                                info[key] = str(val)
                    # Shadow
                    shadow_enable = prim.GetAttribute("inputs:shadow:enable")
                    if shadow_enable and shadow_enable.Get() is not None:
                        info["shadow_enabled"] = bool(shadow_enable.Get())
                    print(json.dumps(info))
        """)
        return await self._execute_json_script(script)

    async def create_isaac_light(
        self,
        prim_path: str,
        light_type: str = "DomeLight",
        intensity: float = 1000.0,
        color: Optional[FloatList] = None,
        color_temperature: Optional[float] = None,
        angle: Optional[float] = None,
        texture_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a light prim in the scene.

        Args:
            prim_path: USD path for the new light (e.g. "/World/Lights/DomeLight").
            light_type: One of "DomeLight", "DistantLight", "SphereLight",
                        "RectLight", "DiskLight", "CylinderLight".
            intensity: Light intensity written to UsdLux inputs:intensity, a
                unitless radiance multiplier. RTX-scale starting points:
                DomeLight ~1000, DistantLight ~3000, SphereLight/RectLight
                ~30000 or more.
            color: RGB color as [r, g, b] floats 0-1. Defaults to [1, 1, 1].
            color_temperature: Optional color temperature in Kelvin (enables
                temperature mode).
            angle: Angular diameter for DistantLight in degrees. Ignored for
                other types.
            texture_file: Texture file path for DomeLight (e.g. HDRI map).

        Returns:
            Dict confirming the light was created with its properties.
        """
        _prim_path = _pyval(prim_path)
        _light_type = _pyval(light_type)
        rgb = color or [1.0, 1.0, 1.0]
        _tex = _pyval(texture_file or None)
        valid_types = [
            "DomeLight", "DistantLight", "SphereLight",
            "RectLight", "DiskLight", "CylinderLight",
        ]
        if light_type not in valid_types:
            return ErrorResponse(
                error=f"Invalid light_type '{light_type}'. Must be one of: {valid_types}",
                error_type="ValueError",
            ).model_dump()
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import UsdLux, Gf, Sdf

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                existing = stage.GetPrimAtPath({_prim_path})
                if existing.IsValid():
                    print(json.dumps({{"error": "Prim already exists at " + {_prim_path}}}))
                else:
                    light_prim = stage.DefinePrim({_prim_path}, {_light_type})
                    if not light_prim.IsValid():
                        print(json.dumps({{"error": "Failed to create light prim"}}))
                    else:
                        light_prim.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set({intensity})
                        light_prim.CreateAttribute("inputs:color", Sdf.ValueTypeNames.Color3f).Set(
                            Gf.Vec3f({rgb[0]}, {rgb[1]}, {rgb[2]})
                        )
                        result = {{
                            "prim_path": {_prim_path},
                            "light_type": {_light_type},
                            "intensity": {intensity},
                            "color": [{rgb[0]}, {rgb[1]}, {rgb[2]}],
                            "created": True,
                        }}
                        temp = {color_temperature if color_temperature is not None else "None"}
                        if temp is not None:
                            light_prim.CreateAttribute(
                                "inputs:enableColorTemperature", Sdf.ValueTypeNames.Bool
                            ).Set(True)
                            light_prim.CreateAttribute(
                                "inputs:colorTemperature", Sdf.ValueTypeNames.Float
                            ).Set(float(temp))
                            result["color_temperature"] = float(temp)
                        angle_val = {angle if angle is not None else "None"}
                        if angle_val is not None:
                            angle_attr = light_prim.GetAttribute("inputs:angle")
                            if not angle_attr or not angle_attr.IsValid():
                                angle_attr = light_prim.CreateAttribute(
                                    "inputs:angle", Sdf.ValueTypeNames.Float
                                )
                            angle_attr.Set(float(angle_val))
                            result["angle"] = float(angle_val)
                        tex = {_tex}
                        if tex is not None:
                            tex_attr = light_prim.GetAttribute("inputs:texture:file")
                            if not tex_attr or not tex_attr.IsValid():
                                tex_attr = light_prim.CreateAttribute(
                                    "inputs:texture:file", Sdf.ValueTypeNames.Asset
                                )
                            tex_attr.Set(Sdf.AssetPath(tex))
                            result["texture_file"] = tex
                        print(json.dumps(result))
        """)
        return await self._execute_json_script(script)

    async def get_isaac_prim_ancestors(self, prim_path: str) -> Dict[str, Any]:
        """
        Get the ancestor chain from root to the specified prim.

        Args:
            prim_path: USD path of the prim.

        Returns:
            Dict with the ordered list of ancestors from root to prim.
        """
        _prim_path = _pyval(prim_path)
        script = textwrap.dedent(f"""\
            import json
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                prim = stage.GetPrimAtPath({_prim_path})
                if not prim.IsValid():
                    print(json.dumps({{"error": "Prim not found: " + {_prim_path}}}))
                else:
                    ancestors = []
                    current = prim
                    while current and current.GetPath() != current.GetParent().GetPath():
                        ancestors.insert(0, {{
                            "path": str(current.GetPath()),
                            "name": current.GetName(),
                            "type": current.GetTypeName(),
                        }})
                        current = current.GetParent()
                    print(json.dumps({{
                        "prim_path": {_prim_path},
                        "depth": len(ancestors),
                        "ancestors": ancestors,
                    }}))
        """)
        return await self._execute_json_script(script)

    async def get_isaac_subtree(
        self,
        root_path: str = "/",
        max_depth: int = 5,
        max_results: int = 150,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        Get a full subtree as a flat list with depth info.

        Args:
            root_path: USD prim path of the subtree root.
            max_depth: Maximum depth to traverse below root_path.
            max_results: Maximum number of prims to return per page. Clamped
                to [1, 1000]; the effective cap is reported as applied_limit.
            offset: Number of prims (in traversal order) to skip before the
                page starts; pass the previous page's next_offset to continue.

        Returns:
            Dict with a page of prims showing path, type, name, depth, and
            child count.
        """
        _root_path = _pyval(root_path)
        max_results = max(1, min(max_results, 1000))
        offset = max(0, offset)
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import Usd

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                root = stage.GetPrimAtPath({_root_path})
                if not root.IsValid():
                    print(json.dumps({{"error": "Root path not found: " + {_root_path}}}))
                else:
                    root_depth = len({_root_path}.rstrip("/").split("/")) - 1
                    prims = []
                    truncated = False
                    matched = 0
                    offset = {offset}
                    limit = {max_results}
                    prim_iter = iter(Usd.PrimRange(root))
                    for p in prim_iter:
                        path_str = str(p.GetPath())
                        depth = len(path_str.rstrip("/").split("/")) - 1 - root_depth
                        if depth >= {max_depth}:
                            prim_iter.PruneChildren()
                        if depth > {max_depth}:
                            continue
                        matched += 1
                        if matched <= offset:
                            continue
                        if len(prims) >= limit:
                            truncated = True
                            break
                        prims.append({{
                            "path": path_str,
                            "name": p.GetName(),
                            "type": p.GetTypeName(),
                            "depth": depth,
                            "child_count": len(p.GetChildren()),
                        }})
                    print(json.dumps({{
                        "root_path": {_root_path},
                        "count": len(prims),
                        "offset": offset,
                        "applied_limit": limit,
                        "truncated": truncated,
                        "next_offset": offset + len(prims) if truncated else None,
                        "max_depth": {max_depth},
                        "prims": prims,
                    }}))
        """)
        return await self._execute_json_script(script)

    async def get_isaac_prim_relationships(self, prim_path: str) -> Dict[str, Any]:
        """
        Get relationships, material bindings, references, and payloads.

        Args:
            prim_path: USD path of the prim.

        Returns:
            Dict with material_binding, references, payloads, variant_sets.
        """
        _prim_path = _pyval(prim_path)
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import UsdShade, Sdf

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                prim = stage.GetPrimAtPath({_prim_path})
                if not prim.IsValid():
                    print(json.dumps({{"error": "Prim not found: " + {_prim_path}}}))
                else:
                    result = {{"prim_path": {_prim_path}}}
                    # Material binding
                    binding_api = UsdShade.MaterialBindingAPI(prim)
                    mat, _ = binding_api.ComputeBoundMaterial()
                    if mat:
                        result["material_binding"] = str(mat.GetPath())
                    else:
                        result["material_binding"] = None
                    # References
                    refs_meta = prim.GetMetadata("references")
                    ref_list = []
                    if refs_meta:
                        for ref in refs_meta.GetAddedOrExplicitItems():
                            ref_list.append({{
                                "asset_path": str(ref.assetPath) if ref.assetPath else None,
                                "prim_path": str(ref.primPath) if ref.primPath else None,
                            }})
                    result["references"] = ref_list
                    # Payloads
                    pay_meta = prim.GetMetadata("payload")
                    pay_list = []
                    if pay_meta:
                        items = pay_meta.GetAddedOrExplicitItems() if hasattr(pay_meta, "GetAddedOrExplicitItems") else []
                        for pay in items:
                            pay_list.append({{
                                "asset_path": str(pay.assetPath) if pay.assetPath else None,
                                "prim_path": str(pay.primPath) if pay.primPath else None,
                            }})
                    result["payloads"] = pay_list
                    # Variant sets
                    vsets = prim.GetVariantSets()
                    variant_info = {{}}
                    for name in vsets.GetNames():
                        vs = vsets.GetVariantSet(name)
                        variant_info[name] = {{
                            "variants": vs.GetVariantNames(),
                            "selection": vs.GetVariantSelection(),
                        }}
                    result["variant_sets"] = variant_info
                    # Relationships
                    rels = []
                    for rel in prim.GetRelationships():
                        targets = rel.GetTargets()
                        if targets:
                            rels.append({{
                                "name": rel.GetName(),
                                "targets": [str(t) for t in targets],
                            }})
                    result["relationships"] = rels
                    print(json.dumps(result))
        """)
        return await self._execute_json_script(script)

    async def get_isaac_layer_info(self) -> Dict[str, Any]:
        """
        Get USD layer stack information for the current stage.

        Returns:
            Dict with root layer, sublayers, and session layer info.
        """
        script = textwrap.dedent("""\
            import json
            import omni.usd
            from pxr import Sdf

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({"error": "No stage is currently open"}))
            else:
                root_layer = stage.GetRootLayer()
                session_layer = stage.GetSessionLayer()
                sublayers = []
                for path in root_layer.subLayerPaths:
                    resolved = root_layer.ComputeAbsolutePath(path)
                    layer = Sdf.Layer.FindOrOpen(resolved)
                    sublayers.append({
                        "path": path,
                        "resolved_path": resolved,
                        "exists": layer is not None,
                    })
                layer_stack = []
                for layer in stage.GetLayerStack():
                    layer_stack.append({
                        "identifier": layer.identifier,
                        "display_name": layer.GetDisplayName(),
                        "dirty": layer.dirty,
                    })
                print(json.dumps({
                    "root_layer": root_layer.identifier,
                    "root_layer_path": root_layer.realPath,
                    "session_layer": session_layer.identifier if session_layer else None,
                    "sublayer_count": len(sublayers),
                    "sublayers": sublayers,
                    "layer_stack_count": len(layer_stack),
                    "layer_stack": layer_stack,
                }))
        """)
        return await self._execute_json_script(script)

    async def get_isaac_scene_stats(self, root_path: str = "/") -> Dict[str, Any]:
        """
        Get aggregate scene statistics: vertex/face counts, material usage.

        Args:
            root_path: USD prim path to compute stats under.

        Returns:
            Dict with total_prims, total_vertices, total_faces, etc. Prims are
            counted the same way as get_isaac_scene_summary.
        """
        _root_path = _pyval(root_path)
        script = COUNTED_PRIMS_HELPER + textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import Usd, UsdGeom, UsdShade, UsdLux

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                root = stage.GetPrimAtPath({_root_path})
                if not root.IsValid():
                    print(json.dumps({{"error": "Root path not found: " + {_root_path}}}))
                else:
                    total_prims = 0
                    total_meshes = 0
                    total_vertices = 0
                    total_faces = 0
                    total_lights = 0
                    total_cameras = 0
                    total_materials = 0
                    total_xforms = 0
                    type_counts = {{}}
                    for p in _counted_prims(stage, {_root_path}):
                        total_prims += 1
                        tname = _prim_type_label(p)
                        type_counts[tname] = type_counts.get(tname, 0) + 1
                        if p.IsA(UsdGeom.Mesh):
                            total_meshes += 1
                            mesh = UsdGeom.Mesh(p)
                            pts = mesh.GetPointsAttr().Get()
                            fcs = mesh.GetFaceVertexCountsAttr().Get()
                            if pts:
                                total_vertices += len(pts)
                            if fcs:
                                total_faces += len(fcs)
                        elif p.IsA(UsdGeom.Camera):
                            total_cameras += 1
                        elif p.IsA(UsdGeom.Xform):
                            total_xforms += 1
                        if p.HasAPI(UsdLux.LightAPI):
                            total_lights += 1
                        if p.IsA(UsdShade.Material):
                            total_materials += 1
                    print(json.dumps({{
                        "root_path": {_root_path},
                        "total_prims": total_prims,
                        "total_meshes": total_meshes,
                        "total_vertices": total_vertices,
                        "total_faces": total_faces,
                        "total_lights": total_lights,
                        "total_cameras": total_cameras,
                        "total_materials": total_materials,
                        "total_xforms": total_xforms,
                        "type_counts": dict(sorted(type_counts.items(), key=lambda x: -x[1])),
                    }}))
        """)
        return await self._execute_json_script(script)

    async def get_isaac_texture_dependencies(
        self, root_path: str = "/"
    ) -> Dict[str, Any]:
        """
        List all external texture files referenced by scene materials.

        Args:
            root_path: USD path to search under.

        Returns:
            Dict with unique texture file paths and their referencing materials.
        """
        _root_path = _pyval(root_path)
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import Usd, UsdShade, Sdf

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                root = stage.GetPrimAtPath({_root_path})
                if not root.IsValid():
                    print(json.dumps({{"error": "Root path not found: " + {_root_path}}}))
                else:
                    textures = {{}}
                    for p in Usd.PrimRange(root):
                        if not p.IsA(UsdShade.Shader):
                            continue
                        shader = UsdShade.Shader(p)
                        mat_path = str(p.GetParent().GetPath()) if p.GetParent() else ""
                        for inp in shader.GetInputs():
                            val = inp.Get()
                            if isinstance(val, Sdf.AssetPath):
                                asset = val.resolvedPath or val.path
                                if asset and any(asset.lower().endswith(ext) for ext in (
                                    ".png", ".jpg", ".jpeg", ".tga", ".bmp", ".exr",
                                    ".hdr", ".dds", ".tif", ".tiff"
                                )):
                                    if asset not in textures:
                                        textures[asset] = []
                                    textures[asset].append({{
                                        "material": mat_path,
                                        "input": inp.GetBaseName(),
                                    }})
                    print(json.dumps({{
                        "root_path": {_root_path},
                        "unique_textures": len(textures),
                        "textures": [
                            {{"path": k, "referenced_by": v}}
                            for k, v in sorted(textures.items())
                        ],
                    }}))
        """)
        return await self._execute_json_script(script)

    async def get_isaac_prim_variants(self, prim_path: str) -> Dict[str, Any]:
        """
        Get variant sets, available variants, and current selections.

        Args:
            prim_path: USD path of the prim.

        Returns:
            Dict with variant sets and their selections.
        """
        _prim_path = _pyval(prim_path)
        script = textwrap.dedent(f"""\
            import json
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                prim = stage.GetPrimAtPath({_prim_path})
                if not prim.IsValid():
                    print(json.dumps({{"error": "Prim not found: " + {_prim_path}}}))
                else:
                    vsets = prim.GetVariantSets()
                    names = vsets.GetNames()
                    variant_sets = {{}}
                    for name in names:
                        vs = vsets.GetVariantSet(name)
                        variant_sets[name] = {{
                            "variants": vs.GetVariantNames(),
                            "selection": vs.GetVariantSelection(),
                        }}
                    print(json.dumps({{
                        "prim_path": {_prim_path},
                        "variant_set_count": len(names),
                        "variant_sets": variant_sets,
                    }}))
        """)
        return await self._execute_json_script(script)

    async def focus_isaac_viewport(self, prim_path: str) -> Dict[str, Any]:
        """
        Frame the viewport camera to focus on a specific prim.

        Args:
            prim_path: USD path of the prim to focus on.

        Returns:
            Dict confirming the viewport was focused.
        """
        _prim_path = _pyval(prim_path)
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            import omni.kit.commands

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                prim = stage.GetPrimAtPath({_prim_path})
                if not prim.IsValid():
                    print(json.dumps({{"error": "Prim not found: " + {_prim_path}}}))
                else:
                    ctx = omni.usd.get_context()
                    selection = ctx.get_selection()
                    selection.set_selected_prim_paths([{_prim_path}], True)
                    # Frame the selection in the viewport
                    try:
                        import omni.kit.viewport.utility as vp_utils
                        vp_win = vp_utils.get_active_viewport_window()
                        if vp_win:
                            vp_win.viewport_api.frame_viewport_selection()
                    except Exception:
                        omni.kit.commands.execute("FrameSelectedCommand")
                    # Wait for viewport update
                    await omni.kit.app.get_app().next_update_async()
                    print(json.dumps({{
                        "prim_path": {_prim_path},
                        "focused": True,
                    }}))
        """)
        return await self._execute_json_script(script)

    async def get_isaac_selection(self) -> Dict[str, Any]:
        """
        Get the currently selected prims in the Isaac Sim viewport.

        Returns:
            Dict with list of selected prim paths and their types.
        """
        script = textwrap.dedent("""\
            import json
            import omni.usd

            ctx = omni.usd.get_context()
            stage = ctx.get_stage()
            if stage is None:
                print(json.dumps({"error": "No stage is currently open"}))
            else:
                selection = ctx.get_selection()
                paths = selection.get_selected_prim_paths()
                selected = []
                for path in paths:
                    prim = stage.GetPrimAtPath(path)
                    if prim.IsValid():
                        selected.append({
                            "path": path,
                            "name": prim.GetName(),
                            "type": prim.GetTypeName(),
                        })
                print(json.dumps({
                    "count": len(selected),
                    "selected_prims": selected,
                }))
        """)
        return await self._execute_json_script(script)

    async def get_isaac_animation_info(self, prim_path: str) -> Dict[str, Any]:
        """
        Get animation information for a prim: animated attributes and time samples.

        Args:
            prim_path: USD path of the prim.

        Returns:
            Dict with animated attributes, sample counts, and time ranges.
        """
        _prim_path = _pyval(prim_path)
        script = textwrap.dedent(f"""\
            import json
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                prim = stage.GetPrimAtPath({_prim_path})
                if not prim.IsValid():
                    print(json.dumps({{"error": "Prim not found: " + {_prim_path}}}))
                else:
                    animated_attrs = []
                    for attr in prim.GetAttributes():
                        num_samples = attr.GetNumTimeSamples()
                        if num_samples > 0:
                            times = attr.GetTimeSamples()
                            animated_attrs.append({{
                                "name": attr.GetName(),
                                "num_samples": num_samples,
                                "time_range": [float(times[0]), float(times[-1])] if times else None,
                            }})
                    print(json.dumps({{
                        "prim_path": {_prim_path},
                        "is_animated": len(animated_attrs) > 0,
                        "animated_attribute_count": len(animated_attrs),
                        "animated_attributes": animated_attrs,
                    }}))
        """)
        return await self._execute_json_script(script)

    async def raycast_isaac_scene(
        self,
        origin: FloatList,
        direction: FloatList,
        max_distance: float = 10000.0,
    ) -> Dict[str, Any]:
        """
        Cast a ray into the scene and return the closest hit.

        Args:
            origin: Ray origin as [x, y, z] in world-space stage units (metres
                by default; Isaac Sim stages are Z-up).
            direction: Ray direction as [x, y, z] (will be normalized).
            max_distance: Maximum ray distance in stage units.

        Returns:
            Dict with hit prim, position, normal, and distance.
        """
        script = textwrap.dedent(f"""\
            import json
            try:
                from omni.physx import get_physx_scene_query_interface
                import carb

                origin = carb.Float3({origin[0]}, {origin[1]}, {origin[2]})
                direction = carb.Float3({direction[0]}, {direction[1]}, {direction[2]})

                result = get_physx_scene_query_interface().raycast_closest(
                    origin, direction, {max_distance}
                )
                if result and result.get("hit", False):
                    pos = result.get("position", (0, 0, 0))
                    nrm = result.get("normal", (0, 0, 0))
                    print(json.dumps({{
                        "hit": True,
                        "prim_path": result.get("rigidBody", ""),
                        "position": [pos[0], pos[1], pos[2]],
                        "normal": [nrm[0], nrm[1], nrm[2]],
                        "distance": result.get("distance", 0),
                    }}))
                else:
                    print(json.dumps({{
                        "hit": False,
                        "message": "No hit detected within max distance",
                    }}))
            except ImportError:
                print(json.dumps({{
                    "error": "PhysX scene query not available. Is physics enabled?",
                }}))
            except Exception as e:
                print(json.dumps({{"error": f"Raycast failed: {{e}}"}}))
        """)
        return await self._execute_json_script(script)

    async def find_isaac_prims_in_area(
        self,
        center: FloatList,
        radius: float,
        prim_type: Optional[str] = None,
        root_path: str = "/",
        max_results: int = 100,
    ) -> Dict[str, Any]:
        """
        Find prims whose bounding box center is within a given radius.

        Args:
            center: Center point [x, y, z] to search around.
            radius: Search radius in stage units.
            prim_type: Optional prim type filter (e.g. "Mesh", "Xform").
            root_path: USD root path to search under.
            max_results: Maximum number of prims to return.

        Returns:
            Dict with matching prims sorted by distance from center.
        """
        _root_path = _pyval(root_path)
        _prim_type = _pyval(prim_type or None)
        script = textwrap.dedent(f"""\
            import json
            import math
            import omni.usd
            from pxr import Usd, UsdGeom, Gf

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                root = stage.GetPrimAtPath({_root_path})
                if not root.IsValid():
                    print(json.dumps({{"error": "Root path not found: " + {_root_path}}}))
                else:
                    center = Gf.Vec3d({center[0]}, {center[1]}, {center[2]})
                    radius = {radius}
                    prim_type_filter = {_prim_type}
                    bbox_cache = UsdGeom.BBoxCache(
                        Usd.TimeCode.Default(), ["default", "render"]
                    )
                    matches = []
                    for p in Usd.PrimRange(root):
                        if prim_type_filter and p.GetTypeName() != prim_type_filter:
                            continue
                        try:
                            bbox = bbox_cache.ComputeWorldBound(p)
                            rng = bbox.ComputeAlignedRange()
                            if rng.IsEmpty():
                                continue
                            mn = rng.GetMin()
                            mx = rng.GetMax()
                            prim_center = (mn + mx) * 0.5
                            dist = (prim_center - center).GetLength()
                            if dist <= radius:
                                matches.append({{
                                    "path": str(p.GetPath()),
                                    "name": p.GetName(),
                                    "type": p.GetTypeName(),
                                    "distance": round(dist, 4),
                                    "center": [prim_center[0], prim_center[1], prim_center[2]],
                                }})
                        except Exception:
                            continue
                    matches.sort(key=lambda x: x["distance"])
                    truncated = len(matches) > {max_results}
                    matches = matches[:{max_results}]
                    print(json.dumps({{
                        "center": [{center[0]}, {center[1]}, {center[2]}],
                        "radius": radius,
                        "count": len(matches),
                        "truncated": truncated,
                        "matches": matches,
                    }}))
        """)
        return await self._execute_json_script(script)

