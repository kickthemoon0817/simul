"""Scene Inspection (Read-only) tools for Isaac Sim."""

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


# Prefix for every generated script that counts prims. get_isaac_scene_summary
# and get_isaac_scene_stats embed the same generator so their totals and type
# labels agree: every prim Usd.PrimRange visits under the root with the default
# predicate (active, defined, loaded, non-abstract), minus the pseudo-root,
# which is a container and not a scene prim.
COUNTED_PRIMS_HELPER = textwrap.dedent("""\
    from pxr import Usd as _CountUsd

    def _counted_prims(stage, root_path):
        root = stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            return
        for prim in _CountUsd.PrimRange(root):
            if prim.IsPseudoRoot():
                continue
            yield prim

    def _prim_type_label(prim):
        return prim.GetTypeName() or "Typeless"
""")


class SceneInspectionMixin:
    # ------------------------------------------------------------------
    # Phase 1: Scene Inspection (Read-only)
    # ------------------------------------------------------------------

    async def get_isaac_stage_info(self) -> Dict[str, Any]:
        """
        Get current stage metadata from the running Isaac Sim instance.

        Returns:
            Stage info dict with up_axis, meters_per_unit, total_prims,
            root_prims, layer_count, default_prim, etc.
        """
        bridge_result = await self._execute_bridge_action("get_stage_info")
        if bridge_result is not None:
            return bridge_result

        script = textwrap.dedent("""\
            import json
            import omni.usd
            from pxr import Usd, UsdGeom

            ctx = omni.usd.get_context()
            stage = ctx.get_stage()
            if stage is None:
                print(json.dumps({"error": "No stage is currently open"}))
            else:
                root = stage.GetPseudoRoot()
                root_prims = [str(p.GetPath()) for p in root.GetChildren()]
                up_axis = UsdGeom.GetStageUpAxis(stage)
                meters = UsdGeom.GetStageMetersPerUnit(stage)
                tps = stage.GetTimeCodesPerSecond()
                start = stage.GetStartTimeCode()
                end = stage.GetEndTimeCode()
                fps = stage.GetFramesPerSecond()
                total = 0
                for _ in stage.Traverse():
                    total += 1
                layer_stack = stage.GetLayerStack()
                default_prim = stage.GetDefaultPrim()
                dp_path = str(default_prim.GetPath()) if default_prim else None
                print(json.dumps({
                    "stage_url": str(ctx.get_stage_url()),
                    "up_axis": up_axis,
                    "meters_per_unit": meters,
                    "time_codes_per_second": tps,
                    "start_time": start,
                    "end_time": end,
                    "frame_rate": fps,
                    "total_prims": total,
                    "root_prims": root_prims,
                    "layer_count": len(layer_stack),
                    "default_prim": dp_path,
                }))
        """)
        mode = self._raw_script_transport_mode
        return await self._execute_json_script(script, transport_mode=mode)

    async def list_isaac_prims(
        self,
        root_path: str = "/",
        prim_type: Optional[str] = None,
        max_depth: int = 5,
        max_results: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        List prims in the current Isaac Sim stage with optional filtering.

        Args:
            root_path: USD prim path to start the traversal from.
            prim_type: Filter by USD prim type name (e.g. "Mesh", "Xform").
            max_depth: Maximum traversal depth below root_path; -1 for
                unlimited.
            max_results: Maximum number of prims to return per page. Clamped
                to [1, 1000]; the effective cap is reported as applied_limit.
            offset: Number of matching prims to skip before the page starts;
                pass the previous page's next_offset to continue.

        Returns:
            Dict with a page of prim entries (path, type, name, active).
        """
        max_results = max(1, min(max_results, 1000))
        offset = max(0, offset)
        bridge_result = await self._execute_bridge_action(
            "list_prims",
            {
                "root_path": root_path,
                "prim_type": prim_type,
                "max_depth": max_depth,
                "max_results": max_results,
                # A bridge published before paging reads this key and ignores
                # the two above; sending it keeps the cap honoured there.
                "max_items": max_results,
                "offset": offset,
            },
        )
        # That older bridge also drops the offset, so its answer only stands
        # for the first page; later pages fall through to the script path.
        if bridge_result is not None and (offset == 0 or "offset" in bridge_result):
            return bridge_result

        _root_path = _pyval(root_path)
        _prim_type = _pyval(prim_type or "")
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
                    print(json.dumps({{"error": "Invalid root path: " + {_root_path}}}))
                else:
                    prims = []
                    type_filter = {_prim_type}
                    max_d = {max_depth}
                    max_n = {max_results}
                    offset = {offset}
                    matched = 0
                    truncated = False
                    root_depth = len(str(root.GetPath()).rstrip("/").split("/"))

                    # continue skips emitting a prim but still descends its
                    # subtree, so the depth limit bounded the output and not the
                    # work. Prune instead.
                    prim_iter = iter(Usd.PrimRange(root))
                    for p in prim_iter:
                        path_str = str(p.GetPath())
                        depth = len(path_str.rstrip("/").split("/")) - root_depth
                        if max_d >= 0 and depth >= max_d:
                            prim_iter.PruneChildren()
                        if max_d >= 0 and depth > max_d:
                            continue
                        ptype = p.GetTypeName()
                        if type_filter and ptype != type_filter:
                            continue
                        matched += 1
                        if matched <= offset:
                            continue
                        if len(prims) >= max_n:
                            truncated = True
                            break
                        prims.append({{
                            "path": path_str,
                            "type": ptype,
                            "name": p.GetName(),
                            "active": p.IsActive(),
                        }})
                    print(json.dumps({{
                        "root_path": {_root_path},
                        "type_filter": type_filter or None,
                        "max_depth": max_d,
                        "count": len(prims),
                        "offset": offset,
                        "applied_limit": max_n,
                        "truncated": truncated,
                        "next_offset": offset + len(prims) if truncated else None,
                        "prims": prims,
                    }}))
        """)
        mode = self._raw_script_transport_mode
        return await self._execute_json_script(script, transport_mode=mode)

    async def get_isaac_prim_info(self, prim_path: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific prim in the running stage.

        Args:
            prim_path: USD path of the prim (e.g. "/World/Cube").

        Returns:
            Dict with prim type, attributes, transform, children, etc.
        """
        bridge_result = await self._execute_bridge_action(
            "get_prim_info",
            {"prim_path": prim_path},
        )
        if bridge_result is not None:
            return bridge_result

        _prim_path = _pyval(prim_path)
        _bulk_attrs = repr(set(BULK_GEOMETRY_ATTRIBUTES))
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
                    # Basic info
                    children = [str(c.GetPath()) for c in prim.GetChildren()]
                    child_types = {{}}
                    for c in prim.GetChildren():
                        t = c.GetTypeName() or "Typeless"
                        child_types[t] = child_types.get(t, 0) + 1

                    # Attributes — custom serializer for USD types
                    def _serialize(v):
                        if v is None:
                            return None
                        if isinstance(v, (bool, int, float, str)):
                            return v
                        if isinstance(v, (Gf.Vec2f, Gf.Vec2d, Gf.Vec2h, Gf.Vec2i,
                                          Gf.Vec3f, Gf.Vec3d, Gf.Vec3h, Gf.Vec3i,
                                          Gf.Vec4f, Gf.Vec4d, Gf.Vec4h, Gf.Vec4i)):
                            return [float(x) for x in v]
                        if isinstance(v, (Gf.Quatf, Gf.Quatd, Gf.Quath)):
                            return [float(v.GetReal())] + [float(x) for x in v.GetImaginary()]
                        if isinstance(v, (Gf.Matrix4d, Gf.Matrix4f, Gf.Matrix3d, Gf.Matrix3f)):
                            return str(type(v).__name__)
                        try:
                            if hasattr(v, '__len__') and not isinstance(v, str):
                                if len(v) > 16:
                                    return f"[{{len(v)}} elements]"
                                return [_serialize(x) for x in v]
                        except Exception:
                            pass
                        try:
                            return float(v) if isinstance(v, (type(Gf.Vec3f()[0]),)) else v
                        except Exception:
                            pass
                        return str(v)

                    BULK_GEOMETRY_ATTRS = {_bulk_attrs}
                    attrs = {{}}
                    for attr in prim.GetAttributes():
                        try:
                            # Bulk geometry only: Get() would decompress the
                            # whole array out of the crate layer just for
                            # _serialize to replace it with a count. Small
                            # arrays still come back in full.
                            attr_name = attr.GetName()
                            if attr_name in BULK_GEOMETRY_ATTRS:
                                type_name = attr.GetTypeName()
                                if getattr(type_name, "isArray", False):
                                    attrs[attr_name] = "<array %s>" % (type_name,)
                                    continue
                            val = attr.Get()
                            if val is not None:
                                attrs[attr.GetName()] = _serialize(val)
                        except Exception:
                            attrs[attr.GetName()] = "<unreadable>"

                    # Transform
                    xform_data = None
                    if prim.IsA(UsdGeom.Xformable):
                        xformable = UsdGeom.Xformable(prim)
                        local_xform = xformable.ComputeLocalToWorldTransform(
                            Usd.TimeCode.Default()
                        )
                        t = local_xform.ExtractTranslation()
                        r = local_xform.ExtractRotation()
                        quat = r.GetQuat()
                        s = Gf.Vec3d(1, 1, 1)
                        try:
                            xform_vecs = xformable.GetOrderedXformOps()
                            for op in xform_vecs:
                                if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                                    s = Gf.Vec3d(op.Get())
                        except Exception:
                            pass
                        xform_data = {{
                            "translation": list(t),
                            "rotation_quat": [quat.GetReal()] + list(quat.GetImaginary()),
                            "scale": list(s),
                        }}

                    # Material bindings
                    mat_paths = []
                    try:
                        from pxr import UsdShade
                        bindings = UsdShade.MaterialBindingAPI(prim)
                        mat, _ = bindings.ComputeBoundMaterial()
                        if mat:
                            mat_paths.append(str(mat.GetPath()))
                    except Exception:
                        pass

                    print(json.dumps({{
                        "path": {_prim_path},
                        "name": prim.GetName(),
                        "type": prim.GetTypeName(),
                        "is_active": prim.IsActive(),
                        "is_defined": prim.IsDefined(),
                        "is_instance": prim.IsInstance(),
                        "purpose": UsdGeom.Imageable(prim).ComputePurpose() if prim.IsA(UsdGeom.Imageable) else None,
                        "visibility": UsdGeom.Imageable(prim).ComputeVisibility() if prim.IsA(UsdGeom.Imageable) else None,
                        "children_count": len(children),
                        "children_types": child_types,
                        "children": children[:50],
                        "material_bindings": mat_paths,
                        "transform": xform_data,
                        "attributes": attrs,
                    }}))
        """)
        mode = self._raw_script_transport_mode
        return await self._execute_json_script(script, transport_mode=mode)

    async def get_isaac_prim_transform(
        self,
        prim_path: str,
        world_space: bool = True,
    ) -> Dict[str, Any]:
        """
        Get the transform of a prim in world or local space.

        Args:
            prim_path: USD path of the prim.
            world_space: If True, return world-space transform; otherwise local.

        Returns:
            Dict with translation, rotation (quaternion), and scale.
        """
        bridge_result = await self._execute_bridge_action(
            "get_prim_transform",
            {"prim_path": prim_path, "world_space": world_space},
        )
        if bridge_result is not None:
            return bridge_result

        _prim_path = _pyval(prim_path)
        world_str = "True" if world_space else "False"
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
                elif not prim.IsA(UsdGeom.Xformable):
                    print(json.dumps({{"error": "Prim is not Xformable: " + {_prim_path}}}))
                else:
                    xformable = UsdGeom.Xformable(prim)
                    world = {world_str}
                    if world:
                        xform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    else:
                        xform = xformable.GetLocalTransformation(Usd.TimeCode.Default())

                    t = xform.ExtractTranslation()
                    r = xform.ExtractRotation().GetQuat()
                    # Extract scale from xform ops
                    s = [1.0, 1.0, 1.0]
                    for op in xformable.GetOrderedXformOps():
                        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                            sv = op.Get()
                            if sv is not None:
                                s = list(Gf.Vec3d(sv))
                            break

                    print(json.dumps({{
                        "prim_path": {_prim_path},
                        "space": "world" if world else "local",
                        "translation": list(t),
                        "rotation_quat": [r.GetReal()] + list(r.GetImaginary()),
                        "scale": s,
                    }}))
        """)
        mode = self._raw_script_transport_mode
        return await self._execute_json_script(script, transport_mode=mode)

    async def search_isaac_prims(
        self,
        query: str,
        search_type: str = "type",
        root_path: str = "/",
        max_results: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        Search for prims by type name or name substring.

        Args:
            query: Exact prim type name (e.g. "Mesh") when search_type is
                "type", or a case-insensitive name substring when it is
                "name".
            search_type: "type" to match the prim type, "name" to match the
                prim name.
            root_path: USD prim path to search under.
            max_results: Maximum number of matches to return per page.
                Clamped to [1, 1000]; the effective cap is reported as
                applied_limit.
            offset: Number of matches to skip before the page starts; pass
                the previous page's next_offset to continue.

        Returns:
            Dict with a page of matching prim paths and types.
        """
        if not query or not query.strip():
            return ErrorResponse(
                error="query is required: a prim type name for search_type='type' "
                "or a name substring for search_type='name'",
                error_type="ValueError",
            ).model_dump()
        max_results = max(1, min(max_results, 1000))
        offset = max(0, offset)
        bridge_result = await self._execute_bridge_action(
            "search_prims",
            {
                "search_type": search_type,
                "query": query,
                "root_path": root_path,
                "max_results": max_results,
                "offset": offset,
            },
        )
        # A bridge published before paging drops the offset, so its answer
        # only stands for the first page; later pages use the script path.
        if bridge_result is not None and (offset == 0 or "offset" in bridge_result):
            return bridge_result

        _root_path = _pyval(root_path)
        _search_type = _pyval(search_type)
        _query = _pyval(query)
        script = textwrap.dedent(f"""\
            import json, re
            import omni.usd
            from pxr import Usd

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                root = stage.GetPrimAtPath({_root_path})
                if not root.IsValid():
                    print(json.dumps({{"error": "Invalid root path: " + {_root_path}}}))
                else:
                    matches = []
                    search_type = {_search_type}
                    query = {_query}
                    max_r = {max_results}
                    offset = {offset}
                    matched = 0
                    truncated = False
                    for p in Usd.PrimRange(root):
                        if search_type == "type":
                            hit = p.GetTypeName() == query
                        elif search_type == "name":
                            hit = query.lower() in p.GetName().lower()
                        else:
                            hit = False
                        if not hit:
                            continue
                        matched += 1
                        if matched <= offset:
                            continue
                        if len(matches) >= max_r:
                            truncated = True
                            break
                        matches.append({{"path": str(p.GetPath()), "type": p.GetTypeName(), "name": p.GetName()}})
                    print(json.dumps({{
                        "search_type": search_type,
                        "query": query,
                        "root_path": {_root_path},
                        "count": len(matches),
                        "offset": offset,
                        "applied_limit": max_r,
                        "truncated": truncated,
                        "next_offset": offset + len(matches) if truncated else None,
                        "matches": matches,
                    }}))
        """)
        mode = self._raw_script_transport_mode
        return await self._execute_json_script(script, transport_mode=mode)

    async def get_isaac_scene_summary(self) -> Dict[str, Any]:
        """
        Get a high-level summary of the current Isaac Sim scene.

        Returns:
            Dict with prim counts by type, total prims, hierarchy depth, etc.
            Prims are counted the same way as get_isaac_scene_stats.
        """
        script = COUNTED_PRIMS_HELPER + textwrap.dedent("""\
            import json
            import omni.usd
            from pxr import Usd, UsdGeom, UsdPhysics

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({"error": "No stage is currently open"}))
            else:
                type_counts = {}
                total = 0
                max_depth = 0
                has_physics = False
                has_animation = stage.GetEndTimeCode() > stage.GetStartTimeCode()

                for p in _counted_prims(stage, "/"):
                    total += 1
                    t = _prim_type_label(p)
                    type_counts[t] = type_counts.get(t, 0) + 1
                    depth = len(str(p.GetPath()).split("/")) - 1
                    if depth > max_depth:
                        max_depth = depth
                    if not has_physics:
                        if p.HasAPI(UsdPhysics.RigidBodyAPI) or p.IsA(UsdPhysics.Scene):
                            has_physics = True

                root_prims = [str(p.GetPath()) for p in stage.GetPseudoRoot().GetChildren()]
                print(json.dumps({
                    "stage_url": str(omni.usd.get_context().get_stage_url()),
                    "total_prims": total,
                    "max_depth": max_depth,
                    "has_physics": has_physics,
                    "has_animation": has_animation,
                    "type_counts": type_counts,
                    "root_prims": root_prims,
                    "up_axis": UsdGeom.GetStageUpAxis(stage),
                    "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
                }))
        """)
        return await self._execute_json_script(script)

