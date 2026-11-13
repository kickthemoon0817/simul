"""Scene Inspection (Read-only) tools for Isaac Sim."""

import textwrap
from typing import Any, Dict, Optional

from ...schemas.common import ErrorResponse
from ._shared import (
    BULK_GEOMETRY_ATTRIBUTES,
    TRAVERSAL_PREDICATE_CORE,
    XFORM_READ_CORE,
    _compose_script,
    _pyval,
)
from .._meta import DeprecatedAlias, tool_meta


def _bridge_listing_stands(result: Optional[Dict[str, Any]], offset: int) -> bool:
    """Whether a bridge listing answers the request, or the script path must.

    A bridge published before paging drops the offset, so its answer only
    stands for the first page. One published before the traversal flags
    ignores them and walks with the default predicate (no instance proxies,
    no inactive prims), which it shows by not echoing them back. A failure
    stands either way: the script path would fail the same way.

    Args:
        result: What ``_execute_bridge_action`` returned.
        offset: The requested page offset.

    Returns:
        True when ``result`` is the answer to return.
    """
    if result is None:
        return False
    if result.get("success") is False:
        return True
    if offset and "offset" not in result:
        return False
    return "include_instance_proxies" in result


# Prefix for every generated script that counts prims. get_isaac_scene_summary
# and get_isaac_scene_stats embed the same generator so their totals and type
# labels agree with each other and with list_isaac_prims: every prim the
# listings' traversal visits under the root (instance proxies included by
# default, inactive prims on request), minus the pseudo-root, which is a
# container and not a scene prim. The default predicate stopped at
# instanceable prims, so a scene built from instanced assets counted no meshes.
COUNTED_PRIMS_HELPER = (
    "from pxr import Usd\n\n"
    + textwrap.dedent(TRAVERSAL_PREDICATE_CORE)
    + textwrap.dedent("""\

    def _counted_prims(stage, root_path, include_instance_proxies=True, include_inactive=False):
        root = stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            return
        predicate = _traversal_predicate(include_instance_proxies, include_inactive)
        for prim in Usd.PrimRange(root, predicate):
            if prim.IsPseudoRoot():
                continue
            yield prim

    def _prim_type_label(prim):
        return prim.GetTypeName() or "Typeless"
""")
)


class SceneInspectionMixin:
    # ------------------------------------------------------------------
    # Phase 1: Scene Inspection (Read-only)
    # ------------------------------------------------------------------

    @tool_meta(
        name="get_isaac_stage_info",
        description=(
            "Get current stage metadata including root layer, up-axis, meters-per-unit, and "
            "default prim. Pass include_prim_count=true to also count every prim "
            "(total_prims); the count walks the whole stage, so it is off by default."
        ),
        read_only=True,
        idempotent=True,
    )
    async def get_isaac_stage_info(
        self, include_prim_count: bool = False
    ) -> Dict[str, Any]:
        """
        Get current stage metadata from the running Isaac Sim instance.

        Args:
            include_prim_count: Also count every prim on the stage for
                ``total_prims``. The count walks the whole stage on Kit's
                main thread, so it is off unless asked for; the key is then
                ``None``.

        Returns:
            Stage info dict with up_axis, meters_per_unit, total_prims,
            root_prims, layer_count, default_prim, etc.
        """
        bridge_result = await self._execute_bridge_action(
            "get_stage_info", {"include_prim_count": include_prim_count}
        )
        if bridge_result is not None:
            return bridge_result

        _include_prim_count = _pyval(bool(include_prim_count))
        script = textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import Usd, UsdGeom

            ctx = omni.usd.get_context()
            stage = ctx.get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                root = stage.GetPseudoRoot()
                root_prims = [str(p.GetPath()) for p in root.GetChildren()]
                up_axis = UsdGeom.GetStageUpAxis(stage)
                meters = UsdGeom.GetStageMetersPerUnit(stage)
                tps = stage.GetTimeCodesPerSecond()
                start = stage.GetStartTimeCode()
                end = stage.GetEndTimeCode()
                fps = stage.GetFramesPerSecond()
                # Counted as the listings and get_isaac_scene_summary count:
                # instance proxies included, the pseudo-root left out.
                total = None
                if {_include_prim_count}:
                    total = sum(1 for _ in stage.Traverse(Usd.TraverseInstanceProxies()))
                layer_stack = stage.GetLayerStack()
                default_prim = stage.GetDefaultPrim()
                dp_path = str(default_prim.GetPath()) if default_prim else None
                print(json.dumps({{
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
                }}))
        """)
        mode = self._raw_script_transport_mode
        return await self._execute_json_script(script, transport_mode=mode)

    @tool_meta(
        name="list_isaac_prims",
        description=(
            "Browse prims under root_path with a depth limit and an optional prim type "
            "filter. Descends into instanceable prims (entries there carry "
            "instance_proxy=true; robot assets keep their meshes and colliders there) "
            "unless include_instance_proxies=false; include_inactive=true also lists "
            "deactivated prims with active=false. Paged (max_results/offset)."
        ),
        read_only=True,
        idempotent=True,
        deprecated_aliases=(DeprecatedAlias("max_items", "max_results"),),
    )
    async def list_isaac_prims(
        self,
        root_path: str = "/",
        prim_type: Optional[str] = None,
        max_depth: int = 5,
        max_results: int = 100,
        offset: int = 0,
        include_instance_proxies: bool = True,
        include_inactive: bool = False,
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
            include_instance_proxies: Descend into instanceable prims. Their
                contents are instance proxies: listed with
                ``instance_proxy: true``, readable but not editable in place.
            include_inactive: Also list deactivated prims (``active: false``).
                Their children are not composed, so none are listed.

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
                "include_instance_proxies": include_instance_proxies,
                "include_inactive": include_inactive,
            },
        )
        if _bridge_listing_stands(bridge_result, offset):
            return bridge_result  # type: ignore[return-value]

        _root_path = _pyval(root_path)
        _prim_type = _pyval(prim_type or "")
        _proxies = _pyval(bool(include_instance_proxies))
        _inactive = _pyval(bool(include_inactive))
        _predicate = f"_traversal_predicate({_proxies}, {_inactive})"
        script = _compose_script(
            """\
            import json
            import omni.usd
            from pxr import Usd
            """,
            TRAVERSAL_PREDICATE_CORE,
            f"""\
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
                    prim_iter = iter(Usd.PrimRange(root, {_predicate}))
                    for p in prim_iter:
                        path_str = str(p.GetPath())
                        depth = len(path_str.rstrip("/").split("/")) - root_depth
                        if max_d >= 0 and depth >= max_d:
                            prim_iter.PruneChildren()
                        if max_d >= 0 and depth > max_d:
                            continue
                        # The pseudo-root is a container, not a scene prim;
                        # the counting tools leave it out too.
                        if p.IsPseudoRoot():
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
                        entry = {{
                            "path": path_str,
                            "type": ptype,
                            "name": p.GetName(),
                            "active": p.IsActive(),
                        }}
                        if p.IsInstanceProxy():
                            entry["instance_proxy"] = True
                        prims.append(entry)
                    print(json.dumps({{
                        "root_path": {_root_path},
                        "type_filter": type_filter or None,
                        "max_depth": max_d,
                        "include_instance_proxies": {_proxies},
                        "include_inactive": {_inactive},
                        "count": len(prims),
                        "offset": offset,
                        "applied_limit": max_n,
                        "truncated": truncated,
                        "next_offset": offset + len(prims) if truncated else None,
                        "prims": prims,
                    }}))
            """,
        )
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
        _bulk_attrs = _pyval(set(BULK_GEOMETRY_ATTRIBUTES))
        script = _compose_script(
            """\
            import json
            import omni.usd
            from pxr import Gf, Sdf, Usd, UsdGeom
            """,
            XFORM_READ_CORE,
            f"""\

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
                        # Anything else (Sdf.AssetPath, Sdf.Path, TfToken
                        # wrappers, enums) must still become JSON, or one such
                        # attribute fails the whole json.dumps below.
                        if isinstance(v, Sdf.AssetPath):
                            return v.path
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

                    # Transform: world pose, with the local one beside it.
                    xform_data = None
                    if prim.IsA(UsdGeom.Xformable):
                        xform_data = _transform_report(
                            UsdGeom.Xformable(prim), True, Usd.TimeCode.Default()
                        )

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
                        "purpose": (
                            UsdGeom.Imageable(prim).ComputePurpose() if prim.IsA(UsdGeom.Imageable) else None
                        ),
                        "visibility": (
                            UsdGeom.Imageable(prim).ComputeVisibility() if prim.IsA(UsdGeom.Imageable) else None
                        ),
                        "children_count": len(children),
                        "children_types": child_types,
                        "children": children[:50],
                        "material_bindings": mat_paths,
                        "transform": xform_data,
                        "attributes": attrs,
                    }}))
            """,
        )
        mode = self._raw_script_transport_mode
        # The transform comes from USD, which GPU physics stops updating.
        return await self._execute_json_script(
            script, transport_mode=mode, annotations=("usd_readback",)
        )

    async def get_isaac_prim_transform(
        self,
        prim_path: str,
        world_space: bool = True,
    ) -> Dict[str, Any]:
        """
        Get the transform of a prim in world or local space.

        Rotation and scale are factored out of the composed matrix
        (Gf.Transform), so they stay right under scaled parents and for
        matrix ops.

        Args:
            prim_path: USD path of the prim.
            world_space: If True, the top-level values are world space;
                otherwise local (relative to the parent).

        Returns:
            Dict with ``space`` and that space's ``translation``,
            ``rotation_quat`` ([w, x, y, z]), ``rotation_euler_xyz``
            (degrees, the order set_isaac_prim_transform takes) and
            ``scale``; ``local`` and ``world`` hold both spaces, plus
            ``xform_op_order``. ``sheared`` and ``note`` flag a transform no
            rotation and per-axis scale reproduce exactly.
        """
        bridge_result = await self._execute_bridge_action(
            "get_prim_transform",
            {"prim_path": prim_path, "world_space": world_space},
        )
        if bridge_result is not None:
            return bridge_result

        _prim_path = _pyval(prim_path)
        world_str = "True" if world_space else "False"
        script = _compose_script(
            """\
            import json
            import omni.usd
            from pxr import Usd, UsdGeom, Gf
            """,
            XFORM_READ_CORE,
            f"""\
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
                    report = {{"prim_path": {_prim_path}}}
                    report.update(_transform_report(
                        UsdGeom.Xformable(prim), {world_str}, Usd.TimeCode.Default()
                    ))
                    print(json.dumps(report))
            """,
        )
        mode = self._raw_script_transport_mode
        return await self._execute_json_script(
            script, transport_mode=mode, annotations=("usd_readback",)
        )

    @tool_meta(
        name="search_isaac_prims",
        description=(
            "Search the whole hierarchy below root_path for prims whose type name equals "
            "query (search_type='type') or whose name contains query (search_type='name'). "
            "Searches inside instanceable prims too (matches there carry "
            "instance_proxy=true) unless include_instance_proxies=false; "
            "include_inactive=true also matches deactivated prims (active=false). Paged "
            "(max_results/offset)."
        ),
        read_only=True,
        idempotent=True,
    )
    async def search_isaac_prims(
        self,
        query: str,
        search_type: str = "type",
        root_path: str = "/",
        max_results: int = 100,
        offset: int = 0,
        include_instance_proxies: bool = True,
        include_inactive: bool = False,
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
            include_instance_proxies: Search inside instanceable prims, where
                Isaac robot assets keep their meshes and colliders. Matches
                there carry ``instance_proxy: true``.
            include_inactive: Also match deactivated prims, which carry
                ``active: false``.

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
                "include_instance_proxies": include_instance_proxies,
                "include_inactive": include_inactive,
            },
        )
        if _bridge_listing_stands(bridge_result, offset):
            return bridge_result  # type: ignore[return-value]

        _root_path = _pyval(root_path)
        _search_type = _pyval(search_type)
        _query = _pyval(query)
        _proxies = _pyval(bool(include_instance_proxies))
        _inactive = _pyval(bool(include_inactive))
        script = _compose_script(
            """\
            import json, re
            import omni.usd
            from pxr import Usd
            """,
            TRAVERSAL_PREDICATE_CORE,
            f"""\
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
                    for p in Usd.PrimRange(root, _traversal_predicate({_proxies}, {_inactive})):
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
                        entry = {{"path": str(p.GetPath()), "type": p.GetTypeName(), "name": p.GetName()}}
                        if p.IsInstanceProxy():
                            entry["instance_proxy"] = True
                        if not p.IsActive():
                            entry["active"] = False
                        matches.append(entry)
                    print(json.dumps({{
                        "search_type": search_type,
                        "query": query,
                        "root_path": {_root_path},
                        "include_instance_proxies": {_proxies},
                        "include_inactive": {_inactive},
                        "count": len(matches),
                        "offset": offset,
                        "applied_limit": max_r,
                        "truncated": truncated,
                        "next_offset": offset + len(matches) if truncated else None,
                        "matches": matches,
                    }}))
            """,
        )
        mode = self._raw_script_transport_mode
        return await self._execute_json_script(script, transport_mode=mode)

    @tool_meta(
        name="get_isaac_scene_summary",
        description=(
            "Get a high-level summary of the whole stage: total prims and counts per prim "
            "type, hierarchy depth, has_physics/has_animation flags, root prims, up axis "
            "and meters per unit. Counts prims the same way as get_isaac_scene_stats. Use "
            "get_isaac_scene_summary to orient before listing; use get_isaac_scene_stats "
            "for geometry totals (vertices, faces, meshes, lights, cameras, materials) "
            "under a chosen root. Counts inside instanceable prims (where Isaac assets keep "
            "their meshes) unless include_instance_proxies=false; include_inactive=true "
            "also counts deactivated prims."
        ),
        read_only=True,
        idempotent=True,
    )
    async def get_isaac_scene_summary(
        self, include_instance_proxies: bool = True, include_inactive: bool = False
    ) -> Dict[str, Any]:
        """
        Get a high-level summary of the current Isaac Sim scene.

        Args:
            include_instance_proxies: Count the prims inside instanceable
                prims, as list_isaac_prims does.
            include_inactive: Also count deactivated prims.

        Returns:
            Dict with prim counts by type, total prims, hierarchy depth, etc.
            Prims are counted the same way as get_isaac_scene_stats.
        """
        _proxies = _pyval(bool(include_instance_proxies))
        _inactive = _pyval(bool(include_inactive))
        script = COUNTED_PRIMS_HELPER + textwrap.dedent(f"""\
            import json
            import omni.usd
            from pxr import Usd, UsdGeom, UsdPhysics

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                print(json.dumps({{"error": "No stage is currently open"}}))
            else:
                type_counts = {{}}
                total = 0
                max_depth = 0
                has_physics = False
                has_animation = stage.GetEndTimeCode() > stage.GetStartTimeCode()

                for p in _counted_prims(stage, "/", {_proxies}, {_inactive}):
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
                print(json.dumps({{
                    "stage_url": str(omni.usd.get_context().get_stage_url()),
                    "total_prims": total,
                    "max_depth": max_depth,
                    "has_physics": has_physics,
                    "has_animation": has_animation,
                    "type_counts": type_counts,
                    "root_prims": root_prims,
                    "up_axis": UsdGeom.GetStageUpAxis(stage),
                    "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
                    "include_instance_proxies": {_proxies},
                    "include_inactive": {_inactive},
                }}))
        """)
        return await self._execute_json_script(script)
